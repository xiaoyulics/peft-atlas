/* Fig. "orbit" — Hyperparameter Orbit Explorer: "LoRA+ is alpha in disguise".
   Thm III.10 (the hyperparameter torus) and Cor III.11 (LoRA+ = alpha * lambda under Adam) of theory/framework.md.

   A seeded toy regression is trained twice, in float64, by a hand-written Adam (bias-corrected, epsilon in
   {0, 1e-8, 1e-6}, convention 0/0 = 0) or by full-batch gradient descent. The reference run and the current run use
   hyperparameters (s = alpha/r, eta_A, eta_B, sigma_A); the 2-torus acts on these by
     Adam: c.(s, eta_A, eta_B, sigma_A) = (s/(c_A c_B), c_A eta_A, c_B eta_B, c_A sigma_A)
     SGD : c.(s, eta_A, eta_B, sigma_A) = (s/(c_A c_B), c_A^2 eta_A, c_B^2 eta_B, c_A sigma_A).
   All four sliders live on a factor-of-two grid, so every torus element reachable on the grid is a power of two and
   rescales IEEE-754 doubles exactly: on the orbit, with epsilon = 0, no weight decay and no clipping, the two weight
   trajectories are bit-identical and the meter max_t ||W_t - W'_t||_F reads exactly 0. */
(function () {
  'use strict';

  /* ---------- fixed setup (theory/visuals.md, section 2) ---------- */
  var M = 8, NI = 6, NS = 40, RK = 2, T = 300, MN = M * NI;
  var B1 = 0.9, B2 = 0.999;           // Adam betas
  var WD = 0.01;                      // decoupled (AdamW-style) weight decay coefficient, PyTorch AdamW default
  var CLIP = 1;                       // global-norm clipping threshold, HF Trainer default max_grad_norm
  var LR_BASE = 1e-3;                 // eta = 1e-3 * 2^k
  var TARGET_SV = [2.0, 0.7];         // singular values of the rank-2 target update W* - W0
  var RANGE = { ks: [-4, 7], kA: [-6, 11], kB: [-6, 11], kS: [-6, 3] };
  var BASE = { adam: { ks: 0, kA: 2, kS: -2 }, sgd: { ks: 0, kA: 5, kS: -2 } };
  var LAMS = [4, 16, 64];
  var EPS = [0, 1e-8, 1e-6];
  var SEED0 = 7;
  var KEYS = ['ks', 'kA', 'kB', 'kS'];

  /* ---------- formatting ---------- */
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sup(n) { return String(n).split('').map(function (ch) { return SUP[ch] || ch; }).join(''); }
  function pow10(e) { return e === 0 ? '1' : e === 1 ? '10' : '10' + sup(e); }
  /* the same power of ten as SVG text with a real superscript (IBM Plex Mono has no superscript minus or digits 0, 4-9) */
  function pow10T(sel, e, fs) {
    if (e === 0 || e === 1) return sel.text(e === 0 ? '1' : '10');
    sel.text('10'); sel.append('tspan').attr('dy', -0.42 * fs).style('font-size', (0.74 * fs).toFixed(1) + 'px').text(String(e).replace('-', '−'));
    return sel;
  }
  /* Unicode superscripts in HTML readouts as <sup> */
  var SUPR = {}; Object.keys(SUP).forEach(function (k) { SUPR[SUP[k]] = k === '-' ? '−' : k; });
  function supHtml(t) { return String(t).replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, function (m) { return '<sup>' + m.split('').map(function (c) { return SUPR[c]; }).join('') + '</sup>'; }); }
  /* scientific with Unicode exponent: 3.4 × 10⁻⁸ */
  function sci(x, d) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '∞';
    var p = x.toExponential(d == null ? 1 : d).split('e');
    var e = parseInt(p[1], 10);
    if (e === 0) return p[0];
    return p[0] + ' × 10' + sup(e);
  }
  /* gap value: decimal in [1e-3, 1e3), scientific otherwise */
  function gapStr(x) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '∞';
    if (x >= 1e-3 && x < 1e3) return String(+x.toPrecision(3));
    return sci(x, 1);
  }
  /* compact ML-style numbers: 6.4e-3, 0.128, 62.5 */
  function compact(x) {
    if (x === 0) return '0';
    var a = Math.abs(x);
    if (a >= 0.1 && a < 1e4) return String(+x.toPrecision(4));
    return x.toExponential(2).replace(/\.?0+e/, 'e').replace('e+', 'e');
  }
  function pow2Str(k) { return String(Math.pow(2, k)); } /* exact decimal for 2^k, |k| <= 8 */
  function fracStr(k) { return k >= 0 ? String(Math.pow(2, k)) : '1/' + Math.pow(2, -k); }
  /* 2^k times a symbol, as HTML: "16<i>s</i>", "<i>s</i>/4", "<i>s</i>" */
  function mulSym(k, sym) { return k === 0 ? sym : k > 0 ? Math.pow(2, k) + sym : sym + '/' + Math.pow(2, -k); }
  /* 2^k as a TeX fraction */
  function texFrac(k) { return k >= 0 ? String(Math.pow(2, k)) : '\\tfrac1{' + Math.pow(2, -k) + '}'; }
  function intStr(n) { return n.toLocaleString('en-US'); }

  /* ---------- hyperparameter grid and the torus action ---------- */
  function vals(c) {
    return { s: Math.pow(2, c.ks), etaA: LR_BASE * Math.pow(2, c.kA), etaB: LR_BASE * Math.pow(2, c.kB), sigA: Math.pow(2, c.kS) };
  }
  function copy(c) { return { ks: c.ks, kA: c.kA, kB: c.kB, kS: c.kS }; }
  function same(a, b) { return a.ks === b.ks && a.kA === b.kA && a.kB === b.kB && a.kS === b.kS; }
  function inRange(c) {
    for (var i = 0; i < KEYS.length; i++) { var k = KEYS[i]; if (c[k] < RANGE[k][0] || c[k] > RANGE[k][1]) return false; }
    return true;
  }
  /* c = (2^a, 2^b) acting on the exponents */
  function act(c, a, b, opt) {
    var e = opt === 'sgd' ? 2 : 1;
    return { ks: c.ks - a - b, kA: c.kA + e * a, kB: c.kB + e * b, kS: c.kS + a };
  }
  /* coordinates of cur relative to ref on the reference orbit (a, b) = (log2 c_A, log2 c_B) */
  function orbitCoords(ref, cur, opt) {
    var da = cur.kA - ref.kA, db = cur.kB - ref.kB;
    if (opt === 'sgd') {
      var a = da / 2, b = db / 2;
      var on = da % 2 === 0 && db % 2 === 0 && cur.ks === ref.ks - a - b && cur.kS === ref.kS + a;
      return { a: a, b: b, on: on };
    }
    return { a: da, b: db, on: cur.ks === ref.ks - da - db && cur.kS === ref.kS + da };
  }
  /* LoRA+ (ratio rho = eta_B/eta_A != 1) -> plain LoRA with alpha absorbed; and back */
  function absorb(c, opt) {
    var rho = c.kB - c.kA;
    if (opt === 'sgd') return rho % 2 === 0 ? act(c, 0, -rho / 2, opt) : null;
    return act(c, 0, -rho, opt);
  }
  function split(c, opt, lam) {
    var L = Math.round(Math.log(lam) / Math.LN2);
    return act(c, 0, opt === 'sgd' ? L / 2 : L, opt);
  }
  function pairFor(opt, lam, base) {
    var L = Math.round(Math.log(lam) / Math.LN2);
    var ref = { ks: base.ks, kA: base.kA, kB: base.kA + L, kS: base.kS };
    while (ref.kB > RANGE.kB[1]) { ref.kA--; ref.kB--; }
    var cur = absorb(ref, opt);
    /* absorbing lambda raises s on the LoRA side: shift both runs down in s so the pair stays on the sliders */
    while (cur.ks > RANGE.ks[1] && ref.ks > RANGE.ks[0]) { ref.ks--; cur.ks--; }
    return { ref: ref, cur: cur };
  }
  function methodName(c) {
    var rho = c.kB - c.kA;
    if (rho === 0) return 'LoRA';
    return 'LoRA+ λ = ' + fracStr(rho);
  }

  /* ---------- seeded toy regression ---------- */
  function makeData(A, seed) {
    var rnd = A.rng(seed);
    var W0 = new Float64Array(MN), X = new Float64Array(NI * NS), U = new Float64Array(RK * NI), Ws = new Float64Array(MN), Y = new Float64Array(M * NS);
    var i, j, k, q;
    for (i = 0; i < MN; i++) W0[i] = rnd.normal() / Math.sqrt(NI);
    for (i = 0; i < NI * NS; i++) X[i] = rnd.normal();
    var us = [], vs = [];
    for (k = 0; k < 2; k++) {
      var u = [], v = [];
      for (i = 0; i < M; i++) u.push(rnd.normal());
      for (i = 0; i < NI; i++) v.push(rnd.normal());
      for (j = 0; j < k; j++) {
        var du = 0, dv = 0;
        for (i = 0; i < M; i++) du += u[i] * us[j][i];
        for (i = 0; i < M; i++) u[i] -= du * us[j][i];
        for (i = 0; i < NI; i++) dv += v[i] * vs[j][i];
        for (i = 0; i < NI; i++) v[i] -= dv * vs[j][i];
      }
      var nu = 0, nv = 0;
      for (i = 0; i < M; i++) nu += u[i] * u[i];
      for (i = 0; i < NI; i++) nv += v[i] * v[i];
      nu = Math.sqrt(nu); nv = Math.sqrt(nv);
      us.push(u.map(function (x) { return x / nu; }));
      vs.push(v.map(function (x) { return x / nv; }));
    }
    for (i = 0; i < M; i++) for (j = 0; j < NI; j++) {
      var w = W0[i * NI + j];
      for (k = 0; k < 2; k++) w += TARGET_SV[k] * us[k][i] * vs[k][j];
      Ws[i * NI + j] = w;
    }
    for (i = 0; i < M; i++) for (q = 0; q < NS; q++) {
      var acc = 0;
      for (j = 0; j < NI; j++) acc += Ws[i * NI + j] * X[j * NS + q];
      Y[i * NS + q] = acc;
    }
    for (i = 0; i < RK * NI; i++) U[i] = rnd.normal();
    return { W0: W0, X: X, Y: Y, U: U, seed: seed };
  }

  /* ---------- the optimizer, by hand ---------- */
  function adamStep(P, g, Mm, V, eta, c1, c2, eps) {
    for (var q = 0; q < P.length; q++) {
      Mm[q] = B1 * Mm[q] + (1 - B1) * g[q];
      V[q] = B2 * V[q] + (1 - B2) * g[q] * g[q];
      var mh = Mm[q] / c1, den = Math.sqrt(V[q] / c2) + eps;
      P[q] -= eta * (den > 0 ? mh / den : 0);   // 0/0 = 0 (only at eps = 0 with an all-zero gradient history)
    }
  }
  /* train rho(B, A) = W0 + s B A on L = ||(W0 + sBA)X - Y||_F^2 / 2N; returns W_0..W_T and the losses */
  function simulate(D, c, opt) {
    var v = vals(c), s = v.s, eA = v.etaA, eB = v.etaB;
    var Bm = new Float64Array(M * RK), Am = new Float64Array(RK * NI);
    var mB = new Float64Array(M * RK), vB = new Float64Array(M * RK), mA = new Float64Array(RK * NI), vA = new Float64Array(RK * NI);
    var W = new Float64Array(MN), Res = new Float64Array(M * NS), G = new Float64Array(MN);
    var gB = new Float64Array(M * RK), gA = new Float64Array(RK * NI);
    var traj = new Float64Array((T + 1) * MN), loss = new Float64Array(T + 1);
    var i, j, k, q, t, acc, L;
    for (i = 0; i < RK * NI; i++) Am[i] = v.sigA * D.U[i];           // A0 = sigma_A U,  B0 = 0
    var steps = T + 1, clipped = 0, wmax = 0;
    for (t = 0; t <= T; t++) {
      for (i = 0; i < M; i++) for (j = 0; j < NI; j++) {
        acc = 0;
        for (k = 0; k < RK; k++) acc += Bm[i * RK + k] * Am[k * NI + j];
        W[i * NI + j] = D.W0[i * NI + j] + s * acc;
      }
      L = 0;
      for (i = 0; i < M; i++) for (q = 0; q < NS; q++) {
        acc = -D.Y[i * NS + q];
        for (j = 0; j < NI; j++) acc += W[i * NI + j] * D.X[j * NS + q];
        Res[i * NS + q] = acc;
        L += acc * acc;
      }
      L /= 2 * NS;
      if (!(L <= 1e12)) { steps = t; break; }                          // diverged: GD at large rates (Adam's steps are bounded)
      traj.set(W, t * MN);
      loss[t] = L;
      var wn = 0;
      for (i = 0; i < MN; i++) wn += W[i] * W[i];
      wn = Math.sqrt(wn);
      if (wn > wmax) wmax = wn;
      if (t === T) break;
      for (i = 0; i < M; i++) for (j = 0; j < NI; j++) {
        acc = 0;
        for (q = 0; q < NS; q++) acc += Res[i * NS + q] * D.X[j * NS + q];
        G[i * NI + j] = acc / NS;                                       // dL/dW
      }
      for (i = 0; i < M; i++) for (k = 0; k < RK; k++) {
        acc = 0;
        for (j = 0; j < NI; j++) acc += G[i * NI + j] * Am[k * NI + j];
        gB[i * RK + k] = s * acc;                                       // dL/dB = s G A^T
      }
      for (k = 0; k < RK; k++) for (j = 0; j < NI; j++) {
        acc = 0;
        for (i = 0; i < M; i++) acc += Bm[i * RK + k] * G[i * NI + j];
        gA[k * NI + j] = s * acc;                                       // dL/dA = s B^T G
      }
      if (opt.clip) {
        var nn = 0;
        for (i = 0; i < gB.length; i++) nn += gB[i] * gB[i];
        for (i = 0; i < gA.length; i++) nn += gA[i] * gA[i];
        nn = Math.sqrt(nn);
        if (nn > opt.clip) {
          var f = opt.clip / nn;
          for (i = 0; i < gB.length; i++) gB[i] *= f;
          for (i = 0; i < gA.length; i++) gA[i] *= f;
          clipped++;
        }
      }
      if (opt.wd) {                                                     // decoupled decay, PyTorch AdamW order
        var fB = 1 - eB * opt.wd, fA = 1 - eA * opt.wd;
        for (i = 0; i < Bm.length; i++) Bm[i] *= fB;
        for (i = 0; i < Am.length; i++) Am[i] *= fA;
      }
      if (opt.kind === 'adam') {
        var c1 = 1 - Math.pow(B1, t + 1), c2 = 1 - Math.pow(B2, t + 1);   // bias correction
        adamStep(Bm, gB, mB, vB, eB, c1, c2, opt.eps);
        adamStep(Am, gA, mA, vA, eA, c1, c2, opt.eps);
      } else {
        for (i = 0; i < Bm.length; i++) Bm[i] -= eB * gB[i];
        for (i = 0; i < Am.length; i++) Am[i] -= eA * gA[i];
      }
    }
    return { W: traj, loss: loss, steps: steps, clipped: clipped, wmax: wmax };
  }
  function compare(a, b) {
    var n = Math.min(a.steps, b.steps), gaps = new Float64Array(n), mx = 0, eq = 0;
    for (var t = 0; t < n; t++) {
      var s2 = 0, o = t * MN;
      for (var q = 0; q < MN; q++) {
        var d = a.W[o + q] - b.W[o + q];
        if (d === 0) eq++;
        s2 += d * d;
      }
      var g = Math.sqrt(s2);
      gaps[t] = g;
      if (g > mx) mx = g;
    }
    var mismatch = a.steps !== b.steps;
    return { gaps: gaps, max: mismatch ? Infinity : mx, eq: eq, total: n * MN, n: n, mismatch: mismatch };
  }

  /* ---------- scoped CSS ---------- */
  function injectCSS() {
    if (document.getElementById('css-orbit')) return;
    var F = '[data-figure="orbit"] ';
    var css = [
      F + '.orb-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.orb-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.orb-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.orb-title i{color:var(--tide)}',
      F + '.orb-title .pl{font-family:var(--f-display);font-weight:var(--w-head);margin-left:.02em}',
      F + '.orb-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.orb-instr{margin:.45rem 0 .8rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:66ch}',
      F + '.orb-instr b{color:var(--ink);font-weight:600}',
      F + '.orb-formula{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem 1.5rem;padding:.5rem .75rem;margin:0 0 1rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.9rem;line-height:1.5;color:var(--ink);overflow-x:auto}',
      F + '.orb-formula > span{display:inline-flex;flex-wrap:wrap;align-items:center;gap:.15rem .5rem;min-width:0}',
      F + '.orb-formula mjx-container{white-space:nowrap}',
      F + '.orb-formula mjx-container svg{max-width:none}',
      F + '.nt{text-transform:none;letter-spacing:0}',
      F + '.orb-tag{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.orb-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.6rem;margin-bottom:1.1rem}',
      F + '.orb-group{min-width:0;border-top:1px solid var(--rule);padding-top:.5rem}',
      F + '.orb-gl{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;margin-bottom:.5rem}',
      F + '.orb-gl .orb-tag{color:var(--ink-2)}',
      F + '.orb-sliders{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6rem .9rem}',
      F + '.orb-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.orb-sl-top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0 .4rem;min-width:0}',
      F + '.orb-sl label{font-family:var(--f-body);font-size:var(--fs-sm);font-style:italic;color:var(--ink);white-space:nowrap}',
      F + '.orb-sl label .nm{font-family:var(--f-ui);font-weight:500;font-style:normal;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.3rem}',
      F + '.orb-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.orb-sl input[type=range]{margin:0;height:1.25rem}',
      F + '.orb-sl .orb-ref{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      F + '.orb-sl .orb-ref .r.diff{color:var(--ochre-ink)}',
      F + '.orb-sl .orb-ref .al{color:var(--ink)}',
      F + '.orb-padwrap{display:grid;justify-items:center;gap:.45rem}',
      F + '.orb-pad{display:block;width:100%;max-width:236px;height:auto;cursor:pointer;border-radius:4px;touch-action:manipulation}',
      F + '.orb-pad:focus{outline:none}',
      F + '.orb-pad:focus-visible{outline:2px solid var(--ochre);outline-offset:3px}',
      F + '.orb-steps{display:flex;flex-wrap:wrap;justify-content:center;gap:.35rem .9rem;font-family:var(--f-ui);font-size:var(--fs-xs);color:var(--ink-2)}',
      F + '.orb-steps > span{display:inline-flex;align-items:center;gap:.3rem}',
      F + '.orb-steps i{font-family:var(--f-body);font-size:1rem;color:var(--ink)}',
      F + '.orb-mini{font-family:var(--f-ui);font-weight:500;font-size:.8rem;line-height:1;min-width:2.3rem;padding:.3rem .4rem;border:1px solid var(--rule);background:var(--paper);color:var(--ink);border-radius:4px;cursor:pointer;font-variant-numeric:tabular-nums}',
      F + '.orb-mini:hover:not(:disabled){border-color:var(--tide);color:var(--tide)}',
      F + '.orb-mini:disabled{opacity:.35;cursor:not-allowed}',
      F + '.orb-hint{font-family:var(--f-ui);font-size:var(--fs-xs);font-variant-numeric:tabular-nums;color:var(--ink-2);text-align:center}',
      F + '.orb-jump{display:grid;gap:.45rem;margin-bottom:.8rem}',
      F + '.orb-jumprow{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem .7rem}',
      F + '.orb-jumpbtn{font-family:var(--f-ui);font-weight:500;font-size:.875rem;letter-spacing:.02em;padding:.5rem .85rem;border-radius:4px;border:1px solid var(--tide);background:var(--tide);color:var(--paper);cursor:pointer;white-space:nowrap}',
      F + '.orb-jumpbtn:hover:not(:disabled){filter:brightness(1.08)}',
      F + '.orb-jumpbtn:disabled{opacity:.45;cursor:not-allowed}',
      F + '.orb-jumpsub{font-family:var(--f-body);font-size:.84rem;color:var(--ink-2);line-height:1.4}',
      F + '.orb-row{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .6rem;margin:.35rem 0}',
      F + '.orb-row .orb-k{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);min-width:5.6rem}',
      F + '.orb-tog{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.02em;padding:.36rem .7rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap}',
      F + '.orb-tog[aria-pressed="true"]{background:var(--seal);border-color:var(--seal);color:var(--paper)}',
      F + '.orb-tog:hover:not([aria-pressed="true"]){border-color:var(--ink-3);color:var(--ink)}',
      F + '.orb-seg-break button[aria-pressed="true"]{background:var(--seal)}',
      F + '.orb-seg-break.is-zero button[aria-pressed="true"]{background:var(--tide)}',
      F + '.seg button:disabled{opacity:.4;cursor:not-allowed}',
      F + '.orb-util{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.85rem}',
      F + '.orb-util .btn{font-weight:500;padding:.34rem .6rem}',
      F + '.orb-results{display:grid;grid-template-columns:minmax(0,1fr);gap:1.1rem 1.5rem;align-items:start}',
      F + '.orb-legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;margin:0 0 .35rem;font-family:var(--f-ui);font-size:var(--fs-xs);font-variant-numeric:tabular-nums;line-height:1.35;color:var(--ink-2)}',
      F + '.orb-li{display:inline-flex;align-items:center;gap:.4rem;font:inherit;color:inherit;background:none;border:0;padding:.1rem 0;text-align:left}',
      F + 'button.orb-li{cursor:pointer;border-radius:3px}',
      F + 'button.orb-li[aria-pressed="false"]{opacity:.45}',
      F + 'button.orb-li[aria-pressed="false"] .orb-lt{text-decoration:line-through}',
      F + '.orb-li b{font-weight:500;color:var(--ink)}',
      F + '.orb-li svg{flex:none;overflow:visible}',
      F + '.orb-chart svg,.orb-gap svg,.orb-bar svg{display:block;width:100%;height:auto;overflow:visible}',
      F + '.orb-chart{touch-action:pan-y}',
      F + '.orb-why{margin:.65rem 0 0;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2)}',
      F + '.orb-why .lead{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;margin-right:.45rem}',
      F + '.orb-why .lead.z{color:var(--moss-ink)}',
      F + '.orb-why .lead.b{color:var(--seal-ink)}',
      F + '.orb-why .lead.o{color:var(--ink-2)}',
      F + '.orb-panel{display:grid;gap:.55rem;align-content:start;min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.85rem .95rem .9rem}',
      F + '.orb-mlabel{font-size:.95rem;color:var(--ink-2);line-height:1.3}',
      F + '.orb-big{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-weight:500;font-size:clamp(2.1rem,1.5rem + 2.6cqi,2.9rem);line-height:1;letter-spacing:-.02em;color:var(--ink);white-space:nowrap}',
      F + '.orb-big.z{color:var(--moss)}',
      F + '.orb-big.b{color:var(--seal)}',
      F + '.orb-chipline{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem .5rem;font-family:var(--f-mono);font-size:.78rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.orb-chip{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;padding:.14rem .5rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap}',
      F + '.orb-chip.z{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.orb-chip.b{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.orb-chip.o{color:var(--ink-2);background:var(--paper-2)}',
      F + '.orb-ctl{font-family:var(--f-mono);font-size:.78rem;color:var(--ink-2);font-variant-numeric:tabular-nums;line-height:1.45}',
      F + '.orb-ctl b{font-weight:500;color:var(--ink)}',
      F + '.orb-sep{border:0;border-top:1px solid var(--rule);margin:.2rem 0}',
      F + 'table.orb-inv{width:100%;border-collapse:collapse;font-size:.8rem;line-height:1.35}',
      F + 'table.orb-inv th,' + F + 'table.orb-inv td{position:static;background:none;padding:.3rem .3rem;border-bottom:1px solid var(--rule);vertical-align:middle}',
      F + 'table.orb-inv tr:last-child td,' + F + 'table.orb-inv tr:last-child th{border-bottom:0}',
      F + 'table.orb-inv thead th{font-family:var(--f-ui);font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);font-weight:500}',
      F + 'table.orb-inv tbody th{font-family:var(--f-body);font-weight:400;color:var(--ink);text-align:left;white-space:nowrap}',
      F + 'table.orb-inv td{font-family:var(--f-mono);font-variant-numeric:tabular-nums;text-align:right;color:var(--ink);white-space:nowrap}',
      F + 'table.orb-inv td.eq{width:1.4rem;text-align:center;font-weight:600}',
      F + 'table.orb-inv td.eq.z{color:var(--moss)}',
      F + 'table.orb-inv td.eq.o{color:var(--seal)}',
      F + '.orb-foot{margin-top:1rem;padding-top:.55rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.8rem;line-height:1.55;color:var(--ink-2)}',
      F + '.orb-foot a{color:var(--ink-2)}',
      F + 'sup{font-size:.72em;line-height:0;vertical-align:.45em}',
      F + '.orb-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
      '@container (max-width: 520px){',
      F + '.orb-sl label .nm{display:none}',
      '}',
      '@container (min-width: 700px){',
      F + '.orb-controls{grid-template-columns:minmax(0,1.05fr) minmax(0,.86fr) minmax(0,1.2fr)}',
      F + '.orb-sliders{grid-template-columns:minmax(0,1fr);gap:.5rem}',
      F + '.orb-results{grid-template-columns:minmax(0,1.62fr) minmax(0,1fr)}',
      '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-orbit';
    s.textContent = css;
    document.head.appendChild(s);
  }

  Atlas.register('orbit', function (el, A) {
    injectCSS();
    var d3 = window.d3;
    var h = A.h;

    /* ---------- state ---------- */
    function freshState(opt, lam, seed) {
      var p = pairFor(opt, lam, BASE[opt]);
      return { opt: opt, eps: 0, wd: false, clip: false, lam: lam, seed: seed, ref: p.ref, cur: p.cur, showCtl: true };
    }
    var st = freshState('adam', 16, SEED0);
    var firstDraw = true, padAnimate = false, lastPadPos = null;

    /* ---------- caches ---------- */
    var dataCache = {}, simCache = {}, simKeys = [];
    function getData(seed) { return dataCache[seed] || (dataCache[seed] = makeData(A, seed)); }
    function optKey(o) { return o.kind + '|' + o.eps + '|' + o.wd + '|' + o.clip; }
    function runSim(D, c, o) {
      var key = D.seed + '|' + c.ks + ',' + c.kA + ',' + c.kB + ',' + c.kS + '|' + optKey(o);
      if (simCache[key]) return simCache[key];
      var r = simulate(D, c, o);
      simCache[key] = r; simKeys.push(key);
      if (simKeys.length > 40) delete simCache[simKeys.shift()];
      return r;
    }

    /* ---------- DOM ---------- */
    var stage = h('div', { class: 'stage orb-stage', role: 'group', 'aria-label': 'Hyperparameter orbit explorer: under Adam with epsilon 0, no weight decay and no clipping, LoRA+ produces exactly the trajectory of LoRA with alpha multiplied by lambda. Two training runs, a meter for their maximum weight difference, and controls that move along the hyperparameter torus or break it.' });
    el.appendChild(stage);

    var head = h('div', { class: 'orb-head' }, [
      h('h3', { class: 'orb-title', html: 'LoRA<span class="pl">+</span> is <i>α</i> in disguise' }),
      h('span', { class: 'orb-kicker', html: 'Hyperparameter orbit · Thm&nbsp;III.10 · Cor&nbsp;III.11' })
    ]);
    var instr = h('p', { class: 'orb-instr', html: 'Press <b>LoRA+ ⇄ α</b> or walk the orbit pad, and the two training runs below stay identical to the last bit. Move a slider to leave the orbit, or switch on an ingredient under <b>Break the torus</b> to open a gap.' });
    var formula = h('div', { class: 'orb-formula', 'aria-label': 'Torus action and invariants' });
    stage.appendChild(head); stage.appendChild(instr); stage.appendChild(formula);

    var controls = h('div', { class: 'orb-controls' });
    stage.appendChild(controls);

    /* group A: sliders */
    var gA = h('div', { class: 'orb-group' }, [h('div', { class: 'orb-gl' }, [h('span', { class: 'orb-tag', text: 'Current run' }), h('span', { class: 'orb-tag', text: 'factor-of-2 grid' })])]);
    var sliderBox = h('div', { class: 'orb-sliders' });
    gA.appendChild(sliderBox);
    controls.appendChild(gA);
    var uid = 'orb' + Math.random().toString(36).slice(2, 7);
    var SL = [
      { key: 'ks', html: 's = α/r', name: 'scale', fmt: function (c) { return pow2Str(c.ks); }, alpha: true },
      { key: 'kA', html: 'η<sub>A</sub>', name: 'lr A', fmt: function (c) { return compact(vals(c).etaA); } },
      { key: 'kB', html: 'η<sub>B</sub>', name: 'lr B', fmt: function (c) { return compact(vals(c).etaB); } },
      { key: 'kS', html: 'σ<sub>A</sub>', name: 'init A₀', fmt: function (c) { return pow2Str(c.kS); } }
    ];
    SL.forEach(function (d) {
      var id = uid + '-' + d.key;
      d.input = h('input', { type: 'range', id: id, min: RANGE[d.key][0], max: RANGE[d.key][1], step: 1, value: st.cur[d.key] });
      d.out = h('output', { for: id });
      d.ref = h('div', { class: 'orb-ref' });
      var lab = h('label', { for: id, html: d.html + '<span class="nm">' + d.name + '</span>' });
      d.el = h('div', { class: 'orb-sl' }, [h('div', { class: 'orb-sl-top' }, [lab, d.out]), d.input, d.ref]);
      d.input.addEventListener('input', function () {
        st.cur[d.key] = +d.input.value;
        schedule();
      });
      sliderBox.appendChild(d.el);
    });

    /* group B: orbit pad */
    var gB = h('div', { class: 'orb-group' }, [h('div', { class: 'orb-gl' }, [h('span', { class: 'orb-tag', text: 'Orbit through the reference' })])]);
    var padWrap = h('div', { class: 'orb-padwrap' });
    var padSvgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    padSvgEl.setAttribute('class', 'orb-pad');
    padSvgEl.setAttribute('tabindex', '0');
    padSvgEl.setAttribute('role', 'group');
    padSvgEl.setAttribute('aria-label', 'Orbit pad. Each lattice point is a hyperparameter setting on the torus orbit of the reference. Arrow keys move the current run along the orbit: left and right change c_A by a factor of 2, up and down change c_B. Home returns to the reference. Click a point to jump there.');
    padWrap.appendChild(padSvgEl);
    function mini(label, aria, fn) { var b = h('button', { type: 'button', class: 'orb-mini', 'aria-label': aria, text: label }); b.addEventListener('click', fn); return b; }
    var stepBtns = {
      aDn: mini('÷2', 'Halve c_A (move along the orbit)', function () { move(-1, 0); }),
      aUp: mini('×2', 'Double c_A (move along the orbit)', function () { move(1, 0); }),
      bDn: mini('÷2', 'Halve c_B (move along the orbit)', function () { move(0, -1); }),
      bUp: mini('×2', 'Double c_B (move along the orbit)', function () { move(0, 1); })
    };
    padWrap.appendChild(h('div', { class: 'orb-steps' }, [
      h('span', null, [h('i', { html: 'c<sub>A</sub>' }), stepBtns.aDn, stepBtns.aUp]),
      h('span', null, [h('i', { html: 'c<sub>B</sub>' }), stepBtns.bDn, stepBtns.bUp])
    ]));
    var padHint = h('div', { class: 'orb-hint' });
    padWrap.appendChild(padHint);
    gB.appendChild(padWrap);
    controls.appendChild(gB);

    /* group C: jump + torus breakers */
    var gC = h('div', { class: 'orb-group' }, [h('div', { class: 'orb-gl' }, [h('span', { class: 'orb-tag', text: 'Jump' })])]);
    var jumpBtn = h('button', { type: 'button', class: 'orb-jumpbtn', text: 'LoRA+ ⇄ α' });
    var jumpSub = h('div', { class: 'orb-jumpsub' });
    var lamSeg = A.seg(LAMS.map(function (l) { return { value: l, label: 'λ = ' + l }; }), st.lam, function (v) {
      var p = pairFor(st.opt, v, { ks: st.ref.ks, kA: st.ref.kA, kS: st.ref.kS });
      st.lam = v; st.ref = p.ref; st.cur = p.cur; padAnimate = true; schedule();
    }, 'LoRA+ ratio lambda = eta_B / eta_A');
    gC.appendChild(h('div', { class: 'orb-jump' }, [h('div', { class: 'orb-jumprow' }, [jumpBtn, lamSeg.el]), jumpSub]));
    jumpBtn.addEventListener('click', function () {
      var tgt = jumpTarget(st.cur);
      if (tgt) { st.cur = tgt; padAnimate = true; schedule(); }
    });

    gC.appendChild(h('div', { class: 'orb-gl', style: 'margin-top:.2rem' }, [h('span', { class: 'orb-tag', text: 'Break the torus' })]));
    var optSeg = A.seg([{ value: 'adam', label: 'Adam' }, { value: 'sgd', label: 'GD' }], st.opt, function (v) {
      if (v === st.opt) return;
      var p = pairFor(v, st.lam, BASE[v]);
      st.opt = v; st.ref = p.ref; st.cur = p.cur; padAnimate = false; lastPadPos = null; schedule();
    }, 'Optimizer: Adam or full-batch gradient descent');
    var epsSeg = A.seg(EPS.map(function (e) { return { value: e, label: e === 0 ? '0' : '1e' + Math.round(Math.log10(e)) }; }), st.eps, function (v) { st.eps = v; schedule(); }, 'Adam epsilon');
    epsSeg.el.classList.add('orb-seg-break');
    var wdBtn = h('button', { type: 'button', class: 'orb-tog', 'aria-pressed': 'false', text: 'decay ' + WD });
    var clipBtn = h('button', { type: 'button', class: 'orb-tog', 'aria-pressed': 'false', html: 'clip ‖g‖ ≤ ' + CLIP });
    wdBtn.setAttribute('aria-label', 'Decoupled weight decay ' + WD + ' (AdamW style)');
    clipBtn.setAttribute('aria-label', 'Global-norm gradient clipping at ' + CLIP);
    wdBtn.addEventListener('click', function () { st.wd = !st.wd; schedule(); });
    clipBtn.addEventListener('click', function () { st.clip = !st.clip; schedule(); });
    gC.appendChild(h('div', { class: 'orb-row' }, [h('span', { class: 'orb-k', text: 'optimizer' }), optSeg.el]));
    var epsRow = h('div', { class: 'orb-row' }, [h('span', { class: 'orb-k', html: 'Adam <span class="nt">ε</span>' }), epsSeg.el]);
    gC.appendChild(epsRow);
    gC.appendChild(h('div', { class: 'orb-row' }, [h('span', { class: 'orb-k', text: 'extras' }), wdBtn, clipBtn]));
    var pinBtn = h('button', { type: 'button', class: 'btn', text: 'Pin as ref', 'aria-label': 'Pin the current run as the new reference' });
    var seedBtn = h('button', { type: 'button', class: 'btn' });
    var resetBtn = h('button', { type: 'button', class: 'btn', text: 'Reset' });
    pinBtn.addEventListener('click', function () { st.ref = copy(st.cur); padAnimate = false; lastPadPos = null; schedule(); });
    seedBtn.addEventListener('click', function () { st.seed = st.seed % 97 + 1; firstDraw = true; schedule(); });
    resetBtn.addEventListener('click', function () {
      st = freshState('adam', 16, SEED0);
      optSeg.set('adam'); lamSeg.set(16); epsSeg.set(0);
      padAnimate = false; lastPadPos = null; firstDraw = true; schedule();
    });
    gA.appendChild(h('div', { class: 'orb-util' }, [pinBtn, seedBtn, resetBtn]));
    controls.appendChild(gC);

    /* results */
    var results = h('div', { class: 'orb-results' });
    stage.appendChild(results);
    var left = h('div', { style: 'min-width:0' });
    var legend = h('div', { class: 'orb-legend' });
    var chartBox = h('div', { class: 'orb-chart' });
    var gapBox = h('div', { class: 'orb-gap' });
    var why = h('p', { class: 'orb-why', 'aria-live': 'polite' });
    left.appendChild(legend); left.appendChild(chartBox); left.appendChild(gapBox); left.appendChild(why);
    results.appendChild(left);

    var panel = h('div', { class: 'orb-panel' });
    var mLabel = h('div', { class: 'orb-mlabel', html: '\\(\\displaystyle\\max_{0\\le t\\le ' + T + '}\\,\\lVert W_t-W\'_t\\rVert_F\\)' });
    var big = h('div', { class: 'orb-big', 'aria-live': 'polite' });
    var chipLine = h('div', { class: 'orb-chipline' });
    var barBox = h('div', { class: 'orb-bar' });
    var ctlLine = h('div', { class: 'orb-ctl' });
    var invTable = h('table', { class: 'orb-inv' });
    var clipLine = h('div', { class: 'orb-ctl' });
    panel.appendChild(h('span', { class: 'orb-tag', text: 'Measured gap, reference vs current' }));
    panel.appendChild(mLabel); panel.appendChild(big); panel.appendChild(chipLine); panel.appendChild(barBox); panel.appendChild(ctlLine);
    panel.appendChild(h('hr', { class: 'orb-sep' }));
    panel.appendChild(invTable); panel.appendChild(clipLine);
    results.appendChild(panel);

    var foot = h('div', { class: 'orb-foot' });
    stage.appendChild(foot);
    var live = h('div', { class: 'orb-live', 'aria-live': 'polite' });
    stage.appendChild(live);

    if (!d3) {
      chartBox.appendChild(h('p', { class: 'muted', text: 'This figure needs D3 v7, which did not load.' }));
    }

    /* ---------- actions ---------- */
    function jumpTarget(c) {
      var rho = c.kB - c.kA, t;
      if (rho !== 0) t = absorb(c, st.opt); else t = split(c, st.opt, st.lam);
      return t && inRange(t) ? t : null;
    }
    function move(a, b) {
      var t = act(st.cur, a, b, st.opt);
      if (!inRange(t)) return;
      st.cur = t; padAnimate = true; schedule();
    }
    padSvgEl.addEventListener('keydown', function (e) {
      var k = e.key, done = true;
      if (k === 'ArrowLeft') move(-1, 0);
      else if (k === 'ArrowRight') move(1, 0);
      else if (k === 'ArrowUp') move(0, 1);
      else if (k === 'ArrowDown') move(0, -1);
      else if (k === 'Home') { st.cur = copy(st.ref); padAnimate = true; schedule(); }
      else done = false;
      if (done) e.preventDefault();
    });

    var pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      var raf = 0, tmo = 0;
      var run = function () { if (!pending) return; pending = false; if (raf && window.cancelAnimationFrame) cancelAnimationFrame(raf); clearTimeout(tmo); update(); };
      if (window.requestAnimationFrame) raf = requestAnimationFrame(run);
      tmo = setTimeout(run, 50);   // hidden tabs throttle rAF; never wait longer than 50 ms
    }

    /* ---------- typeset helpers ---------- */
    var texCache = new WeakMap();
    function setTex(node, html) {
      if (texCache.get(node) === html) return;
      texCache.set(node, html);
      if (window.MathJax && MathJax.typesetClear) { try { MathJax.typesetClear([node]); } catch (e) { /* ignore */ } }
      node.innerHTML = html;
      A.typeset(node);
    }

    /* ---------- compute + render ---------- */
    var R = null;
    function compute() {
      var D = getData(st.seed);
      var o = { kind: st.opt, eps: st.opt === 'adam' ? st.eps : 0, wd: st.wd ? WD : 0, clip: st.clip ? CLIP : 0 };
      var ctlCfg = { ks: st.ref.ks, kA: st.cur.kA, kB: st.cur.kB, kS: st.cur.kS };
      var rRef = runSim(D, st.ref, o), rCur = runSim(D, st.cur, o), rCtl = runSim(D, ctlCfg, o);
      var oc = orbitCoords(st.ref, st.cur, st.opt);
      var g = compare(rRef, rCur), gc = compare(rRef, rCtl);
      var breakers = [];
      if (o.eps) breakers.push('eps');
      if (o.wd) breakers.push('wd');
      if (o.clip && (rRef.clipped || rCur.clipped)) breakers.push('clip');
      var status = same(st.ref, st.cur) ? 'same' : !oc.on ? 'off' : g.max === 0 ? 'zero' : 'broken';
      return { D: D, o: o, ref: rRef, cur: rCur, ctl: rCtl, ctlCfg: ctlCfg, oc: oc, g: g, gc: gc, status: status, breakers: breakers };
    }

    function update() {
      R = compute();
      renderControls();
      renderFormula();
      renderLegend();
      if (d3) { renderPad(); renderChart(); renderGap(); renderBar(); }
      renderPanel();
      renderWhy();
      renderFoot();
      firstDraw = false;
      padAnimate = false;
    }

    function renderControls() {
      SL.forEach(function (d) {
        d.input.value = st.cur[d.key];
        var txt = d.fmt(st.cur);
        d.out.textContent = txt;
        d.input.setAttribute('aria-valuetext', d.alpha ? txt + ', alpha ' + String(vals(st.cur).s * RK) : txt);
        var diff = st.cur[d.key] !== st.ref[d.key];
        d.ref.innerHTML = (d.alpha ? '<span class="al">α = ' + String(vals(st.cur).s * RK) + '</span> · ' : '') +
          '<span class="r' + (diff ? ' diff' : '') + '">ref ' + d.fmt(st.ref) + (diff ? '' : ' · same') + '</span>';
      });
      var t = jumpTarget(st.cur), rho = st.cur.kB - st.cur.kA;
      jumpBtn.disabled = !t;
      if (rho !== 0) {
        var lamTxt = fracStr(rho);
        var sk = st.opt === 'sgd' ? (rho % 2 === 0 ? rho / 2 : null) : rho;
        jumpSub.innerHTML = t
          ? 'Absorb <i>λ</i> = ' + lamTxt + ' into <i>α</i>. (<i>s</i>, <i>η</i>, <i>λη</i>) becomes (' + mulSym(sk, '<i>s</i>') + ', <i>η</i>, <i>η</i>)' + (st.opt === 'sgd' ? ', since GD needs <i>s</i>√<i>λ</i>' : '') + '.'
          : (sk === null ? 'Under GD the jump needs √<i>λ</i>, so <i>λ</i> must be a power of 4 to land on a grid point.' : 'The jump would leave the slider range.');
      } else {
        var Lk = Math.round(Math.log(st.lam) / Math.LN2), sk2 = st.opt === 'sgd' ? Lk / 2 : Lk;
        jumpSub.innerHTML = t
          ? 'Split <i>α</i> into LoRA+ with <i>λ</i> = ' + st.lam + '. (<i>s</i>, <i>η</i>, <i>η</i>) becomes (' + mulSym(-sk2, '<i>s</i>') + ', <i>η</i>, ' + st.lam + '<i>η</i>).'
          : 'The jump would leave the slider range.';
      }
      jumpBtn.setAttribute('aria-label', 'LoRA+ to alpha jump: ' + jumpSub.textContent);
      wdBtn.setAttribute('aria-pressed', String(st.wd));
      clipBtn.setAttribute('aria-pressed', String(st.clip));
      var isAdam = st.opt === 'adam';
      epsSeg.el.querySelectorAll('button').forEach(function (b) { b.disabled = !isAdam; });
      epsSeg.el.classList.toggle('is-zero', st.eps === 0);
      epsRow.style.opacity = isAdam ? '' : '.55';
      epsRow.title = isAdam ? '' : 'epsilon is an Adam ingredient';
      seedBtn.textContent = 'Seed ' + st.seed + ' ↻';
      seedBtn.setAttribute('aria-label', 'Resample the toy data; current seed ' + st.seed);
      pinBtn.disabled = same(st.ref, st.cur);
      // orbit steppers
      stepBtns.aDn.disabled = !inRange(act(st.cur, -1, 0, st.opt));
      stepBtns.aUp.disabled = !inRange(act(st.cur, 1, 0, st.opt));
      stepBtns.bDn.disabled = !inRange(act(st.cur, 0, -1, st.opt));
      stepBtns.bUp.disabled = !inRange(act(st.cur, 0, 1, st.opt));
    }

    function renderFormula() {
      var sgd = st.opt === 'sgd';
      var actTex = '\\(c\\cdot(s,\\eta_A,\\eta_B,\\sigma_A)=\\)' + (sgd
        ? '\\(\\big(\\tfrac{s}{c_Ac_B},\\,c_A^2\\eta_A,\\,c_B^2\\eta_B,\\,c_A\\sigma_A\\big)\\)'
        : '\\(\\big(\\tfrac{s}{c_Ac_B},\\,c_A\\eta_A,\\,c_B\\eta_B,\\,c_A\\sigma_A\\big)\\)');
      var invTex = sgd ? '\\(I_1=s^2\\eta_A\\eta_B,\\ \\ I_2=\\sigma_A^2/\\eta_A\\)' : '\\(I_1=s\\,\\eta_A\\eta_B,\\ \\ I_2=\\sigma_A/\\eta_A\\)';
      var jTex = sgd ? '\\((s,\\eta,\\lambda\\eta)\\sim(\\sqrt{\\lambda}\\,s,\\eta,\\eta)\\)' : '\\((s,\\eta,\\lambda\\eta)\\sim(\\lambda s,\\eta,\\eta)\\)';
      setTex(formula,
        '<span><span class="orb-tag">torus (' + (sgd ? 'GD' : 'Adam') + ')</span>' + actTex + '</span>' +
        '<span><span class="orb-tag">invariants</span>' + invTex + '</span>' +
        '<span><span class="orb-tag">jump</span>' + jTex + '</span>');
    }

    function finalLoss(r) { return r.steps === T + 1 ? sci(r.loss[T], 1) : 'diverged at step ' + r.steps; }
    function finalLossH(r) { return r.steps === T + 1 ? 'L<sub>' + T + '</sub>&nbsp;' + supHtml(sci(r.loss[T], 1)).replace(/ /g, '&nbsp;') : 'diverged at step ' + r.steps; }
    function swatch(kind) {
      var c = kind === 'ref' ? A.css('--tide') : kind === 'cur' ? A.css('--ochre') : A.css('--ink-3');
      var dash = kind === 'cur' ? '5 4' : kind === 'ctl' ? '1.5 3' : '';
      var w = kind === 'ref' ? 2.6 : kind === 'cur' ? 1.8 : 1.6;
      return '<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="' + c + '" stroke-width="' + w + '" stroke-linecap="round"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + '/></svg>';
    }
    function renderLegend() {
      legend.innerHTML = '';
      legend.appendChild(h('span', { class: 'orb-li', html: swatch('ref') + '<span><b>reference</b> · ' + methodName(st.ref) + ', s = ' + pow2Str(st.ref.ks) + ' · ' + finalLossH(R.ref) + '</span>' }));
      legend.appendChild(h('span', { class: 'orb-li', html: swatch('cur') + '<span><b>current</b> · ' + methodName(st.cur) + ', s = ' + pow2Str(st.cur.ks) + ' · ' + finalLossH(R.cur) + '</span>' }));
      var cb = h('button', { type: 'button', class: 'orb-li', 'aria-pressed': String(st.showCtl), title: 'Toggle the control curve', html: swatch('ctl') + '<span class="orb-lt"><b>control</b> · current rates, s = ' + pow2Str(st.ref.ks) + ' (α not compensated) · ' + finalLossH(R.ctl) + '</span>' });
      cb.addEventListener('click', function () { st.showCtl = !st.showCtl; schedule(); });
      legend.appendChild(cb);
    }

    /* ---------- the orbit pad ---------- */
    function padPartner() {
      var rho = st.ref.kB - st.ref.kA, L = Math.round(Math.log(st.lam) / Math.LN2);
      if (rho !== 0) {
        if (st.opt === 'sgd' && rho % 2 !== 0) return null;
        return { a: 0, b: st.opt === 'sgd' ? -rho / 2 : -rho, label: 'LoRA, α×' + (st.opt === 'sgd' ? fracStr(rho / 2) : fracStr(rho)) };
      }
      return { a: 0, b: st.opt === 'sgd' ? L / 2 : L, label: 'LoRA+ λ=' + st.lam };
    }
    function renderPad() {
      var S = 236, ml = 40, mr = 12, mt = 12, mb = 32, n = 8;
      var oc = R.oc, ca = oc.a, cb = oc.b;
      var a0 = Math.round(ca / 2) - n / 2, b0 = Math.round(cb / 2) - n / 2;
      var sx = (S - ml - mr) / n, sy = (S - mt - mb) / n;
      function X(a) { return ml + (a - a0) * sx; }
      function Y(b) { return mt + (b0 + n - b) * sy; }
      var svg = d3.select(padSvgEl).attr('viewBox', '0 0 ' + S + ' ' + S);
      /* the pad scales down with a narrow column: labels grow in viewBox units to keep their size on screen */
      var psc = Math.min(1, (padSvgEl.getBoundingClientRect().width || S) / S);
      function pf(px) { return (px / psc).toFixed(2) + 'px'; }
      var ink3 = A.css('--ink-3'), ink2 = A.css('--ink-2'), rule = A.css('--rule'), tide = A.css('--tide'), ochre = A.css('--ochre'), paper = A.css('--paper');
      svg.selectAll('*').remove();
      var g = svg.append('g');
      var halo = function (sel) { return sel.style('paint-order', 'stroke').style('stroke', paper).style('stroke-width', '3.5px').style('stroke-linejoin', 'round'); };
      // frame and grid
      g.append('rect').attr('x', ml - sx / 2).attr('y', mt - sy / 2).attr('width', n * sx + sx).attr('height', n * sy + sy).attr('rx', 4).attr('fill', paper).attr('stroke', rule);
      for (var a = a0; a <= a0 + n; a++) g.append('line').attr('x1', X(a)).attr('x2', X(a)).attr('y1', Y(b0)).attr('y2', Y(b0 + n)).attr('stroke', rule).attr('stroke-width', a === 0 ? 1 : 0.5).attr('stroke-dasharray', a === 0 ? null : '1 3');
      for (var b = b0; b <= b0 + n; b++) g.append('line').attr('y1', Y(b)).attr('y2', Y(b)).attr('x1', X(a0)).attr('x2', X(a0 + n)).attr('stroke', rule).attr('stroke-width', b === 0 ? 1 : 0.5).attr('stroke-dasharray', b === 0 ? null : '1 3');
      // axes labels
      for (a = a0; a <= a0 + n; a++) if (a % 2 === 0) g.append('text').attr('x', X(a)).attr('y', S - mb + 14).attr('text-anchor', 'middle').attr('class', 'num').style('font-size', pf(11)).style('fill', ink2).text(String(a).replace('-', '−'));
      for (b = b0; b <= b0 + n; b++) if (b % 2 === 0) g.append('text').attr('x', ml - 8).attr('y', Y(b) + 4).attr('text-anchor', 'end').attr('class', 'num').style('font-size', pf(11)).style('fill', ink2).text(String(b).replace('-', '−'));
      var xl = g.append('text').attr('x', ml + n * sx / 2).attr('y', S - 2).attr('text-anchor', 'middle').attr('class', 'serif').style('font-size', pf(12.5)).style('font-style', 'italic').style('fill', ink2);
      xl.append('tspan').text('log₂ c'); xl.append('tspan').attr('dy', 3).style('font-size', pf(9)).text('A');
      var yl = g.append('text').attr('transform', 'translate(12,' + (mt + n * sy / 2) + ') rotate(-90)').attr('text-anchor', 'middle').attr('class', 'serif').style('font-size', pf(12.5)).style('font-style', 'italic').style('fill', ink2);
      yl.append('tspan').text('log₂ c'); yl.append('tspan').attr('dy', 3).style('font-size', pf(9)).text('B');
      // path from reference to current
      var inWin = function (p) { return p.a >= a0 - 0.01 && p.a <= a0 + n + 0.01 && p.b >= b0 - 0.01 && p.b <= b0 + n + 0.01; };
      // lattice points
      var pts = [];
      for (a = a0; a <= a0 + n; a++) for (b = b0; b <= b0 + n; b++) pts.push({ a: a, b: b, ok: inRange(act(st.ref, a, b, st.opt)) });
      var dots = g.append('g');
      pts.forEach(function (p) {
        dots.append('circle').attr('cx', X(p.a)).attr('cy', Y(p.b)).attr('r', p.ok ? 2.4 : 1.1).attr('fill', p.ok ? ink3 : rule);
      });
      // partner point (the other end of the LoRA+ <-> alpha jump)
      var pp = padPartner();
      if (pp && inWin(pp) && inRange(act(st.ref, pp.a, pp.b, st.opt))) {
        var px = X(pp.a), py = Y(pp.b);
        g.append('path').attr('d', 'M' + px + ',' + (py - 6) + 'L' + (px + 6) + ',' + py + 'L' + px + ',' + (py + 6) + 'L' + (px - 6) + ',' + py + 'Z').attr('fill', 'none').attr('stroke', ink2).attr('stroke-width', 1.1);
        var right = px < S / 2 + 10;
        halo(g.append('text').attr('x', px + (right ? 8 : -8)).attr('y', py - sy / 2 + 4).attr('text-anchor', right ? 'start' : 'end').style('font-size', pf(12)).style('font-weight', 500).style('fill', ink2).text(pp.label));
      }
      // reference
      var o0 = { a: 0, b: 0 };
      if (inWin(o0)) {
        g.append('circle').attr('cx', X(0)).attr('cy', Y(0)).attr('r', 7.5).attr('fill', 'none').attr('stroke', tide).attr('stroke-width', 2);
        var rr = X(0) < S / 2 + 10;
        halo(g.append('text').attr('x', X(0) + (rr ? 11 : -11)).attr('y', Y(0) - 7).attr('text-anchor', rr ? 'start' : 'end').style('font-size', pf(12)).style('fill', tide).style('font-weight', 500).text('ref'));
      }
      // current
      var cx = X(ca), cy = Y(cb);
      var from = lastPadPos && padAnimate && !A.reducedMotion() ? lastPadPos : null;
      var link = g.append('line').attr('stroke', ochre).attr('stroke-width', 1).attr('stroke-dasharray', '2 3').attr('x1', X(0)).attr('y1', Y(0));
      var cur = g.append('circle').attr('r', 5.6).attr('stroke', ochre).attr('stroke-width', 2)
        .attr('fill', oc.on ? ochre : paper).attr('stroke-dasharray', oc.on ? null : '2 2');
      if (from) {
        link.attr('x2', from[0]).attr('y2', from[1]).transition().duration(420).ease(d3.easeCubicOut).attr('x2', cx).attr('y2', cy);
        cur.attr('cx', from[0]).attr('cy', from[1]).transition().duration(420).ease(d3.easeCubicOut).attr('cx', cx).attr('cy', cy);
      } else {
        link.attr('x2', cx).attr('y2', cy);
        cur.attr('cx', cx).attr('cy', cy);
      }
      lastPadPos = [cx, cy];
      // hit targets
      var hits = g.append('g');
      pts.forEach(function (p) {
        if (!p.ok) return;
        var cfg = act(st.ref, p.a, p.b, st.opt), v = vals(cfg);
        var c = hits.append('circle').attr('cx', X(p.a)).attr('cy', Y(p.b)).attr('r', Math.min(sx, sy) / 2).attr('fill', 'transparent').style('cursor', 'pointer');
        c.on('mouseenter', function (ev) {
          d3.select(this).attr('fill', A.css('--tide-soft'));
          A.tip.show('<div class="mono" style="font-size:.78rem;line-height:1.5">c = (' + fracStr(p.a) + ', ' + fracStr(p.b) + ')<br>s = ' + pow2Str(cfg.ks) + ', η<sub>A</sub> = ' + compact(v.etaA) + '<br>η<sub>B</sub> = ' + compact(v.etaB) + ', σ<sub>A</sub> = ' + pow2Str(cfg.kS) + '</div>', ev);
        }).on('mousemove', function (ev) { A.tip.move(ev); })
          .on('mouseleave', function () { d3.select(this).attr('fill', 'transparent'); A.tip.hide(); })
          .on('click', function () { st.cur = act(st.ref, p.a, p.b, st.opt); padAnimate = true; A.tip.hide(); schedule(); padSvgEl.focus({ preventScroll: true }); });
      });
      var cT = oc.on ? 'c = (' + fracStr(ca) + ', ' + fracStr(cb) + ')' : 'off the orbit';
      padHint.innerHTML = cT + '&nbsp; · &nbsp;←→ <i>c</i><sub>A</sub>&nbsp; ↑↓ <i>c</i><sub>B</sub>';
    }

    /* ---------- loss chart ---------- */
    var lossSvg = null, gapSvg = null, barSvg = null;
    function logTicks(lo, hi, maxN) {
      var e0 = Math.round(Math.log10(lo)), e1 = Math.round(Math.log10(hi)), span = e1 - e0;
      var step = Math.max(1, Math.ceil(span / maxN)), out = [];
      for (var e = e1; e >= e0; e -= step) out.push(e);
      return out;
    }
    function renderChart() {
      var W = Math.max(260, chartBox.clientWidth || 600), H = W < 460 ? 214 : 250;
      var mg = { l: 44, r: 10, t: 27, b: 30 };
      if (!lossSvg) lossSvg = d3.select(chartBox).append('svg').attr('role', 'img');
      var svg = lossSvg.attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H);
      svg.attr('aria-label', 'Training loss on a log scale over ' + T + ' steps: reference final loss ' + finalLoss(R.ref) + ', current ' + finalLoss(R.cur) + ', control ' + finalLoss(R.ctl) + '.');
      svg.selectAll('*').remove();
      var ink2 = A.css('--ink-2'), ink3 = A.css('--ink-3'), rule = A.css('--rule'), tide = A.css('--tide'), ochre = A.css('--ochre');
      var series = [R.ref, R.cur].concat(st.showCtl ? [R.ctl] : []);
      var lo = Infinity, hi = 0;
      series.forEach(function (r) { for (var t = 0; t < r.steps; t++) { var v = r.loss[t]; if (v > 0 && v < lo) lo = v; if (v > hi) hi = v; } });
      if (!isFinite(lo)) { lo = 1e-6; hi = 1; }
      lo = Math.max(lo, 1e-30);
      var ylo = Math.pow(10, Math.floor(Math.log10(lo))), yhi = Math.pow(10, Math.ceil(Math.log10(hi)));
      if (yhi / ylo < 100) ylo = yhi / 100;
      var x = d3.scaleLinear().domain([0, T]).range([mg.l, W - mg.r]);
      var y = d3.scaleLog().domain([ylo, yhi]).range([H - mg.b, mg.t]);
      var gAx = svg.append('g');
      logTicks(ylo, yhi, H < 230 ? 4 : 6).forEach(function (e) {
        var yy = y(Math.pow(10, e));
        gAx.append('line').attr('x1', mg.l).attr('x2', W - mg.r).attr('y1', yy).attr('y2', yy).attr('stroke', rule).attr('stroke-width', 0.75);
        pow10T(gAx.append('text').attr('x', mg.l - 6).attr('y', yy + 4).attr('text-anchor', 'end').attr('class', 'num').style('font-size', '11px').style('fill', ink2), e, 11);
      });
      var xt = W < 460 ? [0, 100, 200, 300] : [0, 50, 100, 150, 200, 250, 300];
      xt.forEach(function (t) {
        gAx.append('line').attr('x1', x(t)).attr('x2', x(t)).attr('y1', H - mg.b).attr('y2', H - mg.b + 4).attr('stroke', ink3);
        gAx.append('text').attr('x', x(t)).attr('y', H - mg.b + 15).attr('text-anchor', 'middle').attr('class', 'num').style('font-size', '11px').style('fill', ink2).text(t);
      });
      gAx.append('line').attr('x1', mg.l).attr('x2', W - mg.r).attr('y1', H - mg.b).attr('y2', H - mg.b).attr('stroke', ink3).attr('stroke-width', 0.75);
      gAx.append('text').attr('x', W - mg.r).attr('y', H - 1).attr('text-anchor', 'end').style('font-size', '12px').style('font-weight', 500).style('fill', ink2).text('step t');
      var yl = gAx.append('text').attr('x', 2).attr('y', 12).style('font-size', '12px').style('font-weight', 500).style('fill', ink2);
      yl.append('tspan').text('loss L(W'); yl.append('tspan').attr('dy', 3).style('font-size', '9px').text('t'); yl.append('tspan').attr('dy', -3).text(') · log scale');
      var clipId = uid + '-clip';
      var cp = svg.append('defs').append('clipPath').attr('id', clipId).append('rect').attr('x', mg.l - 2).attr('y', 0).attr('height', H).attr('width', W - mg.l - mg.r + 4);
      var lines = svg.append('g').attr('clip-path', 'url(#' + clipId + ')');
      function path(r) {
        return d3.line().defined(function (t) { return t < r.steps && r.loss[t] > 0; }).x(function (t) { return x(t); }).y(function (t) { return y(Math.max(r.loss[t], ylo)); })(d3.range(T + 1));
      }
      if (st.showCtl) lines.append('path').attr('d', path(R.ctl)).attr('fill', 'none').attr('stroke', ink3).attr('stroke-width', 1.6).attr('stroke-dasharray', '1.5 3').attr('stroke-linecap', 'round');
      lines.append('path').attr('d', path(R.ref)).attr('fill', 'none').attr('stroke', tide).attr('stroke-width', 2.6).attr('stroke-linejoin', 'round');
      lines.append('path').attr('d', path(R.cur)).attr('fill', 'none').attr('stroke', ochre).attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4').attr('stroke-linejoin', 'round');
      if (firstDraw && !A.reducedMotion()) {
        var fullW = W - mg.l - mg.r + 4;
        cp.attr('width', 0).transition().duration(1100).ease(d3.easeCubicInOut).attr('width', fullW);
        setTimeout(function () { cp.interrupt().attr('width', fullW); }, 1400);   // rAF can be throttled (hidden tab): never leave curves clipped
      }
      // hover read-out
      var hov = svg.append('g').style('pointer-events', 'none').style('display', 'none');
      var hl = hov.append('line').attr('y1', mg.t).attr('y2', H - mg.b).attr('stroke', ink2).attr('stroke-width', 0.75).attr('stroke-dasharray', '2 2');
      var hd1 = hov.append('circle').attr('r', 3.5).attr('fill', tide);
      var hd2 = hov.append('circle').attr('r', 2.4).attr('fill', ochre);
      svg.append('rect').attr('x', mg.l).attr('y', mg.t).attr('width', W - mg.l - mg.r).attr('height', H - mg.t - mg.b).attr('fill', 'transparent')
        .on('pointermove', function (ev) {
          var p = d3.pointer(ev, this), t = Math.max(0, Math.min(T, Math.round(x.invert(p[0]))));
          hov.style('display', null);
          hl.attr('x1', x(t)).attr('x2', x(t));
          var okR = t < R.ref.steps, okC = t < R.cur.steps;
          hd1.style('display', okR ? null : 'none'); hd2.style('display', okC ? null : 'none');
          if (okR) hd1.attr('cx', x(t)).attr('cy', y(Math.max(R.ref.loss[t], ylo)));
          if (okC) hd2.attr('cx', x(t)).attr('cy', y(Math.max(R.cur.loss[t], ylo)));
          var gv = t < R.g.n ? gapStr(R.g.gaps[t]) : '—';
          A.tip.show('<div class="mono" style="font-size:.76rem;line-height:1.55">step ' + t + '<br><span style="color:' + tide + '">reference</span> ' + (okR ? supHtml(sci(R.ref.loss[t], 3)) : '—') + '<br><span style="color:' + ochre + '">current</span>&nbsp;&nbsp; ' + (okC ? supHtml(sci(R.cur.loss[t], 3)) : '—') + '<br>‖W<sub>t</sub> − W′<sub>t</sub>‖<sub>F</sub> ' + supHtml(gv) + '</div>', ev);
        })
        .on('pointerleave', function () { hov.style('display', 'none'); A.tip.hide(); });
    }

    /* ---------- gap strip: ||W_t - W'_t||_F per step ---------- */
    var GAP_LO = -16, GAP_HI = 1;
    function renderGap() {
      var W = Math.max(260, gapBox.clientWidth || 600), H = 112;
      var mg = { l: 44, r: 10, t: 20, b: 4 }, floorH = 16;
      if (!gapSvg) gapSvg = d3.select(gapBox).append('svg').attr('role', 'img');
      var svg = gapSvg.attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H);
      svg.attr('aria-label', 'Per-step weight gap between the runs, log scale with an exact-zero floor. Maximum ' + gapStr(R.g.max) + '.');
      svg.selectAll('*').remove();
      var ink3 = A.css('--ink-3'), rule = A.css('--rule'), moss = A.css('--moss'), seal = A.css('--seal'), ink2 = A.css('--ink-2'), paper3 = A.css('--paper-3');
      var x = d3.scaleLinear().domain([0, T]).range([mg.l, W - mg.r]);
      var yTop = mg.t, yBot = H - mg.b - floorH - 4;
      var y = d3.scaleLog().domain([Math.pow(10, GAP_LO), Math.pow(10, GAP_HI)]).range([yBot, yTop]).clamp(true);
      var yFloor = H - mg.b - floorH / 2;
      svg.append('rect').attr('x', mg.l).attr('y', H - mg.b - floorH).attr('width', W - mg.l - mg.r).attr('height', floorH).attr('fill', paper3).attr('rx', 2);
      svg.append('text').attr('x', mg.l - 6).attr('y', yFloor + 4).attr('text-anchor', 'end').attr('class', 'num').style('font-size', '11px').style('fill', ink2).text('≡ 0');
      [-15, -10, -5, 0].forEach(function (e) {
        var yy = y(Math.pow(10, e));
        svg.append('line').attr('x1', mg.l).attr('x2', W - mg.r).attr('y1', yy).attr('y2', yy).attr('stroke', rule).attr('stroke-width', 0.6);
        pow10T(svg.append('text').attr('x', mg.l - 6).attr('y', yy + 4).attr('text-anchor', 'end').attr('class', 'num').style('font-size', '11px').style('fill', ink2), e, 11);
      });
      var tl = svg.append('text').attr('x', 2).attr('y', 12).style('font-size', '12px').style('font-weight', 500).style('fill', ink2);
      tl.append('tspan').text('‖W'); tl.append('tspan').attr('dy', 3).style('font-size', '9px').text('t');
      tl.append('tspan').attr('dy', -3).text(' − W′'); tl.append('tspan').attr('dy', 3).style('font-size', '9px').text('t');
      tl.append('tspan').attr('dy', -3).text('‖'); tl.append('tspan').attr('dy', 3).style('font-size', '9px').text('F');
      tl.append('tspan').attr('dy', -3).text(' per step');
      function yv(v) { return v === 0 ? yFloor : y(v); }
      function path(gp) { return d3.line().x(function (t) { return x(t); }).y(function (t) { return yv(gp.gaps[t]); })(d3.range(gp.n)); }
      if (st.showCtl && R.gc.n) svg.append('path').attr('d', path(R.gc)).attr('fill', 'none').attr('stroke', ink3).attr('stroke-width', 1.4).attr('stroke-dasharray', '1.5 3').attr('stroke-linecap', 'round');
      var col = R.status === 'zero' || R.status === 'same' ? moss : R.status === 'broken' ? seal : ink2;
      if (R.g.n) svg.append('path').attr('d', path(R.g)).attr('fill', 'none').attr('stroke', col).attr('stroke-width', R.g.max === 0 ? 2.6 : 1.8).attr('stroke-linejoin', 'round');
      if (R.g.max === 0) svg.append('text').attr('x', W - mg.r).attr('y', 12).attr('text-anchor', 'end').style('font-size', '12px').style('fill', A.css('--moss-ink')).style('font-weight', 500).text('≡ 0 at all ' + R.g.n + ' steps');
    }

    /* ---------- the meter ---------- */
    var BAR_LO = -16, BAR_HI = 1;
    function renderBar() {
      var W = Math.max(220, barBox.clientWidth || 300), H = 50;
      if (!barSvg) barSvg = d3.select(barBox).append('svg').attr('aria-hidden', 'true');
      var svg = barSvg.attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H);
      svg.selectAll('*').remove();
      var ink3 = A.css('--ink-3'), ink2 = A.css('--ink-2'), rule = A.css('--rule'), moss = A.css('--moss'), seal = A.css('--seal'), ink = A.css('--ink'), paper2 = A.css('--paper-2'), ochre = A.css('--ochre');
      var z0 = 4, zW = 22, gap = 24, x0 = z0 + zW + gap, x1 = W - 10, by = 18, bh = 9;
      var x = d3.scaleLog().domain([Math.pow(10, BAR_LO), Math.pow(10, BAR_HI)]).range([x0, x1]).clamp(true);
      svg.append('rect').attr('x', z0).attr('y', by).attr('width', zW).attr('height', bh).attr('rx', 2).attr('fill', paper2).attr('stroke', rule);
      svg.append('rect').attr('x', x0).attr('y', by).attr('width', x1 - x0).attr('height', bh).attr('rx', 2).attr('fill', paper2).attr('stroke', rule);
      // float64 round-off scale of the weights: max_t ||W_t||_F * 2^-52
      var ro = Math.max(R.ref.wmax, R.cur.wmax) * Math.pow(2, -52);
      var xr = x(ro);
      svg.append('line').attr('x1', xr).attr('x2', xr).attr('y1', by - 3).attr('y2', by + bh + 3).attr('stroke', ink3).attr('stroke-dasharray', '1.5 1.5');
      var roE = ro.toExponential(0).split('e');
      var roT = svg.append('text').attr('x', xr + 3).attr('y', by - 5).style('font-size', '12px').style('fill', ink2).text('round-off ' + roE[0] + '\u2009×\u2009');
      pow10T(roT.append('tspan'), +roE[1], 12);
      var roW = 0; try { roW = roT.node().getComputedTextLength(); } catch (er) { roW = 120; }
      svg.append('text').attr('x', z0 + zW / 2).attr('y', by + bh + 14).attr('text-anchor', 'middle').attr('class', 'num').style('font-size', '11px').style('fill', ink2).text('0');
      [-16, -12, -8, -4, 0].forEach(function (e) {
        var xx = x(Math.pow(10, e));
        svg.append('line').attr('x1', xx).attr('x2', xx).attr('y1', by + bh).attr('y2', by + bh + 3).attr('stroke', ink3);
        pow10T(svg.append('text').attr('x', xx).attr('y', by + bh + 14).attr('text-anchor', 'middle').attr('class', 'num').style('font-size', '11px').style('fill', ink2), e, 11);
      });
      var v = R.g.max, col = R.status === 'zero' || R.status === 'same' ? moss : R.status === 'broken' ? seal : ink;
      var mx = v === 0 ? z0 + zW / 2 : !isFinite(v) ? x1 : x(v);
      if (v > 0) svg.append('rect').attr('x', x0).attr('y', by + 1).attr('width', Math.max(0, mx - x0)).attr('height', bh - 2).attr('rx', 1.5).attr('fill', col).attr('opacity', 0.28);
      else svg.append('rect').attr('x', z0 + 1).attr('y', by + 1).attr('width', zW - 2).attr('height', bh - 2).attr('rx', 1.5).attr('fill', col).attr('opacity', 0.35);
      svg.append('path').attr('d', 'M' + mx + ',' + (by - 1) + 'l-5,-8h10z').attr('fill', col);
      svg.append('line').attr('x1', mx).attr('x2', mx).attr('y1', by - 1).attr('y2', by + bh + 1).attr('stroke', col).attr('stroke-width', 2);
      // control marker
      if (st.showCtl && isFinite(R.gc.max) && R.gc.max > 0) {
        var cx = x(R.gc.max);
        svg.append('line').attr('x1', cx).attr('x2', cx).attr('y1', by).attr('y2', by + bh).attr('stroke', ink3).attr('stroke-width', 1.5).attr('stroke-dasharray', '1.5 1.5');
        /* 'control' ends at its marker: clear of the meter's arrow and of the round-off label */
        if (Math.abs(cx - mx) > 50 && (cx < xr - 2 || cx - 48 > xr + 3 + roW)) svg.append('text').attr('x', cx - 2).attr('y', by - 5).attr('text-anchor', 'end').style('font-size', '12px').style('fill', ink2).text('control');
      }
    }

    function renderPanel() {
      var g = R.g, cls = R.status === 'zero' || R.status === 'same' ? 'z' : R.status === 'broken' ? 'b' : '';
      big.className = 'orb-big ' + cls;
      var gs = gapStr(g.max), mm = gs.match(/^(.*) × 10(.*)$/);
      if (mm) {
        var ex = mm[2].split('').map(function (ch) { for (var k in SUP) if (SUP[k] === ch) return k === '-' ? '−' : k; return ch; }).join('');
        big.innerHTML = A.esc(mm[1]) + '<span style="font-size:.62em;margin:0 .12em">×</span>10<sup style="font-size:.5em;vertical-align:.95em;margin-left:.04em">' + ex + '</sup>';
      } else big.textContent = gs;
      big.setAttribute('aria-label', 'maximum weight gap ' + (g.max === 0 ? 'exactly zero' : String(g.max)));
      var chip;
      if (R.status === 'zero') chip = '<span class="orb-chip z">exactly zero</span>';
      else if (R.status === 'same') chip = '<span class="orb-chip z">same point</span>';
      else if (R.status === 'broken') chip = '<span class="orb-chip b">torus broken</span>';
      else chip = '<span class="orb-chip o">off the orbit</span>';
      var bits = intStr(g.eq) + ' / ' + intStr(g.total) + ' entries of W bit-identical';
      if (g.mismatch) bits = 'a run diverged, so the gap is unbounded';
      chipLine.innerHTML = chip + '<span>' + bits + '</span>';
      ctlLine.innerHTML = st.showCtl
        ? 'control (α not compensated): <b>' + supHtml(gapStr(R.gc.max)) + '</b>'
        : 'control curve hidden';
      // invariants
      var sgd = st.opt === 'sgd', vr = vals(st.ref), vc = vals(st.cur);
      function inv(v) { return sgd ? [v.s * v.s * v.etaA * v.etaB, v.sigA * v.sigA / v.etaA] : [v.s * v.etaA * v.etaB, v.sigA / v.etaA]; }
      var ir = inv(vr), ic = inv(vc);
      var eq1 = sgd ? 2 * st.ref.ks + st.ref.kA + st.ref.kB === 2 * st.cur.ks + st.cur.kA + st.cur.kB : st.ref.ks + st.ref.kA + st.ref.kB === st.cur.ks + st.cur.kA + st.cur.kB;
      var eq2 = sgd ? 2 * st.ref.kS - st.ref.kA === 2 * st.cur.kS - st.cur.kA : st.ref.kS - st.ref.kA === st.cur.kS - st.cur.kA;
      var key = 'inv-' + st.opt;
      if (invTable.getAttribute('data-k') !== key) {
        invTable.setAttribute('data-k', key);
        invTable.innerHTML = '<thead><tr><th scope="col"><span class="sr-only">invariant</span></th><th scope="col" style="text-align:right">reference</th><th scope="col" style="text-align:right">current</th><th scope="col"><span class="sr-only">equal</span></th></tr></thead>' +
          '<tbody><tr><th scope="row">' + (sgd ? '\\(I_1=s^2\\eta_A\\eta_B\\)' : '\\(I_1=s\\,\\eta_A\\eta_B\\)') + '</th><td class="r1"></td><td class="c1"></td><td class="eq e1"></td></tr>' +
          '<tr><th scope="row">' + (sgd ? '\\(I_2=\\sigma_A^2/\\eta_A\\)' : '\\(I_2=\\sigma_A/\\eta_A\\)') + '</th><td class="r2"></td><td class="c2"></td><td class="eq e2"></td></tr>' +
          '<tr><th scope="row">\\(c=(c_A,c_B)\\)</th><td class="r3">(1, 1)</td><td class="c3"></td><td class="eq e3"></td></tr></tbody>';
        if (window.MathJax && MathJax.typesetClear) { try { MathJax.typesetClear([invTable]); } catch (e) { /* ignore */ } }
        A.typeset(invTable);
      }
      var q = function (s) { return invTable.querySelector(s); };
      q('.r1').textContent = compact(ir[0]); q('.c1').textContent = compact(ic[0]);
      q('.r2').textContent = compact(ir[1]); q('.c2').textContent = compact(ic[1]);
      var e1 = q('.e1'), e2 = q('.e2');
      e1.textContent = eq1 ? '=' : '≠'; e1.className = 'eq e1 ' + (eq1 ? 'z' : 'o');
      e2.textContent = eq2 ? '=' : '≠'; e2.className = 'eq e2 ' + (eq2 ? 'z' : 'o');
      q('.c3').textContent = R.oc.on ? '(' + fracStr(R.oc.a) + ', ' + fracStr(R.oc.b) + ')' : 'off orbit';
      var parts = [];
      if (R.o.clip) parts.push('clipping fired on ' + R.ref.clipped + ' / ' + T + ' reference steps, ' + R.cur.clipped + ' / ' + T + ' current steps');
      if (R.ref.steps < T + 1) parts.push('reference diverged at step ' + R.ref.steps);
      if (R.cur.steps < T + 1) parts.push('current diverged at step ' + R.cur.steps);
      clipLine.textContent = parts.join(' · ');
      clipLine.style.display = parts.length ? '' : 'none';
      live.textContent = 'Meter ' + gapStr(g.max) + '. ' + (R.status === 'zero' ? 'Exactly zero: the runs are identical.' : R.status === 'broken' ? 'On the orbit but the torus is broken.' : R.status === 'off' ? 'Off the orbit.' : 'Same hyperparameters.');
    }

    function renderWhy() {
      var html, oc = R.oc;
      if (R.status === 'zero') {
        /* the current run is the reference rescaled by c = (2^a, 2^b): A -> c_A A, B -> c_B B, s -> s/(c_A c_B) */
        var parts = [];
        if (oc.a !== 0) parts.push('\\(A\\mapsto ' + mulSym(oc.a, 'A') + '\\)');
        if (oc.b !== 0) parts.push('\\(B\\mapsto ' + mulSym(oc.b, 'B') + '\\)');
        parts.push('\\(s\\mapsto ' + mulSym(-oc.a - oc.b, 's') + '\\)');
        var lead = '<span class="lead z">Why 0</span>The current run is the reference rescaled by \\(c=(' + texFrac(oc.a) + ',' + texFrac(oc.b) + ')\\), that is ' + parts.join(', ') + ', so the weights \\(W_0+sBA\\) are unchanged. ';
        html = lead + (st.opt === 'adam'
          ? 'Each block’s gradients are divided by its \\(c_i\\), Adam’s step \\(\\hat m/\\sqrt{\\hat v}\\) ignores that, and its rate is multiplied by \\(c_i\\), so each step rescales with its block. '
          : 'Each block’s gradients are divided by its \\(c_i\\), a GD step is rate times gradient, and its rate is multiplied by \\(c_i^2\\), so each step rescales with its block. ') +
          'The starts match since \\(B_0=0\\) and \\(\\sigma_A\\mapsto c_A\\sigma_A\\). Powers of two rescale floats without rounding, so the runs agree bit for bit.';
      } else if (R.status === 'broken') {
        var reasons = [];
        if (R.breakers.indexOf('eps') >= 0) reasons.push('\\(\\varepsilon\\) is added to \\(\\sqrt{\\hat v}\\) and does not rescale with the gradients');
        if (R.breakers.indexOf('wd') >= 0) reasons.push('decay multiplies each block by \\(1-\\eta\\,\\mathrm{wd}\\), which changes when \\(\\eta\\) is rescaled');
        if (R.breakers.indexOf('clip') >= 0) reasons.push('clipping scales both blocks by one factor, set by their joint norm');
        var cap = reasons.length ? reasons.join('; ') : 'a torus-breaking ingredient is active';
        html = '<span class="lead b">Why not 0</span>' + cap.charAt(0).toUpperCase() + cap.slice(1) + '. Setting \\(\\varepsilon\\), the decay and the clipping threshold per block, rescaled as Thm III.10 prescribes, restores the match. Prediction X.1 puts any gain of LoRA+ over a re-tuned \\(\\alpha\\) in ingredients like these.';
      } else if (R.status === 'off') {
        html = '<span class="lead o">Off the orbit</span>The invariants \\((I_1,I_2)\\) differ, so no element of the torus relates the two runs and the meter measures a real difference. Click a lattice point on the orbit pad, or press Home on it, to return to the reference orbit.';
      } else {
        html = '<span class="lead o">Same point</span>Reference and current have identical hyperparameters. Walk the orbit pad or press LoRA+ ⇄ α to compare two different settings.';
      }
      setTex(why, html);
    }

    function renderFoot() {
      var lp = A.method && A.method('lora-plus');
      var cite = lp ? ' · LoRA+: ' + A.esc(lp.authors || 'Hayou et al.') + ' ' + A.esc(String(lp.year || '')) + (lp.arxiv ? ', <a href="https://arxiv.org/abs/' + A.esc(lp.arxiv) + '" target="_blank" rel="noopener">arXiv ' + A.esc(lp.arxiv) + '</a>' : '') : '';
      var optTxt = st.opt === 'adam'
        ? 'Adam β = (' + B1 + ', ' + B2 + '), bias-corrected, ε = ' + (st.eps === 0 ? '0 (0/0 := 0)' : String(st.eps))
        : 'full-batch gradient descent, no momentum';
      foot.innerHTML = 'Toy: W₀ ∈ ℝ<sup>' + M + '×' + NI + '</sup>, X ∈ ℝ<sup>' + NI + '×' + NS + '</sup>, Y = W*X with rank(W* − W₀) = 2 · L = ‖(W₀ + sBA)X − Y‖²<sub>F</sub>/2N · r = ' + RK + ', B₀ = 0, A₀ = σ<sub>A</sub>U · ' +
        optTxt + ' · toggles: decoupled decay ' + WD + ' (' + (st.wd ? 'on' : 'off') + '), global-norm clip at ' + CLIP + ' (' + (st.clip ? 'on' : 'off') + ') · ' + T + ' steps in float64, seed ' + st.seed + ' · LoRA/Adam statement after Schulman et al. 2025' + cite + '.';
    }

    /* ---------- test hook (read state; apply a state change and redraw) ---------- */
    el.__orbit = {
      get: function () { return { st: st, R: R }; },
      apply: function (fn) { fn(st); optSeg.set(st.opt); lamSeg.set(st.lam); epsSeg.set(st.eps); update(); return { st: st, R: R }; },
      simulate: function (seed, c, o) { return simulate(getData(seed), c, o); }
    };

    /* ---------- boot ---------- */
    update();
    A.typeset(mLabel);
    A.onTheme(function () { padAnimate = false; update(); });
    if ('ResizeObserver' in window) {
      var lastW = results.clientWidth;
      new ResizeObserver(A.debounce(function () {
        var w = results.clientWidth;
        if (Math.abs(w - lastW) < 2) return;
        lastW = w; padAnimate = false; update();
      }, 60)).observe(results);
    } else {
      window.addEventListener('resize', A.debounce(update, 100));
    }
  });
})();
