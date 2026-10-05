/* Fig. "apex" — The Apex and the Smooth Point.
   theory/framework.md: Def I.5 (apex = positive first-order deficit), Thm II.2 (three strata of LoRA initialisations),
   Prop II.3 (first-order rank formula), Thm II.7 (HRA's paired initialisation is an apex), Prop V.2(a) (QLoRA is LoRA
   pointed at kappa(theta0), with defect e).

   The 3D picture is the symmetric 2x2 slice  S = [[x+y, z], [z, x-y]]  <->  (x, y, z).  On it
       ||S||_F^2 = 2 (x^2 + y^2 + z^2)      (the coordinates are conformal: angles and projections are Frobenius-exact)
       det S = x^2 - y^2 - z^2              (rank <= 1  <=>  the double cone x^2 = y^2 + z^2)
       eig S = x +- sqrt(y^2 + z^2)          (so dist_F(S, rank <= 1) = | |x| - sqrt(y^2+z^2) |, Eckart-Young).
   The illustrative method is the one-parameter-per-factor rank-one model  rho(c, phi) = W_res + c v v^T,
   v = (cos phi, sin phi); c plays B and v plays A.  d rho = [ v v^T ,  c (v vdot^T + vdot v^T) ].
     zero-init : q0 = (0, phi0),  W_res = theta0          -> S_1 = R v0 v0^T (a line, one ruling); deficit 1: apex.
     split     : q0 = (c0, phi0), W_res = theta0 - P,  P = c0 v0 v0^T -> S_1 = tangent plane; deficit 0: smooth point.
     QLoRA     : q0 = (0, phi0),  W_res = kappa(theta0)  -> starts at kappa(theta0), again a vertex; defect e.
   The counters are the exact formulas of Thm II.2(b), Thm II.2(e) and Thm II.7(b), and a Jacobian-rank check
   (run in the browser on mount) reproduces the [Num] lines of the theory. */
(function () {
  'use strict';

  /* ---------- fixed setup ---------- */
  var W0 = [0.62, 0.36, 0.88];          // theta0 = W0 = [[0.62, 0.36], [0.36, 0.88]]  (symmetric 2x2 stored as [a, b, d])
  var H = 0.62;                         // drawn half-height of the cone: t = c/2 in [-H, H]
  var DEF = { mode: 'zero', phi: 45, c0: 0.8, grid: 0.25, az: 30, el: 20 };
  var T_PHI = 100, T_C = 1.0;            // default target T = W0 + T_C v v^T (a rank-one update of W0)
  var SNAP = 0.02;                      // Frobenius distance under which a dragged T is placed exactly on Im
  var TMAX = 0.78;                      // |T - vertex| (slice units) is kept <= TMAX so T stays on stage
  var GRIDS = [0.125, 0.25, 0.5];
  var DIMS = [16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 8192, 11008, 14336, 16384];
  var RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  var PRESETS = [
    { label: '4096 × 4096', sub: 'q_proj, LLaMA-2-7B', m: 4096, n: 4096 },
    { label: '11008 × 4096', sub: 'up_proj', m: 11008, n: 4096 },
    { label: '4096 × 11008', sub: 'down_proj', m: 4096, n: 11008 }
  ];
  var MODES = [
    { value: 'zero', label: 'Zero-init LoRA' },
    { value: 'split', label: 'Split · PiSSA' },
    { value: 'qlora', label: 'QLoRA' }
  ];
  var DEG = Math.PI / 180;

  /* ---------- symmetric 2x2 algebra: S = [a, b, d] = [[a, b], [b, d]] ---------- */
  function add(S, T) { return [S[0] + T[0], S[1] + T[1], S[2] + T[2]]; }
  function sub(S, T) { return [S[0] - T[0], S[1] - T[1], S[2] - T[2]]; }
  function scl(S, k) { return [S[0] * k, S[1] * k, S[2] * k]; }
  function lerpS(S, T, u) { return add(S, scl(sub(T, S), u)); }
  function ip(S, T) { return S[0] * T[0] + 2 * S[1] * T[1] + S[2] * T[2]; }      // Frobenius inner product
  function fro(S) { return Math.sqrt(ip(S, S)); }
  function det(S) { return S[0] * S[2] - S[1] * S[1]; }
  function eig(S) { var m = (S[0] + S[2]) / 2, r = Math.hypot((S[0] - S[2]) / 2, S[1]); return [m + r, m - r]; }
  function minAbsEig(S) { var l = eig(S); return Math.min(Math.abs(l[0]), Math.abs(l[1])); }  // = dist_F(S, rank <= 1)
  function xyz(S) { return [(S[0] + S[2]) / 2, (S[0] - S[2]) / 2, S[1]]; }
  function fromXYZ(p) { return [p[0] + p[1], p[2], p[0] - p[1]]; }
  function vvT(phi) { var c = Math.cos(phi), s = Math.sin(phi); return [c * c, c * s, s * s]; }
  function symOuter(v, w) { return [2 * v[0] * w[0], v[0] * w[1] + v[1] * w[0], 2 * v[1] * w[1]]; } // v w^T + w v^T
  function vec4(S) { return [S[0], S[1], S[1], S[2]]; }
  function quant(S, g) { return S.map(function (x) { return Math.round(x / g) * g; }); }
  /* nearest point of {rank <= 1} to S in Frobenius norm (Eckart-Young: drop the eigenvalue of smaller modulus) */
  function nearestRank1(S) {
    var p = xyz(S), x = p[0], rho = Math.hypot(p[1], p[2]);
    var uy = rho > 1e-15 ? p[1] / rho : 1, uz = rho > 1e-15 ? p[2] / rho : 0;
    var t = (Math.abs(x) + rho) / 2, sx = x >= 0 ? 1 : -1;
    return fromXYZ([sx * t, t * uy, t * uz]);
  }
  function norm3(p) { return Math.hypot(p[0], p[1], p[2]); }

  /* ---------- the slice model ---------- */
  function vertexOf(mode, st) {
    if (mode === 'split') return sub(W0, scl(vvT(st.phi * DEG), st.c0));
    if (mode === 'qlora') return quant(W0, st.grid);
    return W0.slice();
  }
  function startOf(mode, st) { return mode === 'qlora' ? quant(W0, st.grid) : W0.slice(); }
  function jacobian(c, phi) {
    var v = [Math.cos(phi), Math.sin(phi)], vd = [-Math.sin(phi), Math.cos(phi)];
    return [vvT(phi), scl(symOuter(v, vd), c)];
  }
  function svalsOf(cols, A) {
    var J = [0, 1, 2, 3].map(function (i) { return [vec4(cols[0])[i], vec4(cols[1])[i]]; });
    return A.LA.svals(J);
  }
  function compute(st, A) {
    var phi = st.phi * DEG, mode = st.mode;
    var c = mode === 'split' ? st.c0 : 0;
    var start = startOf(mode, st), vertex = vertexOf(mode, st);
    var cols = jacobian(c, phi);
    var sv = svalsOf(cols, A);
    var dimS1 = A.LA.numRank(sv, 1e-9);
    var d = A.LA.numRank(svalsOf(jacobian(1, 0.3), A), 1e-9);          // d = max rank of d rho (generic point)
    /* orthonormal basis of S_1 (Frobenius), Gram-Schmidt on the non-zero columns */
    var basis = [];
    cols.forEach(function (C) {
      var u = C.slice();
      basis.forEach(function (q) { u = sub(u, scl(q, ip(u, q))); });
      var nu = fro(u);
      if (nu > 1e-9) basis.push(scl(u, 1 / nu));
    });
    var r = sub(st.T, start), proj = [0, 0, 0];
    basis.forEach(function (q) { proj = add(proj, scl(q, ip(r, q))); });
    var nr = fro(r);
    var memb = {};
    MODES.forEach(function (M) {
      var V = vertexOf(M.value, st), D = sub(st.T, V), dist = minAbsEig(D);
      memb[M.value] = { dist: dist, inIm: dist <= 1e-9 * (1 + fro(D)), det: det(D) };
    });
    var out = {
      mode: mode, phi: phi, c: c, start: start, vertex: vertex, cols: cols, sv: sv, dimS1: dimS1, d: d,
      deficit: d - dimS1, basis: basis, r: r, proj: proj, nr: nr, frac: nr > 1e-12 ? fro(proj) / nr : 1,
      resid: fro(sub(r, proj)), memb: memb, cur: memb[mode], P: scl(vvT(phi), st.c0)
    };
    if (mode === 'qlora') {
      var e = sub(start, W0);
      out.e = e; out.eFro = fro(e); out.eDet = det(e); out.eRank = Math.abs(det(e)) > 1e-12 ? 2 : (fro(e) > 1e-12 ? 1 : 0);
      out.w0Dist = minAbsEig(scl(e, -1));
    }
    return out;
  }

  /* ---------- exact counters (Thm II.2(b), Thm II.2(e), Thm II.7(b)) ---------- */
  function loraCounts(m, n, r) {
    var ok = r >= 1 && r < Math.min(m, n);
    var d = r * (m + n - r);
    return {
      ok: ok, d: d,
      rows: [
        { key: 'TA', s1: m * r, d: d },
        { key: 'TB', s1: n * r, d: d },
        { key: 'S', s1: d, d: d },
        { key: 'HFo', s1: r % 2 === 0 ? (r / 2) * (m + n - r / 2) : NaN, d: d, needEven: true }
      ]
    };
  }
  function hraCounts(m, n, r) {
    var ok = r % 2 === 0 && r >= 2 && r <= n - 2 && m >= n;
    var k = r / 2;
    var dH = r * n - r * (r + 1) / 2, s1 = k * n - k * (k + 1) / 2;
    return { ok: ok, k: k, d: dH, s1: s1, deficit: k * (2 * n - 3 * k - 1) / 2, loraTA: m * r, loraD: r * (m + n - r) };
  }

  /* ---------- Jacobian-rank check (computed on mount; mirrors theory/framework_checks.py) ---------- */
  function jacRank(cols) {
    /* numerical rank by modified Gram-Schmidt with column pivoting: residual norms are accurate to ~1e-15 of the
       largest column, so a 1e-9 relative cut separates the zero singular values (~1e-16) from the rest (>= 0.1) */
    var V = cols.map(function (c) { return Float64Array.from(c); }), L = V[0].length, rank = 0, nmax = 0, i, k;
    function nrm(v) { var s = 0; for (var t = 0; t < L; t++) s += v[t] * v[t]; return Math.sqrt(s); }
    V.forEach(function (v) { nmax = Math.max(nmax, nrm(v)); });
    while (V.length) {
      var bi = 0, bn = -1;
      for (i = 0; i < V.length; i++) { var nv = nrm(V[i]); if (nv > bn) { bn = nv; bi = i; } }
      if (bn <= 1e-9 * nmax) break;
      var q = V.splice(bi, 1)[0];
      for (k = 0; k < L; k++) q[k] /= bn;
      V.forEach(function (v) { var d = 0, t; for (t = 0; t < L; t++) d += v[t] * q[t]; for (t = 0; t < L; t++) v[t] -= d * q[t]; });
      rank++;
    }
    return rank;
  }
  function matmul(X, Y) { return Atlas.LA.mul(X, Y); }
  function loraS1Rank(B0, A0, m, n, r, A) {
    var cols = [], i, j, p, q, col;
    for (i = 0; i < m; i++) for (j = 0; j < r; j++) {           // dB = E_ij -> E_ij A0
      col = new Float64Array(m * n);
      for (q = 0; q < n; q++) col[i * n + q] = A0[j][q];
      cols.push(col);
    }
    for (i = 0; i < r; i++) for (j = 0; j < n; j++) {           // dA = E_ij -> B0 E_ij
      col = new Float64Array(m * n);
      for (p = 0; p < m; p++) col[p * n + j] = B0[p][i];
      cols.push(col);
    }
    return jacRank(cols);
  }
  function householder(u, n) {
    var H = Atlas.LA.eye(n), nn = 0, i, j;
    for (i = 0; i < n; i++) nn += u[i] * u[i];
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) H[i][j] -= 2 * u[i] * u[j] / nn;
    return H;
  }
  function hraRank(W, U, n, A) {
    var r = U.length, Hs = U.map(function (u) { return householder(u, n); }), cols = [];
    for (var i = 0; i < r; i++) {
      var Lm = Atlas.LA.eye(n), Rm = Atlas.LA.eye(n), t;
      for (t = 0; t < i; t++) Lm = matmul(Lm, Hs[t]);
      for (t = i + 1; t < r; t++) Rm = matmul(Rm, Hs[t]);
      var WL = matmul(W, Lm), u = U[i], nn = 0, a;
      for (a = 0; a < n; a++) nn += u[a] * u[a];
      for (var j = 0; j < n; j++) {
        // dH_u[e_j] = -2 (e_j u^T + u e_j^T)/|u|^2 + 4 u_j u u^T/|u|^4
        var D = Atlas.LA.zeros(n, n), p, q;
        for (p = 0; p < n; p++) for (q = 0; q < n; q++) {
          D[p][q] = -2 * ((p === j ? u[q] : 0) + (q === j ? u[p] : 0)) / nn + 4 * u[j] * u[p] * u[q] / (nn * nn);
        }
        var M = matmul(matmul(WL, D), Rm), col = new Float64Array(W.length * n);
        for (p = 0; p < W.length; p++) for (q = 0; q < n; q++) col[p * n + q] = M[p][q];
        cols.push(col);
      }
    }
    return jacRank(cols);
  }
  function runChecks(A) {
    var rnd = A.rng(20241), LA = A.LA, m = 7, n = 5, r = 2, res = {};
    res.TA = loraS1Rank(LA.zeros(m, r), LA.randn(r, n, rnd), m, n, r, A);
    res.TB = loraS1Rank(LA.randn(m, r, rnd), LA.zeros(r, n), m, n, r, A);
    res.S = loraS1Rank(LA.randn(m, r, rnd), LA.randn(r, n, rnd), m, n, r, A);
    /* HF init_lora_weights="orthogonal": one Q in O(r); A0 = (G_A Q_even)^T/10, B0 = G_B Q_odd/10, so B0 A0 = 0 */
    var Q = LA.randOrth(r, rnd), Qe = [], Qo = [];
    Q.forEach(function (row, i) { (i % 2 === 0 ? Qe : Qo).push(row); });
    var A0o = LA.scale(LA.T(LA.mul(LA.randn(n, r / 2, rnd), Qe)), 0.1), B0o = LA.scale(LA.mul(LA.randn(m, r / 2, rnd), Qo), 0.1);
    res.HFo = loraS1Rank(B0o, A0o, m, n, r, A);
    res.HFoProd = LA.frob(LA.mul(B0o, A0o));
    res.lora = { m: m, n: n, r: r, pred: { TA: m * r, TB: n * r, S: r * (m + n - r), HFo: (r / 2) * (m + n - r / 2) } };
    /* HRA at n = 9 (W0 is 11 x 9, injective): paired vs generic for r = 2, 4 */
    var nH = 9, W = LA.randn(nH + 2, nH, rnd);
    res.hra = [2, 4].map(function (rr) {
      var k = rr / 2, gen = [], pair = [], i, base = [];
      for (i = 0; i < rr; i++) gen.push(Array.prototype.slice.call(LA.randn(1, nH, rnd)[0]));
      for (i = 0; i < k; i++) base.push(Array.prototype.slice.call(LA.randn(1, nH, rnd)[0]));
      for (i = 0; i < rr; i++) pair.push(base[Math.floor(i / 2)].slice());
      return {
        r: rr, paired: hraRank(W, pair, nH, A), generic: hraRank(W, gen, nH, A),
        predP: k * nH - k * (k + 1) / 2, predG: rr * nH - rr * (rr + 1) / 2
      };
    });
    return res;
  }

  /* ---------- formatting ---------- */
  function intStr(x) { return isFinite(x) ? Math.round(x).toLocaleString('en-US') : '—'; }
  function f3(x) { var s = (Math.abs(x) < 5e-4 ? 0 : x).toFixed(3); return s === '-0.000' ? '0.000' : s; }
  function f4(x) { var s = (Math.abs(x) < 5e-5 ? 0 : x).toFixed(4); return s === '-0.0000' ? '0.0000' : s; }
  function fm(x) { var s = (Math.abs(x) < 5e-4 ? 0 : x).toFixed(3).replace(/0$/, ''); return s.charAt(0) === '-' ? '−' + s.slice(1) : s; }
  function signed(s) { return s.charAt(0) === '-' ? '−' + s.slice(1) : s; }
  function pct(x) { return (100 * x).toFixed(1) + '%'; }
  /* Unicode subscript digits -> <sub> (the body serif draws U+2080 like a small "o") */
  function hs(t) { return String(t).replace(/[₀-₉]+/g, function (m) { return '<sub>' + m.replace(/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 8320); }) + '</sub>'; }); }
  /* math inside an uppercase mono tag: keep its case (ρ must not become Ρ, κ must not become Κ) */
  /* legend item: the label after the swatch goes in its own span, so a <sub> inside it is not a separate flex item */
  function liText(t) { return t.replace(/(<\/svg>)([\s\S]*)(<\/span>)$/, '$1<span>$2</span>$3'); }
  function mt(t) { return '<span class="mt">' + hs(t).replace(/Im\b/g, '<span class="up">Im</span>').replace(/ᵀ/g, '<sup>⊤</sup>') + '</span>'; }

  /* ---------- scoped CSS ---------- */
  function injectCSS() {
    if (document.getElementById('css-apex')) return;
    var F = '[data-figure="apex"] ';
    var css = [
      F + '.apx-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.apx-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.apx-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.apx-title i{color:var(--tide)}',
      F + '.apx-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.apx-instr{margin:.45rem 0 .8rem;font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:68ch}',
      F + '.apx-instr b{color:var(--ink);font-weight:600}',
      F + '.apx-formula{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem 1.4rem;padding:.5rem .75rem;margin:0 0 1rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.88rem;line-height:1.5;color:var(--ink);overflow-x:auto}',
      F + '.apx-formula > span{display:inline-flex;flex-wrap:nowrap;align-items:center;gap:.2rem .5rem;min-width:0}',
      F + '.apx-formula mjx-container{white-space:nowrap}',
      F + '.apx-tag{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.mt{font-family:var(--f-body);font-weight:var(--w-body);font-style:italic;font-size:1.22em;letter-spacing:0;text-transform:none;line-height:1}',
      F + '.mt sub{font-size:.72em;line-height:0}',
      F + '.mt .up{font-style:normal}',
      F + '.apx-modes{display:flex;flex-wrap:nowrap;width:100%;border-radius:6px}',
      F + '.apx-modes button{flex:1 1 auto;white-space:nowrap;font-size:var(--fs-xs);padding:.42rem .5rem;text-align:center}',
      F + '.seg button{font-weight:500}',
      F + '.apx-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.6rem;margin-bottom:1rem}',
      F + '.apx-group{min-width:0;border-top:1px solid var(--rule);padding-top:.5rem}',
      F + '.apx-gl{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;margin-bottom:.5rem}',
      F + '.apx-gl .apx-tag{color:var(--ink-2)}',
      F + '.apx-mdesc{margin:.55rem 0 0;font-size:.86rem;line-height:1.45;color:var(--ink-2)}',
      F + '.apx-mdesc > div{display:none}',
      F + '.apx-mdesc > div.on{display:block}',
      F + '.apx-mdesc mjx-container{font-size:96%}',
      F + '.apx-point{display:grid;grid-template-columns:auto minmax(0,1fr);gap:.4rem .9rem;align-items:center}',
      F + '.apx-dial{display:block;width:96px;height:96px;cursor:grab;touch-action:none}',
      F + '.apx-dial:active{cursor:grabbing}',
      F + '.apx-sls{display:grid;gap:.55rem;min-width:0}',
      F + '.apx-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.apx-sl-top{display:flex;justify-content:space-between;align-items:baseline;gap:.4rem}',
      F + '.apx-sl label{font-family:var(--f-body);font-size:var(--fs-sm);font-style:italic;color:var(--ink);white-space:nowrap}',
      F + '.apx-sl label .nm{font-family:var(--f-ui);font-style:normal;font-weight:500;font-size:var(--fs-xs);letter-spacing:.02em;color:var(--ink-2);margin-left:.35rem}',
      F + '.apx-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.apx-sl input[type=range]{margin:0;height:1.25rem}',
      F + '.apx-sl.off{opacity:.42}',
      F + '.apx-note{font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
      F + '.apx-row{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem .6rem;margin:.1rem 0 .55rem}',
      F + '.apx-row .apx-k{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);min-width:6.4rem}',
      F + '.apx-row.off{opacity:.42}',
      F + '.seg button:disabled{cursor:not-allowed}',
      F + '.apx-btns{display:flex;flex-wrap:wrap;gap:.45rem}',
      F + '.apx-btns .btn{font-size:var(--fs-xs);font-weight:500;padding:.36rem .65rem;text-transform:none;letter-spacing:.02em}',
      F + '.apx-main{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.4rem;align-items:start}',
      F + '.apx-scene{position:relative;min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);overflow:hidden}',
      F + '.apx-scene svg{display:block;width:100%;height:auto;touch-action:pan-y;cursor:grab;user-select:none;-webkit-user-select:none}',
      F + '.apx-scene svg:active{cursor:grabbing}',
      F + '.apx-scene svg:focus{outline:none}',
      F + '.apx-scene svg:focus-visible{outline:2px solid var(--ochre);outline-offset:-3px}',
      F + '.apx-scene .apx-hint{position:absolute;right:.55rem;bottom:.4rem;font-family:var(--f-ui);font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);pointer-events:none}',
      F + '.apx-scene .apx-badge{position:absolute;left:.6rem;bottom:.4rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);pointer-events:none}',
      F + '.apx-legend{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;margin:.6rem 0 0;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);line-height:1.4;color:var(--ink-2)}',
      F + '.apx-li{display:inline-flex;align-items:center;gap:.4rem}',
      F + '.apx-li svg{flex:none;overflow:visible}',
      F + '.apx-panel{display:grid;gap:.5rem;align-content:start;min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.85rem .95rem .9rem}',
      F + '.apx-ph{display:flex;justify-content:space-between;align-items:center;gap:.5rem;flex-wrap:wrap}',
      F + '.apx-chip{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;padding:.14rem .55rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap}',
      F + '.apx-chip.m{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.apx-chip.s{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.apx-chip.o{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + '.apx-big{display:flex;align-items:baseline;gap:.55rem;flex-wrap:wrap}',
      F + '.apx-big .n{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-weight:500;font-size:clamp(1.9rem,1.4rem + 2cqi,2.5rem);line-height:1;letter-spacing:-.02em;color:var(--ink)}',
      F + '.apx-big .n.s{color:var(--seal)}',
      F + '.apx-big .n.m{color:var(--moss)}',
      F + '.apx-big .l{font-size:.86rem;color:var(--ink-2);line-height:1.3}',
      F + '.apx-mats{display:flex;flex-wrap:wrap;gap:.4rem .9rem;align-items:flex-end}',
      F + '.apx-matb{display:grid;gap:.15rem;justify-items:start}',
      F + '.apx-matb .ml{font-family:var(--f-body);font-style:italic;font-size:.82rem;color:var(--ink-2);white-space:nowrap}',
      F + '.apx-mat{position:relative;display:inline-grid;grid-template-columns:auto auto;gap:0 .55rem;padding:.08rem .4rem;font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);text-align:right;line-height:1.35}',
      F + '.apx-mat::before,' + F + '.apx-mat::after{content:"";position:absolute;top:0;bottom:0;width:.28rem;border:1.25px solid var(--ink-3)}',
      F + '.apx-mat::before{left:0;border-right:0}',
      F + '.apx-mat::after{right:0;border-left:0}',
      F + '.apx-mat.q{color:var(--seal)}',
      F + 'table.apx-tbl{width:100%;border-collapse:collapse;font-size:.78rem;line-height:1.35}',
      F + 'table.apx-tbl th,' + F + 'table.apx-tbl td{position:static;background:none;padding:.28rem .2rem;border-bottom:1px solid var(--rule);vertical-align:middle}',
      F + 'table.apx-tbl tr:last-child th,' + F + 'table.apx-tbl tr:last-child td{border-bottom:0}',
      F + 'table.apx-tbl th{font-family:var(--f-body);font-weight:400;font-size:.84rem;color:var(--ink);text-align:left;letter-spacing:0;text-transform:none}',
      F + 'table.apx-tbl th .r{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.4rem;white-space:nowrap}',
      F + 'table.apx-tbl td{font-family:var(--f-mono);font-variant-numeric:tabular-nums;text-align:right;color:var(--ink);white-space:nowrap}',
      F + 'table.apx-tbl td.s{color:var(--seal)}',
      F + 'table.apx-tbl td.m{color:var(--moss)}',
      F + '.apx-reach{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:.5rem;align-items:center}',
      F + '.apx-rl{font-size:.84rem;color:var(--ink);margin-bottom:-.2rem}',
      F + '.apx-rl .r{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.4rem}',
      F + '.apx-rbar{position:relative;height:.5rem;border-radius:999px;background:var(--paper-3);overflow:hidden}',
      F + '.apx-rbar > i{position:absolute;left:0;top:0;bottom:0;background:var(--ochre);border-radius:999px}',
      F + '.apx-memb{display:flex;flex-wrap:wrap;gap:.3rem .35rem;align-items:center}',
      F + '.apx-memb .apx-chip{font-size:.75rem;padding:.1rem .48rem;opacity:.62;text-transform:none;letter-spacing:.02em}',
      F + '.apx-memb .apx-chip.cur{opacity:1;box-shadow:0 0 0 1px currentColor}',
      F + '.apx-sep{border:0;border-top:1px solid var(--rule);margin:.15rem 0}',
      F + '.apx-why{margin:.8rem 0 0;max-width:62ch;font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2)}',
      F + '.apx-why b{color:var(--ink);font-weight:600}',
      F + '.apx-why .lead{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;margin-right:.45rem}',
      F + '.apx-why .lead.m{color:var(--moss-ink)}',
      F + '.apx-why .lead.s{color:var(--seal-ink)}',
      F + '.apx-q{display:none}',
      F + '.apx-q.on{display:grid;gap:.45rem}',
      /* counters */
      F + '.apx-count{margin-top:1.3rem;padding-top:.8rem;border-top:1px solid var(--rule)}',
      F + '.apx-ch{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.2rem 1rem;margin-bottom:.6rem}',
      F + '.apx-ch h4{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.15rem,1rem + .8cqi,1.4rem);margin:0;color:var(--ink)}',
      F + '.apx-cctl{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6rem 1rem;margin-bottom:.6rem}',
      F + '.apx-crow2{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem .7rem;margin-bottom:.85rem}',
      F + '.apx-preset{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);font-variant-numeric:tabular-nums;padding:.3rem .6rem;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);border-radius:4px;cursor:pointer;white-space:nowrap}',
      F + '.apx-preset:hover{border-color:var(--tide);color:var(--tide)}',
      F + '.apx-preset[aria-pressed="true"]{border-color:var(--tide);color:var(--tide);background:var(--tide-soft)}',
      F + '.apx-preset .s{font-weight:400;color:var(--ink-2);margin-left:.35rem}',
      F + '.apx-sentence{font-family:var(--f-body);font-size:clamp(1.02rem,.95rem + .45cqi,1.2rem);line-height:1.45;color:var(--ink);margin:0 0 .9rem;max-width:62ch}',
      F + '.apx-sentence .nb{font-family:var(--f-mono);font-size:.9em;font-variant-numeric:tabular-nums;white-space:nowrap}',
      F + '.apx-sentence .nb.s{color:var(--seal)}',
      F + '.apx-sentence .nb.m{color:var(--moss)}',
      F + '.apx-bars{display:grid;gap:.75rem}',
      F + '.apx-bars.off{display:none}',
      F + '.apx-brow{display:grid;grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"name num" "bar bar";gap:.25rem .8rem;align-items:center}',
      F + '.apx-bname{grid-area:name;min-width:0;font-size:.9rem;line-height:1.3;color:var(--ink)}',
      F + '.apx-bname .sb{display:block;font-family:var(--f-body);font-size:.82rem;color:var(--ink-2);line-height:1.35;margin-top:.05rem}',
      F + '.apx-bname mjx-container{font-size:92%}',
      F + '.apx-bnum{grid-area:num;font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);text-align:right;white-space:nowrap;line-height:1.35}',
      F + '.apx-bnum .df{display:block;font-size:.75rem;color:var(--seal-ink)}',
      F + '.apx-bnum .df.z{color:var(--moss-ink)}',
      F + '.apx-btrack{grid-area:bar;position:relative;height:.95rem;border-radius:3px;background:var(--paper-2);overflow:hidden}',
      F + '.apx-bd{position:absolute;left:0;top:0;bottom:0;border-radius:3px;border:1px solid var(--seal);background:repeating-linear-gradient(135deg,var(--seal-soft) 0 4px,transparent 4px 8px)}',
      F + '.apx-bs{position:absolute;left:0;top:0;bottom:0;border-radius:3px 0 0 3px;background:var(--moss)}',
      F + '.apx-brow.na .apx-btrack{opacity:.35}',
      F + '.apx-cwarn{font-family:var(--f-body);font-size:.88rem;line-height:1.45;color:var(--ochre-ink);margin:.1rem 0 .7rem}',
      F + '.apx-cwarn:empty{display:none}',
      F + '.apx-foot{margin-top:1.1rem;padding-top:.6rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.84rem;line-height:1.55;color:var(--ink-2)}',
      F + '.apx-foot p{margin:0 0 .45rem;max-width:none}',
      F + '.apx-foot p:last-child{margin-bottom:0}',
      F + '.apx-foot sub,' + F + '.apx-foot sup{font-size:.78em;line-height:0}',
      F + '.apx-foot b{font-weight:600;color:var(--ink)}',
      F + '.apx-foot .ok{color:var(--moss-ink)}',
      F + '.apx-foot .bad{color:var(--seal-ink)}',
      F + 'sub{font-variant-numeric:lining-nums;font-style:normal}',
      F + '.apx-svg text,' + F + '.apx-dial text{font-variant-numeric:lining-nums}',
      F + '.apx-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
      /* svg classes */
      F + '.apx-svg text{font-family:var(--f-ui);font-size:12px;fill:var(--ink-2)}',
      F + '.apx-svg text.lab{font-family:var(--f-body);font-style:italic;font-size:15px;fill:var(--ink)}',
      F + '.apx-svg text.lab.ochre{fill:var(--ochre)}',
      F + '.apx-svg text.lab.seal{fill:var(--seal)}',
      F + '.apx-svg text.lab.moss{fill:var(--moss)}',
      F + '.apx-svg text.sm{font-weight:500;font-size:12px;fill:var(--ink-2);letter-spacing:.02em}',
      F + '.apx-svg text.halo{stroke:var(--paper);stroke-width:3.2px;stroke-linejoin:round;paint-order:stroke}',
      F + '.apx-svg .rul{fill:none;stroke:var(--tide);stroke-width:1}',
      F + '.apx-svg .lat{fill:none;stroke:var(--tide);stroke-width:1}',
      F + '.apx-svg .nap{fill:var(--tide-soft);stroke:none}',
      F + '.apx-svg .axis{fill:none;stroke:var(--ink-3);stroke-width:1;stroke-dasharray:2 4}',
      F + '.apx-svg .s1line{fill:none;stroke:var(--ochre);stroke-width:3.2;stroke-linecap:round}',
      F + '.apx-svg .s1plane{fill:var(--ochre-soft);stroke:var(--ochre);stroke-width:1.25;stroke-linejoin:round}',
      F + '.apx-svg .s1ruling{fill:none;stroke:var(--ochre);stroke-width:1.6;stroke-dasharray:5 3}',
      F + '.apx-svg .arrow{fill:none;stroke:var(--ink);stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}',
      F + '.apx-svg .resid{fill:none;stroke:var(--ink-2);stroke-width:1.25;stroke-dasharray:4 3}',
      F + '.apx-svg .rmark{fill:none;stroke:var(--ink-3);stroke-width:1}',
      F + '.apx-svg .defect{fill:none;stroke:var(--seal);stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}',
      F + '.apx-svg .w0{fill:var(--paper);stroke:var(--ink);stroke-width:2}',
      F + '.apx-svg .w0dot{fill:var(--ink)}',
      F + '.apx-svg .vtx{fill:var(--paper);stroke:var(--ink-2);stroke-width:1.5}',
      F + '.apx-svg .start{fill:var(--ink);stroke:var(--paper);stroke-width:1.5}',
      F + '.apx-svg .tdot{stroke:var(--paper);stroke-width:2.2;cursor:move;touch-action:none}',
      F + '.apx-svg .tdot.m{fill:var(--moss)}',
      F + '.apx-svg .tdot.s{fill:var(--seal)}',
      F + '.apx-svg .thalo{fill:none;stroke-width:1.25;opacity:.55;pointer-events:none}',
      F + '.apx-svg .thalo.m{stroke:var(--moss)}',
      F + '.apx-svg .thalo.s{stroke:var(--seal)}',
      F + '.apx-svg .thit{fill:transparent;cursor:move;touch-action:none}',
      F + '.apx-svg g.tg:focus{outline:none}',
      F + '.apx-svg g.tg:focus-visible .thalo{stroke:var(--ochre);opacity:1;stroke-width:2}',
      F + '.apx-svg .drop{fill:none;stroke:var(--ink-3);stroke-width:1;stroke-dasharray:1 3}',
      F + '.apx-dial .ring{fill:var(--paper);stroke:var(--rule);stroke-width:1}',
      F + '.apx-dial .ax{stroke:var(--rule);stroke-width:1}',
      F + '.apx-dial .vline{stroke:var(--ochre);stroke-width:2.2;stroke-linecap:round}',
      F + '.apx-dial .vdot{fill:var(--ochre);stroke:var(--paper);stroke-width:1.5}',
      F + '.apx-dial .vneg{fill:var(--paper);stroke:var(--ochre);stroke-width:1.5}',
      F + '.apx-dial .arc{fill:none;stroke:var(--ink-3);stroke-width:1}',
      F + '.apx-dial text{font-family:var(--f-body);font-style:italic;font-size:12px;fill:var(--ink-2)}',
      F + '.apx-dial text.t{font-size:12px}',
      '@container (min-width: 600px){',
      F + '.apx-controls{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}',
      F + '.apx-controls > .apx-group:nth-child(3){grid-column:1 / -1}',
      '}',
      '@container (min-width: 860px){',
      F + '.apx-controls{grid-template-columns:minmax(0,1.25fr) minmax(0,1.15fr) minmax(0,1fr)}',
      F + '.apx-controls > .apx-group:nth-child(3){grid-column:auto}',
      '}',
      '@container (min-width: 640px){',
      F + '.apx-brow{grid-template-columns:minmax(12.5rem,15rem) minmax(0,1fr) 8.6rem;grid-template-areas:"name bar num"}',
      '}',
      '@container (min-width: 760px){',
      F + '.apx-main{grid-template-columns:minmax(0,1.5fr) minmax(0,1fr)}',
      '}',
      '@container (max-width: 460px){',
      F + '.apx-cctl{grid-template-columns:minmax(0,1fr)}',
      F + '.apx-sl label .nm{display:none}',
      '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-apex';
    s.textContent = css;
    document.head.appendChild(s);
  }

  Atlas.register('apex', function (el, A) {
    injectCSS();
    var h = A.h;
    var uid = 'apx' + Math.random().toString(36).slice(2, 7);
    var reduced = A.reducedMotion();

    /* ---------- state ---------- */
    var st = {
      mode: DEF.mode, phi: DEF.phi, c0: DEF.c0, grid: DEF.grid, az: DEF.az, el: DEF.el,
      T: add(W0, scl(vvT(T_PHI * DEG), T_C)),
      mi: DIMS.indexOf(4096), ni: DIMS.indexOf(4096), ri: RANKS.indexOf(16), hra: false
    };
    var R = null;                     // computed slice state
    var vShown = vertexOf(st.mode, st); // vertex currently drawn (tweened on mode change)
    var s1Alpha = 1, tween = null, raf = 0;

    /* ---------- DOM: head ---------- */
    var stage = h('div', { class: 'stage apx-stage', role: 'group', 'aria-label': 'The apex and the smooth point. A 3D view of the symmetric 2 by 2 slice: the rank-one cone, the base point theta zero, the first-order image S1 and a draggable target T, for zero-init LoRA, a split initialisation and QLoRA; below, exact first-order dimension counters at real matrix sizes.' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'apx-head' }, [
      h('h3', { class: 'apx-title', html: 'The apex and the <i>smooth point</i>' }),
      h('span', { class: 'apx-kicker', text: 'Pointings of LoRA · Thm II.2 · Prop II.3 · Thm II.7 · Prop V.2' })
    ]));
    stage.appendChild(h('p', { class: 'apx-instr', html: 'Pick an <b>initialisation</b>, turn the cone and drag the target <b>T</b>. The arrow is the first-order move that gets closest to T, and T turns green when the image contains it.' }));
    var formula = h('div', { class: 'apx-formula', 'aria-label': 'Slice coordinates, cone, method and metric' });
    formula.innerHTML =
      '<span><span class="apx-tag">slice</span>\\(S=\\begin{pmatrix}x+y&z\\\\ z&x-y\\end{pmatrix}\\leftrightarrow(x,y,z)\\)</span>' +
      '<span><span class="apx-tag">cone</span>\\(\\operatorname{rank}S\\le1\\iff\\det S=x^2-y^2-z^2=0\\)</span>' +
      '<span><span class="apx-tag">method</span>\\(\\rho(c,\\phi)=W_{\\mathrm{res}}+c\\,v_\\phi v_\\phi^{\\top}\\)</span>' +
      '<span><span class="apx-tag">metric</span>\\(\\|S\\|_F^2=2(x^2+y^2+z^2)\\)</span>';
    stage.appendChild(formula);

    /* ---------- DOM: controls ---------- */
    var controls = h('div', { class: 'apx-controls' });
    stage.appendChild(controls);

    // group 1: initialisation
    var g1 = h('div', { class: 'apx-group' }, [h('div', { class: 'apx-gl' }, [h('span', { class: 'apx-tag', text: 'Initialisation' }), h('span', { class: 'apx-tag', html: 'pointing ' + mt('q₀') })])]);
    var modeSeg = A.seg(MODES, st.mode, function (v) { setMode(v); }, 'Initialisation');
    modeSeg.el.classList.add('apx-modes');
    g1.appendChild(modeSeg.el);
    var mdesc = h('div', { class: 'apx-mdesc' });
    mdesc.innerHTML =
      '<div data-m="zero">LoRA’s default \\(B_0=0\\): \\(q_0=(0,\\phi_0)\\), \\(W_{\\mathrm{res}}=\\theta_0\\). The base point is the vertex of the image, and an apex.</div>' +
      '<div data-m="split">PiSSA, LoRA-GA, CorDA, OLoRA: \\(q_0=(c_0,\\phi_0)\\), \\(W_{\\mathrm{res}}=\\theta_0-P\\), \\(P=c_0v_0v_0^{\\top}\\). The vertex moves to \\(\\theta_0-P\\).</div>' +
      '<div data-m="qlora">Zero-init over a quantised base, \\(W_{\\mathrm{res}}=\\kappa\\theta_0\\). Not pointed at \\(\\theta_0\\): training starts at \\(\\kappa\\theta_0\\), with defect \\(e=\\kappa\\theta_0-\\theta_0\\).</div>';
    g1.appendChild(mdesc);
    controls.appendChild(g1);

    // group 2: pointing v0 (dial + sliders)
    var g2 = h('div', { class: 'apx-group' }, [h('div', { class: 'apx-gl' }, [h('span', { class: 'apx-tag', html: 'Pointing ' + mt('v₀') }), h('span', { class: 'apx-tag', html: mt('±v₀') + ' give one ' + mt('v₀v₀ᵀ') })])]);
    var NS = 'http://www.w3.org/2000/svg';
    var dial = document.createElementNS(NS, 'svg');
    dial.setAttribute('class', 'apx-dial');
    dial.setAttribute('viewBox', '0 0 96 96');
    dial.setAttribute('aria-hidden', 'true');
    var sliders = h('div', { class: 'apx-sls' });
    function mkSlider(key, html, name, min, max, step, val, fmt, onInput) {
      var id = uid + '-' + key;
      var input = h('input', { type: 'range', id: id, min: min, max: max, step: step, value: val });
      var out = h('output', { for: id });
      var wrap = h('div', { class: 'apx-sl' }, [
        h('div', { class: 'apx-sl-top' }, [h('label', { for: id, html: hs(html) + '<span class="nm">' + hs(name) + '</span>' }), out]),
        input
      ]);
      function refresh() { out.textContent = fmt(+input.value); }
      input.addEventListener('input', function () { refresh(); onInput(+input.value); });
      refresh();
      return { el: wrap, input: input, refresh: refresh };
    }
    var phiS = mkSlider('phi', 'φ₀', 'angle of v₀', 0, 179, 1, st.phi, function (v) { return v + '°'; }, function (v) { st.phi = v; changed(); });
    var c0S = mkSlider('c0', 'c₀', 'split scale', 0.15, 0.9, 0.05, st.c0, function (v) { return v.toFixed(2); }, function (v) { st.c0 = v; changed(); });
    sliders.appendChild(phiS.el); sliders.appendChild(c0S.el);
    g2.appendChild(h('div', { class: 'apx-point' }, [dial, sliders]));
    controls.appendChild(g2);

    // group 3: quantiser, target, view
    var g3 = h('div', { class: 'apx-group' }, [h('div', { class: 'apx-gl' }, [h('span', { class: 'apx-tag', text: 'Quantiser · target · view' })])]);
    var gridRow = h('div', { class: 'apx-row' }, [h('span', { class: 'apx-k', html: mt('κ') + ' grid step' })]);
    var gridSeg = A.seg(GRIDS.map(function (g) { return { value: g, label: String(g) }; }), st.grid, function (v) { st.grid = v; changed(); }, 'Quantisation grid step');
    gridRow.appendChild(gridSeg.el);
    g3.appendChild(gridRow);
    var btns = h('div', { class: 'apx-btns' });
    var snapBtn = h('button', { type: 'button', class: 'btn', text: 'Put T on Im', title: 'Move T to the nearest point of the current image (Eckart–Young)' });
    var rank1Btn = h('button', { type: 'button', class: 'btn', text: 'Reset T', title: 'T = θ₀ + ' + T_C + '·vvᵀ at φ = ' + T_PHI + '°: a rank-one update of θ₀' });
    var viewBtn = h('button', { type: 'button', class: 'btn', text: 'Reset view' });
    btns.appendChild(snapBtn); btns.appendChild(rank1Btn); btns.appendChild(viewBtn);
    g3.appendChild(btns);
    g3.appendChild(h('div', { class: 'apx-note', style: 'margin-top:.5rem', text: 'T keys: arrows move in the screen plane, PgUp/PgDn in depth, Enter puts T on Im.' }));
    controls.appendChild(g3);

    /* ---------- DOM: main (scene + panel) ---------- */
    var main = h('div', { class: 'apx-main' });
    stage.appendChild(main);
    var left = h('div', { style: 'min-width:0' });
    var scene = h('div', { class: 'apx-scene' });
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'apx-svg');
    svg.setAttribute('tabindex', '0');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Rotatable 3D view of the rank-one cone in the symmetric slice. Left and right arrow keys turn it; up and down tilt it.');
    var gScene = document.createElementNS(NS, 'g');
    var gT = document.createElementNS(NS, 'g');
    gT.setAttribute('class', 'tg');
    gT.setAttribute('tabindex', '0');
    gT.setAttribute('role', 'button');
    gT.setAttribute('aria-label', 'Target T. Arrow keys move it in the screen plane, Page Up and Page Down in depth, Enter puts it on the image.');
    var tHalo = document.createElementNS(NS, 'circle'); tHalo.setAttribute('r', '12');
    var tDot = document.createElementNS(NS, 'circle'); tDot.setAttribute('r', '7.5');
    var tHit = document.createElementNS(NS, 'circle'); tHit.setAttribute('r', '18'); tHit.setAttribute('class', 'thit');
    var tLab = document.createElementNS(NS, 'text');
    tLab.setAttribute('class', 'lab halo');
    tLab.textContent = 'T';
    gT.appendChild(tHalo); gT.appendChild(tDot); gT.appendChild(tLab); gT.appendChild(tHit);
    svg.appendChild(gScene); svg.appendChild(gT);
    scene.appendChild(svg);
    var badge = h('div', { class: 'apx-badge', text: 'symmetric 2×2 slice' });
    scene.appendChild(badge);
    scene.appendChild(h('div', { class: 'apx-hint', text: 'drag to turn' }));
    left.appendChild(scene);
    var legend = h('div', { class: 'apx-legend' });
    left.appendChild(legend);
    main.appendChild(left);

    var panel = h('div', { class: 'apx-panel', 'aria-live': 'off' });
    main.appendChild(panel);
    var pState = h('div', { class: 'apx-ph' }, [h('span', { class: 'apx-tag', html: 'At the start ' + mt('ρ(q₀)') })]);
    var stChip = h('span', { class: 'apx-chip' });
    pState.appendChild(stChip);
    panel.appendChild(pState);
    var bigRow = h('div', { class: 'apx-big' });
    var bigN = h('span', { class: 'n' }), bigL = h('span', { class: 'l' });
    bigRow.appendChild(bigN); bigRow.appendChild(bigL);
    panel.appendChild(bigRow);
    var mats = h('div', { class: 'apx-mats' });
    panel.appendChild(mats);
    var tbl1 = h('table', { class: 'apx-tbl' });
    tbl1.innerHTML = hs(
      '<tbody>' +
      '<tr><th>σ(dρ<sub>q₀</sub>)<span class="r">Frobenius</span></th><td data-k="sv"></td></tr>' +
      '<tr><th>dim S₁ = rank dρ<sub>q₀</sub></th><td data-k="s1"></td></tr>' +
      '<tr><th>d = max rank dρ</th><td data-k="d"></td></tr>' +
      '<tr><th>deficit<span class="r">Def I.5</span></th><td data-k="def"></td></tr>' +
      '</tbody>');
    panel.appendChild(tbl1);
    panel.appendChild(h('hr', { class: 'apx-sep' }));
    var pT = h('div', { class: 'apx-ph' }, [h('span', { class: 'apx-tag', html: 'Target ' + mt('T') })]);
    var tChip = h('span', { class: 'apx-chip' });
    pT.appendChild(tChip);
    panel.appendChild(pT);
    var reach = h('div', { class: 'apx-reach' });
    var rbar = h('div', { class: 'apx-rbar' }), rfill = h('i');
    rbar.appendChild(rfill);
    var rnum = h('span', { class: 'mono num', style: 'font-size:.8rem;color:var(--ink)' });
    reach.appendChild(rbar); reach.appendChild(rnum);
    var tbl2 = h('table', { class: 'apx-tbl' });
    tbl2.innerHTML = hs(
      '<tbody>' +
      '<tr><th>‖T − ρ(q₀)‖<sub>F</sub></th><td data-k="nr"></td></tr>' +
      '<tr><th>dist<sub>F</sub>(T, Im)<span class="r">= min<sub>i</sub> |λ<sub>i</sub>(T − vertex)|</span></th><td data-k="dist"></td></tr>' +
      '</tbody>');
    panel.appendChild(tbl2);
    panel.appendChild(h('div', { class: 'apx-rl', html: 'first-order reach <span class="r">‖Π<sub>S₁</sub>(T − ρ(q₀))‖ / ‖T − ρ(q₀)‖</span>' }));
    panel.appendChild(reach);
    var membRow = h('div', { class: 'apx-memb' }, [h('span', { class: 'apx-tag', html: mt('T ∈ Im') + ' for' })]);
    var membChips = {};
    MODES.forEach(function (M) { membChips[M.value] = h('span', { class: 'apx-chip' }); membRow.appendChild(membChips[M.value]); });
    panel.appendChild(membRow);
    var qBox = h('div', { class: 'apx-q' });
    qBox.appendChild(h('hr', { class: 'apx-sep' }));
    qBox.appendChild(h('div', { class: 'apx-ph' }, [h('span', { class: 'apx-tag', html: 'Defect ' + mt('e = κθ₀ − θ₀') + ' · Prop V.2(a)' })]));
    var tbl3 = h('table', { class: 'apx-tbl' });
    tbl3.innerHTML = hs(
      '<tbody>' +
      '<tr><th>‖e‖<sub>F</sub></th><td data-k="ef"></td></tr>' +
      '<tr><th>det e<span class="r">≠ 0 ⇒ rank e = 2 &gt; r</span></th><td data-k="ed"></td></tr>' +
      '<tr><th>dist<sub>F</sub>(θ₀, Im)</th><td data-k="ew"></td></tr>' +
      '</tbody>');
    qBox.appendChild(tbl3);
    panel.appendChild(qBox);
    var why = h('p', { class: 'apx-why' });
    left.appendChild(why);
    var live = h('div', { class: 'apx-live', 'aria-live': 'polite' });
    stage.appendChild(live);

    /* ---------- DOM: counters ---------- */
    var count = h('div', { class: 'apx-count' });
    stage.appendChild(count);
    count.appendChild(h('div', { class: 'apx-ch' }, [
      h('h4', { html: 'At real scale: <i>exact</i> first-order counts' }),
      h('span', { class: 'apx-kicker', text: 'Thm II.2(b) · Thm II.2(e) · Thm II.7(b)' })
    ]));
    var cctl = h('div', { class: 'apx-cctl' });
    count.appendChild(cctl);
    var mS = mkSlider('m', 'm', 'd_out', 0, DIMS.length - 1, 1, st.mi, function (v) { return intStr(DIMS[v]); }, function (v) { st.mi = v; updCounts(); });
    var nS = mkSlider('n', 'n', 'd_in', 0, DIMS.length - 1, 1, st.ni, function (v) { return intStr(DIMS[v]); }, function (v) { st.ni = v; updCounts(); });
    var rS = mkSlider('r', 'r', 'rank', 0, RANKS.length - 1, 1, st.ri, function (v) { return String(RANKS[v]); }, function (v) { st.ri = v; updCounts(); });
    [mS, nS, rS].forEach(function (s) { s.input.setAttribute('aria-valuetext', ''); cctl.appendChild(s.el); });
    var crow2 = h('div', { class: 'apx-crow2' });
    var famSeg = A.seg([{ value: false, label: 'LoRA strata' }, { value: true, label: 'HRA' }], st.hra, function (v) { st.hra = v; updCounts(); }, 'Counter family');
    crow2.appendChild(famSeg.el);
    var presetBtns = PRESETS.map(function (p) {
      var b = h('button', { type: 'button', class: 'apx-preset', 'aria-pressed': 'false', html: A.esc(p.label) + '<span class="s">' + A.esc(p.sub) + '</span>' });
      b.addEventListener('click', function () {
        st.mi = DIMS.indexOf(p.m); st.ni = DIMS.indexOf(p.n);
        mS.input.value = st.mi; nS.input.value = st.ni; mS.refresh(); nS.refresh(); updCounts();
      });
      crow2.appendChild(b);
      return b;
    });
    count.appendChild(crow2);
    var sentence = h('p', { class: 'apx-sentence' });
    count.appendChild(sentence);
    var cwarn = h('div', { class: 'apx-cwarn' });
    count.appendChild(cwarn);

    function barRow(nameHtml, subHtml) {
      var row = h('div', { class: 'apx-brow' });
      var name = h('div', { class: 'apx-bname', html: nameHtml + (subHtml ? '<span class="sb">' + hs(subHtml) + '</span>' : '') });
      var track = h('div', { class: 'apx-btrack' });
      var bd = h('div', { class: 'apx-bd' }), bs = h('div', { class: 'apx-bs' });
      track.appendChild(bd); track.appendChild(bs);
      var num = h('div', { class: 'apx-bnum' });
      row.appendChild(name); row.appendChild(track); row.appendChild(num);
      return { row: row, bd: bd, bs: bs, num: num };
    }
    var loraBars = h('div', { class: 'apx-bars' });
    var LB = {
      TA: barRow('Zero-init LoRA · \\(\\dim S_1=mr\\)', 'B₀ = 0 · stratum T<sub>A</sub> · apex · QLoRA: same, at κθ₀'),
      TB: barRow('\\(A_0=0\\) · \\(\\dim S_1=nr\\)', 'stratum T<sub>B</sub> (“Init[B]”) · apex'),
      S: barRow('Split · \\(\\dim S_1=r(m+n-r)\\)', 'PiSSA, LoRA-GA, CorDA, OLoRA · stratum S · smooth'),
      HFo: barRow('HF orthogonal · \\(\\tfrac r2(m+n-\\tfrac r2)\\)', 'B₀A₀ = 0, ranks (r/2, r/2) · outside the strata')
    };
    ['TA', 'TB', 'S', 'HFo'].forEach(function (k) { loraBars.appendChild(LB[k].row); });
    var hraBars = h('div', { class: 'apx-bars off' });
    var HB = {
      P: barRow('HRA paired init · \\(\\dim S_1=kn-\\tfrac{k(k+1)}2\\)', 'u<sub>2i−1</sub> = u<sub>2i</sub> (HF default) · k = r/2 · apex'),
      L: barRow('Zero-init LoRA, same r · \\(mr\\)', 'for comparison, against d(LoRA) = r(m+n−r)')
    };
    hraBars.appendChild(HB.P.row); hraBars.appendChild(HB.L.row);
    count.appendChild(loraBars); count.appendChild(hraBars);
    var cLegend = h('div', { class: 'apx-legend', style: 'margin-top:.75rem' });
    cLegend.innerHTML = hs([
      '<span class="apx-li"><svg width="18" height="10"><rect x="0" y="1" width="18" height="8" rx="2" style="fill:var(--moss)"/></svg>dim S₁ (first-order image)</span>',
      '<span class="apx-li"><svg width="18" height="10"><rect x="0.5" y="1.5" width="17" height="7" rx="2" style="fill:var(--seal-soft);stroke:var(--seal)"/></svg>deficit d − dim S₁</span>',
      '<span class="apx-li"><svg width="18" height="10"><rect x="0.5" y="1.5" width="17" height="7" rx="2" style="fill:var(--paper-2);stroke:var(--ink-3);stroke-width:1"/></svg>bar length = d (expressive dimension)</span>'
    ].map(liText).join(''));
    count.appendChild(cLegend);

    var foot = h('div', { class: 'apx-foot' });
    stage.appendChild(foot);

    /* ---------- dial ---------- */
    function drawDial() {
      var cx = 48, cy = 48, Rr = 34, p = st.phi * DEG, c = Math.cos(p), s = Math.sin(p);
      var x1 = cx + Rr * c, y1 = cy - Rr * s, x2 = cx - Rr * c, y2 = cy + Rr * s;
      var ar = 14, large = 0;
      var ax = cx + ar * c, ay = cy - ar * s;
      var lx = cx + (Rr + 0) * Math.cos(p + 0.42), ly = cy - (Rr + 0) * Math.sin(p + 0.42);
      dial.innerHTML =
        '<circle class="ring" cx="' + cx + '" cy="' + cy + '" r="' + (Rr + 8) + '"/>' +
        '<line class="ax" x1="' + (cx - Rr - 6) + '" y1="' + cy + '" x2="' + (cx + Rr + 6) + '" y2="' + cy + '"/>' +
        '<line class="ax" x1="' + cx + '" y1="' + (cy - Rr - 6) + '" x2="' + cx + '" y2="' + (cy + Rr + 6) + '"/>' +
        '<path class="arc" d="M' + (cx + ar) + ',' + cy + ' A' + ar + ',' + ar + ' 0 ' + large + ' 0 ' + ax.toFixed(2) + ',' + ay.toFixed(2) + '"/>' +
        '<line class="vline" x1="' + x2.toFixed(2) + '" y1="' + y2.toFixed(2) + '" x2="' + x1.toFixed(2) + '" y2="' + y1.toFixed(2) + '"/>' +
        '<circle class="vneg" cx="' + x2.toFixed(2) + '" cy="' + y2.toFixed(2) + '" r="3.6"/>' +
        '<circle class="vdot" cx="' + x1.toFixed(2) + '" cy="' + y1.toFixed(2) + '" r="5.2"/>' +
        '<text x="' + (lx - 4).toFixed(1) + '" y="' + (ly + 4).toFixed(1) + '">' + svgSubs('v₀') + '</text>' +
        '<text class="t" x="' + (cx + 16) + '" y="' + (cy + 12) + '">' + svgSubs('φ₀') + '</text>';
    }
    function dialFromEvent(e) {
      var b = dial.getBoundingClientRect();
      var x = (e.clientX - b.left) / b.width * 96 - 48, y = 48 - (e.clientY - b.top) / b.height * 96;
      if (Math.hypot(x, y) < 3) return;
      var a = Math.atan2(y, x) / DEG;
      a = ((a % 180) + 180) % 180;
      var v = Math.round(a) % 180;
      if (v !== st.phi) { st.phi = v; phiS.input.value = v; phiS.refresh(); changed(); }
    }
    var dialDrag = false;
    dial.addEventListener('pointerdown', function (e) { dialDrag = true; try { dial.setPointerCapture(e.pointerId); } catch (er) {} dialFromEvent(e); e.preventDefault(); });
    dial.addEventListener('pointermove', function (e) { if (dialDrag) dialFromEvent(e); });
    dial.addEventListener('pointerup', function () { dialDrag = false; });
    dial.addEventListener('pointercancel', function () { dialDrag = false; });

    /* ---------- camera ---------- */
    var view = { w: 560, h: 470, K: 230, cx: 280, cy: 235 };
    function cam() {
      var a = st.az * DEG, e = st.el * DEG;
      return { ca: Math.cos(a), sa: Math.sin(a), ce: Math.cos(e), se: Math.sin(e) };
    }
    function proj(p, C) {
      var u = p[1] * C.ca + p[2] * C.sa, w = -p[1] * C.sa + p[2] * C.ca;
      return [view.cx + view.K * u, view.cy - view.K * (p[0] * C.ce - w * C.se), p[0] * C.se + w * C.ce];
    }
    function screenBasis(C) {
      return {
        right: [0, C.ca, C.sa],
        up: [C.ce, C.se * C.sa, -C.se * C.ca],
        depth: [C.se, -C.ce * C.sa, C.ce * C.ca]
      };
    }

    /* ---------- scene drawing ---------- */
    function P2(q) { return q[0].toFixed(1) + ',' + q[1].toFixed(1); }
    function pathOf(pts) { return 'M' + pts.map(P2).join('L'); }
    function arrowHead(tip, from, size) {
      var dx = tip[0] - from[0], dy = tip[1] - from[1], L = Math.hypot(dx, dy);
      if (L < 1e-6) return '';
      dx /= L; dy /= L;
      var s = size || 9, c = Math.cos(0.42), sn = Math.sin(0.42);
      var l = [tip[0] - s * (dx * c - dy * sn), tip[1] - s * (dy * c + dx * sn)];
      var r = [tip[0] - s * (dx * c + dy * sn), tip[1] - s * (dy * c - dx * sn)];
      return 'M' + P2(l) + 'L' + P2(tip) + 'L' + P2(r);
    }
    function txt(cls, q, dx, dy, s, anchor) {
      var at = ' x="' + (q[0] + dx).toFixed(1) + '" y="' + (q[1] + dy).toFixed(1) + '"' + (anchor ? ' text-anchor="' + anchor + '"' : '') + '>' + svgSubs(s) + '</text>';
      /* a haloed label is drawn as a paper-coloured halo copy under a plain copy: one text element would paint each
         tspan's halo over the previous tspan, clipping a subscript against the glyph that follows it */
      if (/\bhalo\b/.test(cls)) return '<text class="' + cls + '" style="fill:var(--paper)" aria-hidden="true"' + at + '<text class="' + cls.replace(/\s*\bhalo\b/, '') + '"' + at;
      return '<text class="' + cls + '"' + at;
    }
    function rel(S) { return xyz(sub(S, vShown)); }
    /* Unicode subscript digits -> lowered tspans (the body serif draws U+2080 like an "o") */
    function svgSubs(t) {
      var out = '', parts = t.split(/([₀-₉]+)/);
      parts.forEach(function (p, i) {
        if (!p) return;
        if (i % 2) out += '<tspan dy="3.5" style="font-size:78%;font-style:normal">' + p.replace(/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 8320); }) + '</tspan>';
        else out += (i > 0 ? '<tspan dy="-3.5">' + A.esc(p) + '</tspan>' : A.esc(p));
      });
      return out;
    }

    /* rendered width of a scene label in its own class (fonts differ in width, so measure rather than estimate);
       cached per class and text, cleared when the web fonts arrive */
    var twCache = {};
    function textW(text, cls, wPer) {
      var key = cls + '|' + text;
      if (twCache[key] != null) return twCache[key];
      var est = text.replace(/[₀-₉]/g, '').length * wPer, w = est;
      try {
        var probe = document.createElementNS(NS, 'text');
        probe.setAttribute('class', cls);
        probe.setAttribute('visibility', 'hidden');
        probe.innerHTML = svgSubs(text);
        svg.appendChild(probe);
        var len = probe.getComputedTextLength();
        svg.removeChild(probe);
        if (len > 0) { w = len; twCache[key] = w; }
      } catch (e) {}
      return w;
    }

    function drawScene() {
      if (!R) return;
      var w = Math.max(260, svg.getBoundingClientRect().width || scene.clientWidth || 560);   // exact width: the scene draws 1:1
      var hh = Math.round(Math.max(300, Math.min(540, w * 0.9)));
      var C = cam();
      var vext = 2 * H * (C.ce + C.se);
      view.w = w; view.h = hh;
      view.cx = w / 2; view.cy = hh / 2 + 4;
      /* zoom out (rotation-invariantly) when T, theta0, the start or the arrow tip lie far from the vertex, e.g. after a
         large split moved the vertex away from a T placed in another mode; within the drag radius TMAX this never binds */
      var sR = rel(R.start), pR = xyz(R.proj);
      var far = Math.max(norm3(sR), norm3(rel(st.T)), norm3(rel(W0)), norm3([sR[0] + pR[0], sR[1] + pR[1], sR[2] + pR[2]]));
      var room = Math.min(w / 2, view.cy, hh - view.cy) - 24;
      view.K = Math.min(0.31 * w / H, 0.8 * hh / vext, far > 1e-9 ? room / far : Infinity);
      svg.setAttribute('viewBox', '0 0 ' + w + ' ' + hh);
      svg.setAttribute('width', w); svg.setAttribute('height', hh);

      var back = [], front = [], fills = '', out = '';
      var O = proj([0, 0, 0], C);
      /* rulings: t (1, cos psi, sin psi), t in [-H, H]  (t = c/2) */
      for (var k = 0; k < 24; k++) {
        var psi = k * 15 * DEG, cp = Math.cos(psi), sp = Math.sin(psi);
        [1, -1].forEach(function (sg) {
          var end = proj([sg * H, sg * H * cp, sg * H * sp], C), mid = proj([sg * H / 2, sg * H * cp / 2, sg * H * sp / 2], C);
          var item = '<path class="rul" d="M' + P2(O) + 'L' + P2(end) + '" style="stroke-opacity:' + (mid[2] >= O[2] ? 0.62 : 0.22) + '"/>';
          (mid[2] >= O[2] ? front : back).push(item);
        });
      }
      /* latitude circles at t = +-H/3, +-2H/3, +-H, split into front and back arcs */
      [1, 2, 3].forEach(function (j) {
        [1, -1].forEach(function (sg) {
          var t = sg * H * j / 3, rad = Math.abs(t), runs = [], cur = null, curFront = null;
          for (var q = 0; q <= 96; q++) {
            var ps = q / 96 * 2 * Math.PI;
            var pt = [t, rad * Math.cos(ps), rad * Math.sin(ps)], pr = proj(pt, C);
            var wv = -pt[1] * C.sa + pt[2] * C.ca, isF = wv >= 0;
            if (cur === null || isF !== curFront) { if (cur) { cur.push(pr); runs.push({ f: curFront, pts: cur }); } cur = [pr]; curFront = isF; }
            else cur.push(pr);
          }
          runs.push({ f: curFront, pts: cur });
          var op = j === 3 ? 1 : 0.75;
          runs.forEach(function (rn) {
            var item = '<path class="lat" d="' + pathOf(rn.pts) + '" style="stroke-opacity:' + ((rn.f ? 0.62 : 0.22) * op).toFixed(2) + (j === 3 ? '' : ';stroke-dasharray:' + (rn.f ? '0' : '2 3')) + '"/>';
            (rn.f ? front : back).push(item);
          });
          if (j === 3) {
            var rim = [];
            for (q = 0; q < 72; q++) { var ps2 = q / 72 * 2 * Math.PI; rim.push(proj([t, rad * Math.cos(ps2), rad * Math.sin(ps2)], C)); }
            rim.push(O);
            var hull = d3.polygonHull(rim.map(function (p) { return [p[0], p[1]]; }));
            if (hull) fills += '<path class="nap" d="M' + hull.map(P2).join('L') + 'Z"/>';
          }
        });
      });
      /* axis */
      var ax1 = proj([H * 1.12, 0, 0], C), ax2 = proj([-H * 1.12, 0, 0], C);
      out += '<path class="axis" d="M' + P2(ax2) + 'L' + P2(ax1) + '"/>';
      out += back.join('') + fills;

      /* S_1 at the starting point */
      var start = rel(R.start), S0 = proj(start, C);
      var s1 = '';
      var psi0 = 2 * R.phi;
      var t1 = [1 / Math.SQRT2, Math.cos(psi0) / Math.SQRT2, Math.sin(psi0) / Math.SQRT2];   // unit ruling direction (xyz)
      var t2 = [0, -Math.sin(psi0), Math.cos(psi0)];                                         // unit circumferential direction
      function at(a, b) { return [start[0] + a * t1[0] + b * t2[0], start[1] + a * t1[1] + b * t2[1], start[2] + a * t1[2] + b * t2[2]]; }
      var s1LabelPos, s1Seg = null, s1Poly = [];
      if (R.dimS1 === 1) {
        var L1 = H * Math.SQRT2 * 1.16;
        var e1 = proj(at(L1, 0), C), e2 = proj(at(-L1, 0), C);
        s1 += '<path class="s1line" d="M' + P2(e2) + 'L' + P2(e1) + '"/>';
        s1LabelPos = e1; s1Seg = [e2, e1];
      } else {
        var dv = norm3(start);                    // distance from the vertex to the start along the ruling
        var a0 = -(dv + 0.08), a1 = 0.36, b = 0.36;
        var cs = [at(a0, -b), at(a1, -b), at(a1, b), at(a0, b)].map(function (p) { return proj(p, C); });
        s1 += '<path class="s1plane" d="M' + cs.map(P2).join('L') + 'Z"/>';
        s1Poly = cs;
        var rr1 = proj(at(-dv, 0), C), rr2 = proj(at(a1, 0), C);
        s1 += '<path class="s1ruling" d="M' + P2(rr1) + 'L' + P2(rr2) + '"/>';
        s1LabelPos = cs[1][1] < cs[2][1] ? cs[1] : cs[2];
      }
      out += '<g style="opacity:' + s1Alpha.toFixed(3) + '">' + s1 + '</g>';
      out += front.join('');

      /* first-order move: Pi_{S1}(T - start), drawn from the start; residual and right-angle mark */
      var T3 = rel(st.T), Tp = proj(T3, C);
      var pr3 = xyz(R.proj), tip3 = [start[0] + pr3[0], start[1] + pr3[1], start[2] + pr3[2]], tipP = proj(tip3, C);
      var res3 = [T3[0] - tip3[0], T3[1] - tip3[1], T3[2] - tip3[2]], nres = norm3(res3), npr = norm3(pr3);
      var obst = [];                                   // screen-space segments that labels should avoid
      if (R.dimS1 === 1) obst.push(s1Seg);
      else s1Poly.forEach(function (q, i) { obst.push([q, s1Poly[(i + 1) % s1Poly.length]]); });
      if (nres > 1e-3) {
        out += '<path class="resid" d="M' + P2(tipP) + 'L' + P2(Tp) + '"/>';
        obst.push([tipP, Tp]);
        if (npr > 0.04) {
          var m = Math.min(0.045, npr / 3, nres / 3);
          var ua = [-pr3[0] / npr * m, -pr3[1] / npr * m, -pr3[2] / npr * m], ub = [res3[0] / nres * m, res3[1] / nres * m, res3[2] / nres * m];
          var q1 = proj([tip3[0] + ua[0], tip3[1] + ua[1], tip3[2] + ua[2]], C);
          var q2 = proj([tip3[0] + ua[0] + ub[0], tip3[1] + ua[1] + ub[1], tip3[2] + ua[2] + ub[2]], C);
          var q3 = proj([tip3[0] + ub[0], tip3[1] + ub[1], tip3[2] + ub[2]], C);
          out += '<path class="rmark" d="M' + P2(q1) + 'L' + P2(q2) + 'L' + P2(q3) + '"/>';
        }
      }
      if (Math.hypot(tipP[0] - S0[0], tipP[1] - S0[1]) > 3) {
        out += '<path class="arrow" d="M' + P2(S0) + 'L' + P2(tipP) + '"/>';
        out += '<path class="arrow" d="' + arrowHead(tipP, S0, 10) + '"/>';
        obst.push([S0, tipP]);
      }

      /* defect e (QLoRA): from theta0 to kappa(theta0) */
      var W3 = rel(W0), Wp = proj(W3, C);
      if (R.mode === 'qlora' && R.eFro > 1e-9) {
        out += '<path class="defect" d="M' + P2(Wp) + 'L' + P2(S0) + '" style="stroke-dasharray:4 3"/>';
        out += '<path class="defect" d="' + arrowHead(S0, Wp, 8) + '"/>';
        obst.push([Wp, S0]);
      }

      /* vertex, base point */
      var Vp = O;
      out += '<rect class="vtx" x="' + (Vp[0] - 3.6).toFixed(1) + '" y="' + (Vp[1] - 3.6).toFixed(1) + '" width="7.2" height="7.2" transform="rotate(45 ' + P2(Vp) + ')"/>';
      out += '<circle class="w0" cx="' + Wp[0].toFixed(1) + '" cy="' + Wp[1].toFixed(1) + '" r="6"/>';
      out += '<circle class="w0dot" cx="' + Wp[0].toFixed(1) + '" cy="' + Wp[1].toFixed(1) + '" r="2.2"/>';
      obst.push([Vp, Vp], [Wp, Wp], [Tp, Tp]);

      /* labels: greedy placement among candidate offsets, away from the drawn segments and earlier labels */
      var boxes = [];
      function segDist(px, py, A2, B2) {
        var vx = B2[0] - A2[0], vy = B2[1] - A2[1], L2 = vx * vx + vy * vy, t = L2 > 0 ? ((px - A2[0]) * vx + (py - A2[1]) * vy) / L2 : 0;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(px - A2[0] - t * vx, py - A2[1] - t * vy);
      }
      function place(anchorPt, text, cls, wPer, cands) {
        var w = textW(text, cls, wPer) + 4, best = null;
        cands.forEach(function (c, ci) {
          var x0 = anchorPt[0] + c[0], y0 = anchorPt[1] + c[1];
          var left = c[2] === 'end' ? x0 - w : c[2] === 'middle' ? x0 - w / 2 : x0;
          var bx = left + w / 2, by = y0 - 5;
          var sc = Infinity;
          for (var i = 0; i < 9; i++) {           // sample points across the label box
            var px = left + (i % 3) * w / 2, py = y0 - 11 + Math.floor(i / 3) * 6;
            obst.forEach(function (sg) { sc = Math.min(sc, segDist(px, py, sg[0], sg[1])); });
          }
          if (s1Poly.length) for (var j = 0; j < 9; j++) {
            /* inside the translucent S1 plane a haloed label stays legible: a mild penalty, milder than touching a line */
            if (d3.polygonContains(s1Poly.map(function (q) { return [q[0], q[1]]; }), [left + (j % 3) * w / 2, y0 - 11 + Math.floor(j / 3) * 6])) { sc = Math.min(sc, 5); break; }
          }
          boxes.forEach(function (bb) { if (Math.abs(bx - bb.x) < (w + bb.w) / 2 + 4 && Math.abs(by - bb.y) < 15) sc = Math.min(sc, -50); });
          if (left < 4 || left + w > view.w - 4 || y0 - 12 < 4 || y0 > view.h - 4) sc -= 100;
          sc -= ci * 0.8;                          // mild preference for the first candidates
          if (!best || sc > best.sc) best = { sc: sc, x: x0, y: y0, a: c[2], bx: bx, by: by, w: w };
        });
        boxes.push({ x: best.bx, y: best.by, w: w });
        return txt(cls, [best.x, best.y], 0, 0, text, best.a);
      }
      var C8 = [[-12, 22, 'end'], [12, 22, 'start'], [-12, -10, 'end'], [12, -10, 'start'], [0, 26, 'middle'], [0, -14, 'middle']];
      var w0Text = R.mode === 'zero' ? 'θ₀ = vertex: apex' : 'θ₀';
      out += place(Wp, w0Text, 'lab halo', 7.6, R.mode === 'zero' ? C8 : [[-11, 5, 'end'], [11, 5, 'start'], [0, -12, 'middle'], [0, 22, 'middle'], [-9, -9, 'end'], [9, -9, 'start'], [-9, 19, 'end'], [9, 19, 'start']]);
      /* the vertex labels try the waist first: level with the vertex, outside both nappes */
      if (R.mode === 'split') out += place(Vp, 'vertex θ₀ − P', 'sm halo', 6.6, [[-14, 4, 'end'], [14, 4, 'start'], [-22, 4, 'end'], [22, 4, 'start'], [9, 15, 'start'], [-9, 15, 'end'], [9, -8, 'start'], [-9, -8, 'end']]);
      else if (R.mode === 'qlora') out += place(Vp, 'vertex = κθ₀ = ρ(q₀)', 'sm halo', 6.6, [[-14, 4, 'end'], [14, 4, 'start'], [-22, 4, 'end'], [22, 4, 'start'], [10, 16, 'start'], [-10, 16, 'end'], [10, -9, 'start'], [-10, -9, 'end']]);
      if (R.mode === 'qlora' && R.eFro > 1e-9) {
        var mx = [(Wp[0] + S0[0]) / 2, (Wp[1] + S0[1]) / 2];
        out += place(mx, 'e', 'lab seal halo', 8, [[-9, 4, 'end'], [9, 4, 'start'], [0, -7, 'middle'], [0, 16, 'middle']]);
      }
      if (s1LabelPos) out += place(s1LabelPos, 'S₁', 'lab ochre halo', 8, [[6, -6, 'start'], [-6, -6, 'end'], [8, 12, 'start'], [-8, 12, 'end']]);
      /* cone labels on the side away from the S_1 label */
      var sideL = s1LabelPos && s1LabelPos[0] > view.cx;
      var rimU = proj([H, (sideL ? -1 : 1) * H * C.ca, (sideL ? -1 : 1) * H * C.sa], C), rimLo = proj([-H, (sideL ? -1 : 1) * H * C.ca, (sideL ? -1 : 1) * H * C.sa], C);
      /* keep the side labels inside the viewBox: if they do not fit beside the rim, lift them above the top rim
         (resp. drop them below the bottom rim), where the cone outline has already curved away */
      function sideLabel(pt, text, upper) {
        var w = textW(text, 'sm halo', 6.6), x = pt[0] + (sideL ? -8 : 8), y = pt[1] + 4, anchor = sideL ? 'end' : 'start';
        if (sideL && x - w < 4) { x = 4; anchor = 'start'; y = pt[1] + (upper ? -14 : 18); }
        else if (!sideL && x + w > view.w - 4) { x = view.w - 4; anchor = 'end'; y = pt[1] + (upper ? -14 : 18); }
        return txt('sm halo', [x, y], 0, 0, text, anchor);
      }
      out += sideLabel(rimU, 'c > 0 · PSD', true);
      out += sideLabel(rimLo, 'c < 0 · NSD', false);
      gScene.innerHTML = out;

      /* T handle (persistent node) and its label */
      var cls = R.cur.inIm ? 'm' : 's';
      tDot.setAttribute('class', 'tdot ' + cls);
      tHalo.setAttribute('class', 'thalo ' + cls);
      [tDot, tHalo, tHit].forEach(function (c) { c.setAttribute('cx', Tp[0].toFixed(1)); c.setAttribute('cy', Tp[1].toFixed(1)); });
      obst.pop();                                       // T's own point is not an obstacle for its label
      var tl = place(Tp, 'T', 'lab x', 9, [[11, -9, 'start'], [-11, -9, 'end'], [12, 16, 'start'], [-12, 16, 'end']]);
      var mX = /x="([-\d.]+)" y="([-\d.]+)" text-anchor="(\w+)"/.exec(tl);
      tLab.setAttribute('x', mX[1]); tLab.setAttribute('y', mX[2]); tLab.setAttribute('text-anchor', mX[3]);
      tLab.setAttribute('class', 'lab halo ' + (R.cur.inIm ? 'moss' : 'seal'));
    }

    /* ---------- legend ---------- */
    function drawLegend() {
      var items = [
        '<span class="apx-li"><svg width="20" height="12"><path d="M1,11 L10,1 L19,11" style="fill:var(--tide-soft);stroke:var(--tide);stroke-width:1"/></svg><span>Im = vertex + {rank ≤ 1}, a double cone <span style="white-space:nowrap">(d = 2)</span></span></span>',
        R.dimS1 === 1
          ? '<span class="apx-li"><svg width="20" height="12"><path d="M1,6 L19,6" style="stroke:var(--ochre);stroke-width:3.2;stroke-linecap:round"/></svg>S₁: one ruling (line)</span>'
          : '<span class="apx-li"><svg width="20" height="12"><path d="M3,2 L19,2 L17,10 L1,10 Z" style="fill:var(--ochre-soft);stroke:var(--ochre);stroke-width:1.25"/></svg>S₁: tangent plane</span>',
        '<span class="apx-li"><svg width="22" height="12"><path d="M1,6 L19,6 M14,2 L19,6 L14,10" style="fill:none;stroke:var(--ink);stroke-width:2"/></svg>T − ρ(q₀) projected onto S₁: the closest first-order move</span>',
        '<span class="apx-li"><svg width="12" height="12"><circle cx="6" cy="6" r="5" style="fill:var(--moss)"/></svg>' + mt('T ∈ Im') + '</span>',
        '<span class="apx-li"><svg width="12" height="12"><circle cx="6" cy="6" r="5" style="fill:var(--seal)"/></svg>' + mt('T ∉ Im') + '</span>'
      ];
      if (R.mode === 'qlora') items.push('<span class="apx-li"><svg width="22" height="12"><path d="M1,6 L19,6" style="stroke:var(--seal);stroke-width:2;stroke-dasharray:4 3"/></svg>defect e</span>');
      legend.innerHTML = hs(items.map(liText).join(''));
    }

    /* ---------- panel ---------- */
    function matHTML(S, cls) {
      return '<span class="apx-mat' + (cls ? ' ' + cls : '') + '"><span>' + fm(S[0]) + '</span><span>' + fm(S[1]) + '</span><span>' + fm(S[1]) + '</span><span>' + fm(S[2]) + '</span></span>';
    }
    function matBlock(label, S, cls) { return '<div class="apx-matb"><span class="ml">' + hs(label) + '</span>' + matHTML(S, cls) + '</div>'; }
    function setCell(tbl, k, html, cls) { var c = tbl.querySelector('[data-k="' + k + '"]'); c.innerHTML = html; c.className = cls || ''; }
    function updPanel() {
      var apex = R.deficit > 0;
      stChip.className = 'apx-chip ' + (apex ? 's' : 'm');
      stChip.textContent = apex ? 'apex' : 'smooth point';
      bigN.className = 'n ' + (apex ? 's' : 'm');
      bigN.textContent = R.dimS1 + ' / ' + R.d;
      bigL.innerHTML = hs('first-order directions<br>dim S₁ / d');
      var mh = matBlock('θ₀ = W₀', W0);
      if (R.mode === 'split') mh += matBlock('P = c₀v₀v₀ᵀ', R.P) + matBlock('vertex θ₀ − P', R.vertex);
      else if (R.mode === 'qlora') mh += matBlock('κθ₀ = ρ(q₀)', R.start) + matBlock('e = κθ₀ − θ₀', R.e, 'q');
      else mh += '<div class="apx-matb"><span class="ml">' + hs('vertex = ρ(q₀) = θ₀') + '</span></div>';
      mats.innerHTML = mh;
      setCell(tbl1, 'sv', f3(R.sv[0]) + ' · ' + f3(R.sv[1]));
      setCell(tbl1, 's1', String(R.dimS1));
      setCell(tbl1, 'd', String(R.d));
      setCell(tbl1, 'def', R.deficit + (apex ? ' · apex' : ' · regular'), apex ? 's' : 'm');

      var inIm = R.cur.inIm;
      tChip.className = 'apx-chip ' + (inIm ? 'm' : 's');
      tChip.innerHTML = mt(inIm ? 'T ∈ Im' : 'T ∉ Im');
      setCell(tbl2, 'nr', f3(R.nr));
      setCell(tbl2, 'dist', f3(R.cur.dist), inIm ? 'm' : 's');
      rfill.style.width = (100 * Math.min(1, R.frac)).toFixed(2) + '%';
      rnum.textContent = pct(R.frac);
      MODES.forEach(function (M) {
        var mb = R.memb[M.value], c = membChips[M.value];
        c.className = 'apx-chip ' + (mb.inIm ? 'm' : 's') + (M.value === R.mode ? ' cur' : '');
        c.textContent = { zero: 'zero-init', split: 'split', qlora: 'QLoRA' }[M.value] + (mb.inIm ? ' ✓' : ' ✗');
      });
      qBox.className = 'apx-q' + (R.mode === 'qlora' ? ' on' : '');
      if (R.mode === 'qlora') {
        setCell(tbl3, 'ef', f3(R.eFro));
        setCell(tbl3, 'ed', signed(f4(R.eDet)), Math.abs(R.eDet) > 1e-12 ? 's' : 'm');
        setCell(tbl3, 'ew', f3(R.w0Dist), R.w0Dist > 1e-9 ? 's' : 'm');
      }
      /* the sentence: built from the computed state */
      var reachS = '<b>' + pct(R.frac) + '</b>', s = '', full = R.frac > 0.9995;
      if (R.mode === 'zero') {
        s = '<span class="lead s">apex · Thm II.2(c)</span>At c = 0 the v-direction of dρ vanishes, so S₁ is the single ruling along v₀ although the image is the whole cone. ' +
          (full ? 'T lies on that ruling, so here the first-order move reaches it.' :
            inIm ? 'T is in the image, yet the closest first-order move covers only ' + reachS + ' of T − θ₀.' : 'T is not in the image; the closest first-order move covers ' + reachS + ' of T − θ₀.');
      } else if (R.mode === 'split') {
        s = '<span class="lead m">smooth point · Thm II.2(c)</span>Splitting moves the vertex to θ₀ − P. θ₀ now sits on the smooth sheet, S₁ is the whole tangent plane (dim S₁ = d = 2), and the closest first-order move covers ' + reachS + ' of T − θ₀. ' +
          (inIm ? 'T also lies in this image.' : (R.memb.zero.inIm ? 'The price: T is no longer in the image. Split and zero-init images are incomparable (Thm II.2(d)).' : 'T is not in this image either.'));
      } else {
        s = '<span class="lead s">defect · Prop V.2(a)</span>QLoRA is zero-init LoRA pointed at κθ₀ instead of θ₀: training starts at the vertex κθ₀, an apex of that object. ' +
          (R.eRank === 2 ? 'θ₀ itself is off the image, since det e ≠ 0 means rank e = 2 > r = 1.' : 'Here rank e ≤ 1, so θ₀ happens to lie in the image.');
      }
      why.innerHTML = hs(s);
    }

    /* ---------- counters ---------- */
    function setBar(B, s1, dd, dmax, na) {
      B.row.className = 'apx-brow' + (na ? ' na' : '');
      B.bd.style.display = na ? 'none' : '';
      if (na) { B.bd.style.width = '0'; B.bs.style.width = '0'; B.num.innerHTML = '—<span class="df">' + na + '</span>'; return; }
      B.bd.style.width = (100 * dd / dmax).toFixed(3) + '%';
      B.bs.style.width = (100 * s1 / dmax).toFixed(3) + '%';
      var def = dd - s1;
      B.num.innerHTML = intStr(s1) + ' / ' + intStr(dd) + '<span class="df' + (def === 0 ? ' z' : '') + '">' + (def === 0 ? 'deficit 0' : 'deficit ' + intStr(def)) + '</span>';
    }
    function updCounts() {
      var m = DIMS[st.mi], n = DIMS[st.ni], r = RANKS[st.ri];
      presetBtns.forEach(function (b, i) { b.setAttribute('aria-pressed', String(PRESETS[i].m === m && PRESETS[i].n === n)); });
      mS.input.setAttribute('aria-valuetext', String(m)); nS.input.setAttribute('aria-valuetext', String(n)); rS.input.setAttribute('aria-valuetext', String(r));
      loraBars.className = 'apx-bars' + (st.hra ? ' off' : '');
      hraBars.className = 'apx-bars' + (st.hra ? '' : ' off');
      var size = '<span class="nb">' + intStr(m) + ' × ' + intStr(n) + '</span>, <span class="nb">r = ' + r + '</span>';
      if (!st.hra) {
        var Lc = loraCounts(m, n, r);
        if (!Lc.ok) {
          cwarn.textContent = 'Thm II.2 assumes 1 ≤ r < min(m, n) = ' + Math.min(m, n) + '. Lower r or raise m, n.';
          ['TA', 'TB', 'S', 'HFo'].forEach(function (k) { setBar(LB[k], 0, 0, 1, 'needs r < min(m, n)'); });
          sentence.innerHTML = 'At ' + size + ', the strata of Thm II.2 are not defined.';
          return;
        }
        cwarn.textContent = '';
        Lc.rows.forEach(function (row) { setBar(LB[row.key], row.s1, row.d, Lc.d, row.needEven && r % 2 ? 'needs even r' : null); });
        var ta = Lc.rows[0];
        sentence.innerHTML = hs('At ' + size + ', zero-init LoRA has <span class="nb">dim S₁ = ' + intStr(ta.s1) + '</span> of the <span class="nb">d = ' + intStr(Lc.d) +
          '</span> dimensions of its image: it is short of <span class="nb s">' + intStr(Lc.d - ta.s1) + '</span> first-order directions (' + pct((Lc.d - ta.s1) / Lc.d) +
          '). A split init reaches all <span class="nb m">' + intStr(Lc.d) + '</span>.');
      } else {
        var Hc = hraCounts(m, n, r), why2 = [];
        if (r % 2) why2.push('r even');
        if (r < 2 || r > n - 2) why2.push('2 ≤ r ≤ n − 2');
        if (m < n) why2.push('an injective W₀, so m ≥ n (not down_proj)');
        var Lc2 = loraCounts(m, n, r), dmax = Math.max(Hc.d, Lc2.d);
        if (!Hc.ok) {
          cwarn.innerHTML = hs(A.esc('Thm II.7 needs ' + why2.join(', ') + '.'));
          setBar(HB.P, 0, 0, 1, 'outside Thm II.7');
          if (Lc2.ok) setBar(HB.L, Lc2.rows[0].s1, Lc2.d, Lc2.d, null); else setBar(HB.L, 0, 0, 1, 'needs r < min(m, n)');
          sentence.innerHTML = 'At ' + size + ', HRA falls outside the hypotheses of Thm II.7.';
          return;
        }
        cwarn.textContent = '';
        setBar(HB.P, Hc.s1, Hc.d, dmax, null);
        if (Lc2.ok) setBar(HB.L, Lc2.rows[0].s1, Lc2.d, dmax, null); else setBar(HB.L, 0, 0, 1, 'needs r < min(m, n)');
        sentence.innerHTML = hs('At ' + size + ' (k = ' + Hc.k + '), HRA’s paired init has <span class="nb">dim S₁ = ' + intStr(Hc.s1) + '</span> of <span class="nb">d = ' + intStr(Hc.d) +
          '</span>: short of <span class="nb s">k(2n − 3k − 1)/2 = ' + intStr(Hc.deficit) + '</span> directions (' + pct(Hc.deficit / Hc.d) + '). Its image is the meet (W₀ + 𝓜<sub>≤r</sub>) ∩ W₀·SO(n), an image-level meet, not a categorical product (Thm II.7(a), (d)).');
      }
    }

    /* ---------- footer (with the in-browser Jacobian check) ---------- */
    function drawFoot() {
      var ck;
      try { ck = runChecks(A); } catch (err) { ck = null; }
      var line = '';
      if (ck) {
        var L = ck.lora, ok = function (a, b) { return '<span class="' + (a === b ? 'ok' : 'bad') + '">' + a + (a === b ? ' ✓' : ' ≠ ' + b) + '</span>'; };
        line = '<b>Jacobian-rank check, run in this browser.</b> LoRA at (m, n, r) = (' + L.m + ', ' + L.n + ', ' + L.r + '): T<sub>A</sub> ' + ok(ck.TA, L.pred.TA) + ', T<sub>B</sub> ' + ok(ck.TB, L.pred.TB) +
          ', S ' + ok(ck.S, L.pred.S) + ', HF orthogonal ' + ok(ck.HFo, L.pred.HFo) + ' (‖B₀A₀‖ = ' + ck.HFoProd.toExponential(0) + '). HRA at n = 9: ' +
          ck.hra.map(function (x) { return 'r = ' + x.r + ' paired ' + ok(x.paired, x.predP) + ' vs generic ' + ok(x.generic, x.predG); }).join('; ') + '. ';
      }
      foot.innerHTML = hs(
        (line ? '<p>' + line + '</p>' : '') +
        '<p><b>The 3D picture is the symmetric 2×2 slice, not ℝ<sup>m×n</sup>.</b> It shows the rank-one model ρ(c, φ) = W<sub>res</sub> + c v<sub>φ</sub>v<sub>φ</sub><sup>⊤</sup> (c plays B, v<sub>φ</sub> plays A) with θ<sub>0</sub> = W<sub>0</sub> = [0.62 0.36; 0.36 0.88]. ' +
        'Slice numbers come from the analytic Jacobian (σ by Jacobi SVD) and from Eckart–Young, dist<sub>F</sub>(S, rank ≤ 1) = min<sub>i</sub> |λ<sub>i</sub>(S)|. ' +
        'The arrow is the Frobenius-orthogonal projection onto S<sub>1</sub>, the closest first-order move; an optimizer’s first step also depends on its metric, so this is not a claim about gradient steps or speed (Prop II.3). ' +
        'κ rounds each entry to a uniform Δ-grid, an illustrative stand-in for NF4. A dragged T within ' + SNAP + ' (Frobenius) of Im is placed exactly on it.</p>' +
        '<p><b>The counters are exact.</b> Thm II.2(b), (e) need 1 ≤ r &lt; min(m, n). Thm II.7(b) concerns HRA with λ &lt; ∞ (HF apply_GS=False) at HF’s paired initialisation, and needs injective W<sub>0</sub> (so m ≥ n), even r and 2 ≤ r ≤ n − 2.</p>');
    }

    /* ---------- update cycle ---------- */
    function syncControls() {
      c0S.el.className = 'apx-sl' + (st.mode === 'split' ? '' : ' off');
      c0S.input.disabled = st.mode !== 'split';
      gridRow.className = 'apx-row' + (st.mode === 'qlora' ? '' : ' off');
      gridSeg.el.querySelectorAll('button').forEach(function (b) { b.disabled = st.mode !== 'qlora'; });
      mdesc.querySelectorAll('[data-m]').forEach(function (d) { d.className = d.getAttribute('data-m') === st.mode ? 'on' : ''; });
    }
    function render() {
      drawScene();
      drawDial();
    }
    function schedule() { if (!raf) raf = requestAnimationFrame(function () { raf = 0; render(); }); }
    var announce = A.debounce(function () {
      live.textContent = { zero: 'Zero-init', split: 'Split', qlora: 'QLoRA' }[R.mode] + ': ' + (R.deficit > 0 ? 'apex' : 'smooth point') + ', dim S1 ' + R.dimS1 + ' of ' + R.d +
        '. T ' + (R.cur.inIm ? 'is' : 'is not') + ' in the image; first-order reach ' + pct(R.frac) + '.';
    }, 400);
    function changed(opts) {
      R = compute(st, A);
      if (!tween) vShown = R.vertex.slice();
      syncControls();
      updPanel();
      drawLegend();
      if (opts && opts.now) render(); else schedule();
      announce();
    }
    function setMode(v) {
      if (v === st.mode) return;
      var from = vShown.slice();
      st.mode = v;
      modeSeg.set(v);
      R = compute(st, A);
      if (reduced) { vShown = R.vertex.slice(); s1Alpha = 1; tween = null; changed(); return; }
      var to = R.vertex.slice(), t0 = performance.now(), dur = 700;
      /* each tween owns its token: a mode change mid-flight starts a new tween from the drawn vertex, and the
         superseded loop exits at its next frame instead of finishing (or nulling) the new one */
      var my = { from: from, to: to };
      tween = my;
      s1Alpha = 0;
      var ease = (window.d3 && d3.easeCubicInOut) || function (u) { return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; };
      (function step() {
        if (tween !== my) return;
        var u = Math.min(1, (performance.now() - t0) / dur), e = ease(u);
        vShown = lerpS(my.from, my.to, e);
        s1Alpha = Math.max(0, Math.min(1, (u - 0.35) / 0.65));
        drawScene();
        if (u < 1) requestAnimationFrame(step);
        else { tween = null; vShown = R.vertex.slice(); s1Alpha = 1; drawScene(); }
      })();
      changed();
    }

    /* ---------- T placement ---------- */
    function placeT(relXYZ, snap) {
      var n3 = norm3(relXYZ);
      if (n3 > TMAX) relXYZ = relXYZ.map(function (v) { return v * TMAX / n3; });
      var T = add(R.vertex, fromXYZ(relXYZ));
      if (snap) {
        var D = sub(T, R.vertex);
        if (minAbsEig(D) < SNAP) T = add(R.vertex, nearestRank1(D));
      }
      st.T = T;
      changed({ now: true });
    }
    function moveT(dRight, dUp, dDepth) {
      var B = screenBasis(cam()), p = xyz(sub(st.T, R.vertex));
      for (var i = 0; i < 3; i++) p[i] += dRight * B.right[i] + dUp * B.up[i] + dDepth * B.depth[i];
      placeT(p, false);                                 // no snapping for key steps (Enter snaps), or T could never leave Im
    }
    function snapT() { st.T = add(R.vertex, nearestRank1(sub(st.T, R.vertex))); changed({ now: true }); }
    snapBtn.addEventListener('click', snapT);
    rank1Btn.addEventListener('click', function () { st.T = add(W0, scl(vvT(T_PHI * DEG), T_C)); changed({ now: true }); });
    viewBtn.addEventListener('click', function () { st.az = DEF.az; st.el = DEF.el; schedule(); });

    /* ---------- pointer: rotate (background) and drag T ---------- */
    var drag = null;
    function svgScale() { var b = svg.getBoundingClientRect(); return b.width > 0 ? view.w / b.width : 1; }
    svg.addEventListener('pointerdown', function (e) {
      var onT = e.target === tHit || e.target === tDot;
      drag = { kind: onT ? 'T' : 'rot', x: e.clientX, y: e.clientY, az: st.az, el: st.el, T0: xyz(sub(st.T, R.vertex)), k: svgScale(), id: e.pointerId };
      if (onT) { try { svg.setPointerCapture(e.pointerId); } catch (er) {} e.preventDefault(); gT.focus({ preventScroll: true }); }
      else if (e.pointerType === 'mouse') { try { svg.setPointerCapture(e.pointerId); } catch (er) {} }
    });
    svg.addEventListener('pointermove', function (e) {
      if (!drag || e.pointerId !== drag.id) return;
      var dx = (e.clientX - drag.x) * drag.k, dy = (e.clientY - drag.y) * drag.k;
      if (drag.kind === 'rot') {
        st.az = drag.az - dx * 0.45;
        if (e.pointerType === 'mouse') st.el = Math.max(4, Math.min(48, drag.el + dy * 0.25));
        schedule();
      } else {
        var B = screenBasis(cam()), p = drag.T0.slice();
        for (var i = 0; i < 3; i++) p[i] += (dx / view.K) * B.right[i] + (-dy / view.K) * B.up[i];
        placeT(p, true);
      }
    });
    function endDrag() { drag = null; }
    svg.addEventListener('pointerup', endDrag);
    svg.addEventListener('pointercancel', endDrag);
    svg.addEventListener('keydown', function (e) {
      if (e.target !== svg) return;
      var used = true;
      if (e.key === 'ArrowLeft') st.az += 6;
      else if (e.key === 'ArrowRight') st.az -= 6;
      else if (e.key === 'ArrowUp') st.el = Math.min(48, st.el + 3);
      else if (e.key === 'ArrowDown') st.el = Math.max(4, st.el - 3);
      else used = false;
      if (used) { e.preventDefault(); schedule(); }
    });
    gT.addEventListener('keydown', function (e) {
      var s = e.shiftKey ? 0.06 : 0.02, used = true;
      if (e.key === 'ArrowLeft') moveT(-s, 0, 0);
      else if (e.key === 'ArrowRight') moveT(s, 0, 0);
      else if (e.key === 'ArrowUp') moveT(0, s, 0);
      else if (e.key === 'ArrowDown') moveT(0, -s, 0);
      else if (e.key === 'PageUp') moveT(0, 0, s);
      else if (e.key === 'PageDown') moveT(0, 0, -s);
      else if (e.key === 'Enter' || e.key === ' ') snapT();
      else used = false;
      if (used) { e.preventDefault(); e.stopPropagation(); }
    });

    /* ---------- mount ---------- */
    changed({ now: true });
    updCounts();
    drawFoot();
    A.typeset(stage);
    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(function () { drawScene(); }, 60)).observe(scene);
    else window.addEventListener('resize', A.debounce(function () { drawScene(); }, 100));
    A.onTheme(function () { render(); drawLegend(); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { twCache = {}; drawScene(); });

    /* read-only hook for headless verification pages */
    el.__apex = {
      state: st, R: function () { return R; }, setMode: setMode, snap: snapT,
      set: function (o) {
        if (o.phi != null) { st.phi = o.phi; phiS.input.value = o.phi; phiS.refresh(); }
        if (o.c0 != null) { st.c0 = o.c0; c0S.input.value = o.c0; c0S.refresh(); }
        if (o.grid != null) { st.grid = o.grid; gridSeg.set(o.grid); }
        if (o.az != null) st.az = o.az;
        if (o.el != null) st.el = o.el;
        if (o.T) st.T = o.T.slice();
        changed({ now: true });
      },
      counts: function (m, n, r, hra) {
        st.mi = DIMS.indexOf(m); st.ni = DIMS.indexOf(n); st.ri = RANKS.indexOf(r); st.hra = !!hra;
        mS.input.value = st.mi; nS.input.value = st.ni; rS.input.value = st.ri; mS.refresh(); nS.refresh(); rS.refresh(); famSeg.set(!!hra); updCounts();
      },
      view: function () { return { w: view.w, h: view.h, K: view.K }; },
      shown: function () { return { vShown: vShown.slice(), tween: !!tween, s1Alpha: s1Alpha }; }
    };
  });
})();
