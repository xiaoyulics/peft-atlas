/* Figure 13 (#fig-prefix) — Prefix Is a Gate (Thm VI.5; He et al. 2022; Petrov et al. 2024; Zhang et al. 2024).
   The kicker reads the figure number from the enclosing <figure>'s .fig-n, so renumbering the page cannot strand it.
   One query q in R^4 attends to six content tokens C and to l prefix tokens P.
   Joint softmax (prefix tuning):  Attn_q(P ⊎ C) = (1 - λ_q) Attn_q(C) + λ_q Attn_q(P),
                                   λ_q = Z_q(P) / (Z_q(P) + Z_q(C)),  Z_q(S) = Σ_s exp(<q,k_s>/√4).
   Separate softmax (LLaMA-Adapter): out = Attn_q(C) + g Attn_q(P).
   Single-token panel: a frozen head on a one-token context is x ↦ W_v x (softmax over one token = 1),
   affine for every weight setting; the prefixed head is not, which the printed second difference shows.
   Every number on screen is computed below from seeded synthetic tokens. */
(function () {
  'use strict';

  var CSS_ID = 'css-prefix';
  var CSS = [
    '[data-figure="prefix"] .pg-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem;margin:0 0 .2rem}',
    '[data-figure="prefix"] .pg-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.55rem,3.6vw,2.15rem);line-height:1.08;letter-spacing:-.006em;margin:0;color:var(--ink)}',
    '[data-figure="prefix"] .pg-title em{font-style:italic;color:var(--ochre)}',
    '[data-figure="prefix"] .pg-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    '[data-figure="prefix"] .pg-instr{font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);margin:.2rem 0 .85rem;max-width:50rem}',
    '[data-figure="prefix"] .label{letter-spacing:.06em;color:var(--ink-2)}',
    '[data-figure="prefix"] .pg-controls{display:flex;flex-wrap:wrap;gap:.7rem 1.2rem;align-items:flex-end;margin:0 0 .7rem;padding:.7rem .8rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius)}',
    '[data-figure="prefix"] .pg-controls .control{min-width:7.6rem;flex:1 1 7.6rem}',
    '[data-figure="prefix"] .control>label{color:var(--ink-2)}',
    '[data-figure="prefix"] .pg-controls .control>label{white-space:nowrap;letter-spacing:.04em}',
    '[data-figure="prefix"] .pg-modegrp{display:grid;gap:.25rem;flex:1 1 15rem}',
    '[data-figure="prefix"] .pg-modegrp .seg{width:max-content;max-width:100%}',
    '[data-figure="prefix"] .pg-modegrp .seg button{font-weight:500}',
    '[data-figure="prefix"] .pg-dialgrp{display:flex;gap:.65rem;align-items:center;flex:1.4 1 14rem;min-width:0}',
    '[data-figure="prefix"] .pg-dialgrp .control{flex:1 1 auto;min-width:6rem}',
    '[data-figure="prefix"] .pg-dial{width:62px;height:62px;flex:none;touch-action:none;cursor:grab;border-radius:50%}',
    '[data-figure="prefix"] .pg-dial:active{cursor:grabbing}',
    '[data-figure="prefix"] .pg-btns{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center}',
    '[data-figure="prefix"] .pg-btns .btn{padding:.36rem .62rem;letter-spacing:.05em}',
    '[data-figure="prefix"] .pg-btns .btn[aria-pressed="true"]{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
    '[data-figure="prefix"] .pg-btns .btn:disabled{opacity:.45;cursor:not-allowed}',
    '[data-figure="prefix"] .pg-seed{font-family:var(--f-mono);font-size:var(--fs-xs);color:var(--ink-2);font-variant-numeric:tabular-nums}',
    '[data-figure="prefix"] .pg-var{text-transform:none;letter-spacing:0;font-family:var(--f-body);font-style:italic;font-size:1.12em}',
    '[data-figure="prefix"] .pg-nt{text-transform:none;letter-spacing:0;font-family:var(--f-body);font-weight:var(--w-body);font-size:1.12em}',
    '[data-figure="prefix"] .control.is-off{opacity:.45}',
    '[data-figure="prefix"] .control.is-off input{cursor:not-allowed}',
    '[data-figure="prefix"] .pg-legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:0;color:var(--ink-2);margin:0 0 .6rem}',
    '[data-figure="prefix"] .pg-legend span{display:inline-flex;align-items:center;gap:.35rem;white-space:nowrap}',
    '[data-figure="prefix"] .pg-legend svg{width:auto;display:inline-block;overflow:visible}',
    '[data-figure="prefix"] .pg-main{display:grid;gap:.75rem;grid-template-columns:minmax(0,1fr)}',
    '@media (min-width:760px){[data-figure="prefix"] .pg-main{grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}}',
    '[data-figure="prefix"] .pg-panel{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.6rem .7rem .5rem;min-width:0}',
    '[data-figure="prefix"] .pg-ph{display:flex;justify-content:space-between;align-items:baseline;gap:.2rem .6rem;flex-wrap:wrap;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:0 0 .3rem}',
    '[data-figure="prefix"] .pg-ph b{font-weight:600;color:var(--ink)}',
    '[data-figure="prefix"] .pg-ph .num{font-weight:400;letter-spacing:0;text-transform:none;font-variant-numeric:tabular-nums}',
    '[data-figure="prefix"] .pg-svg svg{display:block;width:100%;height:auto;overflow:visible}',
    '[data-figure="prefix"] .pg-eqrow{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:.2rem 2rem;margin:.75rem 0 .1rem;color:var(--ink);font-size:.98rem;min-height:2.4em}',
    '[data-figure="prefix"] .pg-eqrow>div{max-width:100%;overflow-x:auto;overflow-y:hidden}',
    '[data-figure="prefix"] [hidden]{display:none !important}',
    '[data-figure="prefix"] .pg-eqtext{font-family:var(--f-body);font-size:.95rem;color:var(--ink-2);text-align:center;line-height:1.7}',
    '[data-figure="prefix"] .pg-read{display:grid;gap:.55rem 1rem;grid-template-columns:repeat(auto-fit,minmax(12.5rem,1fr));margin:.5rem 0 .9rem}',
    '[data-figure="prefix"] .pg-cell{border-top:1px solid var(--rule);padding-top:.4rem;min-width:0}',
    '[data-figure="prefix"] .pg-cell .k{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);line-height:1.35;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    '[data-figure="prefix"] .pg-cell .v{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:1.02rem;line-height:1.35;color:var(--ink);overflow-wrap:anywhere}',
    '[data-figure="prefix"] .pg-cell .v.big{font-size:1.75rem;line-height:1.15;color:var(--ochre-ink)}',
    '[data-figure="prefix"] .pg-cell .v.big.tide{color:var(--tide)}',
    '[data-figure="prefix"] .pg-cell .s{font-family:var(--f-body);font-variant-numeric:tabular-nums;font-size:.84rem;line-height:1.45;color:var(--ink-2);overflow-wrap:anywhere}',
    '[data-figure="prefix"] sub,[data-figure="prefix"] sup{font-size:max(.78em,12px);line-height:0}',
    '[data-figure="prefix"] .pg-single{display:grid;gap:.75rem;grid-template-columns:minmax(0,1fr)}',
    '@media (min-width:760px){[data-figure="prefix"] .pg-single{grid-template-columns:minmax(0,1.25fr) minmax(0,1fr)}}',
    '[data-figure="prefix"] .pg-card{display:grid;gap:.5rem;align-content:start;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2)}',
    '[data-figure="prefix"] .pg-card p{margin:0}',
    '[data-figure="prefix"] .pg-card .pg-eq2{color:var(--ink);overflow-x:auto;overflow-y:hidden;font-size:.95rem}',
    '[data-figure="prefix"] .pg-d2{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.2rem .9rem;align-items:baseline;font-size:.84rem}',
    '[data-figure="prefix"] .pg-d2 dt{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);color:var(--ink-2);margin:0}',
    '[data-figure="prefix"] .pg-d2 dd{font-family:var(--f-mono);font-variant-numeric:tabular-nums;margin:0;color:var(--ink);text-align:right}',
    '[data-figure="prefix"] .pg-d2 dd.hot{color:var(--seal-ink);font-weight:500}',
    '[data-figure="prefix"] .pg-d2 dd.ok{color:var(--moss-ink);font-weight:500}',
    '[data-figure="prefix"] .chip{font-size:var(--fs-xs);font-weight:500;letter-spacing:.02em}',
    '[data-figure="prefix"] .pg-chip{justify-self:start;white-space:normal;line-height:1.4;padding:.24rem .55rem;border-radius:4px;font-size:var(--fs-xs)}',
    '[data-figure="prefix"] .pg-foot{margin-top:.85rem;padding-top:.55rem;border-top:1px solid var(--rule);display:grid;gap:.4rem;font-family:var(--f-body);font-size:.84rem;line-height:1.5;color:var(--ink-2)}',
    '[data-figure="prefix"] .pg-foot a{color:var(--ink)}',
    '[data-figure="prefix"] .pg-meths{display:flex;flex-wrap:wrap;gap:.35rem .9rem;align-items:center}',
    '[data-figure="prefix"] .pg-meth{display:inline-flex;flex-wrap:wrap;gap:.25rem;align-items:center}',
    '[data-figure="prefix"] .pg-meth>a,[data-figure="prefix"] .pg-meth>b{font-family:var(--f-display);font-size:.95rem;font-weight:var(--w-head);color:var(--ink);text-decoration:none;margin-right:.1rem}',
    '[data-figure="prefix"] .pg-meth>a:hover{color:var(--tide)}',
  ].join('\n');

  var SUB = '₀₁₂₃₄₅₆₇₈₉';
  function sub(n) { return String(n).split('').map(function (c) { return SUB[+c] || c; }).join(''); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function lin(a, x, b, y) { var r = new Float64Array(x.length); for (var i = 0; i < x.length; i++) r[i] = a * x[i] + b * y[i]; return r; }
  function matvec(W, x) { var r = new Float64Array(W.length); for (var i = 0; i < W.length; i++) r[i] = dot(W[i], x); return r; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function copy(o) { var r = {}; for (var k in o) r[k] = o[k]; return r; }
  /* typographic minus, fixed decimals */
  function fx(x, d) { if (!isFinite(x)) return '—'; var s = Math.abs(x).toFixed(d); return (x < 0 && +s !== 0 ? '−' : '') + s; }
  function fe(x) {
    if (!isFinite(x)) return '—';
    if (x === 0) return '0';
    var s = Math.abs(x).toExponential(1).replace('e-', 'e−').replace('e+', 'e+');
    return (x < 0 ? '−' : '') + s;
  }

  Atlas.register('prefix', function (el, A) {
    if (!document.getElementById(CSS_ID)) {
      var st = document.createElement('style'); st.id = CSS_ID; st.textContent = CSS; document.head.appendChild(st);
    }
    var LA = A.LA, H = A.h, esc = A.esc;
    var UID = 'pg' + Math.random().toString(36).slice(2, 7);
    var DK = 4, SQ = Math.sqrt(DK), NC = 6, LMAX = 8, RHO = 2.4;
    var TMAX = 3, HSTEP = 1, NT = 241, COORD = 0;
    var DEF = { mode: 'joint', ell: 3, s: 1.2, g: 0.5, phi: 35, t0: -0.5, seed: 40 };
    var S = copy(DEF);
    var M = gen(S.seed);
    var mix = 0;              /* display tween: 0 = joint softmax, 1 = separate softmax */
    var tween = null, sweeping = false, sweepRaf = 0, visible = true;

    /* ---------------- data: seeded synthetic tokens and a seeded single-token head ---------------- */
    function gen(seed) {
      var R = A.rng(1000 + 7919 * seed);
      var Q = LA.randOrth(DK, R);
      var u0 = Q[0], u1 = Q[1], u2 = Q[2], u3 = Q[3];
      function mk(a, th, o2, o3) {
        var v = new Float64Array(DK);
        for (var i = 0; i < DK; i++) v[i] = a * Math.cos(th) * u0[i] + a * Math.sin(th) * u1[i] + o2 * u2[i] + o3 * u3[i];
        return v;
      }
      function rv(sc) { var v = new Float64Array(DK); for (var i = 0; i < DK; i++) v[i] = sc * R.normal(); return v; }
      var base = 2 * Math.PI * R();
      var kc = [], kp = [], vc = [], vp = [], i, j, d;
      for (j = 0; j < NC; j++) kc.push(mk(1.4 + 0.75 * R(), base + 2 * Math.PI * j / NC + 0.5 * (R() - 0.5), 0.6 * R.normal(), 0.6 * R.normal()));
      var thP = base + Math.PI * (0.75 + 0.5 * R());
      for (i = 0; i < LMAX; i++) kp.push(mk(1.05 + 0.6 * R(), thP + 1.7 * (R() - 0.5), 0.5 * R.normal(), 0.5 * R.normal()));
      for (j = 0; j < NC; j++) vc.push(rv(0.85));
      var cP = rv(1), ncp = norm(cP);
      for (d = 0; d < DK; d++) cP[d] *= 2.5 / ncp;
      for (i = 0; i < LMAX; i++) { var e = rv(0.72); for (d = 0; d < DK; d++) e[d] += cP[d]; vp.push(e); }
      var Wq = LA.randn(DK, DK, R, 1.2), Wk = LA.randn(DK, DK, R, 0.55), Wv = LA.randn(DK, DK, R, 0.6);
      /* probe ray x1: random, tilted toward row 1 of W_v so the frozen head's line visibly slopes
         (any ray works for the theorem; this only makes the straight line easy to see) */
      var x1 = rv(0.5), wr = norm(Wv[0]), nx;
      for (d = 0; d < DK; d++) x1[d] += 0.45 * Wv[0][d] / wr;
      nx = norm(x1);
      for (d = 0; d < DK; d++) x1[d] /= nx;

      /* PCA plane of the fourteen value vectors (fixed per draw, so the picture does not jump with l) */
      var all = vc.concat(vp), n = all.length, mu = new Float64Array(DK);
      all.forEach(function (v) { for (var a = 0; a < DK; a++) mu[a] += v[a] / n; });
      var Cv = LA.zeros(DK, DK);
      all.forEach(function (v) { for (var a = 0; a < DK; a++) for (var b = 0; b < DK; b++) Cv[a][b] += (v[a] - mu[a]) * (v[b] - mu[b]) / n; });
      var E = LA.symEig(Cv), idx = [0, 1, 2, 3].sort(function (a, b) { return E.values[b] - E.values[a]; });
      function col(k) { var c = new Float64Array(DK); for (var a = 0; a < DK; a++) c[a] = E.vectors[a][k]; return c; }
      var e1 = col(idx[0]), e2 = col(idx[1]);
      var mp = new Float64Array(DK);
      vp.forEach(function (v) { for (var a = 0; a < DK; a++) mp[a] += v[a] / LMAX; });
      var dm = lin(1, mp, -1, mu);
      if (dot(e1, dm) < 0) e1 = lin(-1, e1, 0, e1);
      if (dot(e2, dm) < 0) e2 = lin(-1, e2, 0, e2);
      var tot = E.values.reduce(function (a, b) { return a + Math.max(0, b); }, 0);
      var frac = (Math.max(0, E.values[idx[0]]) + Math.max(0, E.values[idx[1]])) / tot;
      function proj(v) { var w = lin(1, v, -1, mu); return [dot(e1, w), dot(e2, w)]; }
      function projVec(v) { return [dot(e1, v), dot(e2, v)]; }
      /* token order is arbitrary: list prefix pairs in farthest-point order of their projected values,
         so that the first few already span a visible hull (keys travel with their values) */
      (function () {
        var P2 = vp.map(proj), cx = 0, cy = 0;
        P2.forEach(function (p) { cx += p[0] / LMAX; cy += p[1] / LMAX; });
        var left = P2.map(function (p, k) { return k; }), ord = [];
        var dist = function (a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); };
        var first = left.reduce(function (b, k) { return dist(P2[k], [cx, cy]) > dist(P2[b], [cx, cy]) ? k : b; }, 0);
        ord.push(first); left.splice(left.indexOf(first), 1);
        while (left.length) {
          var best = left[0], bd = -1;
          left.forEach(function (k) { var md = Math.min.apply(null, ord.map(function (o) { return dist(P2[k], P2[o]); })); if (md > bd) { bd = md; best = k; } });
          ord.push(best); left.splice(left.indexOf(best), 1);
        }
        kp = ord.map(function (k) { return kp[k]; }); vp = ord.map(function (k) { return vp[k]; });
      })();
      var o2 = proj(new Float64Array(DK));
      var pts = all.map(proj).concat([o2]);
      var extJ = ext(pts), extL;
      var more = pts.slice();
      vc.forEach(function (a) { var pa = proj(a); vp.forEach(function (b) { var pb = projVec(b); more.push([pa[0] + pb[0], pa[1] + pb[1]]); }); });
      extL = ext(more);
      function ext(ps) {
        var e = [Infinity, -Infinity, Infinity, -Infinity];
        ps.forEach(function (p) { e[0] = Math.min(e[0], p[0]); e[1] = Math.max(e[1], p[0]); e[2] = Math.min(e[2], p[1]); e[3] = Math.max(e[3], p[1]); });
        return e;
      }
      /* in-plane polar coordinates of keys (q lives in span(u0,u1), so <q,k> = ρ |k∥| cos(φ − θ_k)) */
      function polar(k) { var a = dot(u0, k), b = dot(u1, k); return { r: Math.hypot(a, b), th: Math.atan2(b, a) }; }
      return {
        seed: seed, u0: u0, u1: u1, kc: kc, kp: kp, vc: vc, vp: vp, Wq: Wq, Wk: Wk, Wv: Wv, x1: x1,
        proj: proj, projVec: projVec, origin2: o2, pcaFrac: frac, extJ: extJ, extL: extL,
        polC: kc.map(polar), polP: kp.map(polar),
      };
    }

    /* ---------------- attention ---------------- */
    function queryVec(phiDeg) {
      var r = phiDeg * Math.PI / 180, q = new Float64Array(DK);
      for (var i = 0; i < DK; i++) q[i] = RHO * (Math.cos(r) * M.u0[i] + Math.sin(r) * M.u1[i]);
      return q;
    }
    /* softmax attention of q over n key/value pairs; ks = key scale (number or per-token array); stable log-sum-exp */
    function attend(q, keys, vals, n, ks) {
      var lg = new Array(n), m = -Infinity, i, d;
      for (i = 0; i < n; i++) { lg[i] = dot(q, keys[i]) * (typeof ks === 'number' ? ks : ks[i]) / SQ; if (lg[i] > m) m = lg[i]; }
      var w = new Array(n), z = 0;
      for (i = 0; i < n; i++) { w[i] = Math.exp(lg[i] - m); z += w[i]; }
      var out = new Float64Array(vals[0].length);
      for (i = 0; i < n; i++) { w[i] /= z; for (d = 0; d < out.length; d++) out[d] += w[i] * vals[i][d]; }
      return { logits: lg, logZ: m + Math.log(z), w: w, out: out };
    }
    function order(w) { return w.map(function (v, i) { return i; }).sort(function (a, b) { return w[b] - w[a] || a - b; }).join(','); }
    function segDist(x, a, b) {
      var ab = lin(1, b, -1, a), L2 = dot(ab, ab);
      var t = L2 > 0 ? dot(lin(1, x, -1, a), ab) / L2 : 0;
      var tc = clamp(t, 0, 1);
      return { t: t, off: norm(lin(1, x, -1, lin(1, a, tc, ab))), offLine: norm(lin(1, x, -1, lin(1, a, t, ab))) };
    }

    function compute() {
      var q = queryVec(S.phi), ell = S.ell;
      var C = attend(q, M.kc, M.vc, NC, 1);
      var P = ell > 0 ? attend(q, M.kp, M.vp, ell, S.s) : null;
      /* joint softmax over P ⊎ C, computed directly (not via the identity) */
      var keys = [], vals = [], scl = [], i, j;
      for (i = 0; i < ell; i++) { keys.push(M.kp[i]); vals.push(M.vp[i]); scl.push(S.s); }
      for (j = 0; j < NC; j++) { keys.push(M.kc[j]); vals.push(M.vc[j]); scl.push(1); }
      var J = attend(q, keys, vals, ell + NC, scl);
      var lam = P ? 1 / (1 + Math.exp(C.logZ - P.logZ)) : 0;
      var prefJ = J.w.slice(0, ell), afterJ = J.w.slice(ell);
      var ratiosJ = afterJ.map(function (a, k) { return a / C.w[k]; });
      var spreadJ = 0; ratiosJ.forEach(function (r) { spreadJ = Math.max(spreadJ, Math.abs(r - (1 - lam))); });
      var ident = P ? lin(1 - lam, C.out, lam, P.out) : C.out;
      var residJ = norm(lin(1, J.out, -1, ident));
      var sd = P ? segDist(J.out, C.out, P.out) : { t: 0, off: 0, offLine: 0 };
      /* separate softmax (LLaMA-Adapter): content softmax recomputed on its own, prefix softmax scaled by g */
      var C2 = attend(q, M.kc, M.vc, NC, 1);
      var prefL = P ? P.w.map(function (w) { return S.g * w; }) : [];
      var outL = P ? lin(1, C2.out, S.g, P.out) : C2.out;
      var devL = 0; C2.w.forEach(function (w, k) { devL = Math.max(devL, Math.abs(w - C.w[k])); });
      var sdL = P ? segDist(outL, C.out, P.out) : { t: 0, off: 0, offLine: 0 };
      return {
        q: q, C: C, P: P, J: J, lam: lam, ZP: P ? Math.exp(P.logZ) : 0, ZC: Math.exp(C.logZ),
        prefJ: prefJ, afterJ: afterJ, ratiosJ: ratiosJ, spreadJ: spreadJ, residJ: residJ, sdJ: sd,
        orderJ: order(afterJ) === order(C.w),
        C2: C2, prefL: prefL, afterL: C2.w, ratiosL: C2.w.map(function (w, k) { return w / C.w[k]; }), devL: devL, outL: outL, sdL: sdL,
        orderL: order(C2.w) === order(C.w),
        pushL: norm(lin(1, outL, -1, C.out)), gNormP: P ? S.g * norm(P.out) : 0,
      };
    }

    /* locus of the output as the query turns through 360° (current l, s, g) */
    function locus(llama) {
      var pts = [];
      for (var a = 0; a < 360; a += 4) {
        var q = queryVec(a), C = attend(q, M.kc, M.vc, NC, 1), out;
        if (S.ell === 0) out = C.out;
        else {
          var P = attend(q, M.kp, M.vp, S.ell, S.s);
          if (llama) out = lin(1, C.out, S.g, P.out);
          else { var lam = 1 / (1 + Math.exp(C.logZ - P.logZ)); out = lin(1 - lam, C.out, lam, P.out); }
        }
        pts.push(M.proj(out));
      }
      return pts;
    }

    /* bar y-domain: smallest step covering every content bar over a full turn of the dial (stable while
       sweeping) and every bar on screen now; it only grows when a prefix bar is taller than that */
    var yCache = { key: '', v: 1 };
    var YSTEPS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1];
    function stepAbove(x) { for (var k = 0; k < YSTEPS.length; k++) if (YSTEPS[k] >= x - 1e-12) return YSTEPS[k]; return 1; }
    function yDomain(R) {
      var key = String(M.seed);
      if (yCache.key !== key) {
        var mx = 0;
        for (var a = 0; a < 360; a += 3) {
          var q = queryVec(a), C = attend(q, M.kc, M.vc, NC, 1), k;
          for (k = 0; k < NC; k++) mx = Math.max(mx, C.w[k]);
        }
        yCache = { key: key, v: mx };
      }
      var cur = yCache.v;
      R.C.w.forEach(function (w) { cur = Math.max(cur, w); });
      R.prefJ.forEach(function (w) { cur = Math.max(cur, w); });
      R.prefL.forEach(function (w) { cur = Math.max(cur, w); });
      return stepAbove(cur);
    }

    /* ---------------- single-token head ---------------- */
    function headJoint(x) {
      var q = matvec(M.Wq, x), k = matvec(M.Wk, x), v = matvec(M.Wv, x);
      var keys = [], vals = [], scl = [];
      for (var i = 0; i < S.ell; i++) { keys.push(M.kp[i]); vals.push(M.vp[i]); scl.push(S.s); }
      keys.push(k); vals.push(v); scl.push(1);
      return attend(q, keys, vals, S.ell + 1, scl).out;
    }
    function headLlama(x) {
      var v = matvec(M.Wv, x);
      if (S.ell === 0) return v;
      var q = matvec(M.Wq, x), P = attend(q, M.kp, M.vp, S.ell, S.s);
      return lin(1, v, S.g, P.out);
    }
    function headFrozen(x) {
      /* softmax over a single token is exactly 1: the frozen head is x ↦ W_v x for every weight setting */
      var q = matvec(M.Wq, x), k = matvec(M.Wk, x), v = matvec(M.Wv, x);
      return attend(q, [k], [v], 1, 1).out;
    }
    function fAt(fn, t) { return fn(lin(t, M.x1, 0, M.x1))[COORD]; }
    function d2(fn, t0) { return fAt(fn, t0 + HSTEP) - 2 * fAt(fn, t0) + fAt(fn, t0 - HSTEP); }
    function computeSingle() {
      var ts = [], f0 = [], fJ = [], fL = [];
      for (var i = 0; i < NT; i++) {
        var t = -TMAX + 2 * TMAX * i / (NT - 1);
        ts.push(t); f0.push(fAt(headFrozen, t)); fJ.push(fAt(headJoint, t)); fL.push(fAt(headLlama, t));
      }
      var t0 = S.t0;
      var res = { ts: ts, f0: f0, fJ: fJ, fL: fL, t0: t0, d0: d2(headFrozen, t0), dJ: d2(headJoint, t0), dL: d2(headLlama, t0) };
      res.p0 = [fAt(headFrozen, t0 - HSTEP), fAt(headFrozen, t0), fAt(headFrozen, t0 + HSTEP)];
      res.pJ = [fAt(headJoint, t0 - HSTEP), fAt(headJoint, t0), fAt(headJoint, t0 + HSTEP)];
      res.pL = [fAt(headLlama, t0 - HSTEP), fAt(headLlama, t0), fAt(headLlama, t0 + HSTEP)];
      /* scale for the affinity tolerance */
      var sc = 1; f0.concat(fJ, fL).forEach(function (v) { sc = Math.max(sc, Math.abs(v)); });
      res.tol = 1e-10 * sc;
      return res;
    }

    /* ---------------- colours (tokens, read on every render) ---------------- */
    function colors() {
      return {
        paper: A.css('--paper'), paper2: A.css('--paper-2'), paper3: A.css('--paper-3'),
        ink: A.css('--ink'), ink2: A.css('--ink-2'), ink3: A.css('--ink-3'), rule: A.css('--rule'),
        tide: A.css('--tide'), tideS: A.css('--tide-soft'), ochre: A.css('--ochre'), ochreS: A.css('--ochre-soft'),
        seal: A.css('--seal'), sealS: A.css('--seal-soft'), moss: A.css('--moss'), mossS: A.css('--moss-soft'),
        ochreI: A.css('--ochre-ink'), sealI: A.css('--seal-ink'),
      };
    }
    function T(x, y, s, sty, extra) {
      return '<text x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '"' + (extra || '') + ' style="' + sty + '">' + s + '</text>';
    }

    /* ---------------- DOM ---------------- */
    function figNumber() {
      var f = el.closest ? el.closest('figure') : null, n = f && f.querySelector('.fig-n');
      var t = n ? n.textContent.replace(/\s+/g, ' ').trim() : '';
      return /^Figure \d+$/i.test(t) ? t.replace(/^Figure/i, 'Fig.') + ' · ' : '';
    }
    var stage = H('div', { class: 'stage', role: 'group', 'aria-label': 'Interactive figure: Prefix Is a Gate. Within one attention layer a prefix rescales all content attention weights by one common factor and cannot reorder them.' });
    el.appendChild(stage);
    var head = H('div', { class: 'pg-head' }, [
      H('h3', { class: 'pg-title', html: 'Prefix Is a <em>Gate</em>' }),
      H('span', { class: 'pg-kicker', text: figNumber() + 'Thm VI.5 · one layer' }),
    ]);
    var instr = H('p', { class: 'pg-instr', text: 'Turn the query dial and lengthen the prefix. Within this layer every content weight shrinks by the same factor, so their order never changes. In LLaMA-Adapter mode the content bars match their no-prefix outlines exactly.' });
    stage.appendChild(head); stage.appendChild(instr);

    /* controls */
    var controls = H('div', { class: 'pg-controls' });
    var modeGrp = H('div', { class: 'pg-modegrp' });
    var modeLab = H('span', { class: 'label', id: UID + '-mode', text: 'attention mode' });
    var seg = A.seg([{ value: 'joint', label: 'Prefix-tuning' }, { value: 'llama', label: 'LLaMA-Adapter' }], S.mode, function (v) { setMode(v); }, 'attention mode');
    seg.el.setAttribute('aria-labelledby', UID + '-mode');
    modeGrp.appendChild(modeLab); modeGrp.appendChild(seg.el);
    controls.appendChild(modeGrp);

    var sEll = A.slider({ id: UID + '-ell', label: 'prefix length ℓ', min: 0, max: LMAX, step: 1, value: S.ell,
      format: function (v) { return v + (v === 1 ? ' token' : ' tokens'); }, onInput: function (v) { S.ell = v; renderAll(); } });
    var sS = A.slider({ id: UID + '-s', label: 'prefix strength s', min: 0, max: 3, step: 0.05, value: S.s,
      format: function (v) { return 'keys × ' + v.toFixed(2); }, onInput: function (v) { S.s = v; renderAll(); } });
    var sG = A.slider({ id: UID + '-g', label: 'gate g', min: 0, max: 1, step: 0.01, value: S.g,
      format: function (v) { return 'g = ' + v.toFixed(2); }, onInput: function (v) { S.g = v; renderAll(); } });
    function relabel(sl, html) { var lab = sl.el.querySelector('label'); if (lab) lab.innerHTML = html; }
    relabel(sEll, 'prefix length <span class="pg-var">ℓ</span>');
    relabel(sS, 'prefix strength <span class="pg-var">s</span>');
    relabel(sG, 'gate <span class="pg-var">g</span> · LLaMA mode');
    controls.appendChild(sEll.el); controls.appendChild(sS.el); controls.appendChild(sG.el);

    var dialGrp = H('div', { class: 'pg-dialgrp' });
    var dialBox = H('div', { class: 'pg-dial', 'aria-hidden': 'true', title: 'Drag to turn the query q. Ticks are key directions in the query plane (teal: content, ochre: prefix); tick length is proportional to the in-plane norm, since ⟨q,k⟩ = |q|·|k∥|·cos(φ − θₖ).' });
    var sPhi = A.slider({ id: UID + '-phi', label: 'query direction φ', min: 0, max: 359, step: 1, value: S.phi,
      format: function (v) { return v + '°'; }, onInput: function (v) { stopSweep(); S.phi = v; renderAll(); } });
    relabel(sPhi, 'query direction <span class="pg-var">φ</span>');
    dialGrp.appendChild(dialBox); dialGrp.appendChild(sPhi.el);
    controls.appendChild(dialGrp);

    var btns = H('div', { class: 'pg-btns' });
    var bSweep = H('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', text: 'Sweep query' });
    var bNew = H('button', { type: 'button', class: 'btn', text: 'New draw' });
    var bReset = H('button', { type: 'button', class: 'btn', text: 'Reset' });
    var seedOut = H('span', { class: 'pg-seed', 'aria-live': 'off' });
    btns.appendChild(bSweep); btns.appendChild(bNew); btns.appendChild(bReset); btns.appendChild(seedOut);
    controls.appendChild(btns);
    stage.appendChild(controls);

    /* legend */
    var legend = H('div', { class: 'pg-legend', 'aria-label': 'Legend' });
    stage.appendChild(legend);

    /* main panels */
    var main = H('div', { class: 'pg-main' });
    var pA = H('div', { class: 'pg-panel' });
    var pAh = H('div', { class: 'pg-ph' });
    var svgA = H('div', { class: 'pg-svg', role: 'img' });
    pA.appendChild(pAh); pA.appendChild(svgA);
    var pB = H('div', { class: 'pg-panel' });
    var pBh = H('div', { class: 'pg-ph' });
    var svgB = H('div', { class: 'pg-svg', role: 'img' });
    pB.appendChild(pBh); pB.appendChild(svgB);
    main.appendChild(pA); main.appendChild(pB);
    stage.appendChild(main);

    /* formula row (MathJax, typeset once; the two modes toggle visibility) */
    var eqRow = H('div', { class: 'pg-eqrow' });
    var eqJ1 = H('div', { html: '\\(\\mathrm{Attn}_q(P\\uplus C)=(1-\\lambda_q)\\,\\mathrm{Attn}_q(C)+\\lambda_q\\,\\mathrm{Attn}_q(P)\\)' });
    var eqJ2 = H('div', { html: '\\(\\lambda_q=\\dfrac{Z_q(P)}{Z_q(P)+Z_q(C)},\\quad Z_q(S)=\\textstyle\\sum_{s\\in S}e^{\\langle q,k_s\\rangle/2}\\)' });
    var eqL1 = H('div', { html: '\\(\\mathrm{out}_q=\\mathrm{Attn}_q(C)+g\\,\\mathrm{Attn}_q(P)\\)' });
    var eqL2 = H('div', { class: 'pg-eqtext', html: 'Two softmaxes. The content weights \\(e^{\\langle q,k_j\\rangle/2}/Z_q(C)\\) are untouched, and \\(g=0\\) gives \\(\\mathrm{out}_q=\\mathrm{Attn}_q(C)\\).' });
    [eqJ1, eqJ2, eqL1, eqL2].forEach(function (n) { eqRow.appendChild(n); });
    stage.appendChild(eqRow);

    /* readouts */
    var read = H('div', { class: 'pg-read' });
    function cell() { var c = H('div', { class: 'pg-cell' }); c.k = H('div', { class: 'k' }); c.v = H('div', { class: 'v' }); c.s = H('div', { class: 's' }); c.appendChild(c.k); c.appendChild(c.v); c.appendChild(c.s); read.appendChild(c); return c; }
    var c1 = cell(), c2 = cell(), c3 = cell(), c4 = cell();
    c1.v.className = 'v big';
    stage.appendChild(read);

    /* single-token panel */
    var single = H('div', { class: 'pg-single' });
    var pC = H('div', { class: 'pg-panel' });
    var pCh = H('div', { class: 'pg-ph', html: '<b>One token in, coordinate 1 out</b><span class="num">x = t·x₁, t ∈ [−3, 3]</span>' });
    var legC = H('div', { class: 'pg-legend' });
    var svgC = H('div', { class: 'pg-svg', role: 'img' });
    pC.appendChild(pCh); pC.appendChild(legC); pC.appendChild(svgC);
    var card = H('div', { class: 'pg-panel pg-card' });
    var cardH = H('div', { class: 'pg-ph', html: '<b>Mergeability test</b><span class="pg-nt">Thm VI.5(f)</span>' });
    var cardP = H('p');
    var cardEq = H('div', { class: 'pg-eq2', html: '\\(\\Delta_h^2 f(t_0)=f(t_0{+}h)-2f(t_0)+f(t_0{-}h)\\)' });
    var d2list = H('dl', { class: 'pg-d2' });
    var chip = H('span', { class: 'chip pg-chip' });
    var sT0 = A.slider({ id: UID + '-t0', label: 'probe t₀', min: -(TMAX - HSTEP), max: TMAX - HSTEP, step: 0.05, value: S.t0,
      format: function (v) { return 't₀ = ' + fx(v, 2); }, onInput: function (v) { S.t0 = v; renderSingle(); } });
    relabel(sT0, 'probe <span class="pg-var">t</span>₀ &nbsp;(<span class="pg-var">h</span> = ' + HSTEP + ')');
    [cardH, cardP, cardEq, d2list, chip, sT0.el].forEach(function (n) { card.appendChild(n); });
    single.appendChild(pC); single.appendChild(card);
    stage.appendChild(single);

    /* footer: sources and the atlas classification of the three methods */
    var foot = H('div', { class: 'pg-foot' });
    stage.appendChild(foot);
    var live = H('div', { class: 'sr-only', 'aria-live': 'polite' });
    stage.appendChild(live);

    /* ---------------- panel A: attention bars ---------------- */
    function svgBars(R, W, c) {
      var narrow = W < 400;
      var padL = 30, padR = 6, Lw = W - padL - padR;
      var out = [];
      /* mass block */
      var lamD = lerp(R.lam, 0, mix);                      /* prefix share of the unit bar (joint) */
      var contD = lerp(1 - R.lam, 1, mix);                  /* content mass */
      var prefMassL = R.P ? S.g : 0;
      var yT = 13, yB1 = 19, bh = 10, yB2 = yB1 + bh + 6;
      var xs = function (m) { return padL + m * Lw; };
      var cm = narrow ? 'content ' : 'content mass ', pm = narrow ? 'prefix ' : 'prefix mass ';
      var lab12 = ';font-size:12.5px;font-weight:500;font-variant-numeric:tabular-nums';
      out.push(T(padL, yT, cm + fx(mix > 0.5 ? 1 : 1 - R.lam, 3) + (mix > 0.5 ? (narrow ? ', own softmax' : ' (own softmax)') : ' = 1 − λ'), 'fill:' + c.tide + lab12));
      var prefTxt = mix > 0.5 ? pm + 'g = ' + fx(prefMassL, 3) : pm + 'λ = ' + fx(R.lam, 3);
      out.push(T(W - padR, yT, prefTxt, 'fill:' + c.ochreI + lab12, ' text-anchor="end"'));
      out.push('<rect x="' + padL + '" y="' + yB1 + '" width="' + Lw.toFixed(1) + '" height="' + bh + '" fill="none" stroke="' + c.rule + '" stroke-width="1" rx="2"/>');
      out.push('<rect x="' + padL + '" y="' + yB1 + '" width="' + (contD * Lw).toFixed(1) + '" height="' + bh + '" fill="' + c.tide + '" rx="2"/>');
      if (R.P) {
        /* joint: prefix slice completes the unit bar; separate: its own bar of length g on the second row */
        var jx = xs(1 - lamD), jw = lamD * Lw;
        out.push('<rect x="' + jx.toFixed(1) + '" y="' + yB1 + '" width="' + Math.max(0, jw).toFixed(1) + '" height="' + bh + '" fill="' + c.ochre + '" opacity="' + (1 - mix).toFixed(3) + '" rx="2"/>');
        out.push('<rect x="' + padL + '" y="' + yB2 + '" width="' + (prefMassL * Lw * mix).toFixed(1) + '" height="' + bh + '" fill="' + c.ochre + '" rx="2"/>');
      }
      /* captions in the body serif; the second-row note is drawn only where it fits beside the g bar */
      var capSty = 'fill:' + c.ink2 + ';font-family:var(--f-body);font-style:italic;font-size:12.5px', capY = yB2 + bh + 1;
      if (mix <= 0.5) out.push(T(W - padR, capY, R.P ? 'one softmax, so the two masses sum to 1' : 'no prefix tokens, so λ = 0', capSty + ';opacity:' + (1 - 2 * mix).toFixed(2), ' text-anchor="end"'));
      else if (!R.P) out.push(T(padL, capY, 'no prefix tokens', capSty));
      else if (xs(prefMassL) + 6 + 27 * 5.9 <= W - padR) out.push(T(xs(prefMassL) + 6, capY, 'second softmax, scaled by g', capSty));

      /* bars */
      var gT = yB2 + bh + 30, bT = gT + 10, bH = narrow ? 118 : 150, bB = bT + bH;
      var gap = narrow ? 12 : 18, pw = (Lw - gap) * 0.4, cw = Lw - gap - pw;
      var x0P = padL, x0C = padL + pw + gap, sp = pw / LMAX, sc = cw / NC;
      var bwP = sp * 0.62, bwC = sc * 0.58;
      var yMax = yDomain(R);
      var y = function (v) { return bB - v / yMax * bH; };
      [0, 0.5, 1].forEach(function (f) {
        var yy = y(yMax * f);
        out.push('<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + yy.toFixed(1) + '" y2="' + yy.toFixed(1) + '" stroke="' + c.rule + '" stroke-width="1"' + (f ? ' stroke-dasharray="2 3"' : '') + '/>');
        out.push(T(padL - 5, yy + 4, (yMax * f).toFixed(f === 0 ? 0 : 2).replace(/0$/, '').replace(/\.$/, ''), 'fill:' + c.ink2 + ';font-size:11px', ' text-anchor="end" class="mono"'));
      });
      out.push(T(x0P, gT, (narrow ? 'P · Σ ' : 'prefix P · Σ ') + fx(mix > 0.5 ? prefMassL : R.lam, 3), 'fill:' + c.ochreI + lab12));
      out.push(T(x0C, gT, (narrow ? 'C · Σ ' : 'content C · Σ ') + fx(mix > 0.5 ? 1 : 1 - R.lam, 3), 'fill:' + c.tide + lab12));
      var i, j, xx, v;
      for (i = 0; i < LMAX; i++) {
        xx = x0P + sp * (i + 0.5);
        var on = i < S.ell;
        if (on) {
          v = lerp(R.prefJ[i], R.prefL[i], mix);
          var tipP = '<b>p' + sub(i + 1) + '</b> prefix weight ' + fx(mix > 0.5 ? R.prefL[i] : R.prefJ[i], 4) +
            (mix > 0.5 ? ' = g · ' + fx(R.P.w[i], 4) : ' = λ · ' + fx(R.P.w[i], 4));
          out.push('<rect data-tip="' + esc(tipP) + '" x="' + (xx - bwP / 2).toFixed(1) + '" y="' + y(v).toFixed(1) + '" width="' + bwP.toFixed(1) + '" height="' + Math.max(0, bB - y(v)).toFixed(1) + '" fill="' + c.ochre + '" rx="1.5"/>');
        } else {
          out.push('<line x1="' + (xx - bwP / 2).toFixed(1) + '" x2="' + (xx + bwP / 2).toFixed(1) + '" y1="' + (bB - 0.5) + '" y2="' + (bB - 0.5) + '" stroke="' + c.ink3 + '" stroke-width="1" stroke-dasharray="1.5 2" opacity="0.6"/>');
        }
        if (on || !narrow) out.push(T(xx, bB + 15, 'p' + sub(i + 1), 'fill:' + (on ? c.ochreI : c.ink3) + ';font-size:12px;font-weight:500;opacity:' + (on ? 1 : 0.5), ' text-anchor="middle"'));
      }
      for (j = 0; j < NC; j++) {
        xx = x0C + sc * (j + 0.5);
        var bef = R.C.w[j], aft = lerp(R.afterJ[j], R.afterL[j], mix);
        var tipC = '<b>c' + sub(j + 1) + '</b> before ' + fx(bef, 4) + ' → after ' + fx(mix > 0.5 ? R.afterL[j] : R.afterJ[j], 4) +
          ' (× ' + fx(mix > 0.5 ? R.ratiosL[j] : R.ratiosJ[j], 4) + ')';
        out.push('<rect x="' + (xx - bwC / 2).toFixed(1) + '" y="' + y(bef).toFixed(1) + '" width="' + bwC.toFixed(1) + '" height="' + Math.max(0, bB - y(bef)).toFixed(1) + '" fill="' + c.tide + '" fill-opacity="0.10" stroke="' + c.tide + '" stroke-width="1" stroke-dasharray="3 2" rx="1.5"/>');
        out.push('<rect data-tip="' + esc(tipC) + '" x="' + (xx - bwC / 2 + 2).toFixed(1) + '" y="' + y(aft).toFixed(1) + '" width="' + Math.max(1, bwC - 4).toFixed(1) + '" height="' + Math.max(0, bB - y(aft)).toFixed(1) + '" fill="' + c.tide + '" rx="1"/>');
        out.push('<rect data-tip="' + esc(tipC) + '" x="' + (xx - sc / 2).toFixed(1) + '" y="' + bT + '" width="' + sc.toFixed(1) + '" height="' + bH + '" fill="' + c.paper + '" fill-opacity="0"/>');
        out.push(T(xx, bB + 15, 'c' + sub(j + 1), 'fill:' + c.tide + ';font-size:12px;font-weight:500', ' text-anchor="middle"'));
      }
      out.push('<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + bB + '" y2="' + bB + '" stroke="' + c.ink3 + '" stroke-width="1"/>');

      /* ratio strip: after ÷ before for every content token, flat at 1 − λ_q */
      var rT = bB + 34, rH = 46, rB = rT + rH;
      var ry = function (r) { return rB - r * rH; };
      out.push('<line x1="' + x0C + '" x2="' + (W - padR) + '" y1="' + ry(1) + '" y2="' + ry(1) + '" stroke="' + c.ink3 + '" stroke-width="1" stroke-dasharray="2 3"/>');
      out.push('<line x1="' + x0C + '" x2="' + (W - padR) + '" y1="' + ry(0) + '" y2="' + ry(0) + '" stroke="' + c.rule + '" stroke-width="1"/>');
      out.push(T(x0C - 4, ry(1) + 4, '1', 'fill:' + c.ink2 + ';font-size:11px', ' text-anchor="end" class="mono"'));
      out.push(T(x0C - 4, ry(0) + 4, '0', 'fill:' + c.ink2 + ';font-size:11px', ' text-anchor="end" class="mono"'));
      var rats = [];
      for (j = 0; j < NC; j++) rats.push([x0C + sc * (j + 0.5), ry(lerp(R.ratiosJ[j], R.ratiosL[j], mix))]);
      out.push('<line x1="' + x0C + '" x2="' + (W - padR) + '" y1="' + rats[0][1].toFixed(2) + '" y2="' + rats[0][1].toFixed(2) + '" stroke="' + c.ink + '" stroke-width="1.25" opacity="0.35"/>');
      out.push('<path d="M' + rats.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(2); }).join('L') + '" fill="none" stroke="' + c.ink + '" stroke-width="1.6"/>');
      rats.forEach(function (p) { out.push('<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(2) + '" r="3.2" fill="' + c.paper + '" stroke="' + c.ink + '" stroke-width="1.5"/>'); });
      var rl = mix > 0.5 ? 'all = 1 (untouched)' : 'all = 1 − λ = ' + fx(1 - R.lam, 3);
      out.push(T(padL - 26, rT + 5, 'after ÷ before', 'fill:' + c.ink2 + ';font-size:12px;font-weight:500'));
      out.push(T(padL - 26, rT + 21, rl, 'fill:' + c.ink + ';font-size:12px;font-weight:600;font-variant-numeric:tabular-nums'));
      out.push(T(padL - 26, rT + 37, R.P || mix > 0.5 ? (mix > 0.5 ? 'max |a′−a| ' + fe(R.devL) : 'spread ' + fe(R.spreadJ)) : 'no prefix, so λ = 0', 'fill:' + c.ink2 + ';font-size:12px;font-variant-numeric:tabular-nums'));
      var Ht = rB + 6;
      return '<svg viewBox="0 0 ' + W + ' ' + Ht + '" width="' + W + '" height="' + Ht + '" aria-hidden="true">' + out.join('') + '</svg>';
    }

    /* ---------------- panel B: value space (PCA plane) ---------------- */
    function hull(pts) {
      if (pts.length < 3) return pts.slice();
      var p = pts.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
      var cr = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); };
      var lo = [], up = [], i;
      for (i = 0; i < p.length; i++) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p[i]) <= 0) lo.pop(); lo.push(p[i]); }
      for (i = p.length - 1; i >= 0; i--) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p[i]) <= 0) up.pop(); up.push(p[i]); }
      up.pop(); lo.pop();
      return lo.concat(up);
    }
    function svgSpace(R, W, c) {
      var Ht = Math.round(clamp(W * 0.8, 250, 340));
      var pad = 22;
      var eJ = M.extJ, eL = M.extL;
      var e = [lerp(eJ[0], eL[0], mix), lerp(eJ[1], eL[1], mix), lerp(eJ[2], eL[2], mix), lerp(eJ[3], eL[3], mix)];
      var sx = (W - 2 * pad) / (e[1] - e[0]), sy = (Ht - 2 * pad - 8) / (e[3] - e[2]), k = Math.min(sx, sy);
      var cx = (e[0] + e[1]) / 2, cy = (e[2] + e[3]) / 2;
      var X = function (p) { return [W / 2 + (p[0] - cx) * k, Ht / 2 + 4 - (p[1] - cy) * k]; };
      var out = [];
      var defs = '<defs><marker id="' + UID + '-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1,1L9,5L1,9" fill="none" stroke="' + c.ink + '" stroke-width="1.5"/></marker>' +
        '<marker id="' + UID + '-ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1,1L9,5L1,9" fill="none" stroke="' + c.ochre + '" stroke-width="1.5"/></marker></defs>';
      var pc = M.vc.map(function (v) { return X(M.proj(v)); });
      var pp = M.vp.map(function (v) { return X(M.proj(v)); });
      var ph = function (pts) { return 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L') + 'Z'; };
      /* PC axes through the projected mean, faint */
      out.push(T(W - 4, Ht - 6, 'PC1 →', 'fill:' + c.ink2 + ';font-size:12px;font-weight:500', ' text-anchor="end"'));
      out.push(T(4, 13, '↑ PC2', 'fill:' + c.ink2 + ';font-size:12px;font-weight:500'));
      /* content hull */
      out.push('<path d="' + ph(hull(pc)) + '" fill="' + c.tide + '" fill-opacity="0.07" stroke="' + c.tide + '" stroke-opacity="0.45" stroke-width="1" stroke-dasharray="3 3"/>');
      /* prefix hull (active tokens) */
      var act = pp.slice(0, S.ell);
      if (act.length >= 3) out.push('<path d="' + ph(hull(act)) + '" fill="' + c.ochre + '" fill-opacity="0.16" stroke="' + c.ochre + '" stroke-width="1.25"/>');
      else if (act.length === 2) out.push('<line x1="' + act[0][0].toFixed(1) + '" y1="' + act[0][1].toFixed(1) + '" x2="' + act[1][0].toFixed(1) + '" y2="' + act[1][1].toFixed(1) + '" stroke="' + c.ochre + '" stroke-width="3" stroke-opacity="0.35" stroke-linecap="round"/>');
      /* output locus as q turns */
      var loc = locus(mix > 0.5).map(X);
      out.push('<path d="' + ph(loc) + '" fill="none" stroke="' + c.ink3 + '" stroke-width="1" stroke-dasharray="1.5 3" opacity="0.9"/>');
      /* key points and their label boxes first, so the small value labels can give way to them */
      var pC_ = X(M.proj(R.C.out)), pP_ = R.P ? X(M.proj(R.P.out)) : null;
      var pO = X(M.proj(R.J.out)), pOL = X(M.proj(R.outL));
      var pOut = [lerp(pO[0], pOL[0], mix), lerp(pO[1], pOL[1], mix)];
      var labs = [];
      var dots = [pC_, pOut].concat(pP_ ? [pP_] : []);
      /* each label tries its preferred side, then above, below and the opposite side, and takes the first spot
         that covers no key dot and no label already placed */
      function place(p, dir, off, text, col, size, weight) {
        function lay(d, o) {
          var dx = d[0], dy = d[1], dn = Math.hypot(dx, dy) || 1; dx /= dn; dy /= dn;
          var anc = dx > 0.35 ? 'start' : dx < -0.35 ? 'end' : 'middle', wpx = text.length * size * 0.6;
          var tx = p[0] + dx * o, ty = clamp(p[1] + dy * o + size * 0.35, size + 2, Ht - 4);
          if (anc === 'start') tx = clamp(tx, 4, W - 4 - wpx); else if (anc === 'end') tx = clamp(tx, 4 + wpx, W - 4); else tx = clamp(tx, 4 + wpx / 2, W - 4 - wpx / 2);
          var x0 = anc === 'start' ? tx : anc === 'end' ? tx - wpx : tx - wpx / 2;
          return { x: tx, y: ty, anc: anc, text: text, col: col, size: size, weight: weight, box: [x0 - 3, ty - size - 1, x0 + wpx + 3, ty + 4] };
        }
        function bad(L) {
          var b = L.box;
          return dots.some(function (d) { return d[0] > b[0] - 4 && d[0] < b[2] + 4 && d[1] > b[1] - 4 && d[1] < b[3] + 4; }) ||
            labs.some(function (M) { return b[0] < M.box[2] && b[2] > M.box[0] && b[1] < M.box[3] && b[3] > M.box[1]; });
        }
        var cands = [[dir, off], [[0, -1], off + 3], [[0, 1], off + 6], [[-dir[0], -dir[1]], off]], L = null;
        for (var k = 0; k < cands.length && !L; k++) { var T0 = lay(cands[k][0], cands[k][1]); if (!bad(T0)) L = T0; }
        labs.push(L || lay(dir, off));
      }
      var awayC = pP_ || [pC_[0] + 1, pC_[1]];
      var ux = awayC[0] - pC_[0], uy = awayC[1] - pC_[1], un = Math.hypot(ux, uy) || 1, nx = -uy / un, ny = ux / un;
      if (ny > 0) { nx = -nx; ny = -ny; }
      place(pOut, [nx, ny], 13, 'out', c.ink, 13, 600);
      if (pP_) place(pP_, [pP_[0] - pC_[0], pP_[1] - pC_[1]], 11, 'Attn(P)', c.ochreI, 12.5, 500);
      place(pC_, [pC_[0] - awayC[0], pC_[1] - awayC[1]], 11, 'Attn(C)', c.tide, 12.5, 500);
      function clash(b) {
        return labs.some(function (L) { return b[0] < L.box[2] && b[2] > L.box[0] && b[1] < L.box[3] && b[3] > L.box[1]; }) ||
          dots.some(function (d) { return d[0] > b[0] - 7 && d[0] < b[2] + 7 && d[1] > b[1] - 7 && d[1] < b[3] + 7; });
      }
      pc.forEach(function (p, j) {
        out.push('<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3.4" fill="' + c.tide + '" opacity="0.85"/>');
        if (!clash([p[0] + 4, p[1] - 15, p[0] + 21, p[1] - 1])) out.push(T(p[0] + 5, p[1] - 4, 'v' + sub(j + 1), 'fill:' + c.tide + ';font-size:12px;font-weight:500'));
      });
      pp.forEach(function (p, i) {
        var on = i < S.ell;
        out.push('<rect x="' + (p[0] - 3).toFixed(1) + '" y="' + (p[1] - 3).toFixed(1) + '" width="6" height="6" transform="rotate(45 ' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ')" fill="' + (on ? c.ochre : 'none') + '" stroke="' + c.ochre + '" stroke-width="1" opacity="' + (on ? 0.95 : 0.35) + '"/>');
      });
      if (pP_) {
        var p0 = X(M.origin2);
        out.push('<line x1="' + pC_[0].toFixed(1) + '" y1="' + pC_[1].toFixed(1) + '" x2="' + pP_[0].toFixed(1) + '" y2="' + pP_[1].toFixed(1) + '" stroke="' + c.ink2 + '" stroke-width="1.25" stroke-dasharray="4 3" opacity="' + (1 - 0.6 * mix).toFixed(2) + '"/>');
        if (mix > 0) {
          /* separate softmax: out = Attn(C) + g·Attn(P); the vector Attn(P) from the origin, and its g-multiple placed at Attn(C) */
          out.push('<g opacity="' + mix.toFixed(3) + '">');
          out.push('<path d="M' + (p0[0] - 4) + ',' + p0[1] + 'h8M' + p0[0] + ',' + (p0[1] - 4) + 'v8" stroke="' + c.ink2 + '" stroke-width="1.25"/>');
          out.push(T(p0[0] - 6, p0[1] + 13, '0', 'fill:' + c.ink2 + ';font-size:12px', ' text-anchor="end" class="mono"'));
          out.push('<line x1="' + p0[0].toFixed(1) + '" y1="' + p0[1].toFixed(1) + '" x2="' + pP_[0].toFixed(1) + '" y2="' + pP_[1].toFixed(1) + '" stroke="' + c.ochre + '" stroke-width="1.1" stroke-dasharray="3 3" marker-end="url(#' + UID + '-ao)" opacity="0.8"/>');
          if (S.g > 0.02) out.push('<line x1="' + pC_[0].toFixed(1) + '" y1="' + pC_[1].toFixed(1) + '" x2="' + pOL[0].toFixed(1) + '" y2="' + pOL[1].toFixed(1) + '" stroke="' + c.ink + '" stroke-width="1.5" marker-end="url(#' + UID + '-ah)"/>');
          out.push('</g>');
        }
        out.push('<circle cx="' + pP_[0].toFixed(1) + '" cy="' + pP_[1].toFixed(1) + '" r="5.5" fill="' + c.ochre + '" stroke="' + c.paper + '" stroke-width="1.5"/>');
      }
      out.push('<circle cx="' + pC_[0].toFixed(1) + '" cy="' + pC_[1].toFixed(1) + '" r="5.5" fill="' + c.tide + '" stroke="' + c.paper + '" stroke-width="1.5"/>');
      out.push('<circle cx="' + pOut[0].toFixed(1) + '" cy="' + pOut[1].toFixed(1) + '" r="6.5" fill="' + c.ink + '" stroke="' + c.paper + '" stroke-width="2"/>');
      labs.forEach(function (L) {
        out.push(T(L.x, L.y, L.text, 'fill:' + L.col + ';font-size:' + L.size + 'px;font-weight:' + L.weight + ';paint-order:stroke;stroke:' + c.paper + ';stroke-width:3px', ' text-anchor="' + L.anc + '"'));
      });
      return '<svg viewBox="0 0 ' + W + ' ' + Ht + '" width="' + W + '" height="' + Ht + '" aria-hidden="true">' + defs + out.join('') + '</svg>';
    }

    /* ---------------- panel C: single-token curves ---------------- */
    function niceTicks(lo, hi, n) {
      var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
      if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
      var t = [], v = Math.ceil(lo / step) * step;
      for (; v <= hi + 1e-9; v += step) t.push(Math.abs(v) < 1e-12 ? 0 : v);
      return { t: t, step: step };
    }
    function svgSingle(Sg, W, c) {
      var Ht = Math.round(clamp(W * 0.62, 220, 300));
      var padL = 36, padR = 10, padT = 10, padB = 26;
      /* y-range: all three curves (both modes, so switching modes does not rescale), rounded out to 0.5 */
      var lo = Infinity, hi = -Infinity;
      [Sg.f0, Sg.fJ, Sg.fL].forEach(function (a) { a.forEach(function (v) { if (v < lo) lo = v; if (v > hi) hi = v; }); });
      var yr = [Math.floor(lo * 2 - 0.15) / 2, Math.ceil(hi * 2 + 0.15) / 2];
      var X = function (t) { return padL + (t + TMAX) / (2 * TMAX) * (W - padL - padR); };
      var Y = function (v) { return padT + (yr[1] - v) / (yr[1] - yr[0]) * (Ht - padT - padB); };
      var out = [];
      var yt = niceTicks(yr[0], yr[1], 4);
      yt.t.forEach(function (v) {
        out.push('<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + Y(v).toFixed(1) + '" y2="' + Y(v).toFixed(1) + '" stroke="' + c.rule + '" stroke-width="1"' + (v === 0 ? '' : ' stroke-dasharray="2 3"') + '/>');
        out.push(T(padL - 5, Y(v) + 4, fx(v, yt.step < 1 ? (yt.step < 0.1 ? 2 : 1) : 0), 'fill:' + c.ink2 + ';font-size:11px', ' text-anchor="end" class="mono"'));
      });
      for (var t = -3; t <= 3; t++) {
        out.push('<line x1="' + X(t).toFixed(1) + '" x2="' + X(t).toFixed(1) + '" y1="' + (Ht - padB) + '" y2="' + (Ht - padB + 4) + '" stroke="' + c.ink3 + '" stroke-width="1"/>');
        out.push(T(X(t), Ht - padB + 16, fx(t, 0), 'fill:' + c.ink2 + ';font-size:11px', ' text-anchor="middle" class="mono"'));
      }
      out.push(T(W - padR, Ht - padB - 6, 't', 'fill:' + c.ink2 + ';font-size:15px;font-style:italic', ' text-anchor="end" class="serif"'));
      out.push('<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + (Ht - padB) + '" y2="' + (Ht - padB) + '" stroke="' + c.ink3 + '" stroke-width="1"/>');
      /* probe band */
      var t0 = Sg.t0, tl = t0 - HSTEP, tr = t0 + HSTEP;
      out.push('<rect x="' + X(tl).toFixed(1) + '" y="' + padT + '" width="' + (X(tr) - X(tl)).toFixed(1) + '" height="' + (Ht - padT - padB) + '" fill="' + c.ink3 + '" fill-opacity="0.06"/>');
      var path = function (fs) { return 'M' + Sg.ts.map(function (tt, i) { return X(tt).toFixed(1) + ',' + Y(fs[i]).toFixed(2); }).join('L'); };
      var fP = Sg.fJ.map(function (v, i) { return lerp(v, Sg.fL[i], mix); });
      out.push('<path d="' + path(Sg.f0) + '" fill="none" stroke="' + c.ink2 + '" stroke-width="1.6"/>');
      out.push('<path d="' + path(fP) + '" fill="none" stroke="' + c.ochre + '" stroke-width="2.4" stroke-linejoin="round"/>');
      /* chord and gap = Δ²/2 on the prefixed curve; the frozen line's chord lies on the line */
      var pP = [lerp(Sg.pJ[0], Sg.pL[0], mix), lerp(Sg.pJ[1], Sg.pL[1], mix), lerp(Sg.pJ[2], Sg.pL[2], mix)];
      var mid = (pP[0] + pP[2]) / 2;
      out.push('<line x1="' + X(tl).toFixed(1) + '" y1="' + Y(pP[0]).toFixed(2) + '" x2="' + X(tr).toFixed(1) + '" y2="' + Y(pP[2]).toFixed(2) + '" stroke="' + c.ochre + '" stroke-width="1.1" stroke-dasharray="3 2.5"/>');
      out.push('<line x1="' + X(t0).toFixed(1) + '" y1="' + Y(mid).toFixed(2) + '" x2="' + X(t0).toFixed(1) + '" y2="' + Y(pP[1]).toFixed(2) + '" stroke="' + c.seal + '" stroke-width="2.6"/>');
      var gapPx = Math.abs(Y(mid) - Y(pP[1]));
      if (gapPx > 6) {
        /* label beside the gap, on a side where no curve, line or chord crosses its box; if both sides are
           crossed the label is left out (the legend names the red bar) */
        var ly, lw = 28;
        var chordAt = function (t) { return lerp(pP[0], pP[2], (t - tl) / (tr - tl)); };
        var free = function (x0, x1) {
          if (x0 < padL || x1 > W - padR) return false;
          for (var i = 0; i < Sg.ts.length; i++) {
            var xx = X(Sg.ts[i]);
            if (xx < x0 - 2 || xx > x1 + 2) continue;
            var ys = [Y(fP[i]), Y(Sg.f0[i])];
            if (Sg.ts[i] >= tl && Sg.ts[i] <= tr) ys.push(Y(chordAt(Sg.ts[i])));
            for (var k = 0; k < ys.length; k++) if (ys[k] > ly - 12 && ys[k] < ly + 5) return false;
          }
          return true;
        };
        var xr = X(t0) + 6, xl = X(t0) - 6, labSty = 'fill:' + c.sealI + ';font-size:12.5px;font-weight:500;paint-order:stroke;stroke:' + c.paper + ';stroke-width:3px';
        [0.5, 0.25, 0.75, 0.1, 0.9].some(function (fr) {
          ly = lerp(Y(mid), Y(pP[1]), fr) + 3.5;
          if (free(xr, xr + lw)) { out.push(T(xr, ly, '½Δ²f', labSty)); return true; }
          if (free(xl - lw, xl)) { out.push(T(xl, ly, '½Δ²f', labSty, ' text-anchor="end"')); return true; }
          return false;
        });
      }
      [0, 1, 2].forEach(function (k) {
        var tt = [tl, t0, tr][k];
        out.push('<circle cx="' + X(tt).toFixed(1) + '" cy="' + Y(Sg.p0[k]).toFixed(2) + '" r="3" fill="' + c.paper + '" stroke="' + c.ink2 + '" stroke-width="1.4"/>');
        out.push('<circle cx="' + X(tt).toFixed(1) + '" cy="' + Y(pP[k]).toFixed(2) + '" r="3.4" fill="' + c.paper + '" stroke="' + c.ochre + '" stroke-width="1.6"/>');
      });
      return '<svg viewBox="0 0 ' + W + ' ' + Ht + '" width="' + W + '" height="' + Ht + '" aria-hidden="true">' + out.join('') + '</svg>';
    }

    /* ---------------- dial ---------------- */
    function drawDial(c) {
      var r0 = 26, cx = 31, cy = 31, out = [];
      out.push('<circle cx="' + cx + '" cy="' + cy + '" r="' + r0 + '" fill="' + c.paper2 + '" stroke="' + c.rule + '" stroke-width="1"/>');
      var rmax = 0; M.polC.concat(M.polP).forEach(function (p) { rmax = Math.max(rmax, p.r); });
      function tick(p, col, w, op) {
        var L = 4 + 9 * p.r / rmax, a = p.th;
        var x1 = cx + (r0 - L) * Math.cos(a), y1 = cy - (r0 - L) * Math.sin(a), x2 = cx + r0 * Math.cos(a), y2 = cy - r0 * Math.sin(a);
        out.push('<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '" stroke="' + col + '" stroke-width="' + w + '" stroke-linecap="round" opacity="' + op + '"/>');
      }
      M.polC.forEach(function (p) { tick(p, c.tide, 2, 1); });
      M.polP.forEach(function (p, i) { tick(p, c.ochre, 1.6, i < S.ell ? 1 : 0.25); });
      var a = S.phi * Math.PI / 180;
      out.push('<line x1="' + cx + '" y1="' + cy + '" x2="' + (cx + (r0 - 3) * Math.cos(a)).toFixed(1) + '" y2="' + (cy - (r0 - 3) * Math.sin(a)).toFixed(1) + '" stroke="' + c.ink + '" stroke-width="2" stroke-linecap="round"/>');
      out.push('<circle cx="' + cx + '" cy="' + cy + '" r="3" fill="' + c.ink + '"/>');
      dialBox.innerHTML = '<svg viewBox="0 0 62 62" width="62" height="62">' + out.join('') + '</svg>';
    }
    function dialAngle(e) {
      var rc = dialBox.getBoundingClientRect();
      var dx = e.clientX - (rc.left + rc.width / 2), dy = e.clientY - (rc.top + rc.height / 2);
      return Math.round(((Math.atan2(-dy, dx) * 180 / Math.PI) + 360) % 360) % 360;
    }
    var dragging = false;
    dialBox.addEventListener('pointerdown', function (e) {
      dragging = true; stopSweep();
      try { dialBox.setPointerCapture(e.pointerId); } catch (err) {}
      setPhi(dialAngle(e)); e.preventDefault();
    });
    dialBox.addEventListener('pointermove', function (e) { if (dragging) setPhi(dialAngle(e)); });
    dialBox.addEventListener('pointerup', function () { dragging = false; });
    dialBox.addEventListener('pointercancel', function () { dragging = false; });
    function setPhi(v) { S.phi = v; sPhi.input.value = v; sPhi.refresh(); renderAll(); }

    /* ---------------- legend + footer ---------------- */
    function sw(kind, c) {
      if (kind === 'ghost') return '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="1" y="1" width="10" height="10" fill="' + c.tide + '" fill-opacity=".1" stroke="' + c.tide + '" stroke-dasharray="2 1.5" rx="1"/></svg>';
      if (kind === 'out') return '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="' + c.ink + '"/></svg>';
      if (kind === 'ratio') return '<svg width="20" height="12" viewBox="0 0 20 12" aria-hidden="true"><path d="M1,6H19" stroke="' + c.ink + '" stroke-width="1.6"/><circle cx="10" cy="6" r="3" fill="' + c.paper + '" stroke="' + c.ink + '" stroke-width="1.4"/></svg>';
      if (kind === 'locus') return '<svg width="20" height="12" viewBox="0 0 20 12" aria-hidden="true"><path d="M1,6H19" stroke="' + c.ink3 + '" stroke-width="1.2" stroke-dasharray="1.5 3"/></svg>';
      if (kind === 'line') return '<svg width="20" height="12" viewBox="0 0 20 12" aria-hidden="true"><path d="M1,9L19,3" stroke="' + c.ink2 + '" stroke-width="1.6"/></svg>';
      if (kind === 'curve') return '<svg width="20" height="12" viewBox="0 0 20 12" aria-hidden="true"><path d="M1,10C7,10 11,2 19,3" fill="none" stroke="' + c.ochre + '" stroke-width="2.2"/></svg>';
      if (kind === 'vC') return '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="3.4" fill="' + c.tide + '"/></svg>';
      if (kind === 'vP') return '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="3" y="3" width="6" height="6" transform="rotate(45 6 6)" fill="' + c.ochre + '"/></svg>';
      if (kind === 'gap') return '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M6,1V11" stroke="' + c.seal + '" stroke-width="2.6"/></svg>';
      var col = kind === 'P' ? c.ochre : c.tide;
      return '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="1" y="1" width="10" height="10" fill="' + col + '" rx="1.5"/></svg>';
    }
    function drawLegend(c) {
      legend.innerHTML = [
        '<span>' + sw('C', c) + 'content C, with prefix</span>',
        '<span>' + sw('ghost', c) + 'content C, no prefix</span>',
        '<span>' + sw('P', c) + 'prefix P</span>',
        '<span>' + sw('ratio', c) + 'after ÷ before</span>',
        '<span>' + sw('vC', c) + 'content values, hull</span>',
        '<span>' + sw('vP', c) + 'prefix values in use, hull</span>',
        '<span>' + sw('out', c) + 'output</span>',
        '<span>' + sw('locus', c) + 'output as q turns</span>',
      ].join('');
      legC.innerHTML = [
        '<span>' + sw('line', c) + 'frozen head (a line for any weights)</span>',
        '<span>' + sw('curve', c) + (S.mode === 'llama' ? 'with LLaMA-Adapter prefix' : 'with prefix') + '</span>',
        '<span>' + sw('gap', c) + '½Δ²f at t₀</span>',
      ].join('');
    }
    /* citations come from ATLAS_DATA when present (authors + venue), with fallbacks */
    var PAPERS = [
      { id: 'unified-view-peft', claim: 'identity (b)', fb: { authors: 'He et al.', venue: 'ICLR 2022', url: 'https://arxiv.org/abs/2110.04366', title: 'Towards a Unified View of Parameter-Efficient Transfer Learning' } },
      { id: 'prefix-tuning-capabilities-limitations', claim: 'rigidity (c)', fb: { authors: 'Petrov et al.', venue: 'ICLR 2024', url: 'https://arxiv.org/abs/2310.19698', title: 'When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations' } },
    ];
    function cite(claim, au, venue, url, title) {
      return claim + ': <a href="' + esc(url) + '" title="' + esc(title || '') + '" target="_blank" rel="noopener">' + esc(au) + '</a> <span class="num">(' + esc(venue) + ')</span>';
    }
    function drawFoot() {
      var D = A.data(), pap = D.papers || [];
      var src = PAPERS.map(function (p) {
        var r = p.fb; for (var i = 0; i < pap.length; i++) if (pap[i].id === p.id) r = pap[i];
        return cite(p.claim, r.authors, r.venue || r.year, r.url, r.title);
      });
      var la = A.method('llama-adapter');
      src.push(la ? cite('separate softmax (d)', la.authors, la.venue || la.year, la.url, la.paper_title) :
        cite('separate softmax (d)', 'Zhang et al.', 'ICLR 2024', 'https://arxiv.org/abs/2303.16199', 'LLaMA-Adapter'));
      var axes = D.axes || [];
      function axName(key, val) {
        for (var i = 0; i < axes.length; i++) if (axes[i].key === key) for (var j = 0; j < axes[i].values.length; j++) if (axes[i].values[j].key === val) return axes[i].values[j].name;
        return val;
      }
      var meths = ['prefix-tuning', 'p-tuning-v2', 'llama-adapter'].map(function (id) {
        var m = A.method(id);
        if (!m) return '';
        var nm = m.url ? '<a href="' + esc(m.url) + '" target="_blank" rel="noopener" title="' + esc(m.paper_title || '') + '">' + esc(m.name) + '</a>' : '<b>' + esc(m.name) + '</b>';
        var chips;
        if (m.coords && m.coords.base_point) {
          var bp = m.coords.base_point, mg = m.coords.merge;
          chips = '<span class="chip ' + (bp === 'unpointed' ? 'no' : bp === 'neutral' || bp === 'regular' ? 'yes' : 'cond') + '">' + esc(axName('base_point', bp)) + '</span>' +
            (mg ? '<span class="chip ' + (mg === 'Minf' ? 'no' : /^M1/.test(mg) ? 'yes' : 'cond') + '">merge: ' + esc(axName('merge', mg)) + '</span>' : '');
        } else {
          chips = '<span class="chip">classification pending</span>' +
            (m.mergeable ? '<span class="chip ' + (m.mergeable === 'no' ? 'no' : m.mergeable === 'yes' ? 'yes' : 'cond') + '">mergeable: ' + esc(m.mergeable) + '</span>' : '');
        }
        return '<span class="pg-meth">' + nm + chips + '</span>';
      }).filter(Boolean);
      foot.innerHTML = '<div>' + src.join(' · ') + '. Tokens are seeded synthetic data in ℝ⁴ (key scale 1/√4); every number is computed in your browser.</div>' +
        (meths.length ? '<div class="pg-meths"><span class="label">In the atlas</span>' + meths.join('') + '</div>' : '');
    }

    /* ---------------- readouts ---------------- */
    function pct(x) { return (100 * x).toFixed(1) + '%'; }
    function updateReadouts(R) {
      var L = S.mode === 'llama';
      c1.v.className = 'v big' + (L ? ' tide' : '');
      if (!L) {
        c1.k.innerHTML = 'gate <span class="pg-var">λ<sub>q</sub></span> = prefix share of attention';
        c1.v.textContent = fx(R.lam, 3);
        c1.s.textContent = R.P ? 'Z(P) = ' + fx(R.ZP, 3) + '  ·  Z(C) = ' + fx(R.ZC, 3) : 'empty prefix, so Z(P) = 0 and Z(C) = ' + fx(R.ZC, 3);
        c2.k.textContent = 'every content weight is multiplied by';
        c2.v.textContent = fx(1 - R.lam, 3) + ' = 1 − λ';
        c2.s.innerHTML = 'max |<i>a</i>′<sub><i>j</i></sub>/<i>a</i><sub><i>j</i></sub> − (1 − λ)| = ' + fe(R.spreadJ) + ' · order ' + (R.orderJ ? 'preserved' : 'changed');
        c3.k.innerHTML = 'identity check <span class="pg-nt">(Thm VI.5b)</span>';
        c3.v.textContent = 'residual ' + fe(R.residJ);
        c3.s.textContent = 'one softmax over P ⊎ C vs (1 − λ)·Attn(C) + λ·Attn(P), in ℝ⁴';
        c4.k.innerHTML = 'output moves toward <span class="pg-nt">conv <i>V</i><sub>P</sub></span>';
        if (R.P) {
          c4.v.textContent = pct(R.sdJ.t) + ' of the way';
          c4.s.innerHTML = 'from Attn(C) to Attn(P) ∈ conv <i>V</i><sub>P</sub> · off the segment by ' + fe(R.sdJ.offLine);
        } else { c4.v.textContent = 'out = Attn(C)'; c4.s.innerHTML = 'no prefix tokens, nothing to mix in'; }
      } else {
        c1.k.innerHTML = 'gate <span class="pg-var">g</span> on a separate prefix softmax';
        c1.v.textContent = fx(S.g, 3);
        c1.s.textContent = (R.P ? 'prefix mass ' + fx(S.g, 3) + ' + content mass 1 = ' + fx(1 + S.g, 3) : 'empty prefix, nothing to gate') + ' · zero-initialised (g = tanh(gₗ) in the paper, a raw scalar in HF PEFT)';
        c2.k.textContent = 'every content weight is multiplied by';
        c2.v.textContent = '1 exactly';
        c2.s.innerHTML = 'max |<i>a</i>′<sub><i>j</i></sub> − <i>a</i><sub><i>j</i></sub>| = ' + fe(R.devL) + ' · order ' + (R.orderL ? 'preserved' : 'changed');
        c3.k.innerHTML = 'pointed at <span class="pg-var">g</span> = 0 <span class="pg-nt">(Thm VI.5d)</span>';
        c3.v.textContent = '‖out − Attn(C)‖ = ' + fx(R.pushL, 3);
        c3.s.textContent = '= g·‖Attn(P)‖ = ' + fx(R.gNormP, 3) + '; vanishes exactly at g = 0';
        c4.k.textContent = 'additive branch, not a convex mix';
        if (R.P) {
          c4.v.textContent = 'off the segment by ' + fx(R.sdL.off, 3);
          c4.s.innerHTML = 'distance in ℝ⁴ from out to [Attn(C), Attn(P)]';
        } else { c4.v.textContent = 'out = Attn(C)'; c4.s.innerHTML = 'no prefix tokens'; }
      }
      pAh.innerHTML = '<b>Attention weights over [P ; C]</b><span class="num">q at φ = ' + S.phi + '°, |q| = ' + RHO + '</span>';
      pBh.innerHTML = '<b>Value space · PCA plane</b><span class="num">keeps ' + pct(M.pcaFrac) + ' of value variance</span>';
      seedOut.textContent = 'draw ' + M.seed;
      eqJ1.hidden = L; eqJ2.hidden = L; eqL1.hidden = !L; eqL2.hidden = !L;
      var desc = L ? 'LLaMA-Adapter mode, gate g ' + fx(S.g, 2) + '. Content weights unchanged.' :
        'Prefix-tuning mode. Gate lambda ' + fx(R.lam, 3) + '. Every content weight multiplied by ' + fx(1 - R.lam, 3) + '; order ' + (R.orderJ ? 'preserved' : 'changed') + '.';
      svgA.setAttribute('aria-label', 'Bar chart of attention weights. ' + desc);
      svgB.setAttribute('aria-label', 'Value-space projection: the output lies ' + (L ? 'at Attn(C) plus g times Attn(P).' : pct(R.sdJ.t) + ' of the way from Attn(C) to Attn(P).'));
      announce(desc);
    }
    var announce = A.debounce(function (t) { live.textContent = t; }, 700);

    function updateSingleText(Sg) {
      var L = S.mode === 'llama';
      var dP = L ? Sg.dL : Sg.dJ, affine = Math.abs(dP) <= Sg.tol;
      var dPs = affine || Math.abs(dP) < 1e-3 ? fe(dP) : fx(dP, 4);
      d2list.innerHTML = '<dt>frozen head</dt><dd class="ok">Δ²f = ' + fe(Sg.d0) + '</dd>' +
        '<dt>' + (L ? 'with LLaMA-Adapter' : 'with prefix') + '</dt><dd class="' + (affine ? 'ok' : 'hot') + '">Δ²f = ' + dPs + '</dd>';
      if (affine) {
        chip.className = 'chip yes pg-chip';
        chip.textContent = S.ell === 0 ? 'No prefix, so this is the frozen head.' :
          L && S.g === 0 ? 'At g = 0 this is exactly the frozen head (pointed).' :
          L && S.ell === 1 ? 'With ℓ = 1 the separate softmax is constant, so the prefix adds a fixed shift and the head stays affine.' : 'Affine on one-token inputs.';
      } else {
        chip.className = 'chip no pg-chip';
        chip.textContent = 'Not affine, so no weight setting of a standard head reproduces it, and the method is not box-locally mergeable.';
      }
      cardP.innerHTML = 'On a one-token context the softmax equals 1, so a frozen head computes the affine map <i>x</i> ↦ <i>W<sub>o</sub></i>(<i>W<sub>v</sub>x</i> + <i>b<sub>v</sub></i>) + <i>b<sub>o</sub></i>, here <i>W<sub>v</sub>x</i>, which is a straight line in <i>t</i> for every choice of weights. ' +
        (L ? 'LLaMA-Adapter adds <i>g</i>·Attn<sub>q(x)</sub>(P), and the prefix softmax depends on <i>x</i> through <i>q</i> = <i>W<sub>q</sub>x</i>.' :
          'A prefix makes the gate λ(<i>x</i>) depend on the input, which bends the curve.');
      svgC.setAttribute('aria-label', 'Plot of output coordinate 1 against t. Frozen head second difference ' + fe(Sg.d0) + '; prefixed head second difference ' + dPs + '.');
    }

    /* ---------------- render ---------------- */
    function widthOf(node, fb) { var w = node.getBoundingClientRect().width; return w > 40 ? Math.floor(w) : fb; }
    function renderAll() {
      var c = colors();
      var R = compute();
      svgA.innerHTML = svgBars(R, widthOf(svgA, 420), c);
      svgB.innerHTML = svgSpace(R, widthOf(svgB, 380), c);
      drawDial(c);
      sG.el.classList.toggle('is-off', S.mode !== 'llama');
      sG.input.disabled = S.mode !== 'llama';
      updateReadouts(R);
      renderSingle(c);
    }
    function renderSingle(c) {
      c = c || colors();
      var Sg = computeSingle();
      svgC.innerHTML = svgSingle(Sg, widthOf(svgC, 480), c);
      updateSingleText(Sg);
      drawLegend(c);
    }

    var tweenTimer = 0;
    function setMode(v, instant) {
      if (v === S.mode && !instant) return;
      S.mode = v; seg.set(v);
      var target = v === 'llama' ? 1 : 0;
      if (tween) cancelAnimationFrame(tween);
      clearTimeout(tweenTimer);
      if (instant || A.reducedMotion()) { tween = null; mix = target; renderAll(); return; }
      var from = mix, t0 = null, dur = 380;
      var step = function (ts) {
        if (t0 === null) t0 = ts;
        var u = Math.min(1, (ts - t0) / dur), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        mix = lerp(from, target, e); renderAll();
        tween = u < 1 ? requestAnimationFrame(step) : null;
      };
      tween = requestAnimationFrame(step);
      /* rAF is paused in background tabs and throttled headless: guarantee the final frame */
      tweenTimer = setTimeout(function () { if (tween) cancelAnimationFrame(tween); tween = null; mix = target; renderAll(); }, dur + 120);
      renderAll();
    }

    /* sweep the query around the dial (off under reduced motion) */
    function stopSweep() { if (!sweeping) return; sweeping = false; cancelAnimationFrame(sweepRaf); sweepRaf = 0; bSweep.setAttribute('aria-pressed', 'false'); bSweep.textContent = 'Sweep query'; }
    function startSweep() {
      if (A.reducedMotion()) return;
      sweeping = true; bSweep.setAttribute('aria-pressed', 'true'); bSweep.textContent = 'Stop';
      var last = null, acc = S.phi;
      var step = function (ts) {
        if (!sweeping) return;
        /* off screen the loop parks itself (sweepRaf = 0); the visibility observer restarts it */
        if (!visible) { sweepRaf = 0; return; }
        if (last !== null) { acc = (acc + Math.min(ts - last, 100) * 0.024) % 360; var v = Math.floor(acc); if (v !== S.phi) { S.phi = v; sPhi.input.value = v; sPhi.refresh(); renderAll(); } }
        last = ts; sweepRaf = requestAnimationFrame(step);
      };
      resumeSweep = function () { if (sweeping && !sweepRaf) { last = null; acc = S.phi; sweepRaf = requestAnimationFrame(step); } };
      sweepRaf = requestAnimationFrame(step);
    }
    var resumeSweep = function () {};
    bSweep.addEventListener('click', function () { if (sweeping) stopSweep(); else startSweep(); });
    if (A.reducedMotion()) { bSweep.disabled = true; bSweep.title = 'Animation is off because reduced motion is requested; use the dial or the slider.'; }
    bNew.addEventListener('click', function () { S.seed += 1; M = gen(S.seed); yCache.key = ''; renderAll(); });
    function syncControls() {
      [[sEll, S.ell], [sS, S.s], [sG, S.g], [sPhi, S.phi], [sT0, S.t0]].forEach(function (p) { p[0].input.value = p[1]; p[0].refresh(); });
    }
    bReset.addEventListener('click', function () {
      stopSweep();
      var keep = S; for (var k in DEF) keep[k] = DEF[k];
      M = gen(S.seed); yCache.key = '';
      syncControls(); setMode(DEF.mode, true);
    });

    /* tooltips on bars */
    svgA.addEventListener('mousemove', function (e) {
      var t = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
      if (t) A.tip.show(t.getAttribute('data-tip'), e); else A.tip.hide();
    });
    svgA.addEventListener('mouseleave', function () { A.tip.hide(); });

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) { es.forEach(function (en) { visible = en.isIntersecting; }); if (visible) resumeSweep(); }).observe(stage);
    }
    var onResize = A.debounce(function () { renderAll(); }, 80);
    if ('ResizeObserver' in window) {
      var lastW = 0;
      new ResizeObserver(function () { var w = stage.clientWidth; if (Math.abs(w - lastW) > 1) { lastW = w; onResize(); } }).observe(stage);
    } else window.addEventListener('resize', onResize);
    A.onTheme(function () { renderAll(); });

    drawFoot();
    renderAll();
    A.typeset(eqRow); A.typeset(cardEq);

    /* read-only hook for headless verification pages */
    el.__prefix = {
      state: function () { return S; }, model: function () { return M; }, compute: compute, computeSingle: computeSingle, setMode: setMode, render: renderAll,
      set: function (o) {
        for (var k in o) if (k !== 'mode') S[k] = o[k];
        if (o.seed != null) { M = gen(S.seed); yCache.key = ''; }
        syncControls();
        if (o.mode) setMode(o.mode, true); else renderAll();
      },
    };
  });
})();
