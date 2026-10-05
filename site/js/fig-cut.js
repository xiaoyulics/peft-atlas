/* Fig. "cut" (Figure 5): Cut the Tensor. Kronecker sums are LoRA across another cut.
   Thm II.11 (Kronecker incomparability, repaired: general splits, generic rank-one Kronecker rank
   kappa_1 = min(m1,m2) * min(n1,n2), part (e) for constrained LoKr) and Prop II.13 (sum_{i<=k} C_i (x) D_i
   = R^{-1}_* LoRA_k) of theory/framework.md; theory/visuals.md section 8.

   Delta in R^{16x16} is read as a 4-leg tensor Delta[(u1,u2),(v1,v2)] with R^16 = R^{m1} (x) R^{m2} on the output
   side and R^{n1} (x) R^{n2} on the input side, in the row-major Kronecker convention i = u1*m2 + u2,
   j = v1*n2 + v2, so that (C (x) D)[i][j] = C[u1][v1] * D[u2][v2]. The three 2|2 cuts of the four legs are reshapes:
     cut 1  {U1U2 | V1V2}   Delta itself                       m x n          rank       (LoRA's cut)
     cut 2  {U1V1 | U2V2}   R Delta [(u1,v1),(u2,v2)]          m1n1 x m2n2    rk_kron    (Van Loan-Pitsianis)
     cut 3  {U1V2 | U2V1}   R(Delta Pi^T) [(u1,v2),(u2,v1)]    m1n2 x m2n1               (Pi swaps V1 and V2)
   Each reshape permutes the 256 entries, so sum sigma_i^2 = ||Delta||_F^2 on every cut. Singular values come from
   a one-sided (Hestenes) Jacobi SVD in float64, which returns exact zeros at ~1e-16 * ||Delta|| (a Gram-matrix
   eigen-solver would smear them to ~1e-8). Numerical rank = #{sigma_i > 1e-10 ||Delta||_F}.

   Image comparison (Thm II.11(d),(e)) is exact: for M_{<=r} = Im LoRA_r and K_{<=k} = {rk_kron <= k},
     max rk_kron on M_{<=r} = min(r * kappa_1, min(m1n1, m2n2))        (II.11(c) + subadditivity: upper bound)
     max rank   on K_{<=k}  = min(k * min(m1,n1) min(m2,n2), 16)       (II.11(b) + subadditivity: upper bound)
   and both bounds are certified as attained by a seeded random witness, computed live (the figure says so
   if a witness ever falls short). M_{<=r} in K_{<=k} iff the first is <= k; K_{<=k} in M_{<=r} iff the second <= r. */
(function () {
  'use strict';

  var N = 16;                 // Delta in R^{16 x 16}
  var TOL = 1e-10;            // rank threshold, relative to ||Delta||_F (Delta is normalised to 1)
  var LOGMIN = -17;           // floor of the log axis for singular values
  var SPLITS = [
    { id: 's44', m1: 4, m2: 4, n1: 4, n2: 4, label: '4 ⊗ 4' },
    { id: 's28', m1: 2, m2: 8, n1: 2, n2: 8, label: '2 ⊗ 8' },
    { id: 'sx', m1: 2, m2: 8, n1: 8, n2: 2, label: '2⊗8 | 8⊗2' }
  ];
  var PRESETS = [
    { id: 'uv', label: 'uvᵀ', name: 'uvᵀ' },
    { id: 'id', label: 'I₁₆', name: 'I₁₆' },
    { id: 'cd', label: 'C ⊗ D', name: 'C ⊗ D' },
    { id: 'k2', label: 'C₁⊗D₁ + C₂⊗D₂', name: 'C₁⊗D₁+C₂⊗D₂' },
    { id: 'lora2', label: 'LoRA₂', name: 'LoRA₂' },
    { id: 'loha', label: 'LoHa', name: 'LoHa' },
    { id: 'hira', label: 'HiRA', name: 'HiRA' },
    { id: 'lokr', label: 'LoKr r′=1', name: 'LoKr' }
  ];
  var CUTS = [
    { n: '①', name: 'Ordinary cut', legs: 'U₁U₂ | V₁V₂', who: 'LoRA factors here', mat: 'Δ', boxes: ['B', 'A'] },
    { n: '②', name: 'Kronecker cut', legs: 'U₁V₁ | U₂V₂', who: 'Kronecker sums factor here', mat: 'ℛΔ', boxes: ['C', 'D'] },
    { n: '③', name: 'Shuffled cut', legs: 'U₁V₂ | U₂V₁', who: 'cut ② after swapping V₁ ⇄ V₂', mat: 'ℛ(ΔΠᵀ)', boxes: ['E', 'F'] }
  ];

  /* ---------- formatting ---------- */
  var SUPD = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  var SUBD = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
  function sup(n) { return String(n).split('').map(function (c) { return SUPD[c] || c; }).join(''); }
  function sub(n) { return String(n).split('').map(function (c) { return SUBD[c] || c; }).join(''); }
  function sci(x, d) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '∞';
    var p = x.toExponential(d == null ? 1 : d).split('e'), e = parseInt(p[1], 10);
    if (e === 0) return p[0];
    if (e === -1 || e === -2) return String(+x.toPrecision((d == null ? 1 : d) + 1));
    return p[0] + ' × 10' + sup(e);
  }
  function svStr(s) { return s < 1e-17 ? '< 10⁻¹⁷' : sci(s, 2); }
  function intStr(n) { return n.toLocaleString('en-US'); }
  /* theorem part letters such as (c) keep their case inside uppercase tags */
  function lc(t) { return String(t).replace(/\(([a-e](?:,\s?[a-e])*)\)/g, '<span class="lc">($1)</span>'); }
  /* Unicode sub/superscript runs as <sub>/<sup> in HTML, set in the surrounding font (the web fonts do not carry U+2070-209F) */
  var SUPR = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };
  function hsub(t) {
    return String(t).replace(/[₀-₉]+/g, function (m) { return '<sub>' + m.replace(/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 8320); }) + '</sub>'; })
      .replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, function (m) { return '<sup>' + m.replace(/./g, function (c) { return SUPR[c]; }) + '</sup>'; });
  }
  function ordinal(n) { var t = n % 100, u = n % 10; return n + (t >= 11 && t <= 13 ? 'th' : u === 1 ? 'st' : u === 2 ? 'nd' : u === 3 ? 'rd' : 'th'); }

  /* ---------- small dense linear algebra on Float64Array (row-major) ---------- */
  function gm(rnd, len) { var a = new Float64Array(len); for (var i = 0; i < len; i++) a[i] = rnd.normal(); return a; }
  /* (a x b) (x) (c x d) -> (ac x bd), row-major Kronecker convention */
  function kron(C, a, b, D, c, d) {
    var R = new Float64Array(a * c * b * d), W = b * d;
    for (var i = 0; i < a; i++) for (var j = 0; j < b; j++) {
      var x = C[i * b + j];
      if (x === 0) continue;
      for (var p = 0; p < c; p++) for (var q = 0; q < d; q++) R[(i * c + p) * W + j * d + q] = x * D[p * d + q];
    }
    return R;
  }
  function matmul(A, a, k, B, b) {
    var R = new Float64Array(a * b);
    for (var i = 0; i < a; i++) for (var p = 0; p < k; p++) {
      var x = A[i * k + p];
      if (x === 0) continue;
      for (var j = 0; j < b; j++) R[i * b + j] += x * B[p * b + j];
    }
    return R;
  }
  function frob(X) { var s = 0; for (var i = 0; i < X.length; i++) s += X[i] * X[i]; return Math.sqrt(s); }
  function addTo(X, Y) { for (var i = 0; i < X.length; i++) X[i] += Y[i]; return X; }
  /* singular values (descending) of a rows x cols row-major matrix by one-sided Jacobi (Hestenes) */
  function svdVals(M, rows, cols) {
    var tall = rows >= cols, nc = tall ? cols : rows, len = tall ? rows : cols, i, j;
    var V = new Array(nc);
    for (j = 0; j < nc; j++) V[j] = new Float64Array(len);
    for (i = 0; i < rows; i++) for (j = 0; j < cols; j++) { if (tall) V[j][i] = M[i * cols + j]; else V[i][j] = M[i * cols + j]; }
    for (var sweep = 0; sweep < 80; sweep++) {
      var rot = 0;
      for (var p = 0; p < nc - 1; p++) for (var q = p + 1; q < nc; q++) {
        var vp = V[p], vq = V[q], a = 0, b = 0, g = 0;
        for (i = 0; i < len; i++) { a += vp[i] * vp[i]; b += vq[i] * vq[i]; g += vp[i] * vq[i]; }
        if (g === 0 || Math.abs(g) <= 1e-15 * Math.sqrt(a * b)) continue;
        rot++;
        var zeta = (b - a) / (2 * g);
        var t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        var c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (i = 0; i < len; i++) { var x = vp[i], y = vq[i]; vp[i] = c * x - s * y; vq[i] = s * x + c * y; }
      }
      if (!rot) break;
    }
    var sv = [];
    for (j = 0; j < nc; j++) { var s2 = 0; for (i = 0; i < len; i++) s2 += V[j][i] * V[j][i]; sv.push(Math.sqrt(s2)); }
    return sv.sort(function (x, y) { return y - x; });
  }

  /* ---------- splits and cuts ---------- */
  function splitInfo(sp) {
    var m1 = sp.m1, m2 = sp.m2, n1 = sp.n1, n2 = sp.n2;
    return {
      kappa1: Math.min(m1, m2) * Math.min(n1, n2),     // generic rank-one Kronecker rank (Thm II.11(c))
      kmax: Math.min(m1 * n1, m2 * n2),                 // every Delta has rk_kron <= this (Thm II.11(a))
      rkCD: Math.min(m1, n1) * Math.min(m2, n2),        // rank of a generic C (x) D (Thm II.11(b))
      cost: [N + N, m1 * n1 + m2 * n2, m1 * n2 + m2 * n1], // parameters per unit of bond, cut by cut
      matched: m1 === n1 && m2 === n2,
      square: m1 === m2 && n1 === n2 && m1 === n1
    };
  }
  var cutCache = {};
  function cutMaps(sp) {
    if (cutCache[sp.id]) return cutCache[sp.id];
    var m1 = sp.m1, m2 = sp.m2, n1 = sp.n1, n2 = sp.n2;
    var dims = [[N, N], [m1 * n1, m2 * n2], [m1 * n2, m2 * n1]];
    var cuts = dims.map(function (d) { return { rows: d[0], cols: d[1], row: new Int16Array(N * N), col: new Int16Array(N * N), inv: new Int16Array(N * N) }; });
    for (var i = 0; i < N; i++) for (var j = 0; j < N; j++) {
      var e = i * N + j, u1 = Math.floor(i / m2), u2 = i % m2, v1 = Math.floor(j / n2), v2 = j % n2;
      cuts[0].row[e] = i; cuts[0].col[e] = j;
      cuts[1].row[e] = u1 * n1 + v1; cuts[1].col[e] = u2 * n2 + v2;
      cuts[2].row[e] = u1 * n2 + v2; cuts[2].col[e] = u2 * n1 + v1;
    }
    cuts.forEach(function (c) { for (var e = 0; e < N * N; e++) c.inv[c.row[e] * c.cols + c.col[e]] = e; });
    cutCache[sp.id] = cuts;
    return cuts;
  }
  function reshape(X, cut) { var M = new Float64Array(N * N); for (var e = 0; e < N * N; e++) M[cut.row[e] * cut.cols + cut.col[e]] = X[e]; return M; }
  function rankAcross(X, cut) {
    var f = frob(X), sv = svdVals(reshape(X, cut), cut.rows, cut.cols), r = 0;
    for (var i = 0; i < sv.length; i++) if (sv[i] > TOL * f) r++;
    return r;
  }

  /* ---------- the eight updates ---------- */
  function seedFor(pid, sid, draw) {
    var s = 2166136261, str = pid + '|' + sid + '|' + draw;
    for (var i = 0; i < str.length; i++) { s ^= str.charCodeAt(i); s = Math.imul(s, 16777619); }
    return (s >>> 0) || 1;
  }
  function buildDelta(pid, sp, draw, rhoU, rhoV) {
    var rnd = Atlas.rng(seedFor(pid, sp.id, draw));
    var m1 = sp.m1, m2 = sp.m2, n1 = sp.n1, n2 = sp.n2, X, i, j, a, b;
    switch (pid) {
      case 'uv': {
        /* u = sum_{i<rho_u} x_i (x) y_i, v = sum_{j<rho_v} z_j (x) w_j: Schmidt ranks rho_u, rho_v (generically exact) */
        var mu = Math.min(m1, m2), mv = Math.min(n1, n2), xs = [], ys = [], zs = [], ws = [];
        for (i = 0; i < mu; i++) { xs.push(gm(rnd, m1)); ys.push(gm(rnd, m2)); }
        for (j = 0; j < mv; j++) { zs.push(gm(rnd, n1)); ws.push(gm(rnd, n2)); }
        var u = new Float64Array(N), v = new Float64Array(N);
        for (i = 0; i < rhoU; i++) for (a = 0; a < m1; a++) for (b = 0; b < m2; b++) u[a * m2 + b] += xs[i][a] * ys[i][b];
        for (j = 0; j < rhoV; j++) for (a = 0; a < n1; a++) for (b = 0; b < n2; b++) v[a * n2 + b] += zs[j][a] * ws[j][b];
        X = matmul(u, N, 1, v, N);
        break;
      }
      case 'id':
        X = new Float64Array(N * N);
        for (i = 0; i < N; i++) X[i * N + i] = 1;
        break;
      case 'cd':
        X = kron(gm(rnd, m1 * n1), m1, n1, gm(rnd, m2 * n2), m2, n2);
        break;
      case 'k2':
        X = kron(gm(rnd, m1 * n1), m1, n1, gm(rnd, m2 * n2), m2, n2);
        addTo(X, kron(gm(rnd, m1 * n1), m1, n1, gm(rnd, m2 * n2), m2, n2));
        break;
      case 'lora2':
        X = matmul(gm(rnd, N * 2), N, 2, gm(rnd, 2 * N), N);
        break;
      case 'loha': {
        var P1 = matmul(gm(rnd, N * 2), N, 2, gm(rnd, 2 * N), N), P2 = matmul(gm(rnd, N * 2), N, 2, gm(rnd, 2 * N), N);
        X = new Float64Array(N * N);
        for (i = 0; i < N * N; i++) X[i] = P1[i] * P2[i];
        break;
      }
      case 'hira': {
        var W0 = gm(rnd, N * N), bb = gm(rnd, N), aa = gm(rnd, N);
        X = new Float64Array(N * N);
        for (i = 0; i < N; i++) for (j = 0; j < N; j++) X[i * N + j] = W0[i * N + j] * bb[i] * aa[j];
        break;
      }
      case 'lokr': {
        var Cm = gm(rnd, m1 * n1), bl = gm(rnd, m2), al = gm(rnd, n2);
        X = kron(Cm, m1, n1, matmul(bl, m2, 1, al, n2), m2, n2);
        break;
      }
    }
    var f = frob(X);
    for (i = 0; i < X.length; i++) X[i] /= f;
    return X;
  }
  function analyse(X, cuts) {
    var maxAbs = 0, i;
    for (i = 0; i < X.length; i++) maxAbs = Math.max(maxAbs, Math.abs(X[i]));
    var out = cuts.map(function (c) {
      var sv = svdVals(reshape(X, c), c.rows, c.cols), r = 0, e2 = 0;
      for (var k = 0; k < sv.length; k++) { if (sv[k] > TOL) r++; e2 += sv[k] * sv[k]; }
      return { sv: sv, rank: r, rows: c.rows, cols: c.cols, energyDev: Math.abs(e2 - 1) };
    });
    return { X: X, maxAbs: maxAbs, cuts: out };
  }
  /* seeded witnesses certifying that the two image-comparison bounds are attained */
  var witCache = {};
  function witnesses(sp) {
    if (witCache[sp.id]) return witCache[sp.id];
    var info = splitInfo(sp), cuts = cutMaps(sp), rnd = Atlas.rng(seedFor('witness', sp.id, 0));
    var m1 = sp.m1, m2 = sp.m2, n1 = sp.n1, n2 = sp.n2, r, k, out = { lora: [null], kron: [null] };
    for (r = 1; r <= N; r++) {
      var X = matmul(gm(rnd, N * r), N, r, gm(rnd, r * N), N);
      out.lora.push({ bound: Math.min(r * info.kappa1, info.kmax), witness: rankAcross(X, cuts[1]) });
    }
    for (k = 1; k <= info.kmax; k++) {
      var Y = new Float64Array(N * N);
      for (var t = 0; t < k; t++) addTo(Y, kron(gm(rnd, m1 * n1), m1, n1, gm(rnd, m2 * n2), m2, n2));
      out.kron.push({ bound: Math.min(k * info.rkCD, N), witness: rankAcross(Y, cuts[0]) });
    }
    witCache[sp.id] = out;
    return out;
  }

  /* ---------- module CSS ---------- */
  function injectCSS() {
    if (document.getElementById('css-cut')) return;
    var F = '[data-figure="cut"] ';
    var css = [
      F + '.cut-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.cut-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.2rem 1rem}',
      F + '.cut-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.4rem,1rem + 1.9cqi,2.05rem);line-height:1.1;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.cut-title i{color:var(--tide)}',
      F + '.cut-kicker{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.cut-instr{margin:.45rem 0 .8rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:70ch}',
      F + '.cut-instr b{color:var(--ink);font-weight:600}',
      F + '.cut-formula{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem 1.6rem;padding:.5rem .75rem;margin:0 0 1rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.9rem;line-height:1.5;color:var(--ink)}',
      F + '.cut-formula > span{display:inline-flex;flex-wrap:wrap;align-items:center;gap:.15rem .55rem;min-width:0}',
      F + '.cut-formula mjx-container{white-space:nowrap}',
      F + '.cut-tag{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.cut-num{font-family:var(--f-mono);font-variant-numeric:tabular-nums}',
      F + '.lc{text-transform:none}',
      F + '.cut-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem 1.6rem;margin-bottom:.9rem}',
      F + '.cut-group{min-width:0;border-top:1px solid var(--rule);padding-top:.5rem}',
      F + '.cut-gl{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem .6rem;margin-bottom:.45rem}',
      F + '.cut-gl .cut-tag{color:var(--ink-2)}',
      F + '.cut-gl .cut-tag:first-child{color:var(--ink)}',
      F + '.cut-pick{display:flex;flex-wrap:wrap;gap:.35rem .4rem;align-items:center}',
      F + '.cut-pick button{font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;padding:.34rem .7rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap;line-height:1.25}',
      F + '.cut-pick button:hover:not([aria-pressed="true"]):not(:disabled){border-color:var(--ink-2);color:var(--ink)}',
      F + '.cut-pick button[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--paper)}',
      F + '.cut-pick button.cut-util{border-style:dashed;color:var(--ink-2)}',
      F + '.cut-pick button.cut-util:hover{color:var(--tide);border-color:var(--tide)}',
      F + '.cut-sub{font-family:var(--f-mono);font-size:.75rem;line-height:1.5;color:var(--ink-2);margin-top:.4rem;font-variant-numeric:tabular-nums}',
      F + '.cut-sub b{color:var(--ink);font-weight:500}',
      F + '.cut-hint{font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2);margin-top:.45rem}',
      F + '.cut-sliders{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.4rem .9rem}',
      F + '.cut-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.cut-sl-top{display:flex;justify-content:space-between;align-items:baseline;gap:.4rem}',
      F + '.cut-sl label{font-family:var(--f-body);font-style:italic;font-size:var(--fs-sm);color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}',
      F + '.cut-sl label .nm{font-family:var(--f-ui);font-weight:500;font-style:normal;font-size:.75rem;letter-spacing:.01em;color:var(--ink-2);margin-left:.35rem}',
      F + '.cut-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.cut-sl input[type=range]{margin:0;height:1.25rem}',
      F + '.cut-sl.off{opacity:.42}',
      F + '.cut-desc{margin:.1rem 0 .7rem;padding:.55rem .8rem;border-left:3px solid var(--ink-3);background:var(--paper);border-radius:0 var(--radius) var(--radius) 0;font-size:var(--fs-sm);line-height:1.55;color:var(--ink-2)}',
      F + '.cut-desc .who{display:inline-block;white-space:nowrap;font-family:var(--f-mono);font-size:.78rem;color:var(--ink);font-variant-numeric:tabular-nums}',
      F + '.cut-desc mjx-container{white-space:nowrap}',
      F + '.cut-verdict{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .5rem;margin:0 0 .35rem}',
      F + '.cut-chip{display:inline-block;font-family:var(--f-ui);font-weight:500;font-size:.78rem;letter-spacing:.01em;padding:.2rem .6rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap;font-variant-numeric:tabular-nums}',
      F + '.cut-chip.y{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.cut-chip.n{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.cut-chip.o{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + '.cut-chip sub{font-size:.72em;line-height:1;vertical-align:-.25em}',
      F + '.cut-thm{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.cut-cost{font-family:var(--f-ui);font-size:.82rem;line-height:1.5;color:var(--ink-2);margin:0 0 1rem;font-variant-numeric:tabular-nums}',
      F + '.cut-cost b{font-family:var(--f-mono);color:var(--ink);font-weight:500}',
      F + '.cut-cost .c1{color:var(--tide);font-weight:500}',
      F + '.cut-cost .c2{color:var(--ochre-ink);font-weight:500}',
      F + '.cut-bottom{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.4rem;align-items:start}',
      F + '.cut-matgrid{display:grid;grid-template-columns:minmax(0,1fr);gap:.4rem 1.2rem;align-items:center}',
      F + '.cut-side{display:grid;gap:.6rem;align-content:center}',
      F + '.cut-panel{min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.75rem .8rem .7rem}',
      F + '.cut-ph{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem .7rem;margin-bottom:.4rem}',
      F + '.cut-ph h4{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.08rem;margin:0;color:var(--ink)}',
      F + '.cut-mat svg,' + F + '.cut-plane svg,' + F + '.cut-bars svg{display:block;width:auto;max-width:100%;height:auto;overflow:visible}',
      F + 'sub,' + F + 'sup{font-size:.76em;line-height:0;letter-spacing:0}',
      F + '.cut-mat svg{touch-action:pan-y;border-radius:4px}',
      F + '.cut-mat svg:focus-visible{outline:2px solid var(--ochre);outline-offset:3px}',
      F + '.cut-read{font-family:var(--f-body);font-size:.86rem;line-height:1.5;color:var(--ink-2);min-height:4.5em;font-variant-numeric:tabular-nums}',
      F + '.cut-read b{font-family:var(--f-mono);font-size:.92em;color:var(--ochre-ink);font-weight:500}',
      F + '.cut-replay{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;padding:.28rem .6rem;border-radius:4px;border:1px solid var(--rule);background:var(--paper-2);color:var(--ink-2);cursor:pointer;white-space:nowrap}',
      F + '.cut-replay:hover{border-color:var(--ochre);color:var(--ochre)}',
      F + '.cut-cards{display:grid;grid-template-columns:minmax(0,1fr);gap:.6rem;margin:.75rem 0 1rem}',
      F + '.cut-card{min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.55rem .7rem .55rem}',
      F + '.cut-card.on{border-color:var(--cutc);box-shadow:inset 3px 0 0 var(--cutc)}',
      F + '.cut-ch{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem}',
      F + '.cut-cn{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.02rem;color:var(--ink);white-space:nowrap}',
      F + '.cut-cn .b{color:var(--cutc);margin-right:.25rem}',
      F + '.cut-cl{font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--ink-2);letter-spacing:.01em;white-space:nowrap}',
      F + '.cut-rk{font-family:var(--f-mono);font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink-2);font-size:.75rem}',
      F + '.cut-rk b{font-size:1.45rem;font-weight:500;color:var(--cutc);letter-spacing:-.02em;margin-right:.1rem}',
      F + '.cut-cw{font-family:var(--f-body);font-size:.84rem;color:var(--ink-2);margin:.05rem 0 .3rem;line-height:1.4}',
      F + '.cut-cb{display:grid;grid-template-columns:112px minmax(0,1fr);gap:.5rem;align-items:center}',
      F + '.cut-cb svg.dg{display:block;width:112px;height:auto;overflow:visible}',
      F + '.cut-cf{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);margin-top:.3rem;line-height:1.45;font-variant-numeric:tabular-nums}',
      F + '.cut-cf b{color:var(--ink);font-weight:500}',
      F + '.cut-pv{display:grid;gap:.5rem;align-content:start}',
      F + '.cut-pv .ln{font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2)}',
      F + '.cut-pv .ln .cut-num{font-size:.8rem;color:var(--ink)}',
      F + '.cut-rel{font-family:var(--f-display);font-size:1.35rem;line-height:1.15;color:var(--ink)}',
      F + '.cut-rel.inc{color:var(--seal)}',
      F + '.cut-rel.sub{color:var(--moss)}',
      F + '.cut-def{font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
      F + '.cut-legend{display:flex;flex-wrap:wrap;gap:.4rem 1.3rem;margin:1rem 0 0;padding-top:.7rem;border-top:1px solid var(--rule);font-family:var(--f-ui);font-weight:500;font-size:.75rem;line-height:1.4;color:var(--ink-2)}',
      F + '.cut-li{display:inline-flex;align-items:center;gap:.4rem}',
      F + '.cut-li svg{flex:none;overflow:visible}',
      F + '.cut-foot{margin:.65rem 0 0;font-family:var(--f-body);font-size:.82rem;line-height:1.55;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.seg button,' + F + '.btn{font-weight:500}',
      '@container (min-width: 700px){' +
        F + '.cut-controls{grid-template-columns:minmax(0,1.05fr) minmax(0,1fr) minmax(0,.9fr)}' +
        F + '.cut-g-preset{grid-column:1 / -1}' +
        F + '.cut-matgrid{grid-template-columns:minmax(0,1fr) 13.5rem}' +
        F + '.cut-cards{grid-template-columns:repeat(3,minmax(0,1fr))}' +
        F + '.cut-bottom{grid-template-columns:minmax(0,1.12fr) minmax(0,1fr)}' +
      '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-cut';
    s.textContent = css;
    document.head.appendChild(s);
  }

  Atlas.register('cut', function (el, A) {
    injectCSS();
    var d3 = window.d3, h = A.h;
    var SVGNS = 'http://www.w3.org/2000/svg';

    /* ---------- state ---------- */
    var st = { preset: 'uv', split: 0, view: 1, draw: 0, rhoU: 4, rhoV: 4, r: 1, k: 1 };
    var data = null, hover = null, animating = false, lastDescKey = '', played = false;

    function compute() {
      var sp = SPLITS[st.split], info = splitInfo(sp), cuts = cutMaps(sp);
      var all = PRESETS.map(function (p) { return analyse(buildDelta(p.id, sp, st.draw, st.rhoU, st.rhoV), cuts); });
      var idx = 0;
      PRESETS.forEach(function (p, i) { if (p.id === st.preset) idx = i; });
      data = { sp: sp, info: info, cuts: cuts, all: all, idx: idx, cur: all[idx], wit: witnesses(sp) };
    }
    function col() {
      return {
        paper: A.css('--paper'), paper2: A.css('--paper-2'), ink: A.css('--ink'), ink2: A.css('--ink-2'), ink3: A.css('--ink-3'),
        rule: A.css('--rule'), tide: A.css('--tide'), tideSoft: A.css('--tide-soft'), ochre: A.css('--ochre'), ochreSoft: A.css('--ochre-soft'),
        moss: A.css('--moss'), seal: A.css('--seal'), pos: A.css('--c-multiplicative'), neg: A.css('--c-composition'),
        cut: [A.css('--tide'), A.css('--ochre'), A.css('--c-hybrid')]
      };
    }

    /* ---------- DOM skeleton ---------- */
    var stage = h('div', { class: 'stage cut-stage', role: 'group', 'aria-label': 'Cut the tensor: a 16 by 16 update read as a four-leg tensor has a different rank across each of its three two-by-two cuts. LoRA pays the ordinary rank, a sum of Kronecker products pays the rank across the Kronecker cut (Theorem II.11, Proposition II.13).' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'cut-head' }, [
      h('h3', { class: 'cut-title', html: 'Kronecker sums are <i>LoRA</i> across another cut' }),
      h('span', { class: 'cut-kicker', text: 'Cut the tensor · Thm II.11 · Prop II.13' })
    ]));
    stage.appendChild(h('p', { class: 'cut-instr', html: 'Pick an update <b>Δ</b> and a way to split ℝ¹⁶. The same 256 numbers, cut three ways, can have three different ranks. <b>LoRA</b> pays for the rank across cut ①, and a <b>sum of Kronecker products</b> pays for the rank across cut ②.' }));

    var kappaSpan = h('span', { class: 'cut-num' });
    var formula = h('div', { class: 'cut-formula', 'aria-label': 'Key identities' }, [
      h('span', {}, [h('span', { class: 'cut-tag', text: 'bend wires' }), h('span', { html: '\\(\\mathcal R(C\\otimes D)=\\operatorname{vec}C\\,\\operatorname{vec}D^{\\top}\\)' })]),
      h('span', {}, [h('span', { class: 'cut-tag', text: 'Prop II.13' }), h('span', { html: '\\(\\textstyle\\sum_{i\\le k}C_i\\otimes D_i=\\mathcal R^{-1}_{*}\\,\\mathrm{LoRA}_k\\)' })]),
      h('span', {}, [h('span', { class: 'cut-tag', html: lc('Thm II.11(c)') }), h('span', { style: 'white-space:nowrap' }, [h('span', { html: '\\(\\kappa_1=\\min(m_1,m_2)\\min(n_1,n_2)\\)' }), document.createTextNode(' '), kappaSpan])])
    ]);
    stage.appendChild(formula);

    /* controls */
    var controls = h('div', { class: 'cut-controls' });
    stage.appendChild(controls);
    function pickGroup(items, onPick, aria) {
      var wrap = h('div', { class: 'cut-pick', role: 'group', 'aria-label': aria });
      var btns = items.map(function (it) {
        var b = h('button', { type: 'button', 'aria-pressed': 'false', html: hsub(it.label), title: it.title || null });
        b.addEventListener('click', function () { onPick(it.value); });
        wrap.appendChild(b);
        return b;
      });
      return { el: wrap, btns: btns, set: function (v) { btns.forEach(function (b, i) { b.setAttribute('aria-pressed', String(items[i].value === v)); }); } };
    }
    var presetPick = pickGroup(PRESETS.map(function (p) { return { value: p.id, label: p.label }; }), function (v) { st.preset = v; hover = null; update(false); }, 'Update Δ');
    var redraw = h('button', { type: 'button', class: 'cut-util', title: 'Draw new random factors (same seed family)', text: '↻ redraw' });
    redraw.addEventListener('click', function () { st.draw++; update(false); });
    presetPick.el.appendChild(redraw);
    controls.appendChild(h('div', { class: 'cut-group cut-g-preset' }, [
      h('div', { class: 'cut-gl' }, [h('span', { class: 'cut-tag', html: 'Update Δ ∈ ℝ<sup>16×16</sup>' }), h('span', { class: 'cut-tag', text: 'Gaussian factors · seeded' })]),
      presetPick.el
    ]));

    var splitPick = pickGroup(SPLITS.map(function (s, i) { return { value: i, label: s.label }; }), function (v) {
      st.split = v; var sp = SPLITS[v];
      st.rhoU = Math.min(sp.m1, sp.m2); st.rhoV = Math.min(sp.n1, sp.n2);
      st.k = Math.min(st.k, splitInfo(sp).kmax);
      hover = null; update(true);
    }, 'Split of R^16');
    var splitSub = h('div', { class: 'cut-sub' });
    controls.appendChild(h('div', { class: 'cut-group' }, [
      h('div', { class: 'cut-gl' }, [h('span', { class: 'cut-tag', html: hsub('Split of ℝ¹⁶') }), h('span', { class: 'cut-tag', text: 'out | in' })]),
      splitPick.el, splitSub
    ]));

    function mkSlider(id, label, nm, onInput) {
      var inp = h('input', { type: 'range', id: id, min: 1, max: 4, step: 1, value: 4 });
      var out = h('output', { for: id });
      var wrap = h('div', { class: 'cut-sl' }, [h('div', { class: 'cut-sl-top' }, [h('label', { for: id, html: label + '<span class="nm">' + nm + '</span>' }), out]), inp]);
      inp.addEventListener('input', function () { onInput(+inp.value); });
      return { el: wrap, input: inp, out: out };
    }
    var uid = 'cut' + Math.random().toString(36).slice(2, 7);
    var slU = mkSlider(uid + '-ru', 'ρ<sub>u</sub>', 'of u', function (v) { st.rhoU = v; update(false); });
    var slV = mkSlider(uid + '-rv', 'ρ<sub>v</sub>', 'of v', function (v) { st.rhoV = v; update(false); });
    var rhoNote = h('div', { class: 'cut-sub' });
    controls.appendChild(h('div', { class: 'cut-group' }, [
      h('div', { class: 'cut-gl' }, [h('span', { class: 'cut-tag', text: 'Schmidt ranks of u, v' }), h('span', { class: 'cut-tag', html: lc('Thm II.11(c)') })]),
      h('div', { class: 'cut-sliders' }, [slU.el, slV.el]), rhoNote
    ]));

    var viewPick = pickGroup([{ value: 1, label: '② ℛΔ' }, { value: 2, label: '③ ℛ(ΔΠᵀ)' }], function (v) { st.view = v; hover = null; viewPick.set(v); replay.textContent = '▶ replay ' + (v === 1 ? 'ℛ' : 'ℛ∘Π'); renderMatrix(true); renderCards(); }, 'Reshape shown');
    var replay = h('button', { type: 'button', class: 'cut-util', text: '▶ replay' });
    replay.addEventListener('click', function () { renderMatrix(true); });
    viewPick.el.appendChild(replay);
    controls.appendChild(h('div', { class: 'cut-group' }, [
      h('div', { class: 'cut-gl' }, [h('span', { class: 'cut-tag', text: 'Rearrange Δ' }), h('span', { class: 'cut-tag', text: 'a permutation' })]),
      viewPick.el,
      h('div', { class: 'cut-hint', text: 'Hover a cell, or focus the heatmap and use the arrow keys, to see which block (or comb) of Δ becomes its row.' })
    ]));

    var desc = h('div', { class: 'cut-desc', 'aria-live': 'polite' });
    stage.appendChild(desc);
    var verdict = h('div', { class: 'cut-verdict' });
    stage.appendChild(verdict);
    var cost = h('div', { class: 'cut-cost' });
    stage.appendChild(cost);

    /* top row: matrices + cut cards */
    var matTitle = h('h4', { text: 'One tensor, two matrices' });
    var matTag = h('span', { class: 'cut-tag' });
    var matWrap = h('div', { class: 'cut-mat' });
    var matRead = h('div', { class: 'cut-read' });
    var matSide = h('div', { class: 'cut-side' }, [matRead]);
    stage.appendChild(h('div', { class: 'cut-panel' }, [h('div', { class: 'cut-ph' }, [matTitle, matTag]), h('div', { class: 'cut-matgrid' }, [matWrap, matSide])]));
    var cardsWrap = h('div', { class: 'cut-cards' });
    stage.appendChild(cardsWrap);
    var cards = CUTS.map(function (C, c) {
      var rk = h('span', { class: 'cut-rk' });
      var dg = document.createElementNS(SVGNS, 'svg');
      dg.setAttribute('class', 'dg'); dg.setAttribute('viewBox', '0 -5 112 114'); dg.setAttribute('role', 'img');
      var bars = h('div', { class: 'cut-bars' });
      var foot = h('div', { class: 'cut-cf' });
      var card = h('div', { class: 'cut-card' }, [
        h('div', { class: 'cut-ch' }, [h('span', { class: 'cut-cn', html: '<span class="b">' + C.n + '</span>' + C.name }), rk]),
        h('div', { class: 'cut-cw' }, [h('span', { class: 'cut-cl', html: hsub(C.legs) }), h('span', { html: ' · ' + hsub(C.who) })]),
        h('div', { class: 'cut-cb' }, [dg, bars]),
        foot
      ]);
      cardsWrap.appendChild(card);
      return { card: card, rk: rk, dg: dg, bars: bars, foot: foot };
    });

    /* bottom row: image comparison */
    var bottom = h('div', { class: 'cut-bottom' });
    stage.appendChild(bottom);
    var planeWrap = h('div', { class: 'cut-plane' });
    var slR = mkSlider(uid + '-r', 'r', 'LoRA rank', function (v) { st.r = v; slR.out.textContent = String(v); renderPlane(); renderPlaneVerdict(); });
    var slK = mkSlider(uid + '-k', 'k', 'Kronecker terms', function (v) { st.k = v; slK.out.textContent = String(v); renderPlane(); renderPlaneVerdict(); });
    slR.input.max = N; slR.input.value = st.r;
    var planeHead = h('div', { class: 'cut-ph' }, [h('h4', { text: 'Which image contains which?' }), h('span', { class: 'cut-tag', html: lc('Thm II.11(b, c, d)') })]);
    bottom.appendChild(h('div', { class: 'cut-panel' }, [planeHead, planeWrap, h('div', { class: 'cut-sliders', style: 'margin-top:.35rem' }, [slR.el, slK.el])]));
    var pv = h('div', { class: 'cut-pv' });
    bottom.appendChild(h('div', { class: 'cut-panel' }, [pv]));

    var legend = h('div', { class: 'cut-legend' });
    stage.appendChild(legend);
    var foot = h('p', { class: 'cut-foot' });
    stage.appendChild(foot);
    var live = h('div', { class: 'sr-only', 'aria-live': 'polite' });
    stage.appendChild(live);

    /* ---------- helpers ---------- */
    function S(tag, attrs, parent) {
      var e = document.createElementNS(SVGNS, tag);
      for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }
    /* SVG text; subscript / superscript digit runs become shifted tspans in the text's own font */
    function T(parent, x, y, str, attrs) {
      var t = S('text', Object.assign({ x: x, y: y }, attrs || {}), parent), parts = String(str).split(/([₀-₉]+|[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/);
      if (parts.length === 1) { t.textContent = str; return t; }
      parts.forEach(function (p, i) {
        if (!p) return;
        if (i % 2 === 0) { t.appendChild(document.createTextNode(p)); return; }
        var isSub = /[₀-₉]/.test(p);
        S('tspan', { 'baseline-shift': isSub ? '-0.3em' : '0.45em', style: 'font-size:' + (isSub ? '72%' : '78%') }, t).textContent =
          isSub ? p.replace(/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 8320); }) : p.replace(/./g, function (c) { return SUPR[c]; });
      });
      return t;
    }
    function presetName(i) { return PRESETS[i].name; }
    function loraSub(r) { return 'LoRA<sub>' + r + '</sub>'; }
    function kSub(k) { return '<i>K</i><sub>≤' + k + '</sub>'; }

    /* ---------- text panels ---------- */
    function renderControls() {
      var sp = data.sp, info = data.info;
      presetPick.set(st.preset);
      splitPick.set(st.split);
      viewPick.set(st.view);
      replay.textContent = '▶ replay ' + (st.view === 1 ? 'ℛ' : 'ℛ∘Π');
      splitSub.innerHTML = hsub('out ℝ' + sup(sp.m1) + '⊗ℝ' + sup(sp.m2) + ' · in ℝ' + sup(sp.n1) + '⊗ℝ' + sup(sp.n2) + ' · κ₁ = <b>' + info.kappa1 + '</b>' +
        '<br>every Δ: rk⊗ ≤ min(m₁n₁, m₂n₂) = <b>' + info.kmax + '</b>');
      kappaSpan.textContent = '= ' + info.kappa1;
      var mu = Math.min(sp.m1, sp.m2), mv = Math.min(sp.n1, sp.n2), on = st.preset === 'uv';
      slU.input.max = mu; slV.input.max = mv;
      st.rhoU = Math.min(st.rhoU, mu); st.rhoV = Math.min(st.rhoV, mv);
      slU.input.value = st.rhoU; slV.input.value = st.rhoV;
      slU.out.textContent = st.rhoU + ' / ' + mu; slV.out.textContent = st.rhoV + ' / ' + mv;
      [slU, slV].forEach(function (s) { s.input.disabled = !on; s.el.classList.toggle('off', !on); });
      rhoNote.innerHTML = on ? 'rk⊗(uvᵀ) = ρ<sub>u</sub>ρ<sub>v</sub> = <b>' + (st.rhoU * st.rhoV) + '</b> · measured <b>' + data.cur.cuts[1].rank + '</b>' : 'active for Δ = uvᵀ';
      slK.input.max = info.kmax;
      st.k = Math.min(st.k, info.kmax);
      slR.input.value = st.r; slK.input.value = st.k;
      slR.out.textContent = String(st.r);
      slK.out.textContent = String(st.k);
    }

    function descHTML() {
      var sp = data.sp, info = data.info, m1 = sp.m1, m2 = sp.m2, n1 = sp.n1, n2 = sp.n2, c = data.cur.cuts;
      var M = function (tex) { return '\\(' + tex + '\\)'; };
      var R = function (a, b) { return M('\\mathbb R^{' + a + '\\times ' + b + '}'); };
      var who = function (t) { return ' <span class="who">' + hsub(t) + '</span>'; };
      var a = Math.min(m1, n1), b = Math.min(m2, n2), rL = Math.min(m1, n1);
      switch (st.preset) {
        case 'uv': return M('\\Delta=uv^{\\top}') + ' with ' + M('u=\\sum_{i\\le\\rho_u}x_i\\otimes y_i') + ' and ' + M('v=\\sum_{j\\le\\rho_v}z_j\\otimes w_j') +
          ': it reads one input direction ' + M('v') + ' and writes one output direction ' + M('u') + '. Its Kronecker rank is ' + M('\\rho_u\\rho_v') + ' (Thm II.11(c)).' +
          who('LoRA₁ · |M| = m + n = ' + info.cost[0]);
        case 'id': return (info.matched
            ? M('\\Delta=I_{16}=I_{' + m1 + '}\\otimes I_{' + m2 + '}') + ', a target rather than an adapter. It is one Kronecker product of full rank, ' +
              M('\\operatorname{rk}(C\\otimes D)=\\operatorname{rk}C\\cdot\\operatorname{rk}D=16') + ' (Thm II.11(b)), so KronA reaches it with ' + M('C=I,\\ D=I') + '.'
            : M('\\Delta=I_{16}') + ', a target rather than an adapter. On this crossed split it is not one Kronecker product: ' + M('\\operatorname{rk}_\\otimes I_{16}=' + c[1].rank) + '.') +
          who('No LoRA below rank 16 contains it.');
        case 'cd': return M('\\Delta=C\\otimes D') + ' with ' + M('C\\in') + R(m1, n1) + ', ' + M('D\\in') + R(m2, n2) + ': Kronecker rank 1 and rank ' +
          M('\\operatorname{rk}C\\cdot\\operatorname{rk}D=' + a + '\\cdot' + b + '=' + (a * b)) + ' (Thm II.11(b)).' +
          who('KronA · |M| = m₁n₁ + m₂n₂ = ' + info.cost[1]);
        case 'k2': return M('\\Delta=C_1\\otimes D_1+C_2\\otimes D_2') + ', a two-term Kronecker sum. Such sums are the image of ' + M('\\mathcal R^{-1}_*\\mathrm{LoRA}_2') + ' (Prop II.13).' +
          who('|M| = 2(m₁n₁ + m₂n₂) = ' + (2 * info.cost[1]));
        case 'lora2': return M('\\Delta=BA') + ' with ' + M('B\\in') + R(16, 2) + ', ' + M('A\\in') + R(2, 16) + ': ordinary rank 2.' +
          who('LoRA₂ · |M| = 2(m + n) = ' + (2 * info.cost[0]));
        case 'loha': return M('\\Delta=(B_1A_1)\\odot(B_2A_2)') + ' with inner ranks ' + M('r_1=r_2=2') + ': rank ' + M('\\le r_1r_2=4') + ' (Prop II.12).' +
          who('LoHa · |M| = (r₁ + r₂)(m + n) = ' + (4 * info.cost[0]));
        case 'hira': return M('\\Delta=W_0\\odot(BA)') + ' with ' + M('r=1') + ' and a Gaussian stand-in for the pretrained ' + M('W_0') + ': rank ' +
          M('\\le r\\cdot\\operatorname{rk}W_0=16') + ' (Prop II.9(a)).' +
          who('HiRA · |M| = r(m + n) = ' + info.cost[0]);
        case 'lokr': return M('\\Delta=C\\otimes(BA)') + ' with ' + M('C\\in') + R(m1, n1) + ' and inner rank ' + M("r'=1") + ': rank ' + M("\\le\\min(m_1,n_1)\\,r'=" + rL) +
          ', so ' + M('\\mathrm{Im}\\,\\mathrm{LoKr}_1\\subseteq\\mathrm{Im}\\,\\mathrm{LoRA}_{' + rL + '}') + (rL > 1 ? ', and for ' + M('r<' + rL) + ' the two images are incomparable' : '') + ' (Thm II.11(e), which needs ' + M('\\kappa_1=' + info.kappa1 + '>1') + ').' +
          who('LoKr · |M| = m₁n₁ + r′(m₂ + n₂) = ' + (m1 * n1 + m2 + n2));
      }
      return '';
    }
    function thmFor(pid) {
      return { uv: 'Thm II.11(c)', id: 'Thm II.11(a, b)', cd: 'Thm II.11(a, b)', k2: 'Thm II.11(a)', lora2: 'Thm II.11(a)', loha: 'Thm II.11(a) · Prop II.12', hira: 'Thm II.11(a) · Prop II.9(a)', lokr: 'Thm II.11(e)' }[pid];
    }
    function renderDesc() {
      var key = st.preset + '|' + st.split + '|' + (st.preset === 'id' ? data.cur.cuts[1].rank : '');
      if (key === lastDescKey) return;
      lastDescKey = key;
      desc.innerHTML = descHTML();
      A.typeset(desc);
    }
    function renderVerdict() {
      var c = data.cur.cuts, r0 = c[0].rank, k0 = c[1].rank, info = data.info;
      var html = '';
      html += '<span class="cut-chip y">✓ ' + loraSub(r0) + ' ∋ Δ</span>';
      if (r0 > 1) html += '<span class="cut-chip n">✗ LoRA<sub>r</sub> ∌ Δ for r &lt; ' + r0 + '</span>';
      html += '<span class="cut-chip y">✓ ' + (k0 > 1 ? k0 + '-term Kronecker sums' : 'Kronecker products C⊗D') + ' ∋ Δ</span>';
      if (k0 > 1) html += '<span class="cut-chip n">✗ no sum of k &lt; ' + k0 + ' terms ∋ Δ</span>';
      html += '<span class="cut-thm">' + lc(thmFor(st.preset)) + '</span>';
      verdict.innerHTML = hsub(html);
      var cL = r0 * info.cost[0], cK = k0 * info.cost[1];
      cost.innerHTML = 'Cheapest exact fit per family · <span class="c1">LoRA<sub>' + r0 + '</sub></span> <b>' + intStr(cL) + '</b> params · ' +
        '<span class="c2">' + (k0 > 1 ? k0 + '-term Kronecker sum' : 'one C⊗D') + '</span> <b>' + intStr(cK) + '</b> params · full ΔW <b>256</b>' +
        (cL !== cK ? ' · ' + (cL < cK ? 'LoRA' : 'the Kronecker sum') + ' is <b>' + (+(Math.max(cL, cK) / Math.min(cL, cK)).toFixed(2)) + '×</b> cheaper' : ' · a tie');
      live.textContent = presetName(data.idx) + ' on split ' + data.sp.label + ': rank ' + r0 + ', Kronecker rank ' + k0 + ', shuffled-cut rank ' + c[2].rank + '.';
    }

    /* ---------- matrix panel: Delta and its rearrangement ---------- */
    var matSel = null;
    function renderMatrix(animate) {
      var C = col(), sp = data.sp, cut = data.cuts[st.view], X = data.cur.X, mA = data.cur.maxAbs || 1;
      var W = Math.max(260, Math.floor(matWrap.getBoundingClientRect().width || 520)), rr = cut.rows, rc = cut.cols, square = rr === rc;
      var titleH = 30, L = {};
      if (square && W >= 300) {
        var gap = W >= 520 ? 76 : 50, Sz = Math.min(Math.floor((W - gap) / 2), 250), x0 = Math.floor((W - 2 * Sz - gap) / 2);
        L.side = true;
        L.l = { x: x0, y: titleH, cw: Sz / N, ch: Sz / N, w: Sz, h: Sz };
        L.r = { x: x0 + Sz + gap, y: titleH, cw: Sz / rc, ch: Sz / rr, w: Sz, h: Sz };
        L.H = titleH + Sz + 4;
        L.arrow = [x0 + Sz + 8, titleH + Sz / 2, x0 + Sz + gap - 8, titleH + Sz / 2];
      } else {
        var S1 = Math.min(230, Math.floor(W * 0.6)), xl = Math.floor((W - S1) / 2);
        L.l = { x: xl, y: titleH, cw: S1 / N, ch: S1 / N, w: S1, h: S1 };
        var cell = square ? S1 / N : Math.min(W / rc, 16), rw = cell * rc, rh = cell * rr;
        var ry = titleH + S1 + 44 + titleH - 6;
        L.r = { x: Math.floor((W - rw) / 2), y: ry, cw: cell, ch: cell, w: rw, h: rh };
        L.H = ry + rh + 4;
        L.arrow = [W / 2, titleH + S1 + 6, W / 2, titleH + S1 + 40];
      }
      matWrap.innerHTML = '';
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + L.H, width: W, height: L.H, role: 'img', 'aria-label': 'Heatmap of Δ and of its rearrangement across cut ' + (st.view + 1) + ' (' + rr + ' by ' + rc + '). Arrow keys move a cursor through the rearranged matrix and show where each entry came from.' }, matWrap);
      var defs = S('defs', {}, svg);
      var mk = S('marker', { id: uid + '-ah', viewBox: '0 0 10 10', refX: 8.5, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
      S('path', { d: 'M1,1.5 L8.5,5 L1,8.5', fill: 'none', stroke: C.ink2, 'stroke-width': 1.3 }, mk);

      var ipPos = d3.interpolateRgb(C.paper, C.pos), ipNeg = d3.interpolateRgb(C.paper, C.neg);
      function fill(x) { var a = Math.abs(x) / mA; if (a < 1e-12) return C.paper; var t = 0.3 + 0.7 * Math.pow(a, 0.5); return x > 0 ? ipPos(t) : ipNeg(t); }
      var gapL = L.l.cw >= 7 ? 0.6 : 0, gapR = Math.min(L.r.cw, L.r.ch) >= 7 ? 0.6 : 0;

      /* titles */
      T(svg, L.l.x, titleH - 10, 'Δ', { class: 'display', style: 'font-size:17px;fill:' + C.ink });
      var roomy = L.l.w >= 200, rLegs = st.view === 1 ? ['U₁V₁', 'U₂V₂'] : ['U₁V₂', 'U₂V₁'];
      var rTitleW = st.view === 1 ? 32 : 62, rSpace = W - L.r.x - rTitleW;
      T(svg, L.l.x + 18, titleH - 11, roomy ? 'rows U₁U₂ · cols V₁V₂' : 'U₁U₂ | V₁V₂', { style: 'font-family:var(--f-ui);font-weight:500;font-size:12px;fill:' + C.ink2 });
      T(svg, L.r.x, L.r.y - 10, CUTS[st.view].mat, { class: 'display', style: 'font-size:17px;fill:' + C.ink });
      T(svg, L.r.x + rTitleW, L.r.y - 11, rSpace >= 200 ? 'rows ' + rLegs[0] + ' · cols ' + rLegs[1] + ' · ' + rr + '×' + rc : rLegs[0] + ' | ' + rLegs[1], { style: 'font-family:var(--f-ui);font-weight:500;font-size:12px;fill:' + C.ink2 });

      /* frames + cells */
      S('rect', { x: L.l.x - 0.5, y: L.l.y - 0.5, width: L.l.w + 1, height: L.l.h + 1, fill: C.paper, stroke: C.rule }, svg);
      S('rect', { x: L.r.x - 0.5, y: L.r.y - 0.5, width: L.r.w + 1, height: L.r.h + 1, fill: C.paper, stroke: C.rule }, svg);
      var gL = S('g', {}, svg), gR = S('g', {}, svg);
      var cellsL = [], cellsR = [];
      for (var e = 0; e < N * N; e++) {
        var i = Math.floor(e / N), j = e % N, f = fill(X[e]);
        cellsL.push(S('rect', { x: L.l.x + j * L.l.cw, y: L.l.y + i * L.l.ch, width: Math.max(0.5, L.l.cw - gapL), height: Math.max(0.5, L.l.ch - gapL), fill: f }, gL));
        cellsR.push(S('rect', { x: L.r.x + cut.col[e] * L.r.cw, y: L.r.y + cut.row[e] * L.r.ch, width: Math.max(0.5, L.r.cw - gapR), height: Math.max(0.5, L.r.ch - gapR), fill: f }, gR));
      }
      /* block structure */
      var gLines = S('g', { stroke: C.ink3, 'stroke-width': 0.8, 'stroke-opacity': 0.55 }, svg);
      var a;
      for (a = sp.m2; a < N; a += sp.m2) S('line', { x1: L.l.x, x2: L.l.x + L.l.w, y1: L.l.y + a * L.l.ch, y2: L.l.y + a * L.l.ch }, gLines);
      for (a = sp.n2; a < N; a += sp.n2) S('line', { y1: L.l.y, y2: L.l.y + L.l.h, x1: L.l.x + a * L.l.cw, x2: L.l.x + a * L.l.cw }, gLines);
      var rg = st.view === 1 ? sp.n1 : sp.n2, cg = st.view === 1 ? sp.n2 : sp.n1;
      for (a = rg; a < rr; a += rg) S('line', { x1: L.r.x, x2: L.r.x + L.r.w, y1: L.r.y + a * L.r.ch, y2: L.r.y + a * L.r.ch }, gLines);
      for (a = cg; a < rc; a += cg) S('line', { y1: L.r.y, y2: L.r.y + L.r.h, x1: L.r.x + a * L.r.cw, x2: L.r.x + a * L.r.cw }, gLines);

      /* arrow */
      var ar = L.arrow;
      S('line', { x1: ar[0], y1: ar[1], x2: ar[2], y2: ar[3], stroke: C.ink2, 'stroke-width': 1.25, 'marker-end': 'url(#' + uid + '-ah)' }, svg);
      var lab = st.view === 1 ? 'ℛ' : 'ℛ∘Π';
      if (L.side) {
        T(svg, (ar[0] + ar[2]) / 2, ar[1] - 8, lab, { class: 'display', 'text-anchor': 'middle', style: 'font-size:15px;font-style:italic;fill:' + C.ink });
        if (ar[2] - ar[0] > 50) T(svg, (ar[0] + ar[2]) / 2, ar[1] + 17, 'bend wires', { 'text-anchor': 'middle', style: 'font-family:var(--f-ui);font-weight:500;font-size:12px;fill:' + C.ink2 });
      } else {
        T(svg, ar[0] + 10, (ar[1] + ar[3]) / 2 + 4, lab + ' · bend wires', { style: 'font-family:var(--f-ui);font-weight:500;font-size:12px;fill:' + C.ink2 });
      }

      /* highlight layer */
      var gHi = S('g', { fill: 'none', 'pointer-events': 'none' }, svg);
      matSel = { svg: svg, L: L, cut: cut, cellsL: cellsL, cellsR: cellsR, gHi: gHi, C: C };
      drawHighlight();

      /* hover */
      svg.addEventListener('pointermove', function (ev) {
        if (animating) return;
        var p = d3.pointer(ev, svg), hit = null;
        var inL = p[0] >= L.l.x && p[0] < L.l.x + L.l.w && p[1] >= L.l.y && p[1] < L.l.y + L.l.h;
        var inR = p[0] >= L.r.x && p[0] < L.r.x + L.r.w && p[1] >= L.r.y && p[1] < L.r.y + L.r.h;
        if (inL) { var ii = Math.floor((p[1] - L.l.y) / L.l.ch), jj = Math.floor((p[0] - L.l.x) / L.l.cw); hit = ii * N + jj; }
        else if (inR) { var ri = Math.floor((p[1] - L.r.y) / L.r.ch), ci = Math.floor((p[0] - L.r.x) / L.r.cw); hit = cut.inv[Math.min(rr - 1, ri) * rc + Math.min(rc - 1, ci)]; }
        if (hit !== hover) { hover = hit; drawHighlight(); }
      });
      svg.addEventListener('pointerleave', function () { if (hover != null && !svg.__kbd) { hover = null; drawHighlight(); } });
      /* keyboard: arrows walk the rearranged matrix cell by cell, Escape clears */
      svg.setAttribute('tabindex', '0');
      svg.setAttribute('aria-keyshortcuts', 'ArrowUp ArrowDown ArrowLeft ArrowRight Escape');
      svg.addEventListener('keydown', function (ev) {
        var keys = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
        if (ev.key === 'Escape') { svg.__kbd = false; hover = null; drawHighlight(); return; }
        if (!keys[ev.key] || animating) return;
        ev.preventDefault();
        var ri, ci;
        if (hover == null) { ri = Math.min(1, rr - 1); ci = 0; }
        else { ri = cut.row[hover] + keys[ev.key][0]; ci = cut.col[hover] + keys[ev.key][1]; }
        ri = (ri + rr) % rr; ci = (ci + rc) % rc;
        svg.__kbd = true; hover = cut.inv[ri * rc + ci]; drawHighlight();
      });
      svg.addEventListener('blur', function () { if (svg.__kbd) { svg.__kbd = false; hover = null; drawHighlight(); } });

      /* the rearrangement */
      var reduce = A.reducedMotion();
      if (animate && !reduce) {
        animating = true;
        var sel = d3.selectAll(cellsR).data(d3.range(N * N));
        sel.interrupt()
          .attr('x', function (e) { return L.l.x + (e % N) * L.l.cw; })
          .attr('y', function (e) { return L.l.y + Math.floor(e / N) * L.l.ch; })
          .attr('width', Math.max(0.5, L.l.cw - gapL)).attr('height', Math.max(0.5, L.l.ch - gapL));
        gHi.style.opacity = 0;
        var step = Math.min(70, 900 / rr), last = 0;
        sel.transition()
          .delay(function (e) { var d = 150 + cut.row[e] * step; last = Math.max(last, d); return d; })
          .duration(760).ease(d3.easeCubicInOut)
          .attr('x', function (e) { return L.r.x + cut.col[e] * L.r.cw; })
          .attr('y', function (e) { return L.r.y + cut.row[e] * L.r.ch; })
          .attr('width', Math.max(0.5, L.r.cw - gapR)).attr('height', Math.max(0.5, L.r.ch - gapR));
        /* settle: snap every cell to its target even if animation frames were throttled (background tab, headless) */
        setTimeout(function () {
          if (!matSel || matSel.gHi !== gHi) return;
          sel.interrupt()
            .attr('x', function (e) { return L.r.x + cut.col[e] * L.r.cw; })
            .attr('y', function (e) { return L.r.y + cut.row[e] * L.r.ch; })
            .attr('width', Math.max(0.5, L.r.cw - gapR)).attr('height', Math.max(0.5, L.r.ch - gapR));
          animating = false;
          gHi.style.transition = 'opacity .3s'; gHi.style.opacity = 1;
        }, last + 820);
      } else {
        animating = false;
      }
      matTag.textContent = 'split ' + sp.label + ' · ' + (st.view === 1 ? 'cut ②' : 'cut ③');
    }

    function sourceRects(rowSet, L) {
      /* union of the Delta cells e with cut.row[e] in the target row, as merged rectangles */
      var cut = matSel.cut, byCol = {}, e, runs = [];
      for (e = 0; e < N * N; e++) if (cut.row[e] === rowSet) { var j = e % N; (byCol[j] = byCol[j] || []).push(Math.floor(e / N)); }
      Object.keys(byCol).map(Number).sort(function (a, b) { return a - b; }).forEach(function (j) {
        var is = byCol[j].sort(function (a, b) { return a - b; }), s0 = is[0], prev = is[0];
        for (var t = 1; t <= is.length; t++) {
          if (t < is.length && is[t] === prev + 1) { prev = is[t]; continue; }
          runs.push({ j0: j, j1: j, i0: s0, i1: prev });
          if (t < is.length) { s0 = is[t]; prev = is[t]; }
        }
      });
      var merged = [];
      runs.forEach(function (r) {
        var m = merged.length ? merged[merged.length - 1] : null;
        if (m && m.j1 === r.j0 - 1 && m.i0 === r.i0 && m.i1 === r.i1) m.j1 = r.j1; else merged.push({ j0: r.j0, j1: r.j1, i0: r.i0, i1: r.i1 });
      });
      return merged.map(function (m) { return { x: L.l.x + m.j0 * L.l.cw, y: L.l.y + m.i0 * L.l.ch, w: (m.j1 - m.j0 + 1) * L.l.cw, h: (m.i1 - m.i0 + 1) * L.l.ch }; });
    }
    function drawHighlight() {
      if (!matSel) return;
      var L = matSel.L, cut = matSel.cut, C = matSel.C, g = matSel.gHi, sp = data.sp, X = data.cur.X;
      while (g.firstChild) g.removeChild(g.firstChild);
      var row = hover != null ? cut.row[hover] : Math.min(1, cut.rows - 1);
      var rects = sourceRects(row, L);
      function box(x, y, w, hh, sw) {
        S('rect', { x: x, y: y, width: w, height: hh, stroke: C.paper, 'stroke-width': sw + 2.2, 'stroke-opacity': 0.9 }, g);
        S('rect', { x: x, y: y, width: w, height: hh, stroke: C.ochre, 'stroke-width': sw }, g);
      }
      rects.forEach(function (r) { box(r.x, r.y, r.w, r.h, 1.6); });
      box(L.r.x, L.r.y + row * L.r.ch, L.r.w, L.r.ch, 1.6);
      var dim = hover != null;
      for (var e = 0; e < N * N; e++) {
        var inRow = cut.row[e] === row;
        matSel.cellsL[e].setAttribute('opacity', dim && !inRow ? 0.3 : 1);
        matSel.cellsR[e].setAttribute('opacity', dim && !inRow ? 0.3 : 1);
      }
      var u1 = Math.floor(row / (st.view === 1 ? sp.n1 : sp.n2)), w2 = row % (st.view === 1 ? sp.n1 : sp.n2);
      if (hover != null) {
        var e0 = hover, i = Math.floor(e0 / N), j = e0 % N;
        var cw = Math.min(L.l.cw, L.l.ch);
        S('rect', { x: L.l.x + j * L.l.cw + 0.5, y: L.l.y + i * L.l.ch + 0.5, width: Math.max(1, cw - 1), height: Math.max(1, L.l.ch - 1), stroke: C.ink, 'stroke-width': 1.4 }, g);
        S('rect', { x: L.r.x + cut.col[e0] * L.r.cw + 0.5, y: L.r.y + cut.row[e0] * L.r.ch + 0.5, width: Math.max(1, L.r.cw - 1), height: Math.max(1, L.r.ch - 1), stroke: C.ink, 'stroke-width': 1.4 }, g);
        var a1 = Math.floor(i / sp.m2), a2 = i % sp.m2, b1 = Math.floor(j / sp.n2), b2 = j % sp.n2;
        var x = X[e0], xs = (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3);
        matRead.innerHTML = hsub('Δ[(u₁,u₂),(v₁,v₂)] = Δ[(' + a1 + ',' + a2 + '),(' + b1 + ',' + b2 + ')] = <b>' + xs + '</b><br>→ row ' + cut.row[e0] + ', column ' + cut.col[e0] + ' of ' + CUTS[st.view].mat + ' (row = ' + (st.view === 1 ? '(u₁,v₁)' : '(u₁,v₂)') + ')');
      } else if (st.view === 1) {
        matRead.innerHTML = hsub('Block <b>(u₁,v₁) = (' + u1 + ',' + w2 + ')</b> of Δ, ' + (sp.m2 === 8 || sp.m2 === 11 || sp.m2 === 18 ? 'an ' : 'a ') + sp.m2 + '×' + sp.n2 + ' matrix, becomes <b>row ' + row + '</b> of ℛΔ. Every block of C⊗D is a multiple of D, so ℛ(C⊗D) = vec C · vec Dᵀ has rank 1, and a single Kronecker product is a rank-one update across this cut.');
      } else {
        matRead.innerHTML = hsub('The cells of Δ with <b>u₁ = ' + u1 + '</b> and <b>v₂ = ' + w2 + '</b> (a comb, every ' + ordinal(sp.n2) + ' column of one block row) become <b>row ' + row + '</b> of ℛ(ΔΠᵀ): the Kronecker cut after swapping the input legs.');
      }
    }

    /* ---------- cut cards ---------- */
    function drawDiagram(svg, c, rank, slots, C) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var cc = C.cut[c], P = { U1: [26, 14], U2: [86, 14], V1: [26, 90], V2: [86, 90] };
      var lab = { style: 'font-family:var(--f-body);font-style:italic;font-size:13px;fill:' + C.ink2, 'text-anchor': 'middle' };
      legLabel(26, 10, 'U', 1); legLabel(86, 10, 'U', 2); legLabel(26, 103, 'V', 1); legLabel(86, 103, 'V', 2);
      function legLabel(x, y, a, k) { var t = T(svg, x, y, a, lab); var s = S('tspan', { 'baseline-shift': '-0.3em', style: 'font-size:72%;font-style:normal' }, t); s.textContent = String(k); }
      var wire = { fill: 'none', stroke: C.ink2, 'stroke-width': 1.25 };
      var bw = 1.25 + 3 * (rank - 1) / Math.max(1, N - 1);
      function boxAt(cx, cy, name) {
        S('rect', { x: cx - 11, y: cy - 9, width: 22, height: 18, rx: 2.5, fill: C.paper2, stroke: C.ink2, 'stroke-width': 1.25 }, svg);
        T(svg, cx, cy + 4.5, name, { 'text-anchor': 'middle', class: 'display', style: 'font-size:13px;font-style:italic;fill:' + C.ink });
      }
      if (c === 0) {
        S('path', Object.assign({ d: 'M26,16 C26,26 50,24 51,30' }, wire), svg);
        S('path', Object.assign({ d: 'M86,16 C86,26 62,24 61,30' }, wire), svg);
        S('path', Object.assign({ d: 'M51,74 C50,80 26,78 26,88' }, wire), svg);
        S('path', Object.assign({ d: 'M61,74 C62,80 86,78 86,88' }, wire), svg);
        S('line', { x1: 56, y1: 46, x2: 56, y2: 58, stroke: cc, 'stroke-width': bw }, svg);
        boxAt(56, 38, CUTS[0].boxes[0]); boxAt(56, 66, CUTS[0].boxes[1]);
        T(svg, 65, 56.5, String(rank), { style: 'font-family:var(--f-mono);font-size:12px;font-weight:500;fill:' + cc });
      } else {
        S('line', { x1: 26, y1: 16, x2: 26, y2: 44, stroke: C.ink2, 'stroke-width': 1.25 }, svg);
        S('line', { x1: 86, y1: 16, x2: 86, y2: 44, stroke: C.ink2, 'stroke-width': 1.25 }, svg);
        if (c === 1) {
          S('line', { x1: 26, y1: 60, x2: 26, y2: 88, stroke: C.ink2, 'stroke-width': 1.25 }, svg);
          S('line', { x1: 86, y1: 60, x2: 86, y2: 88, stroke: C.ink2, 'stroke-width': 1.25 }, svg);
        } else {
          S('path', Object.assign({ d: 'M86,60 C86,76 26,72 26,88' }, wire), svg);
          S('path', { d: 'M26,60 C26,76 86,72 86,88', fill: 'none', stroke: C.paper, 'stroke-width': 5 }, svg);
          S('path', Object.assign({ d: 'M26,60 C26,76 86,72 86,88' }, wire), svg);
        }
        S('line', { x1: 37, y1: 52, x2: 75, y2: 52, stroke: cc, 'stroke-width': bw }, svg);
        boxAt(26, 52, CUTS[c].boxes[0]); boxAt(86, 52, CUTS[c].boxes[1]);
        T(svg, 56, 45, String(rank), { 'text-anchor': 'middle', style: 'font-family:var(--f-mono);font-size:12px;font-weight:500;fill:' + cc });
      }
      svg.setAttribute('aria-label', 'String diagram: Δ factored through cut ' + (c + 1) + ' with a bond of dimension ' + rank + '.');
      void slots;
    }
    function drawBars(wrap, c, res, C) {
      var W = Math.max(130, Math.floor(wrap.getBoundingClientRect().width || 180)), H = 100, m = { l: 40, r: 6, t: 6, b: 18 };
      wrap.innerHTML = '';
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': 'Singular values across cut ' + (c + 1) + ', log scale: ' + res.rank + ' of ' + res.sv.length + ' exceed the rank threshold.' }, wrap);
      var y = function (s) { var l = Math.log10(Math.max(s, Math.pow(10, LOGMIN))); return m.t + (0 - l) / (0 - LOGMIN) * (H - m.t - m.b); };
      var x0 = m.l, x1 = W - m.r, n = res.sv.length, slot = (x1 - x0) / n, bw = Math.max(1.5, Math.min(slot * 0.68, 22));
      S('line', { x1: x0, x2: x1, y1: H - m.b, y2: H - m.b, stroke: C.rule }, svg);
      [[0, '1'], [-10, '10⁻¹⁰'], [LOGMIN, '10⁻¹⁷']].forEach(function (tk) {
        var yy = y(Math.pow(10, tk[0]));
        T(svg, x0 - 4, yy + 4, tk[1], { 'text-anchor': 'end', style: 'font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:' + C.ink2 });
        if (tk[0] === 0) S('line', { x1: x0, x2: x1, y1: yy, y2: yy, stroke: C.rule, 'stroke-dasharray': '1 3' }, svg);
      });
      S('line', { x1: x0, x2: x1, y1: y(TOL), y2: y(TOL), stroke: C.ink3, 'stroke-dasharray': '4 3', 'stroke-width': 1 }, svg);
      res.sv.forEach(function (s, i) {
        var xx = x0 + i * slot + (slot - bw) / 2, yy = y(s), on = s > TOL;
        var r = S('rect', { x: xx, y: Math.min(yy, H - m.b - 1), width: bw, height: Math.max(1, H - m.b - yy), fill: on ? C.cut[c] : C.ink3, 'fill-opacity': on ? 0.92 : 0.35 }, svg);
        var tt = S('title', {}, r); tt.textContent = 'σ' + sub(i + 1) + ' / ‖Δ‖_F = ' + svStr(s) + (on ? '' : '  (numerically zero)');
      });
      var t1 = T(svg, x0 + slot / 2, H - 4, '1', { 'text-anchor': 'middle', style: 'font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:' + C.ink2 });
      var tn = T(svg, x0 + (n - 0.5) * slot, H - 4, String(n), { 'text-anchor': 'middle', style: 'font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:' + C.ink2 });
      var tc = T(svg, (x0 + x1) / 2, H - 4, 'σᵢ / ‖Δ‖', { 'text-anchor': 'middle', style: 'font-family:var(--f-body);font-style:italic;font-size:12.5px;fill:' + C.ink2 });
      /* the axis caption shortens, then steps aside, when the first and last index labels leave it no room */
      try {
        var room = function () { var b = tc.getBBox(), b1 = t1.getBBox(), bn = tn.getBBox(); return b.x > b1.x + b1.width + 4 && b.x + b.width < bn.x - 4; };
        if (!room()) { tc.textContent = 'σᵢ'; if (!room()) tc.remove(); }
      } catch (e) { /* not laid out yet */ }
    }
    function renderCards() {
      var C = col(), info = data.info, res = data.cur.cuts;
      cards.forEach(function (k, c) {
        var r = res[c];
        k.card.style.setProperty('--cutc', C.cut[c]);
        k.card.classList.toggle('on', c === st.view);
        k.rk.innerHTML = '<b>' + r.rank + '</b>/ ' + r.sv.length;
        drawDiagram(k.dg, c, r.rank, r.sv.length, C);
        drawBars(k.bars, c, r, C);
        var shape = r.rows + '×' + r.cols, per = info.cost[c];
        var form = c === 0 ? 'Δ = BA' : c === 1 ? 'Δ = Σ Cᵢ⊗Dᵢ' : 'ΔΠᵀ = Σ Eᵢ⊗Fᵢ';
        var perStr = c === 0 ? '(m+n)' : c === 1 ? '(m₁n₁+m₂n₂)' : '(m₁n₂+m₂n₁)';
        k.foot.innerHTML = hsub(CUTS[c].mat + ' is ' + shape + ' · ' + form + ' with bond <b>' + r.rank + '</b> → ' + r.rank + '·' + perStr + ' = <b>' + intStr(r.rank * per) + '</b> params');
      });
    }

    /* ---------- the image plane ---------- */
    function renderPlane() {
      var C = col(), info = data.info, W = Math.max(260, Math.floor(planeWrap.getBoundingClientRect().width || 420)), H = Math.round(Math.max(210, Math.min(270, W * 0.56)));
      var m = { l: 42, r: 26, t: 24, b: 36 }, kmax = info.kmax;
      var x = d3.scaleLinear().domain([0, N + 0.7]).range([m.l, W - m.r]);
      var ytop = kmax + (kmax > 4 ? 0.7 : 0.4);
      var y = d3.scaleLinear().domain([0, ytop]).range([H - m.b, m.t]);
      planeWrap.innerHTML = '';
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': 'Plane of ordinary rank against Kronecker rank. LoRA with rank ' + st.r + ' is the strip of rank at most ' + st.r + '; sums of ' + st.k + ' Kronecker products are the strip of Kronecker rank at most ' + st.k + '.' }, planeWrap);
      /* strips */
      var xr = x(Math.min(st.r + 0.5, N + 0.7)), yk = y(Math.min(st.k + 0.5, ytop));
      S('rect', { x: m.l, y: m.t, width: xr - m.l, height: H - m.b - m.t, fill: C.tideSoft }, svg);
      S('rect', { x: m.l, y: yk, width: W - m.r - m.l, height: H - m.b - yk, fill: C.ochreSoft }, svg);
      S('line', { x1: xr, x2: xr, y1: m.t, y2: H - m.b, stroke: C.tide, 'stroke-width': 1.5 }, svg);
      S('line', { x1: m.l, x2: W - m.r, y1: yk, y2: yk, stroke: C.ochre, 'stroke-width': 1.5 }, svg);
      /* axes */
      S('rect', { x: m.l, y: m.t, width: W - m.r - m.l, height: H - m.b - m.t, fill: 'none', stroke: C.rule }, svg);
      var xt = [1, 4, 8, 12, 16], yt = kmax > 4 ? [1, 4, 8, 12, 16] : [1, 2, 3, 4];
      xt.forEach(function (v) { T(svg, x(v), H - m.b + 14, String(v), { 'text-anchor': 'middle', style: 'font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:' + C.ink2 }); S('line', { x1: x(v), x2: x(v), y1: H - m.b, y2: H - m.b + 3, stroke: C.ink3 }, svg); });
      yt.forEach(function (v) { T(svg, m.l - 5, y(v) + 4, String(v), { 'text-anchor': 'end', style: 'font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:' + C.ink2 }); S('line', { x1: m.l - 3, x2: m.l, y1: y(v), y2: y(v), stroke: C.ink3 }, svg); });
      T(svg, W - m.r, H - 4, 'rk Δ  (cut ①) →', { 'text-anchor': 'end', style: 'font-family:var(--f-ui);font-weight:500;font-size:12px;fill:' + C.ink2 });
      var yl = T(svg, 0, 0, 'rk⊗ Δ  (cut ②) →', { 'text-anchor': 'middle', style: 'font-family:var(--f-ui);font-weight:500;font-size:12px;fill:' + C.ink2 });
      yl.setAttribute('transform', 'translate(12,' + ((m.t + H - m.b) / 2) + ') rotate(-90)');
      /* kappa_1: annotated in the right margin, explained in the legend */
      var yk1 = y(info.kappa1);
      S('line', { x1: m.l, x2: W - m.r, y1: yk1, y2: yk1, stroke: C.ink3, 'stroke-dasharray': '3 3' }, svg);
      var kl = T(svg, W - m.r + 4, yk1 + 4, 'κ', { style: 'font-family:var(--f-body);font-style:italic;font-size:13px;fill:' + C.ink2 });
      S('tspan', { 'baseline-shift': '-0.3em', style: 'font-size:72%;font-style:normal' }, kl).textContent = '1';
      S('title', {}, kl).textContent = 'κ₁ = min(m₁,m₂)·min(n₁,n₂) = ' + info.kappa1 + ': the Kronecker rank of a generic rank-one update (Thm II.11(c))';
      /* strip labels */
      var lx = Math.max(m.l + 14, Math.min(W - m.r - 16, xr));
      T(svg, lx, m.t - 8, 'LoRA' + sub(st.r), { 'text-anchor': 'middle', style: 'font-family:var(--f-ui);font-size:12px;font-weight:500;fill:' + C.tide });
      /* preset points (positions first, so labels can avoid them) */
      var pts = data.all.map(function (res, i) { return { i: i, xr: res.cuts[0].rank, yr: res.cuts[1].rank }; });
      var groups = {};
      pts.forEach(function (p) { var key = p.xr + ',' + p.yr; (groups[key] = groups[key] || []).push(p); });
      Object.keys(groups).forEach(function (key) {
        var g = groups[key];
        g.forEach(function (p, t) { p.dx = (t - (g.length - 1) / 2) * 9; p.names = g.map(function (q) { return presetName(q.i); }).join(', '); });
      });
      var curP = null, others = [];
      pts.forEach(function (p) { var d = { p: p, cx: x(p.xr) + p.dx, cy: y(p.yr) }; if (p.i === data.idx) curP = d; else others.push(d); });
      function hitsOthers(bx0, bx1, by0, by1) { var n = 0; others.forEach(function (d) { if (d.cx + 6 > bx0 && d.cx - 6 < bx1 && d.cy + 6 > by0 && d.cy - 6 < by1) n++; }); return n; }
      /* 1. label of the selected update: the spot crossing the fewest lines and dots (it has priority) */
      var curLab = null;
      if (curP) {
        var nm = presetName(curP.p.i) + (W >= 400 ? '  (' + curP.p.xr + ', ' + curP.p.yr + ')' : ''), tw = 7 * nm.length;
        var cand = [[12, -12, 1], [-12, -12, 0], [12, 20, 1], [-12, 20, 0], [14, 4, 1], [-14, 4, 0], [0, -15, 2], [0, 24, 2]];
        cand.forEach(function (c, ci) {
          var bx0 = c[2] === 1 ? curP.cx + c[0] : c[2] === 2 ? curP.cx - tw / 2 : curP.cx + c[0] - tw, bx1 = bx0 + tw, by0 = curP.cy + c[1] - 10, by1 = curP.cy + c[1] + 3, sc = ci * 0.01;
          if (bx0 < m.l + 2 || bx1 > W - m.r - 2 || by0 < m.t + 1 || by1 > H - m.b - 1) sc += 100;
          [yk, yk1].forEach(function (ly) { if (ly > by0 && ly < by1) sc += 3; });
          if (xr > bx0 && xr < bx1) sc += 2;
          sc += 4 * hitsOthers(bx0, bx1, by0, by1);
          if (!curLab || sc < curLab.sc) curLab = { sc: sc, c: c, nm: nm, box: { x0: bx0 - 2, x1: bx1 + 2, y0: by0, y1: by1 } };
        });
      }
      /* 2. K strip label: by its edge, at the foot of the strip, or inside it, avoiding dots and the label above */
      var kTxt = 'K≤' + st.k + ' · ' + st.k + ' term' + (st.k > 1 ? 's' : ''), kw = 7 * kTxt.length;
      var kc = [[xr + 6, yk - 4], [m.l + 5, yk - 4], [xr + 6, yk + 13], [m.l + 5, yk + 13], [W - m.r - 5 - kw, yk - 4], [W - m.r - 5 - kw, yk + 13],
        [W - m.r - 5 - kw, H - m.b - 5], [xr + 6, H - m.b - 5], [m.l + 5, H - m.b - 5]], kb = null;
      /* sliding positions along the strip edge, preferring those nearest the LoRA line */
      for (var gx = m.l + 5; gx <= W - m.r - 5 - kw; gx += 10) { kc.push([gx, yk - 4, 0, Math.abs(gx - xr - 6) * 0.003 + 0.2]); kc.push([gx, yk + 13, 0, Math.abs(gx - xr - 6) * 0.003 + 0.25]); }
      [0.5, 0.3, 0.7].forEach(function (fy) {
        var yy = yk + (H - m.b - yk) * fy + 4;
        [xr + 6, m.l + 5, (m.l + W - m.r - kw) / 2, W - m.r - 5 - kw].forEach(function (xx) { kc.push([xx, yy, 1]); });
      });
      kc.forEach(function (c, ci) {
        var bx0 = c[0] - 2, bx1 = c[0] + kw + 2, by0 = c[1] - 10, by1 = c[1] + 3, sc = c[3] != null ? c[3] : ci * 0.01;
        if (bx0 < m.l + 1 || bx1 > W - m.r - 1 || by0 < m.t + 1 || by1 > H - m.b - 1) sc += 100;
        if (c[1] > yk + 1 && c[1] - 10 < yk) sc += 100;
        if (c[1] < yk && c[1] + 3 > yk) sc += 100;
        if (c[2]) sc += 1;   /* interior positions only when the edges are crowded */
        if (xr > bx0 && xr < bx1) sc += 3;
        if (yk1 > by0 && yk1 < by1) sc += 2;
        sc += 4 * hitsOthers(bx0, bx1, by0, by1);
        if (curP && curP.cx + 10 > bx0 && curP.cx - 10 < bx1 && curP.cy + 10 > by0 && curP.cy - 10 < by1) sc += 8;
        if (curLab) { var b = curLab.box; if (bx0 < b.x1 && bx1 > b.x0 && by0 < b.y1 && by1 > b.y0) sc += 10; }
        if (!kb || sc < kb.sc) kb = { sc: sc, x: c[0], y: c[1] };
      });
      var kt = T(svg, kb.x, kb.y, kTxt, { style: 'font-family:var(--f-ui);font-size:12px;font-weight:500;fill:' + A.css('--ochre-ink') });
      kt.setAttribute('paint-order', 'stroke'); kt.setAttribute('stroke', C.paper); kt.setAttribute('stroke-width', 3);
      /* 3. dots (click to select), then the selected one and its label on top */
      others.forEach(function (d) {
        var p = d.p;
        var dot = S('circle', { cx: d.cx, cy: d.cy, r: 4, fill: C.paper, stroke: C.ink2, 'stroke-width': 1.25, style: 'cursor:pointer' }, svg);
        S('title', {}, dot).textContent = presetName(p.i) + ': rk ' + p.xr + ', rk⊗ ' + p.yr + ' (click to select)';
        dot.addEventListener('click', function () { st.preset = PRESETS[p.i].id; hover = null; update(false); });
      });
      if (curP) {
        S('circle', { cx: curP.cx, cy: curP.cy, r: 9, fill: 'none', stroke: C.ochre, 'stroke-width': 2 }, svg);
        S('circle', { cx: curP.cx, cy: curP.cy, r: 4.6, fill: C.ink }, svg);
        var t = T(svg, curP.cx + curLab.c[0], curP.cy + curLab.c[1], curLab.nm, { 'text-anchor': ['end', 'start', 'middle'][curLab.c[2]], style: 'font-family:var(--f-ui);font-size:12px;font-weight:500;fill:' + C.ink });
        t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke', C.paper); t.setAttribute('stroke-width', 3);
      }
    }
    function renderPlaneVerdict() {
      var info = data.info, wit = data.wit, r = st.r, k = st.k, c = data.cur.cuts, r0 = c[0].rank, k0 = c[1].rank;
      var wl = wit.lora[r], wk = wit.kron[k];
      var A1 = wl.bound, B1 = wk.bound, okA = wl.witness === wl.bound, okB = wk.witness === wk.bound;
      /* containment is certain when the upper bound fits; non-containment is certain when the witness escapes */
      var loraInK = A1 <= k, kInLora = B1 <= r, loraOut = wl.witness > k, kOut = wk.witness > r;
      var rel, cls, RR = '\\(\\mathbb R^{16\\times16}\\)';
      if (loraInK && kInLora) { rel = 'Both are all of ' + RR; cls = 'sub'; }
      else if (loraInK && kOut) { rel = loraSub(r) + ' ⊊ ' + kSub(k) + (k >= info.kmax ? ' = ' + RR : ''); cls = 'sub'; }
      else if (kInLora && loraOut) { rel = kSub(k) + ' ⊊ ' + loraSub(r) + (r >= N ? ' = ' + RR : ''); cls = 'sub'; }
      else if (loraOut && kOut) { rel = loraSub(r) + ' and ' + kSub(k) + ' are incomparable'; cls = 'inc'; }
      else { rel = 'Undetermined: a witness fell short of its bound'; cls = ''; }
      /* (d) covers 1 <= r < nu, 1 <= k < kappa_1; beyond it the verdict rests on the bounds of (b), (c) and the live witnesses */
      var thm = cls === 'inc' ? (r < info.rkCD && k < info.kappa1 ? 'Thm II.11(d)' : 'Thm II.11(b, c)') :
        loraInK && kInLora ? 'Thm II.11(a)' : loraInK ? 'Thm II.11(a, c)' : kInLora ? 'Thm II.11(b)' : '';
      var html = '';
      html += '<div class="cut-ph" style="margin:0"><h4>' + 'Images at r = ' + r + ', k = ' + k + '</h4><span class="cut-tag">' + lc(thm) + '</span></div>';
      html += '<div class="cut-rel ' + cls + '">' + rel + '</div>';
      html += '<div class="cut-verdict" style="margin:0">' +
        '<span class="cut-chip ' + (r0 <= r ? 'y' : 'n') + '">' + (r0 <= r ? '✓ Δ ∈ ' : '✗ Δ ∉ ') + loraSub(r) + ' (rk ' + r0 + ')</span>' +
        '<span class="cut-chip ' + (k0 <= k ? 'y' : 'n') + '">' + (k0 <= k ? '✓ Δ ∈ ' : '✗ Δ ∉ ') + kSub(k) + ' (rk⊗ ' + k0 + ')</span></div>';
      html += '<div class="ln">Largest Kronecker rank on ' + loraSub(r) + ' is min(r·κ₁, ' + info.kmax + ') = <span class="cut-num">' + A1 + '</span>' +
        (okA ? '' : ' (the witness reached only ' + wl.witness + ')') + (loraInK ? ' ≤ k, so every rank-' + r + ' update is a ' + k + '-term Kronecker sum.' :
        ' &gt; k, so ' + (k < info.kappa1 ? 'a generic uvᵀ already escapes (rk⊗ = κ₁ = ' + info.kappa1 + ').' : 'a generic rank-' + r + ' update escapes.')) + '</div>';
      html += '<div class="ln">Largest rank on ' + kSub(k) + ' is min(k·' + info.rkCD + ', 16) = <span class="cut-num">' + B1 + '</span>' +
        (okB ? '' : ' (the witness reached only ' + wk.witness + ')') + (kInLora ? ' ≤ r, so every ' + k + '-term Kronecker sum has rank ≤ ' + r + '.' :
        ' &gt; r, so ' + (k === 1 ? 'a generic C⊗D (rank ' + info.rkCD + ') escapes.' : 'a generic ' + k + '-term sum escapes.')) + '</div>';
      html += '<div class="cut-num" style="font-size:.75rem;color:var(--ink-2)">Budgets · ' + loraSub(r) + ' ' + intStr(r * info.cost[0]) + ' · ' + kSub(k) + ' ' + intStr(k * info.cost[1]) + ' params' +
        (r * info.cost[0] === k * info.cost[1] ? ' · <span style="color:var(--ochre-ink)">equal budget</span>' : '') + '</div>';
      html += '<div class="cut-def">Here LoRA<sub>r</sub> stands for \\(\\mathcal M_{\\le r}\\), the updates of rank ≤ r (so \\(\\mathrm{Im}\\,\\mathrm{LoRA}_r=W_0+\\mathcal M_{\\le r}\\)): the strip rk ≤ r. ' +
        '<i>K</i><sub>≤k</sub> is \\(\\mathcal K_{\\le k}=\\{\\operatorname{rk}_\\otimes\\le k\\}\\), the sums of k unconstrained Kronecker products (Thm II.11(a)): the strip rk⊗ ≤ k. ' +
        'Both maxima are upper bounds by Thm II.11(b, c) and subadditivity; each is attained by a seeded random draw computed live.</div>';
      pv.innerHTML = hsub(html);
      A.typeset(pv);
    }

    /* ---------- legend & footnote ---------- */
    function renderLegend() {
      var C = col(), html = '';
      function sw(fill, stroke, dash) { return '<svg width="16" height="10" aria-hidden="true"><rect x="0.5" y="0.5" width="15" height="9" rx="1.5" fill="' + fill + '" stroke="' + (stroke || 'none') + '"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + '/></svg>'; }
      var ramp = '<svg width="58" height="10" aria-hidden="true"><defs><linearGradient id="' + uid + '-g" x1="0" x2="1"><stop offset="0" stop-color="' + C.neg + '"/><stop offset=".5" stop-color="' + C.paper + '"/><stop offset="1" stop-color="' + C.pos + '"/></linearGradient></defs><rect x="0.5" y="0.5" width="57" height="9" rx="1.5" fill="url(#' + uid + '-g)" stroke="' + C.rule + '"/></svg>';
      /* each item is a swatch plus one text span, so <sub>/<i> stay inline instead of becoming flex items */
      function li(mark, text) { html += '<span class="cut-li">' + mark + '<span>' + hsub(text) + '</span></span>'; }
      li(ramp, 'entry of Δ: − · 0 · +');
      CUTS.forEach(function (K, c) { li(sw(C.cut[c]), K.n + ' ' + K.name + [' · LoRA<sub>r</sub> strip', ' · <i>K</i><sub>≤k</sub> strip', ''][c]); });
      li('<svg width="18" height="10" aria-hidden="true"><line x1="0" x2="18" y1="5" y2="5" stroke="' + C.ink3 + '" stroke-dasharray="4 3"/></svg>', 'rank threshold 10⁻¹⁰‖Δ‖');
      li('<svg width="18" height="10" aria-hidden="true"><line x1="0" x2="18" y1="5" y2="5" stroke="' + C.ink3 + '" stroke-dasharray="3 3"/></svg>', 'κ₁ = ' + data.info.kappa1 + ': rk⊗ of a generic rank-one Δ');
      li(sw(C.paper, C.ochre), 'block → row under the reshape');
      li('<svg width="30" height="12" aria-hidden="true"><circle cx="5" cy="6" r="3.4" fill="' + C.paper + '" stroke="' + C.ink2 + '" stroke-width="1.2"/>' +
        '<circle cx="22" cy="6" r="5" fill="none" stroke="' + C.ochre + '" stroke-width="1.5"/><circle cx="22" cy="6" r="2.6" fill="' + C.ink + '"/></svg>', 'the presets in the plane; the ringed one is Δ (click to switch)');
      html += '<span class="cut-li"><span style="color:var(--moss-ink)">✓ contains</span><span style="color:var(--seal-ink);margin-left:.6rem">✗ cannot contain</span></span>';
      legend.innerHTML = html;
    }
    function renderFoot() {
      var dev = 0;
      data.all.forEach(function (a) { a.cuts.forEach(function (c) { dev = Math.max(dev, c.energyDev); }); });
      foot.innerHTML = hsub('Δ is normalised to ‖Δ‖<sub>F</sub> = 1; entries use i = u₁m₂ + u₂, j = v₁n₂ + v₂, so (C⊗D)[i, j] = C[u₁, v₁]·D[u₂, v₂]. ' +
        'Singular values by one-sided Jacobi in float64; rank = #{σᵢ > 10⁻¹⁰}. Every reshape permutes entries, so Σσᵢ² = 1 on each cut (largest deviation over all 24 spectra now: ' + sci(dev, 1) + '). ' +
        'On a 768 × 768 layer split 24 ⊗ 32 on both sides (HF’s default factorisation), κ₁ = 24·24 = ' + (24 * 24) + ' equals the largest possible Kronecker rank min(m₁n₁, m₂n₂) = ' + Math.min(24 * 24, 32 * 32) + ', as on every aligned split (Thm II.11(c)). ' +
        'Each string diagram draws its bond with dimension = the rank across that cut, the smallest bond of any two-box factorisation through it (SVD; Thm II.10 is the inequality rank ≤ bond).');
    }

    /* ---------- orchestration ---------- */
    function renderAll(animate) {
      renderControls();
      renderDesc();
      renderVerdict();
      renderMatrix(animate);
      renderCards();
      renderPlane();
      renderPlaneVerdict();
      renderLegend();
      renderFoot();
    }
    function update(animate) { compute(); renderAll(animate); }

    compute();
    renderAll(false);
    A.typeset(formula);

    /* play the rearrangement once, when the figure is actually on screen */
    function playOnce() { if (played) return; played = true; if (!A.reducedMotion()) renderMatrix(true); }
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (ents) { ents.forEach(function (en) { if (en.isIntersecting) { io.disconnect(); setTimeout(playOnce, 250); } }); }, { threshold: 0.3 });
      io.observe(matWrap);
    }

    A.onTheme(function () { renderMatrix(false); renderCards(); renderPlane(); renderLegend(); });
    var lastW = stage.clientWidth;
    var onResize = A.debounce(function () {
      if (Math.abs(stage.clientWidth - lastW) < 2) return;
      lastW = stage.clientWidth;
      renderMatrix(false); renderCards(); renderPlane();
    }, 120);
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(stage); else window.addEventListener('resize', onResize);

    /* test hook: pure math for headless verification */
    el.__cut = { state: st, data: function () { return data; }, svdVals: svdVals, cutMaps: cutMaps, buildDelta: buildDelta, analyse: analyse, witnesses: witnesses, splitInfo: splitInfo, SPLITS: SPLITS, PRESETS: PRESETS, TOL: TOL };
  });
})();
