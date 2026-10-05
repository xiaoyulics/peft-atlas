/* Fig. "gauge" — Gauge Lab: optimizers and merges.
   Thm III.3, Prop III.4, Thm III.12 and Prop V.4 of theory/framework.md (repaired statements).

   Tab A (optimizers). Fixed seeded B in R^{5x2}, A in R^{2x4}, loss gradient G in R^{5x4} (s = eta = 1).
   A gauge g in GL_2, g = R_theta . diag(e^t1, e^t2) . U_h . Pi, acts by (B, A) -> (B g^-1, g A); W = W0 + sBA is fixed and
   grad_B -> grad_B g^T, grad_A -> g^-T grad_A. For each update rule we compute the step at (B, A) and at (Bg^-1, gA) and
   report the relative defect, either of the first-order W-velocity dW = dB A + B dA, or of the parameter step
   (dB', dA') against the transported step (dB g^-1, g dA). Rules:
     SGD                         dB = grad_B, dA = grad_A                                    group O(2)   Thm III.3(b), III.12(a)
     Adam, first step            dB = phi(grad_B), phi(x) = x/(|x|+eps) (bias-corrected,     group B_2    Thm III.12(b)
                                 zero state: m^ = grad, v^ = grad^2)
     Adam + per-channel eta      eta_{B,j} = 1/r_j, eta_{A,j} = r_j, r_j = ||g_{j,:}||       Mon_2 jointly at eps = 0, else B_2
     Adam + per-channel eta, eps also eps_{B,j} = r_j eps, eps_{A,j} = eps / r_j            Mon_2 jointly           (Thm III.12(b))
                                 (r_j = |d_j| for a monomial g = D.P, so this is exactly the theorem's rescaling)
     ScaledGD, damped delta=0.1  dB = grad_B (AA^T + dI)^-1, dA = (B^TB + dI)^-1 grad_A      group O(2)   Thm III.12, published variants
     ScaledGD, undamped          delta = 0;  dW = P_U G + G P_V                              group GL_2   Prop III.4(a), Thm III.12(c)
     LoRA-Pro, Sylvester X       Wang et al. 2024, Thms 2.1-2.2; dW = P_U G + G P_V - P_U G P_V  GL_2 for dW (Prop III.4(b)),
                                                                                             O(2) for the parameter step (Thm III.12)
   Cometric ellipses: K restricted to a fixed 2-plane of gradients (compression <G_i, K G_j>), SGD vs undamped ScaledGD.

   Tab B (merges). N = 3 seeded modules B_i in R^{6x2}, A_i in R^{2x8}, LoraHub weights w = (0.6, 0.5, 0.4), s_i = 1.
   Module 2 is re-gauged by (B_2 g^-1, g A_2), g = c^-1 R_phi (= (cB_2, A_2/c) at phi = 0). Task arithmetic sum w_i B_i A_i is
   invariant; LoraHub (sum w_i B_i)(sum w_j A_j) changes by sum_{l != 2} w_l w_2 [B_2 (g^-1 - I) A_l + B_l (g - I) A_2], which at
   phi = 0 is Prop V.4's sum_{l != 2} w_l w_2 [(c-1) B_2 A_l + (c^-1 - 1) B_l A_2]. With a frozen shared A the only gauge left is
   joint and LH = (sum_j w_j) TA exactly (Prop V.4(b)). Every number on screen is computed here, in float64. */
(function () {
  'use strict';

  /* ---------- fixed setup ---------- */
  var SEED_A = 1, SEED_PLANE = 1051, SEED_B = 4;
  var MA = 5, NA = 4, RK = 2;
  var MB = 6, NB = 8, NMOD = 3;
  var WTS = [0.6, 0.5, 0.4];
  var DELTA = 0.1;                       // damping of the published ScaledGD variant
  var TOL = 1e-10;                       // relative defect below this = equivariant up to round-off
  var FLOOR = 1e-17, CEIL = 10;          // log axis of the defect bars
  var EPS_OPTS = [{ v: 0, label: '0' }, { v: 1e-8, label: '10⁻⁸' }, { v: 1e-2, label: '10⁻²' }];

  /* ---------- formatting ---------- */
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sup(n) { return String(n).split('').map(function (ch) { return SUP[ch] || ch; }).join(''); }
  function sci(x, d) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '∞';
    var p = Math.abs(x).toExponential(d == null ? 1 : d).split('e');
    var e = parseInt(p[1], 10);
    var s = x < 0 ? '−' : '';
    if (e === 0) return s + p[0];
    return s + p[0] + ' × 10' + sup(e);
  }
  function num(x, d) { /* fixed decimals with a true minus sign */
    var s = (Math.abs(x) < 0.5 * Math.pow(10, -(d == null ? 3 : d)) ? 0 : x).toFixed(d == null ? 3 : d);
    return s.charAt(0) === '-' ? '−' + s.slice(1) : s;
  }
  function defStr(x) { /* relative defect: 0, scientific below 1e-3, else 3 significant digits */
    if (x === 0) return '0';
    if (x < 1e-3) return sci(x, 1);
    return x.toPrecision(3);
  }
  function magStr(x) { /* a Frobenius norm */
    if (x === 0) return '0';
    if (x < 1e-3) return sci(x, 1);
    return String(+x.toPrecision(4));
  }
  function pctStr(x) { return x === 0 ? '0 %' : x < 1e-6 ? sci(x * 100, 1) + ' %' : (x * 100).toFixed(1) + ' %'; }
  /* Plex Mono's superscript glyphs are about 5px tall at 12px; readouts set their exponent as real digits, raised */
  var SUPINV = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };
  function expParts(str) {
    var m = /^(.*?)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)(.*)$/.exec(String(str));
    return m ? [m[1], m[2].split('').map(function (c) { return SUPINV[c]; }).join(''), m[3]] : null;
  }
  function escH(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function supHTML(str) { var p = expParts(str); return p ? escH(p[0]) + '<sup>' + p[1] + '</sup>' + escH(p[2]) : escH(str); }
  function supSVG(sel, str, px, minExp) {
    var p = expParts(str);
    if (!p) return sel.text(str);
    var ep = Math.max(minExp || 11, px * 0.8), up = +(px * 0.4).toFixed(2);
    sel.text(null);
    sel.append('tspan').text(p[0]);
    sel.append('tspan').attr('dy', -up).style('font-size', ep + 'px').text(p[1]);
    if (p[2]) sel.append('tspan').attr('dy', up).text(p[2]);
    return sel;
  }

  /* ---------- dense helpers (arrays of arrays, float64) ---------- */
  function zeros(m, n) { var X = []; for (var i = 0; i < m; i++) { X.push(new Array(n).fill(0)); } return X; }
  function eye(n) { var X = zeros(n, n); for (var i = 0; i < n; i++) X[i][i] = 1; return X; }
  function arr(F) { return Array.prototype.map.call(F, function (r) { return Array.prototype.slice.call(r); }); }
  function mul(X, Y) {
    var m = X.length, k = Y.length, n = Y[0].length, C = zeros(m, n);
    for (var i = 0; i < m; i++) for (var p = 0; p < k; p++) { var a = X[i][p]; if (a === 0) continue; for (var j = 0; j < n; j++) C[i][j] += a * Y[p][j]; }
    return C;
  }
  function tp(X) { var m = X.length, n = X[0].length, Y = zeros(n, m); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) Y[j][i] = X[i][j]; return Y; }
  function add(X, Y, b) { var bb = b == null ? 1 : b; return X.map(function (r, i) { return r.map(function (v, j) { return v + bb * Y[i][j]; }); }); }
  function scl(X, s) { return X.map(function (r) { return r.map(function (v) { return v * s; }); }); }
  function fro2(X) { var s = 0; for (var i = 0; i < X.length; i++) for (var j = 0; j < X[i].length; j++) s += X[i][j] * X[i][j]; return s; }
  function fro(X) { return Math.sqrt(fro2(X)); }
  function dot(X, Y) { var s = 0; for (var i = 0; i < X.length; i++) for (var j = 0; j < X[i].length; j++) s += X[i][j] * Y[i][j]; return s; }
  function inv2(M) {
    var a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1], det = a * d - b * c;
    return [[d / det, -b / det], [-c / det, a / det]];
  }
  function addI(M, d) { return M.map(function (r, i) { return r.map(function (v, j) { return v + (i === j ? d : 0); }); }); }
  /* Gaussian elimination with partial pivoting */
  function solve(L, b) {
    var n = b.length, M = L.map(function (r, i) { return r.concat([b[i]]); });
    for (var c = 0; c < n; c++) {
      var p = c;
      for (var r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      var tmp = M[c]; M[c] = M[p]; M[p] = tmp;
      for (r = c + 1; r < n; r++) { var f = M[r][c] / M[c][c]; for (var k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    var x = new Array(n).fill(0);
    for (var i = n - 1; i >= 0; i--) { var s = M[i][n]; for (var j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  }
  /* 2x2 Sylvester equation  M X + X N = C */
  function sylv2(M, N, C) {
    var L = zeros(4, 4), b = [C[0][0], C[0][1], C[1][0], C[1][1]];
    for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++) {
      var p = 2 * i + j;
      for (var k = 0; k < 2; k++) {
        L[p][2 * k + j] += M[i][k];      // (M X)_{ij} = sum_k M_ik X_kj
        L[p][2 * i + k] += N[k][j];      // (X N)_{ij} = sum_k X_ik N_kj
      }
    }
    var x = solve(L, b);
    return [[x[0], x[1]], [x[2], x[3]]];
  }
  /* symmetric 2x2 eigen-decomposition: values descending, unit vectors */
  function eig2(M) {
    var a = M[0][0], b = 0.5 * (M[0][1] + M[1][0]), d = M[1][1];
    var tr = a + d, disc = Math.sqrt(Math.max(0, 0.25 * (a - d) * (a - d) + b * b));
    var l1 = 0.5 * tr + disc, l2 = 0.5 * tr - disc;
    var ang = 0.5 * Math.atan2(2 * b, a - d);
    return { l1: l1, l2: l2, ang: ang };
  }

  /* ---------- the gauge element ---------- */
  function rotm(deg) {
    var c, s;
    if (Math.abs(deg % 90) < 1e-12) {
      var k = ((Math.round(deg / 90) % 4) + 4) % 4;
      c = [1, 0, -1, 0][k]; s = [0, 1, 0, -1][k];
    } else { var t = deg * Math.PI / 180; c = Math.cos(t); s = Math.sin(t); }
    return [[c, -s], [s, c]];
  }
  function gaugeOf(st) {
    var P = eye(2);
    if (st.swap) P = mul([[0, 1], [1, 0]], P);
    if (st.flip) P = mul(P, [[-1, 0], [0, 1]]);
    var D = [[Math.exp(st.t1), 0], [0, Math.exp(st.t2)]], U = [[1, st.h], [0, 1]];
    return mul(mul(mul(rotm(st.th), D), U), P);
  }
  function classify(g) {
    var gtg = mul(tp(g), g), od = fro(add(gtg, eye(2), -1));
    var mx = Math.max(Math.abs(g[0][0]), Math.abs(g[0][1]), Math.abs(g[1][0]), Math.abs(g[1][1]));
    var z = function (v) { return Math.abs(v) <= 1e-12 * mx; };
    var mono = (z(g[0][1]) && z(g[1][0]) && !z(g[0][0]) && !z(g[1][1])) || (z(g[0][0]) && z(g[1][1]) && !z(g[0][1]) && !z(g[1][0]));
    var orth = od <= 1e-12;
    var isId = fro(add(g, eye(2), -1)) <= 1e-12;
    return { orth: orth, mono: mono, isId: isId, orthDef: od, region: orth && mono ? 'B' : orth ? 'O' : mono ? 'M' : 'GL' };
  }

  /* ---------- update rules (eta = s = 1; a common sign and scale drop out of relative defects) ---------- */
  function phi(x, eps) { return eps === 0 ? (x > 0 ? 1 : x < 0 ? -1 : 0) : x / (Math.abs(x) + eps); }
  function grads(B, A, G) { return { gB: mul(G, tp(A)), gA: mul(tp(B), G) }; }
  function stepSGD(B, A, G) { var q = grads(B, A, G); return [q.gB, q.gA]; }
  function stepAdam(B, A, G, eps, rB, rA, eB, eA) {
    var q = grads(B, A, G);
    var dB = q.gB.map(function (row) { return row.map(function (v, j) { return (rB ? rB[j] : 1) * phi(v, eB ? eB[j] : eps); }); });
    var dA = q.gA.map(function (row, j) { return row.map(function (v) { return (rA ? rA[j] : 1) * phi(v, eA ? eA[j] : eps); }); });
    return [dB, dA];
  }
  function stepScaled(B, A, G, delta) {
    var q = grads(B, A, G);
    return [mul(q.gB, inv2(addI(mul(A, tp(A)), delta))), mul(inv2(addI(mul(tp(B), B), delta)), q.gA)];
  }
  function stepLoraPro(B, A, G) {
    var q = grads(B, A, G), BtB = mul(tp(B), B), AAt = mul(A, tp(A)), iB = inv2(BtB), iA = inv2(AAt);
    var X = sylv2(BtB, AAt, scl(mul(mul(iB, q.gA), tp(A)), -1));
    var PBperp = add(eye(B.length), mul(mul(B, iB), tp(B)), -1);
    var hA = add(mul(iB, q.gA), mul(X, A));
    var hB = add(mul(mul(PBperp, q.gB), iA), mul(B, X), -1);
    return [hB, hA];
  }
  function dWof(B, A, d) { return add(mul(d[0], A), mul(B, d[1])); }

  var OPTS = [
    { key: 'sgd', name: 'SGD', short: 'SGD', thm: 'Thm III.3(b), III.12(a)', tip: 'δB = ∇<sub>B</sub>, δA = ∇<sub>A</sub>, so δW ∝ G AᵀA + BBᵀG. Natural exactly along O(2) (Thm III.3(b), III.12(a)).' },
    { key: 'adam', name: 'Adam', short: 'Adam', thm: 'Thm III.12(b)', tip: 'First Adam step from zero state (bias-corrected): δB = φ(∇<sub>B</sub>), δA = φ(∇<sub>A</sub>), φ(x) = x/(|x|+ε) per coordinate. Group: signed permutations B₂ (Thm III.12(b)).' },
    { key: 'adamEta', name: 'Adam · per-channel η', short: 'Adam·η', thm: 'Thm III.12(b), joint', tip: 'Adam with per-rank-channel rates η<sub>B,j</sub> = η/r<sub>j</sub> and η<sub>A,j</sub> = r<sub>j</sub>·η, r<sub>j</sub> = ‖g<sub>j,:</sub>‖ (= |d<sub>j</sub>| for monomial g); ε shared. A joint symmetry with the hyperparameters on Mon₂ at ε = 0 only (Thm III.12(b)).' },
    { key: 'adamEtaEps', name: 'Adam · per-channel η, ε', short: 'Adam·η,ε', thm: 'Thm III.12(b), joint', tip: 'As before, plus ε<sub>B,j</sub> = r<sub>j</sub>·ε and ε<sub>A,j</sub> = ε/r<sub>j</sub>. A joint symmetry on Mon₂ for every ε (Thm III.12(b)).' },
    { key: 'scaledD', name: 'ScaledGD · damped, δ = 0.1', short: 'ScaledGD δ>0', thm: 'Thm III.12, variants', tip: 'δB = ∇<sub>B</sub>(AAᵀ+δI)⁻¹, δA = (BᵀB+δI)⁻¹∇<sub>A</sub> with δ = 0.1 here (Zhang–Pilanci damp with an unspecified δ > 0). Only O(2): g⁻¹(gAAᵀgᵀ+δI)g⁻ᵀ = AAᵀ + δg⁻¹g⁻ᵀ (Thm III.12).' },
    { key: 'scaled', name: 'ScaledGD · undamped', short: 'ScaledGD δ=0', thm: 'Prop III.4(a), III.12(c)', tip: 'Riemannian preconditioning (Tong–Ma–Chi 2021; Zhang–Pilanci 2024) with δ = 0, no Adam: δW ∝ P<sub>U</sub> G + G P<sub>V</sub>, a function of W. Group GL₂ (Prop III.4(a), Thm III.12(c)).' },
    { key: 'lorapro', name: 'LoRA-Pro · Sylvester X', short: 'LoRA-Pro', thm: 'Prop III.4(b), III.12', tip: 'LoRA-Pro’s SGD variant (Wang et al. 2024, Alg. 1) with the Sylvester-optimal free matrix X. Its first-order δW = P<sub>U</sub> G + G P<sub>V</sub> − P<sub>U</sub> G P<sub>V</sub> is the projection onto the tangent space of the rank-2 manifold, for any X (GL₂, Prop III.4(b)); its parameter step is only O(2)-equivariant (Thm III.12).' }
  ];
  /* theoretical group of each rule, given the measured level and epsilon */
  function groupOf(key, level, eps) {
    switch (key) {
      case 'sgd': case 'scaledD': return 'O';
      case 'adam': return 'B';
      case 'adamEta': return eps === 0 ? 'M' : 'B';
      case 'adamEtaEps': return 'M';
      case 'scaled': return 'GL';
      case 'lorapro': return level === 'W' ? 'GL' : 'O';
    }
    return 'GL';
  }
  var GROUP_NAME = { B: 'B₂', O: 'O(2)', M: 'Mon₂', GL: 'GL₂' };
  /* g in group X ?  (B subset O, B subset M, everything in GL) */
  function inGroup(region, grp) {
    if (grp === 'GL') return true;
    if (grp === 'B') return region === 'B';
    if (grp === 'O') return region === 'B' || region === 'O';
    if (grp === 'M') return region === 'B' || region === 'M';
    return false;
  }

  function computeA(base, g, eps) {
    var gi = inv2(g), B1 = mul(base.B, gi), A1 = mul(g, base.A), G = base.G;
    var rn = [Math.hypot(g[0][0], g[0][1]), Math.hypot(g[1][0], g[1][1])];
    var res = {};
    OPTS.forEach(function (o) {
      var d0, d1;
      switch (o.key) {
        case 'sgd': d0 = stepSGD(base.B, base.A, G); d1 = stepSGD(B1, A1, G); break;
        case 'adam': d0 = stepAdam(base.B, base.A, G, eps); d1 = stepAdam(B1, A1, G, eps); break;
        case 'adamEta':
          d0 = stepAdam(base.B, base.A, G, eps);
          d1 = stepAdam(B1, A1, G, eps, [1 / rn[0], 1 / rn[1]], rn); break;
        case 'adamEtaEps':
          d0 = stepAdam(base.B, base.A, G, eps);
          d1 = stepAdam(B1, A1, G, eps, [1 / rn[0], 1 / rn[1]], rn, [rn[0] * eps, rn[1] * eps], [eps / rn[0], eps / rn[1]]); break;
        case 'scaledD': d0 = stepScaled(base.B, base.A, G, DELTA); d1 = stepScaled(B1, A1, G, DELTA); break;
        case 'scaled': d0 = stepScaled(base.B, base.A, G, 0); d1 = stepScaled(B1, A1, G, 0); break;
        case 'lorapro': d0 = stepLoraPro(base.B, base.A, G); d1 = stepLoraPro(B1, A1, G); break;
      }
      var w0 = dWof(base.B, base.A, d0), w1 = dWof(B1, A1, d1);
      var tB = mul(d0[0], gi), tA = mul(g, d0[1]);
      var pnum = Math.sqrt(fro2(add(d1[0], tB, -1)) + fro2(add(d1[1], tA, -1)));
      var pden = Math.sqrt(fro2(tB) + fro2(tA));
      res[o.key] = { W: fro(add(w1, w0, -1)) / fro(w0), P: pnum / pden };
    });
    /* cometric compressions on the fixed gradient plane */
    function Ksgd(X) { return add(mul(X, mul(tp(A1), A1)), mul(mul(B1, tp(B1)), X)); }
    var PU = mul(mul(B1, inv2(mul(tp(B1), B1))), tp(B1)), PV = mul(mul(tp(A1), inv2(mul(A1, tp(A1)))), A1);
    function Ksc(X) { return add(mul(PU, X), mul(X, PV)); }
    function comp(K) { var P = base.plane; return [[dot(P[0], K(P[0])), dot(P[0], K(P[1]))], [dot(P[1], K(P[0])), dot(P[1], K(P[1]))]]; }
    return { g: g, gi: gi, rn: rn, res: res, Msgd: comp(Ksgd), Msc: comp(Ksc), cls: classify(g) };
  }

  function makeBaseA(A) {
    var rnd = A.rng(SEED_A);
    var B = arr(A.LA.randn(MA, RK, rnd)), Am = arr(A.LA.randn(RK, NA, rnd)), G = arr(A.LA.randn(MA, NA, rnd));
    var pr = A.rng(SEED_PLANE);
    var H1 = arr(A.LA.randn(MA, NA, pr)), H2 = arr(A.LA.randn(MA, NA, pr));
    var G1 = scl(H1, 1 / fro(H1));
    var H2p = add(H2, G1, -dot(H2, G1)), G2 = scl(H2p, 1 / fro(H2p));
    return { B: B, A: Am, G: G, plane: [G1, G2] };
  }

  /* ---------- Tab B: merges ---------- */
  function makeBaseB(A) {
    var rnd = A.rng(SEED_B), Bs = [], As = [];
    for (var i = 0; i < NMOD; i++) { Bs.push(arr(A.LA.randn(MB, RK, rnd))); As.push(arr(A.LA.randn(RK, NB, rnd))); }
    return { Bs: Bs, As: As };
  }
  function mergeRules(Bs, As) {
    var TA = zeros(MB, NB), SB = zeros(MB, RK), SA = zeros(RK, NB);
    for (var i = 0; i < Bs.length; i++) {
      TA = add(TA, mul(Bs[i], As[i]), WTS[i]);
      SB = add(SB, Bs[i], WTS[i]); SA = add(SA, As[i], WTS[i]);
    }
    return { TA: TA, LH: mul(SB, SA) };
  }
  function computeB(base, st) {
    var Bs = base.Bs, As = st.share ? base.As.map(function () { return base.As[0]; }) : base.As;
    var c = Math.pow(2, st.lc), g = scl(rotm(st.phi), 1 / c), gi = inv2(g);
    var joint = st.share || st.target === 'all';
    var Bs1 = Bs.map(function (B, i) { return joint || i === 1 ? mul(B, gi) : B; });
    var As1 = As.map(function (Am, i) { return joint || i === 1 ? mul(g, Am) : Am; });
    var m0 = mergeRules(Bs, As), m1 = mergeRules(Bs1, As1);
    var dLH = add(m1.LH, m0.LH, -1), dTA = add(m1.TA, m0.TA, -1);
    var closed = zeros(MB, NB);
    if (!joint) {
      var I = eye(2);
      for (var l = 0; l < NMOD; l++) {
        if (l === 1) continue;
        closed = add(closed, add(mul(mul(Bs[1], add(gi, I, -1)), As[l]), mul(mul(Bs[l], add(g, I, -1)), As[1])), WTS[l] * WTS[1]);
      }
    }
    var sumW = WTS.reduce(function (a, b) { return a + b; }, 0);
    return {
      c: c, g: g, joint: joint, B2: Bs1[1], A2: As1[1], P2: mul(Bs1[1], As1[1]), P2ref: mul(Bs[1], As[1]),
      B2ref: Bs[1], A2ref: As[1],
      TA: m1.TA, LH: m1.LH, TA0: m0.TA, LH0: m0.LH, dLH: dLH,
      nDLH: fro(dLH), nClosed: fro(closed), resid: fro(add(dLH, closed, -1)), rel: fro(dLH) / fro(m0.LH),
      nDTA: fro(dTA), nDP: fro(add(mul(Bs1[1], As1[1]), mul(Bs[1], As[1]), -1)),
      sumW: sumW, shareRes: fro(add(m1.LH, m1.TA, -sumW)), nB2: fro(Bs1[1]), nA2: fro(As1[1])
    };
  }

  /* Closed-form ‖ΔLH‖_F along the knob, log2 c in [-3, 3], at a fixed φ, for a gauge on module 2 alone of the
     unshared modules: sum_{l != 2} w_l w_2 [B_2 (g^-1 - I) A_l + B_l (g - I) A_2], g = c^-1 R_φ (Prop V.4(a)). */
  function curveB(base, phi, n) {
    var Bs = base.Bs, As = base.As, I = eye(2), R = rotm(phi), pts = [];
    for (var k = 0; k <= n; k++) {
      var lc = -3 + 6 * k / n, g = scl(R, Math.pow(2, -lc)), gi = inv2(g), D = zeros(MB, NB);
      for (var l = 0; l < NMOD; l++) {
        if (l === 1) continue;
        D = add(D, add(mul(mul(Bs[1], add(gi, I, -1)), As[l]), mul(mul(Bs[l], add(g, I, -1)), As[1])), WTS[l] * WTS[1]);
      }
      pts.push([lc, fro(D)]);
    }
    return pts;
  }

  /* ---------- module CSS ---------- */
  function injectCSS() {
    if (document.getElementById('css-gauge')) return;
    var F = '[data-figure="gauge"] ';
    var css = [
      F + '.gl-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      /* sub/superscripts inside running text never drop below the 12px reading floor */
      F + 'sub,' + F + 'sup{font-size:max(.8em,12px);line-height:0}',
      F + '.gl-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.gl-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.gl-title i{color:var(--tide)}',
      F + '.gl-kicker{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.gl-tabs{display:flex;flex-wrap:wrap;margin:.75rem 0 0;border-bottom:1px solid var(--rule)}',
      F + '.gl-tab{font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;background:none;border:0;border-bottom:2px solid transparent;color:var(--ink-2);padding:.5rem .85rem .45rem;margin-bottom:-1px;cursor:pointer}',
      F + '.gl-tab:hover{color:var(--ink)}',
      F + '.gl-tab[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--tide)}',
      F + '.gl-tab b{font-weight:600;color:var(--tide);margin-right:.4em}',
      F + '.gl-panel[hidden]{display:none}',
      F + '.gl-instr{margin:.7rem 0 .75rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:70ch}',
      F + '.gl-instr b{color:var(--ink);font-weight:600}',
      F + '.gl-formula{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem 1.6rem;padding:.5rem .75rem;margin:0 0 1rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.9rem;line-height:1.55;color:var(--ink);overflow-x:auto}',
      F + '.gl-formula > span{display:inline-flex;flex-wrap:wrap;align-items:center;gap:.15rem .5rem;min-width:0}',
      F + '.gl-formula mjx-container{white-space:nowrap}',
      F + '.gl-formula mjx-container svg{max-width:none}',
      /* UI labels: Plex Sans 500, small caps-style uppercase, letter-spacing kept at .06em */
      F + '.gl-tag{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.gl-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem 1.6rem;margin-bottom:1.1rem}',
      F + '.gl-group{min-width:0;border-top:1px solid var(--rule);padding-top:.5rem}',
      F + '.gl-gl{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem .6rem;margin-bottom:.45rem}',
      F + '.gl-gform{font-family:var(--f-body);font-style:italic;font-size:.9rem;color:var(--ink-2);white-space:nowrap}',
      F + '.gl-sliders{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem .9rem}',
      F + '.gl-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.gl-sl-top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0 .4rem;min-width:0}',
      F + '.gl-sl label{font-family:var(--f-body);font-size:var(--fs-sm);font-style:italic;color:var(--ink);white-space:nowrap}',
      F + '.gl-sl label .nm{font-family:var(--f-ui);font-style:normal;font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.3rem}',
      F + '.gl-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.gl-sl input[type=range]{margin:0;height:1.25rem}',
      F + '.gl-btnrow{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin-top:.55rem}',
      F + '.gl-btnrow .gl-k{margin-right:.15rem}',
      F + '.gl-k{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.gl-pill{font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;padding:.28rem .66rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap;font-variant-numeric:tabular-nums}',
      F + '.gl-pill:hover:not([aria-pressed="true"]):not(:disabled){border-color:var(--ink-3);color:var(--ink)}',
      F + '.gl-pill[aria-pressed="true"]{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
      F + '.gl-pill.cur{border-color:var(--ochre);color:var(--ink)}',
      F + '.gl-pill:disabled{opacity:.4;cursor:not-allowed}',
      F + '.gl-pill.play{border-color:var(--ink-3);color:var(--ink)}',
      F + '.gl-row{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .6rem;margin:.3rem 0 .45rem}',
      F + '.gl-row .gl-k{min-width:4.8rem}',
      F + '.gl-seg button:disabled{opacity:.4;cursor:not-allowed}',
      F + '.gl-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.5rem;align-items:start}',
      F + '.gl-col{display:grid;gap:1rem;min-width:0}',
      F + '.gl-card{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.7rem .8rem .75rem;min-width:0}',
      F + '.gl-cap{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.15rem .6rem;margin-bottom:.4rem}',
      F + '.gl-cap .gl-sub{font-family:var(--f-body);font-size:.84rem;color:var(--ink-2)}',
      /* g glyph + matrix */
      F + '.gl-gbox{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem 1.1rem}',
      F + '.gl-glyph{width:128px;height:128px;flex:none;display:block}',
      F + '.gl-gread{display:grid;gap:.4rem;min-width:0;flex:1 1 10rem}',
      F + '.gl-mat{display:flex;align-items:center;gap:.45rem;font-family:var(--f-mono);font-size:.82rem;font-variant-numeric:tabular-nums;color:var(--ink)}',
      F + '.gl-mat .lhs{font-family:var(--f-body);font-style:italic;font-size:1.05rem;white-space:nowrap}',
      F + '.gl-matb{position:relative;display:grid;grid-template-columns:auto auto;gap:.1rem .7rem;padding:.15rem .55rem;text-align:right}',
      F + '.gl-matb::before,.gl-matb::after{content:"";position:absolute;top:0;bottom:0;width:.32rem;border:1.5px solid var(--ink-2)}',
      F + '.gl-matb::before{left:0;border-right:0}',
      F + '.gl-matb::after{right:0;border-left:0}',
      /* labels are formulas (serif), values are readouts (mono) */
      F + '.gl-gmeta{font-family:var(--f-body);font-size:.88rem;line-height:1.55;color:var(--ink-2);display:grid}',
      F + '.gl-gmeta span{white-space:nowrap}',
      F + '.gl-gmeta b{font-family:var(--f-mono);font-size:.8rem;font-weight:500;color:var(--ink);font-variant-numeric:tabular-nums}',
      F + '.gl-member{font-family:var(--f-display);font-size:1.02rem;color:var(--ink);line-height:1.2}',
      F + '.gl-member .x{color:var(--ochre)}',
      F + '.gl-venn{display:block;width:100%;height:auto;max-width:440px;margin:0 auto}',
      F + '.gl-bars{width:100%;min-height:120px}',
      F + '.gl-bars svg{display:block;overflow:visible}',
      F + '.gl-legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;margin:.55rem 0 0;font-family:var(--f-ui);font-weight:500;font-size:.78rem;line-height:1.4;color:var(--ink-2)}',
      F + '.gl-legend span{display:inline-flex;align-items:center;gap:.4rem}',
      F + '.gl-legend i{flex:none;display:inline-block;width:.8rem;height:.55rem;border-radius:2px}',
      F + '.gl-verdict{margin:.7rem 0 0;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink);border-left:2px solid var(--ochre);padding:.05rem 0 .05rem .7rem}',
      F + '.gl-verdict .ok{color:var(--moss-ink);font-weight:600}',
      F + '.gl-verdict .no{color:var(--seal-ink);font-weight:600}',
      F + '.gl-verdict .thm{font-family:var(--f-ui);font-weight:500;font-size:.78rem;letter-spacing:.01em;color:var(--ink-2);white-space:nowrap}',
      F + '.gl-ells{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem 1.2rem}',
      F + '.gl-ells-note{grid-column:1/-1;margin-top:0}',
      F + '.gl-ell{min-width:0}',
      F + '.gl-ell svg{display:block;width:100%;height:auto;max-width:210px;margin:0 auto}',
      F + '.gl-ell-h{font-family:var(--f-display);font-weight:600;font-size:.92rem;color:var(--ink);margin-bottom:.1rem}',
      F + '.gl-ell-h small{display:block;font-family:var(--f-body);font-weight:var(--w-body);font-size:.88rem;color:var(--ink-2)}',
      F + '.gl-ell-r{font-family:var(--f-ui);font-size:.78rem;line-height:1.5;color:var(--ink-2);text-align:center;margin-top:.15rem}',
      F + '.gl-ell-r .n{font-family:var(--f-mono);font-variant-numeric:tabular-nums;color:var(--ink)}',
      /* notes and footnotes are prose: Source Serif at reading size */
      F + '.gl-note{font-family:var(--f-body);font-size:.86rem;line-height:1.5;color:var(--ink-2);margin-top:.45rem}',
      F + '.gl-foot{margin:1.1rem 0 0;padding-top:.7rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.84rem;line-height:1.55;color:var(--ink-2)}',
      /* Tab B */
      F + '.gl-fib{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,8fr);gap:.35rem;max-width:300px;margin:0 auto}',
      F + '.gl-hm{position:relative;min-width:0}',
      F + '.gl-hm svg{display:block;width:100%;height:auto;border-radius:2px}',
      F + '.gl-hm-l{font-family:var(--f-body);font-style:italic;font-size:.9rem;color:var(--ink);line-height:1.2;margin:0 0 .15rem;white-space:nowrap}',
      F + '.gl-hm-l small{font-family:var(--f-mono);font-style:normal;font-size:.75rem;color:var(--ink-2);margin-left:.3rem}',
      F + '.gl-fib .gl-corner{display:flex;align-items:flex-end;justify-content:center}',
      F + '.gl-merges{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem}',
      F + '.gl-mcard{border:1.5px solid var(--rule);border-radius:var(--radius);padding:.45rem .5rem .5rem;background:var(--paper);min-width:0}',
      F + '.gl-mcard.inv{border-color:var(--moss)}',
      F + '.gl-mcard.mov{border-color:var(--seal)}',
      F + '.gl-mhead{display:flex;flex-direction:column;align-items:flex-start;gap:.25rem;margin-bottom:.4rem}',
      F + '.gl-mname{font-family:var(--f-display);font-weight:var(--w-head);font-size:.95rem;color:var(--ink);line-height:1.15}',
      F + '.gl-mrule{font-family:var(--f-body);font-weight:var(--w-body);font-size:.82rem;color:var(--ink-2);display:block;margin-top:.15rem;line-height:1.3}',
      F + '.gl-mval{font-family:var(--f-body);font-size:.86rem;color:var(--ink-2);margin-top:.3rem;line-height:1.45}',
      F + '.gl-mval b{font-family:var(--f-mono);font-size:.8rem;font-weight:500;font-variant-numeric:tabular-nums;white-space:nowrap}',
      F + '.gl-chip{display:inline-flex;align-items:center;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;padding:.06rem .45rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap}',
      F + '.gl-chip.yes{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.gl-chip.no{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.gl-chip.neu{color:var(--ink-2)}',
      F + '.gl-meter{display:grid;grid-template-columns:minmax(0,1fr);gap:.5rem 1rem;align-content:center;padding:.2rem .15rem}',
      F + '.gl-mt{min-width:0;display:grid;align-content:end;gap:.1rem}',
      F + '.gl-mt .k{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2);display:block;line-height:1.3}',
      F + '.gl-mt .v{font-family:var(--f-mono);font-size:1.02rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.gl-mt .v.ok{color:var(--moss-ink)}',
      F + '.gl-mt .v.no{color:var(--seal-ink)}',
      F + '.gl-scale{display:flex;flex-wrap:wrap;align-items:center;gap:.2rem .45rem;font-family:var(--f-mono);font-size:.75rem;font-variant-numeric:tabular-nums;color:var(--ink-2)}',
      F + '.gl-scale .bar{width:6rem;height:.5rem;border-radius:2px;border:1px solid var(--rule)}',
      F + '.gl-scale .lab{flex-basis:100%;font-family:var(--f-ui);font-weight:500;font-size:.78rem}',
      F + '.gl-gB{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem .6rem;margin-top:.6rem}',
      F + '.gl-gB .gl-mat{font-size:.78rem}',
      F + '.gl-gB .gl-mat .lhs{font-size:.98rem}',
      F + '.gl-zero{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none}',
      F + '.gl-zero[hidden]{display:none}',
      F + '.gl-zero span{font-family:var(--f-mono);font-size:.8rem;color:var(--moss-ink);background:var(--paper);border:1px solid var(--moss);border-radius:999px;padding:.12rem .55rem;font-variant-numeric:tabular-nums}',
      F + '.gl-curve{margin-top:1rem}',
      F + '.gl-cbox{width:100%;min-height:150px}',
      F + '.gl-cbox svg{display:block;overflow:visible}',
      F + '.gl-legend i.ln{height:0;border-top:2px solid;border-radius:0;width:1.1rem}',
      F + '.gl-legend i.ln.dash{border-top-style:dashed}',
      F + '.gl-legend i.dot{width:.55rem;height:.55rem;border-radius:50%}',
      '@container (min-width:560px){' + F + '.gl-merges{grid-template-columns:repeat(3,minmax(0,1fr))}' + F + '.gl-meter{grid-column:1/-1;grid-template-columns:repeat(4,minmax(0,1fr));padding:.35rem 0 0}}',
      '@container (max-width:480px){' + F + '.gl-formula{font-size:.84rem;gap:.3rem 1rem;padding:.45rem .6rem}}',
      '@container (min-width:720px){' +
        F + '.gl-controls{grid-template-columns:minmax(0,1.65fr) minmax(0,1fr)}' +
        F + '.gl-sliders.four{grid-template-columns:repeat(4,minmax(0,1fr))}' +
        /* the left column keeps the Venn wide enough for 12px labels inside its lunes */
        F + '.gl-grid.a{grid-template-columns:minmax(0,.84fr) minmax(0,1fr)}' +
        F + '.gl-grid.b{grid-template-columns:minmax(0,.62fr) minmax(0,1.38fr)}' +
        F + '.gl-fib{max-width:none}' +
        F + '.gl-ells{grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1.3fr);align-items:center}' +
        F + '.gl-ells-note{grid-column:auto}' +
      '}',
      '@container (max-width:420px){' + F + '.gl-merges{gap:.4rem}' + F + '.gl-mcard{padding:.35rem .35rem .4rem}' + F + '.gl-mname{font-size:.9rem}' + F + '.gl-glyph{width:120px;height:120px}' + '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-gauge';
    s.textContent = css;
    document.head.appendChild(s);
  }

  Atlas.register('gauge', function (el, A) {
    injectCSS();
    var d3 = window.d3;
    var h = A.h;
    var uid = 'gl' + Math.random().toString(36).slice(2, 7);
    var reduce = A.reducedMotion();

    var baseA = makeBaseA(A), baseB = makeBaseB(A);
    var PRESETS = {
      I: { th: 0, t1: 0, t2: 0, h: 0, swap: false, flip: false },
      rotation: { th: 30, t1: 0, t2: 0, h: 0, swap: false, flip: false },
      diagonal: { th: 0, t1: 0.5, t2: -0.4, h: 0, swap: false, flip: false },
      sperm: { th: 0, t1: 0, t2: 0, h: 0, swap: true, flip: true },
      generic: { th: 25, t1: 0.4, t2: -0.3, h: 0.35, swap: false, flip: false }
    };
    var stA = Object.assign({}, PRESETS.rotation, { level: 'W', eps: 0 });
    var stB = { lc: 1, phi: 0, target: 'm2', share: false };
    var tab = 'A';
    var curA = null, curB = null;

    /* ---------- shell ---------- */
    var stage = h('div', { class: 'stage gl-stage', role: 'group', 'aria-label': 'Gauge lab. Tab A: a gauge g in GL2 re-writes the LoRA factors (B, A) as (B g inverse, g A) without changing the model; bars show which optimizers change their step. Tab B: re-gauging one LoRA module leaves task arithmetic unchanged but moves the LoraHub merge.' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'gl-head' }, [
      h('h3', { class: 'gl-title', html: 'Update rules that see the <i>gauge</i>' }),
      h('span', { class: 'gl-kicker', text: 'Gauge lab · Thm III.3 · Prop III.4 · Thm III.12 · Prop V.4' })
    ]));
    var tabs = h('div', { class: 'gl-tabs', role: 'tablist', 'aria-label': 'Gauge lab tabs' });
    stage.appendChild(tabs);
    var tabBtns = {}, panels = {};
    [['A', 'A', 'Optimizers'], ['B', 'B', 'Merges']].forEach(function (t) {
      var b = h('button', { type: 'button', class: 'gl-tab', role: 'tab', id: uid + '-tab' + t[0], 'aria-controls': uid + '-pan' + t[0], 'aria-selected': String(t[0] === tab), tabindex: t[0] === tab ? '0' : '-1', html: '<b>' + t[1] + '</b>' + t[2] });
      b.addEventListener('click', function () { selectTab(t[0]); });
      b.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); var nx = tab === 'A' ? 'B' : 'A'; selectTab(nx); tabBtns[nx].focus(); }
      });
      tabs.appendChild(b); tabBtns[t[0]] = b;
      var p = h('section', { class: 'gl-panel', role: 'tabpanel', id: uid + '-pan' + t[0], 'aria-labelledby': uid + '-tab' + t[0] });
      if (t[0] !== tab) p.hidden = true;
      stage.appendChild(p); panels[t[0]] = p;
    });
    function selectTab(t) {
      if (t === tab) return;
      tab = t;
      ['A', 'B'].forEach(function (k) {
        tabBtns[k].setAttribute('aria-selected', String(k === t)); tabBtns[k].setAttribute('tabindex', k === t ? '0' : '-1');
        panels[k].hidden = k !== t;
      });
      stopTour(); stopSweep();
      if (t === 'A') renderA(); else renderB();
    }

    /* ---------- small UI builders ---------- */
    function slider(o) {
      var id = uid + '-' + o.key;
      var input = h('input', { type: 'range', id: id, min: o.min, max: o.max, step: o.step, value: o.value, 'aria-valuetext': o.fmt(o.value) });
      var out = h('output', { for: id, text: o.fmt(o.value) });
      input.addEventListener('input', function () { var v = +input.value; out.textContent = o.fmt(v); input.setAttribute('aria-valuetext', o.fmt(v)); o.onInput(v); });
      var wrap = h('div', { class: 'gl-sl' }, [h('div', { class: 'gl-sl-top' }, [h('label', { for: id, html: o.label + (o.name ? '<span class="nm">' + o.name + '</span>' : '') }), out]), input]);
      return { el: wrap, set: function (v) { input.value = v; var vv = +input.value; out.textContent = o.fmt(vv); input.setAttribute('aria-valuetext', o.fmt(vv)); } };
    }
    function pill(label, onClick, attrs) {
      var b = h('button', Object.assign({ type: 'button', class: 'gl-pill', html: label }, attrs || {}));
      b.addEventListener('click', onClick);
      return b;
    }
    function formulaItem(tag, tex) { return h('span', {}, [h('span', { class: 'gl-tag', text: tag }), h('span', { html: tex })]); }

    /* ======================================================================
       TAB A — optimizers
       ====================================================================== */
    var pA = panels.A;
    pA.appendChild(h('p', { class: 'gl-instr', html: 'Move <b>g</b> through GL₂. The model <i>W</i> = <i>W</i>₀ + <i>sBA</i> never changes; the bars show which optimizers’ steps do.' }));
    var fA = h('div', { class: 'gl-formula', 'aria-label': 'Gauge action and defect' });
    pA.appendChild(fA);
    var fDef = h('span', { html: '' });
    fA.appendChild(formulaItem('gauge', '\\((B,A)\\mapsto(Bg^{-1},\\,gA)\\) fixes \\(W=W_0+sBA\\);\\(\\ \\nabla_B\\mapsto\\nabla_B\\,g^{\\top},\\ \\nabla_A\\mapsto g^{-\\top}\\nabla_A\\)'));
    fA.appendChild(h('span', {}, [h('span', { class: 'gl-tag', text: 'defect' }), fDef]));
    function setDefFormula() {
      fDef.innerHTML = stA.level === 'W'
        ? '\\(\\|\\delta W(g)-\\delta W(I)\\|_F\\,/\\,\\|\\delta W(I)\\|_F\\)'
        : '\\(\\|(\\delta B\',\\delta A\')-(\\delta B\\,g^{-1},\\,g\\,\\delta A)\\|\\,/\\,\\|(\\delta B\\,g^{-1},\\,g\\,\\delta A)\\|\\)';
      A.typeset(fDef);
    }

    var ctlA = h('div', { class: 'gl-controls' });
    pA.appendChild(ctlA);
    var gG = h('div', { class: 'gl-group' }, [h('div', { class: 'gl-gl' }, [h('span', { class: 'gl-tag', text: 'Gauge g ∈ GL₂' }), h('span', { class: 'gl-gform', html: 'g = R<sub>θ</sub> · diag(e<sup>t₁</sup>, e<sup>t₂</sup>) · U<sub>h</sub> · Π' })])]);
    ctlA.appendChild(gG);
    var slBox = h('div', { class: 'gl-sliders four' });
    gG.appendChild(slBox);
    var SL = {};
    [
      { key: 'th', label: 'θ', name: 'rotate', min: -180, max: 180, step: 1, fmt: function (v) { return (v < 0 ? '−' : '') + Math.abs(Math.round(v)) + '°'; } },
      { key: 't1', label: 't₁', name: 'stretch', min: -1, max: 1, step: 0.05, fmt: function (v) { return num(v, 2); } },
      { key: 't2', label: 't₂', name: 'stretch', min: -1, max: 1, step: 0.05, fmt: function (v) { return num(v, 2); } },
      { key: 'h', label: 'h', name: 'shear', min: -1, max: 1, step: 0.05, fmt: function (v) { return num(v, 2); } }
    ].forEach(function (o) {
      o.value = stA[o.key];
      o.onInput = function (v) { stopTour(); stA[o.key] = v; renderA(); };
      SL[o.key] = slider(o);
      slBox.appendChild(SL[o.key].el);
    });
    var rowPi = h('div', { class: 'gl-btnrow' }, [h('span', { class: 'gl-k', text: 'Π' })]);
    var bSwap = pill('swap J', function () { stopTour(); stA.swap = !stA.swap; renderA(); }, { 'aria-pressed': 'false', title: 'J = [[0,1],[1,0]] swaps the two rank channels' });
    var bFlip = pill('flip F', function () { stopTour(); stA.flip = !stA.flip; renderA(); }, { 'aria-pressed': 'false', title: 'F = diag(−1, 1) flips the sign of channel 1' });
    rowPi.appendChild(bSwap); rowPi.appendChild(bFlip);
    gG.appendChild(rowPi);
    var rowPre = h('div', { class: 'gl-btnrow' }, [h('span', { class: 'gl-k', text: 'Presets' })]);
    var preBtns = {};
    [['I', 'I'], ['rotation', 'rotation'], ['diagonal', 'diagonal'], ['sperm', 'signed perm'], ['generic', 'generic']].forEach(function (p) {
      var b = pill(p[1], function () { stopTour(); applyPreset(p[0]); });
      preBtns[p[0]] = b; rowPre.appendChild(b);
    });
    var bTour = pill(reduce ? 'next ▸' : '▶ tour', function () { toggleTour(); }, { class: 'gl-pill play', 'aria-pressed': 'false', title: reduce ? 'Step g along a path through GL₂' : 'Animate g along a path through B₂, O(2), Mon₂ and GL₂' });
    rowPre.appendChild(bTour);
    gG.appendChild(rowPre);

    var gM = h('div', { class: 'gl-group' }, [h('div', { class: 'gl-gl' }, [h('span', { class: 'gl-tag', text: 'Measure' })])]);
    ctlA.appendChild(gM);
    var segLevel = A.seg([{ value: 'W', label: 'first-order δW' }, { value: 'P', label: 'step (δB, δA)' }], stA.level, function (v) { stopTour(); stA.level = v; setDefFormula(); renderA(); }, 'Measured quantity');
    gM.appendChild(h('div', { class: 'gl-row' }, [h('span', { class: 'gl-k', text: 'Level' }), segLevel.el]));
    var segEps = A.seg(EPS_OPTS.map(function (e, i) { return { value: i, label: 'ε = ' + e.label }; }), 0, function (v) { stopTour(); stA.eps = EPS_OPTS[v].v; renderA(); }, 'Adam epsilon');
    gM.appendChild(h('div', { class: 'gl-row' }, [h('span', { class: 'gl-k', html: 'Adam <span style="text-transform:none">ε</span>' }), segEps.el]));
    gM.appendChild(h('div', { class: 'gl-note', html: 'Gradients here are O(1); PyTorch’s default ε is 10⁻⁸. The per-channel rows use r<sub>j</sub> = ‖g<sub>j,:</sub>‖, which is |d<sub>j</sub>| when g is monomial.' }));

    var gridA = h('div', { class: 'gl-grid a' });
    pA.appendChild(gridA);
    var colL = h('div', { class: 'gl-col' }), colR = h('div', { class: 'gl-col' });
    gridA.appendChild(colL); gridA.appendChild(colR);

    /* card: g itself */
    var cardG = h('div', { class: 'gl-card' }, [h('div', { class: 'gl-cap' }, [h('span', { class: 'gl-tag', text: 'g acting on ℝʳ, r = 2' }), h('span', { class: 'gl-sub', text: 'unit circle ↦ g·S¹' })])]);
    var gbox = h('div', { class: 'gl-gbox' });
    cardG.appendChild(gbox);
    var glyph = d3.select(gbox).append('svg').attr('class', 'gl-glyph').attr('viewBox', '-66 -66 132 132').attr('role', 'img');
    var gread = h('div', { class: 'gl-gread' });
    gbox.appendChild(gread);
    var matEl = h('div', { class: 'gl-mat' });
    var memberEl = h('div', { class: 'gl-member' });
    var gmeta = h('div', { class: 'gl-gmeta' });
    gread.appendChild(matEl); gread.appendChild(memberEl); gread.appendChild(gmeta);
    colL.appendChild(cardG);

    /* card: where g lives */
    var cardV = h('div', { class: 'gl-card' }, [h('div', { class: 'gl-cap' }, [h('span', { class: 'gl-tag', text: 'Where g lives · who is blind to it' }), h('span', { class: 'gl-sub', text: 'O(2) ∩ Mon₂ = B₂' })])]);
    var venn = d3.select(cardV).append('svg').attr('class', 'gl-venn').attr('viewBox', '0 0 360 236').attr('role', 'img');
    cardV.appendChild(h('div', { class: 'gl-note', html: 'Each rule sits in its symmetry group, the largest subgroup of GL₂ under which it is equivariant, and it ignores g exactly when the dot lies inside that group. Mon₂ = signed permutations × positive diagonals. * = a <i>joint</i> symmetry: g acts together with a per-channel rescaling of Adam’s rates (and of ε when ε &gt; 0).' }));
    colL.appendChild(cardV);

    /* card: defect bars */
    var cardBars = h('div', { class: 'gl-card' });
    var barsCap = h('span', { class: 'gl-sub', text: '' });
    cardBars.appendChild(h('div', { class: 'gl-cap' }, [h('span', { class: 'gl-tag', text: 'Gauge defect per update rule' }), barsCap]));
    var barsBox = h('div', { class: 'gl-bars' });
    cardBars.appendChild(barsBox);
    cardBars.appendChild(h('div', { class: 'gl-legend' }, [
      h('span', { html: '<i style="background:var(--moss)"></i>equivariant: blind to g (≤ 10⁻¹⁰, round-off)' }),
      h('span', { html: '<i style="background:var(--seal)"></i>gauge-dependent: the step depends on the chart' })
    ]));
    var verdictA = h('p', { class: 'gl-verdict', 'aria-live': 'polite' });
    cardBars.appendChild(verdictA);
    colR.appendChild(cardBars);

    /* card: cometric ellipses */
    var cardE = h('div', { class: 'gl-card', style: 'margin-top:1rem' }, [h('div', { class: 'gl-cap' }, [h('span', { class: 'gl-tag', text: 'Cometric K on a fixed plane of gradients' }), h('span', { class: 'gl-sub', text: 'dashed: g = I' })])]);
    var ells = h('div', { class: 'gl-ells' });
    cardE.appendChild(ells);
    var ellParts = {};
    [['sgd', 'SGD', 'K(G) = G AᵀA + BBᵀG'], ['sc', 'ScaledGD, δ = 0', 'K(G) = P<sub>U</sub>G + GP<sub>V</sub>']].forEach(function (e) {
      var box = h('div', { class: 'gl-ell' }, [h('div', { class: 'gl-ell-h', html: e[1] + '<small>' + e[2] + '</small>' })]);
      var svg = d3.select(box).append('svg').attr('viewBox', '-80 -66 160 132').attr('role', 'img');
      var rd = h('div', { class: 'gl-ell-r' });
      box.appendChild(rd);
      ells.appendChild(box);
      ellParts[e[0]] = { svg: svg, rd: rd };
    });
    ells.appendChild(h('div', { class: 'gl-note gl-ells-note', html: 'The unit circle of the gradient plane G = cos φ G₁ + sin φ G₂, mapped by K and projected back onto the plane (a symmetric 2×2 operator, so an ellipse). Orthogonal g leave the SGD ellipse fixed and stretches or shears deform it (Thm III.3(b)); the undamped ScaledGD ellipse never moves (Prop III.4(a)). Adam’s step is not linear in the gradient, so it has no ellipse.' }));
    pA.appendChild(cardE);

    pA.appendChild(h('p', { class: 'gl-foot', html: 'Setup: B ∈ ℝ⁵ˣ², A ∈ ℝ²ˣ⁴ and the loss gradient G ∈ ℝ⁵ˣ⁴ are fixed (seed ' + SEED_A + '), s = η = 1; A has full row rank and r = 2 &lt; n = 4, as Thm III.3(b) needs. Adam is its first step from zero state (bias-corrected, so the step is φ(∇) = ∇/(|∇|+ε)). ScaledGD and LoRA-Pro run without Adam (for LoRA-Pro, its SGD variant). At the δW level, the first-order W-velocity, undamped ScaledGD and LoRA-Pro ignore all of GL₂ (Prop III.4). Damped ScaledGD is only O(2)-equivariant at both levels, and LoRA-Pro’s parameter update with the Sylvester-optimal X is only O(2)-equivariant (Thm III.12). Published runs damp the inverses or wrap the step in AdamW, which leaves O(2) or B₂. ScaledGD: Tong–Ma–Chi 2021, Zhang–Pilanci 2024; LoRA-Pro: Wang et al. 2024. Float64 throughout.' }));

    function applyPreset(k) {
      var p = PRESETS[k];
      Object.keys(p).forEach(function (q) { stA[q] = p[q]; });
      ['th', 't1', 't2', 'h'].forEach(function (q) { SL[q].set(stA[q]); });
      renderA(true);
    }

    /* ---- tour (animation along a path in GL2) ---- */
    var KEYS = [
      [0, 0, 0, 0, 0], [2.2, 90, 0, 0, 0], [2.9, 90, 0, 0, 0], [4.3, 90, 0.55, -0.45, 0], [5.0, 90, 0.55, -0.45, 0],
      [6.4, 90, 0.55, -0.45, 0.5], [7.1, 90, 0.55, -0.45, 0.5], [9.1, 0, 0, 0, 0], [9.7, 0, 0, 0, 0]
    ];
    /* reduced motion: discrete stops, one per region (O(2) \ B2, B2, Mon2 \ B2, GL2 \ (O(2) ∪ Mon2), identity) */
    var STEPS = [[0, 45, 0, 0, 0], [0, 90, 0, 0, 0], [0, 90, 0.55, -0.45, 0], [0, 90, 0.55, -0.45, 0.5], [0, 0, 0, 0, 0]];
    var tourRaf = 0, tourT0 = 0, tourStep = -1;
    function setTourState(k) {
      stA.th = k[1]; stA.t1 = k[2]; stA.t2 = k[3]; stA.h = k[4]; stA.swap = false; stA.flip = false;
      ['th', 't1', 't2', 'h'].forEach(function (q) { SL[q].set(stA[q]); });
    }
    function toggleTour() {
      if (reduce) { tourStep = (tourStep + 1) % STEPS.length; setTourState(STEPS[tourStep]); renderA(true); return; }
      if (tourRaf) { stopTour(); return; }
      bTour.setAttribute('aria-pressed', 'true'); bTour.textContent = '❚❚ pause';
      tourT0 = performance.now();
      var period = KEYS[KEYS.length - 1][0];
      var frame = function (now) {
        var t = ((now - tourT0) / 1000) % period, i = 0;
        while (i < KEYS.length - 2 && KEYS[i + 1][0] <= t) i++;
        var a = KEYS[i], b = KEYS[i + 1], u = (t - a[0]) / Math.max(1e-9, b[0] - a[0]);
        u = u * u * (3 - 2 * u);
        /* snap to the slider grid (1°, 0.05): every frame is a state the sliders can reach, so no frame shows a
           sub-round-off stretch whose tiny but real defect the 10⁻¹⁰ threshold would misread as equivariance */
        var k = [t]; for (var j = 1; j < 5; j++) { var v = a[j] + (b[j] - a[j]) * u; k.push(j === 1 ? Math.round(v) : Math.round(v / 0.05) / 20); }
        if (k[1] !== stA.th || k[2] !== stA.t1 || k[3] !== stA.t2 || k[4] !== stA.h || stA.swap || stA.flip) { setTourState(k); renderA(); }
        tourRaf = requestAnimationFrame(frame);
      };
      tourRaf = requestAnimationFrame(frame);
    }
    function stopTour() {
      if (!tourRaf) return;
      cancelAnimationFrame(tourRaf); tourRaf = 0;
      bTour.setAttribute('aria-pressed', 'false'); bTour.textContent = '▶ tour';
    }

    /* ---- drawing: g glyph + matrix ---- */
    function drawGlyph(g) {
      glyph.selectAll('*').remove();
      var sv = Math.sqrt(Math.max(eig2(mul(tp(g), g)).l1, 1e-12));
      var u = 44 / Math.max(1.35, sv);
      glyph.attr('aria-label', 'The unit circle and its image under g; arrows g e1 and g e2.');
      glyph.append('line').attr('x1', -62).attr('x2', 62).attr('y1', 0).attr('y2', 0).style('stroke', 'var(--rule)');
      glyph.append('line').attr('y1', -62).attr('y2', 62).attr('x1', 0).attr('x2', 0).style('stroke', 'var(--rule)');
      glyph.append('circle').attr('r', u).style('fill', 'none').style('stroke', 'var(--ink-3)').style('stroke-dasharray', '3 3');
      var pts = d3.range(0, 97).map(function (i) { var t = 2 * Math.PI * i / 96, x = Math.cos(t), y = Math.sin(t); return [u * (g[0][0] * x + g[0][1] * y), -u * (g[1][0] * x + g[1][1] * y)]; });
      glyph.append('path').attr('d', d3.line()(pts)).style('fill', 'var(--ochre-soft)').style('stroke', 'var(--ochre)').style('stroke-width', 1.6);
      var defs = glyph.append('defs');
      defs.append('marker').attr('id', uid + '-ah').attr('viewBox', '0 0 8 8').attr('refX', 7).attr('refY', 4).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto-start-reverse')
        .append('path').attr('d', 'M0.5,0.8 L7,4 L0.5,7.2').style('fill', 'none').style('stroke', 'var(--ink)').style('stroke-width', 1.2);
      [[0, 'ge₁'], [1, 'ge₂']].forEach(function (c) {
        var x = u * g[0][c[0]], y = -u * g[1][c[0]];
        glyph.append('line').attr('x1', 0).attr('y1', 0).attr('x2', x).attr('y2', y).attr('marker-end', 'url(#' + uid + '-ah)').style('stroke', 'var(--ink)').style('stroke-width', 1.3);
        /* 14 units: 12.7px rendered at 120px (phones), 13.6px at 128px; the label stays inside the viewBox */
        var L = Math.hypot(x, y) || 1, ox = x / L * 12.5, oy = y / L * 12.5;
        var lx = Math.max(-51, Math.min(51, x + ox)), ly = Math.max(-52, Math.min(62, y + oy + 4.8));
        glyph.append('text').attr('x', lx).attr('y', ly).attr('text-anchor', 'middle').attr('class', 'serif').style('font-size', '14px').style('font-style', 'italic').style('fill', 'var(--ink)').text(c[1]);
      });
    }
    function drawMatrix(g, cls) {
      matEl.innerHTML = '';
      matEl.appendChild(h('span', { class: 'lhs', text: 'g =' }));
      var mb = h('div', { class: 'gl-matb' });
      [g[0][0], g[0][1], g[1][0], g[1][1]].forEach(function (v) { mb.appendChild(h('span', { text: num(v, 3) })); });
      matEl.appendChild(mb);
      var nm = { B: 'g <span class="x">∈</span> B₂', O: 'g <span class="x">∈</span> O(2) ∖ B₂', M: 'g <span class="x">∈</span> Mon₂ ∖ B₂', GL: 'g <span class="x">∉</span> O(2) ∪ Mon₂' }[cls.region];
      memberEl.innerHTML = cls.isId ? 'g = I <span class="x">·</span> the identity chart' : nm;
      var det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
      gmeta.innerHTML = '<span>det g = <b>' + num(det, 3) + '</b></span><span>‖gᵀg − I‖<sub>F</sub> = <b>' + (cls.orthDef < 1e-12 ? '0' : supHTML(magStr(cls.orthDef))) + '</b></span><span>r<sub>j</sub> = ‖g<sub>j,:</sub>‖ = <b>' + num(curA.rn[0], 3) + '</b>, <b>' + num(curA.rn[1], 3) + '</b></span>';
    }

    /* ---- drawing: Venn of subgroups ---- */
    /* the ellipses overlap less than before, so each lune is 116 units wide: room for the rule names at 12px+ */
    var VENN = { O: { cx: 122, cy: 124, rx: 106, ry: 80 }, M: { cx: 238, cy: 124, rx: 106, ry: 80 } };
    var DOT = { B: [180, 168], O: [58, 168], M: [302, 168], GL: [180, 26] };
    var lastDot = null;
    function drawVenn(cls, level, eps, animate) {
      venn.selectAll('*').remove();
      /* font sizes are set in rendered pixels: the viewBox is 360 units wide, so divide by the current scale */
      var vs = (venn.node().getBoundingClientRect().width || 360) / 360;
      var U = function (px) { return Math.min(px * 1.4, px / vs); };
      venn.attr('aria-label', 'Subgroups of GL2: O(2) and the monomial group Mon2 overlap in the signed permutations B2. The current g lies in ' + ({ B: 'B2', O: 'O(2) but not B2', M: 'Mon2 but not B2', GL: 'neither O(2) nor Mon2' }[cls.region]) + '.');
      var defs = venn.append('defs');
      function ell(sel, e) { return sel.append('ellipse').attr('cx', e.cx).attr('cy', e.cy).attr('rx', e.rx).attr('ry', e.ry); }
      var mO = defs.append('mask').attr('id', uid + '-mO'); mO.append('rect').attr('width', 360).attr('height', 236).style('fill', '#fff'); ell(mO, VENN.M).style('fill', '#000');
      var mM = defs.append('mask').attr('id', uid + '-mM'); mM.append('rect').attr('width', 360).attr('height', 236).style('fill', '#fff'); ell(mM, VENN.O).style('fill', '#000');
      var mG = defs.append('mask').attr('id', uid + '-mG'); mG.append('rect').attr('width', 360).attr('height', 236).style('fill', '#fff'); ell(mG, VENN.O).style('fill', '#000'); ell(mG, VENN.M).style('fill', '#000');
      var cO = defs.append('clipPath').attr('id', uid + '-cO'); ell(cO, VENN.O);
      /* highlight the region that contains g */
      var hi = 'var(--ochre-soft)';
      if (cls.region === 'GL') venn.append('rect').attr('x', 4).attr('y', 4).attr('width', 352).attr('height', 228).attr('rx', 6).attr('mask', 'url(#' + uid + '-mG)').style('fill', hi);
      if (cls.region === 'O') ell(venn, VENN.O).attr('mask', 'url(#' + uid + '-mO)').style('fill', hi);
      if (cls.region === 'M') ell(venn, VENN.M).attr('mask', 'url(#' + uid + '-mM)').style('fill', hi);
      if (cls.region === 'B') ell(venn, VENN.M).attr('clip-path', 'url(#' + uid + '-cO)').style('fill', hi);
      venn.append('rect').attr('x', 4).attr('y', 4).attr('width', 352).attr('height', 228).attr('rx', 6).style('fill', 'none').style('stroke', 'var(--ink-3)').style('stroke-width', 1);
      ell(venn, VENN.O).style('fill', 'none').style('stroke', 'var(--ink-2)').style('stroke-width', 1.25);
      ell(venn, VENN.M).style('fill', 'none').style('stroke', 'var(--ink-2)').style('stroke-width', 1.25);
      /* labels: group names */
      function T(x, y, txt, cls2, size, anchor) {
        return venn.append('text').attr('x', x).attr('y', y).attr('text-anchor', anchor || 'middle').attr('class', cls2 || '').style('font-size', (+size.toFixed(2)) + 'px').text(txt);
      }
      /* group names: display serif, at least 15px rendered; rule names: Plex Sans 500 at 12.5px rendered */
      var gN = Math.max(18, U(15)), fL = U(12.5), fS = U(12), lh = fL * 1.3;
      T(16, 14 + gN * 0.8, 'GL₂', 'display', gN + 1, 'start').style('fill', 'var(--ink)').style('font-weight', 600);
      T(74, 92, 'O(2)', 'display', gN).style('fill', 'var(--ink)').style('font-weight', 600);
      T(286, 92, 'Mon₂', 'display', gN).style('fill', 'var(--ink)').style('font-weight', 600);
      T(180, 92, 'B₂', 'display', gN).style('fill', 'var(--ink)').style('font-weight', 600);
      var y8 = 92 + gN * 0.38 + fS * 1.05;
      T(180, y8, '8 elements', '', fS).style('fill', 'var(--ink-2)');
      /* rules by smallest group */
      var lists = { B: [], O: [], M: [], GL: [] };
      OPTS.forEach(function (o) { lists[groupOf(o.key, level, eps)].push(o.short + (o.key === 'lorapro' ? (level === 'W' ? ' δW' : ' step') : '')); });
      function list(arrL, x, y, anchor) {
        arrL.forEach(function (s, i) { T(x, y + i * lh, s, '', fL, anchor).style('fill', 'var(--ink)').style('font-weight', 500); });
      }
      list(lists.O, 74, 92 + lh * 1.25, 'middle');
      list(lists.M.map(function (s) { return s + '*'; }), 286, 92 + lh * 1.25, 'middle');
      list(lists.B, 180, y8 + lh * 1.2, 'middle');
      var glTxt = lists.GL.join(' · ');
      T(180, 226, glTxt, '', fL).style('fill', 'var(--ink)').style('font-weight', 500);
      /* the dot */
      var p = DOT[cls.region];
      var gdot = venn.append('g').attr('transform', 'translate(' + (animate && lastDot && !reduce ? lastDot : p).join(',') + ')');
      gdot.append('circle').attr('r', 9).style('fill', 'var(--ochre-soft)').style('stroke', 'var(--ochre)').style('stroke-width', 1);
      gdot.append('circle').attr('r', 4).style('fill', 'var(--ochre)');
      gdot.append('text').attr('x', 12).attr('y', 5).attr('class', 'serif').style('font-size', (+U(14).toFixed(2)) + 'px').style('font-style', 'italic').style('fill', 'var(--ink)').text('g');
      if (animate && lastDot && !reduce && (lastDot[0] !== p[0] || lastDot[1] !== p[1])) gdot.transition().duration(420).ease(d3.easeCubicOut).attr('transform', 'translate(' + p.join(',') + ')');
      lastDot = p;
    }

    /* ---- drawing: defect bars ---- */
    var tipOwned = false;
    function drawBars(animate) {
      var Wd = barsBox.clientWidth;
      if (!Wd) return;
      /* the bars are rebuilt on every change; a tooltip opened on the old ones would go stale */
      if (tipOwned) { tipOwned = false; A.tip.hide(); }
      var medium = Wd < 600;
      var level = stA.level, eps = stA.eps, cls = curA.cls;
      var rows = OPTS.map(function (o) {
        var v = curA.res[o.key][level], grp = groupOf(o.key, level, eps);
        return { o: o, v: v, grp: grp, ok: v <= TOL, pred: inGroup(cls.region, grp) };
      });
      barsBox.innerHTML = '';
      var svg = d3.select(barsBox).append('svg').attr('width', Wd).attr('role', 'img')
        .attr('aria-label', 'Relative gauge defect of each update rule on a log scale. ' + rows.map(function (r) { return r.o.name + ': ' + defStr(r.v) + (r.ok ? ' (equivariant)' : ' (gauge-dependent)'); }).join('; '));
      /* column widths come from the labels as set (Plex Sans names and chips, Plex Mono values), so they follow the
         page fonts; every candidate string is measured, so the columns do not jump while g moves */
      function tw(txt, cl, size, weight) {
        var t = svg.append('text').attr('class', cl || null).style('font-size', size + 'px').style('font-weight', weight || 400); supSVG(t, txt, size, size >= 13 ? 12 : 11);
        var wv = t.node().getComputedTextLength ? t.node().getComputedTextLength() : txt.length * size * 0.6; t.remove(); return wv;
      }
      var thmOf = function (o) { return medium ? o.thm.replace(/^(Thm|Prop) /, '') : o.thm; };
      var nameW = 0, thmW = 0, chipTW = 0;
      OPTS.forEach(function (o) { nameW = Math.max(nameW, tw(o.name, '', 13, 500)); thmW = Math.max(thmW, tw(thmOf(o), '', 12, 400)); });
      ['B₂', 'O(2)', 'Mon₂*', 'GL₂'].forEach(function (s) { chipTW = Math.max(chipTW, tw(s, '', 12, 500)); });
      var chipW = Math.ceil(chipTW + 14), valTW = tw('8.8 × 10⁻¹⁸', 'num', 13, 500);
      /* wide: values in their own column; medium: the value sits on the name line, above the bar's end */
      var labelW = Math.ceil(Math.max(nameW, chipW + 6 + thmW) + 14), valW = medium ? 0 : Math.ceil(valTW + 12);
      /* too little room for a readable bar beside the labels: stack name above bar, as on a phone */
      var narrow = Wd - labelW - valW - 10 < 150;
      if (narrow) { labelW = 0; valW = 0; }
      var rowH = narrow ? 46 : 44, top = 40, axisH = 26;
      var x0 = labelW, x1 = Wd - valW - (narrow || medium ? 2 : 10);
      var Ht = top + rows.length * rowH + axisH;
      svg.attr('height', Ht);
      var x = d3.scaleLog().domain([FLOOR, CEIL]).range([x0, x1]).clamp(true);
      /* round-off band: one column on wide layouts; on narrow ones a strip behind each bar, so the rule names
         above the bars sit on a clean background */
      var spans = narrow ? rows.map(function (r, i) { return [top + i * rowH + 19, top + i * rowH + 37]; }) : [[top - 4, top + rows.length * rowH]];
      spans.forEach(function (sp) {
        svg.append('rect').attr('x', x0).attr('y', sp[0]).attr('width', x(TOL) - x0).attr('height', sp[1] - sp[0]).style('fill', 'var(--moss-soft)');
        svg.append('line').attr('x1', x(TOL)).attr('x2', x(TOL)).attr('y1', sp[0]).attr('y2', sp[1]).style('stroke', 'var(--moss)').style('stroke-dasharray', '2 3').style('stroke-width', 1);
      });
      /* two header lines: the column heading and the 10⁻¹⁰ mark above, the band's name below */
      if (!narrow) svg.append('text').attr('x', 0).attr('y', 15).style('font-size', '12px').style('font-weight', 500).style('letter-spacing', '.06em').style('fill', 'var(--ink-2)').text(medium ? 'RULE · GROUP · RESULT' : 'RULE · GROUP · THEOREM');
      supSVG(svg.append('text').attr('x', x(TOL)).attr('y', 15).attr('text-anchor', 'middle').attr('class', 'num').style('font-size', '11px').style('fill', 'var(--ink-2)'), '10⁻¹⁰', 11);
      var roW = tw('round-off', '', 12, 500), roIn = roW + 8 <= x(TOL) - x0;
      svg.append('text').attr('x', roIn ? x0 + 4 : x(TOL) - 4).attr('y', 31).attr('text-anchor', roIn ? 'start' : 'end').style('font-size', '12px').style('font-weight', 500).style('fill', 'var(--moss-ink)').text('round-off');
      /* grid ticks: dense tick values in mono at 11px */
      var ticks = narrow || medium ? [1e-16, 1e-12, 1e-8, 1e-4, 1] : [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1];
      /* labelled ticks keep a gap of at least 10px between neighbours */
      var tickW = tw('10⁻¹⁶', 'num', 11, 400), labStep = (x(1e-12) - x(1e-16)) >= tickW + 10 ? 4 : 8;
      var axY = top + rows.length * rowH + 4;
      ticks.forEach(function (t) {
        if (narrow) rows.forEach(function (r, i) { svg.append('line').attr('x1', x(t)).attr('x2', x(t)).attr('y1', top + i * rowH + 19).attr('y2', i === rows.length - 1 ? axY : top + i * rowH + 37).style('stroke', 'var(--rule)').style('stroke-width', 0.75); });
        else svg.append('line').attr('x1', x(t)).attr('x2', x(t)).attr('y1', top - 2).attr('y2', axY).style('stroke', 'var(--rule)').style('stroke-width', 0.75);
        var e = Math.round(Math.log10(t)), major = e % 4 === 0;
        if ((narrow || medium || major) && e % labStep === 0) svg.append('text').attr('x', x(t)).attr('y', axY + 14).attr('text-anchor', narrow && e === 0 ? 'end' : 'middle').attr('class', 'num').style('font-size', '11px').style('fill', 'var(--ink-2)').call(function (sel) { supSVG(sel, e === 0 ? '1' : '10' + sup(e), 11); });
      });
      if (!narrow && !medium) svg.append('text').attr('x', x1).attr('y', axY + 14).attr('dx', 8).style('font-size', '12px').style('fill', 'var(--ink-2)').text('rel.');
      rows.forEach(function (r, i) {
        var y = top + i * rowH, col = r.ok ? 'var(--moss)' : 'var(--seal)';
        var g = svg.append('g').attr('transform', 'translate(0,' + y + ')').style('cursor', 'default');
        g.append('rect').attr('x', 0).attr('y', 0).attr('width', Wd).attr('height', rowH).style('fill', 'transparent')
          .on('mouseenter', function (ev) { tipOwned = true; A.tip.show('<div class="t">' + A.esc(r.o.name) + '</div><div>' + r.o.tip + '</div><div style="margin-top:.3rem;font-family:var(--f-ui);font-weight:500;font-size:.8rem;font-variant-numeric:tabular-nums">defect ' + A.esc(defStr(r.v)) + ' · expected group ' + GROUP_NAME[r.grp] + '</div>', ev); })
          .on('mousemove', function (ev) { A.tip.move(ev); }).on('mouseleave', function () { tipOwned = false; A.tip.hide(); });
        var barY, barH = narrow ? 10 : 12;
        /* the defect number takes the text variant of its colour, which keeps AA contrast at 12px */
        var colT = r.ok ? 'var(--moss-ink)' : 'var(--seal-ink)';
        if (narrow) {
          var nt = g.append('text').attr('x', 0).attr('y', 15).style('font-size', '13px').style('font-weight', 500).style('fill', 'var(--ink)').style('paint-order', 'stroke').style('stroke', 'var(--paper)').style('stroke-width', 3).style('stroke-linejoin', 'round').text(r.o.name);
          nt.append('tspan').style('font-size', '12px').style('font-weight', 400).style('fill', 'var(--ink-2)').text('  ' + GROUP_NAME[r.grp] + (r.grp === 'M' ? '*' : ''));
          g.append('text').attr('x', Wd).attr('y', 15).attr('class', 'num').style('paint-order', 'stroke').style('stroke', 'var(--paper)').style('stroke-width', 3).style('stroke-linejoin', 'round').attr('text-anchor', 'end').style('font-size', '13px').style('fill', colT).style('font-weight', 500).call(function (sel) { supSVG(sel, defStr(r.v), 13, 12); });
          barY = 23;
        } else {
          g.append('text').attr('x', 0).attr('y', 15).style('font-size', '13px').style('font-weight', 500).style('fill', 'var(--ink)').text(r.o.name);
          var chip = g.append('g').attr('transform', 'translate(0,21)');
          chip.append('rect').attr('width', chipW).attr('height', 17).attr('rx', 8.5).style('fill', 'var(--paper-2)').style('stroke', 'var(--rule)');
          chip.append('text').attr('x', chipW / 2).attr('y', 12.6).attr('text-anchor', 'middle').style('font-size', '12px').style('font-weight', 500).style('fill', 'var(--ink)').text(GROUP_NAME[r.grp] + (r.grp === 'M' ? '*' : ''));
          g.append('text').attr('x', chipW + 6).attr('y', 33.6).style('font-size', '12px').style('fill', 'var(--ink-2)').text(thmOf(r.o));
          g.append('text').attr('x', Wd).attr('y', medium ? 13 : 26).attr('text-anchor', 'end').attr('class', 'num').style('paint-order', 'stroke').style('stroke', 'var(--paper)').style('stroke-width', 3).style('stroke-linejoin', 'round').style('font-size', '13px').style('fill', colT).style('font-weight', 500).call(function (sel) { supSVG(sel, defStr(r.v), 13, 12); });
          barY = 16;
        }
        var xv = x(Math.max(r.v, FLOOR));
        g.append('rect').attr('x', x0).attr('y', barY).attr('height', barH).attr('width', Math.max(3, xv - x0)).attr('rx', 1.5).style('fill', col).style('opacity', 0.9);
        if (r.ok !== r.pred) console.warn('[gauge] defect disagrees with the theorem for', r.o.key, r.v, cls.region, r.grp);
      });
      barsCap.textContent = (level === 'W' ? 'first-order δW' : 'parameter step') + ' · Adam ε = ' + (eps === 0 ? '0' : sci(eps, 0).replace('1 × ', ''));
      return rows;
    }

    /* ---- drawing: cometric ellipses ---- */
    var M0 = null;
    function drawEllipses() {
      if (!M0) { var c0 = computeA(baseA, eye(2), 0); M0 = { sgd: c0.Msgd, sc: c0.Msc }; }
      [['sgd', curA.Msgd], ['sc', curA.Msc]].forEach(function (p) {
        var part = ellParts[p[0]], ref = M0[p[0]], M = p[1];
        var e0 = eig2(ref), e1 = eig2(M);
        var same = fro(add(M, ref, -1)) <= TOL * fro(ref);
        var R = 56 / Math.max(e0.l1, e1.l1);
        var svg = part.svg; svg.selectAll('*').remove();
        svg.append('line').attr('x1', -74).attr('x2', 74).style('stroke', 'var(--rule)');
        svg.append('line').attr('y1', -62).attr('y2', 62).style('stroke', 'var(--rule)');
        function path(Mx) {
          return d3.line()(d3.range(0, 129).map(function (i) { var t = 2 * Math.PI * i / 128, c = Math.cos(t), s = Math.sin(t); return [R * (Mx[0][0] * c + Mx[0][1] * s), -R * (Mx[1][0] * c + Mx[1][1] * s)]; }));
        }
        var col = same ? 'var(--moss)' : 'var(--seal)';
        svg.append('path').attr('d', path(M)).style('fill', same ? 'var(--moss-soft)' : 'var(--seal-soft)').style('stroke', col).style('stroke-width', 1.8);
        svg.append('path').attr('d', path(ref)).style('fill', 'none').style('stroke', 'var(--ink-2)').style('stroke-width', 1.1).style('stroke-dasharray', '4 3');
        /* axis names at 12px rendered: the viewBox is 160 units wide */
        var es = (svg.node().getBoundingClientRect().width || 160) / 160, ef = +Math.min(16, 12 / es).toFixed(2);
        svg.append('text').attr('x', 78).attr('y', 62).attr('text-anchor', 'end').style('font-size', ef + 'px').style('font-weight', 500).style('fill', 'var(--ink-2)').text('G₁ →');
        svg.append('text').attr('x', 5).attr('y', -64 + ef * 0.8).style('font-size', ef + 'px').style('font-weight', 500).style('fill', 'var(--ink-2)').text('G₂');
        svg.attr('aria-label', (p[0] === 'sgd' ? 'SGD' : 'ScaledGD') + ' cometric ellipse: semi-axes ' + e1.l1.toFixed(3) + ' and ' + e1.l2.toFixed(3) + ', reference ' + e0.l1.toFixed(3) + ' and ' + e0.l2.toFixed(3));
        var colT = same ? 'var(--moss-ink)' : 'var(--seal-ink)';
        part.rd.innerHTML = 'semi-axes <span class="n" style="color:' + colT + '">' + num(e1.l1, 2) + ' · ' + num(e1.l2, 2) + '</span><br>at g = I: <span class="n">' + num(e0.l1, 2) + ' · ' + num(e0.l2, 2) + '</span>';
      });
    }

    /* ---- verdict ---- */
    function verdictA_(rows) {
      var cls = curA.cls;
      if (cls.isId) { verdictA.innerHTML = 'With g = I both charts coincide, so every defect is exactly 0. Pick a preset or move a slider.'; return; }
      var ok = rows.filter(function (r) { return r.ok; }).map(function (r) { return r.o.short; });
      var no = rows.filter(function (r) { return !r.ok; }).map(function (r) { return r.o.short; });
      var lead = {
        B: 'g is a signed permutation, the finite shadow of the gauge that every rule here respects',
        O: 'g is orthogonal but not a signed permutation. Gradient descent is natural exactly along O(2); Adam’s coordinatewise step is not',
        M: 'g rescales the rank channels. Plain Adam sees it, and per-channel rates absorb it only jointly' + (stA.eps > 0 ? ', with ε rescaled as well' : ''),
        GL: 'g is a generic change of chart, so only rules whose ' + (stA.level === 'W' ? 'δW is a function of W' : 'step is GL₂-equivariant') + ' ignore it'
      }[cls.region];
      var thm = { B: 'Thm III.12(b)', O: 'Thm III.3(b), III.12(a,b)', M: 'Thm III.12(b)', GL: stA.level === 'W' ? 'Prop III.4, Thm III.12(c)' : 'Thm III.12(c); LoRA-Pro’s step: Thm III.12' }[cls.region];
      verdictA.innerHTML = A.esc(lead) + '. <span class="ok">Blind:</span> ' + (ok.length ? A.esc(ok.join(', ')) : 'none') + '. <span class="no">Sees g:</span> ' + (no.length ? A.esc(no.join(', ')) : 'none') + '. <span class="thm">' + thm + '</span>';
    }

    function renderA(animate) {
      var g = gaugeOf(stA);
      curA = computeA(baseA, g, stA.eps);
      bSwap.setAttribute('aria-pressed', String(stA.swap)); bFlip.setAttribute('aria-pressed', String(stA.flip));
      Object.keys(PRESETS).forEach(function (k) {
        var p = PRESETS[k], match = Object.keys(p).every(function (q) { return p[q] === stA[q]; });
        preBtns[k].classList.toggle('cur', match);
      });
      drawGlyph(g); drawMatrix(g, curA.cls);
      drawVenn(curA.cls, stA.level, stA.eps, animate);
      var rows = drawBars(animate);
      if (rows) verdictA_(rows);
      drawEllipses();
    }

    /* ======================================================================
       TAB B — merges
       ====================================================================== */
    var pB = panels.B;
    pB.appendChild(h('p', { class: 'gl-instr', html: 'Turn the knob to re-gauge <b>module 2</b>. Its adapter <i>B</i>₂<i>A</i>₂ never changes, yet one merge rule moves.' }));
    var fB = h('div', { class: 'gl-formula', 'aria-label': 'Merge rules and the gauge' });
    fB.appendChild(formulaItem('task arithmetic', '\\(\\mathrm{TA}=\\sum_i w_iB_iA_i\\)'));
    fB.appendChild(formulaItem('LoraHub', '\\(\\mathrm{LH}=\\big(\\sum_i w_iB_i\\big)\\big(\\sum_j w_jA_j\\big)\\)'));
    fB.appendChild(formulaItem('gauge', '\\((B_2,A_2)\\mapsto(B_2g^{-1},\\,gA_2),\\ \\ g=c^{-1}R_\\varphi\\)'));
    fB.appendChild(formulaItem('change', '\\(\\Delta\\mathrm{LH}=\\sum_{l\\neq2}w_lw_2\\big[B_2(g^{-1}-I)A_l+B_l(g-I)A_2\\big]\\)'));
    pB.appendChild(fB);

    var ctlB = h('div', { class: 'gl-controls' });
    pB.appendChild(ctlB);
    var gK = h('div', { class: 'gl-group' }, [h('div', { class: 'gl-gl' }, [h('span', { class: 'gl-tag', text: 'Gauge knob' }), h('span', { class: 'gl-gform', html: 'φ = 0: (cB₂, A₂/c), Prop V.4' })])]);
    ctlB.appendChild(gK);
    var slB = h('div', { class: 'gl-sliders' });
    gK.appendChild(slB);
    var SLB = {};
    function SLB_FMT(v) { var c = Math.pow(2, v); return (c >= 10 ? c.toFixed(1) : c >= 1 ? c.toFixed(2) : c.toFixed(3)); }
    SLB.lc = slider({ key: 'lc', label: 'c', name: 'scale', min: -3, max: 3, step: 0.05, value: stB.lc, fmt: SLB_FMT, onInput: function (v) { stopSweep(); stB.lc = v; renderB(); } });
    SLB.phi = slider({ key: 'phi', label: 'φ', name: 'rotate', min: -180, max: 180, step: 1, value: stB.phi, fmt: function (v) { return (v < 0 ? '−' : '') + Math.abs(Math.round(v)) + '°'; }, onInput: function (v) { stopSweep(); stB.phi = v; renderB(); } });
    slB.appendChild(SLB.lc.el); slB.appendChild(SLB.phi.el);
    var segTarget = A.seg([{ value: 'm2', label: 'module 2' }, { value: 'all', label: 'all modules' }], stB.target, function (v) { stopSweep(); stB.target = v; renderB(); }, 'Gauge acts on');
    segTarget.el.classList.add('gl-seg');
    gK.appendChild(h('div', { class: 'gl-row', style: 'margin-top:.55rem' }, [h('span', { class: 'gl-k', text: 'Acts on' }), segTarget.el]));

    var gS = h('div', { class: 'gl-group' }, [h('div', { class: 'gl-gl' }, [h('span', { class: 'gl-tag', text: 'Modules' })])]);
    ctlB.appendChild(gS);
    var bShare = pill('share frozen A', function () { stopSweep(); stB.share = !stB.share; renderB(); }, { 'aria-pressed': 'false', title: 'All modules use the same frozen A (FFA-LoRA / LoRA-FA style)' });
    var bSweep = pill(reduce ? 'next c ▸' : '▶ sweep c', function () { toggleSweep(); }, { class: 'gl-pill play', 'aria-pressed': 'false' });
    var bResetB = pill('reset', function () { stopSweep(); stB.lc = 1; stB.phi = 0; stB.target = 'm2'; stB.share = false; SLB.lc.set(1); SLB.phi.set(0); renderB(); });
    gS.appendChild(h('div', { class: 'gl-btnrow', style: 'margin-top:0' }, [bShare, bSweep, bResetB]));
    gS.appendChild(h('div', { class: 'gl-note', html: 'N = 3 modules, LoraHub weights w = (' + WTS.join(', ') + '), unnormalised, Σw = ' + WTS.reduce(function (a, b) { return a + b; }, 0) + '.' }));

    var gridB = h('div', { class: 'gl-grid b' });
    pB.appendChild(gridB);
    var cardF = h('div', { class: 'gl-card' }, [h('div', { class: 'gl-cap' }, [h('span', { class: 'gl-tag', text: 'Module 2 in its fibre' }), h('span', { class: 'gl-sub', text: 'B₂A₂ fixed' })])]);
    gridB.appendChild(cardF);
    var cardM = h('div', { class: 'gl-card' }, [h('div', { class: 'gl-cap' }, [h('span', { class: 'gl-tag', text: 'Two ways to merge three adapters' }), h('span', { class: 'gl-sub', text: 'Prop V.4' })])]);
    gridB.appendChild(cardM);

    function heatmap(parent, m, n, label, sub) {
      var wrap = h('div', { class: 'gl-hm' });
      if (label) wrap.appendChild(h('div', { class: 'gl-hm-l', html: label + (sub ? '<small>' + sub + '</small>' : '') }));
      var svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 ' + n + ' ' + m).attr('preserveAspectRatio', 'xMidYMid meet').attr('role', 'img').attr('shape-rendering', 'crispEdges');
      var cells = [];
      for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) cells.push(svg.append('rect').attr('x', j + 0.04).attr('y', i + 0.04).attr('width', 0.92).attr('height', 0.92).attr('rx', 0.08));
      parent.appendChild(wrap);
      return {
        wrap: wrap, svg: svg,
        update: function (X, s, color, aria) {
          for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) cells[i * n + j].style('fill', color(X[i][j] / s));
          if (aria) svg.attr('aria-label', aria);
        }
      };
    }
    var fib = h('div', { class: 'gl-fib' });
    cardF.appendChild(fib);
    fib.appendChild(h('div', { class: 'gl-corner', text: '' }));
    var hmA2 = heatmap(fib, RK, NB, 'A₂', '2 × 8');
    var hmB2 = heatmap(fib, MB, RK, 'B₂', '');
    var hmP2 = heatmap(fib, MB, NB, 'B₂A₂', '6 × 8');
    var gBox = h('div', { class: 'gl-gB' });
    cardF.appendChild(gBox);
    var gBmat = h('div', { class: 'gl-mat' });
    gBox.appendChild(gBmat);
    var fibRead = h('div', { class: 'gl-mval', style: 'margin-top:.45rem' });
    cardF.appendChild(fibRead);
    var scaleEl = h('div', { class: 'gl-scale', style: 'margin-top:.6rem' });
    cardF.appendChild(scaleEl);

    var merges = h('div', { class: 'gl-merges' });
    cardM.appendChild(merges);
    function mcard(name, rule) {
      var c = h('div', { class: 'gl-mcard' });
      var chip = h('span', { class: 'gl-chip neu', text: '' });
      c.appendChild(h('div', { class: 'gl-mhead' }, [h('span', { class: 'gl-mname', html: name + '<span class="gl-mrule">' + rule + '</span>' }), chip]));
      merges.appendChild(c);
      var hm = heatmap(c, MB, NB, '', '');
      var val = h('div', { class: 'gl-mval' });
      c.appendChild(val);
      return { card: c, chip: chip, hm: hm, val: val };
    }
    /* indices as <sub>: Source Serif 4 has no ⱼ (U+2C7C), so the Unicode form fell back to another face */
    var mTA = mcard('Task arithmetic', 'Σ<sub>i</sub> w<sub>i</sub>B<sub>i</sub>A<sub>i</sub>');
    var mLH = mcard('LoraHub', '(Σ<sub>i</sub> w<sub>i</sub>B<sub>i</sub>)(Σ<sub>j</sub> w<sub>j</sub>A<sub>j</sub>)');
    var mD = mcard('ΔLH', 'LH(g) − LH(I), on LH’s colour scale');
    var zeroD = h('div', { class: 'gl-zero', hidden: true }, [h('span', { text: '≡ 0' })]);
    mD.hm.wrap.appendChild(zeroD);
    var meter = h('div', { class: 'gl-meter' });
    merges.appendChild(meter);
    function mt(k) { var v = h('span', { class: 'v', text: '' }); var kk = h('span', { class: 'k', text: k }); meter.appendChild(h('div', { class: 'gl-mt' }, [kk, v])); return { v: v, k: kk }; }
    var mt1 = mt('‖ΔLH‖ measured'), mt2 = mt('closed form'), mt3 = mt('residual'), mt4 = mt('relative change');

    /* card: the change as a function of the knob */
    var capC = h('span', { class: 'gl-sub', text: '' });
    var cardC = h('div', { class: 'gl-card gl-curve' }, [h('div', { class: 'gl-cap' }, [h('span', { class: 'gl-tag', text: 'How far each merge moves as c turns' }), capC])]);
    var curveBox = h('div', { class: 'gl-cbox' });
    cardC.appendChild(curveBox);
    var legC = h('div', { class: 'gl-legend' });
    cardC.appendChild(legC);
    pB.appendChild(cardC);

    var verdictB = h('p', { class: 'gl-verdict', 'aria-live': 'polite' });
    pB.appendChild(verdictB);
    pB.appendChild(h('p', { class: 'gl-foot', html: 'Setup: N = 3 modules, Bᵢ ∈ ℝ⁶ˣ², Aᵢ ∈ ℝ²ˣ⁸ (seed ' + SEED_B + '), sᵢ = αᵢ/rᵢ = 1. Changes are measured against g = I in float64. Colours use a fixed scale per matrix (tanh of the entry over 1.6 × its g = I RMS), so a churning factor saturates rather than rescales. LoraHub: Huang et al. 2023; task arithmetic: Ilharco et al. 2023. HF’s add_weighted_adapter(…, "linear") is a factor-wise rule of the same kind (Prop V.4(a)).' }));

    var colorFn = null;
    function makeColor() {
      var pos = d3.interpolateLab(A.css('--paper'), A.css('--tide')), neg = d3.interpolateLab(A.css('--paper'), A.css('--c-multiplicative'));
      colorFn = function (v) { var t = Math.tanh(v / 1.6); return t >= 0 ? pos(t) : neg(-t); };
      scaleEl.innerHTML = '';
      scaleEl.appendChild(h('span', { class: 'lab', text: 'colour: entry ÷ its RMS at g = I' }));
      scaleEl.appendChild(h('span', { text: '−3' }));
      var stops = d3.range(0, 11).map(function (i) { var v = -3 + 6 * i / 10; return colorFn(v) + ' ' + (i * 10) + '%'; }).join(',');
      scaleEl.appendChild(h('span', { class: 'bar', style: 'background:linear-gradient(90deg,' + stops + ')' }));
      scaleEl.appendChild(h('span', { text: '+3' }));
    }
    function rms(X) { return Math.sqrt(fro2(X) / (X.length * X[0].length)); }

    var sweepRaf = 0, sweepT0 = 0, sweepStep = 0;
    var SWEEP_STEPS = [1, 2, 3, -1, -2, 0];
    function toggleSweep() {
      if (reduce) { sweepStep = (sweepStep + 1) % SWEEP_STEPS.length; stB.lc = SWEEP_STEPS[sweepStep]; SLB.lc.set(stB.lc); renderB(); return; }
      if (sweepRaf) { stopSweep(); return; }
      bSweep.setAttribute('aria-pressed', 'true'); bSweep.textContent = '❚❚ pause';
      var lc0 = stB.lc, a0 = Math.asin(Math.max(-1, Math.min(1, lc0 / 2)));
      sweepT0 = performance.now();
      var frame = function (now) {
        var t = (now - sweepT0) / 1000;
        stB.lc = 2 * Math.sin(a0 + 2 * Math.PI * t / 6);
        SLB.lc.set(Math.round(stB.lc / 0.05) * 0.05);
        renderB();
        sweepRaf = requestAnimationFrame(frame);
      };
      sweepRaf = requestAnimationFrame(frame);
    }
    function stopSweep() {
      if (!sweepRaf) return;
      cancelAnimationFrame(sweepRaf); sweepRaf = 0;
      bSweep.setAttribute('aria-pressed', 'false'); bSweep.textContent = '▶ sweep c';
    }

    /* ---- drawing: ‖Δ‖ against the knob ---- */
    var curveCache = { phi: null, pts: null };
    var C_LABELS = { '-3': '1/8', '-2': '1/4', '-1': '1/2', '0': '1', '1': '2', '2': '4', '3': '8' };
    function drawCurve() {
      var Wd = curveBox.clientWidth;
      if (!Wd || !curB) return;
      var r = curB, narrow = Wd < 460;
      if (curveCache.phi !== stB.phi) { curveCache.phi = stB.phi; curveCache.pts = curveB(baseB, stB.phi, 240); }
      var ref = curveCache.pts, joint = r.joint;
      var Ht = narrow ? 188 : 204, mL = narrow ? 36 : 46, mR = narrow ? 8 : 16, mT = 24, mB = 38;
      var ymax = d3.max(ref, function (p) { return p[1]; });
      var x = d3.scaleLinear().domain([-3, 3]).range([mL, Wd - mR]);
      var y = d3.scaleLinear().domain([0, ymax * 1.08]).nice(4).range([Ht - mB, mT]);
      curveBox.innerHTML = '';
      var svg = d3.select(curveBox).append('svg').attr('width', Wd).attr('height', Ht).attr('role', 'img')
        .attr('aria-label', 'Frobenius norm of the change of each merge rule as log2 c runs from −3 to 3 at φ = ' + Math.round(stB.phi) + '°. Task arithmetic stays at 0. ' +
          (joint ? 'With a joint gauge LoraHub also stays at 0.' : 'LoraHub follows the closed form of Prop V.4; at the current c it moved by ' + magStr(r.nDLH) + '.'));
      /* grid + axes */
      y.ticks(4).forEach(function (t) {
        svg.append('line').attr('x1', mL).attr('x2', Wd - mR).attr('y1', y(t)).attr('y2', y(t)).style('stroke', 'var(--rule)').style('stroke-width', t === 0 ? 1 : 0.75);
        svg.append('text').attr('x', mL - 6).attr('y', y(t) + 4).attr('text-anchor', 'end').attr('class', 'num').style('font-size', '11px').style('fill', 'var(--ink-2)').text(d3.format('~g')(t));
      });
      d3.range(-3, 4).forEach(function (k) {
        svg.append('line').attr('x1', x(k)).attr('x2', x(k)).attr('y1', Ht - mB).attr('y2', Ht - mB + 4).style('stroke', 'var(--ink-3)');
        svg.append('text').attr('x', x(k)).attr('y', Ht - mB + 16).attr('text-anchor', 'middle').attr('class', 'num').style('font-size', '11px').style('fill', k === 0 ? 'var(--ink)' : 'var(--ink-2)').text(C_LABELS[String(k)]);
      });
      /* axis titles: Plex Sans 500 at 12px */
      svg.append('text').attr('x', Wd - mR).attr('y', Ht - 3).attr('text-anchor', 'end').style('font-size', '12px').style('font-weight', 500).style('fill', 'var(--ink-2)').text('c  (log scale)');
      svg.append('text').attr('x', 0).attr('y', 11).style('font-size', '12px').style('font-weight', 500).style('fill', 'var(--ink-2)').text('‖Δ‖').append('tspan').attr('dy', 3).style('font-size', '12px').text('F');
      /* identity marker */
      svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', mT).attr('y2', Ht - mB).style('stroke', 'var(--ink-3)').style('stroke-width', 0.75).style('stroke-dasharray', '1 3');
      var line = d3.line().x(function (p) { return x(p[0]); }).y(function (p) { return y(p[1]); });
      /* LoraHub: closed form along the knob (module 2 alone), or flat when the gauge is joint */
      if (joint) {
        svg.append('path').attr('d', line(ref)).style('fill', 'none').style('stroke', 'var(--ink-3)').style('stroke-width', 1.2).style('stroke-dasharray', '4 3');
      } else {
        svg.append('path').attr('d', line(ref) + 'L' + x(3) + ',' + y(0) + 'L' + x(-3) + ',' + y(0) + 'Z').style('fill', 'var(--seal-soft)').style('stroke', 'none');
        svg.append('path').attr('d', line(ref)).style('fill', 'none').style('stroke', 'var(--seal)').style('stroke-width', 2);
      }
      /* task arithmetic (and a joint-gauge LoraHub): identically zero */
      svg.append('line').attr('x1', x(-3)).attr('x2', x(3)).attr('y1', y(0)).attr('y2', y(0)).style('stroke', 'var(--moss)').style('stroke-width', joint ? 3 : 2.25);
      /* current knob position + measured values */
      var cx = x(Math.max(-3, Math.min(3, stB.lc)));
      svg.append('line').attr('x1', cx).attr('x2', cx).attr('y1', mT - 2).attr('y2', Ht - mB).style('stroke', 'var(--ochre)').style('stroke-width', 1.25);
      svg.append('circle').attr('cx', cx).attr('cy', y(r.nDTA)).attr('r', 4.5).style('fill', 'var(--moss)').style('stroke', 'var(--paper)').style('stroke-width', 1.5);
      svg.append('circle').attr('cx', cx).attr('cy', y(r.nDLH)).attr('r', 5.5).style('fill', joint ? 'var(--moss)' : 'var(--seal)').style('stroke', 'var(--paper)').style('stroke-width', 1.75);
      /* direct labels */
      var cStr = SLB_FMT(stB.lc);
      var lab = svg.append('text').attr('y', mT - 6).attr('class', 'num').style('font-size', '12px').style('fill', 'var(--ink)')
        .style('paint-order', 'stroke').style('stroke', 'var(--paper)').style('stroke-width', 3).style('stroke-linejoin', 'round');
      supSVG(lab, 'c = ' + cStr + (joint ? '' : ' · ‖ΔLH‖ = ' + magStr(r.nDLH)), 12, 12);
      var lw = lab.node().getComputedTextLength ? lab.node().getComputedTextLength() : 120;
      var lx = Math.max(mL, Math.min(Wd - mR - lw, cx - lw / 2));
      lab.attr('x', lx);
      var ts = { 'font-size': narrow ? '12.5px' : '13px' };
      /* direct series labels: Plex Sans 500, in the text variant of the series colour */
      function serif(xx, yy, txt, col, anchor) {
        var t = svg.append('text').attr('x', xx).attr('y', yy).attr('text-anchor', anchor || 'start').style('font-size', ts['font-size']).style('font-weight', 500).style('fill', col)
          .style('paint-order', 'stroke').style('stroke', 'var(--paper)').style('stroke-width', 3.5).style('stroke-linejoin', 'round').text(txt);
        return t;
      }
      /* task-arithmetic label sits just above the zero line at the end away from the knob, where LoraHub is high */
      var tLeft = stB.lc > 0;
      serif(tLeft ? x(-3) + 6 : x(3) - 6, y(0) - 7, joint ? 'ΔTA ≡ ΔLH ≡ 0' : 'ΔTA ≡ 0', 'var(--moss-ink)', tLeft ? 'start' : 'end');
      /* LoraHub (or its ghost) labelled above its own curve, on the branch away from the knob */
      var useRight = stB.lc < -0.4;
      var at = useRight ? 2.4 : -2.4, kk = Math.round((at + 3) / 6 * (ref.length - 1)), pq = ref[kk];
      var lhLab = serif(x(pq[0]) + (useRight ? -8 : 8), Math.max(mT + 10, y(pq[1]) - 8), joint ? (narrow ? 'module 2 alone' : 'LoraHub, module 2 alone') : 'LoraHub', joint ? 'var(--ink-2)' : 'var(--seal-ink)', useRight ? 'end' : 'start');
      if (joint) lhLab.style('font-size', '12px').style('font-style', 'italic').style('font-weight', 400);
      capC.textContent = 'φ = ' + (stB.phi < 0 ? '−' : '') + Math.abs(Math.round(stB.phi)) + '° · curve: closed form, Prop V.4 · dots: measured';
      legC.innerHTML = '';
      legC.appendChild(h('span', { html: '<i class="ln" style="color:var(--moss)"></i>task arithmetic' + (joint ? ' and LoraHub (joint gauge)' : '') }));
      if (!joint) legC.appendChild(h('span', { html: '<i class="ln" style="color:var(--seal)"></i>LoraHub, closed form of Prop V.4' }));
      else legC.appendChild(h('span', { html: '<i class="ln dash" style="color:var(--ink-3)"></i>LoraHub if module 2 alone were re-gauged (unshared)' }));
      legC.appendChild(h('span', { html: '<i class="ln" style="color:var(--ochre)"></i>current c; dots are the measured ‖Δ‖<sub>F</sub>' }));
    }

    function renderB() {
      if (!colorFn) makeColor();
      curB = computeB(baseB, stB);
      var r = curB;
      bShare.setAttribute('aria-pressed', String(stB.share));
      var segBtns = segTarget.el.querySelectorAll('button');
      segBtns[0].disabled = stB.share;
      segTarget.set(r.joint ? 'all' : 'm2');
      var isId = Math.abs(stB.lc) < 1e-12 && Math.abs(stB.phi) < 1e-12;
      /* fibre heatmaps (fixed scales from g = I) */
      var sB = rms(r.B2ref), sA = rms(r.A2ref), sP = rms(r.P2ref), sT = rms(r.TA0), sL = rms(r.LH0);
      hmB2.update(r.B2, sB, colorFn, 'B2 after the gauge');
      hmA2.update(r.A2, sA, colorFn, 'A2 after the gauge');
      hmP2.update(r.P2, sP, colorFn, 'B2 A2, unchanged');
      gBmat.innerHTML = '';
      gBmat.appendChild(h('span', { class: 'lhs', html: 'g = c⁻¹R<sub>φ</sub> =' }));
      var mbB = h('div', { class: 'gl-matb' });
      [r.g[0][0], r.g[0][1], r.g[1][0], r.g[1][1]].forEach(function (v) { mbB.appendChild(h('span', { text: num(v, 3) })); });
      gBmat.appendChild(mbB);
      fibRead.innerHTML = '‖B₂‖<sub>F</sub> = <b>' + num(r.nB2, 3) + '</b> · ‖A₂‖<sub>F</sub> = <b>' + num(r.nA2, 3) + '</b><br>‖Δ(B₂A₂)‖<sub>F</sub> = <b style="color:var(--moss-ink)">' + supHTML(magStr(r.nDP)) + '</b>' + (r.joint ? ' · gauge on every module' : '');
      /* merges */
      mTA.hm.update(r.TA, sT, colorFn, 'Task arithmetic merge');
      mLH.hm.update(r.LH, sL, colorFn, 'LoraHub merge');
      mD.hm.update(r.dLH, sL, colorFn, 'Change of the LoraHub merge');
      var lhInv = r.nDLH <= TOL * fro(r.LH0);
      mTA.card.className = 'gl-mcard inv';
      mTA.chip.className = 'gl-chip yes'; mTA.chip.textContent = 'invariant';
      mTA.val.innerHTML = '‖ΔTA‖<sub>F</sub> = <b style="color:var(--moss-ink)">' + supHTML(magStr(r.nDTA)) + '</b>';
      mLH.card.className = 'gl-mcard ' + (isId ? '' : lhInv ? 'inv' : 'mov');
      mLH.chip.className = 'gl-chip ' + (isId ? 'neu' : lhInv ? 'yes' : 'no');
      mLH.chip.textContent = isId ? 'g = I' : lhInv ? 'invariant' : 'moves';
      mLH.val.innerHTML = stB.share
        ? '‖LH − (Σw)·TA‖ = <b style="color:var(--moss-ink)">' + supHTML(magStr(r.shareRes)) + '</b>'
        : '‖LH‖<sub>F</sub> = <b>' + num(fro(r.LH), 3) + '</b>';
      mD.card.className = 'gl-mcard';
      mD.chip.className = 'gl-chip ' + (isId ? 'neu' : lhInv ? 'yes' : 'no');
      mD.chip.textContent = isId ? 'g = I' : lhInv ? '= 0, round-off' : '≠ 0';
      zeroD.hidden = !lhInv;
      mD.val.innerHTML = 'max |ΔLH<sub>ij</sub>| = <b>' + supHTML(magStr(d3.max(r.dLH, function (row) { return d3.max(row, Math.abs); }))) + '</b>';
      /* meter */
      mt1.v.innerHTML = supHTML(magStr(r.nDLH)); mt1.v.className = 'v ' + (lhInv ? 'ok' : 'no');
      if (r.joint) {
        mt2.k.textContent = 'closed form'; mt2.v.innerHTML = '0';
      } else {
        mt2.k.textContent = Math.abs(stB.phi) < 1e-12 ? 'closed form, V.4' : 'closed form, any g';
        mt2.v.innerHTML = supHTML(magStr(r.nClosed));
      }
      mt2.v.className = 'v';
      mt3.v.innerHTML = supHTML(magStr(r.resid)); mt3.v.className = 'v ok';
      mt4.v.innerHTML = supHTML(pctStr(r.rel)); mt4.v.className = 'v ' + (lhInv ? 'ok' : 'no');
      drawCurve();
      /* verdict */
      if (stB.share) {
        verdictB.innerHTML = 'With a frozen shared <i>A</i>, the only gauge left acts on every module at once, and LoraHub becomes a function of the adapters. It equals (Σ<sub>j</sub>w<sub>j</sub>)·TA, here to ' + A.esc(magStr(r.shareRes)) + ', so it matches task arithmetic only up to the factor Σ<sub>j</sub>w<sub>j</sub> = ' + r.sumW + ', because LoraHub’s weights are unnormalised. <span class="thm">Prop V.4(b)</span>';
      } else if (r.joint) {
        verdictB.innerHTML = 'One <i>g</i> applied to every module (the diagonal GL₂, which also leaves Poly and MHR unchanged) moves neither rule. The fibre of (<i>B</i><sub>i</sub><i>A</i><sub>i</sub>)<sub>i</sub> is an orbit of the whole product ∏<sub>i</sub>GL₂, though, and a gauge on <b>module 2</b> alone moves LoraHub. <span class="thm">Prop V.4(a)</span>';
      } else if (isId) {
        verdictB.innerHTML = 'g = I: nothing has moved yet. Turn <i>c</i> or φ: B₂ and A₂ change, B₂A₂ does not.';
      } else {
        verdictB.innerHTML = 'Every adapter <i>B</i><sub>i</sub><i>A</i><sub>i</sub> is unchanged (<span class="mono" style="font-size:.9em;white-space:nowrap">‖Δ(B₂A₂)‖ = ' + supHTML(magStr(r.nDP)) + '</span>), so task arithmetic cannot move, and it does not. LoraHub moved by <span class="no">' + A.esc(pctStr(r.rel)) + '</span> because it reads the factors themselves; its change matches the closed form to ' + A.esc(magStr(r.resid)) + '. <span class="thm">Prop V.4(a)</span>';
      }
    }

    /* ---------- boot ---------- */
    setDefFormula();
    A.typeset(fA); A.typeset(fB);
    renderA();
    renderB();
    if ('ResizeObserver' in window) {
      var lastW = 0;
      /* the Venn and the ellipses size their labels from their rendered width, so they follow the bars */
      new ResizeObserver(A.debounce(function () { var w = barsBox.clientWidth; if (w && w !== lastW) { lastW = w; if (curA) { var rows = drawBars(false); if (rows) verdictA_(rows); drawVenn(curA.cls, stA.level, stA.eps, false); drawEllipses(); } } }, 60)).observe(barsBox);
      var lastWC = 0;
      new ResizeObserver(A.debounce(function () { var w = curveBox.clientWidth; if (w && w !== lastWC) { lastWC = w; drawCurve(); } }, 60)).observe(curveBox);
    } else window.addEventListener('resize', A.debounce(function () { if (tab === 'A') renderA(); else renderB(); }, 100));
    A.onTheme(function () { colorFn = null; if (tab === 'A') renderA(); renderB(); });
    /* animation loops stop when the figure leaves the viewport (they would otherwise redraw unseen forever) */
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (!e.isIntersecting) { stopTour(); stopSweep(); } });
      }, { threshold: 0 }).observe(el);
    }
    /* test hook: full-precision state for headless verification */
    el.__gauge = {
      a: function () { return { res: curA.res, cls: curA.cls, g: curA.g, Msgd: curA.Msgd, Msc: curA.Msc, st: Object.assign({}, stA) }; },
      b: function () { return { nDLH: curB.nDLH, nClosed: curB.nClosed, resid: curB.resid, rel: curB.rel, nDTA: curB.nDTA, nDP: curB.nDP, shareRes: curB.shareRes, curve: curveB(baseB, stB.phi, 6), st: Object.assign({}, stB) }; },
      setA: function (o) { stopTour(); Object.keys(o).forEach(function (k) { stA[k] = o[k]; }); ['th', 't1', 't2', 'h'].forEach(function (q) { SL[q].set(stA[q]); }); segLevel.set(stA.level); EPS_OPTS.forEach(function (e, i) { if (e.v === stA.eps) segEps.set(i); }); setDefFormula(); renderA(); },
      setB: function (o) { stopSweep(); Object.keys(o).forEach(function (k) { stB[k] = o[k]; }); SLB.lc.set(stB.lc); SLB.phi.set(stB.phi); renderB(); },
      tab: selectTab
    };
  });
})();
