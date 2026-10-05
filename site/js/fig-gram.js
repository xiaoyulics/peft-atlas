/* Fig. "gram": Gram Lab. What a fine-tuning method can never change (Klein's Erlangen programme for PEFT).
   theory/framework.md, Exposé II.B (repaired statements):
     Thm II.5  complete invariants of the basic actions: W' = WR (R in O(n)) iff K_out = WW^T agrees; W' = PW iff
               K_in = W^T W agrees; two-sided iff the singular values agree; full torus T_m: row lines (+ zero rows);
     Cor II.6  an exactly orthogonal input-side method keeps K_out (norms, angles) and the spectrum. Remarks used here:
               BOFT diag(s) W0 R^T keeps |cos| and keeps cos iff s > 0; output-side OFT keeps K_in and the spectrum;
               HF's default Cayley-Neumann OFT has R = Cay(Q)(I - Q^4), R^T R = (I - Q^4)^2, so nothing is kept exactly;
               quasi-orthogonal qGOFT blocks keep neither K_out nor the cosines;
     Thm II.7  Im HRA_r = (W0 + M_{<=r}) ∩ W0·SO(n) for r even, 2 <= r <= n-2: an image-level meet, not a categorical
               product (fibre product dim d + r^2 = rn + r(r-1)/2); the paired init is an apex, dim S1 = kn - k(k+1)/2;
     Prop II.8 Im DoRA_r = Diag_m·(W0 + M_{<=r}); at B = 0 DoRA is an output torus, Delta W escapes rank r;
     Prop II.9 HiRA W0 ⊙ (J + BA): Z(W0) ⊆ Z(W), rank Delta W <= r rank W0; (IA)^3 keeps row lines while l_i != 0;
               RoAd_1 = (IA)^3 over C: each output pair stays on its complex line C·X and its input Gram scales by alpha^2.
   Toy: W0 in R^{4x4}, entries in Z/4, exactly one zero (row 2, column 3), invertible, distinct singular values.
   Every lamp is a float64 equality test, to 1e-10, between an invariant of W = rho(q) and the same invariant of W0.
   Numerical ranks use a one-sided Jacobi SVD (tiny singular values to ~1e-16 absolute); Jacobians are analytic. */
(function () {
  'use strict';

  /* ---------- fixed data ---------- */
  var TOL = 1e-10;
  var W0 = [[0.5, 0.25, -0.5, 1.0], [-1.0, 1.0, 0.0, -0.5], [-1.0, -0.25, -0.5, 0.5], [-0.5, -1.0, 0.5, 0.25]];
  var A0 = [[0.5, -0.25, 0.75, 0.25], [-0.5, 0.75, 0.25, 0.5]];          // LoRA/HiRA/DoRA A at q0 (rank 2, fixed seed-free)
  var HW = [0.5, -0.25, 0.75, 0.5];                                        // HRA paired init u1 = u2 = HW
  var BD = [0.4, 0.15, 0.1, 0.5, -0.3, 0.2, 0.25, -0.35];                 // demo B (4x2, row-major)
  var OMD = [0.35, -0.2, 0.15, 0.3, -0.25, 0.4];                          // demo skew entries (12,13,14,23,24,34)
  var PAIRS = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  var ZERO0 = [[1, 2]];                                                    // the zero entry of W0 (0-based)

  /* ---------- small dense linear algebra (arrays of arrays, float64) ---------- */
  function zeros(m, n) { var A = new Array(m); for (var i = 0; i < m; i++) { A[i] = new Array(n); for (var j = 0; j < n; j++) A[i][j] = 0; } return A; }
  function eye(n) { var I = zeros(n, n); for (var i = 0; i < n; i++) I[i][i] = 1; return I; }
  function copyM(A) { return A.map(function (r) { return r.slice(); }); }
  function T(A) { var m = A.length, n = A[0].length, B = zeros(n, m); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) B[j][i] = A[i][j]; return B; }
  function mul(A, B) {
    var m = A.length, k = B.length, n = B[0].length, C = zeros(m, n);
    for (var i = 0; i < m; i++) for (var p = 0; p < k; p++) { var a = A[i][p]; if (a === 0) continue; for (var j = 0; j < n; j++) C[i][j] += a * B[p][j]; }
    return C;
  }
  function add(A, B, b) { var m = A.length, n = A[0].length, C = zeros(m, n), s = b == null ? 1 : b; for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) C[i][j] = A[i][j] + s * B[i][j]; return C; }
  function scl(A, s) { return A.map(function (r) { return r.map(function (x) { return x * s; }); }); }
  function had(A, B) { return A.map(function (r, i) { return r.map(function (x, j) { return x * B[i][j]; }); }); }
  function rowScale(v, M) { return M.map(function (r, i) { return r.map(function (x) { return x * v[i]; }); }); }
  function frob(A) { var s = 0; for (var i = 0; i < A.length; i++) for (var j = 0; j < A[i].length; j++) s += A[i][j] * A[i][j]; return Math.sqrt(s); }
  function maxAbs(A) { var s = 0; for (var i = 0; i < A.length; i++) for (var j = 0; j < A[i].length; j++) s = Math.max(s, Math.abs(A[i][j])); return s; }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function Eij(i, j) { var E = zeros(4, 4); E[i][j] = 1; return E; }
  var I4 = eye(4);
  /* A X = B by Gaussian elimination with partial pivoting */
  function solve(Ain, Bin) {
    var n = Ain.length, m = Bin[0].length, M = copyM(Ain), X = copyM(Bin), i, j, k, t;
    for (k = 0; k < n; k++) {
      var p = k, best = Math.abs(M[k][k]);
      for (i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > best) { best = Math.abs(M[i][k]); p = i; }
      if (best === 0) return null;
      if (p !== k) { t = M[k]; M[k] = M[p]; M[p] = t; t = X[k]; X[k] = X[p]; X[p] = t; }
      for (i = k + 1; i < n; i++) {
        var f = M[i][k] / M[k][k];
        if (f === 0) continue;
        for (j = k; j < n; j++) M[i][j] -= f * M[k][j];
        for (j = 0; j < m; j++) X[i][j] -= f * X[k][j];
      }
    }
    for (k = n - 1; k >= 0; k--) for (j = 0; j < m; j++) { var s = X[k][j]; for (i = k + 1; i < n; i++) s -= M[k][i] * X[i][j]; X[k][j] = s / M[k][k]; }
    return X;
  }
  /* singular values, descending, by one-sided (Hestenes) Jacobi on the thinner orientation */
  function svals(Min) {
    var M = Min[0].length > Min.length ? T(Min) : Min, r = M.length, c = M[0].length, cols = [], i, j, p, q, sweep;
    for (j = 0; j < c; j++) { var col = new Float64Array(r); for (i = 0; i < r; i++) col[i] = M[i][j]; cols.push(col); }
    for (sweep = 0; sweep < 80; sweep++) {
      var rotated = false;
      for (p = 0; p < c - 1; p++) for (q = p + 1; q < c; q++) {
        var cp = cols[p], cq = cols[q], al = 0, be = 0, ga = 0;
        for (i = 0; i < r; i++) { al += cp[i] * cp[i]; be += cq[i] * cq[i]; ga += cp[i] * cq[i]; }
        if (ga === 0 || Math.abs(ga) <= 1e-15 * Math.sqrt(al * be)) continue;
        rotated = true;
        var ze = (be - al) / (2 * ga), t = (ze >= 0 ? 1 : -1) / (Math.abs(ze) + Math.sqrt(1 + ze * ze)), cs = 1 / Math.sqrt(1 + t * t), sn = cs * t;
        for (i = 0; i < r; i++) { var x = cp[i], y = cq[i]; cp[i] = cs * x - sn * y; cq[i] = sn * x + cs * y; }
      }
      if (!rotated) break;
    }
    return cols.map(function (cl) { var s = 0; for (var k = 0; k < cl.length; k++) s += cl[k] * cl[k]; return Math.sqrt(s); }).sort(function (a, b) { return b - a; });
  }
  /* Cholesky K = L L^T (lower), tolerant of semidefinite input (zero pivots give zero columns) */
  function chol(K) {
    var n = K.length, L = zeros(n, n), sc = 0, i, j, k;
    for (i = 0; i < n; i++) sc = Math.max(sc, K[i][i]);
    for (j = 0; j < n; j++) {
      var d = K[j][j];
      for (k = 0; k < j; k++) d -= L[j][k] * L[j][k];
      d = d > 1e-14 * sc ? Math.sqrt(d) : 0;
      L[j][j] = d;
      for (i = j + 1; i < n; i++) { var s = K[i][j]; for (k = 0; k < j; k++) s -= L[i][k] * L[j][k]; L[i][j] = d > 0 ? s / d : 0; }
    }
    return L;
  }

  /* ---------- building blocks of the method maps ---------- */
  function skewFrom(q, off) { var S = zeros(4, 4); for (var k = 0; k < 6; k++) { var a = PAIRS[k][0], b = PAIRS[k][1]; S[a][b] = q[off + k]; S[b][a] = -q[off + k]; } return S; }
  function skewE(k) { var S = zeros(4, 4); S[PAIRS[k][0]][PAIRS[k][1]] = 1; S[PAIRS[k][1]][PAIRS[k][0]] = -1; return S; }
  /* exact Cayley map (I + S)(I - S)^{-1} = (I - S)^{-1}(I + S) */
  function cayley(S) { return solve(add(I4, S, -1), add(I4, S, 1)); }
  /* HF PEFT default: 5-term Cayley-Neumann series R = I + 2Q + 2Q^2 + 2Q^3 + Q^4 = Cay(Q)(I - Q^4) */
  function cnMap(Q) { var Q2 = mul(Q, Q), Q3 = mul(Q2, Q), Q4 = mul(Q3, Q); return add(add(add(add(I4, Q, 2), Q2, 2), Q3, 2), Q4, 1); }
  function rot2c(w) { var d = 1 + w * w; return [[(1 - w * w) / d, -2 * w / d], [2 * w / d, (1 - w * w) / d]]; }   // Cay of [[0,-w],[w,0]]
  function drot2c(w) { var d = (1 + w * w) * (1 + w * w); return [[-4 * w / d, -2 * (1 - w * w) / d], [2 * (1 - w * w) / d, -4 * w / d]]; }
  function embed(pairs, blocks) {
    var B = zeros(4, 4);
    pairs.forEach(function (pr, k) { var a = pr[0], b = pr[1], m = blocks[k]; B[a][a] = m[0][0]; B[a][b] = m[0][1]; B[b][a] = m[1][0]; B[b][b] = m[1][1]; });
    return B;
  }
  var BF1 = [[0, 1], [2, 3]], BF2 = [[0, 2], [1, 3]];                     // butterfly factors B1, B2 (block size 2)
  function bfly(om) { var B1 = embed(BF1, [rot2c(om[0]), rot2c(om[1])]), B2 = embed(BF2, [rot2c(om[2]), rot2c(om[3])]); return { B1: B1, B2: B2, R: mul(B2, B1) }; }
  function rot(t) { return [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]]; }
  function drot(t) { return [[-Math.sin(t), -Math.cos(t)], [Math.cos(t), -Math.sin(t)]]; }
  function hh(u) { var n2 = dot(u, u), H = eye(4); for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) H[i][j] -= 2 * u[i] * u[j] / n2; return H; }
  function dhh(u, a) {
    var n2 = dot(u, u), ua = dot(u, a), D = zeros(4, 4);
    for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) D[i][j] = -2 * (a[i] * u[j] + u[i] * a[j]) / n2 + 4 * ua * u[i] * u[j] / (n2 * n2);
    return D;
  }
  function ek(k) { var e = [0, 0, 0, 0]; e[k] = 1; return e; }
  function loraBA(q, off) {
    var B = [[q[off], q[off + 1]], [q[off + 2], q[off + 3]], [q[off + 4], q[off + 5]], [q[off + 6], q[off + 7]]];
    var A = [q.slice(off + 8, off + 12), q.slice(off + 12, off + 16)];
    return { B: B, A: A, BA: mul(B, A) };
  }
  /* directions dV = dB A + B dA for every scalar entry, in the flat order (B row-major, then A row-major) */
  function loraDirs(q, off) {
    var f = loraBA(q, off), out = [], i, k, j, D;
    for (i = 0; i < 4; i++) for (k = 0; k < 2; k++) { D = zeros(4, 4); D[i] = f.A[k].slice(); out.push(D); }
    for (k = 0; k < 2; k++) for (j = 0; j < 4; j++) { D = zeros(4, 4); for (i = 0; i < 4; i++) D[i][j] = f.B[i][k]; out.push(D); }
    return out;
  }
  function allZero(q, a, b) { for (var i = a; i < b; i++) if (q[i] !== 0) return false; return true; }
  /* Unicode subscript digits -> <sub> in serif text (the body serif draws U+2080 like a small "o"); mono text keeps them */
  var SUPR = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };
  function hsub(t) {
    return String(t).replace(/[₀-₉]+/g, function (m) { return '<sub>' + m.replace(/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 8320); }) + '</sub>'; })
      .replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, function (m) { return '<sup>' + m.replace(/./g, function (c) { return SUPR[c]; }) + '</sup>'; });   // the serif's superscript minus is a speck
  }
  /* the same for SVG text: subscript / superscript runs become shifted tspans set in the text's own font */
  function tsub(t) {
    return String(t).replace(/[₀-₉]+/g, function (m) { return '<tspan baseline-shift="-0.3em" style="font-size:72%;font-style:normal">' + m.replace(/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 8320); }) + '</tspan>'; })
      .replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, function (m) { return '<tspan baseline-shift="0.45em" style="font-size:78%">' + m.replace(/./g, function (c) { return SUPR[c]; }) + '</tspan>'; });
  }
  var SUBD = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
  function sub(n) { return String(n).split('').map(function (ch) { return SUBD[ch] || ch; }).join(''); }
  /* first index violating a sign condition on q[off..off+3], or -1 */
  function firstBad(q, off, len, pred) { for (var i = 0; i < len; i++) if (!pred(q[off + i])) return i; return -1; }
  /* row scalings diag(s) W: cos_ij -> sign(s_i s_j) cos_ij (Cor II.6 remark). |cos| survives while every s_i != 0;
     cos survives iff, in addition, all s_i share one sign (necessary too, since no cosine of W0 vanishes). */
  function signGuar(q, off, len, sym) {
    var z = firstBad(q, off, len, function (v) { return v !== 0; });
    if (z >= 0) { var w = no(sym + sub(z + 1) + ' = 0'); return { abs: w, cos: w, z: z, flip: -1 }; }
    var f = -1;
    for (var i = 1; i < len; i++) if ((q[off + i] > 0) !== (q[off] > 0)) { f = i; break; }
    return { abs: ok(), cos: f < 0 ? ok() : no(sym + sub(1) + sym + sub(f + 1) + ' < 0'), z: -1, flip: f };
  }

  /* ---------- the eleven methods (OFT carries a Cayley-Neumann switch) ---------- */
  var ROW_NORM0 = W0.map(function (r) { return Math.sqrt(dot(r, r)); });   // = (1.25, 1.5, 1.25, 1.25) exactly
  function ok() { return { ok: true }; }
  function no(why) { return { ok: false, why: why }; }
  var METHODS = [
    {
      id: 'oft', label: 'OFT', group: 0, name: 'OFT', p: 6,
      blocks: [{ type: 'skew', label: 'Ω', off: 0, min: -3, max: 3, mode: 'add', amp: 0.3, spread: 0.6, cap: 'skew: Ωᵀ = −Ω' }],
      q0: [0, 0, 0, 0, 0, 0], demo: OMD,
      R: function (q, f) { var S = skewFrom(q, 0); return f.cn ? cnMap(S) : cayley(S); },
      map: function (q, f) { return mul(W0, T(this.R(q, f))); },
      dR: function (q, f) {
        var S = skewFrom(q, 0), out = [], k;
        if (!f.cn) {
          var Xi = solve(add(I4, S, -1), I4), Rm = mul(Xi, add(I4, S, 1)), IR = add(I4, Rm, 1);
          for (k = 0; k < 6; k++) out.push(mul(mul(IR, skewE(k)), Xi));          // dR = (I + R) dΩ (I - Ω)^{-1}
        } else {
          var Q = S, Q2 = mul(Q, Q), Q3 = mul(Q2, Q);
          for (k = 0; k < 6; k++) {
            var D = skewE(k);
            var t2 = add(mul(D, Q), mul(Q, D));
            var t3 = add(add(mul(D, Q2), mul(mul(Q, D), Q)), mul(Q2, D));
            var t4 = add(add(add(mul(D, Q3), mul(mul(Q, D), Q2)), mul(mul(Q2, D), Q)), mul(Q3, D));
            out.push(add(add(add(scl(D, 2), t2, 2), t3, 2), t4, 1));
          }
        }
        return out;
      },
      jac: function (q, f) { return this.dR(q, f).map(function (dR) { return mul(W0, T(dR)); }); },
      guar: function (q, f) {
        if (f.cn) { var w = no('Cayley–Neumann'); return { Kout: w, cos: w, abscos: w, spec: w }; }
        return { Kout: ok(), cos: ok(), abscos: ok(), spec: ok() };
      },
      tex: function (f) {
        return f.cn ? '\\(W=W_0R^{\\top},\\)  \\(R=I+2Q+2Q^2+2Q^3+Q^4\\)  \\(=\\mathrm{Cay}(Q)\\,(I-Q^4)\\)'
          : '\\(W=W_0R^{\\top},\\)  \\(R=\\mathrm{Cay}(\\Omega)=(I+\\Omega)(I-\\Omega)^{-1}\\)';
      },
      theory: function (f) {
        return f.cn ? '<b>HF PEFT default</b> since 0.18 (<code>use_cayley_neumann=True</code>). <span class="nw"><i>R</i>ᵀ<i>R</i> = (<i>I</i> − <i>Q</i>⁴)² ≠ <i>I</i></span>: quasi-orthogonal, so nothing is kept exactly, already before any merge, and merging compounds the drift <span class="gl-ref">Cor II.6</span>'
          : 'Exact Cayley (the OFT paper; HF with <code>use_cayley_neumann=False</code>). Image ⊂ <i>W</i>₀·SO(4): for every Ω the neuron Gram, the cosines and σ(<i>W</i>) are fixed <span class="gl-ref">Cor II.6</span>';
      },
      title: function (f) { return f.cn ? 'OFT · HF default (Cayley–Neumann)' : 'OFT · exact Cayley, input side'; }
    },
    {
      id: 'out', label: 'OFT · out', group: 0, name: 'output-side OFT', p: 6,
      blocks: [{ type: 'skew', label: 'Ω', off: 0, min: -3, max: 3, mode: 'add', amp: 0.3, spread: 0.6, cap: 'skew: Ωᵀ = −Ω' }],
      q0: [0, 0, 0, 0, 0, 0], demo: OMD,
      map: function (q) { return mul(T(cayley(skewFrom(q, 0))), W0); },
      jac: function (q) { return METHODS[0].dR(q, { cn: false }).map(function (dR) { return mul(T(dR), W0); }); },
      guar: function () { return { Kin: ok(), spec: ok() }; },
      tex: function () { return '\\(W=R^{\\top}W_0,\\)  \\(R=\\mathrm{Cay}(\\Omega)\\)'; },
      theory: function () { return 'Output side (LyCORIS diag-OFT at its defaults; HF PEFT ≤ 0.13 merged to <i>R</i>ᵀ<i>W</i>₀). Image ⊂ O(4)·<i>W</i>₀: <i>K</i><sub>in</sub> and σ(<i>W</i>) are fixed, <i>K</i><sub>out</sub> is not <span class="gl-ref">Thm II.5(2)–(3)</span>'; },
      title: function () { return 'OFT · output side'; }
    },
    {
      id: 'boft', label: 'BOFT', group: 0, name: 'BOFT', p: 8,
      blocks: [
        { type: 'mat', label: 's', off: 0, rows: 4, cols: 1, min: -2, max: 2, mode: 'mul', amp: 0.22, spread: 0.45, cap: 'output scale' },
        { type: 'mat', label: 'ω', off: 4, rows: 2, cols: 2, min: -3, max: 3, mode: 'add', amp: 0.3, spread: 0.6, names: ['ω(1,2)', 'ω(3,4)', 'ω(1,3)', 'ω(2,4)'], cap: 'butterfly: B₁ on (12)(34), B₂ on (13)(24)' }
      ],
      q0: [1, 1, 1, 1, 0, 0, 0, 0], demo: [1.3, 0.75, 1.15, 0.9, 0.3, -0.25, 0.2, 0.35],
      map: function (q) { return rowScale(q.slice(0, 4), mul(W0, T(bfly(q.slice(4, 8)).R))); },
      jac: function (q) {
        var s = q.slice(0, 4), om = q.slice(4, 8), P = bfly(om), WR = mul(W0, T(P.R)), out = [], i, k;
        for (i = 0; i < 4; i++) { var D = zeros(4, 4); D[i] = WR[i].slice(); out.push(D); }
        for (k = 0; k < 4; k++) {
          var dB = k < 2 ? embed([BF1[k]], [drot2c(om[k])]) : embed([BF2[k - 2]], [drot2c(om[k])]);
          var dR = k < 2 ? mul(P.B2, dB) : mul(dB, P.B1);
          out.push(rowScale(s, mul(W0, T(dR))));
        }
        return out;
      },
      guar: function (q) { var g = signGuar(q, 0, 4, 's'); return { abscos: g.abs, cos: g.cos }; },
      tex: function () { return '\\(W=\\mathrm{diag}(s)\\,W_0R^{\\top},\\)  \\(R=B_2B_1\\)  <span class="gl-tx">exact Cayley 2×2 blocks</span>'; },
      theory: function () { return 'HF BOFT, block size 2, two butterfly factors. <span class="nw"><i>K</i><sub>out</sub> ↦ diag(<i>s</i>)<i>K</i><sub>out</sub>diag(<i>s</i>)</span>, so cos<sub><i>ij</i></sub> picks up the sign of <i>s</i><sub><i>i</i></sub><i>s</i><sub><i>j</i></sub>. |cos| is kept while every <i>s</i><sub><i>i</i></sub> ≠ 0, and cos while all <i>s</i><sub><i>i</i></sub> are nonzero with one sign <span class="gl-ref">Cor II.6</span>'; },
      title: function () { return 'BOFT · butterfly OFT with an output scale'; }
    },
    {
      id: 'hra', label: 'HRA', group: 0, name: 'HRA', p: 8,
      blocks: [
        { type: 'mat', label: 'u₁', off: 0, rows: 4, cols: 1, min: -2, max: 2, mode: 'add', amp: 0.35, spread: 1, center: 'zero' },
        { type: 'mat', label: 'u₂', off: 4, rows: 4, cols: 1, min: -2, max: 2, mode: 'add', amp: 0.35, spread: 1, center: 'zero' }
      ],
      q0: HW.concat(HW), demo: HW.concat([0.25, 0.5, 0.5, -0.5]),
      check: function (q) {
        if (dot(q.slice(0, 4), q.slice(0, 4)) < 1e-12) return 'u₁ = 0, and H_u needs u ≠ 0.';
        if (dot(q.slice(4, 8), q.slice(4, 8)) < 1e-12) return 'u₂ = 0, and H_u needs u ≠ 0.';
        return null;
      },
      map: function (q) { return mul(mul(W0, hh(q.slice(0, 4))), hh(q.slice(4, 8))); },
      jac: function (q) {
        var u1 = q.slice(0, 4), u2 = q.slice(4, 8), H1 = hh(u1), H2 = hh(u2), WH1 = mul(W0, H1), out = [], k;
        for (k = 0; k < 4; k++) out.push(mul(mul(W0, dhh(u1, ek(k))), H2));
        for (k = 0; k < 4; k++) out.push(mul(WH1, dhh(u2, ek(k))));
        return out;
      },
      guar: function () { return { Kout: ok(), cos: ok(), abscos: ok(), spec: ok(), rank: ok() }; },
      tex: function () { return '\\(W=W_0H_{u_1}H_{u_2},\\)  \\(H_u=I-2uu^{\\top}/\\lVert u\\rVert^2\\)'; },
      theory: function () { return 'HF HRA, <i>r</i> = 2, <code>apply_GS=False</code>, paired init <i>u</i>₁ = <i>u</i>₂. <span class="nw">Im HRA₂ = (<i>W</i>₀ + 𝓜<sub>≤2</sub>) ∩ <i>W</i>₀·SO(4)</span>: Gram, cosines, spectrum <em>and</em> rank Δ<i>W</i> ≤ 2 <span class="gl-ref">Thm II.7(a)</span>'; },
      title: function () { return 'HRA · two Householder reflections'; }
    },
    {
      id: 'ia3', label: '(IA)³', group: 1, name: '(IA)³', p: 4,
      blocks: [{ type: 'mat', label: 'ℓ', off: 0, rows: 4, cols: 1, min: -2, max: 2, mode: 'mul', amp: 0.3, spread: 0.6, cap: 'per-neuron gain' }],
      q0: [1, 1, 1, 1], demo: [1.4, 0.6, 1.2, 0.8],
      map: function (q) { return rowScale(q, W0); },
      jac: function () { var out = []; for (var i = 0; i < 4; i++) { var D = zeros(4, 4); D[i] = W0[i].slice(); out.push(D); } return out; },
      guar: function (q) { var g = signGuar(q, 0, 4, 'ℓ'); return { zeros: ok(), lines: g.abs, abscos: g.abs, cos: g.cos }; },
      tex: function () { return '\\(W=\\mathrm{diag}(\\ell)\\,W_0\\)'; },
      theory: function () { return 'Output torus <i>T</i>₄. While every ℓ<sub><i>i</i></sub> ≠ 0 each neuron stays on its own line <span class="gl-ref">Thm II.5(4)</span>, and the cosines survive while the ℓ<sub><i>i</i></sub> share one sign; the zeros of <i>W</i>₀ stay zero for every ℓ <span class="gl-ref">Prop II.9(e)</span>'; },
      title: function () { return '(IA)³ · output gains'; }
    },
    {
      id: 'road', label: 'RoAd', group: 1, name: 'RoAd₁', p: 4,
      blocks: [
        { type: 'mat', label: 'θ', off: 0, rows: 2, cols: 1, min: -3.14, max: 3.14, mode: 'add', amp: 0.5, spread: 1.2, cap: 'angles' },
        { type: 'mat', label: 'α', off: 2, rows: 2, cols: 1, min: -2, max: 2, mode: 'mul', amp: 0.25, spread: 0.5, cap: 'scales' }
      ],
      q0: [0, 0, 1, 1], demo: [0.6, -0.4, 1.3, 0.8],
      map: function (q) {
        var W = copyM(W0);
        for (var p = 0; p < 2; p++) { var a = 2 * p, b = a + 1, Rm = scl(rot(q[p]), q[2 + p]); for (var j = 0; j < 4; j++) { W[a][j] = Rm[0][0] * W0[a][j] + Rm[0][1] * W0[b][j]; W[b][j] = Rm[1][0] * W0[a][j] + Rm[1][1] * W0[b][j]; } }
        return W;
      },
      jac: function (q) {
        var out = [];
        function dir(p, Rm) { var D = zeros(4, 4), a = 2 * p, b = a + 1; for (var j = 0; j < 4; j++) { D[a][j] = Rm[0][0] * W0[a][j] + Rm[0][1] * W0[b][j]; D[b][j] = Rm[1][0] * W0[a][j] + Rm[1][1] * W0[b][j]; } return D; }
        out.push(dir(0, scl(drot(q[0]), q[2]))); out.push(dir(1, scl(drot(q[1]), q[3])));
        out.push(dir(0, rot(q[0]))); out.push(dir(1, rot(q[1])));
        return out;
      },
      guar: function (q) { var z = firstBad(q, 2, 2, function (v) { return v !== 0; }); return { clines: z < 0 ? ok() : no('α' + sub(z + 1) + ' = 0') }; },
      tex: function () { return '\\(W=\\begin{pmatrix}\\alpha_1R_{\\theta_1}&0\\\\0&\\alpha_2R_{\\theta_2}\\end{pmatrix}W_0\\)'; },
      theory: function () { return 'RoAd₁ (HF <code>road_1</code>) on output pairs (1,2), (3,4); HF pairs <i>j</i> with <i>j</i> + <i>g</i>/2, which changes nothing here. It is (IA)³ over ℂ: each pair stays on its complex line ℂ·<i>X</i> while α<sub><i>i</i></sub> ≠ 0, and its input Gram scales by α<sub><i>i</i></sub>² <span class="gl-ref">Prop II.9(d)</span>'; },
      title: function () { return 'RoAd₁ · scaled rotations of output pairs'; }
    },
    {
      id: 'dora', label: 'DoRA', group: 1, name: 'DoRA', p: 20,
      blocks: [
        { type: 'mat', label: 'μ', off: 0, rows: 4, cols: 1, min: -3, max: 3, mode: 'mul', amp: 0.25, spread: 0.3, cap: 'magnitude' },
        { type: 'mat', label: 'B', off: 4, rows: 4, cols: 2, min: -2, max: 2, mode: 'add', amp: 0.25, spread: 0.5, center: 'zero' },
        { type: 'mat', label: 'A', off: 12, rows: 2, cols: 4, min: -2, max: 2, mode: 'add', amp: 0.25, spread: 0.4 }
      ],
      q0: ROW_NORM0.concat([0, 0, 0, 0, 0, 0, 0, 0], A0[0], A0[1]), demo: [1.4, 1.3, 1.1, 1.25].concat(BD, A0[0], A0[1]),
      V: function (q) { return add(W0, loraBA(q, 4).BA); },
      check: function (q) { var V = this.V(q); for (var i = 0; i < 4; i++) if (Math.sqrt(dot(V[i], V[i])) < 1e-9) return 'row ' + (i + 1) + ' of W₀ + BA vanished, so N(·) is undefined.'; return null; },
      map: function (q) { var V = this.V(q); return rowScale(V.map(function (r, i) { return q[i] / Math.sqrt(dot(r, r)); }), V); },
      jac: function (q) {
        var V = this.V(q), nu = V.map(function (r) { return Math.sqrt(dot(r, r)); }), out = [], i;
        for (i = 0; i < 4; i++) { var D = zeros(4, 4); D[i] = V[i].map(function (x) { return x / nu[i]; }); out.push(D); }
        loraDirs(q, 4).forEach(function (dV) {
          var D = zeros(4, 4);
          for (var r = 0; r < 4; r++) { var n = V[r].map(function (x) { return x / nu[r]; }), c = dot(n, dV[r]); for (var j = 0; j < 4; j++) D[r][j] = q[r] / nu[r] * (dV[r][j] - c * n[j]); }
          out.push(D);
        });
        return out;
      },
      guar: function (q) {
        if (!allZero(q, 4, 12)) return {};
        var g = signGuar(q, 0, 4, 'μ');
        return { zeros: ok(), lines: g.abs, abscos: g.abs, cos: g.cos };
      },
      tex: function () { return '\\(W=\\mathrm{diag}(\\mu)\\,N(W_0+BA),\\)  <span class="gl-tx">N = row normalisation</span>'; },
      theory: function () { return 'HF DoRA, <i>r</i> = 2, μ per output neuron (<span class="nw">μ = ‖<i>w</i><sub>0<i>i</i></sub>‖</span> at <i>q</i>₀). <span class="nw">Im DoRA₂ = Diag₄·(<i>W</i>₀ + 𝓜<sub>≤2</sub>)</span>: LoRA₂ followed by an output scale. At <i>B</i> = 0 it is an output torus; for <i>B</i> ≠ 0 nothing here is guaranteed <span class="gl-ref">Prop II.8</span>'; },
      title: function () { return 'DoRA · LoRA₂ then an output scale'; }
    },
    {
      id: 'hira', label: 'HiRA', group: 1, name: 'HiRA', p: 16,
      blocks: [
        { type: 'mat', label: 'B', off: 0, rows: 4, cols: 2, min: -2, max: 2, mode: 'add', amp: 0.25, spread: 0.5, center: 'zero' },
        { type: 'mat', label: 'A', off: 8, rows: 2, cols: 4, min: -2, max: 2, mode: 'add', amp: 0.25, spread: 0.4 }
      ],
      q0: [0, 0, 0, 0, 0, 0, 0, 0].concat(A0[0], A0[1]), demo: BD.concat(A0[0], A0[1]),
      map: function (q) { var BA = loraBA(q, 0).BA; return had(W0, BA.map(function (r) { return r.map(function (x) { return 1 + x; }); })); },
      jac: function (q) { return loraDirs(q, 0).map(function (dV) { return had(W0, dV); }); },
      guar: function () { return { zeros: ok() }; },
      tex: function () { return '\\(W=W_0\\odot(J+BA)\\)  \\(=W_0+W_0\\odot BA\\)'; },
      theory: function () { return 'HF HiRA on a linear layer, <i>r</i> = 2, <i>B</i>₀ = 0. Zeros of <i>W</i>₀ stay zero <span class="gl-ref">Prop II.9(e)</span>, but the update is not low-rank: <span class="nw">rank Δ<i>W</i> ≤ <i>r</i>·rank <i>W</i>₀ = 8</span> <span class="gl-ref">Prop II.9(a)</span>'; },
      title: function () { return 'HiRA · Hadamard low-rank'; }
    },
    {
      id: 'lora', label: 'LoRA', group: 2, name: 'LoRA', p: 16,
      blocks: [
        { type: 'mat', label: 'B', off: 0, rows: 4, cols: 2, min: -2, max: 2, mode: 'add', amp: 0.25, spread: 0.5, center: 'zero' },
        { type: 'mat', label: 'A', off: 8, rows: 2, cols: 4, min: -2, max: 2, mode: 'add', amp: 0.25, spread: 0.4 }
      ],
      q0: [0, 0, 0, 0, 0, 0, 0, 0].concat(A0[0], A0[1]), demo: BD.concat(A0[0], A0[1]),
      map: function (q) { return add(W0, loraBA(q, 0).BA); },
      jac: function (q) { return loraDirs(q, 0); },
      guar: function () { return { rank: ok() }; },
      tex: function () { return '\\(W=W_0+BA,\\)  \\(B\\in\\mathbb R^{4\\times2},\\ A\\in\\mathbb R^{2\\times4}\\)'; },
      theory: function () { return 'LoRA, <i>r</i> = 2, scale α/<i>r</i> = 1, <i>B</i>₀ = 0. Im = <i>W</i>₀ + 𝓜<sub>≤2</sub>: rank Δ<i>W</i> ≤ 2 is the only lamp it guarantees.'; },
      title: function () { return 'LoRA · additive low-rank'; }
    },
    {
      id: 'qgoft', label: 'qGOFT', group: 2, name: 'qGOFT', p: 8,
      blocks: [
        { type: 'mat', label: 'M₁', off: 0, rows: 2, cols: 2, min: -2, max: 2, mode: 'add', amp: 0.2, spread: 0.4, cap: 'inputs (1,2)' },
        { type: 'mat', label: 'M₂', off: 4, rows: 2, cols: 2, min: -2, max: 2, mode: 'add', amp: 0.2, spread: 0.4, cap: 'inputs (3,4)' }
      ],
      q0: [1, 0, 0, 1, 1, 0, 0, 1], demo: [1.15, 0.35, -0.2, 0.9, 0.95, -0.3, 0.25, 1.1],
      G: function (q) { return embed([[0, 1], [2, 3]], [[[q[0], q[1]], [q[2], q[3]]], [[q[4], q[5]], [q[6], q[7]]]]); },
      map: function (q) { return mul(W0, T(this.G(q))); },
      jac: function () {
        var out = [];
        for (var b = 0; b < 2; b++) for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++) out.push(mul(W0, T(Eij(2 * b + i, 2 * b + j))));
        return out;
      },
      guar: function () { var w = no('quasi-orthogonal'); return { Kout: w, cos: w }; },
      tex: function () { return '\\(W=W_0G^{\\top},\\)  \\(G=\\mathrm{diag}(M_1,M_2),\\ M_i\\in GL_2\\)'; },
      theory: function () { return 'qGOFT layer: free 2×2 blocks, held near O(2) only by a soft penalty (off here). Once <i>n</i> &gt; 2, one non-orthogonal block, even a scaled rotation, generically moves a cosine; neither <i>K</i><sub>out</sub> nor the cosines are kept <span class="gl-ref">Cor II.6</span>'; },
      title: function () { return 'qGOFT · quasi-orthogonal blocks'; }
    }
  ];
  var GROUPS = ['orthogonal', 'torus · Hadamard', 'contrast'];
  function methodById(id) { for (var i = 0; i < METHODS.length; i++) if (METHODS[i].id === id) return METHODS[i]; return null; }

  /* ---------- invariants ---------- */
  function cosMat(K) {
    var n = K.length, C = zeros(n, n), d = K.map(function (r, i) { return Math.sqrt(Math.max(0, r[i])); });
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) C[i][j] = d[i] > 0 && d[j] > 0 ? K[i][j] / (d[i] * d[j]) : NaN;
    return C;
  }
  var REF = (function () {
    var K = mul(W0, T(W0)), Ki = mul(T(W0), W0), sv = svals(W0);
    return { K: K, Ki: Ki, C: cosMat(K), sv: sv, nK: frob(K), nKi: frob(Ki), mx: maxAbs(W0), L: chol(K) };
  })();
  var JPAIR = [[0, -1], [1, 0]];
  function invariants(W) {
    var K = mul(W, T(W)), Ki = mul(T(W), W), sv = svals(W), C = cosMat(K), i, j, r = {};
    r.W = W; r.K = K; r.Ki = Ki; r.C = C; r.sv = sv;
    r.d = {};
    r.d.Kout = frob(add(K, REF.K, -1)) / REF.nK;
    r.d.Kin = frob(add(Ki, REF.Ki, -1)) / REF.nKi;
    var dc = 0, da = 0, undef = false;
    for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) {
      if (isNaN(C[i][j])) { undef = true; continue; }
      dc = Math.max(dc, Math.abs(C[i][j] - REF.C[i][j])); da = Math.max(da, Math.abs(Math.abs(C[i][j]) - Math.abs(REF.C[i][j])));
    }
    r.d.cos = undef ? Infinity : dc; r.d.abscos = undef ? Infinity : da;
    r.d.spec = 0; for (i = 0; i < 4; i++) r.d.spec = Math.max(r.d.spec, Math.abs(sv[i] - REF.sv[i]) / REF.sv[0]);
    /* row lines: w_i in R^x w0_i (sine of the angle; a zero row leaves the T_4-orbit) */
    r.collapsed = []; var dl = 0;
    for (i = 0; i < 4; i++) {
      var nw = Math.sqrt(dot(W[i], W[i]));
      if (nw <= 1e-12 * ROW_NORM0[i]) { r.collapsed.push(i); continue; }
      var c = dot(W[i], W0[i]) / (ROW_NORM0[i] * ROW_NORM0[i]), res = 0;
      for (j = 0; j < 4; j++) { var e = W[i][j] - c * W0[i][j]; res += e * e; }
      dl = Math.max(dl, Math.sqrt(res) / nw);
    }
    r.d.lines = r.collapsed.length ? Infinity : dl;
    /* complex lines of the output pairs (1,2), (3,4): W_I in C^x X with X = W0_I, i.e. in span{X, JX} and nonzero */
    var dcl = 0; r.pairCollapsed = []; r.pairRatio = [];
    for (var p = 0; p < 2; p++) {
      var X = [W0[2 * p], W0[2 * p + 1]], Y = [W[2 * p], W[2 * p + 1]], JX = mul(JPAIR, X), nX2 = frob(X) * frob(X), nY = frob(Y);
      r.pairRatio.push(frob(mul(T(Y), Y)) / frob(mul(T(X), X)));
      if (nY <= 1e-12 * Math.sqrt(nX2)) { r.pairCollapsed.push(p); continue; }
      var a = 0, b = 0; for (i = 0; i < 2; i++) for (j = 0; j < 4; j++) { a += Y[i][j] * X[i][j]; b += Y[i][j] * JX[i][j]; }
      var rr = 0; for (i = 0; i < 2; i++) for (j = 0; j < 4; j++) { var ee = Y[i][j] - (a * X[i][j] + b * JX[i][j]) / nX2; rr += ee * ee; }
      dcl = Math.max(dcl, Math.sqrt(rr) / nY);
    }
    r.d.clines = r.pairCollapsed.length ? Infinity : dcl;
    /* zeros of W0 stay zero (one-way), and new zeros */
    var dz = 0; ZERO0.forEach(function (z) { dz = Math.max(dz, Math.abs(W[z[0]][z[1]]) / REF.mx); }); r.d.zeros = dz;
    r.newZeros = [];
    for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) if (Math.abs(W[i][j]) <= TOL * REF.mx && !(i === ZERO0[0][0] && j === ZERO0[0][1])) r.newZeros.push([i, j]);
    /* numerical rank of Delta W */
    r.dW = add(W, W0, -1); r.svd = svals(r.dW); r.thr = TOL * REF.sv[0];
    r.rank = r.svd.filter(function (s) { return s > r.thr; }).length;
    r.d.rank = r.svd[2];
    r.lit = {};
    ['Kout', 'cos', 'abscos', 'Kin', 'spec', 'lines', 'clines', 'zeros'].forEach(function (k) { r.lit[k] = r.d[k] < TOL; });
    r.lit.rank = r.rank <= 2;
    return r;
  }
  function flatCols(cols) { var J = zeros(16, cols.length); cols.forEach(function (D, k) { for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) J[4 * i + j][k] = D[i][j]; }); return J; }
  function evaluate(m, q, f) {
    var err = m.check ? m.check(q, f) : null;
    if (err) return { err: err };
    var r = invariants(m.map(q, f));
    r.jsv = svals(flatCols(m.jac(q, f)));
    r.jrank = r.jsv.filter(function (s) { return s > TOL * r.jsv[0]; }).length;
    r.guar = m.guar(q, f);
    return r;
  }
  function sameVec(a, b) { if (a.length !== b.length) return false; for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }

  /* ---------- formatting ---------- */
  var SUPD = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sup(n) { return String(n).split('').map(function (ch) { return SUPD[ch] || ch; }).join(''); }
  function sci(x, d) {
    if (x === 0) return '0';
    var p = x.toExponential(d == null ? 1 : d).split('e'), e = parseInt(p[1], 10);
    return e === 0 ? p[0] : p[0] + '×10' + sup(e);
  }
  function fmtD(x) { if (!isFinite(x)) return '—'; if (x === 0) return '0'; if (x < 1e-3) return sci(x, 1); return String(+x.toPrecision(3)); }
  function mns(s) { return s.replace(/-/g, '−'); }
  /* two decimals; snap to 1e-8 first so values equal up to round-off (e.g. -0.625 vs -0.62499999999) print the same */
  function f2(x) { if (!isFinite(x)) return '—'; x = Math.round(x * 1e8) / 1e8; if (Math.abs(x) < 0.005) return '0.00'; return mns(x.toFixed(2)); }
  function f3(x) { return mns(String(+x.toFixed(3))); }

  /* ---------- colours ---------- */
  function parseColor(s) {
    if (window.d3 && d3.color) { var c = d3.color(s); if (c) { c = c.rgb(); return [c.r, c.g, c.b]; } }
    var m = /^#([0-9a-f]{6})$/i.exec(s || '');
    if (m) { var v = parseInt(m[1], 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
    m = /rgba?\(([^)]+)\)/.exec(s || '');
    if (m) { var p = m[1].split(',').map(parseFloat); return [p[0], p[1], p[2]]; }
    return [128, 128, 128];
  }
  function lerpC(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function rgbStr(c) { return 'rgb(' + Math.round(c[0]) + ',' + Math.round(c[1]) + ',' + Math.round(c[2]) + ')'; }
  function lum(c) { var f = c.map(function (v) { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }); return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]; }
  function contrast(a, b) { var la = lum(a), lb = lum(b); return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05); }

  /* ---------- scoped CSS ---------- */
  function injectCSS() {
    if (document.getElementById('css-gram')) return;
    var F = '[data-figure="gram"] ';
    var css = [
      F + '.gl-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.gl-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.gl-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.gl-title i{color:var(--tide)}',
      F + '.gl-kicker{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.gl-instr{margin:.45rem 0 .9rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:70ch}',
      F + '.gl-instr b{color:var(--ink);font-weight:600}',
      F + '.gl-tag{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.seg button,' + F + '.btn{font-weight:500}',
      F + '.nt{text-transform:none;letter-spacing:0;font-family:var(--f-body);font-weight:400;font-style:italic;font-size:1.2em;line-height:1}',
      F + '.btn .nt{font-size:1.15em}',
      F + '.gl-ctl{display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem 1.6rem;margin-bottom:1rem}',
      F + '.gl-group{min-width:0;border-top:1px solid var(--rule);padding-top:.5rem;display:grid;gap:.55rem;align-content:start}',
      F + '.gl-gl{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem .6rem}',
      F + '.gl-gl .gl-tag{color:var(--ink)}',
      F + '.gl-hint{font-family:var(--f-body);font-size:.8rem;color:var(--ink-2)}',
      F + '.gl-mrow{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem .55rem}',
      F + '.gl-mk{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2);min-width:8.6rem}',
      F + '.gl-mrow .seg button{padding:.36rem .66rem}',
      F + '.gl-map{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.6rem .75rem .65rem;display:grid;gap:.4rem;min-width:0}',
      F + '.gl-mt{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.12rem;line-height:1.25;color:var(--ink)}',
      F + '.gl-tex{font-size:.98rem;line-height:1.5;color:var(--ink);overflow-x:auto;overflow-y:hidden;padding:.1rem 0;display:flex;flex-wrap:wrap;align-items:baseline;column-gap:.6em}',
      F + '.gl-tx{font-family:var(--f-body);font-size:.85rem;color:var(--ink-2);font-style:italic}',
      F + '.gl-tex mjx-container{white-space:nowrap}',
      F + '.gl-tex mjx-container svg{max-width:none}',
      F + '.gl-theory{margin:0;font-size:.86rem;line-height:1.45;color:var(--ink-2)}',
      F + '.gl-theory code{font-size:.78rem}',
      F + '.nw{white-space:nowrap}',
      F + 'sub,' + F + 'sup{font-size:.76em;line-height:0;letter-spacing:0;font-variant-numeric:lining-nums}',
      F + '.gl-ref{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--tide);white-space:nowrap;margin-left:.15rem}',
      F + '.gl-chips{display:flex;flex-wrap:wrap;gap:.3rem .45rem}',
      F + '.gl-chip{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.01em;padding:.14rem .5rem;border-radius:999px;border:1px solid var(--rule);color:var(--ink-2);font-variant-numeric:tabular-nums;white-space:nowrap;background:var(--paper-2)}',
      F + '.gl-chip i{font-family:var(--f-body);font-weight:400;font-size:1.12em}',
      F + '.gl-chip b{font-family:var(--f-mono);font-weight:500;color:var(--ink)}',
      /* parameter matrices */
      F + '.gl-mats{display:flex;flex-wrap:wrap;align-items:flex-start;gap:.7rem 1.1rem;--cw:3.15rem}',
      F + '.gl-mat{display:grid;grid-template-columns:auto auto;align-items:center;column-gap:.35rem;row-gap:.15rem}',
      F + '.gl-ml{font-family:var(--f-body);font-style:italic;font-size:1.05rem;color:var(--ink);white-space:nowrap}',
      F + '.gl-br{justify-self:start;display:grid;gap:2px;padding:3px 5px;border-left:1.5px solid var(--ink-2);border-right:1.5px solid var(--ink-2);border-radius:2px;' +
        'background:linear-gradient(var(--ink-2),var(--ink-2)) left top/5px 1.5px no-repeat,linear-gradient(var(--ink-2),var(--ink-2)) left bottom/5px 1.5px no-repeat,' +
        'linear-gradient(var(--ink-2),var(--ink-2)) right top/5px 1.5px no-repeat,linear-gradient(var(--ink-2),var(--ink-2)) right bottom/5px 1.5px no-repeat}',
      F + '.gl-mcap{grid-column:2;font-family:var(--f-body);font-size:.8rem;color:var(--ink-2);max-width:14rem;line-height:1.3}',
      F + '.gl-cell{box-sizing:border-box;width:var(--cw);height:1.7rem;margin:0;padding:0 .2rem;font-family:var(--f-mono);font-size:.78rem;font-variant-numeric:tabular-nums;text-align:center;' +
        'color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:3px;cursor:ew-resize;touch-action:pan-y;-webkit-user-select:none;user-select:none}',
      F + 'input.gl-cell:hover{border-color:var(--tide)}',
      F + 'input.gl-cell:focus{cursor:text;-webkit-user-select:text;user-select:text;border-color:var(--ochre);outline:none;box-shadow:0 0 0 2px var(--ochre-soft)}',
      F + 'input.gl-cell.scrub{border-color:var(--tide);background:var(--tide-soft)}',
      F + 'input.gl-cell.moved{color:var(--tide)}',
      F + 'span.gl-cell{display:inline-flex;align-items:center;justify-content:center;cursor:default;color:var(--ink-2);background:transparent;border-color:transparent}',
      F + '.gl-actions{display:flex;flex-wrap:wrap;gap:.4rem .45rem;align-items:center}',
      F + '.gl-actions .btn{font-size:.75rem;padding:.36rem .62rem}',
      F + '.gl-actions .btn[aria-pressed="true"]{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
      F + '.gl-actions .btn.warn[aria-pressed="true"]{background:var(--seal);border-color:var(--seal)}',
      F + '.gl-actions .btn:disabled{opacity:.4;cursor:not-allowed}',
      /* lamps */
      F + '.gl-board{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.45rem;margin:.2rem 0 .7rem}',
      F + '.gl-lamp{display:grid;grid-template-columns:auto minmax(0,1fr);column-gap:.5rem;row-gap:.1rem;align-items:center;padding:.42rem .55rem .45rem;border:1px solid var(--rule);border-radius:var(--radius);background:var(--paper);min-width:0;transition:background-color .18s,border-color .18s}',
      F + '.gl-dot{grid-row:1 / span 2;width:.72rem;height:.72rem;border-radius:50%;border:1.5px solid var(--ink-3);box-sizing:border-box;transition:background-color .18s,box-shadow .18s}',
      F + '.gl-ln{font-family:var(--f-body);font-size:.9rem;line-height:1.15;color:var(--ink);min-width:0;display:flex;flex-wrap:wrap;justify-content:space-between;gap:0 .4rem;align-items:baseline}',
      F + '.gl-ln .gl-lr{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.01em;color:var(--ink-2);white-space:nowrap}',
      F + '.gl-ls{font-family:var(--f-mono);font-size:.75rem;line-height:1.3;color:var(--ink-2);font-variant-numeric:tabular-nums;min-width:0;overflow-wrap:anywhere}',
      F + '.gl-lamp.on.thm{border-color:var(--moss);background:var(--moss-soft)}',
      F + '.gl-lamp.on.thm .gl-dot{background:var(--moss);border-color:var(--moss);box-shadow:0 0 0 3px var(--moss-soft),0 0 10px 1px var(--moss)}',
      F + '.gl-lamp.on.thm .gl-ls{color:var(--moss)}',
      F + '.gl-lamp.on.here{border-color:var(--moss);border-style:dashed}',
      F + '.gl-lamp.on.here .gl-dot{border-color:var(--moss);background:var(--moss-soft)}',
      F + '.gl-lamp.on.here .gl-ls{color:var(--moss)}',
      F + '.gl-lamp.lost{border-color:var(--seal);background:var(--seal-soft)}',
      F + '.gl-lamp.lost .gl-dot{border-color:var(--seal);background:transparent}',
      F + '.gl-lamp.lost .gl-ls{color:var(--seal)}',
      F + '.gl-lamp.off .gl-ln{color:var(--ink-2)}',
      F + '.gl-tq{font-weight:600;margin-right:.15rem}',
      /* banner */
      F + '.gl-banner{display:flex;gap:.6rem;align-items:flex-start;padding:.55rem .75rem;margin:0 0 1rem;border-radius:var(--radius);border:1px solid var(--rule);border-left-width:3px;background:var(--paper);font-size:.9rem;line-height:1.45;color:var(--ink-2)}',
      F + '.gl-banner b{color:var(--ink);font-weight:600}',
      F + '.gl-banner .gl-bk{flex:none;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;padding-top:.12rem;min-width:5.9rem}',
      F + '.gl-banner.moss{border-left-color:var(--moss)} ' + F + '.gl-banner.moss .gl-bk{color:var(--moss)}',
      F + '.gl-banner.seal{border-left-color:var(--seal)} ' + F + '.gl-banner.seal .gl-bk{color:var(--seal)}',
      F + '.gl-banner.ochre{border-left-color:var(--ochre)} ' + F + '.gl-banner.ochre .gl-bk{color:var(--ochre)}',
      F + '.gl-banner.ink{border-left-color:var(--ink-3)} ' + F + '.gl-banner.ink .gl-bk{color:var(--ink-2)}',
      /* main panels */
      F + '.gl-main{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.2rem;align-items:stretch}',
      F + '.gl-panel{min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.6rem .7rem .55rem;display:flex;flex-direction:column}',
      F + '.gl-panel > .gl-svg{flex:1 1 auto;display:flex;flex-direction:column;justify-content:center}',
      F + '.gl-panel > .gl-svg svg{max-width:27rem;margin:0 auto}',
      F + '.gl-ph{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:.3rem .6rem;margin-bottom:.25rem}',
      F + '.gl-ph .seg button{padding:.26rem .6rem;font-size:.75rem}',
      F + '.gl-svg svg{display:block;width:100%;height:auto;overflow:visible}',
      F + '.gl-note{margin:.3rem 0 0;font-family:var(--f-body);font-size:.82rem;line-height:1.45;color:var(--ink-2)}',
      F + '.gl-note b{color:var(--moss);font-weight:600}',
      F + '.gl-heats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6rem}',
      F + '.gl-svg svg.gl-px{width:auto;max-width:100%;margin:0 auto}',
      F + '.gl-hm{min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.45rem .5rem .4rem;transition:border-color .18s,box-shadow .18s}',
      F + '.gl-hm.on{border-color:var(--moss);box-shadow:0 0 0 3px var(--moss-soft),0 0 14px -2px var(--moss)}',
      F + '.gl-hm.half{border-color:var(--ochre);box-shadow:0 0 0 3px var(--ochre-soft)}',
      F + '.gl-hm.lost{border-color:var(--seal)}',
      F + '.gl-hmh{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0 .4rem;margin-bottom:.15rem;min-width:0}',
      F + '.gl-hmt{font-family:var(--f-body);font-size:.92rem;color:var(--ink);white-space:nowrap}',
      F + '.gl-hmd{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums;white-space:nowrap}',
      F + '.gl-hm.on .gl-hmd{color:var(--moss)} ' + F + '.gl-hm.half .gl-hmd{color:var(--ochre)} ' + F + '.gl-hm.lost .gl-hmd{color:var(--seal)}',
      F + '.gl-hms{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.01em;color:var(--ink-2);margin:0 0 .25rem}',
      F + '.gl-strips{display:grid;grid-template-columns:minmax(0,1fr);gap:.6rem;margin-top:1rem}',
      F + '.gl-st{min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.5rem .6rem .4rem}',
      F + '.gl-sth{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0 .5rem}',
      F + '.gl-stv{font-family:var(--f-mono);font-size:.75rem;color:var(--ink);font-variant-numeric:tabular-nums;white-space:nowrap}',
      F + '.gl-stv.ok{color:var(--moss)} ' + F + '.gl-stv.ap{color:var(--ochre)}',
      F + '.gl-sts{font-family:var(--f-body);font-size:.82rem;line-height:1.45;color:var(--ink-2);margin-top:.15rem}',
      F + '.gl-legend{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem 1.3rem;margin-top:.9rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--ink-2)}',
      F + '.gl-legend .gl-li{display:inline-flex;align-items:center;gap:.4rem}',
      F + '.gl-legend svg{flex:none;overflow:visible}',
      F + '.gl-foot{margin-top:.7rem;padding-top:.55rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.82rem;line-height:1.55;color:var(--ink-2)}',
      F + '.gl-foot code{font-size:.8rem}',
      F + '.gl-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
      F + '.gl-err{padding:.5rem .7rem;border:1px solid var(--seal);border-radius:var(--radius);color:var(--seal);background:var(--seal-soft);font-size:.86rem;margin-bottom:.8rem}',
      '@container (max-width: 459px){',
      F + '.gl-heats{grid-template-columns:minmax(0,1fr)}',
      F + '.gl-hm{display:grid;grid-template-columns:12.75rem minmax(0,1fr);column-gap:.75rem;align-items:start}',
      F + '.gl-hm .gl-svg{grid-column:1;grid-row:1 / span 3}',
      F + '.gl-hm .gl-hmh{grid-column:2;grid-row:1;flex-direction:column;align-items:flex-start;gap:.15rem;padding-top:.2rem}',
      F + '.gl-hm .gl-hms{grid-column:2;grid-row:2}',
      F + '.gl-hmd{white-space:normal}',
      '}',
      '@container (max-width: 430px){',
      F + '.gl-mats{--cw:2.85rem;gap:.6rem .8rem}',
      F + '.gl-cell{font-size:.75rem}',
      F + '.gl-mk{min-width:0;width:100%}',
      F + '.gl-banner{flex-direction:column;gap:.15rem}',
      '}',
      '@container (min-width: 560px){',
      F + '.gl-board{grid-template-columns:repeat(3,minmax(0,1fr))}',
      '}',
      '@container (min-width: 700px){',
      F + '.gl-strips{grid-template-columns:repeat(3,minmax(0,1fr))}',
      F + '.gl-sth{min-height:2.6rem;align-content:flex-start}',
      F + '.gl-sts{min-height:2.9em}',
      '}',
      '@container (min-width: 780px){',
      F + '.gl-ctl{grid-template-columns:minmax(0,1fr) minmax(0,1.08fr)}',
      F + '.gl-main{grid-template-columns:minmax(0,1fr) 28.6rem}',
      '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-gram';
    s.textContent = css;
    document.head.appendChild(s);
  }

  var LAMPS = [
    { key: 'Kout', name: 'neuron Gram <i>K</i><sub>out</sub>', ref: 'Thm II.5(1)', tip: 'K_out = WWᵀ: every neuron norm and every angle between neurons.' },
    { key: 'cos', name: 'cosines', ref: 'Cor II.6', tip: 'The normalised neuron Gram matrix cos(wᵢ, wⱼ).' },
    { key: 'abscos', name: '|cosines|', ref: 'Cor II.6', tip: 'Absolute cosines |cos(wᵢ, wⱼ)|: angles up to the sign of each neuron.' },
    { key: 'Kin', name: 'input Gram <i>K</i><sub>in</sub>', ref: 'Thm II.5(2)', tip: 'K_in = WᵀW, the complete invariant of left orthogonal actions.' },
    { key: 'spec', name: 'spectrum σ(<i>W</i>)', ref: 'Thm II.5(3)', tip: 'The singular values, the complete invariant of two-sided orthogonal actions.' },
    { key: 'rank', name: 'rank Δ<i>W</i> ≤ 2', ref: 'Im LoRA₂', tip: 'ΔW = W − W₀ lies in 𝓜≤2, the image of LoRA with r = 2.' },
    { key: 'lines', name: 'row lines', ref: 'Thm II.5(4)', tip: 'Every neuron wᵢ is a nonzero multiple of w₀ᵢ: the T₄-orbit invariant.' },
    { key: 'clines', name: 'pair ℂ-lines', ref: 'Prop II.9(d)', tip: 'Each output pair (rows 1–2, rows 3–4), read as a vector of ℂ⁴, is a nonzero complex multiple of its value at W₀.' },
    { key: 'zeros', name: 'zeros of <i>W</i>₀ stay 0', ref: 'Prop II.9(e)', tip: 'Z(W₀) ⊆ Z(W). One-way: new zeros may appear.' }
  ];

  Atlas.register('gram', function (el, A) {
    injectCSS();
    var h = A.h;
    var reduced = A.reducedMotion();
    var st = { mid: 'oft', frame: 'lab', wander: false, per: {} };
    METHODS.forEach(function (m) { st.per[m.id] = { q: m.demo.slice(), f: { cn: false }, draws: 0 }; });
    function cur() { return methodById(st.mid); }
    function curS() { return st.per[st.mid]; }

    /* expressive dimension d(M) = rank d(rho) at a seeded generic point (computed once) */
    var GEN = {};
    (function () {
      var rnd = A.rng(20261001);
      METHODS.forEach(function (m, mi) {
        var q = m.demo.map(function (v) { return v + 0.3 * rnd.normal(); });
        var fl = m.id === 'oft' ? [false, true] : [false];
        fl.forEach(function (cn) {
          var r = evaluate(m, q, { cn: cn });
          GEN[m.id + (cn ? '-cn' : '')] = r.err ? NaN : r.jrank;
        });
      });
    })();
    function dOf(m, f) { return GEN[m.id + (m.id === 'oft' && f.cn ? '-cn' : '')]; }

    /* lab frame: W0's top three right singular vectors (from the symmetric eigenproblem of W0^T W0) */
    var V3 = (function () {
      var E = A.LA.symEig(mul(T(W0), W0)), idx = [0, 1, 2, 3].sort(function (a, b) { return E.values[b] - E.values[a]; });
      return idx.slice(0, 3).map(function (k) {
        var v = [0, 1, 2, 3].map(function (i) { return E.vectors[i][k]; }), big = 0;
        v.forEach(function (x, i) { if (Math.abs(x) > Math.abs(v[big])) big = i; });
        return v[big] < 0 ? v.map(function (x) { return -x; }) : v;
      });
    })();
    var ENERGY = (REF.sv[0] * REF.sv[0] + REF.sv[1] * REF.sv[1] + REF.sv[2] * REF.sv[2]) / REF.sv.reduce(function (s, x) { return s + x * x; }, 0);

    /* ---------- DOM ---------- */
    var stage = h('div', { class: 'stage gl-stage', role: 'group', 'aria-label': 'Gram lab: choose a fine-tuning method and move its parameters; lamps show which invariants of the pretrained weight matrix W0 the method keeps exactly (neuron Gram matrix, cosines, input Gram matrix, singular values, row lines, zeros, rank of the update), with heatmaps, neuron arrows and singular-value strips.' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'gl-head' }, [
      h('h3', { class: 'gl-title', html: 'What a method can <i>never</i> change' }),
      h('span', { class: 'gl-kicker', text: 'Gram lab · Erlangen · Thm II.5 · Cor II.6 · Thm II.7' })
    ]));
    stage.appendChild(h('p', { class: 'gl-instr', html: hsub('Pick a method, then drag any parameter cell sideways (or type in it, or press <b>Random draw</b>). A lamp lights when that invariant of <i>W</i> equals its value at <i>W</i>₀ to 10⁻¹⁰. Lamps marked <b>∀</b> are guaranteed by a theorem; a red lamp means you have just broken that theorem’s hypothesis.') }));
    if (!window.d3) stage.appendChild(h('p', { class: 'gl-err', text: 'D3 v7 did not load; colours fall back to a plain interpolation.' }));

    var ctl = h('div', { class: 'gl-ctl' });
    stage.appendChild(ctl);
    /* method selector + map card */
    var gM = h('div', { class: 'gl-group' }, [h('div', { class: 'gl-gl' }, [h('span', { class: 'gl-tag', html: 'Method <span class="nt">ρ : Q → Θ</span>' })])]);
    var mButtons = {};
    GROUPS.forEach(function (g, gi) {
      var seg = h('div', { class: 'seg', role: 'group', 'aria-label': g + ' methods' });
      METHODS.filter(function (m) { return m.group === gi; }).forEach(function (m) {
        var b = h('button', { type: 'button', 'aria-pressed': 'false', text: m.label });
        b.addEventListener('click', function () { selectMethod(m.id); });
        mButtons[m.id] = b; seg.appendChild(b);
      });
      gM.appendChild(h('div', { class: 'gl-mrow' }, [h('span', { class: 'gl-mk', text: g }), seg]));
    });
    var mapTitle = h('div', { class: 'gl-mt' }), mapTex = h('div', { class: 'gl-tex' }), mapTheory = h('p', { class: 'gl-theory' }), mapChips = h('div', { class: 'gl-chips' });
    gM.appendChild(h('div', { class: 'gl-map' }, [mapTitle, mapTex, mapTheory, mapChips]));
    ctl.appendChild(gM);
    /* parameters */
    var gP = h('div', { class: 'gl-group' }, [h('div', { class: 'gl-gl' }, [h('span', { class: 'gl-tag', html: 'Parameters <span class="nt">q ∈ Q</span>' }), h('span', { class: 'gl-hint', text: 'drag ↔ · type · ↑↓ (⇧ ×5)' })])]);
    var mats = h('div', { class: 'gl-mats' });
    gP.appendChild(mats);
    var bRand = h('button', { type: 'button', class: 'btn', text: 'Random draw' });
    var bWander = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', text: 'Wander' });
    var bReset = h('button', { type: 'button', class: 'btn', html: 'Reset to <span class="nt">q<sub>0</sub></span>' });
    var bExtra = h('button', { type: 'button', class: 'btn' });
    var actions = h('div', { class: 'gl-actions' }, [bRand, bWander, bReset, bExtra]);
    gP.appendChild(actions);
    if (reduced) { bWander.disabled = true; bWander.title = 'Animation is off because reduced motion is requested; use Random draw.'; }
    ctl.appendChild(gP);

    /* lamp board */
    var board = h('div', { class: 'gl-board', role: 'list', 'aria-label': 'Invariant lamps' });
    var lampEls = {};
    LAMPS.forEach(function (L) {
      var ls = h('span', { class: 'gl-ls' });
      var e = h('div', { class: 'gl-lamp off', role: 'listitem', title: L.tip }, [
        h('span', { class: 'gl-dot', 'aria-hidden': 'true' }),
        h('span', { class: 'gl-ln', html: '<span>' + hsub(L.name) + '</span><span class="gl-lr">' + L.ref + '</span>' }),
        ls
      ]);
      lampEls[L.key] = { el: e, ls: ls };
      board.appendChild(e);
    });
    stage.appendChild(board);
    var banner = h('div', { class: 'gl-banner ink' });                 // announced (debounced) through the live region below
    stage.appendChild(banner);
    var errBox = h('div', { class: 'gl-err', style: 'display:none' });
    stage.appendChild(errBox);

    /* arrows + heatmaps */
    var main = h('div', { class: 'gl-main' });
    stage.appendChild(main);
    var frameSeg = A.seg([{ value: 'lab', label: 'lab frame' }, { value: 'gram', label: 'Gram frame' }], 'lab', function (v) { st.frame = v; render(); }, 'Frame for the neuron arrows');
    var arrowSvg = h('div', { class: 'gl-svg' }), arrowNote = h('p', { class: 'gl-note' });
    main.appendChild(h('div', { class: 'gl-panel' }, [h('div', { class: 'gl-ph' }, [h('span', { class: 'gl-tag', html: 'Neurons <span class="nt">w<sub>1</sub>…w<sub>4</sub></span> = rows of <span class="nt">W</span>' }), frameSeg.el]), arrowSvg, arrowNote]));
    var heats = h('div', { class: 'gl-heats' });
    main.appendChild(heats);
    var HM = [
      { key: 'Kout', title: '<i>K</i><sub>out</sub> = <i>WW</i>ᵀ', sub: 'neurons × neurons', rl: 'w', cl: 'w' },
      { key: 'cos', title: 'cos(<i>w</i><sub><i>i</i></sub>, <i>w</i><sub><i>j</i></sub>)', sub: 'normalised neuron Gram', rl: 'w', cl: 'w' },
      { key: 'Kin', title: '<i>K</i><sub>in</sub> = <i>W</i>ᵀ<i>W</i>', sub: 'inputs × inputs', rl: 'x', cl: 'x' },
      { key: 'W', title: '<i>W</i> and its zeros', sub: 'neurons × inputs', rl: 'w', cl: 'x' }
    ].map(function (d) {
      d.dEl = h('span', { class: 'gl-hmd' }); d.svg = h('div', { class: 'gl-svg' });
      d.el = h('div', { class: 'gl-hm' }, [h('div', { class: 'gl-hmh' }, [h('span', { class: 'gl-hmt', html: d.title }), d.dEl]), h('div', { class: 'gl-hms', text: d.sub }), d.svg]);
      heats.appendChild(d.el);
      return d;
    });
    /* strips */
    var strips = h('div', { class: 'gl-strips' });
    stage.appendChild(strips);
    var STR = [
      { key: 'spec', tag: 'Spectrum <span class="nt">σ(W)</span>' },
      { key: 'rank', tag: 'Numerical rank of <span class="nt">ΔW</span>' },
      { key: 'jac', tag: 'First order <span class="nt">σ(dρ<sub>q</sub>)</span>' }
    ].map(function (d) {
      d.v = h('span', { class: 'gl-stv' }); d.svg = h('div', { class: 'gl-svg' }); d.s = h('div', { class: 'gl-sts' });
      d.el = h('div', { class: 'gl-st' }, [h('div', { class: 'gl-sth' }, [h('span', { class: 'gl-tag', html: d.tag }), d.v]), d.svg, d.s]);
      strips.appendChild(d.el);
      return d;
    });
    var legend = h('div', { class: 'gl-legend' });
    stage.appendChild(legend);
    var foot = h('div', { class: 'gl-foot' });
    stage.appendChild(foot);
    var live = h('div', { class: 'gl-live', 'aria-live': 'polite' });
    stage.appendChild(live);

    /* ---------- typeset helper ---------- */
    var texLast = '';
    function setTex(node, html) {
      if (texLast === html) return;
      texLast = html;
      if (window.MathJax && MathJax.typesetClear) { try { MathJax.typesetClear([node]); } catch (e) { /* ignore */ } }
      node.innerHTML = html;
      A.typeset(node);
    }

    /* ---------- parameter cells ---------- */
    var cells = [];                                    // {input, idx, blk}
    function clamp(v, b) { return Math.max(b.min, Math.min(b.max, v)); }
    function q2(v) { return Math.round(v * 100) / 100; }
    function cellText(v) { return f2(v); }
    function setParam(idx, v, blk, opts) {
      var S = curS();
      v = clamp(v, blk);
      if (S.q[idx] === v) return;
      S.q[idx] = v;
      if (!opts || !opts.keepText) syncCells();
      else syncStatic();
      schedule();
    }
    function makeInput(idx, blk, name) {
      var inp = h('input', { type: 'text', inputmode: 'decimal', class: 'gl-cell', role: 'spinbutton', 'aria-label': name, 'aria-valuemin': blk.min, 'aria-valuemax': blk.max, autocomplete: 'off', spellcheck: 'false' });
      var drag = null;
      inp.addEventListener('pointerdown', function (e) {
        if (document.activeElement === inp || e.button !== 0) return;
        e.preventDefault();
        stopWander();
        drag = { x: e.clientX, v: curS().q[idx], moved: false, id: e.pointerId };
        try { inp.setPointerCapture(e.pointerId); } catch (er) { /* ignore */ }
      });
      inp.addEventListener('pointermove', function (e) {
        if (!drag || e.pointerId !== drag.id) return;
        var dx = e.clientX - drag.x;
        if (!drag.moved && Math.abs(dx) < 3) return;
        drag.moved = true; inp.classList.add('scrub');
        setParam(idx, q2(drag.v + dx * (blk.max - blk.min) / 320), blk);
      });
      function end(e) {
        if (!drag || (e && e.pointerId !== drag.id)) return;
        var moved = drag.moved; drag = null; inp.classList.remove('scrub');
        if (!moved && e && e.type === 'pointerup') { inp.focus(); inp.select(); }
      }
      inp.addEventListener('pointerup', end);
      inp.addEventListener('pointercancel', end);
      inp.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault(); stopWander();
          var stp = e.shiftKey ? 0.25 : e.altKey ? 0.01 : 0.05;
          setParam(idx, q2(curS().q[idx] + (e.key === 'ArrowUp' ? stp : -stp)), blk);
          inp.select();
        } else if (e.key === 'Enter') { inp.blur(); }
      });
      inp.addEventListener('input', function () {
        var v = parseFloat(inp.value.replace(/−/g, '-'));
        if (isFinite(v)) { stopWander(); setParam(idx, v, blk, { keepText: true }); }
      });
      inp.addEventListener('blur', function () { syncCells(); });
      cells.push({ input: inp, idx: idx, blk: blk });
      return inp;
    }
    var statics = [];                                  // read-only skew mirrors {el, idx}
    function buildMats() {
      cells = []; statics = [];
      mats.innerHTML = '';
      var m = cur();
      m.blocks.forEach(function (b) {
        var grid = h('div', { class: 'gl-br' });
        var rows = b.type === 'skew' ? 4 : b.rows, cols = b.type === 'skew' ? 4 : b.cols;
        grid.style.gridTemplateColumns = 'repeat(' + cols + ', var(--cw))';
        for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) {
          if (b.type === 'skew') {
            if (i === j) grid.appendChild(h('span', { class: 'gl-cell', 'aria-hidden': 'true', text: '0' }));
            else {
              var k = -1; PAIRS.forEach(function (pr, kk) { if ((pr[0] === Math.min(i, j)) && (pr[1] === Math.max(i, j))) k = kk; });
              if (i < j) grid.appendChild(makeInput(b.off + k, b, b.label + sub(i + 1) + sub(j + 1)));
              else { var sp = h('span', { class: 'gl-cell', 'aria-hidden': 'true' }); statics.push({ el: sp, idx: b.off + k }); grid.appendChild(sp); }
            }
          } else {
            var ix = b.off + i * cols + j;
            var nm = b.names ? b.names[i * cols + j] : b.label + (cols === 1 ? sub(i + 1) : rows === 1 ? sub(j + 1) : sub(i + 1) + sub(j + 1));
            grid.appendChild(makeInput(ix, b, nm));
          }
        }
        var lab = h('span', { class: 'gl-ml', html: hsub(b.label) + ' =' });
        var box = h('div', { class: 'gl-mat' }, [lab, grid]);
        if (b.cap) box.appendChild(h('span', { class: 'gl-mcap', html: hsub(A.esc(b.cap)) }));
        mats.appendChild(box);
      });
      syncCells();
    }
    function syncStatic() {
      var S = curS(), m = cur();
      statics.forEach(function (s) { s.el.textContent = cellText(-S.q[s.idx]); });
      cells.forEach(function (c) { c.input.classList.toggle('moved', S.q[c.idx] !== m.q0[c.idx]); c.input.setAttribute('aria-valuenow', S.q[c.idx]); });
    }
    function syncCells() {
      var S = curS();
      cells.forEach(function (c) { if (document.activeElement !== c.input || st.wander) c.input.value = cellText(S.q[c.idx]); });
      syncStatic();
    }

    /* ---------- method switching, buttons ---------- */
    var EXTRA = {
      oft: { label: 'Cayley–Neumann (HF default)', toggle: true, run: function (S) { S.f.cn = !S.f.cn; } },
      boft: { label: 's₁ ↦ −s₁', run: function (S) { S.q[0] = -S.q[0]; } },
      ia3: { label: 'ℓ₂ ↦ 0', run: function (S) { S.q[1] = 0; } },
      road: { label: 'α₂ ↦ 0', run: function (S) { S.q[3] = 0; } },
      dora: { label: 'B ↦ 0', run: function (S) { for (var i = 4; i < 12; i++) S.q[i] = 0; } },
      hra: { label: 'u = (e₁, e₂)', run: function (S) { S.q = [1, 0, 0, 0, 0, 1, 0, 0]; } },
      qgoft: { label: 'orthogonal Mᵢ', run: function (S) { S.q = [0.6, -0.8, 0.8, 0.6, 0.8, 0.6, -0.6, 0.8]; } }
    };
    function chrome() {
      var m = cur(), S = curS();
      METHODS.forEach(function (mm) { mButtons[mm.id].setAttribute('aria-pressed', String(mm.id === m.id)); });
      mapTitle.innerHTML = hsub(A.esc(m.title(S.f)));
      setTex(mapTex, m.tex(S.f));
      mapTheory.innerHTML = hsub(m.theory(S.f));
      var d = dOf(m, S.f);
      mapChips.innerHTML = '<span class="gl-chip">|<i>M</i>| = <b>' + m.p + '</b> trainable</span>' +
        '<span class="gl-chip"><i>d</i>(<i>M</i>) = <b>' + d + '</b></span>' +
        '<span class="gl-chip">generic fibre dim <b>' + (m.p - d) + '</b></span>';
      var ex = EXTRA[m.id];
      bExtra.style.display = ex ? '' : 'none';
      if (ex) {
        bExtra.textContent = ex.label;
        bExtra.style.textTransform = 'none';
        bExtra.classList.toggle('warn', !!ex.toggle);
        if (ex.toggle) bExtra.setAttribute('aria-pressed', String(!!S.f.cn)); else bExtra.removeAttribute('aria-pressed');
      }
    }
    function selectMethod(id) {
      if (st.mid === id) return;
      stopWander();
      st.mid = id;
      chrome(); buildMats(); render();
    }
    bRand.addEventListener('click', function () {
      stopWander();
      var m = cur(), S = curS();
      S.draws++;
      var rnd = A.rng(7919 * (METHODS.indexOf(m) + 1) + 104729 * S.draws);
      var q = m.q0.slice();
      m.blocks.forEach(function (b) {
        var n = b.type === 'skew' ? 6 : b.rows * b.cols;
        var tries = 0;
        do {
          for (var i = 0; i < n; i++) {
            var ix = b.off + i, u = 2 * rnd() - 1;
            if (b.mode === 'mul') q[ix] = q2(m.q0[ix] * (1 + b.spread * u));
            else q[ix] = q2((b.center === 'zero' ? 0 : m.q0[ix]) + b.spread * u);
            q[ix] = clamp(q[ix], b);
          }
          tries++;
        } while (b.center === 'zero' && b.rows === 4 && b.cols === 1 && dot(q.slice(b.off, b.off + 4), q.slice(b.off, b.off + 4)) < 0.25 && tries < 20);
      });
      S.q = q;
      syncCells(); render();
    });
    bReset.addEventListener('click', function () { stopWander(); var S = curS(); S.q = cur().q0.slice(); syncCells(); render(); });
    bExtra.addEventListener('click', function () { stopWander(); var S = curS(), ex = EXTRA[st.mid]; if (!ex) return; ex.run(S); chrome(); syncCells(); render(); });

    /* ---------- wander (smooth path in Q; multiplicative blocks keep their signs) ---------- */
    var wz = null, wid = 0, onScreen = true;
    /* the wander loop sleeps while the figure is off screen and resumes, without a jump, when it returns */
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) {
        onScreen = es[es.length - 1].isIntersecting;
        if (onScreen && st.wander && wz && !wid) { wz.last = 0; wid = requestAnimationFrame(step); }
      }).observe(stage);
    }
    function startWander() {
      if (reduced) return;
      var m = cur(), S = curS(), rnd = A.rng(31337 + S.draws);
      wz = { m: m, base: S.q.slice(), t: 0, last: 0, ph: S.q.map(function () { return rnd() * 6.283; }), om: S.q.map(function () { return 0.45 + 0.7 * rnd(); }), blk: [] };
      m.blocks.forEach(function (b) { var n = b.type === 'skew' ? 6 : b.rows * b.cols; for (var i = 0; i < n; i++) wz.blk[b.off + i] = b; });
      st.wander = true; bWander.setAttribute('aria-pressed', 'true'); bWander.textContent = 'Pause';
      wid = requestAnimationFrame(step);
    }
    function step(ts) {
      wid = 0;
      if (!st.wander || !wz || !onScreen) return;
      if (!wz.last) wz.last = ts;
      var dt = Math.min(0.05, (ts - wz.last) / 1000); wz.last = ts; wz.t += dt;
      var S = curS();
      for (var i = 0; i < S.q.length; i++) {
        var b = wz.blk[i]; if (!b) continue;
        var s = Math.sin(wz.om[i] * wz.t + wz.ph[i]) - Math.sin(wz.ph[i]);
        S.q[i] = clamp(b.mode === 'mul' ? wz.base[i] * Math.exp(b.amp * s) : wz.base[i] + b.amp * s, b);
      }
      syncCells(); render();
      wid = requestAnimationFrame(step);
    }
    function stopWander() {
      if (!st.wander) return;
      st.wander = false; cancelAnimationFrame(wid); wid = 0; wz = null;
      var S = curS(); S.q = S.q.map(q2);
      bWander.setAttribute('aria-pressed', 'false'); bWander.textContent = 'Wander';
      syncCells(); schedule();
    }
    bWander.addEventListener('click', function () { if (st.wander) stopWander(); else startWander(); });

    /* ---------- rendering ---------- */
    var raf = 0;
    function schedule() { if (raf) return; raf = requestAnimationFrame(function () { raf = 0; render(); }); }
    var last = null, liveT = 0;

    function lampState(k, r) {
      var g = r.guar[k], lit = r.lit[k];
      if (lit && g && g.ok) return 'thm';
      if (lit) return 'here';
      if (g && !g.ok) return 'lost';
      if (g && g.ok) return 'bad';
      return 'off';
    }
    function lampStatus(k, r, s) {
      var g = r.guar[k];
      if (k === 'rank') {
        var t = 'rank ' + r.rank + (r.rank <= 2 ? ' · σ₃ ' + fmtD(r.svd[2]) : '');
        return s === 'thm' ? '<span class="gl-tq">∀</span>' + t : t;
      }
      var dv = r.d[k];
      var und = !isFinite(dv);
      var val = und ? (k === 'lines' || k === 'cos' || k === 'abscos' ? 'w' + sub(r.collapsed[0] + 1) + ' = 0' : k === 'clines' ? 'pair ' + (r.pairCollapsed[0] + 1) + ' = 0' : '—') : 'δ ' + fmtD(dv);
      if (s === 'thm') return '<span class="gl-tq">∀</span>kept · ' + val;
      if (s === 'here') return 'holds here · ' + val;
      if (s === 'lost') return 'lost · ' + A.esc(g.why) + (und ? '' : ' · ' + fmtD(dv));
      if (s === 'bad') return 'check failed · ' + val;
      return (und ? 'not defined · ' : 'moves · ') + val;
    }

    function render() {
      var m = cur(), S = curS(), r = evaluate(m, S.q, S.f);
      if (r.err) {
        errBox.style.display = ''; errBox.innerHTML = hsub(A.esc('Not a point of ' + m.name + ': ' + r.err + ' Showing the last valid state.'));
        return;
      }
      errBox.style.display = 'none';
      r.atQ0 = sameVec(S.q, m.q0);
      r.d0 = dOf(m, S.f);
      last = r;
      LAMPS.forEach(function (L) {
        var s = lampState(L.key, r), le = lampEls[L.key];
        le.el.className = 'gl-lamp ' + (s === 'thm' || s === 'here' ? 'on ' + s : s === 'bad' ? 'lost' : s);
        le.ls.innerHTML = hsub(lampStatus(L.key, r, s));
      });
      renderBanner(m, S, r);
      renderArrows(r);
      renderHeats(r);
      renderStrips(m, S, r);
      clearTimeout(liveT);
      liveT = setTimeout(function () {
        var kept = LAMPS.filter(function (L) { return r.lit[L.key]; }).map(function (L) { return L.name.replace(/<[^>]+>/g, ''); });
        live.textContent = m.name + ': ' + (kept.length ? 'kept: ' + kept.join(', ') : 'no invariant kept') + '. rank of the update ' + r.rank + '.';
      }, 500);
    }

    function setBanner(tone, key, html) { banner.className = 'gl-banner ' + tone; banner.innerHTML = '<span class="gl-bk">' + hsub(key) + '</span><span>' + hsub(html) + '</span>'; }
    function renderBanner(m, S, r) {
      var q = S.q, i;
      if (r.atQ0) {
        var dd = r.d0, s1 = r.jrank;
        var apex = s1 < dd ? ' The Jacobian strip shows <b>dim <i>S</i>₁ = ' + s1 + ' &lt; <i>d</i> = ' + dd + '</b>: this base point is an <b>apex</b>' + (m.id === 'hra' ? ' (Thm II.7(b): <i>kn</i> − <i>k</i>(<i>k</i>+1)/2 = 3 against <i>rn</i> − <i>r</i>(<i>r</i>+1)/2 = 5; the paired init is HF’s default).' : m.id === 'lora' ? ' (Thm II.2(b): <i>mr</i> = 8 against <i>r</i>(<i>m</i>+<i>n</i>−<i>r</i>) = 12).' : '.') : '';
        setBanner(m.id === 'hra' || s1 < dd ? 'ochre' : 'ink', 'at <span class="nt">q₀</span>', 'At <i>q</i>₀ the method sits at <i>W</i>₀, so every lamp is lit, most of them trivially. Move any parameter: only the <b>∀</b> lamps are guaranteed to stay lit.' + apex);
        return;
      }
      if (m.id === 'oft' && !S.f.cn) {
        setBanner('moss', 'kept', 'The update is exactly orthogonal, <i>W</i> = <i>W</i>₀<i>R</i>ᵀ with <i>R</i> ∈ SO(4). Every neuron norm, every angle between neurons and σ(<i>W</i>) are frozen (Cor II.6), while <i>K</i><sub>in</sub> moved by δ = ' + fmtD(r.d.Kin) + ' and rank Δ<i>W</i> = ' + r.rank + '. In exact arithmetic you can merge as often as you like, and the image never leaves <i>W</i>₀·SO(4).');
      } else if (m.id === 'oft') {
        var Q = skewFrom(q, 0), R = m.R(q, S.f), Q4 = mul(mul(Q, Q), mul(Q, Q)), IQ = add(I4, Q4, -1);
        var idErr = frob(add(mul(T(R), R), mul(IQ, IQ), -1)), orth = frob(add(mul(T(R), R), I4, -1));
        setBanner('seal', 'drift', 'HF PEFT’s default OFT (since 0.18) is quasi-orthogonal: <span class="nw"><i>R</i> = Cay(<i>Q</i>)(<i>I</i> − <i>Q</i>⁴)</span>, so <span class="nw"><i>R</i>ᵀ<i>R</i> = (<i>I</i> − <i>Q</i>⁴)²</span> (checked here to ' + fmtD(idErr) + ') and <span class="nw">‖<i>R</i>ᵀ<i>R</i> − <i>I</i>‖<sub>F</sub> = ' + fmtD(orth) + '</span>. The adapted weight already moves <i>K</i><sub>out</sub> by δ = ' + fmtD(r.d.Kout) + ' and σ(<i>W</i>) by ' + fmtD(r.d.spec) + ', and each merge cycle compounds the drift (Cor II.6).');
      } else if (m.id === 'out') {
        setBanner('moss', 'kept', 'Output side: <i>W</i> = <i>R</i>ᵀ<i>W</i>₀ keeps <i>K</i><sub>in</sub> = <i>W</i>ᵀ<i>W</i> and σ(<i>W</i>) but moves <i>K</i><sub>out</sub> (δ = ' + fmtD(r.d.Kout) + '). Left and right actions trade the two Gram matrices (Thm II.5(1)–(2)), which is why the side an implementation merges on matters.');
      } else if (m.id === 'boft') {
        var sg = signGuar(q, 0, 4, 's');
        if (sg.z >= 0) setBanner('seal', 'collapse', '<i>s</i>' + sub(sg.z + 1) + ' = 0 deletes neuron ' + (sg.z + 1) + ', so its cosines are not defined and neither cosine lamp can hold.');
        else if (sg.flip >= 0) setBanner('ochre', 'up to sign', '<i>s</i>₁ and <i>s</i>' + sub(sg.flip + 1) + ' have opposite signs, so every cosine between a neuron with <i>s</i><sub><i>i</i></sub> &gt; 0 and one with <i>s</i><sub><i>j</i></sub> &lt; 0 flips. The cosines move by up to ' + fmtD(r.d.cos) + ', while |cos| stays within ' + fmtD(r.d.abscos) + '. By Cor II.6, BOFT keeps |cos| while every <i>s</i><sub><i>i</i></sub> ≠ 0, and keeps the cosines exactly when the <i>s</i><sub><i>i</i></sub> also share one sign, since no cosine of <i>W</i>₀ is 0.');
        else setBanner('moss', 'kept', '<i>K</i><sub>out</sub> ↦ diag(<i>s</i>) <i>K</i><sub>out</sub> diag(<i>s</i>): neuron norms change (<i>K</i><sub>out</sub> δ = ' + fmtD(r.d.Kout) + ') but every cosine survives, because all <i>s</i><sub><i>i</i></sub> are ' + (q[0] > 0 ? 'positive' : 'negative, and each cosine picks up sign(<i>s</i><sub><i>i</i></sub><i>s</i><sub><i>j</i></sub>) = +1') + ' (Cor II.6). Press <b>s₁ ↦ −s₁</b> to see what one sign does.');
      } else if (m.id === 'hra') {
        var R2 = mul(hh(q.slice(0, 4)), hh(q.slice(4, 8))), tr = R2[0][0] + R2[1][1] + R2[2][2] + R2[3][3];
        var phi = Math.acos(Math.max(-1, Math.min(1, (tr - 2) / 2))) * 180 / Math.PI;
        var minus = Math.abs(tr) < 1e-12;
        var both = r.lit.Kout && r.lit.rank;
        setBanner(both ? 'moss' : 'ink', 'meet', (both ? 'HRA₂ keeps the neuron Gram <em>and</em> rank Δ<i>W</i> ≤ 2 at once. ' : '') + 'Im HRA₂ = (<i>W</i>₀ + 𝓜<sub>≤2</sub>) ∩ <i>W</i>₀·SO(4), an <b>image-level meet</b> (Thm II.7(a), for injective <i>W</i>₀ and even <i>r</i> ≤ <i>n</i> − 2). It is not a categorical product. Over a generic point of the meet, the fibre product of LoRA₂ and the rotation orbit has local dimension <i>d</i> + <i>r</i>² = 9, one more than HRA₂’s <i>rn</i> = 8 parameters (II.7(d)). Here <i>R</i> = <i>H</i><sub><i>u</i>₁</sub><i>H</i><sub><i>u</i>₂</sub> turns one plane by ' + phi.toFixed(1) + '°' +
          (minus ? ': eigenvalue −1, which no exact Cayley map produces, so exactly orthogonal OFT meets <i>W</i>₀ + 𝓜<sub>≤2</sub> only in a dense part of Im HRA₂ (II.7(d)).' : '.'));
      } else if (m.id === 'ia3') {
        var z = -1, ng = -1; for (i = 0; i < 4; i++) { if (q[i] === 0 && z < 0) z = i; if (q[i] < 0 && ng < 0) ng = i; }
        if (z >= 0) setBanner('seal', 'closure', 'ℓ' + sub(z + 1) + ' = 0 collapses neuron ' + (z + 1) + ': <i>W</i> leaves the <i>T</i>₄-orbit for its closure, so the row-line lamp goes out. The zeros of <i>W</i>₀ still stay zero, and ' + r.newZeros.length + ' new zeros appear (Prop II.9(e): the zero pattern itself is not invariant).');
        else if (ng >= 0 && signGuar(q, 0, 4, 'ℓ').flip >= 0) setBanner('ochre', 'up to sign', 'ℓ' + sub(ng + 1) + ' &lt; 0 reverses neuron ' + (ng + 1) + ' while some ℓ<sub><i>j</i></sub> &gt; 0. Every neuron stays on its line and |cos| is kept, but the cosines between neurons of opposite sign flip (δ = ' + fmtD(r.d.cos) + ').');
        else setBanner('moss', 'kept', 'Each neuron slides along its own line through 0 (Thm II.5(4)), so the cosines are kept while the ℓ<sub><i>i</i></sub> share one sign' + (ng >= 0 ? ', as they do here, all negative' : '') + ', and the zero of <i>W</i>₀ at (2,3) stays 0 (Prop II.9(e)). rank Δ<i>W</i> = ' + r.rank + (r.rank > 2 ? ': a torus is not low-rank.' : '.'));
      } else if (m.id === 'road') {
        var zc = q[2] === 0 ? 1 : q[3] === 0 ? 2 : 0;
        if (zc) setBanner('seal', 'closure', 'α' + sub(zc) + ' = 0 sends output pair ' + zc + ' to 0: still in the orbit closure (a real-linear space of dimension 4, Prop II.9(d)), but no longer a nonzero complex multiple of <i>X</i>, so the ℂ-line lamp goes out.');
        else setBanner('moss', 'kept', 'RoAd₁ is (IA)³ over ℂ: output pair <i>X</i> ∈ ℝ<sup>2×4</sup> ≅ ℂ⁴ is multiplied by α<i>e</i><sup><i>i</i>θ</sup>, so it stays on its complex line ℂ·<i>X</i>, and its input Gram scales by α²: measured ' + f3(r.pairRatio[0]) + ' and ' + f3(r.pairRatio[1]) + ' against α₁² = ' + f3(q[2] * q[2]) + ', α₂² = ' + f3(q[3] * q[3]) + ' (Prop II.9(d)).');
      } else if (m.id === 'dora') {
        if (allZero(q, 4, 12)) setBanner('moss', 'torus', 'At <i>B</i> = 0 DoRA is an output torus, and Δ<i>W</i> = (diag(μ<sub><i>i</i></sub>/‖<i>w</i><sub>0<i>i</i></sub>‖) − <i>I</i>)<i>W</i>₀ has rank ' + r.rank + (r.rank > 2 ? ', so DoRA already escapes LoRA’s rank bound here' : ', and up to rank <i>W</i>₀ = 4 in general, beyond LoRA’s bound') + ' (Prop II.8). Rows keep their lines while every μ<sub><i>i</i></sub> ≠ 0, and the cosines while the μ<sub><i>i</i></sub> also share one sign.');
        else setBanner('ink', 'image', 'Im DoRA₂ = Diag₄·(<i>W</i>₀ + 𝓜<sub>≤2</sub>): LoRA₂ followed by an output scale (Prop II.8). Once <i>B</i> ≠ 0 nothing on the board is guaranteed; here rank Δ<i>W</i> = ' + r.rank + (r.d0 === 16 ? ', and <i>d</i>(DoRA₂) = 16 = <i>mn</i>: at 4×4 the image has full dimension.' : ', and <i>d</i>(DoRA₂) = ' + r.d0 + '.') + ' Press <b>B ↦ 0</b> to return to the torus.');
      } else if (m.id === 'hira') {
        setBanner('moss', 'kept', 'Only the zeros of <i>W</i>₀ are guaranteed (Prop II.9(e)): <i>W</i><sub>23</sub> = ' + f2(r.W[1][2]) + '. ' + (r.rank > 2 ? 'The update is not low-rank: rank Δ<i>W</i> = ' + r.rank + ' &gt; <i>r</i> = 2' : 'Here rank Δ<i>W</i> = ' + r.rank + ', but HiRA’s update is not low-rank in general') + ', since <i>W</i>₀ ⊙ <i>BA</i> = Σ<sub><i>k</i></sub> diag(<i>b</i><sub><i>k</i></sub>)<i>W</i>₀diag(<i>a</i><sub><i>k</i></sub>) has rank at most <i>r</i>·rank <i>W</i>₀ = 8, which does not constrain a 4×4 matrix (Prop II.9(a)).' + (r.newZeros.length ? ' New zeros: ' + r.newZeros.length + '.' : ''));
      } else if (m.id === 'lora') {
        setBanner('moss', 'kept', 'LoRA keeps exactly one thing on this board: rank Δ<i>W</i> ≤ 2 (σ₃(Δ<i>W</i>) = ' + fmtD(r.svd[2]) + '). Neuron Gram, cosines, input Gram and spectrum all move.');
      } else if (m.id === 'qgoft') {
        if (r.lit.Kout) setBanner('ochre', 'orthogonal', 'Both blocks are exactly orthogonal at this point, so <i>K</i><sub>out</sub> holds <em>here</em>, but nothing forces it: qGOFT only penalises non-orthogonality, and any drift of a block breaks it (Cor II.6).');
        else if (r.lit.cos) setBanner('ochre', 'conformal', 'The cosines hold at this point, as they do whenever both blocks are orthogonal matrices times one common scale <i>c</i>, which only multiplies <i>K</i><sub>out</sub> by <i>c</i>². Nothing forces this. Give the two blocks different scales and the cosines move, since once <i>n</i> &gt; 2 a non-orthogonal block, even a conformal one, generically moves a cosine (Cor II.6).');
        else setBanner('seal', 'quasi', '<i>M</i>₁ and <i>M</i>₂ are free 2×2 blocks, and qGOFT only penalises non-orthogonality. Once <i>n</i> &gt; 2 a non-orthogonal block, conformal or not, generically moves a cosine, here by up to ' + fmtD(r.d.cos) + '; neither <i>K</i><sub>out</sub> nor the cosines are kept (Cor II.6).');
      }
    }

    /* ---------- neuron arrows ---------- */
    var CAM = { yaw: -0.62, pitch: 0.38 };
    function proj(p) {   // p = (c1, c2, c3) -> screen; c1 right, c2 up, c3 depth
      var X = p[0], Z = p[1], Y = p[2];
      var cy = Math.cos(CAM.yaw), sy = Math.sin(CAM.yaw), cp = Math.cos(CAM.pitch), sp = Math.sin(CAM.pitch);
      var sx = X * cy + Y * sy, depth = -X * sy + Y * cy, up = Z * cp + depth * sp;
      return [sx, up, depth];
    }
    function renderArrows(r) {
      var Wd = 360, Hd = 330, cx = 182, cy = 178, Rmax = 1.15 * Math.max.apply(null, ROW_NORM0), S;
      function P(c) { var s = proj(c); return [cx + S * s[0], cy - S * s[1], s[2]]; }
      var coords, ghost, extra = '', labs;
      if (st.frame === 'lab') {
        coords = r.W.map(function (w) { return V3.map(function (v) { return dot(w, v); }); });
        ghost = W0.map(function (w) { return V3.map(function (v) { return dot(w, v); }); });
        labs = ['v₁', 'v₂', 'v₃'];
      } else {
        var L = chol(r.K);
        coords = L.map(function (row) { return row.slice(0, 3); });
        ghost = REF.L.map(function (row) { return row.slice(0, 3); });
        labs = ['e₁', 'e₂', 'e₃'];
      }
      /* zoom out when a neuron outgrows W0's frame (large gains or free blocks), so no arrow is clipped */
      var big = 0;
      coords.forEach(function (c) { big = Math.max(big, Math.sqrt(dot(c, c))); });
      var Rv = Math.max(Rmax, 1.12 * big);
      S = 140 / Rv;
      var o = [];
      o.push('<svg viewBox="0 0 ' + Wd + ' ' + Hd + '" role="img" aria-label="Rows of W drawn as arrows in 3D, with the rows of W0 dashed">');
      o.push('<defs><clipPath id="gl-clip"><rect x="2" y="2" width="' + (Wd - 4) + '" height="' + (Hd - 4) + '"/></clipPath></defs><g clip-path="url(#gl-clip)">');
      /* floor grid in the (c1, c3) plane */
      var g = Rv, k;
      for (k = -2; k <= 2; k++) {
        var a1 = P([k * g / 2, 0, -g]), a2 = P([k * g / 2, 0, g]), b1 = P([-g, 0, k * g / 2]), b2 = P([g, 0, k * g / 2]);
        o.push('<line x1="' + a1[0].toFixed(1) + '" y1="' + a1[1].toFixed(1) + '" x2="' + a2[0].toFixed(1) + '" y2="' + a2[1].toFixed(1) + '" style="stroke:var(--rule)" stroke-width="0.8"/>');
        o.push('<line x1="' + b1[0].toFixed(1) + '" y1="' + b1[1].toFixed(1) + '" x2="' + b2[0].toFixed(1) + '" y2="' + b2[1].toFixed(1) + '" style="stroke:var(--rule)" stroke-width="0.8"/>');
      }
      /* axes */
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach(function (ax, ai) {
        var e = P(ax.map(function (x) { return x * g * 1.12; })), b = P(ax.map(function (x) { return -x * g * 0.35; }));
        o.push('<line x1="' + b[0].toFixed(1) + '" y1="' + b[1].toFixed(1) + '" x2="' + e[0].toFixed(1) + '" y2="' + e[1].toFixed(1) + '" style="stroke:var(--ink-3)" stroke-width="0.9" stroke-dasharray="2 3"/>');
        o.push('<text x="' + (e[0] + (ai === 0 ? 4 : ai === 2 ? 4 : -4)).toFixed(1) + '" y="' + (e[1] + (ai === 1 ? -4 : 4)).toFixed(1) + '" text-anchor="' + (ai === 1 ? 'end' : 'start') + '" style="fill:var(--ink-2);font-family:var(--f-body);font-size:15.5px;font-style:italic;paint-order:stroke;stroke:var(--paper);stroke-width:3px;stroke-linejoin:round">' + tsub(labs[ai]) + '</text>');
      });
      /* locked row lines (torus methods) */
      if (st.frame === 'lab' && r.lit.lines && r.guar.lines && r.guar.lines.ok) {
        ghost.forEach(function (gc) {
          var n = Math.sqrt(dot(gc, gc)), f = 1.75 * Rv / n, a = P(gc.map(function (x) { return x * f; })), b = P(gc.map(function (x) { return -x * f; }));
          o.push('<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' + b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) + '" style="stroke:var(--moss)" stroke-width="1.1" stroke-dasharray="1 3" stroke-linecap="round" opacity="0.9"/>');
        });
      }
      var O = P([0, 0, 0]);
      function arrow(c, cls, lab) {
        var t = P(c), f = P([c[0], 0, c[2]]), dx = t[0] - O[0], dy = t[1] - O[1], len = Math.sqrt(dx * dx + dy * dy) || 1, ux = dx / len, uy = dy / len, s = [];
        if (cls === 'cur') s.push('<line x1="' + t[0].toFixed(1) + '" y1="' + t[1].toFixed(1) + '" x2="' + f[0].toFixed(1) + '" y2="' + f[1].toFixed(1) + '" style="stroke:var(--ink-3)" stroke-width="0.7" stroke-dasharray="1 2.2"/><circle cx="' + f[0].toFixed(1) + '" cy="' + f[1].toFixed(1) + '" r="1.6" style="fill:var(--ink-3)"/>');
        var hl = Math.min(8, len * 0.45), bx = t[0] - ux * hl, by = t[1] - uy * hl;
        if (cls === 'cur') {
          s.push('<line x1="' + O[0].toFixed(1) + '" y1="' + O[1].toFixed(1) + '" x2="' + bx.toFixed(1) + '" y2="' + by.toFixed(1) + '" style="stroke:var(--tide)" stroke-width="2" stroke-linecap="round"/>');
          s.push('<path d="M' + t[0].toFixed(1) + ',' + t[1].toFixed(1) + 'L' + (bx - uy * 3.6).toFixed(1) + ',' + (by + ux * 3.6).toFixed(1) + 'L' + (bx + uy * 3.6).toFixed(1) + ',' + (by - ux * 3.6).toFixed(1) + 'Z" style="fill:var(--tide)"/>');
          s.push('<text x="' + (t[0] + ux * 12).toFixed(1) + '" y="' + (t[1] + uy * 12 + 5).toFixed(1) + '" text-anchor="middle" style="fill:var(--ink);font-family:var(--f-body);font-size:16px;font-style:italic;paint-order:stroke;stroke:var(--paper);stroke-width:3.5px;stroke-linejoin:round">' + tsub(lab) + '</text>');
        } else {
          s.push('<line x1="' + O[0].toFixed(1) + '" y1="' + O[1].toFixed(1) + '" x2="' + t[0].toFixed(1) + '" y2="' + t[1].toFixed(1) + '" style="stroke:var(--ink-3)" stroke-width="1.2" stroke-dasharray="4 3"/>');
          s.push('<circle cx="' + t[0].toFixed(1) + '" cy="' + t[1].toFixed(1) + '" r="2.2" style="fill:none;stroke:var(--ink-3)" stroke-width="1"/>');
        }
        return { z: t[2], svg: s.join('') };
      }
      ghost.forEach(function (c) { o.push(arrow(c, 'ghost').svg); });
      coords.map(function (c, i) { return arrow(c, 'cur', 'w' + sub(i + 1)); }).sort(function (a, b) { return b.z - a.z; }).forEach(function (a) { o.push(a.svg); });
      o.push('<circle cx="' + O[0].toFixed(1) + '" cy="' + O[1].toFixed(1) + '" r="2.4" style="fill:var(--ink)"/>');
      o.push('</g></svg>');
      arrowSvg.innerHTML = o.join('') + extra;
      var frozen = r.lit.Kout && !r.atQ0;
      if (st.frame === 'lab') arrowNote.innerHTML = hsub('Rows of <i>W</i> on <i>v</i>₁, <i>v</i>₂, <i>v</i>₃, the top right singular vectors of <i>W</i>₀ (' + (100 * ENERGY).toFixed(1) + '% of ‖<i>W</i>₀‖²<sub>F</sub>); dashed: <i>W</i>₀.' + (r.lit.lines && r.guar.lines && r.guar.lines.ok ? ' <b>Dotted: the locked row lines.</b>' : '') + (Rv > Rmax ? ' Zoomed out to fit the longer rows.' : ''));
      else arrowNote.innerHTML = hsub('Rows of <i>L</i>, where <i>K</i><sub>out</sub> = <i>LL</i>ᵀ (Cholesky): a picture of <i>K</i><sub>out</sub> alone, exact for <i>w</i>₁–<i>w</i>₃; <i>w</i>₄’s fourth coordinate (' + f2(chol(r.K)[3][3]) + ') is not drawn.' + (frozen ? ' <b><i>K</i><sub>out</sub> is fixed, so this picture cannot move.</b>' : '') + (Rv > Rmax ? ' Zoomed out to fit the longer rows.' : ''));
    }

    /* ---------- heatmaps ---------- */
    var COL = {};
    function readColors() {
      COL.paper = parseColor(A.css('--paper')); COL.pos = parseColor(A.css('--tide')); COL.neg = parseColor(A.css('--c-multiplicative'));
      COL.ink = parseColor(A.css('--ink'));
    }
    function cfill(t) {
      if (!isFinite(t)) return COL.paper;
      t = Math.max(-1, Math.min(1, t));
      var e = Math.pow(Math.abs(t), 0.8);
      if (window.d3 && d3.interpolateLab) {
        var c = d3.color(d3.interpolateLab(rgbStr(COL.paper), rgbStr(t >= 0 ? COL.pos : COL.neg))(e)).rgb();
        return [c.r, c.g, c.b];
      }
      return lerpC(COL.paper, t >= 0 ? COL.pos : COL.neg, e);
    }
    /* drawn 1:1 in CSS px (the svg is never scaled up or down), so the 12px mono cell values stay 12px at every width;
       the cell size follows the panel width, and the layout keeps it at 44px or more */
    function heatSVG(M, M0, dom, d, r, avail) {
      var ox = 20, oy = 17, c = Math.max(40, Math.min(46, Math.floor(((avail || 204) - ox - 2) / 4)));
      var w = ox + 4 * c + 2, hh2 = oy + 4 * c + 2, ck = Math.round(c * 0.3), o = [], i, j;
      var LBL = 'font-family:var(--f-body);font-style:italic;font-size:13px;fill:var(--ink-2)';
      var NUM = 'font-family:var(--f-mono);font-size:12px;font-variant-numeric:tabular-nums;';
      o.push('<svg class="gl-px" width="' + w + '" height="' + hh2 + '" viewBox="0 0 ' + w + ' ' + hh2 + '" role="img" aria-label="4 by 4 heatmap">');
      for (i = 0; i < 4; i++) {
        o.push('<text x="' + (ox + i * c + c / 2) + '" y="12" text-anchor="middle" style="' + LBL + '">' + tsub(d.cl + sub(i + 1)) + '</text>');
        o.push('<text x="9" y="' + (oy + i * c + c / 2 + 4.5) + '" text-anchor="middle" style="' + LBL + '">' + tsub(d.rl + sub(i + 1)) + '</text>');
      }
      for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) {
        var v = M[i][j], v0 = M0[i][j], x = ox + j * c, y = oy + i * c, f = cfill(v / dom), f0 = cfill(v0 / dom);
        o.push('<rect x="' + x + '" y="' + y + '" width="' + c + '" height="' + c + '" fill="' + rgbStr(f) + '"/>');
        o.push('<path d="M' + x + ',' + y + 'h' + ck + 'L' + x + ',' + (y + ck) + 'Z" fill="' + rgbStr(f0) + '"/>');
        var tc = contrast(f, COL.ink) >= contrast(f, COL.paper) ? 'var(--ink)' : 'var(--paper)';
        o.push('<text x="' + (x + c / 2) + '" y="' + (y + c / 2 + 4.2) + '" text-anchor="middle" style="' + NUM + 'fill:' + tc + '">' + f2(v) + '</text>');
      }
      for (i = 0; i <= 4; i++) {
        o.push('<line x1="' + ox + '" y1="' + (oy + i * c) + '" x2="' + (ox + 4 * c) + '" y2="' + (oy + i * c) + '" style="stroke:var(--rule)" stroke-width="0.8"/>');
        o.push('<line x1="' + (ox + i * c) + '" y1="' + oy + '" x2="' + (ox + i * c) + '" y2="' + (oy + 4 * c) + '" style="stroke:var(--rule)" stroke-width="0.8"/>');
      }
      if (d.key === 'W') {
        ZERO0.forEach(function (z) {
          var kept = r.lit.zeros;
          o.push('<rect x="' + (ox + z[1] * c + 1.5) + '" y="' + (oy + z[0] * c + 1.5) + '" width="' + (c - 3) + '" height="' + (c - 3) + '" fill="none" style="stroke:' + (kept ? 'var(--moss)' : 'var(--ink-3)') + '" stroke-width="' + (kept ? 2.2 : 1.2) + '"' + (kept ? '' : ' stroke-dasharray="3 2"') + '/>');
        });
        r.newZeros.forEach(function (z) {
          o.push('<rect x="' + (ox + z[1] * c + 2) + '" y="' + (oy + z[0] * c + 2) + '" width="' + (c - 4) + '" height="' + (c - 4) + '" fill="none" style="stroke:var(--ochre)" stroke-width="1.6" stroke-dasharray="2 2"/>');
        });
      }
      o.push('</svg>');
      return o.join('');
    }
    function renderHeats(r) {
      HM.forEach(function (d) {
        var M, M0, dom, cls = '', dt = '';
        if (d.key === 'Kout') { M = r.K; M0 = REF.K; dom = maxAbs(REF.K); cls = r.lit.Kout ? (r.guar.Kout && r.guar.Kout.ok ? 'on' : 'on') : r.guar.Kout && !r.guar.Kout.ok ? 'lost' : ''; dt = (r.lit.Kout ? 'fixed · ' : 'δ ') + fmtD(r.d.Kout); }
        else if (d.key === 'cos') {
          M = r.C; M0 = REF.C; dom = 1;
          cls = r.lit.cos ? 'on' : r.lit.abscos ? 'half' : r.guar.cos && !r.guar.cos.ok ? 'lost' : '';
          dt = r.lit.cos ? 'fixed · ' + fmtD(r.d.cos) : r.lit.abscos ? '|cos| fixed · ' + fmtD(r.d.abscos) : 'δ ' + fmtD(r.d.cos);
        }
        else if (d.key === 'Kin') { M = r.Ki; M0 = REF.Ki; dom = maxAbs(REF.Ki); cls = r.lit.Kin ? 'on' : ''; dt = (r.lit.Kin ? 'fixed · ' : 'δ ') + fmtD(r.d.Kin); }
        else { M = r.W; M0 = W0; dom = REF.mx; dt = '‖Δ<i>W</i>‖<sub>F</sub> ' + fmtD(frob(r.dW)); }
        d.el.className = 'gl-hm' + (cls ? ' ' + cls : '');
        d.dEl.innerHTML = hsub(dt);
        d.svg.innerHTML = heatSVG(M, M0, dom, d, r, Math.floor(d.svg.getBoundingClientRect().width));
      });
    }

    /* ---------- strips ---------- */
    function logY(v, lo, hi, top, bot) { var e = v > 0 ? Math.log(v) / Math.LN10 : lo; e = Math.max(lo, Math.min(hi, e)); return top + (hi - e) / (hi - lo) * (bot - top); }
    /* strips are drawn 1:1 in CSS px at the width of their panel: tick values 11px mono, labels and readouts 12px */
    var TICK = 'font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:var(--ink-2)';
    function stemSVG(vals, thr, w, hgt) {
      var lo = -17, hi = 1, pl = 42, pr = 26, top = 8, bot = hgt - 14, o = [], i;
      o.push('<svg class="gl-px" width="' + w + '" height="' + hgt + '" viewBox="0 0 ' + w + ' ' + hgt + '" role="img" aria-label="singular values on a log scale">');
      [0, -4, -8, -12, -16].forEach(function (e) {
        var y = logY(Math.pow(10, e), lo, hi, top, bot);
        o.push('<line x1="' + pl + '" y1="' + y.toFixed(1) + '" x2="' + (w - pr) + '" y2="' + y.toFixed(1) + '" style="stroke:var(--rule)" stroke-width="0.7"/>');
        o.push('<text x="' + (pl - 5) + '" y="' + (y + 4).toFixed(1) + '" text-anchor="end" style="' + TICK + '">' + (e === 0 ? '1' : tsub('10' + sup(e))) + '</text>');
      });
      var yt = logY(thr, lo, hi, top, bot);
      o.push('<line x1="' + pl + '" y1="' + yt.toFixed(1) + '" x2="' + (w - pr) + '" y2="' + yt.toFixed(1) + '" style="stroke:var(--ochre)" stroke-width="1" stroke-dasharray="4 3"/>');
      o.push('<text x="' + (w - pr + 3) + '" y="' + (yt + 4).toFixed(1) + '" text-anchor="start" style="font-family:var(--f-ui);font-weight:500;font-size:12px;fill:var(--ochre-ink)">tol</text>');
      var n = vals.length, span = (w - pl - pr), dx = span / n;
      for (i = 0; i < n; i++) {
        var x = pl + dx * (i + 0.5), y = logY(vals[i], lo, hi, top, bot), above = vals[i] > thr;
        o.push('<line x1="' + x.toFixed(1) + '" y1="' + bot + '" x2="' + x.toFixed(1) + '" y2="' + y.toFixed(1) + '" style="stroke:' + (above ? 'var(--tide)' : 'var(--ink-3)') + '" stroke-width="' + (above ? 1.6 : 1) + '"/>');
        o.push('<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (n > 10 ? 2.4 : 3.2) + '" style="' + (above ? 'fill:var(--tide)' : 'fill:var(--paper);stroke:var(--ink-3)') + '" stroke-width="1"/>');
      }
      o.push('<line x1="' + pl + '" y1="' + bot + '" x2="' + (w - pr) + '" y2="' + bot + '" style="stroke:var(--ink-3)" stroke-width="0.8"/>');
      o.push('</svg>');
      return o.join('');
    }
    function specSVG(sv, sv0, w, hgt) {
      var pl = 34, pr = 8, top = 10, bot = hgt - 34, ymax = Math.max(1.35 * sv0[0], 1.05 * sv[0]), o = [], i;
      function Y(v) { return bot - v / ymax * (bot - top); }
      o.push('<svg class="gl-px" width="' + w + '" height="' + hgt + '" viewBox="0 0 ' + w + ' ' + hgt + '" role="img" aria-label="singular values of W against W0">');
      var tk = ymax <= 2.6 ? 0.5 : ymax <= 5.5 ? 1 : ymax <= 11 ? 2 : 5;  // at most ~6 gridlines in a 92px plot
      for (var t = 0; t <= ymax + 1e-9; t += tk) {
        o.push('<line x1="' + pl + '" y1="' + Y(t).toFixed(1) + '" x2="' + (w - pr) + '" y2="' + Y(t).toFixed(1) + '" style="stroke:var(--rule)" stroke-width="0.7"/>');
        o.push('<text x="' + (pl - 5) + '" y="' + (Y(t) + 4).toFixed(1) + '" text-anchor="end" style="' + TICK + '">' + (tk < 1 ? t.toFixed(1) : String(t)) + '</text>');
      }
      var dx = (w - pl - pr) / 4, bw = Math.min(dx * 0.46, 40);
      for (i = 0; i < 4; i++) {
        var x = pl + dx * (i + 0.5);
        o.push('<rect x="' + (x - bw / 2).toFixed(1) + '" y="' + Y(sv[i]).toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + (bot - Y(sv[i])).toFixed(1) + '" style="fill:var(--tide)" opacity="0.85"/>');
        o.push('<line x1="' + (x - bw / 2 - 5).toFixed(1) + '" y1="' + Y(sv0[i]).toFixed(1) + '" x2="' + (x + bw / 2 + 5).toFixed(1) + '" y2="' + Y(sv0[i]).toFixed(1) + '" style="stroke:var(--ink)" stroke-width="1.3" stroke-dasharray="3 2"/>');
        o.push('<text x="' + x.toFixed(1) + '" y="' + (bot + 15) + '" text-anchor="middle" style="font-family:var(--f-body);font-style:italic;font-size:13px;fill:var(--ink-2)">' + tsub('σ' + sub(i + 1)) + '</text>');
        o.push('<text x="' + x.toFixed(1) + '" y="' + (bot + 30) + '" text-anchor="middle" style="font-family:var(--f-mono);font-size:12px;font-variant-numeric:tabular-nums;fill:var(--ink)">' + sv[i].toFixed(3) + '</text>');
      }
      o.push('<line x1="' + pl + '" y1="' + bot + '" x2="' + (w - pr) + '" y2="' + bot + '" style="stroke:var(--ink-3)" stroke-width="0.8"/>');
      o.push('</svg>');
      return o.join('');
    }
    function renderStrips(m, S, r) {
      var hgt = 126;
      function wOf(d) { return Math.max(200, Math.floor(d.svg.getBoundingClientRect().width || 250)); }
      STR[0].svg.innerHTML = specSVG(r.sv, REF.sv, wOf(STR[0]), hgt);
      STR[0].v.className = 'gl-stv' + (r.lit.spec ? ' ok' : '');
      STR[0].v.innerHTML = hsub(r.lit.spec ? 'fixed · δ ' + fmtD(r.d.spec) : 'δ ' + fmtD(r.d.spec));
      STR[0].s.innerHTML = hsub('Bars: σ(<i>W</i>). Dashed: σ(<i>W</i>₀) = ' + REF.sv.map(function (x) { return x.toFixed(3); }).join(', ') + '.');
      STR[1].svg.innerHTML = stemSVG(r.svd, r.thr, wOf(STR[1]), hgt);
      STR[1].v.className = 'gl-stv' + (r.rank <= 2 ? ' ok' : '');
      STR[1].v.textContent = 'rank ΔW = ' + r.rank;
      STR[1].s.innerHTML = hsub('σ(Δ<i>W</i>), log scale. Tolerance 10⁻¹⁰·σ₁(<i>W</i>₀); exact zeros sit at the floor.');
      STR[2].svg.innerHTML = stemSVG(r.jsv, TOL * r.jsv[0], wOf(STR[2]), hgt);
      var ap = r.atQ0 && r.jrank < r.d0;
      STR[2].v.className = 'gl-stv' + (ap ? ' ap' : '');
      STR[2].v.innerHTML = hsub(r.atQ0 ? 'dim S₁ = ' + r.jrank + (r.jrank < r.d0 ? ' &lt; d = ' : ' = d = ') + r.d0 : 'rank dρ = ' + r.jrank + ' · d = ' + r.d0);
      STR[2].s.innerHTML = hsub(r.atQ0 ? (ap ? 'At <i>q</i>₀: deficit ' + (r.d0 - r.jrank) + ' &gt; 0, an <b>apex</b>' + (m.id === 'hra' ? ' (Thm II.7(b))' : m.id === 'lora' ? ' (Thm II.2(b))' : '') + '.' : 'At <i>q</i>₀: deficit 0, a regular point.') + ' |<i>M</i>| = ' + m.p + '.'
        : 'Analytic Jacobian, |<i>M</i>| = ' + m.p + ' columns; <i>d</i>(<i>M</i>) = rank at a generic point.');
    }

    function renderLegend() {
      readColors();
      var stops = [-1, -0.5, 0, 0.5, 1].map(function (t, i) { return '<stop offset="' + (i * 25) + '%" stop-color="' + rgbStr(cfill(t)) + '"/>'; }).join('');
      legend.innerHTML = hsub('<span class="gl-li"><svg width="74" height="10" aria-hidden="true"><defs><linearGradient id="gl-grad">' + stops + '</linearGradient></defs><rect width="74" height="10" rx="2" fill="url(#gl-grad)" style="stroke:var(--rule)"/></svg><span>− 0 + (per panel, scaled to W₀)</span></span>' +
        '<span class="gl-li"><svg width="16" height="16" aria-hidden="true"><rect width="16" height="16" fill="' + rgbStr(cfill(0.7)) + '"/><path d="M0,0h8L0,8Z" fill="' + rgbStr(cfill(-0.7)) + '"/></svg><span>corner = value at W₀ (invisible when unchanged)</span></span>' +
        '<span class="gl-li"><svg width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" fill="none" style="stroke:var(--moss)" stroke-width="2"/></svg><span>zero of W₀ kept</span></span>' +
        '<span class="gl-li"><svg width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" fill="none" style="stroke:var(--ochre)" stroke-width="1.6" stroke-dasharray="2 2"/></svg><span>new zero</span></span>');
    }
    foot.innerHTML = hsub('Toy: <i>W</i>₀ ∈ ℝ<sup>4×4</sup> with entries in ¼ℤ, one zero at (2,3), σ(<i>W</i>₀) = ' + REF.sv.map(function (x) { return x.toFixed(4); }).join(', ') +
      '; float64. A lamp is lit when its invariant equals the value at <i>W</i>₀ to 10⁻¹⁰: relative Frobenius error for <i>K</i><sub>out</sub>, <i>K</i><sub>in</sub>; absolute for cosines; relative to σ₁ for spectra; sine of the angle for lines. Ranks count singular values above 10⁻¹⁰·σ₁ (one-sided Jacobi SVD). ' +
      'LoRA, HiRA, DoRA: <i>r</i> = 2, <i>s</i> = 1, <i>B</i>₀ = 0, fixed <i>A</i>₀. HRA: <i>r</i> = 2, <code>apply_GS=False</code>, paired init. BOFT: exact Cayley 2×2 blocks, two butterfly factors. <i>d</i>(<i>M</i>) is the Jacobian rank at a seeded generic point. ' +
      'Theorem numbers refer to the companion paper, Exposé II.');

    /* ---------- boot ---------- */
    chrome(); buildMats(); renderLegend(); render();
    A.onTheme(function () { readColors(); renderLegend(); render(); });
    /* heatmaps and strips are drawn 1:1 in CSS px, so they are redrawn when their panels change width */
    if ('ResizeObserver' in window) {
      var lastHW = -1, lastSW = -1;
      new ResizeObserver(A.debounce(function () {
        var hw = HM[0].svg.clientWidth, sw = STR[0].svg.clientWidth;
        if (!last || (Math.abs(hw - lastHW) < 1 && Math.abs(sw - lastSW) < 1)) return;
        lastHW = hw; lastSW = sw;
        renderHeats(last); renderStrips(cur(), curS(), last);
      }, 80)).observe(stage);
    }

    /* read-only hook for headless verification pages */
    el.__gram = {
      evaluate: function (id, q, f) { return evaluate(methodById(id), q, f || { cn: false }); },
      methods: METHODS.map(function (m) { return { id: m.id, p: m.p, q0: m.q0.slice(), demo: m.demo.slice() }; }),
      gen: GEN, last: function () { return last; }, select: selectMethod, state: st,
      jac: function (id, q, f) { return methodById(id).jac(q, f || { cn: false }); },
      map: function (id, q, f) { return methodById(id).map(q, f || { cn: false }); },
      extra: function () { bExtra.click(); }, reset: function () { bReset.click(); }, random: function () { bRand.click(); },
      wander: function () { bWander.click(); },
      frame: function (v) { frameSeg.set(v); st.frame = v; render(); },
      setQ: function (q) { stopWander(); curS().q = q.slice(); syncCells(); render(); },
      lamps: function () { var o = {}; LAMPS.forEach(function (L) { o[L.key] = lampEls[L.key].el.className.replace('gl-lamp ', '') + ' | ' + lampEls[L.key].ls.textContent; }); return o; },
      banner: function () { return banner.textContent; }
    };
  });
})();
