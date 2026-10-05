/* Fig. "rebase" — Rebasing reach: what merge-and-restart can and cannot reach.
   Thm V.3 (rebasing closure) of theory/framework.md, with its repaired corollaries 1, 2, 4, 5 and 7.

   Everything is float64 and seeded. W0 in R^{24x16} is injective, with singular values 2 -> 0.5 evenly spaced
   (so every gap sigma_r - sigma_{r+1} equals 0.1), and n = 16 input coordinates.

   Tab A (Thm V.3(a), Cor 1, Cor 2). ReLoRA adds a fresh random rank-r product B_k A_k each cycle; the rank of the
   accumulated update is measured by a one-sided Jacobi SVD and compared with min(Kr, m, n). LoRA-XS updates
   U_r R_k V_r^T with the frames frozen at W0 (idempotent: R_K = R_1, r^2 dimensions) or recomputed by SVD of the
   merged weight. An update inside the current frame keeps that frame's span (the SVD is block diagonal), so the
   recomputed frame moves only when an update reorders singular values across the r-th gap (never while
   ||R_k||_2 < sigma_r - sigma_{r+1}, by Weyl); the figure counts the cycles in which it moved.

   Tab B (Thm V.3(b), (c), Cor 2, Cor 4, Cor 7). Factors act on coordinate blocks I_j of {1..16}. Union-find gives
   the components C of the hypergraph, Lie<so(I_j)> = (+)_C so(C) has dimension sum_C C(|C|,2), and one partial
   Gram W_{:,C} W_{:,C}^T per component is preserved by exactly orthogonal factors. dim R_K is measured as the rank of
   the right-trivialised Jacobian of the K-cycle product map at a random point (incremental Gram-Schmidt).
   The Cayley-Neumann switch (HF PEFT default) uses R = I + 2Q + 2Q^2 + 2Q^3 + Q^4, which is not orthogonal.

   Tab C (Cor 5). A target R in SO(16) with rank(R - I) = k is written as exactly k reflections (two per rotation
   plane, as in the Cartan-Dieudonne-Scherk step of the proof), grouped into HRA_r elements, padded with H_u H_u = I. */
(function () {
  'use strict';
  var NSV = 'http://www.w3.org/2000/svg';
  var M = 24, N = 16;
  var DIM_SO = N * (N - 1) / 2;     // 120
  var RTOL = 1e-9;                  // rank tolerance relative to sigma_1 (one-sided Jacobi SVD, no squaring)
  var ATOL = 1e-12;                 // absolute floor, so that a rounding-level matrix has rank 0
  var SIG = (function () { var s = []; for (var i = 0; i < N; i++) s.push(2 - 1.5 * i / (N - 1)); return s; })();
  var UPD = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
  var KB = 16;                      // cycles computed in tab B

  /* ---------- formatting ---------- */
  var SUPC = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  var SUBC = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
  function sup(n) { return String(n).split('').map(function (c) { return SUPC[c] || c; }).join(''); }
  function subs(n) { return String(n).split('').map(function (c) { return SUBC[c] || c; }).join(''); }
  function sci(x, d) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '∞';
    var p = x.toExponential(d == null ? 1 : d).split('e'), e = parseInt(p[1], 10);
    if (e === 0) return p[0];
    return p[0] + '×10' + sup(e);
  }
  function small(x) { if (x === 0) return '0'; if (Math.abs(x) < 1e-3) return sci(x, 1); return String(+x.toPrecision(3)); }
  function tiny(x) { return x === 0 ? '0' : sci(x, 0).replace(/^1×/, ''); }
  function pct(x) { if (x === 0) return '0%'; if (x < 1e-4) return sci(x * 100, 1) + '%'; return (+(x * 100).toPrecision(2)) + '%'; }
  function binom2(k) { return k * (k - 1) / 2; }
  /* Plex Mono's superscript glyphs are about 5px tall at 12px; mono readouts set exponents as real digits in <sup> */
  var SUPINV = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };
  function supify(html) { return String(html).replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, function (m) { return '<sup>' + m.split('').map(function (c) { return SUPINV[c]; }).join('') + '</sup>'; }); }

  /* ================= engine (pure numerics; exposed as Atlas._rebase for headless tests) ================= */
  function makeEngine(At) {
    var LA = At.LA;
    function mat(m, n) { return LA.zeros(m, n); }
    function eye(n) { return LA.eye(n); }
    function mul(X, Y) { return LA.mul(X, Y); }
    function tr(X) { return LA.T(X); }
    function frob(X) { return LA.frob(X); }
    function clone(X) { return X.map(function (r) { return Float64Array.from(r); }); }
    function cols(X, r) { var m = X.length, Y = mat(m, r); for (var i = 0; i < m; i++) for (var j = 0; j < r; j++) Y[i][j] = X[i][j]; return Y; }

    /* one-sided (Hestenes) Jacobi SVD, m >= n; singular values accurate to ~eps * sigma_1 */
    function svd(Ain) {
      var m = Ain.length, n = Ain[0].length, i, j, p, q, k, a, b;
      var U = clone(Ain), V = eye(n);
      for (var sweep = 0; sweep < 80; sweep++) {
        var rot = 0;
        for (p = 0; p < n - 1; p++) for (q = p + 1; q < n; q++) {
          var al = 0, be = 0, ga = 0;
          for (i = 0; i < m; i++) { var up = U[i][p], uq = U[i][q]; al += up * up; be += uq * uq; ga += up * uq; }
          if (ga === 0 || Math.abs(ga) <= 1e-15 * Math.sqrt(al * be)) continue;
          rot++;
          var ze = (be - al) / (2 * ga);
          var t = (ze >= 0 ? 1 : -1) / (Math.abs(ze) + Math.sqrt(1 + ze * ze));
          var c = 1 / Math.sqrt(1 + t * t), s = c * t;
          for (i = 0; i < m; i++) { a = U[i][p]; b = U[i][q]; U[i][p] = c * a - s * b; U[i][q] = s * a + c * b; }
          for (i = 0; i < n; i++) { a = V[i][p]; b = V[i][q]; V[i][p] = c * a - s * b; V[i][q] = s * a + c * b; }
        }
        if (!rot) break;
      }
      var sv = [], idx = [];
      for (j = 0; j < n; j++) { var nr = 0; for (i = 0; i < m; i++) nr += U[i][j] * U[i][j]; sv.push(Math.sqrt(nr)); idx.push(j); }
      idx.sort(function (x, y) { return sv[y] - sv[x]; });
      var Uo = mat(m, n), Vo = mat(n, n), so = [];
      for (k = 0; k < n; k++) {
        j = idx[k]; so.push(sv[j]);
        var inv = sv[j] > 0 ? 1 / sv[j] : 0;
        for (i = 0; i < m; i++) Uo[i][k] = U[i][j] * inv;
        for (i = 0; i < n; i++) Vo[i][k] = V[i][j];
      }
      return { s: so, U: Uo, V: Vo };
    }
    /* relative tolerance RTOL * sigma_1, with an absolute floor ATOL (every matrix here has entries of order 0.01..2) */
    function rankS(s) { var tol = Math.max(RTOL * s[0], ATOL), r = 0; for (var i = 0; i < s.length; i++) if (s[i] > tol) r++; return r; }
    function rank(X) { return rankS(svd(X).s); }
    /* Gauss-Jordan inverse with partial pivoting */
    function inv(X) {
      var n = X.length, Aa = clone(X), I = eye(n), i, j, k;
      for (k = 0; k < n; k++) {
        var p = k, mx = Math.abs(Aa[k][k]);
        for (i = k + 1; i < n; i++) if (Math.abs(Aa[i][k]) > mx) { mx = Math.abs(Aa[i][k]); p = i; }
        if (p !== k) { var t = Aa[p]; Aa[p] = Aa[k]; Aa[k] = t; t = I[p]; I[p] = I[k]; I[k] = t; }
        var d = Aa[k][k];
        for (j = 0; j < n; j++) { Aa[k][j] /= d; I[k][j] /= d; }
        for (i = 0; i < n; i++) if (i !== k) {
          var f = Aa[i][k]; if (f === 0) continue;
          for (j = 0; j < n; j++) { Aa[i][j] -= f * Aa[k][j]; I[i][j] -= f * I[k][j]; }
        }
      }
      return I;
    }
    function det(X) {
      var n = X.length, Aa = clone(X), dd = 1, i, j, k;
      for (k = 0; k < n; k++) {
        var p = k, mx = Math.abs(Aa[k][k]);
        for (i = k + 1; i < n; i++) if (Math.abs(Aa[i][k]) > mx) { mx = Math.abs(Aa[i][k]); p = i; }
        if (mx === 0) return 0;
        if (p !== k) { var t = Aa[p]; Aa[p] = Aa[k]; Aa[k] = t; dd = -dd; }
        dd *= Aa[k][k];
        for (i = k + 1; i < n; i++) { var f = Aa[i][k] / Aa[k][k]; for (j = k; j < n; j++) Aa[i][j] -= f * Aa[k][j]; }
      }
      return dd;
    }
    /* m x n with orthonormal columns (Gram-Schmidt twice) */
    function orthCols(m, n, rnd) {
      var C = [], j, k, i;
      for (j = 0; j < n; j++) {
        var v = new Float64Array(m);
        for (i = 0; i < m; i++) v[i] = rnd.normal();
        for (var pass = 0; pass < 2; pass++) for (k = 0; k < C.length; k++) { var d = 0; for (i = 0; i < m; i++) d += v[i] * C[k][i]; for (i = 0; i < m; i++) v[i] -= d * C[k][i]; }
        var nr = 0; for (i = 0; i < m; i++) nr += v[i] * v[i]; nr = Math.sqrt(nr);
        for (i = 0; i < m; i++) v[i] /= nr;
        C.push(v);
      }
      var X = mat(m, n); for (i = 0; i < m; i++) for (j = 0; j < n; j++) X[i][j] = C[j][i];
      return X;
    }
    var BASE = null;
    function base() {
      if (BASE) return BASE;
      var rnd = At.rng(20261001);
      var U = orthCols(M, N, rnd), V = orthCols(N, N, rnd), W0 = mat(M, N);
      for (var i = 0; i < M; i++) for (var j = 0; j < N; j++) { var s = 0; for (var k = 0; k < N; k++) s += U[i][k] * SIG[k] * V[j][k]; W0[i][j] = s; }
      var f = svd(W0);
      BASE = { W0: W0, s: f.s, rank: rankS(f.s), K0: mul(W0, tr(W0)) };
      return BASE;
    }

    /* ---------- Tab A: additive rebasing ---------- */
    function runA(r, upd, seed, Kmax) {
      var W0 = base().W0;
      var rA = At.rng(1000 + 31 * seed + r), rX = At.rng(5000 + 31 * seed + r);
      var f0 = svd(W0), U0 = cols(f0.U, r), V0 = cols(f0.V, r);
      var D = mat(M, N), Df = mat(M, N), W = clone(W0), prevU = null;
      var out = { r: r, upd: upd, Kmax: Kmax, relora: [0], bound: [0], dimRe: [0], xsF: [0], xsR: [0], moved: [false], nMoved: [0], gap0: f0.s[r - 1] - f0.s[r] };
      var nm = 0;
      for (var k = 1; k <= Kmax; k++) {
        var B = LA.randn(M, r, rA), Ak = LA.randn(r, N, rA, 1 / Math.sqrt(N));
        D = LA.add(D, mul(B, Ak));
        out.relora.push(rank(D));
        var sb = Math.min(k * r, M, N);
        out.bound.push(sb); out.dimRe.push(sb * (M + N - sb));
        var R = LA.randn(r, r, rX, upd / r);          // E ||R_k||_F^2 = upd^2
        Df = LA.add(Df, mul(mul(U0, R), tr(V0)));
        out.xsF.push(rank(Df));
        var fk = svd(W), Uk = cols(fk.U, r), Vk = cols(fk.V, r), mv = false;
        if (prevU) { var cc = frob(mul(tr(prevU), Uk)); mv = 2 * r - 2 * cc * cc > 1e-12; }
        if (mv) nm++;
        out.moved.push(mv); out.nMoved.push(nm);
        prevU = Uk;
        W = LA.add(W, mul(mul(Uk, R), tr(Vk)));
        out.xsR.push(rank(LA.add(W, W0, -1)));
      }
      out.Kstar = Math.ceil(Math.min(M, N) / r);
      return out;
    }

    /* ---------- Tab B: block-orthogonal factors ---------- */
    function contiguous(b) { var bl = []; for (var s = 0; s < N; s += b) { var x = []; for (var i = s; i < s + b; i++) x.push(i); bl.push(x); } return bl; }
    /* BOFT butterfly factor at level l with block size b: half-blocks of size b/2, half-block c paired with c XOR 2^l
       (level 0 is the contiguous partition, i.e. block OFT) */
    function butterflyLevel(b, l) {
      var hh = b / 2, H = N / hh, bl = [];
      for (var c = 0; c < H; c++) {
        var c2 = c ^ (1 << l);
        if (c2 > c && c2 < H) { var x = [], i; for (i = c * hh; i < c * hh + hh; i++) x.push(i); for (i = c2 * hh; i < c2 * hh + hh; i++) x.push(i); bl.push(x); }
      }
      return bl;
    }
    function maxDepth(b) { return 1 + Math.round(Math.log(N / b) / Math.LN2); }
    function randomPartition(b, rnd) {
      var p = []; for (var i = 0; i < N; i++) p.push(i);
      for (i = N - 1; i > 0; i--) { var j = Math.floor(rnd() * (i + 1)), t = p[i]; p[i] = p[j]; p[j] = t; }
      var bl = []; for (var s = 0; s < N; s += b) bl.push(p.slice(s, s + b).sort(function (x, y) { return x - y; }));
      return bl;
    }
    function cyclesFor(sc, b, mp, planes, seed, Kmax) {
      var cyc = [], rnd = At.rng(777 + 13 * seed + b);
      for (var k = 0; k < Kmax; k++) {
        var fs = [];
        if (sc === 'block') fs.push({ blocks: contiguous(b), level: 0 });
        else if (sc === 'boft') for (var l = 0; l < mp; l++) fs.push({ blocks: butterflyLevel(b, l), level: l });
        else if (sc === 'rotate') fs.push({ blocks: k === 0 ? contiguous(b) : randomPartition(b, rnd), level: 0 });
        else planes.forEach(function (p, pi) { fs.push({ blocks: [p.slice()], level: pi }); });
        cyc.push(fs);
      }
      return cyc;
    }
    function components(cyc, K) {
      var par = []; for (var i = 0; i < N; i++) par.push(i);
      function find(x) { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; }
      for (var k = 0; k < K && k < cyc.length; k++) cyc[k].forEach(function (f) { f.blocks.forEach(function (I) { for (var t = 1; t < I.length; t++) { var a = find(I[0]), c = find(I[t]); if (a !== c) par[c] = a; } }); });
      var roots = {}, lab = [], sizes = [], order = [];
      for (i = 0; i < N; i++) { var rt = find(i); if (!(rt in roots)) { roots[rt] = order.length; order.push(rt); sizes.push(0); } lab.push(roots[rt]); sizes[roots[rt]]++; }
      var lie = 0; sizes.forEach(function (s) { lie += binom2(s); });
      return { lab: lab, sizes: sizes, count: sizes.length, lie: lie };
    }
    function skewOn(blocks, rnd, scale) {
      var Q = mat(N, N);
      blocks.forEach(function (I) {
        var sc = scale / Math.sqrt(I.length);
        for (var a = 0; a < I.length; a++) for (var c = a + 1; c < I.length; c++) { var x = rnd.normal() * sc; Q[I[a]][I[c]] = x; Q[I[c]][I[a]] = -x; }
      });
      return Q;
    }
    function cayley(Q) { var I = eye(N); return mul(LA.add(I, Q), inv(LA.add(I, Q, -1))); }
    function neumann(Q) { var Q2 = mul(Q, Q), Q3 = mul(Q2, Q), Q4 = mul(Q3, Q), I = eye(N), R = mat(N, N); for (var i = 0; i < N; i++) for (var j = 0; j < N; j++) R[i][j] = I[i][j] + 2 * Q[i][j] + 2 * Q2[i][j] + 2 * Q3[i][j] + Q4[i][j]; return R; }
    /* incremental span with two-pass Gram-Schmidt */
    function Span(dim) { this.dim = dim; this.B = []; this.minIn = Infinity; this.maxOut = 0; }
    Span.prototype.add = function (v) {
      var n = this.dim, i, k, nv = 0;
      for (i = 0; i < n; i++) nv += v[i] * v[i];
      nv = Math.sqrt(nv); if (nv === 0) return false;
      var w = Float64Array.from(v), nw = nv;
      for (var pass = 0; pass < 2; pass++) {
        for (k = 0; k < this.B.length; k++) { var b = this.B[k], d = 0; for (i = 0; i < n; i++) d += b[i] * w[i]; for (i = 0; i < n; i++) w[i] -= d * b[i]; }
        var prev = nw; nw = 0; for (i = 0; i < n; i++) nw += w[i] * w[i]; nw = Math.sqrt(nw);
        if (nw > 0.7 * prev) break;            /* twice is enough (Kahan); one pass when little was removed */
      }
      var rel = nw / nv;
      if (rel <= 1e-8) { if (rel > this.maxOut) this.maxOut = rel; return false; }
      if (rel < this.minIn) this.minIn = rel;
      for (i = 0; i < n; i++) w[i] /= nw;
      this.B.push(w); return true;
    };
    /* dim R_K for K = 1..cyc.length: rank of the right-trivialised Jacobian of (factors) -> F_1 F_2 ... F_P.
       Every factor is block diagonal on disjoint blocks, so all per-block work is done in b x b coordinates. */
    function lmul(X, Y) { var n = X.length, k = Y.length, m = Y[0].length, Z = mat(n, m); for (var i = 0; i < n; i++) for (var p = 0; p < k; p++) { var x = X[i][p]; if (x === 0) continue; for (var j = 0; j < m; j++) Z[i][j] += x * Y[p][j]; } return Z; }
    function cayleyL(Q) { var b = Q.length, I = eye(b); return lmul(LA.add(I, Q), inv(LA.add(I, Q, -1))); }
    function neumannL(Q) { var b = Q.length, Q2 = lmul(Q, Q), Q3 = lmul(Q2, Q), Q4 = lmul(Q3, Q), R = eye(b); for (var i = 0; i < b; i++) for (var j = 0; j < b; j++) R[i][j] += 2 * Q[i][j] + 2 * Q2[i][j] + 2 * Q3[i][j] + Q4[i][j]; return R; }
    function jacDims(cyc, map, seed) {
      var rnd = At.rng(4242 + seed), exact = map === 'exact';
      var full = exact ? DIM_SO : N * N, span = new Span(full);
      var T = eye(N), Ti = eye(N), dims = [];
      for (var k = 0; k < cyc.length; k++) {
        cyc[k].forEach(function (f) {
          var loc = f.blocks.map(function (I) {
            var b = I.length, sc = 0.3 / Math.sqrt(b), Q = mat(b, b);
            for (var a = 0; a < b; a++) for (var c = a + 1; c < b; c++) { var x = rnd.normal() * sc; Q[a][c] = x; Q[c][a] = -x; }
            var F = exact ? cayleyL(Q) : neumannL(Q);
            return { I: I, Q: Q, F: F, Fi: exact ? null : inv(F) };
          });
          if (span.B.length < full) loc.forEach(function (B) {
            var I = B.I, b = I.length, i, j, t;
            var Q2 = exact ? null : lmul(B.Q, B.Q), Q3 = exact ? null : lmul(Q2, B.Q);
            var TI = exact ? null : mat(N, b), TiI = exact ? null : mat(b, N);
            if (!exact) for (i = 0; i < N; i++) for (j = 0; j < b; j++) { TI[i][j] = T[i][I[j]]; TiI[j][i] = Ti[I[j]][i]; }
            for (var a = 0; a < b; a++) for (var c = a + 1; c < b; c++) {
              if (span.B.length >= full) return;
              var v; t = 0;
              if (exact) {
                /* dF F^{-1} ranges over (+) so(I) (the Cayley map is a local diffeomorphism and F is block
                   diagonal and orthogonal), so T L_ac T^T with T orthogonal spans the tangent */
                var ia = I[a], ic = I[c];
                v = new Float64Array(DIM_SO);
                for (i = 0; i < N; i++) for (j = i + 1; j < N; j++) v[t++] = T[i][ia] * T[j][ic] - T[i][ic] * T[j][ia];
              } else {
                /* d/dQ of I + 2Q + 2Q^2 + 2Q^3 + Q^4 along L, times F^{-1}, conjugated by the prefix T */
                var Q = B.Q, L = mat(b, b); L[a][c] = 1; L[c][a] = -1;
                var LQ = lmul(L, Q), QL = lmul(Q, L), LQ2 = lmul(L, Q2), QLQ = lmul(Q, LQ), Q2L = lmul(Q2, L);
                var LQ3 = lmul(L, Q3), QLQ2 = lmul(Q, LQ2), Q2LQ = lmul(Q2, LQ), Q3L = lmul(Q3, L), dN = mat(b, b);
                for (i = 0; i < b; i++) for (j = 0; j < b; j++) dN[i][j] = 2 * L[i][j] + 2 * (LQ[i][j] + QL[i][j]) + 2 * (LQ2[i][j] + QLQ[i][j] + Q2L[i][j]) + LQ3[i][j] + QLQ2[i][j] + Q2LQ[i][j] + Q3L[i][j];
                var X = lmul(lmul(TI, lmul(dN, B.Fi)), TiI);
                v = new Float64Array(N * N);
                for (i = 0; i < N; i++) for (j = 0; j < N; j++) v[t++] = X[i][j];
              }
              span.add(v);
            }
          });
          /* T <- T F, Ti <- F^{-1} Ti, block by block */
          loc.forEach(function (B) {
            var I = B.I, b = I.length, i, j, p, row = new Float64Array(b);
            for (i = 0; i < N; i++) {
              for (j = 0; j < b; j++) { var sacc = 0; for (p = 0; p < b; p++) sacc += T[i][I[p]] * B.F[p][j]; row[j] = sacc; }
              for (j = 0; j < b; j++) T[i][I[j]] = row[j];
            }
            if (!exact) {
              var blk = mat(b, N);
              for (p = 0; p < b; p++) for (i = 0; i < N; i++) { var s2 = 0; for (j = 0; j < b; j++) s2 += B.Fi[p][j] * Ti[I[j]][i]; blk[p][i] = s2; }
              for (p = 0; p < b; p++) for (i = 0; i < N; i++) Ti[I[p]][i] = blk[p][i];
            }
          });
        });
        dims.push(span.B.length);
      }
      return { dims: dims, minIn: span.minIn, maxOut: span.maxOut, full: full };
    }
    /* merge-and-restart run with small factors (skew entries ~0.05, as in the Num of Thm V.3): what is preserved */
    function driftRun(cyc, K, map, seed, comp, b, sc) {
      var bs = base(), W0 = bs.W0, rnd = At.rng(9090 + seed), exact = map === 'exact';
      var R = eye(N), i, j, a, loew = Infinity, rhoMax = 0;
      /* K_out = W0 (R R^T) W0^T with W0 injective, so K_out decreases in the Loewner order at a merge
         iff R R^T does: track the smallest eigenvalue of R_prev R_prev^T - R R^T over every merge */
      var RRprev = eye(N);
      for (var k = 0; k < K; k++) {
        cyc[k].forEach(function (f) {
          var Q = mat(N, N);
          f.blocks.forEach(function (I) { for (var x = 0; x < I.length; x++) for (var y = x + 1; y < I.length; y++) { var q = (rnd.normal() - rnd.normal()) * 0.07 / 2; Q[I[x]][I[y]] = q; Q[I[y]][I[x]] = -q; } });
          if (!exact) { var ev = LA.symEig(mul(Q, tr(Q))).values; for (var e = 0; e < ev.length; e++) rhoMax = Math.max(rhoMax, Math.sqrt(Math.max(0, ev[e]))); }
          R = mul(R, exact ? cayley(Q) : neumann(Q));
        });
        if (!exact) {
          var RR = mul(R, tr(R)), dlt = LA.add(RRprev, RR, -1), mn = Math.min.apply(null, LA.symEig(dlt).values);
          loew = Math.min(loew, mn); RRprev = RR;
        }
      }
      var W = mul(W0, R), Kw = mul(W, tr(W));
      var kout = frob(LA.add(Kw, bs.K0, -1)) / frob(bs.K0);
      var tr0 = 0, trK = 0; for (i = 0; i < M; i++) { tr0 += bs.K0[i][i]; trK += Kw[i][i]; }
      var shrink = 1 - trK / tr0;
      var orth = frob(LA.add(mul(tr(R), R), eye(N), -1));
      var cross = 0;
      for (i = 0; i < N; i++) for (j = 0; j < N; j++) if (comp.lab[i] !== comp.lab[j]) cross = Math.max(cross, Math.abs(R[i][j]));
      function pgDrift(set) {
        var G0 = mat(M, M), G1 = mat(M, M);
        set.forEach(function (c) { for (var x = 0; x < M; x++) for (var y = 0; y < M; y++) { G0[x][y] += W0[x][c] * W0[y][c]; G1[x][y] += W[x][c] * W[y][c]; } });
        return frob(LA.add(G1, G0, -1)) / frob(G0);
      }
      var pmax = 0;
      for (a = 0; a < comp.count; a++) { var set = []; for (i = 0; i < N; i++) if (comp.lab[i] === a) set.push(i); pmax = Math.max(pmax, pgDrift(set)); }
      var J1 = null;
      if (sc !== 'givens') { var s1 = []; for (i = 0; i < b; i++) s1.push(i); J1 = { drift: pgDrift(s1), isComp: comp.sizes[comp.lab[0]] === b } ; }
      var sv = svd(W).s, sd = 0, svUp = 0; for (i = 0; i < N; i++) { sd = Math.max(sd, Math.abs(sv[i] - bs.s[i]) / bs.s[i]); svUp = Math.max(svUp, sv[i] - bs.s[i]); }
      /* loewner: true when every merge kept R_prev R_prev^T - R R^T positive semidefinite (to rounding) */
      return { kout: kout, orth: orth, cross: cross, pmax: pmax, J1: J1, spec: sd, shrink: shrink, svDown: svUp <= 1e-12, loewner: exact ? null : loew >= -1e-12, rho: rhoMax };
    }
    function runB(sc, b, mp, map, planes, seed) {
      var cyc = cyclesFor(sc, b, mp, planes, seed, KB);
      var comps = []; for (var k = 1; k <= KB; k++) comps.push(components(cyc, k));
      var jd = jacDims(cyc, map, seed);
      var single = sc === 'boft' ? mp * (N / b) * binom2(b) : null;
      return { sc: sc, b: b, mp: mp, map: map, cyc: cyc, comps: comps, dims: jd.dims, gs: jd, single: single, depth: maxDepth(b) };
    }

    /* ---------- Tab C: re-HRA ---------- */
    function householderRight(P, v) { /* P <- P H_v */
      var n = P.length, vv = 0, i, j; for (i = 0; i < n; i++) vv += v[i] * v[i];
      for (i = 0; i < n; i++) { var d = 0; for (j = 0; j < n; j++) d += P[i][j] * v[j]; var f = 2 * d / vv; for (j = 0; j < n; j++) P[i][j] -= f * v[j]; }
    }
    function eigAngles(P) {
      var n = P.length, S = mat(n, n), i, j;
      for (i = 0; i < n; i++) for (j = 0; j < n; j++) S[i][j] = (P[i][j] + P[j][i]) / 2;
      var vals = LA.symEig(S).values.slice().sort(function (x, y) { return y - x; });
      var fixed = 0, rest = [];
      vals.forEach(function (v) { if (1 - v < 1e-9) fixed++; else rest.push(v); });
      var ang = []; for (i = 0; i < rest.length; i += 2) ang.push(Math.acos(Math.max(-1, Math.min(1, (rest[i] + (rest[i + 1] == null ? rest[i] : rest[i + 1])) / 2))));
      return { fixed: fixed, angles: ang };
    }
    function runC(r, k, seed) {
      var bs = base(), W0 = bs.W0, rnd = At.rng(31337 + 7 * seed);
      var Q = LA.randOrth(N, rnd), th = [], t, i;
      for (t = 0; t < N / 2; t++) th.push(0.55 + 2.35 * rnd());
      var u = new Float64Array(N); for (i = 0; i < N; i++) u[i] = rnd.normal();
      var refl = [];
      for (t = 0; t < k / 2; t++) {
        var q1 = Q[2 * t], q2 = Q[2 * t + 1], hh = th[t] / 2, bv = new Float64Array(N);
        for (i = 0; i < N; i++) bv[i] = Math.cos(hh) * q1[i] + Math.sin(hh) * q2[i];
        refl.push({ v: bv, plane: t }); refl.push({ v: Float64Array.from(q1), plane: t });   // H_b H_a = rotation by theta_t
      }
      var Rt = eye(N); refl.forEach(function (x) { householderRight(Rt, x.v); });
      var Kstar = Math.ceil(k / r), Jmax = Math.ceil(N / r), cycles = [];
      for (var j = 0; j < Kstar; j++) {
        var slots = refl.slice(j * r, j * r + r).map(function (x) { return { v: x.v, plane: x.plane, pad: false }; });
        while (slots.length < r) slots.push({ v: u, plane: -1, pad: true });
        cycles.push(slots);
      }
      var WR = mul(W0, Rt), P = eye(N), steps = [];
      var meas = function (P, j, C) {
        var W = mul(W0, P), D = LA.add(P, eye(N), -1), Kw = mul(W, tr(W));
        return {
          j: j, rank: rank(D), rankW: rank(LA.add(W, W0, -1)), rankC: C ? rank(LA.add(C, eye(N), -1)) : 0,
          dist: frob(LA.add(W, WR, -1)), kout: frob(LA.add(Kw, bs.K0, -1)) / frob(bs.K0),
          det: det(P), orth: frob(LA.add(mul(tr(P), P), eye(N), -1)), eig: eigAngles(P)
        };
      };
      steps.push(meas(P, 0, null));
      cycles.forEach(function (slots, jj) {
        var C = eye(N); slots.forEach(function (x) { householderRight(C, x.v); });
        P = mul(P, C);
        steps.push(meas(P, jj + 1, C));
      });
      var rg = At.rng(4711 + 7 * seed + r), G = eye(N), gen = [0];
      for (j = 0; j < Jmax; j++) { for (var s = 0; s < r; s++) { var v = new Float64Array(N); for (i = 0; i < N; i++) v[i] = rg.normal(); householderRight(G, v); } gen.push(rank(LA.add(G, eye(N), -1))); }
      return { r: r, k: k, Kstar: Kstar, Jmax: Jmax, cycles: cycles, steps: steps, gen: gen, target: { rank: rank(LA.add(Rt, eye(N), -1)), eig: eigAngles(Rt), det: det(Rt) }, injective: bs.rank };
    }
    return { svd: svd, rank: rank, base: base, runA: runA, runB: runB, runC: runC, driftRun: driftRun, contiguous: contiguous, butterflyLevel: butterflyLevel, maxDepth: maxDepth, components: components, cyclesFor: cyclesFor, jacDims: jacDims, eigAngles: eigAngles, M: M, N: N };
  }

  /* ================= scoped CSS ================= */
  function injectCSS() {
    if (document.getElementById('css-rebase')) return;
    var F = '[data-figure="rebase"] ';
    var css = [
      F + '.rb-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      /* sub/superscripts inside running text never drop below the 12px reading floor */
      F + 'sub,' + F + 'sup{font-size:max(.8em,12px);line-height:0}',
      F + 'code{font-size:max(.88em,12px)}',
      F + '.rb-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.rb-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.rb-title i{color:var(--tide)}',
      F + '.rb-kicker{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.rb-instr{margin:.45rem 0 .85rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:68ch}',
      F + '.rb-instr b{color:var(--ink);font-weight:600}',
      F + '.rb-tabs{display:grid;grid-template-columns:minmax(0,1fr);gap:.4rem;margin:0 0 .95rem}',
      F + '.rb-tab{display:grid;grid-template-columns:auto minmax(0,1fr);column-gap:.6rem;align-items:baseline;text-align:left;font:inherit;color:var(--ink-2);background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.5rem .7rem .55rem;cursor:pointer;min-width:0}',
      F + '.rb-tab:hover{border-color:var(--ink-3);color:var(--ink)}',
      F + '.rb-tab[aria-selected="true"]{border-color:var(--tide);box-shadow:inset 0 -3px 0 var(--tide);color:var(--ink)}',
      /* tab labels: Plex Sans 500; the verdict line under each is a readout in Plex Sans with tabular figures */
      F + '.rb-tab-n{font-family:var(--f-ui);font-weight:600;font-size:.82rem;letter-spacing:.04em;color:var(--tide);grid-row:span 2}',
      F + '.rb-tab-t{font-family:var(--f-ui);font-weight:500;font-size:.95rem;line-height:1.25;color:var(--ink)}',
      F + '.rb-tab-v{font-family:var(--f-ui);font-size:.8rem;line-height:1.4;color:var(--ink-2);font-variant-numeric:tabular-nums;margin-top:.15rem;overflow-wrap:anywhere}',
      F + '.rb-tab-v .m{color:var(--moss-ink)}',
      F + '.rb-tab-v .s{color:var(--seal-ink)}',
      F + '.rb-stmt{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem 1.4rem;padding:.5rem .75rem;margin:0 0 .95rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.9rem;line-height:1.5;color:var(--ink);overflow-x:auto}',
      F + '.rb-stmt > span{display:inline-flex;flex-wrap:wrap;align-items:center;gap:.15rem .5rem;min-width:0}',
      F + '.rb-stmt mjx-container{white-space:nowrap}',
      F + '.rb-stmt mjx-container svg{max-width:none}',
      /* UI labels: Plex Sans 500, uppercase, letter-spacing kept at .06em */
      F + '.rb-tag{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.rb-tag .nt{text-transform:none;letter-spacing:0}',
      F + '.rb-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:.8rem 1.6rem;margin-bottom:1rem}',
      F + '.rb-group{min-width:0;border-top:1px solid var(--rule);padding-top:.5rem;display:grid;gap:.55rem;align-content:start}',
      F + '.rb-row{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .6rem;min-width:0}',
      F + '.rb-k{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);min-width:4.6rem}',
      F + '.rb-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.rb-sl-top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0 .4rem;min-width:0}',
      F + '.rb-sl label{font-family:var(--f-body);font-size:var(--fs-sm);color:var(--ink);white-space:nowrap}',
      F + '.rb-sl label .nm{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.35rem}',
      F + '.rb-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.rb-sl input[type=range]{margin:0;height:1.25rem}',
      F + '.rb-hint{font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.seg button:disabled{opacity:.4;cursor:not-allowed}',
      F + '.rb-seg-neu button[aria-pressed="true"]{background:var(--seal)}',
      F + '.rb-mini{font-family:var(--f-ui);font-weight:500;font-size:.8rem;line-height:1.1;letter-spacing:.01em;padding:.34rem .6rem;border:1px solid var(--rule);background:var(--paper);color:var(--ink);border-radius:4px;cursor:pointer;white-space:nowrap}',
      F + '.rb-mini:hover:not(:disabled){border-color:var(--tide);color:var(--tide)}',
      F + '.rb-mini:disabled{opacity:.4;cursor:not-allowed}',
      F + '.rb-mini.pri{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
      F + '.rb-mini.pri:hover:not(:disabled){filter:brightness(1.08);color:var(--paper)}',
      F + '.rb-sel{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;padding:.25rem .3rem;background:var(--paper);color:var(--ink);border:1px solid var(--rule);border-radius:4px}',
      F + '.rb-res{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.5rem;align-items:start}',
      F + '.rb-legend{display:flex;flex-wrap:wrap;gap:.25rem 1rem;margin:0 0 .35rem;font-family:var(--f-ui);font-weight:500;font-size:.78rem;line-height:1.45;color:var(--ink-2)}',
      F + '.rb-li{display:inline-flex;align-items:center;gap:.4rem}',
      F + '.rb-li b{font-weight:600;color:var(--ink)}',
      F + '.rb-legend i,' + F + '.rb-cap i,' + F + '.rb-kv dd i,' + F + '.rb-tab-v i,' + F + '.rb-hint i,' + F + '.rb-banner .bs i,' + F + 'table.rb-tbl i{font-family:var(--f-body);font-style:italic;font-size:1.1em;line-height:1}',
      F + '.rb-li svg{flex:none;overflow:visible}',
      F + '.rb-chart > svg,' + F + '.rb-graph > svg,' + F + '.rb-circ > svg,' + F + '.rb-mchart > svg{display:block;width:100%;height:auto;overflow:visible}',
      F + '.rb-graph,.rb-circ{max-width:400px;margin:0 auto;width:100%}',
      /* captions under a diagram are prose: Source Serif */
      F + '.rb-cap{font-family:var(--f-body);font-size:.84rem;color:var(--ink-2);text-align:center;margin-top:.25rem;line-height:1.45}',
      F + '.rb-card{display:grid;gap:.6rem;align-content:start;min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.85rem .95rem .9rem}',
      F + '.rb-banner{display:grid;gap:.2rem;padding:.55rem .7rem;border-radius:4px;border:1px solid currentColor}',
      F + '.rb-banner .bt{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.22rem;line-height:1.2}',
      F + '.rb-banner .bs{font-family:var(--f-body);font-size:.86rem;line-height:1.45;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.rb-banner.m{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.rb-banner.s{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.rb-banner.o{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + '.rb-kv{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:.32rem .8rem;font-family:var(--f-body);font-size:.86rem;line-height:1.35;align-items:baseline}',
      F + '.rb-kv dt{color:var(--ink-2);min-width:0}',
      F + '.rb-kv dt .th{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.35rem;white-space:nowrap}',
      F + '.rb-kv dd{margin:0;font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);text-align:right;white-space:nowrap}',
      F + '.rb-kv dd.m{color:var(--moss-ink)}',
      F + '.rb-kv dd.s{color:var(--seal-ink)}',
      F + '.rb-kv dd.o{color:var(--ochre-ink)}',
      F + '.rb-sep{border:0;border-top:1px solid var(--rule);margin:.1rem 0}',
      F + '.rb-note{margin:0;font-family:var(--f-body);font-size:.88rem;line-height:1.5;color:var(--ink-2)}',
      F + '.rb-note b{color:var(--ink);font-weight:600}',
      F + '.rb-lead{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;margin-right:.45rem}',
      F + '.rb-lead.m{color:var(--moss-ink)} ' + F + '.rb-lead.s{color:var(--seal-ink)} ' + F + '.rb-lead.o{color:var(--ochre-ink)} ' + F + '.rb-lead.t{color:var(--tide)}',
      F + 'table.rb-tbl{width:100%;border-collapse:collapse;font-size:.8rem;line-height:1.35}',
      F + 'table.rb-tbl th,' + F + 'table.rb-tbl td{position:static;background:none;padding:.32rem .3rem;border-bottom:1px solid var(--rule);vertical-align:baseline}',
      F + 'table.rb-tbl tr:last-child td,' + F + 'table.rb-tbl tr:last-child th{border-bottom:0}',
      F + 'table.rb-tbl thead th{font-family:var(--f-ui);font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2);font-weight:500;text-align:right}',
      F + 'table.rb-tbl thead th:first-child{text-align:left}',
      F + 'table.rb-tbl tbody th{font-family:var(--f-body);font-size:.9rem;font-weight:400;letter-spacing:0;text-transform:none;color:var(--ink);text-align:left}',
      F + 'table.rb-tbl td .u{font-family:var(--f-ui);color:var(--ink-2);font-size:.78rem}',
      F + 'table.rb-tbl tbody th .sw{display:inline-block;width:.75rem;height:3px;border-radius:2px;vertical-align:middle;margin-right:.4rem}',
      F + 'table.rb-tbl td{font-family:var(--f-mono);font-size:.82rem;font-variant-numeric:tabular-nums;text-align:right;color:var(--ink);white-space:nowrap}',
      F + '.rb-chip{display:inline-flex;align-items:center;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;padding:.08rem .45rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap}',
      F + '.rb-chip.m{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.rb-chip.s{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.rb-chip.o{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + '.rb-strip{display:flex;flex-wrap:wrap;gap:.45rem .5rem;margin:.2rem 0 0}',
      F + '.rb-cyc{display:grid;gap:.2rem;padding:.3rem .35rem .35rem;border:1px solid var(--rule);border-radius:4px;background:var(--paper)}',
      F + '.rb-cyc.on{border-color:var(--tide);box-shadow:inset 0 0 0 1px var(--tide)}',
      F + '.rb-cyc.off{opacity:.55}',
      F + '.rb-cyc-l{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;color:var(--ink-2)}',
      F + '.rb-slots{display:flex;gap:2px}',
      F + '.rb-slot{width:1.2rem;height:1.5rem;display:grid;place-items:center;font-family:var(--f-body);font-style:italic;font-size:.82rem;border-radius:3px;border:1px solid var(--ochre);background:var(--ochre-soft);color:var(--ink)}',
      F + '.rb-slot.alt{border-color:var(--ink-3);background:var(--paper-3)}',
      F + '.rb-slot.pad{border:1px dashed var(--ink-3);background:transparent;color:var(--ink-2)}',
      /* the footnote is prose: Source Serif at reading size */
      F + '.rb-foot{margin-top:1rem;padding-top:.6rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.84rem;line-height:1.55;color:var(--ink-2)}',
      F + '.rb-foot a{color:var(--ink-2)}',
      F + '.rb-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
      F + '.rb-node{cursor:default}',
      F + '.rb-node.pick{cursor:pointer}',
      F + '.rb-node.pick:focus{outline:none}',
      F + '.rb-node.pick:focus-visible circle.hit{stroke:var(--ochre);stroke-width:2.5}',
      F + '[hidden]{display:none !important}',
      '@container (max-width: 520px){',
      F + '.rb-kv{font-size:.84rem;column-gap:.5rem}',
      F + '.rb-kv dd{white-space:normal;max-width:11rem}',
      F + '.rb-sl label .nm{display:none}',
      '}',
      '@container (min-width: 640px){',
      F + '.rb-tabs{grid-template-columns:repeat(3,minmax(0,1fr))}',
      '}',
      '@container (min-width: 720px){',
      F + '.rb-controls{grid-template-columns:repeat(2,minmax(0,1fr))}',
      F + '.rb-res{grid-template-columns:minmax(0,1.5fr) minmax(0,1fr)}',
      F + '.rb-res.b{grid-template-columns:minmax(0,1fr) minmax(0,1.08fr)}',
      F + '.rb-res.c{grid-template-columns:minmax(0,.9fr) minmax(0,1.2fr)}',
      '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-rebase';
    s.textContent = css;
    document.head.appendChild(s);
  }

  var ENGINE = makeEngine(window.Atlas);
  window.Atlas._rebase = ENGINE;

  /* ================= figure ================= */
  Atlas.register('rebase', function (el, A) {
    injectCSS();
    var E = ENGINE, h = A.h;
    var uid = 'rb' + Math.random().toString(36).slice(2, 7);
    function S(tag, attrs, parent) {
      var e = document.createElementNS(NSV, tag);
      if (attrs) for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }
    function T(parent, x, y, txt, attrs) { var a = attrs || {}; a.x = x; a.y = y; var t = S('text', a, parent); t.textContent = txt; return t; }
    function slider(o) {
      var id = uid + '-' + o.key;
      var input = h('input', { type: 'range', id: id, min: o.min, max: o.max, step: o.step || 1, value: o.value });
      var out = h('output', { for: id });
      var hint = o.hint ? h('div', { class: 'rb-hint' }) : null;
      var lab = h('label', { for: id, html: o.label + (o.nm ? '<span class="nm">' + o.nm + '</span>' : '') });
      var wrap = h('div', { class: 'rb-sl' }, [h('div', { class: 'rb-sl-top' }, [lab, out]), input, hint]);
      input.addEventListener('input', function () { o.onInput(+input.value); });
      return { el: wrap, input: input, out: out, hint: hint, set: function (v, mx) { if (mx != null) input.max = mx; input.value = v; } };
    }
    function mini(label, aria, fn, cls) { var b = h('button', { type: 'button', class: 'rb-mini' + (cls ? ' ' + cls : ''), 'aria-label': aria, html: label }); b.addEventListener('click', fn); return b; }
    function sw(color, dash, dot) {
      var s = S('svg', { width: 22, height: 10, viewBox: '0 0 22 10', 'aria-hidden': 'true' });
      S('line', { x1: 1, y1: 5, x2: 21, y2: 5, style: 'stroke:' + color + ';stroke-width:2;' + (dash ? 'stroke-dasharray:' + dash : '') }, s);
      if (dot) S('circle', { cx: 11, cy: 5, r: 3, style: 'fill:' + color }, s);
      return s;
    }
    function li(svgEl, html) { return h('span', { class: 'rb-li' }, [svgEl, h('span', { html: html })]); }
    function kv(dl, label, th, val, cls) {
      dl.appendChild(h('dt', { html: label + (th ? '<span class="th">' + th + '</span>' : '') }));
      dl.appendChild(h('dd', { class: cls || '', html: supify(val) }));
    }
    function observe(node, fn) {
      if ('ResizeObserver' in window) { var last = 0; new ResizeObserver(A.debounce(function () { var w = node.clientWidth; if (w && Math.abs(w - last) > 2) { last = w; fn(); } }, 60)).observe(node); }
      else window.addEventListener('resize', A.debounce(fn, 100));
    }

    /* ---------- state ---------- */
    var stA = { r: 4, ui: 6, seed: 1, K: null };
    var stB = { sc: 'boft', b: 2, mp: 3, K: 4, map: 'exact', seed: 1, planes: [[0, 1], [1, 5], [5, 6], [9, 10], [10, 14], [3, 12], [7, 8]], pick: null };
    var stC = { r: 4, k: 10, j: null, seed: 1 };
    var resA = null, resB = null, resC = null, bKey = '';
    var tab = String(el.getAttribute('data-tab') || 'a').toLowerCase();
    if (!/^[abc]$/.test(tab)) tab = 'a';

    /* ---------- DOM: head ---------- */
    var stage = h('div', { class: 'stage rb-stage', role: 'group', 'aria-label': 'Rebasing reach. Three panels on what merge-and-restart can reach: A, additive methods (ReLoRA climbs one rank-r step per cycle, LoRA-XS with frozen frames is idempotent); B, exactly orthogonal block factors (OFT, BOFT, Givens planes) reach exactly as far as their coordinate hypergraph is connected, while HF’s Cayley-Neumann OFT shrinks K_out at every merge; C, re-HRA reaches any rotation R in ceil(rank(R - I)/r) cycles.' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'rb-head' }, [
      h('h3', { class: 'rb-title', html: 'Rebasing <i>reach</i>' }),
      h('span', { class: 'rb-kicker', text: 'Merge-and-restart · Thm V.3 · Cor 1, 2, 4, 5, 7' })
    ]));
    stage.appendChild(h('p', { class: 'rb-instr', html: 'Pick a panel and add <b>merge cycles</b>. Each cycle adds one more copy of the shape, so ReLoRA’s rank-<i>r</i> cone grows until it fills every matrix, while a fixed subspace or subgroup gains nothing. Orthogonal blocks reach only as far as their coordinate hypergraph connects.' }));

    var tabsEl = h('div', { class: 'rb-tabs', role: 'tablist', 'aria-label': 'Rebasing panels' });
    stage.appendChild(tabsEl);
    var TABS = [
      { key: 'a', n: 'A', t: 'Additive: ReLoRA vs frozen frames' },
      { key: 'b', n: 'B', t: 'Orthogonal blocks: the hypergraph' },
      { key: 'c', n: 'C', t: 'Re-HRA: cycles to a rotation' }
    ];
    var panels = {}, tabBtns = {}, verdicts = {};
    TABS.forEach(function (d, i) {
      var b = h('button', { type: 'button', class: 'rb-tab', role: 'tab', id: uid + '-tab-' + d.key, 'aria-controls': uid + '-pan-' + d.key, 'aria-selected': 'false', tabindex: '-1' });
      b.appendChild(h('span', { class: 'rb-tab-n', text: d.n }));
      b.appendChild(h('span', { class: 'rb-tab-t', text: d.t }));
      verdicts[d.key] = h('span', { class: 'rb-tab-v' });
      b.appendChild(verdicts[d.key]);
      b.addEventListener('click', function () { setTab(d.key); });
      b.addEventListener('keydown', function (e) {
        var j = null;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = (i + 1) % 3;
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = (i + 2) % 3;
        else if (e.key === 'Home') j = 0; else if (e.key === 'End') j = 2;
        if (j != null) { e.preventDefault(); setTab(TABS[j].key); tabBtns[TABS[j].key].focus(); }
      });
      tabBtns[d.key] = b; tabsEl.appendChild(b);
      panels[d.key] = h('div', { class: 'rb-panel', role: 'tabpanel', id: uid + '-pan-' + d.key, 'aria-labelledby': uid + '-tab-' + d.key });
      stage.appendChild(panels[d.key]);
    });
    function setTab(k) {
      tab = k;
      TABS.forEach(function (d) {
        var on = d.key === k;
        tabBtns[d.key].setAttribute('aria-selected', String(on));
        tabBtns[d.key].setAttribute('tabindex', on ? '0' : '-1');
        if (on) panels[d.key].removeAttribute('hidden'); else panels[d.key].setAttribute('hidden', '');
      });
      if (k === 'a') drawA(); else if (k === 'b') drawB(); else drawC();
    }

    /* =============== panel A =============== */
    var pA = panels.a;
    pA.appendChild(h('div', { class: 'rb-stmt' }, [
      h('span', null, [h('span', { class: 'rb-tag', text: 'Cor 1 · ReLoRA' }), h('span', { html: '\\(R_K=W_0+\\mathcal M_{\\le\\min(Kr,\\,m,\\,n)}\\)' })]),
      h('span', null, [h('span', { class: 'rb-tag', text: 'Cor 2 · frozen frames' }), h('span', { html: '\\(R_K=R_1=W_0+U_r\\,\\mathbb R^{r\\times r}\\,V_r^{\\top}\\)' })]),
      h('span', null, [h('span', { class: 'rb-tag', text: 'Thm V.3(a)' }), h('span', { html: '\\(R_K=\\theta_0+C^{+K}\\)' })])
    ]));
    var ctlA = h('div', { class: 'rb-controls' }); pA.appendChild(ctlA);
    var gA1 = h('div', { class: 'rb-group' }, [h('span', { class: 'rb-tag', text: 'Each merge cycle' })]);
    var slR = slider({ key: 'ar', label: 'rank per cycle <i>r</i>', nm: 'ReLoRA & LoRA-XS', min: 1, max: 8, value: stA.r, onInput: function (v) { stA.r = v; stA.K = null; slR.out.textContent = v; computeAd(); } });
    var slK = slider({ key: 'ak', label: 'merge cycles <i>K</i>', min: 1, max: 8, value: 4, onInput: function (v) { stA.K = v; drawA(); } });
    gA1.appendChild(slR.el); gA1.appendChild(slK.el); ctlA.appendChild(gA1);
    var gA2 = h('div', { class: 'rb-group' }, [h('span', { class: 'rb-tag', text: 'LoRA-XS update per cycle' })]);
    var slU = slider({ key: 'au', label: 'update size <i>s</i>', nm: 'R<sub>k</sub> = s·G/r', min: 0, max: UPD.length - 1, value: stA.ui, hint: true, onInput: function (v) { stA.ui = v; slU.out.textContent = String(UPD[v]); computeAd(); } });
    gA2.appendChild(slU.el);
    gA2.appendChild(h('div', { class: 'rb-row' }, [mini('new draws ↻', 'Draw new random factors for every cycle', function () { stA.seed = stA.seed % 97 + 1; computeA(); }), h('span', { class: 'rb-hint', html: 'same <i>R<sub>k</sub></i> for both LoRA-XS rows' })]));
    ctlA.appendChild(gA2);
    var resAEl = h('div', { class: 'rb-res' }); pA.appendChild(resAEl);
    var leftA = h('div', { style: 'min-width:0' });
    var legA = h('div', { class: 'rb-legend' });
    var chartA = h('div', { class: 'rb-chart' });
    var whyA = h('p', { class: 'rb-note', style: 'margin-top:.55rem' });
    leftA.appendChild(legA); leftA.appendChild(chartA); leftA.appendChild(whyA); resAEl.appendChild(leftA);
    var cardA = h('div', { class: 'rb-card' }); resAEl.appendChild(cardA);

    var computeAd = A.debounce(function () { computeA(); }, 50);
    function computeA() {
      var r = stA.r, Kst = Math.ceil(N / r), Kmax = Math.min(20, Math.max(8, Kst + 3));
      resA = E.runA(r, UPD[stA.ui], stA.seed, Kmax);
      if (stA.K == null || stA.K > Kmax) stA.K = Math.min(Kst, Kmax);
      slK.set(stA.K, Kmax);
      slR.out.textContent = r;
      drawA();
    }
    function drawA() {
      if (!resA) return;
      var R = resA, K = stA.K, r = R.r;
      slK.out.textContent = String(K);
      slU.out.textContent = String(R.upd);
      slU.hint.innerHTML = 'E‖R<sub>k</sub>‖²<sub>F</sub> = s²; the gap σ<sub>' + r + '</sub> − σ<sub>' + (r + 1) + '</sub> of W<sub>0</sub> is ' + (+R.gap0.toFixed(3));
      verdicts.a.innerHTML = 'ReLoRA: full FT at <span class="m">K* = ⌈16/' + r + '⌉ = ' + R.Kstar + '</span> · frozen frame: <span class="s">rank ' + R.xsF[R.Kmax] + ' forever</span>';
      /* legend */
      legA.innerHTML = '';
      legA.appendChild(li(sw('var(--tide)', null, true), '<b>ReLoRA</b> rank ΔW<sub>K</sub>'));
      legA.appendChild(li(sw('var(--ink-3)', '4 3'), 'bound min(<i>Kr</i>, <i>m</i>, <i>n</i>)'));
      legA.appendChild(li(sw('var(--ochre)', null, true), '<b>LoRA-XS</b>, frame recomputed'));
      legA.appendChild(li(sw('var(--seal)', null, true), '<b>LoRA-XS</b>, frame frozen'));
      /* chart */
      /* the fractional box width, floored: a 1px-wider SVG would be scaled below 12px text */
      var W = Math.max(280, Math.floor(chartA.getBoundingClientRect().width) || chartA.clientWidth || 600), H = W < 460 ? 250 : 290;
      var ml = 34, mr = 14, mt = 30, mb = 38, iw = W - ml - mr, ih = H - mt - mb;
      chartA.innerHTML = '';
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Rank of the accumulated update after K merge cycles. ReLoRA climbs by r per cycle to full rank 16 at K* = ' + R.Kstar + '; LoRA-XS with a frozen frame stays at rank ' + r + '; with a recomputed frame it reaches rank ' + R.xsR[K] + ' after ' + K + ' cycles.' }, chartA);
      var x = function (k) { return ml + iw * k / R.Kmax; }, y = function (v) { return mt + ih * (1 - v / N); };
      var g = S('g', null, svg);
      [0, 4, 8, 12, 16].forEach(function (v) {
        S('line', { x1: ml, x2: ml + iw, y1: y(v), y2: y(v), style: 'stroke:var(--rule);stroke-width:1' + (v === N ? ';stroke-dasharray:2 3' : '') }, g);
        T(g, ml - 6, y(v) + 4, String(v), { 'text-anchor': 'end', class: 'num', style: 'font-size:11px;fill:var(--ink-2)' });
      });
      var step = R.Kmax > 12 ? 2 : 1;
      for (var k = 0; k <= R.Kmax; k += step) T(g, x(k), mt + ih + 16, String(k), { 'text-anchor': 'middle', class: 'num', style: 'font-size:11px;fill:var(--ink-2)' });
      /* axis titles and direct labels: Plex Sans 500, 12px */
      T(g, ml + iw, mt + ih + 32, 'merge cycles K', { 'text-anchor': 'end', style: 'font-size:12px;font-weight:500;fill:var(--ink-2)' });
      T(g, 0, 13, 'rank ΔW', { style: 'font-size:12px;font-weight:500;fill:var(--ink-2)' });
      T(g, ml + iw - 2, y(N) - 7, W < 460 ? 'rank 16: full fine-tuning' : 'rank 16 = every ΔW ∈ ℝ²⁴ˣ¹⁶: full fine-tuning', { 'text-anchor': 'end', style: 'font-size:12px;font-weight:500;fill:var(--moss-ink)' });
      /* K* marker */
      if (R.Kstar <= R.Kmax) {
        S('line', { x1: x(R.Kstar), x2: x(R.Kstar), y1: y(N), y2: mt + ih, style: 'stroke:var(--moss);stroke-width:1;stroke-dasharray:2 3;opacity:.8' }, g);
      }
      /* cursor */
      S('rect', { x: x(K) - Math.min(14, iw / R.Kmax / 2), y: mt, width: Math.min(28, iw / R.Kmax), height: ih, rx: 3, style: 'fill:var(--tide-soft)' }, g);
      function stepPath(arr) {
        var d = 'M' + x(0) + ',' + y(arr[0]);
        for (var i = 1; i < arr.length; i++) d += 'H' + x(i) + 'V' + y(arr[i]);
        return d;
      }
      S('path', { d: stepPath(R.bound), style: 'fill:none;stroke:var(--ink-3);stroke-width:1.25;stroke-dasharray:4 3' }, g);
      /* the two LoRA-XS series are dodged by 1.5 px so that coinciding values stay visible */
      var series = [
        { a: R.xsF, c: 'var(--seal)', off: 1.5 },
        { a: R.xsR, c: 'var(--ochre)', off: -1.5 },
        { a: R.relora, c: 'var(--tide)', off: 0 }
      ];
      series.forEach(function (s) {
        var gs = S('g', { transform: 'translate(0,' + s.off + ')' }, g);
        S('path', { d: stepPath(s.a), style: 'fill:none;stroke:' + s.c + ';stroke-width:2;stroke-linejoin:round' }, gs);
        for (var i = 1; i < s.a.length; i++) S('circle', { cx: x(i), cy: y(s.a[i]), r: i === K ? 4.2 : 2.6, style: 'fill:' + s.c + ';stroke:var(--paper);stroke-width:1' }, gs);
      });
      /* frame-moved rings for the recomputed series */
      for (var i = 1; i < R.moved.length; i++) if (R.moved[i]) S('circle', { cx: x(i), cy: y(R.xsR[i]) - 1.5, r: 6.5, style: 'fill:none;stroke:var(--ochre);stroke-width:1' }, g);
      if (R.Kstar <= R.Kmax) {
        /* the label is measured as set, so the fallback to the short form follows the font */
        var kx = x(R.Kstar), lab = 'K* = ⌈16/' + r + '⌉ = ' + R.Kstar, kst = 'font-size:12px;font-weight:500;fill:var(--moss-ink)';
        var kt = T(g, kx + 6, y(N) + 16, lab, { style: kst }), kw = kt.getComputedTextLength ? kt.getComputedTextLength() : lab.length * 7;
        if (kx + 6 + kw > ml + iw) { lab = 'K* = ' + R.Kstar; kt.textContent = lab; kw = kt.getComputedTextLength ? kt.getComputedTextLength() : lab.length * 7; }
        if (kx + 6 + kw > ml + iw) { kt.setAttribute('x', kx - 6); kt.setAttribute('text-anchor', 'end'); }
      }
      /* card */
      cardA.innerHTML = '';
      cardA.appendChild(h('span', { class: 'rb-tag', html: 'After <span class="nt">K = ' + K + '</span> merge cycles · <span class="nt">m×n = 24×16</span>' }));
      var tbl = h('table', { class: 'rb-tbl' });
      tbl.innerHTML = '<thead><tr><th>scheme</th><th>rank ΔW<sub>K</sub></th><th>dim R<sub>K</sub></th></tr></thead>';
      var tb = h('tbody');
      var full = R.relora[K] === N;
      function row(swc, name, rk, dim) { var tr_ = h('tr'); tr_.innerHTML = '<th><span class="sw" style="background:' + swc + '"></span>' + name + '</th><td>' + rk + '</td><td>' + dim + '</td>'; tb.appendChild(tr_); }
      row('var(--tide)', 'ReLoRA', R.relora[K] + ' <span class="u">/ ' + R.bound[K] + '</span>', R.dimRe[K]);
      row('var(--seal)', 'LoRA-XS, frozen frame', R.xsF[K], '<span class="u">r² =</span> ' + r * r);
      row('var(--ochre)', 'LoRA-XS, recomputed', R.xsR[K], '—');
      tbl.appendChild(tb); cardA.appendChild(tbl);
      var chips = h('div', { class: 'rb-row' });
      chips.appendChild(h('span', { class: 'rb-chip ' + (full ? 'm' : 'o'), text: full ? 'ReLoRA = full FT' : 'ReLoRA climbing' }));
      chips.appendChild(h('span', { class: 'rb-chip s', text: 'frozen: idempotent' }));
      chips.appendChild(h('span', { class: 'rb-chip o', text: 'frame moved ' + R.nMoved[K] + '/' + Math.max(0, K - 1) }));
      cardA.appendChild(chips);
      cardA.appendChild(h('hr', { class: 'rb-sep' }));
      var dl = h('dl', { class: 'rb-kv' });
      kv(dl, 'ReLoRA rank = bound, all <i>K</i> ≤ ' + R.Kmax, 'Cor 1', R.relora.every(function (v, i) { return v === R.bound[i]; }) ? 'yes' : 'no', R.relora.every(function (v, i) { return v === R.bound[i]; }) ? 'm' : 's');
      kv(dl, 'ReLoRA reaches full FT at', 'K* = ⌈min(m,n)/r⌉', 'K* = ' + R.Kstar, 'm');
      kv(dl, 'full fine-tuning', 'mn', String(M * N));
      cardA.appendChild(dl);
      whyA.innerHTML = '<span class="rb-lead t">Why</span>Each ReLoRA cycle adds one point of the rank-<i>r</i> cone, and rank is subadditive, so <i>K</i> cycles reach exactly the matrices of rank ≤ <i>Kr</i>. A frozen frame is a linear subspace: merging gives nothing a single run could not (Cor 2). A frame recomputed by SVD of the merged weight moves only when an update reorders singular values across the gap σ<sub>r</sub> > σ<sub>r+1</sub> (an update inside the frame keeps its span; by Weyl, never while ‖<i>R<sub>k</sub></i>‖<sub>2</sub> &lt; σ<sub>r</sub> − σ<sub>r+1</sub>). Circles mark the cycles where it moved.';
    }

    /* =============== panel B =============== */
    var pB = panels.b;
    pB.appendChild(h('div', { class: 'rb-stmt' }, [
      h('span', null, [h('span', { class: 'rb-tag', text: 'Thm V.3(c)' }), h('span', { html: '\\(\\mathrm{Lie}\\langle\\mathfrak{so}(I_j)\\rangle=\\bigoplus_{C}\\mathfrak{so}(C)\\)' })]),
      h('span', null, [h('span', { class: 'rb-tag', text: 'Thm V.3(b)' }), h('span', { html: '\\(R_\\infty=W_0\\cdot\\textstyle\\prod_C SO(C)\\)' })]),
      h('span', null, [h('span', { class: 'rb-tag', text: 'Cor 4' }), h('span', { html: '\\(SO(n)\\iff b\\,2^{m\'-1}=n\\)' })])
    ]));
    var ctlB = h('div', { class: 'rb-controls' }); pB.appendChild(ctlB);
    var gB1 = h('div', { class: 'rb-group' }, [h('span', { class: 'rb-tag', text: 'Factor family (input side, W = W₀R)' })]);
    var scSeg = A.seg([{ value: 'block', label: 'block OFT' }, { value: 'boft', label: 'BOFT' }, { value: 'rotate', label: 'rotating' }, { value: 'givens', label: 'Givens' }], stB.sc, function (v) { stB.sc = v; stB.pick = null; if (v === 'boft' && stB.mp > E.maxDepth(stB.b)) stB.mp = E.maxDepth(stB.b); computeB(); }, 'Factor family');
    gB1.appendChild(h('div', { class: 'rb-row' }, [scSeg.el]));
    var bSeg = A.seg([{ value: 2, label: 'b = 2' }, { value: 4, label: 'b = 4' }, { value: 8, label: 'b = 8' }], stB.b, function (v) { stB.b = v; if (stB.mp > E.maxDepth(v)) stB.mp = E.maxDepth(v); computeB(); }, 'Block size b');
    var bRow = h('div', { class: 'rb-row' }, [h('span', { class: 'rb-k', text: 'block size' }), bSeg.el]);
    gB1.appendChild(bRow);
    var mapSeg = A.seg([{ value: 'exact', label: 'exact Cayley' }, { value: 'neumann', label: 'Cayley–Neumann (HF OFT)' }], stB.map, function (v) { stB.map = v; mapSeg.el.classList.toggle('rb-seg-neu', v === 'neumann'); computeB(); }, 'Parametrisation of each orthogonal factor');
    gB1.appendChild(h('div', { class: 'rb-row' }, [h('span', { class: 'rb-k', text: 'factor map' }), mapSeg.el]));
    ctlB.appendChild(gB1);
    var gB2 = h('div', { class: 'rb-group' }, [h('span', { class: 'rb-tag', text: 'Depth and cycles' })]);
    var slMp = slider({ key: 'bmp', label: 'butterfly factors <i>m′</i>', nm: 'boft_n_butterfly_factor', min: 1, max: 4, value: stB.mp, hint: true, onInput: function (v) { stB.mp = v; slMp.out.textContent = v; computeBd(); } });
    var slKB = slider({ key: 'bk', label: 'merge cycles <i>K</i>', min: 1, max: KB, value: stB.K, onInput: function (v) { stB.K = v; drawB(); } });
    gB2.appendChild(slMp.el); gB2.appendChild(slKB.el);
    var selI = h('select', { class: 'rb-sel', 'aria-label': 'First coordinate of the Givens plane' }), selJ = h('select', { class: 'rb-sel', 'aria-label': 'Second coordinate of the Givens plane' });
    for (var ii = 0; ii < N; ii++) { selI.appendChild(h('option', { value: ii, text: String(ii + 1) })); selJ.appendChild(h('option', { value: ii, text: String(ii + 1) })); }
    selJ.value = '2';
    var givRow = h('div', { class: 'rb-row' }, [
      h('span', { class: 'rb-k', text: 'plane' }), selI, selJ,
      mini('add / remove', 'Add or remove the Givens plane between the two selected coordinates', function () { togglePlane(+selI.value, +selJ.value); }),
      mini('random', 'Add a random Givens plane', function () { var rr = A.rng(stB.planes.length * 31 + 7 + stB.seed * 101); var a = Math.floor(rr() * N), c = (a + 1 + Math.floor(rr() * (N - 1))) % N; togglePlane(a, c, true); }),
      mini('clear', 'Remove every Givens plane', function () { stB.planes = []; stB.pick = null; computeB(); })
    ]);
    gB2.appendChild(givRow);
    gB2.appendChild(h('div', { class: 'rb-row' }, [mini('new draws ↻', 'Draw new random factors and partitions', function () { stB.seed = stB.seed % 97 + 1; computeB(); })]));
    ctlB.appendChild(gB2);
    function togglePlane(a, c, addOnly) {
      if (a === c) return;
      var lo = Math.min(a, c), hi = Math.max(a, c), idx = -1;
      stB.planes.forEach(function (p, i) { if (p[0] === lo && p[1] === hi) idx = i; });
      if (idx >= 0) { if (addOnly) return; stB.planes.splice(idx, 1); } else stB.planes.push([lo, hi]);
      computeB();
    }
    var resBEl = h('div', { class: 'rb-res b' }); pB.appendChild(resBEl);
    var leftB = h('div', { style: 'min-width:0' });
    var legB = h('div', { class: 'rb-legend' });
    var graphB = h('div', { class: 'rb-graph' });
    var capB = h('div', { class: 'rb-cap' });
    leftB.appendChild(legB); leftB.appendChild(graphB); leftB.appendChild(capB); resBEl.appendChild(leftB);
    var noteB = h('p', { class: 'rb-note', style: 'margin-top:.8rem' });
    leftB.appendChild(noteB);
    var cardB = h('div', { class: 'rb-card' }); resBEl.appendChild(cardB);
    var bannerB = h('div', { class: 'rb-banner', 'aria-live': 'polite' });
    var kvB = h('dl', { class: 'rb-kv' });
    var mchartB = h('div', { class: 'rb-mchart' });
    cardB.appendChild(bannerB); cardB.appendChild(kvB); cardB.appendChild(h('hr', { class: 'rb-sep' }));
    cardB.appendChild(h('span', { class: 'rb-tag', html: 'dim <span class="nt">R<sub>K</sub></span>, measured (Jacobian rank of the product map)' }));
    cardB.appendChild(mchartB);

    var computeBd = A.debounce(function () { computeB(); }, 50);
    function computeB() {
      var key = [stB.sc, stB.b, stB.mp, stB.map, stB.seed, stB.sc === 'givens' ? JSON.stringify(stB.planes) : ''].join('|');
      if (key !== bKey) { resB = E.runB(stB.sc, stB.b, stB.mp, stB.map, stB.planes, stB.seed); bKey = key; }
      drawB();
    }
    var PAL = ['--c-additive', '--c-multiplicative', '--c-architectural', '--c-augmentation', '--c-optimizer', '--c-composition', '--c-hybrid', '--ink-2'];
    function drawB() {
      if (!resB) return;
      var R = resB, K = stB.K, comp = R.comps[K - 1], b = R.b, exact = R.map === 'exact';
      var giv = R.sc === 'givens', boft = R.sc === 'boft';
      bRow.hidden = giv; slMp.el.hidden = !boft; givRow.hidden = !giv;
      slMp.set(stB.mp, R.depth); slMp.out.textContent = stB.mp;
      slMp.hint.innerHTML = stB.mp === 1 ? 'HF default = block OFT' : stB.mp === R.depth ? 'full depth 1 + log₂(16/' + b + ') = ' + R.depth : 'full depth is ' + R.depth;
      slKB.out.textContent = K;
      var dr = E_drift(R, K, comp);
      var connected = comp.count === 1;
      /* tab verdict */
      verdicts.b.innerHTML = !exact ? '<span class="s">Neumann: not orthogonal</span> · tr K<sub>out</sub> shrinks ' + pct(dr.shrink) : connected ? '<span class="m">connected · SO(16) generated</span>' : '<span class="s">' + comp.count + ' components · stuck in ∏ SO(C)</span>';
      /* legend */
      legB.innerHTML = '';
      var dotS = function (fill, stroke) { var s = S('svg', { width: 12, height: 12, viewBox: '0 0 12 12', 'aria-hidden': 'true' }); S('circle', { cx: 6, cy: 6, r: 4.5, style: 'fill:' + fill + ';stroke:' + stroke + ';stroke-width:1.2' }, s); return s; };
      legB.appendChild(li(dotS('var(--c-multiplicative)', 'var(--c-multiplicative)'), 'colour = component <i>C</i>'));
      legB.appendChild(li(dotS('var(--moss)', 'var(--moss)'), 'one component: SO(16)'));
      legB.appendChild(li(dotS('var(--paper)', 'var(--ink-3)'), 'untouched coordinate'));
      var nwS = S('svg', { width: 22, height: 10, viewBox: '0 0 22 10', 'aria-hidden': 'true' }); S('line', { x1: 1, y1: 5, x2: 21, y2: 5, style: 'stroke:var(--ink-2);stroke-width:3.2;stroke-linecap:round' }, nwS);
      legB.appendChild(li(nwS, 'bold = newest factor'));
      /* hypergraph */
      drawGraph(R, K, comp);
      capB.innerHTML = 'nodes = the ' + N + ' input coordinates of <i>W</i>₀ ∈ ℝ²⁴ˣ¹⁶ · edges = coordinate blocks <i>I<sub>j</sub></i> of the factors in cycles 1..' + K + (giv ? ' · click two nodes to add or remove a plane' : '');
      /* banner */
      var sizes = comp.sizes.slice().sort(function (x, y) { return y - x; });
      var nonSingle = sizes.filter(function (s) { return s > 1; });
      var sizeStr = (function () { var m = {}; sizes.forEach(function (s) { m[s] = (m[s] || 0) + 1; }); return Object.keys(m).sort(function (x, y) { return y - x; }).map(function (s) { return m[s] > 1 ? m[s] + '×' + s : s; }).join(' + '); })();
      var prodStr = connected ? 'SO(16)' : (function () { var m = {}; nonSingle.forEach(function (s) { m[s] = (m[s] || 0) + 1; }); var ks = Object.keys(m).sort(function (x, y) { return y - x; }); return ks.length ? ks.map(function (s) { return 'SO(' + s + ')' + (m[s] > 1 ? sup(m[s]) : ''); }).join(' × ') : '{I}'; })();
      bannerB.className = 'rb-banner ' + (!exact ? 's' : connected ? 'm' : 's');
      if (!exact) bannerB.innerHTML = '<span class="bt">Not orthogonal: each merge shrinks</span><span class="bs">R = I + 2Q + 2Q² + 2Q³ + Q⁴ = C(Q)(I − Q⁴), so R<sup>⊤</sup>R = (I − Q⁴)² · after ' + K + ' merge' + (K > 1 ? 's' : '') + ' tr K<sub>out</sub> is down ' + pct(dr.shrink) + (dr.loewner ? ', and K<sub>out</sub> fell in the Loewner order at every merge' : '') + ' · Cor 4</span>';
      else if (connected) bannerB.innerHTML = '<span class="bt">SO(16) generated</span><span class="bs">R<sub>∞</sub> = W₀·SO(16)' + (boft ? ' (rotation part; BOFT adds a row scale diag(s))' : '') + (R.dims[K - 1] < DIM_SO ? ' · filled so far: dim R<sub>K</sub> = ' + R.dims[K - 1] + ' of 120' : ' · filled at K = ' + (R.dims.indexOf(DIM_SO) + 1)) + ' · K<sub>out</sub> = WW<sup>⊤</sup> and the spectrum are still kept (Cor 7)</span>';
      else bannerB.innerHTML = '<span class="bt">Stuck in ' + prodStr + '</span><span class="bs">R<sub>∞</sub> = W₀·∏<sub>C</sub> SO(C) · ' + comp.count + ' partial Grams W<sub>:,C</sub>W<sub>:,C</sub><sup>⊤</sup> can never change' + (boft ? ' (rotation part; BOFT’s diag(s) rescales them)' : '') + '</span>';
      /* readouts */
      kvB.innerHTML = '';
      kv(kvB, 'components <i>C</i>', 'union-find', comp.count + ' <span style="color:var(--ink-2)">(' + sizeStr + ')</span>');
      kv(kvB, 'dim ⊕<sub>C</sub> so(C) = Σ C(|C|,2)', 'Thm V.3(c)', comp.lie + ' <span style="color:var(--ink-2)">/ ' + DIM_SO + '</span>', connected ? 'm' : '');
      var dK = R.dims[K - 1];
      kv(kvB, 'dim R<sub>K</sub>, measured', 'Jacobian', String(dK), exact ? (dK === comp.lie ? 'm' : '') : (dK > comp.lie ? 's' : ''));
      if (boft) kv(kvB, 'one BOFT: m′(n/b)C(b,2) bound', 'Cor 4', R.single + ' <span style="color:var(--ink-2)">· measured ' + R.dims[0] + '</span>');
      kv(kvB, 'cross-component |R<sub>ab</sub>|, max', 'block-diagonal', dr.cross === 0 ? '0 exactly' : sci(dr.cross), dr.cross === 0 ? 'm' : 's');
      kv(kvB, exact ? 'partial Grams preserved' : 'partial Grams preserved', exact ? 'Cor II.6' : 'not orthogonal', exact ? comp.count + ' <span style="color:var(--ink-2)">(drift ' + tiny(dr.pmax) + ')</span>' : 'none · drift ' + sci(dr.pmax, 1), exact ? 'm' : 's');
      if (dr.J1 && exact && !dr.J1.isComp) kv(kvB, 'block J₁ = {1..' + b + '}: its partial Gram', 'absorbed', 'changes ' + sci(dr.J1.drift, 1), 'o');
      kv(kvB, 'K<sub>out</sub> = WW<sup>⊤</sup> after ' + K + ' merge' + (K > 1 ? 's' : ''), exact ? 'Cor 7, input side' : (R.sc === 'block' ? 'HF OFT default' : 'what-if'), exact ? 'drift ' + tiny(dr.kout) : 'trace −' + pct(dr.shrink), exact ? 'm' : 's');
      if (!exact) kv(kvB, 'K<sub>out</sub> falls in Loewner order at every merge', '‖Q‖₂ ≤ ' + (+dr.rho.toFixed(2)) + (dr.rho < Math.pow(2, 0.25) ? ' &lt; 2<sup>1/4</sup>' : ''), dr.loewner ? 'yes' : 'no', dr.loewner ? 's' : 'o');
      drawMiniB(R, K);
      /* note */
      var note = '';
      if (!exact) note = '<span class="rb-lead s">' + (R.sc === 'block' ? 'HF OFT default' : 'What-if') + '</span>Each Cayley–Neumann factor is C(Q)(I − Q⁴), with R<sup>⊤</sup>R = (I − Q⁴)² ≠ I. With small generators (spectral norm of Q at most 2<sup>1/4</sup>, the regime where the series approximates the Cayley map) every factor is a contraction, so each merge can only shrink K<sub>out</sub> in the Loewner order and lower every singular value. ' + (R.sc === 'block' ? 'On a fixed partition the measured dimension climbs from ' + R.dims[0] + ' to ' + R.dims[KB - 1] + (b === 2 ? ', the dimension of ∏ CO⁺(2) (rotations with a scale)' : ', the dimension of ∏ GL(' + b + ')') + ': the reach is full-dimensional, and it stays among contractions.' : 'The measured dimension climbs past the orthogonal ceiling Σ C(|C|,2), so these factors are not confined to SO(16). ' + (R.sc === 'boft' ? 'HF’s BOFT uses the exact Cayley map, so here the switch is a what-if.' : 'For this family the switch is a what-if.')) + ' OFT is exactly orthogonal only with <code>use_cayley_neumann=False</code>.';
      else if (R.sc === 'block') note = '<span class="rb-lead s">Cor 2</span>A fixed partition never connects. One exact-Cayley cycle is dense in ∏ SO(' + b + ') but misses the block rotations with an eigenvalue −1; a second cycle adds exactly those, so R<sub>2</sub> = R<sub>∞</sub> = W₀·∏ SO(' + b + '), the closure of R<sub>1</sub>, and the measured dimension never moves. <span class="rb-lead t">Pred</span>To reach beyond ∏ SO(' + b + '), blocks must overlap across cycles so that components grow: re-draw the partition between cycles, or add butterfly levels. This is a statement about reach; it says nothing about trained quality.';
      else if (boft) note = '<span class="rb-lead ' + (connected ? 'm' : 's') + '">Cor 4</span>Butterfly level ℓ pairs half-block c with c ⊕ 2<sup>ℓ</sup>, so components have size b·2<sup>m′−1</sup> = ' + (b * Math.pow(2, stB.mp - 1)) + '. SO(16) needs the full depth m′ = 1 + log₂(16/' + b + ') = ' + R.depth + '; HF’s default <code>boft_n_butterfly_factor=1</code> is block OFT times the row scale diag(<i>s</i>). A single full-depth BOFT is a dense <i>matrix</i>, not a dense <i>set</i>: its dimension is below ' + DIM_SO + ', and merge cycles fill the group.';
      else if (R.sc === 'rotate') note = '<span class="rb-lead t">Pred</span>Re-basing with a fresh random partition each cycle (cycle 1 is the fixed one): blocks from different cycles overlap, components merge, and the hypergraph connects. Connected means R<sub>∞</sub> = W₀·SO(16) (Thm V.3(b), (c)); each cycle adds at most ' + (N / b) * binom2(b) + ' dimensions, so filling SO(16) takes at least ' + Math.ceil(DIM_SO / ((N / b) * binom2(b))) + ' cycles.';
      else note = '<span class="rb-lead t">Givens</span>Each plane (i, j) is a factor so({i, j}). Planes that share a coordinate generate so of the union, because [L<sub>ac</sub>, L<sub>cb</sub>] = L<sub>ab</sub>. ' + (stB.planes.length ? stB.planes.length + ' plane' + (stB.planes.length === 1 ? ' gives' : 's give') + ' at most ' + stB.planes.length + ' dimension' + (stB.planes.length === 1 ? '' : 's') + ' per cycle. ' : 'No planes yet: click two nodes, or press random. ') + 'A spanning tree of 15 planes connects all 16 coordinates.';
      noteB.innerHTML = note;
      live('Panel B: ' + comp.count + ' components, Lie algebra dimension ' + comp.lie + ' of 120, measured dim R_K ' + dK + '.');
    }
    var driftCache = {};
    function E_drift(R, K, comp) {
      var key = bKey + '|' + K;
      if (!driftCache[key]) {
        if (Object.keys(driftCache).length > 64) driftCache = {};
        driftCache[key] = E.driftRun(R.cyc, K, R.map, stB.seed, comp, R.b, R.sc);
      }
      return driftCache[key];
    }
    function drawGraph(R, K, comp) {
      graphB.innerHTML = '';
      var Wd = 380, cx = 190, cy = 190, rad = 138;
      var svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Wd, role: 'img', 'aria-label': 'Factor hypergraph on 16 coordinates after ' + K + ' cycles: ' + comp.count + ' connected components.' }, graphB);
      /* labels are sized in rendered pixels: the viewBox is 380 units wide and the graph is drawn at up to 400px */
      var gsc = (graphB.getBoundingClientRect().width || Wd) / Wd, U = function (px) { return +Math.min(px * 1.5, px / gsc).toFixed(2); };
      var pos = []; for (var i = 0; i < N; i++) { var a = -Math.PI / 2 + 2 * Math.PI * i / N; pos.push([cx + rad * Math.cos(a), cy + rad * Math.sin(a), a]); }
      S('circle', { cx: cx, cy: cy, r: rad, style: 'fill:none;stroke:var(--rule);stroke-width:1;stroke-dasharray:1 4' }, svg);
      var connected = comp.count === 1;
      function compColor(c) { if (connected) return 'var(--moss)'; var nonS = []; comp.sizes.forEach(function (s, ci) { if (s > 1) nonS.push(ci); }); var k = nonS.indexOf(c); return k < 0 ? 'var(--ink-3)' : 'var(' + PAL[k % PAL.length] + ')'; }
      /* collect hyperedges with first appearance */
      var seen = {}, edges = [];
      var newest = null;
      for (var k = 0; k < K; k++) R.cyc[k].forEach(function (f, fi) {
        f.blocks.forEach(function (I) {
          var key = I.join(',');
          if (seen[key]) return;
          seen[key] = 1; edges.push({ I: I, k: k, fi: fi, level: f.level });
        });
      });
      if (R.sc === 'boft') newest = function (e) { return e.level === stB.mp - 1 && stB.mp > 1; };
      else if (R.sc === 'rotate') newest = function (e) { return K > 1 && e.k === K - 1; };
      else if (R.sc === 'givens') newest = function (e) { return stB.planes.length > 0 && e.I[0] === stB.planes[stB.planes.length - 1][0] && e.I[1] === stB.planes[stB.planes.length - 1][1]; };
      else newest = function () { return false; };
      var gE = S('g', null, svg), gN = S('g', null, svg);
      edges.sort(function (x, y) { return (newest(x) ? 1 : 0) - (newest(y) ? 1 : 0); });
      edges.forEach(function (e) {
        var col = compColor(comp.lab[e.I[0]]), nw = newest(e);
        var stroke = col;
        if (e.I.length === 2) {
          var p = pos[e.I[0]], q = pos[e.I[1]], mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
          var qx = cx + (mx - cx) * 0.7, qy = cy + (my - cy) * 0.7;
          S('path', { d: 'M' + p[0].toFixed(1) + ',' + p[1].toFixed(1) + 'Q' + qx.toFixed(1) + ',' + qy.toFixed(1) + ' ' + q[0].toFixed(1) + ',' + q[1].toFixed(1), style: 'fill:none;stroke:' + stroke + ';stroke-width:' + (nw ? 3.2 : 1.5) + ';opacity:' + (nw ? 1 : 0.62) + ';stroke-linecap:round' }, gE);
        } else {
          var pts = e.I.map(function (i) { return pos[i]; }).slice().sort(function (u, v) { return u[2] - v[2]; });
          var d = pts.map(function (p, j) { return (j ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('') + 'Z';
          S('path', { d: d, style: 'fill:' + stroke + ';fill-opacity:' + (nw ? 0.06 : 0.035) + ';stroke:' + stroke + ';stroke-width:' + (nw ? 2.8 : 1.2) + ';stroke-opacity:' + (nw ? 1 : 0.6) + ';stroke-linejoin:round' }, gE);
        }
      });
      for (i = 0; i < N; i++) {
        var c = comp.lab[i], single = comp.sizes[c] === 1, col = compColor(c), p = pos[i];
        var g = S('g', { class: 'rb-node' + (R.sc === 'givens' ? ' pick' : '') }, gN);
        var picked = stB.pick === i;
        S('circle', { class: 'hit', cx: p[0], cy: p[1], r: 10.5, style: 'fill:' + (single ? 'var(--paper)' : col) + ';stroke:' + (picked ? 'var(--ochre)' : single ? 'var(--ink-3)' : 'var(--paper)') + ';stroke-width:' + (picked ? 3 : 1.5) }, g);
        var lr = rad + 25, nf = U(12.5);
        T(g, cx + lr * Math.cos(p[2]), cy + lr * Math.sin(p[2]) + nf * 0.36, String(i + 1), { 'text-anchor': 'middle', class: 'num', style: 'font-size:' + nf + 'px;fill:var(--ink-2)' });
        if (R.sc === 'givens') {
          g.setAttribute('tabindex', '0'); g.setAttribute('role', 'button');
          g.setAttribute('aria-label', 'Coordinate ' + (i + 1) + (picked ? ', selected' : '') + '. Press Enter to pick it as one end of a Givens plane.');
          (function (i) {
            var act = function () { if (stB.pick == null) { stB.pick = i; drawB(); } else if (stB.pick === i) { stB.pick = null; drawB(); } else { var a = stB.pick; stB.pick = null; togglePlane(a, i); } };
            g.addEventListener('click', act);
            g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
          })(i);
        }
      }
      /* centre label */
      var lab = !connected ? comp.count + ' components' : 'connected';
      var cl = S('g', null, svg), cf = U(13);
      var cbox = S('rect', { x: cx - 52, y: cy - cf * 1.1, width: 104, height: cf * 2.1, rx: 4, style: 'fill:var(--paper-2);stroke:var(--rule);stroke-width:1;opacity:.92' }, cl);
      var ct = T(cl, cx, cy + cf * 0.35, lab, { 'text-anchor': 'middle', style: 'font-size:' + cf + 'px;font-weight:500;fill:' + (connected ? 'var(--moss-ink)' : 'var(--ink)') });
      var cw0 = ct.getComputedTextLength ? ct.getComputedTextLength() : lab.length * cf * 0.55;
      cbox.setAttribute('x', cx - cw0 / 2 - cf * 0.7); cbox.setAttribute('width', cw0 + cf * 1.4);
    }
    function drawMiniB(R, K) {
      mchartB.innerHTML = '';
      var W = Math.max(240, Math.floor(mchartB.getBoundingClientRect().width) || mchartB.clientWidth || 300), H = 160, ml = 32, mr = 10, mt = 18, mb = 26, iw = W - ml - mr, ih = H - mt - mb;
      var exact = R.map === 'exact';
      var dmax = Math.max.apply(null, R.dims), ymax = dmax <= DIM_SO ? DIM_SO : Math.ceil(dmax / 64) * 64;
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Measured dimension of R_K for K = 1 to 16, against the Lie algebra dimension of the union hypergraph and dim SO(16) = 120.' }, mchartB);
      var x = function (k) { return ml + iw * (k - 1) / (KB - 1); }, y = function (v) { return mt + ih * (1 - v / ymax); };
      var ticks = ymax === DIM_SO ? [0, 40, 80, 120] : [0, 64, 128, 192, 256].filter(function (v) { return v <= ymax; });
      ticks.forEach(function (v) { S('line', { x1: ml, x2: ml + iw, y1: y(v), y2: y(v), style: 'stroke:var(--rule);stroke-width:1' }, svg); T(svg, ml - 5, y(v) + 4, String(v), { 'text-anchor': 'end', class: 'num', style: 'font-size:11px;fill:var(--ink-2)' }); });
      [1, 4, 8, 12, 16].forEach(function (k) { T(svg, x(k), mt + ih + 16, String(k), { 'text-anchor': 'middle', class: 'num', style: 'font-size:11px;fill:var(--ink-2)' }); });
      S('line', { x1: ml, x2: ml + iw, y1: y(DIM_SO), y2: y(DIM_SO), style: 'stroke:var(--moss);stroke-width:1;stroke-dasharray:3 3' }, svg);
      T(svg, ml + iw, ymax > DIM_SO ? y(DIM_SO) + 14 : y(DIM_SO) - 5, 'dim SO(16) = 120', { 'text-anchor': 'end', style: 'font-size:12px;font-weight:500;fill:var(--moss-ink)' });
      S('rect', { x: x(K) - 6, y: mt, width: 12, height: ih, rx: 3, style: 'fill:var(--tide-soft)' }, svg);
      var lie = R.comps.map(function (c) { return c.lie; });
      var dl = 'M' + x(1) + ',' + y(lie[0]); for (var k = 1; k < KB; k++) dl += 'H' + x(k + 1) + 'V' + y(lie[k]);
      S('path', { d: dl, style: 'fill:none;stroke:var(--ink-3);stroke-width:1.25;stroke-dasharray:4 3' }, svg);
      var col = exact ? 'var(--tide)' : 'var(--seal)';
      var dd = 'M' + x(1) + ',' + y(R.dims[0]); for (k = 1; k < KB; k++) dd += 'H' + x(k + 1) + 'V' + y(R.dims[k]);
      S('path', { d: dd, style: 'fill:none;stroke:' + col + ';stroke-width:2;stroke-linejoin:round' }, svg);
      for (k = 0; k < KB; k++) S('circle', { cx: x(k + 1), cy: y(R.dims[k]), r: k === K - 1 ? 4 : 2.3, style: 'fill:' + col + ';stroke:var(--paper);stroke-width:1' }, svg);
      var leg = h('div', { class: 'rb-legend', style: 'margin:.25rem 0 0' });
      leg.appendChild(li(sw(col, null, true), 'dim R<sub>K</sub> measured'));
      leg.appendChild(li(sw('var(--ink-3)', '4 3'), 'Σ C(|C|,2) of cycles 1..K'));
      mchartB.appendChild(leg);
    }

    /* =============== panel C =============== */
    var pC = panels.c;
    pC.appendChild(h('div', { class: 'rb-stmt' }, [
      h('span', null, [h('span', { class: 'rb-tag', text: 'Cor 5 · re-HRA' }), h('span', { html: '\\(R_K=(W_0+\\mathcal M_{\\le Kr})\\cap W_0\\cdot SO(n)\\)' })]),
      h('span', null, [h('span', { class: 'rb-tag', text: 'cycles to reach R' }), h('span', { html: '\\(\\lceil \\operatorname{rank}(R-I)/r\\rceil\\le\\lceil n/r\\rceil\\)' })]),
      h('span', { class: 'rb-tag', html: '<span class="nt">W₀ injective, r even · rebasing commutes with the meet</span>' })
    ]));
    var ctlC = h('div', { class: 'rb-controls' }); pC.appendChild(ctlC);
    var gC1 = h('div', { class: 'rb-group' }, [h('span', { class: 'rb-tag', text: 'Target and method' })]);
    var rSegC = A.seg([2, 4, 6, 8].map(function (v) { return { value: v, label: 'r = ' + v }; }), stC.r, function (v) { stopPlay(); stC.r = v; stC.j = null; computeC(); }, 'HRA reflections per cycle r');
    gC1.appendChild(h('div', { class: 'rb-row' }, [h('span', { class: 'rb-k', text: 'HRA rank' }), rSegC.el]));
    var slk = slider({ key: 'ck', label: 'target <i>k</i> = rank(<i>R</i> − <i>I</i>)', min: 2, max: N, step: 2, value: stC.k, onInput: function (v) { stopPlay(); stC.k = v; stC.j = null; computeC(); } });
    gC1.appendChild(slk.el);
    ctlC.appendChild(gC1);
    var gC2 = h('div', { class: 'rb-group' }, [h('span', { class: 'rb-tag', text: 'Merge and restart' })]);
    var slj = slider({ key: 'cj', label: 'cycles run <i>j</i>', min: 0, max: 3, value: 3, onInput: function (v) { stopPlay(); stC.j = v; drawC(); } });
    gC2.appendChild(slj.el);
    var playBtn = mini('▶ play cycles', 'Play the merge cycles from j = 0', function () { play(); }, 'pri');
    gC2.appendChild(h('div', { class: 'rb-row' }, [playBtn, mini('new target ↻', 'Draw a new random target rotation', function () { stopPlay(); stC.seed = stC.seed % 97 + 1; stC.j = null; computeC(); })]));
    ctlC.appendChild(gC2);
    var resCEl = h('div', { class: 'rb-res c' }); pC.appendChild(resCEl);
    var leftC = h('div', { style: 'min-width:0' });
    var legC = h('div', { class: 'rb-legend' });
    var circC = h('div', { class: 'rb-circ' });
    var capC = h('div', { class: 'rb-cap' });
    leftC.appendChild(legC); leftC.appendChild(circC); leftC.appendChild(capC); resCEl.appendChild(leftC);
    var rightC = h('div', { style: 'min-width:0;display:grid;gap:.75rem' });
    var legC2 = h('div', { class: 'rb-legend' });
    var barsC = h('div', { class: 'rb-chart' });
    var stripC = h('div');
    var cardC = h('div', { class: 'rb-card' });
    rightC.appendChild(h('div', null, [legC2, barsC])); rightC.appendChild(stripC); rightC.appendChild(cardC);
    resCEl.appendChild(rightC);

    function computeC() {
      resC = E.runC(stC.r, stC.k, stC.seed);
      if (stC.j == null || stC.j > resC.Kstar) stC.j = resC.Kstar;
      slj.set(stC.j, resC.Kstar);
      drawC();
    }
    var playT = null;
    function stopPlay() { if (playT) { clearTimeout(playT); playT = null; playBtn.innerHTML = '▶ play cycles'; } }
    function play() {
      if (playT) { stopPlay(); return; }
      if (A.reducedMotion()) { stC.j = resC.Kstar; slj.set(stC.j); drawC(); return; }
      stC.j = 0; slj.set(0); drawC(); playBtn.innerHTML = '■ stop';
      var tick = function () {
        stC.j = Math.min(stC.j + 1, resC.Kstar); slj.set(stC.j); drawC();
        if (stC.j >= resC.Kstar) { playT = null; playBtn.innerHTML = '▶ play cycles'; return; }
        playT = setTimeout(tick, 850);
      };
      playT = setTimeout(tick, 650);
    }
    function drawC() {
      if (!resC) return;
      var R = resC, j = stC.j, st = R.steps[j], r = R.r, k = R.k;
      slk.out.textContent = k; slj.out.textContent = j + ' / ' + R.Kstar;
      var reached = j === R.Kstar;
      verdicts.c.innerHTML = 'rank(R − I) = ' + k + ' reached in <span class="m">⌈' + k + '/' + r + '⌉ = ' + R.Kstar + ' cycle' + (R.Kstar > 1 ? 's' : '') + '</span> · any R in ≤ ' + R.Jmax;
      /* legends */
      legC.innerHTML = '';
      var dot = function (fill, stroke, rr) { var s = S('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' }); S('circle', { cx: 7, cy: 7, r: rr || 4, style: 'fill:' + fill + ';stroke:' + stroke + ';stroke-width:1.5' }, s); return s; };
      legC.appendChild(li(dot('none', 'var(--ochre)', 5.5), 'eigenvalues of target <i>R</i>'));
      legC.appendChild(li(dot('var(--moss)', 'var(--paper)'), 'of <i>R</i>₁⋯<i>R<sub>j</sub></i>'));
      legC.appendChild(li(dot('var(--ink-3)', 'var(--paper)'), 'still at 1'));
      drawCircle(R, st);
      capC.innerHTML = 'each cycle moves at most r = ' + r + ' eigenvalues off 1 · rank(<i>P</i> − <i>I</i>) = # eigenvalues ≠ 1';
      legC2.innerHTML = '';
      legC2.appendChild(li(sw('var(--tide)', null, true), 'rank(<i>R</i>₁⋯<i>R<sub>j</sub></i> − <i>I</i>)'));
      legC2.appendChild(li(sw('var(--ink-3)', '4 3'), 'bound <i>jr</i>'));
      legC2.appendChild(li(sw('var(--ochre)'), 'target <i>k</i>'));
      legC2.appendChild(li(sw('var(--ink-2)', '1.5 3', false), 'random HRA cycles'));
      drawBars(R, j);
      /* reflection strip */
      stripC.innerHTML = '';
      stripC.appendChild(h('span', { class: 'rb-tag', html: '<span class="nt">R</span> = product of exactly <span class="nt">' + k + '</span> reflections, grouped into HRA<span class="nt">' + subs(r) + '</span> cycles' }));
      var strip = h('div', { class: 'rb-strip' });
      R.cycles.forEach(function (slots, jj) {
        var box = h('div', { class: 'rb-cyc' + (jj + 1 === j ? ' on' : jj + 1 > j ? ' off' : '') });
        box.appendChild(h('span', { class: 'rb-cyc-l', text: 'cycle ' + (jj + 1) }));
        var row = h('div', { class: 'rb-slots' });
        slots.forEach(function (s) {
          row.appendChild(h('span', { class: 'rb-slot' + (s.pad ? ' pad' : s.plane % 2 ? ' alt' : ''), title: s.pad ? 'padding H_u H_u = I' : 'reflection in rotation plane ' + (s.plane + 1), text: s.pad ? 'u' : 'H' }));
        });
        box.appendChild(row);
        strip.appendChild(box);
      });
      stripC.appendChild(strip);
      var pads = R.cycles.length * r - k;
      stripC.appendChild(h('div', { class: 'rb-cap', style: 'text-align:left;margin-top:.3rem', html: 'two reflections per rotation plane (shaded alternately)' + (pads ? ' · ' + pads + ' padding slot' + (pads > 1 ? 's' : '') + ' <i>H<sub>u</sub>H<sub>u</sub></i> = <i>I</i>' : '') }));
      /* card */
      cardC.innerHTML = '';
      var b = h('div', { class: 'rb-banner ' + (reached ? 'm' : 'o') });
      b.innerHTML = reached ? '<span class="bt">Target reached in ' + R.Kstar + ' cycle' + (R.Kstar > 1 ? 's' : '') + '</span><span class="bs">‖W<sub>j</sub> − W₀R‖<sub>F</sub> = ' + sci(st.dist, 1) + ' · ' + (R.Kstar === 1 ? 'k ≤ r: one HRA' + subs(r) + ' element is enough' : 'minimal: ' + (R.Kstar - 1) + ' cycle' + (R.Kstar === 2 ? '' : 's') + ' reach only rank ≤ ' + ((R.Kstar - 1) * r) + ' &lt; ' + k) + '</span>'
        : '<span class="bt">' + (R.Kstar - j) + ' more cycle' + (R.Kstar - j > 1 ? 's' : '') + ' to go</span><span class="bs">‖W<sub>j</sub> − W₀R‖<sub>F</sub> = ' + small(st.dist) + ' · rank so far ' + st.rank + ' of ' + k + '</span>';
      cardC.appendChild(b);
      var dl = h('dl', { class: 'rb-kv' });
      kv(dl, 'rank(W<sub>j</sub> − W₀) = rank(R₁⋯R<sub>j</sub> − I)', 'Cor 5', st.rankW + ' ≤ min(jr, n) = ' + Math.min(j * r, N), st.rankW <= j * r && st.rankW === st.rank ? 'm' : 's');
      var maxC = 0; for (var q = 1; q <= j; q++) maxC = Math.max(maxC, R.steps[q].rankC);
      kv(dl, 'each cycle: rank(R<sub>i</sub> − I)', 'HRA' + subs(r), j ? 'max ' + maxC + ' ≤ r = ' + r : '—');
      kv(dl, 'det(R₁⋯R<sub>j</sub>)', 'in SO(16)', '+' + st.det.toFixed(12).replace(/0+$/, '').replace(/\.$/, ''), Math.abs(st.det - 1) < 1e-9 ? 'm' : 's');
      kv(dl, 'K<sub>out</sub> = WW<sup>⊤</sup>, relative drift', 'Cor 7', tiny(st.kout), 'm');
      kv(dl, 'rank W₀', 'injective, 24×16', String(R.injective), R.injective === N ? 'm' : 's');
      cardC.appendChild(dl);
      live('Panel C: after ' + j + ' of ' + R.Kstar + ' cycles, rank(R1...Rj - I) = ' + st.rank + ', target rank ' + k + '.');
    }
    function drawCircle(R, st) {
      circC.innerHTML = '';
      var Wd = 320, cx = 160, cy = 160, rad = 118;
      var svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Wd, role: 'img', 'aria-label': 'Eigenvalues on the unit circle: ' + st.eig.angles.length + ' conjugate pairs have left 1, ' + st.eig.fixed + ' eigenvalues remain at 1.' }, circC);
      /* labels are sized in rendered pixels: the viewBox is 320 units wide */
      var csc = (circC.getBoundingClientRect().width || Wd) / Wd, U = function (px) { return +Math.min(px * 1.5, px / csc).toFixed(2); };
      S('line', { x1: cx - rad - 16, x2: cx + rad + 16, y1: cy, y2: cy, style: 'stroke:var(--rule);stroke-width:1' }, svg);
      S('line', { x1: cx, x2: cx, y1: cy - rad - 16, y2: cy + rad + 16, style: 'stroke:var(--rule);stroke-width:1' }, svg);
      S('circle', { cx: cx, cy: cy, r: rad, style: 'fill:none;stroke:var(--ink-3);stroke-width:1.2' }, svg);
      var af = U(12);
      T(svg, cx + rad + 6, cy - 6, '1', { class: 'num', style: 'font-size:' + af + 'px;fill:var(--ink-2)' });
      T(svg, cx - rad - 6, cy - 6, '−1', { 'text-anchor': 'end', class: 'num', style: 'font-size:' + af + 'px;fill:var(--ink-2)' });
      T(svg, cx + 6, cy - rad - 6, 'i', { class: 'serif', style: 'font-size:' + U(13) + 'px;fill:var(--ink-2);font-style:italic' });
      var P = function (a) { return [cx + rad * Math.cos(a), cy - rad * Math.sin(a)]; };
      /* arcs for reached planes */
      /* one concentric arc per rotation plane already built: the angle it turns (upper half only) */
      st.eig.angles.slice().sort(function (x, y) { return y - x; }).forEach(function (a, t) {
        var rr = rad * (0.88 - 0.065 * t), p0 = [cx + rr, cy], p1 = [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
        S('path', { d: 'M' + p0[0].toFixed(2) + ',' + p0[1] + 'A' + rr.toFixed(2) + ',' + rr.toFixed(2) + ' 0 0 0 ' + p1[0].toFixed(2) + ',' + p1[1].toFixed(2), style: 'fill:none;stroke:var(--moss);stroke-width:1.4;opacity:.55;stroke-linecap:round' }, svg);
        S('circle', { cx: p1[0], cy: p1[1], r: 1.8, style: 'fill:var(--moss);opacity:.8' }, svg);
      });
      R.target.eig.angles.forEach(function (a) { [a, -a].forEach(function (b) { var p = P(b); S('circle', { cx: p[0], cy: p[1], r: 7.5, style: 'fill:none;stroke:var(--ochre);stroke-width:1.6' }, svg); }); });
      st.eig.angles.forEach(function (a) { [a, -a].forEach(function (b) { var p = P(b); S('circle', { cx: p[0], cy: p[1], r: 4.6, style: 'fill:var(--moss);stroke:var(--paper);stroke-width:1' }, svg); }); });
      if (st.eig.fixed) {
        var p1 = P(0);
        S('circle', { cx: p1[0], cy: p1[1], r: 5.5, style: 'fill:var(--ink-3);stroke:var(--paper);stroke-width:1' }, svg);
        T(svg, p1[0] - 9, p1[1] + 6 + af, '×' + st.eig.fixed, { 'text-anchor': 'end', class: 'num', style: 'font-size:' + af + 'px;fill:var(--ink-2)' });
      }
      /* R₁⋯R_j in Source Serif with a real subscript (neither face has a ⱼ glyph) */
      var HALO = 'paint-order:stroke;stroke:var(--paper-2);stroke-width:4px;stroke-linejoin:round;';
      var mf = U(14), sf = Math.max(U(12), mf * 0.8), mt0 = T(svg, cx, cy + rad * 0.42, 'R', { 'text-anchor': 'middle', class: 'serif', style: HALO + 'font-size:' + mf + 'px;fill:var(--ink);font-style:italic' });
      mt0.textContent = '';
      [['R', 0], ['1', 1], ['⋯', -1], ['R', 0], [st.j === 0 ? 'j' : String(st.j), 1]].forEach(function (pt, q, arr) {
        var prev = q ? arr[q - 1][1] : 0, dy = (pt[1] === 1 ? mf * 0.25 : 0) - (prev === 1 ? mf * 0.25 : 0);
        /* numerals and the ellipsis upright, the variables R and j italic */
        var upright = pt[1] === -1 || /^[0-9]+$/.test(pt[0]);
        var tsp = S('tspan', { dy: dy || null, style: (pt[1] === 1 ? 'font-size:' + sf + 'px;' : '') + (upright ? 'font-style:normal' : '') }, mt0); tsp.textContent = pt[0];
      });
      if (st.j === 0) { var eqI = S('tspan', { dy: -mf * 0.25, style: 'font-style:normal' }, mt0); eqI.textContent = ' = '; var iI = S('tspan', null, mt0); iI.textContent = 'I'; }
      T(svg, cx, cy + rad * 0.42 + mf * 1.25, 'rank − I = ' + st.rank, { 'text-anchor': 'middle', style: HALO + 'font-size:' + U(12) + 'px;font-weight:500;fill:var(--ink-2)' });
    }
    function drawBars(R, j) {
      barsC.innerHTML = '';
      var W = Math.max(260, Math.floor(barsC.getBoundingClientRect().width) || barsC.clientWidth || 420), H = 198, ml = 30, mr = 12, mt = 18, mb = 34, iw = W - ml - mr, ih = H - mt - mb;
      var J = R.Jmax, bw = Math.min(34, iw / J * 0.56);
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Rank of R1...Rj - I per cycle, against the bound jr and the target rank ' + R.k + '.' }, barsC);
      var x = function (jj) { return ml + iw * (jj - 0.5) / J; }, y = function (v) { return mt + ih * (1 - v / N); };
      [0, 4, 8, 12, 16].forEach(function (v) { S('line', { x1: ml, x2: ml + iw, y1: y(v), y2: y(v), style: 'stroke:var(--rule);stroke-width:1' }, svg); T(svg, ml - 5, y(v) + 4, String(v), { 'text-anchor': 'end', class: 'num', style: 'font-size:11px;fill:var(--ink-2)' }); });
      for (var jj = 1; jj <= J; jj++) {
        var bd = Math.min(jj * R.r, N);
        S('rect', { x: x(jj) - bw / 2, y: y(bd), width: bw, height: y(0) - y(bd), rx: 2, style: 'fill:none;stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 2' }, svg);
        T(svg, x(jj), mt + ih + 16, String(jj), { 'text-anchor': 'middle', class: 'num', style: 'font-size:11px;fill:' + (jj === j ? 'var(--ink)' : 'var(--ink-2)') });
        if (jj <= R.Kstar) {
          var v = R.steps[jj].rank, on = jj <= j;
          var fill = jj === R.Kstar ? 'var(--moss)' : 'var(--tide)';
          S('rect', { x: x(jj) - bw / 2 + 2, y: y(v), width: bw - 4, height: y(0) - y(v), rx: 2, style: 'fill:' + fill + ';opacity:' + (on ? 0.9 : 0.18) }, svg);
          if (on) T(svg, x(jj), y(v) - 5, String(v), { 'text-anchor': 'middle', class: 'num', style: 'paint-order:stroke;stroke:var(--paper-2);stroke-width:3px;stroke-linejoin:round;font-size:12px;font-weight:500;fill:var(--ink)' });
        }
      }
      T(svg, ml + iw, mt + ih + 31, 'cycle j', { 'text-anchor': 'end', style: 'font-size:12px;font-weight:500;fill:var(--ink-2)' });
      S('line', { x1: ml, x2: ml + iw, y1: y(R.k), y2: y(R.k), style: 'stroke:var(--ochre);stroke-width:1.5' }, svg);
      T(svg, ml + 4, y(R.k) - 5, 'k = ' + R.k, { style: 'font-size:12px;font-weight:500;fill:var(--ochre-ink);paint-order:stroke;stroke:var(--paper-2);stroke-width:3px;stroke-linejoin:round' });
      var gd = ''; for (jj = 0; jj <= J; jj++) gd += (jj ? 'L' : 'M') + (jj === 0 ? ml : x(jj)).toFixed(1) + ',' + y(R.gen[jj]).toFixed(1);
      S('path', { d: gd, style: 'fill:none;stroke:var(--ink-2);stroke-width:1.3;stroke-dasharray:1.5 3' }, svg);
      for (jj = 1; jj <= J; jj++) S('circle', { cx: x(jj), cy: y(R.gen[jj]), r: 2.2, style: 'fill:var(--ink-2)' }, svg);
    }

    /* ---------- footer, live region ---------- */
    stage.appendChild(h('div', { class: 'rb-foot', html: 'Setup: <i>W</i>₀ ∈ ℝ²⁴ˣ¹⁶, injective, singular values 2 → 0.5 evenly spaced; float64 in your browser, seeded. Ranks by one-sided Jacobi SVD at tolerance 10⁻⁹·σ₁. dim <i>R<sub>K</sub></i> in B is the rank of the right-trivialised Jacobian of the product of all factors at a random point; partial-Gram and <i>K</i><sub>out</sub> drifts use skew entries ≈ 0.05, as in the check of Thm V.3. B shows the rotation part: BOFT’s row scale diag(<i>s</i>) acts separately, and GOFT’s released code also trains an input scale by default, which this panel leaves out. Re-HRA multiplies plain reflections (HF <code>apply_GS=False</code>). Papers: ReLoRA <a href="https://arxiv.org/abs/2307.05695">2307.05695</a>, LoRA-XS <a href="https://arxiv.org/abs/2405.17604">2405.17604</a>, OFT <a href="https://arxiv.org/abs/2306.07280">2306.07280</a>, BOFT <a href="https://arxiv.org/abs/2311.06243">2311.06243</a>, GOFT <a href="https://arxiv.org/abs/2404.04316">2404.04316</a>, HRA <a href="https://arxiv.org/abs/2405.17484">2405.17484</a>.' }));
    var liveEl = h('div', { class: 'rb-live', 'aria-live': 'polite' });
    stage.appendChild(liveEl);
    var liveT = null;
    function live(msg) { clearTimeout(liveT); liveT = setTimeout(function () { liveEl.textContent = msg; }, 400); }

    /* ---------- boot ---------- */
    computeA(); computeB(); computeC();
    setTab(tab);
    A.typeset(stage);
    observe(chartA, function () { if (tab === 'a') drawA(); });
    observe(mchartB, function () { if (tab === 'b') drawB(); });
    observe(barsC, function () { if (tab === 'c') drawC(); });
    A.onTheme(function () { if (tab === 'a') drawA(); else if (tab === 'b') drawB(); else drawC(); });
  });
})();
