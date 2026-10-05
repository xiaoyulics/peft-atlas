/* fig-workbench.js — Figure 6, "The plug workbench: the hole and the plug".
   Cut W0 out of a linear layer and a hole V -> U remains (V = R^n, U = R^m, y = W x). A PEFT method is a
   plug: a small tensor network of frozen and trainable boxes built from series composition, formal sums,
   Hadamard diamonds (copy spiders), Kronecker reshapes, diagonal spiders, a parameter-side row norm and
   sigma boxes. Every readout is computed live from the plug at m = n = 8:
     |D|         trainable entries (= |M| = dim Q)                                        Def I.2
     degree      maximum number of trainable boxes on one path of the expanded formal sum
     rank bound  brute-force minimum over vertex bipartitions of each path's cut weight,  Thm II.10
                 copy spiders counted once, summed over paths (subadditivity)
     d(M)        numerical Jacobian rank of q -> rho(q): central differences (exact for   Def I.5
                 multilinear plugs), one-sided Jacobi SVD, threshold 1e-8 sigma_max, Float64
     gauge       exhibited Lie-algebra generators (g g^-1 on wires, Hadamard spider tori,  Prop II.12(a)
                 scalar trades), each verified to lie in ker d rho; orbit dim = their rank
     merge       M-infinity iff a sigma sits on the activation path, else M1              Thm VI.3
     identity    structural match against the presets, else nearest invariant signature
   Classic script; registers Atlas.register('workbench', ...). */
(function () {
  'use strict';

  var M = 8, N = 8, MN = M * N, PMAX = 200, VMAX = 18;
  var SUBS = '₀₁₂₃₄₅₆₇₈₉';
  var SUPS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  function sub(k) { var s = String(k), o = ''; for (var i = 0; i < s.length; i++) { var c = s.charAt(i); o += (c >= '0' && c <= '9') ? SUBS.charAt(+c) : c; } return o; }
  function sup(k) { var s = String(k), o = ''; for (var i = 0; i < s.length; i++) { var c = s.charAt(i); o += (c >= '0' && c <= '9') ? SUPS.charAt(+c) : c === '-' ? '⁻' : c; } return o; }
  /* HTML readouts: Unicode sub/superscript digits become <sub>/<sup> (Plex Mono has no glyphs for most of them),
     and σ_max gets a real subscript */
  function subHtml(h) {
    return String(h).replace(/[₀-₉]+/g, function (m) { var o = ''; for (var i = 0; i < m.length; i++) o += SUBS.indexOf(m.charAt(i)); return '<sub>' + o + '</sub>'; })
      .replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+/g, function (m) { var o = ''; for (var i = 0; i < m.length; i++) { var c = m.charAt(i); o += c === '⁻' ? '−' : SUPS.indexOf(c); } return '<sup>' + o + '</sup>'; })
      .replace(/σ_max/g, 'σ<sub>max</sub>');
  }
  function fmtExp(v) { if (!(v > 0)) return '0'; var e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e); if (m >= 9.5) { m = 1; e += 1; } return Math.round(m) + '×10' + sup(e); }

  /* =====================================================================================
     1. The plug grammar. A plug is a formal sum (root 'sum') of paths; a path is a chain of
        nodes in flow order (input first). Node types:
          box  {role: train|frozen, st: dense|diag, fk: W0|rand|svdU|svdV|sum|eye|blk, out, inp}
          sig  elementwise sigma (GELU, tanh form) on a wire
          had  Hadamard diamond: copy spider, branches a and b, merge spider   (X ⊙ Y)
          kron Kronecker reshape: R^8 = R^{a} ⊗ R^{b}, factors a and b          (X ⊗ Y)
          norm parameter-side row normalisation of the inner sum (DoRA)
          sum  formal sum of chains
     ===================================================================================== */
  var UIDC = 0;
  function nid(p) { UIDC += 1; return p + UIDC; }
  function TB(label, out, inp) { return { t: 'box', id: nid('b'), role: 'train', st: 'dense', fk: null, label: label, out: out, inp: inp }; }
  function TD(label, dim) { return { t: 'box', id: nid('b'), role: 'train', st: 'diag', fk: null, label: label, out: dim, inp: dim }; }
  function FB(label, fk, out, inp, blk) { var b = { t: 'box', id: nid('b'), role: 'frozen', st: 'dense', fk: fk, label: label, out: out, inp: inp }; if (blk != null) b.blk = blk; return b; }
  function W0B() { return FB('W₀', 'W0', M, N); }
  function SG(dim) { return { t: 'sig', id: nid('s'), label: 'σ', out: dim, inp: dim }; }
  function HAD(a, b) { return { t: 'had', id: nid('h'), a: a, b: b }; }
  function KRON(a, b) { return { t: 'kron', id: nid('k'), a: a, b: b }; }
  function SUM(lanes) { return { t: 'sum', id: nid('u'), lanes: lanes }; }
  function NORM(lanes) { return { t: 'norm', id: nid('n'), inner: SUM(lanes) }; }
  function clone(x) { return JSON.parse(JSON.stringify(x)); }

  /* =====================================================================================
     2. Presets. chk(r) gives the closed forms of the framework that the live numbers must
        reproduce (Prop II.12 / Prop II.13 / Thm II.10 / Thm II.11); null fields have no
        closed form in the framework and are shown as computed only.
     ===================================================================================== */
  function loraD(r) { return r * (M + N - r); }
  var PRESETS = [
    { id: 'lora', name: 'LoRA', rs: [1, 2, 3, 4], title: function (r) { return 'LoRA' + sub(r); },
      build: function (r) { return SUM([[W0B()], [TB('A', r, N), TB('B', M, r)]]); },
      note: 'rsLoRA and LoRA+ are this same plug with other hyperparameters (Cor III.11).',
      chk: function (r) { return { d: loraD(r), dTxt: 'r(m+n−r)', dRef: 'Prop II.12', eq: true, rank: r, rTxt: 'r', rRef: 'Thm II.10' }; } },
    { id: 'lorafa', name: 'LoRA-FA', rs: [1, 2, 3, 4], title: function (r) { return 'LoRA-FA' + sub(r); },
      build: function (r) { return SUM([[W0B()], [FB('A₀', 'rand', r, N), TB('B', M, r)]]); },
      note: 'A₀ is a frozen random frame, so ρ is linear in B and the image W₀ + {BA₀} is an affine space of dimension mr.',
      chk: function (r) { return { d: M * r, dTxt: 'mr (linear, injective)', dRef: 'Prop II.12', eq: true, rank: r, rTxt: 'r', rRef: 'Thm II.10' }; } },
    { id: 'vera', name: 'VeRA', rs: [1, 2, 3, 4], title: function (r) { return 'VeRA (r = ' + r + ')'; },
      build: function (r) { return SUM([[W0B()], [FB('A₀', 'rand', r, N), TD('d', r), FB('B₀', 'rand', M, r), TD('b', M)]]); },
      note: 'Only the two diagonal spiders train; B₀ and A₀ are frozen random frames.',
      chk: function (r) { return { d: M + r - 1, dTxt: 'm+r−1', dRef: 'Prop II.12', eq: true, rank: r, rTxt: 'r', rRef: 'Thm II.10' }; } },
    { id: 'loraxs', name: 'LoRA-XS', rs: [1, 2, 3, 4], title: function (r) { return 'LoRA-XS (r = ' + r + ')'; },
      build: function (r) { return SUM([[W0B()], [FB('ΣVᵀ', 'svdV', r, N), TB('R', r, r), FB('U', 'svdU', M, r)]]); },
      note: 'The frozen frames are the top-r SVD frames of W₀; Prop I.8(a) builds it as a product of LoRA-FA and LoRA-FB.',
      chk: function (r) { return { d: r * r, dTxt: 'r² (linear, injective)', dRef: 'Prop II.12', eq: true, rank: r, rTxt: 'r', rRef: 'Thm II.10' }; } },
    { id: 'miss', name: 'MiSS', rs: [1, 2, 4], title: function (r) { return 'MiSS (r = ' + r + ')'; },
      build: function (r) { return SUM([[W0B()], [FB('S', 'sum', r, N), TB('D', M, r)]]); },
      note: 'MiSS in its default (balance) mode is LoRA-FA with a frozen summing A, S = 1ᵀ ⊗ I (Prop II.13).',
      chk: function (r) { return { d: M * r, dTxt: 'mr (linear, injective)', dRef: 'Prop II.12', eq: true, rank: r, rTxt: 'r', rRef: 'Thm II.10' }; } },
    { id: 'gralora', name: 'GraLoRA', rs: [2, 4], title: function (r) { return 'GraLoRA (k = 2, r = ' + r + ')'; },
      build: function (r) {
        var lanes = [[W0B()]], h = r / 2;
        for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++) {
          lanes.push([FB('P' + sub(j + 1), 'blk', N / 2, N, j), TB('A' + sub('' + (i + 1) + (j + 1)), h, N / 2),
            TB('B' + sub('' + (i + 1) + (j + 1)), M / 2, h), FB('E' + sub(i + 1), 'blk', M, M / 2, i)]);
        }
        return SUM(lanes);
      },
      note: 'Four block LoRAs of rank r/k. GraLoRA has the dimension of LoRAᵣ and k times its maximal rank, with every block capped at rank r/k (Prop II.12(c)).',
      chk: function (r) { return { d: loraD(r), dTxt: 'r(m+n−r)', dRef: 'Prop II.12', eq: true, rank: Math.min(2 * r, M, N), rTxt: 'min(kr, m, n)', rRef: 'Prop II.12' }; } },
    { id: 'loha', name: 'LoHa', rs: [1, 2, 3, 4], title: function (r) { return 'LoHa' + sub(r) + ' (r₁ = r₂ = ' + r + ')'; },
      build: function (r) { return SUM([[W0B()], [HAD([TB('A₁', r, N), TB('B₁', M, r)], [TB('A₂', r, N), TB('B₂', M, r)])]]); },
      note: 'The copy spiders multiply the cut, so the rank can reach r². Their torus, of dimension m+n−1, is pure gauge and costs dimension (Prop II.12).',
      chk: function (r) { return { d: Math.min(MN, 2 * r * (M + N - r) - (M + N - 1)), dTxt: 'min(mn, 2r(m+n−r) − (m+n−1))', dRef: 'Prop II.12, upper bound', eq: false, rank: Math.min(r * r, M, N), rTxt: 'min(r², m, n)', rRef: 'Prop II.12' }; } },
    { id: 'hira', name: 'HiRA', rs: [1, 2, 3, 4], title: function (r) { return 'HiRA' + sub(r); },
      build: function (r) { return SUM([[W0B()], [HAD([W0B()], [TB('A', r, N), TB('B', M, r)])]]); },
      note: 'HiRA is LoRA transported by X ↦ W₀ ⊙ X, which is invertible because this W₀ has no zero entries. The transport keeps d and changes the rank (Prop II.13).',
      chk: function (r) { return { d: loraD(r), dTxt: 'r(m+n−r)', dRef: 'Prop II.13, transport of LoRA', eq: true, rank: Math.min(M, N, r * Math.min(M, N)), rTxt: 'r · rk W₀, capped', rRef: 'Thm II.10' }; } },
    { id: 'lokr', name: 'LoKr', rs: [1, 2, 3, 4], title: function (r) { return 'LoKr (r′ = ' + r + ')'; },
      build: function (r) { return SUM([[W0B()], [KRON([TB('C', 2, 2)], [TB('A', r, 4), TB('B', 4, r)])]]); },
      note: 'C ⊗ BA on ℝ⁸ = ℝ² ⊗ ℝ⁴ has rank at most min(m₁, n₁)·r′, so its image lies inside LoRA₂ᵣ′’s (Thm II.11(e)).',
      chk: function (r) { return { d: null, rank: Math.min(2 * r, M, N), rTxt: 'min(m₁,n₁) · r′', rRef: 'Thm II.11(e)' }; } },
    { id: 'krona', name: 'KronA', rs: [0], title: function () { return 'KronA (2×2 ⊗ 4×4)'; },
      build: function () { return SUM([[W0B()], [KRON([TB('C', 2, 2)], [TB('D', 4, 4)])]]); },
      note: 'With unconstrained factors, KronA is LoRA₁ across the other cut of the 4-leg tensor (Prop II.13).',
      chk: function () { return { d: 2 * 2 + 4 * 4 - 1, dTxt: 'm₁n₁ + m₂n₂ − 1', dRef: 'Prop II.12', eq: true, rank: M, rTxt: 'rk C · rk D', rRef: 'Thm II.11(b)' }; } },
    { id: 'mora', name: 'MoRA', rs: [0], title: function () { return 'MoRA-style block reshape (r̂ = 4)'; },
      build: function () { return SUM([[W0B()], [KRON([FB('I₂', 'eye', 2, 2)], [TB('M', 4, 4)])]]); },
      note: 'One square M acts on both blocks of the reshaped input, Δ = I₂ ⊗ M, with no rotation. The map is linear and injective, and the rank can reach full.',
      chk: function () { return { d: 16, dTxt: 'r̂² (linear, injective)', dRef: 'Prop II.12', eq: true, rank: M, rTxt: 'rk I₂ · rk M', rRef: 'Thm II.11(b)' }; } },
    { id: 'dora', name: 'DoRA', rs: [1, 2, 3, 4], title: function (r) { return 'DoRA' + sub(r); },
      build: function (r) { return SUM([[NORM([[W0B()], [TB('A', r, N), TB('B', M, r)]]), TD('m', M)]]); },
      note: 'The row norm acts on the weight matrix, so the trained layer is still one m×n matrix and merges (Thm VI.3).',
      chk: function (r) { return { d: Math.min(MN, loraD(r) + M), dTxt: 'min(mn, r(m+n−r)+m)', dRef: 'Prop II.12, generic W₀', eq: true, rank: M, rTxt: 'min(m, n)', rRef: 'Prop II.12' }; } },
    { id: 'ia3', name: '(IA)³', rs: [0], title: function () { return '(IA)³ (output scaling)'; },
      build: function () { return SUM([[W0B(), TD('ℓ', M)]]); },
      note: 'W = diag(ℓ)W₀ is linear in ℓ; it fuses into its own box.',
      chk: function () { return { d: M, dTxt: 'm (linear, injective)', dRef: 'Prop II.12', eq: true, rank: M, rTxt: 'min(m, n)', rRef: 'Thm II.10' }; } },
    { id: 'full', name: 'Full FT', rs: [0], title: function () { return 'Full fine-tuning'; },
      build: function () { return SUM([[W0B()], [TB('ΔW', M, N)]]); },
      note: 'The terminal object of Meth(Θ, θ₀) (Obs I.4).',
      chk: function () { return { d: MN, dTxt: 'mn', dRef: 'Obs I.4', eq: true, rank: M, rTxt: 'min(m, n)', rRef: 'Thm II.10' }; } },
    { id: 'houlsby', name: 'Houlsby', rs: [1, 2, 3, 4], title: function (r) { return 'Houlsby adapter (width ' + r + ', GELU)'; },
      build: function (r) { return SUM([[W0B()], [W0B(), TB('D', r, M), SG(r), TB('U', M, r)]]); },
      note: 'A serial adapter after the layer, without biases. GELU leaves only a discrete permutation gauge on the hidden units (Lemma VI.2).',
      chk: function () { return null; } }
  ];
  var HIDDEN = [
    { id: 'serial', name: 'serial linear adapter', rs: [1, 2, 3, 4], title: function (r) { return 'serial linear adapter (width ' + r + ')'; },
      build: function (r) { return SUM([[W0B()], [W0B(), TB('D', r, M), TB('U', M, r)]]); },
      note: 'Isomorphic to LoRA when W₀ is invertible (Prop VI.4(b)); it is the Houlsby adapter with σ = id.',
      chk: function (r) { return { d: loraD(r), dTxt: 'r(m+n−r)', dRef: 'Prop VI.4(b), W₀ invertible', eq: true, rank: r, rTxt: 'r', rRef: 'Thm II.10' }; } },
    { id: 'parallel', name: 'parallel adapter', rs: [1, 2, 3, 4], title: function (r) { return 'parallel adapter (width ' + r + ', GELU)'; },
      build: function (r) { return SUM([[W0B()], [TB('D', r, N), SG(r), TB('U', M, r)]]); },
      note: 'With σ = id it has the same ρ as LoRA (Prop VI.4(a)). With GELU on the activation path it extends the layer, and no single weight matrix represents it.',
      chk: function () { return null; } },
    { id: 'frozen', name: 'Frozen', rs: [0], title: function () { return 'the frozen model'; },
      build: function () { return SUM([[W0B()]]); },
      note: 'The initial object of Meth(Θ, θ₀) (Obs I.4).',
      chk: function () { return { d: 0, dTxt: '0', dRef: 'Obs I.4', eq: true, rank: 0, rTxt: '0', rRef: 'Thm II.10' }; } }
  ];
  function presetById(id) { var all = PRESETS.concat(HIDDEN); for (var i = 0; i < all.length; i++) if (all[i].id === id) return all[i]; return null; }

  /* =====================================================================================
     3. Small dense linear algebra (rows of Float64Array) and the one-sided Jacobi SVD
     ===================================================================================== */
  function zeros(r, c) { var A = new Array(r); for (var i = 0; i < r; i++) A[i] = new Float64Array(c); return A; }
  function eyeM(r, c) { var A = zeros(r, c); for (var i = 0; i < Math.min(r, c); i++) A[i][i] = 1; return A; }
  function mul(A, B) {
    var m = A.length, k = B.length, n = B[0].length, C = zeros(m, n);
    for (var i = 0; i < m; i++) { var Ai = A[i], Ci = C[i]; for (var p = 0; p < k; p++) { var a = Ai[p]; if (a === 0) continue; var Bp = B[p]; for (var j = 0; j < n; j++) Ci[j] += a * Bp[j]; } }
    return C;
  }
  function addM(A, B, s) { var m = A.length, n = A[0].length, C = zeros(m, n), b = s == null ? 1 : s; for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) C[i][j] = A[i][j] + b * B[i][j]; return C; }
  function hadM(A, B) { var m = A.length, n = A[0].length, C = zeros(m, n); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) C[i][j] = A[i][j] * B[i][j]; return C; }
  function kronM(A, B) {
    var ma = A.length, na = A[0].length, mb = B.length, nb = B[0].length, C = zeros(ma * mb, na * nb);
    for (var i = 0; i < ma; i++) for (var j = 0; j < na; j++) { var a = A[i][j]; for (var p = 0; p < mb; p++) for (var q = 0; q < nb; q++) C[i * mb + p][j * nb + q] = a * B[p][q]; }
    return C;
  }
  function rowNorm(X) {
    var m = X.length, n = X[0].length, C = zeros(m, n);
    for (var i = 0; i < m; i++) { var s = 0; for (var j = 0; j < n; j++) s += X[i][j] * X[i][j]; s = Math.sqrt(s) || 1; for (j = 0; j < n; j++) C[i][j] = X[i][j] / s; }
    return C;
  }
  function gaussM(r, c, rnd, s) { var A = zeros(r, c); for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) A[i][j] = rnd.normal() * s; return A; }
  function hash(str) { var h = 2166136261; for (var i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
  function invM(A) {
    var n = A.length, M2 = A.map(function (r, i) { var o = new Float64Array(2 * n); for (var j = 0; j < n; j++) o[j] = r[j]; o[n + i] = 1; return o; }), i, j, k;
    for (i = 0; i < n; i++) {
      var p = i; for (k = i + 1; k < n; k++) if (Math.abs(M2[k][i]) > Math.abs(M2[p][i])) p = k;
      if (Math.abs(M2[p][i]) < 1e-12) return null;
      var t = M2[i]; M2[i] = M2[p]; M2[p] = t;
      var piv = M2[i][i]; for (j = 0; j < 2 * n; j++) M2[i][j] /= piv;
      for (k = 0; k < n; k++) if (k !== i) { var f = M2[k][i]; if (f) for (j = 0; j < 2 * n; j++) M2[k][j] -= f * M2[i][j]; }
    }
    return M2.map(function (r) { return r.slice(n); });
  }
  function gelu(z) { return 0.5 * z * (1 + Math.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z))); }

  /* singular values of the matrix whose columns are `cols`, descending (Hestenes one-sided Jacobi;
     works on the side with fewer vectors, accurate to roughly machine precision relative to sigma_max). */
  function svalsCols(cols) {
    var c = cols.length; if (!c) return [];
    var L = cols[0].length; if (!L) return [];
    var U, i, k;
    if (c <= L) { U = cols.map(function (v) { return Float64Array.from(v); }); }
    else { U = []; for (i = 0; i < L; i++) { var row = new Float64Array(c); for (k = 0; k < c; k++) row[k] = cols[k][i]; U.push(row); } }
    var n = U.length, len = U[0].length;
    for (var sweep = 0; sweep < 80; sweep++) {
      var rot = 0;
      for (var p = 0; p < n - 1; p++) {
        var up = U[p];
        for (var qi = p + 1; qi < n; qi++) {
          var uq = U[qi], al = 0, be = 0, ga = 0, t, a, b;
          for (t = 0; t < len; t++) { a = up[t]; b = uq[t]; al += a * a; be += b * b; ga += a * b; }
          if (ga === 0 || Math.abs(ga) <= 1e-15 * Math.sqrt(al * be)) continue;
          rot++;
          var ze = (be - al) / (2 * ga);
          var tt = (ze >= 0 ? 1 : -1) / (Math.abs(ze) + Math.sqrt(1 + ze * ze));
          var cs = 1 / Math.sqrt(1 + tt * tt), sn = cs * tt;
          for (t = 0; t < len; t++) { a = up[t]; b = uq[t]; up[t] = cs * a - sn * b; uq[t] = sn * a + cs * b; }
        }
      }
      if (!rot) break;
    }
    return U.map(function (v) { var x = 0; for (var t = 0; t < v.length; t++) x += v[t] * v[t]; return Math.sqrt(x); })
      .sort(function (x, y) { return y - x; });
  }
  function rankOf(sv, rtol) { if (!sv.length || !(sv[0] > 1e-300)) return 0; var tol = rtol * sv[0], r = 0; for (var i = 0; i < sv.length; i++) if (sv[i] > tol) r++; return r; }
  function colsOf(A) { var m = A.length, n = A[0].length, out = []; for (var j = 0; j < n; j++) { var v = new Float64Array(m); for (var i = 0; i < m; i++) v[i] = A[i][j]; out.push(v); } return out; }

  /* =====================================================================================
     4. Frozen boxes: a generic W0 (Gaussian, fixed seed), random frames, SVD frames of W0,
        summing boxes S = 1ᵀ ⊗ I, identities, block embeddings / selections.
     ===================================================================================== */
  var _W0 = null, _U = null;
  function W0mat() { if (!_W0) _W0 = gaussM(M, N, Atlas.rng(20261001), 1 / Math.sqrt(N)); return _W0; }
  function svdFrame() {
    if (_U) return _U;
    var W = W0mat(), G = zeros(M, M), i, j, k;
    for (i = 0; i < M; i++) for (j = 0; j < M; j++) { var s = 0; for (k = 0; k < N; k++) s += W[i][k] * W[j][k]; G[i][j] = s; }
    var e = Atlas.LA.symEig(G), idx = [];
    for (i = 0; i < M; i++) idx.push(i);
    idx.sort(function (a, b) { return e.values[b] - e.values[a]; });
    _U = idx.map(function (p) { var u = new Float64Array(M); for (var t = 0; t < M; t++) u[t] = e.vectors[t][p]; return u; });
    return _U;
  }
  function frozenMat(b) {
    var r = b.out, c = b.inp, A, i, j, p;
    if (b.st === 'diag') { var rd = Atlas.rng(b._seed); A = zeros(r, r); for (i = 0; i < r; i++) A[i][i] = 0.5 + rd(); return A; }
    switch (b.fk) {
      case 'W0': return W0mat();
      case 'svdU': { var U = svdFrame(); A = zeros(M, c); for (j = 0; j < c; j++) for (i = 0; i < M; i++) A[i][j] = U[j][i]; return A; }
      case 'svdV': { var U2 = svdFrame(), W = W0mat(); A = zeros(r, N); for (p = 0; p < r; p++) for (j = 0; j < N; j++) { var s = 0; for (i = 0; i < M; i++) s += U2[p][i] * W[i][j]; A[p][j] = s; } return A; }
      case 'sum': A = zeros(r, c); if (r <= c) { for (j = 0; j < c; j++) A[j % r][j] = 1; } else { for (i = 0; i < r; i++) A[i][i % c] = 1; } return A;
      case 'eye': return eyeM(r, c);
      case 'blk': A = zeros(r, c); if (r > c) { for (p = 0; p < c; p++) A[b.blk * c + p][p] = 1; } else if (r < c) { for (p = 0; p < r; p++) A[p][b.blk * r + p] = 1; } else return eyeM(r, c); return A;
      default: return gaussM(r, c, Atlas.rng(b._seed), 1 / Math.sqrt(c));
    }
  }
  var FK_NAMES = { W0: 'pretrained W₀ = θ₀', rand: 'random frame', svdU: 'SVD frame of W₀ (left)', svdV: 'SVD frame of W₀ (ΣVᵀ)', sum: 'summing box 1ᵀ ⊗ I', eye: 'identity', blk: 'block embedding' };
  var FK_SHORT = { W0: 'W₀', rand: 'random', svdU: 'SVD frame', svdV: 'SVD frame', sum: 'summing', eye: 'identity', blk: 'block' };

  /* =====================================================================================
     5. Compile: dims, validation, parameter offsets, scopes (for scalar trades), seeds.
     ===================================================================================== */
  function compile(root, salt) {
    var C = { root: root, info: {}, boxes: [], train: [], nP: 0, sigma: false, norm: false, errs: [], fz: {} };
    function err(s) { if (C.errs.indexOf(s) < 0) C.errs.push(s); }
    if (!root || root.t !== 'sum' || !root.lanes || !root.lanes.length) { err('The plug has no path.'); return C; }
    function walkChain(ch, ctx) {
      if (!ch || !ch.length) { err('A path is empty.'); return null; }
      var d0 = null, cur = null, scope = ctx.scope;
      for (var k = 0; k < ch.length; k++) {
        var nd = ch[k];
        var inf = { chain: ch, idx: k, lane: ctx.lane, scope: scope, inHad: ctx.inHad, inKron: ctx.inKron, inNorm: ctx.inNorm, top: ctx.top, path: ctx.path + '.' + k };
        C.info[nd.id] = inf;
        var d = walkNode(nd, inf);
        if (!d) return null;
        if (k === 0) d0 = d; else if (d.i !== cur) err('Shapes do not meet at ' + (nd.label || nd.t) + ', where a wire of size ' + cur + ' cannot feed a box that expects ' + d.i + '.');
        cur = d.o;
        if (nd.t === 'sig') scope = scope + '|' + nd.id;
      }
      return { i: d0.i, o: cur };
    }
    function walkNode(nd, inf) {
      var a, b, d;
      if (nd.t === 'box') {
        if (!(nd.out >= 1 && nd.inp >= 1 && nd.out <= 16 && nd.inp <= 16)) { err('Box sizes must lie between 1 and 16.'); return null; }
        if (nd.st === 'diag' && nd.out !== nd.inp) err('A diagonal spider must be square.');
        if (nd.role === 'frozen') {
          if (nd.fk === 'W0' && (nd.out !== M || nd.inp !== N)) err('W₀ is ' + M + '×' + N + '; it cannot be resized.');
          if (nd.fk === 'svdU' && (nd.out !== M || nd.inp > Math.min(M, N))) err('The left SVD frame of W₀ is m×k with k ≤ 8.');
          if (nd.fk === 'svdV' && (nd.inp !== N || nd.out > Math.min(M, N))) err('The right SVD frame of W₀ is k×n with k ≤ 8.');
          if (nd.fk === 'blk' && !(nd.out === nd.inp || (nd.out > nd.inp && (nd.blk + 1) * nd.inp <= nd.out) || (nd.out < nd.inp && (nd.blk + 1) * nd.out <= nd.inp))) err('A block embedding cannot take that shape.');
        } else {
          nd._off = C.nP; nd._sz = nd.st === 'diag' ? nd.out : nd.out * nd.inp; C.nP += nd._sz; C.train.push(nd);
        }
        nd._seed = (hash(inf.path + '|' + nd.label + '|' + nd.fk) ^ Math.imul(salt + 1, 2654435761)) >>> 0;
        nd._i = nd.inp; nd._o = nd.out;
        C.boxes.push(nd);
        return { i: nd.inp, o: nd.out };
      }
      if (nd.t === 'sig') {
        if (!inf.top) err('σ belongs on an activation path at the top level, not inside ⊙, ⊗ or a norm.');
        C.sigma = true; nd._i = nd.inp; nd._o = nd.out;
        return { i: nd.inp, o: nd.out };
      }
      if (nd.t === 'had' || nd.t === 'kron') {
        var sc = { lane: inf.lane, scope: inf.scope, inHad: inf.inHad || nd.t === 'had', inKron: inf.inKron || nd.t === 'kron', inNorm: inf.inNorm, top: false };
        a = walkChain(nd.a, Object.assign({}, sc, { path: inf.path + 'a' }));
        b = walkChain(nd.b, Object.assign({}, sc, { path: inf.path + 'b' }));
        if (!a || !b) return null;
        if (nd.t === 'had') {
          if (a.i !== b.i || a.o !== b.o) err('The two branches of a Hadamard diamond must have the same shape.');
          d = { i: a.i, o: a.o };
        } else d = { i: a.i * b.i, o: a.o * b.o };
        nd._i = d.i; nd._o = d.o;
        return d;
      }
      if (nd.t === 'norm') {
        C.norm = true;
        d = walkSum(nd.inner, { lane: inf.lane, scope: inf.scope + '|' + nd.id, inHad: inf.inHad, inKron: inf.inKron, inNorm: true, top: false, path: inf.path + 'n' });
        if (!d) return null;
        nd._i = d.i; nd._o = d.o;
        return d;
      }
      err('Unknown node.'); return null;
    }
    function walkSum(sm, ctx) {
      var d = null;
      if (!sm.lanes || !sm.lanes.length) { err('A sum has no path.'); return null; }
      for (var li = 0; li < sm.lanes.length; li++) {
        var dl = walkChain(sm.lanes[li], Object.assign({}, ctx, { path: ctx.path + 'l' + li }));
        if (!dl) return null;
        if (d && (d.i !== dl.i || d.o !== dl.o)) err('Every term of a sum must have the same shape.');
        d = d || dl;
      }
      sm._i = d.i; sm._o = d.o;
      return d;
    }
    root.lanes.forEach(function (lane, li) {
      var d = walkChain(lane, { lane: li, scope: 'root', inHad: false, inKron: false, inNorm: false, top: true, path: 'L' + li });
      if (d && (d.i !== N || d.o !== M)) err('Every path must map V = ℝ⁸ to U = ℝ⁸.');
    });
    root._i = N; root._o = M;
    return C;
  }

  /* =====================================================================================
     6. Evaluation: W(q) for linear plugs; y(q) on probe inputs for plugs with sigma.
     ===================================================================================== */
  function boxMat(b, q, C) {
    var A, i, j;
    if (b.role === 'train') {
      if (b.st === 'diag') { A = zeros(b.out, b.out); for (i = 0; i < b.out; i++) A[i][i] = q[b._off + i]; return A; }
      A = zeros(b.out, b.inp); for (i = 0; i < b.out; i++) for (j = 0; j < b.inp; j++) A[i][j] = q[b._off + i * b.inp + j];
      return A;
    }
    if (!C.fz[b.id]) C.fz[b.id] = frozenMat(b);
    return C.fz[b.id];
  }
  function evalMat(nd, q, C) {
    var S = null;
    switch (nd.t) {
      case 'box': return boxMat(nd, q, C);
      case 'had': return hadM(chainMat(nd.a, q, C), chainMat(nd.b, q, C));
      case 'kron': return kronM(chainMat(nd.a, q, C), chainMat(nd.b, q, C));
      case 'norm': return rowNorm(evalMat(nd.inner, q, C));
      case 'sum': nd.lanes.forEach(function (l) { var X = chainMat(l, q, C); S = S ? addM(S, X) : X; }); return S;
    }
    throw new Error('non-linear node in a linear evaluation');
  }
  function chainMat(ch, q, C) { var P = null; for (var k = 0; k < ch.length; k++) { var X = evalMat(ch[k], q, C); P = P ? mul(X, P) : X; } return P; }
  function applyChain(ch, H, q, C) {
    for (var k = 0; k < ch.length; k++) {
      var nd = ch[k];
      if (nd.t === 'sig') { var G = zeros(H.length, H[0].length); for (var i = 0; i < H.length; i++) for (var j = 0; j < H[0].length; j++) G[i][j] = gelu(H[i][j]); H = G; }
      else H = mul(evalMat(nd, q, C), H);
    }
    return H;
  }
  function applyRoot(root, X, q, C) { var S = null; root.lanes.forEach(function (l) { var Y = applyChain(l, X, q, C); S = S ? addM(S, Y) : Y; }); return S; }
  function Fvec(q, C) {
    var R = C.sigma ? applyRoot(C.root, C.X, q, C) : evalMat(C.root, q, C);
    var m = R.length, n = R[0].length, v = new Float64Array(m * n);
    for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) v[i * n + j] = R[i][j];
    return v;
  }
  function jacobianCols(C, q) {
    /* central differences; for multilinear plugs every coordinate enters with degree 1, so the
       difference quotient is exact for any step and h = 1/8 keeps rounding at machine level. */
    var P = C.nP, h = (C.sigma || C.norm) ? 1e-5 : 0.125, cols = new Array(P), qq = Float64Array.from(q);
    for (var k = 0; k < P; k++) {
      var v = qq[k];
      qq[k] = v + h; var fp = Fvec(qq, C);
      qq[k] = v - h; var fm = Fvec(qq, C);
      qq[k] = v;
      var col = new Float64Array(fp.length);
      for (var i = 0; i < fp.length; i++) col[i] = (fp[i] - fm[i]) / (2 * h);
      cols[k] = col;
    }
    C.h = h;
    return cols;
  }

  /* =====================================================================================
     7. Expansion into a formal sum of pure tensor networks, and the min-cut (Thm II.10)
     ===================================================================================== */
  function expandChain(ch) {
    var res = [[]];
    for (var k = 0; k < ch.length; k++) {
      var alts = expandNode(ch[k]), nx = [];
      for (var a = 0; a < res.length; a++) for (var b = 0; b < alts.length; b++) nx.push(res[a].concat(alts[b]));
      res = nx;
    }
    return res;
  }
  function expandNode(nd) {
    var out = [];
    if (nd.t === 'box' || nd.t === 'sig') return [[nd]];
    if (nd.t === 'sum') { nd.lanes.forEach(function (l) { out = out.concat(expandChain(l)); }); return out; }
    if (nd.t === 'norm') return expandNode(nd.inner).map(function (seq) { return seq.concat([{ t: 'nrm', id: nd.id, out: nd._o, inp: nd._i }]); });
    var A = expandChain(nd.a), B = expandChain(nd.b);
    A.forEach(function (ea) { B.forEach(function (eb) { out.push([{ t: nd.t, id: nd.id, a: ea, b: eb, out: nd._o, inp: nd._i }]); }); });
    return out;
  }
  function walkPure(seq, fn) { seq.forEach(function (x) { fn(x); if (x.t === 'had' || x.t === 'kron') { walkPure(x.a, fn); walkPure(x.b, fn); } }); }
  function termInfo(t) {
    var ids = [], sig = false, nrm = false, w0 = 0;
    walkPure(t, function (x) {
      if (x.t === 'box') { if (x.role === 'train') ids.push(x.id); else if (x.fk === 'W0') w0++; }
      else if (x.t === 'sig') sig = true; else if (x.t === 'nrm') nrm = true;
    });
    return { ids: ids, nTr: ids.length, sig: sig, nrm: nrm, w0: w0,
      bare: t.length === 1 && t[0].t === 'box' && t[0].role === 'frozen' && t[0].fk === 'W0',
      constant: !ids.length && !nrm };
  }
  function firstId(seq) { return seq.length ? seq[0].id : ''; }
  /* one pure path -> tensor network (vertices = boxes; hyperedges = wires and copy spiders); brute-force cut */
  function minCut(t) {
    var V = [], H = [];
    function he(dim, wire) { var h = { dim: dim, ends: [], li: false, lo: false, wire: wire }; H.push(h); return h; }
    function vert(x) { V.push(x); return V.length - 1; }
    function chain(seq, hin) { var h = hin; for (var i = 0; i < seq.length; i++) h = node(seq[i], h); return h; }
    function node(x, hin) {
      if (x.t === 'had') {
        var ha = chain(x.a, hin), hb = chain(x.b, hin);
        hb.ends.forEach(function (v) { ha.ends.push(v); }); hb.dead = true; ha.wire = 'o:' + x.id; ha.spider = true;
        return ha;
      }
      if (x.t === 'kron') {
        var ri = vert({ t: 'reshape' }); hin.ends.push(ri);
        var a0 = x.a[0], b0 = x.b[0];
        var ha0 = he(a0.inp, 'i:' + a0.id), hb0 = he(b0.inp, 'i:' + b0.id);
        ha0.ends.push(ri); hb0.ends.push(ri);
        var ha2 = chain(x.a, ha0), hb2 = chain(x.b, hb0);
        var ro = vert({ t: 'reshape' }); ha2.ends.push(ro); hb2.ends.push(ro);
        var ho = he(x.out, 'o:' + x.id); ho.ends.push(ro);
        return ho;
      }
      var v = vert(x); hin.ends.push(v);
      var h = he(x.out, 'o:' + x.id); h.ends.push(v);
      return h;
    }
    var h0 = he(N, 'i:' + firstId(t)); h0.li = true;
    var hl = chain(t, h0); hl.lo = true;
    H = H.filter(function (h) { return !h.dead; });
    var nv = V.length, best = { w: Infinity, cnt: Infinity, legs: Infinity, mask: 0 };
    if (nv > VMAX) return { w: Math.min(M, N), cross: [], trivial: true, V: V, H: H };
    for (var mask = 0; mask < (1 << nv); mask++) {
      var w = 1, cnt = 0, legs = 0;
      for (var e = 0; e < H.length; e++) {
        var h = H[e], hasI = h.li, hasO = h.lo;
        for (var k = 0; k < h.ends.length; k++) { if ((mask >> h.ends[k]) & 1) hasI = true; else hasO = true; }
        if (hasI && hasO) { w *= h.dim; cnt++; if (h.li || h.lo) legs++; }
      }
      if (w < best.w || (w === best.w && (cnt < best.cnt || (cnt === best.cnt && legs < best.legs)))) best = { w: w, cnt: cnt, legs: legs, mask: mask };
    }
    var cross = [];
    H.forEach(function (h) {
      var hasI = h.li, hasO = h.lo;
      h.ends.forEach(function (v) { if ((best.mask >> v) & 1) hasI = true; else hasO = true; });
      if (hasI && hasO) cross.push({ wire: h.wire, dim: h.dim, leg: h.li || h.lo });
    });
    return { w: best.w, cross: cross, nv: nv, nh: H.length, masks: 1 << nv };
  }
  function rankBound(terms) {
    var parts = [], total = 0, bare = 0, consts = 0, nonlin = false, masks = 0;
    terms.forEach(function (t, ti) {
      var inf = termInfo(t);
      if (inf.sig) { nonlin = true; return; }
      if (inf.bare) { bare++; return; }
      var c = minCut(t);
      masks += c.masks || 0;
      parts.push({ ti: ti, w: c.w, cut: c, inf: inf, lane: null });
      total += c.w;
      if (inf.constant) consts++;
    });
    if (nonlin) return { nonlin: true, parts: [], bare: bare, consts: consts };
    var rk0 = Math.min(M, N);                        /* rank of the generic W0 */
    var extra = bare === 0 ? rk0 : (bare - 1) * rk0;   /* Delta W = (plug) - W0 */
    return { nonlin: false, parts: parts, total: total, extra: extra, bare: bare, consts: consts, bound: Math.min(M, N, total + extra), masks: masks };
  }

  /* =====================================================================================
     8. Gauge: Lie-algebra generators of the exhibited gauge, as tangent vectors at q
     ===================================================================================== */
  function forEachChain(root, fn) {
    function ch(c) { fn(c); c.forEach(nd); }
    function nd(x) { if (x.t === 'had' || x.t === 'kron') { ch(x.a); ch(x.b); } else if (x.t === 'norm') x.inner.lanes.forEach(ch); }
    root.lanes.forEach(ch);
  }
  function forEachNode(root, fn) { forEachChain(root, function (c) { c.forEach(fn); }); }
  function isT(x) { return x && x.t === 'box' && x.role === 'train'; }
  function gaugeGens(C, q, terms) {
    var P = C.nP, gens = [], tags = [], pieces = [];
    function vec() { return new Float64Array(P); }
    function colScale(v, b, j, s) { if (b.st === 'diag') v[b._off + j] += s * q[b._off + j]; else for (var i = 0; i < b.out; i++) v[b._off + i * b.inp + j] += s * q[b._off + i * b.inp + j]; }
    function rowScale(v, b, i, s) { if (b.st === 'diag') v[b._off + i] += s * q[b._off + i]; else for (var j = 0; j < b.inp; j++) v[b._off + i * b.inp + j] += s * q[b._off + i * b.inp + j]; }
    /* (1) g g^-1 on a wire between two trainable boxes: X -> gX, Y -> Y g^-1 (all of GL_k when both
           are dense, the torus T_k when either is diagonal) */
    forEachChain(C.root, function (ch) {
      for (var k = 1; k < ch.length; k++) {
        var X = ch[k - 1], Y = ch[k];
        if (!isT(X) || !isT(Y)) continue;
        var kk = X.out, dense = X.st === 'dense' && Y.st === 'dense', pi = pieces.length;
        for (var a = 0; a < kk; a++) for (var b = 0; b < kk; b++) {
          if (!dense && a !== b) continue;
          var v = vec(), i, j;
          if (X.st === 'diag') v[X._off + a] += q[X._off + a];
          else for (j = 0; j < X.inp; j++) v[X._off + a * X.inp + j] += q[X._off + b * X.inp + j];
          if (Y.st === 'diag') v[Y._off + a] -= q[Y._off + a];
          else for (i = 0; i < Y.out; i++) v[Y._off + i * Y.inp + b] -= q[Y._off + i * Y.inp + a];
          gens.push(v); tags.push(pi);
        }
        pieces.push({ kind: dense ? 'GL' : 'T', k: kk, wire: 'o:' + X.id, from: X.label, to: Y.label });
      }
    });
    /* (1b) GL_k passing through square boxes F between two dense trainable boxes:
            Y F X = (Y F g^-1 F^-1) F (g X); infinitesimally dX = xi X, dY = -Y F xi F^-1 */
    forEachChain(C.root, function (ch) {
      for (var a = 0; a < ch.length; a++) {
        var X = ch[a];
        if (!isT(X) || X.st !== 'dense') continue;
        for (var b = a + 2; b < ch.length; b++) {
          var mid = ch.slice(a + 1, b);
          if (!mid.every(function (z) { return z.t === 'box' && z.out === z.inp; })) break;
          var Y = ch[b];
          if (!isT(Y) || Y.st !== 'dense') continue;
          var Fm = null; mid.forEach(function (z) { var Z = boxMat(z, q, C); Fm = Fm ? mul(Z, Fm) : Z; });
          var Fi = invM(Fm);
          if (!Fi) break;
          var kk = X.out, pi = pieces.length, YF = mul(boxMat(Y, q, C), Fm);
          for (var i1 = 0; i1 < kk; i1++) for (var j1 = 0; j1 < kk; j1++) {
            var v = vec(), j, p, c;
            for (j = 0; j < X.inp; j++) v[X._off + i1 * X.inp + j] += q[X._off + j1 * X.inp + j];
            for (p = 0; p < Y.out; p++) for (c = 0; c < kk; c++) v[Y._off + p * Y.inp + c] -= YF[p][i1] * Fi[j1][c];
            gens.push(v); tags.push(pi);
          }
          pieces.push({ kind: 'GL', k: kk, wire: 'o:' + X.id, through: mid.map(function (z) { return z.label; }).join('') });
          break;
        }
      }
    });
    /* (2) Hadamard spider tori: (X diag t) ⊙ (Y diag t^-1) = X ⊙ Y, likewise on the output side */
    forEachNode(C.root, function (nd) {
      if (nd.t !== 'had') return;
      var a0 = nd.a[0], b0 = nd.b[0], aL = nd.a[nd.a.length - 1], bL = nd.b[nd.b.length - 1], j, pi;
      if (isT(a0) && isT(b0)) {
        pi = pieces.length;
        for (j = 0; j < a0.inp; j++) { var v = vec(); colScale(v, a0, j, 1); colScale(v, b0, j, -1); gens.push(v); tags.push(pi); }
        pieces.push({ kind: 'T', k: a0.inp, spider: nd.id + ':in' });
      }
      if (isT(aL) && isT(bL)) {
        pi = pieces.length;
        for (j = 0; j < aL.out; j++) { var w = vec(); rowScale(w, aL, j, 1); rowScale(w, bL, j, -1); gens.push(w); tags.push(pi); }
        pieces.push({ kind: 'T', k: aL.out, spider: nd.id + ':out' });
      }
    });
    /* (3) scalar trades P -> cP, Q -> Q/c between trainable boxes that occur in exactly the same paths of
           the expansion and are not separated by sigma or the norm (both break multilinearity) */
    var memb = {};
    terms.forEach(function (t, ti) { termInfo(t).ids.forEach(function (id) { (memb[id] = memb[id] || []).push(ti); }); });
    var tb = C.train, pS = pieces.length, nS = 0;
    for (var x = 0; x < tb.length; x++) for (var y = x + 1; y < tb.length; y++) {
      var P1 = tb[x], P2 = tb[y];
      if (C.info[P1.id].scope !== C.info[P2.id].scope) continue;
      if (String(memb[P1.id] || '') !== String(memb[P2.id] || '') || !memb[P1.id]) continue;
      var u = vec(), k2;
      for (k2 = 0; k2 < P1._sz; k2++) u[P1._off + k2] += q[P1._off + k2];
      for (k2 = 0; k2 < P2._sz; k2++) u[P2._off + k2] -= q[P2._off + k2];
      gens.push(u); tags.push(pS); nS++;
    }
    if (nS) pieces.push({ kind: 'S', k: 1, n: nS });
    return { gens: gens, tags: tags, pieces: pieces };
  }

  /* =====================================================================================
     9. Full analysis of one plug
     ===================================================================================== */
  function randomPoint(P, salt) { var rd = Atlas.rng(90001 + 7919 * salt), q = new Float64Array(P); for (var i = 0; i < P; i++) q[i] = rd.normal(); return q; }
  function analyse(root, salt) {
    salt = salt || 0;
    var C = compile(root, salt), R = { C: C, errs: C.errs };
    if (C.errs.length) return R;
    var terms = expandNode(root);
    R.terms = terms;
    R.P = C.nP;
    R.deg = 0; terms.forEach(function (t) { R.deg = Math.max(R.deg, termInfo(t).nTr); });
    R.rank = rankBound(terms);
    R.linear = !C.sigma;
    R.norm = C.norm;
    var q = randomPoint(C.nP, salt);
    if (R.linear) {
      var dW = addM(evalMat(root, q, C), W0mat(), -1);
      R.attSv = svalsCols(colsOf(dW));
      R.att = R.attSv[0] > 1e-12 ? rankOf(R.attSv, 1e-9) : 0;
    }
    if (C.sigma) { C.K = Math.max(24, Math.ceil(2 * C.nP / M) + 8); C.X = gaussM(N, C.K, Atlas.rng(4242), 1); }
    var J = null;
    if (C.nP === 0) { R.d = 0; R.sv = []; }
    else if (C.nP > PMAX) { R.d = null; R.tooBig = true; }
    else { J = jacobianCols(C, q); R.sv = svalsCols(J); R.d = rankOf(R.sv, 1e-8); R.h = C.h; }
    /* gauge generators, each checked against the Jacobian */
    var G = gaugeGens(C, q, terms), keep = [], keepTags = [], bad = 0;
    if (J && G.gens.length) {
      var smax = R.sv[0] || 1;
      G.gens.forEach(function (v, gi) {
        var L = J[0].length, Jv = new Float64Array(L), nv = 0, i, k;
        for (k = 0; k < v.length; k++) { var c = v[k]; nv += c * c; if (c === 0) continue; var col = J[k]; for (i = 0; i < L; i++) Jv[i] += c * col[i]; }
        var nj = 0; for (i = 0; i < L; i++) nj += Jv[i] * Jv[i];
        if (Math.sqrt(nj) <= 1e-7 * smax * Math.sqrt(nv)) { keep.push(v); keepTags.push(G.tags[gi]); } else bad++;
      });
    } else if (!J) { keep = G.gens.slice(); keepTags = G.tags.slice(); }
    if (bad && typeof console !== 'undefined') console.warn('[workbench] ' + bad + ' gauge generator(s) failed the kernel check');
    R.badGens = bad; R.nGens = G.gens.length;
    /* orbit dimension, and what each generator group adds in order (wires, through-gauges, spiders, scalars) */
    var byTag = {}, acc = [], prev = 0;
    keep.forEach(function (v, i) { (byTag[keepTags[i]] = byTag[keepTags[i]] || []).push(v); });
    R.pieces = [];
    G.pieces.forEach(function (p, i) {
      var vs = byTag[i]; if (!vs) return;
      p.own = rankOf(svalsCols(vs), 1e-9);
      acc = acc.concat(vs);
      var now = rankOf(svalsCols(acc), 1e-9);
      p.add = now - prev; prev = now;
      if (p.kind === 'S') p.own = p.add;          /* scalar trades: count only what they add */
      if (p.add > 0) R.pieces.push(p);
    });
    R.orbit = prev;
    var hasGL = R.pieces.some(function (p) { return p.kind === 'GL'; }), hasT = R.pieces.some(function (p) { return p.kind === 'T'; });
    R.gt = hasGL && hasT ? 'GL×torus' : hasGL ? 'GL' : hasT ? 'torus' : R.orbit > 0 ? 'scalar' : 'none';
    R.fibre = R.d == null ? null : R.P - R.d;
    R.merge = C.sigma ? 'M∞' : 'M1';
    var fz = {}; C.boxes.forEach(function (b) { if (b.role === 'frozen') fz[b.st === 'diag' ? 'diag' : b.fk] = 1; });
    R.fz = Object.keys(fz).sort();
    R.hasW0 = !!fz.W0;
    R.key = keyNode(root);
    R.sig = { P: R.P, deg: R.deg, d: R.d, rb: R.rank.nonlin ? null : R.rank.bound, gt: R.gt, fz: R.fz.join(','), mg: R.merge };
    return R;
  }

  /* =====================================================================================
     10. Identification: structural keys, the preset table, nearest signature
     ===================================================================================== */
  function keyNode(nd) {
    switch (nd.t) {
      case 'box': return (nd.role === 'train' ? 'T' : 'F' + nd.fk + (nd.blk != null ? nd.blk : '')) + (nd.st === 'diag' ? 'd' : '') + nd.out + 'x' + nd.inp;
      case 'sig': return 's' + nd.out;
      case 'had': { var ka = keyChain(nd.a), kb = keyChain(nd.b); return 'H(' + (ka < kb ? ka + '|' + kb : kb + '|' + ka) + ')'; }
      case 'kron': return 'K(' + keyChain(nd.a) + '|' + keyChain(nd.b) + ')';
      case 'norm': return 'N(' + keyNode(nd.inner) + ')';
      case 'sum': return 'S(' + nd.lanes.map(keyChain).sort().join('+') + ')';
    }
    return '?';
  }
  function keyChain(ch) { return ch.map(keyNode).join('>'); }
  var TABLE = null;
  function buildTable() {
    if (TABLE) return TABLE;
    TABLE = [];
    PRESETS.concat(HIDDEN).forEach(function (p) {
      p.rs.forEach(function (r) { var root = p.build(r); TABLE.push({ p: p, r: r, root: root, key: keyNode(root), title: p.title(r), sig: null }); });
    });
    return TABLE;
  }
  function sigDist(a, b) {
    var s = 0;
    if (a.mg !== b.mg) s += 6;
    if (a.fz !== b.fz) s += 1.5;
    if (a.gt !== b.gt) s += 2;
    s += Math.abs(a.deg - b.deg);
    s += Math.abs((a.d == null ? 0 : a.d) - (b.d == null ? 0 : b.d)) / 8;
    s += Math.abs((a.rb == null ? 8 : a.rb) - (b.rb == null ? 8 : b.rb)) / 2;
    s += Math.abs(a.P - b.P) / 24;
    return s;
  }
  function fzTxt(s) { if (!s) return '∅'; return '{' + s.split(',').map(function (k) { return k === 'diag' ? 'diagonal' : (FK_SHORT[k] || k); }).filter(function (v, i, arr) { return arr.indexOf(v) === i; }).join(', ') + '}'; }
  function sigDiffs(a, b) {
    var out = [];
    if (a.mg !== b.mg) out.push('merge ' + a.mg + ' vs ' + b.mg);
    if (a.deg !== b.deg) out.push('degree ' + a.deg + ' vs ' + b.deg);
    if (a.d !== b.d) out.push('d ' + (a.d == null ? '—' : a.d) + ' vs ' + (b.d == null ? '—' : b.d));
    if (a.rb !== b.rb) out.push('rank ≤ ' + (a.rb == null ? 'n/a' : a.rb) + ' vs ≤ ' + (b.rb == null ? 'n/a' : b.rb));
    if (a.gt !== b.gt) out.push('gauge ' + a.gt + ' vs ' + b.gt);
    if (a.P !== b.P) out.push('|D| ' + a.P + ' vs ' + b.P);
    if (a.fz !== b.fz) out.push('frozen boxes ' + fzTxt(a.fz) + ' vs ' + fzTxt(b.fz));
    return out;
  }
  function sigEq(a, b) { return a.mg === b.mg && a.deg === b.deg && a.d === b.d && a.rb === b.rb && a.gt === b.gt && a.P === b.P && a.fz === b.fz; }
  function identify(R) {
    var T = buildTable(), i;
    for (i = 0; i < T.length; i++) if (T[i].key === R.key) return { kind: 'exact', e: T[i] };
    for (i = 0; i < T.length; i++) if (!T[i].sig) return { kind: 'pending' };
    for (i = 0; i < T.length; i++) if (sigEq(T[i].sig, R.sig)) return { kind: 'same', e: T[i] };
    var best = null, bd = Infinity;
    T.forEach(function (e) { var d = sigDist(R.sig, e.sig); if (d < bd) { bd = d; best = e; } });
    return { kind: 'near', e: best, diffs: sigDiffs(R.sig, best.sig) };
  }
  function isLoraLane(l) { return l.length === 2 && isT(l[0]) && isT(l[1]) && l[0].st === 'dense' && l[1].st === 'dense' && l[0].inp === N && l[1].out === M; }
  function loraLanes(root) { var n = 0; root.lanes.forEach(function (l) { if (isLoraLane(l)) n++; }); return n; }
  function concatLoRA(root) {
    var base = 0, ok = true, rs = [];
    root.lanes.forEach(function (l) { if (l.length === 1 && l[0].t === 'box' && l[0].role === 'frozen' && l[0].fk === 'W0') base++; else if (isLoraLane(l)) rs.push(l[0].out); else ok = false; });
    return ok && base === 1 && rs.length >= 2 ? rs : null;
  }

  /* =====================================================================================
     11. Edit operations (each returns a new root or an error string)
     ===================================================================================== */
  function find(root, id) {
    var res = null;
    function inChain(ch, owner) {
      for (var k = 0; k < ch.length && !res; k++) {
        var nd = ch[k];
        if (nd.id === id) { res = { chain: ch, idx: k, node: nd, owner: owner }; return; }
        inNode(nd, { chain: ch, idx: k, node: nd, owner: owner });
      }
    }
    function inNode(nd, ctx) {
      if (nd.t === 'had' || nd.t === 'kron') { inChain(nd.a, { type: nd.t, node: nd, part: 'a', ctx: ctx }); if (!res) inChain(nd.b, { type: nd.t, node: nd, part: 'b', ctx: ctx }); }
      else if (nd.t === 'norm') nd.inner.lanes.forEach(function (l, li) { if (!res) inChain(l, { type: 'norm', node: nd, part: li, ctx: ctx }); });
    }
    root.lanes.forEach(function (l, li) { if (!res) inChain(l, { type: 'root', part: li }); });
    return res;
  }
  function rootLaneOf(f) { var o = f.owner; while (o && o.type !== 'root') o = o.ctx.owner; return o ? o.part : -1; }
  function isBase(lane) { return lane.length === 1 && lane[0].t === 'box' && lane[0].role === 'frozen' && lane[0].fk === 'W0'; }
  function hasSig(ch) { return ch.some(function (x) { return x.t === 'sig'; }); }
  function plainLane(ch) { return ch.every(function (x) { return x.t === 'box' && !(x.role === 'frozen' && (x.fk === 'W0' || x.fk === 'blk' || x.fk === 'svdU' || x.fk === 'svdV')); }); }
  function labelsIn(root) { var s = {}; forEachNode(root, function (x) { if (x.label) s[x.label] = 1; }); return s; }
  function fresh(root, base) { var s = labelsIn(root); for (var k = 1; k < 99; k++) if (!s[base + sub(k)]) return base + sub(k); return base; }
  function freshPair(root) { var s = labelsIn(root); for (var k = 1; k < 99; k++) if (!s['A' + sub(k)] && !s['B' + sub(k)]) return ['A' + sub(k), 'B' + sub(k)]; return ['A', 'B']; }
  function dIn(x) { return x.t === 'box' || x.t === 'sig' ? x.inp : x._i; }
  function dOut(x) { return x.t === 'box' || x.t === 'sig' ? x.out : x._o; }
  function canResize(x) { if (x.t === 'sig') return true; if (x.t !== 'box') return false; return !(x.role === 'frozen' && (x.fk === 'W0' || x.fk === 'blk')); }
  function resizeNode(ch, idx, side, dim) {
    var x = ch[idx];
    if (!x || !canResize(x)) return false;
    var keepSq = x.t === 'sig' || x.st === 'diag';
    if (!keepSq) {
      if ((x.fk === 'svdU' && side === 'out') || (x.fk === 'svdV' && side === 'in')) return false;
      if (side === 'in') x.inp = dim; else x.out = dim;
      return true;
    }
    x.inp = x.out = dim;
    if (side === 'in') return idx + 1 < ch.length && resizeNode(ch, idx + 1, 'in', dim);
    return idx - 1 >= 0 && resizeNode(ch, idx - 1, 'out', dim);
  }
  function setWire(ch, k, dim) { return resizeNode(ch, k - 1, 'out', dim) && resizeNode(ch, k, 'in', dim); }
  function relabelFrozen(b) {
    if (b.fk === 'W0') b.label = 'W₀';
    else if (b.fk === 'svdU') b.label = 'U';
    else if (b.fk === 'svdV') b.label = 'ΣVᵀ';
    else if (b.fk === 'sum') b.label = 'S';
    else if (b.fk === 'eye') b.label = 'I';
    else if (!/₀$/.test(b.label)) b.label = b.label.replace(/[₁-₉]+$/, '') + '₀';
  }
  var OPS = {
    role: function (R, id) {
      var f = find(R, id), b = f && f.node;
      if (!b || b.t !== 'box') return 'Select a box first.';
      if (b.role === 'train') { b.role = 'frozen'; b.fk = 'rand'; relabelFrozen(b); }
      else { b.role = 'train'; b.fk = null; delete b.blk; b.label = b.label === 'W₀' ? 'W' : b.label.replace(/₀$/, '') || 'T'; if (b.label === 'ΣVᵀ' || b.label === 'S' || b.label === 'I' || b.label === 'U') b.label = fresh(R, 'T'); }
      return R;
    },
    diag: function (R, id) {
      var f = find(R, id), b = f && f.node;
      if (!b || b.t !== 'box') return 'Select a box first.';
      if (b.out !== b.inp) return 'Only a square box can become a diagonal spider.';
      if (b.st === 'diag') { b.st = 'dense'; } else { b.st = 'diag'; if (b.role === 'frozen') { b.fk = 'rand'; delete b.blk; relabelFrozen(b); } }
      return R;
    },
    kind: function (R, id, fk) {
      var f = find(R, id), b = f && f.node;
      if (!b || b.t !== 'box' || b.role !== 'frozen') return 'Select a frozen box first.';
      if (b.st === 'diag' && fk !== 'rand') return 'A frozen diagonal spider holds a random diagonal.';
      b.fk = fk; delete b.blk; relabelFrozen(b);
      return R;
    },
    wire: function (R, id, delta) {
      var f = find(R, id);
      if (!f) return 'Select a box first.';
      if (f.idx >= f.chain.length - 1) return 'This box’s output is the boundary of its path; its size is fixed.';
      var cur = f.node.out, d = cur + delta;
      if (d < 1 || d > 8) return 'Internal wires range from 1 to 8.';
      if (!setWire(f.chain, f.idx + 1, d)) return 'That wire cannot change size, because a fixed box (W₀ or a frame) sits on it.';
      return R;
    },
    insert: function (R, id, what) {
      var f = find(R, id);
      if (!f) return 'Select a box first.';
      var x = f.node, dim = x.out, nn;
      if (what === 'sig') {
        if (f.owner.type !== 'root') return 'σ belongs on an activation path at the top level, not inside ⊙, ⊗ or a norm.';
        if (isBase(f.chain)) return 'The base path is θ₀ itself; put σ on an update path.';
        nn = SG(dim);
      } else if (what === 'diag') nn = TD(fresh(R, 'ℓ'), dim);
      else nn = TB(fresh(R, 'T'), dim, dim);
      f.chain.splice(f.idx + 1, 0, nn);
      return R;
    },
    remove: function (R, id) {
      var f = find(R, id);
      if (!f) return 'Select a box first.';
      var ch = f.chain, i = f.idx, X = ch[i], ok = true;
      if (ch.length === 1) return removeChain(R, f);
      ch.splice(i, 1);
      if (i === 0) ok = dIn(ch[0]) === dIn(X) || resizeNode(ch, 0, 'in', dIn(X));
      else if (i === ch.length) ok = dOut(ch[i - 1]) === dOut(X) || resizeNode(ch, i - 1, 'out', dOut(X));
      else if (dOut(ch[i - 1]) !== dIn(ch[i])) ok = setWire(ch, i, Math.min(dIn(X), dOut(X)));
      if (!ok) return 'Removing ' + X.label + ' would leave wires that do not meet.';
      return R;
    },
    hadNew: function (R, id, r) {
      var f = find(R, id); if (!f) return 'Select a box first.';
      var li = rootLaneOf(f), lane = R.lanes[li];
      if (isBase(lane)) return 'The base path is θ₀; choose a box on an update path.';
      if (hasSig(lane)) return 'A Hadamard diamond needs a linear path (no σ).';
      var nm = freshPair(R);
      R.lanes[li] = [HAD(lane, [TB(nm[0], r, N), TB(nm[1], M, r)])];
      return R;
    },
    hadW0: function (R, id) {
      var f = find(R, id); if (!f) return 'Select a box first.';
      var li = rootLaneOf(f), lane = R.lanes[li];
      if (isBase(lane)) return 'The base path is θ₀; choose a box on an update path.';
      if (hasSig(lane)) return 'A Hadamard diamond needs a linear path (no σ).';
      R.lanes[li] = [HAD([W0B()], lane)];
      return R;
    },
    kron: function (R, id) {
      var f = find(R, id); if (!f) return 'Select a box first.';
      var li = rootLaneOf(f), lane = R.lanes[li];
      if (isBase(lane)) return 'The base path is θ₀; choose a box on an update path.';
      if (!plainLane(lane)) return 'A Kronecker split needs a path of plain boxes (no W₀, frames, σ, ⊙ or ⊗).';
      var L = clone(lane);
      L.forEach(function (b) { b.inp = b.inp === 8 ? 4 : Math.min(b.inp, 4); b.out = b.out === 8 ? 4 : Math.min(b.out, 4); b.id = nid('b'); });
      R.lanes[li] = [KRON([TB(fresh(R, 'C'), 2, 2)], L)];
      return R;
    },
    addPath: function (R, id, r) { var nm = freshPair(R); R.lanes.push([TB(nm[0], r, N), TB(nm[1], M, r)]); return R; },
    removePath: function (R, id) {
      var f = find(R, id); if (!f) return 'Select a box first.';
      if (R.lanes.length < 2) return 'A plug needs at least one path.';
      R.lanes.splice(rootLaneOf(f), 1);
      return R;
    }
  };
  function removeChain(R, f) {
    var o = f.owner;
    if (o.type === 'root') { if (R.lanes.length < 2) return 'A plug needs at least one path.'; R.lanes.splice(o.part, 1); return R; }
    if (o.type === 'had') { var other = o.part === 'a' ? o.node.b : o.node.a; var c = o.ctx; c.chain.splice.apply(c.chain, [c.idx, 1].concat(other)); return R; }
    if (o.type === 'kron') return 'Removing a whole Kronecker factor would change the wire sizes; remove the ⊗ path instead.';
    if (o.type === 'norm') {
      o.node.inner.lanes.splice(o.part, 1);
      if (!o.node.inner.lanes.length) { var cc = o.ctx; if (cc.chain.length === 1) return removeChain(R, cc); cc.chain.splice(cc.idx, 1); }
      return R;
    }
    return 'Cannot remove that.';
  }

  /* =====================================================================================
     12. Formulas (TeX for MathJax, Unicode fallback)
     ===================================================================================== */
  var GREEK = { 'Σ': '\\Sigma ', 'Δ': '\\Delta ', 'ℓ': '\\ell ', 'σ': '\\sigma ', 'λ': '\\lambda ', 'μ': '\\mu ' };
  function texLabel(lbl) {
    var out = '', subs = '', i, c;
    for (i = 0; i < lbl.length; i++) {
      c = lbl.charAt(i);
      var si = SUBS.indexOf(c);
      if (si >= 0) { subs += si; continue; }
      if (subs) { out += '_{' + subs + '}'; subs = ''; }
      if (c === 'ᵀ') out += '^{\\top}';
      else if (c === '′') out += "'";
      else out += GREEK[c] || c;
    }
    if (subs) out += '_{' + subs + '}';
    return out;
  }
  function fmlNode(x, inProd, tex) {
    var L = tex ? texLabel : function (s) { return s; };
    switch (x.t) {
      case 'box': return x.st === 'diag' ? (tex ? '\\operatorname{diag}(' + L(x.label) + ')' : 'diag(' + x.label + ')') : L(x.label);
      case 'had': { var s = fmlChain(x.a, tex) + (tex ? ' \\odot ' : ' ⊙ ') + fmlChain(x.b, tex); return inProd ? '(' + s + ')' : s; }
      case 'kron': { var s2 = fmlChain(x.a, tex) + (tex ? ' \\otimes ' : ' ⊗ ') + fmlChain(x.b, tex); return inProd ? '(' + s2 + ')' : s2; }
      case 'norm': return (tex ? '\\mathcal N' : 'N') + '(' + x.inner.lanes.map(function (l) { return fmlChain(l, tex); }).join(' + ') + ')';
      case 'sig': return tex ? '\\sigma' : 'σ';
    }
    return '';
  }
  function fmlChain(ch, tex) {
    var parts = [];
    for (var k = ch.length - 1; k >= 0; k--) parts.push(fmlNode(ch[k], ch.length > 1, tex));
    return parts.join(tex ? '\\,' : '');
  }
  function fmlApply(ch, tex) {
    var s = 'x';
    for (var k = 0; k < ch.length; k++) {
      var x = ch[k];
      if (x.t === 'sig') s = (tex ? '\\sigma' : 'σ') + '(' + s + ')';
      else s = fmlNode(x, true, tex) + (tex ? '\\,' : ' ') + s;
    }
    return s;
  }
  /* the formula as one string per term, so that a long formal sum can wrap between terms */
  function formulaParts(root, tex) {
    var sigma = false; forEachNode(root, function (x) { if (x.t === 'sig') sigma = true; });
    var terms = root.lanes.map(function (l) { return sigma ? fmlApply(l, tex) : fmlChain(l, tex); });
    return terms.map(function (t, i) { return i ? (tex ? '{}+{} ' : '+ ') + t : (sigma ? 'y = ' : 'W = ') + t; });
  }
  function formula(root, tex) { return formulaParts(root, tex).join(' '); }

  /* expose the engine for headless tests */
  Atlas._workbench = { PRESETS: PRESETS, HIDDEN: HIDDEN, analyse: analyse, identify: identify, buildTable: buildTable, OPS: OPS, clone: clone, find: find, formula: formula, keyNode: keyNode, svalsCols: svalsCols, minCut: minCut, expandNode: expandNode };

  /* =====================================================================================
     13. CSS
     ===================================================================================== */
  var F = '[data-figure="workbench"] ';
  var CSS = [
    F + '.wb{padding:1.15rem 1.2rem 1.1rem}',
    F + '.wb-title-row{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.2rem 1rem}',
    F + '.wb-title{font-family:var(--f-display);font-size:clamp(1.5rem,3.6vw,1.95rem);font-weight:var(--w-head);letter-spacing:-.01em;line-height:1.08;margin:0}',
    F + 'sub{font-size:.68em;line-height:0;vertical-align:-.22em}',
    F + 'sup{font-size:.68em;line-height:0;vertical-align:.42em}',
    F + '.nt{text-transform:none;letter-spacing:0}',
    F + '.wb-tag{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide)}',
    F + '.wb-dek{margin:.25rem 0 .55rem;font-family:var(--f-body);font-style:italic;font-size:1.1rem;line-height:1.35;color:var(--ink-2)}',
    F + '.wb-instr{margin:0;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:48rem}',
    F + '.wb-instr b{font-weight:600;color:var(--ink)}',
    F + '.wb-controls{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:.7rem;margin:1rem 0 .7rem}',
    F + '.wb.is-narrow .wb-controls{grid-template-columns:minmax(0,1fr)}',
    /* phones: the diagram comes straight after the presets, the editor after the legend */
    F + '.wb.is-narrow{display:flex;flex-direction:column}',
    F + '.wb.is-narrow>.wb-head{order:0}', F + '.wb.is-narrow>.wb-controls{order:1}', F + '.wb.is-narrow>.wb-formula{order:2}', F + '.wb.is-narrow>.wb-dia{order:3}',
    F + '.wb.is-narrow>.wb-legend{order:4}', F + '.wb.is-narrow>.wb-edit{order:5}', F + '.wb.is-narrow>.wb-tiles{order:6}', F + '.wb.is-narrow>.wb-foot{order:7}',
    F + '.wb-group{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:.6rem .75rem .7rem;min-width:0;display:grid;gap:.45rem;align-content:start}',
    F + '.wb-glabel{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    F + '.wb-chips{display:flex;flex-wrap:wrap;gap:.3rem}',
    F + '.wb-chip{font-family:var(--f-ui);font-weight:500;font-size:.8rem;padding:.24rem .55rem;border:1px solid var(--rule);background:var(--paper-2);color:var(--ink-2);border-radius:4px;cursor:pointer;line-height:1.3}',
    F + '.wb-chip:hover{border-color:var(--tide);color:var(--ink)}',
    F + '.wb-chip[aria-pressed="true"]{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
    F + '.wb-step{display:inline-flex;align-items:stretch;border:1px solid var(--rule);border-radius:4px;overflow:hidden;background:var(--paper-2);width:max-content}',
    F + '.wb-step button{font-family:var(--f-ui);font-weight:500;font-size:.95rem;width:2rem;border:0;background:transparent;color:var(--ink);cursor:pointer}',
    F + '.wb-step button:hover:not(:disabled){background:var(--tide-soft)}',
    F + '.wb-step button:disabled{color:var(--ink-3);cursor:not-allowed;opacity:.5}',
    F + '.wb-step output{font-family:var(--f-mono);font-variant-numeric:tabular-nums;min-width:3.4rem;text-align:center;padding:.28rem .3rem;border-left:1px solid var(--rule);border-right:1px solid var(--rule);color:var(--ink);font-size:.84rem}',
    F + '.wb-help{font-family:var(--f-body);font-size:.8rem;line-height:1.4;color:var(--ink-2);max-width:20rem}',
    F + '.wb-edit{margin:0 0 .9rem}',
    F + '.wb-edit-top{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .7rem}',
    F + '.wb-edit-top select{font-family:var(--f-ui);font-size:.8rem;padding:.25rem .4rem;max-width:100%}',
    F + '.wb-desc{font-family:var(--f-body);font-size:.84rem;color:var(--ink-2);line-height:1.4}',
    F + '.wb-rows{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.35rem .7rem;align-items:center}',
    F + '.wb.is-narrow .wb-rows{grid-template-columns:minmax(0,1fr)}',
    F + '.wb-rl{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    F + '.wb-btns{display:flex;flex-wrap:wrap;gap:.3rem;align-items:center}',
    F + '.wb-btn{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);padding:.26rem .55rem;border:1px solid var(--rule);background:var(--paper-2);color:var(--ink);border-radius:4px;cursor:pointer;white-space:nowrap;line-height:1.3}',
    F + '.wb-btn:hover:not(:disabled){border-color:var(--tide);color:var(--tide)}',
    F + '.wb-btn:disabled{opacity:.4;cursor:not-allowed}',
    F + '.wb-btn.ghost{background:transparent}',
    F + '.wb-kind{font-family:var(--f-ui);font-size:var(--fs-xs);padding:.22rem .35rem}',
    F + '.wb-wire{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);color:var(--ink-2)}',
    F + '.wb-msg{font-family:var(--f-ui);font-size:.8rem;color:var(--seal);min-height:1.1em}',
    F + '.wb-msg.ok{color:var(--ink-2)}',
    F + '.wb-formula{font-family:var(--f-display);font-size:1.22rem;margin:.2rem 0 .45rem;color:var(--ink);overflow-x:auto;overflow-y:hidden;padding:.1rem 0;display:flex;flex-wrap:wrap;justify-content:center;justify-content:safe center;align-items:baseline;row-gap:.15rem}',
    F + '.wb-fpart{white-space:nowrap;max-width:100%}',
    F + '.wb-formula.plain .wb-fpart + .wb-fpart{margin-left:.3em}',
    F + '.wb.is-narrow .wb-formula{font-size:1.08rem}',
    F + '.wb-formula mjx-container{margin:0 !important}',
    F + '.wb-dia{background:var(--paper);border:1px solid var(--rule);border-radius:6px;overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch}',
    F + '.wb-dia svg{display:block;margin:0 auto;max-width:none}',
    F + '.wb-dia svg text{font-family:var(--f-ui);fill:var(--ink-2)}',
    F + '.wb-dia svg text.mono{font-family:var(--f-mono);font-variant-numeric:tabular-nums}',
    F + '.wb-dia svg text.serif{font-family:var(--f-body)}',
    F + '.wb-dia svg text.ui{font-weight:500}',
    F + '.wb-node{cursor:pointer;outline:none}',
    F + '.wb-node:focus-visible .wb-shape,.wb-node.is-sel .wb-shape{stroke:var(--ochre) !important;stroke-width:2.6px !important}',
    F + '.wb-node:hover .wb-shape{stroke-width:2px}',
    F + '.wb-legend{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;font-family:var(--f-ui);font-size:var(--fs-xs);color:var(--ink-2);margin:.55rem 0 1rem;align-items:center}',
    F + '.wb-legend span{display:inline-flex;align-items:center;gap:.35rem}',
    F + '.wb-legend svg{flex:none}',
    F + '.wb-tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem}',
    F + '.wb.is-mid .wb-tiles{grid-template-columns:repeat(2,minmax(0,1fr))}',
    F + '.wb.is-narrow .wb-tiles{grid-template-columns:minmax(0,1fr)}',
    F + '.wb-tile{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:.6rem .75rem .65rem;min-width:0;display:grid;gap:.25rem;align-content:start}',
    F + '.wb-tile.span2{grid-column:span 2}',
    F + '.wb.is-narrow .wb-tile.span2{grid-column:auto}',
    F + '.wb-k{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);line-height:1.35}',
    F + '.wb-v{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:1.6rem;color:var(--ink);line-height:1.1;letter-spacing:-.01em}',
    F + '.wb-v small{font-size:.8rem;color:var(--ink-2);letter-spacing:0;margin-left:.45rem}',
    F + '.wb-s{font-family:var(--f-body);font-size:.84rem;color:var(--ink-2);line-height:1.42}',
    F + '.wb-s.mono,' + F + '.wb-mono{font-family:var(--f-mono);font-size:.78rem;font-variant-numeric:tabular-nums}',
    F + '.wb-ok{color:var(--moss)}',
    F + '.wb-bad{color:var(--seal)}',
    F + '.wb-hi{color:var(--ochre)}',
    F + '.wb-chk{font-family:var(--f-body);font-size:.84rem;font-variant-numeric:tabular-nums;line-height:1.4;color:var(--ink-2);border-top:1px solid var(--rule);padding-top:.3rem;margin-top:.15rem}',
    F + '.wb-badge{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.04em;padding:.12rem .5rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap;width:max-content}',
    F + '.wb-badge.yes{color:var(--moss);background:var(--moss-soft)}',
    F + '.wb-badge.no{color:var(--seal);background:var(--seal-soft)}',
    F + '.wb-badge.cond{color:var(--ochre);background:var(--ochre-soft)}',
    F + '.wb-idname{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.45rem;line-height:1.15;color:var(--ink)}',
    F + '.wb-idpar{font-family:var(--f-body);font-weight:var(--w-body);font-size:.72em;color:var(--ink-2);letter-spacing:0}',
    F + '.wb-bars{display:grid;grid-template-columns:max-content minmax(0,1fr) max-content;gap:.25rem .5rem;align-items:center;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);color:var(--ink-2);font-variant-numeric:tabular-nums}',
    F + '.wb-bars > span:nth-child(3n){font-family:var(--f-mono);font-weight:400}',
    F + '.wb-bar{height:10px;border-radius:2px;background:var(--paper-3);position:relative;overflow:hidden}',
    F + '.wb-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:2px}',
    F + '.wb-spark{display:block;width:100%;height:auto}',
    F + '.wb-spark text{font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:var(--ink-2)}',
    F + '.wb-foot{margin-top:.8rem;font-family:var(--f-body);font-size:.8rem;line-height:1.5;color:var(--ink-2);display:flex;flex-wrap:wrap;gap:.4rem 1rem;align-items:baseline;justify-content:space-between}',
    F + '.wb-foot p{margin:0;max-width:52rem}',
    F + '.wb-pulse{animation:wbPulse .5s ease-out}',
    '@keyframes wbPulse{from{background:var(--ochre-soft)}to{background:var(--paper)}}'
  ].join('\n');
  function injectCss() {
    if (document.getElementById('css-workbench')) return;
    var s = document.createElement('style'); s.id = 'css-workbench'; s.textContent = CSS; document.head.appendChild(s);
  }

  /* =====================================================================================
     14. Diagram layout and SVG
     ===================================================================================== */
  var NS = 'http://www.w3.org/2000/svg';
  function S(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  var GEO = { bw: 58, bh: 34, dw: 34, sw: 40, wl: 40, pad: 22, vgap: 20, lane: 24, port: 44, fan: 30, top: 50, bot: 34, kcap: 11.4 };
  function measure(x, MS) {
    var m;
    switch (x.t) {
      case 'box': m = x.st === 'diag' ? { w: GEO.dw, h: GEO.bh + 8 } : { w: GEO.bw, h: GEO.bh + 12 }; break;
      case 'sig': m = { w: GEO.sw, h: GEO.bh }; break;
      case 'had': case 'kron': {
        var a = measureChain(x.a, MS), b = measureChain(x.b, MS), pad = x.t === 'kron' ? GEO.pad + 8 : GEO.pad;
        /* a Kronecker block keeps a line below its lower branch for its R^n = R^a ⊗ R^b caption */
        m = { w: Math.max(a.w, b.w) + 2 * pad, h: a.h + GEO.vgap + b.h + (x.t === 'kron' ? GEO.kcap : 0), a: a, b: b, pad: pad }; break;
      }
      case 'norm': { var s = measureSum(x.inner, MS); m = { w: s.w + 20, h: s.h + 30, s: s }; break; }
      default: m = { w: 40, h: 30 };
    }
    MS[x.id] = m;
    return m;
  }
  function measureChain(ch, MS) { var w = GEO.wl, h = 0; ch.forEach(function (x) { var m = measure(x, MS); w += m.w + GEO.wl; h = Math.max(h, m.h); }); return { w: w, h: h }; }
  function measureSum(sm, MS) {
    var w = 0, h = 0, ls = sm.lanes.map(function (l) { var m = measureChain(l, MS); w = Math.max(w, m.w); return m; });
    ls.forEach(function (m, i) { h += m.h + (i ? GEO.lane : 0); });
    var r = { w: w + 2 * GEO.pad + 16, h: h, lanes: ls }; MS[sm.id] = r; return r;
  }

  /* SVG text with real subscripts (digits) and a superscript T, positioned in px for font size fs */
  function richText(t, label, fs) {
    var segs = [], cur = '', mode = null, i;
    for (i = 0; i < label.length; i++) {
      var c = label.charAt(i), si = SUBS.indexOf(c), m = si >= 0 ? 's' : c === 'ᵀ' ? 'p' : 'n';
      var ch = si >= 0 ? String(si) : c === 'ᵀ' ? 'T' : c;
      if (m !== mode && cur) { segs.push([mode, cur]); cur = ''; }
      mode = m; cur += ch;
    }
    if (cur) segs.push([mode, cur]);
    var shift = 0;
    segs.forEach(function (sg) {
      var want = sg[0] === 's' ? 0.3 * fs : sg[0] === 'p' ? -0.42 * fs : 0;
      var a = { dy: (want - shift).toFixed(2) };
      if (sg[0] !== 'n') a['font-size'] = (0.66 * fs).toFixed(1) + 'px';
      S('tspan', a, t, sg[1]);
      shift = want;
    });
    return t;
  }

  Atlas.register('workbench', function (el, A) {
    injectCss();
    var UID = 'wb' + Math.random().toString(36).slice(2, 7);
    var state = { preset: 'loha', r: 2, root: null, sel: null, hist: [], salt: 0 };
    var R = null;           /* the current analysis */

    /* ---------------- skeleton ---------------- */
    var stage = A.h('div', { class: 'stage wb', role: 'group', 'aria-label': 'Plug workbench: build a parameter-efficient fine-tuning method as a tensor network and read off its invariants' });
    el.appendChild(stage);
    stage.appendChild(A.h('div', { class: 'wb-head' }, [
      A.h('div', { class: 'wb-title-row' }, [
        A.h('h3', { class: 'wb-title', text: 'The plug workbench' }),
        A.h('span', { class: 'wb-tag', text: 'Def I.5 · Thm II.10 · Prop II.12 · Thm VI.3' })
      ]),
      A.h('p', { class: 'wb-dek', html: 'Cut <i>W</i><sub>0</sub> out of the layer and a hole <i>V</i> → <i>U</i> remains. A method is the plug you put back, a small tensor network of frozen and trainable boxes.' }),
      A.h('p', { class: 'wb-instr', html: 'Pick a preset, then <b>click any box</b> in the diagram (or choose it from the list) and change the plug with the edit buttons. Every number below is recomputed from the diagram you see, at <b>m = n = 8</b>.' })
    ]));

    /* presets + r */
    var controls = A.h('div', { class: 'wb-controls' });
    stage.appendChild(controls);
    var chipWrap = A.h('div', { class: 'wb-chips', role: 'group', 'aria-label': 'Preset method' });
    var chips = {};
    PRESETS.forEach(function (p) {
      var b = A.h('button', { type: 'button', class: 'wb-chip', 'aria-pressed': 'false', text: p.name });
      b.addEventListener('click', function () { loadPreset(p.id); });
      chips[p.id] = b; chipWrap.appendChild(b);
    });
    controls.appendChild(A.h('div', { class: 'wb-group' }, [A.h('div', { class: 'wb-glabel', text: 'Preset' }), chipWrap]));
    var rMinus = A.h('button', { type: 'button', 'aria-label': 'Decrease the rank r' }, ['−']);
    var rOut = A.h('output', { 'aria-live': 'polite' });
    var rPlus = A.h('button', { type: 'button', 'aria-label': 'Increase the rank r' }, ['+']);
    var rHelp = A.h('div', { class: 'wb-help' });
    rMinus.addEventListener('click', function () { stepR(-1); });
    rPlus.addEventListener('click', function () { stepR(1); });
    controls.appendChild(A.h('div', { class: 'wb-group' }, [A.h('div', { class: 'wb-glabel', html: 'Rank <span class="nt"><i>r</i></span> of the preset' }), A.h('div', { class: 'wb-step' }, [rMinus, rOut, rPlus]), rHelp]));

    /* editor */
    var edit = A.h('div', { class: 'wb-group wb-edit' });
    stage.appendChild(edit);
    var selSelect = A.h('select', { 'aria-label': 'Selected box' });
    selSelect.addEventListener('change', function () { state.sel = selSelect.value; renderDiagram(); renderEditor(); });
    var selDesc = A.h('span', { class: 'wb-desc' });
    edit.appendChild(A.h('div', { class: 'wb-edit-top' }, [A.h('span', { class: 'wb-glabel', text: 'Edit the plug · selected' }), selSelect, selDesc]));
    function btn(label, aria, fn) { var b = A.h('button', { type: 'button', class: 'wb-btn', 'aria-label': aria || label, text: label }); b.addEventListener('click', fn); return b; }
    var bRole = btn('trainable ⇄ frozen', 'Toggle the selected box between trainable and frozen', function () { apply('role'); });
    var bDiag = btn('dense ⇄ diagonal', 'Toggle the selected box between dense and diagonal spider', function () { apply('diag'); });
    var kindSel = A.h('select', { class: 'wb-kind', 'aria-label': 'Kind of frozen box' });
    ['W0', 'rand', 'svdU', 'svdV', 'sum', 'eye'].forEach(function (k) { kindSel.appendChild(A.h('option', { value: k, text: 'frozen: ' + FK_SHORT[k] + (k === 'svdU' ? ' (U)' : k === 'svdV' ? ' (ΣVᵀ)' : '') })); });
    kindSel.addEventListener('change', function () { apply('kind', kindSel.value); });
    var wMinus = btn('−', 'Shrink the wire leaving the selected box', function () { apply('wire', -1); });
    var wOut = A.h('output', { class: 'wb-mono' });
    var wPlus = btn('+', 'Grow the wire leaving the selected box', function () { apply('wire', 1); });
    var bRemove = btn('remove', 'Remove the selected box', function () { apply('remove'); });
    var bInsBox = btn('+ trainable box', 'Insert a trainable box after the selected box', function () { apply('insert', 'box'); });
    var bInsDiag = btn('+ diagonal spider', 'Insert a trainable diagonal spider after the selected box', function () { apply('insert', 'diag'); });
    var bInsSig = btn('+ σ', 'Insert a sigma box after the selected box', function () { apply('insert', 'sig'); });
    var bHadNew = btn('⊙ new LoRA branch', 'Hadamard the selected path with a new rank r branch', function () { apply('hadNew', state.r || 2); });
    var bHadW0 = btn('⊙ W₀', 'Hadamard the selected path with the frozen W0', function () { apply('hadW0'); });
    var bKron = btn('⊗ Kronecker split', 'Reshape the selected path as a Kronecker product on R2 tensor R4', function () { apply('kron'); });
    var bAddPath = btn('+ LoRA path', 'Add a new rank r LoRA path to the formal sum', function () { apply('addPath', state.r || 2); });
    var bRemPath = btn('− path', 'Remove the path containing the selected box', function () { apply('removePath'); });
    var bUndo = btn('undo', 'Undo the last edit', function () { undo(); }); bUndo.classList.add('ghost');
    var bReset = btn('reset preset', 'Reset to the preset', function () { loadPreset(state.preset); }); bReset.classList.add('ghost');
    var msg = A.h('div', { class: 'wb-msg ok', 'aria-live': 'polite' });
    edit.appendChild(A.h('div', { class: 'wb-rows' }, [
      A.h('span', { class: 'wb-rl', text: 'Box' }), A.h('div', { class: 'wb-btns' }, [bRole, bDiag, kindSel, A.h('span', { class: 'wb-wire' }, ['out-wire', wMinus, wOut, wPlus]), bRemove]),
      A.h('span', { class: 'wb-rl', text: 'Insert after' }), A.h('div', { class: 'wb-btns' }, [bInsBox, bInsDiag, bInsSig]),
      A.h('span', { class: 'wb-rl', text: 'Path' }), A.h('div', { class: 'wb-btns' }, [bHadNew, bHadW0, bKron, bAddPath, bRemPath, bUndo, bReset])
    ]));
    edit.appendChild(msg);

    /* formula, diagram, legend */
    var fml = A.h('div', { class: 'wb-formula', 'aria-live': 'polite' });
    stage.appendChild(fml);
    var dia = A.h('div', { class: 'wb-dia' });
    stage.appendChild(dia);
    var legend = A.h('div', { class: 'wb-legend', 'aria-label': 'Legend' });
    stage.appendChild(legend);

    /* readouts */
    var tiles = A.h('div', { class: 'wb-tiles' });
    stage.appendChild(tiles);
    function tile(cls) { var t = A.h('div', { class: 'wb-tile ' + (cls || '') }); tiles.appendChild(t); return t; }
    var tP = tile(), tDeg = tile(), tRank = tile(), tMerge = tile(), tD = tile('span2'), tGauge = tile('span2'), tId = tile('span2'), tCmp = tile('span2');
    var foot = A.h('div', { class: 'wb-foot' });
    stage.appendChild(foot);
    var footP = A.h('p');
    var bPoint = btn('new random point', 'Draw a new random point q, and new random frozen frames, then recompute', function () { state.salt += 1; recompute(true); });
    foot.appendChild(footP); foot.appendChild(bPoint);

    /* ---------------- state transitions ---------------- */
    function presetOf() { return presetById(state.preset); }
    function validR(p, r) { if (p.rs.indexOf(r) >= 0) return r; if (p.rs[0] === 0) return 0; var best = p.rs[0]; p.rs.forEach(function (x) { if (Math.abs(x - r) < Math.abs(best - r)) best = x; }); return best; }
    function firstTrain(root) { var id = null; forEachNode(root, function (x) { if (!id && x.t === 'box' && x.role === 'train') id = x.id; }); return id; }
    function loadPreset(id) {
      state.preset = id;
      var p = presetOf();
      var want = state.rWanted || state.r || 2;
      state.r = validR(p, want);
      state.root = p.build(state.r);
      state.sel = firstTrain(state.root);
      state.hist = [];
      setMsg('', true);
      renderAll();
    }
    function stepR(dir) {
      var p = presetOf(); if (p.rs[0] === 0) return;
      var i = p.rs.indexOf(state.r), j = Math.max(0, Math.min(p.rs.length - 1, i + dir));
      if (j === i) return;
      state.r = p.rs[j]; state.rWanted = state.r;
      state.root = p.build(state.r);
      state.sel = firstTrain(state.root);
      state.hist = [];
      setMsg('', true);
      renderAll();
    }
    function setMsg(s, ok) { msg.textContent = s || ''; msg.className = 'wb-msg' + (ok ? ' ok' : ''); }
    function apply(op, arg) {
      if (!state.sel) { setMsg('Select a box in the diagram first.'); return; }
      var next = OPS[op](clone(state.root), state.sel, arg);
      if (typeof next === 'string') { setMsg(next); return; }
      var C = compile(next, 0);
      if (C.errs.length) { setMsg(C.errs[0]); return; }
      if (C.nP > PMAX) { setMsg('That plug would have ' + C.nP + ' trainable entries; the live Jacobian is capped at ' + PMAX + '.'); return; }
      state.hist.push({ root: state.root, sel: state.sel });
      if (state.hist.length > 60) state.hist.shift();
      state.root = next;
      if (!find(next, state.sel)) state.sel = firstTrain(next) || (next.lanes[0][0] && next.lanes[0][0].id);
      setMsg('', true);
      renderAll();
    }
    function undo() {
      var h = state.hist.pop(); if (!h) { setMsg('Nothing to undo.'); return; }
      state.root = h.root; state.sel = h.sel; setMsg('', true); renderAll();
    }

    /* ---------------- compute ---------------- */
    function recompute(pulse) {
      var t0 = (window.performance && performance.now) ? performance.now() : 0;
      R = analyse(state.root, state.salt);
      R.ms = ((window.performance && performance.now) ? performance.now() : 0) - t0;
      lastMs = R.ms;
      if (!R.errs.length) {
        /* attach each path of the expansion to the top-level lane it lives in */
        var laneOf = {};
        state.root.lanes.forEach(function (l, li) { forEachChain({ lanes: [l] }, function (c) { c.forEach(function (x) { laneOf[x.id] = li; }); }); });
        R.rank.parts.forEach(function (pt) { var t = R.terms[pt.ti], lid = null; walkPure(t, function (x) { if (lid == null && laneOf[x.id] != null) lid = laneOf[x.id]; }); pt.lane = lid; });
        R.id = identify(R);
      }
      renderDiagram();
      renderReadouts(pulse);
    }
    var recomputeSoon = A.debounce(function () { recompute(true); }, 40);

    /* ---------------- rendering: controls ---------------- */
    function renderControls() {
      var p = presetOf();
      Object.keys(chips).forEach(function (k) { chips[k].setAttribute('aria-pressed', String(k === state.preset)); });
      var noR = p.rs[0] === 0;
      rOut.textContent = noR ? 'fixed' : 'r = ' + state.r;
      rMinus.disabled = noR || p.rs.indexOf(state.r) <= 0;
      rPlus.disabled = noR || p.rs.indexOf(state.r) >= p.rs.length - 1;
      rHelp.textContent = noR ? p.name + ' has no rank knob.' : p.rs.length < 4 ? p.name + ' allows r ∈ {' + p.rs.join(', ') + '}. Changing r rebuilds the preset.' : 'Changing r rebuilds the preset (r = 1 … 4).';
    }
    function nodeDesc(x) {
      if (!x) return '';
      if (x.t === 'sig') return 'σ: GELU on a wire of size ' + x.out + ', on the activation path';
      if (x.t !== 'box') return '';
      var shape = x.out + '×' + x.inp;
      if (x.role === 'train') return x.label + ': trainable ' + (x.st === 'diag' ? 'diagonal spider on ℝ' + sup(x.out) + ' (' + x.out + ' entries)' : 'box ' + shape + ' (' + x.out * x.inp + ' entries)');
      if (x.st === 'diag') return x.label + ': frozen random diagonal on ℝ' + sup(x.out);
      return x.label + ': frozen ' + (FK_NAMES[x.fk] || x.fk) + ', ' + shape;
    }
    function renderEditor() {
      var opts = [];
      forEachNode(state.root, function (x) { if (x.t === 'box' || x.t === 'sig') opts.push(x); });
      selSelect.innerHTML = '';
      opts.forEach(function (x) {
        var t = x.t === 'sig' ? 'σ · ' + x.out : x.label + ' · ' + (x.role === 'train' ? 'trainable' : 'frozen') + (x.st === 'diag' ? ' diag ' + x.out : ' ' + x.out + '×' + x.inp);
        selSelect.appendChild(A.h('option', { value: x.id, text: t }));
      });
      if (state.sel) selSelect.value = state.sel;
      var f = state.sel ? find(state.root, state.sel) : null, x = f && f.node;
      selDesc.innerHTML = subHtml(A.esc(nodeDesc(x)));
      var isBox = x && x.t === 'box';
      bRole.disabled = !isBox;
      bDiag.disabled = !isBox || x.out !== x.inp;
      kindSel.disabled = !isBox || x.role !== 'frozen' || x.st === 'diag' || x.fk === 'blk';
      kindSel.style.display = isBox && x.role === 'frozen' && x.st !== 'diag' && x.fk !== 'blk' ? '' : 'none';
      if (isBox && x.role === 'frozen') {
        Array.prototype.forEach.call(kindSel.options, function (o) {
          var k = o.value, ok = true;
          if (k === 'W0') ok = x.out === M && x.inp === N;
          if (k === 'svdU') ok = x.out === M && x.inp <= 8;
          if (k === 'svdV') ok = x.inp === N && x.out <= 8;
          o.disabled = !ok;
        });
        kindSel.value = x.fk;
      }
      var internal = f && f.idx < f.chain.length - 1;
      wOut.textContent = internal ? String(x.out) : '—';
      wMinus.disabled = !internal || x.out <= 1;
      wPlus.disabled = !internal || x.out >= 8;
      bRemove.disabled = !x;
      var li = f ? rootLaneOf(f) : -1, lane = li >= 0 ? state.root.lanes[li] : null;
      var upd = lane && !isBase(lane);
      bInsBox.disabled = !x; bInsDiag.disabled = !x;
      bInsSig.disabled = !x || f.owner.type !== 'root' || !upd;
      bHadNew.disabled = !upd || hasSig(lane);
      bHadW0.disabled = !upd || hasSig(lane);
      bKron.disabled = !upd || !plainLane(lane);
      bRemPath.disabled = !lane || state.root.lanes.length < 2;
      bUndo.disabled = !state.hist.length;
    }
    function renderFormula() {
      var MJ = window.MathJax, tex = !!(MJ && MJ.typesetPromise);
      if (tex && MJ.typesetClear) { try { MJ.typesetClear([fml]); } catch (e) { /* ignore */ } }
      fml.innerHTML = '';
      fml.classList.toggle('plain', !tex);
      formulaParts(state.root, tex).forEach(function (p) {
        fml.appendChild(A.h('span', { class: 'wb-fpart', text: tex ? '\\(' + p + '\\)' : p }));
      });
      if (tex) A.typeset(fml);
    }

    /* ---------------- rendering: diagram ---------------- */
    var lastW = 0;
    function renderDiagram() {
      A.tip.hide();                                /* the hovered node is about to be replaced */
      var MS = {}, root = state.root, cw0 = dia.clientWidth, compact = cw0 > 0 && cw0 < 520;
      /* phones: tighter wires and boxes so that most plugs fit the width without scrolling */
      GEO.wl = compact ? 24 : 40; GEO.port = compact ? 32 : 44; GEO.fan = compact ? 20 : 30; GEO.pad = compact ? 14 : 22;
      GEO.bw = compact ? 50 : 58; GEO.dw = compact ? 30 : 34; GEO.sw = compact ? 34 : 40;
      GEO.lane = 24; GEO.vgap = 20; GEO.top = 50; GEO.bot = 34; GEO.kcap = 11.4;
      var lanesM, innerW, innerH;
      function measureAll() {
        MS = {};
        lanesM = root.lanes.map(function (l) { return measureChain(l, MS); });
        innerW = 0; innerH = 0;
        lanesM.forEach(function (m, i) { innerW = Math.max(innerW, m.w); innerH += m.h + (i ? GEO.lane : 0); });
        innerW = Math.max(innerW, compact ? 240 : 300);
      }
      measureAll();
      var JX = 11;                                 /* room for the sum junction before the arrow into U */
      var W = 2 * GEO.port + 2 * GEO.fan + JX + innerW;
      var cw = dia.clientWidth, avail = cw > 60 ? cw - 2 : 600;
      lastW = dia.clientWidth;
      var scale = Math.min(1.45, avail / W);
      if (scale < 0.64) scale = 0.64;              /* below this the diagram scrolls sideways */
      /* reader-facing text keeps a minimum size on screen: fz(d, px) is d viewBox units, or more when the
         drawing is scaled down so far that d units would render below px pixels */
      function fz(d, px) { return +Math.max(d, px / scale).toFixed(2); }
      var capFs = fz(9.5, 12.5), tagFs = fz(10, 12.5), glFs = fz(12.5, 13), boxFs = fz(17, 14), frzFs = fz(16, 14),
        dgFs = fz(14, 13), symFs = fz(17, 14), portFs = fz(19, 15), hdFs = fz(9, 12.5);
      /* larger labels need more room between lanes and between the branches of a diamond; only heights
         change, so W and the scale stay put */
      var grow = Math.max(0, capFs - 9.5);
      if (grow > 0) { GEO.lane = 24 + 3.7 * grow; GEO.vgap = 20 + grow; GEO.top = 50 + grow; GEO.bot = 34 + grow; GEO.kcap = 1.2 * capFs; measureAll(); }
      var H = GEO.top + innerH + GEO.bot;
      function estW(txt, f, k) { return txt.length * f * (k || 0.56); }
      dia.innerHTML = '';
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, width: Math.round(W * scale), height: Math.round(H * scale), role: 'group', 'aria-label': 'Plug diagram: ' + formula(root, false) }, dia);
      var defs = S('defs', null, svg);
      var pat = S('pattern', { id: UID + '-hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
      S('line', { x1: 0, y1: 0, x2: 0, y2: 5, style: 'stroke:var(--rule);stroke-width:2.2' }, pat);
      var gHalo = S('g', null, svg), gWire = S('g', null, svg), gNode = S('g', null, svg), gOver = S('g', null, svg);
      var WIRES = {}, SPIDERS = {};
      var sel = state.sel;
      function wire(x1, y1, x2, y2, id, dim, labelDim) {
        S('line', { x1: x1, y1: y1, x2: x2, y2: y2, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
        if (id) WIRES[id] = { x1: x1, y1: y1, x2: x2, y2: y2, dim: dim, lab: !!labelDim };
      }
      function curve(x1, y1, x2, y2) {
        var mx = (x1 + x2) / 2;
        S('path', { d: 'M' + x1 + ',' + y1 + ' C' + mx + ',' + y1 + ' ' + mx + ',' + y2 + ' ' + x2 + ',' + y2, style: 'fill:none;stroke:var(--ink-2);stroke-width:1.25' }, gWire);
      }
      function nodeGroup(x, extra) {
        var g = S('g', { class: 'wb-node' + (x.id === sel ? ' is-sel' : ''), tabindex: 0, role: 'button', 'aria-label': nodeDesc(x) + '. Select to edit.', 'aria-pressed': String(x.id === sel) }, gNode);
        g.addEventListener('click', function () { state.sel = x.id; renderDiagram(); renderEditor(); });
        g.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); state.sel = x.id; renderDiagram(); renderEditor(); var n = dia.querySelector('[data-id="' + x.id + '"]'); if (n) n.focus(); } });
        g.addEventListener('mouseenter', function (ev) { A.tip.show('<div class="t">' + A.esc(x.label || 'σ') + '</div>' + A.esc(nodeDesc(x).replace(/^[^:]*: /, '')), ev); });
        g.addEventListener('mousemove', function (ev) { A.tip.move(ev); });
        g.addEventListener('mouseleave', function () { A.tip.hide(); });
        g.setAttribute('data-id', x.id);
        return g;
      }
      function drawBox(x, x0, yc) {
        var g = nodeGroup(x), w = GEO.bw, h = GEO.bh, y0 = yc - h / 2;
        if (x.st === 'diag') {
          var cx = x0 + GEO.dw / 2, tr = x.role === 'train';
          S('line', { x1: cx, y1: yc - 9, x2: cx, y2: yc - 19, style: 'stroke:var(--ink-2);stroke-width:1.25' }, g);
          S('path', { d: 'M' + (cx - 6) + ',' + (yc - 19) + ' L' + (cx + 6) + ',' + (yc - 19) + ' L' + cx + ',' + (yc - 27) + ' Z', class: 'wb-shape', style: tr ? 'fill:var(--tide);stroke:var(--tide);stroke-width:1' : 'fill:var(--paper);stroke:var(--ink-3);stroke-width:1' }, g);
          S('circle', { cx: cx, cy: yc, r: 8.5, class: 'wb-shape', style: tr ? 'fill:var(--tide);stroke:var(--tide);stroke-width:1' : 'fill:var(--paper);stroke:var(--ink-3);stroke-width:1.2' }, g);
          richText(S('text', { x: cx, y: yc + 12.5 + 0.75 * dgFs, 'text-anchor': 'middle', style: 'font-family:var(--f-display);font-style:italic;font-size:' + dgFs + 'px;fill:var(--ink)' }, g), x.label, dgFs);
          return;
        }
        if (x.role === 'train') {
          S('rect', { x: x0, y: y0, width: w, height: h, rx: 5, class: 'wb-shape', style: 'fill:var(--tide);stroke:var(--tide);stroke-width:1' }, g);
          richText(S('text', { x: x0 + w / 2, y: yc + 0.3 * boxFs, 'text-anchor': 'middle', style: 'font-family:var(--f-display);font-style:italic;font-size:' + boxFs + 'px;fill:var(--paper)' }, g), x.label, boxFs);
        } else {
          S('rect', { x: x0, y: y0, width: w, height: h, rx: 5, style: 'fill:var(--paper)' }, g);
          S('rect', { x: x0, y: y0, width: w, height: h, rx: 5, class: 'wb-shape', style: 'fill:url(#' + UID + '-hatch);stroke:var(--ink-3);stroke-width:1.2' }, g);
          var lw = Math.min(w - 6, Math.max(16, x.label.length * 0.53 * frzFs + 8)), lh = 1.25 * frzFs;
          S('rect', { x: x0 + w / 2 - lw / 2, y: yc - lh / 2, width: lw, height: lh, rx: 3, style: 'fill:var(--paper)' }, g);
          richText(S('text', { x: x0 + w / 2, y: yc + 0.31 * frzFs, 'text-anchor': 'middle', style: 'font-family:var(--f-display);font-size:' + frzFs + 'px;fill:var(--ink)' }, g), x.label, frzFs);
        }
        var isW0 = x.fk === 'W0', cap = isW0 ? 'θ₀' : x.out + '×' + x.inp;
        richText(S('text', { x: x0 + w / 2, y: y0 + h + 3 + 0.95 * capFs, 'text-anchor': 'middle', class: isW0 ? 'serif' : 'mono', style: 'font-size:' + capFs + 'px;fill:var(--ink-2)' + (isW0 ? ';font-style:italic' : '') }, g), cap, capFs);
      }
      function drawSig(x, x0, yc) {
        var g = nodeGroup(x), w = GEO.sw, h = GEO.bh - 6;
        S('rect', { x: x0 + 2, y: yc - h / 2, width: w - 4, height: h, rx: h / 2, class: 'wb-shape', style: 'fill:var(--paper);stroke:var(--seal);stroke-width:1.5' }, g);
        S('text', { x: x0 + w / 2, y: yc + 0.32 * symFs, 'text-anchor': 'middle', style: 'font-family:var(--f-display);font-style:italic;font-size:' + symFs + 'px;fill:var(--seal)' }, g, 'σ');
      }
      function drawChain(ch, x0, x1, yc) {
        var x = x0;
        if (!ch.length) return;
        wire(x, yc, x + GEO.wl, yc, 'i:' + ch[0].id, ch[0]._i, true);
        x += GEO.wl;
        for (var k = 0; k < ch.length; k++) {
          var nd = ch[k], m = MS[nd.id];
          drawNode(nd, x, yc, m);
          x += m.w;
          var xe = k === ch.length - 1 ? x1 : x + GEO.wl;
          wire(x, yc, xe, yc, 'o:' + nd.id, nd._o, true);
          x = xe;
        }
      }
      function drawNode(nd, x0, yc, m) {
        if (nd.t === 'box') return drawBox(nd, x0, yc);
        if (nd.t === 'sig') return drawSig(nd, x0, yc);
        if (nd.t === 'had' || nd.t === 'kron') {
          var top = yc - m.h / 2, ya = top + m.a.h / 2, yb = top + m.a.h + GEO.vgap + m.b.h / 2, pad = m.pad, xr = x0 + m.w;
          if (nd.t === 'had') {
            curve(x0 + 4, yc, x0 + pad, ya); curve(x0 + 4, yc, x0 + pad, yb);
            curve(xr - pad, ya, xr - 4, yc); curve(xr - pad, yb, xr - 4, yc);
            S('line', { x1: x0, y1: yc, x2: x0 + 4, y2: yc, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
            S('line', { x1: xr - 4, y1: yc, x2: xr, y2: yc, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
            S('circle', { cx: x0 + 4, cy: yc, r: 3.6, style: 'fill:var(--ink)' }, gNode);
            S('circle', { cx: xr - 4, cy: yc, r: 3.6, style: 'fill:var(--ink)' }, gNode);
            SPIDERS[nd.id + ':in'] = { x: x0 + 4, y: yc }; SPIDERS[nd.id + ':out'] = { x: xr - 4, y: yc };
            S('text', { x: x0 + pad + 13, y: yc + 0.35 * symFs, 'text-anchor': 'middle', style: 'font-family:var(--f-display);font-size:' + symFs + 'px;fill:var(--ink-2)' }, gNode, '⊙');
          } else {
            var ch = m.h - GEO.kcap;               /* the branches; the caption line sits below them */
            [x0 + 2, xr - 9].forEach(function (bx) { S('rect', { x: bx, y: top + 3, width: 7, height: ch - 6, rx: 2, style: 'fill:var(--paper-3);stroke:var(--ink-2);stroke-width:1' }, gNode); });
            S('line', { x1: x0 + 9, y1: ya, x2: x0 + pad, y2: ya, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
            S('line', { x1: x0 + 9, y1: yb, x2: x0 + pad, y2: yb, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
            S('line', { x1: xr - pad, y1: ya, x2: xr - 9, y2: ya, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
            S('line', { x1: xr - pad, y1: yb, x2: xr - 9, y2: yb, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
            S('text', { x: x0 + pad + 13, y: (ya + yb) / 2 + 0.35 * symFs, 'text-anchor': 'middle', style: 'font-family:var(--f-display);font-size:' + symFs + 'px;fill:var(--ink-2)' }, gNode, '⊗');
            S('text', { x: x0 + 2, y: top + ch - 3 + 2.15 * capFs, 'text-anchor': 'start', class: 'serif', style: 'font-size:' + capFs + 'px;fill:var(--ink-2)' }, gNode, 'ℝ' + sup(nd._i) + ' = ℝ' + sup(nd.a[0]._i) + '⊗ℝ' + sup(nd.b[0]._i));
          }
          drawChain(nd.a, x0 + pad, xr - pad, ya);
          drawChain(nd.b, x0 + pad, xr - pad, yb);
          return;
        }
        if (nd.t === 'norm') {
          var t2 = yc - m.h / 2, ic = t2 + 20 + m.s.h / 2;
          S('rect', { x: x0 + 2, y: t2, width: m.w - 4, height: m.h, rx: 6, style: 'fill:none;stroke:var(--ink-3);stroke-width:1.1;stroke-dasharray:4 3' }, gNode);
          var nTxt = estW('N(·) row norm, parameter side', capFs) < m.w - 16 ? 'N(·) row norm, parameter side' : 'N(·) row norm';
          S('text', { x: x0 + 9, y: t2 + 4 + 0.95 * capFs, class: 'ui', style: 'font-size:' + capFs + 'px;fill:var(--ink-2)' }, gNode, nTxt);
          curve(x0, yc, x0 + 10, ic); curve(x0 + m.w - 10, ic, x0 + m.w, yc);
          drawSum(nd.inner, x0 + 10, x0 + m.w - 10, ic, m.s);
          return;
        }
      }
      function plusJunction(x, y, parent) {
        S('circle', { cx: x, cy: y, r: 7.5, style: 'fill:var(--paper);stroke:var(--ink-2);stroke-width:1.25' }, parent);
        S('line', { x1: x - 4, y1: y, x2: x + 4, y2: y, style: 'stroke:var(--ink-2);stroke-width:1.25' }, parent);
        S('line', { x1: x, y1: y - 4, x2: x, y2: y + 4, style: 'stroke:var(--ink-2);stroke-width:1.25' }, parent);
      }
      function drawSum(sm, x0, x1, yc, ms) {
        var y = yc - ms.h / 2, ys = [], jx = x1 - 8;
        ms.lanes.forEach(function (lm) { ys.push(y + lm.h / 2); y += lm.h + GEO.lane; });
        ys.forEach(function (c, i) {
          curve(x0, yc, x0 + GEO.pad, c);
          drawChain(sm.lanes[i], x0 + GEO.pad, x1 - GEO.pad - 16, c);
          curve(x1 - GEO.pad - 16, c, jx, yc);
        });
        S('circle', { cx: x0, cy: yc, r: 2.6, style: 'fill:var(--ink-2)' }, gNode);
        plusJunction(jx, yc, gNode);
        S('line', { x1: jx + 7.5, y1: yc, x2: x1, y2: yc, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
      }
      /* root: ports, hole, lanes */
      var yMid = GEO.top + innerH / 2;
      var hx0 = GEO.port - 6, hx1 = W - GEO.port + 6;
      S('rect', { x: hx0, y: GEO.top - 22, width: hx1 - hx0, height: innerH + 38, rx: 8, style: 'fill:var(--paper-2);stroke:var(--ink-3);stroke-width:1;stroke-dasharray:5 4' }, gHalo);
      var hdTxt = ['THE HOLE V → U, FILLED BY THE PLUG ρ(q)', 'THE HOLE V → U, FILLED BY ρ(q)', 'THE HOLE V → U'].filter(function (t) { return estW(t, hdFs, 0.7) <= hx1 - hx0 - 14; })[0] || 'THE HOLE V → U';
      S('text', { x: hx0 + 8, y: GEO.top - 27, class: 'ui', style: 'font-size:' + hdFs + 'px;letter-spacing:.06em;fill:var(--ink-2)' }, gHalo, hdTxt);
      /* ports */
      [[GEO.port - 6, 'V', 'ℝ' + sup(N)], [W - GEO.port + 6, 'U', 'ℝ' + sup(M)]].forEach(function (pt, i) {
        var px = pt[0];
        S('rect', { x: px - 5, y: yMid - 13, width: 10, height: 26, rx: 3, style: 'fill:var(--ink);stroke:none' }, gNode);
        S('text', { x: i ? px + 9 : px - 9, y: yMid + 0.32 * portFs, 'text-anchor': i ? 'start' : 'end', style: 'font-family:var(--f-display);font-style:italic;font-size:' + portFs + 'px;fill:var(--ink)' }, gNode, pt[1]);
        S('text', { x: px, y: yMid + 17 + 0.95 * capFs, 'text-anchor': 'middle', class: 'serif', style: 'font-size:' + capFs + 'px;fill:var(--ink-2);paint-order:stroke;stroke:var(--paper);stroke-width:3px;stroke-linejoin:round' }, gNode, pt[2]);
      });
      var xa = GEO.port + 4, xb = W - GEO.port - 4;
      S('line', { x1: GEO.port - 1, y1: yMid, x2: xa, y2: yMid, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
      var jx0 = xb - JX;                           /* the sum junction, clear of the arrowhead into U */
      S('line', { x1: jx0 + 7.5, y1: yMid, x2: W - GEO.port + 1, y2: yMid, style: 'stroke:var(--ink-2);stroke-width:1.25' }, gWire);
      var y = GEO.top, laneY = [];
      lanesM.forEach(function (m) { laneY.push(y + m.h / 2); y += m.h + GEO.lane; });
      var laneX0 = GEO.port + GEO.fan, laneX1 = W - GEO.port - GEO.fan - JX;
      laneY.forEach(function (c, i) {
        curve(xa, yMid, laneX0, c);
        drawChain(root.lanes[i], laneX0, laneX1, c);
        curve(laneX1, c, jx0, yMid);
      });
      S('circle', { cx: xa, cy: yMid, r: 2.8, style: 'fill:var(--ink-2)' }, gNode);
      plusJunction(jx0, yMid, gNode);
      root.lanes.forEach(function (l, i) {
        if (isBase(l) && laneX1 - (laneX0 + GEO.wl + GEO.bw) > estW('base point θ₀', capFs) + 16) S('text', { x: laneX1 - 4, y: laneY[i] - 6, 'text-anchor': 'end', style: 'font-size:' + capFs + 'px;fill:var(--ink-2)' }, gOver, 'base point θ₀');
      });

      /* overlays from the last analysis: min cuts (Thm II.10), gauge halos, then wire sizes */
      var cutX = {}, okR = R && !R.errs.length && R.C && R.C.root === state.root;
      function tag(x, y, txt, color, anchor) {
        var tw = txt.length * tagFs * 0.6 + 8, x0 = anchor === 'end' ? x - tw + 4 : anchor === 'start' ? x - 4 : x - tw / 2;
        S('rect', { x: x0, y: y - 1.05 * tagFs, width: tw, height: 1.4 * tagFs, rx: 3, style: 'fill:var(--paper-2);fill-opacity:.92' }, gOver);
        S('text', { x: x, y: y, 'text-anchor': anchor || 'middle', class: 'mono', style: 'font-size:' + tagFs + 'px;fill:' + color }, gOver, txt);
      }
      if (okR && !R.rank.nonlin) {
        R.rank.parts.forEach(function (pt) {
          if (pt.w >= Math.min(M, N)) return;
          var P = [];
          pt.cut.cross.forEach(function (c) { var w = WIRES[c.wire]; if (w) { P.push({ x: (w.x1 + w.x2) / 2, y: w.y1, dim: c.dim }); cutX[c.wire] = true; } });
          if (!P.length) return;
          P.sort(function (u, v) { return u.y - v.y; });
          var last = P[P.length - 1], aligned = P.every(function (u) { return Math.abs(u.x - P[0].x) < 4; });
          var cutStyle = 'fill:none;stroke:var(--ochre);stroke-width:2;stroke-dasharray:4 3;stroke-linecap:round';
          if (aligned) S('path', { d: 'M' + P[0].x + ',' + (P[0].y - 14) + ' L' + last.x + ',' + (last.y + 14), style: cutStyle }, gOver);
          else P.forEach(function (u) { S('path', { d: 'M' + u.x + ',' + (u.y - 14) + ' L' + u.x + ',' + (u.y + 14), style: cutStyle }, gOver); });
          var txt = 'cut ' + (P.length > 1 ? P.map(function (u) { return u.dim; }).join('·') + ' = ' : '') + pt.w;
          tag(P[0].x, P[0].y - 9 - 1.4 * tagFs, txt, 'var(--ochre)');
        });
      }
      if (okR) {
        var wireBest = {};
        R.pieces.forEach(function (p) { if (p.wire && (!wireBest[p.wire] || (p.kind === 'GL' && wireBest[p.wire].kind !== 'GL'))) wireBest[p.wire] = p; });
        R.pieces.forEach(function (p) {
          if (p.wire && wireBest[p.wire] !== p) return;
          if (p.wire && WIRES[p.wire]) {
            var w = WIRES[p.wire], xm = (w.x1 + w.x2) / 2, gl = (p.kind === 'GL' ? 'GL' : 'T') + sub(p.k), gw = (p.kind === 'GL' ? 23 : 13) * glFs / 12.5;
            S('line', { x1: w.x1 + 3, y1: w.y1, x2: w.x2 - 3, y2: w.y2, style: 'stroke:var(--tide);stroke-opacity:.26;stroke-width:9;stroke-linecap:round' }, gHalo);
            if (cutX[p.wire]) S('rect', { x: xm - gw / 2 - 2, y: w.y1 + 7, width: gw + 4, height: 1.12 * glFs, rx: 3, style: 'fill:var(--paper-2)' }, gOver);
            richText(S('text', { x: xm, y: w.y1 + 7 + 0.8 * glFs, 'text-anchor': 'middle', style: 'font-family:var(--f-display);font-style:italic;font-size:' + glFs + 'px;fill:var(--tide)' }, gOver), gl, glFs);
          } else if (p.spider && SPIDERS[p.spider]) {
            var sp = SPIDERS[p.spider], isIn = /:in$/.test(p.spider);
            S('circle', { cx: sp.x, cy: sp.y, r: 9.5, style: 'fill:var(--tide);fill-opacity:.22;stroke:none' }, gHalo);
            richText(S('text', { x: sp.x + (isIn ? -7 : 7), y: sp.y - 0.88 * glFs, 'text-anchor': isIn ? 'end' : 'start', style: 'font-family:var(--f-display);font-style:italic;font-size:' + glFs + 'px;fill:var(--tide)' }, gOver), 'T' + sub(p.k), glFs);
          }
        });
      }
      Object.keys(WIRES).forEach(function (id) {
        var w = WIRES[id];
        if (!w.lab || w.dim == null || w.dim === 8 || w.x2 - w.x1 < 14) return;
        var xm = (w.x1 + w.x2) / 2;
        if (cutX[id]) S('rect', { x: xm - 0.6 * tagFs, y: w.y1 - 4 - 1.1 * tagFs, width: 1.2 * tagFs, height: 1.2 * tagFs, rx: 2, style: 'fill:var(--paper-2)' }, gOver);
        S('text', { x: xm, y: w.y1 - 5, 'text-anchor': 'middle', class: 'mono', style: 'font-size:' + tagFs + 'px;fill:var(--ink-2)' }, gOver, String(w.dim));
      });
      /* arrowhead into U */
      S('path', { d: 'M' + (W - GEO.port - 6) + ',' + (yMid - 4) + ' L' + (W - GEO.port + 0) + ',' + yMid + ' L' + (W - GEO.port - 6) + ',' + (yMid + 4), style: 'fill:none;stroke:var(--ink-2);stroke-width:1.25' }, gOver);
    }

    function renderLegend() {
      legend.innerHTML = '';
      function item(svg, text) { var s = A.h('span'); s.innerHTML = svg; s.appendChild(document.createTextNode(text)); legend.appendChild(s); }
      item('<svg width="22" height="14" aria-hidden="true"><rect x="1" y="1" width="20" height="12" rx="3" style="fill:var(--tide)"/></svg>', 'trainable box');
      item('<svg width="22" height="14" aria-hidden="true"><defs><pattern id="' + UID + '-lh" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4" style="stroke:var(--rule);stroke-width:2"/></pattern></defs><rect x="1" y="1" width="20" height="12" rx="3" style="fill:url(#' + UID + '-lh);stroke:var(--ink-3)"/></svg>', 'frozen box');
      item('<svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="10" r="5" style="fill:var(--tide)"/><path d="M4,4 L12,4 L8,0 Z" style="fill:var(--tide)"/></svg>', 'diagonal spider');
      item('<svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="3.5" style="fill:var(--ink)"/></svg>', 'copy spider (⊙)');
      item('<svg width="24" height="14" aria-hidden="true"><rect x="2" y="2" width="20" height="10" rx="5" style="fill:var(--paper);stroke:var(--seal);stroke-width:1.5"/></svg>', 'σ (activation)');
      item('<svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="6.5" style="fill:var(--paper);stroke:var(--ink-2);stroke-width:1.2"/><path d="M4.5,8 H11.5 M8,4.5 V11.5" style="stroke:var(--ink-2);stroke-width:1.2"/></svg>', 'formal sum of paths');
      item('<svg width="20" height="16" aria-hidden="true"><rect x="2" y="1" width="5" height="14" rx="1.5" style="fill:var(--paper-3);stroke:var(--ink-2)"/><rect x="13" y="1" width="5" height="14" rx="1.5" style="fill:var(--paper-3);stroke:var(--ink-2)"/></svg>', 'Kronecker reshape (⊗)');
      item('<svg width="26" height="14" aria-hidden="true"><line x1="1" y1="7" x2="25" y2="7" style="stroke:var(--tide);stroke-opacity:.3;stroke-width:8;stroke-linecap:round"/><line x1="1" y1="7" x2="25" y2="7" style="stroke:var(--ink-2);stroke-width:1.25"/></svg>', 'gauge on a wire or spider');
      item('<svg width="14" height="18" aria-hidden="true"><line x1="7" y1="1" x2="7" y2="17" style="stroke:var(--ochre);stroke-width:2;stroke-dasharray:3.5 2.5"/></svg>', 'minimum cut');
      item('<svg width="22" height="16" aria-hidden="true"><line x1="1" y1="12" x2="21" y2="12" style="stroke:var(--ink-2);stroke-width:1.25"/><text x="11" y="9" text-anchor="middle" style="font-family:var(--f-mono);font-size:11px;fill:var(--ink-2)">2</text></svg>', 'wire size (8 unless marked)');
    }

    /* ---------------- rendering: readouts ---------------- */
    /* singular values on a log scale, drawn at the tile's own pixel width so the tick labels keep their size */
    function spark(sv, d, wpx) {
      var w = Math.max(200, Math.round(wpx || 300)), h = 62, n = sv.length, lx = 42;
      if (!n) return '';
      var s0 = sv[0] || 1, lo = -17, bw = Math.max(1, (w - lx - 2) / n);
      var out = '<svg class="wb-spark" viewBox="0 0 ' + w + ' ' + h + '" role="img" aria-label="Singular values of the Jacobian on a log scale: ' + d + ' of ' + n + ' lie above 1e-8 times the largest">';
      function yOf(v) { var l = Math.max(lo, Math.log10(Math.max(v / s0, 1e-300))); return 6 + (0 - l) / (0 - lo) * (h - 18); }
      out += '<line x1="' + (lx - 3) + '" x2="' + w + '" y1="' + yOf(1e-8 * s0).toFixed(1) + '" y2="' + yOf(1e-8 * s0).toFixed(1) + '" style="stroke:var(--ochre);stroke-width:1;stroke-dasharray:3 2"/>';
      for (var i = 0; i < n; i++) {
        var yv = yOf(sv[i]), x = lx + i * bw;
        out += '<rect x="' + x.toFixed(2) + '" y="' + yv.toFixed(1) + '" width="' + Math.max(0.8, bw - (bw > 3 ? 1 : 0.3)).toFixed(2) + '" height="' + Math.max(0.8, h - 12 - yv).toFixed(1) + '" style="fill:' + (i < d ? 'var(--tide)' : 'var(--ink-3);fill-opacity:.45') + '"/>';
      }
      function pow(y, e, st) { return '<text x="0" y="' + y.toFixed(1) + '"' + (st ? ' style="' + st + '"' : '') + '>10<tspan dy="-4.5" style="font-size:8.5px">' + e + '</tspan></text>'; }
      out += pow(yOf(s0) + 9, '0') + pow(yOf(1e-8 * s0) + 4, '−8', 'fill:var(--ochre-ink)') + pow(h - 12, '−17');
      return out + '</svg>';
    }
    function set(elm, html, pulse) {
      if (elm.__h === html) return;
      elm.__h = html; elm.innerHTML = subHtml(html);
      if (pulse && !A.reducedMotion()) { elm.classList.remove('wb-pulse'); void elm.offsetWidth; elm.classList.add('wb-pulse'); }
    }
    function esc(s) { return A.esc(s); }
    /* a method title, with its parameters in a quieter face: 'LoHa₂ (r₁ = r₂ = 2)' */
    function idTitle(t) { var m = /^(.*?)( \(.*\))$/.exec(t); return m ? esc(m[1]) + '<span class="wb-idpar">' + esc(m[2]) + '</span>' : esc(t); }
    function renderReadouts(pulse) {
      if (!R || R.errs.length) {
        set(tP, '<div class="wb-k">plug</div><div class="wb-s wb-bad">' + esc(R && R.errs[0] || '') + '</div>');
        return;
      }
      var e = R.id && R.id.e, exact = R.id && R.id.kind === 'exact', chk = exact && e.p.chk ? e.p.chk(e.r) : null;
      /* |D| */
      var parts = R.C.train.map(function (b) { return esc(b.label) + ' ' + b._sz; });
      set(tP, '<div class="wb-k">|D| · trainable entries</div><div class="wb-v">' + R.P + '</div><div class="wb-s mono">' + (parts.length ? (parts.length > 6 ? parts.slice(0, 6).join(' · ') + ' · …' : parts.join(' · ')) : 'nothing trains') + '</div>', pulse);
      /* degree */
      var degNote = !R.linear ? 'With σ on a path, ρ is no longer a polynomial in q; this counts trainable boxes on one path.' : R.norm ? 'The row norm divides by a Euclidean norm, so ρ is not a polynomial in q.' : R.deg === 0 ? 'Nothing trains, so ρ is constant.' : R.deg === 1 ? 'ρ is affine in q, so the image is an affine space.' : 'ρ is a polynomial of degree ' + R.deg + ' in q, linear in each box.';
      set(tDeg, '<div class="wb-k">degree · trainable boxes on a path</div><div class="wb-v">' + R.deg + '</div><div class="wb-s">' + degNote + '</div>', pulse);
      /* rank */
      var rk = R.rank, rkHtml;
      if (rk.nonlin) rkHtml = '<div class="wb-k">rank bound · min cut (Thm II.10)</div><div class="wb-v">n/a</div><div class="wb-s">σ is not a linear map, and Thm II.10 is a statement about linear networks (FinVect).</div>';
      else {
        var sumTxt = rk.parts.length ? rk.parts.map(function (p) { return p.w; }).join(' + ') : '0';
        var extra = rk.extra ? ' + ' + rk.extra + (rk.bare ? ' (extra W₀)' : ' (−W₀)') : '';
        rkHtml = '<div class="wb-k">rank of <span class="nt">ΔW</span> · min cut (Thm II.10)</div><div class="wb-v">≤ ' + rk.bound + '<small>of ' + Math.min(M, N) + '</small></div>' +
          '<div class="wb-s mono">Σ cuts = ' + sumTxt + extra + (rk.total + rk.extra > Math.min(M, N) ? ' → capped at min(m, n)' : '') + '</div>' +
          '<div class="wb-s">Attained at a random point: <b class="' + (R.att === rk.bound ? 'wb-ok' : 'wb-hi') + '">' + R.att + '</b>' + (R.att === rk.bound ? ', so the bound is sharp here.' : '. The cut bounds the rank from above only.') + '</div>';
        if (chk && chk.rank != null) rkHtml += '<div class="wb-chk">' + esc(chk.rRef) + ': max rank ' + esc(chk.rTxt) + ' = ' + chk.rank + ' ' + (chk.rank === rk.bound ? '<span class="wb-ok">✓ the cut agrees</span>' : '<span class="wb-bad">✗ the cut differs</span>') + '</div>';
      }
      set(tRank, rkHtml, pulse);
      /* merge */
      var mg = R.merge === 'M1'
        ? '<span class="wb-badge yes">M1 · fuses into W</span><div class="wb-s">The plug is linear in x, so after training it is one m×n matrix that replaces W₀ (rule R1, Thm VI.3).' + (R.norm ? ' The row norm acts on the weight matrix, so it does not change this.' : '') + '</div>'
        : '<span class="wb-badge no">M∞ · does not fuse</span><div class="wb-s">σ sits on the activation path, so for generic q the layer is non-affine in x and no single m×n matrix reproduces it. The plug is not box-locally mergeable (Thm VI.3(c)).</div>';
      set(tMerge, '<div class="wb-k">merge verdict (Thm VI.3)</div>' + mg, pulse);
      /* d(M) */
      var dHtml = '<div class="wb-k"><span class="nt">d(M)</span> · Jacobian rank at <span class="nt">m = n = 8</span> (Def I.5)' + (R.linear ? '' : ', functional') + '</div>';
      if (R.tooBig) dHtml += '<div class="wb-v">—</div><div class="wb-s">More than ' + PMAX + ' trainable entries; the live Jacobian is skipped.</div>';
      else {
        dHtml += '<div class="wb-v">' + R.d + '<small>of ' + (R.linear ? MN + ' = mn' : (R.C.K * M) + ' probe outputs') + '</small></div>';
        if (R.sv && R.sv.length) dHtml += spark(R.sv, R.d, tD.clientWidth - 26) + '<div class="wb-s mono">' + R.d + ' of ' + R.sv.length + ' singular values lie above 10⁻⁸σ_max' + (R.sv.length > R.d ? '; the next is ' + fmtExp(R.sv[R.d] / R.sv[0]) + ' σ_max' : '') + '.</div>';
        if (!R.linear) dHtml += '<div class="wb-s">ρ lands in functions, so d is the rank of q ↦ (y(x₁), …, y(x' + sub(R.C.K) + ')) at ' + R.C.K + ' random inputs.</div>';
        if (chk && chk.d != null) {
          var okD = chk.eq ? R.d === chk.d : R.d <= chk.d;
          dHtml += '<div class="wb-chk">' + esc(chk.dRef) + ': ' + esc(chk.dTxt) + ' = ' + chk.d + ' ' + (okD ? '<span class="wb-ok">✓ ' + (chk.eq ? 'agrees' : (R.d === chk.d ? 'attains the bound' : 'within the bound')) + '</span>' : '<span class="wb-bad">✗ differs</span>') + '</div>';
        } else if (exact && e.p.id === 'lokr') dHtml += '<div class="wb-chk">No closed form for LoKr’s d in the framework; KronA’s row of Prop II.12 is the case r′ = 4.</div>';
        else if (exact && (e.p.id === 'houlsby' || e.p.id === 'parallel')) dHtml += '<div class="wb-chk">No closed form in the framework. ' + (R.d === R.P ? 'The computed d = |D| fits Lemma VI.2(c), which leaves GELU no continuous hidden-unit gauge.' : 'Here d = ' + R.d + ' is below |D| = ' + R.P + '.') + '</div>';
      }
      set(tD, dHtml, pulse);
      /* gauge */
      var agg = {}, order = [];
      R.pieces.forEach(function (p) {
        var k = p.kind === 'S' ? 'ℝ^× scalar trade' : (p.kind === 'GL' ? 'GL' : 'T') + sub(p.k) + (p.spider ? ' on a ⊙ spider' : p.through ? ' through ' + p.through : ' on ' + p.from + '→' + p.to);
        if (p.kind !== 'S' && !p.spider && !p.through) k = (p.kind === 'GL' ? 'GL' : 'T') + sub(p.k) + ' wire';
        if (!agg[k]) { agg[k] = 0; order.push(k); } agg[k]++;
      });
      var pieceTxt = order.length ? order.map(function (k) { return (agg[k] > 1 ? agg[k] + ' × ' : '') + k; }).join(' · ') : 'none exhibited';
      var owns = R.pieces.map(function (p) { return p.own; }), tot = owns.reduce(function (a, b) { return a + b; }, 0), shared = tot - R.orbit;
      var sumTxt = owns.length > 1 || shared ? owns.join(' + ') + (shared ? ' − ' + shared + ' shared' : '') + ' = ' + R.orbit : '';
      var gHtml = '<div class="wb-k">gauge · exhibited orbit (Prop II.12<span class="nt">(a)</span>)</div><div class="wb-v">' + R.orbit + '<small>' + esc(R.gt) + '</small></div><div class="wb-s mono">' + esc(pieceTxt).replace(/ℝ\^×/g, 'ℝ<sup>×</sup>') + (sumTxt ? '<br>dimensions ' + esc(sumTxt) : '') + '</div>';
      if (R.fibre != null) {
        var fTxt = 'Fibre |D| − d = ' + R.P + ' − ' + R.d + ' = ';
        if (R.fibre === 0 && R.orbit === 0) gHtml += '<div class="wb-s">' + fTxt + '<b class="wb-ok">0</b>, so ρ is an immersion at this point and has no continuous gauge.</div>';
        else if (R.fibre === R.orbit) gHtml += '<div class="wb-s">' + fTxt + '<b class="wb-ok">' + R.fibre + '</b>, the dimension of the exhibited orbit, so the gauge accounts for the whole fibre (the equality case of Prop II.12(a)).</div>';
        else {
          var why = R.d === MN ? 'Once d = mn the image is open in Θ, and every further parameter only adds to the fibre.' : loraLanes(state.root) >= 2 ? 'Parallel LoRA paths concatenate, LoRAᵣ ⊕ LoRAₛ ≅ LoRAᵣ₊ₛ (Obs I.7), and the larger GL gauge of the concatenation mixes the paths.' : 'The rest of the fibre comes from symmetries that the workbench does not search for.';
          gHtml += '<div class="wb-s">' + fTxt + '<b class="wb-hi">' + R.fibre + '</b> exceeds the exhibited ' + R.orbit + ', as Prop II.12(a) allows. ' + why + '</div>';
        }
      }
      if (R.nGens) gHtml += '<div class="wb-chk">Each of the ' + R.nGens + ' generators is checked to satisfy ‖dρ·ξ‖ ≤ 10⁻⁷σ_max‖ξ‖' + (R.badGens ? '; <span class="wb-bad">' + R.badGens + ' failed</span>' : ' (all pass)') + '.</div>';
      set(tGauge, gHtml, pulse);
      /* identity */
      var idHtml = '<div class="wb-k">nearest named method</div>';
      if (!R.id || R.id.kind === 'pending') idHtml += '<div class="wb-s">Matching against the preset table…</div>';
      else if (exact) {
        idHtml += '<span class="wb-badge yes">exact plug</span><div class="wb-idname">This is ' + idTitle(e.title) + '</div><div class="wb-s">' + esc(e.p.note || '') + '</div>';
      } else if (concatLoRA(state.root)) {
        var rs = concatLoRA(state.root), rt = rs.reduce(function (a, b) { return a + b; }, 0);
        idHtml += '<span class="wb-badge yes">isomorphic</span><div class="wb-idname">This is LoRA' + sub(rt) + '</div><div class="wb-s">Concatenating the factors, ([B₁ B₂ …], [A₁; A₂; …]), is an isomorphism ' + rs.map(function (r) { return 'LoRA' + sub(r); }).join(' ⊕ ') + ' ≅ LoRA' + sub(rt) + ' of unpointed objects (Obs I.7).</div>';
      } else if (R.id.kind === 'same') idHtml += '<span class="wb-badge cond">same invariants</span><div class="wb-idname">Same signature as ' + idTitle(e.title) + '</div><div class="wb-s">|D|, degree, d, rank bound, gauge type, frozen boxes and merge type all agree, but the plug is drawn differently. Equal invariants do not make two methods isomorphic.</div>';
      else idHtml += '<span class="wb-badge cond">new gadget</span><div class="wb-idname">Nearest: ' + idTitle(e.title) + '</div><div class="wb-s">Differs in ' + esc(R.id.diffs.slice(0, 4).join('; ')) + '.</div>';
      set(tId, idHtml, pulse);
      /* equal budget vs LoRA */
      var cmp = '<div class="wb-k">against <span class="nt">LoRA</span> at the same budget</div>';
      var req = Math.min(8, Math.floor(R.P / (M + N)));
      if (req < 1) cmp += '<div class="wb-s">|D| = ' + R.P + ' is below LoRA₁’s 16 entries; LoRA₁ reaches d = 15 with rank 1.</div>';
      else {
        var dL = req * (M + N - req), mine = R.d == null ? 0 : R.d;
        cmp += '<div class="wb-bars">' +
          '<span>this plug</span><span class="wb-bar"><i style="width:' + Math.min(100, 100 * mine / MN).toFixed(1) + '%;background:var(--tide)"></i></span><span>d ' + (R.d == null ? '—' : R.d) + ' · rank ≤ ' + (R.rank.nonlin ? 'n/a' : R.rank.bound) + '</span>' +
          '<span>LoRA' + sub(req) + '</span><span class="wb-bar"><i style="width:' + (100 * dL / MN).toFixed(1) + '%;background:var(--ink-3)"></i></span><span>d ' + dL + ' · rank ≤ ' + req + '</span></div>';
        var used = req * (M + N);
        cmp += '<div class="wb-s">LoRA' + sub(req) + ' trains ' + used + (used === R.P ? ' entries, exactly this budget' : ' ≤ ' + R.P + ' entries') + '; its d = r(m+n−r) = ' + dL + ' (Prop II.12).' + (R.d != null ? ' Δd = ' + (mine - dL >= 0 ? '+' : '−') + Math.abs(mine - dL) + '.' : '') + (R.linear ? '' : ' This plug’s d counts directions in a space of functions and LoRA’s counts weights, so the two are not on one scale.') + '</div>';
        if (exact && e.p.id === 'loha' && req === 2 * e.r) {
          var r2 = 2 * e.r * e.r, gap = M + N - 1 - r2, mn0 = Math.min(M, N);
          if (gap > 0 && 2 * e.r <= mn0 && e.r * e.r <= mn0) cmp += '<div class="wb-chk">Prop II.12(c) applies here (2r ≤ min(m, n), r² ≤ min(m, n), and 2r² = ' + r2 + ' &lt; m+n−1 = ' + (M + N - 1) + '). At the same cost, LoHa' + sub(e.r) + ' reaches at least (m+n−1) − 2r² = ' + gap + ' fewer dimensions than LoRA' + sub(2 * e.r) + ' (computed: ' + (dL - mine) + ')' + (e.r * e.r <= 2 * e.r ? ', and since r² ≤ 2r its image lies strictly inside LoRA' + sub(2 * e.r) + '’s.' : ', in exchange for rank up to r² = ' + e.r * e.r + '.') + '</div>';
          else cmp += '<div class="wb-chk">Here 2r² = ' + r2 + ' ≥ m+n−1 = ' + (M + N - 1) + ', outside Prop II.12(c). The computed d of LoHa' + sub(e.r) + ' ' + (mine > dL ? 'exceeds' : mine === dL ? 'equals' : 'stays below') + ' LoRA' + sub(2 * e.r) + '’s (' + mine + ' against ' + dL + '), with rank up to ' + (R.rank.nonlin ? 'n/a' : R.rank.bound) + ' against ' + req + '.</div>';
        }
        else if (exact && e.p.id === 'gralora') cmp += '<div class="wb-chk">Prop II.12(c): GraLoRA has the d of LoRA' + sub(e.r) + ' and rank up to min(kr, m, n) = ' + Math.min(2 * e.r, M, N) + ', with every block capped at rank r/k = ' + e.r / 2 + '. Since kr = ' + 2 * e.r + ' ≤ min(m, n), the two images are incomparable.</div>';
      }
      set(tCmp, cmp, pulse);
      /* foot */
      footP.innerHTML = subHtml('Computed live in Float64 at m = n = 8 with a fixed Gaussian W₀; random draw #' + (state.salt + 1) + ' of the point q and of any random frozen frames. d(M): rank of the central-difference Jacobian (h = ' + (R.h === 0.125 ? '1/8, exact for multilinear plugs' : '10⁻⁵') + '), one-sided Jacobi SVD, threshold 10⁻⁸σ_max. ' + (R.rank.nonlin ? 'Rank bound: not defined once σ is on a path. ' : 'Rank bound: brute force over ' + (R.rank.masks || 0) + ' vertex bipartitions, copy spiders counted once, plus subadditivity over paths. ') + (R.ms != null ? 'Recomputed in ' + Math.max(1, Math.round(R.ms)) + ' ms.' : ''));
    }

    var lastMs = 0;
    function renderAll() {
      compile(state.root, state.salt);           /* sets wire sizes for the diagram */
      renderControls();
      renderEditor();
      renderFormula();
      if (lastMs < 30) recompute(false);          /* small plugs: synchronous, no flicker */
      else { renderDiagram(); recomputeSoon(); }  /* heavy plugs: debounce the Jacobian */
    }

    /* idle-time fill of the preset table (signatures for nearest-method matching) */
    function fillTable() {
      var T = buildTable(), i = 0;
      function step() {
        var t0 = Date.now();
        while (i < T.length && Date.now() - t0 < 12) { if (!T[i].sig) { var a = analyse(clone(T[i].root), 0); T[i].sig = a.sig; } i++; }
        if (i < T.length) setTimeout(step, 0);
        else if (R && R.id && R.id.kind === 'pending') { R.id = identify(R); renderReadouts(false); }
      }
      setTimeout(step, 30);
    }

    function layoutClass() {
      var w = stage.clientWidth;
      stage.classList.toggle('is-narrow', w < 560);
      stage.classList.toggle('is-mid', w >= 560 && w < 860);
    }
    layoutClass();
    renderLegend();
    loadPreset('loha');
    fillTable();
    A.onTheme(function () { renderDiagram(); renderLegend(); });
    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(function () { layoutClass(); if (Math.abs(dia.clientWidth - lastW) > 4) { renderDiagram(); renderReadouts(false); } }, 80)).observe(stage);
    else window.addEventListener('resize', A.debounce(function () { layoutClass(); renderDiagram(); }, 120));
  });
})();
