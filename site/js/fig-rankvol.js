/* Fig. "rankvol" (Figure 4): Rank Is Not Volume.
   Prop II.12 (gauge census: rank is a cut, dimension is a volume) and Thm II.10 (cut-rank bound) of
   theory/framework.md; HiRA through Prop II.9(a) and Prop II.13 (transport of structure).

   Every method is placed at the same budget |M| = R(m+n), the parameter count of LoRA_R, on one m x n box:
     LoRA_R          d = R(m+n-R)                                     max rank R
     LoHa_{r1,r2}    d = min(mn, r1(m+n-r1) + r2(m+n-r2) - (m+n-1))   max rank min(r1 r2, m, n)
                     (r1 = floor(R/2), r2 = ceil(R/2); the second argument is a proved upper bound, equality [Num])
     GraLoRA_k       d = R(m+n-R)                                     max rank min(kR, m, n)   (k | m, n, R)
     HiRA_R          d = R(m+n-R)  (W0 with no zero entries)          max rank min(R rk W0, m, n) = min(m,n), generic W0
     FourierFT       d = n_c - #conjugate pairs, n_c = |M|            max rank <= min(2 n_c, m, n)   (Thm II.10)
     C3A             d = |M| = mn/b (only when such a b divides m, n) max rank min(m,n)
   The x axis is the maximum rank (a cut), the y axis the image dimension d(M) (a volume), and
   d(M) = |M| - dim(generic fibre) <= |M| - dim(gauge orbit)  (Prop II.12(a)).

   The live check rebuilds each parametrisation, forms its Jacobian at a seeded random point by central differences
   (exact, since every map here is affine in each single coordinate), and takes the numerical rank by Householder QR
   with column pivoting, in float64. It runs on the current layer when m, n <= LIVE_MAX, and otherwise on a small square
   copy with the same R and k (the census formulas are the same polynomials in m, n and R). */
(function () {
  'use strict';

  var LIVE_MAX = 20;
  var R_CAP = 64;
  var SIZES = [4, 6, 8, 9, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256, 512, 768, 1024, 2048, 3072, 4096, 5120, 8192, 11008, 13824];
  var PRESETS = [
    { id: 'toy', label: 'toy 12 × 12', m: 12, n: 12, R: 6 },
    { id: 'bert', label: 'BERT-base 768 × 768', m: 768, n: 768, R: 8 },
    { id: 'q', label: 'LLaMA-7B q_proj', m: 4096, n: 4096, R: 8 },
    { id: 'up', label: 'LLaMA-7B up_proj', m: 11008, n: 4096, R: 8 }
  ];
  var ORDER = ['lora', 'loha', 'gralora', 'hira', 'fourier', 'c3a'];
  var META = {
    lora: { name: 'LoRA', token: '--ink' },
    loha: { name: 'LoHa', token: '--seal' },
    gralora: { name: 'GraLoRA', token: '--tide' },
    hira: { name: 'HiRA', token: '--c-multiplicative' },
    fourier: { name: 'FourierFT', token: '--ochre' },
    c3a: { name: 'C3A', token: '--moss' }
  };

  /* ---------- formatting ---------- */
  function int(x) { return Math.round(x).toLocaleString('en-US'); }
  function signed(x) { return (x > 0 ? '+' : x < 0 ? '−' : '') + int(Math.abs(x)); }
  var SUPD = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sci(x) {
    if (x === 0) return '0';
    var p = x.toExponential(0).split('e'), e = parseInt(p[1], 10);
    return p[0] + '×10' + String(e).split('').map(function (c) { return SUPD[c] || c; }).join('');
  }
  /* superscript runs as <sup> in HTML, set in the surrounding font (the web fonts do not carry U+2070-209F) */
  function hsup(t) { return String(t).replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, function (m) { return '<sup>' + m.replace(/./g, function (c) { return c === '⁻' ? '−' : String('⁰¹²³⁴⁵⁶⁷⁸⁹'.indexOf(c)); }) + '</sup>'; }); }
  /* theorem part letters such as (c) keep their case inside uppercase tags */
  function lc(t) { return String(t).replace(/\(([a-e](?:,\s?[a-e])*)\)/g, '<span class="lc">($1)</span>'); }
  function gcd(a, b) { while (b) { var t = a % b; a = b; b = t; } return a; }
  function validKs(m, n, R) {
    var g = gcd(gcd(m, n), R), ks = [];
    for (var k = 2; k <= g; k++) if (g % k === 0) ks.push(k);
    return ks;
  }

  /* ---------- FourierFT: seeded draw of n_c distinct frequencies, uniform without replacement ----------
     (the distribution of HF PEFT's randperm(m*n)[:n_frequency]); counts conjugate pairs p, -p (p != -p). */
  var drawCache = {}, drawKeys = [];
  function fourierDraw(A, m, n, nc, seed, keepIdx) {
    var key = m + 'x' + n + ':' + nc + ':' + seed + (keepIdx ? ':i' : '');
    if (drawCache[key]) return drawCache[key];
    var mn = m * n, rnd = A.rng(1000003 * seed + 7), idx = new Int32Array(nc), i, has, res;
    if (mn <= (1 << 22)) {
      var perm = new Int32Array(mn), flag = new Uint8Array(mn);
      for (i = 0; i < mn; i++) perm[i] = i;
      for (i = 0; i < nc; i++) {
        var j = i + Math.floor(rnd() * (mn - i)); if (j >= mn) j = mn - 1;
        var t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        idx[i] = perm[i]; flag[perm[i]] = 1;
      }
      has = function (x) { return flag[x] === 1; };
    } else {
      var cap = 1, bits = 0; while (cap < 2 * nc) { cap <<= 1; bits++; }
      var tab = new Int32Array(cap).fill(-1), mask = cap - 1, sh = 32 - bits, cnt = 0;
      var slot = function (x) { var hh = (Math.imul(x, 0x9E3779B1) >>> sh) & mask; while (tab[hh] !== -1 && tab[hh] !== x) hh = (hh + 1) & mask; return hh; };
      while (cnt < nc) {
        var x = Math.floor(rnd() * mn); if (x >= mn) x = mn - 1;
        var s = slot(x);
        if (tab[s] === -1) { tab[s] = x; idx[cnt++] = x; }
      }
      has = function (x) { return tab[slot(x)] === x; };
    }
    var pairs = 0, self = 0;
    for (i = 0; i < nc; i++) {
      var tt = idx[i], u = Math.floor(tt / n), v = tt - u * n;
      var c = ((m - u) % m) * n + ((n - v) % n);
      if (c === tt) self++;
      else if (tt < c && has(c)) pairs++;
    }
    res = { pairs: pairs, self: self, idx: keepIdx ? idx : null };
    drawCache[key] = res; drawKeys.push(key);
    if (drawKeys.length > 24) delete drawCache[drawKeys.shift()];
    return res;
  }

  /* ---------- the census at equal budget (Prop II.12(b)) ---------- */
  function census(A, m, n, R, kSel, seed) {
    var s = m + n, mn = m * n, mi = Math.min(m, n), P = R * s, dL = R * (s - R);
    var C = { m: m, n: n, R: R, P: P, mn: mn, mi: mi, s: s, dLoRA: dL, L: {} };
    C.ceil = Math.min(P, mn);
    C.L.lora = { id: 'lora', avail: true, P: P, d: dL, rank: R, bound: false, gdim: R * R, sub: String(R),
      gauge: 'GL<sub>' + R + '</sub>' };
    if (R >= 2) {
      var r1 = Math.floor(R / 2), r2 = R - r1;
      var F = r1 * (s - r1) + r2 * (s - r2) - (s - 1);
      C.L.loha = { id: 'loha', avail: true, P: P, F: F, d: Math.min(mn, F), capped: F > mn, r1: r1, r2: r2,
        rank: Math.min(r1 * r2, m, n), bound: false, gdim: r1 * r1 + r2 * r2 + s - 1, sub: r1 + ',' + r2,
        gauge: 'GL<sub>' + r1 + '</sub>×GL<sub>' + r2 + '</sub>×torus',
        inC: r1 === r2 && 2 * r1 <= mi && r1 * r1 <= mi };
    } else C.L.loha = { id: 'loha', avail: false, why: 'needs R ≥ 2 (one inner rank per factor)' };
    var ks = validKs(m, n, R), k = ks.indexOf(kSel) >= 0 ? kSel : (ks.length ? ks[0] : null);
    C.ks = ks; C.k = k;
    if (k) {
      C.L.gralora = { id: 'gralora', avail: true, P: P, d: dL, rank: Math.min(k * R, m, n), bound: false, k: k, gdim: R * R,
        sub: '', tail: ' k=' + k, gauge: 'GL<sub>' + (R / k) + '</sub><sup>×' + (k * k) + '</sup>', inC: k * R <= mi,
        ghosts: ks.filter(function (kk) { return kk !== k; }).map(function (kk) { return { k: kk, rank: Math.min(kk * R, m, n) }; }) };
    } else C.L.gralora = { id: 'gralora', avail: false, why: 'needs a block count k ≥ 2 dividing m, n and R' };
    C.L.hira = { id: 'hira', avail: true, P: P, d: dL, rank: mi, bound: false, gdim: R * R, sub: String(R),
      gauge: 'GL<sub>' + R + '</sub>, transported' };
    if (P <= mn) {
      var dr = fourierDraw(A, m, n, P, seed, false);
      C.L.fourier = { id: 'fourier', avail: true, P: P, d: P - dr.pairs, pairs: dr.pairs, self: dr.self, rank: Math.min(2 * P, m, n),
        bound: true, gdim: dr.pairs, sub: '', gauge: 'kernel: ' + int(dr.pairs) + ' conjugate pair' + (dr.pairs === 1 ? '' : 's') };
    } else C.L.fourier = { id: 'fourier', avail: false, why: 'n_c = |M| would exceed the mn = ' + int(mn) + ' frequencies' };
    var b = mn % P === 0 ? mn / P : 0;
    if (b && m % b === 0 && n % b === 0) {
      C.L.c3a = { id: 'c3a', avail: true, P: P, d: P, rank: mi, bound: false, b: b, gdim: 0, sub: '', tail: ' b=' + b, gauge: 'none (injective)' };
    } else C.L.c3a = { id: 'c3a', avail: false, why: 'no block size b with mn/b = |M| dividing both m and n' };
    return C;
  }

  /* ---------- live check: parametrisations, Jacobians, pivoted QR ---------- */
  function normals(rnd, len) { var q = new Float64Array(len); for (var i = 0; i < len; i++) q[i] = rnd.normal(); return q; }
  function mapLoRA(m, n, R) {
    return { P: R * (m + n), f: function (q, out) {
      out.fill(0);
      for (var i = 0; i < m; i++) for (var a = 0; a < R; a++) { var b = q[i * R + a]; if (b === 0) continue; var o = m * R + a * n; for (var j = 0; j < n; j++) out[i * n + j] += b * q[o + j]; }
    } };
  }
  function mapLoHa(m, n, r1, r2) {
    var X = new Float64Array(m * n), Y = new Float64Array(m * n);
    function lr(q, off, r, out) {
      out.fill(0);
      for (var i = 0; i < m; i++) for (var a = 0; a < r; a++) { var b = q[off + i * r + a]; var o = off + m * r + a * n; for (var j = 0; j < n; j++) out[i * n + j] += b * q[o + j]; }
    }
    return { P: (r1 + r2) * (m + n), f: function (q, out) {
      lr(q, 0, r1, X); lr(q, r1 * (m + n), r2, Y);
      for (var t = 0; t < m * n; t++) out[t] = X[t] * Y[t];
    } };
  }
  function mapGraLoRA(m, n, R, k) {
    var bm = m / k, bn = n / k, br = R / k;
    return { P: R * (m + n), f: function (q, out) {
      out.fill(0);
      var idx = 0;
      for (var I = 0; I < k; I++) for (var J = 0; J < k; J++) {
        var oB = idx; idx += bm * br; var oA = idx; idx += br * bn;
        for (var i = 0; i < bm; i++) for (var a = 0; a < br; a++) {
          var b = q[oB + i * br + a], row = (I * bm + i) * n + J * bn;
          for (var j = 0; j < bn; j++) out[row + j] += b * q[oA + a * bn + j];
        }
      }
    } };
  }
  function mapHiRA(m, n, R, W0) {
    var base = mapLoRA(m, n, R);
    return { P: base.P, f: function (q, out) { base.f(q, out); for (var t = 0; t < m * n; t++) out[t] *= W0[t]; } };
  }
  function mapFourier(m, n, idx) {
    var nc = idx.length, pat = new Float64Array(nc * m * n), mn = m * n;
    for (var c = 0; c < nc; c++) {
      var u = Math.floor(idx[c] / n), v = idx[c] - u * n;
      for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) pat[c * mn + i * n + j] = Math.cos(2 * Math.PI * (u * i / m + v * j / n)) / mn;
    }
    return { P: nc, linear: true, f: function (q, out) {
      out.fill(0);
      for (var c = 0; c < nc; c++) { var w = q[c]; if (w === 0) continue; for (var t = 0; t < mn; t++) out[t] += w * pat[c * mn + t]; }
    } };
  }
  function mapC3A(m, n, b) {
    var gm = m / b, gn = n / b;
    return { P: gm * gn * b, linear: true, f: function (q, out) {
      var off = 0;
      for (var I = 0; I < gm; I++) for (var J = 0; J < gn; J++) {
        for (var r = 0; r < b; r++) for (var c = 0; c < b; c++) out[(I * b + r) * n + J * b + c] = q[off + ((c - r + b) % b)];
        off += b;
      }
    } };
  }
  /* Jacobian, column-major (mn x P): exact central differences for maps affine in each single coordinate */
  function jacobian(map, q, mn) {
    var P = map.P, J = new Float64Array(mn * P), fp = new Float64Array(mn), fm = new Float64Array(mn), e;
    if (map.linear) {
      e = new Float64Array(P);
      for (var c = 0; c < P; c++) { e[c] = 1; map.f(e, fp); e[c] = 0; J.set(fp, c * mn); }
      return J;
    }
    var qq = Float64Array.from(q);
    for (var i = 0; i < P; i++) {
      var q0 = qq[i];
      qq[i] = q0 + 1; map.f(qq, fp);
      qq[i] = q0 - 1; map.f(qq, fm);
      qq[i] = q0;
      var o = i * mn;
      for (var t = 0; t < mn; t++) J[o + t] = 0.5 * (fp[t] - fm[t]);
    }
    return J;
  }
  /* Householder QR with column pivoting on a column-major rows x cols matrix (destroyed).
     Returns |R_kk| until the first one below 1e-13 |R_11| (that pivot included). */
  function qrcpDiag(A, rows, cols) {
    var kmax = Math.min(rows, cols), nr = new Float64Array(cols), ref = new Float64Array(cols), v = new Float64Array(rows), diag = [];
    var i, j, k;
    for (j = 0; j < cols; j++) { var s0 = 0, o0 = j * rows; for (i = 0; i < rows; i++) s0 += A[o0 + i] * A[o0 + i]; nr[j] = s0; ref[j] = s0; }
    for (k = 0; k < kmax; k++) {
      var p = k, best = nr[k];
      for (j = k + 1; j < cols; j++) if (nr[j] > best) { best = nr[j]; p = j; }
      if (p !== k) {
        var op = p * rows, ok0 = k * rows;
        for (i = 0; i < rows; i++) { var tmp = A[op + i]; A[op + i] = A[ok0 + i]; A[ok0 + i] = tmp; }
        tmp = nr[p]; nr[p] = nr[k]; nr[k] = tmp; tmp = ref[p]; ref[p] = ref[k]; ref[k] = tmp;
      }
      var ok = k * rows, al = 0;
      for (i = k; i < rows; i++) al += A[ok + i] * A[ok + i];
      al = Math.sqrt(al);
      diag.push(al);
      if (al === 0 || (diag.length > 1 && al < 1e-13 * diag[0])) break;
      if (A[ok + k] > 0) al = -al;
      var vn = 0;
      for (i = k; i < rows; i++) v[i] = A[ok + i];
      v[k] -= al;
      for (i = k; i < rows; i++) vn += v[i] * v[i];
      if (vn === 0) continue;
      var inv = 2 / vn;
      for (j = k + 1; j < cols; j++) {
        var oj = j * rows, dot = 0;
        for (i = k; i < rows; i++) dot += v[i] * A[oj + i];
        dot *= inv;
        for (i = k; i < rows; i++) A[oj + i] -= dot * v[i];
        var akj = A[oj + k];
        nr[j] -= akj * akj;
        if (nr[j] <= 1e-10 * ref[j]) { var s1 = 0; for (i = k + 1; i < rows; i++) s1 += A[oj + i] * A[oj + i]; nr[j] = s1; ref[j] = s1; }
      }
    }
    return diag;
  }
  function numRank(diag) {
    if (!diag.length || diag[0] === 0) return { rank: 0, last: 0, next: 0 };
    var r = 0; for (var i = 0; i < diag.length; i++) if (diag[i] > 1e-9 * diag[0]) r++;
    return { rank: r, last: diag[r - 1] / diag[0], next: r < diag.length ? diag[r] / diag[0] : 0 };
  }

  /* ---------- styles ---------- */
  function injectCSS() {
    if (document.getElementById('css-rankvol')) return;
    var F = '[data-figure="rankvol"] ';
    var css = [
      F + '.rv-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.rv-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.rv-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.5rem,1.05rem + 2cqi,2.1rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.rv-title i{color:var(--seal)}',
      F + '.rv-kicker{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.rv-instr{margin:.45rem 0 .8rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:70ch}',
      F + '.rv-instr b{color:var(--ink);font-weight:600}',
      F + '.rv-formula{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem 1.5rem;padding:.5rem .75rem;margin:0 0 1rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.86rem;line-height:1.5;color:var(--ink);overflow-x:auto}',
      F + '.rv-formula > span{display:inline-flex;flex-wrap:wrap;align-items:center;gap:.15rem .5rem;min-width:0}',
      F + '.rv-formula mjx-container{white-space:nowrap}',
      F + '.rv-formula mjx-container svg{max-width:none}',
      F + '.rv-tag{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.rv-tag .nt{text-transform:none;letter-spacing:0;font-family:var(--f-body);font-weight:400;font-style:italic;font-size:1.12em}',
      F + '.seg button,' + F + '.btn{font-weight:500}',
      F + '.lc{text-transform:none}',
      F + '.rv-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem 1.6rem;margin-bottom:1.1rem}',
      F + '.rv-group{min-width:0;border-top:1px solid var(--rule);padding-top:.5rem}',
      F + '.rv-gl{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;margin-bottom:.45rem}',
      F + '.rv-gl .rv-tag:first-child{color:var(--ink)}',
      F + '.rv-sl2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.4rem .9rem}',
      F + '.rv-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.rv-sl-top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0 .4rem;min-width:0}',
      F + '.rv-sl label{font-family:var(--f-body);font-size:var(--fs-sm);font-style:italic;color:var(--ink);white-space:nowrap}',
      F + '.rv-sl label .nm{font-family:var(--f-ui);font-weight:500;font-style:normal;font-size:.75rem;letter-spacing:.01em;color:var(--ink-2);margin-left:.35rem}',
      F + '.rv-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.rv-sl input[type=range]{margin:0;height:1.3rem}',
      F + '.rv-sub{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums;line-height:1.45}',
      F + '.rv-sub b{color:var(--ink);font-weight:500}',
      F + '.rv-presets{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.45rem}',
      F + '.rv-pre{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.01em;font-variant-numeric:tabular-nums;padding:.26rem .55rem;border-radius:4px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap}',
      F + '.rv-pre:hover{border-color:var(--ink-2);color:var(--ink)}',
      F + '.rv-pre[aria-pressed="true"]{border-color:var(--tide);color:var(--tide);background:var(--tide-soft)}',
      F + '.rv-krow{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .6rem;margin-top:.55rem}',
      F + '.rv-krow .seg button{padding:.3rem .62rem}',
      F + '.rv-none{font-family:var(--f-ui);font-size:.75rem;color:var(--ink-2)}',
      F + '.rv-chips{display:flex;flex-wrap:wrap;gap:.35rem .4rem}',
      F + '.rv-chip{display:inline-flex;align-items:center;gap:.4rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.01em;padding:.3rem .62rem .3rem .5rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink);cursor:pointer;white-space:nowrap}',
      F + '.rv-chip .sw{width:.66rem;height:.66rem;border-radius:50%;background:currentColor;flex:none;border:1.5px solid currentColor}',
      F + '.rv-chip .tx{color:var(--ink)}',
      F + '.rv-chip[aria-pressed="false"]{color:var(--ink-2)}',
      F + '.rv-chip[aria-pressed="false"] .sw{background:transparent}',
      F + '.rv-chip[aria-pressed="false"] .tx{color:var(--ink-2);text-decoration:line-through;text-decoration-thickness:1px}',
      F + '.rv-chip.ref{cursor:default}',
      F + '.rv-chip:hover:not(.ref){border-color:var(--ink-2)}',
      F + '.rv-main{display:grid;grid-template-columns:minmax(0,1fr);gap:1.1rem 1.6rem;align-items:start}',
      F + '.rv-chartbox{min-width:0;position:relative}',
      F + '.rv-chartbox svg{display:block;width:auto;max-width:100%;height:auto;overflow:visible}',
      F + '.rv-chartbox svg text{font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:var(--ink-2)}',
      F + '.rv-chartbox svg .ax{font-family:var(--f-ui);font-weight:500;font-size:12px;letter-spacing:.01em;fill:var(--ink-2)}',
      F + '.rv-chartbox svg .ann{font-family:var(--f-ui);font-weight:500;font-size:12px;font-variant-numeric:tabular-nums;fill:var(--ink-2)}',
      F + '.rv-chartbox svg .lbl{font-family:var(--f-body);font-size:14px;fill:var(--ink);paint-order:stroke;stroke:var(--paper-2);stroke-width:3.5px;stroke-linejoin:round}',
      F + '.rv-chartbox svg .lbl.ghost{font-family:var(--f-ui);font-weight:500;font-size:12px;fill:var(--ink-2)}',
      F + '.rv-chartbox svg .halo{paint-order:stroke;stroke:var(--paper-2);stroke-width:3.5px;stroke-linejoin:round}',
      F + '.rv-chartbox svg .pt{cursor:pointer}',
      F + '.rv-chartbox svg .pt:focus{outline:none}',
      F + '.rv-chartbox svg .pt:focus-visible .ring{stroke:var(--ochre);stroke-width:2.5px}',
      F + '.rv-mlegend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;margin-top:.4rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--ink-2);line-height:1.4}',
      F + '.rv-mlegend > span{display:inline-flex;align-items:center;gap:.4rem}',
      F + '.rv-mlegend i{display:inline-block;width:.62rem;height:.62rem;border-radius:50%;border:1.5px solid var(--ink-2)}',
      F + '.rv-mlegend i.f{background:var(--ink-2)}',
      F + '.rv-mlegend i.h{border-radius:2px;border:1px solid var(--rule);background:repeating-linear-gradient(135deg,var(--rule) 0 1.2px,transparent 1.2px 3.6px)}',
      F + '.rv-mlegend i.gh{width:.46rem;height:.46rem;border:1.4px solid var(--tide);background:transparent}',
      F + '.rv-mlegend i.it{display:inline;width:auto;height:auto;border:0;border-radius:0;font-family:var(--f-body);font-weight:400;font-size:1.12em}',
      F + '.rv-chartbox svg .ghost-pt{cursor:pointer}',
      F + '.rv-chartbox svg .ghost-pt:focus{outline:none}',
      F + '.rv-chartbox svg .ghost-pt:focus-visible circle{stroke:var(--ochre);stroke-width:2.2px}',
      F + '.rv-side{min-width:0;display:grid;gap:.85rem}',
      F + '.rv-budget{display:grid;gap:.1rem}',
      F + '.rv-big{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:clamp(1.55rem,1.1rem + 1.6cqi,2.05rem);line-height:1.05;color:var(--ink);letter-spacing:-.01em}',
      F + '.rv-big small{font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--ink-2);letter-spacing:.01em;margin-left:.4rem}',
      F + '.rv-ledger{display:grid;gap:.6rem}',
      F + '.rv-lhead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem .8rem;border-top:1px solid var(--rule);padding-top:.45rem}',
      F + '.rv-lkey{display:inline-flex;flex-wrap:wrap;gap:.2rem .8rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--ink-2)}',
      F + '.rv-lkey span{display:inline-flex;align-items:center;gap:.3rem}',
      F + '.rv-lkey i{display:inline-block;width:.9rem;height:.5rem;border-radius:1px;background:var(--ink-2)}',
      F + '.rv-lkey i.g{background:repeating-linear-gradient(135deg,var(--ink-3) 0 1.2px,transparent 1.2px 3.4px);outline:1px solid var(--rule)}',
      F + '.rv-row{display:grid;gap:.18rem;min-width:0}',
      F + '.rv-row.off{opacity:.55}',
      F + '.rv-rtop{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;min-width:0}',
      F + '.rv-nm{font-family:var(--f-body);font-size:.98rem;color:var(--ink);white-space:nowrap}',
      F + '.rv-nm .dot{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;margin-right:.4rem;vertical-align:.06em}',
      F + '.rv-nm sub{font-size:.68em}',
      F + '.rv-rk{font-family:var(--f-mono);font-size:.78rem;color:var(--ink);font-variant-numeric:tabular-nums;white-space:nowrap}',
      F + '.rv-bar{display:flex;height:9px;border-radius:2px;overflow:hidden;background:var(--paper);border:1px solid var(--rule)}',
      F + '.rv-bar .d{height:100%;transition:width .25s ease}',
      F + '.rv-bar .g{height:100%;background:repeating-linear-gradient(135deg,var(--ink-3) 0 1.2px,transparent 1.2px 3.4px);transition:width .25s ease}',
      F + '.rv-rsub{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums;line-height:1.45;overflow-wrap:anywhere}',
      F + '.rv-rsub sub,' + F + '.rv-rsub sup{font-size:.8em;line-height:0}',
      F + '.rv-redraw{font:inherit;font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--tide);background:none;border:0;padding:0 .1rem;cursor:pointer;text-decoration:underline;text-underline-offset:.15em}',
      F + '.rv-cards{display:grid;grid-template-columns:minmax(0,1fr);gap:.75rem;margin-top:1.2rem}',
      F + '.rv-card{min-width:0;border:1px solid var(--rule);border-radius:var(--radius);background:var(--paper);padding:.6rem .8rem .7rem;font-size:.86rem;line-height:1.5;color:var(--ink-2)}',
      F + '.rv-card h4{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.12rem;color:var(--ink);margin:0 0 .3rem;display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.25rem .6rem}',
      F + '.rv-card h4 sub{font-size:.62em}',
      F + '.rv-card p{margin:0}',
      F + '.rv-card b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}',
      F + '.rv-card .mono{font-family:var(--f-mono);font-size:.88em}',
      F + '.rv-v{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;padding:.12rem .45rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap}',
      F + '.rv-v.seal{color:var(--seal);background:var(--seal-soft)}',
      F + '.rv-v.ochre{color:var(--ochre);background:var(--ochre-soft)}',
      F + '.rv-v.moss{color:var(--moss);background:var(--moss-soft)}',
      F + '.rv-v.ink{color:var(--ink-2)}',
      F + '.rv-live{margin-top:1.2rem;border-top:1px solid var(--rule);padding-top:.6rem}',
      F + '.rv-livehead{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.4rem 1rem;margin-bottom:.45rem}',
      F + '.rv-livehead .btn{font-size:.75rem;padding:.32rem .6rem}',
      F + '.rv-where{font-size:.84rem;line-height:1.45;color:var(--ink-2);margin:0 0 .5rem;max-width:72ch}',
      F + '.rv-where b{color:var(--ink);font-weight:600}',
      F + '.rv-tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:var(--radius);background:var(--paper)}',
      F + '.rv-t{width:100%;border-collapse:collapse;font-family:var(--f-mono);font-size:.75rem;font-variant-numeric:tabular-nums}',
      F + '.rv-t th,' + F + '.rv-t td{padding:.34rem .55rem;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap;vertical-align:baseline}',
      F + '.rv-t th{position:static;background:var(--paper-2);font-family:var(--f-ui);font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-2);font-weight:500;white-space:normal;line-height:1.35;vertical-align:bottom}',
      F + '.rv-t tr:last-child td{border-bottom:0}',
      F + '.rv-t sup,' + F + '.rv-livesum sup{font-size:.78em;line-height:0;letter-spacing:0}',
      F + '.rv-t td:first-child,' + F + '.rv-t th:first-child{text-align:left}',
      F + '.rv-t td.m{font-family:var(--f-body);font-size:.88rem;color:var(--ink)}',
      F + '.rv-t td.m sub{font-size:.68em}',
      F + '.rv-t td.m small{display:block;font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2)}',
      F + '.rv-t .ok{color:var(--moss)}',
      F + '.rv-t .bad{color:var(--seal)}',
      F + '.rv-t .dim{color:var(--ink-2)}',
      F + '.rv-t .hide-n{}',
      F + '.rv-livesum{display:flex;flex-wrap:wrap;gap:.3rem 1rem;margin-top:.45rem;font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2)}',
      F + '.rv-livesum .ok{color:var(--moss)}',
      F + '.rv-livesum .bad{color:var(--seal)}',
      F + '.rv-foot{margin-top:1rem;padding-top:.55rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.82rem;line-height:1.55;color:var(--ink-2)}',
      '@container (max-width: 520px){' + F + '.rv-t .hide-n{display:none}' + F + '.rv-t th,' + F + '.rv-t td{padding:.32rem .4rem}}',
      '@container (min-width: 700px){' + F + '.rv-controls{grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1.05fr)}}',
      '@container (min-width: 760px){' + F + '.rv-main{grid-template-columns:minmax(0,1.6fr) minmax(0,1fr)}' + F + '.rv-cards{grid-template-columns:repeat(3,minmax(0,1fr))}}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-rankvol';
    s.textContent = css;
    document.head.appendChild(s);
  }

  Atlas.register('rankvol', function (el, A) {
    injectCSS();
    var d3 = window.d3, h = A.h;
    var uid = 'rv' + Math.random().toString(36).slice(2, 7);
    var NS = 'http://www.w3.org/2000/svg';

    var st = {
      mIdx: SIZES.indexOf(4096), nIdx: SIZES.indexOf(4096), R: 8, k: 2,
      on: { loha: true, gralora: true, hira: true, fourier: true, c3a: true },
      seed: 1, liveSeed: 1
    };
    var C = null, kSeg = null, kSegKey = null;

    /* ---------- DOM ---------- */
    var stage = h('div', { class: 'stage rv-stage', role: 'group', 'aria-label': 'Rank is not volume: at an equal parameter budget, a scatter of the maximum rank each PEFT method can reach against the dimension of its image, with a ledger of where the budget goes and a live Jacobian check.' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'rv-head' }, [
      h('h3', { class: 'rv-title', html: 'Rank is <i>not</i> volume' }),
      h('span', { class: 'rv-kicker', text: 'Gauge census · Prop II.12 · Thm II.10' })
    ]));
    stage.appendChild(h('p', { class: 'rv-instr', html: 'Drag the <b>budget</b> <i>R</i> or the <b>layer shape</b> <i>m</i> × <i>n</i>. Every dot spends exactly <i>R</i>(<i>m</i>+<i>n</i>) parameters. <b>Across</b> is the highest rank it can reach (a cut); <b>up</b> is the dimension of everything it can reach (a volume). Hover or focus a dot for its numbers.' }));
    var formula = h('div', { class: 'rv-formula', 'aria-label': 'Key formulas' }, [
      h('span', {}, [h('span', { class: 'rv-tag', html: lc('volume · II.12(a)') }), h('span', { html: '\\(d(M)=|M|-\\dim(\\text{generic fibre})\\)' }), h('span', { html: '\\(\\le |M|-\\dim(\\text{gauge orbit})\\)' })]),
      h('span', {}, [h('span', { class: 'rv-tag', text: 'cut · II.10' }), h('span', { html: '\\(\\operatorname{rank}\\Delta W\\le\\min_{\\text{cuts}}w\\)' })]),
      h('span', {}, [h('span', { class: 'rv-tag', html: lc('equal budget · II.12(c)') + ' <span class="nt">· 2r ≤ min(m,n)</span>' }), h('span', { html: '\\(d(\\mathrm{LoRA}_{2r})-d(\\mathrm{LoHa}_{r,r})\\ge(m+n-1)-2r^2\\)' })])
    ]);
    stage.appendChild(formula);

    /* controls */
    var controls = h('div', { class: 'rv-controls' });
    stage.appendChild(controls);
    function mkSlider(id, labelHtml, nm, min, max, val, onInput) {
      var input = h('input', { type: 'range', id: id, min: min, max: max, step: 1, value: val });
      var out = h('output', { for: id });
      var lab = h('label', { for: id, html: labelHtml + '<span class="nm">' + nm + '</span>' });
      input.addEventListener('input', function () { onInput(+input.value); });
      return { el: h('div', { class: 'rv-sl' }, [h('div', { class: 'rv-sl-top' }, [lab, out]), input]), input: input, out: out };
    }
    var gLayer = h('div', { class: 'rv-group' }, [h('div', { class: 'rv-gl' }, [h('span', { class: 'rv-tag', html: 'Layer <span class="nt">W ∈ ℝ<sup>m×n</sup></span>' }), h('span', { class: 'rv-tag', html: '<span class="nt">y = Wx</span>' })])]);
    var sl2 = h('div', { class: 'rv-sl2' });
    var sM = mkSlider(uid + '-m', 'm', 'd_out', 0, SIZES.length - 1, st.mIdx, function (v) { st.mIdx = v; schedule(); });
    var sN = mkSlider(uid + '-n', 'n', 'd_in', 0, SIZES.length - 1, st.nIdx, function (v) { st.nIdx = v; schedule(); });
    sl2.appendChild(sM.el); sl2.appendChild(sN.el);
    gLayer.appendChild(sl2);
    var presetBox = h('div', { class: 'rv-presets', role: 'group', 'aria-label': 'Layer presets' });
    var presetBtns = PRESETS.map(function (p) {
      var b = h('button', { type: 'button', class: 'rv-pre', 'aria-pressed': 'false', text: p.label, title: p.m + ' × ' + p.n + ', R = ' + p.R });
      b.addEventListener('click', function () { st.mIdx = SIZES.indexOf(p.m); st.nIdx = SIZES.indexOf(p.n); st.R = p.R; render(); });
      presetBox.appendChild(b);
      return b;
    });
    gLayer.appendChild(presetBox);
    controls.appendChild(gLayer);

    var gBudget = h('div', { class: 'rv-group' }, [h('div', { class: 'rv-gl' }, [h('span', { class: 'rv-tag', text: 'Equal budget' }), h('span', { class: 'rv-tag', html: '<span class="nt">|M| = R(m+n)</span>' })])]);
    var sR = mkSlider(uid + '-r', 'R', 'LoRA rank of the budget', 1, 64, st.R, function (v) { st.R = v; schedule(); });
    gBudget.appendChild(sR.el);
    var budgetSub = h('div', { class: 'rv-sub' });
    gBudget.appendChild(budgetSub);
    var kRow = h('div', { class: 'rv-krow' });
    gBudget.appendChild(kRow);
    controls.appendChild(gBudget);

    var gMeth = h('div', { class: 'rv-group' }, [h('div', { class: 'rv-gl' }, [h('span', { class: 'rv-tag', text: 'Methods' }), h('span', { class: 'rv-tag', text: 'click to hide' })])]);
    var chipBox = h('div', { class: 'rv-chips', role: 'group', 'aria-label': 'Show or hide methods' });
    var chips = {};
    ORDER.forEach(function (id) {
      var isRef = id === 'lora';
      var c = h(isRef ? 'span' : 'button', isRef ? { class: 'rv-chip ref', title: 'LoRA_R is the reference at this budget' } : { type: 'button', class: 'rv-chip', 'aria-pressed': 'true' }, [
        h('span', { class: 'sw', 'aria-hidden': 'true' }), h('span', { class: 'tx', text: META[id].name + (isRef ? ' (reference)' : '') })
      ]);
      c.style.color = 'var(' + META[id].token + ')';
      if (!isRef) c.addEventListener('click', function () { st.on[id] = !st.on[id]; c.setAttribute('aria-pressed', String(st.on[id])); render(); });
      chipBox.appendChild(c);
      chips[id] = c;
    });
    gMeth.appendChild(chipBox);
    controls.appendChild(gMeth);

    /* main: chart + side */
    var main = h('div', { class: 'rv-main' });
    stage.appendChild(main);
    var chartCol = h('div', { class: 'rv-chartbox' });
    main.appendChild(chartCol);
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('role', 'group');
    chartCol.appendChild(svg);
    var mlegend = h('div', { class: 'rv-mlegend', 'aria-hidden': 'true' });
    chartCol.appendChild(mlegend);
    function drawLegend(hasGhosts) {
      mlegend.innerHTML = '<span><i class="f"></i>max rank attained</span><span><i></i>max rank is an upper bound (Thm II.10)</span>' +
        (hasGhosts ? '<span><i class="gh"></i>GraLoRA at another <i class="it">k</i> (click to switch)</span>' : '') +
        '<span><i class="h"></i>out of bounds: d ≤ min(|M|, mn), rank ≤ min(m,n)</span>';
    }
    var side = h('div', { class: 'rv-side' });
    main.appendChild(side);
    var budgetBox = h('div', { class: 'rv-budget' });
    side.appendChild(budgetBox);
    var ledger = h('div', { class: 'rv-ledger' });
    side.appendChild(ledger);

    var cards = h('div', { class: 'rv-cards' });
    stage.appendChild(cards);

    var live = h('div', { class: 'rv-live' });
    stage.appendChild(live);
    var liveHead = h('div', { class: 'rv-livehead' });
    var liveBtn = h('button', { type: 'button', class: 'btn', text: 'New random point' });
    liveHead.appendChild(h('span', { class: 'rv-tag', text: 'Live check · Jacobian rank in float64' }));
    liveHead.appendChild(liveBtn);
    live.appendChild(liveHead);
    var liveWhere = h('p', { class: 'rv-where' });
    live.appendChild(liveWhere);
    var liveTable = h('div', { class: 'rv-tablewrap' });
    live.appendChild(liveTable);
    var liveSum = h('div', { class: 'rv-livesum', 'aria-live': 'polite' });
    live.appendChild(liveSum);
    liveBtn.addEventListener('click', function () { st.liveSeed++; runLive(true); });

    stage.appendChild(h('div', { class: 'rv-foot', html:
      'Budget R(m+n) is LoRA<sub>R</sub>’s parameter count on one m×n box. LoHa<sub>⌊R/2⌋,⌈R/2⌉</sub> splits it between its two Hadamard factors (HF LoHa uses one r for both, i.e. even R); GraLoRA uses k×k blocks of rank R/k; HiRA is ΔW = W<sub>0</sub> ⊙ BA with a generic W<sub>0</sub> (no zero entries, full rank; Prop II.9(a)); ' +
      'FourierFT has n<sub>c</sub> = |M| frequencies drawn uniformly without replacement (seeded), as HF’s randperm does; C3A appears only when a block size b with mn/b = |M| divides m and n. ' +
      'LoHa’s d is the proved upper bound of Prop II.12(b), with equality checked numerically there and live below. KronA’s budget m<sub>1</sub>n<sub>1</sub> + m<sub>2</sub>n<sub>2</sub> is fixed by its factor shapes, so it has no point at this budget; Figure 5 compares it with LoRA cut by cut. Every number is computed in the browser.' }));

    /* ---------- state helpers ---------- */
    function recompute() {
      var m = SIZES[st.mIdx], n = SIZES[st.nIdx];
      var Rmax = Math.min(m, n, R_CAP);
      if (st.R > Rmax) st.R = Rmax;
      if (st.R < 1) st.R = 1;
      C = census(A, m, n, st.R, st.k, st.seed);
      if (C.k) st.k = C.k;
    }
    function syncControls() {
      var m = C.m, n = C.n, Rmax = Math.min(m, n, R_CAP);
      sM.input.value = st.mIdx; sN.input.value = st.nIdx;
      sM.out.textContent = int(m); sN.out.textContent = int(n);
      sR.input.max = Rmax; sR.input.value = st.R;
      sR.out.textContent = String(st.R);
      budgetSub.innerHTML = '|M| = ' + st.R + ' · (' + int(m) + ' + ' + int(n) + ') = <b>' + int(C.P) + '</b> · ' + fracPct(C.P / C.mn) + ' of mn';
      presetBtns.forEach(function (b, i) { var p = PRESETS[i]; b.setAttribute('aria-pressed', String(p.m === m && p.n === n && p.R === st.R)); });
      // k selector (rebuilt only when the list of valid k changes, so keyboard focus survives a choice)
      var ksKey = C.ks.slice(0, 7).join(',');
      if (ksKey !== kSegKey) {
        kSegKey = ksKey; kSeg = null;
        kRow.innerHTML = '';
        kRow.appendChild(h('span', { class: 'rv-tag', html: 'GraLoRA <span class="nt">k × k</span> blocks' }));
        if (C.ks.length) {
          var opts = C.ks.slice(0, 7).map(function (k) { return { value: k, label: 'k=' + k }; });
          kSeg = A.seg(opts, C.k, function (v) { st.k = v; render(); }, 'GraLoRA block count k');
          kRow.appendChild(kSeg.el);
        } else kRow.appendChild(h('span', { class: 'rv-none', text: 'none: k ≥ 2 must divide m, n and R' }));
      } else if (kSeg) kSeg.set(C.k);
    }
    function fracPct(x) { return x >= 0.1 ? (100 * x).toFixed(0) + '%' : x >= 0.01 ? (100 * x).toFixed(1) + '%' : (100 * x).toFixed(2) + '%'; }
    function shown(id) { return id === 'lora' || st.on[id]; }
    function color(id) { return 'var(' + META[id].token + ')'; }
    function nameHTML(id, E) {
      var base = META[id].name;
      if (!E || !E.avail) return base;
      return base + (E.sub ? '<sub>' + E.sub + '</sub>' : '') + (E.tail ? '<span style="font-size:.86em"> ' + E.tail.trim().replace('k=', '<i>k</i>=').replace('b=', '<i>b</i>=') + '</span>' : '');
    }
    var FORM = {
      lora: 'ΔW = BA,  B ∈ ℝ<sup>m×R</sup>, A ∈ ℝ<sup>R×n</sup>',
      loha: 'ΔW = (B<sub>1</sub>A<sub>1</sub>) ⊙ (B<sub>2</sub>A<sub>2</sub>), inner ranks r<sub>1</sub>, r<sub>2</sub>',
      gralora: 'k×k grid of blocks B<sub>ij</sub>A<sub>ij</sub>, each of rank R/k',
      hira: 'ΔW = W<sub>0</sub> ⊙ (BA)',
      fourier: 'ΔW = Re ifft2(S), S supported on n<sub>c</sub> sampled frequencies',
      c3a: 'ΔW = [circ(w<sub>ij</sub>)], b×b circulant blocks'
    };
    var REF = {
      lora: 'Prop II.12(b)', loha: 'Prop II.12(b); d is a proved upper bound, equality [Num]', gralora: 'Prop II.12(b)',
      hira: 'Prop II.9(a), II.13; rank ≤ R·rk W<sub>0</sub> (Thm II.10)', fourier: 'Prop II.12(b); rank bound Thm II.10', c3a: 'Prop II.12(b), linear row'
    };
    function tipHTML(id) {
      var E = C.L[id];
      return '<div class="t">' + nameHTML(id, E) + '</div><div style="font-size:.86rem;color:var(--ink-2)">' + FORM[id] + '</div>' +
        '<div class="mono" style="font-size:.78rem;font-variant-numeric:tabular-nums;margin-top:.3rem;line-height:1.55">|M| ' + int(E.P) + ' · d ' + (id === 'loha' ? '≤ ' : '') + int(E.d) + '<br>max rank ' + (E.bound ? '≤ ' : '') + int(E.rank) +
        ' · fibre ' + (id === 'loha' ? '≥ ' : '') + int(E.P - E.d) + '<br>gauge ' + E.gauge + '</div><div style="font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--ink-2);margin-top:.25rem">' + REF[id] + '</div>';
    }

    /* ---------- chart ---------- */
    function sv(tag, attrs, parent) {
      var e = document.createElementNS(NS, tag);
      for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }
    /* SVG text; subscript digit runs become shifted tspans in the text's own font (the web fonts lack U+2080-2089) */
    function svText(parent, x, y, str, attrs) {
      var t = sv('text', Object.assign({ x: x, y: y }, attrs || {}), parent), parts = String(str).split(/([₀-₉]+)/);
      if (parts.length === 1) { t.textContent = str; return t; }
      parts.forEach(function (p, i) {
        if (!p) return;
        if (i % 2 === 0) { t.appendChild(document.createTextNode(p)); return; }
        sv('tspan', { 'baseline-shift': '-0.3em', style: 'font-size:72%' }, t).textContent = p.replace(/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 8320); });
      });
      return t;
    }
    function labelNode(parent, id, E, ghostK) {
      var t = sv('text', { class: ghostK ? 'lbl ghost' : 'lbl' }, parent);
      if (ghostK) { t.textContent = 'k=' + ghostK; return t; }
      var b = sv('tspan', {}, t); b.textContent = META[id].name;
      if (E.sub) { var s = sv('tspan', { 'baseline-shift': '-0.3em', style: 'font-size:10px' }, t); s.textContent = E.sub; }
      if (E.tail) { var tl = sv('tspan', { style: 'font-size:12.5px' }, t); tl.textContent = E.tail; }
      return t;
    }
    function candidates(it) {
      var g = it.r + 5, w = it.w, hh = it.h, px = it.px, py = it.py, base = hh * 0.34, out = {};
      out.r = { x0: px + g, x1: px + g + w, y0: py - hh / 2, y1: py + hh / 2, tx: px + g, ty: py + base, an: 'start' };
      out.l = { x0: px - g - w, x1: px - g, y0: py - hh / 2, y1: py + hh / 2, tx: px - g, ty: py + base, an: 'end' };
      out.t = { x0: px - w / 2, x1: px + w / 2, y0: py - g - hh, y1: py - g, tx: px, ty: py - g - hh / 2 + base, an: 'middle' };
      out.b = { x0: px - w / 2, x1: px + w / 2, y0: py + g, y1: py + g + hh, tx: px, ty: py + g + hh / 2 + base, an: 'middle' };
      out.br = { x0: px + 3, x1: px + 3 + w, y0: py + g - 2, y1: py + g - 2 + hh, tx: px + 3, ty: py + g - 2 + hh / 2 + base, an: 'start' };
      out.bl = { x0: px - 3 - w, x1: px - 3, y0: py + g - 2, y1: py + g - 2 + hh, tx: px - 3, ty: py + g - 2 + hh / 2 + base, an: 'end' };
      out.tr = { x0: px + 3, x1: px + 3 + w, y0: py - g + 2 - hh, y1: py - g + 2, tx: px + 3, ty: py - g + 2 - hh / 2 + base, an: 'start' };
      out.tl = { x0: px - 3 - w, x1: px - 3, y0: py - g + 2 - hh, y1: py - g + 2, tx: px - 3, ty: py - g + 2 - hh / 2 + base, an: 'end' };
      out.l2 = { x0: px - g - w, x1: px - g, y0: py + 4, y1: py + 4 + hh, tx: px - g, ty: py + 4 + hh / 2 + base, an: 'end' };
      out.l3 = { x0: px - g - w, x1: px - g, y0: py - 4 - hh, y1: py - 4, tx: px - g, ty: py - 4 - hh / 2 + base, an: 'end' };
      return it.cands.map(function (c) { return out[c]; });
    }
    function overlap(a, b) { return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1; }

    function drawChart() {
      A.tip.hide();   // the dot under a visible tooltip is about to be replaced
      var W = Math.max(280, Math.floor(chartCol.getBoundingClientRect().width || 600));   // drawn 1:1, never rescaled
      var narrow = W < 500;
      var H = Math.round(Math.max(300, Math.min(470, W * (narrow ? 1.0 : 0.84))));
      var pts = ORDER.filter(function (id) { return shown(id) && C.L[id].avail; }).map(function (id) { return C.L[id]; });
      var ceil = C.ceil, minD = Infinity;
      pts.forEach(function (E) { minD = Math.min(minD, E.d); });
      var lo0 = Math.min(minD, 0.9 * ceil);
      var yLo = Math.max(0, lo0 - 0.1 * (ceil - lo0)), yHi = ceil + (narrow ? 0.15 : 0.1) * (ceil - yLo);
      // the left margin fits the widest y tick label (11px mono, about 6.7px per character)
      var tickLen = 0;
      d3.scaleLinear().domain([yLo, yHi]).ticks(narrow ? 4 : 6).forEach(function (v) { if (v <= ceil) tickLen = Math.max(tickLen, int(v).length); });
      var mg = { l: Math.max(narrow ? 50 : 60, Math.ceil(12 + 6.7 * tickLen)), r: narrow ? 6 : 10, t: 30, b: 44 };
      var pw = W - mg.l - mg.r, ph = H - mg.t - mg.b;
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      svg.setAttribute('width', W); svg.setAttribute('height', H);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var x = d3.scaleLog().base(2).domain([0.7, C.mi * 1.6]).range([0, pw]);
      var y = d3.scaleLinear().domain([yLo, yHi]).range([ph, 0]);

      var defs = sv('defs', {}, svg);
      var pat = sv('pattern', { id: uid + '-hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
      sv('line', { x1: 0, y1: 0, x2: 0, y2: 5, style: 'stroke:var(--rule);stroke-width:1.3' }, pat);
      var clip = sv('clipPath', { id: uid + '-clip' }, defs);
      sv('rect', { x: -12, y: -6, width: pw + 24, height: ph + 12 }, clip);

      var g = sv('g', { transform: 'translate(' + mg.l + ',' + mg.t + ')' }, svg);
      // impossible regions
      var yc = y(ceil), xf = x(C.mi);
      sv('rect', { x: 0, y: 0, width: pw, height: Math.max(0, yc), fill: 'url(#' + uid + '-hatch)' }, g);
      sv('rect', { x: xf, y: yc, width: Math.max(0, pw - xf), height: ph - yc, fill: 'url(#' + uid + '-hatch)' }, g);
      // grid + y axis
      var yt = y.ticks(narrow ? 4 : 6).filter(function (v) { return v >= yLo && v <= ceil && Math.abs(v - Math.round(v)) < 1e-9 && (yLo === 0 || y(v) < ph - 9); });
      var tickBoxes = [];
      yt.forEach(function (v) {
        sv('line', { x1: 0, x2: xf, y1: y(v), y2: y(v), style: 'stroke:var(--rule);stroke-width:1;stroke-dasharray:2 3' }, g);
        var tt = svText(g, -7, y(v) + 3.5, int(v), { 'text-anchor': 'end' });
        tickBoxes.push({ x0: -7 - tt.getComputedTextLength() - 2, x1: -1, y0: y(v) - 6, y1: y(v) + 6 });
      });
      sv('line', { x1: 0, x2: 0, y1: 0, y2: ph, style: 'stroke:var(--ink-3);stroke-width:1' }, g);
      if (yLo > 0) { // axis break marks: the axis does not start at 0
        sv('path', { d: 'M-5,' + (ph - 3) + 'l10,-4M-5,' + (ph + 1) + 'l10,-4', style: 'stroke:var(--ink-3);stroke-width:1;fill:none' }, g);
      }
      // x axis
      sv('line', { x1: 0, x2: pw, y1: ph, y2: ph, style: 'stroke:var(--ink-3);stroke-width:1' }, g);
      var L = Math.floor(Math.log2(C.mi) + 1e-9), octPx = x(2) - x(1), step = Math.max(1, Math.ceil((narrow ? 26 : 30) / octPx));
      for (var e = 0; e <= L; e++) {
        var v = Math.pow(2, e), xx = x(v);
        sv('line', { x1: xx, x2: xx, y1: ph, y2: ph + 4, style: 'stroke:var(--ink-3);stroke-width:1' }, g);
        if (e % step === 0 && !(v === C.mi) && xf - xx > 6.8 * (String(v).length + String(C.mi).length) / 2 + 6) svText(g, xx, ph + 16, String(v), { 'text-anchor': 'middle' });
      }
      svText(g, xf, ph + 16, String(C.mi), { 'text-anchor': 'middle', style: 'fill:var(--ink);font-weight:500' });
      // axis titles
      svText(g, -mg.l + 2, -14, 'd(M) ↑  dimension of the image · a volume', { class: 'ax' });
      svText(g, pw, ph + 36, 'max rank of ΔW → a cut · log scale', { 'text-anchor': 'end', class: 'ax' });
      // full-rank line
      sv('line', { x1: xf, x2: xf, y1: yc, y2: ph, style: 'stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 3' }, g);
      var fr = svText(g, 0, 0, 'full rank', { transform: 'translate(' + (xf + 13) + ',' + (ph - 6) + ') rotate(-90)', class: 'halo ann' });
      fr.setAttribute('aria-hidden', 'true');
      // reference lines (their labels are placed last, in free space)
      sv('line', { x1: 0, x2: pw, y1: yc, y2: yc, style: 'stroke:var(--ink-2);stroke-width:1.1;stroke-dasharray:5 3' }, g);
      var yl = y(C.dLoRA);
      sv('line', { x1: 0, x2: xf, y1: yl, y2: yl, style: 'stroke:var(--ink);stroke-width:1;stroke-dasharray:1 3;opacity:.75' }, g);
      var obstacles = tickBoxes.slice();

      // points, dodged horizontally when two share a position (toward the interior)
      var items = pts.map(function (E) { return { id: E.id, E: E, px: x(E.rank), py: y(E.d), r: 6.5 }; });
      items.forEach(function (it, i) {
        it.tx = it.px;
        for (var guard = 0; guard < 8; guard++) {
          var hit = false;
          for (var j = 0; j < i; j++) if (Math.abs(items[j].px - it.px) < 12 && Math.abs(items[j].py - it.py) < 12) { hit = true; break; }
          if (!hit) break;
          it.px += it.tx > pw * 0.6 ? -13 : 13;
        }
      });
      var ghosts = [];
      if (shown('gralora') && C.L.gralora.avail) {
        var seen = {}; seen[C.L.gralora.rank] = 1;
        C.L.gralora.ghosts.forEach(function (gk) {
          if (seen[gk.rank]) return; seen[gk.rank] = 1;
          var gx = x(gk.rank), gy = y(C.dLoRA);
          for (var j = 0; j < items.length; j++) if (Math.abs(items[j].px - gx) < 11 && Math.abs(items[j].py - gy) < 11) return;
          ghosts.push({ k: gk.k, rank: gk.rank, px: gx, py: gy, r: 4 });
        });
      }
      ghosts.forEach(function (gh) {
        var gg = sv('g', { class: 'ghost-pt', tabindex: 0, role: 'button', 'aria-label': 'GraLoRA with k = ' + gh.k + ': maximum rank ' + gh.rank + ' at the same dimension. Activate to switch to k = ' + gh.k + '.' }, g);
        sv('circle', { cx: gh.px, cy: gh.py, r: gh.r + 6, style: 'fill:transparent' }, gg);
        sv('circle', { cx: gh.px, cy: gh.py, r: gh.r, style: 'fill:var(--paper-2);stroke:var(--tide);stroke-width:1.4;opacity:.85' }, gg);
        var tt = sv('title', {}, gg); tt.textContent = 'GraLoRA k = ' + gh.k + ' (click to switch)';
        var pick = function () { st.k = gh.k; render(); };
        gg.addEventListener('click', pick);
        gg.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); } });
      });
      drawLegend(ghosts.length > 0);
      // LoHa: equal-budget gap to LoRA_R's volume (Prop II.12(c)), drawn as a dimension bracket by the y axis
      var Eh = C.L.loha, gapDims = Eh.avail ? C.dLoRA - Eh.d : 0;
      if (shown('loha') && Eh.avail && gapDims !== 0) {
        var it0 = items.filter(function (it) { return it.id === 'loha'; })[0];
        var bx = 14;
        if (Math.abs(it0.px - bx) < 34) bx = it0.px + 20;
        var y1 = y(C.dLoRA), y2 = y(Eh.d), col = gapDims > 0 ? 'var(--seal)' : 'var(--moss)';
        if (Math.abs(y2 - y1) >= 6) {
          if (it0.px - bx > 14) sv('line', { x1: bx, x2: it0.px - 9, y1: y2, y2: y2, style: 'stroke:' + col + ';stroke-width:1;stroke-dasharray:1 3;opacity:.8' }, g);
          sv('path', { d: 'M' + (bx - 3.5) + ',' + y1 + 'h7M' + bx + ',' + y1 + 'V' + y2 + 'M' + (bx - 3.5) + ',' + y2 + 'h7', style: 'fill:none;stroke:' + col + ';stroke-width:1.3' }, g);
          obstacles.push({ x0: bx - 4, x1: bx + 4, y0: Math.min(y1, y2), y1: Math.max(y1, y2) });
          // LoHa's d is a proved upper bound (equality numerical), so a gap below LoRA is "at least"
          var gm = (y1 + y2) / 2, ag = Math.abs(gapDims), dimW = ag === 1 ? ' dim' : ' dims';
          var gTxt = gapDims > 0 ? '≥ ' + int(ag) + ' fewer' + dimW : '+' + int(ag) + dimW;
          if (Math.abs(y2 - y1) >= 30) {
            var gl = svText(g, bx + 7, gm - 3, gTxt, { class: 'halo ann', style: 'fill:' + col });
            var gl2 = svText(g, bx + 7, gm + 12, gapDims > 0 ? 'LoHa vs LoRA' + subDigits(C.R) : 'LoHa above LoRA' + subDigits(C.R) + ' (numerical)', { class: 'halo ann', style: 'fill:' + col + ';font-weight:400' });
            var gb = gl.getBBox(), gb2 = gl2.getBBox();
            obstacles.push({ x0: gb.x - 2, x1: Math.max(gb.x + gb.width, gb2.x + gb2.width) + 2, y0: gb.y - 1, y1: gb2.y + gb2.height + 1 });
          } else if (Math.abs(y2 - y1) >= 12) {
            var gs = svText(g, bx + 7, gm + 4, gTxt, { class: 'halo ann', style: 'fill:' + col });
            var gsb = gs.getBBox();
            obstacles.push({ x0: gsb.x - 2, x1: gsb.x + gsb.width + 2, y0: gsb.y - 1, y1: gsb.y + gsb.height + 1 });
          }
        }
      }
      // dodge guides + point marks
      var ptLayer = sv('g', {}, g);
      items.forEach(function (it) {
        if (Math.abs(it.px - it.tx) > 0.5) sv('line', { x1: it.tx, x2: it.px, y1: it.py, y2: it.py, style: 'stroke:' + color(it.id) + ';stroke-width:1;opacity:.55' }, ptLayer);
      });
      items.slice().reverse().forEach(function (it) {
        var E = it.E, gg = sv('g', { class: 'pt', tabindex: 0, role: 'img', 'aria-label': META[it.id].name + (E.sub ? ' ' + E.sub : '') + (E.tail || '') + ': maximum rank ' + (E.bound ? 'at most ' : '') + E.rank + ', image dimension ' + (it.id === 'loha' ? 'at most ' : '') + E.d + ' of budget ' + E.P }, ptLayer);
        sv('circle', { cx: it.px, cy: it.py, r: it.r + 6, style: 'fill:transparent' }, gg);
        if (E.bound) sv('circle', { class: 'ring', cx: it.px, cy: it.py, r: it.r - 0.5, style: 'fill:var(--paper-2);stroke:' + color(it.id) + ';stroke-width:2.2' }, gg);
        else sv('circle', { class: 'ring', cx: it.px, cy: it.py, r: it.r, style: 'fill:' + color(it.id) + ';stroke:var(--paper-2);stroke-width:1.5' }, gg);
        var show = function (ev) { A.tip.show(tipHTML(it.id), ev); };
        gg.addEventListener('mouseenter', show);
        gg.addEventListener('mousemove', function (ev) { A.tip.move(ev); });
        gg.addEventListener('mouseleave', function () { A.tip.hide(); });
        gg.addEventListener('focus', function () { var r = gg.getBoundingClientRect(); show({ clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }); });
        gg.addEventListener('blur', function () { A.tip.hide(); });
      });

      // labels: direct where there is room, otherwise a leader-line stack under the point
      // the vertical "full rank" label: at the foot of its line unless a dot sits there
      var frLen = fr.getBBox().width, frY = ph - 6;
      var nearFull = items.filter(function (it) { return Math.abs(it.px - xf) < 20; });
      var frHits = function (yb) { return nearFull.some(function (it) { return it.py + it.r + 2 > yb - frLen && it.py - it.r - 2 < yb; }); };
      if (frHits(frY)) {
        var frBest = null;
        for (var yb = ph - 6; yb - frLen > yc + 8; yb -= 4) if (!frHits(yb)) { frBest = yb; break; }
        if (frBest != null) frY = frBest; else { fr.remove(); frLen = 0; }
      }
      fr.setAttribute('transform', 'translate(' + (xf + 13) + ',' + frY + ') rotate(-90)');
      if (frLen) obstacles.push({ x0: xf + 2, x1: xf + 16, y0: frY - 2 - frLen, y1: frY });
      var placed = obstacles.slice();
      items.forEach(function (it) { placed.push({ x0: it.px - it.r - 1, x1: it.px + it.r + 1, y0: it.py - it.r - 1, y1: it.py + it.r + 1 }); });
      ghosts.forEach(function (gh) { placed.push({ x0: gh.px - 5, x1: gh.px + 5, y0: gh.py - 5, y1: gh.py + 5 }); });
      var bounds = { x0: 1, x1: pw + mg.r - 2, y0: -mg.t + 18, y1: ph - 3 };
      function free(c) {
        if (c.x0 < bounds.x0 || c.x1 > bounds.x1 || c.y0 < bounds.y0 || c.y1 > bounds.y1) return false;
        for (var j = 0; j < placed.length; j++) if (overlap(c, placed[j])) return false;
        return true;
      }
      function put(q, c) {
        var bg = document.createElementNS(NS, 'rect');
        bg.setAttribute('x', c.x0 - 1); bg.setAttribute('y', c.y0 + 1); bg.setAttribute('width', c.x1 - c.x0 + 2); bg.setAttribute('height', Math.max(0, c.y1 - c.y0 - 1));
        bg.setAttribute('rx', 2); bg.setAttribute('style', 'fill:var(--paper-2)'); bg.setAttribute('aria-hidden', 'true');
        q.node.parentNode.insertBefore(bg, q.node);
        q.node.setAttribute('x', c.tx); q.node.setAttribute('y', c.ty); q.node.setAttribute('text-anchor', c.an);
        q.node.setAttribute('aria-hidden', 'true');
        placed.push({ x0: c.x0 - 2, x1: c.x1 + 2, y0: c.y0, y1: c.y1 });
      }
      var CANDS = {
        lora: ['l', 'b', 'bl', 'br', 't', 'r'], loha: ['r', 'l', 'b', 'br', 'bl', 't'], gralora: ['b', 'br', 'r', 'bl', 't'],
        hira: ['l', 'b', 'r'], fourier: ['l', 'b', 'r'], c3a: ['l', 'b', 'r']
      };
      // members of a coincident cluster (same true position, dodged apart) are labelled through the leader stack
      items.forEach(function (it) {
        it.cluster = items.some(function (o) { return o !== it && Math.abs(o.tx - it.tx) < 6 && Math.abs(o.py - it.py) < 12; });
      });
      // cluster members go first (their leaders need clear columns), then the free labels, then any label left over
      var stack = [], rest = [], stack2 = [];
      items.forEach(function (it) {
        var q = { node: labelNode(g, it.id, it.E), px: it.px, py: it.py, r: it.r, cands: CANDS[it.id] };
        var bb = q.node.getBBox(); q.w = bb.width; q.h = Math.max(11, bb.height);
        (it.cluster ? stack : rest).push(q);
      });
      placeStack(stack);
      rest.forEach(function (q) {
        var cs = candidates(q), chosen = null;
        for (var i = 0; i < cs.length && !chosen; i++) if (free(cs[i])) chosen = cs[i];
        if (chosen) put(q, chosen); else stack2.push(q);
      });
      placeStack(stack2);
      function placeStack(list) { list.sort(function (a, b) { return a.px - b.px; }).forEach(function (q) {
        var base = q.h * 0.34, done = false;
        for (var dir = 1; dir >= -1 && !done; dir -= 2) {
          for (var k = 1; k <= 18 && !done; k++) {
            var ly = q.py + dir * (q.r + 2 + k * 14);
            var right = q.px - 6 - q.w > bounds.x0;
            var c = right ? { x0: q.px - 6 - q.w, x1: q.px - 6, tx: q.px - 6, an: 'end' } : { x0: q.px + 6, x1: q.px + 6 + q.w, tx: q.px + 6, an: 'start' };
            c.y0 = ly - q.h / 2; c.y1 = ly + q.h / 2; c.ty = ly + base;
            var lead = { x0: q.px - 1.5, x1: q.px + 1.5, y0: dir > 0 ? q.py + q.r + 1 : ly, y1: dir > 0 ? ly : q.py - q.r - 1 };
            var leadBad = false;
            for (var j = 0; j < placed.length; j++) if (overlap(lead, placed[j])) { leadBad = true; break; }
            if (!leadBad && free(c)) {
              var y0 = q.py + dir * (q.r + 1);
              sv('path', { d: 'M' + q.px + ',' + y0 + 'V' + ly + 'H' + (right ? q.px - 4 : q.px + 4), style: 'fill:none;stroke:var(--ink-3);stroke-width:.9' }, g);
              put(q, c);
              placed.push(lead);
              done = true;
            }
          }
        }
        if (!done) { var cs = candidates(q); put(q, cs[0]); }
      }); }
      ghosts.forEach(function (gh) {
        var q = { node: labelNode(g, 'gralora', null, gh.k), px: gh.px, py: gh.py, r: gh.r, cands: ['t', 'b', 'tr', 'br'] };
        var bb = q.node.getBBox(); q.w = bb.width; q.h = Math.max(10, bb.height);
        var cs = candidates(q), chosen = null;
        for (var i = 0; i < cs.length && !chosen; i++) if (free(cs[i])) chosen = cs[i];
        if (chosen) put(q, chosen); else q.node.remove();
      });
      // reference-line labels in the free stretch of each line
      function lineLabel(txt, rows, cls, mandatory) {
        var t = svText(g, 0, rows[0], txt, { class: 'halo ann', style: cls });
        var bb = t.getBBox(), w = bb.width, hh = bb.height, xs = [], f;
        [0.42, 0.54, 0.3, 0.66, 0.18].forEach(function (fr) { xs.push(fr * xf); });
        for (f = 0; f + w + 6 <= xf; f += 4) xs.push(f);
        for (var r = 0; r < rows.length; r++) for (var i = 0; i < xs.length; i++) {
          var x0 = xs[i], c = { x0: x0 - 2, x1: x0 + w + 2, y0: rows[r] - hh * 0.78, y1: rows[r] + hh * 0.25 };
          if (c.x0 < -1 || c.x1 > xf - 2) continue;
          if (free(c)) { t.setAttribute('x', x0); t.setAttribute('y', rows[r]); placed.push(c); return t; }
        }
        if (mandatory) { t.setAttribute('x', 2); return t; }
        t.remove(); return null;
      }
      lineLabel(C.P <= C.mn ? (narrow ? '|M| = ' + int(C.P) : '|M| = ' + int(C.P) + ' · no gauge') : (narrow ? 'mn = ' + int(C.mn) : 'mn = ' + int(C.mn) + ' · whole space'), [yc - 5, yc - 17], 'fill:var(--ink)', true);
      lineLabel('d(LoRA' + subDigits(C.R) + ') = ' + int(C.dLoRA), [yl + 13, yl + 26], 'fill:var(--ink-2)', false);
      // span annotation: one volume, many ranks (methods on LoRA_R's volume line)
      var onLine = items.filter(function (it) { return it.E.d === C.dLoRA; }), rMax = 0;
      onLine.forEach(function (it) { rMax = Math.max(rMax, it.E.rank); });
      if (onLine.length > 1 && rMax > C.R) {
        var xa = x(C.R) + 10, xb = x(rMax) - 10;
        var at = svText(g, (xa + xb) / 2, 0, 'same volume ' + int(C.dLoRA) + ' · max rank ' + int(C.R) + ' → ' + int(rMax), { 'text-anchor': 'middle', class: 'halo ann', style: 'font-weight:400' });
        var aw = at.getComputedTextLength();
        var okA = false;
        if (xb - xa > aw + 30) {
          for (var ay = yl + 44; ay < ph - 20 && !okA; ay += 12) {
            var abox = { x0: xa - 2, x1: xb + 2, y0: ay - 13, y1: ay + 4 };
            if (free(abox)) {
              at.setAttribute('y', ay - 4);
              var mid = (xa + xb) / 2;
              sv('path', { d: 'M' + xa + ',' + (ay - 7.5) + 'H' + (mid - aw / 2 - 6) + 'M' + (mid + aw / 2 + 6) + ',' + (ay - 7.5) + 'H' + xb + 'M' + (xa + 5) + ',' + (ay - 10.5) + 'L' + xa + ',' + (ay - 7.5) + 'L' + (xa + 5) + ',' + (ay - 4.5) + 'M' + (xb - 5) + ',' + (ay - 10.5) + 'L' + xb + ',' + (ay - 7.5) + 'L' + (xb - 5) + ',' + (ay - 4.5), style: 'fill:none;stroke:var(--ink-3);stroke-width:1' }, g);
              placed.push(abox); okA = true;
            }
          }
        }
        if (!okA) at.remove();
      }
      svg.setAttribute('aria-label', chartSummary());
    }
    function subDigits(n) { return String(n).replace(/[0-9]/g, function (c) { return '₀₁₂₃₄₅₆₇₈₉'[+c]; }); }
    function chartSummary() {
      var parts = ['Scatter at an equal budget of ' + int(C.P) + ' parameters on a ' + C.m + ' by ' + C.n + ' layer.'];
      ORDER.forEach(function (id) {
        if (!shown(id) || !C.L[id].avail) return;
        var E = C.L[id];
        parts.push(META[id].name + (E.sub ? ' ' + E.sub : '') + (E.tail || '') + ': max rank ' + (E.bound ? 'at most ' : '') + E.rank + ', dimension ' + (id === 'loha' ? 'at most ' : '') + E.d + '.');
      });
      return parts.join(' ');
    }

    /* ---------- side: budget readout + ledger ---------- */
    function drawSide() {
      budgetBox.innerHTML = '';
      budgetBox.appendChild(h('span', { class: 'rv-tag', text: 'Every method spends' }));
      budgetBox.appendChild(h('div', { class: 'rv-big', html: int(C.P) + '<small>parameters</small>' }));
      budgetBox.appendChild(h('div', { class: 'rv-sub', html: 'on a ' + int(C.m) + ' × ' + int(C.n) + ' box (mn = ' + int(C.mn) + ')' }));
      ledger.innerHTML = '';
      ledger.appendChild(h('div', { class: 'rv-lhead' }, [
        h('span', { class: 'rv-tag', text: 'Where the budget goes' }),
        h('span', { class: 'rv-lkey', html: '<span><i></i>d(M)</span><span><i class="g"></i>fibre ⊇ gauge orbit</span>' })
      ]));
      ORDER.forEach(function (id) {
        if (!shown(id)) return;
        var E = C.L[id];
        if (!E.avail) {
          ledger.appendChild(h('div', { class: 'rv-row off' }, [
            h('div', { class: 'rv-rtop' }, [h('span', { class: 'rv-nm', html: '<span class="dot" style="background:' + color(id) + '"></span>' + META[id].name }), h('span', { class: 'rv-rk', text: 'n/a' })]),
            h('div', { class: 'rv-rsub', text: E.why })
          ]));
          return;
        }
        var fd = E.d / E.P, fg = 1 - fd, fib = E.P - E.d;
        var subTxt = 'd ' + int(E.d) + ' · fibre ' + int(fib);
        // LoHa: d is the proved upper bound min(mn, F); equality is numerical (Prop II.12(b))
        if (id === 'loha') subTxt = 'd ≤ ' + int(E.d) + ' · fibre ≥ ' + int(fib) + (E.capped ? ' > gauge ' + int(E.gdim) + ' (d capped at mn)' : ' = dim ' + E.gauge + ' (' + int(E.gdim) + ')');
        else if (id === 'fourier') subTxt += ' = ' + int(E.pairs) + ' conjugate pair' + (E.pairs === 1 ? '' : 's') + ' ';
        else if (id === 'c3a') subTxt += ' · injective';
        else subTxt += (fib === E.gdim ? ' = dim ' : ' ≥ dim ') + E.gauge + ' (' + int(E.gdim) + ')';
        var row = h('div', { class: 'rv-row' }, [
          h('div', { class: 'rv-rtop' }, [
            h('span', { class: 'rv-nm', html: '<span class="dot" style="background:' + (E.bound ? 'transparent;border:1.6px solid ' + color(id) : color(id)) + '"></span>' + nameHTML(id, E) }),
            h('span', { class: 'rv-rk', text: 'rank ' + (E.bound ? '≤ ' : '') + int(E.rank) })
          ]),
          h('div', { class: 'rv-bar', role: 'img', 'aria-label': 'dimension ' + E.d + ' of ' + E.P + ' parameters; fibre ' + fib }, [
            h('span', { class: 'd', style: 'width:' + (100 * fd).toFixed(3) + '%;background:' + color(id) }),
            h('span', { class: 'g', style: 'width:' + (100 * fg).toFixed(3) + '%' })
          ]),
          h('div', { class: 'rv-rsub', html: subTxt })
        ]);
        if (id === 'fourier') {
          var rb = h('button', { type: 'button', class: 'rv-redraw', text: 'redraw', 'aria-label': 'Redraw the FourierFT frequencies' });
          rb.addEventListener('click', function () { st.seed++; render(); var nb = ledger.querySelector('.rv-redraw'); if (nb) nb.focus(); });
          row.lastChild.appendChild(rb);
        }
        ledger.appendChild(row);
      });
    }

    /* ---------- verdict cards (Prop II.12(c), Prop II.13) ---------- */
    function drawCards() {
      cards.innerHTML = '';
      var R = C.R, m = C.m, n = C.n, mi = C.mi, s1 = C.s - 1;
      var loraH = 'LoRA<sub>' + R + '</sub>';
      function card(titleHTML, chipCls, chipTxt, bodyHTML) {
        cards.appendChild(h('div', { class: 'rv-card' }, [
          h('h4', { html: '<span>' + titleHTML + '</span><span class="rv-v ' + chipCls + '">' + lc(chipTxt) + '</span>' }),
          h('p', { html: bodyHTML })
        ]));
      }
      // LoHa
      if (shown('loha')) {
        var E = C.L.loha;
        if (!E.avail) card('LoHa vs ' + loraH, 'ink', 'n/a', 'LoHa needs R ≥ 2: one inner rank per Hadamard factor.');
        else {
          // E.d = min(mn, F) is a proved upper bound on d(LoHa); equality is numerical (Prop II.12(b)).
          // So a positive gap is "at least", and a zero or negative gap rests on the numerical equality.
          var gap = C.dLoRA - E.d, lh = 'LoHa<sub>' + E.sub + '</sub>', r = E.r1, rr = E.r1 * E.r2, ag = Math.abs(gap);
          var plural = ag === 1 ? '' : 's';
          var gapTxt = gap > 0 ? 'at least <b>' + int(ag) + '</b> fewer dimension' + plural : gap < 0 ? '<b>' + int(ag) + '</b> more dimension' + plural : 'the same number of dimensions';
          var dims = ' (' + (gap > 0 ? 'at most ' : '') + int(E.d) + ' against ' + int(C.dLoRA) + ')';
          var liveHere = C.m <= LIVE_MAX && C.n <= LIVE_MAX;
          if (E.inC) {
            var formulaGap = s1 - 2 * r * r;
            if (r <= 2 && gap > 0) {
              card(lh + ' vs ' + loraH, 'seal', 'strictly dominated · II.12(c)',
                'Every ' + lh + ' update has rank ≤ <i>r</i>² = ' + rr + ' ≤ 2<i>r</i> = ' + R + ', so it is already a ' + loraH + ' update. Its image also has ' + gapTxt + dims + ', so Im ' + lh + ' ⊊ Im ' + loraH + '.');
            } else if (r <= 2) {
              card(lh + ' vs ' + loraH, 'ink', 'contained · II.12(b)',
                'Every ' + lh + ' update has rank ≤ <i>r</i>² = ' + rr + ' ≤ 2<i>r</i> = ' + R + ', so Im ' + lh + ' ⊆ Im ' + loraH + '. Both images are ' + int(E.d) + '-dimensional here (' +
                (E.capped ? 'the LoHa formula gives ' + int(E.F) + ', capped at mn = ' + int(C.mn) : 'by the LoHa formula') + '; equality for LoHa is checked numerically' + (liveHere ? ', live below' : '') + '), so the dimension count cannot make the inclusion strict at this size.');
            } else if (gap > 0) {
              card(lh + ' vs ' + loraH, 'ochre', 'trade · II.12(c)',
                'Same budget. LoHa reaches rank <i>r</i>² = <b>' + rr + '</b> &gt; 2<i>r</i> = ' + R + ', and its image has at least (<i>m</i>+<i>n</i>−1) − 2<i>r</i>² = ' + int(s1) + ' − ' + (2 * r * r) + ' = <b>' + int(ag) + '</b> fewer dimension' + plural + dims + '.');
            } else {
              card(lh + ' vs ' + loraH, 'moss', 'no trade · <span class="lc">m+n ≤ 2r²+1</span>',
                'Here (<i>m</i>+<i>n</i>−1) − 2<i>r</i>² = ' + signed(formulaGap) + ', so Prop II.12(c) gives no gap. ' + lh + ' reaches rank <b>' + int(E.rank) + '</b> &gt; ' + R + ', and its dimension formula gives <b>' + int(E.d) + '</b> against ' + int(C.dLoRA) +
                (liveHere ? ' (both checked live below). At this shape LoHa ' + (gap < 0 ? 'beats ' + loraH + ' on both counts.' : 'matches ' + loraH + ' on dimension and beats it on rank.')
                  : ', with equality checked numerically at smaller sizes. At this shape LoHa matches or beats ' + loraH + ' on dimension and beats it on rank.'));
            }
          } else {
            var why = E.r1 !== E.r2 ? 'odd R splits the budget as <i>r</i><sub>1</sub> = ' + E.r1 + ', <i>r</i><sub>2</sub> = ' + E.r2 : '<i>r</i>² = ' + (r * r) + ' &gt; min(<i>m</i>,<i>n</i>) = ' + int(mi);
            var body;
            if (rr <= R && gap > 0) body = 'Every update has rank ≤ <i>r</i><sub>1</sub><i>r</i><sub>2</sub> = ' + rr + ' ≤ ' + R + ', so Im ' + lh + ' ⊊ Im ' + loraH + ', with ' + gapTxt + dims + '.';
            else if (gap > 0) body = 'LoHa reaches rank <b>' + int(E.rank) + '</b> &gt; ' + R + ' with ' + gapTxt + dims + '.';
            else body = 'LoHa reaches rank <b>' + int(E.rank) + '</b>' + (E.rank > R ? ' &gt; ' : ' against ') + R + ' and, by its dimension formula, ' + gapTxt + dims + (E.capped ? ', capped at mn = ' + int(C.mn) : '') + '; equality is checked numerically' + (liveHere ? ', live below' : '') + '.';
            card(lh + ' vs ' + loraH, 'ink', 'values from II.12(b)', body + ' <span style="color:var(--ink-2);font-style:italic">Outside (c)’s hypotheses: ' + why + '.</span>');
          }
        }
      }
      // GraLoRA
      if (shown('gralora')) {
        var G = C.L.gralora;
        if (!G.avail) card('GraLoRA vs ' + loraH, 'ink', 'n/a', 'GraLoRA needs a block count <i>k</i> ≥ 2 dividing <i>m</i>, <i>n</i> and <i>R</i>; there is none for (' + int(m) + ', ' + int(n) + ', ' + R + ').');
        else {
          var gh = 'GraLoRA <i>k</i>=' + G.k;
          if (G.inC) {
            card(gh + ' vs ' + loraH, 'ochre', 'incomparable · II.12(c)',
              'Same budget, same dimension <b>' + int(C.dLoRA) + '</b>. GraLoRA reaches rank <i>kR</i> = <b>' + int(G.rank) + '</b>, which ' + loraH + ' cannot; but each of its ' + (G.k * G.k) + ' blocks has rank ≤ <i>R</i>/<i>k</i> = ' + (R / G.k) + ', while a generic rank-' + R + ' update has rank ' + R + ' on every block. Neither image contains the other.');
          } else {
            card(gh + ' vs ' + loraH, 'ink', 'rank saturates',
              '<i>kR</i> = ' + int(G.k * R) + ' &gt; min(<i>m</i>,<i>n</i>) = ' + int(mi) + ', so GraLoRA’s maximum rank is ' + int(G.rank) + ', at LoRA’s dimension ' + int(C.dLoRA) + '. Prop II.12(c) states incomparability for <i>kR</i> ≤ min(<i>m</i>,<i>n</i>).');
          }
        }
      }
      // HiRA
      if (shown('hira')) {
        card('HiRA<sub>' + R + '</sub> vs ' + loraH, 'moss', 'same volume · II.13',
          'ΔW = W<sub>0</sub> ⊙ BA is LoRA transported by the linear automorphism X ↦ W<sub>0</sub> ⊙ X, which depends on W<sub>0</sub> and needs it to have no zero entries (Prop II.9(a)). Transport keeps <i>d</i> = <b>' + int(C.dLoRA) + '</b> and the gauge GL<sub>' + R + '</sub>, but not the rank: here up to <b>' + int(mi) + '</b> for generic W<sub>0</sub> (already W<sub>0</sub> ⊙ uv<sup>⊤</sup> = diag(u) W<sub>0</sub> diag(v) has rank rk W<sub>0</sub>).');
      }
    }

    /* ---------- live check ---------- */
    var liveToken = 0, liveTimer = null;
    function liveCopy() {
      if (C.m <= LIVE_MAX && C.n <= LIVE_MAX) return { m: C.m, n: C.n, own: true };
      if (C.R > LIVE_MAX) return null;
      var best = null;
      for (var t = LIVE_MAX; t >= Math.max(C.R, 2); t--) {
        if (C.k && t % C.k) continue;
        var sc = 2 * C.R <= t ? 1 : 0;
        if (!best || sc > best.sc) best = { t: t, sc: sc };
        if (sc) break;
      }
      return best ? { m: best.t, n: best.t, own: false } : null;
    }
    function liveSpecs(cp) {
      var m = cp.m, n = cp.n, R = C.R, mn = m * n, mi = Math.min(m, n), s = m + n;
      var C2 = census(A, m, n, R, C.k, st.seed), specs = [], rnd = A.rng(9001 + 131 * st.liveSeed);
      specs.push({ id: 'lora', html: 'LoRA<sub>' + R + '</sub>', map: mapLoRA(m, n, R), d: R * (s - R), rank: R, bound: false });
      if (shown('loha') && R >= 2) {
        var E = C2.L.loha;
        specs.push({ id: 'loha', html: 'LoHa<sub>' + E.sub + '</sub>', map: mapLoHa(m, n, E.r1, E.r2), d: E.d, rank: E.rank, bound: false, note: 'upper bound' });
      }
      if (shown('gralora') && C2.k && C.k && C2.k === C.k) specs.push({ id: 'gralora', html: 'GraLoRA <i>k</i>=' + C.k, map: mapGraLoRA(m, n, R, C.k), d: R * (s - R), rank: Math.min(C.k * R, m, n), bound: false });
      if (shown('hira')) specs.push({ id: 'hira', html: 'HiRA<sub>' + R + '</sub>', map: mapHiRA(m, n, R, normals(rnd, mn)), d: R * (s - R), rank: mi, bound: false });
      if (shown('fourier')) {
        var nc = Math.min(R * s, mn), dr = fourierDraw(A, m, n, nc, st.seed, true);
        specs.push({ id: 'fourier', html: 'FourierFT', extra: 'n<sub>c</sub> = ' + nc + (nc < R * s ? ' (all)' : ''), map: mapFourier(m, n, dr.idx), d: nc - dr.pairs, rank: Math.min(2 * nc, m, n), bound: true });
      }
      if (shown('c3a')) {
        var b = C2.L.c3a.avail ? C2.L.c3a.b : 0, g = gcd(m, n);
        if (!b) { b = 1; for (var dv = 2; dv <= g; dv++) if (g % dv === 0) { b = dv; break; } }
        specs.push({ id: 'c3a', html: 'C3A', extra: '<i>b</i> = ' + b + (C2.L.c3a.avail ? '' : ', own budget'), map: mapC3A(m, n, b), d: mn / b, rank: mi, bound: false });
      }
      specs.forEach(function (sp) { sp.q = normals(rnd, sp.map.P); sp.P = sp.map.P; });
      return { specs: specs, m: m, n: n };
    }
    function scheduleLive() {
      clearTimeout(liveTimer);
      liveTimer = setTimeout(function () { runLive(false); }, 380);
    }
    function runLive(force) {
      var token = ++liveToken;
      var cp = liveCopy();
      liveTable.innerHTML = ''; liveSum.innerHTML = '';
      if (!cp) {
        liveWhere.innerHTML = 'A live check needs a layer at least <i>R</i> = ' + C.R + ' wide, and dense Jacobians stop at ' + LIVE_MAX + ' × ' + LIVE_MAX + ' here. Lower <i>R</i> to ' + LIVE_MAX + ' or less to run it.';
        liveBtn.disabled = true;
        return;
      }
      liveBtn.disabled = false;
      var L = liveSpecs(cp), mn = L.m * L.n;
      var cost = 0; L.specs.forEach(function (sp) { cost += mn * sp.P * Math.min(mn, sp.P); });
      liveWhere.innerHTML = (cp.own ? 'On <b>this layer</b> (' + L.m + ' × ' + L.n + ')' :
        'On a <b>' + L.m + ' × ' + L.n + ' copy</b> with the same <i>R</i> = ' + C.R + (C.k ? ' and <i>k</i> = ' + C.k : '') + '; your layer’s Jacobian would be ' + int(C.mn) + ' × ' + int(C.P) + '. The census formulas are the same polynomials in <i>m</i>, <i>n</i>, <i>R</i>') +
        '. Each parametrisation is rebuilt and differentiated at a seeded random point; <i>d</i> is the numerical rank of its Jacobian (pivoted QR), and the max rank is the rank of ΔW there.';
      var tbl = h('table', { class: 'rv-t' });
      tbl.innerHTML = '<thead><tr><th>method</th><th class="hide-n">|M|</th><th>d: formula → Jacobian</th><th>max rank: formula → ΔW</th><th class="hide-n">pivot drop at d</th></tr></thead>';
      var tb = h('tbody'); tbl.appendChild(tb);
      liveTable.appendChild(tbl);
      var rows = L.specs.map(function (sp) {
        var tr = h('tr');
        tr.innerHTML = '<td class="m"><span style="display:inline-block;width:.5rem;height:.5rem;border-radius:50%;margin-right:.35rem;background:' + color(sp.id) + '"></span>' + sp.html + (sp.extra ? '<small>' + sp.extra + '</small>' : '') + '</td>' +
          '<td class="hide-n dim">' + int(sp.P) + '</td><td>' + int(sp.d) + ' <span class="dim">→</span> <span class="dim">…</span></td><td>' + (sp.bound ? '≤ ' : '') + int(sp.rank) + ' <span class="dim">→</span> <span class="dim">…</span></td><td class="hide-n dim">…</td>';
        tb.appendChild(tr);
        return tr;
      });
      if (!force && cost > 6e8) {
        liveSum.innerHTML = '<span>Estimated ' + (cost / 1e9).toFixed(1) + ' GFLOP: press “New random point” to run.</span>';
        return;
      }
      var i = 0, okAll = 0, t0 = (window.performance || Date).now();
      function step() {
        if (token !== liveToken) return;
        if (i >= L.specs.length) {
          var ms = Math.round((window.performance || Date).now() - t0);
          liveSum.innerHTML = '<span class="' + (okAll === L.specs.length ? 'ok' : 'bad') + '">' + okAll + ' / ' + L.specs.length + ' census rows reproduced</span><span>tolerance 10<sup>−9</sup> · ' + ms + ' ms · point #' + st.liveSeed + '</span>';
          return;
        }
        var sp = L.specs[i], tr = rows[i];
        var J = jacobian(sp.map, sp.q, mn);
        var rk = numRank(qrcpDiag(J, mn, sp.P));
        var Dv = new Float64Array(mn); sp.map.f(sp.q, Dv);
        var Dc = new Float64Array(mn); // column-major copy of the m x n matrix
        for (var a = 0; a < L.m; a++) for (var b2 = 0; b2 < L.n; b2++) Dc[b2 * L.m + a] = Dv[a * L.n + b2];
        var rD = numRank(qrcpDiag(Dc, L.m, L.n)).rank;
        var okD = rk.rank === sp.d, okR = sp.bound ? rD <= sp.rank : rD === sp.rank;
        if (okD && okR) okAll++;
        var tds = tr.children;
        var sD = tds[2].lastChild, sR2 = tds[3].lastChild;
        sD.className = okD ? 'ok' : 'bad'; sD.textContent = int(rk.rank) + (okD ? ' ✓' : ' ✗');
        sR2.className = okR ? 'ok' : 'bad'; sR2.textContent = int(rD) + (okR ? ' ✓' : ' ✗');
        tds[4].innerHTML = hsup(sci(rk.last) + ' → ' + (rk.next ? sci(rk.next) : '—'));
        i++;
        setTimeout(step, 0);
      }
      setTimeout(step, 0);
    }

    /* ---------- render loop ---------- */
    var pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      (window.requestAnimationFrame || setTimeout)(function () { pending = false; render(); });
    }
    function render() {
      recompute();
      syncControls();
      drawChart();
      drawSide();
      drawCards();
      scheduleLive();
    }
    if (!d3) { stage.appendChild(h('p', { class: 'rv-where', text: 'D3 did not load, so the chart cannot be drawn.' })); return; }
    render();
    A.typeset(formula);
    A.onTheme(function () { drawChart(); });
    if ('ResizeObserver' in window) {
      var lastW = chartCol.clientWidth;
      new ResizeObserver(A.debounce(function () { if (Math.abs(chartCol.clientWidth - lastW) > 2) { lastW = chartCol.clientWidth; drawChart(); } }, 80)).observe(chartCol);
    } else window.addEventListener('resize', A.debounce(drawChart, 120));

    /* test hook: state and census for headless verification */
    el.__rankvol = { st: st, C: function () { return C; }, render: render, census: function (m, n, R, k, seed) { return census(A, m, n, R, k, seed || 1); } };
  });
})();
