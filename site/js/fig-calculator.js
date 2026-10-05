/* fig-calculator.js — "The budget calculator": parameters trained versus directions reachable.
   Every number on screen is computed here from (i) a base model's config.json and (ii) the parametrisation
   that the method's paper / Hugging Face PEFT actually instantiates. Sources are listed in SOURCES below and
   rendered in the figure. Notation follows the framework: W in R^{m x n}, y = Wx, m = d_out, n = d_in. */
(function () {
  'use strict';

  /* =====================================================================================
     1. Base models. Numbers from each repo's config.json (fetched 2026-10-01).
        hfTotal = the official repo's safetensors parameter count reported by the HF API, less non-parameter
        buffers stored in the checkpoint (buffers: Llama-2's 32 x 64 F32 rotary inv_freq entries);
        totalParams() must reproduce it exactly (checked on screen).
     ===================================================================================== */
  var MODELS = [
    { key: 'llama3-8b', label: 'Meta-Llama-3-8B', repo: 'meta-llama/Meta-Llama-3-8B', type: 'llama',
      d: 4096, ff: 14336, L: 32, heads: 32, kv: 8, V: 128256, tie: false, qkvBias: false, hfTotal: 8030261248 },
    { key: 'llama2-7b', label: 'Llama-2-7b-hf', repo: 'meta-llama/Llama-2-7b-hf', type: 'llama',
      d: 4096, ff: 11008, L: 32, heads: 32, kv: 32, V: 32000, tie: false, qkvBias: false, hfTotal: 6738415616, buffers: 2048 },
    { key: 'mistral-7b', label: 'Mistral-7B-v0.1', repo: 'mistralai/Mistral-7B-v0.1', type: 'mistral',
      d: 4096, ff: 14336, L: 32, heads: 32, kv: 8, V: 32000, tie: false, qkvBias: false, hfTotal: 7241732096 },
    { key: 'qwen25-7b', label: 'Qwen2.5-7B', repo: 'Qwen/Qwen2.5-7B', type: 'qwen2',
      d: 3584, ff: 18944, L: 28, heads: 28, kv: 4, V: 152064, tie: false, qkvBias: true, hfTotal: 7615616512 },
    { key: 'qwen25-05b', label: 'Qwen2.5-0.5B', repo: 'Qwen/Qwen2.5-0.5B', type: 'qwen2',
      d: 896, ff: 4864, L: 24, heads: 14, kv: 2, V: 151936, tie: true, qkvBias: true, hfTotal: 494032768 }
  ];
  var MOD_KEYS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
  var PRESETS = { qv: ['q', 'v'], attn: ['q', 'k', 'v', 'o'], all: MOD_KEYS.slice() };

  function headDim(M) { return M.d / M.heads; }            // none of these configs sets head_dim
  function kvDim(M) { return M.kv * headDim(M); }
  /* the seven linear maps of one decoder layer, as (m = d_out) x (n = d_in) */
  function arch(M) {
    var q = M.heads * headDim(M), kv = kvDim(M);
    return [
      { key: 'q', name: 'q_proj', m: q, n: M.d, bias: M.qkvBias },
      { key: 'k', name: 'k_proj', m: kv, n: M.d, bias: M.qkvBias },
      { key: 'v', name: 'v_proj', m: kv, n: M.d, bias: M.qkvBias },
      { key: 'o', name: 'o_proj', m: M.d, n: q, bias: false },
      { key: 'gate', name: 'gate_proj', m: M.ff, n: M.d, bias: false },
      { key: 'up', name: 'up_proj', m: M.ff, n: M.d, bias: false },
      { key: 'down', name: 'down_proj', m: M.d, n: M.ff, bias: false }
    ];
  }
  function modByKey(M, k) { var a = arch(M); for (var i = 0; i < a.length; i++) if (a[i].key === k) return a[i]; return null; }
  /* N = V d (x2 if untied) + L [ sum of the seven maps (+ q,k,v biases) + two RMSNorm gains ] + final norm */
  function totalParams(M) {
    var per = 2 * M.d;
    arch(M).forEach(function (mo) { per += mo.m * mo.n + (mo.bias ? mo.m : 0); });
    return M.V * M.d * (M.tie ? 1 : 2) + M.L * per + M.d;
  }

  /* =====================================================================================
     2. Exact ports of the HF PEFT helpers that change counts
     ===================================================================================== */
  /* peft/tuners/lokr/layer.py: factorization(dimension, factor=-1) -> (m, n), m <= n, m*n = dimension */
  function factorization(dimension, factor) {
    if (factor > 0 && dimension % factor === 0) return [factor, dimension / factor];
    if (factor === -1 || factor == null) factor = dimension;
    var m = 1, n = dimension, length = m + n;
    while (m < n) {
      var nm = m + 1;
      while (dimension % nm !== 0) nm++;
      var nn = dimension / nm;
      if (nm + nn > length || nm > factor) break;
      m = nm; n = nn;
    }
    if (m > n) { var t = m; m = n; n = t; }
    return [m, n];
  }
  /* peft/tuners/oft/layer.py: OFTLayer.adjust_oft_parameters(in_features, params) */
  function adjustOft(inF, p) {
    var hi;
    if (p < inF) { hi = p; while (hi <= inF && inF % hi !== 0) hi++; } else return inF;
    var lo = p;
    while (lo > 1 && inF % lo !== 0) lo--;
    return (p - lo) <= (hi - p) ? lo : hi;
  }
  /* dimension of the determinantal variety M_{<=r}(m x n) */
  function coneDim(r, m, n) { var k = Math.min(r, m, n); return k * (m + n - k); }

  /* =====================================================================================
     3. Methods. per(m, n, K, mo, M) returns, for ONE adapted m x n matrix:
        P = trainable parameters |M|, d = expressive dimension (null where the framework
        gives no closed form), s = structure string, bound = d is an upper bound.
     ===================================================================================== */
  var STEP_RANK = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  var STEP_B = [2, 4, 8, 16, 32, 64, 128, 256];
  var STEP_F = [1, 2, 3, 4, 5, 6];
  var STEP_NF = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
  var STEP_L = [1, 5, 10, 20, 50, 100, 200, 500];
  var STEP_HR = [4, 8, 16, 32, 64, 128, 256, 512];
  /* FourierFT: PEFT samples its frequencies as torch.randperm(m*n, generator=manual_seed(777))[:n_f] on the
     (out, in) grid, the same set for every layer of one shape. Re ifft2 identifies a conjugate pair (u,v), (-u,-v),
     so d = n_f - c with c the number of such pairs inside the sampled set (Prop II.12(b)). c below was computed
     with torch.randperm exactly as PEFT calls it (torch 2.13; the same call gives the 2 pairs at 768x768, n_f = 1000,
     that the referees report), for every m x n shape of the five models and every n_f in STEP_NF. */
  var FOURIER_PAIRS = {
    '128x896': [0, 0, 2, 3, 24, 97, 417, 1744, 10976, 43597],
    '512x3584': [0, 0, 0, 1, 2, 6, 31, 97, 683, 2743],
    '896x896': [0, 0, 0, 0, 2, 12, 59, 240, 1558, 6161],
    '896x4864': [0, 0, 0, 0, 0, 3, 10, 44, 262, 1146],
    '1024x4096': [0, 0, 0, 0, 0, 1, 6, 40, 284, 1154],
    '3584x3584': [0, 0, 0, 0, 0, 0, 4, 16, 100, 393],
    '3584x18944': [0, 0, 0, 0, 0, 1, 1, 3, 14, 64],
    '4096x4096': [0, 0, 0, 0, 0, 0, 5, 15, 69, 289],
    '4096x11008': [0, 0, 0, 0, 0, 0, 0, 3, 29, 102],
    '4096x14336': [0, 0, 0, 0, 0, 0, 0, 5, 24, 87],
    '4864x896': [0, 0, 0, 0, 0, 0, 11, 57, 298, 1162],
    '11008x4096': [0, 0, 0, 0, 0, 2, 4, 11, 30, 109],
    '14336x4096': [0, 0, 0, 0, 0, 0, 0, 1, 22, 85],
    '18944x3584': [0, 0, 0, 0, 0, 0, 1, 2, 17, 75]
  };
  function fourierPairs(m, n, nf) {
    var row = FOURIER_PAIRS[m + 'x' + n], i = STEP_NF.indexOf(nf);
    return row && i >= 0 && row[i] != null ? row[i] : null;
  }
  /* expected number of conjugate pairs among n_f distinct uniform indices of an m x n grid */
  function expectedPairs(m, n, nf) {
    var N = m * n, self = (m % 2 ? 1 : 2) * (n % 2 ? 1 : 2);
    if (N < 2) return 0;
    return (nf * (1 - self / N)) * (nf - 1) / (N - 1) / 2;
  }
  function rankKnob() { return { k: 'r', sym: 'r', label: 'rank r', steps: STEP_RANK, min: 1, max: 4096, def: 8, rank: true }; }
  function fmtShape(m, n) { return m + '×' + n; }
  function plural(n, one) { return n + ' ' + one + (n === 1 ? '' : 's'); }

  var METHODS = [
    { id: 'lora', name: 'LoRA', alias: 'also rsLoRA, PiSSA', data: ['lora', 'rslora', 'pissa'], scope: 'sel', knobs: [rankKnob()],
      per: function (m, n, K) {
        var r = K.r;
        return { P: r * (m + n), d: coneDim(r, m, n), s: 'A ' + fmtShape(r, n) + ', B ' + fmtShape(m, r) };
      },
      tex: { P: '\\lvert M\\rvert = r(m+n)', d: 'd(M) = k(m+n-k),\\ \\ k = \\min(r,m,n)', g: '\\lvert M\\rvert - d(M) = r^2\\ \\ (r \\le \\min(m,n))' },
      srcP: 'PEFT tuners/lora/layer.py (lora_A, lora_B; lora_bias=False). use_rslora and init_lora_weights="pissa" change scaling / initialisation only.',
      srcD: 'Thm II.2(b): the same d in every initialisation stratum (zero-init, PiSSA split) for r < min(m,n); d = mn once r ≥ min(m,n). rsLoRA ≅ LoRA (rule A3).' },

    { id: 'dora', name: 'DoRA', data: ['dora'], scope: 'sel', knobs: [rankKnob()],
      per: function (m, n, K) {
        var r = K.r, P = r * (m + n) + m, d;
        /* Prop II.12(b) / Prop II.8: d = min(mn, r(m+n-r)+m) for generic W0, checked numerically [Num] */
        if (r >= Math.min(m, n)) d = m * n;
        else d = Math.min(m * n, r * (m + n - r) + m);
        return { P: P, d: d, num: r < Math.min(m, n), s: 'LoRA + magnitude ' + m };
      },
      tex: { P: '\\lvert M\\rvert = r(m+n)+m', d: 'd(M) = \\min\\big(mn,\\ r(m+n-r)+m\\big)', g: '\\lvert M\\rvert - d(M) = r^2\\ \\ \\text{(below the cap)}' },
      srcP: 'PEFT tuners/lora/dora.py: the magnitude is the row norm ‖W‖ (dim=1), one entry per output, size m.',
      srcD: 'Prop II.8: DoRA reaches exactly what rank-r LoRA followed by a free per-output scale reaches. Prop II.12(b): that set has dimension min(mn, r(m+n−r)+m) for generic W₀, checked numerically [Num]; special W₀ give less.' },

    { id: 'vera', name: 'VeRA', data: ['vera'], scope: 'sel',
      knobs: [{ k: 'r', sym: 'r', label: 'shared rank r', steps: STEP_RANK, min: 1, max: 4096, def: 256 }],
      per: function (m, n, K) {
        var r = K.r, ok = r <= Math.min(m, n);
        return { P: m + r, d: ok ? m + r - 1 : null, num: ok, why: ok ? null : 'r > min(m,n) = ' + Math.min(m, n) + ', outside the range of Prop II.12', s: 'λ_b ' + m + ', λ_d ' + r };
      },
      tex: { P: '\\lvert M\\rvert = m+r', d: 'd(M) = m+r-1', g: '\\lvert M\\rvert - d(M) = 1' },
      srcP: 'PEFT tuners/vera/layer.py: vera_lambda_b ∈ ℝ^m, vera_lambda_d ∈ ℝ^r; the shared random A, B are frozen buffers.',
      srcD: 'Prop II.12(b), VeRA row (scalar gauge ℝ^×), checked numerically [Num]; applied for r ≤ min(m,n).' },

    { id: 'loha', name: 'LoHa', data: ['loha'], scope: 'sel', knobs: [rankKnob()], bound: true,
      per: function (m, n, K) {
        /* Prop II.12(b): d <= min(mn, 2r(m+n-r) - (m+n-1)) for r <= min(m,n), a proved upper bound
           (equality checked numerically). Past min(m,n) only the trivial bound d <= mn is claimed. */
        var r = K.r, P = 2 * r * (m + n), d;
        if (r <= Math.min(m, n)) d = Math.min(m * n, 2 * r * (m + n - r) - (m + n - 1));
        else d = m * n;
        return { P: P, d: d, bound: true, s: '2 × (' + fmtShape(m, r) + ', ' + fmtShape(r, n) + ')' };
      },
      tex: { P: '\\lvert M\\rvert = 2r(m+n)', d: 'd(M) \\le \\min\\big(mn,\\ 2r(m+n-r) - (m+n-1)\\big)', g: '\\lvert M\\rvert - d(M) \\ge 2r^2 + m + n - 1\\ \\ (r \\le \\min(m,n))' },
      srcP: 'PEFT tuners/loha/layer.py (Linear): hada_w1_a, hada_w2_a ∈ ℝ^{m×r}; hada_w1_b, hada_w2_b ∈ ℝ^{r×n}.',
      srcD: 'Prop II.12(b), LoHa row with r₁ = r₂ = r ≤ min(m,n): an upper bound from the gauge GL_r × GL_r × diagonal torus, with equality checked numerically. For r > min(m,n) only d ≤ mn is shown.' },

    { id: 'lokr', name: 'LoKr', data: ['lokr'], scope: 'sel', knobs: [rankKnob()],
      per: function (m, n, K) {
        var r = K.r, fo = factorization(m, -1), fi = factorization(n, -1);
        var a = fo[0], b = fo[1], c = fi[0], e = fi[1];
        var fac = r < Math.max(b, e) / 2;
        var P = a * c + (fac ? r * (b + e) : b * e);
        var d = a * c + (fac ? coneDim(r, b, e) : b * e) - 1;
        return { P: P, d: d, s: fmtShape(a, c) + ' ⊗ ' + (fac ? '(' + fmtShape(b, r) + '·' + fmtShape(r, e) + ')' : fmtShape(b, e)) };
      },
      tex: { P: '\\lvert M\\rvert = ac + r(b+e)\\ \\ (m=ab,\\ n=ce)', d: 'd(M) = ac + r(b+e-r) - 1', g: '\\lvert M\\rvert - d(M) = r^2 + 1\\ \\ (r \\le \\min(b,e))' },
      srcP: 'PEFT tuners/lokr/layer.py with defaults decompose_both=False, decompose_factor=−1: m = a·b and n = c·e split near √ by factorization(); w₁ ∈ ℝ^{a×c} is full; w₂ = w2_a·w2_b (b×r, r×e) when r < max(b,e)/2, else a full b×e matrix (then |M| = ac + be, d = ac + be − 1); ΔW = w₁ ⊗ w₂.',
      srcD: 'Derived from Prop II.13 (LoKr is a transported LoRA₁ whose second factor lies in 𝓜≤r; transport preserves d) and the KronA row of Prop II.12: the image {C ⊗ D : D ∈ 𝓜≤r} has dimension ac + dim 𝓜≤r(b×e) − 1, the gauge being ℝ^× × GL_r as in the periodic table.' },

    { id: 'lora-xs', name: 'LoRA-XS', data: ['lora-xs'], scope: 'sel', knobs: [rankKnob()],
      per: function (m, n, K) {
        var r = K.r;
        if (r > Math.min(m, n)) return { invalid: 'r must be ≤ min(m,n) = ' + Math.min(m, n) };
        return { P: r * r, d: r * r, s: 'R ' + fmtShape(r, r) };
      },
      tex: { P: '\\lvert M\\rvert = r^2', d: 'd(M) = r^2', g: '\\lvert M\\rvert - d(M) = 0' },
      srcP: 'Bałazy et al. 2024: one trainable r×r matrix R between frozen truncated-SVD factors of W₀. Not in HF PEFT; count from the paper and its official code.',
      srcD: 'Prop II.12(b), linear methods: trivial gauge, d = |M|.' },

    { id: 'fourierft', name: 'FourierFT', data: ['fourierft'], scope: 'sel',
      knobs: [{ k: 'nf', sym: 'n<sub>f</sub>', label: 'spectral coefficients per matrix', steps: STEP_NF, min: 1, max: 100000000, def: 1000 }],
      per: function (m, n, K) {
        var c = K.nf;
        if (c > m * n) return { invalid: 'n_f must be ≤ mn = ' + (m * n).toLocaleString('en-US') };
        var pairs = fourierPairs(m, n, c);
        if (pairs == null) return { P: c, d: c, bound: true, s: c + ' coefficients', exp: expectedPairs(m, n, c) };
        return { P: c, d: c - pairs, pairs: pairs, s: c + ' coefficients' + (pairs ? ', ' + pairs + ' conjugate pair' + (pairs === 1 ? '' : 's') : '') };
      },
      tex: { P: '\\lvert M\\rvert = n_f', d: 'd(M) = n_f - c', g: '\\lvert M\\rvert - d(M) = c\\ \\ (c = \\text{sampled conjugate pairs})' },
      srcP: 'PEFT tuners/fourierft/layer.py: fourierft_spectrum ∈ ℝ^{n_f} at the first n_f entries of torch.randperm(mn) with random_loc_seed = 777 (defaults n_frequency = 1000, one seed for every layer); ΔW = Re ifft2(spectrum)·scaling; n_f ≤ mn is enforced.',
      srcD: 'Prop II.12(b): a linear method with d = n_f − c, where c counts sampled conjugate pairs (u,v), (−u,−v), which Re ifft2 identifies. For the listed n_f values c is computed from PEFT’s default index set (torch.randperm, seed 777); for other n_f only d ≤ n_f is shown, and about n_f²/(2mn) pairs are expected.' },

    { id: 'ia3', name: '(IA)³', data: ['ia3'], scope: 'ia3', knobs: [],
      tex: { P: '\\lvert M\\rvert = m\\ \\text{(attention)},\\ \\ n\\ \\text{(down\\_proj)}', d: 'd(M) = \\lvert M\\rvert', g: '\\lvert M\\rvert - d(M) = 0' },
      srcP: 'PEFT tuners/ia3/layer.py: ia3_l ∈ ℝ^{m} scales outputs, or ℝ^{n} (inputs) for feed-forward modules. Default targets, peft/utils/constants.py: llama, mistral → k_proj, v_proj, down_proj; qwen2 → q_proj, v_proj, down_proj; feed-forward: down_proj.',
      srcD: 'Periodic table: (IA)³ is Lin with no gauge, so d = |M| whenever W₀ has no zero row (zero column for down_proj).' },

    { id: 'oft', name: 'OFT', data: ['oft'], scope: 'sel',
      knobs: [{ k: 'b', sym: 'b', label: 'block size b', steps: STEP_B, min: 2, max: 65536, def: 32 }],
      per: function (m, n, K) {
        var b = K.b, adj = null;
        if (n % b !== 0 || b > n) { adj = adjustOft(n, b); }
        var bb = adj == null ? b : adj, blocks = n / bb;
        return { P: blocks * bb * (bb - 1) / 2, d: null, why: 'orbit of ∏SO(b); no closed form in Prop II.12', adj: adj, s: blocks + ' blocks of ' + bb + (adj != null ? ' (b→' + bb + ')' : '') };
      },
      tex: { P: '\\lvert M\\rvert = \\tfrac{n}{b}\\binom{b}{2} = \\tfrac{n(b-1)}{2}', d: '\\text{no closed form (orbit of }\\textstyle\\prod SO(b)\\text{)}', g: null },
      srcP: 'PEFT tuners/oft/layer.py: OFTRotationModule weight ∈ ℝ^{(n/b) × b(b−1)/2} (strict upper triangles of n/b skew blocks); default oft_block_size = 32, block_share = False; a b that does not divide n is moved to the nearest divisor (adjust_oft_parameters). use_cayley_neumann = True (default) truncates the Cayley inverse, so R is only approximately orthogonal.',
      srcD: 'The framework classifies block OFT as an orbit of ∏SO(b) (periodic table) without a closed form for d in Prop II.12, so no value is shown.' },

    { id: 'boft', name: 'BOFT', data: ['boft'], scope: 'sel',
      knobs: [{ k: 'b', sym: 'b', label: 'block size b', steps: STEP_B, min: 2, max: 65536, def: 4 },
        { k: 'f', sym: 'f', label: 'butterfly factors f', steps: STEP_F, min: 1, max: 16, def: 1 }],
      per: function (m, n, K) {
        var b = K.b, f1 = K.f - 1;
        if (n % b !== 0) return { invalid: 'b must divide n = ' + n };
        if (f1 !== 0) {
          var span = b * Math.pow(2, f1);
          if (n < span || n % span !== 0) return { invalid: 'b·2^(f−1) = ' + span + ' must divide n = ' + n };
          if ((n / b) % 2 !== 0 || b % 2 !== 0) return { invalid: 'b and n/b must be even when f > 1' };
        }
        var blocks = n / b;
        return { P: K.f * blocks * b * b + m, eff: K.f * blocks * b * (b - 1) / 2 + m, d: null, why: 'butterfly product; no closed form in Prop II.12',
          s: K.f + '×' + blocks + ' blocks ' + fmtShape(b, b) + ' + s ' + m };
      },
      tex: { P: '\\lvert M\\rvert = f\\,\\tfrac{n}{b}\\,b^2 + m', d: '\\text{no closed form (butterfly product)}', g: null },
      srcP: 'PEFT tuners/boft/layer.py: boft_R ∈ ℝ^{f × (n/b) × b × b} and boft_s ∈ ℝ^{m}; defaults boft_block_size = 4, boft_n_butterfly_factor = 1. The Cayley map uses ½(R − Rᵀ), so the symmetric half of every block (b(b+1)/2 entries) never affects the weight; effective count f·(n/b)·b(b−1)/2 + m. f > 1 needs PEFT’s fbd CUDA extension; without it PEFT warns and resets f to 1.',
      srcD: 'No closed form for d in the framework (Prop II.12), so no value is shown.' },

    { id: 'bitfit', name: 'BitFit', data: ['bitfit'], scope: 'bitfit', knobs: [],
      tex: { P: '\\lvert M\\rvert = \\textstyle\\sum \\text{bias lengths}', d: 'd(M) = \\lvert M\\rvert', g: '\\lvert M\\rvert - d(M) = 0' },
      srcP: 'Ben Zaken et al. 2021: train every bias term and nothing else. Llama and Mistral have bias-free linear layers and RMSNorm (gain only), so BitFit trains 0 parameters there; Qwen2 has biases on q_proj, k_proj, v_proj.',
      srcD: 'Prop II.12(b), linear methods (a translation of the bias block): d = |M|.' },

    { id: 'prompt-tuning', name: 'Prompt tuning', data: ['prompt-tuning'], scope: 'prompt',
      knobs: [{ k: 'l', sym: 'ℓ', label: 'virtual tokens ℓ', steps: STEP_L, min: 1, max: 4096, def: 20 }],
      tex: { P: '\\lvert M\\rvert = \\ell\\, d_{\\mathrm{model}}', d: '\\text{none (acts on the input, not on }\\Theta\\text{)}', g: null },
      srcP: 'PEFT tuners/prompt_tuning/model.py: Embedding(ℓ · num_transformer_submodules, token_dim), with one sub-module for a decoder-only LM and token_dim = hidden_size.',
      srcD: 'An unpointed extension acting on the input sequence (shape Fun in the periodic table): not a subset of weight space, so no d.' },

    { id: 'prefix-tuning', name: 'Prefix tuning', data: ['prefix-tuning'], scope: 'prefix',
      knobs: [{ k: 'l', sym: 'ℓ', label: 'prefix length ℓ', steps: STEP_L, min: 1, max: 4096, def: 20 }],
      tex: { P: '\\lvert M\\rvert = 2L\\ell\\, d_{kv},\\ \\ d_{kv} = n_{kv} d_{\\mathrm{head}}', d: '\\text{none (acts on activations, not on }\\Theta\\text{)}', g: null },
      srcP: 'PEFT tuners/prefix_tuning/model.py: Embedding(ℓ, 2·L·token_dim); peft/utils/other.py sets token_dim = head_dim · n_kv for grouped-query models (= d_model without GQA). With prefix_projection = True (Li & Liang’s reparametrisation) training adds Embedding(ℓ,t) + Linear(t,h) + Linear(h,2Lt), h = encoder_hidden_size (default t), discarded afterwards.',
      srcD: 'Unpointed extension of shape Fun (a gated parallel adapter on attention): no weight-space d.' },

    { id: 'houlsby-adapter', name: 'Houlsby adapter', data: ['houlsby-adapter'], scope: 'houlsby',
      knobs: [{ k: 'r', sym: 'r', label: 'bottleneck r', steps: STEP_HR, min: 1, max: 65536, def: 64 }],
      tex: { P: '\\lvert M\\rvert = 2L\\,(2 d_{\\mathrm{model}} r + r + d_{\\mathrm{model}})', d: '\\text{none (acts on activations, not on }\\Theta\\text{)}', g: null },
      srcP: 'Houlsby et al. 2019, §2.1: “2md + d + m” parameters per adapter including biases (their m is our r); two adapters per layer. Not in HF PEFT. Excludes the layer-norm parameters the paper also trains.',
      srcD: 'Pointed extension of shape Fun (serial, non-linear): no weight-space d.' },

    { id: 'full', name: 'Full fine-tuning', data: [], scope: 'full', knobs: [], ref: true,
      tex: { P: '\\lvert M\\rvert = N', d: 'd(M) = N', g: '\\lvert M\\rvert - d(M) = 0' },
      srcP: 'Every parameter of the base model (embeddings counted once when tied).',
      srcD: 'Obs I.4: full fine-tuning is the terminal object, ρ = id_Θ, so d = |M|.' }
  ];
  var MODEL_SRC = 'Architectures from each repo’s config.json: Mistral and Qwen fetched directly; Meta’s Llama repos are gated, so Llama-3-8B was read from the NousResearch mirror whose config.json is byte-identical (same git blob 7784fbf), and Llama-2-7b from the NousResearch/Llama-2-7b-hf mirror. None sets head_dim, so d_head = d_model / heads. Each computed total equals the official repo’s safetensors parameter count reported by the Hugging Face API, except that the Llama-2 checkpoint also stores 2,048 F32 rotary inv_freq entries, which are buffers rather than parameters.';
  var THEORY_SRC = 'Reachable directions d(M) = dim Im M, the expressive dimension of Def I.5, counted in weight space Θ; the gauge gap |M| − d(M) is the gauge waste g(M) of Def I.5. Adapted matrices are independent boxes, so |M|, d(M) and the gap add over them. PEFT version: main @ 85be3f3 (0.21.1.dev0).';

  /* ---------- per-method computation over a model and a target-module selection ---------- */
  function ia3Targets(M) { return M.type === 'qwen2' ? ['q', 'v', 'down'] : ['k', 'v', 'down']; }
  function compute(me, M, sel, K) {
    var units = [], invalid = null, notes = [], L = M.L, dm = M.d;
    if (me.scope === 'sel') {
      sel.forEach(function (k) {
        var mo = modByKey(M, k), r = me.per(mo.m, mo.n, K, mo, M);
        if (r.invalid) { invalid = invalid || (mo.name + ': ' + r.invalid); return; }
        units.push({ label: mo.name, shape: fmtShape(mo.m, mo.n), count: L, P: r.P, d: r.d, why: r.why, bound: r.bound, num: r.num, pairs: r.pairs, exp: r.exp, s: r.s, eff: r.eff, adj: r.adj, b: K.b });
      });
      if (me.id === 'oft') units.forEach(function (u) { if (u.adj != null) notes.push(u.label + ': b = ' + K.b + ' does not divide n, PEFT adjusts it to ' + u.adj + '.'); });
      if (me.id === 'boft' && K.f > 1) notes.push('f = ' + K.f + ' needs PEFT’s fbd CUDA extension; without it PEFT warns and resets f to 1.');
      if (me.id === 'fourierft' && units.length) {
        var exact = units.filter(function (u) { return u.pairs != null; }), est = units.filter(function (u) { return u.exp != null; });
        if (exact.length) {
          var hit = exact.filter(function (u) { return u.pairs > 0; });
          notes.push('PEFT’s default index set (seed 777) at n_f = ' + K.nf + ' holds ' + (hit.length
            ? hit.map(function (u) { return plural(u.pairs, 'conjugate pair') + ' on ' + u.label; }).join(', ') + (hit.length < exact.length ? ' and none on the other matrices' : '') + '. Each pair costs one direction.'
            : 'no conjugate pair on any adapted matrix, so d = n_f.'));
        }
        if (est.length) notes.push('n_f = ' + K.nf + ' is not one of the stepped values, so only d ≤ n_f is shown. A uniform index set of this size holds on average ' + est.map(function (u) { return (u.exp < 0.01 ? u.exp.toExponential(1) : u.exp < 10 ? u.exp.toFixed(2) : String(Math.round(u.exp))) + ' conjugate pairs on ' + u.label; }).join(', ') + '.');
      }
    } else if (me.scope === 'ia3') {
      var ff = { down: true };
      ia3Targets(M).forEach(function (k) {
        var mo = modByKey(M, k), p = ff[k] ? mo.n : mo.m;
        units.push({ label: mo.name, shape: fmtShape(mo.m, mo.n), count: L, P: p, d: p, s: (ff[k] ? 'ℓ ∈ ℝ^' + mo.n + ' (inputs)' : 'ℓ ∈ ℝ^' + mo.m + ' (outputs)') });
      });
    } else if (me.scope === 'bitfit') {
      arch(M).forEach(function (mo) { if (mo.bias) units.push({ label: mo.name + '.bias', shape: String(mo.m), count: L, P: mo.m, d: mo.m, s: 'bias ' + mo.m }); });
    } else if (me.scope === 'prompt') {
      units.push({ label: 'soft prompt', shape: fmtShape(K.l, dm), count: 1, P: K.l * dm, d: null, s: K.l + ' tokens × ' + dm });
    } else if (me.scope === 'prefix') {
      var t = kvDim(M), h = t;
      units.push({ label: 'K, V prefixes', shape: '2×' + K.l + '×' + t, count: L, P: 2 * K.l * t, d: null, s: 'd_kv = ' + M.kv + '×' + headDim(M) });
      var mlp = K.l * t + (t * h + h) + (h * 2 * L * t + 2 * L * t);
      notes.push('Training with prefix_projection = True (h = d_kv = ' + t + '): ' + mlp.toLocaleString('en-US') + ' parameters, an MLP that is discarded after training.');
    } else if (me.scope === 'houlsby') {
      var r = K.r;
      units.push({ label: '2 adapters', shape: dm + '→' + r + '→' + dm, count: L, P: 2 * (2 * dm * r + r + dm), d: null, s: 'down + up, with biases' });
      notes.push('If the norms are also trained, as in the paper, add (2L+1)·d_model = ' + ((2 * L + 1) * dm).toLocaleString('en-US') + ' RMSNorm gains.');
    } else if (me.scope === 'full') {
      units.push({ label: 'embed_tokens', shape: fmtShape(M.V, dm), count: 1, P: M.V * dm, d: M.V * dm, s: 'V × d' });
      arch(M).forEach(function (mo) { var p = mo.m * mo.n + (mo.bias ? mo.m : 0); units.push({ label: mo.name, shape: fmtShape(mo.m, mo.n), count: L, P: p, d: p, s: mo.bias ? 'W + bias' : 'W' }); });
      units.push({ label: 'norms', shape: '2×' + dm, count: L, P: 2 * dm, d: 2 * dm, s: 'RMSNorm gains' });
      units.push({ label: 'final norm', shape: String(dm), count: 1, P: dm, d: dm, s: 'RMSNorm gain' });
      if (!M.tie) units.push({ label: 'lm_head', shape: fmtShape(M.V, dm), count: 1, P: M.V * dm, d: M.V * dm, s: 'untied' });
      else notes.push('lm_head is tied to embed_tokens and counted once.');
    }
    var P = 0, d = 0, dOk = units.length > 0, bound = false, num = false, mats = 0;
    units.forEach(function (u) {
      P += u.count * u.P; mats += u.count;
      if (u.d == null) dOk = false; else d += u.count * u.d;
      if (u.bound) bound = true;
      if (u.num) num = true;
    });
    if (me.id === 'boft' && units.length) {
      var eff = 0; units.forEach(function (u) { eff += u.count * u.eff; });
      notes.unshift('Of these, ' + (P - eff).toLocaleString('en-US') + ' are the symmetric halves of the b×b blocks, which the Cayley map discards; only ' + eff.toLocaleString('en-US') + ' entries affect the weight.');
    }
    if (!units.length && !invalid && me.tex.g) { dOk = true; d = 0; }   /* nothing to train: d = 0 */
    var res = { units: units, P: P, d: dOk ? d : null, bound: bound, num: num && dOk, mats: mats, invalid: invalid, notes: notes };
    res.gap = res.d == null ? null : P - res.d;
    return res;
  }

  /* exposed for headless verification against PEFT (no effect on the page) */
  Atlas._calculatorCore = { MODELS: MODELS, METHODS: METHODS, arch: arch, totalParams: totalParams, compute: compute, factorization: factorization, adjustOft: adjustOft };

  /* =====================================================================================
     4. Figure
     ===================================================================================== */
  var CSS = [
    '[data-figure="calculator"] .calc{padding:1.15rem 1.2rem 1.1rem}',
    '[data-figure="calculator"] .calc-head{margin-bottom:1rem}',
    '[data-figure="calculator"] .calc-title-row{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.2rem 1rem}',
    '[data-figure="calculator"] .calc-title{margin:0;font-family:var(--f-display);font-size:clamp(1.5rem,3.6vw,1.95rem);font-weight:var(--w-head);letter-spacing:-.01em;line-height:1.08}',
    '[data-figure="calculator"] .calc-tag{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide);white-space:nowrap}',
    '[data-figure="calculator"] .calc-dek{margin:.25rem 0 .55rem;font-family:var(--f-body);font-style:italic;font-size:1.1rem;line-height:1.35;color:var(--ink-2)}',
    '[data-figure="calculator"] .calc-instr{margin:0;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:46rem}',
    '[data-figure="calculator"] .calc-instr b{font-weight:600;color:var(--ink)}',
    /* controls */
    '[data-figure="calculator"] .calc-controls{display:grid;grid-template-columns:minmax(0,1.08fr) minmax(0,1.15fr) minmax(0,.8fr);gap:.7rem;margin:0 0 1.25rem}',
    '[data-figure="calculator"] .calc.is-narrow .calc-controls{grid-template-columns:minmax(0,1fr)}',
    '[data-figure="calculator"] .calc-group{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:.6rem .75rem .7rem;min-width:0;display:grid;gap:.45rem;align-content:start}',
    '[data-figure="calculator"] .calc-glabel,[data-figure="calculator"] .calc-group .control>label{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    '[data-figure="calculator"] .calc-select{width:100%;font-family:var(--f-ui);font-weight:500;font-size:.875rem;padding:.32rem .45rem}',
    '[data-figure="calculator"] .calc-arch{display:grid;grid-template-columns:repeat(3,max-content);gap:.1rem .8rem;font-family:var(--f-mono);font-size:.78rem;line-height:1.5;color:var(--ink);font-variant-numeric:tabular-nums}',
    '[data-figure="calculator"] .calc-arch i{font-family:var(--f-ui);font-style:normal;color:var(--ink-2);margin-right:.35em}',
    '[data-figure="calculator"] .calc-arch sub{font-size:.96em;line-height:0;vertical-align:-.28em}',
    '[data-figure="calculator"] .calc-total{font-family:var(--f-ui);font-size:.8rem;line-height:1.45;color:var(--ink-2);font-variant-numeric:tabular-nums;border-top:1px solid var(--rule);padding-top:.4rem}',
    '[data-figure="calculator"] .calc-total b{font-family:var(--f-mono);font-weight:500;color:var(--ink);font-size:.9rem}',
    '[data-figure="calculator"] .calc-total .ok{color:var(--moss-ink)}',
    '[data-figure="calculator"] .calc-total .bad{color:var(--seal-ink)}',
    '[data-figure="calculator"] .calc-group .seg{justify-self:start;max-width:100%}',
    '[data-figure="calculator"] .calc-mods{display:flex;flex-wrap:wrap;gap:.3rem}',
    '[data-figure="calculator"] .calc-mods button{font-family:var(--f-ui);font-weight:500;font-size:.8rem;line-height:1.25;padding:.24rem .55rem;border:1px solid var(--rule);background:var(--paper-2);color:var(--ink-2);border-radius:4px;cursor:pointer;min-width:2rem}',
    '[data-figure="calculator"] .calc-mods button:hover{border-color:var(--tide);color:var(--ink)}',
    '[data-figure="calculator"] .calc-mods button[aria-pressed="true"]{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
    '[data-figure="calculator"] .calc-modsum{font-family:var(--f-mono);font-size:.75rem;line-height:1.5;color:var(--ink-2);font-variant-numeric:tabular-nums}',
    '[data-figure="calculator"] .calc-group .control{min-width:0;gap:.25rem}',
    '[data-figure="calculator"] .calc-group .control output{font-size:.95rem}',
    '[data-figure="calculator"] .calc-help{font-family:var(--f-body);font-size:.8rem;line-height:1.45;color:var(--ink-2)}',
    /* chart rows */
    '[data-figure="calculator"] .calc-ctx{font-family:var(--f-ui);font-size:.8rem;line-height:1.5;color:var(--ink-2);margin:0 0 .4rem;font-variant-numeric:tabular-nums}',
    '[data-figure="calculator"] .calc-ctx b{color:var(--ink);font-weight:500}',
    '[data-figure="calculator"] .cr{display:grid;grid-template-columns:170px 116px minmax(0,1fr) 60px 72px 60px 54px;column-gap:8px;align-items:center;min-height:34px;padding:0 6px;border-bottom:1px solid var(--rule)}',
    '[data-figure="calculator"] .cr-nums{display:contents}',
    '[data-figure="calculator"] .cr-l{display:none}',
    '[data-figure="calculator"] .cr-n{text-align:right;font-family:var(--f-mono);font-size:.78rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
    '[data-figure="calculator"] .cr-n b{font-weight:400}',
    '[data-figure="calculator"] .cr-pct b{color:var(--ink-2)}',
    '[data-figure="calculator"] .cr-g b.pos{color:var(--seal-ink)}',
    '[data-figure="calculator"] .cr-g b.zero{color:var(--moss-ink)}',
    '[data-figure="calculator"] .cr-n b.na{color:var(--ink-3)}',
    '[data-figure="calculator"] .cr-dag{font-family:var(--f-mono);font-size:.96em;color:var(--ochre-ink);margin-left:.06em;vertical-align:.24em;line-height:0;cursor:help}',
    '[data-figure="calculator"] .cr-bar{align-self:stretch;position:relative;min-width:0;min-height:30px}',
    '[data-figure="calculator"] .cr-bar svg{position:absolute;left:0;top:0;width:100%;height:100%;overflow:visible;display:block}',
    '[data-figure="calculator"] .cr.off .cr-name,[data-figure="calculator"] .cr.off .cr-knob,[data-figure="calculator"] .cr.off .cr-nums{opacity:.5}',
    '[data-figure="calculator"] .cr.focus{background:var(--tide-soft);box-shadow:inset 2px 0 0 var(--tide)}',
    '[data-figure="calculator"] .cr-name{display:grid;grid-template-columns:auto auto minmax(0,1fr);align-items:center;column-gap:.38rem;min-width:0;padding:.3rem 0}',
    '[data-figure="calculator"] .cr-name input{margin:0;accent-color:var(--tide);width:.85rem;height:.85rem}',
    '[data-figure="calculator"] .cr-sw{width:.55rem;height:.55rem;border-radius:2px}',
    '[data-figure="calculator"] .cr-mname{appearance:none;border:0;background:none;padding:0;margin:0;text-align:left;cursor:pointer;font-family:var(--f-body);font-size:.97rem;line-height:1.2;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}',
    '[data-figure="calculator"] .cr-mname:hover{color:var(--tide)}',
    '[data-figure="calculator"] .cr-mname sup{font-family:var(--f-body);font-size:.8em;line-height:0;color:var(--ink-2);margin-left:.06em;vertical-align:.42em}',
    '[data-figure="calculator"] .cr-scope{grid-column:3;font-family:var(--f-ui);font-size:.75rem;line-height:1.3;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    '[data-figure="calculator"] .cr-knob{display:grid;gap:.18rem;justify-items:start;min-width:0}',
    '[data-figure="calculator"] .cr-none{font-family:var(--f-mono);font-size:.8rem;color:var(--ink-3)}',
    '[data-figure="calculator"] .cr-hdr,[data-figure="calculator"] .cr-ftr{min-height:28px}',
    '[data-figure="calculator"] .cr-hdr{border-bottom:1px solid var(--ink-3)}',
    '[data-figure="calculator"] .cr-ftr{border-bottom:0}',
    '[data-figure="calculator"] .cr-h{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-2);white-space:nowrap}',
    '[data-figure="calculator"] .cr-h.num{text-align:right}',
    '[data-figure="calculator"] .cr-ftrlab{grid-column:1 / span 2;text-transform:none;letter-spacing:0;font-weight:400;font-size:.75rem;text-align:right;padding-right:2px;align-self:start;padding-top:5px}',
    '[data-figure="calculator"] .cr-hdr .cr-bar,[data-figure="calculator"] .cr-ftr .cr-bar{min-height:26px}',
    /* stepper */
    '[data-figure="calculator"] .ck{display:inline-flex;align-items:center}',
    '[data-figure="calculator"] .ck-sym{font-family:var(--f-body);font-style:italic;font-size:.98rem;color:var(--ink-2);width:1.15rem;text-align:left}',
    '[data-figure="calculator"] .ck-sym sub{font-size:.8em;line-height:0}',
    '[data-figure="calculator"] .ck button{width:1.4rem;height:1.5rem;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;font-family:var(--f-mono);font-size:.82rem;line-height:1;padding:0}',
    '[data-figure="calculator"] .ck button:hover{color:var(--tide);border-color:var(--tide)}',
    '[data-figure="calculator"] .ck .ck-dec{border-radius:4px 0 0 4px}',
    '[data-figure="calculator"] .ck .ck-inc{border-radius:0 4px 4px 0}',
    '[data-figure="calculator"] .ck input{width:3.2rem;height:1.5rem;text-align:center;border:1px solid var(--rule);border-left:0;border-right:0;border-radius:0;padding:0 .15rem;font-family:var(--f-mono);font-size:.78rem;font-variant-numeric:tabular-nums;background:var(--paper);color:var(--ink);-moz-appearance:textfield;appearance:textfield}',
    '[data-figure="calculator"] .ck input::-webkit-outer-spin-button,[data-figure="calculator"] .ck input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}',
    /* legend */
    '[data-figure="calculator"] .calc-legend{display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;align-items:center;margin-top:.6rem;font-family:var(--f-ui);font-size:.75rem;line-height:1.4;color:var(--ink-2)}',
    '[data-figure="calculator"] .lg{display:inline-flex;align-items:center;gap:.4rem;white-space:nowrap}',
    '[data-figure="calculator"] .lg svg{flex:none}',
    '[data-figure="calculator"] .lg-kinds{display:flex;flex-wrap:wrap;gap:.3rem .85rem;align-items:center;flex-basis:100%;color:var(--ink-2)}',
    '[data-figure="calculator"] .lg-marks{gap:.3rem}',
    '[data-figure="calculator"] .lg-marks b{font-family:var(--f-mono);font-weight:500;color:var(--ink);font-size:.875rem}',
    '[data-figure="calculator"] .lg-marks b:nth-of-type(2){color:var(--ochre-ink);margin-left:.6rem}',
    '[data-figure="calculator"] .lg-sw{display:inline-block;width:.6rem;height:.6rem;border-radius:2px;margin-right:.3rem;vertical-align:-.05rem}',
    /* focus panel */
    '[data-figure="calculator"] .calc-focus{margin-top:1.15rem;border:1px solid var(--rule);border-radius:6px;background:var(--paper);padding:.85rem .95rem .9rem}',
    '[data-figure="calculator"] .cf-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:.15rem .8rem;margin-bottom:.6rem}',
    '[data-figure="calculator"] .cf-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide);flex-basis:100%}',
    '[data-figure="calculator"] .cf-title{font-family:var(--f-display);font-size:1.4rem;font-weight:var(--w-head);margin:0}',
    '[data-figure="calculator"] .cf-meta{font-family:var(--f-body);font-style:italic;color:var(--ink-2);font-size:.9rem;line-height:1.4}',
    '[data-figure="calculator"] .cf-forms{display:grid;grid-template-columns:max-content minmax(0,1.05fr) minmax(0,1.2fr);gap:.45rem 1rem;align-items:baseline;margin:0 0 .75rem}',
    '[data-figure="calculator"] .cf-k{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    '[data-figure="calculator"] .cf-f{font-size:.95rem;min-width:0;overflow-x:auto;overflow-y:hidden}',
    '[data-figure="calculator"] .cf-s{font-family:var(--f-body);font-size:.8rem;line-height:1.45;color:var(--ink-2)}',
    '[data-figure="calculator"] .calc.is-narrow .cf-forms{grid-template-columns:max-content minmax(0,1fr)}',
    '[data-figure="calculator"] .calc.is-narrow .cf-s{grid-column:1 / -1;margin:-.15rem 0 .25rem}',
    '[data-figure="calculator"] .cf-tw{overflow-x:auto;border:1px solid var(--rule);border-radius:4px}',
    '[data-figure="calculator"] .cf-table{font-size:.8rem;line-height:1.35}',
    '[data-figure="calculator"] .cf-table th{position:static;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;color:var(--ink-2);padding:.4rem .55rem;background:var(--paper-2);white-space:nowrap}',
    '[data-figure="calculator"] .cf-table td{font-family:var(--f-mono);font-variant-numeric:tabular-nums;padding:.32rem .55rem;white-space:nowrap}',
    '[data-figure="calculator"] .cf-table td.num,[data-figure="calculator"] .cf-table th.num{text-align:right}',
    '[data-figure="calculator"] .cf-table tr.tot td{border-top:1px solid var(--ink-3);color:var(--ink);font-weight:500}',
    '[data-figure="calculator"] .cf-table td.st{color:var(--ink-2)}',
    '[data-figure="calculator"] .cf-table .cf-shp{display:none}',
    '[data-figure="calculator"] .cf-notes{margin:.6rem 0 0;padding-left:1.1rem;font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
    '[data-figure="calculator"] .cf-notes li{margin-bottom:.2rem}',
    '[data-figure="calculator"] .cf-notes li.warn{color:var(--seal-ink)}',
    '[data-figure="calculator"] .cf-coords{margin-top:.6rem;font-family:var(--f-ui);font-size:.8rem;line-height:1.55;color:var(--ink-2)}',
    '[data-figure="calculator"] .cf-coords b{font-weight:500;color:var(--ink)}',
    '[data-figure="calculator"] .cf-read{display:block;margin-top:.55rem;font-family:var(--f-body);font-size:.9rem;line-height:1.5;color:var(--ink-2)}',
    /* sources */
    '[data-figure="calculator"] .calc-src{margin-top:.95rem;font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
    '[data-figure="calculator"] .calc-src summary{cursor:pointer;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    '[data-figure="calculator"] .calc-src summary:hover{color:var(--tide)}',
    '[data-figure="calculator"] .calc-src ol{margin:.6rem 0 0;padding-left:1.5rem;columns:2 22rem;column-gap:2.2rem}',
    '[data-figure="calculator"] .calc-src li{break-inside:avoid;margin-bottom:.45rem}',
    '[data-figure="calculator"] .calc-src li b{font-weight:600;color:var(--ink)}',
    '[data-figure="calculator"] .calc-src p{margin:.5rem 0 0}',
    /* narrow layout */
    '[data-figure="calculator"] .calc.is-narrow{padding:.95rem .85rem .9rem}',
    '[data-figure="calculator"] .calc.is-narrow .cr{grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"name knob" "bar bar" "nums nums";row-gap:3px;padding:.4rem 2px .45rem}',
    '[data-figure="calculator"] .calc.is-narrow .cr-name{grid-area:name;padding:0}',
    '[data-figure="calculator"] .calc.is-narrow .cr-knob{grid-area:knob;justify-items:end}',
    '[data-figure="calculator"] .calc.is-narrow .cr-bar{grid-area:bar;min-height:16px}',
    '[data-figure="calculator"] .calc.is-narrow .cr-nums{grid-area:nums;display:flex;flex-wrap:wrap;gap:.05rem .7rem}',
    '[data-figure="calculator"] .calc.is-narrow .cr-n{text-align:left;font-size:.78rem}',
    '[data-figure="calculator"] .calc.is-narrow .cr-pct b::before{content:"(";color:var(--ink-3)}',
    '[data-figure="calculator"] .calc.is-narrow .cr-pct b::after{content:")";color:var(--ink-3)}',
    '[data-figure="calculator"] .calc.is-narrow .cr-l{display:inline;font-family:var(--f-ui);font-style:normal;font-size:.96em;color:var(--ink-2);margin-right:.32em}',
    '[data-figure="calculator"] .calc.is-narrow .cr-hdr,[data-figure="calculator"] .calc.is-narrow .cr-ftr{grid-template-areas:"bar bar";padding:0 2px}',
    '[data-figure="calculator"] .calc.is-narrow .cr-hdr .cr-h:not(.bar),[data-figure="calculator"] .calc.is-narrow .cr-ftr .cr-h:not(.bar),[data-figure="calculator"] .calc.is-narrow .cr-hdr .cr-nums,[data-figure="calculator"] .calc.is-narrow .cr-ftr .cr-nums{display:none}',
    '[data-figure="calculator"] .calc.is-narrow .cr-hdr .cr-bar,[data-figure="calculator"] .calc.is-narrow .cr-ftr .cr-bar{min-height:24px}',
    '[data-figure="calculator"] .calc.is-narrow .cr-ftr{grid-template-areas:"bar bar" "lab lab"}',
    '[data-figure="calculator"] .calc.is-narrow .cr-ftr .cr-h.cr-ftrlab{display:block;grid-area:lab;grid-column:auto;text-align:left;padding:2px 0 0}',
    '[data-figure="calculator"] .calc.is-narrow .cf-table .cnt{display:none}',
    '[data-figure="calculator"] .calc.is-narrow .calc-arch{grid-template-columns:repeat(2,max-content)}',
    '[data-figure="calculator"] .calc.is-narrow .calc-src ol{columns:1}',
    '[data-figure="calculator"] .calc.is-narrow .lg{white-space:normal;align-items:flex-start}',
    '[data-figure="calculator"] .calc.is-narrow .lg svg{margin-top:.2rem}',
    '[data-figure="calculator"] .calc.is-narrow .cf-table .st{display:none}',
    '[data-figure="calculator"] .calc.is-narrow .cf-table td,[data-figure="calculator"] .calc.is-narrow .cf-table th{padding:.32rem .3rem;font-size:.78rem}',
    '[data-figure="calculator"] .calc.is-narrow .cf-table .shp{display:none}',
    '[data-figure="calculator"] .calc.is-narrow .cf-table .cf-shp{display:block;font-size:.97em;color:var(--ink-2);line-height:1.25}',
    '[data-figure="calculator"] .calc.is-narrow .cf-table .cr-dag{vertical-align:.2em}',
    '[data-figure="calculator"] .calc.is-narrow .cf-table th{font-size:.75rem;letter-spacing:.02em}'
  ].join('\n');

  function injectCss() {
    if (document.getElementById('css-calculator')) return;
    var s = document.createElement('style');
    s.id = 'css-calculator';
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  var NS = 'http://www.w3.org/2000/svg';
  function nf(n) { return Number(n).toLocaleString('en-US'); }
  function pctTxt(x) {
    if (!isFinite(x)) return '—';
    var v = x * 100;
    if (v === 0) return '0%';
    if (v >= 99.95 && v < 100.5) return '100%';
    if (v >= 100) return Math.round(v).toLocaleString('en-US') + '%';
    if (v >= 10) return v.toFixed(0) + '%';
    if (v >= 1) return v.toFixed(1) + '%';
    var dg = Math.min(8, 1 - Math.floor(Math.log10(v)));
    return v.toFixed(dg) + '%';
  }
  function logLabel(v) {
    var e = Math.round(Math.log10(v));
    var units = [[12, 'T'], [9, 'B'], [6, 'M'], [3, 'K']];
    for (var i = 0; i < units.length; i++) if (e >= units[i][0]) return Math.pow(10, e - units[i][0]) + units[i][1];
    return String(Math.pow(10, e));
  }
  function nearestStepUp(steps, v, max) { for (var i = 0; i < steps.length; i++) if (steps[i] > v) return Math.min(steps[i], max); return Math.min(max, v * 2); }
  function nearestStepDown(steps, v, min) { for (var i = steps.length - 1; i >= 0; i--) if (steps[i] < v) return Math.max(steps[i], min); return Math.max(min, Math.floor(v / 2)); }

  Atlas.register('calculator', function (el, A) {
    injectCss();
    var UID = 'calc' + Math.random().toString(36).slice(2, 7);
    var hasD3 = typeof window.d3 !== 'undefined';
    var state = { model: 0, mods: { q: true, v: true }, knobs: {}, on: {}, focus: 'lora', res: {}, rankIdx: 3 };
    METHODS.forEach(function (me) {
      state.on[me.id] = true;
      state.knobs[me.id] = {};
      me.knobs.forEach(function (k) { state.knobs[me.id][k.k] = k.def; });
    });
    function kindOf(me) {
      if (me.ref) return null;
      var dm = A.method(me.data[0]);
      return dm && dm.modification_kind ? dm.modification_kind : 'hybrid';
    }
    function kindToken(kind) {
      if (!kind) return '--ink';
      for (var i = 0; i < A.KINDS.length; i++) if (A.KINDS[i].key === kind) return A.KINDS[i].token;
      return '--c-hybrid';
    }
    function colorOf(me) { return A.css(kindToken(kindOf(me))); }

    /* ---------------- skeleton ---------------- */
    var stage = A.h('div', { class: 'stage calc', role: 'group', 'aria-label': 'Budget calculator: trainable parameters versus reachable directions for parameter-efficient fine-tuning methods' });
    el.appendChild(stage);

    var head = A.h('div', { class: 'calc-head' }, [
      A.h('div', { class: 'calc-title-row' }, [
        A.h('h3', { class: 'calc-title', text: 'The budget calculator' }),
        A.h('span', { class: 'calc-tag', text: 'Def I.5 · Thm II.2 · Prop II.12' })
      ]),
      A.h('p', { class: 'calc-dek', text: 'Parameters trained versus directions reachable, on real base models.' }),
      A.h('p', { class: 'calc-instr', html: 'Pick a base model and the matrices to adapt, then set each method’s knob. The <b>pale bar</b> counts the parameters a method trains, the <b>solid bar</b> the independent directions of weight space it can reach. Both use a log scale. Click a method name for its per-matrix formulas and sources.' })
    ]);
    stage.appendChild(head);

    /* controls */
    var controls = A.h('div', { class: 'calc-controls' });
    stage.appendChild(controls);

    var selId = UID + '-model';
    var modelSel = A.h('select', { id: selId, class: 'calc-select' });
    MODELS.forEach(function (M, i) { modelSel.appendChild(A.h('option', { value: String(i), text: M.label })); });
    var archEl = A.h('div', { class: 'calc-arch', 'aria-live': 'polite' });
    var totalEl = A.h('div', { class: 'calc-total' });
    controls.appendChild(A.h('div', { class: 'calc-group' }, [
      A.h('label', { class: 'calc-glabel', for: selId, text: 'Base model' }), modelSel, archEl, totalEl
    ]));
    modelSel.addEventListener('change', function () { state.model = +modelSel.value; refresh(); });

    var modLabelId = UID + '-mods';
    var presetSeg = A.seg([{ value: 'qv', label: 'q, v' }, { value: 'attn', label: 'attention' }, { value: 'all', label: 'all linear' }], 'qv', function (v) {
      state.mods = {}; PRESETS[v].forEach(function (k) { state.mods[k] = true; }); refresh();
    }, 'Target-module preset');
    var modWrap = A.h('div', { class: 'calc-mods', role: 'group', 'aria-labelledby': modLabelId });
    var modBtns = {};
    MOD_KEYS.forEach(function (k) {
      var b = A.h('button', { type: 'button', 'aria-pressed': 'false', text: k });
      b.addEventListener('click', function () { state.mods[k] = !state.mods[k]; refresh(); });
      modBtns[k] = b; modWrap.appendChild(b);
    });
    var modSum = A.h('div', { class: 'calc-modsum' });
    controls.appendChild(A.h('div', { class: 'calc-group' }, [
      A.h('div', { class: 'calc-glabel', id: modLabelId, text: 'Adapted matrices' }), presetSeg.el, modWrap, modSum
    ]));

    var rankSlider = A.slider({
      id: UID + '-rank', label: 'Rank · low-rank rows', min: 0, max: STEP_RANK.length - 1, step: 1, value: state.rankIdx,
      format: function (i) { return 'r = ' + STEP_RANK[i]; },
      onInput: function (i) {
        state.rankIdx = i;
        METHODS.forEach(function (me) { me.knobs.forEach(function (k) { if (k.rank) state.knobs[me.id][k.k] = STEP_RANK[i]; }); });
        syncKnobs(); refreshSoon();
      }
    });
    controls.appendChild(A.h('div', { class: 'calc-group' }, [
      rankSlider.el,
      A.h('div', { class: 'calc-help', text: 'Sets r for LoRA, DoRA, LoHa, LoKr and LoRA-XS at once; every knob below can also be set on its own.' })
    ]));

    /* chart */
    var chart = A.h('div', { class: 'calc-chart' });
    stage.appendChild(chart);
    var ctxEl = A.h('div', { class: 'calc-ctx' });
    chart.appendChild(ctxEl);

    function svgEl(tag, attrs) { var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); return e; }

    var hdr = A.h('div', { class: 'cr cr-hdr' }, [
      A.h('div', { class: 'cr-h', text: 'method' }), A.h('div', { class: 'cr-h', text: 'knob' }),
      A.h('div', { class: 'cr-bar cr-h bar' }),
      A.h('div', { class: 'cr-nums' }, [A.h('div', { class: 'cr-h num', text: 'trained' }), A.h('div', { class: 'cr-h num', text: '% model' }), A.h('div', { class: 'cr-h num', text: 'reached' }), A.h('div', { class: 'cr-h num', text: 'gap' })])
    ]);
    var axisSvg = svgEl('svg', { 'aria-hidden': 'true' });
    hdr.children[2].appendChild(axisSvg);
    chart.appendChild(hdr);

    var rows = {};
    METHODS.forEach(function (me, idx) {
      var cbId = UID + '-on-' + me.id;
      var cb = A.h('input', { type: 'checkbox', id: cbId, checked: 'checked', 'aria-label': 'Show ' + me.name + ' in the chart' });
      cb.addEventListener('change', function () { state.on[me.id] = cb.checked; refresh(); });
      var sw = A.h('span', { class: 'cr-sw', 'aria-hidden': 'true' });
      var nameBtn = A.h('button', { type: 'button', class: 'cr-mname', 'aria-controls': UID + '-focus', title: 'Show per-matrix formulas and sources for ' + me.name, text: me.name });
      nameBtn.addEventListener('click', function () { state.focus = me.id; refresh(true); });
      var scope = A.h('span', { class: 'cr-scope' });
      var nameCell = A.h('div', { class: 'cr-name' }, [cb, sw, nameBtn, scope]);
      var knobCell = A.h('div', { class: 'cr-knob' });
      var inputs = {};
      if (!me.knobs.length) knobCell.appendChild(A.h('span', { class: 'cr-none', text: '—', title: 'no hyperparameter' }));
      me.knobs.forEach(function (k) {
        var id = UID + '-' + me.id + '-' + k.k;
        var input = A.h('input', { type: 'number', id: id, min: k.min, max: k.max, step: 1, value: k.def, inputmode: 'numeric' });
        var lab = A.h('label', { class: 'ck-sym', for: id, html: k.sym, title: me.name + ': ' + k.label });
        var srLab = A.h('span', { class: 'sr-only', text: me.name + ' ' + k.label });
        lab.appendChild(srLab);
        var dec = A.h('button', { type: 'button', class: 'ck-dec', 'aria-label': 'Decrease ' + me.name + ' ' + k.label, text: '−' });
        var inc = A.h('button', { type: 'button', class: 'ck-inc', 'aria-label': 'Increase ' + me.name + ' ' + k.label, text: '+' });
        function commit(v) {
          v = Math.round(+v);
          if (!isFinite(v)) v = state.knobs[me.id][k.k];
          v = Math.max(k.min, Math.min(k.max, v));
          state.knobs[me.id][k.k] = v; input.value = v; refreshSoon();
        }
        dec.addEventListener('click', function () { commit(nearestStepDown(k.steps, state.knobs[me.id][k.k], k.min)); });
        inc.addEventListener('click', function () { commit(nearestStepUp(k.steps, state.knobs[me.id][k.k], k.max)); });
        var deb = A.debounce(function () { if (input.value !== '' && isFinite(+input.value)) commit(input.value); }, 160);
        input.addEventListener('input', deb);
        input.addEventListener('change', function () { commit(input.value === '' ? state.knobs[me.id][k.k] : input.value); });
        inputs[k.k] = input;
        knobCell.appendChild(A.h('span', { class: 'ck' }, [lab, dec, input, inc]));
      });
      var barCell = A.h('div', { class: 'cr-bar' });
      var svg = svgEl('svg', { role: 'img' });
      barCell.appendChild(svg);
      function num(cls, lab) { var b = A.h('b'); var s = A.h('span', { class: 'cr-n ' + cls }, [lab ? A.h('i', { class: 'cr-l', text: lab }) : null, b]); return { el: s, b: b }; }
      var nP = num('cr-P', 'trained'), nPct = num('cr-pct', ''), nD = num('cr-d', 'reached'), nG = num('cr-g', 'gap');
      var numsCell = A.h('div', { class: 'cr-nums' }, [nP.el, nPct.el, nD.el, nG.el]);
      var row = A.h('div', { class: 'cr', 'data-id': me.id }, [nameCell, knobCell, barCell, numsCell]);
      chart.appendChild(row);
      rows[me.id] = { me: me, row: row, cb: cb, sw: sw, scope: scope, inputs: inputs, svg: svg, nP: nP.b, nPct: nPct.b, nD: nD.b, nG: nG.b, nameBtn: nameBtn };
    });

    var ftrLabel = A.h('div', { class: 'cr-h cr-ftrlab' });
    var ftr = A.h('div', { class: 'cr cr-ftr' }, [
      ftrLabel, A.h('div', { class: 'cr-bar cr-h bar' }), A.h('div', { class: 'cr-nums' })
    ]);
    var pctSvg = svgEl('svg', { 'aria-hidden': 'true' });
    ftr.children[1].appendChild(pctSvg);
    chart.appendChild(ftr);

    /* legend */
    function legendBar(kind) {
      var s = svgEl('svg', { width: 30, height: 10, viewBox: '0 0 30 10', 'aria-hidden': 'true' });
      if (kind === 'solid') s.appendChild(svgEl('rect', { x: 0, y: 1, width: 30, height: 8, rx: 1.5, style: 'fill: var(--ink-2)' }));
      if (kind === 'pale') s.appendChild(svgEl('rect', { x: 0, y: 1, width: 30, height: 8, rx: 1.5, style: 'fill: var(--ink-2); fill-opacity: .26' }));
      if (kind === 'tail') {
        s.appendChild(svgEl('rect', { x: 0, y: 1, width: 30, height: 8, rx: 1.5, style: 'fill: var(--ink-2); fill-opacity: .26' }));
        s.appendChild(svgEl('rect', { x: 0, y: 1, width: 19, height: 8, rx: 1.5, style: 'fill: var(--ink-2)' }));
        s.appendChild(svgEl('line', { x1: 29.5, x2: 29.5, y1: 0, y2: 10, style: 'stroke: var(--seal); stroke-width: 1.4' }));
      }
      if (kind === 'dash') s.appendChild(svgEl('rect', { x: .6, y: 1.6, width: 28.8, height: 6.8, rx: 1.5, style: 'fill: var(--ink-2); fill-opacity: .14; stroke: var(--ink-2); stroke-dasharray: 2.5 2' }));
      return s;
    }
    var legend = A.h('div', { class: 'calc-legend' }, [
      A.h('span', { class: 'lg' }, [legendBar('solid'), document.createTextNode('reachable directions d(M)')]),
      A.h('span', { class: 'lg' }, [legendBar('pale'), document.createTextNode('trainable parameters |M|')]),
      A.h('span', { class: 'lg' }, [legendBar('tail'), document.createTextNode('pale tail to the red tick = gauge gap |M| − d(M)')]),
      A.h('span', { class: 'lg' }, [legendBar('dash'), document.createTextNode('d(M) has no closed form, or no meaning off weight space')]),
      A.h('span', { class: 'lg lg-marks' }, [A.h('b', { text: '≤' }), document.createTextNode('upper bound'), A.h('b', { text: '†' }), document.createTextNode('checked numerically, not proved')])
    ]);
    var kindsSeen = [], kindsWrap = A.h('div', { class: 'lg-kinds' }, [document.createTextNode('colour = modification kind:')]);
    METHODS.forEach(function (me) {
      var kd = kindOf(me);
      if (kindsSeen.indexOf(kd) >= 0) return;
      kindsSeen.push(kd);
      var label = kd ? A.kindName(kd).toLowerCase() : 'reference';
      kindsWrap.appendChild(A.h('span', {}, [A.h('i', { class: 'lg-sw', style: 'background: var(' + kindToken(kd) + ')' }), document.createTextNode(label)]));
    });
    legend.appendChild(kindsWrap);
    chart.appendChild(legend);

    /* focus panel */
    var focus = A.h('section', { class: 'calc-focus', id: UID + '-focus', 'aria-live': 'polite' });
    stage.appendChild(focus);
    var fHead = A.h('div', { class: 'cf-head' });
    var fForms = A.h('div', { class: 'cf-forms' });
    var fTableWrap = A.h('div', { class: 'cf-tw' });
    var fNotes = A.h('ul', { class: 'cf-notes' });
    var fCoords = A.h('div', { class: 'cf-coords' });
    focus.appendChild(fHead); focus.appendChild(fForms); focus.appendChild(fTableWrap); focus.appendChild(fNotes); focus.appendChild(fCoords);

    /* sources */
    var src = A.h('details', { class: 'calc-src' });
    src.appendChild(A.h('summary', { text: 'Sources and assumptions · ' + METHODS.length + ' methods' }));
    var ol = A.h('ol');
    METHODS.forEach(function (me) {
      var dm = me.data.length ? A.method(me.data[0]) : null;
      var who = me.data.map(function (id) { var m = A.method(id); return m ? (m.authors || m.name) + ' ' + (m.year || '') + (m.arxiv ? ' (arXiv ' + m.arxiv + ')' : '') : null; }).filter(Boolean).join('; ');
      ol.appendChild(A.h('li', { html: '<b>' + A.esc(me.name) + '</b>' + (who ? ' · ' + A.esc(who) : '') + '. <i>Count:</i> ' + A.esc(me.srcP) + ' <i>Reach:</i> ' + A.esc(me.srcD) }));
      void dm;
    });
    src.appendChild(ol);
    src.appendChild(A.h('p', { html: '<b>Base models.</b> ' + A.esc(MODEL_SRC) }));
    src.appendChild(A.h('p', { html: '<b>Theory.</b> ' + A.esc(THEORY_SRC) }));
    stage.appendChild(src);

    /* ---------------- state → DOM ---------------- */
    function selected() { return MOD_KEYS.filter(function (k) { return state.mods[k]; }); }
    function syncKnobs() {
      METHODS.forEach(function (me) { me.knobs.forEach(function (k) { var inp = rows[me.id].inputs[k.k]; if (inp && +inp.value !== state.knobs[me.id][k.k]) inp.value = state.knobs[me.id][k.k]; }); });
    }
    function scopeText(me, M) {
      if (me.scope === 'sel') return me.alias || '';
      if (me.scope === 'ia3') return 'HF default ' + ia3Targets(M).join(', ');
      if (me.scope === 'bitfit') return M.qkvBias ? 'biases: q, k, v' : 'biases: none here';
      if (me.scope === 'prompt') return 'input sequence';
      if (me.scope === 'prefix') return 'K, V of every layer';
      if (me.scope === 'houlsby') return 'two per layer';
      if (me.scope === 'full') return 'every weight';
      return '';
    }

    var lastFocus = null;
    function refresh(focusChanged) {
      var M = MODELS[state.model], sel = selected(), N = totalParams(M);
      METHODS.forEach(function (me) { state.res[me.id] = compute(me, M, sel, state.knobs[me.id]); });

      /* controls readouts */
      var h = headDim(M);
      archEl.innerHTML = [
        ['d<sub>model</sub>', nf(M.d)], ['d<sub>ff</sub>', nf(M.ff)], ['layers', M.L],
        ['heads', M.heads], ['kv heads', M.kv], ['d<sub>head</sub>', h],
        ['vocab', nf(M.V)], ['embed', M.tie ? 'tied' : 'untied'], ['bias', M.qkvBias ? 'q, k, v' : 'none']
      ].map(function (p) { return '<span><i>' + p[0] + '</i>' + p[1] + '</span>'; }).join('');
      var match = N === M.hfTotal;
      totalEl.innerHTML = '<b>' + nf(N) + '</b> parameters<br><span class="' + (match ? 'ok' : 'bad') + '">' + (match ? '✓ equals the HF safetensors total' + (M.buffers ? ' less ' + nf(M.buffers) + ' rotary buffers' : '') : '≠ HF total ' + nf(M.hfTotal)) + '</span>';
      MOD_KEYS.forEach(function (k) {
        var mo = modByKey(M, k);
        modBtns[k].setAttribute('aria-pressed', String(!!state.mods[k]));
        modBtns[k].title = mo.name + ': ' + mo.m + ' × ' + mo.n + ' (m × n)';
        modBtns[k].setAttribute('aria-label', mo.name + ', ' + mo.m + ' by ' + mo.n);
      });
      var preset = null;
      ['qv', 'attn', 'all'].forEach(function (p) { if (PRESETS[p].length === sel.length && PRESETS[p].every(function (k) { return state.mods[k]; })) preset = p; });
      presetSeg.set(preset);
      modSum.textContent = sel.length
        ? sel.length + ' per layer × ' + M.L + ' layers = ' + (sel.length * M.L) + ' matrices · ' + sel.map(function (k) { var mo = modByKey(M, k); return k + '\u00a0' + mo.m + '×' + mo.n; }).join(', ')
        : 'No matrix selected, so the low-rank rows train nothing.';
      ctxEl.innerHTML = 'on <b>' + A.esc(M.label) + '</b> · ' + A.fmtCount(N) + ' parameters · adapting <b>' + (sel.length ? sel.join(', ') : 'nothing') + '</b> → ' + (sel.length * M.L) + ' matrices · log scale';

      /* rows */
      METHODS.forEach(function (me) {
        var R = rows[me.id], res = state.res[me.id];
        R.row.classList.toggle('off', !state.on[me.id]);
        R.row.classList.toggle('focus', state.focus === me.id);
        R.nameBtn.setAttribute('aria-pressed', String(state.focus === me.id));
        R.sw.style.background = 'var(' + kindToken(kindOf(me)) + ')';
        R.scope.textContent = scopeText(me, M);
        R.scope.style.display = R.scope.textContent ? '' : 'none';
        if (res.invalid) {
          R.nP.textContent = '—'; R.nP.className = 'na'; R.nPct.textContent = '—'; R.nPct.className = 'na';
          R.nD.textContent = '—'; R.nD.className = 'na'; R.nG.textContent = '—'; R.nG.className = 'na';
          return;
        }
        R.nP.textContent = A.fmtCount(res.P); R.nP.className = '';
        R.nPct.textContent = pctTxt(res.P / N); R.nPct.className = '';
        if (res.d == null) { R.nD.textContent = '—'; R.nD.className = 'na'; R.nG.textContent = '—'; R.nG.className = 'na'; }
        else {
          R.nD.innerHTML = (res.bound ? '≤' : '') + A.esc(A.fmtCount(res.d)) + (res.num ? '<span class="cr-dag" title="checked numerically, not proved">†</span>' : '');
          R.nD.className = '';
          R.nG.textContent = (res.bound ? '≥' : '') + A.fmtCount(res.gap); R.nG.className = res.gap > 0 ? 'pos' : res.bound ? '' : 'zero';
        }
      });
      drawBars(true);
      renderFocus(focusChanged || lastFocus !== state.focus);
      lastFocus = state.focus;
    }
    var refreshSoon = (function () {
      var pending = false;
      return function () {
        if (pending) return; pending = true;
        (window.requestAnimationFrame || setTimeout)(function () { pending = false; refresh(); });
      };
    })();

    /* ---------------- bars ---------------- */
    var narrow = false;
    function scaleFor(w, N) {
      var vals = [];
      METHODS.forEach(function (me) {
        var r = state.res[me.id];
        if (!state.on[me.id] || !r || r.invalid) return;
        if (r.P > 0) vals.push(r.P);
        if (r.d > 0) vals.push(r.d);
      });
      var mn = vals.length ? Math.min.apply(null, vals) : 1000;
      var lo = Math.pow(10, Math.floor(Math.log10(mn)));
      var mx = vals.length ? Math.max.apply(null, vals) : N;
      var hi = Math.pow(10, Math.ceil(Math.log10(Math.max(N, mx))));
      if (hi / lo < 100) lo = hi / 100;
      lo = Math.max(1, lo);
      var x = function (v) { return Math.max(0, Math.min(w, (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * w)); };
      x.lo = lo; x.hi = hi;
      return x;
    }
    function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
    var mctx = document.createElement('canvas').getContext('2d');
    function decades(x) { var t = []; for (var e = Math.round(Math.log10(x.lo)); e <= Math.round(Math.log10(x.hi)); e++) t.push(Math.pow(10, e)); return t; }

    function pctTick(e) {            /* e = log10 of the share: 0 -> 100%, -1 -> 10%, -4 -> 0.01% */
      var k = e + 2;
      if (k >= 0) return Math.pow(10, k) + '%';
      return '0.' + new Array(-k).join('0') + '1%';
    }
    function drawBars(animate) {
      if (!state.res[METHODS[0].id]) return;
      var cell = hdr.children[2];
      var w = cell.getBoundingClientRect().width;   /* fractional, so the 1:1 viewBox never rescales the labels */
      if (!w) return;
      var M = MODELS[state.model], N = totalParams(M), x = scaleFor(w, N), ticks = decades(x);
      var cRule = A.css('--rule'), cInk3 = A.css('--ink-3'), cInk2 = A.css('--ink-2'), cInk = A.css('--ink'), cSeal = A.css('--seal-ink');
      /* fonts from the page tokens at draw time (tick values in mono at 11px, row messages in the UI face at 12px) */
      var monoFont = A.css('--f-mono'), uiFont = A.css('--f-ui'), TICK = 11;
      mctx.font = TICK + 'px ' + monoFont;
      function txt(parent, str, xx, yy, anchor, color) {
        var t = svgEl('text', { x: xx, y: yy, 'text-anchor': anchor, style: 'font-family:' + monoFont + ';font-size:' + TICK + 'px;font-weight:400;font-variant-numeric:tabular-nums;letter-spacing:0;text-transform:none;fill:' + color });
        t.textContent = str; parent.appendChild(t); return t;
      }
      /* place labels greedily (most important first) so that none overlap; widths measured in the live font */
      function placeLabels(parent, items, yy, ticksUp, hgt) {
        var kept = [];
        items.forEach(function (it) {
          var wd = Math.max(it.text.length * TICK * 0.6, mctx.measureText(it.text).width), anchor = 'middle', l = it.x - wd / 2;
          if (l < 0) { anchor = 'start'; l = it.x; }
          if (l + wd > w) { anchor = 'end'; l = it.x - wd; }
          var r = l + wd;
          for (var i = 0; i < kept.length; i++) if (!(r + 7 < kept[i][0] || l - 7 > kept[i][1])) return;
          kept.push([l, r]);
          txt(parent, it.text, it.x, yy, anchor, it.color || cInk2);
        });
        items.forEach(function (it) {
          parent.appendChild(svgEl('line', { x1: it.x, x2: it.x, y1: ticksUp ? hgt - 5 : 0, y2: ticksUp ? hgt : 5, stroke: cInk3, 'stroke-width': 1 }));
        });
      }

      /* top axis: counts (log10) */
      var hh = cell.getBoundingClientRect().height || 26;
      clear(axisSvg);
      axisSvg.setAttribute('viewBox', '0 0 ' + w + ' ' + hh);
      var topItems = ticks.map(function (t) { return { x: x(t), text: logLabel(t) }; });
      var order = topItems.map(function (_, i) { return i; }).sort(function (a, b) { return (a % 2) - (b % 2) || a - b; });
      placeLabels(axisSvg, order.map(function (i) { return topItems[i]; }), hh - 9, true, hh);

      /* bottom axis: share of the model, ticks at N·10^e */
      var fcell = ftr.children[1], fh = fcell.getBoundingClientRect().height || 26;
      clear(pctSvg);
      pctSvg.setAttribute('viewBox', '0 0 ' + w + ' ' + fh);
      var pItems = [];
      for (var e = 0; e > -14; e--) { var v = N * Math.pow(10, e); if (v < x.lo * 0.999) break; pItems.push({ x: x(v), text: pctTick(e), color: e === 0 ? cInk : cInk2 }); }
      placeLabels(pctSvg, pItems, 17, false, fh);
      ftrLabel.textContent = 'share of ' + M.label;

      var reduce = A.reducedMotion() || !hasD3;
      var anim = !!animate && !reduce;
      function grow(node, attrs) {
        if (!anim) { for (var k in attrs) node.setAttribute(k, attrs[k][1]); return; }
        var sel = d3.select(node);
        for (var k2 in attrs) sel.attr(k2, attrs[k2][0]);
        var tr = sel.transition().duration(280).ease(d3.easeCubicOut);
        for (var k3 in attrs) tr.attr(k3, attrs[k3][1]);
      }
      METHODS.forEach(function (me) {
        var R = rows[me.id], res = state.res[me.id], svg = R.svg;
        var bh = narrow ? 10 : 11;
        clear(svg);
        var sh = svg.parentNode.getBoundingClientRect().height || (narrow ? 16 : 30);
        svg.setAttribute('viewBox', '0 0 ' + w + ' ' + sh);
        var textOnly = state.on[me.id] && (res.invalid || res.P === 0);   /* a message row: keep the grid out of the words */
        if (!textOnly) ticks.forEach(function (t) { svg.appendChild(svgEl('line', { x1: x(t), x2: x(t), y1: 0, y2: sh, stroke: cRule, 'stroke-width': 1 })); });
        svg.appendChild(svgEl('line', { x1: x(N), x2: x(N), y1: 0, y2: sh, stroke: cInk3, 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
        var cy = sh / 2;
        function label(t, color) {
          var el2 = svgEl('text', { x: 3, y: cy + 4.2, style: 'font-family:' + uiFont + ';font-size:12px;font-weight:400;fill:' + color + ';paint-order:stroke;stroke:var(--paper-2);stroke-width:4px;stroke-linejoin:round' });
          el2.textContent = t; svg.appendChild(el2);
        }
        var prev = R.prev || { P: 0, D: 0 };
        R.prev = { P: 0, D: 0 };
        if (!state.on[me.id]) { svg.setAttribute('aria-label', me.name + ': hidden'); return; }
        if (res.invalid) { label(res.invalid, cSeal); svg.setAttribute('aria-label', me.name + ': ' + res.invalid); return; }
        if (res.P === 0) {
          label(me.scope === 'sel' ? 'no matrix selected' : 'no bias terms in this model, so nothing to train', cInk2);
          svg.setAttribute('aria-label', me.name + ': 0 trainable parameters'); return;
        }
        var col = colorOf(me), xP = x(res.P), xD = res.d != null && res.d > 0 ? x(res.d) : 0;
        R.prev = { P: xP, D: xD };
        var g = svgEl('g', { class: 'cr-g-bars' });
        g.appendChild(svgEl('rect', { x: 0, y: 0, width: w, height: sh, fill: 'transparent' }));  /* hover target */
        var pale = svgEl('rect', { x: 0, y: cy - bh / 2, height: bh, rx: 2, fill: col, 'fill-opacity': 0.26 });
        if (res.d == null) { pale.setAttribute('stroke', col); pale.setAttribute('stroke-dasharray', '3 2'); pale.setAttribute('fill-opacity', 0.16); }
        g.appendChild(pale);
        grow(pale, { width: [prev.P, xP] });
        if (res.d != null) {
          var solid = svgEl('rect', { x: 0, y: cy - bh / 2, height: bh, rx: 2, fill: col });
          g.appendChild(solid);
          grow(solid, { width: [prev.D, xD] });
        }
        if (res.d != null && res.gap > 0) {
          var tick = svgEl('line', { y1: cy - bh / 2 - 2.5, y2: cy + bh / 2 + 2.5, stroke: cSeal, 'stroke-width': 1.5 });
          g.appendChild(tick);
          grow(tick, { x1: [Math.max(0, prev.P - 0.75), xP - 0.75], x2: [Math.max(0, prev.P - 0.75), xP - 0.75] });
        }
        svg.appendChild(g);
        var noD = me.scope === 'sel' ? 'no closed form for reachable directions' : 'acts outside weight space, so no reachable-direction count';
        var aria = me.name + ': ' + nf(res.P) + ' trainable parameters (' + pctTxt(res.P / N) + ' of the model); ' + (res.d == null ? noD : (res.bound ? 'at most ' : '') + nf(res.d) + ' reachable directions' + (res.num ? ' (checked numerically)' : '') + ', gauge gap ' + (res.bound ? 'at least ' : '') + nf(res.gap));
        svg.setAttribute('aria-label', aria);
        g.addEventListener('mousemove', function (ev) {
          A.tip.show('<div class="t">' + A.esc(me.name) + '</div><div class="mono num" style="font-size:.8rem;line-height:1.55">trained ' + nf(res.P) + '<br>' + (res.d == null ? 'reached: ' + (me.scope === 'sel' ? 'no closed form' : 'not in weight space') : 'reached ' + (res.bound ? '≤ ' : '') + nf(res.d) + (res.num ? ' †' : '') + '<br>gauge gap ' + (res.bound ? '≥ ' : '') + nf(res.gap)) + '<br>' + pctTxt(res.P / N) + ' of ' + A.esc(M.label) + '</div>', ev);
        });
        g.addEventListener('mouseleave', function () { A.tip.hide(); });
      });
    }

    /* ---------------- focus panel ---------------- */
    function axisValueName(axisKey, valKey) {
      var ax = (A.data().axes || []).filter(function (a) { return a.key === axisKey; })[0];
      if (!ax || !ax.values) return valKey;
      var v = ax.values.filter(function (x) { return x.key === valKey; })[0];
      return v ? v.name : valKey;
    }
    function renderFocus(full) {
      var me = null; METHODS.forEach(function (m) { if (m.id === state.focus) me = m; });
      if (!me) return;
      var M = MODELS[state.model], res = state.res[me.id], idx = METHODS.indexOf(me) + 1;
      if (full) {
        var who = me.data.map(function (id) {
          var m = A.method(id); if (!m) return null;
          var t = A.esc((m.authors || m.name) + ' ' + (m.year || ''));
          return m.url ? '<a href="' + A.esc(m.url) + '" target="_blank" rel="noopener">' + t + '</a>' : t;
        }).filter(Boolean).join(' · ');
        fHead.innerHTML = '<span class="cf-kicker">Per adapted matrix · source ' + idx + '</span><h4 class="cf-title">' + A.esc(me.name) + '</h4>' + (who ? '<span class="cf-meta">' + who + '</span>' : '<span class="cf-meta">the terminal object of the slice</span>');
        if (window.MathJax && MathJax.typesetClear) { try { MathJax.typesetClear([fForms, fCoords]); } catch (e) {} }
        var rowsF = [
          ['trainable', me.tex.P, me.srcP],
          ['reachable', me.tex.d, me.srcD]
        ];
        if (me.tex.g) rowsF.push(['gauge gap', me.tex.g, 'Def I.5: the gauge waste g(M) = |M| − d(M), the dimension of a generic fibre.']);
        fForms.innerHTML = rowsF.map(function (r) {
          return '<span class="cf-k">' + r[0] + '</span><span class="cf-f">\\(' + r[1] + '\\)</span><span class="cf-s">' + A.esc(r[2]) + '</span>';
        }).join('');
        A.typeset(fForms);
        var dm = me.data.length ? A.method(me.data[0]) : null;
        if (me.ref) fCoords.innerHTML = '<b>Atlas coordinates</b> · terminal object: ρ = id, regular base point, no gauge';
        else if (dm && dm.coords) {
          var c = dm.coords, keys = ['kind', 'shape', 'base_point', 'gauge', 'covariance', 'merge', 'rebasing'];
          fCoords.innerHTML = '<b>Atlas coordinates</b> · ' + keys.filter(function (k) { return c[k] != null; }).map(function (k) { return A.esc(k.replace('_', ' ')) + ' <b>' + A.esc(axisValueName(k, c[k])) + '</b>'; }).join(' · ');
        } else fCoords.innerHTML = '<b>Atlas coordinates</b> · classification pending';
        if (dm && dm.categorical_reading) fCoords.innerHTML += '<span class="cf-read">' + A.esc(dm.categorical_reading) + '</span>';
        A.typeset(fCoords);
      }
      /* numbers table */
      var html = '<table class="cf-table"><thead><tr><th>' + (me.scope === 'sel' || me.scope === 'ia3' ? 'matrix' : 'block') + '</th><th class="shp">' + (me.scope === 'prompt' || me.scope === 'prefix' || me.scope === 'houlsby' ? 'shape' : 'm × n') + '</th><th class="num cnt">count</th><th class="st">structure</th><th class="num">|M| each</th><th class="num">d each</th><th class="num">gap each</th></tr></thead><tbody>';
      if (res.invalid) html += '<tr><td colspan="7" style="color:var(--seal)">' + A.esc(res.invalid) + '</td></tr>';
      var dag = '<span class="cr-dag" title="checked numerically, not proved">†</span>';
      res.units.forEach(function (u) {
        var dTxt = u.d == null ? '—' : (u.bound ? '≤ ' : '') + nf(u.d) + (u.num ? dag : ''), gTxt = u.d == null ? '—' : (u.bound ? '≥ ' : '') + nf(u.P - u.d);
        html += '<tr><td>' + A.esc(u.label) + '<span class="cf-shp">' + A.esc(u.shape) + '</span></td><td class="shp">' + A.esc(u.shape) + '</td><td class="num cnt">' + u.count + '</td><td class="st">' + A.esc(u.s || '') + '</td><td class="num">' + nf(u.P) + '</td><td class="num">' + dTxt + '</td><td class="num">' + gTxt + '</td></tr>';
      });
      if (!res.invalid) {
        html += '<tr class="tot"><td>total</td><td class="shp"></td><td class="num cnt">' + res.mats + '</td><td class="st">' + pctTxt(res.P / totalParams(M)) + ' of the model</td><td class="num">' + nf(res.P) + '</td><td class="num">' + (res.d == null ? '—' : (res.bound ? '≤ ' : '') + nf(res.d) + (res.num ? dag : '')) + '</td><td class="num">' + (res.d == null ? '—' : (res.bound ? '≥ ' : '') + nf(res.gap)) + '</td></tr>';
      }
      html += '</tbody></table>';
      fTableWrap.innerHTML = html;
      var notes = res.notes.slice();
      var why = []; res.units.forEach(function (u) { if (u.d == null && u.why && why.indexOf(u.why) < 0) why.push(u.why); });
      if (me.scope === 'sel' && why.length && res.units.some(function (u) { return u.d != null; })) notes.push('d is shown only where the framework’s formula applies: ' + why.join('; ') + '.');
      if (res.P === 0 && me.scope === 'bitfit') notes.push(M.label + ' has no bias terms (bias-free linear layers, RMSNorm without bias), so BitFit has nothing to train.');
      if (me.scope === 'sel' && !res.units.length && !res.invalid) notes.push('Select at least one matrix above.');
      fNotes.innerHTML = notes.map(function (n) { return '<li>' + A.esc(n) + '</li>'; }).join('');
      if (res.invalid) fNotes.insertAdjacentHTML('afterbegin', '<li class="warn">PEFT would refuse this configuration: ' + A.esc(res.invalid) + '.</li>');
      fNotes.style.display = fNotes.children.length ? '' : 'none';
    }

    /* ---------------- layout + theme ---------------- */
    function layout() {
      var w = stage.clientWidth;
      var nw = w < 800;   /* below this the seven-column row leaves the bars too little room at the 12px type sizes */
      if (nw !== narrow) { narrow = nw; stage.classList.toggle('is-narrow', narrow); }
      drawBars(false);
    }
    modelSel.value = String(state.model);
    syncKnobs();
    refresh(true);
    layout();
    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(layout, 60)).observe(stage);
    else window.addEventListener('resize', A.debounce(layout, 100));
    A.onTheme(function () { drawBars(false); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { drawBars(false); });
  });
})();
