/* Fig. "lens" — Tangent and cotangent: the lens. Exposé IV of theory/framework.md (Def IV.1, Prop IV.2,
   Prop IV.3, Obs IV.4), with the corrections of the round-2 referee pass (theory/round2_verdicts.json):
     - Flora as published compresses only optimizer state and lets Adafactor act in the full space, so it is not of
       Def IV.1 form; the covered random-projector method is GaLore with a random P (Torroba-Hennigen et al. 2025);
     - the full-batch gradient-subspace field Hom(R^n, U_r(grad L)) is generically non-involutive for every 1 <= r < min(m, n);
     - LISA gets no cone truncation but does save activations on the cone (frozen linears keep nothing).

   Panel A — the cotangent family as projector (compression c_t) x schedule (anchor a_t). Each method has a card with
   its update rule in the framework's notation and its verdict, and two live float64 checks run on mount:
     Prop IV.2: GaLore with a fixed projector per period vs one-sided LoRA W_k + P_k C re-based every period
                (Adam, state carried / reset / AdamW / full-gradient clipping);
     Prop IV.3: the Lie bracket of the theta-dependent field D_W = Hom(R^n, U_r(grad L(W))), least squares, with
                D Pi from first-order perturbation of the top-r eigenspace of G G^T.
   Panel B — the backprop cone (Obs IV.4) on a schematic decoder (Llama-style blocks). The cone is computed by
   forward reachability from the trainable boxes; the kept tensors follow Obs IV.4 box by box; the memory bars are
   the formulas printed in the footnote. Every number is computed here; nothing is typed in by hand. */
(function () {
  'use strict';

  /* ======================================================================================
     Panel B data: model configs (HF config.json values) and box layout of one decoder block
     ====================================================================================== */
  var MODELS = [
    { id: 'llama3-8b', name: 'Llama-3-8B', L: 32, d: 4096, f: 14336, kv: 1024, h: 32, V: 128256 },
    { id: 'llama2-7b', name: 'Llama-2-7B', L: 32, d: 4096, f: 11008, kv: 4096, h: 32, V: 32000 },
    { id: 'mistral-7b', name: 'Mistral-7B', L: 32, d: 4096, f: 14336, kv: 1024, h: 32, V: 32000 },
    { id: 'llama3-70b', name: 'Llama-3-70B', L: 80, d: 8192, f: 28672, kv: 1024, h: 64, V: 128256 }
  ];
  /* rows of a block, in topological order (residual adds and RoPE are deterministic linear: they keep nothing) */
  var ROWS = [
    { key: 'n1', lab: 'norm', kind: 'norm', name: 'RMSNorm (attention)' },
    { key: 'q', lab: 'q', kind: 'lin', name: 'W<sub>q</sub>' },
    { key: 'k', lab: 'k', kind: 'lin', name: 'W<sub>k</sub>' },
    { key: 'v', lab: 'v', kind: 'lin', name: 'W<sub>v</sub>' },
    { key: 'at', lab: 'attn', kind: 'nl', name: 'attention core softmax(qkᵀ)v' },
    { key: 'o', lab: 'o', kind: 'lin', name: 'W<sub>o</sub>' },
    { key: 'n2', lab: 'norm', kind: 'norm', name: 'RMSNorm (MLP)' },
    { key: 'gt', lab: 'gate', kind: 'lin', name: 'W<sub>gate</sub>' },
    { key: 'up', lab: 'up', kind: 'lin', name: 'W<sub>up</sub>' },
    { key: 'ac', lab: 'σ⊙', kind: 'nl', name: 'SwiGLU product silu(g)⊙u' },
    { key: 'dn', lab: 'down', kind: 'lin', name: 'W<sub>down</sub>' }
  ];
  var LIN = ['q', 'k', 'v', 'o', 'gt', 'up', 'dn'];
  var INP = { q: 'xn', k: 'xn', v: 'xn', o: 'o', gt: 'hn', up: 'hn', dn: 'a' };
  /* tensors a block may keep: c = symbolic size per position, B = bytes per element */
  var TL = {
    x: { c: { d: 1 }, B: 2, t: 'x' }, r1: { c: { one: 1 }, B: 4, t: '1/rms' }, xn: { c: { d: 1 }, B: 2, t: 'x̂' },
    q: { c: { d: 1 }, B: 2, t: 'q' }, k: { c: { kv: 1 }, B: 2, t: 'k' }, v: { c: { kv: 1 }, B: 2, t: 'v' },
    lse: { c: { h: 1 }, B: 4, t: 'lse' }, P: { c: { hS: 1 }, B: 2, t: 'softmax' }, o: { c: { d: 1 }, B: 2, t: 'o' },
    h: { c: { d: 1 }, B: 2, t: 'h' }, r2: { c: { one: 1 }, B: 4, t: '1/rms' }, hn: { c: { d: 1 }, B: 2, t: 'ĥ' },
    g: { c: { f: 1 }, B: 2, t: 'g' }, u: { c: { f: 1 }, B: 2, t: 'u' }, a: { c: { f: 1 }, B: 2, t: 'a' }
  };
  LIN.forEach(function (k) { TL['z_' + k] = { c: { r: 1 }, B: 2, t: 'Ax' }; });
  var TORDER = ['x', 'r1', 'xn', 'q', 'k', 'v', 'lse', 'P', 'o', 'h', 'r2', 'hn', 'g', 'u', 'a'].concat(LIN.map(function (k) { return 'z_' + k; }));

  var PRESETS = [
    { id: 'full', label: 'Full FT' },
    { id: 'lora-qv', label: 'LoRA q,v' },
    { id: 'lora-all', label: 'LoRA all' },
    { id: 'bitfit', label: 'BitFit' },
    { id: 'lisa', label: 'LISA' },
    { id: 'topk', label: 'Top-k' },
    { id: 'prompt', label: 'Prompt' },
    { id: 'mezo', label: 'MeZO' }
  ];
  var RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256];

  /* ======================================================================================
     Panel A data: the projector x schedule table (Exposé IV, with round-2 corrections)
     ====================================================================================== */
  var AROWS = [
    { id: 'rand', lab: 'random', sub: 'P drawn at random' },
    { id: 'svd', lab: 'gradient / data SVD', sub: 'P read off an SVD' },
    { id: 'coord', lab: 'coordinate blocks', sub: 'P = ι<sub>S</sub>' },
    { id: 'two', lab: 'two-sided, state-dependent', sub: 'contrast row' }
  ];
  var ACOLS = [
    { id: 'frozen', lab: 'frozen', sub: 'a_t = a_0' },
    { id: 'period', lab: 're-based every T', sub: 'a_t = a_kT' },
    { id: 'step', lab: 'every step', sub: 'T = 1' }
  ];
  var CELLNOTE = {
    'rand|step': 'rank 1, global', 'svd|frozen': 'first order only', 'two|step': '—'
  };
  var VERDICT = {
    fwd: 'Forward lens: a reparametrisation outright',
    per: 'Secretly a reparametrisation, period by period',
    out: 'Not covered: outside Def IV.1'
  };
  var IV2_HYP = [
    'the optimizer reads only the gradients it is fed: SGD, heavy-ball or Nesterov momentum, Adam/AMSGrad, or Adafactor with <code>scale_parameter=False</code>, the same implementation on both sides;',
    'both sides start each period from the same optimizer state (fresh, or carried over);',
    'weight decay \\(0\\) and no global-norm clipping of the full gradient;'
  ];
  var IV2_BASIS = 'the frozen factor is exactly the method’s projector \\(P_k\\) (signs and basis), not only its span.';
  var IV2_BREAK = 'Broken by AdamW with \\(\\lambda>0\\), LAMB/LARS trust ratios, parameter-scaled Adafactor (HF default <code>scale_parameter=True</code>), full-gradient clipping, or a state reset.';

  var METHODS = [
    {
      id: 'lora-fa', name: 'LoRA-FA', row: 'rand', col: 'frozen', v: 'fwd', link: 'lora-fa',
      cite: 'Zhang et al. 2023', url: 'https://arxiv.org/abs/2308.03303', ax: '2308.03303',
      vline: 'A forward linear method; it equals its backward form exactly (Prop IV.2).',
      c: '\\(c(G)=GA_0^{\\top}\\)', a: '\\(a(S)=SA_0\\)', cnote: '\\(A_0\\in\\mathbb R^{r\\times n}\\) random, frozen',
      upd: '\\[\\begin{aligned}&W=W_0+BA_0:\\\\ &W\\leftarrow W-\\eta\\,\\mathrm{opt}\\big(GA_0^{\\top}\\big)\\,A_0\\end{aligned}\\]',
      read: 'Training \\(B\\) is the backward method with \\(c=\\mathcal L^{*}\\), \\(a=\\mathcal L\\) for \\(\\mathcal L(C)=CA_0\\), so the two give the same \\(W\\)-trajectory (Prop IV.2). Under SGD its cometric is \\(K(G)=GA_0^{\\top}A_0\\); a preconditioned or adaptive step changes \\(K\\).',
      hyp: IV2_HYP,
      cone: '\\(B\\) keeps \\(A_0x\\) (\\(r\\) numbers per position) instead of \\(x\\) (\\(n\\) numbers), with LoRA dropout 0 (Obs IV.4).'
    },
    {
      id: 'galore-rand', name: 'GaLore, random P', row: 'rand', col: 'period', v: 'per',
      cite: 'Torroba-Hennigen, Lang, Guo, Kim 2025, Table 1', url: 'https://arxiv.org/abs/2502.13811', ax: '2502.13811',
      vline: 'One-sided LoRA with a random frozen factor, re-based every \\(T\\) steps.',
      c: '\\(c_t=P_k^{\\top}\\)', a: '\\(a_t=P_k\\)', cnote: '\\(P_k\\) random (Gaussian, Rademacher or semi-orthogonal), redrawn every \\(T\\) steps',
      upd: '\\[\\begin{aligned}&W\\leftarrow W-\\eta\\,P_k\\,\\mathrm{opt}\\big(P_k^{\\top}G_t\\big),\\\\ &kT\\le t\\lt(k+1)T\\end{aligned}\\]',
      read: 'The anchor is constant on each period, so its distribution is constant and trivially involutive with affine leaves (Prop IV.3). Prop IV.2 then makes each period one-sided LoRA \\(W_k+P_kC\\) with frozen \\(B=P_k\\), merged and re-based every \\(T\\) steps. The frozen-\\(A\\) SGD case is Hao, Cao and Mou 2024, eqs. (19)–(20).',
      hyp: IV2_HYP.concat([IV2_BASIS]), brk: IV2_BREAK,
      cone: 'Reverse route: every weight trains, so its cone and kept activations are those of full fine-tuning; the saving is optimizer state.'
    },
    {
      id: 'flora', name: 'Flora', row: 'rand', col: 'period', v: 'out', note: 'as published',
      cite: 'Hao, Cao, Mou 2024', url: 'https://arxiv.org/abs/2402.03293', ax: '2402.03293',
      vline: 'As published, Flora is not a backward method of Def IV.1, so Prop IV.2 does not apply.',
      c: 'random \\(A_k\\) compress only optimizer state', a: 'decompression by \\(A_k\\), mixed with a full-space step', cnote: 'momentum or gradient accumulator stored as \\(MA_k^{\\top}\\), redrawn every \\(\\kappa\\) steps',
      upd: '\\[\\text{step}=\\mathrm{opt}\\big(a\\,c\\,\\nabla L\\big)\\ \\neq\\ a\\big(\\mathrm{opt}(c\\,\\nabla L)\\big)\\]',
      read: 'Flora compresses the momentum (or the gradient-accumulation buffer) with a random projection and decompresses it. The result goes to Adafactor acting in the full space, and on resampling the momentum is transported, \\(M\\mapsto MA_{\\mathrm{old}}A_{\\mathrm{new}}^{\\top}\\). Its step leaves \\(\\mathrm{im}\\,a\\), so no one-sided LoRA \\(W_k+CA_k\\) reproduces its trajectory. Covered instead: GaLore with a random projector (the chip beside it). An idealised Flora with an SGD or heavy-ball base (paper Alg. 2) has Def IV.1 form within each period.',
      cone: 'Reverse route: every weight trains; the saving is optimizer state.'
    },
    {
      id: 'mezo', name: 'MeZO', row: 'rand', col: 'step', v: 'per', link: 'mezo',
      cite: 'Malladi et al. 2023', url: 'https://arxiv.org/abs/2305.17333', ax: '2305.17333',
      vline: 'Each step is a step of a rank-1 linear method, up to finite-difference error.',
      c: '\\(c_t(g)=\\langle z_t,g\\rangle\\), estimated by \\(\\hat g_t\\)', a: '\\(a_t(\\lambda)=\\lambda z_t\\)', cnote: '\\(z_t\\sim\\mathcal N(0,I)\\) on all of \\(\\theta\\), regenerated from a seed',
      upd: '\\[\\begin{aligned}&\\theta\\leftarrow\\theta-\\eta\\,\\hat g_t\\,z_t,\\\\ &\\hat g_t=\\frac{L(\\theta+\\varepsilon z_t)-L(\\theta-\\varepsilon z_t)}{2\\varepsilon}\\end{aligned}\\]',
      read: 'Each step is a step of the rank-1 linear forward method \\(\\lambda\\mapsto\\theta_t+\\lambda z_t\\), up to the SPSA (finite-difference) error. At \\(T=1\\) this period-wise reading holds for any update that lies in \\(\\mathrm{im}\\,a\\), so on its own it says little; MeZO differs from LoRA in its schedule, and integrability plays no part. Its anchor depends on step-wise randomness, so it is not a time-independent distribution. With weight decay \\(>0\\) the step leaves the line.',
      hyp: ['up to the finite-difference error of \\(\\hat g_t\\);', 'weight decay \\(0\\).'], hl: 'The reading holds',
      cone: '\\(k=1\\) with no reverse pass: no backward cone (Obs IV.4). \\(\\hat g_t\\) is unbiased for the gradient of the Gaussian-smoothed loss, with \\(O(\\varepsilon^2)\\) bias against \\(\\nabla L\\) for a Lipschitz Hessian; the exact-JVP forward gradient satisfies \\(\\mathbb E[(\\nabla L\\cdot z)z]=\\nabla L\\).'
    },
    {
      id: 'eva', name: 'EVA', row: 'svd', col: 'frozen', v: 'fwd',
      cite: 'Paischer et al. 2024', url: 'https://arxiv.org/abs/2410.07170', ax: '2410.07170',
      vline: 'A forward lens; it sits in this cell at first order only.',
      c: '\\(c=\\cdot A_0^{\\top}\\) at the first step', a: '\\(a=\\cdot A_0\\) at the first step', cnote: '\\(A_0\\) = top-\\(r\\) right singular vectors of minibatches of layer inputs',
      upd: '\\[\\begin{aligned}&W=W_0+BA,\\quad B_0=0,\\\\ &A_0=V_{[:,1:r]}^{\\top}\\ \\text{of}\\ \\mathrm{SVD}(X)\\end{aligned}\\]',
      read: 'LoRA with a data-SVD initialisation (EVA also redistributes ranks across layers). Since \\(B_0=0\\), \\(\\partial L/\\partial A=B^{\\top}G=0\\) at the first step, so that step is the frozen-projector method \\(c=\\cdot A_0^{\\top}\\), \\(a=\\cdot A_0\\). Afterwards both factors move and it is two-sided LoRA.',
      cone: 'As LoRA: \\(A\\) keeps its input, \\(B\\) keeps \\(Ax\\).'
    },
    {
      id: 'lora-ga', name: 'LoRA-GA', row: 'svd', col: 'frozen', v: 'fwd',
      cite: 'Wang et al. 2024', url: 'https://arxiv.org/abs/2407.05000', ax: '2407.05000',
      vline: 'A forward lens; it sits in this cell at first order only.',
      c: 'frames from \\(\\mathrm{SVD}(G_0)\\)', a: 'the same frames', cnote: 'read off the SVD of the first full gradient \\(G_0\\)',
      upd: '\\[\\begin{aligned}&W=(W_0-sB_0A_0)+sBA,\\\\ &(B_0,A_0)\\ \\text{from}\\ \\mathrm{SVD}(G_0)\\end{aligned}\\]',
      read: 'LoRA with a split initialisation: \\(W_0\\) is offset so that \\(\\rho(B_0,A_0)=\\theta_0\\), with both factors read off the SVD of the first full gradient. Its first step moves along gradient-SVD frames, which is why it sits in this cell; afterwards both factors move.',
      cone: 'As LoRA: \\(A\\) keeps its input, \\(B\\) keeps \\(Ax\\).'
    },
    {
      id: 'galore', name: 'GaLore', row: 'svd', col: 'period', v: 'per',
      cite: 'Zhao et al. 2024', url: 'https://arxiv.org/abs/2403.03507', ax: '2403.03507',
      vline: 'Secretly one-sided LoRA with a moving frame (Prop IV.2).',
      c: '\\(c_t=P_k^{\\top}\\)', a: '\\(a_t=P_k\\)', cnote: '\\(P_k=U_{[:,1:r]}\\big(\\mathrm{SVD}(G_{kT})\\big)\\), refreshed every \\(T\\) steps; left projection shown (\\(m\\lt n\\)), GaLore projects on the right when \\(m\\ge n\\)',
      upd: '\\[\\begin{aligned}&W\\leftarrow W-\\eta\\,\\alpha\\,P_k\\,\\mathrm{opt}\\big(P_k^{\\top}G_t\\big),\\\\ &kT\\le t\\lt(k+1)T\\end{aligned}\\]',
      read: 'With \\(\\mathcal L(C)=P_kC\\), Prop IV.2 identifies each period with one-sided LoRA \\(W_k+P_kC\\) (frozen \\(B=P_k\\)), merged and re-based every \\(T\\) steps, with \\(C\\)’s optimizer state (moments and step counter) carried over unchanged. That carry-over is neither a reset (ReLoRA) nor a transport (LDAdam); moments computed in \\(P_{k-1}\\)’s coordinates are reused as if they were in \\(P_k\\)’s. The scale \\(\\alpha\\) multiplies the optimizer output, so it is a learning-rate factor. Credit: Torroba-Hennigen, Lang, Guo, Kim 2025 (Thm 1, Cor. 3).',
      hyp: IV2_HYP.concat([IV2_BASIS]), brk: IV2_BREAK,
      cone: 'Reverse route: every weight trains, so its cone and kept activations are those of full fine-tuning; the saving is optimizer state, \\(2rn+mr\\) floats per \\(m\\times n\\) matrix instead of \\(2mn\\).'
    },
    {
      id: 'galore1', name: 'GaLore, T = 1', short: 'GaLore T=1', row: 'svd', col: 'step', v: 'per', v2: 'out',
      cite: 'Zhao et al. 2024, with T = 1', url: 'https://arxiv.org/abs/2403.03507', ax: '2403.03507',
      vline: 'Step by step a re-based one-sided LoRA, which is automatic at \\(T=1\\). Read as a \\(\\theta\\)-dependent field it is generically non-involutive.',
      c: '\\(c_t=P_t^{\\top}\\)', a: '\\(a_t=P_t\\)', cnote: '\\(P_t\\) recomputed from \\(G_t\\) at every step',
      upd: '\\[\\begin{aligned}&W\\leftarrow W-\\eta\\,P_t\\,\\mathrm{opt}\\big(P_t^{\\top}G_t\\big)\\\\ &\\text{SGD: }\\Delta W=-\\eta\\,P_tP_t^{\\top}G_t\\end{aligned}\\]',
      read: 'At \\(T=1\\) a period is one step, so the period-wise reading is automatic and says nothing. The content is the other reading: in full batch the anchor is a function of \\(\\theta\\), giving the field \\(D_W=\\mathrm{Hom}\\big(\\mathbb R^n,U_r(\\nabla L(W))\\big)\\) wherever a singular-value gap exists (\\(\\mathrm{Hom}(V_r,\\mathbb R^m)\\) for the right projection). That field is generically non-involutive for every \\(r\\) with \\(1\\le r\\lt\\min(m,n)\\), so it is not foliated by images of forward methods (Prop IV.3, Frobenius). The live probe below measures the bracket. Minibatch \\(T=1\\) is not time-independent.',
      hyp: IV2_HYP.concat([IV2_BASIS]), brk: IV2_BREAK,
      cone: 'Reverse route, as GaLore.'
    },
    {
      id: 'bitfit', name: 'BitFit', row: 'coord', col: 'frozen', v: 'fwd', link: 'bitfit',
      cite: 'Ben Zaken, Ravfogel, Goldberg 2021', url: 'https://arxiv.org/abs/2106.10199', ax: '2106.10199',
      vline: 'A selective forward method; trivially of Def IV.1 form.',
      c: '\\(c=\\iota_b^{\\top}\\) (restrict to biases)', a: '\\(a=\\iota_b\\) (include)', cnote: '\\(\\iota_b\\) = inclusion of the bias coordinates',
      upd: '\\[\\theta\\leftarrow\\theta-\\eta\\,\\iota_b\\,\\mathrm{opt}\\big(\\iota_b^{\\top}\\nabla L\\big)\\]',
      read: 'The forward method \\(\\rho(\\delta)=\\theta_0+\\iota_b(\\delta)\\) is linear in \\(\\delta\\), so by Prop IV.2 it equals the backward method \\(c=\\iota_b^{\\top}\\), \\(a=\\iota_b\\).',
      hyp: IV2_HYP,
      cone: 'Biases are additive boxes: their parameter cotangent is the incoming cotangent, so they keep nothing. They still sit in block 1, so every nonlinear box downstream keeps its residual (Obs IV.4).'
    },
    {
      id: 'fish', name: 'FISH Mask', row: 'coord', col: 'frozen', v: 'fwd',
      cite: 'Sung, Nair, Raffel 2021', url: 'https://arxiv.org/abs/2111.09839', ax: '2111.09839',
      vline: 'A selective forward method on a fixed sparse mask.',
      c: '\\(c=\\iota_S^{\\top}\\)', a: '\\(a=\\iota_S\\)', cnote: '\\(S\\) = top coordinates of the empirical Fisher, fixed before training',
      upd: '\\[\\theta\\leftarrow\\theta-\\eta\\,\\iota_S\\,\\mathrm{opt}\\big(\\iota_S^{\\top}\\nabla L\\big)\\]',
      read: '\\(\\rho(s)=\\theta_0+\\iota_S(s)\\) is linear, so the forward and backward readings coincide (Prop IV.2).',
      hyp: IV2_HYP,
      cone: 'Sparse entries of linear maps are multiplicative: a trained entry of \\(W\\) keeps that box’s input.'
    },
    {
      id: 'ln', name: 'LN tuning', row: 'coord', col: 'frozen', v: 'fwd',
      cite: 'Zhao et al. 2023', url: 'https://arxiv.org/abs/2312.11420', ax: '2312.11420',
      vline: 'A selective forward method on the normalisation parameters.',
      c: '\\(c=\\iota_{\\mathrm{LN}}^{\\top}\\)', a: '\\(a=\\iota_{\\mathrm{LN}}\\)', cnote: 'only the normalisation gains (and biases) train',
      upd: '\\[\\theta\\leftarrow\\theta-\\eta\\,\\iota_{\\mathrm{LN}}\\,\\mathrm{opt}\\big(\\iota_{\\mathrm{LN}}^{\\top}\\nabla L\\big)\\]',
      read: 'A coordinate inclusion, hence linear: forward and backward readings coincide (Prop IV.2).',
      hyp: IV2_HYP,
      cone: 'Norm gains are multiplicative boxes: each keeps its normalised input. Click the norm rows in B to see it.'
    },
    {
      id: 'lisa', name: 'LISA', row: 'coord', col: 'period', v: 'per', link: 'lisa',
      cite: 'Pan et al. 2024', url: 'https://arxiv.org/abs/2403.17919', ax: '2403.17919',
      vline: 'Per period a selective linear method; LISA’s AdamW decay is outside Prop IV.2.',
      c: '\\(c_t=\\iota_{S_k}^{\\top}\\)', a: '\\(a_t=\\iota_{S_k}\\)', cnote: '\\(S_k\\) = embedding, LM head and \\(\\gamma\\) blocks sampled every \\(K\\) steps',
      upd: '\\[\\begin{aligned}&\\theta\\leftarrow\\theta-\\eta\\,\\iota_{S_k}\\,\\mathrm{opt}\\big(\\iota_{S_k}^{\\top}\\nabla L\\big),\\\\ &kK\\le t\\lt(k+1)K\\end{aligned}\\]',
      read: 'On each period the anchor is constant, so this is the coordinate method \\(\\theta_k+\\iota_{S_k}(c)\\): under Prop IV.2’s optimizer conditions (weight decay \\(0\\) included), a rebasing scheme of linear forward methods. Decoupled decay (AdamW with \\(\\lambda>0\\)) falls outside that identity.',
      hyp: IV2_HYP,
      cone: 'The embedding trains at every step, so the cone is the whole network: no truncation. On the cone, the frozen linears of unsampled blocks keep nothing, so it stores fewer activations than full fine-tuning; most of its saving is gradients and optimizer state (Obs IV.4).'
    },
    {
      id: 'bcd', name: 'Block coordinate descent', short: 'Block CD', row: 'coord', col: 'step', v: 'per',
      cite: 'classical; one block per step', url: '', ax: '',
      vline: 'Each step is a step of a coordinate method; automatic at \\(T=1\\).',
      c: '\\(c_t=\\iota_{S_t}^{\\top}\\)', a: '\\(a_t=\\iota_{S_t}\\)', cnote: 'one block \\(S_t\\) per step, cyclic or sampled',
      upd: '\\[\\theta\\leftarrow\\theta-\\eta\\,\\iota_{S_t}\\,\\mathrm{opt}\\big(\\iota_{S_t}^{\\top}\\nabla L\\big)\\]',
      read: 'Each step is a step of \\(\\theta_t+\\iota_{S_t}(c)\\). At \\(T=1\\) the period-wise reading is automatic; its anchor depends on \\(t\\), not on \\(\\theta\\), so Frobenius has nothing to test.',
      hyp: IV2_HYP,
      cone: 'The cone starts at the lowest active block of the step.'
    },
    {
      id: 'lora', name: 'LoRA', row: 'two', col: 'frozen', v: 'fwd', link: 'lora-qv',
      cite: 'Hu et al. 2021', url: 'https://arxiv.org/abs/2106.09685', ax: '2106.09685',
      vline: 'A forward lens by definition. Contrast row: its anchor reads the factor state.',
      c: '\\(c(G)=(GA^{\\top},\\,B^{\\top}G)\\)', a: '\\((S,T)\\mapsto SA+BT\\)', cnote: 'both depend on the factors \\((B,A)\\), not on \\(\\theta\\) alone',
      upd: '\\[\\begin{aligned}&W=W_0+BA,\\quad B_0=0:\\\\ &B\\leftarrow B-\\eta\\,\\mathrm{opt}(GA^{\\top}),\\quad A\\leftarrow A-\\eta\\,\\mathrm{opt}(B^{\\top}G)\\end{aligned}\\]',
      read: 'The lens \\((\\rho,\\mathsf R[\\rho])\\) itself. Under SGD, to first order, \\(\\Delta W=-\\eta\\,(GA^{\\top}A+BB^{\\top}G)\\): a two-sided anchor that reads the factor state \\((B,A)\\). So it sits outside Def IV.1 and outside Prop IV.2; it is listed for contrast.',
      cone: '\\(A\\) keeps its input \\(x\\), \\(B\\) keeps \\(Ax\\), and every nonlinear box downstream keeps its residual: stored activations stay close to full fine-tuning (Obs IV.4).'
    },
    {
      id: 'relora', name: 'ReLoRA', row: 'two', col: 'period', v: 'fwd',
      cite: 'Lialin et al. 2023', url: 'https://arxiv.org/abs/2307.05695', ax: '2307.05695',
      vline: 'LoRA re-based every \\(T\\) steps with a state reset. Contrast row.',
      c: 'as LoRA, within a period', a: 'as LoRA, within a period', cnote: 'every \\(T\\) steps: merge, re-initialise, partially reset the optimizer',
      upd: '\\[\\begin{aligned}&W_{k+1}=W_k+B_kA_k,\\quad (B,A)\\leftarrow(0,\\ \\text{random}),\\\\ &\\text{optimizer state partially reset}\\end{aligned}\\]',
      read: 'A forward method re-based every \\(T\\) steps (Exposé V). Its reset is exactly the ingredient Prop IV.2 isolates: carried-over versus reset optimizer state gives different trajectories (live meter below, “reset state”).',
      cone: 'As LoRA within each period.'
    }
  ];

  /* ======================================================================================
     Numerics shared by the live checks
     ====================================================================================== */
  function randMat(rnd, m, n, s) {
    var M = [];
    for (var i = 0; i < m; i++) { var row = new Float64Array(n); for (var j = 0; j < n; j++) row[j] = rnd.normal() * s; M.push(row); }
    return M;
  }
  function zeros(m, n) { var M = []; for (var i = 0; i < m; i++) M.push(new Float64Array(n)); return M; }
  function copyM(A) { return A.map(function (r) { return Float64Array.from(r); }); }
  function mul(A, B) {
    var m = A.length, k = B.length, n = B[0].length, C = zeros(m, n);
    for (var i = 0; i < m; i++) for (var p = 0; p < k; p++) { var a = A[i][p]; if (a === 0) continue; var Bp = B[p], Ci = C[i]; for (var j = 0; j < n; j++) Ci[j] += a * Bp[j]; }
    return C;
  }
  function mulT(A, B) { /* A B^T */
    var m = A.length, n = B.length, k = A[0].length, C = zeros(m, n);
    for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) { var s = 0; for (var p = 0; p < k; p++) s += A[i][p] * B[j][p]; C[i][j] = s; }
    return C;
  }
  function frob(A) { var s = 0; for (var i = 0; i < A.length; i++) for (var j = 0; j < A[i].length; j++) s += A[i][j] * A[i][j]; return Math.sqrt(s); }
  /* cyclic Jacobi to full double precision; returns eigenpairs sorted by value (descending), each eigenvector
     normalised so that its largest-magnitude entry is positive (the convention of the Python reference) */
  function eigDesc(S) {
    var n = S.length, a = S.map(function (r) { return Float64Array.from(r); }), V = zeros(n, n), i, k, p, q;
    for (i = 0; i < n; i++) V[i][i] = 1;
    var nrm = 0; for (i = 0; i < n; i++) for (k = 0; k < n; k++) nrm += a[i][k] * a[i][k];
    var prev = Infinity;
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0;
      for (p = 0; p < n; p++) for (q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off === 0 || off <= 1e-34 * nrm || off >= prev) break;
      prev = off;
      for (p = 0; p < n; p++) for (q = p + 1; q < n; q++) {
        var apq = a[p][q];
        if (apq === 0) continue;
        var th = (a[q][q] - a[p][p]) / (2 * apq);
        var t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        var c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (k = 0; k < n; k++) { var akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (k = 0; k < n; k++) { var apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (k = 0; k < n; k++) { var vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    var idx = []; for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (x, y) { return a[y][y] - a[x][x]; });
    var lam = idx.map(function (j) { return a[j][j]; }), U = [];
    idx.forEach(function (j) {
      var u = new Float64Array(n), best = 0;
      for (k = 0; k < n; k++) u[k] = V[k][j];
      for (k = 1; k < n; k++) if (Math.abs(u[k]) > Math.abs(u[best])) best = k;
      if (u[best] < 0) for (k = 0; k < n; k++) u[k] = -u[k];
      U.push(u);
    });
    return { lam: lam, U: U };
  }

  /* ---------- Prop IV.2 live check: GaLore vs one-sided LoRA re-based (theory reference: lens/ref_toys.py) ---------- */
  var IV2 = { seed: 11, m: 6, n: 10, N: 24, r: 2, T: 10, K: 4, eta: 0.01, lam: 0.1, clip: 1, b1: 0.9, b2: 0.999, eps: 1e-8 };
  function iv2Run(A, variant) {
    var c = IV2, m = c.m, n = c.n, N = c.N, r = c.r, i, j, a, q;
    var rnd = A.rng(c.seed);
    var W0 = randMat(rnd, m, n, 1 / Math.sqrt(n)), X = randMat(rnd, n, N, 1), D = randMat(rnd, m, n, 0.6 / Math.sqrt(n));
    var Wst = zeros(m, n); for (i = 0; i < m; i++) for (j = 0; j < n; j++) Wst[i][j] = W0[i][j] + D[i][j];
    var Y = mul(Wst, X);
    function grad(W) {
      var R = mul(W, X);
      for (i = 0; i < m; i++) for (q = 0; q < N; q++) R[i][q] -= Y[i][q];
      var G = mulT(R, X);
      for (i = 0; i < m; i++) for (j = 0; j < n; j++) G[i][j] /= N;
      return G;
    }
    function newState() { return { t: 0, m: zeros(r, n), v: zeros(r, n) }; }
    function adam(st, g) {
      st.t += 1;
      var bc1 = 1 - Math.pow(c.b1, st.t), bc2 = 1 - Math.pow(c.b2, st.t), out = zeros(r, n);
      for (var x = 0; x < r; x++) for (var y = 0; y < n; y++) {
        st.m[x][y] = c.b1 * st.m[x][y] + (1 - c.b1) * g[x][y];
        st.v[x][y] = c.b2 * st.v[x][y] + (1 - c.b2) * g[x][y] * g[x][y];
        out[x][y] = (st.m[x][y] / bc1) / (Math.sqrt(st.v[x][y] / bc2) + c.eps);
      }
      return out;
    }
    function PtG(P, G) { var R = zeros(r, n); for (a = 0; a < r; a++) for (j = 0; j < n; j++) { var s = 0; for (i = 0; i < m; i++) s += P[i][a] * G[i][j]; R[a][j] = s; } return R; }
    function PC(P, C) { var R = zeros(m, n); for (i = 0; i < m; i++) for (j = 0; j < n; j++) { var s = 0; for (a = 0; a < r; a++) s += P[i][a] * C[a][j]; R[i][j] = s; } return R; }
    function scaleIfOver(G, lim) { var nr = frob(G); if (nr > lim) { var f = lim / nr; return G.map(function (row) { return row.map(function (v) { return v * f; }); }); } return G; }
    var Wg = copyM(W0), sg = newState(), Wk = copyM(W0), C = zeros(r, n), sl = newState(), gaps = [];
    for (var k = 0; k < c.K; k++) {
      var Gs = grad(Wg), E = eigDesc(mulT(Gs, Gs)), P = zeros(m, r);
      for (i = 0; i < m; i++) for (a = 0; a < r; a++) P[i][a] = E.U[a][i];
      if (variant === 'reset') sl = newState();
      for (var t = 0; t < c.T; t++) {
        /* GaLore side: decoupled decay and clipping act on the full W and the full G */
        var G = grad(Wg);
        if (variant === 'clip') G = scaleIfOver(G, c.clip);
        if (variant === 'adamw') for (i = 0; i < m; i++) for (j = 0; j < n; j++) Wg[i][j] -= c.eta * c.lam * Wg[i][j];
        var U = PC(P, adam(sg, PtG(P, G)));
        for (i = 0; i < m; i++) for (j = 0; j < n; j++) Wg[i][j] -= c.eta * U[i][j];
        /* one-sided LoRA side: W = W_k + P_k C; decay and clipping see C and P_k^T G */
        var BC = PC(P, C), Wl = zeros(m, n);
        for (i = 0; i < m; i++) for (j = 0; j < n; j++) Wl[i][j] = Wk[i][j] + BC[i][j];
        var gC = PtG(P, grad(Wl));
        if (variant === 'clip') gC = scaleIfOver(gC, c.clip);
        if (variant === 'adamw') for (a = 0; a < r; a++) for (j = 0; j < n; j++) C[a][j] -= c.eta * c.lam * C[a][j];
        var Uc = adam(sl, gC);
        for (a = 0; a < r; a++) for (j = 0; j < n; j++) C[a][j] -= c.eta * Uc[a][j];
        BC = PC(P, C);
        var g2 = 0;
        for (i = 0; i < m; i++) for (j = 0; j < n; j++) { var dd = Wg[i][j] - (Wk[i][j] + BC[i][j]); g2 += dd * dd; }
        gaps.push(Math.sqrt(g2));
      }
      BC = PC(P, C);
      for (i = 0; i < m; i++) for (j = 0; j < n; j++) Wk[i][j] += BC[i][j];
      C = zeros(r, n);
    }
    var mv = 0; for (i = 0; i < m; i++) for (j = 0; j < n; j++) mv += (Wg[i][j] - W0[i][j]) * (Wg[i][j] - W0[i][j]);
    return { gaps: gaps, max: Math.max.apply(null, gaps), move: Math.sqrt(mv) };
  }

  /* ---------- Prop IV.3 live probe: bracket of D_W = Hom(R^n, U_r(grad L(W))) ----------
     W is 3 x 4 (m < n), where GaLore itself projects on the left, so this is the field GaLore T = 1 follows. */
  var IV3 = { seed: 5, m: 3, n: 4, N: 7 };
  function iv3Run(A) {
    var m = IV3.m, n = IV3.n, rnd = A.rng(IV3.seed);
    var W = randMat(rnd, m, n, 1), X7 = randMat(rnd, n, IV3.N, 1), Y7 = randMat(rnd, m, IV3.N, 1), x1 = randMat(rnd, n, 1, 1), y1 = randMat(rnd, m, 1, 1);
    var pairs = []; for (var i = 0; i < 8; i++) pairs.push([randMat(rnd, m, n, 1), randMat(rnd, m, n, 1)]);
    function field(X, Y, r) {
      var N = X[0].length, R = mul(W, X), a, b;
      for (a = 0; a < m; a++) for (b = 0; b < N; b++) R[a][b] -= Y[a][b];
      var G = mulT(R, X); for (a = 0; a < m; a++) for (b = 0; b < n; b++) G[a][b] /= N;
      var E = eigDesc(mulT(G, G)), Pi = zeros(m, m);
      for (var s = 0; s < r; s++) for (a = 0; a < m; a++) for (b = 0; b < m; b++) Pi[a][b] += E.U[s][a] * E.U[s][b];
      var XXt = mulT(X, X); for (a = 0; a < n; a++) for (b = 0; b < n; b++) XXt[a][b] /= N;
      /* D Pi[V]: first-order perturbation of the top-r eigenspace of S = G G^T, dG = V X X^T / N */
      function dPi(Vm) {
        var dG = mul(Vm, XXt), dS = mulT(dG, G), GdGt = mulT(G, dG), out = zeros(m, m), ii, jj, p, q;
        for (p = 0; p < m; p++) for (q = 0; q < m; q++) dS[p][q] += GdGt[p][q];
        for (ii = 0; ii < r; ii++) for (jj = r; jj < m; jj++) {
          var ui = E.U[ii], uj = E.U[jj], s2 = 0;
          for (p = 0; p < m; p++) for (q = 0; q < m; q++) s2 += uj[p] * dS[p][q] * ui[q];
          var cc = s2 / (E.lam[ii] - E.lam[jj]);
          for (p = 0; p < m; p++) for (q = 0; q < m; q++) out[p][q] += cc * (uj[p] * ui[q] + ui[p] * uj[q]);
        }
        return out;
      }
      return { Pi: Pi, dPi: dPi };
    }
    function run(X, Y, r) {
      var F = field(X, Y, r), ratios = [], brs = [], nrs = [];
      pairs.forEach(function (pr) {
        var XN = mul(F.Pi, pr[0]), XM = mul(F.Pi, pr[1]);
        var t1 = mul(F.dPi(XN), pr[1]), t2 = mul(F.dPi(XM), pr[0]), br = zeros(m, n), a, b;
        for (a = 0; a < m; a++) for (b = 0; b < n; b++) br[a][b] = t1[a][b] - t2[a][b];
        var PB = mul(F.Pi, br), nor = zeros(m, n);
        for (a = 0; a < m; a++) for (b = 0; b < n; b++) nor[a][b] = br[a][b] - PB[a][b];
        var nb = frob(br), nn = frob(nor);
        brs.push(nb); nrs.push(nn); ratios.push(nb > 0 ? nn / nb : 0);
      });
      return { rmin: Math.min.apply(null, ratios), rmax: Math.max.apply(null, ratios), brmax: Math.max.apply(null, brs), nmax: Math.max.apply(null, nrs) };
    }
    return { r1N7: run(X7, Y7, 1), r2N7: run(X7, Y7, 2), r1N1: run(x1, y1, 1) };
  }

  /* ======================================================================================
     Panel B accounting (Obs IV.4): cone by forward reachability, kept tensors box by box
     ====================================================================================== */
  function emptyTrain(L) { var ls = []; for (var i = 0; i < L; i++) ls.push({}); return { emb: null, pr: false, fn: null, hd: null, layers: ls }; }
  function presetTrain(id, cfg, o) {
    var tr = emptyTrain(cfg.L), L = cfg.L;
    var full = function (l) { ROWS.forEach(function (rw) { if (rw.kind !== 'nl') tr.layers[l][rw.key] = 'W'; }); };
    var l;
    if (id === 'full' || id === 'mezo') { tr.emb = tr.fn = tr.hd = 'W'; for (l = 0; l < L; l++) full(l); }
    else if (id === 'lora-qv') { for (l = 0; l < L; l++) { tr.layers[l].q = 'lora'; tr.layers[l].v = 'lora'; } }
    else if (id === 'lora-all') { for (l = 0; l < L; l++) LIN.forEach(function (k) { tr.layers[l][k] = 'lora'; }); }
    else if (id === 'bitfit') { for (l = 0; l < L; l++) LIN.forEach(function (k) { tr.layers[l][k] = 'bias'; }); }
    else if (id === 'lisa') { tr.emb = tr.hd = 'W'; (o.lisaLayers || []).forEach(function (x) { if (x < L) full(x); }); }
    else if (id === 'topk') { for (l = Math.max(0, L - o.k); l < L; l++) full(l); }
    else if (id === 'prompt') { tr.pr = true; }
    return tr;
  }
  function polyAdd(acc, c, w) { for (var k in c) acc[k] = (acc[k] || 0) + c[k] * (w || 1); }
  function analyse(cfg, tr, o) {
    var d = cfg.d, f = cfg.f, kv = cfg.kv, H = cfg.h, V = cfg.V, L = cfg.L, r = o.r;
    var S = o.s + (tr.pr ? o.p : 0);
    var dim = { d: d, kv: kv, f: f, r: r, h: H, hS: H * S, one: 1 };
    var IO = { q: [d, d], k: [d, kv], v: [d, kv], o: [d, d], gt: [d, f], up: [d, f], dn: [f, d] };
    var P = 2 * V * d + d + L * (2 * d * d + 2 * d * kv + 3 * d * f + 2 * d);
    var Pnew = 0, M = 0, nTrain = 0;
    if (tr.emb === 'W') { M += V * d; nTrain++; }
    if (tr.fn === 'W') { M += d; nTrain++; }
    if (tr.hd === 'W') { M += V * d; nTrain++; }
    if (tr.pr) { Pnew += o.p * d; M += o.p * d; nTrain++; }
    var reach = tr.emb === 'W' || tr.pr;
    var embR = reach, layers = [], coneStart = null, blocksOn = 0;
    function size(key) { var t = TL[key], s = 0; for (var k in t.c) s += t.c[k] * dim[k]; return s * t.B; }
    for (var l = 0; l < L; l++) {
      var t = tr.layers[l], R = {}, need = {}, T = function (k) { return !!t[k]; };
      ['n1', 'n2'].forEach(function (k) { if (t[k] === 'W') { M += d; nTrain++; } });
      LIN.forEach(function (k) {
        var io = IO[k];
        if (t[k] === 'W') { M += io[0] * io[1]; nTrain++; }
        else if (t[k] === 'lora') { Pnew += r * (io[0] + io[1]); M += o.fa ? r * io[1] : r * (io[0] + io[1]); nTrain++; }
        else if (t[k] === 'bias') { Pnew += io[1]; M += io[1]; nTrain++; }
      });
      if (o.mezo) { layers.push({ R: {}, need: {}, nlB: 0, inB: 0, sig: '' }); continue; }
      R.n1 = reach || T('n1');
      R.q = R.n1 || T('q'); R.k = R.n1 || T('k'); R.v = R.n1 || T('v');
      R.at = R.q || R.k || R.v;
      R.o = R.at || T('o');
      var hR = reach || R.o;
      R.n2 = hR || T('n2');
      R.gt = R.n2 || T('gt'); R.up = R.n2 || T('up');
      R.ac = R.gt || R.up;
      R.dn = R.ac || T('dn');
      var add = function (key, cat) { if (!need[key]) need[key] = cat; };
      /* (ii) nonlinear boxes on the cone keep a VJP residual */
      if (R.n1) { add('x', 'nl'); add('r1', 'nl'); }
      if (R.at) { add('q', 'nl'); add('k', 'nl'); add('v', 'nl'); if (o.eager) add('P', 'nl'); else { add('o', 'nl'); add('lse', 'nl'); } }
      if (R.n2) { add('h', 'nl'); add('r2', 'nl'); }
      if (R.ac) { add('g', 'nl'); add('u', 'nl'); }
      /* (i) trainable linear / multiplicative boxes keep their input; additive ones (biases) keep nothing */
      LIN.forEach(function (k) {
        if (t[k] === 'W') add(INP[k], 'in');
        else if (t[k] === 'lora') { if (!o.fa) add(INP[k], 'in'); add('z_' + k, 'in'); }
      });
      var nlB = 0, inB = 0, sig = [];
      TORDER.forEach(function (key) { if (need[key]) { if (need[key] === 'nl') nlB += size(key); else inB += size(key); sig.push(key + need[key]); } });
      if (coneStart === null) for (var i = 0; i < ROWS.length; i++) if (R[ROWS[i].key]) { coneStart = { l: l, key: ROWS[i].key }; break; }
      var anyR = false; for (var kk in R) if (R[kk]) anyR = true;
      if (anyR) blocksOn++;
      layers.push({ R: R, need: need, nlB: nlB, inB: inB, sig: sig.join(',') });
      reach = hR || R.dn;
    }
    var g = { embR: embR && !o.mezo, fnR: false, hdR: false, lsR: false, need: {}, nlB: 0, inB: 0, E: 0, N: 0, Hh: 0, Ls: 0 };
    if (!o.mezo) {
      g.fnR = reach || tr.fn === 'W'; g.hdR = g.fnR || tr.hd === 'W'; g.lsR = g.hdR;
      if (tr.emb === 'W') { g.need.ids = 'in'; g.E = 8; }
      if (g.fnR) { g.need.xL = 'nl'; g.need.rf = 'nl'; g.N = 2 * d + 4; }
      if (tr.hd === 'W') { g.need.xLn = 'in'; g.Hh = 2 * d; }
      if (g.lsR) { g.need.logp = 'nl'; g.Ls = 4 * V; }
      g.nlB = g.N + g.Ls; g.inB = g.E + g.Hh;
    }
    var tokL = o.b * S, tok = o.b * o.s, actNl = 0, actIn = 0;
    layers.forEach(function (ly) { actNl += tokL * ly.nlB; actIn += tokL * ly.inB; });
    actNl += tok * g.nlB; actIn += tok * g.inB;
    var res = {
      P: P, Pnew: Pnew, M: M, nTrain: nTrain, S: S, tokL: tokL, tok: tok, layers: layers, g: g, coneStart: coneStart, blocksOn: blocksOn,
      any: g.lsR, mezo: !!o.mezo, dim: dim
    };
    res.weights = 2 * (P + Pnew);
    res.grads = o.mezo ? 0 : 2 * M;
    res.opt = o.mezo ? 0 : (8 + (o.master ? 4 : 0)) * M;
    res.actNl = o.mezo ? 0 : actNl; res.actIn = o.mezo ? 0 : actIn; res.act = res.actNl + res.actIn;
    res.total = res.weights + res.grads + res.opt + res.act;
    return res;
  }
  /* symbolic size of a kept-tensor set, split by element width */
  function polyOf(need, dimKeys) {
    var p2 = {}, p4 = {};
    Object.keys(need).forEach(function (key) { var t = TL[key]; polyAdd(t.B === 4 ? p4 : p2, t.c); });
    return { p2: p2, p4: p4 };
  }
  var PNAME = { d: 'd', kv: 'd<sub>kv</sub>', f: 'f', r: 'r', hS: 'h·S', h: 'h', one: '' };
  function polyStr(p, keys) {
    var out = [];
    keys.forEach(function (k) {
      var v = p[k] || 0; if (!v) return;
      out.push(k === 'one' ? String(v) : (v === 1 ? '' : v) + PNAME[k]);
    });
    return out.join(' + ');
  }

  /* ======================================================================================
     Formatting
     ====================================================================================== */
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sup(n) { return String(n).split('').map(function (ch) { return SUP[ch] || ch; }).join(''); }
  function sci(x, dgt) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '∞';
    var p = x.toExponential(dgt == null ? 1 : dgt).split('e'), e = parseInt(p[1], 10);
    if (e >= -2 && e <= 2) return String(+x.toPrecision((dgt == null ? 1 : dgt) + 1));
    return p[0] + ' × 10' + sup(e);
  }
  /* Plex Mono's superscript glyphs are about 5px tall at 12px; readouts set their exponent as real digits, raised */
  var SUPINV = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };
  function expParts(str) {
    var m = /^(.*?)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)(.*)$/.exec(String(str));
    return m ? [m[1], m[2].split('').map(function (c) { return SUPINV[c]; }).join(''), m[3]] : null;
  }
  function escH(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function supHTML(str) { var p = expParts(str); return p ? escH(p[0]) + '<sup>' + p[1] + '</sup>' + escH(p[2]) : escH(str); }
  function bstr(b) {
    var a = Math.abs(b), u = [['GiB', 1073741824], ['MiB', 1048576], ['KiB', 1024]];
    for (var i = 0; i < u.length; i++) if (a >= u[i][1]) { var v = b / u[i][1]; return (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + ' ' + u[i][0]; }
    return Math.round(b) + ' B';
  }
  function gstr(b) { if (b === 0) return '0'; var v = b / 1073741824; return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3); }
  /* GiB when large, otherwise the binary unit that keeps three significant digits */
  function mstr(b) { return b === 0 ? '0' : b >= 0.01 * 1073741824 ? gstr(b) + ' GiB' : bstr(b); }
  function barStr(b) { return b === 0 ? '0' : b >= 0.01 * 1073741824 ? gstr(b) : bstr(b); }
  function intStr(n) { return Math.round(n).toLocaleString('en-US'); }
  function pctStr(x) { if (!isFinite(x)) return '—'; var p = x * 100; return (p >= 10 ? p.toFixed(0) : p >= 1 ? p.toFixed(1) : p >= 0.01 ? p.toFixed(2) : p.toFixed(4)) + '%'; }
  function ranges(list) {
    var out = [], i = 0;
    while (i < list.length) { var j = i; while (j + 1 < list.length && list[j + 1] === list[j] + 1) j++; out.push(i === j ? String(list[i]) : list[i] + '–' + list[j]); i = j + 1; }
    return out.join(', ');
  }

  /* ======================================================================================
     Scoped CSS
     ====================================================================================== */
  function injectCSS() {
    if (document.getElementById('css-lens')) return;
    var F = '[data-figure="lens"] ';
    var css = [
      F + '.ln-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      /* sub/superscripts inside running text never drop below the 12px reading floor */
      F + 'sub,' + F + 'sup{font-size:max(.8em,12px);line-height:0}',
      F + 'code{font-size:max(.88em,12px)}',
      F + '.ln-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.ln-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.ln-title i{color:var(--tide)}',
      F + '.ln-kicker{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.ln-instr{margin:.45rem 0 .85rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:72ch}',
      F + '.ln-instr b{color:var(--ink);font-weight:600}',
      F + '.ln-formula{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem 1.5rem;padding:.5rem .75rem;margin:0 0 1rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.9rem;line-height:1.5;color:var(--ink);overflow-x:auto}',
      F + '.ln-formula > span{display:inline-flex;flex-wrap:wrap;align-items:center;gap:.15rem .5rem;min-width:0}',
      F + '.ln-formula mjx-container{white-space:nowrap}',
      F + '.ln-formula mjx-container svg{max-width:none}',
      /* UI labels: Plex Sans 500, uppercase, letter-spacing kept at .06em */
      F + '.ln-tag{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.ln-panel{border-top:1px solid var(--ink-3);padding-top:.75rem;margin-top:.4rem}',
      F + '.ln-panel + .ln-panel{margin-top:1.9rem}',
      F + '.ln-phead{display:flex;flex-wrap:wrap;align-items:baseline;gap:.2rem .7rem;margin-bottom:.35rem}',
      F + '.ln-badge{font-family:var(--f-ui);font-weight:600;font-size:.78rem;letter-spacing:.04em;color:var(--paper);background:var(--ink-2);border-radius:3px;padding:.04rem .4rem;align-self:center}',
      F + '.ln-ptitle{font-family:var(--f-display);font-size:clamp(1.12rem,.95rem + .8cqi,1.38rem);font-weight:var(--w-head);color:var(--ink);line-height:1.2}',
      F + '.ln-ptitle i{font-family:var(--f-body);font-weight:var(--w-body);color:var(--tide)}',
      F + '.ln-pthm{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.ln-pinstr{margin:.1rem 0 .8rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:74ch}',
      F + '.ln-pinstr b{color:var(--ink);font-weight:600}',
      /* ---- Panel A ---- */
      F + '.ln-abody{display:grid;grid-template-columns:minmax(0,1fr);grid-template-areas:"tab" "card" "chk";gap:1rem 1.4rem;align-items:start}',
      F + '.ln-tabwrap{grid-area:tab;min-width:0}',
      F + '.ln-cardwrap{grid-area:card;min-width:0}',
      F + '.ln-checks{grid-area:chk;display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem;min-width:0}',
      F + 'table.ln-grid{width:100%;border-collapse:collapse;table-layout:fixed;font-size:.84rem;line-height:1.3;margin:0}',
      F + 'table.ln-grid th,' + F + 'table.ln-grid td{position:static;background:none;padding:.42rem .3rem;border-bottom:1px solid var(--rule);vertical-align:top;text-transform:none;letter-spacing:0;font-weight:400}',
      F + 'table.ln-grid tr:last-child th,' + F + 'table.ln-grid tr:last-child td{border-bottom:0}',
      F + 'table.ln-grid thead th{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-2);vertical-align:bottom;border-bottom:1px solid var(--ink-3)}',
      F + 'table.ln-grid thead th .s{display:block;font-family:var(--f-body);font-weight:var(--w-body);font-style:italic;text-transform:none;letter-spacing:0;color:var(--ink-2);font-size:.84rem;margin-top:.1rem}',
      F + 'table.ln-grid thead th.corner{text-transform:none;letter-spacing:0;line-height:1.35}',
      F + 'table.ln-grid tbody th{font-family:var(--f-body);font-size:.88rem;color:var(--ink);line-height:1.2;text-align:left}',
      F + 'table.ln-grid tbody th .s{display:block;font-family:var(--f-body);font-style:italic;font-size:.8rem;color:var(--ink-2);margin-top:.15rem}',
      F + 'table.ln-grid tbody tr.contrast th,' + F + 'table.ln-grid tbody tr.contrast td{border-top:1px dashed var(--ink-3)}',
      F + 'table.ln-grid td.cell{border-left:1px solid var(--rule)}',
      F + 'table.ln-grid col.c0{width:27%}',
      F + '.ln-cnote{display:block;font-family:var(--f-body);font-style:italic;font-size:.78rem;color:var(--ink-2);margin:.15rem 0 0 .15rem}',
      F + '.ln-m{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:0;padding:.22rem .55rem;border-radius:999px;border:1px solid currentColor;cursor:pointer;line-height:1.2;margin:.12rem .18rem .12rem 0;text-align:left;max-width:100%;white-space:normal}',
      F + '.ln-m.fwd{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.ln-m.per{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + '.ln-m.out{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.ln-m[aria-pressed="true"]{color:var(--paper)}',
      F + '.ln-m.fwd[aria-pressed="true"]{background:var(--moss);border-color:var(--moss)}',
      F + '.ln-m.per[aria-pressed="true"]{background:var(--ochre);border-color:var(--ochre)}',
      F + '.ln-m.out[aria-pressed="true"]{background:var(--seal);border-color:var(--seal)}',
      F + '.ln-check .seg button{padding:.36rem .58rem;letter-spacing:.01em}',
      F + '.ln-m .dot{flex:none;width:.5rem;height:.5rem;border-radius:50%;background:var(--seal);box-shadow:0 0 0 1.5px var(--paper)}',
      F + '.ln-m:focus-visible{outline:2px solid var(--ochre);outline-offset:2px}',
      F + '.ln-legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;margin:.55rem 0 0;font-family:var(--f-ui);font-weight:500;font-size:.78rem;line-height:1.4;color:var(--ink-2)}',
      F + '.ln-li{display:inline-flex;align-items:center;gap:.4rem}',
      F + '.ln-sw{flex:none;display:inline-block;width:.8rem;height:.8rem;border-radius:999px;border:1px solid currentColor}',
      F + '.ln-sw.fwd{color:var(--moss);background:var(--moss-soft)}',
      F + '.ln-sw.per{color:var(--ochre);background:var(--ochre-soft)}',
      F + '.ln-sw.out{color:var(--seal);background:var(--seal-soft)}',
      F + '.ln-sw.dot{width:.5rem;height:.5rem;background:var(--seal);border:0}',
      F + '.ln-card{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.85rem 1rem .95rem;display:grid;grid-template-columns:minmax(0,1fr);gap:.55rem;min-width:0;font-family:var(--f-body);font-size:.9rem;line-height:1.5;color:var(--ink)}',
      F + '.ln-card .top{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.1rem .8rem}',
      F + '.ln-card h4{font-family:var(--f-display);font-size:1.5rem;font-weight:var(--w-head);margin:0;line-height:1.1;color:var(--ink)}',
      F + '.ln-cite{font-family:var(--f-ui);font-size:.78rem;color:var(--ink-2);letter-spacing:0}',
      F + '.ln-cite a{color:var(--ink-2)}',
      F + '.ln-where{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.ln-vchip{display:inline-flex;align-items:center;gap:.35rem;justify-self:start;font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;padding:.14rem .6rem;border-radius:999px;border:1px solid currentColor;line-height:1.35}',
      F + '.ln-vchip.fwd{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.ln-vchip.per{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + '.ln-vchip.out{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.ln-vline{margin:0;font-size:.95rem;line-height:1.45;color:var(--ink)}',
      F + '.ln-dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.25rem .85rem;margin:0;font-size:.88rem}',
      F + '.ln-dl dt{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);padding-top:.22em}',
      F + '.ln-dl dd{margin:0;min-width:0;color:var(--ink)}',
      F + '.ln-dl dd .n{display:block;font-size:.84rem;color:var(--ink-2)}',
      F + '.ln-upd{overflow-x:auto;overflow-y:hidden;padding:.1rem 0;font-size:.92rem}',
      F + '.ln-upd mjx-container[display="true"]{margin:.25em 0 !important}',
      F + '.ln-upd mjx-container svg,' + F + '.ln-foot mjx-container svg,' + F + '.ln-card mjx-container svg{max-width:none}',
      F + '.ln-txt{margin:0;font-size:.88rem;line-height:1.52;color:var(--ink-2)}',
      F + '.ln-sub{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:.15rem 0 -.2rem}',
      F + '.ln-hyp{margin:0;padding-left:1.1rem;font-size:.86rem;line-height:1.45;color:var(--ink-2)}',
      F + '.ln-hyp li{margin:.12rem 0}',
      F + '.ln-brk{margin:0;font-size:.86rem;line-height:1.45;color:var(--seal-ink)}',
      F + '.ln-cone{margin:0;font-size:.86rem;line-height:1.5;color:var(--ink-2);border-left:2px solid var(--tide);padding-left:.6rem}',
      F + '.ln-cone b{font-family:var(--f-ui);font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--tide);font-weight:500;margin-right:.4rem}',
      F + '.ln-go{justify-self:start;font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;padding:.3rem .65rem;border-radius:4px;border:1px solid var(--tide);background:transparent;color:var(--tide);cursor:pointer}',
      F + '.ln-go:hover{background:var(--tide);color:var(--paper)}',
      F + '.ln-check{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.75rem .9rem .8rem;display:grid;grid-template-columns:minmax(0,1fr);gap:.45rem;min-width:0;font-family:var(--f-body)}',
      F + '.ln-check h5{margin:0;font-family:var(--f-display);font-size:.98rem;font-weight:var(--w-head);color:var(--ink);line-height:1.3}',
      F + '.ln-check h5 .ln-tag{margin-left:.4rem}',
      F + '.ln-meter{display:flex;flex-wrap:wrap;align-items:baseline;gap:.2rem .8rem}',
      F + '.ln-big{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-weight:500;font-size:clamp(1.55rem,1.15rem + 1.6cqi,2.1rem);line-height:1;letter-spacing:-.02em;color:var(--ink);white-space:nowrap}',
      F + '.ln-big.z{color:var(--moss-ink)}',
      F + '.ln-big.b{color:var(--seal-ink)}',
      F + '.ln-blab{font-size:.88rem;color:var(--ink-2)}',
      F + '.ln-chip{display:inline-flex;align-items:center;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;padding:.1rem .5rem;border-radius:999px;border:1px solid currentColor;white-space:nowrap;line-height:1.3}',
      F + '.ln-chip.z{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.ln-chip.b{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.ln-spark svg{display:block;width:100%;height:auto;overflow:visible}',
      F + '.ln-why{margin:0;font-size:.86rem;line-height:1.45;color:var(--ink-2)}',
      /* small print under a check: prose, so Source Serif rather than mono */
      F + '.ln-mini{font-family:var(--f-body);font-size:.8rem;line-height:1.5;color:var(--ink-2)}',
      F + 'table.ln-pt{width:100%;border-collapse:collapse;font-size:.8rem;line-height:1.35;margin:0}',
      F + 'table.ln-pt th,' + F + 'table.ln-pt td{position:static;background:none;padding:.3rem .25rem;border-bottom:1px solid var(--rule);vertical-align:baseline;text-transform:none;letter-spacing:0;font-weight:400}',
      F + 'table.ln-pt tr:last-child td,' + F + 'table.ln-pt tr:last-child th{border-bottom:0}',
      F + 'table.ln-pt thead th{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-2);text-align:right;white-space:normal;vertical-align:bottom}',
      F + 'table.ln-pt thead th:first-child{text-align:left}',
      F + 'table.ln-pt tbody th{font-family:var(--f-mono);font-size:.78rem;color:var(--ink);text-align:left;white-space:nowrap}',
      F + 'table.ln-pt td{font-family:var(--f-mono);font-size:.78rem;font-variant-numeric:tabular-nums;text-align:right;color:var(--ink);white-space:nowrap}',
      F + 'table.ln-pt td:last-child{font-family:var(--f-ui);font-weight:500;font-size:.8rem}',
      F + 'table.ln-pt td.w{white-space:nowrap}',
      F + 'table.ln-pt td.z{color:var(--moss-ink)}',
      F + 'table.ln-pt td.b{color:var(--seal-ink)}',
      /* ---- Panel B ---- */
      F + '.ln-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:.75rem 1.4rem;margin-bottom:.9rem}',
      F + '.ln-group{min-width:0;border-top:1px solid var(--rule);padding-top:.45rem;display:grid;grid-template-columns:minmax(0,1fr);gap:.45rem;align-content:start}',
      F + '.ln-row{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .6rem}',
      F + '.ln-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.ln-sl-top{display:flex;justify-content:space-between;align-items:baseline;gap:0 .5rem;min-width:0}',
      F + '.ln-sl label{font-family:var(--f-body);font-size:var(--fs-sm);color:var(--ink);white-space:nowrap}',
      F + '.ln-sl label .nm{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.35rem}',
      F + '.ln-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.ln-sl input[type=range]{margin:0;height:1.25rem}',
      F + '.ln-hint{font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
      F + '.ln-hint.num{font-family:var(--f-mono);font-size:.75rem;font-variant-numeric:tabular-nums}',
      F + '.ln-tog{font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;padding:.3rem .68rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap}',
      F + '.ln-tog[aria-pressed="true"]{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
      F + '.ln-tog:hover:not([aria-pressed="true"]){border-color:var(--ink-3);color:var(--ink)}',
      F + '.ln-btn{font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;padding:.3rem .62rem;border-radius:4px;border:1px solid var(--rule);background:var(--paper);color:var(--ink);cursor:pointer;white-space:nowrap}',
      F + '.ln-btn:hover{border-color:var(--tide);color:var(--tide)}',
      F + '.ln-sel{font-family:var(--f-ui);font-size:.84rem;padding:.3rem .45rem;width:100%;max-width:100%}',
      F + '.ln-custom{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ochre-ink);border:1px solid currentColor;border-radius:999px;padding:.08rem .45rem;background:var(--ochre-soft)}',
      F + '.ln-strip{position:relative;min-width:0;margin:.2rem 0 .3rem}',
      F + '.ln-strip svg{display:block;max-width:100%;touch-action:manipulation;cursor:pointer;border-radius:4px}',
      F + '.ln-strip svg:focus{outline:none}',
      F + '.ln-strip svg:focus-visible{outline:2px solid var(--ochre);outline-offset:3px}',
      /* row names are labels (Plex Sans); column numbers and byte counts are tick values and readouts (mono) */
      F + '.ln-strip svg text{font-family:var(--f-ui);font-weight:500;fill:var(--ink-2)}',
      F + '.ln-strip .lab{font-size:12px;fill:var(--ink-2)}',
      F + '.ln-strip .lab.col{font-family:var(--f-mono);font-weight:400;font-size:11px;fill:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.ln-strip .cone{fill:var(--tide-soft)}',
      F + '.ln-strip .lin{fill:var(--paper-2);stroke:var(--ink-3);stroke-opacity:.55;stroke-width:1}',
      F + '.ln-strip .lin.tw{fill:var(--ochre);stroke:var(--ochre);stroke-opacity:1}',
      F + '.ln-strip .keep{fill:none;stroke:var(--tide);stroke-width:1.6}',
      F + '.ln-strip .adp{fill:var(--ochre)}',
      F + '.ln-strip .nl{fill:none;stroke:var(--ink-3);stroke-opacity:.45;stroke-width:1}',
      F + '.ln-strip .nl.on{fill:var(--tide);stroke:var(--tide);stroke-opacity:1}',
      F + '.ln-strip .nl.gain{stroke:var(--ochre);stroke-opacity:1;stroke-width:1.8}',
      F + '.ln-strip .pr{fill:none;stroke:var(--ink-3);stroke-opacity:.5;stroke-width:1}',
      F + '.ln-strip .pr.tw{fill:var(--ochre);stroke:var(--ochre);stroke-opacity:1}',
      F + '.ln-strip .bnl{fill:var(--tide)}',
      F + '.ln-strip .bin{fill:var(--tide);fill-opacity:.38}',
      F + '.ln-strip .base{stroke:var(--ink-3);stroke-width:1}',
      F + '.ln-strip .cur{fill:none;stroke:var(--ochre);stroke-width:1.6;stroke-dasharray:2 2}',
      F + '.ln-strip svg:not(:focus) .cur{display:none}',
      F + '.ln-strip .note{font-size:12px;fill:var(--ink-2)}',
      F + '.ln-strip .val{font-size:12px;font-weight:400;fill:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.ln-slegend{display:flex;flex-wrap:wrap;gap:.25rem 1rem;margin:.15rem 0 .9rem;font-family:var(--f-ui);font-weight:500;font-size:.78rem;line-height:1.4;color:var(--ink-2)}',
      F + '.ln-slegend .ln-li svg{flex:none;overflow:visible}',
      F + '.ln-res{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.5rem;align-items:start}',
      F + '.ln-bars{min-width:0;display:grid;grid-template-columns:minmax(0,1fr);gap:.5rem}',
      F + '.ln-bh{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem .8rem}',
      F + '.ln-bh .est{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ochre-ink)}',
      F + '.ln-br{display:grid;grid-template-columns:7.4rem minmax(0,1fr) 4.8rem;align-items:center;gap:.6rem}',
      F + '.ln-br .k{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink);line-height:1.25}',
      F + '.ln-br .k small{display:block;font-weight:400;text-transform:none;letter-spacing:0;color:var(--ink-2);font-size:.75rem}',
      F + '.ln-br .v{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);text-align:right;white-space:nowrap}',
      F + '.ln-track{position:relative;height:1.05rem;background:var(--paper);border:1px solid var(--rule);border-radius:3px;overflow:hidden}',
      F + '.ln-track.tot{height:1.5rem}',
      F + '.ln-fill{position:absolute;top:0;bottom:0;left:0;transition:width .3s ease}',
      F + '.ln-ghost{position:absolute;top:1px;bottom:1px;left:0;border:1px dashed var(--ink-3);border-left:0;border-radius:0 2px 2px 0;transition:width .3s ease;pointer-events:none}',
      F + '.ln-ref{position:absolute;top:-2px;bottom:-2px;width:0;border-left:1.5px solid var(--seal);pointer-events:none}',
      F + '.c-w{background:var(--ink-3)}',
      F + '.c-g{background:var(--ochre)}',
      F + '.c-o{background:var(--ochre);opacity:.5}',
      F + '.c-anl{background:var(--tide)}',
      F + '.c-ain{background:var(--tide);opacity:.42}',
      F + '.ln-blegend{display:flex;flex-wrap:wrap;gap:.25rem 1rem;font-family:var(--f-ui);font-weight:500;font-size:.78rem;color:var(--ink-2)}',
      F + '.ln-blegend i{display:inline-block;width:.75rem;height:.6rem;border-radius:2px;margin-right:.35rem;vertical-align:-.05rem}',
      F + '.ln-blegend i.gh{border:1px dashed var(--ink-3);background:none}',
      F + '.ln-blegend i.rl{width:0;height:.8rem;border-left:1.5px solid var(--seal);border-radius:0}',
      F + '.ln-readout{display:grid;grid-template-columns:minmax(0,1fr);gap:.45rem;align-content:start;min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.8rem .9rem .85rem}',
      F + '.ln-rl{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.ln-rsub{font-family:var(--f-ui);font-size:.82rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + 'table.ln-kv{width:100%;border-collapse:collapse;font-size:.8rem;line-height:1.35;margin:.1rem 0 0}',
      F + 'table.ln-kv th,' + F + 'table.ln-kv td{position:static;background:none;padding:.3rem .2rem;border-bottom:1px solid var(--rule);vertical-align:baseline;text-transform:none;letter-spacing:0;font-weight:400}',
      F + 'table.ln-kv tr:last-child th,' + F + 'table.ln-kv tr:last-child td{border-bottom:0}',
      F + 'table.ln-kv th{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2);white-space:nowrap;text-align:left;padding-right:.6rem}',
      F + 'table.ln-kv td{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);text-align:left}',
      F + 'table.ln-kv td small{font-family:var(--f-ui);color:var(--ink-2);font-size:.78rem}',
      F + 'table.ln-kv td.w{font-family:var(--f-ui);font-size:.82rem}',
      F + '.ln-note{margin:.1rem 0 0;font-family:var(--f-body);font-size:.88rem;line-height:1.5;color:var(--ink-2)}',
      F + '.ln-foot{margin-top:1.1rem;padding-top:.6rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.84rem;line-height:1.55;color:var(--ink-2);display:grid;grid-template-columns:minmax(0,1fr);gap:.45rem}',
      F + '.ln-foot .fm{overflow-x:auto;overflow-y:hidden}',
      F + '.ln-fmrow{display:flex;flex-wrap:wrap;justify-content:center;gap:.2rem 1.6rem;font-size:.9rem;color:var(--ink);padding-top:.2rem}',
      F + '.ln-foot .fm mjx-container[display="true"]{margin:.2em 0 !important}',
      F + '.ln-foot .eq{font-family:var(--f-mono);font-size:.75rem;line-height:1.65;color:var(--ink-2);font-variant-numeric:tabular-nums;overflow-wrap:anywhere}',
      F + '.ln-foot .eq b{font-weight:500;color:var(--ink)}',
      F + '.ln-foot ul{margin:0;padding-left:1.1rem}',
      F + '.ln-foot li{margin:.1rem 0}',
      F + '.ln-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
      /* segmented controls that wrap: a tidy grid of equal cells instead of a ragged pill */
      F + '.seg.ln-segg{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--rule);border-radius:var(--radius);overflow:hidden}',
      F + '.seg.ln-segg.ln-seg4{grid-template-columns:repeat(4,minmax(0,1fr))}',
      F + '.seg.ln-segg button{border-right:0;background:var(--paper);text-align:center;white-space:nowrap;padding-left:.3rem;padding-right:.3rem;min-width:0}',
      F + '.seg.ln-segg button[aria-pressed="true"]{background:var(--tide);color:var(--paper)}',
      F + '.seg.ln-segg button:hover:not([aria-pressed="true"]){background:var(--tide-soft);color:var(--ink)}',
      '@container (min-width: 471px){',
      F + '.ln-m.nw{white-space:nowrap}',
      F + '.ln-check .seg.ln-segg{grid-template-columns:repeat(4,minmax(0,1fr))}',
      '}',
      '@container (min-width: 640px){',
      F + '.ln-controls{grid-template-columns:repeat(2,minmax(0,1fr))}',
      F + '.ln-checks{grid-template-columns:repeat(2,minmax(0,1fr))}',
      F + '.ln-check .seg.ln-segg{grid-template-columns:repeat(2,minmax(0,1fr))}',
      '}',
      '@container (min-width: 820px){',
      F + '.ln-check .seg.ln-segg{grid-template-columns:repeat(4,minmax(0,1fr))}',
      F + '.ln-abody{grid-template-columns:minmax(0,1.12fr) minmax(0,1fr);grid-template-areas:"tab card" "chk card"}',
      F + '.ln-checks{grid-template-columns:minmax(0,1fr)}',
      F + '.ln-controls{grid-template-columns:minmax(0,1.35fr) minmax(0,1fr) minmax(0,.95fr)}',
      F + '.ln-res{grid-template-columns:minmax(0,1.45fr) minmax(0,1fr)}',
      '}',
      '@container (max-width: 470px){',
      F + '.ln-br{grid-template-columns:6.3rem minmax(0,1fr) 4.4rem;gap:.45rem}',
      F + 'table.ln-grid{font-size:.8rem}',
      F + 'table.ln-grid th,' + F + 'table.ln-grid td{padding:.36rem .2rem}',
      F + '.ln-m{font-size:.75rem;padding:.2rem .45rem}',
      F + 'table.ln-grid col.c0{width:24%}',
      F + '.ln-upd{font-size:.8rem}',
      F + 'table.ln-pt td{font-size:.75rem}',
      F + 'table.ln-pt td{white-space:normal}',
      F + 'table.ln-pt td:nth-child(2),' + F + 'table.ln-pt td.w{white-space:nowrap}',
      F + 'table.ln-pt th,' + F + 'table.ln-pt td{padding:.3rem .15rem}',
      F + 'table.ln-pt td:nth-child(2){padding-left:.6rem}',
      '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-lens';
    s.textContent = css;
    document.head.appendChild(s);
  }

  /* ======================================================================================
     The figure
     ====================================================================================== */
  Atlas.register('lens', function (el, A) {
    injectCSS();
    var h = A.h, SVGNS = 'http://www.w3.org/2000/svg';
    /* MathJax calls are chained so two typesets never run at once */
    var tsq = Promise.resolve();
    function ts(node) { tsq = tsq.then(function () { return A.typeset(node); }); return tsq; }
    function sv(tag, attrs, parent) {
      var e = document.createElementNS(SVGNS, tag);
      if (attrs) for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }

    var stage = h('div', { class: 'stage ln-stage', role: 'group', 'aria-label': 'The backpropagation lens. Panel A classifies cotangent-side fine-tuning methods by projector and schedule, with cards and two live numerical checks (Prop IV.2 and Prop IV.3). Panel B shades the backward cone of a decoder for a chosen set of trainable boxes and estimates training memory.' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'ln-head' }, [
      h('h3', { class: 'ln-title', html: 'Tangent and cotangent: <i>the lens</i>' }),
      h('span', { class: 'ln-kicker', text: 'Def IV.1 · Prop IV.2 · Prop IV.3 · Obs IV.4' })
    ]));
    stage.appendChild(h('p', { class: 'ln-instr', html: 'Backpropagation is a lens \\((\\rho,\\mathsf R[\\rho])\\). Forward methods change what the model can express; backward methods change how the gradient moves it. <b>Hover a method in A</b> for its update rule and verdict; <b>pick a preset or click boxes in B</b> to see the backward cone and the memory it costs.' }));
    var formula = h('div', { class: 'ln-formula', 'aria-label': 'Key formulas' }, [
      h('span', null, [h('span', { class: 'ln-tag', text: 'Lens' }), h('span', { html: '\\(\\mathsf R[f](x,y)=J_f(x)^{\\top}y\\)' })]),
      h('span', null, [h('span', { class: 'ln-tag', text: 'Backward method · Def IV.1' }), h('span', { html: '\\(\\theta\\leftarrow\\theta-\\eta\\,a_t\\big(\\mathrm{opt}(c_t(\\nabla L))\\big)\\)' })]),
      h('span', null, [h('span', { class: 'ln-tag', text: 'Collapse · Prop IV.2 · opt reads only its gradients' }), h('span', { html: '\\(c=\\mathcal L^{*},\\ a=\\mathcal L\\ \\Longleftrightarrow\\ \\text{train }C\\text{ in }W_k+\\mathcal LC\\)' })])
    ]);
    stage.appendChild(formula);
    var live = h('div', { class: 'ln-live', 'aria-live': 'polite' });
    stage.appendChild(live);

    /* ==================================================================================
       PANEL A
       ================================================================================== */
    var pA = h('section', { class: 'ln-panel', 'aria-label': 'Panel A: projector by schedule' });
    stage.appendChild(pA);
    pA.appendChild(h('div', { class: 'ln-phead' }, [
      h('span', { class: 'ln-badge', text: 'A' }),
      h('span', { class: 'ln-ptitle', html: 'Projector × schedule: <i>the cotangent family</i>' }),
      h('span', { class: 'ln-pthm', text: 'Def IV.1 · Prop IV.2 · Prop IV.3' })
    ]));
    pA.appendChild(h('p', { class: 'ln-pinstr', html: 'Rows say how the compression \\(c_t\\) is chosen, columns how often the anchor \\(a_t\\) moves. <b>Hover, tab or tap a method</b>; the colour is its verdict.' }));
    var abody = h('div', { class: 'ln-abody' });
    pA.appendChild(abody);
    var tabwrap = h('div', { class: 'ln-tabwrap' });
    var cardwrap = h('div', { class: 'ln-cardwrap' });
    var checks = h('div', { class: 'ln-checks' });
    abody.appendChild(tabwrap); abody.appendChild(cardwrap); abody.appendChild(checks);

    /* ---- table ---- */
    var table = h('table', { class: 'ln-grid' });
    var cg = h('colgroup', null, [h('col', { class: 'c0' }), h('col'), h('col'), h('col')]);
    table.appendChild(cg);
    var thead = h('thead'), trh = h('tr');
    trh.appendChild(h('th', { class: 'corner', scope: 'col', html: 'projector ↓<br>schedule →' }));
    ACOLS.forEach(function (c) { trh.appendChild(h('th', { scope: 'col', html: A.esc(c.lab) + '<span class="s">' + c.sub.replace('_kT', '<sub>kT</sub>').replace('_0', '<sub>0</sub>').replace('a_t', 'a<sub>t</sub>') + '</span>' })); });
    thead.appendChild(trh); table.appendChild(thead);
    var tbody = h('tbody');
    var chipEls = {};
    AROWS.forEach(function (rw) {
      var tr = h('tr', { class: rw.id === 'two' ? 'contrast' : null });
      tr.appendChild(h('th', { scope: 'row', html: A.esc(rw.lab) + '<span class="s">' + rw.sub + '</span>' }));
      ACOLS.forEach(function (c) {
        var td = h('td', { class: 'cell' });
        METHODS.forEach(function (mt) {
          if (mt.row !== rw.id || mt.col !== c.id) return;
          var b = h('button', { type: 'button', class: 'ln-m ' + mt.v + ((mt.short || mt.name).length <= 11 ? ' nw' : ''), 'aria-pressed': 'false', 'aria-label': mt.name + ': ' + VERDICT[mt.v] + (mt.v2 ? '; its theta-dependent reading is non-involutive' : '') });
          b.appendChild(document.createTextNode(mt.short || mt.name));
          if (mt.v2) b.appendChild(h('span', { class: 'dot', 'aria-hidden': 'true' }));
          b.addEventListener('mouseenter', function () { preview(mt.id); });
          b.addEventListener('focus', function () { select(mt.id, true); });
          b.addEventListener('click', function () { select(mt.id); });
          chipEls[mt.id] = b;
          td.appendChild(b);
          if (mt.note) td.appendChild(h('span', { class: 'ln-cnote', text: mt.note }));
        });
        var cn = CELLNOTE[rw.id + '|' + c.id];
        if (cn) td.appendChild(h('span', { class: 'ln-cnote', text: cn }));
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tabwrap.appendChild(table);
    tabwrap.addEventListener('mouseleave', function () { showCard(selId); });
    tabwrap.appendChild(h('div', { class: 'ln-legend' }, [
      h('span', { class: 'ln-li' }, [h('span', { class: 'ln-sw fwd' }), h('span', { text: 'forward lens ρ (tangent side)' })]),
      h('span', { class: 'ln-li' }, [h('span', { class: 'ln-sw per' }), h('span', { text: 'linear ρ, re-based each period (Prop IV.2)' })]),
      h('span', { class: 'ln-li' }, [h('span', { class: 'ln-sw out' }), h('span', { text: 'outside Def IV.1' })]),
      h('span', { class: 'ln-li' }, [h('span', { class: 'ln-sw dot' }), h('span', { text: 'θ-dependent reading non-involutive (Prop IV.3)' })])
    ]));

    /* ---- cards (built lazily, typeset once, cached) ---- */
    var cards = {}, selId = 'galore', shownId = null;
    function methodById(id) { for (var i = 0; i < METHODS.length; i++) if (METHODS[i].id === id) return METHODS[i]; return null; }
    function buildCard(mt) {
      var rowLab = AROWS.filter(function (r) { return r.id === mt.row; })[0].lab, colLab = ACOLS.filter(function (c) { return c.id === mt.col; })[0].lab;
      var card = h('article', { class: 'ln-card', 'aria-label': mt.name + ' card' });
      var cite = mt.url ? '<a href="' + mt.url + '" target="_blank" rel="noopener">' + A.esc(mt.cite) + (mt.ax ? ' · arXiv ' + mt.ax : '') + '</a>' : A.esc(mt.cite);
      card.appendChild(h('div', { class: 'top' }, [h('h4', { text: mt.name }), h('span', { class: 'ln-cite', html: cite })]));
      card.appendChild(h('div', { class: 'ln-where', text: rowLab + ' × ' + colLab }));
      card.appendChild(h('span', { class: 'ln-vchip ' + mt.v, text: VERDICT[mt.v] }));
      if (mt.v2) card.appendChild(h('span', { class: 'ln-vchip out', text: 'θ-dependent reading: non-involutive' }));
      card.appendChild(h('p', { class: 'ln-vline', html: mt.vline }));
      card.appendChild(h('dl', { class: 'ln-dl' }, [
        h('dt', { text: 'compression c' }), h('dd', { html: mt.c }),
        h('dt', { text: 'anchor a' }), h('dd', { html: mt.a + (mt.cnote ? '<span class="n">' + mt.cnote + '</span>' : '') })
      ]));
      card.appendChild(h('div', { class: 'ln-upd', html: mt.upd }));
      card.appendChild(h('p', { class: 'ln-txt', html: mt.read }));
      if (mt.hyp) {
        card.appendChild(h('div', { class: 'ln-sub', text: mt.hl || (mt.v === 'fwd' ? 'Forward = backward exactly, if' : 'Exact, if (repaired hypotheses)') }));
        var ul = h('ol', { class: 'ln-hyp' });
        mt.hyp.forEach(function (x) { ul.appendChild(h('li', { html: x })); });
        card.appendChild(ul);
        if (mt.brk) card.appendChild(h('p', { class: 'ln-brk', html: mt.brk }));
      }
      card.appendChild(h('p', { class: 'ln-cone', html: '<b>Cone · Obs IV.4</b>' + mt.cone }));
      if (mt.link) {
        var go = h('button', { type: 'button', class: 'ln-go', text: 'Show its cone in B →' });
        go.addEventListener('click', function () { linkToB(mt.link); });
        card.appendChild(go);
      }
      return card;
    }
    function showCard(id) {
      if (shownId === id && cardwrap.firstChild) return;
      var mt = methodById(id); if (!mt) return;
      var fresh = !cards[id];
      if (fresh) cards[id] = buildCard(mt);
      cardwrap.innerHTML = '';
      cardwrap.appendChild(cards[id]);
      shownId = id;
      if (fresh) ts(cards[id]);
    }
    function preview(id) { showCard(id); }
    function select(id, quiet) {
      selId = id;
      Object.keys(chipEls).forEach(function (k) { chipEls[k].setAttribute('aria-pressed', String(k === id)); });
      showCard(id);
      var mt = methodById(id);
      if (!quiet && mt) live.textContent = mt.name + ': ' + VERDICT[mt.v] + '. Its card follows the table.';
    }

    /* ---- live check 1: Prop IV.2 ---- */
    var VARIANTS = [
      { id: 'carry', label: 'carry state' },
      { id: 'reset', label: 'reset state' },
      { id: 'adamw', label: 'AdamW λ=0.1' },
      { id: 'clip', label: 'clip ‖G‖≤1' }
    ];
    var WHY = {
      carry: 'Both sides feed Adam the same sequence \\(P_k^{\\top}\\nabla L(W_t)\\) from the same state, so they get the same outputs and the gap is round-off.',
      reset: 'The LoRA side resets its moments and step counter at every re-basing, as ReLoRA does, while GaLore carries them over. The trajectories split.',
      adamw: 'Decoupled decay pulls \\(W\\) toward \\(0\\) on the GaLore side and \\(C\\) toward \\(0\\) (so \\(W\\) toward \\(W_k\\)) on the LoRA side.',
      clip: 'Clipping reads \\(\\|G\\|\\) on the GaLore side and \\(\\|P_k^{\\top}G\\|\\) on the LoRA side, so the two optimizers are fed different sequences.'
    };
    var iv2Res = {};
    VARIANTS.forEach(function (vv) { iv2Res[vv.id] = iv2Run(A, vv.id); });
    var iv2Var = 'carry';
    var c1 = h('div', { class: 'ln-check' });
    c1.appendChild(h('h5', { html: 'Prop IV.2, live: GaLore vs one-sided LoRA <span class="ln-tag">float64</span>' }));
    var segIV2 = A.seg(VARIANTS.map(function (vv) { return { value: vv.id, label: vv.label }; }), iv2Var, function (v) { iv2Var = v; drawIV2(); }, 'Prop IV.2 variant');
    segIV2.el.classList.add('ln-segg');
    c1.appendChild(segIV2.el);
    var m1 = h('div', { class: 'ln-meter' });
    var m1lab = h('span', { class: 'ln-blab', html: '\\(\\max_t\\|W^{\\text{GaLore}}_t-W^{\\text{LoRA}}_t\\|_F\\)' });
    var m1big = h('span', { class: 'ln-big' }), m1chip = h('span', { class: 'ln-chip' });
    m1.appendChild(m1lab); m1.appendChild(m1big); m1.appendChild(m1chip);
    c1.appendChild(m1);
    var spark = h('div', { class: 'ln-spark' });
    c1.appendChild(spark);
    var why1 = h('p', { class: 'ln-why' });
    c1.appendChild(why1);
    c1.appendChild(h('div', { class: 'ln-mini', html: 'Toy least squares, \\(W\\in\\mathbb R^{' + IV2.m + '\\times' + IV2.n + '}\\), ' + IV2.N + ' samples, \\(r=' + IV2.r + '\\), \\(T=' + IV2.T + '\\), ' + IV2.K + ' periods, Adam \\((0.9,\\,0.999,\\,10^{-8})\\), \\(\\eta=' + IV2.eta + '\\); LoRA’s frozen factor is GaLore’s \\(P_k\\) exactly, column signs included (largest entry positive). Scale: \\(\\|W_T-W_0\\|_F=' + iv2Res.carry.move.toFixed(3) + '\\).' }));
    checks.appendChild(c1);
    var whyCache = {};
    function drawIV2() {
      var R = iv2Res[iv2Var], ok = R.max < 1e-12;
      m1big.innerHTML = supHTML(sci(R.max, 1));
      m1big.className = 'ln-big ' + (ok ? 'z' : 'b');
      m1chip.className = 'ln-chip ' + (ok ? 'z' : 'b');
      m1chip.textContent = ok ? 'identical to round-off' : 'identity broken';
      var fresh = !whyCache[iv2Var];
      if (fresh) whyCache[iv2Var] = h('span', { html: WHY[iv2Var] });
      why1.innerHTML = ''; why1.appendChild(h('b', { text: ok ? 'Holds. ' : 'Breaks. ', style: 'color:var(--' + (ok ? 'moss' : 'seal') + ');font-weight:600' })); why1.appendChild(whyCache[iv2Var]);
      if (fresh) ts(whyCache[iv2Var]);
      drawSpark(R);
    }
    function drawSpark(R) {
      var w = Math.max(240, Math.min(560, Math.floor(spark.getBoundingClientRect().width) || spark.clientWidth || 320)), H = 76, ml = 40, mr = 6, mt = 8, mb = 20;
      var lo = -17, hi = 0, n = R.gaps.length;
      var x = function (i) { return ml + (w - ml - mr) * (i / (n - 1)); };
      var y = function (g) { var lg = g > 0 ? Math.log(g) / Math.LN10 : lo; lg = Math.max(lo, Math.min(hi, lg)); return mt + (H - mt - mb) * (hi - lg) / (hi - lo); };
      spark.innerHTML = '';
      var s = sv('svg', { width: w, height: H, viewBox: '0 0 ' + w + ' ' + H, role: 'img', 'aria-label': 'Gap per step on a log scale; it stays at round-off when the identity holds.' }, spark);
      [-16, -8, 0].forEach(function (e) {
        sv('line', { x1: ml, x2: w - mr, y1: y(Math.pow(10, e)), y2: y(Math.pow(10, e)), stroke: 'var(--rule)', 'stroke-width': 1 }, s);
        /* tick values: mono 11px, the exponent as raised digits */
        var t = sv('text', { x: ml - 5, y: y(Math.pow(10, e)) + 4, 'text-anchor': 'end', class: 'num', style: 'font-size:11px;fill:var(--ink-2)' }, s);
        if (e === 0) t.textContent = '1';
        else { t.textContent = '10'; var ex = sv('tspan', { dy: -4.4, style: 'font-size:11px' }, t); ex.textContent = '−' + String(-e); }
      });
      for (var k = 1; k < IV2.K; k++) {
        var xx = x(k * IV2.T - 0.5);
        sv('line', { x1: xx, x2: xx, y1: mt, y2: H - mb, stroke: 'var(--ink-3)', 'stroke-dasharray': '2 3', 'stroke-width': 1 }, s);
      }
      var tt = sv('text', { x: x(IV2.T - 0.5) + 4, y: H - 4, style: 'font-size:12px;font-weight:500;fill:var(--ink-2)' }, s); tt.textContent = 're-base';
      var t2 = sv('text', { x: w - mr, y: H - 4, 'text-anchor': 'end', style: 'font-size:12px;font-weight:500;fill:var(--ink-2)' }, s); t2.textContent = 'step ' + n;
      var d = '';
      R.gaps.forEach(function (g, i) { d += (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(g).toFixed(1); });
      var ok = R.max < 1e-12;
      sv('path', { d: d, fill: 'none', stroke: ok ? 'var(--moss)' : 'var(--seal)', 'stroke-width': 1.8, 'stroke-linejoin': 'round' }, s);
    }

    /* ---- live check 2: Prop IV.3 ---- */
    var iv3 = iv3Run(A);
    var c2 = h('div', { class: 'ln-check' });
    c2.appendChild(h('h5', { html: 'Prop IV.3, live: is the gradient-subspace field integrable? <span class="ln-tag">float64</span>' }));
    c2.appendChild(h('div', { class: 'ln-why', html: '\\(N\\) = number of samples. Full-batch GaLore with \\(T=1\\) read as \\(D_W=\\mathrm{Hom}(\\mathbb R^n,U_r(\\nabla L(W)))\\). For frame fields \\(X=\\Pi N\\), \\(Y=\\Pi M\\) (\\(\\Pi\\) = projector onto \\(U_r\\)), Frobenius asks whether \\([X,Y]\\) stays in <span style="white-space:nowrap">\\(D\\).</span>' }));
    var pt = h('table', { class: 'ln-pt' });
    pt.appendChild(h('thead', null, [h('tr', null, [h('th', { text: 'field' }), h('th', { html: 'max ‖[X,Y]‖' }), h('th', { html: 'normal part ÷ ‖[X,Y]‖' }), h('th', { text: 'verdict' })])]));
    var ptb = h('tbody');
    [
      { k: 'r1N7', lab: 'r=1, N=' + IV3.N },
      { k: 'r2N7', lab: 'r=2, N=' + IV3.N },
      { k: 'r1N1', lab: 'r=1, N=1' }
    ].forEach(function (row) {
      var R = iv3[row.k], inv = R.brmax < 1e-9;
      ptb.appendChild(h('tr', null, [
        h('th', { text: row.lab }),
        h('td', { html: supHTML(sci(R.brmax, 1)) }),
        h('td', { class: 'w ' + (inv ? '' : 'b'), text: inv ? '—' : R.rmin.toFixed(2) + '–' + R.rmax.toFixed(2) }),
        h('td', { class: inv ? 'z' : 'b', text: inv ? 'involutive' : 'not involutive' })
      ]));
    });
    pt.appendChild(ptb);
    c2.appendChild(pt);
    c2.appendChild(h('p', { class: 'ln-why', html: 'Generic data give a normal component of order one at \\(r=1\\) and \\(r=2\\) alike, so this field is not foliated by images of forward methods. With a single sample \\(U_1=\\mathrm{span}(Wx-y)\\) is constant along \\(D\\), the bracket vanishes and the leaves are affine. Fixed-projector methods (GaLore with \\(T>1\\), LISA, MeZO) have constant anchors on each period, so for them the test is vacuous.' }));
    c2.appendChild(h('div', { class: 'ln-mini', html: 'Least squares \\(L=\\|WX-Y\\|_F^2/2N\\), \\(W\\in\\mathbb R^{' + IV3.m + '\\times' + IV3.n + '}\\) (\\(m\\lt n\\): GaLore projects on the left), 8 random frame pairs \\((N,M)\\), \\(D\\Pi\\) by first-order perturbation of the top-\\(r\\) eigenspace of \\(GG^{\\top}\\).' }));
    checks.appendChild(c2);

    /* ==================================================================================
       PANEL B
       ================================================================================== */
    var pB = h('section', { class: 'ln-panel', 'aria-label': 'Panel B: the backprop cone' });
    stage.appendChild(pB);
    pB.appendChild(h('div', { class: 'ln-phead' }, [
      h('span', { class: 'ln-badge', text: 'B' }),
      h('span', { class: 'ln-ptitle', html: 'The backprop cone: <i>what the backward pass must remember</i>' }),
      h('span', { class: 'ln-pthm', text: 'Obs IV.4 · an upper bound' })
    ]));
    pB.appendChild(h('p', { class: 'ln-pinstr', html: '<b>Pick a preset, or click (or arrow-key and Enter) a box</b> to make it trainable. Shading is the backward cone; the bars are an <b>estimate</b> of training memory under the accounting in the footnote.' }));

    var st = { model: 'llama3-8b', preset: 'lora-qv', base: 'lora-qv', r: 16, gamma: 2, k: 4, p: 20, fa: false, b: 1, s: 4096, eager: false, master: true, lisaSeed: 1, cur: null, tr: null };
    function cfg() { for (var i = 0; i < MODELS.length; i++) if (MODELS[i].id === st.model) return MODELS[i]; return MODELS[0]; }
    function lisaLayers() {
      var L = cfg().L, g = Math.min(st.gamma, L), rnd = A.rng(1000 + st.lisaSeed), idx = [], i;
      for (i = 0; i < L; i++) idx.push(i);
      for (i = 0; i < g; i++) { var j = i + Math.floor(rnd() * (L - i)); var tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; }
      return idx.slice(0, g).sort(function (a, b) { return a - b; });
    }
    function opts(extra) {
      var o = { r: st.r, p: st.p, k: st.k, fa: st.fa, b: st.b, s: st.s, eager: st.eager, master: st.master, mezo: st.preset === 'mezo', lisaLayers: lisaLayers() };
      if (extra) for (var k in extra) o[k] = extra[k];
      return o;
    }
    function applyPreset(id) { st.preset = id; st.base = id; st.tr = presetTrain(id, cfg(), opts()); }

    var controls = h('div', { class: 'ln-controls' });
    pB.appendChild(controls);
    /* group 1: what trains */
    var g1 = h('div', { class: 'ln-group' }, [h('span', { class: 'ln-tag', text: 'What trains' })]);
    var segPre = A.seg(PRESETS.map(function (p) { return { value: p.id, label: p.label }; }), st.preset, function (v) { applyPreset(v); syncControls(); render(); announce(); }, 'Trainable-set preset');
    segPre.el.classList.add('ln-segg', 'ln-seg4');
    g1.appendChild(segPre.el);
    var paramRow = h('div', { class: 'ln-row' });
    g1.appendChild(paramRow);
    var uid = 'ln' + Math.random().toString(36).slice(2, 7);
    function mkSlider(id, label, nm, min, max, val, fmt, on) {
      var inp = h('input', { type: 'range', id: uid + id, min: min, max: max, step: 1, value: val });
      var out = h('output', { for: uid + id, text: fmt(val) });
      inp.addEventListener('input', function () { out.textContent = fmt(+inp.value); on(+inp.value); });
      var wrap = h('div', { class: 'ln-sl', style: 'flex:1 1 11rem' }, [h('div', { class: 'ln-sl-top' }, [h('label', { for: uid + id, html: label + '<span class="nm">' + nm + '</span>' }), out]), inp]);
      return { el: wrap, input: inp, out: out, set: function (v) { inp.value = v; out.textContent = fmt(v); } };
    }
    var slR = mkSlider('r', '<i>r</i>', 'LoRA rank', 0, RANKS.length - 1, RANKS.indexOf(st.r), function (i) { return String(RANKS[i]); }, function (i) { st.r = RANKS[i]; schedule(); });
    var slG = mkSlider('g', '<i>γ</i>', 'sampled blocks', 1, 32, st.gamma, String, function (v) { st.gamma = v; rebuildKeep(); schedule(); });
    var slK = mkSlider('k', '<i>k</i>', 'top blocks', 1, 32, st.k, String, function (v) { st.k = v; rebuildKeep(); schedule(); });
    var slP = mkSlider('p', '<i>p</i>', 'prompt tokens', 1, 200, st.p, String, function (v) { st.p = v; schedule(); });
    var faBtn = h('button', { type: 'button', class: 'ln-tog', 'aria-pressed': 'false', text: 'A₀ frozen (LoRA-FA)' });
    faBtn.addEventListener('click', function () { st.fa = !st.fa; faBtn.setAttribute('aria-pressed', String(st.fa)); schedule(); announce(); });
    var reBtn = h('button', { type: 'button', class: 'ln-btn', text: 'Resample blocks ↻' });
    reBtn.addEventListener('click', function () { st.lisaSeed++; rebuildKeep(); schedule(); announce(); });
    var customTag = h('span', { class: 'ln-custom', text: 'custom' });
    var presetHint = h('div', { class: 'ln-hint' });
    g1.appendChild(presetHint);
    /* rebuild the trainable set from the current base preset (custom edits are dropped when its parameter moves) */
    function rebuildKeep() { if (st.preset !== 'custom') st.tr = presetTrain(st.preset, cfg(), opts()); else { st.preset = st.base; st.tr = presetTrain(st.base, cfg(), opts()); syncControls(); } }

    /* group 2: model and tokens */
    var g2 = h('div', { class: 'ln-group' }, [h('span', { class: 'ln-tag', text: 'Model & tokens' })]);
    var selId2 = uid + '-model';
    var sel = h('select', { id: selId2, class: 'ln-sel' });
    MODELS.forEach(function (m) { sel.appendChild(h('option', { value: m.id, text: m.name })); });
    sel.value = st.model;
    sel.addEventListener('change', function () {
      st.model = sel.value;
      var L = cfg().L;
      slG.input.max = L; slK.input.max = L;
      if (st.gamma > L) st.gamma = L; if (st.k > L) st.k = L;
      slG.set(st.gamma); slK.set(st.k);
      if (st.preset === 'custom') st.preset = st.base;
      applyPreset(st.preset); syncControls(); render(); announce();
    });
    var cfgHint = h('div', { class: 'ln-hint num' });
    g2.appendChild(h('div', { class: 'ln-sl' }, [h('label', { for: selId2, html: 'model<span class="nm">HF config</span>' }), sel, cfgHint]));
    var slB = mkSlider('b', '<i>b</i>', 'batch', 0, 6, Math.log2(st.b), function (i) { return String(Math.pow(2, i)); }, function (i) { st.b = Math.pow(2, i); schedule(); });
    var slS = mkSlider('s', '<i>s</i>', 'sequence', 8, 15, Math.log2(st.s), function (i) { return intStr(Math.pow(2, i)); }, function (i) { st.s = Math.pow(2, i); schedule(); });
    g2.appendChild(h('div', { class: 'ln-row' }, [slB.el, slS.el]));
    /* group 3: accounting */
    var g3 = h('div', { class: 'ln-group' }, [h('span', { class: 'ln-tag', text: 'Accounting' })]);
    var segAtt = A.seg([{ value: 'fused', label: 'attn fused' }, { value: 'eager', label: 'attn eager' }], 'fused', function (v) { st.eager = v === 'eager'; schedule(); }, 'Attention residual');
    g3.appendChild(segAtt.el);
    var masterBtn = h('button', { type: 'button', class: 'ln-tog', 'aria-pressed': 'true', text: 'fp32 master copy' });
    masterBtn.addEventListener('click', function () { st.master = !st.master; masterBtn.setAttribute('aria-pressed', String(st.master)); schedule(); });
    g3.appendChild(h('div', { class: 'ln-row' }, [masterBtn]));
    var accHint = h('div', { class: 'ln-hint' });
    g3.appendChild(accHint);
    controls.appendChild(g1); controls.appendChild(g2); controls.appendChild(g3);

    function syncControls() {
      var p = st.preset === 'custom' ? st.base : st.preset;
      segPre.set(st.preset === 'custom' ? '__none' : st.preset);
      paramRow.innerHTML = '';
      if (st.preset === 'custom') paramRow.appendChild(customTag);
      if (p === 'lora-qv' || p === 'lora-all') { paramRow.appendChild(slR.el); paramRow.appendChild(faBtn); }
      else if (p === 'lisa') { paramRow.appendChild(slG.el); paramRow.appendChild(reBtn); }
      else if (p === 'topk') paramRow.appendChild(slK.el);
      else if (p === 'prompt') paramRow.appendChild(slP.el);
      var HINT = {
        full: 'every weight, norm gain, the embedding and the head',
        'lora-qv': 'rank-<i>r</i> adapters on W<sub>q</sub> and W<sub>v</sub> in every block; W frozen',
        'lora-all': 'rank-<i>r</i> adapters on all seven linears of every block',
        bitfit: 'zero-initialised biases on the seven linears (Llama has none)',
        lisa: 'embedding and LM head always, plus γ sampled blocks (Pan et al. 2024: E + H + 2L)',
        topk: 'the top k blocks, norms included; embedding, final norm and head frozen',
        prompt: 'p soft-prompt vectors before block 1; the model frozen',
        mezo: 'all weights, zeroth order: two forward passes per step, no backward pass'
      };
      presetHint.innerHTML = (st.preset === 'custom' ? 'edited from ' + PRESETS.filter(function (x) { return x.id === st.base; })[0].label + ': ' : '') + HINT[p];
    }

    /* strip */
    var stripBox = h('div', { class: 'ln-strip' });
    pB.appendChild(stripBox);
    var stripLegend = h('div', { class: 'ln-slegend', 'aria-label': 'Legend for the strip' });
    pB.appendChild(stripLegend);
    function legendGlyph(kind) {
      var s = document.createElementNS(SVGNS, 'svg'); s.setAttribute('width', 14); s.setAttribute('height', 12); s.setAttribute('aria-hidden', 'true');
      if (kind === 'cone') sv('rect', { x: 0, y: 0, width: 14, height: 12, class: 'cone', style: 'fill:var(--tide-soft)' }, s);
      if (kind === 'tw') sv('rect', { x: 2, y: 1.5, width: 9, height: 9, style: 'fill:var(--ochre)' }, s);
      if (kind === 'keep') { sv('rect', { x: 2, y: 1.5, width: 9, height: 9, style: 'fill:var(--ochre)' }, s); sv('rect', { x: 0.8, y: 0.3, width: 11.4, height: 11.4, style: 'fill:none;stroke:var(--tide);stroke-width:1.6' }, s); }
      if (kind === 'adp') { sv('rect', { x: 2, y: 1.5, width: 9, height: 9, style: 'fill:var(--paper-2);stroke:var(--ink-3);stroke-opacity:.55' }, s); sv('circle', { cx: 6.5, cy: 6, r: 2.4, style: 'fill:var(--ochre)' }, s); }
      if (kind === 'frz') sv('rect', { x: 2, y: 1.5, width: 9, height: 9, style: 'fill:var(--paper-2);stroke:var(--ink-3);stroke-opacity:.55' }, s);
      if (kind === 'nlon') sv('circle', { cx: 6.5, cy: 6, r: 4.5, style: 'fill:var(--tide)' }, s);
      if (kind === 'nloff') sv('circle', { cx: 6.5, cy: 6, r: 4.3, style: 'fill:none;stroke:var(--ink-3);stroke-opacity:.6' }, s);
      if (kind === 'gain') sv('circle', { cx: 6.5, cy: 6, r: 4.3, style: 'fill:var(--tide);stroke:var(--ochre);stroke-width:1.8' }, s);
      return s;
    }
    [
      ['cone', 'backward cone (cotangents flow back from ℒ)'],
      ['tw', 'trainable weight'],
      ['adp', 'adapter on a frozen box (LoRA, bias)'],
      ['keep', 'tide ring: keeps its input'],
      ['frz', 'frozen linear: keeps nothing'],
      ['nlon', 'nonlinear on the cone: keeps a residual'],
      ['nloff', 'off the cone: keeps nothing'],
      ['gain', 'trainable norm gain']
    ].forEach(function (x) { stripLegend.appendChild(h('span', { class: 'ln-li' }, [legendGlyph(x[0]), h('span', { text: x[1] })])); });

    /* results */
    var res = h('div', { class: 'ln-res' });
    pB.appendChild(res);
    var bars = h('div', { class: 'ln-bars' });
    var readout = h('div', { class: 'ln-readout' });
    res.appendChild(bars); res.appendChild(readout);
    var foot = h('div', { class: 'ln-foot' });
    pB.appendChild(foot);

    var BARS = [
      { key: 'weights', lab: 'Weights', sub: '2 B × (|θ₀| + new)', cls: 'c-w' },
      { key: 'grads', lab: 'Gradients', sub: '2 B × |M|', cls: 'c-g' },
      { key: 'opt', lab: 'Optimizer', sub: 'Adam m, v (+ master)', cls: 'c-o' },
      { key: 'act', lab: 'Activations', sub: 'kept by the cone', cls: 'c-anl' }
    ];
    var barEls = {};
    bars.appendChild(h('div', { class: 'ln-bh' }, [h('span', { class: 'ln-rl', text: 'Training memory, GiB (2³⁰ B)' }), h('span', { class: 'est', text: 'estimate · stated assumptions' })]));
    BARS.forEach(function (b) {
      var track = h('div', { class: 'ln-track' });
      var ghost = h('div', { class: 'ln-ghost' });
      var fill = h('div', { class: 'ln-fill ' + b.cls }), fill2 = b.key === 'act' ? h('div', { class: 'ln-fill c-ain' }) : null;
      track.appendChild(fill); if (fill2) track.appendChild(fill2); track.appendChild(ghost);
      var v = h('div', { class: 'v' });
      var row = h('div', { class: 'ln-br' }, [h('div', { class: 'k', html: b.lab + '<small>' + b.sub + '</small>' }), track, v]);
      bars.appendChild(row);
      barEls[b.key] = { fill: fill, fill2: fill2, ghost: ghost, v: v };
    });
    var totTrack = h('div', { class: 'ln-track tot' }), totV = h('div', { class: 'v', style: 'font-weight:600' }), totGhost = h('div', { class: 'ln-ghost' }), totRef = h('div', { class: 'ln-ref' });
    var totFills = BARS.map(function (b) { return h('div', { class: 'ln-fill ' + b.cls }); });
    var totAin = h('div', { class: 'ln-fill c-ain' });
    totFills.forEach(function (f) { totTrack.appendChild(f); }); totTrack.appendChild(totAin); totTrack.appendChild(totGhost); totTrack.appendChild(totRef);
    bars.appendChild(h('div', { class: 'ln-br' }, [h('div', { class: 'k', html: 'Total<small>sum of the four</small>' }), totTrack, totV]));
    bars.appendChild(h('div', { class: 'ln-blegend' }, [
      h('span', null, [h('i', { class: 'c-anl' }), document.createTextNode('nonlinear residuals')]),
      h('span', null, [h('i', { class: 'c-ain' }), document.createTextNode('inputs of trainable boxes')]),
      h('span', null, [h('i', { class: 'gh' }), document.createTextNode('full fine-tuning, same setting')]),
      h('span', null, [h('i', { class: 'rl' }), document.createTextNode('80 GiB')])
    ]));

    /* readout */
    var rBig = h('div', { class: 'ln-big' }), rSub = h('div', { class: 'ln-rsub' });
    var kv = h('table', { class: 'ln-kv' }), kvb = h('tbody');
    kv.appendChild(kvb);
    var rNote = h('p', { class: 'ln-note' });
    readout.appendChild(h('div', { class: 'ln-rl', text: 'Estimated training memory' }));
    readout.appendChild(rBig); readout.appendChild(rSub); readout.appendChild(kv); readout.appendChild(rNote);

    /* footnote: static formulas (typeset once) + live substitution (plain text) */
    foot.appendChild(h('span', { class: 'ln-tag', text: 'How each bar is computed' }));
    foot.appendChild(h('div', { class: 'ln-fmrow', html: '<span>\\(\\text{Weights}=2\\,\\mathrm B\\,(|\\theta_0|+|\\text{new}|)\\)</span><span>\\(\\text{Gradients}=2\\,\\mathrm B\\,|M|\\)</span><span>\\(\\text{Optimizer}=(4+4+4\\,[\\text{master}])\\,\\mathrm B\\,|M|\\)</span>' }));
    foot.appendChild(h('div', { class: 'fm', html: '\\[\\text{Activations}=b\\,(s+p)\\sum_{\\ell=1}^{L}\\ \\sum_{t\\in\\mathcal K_\\ell}\\mathrm{size}(t)\\;+\\;b\\,s\\sum_{t\\in\\mathcal K_{\\text{head}}}\\mathrm{size}(t)\\]' }));
    foot.appendChild(h('div', { class: 'ln-why', html: '\\(\\mathcal K_\\ell\\) is the set of tensors block \\(\\ell\\) keeps by Obs IV.4: (i) the input of each trainable linear or multiplicative box (an adapter’s \\(A\\) keeps \\(x\\), its \\(B\\) keeps \\(Ax\\); biases and prompts keep nothing), and (ii) a residual for each nonlinear box on the cone. A tensor needed by both is counted once. \\(d\\) hidden size, \\(d_{kv}\\) key/value width, \\(f\\) MLP width, \\(h\\) heads, \\(V\\) vocabulary, \\(S=s+p\\) positions.' }));
    var footLive = h('div', { class: 'eq' });
    foot.appendChild(footLive);
    foot.appendChild(h('span', { class: 'ln-tag', text: 'Assumptions' }));
    var assume = h('ul', { class: 'ln-hyp' });
    foot.appendChild(assume);
    foot.appendChild(h('span', { class: 'ln-tag', text: 'Caveats from the repaired Obs IV.4' }));
    var cav = h('ul', { class: 'ln-hyp' });
    [
      'Obs IV.4 is an upper bound: it says what <i>suffices</i> to keep. A residual may be an input, an output or a sufficient statistic (a ReLU mask); a framework may keep more (whatever each op saves) or less (recomputation).',
      'Activation checkpointing changes the picture: with it, PEFT’s <code>enable_input_require_grads</code> sends cotangents down to the embeddings, so top-k truncation holds only without checkpointing.',
      'Dropout on the cone keeps its mask (here LoRA dropout is 0; with dropout the adapter keeps an \\(n\\)-element mask as well).',
      'Forward mode: \\(k\\) JVPs give \\(P^{\\top}\\nabla L\\) with \\(O(1)\\)-block activation memory at a small constant times \\(k\\) forward passes. That is time-competitive with one backward pass only for \\(k\\approx1\\) (MeZO is \\(k=1\\) with a finite difference), and \\(P\\) must be fixed before the passes, so this route cannot produce GaLore’s SVD refresh more cheaply than a backward pass.'
    ].forEach(function (x) { cav.appendChild(h('li', { html: x })); });
    foot.appendChild(cav);

    /* ---------- link from A ---------- */
    function linkToB(preset) {
      if (preset === 'lora-fa') { st.fa = true; faBtn.setAttribute('aria-pressed', 'true'); preset = 'lora-all'; }
      else if (preset === 'lora-qv' || preset === 'lora-all') { st.fa = false; faBtn.setAttribute('aria-pressed', 'false'); }
      applyPreset(preset); syncControls(); render(); announce();
      try { pB.scrollIntoView({ behavior: A.reducedMotion() ? 'auto' : 'smooth', block: 'start' }); } catch (e) { pB.scrollIntoView(); }
    }

    /* ---------- strip drawing and interaction ---------- */
    var geo = null, cursor = null, an = null, ref = null;
    function colKind(c) { var L = cfg().L; return c === 0 ? 'E' : c <= L ? 'layer' : c === L + 1 ? 'N' : c === L + 2 ? 'H' : 'LS'; }
    function globalRow() { return 5; }
    function drawStrip() {
      var C = cfg(), L = C.L;
      /* the fractional box width, floored: a 1px-wider SVG would be scaled down below 12px text */
      var Wpx = Math.max(280, Math.floor(stripBox.getBoundingClientRect().width || stripBox.clientWidth || 640));
      var phone = Wpx < 560;
      var labW = phone ? 38 : 54, gap = phone ? 5 : 10, ncol = L + 4;
      var cw = Math.min(phone ? 15 : 24, (Wpx - labW - 2 * gap - 4) / ncol);
      var used = labW + 2 * gap + ncol * cw, x0 = Math.max(0, (Wpx - used) / 2);
      var rh = phone ? 13 : 15, top = 18, nR = ROWS.length, gridH = nR * rh;
      var barTop = top + gridH + 10, barH = phone ? 32 : 44, Hpx = barTop + barH + 20;
      var colX = function (c) {
        if (c === 0) return x0 + labW;
        if (c <= L) return x0 + labW + cw + gap + (c - 1) * cw;
        return x0 + labW + cw + gap + L * cw + gap + (c - L - 1) * cw;
      };
      geo = { L: L, cw: cw, rh: rh, top: top, colX: colX, labW: labW, x0: x0, ncol: ncol, nR: nR, W: Wpx, H: Hpx, gap: gap };
      /* the strip is rebuilt on every change (theme, resize, a toggle): keep keyboard focus on it */
      var hadFocus = stripBox.contains(document.activeElement);
      stripBox.innerHTML = '';
      var s = sv('svg', { width: Wpx, height: Hpx, viewBox: '0 0 ' + Wpx + ' ' + Hpx, tabindex: '0', role: 'application',
        'aria-label': 'Backprop cone strip for ' + C.name + ': embedding, ' + L + ' blocks of eleven boxes, final norm, head and loss. Arrow keys move a cursor over the boxes; Enter or Space toggles whether the box is trainable.' }, stripBox);
      var gs = Math.max(2, Math.min(cw - (cw > 9 ? 3.2 : 1.2), rh - 4));
      var cyRow = function (ri) { return top + ri * rh + rh / 2; };
      /* cone shading */
      var gcone = sv('g', null, s);
      var gR = an.g;
      if (!an.mezo) {
        if (gR.embR) sv('rect', { x: colX(0), y: top, width: cw, height: gridH, class: 'cone' }, gcone);
        an.layers.forEach(function (ly, l) {
          ROWS.forEach(function (rw, ri) { if (ly.R[rw.key]) sv('rect', { x: colX(l + 1), y: top + ri * rh, width: cw + 0.4, height: rh + 0.4, class: 'cone' }, gcone); });
        });
        if (gR.fnR) sv('rect', { x: colX(L + 1), y: top, width: cw, height: gridH, class: 'cone' }, gcone);
        if (gR.hdR) sv('rect', { x: colX(L + 2), y: top, width: cw, height: gridH, class: 'cone' }, gcone);
        if (gR.lsR) sv('rect', { x: colX(L + 3), y: top, width: cw, height: gridH, class: 'cone' }, gcone);
      }
      /* row labels */
      ROWS.forEach(function (rw, ri) { var t = sv('text', { x: x0 + labW - 5, y: cyRow(ri) + 4.2, 'text-anchor': 'end', class: 'lab' }, s); t.textContent = rw.lab; });
      /* column labels */
      var step = L <= 40 ? 8 : 20;
      var colLab = function (c, txt) { var t = sv('text', { x: colX(c) + cw / 2, y: top - 6, 'text-anchor': 'middle', class: 'lab col' }, s); t.textContent = txt; };
      colLab(0, 'E');
      for (var l = 1; l <= L; l++) if (l === 1 || l % step === 0) colLab(l, String(l));
      colLab(L + 1, 'N'); colLab(L + 2, 'H'); colLab(L + 3, 'ℒ');
      /* glyphs */
      var gg = sv('g', null, s);
      function linGlyph(cx, cy, kind, keeps) {
        var hw = gs / 2;
        sv('rect', { x: cx - hw, y: cy - hw, width: gs, height: gs, rx: gs > 6 ? 1 : 0, class: 'lin' + (kind === 'W' ? ' tw' : '') }, gg);
        if (kind === 'lora' || kind === 'bias') sv('circle', { cx: cx, cy: cy, r: Math.max(1, gs * 0.27), class: 'adp' }, gg);
        if (keeps) sv('rect', { x: cx - hw - 1.4, y: cy - hw - 1.4, width: gs + 2.8, height: gs + 2.8, rx: 1.5, class: 'keep' }, gg);
      }
      function nlGlyph(cx, cy, on, gain) {
        sv('circle', { cx: cx, cy: cy, r: Math.max(1, gs / 2), class: 'nl' + (on ? ' on' : '') + (gain ? ' gain' : '') }, gg);
      }
      var tr = st.tr;
      an.layers.forEach(function (ly, li) {
        var cx = colX(li + 1) + cw / 2, t = tr.layers[li];
        ROWS.forEach(function (rw, ri) {
          var cy = cyRow(ri);
          if (rw.kind === 'lin') linGlyph(cx, cy, t[rw.key], !an.mezo && ((t[rw.key] === 'W') || (t[rw.key] === 'lora')));
          else nlGlyph(cx, cy, !an.mezo && ly.R[rw.key], rw.kind === 'norm' && t[rw.key] === 'W');
        });
      });
      /* global boxes: prompt (diamond) and embedding in E, final norm N, head H, loss ℒ */
      var gy = cyRow(globalRow()), ex = colX(0) + cw / 2, pyy = cyRow(2), dd = gs / 2 + 0.6;
      sv('path', { d: 'M' + ex + ',' + (pyy - dd) + 'L' + (ex + dd) + ',' + pyy + 'L' + ex + ',' + (pyy + dd) + 'L' + (ex - dd) + ',' + pyy + 'Z', class: 'pr' + (tr.pr ? ' tw' : '') }, gg);
      linGlyph(ex, gy, tr.emb, !an.mezo && tr.emb === 'W');
      nlGlyph(colX(L + 1) + cw / 2, gy, !an.mezo && gR.fnR, tr.fn === 'W');
      linGlyph(colX(L + 2) + cw / 2, gy, tr.hd, !an.mezo && tr.hd === 'W');
      nlGlyph(colX(L + 3) + cw / 2, gy, !an.mezo && gR.lsR, false);
      if (!phone || cw >= 9) {
        var tp = sv('text', { x: ex, y: cyRow(1) + 4, 'text-anchor': 'middle', class: 'note' }, s); tp.textContent = 'P';
      }
      /* per-column kept bytes per position */
      var vals = [];
      vals.push({ c: 0, nl: 0, inn: gR.E });
      an.layers.forEach(function (ly, li) { vals.push({ c: li + 1, nl: ly.nlB, inn: ly.inB }); });
      vals.push({ c: L + 1, nl: gR.N, inn: 0 }, { c: L + 2, nl: 0, inn: gR.Hh }, { c: L + 3, nl: gR.Ls, inn: 0 });
      var vmax = 0; vals.forEach(function (v) { vmax = Math.max(vmax, v.nl + v.inn); });
      sv('line', { x1: colX(0), x2: colX(L + 3) + cw, y1: barTop + barH + 0.5, y2: barTop + barH + 0.5, class: 'base' }, s);
      var bl = sv('text', { x: x0 + labW - 5, y: barTop + barH - 1, 'text-anchor': 'end', class: 'lab' }, s); bl.textContent = 'kept';
      if (vmax > 0) {
        vals.forEach(function (v) {
          var bw = Math.max(1, cw - (cw > 6 ? 2 : 0.6)), bx = colX(v.c) + (cw - bw) / 2;
          var hNl = barH * v.nl / vmax, hIn = barH * v.inn / vmax;
          if (hNl > 0) sv('rect', { x: bx, y: barTop + barH - hNl, width: bw, height: hNl, class: 'bnl' }, s);
          if (hIn > 0) sv('rect', { x: bx, y: barTop + barH - hNl - hIn, width: bw, height: hIn, class: 'bin' }, s);
        });
        /* label the largest block value and the loss column */
        var maxL = 0, maxLc = 1; an.layers.forEach(function (ly, li) { if (ly.nlB + ly.inB > maxL) { maxL = ly.nlB + ly.inB; maxLc = li + 1; } });
        var lt = sv('text', { x: colX(1), y: barTop + barH + 14, class: 'val' }, s);
        lt.textContent = (maxL > 0 ? 'max per block ' + bstr(maxL) : 'blocks keep nothing') + (phone ? '' : ' / position');
        if (gR.Ls > 0) { var lt2 = sv('text', { x: colX(L + 3) + cw, y: barTop + barH + 14, 'text-anchor': 'end', class: 'val' }, s); lt2.textContent = 'ℒ ' + bstr(gR.Ls); }
      } else {
        var nt = sv('text', { x: colX(1), y: barTop + barH - 6, class: 'val' }, s);
        nt.textContent = an.mezo ? 'no reverse pass: nothing is kept for a backward pass' : 'nothing trainable: no backward pass';
      }
      /* keyboard cursor (shown only while the strip has focus) */
      curRect = sv('rect', { x: -99, y: -99, width: cw + 2, height: rh + 2, rx: 2, class: 'cur' }, s);
      placeCursor();
      s.addEventListener('mousemove', onMove);
      s.addEventListener('mouseleave', function () { A.tip.hide(); });
      s.addEventListener('click', onClick);
      s.addEventListener('keydown', onKey);
      s.addEventListener('focus', function () { if (!cursor) cursor = { c: 1, ri: 1 }; placeCursor(); describe(); });
      if (hadFocus) { try { s.focus({ preventScroll: true }); } catch (e) { s.focus(); } }
    }
    var curRect = null;
    function placeCursor() {
      if (!curRect || !cursor || !geo) return;
      var p = cellXY(cursor.c, cursor.ri);
      curRect.setAttribute('x', p.x - 1); curRect.setAttribute('y', p.y - 1);
    }
    function cellXY(c, ri) { if (!geo) return null; return { x: geo.colX(c), y: geo.top + ri * geo.rh }; }
    function hit(evt) {
      var svg = stripBox.querySelector('svg'); if (!svg || !geo) return null;
      var bb = svg.getBoundingClientRect(), x = (evt.clientX - bb.left) * (geo.W / bb.width), y = (evt.clientY - bb.top) * (geo.H / bb.height);
      var ri = Math.floor((y - geo.top) / geo.rh); if (ri < 0 || ri >= geo.nR) return null;
      for (var c = 0; c < geo.ncol; c++) { var cx = geo.colX(c); if (x >= cx && x < cx + geo.cw) return { c: c, ri: ri }; }
      return null;
    }
    /* which box sits at (c, ri)? global columns hold one box each (E holds the prompt and the embedding) */
    function boxAt(c, ri) {
      var k = colKind(c);
      if (k === 'layer') return { type: 'layer', l: c - 1, row: ROWS[ri] };
      if (k === 'E') { if (ri === 2) return { type: 'pr' }; if (ri === globalRow()) return { type: 'emb' }; return null; }
      if (ri !== globalRow()) return null;
      return { type: k === 'N' ? 'fn' : k === 'H' ? 'hd' : 'ls' };
    }
    function clickKind() { var b = st.preset === 'custom' ? st.base : st.preset; return (b === 'lora-qv' || b === 'lora-all') ? 'lora' : b === 'bitfit' ? 'bias' : 'W'; }
    function toggle(c, ri) {
      var bx = boxAt(c, ri); if (!bx) return false;
      if (bx.type === 'layer' && bx.row.kind === 'nl') return false;
      if (bx.type === 'ls') return false;
      var kind = clickKind();
      if (st.preset !== 'custom') { st.base = st.preset === 'mezo' ? 'full' : st.preset; st.preset = 'custom'; }
      var tr = st.tr;
      if (bx.type === 'pr') tr.pr = !tr.pr;
      else if (bx.type === 'emb') tr.emb = tr.emb ? null : 'W';
      else if (bx.type === 'fn') tr.fn = tr.fn ? null : 'W';
      else if (bx.type === 'hd') tr.hd = tr.hd ? null : 'W';
      else { var t = tr.layers[bx.l]; t[bx.row.key] = t[bx.row.key] ? null : (bx.row.kind === 'norm' ? 'W' : kind); }
      syncControls();
      return true;
    }
    function onClick(evt) { var ht = hit(evt); if (!ht) return; if (toggle(ht.c, ht.ri)) { cursor = ht; render(); describe(); showTip(evt, ht); } }
    function onMove(evt) { var ht = hit(evt); if (!ht) { A.tip.hide(); return; } showTip(evt, ht); }
    function showTip(evt, ht) { var html = boxInfo(ht.c, ht.ri, true); if (html) A.tip.show(html, evt); else A.tip.hide(); }
    function onKey(evt) {
      if (!cursor) cursor = { c: 1, ri: 1 };
      var L = cfg().L, k = evt.key, moved = false;
      if (k === 'ArrowRight') { cursor.c = Math.min(L + 3, cursor.c + 1); moved = true; }
      else if (k === 'ArrowLeft') { cursor.c = Math.max(0, cursor.c - 1); moved = true; }
      else if (k === 'ArrowUp') { cursor.ri = Math.max(0, cursor.ri - 1); moved = true; }
      else if (k === 'ArrowDown') { cursor.ri = Math.min(ROWS.length - 1, cursor.ri + 1); moved = true; }
      else if (k === 'Home') { cursor = { c: 1, ri: cursor.ri }; moved = true; }
      else if (k === 'End') { cursor = { c: L, ri: cursor.ri }; moved = true; }
      else if (k === 'Enter' || k === ' ') { evt.preventDefault(); if (toggle(cursor.c, cursor.ri)) render(); describe(); refocus(); return; }
      else return;
      evt.preventDefault();
      /* global columns hold boxes only on their rows: snap the cursor there */
      var ck = colKind(cursor.c);
      if (ck !== 'layer' && !boxAt(cursor.c, cursor.ri)) cursor.ri = (ck === 'E' && cursor.ri < 4) ? 2 : globalRow();
      if (moved) { placeCursor(); describe(); }
    }
    function refocus() { var svg = stripBox.querySelector('svg'); if (svg) svg.focus(); }
    function describe() { if (!cursor) return; var t = boxInfo(cursor.c, cursor.ri, false); live.textContent = t ? t.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ') : 'empty slot'; }
    function keepList(need) { return TORDER.filter(function (k) { return need[k]; }).map(function (k) { return TL[k].t; }); }
    function boxInfo(c, ri, rich) {
      var bx = boxAt(c, ri), C = cfg(), tr = st.tr; if (!bx) return '';
      var on, title, stat, keeps;
      var hd = function (t) { return '<div class="t">' + t + '</div>'; };
      if (an.mezo) {
        title = bx.type === 'layer' ? 'Block ' + (bx.l + 1) + ' · ' + bx.row.name : { pr: 'soft prompt', emb: 'embedding', fn: 'final RMSNorm', hd: 'LM head', ls: 'loss ℒ' }[bx.type];
        return hd(title) + 'MeZO: two forward passes per step, no reverse pass, so nothing is kept for a backward pass.';
      }
      if (bx.type === 'layer') {
        var ly = an.layers[bx.l], t = tr.layers[bx.l], rw = bx.row, k = rw.key;
        on = !!ly.R[k];
        title = 'Block ' + (bx.l + 1) + ' · ' + rw.name;
        if (rw.kind === 'lin') {
          var io = { q: [C.d, C.d], k: [C.d, C.kv], v: [C.d, C.kv], o: [C.d, C.d], gt: [C.d, C.f], up: [C.d, C.f], dn: [C.f, C.d] }[k];
          title += ' (' + io[0] + ' → ' + io[1] + ')';
          var inName = TL[INP[k]].t, inSz = io[0];
          if (t[k] === 'W') { stat = 'trainable weight'; keeps = 'keeps its input ' + inName + ' (' + intStr(inSz) + ' per position): its parameter VJP is y xᵀ'; }
          else if (t[k] === 'lora') { stat = (st.fa ? 'LoRA-FA adapter, A₀ frozen' : 'LoRA adapter') + ', r = ' + st.r; keeps = st.fa ? 'B keeps A₀x (' + st.r + ' per position); A₀ and W keep nothing' : 'A keeps ' + inName + ' (' + intStr(inSz) + '), B keeps Ax (' + st.r + ') per position; W keeps nothing'; }
          else if (t[k] === 'bias') { stat = 'trainable bias (additive)'; keeps = 'keeps nothing: a bias’s cotangent is the incoming cotangent'; }
          else { stat = 'frozen'; keeps = 'keeps nothing: a frozen linear box has R[f](x, y) = Wᵀy'; }
        } else {
          stat = (rw.kind === 'norm' && t[k] === 'W') ? 'trainable gain' : 'no parameters';
          if (rw.kind === 'norm' && t[k] !== 'W') stat = 'gain frozen';
          var res0 = { n1: ['x', '1/rms'], n2: ['h', '1/rms'], at: st.eager ? ['q', 'k', 'v', 'softmax (h × S × S)'] : ['q', 'k', 'v', 'o', 'lse'], ac: ['g', 'u'] }[k];
          keeps = on ? 'on the cone, keeps ' + res0.join(', ') : 'off the cone: keeps nothing';
        }
      } else {
        title = { pr: 'soft prompt (' + st.p + ' × ' + C.d + ')', emb: 'embedding (' + C.V + ' × ' + C.d + ')', fn: 'final RMSNorm', hd: 'LM head (' + C.d + ' → ' + C.V + ')', ls: 'loss ℒ (softmax cross-entropy)' }[bx.type];
        var g = an.g;
        if (bx.type === 'pr') { on = tr.pr; stat = tr.pr ? 'trainable, additive' : 'off'; keeps = tr.pr ? 'keeps nothing, but sits below block 1: the cone is maximal and every block keeps S = s + p positions' : 'click to add a soft prompt'; }
        else if (bx.type === 'emb') { on = g.embR; stat = tr.emb ? 'trainable' : 'frozen'; keeps = tr.emb ? 'keeps the token ids (8 B per token)' : 'keeps nothing'; }
        else if (bx.type === 'fn') { on = g.fnR; stat = tr.fn ? 'trainable gain' : 'gain frozen'; keeps = g.fnR ? 'on the cone, keeps its input and 1/rms' : 'off the cone: keeps nothing'; }
        else if (bx.type === 'hd') { on = g.hdR; stat = tr.hd ? 'trainable' : 'frozen'; keeps = tr.hd ? 'keeps its input (' + intStr(C.d) + ' per token)' : 'keeps nothing (frozen linear)'; }
        else { on = g.lsR; stat = 'nonlinear'; keeps = g.lsR ? 'keeps fp32 log-probabilities (' + intStr(C.V) + ' per token)' : 'no backward pass'; }
      }
      var coneTxt = on ? 'on the backward cone' : 'off the cone';
      return hd(title) + A.esc(stat) + ' · ' + coneTxt + '<br>' + A.esc(keeps);
    }

    /* ---------- bars, readout, footnote ---------- */
    function setW(elm, frac) { elm.style.width = (Math.max(0, Math.min(1, frac)) * 100).toFixed(3) + '%'; }
    function renderBars() {
      var scale = 0;
      BARS.forEach(function (b) { scale = Math.max(scale, an[b.key], ref[b.key]); });
      BARS.forEach(function (b) {
        var e = barEls[b.key];
        if (b.key === 'act') { setW(e.fill, an.actNl / scale); e.fill2.style.left = (an.actNl / scale * 100).toFixed(3) + '%'; setW(e.fill2, an.actIn / scale); }
        else setW(e.fill, an[b.key] / scale);
        setW(e.ghost, ref[b.key] / scale);
        e.v.textContent = barStr(an[b.key]);
      });
      var tmax = Math.max(an.total, ref.total, 80 * 1073741824) * 1.02, acc = 0;
      var parts = [an.weights, an.grads, an.opt, an.actNl];
      totFills.forEach(function (f, i) { f.style.left = (acc / tmax * 100).toFixed(3) + '%'; setW(f, parts[i] / tmax); acc += parts[i]; });
      totAin.style.left = (acc / tmax * 100).toFixed(3) + '%'; setW(totAin, an.actIn / tmax);
      setW(totGhost, ref.total / tmax);
      totRef.style.left = (80 * 1073741824 / tmax * 100).toFixed(3) + '%';
      totV.textContent = gstr(an.total);
    }
    function coneText() {
      if (an.mezo) return 'none: no reverse pass';
      if (!an.any) return 'none: nothing trainable';
      var L = cfg().L;
      if (an.g.embR) return 'from the ' + (st.tr.pr && st.tr.emb !== 'W' ? 'prompt' : 'embedding') + ' to ℒ · ' + an.blocksOn + ' / ' + L + ' blocks';
      if (an.coneStart) { var rw = ROWS.filter(function (r) { return r.key === an.coneStart.key; })[0]; return 'from block ' + (an.coneStart.l + 1) + ' · ' + rw.lab + ' to ℒ · ' + an.blocksOn + ' / ' + L + ' blocks'; }
      return 'head and loss only';
    }
    function renderReadout() {
      rBig.textContent = mstr(an.total);
      var save = 1 - an.total / ref.total;
      rSub.textContent = 'full fine-tuning, same setting: ' + gstr(ref.total) + ' GiB' + (st.preset === 'full' ? '' : ' · ' + (save >= 0 ? '−' : '+') + pctStr(Math.abs(save)));
      var rows = [
        ['trainable |M|', A.fmtCount(an.M) + ' <small>· ' + pctStr(an.M / an.P) + ' of ' + A.fmtCount(an.P) + '</small>'],
        ['backward cone', coneText()],
        ['kept per token', an.mezo ? '0' : bstr(an.act / an.tok) + ' <small>· all blocks, head and loss</small>'],
        ['activations', an.mezo ? '0 <small>(no reverse pass)</small>' : mstr(an.act) + ' <small>· ' + pctStr(an.act / ref.act) + ' of full FT</small>'],
        ['grad + opt', mstr(an.grads + an.opt) + (an.grads + an.opt > 0 ? ' <small>· ' + pctStr((an.grads + an.opt) / (ref.grads + ref.opt)) + ' of full FT</small>' : ' <small>(no reverse pass)</small>')]
      ];
      kvb.innerHTML = '';
      /* the cone row is words, not a number: Plex Sans */
      rows.forEach(function (r) { kvb.appendChild(h('tr', null, [h('th', { text: r[0] }), h('td', { html: r[1], class: r[0] === 'backward cone' ? 'w' : null })])); });
      rNote.innerHTML = presetNote();
    }
    function presetNote() {
      var p = st.preset, actPct = pctStr(an.act / ref.act), L = cfg().L;
      if (p === 'full') return 'Every box trains: every linear keeps its input and every nonlinear box keeps its residual. This is the reference drawn as dashed ghosts.';
      if (p === 'lora-qv' || p === 'lora-all') return st.fa
        ? 'With <i>A</i>₀ frozen (LoRA-FA) each <i>B</i> keeps only <i>A</i>₀<i>x</i>, <i>r</i> numbers per position; the nonlinear residuals on the cone stay. Activations: ' + actPct + ' of full fine-tuning.'
        : 'The frozen <i>W</i> keep nothing, but each adapter’s <i>A</i> keeps its input, <i>B</i> keeps <i>Ax</i>, and every nonlinear box downstream keeps its residual. Activations: ' + actPct + ' of full fine-tuning; the saving is gradients and optimizer state.';
      if (p === 'bitfit') return 'Biases are additive and keep nothing themselves, yet they sit in block 1, so the cone is the whole network and every nonlinear residual is kept: activations ' + actPct + ' of full fine-tuning.';
      if (p === 'lisa') {
        var un = L - Math.min(st.gamma, L);
        return 'The embedding trains at every step, so the cone is the whole network and nothing is truncated. ' +
          (un > 0 ? 'On the cone, the frozen linears of the ' + un + ' unsampled block' + (un === 1 ? '' : 's') + ' keep nothing, so activations fall to ' + actPct + ' of full fine-tuning; most of the saving is gradients and optimizer state.'
            : 'With every block sampled, activations are ' + actPct + ' of full fine-tuning.');
      }
      if (p === 'topk') {
        var kk = Math.min(st.k, L);
        if (kk >= L) return 'All ' + L + ' blocks train (norms included), so the cone starts at block 1 and covers every block. Activations: ' + actPct + ' of full fine-tuning.';
        return 'Only the top ' + (kk === 1 ? 'block trains' : kk + ' blocks train') + ', so the cone starts at block ' + (L - kk + 1) + ' and the ' + (L - kk === 1 ? 'block below it keeps' : (L - kk) + ' blocks below it keep') + ' nothing. Activations: ' + actPct + ' of full fine-tuning.';
      }
      if (p === 'prompt') return 'The prompt is additive and keeps nothing, but it enters below block 1: the cone is maximal, and every kept tensor grows to <i>S</i> = <i>s</i> + <i>p</i> positions. Activations: ' + actPct + ' of full fine-tuning, for ' + A.fmtCount(an.M) + ' trainable numbers.';
      if (p === 'mezo') return 'No reverse pass: no cone, no gradient tensors and no optimizer state (MeZO-SGD regenerates <i>z</i> from a seed). The price is two forward passes per step and many more steps; the transient working set of a forward pass is not counted.';
      return an.any ? 'Custom set of ' + an.nTrain + ' trainable boxes. The cone runs ' + coneText() + '. Activations: ' + actPct + ' of full fine-tuning.' : 'Nothing is trainable, so there is no backward pass and nothing to keep.';
    }
    var assumeKey = null;
    function renderFoot() {
      var C = cfg(), lines = [];
      lines.push('<b>this setting</b> · ' + C.name + ': |θ₀| = 2Vd + d + L(2d² + 2d·d<sub>kv</sub> + 3d·f + 2d) = ' + intStr(an.P) + ' · |new| = ' + intStr(an.Pnew) + ' · |M| = ' + intStr(an.M) + ' · b·s = ' + intStr(an.tok) + ' tokens' + (st.tr.pr ? ', S = s + p = ' + intStr(an.S) : '') + '.');
      if (!an.mezo && an.any) {
        var groups = {}, order = [];
        an.layers.forEach(function (ly, l) { var key = ly.sig || '∅'; if (!groups[key]) { groups[key] = { ls: [], ly: ly }; order.push(key); } groups[key].ls.push(l + 1); });
        order.forEach(function (key) {
          var gq = groups[key], ly = gq.ly;
          var lab = (gq.ls.length === 1 ? 'block ' : 'blocks ') + ranges(gq.ls);
          if (key === '∅') { lines.push(lab + ': keep nothing'); return; }
          var pp = polyOf(ly.need), s2 = polyStr(pp.p2, ['d', 'kv', 'f', 'r', 'hS']), s4 = polyStr(pp.p4, ['one', 'h']);
          lines.push(lab + ': 2 B × (' + s2 + ')' + (s4 ? ' + 4 B × (' + s4 + ')' : '') + ' = <b>' + bstr(ly.nlB + ly.inB) + '</b> per position');
        });
        var gl = [], g = an.g;
        if (g.E) gl.push('ids 8 B');
        if (g.N) gl.push('final norm 2 B × d + 4 B');
        if (g.Hh) gl.push('head input 2 B × d');
        if (g.Ls) gl.push('loss 4 B × V');
        lines.push('head side, per token: ' + gl.join(' + ') + ' = <b>' + bstr(g.nlB + g.inB) + '</b>');
        lines.push('Activations = ' + intStr(an.tokL) + ' × Σ blocks + ' + intStr(an.tok) + ' × head side = <b>' + mstr(an.act) + '</b>');
      } else if (an.mezo) lines.push('MeZO: Gradients = Optimizer = Activations = 0 (no reverse pass; MeZO-SGD keeps no state).');
      lines.push('Weights = 2 × ' + intStr(an.P + an.Pnew) + ' B = <b>' + mstr(an.weights) + '</b> · Gradients = 2 × ' + intStr(an.mezo ? 0 : an.M) + ' B = <b>' + mstr(an.grads) + '</b> · Optimizer = ' + (st.master ? 12 : 8) + ' × ' + intStr(an.mezo ? 0 : an.M) + ' B = <b>' + mstr(an.opt) + '</b>');
      footLive.innerHTML = lines.map(function (x) { return '<div>' + x + '</div>'; }).join('');
      var akey = st.eager + '|' + st.master;
      if (akey === assumeKey) return;
      assumeKey = akey;
      assume.innerHTML = '';
      [
        'bf16 (2 B) weights, adapters, gradients and stored activations; Adam \\(m, v\\) in fp32 (8 B per trainable number)' + (st.master ? ', plus an fp32 master copy (4 B)' : '; no master copy') + '.',
        'Per-box residuals with no activation checkpointing across boxes.',
        st.eager ? 'Attention (eager) keeps \\(q, k, v\\) and the \\(h\\times S\\times S\\) softmax probabilities.' : 'Attention (fused, FlashAttention-style) keeps \\(q, k, v, o\\) and the per-head log-sum-exp in fp32; the \\(S\\times S\\) scores are recomputed inside the box.',
        'RMSNorm keeps its input and \\(1/\\mathrm{rms}\\) (fp32); the SwiGLU product keeps \\(g, u\\); the loss keeps fp32 log-probabilities (\\(V\\) per token); a trainable embedding keeps the token ids.',
        'LoRA dropout 0; untied embeddings; block tensors counted at all \\(S=s+p\\) positions, head and loss at the \\(s\\) real ones.',
        'Not counted: temporaries and kernel workspaces, the CUDA context, allocator fragmentation, the transient working set of the forward pass.'
      ].forEach(function (x) { assume.appendChild(h('li', { html: x })); });
      ts(assume);
      void 0;
      accHint.textContent = (st.eager ? 'eager: softmax probabilities kept, grow as S²' : 'fused: q, k, v, o + lse kept, scores recomputed') + ' · Adam ' + (st.master ? '12' : '8') + ' B per trainable number';
    }

    /* ---------- render ---------- */
    var rafId = 0;
    function schedule() { if (rafId) return; rafId = requestAnimationFrame(function () { rafId = 0; render(); }); }
    function render() {
      var C = cfg(), o = opts();
      if (!st.tr || st.tr.layers.length !== C.L) st.tr = presetTrain(st.preset === 'custom' ? st.base : st.preset, C, o);
      an = analyse(C, st.tr, o);
      ref = analyse(C, presetTrain('full', C, o), opts({ mezo: false, p: 0 }));
      cfgHint.textContent = 'L ' + C.L + ' · d ' + C.d + ' · f ' + C.f + ' · d_kv ' + C.kv + ' · h ' + C.h + ' · V ' + intStr(C.V);
      drawStrip();
      renderBars();
      renderReadout();
      renderFoot();
      typesetFootOnce();
      el.__lens = { an: an, ref: ref, st: st, iv2: iv2Res, iv3: iv3, analyse: analyse, presetTrain: presetTrain, MODELS: MODELS, opts: opts, geo: function () { return geo; } };
    }
    var footTypeset = false;
    function typesetFootOnce() { if (!footTypeset) { footTypeset = true; ts(foot); } }
    function announce() {
      var o = PRESETS.filter(function (x) { return x.id === st.preset; })[0];
      live.textContent = (o ? o.label : 'Custom') + ': ' + gstr(an.total) + ' GiB estimated, activations ' + gstr(an.act) + ' GiB, cone ' + coneText() + '.';
    }

    /* ---------- mount ---------- */
    applyPreset('lora-qv');
    syncControls();
    select(selId, true);
    drawIV2();
    render();
    ts(stage);
    if ('ResizeObserver' in window) {
      var lastW = 0;
      new ResizeObserver(A.debounce(function () {
        var w = stripBox.clientWidth;
        if (Math.abs(w - lastW) < 2) return;
        lastW = w; drawStrip(); drawSpark(iv2Res[iv2Var]);
      }, 80)).observe(stage);
    } else window.addEventListener('resize', A.debounce(function () { drawStrip(); drawSpark(iv2Res[iv2Var]); }, 120));
    A.onTheme(function () { drawStrip(); drawSpark(iv2Res[iv2Var]); });
  });
})();
