/* Fig. "rosetta" (Figure 19) — Rosetta Stone: a dictionary for two communities (Exposé IX).

   Data: ATLAS_DATA.rosetta (= theory/rosetta.json), one record per entry with three ASCII fields
   {peft, category, explanation}. Nothing here is invented: every count on screen (entries, entries per exposé,
   distinct results cited, referee statuses, search hits) is computed from that array and from
   ATLAS_DATA.propositions at mount time or on every filter change. Thirteen first-round glosses claimed more than the
   final statements they cite; FIX (below) shows the narrowed wording until rosetta.json is updated to match it.

   Typesetting. The ASCII fields carry mathematics ("theta0", "M_{<=r}", "Diag_m . (W0 + M_{<=r})"). TEX below holds a
   hand-typeset HTML + inline-TeX version of each field that needs it, keyed by the EXACT raw string (after FIX): if
   rosetta.json is edited later, the lookup misses and the raw text is shown instead of a stale typeset version
   (a console note says which field fell back).

   Cross-references. Any "Thm III.10", "Prop II.12", "Cor III.11", "Lemma VI.2", "Obs I.4", "Def I.5", "Remark I.3",
   and lists or ranges such as "Thms II.10-II.11", become links to #thm-III-10 etc. (id = "thm-" + number with dots
   replaced by hyphens). The exposé of a result is the Roman numeral of its number; an entry belongs to every
   exposé whose results it cites. "Conj A" is a conjecture, not a numbered result, and is left unlinked. */
(function () {
  'use strict';
  var FIG = 'rosetta';

  /* exposé short names (theory/framework.md headings) */
  var PARTS = [
    { key: 'I', name: 'Pointed slice' },
    { key: 'II', name: 'Image germ' },
    { key: 'III', name: 'Fibres & dynamics' },
    { key: 'IV', name: 'The lens' },
    { key: 'V', name: 'Base change' },
    { key: 'VI', name: 'Extensions & merging' }
  ];

  /* Referee-safe wording. Each key is an EXACT raw string of theory/rosetta.json whose claim is stronger than the
     final statement it cites (theory/round2_verdicts.json, and the statement on the page); the value is the wording
     shown instead, in the same ASCII convention, with its typeset form in TEX. Once rosetta.json carries the new
     strings these keys stop matching and the map is inert. */
  var FIX = {
    "For weight-space methods merging is definitional; for extensions zero inference overhead is a lifting problem, decided box-locally by rewriting under side conditions such as position-uniformity, no weight sharing and bias slots (Thm VI.3).":
      "For weight-space methods merging is definitional; for extensions zero inference overhead is a lifting problem, and rewriting under side conditions such as position-uniformity, no weight sharing and bias slots gives sufficient conditions box by box (Thm VI.3).",
    "LoRA wastes r^2 parameters; LoHa wastes r1^2 + r2^2 + m + n - 1 (Prop II.12).":
      "LoRA wastes r^2 parameters; LoHa wastes at least r1^2 + r2^2 + m + n - 1 (Prop II.12).",
    "With equal square splits, unconstrained Kronecker sums are incomparable with LoRA: a generic rank-one update has maximal Kronecker rank. LoKr with a low inner rank can sit inside a LoRA image; Compacter's PHM weights live in a nonlinear adapter (Thm II.11).":
      "With equal square splits of an N x N weight, sums of fewer than N Kronecker products and LoRA of rank below N are incomparable, since a generic rank-one update has maximal Kronecker rank and a Kronecker product of full-rank factors has full rank. LoKr with a low inner rank can sit inside a LoRA image; Compacter's PHM weights live in a nonlinear adapter (Thm II.11).",
    "FourierFT (up to sampled conjugate pairs), WaveFT, C3A, LoRA-XS, SVFT and SHiRA are coordinate masks in transported frames; for LoRA-XS, SVFT and HiRA the frame depends on theta0 (Prop II.13).":
      "FourierFT (when no conjugate frequency pair is sampled), WaveFT, C3A, LoRA-XS, SVFT and SHiRA are coordinate masks in transported frames, and HiRA is LoRA in a transported frame; for LoRA-XS, SVFT and HiRA the frame depends on theta0 (Prop II.13).",
    "Exactly orthogonal methods (OFT with an exact Cayley map, GOFT, HRA) can never change the spectrum or neuron angles, however often they are merged; HF's default Cayley-Neumann OFT is only approximately orthogonal (Cor II.6).":
      "Exactly orthogonal input-side methods (OFT with an exact Cayley map, GOFT without its input scale, HRA) can never change the spectrum or neuron angles, however often they are merged; HF's default Cayley-Neumann OFT is only approximately orthogonal (Cor II.6).",
    "Cayley-parametrised OFT is strictly smaller; HRA's paired identity init is an apex: the germ is the skew-determinantal cone (Thm II.7).":
      "LoRA meets exact-Cayley OFT in a strictly smaller set. At HRA's paired identity init the image germ is the skew-determinantal cone, so that init is an apex (Thm II.7).",
    "Descent depends on which side of each box a gauge acts on: LoRA and (IA)^3 (HF defaults) provably descend through residual-stream rotations; block OFT, LoHa, HiRA and masks fail the sufficient condition (Obs II.15, Conj A).":
      "Descent depends on which side of each box a gauge acts on: LoRA and (IA)^3 (HF defaults) provably descend, as reachable sets, through SliceGPT-style rotations of a gain-fused pre-norm residual stream; block OFT, LoHa, HiRA and masks fail the sufficient condition (Obs II.15, Conj A).",
    "Conserved under gradient flow for any cotangent signal of the form Lambda D rho^T gamma (DoRA's detached norm, clipping), after Zhao et al. 2023 (Thm III.5, Prop III.7).":
      "Conserved under gradient flow with constant rates for any cotangent signal of the form Lambda D rho^T gamma (DoRA's detached norm, clipping), after Zhao et al. 2023 (Thm III.5, Prop III.7).",
    "Only undamped scaled GD is GL_r-equivariant at the parameter level; damped variants and LoRA-Pro's step are O(r)-equivariant, LoRA-RITE is invariant at the level of W, and Adam sees only B_r (Prop III.4, Thm III.12).":
      "On the full-rank locus, undamped scaled GD and LoRA-RITE are GL_r-equivariant in the parameters; damped variants and LoRA-Pro's step are only O(r)-equivariant, and Adam only B_r-equivariant (Prop III.4, Thm III.12).",
    "GaLore = one-sided LoRA with a moving frame, with optimizer state carried across re-basings (not transported) (Prop IV.2, credited to Torroba-Hennigen et al. 2025).":
      "Without weight decay or full-gradient clipping, GaLore = one-sided LoRA with a moving frame, with optimizer state carried across re-basings (not transported) (Prop IV.2, credited to Torroba-Hennigen et al. 2025).",
    "Every practical cotangent method has a theta-independent anchor within its period, hence affine leaves, and is a rebasing scheme of a linear forward method (Prop IV.2); Frobenius integrability matters only for theta-dependent anchors such as full-batch GaLore with T=1 (Prop IV.3).":
      "GaLore, LISA and MeZO keep a theta-independent anchor within each period, hence affine leaves, and under the hypotheses of Prop IV.2 are rebasing schemes of linear forward methods; Frobenius integrability matters only for theta-dependent anchors such as full-batch GaLore with T=1 (Prop IV.3).",
    "ReLoRA reaches exactly M_{<=Kr}; exact-Cayley block OFT is stuck; iterated BOFT reaches SO(n) only at full butterfly depth; re-HRA reaches (W0 + M_{<=Kr}) cap W0 SO(n) (Thm V.3).":
      "ReLoRA reaches exactly W0 + M_{<=Kr}; exact-Cayley block OFT is stuck; iterated BOFT reaches the orbit W0 SO(n), up to output scaling, only at full butterfly depth; re-HRA reaches (W0 + M_{<=Kr}) cap W0 SO(n) (Thm V.3).",
    "Prefix = gated parallel adapter (attention = monoid homomorphism followed by a perspective map); content-attention ratios are rigid only within the layer where the prefix acts; prompt tuning maps into deep prefix tuning in causal decoders (Thm VI.5).":
      "Prefix = gated parallel adapter (attention = monoid homomorphism followed by a perspective map); content-attention ratios are rigid only within the layer where the prefix acts; prompt tuning maps into deep prefix tuning in causal decoders with matching position offsets (Thm VI.5)."
  };

  /* raw ASCII field -> typeset HTML with inline TeX \( … \). Generated from theory/rosetta.json, typeset by hand. */
  var TEX = /*TEX-BEGIN*/{
    "base point theta0: 1 -> Theta; a pointed 1-cell (Theta, theta0, f) of Para(Diff)":
      "base point \\(\\theta_0:1\\to\\Theta\\); a pointed 1-cell \\((\\Theta,\\theta_0,f)\\) of \\(\\mathbf{Para}(\\mathbf{Diff})\\)",
    "object of the pointed slice Meth(Theta,theta0) = Diff_*/(Theta,theta0), equivalently the opposite of the pointed coslice of reparametrisation 2-cells under F":
      "object of the pointed slice \\(\\mathsf{Meth}(\\Theta,\\theta_0)=\\mathbf{Diff}_*/(\\Theta,\\theta_0)\\), equivalently the opposite of the pointed coslice of reparametrisation 2-cells under \\(F\\)",
    "A method is a pointed map rho:(Q,q0)->(Theta,theta0). The slice depends only on (Theta,theta0); Para is bookkeeping for weight-space methods (Remark I.3).":
      "A method is a pointed map \\(\\rho:(Q,q_0)\\to(\\Theta,\\theta_0)\\). The slice depends only on \\((\\Theta,\\theta_0)\\); \\(\\mathbf{Para}\\) is bookkeeping for weight-space methods (Remark I.3).",
    "\"method N can do whatever M can\"":
      "“method \\(N\\) can do whatever \\(M\\) can”",
    "morphism (simulation) h: M -> N with rho_N o h = rho_M":
      "morphism (simulation) \\(h:M\\to N\\) with \\(\\rho_N\\circ h=\\rho_M\\)",
    "Every weight-space method lies in the interval [frozen, full FT]; adapters, prefixes and ReFT are extensions, not objects of Meth (Obs I.4).":
      "Every weight-space method lies in the interval [frozen, full FT]; adapters, prefixes and ReFT are extensions, not objects of \\(\\mathsf{Meth}\\) (Obs I.4).",
    "the canonical morphism to the terminal object (it is rho itself); for extensions, a lift along the realisation Phi_f":
      "the canonical morphism to the terminal object (it is \\(\\rho\\) itself); for extensions, a lift along the realisation \\(\\Phi_f\\)",
    "image germ of rho at theta0; its dimension d(M) = generic rank of d rho":
      "image germ of \\(\\rho\\) at \\(\\theta_0\\); its dimension \\(d(M)\\) = generic rank of \\(d\\rho\\)",
    "first-order image S1 = d rho_{q0}(T Q), a functorial isomorphism invariant":
      "first-order image \\(S_1=d\\rho_{q_0}(T_{q_0}Q)\\), a functorial isomorphism invariant",
    "The lazy kernel has range J_f(S1); S1 separates zero-init pointings (T_A vs T_B, the row space of A0) that the image cannot, while the image already separates split inits (Obs I.6).":
      "The lazy kernel has range \\(J_f(S_1)\\); \\(S_1\\) separates zero-init pointings (\\(\\mathrm T_A\\) vs \\(\\mathrm T_B\\), the row space of \\(A_0\\)) that the image cannot, while the image already separates split inits (Obs I.6).",
    "apex (singular vertex) vs smooth point of the image; Type I (new q0 in the apex fibre) vs Type II (new rho by splitting)":
      "apex (singular vertex) vs smooth point of the image; Type I (new \\(q_0\\) in the apex fibre) vs Type II (new \\(\\rho\\) by splitting)",
    "Zero-init loses r(n-r) first-order directions; split inits are different objects, incomparable with LoRA (Thm II.2, Prop III.8).":
      "Zero-init loses \\(r(n-r)\\) first-order directions; split inits are different objects, incomparable with LoRA (Thm II.2, Prop III.8).",
    "points of three distinguished strata of generic pointings, classified as a set by Gr(r,n) disjoint-union Gr(r,m) disjoint-union M_r":
      "points of three distinguished strata of generic pointings, classified as a set by \\(\\mathrm{Gr}(r,n)\\sqcup\\mathrm{Gr}(r,m)\\sqcup\\mathcal M_r\\)",
    "Vanilla, EVA, Init[B], MiCA, PiSSA, MiLoRA, OLoRA, CorDA, LoRA-GA and Astra are points; HF's 'orthogonal' init lies outside the three strata, and LoftQ/QLoRA are defect pointings. The optimizer sees a finer quotient (Thm II.2, Cor III.13).":
      "Vanilla, EVA, Init[B], MiCA, PiSSA, MiLoRA, OLoRA, CorDA, LoRA-GA and Astra are points; HF’s ‘orthogonal’ init lies outside the three strata, and LoftQ/QLoRA are defect pointings. The optimizer sees a finer quotient (Thm II.2, Cor III.13).",
    "no init buys rank above 2r":
      "no init buys rank above \\(2r\\)",
    "union over pointings of theta0 + (C - C) = theta0 + M_{<=2r}":
      "union over pointings: \\(\\bigcup_{q}\\operatorname{Im}M_q=\\theta_0+(C-C)=\\theta_0+\\mathcal M_{\\le 2r}\\)",
    "An exact initialisation picks a translate of the shape; from any starting point the update has rank at most 2r. This is the known rank-2r PiSSA-to-LoRA conversion (Thm II.1).":
      "An exact initialisation picks a translate of the shape; from any starting point the update has rank at most \\(2r\\). This is the known <span class=\"ro-nw\">rank-\\(2r\\)</span> PiSSA-to-LoRA conversion (Thm II.1).",
    "gauge group = automorphisms of Q over Theta; generic fibre":
      "gauge group = automorphisms of \\(Q\\) over \\(\\Theta\\); generic fibre",
    "tensor networks whose min-cut avoids the bond; transport of structure along GL(Theta)":
      "tensor networks whose min-cut avoids the bond; transport of structure along \\(GL(\\Theta)\\)",
    "High rank is not GL(Theta)-invariant (though rank is GL_m x GL_n-invariant) and need not mean large volume; a large min-cut is necessary, not sufficient (Thms II.10-II.11, Prop II.13).":
      "High rank is not <span class=\"ro-nw\">\\(GL(\\Theta)\\)-invariant</span> (though rank is <span class=\"ro-nw\">\\(GL_m\\times GL_n\\)-invariant</span>) and need not mean large volume; a large min-cut is necessary, not sufficient (Thms II.10–II.11, Prop II.13).",
    "LoRA across a different cut of the same 4-leg tensor (Van Loan-Pitsianis rearrangement, a coordinate permutation)":
      "LoRA across a different cut of the same 4-leg tensor (Van Loan–Pitsianis rearrangement, a coordinate permutation)",
    "\"new method = old method\"":
      "“new method = old method”",
    "transport T_* along T in GL(Theta): an automorphism of the category, not an arrow inside it":
      "transport \\(T_*\\) along \\(T\\in GL(\\Theta)\\): an automorphism of the category, not an arrow inside it",
    "the image lies in a fibre of the complete invariant W -> W W^T (first fundamental theorem for O(n))":
      "the image lies in a fibre of the complete invariant \\(W\\mapsto WW^\\top\\) (first fundamental theorem for \\(O(n)\\))",
    "image-level meet with the full orthogonal orbit: (W0 + M_{<=r}) cap W0 SO(n), not a product of methods":
      "image-level meet with the full orthogonal orbit: \\((W_0+\\mathcal M_{\\le r})\\cap W_0\\,SO(n)\\), not a product of methods",
    "categorical product FA(A0) x FB(B0) of the two one-sided LoRAs":
      "categorical product \\(\\mathrm{FA}(A_0)\\times\\mathrm{FB}(B_0)\\) of the two one-sided LoRAs",
    "DoRA's magnitude/direction decoupling":
      "DoRA’s magnitude/direction decoupling",
    "diagonal saturation Diag_m . (W0 + M_{<=r}), the image of LoRA followed by a trainable output scale":
      "diagonal saturation \\(\\mathrm{Diag}_m\\cdot(W_0+\\mathcal M_{\\le r})\\), the image of LoRA followed by a trainable output scale",
    "(IA)^3, SSF, LN tuning, RoAd, HiRA":
      "(IA)<sup>3</sup>, SSF, LN tuning, RoAd, HiRA",
    "rungs of the Hadamard-torus ladder W0 (.) (J + Z); RoAd_1 is the complex torus (C^x)^{m/2}, RoAd_2/4 are block-GL_2":
      "rungs of the Hadamard-torus ladder \\(W_0\\odot(J+Z)\\); \\(\\mathrm{RoAd}_1\\) is the complex torus \\((\\mathbb C^\\times)^{m/2}\\), \\(\\mathrm{RoAd}_2\\) and \\(\\mathrm{RoAd}_4\\) are <span class=\"ro-nw\">block-\\(GL_2\\)</span>",
    "Zeros of W0 stay zero; covariance Mon x GL on output-scaled boxes, GL x Mon on down_proj, never GL x GL (Prop II.9).":
      "Zeros of \\(W_0\\) stay zero; covariance \\(\\mathrm{Mon}_m\\times GL_n\\) on output-scaled boxes, \\(GL_m\\times\\mathrm{Mon}_n\\) on <span class=\"ro-code\">down_proj</span>, never \\(GL_m\\times GL_n\\) (Prop II.9).",
    "\"the geometry a method lives in\"":
      "“the geometry a method lives in”",
    "covariance group G_M (Klein): the largest (g,h) with Im M(g W0 h^{-1}) = g Im M(W0) h^{-1}":
      "covariance group \\(\\mathcal G_M\\) (Klein): the largest \\((g,h)\\) with \\(\\operatorname{Im}M(gW_0h^{-1})=g\\,\\operatorname{Im}M(W_0)\\,h^{-1}\\)",
    "induced cometric K = D rho Lambda D rho^T (pulled-back gradient of the lens)":
      "induced cometric \\(K=D\\rho\\,\\Lambda\\,D\\rho^\\top\\) (pulled-back gradient of the lens)",
    "Two isomorphic objects can train differently; gradient descent is natural exactly where cometrics agree, which for LoRA's linear gauge means O(r) (Thm III.3).":
      "Two isomorphic objects can train differently; gradient descent is natural exactly where cometrics agree, which for LoRA’s linear gauge means \\(O(r)\\) (Thm III.3).",
    "balancedness B^T B = A A^T":
      "balancedness \\(B^\\top B=AA^\\top\\)",
    "Noether charge of the non-compact half Sym_r of gl_r; the Kempf-Ness slice of the gauge action":
      "Noether charge of the non-compact half \\(\\mathrm{Sym}_r\\) of \\(\\mathfrak{gl}_r\\); the Kempf–Ness slice of the gauge action",
    "LoRA+ ratio, alpha, init scale, rsLoRA":
      "LoRA+ ratio, \\(\\alpha\\), init scale, rsLoRA",
    "Under Adam with eps=0, no weight decay, no clipping and a shared schedule, LoRA+ with ratio lambda is exactly LoRA with alpha times lambda (the LoRA case is Schulman et al. 2025); rsLoRA is isomorphic to LoRA (Thm III.10, Cor III.11).":
      "Under Adam with \\(\\varepsilon=0\\), no weight decay, no clipping and a shared schedule, LoRA+ with ratio \\(\\lambda\\) is exactly LoRA with \\(\\alpha\\) times \\(\\lambda\\) (the LoRA case is Schulman et al. 2025); rsLoRA is isomorphic to LoRA (Thm III.10, Cor III.11).",
    "update rule equivariant under the whole gauge GL_r, so it descends to the quotient M_r":
      "update rule equivariant under the whole gauge \\(GL_r\\), so it descends to the quotient \\(\\mathcal M_r\\)",
    "An upper bound: LST avoids the backbone's backward pass; LoRA-FA stores A0 x; LISA's cone is the whole network because its embedding is always trained; MeZO has no reverse pass (Obs IV.4).":
      "An upper bound: LST avoids the backbone’s backward pass; LoRA-FA stores \\(A_0x\\); LISA’s cone is the whole network because its embedding is always trained; MeZO has no reverse pass (Obs IV.4).",
    "Split-then-quantise, quantise-then-fit (LoftQ's default) and quantise-split-requantise leave three different defects (Prop V.2).":
      "Split-then-quantise, quantise-then-fit (LoftQ’s default) and quantise-split-requantise leave three different defects (Prop V.2).",
    "GL(Theta)-natural vs coordinatewise operations (TIES: homotheties times signed permutations; DARE: the monomial group)":
      "<span class=\"ro-nw\">\\(GL(\\Theta)\\)-natural</span> vs coordinatewise operations (TIES: homotheties times signed permutations; DARE: the monomial group)",
    "a rule on factors that is not invariant under prod GL_r, hence not a function of the adapters":
      "a rule on factors that is not invariant under \\(\\prod GL_r\\), hence not a function of the adapters",
    "Post-hoc LoraHub (and HF's 'linear' combination) needs a joint gauge fixing that produces a shared factor (KnOTS) or a shared frozen factor (FFA-LoRA); Poly/MHR are trained jointly and have only the diagonal GL_r symmetry (Prop V.4).":
      "Post-hoc LoraHub (and HF’s ‘linear’ combination) needs a joint gauge fixing that produces a shared factor (KnOTS) or a shared frozen factor (FFA-LoRA); Poly/MHR are trained jointly and have only the diagonal \\(GL_r\\) symmetry (Prop V.4).",
    "LoReFT is LoRA on an inserted affine identity box (bias confined to col B), applied at selected positions; pyreft's default init is not neutral (Prop VI.4).":
      "LoReFT is LoRA on an inserted affine identity box (bias confined to \\(\\mathrm{col}\\,B\\)), applied at selected positions; pyreft’s default init is not neutral (Prop VI.4).",
    "automorphism group of the elementwise activation (the sigma-spider)":
      "automorphism group of the elementwise activation (the <span class=\"ro-nw\">\\(\\sigma\\)-spider</span>)",
    "RoSA, GraLoRA, HF 'cat' merges, ReLoRA stages":
      "RoSA, GraLoRA, HF ‘cat’ merges, ReLoRA stages",
    "words in the convolution monoidal product (+) of additive methods (Minkowski sum of images)":
      "words in the convolution monoidal product \\(\\oplus\\) of additive methods (Minkowski sum of images)",
    "(+) is not the categorical product, whose image is the intersection when it exists. LoRA-soup CAT is a (+) of frozen line methods (a span), and UniPELT/MAM are not (+)-words (Obs I.7).":
      "\\(\\oplus\\) is not the categorical product, whose image is the intersection when it exists. LoRA-soup CAT is a \\(\\oplus\\) of frozen line methods (a span), and UniPELT/MAM are not <span class=\"ro-nw\">\\(\\oplus\\)-words</span> (Obs I.7).",
    "target_modules / applying a method to every layer":
      "<span class=\"ro-code\">target_modules</span> / applying a method to every layer",
    "assigning a 2-cell to each generator box and composing horizontally; T-points for routing, tasks and tenants":
      "assigning a 2-cell to each generator box and composing horizontally; <span class=\"ro-nw\">\\(T\\)-points</span> for routing, tasks and tenants",
    "LoRA wastes r^2 parameters; LoHa wastes at least r1^2 + r2^2 + m + n - 1 (Prop II.12).":
      "LoRA wastes \\(r^2\\) parameters; LoHa wastes at least \\(r_1^2+r_2^2+m+n-1\\) (Prop II.12).",
    "With equal square splits of an N x N weight, sums of fewer than N Kronecker products and LoRA of rank below N are incomparable, since a generic rank-one update has maximal Kronecker rank and a Kronecker product of full-rank factors has full rank. LoKr with a low inner rank can sit inside a LoRA image; Compacter's PHM weights live in a nonlinear adapter (Thm II.11).":
      "With equal square splits of an \\(N\\times N\\) weight, sums of fewer than \\(N\\) Kronecker products and LoRA of rank below \\(N\\) are incomparable, since a generic rank-one update has maximal Kronecker rank and a Kronecker product of full-rank factors has full rank. LoKr with a low inner rank can sit inside a LoRA image; Compacter’s PHM weights live in a nonlinear adapter (Thm II.11).",
    "FourierFT (when no conjugate frequency pair is sampled), WaveFT, C3A, LoRA-XS, SVFT and SHiRA are coordinate masks in transported frames, and HiRA is LoRA in a transported frame; for LoRA-XS, SVFT and HiRA the frame depends on theta0 (Prop II.13).":
      "FourierFT (when no conjugate frequency pair is sampled), WaveFT, C3A, LoRA-XS, SVFT and SHiRA are coordinate masks in transported frames, and HiRA is LoRA in a transported frame; for LoRA-XS, SVFT and HiRA the frame depends on \\(\\theta_0\\) (Prop II.13).",
    "Exactly orthogonal input-side methods (OFT with an exact Cayley map, GOFT without its input scale, HRA) can never change the spectrum or neuron angles, however often they are merged; HF's default Cayley-Neumann OFT is only approximately orthogonal (Cor II.6).":
      "Exactly orthogonal input-side methods (OFT with an exact Cayley map, GOFT without its input scale, HRA) can never change the spectrum or neuron angles, however often they are merged; HF’s default Cayley–Neumann OFT is only approximately orthogonal (Cor II.6).",
    "LoRA meets exact-Cayley OFT in a strictly smaller set. At HRA's paired identity init the image germ is the skew-determinantal cone, so that init is an apex (Thm II.7).":
      "LoRA meets exact-Cayley OFT in a strictly smaller set. At HRA’s paired identity init the image germ is the skew-determinantal cone, so that init is an apex (Thm II.7).",
    "Descent depends on which side of each box a gauge acts on: LoRA and (IA)^3 (HF defaults) provably descend, as reachable sets, through SliceGPT-style rotations of a gain-fused pre-norm residual stream; block OFT, LoHa, HiRA and masks fail the sufficient condition (Obs II.15, Conj A).":
      "Descent depends on which side of each box a gauge acts on: LoRA and (IA)<sup>3</sup> (HF defaults) provably descend, as reachable sets, through SliceGPT-style rotations of a gain-fused pre-norm residual stream; block OFT, LoHa, HiRA and masks fail the sufficient condition (Obs II.15, Conj A).",
    "Conserved under gradient flow with constant rates for any cotangent signal of the form Lambda D rho^T gamma (DoRA's detached norm, clipping), after Zhao et al. 2023 (Thm III.5, Prop III.7).":
      "Conserved under gradient flow with constant rates for any cotangent signal of the form \\(\\Lambda\\,D\\rho^\\top\\gamma\\) (DoRA’s detached norm, clipping), after Zhao et al. 2023 (Thm III.5, Prop III.7).",
    "On the full-rank locus, undamped scaled GD and LoRA-RITE are GL_r-equivariant in the parameters; damped variants and LoRA-Pro's step are only O(r)-equivariant, and Adam only B_r-equivariant (Prop III.4, Thm III.12).":
      "On the full-rank locus, undamped scaled GD and LoRA-RITE are <span class=\"ro-nw\">\\(GL_r\\)-equivariant</span> in the parameters; damped variants and LoRA-Pro’s step are only <span class=\"ro-nw\">\\(O(r)\\)-equivariant</span>, and Adam only <span class=\"ro-nw\">\\(B_r\\)-equivariant</span> (Prop III.4, Thm III.12).",
    "GaLore, LISA and MeZO keep a theta-independent anchor within each period, hence affine leaves, and under the hypotheses of Prop IV.2 are rebasing schemes of linear forward methods; Frobenius integrability matters only for theta-dependent anchors such as full-batch GaLore with T=1 (Prop IV.3).":
      "GaLore, LISA and MeZO keep a <span class=\"ro-nw\">\\(\\theta\\)-independent</span> anchor within each period, hence affine leaves, and under the hypotheses of Prop IV.2 are rebasing schemes of linear forward methods; Frobenius integrability matters only for <span class=\"ro-nw\">\\(\\theta\\)-dependent</span> anchors such as full-batch GaLore with \\(T=1\\) (Prop IV.3).",
    "ReLoRA reaches exactly W0 + M_{<=Kr}; exact-Cayley block OFT is stuck; iterated BOFT reaches the orbit W0 SO(n), up to output scaling, only at full butterfly depth; re-HRA reaches (W0 + M_{<=Kr}) cap W0 SO(n) (Thm V.3).":
      "ReLoRA reaches exactly \\(W_0+\\mathcal M_{\\le Kr}\\); exact-Cayley block OFT is stuck; iterated BOFT reaches the orbit \\(W_0\\,SO(n)\\), up to output scaling, only at full butterfly depth; re-HRA reaches \\((W_0+\\mathcal M_{\\le Kr})\\cap W_0\\,SO(n)\\) (Thm V.3)."
  }/*TEX-END*/;

  /* Second review (theory/round2_verdicts.json): results whose repaired statement a referee found a false or
     overreaching clause in, on either lens, and which were narrowed again. The page's referee record
     (#referee-record, "repaired twice") is read first when present; this list is the fallback. */
  var TWICE = ['I.7', 'II.6', 'II.7', 'II.8', 'II.11', 'II.12', 'II.13', 'II.15', 'II.16', 'III.5', 'III.12',
    'IV.3', 'IV.4', 'V.2', 'V.3', 'V.4', 'VI.3', 'VI.4', 'VI.5'];

  /* suggested searches (each is shown only if it matches at least one entry; the count is computed) */
  var TRY = ['gauge', 'apex', 'Adam', 'orthogonal', 'product', 'merge', 'III.10'];

  /* ---------------- references ---------------- */
  var KIND_RE = 'Theorems?|Thms?|Propositions?|Props?|Corollar(?:y|ies)|Cors?|Lemmas?|Observations?|Obs|Definitions?|Defs?|Remarks?';
  var NUM_RE = '[IVX]+\\.\\d+';
  var TAIL_RE = '(?:\\s*[-–]\\s*(?:[IVX]+\\.)?\\d+(?!\\d)(?!\\.\\d)|\\s*(?:,|and|&)\\s*' + NUM_RE + ')*';
  function refRegex() { return new RegExp('\\b(' + KIND_RE + ')(\\.?\\s+)(' + NUM_RE + ')(' + TAIL_RE + ')', 'g'); }
  var TAIL_ITEM = /(\s*(?:[-–]|,|and|&)\s*)((?:[IVX]+\.)?\d+)/g;
  function kindName(abbr) {
    var a = abbr.toLowerCase();
    if (a.indexOf('th') === 0) return 'Theorem';
    if (a.indexOf('prop') === 0) return 'Proposition';
    if (a.indexOf('cor') === 0) return 'Corollary';
    if (a.indexOf('lem') === 0) return 'Lemma';
    if (a.indexOf('obs') === 0) return 'Observation';
    if (a.indexOf('def') === 0) return 'Definition';
    if (a.indexOf('rem') === 0) return 'Remark';
    return abbr;
  }
  function partOf(num) { return num.split('.')[0]; }
  function anchorOf(num) { return 'thm-' + num.replace(/\./g, '-'); }
  /* every numbered result a text cites, ranges expanded ("II.10-II.12" -> II.10, II.11, II.12) */
  function parseRefs(text) {
    var out = [], re = refRegex(), m;
    while ((m = re.exec(text))) {
      var kn = kindName(m[1]), part = partOf(m[3]), prev = +m[3].split('.')[1];
      out.push({ num: m[3], kind: kn });
      var t, ti = new RegExp(TAIL_ITEM.source, 'g');
      while ((t = ti.exec(m[4]))) {
        var full = t[2].indexOf('.') >= 0 ? t[2] : part + '.' + t[2];
        var p2 = partOf(full), n2 = +full.split('.')[1];
        if (/[-–]/.test(t[1]) && p2 === part) for (var k = prev + 1; k < n2; k++) out.push({ num: part + '.' + k, kind: kn });
        out.push({ num: full, kind: kn });
        part = p2; prev = n2;
      }
    }
    return out;
  }
  function refLink(num, kn, text) {
    return '<a class="ro-ref" href="#' + anchorOf(num) + '" data-num="' + num + '" data-kind="' + kn + '">' + text + '</a>';
  }
  /* link the references inside one run of plain text (no tags, no math) */
  function linkify(s) {
    return s.replace(refRegex(), function (all, kind, sp, first, tail) {
      var kn = kindName(kind);
      if (!tail) return refLink(first, kn, kind + sp + first);
      var part = partOf(first);
      var out = kind + sp + refLink(first, kn, first);
      out += tail.replace(new RegExp(TAIL_ITEM.source, 'g'), function (x, sep, num) {
        var full = num.indexOf('.') >= 0 ? num : part + '.' + num;
        part = partOf(full);
        return sep.replace('-', '–') + refLink(full, kn, num);
      });
      return out;
    });
  }
  /* link references in an HTML string, leaving tags and \( … \) math untouched */
  function linkifyHTML(html) {
    return html.split(/(\\\([\s\S]*?\\\)|<[^>]*>)/).map(function (seg, i) { return i % 2 ? seg : linkify(seg); }).join('');
  }

  /* ---------------- text helpers ---------------- */
  function smartQuotes(s) {
    return s.replace(/(^|[\s(\[\/])"/g, '$1“').replace(/"/g, '”').replace(/(^|[\s(\[\/])'/g, '$1‘').replace(/'/g, '’');
  }
  var TEXU = {
    '\\theta': 'θ', '\\Theta': 'Θ', '\\rho': 'ρ', '\\alpha': 'α', '\\lambda': 'λ', '\\Lambda': 'Λ', '\\gamma': 'γ',
    '\\sigma': 'σ', '\\varepsilon': 'ε', '\\Phi': 'Φ', '\\top': 'ᵀ', '\\le': '≤', '\\oplus': '⊕', '\\odot': '⊙',
    '\\times': '×', '\\cap': '∩', '\\sqcup': '⊔', '\\bigcup': '⋃', '\\circ': '∘', '\\to': '→', '\\mapsto': '↦',
    '\\in': '∈', '\\prod': '∏', '\\cdot': '·'
  };
  var SUBD = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
  /* small TeX -> Unicode, for tooltips and the search index (never for display in the dictionary itself) */
  function texToText(t) {
    return t
      .replace(/\\(?:mathrm|mathsf|mathbf|mathbb|mathcal|mathfrak|operatorname)\{([^}]*)\}/g, '$1')
      .replace(/\\[,;!]/g, ' ')
      .replace(/\\[a-zA-Z]+/g, function (m) { return TEXU[m] != null ? TEXU[m] : ''; })
      .replace(/\^\{?ᵀ\}?/g, 'ᵀ')
      .replace(/_\{?(\d)\}?/g, function (x, d) { return SUBD[d]; })
      .replace(/_\{([^}]*)\}/g, '$1').replace(/_(\w)/g, '$1')
      .replace(/\^\{([^}]*)\}/g, '^$1')
      .replace(/[{}]/g, '').replace(/\s+/g, ' ').trim();
  }
  function htmlToText(html) {
    return html
      .replace(/\\\(([\s\S]*?)\\\)/g, function (x, t) { return texToText(t); })
      .replace(/<sup>(\d)<\/sup>/g, function (x, d) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]; })
      .replace(/<[^>]*>/g, '')
      .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  }
  /* same-length normalisation used for matching (curly quotes and dashes -> ASCII) */
  function normChars(s) { return s.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[–—]/g, '-'); }
  /* query transliteration: Unicode maths -> the ASCII spelling used in rosetta.json */
  var QMAP = [
    [/θ/g, 'theta'], [/Θ/g, 'theta'], [/ρ/g, 'rho'], [/[λΛ]/g, 'lambda'], [/α/g, 'alpha'], [/γ/g, 'gamma'], [/σ/g, 'sigma'],
    [/[εϵ]/g, 'eps'], [/⊕/g, '(+)'], [/⊙/g, '(.)'], [/×/g, ' x '], [/∩/g, ' cap '], [/≤/g, '<='], [/ᵀ/g, '^t'],
    [/⊔/g, 'disjoint-union'], [/∘/g, ' o '], [/[₀-₉]/g, function (c) { return String(c.charCodeAt(0) - 0x2080); }]
  ];
  function queryTerms(q) {
    var s = normChars(String(q || ''));
    QMAP.forEach(function (p) { s = s.replace(p[0], p[1]); });
    s = s.toLowerCase();
    var terms = [], m, re = /"([^"]+)"|(\S+)/g;
    while ((m = re.exec(s))) { var t = (m[1] || m[2]).trim(); if (t && terms.indexOf(t) < 0) terms.push(t); }
    return terms;
  }
  var ROMAN_NUM = /^[ivx]+\.\d+$/;
  /* is the occurrence of term at p a word start? Camel-case starts count (ReLoRA, QLoRA), mid-word ones do not
     (so "adam" does not hit "Hadamard"); result numbers must match whole ("I.4" is not inside "VI.4" or "I.40"). */
  function startOK(o, p, term) {
    var prev = p > 0 ? o.charAt(p - 1) : '', cur = o.charAt(p), next = o.charAt(p + 1);
    if (ROMAN_NUM.test(term)) return !/[A-Za-z0-9.]/.test(prev) && !/\d/.test(o.charAt(p + term.length));
    if (!/[a-z]/.test(term.charAt(0)) || !prev) return true;
    if (!/[A-Za-z0-9]/.test(prev)) return true;
    if (/[A-Z]/.test(cur) && /[a-z0-9]/.test(prev)) return true;
    if (/[A-Z]/.test(cur) && /[A-Z]/.test(prev) && /[a-z]/.test(next)) return true;
    return false;
  }
  function occurrences(orig, term) {
    var low = orig.toLowerCase(), out = [], p = low.indexOf(term);
    while (p >= 0) { if (startOK(orig, p, term)) out.push([p, p + term.length]); p = low.indexOf(term, p + 1); }
    return out;
  }
  function pad2(n) { return n < 10 ? '0' + n : String(n); }

  /* ---------------- CSS ---------------- */
  function injectCSS() {
    if (document.getElementById('css-' + FIG)) return;
    var F = '[data-figure="' + FIG + '"] ';
    var css = [
      F + '.ro-stage{padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.ro-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.2rem 1rem}',
      F + '.ro-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.5rem,1.1rem + 1.6vw,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.ro-title .sub{font-family:var(--f-body);font-style:italic;font-weight:var(--w-body);font-size:.6em;letter-spacing:0;color:var(--tide);margin-left:.4em;white-space:nowrap}',
      F + '.ro-title .nm{white-space:nowrap}',
      F + '.ro-narrow .ro-title .sub{display:block;margin:.25rem 0 0;font-size:.58em}',
      F + '.ro-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.ro-instr{margin:.45rem 0 .95rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:70ch}',
      F + '.ro-controls{display:grid;grid-template-columns:minmax(13rem,1fr) auto;gap:.8rem 1.6rem;align-items:start;border-top:1px solid var(--rule);padding-top:.65rem}',
      F + '.ro-narrow .ro-controls{grid-template-columns:minmax(0,1fr)}',
      F + '.ro-group{display:grid;gap:.35rem;min-width:0}',
      F + '.ro-gl{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.ro-group.full{grid-column:1 / -1}',
      F + '.ro-search{position:relative;display:flex;align-items:center;min-width:0}',
      F + '.ro-search svg{position:absolute;left:.75rem;width:15px;height:15px;fill:none;stroke:var(--ink-2);stroke-width:1.6;pointer-events:none}',
      F + '.ro-search input{width:100%;min-width:0;font-family:var(--f-body);font-size:1rem;line-height:1.3;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:999px;padding:.5rem 2.3rem .5rem 2.2rem;-webkit-appearance:none;appearance:none}',
      F + '.ro-search input::placeholder{color:var(--ink-2);font-style:italic;opacity:1}',
      F + '.ro-search input::-webkit-search-cancel-button{-webkit-appearance:none;display:none}',
      F + '.ro-search input:focus{outline:none;border-color:var(--tide);box-shadow:0 0 0 3px var(--tide-soft)}',
      F + '.ro-clear{position:absolute;right:.3rem;width:1.75rem;height:1.75rem;border-radius:999px;border:0;background:transparent;color:var(--ink-2);font-family:var(--f-ui);font-size:1.1rem;line-height:1;cursor:pointer}',
      F + '.ro-clear:hover{color:var(--ink);background:var(--paper-3)}',
      F + '.ro-clear[hidden]{display:none}',
      F + '.ro-pills{display:flex;flex-wrap:wrap;gap:.3rem}',
      F + '.ro-pill{display:inline-flex;align-items:baseline;gap:.32rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);line-height:1.2;padding:.32rem .62rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap}',
      F + '.ro-pill .pn{font-family:var(--f-display);font-size:1rem;font-weight:var(--w-head);color:var(--ink);line-height:1}',
      F + '.ro-pill .pc{font-family:var(--f-mono);font-weight:400;font-size:.76rem;font-variant-numeric:tabular-nums;color:var(--ink-2)}',
      F + '.ro-pill:hover:not([aria-pressed="true"]){border-color:var(--ink-3);color:var(--ink)}',
      F + '.ro-pill[aria-pressed="true"]{background:var(--tide);border-color:var(--tide);color:var(--paper)}',
      F + '.ro-pill[aria-pressed="true"] .pn,[data-figure="rosetta"] .ro-pill[aria-pressed="true"] .pc{color:var(--paper)}',
      F + '.ro-try{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem .35rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);color:var(--ink-2)}',
      F + '.ro-try .lab{letter-spacing:.06em;text-transform:uppercase;margin-right:.15rem}',
      F + '.ro-chip{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);line-height:1.25;padding:.16rem .52rem;border-radius:999px;border:1px dashed var(--rule);background:transparent;color:var(--ink);cursor:pointer}',
      F + '.ro-chip .c{font-family:var(--f-mono);font-weight:400;font-size:.76rem;font-variant-numeric:tabular-nums;color:var(--ink-2);margin-left:.35em}',
      F + '.ro-chip:hover{border-style:solid;border-color:var(--ochre);color:var(--ink)}',
      F + '.ro-chip[aria-pressed="true"]{border-style:solid;border-color:var(--ochre);background:var(--ochre-soft);color:var(--ink)}',
      F + '.ro-map{margin-top:.9rem;border-top:1px solid var(--rule);padding-top:.55rem}',
      F + '.ro-maphead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.2rem 1rem;margin-bottom:.3rem}',
      F + '.ro-svgwrap{position:relative;width:100%}',
      F + '.ro-svgwrap svg{display:block;width:100%;height:auto;overflow:visible}',
      F + 'svg .rm-band{fill:transparent}',
      F + 'svg .rm-band.sel{fill:var(--tide-soft)}',
      F + 'svg .rm-guide{stroke:var(--rule);stroke-width:1;stroke-dasharray:1 3}',
      F + 'svg .rm-num{font-family:var(--f-display);font-size:14px;font-weight:var(--w-head);fill:var(--ink)}',
      F + 'svg .rm-name{font-family:var(--f-body);font-style:italic;font-size:13px;fill:var(--ink-2)}',
      F + 'svg .rm-cnt{font-family:var(--f-mono);font-size:12px;font-variant-numeric:tabular-nums;fill:var(--ink-2)}',
      F + 'svg .rm-lab{cursor:pointer}',
      F + 'svg .rm-lab:hover .rm-num,[data-figure="rosetta"] svg .rm-lab:hover .rm-name{fill:var(--tide)}',
      F + 'svg .rm-lab.sel .rm-num,[data-figure="rosetta"] svg .rm-lab.sel .rm-name{fill:var(--tide)}',
      F + 'svg .rm-dot{fill:var(--tide)}',
      F + 'svg .rm-dot.hit{fill:var(--ochre)}',
      F + 'svg .rm-dot.dim{fill:var(--ink-3);fill-opacity:.28}',
      F + 'svg .rm-none{fill:none;stroke:var(--ink-2);stroke-width:1}',
      F + 'svg .rm-none.dim{stroke-opacity:.3}',
      F + 'svg .rm-hover{fill:var(--ink);opacity:0;pointer-events:none}',
      F + 'svg .rm-hover.on{opacity:.07}',
      F + 'svg .rm-col{fill:transparent;cursor:pointer}',
      F + 'svg .rm-ax{font-family:var(--f-mono);font-size:11px;font-variant-numeric:tabular-nums;fill:var(--ink-2)}',
      F + 'svg .rm-ax.dim{fill:var(--ink-3);fill-opacity:.45}',
      F + 'svg .rm-axt{font-family:var(--f-ui);font-weight:500;font-size:12px;fill:var(--ink-2)}',
      F + '.ro-legend{display:flex;flex-wrap:wrap;gap:.25rem 1.1rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);line-height:1.4;color:var(--ink-2);margin-top:.4rem}',
      F + '.ro-legend > span{display:inline-flex;align-items:center;gap:.35rem}',
      F + '.ro-legend svg{display:block;overflow:visible}',
      F + '.ro-legend .d{fill:var(--tide)}',
      F + '.ro-legend .d.hit{fill:var(--ochre)}',
      F + '.ro-legend .d.dim{fill:var(--ink-3);fill-opacity:.28}',
      F + '.ro-legend .r{fill:none;stroke:var(--ink-2)}',
      F + '.ro-bar{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.3rem 1rem;margin-top:.9rem;padding-top:.6rem;border-top:1px solid var(--rule)}',
      F + '.ro-count{font-family:var(--f-ui);font-size:var(--fs-sm);color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.ro-count b{font-family:var(--f-mono);font-weight:500;color:var(--ink);font-size:1.15rem;margin-right:.25em}',
      F + '.ro-count .sep{color:var(--ink-3);margin:0 .2em}',
      F + '.ro-count .seg-t{white-space:nowrap}',
      F + '.ro-dir{display:flex;align-items:center;gap:.5rem;flex:none;max-width:100%}',
      F + '.ro-dir .seg{flex-wrap:nowrap;flex:none}',
      F + '.ro-dir .seg button{white-space:nowrap;font-weight:500}',
      F + '.ro-narrow .ro-dir .seg button{padding:.38rem .55rem;letter-spacing:0}',
      F + '.ro-listwrap{position:relative;margin-top:.5rem}',
      F + '.ro-listwrap::after{content:"";position:absolute;left:0;right:0;bottom:0;height:2.4rem;pointer-events:none;background:linear-gradient(to bottom,transparent,var(--paper-2));opacity:1;transition:opacity .2s}',
      F + '.ro-listwrap.at-end::after{opacity:0}',
      F + '.ro-list{max-height:clamp(24rem,68vh,46rem);overflow-y:auto;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);scrollbar-width:thin;scrollbar-color:var(--rule) transparent}',
      F + '.ro-list:focus-visible{outline:2px solid var(--ochre);outline-offset:2px}',
      F + '.ro-cols,[data-figure="rosetta"] .ro-row{display:grid;grid-template-columns:2.7rem minmax(0,.8fr) minmax(0,1.45fr);column-gap:1.5rem}',
      F + '.ct-first .ro-cols,[data-figure="rosetta"] .ct-first .ro-row{grid-template-columns:2.7rem minmax(0,1.45fr) minmax(0,.8fr)}',
      F + '.ro-cols{position:sticky;top:0;z-index:2;background:var(--paper-2);border-bottom:1px solid var(--rule);padding:.5rem .3rem .4rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.ro-cols .c-peft{grid-column:2}',
      F + '.ro-cols .c-cat{grid-column:3}',
      F + '.ct-first .ro-cols .c-peft{grid-column:3;grid-row:1}',
      F + '.ct-first .ro-cols .c-cat{grid-column:2;grid-row:1}',
      F + '.ro-row{row-gap:.4rem;padding:.95rem .3rem .9rem;border-bottom:1px solid var(--rule);scroll-margin-top:2.2rem}',
      F + '.ro-row:last-of-type{border-bottom:0}',
      F + '.ro-row[hidden],[data-figure="rosetta"] .ro-cols[hidden]{display:none}',
      F + '.ro-row.hot{background:color-mix(in srgb,var(--paper-3) 55%,transparent)}',
      F + '.ro-row.target{box-shadow:inset 3px 0 0 var(--ochre)}',
      F + '.ro-row.flash{animation:ro-flash 1.8s ease-out}',
      '@keyframes ro-flash{0%{background:var(--ochre-soft)}100%{background:transparent}}',
      F + '.ro-n{grid-column:1;grid-row:1 / span 2;display:flex;flex-direction:column;gap:.3rem;padding-top:.22rem}',
      F + '.ro-n .no{font-family:var(--f-mono);font-size:.8rem;color:var(--ink-2);font-variant-numeric:tabular-nums;line-height:1}',
      F + '.ro-n .pt{display:flex;flex-direction:column;gap:.12rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.02em;color:var(--tide);line-height:1.2}',
      F + '.ro-narrow .ro-n .pt{flex-direction:row;gap:.45rem}',
      F + '.ro-peft{grid-column:2;grid-row:1;font-family:var(--f-display);font-size:1.2rem;font-weight:var(--w-head);line-height:1.24;color:var(--ink);min-width:0;overflow-wrap:break-word}',
      F + '.ro-cat{grid-column:3;grid-row:1;font-family:var(--f-body);font-size:1.02rem;line-height:1.5;color:var(--ink);min-width:0;overflow-wrap:break-word}',
      F + '.ct-first .ro-peft{grid-column:3}',
      F + '.ct-first .ro-cat{grid-column:2}',
      F + '.ro-expl{grid-column:2 / 4;grid-row:2;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.56;color:var(--ink-2);min-width:0;overflow-wrap:break-word;max-width:78ch}',
      F + '.ro-code{font-family:var(--f-mono);font-size:.86em;letter-spacing:-.01em}',
      F + '.ro-nw{white-space:nowrap}',
      F + '.ro-peft .ro-code{font-size:.78em;font-weight:500}',
      F + '.ro-row mjx-container{color:inherit}',
      F + 'a.ro-ref{font-family:var(--f-ui);font-weight:500;font-size:.9em;letter-spacing:.01em;font-variant-numeric:tabular-nums;color:var(--tide);text-decoration:none;border-bottom:1px solid currentColor;padding-bottom:.02em;white-space:nowrap}',
      F + 'a.ro-ref:hover{background:var(--tide-soft)}',
      F + 'mark.ro-hit{background:var(--ochre-soft);color:inherit;border-radius:2px;box-shadow:inset 0 -1.5px 0 var(--ochre);padding:0 .04em}',
      F + '.ro-empty{padding:1.6rem .4rem;font-family:var(--f-body);font-size:var(--fs-sm);color:var(--ink-2);display:grid;gap:.6rem;justify-items:start}',
      F + '.ro-empty[hidden]{display:none}',
      F + '.ro-empty q{font-style:italic;color:var(--ink)}',
      F + '.ro-foot{margin-top:.8rem;font-family:var(--f-body);font-size:.84rem;line-height:1.55;color:var(--ink-2);max-width:92ch}',
      F + '.ro-narrow .ro-cols{display:none}',
      F + '.ro-narrow .ro-row,[data-figure="rosetta"] .ro-narrow.ct-first .ro-row{grid-template-columns:minmax(0,1fr);row-gap:.45rem;padding:.95rem .15rem}',
      F + '.ro-narrow .ro-n{grid-column:1;grid-row:1;flex-direction:row;align-items:baseline;gap:.6rem;padding-top:0}',
      F + '.ro-narrow .ro-peft{grid-column:1;grid-row:2}',
      F + '.ro-narrow .ro-cat{grid-column:1;grid-row:3}',
      F + '.ro-narrow.ct-first .ro-cat{grid-row:2}',
      F + '.ro-narrow.ct-first .ro-peft{grid-row:3}',
      F + '.ro-narrow .ro-expl{grid-column:1;grid-row:4}',
      F + '.ro-narrow .ro-peft::before,[data-figure="rosetta"] .ro-narrow .ro-cat::before{display:block;font-family:var(--f-ui);font-size:var(--fs-xs);font-weight:500;font-style:normal;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin-bottom:.15rem}',
      F + '.ro-narrow .ro-peft::before{content:"PEFT"}',
      F + '.ro-narrow .ro-cat::before{content:"Category theory · geometry"}',
      F + '.ro-narrow .ro-list{max-height:clamp(24rem,72vh,40rem)}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-' + FIG;
    s.textContent = css;
    document.head.appendChild(s);
  }

  /* ---------------- figure ---------------- */
  Atlas.register(FIG, function (el, A) {
    injectCSS();
    var h = A.h;
    var D = A.data();
    var raw = (D && D.rosetta) || [];
    var props = {};
    ((D && D.propositions) || []).forEach(function (p) { props[p.id] = p; });
    var SVGNS = 'http://www.w3.org/2000/svg';
    function S(tag, attrs) {
      var n = document.createElementNS(SVGNS, tag);
      for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
      return n;
    }

    var stage = h('div', { class: 'stage ro-stage', role: 'group' });
    if (!el.getAttribute('aria-label')) stage.setAttribute('aria-label', 'Rosetta Stone, a searchable dictionary from parameter-efficient fine-tuning terms to their categorical and geometric counterparts, with links to the results each entry cites.');
    el.appendChild(stage);
    if (!raw.length) {
      stage.appendChild(h('p', { class: 'ro-instr', text: 'The dictionary data (ATLAS_DATA.rosetta) did not load.' }));
      return;
    }

    /* ---------- model ---------- */
    function safe(x) { var t = String(x == null ? '' : x); return Object.prototype.hasOwnProperty.call(FIX, t) ? FIX[t] : t; }
    var fellBack = [];
    function rich(rawText, where) {
      var t = TEX[rawText];
      if (t != null) return linkifyHTML(t);
      if (/[\\_^]|<=|theta|rho|Lambda|\(\+\)/.test(rawText)) fellBack.push(where);
      return '<span class="tex2jax_ignore">' + linkify(A.esc(smartQuotes(rawText)).replace(/&quot;|&#39;/g, function (m) { return m === '&quot;' ? '”' : '’'; })) + '</span>';
    }
    var entries = raw.map(function (r, i) {
      var f = { peft: safe(r.peft), category: safe(r.category), explanation: safe(r.explanation) };
      var refs = parseRefs(f.peft + ' \n ' + f.category + ' \n ' + f.explanation);
      var byPart = {}, nums = [];
      refs.forEach(function (x) {
        if (nums.indexOf(x.num) >= 0) return;
        nums.push(x.num);
        var p = partOf(x.num);
        (byPart[p] = byPart[p] || []).push(x.num);
      });
      var html = {
        peft: rich(f.peft, (i + 1) + '.peft'),
        category: rich(f.category, (i + 1) + '.category'),
        explanation: rich(f.explanation, (i + 1) + '.explanation')
      };
      var plain = {};
      ['peft', 'category', 'explanation'].forEach(function (k) {
        plain[k] = TEX[f[k]] != null ? normChars(htmlToText(TEX[f[k]])) : normChars(smartQuotes(f[k]));
      });
      var hay = [f.peft, f.category, f.explanation, plain.peft, plain.category, plain.explanation].join(' \n ');
      return { i: i, n: i + 1, raw: f, html: html, plain: plain, refs: nums, byPart: byPart,
        parts: PARTS.map(function (p) { return p.key; }).filter(function (k) { return byPart[k]; }), hay: hay };
    });
    if (fellBack.length && window.console) console.info('[rosetta] fields shown untypeset (rosetta.json changed?):', fellBack.join(', '));
    var N = entries.length;
    var allNums = [];
    entries.forEach(function (e) { e.refs.forEach(function (n) { if (allNums.indexOf(n) < 0) allNums.push(n); }); });
    var partCount = {};
    PARTS.forEach(function (p) { partCount[p.key] = entries.filter(function (e) { return e.byPart[p.key]; }).length; });
    var nUncited = entries.filter(function (e) { return !e.refs.length; }).length;
    /* referee outcome per result: the page's referee record when present, else propositions.json + TWICE */
    var pageRecord = {};
    try {
      var trs = document.querySelectorAll('#referee-record ~ * tr');
      for (var ti = 0; ti < trs.length; ti++) {
        var ta = trs[ti].querySelector('td a[href^="#thm-"]'), tc = trs[ti].querySelector('td .chip');
        if (ta && tc) pageRecord[ta.getAttribute('href').slice(5).replace(/-/g, '.')] = tc.textContent.trim().toLowerCase();
      }
    } catch (err) {}
    function outcome(num) {
      if (pageRecord[num]) return pageRecord[num];
      var p = props[num];
      if (!p || !p.status) return '';
      return p.status === 'repaired' && TWICE.indexOf(num) >= 0 ? 'repaired twice' : p.status;
    }
    var statusCount = { verified: 0, repaired: 0, twice: 0, other: 0 }, nRecord = 0;
    allNums.forEach(function (n) {
      var o = outcome(n);
      if (!o) return;
      nRecord++;
      if (o === 'verified') statusCount.verified++;
      else if (o.indexOf('repaired') === 0) { statusCount.repaired++; if (o !== 'repaired') statusCount.twice++; }
      else statusCount.other++;
    });
    function matches(e, terms) {
      for (var t = 0; t < terms.length; t++) if (!occurrences(e.hay, terms[t]).length) return false;
      return true;
    }

    /* ---------- state ---------- */
    var st = { q: '', part: 'all', dir: 'ml' };
    var mathReady = false;

    /* ---------- header ---------- */
    stage.appendChild(h('div', { class: 'ro-head' }, [
      h('h3', { class: 'ro-title', html: '<span class="nm">Rosetta Stone</span><span class="sub">a dictionary for two communities</span>' }),
      h('div', { class: 'ro-kicker', text: 'Exposé IX · ' + N + ' entries · ' + allNums.length + ' results' })
    ]));
    stage.appendChild(h('p', { class: 'ro-instr', html: 'Search in either language by method, concept or result number, or pick an exposé, and click a column of the map to jump to its entry. Every result number links to its statement, and hovering one shows its name and referee status.' }));

    /* ---------- controls ---------- */
    var uid = 'ro-' + Math.random().toString(36).slice(2, 7);
    var input = h('input', { type: 'search', id: uid + '-q', placeholder: 'LoRA, gauge, apex, Adam, III.10 …', autocomplete: 'off', spellcheck: 'false', 'aria-describedby': uid + '-count' });
    var clearBtn = h('button', { type: 'button', class: 'ro-clear', 'aria-label': 'Clear search', hidden: 'hidden', text: '×' });
    var searchBox = h('div', { class: 'ro-search' }, [input, clearBtn]);
    searchBox.insertAdjacentHTML('afterbegin', '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="6.8" cy="6.8" r="4.9"/><path d="M10.5 10.5 14.5 14.5" stroke-linecap="round"/></svg>');

    var pillBtns = {};
    var pills = h('div', { class: 'ro-pills', role: 'group', 'aria-label': 'Filter by exposé' });
    function addPill(key, label, count, title) {
      var b = h('button', { type: 'button', class: 'ro-pill', 'aria-pressed': String(st.part === key), title: title, 'aria-label': title });
      b.innerHTML = '<span class="pn">' + label + '</span><span class="pc">' + count + '</span>';
      b.addEventListener('click', function () { setPart(st.part === key && key !== 'all' ? 'all' : key); });
      pills.appendChild(b);
      pillBtns[key] = b;
    }
    addPill('all', 'All', N, 'All ' + N + ' entries');
    PARTS.forEach(function (p) {
      addPill(p.key, p.key, partCount[p.key], 'Exposé ' + p.key + ', ' + p.name + ': ' + partCount[p.key] + ' entries cite a result stated there');
    });

    var tryRow = h('div', { class: 'ro-try' }, [h('span', { class: 'lab', text: 'Try' })]);
    var tryBtns = [];
    TRY.forEach(function (t) {
      var terms = queryTerms(t);
      var c = entries.filter(function (e) { return matches(e, terms); }).length;
      if (!c) return;
      var b = h('button', { type: 'button', class: 'ro-chip', 'aria-pressed': 'false', 'aria-label': 'Search ' + t + ' (' + c + ' ' + (c === 1 ? 'entry' : 'entries') + ')' });
      b.innerHTML = A.esc(t) + '<span class="c">' + c + '</span>';
      b.addEventListener('click', function () { var on = st.q.trim().toLowerCase() === t.toLowerCase(); input.value = on ? '' : t; onQuery(true); });
      b._q = t;
      tryBtns.push(b);
      tryRow.appendChild(b);
    });

    stage.appendChild(h('div', { class: 'ro-controls' }, [
      h('div', { class: 'ro-group' }, [h('label', { class: 'ro-gl', for: uid + '-q', text: 'Search both columns' }), searchBox]),
      h('div', { class: 'ro-group' }, [h('span', { class: 'ro-gl', id: uid + '-pl', text: 'Exposé · entries citing it' }), pills]),
      h('div', { class: 'ro-group full' }, [tryRow])
    ]));
    pills.setAttribute('aria-labelledby', uid + '-pl');

    /* ---------- citation map ---------- */
    var mapBox = h('div', { class: 'ro-map' });
    var mapHead = h('div', { class: 'ro-maphead' }, [
      h('span', { class: 'ro-gl', text: 'Citation map · the exposés each entry cites' }),
      h('span', { class: 'ro-gl', text: 'columns = entries 1–' + N + ' · rows = exposés' })
    ]);
    var svgWrap = h('div', { class: 'ro-svgwrap' });
    var svg = S('svg', { role: 'img' });
    svg.setAttribute('aria-label', 'Citation map: ' + PARTS.map(function (p) { return 'Exposé ' + p.key + ' is cited by ' + partCount[p.key] + ' entries'; }).join('; ') + (nUncited ? '; ' + nUncited + ' entry cites no numbered result.' : '.'));
    svgWrap.appendChild(svg);
    var legend = h('div', { class: 'ro-legend', 'aria-hidden': 'true' });
    function dotSVG(r, cls) { return '<svg width="' + (2 * r + 2) + '" height="12" viewBox="' + (-r - 1) + ' -6 ' + (2 * r + 2) + ' 12"><circle class="d ' + (cls || '') + '" r="' + r + '"/></svg>'; }
    mapBox.appendChild(mapHead);
    mapBox.appendChild(svgWrap);
    mapBox.appendChild(legend);
    stage.appendChild(mapBox);

    /* ---------- readout + direction ---------- */
    var countEl = h('div', { class: 'ro-count', id: uid + '-count', 'aria-live': 'polite' });
    var dirSeg = A.seg([{ value: 'ml', label: 'PEFT → category' }, { value: 'ct', label: 'category → PEFT' }], st.dir, function (v) { setDir(v); }, 'Column order');
    stage.appendChild(h('div', { class: 'ro-bar' }, [countEl, h('div', { class: 'ro-dir' }, [h('span', { class: 'ro-gl', text: 'Read' }), dirSeg.el])]));

    /* ---------- dictionary ---------- */
    var listWrap = h('div', { class: 'ro-listwrap' });
    var list = h('div', { class: 'ro-list', role: 'list', tabindex: '0', 'aria-label': 'Dictionary entries (scrollable)' });
    var colsHead = h('div', { class: 'ro-cols', 'aria-hidden': 'true' }, [
      h('span', { class: 'c-peft', text: 'PEFT' }),
      h('span', { class: 'c-cat', text: 'Category theory · geometry' })
    ]);
    list.appendChild(colsHead);
    var rows = entries.map(function (e) {
      var row = h('article', { class: 'ro-row', role: 'listitem', id: uid + '-e' + e.n, 'aria-label': 'Entry ' + e.n });
      var nCell = h('div', { class: 'ro-n' }, [h('span', { class: 'no', text: pad2(e.n) })]);
      if (e.parts.length) {
        var pt = h('span', { class: 'pt', title: 'Cites results of ' + e.parts.map(function (k) { return 'Exposé ' + k; }).join(' and ') });
        e.parts.forEach(function (k) { pt.appendChild(h('span', { text: k })); });
        nCell.appendChild(pt);
      }
      row.appendChild(nCell);
      row.appendChild(h('div', { class: 'ro-peft', html: e.html.peft }));
      row.appendChild(h('div', { class: 'ro-cat', html: e.html.category }));
      row.appendChild(h('div', { class: 'ro-expl', html: e.html.explanation }));
      row.addEventListener('mouseenter', function () { hoverCol(e.i, true); });
      row.addEventListener('mouseleave', function () { hoverCol(e.i, false); });
      list.appendChild(row);
      return row;
    });
    var emptyMsg = h('div', { class: 'ro-empty', hidden: 'hidden' });
    list.appendChild(emptyMsg);
    listWrap.appendChild(list);
    stage.appendChild(listWrap);

    var footBits = ['Typeset from theory/rosetta.json (Exposé IX).',
      N + ' entries cite ' + allNums.length + ' distinct numbered results' + (nUncited ? '; ' + nUncited + ' cites none' : '') + '.'];
    if (nRecord) footBits.push(nRecord + ' of the ' + allNums.length + ' have a referee record: ' + statusCount.verified + ' verified and ' + statusCount.repaired + ' repaired' + (statusCount.twice ? ', ' + statusCount.twice + ' of them twice' : '') + (statusCount.other ? ', ' + statusCount.other + ' other' : '') + '.');
    footBits.push('Conjectures (Conj A) are not numbered results and carry no link.');
    stage.appendChild(h('p', { class: 'ro-foot', text: footBits.join(' ') }));

    /* ---------- reference tooltips ---------- */
    function refName(num) {
      var p = props[num];
      if (p && p.name) return texToText(String(p.name).replace(/\$([^$]*)\$/g, function (x, t) { return t; }));
      var node = document.getElementById(anchorOf(num));
      var nm = node && node.querySelector('.env-name');
      if (nm && !nm.querySelector('mjx-container')) return nm.textContent.replace(/^\s*\(|\)\s*\.?\s*$/g, '').trim();
      return '';
    }
    function refTipHTML(a) {
      var num = a.getAttribute('data-num'), p = props[num];
      var kind = p && p.kind ? p.kind.charAt(0).toUpperCase() + p.kind.slice(1) : a.getAttribute('data-kind');
      var name = refName(num);
      var s = '<div class="t">' + A.esc(kind + ' ' + num) + '</div>';
      if (name) s += '<div style="font-style:italic;color:var(--ink-2)">' + A.esc(name) + '</div>';
      var o = outcome(num);
      /* the tooltip lives outside the figure, so its chip is sized here (12px floor) */
      var cs = ' style="font-size:var(--fs-xs);font-weight:500"';
      var chip = !o ? '' : o === 'verified' ? '<span class="chip yes"' + cs + '>referee: verified</span>' :
        o.indexOf('repaired') === 0 ? '<span class="chip cond"' + cs + '>referee: ' + A.esc(o) + '</span>' : '<span class="chip"' + cs + '>' + A.esc(o) + '</span>';
      s += '<div style="margin-top:.4rem;display:flex;flex-wrap:wrap;gap:.4rem;align-items:center">' + chip +
        '<span style="font-family:var(--f-mono);font-size:.78rem;color:var(--ink-2)">#' + anchorOf(num) + '</span></div>';
      return s;
    }
    function rectEvt(node) { var r = node.getBoundingClientRect(); return { clientX: r.left, clientY: r.bottom }; }
    list.addEventListener('mouseover', function (ev) { var a = ev.target.closest && ev.target.closest('a.ro-ref'); if (a) A.tip.show(refTipHTML(a), ev); });
    list.addEventListener('mousemove', function (ev) { if (ev.target.closest && ev.target.closest('a.ro-ref')) A.tip.move(ev); });
    list.addEventListener('mouseout', function (ev) { var a = ev.target.closest && ev.target.closest('a.ro-ref'); if (a && !(ev.relatedTarget && a.contains(ev.relatedTarget))) A.tip.hide(); });
    list.addEventListener('focusin', function (ev) { var a = ev.target.closest && ev.target.closest('a.ro-ref'); if (a) A.tip.show(refTipHTML(a), rectEvt(a)); });
    list.addEventListener('focusout', function () { A.tip.hide(); });
    list.addEventListener('scroll', function () { A.tip.hide(); updateFade(); }, { passive: true });

    /* ---------- map drawing ---------- */
    var mapGeom = null, dotEls = [], colHover = null, labEls = {}, axEls = [];
    function drawMap() {
      var W = Math.max(260, Math.floor(svgWrap.getBoundingClientRect().width || stage.clientWidth || 600));
      var narrow = W < 560;
      var labW = narrow ? 32 : 196, padR = narrow ? 3 : 8, top = 3, rowH = narrow ? 18 : 20, axisH = 21;
      var colW = (W - labW - padR) / N;
      var rb = Math.max(1.9, Math.min(3.1, colW * 0.27));
      var H = top + PARTS.length * rowH + axisH;
      mapGeom = { W: W, labW: labW, colW: colW, top: top, rowH: rowH, rb: rb, narrow: narrow };
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      svg.setAttribute('width', W);
      svg.setAttribute('height', H);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      labEls = {}; dotEls = []; axEls = [];
      var gx0 = labW, gx1 = W - padR;
      PARTS.forEach(function (p, k) {
        var y = top + k * rowH, cy = y + rowH / 2;
        var g = S('g', { class: 'rm-lab', 'data-part': p.key });
        var band = S('rect', { class: 'rm-band', x: 0, y: y, width: W, height: rowH, rx: 3 });
        g.appendChild(band);
        g.appendChild(S('line', { class: 'rm-guide', x1: gx0, x2: gx1, y1: cy, y2: cy }));
        var num = S('text', { class: 'rm-num', x: narrow ? labW - 7 : 25, y: cy + 5, 'text-anchor': 'end' });
        num.textContent = p.key;
        g.appendChild(num);
        if (!narrow) {
          var nm = S('text', { class: 'rm-name', x: 33, y: cy + 4.5 });
          nm.textContent = p.name;
          g.appendChild(nm);
          var cnt = S('text', { class: 'rm-cnt', x: labW - 10, y: cy + 4, 'text-anchor': 'end' });
          cnt.textContent = partCount[p.key];
          g.appendChild(cnt);
        }
        var tt = S('title', {});
        tt.textContent = 'Exposé ' + p.key + ' (' + p.name + '): ' + partCount[p.key] + ' entries. Click to filter.';
        g.appendChild(tt);
        g.addEventListener('click', function () { setPart(st.part === p.key ? 'all' : p.key); });
        svg.appendChild(g);
        labEls[p.key] = { g: g, band: band };
      });
      colHover = S('rect', { class: 'rm-hover', x: 0, y: top, width: colW, height: PARTS.length * rowH, rx: 2 });
      svg.appendChild(colHover);
      entries.forEach(function (e) {
        var cx = labW + (e.i + 0.5) * colW, ds = [];
        PARTS.forEach(function (p, k) {
          var c = e.byPart[p.key];
          if (!c) return;
          var d = S('circle', { class: 'rm-dot', cx: cx.toFixed(2), cy: (top + k * rowH + rowH / 2).toFixed(2), r: (rb * Math.sqrt(c.length)).toFixed(2) });
          svg.appendChild(d);
          ds.push(d);
        });
        if (!e.refs.length) {
          var ring = S('circle', { class: 'rm-none', cx: cx.toFixed(2), cy: (top + PARTS.length * rowH + 3).toFixed(2), r: Math.max(1.5, rb * 0.62).toFixed(2) });
          svg.appendChild(ring);
          ds.push(ring);
        }
        dotEls[e.i] = ds;
        var hit = S('rect', { class: 'rm-col', x: (labW + e.i * colW).toFixed(2), y: top, width: colW.toFixed(2), height: PARTS.length * rowH + 8 });
        hit.addEventListener('mouseenter', function (ev) { hoverCol(e.i, true); A.tip.show(colTipHTML(e), ev); });
        hit.addEventListener('mousemove', function (ev) { A.tip.move(ev); });
        hit.addEventListener('mouseleave', function () { hoverCol(e.i, false); A.tip.hide(); });
        hit.addEventListener('click', function () { A.tip.hide(); jumpTo(e.i); });
        svg.appendChild(hit);
      });
      var ticks = narrow ? [1, 10, 20, 30, N] : [1, 5, 10, 15, 20, 25, 30, 35, N];
      ticks = ticks.filter(function (t, j) { return t >= 1 && t <= N && ticks.indexOf(t) === j; });
      if (ticks.length > 1 && ticks[ticks.length - 1] - ticks[ticks.length - 2] < (narrow ? 4 : 2)) ticks.splice(ticks.length - 2, 1);
      ticks.forEach(function (t) {
        var tx = S('text', { class: 'rm-ax', x: (labW + (t - 0.5) * colW).toFixed(2), y: top + PARTS.length * rowH + 17, 'text-anchor': 'middle' });
        tx.textContent = t;
        tx._n = t;
        svg.appendChild(tx);
        axEls.push(tx);
      });
      if (!narrow) {
        var ax = S('text', { class: 'rm-axt', x: labW - 10, y: top + PARTS.length * rowH + 17, 'text-anchor': 'end' });
        ax.textContent = 'entry';
        svg.appendChild(ax);
      }
      legend.innerHTML =
        '<span>' + dotSVG(rb, '') + dotSVG(rb * Math.SQRT2, '') + dotSVG(rb * Math.sqrt(3), '') + ' 1 · 2 · 3 results cited from that exposé</span>' +
        '<span>' + dotSVG(rb * 1.2, 'hit') + ' matches the search or filter</span>' +
        '<span>' + dotSVG(rb * 1.2, 'dim') + ' filtered out</span>' +
        (nUncited ? '<span><svg width="10" height="12" viewBox="-5 -6 10 12"><circle class="r" r="' + Math.max(1.5, rb * 0.62).toFixed(2) + '"/></svg> cites no numbered result</span>' : '');
      updateMap();
    }
    function colTipHTML(e) {
      var s = '<div class="t">Entry ' + e.n + '</div><div>' + A.esc(htmlToText(TEX[e.raw.peft] != null ? TEX[e.raw.peft] : smartQuotes(e.raw.peft))) + '</div>';
      s += '<div style="margin-top:.3rem;font-family:var(--f-ui);font-size:.8rem;color:var(--ink-2);font-variant-numeric:tabular-nums">' +
        (e.refs.length ? 'cites ' + A.esc(e.refs.join(', ')) : 'cites no numbered result') + '</div>';
      if (rows[e.i].hidden) s += '<div style="font-family:var(--f-ui);font-size:.78rem;color:var(--ink-2)">hidden by the filter · click to show</div>';
      return s;
    }
    function hoverCol(i, on) {
      if (!mapGeom || !colHover) return;
      if (on) { colHover.setAttribute('x', (mapGeom.labW + i * mapGeom.colW).toFixed(2)); colHover.setAttribute('width', mapGeom.colW.toFixed(2)); }
      colHover.classList.toggle('on', !!on);
    }

    /* ---------- filtering ---------- */
    var visible = [];
    function apply() {
      var terms = queryTerms(st.q);
      var active = terms.length > 0 || st.part !== 'all';
      visible = entries.map(function (e) { return (st.part === 'all' || !!e.byPart[st.part]) && matches(e, terms); });
      var nVis = 0, nums = [];
      entries.forEach(function (e, i) {
        rows[i].hidden = !visible[i];
        if (visible[i]) { nVis++; e.refs.forEach(function (n) { if (nums.indexOf(n) < 0) nums.push(n); }); }
      });
      countEl.innerHTML = '<span class="seg-t">' + (active ? '<b>' + nVis + '</b>of ' + N + ' entries' : '<b>' + N + '</b>entries') + '</span>' +
        ' <span class="sep">·</span> <span class="seg-t">' + nums.length + ' result' + (nums.length === 1 ? '' : 's') + ' cited</span>' +
        (st.part !== 'all' ? ' <span class="sep">·</span> <span class="seg-t">Exposé ' + st.part + '</span>' : '');
      emptyMsg.hidden = nVis > 0;
      if (!nVis) {
        emptyMsg.innerHTML = '';
        var msg = 'No entry matches' + (st.q.trim() ? ' <q>' + A.esc(st.q.trim()) + '</q>' : '') + (st.part !== 'all' ? ' in Exposé ' + st.part : '') + '. Try a method (LoRA, DoRA), a concept (gauge, apex) or a result number (III.10).';
        emptyMsg.appendChild(h('div', { html: msg }));
        emptyMsg.appendChild(h('button', { type: 'button', class: 'btn', text: 'Clear search and filter', onclick: function () { input.value = ''; st.q = ''; setPart('all'); input.focus(); } }));
      }
      colsHead.hidden = nVis === 0;
      clearBtn.hidden = !st.q;
      tryBtns.forEach(function (b) { b.setAttribute('aria-pressed', String(st.q.trim().toLowerCase() === b._q.toLowerCase())); });
      Object.keys(pillBtns).forEach(function (k) { pillBtns[k].setAttribute('aria-pressed', String(st.part === k)); });
      highlight(terms);
      updateMap();
      updateFade();
    }
    function updateMap() {
      if (!mapGeom) return;
      var terms = queryTerms(st.q), active = terms.length > 0 || st.part !== 'all';
      entries.forEach(function (e, i) {
        var vis = visible.length ? visible[i] : true;
        (dotEls[i] || []).forEach(function (d) {
          d.classList.toggle('hit', active && vis && d.getAttribute('class').indexOf('rm-dot') >= 0);
          d.classList.toggle('dim', active && !vis);
        });
      });
      axEls.forEach(function (t) { t.classList.toggle('dim', active && !(visible.length ? visible[t._n - 1] : true)); });
      PARTS.forEach(function (p) {
        var L = labEls[p.key];
        if (!L) return;
        L.g.classList.toggle('sel', st.part === p.key);
        L.band.classList.toggle('sel', st.part === p.key);
      });
    }
    function updateFade() {
      var atEnd = list.scrollTop + list.clientHeight >= list.scrollHeight - 4;
      listWrap.classList.toggle('at-end', atEnd);
    }
    function setPart(k) { st.part = k; list.scrollTop = 0; apply(); }
    function setDir(v) { st.dir = v; stage.classList.toggle('ct-first', v === 'ct'); dirSeg.set(v); }
    var applyDebounced = A.debounce(function () { list.scrollTop = 0; apply(); }, 90);
    function onQuery(now) {
      st.q = input.value;
      clearBtn.hidden = !st.q;
      if (now) { list.scrollTop = 0; apply(); } else applyDebounced();
    }
    input.addEventListener('input', function () { onQuery(false); });
    input.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && input.value) { ev.preventDefault(); input.value = ''; onQuery(true); }
      else if (ev.key === 'Enter') { ev.preventDefault(); onQuery(true); }
    });
    clearBtn.addEventListener('click', function () { input.value = ''; onQuery(true); input.focus(); });

    /* ---------- jump from the map ---------- */
    var lastTarget = null, scrollTok = 0;
    /* eased scroll inside the list (a timeout guarantees the end state if animation frames are throttled) */
    function scrollListTo(y, instant) {
      var tok = ++scrollTok;
      if (instant) { list.scrollTop = y; return; }
      var y0 = list.scrollTop, t0 = null, dur = 360;
      function step(ts) {
        if (tok !== scrollTok) return;
        if (t0 == null) t0 = ts;
        var k = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - k, 3);
        list.scrollTop = y0 + (y - y0) * e;
        if (k < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
      setTimeout(function () { if (tok === scrollTok) { scrollTok++; list.scrollTop = y; } }, dur + 250);
    }
    function jumpTo(i) {
      if (rows[i].hidden) { input.value = ''; st.q = ''; st.part = 'all'; apply(); }
      var row = rows[i];
      var y = list.scrollTop + (row.getBoundingClientRect().top - list.getBoundingClientRect().top) - (colsHead.hidden ? 0 : colsHead.offsetHeight) - 2;
      var rm = A.reducedMotion();
      scrollListTo(Math.max(0, y), rm);
      if (lastTarget) lastTarget.classList.remove('target', 'flash');
      row.classList.add('target');
      if (!rm) { void row.offsetWidth; row.classList.add('flash'); }
      lastTarget = row;
    }

    /* ---------- search highlighting (text nodes only; never inside typeset maths) ---------- */
    function unmark() {
      var ms = list.querySelectorAll('mark.ro-hit'), parents = [];
      for (var i = 0; i < ms.length; i++) {
        var m = ms[i], p = m.parentNode;
        p.replaceChild(document.createTextNode(m.textContent), m);
        if (parents.indexOf(p) < 0) parents.push(p);
      }
      parents.forEach(function (p) { p.normalize(); });
    }
    function highlight(terms) {
      unmark();
      if (!terms.length || !mathReady) return;
      rows.forEach(function (row) {
        if (row.hidden) return;
        var cells = row.querySelectorAll('.ro-peft, .ro-cat, .ro-expl');
        for (var c = 0; c < cells.length; c++) {
          var walker = document.createTreeWalker(cells[c], NodeFilter.SHOW_TEXT, {
            acceptNode: function (n) {
              var pe = n.parentElement;
              if (!pe || pe.closest('mjx-container')) return NodeFilter.FILTER_REJECT;
              if (/\\\(|\\\)/.test(n.data)) return NodeFilter.FILTER_REJECT;
              return n.data.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
            }
          });
          var nodes = [], n;
          while ((n = walker.nextNode())) nodes.push(n);
          nodes.forEach(function (node) {
            var text = normChars(node.data), spans = [];
            terms.forEach(function (t) { spans = spans.concat(occurrences(text, t)); });
            if (!spans.length) return;
            spans.sort(function (a, b) { return a[0] - b[0]; });
            var merged = [];
            spans.forEach(function (s) { var last = merged[merged.length - 1]; if (last && s[0] <= last[1]) last[1] = Math.max(last[1], s[1]); else merged.push([s[0], s[1]]); });
            var frag = document.createDocumentFragment(), pos = 0;
            merged.forEach(function (s) {
              if (s[0] > pos) frag.appendChild(document.createTextNode(node.data.slice(pos, s[0])));
              var mk = document.createElement('mark');
              mk.className = 'ro-hit';
              mk.textContent = node.data.slice(s[0], s[1]);
              frag.appendChild(mk);
              pos = s[1];
            });
            if (pos < node.data.length) frag.appendChild(document.createTextNode(node.data.slice(pos)));
            node.parentNode.replaceChild(frag, node);
          });
        }
      });
    }

    /* ---------- responsive + theme ---------- */
    var lastW = 0;
    function layout() {
      var w = stage.clientWidth || 0;
      stage.classList.toggle('ro-narrow', w > 0 && w < 600);
      if (Math.abs(w - lastW) > 1) { lastW = w; drawMap(); }
    }
    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(layout, 60)).observe(stage);
    else window.addEventListener('resize', A.debounce(layout, 100));
    A.onTheme(function () { drawMap(); });

    /* ---------- first paint: complete at rest ---------- */
    layout();
    if (!mapGeom) drawMap();
    apply();
    function afterMath() { mathReady = true; highlight(queryTerms(st.q)); updateFade(); }
    A.typeset(list).then(afterMath);
    document.addEventListener('atlas:mathjax', function () { A.typeset(list).then(afterMath); });
  });
})();
