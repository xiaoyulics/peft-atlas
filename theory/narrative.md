# Narrative arc for the website

## Title options

Each title is 2–4 words and works on X.

1. **Fibres of Fine-Tuning** (recommended). It names the method: study the fibres, the image and the base change of the map ρ. It reads well as a paper title and as a site title.
2. **Over θ₀**. Minimal and Grothendieckian: every method is an object over the pretrained point. It doubles as the logo, θ₀ with a small arrow landing on it.
3. **Rank Is Not Volume**. The most shareable single claim, usable as a tagline for the thread.
4. **The PEFT Atlas**. Plain and discoverable; good for search and as the hub's name.

## Tagline

> *Every PEFT method is a morphism into the pretrained point. Its image is what it can say, its fibres are how it learns, and base change is how it travels.*

Short form for X: *"LoRA, DoRA, OFT, prefixes, GaLore: one category, three invariants, ninety methods."*

## Epigraphs

These are paraphrases in the spirit of Grothendieck and Klein, not quotations.

1. *Do not attack the hard nut with a hammer. Let the sea rise around it until it opens of its own accord.* (After Grothendieck's image of the rising sea.)
2. *A method is not a thing but an arrow. Ask what it sees of the point where it lands.*
3. *A geometry is the study of what a group leaves unchanged; so is a fine-tuning method.* (After Klein's Erlangen programme.)
4. *Rank lives on cuts; dimension lives in volumes. Confuse them and you will count the wrong thing.*
5. *The right generality makes the hard theorem into a remark, and the remark into a design rule.*
6. *Before asking how a method learns, ask what it cannot help keeping.*

The site uses these one per section, in small caps, with the attribution line "after Grothendieck" or "after Klein" where it applies. No copyrighted text is quoted.

---

## Section-by-section outline

### 0. Hero: "Fine-tuning moves a point."

- **Hook.** An animated θ₀ sits in weight space. Small manifolds of trainable parameters (LoRA's cone, OFT's sphere, (IA)³'s line) map onto it. A single sentence: *PEFT picks the space the point may move in, and the metric it moves with.*
- **Above the fold.** The tagline, three buttons (*The idea*, *The atlas*, *The surprises*), and a running counter: "90+ methods · 7 axes · 41 proved statements · 11 falsifiable predictions".
- **Figure.** A miniature of the Periodic Table (Fig. 12) that is clickable from the first screen.

### 1. The Idea: "Study the morphism, not the object." (Exposés 0–I)

- **Hook.** The page opens with "Everyone compares methods by parameter count. Grothendieck would compare the *maps*." A method is a pointed map ρ : (Q, q₀) → (Θ, θ₀). The frozen model is initial and full fine-tuning is terminal, so every weight-space method lives in that interval.
- **Beats.** The pointed slice. Merging as the canonical map to the terminal object. Why Para adds nothing for weight-space methods, which buys credibility with category theorists. The seven places where the category theory actually works.
- **Figure.** An interval diagram, frozen → M → Full FT, with the merge arrow labelled ρ.

### 2. What a method can say: the image germ (Exposé II)

- **Hook (shareable).** *"No exact initialisation can buy LoRA rank above 2r — and the standard LoRA inits are points of three strata, Gr(r,n) ⊔ Gr(r,m) ⊔ M_r."* HF's `orthogonal` init lies outside the three strata, which the page shows as a fourth marker.
- **Beats.**
  - Zero-init LoRA sits at the **apex** of a cone and loses r(n−r) first-order directions.
  - PiSSA, LoRA-GA and CorDA move the apex away, so they are *different objects*, incomparable with LoRA.
  - **New result:** HRA's default initialisation is an apex too.
  - Smoothness, alignment and scale are three distinct mechanisms of "better init".
- **Figures.** The Apex and the Smooth Point (Fig. 1), plus an inline strata map with the init schemes as labelled points, and the exceptions (HF `orthogonal`, defect pointings) marked separately.

### 3. What a method can never change: Erlangen (Exposé II.B)

- **Hook (shareable).** *"In exact arithmetic, exactly orthogonal fine-tuning can never change the spectrum, however many times you merge."* A footnote says that HF's default Cayley–Neumann OFT is only approximately orthogonal, and with small parameters it shrinks the spectrum at every merge.
- **Beats.**
  - Complete invariants: neuron Gram and spectrum for exactly orthogonal methods; only the zeros of W₀ for torus and Hadamard methods. Sort out the left/right convention once and for all.
  - BOFT keeps |cosines| when no scale entry is zero (the cosines themselves when all scale entries share a sign).
  - **HRA's image = LoRA ∩ W₀·SO(n)** for injective W₀ and even r, an image-level meet. **LoRA-XS = FA × FB**, a genuine categorical product.
  - The torus ladder: (IA)³ ⊂ HiRA₁, DoRA = LoRA + output scale (same image, a theorem), RoAd₁ = (IA)³ over ℂ.
- **Figure.** Gram Lab (Fig. 5).

### 4. Rank is not volume (Exposé II.C)

- **Hook (shareable).** *"LoHa reaches rank r² (for r ≥ 3) — on a set ~m+n dimensions thinner than LoRA₂ᵣ at the same budget; for r ≤ 2 it is simply dominated. GraLoRA multiplies LoRA's maximum rank by k at exactly LoRA's dimension, but caps every block at rank r/k."*
- **Beats.**
  - Rank is a min-cut (symmetric monoidal structure).
  - Unconstrained Kronecker sums are LoRA across another cut, and with aligned splits (equal square splits, and every split HF PEFT produces) a generic rank-one update has *maximal* Kronecker rank. LoKr with inner rank r′ sits inside LoRA_r only when r ≥ min(m₁,n₁)r′.
  - Transport of structure: FourierFT, WaveFT, C3A, LoRA-XS and MiSS are coordinate masks in other frames.
  - **Design rule:** report d(M) next to |M|.
- **Figures.** Rank Is Not Volume (Fig. 7), Cut the Tensor (Fig. 8), Plug Workbench (Fig. 6).

### 5. How a method learns: fibres and metrics (Exposé III)

- **Hook (the viral one).** *"LoRA+ is α in disguise. Under Adam (ε=0, no weight decay or clipping), LoRA+ with ratio λ is* exactly *LoRA with α·λ — max trajectory difference 0.0. Try it."* Credit Schulman et al. (2025) for the LoRA/Adam case in the same breath; our addition is the general torus (any multihomogeneous method, SGD's c² law, per-block ε and decay).
- **Beats.**
  - The cometric K = DρΛDρᵀ: two isomorphic methods can train differently.
  - Gradient descent is natural exactly where cometrics agree; for LoRA's gauge that means O(r), so plain-GD "LoRA" is a family of dynamics indexed by O(r)\GL_r.
  - Noether charges (Zhao et al. 2023) hold under gradient flow for *any* cotangent signal, which covers DoRA's detached norm.
  - The crossover scale σ* between LoRA-FA-like and balanced regimes, credited to the linear-network literature.
  - The hyperparameter torus.
  - The optimizer hierarchy B_r ⊂ O(r) ⊂ GL_r: Adam sees a finite shadow of the gauge.
- **Figures.** Hyperparameter Orbit Explorer (Fig. 2), Noether Hyperbolas and Crossover (Fig. 3), Gauge Lab (Fig. 4).

### 6. Tangent and cotangent: the lens (Exposé IV)

- **Hook.** *"GaLore is LoRA with a moving frame."* Credit Torroba-Hennigen et al. up front, then show the categorical surplus: a projector × schedule table that classifies LoRA-FA, random-projector GaLore, MeZO, EVA, GaLore, BitFit, LISA and ReLoRA, and the observation that every gradient-projection method whose optimizer reads only its gradients (no weight decay) is, period by period, a forward linear method on a moving base. Flora, which compresses only its momentum, is not of this kind. Frobenius integrability only matters for anchors that depend on θ.
- **Figure.** An inline table with hover cards, plus a backprop-cone strip (memory bars).

### 7. Moving the base point (Exposé V)

- **Hook.** *"QLoRA needs no new theory: it is LoRA pointed at the wrong point."*
- **Beats.**
  - Pointing defects: three orders of quantising and splitting, three different defects.
  - Rebasing closure: ReLoRA reaches exactly rank Kr, exact-Cayley block OFT is stuck forever, the butterfly connects only at full depth, and re-HRA climbs inside SO(n).
  - Merging is gauge-sensitive: the LoRAHub trap.
- **Figures.** Rebasing Reach (Fig. 10), and the LoRAHub tab of the Gauge Lab.

### 8. Extensions and merging (Exposé VI)

- **Hook (shareable).** *"Which weight absorbs (IA)³ depends on your activation function: W_up under SwiGLU, never under GELU (W_down always works)."*
- **Beats.**
  - Merging as a lifting problem, certified by seven rewrite rules with explicit side conditions (RoPE, GQA sharing, bias slots).
  - The activation function fixes the privileged basis (permutation test).
  - Prefix tuning is a gated adapter that cannot reorder content attention at the layer where it acts; LLaMA-Adapter's contribution is *pointedness*.
  - ReFT is LoRA on an affine identity box, applied at selected positions.
- **Figures.** Merge Game (Fig. 9), Prefix Is a Gate (Fig. 11).

### 9. The Atlas (Exposés VII–VIII)

- **Hook.** *"Ninety methods, seven coordinates, every arrow with its proof."*
- **Beats.** Fourteen worked cards (LoRA, DoRA, OFT/BOFT, (IA)³, VeRA, LoHa/LoKr, FourierFT, BitFit, adapters, prefix and prompt tuning, PiSSA/LoftQ/QLoRA, ReLoRA, GaLore, merging), then the Periodic Table with filters, then the arrow rules.
- **Figure.** The Periodic Table (Fig. 12), full screen.

### 10. Rosetta stone

- **Hook.** *"A dictionary for two communities."* A two-column searchable table (`rosetta.json`). Each row links to the theorem that makes the translation exact.

### 11. Predictions and honesty (Exposés X–XI)

- **Hook.** *"Eleven experiments that could prove us wrong."* Each prediction carries its optimizer scope and a "run it" link where a toy version exists (LoRA+ vs α, permutation test, rank-one Kronecker test).
- **Beats.** An **"Analogy vs theorem"** box ("Noether", "Erlangen", "gauge" and "moduli" each say exactly what they mean here). Open problems. A **"Known results we re-derive"** box crediting Du–Hu–Lee, Arora–Cohen–Hazan, Tarmoun et al., Min et al., He et al., Petrov et al., Torroba-Hennigen et al., Kempf–Ness and Srebro et al. Add Zhao et al. (charges), Schulman et al. (LoRA+ and α), Mishra–Sepulchre (scaled GD on the quotient), Meng et al. (the rank-2r PiSSA conversion), Sussmann and Chen–Lu–Hecht-Nielsen (activation symmetries) and Liu et al. ((IA)³ merging). A **referee-status page** generated from `theory/propositions.json` lists every numbered result with its status (verified or repaired) and the referee notes. This is the part that earns citations rather than ridicule.

### 12. Cite / colophon

- A BibTeX block, the reproducibility note (`theory/framework_checks.py`; every [Num] claim re-runs in about a minute), and the status-label legend.

---

## X / social thread skeleton

Eight posts, one figure GIF each.

1. "Every PEFT method is a morphism into the pretrained point. We rebuilt the whole field — LoRA to prefixes to GaLore — from that one idea." Use the hero GIF.
2. LoRA+ ≡ α·λ under Adam (ε=0), exactly, after Schulman et al. Orbit explorer GIF.
3. No exact init buys rank > 2r; the standard LoRA inits live in three strata, Gr ⊔ Gr ⊔ M_r. Apex GIF.
4. Zero-init LoRA *and* HRA start at a cone's apex. Cone GIF.
5. Rank is not volume: LoHa vs LoRA₂ᵣ, GraLoRA. Scatter.
6. Exactly orthogonal methods can never change the spectrum. Gram Lab.
7. (IA)³ can fuse into W_up under SwiGLU, never under GELU. Merge Game.
8. Eleven falsifiable predictions, plus an honest "analogy vs theorem" page. Link.
