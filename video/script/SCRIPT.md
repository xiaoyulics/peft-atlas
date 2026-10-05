# Fine-tuning moves a point — explainer script (draft 2, after an independent fact-check)

*A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges.* Narration by Gemini 3.8 Flash TTS,
voice **Sulafat**. Visual style: **quiet C** (deep navy, Source Serif 4 and IBM Plex, tide and gold accents, a halo on
θ₀ only), with **paper cards** in the light atlas style for statements and credits. Target length about 17 minutes (narration at Sulafat's pace, about 130 words a minute).

How to read this file. Each beat has an id, an estimated length, tags for the kind of animation, what is **on screen**,
the **narration** (every line starting with `>` is sent to the voice, exactly as written; an optional **Delivery** line
adds a direction for that beat only), and what each claim was
**checked against** in `theory/propositions.json` and the paper. Tags:
`[3D]` geometry in weight space · `[SIM]` a real computation whose numbers appear on screen · `[STORY]` narrative
devices (nuts, the sea, chapter cards) · `[CARD]` a paper card · `[CUT]` tensor diagram · `[DATA]` the catalogue.

**The four nuts.** Shown on the shore in the prologue; each dissolves in its own chapter.

| | Nut (as written on its card) | Scope tag on the card | Dissolves in | Result |
|---|---|---|---|---|
| ① | No exact initialisation lets LoRA reach an update of rank above 2r. | any exact init | II · The image | Thm II.1 |
| ② | LoRA+ is LoRA with α rescaled. | Adam, ε = 0 (0/0 := 0), no decay, no clipping, B₀ = 0, shared schedule | III · Fibres | Thm III.10, Cor III.11 |
| ③ | GaLore is one-sided LoRA with a moving frame. | gradient-only optimizer (Adam or SGD), no weight decay, no clipping | IV · The lens | Prop IV.2 |
| ④ | Exactly orthogonal methods never change singular values, however often you merge. | exact arithmetic; one orthogonal factor per weight | V · Base change | Cor II.6, Thm V.3 |

**The sea.** No water during the mathematics. Between chapters the stage clears, a chapter card appears and the sea
rises one notch (six notches). When a nut dissolves, the shot cuts to the shore for about four seconds. In the coda the
sea rises over everything.

---

## 0 · Cold open (0:00–0:55)

### 0.1 · Two runs, one path  ⏱ 0:26 · `[SIM]`
**On screen.** Two panels, then one. Left: "LoRA+, λ = 16". Right: "LoRA, α × 16". Each draws its weight trajectory
(a 2-D projection of W(t)) and its loss curve, step by step. The panels slide together and the two paths overlay
perfectly. A counter under them: `max |W_LoRA+ − W_LoRA| = 0.0`, which stays at 0.0 to the last step.
Numbers come from our own run, `video/sim/lora_plus.py` (zero-B start, Adam with ε = 0, no decay, no clipping, shared
schedule, λ = 16): the difference is exactly 0.0 over 400 steps, against 1.41 for the control. Footnote on screen:
"ε = 0 with the convention 0/0 := 0 (stock Adam: ε on A and 16ε on B, still bit-exact) · after Schulman et al. 2025".

> Here are two fine-tuning runs. <short pause> One is LoRA-plus, which trains its two factors at different learning rates. The other is plain LoRA, with its scale, alpha, turned up. <short pause> Train both with Adam, with epsilon, weight decay and clipping switched off, and watch their weights. <long pause> They're identical. Not close. Identical, step after step.

**Checked against.** Cor III.11 (exact equality under Adam, ε = 0, no decay, no clipping, shared schedule, same data
order and dropout masks, from B₀ = 0); [Num] in III.11: maximum difference exactly 0.0 for power-of-two λ, 0.93 for the
unrescaled control; the 0/0 := 0 convention for ε = 0 (remark in `10-predictions.tex`). Credit: the footnote here and
the card in III.4 (Schulman et al. 2025).

### 0.2 · 382 names  ⏱ 0:25 · `[DATA]`
**On screen.** The overlaid paths fade. Method names from `research/catalog.json` fall like rain (LoRA, DoRA, OFT,
BOFT, VeRA, (IA)³, BitFit, QLoRA, GaLore, HRA, LoHa, LoKr, AdaLoRA, prefix tuning, …) and pile up; a counter climbs to
382. One name card flips to show two empty slots, "trainable parameters" and "benchmark score" (schematic, no values).

> That isn't luck, and it isn't the only coincidence of its kind. Parameter-efficient fine-tuning now has hundreds of methods; this atlas lists three hundred and eighty-two. We usually compare them with two numbers: how many parameters they train, and how well they score. Those numbers can't see a coincidence like this one. To see it, we have to compare methods by their shape.

**Checked against.** Abstract ("has hundreds of methods, usually compared by parameter count and benchmark score");
README and `tables/counts.tex` (382 catalogued).

### 0.3 · Title  ⏱ 0:08 · `[STORY]`
**On screen.** Title card with contours (the only other place contours appear is the end card): *A Categorical Atlas of
Parameter-Efficient Fine-Tuning* / *Cones, Orbits, and Gauges*. Music only.

---

## Prologue · The nut and the sea (0:55–2:10)

### P.1 · Hammer or sea  ⏱ 0:34 · `[STORY]`
**On screen.** A nut on a rock. A hammer and chisel strike it at one point after another; hairline cracks. Then the
water rises around the nut, very slowly; the shell softens; a light touch opens it.

> The mathematician Alexander Grothendieck described two ways to open a hard nut. <short pause> You can take a hammer and chisel and strike, point after point, until the shell cracks. <short pause> Or you can soak it, and let time pass, until the pressure of a hand is enough. On a larger scale, he called this second way the rising sea. So slowly that nothing seems to happen, the water rises around the problem, until it dissolves into a theory that reaches far beyond it.

**Checked against.** Paraphrase of *Récoltes et Semailles* as given in Appendix F (`F-making-of.tex`, "Who did
what"); no direct quotation.

### P.2 · Four nuts on the shore  ⏱ 0:36 · `[STORY]`
**On screen.** The shore, with four nut cards in a row (text and scope tags as in the table above). Card ② glows
briefly when the narration says "You've already met one".

> Here are four nuts from fine-tuning. You've already met one. <short pause> No exact initialisation lets LoRA reach an update of rank above twice its own. LoRA-plus is LoRA with alpha rescaled. GaLore is one-sided LoRA with a moving frame. And exactly orthogonal methods never change a weight's singular values, however often you merge them. <short pause> Each sounds like a separate fact with its own clever proof, and most were first proved that way. We won't crack them again. We'll raise the sea, and watch them dissolve.

**Checked against.** Abstract (statements of ①, ②, ④); Thesis, `00-thesis.tex` (③). Scope tags carried on the cards.
Conclusion: three of the four were known before this paper; the setting adds a short proof, an exact scope and one proof
for every method of the same shape.

---

## I · The slice — fine-tuning moves a point (2:10–3:50)

### I.0 · Chapter card  ⏱ 0:05 · `[STORY]`
**On screen.** "EXPOSÉ I · The slice — *fine-tuning moves a point*". Sea rises to notch 1.

### I.1 · One point  ⏱ 0:16 · `[3D]`
**On screen.** Dark 3-D space. One point appears with its halo: θ₀, "the pretrained model". A full fine-tuning run
carries a copy of the point along a free curve to somewhere else.

> Start with the pretrained model. All of its weights together are a single point in an enormous space. Call it theta-zero. <short pause> Fine-tuning moves that point. Full fine-tuning can move it anywhere.

### I.2 · A method is a map  ⏱ 0:18 · `[3D]`
**On screen.** A small disc on the left, "Q — trainable parameters", with a marked point q₀. An arrow ρ carries the disc
into weight space; q₀ lands on θ₀. As a probe point wanders in Q, ρ of it traces a shape through θ₀. The formula from
the style sample: ρ : (Q, q₀) → (Θ, θ₀), ρ(q₀) = θ₀.

> A parameter-efficient method trains only a few numbers. So think of it as a map, rho, from a small space of trainable parameters into weight space, with one rule: the starting parameters land exactly on theta-zero.

**Checked against.** Definition I.2; Thesis.

### I.3 · The slice  ⏱ 0:29 · `[STORY]` `[CARD]`
**On screen.** "Frozen" (a single dot) at the left, "full fine-tuning" at the right, and methods between them joined
by arrows rather than lined up on one axis: one pair, zero-init LoRA and split LoRA, has no arrow either way. Tag:
schematic. An arrow M → N labelled "N simulates M", drawn as a small commuting triangle into Θ.
Paper card: *Observation I.4 — the frozen model is initial and full fine-tuning is terminal.*

> Maps like this form what mathematicians call a slice category. An arrow from one method to another means the second can simulate the first; in particular, wherever the first can take theta-zero, the second can follow. At one end sits the frozen model, which can't move at all. At the other sits full fine-tuning, which can simulate everything. Every method that starts exactly at theta-zero lives in between.

**Checked against.** Definition I.2 (morphism h with ρ_N ∘ h = ρ_M, read "N simulates M"); Observation I.4 (image
inclusion alone gives only a set-level simulation); `01-slice.tex` (adapters and prefixes extend the network and fall
outside the interval); Thm II.2 (zero-init and split images are incomparable).

### I.4 · Three questions  ⏱ 0:35 · `[STORY]`
**On screen.** Three words appear one by one beside θ₀ and stay for the rest of the film as a small legend:
**image** (what it can say), **fibres** (how it learns), **base change** (how it travels).

> So instead of asking how many parameters a method has, we can ask three questions about its map. What is its image: where can it go? What are its fibres: which parameter settings give the same weights, and how does training move through them? And what happens when the base point itself moves? <short pause> Its image is what it can say. Its fibres are how it learns. Base change is how it travels.

**Checked against.** Thesis ("three invariants of the morphism"); site hero line.

---

## II · The image — what a method can say (3:50–7:20)

### II.0 · Chapter card  ⏱ 0:05 · `[STORY]`
"EXPOSÉ II · The image — *what a method can say*". Sea rises to notch 2.

### II.1 · Cone, orbit, line  ⏱ 0:36 · `[3D]`
**On screen.** The style-sample shot, extended: LoRA's double cone with its tip at θ₀; OFT's arc of the circle about
0 through θ₀; (IA)³'s line through 0 and θ₀ (tag: schematic; for m output neurons the closure is an m-dimensional
flat through 0); BitFit moving parallel to a coordinate axis. The cone turns slowly.

> Let's look at images. <short pause> LoRA adds a low-rank matrix to each weight, and the matrices of rank at most r don't form a flat space. They form a cone, and for standard LoRA its tip sits right at theta-zero. <short pause> OFT rotates the weights, so it moves theta-zero along an orbit. I-A-cubed rescales neurons; in this picture, it slides along a line through the origin. And BitFit, which trains only the biases, moves parallel to a few coordinate axes.

**Checked against.** Teaser figure caption (`00-thesis.tex`): LoRA is θ₀ + M_≤r with apex at θ₀ (Thm II.2); OFT is the
orbit of θ₀ under rotations; (IA)³ is a torus orbit whose closure in the picture is the line through 0 and θ₀ (Thm II.5);
BitFit moves along a coordinate line.

### II.2 · Disguises  ⏱ 0:20 · `[3D]` `[CARD]`
**On screen.** DoRA's name card splits into "magnitude" and "direction", then reassembles as "LoRA → one gain per
output neuron". Paper card: *Proposition II.8.*

> Images also expose disguises. DoRA, which splits each weight into a magnitude and a direction, reaches exactly the same weights as LoRA followed by one gain per output neuron. Beyond those gains, anything DoRA adds, it adds in training, not in reach.

**Checked against.** Prop II.8 (HF PEFT's DoRA; W₀ with no zero rows; r ≤ min(m, n); "any difference ... comes from
training dynamics").

### II.3 · Tips and smooth points  ⏱ 0:50 · `[3D]` `[CARD]`
**On screen.** A smooth point on the cone's side with its flat tangent plane; then the tip, where the reachable
first-order directions form a smaller set. Counters: "first-order directions: mr" at the tip versus "dimension:
r(m + n − r)". A split initialisation shown as a point on the cone's side. Paper card: *Theorem II.2*; a short mention
card for *Theorem II.7* (HRA).

> Where you start on an image matters. At a smooth point, the directions you can move in, to first order, form a flat plane as large as the image itself. At the tip of a cone, they don't. <short pause> Zero-initialised LoRA, the usual default, starts exactly at the tip. With B at zero, nudging A changes nothing, so at first only B can move the weights, and fewer directions are open. Split initialisations, which carve a low-rank piece out of the frozen weight, start at a smooth point instead. <short pause> The same lens finds tips where you might not expect them: HRA, a method built from reflections, also starts at one at Hugging Face's default initialisation, when r is even and the frozen weight has full column rank.

**Checked against.** Thm II.2 (zero-factor inits at the apex with r(n − r) or r(m − r) fewer first-order directions;
split inits at a smooth point; set-level, exact arithmetic); Lemma II.4 (zero gate); Prop II.3 (mr directions at a
zero-factor init); Thm II.7 (HRA at HF's paired default init, which the HRA paper does not state; W₀ of full column rank, r ≤ n − 2 even).

### II.4 · Nut ① dissolves  ⏱ 1:00 · `[3D]` `[SIM]` `[CARD]` `[STORY]`
**On screen.** A point elsewhere on the cone; the whole cone slides until it passes through θ₀ again ("subtract a frozen
residual"). Many translated copies sweep through θ₀; their union fills a region labelled θ₀ + (C − C). Then a histogram
from our own sampling (for example m = n = 8, r = 2): the rank of reachable updates over thousands of random split
initialisations, with bars up to 2r = 4 and nothing above. Paper card: *Theorem II.1*, with the credit line
"known in practice: converting a PiSSA adapter to plain LoRA takes rank 2r (Meng, Wang & Zhang 2024; HF PEFT) · new
here: 2r is needed in general, and the argument covers every additive method". Cut to the shore:
nut ① dissolves.

> Now the first nut. Split initialisations start with a nonzero adapter, and subtract that piece from the frozen weights, so the model still begins at theta-zero. In the picture, that's sliding the whole cone until it passes through theta-zero again. <short pause> Slide it every possible way, and together the copies cover theta-zero plus all differences of two points on the cone. The difference of two matrices of rank at most r has rank at most two r. <short pause> So no exact initialisation, zero or split, lets LoRA reach an update of rank above two r. Practitioners already knew this ceiling: converting a split adapter back into plain LoRA takes rank two r. The picture adds that two r is really needed, and that the same few lines work for any additive method.

**Checked against.** Thm II.1 (a)–(c); split inits attain 2r when 2r ≤ min(m, n); `02a-strata.tex` after Thm II.1
(Meng et al. 2024 convert a trained PiSSA adapter into a rank-2r LoRA; the theorem adds that the conversion loses
nothing and that rank 2r is needed in general).

### II.5 · Rank is a cut, dimension is a volume  ⏱ 0:55 · `[CUT]` `[CARD]`
**On screen.** LoRA as a tensor diagram: input wire n → box A → wire r → box B → output wire m. A vertical cut line
sweeps across and stops on the r-wire. LoHa's diagram: two such chains joined by an entrywise product; the narrowest cut
crosses both r-wires, r × r. Then two bars at equal budget: rank (LoHa ahead for r ≥ 3) and dimension (LoRA₂ᵣ ahead on
large layers). Paper cards: *Theorem II.10* (standard), *Proposition II.12*.

**Delivery.** a little slower here: this is the densest passage, so give each sentence room to land.

> Rank has a picture of its own. <short pause> Draw an adapter as boxes joined by wires. Cut the diagram anywhere between input and output, and the rank can't exceed the product of the sizes of the wires you cut. <short pause> For LoRA, the narrowest cut is the single wire of size r. LoHa multiplies two low-rank factors entry by entry, and its narrowest cut is r times r. <short pause> But rank is a cut, and dimension, the number of independent directions a method can actually move in, is a volume. <short pause> At the same budget as LoRA of rank two r, LoHa reaches rank up to r squared, but on large layers it gives up directions to get there: a trade of dimension for rank when r is three or more, and a pure loss when r is two or less.

**Checked against.** Thm II.10 (cut-rank bound, for a chosen presentation); Prop II.12 (equal budget 2r(m + n); when
2r² < m + n − 1 LoHa is a dimension-for-rank trade for r ≥ 3 and strictly dominated for r ≤ 2; the bound stops
separating them when 2r² ≥ m + n − 1, hence "on large layers").

---

## III · Fibres — how a method learns (7:20–10:20)

### III.0 · Chapter card  ⏱ 0:05 · `[STORY]`
"EXPOSÉ III · Fibres — *how a method learns*". Sea rises to notch 3.

### III.1 · The gauge  ⏱ 0:32 · `[SIM]`
**On screen.** Three heat maps: B (m × r), A (r × n) and their product BA (m × n). A slider moves an invertible r × r
matrix g along a path; B becomes Bg⁻¹ and A becomes gA, and both maps change visibly while BA stays fixed to the last
digit (a live readout of max |ΔBA| at machine precision). Label: "same weights, different parameters — a fibre".

> Images say where a method can go. Fibres say how it gets there. <short pause> Take a LoRA update, B times A, and slip any invertible r-by-r matrix g into the middle: B times g-inverse, times g times A. The factors change. Their product doesn't. <short pause> All of these settings give the same weights. When the product has full rank, they make up the whole fibre, and the group of all such g is LoRA's gauge.

**Checked against.** Thesis and Thm III.3 (the linear gauge (B, A) ↦ (Bg⁻¹, gA)); Exposé XI table (fibres are gauge
orbits on the full-rank stratum).

### III.2 · A conserved charge  ⏱ 0:32 · `[SIM]`
**On screen.** A small LoRA (for example m = n = 6, r = 2) trained three ways on the same task. Above each run, the
charge Φ = BᵀB/η_B − AAᵀ/η_A as a 2 × 2 tile with a drift meter. Gradient flow: the meter stays flat (drift 8 × 10⁻¹³
in `video/sim/fibres.py`). Then one step from the same mid-training state at step sizes 0.02, 0.01 and 0.005: plain
SGD's drift falls fourfold at each halving (8.2 × 10⁻⁵, 2.1 × 10⁻⁵, 5.1 × 10⁻⁶), an Adam step's only twofold
(2.2 × 10⁻², 1.1 × 10⁻², 5.6 × 10⁻³). Credit on screen: Du, Hu & Lee 2018; Arora, Cohen & Hazan 2018; Kunin et al. 2020;
Zhao et al. 2022.

> Training moves through the fibres too, and the symmetry leaves a fingerprint. Under gradient flow, one quantity never changes: the balance between the two factors, B-transpose-B minus A-A-transpose, each divided by its own learning rate. It's a conserved charge, known for factorised networks since 2018, in the spirit of Noether's theorem. <short pause> With plain SGD steps, it drifts, but only at second order in the step size. Under Adam, it isn't conserved at all.

**Checked against.** Thm III.5 and Cor III.6 (Φ conserved under gradient flow with block-scalar rates; no Adam, momentum
or discrete steps beyond first order); Exposé X prediction (O(η²) per SGD step); [Num] in III.6 (one Adam sign step moves
Φ by an amount proportional to the learning rate); III.5's own note (known: Zhao et al.; the principle: Du–Hu–Lee 2018,
Arora–Cohen–Hazan 2018, Kunin et al.); X.8 scope (SGD without momentum or weight decay).

### III.3 · What each optimizer sees  ⏱ 0:33 · `[SIM]` `[CARD]`
**On screen.** Two zero-B starts that differ by g acting on A₀. Three rows, each with two overlaid weight paths and a
gap readout from `video/sim/fibres.py`: SGD with g a generic rotation (paths coincide, 7 × 10⁻¹⁶); Adam with the same g
(paths separate, 0.345); Adam with g a signed permutation that is not a rotation, swapping the channels and flipping
both signs (paths coincide, about 10⁻¹⁶). A nested-group diagram: signed permutations ⊂ O(r) ⊂ GL_r. Paper card:
*Theorem III.12* (with its note that undamped scaled GD and LoRA-RITE respect all of GL_r), *Corollary III.13*.

> Here's the surprise. The gauge is a whole group, but the usual optimizers respect only part of it. Gradient descent respects the rotations and reflections. Adam respects far less: only the signed permutations, which shuffle the r channels and flip their signs. <short pause> Start Adam from two initialisations related by a generic rotation, and the runs drift apart. Relate them by a signed permutation, and they stay identical. The optimizer sees more of where you start than the image does.

**Checked against.** Thm III.12 (full-rank locus, channel-uniform hyperparameters: O(r) for SGD with or without
momentum, signed permutations B_r for Adam/AdamW); Cor III.13 (from zero-B: SGD sees A₀ᵀA₀, Adam sees A₀ up to signed
permutations; the row-space classification is coarser than what training sees); [Num] in III.12.

### III.4 · Nut ② dissolves  ⏱ 0:55 · `[SIM]` `[CARD]` `[STORY]`
**On screen.** A torus of rescalings, tagged schematic (the group is (ℝ>0)², one factor per block): one circle
"rescale A: its start and its rate together", the other "rescale B: its start and its rate together", with a point
marked "LoRA" and another "LoRA+, λ" joined along the torus. The opening simulation returns, now labelled with its exact
settings, and the control run (LoRA+ against unrescaled LoRA) peels away (1.41 on our toy problem; the paper's 0.93 is on a
different one). Paper card: *Theorem III.10 / Corollary
III.11*, credit "LoRA and Adam, ε included: Schulman & Thinking Machines Lab 2025 · new here: SGD (factor √λ)
and every multihomogeneous method". Cut to the shore: nut ② dissolves.

> One more symmetry hides in the hyperparameters. Rescale a block's starting size and its learning rate, adjust alpha to match, and under Adam, with epsilon, decay and clipping off, the weights follow exactly the same path. These rescalings form a torus. <short pause> LoRA-plus, which gives B a learning rate lambda times larger than A's, is just a point on that torus. That's the coincidence from the start, and our second nut: from the usual zero start, LoRA-plus and plain LoRA with alpha times lambda take the same path, exactly. <short pause> Schulman and colleagues found this for LoRA and Adam in 2025, epsilon included. The torus carries it to SGD, and to any method whose update is a product of its blocks.

**Checked against.** Thm III.10 (rescaling each block's initial scale and learning rate together with the overall scale
s, and under Adam the per-block ε, decoupled decay and clipping thresholds, leaves the trajectory unchanged; same data
order and dropout masks); Cor III.11 (conditions as on the nut card; with ε > 0 or AdamW the per-block ε and decay must be
rescaled); [Num] in III.11 (0.0 for power-of-two λ, 0.93 for the control); III.10's own note (known for LoRA and Adam,
including ε > 0: Schulman et al. 2025; new: arbitrary multihomogeneous updates and the SGD c² law).

---

## IV · The lens — methods that act on gradients (10:20–11:20)

### IV.0 · Chapter card  ⏱ 0:05 · `[STORY]`
"EXPOSÉ IV · The lens — *when a method compresses the gradient*". Sea rises to notch 4.

### IV.1 · Nut ③ dissolves  ⏱ 0:50 · `[SIM]` `[CARD]` `[STORY]`
**On screen.** A forward arrow (weights → loss) paired with a backward arrow (gradient), drawn as a lens. Two procedures
run side by side on the same data: GaLore (project the gradient, Adam step in the small space, map back) and one-sided
LoRA with GaLore's projector as the frozen factor (train, merge, restart each period, optimizer state carried over).
Their weight heat maps evolve identically over several periods; readout "difference ≈ 6.1 × 10⁻¹⁶ (machine precision)",
from `video/sim/galore.py`. A
third run resets the LoRA side's optimizer state at each period and separates (0.115 on our toy problem; the paper's
0.37 is on a different one). Paper card: *Proposition IV.2*, credit
"random projections: Hao, Cao & Mou 2024 · GaLore: Torroba-Hennigen, Lang, Guo & Kim 2025 · backprop as a lens: Fong,
Spivak & Tuyéras 2017; Cruttwell et al. 2021". Cut to the shore: nut ③ dissolves.

> Some methods add no adapter at all: they train the weights directly, but compress the gradient. GaLore projects each gradient onto a small subspace and runs Adam there. <short pause> Backpropagation has the shape of a lens: a forward map paired with a backward one. Through that lens, a linear adapter and a compressed gradient are two views of the same thing. <short pause> So the third nut: without weight decay or gradient clipping, GaLore is exactly one-sided LoRA, with GaLore's projector as the frozen factor, merged and restarted every period, its optimizer state carried over. Torroba-Hennigen and colleagues proved this for GaLore in 2025. The lens gives it a short proof by induction, for every linear adapter.

**Checked against.** Prop IV.2 (degree-one collapse; optimizers that depend only on received gradients; GaLore with
weight decay 0; decoupled decay, parameter-scaled Adafactor, trust-ratio optimizers and full-gradient clipping break it;
proof by induction); [Num] in IV.2 (7 × 10⁻¹⁶ under Adam over one period; 0.37 when the LoRA side resets); credit
Torroba-Hennigen et al. 2025.

---

## V · Base change — how a method travels (11:20–13:20)

### V.0 · Chapter card  ⏱ 0:05 · `[STORY]`
"EXPOSÉ V · Base change — *how a method travels*". Sea rises to notch 5.

### V.1 · Quantisation moves the base point  ⏱ 0:32 · `[3D]` `[CARD]`
**On screen.** The only scene with a grid: the grid of a 4-bit quantiser, tagged schematic (NF4's levels
are not evenly spaced). θ₀ snaps to its nearest node κθ₀; LoRA's
cone moves with it; a red arrow e = κθ₀ − θ₀ shows the miss. Paper card: *Proposition V.2.*

> So far theta-zero has stayed put. But plenty of things move it. <short pause> Quantise the model to four bits, and the frozen weights snap to a grid. Q-LoRA is ordinary LoRA pointed at the snapped weights, not the real ones. So it starts off target by the quantisation error, which is generically full rank, and it can only get back to the original point if that error has rank at most r.

**Checked against.** Prop V.2 (QLoRA anchored at κW misses W by e = κW − W, generically full rank; reaches W only if
rank e ≤ r); teaser figure caption.

### V.2 · Merge and restart  ⏱ 0:22 · `[SIM]` `[3D]`
**On screen.** Train, merge, restart: θ₀ hops to a new base point each cycle and a fresh cone grows from it. A readout of
the largest reachable rank steps r, 2r, 3r, … (computed on a small example).

> Merging moves theta-zero too. Train an adapter, merge it into the weights, and start a fresh one from the new point. Every cycle re-points the method, and the reach grows: after K cycles, Re-LoRA reaches exactly the updates of rank at most K times r.

**Checked against.** Thm V.3 (ReLoRA reaches exactly W₀ + {rank ≤ min(Kr, m, n)}).

### V.3 · Nut ④ dissolves  ⏱ 0:45 · `[SIM]` `[CARD]` `[STORY]`
**On screen.** The singular values of a weight as a row of bars. Fifty merge cycles of exactly orthogonal (Cayley) OFT:
the bars do not move (readout at machine precision). The same fifty cycles with LoRA merges: the bars change. Then HF
PEFT's default Cayley–Neumann OFT (PEFT ≥ 0.18): the bars sink slowly, with a counter of how far the neuron Gram matrix
has moved. Paper cards: *Corollary II.6*, credit "OFT was built to keep these angles: Qiu et al. 2023, after Liu et al.
2018", and *Theorem V.3*. Cut to the shore: nut ④ dissolves; all four nuts are gone.

> Now the last nut. A method that rotates the weights from the input side, exactly, never changes the inner products between neurons. That is what OFT was designed for. So it never changes a single singular value, and in exact arithmetic you can merge as often as you like: a rotation times a rotation is still a rotation. <short pause> One caution that matters in practice. The default OFT in recent versions of Hugging Face's PEFT library uses an approximation that isn't exactly orthogonal. Each merge shrinks the weights a little, and in the paper's test, two hundred merges move the neuron inner products by about a fifth.

**Checked against.** Cor II.6 (W = W₀R with R exactly orthogonal on the input side, including after merge-and-restart:
WWᵀ fixed, so neuron norms, angles and singular values are fixed; excludes HF's default Cayley–Neumann OFT); Thm V.3 (with
small generators each default-OFT merge is a contraction); [Num] in V.3 (the neuron Gram matrix K_out moves by 0.2 %,
1.3 %, 6 % and 22 % after 1, 10, 50 and 200 default-OFT merges, block size 8, generator entries about 0.05).

---

## VIII–XI · The atlas (13:20–14:30)

### A.0 · Chapter card  ⏱ 0:05 · `[STORY]`
"EXPOSÉS VIII–XI · The atlas — *every method, seven coordinates*". Sea rises to notch 6.

### A.1 · Seven coordinates  ⏱ 0:38 · `[DATA]`
**On screen.** The 382 names from the opening return and settle into a table, one column per coordinate, coloured by
value. Then the 403 simulation arrows light up as a network, and the 485 obstructions appear as short red bars.

> Do this for every method, and you get an atlas. The paper reads seven coordinates off each method's formula: where it acts; the shape of its image; where it starts, at a tip, a smooth point, or off target; its gauge; which changes of basis it respects; whether it can be merged; and what merge-and-restart can reach. <short pause> Three hundred and eighty-two methods from a hundred and thirty-eight papers. Four hundred and three arrows saying which method simulates which, each with an explicit map, and four hundred and eighty-five obstructions saying which can't.

**Checked against.** `theory/axes.json` (the seven coordinates: kind, shape of the image germ, base point, gauge,
covariance group, merge type, rebasing closure); abstract and README counts; Conclusion ("each backed by an explicit map
or a named test").

### A.2 · Predictions and limits  ⏱ 0:37 · `[CARD]`
**On screen.** Two prediction cards from Exposé X, then three limit cards from the Conclusion and Exposé XI.

> A theory should also risk being wrong, so the atlas makes predictions a training run can test. Under Adam, a fair LoRA-plus baseline should multiply alpha by lambda. LoRA's charge should drift by a second-order amount per step of SGD, a cheap diagnostic of what your optimizer adds. <short pause> And it marks its own limits. Reaching a weight isn't the same as learning it. Most results are about images, in exact arithmetic. And words like gauge and Noether are analogies; the paper says exactly where each one stops.

**Checked against.** Exposé X "In brief"; Conclusion ("Expressivity is not learnability"; set-level, exact arithmetic);
Exposé XI table of borrowed words.

---

## Coda (14:30–15:15)

### C.1 · Three questions  ⏱ 0:25 · `[STORY]`
**On screen.** The sea rises over the shore; the legend from I.4 returns as three questions, one at a time.

> So the next time a new method appears, you don't have to reach for the hammer. Ask three questions. What can its map reach? How do its fibres meet the optimizer? And what happens when the base point moves? <short pause> Often, the answer is already under the water.

**Checked against.** Conclusion, last sentence.

### C.2 · End card  ⏱ 0:20 · `[STORY]`
**On screen.** Contours return. Title; Xiaoyu Li, Zhizhou Sha, Chiwun Yang, Dai Shi; "with Claude Opus 5.5 (Anthropic),
in the spirit of Grothendieck's rising sea"; xiaoyulics.com/peft-atlas; arXiv link once assigned.

> The proofs, the full catalogue, and twenty interactive figures are in the paper and on the companion site. The link is below. Thanks for watching.

---

## Production notes

**Voice.** `video/tools/tts.py` defaults: model `gemini-3.8-flash-tts`, voice `sulafat`, style "gentle, warm and
soft-spoken; an unhurried young woman explaining mathematics to a friend, with a light smile in her voice". One request
per beat; the sentence gaps found by silence detection give the cue times for the animation.

**Pronunciation spellings.** The narration is written for the ear; subtitles use the printed forms.

| Narration | Subtitle | | Narration | Subtitle |
|---|---|---|---|---|
| theta-zero | θ₀ | | Q-LoRA | QLoRA |
| LoRA-plus | LoRA+ | | Re-LoRA | ReLoRA |
| I-A-cubed | (IA)³ | | B-transpose-B minus A-A-transpose | BᵀB − AAᵀ |

**Numbers on screen.** Every number in a `[SIM]` beat comes from a script in `video/sim/` run at render time; where the
paper reports the same quantity ([Num] notes), the two must agree before the beat is rendered.

**Credits.** Author lists and years on screen follow `paper/bib/refs.bib`.

**Music.** Quiet piano or pad under the cold open, the title, the chapter cards and the coda; none under dense
explanation.
