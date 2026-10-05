# Interactive visualisations

Twelve figures. Each one is tied to a numbered result in `theory/framework.md`, computes every number it shows, and renders a meaningful default state with no user input ("complete at rest"). Builders follow `site/SPEC.md`:
- classic scripts only, registered with `Atlas.register`;
- D3 v7 and MathJax 3 are the only libraries;
- linear algebra comes from `Atlas.LA` (`randn`, `mul`, `T`, `hadamard`, `kron`, `svals`, `symEig`, `numRank`, `randOrth`), with seeded randomness from `Atlas.rng`;
- colours come from tokens: `--moss` = holds or mergeable, `--seal` = obstruction, `--ochre` = highlight;
- layouts must work at 380px, and every animation needs a static reduced-motion frame.

All numerics use `Float64Array` and a fixed seed, so screenshots are reproducible.

---

## 1. The Apex and the Smooth Point

**Concept.** Zero-initialised LoRA, and HRA's paired initialisation, start at the *vertex* of a cone, so their first-order reach is deficient. Split initialisations (PiSSA, LoRA-GA, CorDA) start at a *smooth point*. QLoRA starts off the base point. *(Thm II.2, Prop II.3, Thm II.7.)*

**Exact math.**
- *3D illustration: the symmetric slice.* Write $S=\begin{psmallmatrix}x+y&z\\z&x-y\end{psmallmatrix}\cong(x,y,z)\in\mathbb R^3$. Rank $\le1$ is the double cone $x^2=y^2+z^2$. The illustrative one-parameter-per-factor method is $\Delta=c\,vv^\top$ with $v=(\cos\phi,\sin\phi)$:
  - at $c=0$ (apex), $S_1=\mathbb R\,v_0v_0^\top$ is a line;
  - at $c_0\ne0$ (smooth point), $S_1=\mathrm{span}\{v_0v_0^\top,\ v_0\dot v^\top+\dot vv_0^\top\}$ is the tangent plane.
- *Split.* Move the apex to $W_0-P$ with $P=c_0v_0v_0^\top$. The base point $W_0$ then lies on the smooth sheet.
- *QLoRA.* Draw the apex at $\kappa W_0$, with the defect vector $e=\kappa W_0-W_0$ (κ rounds to a 0.25 grid).
- *Counters (exact formulas, not the slice).* For user-chosen $(m,n,r)$:
  - zero-init: $\dim S_1=mr$;
  - split: $r(m+n-r)$;
  - deficit: $r(n-r)$;
  - HRA paired init ($k=r/2$): $kn-\tfrac{k(k+1)}2$ against $rn-\tfrac{r(r+1)}2$.

**User manipulates.** An init toggle (LoRA zero-init / PiSSA split / QLoRA); the angle $\phi$ of $v_0$ (drag); a target point $T$ (drag); sliders for $m,n,r$; an HRA toggle for the counters.

**User sees.**
- The cone, rendered as rulings, with the base point marked.
- $S_1$ drawn as a line (apex) or a translucent plane (smooth point).
- The first-step arrow, which is the orthogonal projection of $T-W_0$ onto $S_1$.
- $T$ coloured moss if it lies in the image and seal if not.
- The counters panel, with a sentence such as "zero-init loses 65,280 of 130,816 first-order directions" at $4096^2$, $r=16$.

**Why it is correct.**
- The cone equation is $\det S=0$.
- $S_1$ is computed from the analytic derivative of $c\,vv^\top$.
- The counters are Thm II.2(b) and Thm II.7(b). The latter was verified by Jacobian rank: $8$ vs $15$ at $n=9,r=2$; $15$ vs $26$ at $r=4$; see `framework_checks.py`.
- The caption states that the 3D picture is the symmetric slice, not $\mathbb R^{m\times n}$.

**Implementation.** SVG with a fixed oblique projection. Drag rotates about the vertical axis; under reduced motion the view stays at a fixed 30° angle. The cone is drawn as 24 rulings plus 3 latitude ellipses, the plane as a clipped polygon, and the projection uses a $3\times3$ least-squares solve. Counters are rendered with MathJax.

---

## 2. Hyperparameter Orbit Explorer ("LoRA+ is α in disguise")

**Concept.** LoRA's four knobs $(\alpha,\eta_A,\eta_B,\sigma_A)$ live on the orbit space of a 2-torus. Moving along an orbit leaves training unchanged; under Adam with $\varepsilon=0$, no weight decay, no clipping and a shared schedule, LoRA+ is exactly α-rescaling. The LoRA/Adam case is due to Schulman et al. (2025), and the caption says so. *(Thm III.10, Cor III.11.)*

**Exact math.**
- Toy regression: $W_0\in\mathbb R^{8\times6}$, $X\in\mathbb R^{6\times40}$, $Y=W^\star X$, loss $\tfrac1{2N}\|(W_0+sBA)X-Y\|^2$, $r=2$, $B_0=0$, $A_0=\sigma_AU$ with $U$ fixed by the seed.
- Optimizer: Adam with $\beta=(0.9,0.999)$, bias correction, and $\varepsilon\in\{0,10^{-8},10^{-6}\}$ (the convention $0/0=0$ at $\varepsilon=0$), for 300 steps. SGD is optional.
- Displayed invariants: under Adam, $I_1=s\eta_A\eta_B$ and $I_2=\sigma_A/\eta_A$; under SGD, $s^2\eta_A\eta_B$ and $\sigma_A^2/\eta_A$.

**User manipulates.**
- Four log sliders for $s=\alpha/r$, $\eta_A$, $\eta_B$ and $\sigma_A$.
- Two "orbit handles" $(c_A,c_B)$ that move all four sliders together while preserving $(I_1,I_2)$.
- A **LoRA+ ↔ α** button, which jumps between $(s,\eta_A,\lambda\eta_A)$ and $(\lambda s,\eta_A,\eta_A)$.
- Toggles for $\varepsilon$, decoupled weight decay, global-norm clipping, a shared warm-up schedule, "rescale $\varepsilon$ and decay per block", and the optimizer.

**User sees.**
- Two overlaid loss curves (reference and current) and a meter for $\max_t\|W_t-W_t'\|_F$.
- On an orbit with every torus-breaking toggle off, the meter reads exactly `0` and the curves coincide pixel for pixel.
- Turning on a shared $\varepsilon$, shared decay or clipping makes the meter jump, which visualises the [Pred] about where LoRA+ gains can come from. A shared warm-up, or $\varepsilon$ and decay rescaled per block ($\varepsilon_i/c_i$, $\lambda_i/c_i$), keeps the meter at `0`.

**Why it is correct.**
- Thm III.10: the parameter rescaling $q_i\mapsto c_iq_i$ with compensating $s$ is a slice isomorphism, and Adam at $\varepsilon=0$ is invariant to per-block gradient scale; shared multiplicative schedules commute with the torus, and per-block $\varepsilon$ and decoupled decay transform as $\varepsilon_i/c_i$, $\lambda_i/c_i$.
- Exactness is verified in `framework_checks.py`: the maximum difference is $0.0$, against $0.93$ for the control.

**Implementation.** Plain JS loops over `Float64Array`; 300 steps on $8\times6$ matrices take under 30 ms. Recompute on input (debounced 50 ms). The curves are drawn on Canvas, with a log-scale y-axis drawn by D3 in an overlaid SVG.

---

## 3. Noether Hyperbolas and the Crossover

**Concept.** Under gradient flow, the non-compact half of LoRA's gauge produces a conserved charge (after Zhao et al. 2023). Each weight-decay convention pulls the trajectory to its own balanced slice: per-time decay to $\Phi=0$, learning-rate-coupled decay to $\mu=B^\top B-AA^\top=0$. The charge fixes a closed-form crossover between LoRA-FA-like and balanced dynamics. *(Thm III.5, Cor III.6, Thm III.9.)*

**Exact math.**

*Panel A (scalar LoRA).* Use $w=s\,ba$ and $\ell=\tfrac12(w_0+sba-w^\star)^2$.
- The level sets $b^2/\eta_B-a^2/\eta_A=\Phi$ are hyperbolas.
- Dynamics: gradient flow (RK4, $h=10^{-3}$), discrete GD, or Adam.
- Weight decay comes in two conventions:
  - per-time decay $-\lambda(a,b)$, which gives $\Phi(t)=e^{-2\lambda t}\Phi_0$;
  - learning-rate-scaled decay $-\eta(\cdot+\lambda\cdot)$, which gives the closed form only when $\eta_A=\eta_B$; at other rates its equilibria still lie on the unweighted slice $\mu=0$.

*Panel B (matrices).* Use $m=n=6$, $r=2$, with $A_0$ having orthonormal rows scaled by $\sqrt c$, so that $A_0A_0^\top=cI$ exactly.
- The closed-form functions are $f_\kappa(x)=\tfrac12(\sqrt{\kappa^2+4x}-\kappa)$ and $g_\kappa=f_\kappa+\kappa$, with $\kappa=c/\eta_A$.
- The crossover sits at $\sigma^*=\mu\kappa/2$ with $\mu=s\sqrt{\eta_A\eta_B}$.
- The singular values of $\Delta W(t)$ come from integrating the matrix flow.

**User manipulates.** In Panel A: $(a_0,b_0)$ by drag, $\lambda=\eta_B/\eta_A$, the decay coefficient and its convention, and the optimizer. In Panel B: $s$, $c$, $\lambda$, and a PiSSA preset ($\kappa=0$).

**User sees.**
- *Panel A.* The trajectory rides its hyperbola, and a Φ meter beside it reports:
  - constant under the flow;
  - an $O(\eta^2)$ wobble under GD;
  - drift under Adam;
  - exponential decay under per-time decay;
  - under scaled decay with $\lambda\ne1$, visible failure of the closed-form $\Phi$ law, while a second meter shows $\mu\to0$.
- *Panel B.* The curves $f_\kappa$ and $g_\kappa$ against $\sigma$, with $\sigma^*$ marked and the bands "one-sided (LoRA-FA)" and "balanced (Arora–Cohen–Hazan)" shaded. Simulated singular values move across $\sigma^*$ as dots. A residual meter compares the closed-form $\dot W$ with the integrated one ($\sim10^{-12}$).

**Why it is correct.** Thm III.5 gives conservation for any cotangent signal of the form $\Lambda D\rho^\top\gamma$ under gradient flow. Cor III.6(b) was checked numerically: $10.7$ relative error for the closed form under scaled decay at unequal rates, and $2\times10^{-4}$ (Euler error) for the valid conventions; at $\eta_B/\eta_A=4$ scaled decay converges to $\mu=0$ to $10^{-12}$ and per-time decay to $\Phi=0$. The Thm III.9 formula matches direct integration to $10^{-12}$. The caption credits Tarmoun et al. 2021 and Min et al. 2021.

**Implementation.** SVG for both panels. The hyperbolas are drawn analytically, with branches for both signs of Φ. RK4 runs in JS, and the matrix flow uses $6\times6$ products with `Atlas.LA.mul`.

---

## 4. Gauge Lab: optimizers and merges

**Concept.** The gauge does not act on the model; it acts on *optimizers* and on *merging rules*. Adam sees only signed permutations, SGD sees rotations, and undamped Riemannian preconditioning sees all of $GL_r$. Factor-wise merging of independently trained modules is not a function of the adapters. *(Thm III.3, Prop III.4, Thm III.12, Prop V.4.)*

**Exact math.**

*Tab A.* Fix random $B\in\mathbb R^{5\times2}$, $A\in\mathbb R^{2\times4}$ and $G$. Parametrise $g=R_\theta\,\mathrm{diag}(e^{t_1},e^{t_2})\,\begin{psmallmatrix}1&h\\0&1\end{psmallmatrix}$, with buttons for the swap and a sign flip. Then compute, for $(Bg^{-1},gA)$, the first-order $\delta W$ of:
- SGD: $GA^\top A+BB^\top G$;
- Adam's first step: $\mathrm{sign}(\nabla_B)A+B\,\mathrm{sign}(\nabla_A)$;
- ScaledGD: $P_UG+GP_V$;
- LoRA-Pro: $P_UG+GP_V-P_UGP_V$;
- "Adam + per-channel rates": rates rescaled by $\mathrm{diag}(g)$ when $g$ is diagonal (at $\varepsilon=0$; with $\varepsilon>0$ the toggle also rescales $\varepsilon$ per channel).

*Tab B.* Two modules $(B_i,A_i)$ with weights $w_1,w_2$. A gauge $(cB_2,A_2/c)$, or a general $g$, acts on module 2.
- The LoraHub merge $(\sum w_iB_i)(\sum w_iA_i)$ changes by $w_1w_2[(c-1)B_2A_1+(c^{-1}-1)B_1A_2]$ (two modules; with $N$ modules the change is $\sum_{l\ne2}w_lw_2[(c-1)B_2A_l+(c^{-1}-1)B_lA_2]$).
- Task arithmetic $\sum w_iB_iA_i$ is invariant.

**User manipulates.** In Tab A: the rotation, stretch and shear sliders and the permutation/sign buttons. In Tab B: the gauge knob, and a "share frozen $A$" toggle.

**User sees.**
- *Tab A.* A defect bar $\|\delta W(g)-\delta W(I)\|$ per optimizer.
  - SGD is zero only when $t_1=t_2=0$ and $h=0$.
  - Adam is zero only on signed permutations, or on diagonals with the per-channel toggle on.
  - ScaledGD and LoRA-Pro stay at zero throughout. The caption says this is the first-order $W$-velocity: damped scaled GD and LoRA-Pro's discrete parameter update with the Sylvester-optimal $X$ are only $O(r)$-equivariant (Thm III.12).
  - An ellipse shows the image of the unit circle of a 2D family of $G$ under the cometric $K$: rotations leave it fixed, and stretches deform it (SGD).
- *Tab B.* Heatmaps of $B_2$ and $A_2$ churn while $B_2A_2$ stays fixed. The task-arithmetic panel stays still (moss); the LoraHub panel moves (seal), with a ‖change‖ meter that matches the closed form. The "share A" toggle turns it green.

**Why it is correct.** Thm III.12 gives the group, including the proof that first-step sign descent forces $B_r$. Numerically the defects are $10^{-15}$, $11.3$ and $6.6$, and $5\times10^{-16}$ for the joint torus. Prop V.4's formula is exact algebra.

**Implementation.** `Atlas.LA` throughout, with `inv` via a $2\times2$ closed form. Heatmaps are drawn on Canvas, defect bars in SVG.

---

## 5. Gram Lab (Erlangen)

**Concept.** Each method preserves a complete invariant, which is what it can never change.
- Exactly orthogonal input-side methods preserve $K_{\rm out}$ and the spectrum; HF's default Cayley–Neumann OFT only approximately.
- BOFT preserves the cosines when all output-scale entries share a sign; mixed signs keep only $\lvert\cos\rvert$.
- Torus methods keep row lines while their scales are nonzero.
- Hadamard methods keep the zeros of $W_0$ (they can create new ones).
- HRA preserves both "Gram fixed" and "rank $\le r$": it is the meet.

*(Thm II.5, Cor II.6, Thm II.7(a), Props II.8–II.9.)*

**Exact math.** $W_0\in\mathbb R^{4\times4}$, invertible and with one zero entry. Neurons (rows) are drawn as arrows in the 3D projection given by $W_0$'s top three right singular vectors. Live readouts: $K_{\rm out}=WW^\top$, $K_{\rm in}=W^\top W$, the normalised $K_{\rm out}$ (cosines), the singular values, $\operatorname{rank}(W-W_0)$, and the zero pattern.

| method | map | parameters |
|---|---|---|
| OFT | $W_0\,\mathrm{Cay}(\Omega)^\top$ (exact Cayley; a switch selects HF's 5-term Cayley–Neumann map) | $\Omega$: 6 skew sliders |
| output rotation | $\mathrm{Cay}(\Omega)W_0$ | $\Omega$ |
| BOFT | $\mathrm{diag}(s)W_0R^\top$ | $s$, $R$ |
| (IA)$^3$ | $\mathrm{diag}(\ell)W_0$ | $\ell$ |
| RoAd | scaled $2\times2$ rotations on the row pairs $(1,2),(3,4)$ | angles, scales |
| DoRA | $\mathrm{diag}(\mu)N(W_0+BA)$ | $\mu$, $B$, $A$ |
| HiRA | $W_0\odot(J+BA)$ | $B$, $A$ |
| LoRA | $W_0+BA$ | $B$, $A$ |
| HRA | $W_0H_{u_1}H_{u_2}$ | $u_1$, $u_2$ |
| qGOFT | a $GL_2$ block | 4 entries |

**User manipulates.** The method selector and its parameter sliders, plus a **reset to identity** button.

**User sees.** Heatmaps that *glow* when locked, with badges: "neuron Gram fixed", "cosines fixed", "|cosines| fixed", "spectrum fixed", "row lines fixed", "zeros of $W_0$ kept", "rank ≤ r".
- OFT lights Gram and spectrum; with the Cayley–Neumann switch on, both badges go dark ($R^\top R=(I-Q^4)^2$).
- BOFT lights cosines when all $s_i$ share a sign, and only $\lvert\cos\rvert$ when the signs are mixed.
- (IA)$^3$ lights row lines and zeros-of-$W_0$ while every $\ell_i\ne0$.
- HRA lights Gram, spectrum and rank ≤ 2 together, and an "intersection" banner names it LoRA ∩ $W_0\cdot SO(n)$, an image-level meet.
- qGOFT lights nothing.

**Why it is correct.** Every badge is a numerical equality test to $10^{-10}$ on the computed invariant, and the expectations follow from Thm II.5 and Cor II.6. The figure states that HRA's apex property needs $r\le n-2$ (here $r=2$, $n=4$) and *shows the apex*: at the paired init, an "$S_1$ dimension" readout shows $3$ against generic $5$. That is $kn-k(k+1)/2=4-1=3$ against $rn-r(r+1)/2=8-3=5$.

**Implementation.** Arrows in SVG with a fixed oblique projection; heatmaps on Canvas with a diverging scale from tokens. The Cayley map uses a $4\times4$ solve, and the eigen/SVD routines come from `Atlas.LA`.

---

## 6. Plug Workbench (the hole and the plug)

**Concept.** Cut $W_0$ out of the network and a *hole* $V\to U$ remains. A method is a *plug*: a small tensor network with frozen and trainable boxes. Parameter count, rank, gauge, dimension and mergeability are all invariants of the plug. *(Def I.5, Thm II.10, Prop II.12, Thm VI.3.)*

**Exact math.** A plug is a formal sum of paths built from a grammar. Product: matrix boxes in series. Hadamard: a copy spider, two branches, and a merge spider. Kronecker: a reshape. Diagonal spider-with-state. $\sigma$ boxes. The plug computes:
- $\lvert D\rvert$, the total size of the trainable boxes;
- the degree, the maximum number of trainable boxes on any path;
- the **min-cut** rank bound, by brute force over vertex bipartitions (at most $2^{10}$);
- the **gauge wires**, the internal wires joining two unconstrained trainable boxes;
- $d(M)$, the numerical Jacobian rank at $m=n=8$, at a random point with central differences and an SVD threshold of $10^{-8}\sigma_{\max}$ in Float64;
- the **merge verdict**: M∞ if a $\sigma$ sits on the activation path, else M1;
- the **nearest named method**, by matching the signature (degree, $d$, rank bound, gauge type, frozen-box types) against a table of about 25 presets.

**User manipulates.** Drag boxes from a palette:
- trainable matrix, diagonal or skew;
- frozen $W_0$, random, DFT, SVD-frame or summing box;
- copy and merge spiders, reshape, $\sigma$.

Wire them into the hole. Presets include LoRA, VeRA, LoRA-XS, LoHa, LoKr, HiRA, DoRA, GraLoRA, MiSS and a Houlsby adapter.

**User sees.** Live readouts. The minimum cut is highlighted with its weight, gauge wires glow, and the equal-budget comparison against LoRA is shown. A badge reads "this is MiSS = LoRA-FA with a summing $A$" or "new gadget".

**Why it is correct.** The rank bound is Thm II.10. Generic Jacobian rank equals $\dim\mathrm{Im}$ (Def I.5), and random Float64 points are generic with probability one. The LoHa and GraLoRA presets reproduce the [Num] values in Prop II.12.

**Implementation.** SVG canvas with d3-drag. The grammar is restricted (series, parallel sum, Hadamard diamond, Kronecker) so that contraction is a short JS interpreter over small matrices; $\lvert D\rvert\le200$ keeps the Jacobian at most 200 forward passes.

---

## 7. Rank Is Not Volume

**Concept.** At equal budget, "high-rank" methods reach high rank on thin sets. When $2\rho^2<m+n-1$, LoHa gives up $(m+n-1)-2\rho^2$ dimensions relative to LoRA$_{2\rho}$, in exchange for rank up to $\rho^2$ when $\rho\ge3$, and for $\rho\le2$ it is strictly dominated; when $2\rho^2\ge m+n-1$ it matches or beats LoRA$_{2\rho}$ on both counts. GraLoRA matches LoRA's dimension and multiplies its maximum rank by $k$, but caps each block at rank $r/k$, so neither image contains the other. *(Prop II.12, Thm II.10.)*

**Exact math.** For layer size $m\times n$ and budget $P$:

| method | rank | dimension |
|---|---|---|
| LoRA$_r$, $r=P/(m+n)$ | $r$ | $r(m+n-r)$ |
| LoHa$_{\rho}$, $\rho=P/(2(m+n))$ | $\min(\rho^2,m,n)$ | $\min\big(mn,\ 2\rho(m+n)-2\rho^2-(m+n-1)\big)$ (equality [Num]) |
| GraLoRA$_k$ | $\min(kr,m,n)$ | $r(m+n-r)$ |
| HiRA$_r$ | $r\operatorname{rk}W_0$ | $r(m+n-r)$ |
| KronA, balanced factors | — | $m_1n_1+m_2n_2-1$ |
| MoRA | up to $\min(m,n)$ | $P$ |
| FourierFT, $n_c=P$ | $\le\min(2n_c,m,n)$ | $P$ minus the number of sampled conjugate pairs |
| C3A | full | $P=mn/b$ |

**User manipulates.** Sliders for $m$, $n$ and $P$, toggles for methods, and a "verify" button that runs an in-browser Jacobian rank at $m,n\le12$.

**User sees.** A scatter with x = maximum rank (log scale), y = $d(M)/(mn)$, and point size = budget. LoHa sits high on x and low on y; GraLoRA sits directly above LoRA's y-level at $k$ times its rank. A caption gives the equal-budget gap $(m+n-1)-2\rho^2$.

**Why it is correct.** The formulas are Prop II.12. The LoHa formula is marked as an upper bound proved and checked numerically, and the verify button re-checks it live.

**Implementation.** D3 scatter in SVG, with the formulas in JS. The verify path reuses the Workbench's Jacobian routine.

---

## 8. Cut the Tensor (Kronecker incomparability)

**Concept.** Unconstrained Kronecker sums are LoRA across a different cut. With equal square splits, a generic rank-one update has *maximal* Kronecker rank, while the identity has Kronecker rank one. With unequal splits the generic rank-one value is $\min(m_1,m_2)\min(n_1,n_2)$, which is still the maximal Kronecker rank when the splits are aligned, and LoKr with inner rank $r'$ sits inside LoRA$_r$ only when $r\ge\min(m_1,n_1)r'$. *(Thm II.11, Prop II.13.)*

**Exact math.** Take a $16\times16$ matrix $\Delta$ with $\mathbb R^{16}=\mathbb R^4\otimes\mathbb R^4$ on both sides, viewed as the 4-leg tensor $\Delta_{(u_1u_2)(v_1v_2)}$. There are three bipartitions:
- $\{U_1U_2\}\mid\{V_1V_2\}$, the ordinary rank;
- $\{U_1V_1\}\mid\{U_2V_2\}$, the Kronecker rank, via the Van Loan–Pitsianis rearrangement $\mathcal R$;
- $\{U_1V_2\}\mid\{U_2V_1\}$.

Each cut's singular values are the `svals` of the corresponding $16\times16$ reshaping.

**User manipulates.** A preset: random $uv^\top$, $I_{16}$, $C\otimes D$, a sum of 2 Kronecker terms, a LoRA$_2$ update, LoHa, HiRA, or LoKr with inner rank 1 (rank $\le4$, so inside LoRA$_4$). Also a cut selector, a split selector ($4\otimes4$ or $2\otimes8$), and a "rearrange" animation.

**User sees.** Three bar charts. A random rank-one update lights 16 Kronecker singular values; the identity lights one Kronecker value and 16 ordinary ones. A verdict line reads "LoRA$_1$ ∋ this; Kronecker-sum with $k<16$ ∌ this", or the reverse.

**Why it is correct.** $\mathcal R$ is an index permutation, so the reshaped singular values are the operator-Schmidt coefficients (Thm II.11(c)).

**Implementation.** An index map computed once; `Atlas.LA.svals`; bars on Canvas. The animation morphs cell positions in SVG, with a static frame under reduced motion.

---

## 9. Merge Game

**Concept.** Exact merging is certified by local rewriting, a sufficient test with box-local obstructions. The rewrite also says *which* weight absorbs the update, and the answer depends on the activation function. *(Thm VI.3, Lemma VI.2.)*

**Exact math.** Rules R1–R6:
- fusion;
- permutation slide;
- positive-diagonal slide, valid iff $\sigma$ is positively homogeneous;
- spider slide through the Hadamard gate of GLU units;
- cup slide around $\langle q,k\rangle$ (blocked under RoPE unless the scale is constant on rotary pairs);
- norm absorption (the shift needs a bias slot);
- copy slide, which sends an edit to every consumer of a wire.

Each rewrite is also checked numerically. On random weights at $d=8$, the figure compares the adapted block output with the merged-weight output and shows $\|\Delta\|$ (about $10^{-15}$ when legal).

**User manipulates.** A pre-LN transformer block with an activation toggle (GELU / ReLU / SwiGLU) and a RoPE toggle. Drop (IA)$^3$ (on $k$, $v$ or the FFN), SSF, LN-tuning, LoRA, OFT, a Houlsby adapter or a prefix onto a site, then press **rewrite**. A "force merge" button attempts an illegal merge.

**User sees.** Rules animate step by step. The run ends either fused into a highlighted weight (moss) or blocked at the offending σ or softmax box (seal), with the failing rule named. For example, (IA)$^3$ on the FFN fuses into $W_{\rm up}$ under SwiGLU for every sign, into $W_{\rm up}$ under ReLU only for $\ell>0$, and never into $W_{\rm up}$ under GELU; it always fuses into $W_{\rm down}$. A forced merge shows a large $\|\Delta\|$.

**Why it is correct.** Each rule is an identity of continuous, piecewise-smooth maps, with the side conditions of Thm VI.3 (position-independent paths, unshared target boxes, bias slots). R3's condition is positive homogeneity, verified numerically for ReLU versus GELU in the string-diagram checks; with RoPE on, the query-scaling slide onto $W_K$ leaves a nonzero residual unless $\ell$ is pair-constant.

**Implementation.** An SVG block diagram and a small rule engine, a table keyed by (method, site, σ) that lists the rewrite sequence. The numeric check uses `Atlas.LA` with a tanh-approximated GELU and an exact SiLU.

---

## 10. Rebasing Reach

**Concept.** What merge-and-restart can and cannot reach. ReLoRA climbs to rank $Kr$, fixed linear shapes with frozen frames are idempotent, exact-Cayley block OFT is stuck, the butterfly connects only at full depth ($b\,2^{m'-1}=n$), and re-HRA climbs inside $SO(n)$. *(Thm V.3.)*

**Exact math.**

*Tab A.* Accumulate $\sum_{k\le K}B_kA_k$ with random rank-$r$ factors and show its numerical rank against $\min(Kr,m,n)$. A LoRA-XS comparison keeps a fixed frame (stuck at $r^2$ dimensions) or recomputes the frame each cycle.

*Tab B.* Coordinates $1..n$ ($n=16$) are nodes. Each factor adds hyperedges:
- block OFT with block size $b$ (a fixed partition);
- butterfly level $\ell$ ($i\sim i\oplus2^\ell$);
- random pairs;
- Givens planes.

Union-find gives the components $C$. The generated Lie algebra has dimension $\sum_C\binom{\lvert C\rvert}2$, against $\binom n2$, and the number of preserved partial Grams equals the number of components.

*Tab C.* For a random target $R\in SO(n)$ with $\operatorname{rank}(R-I)=k$, re-HRA needs $\lceil k/r\rceil$ cycles. A product of HRA blocks is shown with $\operatorname{rank}(R_{1..K}-I)\le Kr$.

**User manipulates.** $K$, $r$, the factor types and block size, and the frame-recompute toggle.

**User sees.**
- Tab A: a rank staircase.
- Tab B: components merging until "SO(n) reached"; with butterfly levels the banner appears only at full depth, and HF's default (one level) is labelled "block OFT".
- Tab C: a cycle counter, with rank bars per cycle.

**Why it is correct.** These are Thm V.3(a)–(c) and corollaries 1, 2, 4 and 5. The ranks are computed with `numRank`.

**Implementation.** SVG with a circular node layout (deterministic, no force simulation), union-find in JS, and `Atlas.LA` for the rank computations.

---

## 11. Prefix Is a Gate

**Concept.** Attention is a monoid homomorphism followed by a perspective map. So a prefix acts as a gated parallel adapter: at the layer where it acts it rescales all content attention uniformly and cannot reorder it (higher layers can change, which the caption states). LLaMA-Adapter's separate softmax leaves content attention untouched. *(Thm VI.5.)*

**Exact math.**
- One query $q\in\mathbb R^4$, six content tokens $(k_j,v_j)$, and $\ell$ prefix tokens.
- $Z_q(S)=\sum_{s\in S}e^{\langle q,k_s\rangle/2}$, with $\lambda_q=Z_q(P)/(Z_q(P)+Z_q(C))$ and output $(1-\lambda_q)\mathrm{Attn}_q(C)+\lambda_q\mathrm{Attn}_q(P)$.
- LLaMA-Adapter mode outputs $\mathrm{Attn}_q(C)+g\,\mathrm{Attn}_q(P)$.
- Single-token panel: plot output coordinate 1 along $x=tx_1$, $t\in[-3,3]$. The original head gives a straight line; the prefixed head gives a curve.

**User manipulates.** Prefix length, a prefix logit-strength slider, the query direction (dial), LLaMA-Adapter mode, and the gate $g$.

**User sees.**
- Content attention bars before and after, with a "ratio" line across them that stays flat.
- The $\lambda_q$ readout.
- The output point moving along the segment from $\mathrm{Attn}(C)$ toward the convex hull of the prefix values (2D projection).
- In LLaMA-Adapter mode the content bars do not move.
- The single-token plot visualises the mergeability obstruction: no weight setting produces a curve.

**Why it is correct.** These are exact identities (Thm VI.5(b)–(d)). The non-affinity in (f) shows as a nonzero second difference, which the figure prints.

**Implementation.** SVG bars and a 2D scatter with a PCA projection of the values, all computed in JS.

---

## 12. The Periodic Table of PEFT (atlas hub)

**Concept.** Every method has seven coordinates, and its arrows (simulations) connect it to the rest. *(Exposé VIII; `axes.json`; `rosetta.json`.)*

**Exact math.** The data come from Exposé VIII.B and `axes.json`.
- Rows are kind and coupling: R-additive, R-action, R-Hadamard, R-selective, E, E°, B, P, F$_T$.
- Columns are shape: Lin, Cone, Cone*, Orb, Sat, Meet, Fun.
- Arrows come from the arrow rules A1–A5 together with the obstruction tests O1–O3.

**User manipulates.**
- Filter chips: base point (regular / apex / neutral / defect / unpointed), gauge, covariance, merge, rebasing.
- Hover a method to see its seven coordinates, formula and budget.
- Click a method to open a mini-Hasse panel of its in- and out-arrows, each labelled with its $h$. Obstructed pairs are listed in seal with the failing test.
- "Open in Workbench / Gram Lab / Apex" links.

**User sees.** A dense, colour-coded table (method chips coloured by `Atlas.kindColor`) that doubles as the site's navigation. Filtering "apex ∧ M1" lights LoRA, DoRA, VeRA, LoHa, HiRA and HRA. Filtering "idempotent ∧ regular" lights the linear methods.

**Why it is correct.** Every cell is generated from `axes.json`, which mirrors the proved table, and every arrow carries an explicit $h$ or a cited obstruction.

**Implementation.** CSS grid plus SVG mini-diagrams. Data are compiled into `data/atlas-data.js` from `axes.json` and the research catalog. The layout is responsive: below 600px it collapses to a filterable list.
