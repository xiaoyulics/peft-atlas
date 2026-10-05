# PEFT as pointed 2-cells: Para, gauge and the lens duality

*Proposal for the "Para, 2-cells and parametric lenses" lens.*

**Status tags.** **[Thm]** and **[Prop]**: proved here (a Thm is the harder kind). **[Obs]**: true but trivial; it organises things and nothing more. **[Sketch]**: proof sketch only. **[Pred]**: a prediction someone can falsify. **[Analogy]**: not a theorem. Numerical claims marked ✓ are checked by `theory/proposals/para-lens-checks.py`, which integrates the ODEs and agrees to relative error $\le 10^{-11}$.

**Notation.** We use the convention $y=Wx$ with $W_0\in\mathbb R^{m\times n}$. Write $\mathcal M_{\le r}=\{X:\operatorname{rank}X\le r\}$ and $\mathcal M_r$ for rank exactly $r$. LoRA is $\rho(B,A)=W_0+sBA$ with $B\in\mathbb R^{m\times r}$, $A\in\mathbb R^{r\times n}$, scale $s$ ($\alpha/r$ in LoRA, $\alpha/\sqrt r$ in rsLoRA), and $G=\nabla_W L$.

---

## 1. Thesis

Fine-tuning moves a point. PEFT picks the space the point may move in and the metric it moves with. A pretrained network is a **pointed** 1-cell $F=(P,\theta_0,f)$ of $\mathbf{Para}$. A PEFT method is a **pointed 2-cell** $\rho:(Q,q_0)\to(P,\theta_0)$ into $F$. An architectural method is a pointed 2-cell into a *conservative extension* $\tilde F$ whose restriction along a neutral point is $F$. Backpropagation is the 2-functor $\mathbf{Para}(\mathsf R)$ into parametric lenses, so $\rho$ acts on both sides of the lens. On the tangent side, its image $\operatorname{im}\rho$ is what the method can **express**. On the cotangent side, the pulled-back gradient $\mathsf R[\rho]$ induces a **cometric** $K_q=D\rho_q D\rho_q^{\top}$ on weight space, and that cometric is how the method **moves**. Two further invariants complete the picture: the **gauge group** of $\rho$'s fibres (its Cartan split separates isometries, which optimisers may ignore, from conserved Noether charges, which they may not) and the **merge type** (whether the trained family lifts back through the frozen architecture). Together the four classify the field. Cotangent-side methods (GaLore, Flora, LISA, MeZO) are lens 2-cells *outside* the image of $\mathbf{Para}(\mathsf R)$. We prove they are exactly forward methods whose anchor is re-based over time (Thm 3.10).

---

## 2. Definitions

### 2.1 Three ambient categories, each for a reason

- **$\mathbf{Euc}$** (Euclidean spaces, smooth maps) is where **dynamics** lives. It is cartesian and a Cartesian reverse-derivative category with $\mathsf R[f](x,y)=J_f(x)^{\top}y$. We need the inner product, because gradient descent is not invariant under general diffeomorphisms (Prop 3.4).
- **$\mathbf{SA}$** (semialgebraic sets and maps) is where **expressivity** lives. Every $\rho$ below is polynomial or rational. By Tarski–Seidenberg, $\operatorname{im}\rho$ is then semialgebraic, and its dimension equals the generic rank of $D\rho$. Images such as $\mathcal M_{\le r}$ are singular varieties, so a category of manifolds is the wrong home.
- **$\mathbf{Set}$** (or diffeological spaces, which are cartesian closed) is used for **mergeability**, which needs currying $f\mapsto f^\flat:P\to B^A$.

### 2.2 Para and pointed Para

**Definition 2.1 (Para).** Let $(\mathcal C,\times,1)$ be cartesian. A 1-cell $A\to B$ of $\mathbf{Para}(\mathcal C)$ is a pair $(P,\ f:P\times A\to B)$. Composition is $(Q,g)\circ(P,f)=(Q\times P,\ g\circ(Q\times f))$. A 2-cell $(P,f)\Rightarrow(Q,g)$ is a map $r:Q\to P$ with $g=f\circ(r\times A)$. Conventions on direction vary in the literature; ours has the 2-cell point from old learner to new, while $r$ runs backwards on parameters. The result is a bicategory, with associators from those of $\times$.

**Definition 2.2 (pointed Para).** $\mathbf{Para}_*(\mathcal C)$ is the Para construction for the actegory in which the pointed objects $1/\mathcal C$ act on $\mathcal C$ through the forgetful functor (Capucci–Gavranović). Its 1-cells are triples $(P,\theta_0,f)$, composition multiplies the points, and 2-cells are pointed maps. A **pretrained model** is a pointed 1-cell $F=(P,\theta_0,f)$. **Fine-tuning** is a path $\theta(t)$ in $P$ with $\theta(0)=\theta_0$.

### 2.3 Methods

**Definition 2.3 (reparametrisation method).** A PEFT method on $F$ is a pointed 2-cell into $F$: a triple $(Q,q_0,\rho)$ with $\rho(q_0)=\theta_0$. Applying it gives $F^\rho=(Q,q_0,f\circ(\rho\times A))$. The methods form a category $\mathbf{Rep}(F):=(1/\mathcal C)/(P,\theta_0)$, the pointed slice. A morphism $h:\rho\to\rho'$ is a pointed map with $\rho'\circ h=\rho$, read "$\rho'$ simulates $\rho$". A method with $\rho(q_0)\neq\theta_0$ (QLoRA) is an object of the *unpointed* slice over $P$. Its **pointedness defect** is $\delta=\rho(q_0)-\theta_0$.

**Definition 2.4 (extension).** An extension of $F$ is a pointed 1-cell $\tilde F=(\tilde P,\tilde\theta_0,\tilde f)$ together with a pointed 2-cell $\iota:\tilde F\Rightarrow F$, that is, $\iota:(P,\theta_0)\to(\tilde P,\tilde\theta_0)$ with $\tilde f\circ(\iota\times A)=f$. Typically $\tilde P=P\times P'$ and $\iota(\theta)=(\theta,p'_0)$, where $p'_0$ is the **neutral point**. *The pretrained model is a reparametrisation of the extended one.* An extension-type method is a triple $(\tilde F,\iota,\rho)$ with $\rho$ a pointed 2-cell into $\tilde F$. Not every architectural method has a neutral point: soft prompts and prefixes change the output at initialisation. Such methods are extensions *without* $\iota$, and they have a pointedness defect.

**Definition 2.5 (mergeable).** A method $(\tilde F,\iota,\rho)$ is **mergeable** if there is $\mu:Q\to P$ with $\tilde f\circ(\rho\times A)=f\circ(\mu\times A)$. Equivalently, in $\mathbf{Set}$, $\tilde f^\flat\circ\rho$ lifts along the currying $f^\flat:P\to B^A$. **Box-level** mergeability is the same condition for the one box at which the insertion is made.

**Definition 2.6 (dependent 2-cell).** A dependent 2-cell is a map $r:Q\times A\to P$, which yields $(Q,\ f\circ\langle r,\pi_A\rangle)$. Here the weights depend on the input. These are 2-cells of the coKleisli bicategory of the reader comonad $A\times-$. Examples: mixtures of LoRA experts and AdapterFusion.

### 2.4 Lenses, backprop, cotangent methods

$\mathbf{Lens}(\mathcal C)$ has objects $(X,X')$ and morphisms $(f,f^\sharp)$ with $f:X\to Y$ and $f^\sharp:X\times Y'\to X'$. Cruttwell–Gavranović–Ghani–Wilson–Zanasi (CGGWZ) show that $f\mapsto(f,\mathsf R[f])$ is a product-preserving functor $\mathsf R:\mathcal C\to\mathbf{Lens}(\mathcal C)$. It therefore induces $\mathbf{Para}(\mathsf R):\mathbf{Para}(\mathcal C)\to\mathbf{Para}(\mathbf{Lens}(\mathcal C))$. A learner is that image composed with a loss, a learning rate and an optimiser lens on the parameter port. Fong–Spivak–Tuyeras's functor $L_{\varepsilon,e}:\mathbf{Para}\to\mathbf{Learn}$ is the special case of plain gradient descent.

**Definition 2.7 (forward and backward methods).** A **forward** (tangent-side) method acts on the parameter port by a lens in the image of $\mathsf R$, namely $(\rho,\mathsf R[\rho])$. A **backward** (cotangent-side) method acts by a lens *not* of this form: its forward part is $\mathrm{id}_P$, and its backward part is a **compression** $c_t:T^*_\theta P\to S$ into a small state space. It is paired with an **anchor** $a_t:S\to T_\theta P$, and the update is $\theta\leftarrow\theta-\eta\,a_t(\mathrm{opt}(c_t(\nabla L)))$. GaLore uses $c_t=P_t^{\top}$ and $a_t=P_t$. MeZO uses $c_t(g)=\langle z_t,g\rangle$, estimated without a backward pass, and $a_t(\lambda)=\lambda z_t$.

**Definition 2.8 (induced cometric; three equivalences).** Gradient flow on $Q$ moves $\theta=\rho(q)$ by $\dot\theta=-K_q\nabla_\theta L$, where $K_q=D\rho_qD\rho_q^{\top}$. For a backward method under SGD, $K_t=a_t c_t$. Two methods are **expressively** equivalent if their images agree, **first-order** equivalent if $K_{q_0}$ agrees, and **dynamically** equivalent if they produce the same $\theta$-trajectory for every loss.

**Definition 2.9 (recipes, locality).** An architecture is a composite in Para of generator boxes (Linear, Attention, Norm, Embedding). A **recipe** assigns a 2-cell to each box type, with identity meaning frozen, and is extended by horizontal composition. A method is **local** if its 2-cell is such a product, and **global** otherwise.

---

## 3. Results

### 3.1 The order-theoretic skeleton

**Prop 3.1 [Obs] (the interval).** In $\mathbf{Rep}(F)$:
- The frozen model $(1,*,\theta_0)$ is initial, and full fine-tuning $(P,\theta_0,\mathrm{id})$ is terminal.
- In $\mathbf{Set}$, a morphism $\rho\to\rho'$ exists iff $\operatorname{im}\rho\subseteq\operatorname{im}\rho'$. So the poset reflection is the lattice of subsets containing $\theta_0$. The meet is the pullback $Q\times_PQ'$, whose image is $\operatorname{im}\rho\cap\operatorname{im}\rho'$.
- In $\mathbf{Euc}$ the slice is strictly finer. $\rho(t)=t$ and $\rho'(u)=u^3$ have the same image, but a morphism $\rho\to\rho'$ would need $h(t)=t^{1/3}$, which is not smooth.

*Proof.* Immediate. In $\mathbf{Set}$, choose preimages, sending $q_0\mapsto q'_0$. ∎

*What this explains.* Every method sits between frozen and full FT, and expressivity comparisons are subset comparisons. The cube-root example is the generic failure at degenerate base points, where $D\rho'$ drops rank. It returns as LoRA's $B_0=0$ apex (Prop 3.8).

**Prop 3.2 [Prop] (rank is strong monoidal).** For additive methods, define $M\circledast N=(Q_M\times Q_N,(q_M,q_N),\rho_M+\rho_N-\theta_0)$. This is symmetric monoidal up to isomorphism, with the frozen model as unit. The map $h(B_1,A_1,B_2,A_2)=([B_1\,B_2],\,[A_1;A_2])$ is a linear isomorphism $\mathrm{LoRA}_r\circledast\mathrm{LoRA}_{r'}\cong\mathrm{LoRA}_{r+r'}$ in $\mathbf{Rep}(F)$, and it preserves base points. Hence $r\mapsto\mathrm{LoRA}_r$ is a strong monoidal functor $(\mathbb N,+,0)\to(\mathbf{Rep}^+(F),\circledast,\text{frozen})$.

*Proof.* $[B_1\,B_2][A_1;A_2]=B_1A_1+B_2A_2$, and $\mathrm{LoRA}_0$ has a one-point parameter space. ∎

*What this explains.* UniPELT- and MAM-style combinations are $\circledast$-products. A sum of $k$ LoRAs (soups, multi-LoRA) *is* one LoRA of rank $kr$. ReLoRA's $k$ cycles reach $\mathcal M_{\le kr}$ (Thm 3.13).

### 3.2 Backprop on 2-cells, and where naturality fails

**Prop 3.3 [Obs] (pullback; chain rule for cometrics).**
- The pulled-back gradient is $\nabla_q=\mathsf R[\rho](q,\nabla_\theta L)$.
- For $Q'\xrightarrow{\sigma}Q\xrightarrow{\rho}P$, the cometrics compose as $K^{\rho\sigma}_{q'}=D\rho\,K^{\sigma}_{q'}\,D\rho^{\top}$.
- For LoRA, $\nabla_B=sGA^{\top}$ and $\nabla_A=sB^{\top}G$, so $K_{(B,A)}(G)=s^2(GA^{\top}A+BB^{\top}G)$.

*What this explains.* Two methods with the same image can train differently. Prefix tuning with Li–Liang's MLP 2-cell $\sigma$ and P-tuning v2 without it share an image, but their cometrics differ by $D\sigma(\cdot)D\sigma^{\top}$.

**Prop 3.4 [Thm] (gradient descent is natural only along isometric 2-cells).** Let $\varphi:\rho\xrightarrow{\cong}\rho'$ be an isomorphism (a diffeomorphism) in $\mathbf{Rep}(F)$. The two methods induce the same $\theta$-flow for every $L$ iff
$$D\rho'\,(D\varphi D\varphi^{\top}-I)\,D\rho'^{\top}=0$$
along the trajectory. This holds in particular if $\varphi$ is an isometry. For LoRA's gauge $\varphi_g(B,A)=(Bg^{-1},gA)$ at full-rank points (with $n>r$), it holds **iff $g\in O(r)$**.

*Proof.* Put $\ell=L\circ\rho'$. Then $\dot q=-D\varphi^{\top}\nabla\ell$, so $\tfrac{d}{dt}\varphi(q)=-D\varphi D\varphi^{\top}\nabla\ell$. Applying $D\rho'$ and using $\nabla\ell=D\rho'^{\top}\nabla L$ gives the criterion.

For LoRA, equality of $K$ for all $G$ requires $G\,(A^{\top}(g^{\top}g-I)A)=(B(I-g^{-1}g^{-\top})B^{\top})\,G$ for every $G$. That forces both brackets to be the same scalar multiple of the identity. Since $A^{\top}(\cdot)A$ has rank $\le r<n$, the scalar is $0$. Full row rank of $A$ then gives $g^{\top}g=I$. ∎

*What this explains.* No learning functor that uses a fixed Euclidean metric can respect all invertible 2-cells. So "LoRA" is really a family of training dynamics indexed by $GL_r/O(r)$. LoRA+, Riemannian-preconditioned LoRA and LoRA-RITE are attempts to repair this failure of naturality.

**Cor 3.5 [Prop] (restoring naturality) ✓.** Consider the scaled GD of Zhang–Pilanci: $\delta B=-\eta\nabla_B(AA^{\top})^{-1}$ and $\delta A=-\eta(B^{\top}B)^{-1}\nabla_A$. It is $GL_r$-equivariant, and its induced cometric is
$$K(G)=s^2(P_UG+GP_V),$$
where $U$ and $V$ are the column and row spaces of $W-W_0$. This depends only on $W$, so the dynamics descend to the quotient $\mathcal M_r=Q^{\mathrm{fr}}/GL_r$.

*Proof.* Substitute and simplify; equivariance follows because $(AA^{\top})^{-1}\mapsto g^{-\top}(AA^{\top})^{-1}g^{-1}$. ∎

### 3.3 Gauge: the Cartan–Noether split

**Thm 3.6 [Thm] (Cartan–Noether split).** Let $\mathcal G\subseteq GL(Q)$ be a linear group with $\rho\circ g=\rho$, and suppose its Lie algebra is closed under transpose. Split it as $\mathfrak g=\mathfrak k\oplus\mathfrak p$ into skew and self-adjoint parts.

- **(i)** $\exp\mathfrak k$ acts by isometries, so the induced $\theta$-dynamics are invariant along $\mathfrak k$ (Prop 3.4).
- **(ii)** For each $\xi\in\mathfrak p$, the charge $J_\xi(q)=\tfrac12\langle q,\xi q\rangle$ is conserved by **every** update of the form $\dot q=-\Lambda\,\mathsf R[\rho](q,\gamma(t))$. Here $\gamma(t)$ is an *arbitrary* cotangent signal (not necessarily a gradient), and $\Lambda$ is a self-adjoint per-block learning-rate operator commuting with $\xi$. With learning rates present, the charge is $\tfrac12\langle q,\Lambda^{-1}\xi q\rangle$.

*Proof.* Invariance gives $D\rho_q(\xi q)=0$, so $\langle\mathsf R[\rho](q,\gamma),\xi q\rangle=\langle\gamma,D\rho_q\xi q\rangle=0$. Since $\Lambda^{-1}\xi$ is self-adjoint, $\tfrac{d}{dt}\tfrac12\langle q,\Lambda^{-1}\xi q\rangle=\langle\dot q,\Lambda^{-1}\xi q\rangle=-\langle\mathsf R[\rho](q,\gamma),\xi q\rangle=0$. ∎

**Instances.**
- **LoRA.** $\mathcal G=GL_r$ acts by $(B,A)\mapsto(Bg^{-1},gA)$, with $\mathfrak k=\mathfrak{so}_r$ and $\mathfrak p=\mathrm{Sym}_r$. The conserved matrix is
$$\Phi=\frac{B^{\top}B}{\eta_B}-\frac{AA^{\top}}{\eta_A},$$
which is $r(r+1)/2$ charges ✓. This is the "balancedness" of Arora–Cohen–Hazan and Du–Hu–Lee, now identified as the $\mathfrak p$-momentum of the gauge.
- **DyLoRA.** Truncation compatibility cuts the gauge down to the diagonal torus, leaving $r$ charges $\operatorname{diag}\Phi$.
- **VeRA.** $(b,d)\mapsto(cb,d/c)$, with charge $\Vert b\Vert^2-\Vert d\Vert^2$.
- **NOLA.** Charge $\Vert\alpha\Vert^2-\Vert\beta\Vert^2$.
- **HRA.** Each $u_i\mapsto cu_i$ leaves the reflection unchanged, so each $\Vert u_i\Vert$ is conserved.
- **ReLU adapters.** Positive rescaling per hidden unit is a symmetry (Du–Hu–Lee).
- **DoRA.** Its *detached-norm* backward is not $\mathsf R$ of DoRA's own forward map. Even so, its $(B,A)$-update still has the form $\mathsf R[\rho_{\rm LoRA}](q,\hat G)$, so $\Phi$ **is still conserved**.

**Cor (weight decay) ✓.** Take equal learning rates $\eta$ and decay $\lambda$ on both factors. Then $\dot\Phi=-2\eta\lambda\Phi$: weight decay drives every LoRA exponentially toward balance.

*What this explains.*
- The charge at initialisation is a coordinate on LoRA initialisations:
  - default LoRA (Kaiming-uniform $A_0$ with variance $1/(3n)$, $B_0=0$): $\Phi_0=-A_0A_0^{\top}\approx-\tfrac13 I_r$;
  - the "Init[A]=0" variant: $\Phi_0=+B_0^{\top}B_0$;
  - PiSSA and MiLoRA ($B_0=U_rS_r^{1/2}$, $A_0=S_r^{1/2}V_r^{\top}$): $\Phi_0=0$, exactly balanced;
  - OLoRA: $\Phi_0=I-R_rR_r^{\top}$;
  - LoRA-GA (orthonormal singular-vector factors): isotropic.
- Adam's update is not of the form $\mathsf R[\rho](q,\gamma)$, so the drift of $\Phi$ during training is a **diagnostic of how far Adam departs from gauge geometry [Pred]**.

### 3.4 A closed form for LoRA dynamics (the main non-obvious result)

**Thm 3.7 [Thm] ✓.** Let $\rho(B,A)=W_{\rm res}+sBA$ be trained by gradient flow with learning rates $\eta_A,\eta_B$. Suppose the charge is **isotropic**, $\Phi=-\kappa I_r$ for some $\kappa\in\mathbb R$. Put $\mu=s\sqrt{\eta_A\eta_B}$ and $Z=sBA=W-W_{\rm res}$, with compact SVD $Z=U\Sigma V^{\top}$ of rank $r$. Then the weights follow the **autonomous** ODE
$$\dot W=-\mu^2\Big[\,G\,V g_\kappa(\Sigma^2/\mu^2)V^{\top}+U f_\kappa(\Sigma^2/\mu^2)U^{\top}G\,\Big],$$
$$f_\kappa(x)=\tfrac12\big(\sqrt{\kappa^2+4x}-\kappa\big),\qquad g_\kappa=f_\kappa+\kappa .$$

*Proof.* Rescale to $\beta=B/\sqrt{\eta_B}$ and $\alpha=A/\sqrt{\eta_A}$. Then $\beta^{\top}\beta-\alpha\alpha^{\top}=-\kappa I$ and $\beta\alpha=Z/\mu$. Set $T=\beta\beta^{\top}$ and $S=\alpha^{\top}\alpha$.
- On the left, $ZZ^{\top}/\mu^2=\beta(\beta^{\top}\beta+\kappa I)\beta^{\top}=T^2+\kappa T$.
- On the right, $Z^{\top}Z/\mu^2=\alpha^{\top}(\alpha\alpha^{\top}-\kappa I)\alpha=S^2-\kappa S$.
- On the rank-$r$ locus, $T$ is supported on $\operatorname{col}Z$ and $S$ on $\operatorname{row}Z$, with positive spectra. The positive roots of $t^2\pm\kappa t=x$ are $f_\kappa(x)$ and $g_\kappa(x)$. Since $T$ and $S$ commute with $ZZ^{\top}$ and $Z^{\top}Z$, we get $T=Uf_\kappa(\Sigma^2/\mu^2)U^{\top}$ and $S=Vg_\kappa(\Sigma^2/\mu^2)V^{\top}$.
- Finally $\dot W=s(\dot BA+B\dot A)=-s^2(\eta_BGA^{\top}A+\eta_ABB^{\top}G)=-\mu^2(GS+TG)$. ∎

*Numerical check.* With $m{=}7$, $n{=}9$, $r{=}3$, a random least-squares loss, $\eta_A{=}0.5$, $\eta_B{=}2$, $s{=}0.7$ and $\kappa\eta_A{=}0.8$, the formula matches direct integration to $10^{-12}$ relative error.

**Corollaries.**
- **(a) Two regimes and a crossover.** Default LoRA has $B_0=0$ and $A_0A_0^{\top}=cI$, so $\kappa=c/\eta_A$. The crossover sits at
$$\sigma^*=\mu\kappa=s\,c\,\sqrt{\eta_B/\eta_A}.$$
  - For $\sigma_i(Z)\ll\sigma^*$: $f\to0$ and $g\to\kappa$, so $\dot W\approx-s^2\eta_B c\,GP_{V}$. This is **LoRA-FA dynamics**; only $B$ effectively moves.
  - For $\sigma_i\gg\sigma^*$: $f\approx g\approx\sigma/\mu$, so $\dot W\approx-\mu\,[G(Z^{\top}Z)^{1/2}+(ZZ^{\top})^{1/2}G]$. These are the **balanced (Arora–Cohen–Hazan) dynamics**, with effective rate $s\sqrt{\eta_A\eta_B}$.
- **(b) LoRA+.** Setting $\lambda=\eta_B/\eta_A$ moves the crossover to $\sigma^*\sqrt\lambda$, and the balanced-regime rate becomes the *geometric mean* of the two learning rates. As $\lambda\to\infty$ we recover LoRA-FA for all time. LoRA-FA is therefore the endpoint of the LoRA+ family.
- **(c) PiSSA and MiLoRA** have $\kappa=0$, so they are *exactly balanced forever*. The early preconditioner scales gradient components by the selected singular values: principal ones for PiSSA, minor ones for MiLoRA. **[Pred, SGD/gradient flow only]** MiLoRA's adapter moves slower early by a factor of about $\sigma_{\min}/\sigma_{\max}$ of the chosen components. Adam's normalisation largely hides this.
- **(d) HF defaults [Pred].** HF PEFT's Kaiming-uniform($a{=}\sqrt5$) init gives $c\approx1/3$, so $\sigma^*\approx(s/3)\sqrt{\eta_B/\eta_A}$. Under SGD this predicts a measurable switch from one-sided to two-sided growth when the singular values of $\Delta W$ cross $\sigma^*$.

*What this explains.* "LoRA vs LoRA-FA", "LoRA+", "PiSSA converges faster" and "init matters asymmetrically" (Hayou et al.) all become statements about one conserved charge and one crossover scale.

### 3.5 Base points: apex versus smooth point

**Prop 3.8 [Prop].** Let $\rho=W_{\rm res}+sBA$.
- **(i) Apex base point.** If $B_0=0$ and $\operatorname{rank}A_0=r$, then $\operatorname{im}D\rho_{q_0}=\{XA_0\}$. This is a *linear* space of dimension $mr$, identical to the tangent image of LoRA-FA. Also $\nabla_A=0$ at $q_0$.
- **(ii) Smooth base point.** If $\operatorname{rank}B_0A_0=r$, then $\operatorname{im}D\rho_{q_0}=T_{B_0A_0}\mathcal M_r$, of dimension $r(m+n-r)>mr$.

*Proof.* $D\rho(\delta B,\delta A)=s(\delta B A_0+B_0\delta A)$. ∎

**Lemma 3.9 [Prop] (zero-gate lemma).** Let $\rho=\theta_0+\beta(u,v)$ with $\beta$ multilinear and $u_0=0$. Then $\nabla_v=0$ at $q_0$, and $K_{q_0}=D_u\beta\,D_u\beta^{\top}$. To first order, the method behaves as if $v$ were frozen.

Instances:
- **LoRA:** $u=B$.
- **VeRA:** $u=b$; the first step moves only $b$.
- **AdaLoRA:** $u=\Lambda$, initialised at 0. The first step moves only the $r$ singular values, so $K_{q_0}$ has rank $r$.
- **LLaMA-Adapter:** $u$ is the zero-initialised gate; the first step moves only the gate, not the prompt.
- **Adapters with a zero up-projection.**
- **LoHa with one zero factor.**

*What this explains.* Initialisations fall into two kinds:
- **Apex inits** (LoRA, EVA, QLoRA) are first-order one-sided. EVA is "LoRA-FA with the activation-PCA projector": $K_{q_0}(G)=GP_X$, which loses little because $G=\sum_i\delta_ix_i^{\top}$.
- **Smooth-point inits** (PiSSA, MiLoRA, OLoRA, LoRA-GA, CorDA, LoftQ) are first-order two-sided. LoRA-GA chooses $q_0$ so that $K_{q_0}(G_0)$ equals the best rank-$2r$ approximation of the first gradient $G_0$, up to its scale factors.

Separately, "splitting" methods move the **apex** $W_{\rm res}$, whereas LoRA, EVA and QLoRA keep the apex and choose only $q_0$ in the fibre.

### 3.6 The lens duality, made exact

**Thm 3.10 [Thm] (GaLore is periodically re-based one-sided LoRA) ✓.** Fix a period with an orthonormal projector $P_k\in\mathbb R^{m\times r}$, and let `opt` be any first-order optimiser whose state and output depend only on the gradients it is fed (SGD, momentum, Adam, Adafactor). Two procedures produce the same $W$-trajectory over that period:
- GaLore, started at $W_k$ with fresh optimiser state;
- the forward method $\rho_k(C)=W_k+P_kC$, with $C\in\mathbb R^{r\times n}$ and $C_0=0$, trained by `opt` and merged at the end of the period.

GaLore's scale factor is absorbed into $\eta$.

*Proof.* By induction, $W_t=W_k+P_kC_t$. GaLore feeds `opt` the matrix $P_k^{\top}\nabla L(W_t)$, and the forward method feeds it $\mathsf R[\rho_k](C_t,\nabla L)=P_k^{\top}\nabla L(W_k+P_kC_t)$. These are the same sequence, so `opt` returns the same outputs $U_t$. GaLore then sets $W\leftarrow W-\eta P_kU_t$, while the forward method sets $C\leftarrow C-\eta U_t$. ∎

The one genuinely new ingredient in GaLore is **state transport across re-basings**: it keeps the Adam moments in the old projected coordinates, which ReLoRA does not. The equivalence classifies the cotangent family:

| projector \ re-basing | frozen | re-based periodically | resampled every step |
|---|---|---|---|
| random | LoRA-FA ($K=GA_0^{\top}A_0$) | Flora | MeZO (rank 1, global) |
| gradient/data SVD | EVA, LoRA-GA (first order) | GaLore | GaLore with $T=1$ |
| coordinate blocks | BitFit, FISH, LN-tuning | LISA | (block coordinate descent) |
| two-sided, state-dependent | LoRA | ReLoRA | — |

**Prop 3.11 [Obs] (forward mode suffices for compression).** The $i$-th entry of $P^{\top}\nabla L$ is $DL(\theta)[p_i]$, so it costs $k$ JVPs and no VJP. MeZO is the case $k=1$, with the JVP estimated by a central difference, $\tfrac{L(\theta+\epsilon z)-L(\theta-\epsilon z)}{2\epsilon}=DL[z]+O(\epsilon^2)$. Its memory is therefore inference memory. Because $\mathbb E[zz^{\top}]=I$, the update is unbiased.

*What this explains.* The forward/backward duality is the defining adjunction $\langle\mathsf R[f](x,y),v\rangle=\langle y,D[f](x,v)\rangle$. Small $k$ favours forward mode (MeZO); large $k$ favours reverse mode (GaLore). MeZO's $z$ is global, so its recipe is non-local.

**Prop 3.12 [Sketch] (integrability).** Take a backward method with a time-independent, smooth, constant-rank anchor distribution $D_\theta=\operatorname{im}a(\theta)$. It is locally a forward method (an injective $\rho$ with $\operatorname{im}D\rho=D$) iff $D$ is involutive. This is Frobenius's theorem. Per-period GaLore has a constant $D$, whose leaves are the affine images of $\rho_k$. GaLore with $T=1$ recomputes its anchor from each new gradient; that anchor is not a fixed distribution on $P$, so it reparametrises nothing. This is where cotangent-side PEFT is genuinely new.

### 3.7 Merge-and-restart: what cycles can reach

**Thm 3.13 [Thm].**
- **(a) Additive methods.** $k$ cycles of train–merge–reinitialise, with displacement set $S\ni0$, reach $\theta_0+S^{+k}$ (Minkowski sum).
  - For $S=\mathcal M_{\le r}$ this is $\mathcal M_{\le kr}$: ReLoRA's reach.
  - For a fixed linear subspace $V$ it is just $V$, so merging gains nothing for diff pruning, FISH, BitFit, LoRA-XS, FourierFT or MoRA.
  - Resampled subspaces (LISA, Flora, GaLore) reach $V_1+\dots+V_k$.
- **(b) Multiplicative methods.** Suppose $S\subseteq O(n)$ contains a neighbourhood of $I$ in each of the connected subgroups $H_1,\dots,H_p$. Then $\bigcup_kS^k$ is the connected Lie subgroup whose Lie algebra is generated by $\mathfrak h_1,\dots,\mathfrak h_p$. When the $H_j$ are block-diagonal groups for partitions $\pi_j$ of $\{1,\dots,n\}$, this Lie algebra is $\bigoplus_C\mathfrak{so}(C)$. The sum runs over the connected components $C$ of the graph $\Gamma$ that joins $i$ and $j$ whenever they share a block in some $\pi_j$.

*Proof of (b).* Write $L_{ij}=E_{ij}-E_{ji}$; then $[L_{ij},L_{jk}]=L_{ik}$. So brackets along paths in $\Gamma$ generate every $L_{ij}$ within a component, and $\bigoplus_C\mathfrak{so}(C)$ is a subalgebra containing every $\mathfrak h_j$. The subgroup generated by connected subgroups is path-connected, hence a Lie subgroup (Yamabe), with the generated Lie algebra. Finally, the Cayley image contains the symmetric neighbourhoods $R\leftrightarrow R^{\top}$, so the monoid generated equals the group generated. ∎

**Consequences.**
- **OFT** with a fixed block partition never leaves its block group, however many merges it receives.
- **BOFT's** butterfly partitions ($i\sim i\oplus 2^\ell$) give the hypercube graph, which is connected, so *iterated* BOFT reaches $SO(n)$. A single BOFT, however, has dimension at most $m\cdot\tfrac nb\binom b2<\dim SO(n)$ for few factors. It is a dense *matrix*, not a dense *set*.
- **HRA$_r$** with $r$ even generates $SO(n)$ in $\lceil n/r\rceil$ cycles.
- **Design rule [Pred].** A re-based OFT must rotate its block partition between cycles; additive methods need no such change.
- **Obstruction.** Every multiplicative cycle preserves the Gram invariant (Thm 3.14a). **Merging never rescues a method from its invariant.**

### 3.8 Erlangen: groups, orbits and complete invariants

**Thm 3.14 [Thm].**

**(a) The Gram matrix is a complete invariant.** For any $W_0$,
$$\{W_0R:R\in O(n)\}=\{W:WW^{\top}=W_0W_0^{\top}\}.$$
Hence OFT, BOFT, qGOFT and HRA, which act on the input index, preserve all inner products between output neurons (the "hyperspherical energy") *and the whole singular spectrum of $W_0$*. So do any number of their merge cycles. Because multiplicative orthogonal methods can never change spectra, any variant that must change them has to add a non-orthogonal factor.

*Proof.* "⊆" is clear. For "⊇": equal Gram matrices give $\Vert W^{\top}x\Vert=\Vert W_0^{\top}x\Vert$, so $W_0^{\top}x\mapsto W^{\top}x$ is a well-defined isometry $\operatorname{im}W_0^{\top}\to\operatorname{im}W^{\top}$. Extend it to an orthogonal $O$ on $\mathbb R^n$; then $W=W_0O^{\top}$. (The First Fundamental Theorem of $O(n)$-invariant theory is the polynomial form of the same fact.) ∎

**(b) HRA is the meet of LoRA and OFT.** If $W_0$ has full column rank and $r$ is even, then
$$\operatorname{im}\mathrm{HRA}_r=(W_0\cdot SO(n))\cap(W_0+\mathcal M_{\le r}).$$
The image has dimension $rn-r(r+1)/2$. In the expressivity lattice, HRA is the **pullback** of OFT (with the full rotation group) and LoRA$_r$.

*Proof.*
- **⊆:** a product of $r$ reflections fixes a subspace of codimension at most $r$.
- **⊇:** injectivity of $W_0$ gives $k:=\operatorname{rank}(R-I)\le r$. An orthogonal map whose moved space has dimension $k$ has determinant $(-1)^k$, so $k$ is even. By Scherk's refinement of Cartan–Dieudonné (positive-definite form), $R$ is a product of exactly $k$ reflections. Pad with $(r-k)/2$ pairs $H_uH_u=I$. ∎

**(c) Tori.** (IA)$^3$ on keys and values gives $\operatorname{diag}(l)W_0$. Its orbit under $(\mathbb R^*)^m$ is $\{W:\ w_i\in\mathbb R^*w_{0,i}\}$, with complete invariant the tuple of row lines. On the FFN, (IA)$^3$ is a right torus acting on $W_2$. SSF is the affine-diagonal group (scale ⋉ shift) acting on activations. LN-tuning is SSF at the LayerNorm sites.

**(d) DoRA is LoRA twisted by a torus.** $\operatorname{im}\mathrm{DoRA}=T\cdot(W_0+\mathcal M_{\le r})$, where $T$ is the diagonal torus acting on the normalised index. This contains both LoRA's image and the torus orbit. With $B=0$, $\Delta W=(D-I)W_0$, whose rank can reach $\operatorname{rank}W_0$. **DoRA escapes the rank bound.**

*What this explains.* "Orthogonal methods preserve knowledge" means precisely that they preserve a complete invariant, so they can *never* change spectra or neuron geometry. HRA "bridging low-rank and orthogonal" is literally a pullback. Part of DoRA's gain over LoRA is rank escape, not just "decoupling".

### 3.9 Transport: structured adapters as LoRA in another basis

**Thm 3.15 [Thm].** Let $\Psi\in GL(\mathbb R^{m\times n})$ be fixed, and let $M$ be an additive method with displacement map $\delta$. Its transport $\Psi_*M$ has displacement $\Psi\circ\delta$. Then:
- $\operatorname{im}\Psi_*M=W_0+\Psi(\operatorname{im}\delta)$;
- $K^{\Psi_*M}=\Psi K^M\Psi^{\top}$;
- if $\Psi$ is Frobenius-orthogonal, $\Psi_*M$ is *dynamically* equivalent to $M$ in rotated coordinates (Prop 3.4).

Instances:
- **Kronecker family ✓.** The rearrangement $\mathcal R$ (Van Loan–Pitsianis) is a coordinate *permutation* with $\mathcal R(A\otimes B)=\operatorname{vec}A\operatorname{vec}B^{\top}$. So $\sum_{i\le k}A_i\otimes B_i$ with free factors is $\mathcal R^{-1}_*\mathrm{LoRA}_k$, including its dynamics.
  - KronA is $\mathcal R^{-1}_*\mathrm{LoRA}_1$.
  - LoKr ($C\otimes BA$) is rank one after rearrangement, with its second factor constrained to $\operatorname{vec}\mathcal M_{\le r}$.
  - Compacter's PHM weights are $\mathcal R^{-1}$-cones, with the $A_i$ shared globally.
- **Tensor-train adapters.** By Oseledets's TT-SVD, the set with TT-ranks $\le(r_k)$ equals $\bigcap_k\mathcal R_k^{-1}(\mathcal M_{\le r_k})$, a **meet** of transported cones over the unfoldings. The gauge is $\prod_kGL_{r_k}$. For loop networks (tensor ring) the representable set is in general not closed (Landsberg–Qi–Ye), an obstruction worth stating.
- **LoHa.** $X\odot Y=\Delta_m^{\top}(X\otimes Y)\Delta_n$, where $\Delta$ is the copy spider $e_i\mapsto e_i\otimes e_i$. Hence $\operatorname{rank}\le\operatorname{rank}X\cdot\operatorname{rank}Y$, and $\operatorname{im}=\mathcal M_{\le r}\odot\mathcal M_{\le r}\subseteq\mathcal M_{\le r^2}$. This is a Hadamard product of determinantal varieties, the same kind of object as in the geometry of restricted Boltzmann machines. It is a compression of the Kronecker product, not a transport.
- **HiRA.** $\Psi=W_0\odot(-)$, which is diagonal and invertible iff $W_0$ has no zero entries. Rank $\le r\cdot\operatorname{rank}W_0$. Because $\Psi$ is not orthogonal, the metric changes as well as the image.
- **FourierFT.** The fixed-support sparse method (diff-pruning/FISH type) transported by the real Fourier synthesis. It is flat.
- **LoRA-XS** is flat: $U_rS_r\mathbb R^{r\times r}V_r^{\top}$, a Tucker core with frozen factors.
- **MoRA** is flat and high-rank when its compress and decompress maps are linear.
- **Intrinsic-dimension projections** ($\theta_0+Pz$) are the archetypal flat method.

*What this explains.* The zoo of structured adapters reduces to *(cone, flat subspace or Hadamard compression) × (a fixed basis $\Psi$)*. The inductive bias of each method is the basis it works in. **[Pred]** With orthogonal $\Psi$ (Kronecker, Fourier), any performance difference from LoRA or sparse methods is about basis alignment, not optimisation.

### 3.10 Extensions, merging and prefixes

**Prop 3.16 [Prop] (linear absorption and its obstruction).**
- **(i) Absorption.** For a dense linear box, $\operatorname{im}f^\flat=\mathrm{Hom}(\mathbb R^n,\mathbb R^m)$, which is closed under $+$ and under linear pre- and post-composition. So an insertion is box-level mergeable iff every trained box function is linear (or affine, for an affine box). (IA)$^3$, SSF, BitFit and LN-tuning are mergeable.
- **(ii) Linear adapters.** A parallel linear adapter is $\cong$ LoRA. A serial linear adapter after $W_0$ has image $W_0+\{X\in\mathcal M_{\le r}:\operatorname{row}X\subseteq\operatorname{row}W_0\}$; it is expressively equivalent to LoRA iff $\operatorname{rank}W_0=n$ (before $W_0$: iff $\operatorname{rank}W_0=m$), and isomorphic to LoRA in $\mathbf{Rep}(F)$ when $W_0$ is invertible.
- **(iii) Obstruction.** Nonlinear serial and parallel adapters are not box-level mergeable, and neither are prefix and prompt tuning. Restrict to a sequence of length 1. The frozen-architecture head then computes $x\mapsto W_oW_vx$, which is linear in $x$. The prefixed head computes $(1-\lambda(x))W_oW_vx+\lambda(x)\bar v(x)$, whose gate $\lambda(x)$ is a non-constant softmax mass, so it is not linear. ∎

**Prop 3.17 [Prop] (prefixes).**
- **(i)** For fixed inputs to an attention layer, prefixes leave the ratios $a_{ij}/a_{ik}$ over content positions unchanged, and $h_i'=(1-\lambda_i)h_i+\lambda_i\bar v_i$ with $\bar v_i\in\operatorname{conv}(\text{prefix values})$. This recovers He et al.'s gated-adapter form and Petrov et al.'s per-layer limitation.
- **(ii)** In a causal decoder with the soft prompt placed first, the prompt's layer-$\ell$ keys and values depend only on the prompt. So $p\mapsto(K_\ell(p),V_\ell(p))_\ell$ is a morphism Prompt → Prefix. **This fails for bidirectional encoders**, where the prompt's states depend on the input.
- **(iii)** The Li–Liang MLP is a 2-cell $\sigma$; P-tuning v2 drops it, keeping the image and changing $K$.
- **(iv)** LLaMA-Adapter's zero-initialised gate supplies the missing neutral point. **Its contribution, categorically, is pointedness**, and Lemma 3.9 then applies.

### 3.11 Backward reach, locality, merging operators

**Prop 3.18 [Obs] (backward reach).** In a composite lens, the parameter cotangents need only the backward maps of boxes downstream of the earliest trainable box. They also need only the inputs of those boxes whose $\mathsf R$ depends on their input.
- **LST.** The side network reads the backbone through copy maps, so the backward reach is the side network alone.
- **Prompt tuning.** It has the *smallest* $Q$ and the *largest* reach.
- **LLaMA-Adapter** reaches only its top $L$ layers; **LISA** reaches down to its lowest sampled layer; **MeZO** has no reach at all.
- **LoRA-FA** stores $A_0x$ ($r$ numbers) instead of $x$ ($n$ numbers), because $\nabla_B=G(A_0x)^{\top}$.

HF PEFT's `target_modules` is literally the generator ↦ 2-cell assignment of a recipe (Def 2.9). Global methods include:
- AdaLoRA, EVA and diff pruning (global budgets);
- FISH (global selection);
- VB-LoRA (a shared vector bank);
- VeRA and FourierFT (shared frozen randomness, with local trainables);
- Compacter (shared $A_i$);
- LISA (global sampling) and MeZO (global $z$).

**Prop 3.19 [Prop] (merging operators).**
- **Task arithmetic** is linear in the abelian group $(T_{\theta_0}P,+)$, so it is $GL$-equivariant (basis-free).
- **TIES** (trim, elect sign, disjoint mean) and **DARE** (drop with rate $p$, rescale by $1/(1-p)$, unbiased) act coordinatewise. They are equivariant under signed coordinate permutations and global rescaling, but *not* under rotations of weight space.
- Because $\rho_{\rm LoRA}$ is bilinear, averaging the factors is not averaging the products. Factor averaging is also gauge-dependent: replacing each $(B_i,A_i)$ by $(B_ig_i^{-1},g_iA_i)$ changes $\bar B\bar A$. **[Pred]** TIES and DARE on factors give gauge-dependent results; on $\Delta W$ they do not.

Related constructions:
- **LoRAHub** is a flat method spanned by task vectors ($Q=\mathbb R^k$), optimised gradient-free.
- **Text-to-LoRA** is Para composition $E\to Q\to P$ (amortisation).
- **Mixtures of LoRA experts** are dependent 2-cells (Def 2.6).
- **S-LoRA** evaluates the *unmerged* string diagram: the shared $W_0$ box is applied to a heterogeneous batch while each adapter branch is applied to its own sub-batch. Merging is the evaluation functor from syntax to semantics. Cost is a functional on syntax, so multi-tenant serving exploits exactly the non-merged form.

---

## 4. Classification

Every method gets a value on eight axes:
- **A1 locus:** F (forward 2-cell), B (backward lens), E (extension), D (dependent 2-cell) or M (operation on several methods);
- **A2 image type:** flat, cone, transported cone, group orbit, Hadamard/torus product, or function class;
- **A3 gauge, its Cartan split, and the charge at initialisation;**
- **A4 base point:** apex, smooth point, defect, or not pointed;
- **A5 merge type;**
- **A6 schedule:** static, filtered, re-based, or resampled;
- **A7 locality;**
- **A8 backward reach and optimiser-state size.**

**Table 1 — forward (reparametrisation) methods.** For all of these, A5 is "mergeable", A1 is F, and reach is the full network unless noted.

| method | A2 image (dim) | A3 gauge / charge $\Phi_0$ | A4 base point | A6 / A7; merge-cycle reach |
|---|---|---|---|---|
| LoRA | cone $W_0+\mathcal M_{\le r}$, $r(m{+}n{-}r)$ | $GL_r$ ($\mathfrak{so}_r\oplus\mathrm{Sym}_r$); $\approx-\tfrac13I$ | apex | static, local; $\mathcal M_{\le kr}$ |
| rsLoRA | = LoRA, $s=\alpha/\sqrt r$ | as LoRA; rescales $K$ and $\sigma^*$ | apex | static, local |
| LoRA+ | = LoRA; metric $\mathrm{diag}(\eta_B^{-1},\eta_A^{-1})$ | $B^{\top}B/\eta_B-AA^{\top}/\eta_A$; $\sigma^*\times\sqrt\lambda$ | apex | static, local |
| LoRA-FA | flat $\{XA_0\}$, $mr$ | trivial | apex | static; no gain without resampling |
| DoRA | $T\cdot$cone; rank unbounded | $GL_r$, charge kept (detached norm) | apex, $m_0=\Vert W_0\Vert$ | static, local |
| PiSSA | cone at apex $W_0-W_r$ | $\Phi_0=0$ (balanced) | smooth, principal | static |
| MiLoRA | cone at apex $W_0-W_{\min}$ | $\Phi_0=0$ | smooth, minor | static |
| OLoRA | cone at apex $W_0-Q_rR_r$ | $I-R_rR_r^{\top}$ | smooth | static |
| LoRA-GA | cone at re-split apex | isotropic | smooth; $K_{q_0}G_0\approx\mathrm{SVD}_{2r}(G_0)$ | static |
| EVA | cone | $-A_0A_0^{\top}$, $A_0$ = activation PCs | apex; $K=GP_X$ | global rank budget |
| CorDA | cone at context-SVD apex | balanced-type | smooth | static |
| LoftQ | cone at quantised apex | — | smooth; defect minimised | static |
| QLoRA | cone at $\mathrm{dq}(q(W_0))$ | as LoRA | apex; defect = quantisation error | static |
| AdaLoRA | union of strata, $P\Lambda Q$ | torus (regulariser breaks it) | apex; first step moves only $\Lambda$ | filtered, global |
| DyLoRA | filtration $\mathrm{LoRA}_1\subset\dots\subset\mathrm{LoRA}_r$ | diagonal torus; $\operatorname{diag}\Phi$ | apex | filtered |
| ReLoRA | $\mathcal M_{\le kr}$ after $k$ cycles | $GL_r$ per cycle | re-based apex | periodic |
| VeRA | toric $\{\Lambda_bB\Lambda_dA\}$, $\le m{+}r{-}1$ | $\mathbb R^*$: $\Vert b\Vert^2-\Vert d\Vert^2$ | apex ($b_0=0$) | shared frozen $A,B$ |
| NOLA | ⊆ cone via linear maps | $\mathbb R^*$ | apex | shared bases |
| VB-LoRA | ⊆ cone (top-$k$ bank mixtures) | permutation-type | apex | global bank |
| LoRA-XS | flat, $r^2$ | trivial | — | no cycle gain |
| FourierFT | flat, $n$ cosine patterns | trivial | — | no cycle gain |
| MoRA | flat, $\hat r^2$, high rank | trivial | — | no cycle gain |
| LoHa | $\mathcal M_{\le r}\odot\mathcal M_{\le r}$ | $GL_r^2\times\mathbb R^*$ | apex | static |
| LoKr / KronA | $\mathcal R^{-1}$-transported cone | $GL$-type | apex | static |
| TT adapters | $\bigcap_k\mathcal R_k^{-1}\mathcal M_{\le r_k}$ | $\prod GL_{r_k}$ | — | static |
| HiRA | $W_0+W_0\odot\mathcal M_{\le r}$ | $GL_r$ | apex | static |
| OFT | $W_0\cdot SO(b)^{n/b}$ ⊆ Gram level set | trivial | $I$ | stuck in block group |
| BOFT | product of $m$ butterfly factors | — | $I$ | iterated → $SO(n)$ |
| qGOFT (Givens) | products of Givens rotations | — | $I$ | → $SO(n)$ if pairing graph connected |
| HRA | $W_0 SO(n)\cap(W_0+\mathcal M_{\le r})$ | $\mathbb R^*$ per $u_i$; $\Vert u_i\Vert$ conserved | $I$ | → $SO(n)$ in $\lceil n/r\rceil$ cycles |
| (IA)$^3$ | torus orbit (row lines fixed) | trivial | $l=1$ | stuck (group) |
| SSF | affine-diagonal orbit | trivial | $(1,0)$ | stuck |
| BitFit | flat: biases | trivial | — | stuck |
| diff pruning | union of $k$-sparse subspaces | trivial | — | global L0 budget |
| FISH mask | fixed coordinate subspace | trivial | — | global selection |
| LN tuning | flat: LN $(\gamma,\beta)$ = SSF at LN sites | trivial | — | stuck |

**Table 2 — extension and dependent methods (A1 = E or D).**

| method | extension type | pointed? | merge | key structure | reach |
|---|---|---|---|---|---|
| Houlsby | serial nonlinear, 2 per layer | ≈ (exact with zero up-projection) | ✗ | ReLU rescaling gauge | full |
| Pfeiffer | serial, 1 per layer | ≈ | ✗ | — | full |
| parallel adapter | parallel nonlinear | ✓ (zero up-projection) | ✗ | linear version ≅ LoRA | full |
| Compacter | serial with PHM ($\mathcal R^{-1}$-cone) weights | ≈ | ✗ | shared $A_i$ (global) | full |
| AdapterFusion | D: attention over frozen adapters | — | ✗ | dependent 2-cell | full |
| LST | side net fed by copy maps | ✗ | ✗ | backbone backward never run | side net only |
| LLaMA-Adapter | gated prefix in top $L$ layers | ✓ ($g_0=0$) | ✗ | pointed prefix; zero-gate | top $L$ |
| prompt tuning | input-state precomposition | ✗ (defect) | ✗ | ≤ prefix (causal only) | full |
| prefix tuning | K/V states per layer, with MLP 2-cell | ✗ | ✗ | gated adapter; ratio invariant | full |
| P-tuning v2 | deep prompts, no MLP | ✗ | ✗ | same image as prefix, other $K$ | full |
| MoLE | D: gated sum of LoRAs | — | ✗ | coKleisli 2-cell | full |

**Table 3 — backward lenses and operations on methods (A1 = B or M).**

| method | anchor / operation | schedule | equivalence | state; reach |
|---|---|---|---|---|
| GaLore | gradient-SVD $P_t$ | periodic re-base | ≡ merged one-sided LoRA (Thm 3.10) | $r\times n$; full |
| Flora | random $P_t$ | periodic resample | ≡ LoRA-FA + resampling | $r\times n$; full |
| LISA | layer-block projector | periodic resample | block-coordinate GaLore | active layers; down to lowest active |
| MeZO | global rank-one $z_t$ | every step | $K=zz^{\top}$; forward-mode JVP | seed only; none |
| task arithmetic | linear in $(T_{\theta_0}P,+)$ | — | $GL$-equivariant | — |
| TIES / DARE | coordinatewise nonlinear | — | signed-permutation equivariant | — |
| LoRAHub | flat F over task vectors | — | gradient-free | $k$ |
| S-LoRA | unmerged-syntax evaluation | — | merging = semantics functor | — |
| Text-to-LoRA | Para composition $E\to Q\to P$ | — | amortised 2-cell | hypernet |

---

## 5. Equivalences and obstructions (summary)

**Equivalences.**
- Linear parallel adapter ≅ LoRA.
- Serial linear adapter is expressively equivalent to LoRA iff $W_0$ is injective (adapter after $W_0$) or surjective (adapter before $W_0$) (3.16).
- $\mathrm{LoRA}_r\circledast\mathrm{LoRA}_{r'}\cong\mathrm{LoRA}_{r+r'}$ (3.2).
- LoRA ≡₁ LoRA-FA at apex init, and the two stay close until $\sigma^*$ (3.7, 3.8).
- LoRA+ with $\lambda\to\infty$ = LoRA-FA (3.7b).
- GaLore per period ≡ re-based one-sided LoRA; Flora ≡ LoRA-FA with resampling (3.10).
- Sum-of-Kronecker adapters ≅ $\mathcal R^{-1}_*$LoRA, dynamically (3.15).
- HRA = OFT ∧ LoRA (3.14b).
- Prefix = gated parallel adapter in activation space; prompt ≤ prefix in causal models (3.17).
- Prefix ≃ P-tuning v2 expressively, but not first-order.
- LLaMA-Adapter = pointed prefix.

**Obstructions.**
- Nonlinear insertions are not box-mergeable (3.16).
- Multiplicative methods cannot change their complete invariant: Gram matrix and spectrum (OFT family) or row lines (tori), even under merging (3.13, 3.14).
- Fixed subspaces gain nothing from merge cycles (3.13a).
- A fixed-partition block OFT never leaves its block group (3.13b).
- Prompt ≤ prefix fails for bidirectional encoders.
- Gradient descent cannot be gauge-natural beyond $O(r)$ (3.4).
- Factor-space merging is gauge-dependent (3.19).
- Loop tensor networks are not closed.

---

## 6. Rosetta stone

| PEFT | categorical / geometric |
|---|---|
| pretrained model | pointed 1-cell $(P,\theta_0,f)$ of $\mathbf{Para}_*$ |
| PEFT method | pointed 2-cell into it; object of $\mathbf{Rep}(F)$ |
| "M can emulate N" | slice morphism $N\to M$ |
| frozen / full FT | initial / terminal object |
| expressivity; effective #params | $\operatorname{im}\rho$; generic rank of $D\rho$ |
| rank $r$ | factorisation through an $r$-dim object |
| combining adapters, stacking ranks | $\circledast$; strong monoidal $r\mapsto\mathrm{LoRA}_r$ |
| HRA between LoRA and OFT | pullback (meet) |
| adapters, prompts | extension $\iota:\tilde F\Rightarrow F$ |
| identity-at-init | neutral point (pointed $\iota$) |
| merge into weights | lift along currying $f^\flat$ |
| `target_modules` | recipe on generators; horizontal composition |
| backprop | 2-functor $\mathbf{Para}(\mathsf R)$ |
| PEFT gradient | pullback $\mathsf R[\rho]$ |
| effective update rule | induced cometric $K=D\rho D\rho^{\top}$ |
| GaLore, Flora, MeZO | lens 2-cells not in $\operatorname{im}\mathbf{Para}(\mathsf R)$ |
| LoRA redundancy | $GL_r$-torsor fibres (gauge) |
| LoRA+, Riemannian LoRA, RITE | choices of metric; restored equivariance |
| balancedness | Noether charge of $\mathfrak p=\mathrm{Sym}_r$ |
| init scheme | apex (splitting) + base point + charge |
| QLoRA | base-point shift (pointedness defect) |
| ReLoRA cycles | generated submonoid |
| OFT, (IA)$^3$ | group orbits; complete invariants |
| Kronecker, Fourier, HiRA | transport by fixed $\Psi$ |
| task arithmetic vs TIES | $GL$-natural vs hyperoctahedral-natural |
| MoE-LoRA | coKleisli (input-dependent) 2-cell |
| activation memory | backward reach of the composite lens |
| S-LoRA | syntax vs semantics |

---

## 7. What is analogy, stated plainly

- **"Erlangen programme" [Analogy].** It is a theorem only for the actual groups $O(n)$, tori and translation subspaces, where complete invariants exist (3.14). The LoRA cone is not a group orbit: $\mathcal M_{\le r}$ is not closed under $+$.
- **"Noether" [Analogy, partly].** Gradient flow is not Hamiltonian. We prove the conservation law directly (3.6); "Noether" names the pattern (symmetry → charge), not a variational theorem.
- **"Sub-Riemannian geometry" [Analogy].** $K$ has varying rank. The Chow–Rashevskii-type reach statement is proved only for groups (3.13b) and for linear distributions.
- **Adam.** Props 3.4–3.8 and Thm 3.7 are statements about gradient flow and SGD. Adam, finite steps and weight decay (except as treated in 3.6) break them. We state each [Pred] with its optimiser scope.
- **Expressivity is not learnability.** A slice morphism says what *can* be represented, not what training *finds*.
- **Learn and lens bookkeeping.** Our substantive content is the chain rule plus a metric. "Backprop as functor" supplies the bookkeeping, not the force.
- **MeZO as a lens [Analogy].** It is a stochastic estimator, whose natural home would be a Markov or probabilistic category; we do not build one.
- **"Natural family" across boxes.** This is horizontal composition, not a natural transformation in a formal 2-categorical sense.
- **S-LoRA as syntax/semantics, Text-to-LoRA as Para composition.** Interpretive and bookkeeping only.
- **Whole-network non-mergeability of prefixes and adapters.** Expected but unproved; we prove box-level obstructions only.

---

## 8. Interactive visualisations (each tied to one result)

1. **Noether hyperbolas** (3.6, $r=1$). *Manipulate:* init $(a_0,b_0)$, learning-rate ratio $\lambda$, weight decay, SGD/Adam toggle. *See:* the trajectory stays on $a^2/\eta_A-b^2/\eta_B=$ const. With weight decay it is pulled to the diagonal (balance); with Adam it visibly leaves the hyperbola, and $\Phi$'s drift is plotted.
2. **Crossover meter** (3.7). *Manipulate:* $s$, $c$, $\lambda$. *See:* the curves $f_\kappa$ and $g_\kappa$ against $\sigma$, with $\sigma^*$ marked, and simulated singular values of $\Delta W$ crossing from the "LoRA-FA band" to the "balanced band". A PiSSA preset shows $\kappa=0$.
3. **Cone and apex** (3.8). *Manipulate:* drag $A_0$; toggle LoRA, PiSSA or QLoRA. *See:* symmetric $2\times2$ matrices drawn in $\mathbb R^3$, with the rank-1 cone $xz=y^2$. The apex tangent plane $\{XA_0\}$ rotates as $A_0$ moves. PiSSA moves the apex to $W_{\rm res}$ and the base point onto the smooth sheet. QLoRA shows a defect vector.
4. **Gauge slider** (3.4, 3.5). *Manipulate:* $g\in GL_2$ for a fixed $W=BA$, split as rotation × stretch. *See:* the ellipse of $K(G)$ over unit $G$. Rotation leaves it unchanged and stretching deforms it. A "Riemannian" toggle freezes it.
5. **Orbit explorer** (3.14). *Manipulate:* choose LoRA, OFT, HRA, (IA)$^3$ or DoRA, and drag the parameters. *See:* live readouts of the Gram matrix, the singular values and $\operatorname{rank}\Delta W$, with "locked" badges. HRA lights both "rank ≤ r" and "Gram fixed".
6. **Merge-cycle reach** (3.13). *Manipulate:* number of cycles, and the partition used per cycle (fixed, butterfly or random). *See:* ReLoRA's rank climbing $r,2r,\dots$, and the block graph's components merging, with $\sum_C\binom{|C|}2$ against $\dim\mathfrak{so}(n)$.
7. **Rearrangement** (3.15). *See:* an animated permutation turning $A\otimes B$ into $\operatorname{vec}A\operatorname{vec}B^{\top}$. Add a second term and the rank becomes 2. A TT mode shows the unfolding ranks.
8. **Lens string diagram** (2.7, 3.18). *Manipulate:* choose a method and the sizes $d$, $L$, $r$. *See:* which backward boxes run, where the compression sits, which activations are stored, and memory bars computed from the formulas.
9. **Prefix ratios** (3.17). *Manipulate:* prefix strength. *See:* attention bars over content tokens shrink proportionally while their ratios stay locked. A comparison with LoRA on $W_k$ reshapes them.
10. **Method atlas** (3.1, 3.2, 3.14b, Table 1). *See:* a Hasse diagram of simulations and transports, with nodes placed by (#params, $\dim\operatorname{im}\rho$). Click a node for its $(Q,q_0,\rho)$, gauge, charge and merge type. HRA is drawn as the pullback square.
11. **JVP vs VJP** (3.11). *Manipulate:* the slider $k$. *See:* the cost of $k$ JVPs against one VJP plus projection, with MeZO at $k=1$.

---

## 9. Where this departs from the seed sketch

- **(S2)** The slice preorder equals image inclusion only in $\mathbf{Set}$; in $\mathbf{Euc}$ it is finer (the cube-root example). We therefore use $\mathbf{SA}$ for expressivity and $\mathbf{Euc}$ for dynamics.
- **(S3)** $\iota$ is a 2-cell $\tilde F\Rightarrow F$: the old model is a reparametrisation of the new one. Prompts and prefixes are **not** identity-at-init; LLaMA-Adapter is what makes them pointed.
- **(S4)** The OFT invariant is the Gram matrix of output neurons *and the whole spectrum*, and it is complete (3.14a). HRA is a pullback, which is sharper than "bridging".
- **(S5)** Only the $\mathrm{Sym}_r$ part of $\mathfrak{gl}_r$ yields charges; $\mathfrak{so}_r$ yields isometries. The charge survives DoRA's detached norm and decays under weight decay. Thm 3.7 turns the charge into a closed form.
- **(S8)** Distinguish apex shifts (PiSSA and the like) from base-point choices (EVA). QLoRA is unpointed.
- **(S9)** The forward/backward split is not a dichotomy. It reduces to integrability plus re-basing (3.10, 3.12).

---

## 10. References

**Category theory of learning.**
- Fong, Spivak, Tuyeras, *Backprop as Functor*, arXiv:1711.10455.
- Cruttwell, Gavranović, Ghani, Wilson, Zanasi, *Categorical Foundations of Gradient-Based Learning*, arXiv:2103.01931.
- Cockett et al., *Reverse Derivative Categories*, arXiv:1910.07065.
- Fong, Johnson, *Lenses and Learners*, arXiv:1903.03671.
- Capucci, Gavranović, Hedges, Rischel, *Towards Foundations of Categorical Cybernetics*, arXiv:2105.06332.
- Capucci, Gavranović, *Actegories for the Working Amthematician*, arXiv:2203.16351.
- Gavranović, *Fundamental Components of Deep Learning* (thesis), arXiv:2403.13001.
- Gavranović et al., *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.
- Spivak, *Learners' Languages*, arXiv:2103.01189.
- Riley, *Categories of Optics*, arXiv:1809.00738.

**LoRA family.**
- LoRA, arXiv:2106.09685; rsLoRA, arXiv:2312.03732; LoRA+, arXiv:2402.12354; LoRA-FA, arXiv:2308.03303; DoRA, arXiv:2402.09353.
- PiSSA, arXiv:2404.02948; MiLoRA, arXiv:2406.09044; OLoRA, arXiv:2406.01775; LoRA-GA, arXiv:2407.05000; EVA, arXiv:2410.07170; CorDA, arXiv:2406.05223.
- LoftQ, arXiv:2310.08659; QLoRA, arXiv:2305.14314; AdaLoRA, arXiv:2303.10512; DyLoRA, arXiv:2210.07558; ReLoRA, arXiv:2307.05695.
- VeRA, arXiv:2310.11454; NOLA, arXiv:2310.02556; VB-LoRA, arXiv:2405.15179; LoRA-XS, arXiv:2405.17604; FourierFT, arXiv:2405.03003.
- LoHa (FedPara), arXiv:2108.06098; LoKr/LyCORIS, arXiv:2309.14859; KronA, arXiv:2212.10650; Compacter, arXiv:2106.04647; LoRETTA (TT), arXiv:2402.11417.
- MoRA, arXiv:2405.12130; HiRA (Huang et al., ICLR 2025).

**Orthogonal and scaling methods.**
- OFT, arXiv:2306.07280; BOFT, arXiv:2311.06243; HRA, arXiv:2405.17484; qGOFT, arXiv:2404.04316.
- (IA)$^3$, arXiv:2205.05638; SSF, arXiv:2210.08823; BitFit, arXiv:2106.10199; diff pruning, arXiv:2012.07463; FISH, arXiv:2111.09839.
- LN-tuning, arXiv:2312.11420; LISA, arXiv:2403.17919.

**Adapters and prompts.**
- Houlsby, arXiv:1902.00751; AdapterFusion, arXiv:2005.00247; MAD-X, arXiv:2005.00052.
- He et al., unified view, arXiv:2110.04366; UniPELT, arXiv:2110.07577; LST, arXiv:2206.06522; LLaMA-Adapter, arXiv:2303.16199.
- Prompt tuning, arXiv:2104.08691; prefix tuning, arXiv:2101.00190; P-tuning v2, arXiv:2110.07602.
- Wang et al., prompt-tuning universality and limits, arXiv:2305.18787; Petrov–Torr–Bibi, arXiv:2310.19698.

**Optimisers and memory.**
- GaLore, arXiv:2403.03507; Flora, arXiv:2402.03293; MeZO, arXiv:2305.17333.
- Riemannian preconditioned LoRA, arXiv:2402.02347; LoRA-RITE, arXiv:2410.20625; Hayou et al., LoRA initialisation, arXiv:2406.08447.

**Merging, serving and generation.**
- Task arithmetic, arXiv:2212.04089; TIES, arXiv:2306.01708; DARE, arXiv:2311.03099.
- LoraHub, arXiv:2307.13269; MoLE, arXiv:2404.13628; S-LoRA, arXiv:2311.03285; Text-to-LoRA, arXiv:2506.06105.

**Theory and geometry.**
- Arora–Cohen–Hazan, arXiv:1802.06509; Du–Hu–Lee, arXiv:1806.00900; Kunin et al., arXiv:2012.04728; Zhao et al., conserved quantities, arXiv:2210.17216.
- Intrinsic dimension: Li et al., arXiv:1804.08838; Aghajanyan et al., arXiv:2012.13255.
- Zeng–Lee, expressive power of LoRA, arXiv:2310.17513; Malladi et al., kernel view, arXiv:2210.05643; weight normalisation, arXiv:1602.07868.
- Cueto–Morton–Sturmfels, RBM geometry, arXiv:0908.4425.
- Van Loan–Pitsianis, *Approximation with Kronecker Products* (1993); Oseledets, *Tensor-Train Decomposition*, SIAM J. Sci. Comput. 33 (2011); Landsberg–Qi–Ye, *On the geometry of tensor network states* (2012).
- Scherk, *On the decomposition of orthogonalities into symmetries*, Proc. AMS (1950).
- Hugging Face PEFT library (github.com/huggingface/peft).
