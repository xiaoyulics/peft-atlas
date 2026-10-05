# Erlangen and Gauge: a Kleinian foundation for parameter-efficient fine-tuning

*Proposal for the "Erlangen programme and gauge symmetry" lens.*

**Status labels used throughout.** **[Def]** definition. **[Thm]** theorem with complete proof. **[Prop]** proposition with complete proof (smaller). **[Sketch]** proof sketch: the idea is complete, but routine details are left out. **[Obs]** a trivial-but-true observation, labelled as such. **[Num]** checked numerically (float64 autograd Jacobian ranks or small-step gradient flow; scripts in the project scratchpad). **[Conj]** conjecture or testable prediction. **[Analogy]** heuristic only, no theorem behind it.

**Conventions (load-bearing).** A linear box computes $y = Wx$ with $W \in \mathbb{R}^{m\times n}$, so $m = d_{\text{out}}$ and $n = d_{\text{in}}$ (PyTorch layout). Row $i$ of $W$ holds the incoming weights of output neuron $i$. Two Gram matrices appear and must not be confused:
$$K_{\text{out}}(W) = WW^{\top}\in\mathrm{Sym}_m \ (\text{neuron Gram: pairwise inner products of neurons}),\qquad K_{\text{in}}(W) = W^{\top}W\in\mathrm{Sym}_n\ (\text{pulled-back input metric}).$$
The group $GL_m\times GL_n$ acts on the **left** of $\mathbb{R}^{m\times n}$ by $(g,h)\cdot W = gWh^{-1}$. The output factor $g$ acts on the left of $W$, mixing neurons. The input factor $h$ acts on the right, reparametrizing the input space. A right multiplication $W\mapsto WR$ with $R\in O(n)$ keeps $K_{\text{out}}$ and conjugates $K_{\text{in}}$. A left multiplication $W\mapsto PW$ with $P\in O(m)$ keeps $K_{\text{in}}$ and conjugates $K_{\text{out}}$. Papers that store $W^{\top}$ (for example OFT's $z = W^{\top}x$) swap these words. We translate every method into the convention above. $M_{\le r}\subset\mathbb{R}^{m\times n}$ is the determinantal variety of matrices of rank at most $r$, and $M_r$ is its smooth open stratum of rank exactly $r$, with $\dim M_r = r(m+n-r)$. $T_m\subset GL_m$ is the diagonal torus and $\mathrm{Mon}_m = T_m\rtimes S_m$ is the group of monomial matrices. $B_r = \{\pm1\}^r\rtimes S_r$ is the hyperoctahedral group of signed permutations.

---

## 1. Thesis

**A PEFT method is a small, structured chart on the weight space of a pretrained model, centered at the pretrained point. Every method is fixed by three groups and one point.** (i) The **covariance group** $\mathcal{G}_M\subseteq GL_m\times GL_n$ fixes the geometry the method belongs to, in Klein's sense of Erlangen. LoRA is a concept of linear geometry, OFT of Euclidean geometry on the input side, and (IA)$^3$, LoHa, HiRA and sparse masks of coordinate geometry. (ii) The **structure group** whose orbit the method explores, with its **absolute invariants** (functions of $W$ that the method cannot change) and **relative invariants** (constraints on the pair $(W_0, W)$, such as $\operatorname{rank}(W-W_0)\le r$). (iii) The **gauge group** $K_M$ of the parametrization $\rho_M$ itself: the symmetries of the trainable parameters that leave the weights unchanged. (iv) The **position of $\theta_0$** in the reachable set: the singular apex of a cone, or a smooth point. The gauge group does not act on the model. It acts on the optimizer, and that is where it does its work. By a Noether-type theorem for gradient flows, the non-compact half of the gauge algebra produces conserved charges, such as $B^{\top}B - AA^{\top}$ for LoRA. The compact half produces equivariance. An optimizer is well defined on the actual model only when it is equivariant under the whole gauge group. Read through these four data, initialization schemes, learning-rate ratios, preconditioners, rank allocation, merging rules, and serving rules all become statements about choosing, fixing, or respecting a gauge. Concrete outputs include exact effective dimensions, a balance–escape trilemma, incomparability theorems, and covariance-based design rules. The headline results below are proved, and the numeric ones are checked by machine.

---

## 2. Definitions

### 2.1 Ambient category and the pretrained point

**[Def 1] Ambient category.** Work in $\mathbf{Diff}$, the category of smooth manifolds and smooth maps. Reachable sets are images of smooth maps and are usually not manifolds; $M_{\le r}$ is singular along $M_{\le r-1}$. All images we meet are semialgebraic (images of polynomial or rational maps, by Tarski–Seidenberg), so dimension, smooth points and tangent cones are well defined. When images matter we work in the category $\mathbf{SA}$ of semialgebraic sets and Nash or semialgebraic maps. **Hom-sets are never used beyond $\mathbf{Diff}$; only images are taken in $\mathbf{SA}$.**

**[Def 2] Parametrized model, pretrained point.** A model is a morphism of $\mathbf{Para}(\mathbf{Diff})$ (Fong–Spivak–Tuyéras; Cruttwell et al.): a pair $(\Theta, f)$ with $f:\Theta\times X\to Y$. Here $\Theta = \prod_{\ell}\Theta_\ell$, one factor per box (linear weights $\mathbb{R}^{m_\ell\times n_\ell}$, biases, norm gains, embeddings). A **pretrained model** is a *pointed* model $(\Theta,\theta_0,f)$.

**[Def 3] PEFT method at $\theta_0$ (weight-space type).** A pointed reparametrization $(Q, q_0, \rho)$ with $\rho:Q\to\Theta$ smooth and $\rho(q_0)=\theta_0$. Usually $\dim Q\ll\dim\Theta$, and $\theta_0$ (or a splitting of it) is frozen *inside* $\rho$. These objects form the pointed slice category $\mathbf{Rep}_*(\Theta,\theta_0) := \mathbf{Diff}_*/(\Theta,\theta_0)$. A morphism $h:(Q,q_0,\rho)\to(Q',q_0',\rho')$ is a smooth pointed map with $\rho'\circ h = \rho$ (a *simulation*: $N$ reproduces every weight $M$ reaches, together with its parameter path). The **reachable set** is $S_M = \rho(Q)\subseteq\Theta$ (semialgebraic). The **expressivity preorder** is $M\preceq N$ iff $S_M\subseteq S_N$. This is weaker than the existence of a morphism, because a morphism also needs a smooth lift.

**[Def 4] Methods as natural families.** Most methods are defined for every $W_0$ at once: they are maps $W_0\mapsto(Q,q_0,\rho_{W_0})$ for each box type. We write $S_M(W_0)$ for the reachable set as a function of the base point. Glueing per-box definitions along a network (S11 of the seed) is the uniformity assumption. Methods that share parameters across layers (VB-LoRA's vector bank, Compacter's shared factors, VeRA's shared random matrices) are *not* products of per-box methods. Their parameter object is a limit over the whole network, not a per-box choice.

**[Def 5] Architectural (extension-type) methods.** An extension is a pointed embedding $\iota:(\Theta,f)\to(\Theta\times P', \tilde f)$ with a neutral point $p_0'$ such that $\tilde f(\theta,p_0',-)=f(\theta,-)$ (identity at init), followed by a weight-space method on $\Theta\times P'$. A trained extension is **mergeable** if there is $\mu:Q\to\Theta$ with $\tilde f\circ(\rho\times X) = f\circ(\mu\times X)$ extensionally.

### 2.2 The Erlangen data

Fix a layer $\Theta_\ell=\mathbb{R}^{m\times n}$ and the ambient affine group $\mathbb{A}_\ell := (GL_m\times GL_n)\ltimes\mathbb{R}^{m\times n}$, acting by $(g,h,X)\cdot W = gWh^{-1}+X$.

**[Def 6] Covariance group.** The covariance group of a natural method $M$ is the largest subgroup $\mathcal{G}_M\subseteq GL_m\times GL_n$ with
$$S_M\big((g,h)\cdot W_0\big) = (g,h)\cdot S_M(W_0)\qquad\text{for all }W_0.$$
If $M$ depends on auxiliary data (random frames, activation statistics, a first-batch gradient), the data transforms too, and the identity may hold only in distribution. $\mathcal{G}_M$ is *the geometry the method lives in*: what $M$ does is meaningful only up to $\mathcal{G}_M$-reparametrization of the layer.

**[Def 7] Structure group, absolute and relative invariants.** If $S_M(W_0) = \Gamma(Q)\cdot W_0$ for a parametrized subset $\Gamma:Q\to H$ of a subgroup $H\subseteq\mathbb{A}_\ell$ with $\Gamma(q_0)=e$, then $M$ is **of type $H$**. It is **group-type** if $S_M(W_0)$ is a whole $H$-orbit (or an open piece of one), and **variety-type** if $S_M(W_0)-W_0$ is a proper cone that is not closed under the group law. An **absolute invariant** is a function $F$ on $\Theta_\ell$ with $F|_{S_M(W_0)}\equiv F(W_0)$ for all $W_0$. A **relative invariant (constraint)** is a function $\Phi(W_0,W)$ of the pair that is bounded or fixed on $S_M(W_0)$. Distance in Euclidean geometry is the classical relative invariant; $\operatorname{rank}(W-W_0)$ plays that role for LoRA.

**[Def 8] Gauge group.** The gauge group of $\rho$ is $\mathrm{Aut}_{\mathbf{Diff}/\Theta}(Q,\rho) = \{\varphi\in\mathrm{Diff}(Q):\rho\circ\varphi=\rho\}$. This is an *unpointed* automorphism, since gauge moves $q_0$ inside its fibre. In practice we exhibit a Lie group $K$ acting on $Q$ with $\rho$ $K$-invariant. The **generic gauge** is $K$ when it acts freely and transitively on the generic fibre $\rho^{-1}(\rho(q))$; then the generic rank of $d\rho$ is $\dim Q - \dim K$. The **effective dimension** of $M$ is $\dim S_M = \operatorname{rank}_{\text{generic}} d\rho$. The **gauge waste** is $\dim Q-\dim S_M$.

**[Def 9] Position of the base point.** $\theta_0$ is an **apex** of $M$ if $S_M(W_0)$ is a cone with vertex $W_0$ and $W_0$ is a singular point; there $\operatorname{rank} d\rho_{q_0}<\dim S_M$. Otherwise $\theta_0$ is a **smooth point**.

**[Def 10] Optimizer gauge group.** An update rule $U$ sends $(q,\nabla_q\mathcal{L},\text{state})$ to a step $\delta q$. Its **optimizer gauge group** is the subgroup $H_U\subseteq K$ under which $U$ is equivariant: $U(k\cdot q, k_*\nabla, k\cdot\text{state}) = dk(\delta q)$. $U$ *descends* to the reachable set (it induces a step on $\Theta$ that depends only on $\rho(q)$) iff $H_U = K$ locally.

**[Def 11] Noether charge.** For gradient flow $\dot q = -\nabla_q(\mathcal{L}\circ\rho)$ (Euclidean, or with a fixed diagonal metric of per-block learning rates), a **Noether charge** is a function $J:Q\to V$ that is constant along every trajectory for every smooth loss $\mathcal{L}$ on $\Theta$.

---

## 3. Propositions and theorems

### 3.1 The interval and the Kleinian invariants

**[Obs 1] The interval $[\text{frozen},\text{full FT}]$.** In $\mathbf{Rep}_*(\Theta,\theta_0)$ the frozen model $(\mathrm{pt},*,\theta_0)$ is initial and full fine-tuning $(\Theta,\theta_0,\mathrm{id})$ is terminal. *Proof.* The only pointed map $\mathrm{pt}\to Q$ is $*\mapsto q_0$, and it commutes over $\Theta$ because $\rho(q_0)=\theta_0$. A map $h$ into the terminal object must satisfy $\mathrm{id}\circ h=\rho$, so $h=\rho$. $\square$
*What this explains.* Trivial but useful: every method sits between "do nothing" and "do everything". The map to the terminal object *is* $\rho$, so the "merge into the base weights" operation is the canonical morphism to full FT. Methods without such a morphism (Def 5, non-mergeable) live in a different slice, over $\Theta\times P'$.

**[Thm 2] Complete invariants of the basic actions (orbit classification).** Let $W,W'\in\mathbb{R}^{m\times n}$.
1. (Right orthogonal, OFT side) $W'=WR$ for some $R\in O(n)$ iff $WW^{\top}=W'W'^{\top}$.
2. (Left orthogonal) $W'=PW$ for some $P\in O(m)$ iff $W^{\top}W=W'^{\top}W'$.
3. (Two-sided, spectral) $W'=PWR$ for some $(P,R)\in O(m)\times O(n)$ iff $W$ and $W'$ have the same singular values.
4. (Left positive torus, (IA)$^3$/DoRA side) for $W,W'$ without zero rows, $W'=DW$ with $D\in T_m^{+}$ iff the rows have the same directions, $N(W)=N(W')$, where $N$ normalizes each row.
5. (Coordinate torus $T_{mn}$ acting by Hadamard product) $W'=E\odot W$ with $E$ entrywise nonzero iff $W$ and $W'$ have the same zero pattern.

*Proof.* (1) Let $w_i,w_i'\in\mathbb{R}^n$ be the rows. If all inner products agree, the linear map $\mathrm{span}(w_i)\to\mathrm{span}(w_i')$, $\sum c_iw_i\mapsto\sum c_iw_i'$, is well defined and isometric: $\|\sum c_iw_i\|^2=c^{\top}WW^{\top}c$. Extend it by an isometry between the orthogonal complements, which have equal dimension, to get $R^{\top}\in O(n)$ with $w_i'=R^{\top}w_i$, that is $W'=WR$. The converse is immediate. (2) is (1) applied to $W^{\top}$. (3) follows from the SVD. (4) and (5) are immediate. $\square$
(1) is the first fundamental theorem of invariant theory for $O(n)$ in its set-theoretic form: the orbit space $\mathbb{R}^{m\times n}/O(n)$ is the cone of PSD $m\times m$ matrices of rank $\le n$, via $W\mapsto WW^{\top}$.
*What this explains.* In our convention OFT, BOFT, HRA and Givens-OFT act on the input side, $W=W_0R$. They therefore preserve exactly the neuron Gram $K_{\text{out}}=WW^{\top}$: all neuron norms and pairwise angles, hence the hyperspherical energy that OFT was designed to preserve. The orbit classification says more: *full* OFT reaches the entire fibre $\{W:WW^{\top}=W_0W_0^{\top}\}$, and that fibre is the largest set any input-side orthogonal method can reach. Block-diagonal OFT with $R=\mathrm{diag}(R_1,\dots,R_{n/b})$ preserves each partial Gram $W_{:,J}W_{:,J}^{\top}$ for every block $J$ of input coordinates. These are $n/b$ invariants, strictly more than full OFT, and they quantify what block-diagonality costs. Output-side orthogonal methods would preserve $K_{\text{in}}$, that is $\|Wx\|$ for every input. "Which Gram is preserved" is not a convention but an invariant of the method.

**[Prop 3] Descent to function space (covariance ⇒ well-defined on the moduli of models).** Let $G_{\text{arch}}$ be a group of function-preserving reparametrizations, so $f(g\theta,-)=f(\theta,-)$. Examples: permutations of hidden neurons; for a head, $(W_Q,W_K)\mapsto(W_Qg,W_Kg^{-\top})$ with $g\in GL_{d_h}$ in the row-vector convention $QK^{\top}=XW_QW_K^{\top}X^{\top}$; rotations of the residual stream after RMSNorm gains are fused, as in QuaRot/SliceGPT. If every element of $G_{\text{arch}}$ acts layerwise through $\mathcal{G}_M$, then the set of *functions* reachable by $M$ from $g\theta_0$ equals the set reachable from $\theta_0$. *Proof.* $\{f(\theta):\theta\in S_M(g\theta_0)\}=\{f(g\theta):\theta\in S_M(\theta_0)\}=\{f(\theta):\theta\in S_M(\theta_0)\}$. $\square$
*What this explains.* This is trivial-but-true, and it sorts methods sharply. LoRA, DoRA's image, BitFit, OFT on the left factor, and full FT have $\mathcal{G}_M\supseteq GL_m\times\{1\}$ or larger (Table 1), so they descend through any function-preserving linear change of basis of the residual stream that they see. (IA)$^3$, LoHa, HiRA, diff pruning, FISH masks and LayerNorm tuning are only $\mathrm{Mon}$-covariant: they descend through neuron permutations but **not** through rotations. **[Conj A]** Fine-tuning a rotated but function-identical model (QuaRot- or SliceGPT-style residual rotation) leaves the reachable function set of LoRA/OFT unchanged. For $\mathrm{Mon}$-covariant methods it changes the reachable set, and measurably so, because LLM residual streams have a privileged basis (outlier features). A second, subtler point: *dynamics* covariance is set by the optimizer (Thm 10), and Adam is only signed-permutation equivariant. LoRA trained with Adam therefore does not descend through rotations either, even though its reachable set does.

### 3.2 The gauge of LoRA and its singular apex

**[Thm 4] LoRA fibres are $GL_r$-torsors; the base point is the apex.** Let $\rho(B,A)=W_0+BA$ on $Q=\mathbb{R}^{m\times r}\times\mathbb{R}^{r\times n}$, with $r\le\min(m,n)$, and let $GL_r$ act by $(B,A)\cdot g=(Bg,g^{-1}A)$.
(a) On $Q^{\circ}=\{\operatorname{rank}B=\operatorname{rank}A=r\}$ the action is free and proper, every fibre of $\rho|_{Q^\circ}$ is exactly one orbit, and $\rho$ induces a diffeomorphism $Q^{\circ}/GL_r\cong W_0+M_r$. Hence $\dim S_{\text{LoRA}}=r(m+n-r)$ and the gauge waste is exactly $r^2$ per matrix.
(b) At the standard initialization $q_0=(0,A_0)$ with $\operatorname{rank}A_0=r$: $d\rho_{q_0}(\delta B,\delta A)=\delta B\,A_0$. So $\operatorname{im}d\rho_{q_0}=\{X:\operatorname{row}(X)\subseteq\operatorname{row}(A_0)\}$ has dimension $mr<r(m+n-r)$. Also $\nabla_A(\mathcal{L}\circ\rho)(q_0)=B_0^{\top}G=0$, and near $q_0$ the fibre is $\{0\}\times\{A:\operatorname{rank}A=r\}$, of dimension $rn$, much larger than the $r^2$-dimensional gauge orbit.
*Proof.* (a) If $BA=B'A'$ with all four of rank $r$, then $\operatorname{col}(B)=\operatorname{col}(BA)=\operatorname{col}(B')$, so $B'=Bg$ for a unique $g\in GL_r$. Then $BA=BgA'$ and the injectivity of $B$ give $A'=g^{-1}A$. Freeness: $Bg=B$ forces $g=I$. Properness: $g=(B^{\top}B)^{-1}B^{\top}B'$ depends continuously on the pair. Since $\rho$ is a submersion onto $W_0+M_r$ on $Q^\circ$ (its differential $(\delta B,\delta A)\mapsto\delta BA+B\delta A$ maps onto $T_{BA}M_r=\{UX+YV^{\top}\}$), the quotient is a diffeomorphism. (b) Direct differentiation; the fibre statement holds because $BA=0$ with $A$ of rank $r$ forces $B=0$. $\square$ **[Num]** At $(m,n,r)=(8,7,2)$: generic rank $26=r(m+n-r)$; at $B=0$, rank $16=mr$.
*What this explains.* (i) LoRA's image is the affine cone $W_0+M_{\le r}$, and **standard LoRA starts at its vertex**, the most singular point. There the tangent space collapses to the tangent cone, and at first order only $B$ moves, inside the random row space of $A_0$. This one fact explains LoRA-FA (freezing $A$ loses nothing at first order), LoRA-GA (choose $\operatorname{row}(A_0)$, and $B_0$, to align with the top singular directions of the first-step full gradient), EVA (choose $\operatorname{row}(A_0)$ as the top right-singular directions of the input activations, the data-optimal frame for the first step), and the slow start that LoRA+ fights (Prop 9). (ii) The $r^2$ redundant parameters are pure gauge. They cost nothing in expressivity, but they *do* change the optimization, because the Euclidean metric on $Q$ is not $GL_r$-invariant.

**[Thm 5] Gauge census and effective dimensions.** For generic parameters, the generic gauge and effective dimension are as follows. **[Num]** Each row was confirmed by Jacobian rank on random instances.

| method | $\rho$ | generic gauge $K$ | $\dim K$ | $\dim S_M$ |
|---|---|---|---|---|
| LoRA, rsLoRA, LoRA+ | $W_0+sBA$ | $GL_r$ | $r^2$ | $r(m+n-r)$ |
| DoRA | $\mathrm{diag}(\mu)N(W_0+BA)$ | $GL_r$ | $r^2$ | $r(m+n-r)+m$ |
| VeRA | $W_0+\Lambda_bB\Lambda_dA$ ($A,B$ frozen) | $\mathbb{R}^{\times}$: $(b,d)\mapsto(\lambda b,\lambda^{-1}d)$ | 1 | $m+r-1$ |
| NOLA | $W_0+(\sum\alpha_iB_i)(\sum\beta_jA_j)$ | $\mathbb{R}^{\times}$ | 1 | $k+l-1$ |
| LoKr ($C\otimes D$) | $W_0+C\otimes D$ | $\mathbb{R}^{\times}$ | 1 | $m_1n_1+m_2n_2-1$ |
| TT / tensor-train adapter | contraction of cores | $\prod_eGL_{r_e}$ over bonds | $\sum_er_e^2$ | $\#\text{params}-\sum_er_e^2$ |
| HRA ($r$ reflections, $W_0$ injective) | $W_0\prod_i(I-2u_iu_i^{\top}/\|u_i\|^2)$ | see Thm 12 | $r(r+1)/2$ | $rn-r(r+1)/2$ |
| **LoHa** | $W_0+(B_1A_1)\odot(B_2A_2)$ | $GL_r^2\times(T_m\times T_n)/\mathbb{R}^{\times}$ | $2r^2+m+n-1$ | $2r(m+n)-2r^2-(m+n-1)$ |
| LoRA-FA, LoRA-XS, FourierFT, BitFit, masks, MoRA | linear in $q$ | trivial (FourierFT: one null direction per sampled conjugate pair $(u,v),(-u,-v)$) | 0 | $\dim Q$ |

*Proof (LoHa row, the non-obvious one).* Besides the two LoRA gauges, the torus $T_m\times T_n$ acts by $(B_1,A_1,B_2,A_2)\mapsto(D_1B_1,A_1D_2,D_1^{-1}B_2,A_2D_2^{-1})$. This preserves the product, because $(D_1XD_2)\odot(D_1^{-1}YD_2^{-1})=X\odot Y$ entrywise. The scalar subgroup $\{(\lambda I,\lambda^{-1}I)\}$ acts trivially on $(X,Y)$ and is already counted, so the torus adds $m+n-1$ dimensions. This gives an upper bound on $\dim S_M$. Equality is the generic-rank computation: **[Num]** $(m,n,r)=(8,7,2),(12,10,2),(12,10,3),(20,16,2)$ give ranks $38,59,93,101$, exactly the formula. The other rows are standard. VeRA and NOLA are rank-1 bilinear in two coefficient vectors. TT is the Holtz–Rohwedder–Schneider dimension formula for fixed-TT-rank manifolds. FourierFT's real-part cosine basis satisfies $\cos(2\pi(uj/m+vk/n))=\cos(2\pi((m-u)j/m+(n-v)k/n))$. $\square$
*What this explains.* (i) **LoHa versus LoRA at equal budget.** LoHa$_r$ and LoRA$_{2r}$ both have $2r(m+n)$ parameters. LoRA$_{2r}$ reaches a $2r(m+n)-4r^2$-dimensional set of rank $\le 2r$. LoHa$_r$ reaches rank up to $r^2$ but only a set of dimension $2r(m+n)-2r^2-(m+n-1)$. Since $m+n-1>2r^2$ for every realistic layer, **LoHa trades about $m+n$ dimensions of reachable set for higher rank.** The Hadamard product spends a whole torus on gauge. This is a sharp, testable version of the folklore "LoHa is high-rank but constrained". (ii) VeRA's whole reachable set is $(m+r-1)$-dimensional: the trainable vectors explore a thin, randomly oriented subvariety, which explains why VeRA needs much larger $r$ to match LoRA. (iii) Gauge waste is a design criterion: for a tensor network, parameters minus $\sum_e r_e^2$ is the honest count to report.

### 3.3 Noether, balance, and the trilemma

**[Thm 6] Noether theorem for gradient flow on a linear gauge.** Let a Lie group $K$ act linearly on $Q=\mathbb{R}^N$ with $\rho\circ k=\rho$. Let $\dot q=-\Lambda\nabla(\mathcal{L}\circ\rho)$ with $\Lambda$ a fixed symmetric positive-definite "learning-rate metric", for example per-block learning rates. For every generator $\xi\in\mathfrak{k}$ (a matrix on $Q$) such that $\Lambda^{-1}\xi$ is symmetric, the quadratic form $J_\xi(q)=\tfrac12\langle q,\Lambda^{-1}\xi q\rangle$ is conserved. Generators with $\Lambda^{-1}\xi$ antisymmetric give no charge; instead the flow is equivariant under them.
*Proof.* Invariance gives $\langle\nabla(\mathcal{L}\circ\rho)(q),\xi q\rangle=0$ for all $q$. Then $\frac{d}{dt}J_\xi=\langle\Lambda^{-1}\xi q,\dot q\rangle=-\langle\Lambda^{-1}\xi q,\Lambda\nabla\rangle=-\langle\xi q,\nabla\rangle=0$, using the symmetry of $\Lambda^{-1}\xi$ and of $\Lambda$. If $\Lambda^{-1}\xi$ is antisymmetric, $e^{t\xi}$ is a $\Lambda^{-1}$-isometry, so it maps trajectories to trajectories. $\square$

**Corollary 6.1 (LoRA charge, with LoRA+ rates and weight decay).** Let $\dot B=-\eta_B(GA^{\top}+\lambda B)$, $\dot A=-\eta_A(B^{\top}G+\lambda A)$ with $G=\nabla\mathcal{L}(W_0+BA)$. Then
$$C_{\eta}:=\frac{B^{\top}B}{\eta_B}-\frac{AA^{\top}}{\eta_A}\in\mathrm{Sym}_r\quad\text{satisfies}\quad\dot C_\eta=-2\lambda\,C_\eta,\ \text{i.e.}\ C_\eta(t)=e^{-2\lambda t}C_\eta(0).$$
*Proof.* Compute directly: $\frac{d}{dt}B^{\top}B=-\eta_B(AG^{\top}B+B^{\top}GA^{\top})-2\eta_B\lambda B^{\top}B$ and $\frac{d}{dt}AA^{\top}=-\eta_A(B^{\top}GA^{\top}+AG^{\top}B)-2\eta_A\lambda AA^{\top}$. Divide by $\eta_B$ and $\eta_A$ and subtract. Equivalently, apply Thm 6 to the generators $\xi_X(B,A)=(BX,-XA)$ with $X$ symmetric. $\square$ **[Num]** Small-step flow gives relative drift $\sim10^{-4}$ (an $O(h)$ Euler error) for $\lambda=0$ and for $\eta_B/\eta_A=4$, and matches $e^{-2\lambda t}$ for $\lambda=0.5$.
Discrete GD instead gives $C_{k+1}=C_k+\eta^2(AG^{\top}GA^{\top}-B^{\top}GG^{\top}B)$: the charge drifts at $O(\eta^2)$. Adam has no such charge.
*What this explains.* (i) **Cartan split of the gauge.** $\mathfrak{gl}_r=\mathfrak{o}(r)\oplus\mathrm{Sym}_r$. The compact half $\mathfrak{o}(r)$ is harmless, since Euclidean GD is equivariant under it. The non-compact half $\mathrm{Sym}_r$, of dimension $r(r+1)/2$, is exactly where the representative matters, and gradient flow *automatically fixes it* through the charge $C$. With standard init $B_0=0$, $C\equiv-A_0A_0^{\top}\approx-\tfrac13I_r$ under the Kaiming-uniform default with bound $1/\sqrt n$, so $\mathrm{Var}=1/(3n)$. The flow is confined to the **unbalanced** level set $AA^{\top}=B^{\top}B+A_0A_0^{\top}$ forever. (ii) Weight decay drives the charge to zero exponentially: **weight decay is gauge fixing to the balanced slice** $B^{\top}B=AA^{\top}$. (iii) The same theorem gives VeRA's charge $\|b\|^2-\|d\|^2\equiv-r\,d_{\text{init}}^2$ (init $b=0$, $d=d_{\text{init}}\mathbf{1}$). **[Conj B]** Under SGD-like training, $\|d\|^2-\|b\|^2$ stays near $r\,d_{\text{init}}^2$; under AdamW it relaxes. This is a cheap, falsifiable diagnostic.

**[Thm 7] Balance–escape trilemma.** A LoRA initialization cannot be all three of: (a) *identity at init with no splitting*, $B_0A_0=0$; (b) *balanced*, $B_0^{\top}B_0=A_0A_0^{\top}$; (c) *non-stationary*, with nonzero initial velocity of $W$ under gradient flow for some loss.
*Proof.* Assume (a) and (b). Then $0=B_0^{\top}B_0A_0=A_0A_0^{\top}A_0$, so $(A_0^{\top}A_0)^2=A_0^{\top}(A_0A_0^{\top}A_0)=0$. Since $A_0^{\top}A_0$ is PSD, it vanishes, so $A_0=0$, and then $B_0^{\top}B_0=0$ gives $B_0=0$. At $(0,0)$ we have $\dot W=-GA^{\top}A-BB^{\top}G=0$ for every $G$: a critical point (a saddle) of $\mathcal{L}\circ\rho$. $\square$
*What this explains.* This is the structural reason the field splits into two families. **Zero-product inits** (LoRA, LoRA-FA, EVA, DoRA's LoRA part, VeRA, AdaLoRA with $\Lambda_0=0$) must be unbalanced; their charge is $C_0=-A_0A_0^{\top}\neq0$. **Balanced or "informative" inits** (PiSSA, MiLoRA, and in spirit CorDA, OLoRA, LoRA-GA, LoftQ) must have $B_0A_0\neq0$. To keep $\rho(q_0)=\theta_0$ they are *forced* to subtract and freeze a residual $W_{\text{res}}=W_0-B_0A_0$. PiSSA's residual is not a trick; it is the price of balance. Indeed PiSSA's $B_0=U_rS_r^{1/2}$, $A_0=S_r^{1/2}V_r^{\top}$ is exactly balanced: $B_0^{\top}B_0=A_0A_0^{\top}=S_r$ (**[Num]** $10^{-14}$). By Cor 6.1, PiSSA then stays balanced under gradient flow forever. On the balanced slice, with $\Delta=BA$, the induced dynamics closes on $W$ alone (Arora–Cohen–Hazan, depth 2):
$$\dot W=-\big(\Delta\Delta^{\top}\big)^{1/2}G-G\big(\Delta^{\top}\Delta\big)^{1/2}.$$
This ODE is gauge-free, and it is stationary at $\Delta=0$, which is the trilemma again. **HRA analogue.** Reflections $H_u=I-2uu^{\top}/\|u\|^2$ have determinant $-1$, and a product of reflections in mutually orthogonal hyperplanes is $I-2P_V\neq I$. So HRA's identity-at-init (pairs $u_{2i-1}=u_{2i}$, which give $H_uH_u=I$) is incompatible with the strict orthonormality of its regularizer. HRA too must start off its gauge-fixed slice.

### 3.4 Polar decomposition: what DoRA really adds

**[Thm 8] DoRA is the torus saturation of LoRA.** Let $N$ normalize rows (DoRA's per-output-channel norm, matching the HF PEFT magnitude vector of length $d_{\text{out}}$). On weights without zero rows,
$$S_{\text{DoRA}_r}(W_0)=\{\mathrm{diag}(\mu)N(W_0+BA)\}=T_m\cdot\big(W_0+M_{\le r}\big)=S_{(\text{IA})^3_{\text{out}}\circ\text{LoRA}_r}(W_0),$$
the reachable set of "LoRA followed by a trainable output-channel scale". Its dimension is $r(m+n-r)+m$ (**[Num]** 34 at $(8,7,2)$). The image covariance group is $\mathrm{Mon}_m\times GL_n$. The *parametrization* (the normalization) is only $\mathrm{Mon}_m\times O(n)$-covariant.
*Proof.* $N(V)=\mathrm{diag}(\|v_i\|^{-1})V\in T_m\cdot V$, so $\mathrm{diag}(\mu)N(V)$ ranges over $T_m\cdot V$ as $\mu$ ranges over $\mathbb{R}^m$. Dimension: the gauge of $(\mu,B,A)\mapsto\mathrm{diag}(\mu)(W_0+BA)$ is generically just $GL_r$, since $\mathrm{diag}(c)(W_0+X)=W_0+X'$ forces $(\mathrm{diag}(c)-I)W_0$ to have rank $\le2r$, which is non-generic. Covariance: $g\,T_m(W_0+M_{\le r})h^{-1}=T_m(gW_0h^{-1}+M_{\le r})$ iff $g$ normalizes $T_m$. $\square$
*What this explains.* At the level of reachable sets, DoRA's "magnitude–direction decoupling" is exactly LoRA plus $m$ free output-channel gains. It is a semidirect combination of a translation-cone method and the positive diagonal torus, so polar decomposition is the right word. Everything else DoRA changes lives in the metric: the gradient with respect to the direction is projected orthogonally to each row, as in weight normalization, and the paper further detaches the norm from the graph. **[Conj D]** "LoRA $+$ trainable output scale" (no normalization) has *identical* expressivity to DoRA. Any accuracy gap between them therefore isolates the effect of the normalization metric. This controlled experiment is suggested directly by Thm 8.

### 3.5 Optimizer geometry: the gauge acts on the optimizer

**[Thm 9] LoRA+ is a gauge transformation of initialization (for SGD).** Run (stochastic) gradient descent on LoRA with no weight decay, $B_0=0$, step sizes $(\eta_B,\eta_A)$, and ratio $\lambda=\eta_B/\eta_A$. It produces *exactly* the same sequence $W_k$ as plain LoRA with the single step size $\bar\eta=\sqrt{\eta_A\eta_B}$ and initialization $A_0'=\lambda^{1/4}A_0$. The same holds for gradient flow and for heavy-ball momentum.
*Proof.* Put $B'=\kappa B$ and $A'=\kappa^{-1}A$ with $\kappa=\lambda^{-1/4}$. This is a gauge transformation, so $B'A'=BA$ and $G$ is unchanged. Then $B'_{k+1}=\kappa(B_k-\eta_BGA_k^{\top})=B'_k-\kappa^2\eta_BGA_k'^{\top}$ and $A'_{k+1}=A'_k-\kappa^{-2}\eta_AB_k'^{\top}G$, with $\kappa^2\eta_B=\kappa^{-2}\eta_A=\bar\eta$. Initial data: $B'_0=0$ and $A'_0=\lambda^{1/4}A_0$. Momentum buffers are linear in the gradients and transform the same way. $\square$ **[Num]** The relative trajectory difference is $0.0$ to machine precision, against $0.45$ for the control.
*Scalar closed form.* For $w=ab$ under gradient flow, the charge $b^2/\eta_B-a^2/\eta_A=-a_0^2/\eta_A$ eliminates the gauge and gives the gauge-free ODE
$$\dot w=-\sqrt{\eta_B^2a_0^4+4\eta_A\eta_B\,w^2}\;\ell'(w).$$
The speed near the apex is $\eta_Ba_0^2$, which depends only on $\eta_B$; far from it the speed is $\approx2\sqrt{\eta_A\eta_B}|w|$, which depends only on the geometric mean.
*What this explains.* (i) For SGD, LoRA+'s ratio is not a new degree of freedom: it is an init scale, $\lambda^{1/4}$, in disguise. It speeds the escape from the singular apex (Thm 4) and is irrelevant far from it. (ii) LoRA+ is used with Adam, and Adam's optimizer gauge group (Thm 10) contains no scalings, so with Adam the ratio is *not* absorbable. This is consistent with Hayou–Ghosh–Yu's analysis, which is genuinely about adaptive updates at large width. The Erlangen view tells you *where* their effect has to come from. (iii) rsLoRA's $\alpha/\sqrt r$ is likewise invisible in $\mathbf{Rep}_*$: $(B,A)\mapsto(sB,A)$ is an isomorphism of pointed reparametrizations. Its content is entirely in how the metric scales with $r$.

**[Thm 10] Optimizer gauge groups.** On LoRA, with $K=GL_r$:
1. GD/SGD (any per-factor step sizes): $H=O(r)$.
2. Adam/AdamW (coordinatewise; any $\beta_1,\beta_2,\epsilon$): $H=B_r$, the finite signed-permutation group of order $2^rr!$.
3. Scaled GD / Riemannian-preconditioned LoRA, with $\delta B=-\eta\nabla_B(AA^{\top})^{-1}$ and $\delta A=-\eta(B^{\top}B)^{-1}\nabla_A$: $H=GL_r$. The induced step is
$$\delta W=-\eta\big(G\,P_{\operatorname{row}A}+P_{\operatorname{col}B}\,G\big)+\eta^2\,GA^{\top}(AA^{\top})^{-1}(B^{\top}B)^{-1}B^{\top}G,$$
and both terms depend only on $W$. It therefore descends to the manifold $W_0+M_r$.
*Proof.* Under $(Bg,g^{-1}A)$ the gradients transform as $\nabla_B\mapsto\nabla_Bg^{-\top}$ and $\nabla_A\mapsto g^{\top}\nabla_A$. (1) $\delta B'=-\eta\nabla_Bg^{-\top}$ equals $\delta B\,g$ iff $g^{-\top}=g$. (2) Coordinatewise rules commute with permutations and with sign flips (the moments $m$ flip sign, $v$ is unchanged). A non-monomial $g$ mixes coordinates, and a non-unit monomial scale $c$ rescales gradients by $c^{-1}$ while the Adam step is scale-free, so $\delta B'\neq\delta Bg$. (3) $(B'^{\top}B')^{-1}\nabla_{A'}=g^{-1}(B^{\top}B)^{-1}\nabla_A$ and $\nabla_{B'}(A'A'^{\top})^{-1}=\nabla_B(AA^{\top})^{-1}g$. Substitute to get $\delta W$, whose terms are visibly invariant. $\square$ **[Num]** Equivariance defects ($\approx10^{-14}$ means equivariant): SGD gives $O(r)$: $10^{-14}$, $GL_r$: $5.4$. Scaled GD gives $10^{-13}$ for all three. First-step Adam gives signed permutations: $10^{-14}$, $O(r)$: $0.61$.
*What this explains.* This is a strict hierarchy $B_r\subset O(r)\subset GL_r$ of optimizer geometries. Only the top level (Riemannian preconditioning à la Mishra–Sepulchre, ScaledGD, Riemannian-preconditioned LoRA, and LoRA-RITE, which is built to be transformation-invariant *and* adaptive) defines an optimizer on the model rather than on a chart. Plain Adam, the default, sees only a finite shadow of the gauge. That is why ad-hoc fixes (LoRA+, balanced or orthonormal inits, rsLoRA scaling) work: each fixes part of the non-compact gauge by hand.

**[Cor 10.1] What a zero-$B$ initialization really chooses.** At $q_0=(0,A_0)$ the gauge-invariant content of $A_0$ depends on the optimizer. For a $GL_r$-equivariant optimizer it is $\operatorname{row}(A_0)\in\mathrm{Gr}(r,n)$. For SGD it is the rank-$r$ PSD matrix $A_0^{\top}A_0$ (Thm 2.2), which is exactly the initial preconditioner $\dot W(0)=-\eta_BGA_0^{\top}A_0$. For Adam it is $A_0$ up to signed permutation. Random-$A$ LoRA, EVA (activation-SVD frame) and LoRA-FA are therefore **different points of the same fibre $\rho^{-1}(\theta_0)$**, which is jumped-up at the apex. Their reachable set is identical, $W_0+M_{\le r}$. They differ only in where they start inside the fibre.

### 3.6 Apex, pullbacks, generated groups

**[Thm 11] Splitting the base point changes the reachable set (LoRA vs PiSSA).** Let $P=U_rS_rV_r^{\top}$ be the top-$r$ SVD part of $W_0$ (rank $r$, with $r<\min(m,n)$) and $W_{\text{res}}=W_0-P$. Then (a) $S_{\text{LoRA}_r}=W_0+M_{\le r}$ has $W_0$ as apex, while $S_{\text{PiSSA}_r}=W_{\text{res}}+M_{\le r}$ contains $W_0$ as a **smooth** point; (b) the two sets are **incomparable**; (c) $\text{PiSSA}_r\preceq\text{LoRA}_{2r}$ and $\text{LoRA}_r\preceq\text{PiSSA}_{2r}$.
*Proof.* (a) $W_0-W_{\text{res}}=P\in M_r$. (b) Choose $u\notin\operatorname{col}P$ and $v\notin\operatorname{row}P$. Then $\operatorname{rank}(P+uv^{\top})=\operatorname{rank}(uv^{\top}-P)=r+1$, so $W_0+uv^{\top}\notin S_{\text{PiSSA}}$ and $W_{\text{res}}+uv^{\top}\notin S_{\text{LoRA}}$. (c) $W_{\text{res}}+Y=W_0+(Y-P)$ and $W_0+X=W_{\text{res}}+(P+X)$, and rank is subadditive. $\square$
*What this explains.* "Initialization methods" that freeze a residual (PiSSA, MiLoRA, OLoRA, LoRA-GA, CorDA, LoftQ) are not initializations of LoRA. They are **different objects** of $\mathbf{Rep}_*(\theta_0)$: the same cone, moved so that its apex lies away from $\theta_0$. There is no morphism to or from LoRA. The dichotomy is: **Type I** (same $\rho$, new $q_0$ in the apex fibre: LoRA, EVA, LoRA-FA) versus **Type II** (new $\rho$ via a splitting $\theta_0=\theta_{\text{res}}+\rho(q_0)$: PiSSA and friends). For Type II, which part of $W_0$ is moved into the adapter (principal for PiSSA, minor for MiLoRA, the QR head for OLoRA, gradient-aligned for LoRA-GA, context-weighted for CorDA, quantization residual for LoftQ) selects *which translate* of $M_{\le r}$ you explore. Morphisms do exist inside Type II: LoRA-XS, with $S=W_0+U_rRV_r^{\top}$, maps into PiSSA via $R\mapsto(U_r(S_r+R)S_r^{-1/2},\,S_r^{1/2}V_r^{\top})$. **QLoRA** is not over $\theta_0$ at all: its base point is $\hat\theta_0=\mathrm{deq}(\mathrm{quant}(\theta_0))$, a change of base point $\mathbf{Rep}_*(\theta_0)\rightsquigarrow\mathbf{Rep}_*(\hat\theta_0)$. **LoftQ** is the best *approximately pointed* Type II object over $\hat\theta_0$: it minimizes $\|\theta_0-\hat\theta_0-B_0A_0\|$, so that $\rho(q_0)\approx\theta_0$.

**[Thm 12] HRA is the pullback of LoRA and OFT; its gauge is $r(r+1)/2$.** Let $W_0$ be injective ($\operatorname{rank}W_0=n$, e.g. square invertible) and $r$ even (HRA's identity-at-init pairing). Then
$$S_{\text{HRA}_r}(W_0)=W_0\cdot\{R\in SO(n):\operatorname{rank}(R-I)\le r\}=\big(W_0+M_{\le r}\big)\ \cap\ W_0\cdot SO(n),$$
the fibre product (in $\mathbf{Set}$, hence of the images) of $\mathrm{im}(\text{LoRA}_r)\hookrightarrow\Theta\hookleftarrow\mathrm{im}(\text{OFT}_{SO})$. Its dimension is $r(n-r)+r(r-1)/2$, so the gauge has dimension $rn-\dim S=r(r+1)/2$.
*Proof.* ($\subseteq$) Each $H_u-I$ has rank 1, so a product of $r$ reflections fixes $\bigcap u_i^{\perp}$, and therefore $\operatorname{rank}(R-I)\le r$. Then $\operatorname{rank}(W_0R-W_0)\le r$. ($\supseteq$) If $W_0R=W_0+X$ with $\operatorname{rank}X\le r$, injectivity gives $\operatorname{rank}(R-I)\le r$. By the refined Cartan–Dieudonné theorem, an orthogonal map whose moving space $\operatorname{im}(R-I)$ has dimension $k$ is a product of exactly $k$ reflections. Since $\det R=1$, $k$ is even, so pad with pairs $H_uH_u=I$. Dimension: choose the $k$-dimensional moving space ($\dim\mathrm{Gr}(r,n)=r(n-r)$) and a fixed-point-free element of $O(r)$ (open, of dimension $r(r-1)/2$). $\square$ **[Num]** Jacobian ranks $8,15,21,26$ for $r=1,\dots,4$, $n=9$; this matches $rn-r(r+1)/2$ (odd $r$ uses the other component).
*What this explains.* HRA's title claim, "bridging low-rank and orthogonal adaptation", becomes a precise universal property: HRA is the **meet** of LoRA and OFT in the expressivity lattice. It preserves the neuron Gram $K_{\text{out}}$ *and* has rank-$\le r$ displacement. Its orthogonality regularizer is a soft gauge fixing. Enforced strictly, it collapses the image to $\{W_0(I-2P_V)\}$, of dimension $r(n-r)$, which excludes the identity (Thm 7, HRA analogue).

**[Thm 13] Generated groups: what re-merging can and cannot reach.**
(a) $\underbrace{M_{\le r}+\dots+M_{\le r}}_{k}=M_{\le\min(kr,m,n)}$. So ReLoRA after $k$ merge-and-restart cycles reaches exactly $W_0+M_{\le kr}$, and the group generated by LoRA's displacement cone is all of $(\mathbb{R}^{m\times n},+)$.
(b) **[Obs]** Group-type methods are idempotent under re-merging with a fixed frame: BitFit, fixed masks (FISH), LoRA-XS, FourierFT, MoRA, block-diagonal OFT, (IA)$^3$. Re-merging gains nothing unless the frame is recomputed.
(c) Let $I_1,\dots,I_p\subseteq\{1,\dots,n\}$ be the coordinate blocks of a family of block-orthogonal factors (BOFT's butterfly factors or Givens planes), with $|I_j|\ge2$. The subgroup generated by the $SO(I_j)$ is $SO(n)$ iff the hypergraph $(\{1..n\},\{I_j\})$ is connected. Otherwise it lies in $\prod_cSO(I^{(c)})$ over the connected components.
*Proof.* (a) Subadditivity of rank gives $\subseteq$; split an SVD into $k$ blocks for $\supseteq$. (c) With $L_{ab}=E_{ab}-E_{ba}$, if $a\in I\setminus J$, $c\in I\cap J$, $b\in J\setminus I$, then $[L_{ac},L_{cb}]=L_{ab}$. So $\mathfrak{so}(I)$ and $\mathfrak{so}(J)$ generate $\mathfrak{so}(I\cup J)$. Induct along a spanning tree to get $\mathfrak{so}(n)$. The group generated by connected Lie subgroups is the connected subgroup whose Lie algebra is the generated Lie algebra. If the hypergraph is disconnected, every generator preserves the block decomposition. $\square$
*What this explains.* LoRA is variety-type: one round reaches a cone, and ReLoRA "walks the group the cone generates", which is why it can match full-rank training. Block-diagonal OFT can never escape its $n/b$ partial-Gram invariants (Thm 2), however often it is re-merged. BOFT's butterfly permutations exist precisely to connect the block hypergraph, which destroys those invariants and leaves only the global $K_{\text{out}}$. This is Erlangen in miniature: **the butterfly changes the geometry from $O(b)^{n/b}$ to $SO(n)$**.

### 3.7 Merging, rank allocation, cotangent methods

**[Prop 14] Factor-wise merging is gauge-dependent; freezing $A$ is gauge fixing.** Given LoRA modules $(B_i,A_i)$ and weights $w_i$ with $\sum w_i=1$, let $\Phi_w=(\sum_iw_iB_i)(\sum_jw_jA_j)$ be the factor-wise merge used by LoRAHub-style composition and by FedAvg on factors. Then $\Phi_w-\sum_iw_iB_iA_i=\sum_{i\neq j}w_iw_jB_iA_j-\sum_iw_i(1-w_i)B_iA_i$. This is not invariant under independent gauges $(B_ig_i,g_i^{-1}A_i)$, only under a common $g$. If all modules share a frozen $A_i=A_0$ (LoRA-FA, FFA-LoRA), then $\Phi_w=\sum_iw_iB_iA_0$ exactly.
*Proof.* Expand; for a common $A_0$, linearity in $B$. $\square$
*What this explains.* Task arithmetic, $\theta_0+\sum\lambda_i(\theta_i-\theta_0)$, lives in the translation group of the torsor $\Theta$. It is equivariant under all of $\mathbb{A}_\ell$, and in particular under $G_{\text{arch}}$. TIES (trim, sign election, disjoint mean) and DARE (drop and rescale) are coordinatewise nonlinear, so they are equivariant only under signed coordinate permutations. That is the same Erlangen split as for methods (**[Conj E]**: TIES/DARE outcomes change under function-preserving rotations; task arithmetic does not). Factor-wise merging is *ill-defined on the model* unless the gauge is shared. Freezing $A$ (FFA-LoRA in federated learning) is exactly the gauge fixing that makes it exact. Modules trained from a common seed for $A_0$ start in a common gauge, which partly explains why naive factor merging works at all **[Analogy]**. **Symmetry breaking by the base point.** A generic $\theta_0$ has trivial stabilizer in the discrete group of neuron permutations. Fine-tunes of a common $\theta_0$ therefore inherit a canonical alignment, whereas independently trained models must first be aligned (Git Re-Basin). This is why merging is a fine-tuning phenomenon.

**[Prop 15] Per-rank importance needs a reduced gauge (AdaLoRA, DyLoRA).** (a) The multiset of rank-one summands $\{b_ia_i^{\top}\}$ of $BA$ is preserved by $g\in GL_r$ for all $(B,A)$ iff $g\in\mathrm{Mon}_r=T_r\rtimes S_r$. (b) The nested truncations $(B,A)\mapsto(B_{:,\le k},A_{\le k,:})$ are $g$-compatible for every $k$ iff $g\in T_r$ (diagonal).
*Proof.* (a) $(Bg)_{:,j}=\sum_iB_{:,i}g_{ij}$, so the summands are permuted and rescaled for all $(B,A)$ iff each column of $g$ has a single nonzero entry. (b) For every $k$, $(Bg)_{:,\le k}$ must depend only on $B_{:,\le k}$, which forces $g$ block upper-triangular, and $(g^{-1}A)_{\le k,:}$ must depend only on $A_{\le k,:}$, which forces $g$ block lower-triangular. Together, $g$ is diagonal. $\square$
*What this explains.* Pruning "rank components" of plain LoRA is gauge-dependent. AdaLoRA's SVD-form $P\Lambda Q$ with orthogonality penalty is a soft gauge fixing from $GL_r$ down to $B_r$; on the fixed slice the $\lambda_i$ are gauge-invariant and may be meaningfully scored and pruned. DyLoRA breaks $GL_r$ to the ordered torus $T_r$, a *filtration* $\text{LoRA}_1\rightarrowtail\text{LoRA}_2\rightarrowtail\cdots$ of monomorphisms in $\mathbf{Rep}_*$ (zero-padding commutes with $\rho$) together with retractions. EVA's explained-variance rank allocation and AdaLoRA's budget scheduler are maps from data to a point of this filtration, one per layer.

**[Prop 16] Tangent = cotangent for orthonormal linear frames; GaLore is frozen-$B$ LoRA re-based.** If $\rho(q)=\theta_0+Jq$ with $J^{\top}J=I$, then gradient descent on $q$ is projected gradient descent $\theta\mapsto\theta-\eta JJ^{\top}\nabla\mathcal{L}$, and Adam on $q$ is Adam applied to $J^{\top}\nabla\mathcal{L}$. In particular, within one projection period, GaLore with left projector $P\in\mathbb{R}^{m\times r}$ (with $P^{\top}P=I$, Adam on $P^{\top}G$) is *identical* to LoRA with $B:=P$ frozen and $A$ trained by Adam from $0$, because $\nabla_A\mathcal{L}(W_0+PA)=P^{\top}G$. Switching $P$ every $T$ steps is ReLoRA-style re-basing with a gradient-chosen frame.
*Proof.* Chain rule. $\square$ (Up to GaLore's scale factor and its choice of whether to keep the moments when $P$ changes.)
*What this explains.* The lens-theoretic distinction "restrict the forward chart (LoRA) vs restrict the backward cotangent (GaLore)" is *real only for curved charts*. For linear orthonormal charts the two coincide. What GaLore adds is a **dynamic frame**: a walk on $\mathrm{Gr}(r,m)$ driven by gradient SVD. Flora's random resampling is a random walk on the same Grassmannian. LISA is a random walk on the discrete set of layer-block coordinate subspaces. ReLoRA walks the apex. In each case one round is group-type or variety-type, and the walk generates the full translation group (Thm 13). MeZO is orthogonal to all of this. It replaces the reverse-mode lens by forward evaluations along Gaussian directions, so it is $O(\dim Q)$-equivariant in distribution and composes with any chart (MeZO + LoRA, MeZO + prefix).

### 3.8 Extensions: activation-space methods and mergeability

**[Thm 17] Prefix = gated parallel adapter; the mergeability obstruction.** (a) **[Obs]** A parallel adapter with identity activation, $h=W_0x+W_{\text{up}}W_{\text{down}}x$, *is* LoRA (same $\rho$). (b) (He et al. 2022) For a query $q$, context keys and values $K,V$, and prefix $P_k,P_v\in\mathbb{R}^{\ell\times d}$:
$$\mathrm{Attn}\big(q,[P_k;K],[P_v;V]\big)=(1-\lambda(q))\,\mathrm{Attn}(q,K,V)+\lambda(q)\,\mathrm{softmax}(qP_k^{\top})P_v,\quad\lambda(q)=\frac{\sum_ie^{q\cdot p_i}}{\sum_ie^{q\cdot p_i}+\sum_je^{q\cdot k_j}}.$$
(c) LLaMA-Adapter normalizes the prompt scores separately and multiplies them by a zero-initialized gate $g$. Its output is $\mathrm{Attn}(q,K,V)+g\,\mathrm{softmax}(qP_k^{\top})P_v$: an ungated parallel adapter in activation space, with neutral point $g=0$. (d) **Obstruction.** On single-token inputs every weight setting of a standard attention block is affine in $x$, namely $W_o(W_vx+b_v)+b_o$. The prefix-tuned block is not affine in $x$ for generic $(P_k,P_v)$, so no choice of the original weights reproduces it: prefix tuning, P-tuning v2 and prompt tuning are **not mergeable**. The same holds for nonlinear serial adapters (Houlsby, Pfeiffer) inserted where the base is affine.
*Proof.* (b) Split the softmax normalizer over prefix and context. (d) **[Sketch]** Along $x=tx_1$ the prefix output involves $\mathrm{softmax}(t\,W_qx_1P_k^{\top})$, whose second $t$-derivative at $0$ is nonzero unless $P_v$ is orthogonal to the variance directions of $P_kW_qx_1$. That is a proper algebraic condition. $\square$
*What this explains.* This gives the clean boundary between weight-space methods (an action on $\Theta$, so a morphism to the terminal object exists, which is mergeability) and activation-space methods (an extension, Def 5, which lives over $\Theta\times P'$). Prefix tuning is a parallel adapter whose gate $\lambda(q)$ is *input-dependent*, tied to the softmax mass of the context. LLaMA-Adapter decouples it. Mixtures of LoRA experts and AdapterFusion have input-dependent weights $\Delta W(x)=\sum_ig_i(x)B_iA_i$. They are mergeable iff the gate is constant on the data. Pointwise, each $\Delta W(x)$ lies in $W_0+M_{\le kr}$ for top-$k$ routing. LST runs a side network that reads frozen activations: its backward pass never enters the backbone, so it is the extreme cotangent-free extension. S-LoRA serves thousands of adapters because all of them are objects over the **same base point**: a batch computes one shared $W_0X$ plus per-request $B_iA_ix_i$. Any method of the form $W_i=L_iW_0R_i+\Delta_i$ with cheap structured $L_i,R_i,\Delta_i$ (OFT, (IA)$^3$, DoRA) is batch-servable in the same way. Text-to-LoRA-style hypernetworks output a *representative* $(B,A)$. A reconstruction loss on factors is gauge-dependent, whereas a loss on $BA$, or on an SVD-gauge-fixed representative, is not. That is a concrete design rule.

### 3.9 Torus hierarchy and covariant initialization

**[Prop 18] The torus ladder.** Let HiRA$_r$ be $W=W_0\odot(J+BA)$, where $J$ is the all-ones matrix. Then:
(a) the left (IA)$^3$/SSF scale $\mathrm{diag}(l)W_0=W_0\odot(l\mathbf 1^{\top})\in\text{HiRA}_1$; the right scale (IA)$^3$ on $W_{\text{down}}$, and LayerNorm tuning folded into the next linear map, $W\mathrm{diag}(\gamma)=W\odot(\mathbf 1\gamma^{\top})$, also lie in HiRA$_1$; the LN bias $\beta$ and SSF's shift are BitFit-type translations $Wb\mapsto Wb+W\beta$;
(b) $\operatorname{rank}(W-W_0)\le r\cdot\operatorname{rank}W_0$, and the zero pattern of $W_0$ is an absolute invariant of HiRA, of (IA)$^3$ and of SSF;
(c) the image covariance of all of these is $\mathrm{Mon}\times GL$, $GL\times\mathrm{Mon}$ or $\mathrm{Mon}\times\mathrm{Mon}$, never $GL\times GL$.
*Proof.* (a) Direct; (b) the Hadamard product is a principal submatrix of the Kronecker product, so rank is multiplicative-bounded; (c) monomials normalize the tori and commute with Hadamard products up to relabelling. Conversely, $gT_mg^{-1}=T_m$ forces $g\in\mathrm{Mon}_m$. For HiRA take $W_0=J$ and $(g,h)$ with $gJh^{-1}=E_{11}$: covariance would force $E_{11}\odot M_{\le r}=g(J\odot M_{\le r})h^{-1}=M_{\le r}$, which is false. $\square$
*What this explains.* HiRA is to the coordinate torus $T_{mn}$ what LoRA is to translations: a low-rank chart on its Lie algebra, since $J+BA\approx\exp_{\odot}(BA)$. (IA)$^3$, SSF and LN tuning are the rank-1 row and column subtori. DoRA is the row torus times LoRA (Thm 8). One ladder, four rungs.

**[Prop 19] Covariant data-aware initialization (a design rule).** Let $C=\mathbb{E}[xx^{\top}]$ be the input second moment, with inputs transforming as $x\mapsto hx$ (so $W\mapsto Wh^{-1}$ and $C\mapsto hCh^{\top}$). The split $X^*=\arg\min_{\operatorname{rank}X\le r}\|(W_0-X)L\|_F$ with $LL^{\top}=C$ (whitening) is $GL_n$-covariant. CorDA's split, from the SVD of $W_0C$, minimizes $\|(W_0-X)C\|_F$ and is only $O(n)$-covariant. So are PiSSA, MiLoRA, LoRA-XS (SVD of $W_0$) and EVA (SVD of activations).
*Proof.* $\|(W-X)L\|_F^2=\operatorname{tr}((W-X)C(W-X)^{\top})$ is invariant under $(W,X,C)\mapsto(Wh^{-1},Xh^{-1},hCh^{\top})$. The $C$-weighted version gives $\operatorname{tr}((W-X)C^2(W-X)^{\top})\mapsto\operatorname{tr}((W-X)Ch^{\top}hC(W-X)^{\top})$, which is invariant iff $h^{\top}h=I$. $\square$
*What this explains.* The Erlangen programme yields a concrete design rule: a data-aware Type II init is intrinsic to the network (independent of how the input coordinates of a layer are chosen) iff it uses a square root of $C$. SVD-LLM-style whitening has this property. **[Conj C]** Whitened CorDA or PiSSA is more robust to function-preserving input reparametrizations (for example scaling or rotating the residual stream with compensating weights), and no worse otherwise.

---

## 4. Classification: the Erlangen coordinates of every method

Every method gets a value on seven axes:
**(G)** image covariance group $\mathcal{G}_M$, the geometry (Def 6);
**(T)** type, meaning structure group and orbit shape (Def 7);
**(I)** absolute invariant;
**(R)** relative constraint;
**(K)** generic gauge of $\rho$ (Def 8);
**(P)** position of $\theta_0$ (Def 9);
**(M)** mergeable into $\Theta$ (Def 5).

Abbreviations: *tr-cone* means a translation by a non-linear cone (variety-type); *tr-lin* means a translation by a linear subspace (group-type); *L-$H$* and *R-$H$* mean a left or right action of $H$; *ext* means an extension (activation space); *frame* means covariance only in distribution, through the law of frozen random matrices; $CO(n)=\mathbb{R}^{\times}O(n)$. "Type I" and "Type II" refer to Thm 11.

| method | (G) geometry | (T) type | (I) abs. invariant | (R) relative constraint | (K) gauge | (P) $\theta_0$ | (M) |
|---|---|---|---|---|---|---|---|
| Full FT | $GL\times GL$ | tr-lin, all of $\Theta$ | — | — | 1 | smooth | — |
| LoRA | $GL\times GL$ | tr-cone $M_{\le r}$ | — | $\operatorname{rank}\Delta W\le r$ | $GL_r$ | **apex** | ✓ |
| rsLoRA, LoRA+ | $\cong$ LoRA in $\mathbf{Rep}_*$ | same | — | same | $GL_r$ | apex | ✓ |
| LoRA-FA | $GL_m\times$frame | tr-lin, dim $mr$ | — | $\operatorname{row}\Delta W\subseteq\operatorname{row}A_0$ | 1 | smooth | ✓ |
| EVA | $GL\times GL$ (init $O(n)$, data) | = LoRA, Type I | — | $\operatorname{rank}\le r$, adaptive $r_\ell$ | $GL_r$ | apex | ✓ |
| DoRA | $\mathrm{Mon}_m\times GL_n$ | $T_m\ltimes$ tr-cone (Thm 8) | — | $\exists D:\operatorname{rank}(DW-W_0)\le r$ | $GL_r$ | apex | ✓ |
| PiSSA | $O\times O$ | tr-cone, apex $W_{\text{res}}$ (Type II) | — | $\operatorname{rank}(W-W_{\text{res}})\le r$ | $GL_r$ | smooth, balanced | ✓ |
| MiLoRA | $O\times O$ | tr-cone, apex $W_0-$minor part | — | same form | $GL_r$ | smooth, balanced | ✓ |
| OLoRA | $O(m)\times\mathrm{Borel}(n)$ | tr-cone, apex $W_0-Q_rR_r$ | — | same form | $GL_r$ | smooth, unbalanced | ✓ |
| LoRA-GA | $O\times O$ (gradient data) | tr-cone, gradient-chosen apex | — | same form | $GL_r$ | smooth | ✓ |
| CorDA | $O\times O$ (data, Prop 19) | tr-cone, apex from SVD$(W_0C)$ | — | same form | $GL_r$ | smooth | ✓ |
| LoftQ | $\mathrm{Mon}$ (quant grid) | tr-cone over $\hat\theta_0$ | — | same form | $GL_r$ | smooth, $\approx$pointed | ✓ (to $\hat\Theta$) |
| QLoRA | $\mathrm{Mon}$ (quant grid) | LoRA over $\hat\theta_0\neq\theta_0$ | — | $\operatorname{rank}(W-\hat W_0)\le r$ | $GL_r$ | apex at $\hat\theta_0$ | ✓ (to $\hat\Theta$) |
| AdaLoRA | $GL\times GL$ (param. $O\times O$) | tr-cone, $r_\ell(t)$ scheduled | — | $\operatorname{rank}\le r_\ell(t)$ | $GL_r\to B_r$ (soft) | apex | ✓ |
| DyLoRA | $GL\times GL$ | filtration of cones | — | nested ranks | $GL_r\to T_r$ | apex | ✓ |
| ReLoRA | $GL\times GL$ | walk of apex, reaches $W_0+M_{\le kr}$ | — | $\operatorname{rank}\le kr$ after $k$ cycles | $GL_r$/cycle | apex, re-based | ✓ |
| VeRA | $\mathrm{Mon}_m\times$frame | tr-variety, dim $m+r-1$ | — | $\Delta W\in\Lambda B\Lambda A$ | $\mathbb{R}^{\times}$ | apex | ✓ |
| NOLA | frame | tr-variety, dim $k+l-1$ | — | random-basis bilinear | $\mathbb{R}^{\times}$ | apex | ✓ |
| VB-LoRA | frame (shared bank) | piecewise tr-variety, cross-layer limit | — | top-$k$ bank mixtures | bank symmetries | apex | ✓ |
| LoRA-XS | $O\times O$ | tr-lin, $\Delta W\in U_r\mathfrak{gl}_rV_r^{\top}$ | — | $\operatorname{col}\subseteq U_r$, $\operatorname{row}\subseteq V_r$ | 1 | smooth | ✓ |
| FourierFT | DFT symmetries | tr-lin, dim $n$ (cosine basis) | — | spectral support $\Omega$ | conj. pairs | smooth | ✓ |
| MoRA | compressor structure | tr-lin, dim $\hat r^2$, high rank (e.g. $I\otimes M$) | — | fixed block structure | 1 | smooth | ✓ |
| LoHa | $\mathrm{Mon}\times\mathrm{Mon}$ | tr-variety $\subset M_{\le r^2}$ | — | $\Delta W\in M_{\le r}\odot M_{\le r}$ | $GL_r^2\times T_m\times T_n/\mathbb{R}^{\times}$ | apex | ✓ |
| LoKr | $\otimes$-subgroup | tr-variety $C\otimes D$ | — | Kronecker form | $\mathbb{R}^{\times}(\times GL_r)$ | apex | ✓ |
| TT adapters | tensor-factor | tr-variety (TT manifold) | — | TT-ranks | $\prod_eGL_{r_e}$ | apex/neutral | ✓ / ext |
| Compacter | $\otimes$ (PHM) | ext with $\sum_iA_i\otimes B_i$ weights | — | Kronecker rank | $GL$ on PHM index | neutral | ✗ |
| HiRA | $\mathrm{Mon}\times\mathrm{Mon}$ | chart on coordinate torus $T_{mn}$ | zero pattern | $\operatorname{rank}\Delta W\le r\operatorname{rank}W_0$ | $GL_r$ | apex | ✓ |
| OFT | $GL_m\times CO(n)$ (block: normalizer) | R-$O(b)^{n/b}$ | $K_{\text{out}}$, partial Grams | — | $\mathrm{Stab}\cong O(n-\operatorname{rk}W_0)$ | smooth | ✓ |
| BOFT | $GL_m\times$ butterfly normalizer | R-products of butterfly $O$ | $K_{\text{out}}$ | — | inter-factor | smooth | ✓ |
| HRA | $GL_m\times CO(n)$ | $\text{OFT}\cap\text{LoRA}_r$ (Thm 12) | $K_{\text{out}}$ | $\operatorname{rank}\Delta W\le r$ | dim $r(r+1)/2$ | smooth | ✓ |
| Givens-OFT | $GL_m\times$ plane structure | R-products of Givens rotations | $K_{\text{out}}$ (approx. for quasi-orth.) | $\operatorname{rank}(R-I)\le2k$ | commuting planes | smooth | ✓ |
| (IA)$^3$ | $\mathrm{Mon}_m\times GL_n$ ($k,v$); $GL\times\mathrm{Mon}$ (FFN) | L-$T_m$ / R-$T_n$ | row/col directions, zeros | — | 1 | smooth | ✓ |
| SSF | $\mathrm{Mon}_m\times GL_n$ | L-$(T_m\ltimes\mathbb{R}^m)$ on $(W,b)$ | row directions | — | 1 | smooth | ✓ |
| LayerNorm tuning | $GL_m\times\mathrm{Mon}_n$ (next linear) | R-$T_n$ tied over $Q,K,V$, plus bias | column directions | — | 1 | smooth | ✓ |
| BitFit | $GL\times GL$ | tr-lin on biases | every $W$ | — | 1 | smooth | ✓ |
| Diff pruning | $\mathrm{Mon}\times\mathrm{Mon}$ | union of coordinate subspaces | — | $|\operatorname{supp}\Delta W|\le k$ | 1 | apex (of the union) | ✓ |
| FISH mask | $\mathrm{Mon}\times\mathrm{Mon}$ (Fisher, data) | tr-lin $\mathbb{R}^{\Omega}$ | — | $\operatorname{supp}\subseteq\Omega$ | 1 | smooth | ✓ |
| LISA | $GL\times GL$ per block | random walk on layer subspaces | — | active layers | 1 | smooth | ✓ |
| GaLore | $O\times O$ (gradient data) | dynamic tr-lin = frozen-$B$ LoRA (Prop 16) | — | $\operatorname{col}\Delta W_{\text{period}}\subseteq P$ | 1 | smooth | — (full $W$) |
| Houlsby / Pfeiffer | ext | serial nonlinear bottleneck | — | — | GL on bottleneck if linear | neutral ($W_{\text{up}}=0$) | ✗ |
| Parallel adapter | ext | parallel bottleneck ($\sigma=\mathrm{id}$ ⇒ LoRA) | — | — | $GL_r$ if linear | neutral | ✗ (✓ if linear) |
| AdapterFusion, MoLE / LoRAMoE | ext | input-dependent mixture $\sum g_i(x)\Delta_i$ | — | per-input rank $\le kr$ | per-expert $GL_r$ | neutral | ✗ (✓ iff gate constant) |
| LST | ext | side network, backward-free backbone | backbone | — | — | neutral | ✗ |
| LLaMA-Adapter | ext | ungated parallel adapter in activations | — | — | — | neutral $g=0$ | ✗ |
| Prompt tuning | ext | learned input state | — | — | — | neutral (no prompt) | ✗ |
| Prefix, P-tuning v2 | ext | gated parallel adapter, gate $\lambda(q)$ | — | — | — | neutral | ✗ |

**Dynamics coordinates (optimizer-level).**

| item | object in $\mathbf{Rep}_*$ | optimizer gauge $H$ | Noether charge at init | reading |
|---|---|---|---|---|
| LoRA + SGD | LoRA | $O(r)$ | $C=-A_0A_0^{\top}$ | unbalanced level set |
| LoRA + Adam(W) | LoRA | $B_r$ | none (charge broken) | finite gauge shadow |
| LoRA+ | LoRA | as base optimizer | $C_\eta=-A_0A_0^{\top}/\eta_A$ | SGD: init rescale $\lambda^{1/4}$ (Thm 9) |
| rsLoRA | $\cong$LoRA | as base | scaled by $s^2$ | $r$-dependent metric |
| Riemannian precond. / ScaledGD | LoRA | $GL_r$ | n/a (descends) | optimizer on $W_0+M_r$ |
| LoRA-RITE | LoRA | $GL_r$ (by design) | n/a | adaptive *and* invariant |
| weight decay | — | — | $C(t)=e^{-2\lambda t}C_0$ | gauge fixing to balance |
| PiSSA / MiLoRA init | Type II | — | $C_0=0$ | balanced forever (flow) |
| VeRA | VeRA | $\mathbb{R}^{\times}$ for SGD | $\|b\|^2-\|d\|^2=-rd_{\text{init}}^2$ | Conj B |
| MeZO | any | $O(\dim Q)$ in law | — | forward-only lens |
| Task arithmetic | in torsor $\Theta$ | $\mathbb{A}$ | — | coordinate-free merge |
| TIES / DARE | in torsor | $B_N$ (coordinate) | — | coordinate merge (Conj E) |
| LoRAHub / factor FedAvg | factors | needs common gauge | — | Prop 14 |
| S-LoRA | coproduct over common $\theta_0$ | — | — | shared base GEMM |
| Text-to-LoRA | section $\mathcal{T}\to Q$ | loss must be $GL_r$-invariant | — | design rule |

---

## 5. Equivalences and obstructions (summary)

**Equivalences (all proved above).**
1. Linear parallel adapter = LoRA (Thm 17a).
2. Prefix tuning = parallel adapter with gate $\lambda(q)$ (He et al.); LLaMA-Adapter = the same with the gate decoupled and zero-initialized (Thm 17b,c).
3. DoRA $\equiv_{\text{expressivity}}$ (IA)$^3_{\text{out}}\circ$LoRA (Thm 8).
4. HRA = OFT $\cap$ LoRA$_r$, a pullback of images, for injective $W_0$ (Thm 12).
5. LoRA+ (SGD) = LoRA with $A_0\mapsto\lambda^{1/4}A_0$ and $\bar\eta=\sqrt{\eta_A\eta_B}$, exactly (Thm 9).
6. rsLoRA $\cong$ LoRA in $\mathbf{Rep}_*$ (difference purely metric).
7. GaLore (one period) = frozen-$B$ LoRA with Adam (Prop 16).
8. LoRA-XS $\to$ PiSSA is a morphism; LoRA-FA $\to$ LoRA is a monomorphism.
9. (IA)$^3$, SSF and LN tuning lie in HiRA$_1$, up to bias translations (Prop 18).
10. With a frozen shared $A$, factor-merge = task arithmetic (Prop 14).

**Obstructions.**
1. LoRA and PiSSA (and every Type II init) have incomparable reachable sets at equal rank (Thm 11).
2. Balanced + zero-product + non-stationary is impossible (Thm 7); the HRA analogue holds as well.
3. Prefix, prompt and nonlinear serial adapters are not mergeable: the single-token affinity obstruction (Thm 17d).
4. Block-diagonal OFT never escapes its partial-Gram invariants under re-merging; BOFT does iff its block hypergraph is connected (Thm 13c).
5. Group-type methods gain nothing from ReLoRA-style restarts with a fixed frame (Thm 13b).
6. Per-rank pruning in plain LoRA is not gauge-invariant (Prop 15).
7. Mon-covariant methods and coordinate merges do not descend through rotations of the residual stream (Prop 3).
8. LoHa at the LoRA$_{2r}$ budget loses about $m+n$ dimensions of reachable set (Thm 5).

---

## 6. Rosetta stone

| PEFT concept | Categorical / geometric concept | where |
|---|---|---|
| pretrained model | pointed object $(\Theta,\theta_0)$ of $\mathbf{Para}$ | Def 2 |
| PEFT method | pointed reparametrization, an object of $\mathbf{Diff}_*/(\Theta,\theta_0)$ | Def 3 |
| frozen model / full FT | initial / terminal object | Obs 1 |
| merging adapter into weights | the canonical map to the terminal object ($=\rho$) | Obs 1 |
| "method $N$ can emulate $M$" | morphism in the slice (simulation); image inclusion (preorder) | Def 3 |
| expressivity | semialgebraic image $\rho(Q)$, its dimension and singularities | Def 1, 8 |
| zero-init adapter | base point at the apex (singular vertex) of a cone | Def 9, Thm 4 |
| PiSSA-type init with residual | change of object: translate the cone so $\theta_0$ is a smooth point | Thm 11 |
| quantized base (QLoRA) | change of base point $\theta_0\rightsquigarrow\hat\theta_0$ | Thm 11 |
| redundant parameters | gauge group $\mathrm{Aut}_{/\Theta}(Q,\rho)$; fibres = orbits | Def 8, Thm 4–5 |
| effective parameter count | $\dim Q-\dim K$ = generic rank of $d\rho$ | Thm 5 |
| "geometry" a method lives in | covariance group $\mathcal{G}_M$ (Klein) | Def 6 |
| what a method cannot change | absolute invariants (Gram, directions, zero pattern) | Thm 2 |
| rank budget / sparsity budget | relative (two-point) invariant | Def 7 |
| OFT preserves hyperspherical energy | right $O(n)$-orbit = fibre of $W\mapsto WW^{\top}$ (FFT of invariant theory) | Thm 2 |
| DoRA magnitude/direction | polar decomposition; torus saturation | Thm 8 |
| HRA "bridges" LoRA and OFT | pullback (meet) of images | Thm 12 |
| ReLoRA / restarts | subgroup generated by the reachable displacements | Thm 13 |
| butterfly factors | connecting a hypergraph so that $\mathfrak{so}(I_j)$ generate $\mathfrak{so}(n)$ | Thm 13 |
| balancedness $B^{\top}B=AA^{\top}$ | Noether charge of the symmetric part of $\mathfrak{gl}_r$ | Thm 6 |
| weight decay | exponential gauge fixing to the zero-charge slice | Cor 6.1 |
| LoRA+ learning-rate ratio | gauge transformation of initialization (SGD) | Thm 9 |
| Riemannian / transformation-invariant optimizers | optimizers equivariant under the full gauge, so they descend to the image | Thm 10 |
| Adam on factors | equivariance only under the finite group $B_r$ | Thm 10 |
| rank pruning / nested ranks | gauge reduction $GL_r\to B_r$ (AdaLoRA) or $\to T_r$ (DyLoRA); filtration of subobjects | Prop 15 |
| task vectors | elements of the translation group acting on the torsor $\Theta$ | Prop 14 |
| factor-wise merging | ill-defined on the quotient unless the gauge is shared | Prop 14 |
| GaLore vs LoRA | cotangent vs tangent restriction; equal for linear orthonormal charts | Prop 16 |
| adapters, prompts, prefixes | extension $\Theta\rightarrowtail\Theta\times P'$ with neutral point | Def 5 |
| non-mergeability | non-existence of a factorization through $f$ (lifting obstruction) | Thm 17 |
| multi-adapter serving | many objects over one common base point | Thm 17 |
| robustness to model reparametrization | descent along $G_{\text{arch}}$ (covariance) | Prop 3 |

---

## 7. What is analogy rather than theorem

- **"Gauge".** Here it means the symmetry group of a parametrization, nothing more. No connection, curvature or gauge field is defined. Globally $\rho$ is *not* a principal bundle: fibres jump at the apex (Thm 4b). The torsor statement holds only on the full-rank stratum.
- **Noether.** Thm 6 is proved for gradient flow with a *fixed* metric and a *linear* gauge action. Discrete GD drifts at $O(\eta^2)$, and Adam, gradient clipping and stochasticity (beyond expectation) break the charge. Practical claims about balancedness under AdamW (Conj B) are untested.
- **Erlangen classification.** It assigns coordinates (groups, invariants); it is not a theorem that "every method is a group orbit". Most popular methods are variety-type (cones, Hadamard and Kronecker varieties), and their structure group is not unique. The covariance group of methods depending on random frames or data holds only in distribution or with the data transformed.
- **Expressivity is not performance.** Reachable sets, dimensions and invariants constrain what training *can* reach. They say nothing direct about generalization or sample efficiency. Thm 9 and Thm 10 say where an effect can or cannot come from, for example that LoRA+ under SGD is only an init rescale. They do *not* derive LoRA+'s empirical gains under Adam.
- **Lens duality.** The tangent/cotangent picture of LoRA vs GaLore is rigorous only for linear charts (Prop 16). For curved charts it is a useful description, not a theorem.
- **Symmetry breaking by $\theta_0$** as the reason merging works for fine-tunes is a heuristic. The precise part is only that a generic $\theta_0$ has trivial stabilizer under neuron permutations.
- **Descent through residual rotations** (Prop 3, Conj A) needs RMSNorm/LayerNorm gains fused into adjacent weights, as QuaRot does. Without fusing, the norm layers themselves are only $\mathrm{Mon}$-covariant.
- **Category theory proper** does real work here at a few points: the slice and its terminal map (merging), pullbacks (HRA), change of base point (QLoRA/LoftQ), filtrations (DyLoRA), automorphism groups over $\Theta$ (gauge), and extension vs factorization (mergeability). The heavy lifting is done by group actions and real algebraic geometry. That is Klein's programme, and we say so plainly. "Mixtures as sections of a bundle of adapters" and "S-LoRA as a coproduct" are descriptions, not theorems.

---

## 8. Interactive visualizations (each tied to a precise statement)

1. **Hyperbola explorer (Thm 6, 9, Cor 6.1).** Scalar LoRA $w=ab$. *Manipulate:* $(a_0,b_0)$, $\eta_A,\eta_B$, weight decay, optimizer (GF/SGD/Adam). *See:* trajectories in the $(a,b)$-plane riding the hyperbolas $b^2/\eta_B-a^2/\eta_A=\text{const}$; the fibre $ab=w$ drawn as the gauge orbit; weight decay collapsing onto the balanced diagonals; Adam leaving the hyperbola. A "LoRA+ ≡ rescaled init" toggle overlays two runs whose $w(t)$ coincide exactly, and a side plot shows $\dot w/\ell'=\sqrt{\eta_B^2a_0^4+4\eta_A\eta_Bw^2}$.
2. **Apex vs smooth point (Thm 4, 11).** Symmetric $2\times2$ matrices $\begin{psmallmatrix}x+y&z\\z&x-y\end{psmallmatrix}$, where rank $\le1$ is the double cone $x^2=y^2+z^2$ in $\mathbb{R}^3$. *Manipulate:* drag a target $W$; switch LoRA (cone apex at $W_0$) or PiSSA (same cone translated so $W_0$ is on its surface). *See:* reachable or unreachable coloring, the tangent cone vs tangent plane at $W_0$, the first-step gradient projected onto $\operatorname{row}(A_0)$, and the regions reachable by one method but not the other.
3. **Gram lab (Thm 2, 8, Prop 18).** Neurons as arrows in $\mathbb{R}^2$/$\mathbb{R}^3$ with live heatmaps of $K_{\text{out}}=WW^{\top}$ and $K_{\text{in}}=W^{\top}W$. *Manipulate:* apply OFT (input-side rotation), output-side rotation, (IA)$^3$, DoRA, HiRA, LoRA. *See:* exactly which heatmap freezes (it glows), plus the zero-pattern overlay for torus methods. This fixes the left/right confusion once and for all.
4. **The Erlangen lattice (Def 6, Prop 3).** A Hasse diagram of covariance groups, $GL\times GL\supset GL\times CO\supset O\times O\supset\mathrm{Mon}\times\mathrm{Mon}\supset B\times B$, with every method placed at its geometry. *Manipulate:* a "rotate the residual stream" or "permute neurons" button. *See:* which methods turn red (non-descending) under each reparametrization. Clicking a method shows its seven coordinates.
5. **Gauge census (Thm 5).** *Manipulate:* $(m,n,r)$, choice of methods. *See:* stacked bars of parameters $=$ effective dimension $+$ gauge waste, and the LoHa$_r$ vs LoRA$_{2r}$ equal-budget comparison (rank reach vs dimension). A "verify" button computes the Jacobian rank in-browser for small sizes by finite differences.
6. **Optimizer gauge test (Thm 10).** $r=2$. *Manipulate:* a gauge $g\in GL_2$ via rotation, shear and scale sliders. *See:* the induced $\delta W$ for SGD, Adam and ScaledGD. ScaledGD's arrow stays still for all $g$; SGD's moves under shear and scale only; Adam's moves under everything except signed permutations.
7. **Butterfly connectivity (Thm 13c).** Coordinates as nodes and blocks as hyperedges. *Manipulate:* add block-OFT or BOFT factors and choose the block size. *See:* connected components merging, the dimension of the generated Lie algebra rising to $n(n-1)/2$, and the count of preserved partial Grams falling to 1.
8. **ReLoRA staircase (Thm 13a,b).** *Manipulate:* number of merge cycles $k$, frame recomputation on or off. *See:* the singular-value spectrum of the accumulated $\Delta W$ climbing to rank $kr$ for LoRA, flat for LoRA-XS with a fixed frame, climbing again when the frame is recomputed.
9. **Prefix gate (Thm 17).** *Manipulate:* query direction, prefix length, context length, LLaMA-Adapter gate $g$. *See:* $\lambda(q)$ and the attention mass split. A single-token plot shows output vs $t$ for $x=tx_1$: the straight line of every original-weight setting against the prefix curve, which visualizes the mergeability obstruction.
10. **Trilemma triangle (Thm 7).** Vertices "zero product", "balanced", "moving". Each init (LoRA, EVA, PiSSA, MiLoRA, OLoRA, LoRA-GA, HRA-paired) sits on the edge it satisfies. Clicking one shows its charge $C_0$ and residual.

---

## 9. Key references

*Categorical learning.* B. Fong, D. Spivak, R. Tuyéras, *Backprop as Functor* (arXiv:1711.10455). G. Cruttwell, B. Gavranović, N. Ghani, P. Wilson, F. Zanasi, *Categorical Foundations of Gradient-Based Learning* (arXiv:2103.01931). F. Klein, *Vergleichende Betrachtungen über neuere geometrische Forschungen* (Erlangen, 1872).
*Low-rank family.* Hu et al., LoRA (arXiv:2106.09685). Kalajdzievski, rsLoRA (arXiv:2312.03732). Hayou, Ghosh, Yu, LoRA+ (arXiv:2402.12354). Zhang et al., LoRA-FA (arXiv:2308.03303). Liu et al., DoRA (arXiv:2402.09353). Meng, Wang, Zhang, PiSSA (arXiv:2404.02948). Wang et al., MiLoRA (arXiv:2406.09044). Büyükakyüz, OLoRA (arXiv:2406.01775). Wang, Yu, Li, LoRA-GA (arXiv:2407.05000). Paischer et al., EVA (arXiv:2410.07170). Yang et al., CorDA (arXiv:2406.05223). Li et al., LoftQ (arXiv:2310.08659). Dettmers et al., QLoRA (arXiv:2305.14314). Zhang et al., AdaLoRA (arXiv:2303.10512). Valipour et al., DyLoRA (arXiv:2210.07558). Lialin et al., ReLoRA (arXiv:2307.05695). Kopiczko, Blankevoort, Asano, VeRA (arXiv:2310.11454). Koohpayegani et al., NOLA (arXiv:2310.02556). Li, Han, Ji, VB-LoRA (arXiv:2405.15179). Bałazy et al., LoRA-XS (arXiv:2405.17604). Gao et al., FourierFT (arXiv:2405.03003). Jiang et al., MoRA (arXiv:2405.12130). Huang et al., HiRA (ICLR 2025).
*Structured products.* Hyeon-Woo, Ye-Bin, Oh, FedPara, the origin of LoHa (arXiv:2108.06098). Yeh et al., LyCORIS/LoKr (arXiv:2309.14859). Mahabadi, Henderson, Ruder, Compacter (arXiv:2106.04647). Yang et al., LoRETTA tensor-train adapters (arXiv:2402.11417). Holtz, Rohwedder, Schneider, *On manifolds of tensors of fixed TT-rank* (Numer. Math., 2012).
*Orthogonal and multiplicative.* Qiu et al., OFT (arXiv:2306.07280). Liu et al., BOFT (arXiv:2311.06243). Yuan, Liu, Xu, HRA (arXiv:2405.17484). Ma et al., quasi-orthogonal fine-tuning via Givens rotations (arXiv:2404.04316). Liu et al., (IA)$^3$ (arXiv:2205.05638). Lian et al., SSF (arXiv:2210.08823). Salimans, Kingma, weight normalization (arXiv:1602.07868).
*Selective.* Ben Zaken, Ravfogel, Goldberg, BitFit (arXiv:2106.10199). Guo, Rush, Kim, diff pruning (arXiv:2012.07463). Sung, Nair, Raffel, FISH mask (arXiv:2111.09839). Zhao et al., LayerNorm tuning (arXiv:2312.11420). Pan et al., LISA (arXiv:2403.17919).
*Adapters and prompts.* Houlsby et al. (arXiv:1902.00751). Pfeiffer et al., AdapterFusion (arXiv:2005.00247). He et al., *Towards a Unified View of Parameter-Efficient Transfer Learning* (arXiv:2110.04366). Sung, Cho, Bansal, LST (arXiv:2206.06522). Zhang et al., LLaMA-Adapter (arXiv:2303.16199). Lester, Al-Rfou, Constant, prompt tuning (arXiv:2104.08691). Li, Liang, prefix tuning (arXiv:2101.00190). Liu et al., P-tuning v2 (arXiv:2110.07602). Wang et al., *Universality and Limitations of Prompt Tuning* (arXiv:2305.18787). Petrov, Torr, Bibi, *When Do Prompting and Prefix-Tuning Work?* (arXiv:2310.19698). Mahabadi et al., HyperFormer (arXiv:2106.04489). Charakorn et al., Text-to-LoRA (ICML 2025).
*Optimizers, gradients, zeroth order.* Zhao et al., GaLore (arXiv:2403.03507). Hao, Cao, Mou, Flora (arXiv:2402.03293). Malladi et al., MeZO (arXiv:2305.17333). Zhang, Pilanci, Riemannian preconditioned LoRA (arXiv:2402.02347). Yen et al., LoRA-RITE (arXiv:2410.20625). Tong, Ma, Chi, ScaledGD (arXiv:2005.08898). Mishra, Sepulchre, *Riemannian preconditioning* (arXiv:1405.6055).
*Conservation laws and symmetry.* Du, Hu, Lee, *layers are automatically balanced* (arXiv:1806.00900). Arora, Cohen, Hazan, *implicit acceleration by overparameterization* (arXiv:1802.06509). Kunin et al., *Neural Mechanics* (arXiv:2012.04728). Zhao et al., *Symmetries, flat minima, and the conserved quantities of gradient flow* (arXiv:2210.17216). Marcotte, Gribonval, Peyré, *Abide by the law and follow the flow* (arXiv:2307.00144). Ainsworth, Hayase, Srinivasa, Git Re-Basin (arXiv:2209.04836). Neyshabur, Sedghi, Zhang, *What is being transferred in transfer learning?* (arXiv:2008.11687).
*Merging and serving.* Ilharco et al., task arithmetic (arXiv:2212.04089). Yadav et al., TIES-Merging (arXiv:2306.01708). Yu et al., DARE (arXiv:2311.03099). Huang et al., LoraHub (arXiv:2307.13269). Wu, Huang, Wei, Mixture of LoRA Experts (arXiv:2404.13628). Dou et al., LoRAMoE (arXiv:2312.09979). Sheng et al., S-LoRA (arXiv:2311.03285). Sun et al., FFA-LoRA (arXiv:2403.12313).
*Reparametrized base models.* Ashkboos et al., QuaRot (arXiv:2404.00456). Ashkboos et al., SliceGPT (arXiv:2401.15024). Wang et al., SVD-LLM (arXiv:2403.07378).
*Library.* Hugging Face PEFT documentation (method implementations, including DoRA's per-output-channel magnitude and OLoRA's QR initialization).

---

*Numerical checks (reproducible, float64 autograd).* LoRA rank 26 generic and 16 at $B=0$, at $(8,7,2)$. DoRA 34. VeRA 9. NOLA 8. LoKr 17. TT 37. HRA $8,15,21,26$. LoHa $38,59,93,101$. OFT with wide $W_0$ 12 ($=\dim\mathfrak{so}(6)-\dim\mathfrak{so}(3)$). Noether drift $\sim10^{-4}$ (Euler $O(h)$). LoRA+ ≡ rescaled init: trajectory difference exactly 0. Equivariance defects as in Thm 10.
