# PEFT from the Relative Point of View

*Proposal for the "relative" lens: slices, base change, fibrations.*
*Status labels used throughout:* **[Def]** definition · **[Thm]** theorem with proof · **[Prop]** proposition with proof · **[Trivial]** true and formally easy, kept because it organises · **[Known]** due to the cited authors; we only re-derive or re-read it · **[Num]** checked numerically by `theory/proposals/relative_checks.py` (Jacobian ranks on random instances; exact-arithmetic Adam/SGD runs) · **[Pred]** falsifiable prediction · **[Analogy]** organising language with no theorem behind it.

---

## 0. Thesis

A pretrained checkpoint is a **base point**, and a PEFT method is a **relative object over it**: a pointed map $\rho:(Q,q_0)\to(\Theta,\theta_0)$ from a small manifold of trainable parameters into weight space, together with the way that map transforms when the base point, the coordinates on $\Theta$, or the architecture change. Following Grothendieck, we study the morphism and not the object. Three of its invariants do the work. (i) The **fibres** of $\rho$ are gauge orbits. Together with the optimizer's metric they determine the dynamics: balancedness, the learning-rate ratio of LoRA+, $\alpha$, rsLoRA. (ii) The **image germ** at $\theta_0$ is a semialgebraic set (a cone, a linear space, or a group orbit) with a tangent cone, a natural symmetry group and a closure under iteration, and it determines expressivity. (iii) **Base change**, meaning moving the base point by quantization, splitting, restarting or merging, organises the initialization, schedule and merging literatures. In this setting about fifty methods reduce to a handful of geometric types, and several "new" methods turn out to be old ones in transported coordinates. Folklore facts become one-line consequences, for instance that QLoRA needs no new theory, that ReLoRA reaches rank $Kr$, and that the default HRA initialization needs even $r$. A few sharper statements also follow: a moduli space of LoRA initializations, an exact equivalence LoRA+ $\equiv$ $\alpha$-rescaling under Adam, $\mathrm{HRA}=\mathrm{LoRA}\cap\mathrm{OFT}$, and a closure theorem saying which iterated methods can ever leave their starting subspace.

---

## 1. Setting and definitions

**Conventions.** A linear box is $y=Wx$ with $W\in\mathbb R^{m\times n}$ (rows = neurons). $M_{\le r}\subset\mathbb R^{m\times n}$ is the determinantal variety of rank $\le r$ matrices; $M_r$ (rank exactly $r$) is smooth of dimension $r(m+n-r)$, and $\mathrm{Sing}(M_{\le r})=M_{\le r-1}$. For a whole network $\Theta=\prod_{b}\Theta_b$ over boxes $b$.

**Ambient category.** We work in $\mathbf{Diff}$ (finite-dimensional manifolds, smooth maps) and pass to $\mathbf{Set}$ for statements about images. Every concrete $\rho$ in PEFT is semialgebraic (polynomial, rational, or involving norms) or real-analytic (Lie exponential). So images are semialgebraic or subanalytic sets with a well-defined dimension, and $\dim\rho(Q)=\max_q\operatorname{rank}d\rho_q$ (generic rank). This is why arbitrary smooth maps are not allowed. Images are usually **not** manifolds, and their singularities carry information (Thm 4).

**[Def 1.1] Pretrained model.** A triple $(\Theta,f,\theta_0)$ with $f:\Theta\times X\to Y$ a morphism of $\mathbf{Para}(\mathbf{Diff})$ (Fong–Spivak–Tuyéras; Cruttwell et al.) and $\theta_0\in\Theta$. Its realization is $\Phi_f:\Theta\to C^\infty(X,Y)$, $\theta\mapsto f(\theta,-)$.

**[Def 1.2] Methods at a base point.** $\mathsf{Meth}(\Theta,\theta_0):=\mathbf{Diff}_*/(\Theta,\theta_0)$, the pointed slice. An object is $M=(Q,q_0,\rho)$ with $\rho(q_0)=\theta_0$. A morphism $h:M\to N$ is a pointed smooth map with $\rho_N\circ h=\rho_M$; read it as "$N$ simulates $M$". *Weak* morphisms drop $h(q_0)=q_0'$. Equivalently, $\mathsf{Meth}(\Theta,\theta_0)$ is the pointed slice of the hom-category $\mathbf{Para}(X,Y)$ over $(\Theta,f)$, because $(Q,f\circ(\rho\times X))$ is a reparametrization 2-cell. The **budget** is $|M|=\dim Q$.

**[Def 1.3] Invariants of an object.**
- Image $\mathrm{Im}\,M=\rho(Q)\ni\theta_0$.
- Expressive dimension $d(M)=\dim\rho(Q)$, and gauge deficit $g(M)=\dim Q-d(M)$.
- First-order image $S_1(M)=d\rho_{q_0}(T_{q_0}Q)\subseteq T_{\theta_0}\Theta$.
- Gauge group: the group of diffeomorphisms of $Q$ over $\Theta$, whose generic orbits are the generic fibres of $\rho$.

**[Def 1.4] Families (methods as sections).** A *method family* on a set $U$ of checkpoints is $\rho:U\times Q\to\Theta$ with $q_0:U\to Q$ and $\rho(\theta,q_0(\theta))=\theta$. It comes in four types:
- **Additive** (translation-invariant): $\rho(\theta,q)=\theta+\delta(q)$ with $\delta:(Q,q_0)\to(\Theta,0)$ fixed. These are sections of the constant bundle $\theta\mapsto\mathsf{Meth}(\Theta,\theta)$, trivialised by translations.
- **Action**: $G$ acts on $\Theta$, $Q\subseteq G$ is a submanifold through $e$ (or a product of subgroups), and $\rho(\theta,g)=g\cdot\theta$.
- **Base-dependent**: $\delta$ or $q_0$ depends on $\theta$.
- **Data-dependent**: $\delta$ or $q_0$ depends on activations or gradients at $\theta$.

**[Def 1.5] Pointings / splittings.** For additive $\delta$ and any $q\in Q$, the *re-pointed method* is $M_q=(Q,\,q,\,\theta_0-\delta(q)+\delta(\cdot))$. Its residual base is $\theta_{\rm res}=\theta_0-\delta(q)$.

**[Def 1.6] Networks and extensions.** Let $\mathbf{PNet}(X,Y)$ have objects pointed networks $(\Theta,f,\theta_0)$. Its morphisms are *function-preserving* pointed reparametrizations $\phi:(\Theta',f',\theta_0')\to(\Theta,f,\theta_0)$ with $f\circ(\phi\times X)=f'$ (Net2Net-type embeddings).
- An **extension** is $\iota:(\Theta,f,\theta_0)\to(\Theta\times\Theta',\tilde f,(\theta_0,\theta_0'))$ with $\iota(\theta)=(\theta,\theta_0')$ a $\mathbf{PNet}$-morphism. Here $\theta_0'$ is the *neutral point* ("identity at init").
- A method on an extension is an object of $\mathsf{Meth}(\Theta\times\Theta',(\theta_0,\theta'_0))$.
- An **unpointed extension** has no neutral point (prompt tuning, Prop 12.4).

**[Def 1.7] Mergeability.** A method $M$ on an extension is *mergeable* if there is $\mu:Q\to\Theta$ with $\Phi_{\tilde f}\circ\rho=\Phi_f\circ\mu$, i.e. a lift of $\Phi_{\tilde f}\rho$ along $\Phi_f$. It is *locally mergeable at box $b$* if the modified box function stays in the box's function class $\mathcal F_b$ (linear maps, for a linear box).

**[Def 1.8] Base change.** A $\mathbf{PNet}$-morphism $\phi$ induces $\phi_!:\mathsf{Meth}(\Theta',\theta_0')\to\mathsf{Meth}(\Theta,\theta_0)$ by postcomposition. When $\phi$ is a submersion (always, in $\mathbf{Set}$), it also induces $\phi^*$ by fibre product, and $\phi_!\dashv\phi^*$. The Grothendieck construction $\int\mathsf{Meth}\to\mathbf{PNet}$ is an opfibration whose objects are (network, method) pairs. Maps that do *not* preserve the function, such as a quantizer $\kappa$, are not morphisms of $\mathbf{PNet}$. They move the base point, which is the source of "pointing defects" (Thm 13).

**[Def 1.9] Rebasing scheme.** Given a family, a $K$-stage scheme picks $\theta_{k+1}\in\mathrm{Im}\,M(\theta_k)$. The reachable set is $R_K(\theta_0)$ and $R_\infty=\bigcup_K R_K$.

**[Def 1.10] Locality.** For a set $S$ of boxes let $Q(S)$ be the trainable parameters that act on $S$. The method is **local** if $Q(S\sqcup S')=Q(S)\times Q(S')$, i.e. a sheaf for the discrete coverage. It is *separated* if $Q(S)\to\prod_{b\in S}Q(b)$ is injective, and *glues* if that map is surjective.

**[Def 1.11] Naturality group.** Let $\mathcal G$ be a groupoid of isomorphisms acting on a box's parameter space (for a linear box, $W\mapsto hWg^{-1}$). A family is **$\mathcal G$-natural** if $\mathrm{Im}\,M(hWg^{-1})=h\,\mathrm{Im}\,M(W)\,g^{-1}$. The largest such $\mathcal G$ is the method's *naturality group*, its Erlangen group.

**[Def 1.12] Families of adapters ($T$-points).** For a "variation base" $T$, a $T$-point of a method is a map $a:T\to Q$, giving the family $\rho\circ a:T\to\Theta$. The relevant bases are:
- $T=\ast$: one adapter.
- $T$ = task-description space: hypernetworks, Text-to-LoRA.
- $T=X$: input-dependent routing, as in mixtures of LoRA experts and AdapterFusion.
- $T=\Delta^{k-1}$: mixing weights, as in LoraHub and soups.
- $T=U$: a set of tenants, as in S-LoRA.

**[Def 1.13] Dynamical data.** A *trained method* is a method together with optimizer structure on $Q$: a metric for gradient descent, or a coordinate frame with per-block learning rates and initial scale for Adam. Slice isomorphisms that are not isometries leave expressivity unchanged and change dynamics.

---

## 2. Theorems and propositions

### Theorem 1 (universal properties) [Trivial]
In $\mathsf{Meth}(\Theta,\theta_0)$:
1. the frozen model $(\{\ast\},\ast,\theta_0)$ is initial;
2. full fine-tuning $(\Theta,\theta_0,\mathrm{id})$ is terminal;
3. the product $M\times N$ is the fibre product $Q_M\times_\Theta Q_N$ (in $\mathbf{Set}$ always, in $\mathbf{Diff}$ under transversality), with $\mathrm{Im}(M\times N)=\mathrm{Im}\,M\cap\mathrm{Im}\,N$;
4. $\mathrm{Im}$ is a functor to the poset of subsets containing $\theta_0$. In $\mathbf{Set}$, a morphism $M\to N$ exists iff $\mathrm{Im}\,M\subseteq\mathrm{Im}\,N$; in $\mathbf{Diff}$ only "$\Rightarrow$" holds.

*Proof.* These are standard slice facts; for (4) in $\mathbf{Set}$, choose preimages with $q_0\mapsto q_0'$. ∎

**What this explains.** Every method lies in the interval [frozen, full FT]. "What both $M$ and $N$ can express" is a categorical product (Thm 9 computes a non-trivial one). The gap between $\mathbf{Set}$ and $\mathbf{Diff}$ in (4) is a *smooth lifting obstruction*, and Prop 2 detects it.

### Proposition 2 (first-order functor) [Trivial]
$S_1$ is functorial: $M\to N$ implies $S_1(M)\subseteq S_1(N)$, so isomorphic pointed methods have equal $S_1$.

*Proof.* By the chain rule, $d\rho_N\circ dh=d\rho_M$. ∎

**What this explains.** $S_1$ is what a method can do in the lazy/NTK regime (Malladi et al. 2023). It is an isomorphism invariant that sees initialization, while the image does not (Thm 4).

### Theorem 3 (pointings of an additive method) [Thm]
Let $\delta:Q\to\Theta$ be additive with $C=\delta(Q)$.
1. Pointings at $\theta_0$ are parametrized by $Q$ itself, and $\mathrm{Im}\,M_q=\theta_0+(C-\delta(q))$.
2. $\bigcup_q\mathrm{Im}\,M_q=\theta_0+(C-C)$.
3. For LoRA$_r$, $C-C=M_{\le 2r}$.

*Proof.* (1) and (2) follow from the definitions. For (3), $\operatorname{rank}(X-Y)\le 2r$; conversely, a rank-$\le2r$ matrix splits as $Z_1+Z_2$ with $\operatorname{rank}Z_i\le r$, and $Z_1+Z_2=Z_1-(-Z_2)$. ∎

**What this explains.** An initialization of an additive method chooses which *translate* of the shape $C$ passes through $\theta_0$. No initialization scheme for LoRA$_r$, however clever, can make any update of rank $>2r$ reachable. Splitting inits (PiSSA-type) buy the "spectral replacement" updates $Y-\delta(q_0)$, which can have rank $2r$, and pay by losing generic rank-$r$ updates (Thm 4.4).

### Theorem 4 (moduli of LoRA initializations; apex vs. smooth point) [Thm]
Let $\delta(B,A)=BA$ with $B\in\mathbb R^{m\times r}$, $A\in\mathbb R^{r\times n}$, $1\le r<\min(m,n)$. Consider pointed LoRA objects $M_{q_0}$ in three strata:
- $(\mathrm T_A)$: $B_0=0$ and $\operatorname{rank}A_0=r$;
- $(\mathrm T_B)$: $A_0=0$ and $\operatorname{rank}B_0=r$;
- $(\mathrm S)$: $\operatorname{rank}(B_0A_0)=r$.

1. **Classification.** Within a stratum, $M_{q_0}\cong M_{q_0'}$ iff, respectively, $\mathrm{row}(A_0)=\mathrm{row}(A_0')$, $\mathrm{col}(B_0)=\mathrm{col}(B_0')$, or $B_0A_0=B_0'A_0'$. Objects in different strata are never isomorphic. Hence the isomorphism classes form
$$\mathrm{Gr}(r,\mathbb R^n)\ \sqcup\ \mathrm{Gr}(r,\mathbb R^m)\ \sqcup\ M_r(m\times n).$$
2. **First order.** $\dim S_1$ equals $mr$, $nr$ and $r(m+n-r)$ respectively, while $d=r(m+n-r)$ always. The first-order deficit of the zero-init stratum $\mathrm T_A$ is $r(n-r)$. **[Num]**
3. **Geometry.** In $\mathrm T_A$ and $\mathrm T_B$, $\mathrm{Im}=\theta_0+M_{\le r}$ and $\theta_0$ is its **apex**, the most singular point. In $\mathrm S$, $\mathrm{Im}=\theta_0-P+M_{\le r}$ with $P=B_0A_0$, and $\theta_0$ is a **smooth point**.
4. **Incomparability.** For $P\ne0$, the images of $\mathrm S$-type and $\mathrm T$-type objects are incomparable in the expressivity preorder.

*Proof.*
- **"If" in (1).** Two full-row-rank $A_0,A_0'$ with the same row space satisfy $A_0'=gA_0$ for a unique $g\in GL_r$. Then $h(B,A)=(Bg^{-1},gA)$ is a pointed isomorphism over $\Theta$. The same argument works for $\mathrm T_B$. For $\mathrm S$, the full-rank factorizations of a rank-$r$ matrix $P$ form one $GL_r$-orbit, and the base $\theta_0-P$ agrees.
- **"Only if" in (1).** In $\mathrm T_A$, $d\rho_{(0,A_0)}(\dot B,\dot A)=\dot BA_0$, so $S_1=\{XA_0\}=\{Z:\mathrm{row}Z\subseteq\mathrm{row}A_0\}$, which determines $\mathrm{row}(A_0)$; apply Prop 2. $\mathrm T_B$ is symmetric. The sets $\{XA_0\}$ and $\{B_0Y\}$ differ for $r<\min(m,n)$, which separates $\mathrm T_A$ from $\mathrm T_B$. In $\mathrm S$, the image determines its apex $\theta_0-P$ (iterate $\mathrm{Sing}$ down to the unique point of $M_{\le 0}$), hence determines $P$. $\mathrm S$ differs from $\mathrm T$ because $\theta_0$ is smooth in one image and the apex in the other.
- **(2).** $S_1$ at a full-rank pair is $\{XA_0+B_0Y\}=T_P M_r$.
- **(4).** Pick a rank-one $X$ whose column and row spaces avoid those of $P$. Then $\theta_0+X\in$ LoRA's image, but $X+P\notin M_{\le r}$. The converse inclusion fails the same way. ∎

**What this explains.** Every LoRA initialization scheme in the literature is a *point of this moduli space*:

| Scheme | Stratum | Point |
|---|---|---|
| vanilla LoRA | $\mathrm T_A$ | Haar-random point of $\mathrm{Gr}(r,n)$ |
| LoRA-FA | $\mathrm T_A$ | same point, frozen; it keeps only $S_1$ |
| EVA | $\mathrm T_A$ | top-$r$ principal subspace of input activations |
| "Init[B]" (Hayou et al. 2024) | $\mathrm T_B$ | random point of $\mathrm{Gr}(r,m)$ |
| PiSSA | $\mathrm S$ | $P=[W_0]_r$, top SVD part |
| MiLoRA | $\mathrm S$ | $P=$ bottom-$r$ SVD part |
| OLoRA | $\mathrm S$ | $P=Q_rR_r$, QR truncation |
| CorDA | $\mathrm S$ | $P$ from the SVD of $W_0C$ ($C$ = activation covariance), mapped back by $C^{-1}$; KPM and IPM pick the minor or major part |
| LoRA-GA | $\mathrm S$ | $P=sB_0A_0$ from the gradient SVD, chosen so that $S_1$ best approximates the full-FT step |
| LoftQ | $\mathrm S$ | over a quantized residual (Thm 13) |

Zero-init LoRA starts at a *singular point*. At first order the $A$-direction is dead, so LoRA and LoRA-FA coincide in the lazy regime and differ only at second order. This agrees with the empirical asymmetry findings of Zhu et al. 2024 and with Flora. Splitting inits start at a smooth point with full first-order reach, which is one mechanism (not a proof of speed) behind the faster early progress reported by PiSSA and LoRA-GA.

### Theorem 5 (gauge, moment map, Noether, Kempf–Ness) [Known + Thm]
$GL_r$ acts on $Q$ by $g\cdot(B,A)=(Bg^{-1},gA)$, and $\delta$ is invariant under this action. Over $M_r$ the fibres are free orbits, so $Q_{\rm full}\to M_r$ is a principal $GL_r$-bundle. Let $\mu(B,A)=B^\top B-AA^\top\in\mathrm{Sym}_r$.
1. **(Kempf–Ness.)** For $\xi\in\mathfrak{gl}_r$, $\tfrac{d}{dt}\big|_0\|e^{t\xi}\cdot q\|^2=-2\operatorname{tr}(\xi\mu(q))$. Antisymmetric $\xi$ (the compact part $\mathfrak o(r)$) gives $0$. So the norm-minimizing representatives of a full-rank orbit are exactly the *balanced* ones, $\mu=0$, and they are unique up to $O(r)$. The minimum is $\|B\|^2+\|A\|^2=2\|BA\|_*$ (Srebro et al. 2005).
2. **(Noether, [Known]: Du–Hu–Lee 2018, Arora et al. 2018, Zhao et al. 2023.)** Under gradient flow with block rates, $\dot B=-\eta_BGA^\top$ and $\dot A=-\eta_AB^\top G$ where $G=\nabla L(W)$. Then $B^\top B/\eta_B-AA^\top/\eta_A$ is conserved. The charges are indexed by $\mathrm{Sym}_r=\mathfrak{gl}_r/\mathfrak o(r)$, the non-compact directions.
3. **(Corollary, Loewner monotonicity.)** From zero init ($B_0=0$),
$$B^\top B=\tfrac{\eta_B}{\eta_A}\,(AA^\top-A_0A_0^\top)\succeq0\quad\text{for all }t,$$
so $AA^\top\succeq A_0A_0^\top$: the $A$-factor never shrinks below its initialization. **[Num]**
4. **(Weight decay.)** With decoupled decay $\lambda$ on both factors, $\dot\mu=-2\lambda\mu$, so the imbalance decays like $e^{-2\lambda t}$ and the flow is attracted to the Kempf–Ness (nuclear-norm) slice.

*Proof.*
- (1) $\langle q,(-B\xi,\xi A)\rangle=\operatorname{tr}(\xi(AA^\top-B^\top B))$, and the minimum over a closed orbit is attained.
- (2) $\tfrac{d}{dt}B^\top B=-\eta_B(AG^\top B+B^\top GA^\top)=\tfrac{\eta_B}{\eta_A}\tfrac{d}{dt}AA^\top$.
- (3) is (2) with $B_0=0$.
- (4) Each derivative picks up $-2\lambda(\cdot)$. ∎

**What this explains.** "Balanced" is not a heuristic: it is the Kempf–Ness slice of the gauge action. PiSSA's $S^{1/2}$-split sits exactly on it ($\mu_0=0$). Vanilla LoRA starts maximally unbalanced ($\mu_0=-A_0A_0^\top$) and stays so up to weight decay. Weight decay on factors is nuclear-norm regularization of $\Delta W$ on that slice. LoRA+ ($\eta_B\gg\eta_A$) pushes the flow toward the LoRA-FA regime by an amount the conserved charge quantifies: $\|A\|^2-\|A_0\|^2=\|B\|^2\eta_A/\eta_B$.

### Theorem 6 (hyperparameters live on an orbit space) [Thm; LoRA case Known: Schulman et al. 2025]
**Setting.** Let $\rho(q)=\theta_b+s\,\delta(q_1,\dots,q_k)$ with $\delta$ multihomogeneous of multidegree $d$, i.e. $\delta(c_1q_1,\dots)=\prod c_i^{d_i}\delta(q)$. Train each block with Adam at rate $\eta_i$, with $\varepsilon=0$ (convention $0/0=0$), decoupled decay $\lambda_i$, and initialization $q_i(0)=\sigma_iu_i$.

**Statement.** The trajectory $t\mapsto\rho(q(t))$ is invariant under the torus action
$$c\cdot(s,\eta_i,\sigma_i,\lambda_i)=(s\textstyle\prod c_i^{-d_i},\;c_i\eta_i,\;c_i\sigma_i,\;\lambda_i/c_i).$$
Under gradient descent, the same holds with $\eta_i\mapsto c_i^2\eta_i$.

*Proof.* Rescaling $q_i'=c_iq_i$ with the compensating $s$ is a slice isomorphism, so the model is unchanged. Gradients with respect to $q_i'$ are $c_i^{-1}$ times those with respect to $q_i$, and Adam with $\varepsilon=0$ is invariant to positive rescaling of a block's gradient history. So $\Delta q_i'=c_i\Delta q_i$ requires rate $c_i\eta_i$. For SGD, the gradient factor $c_i^{-1}$ combines with the needed $c_i$ to give $c_i^2$. ∎

**Corollaries.**
- **(a) LoRA.** $k=2$, $d=(1,1)$, $\sigma_B=0$. The effective hyperparameters are $(s\eta_A\eta_B,\ \sigma_A/\eta_A)$ (plus $\eta_i\lambda_i$). This is the pair Schulman et al. (2025) state as $\alpha\cdot\mathrm{init}_A\cdot LR_B$ and $\mathrm{init}_A/LR_A$.
- **(b) LoRA+ $\equiv$ $\alpha$-rescaling.** Without weight decay, LoRA+ with ratio $\lambda$ at scale $s$ produces *exactly* the trajectory of plain LoRA at scale $\lambda s$ (same $\eta_A$, same $A_0$) under Adam, and at scale $\sqrt\lambda\,s$ under SGD. **[Num]**: the maximum trajectory difference is $0.0$ at $\varepsilon\to0$ and $5\times10^{-9}$ at $\varepsilon=10^{-8}$.
- **(c) rsLoRA.** $\alpha/r\mapsto\alpha/\sqrt r$ is a slice isomorphism $h(B,A)=(\sqrt r B,A)$, so rsLoRA and LoRA are *isomorphic objects*. They differ only in metric data: rsLoRA multiplies the effective $\eta_B$ by $\sqrt r$ under Adam and by $r$ under SGD.

**Equivariance of LoRA optimizers [Prop].** The gauge group acts on optimizers, and each optimizer is equivariant under a different subgroup:
- SGD: $O(r)$.
- Adam: the positive diagonal torus of $GL_r$, together with permutations.
- Riemannian-preconditioned GD (Zhang & Pilanci 2024), $\dot B=-GA^\top(AA^\top)^{-1}$, $\dot A=-(B^\top B)^{-1}B^\top G$: all of $GL_r$. It descends to the well-defined flow $\dot W=-(\Pi_{\mathrm{col}B}G+G\Pi_{\mathrm{row}A})$ on $M_r$. The check is a direct substitution.
- LoRA-RITE (Yen et al. 2024): extends full $GL_r$-equivariance to adaptive moments.

**What this explains.** LoRA's four knobs ($\alpha$, $\eta_A$, $\eta_B$, init scale) are coordinates on a 4-dimensional space on which a 2-torus acts. Only the orbit matters, which collapses rsLoRA, LoRA+ and $\alpha$-tuning into one family.

**[Pred]** Any measured gain of LoRA+ over LoRA with $\alpha$ re-tuned by the factor $\lambda$ (same $\eta_A$) must come from $\varepsilon$, weight decay, global-norm gradient clipping, or per-group schedules, since these are exactly the ingredients that break the torus action.

### Theorem 7 (monoidal structure; rank as a functor) [Trivial/standard]
Additive methods carry $M\oplus N=(Q_M\times Q_N,(q_0,q_0'),\delta_M+\delta_N)$, the convolution product of a slice over an abelian group object, with unit the frozen model and $\mathrm{Im}(M\oplus N)=\mathrm{Im}M+\mathrm{Im}N$ (Minkowski sum). Concatenation $[B_1\,B_2],\binom{A_1}{A_2}$ gives an isomorphism $\mathrm{LoRA}_r\oplus\mathrm{LoRA}_s\cong\mathrm{LoRA}_{r+s}$. So $r\mapsto\mathrm{LoRA}_r$ is a strong monoidal functor $(\mathbb N,+)\to(\mathsf{AddMeth},\oplus)$. In $\mathbf{FinVect}$, $\mathrm{Im}\,\mathrm{LoRA}_r$ is the set of maps factoring through an object of dimension $\le r$, i.e. the coend $\int^{R\in\mathbf{FinVect}_{\le r}}\mathrm{Hom}(R,W)\times\mathrm{Hom}(V,R)$. Its colimit over $r$ is $\mathrm{Hom}(V,W)$ by co-Yoneda, so full FT is the colimit of the rank filtration.

**What this explains.**
- UniPELT, MAM, the "design spaces" programme, LoRA soups and low-rank + sparse methods (RoSA) are $\oplus$-products, and their expressivity is a Minkowski sum.
- DyLoRA and AdaLoRA work with the filtration $\mathrm{LoRA}_1\rightarrowtail\mathrm{LoRA}_2\rightarrowtail\cdots$.
- Rank is the functor; it is not a measure of image size (Thm 11).

### Theorem 8 (rebasing closure) [Thm]
For a family with fixed shape:
- **Additive** with $C=\mathrm{Im}\,\delta$: $R_K=\theta_0+C^{+K}$ (the $K$-fold Minkowski sum), and $R_\infty=\theta_0+\langle C\rangle$. When $C=\mathbb RC$ (a cone), this is $\theta_0+\mathrm{span}\,C$.
- **Action** with $S\subseteq G$: $R_K=S^K\theta_0$ and $R_\infty=\langle S\rangle\theta_0$. If $S$ is a union of one-parameter subgroups $\exp(\mathbb R X)$, $X\in\Sigma$, then $\langle S\rangle$ is the connected subgroup with Lie algebra $\mathrm{Lie}(\Sigma)$.

*Proof.* Induct on stages; the last claim is the standard theorem on subgroups generated by one-parameter subgroups. ∎

**Corollaries.**
1. **ReLoRA.** $R_K=\theta_0+M_{\le Kr}$. Full FT is reached exactly at $K^*=\lceil\min(m,n)/r\rceil$; fewer stages cannot reach it, by subadditivity of rank. Iteration in time realises the monoidal power $\mathrm{LoRA}_r^{\oplus K}$ in space.
2. **Idempotent (no gain from rebasing).** LoRA-FA with a fixed $A$, LoRA-XS, MoRA, FourierFT, BitFit, FISH mask, LayerNorm tuning, (IA)$^3$, SSF, and OFT with a *fixed* block structure (a subgroup). Their shapes are subspaces or subgroups.
3. **VeRA** (fixed random $A,B$ with nonzero entries). $\mathrm{span}\{\Lambda_bB\Lambda_dA\}=\{XA\}$, so iterated VeRA converges to LoRA-FA$(A)$ and stops there.
4. **Rotation-plane criterion.** For Givens-type methods with plane set $E$ (GOFT, BOFT, block OFT), $\mathrm{Lie}\{E_{ij}-E_{ji}:(i,j)\in E\}=\bigoplus_{\text{components}}\mathfrak{so}(V_c)$. So $R_\infty=SO(n)\theta_0$ **iff the graph $E$ is connected**:
   - block-diagonal OFT has disconnected cliques and is stuck in $\prod SO(b)$;
   - BOFT's butterfly gives a connected graph, which is the Lie-theoretic content of "butterfly information transmission".
5. **GaLore** (with a fixed projector $P$ for $T$ steps) is identical to one-sided LoRA $\rho(X)=W_s+PX$ followed by merge-and-rebase (Torroba-Hennigen et al. 2025). The same pattern covers:
   - Flora: resampled frozen factor;
   - LISA: moving coordinate blocks;
   - MeZO: a moving random line $\theta+tz$ with a finite-difference derivative.
   
   These are all rebasing schemes of *linear* methods whose shape moves. That is exactly why they reach full rank, while a fixed linear shape cannot (corollary 2).

**[Pred]** "Re-OFT" with fixed blocks provably gains nothing, while shuffling the block structure between stages generates $SO(n)$.

### Theorem 9 (orthogonal methods: invariants, and HRA = LoRA ∩ OFT) [Thm]
Let $O(n)$ act on the right, $W\mapsto WR$.
1. $W_0\,O(n)=\{W:WW^\top=W_0W_0^\top\}$. This is the first fundamental theorem for $O(n)$: tuples with equal Gram matrices are orthogonally equivalent. So every orthogonal method (OFT, BOFT, HRA, GOFT, COFT) has image inside the level set of the neuron Gram matrix, and preserves all neuron norms and angles (the "hyperspherical energy"). Full OFT has $d=mn-m(m+1)/2$ when $\operatorname{rank}W_0=m\le n$. **[Num]**
2. **Transversality.** If $\operatorname{rank}W_0=m$, then $T(\text{OFT})\cap T(\text{row scaling})=0$. From $W_0\Omega=DW_0$ we get $W_0\Omega W_0^\top=DG$ with $G\succ0$; the left side is skew, which forces $d_iG_{ii}=0$. **[Num]**
3. **HRA.** If $W_0$ is injective ($\operatorname{rank}W_0=n$), then $\mathrm{Im}\,\mathrm{HRA}_r\cup\mathrm{Im}\,\mathrm{HRA}_{r-1}=\mathrm{Im}\,\mathrm{LoRA}_r\cap W_0O(n)$. A *pointed* HRA ($R(q_0)=I$) needs $r$ even, and then $\mathrm{Im}\,\mathrm{HRA}_r=\mathrm{Im}\,\mathrm{LoRA}_r\cap W_0SO(n)$, the categorical product of Thm 1.3 at the level of images. Also $d(\mathrm{HRA}_r)=rn-r(r+1)/2$. **[Num]**

*Proof of 3.* $W_0R-W_0=W_0(R-I)$ has rank equal to $\operatorname{rank}(R-I)$ because $W_0$ is injective. By the sharp Cartan–Dieudonné theorem, an orthogonal $R$ with $\operatorname{rank}(R-I)=k$ is a product of $k$ reflections and of no fewer, and $\det R=(-1)^k$. Products of exactly $r$ reflections, padding with repeated pairs, are therefore those with $k\le r$ and $k\equiv r\pmod 2$. If $R(q_0)=I$, then $(-1)^r=1$. ∎

**What this explains.** Multiplicative methods move *inside a fibre of the invariant map* $\Theta\to\Theta/\!\!/G$. Additive methods move along a cone transverse to such fibres. HRA's "bridge between LoRA and OFT" is literally their meet. The HF PEFT note that $r$ should be even "otherwise the default initialization method will not work" is the determinant identity.

### Theorem 10 (the Hadamard-torus family) [Thm, easy]
Let $D_{W_0}(X)=W_0\odot X$.
1. HiRA's image is $W_0\odot(\mathbf 1\mathbf 1^\top+M_{\le r})=(D_{W_0})_*\mathrm{LoRA}_r$. If $W_0$ has no zero entries, $D_{W_0}\in GL(\Theta)$, so HiRA is LoRA transported along a diagonal automorphism of the base. It has the same $d=r(m+n-r)$ and the same gauge **[Num]**, yet $\operatorname{rank}\Delta W$ can reach $\operatorname{rank}(W_0)\cdot r$.
2. (IA)$^3$ is $\mathrm{diag}(l)W_0=W_0\odot(l\mathbf 1^\top)$, i.e. **HiRA$_1$ with $A$ frozen to $\mathbf 1^\top$**, with $l=\mathbf 1\Leftrightarrow B=0$, the apex again.
3. DoRA's magnitude vector is the same construction on the other side. DoRA $=T_+\cdot\mathrm{LoRA}$ is the saturation of LoRA's image by the positive torus acting on rows. The saturation $\mathsf T(M)=(T_+\times Q,\,(g,q)\mapsto g\rho(q))$ is an idempotent-on-images monad. Generically $d(\mathrm{DoRA})=r(m+n-r)+m$ **[Num]**.

**What this explains.** (IA)$^3$, SSF's scale, DoRA's magnitude and HiRA are one family, $W_0\odot(\mathbf 1+Z)$, indexed by the structured cone $Z$ ranges over. This gives a concrete design rule: anything between "rank-1 with one frozen side" and "rank $r$" is available.

### Theorem 11 (transport of structure: new methods as old methods in new coordinates) [Thm, easy]
$GL(\Theta)$ acts on additive methods by $T_*(Q,\delta)=(Q,T\delta)$. This action preserves $d$, $g$, polynomial degree and rebasing behaviour. It preserves *rank* only for $T(X)=hXg^{-1}$.
1. **LoKr.** With two free factors, LoKr is rank-one LoRA in the Van Loan–Pitsianis frame $\mathcal R(C\otimes D)=\mathrm{vec}C\,\mathrm{vec}D^\top$. Because $\mathcal R$ is a coordinate permutation of $\Theta$, and $\mathrm{vec}$ a permutation of $Q$, the equivalence holds **for dynamics too**, under any permutation-equivariant optimizer (SGD, Adam). Sums of $k$ Kronecker terms (Compacter's PHM weights, KronA) are LoRA$_k$ in this frame. **[Num]**
2. **FourierFT, LoRA-XS, SVFT.**
   - FourierFT is a fixed-support coordinate method transported by $Z\mapsto\mathrm{Re}(F_m^{-1}ZF_n^{-1})$.
   - LoRA-XS is the top-left $r\times r$ block mask transported by $Z\mapsto UZV^\top$ (the SVD frame of $W_0$), hence linear with $d=r^2$.
   - SVFT is a sparse mask in that same frame.
3. **LoHa** is a diagonal pushforward of a Kronecker method: $X\odot Y=S_m^\top(X\otimes Y)S_n$, where $S$ is the copy spider $e_i\mapsto e_i\otimes e_i$. Hence $\operatorname{rank}\le r_1r_2$. Its dimension is $d=r_1(m+n-r_1)+r_2(m+n-r_2)-(m+n-1)$; the extra gauge is the Hadamard torus $(D_1XD_2,D_1^{-1}YD_2^{-1})$. The upper bound is proved; equality is **[Num]** on five shapes. Two consequences:
   - LoHa$_{1,1}$ equals LoRA$_1$ exactly, since $(b_1a_1^\top)\odot(b_2a_2^\top)=(b_1\odot b_2)(a_1\odot a_2)^\top$.
   - At equal parameter count, LoHa$_{r,r}$ trades $(m+n-1)-2r^2$ image dimensions relative to LoRA$_{2r}$ for rank up to $r^2$.
4. **MoRA** with reshape compression is $I\otimes M$: a linear method, high rank, $d=\hat r^2$.

**What this explains.** "High rank" claims (MoRA, HiRA, LoKr, LoHa) are coordinate-relative. The transport-invariant content is $(d,g,\text{shape},\text{naturality group})$, and on those coordinates several methods coincide.

### Theorem 12 (mergeability and its obstructions) [Thm]
1. Reparametrizations are mergeable, with $\mu=\rho$.
2. **Linear insertions.** A linear parallel adapter $W_0x+UDx$ is isomorphic to LoRA, as pointed objects. A linear serial adapter $(I+UD)W_0$ is mergeable, with image $W_0+\{Y:\operatorname{rank}Y\le r,\ \mathrm{row}Y\subseteq\mathrm{row}W_0\}$; this equals LoRA's image iff $\operatorname{rank}W_0=n$.
3. **Nonlinear adapters.** For $\sigma$ non-affine and $U\sigma(D\cdot)$ not affine, $x\mapsto W_0x+U\sigma(Dx)$ is not linear, so the adapter is not locally mergeable. This covers Houlsby, Pfeiffer, parallel adapters and Compacter.
4. **Prefix theorem.** With prefix $(P_K,P_V)$, a head's output at query $i$ is $(1-\lambda_i)\sum_ja_{ij}v_j+\lambda_i\,\mathrm{softmax}(q_iP_K^\top)P_V$, where $a_{ij}$ are the *original* weights (He et al. 2022). Consequences:
   - (a) The ratios $a_{ij}/a_{ij'}$ among content tokens are unchanged (Petrov et al. 2024).
   - (b) On a length-1 context the original head, and every reparametrized head, is linear in $x$, while the prefixed head is not, generically in $(P_K,P_V)$. So prefix, prompt and P-tuning v2 are not locally mergeable.
   - (c) Prompt tuning is unpointed: softmax weights are positive, so a soft token always perturbs the output, generically. LLaMA-Adapter restores pointedness with zero-init gating.
5. **$X$-points.** A routed adapter $x\mapsto(W_0+\Delta W(x))x$ is locally mergeable iff $x\mapsto\Delta W(x)x$ is linear. Input-independent mixtures $\sum g_iB_iA_i$ equal LoRA$_{\sum r_i}$; routing (MoLE, LoRAMoE, AdapterFusion) generically is not mergeable.

**What this explains.** Inference-time overhead is a lifting problem along $\Phi_f$, decided box by box. S-LoRA/Punica do the opposite of merging: they compute *relatively* over a shared base ($W_0x+B_iA_ix$ per tenant). That is possible precisely because additive families are translation-invariant sections.

### Theorem 13 (base change: quantization and splitting) [Prop]
1. Additive families commute with translation: $M(\theta+v)=\tau_v\circ M(\theta)$. So for any quantizer $\kappa$, QLoRA $=M(\kappa\theta_0)$ is LoRA pointed at the wrong point, with defect $e=\kappa\theta_0-\theta_0$, and needs no new method.
2. Base-dependent families (PiSSA, DoRA, OFT, HiRA, LoRA-XS) must be recomputed after base change, and the two orders of operations differ:
   - split-then-quantize-residual (QPiSSA, LoftQ) has defect $\kappa\theta_{\rm res}-\theta_{\rm res}$;
   - quantize-then-split has defect $e$.
3. LoftQ alternately solves $\min_{\theta_r,q}\|\theta_0-\kappa\theta_r-\delta(q)\|$. This is an *approximate section* restoring pointedness; $\kappa$ is not a $\mathbf{PNet}$-morphism and LoftQ repairs it.
4. "Fine-tune then quantize" ($\kappa_!$ of the merged model) and "quantize then fine-tune" ($M(\kappa\theta_0)$ with a high-precision adapter) do not commute.

*Proof.* Immediate from the definitions. ∎

### Theorem 14 (factor-wise operations are not gauge-invariant) [Thm]
LoraHub composes $\hat m=(\sum w_iA_i)(\sum w_iB_i)$ factor-wise (Huang et al. 2023, Sec. 3). Expanding $(\sum w_iB_ig_i^{-1})(\sum w_jg_jA_j)$ shows that the cross terms $w_iw_jB_ig_i^{-1}g_jA_j$ depend on the representatives. So factor-wise combination is **not a function of the adapters** $\{B_iA_i\}$. It is well-defined only on the diagonal gauge, i.e. on representatives expressed in a common gauge. Sums of products, $\sum w_iB_iA_i$ (task arithmetic on adapters), are well-defined and land in LoRA$_{kr}$.

**What this explains.** Merging LoRAs needs alignment first. KnOTS gauge-fixes by a joint SVD.

**[Pred]** LoraHub behaves like a linear combination of updates when the modules share the $A$-seed and $A$ moves little, which is the regime Thm 5.3 predicts for zero init. It should degrade when the modules come from independent $A$-seeds.

### Theorem 15 (naturality and functional well-definedness) [Thm]
Let $G$ be the network's parameter symmetries, $\Phi_f(g\theta)=\Phi_f(\theta)$. If a family is $G$-equivariant, then its *functional* image depends only on the function $\Phi_f(\theta_0)$ and not on the checkpoint representative.
- LoRA on attention is equivariant under the query–key gauge ($W_Q\mapsto gW_Q$, $W_K\mapsto g^{-\top}W_K$, $g\in GL(d_h)$) and the value–output gauge, because $M_{\le r}$ is $GL\times GL$-stable.
- (IA)$^3$ on keys is equivariant only under the monomial subgroup, even with RoPE (whose commutant contains rotations inside each 2-plane).

*Proof.* $g\,\mathrm{Im}M(\theta)=\mathrm{Im}M(g\theta)$, and apply $\Phi_f$. ∎

**What this explains.** Coordinate methods are legitimate where the architecture has a privileged basis: FFN hidden units after an elementwise nonlinearity, and LayerNorm gains. They are representative-dependent on gauge-symmetric spaces. For LoRA, the image is $GL$-natural, while its init distribution and SGD metric are only $O$-natural.

### Proposition 16 (locality as a sheaf condition) [Def + Trivial]
- Local methods (LoRA, OFT, adapters) are sheaves.
- VB-LoRA (a shared vector bank), Compacter (shared $A_i$) and FacT (a cross-layer tensor) are separated but fail to glue.
- AdaLoRA and EVA, through a global rank budget, and FISH, through a global top-$k$, also fail to glue.
- MeZO's per-step method, the line $\theta+tz$ with $z$ spanning all layers, is not local.
- VeRA and NOLA are sheaves in their trainable parameters; they share only frozen randomness.

**What this explains.** Whether per-layer adapters can be mixed and matched, how they communicate in distributed training, and how storage scales.

---

## 3. Classification

**Axes.**
- **Kind**: R = reparametrization (lives in the base fibre) · E = pointed extension · E° = unpointed extension · S = schedule of base changes · P = post-hoc operation on trained points · F$_T$ = family over $T$.
- **Shape** of the image germ: Lin = linear subspace · Cone = determinantal-type cone with $\theta_0$ at the apex · Cone* = cone with $\theta_0$ at a smooth point · Orb = group orbit · Prod = product set of subgroups · Sat = torus saturation · ∪Lin = union of coordinate planes · Fun = function-level only.
- **Grp** (structure group): + translations · T⊙ Hadamard/diagonal torus · O orthogonal.
- **Nat** (naturality group): GL = $GL\times GL$ · O = $O\times O$ (spectral) · Mono = monomial/coordinate · Blk = block · F = Fourier · Bor = $O\times$Borel.
- **Base**: ∅ independent · θ base-dependent · D data-dependent.
- **Pt** (pointing): apex · smooth · reg = regular · ≈ = approximate, with a defect · none.
- **deg**: polynomial degree in the trainable parameters; ω = non-polynomial.
- **Loc**: L sheaf · L* sheaf with shared frozen randomness · G shared trainable parameters · B budget-coupled.
- **Mrg**: ✓ mergeable · ✗ not · ✓c mergeable if constant over $T$.
- **Reb**: id = idempotent under rebasing · → $X$ = closure under rebasing.

| Method | Kind | Shape | Grp | Nat | Base | Pt | deg | Loc | Mrg | Reb |
|---|---|---|---|---|---|---|---|---|---|---|
| Frozen / Full FT | R (initial / terminal) | $\{\theta_0\}$ / $\Theta$ | + | all / GL | ∅ | – / reg | 0 / 1 | L | ✓ | id |
| LoRA | R | Cone$_r$ | + | GL | ∅ | apex | 2 | L | ✓ | →Θ, $K^*=\lceil\min/r\rceil$ |
| rsLoRA, LoRA+ | R (≅ LoRA; metric only) | Cone$_r$ | + | GL | ∅ | apex | 2 | L | ✓ | →Θ |
| LoRA-FA | R | Lin $\{XA_0\}$, $d=mr$ | + | GL×O | ∅ | reg | 1 | L | ✓ | id |
| DoRA | R | Sat(Cone$_r$) | +,T⊙ | Mono×O | θ | apex | ω | L | ✓ | →Θ |
| PiSSA / MiLoRA | R | Cone*$_r$ (top / bottom SVD) | + | O | θ | smooth (μ=0) | 2 | L | ✓ | →Θ |
| OLoRA | R | Cone* (QR) | + | Bor | θ | smooth | 2 | L | ✓ | →Θ |
| LoRA-GA | R | Cone* (gradient SVD) | + | O | D | smooth | 2 | L | ✓ | →Θ |
| EVA | R | Cone$_r$, row$(A_0)$ = activation PCA | + | GL×O | D | apex | 2 | B | ✓ | →Θ |
| CorDA | R | Cone* (SVD of $W_0C$) | + | O ($C$-twisted) | D | smooth | 2 | L | ✓ | →Θ |
| QLoRA | R at $\kappa\theta_0$ | Cone$_r$ | + | Mono (quantizer) | ∅ | ≈ ($e$) | 2 | L | ✓ (dequant.) | →Θ |
| LoftQ | R | Cone* over $\kappa\theta_{\rm res}$ | + | Mono | θ | ≈ (min.) | 2 | L | ✓ (high-prec.) | →Θ |
| AdaLoRA | R | $\bigcup_{\sum r_\ell\le R}\prod$Cone$_{r_\ell}$ | + | O (SVD gauge) | ∅ | apex | 3 | B | ✓ | →Θ |
| DyLoRA | R | nested Cone$_1\subset\dots\subset$Cone$_r$ | + | GL | ∅ | apex | 2 | L | ✓ | →Θ |
| ReLoRA | S | Cone$_r$ per stage → $M_{\le Kr}$ | + | GL | ∅ | apex | 2 | L | ✓ | is rebasing |
| VeRA | R | bilinear ⊂ $\{XA\}$, $d=m+r-1$ | + | Mono | ∅ | apex | 2 | L* | ✓ | → LoRA-FA($A$) |
| NOLA | R | bilinear ⊂ span$\{B_jA_i\}$ | + | O (in distribution) | ∅ | apex | 2 | L* | ✓ | → span |
| VB-LoRA | R | ∪ (top-k) bilinear | + | Mono | ∅ | – | ω | G | ✓ | →Θ (generic) |
| LoRA-XS / SVFT | R | Lin in SVD frame ($r^2$ / sparse) | + | O | θ | reg | 1 | L | ✓ | id |
| FourierFT | R | Lin ($n_c$ Fourier modes) | + | F | ∅ | reg | 1 | L* | ✓ | id |
| LoHa | R | Hadamard variety ⊂ $M_{\le r_1r_2}$ | + | Mono | ∅ | apex | 4 | L | ✓ | →Θ |
| LoKr / KronA | R | VLP-rank-1 / VLP-rank-$k$ | + | GL on tensor factors | ∅ | apex | 2–3 | L | ✓ | →Θ |
| TT / Tucker (LoRETTA, FacT) | R | tensor-network variety | + | GL on bonds | ∅ | apex | #cores | L / G (FacT) | ✓ | →Θ |
| MoRA | R | Lin, $d=\hat r^2$, high rank | + | Blk | ∅ | reg | 1 | L | ✓ | id |
| HiRA | R | $D_{W_0}$(Cone$_r$) | T⊙ | Mono | θ | apex | 2 | L | ✓ | →Θ |
| OFT (block) | R | Orb of $\prod SO(b)$ | O | GL×Blk-O | θ | reg | ω | L | ✓ | id |
| BOFT | R | Prod (butterfly) ⊂ $SO(n)\theta_0$ | O | GL×butterfly | θ | reg | ω | L | ✓ | → $SO(n)\theta_0$ |
| HRA | R | LoRA$_r$ ∩ $W_0SO(n)$ | O ∩ + | GL×O | θ | reg ($r$ even) | ω | L | ✓ | → $O(n)\theta_0$ |
| GOFT | R | Prod (Givens) | O | GL×graph | θ | reg | ω | L | ✓ | iff graph connected |
| (IA)$^3$ | R (merged) | Lin ∩ Orb$_{T_+}$ | T⊙ | Mono | θ | reg | 1 | L | ✓ | id |
| SSF | R (merged) | Lin $\mathrm{diag}(\gamma)W_0+\beta$ | T⊙⋉+ | Mono | θ | reg | 1 | L | ✓ | id |
| BitFit | R | Lin (bias block) | + | Blk | ∅ | reg | 1 | L | ✓ | id |
| LayerNorm tuning | R | Lin (LN block) | + | Mono | ∅ | reg | 1 | L | ✓ | id |
| Diff pruning | R | ∪Lin (Sparse$_k$) | + | Mono | ∅ | apex | 2 | L/B | ✓ | →Θ |
| FISH mask | R | Lin $\mathbb R^S$ | + | Mono | D | reg | 1 | B | ✓ | id |
| LISA | S | moving coordinate blocks | + | Blk | ∅ | reg | 1 | L | ✓ | →Θ |
| GaLore | S (≡ one-sided LoRA, rebased) | moving Lin $\{P_tX\}$ | + | O | D | reg | 1 | L | n/a | →Θ |
| Flora | S | moving Lin $\{XA_t\}$ | + | O (in distribution) | ∅ | reg | 1 | L | n/a | →Θ |
| MeZO | S (zeroth order) | moving line $\theta+tz$ | + | O(Θ) (in distribution) | ∅ | reg | 1 | G | n/a | →Θ |
| Houlsby / Pfeiffer | E | Fun (serial, nonlinear) | – | GL (bottleneck) | ∅ | ≈ (near-id) | ω | L | ✗ | – |
| Parallel adapter | E | Fun (linear case ≅ LoRA) | – | GL | ∅ | ≈ | ω | L | ✗ (✓ if linear) | – |
| Compacter | E | Fun; PHM weights | – | GL⊗GL | ∅ | ≈ | ω | G | ✗ | – |
| AdapterFusion | E, F$_X$ | Fun | – | – | ∅ | ≈ | ω | G | ✗ | – |
| LST | E | Fun (side network) | – | – | ∅ | ≈ | ω | G | ✗ | – |
| LLaMA-Adapter | E | Fun (gated prefix) | – | – | ∅ | exact (zero gate) | ω | L | ✗ | – |
| Prompt tuning | E° | Fun (input state) | – | – | ∅ | none | ω | single box | ✗ | – |
| Prefix / P-tuning v2 | E° | Fun (gated parallel adapter on attention) | – | – | ∅ | none | ω | L | ✗ | – |
| Task arithmetic | P | $\theta_0+\sum\lambda_i\tau_i$ | + | GL(Θ) | – | – | 1 | L | ✓ | – |
| TIES / DARE | P | coordinatewise / stochastic | – | Mono (signed permutations) | – | – | ω | L | ✓ | – |
| LoRA soups | P | Minkowski sum ⊂ LoRA$_{kr}$ | + | GL | – | – | – | L | ✓ | – |
| LoraHub | P, F$_\Delta$ | factor-wise: **gauge-dependent** | – | not gauge-natural | – | – | 2 in $w$ | L | ✓ | – |
| MoE-LoRA (MoLE, LoRAMoE) | F$_X$ | Fun | + | – | – | apex | – | L | ✓c | – |
| S-LoRA / Punica | F$_U$ | relative computation over a shared base | + | – | – | – | – | L | deliberately unmerged | – |
| Text-to-LoRA, HyperFormer | F$_T$ | $T$-point of LoRA / adapters | + | – | D | – | – | G | ✓ per task | – |
| UniPELT / MAM | ⊕ of primitives (gates: F$_X$) | Minkowski sum | + | mixed | ∅ | mixed | – | L | partial | – |

---

## 4. Equivalences and obstructions

**Equivalences.**
1. Linear parallel adapter ≅ LoRA, as pointed objects.
2. Linear serial adapter $(I+UD)W_0$ ≡ LoRA in image iff $\operatorname{rank}W_0=n$.
3. Prefix tuning = gated parallel adapter on the attention output (He et al. 2022).
4. rsLoRA ≅ LoRA, an isomorphism in the slice.
5. LoRA+$(\lambda)$ ≡ LoRA$(\lambda\alpha)$ dynamically, under Adam without weight decay; ≡ LoRA$(\sqrt\lambda\,\alpha)$ under SGD.
6. GaLore ≡ one-sided LoRA rebased (Torroba-Hennigen et al. 2025).
7. LoRA-FA ≡ a fixed gradient projection $G\mapsto GA_0^\top A_0$, which is Flora's observation.
8. LoKr (free factors) ≡ LoRA$_1$ in the Van Loan–Pitsianis frame, including dynamics under SGD and Adam.
9. LoHa$_{1,1}$ = LoRA$_1$.
10. HiRA = $(D_{W_0})_*$LoRA.
11. (IA)$^3$ = HiRA-FA$_1$.
12. FourierFT, LoRA-XS and SVFT are coordinate masks in transported frames.
13. $\mathrm{HRA}_r=\mathrm{LoRA}_r\cap\mathrm{OFT}_{SO}$ for injective $W_0$ and even $r$.
14. ReLoRA$^K$ reaches exactly LoRA$_{Kr}$.
15. VeRA$^\infty$ = LoRA-FA($A$).
16. Constant-gate MoE of LoRAs = LoRA$_{\sum r_i}$.
17. QLoRA = LoRA at the base point $\kappa\theta_0$.

**Obstructions.**
1. There is no smooth simulation between initialization strata, by $S_1$ and the apex argument.
2. Splitting inits and zero inits are incomparable in expressivity.
3. No LoRA$_r$ initialization reaches an update of rank $>2r$.
4. Orthogonal methods cannot change any neuron norm or angle.
5. Idempotent methods gain nothing from rebasing.
6. Block-OFT cannot leave $\prod SO(b)$.
7. Prefixes cannot reweight content tokens relative to one another.
8. Nonlinear adapters, prefixes and routed mixtures are not locally mergeable.
9. Factor-wise LoRA arithmetic is not well-defined on adapters.
10. Coordinate methods on gauge-symmetric spaces depend on the checkpoint representative.
11. A pointed HRA requires even $r$.

---

## 5. Rosetta stone

| PEFT | Relative / categorical |
|---|---|
| pretrained checkpoint | base point $\theta_0:1\to\Theta$; object of $\mathbf{PNet}$ |
| PEFT method | object of the pointed slice $\mathbf{Diff}_*/(\Theta,\theta_0)$ ≅ pointed 2-cells into $f$ in $\mathbf{Para}$ |
| "M can simulate N" | morphism in the slice |
| frozen / full FT | initial / terminal object |
| expressivity | image factorization $Q\twoheadrightarrow\mathrm{Im}\rightarrowtail\Theta$; poset reflection |
| what both can express | product = fibre product (meet) |
| stacking adapters additively | convolution monoidal product; Minkowski sum |
| rank / LoRA$_r$ | factorization through a small object; coend; co-Yoneda colimit = full FT |
| initialization | point of the fibre; moduli $\mathrm{Gr}\sqcup\mathrm{Gr}\sqcup M_r$ |
| zero-init vs PiSSA-type init | apex (singular point) vs smooth point of the image |
| parameter redundancy | gauge group = generic fibre ($GL_r$) |
| balancedness | moment map / Noether charge; Kempf–Ness slice |
| weight decay on factors | nuclear norm on the quotient |
| $\alpha$, LR ratios, init scale | metric data; torus action on hyperparameters; orbit space |
| invariant optimizer | optimizer equivariant under the whole gauge group |
| multiplicative method | group orbit; fibre of $\Theta\to\Theta/\!\!/G$ |
| hyperspherical energy | $O(n)$-invariants (first fundamental theorem) |
| magnitude / direction | torus quotient; saturation monad |
| "new method" = old one in new coordinates | transport of structure along $GL(\Theta)$ |
| zero inference overhead | lift along the realization $\Phi_f$ |
| identity-at-init adapter | function-preserving embedding ($\mathbf{PNet}$-morphism) |
| prompt tuning | unpointed extension |
| QLoRA / LoftQ | moving the base point / approximate section |
| ReLoRA, GaLore, LISA, MeZO | iterated base change; generated submonoid, subgroup or Lie algebra |
| per-layer adapters | sheaf on box sets; shared banks = gluing failure |
| "works for every model" | section of the opfibration $\int\mathsf{Meth}\to\mathbf{PNet}$ |
| generator / router / mixer / server | $T$-points for $T$ = tasks / inputs / simplex / tenants |
| task arithmetic vs TIES/DARE | $GL(\Theta)$-natural vs monomial-natural operations |
| GaLore vs LoRA | cotangent vs tangent restriction, which coincide for linear maps (reverse derivative $P\mapsto P^\top$) |

---

## 6. What is analogy rather than theorem

1. **"Sheaf" for locality.** It is correct, but the coverage is discrete, so the content is "is $Q$ a product?".
2. **"Descent" for merging.** This is an analogy. The real content is a lifting problem, which we only decide *locally* (box by box). Global non-mergeability of whole networks is generally unproved.
3. **The opfibration $\int\mathsf{Meth}\to\mathbf{PNet}$.** It is bookkeeping: it makes "uniform method" precise but predicts little by itself.
4. **Tangent/cotangent (lens) duality.** This is exact only for *linear* reparametrizations. For bilinear LoRA, the "projected gradient" view is state-dependent.
5. **Noether.** It holds for gradient flow. Discrete SGD conserves the charge only up to $O(\eta^2)$ per step, and Adam breaks it.
6. **Erlangen.** It is a theorem for orbit methods (OFT, BOFT, HRA, (IA)$^3$). For additive methods it is a naming device: translations act on everything, and the content is the shape of $C$.
7. **Saturation monad, coend formula.** Both are true and low-yield.
8. **"Moduli space".** Here it means a set of isomorphism classes with its natural geometry; no fine moduli stack is constructed. "Haar-random point" refers to the law of $\mathrm{row}(A_0)$ under Gaussian init.
9. **Expressivity is not learnability.** Every image statement concerns reachable sets, not what training finds or how it generalizes. Thms 5, 6 and 8 are the only dynamical ones.
10. **Numerical checks.** The checks marked [Num] are evidence on random small instances. They are not proofs where the proof is listed as missing (the LoHa equality).

---

## 7. Interactive visualizations (each tied to one concept)

1. **The apex and the smooth point (Thm 4).**
   - *Concept:* symmetric $2\times2$ rank-$\le1$ matrices form the double cone $ac=b^2$ in $\mathbb R^3$.
   - *User manipulates:* the init (zero-init at the apex, or PiSSA-type at a smooth point) and a target.
   - *User sees:* $S_1$ drawn as a line versus a tangent plane, the gradient-flow path, and the first-order deficit counter $r(n-r)$ for user-chosen $m,n,r$.
2. **Gauge hyperbolas and the Noether charge (Thm 5).**
   - *Concept:* scalar LoRA $w=ba$ and the conserved charge.
   - *User manipulates:* $\eta_B/\eta_A$, the init $(a_0,0)$, weight decay, and an optimizer toggle (GD vs Adam).
   - *User sees:* trajectories on the level sets $b^2/\eta_B-a^2/\eta_A=$ const; weight decay pulling toward the balanced diagonal; Adam visibly leaving the level set.
3. **Hyperparameter orbit explorer (Thm 6).**
   - *Concept:* the torus action on LoRA's hyperparameters.
   - *User manipulates:* four sliders $(\alpha,\eta_A,\eta_B,\sigma_A)$ on a live toy matrix regression trained with Adam.
   - *User sees:* the two invariants displayed. Moving along an orbit gives *identical* loss curves (overlaid); moving across orbits does not. A "LoRA+ ↔ α" button jumps between equivalent settings. Toggles for $\varepsilon$, weight decay and clipping show where exactness breaks.
4. **Moduli map of initializations (Thm 4).**
   - *Concept:* the three strata $\mathrm{Gr}(r,n)$, $\mathrm{Gr}(r,m)$, $M_r$.
   - *User manipulates:* a method selected from a menu.
   - *User sees:* every init method as a labelled point (vanilla, EVA, Init[B], PiSSA, MiLoRA, OLoRA, CorDA, LoRA-GA, LoftQ), with its pointing defect, $\mu_0$, and $\dim S_1$.
5. **Erlangen orbit lab (Thms 9–10).**
   - *Concept:* the invariants each method preserves.
   - *User manipulates:* the method applied to three neuron vectors in $\mathbb R^3$: OFT, (IA)$^3$, DoRA, LoRA, HRA.
   - *User sees:* a live Gram-matrix heatmap showing which invariants survive, and an "intersection" mode that highlights HRA as LoRA ∩ OFT.
6. **Rebasing reachability (Thm 8).**
   - *Concept:* the closure under rebasing.
   - *User manipulates:* the edges of a graph on $n$ coordinates (rotation planes), or a choice between LoRA-FA (fixed $A$) and ReLoRA.
   - *User sees:* connected components and $\dim\mathrm{Lie}=\sum c_i(c_i-1)/2$ (block-OFT stuck, butterfly connected); a rank bar that grows with stages $K$ or stalls.
7. **Expressivity Hasse diagram.**
   - *Concept:* the expressivity preorder at one $m\times n$ layer.
   - *User manipulates:* hover and click on nodes (methods) and edges (proved inclusions).
   - *User sees:* the vertical axis $d(M)$, with rank shown as colour so that "high rank ≠ big image" is visible. Hovering shows the proof; clicking two nodes computes their meet when known.
8. **String-diagram builder (Thm 11).**
   - *Concept:* methods as diagrams of trainable and frozen boxes.
   - *User manipulates:* drags trainable/frozen boxes, copy spiders and ⊗ into a diagram.
   - *User sees:* the parameter count, polynomial degree, rank bound, and $d$ from a numerical Jacobian rank. A matcher names the known method (LoRA, LoHa, LoKr, VeRA, LoRA-XS, …) or reports "new".
9. **Mergeability lifting square (Thm 12).**
   - *Concept:* locality of mergeability and the prefix obstruction.
   - *User manipulates:* picks a method; for prefix tuning, a slider for prefix strength.
   - *User sees:* the square $Q\to\Theta\times\Theta'\to C^\infty$ versus $\Theta\to C^\infty$ turning green or red. For prefix tuning, attention bars over content tokens rescale uniformly and never reorder.
10. **Base change under quantization (Thm 13).**
    - *Concept:* pointing defects under QLoRA and LoftQ.
    - *User manipulates:* a 2D weight plane with a quantization grid; the order of operations (quantize→split vs split→quantize).
    - *User sees:* $\theta_0$, $\kappa\theta_0$, the LoRA line, QLoRA's pointing defect $e$, and LoftQ's corrected split.
11. **Classification explorer (§3).**
    - *Concept:* the classification axes.
    - *User manipulates:* the axes in a parallel-coordinates view.
    - *User sees:* clicking a method shows its diagram, image type, invariants, equivalences and obstructions; filtering ("mergeable ∧ rebasing-idempotent ∧ base-independent") yields method sets.

---

## 8. Key references

**Categorical background.**
- Fong, Spivak, Tuyéras, *Backprop as Functor*, arXiv:1711.10455.
- Cruttwell, Gavranović, Ghani, Wilson, Zanasi, *Categorical Foundations of Gradient-Based Learning*, arXiv:2103.01931.
- Gavranović et al., *Categorical Deep Learning is an Algebraic Theory of All Architectures*, arXiv:2402.15332.
- Cockett et al., *Reverse Derivative Categories*, arXiv:1910.07065.
- Vistoli, *Notes on Grothendieck topologies, fibered categories and descent theory*, arXiv:math/0412512.

**Low-rank family.**
- Hu et al., LoRA, arXiv:2106.09685.
- Kalajdzievski, rsLoRA, arXiv:2312.03732.
- Hayou, Ghosh, Yu, LoRA+, arXiv:2402.12354.
- Zhang et al., LoRA-FA, arXiv:2308.03303.
- Liu et al., DoRA, arXiv:2402.09353.
- Meng, Wang, Zhang, PiSSA, arXiv:2404.02948.
- Wang et al., MiLoRA, arXiv:2406.09044.
- Büyükakyüz, OLoRA, arXiv:2406.01775.
- Wang, Yu, Li, LoRA-GA, arXiv:2407.05000.
- Paischer et al., EVA, arXiv:2410.07170.
- Yang et al., CorDA, arXiv:2406.05223.
- Li et al., LoftQ, arXiv:2310.08659.
- Dettmers et al., QLoRA, arXiv:2305.14314.
- Zhang et al., AdaLoRA, arXiv:2303.10512.
- Valipour et al., DyLoRA, arXiv:2210.07558.
- Lialin et al., ReLoRA, arXiv:2307.05695.
- Kopiczko et al., VeRA, arXiv:2310.11454.
- Koohpayegani et al., NOLA, arXiv:2310.02556.
- Li, Han, Ji, VB-LoRA, arXiv:2405.15179.
- Bałazy et al., LoRA-XS, arXiv:2405.17604.
- Lingam et al., SVFT, arXiv:2405.19597.
- Gao et al., FourierFT, arXiv:2405.03003.
- Hyeon-Woo et al., FedPara/LoHa, arXiv:2108.06098.
- Yeh et al., LyCORIS, arXiv:2309.14859.
- Edalati et al., KronA, arXiv:2212.10650.
- Mahabadi, Henderson, Ruder, Compacter, arXiv:2106.04647.
- Yang et al., LoRETTA, arXiv:2402.11417.
- Jie, Deng, FacT, arXiv:2212.03145.
- Jiang et al., MoRA, arXiv:2405.12130.
- Huang et al., HiRA, ICLR 2025.

**Orthogonal and multiplicative.**
- Qiu et al., OFT, arXiv:2306.07280.
- Liu et al., BOFT, arXiv:2311.06243.
- Yuan, Liu, Xu, HRA, arXiv:2405.17484.
- Ma et al., GOFT, arXiv:2404.04316.
- Liu et al., (IA)$^3$, arXiv:2205.05638.
- Lian et al., SSF, arXiv:2210.08823.

**Selective and sparse.**
- Ben Zaken et al., BitFit, arXiv:2106.10199.
- Guo, Rush, Kim, Diff pruning, arXiv:2012.07463.
- Sung, Nair, Raffel, FISH mask, arXiv:2111.09839.
- Zhao et al., LayerNorm tuning, arXiv:2312.11420.
- Pan et al., LISA, arXiv:2403.17919.
- Nikdan et al., RoSA, arXiv:2401.04679.

**Adapters and prompts.**
- Houlsby et al., arXiv:1902.00751.
- Pfeiffer et al., AdapterFusion, arXiv:2005.00247.
- He et al., *Towards a Unified View of PETL*, arXiv:2110.04366.
- Mao et al., UniPELT, arXiv:2110.07577.
- Sung, Cho, Bansal, LST, arXiv:2206.06522.
- Zhang et al., LLaMA-Adapter, arXiv:2303.16199.
- Lester et al., Prompt tuning, arXiv:2104.08691.
- Li, Liang, Prefix tuning, arXiv:2101.00190.
- Liu et al., P-tuning v2, arXiv:2110.07602.
- Petrov, Torr, Bibi, arXiv:2310.19698.
- Wang et al., *Universality and Limitations of Prompt Tuning*, arXiv:2305.18787.

**Optimizer side.**
- Zhao et al., GaLore, arXiv:2403.03507.
- Hao, Cao, Mou, Flora, arXiv:2402.03293.
- Malladi et al., MeZO, arXiv:2305.17333.
- Torroba-Hennigen, Lang, Guo, Kim, *On the Duality between Gradient Transformations and Adapters*, arXiv:2502.13811.
- Zhang, Pilanci, Riemannian Preconditioned LoRA, arXiv:2402.02347.
- Yen et al., LoRA-RITE, arXiv:2410.20625.
- Schulman et al., *LoRA Without Regret*, Thinking Machines Lab blog, Sep 2025.

**Merging, serving, generation.**
- Ilharco et al., Task arithmetic, arXiv:2212.04089.
- Yadav et al., TIES, arXiv:2306.01708.
- Yu et al., DARE, arXiv:2311.03099.
- Huang et al., LoraHub, arXiv:2307.13269.
- Stoica et al., KnOTS, arXiv:2410.19735.
- Wu, Huang, Wei, MoLE, arXiv:2404.13628.
- Dou et al., LoRAMoE, arXiv:2312.09979.
- Sheng et al., S-LoRA, arXiv:2311.03285.
- Chen et al., Punica, arXiv:2310.18547.
- Charakorn et al., Text-to-LoRA, arXiv:2506.06105.
- Mahabadi et al., HyperFormer, arXiv:2106.04489.

**Theory.**
- Du, Hu, Lee, arXiv:1806.00900.
- Arora, Cohen, Hazan, arXiv:1802.06509.
- Zhao et al., *Symmetries, Flat Minima, and the Conserved Quantities of Gradient Flow*, arXiv:2210.17216.
- Kunin et al., *Neural Mechanics*, arXiv:2012.04728.
- Malladi et al., *A Kernel-Based View of LM Fine-Tuning*, arXiv:2210.05643.
- Zeng, Lee, *Expressive Power of LoRA*, arXiv:2310.17513.
- Zhu et al., *Asymmetry in Low-Rank Adapters*, arXiv:2402.16842.
- Hayou et al., *Impact of Initialization on LoRA*, arXiv:2406.08447.
- Aghajanyan et al., *Intrinsic Dimensionality*, arXiv:2012.13255.
- Chen, Goodfellow, Shlens, Net2Net, arXiv:1511.05641.
- Elhage et al., *Toy Models of Superposition* (privileged basis), arXiv:2209.10652.

**Classical mathematics.**
- Srebro, Rennie, Jaakkola, *Maximum-Margin Matrix Factorization*, NIPS 2004.
- Kempf, Ness, *The length of vectors in representation spaces*, 1979.
- Van Loan, Pitsianis, *Approximation with Kronecker products*, 1993.
- Cartan–Dieudonné theorem: E. Artin, *Geometric Algebra*, 1957.
- H. Weyl, *The Classical Groups*, 1939.
