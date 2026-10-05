# The Hole and the Plug: String Diagrams, Tensor Networks and the Monoidal Structure of PEFT

*Proposal for the lens "string diagrams, tensor networks and monoidal structure".*

**Status tags.** **[Std]** standard mathematics. **[Known]** known in the ML literature (cited). **[Ours]** not stated in this form anywhere we know of, though the mathematics underneath may be elementary. **[Trivial]** true and trivial, kept because it organises. **[Sketch]** proof sketch only. **[Pred]** a falsifiable prediction. **[Analogy]** heuristic, not a theorem. Claims marked ✓ are checked on random instances by `theory/proposals/string-diagram-checks.py`, where all 33 checks pass.

**Conventions.** We write $h=Wx$, with frozen weight $W_0\in\mathbb R^{m\times n}$ ($m=d_{\rm out}$, $n=d_{\rm in}$) and $r_0=\operatorname{rk}W_0$. LoRA is $\Delta W=BA$ with $B\in\mathbb R^{m\times r}$ and $A\in\mathbb R^{r\times n}$, and $G=\nabla_W L$. $\mathcal M_{\le r}$ denotes the matrices of rank at most $r$. "Generic" means outside a proper real-algebraic subset of parameter space.

---

## 0. Thesis

Cut the frozen weight out of a pretrained network's string diagram and a **hole** of type $V\to U$ remains. A PEFT method is a rule for filling the hole with a small **plug**: a tensor network, that is, a string diagram in $(\mathbf{FinVect},\otimes)$. Some of its boxes are frozen: $W_0$ itself, random or spectral bases, SVD frames, and the copy/merge spiders of the neuron basis. The rest are trainable. Almost everything practitioners argue about is a graph invariant of the plug:

- **Parameter count** is the total size of the trainable boxes.
- **Expressivity** is the image of the contraction map.
- **Maximal rank** is a min-cut.
- **Redundancy** is the gauge group $GL(r_e)$ on internal wires between trainable boxes, and each such wire carries a conserved Noether charge under gradient flow.
- **Mergeability** asks whether the activation side of the plug rewrites into a single linear box. Linear boxes fuse, nonlinear boxes obstruct, and diagonals slide through spiders.
- **Combination** is a biproduct sum $\oplus$ or a serial composite $\rhd$. The category of plugs also has products (intersections) and coproducts (unions), and these products produce named methods.

Read this way, the zoo collapses to a small grammar: *coupling to $W_0$ × graph shape × frozen/trainable labelling × cut × placement*. Several "different" methods turn out to be the same diagram drawn differently:

- Kronecker adapters are LoRA across a different cut.
- HiRA is a sum of two-sided (IA)³ scalings.
- DoRA's image is (IA)³ ⊕ LoRA.
- HRA is the pullback of LoRA and $SO(n)$ (by Cartan–Dieudonné).
- LoRA-XS is the pullback of the two one-sided LoRAs.
- Prefix tuning is a gated parallel adapter.
- GaLore, within one projection period, is one-sided LoRA.

---

## 1. Definitions

### 1.1 Three ambient categories, one per job

**(a) Boxes live in $\mathbf{FinVect}_{\mathbb R}$.** It has finite-dimensional real vector spaces and linear maps. It is symmetric monoidal under $\otimes$ (unit $\mathbb R$) and **compact closed**: every $V$ has a dual $V^*$, with cup and cap. So a morphism $V\to U$ is the same thing as a tensor in $U\otimes V^*$, and wires may be bent; transposition is wire-bending. Hom-sets are vector spaces, composition is bilinear, and $\oplus$ is a biproduct. A tensor network is a morphism of the free compact closed category on a signature of box symbols, and contracting it means evaluating it under a strong monoidal functor $[\![-]\!]$ into $\mathbf{FinVect}$. String diagrams are sound and complete for this calculus (Joyal–Street; Kelly–Laplaza; survey in Selinger), so *diagram equalities are equalities of contractions*.

**(b) Networks live in $\mathbf{Para}(\mathbf{Smooth})$** (Fong–Spivak–Tuyéras; Cruttwell–Gavranović–Ghani–Wilson–Zanasi). Here $\mathbf{Smooth}$ is Euclidean spaces with smooth maps, and is cartesian. A 1-cell $A\to B$ is a pair $(P, f:P\times A\to B)$. A 2-cell is a reparametrisation $r:P'\to P$, which acts by $f\mapsto f\circ(r\times A)$. We draw diagrams in two dimensions:

- **activation wires** run horizontally;
- **parameter wires** enter boxes vertically from above.

Because $\mathbf{Smooth}$ is cartesian, it has copy $\Delta$ and delete $!$. Copying an activation wire gives a residual or side connection. Copying a parameter wire gives weight sharing.

**(c) Methods live in $\mathbf{Def}$**: sets and maps definable in the o-minimal structure $\mathbb R_{\exp}$ (Wilkie). Every parameter-side map used in PEFT is definable: polynomials, Cayley's rational map, norms and square roots, softmax, tanh, top-$k$. Two facts are used.

- **Images are definable.** In the semialgebraic case this is Tarski–Seidenberg. So every update set has a dimension and a stratification into manifolds, even when it is singular, as $\mathcal M_{\le r}$ is at lower rank. A category of manifolds would fail here.
- **Definable choice.** Definable surjections have definable sections (van den Dries, Ch. 6).

Only the parameter side needs to be definable. Activation functions never enter $\mathbf{Def}$.

### 1.2 Spiders, and why the neuron basis is privileged

A basis $\{e_i\}$ of $X$ defines four maps:

- copy $\delta:e_i\mapsto e_i\otimes e_i$;
- merge $\mu:e_i\otimes e_j\mapsto\delta_{ij}e_i$;
- unit $\eta:1\mapsto\sum_i e_i$;
- counit $\varepsilon:e_i\mapsto 1$.

Together they form a special commutative Frobenius algebra. Connected composites of these maps depend only on their arity, and are called **spiders**. In FdHilb, orthonormal bases correspond exactly to special commutative †-Frobenius algebras (Coecke–Pavlovic–Vicary). PEFT uses three spider gadgets:

- **Spider with a state:** $\operatorname{diag}(\ell)=\mu\circ(\ell\otimes\mathrm{id})$. This is every scaling method: (IA)³, SSF, LayerNorm's $\gamma$, DoRA's magnitude, VeRA's vectors, AdaLoRA's $\Lambda$.
- **Hadamard product:** $X\circ Y=\mu\circ(X\otimes Y)\circ\delta$. This covers LoHa and HiRA.
- **Elementwise nonlinearity:** $\sigma^{\times d}$, a box that respects the spider.

**Lemma 1.1 (σ fixes the basis) [Std; elementary].** Let $\sigma\in C^2$ be non-affine, and let $L\in GL_d$ satisfy $L\,\sigma^{\times d}=\sigma^{\times d}L$. Then $L$ is a monomial matrix $PD$, as follows:

- If $\sigma$ is analytic, $D=I$; or $D=\pm I$ entrywise when $\sigma$ is odd (tanh).
- If $\sigma$ is ReLU or leaky ReLU, $D$ is a positive diagonal.

*Proof.* For row $i$, the left side $\sum_jL_{ij}\sigma(x_j)$ has zero mixed partials $\partial_{x_j}\partial_{x_k}$. The right side has mixed partial $L_{ij}L_{ik}\sigma''(\sum_l L_{il}x_l)$. So each row has a single nonzero entry, and $L=PD$. The remaining condition $d\,\sigma(x)=\sigma(dx)$ forces $d=1$ for an analytic non-linear $\sigma$, by comparing Taylor coefficients. It allows $d=-1$ exactly when $\sigma$ is odd, and allows every $d>0$ for positively homogeneous $\sigma$. ∎

*What this explains.* The group of symmetries a method may legitimately ignore depends on σ: permutations for GELU and SiLU, signed permutations for tanh, positive monomials for ReLU. Methods whose frozen structure breaks this group impose structure the pretrained net does not have (obstruction O4).

### 1.3 Gadgets: tensor networks with a hole

**Definition 1 (gadget).** A **linear PEFT gadget** on $W_0\in\mathrm{Hom}(V,U)$ consists of:

1. A finite formal $\mathbb Z$-combination $\Gamma=\sum_a\varepsilon_a\Gamma_a$ of tensor-network diagrams. Each $\Gamma_a$ has one input leg of type $V$ and one output leg of type $U$; legs may be split along fixed reshape isomorphisms $V\cong\bigotimes_iV_i$.
2. A labelling of boxes as **frozen** or **trainable**.
   - Frozen boxes have fixed values: $W_0$, spiders, DFT matrices, random matrices, SVD or data-derived frames, reshape isos.
   - A trainable box $t$ has a **box space** $Q_t\subseteq\mathrm{Hom}(X_t,Y_t)$, a linear subspace: all matrices, diagonals, skew-symmetric matrices, or a fixed support.
3. Definable **parameter-side maps** $c_t:\Psi_t\to Q_t$. These default to the identity; examples are Cayley, normalisation, softmax mixing and hypernetworks.
4. A base point $\psi_0\in\Psi=\prod_t\Psi_t$ with $\kappa_D(\psi_0)=W_0$, where $\kappa_D(\psi)=[\![\Gamma]\!](\text{frozen values},c(\psi))$ is the **contraction map**.

**Definition 2 (statistics of a gadget).**

| Statistic | Definition |
|---|---|
| Size | $\lvert D\rvert=\sum_t\dim\Psi_t$ |
| Degree $\deg D$ | the largest number of trainable boxes in any summand $\Gamma_a$, i.e. the polynomial degree of $\kappa$ in the box values |
| Update set | $\mathrm{Upd}(D)=\kappa_D(\Psi)-W_0$, a definable set with o-minimal dimension $\dim\mathrm{Upd}(D)$ |
| Cut-rank | $\rho(D)=\max\operatorname{rk}$ over $\mathrm{Upd}(D)$ |
| Gauge group $\mathcal G(D)$ | generated by **wire insertions** (put $g_eg_e^{-1}$ on an internal wire $e$, then absorb one factor into each endpoint) and **scalar slides** (scalars commute past frozen boxes), restricted to transformations that keep every trainable box in its box space and fix every frozen box |

### 1.4 The category of gadgets

**Definition 3.** $\mathbf{Gad}(W_0)$ is the pointed slice $\mathbf{Def}_*/(\mathrm{Hom}(V,U),W_0)$. Its objects are triples $(\Psi,\psi_0,\kappa)$. Its morphisms are definable pointed maps $h$ with $\kappa'h=\kappa$, called **simulations**: $D'$ can reproduce every update $D$ can. A presentation $\Gamma$ carries more information than its underlying object: gauge, cut and merge structure live in the presentation.

**Lemma 4 [Std].**

1. The frozen gadget $(\{*\},*,W_0)$ is initial. Full fine-tuning $(\mathrm{Hom}(V,U),W_0,\mathrm{id})$ is terminal.
2. Binary **products** are fibre products $\Psi\times_{\mathrm{Hom}}\Psi'$, and $\mathrm{Upd}(D\times D')=\mathrm{Upd}(D)\cap\mathrm{Upd}(D')$.
3. Binary **coproducts** are wedges $\Psi\vee\Psi'$, realised definably as $\Psi\times\{\psi'_0\}\cup\{\psi_0\}\times\Psi'$, and $\mathrm{Upd}=\mathrm{Upd}(D)\cup\mathrm{Upd}(D')$.
4. A simulation $D\to D'$ exists iff $\mathrm{Upd}(D)\subseteq\mathrm{Upd}(D')$. *Proof:* take a definable section $s$ of $\kappa'$ over its image, reset $s(W_0):=\psi'_0$, and put $h=s\circ\kappa$. ∎

Part 4 is where $\mathbf{Def}$ pays off. In $\mathbf{Diff}$ the simulation preorder is strictly finer, because no smooth section exists through the singular vertex of $\mathcal M_{\le r}$.

### 1.5 Activation-side gadgets and mergeability

**Definition 5.** An **activation-side gadget** at a box $W_0:V\to U$ is a parametrised morphism $(\Phi,e:\Phi\times V\to U)$ in $\mathbf{Para}(\mathbf{Smooth})$ with $e(\phi_0,-)=W_0$. Its diagram may contain $W_0$, linear boxes, $\sigma$, softmax and normalisations. It is **box-locally mergeable** iff $e(\phi,-)$ is linear for every $\phi$.

**Observation [Trivial].** Linear gadgets (Def. 1) are exactly the box-locally mergeable activation-side gadgets, with parameter-side map $\phi\mapsto e(\phi,-)$. This dissolves the "reparametrisation vs extension" dichotomy: the two coincide precisely when the activation side is linear. Nonlinearity on the parameter side (DoRA's normalisation, Cayley in OFT, HRA's $u/\lVert u\rVert$) is irrelevant to merging. It is compile-time.

### 1.6 Methods on whole networks

**Definition 6.** Fix a network: a 2D diagram in $\mathbf{Para}(\mathbf{Smooth})$ over a signature $\Sigma$ of box types (Linear, Attention, Norm, σ, Embedding). A PEFT method consists of:

- **(i)** a placement $S$, a set of boxes;
- **(ii)** a gadget $D_b$ for each $b\in S$;
- **(iii)** a **sharing pattern**: a parameter-side map $s:\Psi_{\rm glob}\to\prod_b\Psi_b$. The identity gives a *box-local* method; diagonal copies give tying; a hypernetwork gives conditional sharing;
- **(iv)** optionally, a **schedule**: a sequence of placements and gadgets separated by merge steps (LISA, ReLoRA, GaLore).

**Remark 7 (naturality) [Trivial].** A box-local method whose $D_b$ depends only on the type of $b$ is a choice per generator of $\Sigma$. By the universal property of free monoidal categories on a signature, it extends uniquely to every diagram. That is all "LoRA works on any architecture" means.

The **non-local** methods are exactly those with a nontrivial sharing pattern:

- VB-LoRA: a trainable bank shared across layers;
- Compacter: shared $A_i$;
- FacT: a cross-layer tensor leg;
- SAID: a global projection;
- Text-to-LoRA: a hypernetwork;
- LoRAHub: global mixing weights.

These are not determined generator-wise. (This corrects seed item S11.)

### 1.7 Two ways to combine gadgets

**Definition 8 (parallel sum).** $D\oplus D':=(\Psi\times\Psi',(\psi_0,\psi_0'),\kappa+\kappa'-W_0)$. As a diagram, the two trainable networks sit side by side between a copy of the input and a sum of the outputs (the biproduct).

**Definition 9 (serial composition).** A **gadget scheme** is a family $W\mapsto D(W)$ defined for every base weight; LoRA, OFT, (IA)³ and DoRA are schemes. Serial composition is
$$(D'\rhd D)(W_0):\ (\psi,\psi')\mapsto\kappa_{D'(\kappa_D(\psi))}(\psi'),$$
that is, apply $D$, then apply $D'$ to the result.

**Proposition 10 [Ours, elementary].**

- **(a)** $(\mathbf{Gad}(W_0),\oplus,\mathrm{Frozen})$ is symmetric monoidal, and $\oplus$ is functorial on simulations. $\mathrm{Upd}(D\oplus D')$ is the Minkowski sum, $\rho$ is subadditive, and $\lvert\cdot\rvert$ is additive. $\oplus$ is *not* the categorical product, which is $\cap$; this corrects seed item S7.
- **(b)** $\mathrm{LoRA}_r\oplus\mathrm{LoRA}_s\cong\mathrm{LoRA}_{r+s}$, an isomorphism via block concatenation (the biproduct $\mathbb R^r\oplus\mathbb R^s$ on the bond).
- **(c)** Schemes under $\rhd$ form a monoid that is not commutative. Take a diagonal scheme $D$ and an orthogonal scheme $R$ acting on the same side, with $W_0=I$. Then $D\rhd R$ produces exactly the matrices with orthogonal rows, and $R\rhd D$ exactly those with orthogonal columns.
- **(d) Distributivity.** For a left-multiplicative scheme $R$ with invertible values, $R\rhd\mathrm{LoRA}\cong\mathrm{LoRA}\rhd R\cong R\oplus\mathrm{LoRA}$. The isomorphism is $(\psi,B,A)\mapsto(\psi,R(\psi)B,A)$, because $R(W_0+BA)=RW_0+(RB)A$.

---

## 2. Results

### 2.1 Cut and rank

**Theorem 11 (cut-rank bound) [Std].** Let $\Gamma$ be a connected tensor network with input legs of total type $V$ and output legs of total type $U$. A **cut** is a partition of its vertices (boxes and spiders) into $\mathcal I\sqcup\mathcal O$. Its weight $w$ is the product of the dimensions of all wires joining the two sides, counting open legs attached on the wrong side. Then
$$\operatorname{rk}[\![\Gamma]\!]\le\min_{\text{cuts}}w,$$
and over formal sums ranks add.

*Proof.* Contract each side separately, bending wrong-side legs across (compact closure). Then $[\![\Gamma]\!]=[\![\Gamma_{\mathcal O}]\!]\circ[\![\Gamma_{\mathcal I}]\!]$ factors through $\bigotimes_{e\in\mathrm{cut}}e$. ∎

*Corollaries.* All bounds are for one $m\times n$ box; ✓ marks a numerical check.

- **Rank $\le r$:** LoRA and every member of the LoRA family (VeRA, NOLA, VB-LoRA, LoRA-XS, LoRA-FA, PiSSA-type, DyLoRA, AdaLoRA), cut through the bond.
- **LoHa:** $r_1r_2$, cut through both bonds of the Hadamard diamond.
- **HiRA:** $r_0r$, cut through $W_0$'s image and the bond (✓, attained).
- **LoKr:** $\operatorname{rk}C\cdot r$.
- **MoRA:** $\hat r$ in its sharing variant; up to $\min(m,n)$ in its reshape variant.
- **FourierFT:** at most $2\min(\#\text{rows},\#\text{cols of the support }\Omega)\le 2n_c$ (✓).
- **HRA:** $r$.
- **Givens circuit with $g$ gates:** $2g$.
- **TT-matrix (MPO):** up to full rank (put every core on one side).
- **LoRA on both $W_Q$ and $W_K$:** the QK bilinear form changes by rank at most $2r$.
- **ReLoRA after $K$ phases:** $Kr$.

*What this explains.* The "high-rank PEFT" literature (LoHa, LoKr, MoRA, HiRA, FourierFT, OFT) is a catalogue of networks whose min-cut avoids an $r$-dimensional bottleneck. Theorems 12, 13 and 16 show why rank alone is the wrong figure of merit.

**Theorem 12 (bipartition incomparability) [linear algebra Std (operator-Schmidt / Van Loan–Pitsianis); PEFT reading Ours].** Let $V=V_1\otimes V_2$ and $U=U_1\otimes U_2$, and view $\Delta$ as a 4-leg tensor. Let $\operatorname{rk}$ be the rank across the cut $\{U_1U_2\}\mid\{V_1V_2\}$, and $\operatorname{rk}_\otimes$ the rank across $\{U_1V_1\}\mid\{U_2V_2\}$.

- **(a)** $\operatorname{rk}_\otimes\Delta\le k$ iff $\Delta=\sum_{i\le k}C_i\otimes D_i$. This is the same wire-bending that underlies the Van Loan rearrangement.
- **(b)** $\operatorname{rk}(C\otimes D)=\operatorname{rk}C\operatorname{rk}D$, while $\operatorname{rk}_\otimes(C\otimes D)=1$.
- **(c)** If $u,v$ have Schmidt ranks $\rho_u,\rho_v$, then $\operatorname{rk}_\otimes(uv^\top)=\rho_u\rho_v$.
- **(d)** Hence, for $d_1=d_2=s$, a generic rank-1 update has the maximal Kronecker rank $s^2=d$, while $I_d$ has Kronecker rank 1 and full rank (✓). For $r<d$ and $k<d$, $\mathrm{LoRA}_r$ and any $k$-term Kronecker-sum gadget are incomparable in $\mathbf{Gad}$.

*Proof of (c).* Write $u=\sum\sigma_ix_i\otimes y_i$ and $v=\sum\tau_jz_j\otimes w_j$ (Schmidt). Then $uv^\top=\sum_{ij}\sigma_i\tau_j(x_iz_j^\top)\otimes(y_iw_j^\top)$. Both families $\{x_iz_j^\top\}$ and $\{y_iw_j^\top\}$ are orthonormal, so this is an operator-Schmidt decomposition with $\rho_u\rho_v$ nonzero coefficients. ∎

*What this explains.* LoKr $C\otimes BA$, KronA $A\otimes B$, Compacter's PHM layers $\sum A_i\otimes B_i$ and TT-matrix adapters are **LoRA across a different cut** of the same 4-leg tensor, not "higher-rank LoRAs". They cannot represent the typical fine-tuning move "write one new feature into many neurons" (a generic rank-1 update). They also rest on an arbitrary reshape $\mathbb R^d\cong\mathbb R^{d_1}\otimes\mathbb R^{d_2}$ that the network does not possess.

**[Pred]** On targets $W^*=W_0+uv^\top$ with generic $u,v$, $\mathrm{LoRA}_1$ reaches zero loss and no Kronecker-sum gadget with $k<d$ terms does. With budgets equalised ($m=n=s^2$; KronA has $2s^2=2d$ parameters, the same as $\mathrm{LoRA}_1$), neither dominates.

**Theorem 13 (Hadamard gadgets: rank is attained, volume is not) [bound Std/Known (FedPara, LyCORIS state it); generic attainment and the dimension bound Ours].** Let $\mathrm{LoHa}_{r_1,r_2}$ be $\Delta=(B_1A_1)\circ(B_2A_2)$.

- **(a) Spider fusion.** $(B_1A_1)\circ(B_2A_2)=(B_1\bullet B_2)(A_1^\top\bullet A_2^\top)^\top$, where $\bullet$ is the face-splitting product, so $(P\bullet Q)_{i,(k,l)}=P_{ik}Q_{il}$ (✓). This gives a simulation $\mathrm{LoHa}_{r_1,r_2}\to\mathrm{LoRA}_{r_1r_2}$.
- **(b)** Generically $\operatorname{rk}\Delta=\min(m,n,r_1r_2)$ (✓, including prime $m=n=7$ with $r_1=r_2=3$).
- **(c)** $\dim\mathrm{Upd}(\mathrm{LoHa}_{r_1,r_2})\le(r_1+r_2)(m+n)-r_1^2-r_2^2-1$, whereas $\dim\mathrm{Upd}(\mathrm{LoRA}_{r_1r_2})=r_1r_2(m+n-r_1r_2)$.
- **(d)** $\mathrm{Upd}(\mathrm{LoHa}_{1,r})=\mathcal M_{\le r}$, because $(ba^\top)\circ Y=\operatorname{diag}(b)\,Y\operatorname{diag}(a)$.

*Proof of (b).* Put $R=r_1r_2$ and assume without loss of generality that $n\le m$.

1. The rows of $X=B_1\bullet B_2$ are points $b_{1i}\otimes b_{2i}$ of the Segre cone $\Sigma\subset\mathbb R^{r_1}\otimes\mathbb R^{r_2}$, which spans the whole space. Any $N$ generic points of an irreducible spanning cone span $\min(N,R)$ dimensions, by induction: a proper subspace cannot contain $\Sigma$.
2. Likewise $Y=A_1^\top\bullet A_2^\top$ has generic rank $\min(n,R)$.
3. If $R\le n$, then $Y^\top$ is onto $\mathbb R^R$ and $\operatorname{rk}XY^\top=\operatorname{rk}X=R$.
4. Otherwise $\operatorname{rk}XY^\top=n-\dim(\operatorname{row}Y\cap\ker X)$. The image of $\Sigma$ in $\mathbb R^R/\ker X$ spans that quotient, which has dimension $\min(m,R)\ge n$. So $n$ generic such points are independent modulo $\ker X$, and $\operatorname{rk}XY^\top=n$.

*Proof of (c).* The group $GL_{r_1}\times GL_{r_2}\times\mathbb R^*$, where the last factor trades a scalar between the two Hadamard factors, acts freely at generic points. Its orbits lie in the fibres, so Prop. 16 applies. ∎

*What this explains.* LoHa, HiRA and LoKr really do reach high rank at LoRA cost. But for $r_1=r_2=r\ge3$ and $m,n\gg r^2$ the set they reach is a **thin, curved subvariety**: its codimension inside the rank-$r^2$ variety is about $(r^2-2r)(m+n)$. Rank is a *cut* invariant and dimension is a *volume* invariant, and the literature conflates the two.

### 2.2 Gauge, Noether charges and dimension

**Theorem 14 (wire gauge and Noether charges) [principle Known (Du–Hu–Lee; Arora–Cohen–Hazan; Kunin et al.; Zhao et al.); diagrammatic localisation and PEFT corollaries Ours].**

- **(a) [Trivial]** Inserting $g\,g^{-1}$ on an internal wire and absorbing the factors leaves the contraction unchanged.
- **(b)** Let a Lie group act linearly and block-diagonally on $\Psi=\prod\Psi_t$, and let $F=L\circ\kappa$ be invariant. Under preconditioned gradient flow $\dot\psi_t=-\eta_t\nabla_tF$ (block-scalar rates), the quantity
$$Q_\xi=\sum_t\eta_t^{-1}\langle\psi_t,X_\xi\psi_t\rangle$$
is conserved for every Lie-algebra element $\xi$ whose generator $X_\xi$ is symmetric on each block.
- **(c) Edge form.** Let $e$ be an internal wire of dimension $r_e$ between unconstrained trainable boxes $T,T'$. Write $T_{(e)}$ for the matricisation of $T$ with the $e$-index as rows. Then
$$N_e=\eta_T^{-1}T_{(e)}T_{(e)}^\top-\eta_{T'}^{-1}T'_{(e)}T'^\top_{(e)}$$
is conserved. If an endpoint is constrained (say diagonal), only the projection of $N_e$ onto the symmetric part of the surviving Lie algebra is conserved; for a torus, that is $\operatorname{diag}N_e$. If an endpoint is frozen, the gauge on $e$ breaks to the stabiliser of the frozen box, which is generically trivial.

*Proof.* Invariance gives $\sum_t\langle\nabla_tF,X_\xi\psi_t\rangle=0$. Then $\dot Q_\xi=\sum_t2\eta_t^{-1}\langle\dot\psi_t,X_\xi\psi_t\rangle=-2\sum_t\langle\nabla_tF,X_\xi\psi_t\rangle=0$. For (c), apply this to $\exp(sE)$ with $E$ symmetric, acting as $E$ on $T$'s $e$-leg and as $-E$ on $T'$'s; then $Q_E=\operatorname{tr}(EN_e)$. ∎

*Caveats.* The law is exact only for continuous flow. For gradient descent the drift is $O(\eta)$ (✓: halving the step halves the drift). Adam, weight decay and gauge-breaking penalties break it, and Kunin et al. quantify the breaking.

*Corollaries (what this explains about PEFT).*

- **C1: LoRA and LoRA+.** The conserved charge is $N=AA^\top/\eta_A-B^\top B/\eta_B$ (✓). With zero-$B$ initialisation, $B^\top B=\lambda(AA^\top-A_0A_0^\top)$ where $\lambda=\eta_B/\eta_A$. Three consequences follow:
  - (i) $AA^\top\succeq A_0A_0^\top$ forever.
  - (ii) $\lVert B\rVert_F^2=\lambda(\lVert A\rVert_F^2-\lVert A_0\rVert_F^2)$. The zero-$B$ start is maximally unbalanced, and B can grow only in proportion to A's growth times λ. LoRA+'s $\lambda\gg1$ is exactly the knob that loosens this constraint, consistent with Hayou et al.'s width analysis.
  - (iii) In the scalar case $m=n=r=1$, $(a,b)$ moves on the hyperbola $a^2/\eta_A-b^2/\eta_B=\text{const}$.
- **C2: VeRA** ($\Delta=\Lambda_bB\Lambda_dA$, with $B,A$ frozen). Frozen boxes break $GL_r$. One scalar slides through them, $(b,d)\mapsto(cb,d/c)$, so $\lVert b\rVert^2/\eta_b-\lVert d\rVert^2/\eta_d$ is conserved (✓). With $b_0=0$ and $d_0=d_{\rm init}\mathbf 1$, VeRA's trajectory lies on a hyperboloid. The same holds for NOLA with $\lVert\alpha\rVert^2-\lVert\beta\rVert^2$.
- **C3: AdaLoRA** ($P\Lambda Q$). There are two torus edges, giving charges $\lVert p_k\rVert^2-\lambda_k^2$ and $\lVert q_k\rVert^2-\lambda_k^2$.
  - The proxy $\lambda_k$ is **not gauge-invariant**: $(PD,D^{-1}\Lambda)$ changes it.
  - AdaLoRA's penalty $\lVert P^\top P-I\rVert^2+\lVert QQ^\top-I\rVert^2$ is therefore a **gauge-fixing term**, and it is what makes $\lambda$-based pruning meaningful.
  - **[Pred]** A pruning score built from the invariant $\lvert\lambda_k\rvert\lVert p_k\rVert\lVert q_k\rVert$ makes the regulariser unnecessary for the pruning step.
- **C4: DyLoRA.** Nested truncations impose a flag on the bond, which reduces $GL_r$ to a torus, with charges $\lVert b_k\rVert^2-\lVert a_k\rVert^2$.
- **C5: TT and Tucker adapters** have one conserved matrix per bond.
- **C6: bottleneck adapters.** With ReLU, positive diagonals slide through σ (Thm 20, rule R3), so each bottleneck unit carries a charge. With GELU only a finite gauge survives.
- **C7: HRA.** A vector enters only through $u/\lVert u\rVert$, so $\lVert u_i\rVert^2$ is conserved.
- **C8: frozen-factor gadgets** (LoRA-FA, LoRA-XS, FourierFT, MoRA, (IA)³) have no gauge, no charges and no redundancy.

**Proposition 15 (optimisers that descend to the quotient) [Known: ScaledGD (Tong–Ma–Chi); Riemannian LoRA (Zhang–Pilanci); LoRA-RITE; the reading is Ours].**

- Plain gradient descent on $(B,A)$ induces $\dot W=-(\eta_B\,GA^\top A+\eta_A\,BB^\top G)$. This depends on the representative, because $A^\top A\mapsto A^\top g^\top gA$ under the gauge.
- The preconditioned flow $\dot B=-GA^\top(AA^\top)^{-1}$, $\dot A=-(B^\top B)^{-1}B^\top G$ induces $\dot W=-(G\,P_{\mathrm{row}A}+P_{\mathrm{col}B}\,G)$, which depends only on $W$.

*What this explains.* "Transformation invariance" (LoRA-RITE) means exactly that the dynamics descend to the quotient by the wire gauge. rsLoRA and LoRA+ are non-invariant metrics, tuned to compensate for the particular unbalanced section chosen by zero-$B$ initialisation.

**Proposition 16 (count ≥ volume + gauge) [Std; TT/HT case Known].** For any gadget,
$$\dim\mathrm{Upd}(D)\le\lvert D\rvert-\dim(\text{generic }\mathcal G(D)\text{-orbit}).$$
Equality holds for LoRA ($r(m+n-r)$) and for TT and hierarchical Tucker with admissible ranks (Holtz–Rohwedder–Schneider; Uschmajew–Vandereycken).

Wasted parameters per gadget:

| Gadget | Wasted |
|---|---|
| LoRA | $r^2$ |
| LoHa | at least $r_1^2+r_2^2+1$ |
| AdaLoRA | $r^2+r$ (see below) |
| VeRA, NOLA (generic) | 1 |
| TT | $\sum_k r_k^2$ |
| Degree-1 gadgets (LoRA-FA, LoRA-XS, FourierFT, MoRA, (IA)³) | 0 |

AdaLoRA's figure needs a word. $P\Lambda Q$ with $P,Q$ free still covers all of $\mathcal M_{\le r}$, so the waste is $r(m+n)+r-r(m+n-r)=r^2+r$. The torus wire gauge explains only $2r$ of it. The rest comes from a rewrite rule, *a spider-with-state next to an unconstrained box is absorbable* ($\Lambda Q$ ranges over all of $\mathbb R^{r\times n}$), and that redundancy acts non-linearly. Without its orthogonality penalty, AdaLoRA's $\Lambda$ is pure redundancy.

**[Pred]** Benchmarks should report $\dim\mathrm{Upd}$ next to $\lvert D\rvert$. At budgets of a few thousand parameters (VeRA, NOLA, VB-LoRA, LoRA-XS) the correction is not negligible for chain gadgets, and rankings at equal *effective* dimension can change.

### 2.3 Base points: the cone and its smooth points

**Theorem 17 (LoRA starts at the cone point) [elementary; reading Ours].** For $\kappa(B,A)=W_{\rm base}+BA$,
$$\operatorname{rk}d\kappa_{(B,A)}=m\operatorname{rk}A+n\operatorname{rk}B-\operatorname{rk}A\operatorname{rk}B\quad(✓).$$

*Proof.* The image of $d\kappa$ is $\{XA\}+\{BY\}$. The first summand has dimension $m\operatorname{rk}A$ and the second $n\operatorname{rk}B$. Their intersection is $\{M:\operatorname{row}M\subseteq\operatorname{row}A,\ \operatorname{col}M\subseteq\operatorname{col}B\}$, of dimension $\operatorname{rk}A\operatorname{rk}B$. ∎

*What this explains.*

- **(i) Zero-factor initialisations sit at the vertex** of the cone $W_{\rm base}+\mathcal M_{\le r}$: LoRA, QLoRA, EVA, LoRA-FA, and HiRA's zero factor. First-order motion is confined to $\{XA_0\}$, which has dimension $mr$ and pins the row space.
- **(ii) Splitting initialisations sit at a smooth point**, where $d\kappa$ is onto the full tangent space of dimension $r(m+n-r)$. These are the inits that rewrite the $W_0$ box as a sum $W_{\rm res}+B_0A_0$: PiSSA (top singular vectors), MiLoRA (bottom), OLoRA (QR), LoRA-GA (gradient SVD), CorDA (SVD of $W_0$ times the activation covariance), LoftQ (quantisation residual). For $m=n=4096$ and $r=16$, that is 130,816 directions against 65,536.
- **(iii) The double-counted overlap** $\{B_0RA_0\}$ has dimension $r^2$ and is exactly LoRA-XS (Thm 18a). The formula above is inclusion–exclusion (✓).
- **(iv) Two mechanisms, not one.** At initialisation $\tfrac{d}{dt}L=-(\eta_B\lVert GA_0^\top\rVert^2+\eta_A\lVert B_0^\top G\rVert^2)$. LoRA-GA aligns the tangent image with $G_0$'s top singular subspaces. EVA keeps the vertex but aligns $\operatorname{row}A_0$ with the dominant activation directions: since $G=\sum\delta x^\top$, the term $GA_0^\top$ captures the most energy that way. "Better initialisation" in the literature mixes **smoothness** with **alignment**.

### 2.4 Products in the gadget category generate named methods

**Theorem 18 (two named methods are categorical products) [Ours, to our knowledge].**

**(a)** Let $A_0$ have full row rank and $B_0$ full column rank. Write $\mathrm{FA}(A_0):B\mapsto W_0+BA_0$ and $\mathrm{FB}(B_0):A\mapsto W_0+B_0A$. Then
$$\mathrm{FA}(A_0)\times\mathrm{FB}(B_0)\;\cong\;(R\mapsto W_0+B_0RA_0,\ R\in\mathbb R^{r\times r}).$$
With SVD frames of $W_0$ this is **LoRA-XS**.

*Proof.* A point of the fibre product satisfies $BA_0=B_0A=:M$, so $M$'s row space lies in $\operatorname{row}A_0$ and its column space in $\operatorname{col}B_0$. Hence $M=B_0RA_0$ for a unique $R$, and $R\mapsto(B_0R,RA_0)$ is a definable bijection commuting with κ. ∎

**(b)** Let $r$ be even and $W_0$ injective on the side acted on. Write $\mathrm{SO}:R\mapsto W_0R$ for $R\in SO(n)$. Then
$$\mathrm{Upd}(\mathrm{HRA}_r)=\mathrm{Upd}(\mathrm{LoRA}_r)\cap\mathrm{Upd}(\mathrm{SO}).$$
So HRA is image-equivalent to the **pullback $\mathrm{LoRA}_r\times\mathrm{SO}$**.

*Proof.*

- (⊆) A product of $r$ reflections fixes $\bigcap u_i^\perp$, so $\operatorname{rk}(R-I)\le r$, and $\det R=(-1)^r=1$.
- (⊇) Suppose $W_0R=W_0+\Delta$ with $\operatorname{rk}\Delta\le r$. By injectivity, $k:=\operatorname{rk}(R-I)=\operatorname{rk}\Delta\le r$. Now peel (Cartan–Dieudonné, Euclidean form): pick $v$ with $Rv\ne v$ and let $H$ be the reflection in $(Rv-v)^\perp$. Then $HRv=v$, and $HR$ still fixes $\mathrm{Fix}(R)$, because $\langle w,Rv-v\rangle=\langle Rw,Rv\rangle-\langle w,v\rangle=0$. So the fixed space grows by one at each step, and $R$ is a product of exactly $k$ reflections (✓).
- Since $\det R=1$, $k$ is even. Pad with $(r-k)/2$ pairs $H_uH_u=I$. This matches HRA's paired initialisation, which is why its implementation asks for even $r$. ∎

*Remarks.* Cayley-parametrised OFT reaches an open dense subset of $SO(n)$, or of its block subgroup. A Givens circuit with $g$ gates lies in $\mathrm{LoRA}_{2g}\times\mathrm{SO}$, and by Prop. 16 parametrises a set of dimension at most $g$, far below $\dim SO(n)$.

*What this explains.* HRA's slogan "bridging low-rank and orthogonal adaptation" becomes a **universal property**. HRA is the largest gadget that is both a rank-$r$ update and an orthogonal transport of neurons. LoRA-XS is the largest gadget that factors through both frozen frames, the overlap a PiSSA init double-counts. Products also **predict methods**: if $W_0$ has full row rank, $\mathrm{LoRA}_r\times(\mathrm{IA})^3$ is "rescale at most $r$ output neurons", since $\operatorname{rk}(\operatorname{diag}(\ell-1)W_0)=\#\{i:\ell_i\ne1\}$.

### 2.5 Degree one: forward and backward restriction coincide

**Theorem 19 [(c) is due to Torroba-Hennigen–Lang–Guo–Kim 2025; (d) to Flora (Hao–Cao–Mou 2024); framing Ours].** Let $\deg D=1$, so $\kappa(q)=W_0+\mathcal Lq$ with $\mathcal L$ linear.

- **(a)** $\mathrm{Upd}(D)=\operatorname{im}\mathcal L$ is a linear subspace. If $\mathcal L$ is injective, the gadget has no gauge, no charges and no singular points.
- **(b)** For any first-order optimiser $\mathcal O$ acting in $Q$, the trajectory is $W_t=W_0+\mathcal L\,\mathcal O(\mathcal L^*G_1,\dots,\mathcal L^*G_t)$. "Restrict the forward parametrisation" and "project the cotangent with $\mathcal L^*$, then lift with $\mathcal L$" are the same algorithm.
- **(c)** GaLore with its projector $P\in\mathbb R^{m\times r}$ fixed during a period, and Adam on $P^\top G$, *is* the gadget $W_0+PA$ (frozen up-projection, trainable $A$) under Adam, with identical iterates (✓, to machine precision). Its scale $\alpha$ is a learning-rate factor. Changing $P$ every $T$ steps is merge-and-rebase. When GaLore projects on the right, the gadget is LoRA-FA. How optimiser state is handled at switches is an implementation choice with no canonical diagram.
- **(d)** For degree ≥ 2 the two differ: the tangent space moves with the point (Prop. 15). At the vertex, LoRA's flow is $\dot W=-\eta_BGA_0^\top A_0$, which is a random-projection gradient compressor.
- **(e)** MeZO's estimate $\frac{L(\theta+\epsilon z)-L(\theta-\epsilon z)}{2\epsilon}z=zz^\top\nabla L+O(\epsilon^2)$ restricts the cotangent to a fresh random line at every step, with no reverse pass. It composes with any gadget.

*What this explains.* The seed's tangent/cotangent split (S9) is real in only two places: for degree ≥ 2 gadgets, and through **re-basing schedules** (GaLore, ReLoRA, Flora, LISA) whose cumulative update escapes every fixed subspace.

### 2.6 Merging as rewriting

**Theorem 20 (mergeability calculus) [rules elementary; consequences Ours].** Each rule below is an equality of smooth maps between activation-side diagrams.

| Rule | Statement | Conditions |
|---|---|---|
| **R1** (fusion) | Composites and parallel sums of linear boxes are linear. | — |
| **R2** (permutation slide) | $P\sigma^{\times d}=\sigma^{\times d}P$ | every σ |
| **R3** (positive-diagonal slide) | $\operatorname{diag}(\ell)\sigma^{\times d}=\sigma^{\times d}\operatorname{diag}(\ell)$ for all $\ell>0$ | iff σ is positively homogeneous: ReLU and leaky ReLU, not GELU, SiLU or tanh (✓) |
| **R4** (spider slide) | $\operatorname{diag}(\ell)\circ\mu=\mu\circ(\operatorname{diag}(\ell)\otimes\mathrm{id})$ | any sign of $\ell$ (✓) |
| **R5** (cup slide) | $\langle Mq,k\rangle=\langle q,M^\top k\rangle$: boxes slide around the QK cup as their transposes | — |
| **R6** (norm absorption) | $W(\gamma\odot n(x)+\beta)=(W\operatorname{diag}\gamma)\,n(x)+W\beta$ | — |

Then:

- **(a)** A gadget whose activation side is all linear is box-locally mergeable.
- **(b)** A diagonal gadget merges into every linear box reachable from it by R1–R6.
- **(c)** If the activation function of a gadget is non-affine for some parameter value, it is not box-locally mergeable.
- **(d)** Merging is **type-relative**: $k$-bit grids are not closed under $+$, so merging QLoRA or LoftQ updates leaves the quantised type.

*Corollaries.*

- **Mergeable:** the LoRA family, DoRA, OFT, BOFT, HRA, Givens, VeRA, NOLA, VB-LoRA, FourierFT, LoHa, LoKr, KronA, HiRA, MoRA, AdaLoRA and PiSSA-type methods. Their nonlinearities, where they have any, are on the parameter side.
- **(IA)³:**
  - $\ell_k,\ell_v$ fuse into $W_K,W_V$.
  - The FFN vector fuses into $W_{\rm down}$ for every σ.
  - It also fuses into $W_{\rm up}$ under ReLU when $\ell>0$, and **under SwiGLU/GeGLU for every $\ell$** (R4), but **not** under GELU.
  - A query scaling can be moved onto $W_K$ (R5).
- **SSF** fuses into the preceding linear box. **LayerNorm tuning** fuses into the following one.
- **Not mergeable:** nonlinear Houlsby, Pfeiffer and parallel adapters; AdapterFusion and MoLE (input-dependent mixing); LST; prefix and LLaMA-Adapter (Thm 21); prompt tuning (it changes the sequence object).

*What this explains.* "Zero inference latency" is decidable by local rewriting, and the rewrite also says *which* weight absorbs the update. That matters for multi-adapter serving and quantised merging.

### 2.7 Sequence-wire gadgets

**Theorem 21 (attention is a homomorphism followed by a perspective map) [identity Std; (a) He et al. 2022; (b) Petrov–Torr–Bibi; organisation and (d) Ours].** Fix a query $q$. Let multisets of key–value pairs form a monoid under $\uplus$, and define
$$\Sigma_q(K)=\Big(\sum e^{\langle q,k\rangle/\sqrt{d}},\ \sum e^{\langle q,k\rangle/\sqrt d}\,v\Big).$$
Then $\Sigma_q$ is a monoid homomorphism into $(\mathbb R_{\ge0}\times\mathbb R^{d_v},+)$, and $\mathrm{Attn}_q=\pi\circ\Sigma_q$ with $\pi(Z,S)=S/Z$.

- **(a) Prefix = gated parallel adapter.** $\mathrm{Attn}_q(P\uplus C)=(1-\lambda_q)\mathrm{Attn}_q(C)+\lambda_q\mathrm{Attn}_q(P)$, with $\lambda_q=Z_q(P)/(Z_q(P)+Z_q(C))$ (✓).
- **(b) Pattern rigidity.** Content attention weights are multiplied by $1-\lambda_q$ (✓). Ratios between content tokens are unchanged, and the added term lies in $\operatorname{conv}V_P$.
- **(c) LLaMA-Adapter.** Normalising $P$ and $C$ separately and adding a zero-initialised gate gives $\mathrm{Attn}_q(C)+g\,\mathrm{Attn}_q(P)$. This uses two perspective maps. The content weights are left exactly unchanged, there is no $(1-\lambda)$ shrinkage, and the output is the identity at initialisation.
- **(d) Prompt tuning embeds in deep prefix tuning (causal decoders).** If soft prompts come first and attention is causal, the hidden states at prompt positions in every layer depend only on the prompt. Running the frozen net on $p$ therefore gives a simulation $\mathrm{Prompt}_\ell\to\mathrm{DeepPrefix}_\ell$, $p\mapsto(K_j(p),V_j(p))_j$. It is generally not onto. In bidirectional encoders it fails, because prompt positions read the content. P-tuning v2 is $\mathrm{DeepPrefix}_\ell$ itself. Li–Liang's MLP reparametrisation is a parameter-side box that collapses after training.
- **(e)** The gate $\lambda_q$ is non-affine in $x$, so prefix tuning does not merge (Thm 20c).

*What this explains.* Prefix-type methods elicit existing skills but cannot impose new attention patterns. LLaMA-Adapter's "zero-init attention" is a structural change, not just a stabilisation trick. And prompt tuning is a strict sub-method of P-tuning v2 in decoders.

### 2.8 Combining many gadgets

**Theorem 22 (gauge obstruction to factor-wise combination) [Ours in this form; related diagnoses: FFA-LoRA, KnOTS].** Take LoRA modules $(B_i,A_i)$ of rank $r$, $i=1,\dots,N$. The map $(B_i,A_i)_i\mapsto(B_iA_i)_i$ is invariant under $\mathcal G=\prod_iGL_r$. A combination rule $\Phi$ defined on factors descends to updates iff $\Phi$ is $\mathcal G$-invariant.

- **(a)** Task arithmetic $\sum w_iB_iA_i$ is invariant (✓), and lands in $\mathrm{LoRA}_{Nr}$ by Prop. 10b.
- **(b)** LoRAHub's rule $(\sum w_iB_i)(\sum w_jA_j)=\sum_{ij}w_iw_jB_iA_j$ is **not** invariant. Applying $g=cI$ to module 2 changes it by $w_1w_2[(c-1)B_1A_2+(c^{-1}-1)B_2A_1]\ne0$ (✓). The same obstruction hits:
  - factor averaging in federated LoRA;
  - "soups" of factors;
  - any hypernetwork loss that regresses factor pairs rather than products, relevant for Text-to-LoRA-style generators, which output $(A,B)$.
- **(c) Gauge fixing restores well-definedness.**
  - A shared frozen factor (FFA-LoRA, LoRA-FA) makes the gadget degree 1, and then factor-wise and update-wise combination agree.
  - An SVD canonical form leaves a residual $(\mathbb Z/2)^r$ of sign flips per module, plus rotations within repeated singular values, and these still change the cross terms.
  - What is needed is a *joint* frame, such as KnOTS's SVD of the concatenated updates.

*What this explains.* Factor-space merging is unstable, FFA-LoRA repairs federated LoRA, and KnOTS's alignment helps. **Design rule:** combine LoRAs on products, or only after a joint gauge fixing.

**Proposition 23 (diagonal methods are Hadamard-coupled) [elementary; reading Ours].**

- **(a) Spider identity.** $\operatorname{diag}(a)W_0\operatorname{diag}(b)=(ab^\top)\circ W_0$ (✓). Consequences:
  - One-sided (IA)³ is $\mathrm{HiRA}_1$ with $B=\ell-\mathbf 1$ and $A=\mathbf 1^\top$.
  - Two-sided scaling lies in $\mathrm{HiRA}_2$.
  - $\mathrm{Upd}(\mathrm{HiRA}_r)=\{\sum_{k\le r}\operatorname{diag}(b_k)W_0\operatorname{diag}(a_k)\}$, a sum of $r$ two-sided scalings.
- **(b) Torus orbit.** $\{\operatorname{diag}(a)W_0\operatorname{diag}(b)\}$ is the orbit of $W_0$ under the torus, and has dimension $m+n-c(W_0)$ (✓). Here $c(W_0)$ is the number of connected components of the bipartite support graph of $W_0$. A dense $W_0$ wastes exactly one parameter, with charge $\lVert a\rVert^2-\lVert b\rVert^2$.
- **(c) DoRA.** $\mathrm{Upd}(\mathrm{DoRA})=\{\operatorname{diag}(s)(W_0+BA)\}-W_0$. Choose the magnitude $m_i=s_i\lVert(W_0+BA)_{i,:}\rVert$; DoRA's magnitude is per output neuron, as in the PEFT implementation. Where $s$ has no zeros this equals $\mathrm{Upd}((\mathrm{IA})^3_{\rm out}\oplus\mathrm{LoRA}_r)$ (substitute $B'=\operatorname{diag}(s)B$), which is dense. DoRA's normalisation changes the geometry of optimisation (the direction becomes scale-invariant), not the image.

**Proposition 24 (backprop cone) [elementary].** Reverse mode must store:

1. the input wire of every trainable box, for its parameter gradient;
2. the input wire of every *nonlinear* box on a directed path from a trainable box to the loss.

Frozen linear boxes need nothing stored. Consequences:

- LST keeps the trainable network downstream of copy maps, so the cone avoids the backbone.
- LoRA-FA's trainable $B$ reads the $r$-dimensional wire $A_0x$ instead of $x$, which is its memory saving.
- Adapting only the top layers truncates the cone.
- LISA's cone starts at its lowest sampled layer.
- Prompt tuning's cone is the whole network.
- MeZO has no cone at all.

---

## 3. Classification

**Axes.**

- **Coupling** to $W_0$:
  - **+** additive/parallel;
  - **∘** serial/multiplicative;
  - **⊙** Hadamard with $W_0$;
  - **ι** internal coordinates of $P$;
  - **σ** nonlinear activation-side insertion;
  - **⊗** a state tensored into the sequence wire;
  - **∇** cotangent side;
  - **Σ** operations on many gadgets.
- **Shape**:
  - **pt** single box or spider-with-state;
  - **ed** edge (two boxes);
  - **ch** chain/MPO;
  - **st** star/core;
  - **kr** Kronecker under a reshape;
  - **hd** Hadamard diamond;
  - **cj** conjugation sandwich;
  - **bf** butterfly or gate circuit;
  - **dc** frozen dictionary with trainable coefficients;
  - **rf** reflection chain.
- **deg** (Def. 2).
- **ρ** (Thm 11).
- **|D|** for one $m\times n$ box.
- **Gauge** (Thm 14).
- **Symmetry** of the update class:
  - **GL** basis-free;
  - **O**;
  - **Mon** neuron-basis monomials (Lemma 1.1);
  - **Π_X** extra structure $X$ imposed (blocks, tensor factorisation ⊗, cyclic order, pairing);
  - **Fr(·)** a frame taken from $W_0$, the gradient, data, or a random draw.
- **Merge**:
  - **M0** internal, nothing to merge;
  - **M1** fuses into the box;
  - **M1→** fuses into a neighbour;
  - **Mq** fusion leaves the quantised type;
  - **M∞** obstructed.
- **Base**:
  - **V** at the vertex of a singular update set;
  - **S** at a smooth point;
  - **lin** the update set is linear.
- **Locality**: **L** box-local, **N** shared.

**Table A: weight-space gadgets.**

| Method | Coupling | Shape | deg | ρ | \|D\| | Gauge | Symmetry | Merge | Base | Loc |
|---|---|---|---|---|---|---|---|---|---|---|
| LoRA | + | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | GL | M1 | V | L |
| rsLoRA | + | ed + scalar $\alpha/\sqrt r$ on bond | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | GL | M1 | V | L |
| LoRA+ | + | ed, metric $\eta_B/\eta_A$ | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | GL | M1 | V | L |
| LoRA-FA | + | ed, $A$ frozen | 1 | $r$ | $mr$ | — | Fr(rand) | M1 | lin | L |
| DoRA | ∘ after + | spider·(W₀+ed), param-side norm | 3 | $\min(m,n)$ | $r(m{+}n){+}m$ | $GL_r$ | Mon_out×GL_in | M1 | V | L |
| PiSSA | + (split) | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | Fr($W_0$ top) | M1 | S | L |
| MiLoRA | + (split) | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | Fr($W_0$ bottom) | M1 | S | L |
| OLoRA | + (split) | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | Fr($W_0$ QR) | M1 | S | L |
| LoRA-GA | + (split) | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | Fr(grad) | M1 | S | L |
| EVA | + | ed | 2 | $r_\ell$ (redistributed) | $r_\ell(m{+}n)$ | $GL_r$ | Fr(act) | M1 | V | L |
| CorDA | + (split) | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | Fr($W_0$·cov) | M1 | S | L |
| LoftQ | + (quantised split) | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | Fr($W_0{-}Q$) | Mq | S | L |
| QLoRA | + (quantised base) | ed | 2 | $r$ | $r(m{+}n)$ | $GL_r$ | GL | Mq | V | L |
| AdaLoRA | + | P–λ–Q (spider core) | 3 | $\le r$ (budgeted) | $r(m{+}n){+}r$ | torus $(\mathbb R^*)^{2r}$; total waste $r^2{+}r$; fixed by penalty | ≈O | M1 | V | L* |
| DyLoRA | + | ed with flag | 2 | $b\le r$ | $r(m{+}n)$ | $(\mathbb R^*)^r$ | GL | M1 | V | L |
| ReLoRA | + iterated, with merges | ed per phase | 2 | $Kr$ | $r(m{+}n)$ | $GL_r$ per phase | GL | M1 | V per phase | L |
| VeRA | + | spider–B̄–spider–Ā (frozen, shared) | 2 | $r$ | $m{+}r$ | $\mathbb R^*$ | Fr(rand) | M1 | V | L |
| NOLA | + | dc·dc | 2 | $r$ | $k{+}l$ | $\mathbb R^*$ | Fr(rand) | M1 | V | L |
| VB-LoRA | + | dc with shared bank, top-$k$ | 2 (bank) | $r$ | global bank + logits | discrete | Fr(bank) | M1 | V | N |
| LoRA-XS | + | Fr–R–Fr | 1 | $r$ | $r^2$ | — | Fr($W_0$) | M1 | lin | L |
| FourierFT | + | cj: Re(DFT·S_Ω·DFT) | 1 | $\le2n_c$ | $n_c$ | — | Π_cyc | M1 | lin | L |
| LoHa | + | hd | 4 | $\min(m,n,r_1r_2)$ | $(r_1{+}r_2)(m{+}n)$ | $GL_{r_1}{\times}GL_{r_2}{\times}\mathbb R^*$ | Mon | M1 | V | L |
| LoKr | + | kr: $C\otimes BA$ | 3 | $\operatorname{rk}C\cdot r$ | $m_1n_1{+}r(m_2{+}n_2)$ | $GL_r{\times}\mathbb R^*$ | Π_⊗ | M1 | V | L |
| KronA | + | kr: $A\otimes B$ | 2 | $\operatorname{rk}A\operatorname{rk}B$ | $a_1a_2{+}b_1b_2$ | $\mathbb R^*$ | Π_⊗ | M1 | V | L |
| TT adapters (LoRETTA_rep, FacT-TT) | + | ch (MPO) | #cores | up to full | $\sum r_{k-1}m_kn_kr_k$ | $\prod GL(r_k)$ | Π_⊗ | M1 | V | L (FacT: N) |
| MoRA | + | Fr–M–Fr (share) / $I\otimes M$ with frozen rotations (reshape) | 1 | $\hat r$ / full | $\hat r^2$, $\hat r=\lfloor\sqrt{(m{+}n)r}\rfloor$ | — | Π_blk | M1 | lin | L |
| HiRA | ⊙ | hd with $W_0$ | 2 | $r_0r$ | $r(m{+}n)$ | $GL_r$ | Mon | M1 | V | L |
| OFT | ∘ | pt: block Cayley | 1 in $R$ | $r_0$ | $k\binom{b}{2}$ | — | Π_blk | M1 | S | L |
| BOFT | ∘ | bf: $m'$ butterfly factors | $m'$ | $r_0$ | $m'\frac{n}{b}\binom b2$ | discrete | Π_⊗ (hypercube) | M1 | S | L |
| HRA | ∘ | rf: $r$ reflections | $r$ | $r$ | $rn$ | $(\mathbb R^*)^r$ (scale of each $u_i$) | O | M1 | S (paired) | L |
| Givens / qGOFT | ∘ | bf: 2-wire gates | #gates | $2\cdot$#gates | $O(d)$ | — | Π_pair | M1 | S | L |
| (IA)³ | ∘ | pt (spider with state) | 1 | $r_0$ | $m$ per site | — | Mon | M1→ | S | L |
| SSF | ∘ (+ shift) | pt | 1 | $r_0$ | $2d$ per site | — | Mon | M1→ | S | L |
| BitFit | ι | coordinate subspace | 1 | n/a (affine) | #biases | — | Mon | M0 | lin | L |
| Diff pruning | ι | mask·values | 2 | $\le k$ | $\le k$ stored | — | Π_perm | M0 | V (sparse variety) | L |
| FISH mask | ι | fixed coordinate subspace | 1 | $\le k$ | $k$ | — | Π_perm | M0 | lin | L |
| LayerNorm tuning | ι (= ∘ spider) | pt | 1 | — | $2d$ per LN | — | Mon | M0 / M1→ | lin | L |
| LISA | ι, scheduled | layer subsets | 1 per phase | full on active layers | γ layers | — | GL | M0 | lin | L |

\*AdaLoRA's budget allocation across layers is global.

**Table B: activation-side, sequence, cotangent and meta methods.**

| Method | Coupling | Diagrammatic form | Merge | Key structural fact |
|---|---|---|---|---|
| Houlsby / Pfeiffer | σ | serial bottleneck $x+U\sigma(Dx)$, ×2 or ×1 per layer | M∞ | gauge is perm ⋉ $(\mathbb R_{>0})^r$ under ReLU, perm only under GELU (C6) |
| Parallel adapter | σ | $W_0x+sU\sigma(Dx)$ | M∞ | with σ = id it *equals* LoRA (E1) |
| Compacter | σ | Houlsby adapter with PHM boxes $\sum A_i\otimes B_i$ (shared $A_i$) | M∞ | Kronecker cut (Thm 12); non-local |
| AdapterFusion | σ+Σ | attention over $N$ adapter outputs | M∞ | input-dependent convex mixing |
| LST | σ | side network fed by copies of backbone activations | M∞ | backprop cone avoids the backbone (Prop. 24) |
| LLaMA-Adapter | ⊗ | $K$ prompts in top $L$ layers, separate softmax, zero gate | M∞ | content attention exactly unchanged (Thm 21c) |
| Prompt tuning | ⊗ | state $p\in X^{\ell}$ at layer 0 | M∞ | factors through deep prefix in decoders (Thm 21d) |
| Prefix tuning | ⊗ | K/V states at each layer, plus a parameter-side MLP | M∞ | gated parallel adapter (Thm 21a) |
| P-tuning v2 | ⊗ | deep prefix with free per-layer K/V | M∞ | largest of the prefix-type gadgets |
| GaLore | ∇ | per-period frozen frame $P$ | M0 | equals one-sided LoRA within a period (Thm 19c) |
| MeZO | ∇ | random rank-1 cotangent line per step, forward-only | — | empty backprop cone |
| Task arithmetic | Σ | sum in the abelian group Hom | M1 | gauge-invariant; ⊕ |
| TIES / DARE | Σ | coordinatewise trim/elect/rescale | M1 | natural only for signed permutations of the spider |
| LoRAHub | Σ | factor-wise quadratic mixing | M1 | **gauge-dependent** (Thm 22b) |
| MoLE | Σ+σ | input-dependent gates over LoRA branches | M∞ | gauge-invariant |
| S-LoRA | Σ | $I_{\rm batch}\otimes W_0+\sum_i\lvert i\rangle\langle i\rvert\otimes B_iA_i$ | deliberately unmerged | multiplexer on the batch leg (E15) |
| Text-to-LoRA | Σ | hypernetwork into the factor slots | M1 (output) | factor-level targets are gauge-sensitive |
| UniPELT / MAM | ⊕ | sums and placements of primitives | mixed | design spaces are ⊕-words plus coproducts |

---

## 4. Equivalences and obstructions

**Equivalences** (each is a diagram equality, a gadget isomorphism, or an equality of images).

- **E1.** A linear parallel adapter is LoRA: $W_0x+sUDx$ gives $\Delta W=sUD$ (He et al.).
- **E2.** A linear serial adapter $(I+UD)W_0$ is a $W_0$-coupled LoRA $\{U(DW_0)\}\subseteq\mathrm{LoRA}_r$. Its update set equals LoRA's iff $W_0$ has full column rank, and the two gadgets are isomorphic iff $W_0$ is invertible.
- **E3.** $\mathrm{LoRA}_r\oplus\mathrm{LoRA}_s\cong\mathrm{LoRA}_{r+s}$. Task arithmetic of $N$ LoRAs lies in $\mathrm{LoRA}_{Nr}$, and ReLoRA's $K$ phases lie in $\mathrm{LoRA}_{Kr}$.
- **E4.** Kronecker-sum gadgets are LoRA across the cut $\{U_1V_1\}\mid\{U_2V_2\}$. LoKr is Kronecker rank 1 with a LoRA inside one factor. TT-matrix extends this to more legs. MoRA's reshape variant is $I\otimes M$ with frozen chunk rotations: a degree-1 LoKr whose first factor is the identity.
- **E5.** $\mathrm{LoHa}_{r_1,r_2}\to\mathrm{LoRA}_{r_1r_2}$ (face-splitting), and $\mathrm{LoHa}_{1,r}\simeq\mathrm{LoRA}_r$ in image.
- **E6.** (IA)³ ⊂ $\mathrm{HiRA}_1$; two-sided scaling ⊂ $\mathrm{HiRA}_2$; and $\mathrm{HiRA}_r$ is an $r$-fold sum of two-sided scalings.
- **E7.** DoRA ≃ (IA)³_out ⊕ $\mathrm{LoRA}_r$, image-level and generic.
- **E8.** OFT ▷ LoRA ≅ LoRA ▷ OFT ≅ OFT ⊕ LoRA, but (IA)³ ▷ OFT ≠ OFT ▷ (IA)³.
- **E9.** $\mathrm{HRA}_r=\mathrm{LoRA}_r\times\mathrm{SO}$; a Givens circuit with $g$ gates lies in $\mathrm{LoRA}_{2g}\times\mathrm{SO}$.
- **E10.** LoRA-XS $=\mathrm{FA}(A_0)\times\mathrm{FB}(B_0)$. The tangent space at a PiSSA initialisation is $\mathrm{Upd}(\mathrm{FA}\oplus\mathrm{FB})$.
- **E11.** Prefix tuning is a gated parallel adapter. LLaMA-Adapter is its separately normalised version. In decoders, prompt tuning embeds in deep prefix tuning, which is P-tuning v2.
- **E12.** GaLore within a period is one-sided LoRA. LoRA at the vertex is a random-projection compressor (Flora). MeZO is a random rank-1 cotangent restriction.
- **E13.** VeRA, NOLA, VB-LoRA, LoRA-XS, LoRA-FA and the PiSSA family all simulate into $\mathrm{LoRA}_r$. They differ only in their frozen boxes, sharing pattern and base point.
- **E14.** FourierFT and SAID (Aghajanyan et al.'s Fastfood random subspace) are both degree-1 frozen-dictionary conjugation sandwiches. They differ in the dictionary and in locality.
- **E15.** S-LoRA batching uses Ab-enrichment, $(W_0+\Delta_i)x_i=W_0x_i+\Delta_ix_i$, to keep the multiplexer on the small box. Merging would materialise $\sum_i\lvert i\rangle\langle i\rvert\otimes(W_0+\Delta_i)$, which means $N$ full copies.

**Obstructions.**

- **O1, nonlinearity.** A non-affine box on the activation path blocks merging (Thm 20c).
- **O2, cut.** $\operatorname{rk}\le\rho$, and different cuts are incomparable (Thms 11–12).
- **O3, volume.** $\dim\mathrm{Upd}\le\lvert D\rvert-\dim\text{gauge}$, and high rank does not imply high dimension (Thm 13c, Prop. 16).
- **O4, symmetry.** A method whose frozen structure is invariant only under $H\subsetneq\mathrm{Aut}$ (Lemma 1.1) is not equivariant under function-preserving neuron permutations. This covers FourierFT (cyclic order), LoKr, KronA, BOFT and TT (tensor factorisation), block OFT and MoRA (blocks), and Givens (pairing).
  - LoRA, DoRA, (IA)³, BitFit, LayerNorm tuning, diff pruning, FISH and HRA are equivariant. VeRA and NOLA are equivariant in distribution.
  - **[Pred]** Permuting hidden units of a pretrained model leaves LoRA's results unchanged up to seed variation, but changes FourierFT, LoKr, BOFT and MoRA.
  - TIES and DARE are natural only for signed permutations, so they should be applied in privileged coordinates, or after gauge-fixing gauges such as $W_Q\mapsto gW_Q$, $W_K\mapsto g^{-\top}W_K$.
- **O5, gauge.** Factor-wise combination is ill-defined without a joint gauge fixing (Thm 22).
- **O6, pattern rigidity.** Prefix-type methods cannot change relative content attention (Thm 21b).
- **O7, type.** Quantised boxes are not closed under fusion (Thm 20d).
- **O8, bidirectionality.** In encoders, prompt tuning does not factor through prefix tuning (Thm 21d).
- **O9, base point.** Zero-factor initialisations halve the first-order directions available (Thm 17). This is dynamical, not expressive.

---

## 5. Rosetta stone

| PEFT concept | Categorical / diagrammatic concept |
|---|---|
| frozen weight $W_0$ / the adapted layer | frozen box / a hole of type $V\to U$ in a $\mathbf{Para}$ diagram |
| a PEFT method on a layer | a gadget: a tensor network with a frozen/trainable labelling plugged into the hole |
| trainable parameters, parameter count | trainable boxes, total box size $\lvert D\rvert$ |
| what a method can express | image of the contraction map (a definable set) |
| "method A can do whatever B can" | morphism (simulation) in the pointed slice $\mathbf{Gad}(W_0)$ |
| frozen model / full fine-tuning | initial / terminal object |
| rank of ΔW | min-cut weight of the network |
| "high-rank" PEFT | a network whose min-cut avoids the bond |
| Kronecker product, Kronecker rank | $\otimes$ of morphisms under a reshape; operator-Schmidt rank across $\{U_1V_1\}\mid\{U_2V_2\}$ |
| Hadamard product | copy and merge spiders of the neuron-basis Frobenius algebra |
| diagonal scaling ((IA)³, SSF, LN γ, DoRA magnitude) | spider with a state |
| DFT, random bases, SVD frames | frozen isomorphisms and frozen dictionaries |
| LoRA's $(Bg^{-1},gA)$ redundancy | gauge $GL(r)$ on an internal wire |
| balancedness, LoRA+ ratio | Noether charge of the wire gauge, and the metric weighting it |
| transformation-invariant optimiser | dynamics that descend to the quotient by the gauge |
| zero-B vs PiSSA-type initialisation | base point at the singular vertex vs at a smooth point |
| splitting $W_0=W_{\rm res}+B_0A_0$ | rewriting the $W_0$ box as a biproduct sum, then labelling one summand trainable |
| merge / zero inference overhead | rewriting the activation subdiagram into a single linear box |
| nonlinear adapter | σ-box on an activation wire, which obstructs fusion |
| Cayley, normalisation, prefix MLP | parameter-side (vertical) boxes; compile-time |
| UniPELT, task arithmetic | $\oplus$ (biproduct): Minkowski sum of update sets |
| stacking multiplicative methods | serial composition $\rhd$, a non-commutative monoid |
| "both a LoRA and an OFT" (HRA); LoRA-XS | categorical product (pullback over Hom) |
| choosing a method or configuration | coproduct (wedge) |
| prompt, prefix | a state tensored into the sequence wire |
| softmax attention | monoid homomorphism followed by a perspective map |
| GaLore, gradient projection | restriction of the backward map of the lens; equals degree-1 forward restriction |
| LST memory savings | a backprop cone that avoids the backbone |
| cross-layer sharing (VB-LoRA, FacT, hypernetworks) | parameter-side copying: a non-local sharing pattern |
| multi-adapter serving | a multiplexer (controlled box) on the batch leg |
| TIES, DARE | operations natural only for automorphisms of the neuron spider |
| neuron-permutation symmetry | automorphisms of the σ-spider (Lemma 1.1) |
| quantisation | changing the box type to a non-additive object |
| rank schedules (AdaLoRA, DyLoRA) | a filtration (flag) on the bond, which cuts the gauge down to a torus |
| ReLoRA and GaLore periods | iterated rewrite: train, fuse, re-open the hole |

---

## 6. What is analogy rather than theorem, and what is open

**Analogies.**

- **[Analogy]** "Noether" under Adam. The conservation laws are exact only for continuous-time flow with block-scalar preconditioning. Under Adam they are a heuristic for *where the symmetry pushes*, not a law.
- **[Analogy]** Quantum-circuit language for butterflies ("multiplexed gates") and "entanglement across a cut". The cut-rank mathematics is real; the physics vocabulary is only intuition.
- **[Analogy]** The Erlangen classification by symmetry group classifies *update classes*. It does not say which symmetry suits a task.
- **[Analogy]** "Tangent/cotangent duality" as a general principle. The degree-1 collapse (Thm 19) is a theorem; a general duality between forward and backward methods is not.

**Blind spots of the image-level category.** rsLoRA, LoRA+, PiSSA, LoRA-GA, EVA and LoRA share an image up to base point, so $\mathbf{Gad}$'s preorder cannot rank them. Their differences live in the **fibre**: metric, base point, and choice of gauge section. Claiming that "category theory explains why PiSSA beats LoRA" would be false. What the framework does supply is the vertex/smooth distinction (Thm 17) and the Noether constraints.

**Limits of the mergeability results.** We prove only *box-local* and *neighbour* mergeability. Global absorption elsewhere in the network is not addressed.

**Open problems.**

1. Is $\mathrm{Upd}(\mathrm{LoHa}_{r_1,r_2})$ closed for $r_1,r_2\ge2$? Loop networks can fail closedness (Landsberg–Qi–Ye; de Silva–Lim for CP), and non-closedness would predict diverging factors.
2. What is the exact dimension of Hadamard-diamond images?
3. Describe the size-$k$ frontier of $\mathbf{Gad}$: its maximal elements are incomparable (Thm 12).
4. When does $\rhd$ commute?
5. Design frozen dictionaries to maximise the expected gradient capture $\mathbb E\lVert P_{\operatorname{im}\mathcal L}G\rVert^2$ on the Grassmannian.

---

## 7. Amendments to the seed sketch

- **S1/S3.** Adopted, and unified by the Observation in §1.5: reparametrisation and extension coincide exactly when the activation side is linear.
- **S2.** Adopted with $\mathbf{Def}$ as the ambient category. Definable choice makes the preorder equal image inclusion, and products and coproducts are intersections and unions.
- **S4.** Refined: the privileged group is *derived* from σ (Lemma 1.1).
- **S5.** Generalised to every internal wire, including constrained and frozen endpoints.
- **S6.** Strengthened: generic attainment of the LoHa bound, plus the volume caveat.
- **S7.** Corrected: $\oplus$ is not the product; distributivity and non-commutativity are made precise.
- **S8.** Recast: base points sit at the vertex or at a smooth point.
- **S9.** Sharpened: the distinction collapses at degree 1, and GaLore is one-sided LoRA (credited).
- **S10.** Derived from a single homomorphism, and extended with LLaMA-Adapter and the prompt-into-prefix embedding.
- **S11.** Corrected: non-local methods are not generator-wise.

---

## 8. Interactive visualisations

All of these are complete at rest, compute every number they display, and use the site's spec tokens.

1. **Plug Workbench** (Defs 1–2; Thm 11; Prop. 16).
   - *Manipulate:* drag trainable boxes, frozen boxes, spiders, reshape isos, DFT and dictionary boxes into the hole $V\to U$.
   - *See:* live $\lvert D\rvert$, degree, the minimum cut highlighted with its weight ρ, gauge wires glowing (trainable–trainable edges), $\dim\mathrm{Upd}$ (numerical Jacobian rank at small sizes), the merge verdict, and the nearest named method.
   - *Default:* LoRA.
2. **Cut the Tensor** (Thm 12).
   - *Manipulate:* a 16×16 weight drawn as a 4-leg square ($s=4$); choose one of three bipartitions; pick a preset Δ (random rank-1, $I$, $C\otimes D$, LoHa, HiRA).
   - *See:* singular-value bars across each cut. A rank-1 update lights up all 16 Kronecker singular values; the identity has one.
3. **LoRA Starts at the Apex** (Thms 17, 18a).
   - *Manipulate:* sliders for $m,n,r$, and for $\operatorname{rk}A_0,\operatorname{rk}B_0$.
   - *See:* a Venn diagram of $\{XA_0\}$ and $\{B_0Y\}$ whose overlap is labelled LoRA-XS with dimension $r^2$; the counts $mr$ against $r(m+n-r)$; a small cone of symmetric rank-1 2×2 matrices (labelled as a symmetric slice) with first-order arrows at the apex and at a smooth point.
4. **Noether Hyperbolas** (Thm 14, C1).
   - *Manipulate:* scalar LoRA on $\tfrac12(w_0+ba-w^*)^2$; the ratio λ = η_B/η_A; the initial $a_0$; step size; an Adam toggle.
   - *See:* trajectories on the hyperbolas $a^2/\eta_A-b^2/\eta_B=\text{const}$, a meter for the conserved value, its $O(\eta)$ drift, and the breaking under Adam.
5. **The LoRAHub Trap** (Thm 22).
   - *Manipulate:* a $GL_2$ knob acting on module 2 (polar parameters).
   - *See:* heatmaps of $B_2$ and $A_2$ churning while $B_2A_2$ stays fixed; a task-arithmetic panel that stays still (moss); a LoRAHub panel that moves (seal) with a ‖change‖ meter; a "share frozen A" button that turns it green.
6. **Merge Game** (Thm 20).
   - *Manipulate:* a pre-LN transformer block with a GELU / ReLU / SwiGLU toggle; drop (IA)³, SSF, LayerNorm tuning, LoRA, an adapter or a prefix onto a wire; press *rewrite*.
   - *See:* rules R1–R6 animated step by step, ending either fused into the highlighted weights or blocked at the offending σ or softmax box.
7. **Prefix Is a Gate** (Thm 21).
   - *Manipulate:* one query and prefix logit strength; a LLaMA-Adapter mode with separate softmax and gate $g$.
   - *See:* content-attention bars before and after, with their ratio line flat (rigidity); the output vector $(1-\lambda)a_C+\lambda a_P$ inside the convex hull of the prefix values. In LLaMA-Adapter mode the content bars do not move.
8. **Periodic Table** (§3).
   - *Layout:* rows are couplings (+, ∘, ⊙, ι, σ, ⊗, ∇, Σ), columns are shapes, and cells are method glyphs.
   - *Manipulate:* hover for axis values; filter by merge, gauge or symmetry; click to open in the Workbench.
9. **Rank Is Not Volume** (Thm 13; Prop. 16).
   - *Manipulate:* sliders for $m,n$ and the budget.
   - *See:* a scatter of ρ against $\dim\mathrm{Upd}$ for LoRA, LoHa, LoKr, KronA, HiRA, MoRA, FourierFT and TT, all computed from formulas. LoHa sits high on ρ and low on dimension.
10. **Permutation Test** (O4).
    - *Manipulate:* a "permute hidden units" button (function-preserving).
    - *See:* support and frame patterns of each method's update class. LoRA and (IA)³ are carried along (✓); FourierFT's cyclic structure, LoKr's tensor structure and block-OFT's blocks break (✗).
11. **Peeling Reflections** (Thm 18b).
    - *Manipulate:* a random $R\in SO(4)$ with $\operatorname{rk}(R-I)=k$; press *peel*.
    - *See:* one Householder reflection removed at a time, the fixed subspace growing, and the count ending at $k$. The reverse view composes HRA vectors and shows $\operatorname{rk}(R-I)\le r$.

---

## 9. Key references

*Category theory and diagrams.*

- Joyal, Street. The geometry of tensor calculus I. Adv. Math. 88 (1991).
- Kelly, Laplaza. Coherence for compact closed categories. JPAA 19 (1980).
- Selinger. A survey of graphical languages for monoidal categories. arXiv:0908.3347.
- Coecke, Pavlovic, Vicary. A new description of orthogonal bases. MSCS (2013); arXiv:0810.0812.
- Coecke, Kissinger. *Picturing Quantum Processes*. CUP (2017).
- Fong, Spivak, Tuyéras. Backprop as Functor. arXiv:1711.10455.
- Cruttwell, Gavranović, Ghani, Wilson, Zanasi. Categorical Foundations of Gradient-Based Learning. arXiv:2103.01931.
- Gavranović. Fundamental Components of Deep Learning: A category-theoretic approach (thesis). arXiv:2403.13001.

*Tensor networks, tensor geometry, o-minimality.*

- Biamonte, Bergholm. Tensor Networks in a Nutshell. arXiv:1708.00006.
- Landsberg, Qi, Ye. On the geometry of tensor network states. arXiv:1105.4449.
- Ye, Lim. Tensor network ranks. arXiv:1801.02662.
- de Silva, Lim. Tensor rank and the ill-posedness of the best low-rank approximation problem. arXiv:math/0607647.
- Oseledets. Tensor-Train decomposition. SIAM J. Sci. Comput. 33 (2011).
- Holtz, Rohwedder, Schneider. On manifolds of tensors of fixed TT-rank. Numer. Math. 120 (2012).
- Uschmajew, Vandereycken. The geometry of algorithms using hierarchical tensors. Linear Algebra Appl. 439 (2013).
- Van Loan, Pitsianis. Approximation with Kronecker products (1993).
- van den Dries. *Tame Topology and O-minimal Structures*. CUP (1998).
- Wilkie. Model completeness results for expansions of the ordered field of real numbers by restricted Pfaffian functions and the exponential function. JAMS (1996).
- Dao et al. Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations. arXiv:1903.05895.

*Symmetry and dynamics.*

- Du, Hu, Lee. Algorithmic regularization in learning deep homogeneous models. arXiv:1806.00900.
- Arora, Cohen, Hazan. On the optimization of deep networks: implicit acceleration by overparameterization. arXiv:1802.06509.
- Kunin et al. Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics. arXiv:2012.04728.
- Zhao, Ganev, Walters, Yu, Dehmamy. Symmetries, flat minima, and the conserved quantities of gradient flow. arXiv:2210.17216.
- Tong, Ma, Chi. Accelerating ill-conditioned low-rank matrix estimation via scaled gradient descent. arXiv:2005.08898.

*LoRA family.*

- Hu et al. LoRA. arXiv:2106.09685.
- Kalajdzievski. rsLoRA. arXiv:2312.03732.
- Hayou, Ghosh, Yu. LoRA+. arXiv:2402.12354.
- Zhang et al. LoRA-FA. arXiv:2308.03303.
- Liu et al. DoRA. arXiv:2402.09353.
- Meng, Wang, Zhang. PiSSA. arXiv:2404.02948.
- Wang et al. MiLoRA. arXiv:2406.09044.
- Büyükakyüz. OLoRA. arXiv:2406.01775.
- Wang, Yu, Li. LoRA-GA. arXiv:2407.05000.
- Paischer et al. EVA. arXiv:2410.07170.
- Yang et al. CorDA. arXiv:2406.05223.
- Li et al. LoftQ. arXiv:2310.08659.
- Dettmers et al. QLoRA. arXiv:2305.14314.
- Zhang et al. AdaLoRA. arXiv:2303.10512.
- Valipour et al. DyLoRA. arXiv:2210.07558.
- Lialin et al. ReLoRA. arXiv:2307.05695.
- Kopiczko et al. VeRA. arXiv:2310.11454.
- Koohpayegani et al. NOLA. arXiv:2310.02556.
- Li, Han, Ji. VB-LoRA. arXiv:2405.15179.
- Bałazy et al. LoRA-XS. arXiv:2405.17604.
- Gao et al. FourierFT. arXiv:2405.03003.
- Hyeon-Woo et al. FedPara (LoHa). arXiv:2108.06098.
- Yeh et al. LyCORIS (LoHa/LoKr). arXiv:2309.14859.
- Edalati et al. KronA. arXiv:2212.10650.
- Zhang et al. PHM layers. arXiv:2102.08597.
- Mahabadi, Henderson, Ruder. Compacter. arXiv:2106.04647.
- Yang et al. LoRETTA. arXiv:2402.11417.
- Jie, Deng. FacT. arXiv:2212.03145.
- Jiang et al. MoRA. arXiv:2405.12130.
- Huang et al. HiRA. ICLR 2025.
- Zhang, Pilanci. Riemannian Preconditioned LoRA. arXiv:2402.02347.
- Yen et al. LoRA-RITE. arXiv:2410.20625.
- Zeng, Lee. The Expressive Power of Low-Rank Adaptation. arXiv:2310.17513.

*Orthogonal and scaling methods.*

- Qiu et al. OFT. arXiv:2306.07280.
- Liu et al. BOFT. arXiv:2311.06243.
- Yuan et al. HRA. arXiv:2405.17484.
- Ma et al. qGOFT. arXiv:2404.04316.
- Liu et al. (IA)³. arXiv:2205.05638.
- Lian et al. SSF. arXiv:2210.08823.

*Selective methods.*

- Ben Zaken et al. BitFit. arXiv:2106.10199.
- Guo, Rush, Kim. Diff pruning. arXiv:2012.07463.
- Sung, Nair, Raffel. FISH mask. arXiv:2111.09839.
- Zhao et al. LayerNorm tuning. arXiv:2312.11420.
- Pan et al. LISA. arXiv:2403.17919.
- Aghajanyan et al. Intrinsic dimensionality. arXiv:2012.13255.
- Li et al. Measuring the intrinsic dimension. arXiv:1804.08838.

*Adapters and prompts.*

- Houlsby et al. arXiv:1902.00751.
- Pfeiffer et al. AdapterFusion. arXiv:2005.00247.
- He et al. Towards a Unified View of PETL. arXiv:2110.04366.
- Mao et al. UniPELT. arXiv:2110.07577.
- Chen et al. PEFT Design Spaces. arXiv:2301.01821.
- Sung, Cho, Bansal. LST. arXiv:2206.06522.
- Zhang et al. LLaMA-Adapter. arXiv:2303.16199.
- Lester, Al-Rfou, Constant. Prompt tuning. arXiv:2104.08691.
- Li, Liang. Prefix-tuning. arXiv:2101.00190.
- Liu et al. P-tuning v2. arXiv:2110.07602.
- Wang et al. Universality and Limitations of Prompt Tuning. arXiv:2305.18787.
- Petrov, Torr, Bibi. When Do Prompting and Prefix-Tuning Work? arXiv:2310.19698.

*Optimiser-side methods.*

- Zhao et al. GaLore. arXiv:2403.03507.
- Hao, Cao, Mou. Flora. arXiv:2402.03293.
- Torroba-Hennigen, Lang, Guo, Kim. On the Duality between Gradient Transformations and Adapters. arXiv:2502.13811.
- Malladi et al. MeZO. arXiv:2305.17333.

*Merging, routing, serving.*

- Ilharco et al. Task arithmetic. arXiv:2212.04089.
- Yadav et al. TIES-Merging. arXiv:2306.01708.
- Yu et al. DARE. arXiv:2311.03099.
- Huang et al. LoraHub. arXiv:2307.13269.
- Stoica et al. KnOTS. arXiv:2410.19735.
- Sun et al. FFA-LoRA. arXiv:2403.12313.
- Wu, Huang, Wei. Mixture of LoRA Experts. arXiv:2404.13628.
- Sheng et al. S-LoRA. arXiv:2311.03285.
- Charakorn et al. Text-to-LoRA. arXiv:2506.06105.
