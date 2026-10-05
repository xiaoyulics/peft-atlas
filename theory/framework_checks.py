"""Numerical checks for theory/framework.md (float64 torch autograd).
Run: python3 theory/framework_checks.py   (about 1 minute on a laptop)
Checks: HRA apex (paired-init Jacobian rank), Adam gauge group B_r and the joint torus,
LoRA+ == alpha*lambda under Adam eps=0, GraLoRA and LoHa dimensions, weight-decay charge
conventions, descending cometrics (ScaledGD, LoRA-Pro), VeRA rebasing span, RoAd orbit
dimension, first-order rank formula, re-HRA rank subadditivity.
Section 12 (referee round): DoRA dimension caps, GraLoRA rank, LoHa cap, LoReFT and serial-adapter
images, FourierFT conjugate pairs, HF Cayley-Neumann OFT, BOFT signs, DARE/TIES groups, covariant splits,
Adam joint torus with eps, LoRA-Pro equivariance, LoRA strata and HF 'orthogonal' init, Obs I.7 / Prop I.8,
HiRA rebasing, Kronecker ranks, butterfly components, weight-decay slices, zero-B rule, Cayley-Neumann Lie algebra.
Section 13 (second referee round): S_1 of sums (Obs I.7), HF orthogonal init (Thm II.2), BOFT/GOFT/Cayley-Neumann
(Cor II.6), Hurwitz moves (Thm II.7), DoRA with a zero magnitude (Prop II.8), LoKr caps (Thm II.11), LoHa vs LoRA_2r and
the LoHa orbit, HRA at r >= n-1 (Prop II.12), transport and rebasing (Prop II.13), bottom-r and damped splits (Prop II.16),
per-channel charges and the GD step of Phi (Thm III.5), Adam drift of Phi (Cor III.6), GaLore brackets and Bures-Wasserstein
(Prop IV.3), NF4 defects (Prop V.2), Cayley-Neumann contraction (Thm V.3), factor-wise DARE, LoraHub and Poly fibres (Prop V.4),
RoPE cup slide (Thm VI.3), serial adapter at m < n (Prop VI.4)."""
import torch, numpy as np
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from torch.autograd.functional import jacobian

def jrank(f, q, tol=1e-9):
    J = jacobian(f, q)
    J = J.reshape(-1, q.numel())
    s = torch.linalg.svdvals(J)
    return int((s > tol * s[0]).sum())

print("== 1. HRA paired init vs generic ==")
def hra(n, r, W0):
    def f(u):
        U = u.reshape(r, n)
        R = torch.eye(n)
        for i in range(r):
            v = U[i]
            R = R @ (torch.eye(n) - 2 * torch.outer(v, v) / (v @ v))
        return W0 @ R
    return f
for n, r in [(9, 2), (9, 4), (10, 2), (10, 4), (12, 6)]:
    W0 = torch.randn(n + 2, n)  # injective
    ug = torch.randn(r * n)
    k = r // 2
    base = torch.randn(k, n)
    up = torch.cat([base[i // 2:i // 2 + 1] for i in range(r)], 0).reshape(-1)
    print(n, r, "generic", jrank(hra(n, r, W0), ug), "pred", r * n - r * (r + 1) // 2,
          "| paired", jrank(hra(n, r, W0), up), "pred", k * n - k * (k + 1) // 2)

print("== 2. Adam first step equivariance (sign descent) ==")
m, n, r = 6, 5, 3
B = torch.randn(m, r); A = torch.randn(r, n); G = torch.randn(m, n)
def adam_first_dW(B, A, etaB=1.0, etaA=1.0):
    gB = G @ A.T; gA = B.T @ G
    dB = -etaB * torch.sign(gB); dA = -etaA * torch.sign(gA)
    return dB @ A + B @ dA  # first-order dW
def adam_first_dW_vec(B, A, etaB_vec, etaA_vec):
    gB = G @ A.T; gA = B.T @ G
    dB = -torch.sign(gB) * etaB_vec[None, :]; dA = -torch.sign(gA) * etaA_vec[:, None]
    return dB @ A + B @ dA
base = adam_first_dW(B, A)
P = torch.eye(r)[[2, 0, 1]] * torch.tensor([1., -1., 1.])[:, None]
D = torch.diag(torch.tensor([2.0, 0.5, 3.0]))
O, _ = torch.linalg.qr(torch.randn(r, r))
for name, g in [("signed perm", P), ("diag torus", D), ("rotation", O)]:
    Bg, Ag = B @ g, torch.linalg.inv(g) @ A
    print(name, "defect", float((adam_first_dW(Bg, Ag) - base).norm()))
d = torch.diagonal(D)
Bg, Ag = B @ D, torch.linalg.inv(D) @ A
print("diag torus with per-channel lr (etaB*d, etaA/d) defect",
      float((adam_first_dW_vec(Bg, Ag, d, 1 / d) - base).norm()))

print("== 3. LoRA+ == alpha*lambda under Adam eps=0 (full trajectories) ==")
def adam_run(s, etaA, etaB, A0, steps=200, eps=0.0):
    m_, n_ = 7, 6; r_ = 2
    torch.manual_seed(1)
    W0 = torch.randn(m_, n_); X = torch.randn(n_, 20); Y = torch.randn(m_, 20)
    B = torch.zeros(m_, r_); A = A0.clone()
    mB = torch.zeros_like(B); vB = torch.zeros_like(B); mA = torch.zeros_like(A); vA = torch.zeros_like(A)
    b1, b2 = 0.9, 0.999
    traj = []
    for t in range(1, steps + 1):
        W = W0 + s * B @ A
        Gr = (W @ X - Y) @ X.T / 20
        gB = s * Gr @ A.T; gA = s * B.T @ Gr
        mB = b1 * mB + (1 - b1) * gB; vB = b2 * vB + (1 - b2) * gB ** 2
        mA = b1 * mA + (1 - b1) * gA; vA = b2 * vA + (1 - b2) * gA ** 2
        def step(mm, vv):
            mh = mm / (1 - b1 ** t); vh = vv / (1 - b2 ** t)
            den = vh.sqrt() + eps
            out = torch.where(den > 0, mh / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(mh))
            return out
        B = B - etaB * step(mB, vB); A = A - etaA * step(mA, vA)
        traj.append((W0 + s * B @ A).clone())
    return torch.stack(traj)
torch.manual_seed(2); A0 = torch.randn(2, 6) * 0.3
lam = 8.0; s = 0.5; etaA = 1e-3
t1 = adam_run(s, etaA, lam * etaA, A0)
t2 = adam_run(lam * s, etaA, etaA, A0)
t3 = adam_run(s, etaA, etaA, A0)
print("LoRA+ vs alpha*lam: max diff", float((t1 - t2).abs().max()), "| control (no rescale):", float((t1 - t3).abs().max()))

print("== 4. GraLoRA dimension ==")
def gralora(m, n, r, k):
    def f(q):
        bm, bn, br = m // k, n // k, r // k
        idx = 0; rows = []
        for i in range(k):
            row = []
            for j in range(k):
                Bij = q[idx:idx + bm * br].reshape(bm, br); idx += bm * br
                Aij = q[idx:idx + br * bn].reshape(br, bn); idx += br * bn
                row.append(Bij @ Aij)
            rows.append(torch.cat(row, 1))
        return torch.cat(rows, 0)
    return f
for (m, n, r, k) in [(8, 8, 4, 2), (12, 8, 4, 2), (12, 12, 6, 3)]:
    npar = r * (m + n)
    q = torch.randn(npar)
    print((m, n, r, k), "rank", jrank(gralora(m, n, r, k), q), "pred r(m+n-r)", r * (m + n - r),
          "| max rank of dW", int(torch.linalg.matrix_rank(gralora(m, n, r, k)(q))), "LoRA rank cap", r)

print("== 5. LoHa dimension ==")
def loha(m, n, r):
    def f(q):
        i = 0
        B1 = q[i:i + m * r].reshape(m, r); i += m * r
        A1 = q[i:i + r * n].reshape(r, n); i += r * n
        B2 = q[i:i + m * r].reshape(m, r); i += m * r
        A2 = q[i:i + r * n].reshape(r, n)
        return (B1 @ A1) * (B2 @ A2)
    return f
for (m, n, r) in [(8, 7, 2), (12, 10, 2), (12, 10, 3)]:
    q = torch.randn(2 * r * (m + n))
    print((m, n, r), jrank(loha(m, n, r), q), "pred", 2 * r * (m + n) - 2 * r * r - (m + n - 1))

print("== 6. weight-decay charge ==")
def wd_flow(etaB, etaA, lam, lr_scaled, T=2.0, h=1e-4):
    torch.manual_seed(3)
    m_, n_, r_ = 5, 4, 2
    W0 = torch.randn(m_, n_); Wt = torch.randn(m_, n_)
    B = torch.randn(m_, r_) * 0.5; A = torch.randn(r_, n_) * 0.5
    C0 = B.T @ B / etaB - A @ A.T / etaA
    for _ in range(int(T / h)):
        G = (W0 + B @ A - Wt)
        if lr_scaled:
            dB = -etaB * (G @ A.T + lam * B); dA = -etaA * (B.T @ G + lam * A)
        else:
            dB = -etaB * G @ A.T - lam * B; dA = -etaA * B.T @ G - lam * A
        B = B + h * dB; A = A + h * dA
    C = B.T @ B / etaB - A @ A.T / etaA
    pred = np.exp(-2 * lam * T) * C0 if not lr_scaled else None
    return C, C0
for etaB, etaA, lrs in [(4.0, 1.0, True), (1.0, 1.0, True), (4.0, 1.0, False)]:
    lam = 0.5; T = 2.0
    C, C0 = wd_flow(etaB, etaA, lam, lrs)
    if lrs and etaA == etaB:
        pred = np.exp(-2 * lam * etaA * T) * C0
    else:
        pred = np.exp(-2 * lam * T) * C0
    print("etaB,etaA,lr_scaled", etaB, etaA, lrs, "rel err vs closed form", float((C - pred).norm() / pred.norm()))

print("== 7. cometrics: ScaledGD and LoRA-Pro ==")
m, n, r = 7, 6, 2
B = torch.randn(m, r); A = torch.randn(r, n); G = torch.randn(m, n)
PU = B @ torch.linalg.inv(B.T @ B) @ B.T; PV = A.T @ torch.linalg.inv(A @ A.T) @ A
K_sgd = lambda B, A: G @ A.T @ A + B @ B.T @ G
K_sc = lambda B, A: G @ A.T @ torch.linalg.inv(A @ A.T) @ A + B @ torch.linalg.inv(B.T @ B) @ B.T @ G
g = torch.randn(r, r)
print("SGD cometric gauge defect", float((K_sgd(B @ g, torch.linalg.inv(g) @ A) - K_sgd(B, A)).norm()))
print("ScaledGD cometric gauge defect", float((K_sc(B @ g, torch.linalg.inv(g) @ A) - K_sc(B, A)).norm()),
      "| equals PU G + G PV:", float((K_sc(B, A) - (PU @ G + G @ PV)).norm()))
# tangent projection
Pi = PU @ G + G @ PV - PU @ G @ PV
# check Pi is orthogonal projection onto T = {B X + Y A}
Xs = torch.randn(m, r); Ys = torch.randn(r, n)
T = Xs @ A + B @ Ys
print("<G - Pi, T> =", float(((G - Pi) * T).sum()), "(0 => Pi is the orthogonal projection)")

print("== 8. VeRA span = {XA} ==")
m, n, r = 6, 5, 2
Bv = torch.randn(m, r); Av = torch.randn(r, n)
mats = []
for _ in range(200):
    b = torch.randn(m); d = torch.randn(r)
    mats.append((torch.diag(b) @ Bv @ torch.diag(d) @ Av).reshape(-1))
M = torch.stack(mats)
print("dim span VeRA images", int(torch.linalg.matrix_rank(M)), "pred m*r", m * r)

print("== 9. RoAd orbit dimension ==")
def road(m, n, W0):
    def f(q):
        th = q[:m // 2]; al = q[m // 2:]
        R = torch.zeros(m, m)
        for i in range(m // 2):
            c, s_ = torch.cos(th[i]), torch.sin(th[i])
            blk = al[i] * torch.stack([torch.stack([c, -s_]), torch.stack([s_, c])])
            R = R.index_put((torch.tensor([2 * i, 2 * i, 2 * i + 1, 2 * i + 1]), torch.tensor([2 * i, 2 * i + 1, 2 * i, 2 * i + 1])), blk.reshape(-1))
        return R @ W0
    return f
m, n = 8, 6
W0 = torch.randn(m, n)
q = torch.cat([torch.randn(m // 2), torch.rand(m // 2) + 0.5])
print("RoAd jac rank", jrank(road(m, n, W0), q), "params", m)

print("== 10. rank formula of dkappa ==")
for (m, n, r, ra, rb) in [(7, 6, 3, 3, 0), (7, 6, 3, 3, 3), (7, 6, 3, 2, 1)]:
    A = torch.randn(r, n); B = torch.randn(m, r)
    if ra < r: A[ra:] = 0
    if rb < r: B[:, rb:] = 0
    def f(q):
        dB = q[:m * r].reshape(m, r); dA = q[m * r:].reshape(r, n)
        return dB @ A + B @ dA
    print((m, n, r, ra, rb), jrank(f, torch.randn(r * (m + n))), "pred", m * ra + n * rb - ra * rb)

print("== 11. re-HRA: rank(R-I)<=Kr reachable; product bound ==")
n = 8
def refl(u): return torch.eye(n) - 2 * torch.outer(u, u) / (u @ u)
R1 = refl(torch.randn(n)) @ refl(torch.randn(n))
R2 = refl(torch.randn(n)) @ refl(torch.randn(n))
print("rank(R1R2 - I)", int(torch.linalg.matrix_rank(R1 @ R2 - torch.eye(n))), "<= 4")

print("== 12. Referee-round checks ==")
import scipy.linalg as sla
torch.manual_seed(12)

print("-- 12a. DoRA dimension (Prop II.8) --")
def dora(m, n, r, W0):
    def f(q):
        B = q[:m * r].reshape(m, r); A = q[m * r:m * r + r * n].reshape(r, n); mu = q[m * r + r * n:]
        V = W0 + B @ A
        return torch.diag(mu / V.norm(dim=1)) @ V
    return f
for (m, n, r, kind) in [(3, 3, 2, "generic"), (8, 8, 2, "generic"), (6, 6, 2, "rank-one W0")]:
    W0 = torch.randn(m, n) if kind == "generic" else torch.outer(torch.rand(m) + 0.5, torch.randn(n))
    q = torch.randn(r * (m + n) + m)
    print((m, n, r), kind, "rank", jrank(dora(m, n, r, W0), q), "| formula r(m+n-r)+m", r * (m + n - r) + m, "| mn", m * n)

print("-- 12b. GraLoRA generic rank of dW (Prop II.12) --")
for (m, n, r, k) in [(16, 16, 4, 2), (12, 12, 6, 3)]:
    q = torch.randn(r * (m + n))
    print((m, n, r, k), "rank dW", int(torch.linalg.matrix_rank(gralora(m, n, r, k)(q))), "min(kr,m,n)", min(k * r, m, n))

print("-- 12c. LoHa at (4,4,2): capped by mn --")
print("rank", jrank(loha(4, 4, 2), torch.randn(2 * 2 * 8)), "formula", 2 * 2 * 8 - 8 - 7, "mn", 16)

print("-- 12d. LoReFT image dimension (Prop VI.4) --")
d, r = 6, 2
def loreft(q):
    R = q[:r * d].reshape(r, d); Wl = q[r * d:2 * r * d].reshape(r, d); b = q[2 * r * d:]
    Rq, _ = torch.linalg.qr(R.T)  # orthonormal columns d x r
    Ro = Rq.T
    return torch.cat([(torch.eye(d) + Ro.T @ (Wl - Ro)).reshape(-1), Ro.T @ b])
print("LoReFT rank", jrank(loreft, torch.randn(2 * r * d + r)), "pred r(2d+1-r)", r * (2 * d + 1 - r),
      "| LoRA_r + free bias", r * (2 * d - r) + d)

print("-- 12e. serial adapter image dimension (Prop VI.4(b)) --")
m, n, r = 5, 3, 1
W0 = torch.randn(m, n)
def serial(q):
    U = q[:m * r].reshape(m, r); Dm = q[m * r:].reshape(r, m)
    return (torch.eye(m) + U @ Dm) @ W0
print("serial rank", jrank(serial, torch.randn(2 * m * r)), "LoRA d", r * (m + n - r))

print("-- 12f. FourierFT conjugate pairs (Props II.12-13) --")
def atom(u, v, N=8):
    S = np.zeros((N, N), complex); S[u, v] = 1
    return np.fft.ifft2(S).real
print("atoms (1,2) and (7,6) identical:", np.allclose(atom(1, 2), atom(7, 6)))
M3 = np.stack([atom(1, 2).ravel(), atom(7, 6).ravel(), atom(3, 5).ravel()])
print("rank with a conjugate pair among 3 frequencies:", np.linalg.matrix_rank(M3))

print("-- 12g. HF Cayley-Neumann OFT (Cor II.6, Thm V.3) --")
def cayley_neumann(Q):
    I = torch.eye(Q.shape[0]); return I + 2 * Q + 2 * Q @ Q + 2 * Q @ Q @ Q + Q @ Q @ Q @ Q
def skew(n, s):
    X = torch.randn(n, n) * s; return (X - X.T) / 2
Q = skew(8, 0.1); R = cayley_neumann(Q); I8 = torch.eye(8)
Q4 = Q @ Q @ Q @ Q
print("||R^T R - (I-Q^4)^2||", float((R.T @ R - (I8 - Q4) @ (I8 - Q4)).norm()), "| ||R^T R - I||", float((R.T @ R - I8).norm()))
W = torch.randn(6, 8); K0 = W @ W.T; drift = {}
for c in range(1, 201):
    W = W @ cayley_neumann(skew(8, 0.07)).T
    if c in (1, 10, 50, 200): drift[c] = float((W @ W.T - K0).norm() / K0.norm())
print("relative K_out drift after merges", {k: round(v, 3) for k, v in drift.items()})

print("-- 12h. BOFT sign of s (Cor II.6) --")
W0 = torch.randn(6, 8); Om = skew(8, 0.5); Rc = torch.linalg.solve(torch.eye(8) - Om, torch.eye(8) + Om)
s = torch.tensor([-0.5, 1, 1, 1, 1, 1.])
def cosm(W):
    Wn = W / W.norm(dim=1, keepdim=True); return Wn @ Wn.T
Wb = torch.diag(s) @ W0 @ Rc.T
print("max |cos change|", float((cosm(Wb) - cosm(W0)).abs().max()), "| max ||cos| change|", float((cosm(Wb).abs() - cosm(W0).abs()).abs().max()))

print("-- 12i. DARE torus equivariance, TIES homogeneity (Prop V.4) --")
N, p = 50, 0.7
tau = torch.randn(N); mask = (torch.rand(N) > p).double(); Dg = torch.exp(torch.randn(N)) * torch.sign(torch.randn(N))
dare = lambda x: mask * x / (1 - p)
print("DARE(D tau) - D DARE(tau):", float((dare(Dg * tau) - Dg * dare(tau)).abs().max()))
def ties(T, k=0.2):
    T = T.clone(); thr = T.abs().flatten().kthvalue(int((1 - k) * T.numel())).values
    T[T.abs() < thr] = 0; sg = torch.sign(T.sum(0)); agree = (torch.sign(T) == sg) & (T != 0)
    return (T * agree).sum(0) / agree.sum(0).clamp(min=1)
T3 = torch.randn(3, N)
print("TIES(3.7 T) - 3.7 TIES(T):", float((ties(3.7 * T3) - 3.7 * ties(T3)).abs().max()),
      "| TIES(-2T) + 2 TIES(T):", float((ties(-2 * T3) + 2 * ties(T3)).abs().max()),
      "| generic positive diagonal:", float((ties(T3 * Dg.abs()) - Dg.abs() * ties(T3)).abs().max()))

print("-- 12j. covariant splits (Prop II.16) --")
m, n, r = 6, 5, 2
X = torch.randn(n, 40); C = X @ X.T / 40; W0 = torch.randn(m, n)
def trunc(M, r):
    U, S, Vh = torch.linalg.svd(M); return U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
def whiten(W, C):
    L = torch.linalg.cholesky(C); return trunc(W @ L, r) @ torch.linalg.inv(L)
corda = lambda W, C: trunc(W @ C, r) @ torch.linalg.inv(C)
pissa = lambda W, C: trunc(W, r)
Oq, _ = torch.linalg.qr(torch.randn(n, n))
for name, h in [("3Q", 3 * Oq), ("generic", torch.randn(n, n) + 2 * torch.eye(n))]:
    hi = torch.linalg.inv(h)
    out = []
    for f in (whiten, corda, pissa):
        out.append(float((f(W0 @ hi, h @ C @ h.T) - f(W0, C) @ hi).norm()))
    print(name, "defects whiten/CorDA/PiSSA", [f"{x:.1e}" for x in out])
hgen = torch.randn(n, n) + 2 * torch.eye(n); hgi = torch.linalg.inv(hgen)
damped = lambda W, C: whiten(W, C + 0.1 * torch.eye(n))
print("damped (0.1) whitened split, generic h: defect", float((damped(W0 @ hgi, hgen @ C @ hgen.T) - damped(W0, C) @ hgi).norm()))
Ev, Evec = torch.linalg.eigh(W0 @ C @ W0.T); Pi = Evec[:, -r:] @ Evec[:, -r:].T
print("whitened split == Pi_r W0:", float((whiten(W0, C) - Pi @ W0).norm()))

print("-- 12k. Adam joint torus needs eps rescaled (Thm III.12(b)) --")
m, n, r = 6, 5, 3
B = torch.randn(m, r); A = torch.randn(r, n); G = torch.randn(m, n) * 1e-6
dvec = torch.tensor([0.3, 1.0, 4.0])
def first_step(B, A, eB, eA, epsB, epsA):
    gB = G @ A.T; gA = B.T @ G
    return (-eB * gB / (gB.abs() + epsB)) @ A + B @ (-eA * gA / (gA.abs() + epsA))
base = first_step(B, A, 1.0, 1.0, 1e-8, 1e-8)
Bg, Ag = B @ torch.diag(1 / dvec), torch.diag(dvec) @ A
fixed = first_step(Bg, Ag, (1 / dvec)[None, :], dvec[:, None], 1e-8, 1e-8)
resc = first_step(Bg, Ag, (1 / dvec)[None, :], dvec[:, None], (dvec * 1e-8)[None, :], (1e-8 / dvec)[:, None])
print("relative defect: eps fixed", float((fixed - base).norm() / base.norm()), "| eps rescaled", float((resc - base).norm() / base.norm()))

print("-- 12l. LoRA-Pro Sylvester X: parameter-level equivariance (Thm III.12) --")
m, n, r, s_ = 8, 7, 3, 1.5
B = torch.randn(m, r); A = torch.randn(r, n); G = torch.randn(m, n)
def lorapro(B, A, Xzero=False):
    gA = s_ * B.T @ G; gB = s_ * G @ A.T
    BtB = B.T @ B; AAt = A @ A.T
    if Xzero:
        Xs = torch.zeros(r, r)
    else:
        rhs = -(1 / s_ ** 2) * torch.linalg.inv(BtB) @ gA @ A.T
        Xs = torch.tensor(sla.solve_sylvester(BtB.numpy(), AAt.numpy(), rhs.numpy()))
    PB = torch.eye(m) - B @ torch.linalg.inv(BtB) @ B.T
    hA = (1 / s_ ** 2) * torch.linalg.inv(BtB) @ gA + Xs @ A
    hB = (1 / s_ ** 2) * PB @ gB @ torch.linalg.inv(AAt) - B @ Xs
    return hB, hA
g = torch.diag(torch.tensor([0.3, 1.0, 4.0])) + 0.5 * torch.randn(r, r)
Og, _ = torch.linalg.qr(torch.randn(r, r))
for gname, gm in [("non-orthogonal g", g), ("orthogonal g", Og)]:
    for xz in (False, True):
        hB, hA = lorapro(B, A, xz); hB2, hA2 = lorapro(B @ torch.linalg.inv(gm), gm @ A, xz)
        print(gname, "X=0" if xz else "Sylvester X", "param defect", float((hB2 - hB @ torch.linalg.inv(gm)).norm() + (hA2 - gm @ hA).norm()))

print("-- 12m. strata dims and HF 'orthogonal' init (Thm II.2) --")
m, n, r = 7, 5, 2
def s1dim(B0, A0):
    def f(q):
        dB = q[:m * r].reshape(m, r); dA = q[m * r:].reshape(r, n); return dB @ A0 + B0 @ dA
    return jrank(f, torch.randn(r * (m + n)))
Qo, _ = torch.linalg.qr(torch.randn(r, r))
A_or = (torch.randn(n, r // 2) @ Qo[0::2]).T / 10; B_or = torch.randn(r // 2, m).T @ Qo[1::2] / 10
print("T_A", s1dim(torch.zeros(m, r), torch.randn(r, n)), "T_B", s1dim(torch.randn(m, r), torch.zeros(r, n)),
      "S", s1dim(torch.randn(m, r), torch.randn(r, n)), "| orthogonal init: |B0A0|", float((B_or @ A_or).abs().max()), "dim S1", s1dim(B_or, A_or))

print("-- 12n. Obs I.7 and Prop I.8 --")
m, n = 7, 6
A0 = torch.randn(2, n); A1 = torch.randn(2, n)
f4 = lambda q: q.reshape(m, 4) @ torch.cat([A0, A1])
f22 = lambda q: q.reshape(m, 4) @ torch.cat([A0, A0])
print("dim S1: standard LoRA_4", jrank(f4, torch.randn(4 * m)), "| LoRA_2 (+) LoRA_2 same A0", jrank(f22, torch.randn(4 * m)))
m, n, r = 6, 8, 2
Af = torch.randn(r, n); Bf = torch.randn(m, r)
fp = lambda q: q[:m * r].reshape(m, r) @ Af - Bf @ q[m * r:].reshape(r, n)  # B A0 - B0 A (linear)
print("fibre product FA x FB dimension", r * (m + n) - jrank(fp, torch.randn(r * (m + n))), "pred r^2", r * r)

print("-- 12o. HiRA: two cycles (Prop II.13) --")
m, n = 6, 5
W0 = torch.rand(m, n) + 0.5; J = torch.ones(m, n)
Z1 = torch.outer(torch.randn(m), torch.randn(n)); Z2 = torch.outer(torch.randn(m), torch.randn(n))
W2 = W0 * (J + Z1) * (J + Z2)
print("rank((W2-W0)/W0)", int(torch.linalg.matrix_rank((W2 - W0) / W0)), "> 2r =", 2)

print("-- 12p. Kronecker ranks (Thm II.11) --")
def kron_rank(D, m1, m2, n1, n2):
    T = D.reshape(m1, m2, n1, n2).permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)
    return int(torch.linalg.matrix_rank(T))
u, v = torch.randn(8), torch.randn(8)
print("2x4 split, generic uv^T: kron rank", kron_rank(torch.outer(u, v), 2, 4, 2, 4), "= maximal Kronecker rank min(m1 n1, m2 n2) = 4 (aligned split)")
mx = 0
for _ in range(200):
    Cc = torch.randn(4, 4); D = torch.outer(torch.randn(4), torch.randn(4))
    mx = max(mx, int(torch.linalg.matrix_rank(torch.kron(Cc, D))))
print("LoKr s=4, r'=1: max rank", mx)

print("-- 12q. butterfly hypergraph components (Thm V.3(c), Cor 4) --")
def components(n, b, mp):
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    h = b // 2
    for l in range(mp):
        for c in range(n // h):
            c2 = c ^ (1 << l)
            if c2 < n // h:
                for i in list(range(c * h, c * h + h)) + list(range(c2 * h, c2 * h + h)):
                    parent[find(i)] = find(c * h)
    sizes = {}
    for i in range(n): sizes[find(i)] = sizes.get(find(i), 0) + 1
    return sorted(set(sizes.values()))
print("n=64, b=4: component sizes for m'=1..5", [components(64, 4, k) for k in range(1, 6)])

print("-- 12r. weight-decay slices (Cor III.6) --")
def decay_flow(lr_scaled, etaB=4.0, etaA=1.0, lam=0.1, T=200.0, h=1e-3):
    torch.manual_seed(5)
    m_, n_, r_ = 12, 10, 3
    Y = torch.randn(m_, n_); B = torch.zeros(m_, r_); A = torch.rand(r_, n_) * 2 / np.sqrt(n_) - 1 / np.sqrt(n_)
    for _ in range(int(T / h)):
        Gm = B @ A - Y
        if lr_scaled:
            dB = -etaB * (Gm @ A.T + lam * B); dA = -etaA * (B.T @ Gm + lam * A)
        else:
            dB = -etaB * Gm @ A.T - lam * B; dA = -etaA * B.T @ Gm - lam * A
        B = B + h * dB; A = A + h * dA
    mu = B.T @ B - A @ A.T; Phi = B.T @ B / etaB - A @ A.T / etaA
    return float(mu.norm() / (B.T @ B).norm()), float(Phi.norm() / (B.T @ B / etaB).norm())
print("coupled decay: rel |mu|, rel |Phi|", decay_flow(True))
print("per-time decay: rel |mu|, rel |Phi|", decay_flow(False))

print("-- 12s. zero-B example rule is GL_r-equivariant (Cor III.13) --")
m, n, r = 7, 6, 3
torch.manual_seed(9)
W0 = torch.randn(m, n); Wt = torch.randn(m, n); A0 = torch.randn(r, n); gg = torch.randn(r, r) + 2 * torch.eye(r)
def run_rule(A, steps=50, eta=1e-2):
    B = torch.zeros(m, r)
    for _ in range(steps):
        Gm = W0 + B @ A - Wt; gB = Gm @ A.T; gA = B.T @ Gm
        B, A = B - eta * gB @ torch.linalg.inv(A @ A.T), A - eta * (A @ A.T) @ gA
    return W0 + B @ A
print("trajectory endpoint difference A0 vs g A0", float((run_rule(A0) - run_rule(gg @ A0)).abs().max()))

print("-- 12t. Lie algebra generated by HF Cayley-Neumann factors (Thm V.3 Cor 4) --")
def lie_closure(gens, tol=1e-8):
    basis = []
    def add(M):
        v = M.reshape(-1)
        if not basis:
            if v.norm() > tol: basis.append(v / v.norm()); return True
            return False
        Bm = torch.stack(basis)
        res = v - Bm.T @ (Bm @ v)
        if res.norm() > tol * max(1.0, float(v.norm())): basis.append(res / res.norm()); return True
        return False
    mats = []
    for G_ in gens:
        if add(G_): mats.append(G_)
    changed = True
    while changed:
        changed = False
        cur = list(mats)
        for X_ in cur:
            for Y_ in cur:
                Z_ = X_ @ Y_ - Y_ @ X_
                if add(Z_): mats.append(Z_); changed = True
    return len(basis)
logs = []
for _ in range(12):
    Rcn = cayley_neumann(skew(4, 0.3))
    logs.append(torch.tensor(sla.logm(Rcn.numpy()).real))
print("dim Lie algebra generated (b=4):", lie_closure(logs), "| dim gl(4) = 16, dim so(4) = 6 (identifies the generated group only; re-merging reaches a monoid, see 13n)")

print("== 13. Second referee round ==")
torch.manual_seed(13)

print("-- 13a. S_1 of sums of LoRAs (Obs I.7), m = 7, n = 6 --")
m_, n_ = 7, 6
A0_, A1_ = torch.randn(2, n_), torch.randn(2, n_)
a1_, a2_, a3_ = torch.randn(1, n_), torch.randn(2, n_), torch.randn(3, n_)
print("LoRA_2 (+) LoRA_2 independent A0:", jrank(lambda q: q.reshape(m_, 4) @ torch.cat([A0_, A1_]), torch.randn(4 * m_)),
      "| F(1) (+) F(2):", jrank(lambda q: q.reshape(m_, 3) @ torch.cat([a1_, a2_]), torch.randn(3 * m_)),
      "| standard F(3):", jrank(lambda q: q.reshape(m_, 3) @ a3_, torch.randn(3 * m_)))

print("-- 13b. HF orthogonal init vs T_A at (m,n,r) = (4,5,2) (Thm II.2(e)) --")
def s1jac(B0, A0):
    mm, rr = B0.shape; nn = A0.shape[1]
    f = lambda q: (q[:mm * rr].reshape(mm, rr) @ A0 + B0 @ q[mm * rr:].reshape(rr, nn)).reshape(-1)
    return jacobian(f, torch.randn(rr * (mm + nn)))
m_, n_, r_ = 4, 5, 2
Qo_, _ = torch.linalg.qr(torch.randn(r_, r_))
A_or_ = (torch.randn(n_, 1) @ Qo_[0::2]).T / 10; B_or_ = torch.randn(1, m_).T @ Qo_[1::2] / 10
J_or = s1jac(B_or_, A_or_); J_ta = s1jac(torch.zeros(m_, r_), torch.randn(r_, n_))
rk = lambda J: int(torch.linalg.matrix_rank(J, atol=1e-9))
print("dim S1 orthogonal", rk(J_or), "| T_A", rk(J_ta), "| mr", m_ * r_, "| dim of the sum of the two S1", rk(torch.cat([J_or, J_ta], 1)))

print("-- 13c. Cor II.6: BOFT with negative s, GOFT scaler, conformal block, singular Cayley-Neumann --")
W0_ = torch.randn(6, 8); Om_ = skew(8, 0.5); Rc_ = torch.linalg.solve(torch.eye(8) - Om_, torch.eye(8) + Om_)
sneg = -(torch.rand(6) + 0.1)
print("all-negative s: max cos change", float((cosm(torch.diag(sneg) @ W0_ @ Rc_.T) - cosm(W0_)).abs().max()))
gsc = 1 + 0.05 * (2 * torch.rand(8) - 1); Wg_ = W0_ @ torch.diag(gsc) @ Rc_
sv0 = torch.linalg.svdvals(W0_); svg = torch.linalg.svdvals(Wg_)
print("GOFT with scaler: max cos change", float((cosm(Wg_) - cosm(W0_)).abs().max()),
      "| max rel singular value change", float(((svg - sv0) / sv0).abs().max()),
      "| K_out - W0 diag(g)^2 W0^T", float((Wg_ @ Wg_.T - W0_ @ torch.diag(gsc ** 2) @ W0_.T).norm()))
Cb = torch.eye(8); th = 0.3
Cb[:2, :2] = np.sqrt(2) * torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
print("conformal block sqrt2 Rot(0.3) on 2 of 8 inputs: max cos change", float((cosm(W0_ @ Cb) - cosm(W0_)).abs().max()))
Jq = torch.tensor([[0., -1.], [1., 0.]])
print("Cayley-Neumann at theta = 1 (b = 2): ||R||", float(cayley_neumann(Jq).norm()))

print("-- 13d. Hurwitz move H_a H_b = H_{H_a b} H_a (Thm II.7) --")
Hh = lambda u: torch.eye(6) - 2 * torch.outer(u, u) / (u @ u)
ua, ub = torch.randn(6), torch.randn(6)
print("defect", float((Hh(ua) @ Hh(ub) - Hh(Hh(ua) @ ub) @ Hh(ua)).norm()))

print("-- 13e. DoRA with one zero magnitude at (8,8,2) (Prop II.8) --")
qz = torch.randn(2 * 16 + 8); qz[-1] = 0.0
print("rank", jrank(dora(8, 8, 2, torch.randn(8, 8)), qz), "| with all mu_i != 0 the formula gives", 2 * 14 + 8)

print("-- 13f. LoKr caps (Thm II.11(e)) --")
def lokr_max(m1, m2, n1, n2, rin, trials=50):
    mx = 0
    for _ in range(trials):
        Cc = torch.randn(m1, n1); Bk = torch.randn(m2, rin); Ak = torch.randn(rin, n2)
        mx = max(mx, int(torch.linalg.matrix_rank(torch.kron(Cc, Bk @ Ak))))
    return mx
print("(2,8,8,2) inner 4: max rank", lokr_max(2, 8, 8, 2, 4), "| (4,2,2,8) inner 5: max rank", lokr_max(4, 2, 2, 8, 5),
      "| predicted min(m1,n1) min(r',m2,n2): 4, 4")

print("-- 13g. LoHa vs LoRA_2r, LoHa orbit, HRA at r >= n-1 (Prop II.12) --")
print("(9,9,3): LoHa", jrank(loha(9, 9, 3), torch.randn(2 * 3 * 18)), "LoRA_6", 6 * (18 - 6),
      "| (4,4,2): LoHa", jrank(loha(4, 4, 2), torch.randn(2 * 2 * 8)), "LoRA_4", 4 * (8 - 4))
def loha_orbit_dim(m, n, r):
    B1, A1, B2, A2 = torch.randn(m, r), torch.randn(r, n), torch.randn(m, r), torch.randn(r, n)
    vecs = []
    flat = lambda a, b, c, d_: torch.cat([a.reshape(-1), b.reshape(-1), c.reshape(-1), d_.reshape(-1)])
    Z = lambda M: torch.zeros_like(M)
    for i in range(r):
        for j in range(r):
            E = torch.zeros(r, r); E[i, j] = 1
            vecs.append(flat(-B1 @ E, E @ A1, Z(B2), Z(A2)))
            vecs.append(flat(Z(B1), Z(A1), -B2 @ E, E @ A2))
    for i in range(m):
        D = torch.zeros(m, m); D[i, i] = 1
        vecs.append(flat(D @ B1, Z(A1), -D @ B2, Z(A2)))
    for i in range(n):
        D = torch.zeros(n, n); D[i, i] = 1
        vecs.append(flat(Z(B1), A1 @ D, Z(B2), -A2 @ D))
    return int(torch.linalg.matrix_rank(torch.stack(vecs)))
print("LoHa (4,4,2): orbit dim", loha_orbit_dim(4, 4, 2), "| fibre dim", 2 * 2 * 8 - jrank(loha(4, 4, 2), torch.randn(32)))
nH = 5; W0H = torch.randn(nH + 2, nH)
print("HRA n = 5: rank at r = 4", jrank(hra(nH, 4, W0H), torch.randn(4 * nH)), "at r = 5", jrank(hra(nH, 5, W0H), torch.randn(5 * nH)),
      "| dim SO(5)", nH * (nH - 1) // 2)

print("-- 13h. transport and rebasing (Prop II.13) --")
th0 = torch.tensor([[1.0, 0.0]]); Rpi = -torch.eye(2); th = th0.clone()
for _ in range(2): th = th @ (2 * Rpi - torch.eye(2))
print("OFT transported by T = 2I, two cycles: theta_2 =", th.tolist(), "norm", float(th.norm()))
W0r = torch.rand(30, 30) + 0.5; Jr = torch.ones(30, 30); hr = []
for rr in (1, 2, 3):
    Z1 = torch.randn(30, rr) @ torch.randn(rr, 30); Z2 = torch.randn(30, rr) @ torch.randn(rr, 30)
    W2r = W0r * (Jr + Z1) * (Jr + Z2); hr.append(int(torch.linalg.matrix_rank((W2r - W0r) / W0r)))
print("HiRA two cycles, relative-update rank at r = 1,2,3:", hr, "| min(2r + r^2, 30):", [3, 8, 15])

print("-- 13i. bottom-r and damped splits (Prop II.16), m,n,r = 7,6,2 --")
m_, n_, r_ = 7, 6, 2; k_ = min(m_, n_)
Xd = torch.randn(n_, 40); Cd = Xd @ Xd.T / 40; W0d = torch.randn(m_, n_)
tr_ = lambda M, rr: trunc(M, rr)
bottom = lambda M, rr: M - tr_(M, k_ - rr)
kpm = lambda W, C: bottom(W @ C, r_) @ torch.linalg.inv(C)
milora = lambda W, C: bottom(W, r_)
def whiten_r(W, C, rr=r_):
    L = torch.linalg.cholesky(C); return tr_(W @ L, rr) @ torch.linalg.inv(L)
absd = lambda W, C: whiten_r(W, C + 0.1 * torch.eye(n_))
reld = lambda W, C: whiten_r(W, C + 0.1 * torch.trace(C) / n_ * torch.eye(n_))
Oq_, _ = torch.linalg.qr(torch.randn(n_, n_))
for name, h in [("3Q", 3 * Oq_), ("generic", torch.randn(n_, n_) + 2 * torch.eye(n_))]:
    hi = torch.linalg.inv(h)
    print(name, "defects KPM(bottom-r)/MiLoRA(bottom-r)/absolute damping/trace-relative damping",
          [f"{float((f(W0d @ hi, h @ Cd @ h.T) - f(W0d, Cd) @ hi).norm()):.1e}" for f in (kpm, milora, absd, reld)])
hg_ = torch.randn(n_, n_) + 2 * torch.eye(n_); hgi_ = torch.linalg.inv(hg_)
print("r = n: PiSSA under a generic h", float((tr_(W0d @ hgi_, n_) - tr_(W0d, n_) @ hgi_).norm()))

print("-- 13j. per-channel charges and the GD step of Phi (Thm III.5) --")
torch.manual_seed(31)
m_, n_, r_ = 5, 4, 3
etaB = torch.tensor([1.0, 1.0, 1.0]); etaA = torch.tensor([1.0, 1.0, 2.0])
Yt = torch.randn(m_, n_)
def gradW(W): return torch.tanh(W) - Yt  # gradient of a non-quadratic loss sum(logcosh(W)) - <Y,W>
def field(B, A):
    Gm = gradW(B @ A); return -(Gm @ A.T) * etaB[None, :], -etaA[:, None] * (B.T @ Gm)
def charge(B, A, i, j):
    return float((B.T @ B)[i, j] - (A @ A.T)[i, j])
B_, A_ = torch.randn(m_, r_), torch.randn(r_, n_)
c01, c02 = charge(B_, A_, 0, 1), charge(B_, A_, 0, 2)
hstep = 1e-3
for _ in range(1000):
    k1 = field(B_, A_); k2 = field(B_ + hstep / 2 * k1[0], A_ + hstep / 2 * k1[1])
    k3 = field(B_ + hstep / 2 * k2[0], A_ + hstep / 2 * k2[1]); k4 = field(B_ + hstep * k3[0], A_ + hstep * k3[1])
    B_ = B_ + hstep / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); A_ = A_ + hstep / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
print("eta_B = (1,1,1), eta_A = (1,1,2): drift of the (0,1) charge", abs(charge(B_, A_, 0, 1) - c01),
      "| drift of the (0,2) form", abs(charge(B_, A_, 0, 2) - c02))
eB_, eA_ = 0.3, 0.7
B_, A_, Gm_ = torch.randn(m_, r_), torch.randn(r_, n_), torch.randn(m_, n_)
Phi = lambda B, A: B.T @ B / eB_ - A @ A.T / eA_
B1_, A1_ = B_ - eB_ * Gm_ @ A_.T, A_ - eA_ * B_.T @ Gm_
pred = eB_ * A_ @ Gm_.T @ Gm_ @ A_.T - eA_ * B_.T @ Gm_ @ Gm_.T @ B_
print("one GD step: Phi change minus the exact formula", float((Phi(B1_, A1_) - Phi(B_, A_) - pred).norm()))

print("-- 13k. Adam moves Phi at first order in the step size (Cor III.6) --")
B_, A_, Gm_ = torch.randn(m_, r_), torch.randn(r_, n_), torch.randn(m_, n_)
ratios = []
for lr in (1e-2, 1e-3, 1e-4):
    B1_ = B_ - lr * torch.sign(Gm_ @ A_.T); A1_ = A_ - lr * torch.sign(B_.T @ Gm_)
    ratios.append(round(float((Phi(B1_, A1_) - Phi(B_, A_)).norm()) / lr, 3))
print("|Delta Phi| / lr at lr = 1e-2, 1e-3, 1e-4:", ratios)

print("-- 13l. GaLore T = 1 bracket at W = diag(2,1); Bures-Wasserstein cometric (Prop IV.3) --")
def u_top(W):
    U, S, Vh = torch.linalg.svd(W); u = U[:, 0]; return u * torch.sign(u[0])
def Xfield(i):
    def f(W):
        e = torch.zeros(2); e[i] = 1.0; return torch.outer(u_top(W), e)  # grad L = W for L = |W|^2 / 2
    return f
W_ = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
def dirderiv(f, W, V):
    return jacobian(lambda w: f(w.reshape(2, 2)).reshape(-1), W.reshape(-1)) @ V.reshape(-1)
br = (dirderiv(Xfield(1), W_, Xfield(0)(W_)) - dirderiv(Xfield(0), W_, Xfield(1)(W_))).reshape(2, 2)
print("bracket [X1, X2] =", [[round(float(x), 6) for x in row] for row in br], "| predicted -(1/3) e2 e1^T")
rho_bw = lambda b: (b.reshape(2, 2) @ b.reshape(2, 2).T).reshape(-1)
Bb = torch.randn(2, 2); Rb, _ = torch.linalg.qr(torch.randn(2, 2))
Jb = jacobian(rho_bw, Bb.reshape(-1)); Jbr = jacobian(rho_bw, (Bb @ Rb).reshape(-1))
print("BB^T cometric at B and at BR:", float((Jb @ Jb.T - Jbr @ Jbr.T).norm()))

print("-- 13m. NF4 defects (Prop V.2) --")
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])
def nf4(W, block=64):
    flat = W.reshape(-1, block); sc = flat.abs().max(1, keepdim=True).values.clamp(min=1e-12)
    idx = ((flat / sc).unsqueeze(-1) - NF4).abs().argmin(-1)
    return (NF4[idx] * sc).reshape(W.shape)
torch.manual_seed(7)
Wq = torch.randn(256, 256) * 0.04; eq = nf4(Wq) - Wq; Pq = trunc(eq, 8)
Wgres = Wq - trunc(Wq, 8)
print("Gaussian W: ||e||", round(float(eq.norm()), 4), "| ||e - SVD_8(e)||", round(float((eq - Pq).norm()), 4),
      "| one-step LoftQ defect with s = 2:", round(float((eq - 2 * Pq).norm()), 4),
      "| QPiSSA defect", round(float((nf4(Wgres) - Wgres).norm()), 4))
Ul, _ = torch.linalg.qr(torch.randn(256, 8)); Vl, _ = torch.linalg.qr(torch.randn(256, 8))
Wl = Ul @ torch.diag(torch.linspace(3.0, 1.0, 8)) @ Vl.T + 0.01 * torch.randn(256, 256)
el = nf4(Wl) - Wl; Wres = Wl - trunc(Wl, 8)
print("low-rank-plus-noise W: ||e||", round(float(el.norm()), 4), "| QPiSSA defect", round(float((nf4(Wres) - Wres).norm()), 4),
      "| LoftQ (T = 1, s = 1) defect", round(float((el - trunc(el, 8)).norm()), 4))
Kq = nf4(Wq); Dq = nf4(0.1 * torch.randn(256, 256) * 0.04)
print("fraction of entries of nf4(W) + nf4(Delta) off the NF4 grid:", round(float((nf4(Kq + Dq) != Kq + Dq).double().mean()), 3))

print("-- 13n. Cayley-Neumann contraction (Thm V.3, corollary 4) --")
mx = 0.0
for _ in range(2000):
    Qc = skew(4, 1.0); Qc = Qc * (2 ** 0.25) * float(torch.rand(1)) / torch.linalg.matrix_norm(Qc, 2)
    mx = max(mx, float(torch.linalg.matrix_norm(cayley_neumann(Qc), 2)))
print("max ||R||_op over random Q with ||Q||_op <= 2^(1/4):", mx)
Wc = torch.randn(6, 8); worst = -1.0
for _ in range(200):
    Wn = Wc @ cayley_neumann(skew(8, 0.05)).T
    worst = max(worst, float(torch.linalg.eigvalsh(Wn @ Wn.T - Wc @ Wc.T).max())); Wc = Wn
print("largest eigenvalue of K_out(t+1) - K_out(t) over 200 merges (<= 0 means Loewner decrease):", worst)

print("-- 13o. factor-wise DARE, LoraHub at N = 3, Poly fibres (Prop V.4) --")
torch.manual_seed(41)
m_, n_, r_, p_ = 8, 7, 3, 0.5
Bs = [torch.randn(m_, r_) for _ in range(2)]; As = [torch.randn(r_, n_) for _ in range(2)]
MB = [(torch.rand(m_, r_) > p_).double() for _ in range(2)]; MA = [(torch.rand(r_, n_) > p_).double() for _ in range(2)]
def dare_fac(Bs, As):
    return (sum(MB[i] * Bs[i] for i in range(2)) / (1 - p_)) @ (sum(MA[i] * As[i] for i in range(2)) / (1 - p_))
Dd = torch.diag(torch.exp(torch.randn(r_)))
base_ = dare_fac(Bs, As)
one = dare_fac([Bs[0], Bs[1] @ Dd], [As[0], torch.linalg.inv(Dd) @ As[1]])
both = dare_fac([b @ Dd for b in Bs], [torch.linalg.inv(Dd) @ a for a in As])
print("diagonal gauge on adapter 2 only:", float((one - base_).norm()), "| shared by both adapters:", float((both - base_).norm()))
w3 = torch.randn(3); B3 = [torch.randn(m_, r_) for _ in range(3)]; A3 = [torch.randn(r_, n_) for _ in range(3)]; cc = 1.7
hub = lambda Bs, As: sum(w3[i] * Bs[i] for i in range(3)) @ sum(w3[j] * As[j] for j in range(3))
chg = hub([B3[0], B3[1], cc * B3[2]], [A3[0], A3[1], A3[2] / cc]) - hub(B3, A3)
form = sum(w3[l] * w3[2] * ((cc - 1) * B3[2] @ A3[l] + (1 / cc - 1) * B3[l] @ A3[2]) for l in range(2))
print("LoraHub change formula at N = 3:", float((chg - form).norm()))
def poly_fibre(K, Tt, r, m, n):
    al = torch.rand(Tt, K); al = al / al.sum(1, keepdim=True)
    def f(q):
        Bk = q[:K * m * r].reshape(K, m, r); Ak = q[K * m * r:].reshape(K, r, n)
        return torch.stack([torch.einsum('k,kmr->mr', al[t], Bk) @ torch.einsum('k,krn->rn', al[t], Ak) for t in range(Tt)]).reshape(-1)
    return K * r * (m + n) - jrank(f, torch.randn(K * r * (m + n)))
print("Poly fibre dims (K skills, T tasks, r = 2, 6x6): K=3,T=2:", poly_fibre(3, 2, 2, 6, 6), "| K=3,T=4:", poly_fibre(3, 4, 2, 6, 6),
      "| dim GL_2 = 4")

print("-- 13p. RoPE cup slide (Thm VI.3, R5) --")
dk = 8; freqs = 10000.0 ** (-torch.arange(0, dk, 2, dtype=torch.float64) / dk)
def Rm(mpos):
    R = torch.zeros(dk, dk)
    for j, f in enumerate(freqs):
        c, s_ = np.cos(mpos * float(f)), np.sin(mpos * float(f))
        R[2 * j:2 * j + 2, 2 * j:2 * j + 2] = torch.tensor([[c, -s_], [s_, c]])
    return R
Mg = torch.randn(dk, dk); qv, kv = torch.randn(dk), torch.randn(dk); mp, npos = 3, 7
lhs = (Rm(mp) @ Mg @ qv) @ (Rm(npos) @ kv)
print("<R_m M q, R_n k> - <q, M^T R_{n-m} k>:", float(lhs - qv @ (Mg.T @ Rm(npos - mp) @ kv)),
      "| slide onto the key side <R_m q, R_n M^T k> differs by", float(lhs - (Rm(mp) @ qv) @ (Rm(npos) @ Mg.T @ kv)))

print("-- 13q. serial adapter at (m,n,r) = (4,8,1) (Prop VI.4(b)) --")
W0s = torch.randn(4, 8)
print("serial rank", jrank(lambda q: ((torch.eye(4) + q[:4].reshape(4, 1) @ q[4:].reshape(1, 4)) @ W0s).reshape(-1), torch.randn(8)),
      "| LoRA_1 d", 1 * (4 + 8 - 1))
