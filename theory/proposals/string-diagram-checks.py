#!/usr/bin/env python3
"""Numerical checks for theory/proposals/string-diagram.md.

Every identity / rank / conservation claim marked "checked" in the proposal is
exercised here on random instances.  Run:  python3 string-diagram-checks.py
"""
import numpy as np

rng = np.random.default_rng(0)
rk = lambda M, tol=1e-8: int(np.linalg.matrix_rank(M, tol=tol))
ok = lambda name, cond, info="": print(("PASS " if cond else "FAIL ") + name, info)


def H(u):
    u = u / np.linalg.norm(u)
    return np.eye(len(u)) - 2 * np.outer(u, u)


# Thm 13: LoHa generic rank = min(m, n, r1 r2)
for (m, n, r1, r2) in [(40, 30, 2, 3), (40, 30, 4, 5), (40, 30, 6, 6), (7, 7, 3, 3)]:
    B1, A1, B2, A2 = [rng.normal(size=s) for s in [(m, r1), (r1, n), (m, r2), (r2, n)]]
    X = (B1 @ A1) * (B2 @ A2)
    ok("Thm13 LoHa generic rank", rk(X) == min(m, n, r1 * r2), f"{(m, n, r1, r2)} -> {rk(X)}")
    # face-splitting identity
    FS = lambda P, Q: np.einsum('ik,il->ikl', P, Q).reshape(P.shape[0], -1)
    ok("Thm13 face-splitting identity", np.allclose(X, FS(B1, B2) @ FS(A1.T, A2.T).T))

# Thm 12: Kronecker (operator-Schmidt) rank
def rearr(D, d1, d2, e1, e2):
    return D.reshape(d1, d2, e1, e2).transpose(0, 2, 1, 3).reshape(d1 * e1, d2 * e2)
s = 4
u, v = rng.normal(size=s * s), rng.normal(size=s * s)
ok("Thm12 generic rank-1 has Kronecker rank s^2", rk(rearr(np.outer(u, v), s, s, s, s)) == s * s)
ok("Thm12 identity has Kronecker rank 1", rk(rearr(np.eye(s * s), s, s, s, s)) == 1)
C, D = rng.normal(size=(s, s)), rng.normal(size=(s, s))
ok("Thm12 rank(C⊗D)=rkC·rkD", rk(np.kron(C, D)) == rk(C) * rk(D))

# Thm 17: rank of the differential of (B,A) -> W + BA
def dk_rank(B, A):
    m, r = B.shape; n = A.shape[1]; cols = []
    for i in range(m):
        for k in range(r):
            dB = np.zeros_like(B); dB[i, k] = 1; cols.append((dB @ A).ravel())
    for k in range(r):
        for j in range(n):
            dA = np.zeros_like(A); dA[k, j] = 1; cols.append((B @ dA).ravel())
    return rk(np.array(cols).T)
m, n, r = 7, 5, 2
A0, B0 = rng.normal(size=(r, n)), rng.normal(size=(m, r))
ok("Thm17 vertex (B=0)", dk_rank(np.zeros((m, r)), A0) == m * r)
ok("Thm17 smooth point", dk_rank(B0, A0) == r * (m + n - r))
B1 = B0.copy(); B1[:, 1] = 0
ok("Thm17 mixed ranks", dk_rank(B1, A0) == m * 2 + n * 1 - 2 * 1)

# Thm 14 (C1): Noether charge AA^T/eta_A - B^T B/eta_B under small-step GD
m, n, r = 6, 5, 3
Wst, W0 = rng.normal(size=(m, n)), rng.normal(size=(m, n))
A = 0.5 * rng.normal(size=(r, n)); B = np.zeros((m, r)); eA, eB = 1e-3, 4e-3
N0 = A @ A.T / eA - B.T @ B / eB
for _ in range(20000):
    G = W0 + B @ A - Wst
    B, A = B - eB * G @ A.T, A - eA * B.T @ G
N1 = A @ A.T / eA - B.T @ B / eB
ok("Thm14 LoRA+ Noether charge (discrete GD, O(eta) drift)",
   np.linalg.norm(N1 - N0) / np.linalg.norm(N0) < 1e-2,
   f"rel drift {np.linalg.norm(N1 - N0) / np.linalg.norm(N0):.2e}")

# Thm 14 (C2): VeRA scalar charge.  Exact for gradient flow; for GD the drift is O(eta):
# halving the step (at fixed training time) should halve the drift.
Bf, Af = rng.normal(size=(m, 4)), rng.normal(size=(4, n))
def vera_drift(scale, T=4.0):
    b, d = np.zeros(m), 0.1 * np.ones(4); eb, ed = 2 * scale, 1 * scale
    Q0 = b @ b / eb - d @ d / ed
    for _ in range(int(T / scale)):
        G = W0 + np.diag(b) @ Bf @ np.diag(d) @ Af - Wst
        gb = np.diag(G @ (Bf @ np.diag(d) @ Af).T)
        gd = np.diag((np.diag(b) @ Bf).T @ G @ Af.T)
        b, d = b - eb * gb, d - ed * gd
    return abs(b @ b / eb - d @ d / ed - Q0) / abs(Q0)
d1, d2 = vera_drift(1e-3), vera_drift(5e-4)
ok("Thm14 VeRA scalar charge: drift is O(eta)", 1.7 < d1 / d2 < 2.3, f"drift {d1:.3e} -> {d2:.3e}")

# Thm 18(b): Cartan-Dieudonne peeling, rank(R-I) reflections
n_ = 8
P, _ = np.linalg.qr(rng.normal(size=(n_, 4)))
G4, _ = np.linalg.qr(rng.normal(size=(4, 4)))
if np.linalg.det(G4) < 0:
    G4[:, 0] *= -1
R = np.eye(n_) - P @ P.T + P @ G4 @ P.T
Rc, refl = R.copy(), []
while rk(Rc - np.eye(n_)) > 0:
    _, _, Vt = np.linalg.svd(Rc - np.eye(n_)); v = Vt[0]
    Hw = H(Rc @ v - v); Rc = Hw @ Rc; refl.append(Hw)
prod = np.eye(n_)
for Hm in refl:
    prod = prod @ Hm
ok("Thm18 Cartan-Dieudonne", len(refl) == rk(R - np.eye(n_)) and np.allclose(prod, R), f"#refl={len(refl)}")
Rh = np.eye(n_)
for _ in range(4):
    Rh = Rh @ H(rng.normal(size=n_))
ok("Thm18 product of 4 reflections has rank(R-I)<=4", rk(Rh - np.eye(n_)) <= 4)

# Thm 19(c): GaLore (fixed P, Adam) == one-sided LoRA W0 + P A with Adam
m, n, r = 8, 6, 3
P, _ = np.linalg.qr(rng.normal(size=(m, r)))
X, Y = rng.normal(size=(20, n)), rng.normal(size=(20, m))
grad = lambda W: (X @ W.T - Y).T @ X / 20
Wb = rng.normal(size=(m, n)); b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-2
A = np.zeros((r, n)); mA = np.zeros_like(A); vA = np.zeros_like(A)
Wg = Wb.copy(); mG = np.zeros((r, n)); vG = np.zeros((r, n))
for t in range(1, 51):
    g = P.T @ grad(Wb + P @ A); mA = b1 * mA + (1 - b1) * g; vA = b2 * vA + (1 - b2) * g * g
    A = A - lr * (mA / (1 - b1 ** t)) / (np.sqrt(vA / (1 - b2 ** t)) + eps)
    R_ = P.T @ grad(Wg); mG = b1 * mG + (1 - b1) * R_; vG = b2 * vG + (1 - b2) * R_ * R_
    Wg = Wg - lr * P @ ((mG / (1 - b1 ** t)) / (np.sqrt(vG / (1 - b2 ** t)) + eps))
ok("Thm19 GaLore == one-sided LoRA (Adam)", np.allclose(Wg, Wb + P @ A, atol=1e-12))

# Thm 20 (R4): diagonal slides through the Hadamard spider of SwiGLU
dm, dff = 5, 7
Wg_, Wu_, Wd_ = rng.normal(size=(dff, dm)), rng.normal(size=(dff, dm)), rng.normal(size=(dm, dff))
l, x = rng.normal(size=dff), rng.normal(size=dm)
silu = lambda z: z / (1 + np.exp(-z))
ok("Thm20 SwiGLU spider slide",
   np.allclose(Wd_ @ (l * (silu(Wg_ @ x) * (Wu_ @ x))), Wd_ @ (silu(Wg_ @ x) * ((np.diag(l) @ Wu_) @ x))))
gelu = lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
lp = np.abs(l) + 0.5
ok("Thm20 positive diagonal does NOT slide through GELU",
   not np.allclose(lp * gelu(Wg_ @ x), gelu(np.diag(lp) @ Wg_ @ x)))
relu = lambda z: np.maximum(z, 0)
ok("Thm20 positive diagonal slides through ReLU", np.allclose(lp * relu(Wg_ @ x), relu(np.diag(lp) @ Wg_ @ x)))

# Thm 21: prefix = gated parallel adapter
dk = 5; q = rng.normal(size=dk)
Kc, Vc, Kp, Vp = rng.normal(size=(7, dk)), rng.normal(size=(7, 3)), rng.normal(size=(2, dk)), rng.normal(size=(2, 3))
def attn(q, K, V):
    s_ = np.exp(K @ q); return s_ @ V / s_.sum(), s_.sum(), s_ / s_.sum()
full, _, wfull = attn(q, np.vstack([Kp, Kc]), np.vstack([Vp, Vc]))
ac, Zc, wc = attn(q, Kc, Vc); ap, Zp, _ = attn(q, Kp, Vp); lam = Zp / (Zp + Zc)
ok("Thm21 prefix identity", np.allclose(full, (1 - lam) * ac + lam * ap))
ok("Thm21 content pattern rescaled by (1-lambda)", np.allclose(wfull[2:], (1 - lam) * wc))

# Thm 22: LoRAHub is gauge dependent, task arithmetic is not
m, n, r = 6, 5, 2
Bs = [rng.normal(size=(m, r)) for _ in range(2)]; As = [rng.normal(size=(r, n)) for _ in range(2)]
w = [0.6, 0.4]; g = rng.normal(size=(r, r))
Bs2, As2 = [Bs[0], Bs[1] @ np.linalg.inv(g)], [As[0], g @ As[1]]
hub = lambda Bs, As: sum(wi * Bi for wi, Bi in zip(w, Bs)) @ sum(wi * Ai for wi, Ai in zip(w, As))
ta = lambda Bs, As: sum(wi * Bi @ Ai for wi, Bi, Ai in zip(w, Bs, As))
ok("Thm22 task arithmetic gauge-invariant", np.allclose(ta(Bs, As), ta(Bs2, As2)))
ok("Thm22 LoRAHub gauge-dependent", not np.allclose(hub(Bs, As), hub(Bs2, As2)),
   f"change {np.linalg.norm(hub(Bs, As) - hub(Bs2, As2)):.3f}")

# Prop 23: torus orbit dimension m + n - c(W0); HiRA rank-1 multiplier identity
def orbit_dim(W):
    m, n = W.shape; J = []
    for i in range(m):
        E = np.zeros((m, m)); E[i, i] = 1; J.append((E @ W).ravel())
    for j in range(n):
        E = np.zeros((n, n)); E[j, j] = 1; J.append((W @ E).ravel())
    return rk(np.array(J).T)
W = rng.normal(size=(6, 5)); ok("Prop23 torus orbit (dense)", orbit_dim(W) == 10)
Wb2 = np.zeros((6, 5)); Wb2[:3, :2] = rng.normal(size=(3, 2)); Wb2[3:, 2:] = rng.normal(size=(3, 3))
ok("Prop23 torus orbit (2 components)", orbit_dim(Wb2) == 9)
a_, b_ = rng.normal(size=6), rng.normal(size=5)
ok("Prop23 (ab^T)∘W0 = diag(a) W0 diag(b)", np.allclose(np.outer(a_, b_) * W, np.diag(a_) @ W @ np.diag(b_)))
m, n, r0, r = 30, 30, 3, 4
W0r = rng.normal(size=(m, r0)) @ rng.normal(size=(r0, n))
ok("Thm11 HiRA cut bound r0*r attained", rk(W0r * (rng.normal(size=(m, r)) @ rng.normal(size=(r, n)))) == r0 * r)

# Thm 11: FourierFT rank <= 2 * (#coefficients)
S = np.zeros((64, 64), complex); idx = rng.choice(64 * 64, 10, replace=False); S.flat[idx] = rng.normal(size=10)
ok("Thm11 FourierFT rank <= 2 n_c", rk(np.real(np.fft.ifft2(S)), 1e-10) <= 20)

# Thm 18(a): LoRA-XS = intersection of the two one-sided LoRAs (dimension r^2)
m, n, r = 7, 6, 2
A0, B0 = rng.normal(size=(r, n)), rng.normal(size=(m, r))
left = np.array([(np.eye(m)[:, [i]] @ np.eye(r)[[k], :] @ A0).ravel() for i in range(m) for k in range(r)]).T
right = np.array([(B0 @ np.eye(r)[:, [k]] @ np.eye(n)[[j], :]).ravel() for k in range(r) for j in range(n)]).T
dim_sum = rk(np.hstack([left, right]))
ok("Thm18 inclusion-exclusion: dim(sum)=mr+nr-r^2", dim_sum == m * r + n * r - r * r)

# Prop 16: AdaLoRA P diag(lam) Q has image dimension r(m+n-r), so waste = r^2 + r (not just the 2r torus)
m, n, r = 7, 6, 2
P, lam, Q = rng.normal(size=(m, r)), rng.normal(size=r), rng.normal(size=(r, n))
eps_ = 1e-6; base = (P @ np.diag(lam) @ Q).ravel(); cols = []
for arr in (P, lam, Q):
    for idx in np.ndindex(arr.shape):
        arr[idx] += eps_; cols.append(((P @ np.diag(lam) @ Q).ravel() - base) / eps_); arr[idx] -= eps_
jr = np.linalg.matrix_rank(np.array(cols).T, tol=1e-4)
ok("Prop16 AdaLoRA image dim r(m+n-r), waste r^2+r", jr == r * (m + n - r) and (r * (m + n) + r - jr) == r * r + r,
   f"jac rank {jr}")
