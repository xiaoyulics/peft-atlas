"""Data for beats III.1 to III.3 (video/film/ch3_fibres.py), from the setup of video/sim/fibres.py.

fibres.py is executed as a module with its printing silenced, so the matrices, the loss and the optimizers are its own
(m = n = 6, r = 2, zero-B start, full-batch quadratic loss); its printout is unchanged.

III.1  The gauge. The mid-training LoRA state (B, A) = flow(1.0) of fibres.py and a path g(u) in GL_2, u in [0, 1]
       (a positive scalar times a rotation times expm(s M)). Along the path: B g^-1, g A, their product and
       max |(B g^-1)(g A) - BA|, which stays at machine precision.
III.2  The charge Phi = B^T B - A A^T (equal block rates). Gradient flow from the zero-B start with the integrator call of
       fibres.py (RK45, rtol = atol = 1e-12, t in [0, 2]); with dense output on, the accepted steps are the same, so the
       final drift is fibres.py's 8.2e-13. Every accepted step is kept, with B, A, Phi and the drift |Phi - Phi_0|.
       Then one step from the same mid-training state flow(1.0) at step sizes 0.02, 0.01, 0.005: plain SGD (drift
       falls 4x per halving) and Adam's first step from fresh state, a sign step (drift falls 2x).
III.3  What each optimizer sees. Weight trajectories W(t), 300 steps at step size 0.01, from A0 and from g A0:
       SGD with the rotation, Adam with the rotation, Adam with the signed permutation (swap the channels, flip both
       signs; det -1). Gaps max_t max |dW| and 2-D projections (PCA of the updates W(t) - W0, one basis for the SGD
       row and one shared by the two Adam rows), with W0 itself prepended as step 0.

  video/.venv/bin/python video/sim/fibres_export.py      # prints the numbers and writes video/sim/out/fibres_export.npz
"""
import contextlib
import importlib.util
import io
import pathlib

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fibres", HERE / "fibres.py")
F = importlib.util.module_from_spec(_spec)
with contextlib.redirect_stdout(io.StringIO()):
    _spec.loader.exec_module(F)
m, n, r = F.m, F.n, F.r

# ---------------------------------------------------------------- III.1: the gauge
B1, A1 = F.flow(1.0)                                  # a generic mid-training state, B != 0
M = np.array([[0.9, 0.5], [0.2, -0.9]])               # a traceless generator (stretch and shear)
U = np.linspace(0.0, 1.0, 1201)


def gauge(u):
    """g(u) in GL_2: a positive scalar times a rotation times expm(s M); g(0) = I."""
    th = 1.25 * np.pi * u
    s = 1.1 * np.sin(np.pi * u)
    k = 0.45 * np.sin(2 * np.pi * u)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.exp(k) * R @ expm(s * M)


BA1 = B1 @ A1
G1 = np.array([gauge(u) for u in U])
Bg = np.array([B1 @ np.linalg.inv(g) for g in G1])
gA = np.array([g @ A1 for g in G1])
prod = np.einsum("kir,krj->kij", Bg, gA)
dBA = np.abs(prod - BA1).max(axis=(1, 2))
detg = np.linalg.det(G1)
print(f"III.1  gauge path: {len(U)} samples, det g in [{detg.min():.3f}, {detg.max():.3f}], "
      f"max |(B g^-1)(g A) - BA| over the path = {dBA.max():.2e} (median {np.median(dBA):.2e})")


# ---------------------------------------------------------------- III.2: the charge under gradient flow
def rhs(_, y):
    B, A = y[:m * r].reshape(m, r), y[m * r:].reshape(r, n)
    gB, gA_ = F.grads(B, A)
    return -np.concatenate([gB.ravel(), gA_.ravel()])


y0 = np.concatenate([np.zeros(m * r), F.A0.ravel()])
sol = solve_ivp(rhs, (0, 2.0), y0, rtol=1e-12, atol=1e-12, dense_output=True)
Bf, Af = F.flow(2.0)
assert np.array_equal(sol.y[:m * r, -1].reshape(m, r), Bf) and np.array_equal(sol.y[m * r:, -1].reshape(r, n), Af)
flow_t = sol.t
flow_B = sol.y[:m * r].T.reshape(-1, m, r)
flow_A = sol.y[m * r:].T.reshape(-1, r, n)
phi0 = F.charge(np.zeros((m, r)), F.A0)
flow_phi = np.array([F.charge(B, A) for B, A in zip(flow_B, flow_A)])
flow_drift = np.abs(flow_phi - phi0).max(axis=(1, 2))
flow_loss = np.array([0.5 * np.sum(((F.W0 + B @ A - F.target) @ F.X) ** 2) / F.X.shape[1] for B, A in zip(flow_B, flow_A)])
print(f"III.2  gradient flow: {len(flow_t)} accepted RK45 steps on [0, 2]; |Phi(2) - Phi0| = {flow_drift[-1]:.2e}, "
      f"largest along the way {flow_drift.max():.2e}; loss {flow_loss[0]:.3f} -> {flow_loss[-1]:.3f}")

Bm, Am = F.flow(1.0)
phim = F.charge(Bm, Am)
gB, gA_ = F.grads(Bm, Am)
etas = np.array([0.02, 0.01, 0.005])
d_sgd = np.array([np.abs(F.charge(Bm - e * gB, Am - e * gA_) - phim).max() for e in etas])
d_adam = np.array([np.abs(F.charge(Bm - e * np.sign(gB), Am - e * np.sign(gA_)) - phim).max() for e in etas])
for e, a, b in zip(etas, d_sgd, d_adam):
    print(f"       one step, step {e:<6} SGD {a:.4e}   Adam {b:.4e}")
print(f"       ratios per halving: SGD {d_sgd[0] / d_sgd[1]:.3f}, {d_sgd[1] / d_sgd[2]:.3f};  "
      f"Adam {d_adam[0] / d_adam[1]:.3f}, {d_adam[1] / d_adam[2]:.3f}")

# ---------------------------------------------------------------- III.3: what each optimizer sees
W0 = F.W0
sgd_base = np.concatenate([W0[None], F.base_sgd])
sgd_rot = np.concatenate([W0[None], F.sgd_traj(F.rot @ F.A0)])
adam_base = np.concatenate([W0[None], F.base_adam])
adam_rot = np.concatenate([W0[None], F.adam(0.01, 300, F.rot @ F.A0)[2]])
adam_perm = np.concatenate([W0[None], F.adam(0.01, 300, F.sperm @ F.A0)[2]])
gap_sgd_rot = np.abs(sgd_rot - sgd_base).max()
gap_adam_rot = np.abs(adam_rot - adam_base).max()
gap_adam_perm = np.abs(adam_perm - adam_base).max()
run_sgd_rot = np.maximum.accumulate(np.abs(sgd_rot - sgd_base).max(axis=(1, 2)))
run_adam_rot = np.maximum.accumulate(np.abs(adam_rot - adam_base).max(axis=(1, 2)))
run_adam_perm = np.maximum.accumulate(np.abs(adam_perm - adam_base).max(axis=(1, 2)))
print(f"III.3  SGD,  rotation:            max |dW| = {gap_sgd_rot:.2e}")
print(f"       Adam, rotation:            max |dW| = {gap_adam_rot:.2e}")
print(f"       Adam, signed permutation:  max |dW| = {gap_adam_perm:.2e}   (det = {np.linalg.det(F.sperm):+.0f})")


def basis(*trajs):
    flat = np.concatenate([(T - W0).reshape(len(T), -1) for T in trajs])
    _, _, Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
    return Vt[:2]


V_sgd, V_adam = basis(sgd_base), basis(adam_base, adam_rot)
proj = lambda T, V: (T - W0).reshape(len(T), -1) @ V.T

out = HERE / "out"
out.mkdir(exist_ok=True)
np.savez(out / "fibres_export.npz",
         # III.1
         B1=B1, A1=A1, BA1=BA1, U=U, G1=G1, Bg=Bg, gA=gA, prod=prod, dBA=dBA,
         # III.2
         flow_t=flow_t, flow_B=flow_B, flow_A=flow_A, flow_phi=flow_phi, flow_drift=flow_drift, flow_loss=flow_loss,
         phi0=phi0, etas=etas, d_sgd=d_sgd, d_adam=d_adam,
         # III.3
         rot=F.rot, sperm=F.sperm,
         P_sgd_base=proj(sgd_base, V_sgd), P_sgd_rot=proj(sgd_rot, V_sgd),
         P_adam_base=proj(adam_base, V_adam), P_adam_rot=proj(adam_rot, V_adam), P_adam_perm=proj(adam_perm, V_adam),
         gap_sgd_rot=gap_sgd_rot, gap_adam_rot=gap_adam_rot, gap_adam_perm=gap_adam_perm,
         run_sgd_rot=run_sgd_rot, run_adam_rot=run_adam_rot, run_adam_perm=run_adam_perm)
print(f"wrote {out / 'fibres_export.npz'}")
