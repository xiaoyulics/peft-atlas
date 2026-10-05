"""Cold open and beat III.4: LoRA+ with ratio lam against LoRA with alpha * lam, under Adam (eps = 0).

Both start from B0 = 0 and the same A0, see the same full-batch loss and share one constant schedule; no weight decay,
no clipping. Corollary III.11 says the weight trajectories agree in exact arithmetic; for a power-of-two lam every
floating-point operation agrees too, so the difference is exactly 0.0. The control is LoRA+ against unrescaled LoRA.

  python3 video/sim/lora_plus.py            # prints the differences and writes video/sim/out/lora_plus.npz
"""
import pathlib

import numpy as np

m, n, r, alpha, lam, lr, steps = 12, 10, 2, 8.0, 16.0, 2e-3, 400
rng = np.random.default_rng(0)
W0 = rng.normal(size=(m, n)) / np.sqrt(n)
target = W0 + rng.normal(size=(m, 3)) @ rng.normal(size=(3, n)) / 3      # a rank-3 shift to learn
X = rng.normal(size=(n, 64))
A0 = rng.uniform(-1, 1, size=(r, n)) / np.sqrt(n)                          # Kaiming-uniform-like


def loss_grad(W):
    R = (W - target) @ X
    return 0.5 * np.sum(R * R) / X.shape[1], R @ X.T / X.shape[1]


def run(scale, lr_B, lr_A, b1=0.9, b2=0.999):
    B, A = np.zeros((m, r)), A0.copy()
    mB, vB, mA, vA = 0 * B, 0 * B, 0 * A, 0 * A
    Ws, Ls = [], []
    for t in range(1, steps + 1):
        W = W0 + scale * B @ A
        L, G = loss_grad(W)
        Ws.append(W); Ls.append(L)
        gB, gA = scale * G @ A.T, scale * B.T @ G
        for p, g, mm, vv, lr_p in ((B, gB, mB, vB, lr_B), (A, gA, mA, vA, lr_A)):
            mm *= b1; mm += (1 - b1) * g
            vv *= b2; vv += (1 - b2) * g * g
            mh, vh = mm / (1 - b1 ** t), vv / (1 - b2 ** t)
            with np.errstate(invalid="ignore", divide="ignore"):
                step = np.where(vh > 0, mh / np.sqrt(vh), 0.0)                # Adam with eps = 0
            p -= lr_p * step
    return np.array(Ws), np.array(Ls)


s = alpha / r
W_plus, L_plus = run(s, lam * lr, lr)              # LoRA+: B's learning rate lam times A's
W_alpha, L_alpha = run(lam * s, lr, lr)            # plain LoRA with alpha multiplied by lam
W_plain, L_plain = run(s, lr, lr)                  # control: plain LoRA, alpha unchanged

same = np.abs(W_plus - W_alpha).max()
ctrl = np.abs(W_plus - W_plain).max()
print(f"max |W_LoRA+ - W_LoRA(alpha*{lam:g})| over {steps} steps = {same!r}")
print(f"control: max |W_LoRA+ - W_LoRA(alpha)| = {ctrl:.3g}")
print(f"loss: start {L_plus[0]:.4f}, end LoRA+ {L_plus[-1]:.4f}, LoRA(alpha*lam) {L_alpha[-1]:.4f}, control {L_plain[-1]:.4f}")
out = pathlib.Path(__file__).with_name("out")
out.mkdir(exist_ok=True)
# a shared 2-D view of the trajectories (beats 0.1 and III.4): PCA of the updates W(t) - W0 of LoRA+ and the control
flat = np.concatenate([(W_plus - W0).reshape(steps, -1), (W_plain - W0).reshape(steps, -1)])
_, _, Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
proj = lambda W: (W - W0).reshape(len(W), -1) @ Vt[:2].T
P_plus, P_alpha, P_plain = proj(W_plus), proj(W_alpha), proj(W_plain)
np.savez(out / "lora_plus.npz", P_plus=P_plus, P_alpha=P_alpha, P_plain=P_plain, W_plus=W_plus, W_alpha=W_alpha, W_plain=W_plain,
         L_plus=L_plus, L_alpha=L_alpha, L_plain=L_plain, same=same, ctrl=ctrl, lam=lam)
