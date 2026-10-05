"""Beat IV.1: GaLore against one-sided LoRA with GaLore's projector as the frozen factor (Prop IV.2).

Least squares L(W) = |W X - Y|^2 / 2N on W in R^{m x n} with m = 10 < n = 12, so GaLore projects on the left (as its
paper's algorithm and galore_torch both do for m < n). Projector rank r = 2, period T = 25 steps, K = 4 periods, one
constant learning rate, Adam (beta = 0.9, 0.999; the same eps = 1e-8 and the same implementation on both sides), no
weight decay, no clipping, float64.

  GaLore    P_k = the top-r left singular vectors of the gradient at the start of period k (signs fixed: the
            largest entry of each column positive). Each step: R = P_k^T G, W <- W - lr P_k adam(R). The Adam state
            (moments and step counter) is carried across periods unchanged, as GaLore does.
  LoRA      one-sided LoRA W = W_k + P_k C: P_k computed the same way from its own gradient at W_k, C = 0 at the start
            of each period and trained by the same Adam on its gradient P_k^T G; merged (W_k <- W_k + P_k C) and
            restarted every period, with C's Adam state carried over.
  control   the LoRA procedure with its Adam state reset at every period, as ReLoRA does.

Prop IV.2 says GaLore and LoRA give the same W at every step in exact arithmetic; in float64 they agree to round-off.

  video/.venv/bin/python video/sim/galore.py      # prints the differences and writes video/sim/out/galore.npz
"""
import pathlib

import numpy as np

m, n, r, N = 10, 12, 2, 32
T, K, lr = 25, 4, 0.01
b1, b2, eps = 0.9, 0.999, 1e-8
rng = np.random.default_rng(7)
W0 = rng.normal(size=(m, n)) / np.sqrt(n)
W_star = W0 + 0.6 * rng.normal(size=(m, n)) / np.sqrt(n)        # a full-rank shift to learn
X = rng.normal(size=(n, N))
Y = W_star @ X


def loss(W):
    R = W @ X - Y
    return 0.5 * np.sum(R * R) / N


def grad(W):
    return (W @ X - Y) @ X.T / N


class Adam:
    """Bias-corrected Adam acting on whatever gradient it is fed (the same code on both sides)."""

    def __init__(self, shape):
        self.m, self.v, self.t = np.zeros(shape), np.zeros(shape), 0

    def __call__(self, g):
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        mh, vh = self.m / (1 - b1 ** self.t), self.v / (1 - b2 ** self.t)
        return mh / (np.sqrt(vh) + eps)


def projector(G):
    U = np.linalg.svd(G)[0][:, :r]
    signs = np.sign(U[np.abs(U).argmax(axis=0), np.arange(r)])
    return U * signs


def galore():
    W, opt, traj, Ps = W0.copy(), Adam((r, n)), [W0.copy()], []
    for _ in range(K):
        P = projector(grad(W))
        Ps.append(P)
        for _ in range(T):
            W = W - lr * P @ opt(P.T @ grad(W))
            traj.append(W)
    return np.array(traj), np.array(Ps)


def lora(reset=False, given_P=None):
    Wk, opt, traj, Cs, Ps = W0.copy(), Adam((r, n)), [W0.copy()], [np.zeros((r, n))], []
    for k in range(K):
        P = projector(grad(Wk)) if given_P is None else given_P[k]
        Ps.append(P)
        if reset:
            opt = Adam((r, n))
        C = np.zeros((r, n))
        for _ in range(T):
            C = C - lr * opt(P.T @ grad(Wk + P @ C))
            traj.append(Wk + P @ C)
            Cs.append(C)
        Wk = Wk + P @ C                                         # merge, then restart with C = 0
    return np.array(traj), np.array(Ps), np.array(Cs)


W_gal, P_gal = galore()
W_lora, P_lora, C_lora = lora()
W_reset, P_reset, C_reset = lora(reset=True)
W_lora_sharedP = lora(given_P=P_gal)[0]                         # variant: LoRA handed GaLore's own P_k

d_same = np.abs(W_gal - W_lora).max(axis=(1, 2))               # entrywise max |W_GaLore - W_LoRA| at each step
d_reset = np.abs(W_gal - W_reset).max(axis=(1, 2))
f_same = np.linalg.norm(W_gal - W_lora, axis=(1, 2))           # Frobenius, as the paper's site reports it
f_reset = np.linalg.norm(W_gal - W_reset, axis=(1, 2))
same, reset = d_same.max(), d_reset.max()
print(f"GaLore vs one-sided LoRA (state carried), max over {K * T} steps of max|dW|: {same:.2e}  "
      f"(first period {d_same[:T + 1].max():.2e}; Frobenius {f_same.max():.2e})")
print(f"  with GaLore's own P_k handed to the LoRA side: {np.abs(W_gal - W_lora_sharedP).max():.2e}")
print(f"  projectors agree to {np.abs(P_gal - P_lora).max():.2e}")
print(f"control (LoRA state reset every period): max|dW| = {reset:.3g}  (Frobenius {f_reset.max():.3g}); "
      f"first period {d_reset[:T + 1].max():.2e}")
mv = np.abs(W_gal[-1] - W0).max()
print(f"total movement max|W_100 - W_0| = {mv:.3f} (Frobenius {np.linalg.norm(W_gal[-1] - W0):.3f}); "
      f"loss {loss(W0):.3f} -> GaLore {loss(W_gal[-1]):.4f}, LoRA {loss(W_lora[-1]):.4f}, control {loss(W_reset[-1]):.4f}")

# the first step, piece by piece (for the opening of the beat)
G0 = grad(W0)
P0 = projector(G0)
R0 = P0.T @ G0
U0 = Adam((r, n))(R0)
dW0 = -lr * P0 @ U0

out = pathlib.Path(__file__).with_name("out")
out.mkdir(exist_ok=True)
np.savez(out / "galore.npz", W0=W0, W_gal=W_gal, W_lora=W_lora, W_reset=W_reset, P_gal=P_gal, P_lora=P_lora,
         C_lora=C_lora, C_reset=C_reset, d_same=d_same, d_reset=d_reset, f_same=f_same, f_reset=f_reset,
         same=same, reset=reset, G0=G0, P0=P0, R0=R0, U0=U0, dW0=dW0,
         L_gal=np.array([loss(W) for W in W_gal]), L_lora=np.array([loss(W) for W in W_lora]),
         L_reset=np.array([loss(W) for W in W_reset]), m=m, n=n, r=r, T=T, K=K, lr=lr, eps=eps)
print(f"wrote {out / 'galore.npz'}")
