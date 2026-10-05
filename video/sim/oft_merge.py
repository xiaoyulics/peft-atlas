"""Beat V.3: what repeated merging does to the singular values of a weight (Cor II.6, Thm V.3).

One weight W (32 x 32: a Gaussian bulk plus four stronger directions), three merge-and-restart schedules:

 (a) exact Cayley OFT on the input side, W <- W R with R = (I - Q)^{-1}(I + Q) block-diagonal (block size 8), a fresh
     small skew Q in every cycle, 50 cycles. R is orthogonal, so W W^T and every singular value stay fixed; in float64
     the largest change is at rounding level.
 (b) LoRA merges, W <- W + B A with a fresh small random rank-2 product in every cycle, 50 cycles. The spectrum moves.
 (c) HF PEFT's default Cayley-Neumann OFT (PEFT >= 0.18), R = (I + Q)(I + Q + Q^2 + Q^3) = C(Q)(I - Q^4), so
     R^T R = (I - Q^4)^2, a contraction for small Q. Block size 8, skew entries of about 0.05 (standard deviation 0.05),
     fresh in every cycle, 200 cycles, averaged over 8 draws of the generators (same W). Reported: the relative
     Frobenius change of the neuron Gram matrix K_out = W W^T after 1, 10, 50 and 200 merges. The paper's [Num] note
     to Thm V.3 reports 0.2 %, 1.3 %, 6 % and 22 %.

  video/.venv/bin/python video/sim/oft_merge.py      # prints the numbers, writes video/sim/out/oft_merge.npz
"""
import pathlib

import numpy as np

N, BLOCK, Q_STD = 32, 8, 0.05
CYCLES_AB, CYCLES_C, DRAWS = 50, 200, 8
LORA_RANK, LORA_STEP = 2, 0.07
MARKS = (1, 10, 50, 200)
PAPER = {1: 0.2, 10: 1.3, 50: 6.0, 200: 22.0}           # percent, [Num] in Thm V.3

rng = np.random.default_rng(3)
U, _ = np.linalg.qr(rng.normal(size=(N, 4)))
V, _ = np.linalg.qr(rng.normal(size=(N, 4)))
W0 = rng.normal(size=(N, N)) / np.sqrt(N) + U @ np.diag([2.6, 2.0, 1.5, 1.2]) @ V.T
I_B = np.eye(BLOCK)


def svals(W):
    return np.linalg.svd(W, compute_uv=False)


def skew(g):
    """A skew-symmetric BLOCK x BLOCK generator with independent N(0, Q_STD^2) entries above the diagonal."""
    T = np.triu(g.normal(scale=Q_STD, size=(BLOCK, BLOCK)), 1)
    return T - T.T


def cayley(Q):
    return np.linalg.solve(I_B - Q, I_B + Q)                       # (I - Q)^{-1}(I + Q), exactly orthogonal


def cayley_neumann(Q):
    return (I_B + Q) @ (I_B + Q + Q @ Q + Q @ Q @ Q)              # HF PEFT default (use_cayley_neumann=True)


def block_diag(blocks):
    R = np.zeros((N, N))
    for k, b in enumerate(blocks):
        R[k * BLOCK:(k + 1) * BLOCK, k * BLOCK:(k + 1) * BLOCK] = b
    return R


def merge_run(W, factor, cycles, g):
    """Input-side merges W <- W R with fresh block generators; returns the singular values and K_out changes."""
    K0 = W @ W.T
    sig, dK, loewner = [svals(W)], [0.0], -np.inf
    for _ in range(cycles):
        Wn = W @ block_diag([factor(skew(g)) for _ in range(N // BLOCK)])
        loewner = max(loewner, np.linalg.eigvalsh(Wn @ Wn.T - W @ W.T).max())
        W = Wn
        sig.append(svals(W))
        dK.append(np.linalg.norm(W @ W.T - K0) / np.linalg.norm(K0))
    return np.array(sig), np.array(dK), loewner


sig0 = svals(W0)

# (a) exact Cayley OFT
sig_a, dK_a, _ = merge_run(W0, cayley, CYCLES_AB, np.random.default_rng(11))
run_a = np.maximum.accumulate(np.abs(sig_a - sig0).max(1))

# (b) LoRA merges
g = np.random.default_rng(12)
W, sig_b = W0.copy(), [sig0]
for _ in range(CYCLES_AB):
    W = W + (LORA_STEP * g.normal(size=(N, LORA_RANK))) @ (g.normal(size=(LORA_RANK, N)) / np.sqrt(N))
    sig_b.append(svals(W))
sig_b = np.array(sig_b)
run_b = np.maximum.accumulate(np.abs(sig_b - sig0).max(1))

# (c) HF default Cayley-Neumann OFT, averaged over draws of the generators
runs = [merge_run(W0, cayley_neumann, CYCLES_C, np.random.default_rng(100 + d)) for d in range(DRAWS)]
sig_c = np.mean([s for s, _, _ in runs], axis=0)
dK_all = np.array([k for _, k, _ in runs])
dK_c, dK_sd = dK_all.mean(0), dK_all.std(0)
loewner_c = max(l for _, _, l in runs)

print(f"W: {N} x {N}, block size {BLOCK}, skew entries N(0, {Q_STD}^2), sigma_max {sig0[0]:.3f}, sigma_min {sig0[-1]:.3f}")
print(f"(a) exact Cayley OFT, {CYCLES_AB} merges: max |d sigma| = {run_a[-1]:.2e}, "
      f"max |dK_out|/|K_out| = {dK_a.max():.2e}")
print(f"(b) LoRA (rank {LORA_RANK}) merges, {CYCLES_AB} merges: max |d sigma| = {run_b[-1]:.3f}")
print(f"(c) HF Cayley-Neumann OFT, mean of {DRAWS} draws: relative Frobenius change of K_out = W W^T")
for c in MARKS:
    print(f"    after {c:>3} merges: {100 * dK_c[c]:6.2f} % (sd {100 * dK_sd[c]:.2f})   paper: {PAPER[c]:g} %")
print(f"    largest eigenvalue of K_out(t+1) - K_out(t), any merge and draw: {loewner_c:.2e} (<= 0: Loewner decrease)")
print(f"    singular values after {CYCLES_C} merges: mean ratio {np.mean(sig_c[-1] / sig0):.4f}")

# cross-check in the paper's toy shape (6 x 8, one block), same generator law
toy = []
for d in range(DRAWS):
    gt = np.random.default_rng(200 + d)
    Wt = gt.normal(size=(6, BLOCK))
    K0 = Wt @ Wt.T
    row = {}
    for c in range(1, CYCLES_C + 1):
        Wt = Wt @ cayley_neumann(skew(gt))
        if c in MARKS:
            row[c] = np.linalg.norm(Wt @ Wt.T - K0) / np.linalg.norm(K0)
    toy.append(row)
print("    cross-check, 6 x 8 Gaussian W, one block:",
      ", ".join(f"{c}: {100 * np.mean([t[c] for t in toy]):.2f} %" for c in MARKS))

out = pathlib.Path(__file__).with_name("out")
out.mkdir(exist_ok=True)
np.savez(out / "oft_merge.npz", sig0=sig0, sig_a=sig_a, run_a=run_a, sig_b=sig_b, run_b=run_b, sig_c=sig_c, dK_c=dK_c,
         dK_sd=dK_sd, marks=np.array(MARKS), dK_marks=np.array([dK_c[c] for c in MARKS]),
         paper=np.array([PAPER[c] for c in MARKS]), meta=np.array([N, BLOCK, Q_STD, DRAWS, LORA_RANK, LORA_STEP]))
