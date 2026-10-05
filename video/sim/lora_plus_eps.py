"""Cold-open footnote: the LoRA+ / LoRA(alpha * lam) coincidence with stock Adam (eps > 0).

Same problem, seed, data and schedule as video/sim/lora_plus.py (copied here so that script's output is untouched).
LoRA+ runs stock Adam with the same eps on both blocks. The rescaled run, plain LoRA with alpha * lam, must then use
eps on A and lam * eps on B (Cor III.11: the per-block eps is rescaled with the block). For a power-of-two lam every
floating-point operation agrees, so the difference is exactly 0.0; with eps left unscaled on B it is small but not 0.

  video/.venv/bin/python video/sim/lora_plus_eps.py     # prints both differences, writes video/sim/out/lora_plus_eps.json
"""
import json
import pathlib

import numpy as np

m, n, r, alpha, lam, lr, steps = 12, 10, 2, 8.0, 16.0, 2e-3, 400
eps = 1e-8                                                                 # stock Adam default
rng = np.random.default_rng(0)
W0 = rng.normal(size=(m, n)) / np.sqrt(n)
target = W0 + rng.normal(size=(m, 3)) @ rng.normal(size=(3, n)) / 3
X = rng.normal(size=(n, 64))
A0 = rng.uniform(-1, 1, size=(r, n)) / np.sqrt(n)


def loss_grad(W):
    R = (W - target) @ X
    return 0.5 * np.sum(R * R) / X.shape[1], R @ X.T / X.shape[1]


def run(scale, lr_B, lr_A, eps_B, eps_A, b1=0.9, b2=0.999):
    B, A = np.zeros((m, r)), A0.copy()
    mB, vB, mA, vA = 0 * B, 0 * B, 0 * A, 0 * A
    Ws = []
    for t in range(1, steps + 1):
        W = W0 + scale * B @ A
        _, G = loss_grad(W)
        Ws.append(W)
        gB, gA = scale * G @ A.T, scale * B.T @ G
        for p, g, mm, vv, lr_p, e in ((B, gB, mB, vB, lr_B, eps_B), (A, gA, mA, vA, lr_A, eps_A)):
            mm *= b1; mm += (1 - b1) * g
            vv *= b2; vv += (1 - b2) * g * g
            mh, vh = mm / (1 - b1 ** t), vv / (1 - b2 ** t)
            p -= lr_p * (mh / (np.sqrt(vh) + e))                           # stock Adam
    return np.array(Ws)


s = alpha / r
W_plus = run(s, lam * lr, lr, eps, eps)                  # LoRA+, stock Adam: eps on both blocks
W_scaled = run(lam * s, lr, lr, lam * eps, eps)          # LoRA, alpha * lam: eps on A, lam * eps on B
W_unscaled = run(lam * s, lr, lr, eps, eps)              # LoRA, alpha * lam, eps left unscaled on B
same = float(np.abs(W_plus - W_scaled).max())
off = float(np.abs(W_plus - W_unscaled).max())
print(f"stock Adam (eps = {eps:g}): max |W_LoRA+ - W_LoRA(alpha*{lam:g})| with {lam:g}*eps on B = {same!r}")
print(f"  with eps unscaled on B: {off:.3g}")
out = pathlib.Path(__file__).with_name("out")
out.mkdir(exist_ok=True)
(out / "lora_plus_eps.json").write_text(json.dumps(dict(eps=eps, lam=lam, steps=steps, same=same, unscaled=off),
                                                   indent=1))
