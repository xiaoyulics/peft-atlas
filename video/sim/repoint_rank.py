"""Beat II.4: the rank of reachable LoRA updates from random split initialisations (Theorem II.1).

One m x n weight, m = n = 8, LoRA rank r = 2. Each sample draws
  - a pretrained weight W0 (= theta_0) and a split initialisation (B0, A0), Gaussian, so rank B0 A0 = r;
    the frozen residual is W0 - B0 A0, so the model starts exactly at theta_0;
  - a reachable point theta = (W0 - B0 A0) + B A, with B A drawn from one stratum of the cone, rank B A = k,
    k uniform in {0, 1, ..., r} (B, A Gaussian with their last r - k channels zeroed);
and records rank(theta - theta_0) by SVD. Theorem II.1 says no update reaches rank above 2r = 4, and split
initialisations attain 2r when 2r <= min(m, n).

  video/.venv/bin/python video/sim/repoint_rank.py     # prints the histogram and writes video/sim/out/repoint_rank.json
"""
import json
import pathlib

import numpy as np

m, n, r = 8, 8, 2
inits, per_init = 2000, 5
rng = np.random.default_rng(4)

counts = np.zeros(min(m, n) + 1, dtype=int)
by_k = {k: np.zeros(min(m, n) + 1, dtype=int) for k in range(r + 1)}
min_gap = np.inf
for _ in range(inits):
    W0 = rng.normal(size=(m, n))
    B0, A0 = rng.normal(size=(m, r)), rng.normal(size=(r, n))
    assert np.linalg.matrix_rank(B0 @ A0) == r
    W_res = W0 - B0 @ A0                                   # frozen residual: the model starts at theta_0
    for _ in range(per_init):
        k = int(rng.integers(0, r + 1))
        B, A = rng.normal(size=(m, r)), rng.normal(size=(r, n))
        B[:, k:], A[k:, :] = 0.0, 0.0                       # rank B A = k
        theta = W_res + B @ A                               # a reachable point
        U = theta - W0                                      # the update, relative to theta_0
        s = np.linalg.svd(U, compute_uv=False)
        rk = int((s > s.max() * max(m, n) * np.finfo(float).eps).sum()) if s.max() > 0 else 0
        assert rk == np.linalg.matrix_rank(B @ A - B0 @ A0)  # same rank from the factors directly
        if rk < s.size:
            min_gap = min(min_gap, s[rk - 1] / max(s[rk], 1e-300))
        counts[rk] += 1
        by_k[k][rk] += 1

N = int(counts.sum())
print(f"m = n = {m}, r = {r}: {inits} random split initialisations x {per_init} reachable points = {N} samples")
for q, c in enumerate(counts):
    print(f"  rank {q}: {c:5d}  " + "#" * int(60 * c / counts.max()))
print(f"largest rank seen: {int(np.flatnonzero(counts).max())}; samples above 2r = {2 * r}: {int(counts[2 * r + 1:].sum())}")
print(f"smallest ratio between the last kept and the first dropped singular value: {min_gap:.2e}")
dest = pathlib.Path(__file__).with_name("out")
dest.mkdir(exist_ok=True)
(dest / "repoint_rank.json").write_text(json.dumps(dict(
    m=m, n=n, r=r, inits=inits, per_init=per_init, samples=N, counts=counts.tolist(),
    counts_by_rank_of_BA={str(k): v.tolist() for k, v in by_k.items()},
    max_rank=int(np.flatnonzero(counts).max()), above_2r=int(counts[2 * r + 1:].sum()),
    min_singular_gap=float(min_gap)), indent=1))
