"""Beat V.2: merge-and-restart (ReLoRA) reach grows by r per cycle (Thm V.3, corollary 1).

After K cycles the update is a sum of K rank-r products, so its rank is at most min(Kr, m, n), and a generic sum
attains it. We draw K fresh random rank-r products B_k A_k (m = n = 8, r = 2), sum them and take the numerical rank,
for K = 1..5, over several draws: the largest reachable rank steps 2, 4, 6, 8, 8 (r, 2r, 3r, 4r = n, then full).

  video/.venv/bin/python video/sim/relora.py      # prints the staircase, writes video/sim/out/relora.json
"""
import json
import pathlib

import numpy as np

m, n, r, K_MAX, DRAWS = 8, 8, 2, 5, 200
rng = np.random.default_rng(5)

ranks = np.zeros((DRAWS, K_MAX), dtype=int)
for d in range(DRAWS):
    delta = np.zeros((m, n))
    for k in range(K_MAX):
        delta += rng.normal(size=(m, r)) @ rng.normal(size=(r, n))      # one cycle: train, merge
        ranks[d, k] = np.linalg.matrix_rank(delta)

largest = ranks.max(0).tolist()                     # largest rank reached after K cycles
typical = [int(np.bincount(ranks[:, k]).argmax()) for k in range(K_MAX)]
bound = [min((k + 1) * r, m, n) for k in range(K_MAX)]
print(f"m = n = {m}, r = {r}, {DRAWS} draws")
for k in range(K_MAX):
    print(f"  K = {k + 1}:  largest rank {largest[k]}   (every draw: {sorted(set(ranks[:, k].tolist()))})"
          f"   bound min(Kr, m, n) = {bound[k]}")
assert largest == bound == typical
out = pathlib.Path(__file__).with_name("out")
out.mkdir(exist_ok=True)
(out / "relora.json").write_text(json.dumps(dict(m=m, n=n, r=r, K=list(range(1, K_MAX + 1)), largest_rank=largest,
                                                 bound=bound, draws=DRAWS), indent=1))
