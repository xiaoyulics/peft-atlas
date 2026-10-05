"""Beat II.5: rank is a cut, dimension is a volume (Theorem II.10, Proposition II.12).

At equal budget 2r(m + n) on one m x n weight, compare
  LoHa_{r,r}:  dW = (B1 A1) * (B2 A2)   (entrywise product; B_i in R^{m x r}, A_i in R^{r x n})
  LoRA_{2r}:   dW = B A                (B in R^{m x 2r}, A in R^{2r x n})
for m = n = 64 at r = 3 (the case on screen) and r = 2. For each method:
  rank       -- rank of dW at random parameter points (the largest seen);
  dimension  -- rank of the Jacobian of the parameter-to-weight map at random points (the largest seen), i.e. the
                number of independent directions the method can move in.
The paper's closed forms: rank min(r^2, m, n) for LoHa (the cut through both r-wires) and 2r for LoRA_{2r};
dimension 2r(m + n - 2r) for LoRA_{2r}, and at most 2r(m + n - r) - (m + n - 1) for LoHa_{r,r} (Prop II.12, an upper
bound, attained in the paper's numerical checks).

  video/.venv/bin/python video/sim/cut_volume.py       # prints the table and writes video/sim/out/cut_volume.json
"""
import json
import pathlib

import numpy as np

m = n = 64
rng = np.random.default_rng(5)
EPS = np.finfo(float).eps


def num_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    k = int((s > s.max() * max(M.shape) * EPS).sum())
    gap = float(s[k - 1] / s[k]) if k < s.size and s[k] > 0 else None
    return k, gap


def lora_jacobian(B, A):
    return np.hstack([np.kron(A.T, np.eye(m)), np.kron(np.eye(n), B)])          # column-major vec


def loha_jacobian(B1, A1, B2, A2):
    P1, P2 = (B1 @ A1).ravel(order="F"), (B2 @ A2).ravel(order="F")
    return np.hstack([P2[:, None] * lora_jacobian(B1, A1), P1[:, None] * lora_jacobian(B2, A2)])


def case(r, points=3):
    R = 2 * r
    lo_rank, lo_dim, ha_rank, ha_dim, gaps = 0, 0, 0, 0, []
    for _ in range(points):
        B, A = rng.normal(size=(m, R)), rng.normal(size=(R, n))
        B1, A1, B2, A2 = (rng.normal(size=(m, r)), rng.normal(size=(r, n)),
                          rng.normal(size=(m, r)), rng.normal(size=(r, n)))
        lo_rank = max(lo_rank, num_rank(B @ A)[0])
        ha_rank = max(ha_rank, num_rank((B1 @ A1) * (B2 @ A2))[0])
        (dl, gl), (dh, gh) = num_rank(lora_jacobian(B, A)), num_rank(loha_jacobian(B1, A1, B2, A2))
        lo_dim, ha_dim = max(lo_dim, dl), max(ha_dim, dh)
        gaps += [g for g in (gl, gh) if g is not None]
    res = dict(r=r, budget_loha=2 * r * (m + n), budget_lora=R * (m + n),
               loha=dict(rank=ha_rank, dim=ha_dim), lora=dict(rank=lo_rank, dim=lo_dim),
               paper=dict(loha_rank=min(r * r, m, n), lora_rank=R, lora_dim=R * (m + n - R),
                          loha_dim_bound=2 * r * (m + n - r) - (m + n - 1)),
               separates=2 * r * r < m + n - 1, min_singular_gap=min(gaps))
    print(f"r = {r}: budget LoHa_{r},{r} = {res['budget_loha']}, LoRA_{R} = {res['budget_lora']}")
    print(f"   rank       LoHa {ha_rank:4d}   LoRA {lo_rank:4d}    (paper: {min(r * r, m, n)} and {R})")
    print(f"   dimension  LoHa {ha_dim:4d}   LoRA {lo_dim:4d}    (paper: <= {res['paper']['loha_dim_bound']} "
          f"and {R * (m + n - R)}; 2r^2 < m + n - 1: {res['separates']})")
    return res


out = dict(m=m, n=n, cases={str(r): case(r) for r in (3, 2)})
dest = pathlib.Path(__file__).with_name("out")
dest.mkdir(exist_ok=True)
(dest / "cut_volume.json").write_text(json.dumps(out, indent=1))
