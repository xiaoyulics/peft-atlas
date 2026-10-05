"""Beat II.3: first-order directions of LoRA at three initialisations (Theorem II.2(b), Proposition II.3).

LoRA's map is kappa(B, A) = W_res + B A on one m x n weight, with B in R^{m x r} and A in R^{r x n}. Its first-order
image at a pointing (B0, A0) is S_1 = { dB A0 + B0 dA }, and dim S_1 is the rank of the Jacobian of kappa there.
We build that Jacobian explicitly (vec(dB A0) = (A0^T kron I_m) vec(dB), vec(B0 dA) = (I_n kron B0) vec(dA)), check
it against central finite differences, and take its rank by SVD at

  zero-B init   B0 = 0, A0 random of rank r        (stratum T_A: the apex)
  zero-A init   A0 = 0, B0 random of rank r        (stratum T_B: the apex)
  split init    B0, A0 random, rank B0 A0 = r       (stratum S: a smooth point)

and the image dimension d, as the formula r(m + n - r) and as the largest Jacobian rank over random points.
The paper reports dim S_1 = (14, 10, 20) at (m, n, r) = (7, 5, 2).

  video/.venv/bin/python video/sim/first_order.py      # prints the ranks and writes video/sim/out/first_order.json
"""
import json
import pathlib

import numpy as np

m, n, r = 7, 5, 2
rng = np.random.default_rng(2)


def jacobian(B0, A0):
    """d vec(B A) / d (vec B, vec A) at (B0, A0); column-major vec, shape (m n) x (m r + r n)."""
    JB = np.kron(A0.T, np.eye(m))            # vec(dB A0)
    JA = np.kron(np.eye(n), B0)              # vec(B0 dA)
    return np.hstack([JB, JA])


def jacobian_fd(B0, A0, h=1e-3):
    """The same Jacobian by central differences (exact up to rounding: the map is bilinear)."""
    q0 = np.concatenate([B0.ravel(order="F"), A0.ravel(order="F")])

    def f(q):
        B = q[:m * r].reshape((m, r), order="F")
        A = q[m * r:].reshape((r, n), order="F")
        return (B @ A).ravel(order="F")
    cols = []
    for k in range(q0.size):
        e = np.zeros_like(q0)
        e[k] = h
        cols.append((f(q0 + e) - f(q0 - e)) / (2 * h))
    return np.stack(cols, axis=1)


def rank(J, rtol=None):
    """Numerical rank by SVD (numpy's default tolerance unless rtol is given), the smallest kept singular value, and
    the gap between it and the largest discarded one."""
    s = np.linalg.svd(J, compute_uv=False)
    tol = s.max() * (rtol if rtol is not None else max(J.shape) * np.finfo(float).eps)
    k = int((s > tol).sum())
    gap = float(s[k - 1] / s[k]) if k < s.size and s[k] > 0 else None    # None: the discarded values are exactly 0
    return k, float(s[k - 1]), gap


inits = {
    "zero_B": (np.zeros((m, r)), rng.normal(size=(r, n))),
    "zero_A": (rng.normal(size=(m, r)), np.zeros((r, n))),
    "split": (rng.normal(size=(m, r)), rng.normal(size=(r, n))),
}
out = dict(m=m, n=n, r=r, dimS1={}, check={})
for name, (B0, A0) in inits.items():
    J = jacobian(B0, A0)
    k, smin, gap = rank(J)
    k_fd, _, _ = rank(jacobian_fd(B0, A0), rtol=1e-8)      # differences carry ~1e-13 rounding noise
    assert k == k_fd, (name, k, k_fd)
    out["dimS1"][name] = k
    out["check"][name] = dict(finite_difference_rank=k_fd, smallest_kept_singular_value=smin, singular_gap=gap,
                              rank_B0=int(np.linalg.matrix_rank(B0)), rank_A0=int(np.linalg.matrix_rank(A0)),
                              rank_B0A0=int(np.linalg.matrix_rank(B0 @ A0)))
    print(f"{name:7s} dim S_1 = {k:2d}   (finite differences: {k_fd}; rank B0 = {np.linalg.matrix_rank(B0)}, "
          f"rank A0 = {np.linalg.matrix_rank(A0)})")

generic = max(rank(jacobian(rng.normal(size=(m, r)), rng.normal(size=(r, n))))[0] for _ in range(20))
out["image_dim_formula"] = r * (m + n - r)
out["image_dim_generic_jacobian_rank"] = generic
out["formulas"] = dict(zero_B=m * r, zero_A=n * r, split=r * (m + n - r))
out["paper"] = dict(zero_B=14, zero_A=10, split=20)
out["agrees_with_paper"] = all(out["dimS1"][k] == out["paper"][k] for k in out["paper"])
print(f"image dimension r(m + n - r) = {r * (m + n - r)}; largest Jacobian rank over 20 random points = {generic}")
print(f"paper: dim S_1 = (14, 10, 20); agrees: {out['agrees_with_paper']}")

dest = pathlib.Path(__file__).with_name("out")
dest.mkdir(exist_ok=True)
(dest / "first_order.json").write_text(json.dumps(out, indent=1))
