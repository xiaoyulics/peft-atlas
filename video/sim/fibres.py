"""Beats III.2 and III.3 on a small LoRA (m = n = 6, r = 2, zero-B start, full-batch quadratic loss).

III.2  The charge Phi = B^T B / eta_B - A A^T / eta_A (equal block rates here, so Phi = B^T B - A A^T):
       constant under gradient flow (RK45 at tight tolerance); after a fixed number of SGD steps its drift scales with
       the square of the step; under Adam it moves at first order.
III.3  Two zero-B starts A0 and g A0: SGD gives the same weights for a rotation g; Adam separates them for a rotation
       and keeps them together for a signed permutation (Thm III.12, Cor III.13).

  video/.venv/bin/python video/sim/fibres.py
"""
import pathlib

import numpy as np
from scipy.integrate import solve_ivp

m, n, r = 6, 6, 2
rng = np.random.default_rng(1)
W0 = rng.normal(size=(m, n)) / np.sqrt(n)
target = W0 + rng.normal(size=(m, 2)) @ rng.normal(size=(2, n)) / 2
X = rng.normal(size=(n, 32))
A0 = rng.uniform(-1, 1, size=(r, n)) / np.sqrt(n)


def grads(B, A):
    G = (W0 + B @ A - target) @ X @ X.T / X.shape[1]
    return G @ A.T, B.T @ G


def charge(B, A):
    return B.T @ B - A @ A.T


# ---------------------------------------------------------------- III.2: the charge
def flow(T):
    def rhs(_, y):
        B, A = y[:m * r].reshape(m, r), y[m * r:].reshape(r, n)
        gB, gA = grads(B, A)
        return -np.concatenate([gB.ravel(), gA.ravel()])
    y0 = np.concatenate([np.zeros(m * r), A0.ravel()])
    y = solve_ivp(rhs, (0, T), y0, rtol=1e-12, atol=1e-12).y[:, -1]
    return y[:m * r].reshape(m, r), y[m * r:].reshape(r, n)


def sgd(eta, steps):
    B, A = np.zeros((m, r)), A0.copy()
    for _ in range(steps):
        gB, gA = grads(B, A)
        B, A = B - eta * gB, A - eta * gA
    return B, A


def adam(eta, steps, A_init=A0, eps=0.0, b1=0.9, b2=0.999):
    B, A = np.zeros((m, r)), A_init.copy()
    st = [np.zeros_like(B), np.zeros_like(B), np.zeros_like(A), np.zeros_like(A)]
    traj = []
    for t in range(1, steps + 1):
        gB, gA = grads(B, A)
        new = []
        for p, g, mm, vv in ((B, gB, st[0], st[1]), (A, gA, st[2], st[3])):
            mm[:] = b1 * mm + (1 - b1) * g
            vv[:] = b2 * vv + (1 - b2) * g * g
            mh, vh = mm / (1 - b1 ** t), vv / (1 - b2 ** t)
            with np.errstate(invalid="ignore", divide="ignore"):
                step = np.where(vh > 0, mh / (np.sqrt(vh) + eps), 0.0)
            new.append(p - eta * step)
        B, A = new
        traj.append(W0 + B @ A)
    return B, A, np.array(traj)


phi0 = charge(np.zeros((m, r)), A0)
Bf, Af = flow(2.0)
print(f"gradient flow, t = 2:   |Phi - Phi0| = {np.abs(charge(Bf, Af) - phi0).max():.2e}")
for eta in (0.02, 0.01, 0.005):
    Bs, As = sgd(eta, 100)
    Ba, Aa, _ = adam(eta, 100)
    print(f"step {eta:<6} 100 steps:  SGD drift {np.abs(charge(Bs, As) - phi0).max():.3e}   "
          f"Adam drift {np.abs(charge(Ba, Aa) - phi0).max():.3e}")

# ---------------------------------------------------------------- III.3: what each optimizer sees
th = 0.7
rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
sperm = np.array([[0.0, -1.0], [-1.0, 0.0]])         # swap the channels, flip both signs: det -1, not a rotation


def sgd_traj(A_init, eta=0.01, steps=300):
    B, A = np.zeros((m, r)), A_init.copy()
    out = []
    for _ in range(steps):
        gB, gA = grads(B, A)
        B, A = B - eta * gB, A - eta * gA
        out.append(W0 + B @ A)
    return np.array(out)


base_sgd = sgd_traj(A0)
_, _, base_adam = adam(0.01, 300)
print(f"SGD,  g = rotation:            max |dW| = {np.abs(sgd_traj(rot @ A0) - base_sgd).max():.2e}")
print(f"Adam, g = rotation:            max |dW| = {np.abs(adam(0.01, 300, rot @ A0)[2] - base_adam).max():.2e}")
print(f"Adam, g = signed permutation:  max |dW| = {np.abs(adam(0.01, 300, sperm @ A0)[2] - base_adam).max():.2e}")
pathlib.Path(__file__).with_name("out").mkdir(exist_ok=True)

# ---------------------------------------------------------------- III.2, cleanly: one step from the same state
Bm, Am = flow(1.0)                                       # a generic mid-training state (B != 0)
phim = charge(Bm, Am)
gB, gA = grads(Bm, Am)
print("one step from the same state (drift of Phi):")
for eta in (0.02, 0.01, 0.005):
    d_sgd = np.abs(charge(Bm - eta * gB, Am - eta * gA) - phim).max()
    d_adam = np.abs(charge(Bm - eta * np.sign(gB), Am - eta * np.sign(gA)) - phim).max()   # Adam's first step from fresh state
    print(f"  step {eta:<6} SGD {d_sgd:.4e}   Adam {d_adam:.4e}")
