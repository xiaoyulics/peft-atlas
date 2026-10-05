"""Figure "gauge" (Expose III): who sees the gauge? (Thm III.12, Prop III.4, Prop V.4.)

A port of site/js/fig-gauge.js (the gauge lab) and of checks 2, 7, 12k, 12l and 13o of theory/framework_checks.py.

(a) Optimizers. Fixed seeded LoRA factors B in R^{5x2}, A in R^{2x4} and a loss gradient G in R^{5x4} (site seed 1,
    s = eta = 1). A gauge g in GL_2 replaces (B, A) by (Bg^{-1}, gA); the weight W = W0 + sBA does not change, and the
    factor gradients become (grad_B g^T, g^{-T} grad_A). For each update rule we compute the first-order weight update
    dW = dB A + B dA at (B, A) and at (Bg^{-1}, gA) and report the relative gauge defect
        ||dW(g) - dW(I)||_F / ||dW(I)||_F.
    Rules (as on the site):
      SGD                       dB = grad_B, dA = grad_A                                       O(2)   Thm III.12(a)
      Adam                      first step from zero state, bias-corrected, eps = 0: sign descent  B_2    Thm III.12(b)
      Adam, per-channel rates   the same at (Bg^{-1}, gA) with eta_{B,j} = 1/r_j, eta_{A,j} = r_j,
                                r_j = ||g_{j,:}|| (= d_j for a positive diagonal g = diag(d))     Mon_2 jointly, Thm III.12(b)
      scaled GD, damped         dB = grad_B (AA^T + 0.1 I)^{-1}, dA = (B^T B + 0.1 I)^{-1} grad_A  O(2)   Thm III.12
      scaled GD, undamped       delta = 0: dW = P_U G + G P_V                                   GL_2   Prop III.4(a), III.12(c)
      LoRA-Pro                  Wang et al. 2024 with the Sylvester-optimal free matrix X:
                                dW = P_U G + G P_V - P_U G P_V                                  GL_2   Prop III.4(b)
    Gauges (the site's presets): a rotation by 30 degrees, the stretch diag(e^0.5, e^-0.4), the signed permutation
    [[0, 1], [1, 0]] diag(-1, 1), and a generic g = R_25 diag(e^0.4, e^-0.3) [[1, 0.35], [0, 1]].

(b) Merges (Prop V.4). N = 3 seeded modules B_i in R^{6x2}, A_i in R^{2x8} (site seed 4), weights w = (0.6, 0.5, 0.4).
    Module 2 is re-gauged by g = c^{-1} I, i.e. (B_2, A_2) -> (c B_2, A_2 / c), for log2 c in [-3, 3]. Task arithmetic
    sum_i w_i B_i A_i does not change; LoraHub (sum_i w_i B_i)(sum_j w_j A_j) changes by
    sum_{l != 2} w_l w_2 [(c - 1) B_2 A_l + (c^{-1} - 1) B_l A_2]. Both merges are recomputed from the factors at every c,
    and the closed form is evaluated separately and checked against the recomputation.

Run from paper/figures/src:  python3 fig_gauge.py
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import apply_style, C, W1, save

M32 = 0xFFFFFFFF
SEED_A, SEED_B = 1, 4
MA, NA, RK = 5, 4, 2
MB, NB, NMOD = 6, 8, 3
WTS = [0.6, 0.5, 0.4]
DELTA = 0.1
TOL = 1e-10                     # relative defect below this = equivariant up to round-off


# ---------------------------------------------------------------------------------------------------------------
# the site's seeded generator (site/js/core.js: Atlas.rng, mulberry32 + Box-Muller; LA.randn fills row by row)
# ---------------------------------------------------------------------------------------------------------------
class Rng:
    def __init__(self, seed):
        self.a = (seed & M32) or 1

    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & M32
        a = self.a
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    def normal(self):
        u = 0.0
        while u == 0:
            u = self()
        v = 0.0
        while v == 0:
            v = self()
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

    def randn(self, m, n):
        return np.array([[self.normal() for _ in range(n)] for _ in range(m)])


# ---------------------------------------------------------------------------------------------------------------
# gauge elements
# ---------------------------------------------------------------------------------------------------------------
def rotm(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def gauge(th=0.0, t1=0.0, t2=0.0, h=0.0, swap=False, flip=False):
    P = np.eye(2)
    if swap:
        P = np.array([[0.0, 1.0], [1.0, 0.0]]) @ P
    if flip:
        P = P @ np.diag([-1.0, 1.0])
    return rotm(th) @ np.diag([math.exp(t1), math.exp(t2)]) @ np.array([[1.0, h], [0.0, 1.0]]) @ P


GAUGES = [  # key, header, g, membership (O(2)?, monomial?)
    ('rot', 'rotation\n' + r'$R_{30^\circ}$', gauge(th=30)),
    ('diag', 'stretch\n' + r'$\mathrm{diag}(e^{0.5},e^{-0.4})$', gauge(t1=0.5, t2=-0.4)),
    ('sperm', 'signed\npermutation', gauge(swap=True, flip=True)),
    ('gen', 'generic\n' + r'$g\in \mathrm{GL}_2$', gauge(th=25, t1=0.4, t2=-0.3, h=0.35)),
]


def classify(g):
    orth = np.linalg.norm(g.T @ g - np.eye(2)) <= 1e-12
    z = np.abs(g) <= 1e-12 * np.abs(g).max()
    mono = (z[0, 1] and z[1, 0] and not z[0, 0] and not z[1, 1]) or (z[0, 0] and z[1, 1] and not z[0, 1] and not z[1, 0])
    return orth, mono


# ---------------------------------------------------------------------------------------------------------------
# update rules (eta = s = 1; a common sign and scale drop out of relative defects)
# ---------------------------------------------------------------------------------------------------------------
def grads(B, A, G):
    return G @ A.T, B.T @ G


def step_sgd(B, A, G):
    return grads(B, A, G)


def step_adam(B, A, G, rB=None, rA=None):
    """First Adam step from zero state with bias correction at eps = 0: m^ = grad, v^ = grad^2, step sign(grad);
    optional per-rank-channel rates (columns of B, rows of A)."""
    gB, gA = grads(B, A, G)
    dB, dA = np.sign(gB), np.sign(gA)
    if rB is not None:
        dB = dB * np.asarray(rB)[None, :]
        dA = dA * np.asarray(rA)[:, None]
    return dB, dA


def step_scaled(B, A, G, delta):
    gB, gA = grads(B, A, G)
    return gB @ np.linalg.inv(A @ A.T + delta * np.eye(RK)), np.linalg.inv(B.T @ B + delta * np.eye(RK)) @ gA


def sylvester(Mx, Nx, Cx):
    """Solve Mx X + X Nx = Cx by vectorisation (row-major)."""
    r = Mx.shape[0]
    L = np.kron(Mx, np.eye(r)) + np.kron(np.eye(r), Nx.T)
    return np.linalg.solve(L, Cx.reshape(-1)).reshape(r, r)


def step_lorapro(B, A, G):
    gB, gA = grads(B, A, G)
    BtB, AAt = B.T @ B, A @ A.T
    iB, iA = np.linalg.inv(BtB), np.linalg.inv(AAt)
    X = sylvester(BtB, AAt, -(iB @ gA @ A.T))
    PBperp = np.eye(B.shape[0]) - B @ iB @ B.T
    hA = iB @ gA + X @ A
    hB = PBperp @ gB @ iA - B @ X
    return hB, hA


def dW(B, A, d):
    return d[0] @ A + B @ d[1]


RULES = [  # key, label, the gauge group the theory predicts, the result
    ('sgd', 'SGD', r'$\mathrm{O}(2)$', 'III.12(a)'),
    ('adam', r'Adam ($\varepsilon=0$)', r'$B_2$', 'III.12(b)'),
    ('adamEta', 'Adam, per-channel\nrates (joint)', r'$\mathrm{Mon}_2$', 'III.12(b)'),
    ('scaledD', r'scaled GD, $\delta=0.1$', r'$\mathrm{O}(2)$', 'III.12'),
    ('scaled', r'scaled GD, $\delta=0$', r'$\mathrm{GL}_2$', 'III.4(a)'),
    ('lorapro', 'LoRA-Pro', r'$\mathrm{GL}_2$', 'III.4(b)'),
]


def defects(B, A, G, g):
    gi = np.linalg.inv(g)
    B1, A1 = B @ gi, g @ A
    rn = np.hypot(g[:, 0], g[:, 1])                  # row norms of g: |d_j| for a monomial g
    out = {}
    for key, *_ in RULES:
        if key == 'sgd':
            d0, d1 = step_sgd(B, A, G), step_sgd(B1, A1, G)
        elif key == 'adam':
            d0, d1 = step_adam(B, A, G), step_adam(B1, A1, G)
        elif key == 'adamEta':
            d0, d1 = step_adam(B, A, G), step_adam(B1, A1, G, 1 / rn, rn)
        elif key == 'scaledD':
            d0, d1 = step_scaled(B, A, G, DELTA), step_scaled(B1, A1, G, DELTA)
        elif key == 'scaled':
            d0, d1 = step_scaled(B, A, G, 0.0), step_scaled(B1, A1, G, 0.0)
        else:
            d0, d1 = step_lorapro(B, A, G), step_lorapro(B1, A1, G)
        w0, w1 = dW(B, A, d0), dW(B1, A1, d1)
        out[key] = np.linalg.norm(w1 - w0) / np.linalg.norm(w0)
    return out


def predicted(key, orth, mono):
    """Is g in the gauge group of this rule (Thm III.12, Prop III.4)?"""
    return {'sgd': orth, 'scaledD': orth, 'adam': orth and mono, 'adamEta': mono,
            'scaled': True, 'lorapro': True}[key]


# ---------------------------------------------------------------------------------------------------------------
# (a) the optimizer table
# ---------------------------------------------------------------------------------------------------------------
rnd = Rng(SEED_A)
B0, A0, G0 = rnd.randn(MA, RK), rnd.randn(RK, NA), rnd.randn(MA, NA)
TAB = {}
for gk, _, g in GAUGES:
    orth, mono = classify(g)
    dd = defects(B0, A0, G0, g)
    for key, *_ in RULES:
        TAB[key, gk] = dd[key]
        assert (dd[key] <= TOL) == predicted(key, orth, mono), (key, gk, dd[key])

# ---------------------------------------------------------------------------------------------------------------
# (b) merges under a gauge of module 2
# ---------------------------------------------------------------------------------------------------------------
rnd = Rng(SEED_B)
Bs, As = [], []
for _ in range(NMOD):
    Bs.append(rnd.randn(MB, RK))
    As.append(rnd.randn(RK, NB))


def merges(Bs, As):
    TA = sum(w * B @ A for w, B, A in zip(WTS, Bs, As))
    LH = sum(w * B for w, B in zip(WTS, Bs)) @ sum(w * A for w, A in zip(WTS, As))
    return TA, LH


TA0, LH0 = merges(Bs, As)
LC = np.linspace(-3, 3, 241)                         # log2 c
dLH, dTA, resid = [], [], 0.0
for lc in LC:
    c = 2.0 ** lc
    g = np.eye(RK) / c
    gi = np.linalg.inv(g)
    Bs1 = [B @ gi if i == 1 else B for i, B in enumerate(Bs)]
    As1 = [g @ A if i == 1 else A for i, A in enumerate(As)]
    TA1, LH1 = merges(Bs1, As1)
    closed = sum(WTS[l] * WTS[1] * ((c - 1) * Bs[1] @ As[l] + (1 / c - 1) * Bs[l] @ As[1]) for l in range(NMOD) if l != 1)
    resid = max(resid, np.linalg.norm((LH1 - LH0) - closed) / np.linalg.norm(LH0))
    dLH.append(np.linalg.norm(LH1 - LH0) / np.linalg.norm(LH0))
    dTA.append(np.linalg.norm(TA1 - TA0) / np.linalg.norm(TA0))
dLH, dTA = np.array(dLH), np.array(dTA)
assert resid < 1e-13 and dTA.max() < 1e-13

print('(a) relative defects ||dW(g) - dW(I)|| / ||dW(I)||')
print('%-10s' % '' + ''.join('%12s' % gk for gk, *_ in GAUGES))
for key, *_ in RULES:
    print('%-10s' % key + ''.join('%12.2e' % TAB[key, gk] for gk, *_ in GAUGES))
print('(b) LoraHub relative change at c = 1/8, 2, 8: %.3f %.3f %.3f; task arithmetic max %.1e; closed-form residual %.1e'
      % (dLH[0], dLH[np.argmin(abs(LC - 1))], dLH[-1], dTA.max(), resid))


# ---------------------------------------------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------------------------------------------
def val_tex(x):
    """A relative defect as printed in a cell: 0, a power-of-ten form below 1e-3, else two significant digits."""
    if x == 0:
        return '$0$'
    if x < 1e-3:
        m, e = ('%.0e' % x).split('e')
        return r'$%s{\times}10^{%d}$' % (m, int(e))
    return '$%s$' % ('%.2g' % x if x < 0.1 else '%.2f' % x)


def panel_label(ax, s, x, y=1.0, transform=None):
    ax.text(x, y, s, transform=transform or ax.transAxes, fontsize=9, fontweight='medium', color=C['ink'],
            va='bottom', ha='left')


apply_style()
fig = plt.figure(figsize=(W1, 2.35))
gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1], wspace=0.2, left=0.0, right=0.985, bottom=0.17, top=0.9)
axT = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1])

# (a) table of relative defects; moss = equivariant up to round-off (g lies in the rule's gauge group), seal = defect
LW, CW, GW = 1.95, 1.05, 1.2                          # widths: row labels, data cells, group column
NR = len(RULES)
HEAD = 1.05
axT.set_xlim(0, LW + 4 * CW + GW)
axT.set_ylim(NR + HEAD + 0.02, 0)
axT.axis('off')
GROUP_OF_G = [r'$\in \mathrm{O}(2)$', r'$\in \mathrm{Mon}_2$', r'$\in B_2$', r'$\in \mathrm{GL}_2$']
HEADS = ['rotation', 'stretch', 'signed perm.', 'generic']
for j in range(4):
    xc = LW + (j + 0.5) * CW
    axT.text(xc, HEAD - 0.62, HEADS[j], ha='center', va='bottom', fontsize=7.5, color=C['ink'])
    axT.text(xc, HEAD - 0.12, GROUP_OF_G[j], ha='center', va='bottom', fontsize=7.5, color=C['ink2'])
axT.text(LW + 4 * CW + GW / 2, HEAD - 0.62, 'gauge', ha='center', va='bottom', fontsize=7.5, color=C['ink'])
axT.text(LW + 4 * CW + GW / 2, HEAD - 0.12, 'group', ha='center', va='bottom', fontsize=7.5, color=C['ink'])
axT.text(0.06, HEAD - 0.12, 'update rule', ha='left', va='bottom', fontsize=7.5, color=C['ink'])
axT.plot([0, LW + 4 * CW + GW], [HEAD, HEAD], color=C['ink3'], lw=0.6, clip_on=False)
ROW_LABELS = ['SGD', r'Adam, $\varepsilon=0$', r'Adam, per-channel $\eta$', r'scaled GD, $\delta=0.1$',
              r'scaled GD, $\delta=0$', 'LoRA-Pro']
GROUPS = [r'$\mathrm{O}(2)$', r'$B_2$', r'$\mathrm{Mon}_2$ (joint)', r'$\mathrm{O}(2)$', r'$\mathrm{GL}_2$', r'$\mathrm{GL}_2$']
PAD = 0.045
for i, (key, *_r) in enumerate(RULES):
    y0 = HEAD + i
    axT.text(0.06, y0 + 0.5, ROW_LABELS[i], ha='left', va='center', fontsize=7.5, color=C['ink'])
    axT.text(LW + 4 * CW + GW / 2, y0 + 0.5, GROUPS[i], ha='center', va='center', fontsize=7.5, color=C['ink'])
    for j, (gk, *_g) in enumerate(GAUGES):
        v = TAB[key, gk]
        ok = v <= TOL
        col = C['moss'] if ok else C['seal']
        axT.add_patch(Rectangle((LW + j * CW + PAD, y0 + PAD), CW - 2 * PAD, 1 - 2 * PAD, facecolor=col,
                                alpha=0.13 if ok else 0.16, edgecolor='none'))
        axT.text(LW + (j + 0.5) * CW, y0 + 0.52, val_tex(v), ha='center', va='center', fontsize=7.5,
                 color=C['moss'] if ok else C['seal'])
axT.plot([0, LW + 4 * CW + GW], [HEAD + NR, HEAD + NR], color=C['ink3'], lw=0.6, clip_on=False)
panel_label(axT, '(a)', x=0.0, y=-0.12, transform=axT.transData)
axT.texts[-1].set_va('bottom')

# (b) the merge change under a gauge of module 2
axM.plot(LC, dLH, color=C['seal'], lw=1.4, zorder=3)
axM.plot(LC, dTA, color=C['moss'], lw=1.4, zorder=3)
ks = np.arange(-3, 4)
closed_pts = []
for k in ks:
    c = 2.0 ** k
    closed = sum(WTS[l] * WTS[1] * ((c - 1) * Bs[1] @ As[l] + (1 / c - 1) * Bs[l] @ As[1])
                 for l in range(NMOD) if l != 1)
    closed_pts.append(np.linalg.norm(closed) / np.linalg.norm(LH0))
axM.plot(ks, closed_pts, ls='none', marker='o', ms=3.6, mfc='white', mec=C['ink'], mew=0.7, zorder=4)
axM.set_xlim(-3.15, 3.15)
axM.set_xticks(ks)
axM.set_xticklabels(['1/8', '1/4', '1/2', '1', '2', '4', '8'])
axM.set_xlabel(r'$c$ in $(B_2,A_2)\mapsto(cB_2,\,A_2/c)$')
axM.set_ylim(-0.42, 3.6)
axM.set_yticks([0, 1, 2, 3])
axM.set_ylabel(r'$\Vert \Delta\,\mathrm{merge}\Vert_F\,/\,\Vert \mathrm{merge}\Vert_F$')
axM.spines['left'].set_bounds(0, 3.6)
axM.text(-1.0, dLH[np.argmin(abs(LC + 1.0))] + 0.25, 'LoraHub', ha='left', va='bottom', fontsize=7.5, color=C['seal'])
axM.text(-3.0, -0.08, 'task arithmetic', ha='left', va='top', fontsize=7.5, color=C['moss'])
axM.plot([0.55], [3.3], ls='none', marker='o', ms=3.6, mfc='white', mec=C['ink'], mew=0.7)
axM.text(0.8, 3.3, 'closed form', ha='left', va='center', fontsize=7, color=C['ink2'])
panel_label(axM, '(b)', x=-0.27, y=1.0)
bb = fig.get_tightbbox(fig.canvas.get_renderer())
print('tight bbox x: %.3f .. %.3f in (figure 0 .. %.3f)' % (bb.x0, bb.x1, W1))
assert bb.width + 0.04 <= W1, 'figure wider than the text width: %.2f in' % bb.width
save(fig, 'gauge')
print('wrote gauge.pdf')
