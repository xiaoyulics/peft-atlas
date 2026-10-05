"""Figure "orbit" (Expose III): LoRA+ is alpha in disguise (Thm III.10, Cor III.11).

A port of site/js/fig-orbit.js (the hyperparameter orbit explorer) and of check 3 of theory/framework_checks.py.

Toy: W0 in R^{8x6}, X in R^{6x40}, Y = W* X with rank(W* - W0) = 2 (singular values 2.0 and 0.7), loss
L(W) = ||W X - Y||_F^2 / 2N, LoRA rho(B, A) = W0 + s B A with r = 2, B0 = 0, A0 = sigma_A U, T = 300 steps of a
hand-written, bias-corrected Adam (beta = (0.9, 0.999), convention 0/0 = 0) in float64. All randomness comes from the
site's mulberry32 generator with seed 7, so the toy is the site's default one.

Two runs on one orbit of the hyperparameter torus (Adam: c.(s, eta_A, eta_B, sigma_A) = (s/(c_A c_B), c_A eta_A,
c_B eta_B, c_A sigma_A)):
  reference  LoRA+ with lambda = eta_B / eta_A = 16:  s = 1,  eta_A = 4e-3, eta_B = 6.4e-2, sigma_A = 1/4
  current    LoRA with alpha multiplied by lambda:     s = 16, eta_A = eta_B = 4e-3,        sigma_A = 1/4
  control    LoRA with alpha not compensated:          s = 1,  eta_A = eta_B = 4e-3,        sigma_A = 1/4
The torus element c = (1, 1/16) is a power of two, so it rescales IEEE-754 doubles exactly and, at eps = 0 with no
decay and no clipping, the two weight trajectories are bit-identical. The ingredients that break the torus when they are
shared by both parameter groups (Adam eps, PyTorch-style decoupled weight decay 0.01, global-norm clipping at 1, the
HF Trainer default) are switched on one at a time; rescaling them per block as Thm III.10 prescribes (eps_B = lambda eps,
decay_B = lambda wd, a per-block clipping threshold lambda tau on B) restores the exact match.

Run from paper/figures/src:  python3 fig_orbit.py
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import apply_style, C, W1, save

# ---------------------------------------------------------------------------------------------------------------
# fixed setup (site/js/fig-orbit.js)
# ---------------------------------------------------------------------------------------------------------------
M, NI, NS, RK, T = 8, 6, 40, 2, 300
B1, B2 = 0.9, 0.999
WD = 0.01                      # decoupled weight decay, PyTorch AdamW default
CLIP = 1.0                     # global-norm clipping threshold, HF Trainer default max_grad_norm
LR_BASE = 1e-3                 # eta = 1e-3 * 2^k
TARGET_SV = [2.0, 0.7]
SEED = 7
LAM = 16                       # LoRA+ ratio lambda = eta_B / eta_A (a power of two)
REF = dict(ks=0, kA=2, kB=2 + 4, kS=-2)          # LoRA+ lambda = 16: eta_B = 2^4 eta_A
CUR = dict(ks=0 + 4, kA=2, kB=2, kS=-2)          # the torus image under c = (1, 2^-4): LoRA at s * 16
CTL = dict(ks=0, kA=2, kB=2, kS=-2)              # control: the current rates at the reference s
M32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------------------
# the site's seeded generator (site/js/core.js: Atlas.rng, mulberry32 + Box-Muller)
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


def vals(c):
    return dict(s=2.0 ** c['ks'], etaA=LR_BASE * 2.0 ** c['kA'], etaB=LR_BASE * 2.0 ** c['kB'], sigA=2.0 ** c['kS'])


def make_data(seed):
    rnd = Rng(seed)
    W0 = [rnd.normal() / math.sqrt(NI) for _ in range(M * NI)]
    X = [rnd.normal() for _ in range(NI * NS)]
    us, vs = [], []
    for k in range(2):
        u = [rnd.normal() for _ in range(M)]
        v = [rnd.normal() for _ in range(NI)]
        for j in range(k):
            du = sum(u[i] * us[j][i] for i in range(M))
            u = [u[i] - du * us[j][i] for i in range(M)]
            dv = sum(v[i] * vs[j][i] for i in range(NI))
            v = [v[i] - dv * vs[j][i] for i in range(NI)]
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(x * x for x in v))
        us.append([x / nu for x in u])
        vs.append([x / nv for x in v])
    Ws = []
    for i in range(M):
        for j in range(NI):
            w = W0[i * NI + j]
            for k in range(2):
                w += TARGET_SV[k] * us[k][i] * vs[k][j]
            Ws.append(w)
    Y = []
    for i in range(M):
        for q in range(NS):
            acc = 0.0
            for j in range(NI):
                acc += Ws[i * NI + j] * X[j * NS + q]
            Y.append(acc)
    U = [rnd.normal() for _ in range(RK * NI)]
    return dict(W0=W0, X=X, Y=Y, U=U)


# ---------------------------------------------------------------------------------------------------------------
# the optimizer, by hand, in the site's operation order (so that the float64 arithmetic is the site's)
# ---------------------------------------------------------------------------------------------------------------
def adam_step(P, g, Mm, V, eta, c1, c2, eps):
    for q in range(len(P)):
        Mm[q] = B1 * Mm[q] + (1 - B1) * g[q]
        V[q] = B2 * V[q] + (1 - B2) * g[q] * g[q]
        mh = Mm[q] / c1
        den = math.sqrt(V[q] / c2) + eps[q]
        P[q] -= eta * (mh / den if den > 0 else 0.0)        # 0/0 = 0


def simulate(D, c, eps=(0.0, 0.0), wd=(0.0, 0.0), clip=None):
    """Train rho(B, A) = W0 + s B A. eps, wd = (block A, block B). clip = ('global', tau) or ('block', tauA, tauB).
    Returns the weight trajectory (T+1, M*NI), the losses and the number of clipped steps."""
    v = vals(c)
    s, eA, eB = v['s'], v['etaA'], v['etaB']
    Bm = [0.0] * (M * RK)
    Am = [v['sigA'] * u for u in D['U']]
    mB, vB, mA, vA = [0.0] * (M * RK), [0.0] * (M * RK), [0.0] * (RK * NI), [0.0] * (RK * NI)
    epsA, epsB = [eps[0]] * (RK * NI), [eps[1]] * (M * RK)
    W0, X, Y = D['W0'], D['X'], D['Y']
    traj = np.zeros((T + 1, M * NI))
    loss = np.zeros(T + 1)
    clipped = 0
    for t in range(T + 1):
        W = [0.0] * (M * NI)
        for i in range(M):
            for j in range(NI):
                acc = 0.0
                for k in range(RK):
                    acc += Bm[i * RK + k] * Am[k * NI + j]
                W[i * NI + j] = W0[i * NI + j] + s * acc
        Res = [0.0] * (M * NS)
        L = 0.0
        for i in range(M):
            Wi = W[i * NI:(i + 1) * NI]
            for q in range(NS):
                acc = -Y[i * NS + q]
                for j in range(NI):
                    acc += Wi[j] * X[j * NS + q]
                Res[i * NS + q] = acc
                L += acc * acc
        L /= 2 * NS
        traj[t] = W
        loss[t] = L
        if t == T:
            break
        G = [0.0] * (M * NI)
        for i in range(M):
            for j in range(NI):
                acc = 0.0
                for q in range(NS):
                    acc += Res[i * NS + q] * X[j * NS + q]
                G[i * NI + j] = acc / NS                                   # dL/dW
        gB = [0.0] * (M * RK)
        for i in range(M):
            for k in range(RK):
                acc = 0.0
                for j in range(NI):
                    acc += G[i * NI + j] * Am[k * NI + j]
                gB[i * RK + k] = s * acc                                   # dL/dB = s G A^T
        gA = [0.0] * (RK * NI)
        for k in range(RK):
            for j in range(NI):
                acc = 0.0
                for i in range(M):
                    acc += Bm[i * RK + k] * G[i * NI + j]
                gA[k * NI + j] = s * acc                                   # dL/dA = s B^T G
        if clip is not None:
            if clip[0] == 'global':
                nn = math.sqrt(sum(x * x for x in gB) + sum(x * x for x in gA))
                if nn > clip[1]:
                    f = clip[1] / nn
                    gB = [x * f for x in gB]
                    gA = [x * f for x in gA]
                    clipped += 1
            else:
                did = False
                for blk, tau in ((gA, clip[1]), (gB, clip[2])):
                    nn = math.sqrt(sum(x * x for x in blk))
                    if nn > tau:
                        f = tau / nn
                        for q in range(len(blk)):
                            blk[q] *= f
                        did = True
                clipped += did
        if wd[0] or wd[1]:                                                # decoupled decay, PyTorch AdamW order
            fB, fA = 1 - eB * wd[1], 1 - eA * wd[0]
            Bm = [x * fB for x in Bm]
            Am = [x * fA for x in Am]
        c1, c2 = 1 - B1 ** (t + 1), 1 - B2 ** (t + 1)
        adam_step(Bm, gB, mB, vB, eB, c1, c2, epsB)
        adam_step(Am, gA, mA, vA, eA, c1, c2, epsA)
    return dict(W=traj, loss=loss, clipped=clipped)


def gap(a, b):
    """Per-step ||W_t - W'_t||_F and the number of bit-identical entries."""
    d = a['W'] - b['W']
    return np.sqrt((d * d).sum(1)), int((d == 0).sum())


# ---------------------------------------------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------------------------------------------
D = make_data(SEED)
L_ = float(LAM)
# (key, label, shared settings for both runs, per-block settings for the current run that Thm III.10 prescribes)
CASES = [
    ('none', r'none ($\varepsilon=0$)', dict(), None),
    ('eps8', r'$\varepsilon=10^{-8}$', dict(eps=(1e-8, 1e-8)), dict(eps=(1e-8, L_ * 1e-8))),
    ('eps6', r'$\varepsilon=10^{-6}$', dict(eps=(1e-6, 1e-6)), dict(eps=(1e-6, L_ * 1e-6))),
    ('wd', 'decay 0.01', dict(wd=(WD, WD)), dict(wd=(WD, L_ * WD))),
    ('clip', r'clip $\Vert g\Vert\leq 1$', dict(clip=('global', CLIP)), None),
]
runs = {}
for key, _, shared, perblock in CASES:
    ref = simulate(D, REF, **shared)
    cur = simulate(D, CUR, **shared)
    g, eq = gap(ref, cur)
    runs[key] = dict(ref=ref, cur=cur, gap=g, eq=eq)
    if key == 'clip':
        # per-block clipping, thresholds rescaled with the blocks: tau_A = 1 on both runs, tau_B = 1 -> lambda
        refb = simulate(D, REF, clip=('block', CLIP, CLIP))
        curb = simulate(D, CUR, clip=('block', CLIP, L_ * CLIP))
        runs[key]['fix'] = gap(refb, curb)[0]
        runs[key]['fix_clipped'] = (refb['clipped'], curb['clipped'])
    elif perblock is not None:
        refb = ref
        curb = simulate(D, CUR, **perblock)
        runs[key]['fix'] = gap(refb, curb)[0]
ctl = simulate(D, CTL)
gctl, _ = gap(runs['none']['ref'], ctl)

base = runs['none']
assert base['gap'].max() == 0.0 and base['eq'] == (T + 1) * M * NI, 'the orbit pair must be bit-identical'
for key, *_ in CASES[1:]:
    assert runs[key]['gap'].max() > 0, key
    assert runs[key]['fix'].max() == 0.0, key + ': per-block rescaling must restore the exact match'
roundoff = max(np.sqrt((base['ref']['W'] ** 2).sum(1)).max(), 1.0) * 2.0 ** -52

print('final losses: LoRA+ %.3e  LoRA alpha*16 %.3e  control %.3e'
      % (base['ref']['loss'][T], base['cur']['loss'][T], ctl['loss'][T]))
print('bit-identical entries: %d / %d' % (base['eq'], (T + 1) * M * NI))
print('max gap, control: %.3g' % gctl.max())
for key, *_ in CASES:
    r = runs[key]
    print('%-5s shared max gap %.3g   per-block rescaled %s   clipped steps ref/cur %d/%d'
          % (key, r['gap'].max(), r.get('fix', np.zeros(1)).max(), r['ref']['clipped'], r['cur']['clipped']))
print('clip per-block clipped steps', runs['clip']['fix_clipped'], ' round-off scale %.1e' % roundoff)
print('eps gap ratio (1e-6 vs 1e-8), per step: %.6f .. %.6f'
      % tuple(f(runs['eps6']['gap'][1:] / runs['eps8']['gap'][1:]) for f in (np.min, np.max)))


# ---------------------------------------------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------------------------------------------
def sci_tex(x, d=1):
    m, e = ('%.*e' % (d, x)).split('e')
    return r'%s{\times}10^{%d}' % (m, int(e))


def pow10_tex(v, _pos=None):
    e = int(round(np.log10(v)))
    return '$1$' if e == 0 else r'$10^{%d}$' % e


def panel_label(ax, s, x):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight='medium', color=C['ink'], va='bottom', ha='left')


apply_style()
fig = plt.figure(figsize=(W1, 2.45))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.42], height_ratios=[1, 0.27], wspace=0.30, hspace=0.04,
                      left=0.08, right=0.975, bottom=0.155, top=0.935)
axL = fig.add_subplot(gs[:, 0])
axG = fig.add_subplot(gs[0, 1])
axZ = fig.add_subplot(gs[1, 1], sharex=axG)
t = np.arange(T + 1)

# (a) loss curves: the orbit pair coincides, the control does not
axL.plot(t, base['ref']['loss'], color=C['tide'], lw=2.4, solid_capstyle='butt', zorder=3,
         label=r'LoRA+, $\lambda=16$:  $s,\ \eta_B=16\,\eta_A$')
axL.plot(t, base['cur']['loss'], color=C['ochre'], lw=1.0, dashes=(3.2, 2.2), zorder=4,
         label=r'LoRA, $\alpha\times16$:  $16s,\ \eta_B=\eta_A$')
axL.plot(t, ctl['loss'], color=C['ink3'], lw=1.0, dashes=(1, 1.6), zorder=2,
         label=r'control, LoRA:  $s,\ \eta_B=\eta_A$')
axL.set_yscale('log')
axL.set_xlim(0, T)
axL.set_ylim(1e-7, 10)
axL.set_yticks([1e-6, 1e-4, 1e-2, 1])
axL.set_yticks([], minor=True)
axL.yaxis.set_major_formatter(pow10_tex)
axL.set_xticks([0, 100, 200, 300])
axL.set_xlabel('step $t$')
axL.set_ylabel(r'training loss $L(W_t)$')
axL.legend(loc='lower left', handlelength=2.4, borderaxespad=0.1, labelspacing=0.35, fontsize=7)
panel_label(axL, '(a)', x=-0.2)

# (b) per-step weight gap of the orbit pair, log scale over an exact-zero floor; each line is the pair run with one
# ingredient switched on and shared by both parameter groups.
LINES = [  # key, dashes, label, t of the label, side of the line
    ('clip', None, 'global clipping', 258, 'mid'),
    ('wd', (4, 1.6), 'decoupled decay', 150, 'below'),
    ('eps6', (1.2, 1.2), r'$\varepsilon=10^{-6}$', 150, 'below'),
    ('eps8', (1.2, 1.2), r'$\varepsilon=10^{-8}$', 150, 'below'),
]
for key, dash, lab, tl, side in LINES:
    g = runs[key]['gap'].copy()
    g[g == 0] = np.nan                       # zero only at the shared start; drawn on the floor below
    kw = dict(dashes=dash) if dash else {}
    axG.plot(t, g, color=C['seal'], lw=1.0, **kw)
    if side == 'below':
        axG.text(tl, runs[key]['gap'][tl] / 2.6, lab, ha='center', va='top', fontsize=7, color=C['seal'])
    else:                                    # between this line and the control line (log-scale interpolation)
        axG.text(tl, runs[key]['gap'][tl] ** 0.4 * gctl[tl] ** 0.6, lab, ha='center', va='center', fontsize=7,
                 color=C['seal'])
gc = gctl.copy(); gc[gc == 0] = np.nan
axG.plot(t, gc, color=C['ink3'], lw=1.0, dashes=(1, 1.6))
axG.text(150, gctl[150] * 2.6, r'control ($\alpha$ not compensated)', ha='center', va='bottom', fontsize=7,
         color=C['ink2'])
axG.set_yscale('log')
axG.set_ylim(1e-16, 300)
axG.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
axG.set_yticks([], minor=True)
axG.yaxis.set_major_formatter(pow10_tex)
axG.axhline(roundoff, color=C['ink3'], lw=0.5, dashes=(2, 2), zorder=1)
axG.text(T, roundoff * 2.0, r'float64 round-off of $W$', ha='right', va='bottom', fontsize=7, color=C['ink3'])
axG.set_ylabel(r"$\Vert W_t-W_t^{\prime}\Vert_F$")
axG.spines['bottom'].set_visible(False)
axG.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
panel_label(axG, '(b)', x=-0.135)

axZ.plot(t, base['gap'], color=C['moss'], lw=2.0, solid_capstyle='butt')
axZ.text(3, 0.6, r'$\varepsilon=0$: bit-identical at all %d steps' % (T + 1) + '\n'
         r'$\varepsilon$, decay, clipping rescaled per block: bit-identical too', ha='left', va='bottom', fontsize=7,
         color=C['moss'], linespacing=1.25)
axZ.set_ylim(-0.5, 3.6)
axZ.set_yticks([0])
axZ.set_yticklabels([r'$\equiv 0$'])
axZ.spines['left'].set_bounds(-0.5, 3.6)
axZ.set_xlim(0, T)
axZ.set_xticks([0, 100, 200, 300])
axZ.set_xlabel('step $t$')
bb = fig.get_tightbbox(fig.canvas.get_renderer())
print('tight bbox x: %.3f .. %.3f in (figure 0 .. %.3f)' % (bb.x0, bb.x1, W1))
assert bb.width + 0.04 <= W1, 'figure wider than the text width: %.2f in' % bb.width
save(fig, 'orbit')
print('wrote orbit.pdf')
