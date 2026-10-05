"""Figure "noether" (Expose III): Noether hyperbolas and the crossover (Thm III.5, Cor III.6, Thm III.9).

A port of site/js/fig-noether.js (the Noether-hyperbola and crossover explorer).

(a) Scalar LoRA. w = w0 + s b a with s = 1, loss l = (w - w*)^2 / 2 with w* - w0 = 1, rates eta_A = 1 and
    eta_B = lambda = 2 (time in units of 1/eta_A). The charge Phi = b^2/eta_B - a^2/eta_A is integrated, never assumed:
      gradient flow  RK4 with h = 1e-3 (the site's integrator);
      GD             Euler steps of size eta (learning rates eta*eta_A, eta*eta_B);
      Adam           beta = (0.9, 0.999), eps = 1e-8, bias-corrected, learning rates eta*eta_A, eta*eta_B.
    Six starts ride their own hyperbolas b^2/2 - a^2 = Phi_0 to the fibre {s a b = w* - w0}. From the zero-B start
    (1.2, 0) the panel below shows Phi(t) under the flow, under GD and under Adam at eta = 0.1 and 0.025: a GD step
    changes Phi by O(eta^2) (the exact scalar identity Phi_{k+1} = (1 - lambda eta^2 (s e_k)^2) Phi_k is checked), so the
    drift over a fixed time vanishes like eta, while Adam's drift stays of order one as eta -> 0.

(b) Matrix LoRA, m = n = 6, r = 2, W = W_res + s B A, loss ||W - W*||_F^2 / 2 with a seeded rank-2 target update
    Delta* (singular values 2.0 and 0.3), s = 1, c = 1, eta_A = eta_B = 1. LoRA initialisation B0 = 0,
    A0 A0^T = c I exactly (orthonormal rows times sqrt c), so Phi = -kappa I with kappa = c/eta_A = 1 and
    mu = s sqrt(eta_A eta_B) = 1, crossover sigma* = mu |kappa| / 2 = 0.5. The flow is integrated by the implicit
    midpoint rule (Newton-solved with the analytic Jacobian), which keeps quadratic invariants to round-off. At
    log-spaced times we take the SVD of Z = s B A from the factors, read f_i = u_i^T (B B^T/eta_B) u_i and
    g_i = v_i^T (A^T A/eta_A) v_i off the factors, and compare the closed-form W-velocity of Thm III.9 with
    s (dB A + B dA).

Run from paper/figures/src:  python3 fig_noether.py
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from style import apply_style, C, W1, save

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


# ===============================================================================================================
# (a) scalar LoRA
# ===============================================================================================================
SA, DA = 1.0, 1.0                 # s and delta = w* - w0: the fibre over w* is {s a b = delta}
LAM = 2.0                         # eta_B / eta_A, with eta_A = 1
LIM = 2.4                         # half-width of the (a, b) window
H_FLOW = 1e-3                     # RK4 step
T_A = 10.0                        # horizon
AB1, AB2, AEPS = 0.9, 0.999, 1e-8
STARTS = [(1.2, 0.0), (0.0, 1.2), (0.25, 0.25 * math.sqrt(LAM)), (2.0, -0.6), (-1.5, 0.5), (1.0, -2.0)]
ETAS = [0.1, 0.025]
ETA_SWEEP = np.geomspace(0.0025, 0.2, 9)        # step sizes for the drift-versus-eta panel


def rhs_a(a, b, lam):
    e = SA * a * b - DA
    return -SA * e * b, -lam * SA * e * a


def sim_scalar(a0, b0, lam, opt, eta=None, T=T_A):
    """Port of simScalar (no weight decay). Returns t, a, b, Phi at every step, and the GD identity residual."""
    h = H_FLOW if opt == 'flow' else eta
    N = int(round(T / h))
    a, b = a0, b0
    ts, As, Bs = [0.0], [a], [b]
    ma = mb = va = vb = 0.0
    id_res = 0.0
    for k in range(1, N + 1):
        Pk, ek, scale = b * b / lam - a * a, SA * a * b - DA, a * a + b * b / lam
        if opt == 'flow':
            k1 = rhs_a(a, b, lam)
            k2 = rhs_a(a + 0.5 * h * k1[0], b + 0.5 * h * k1[1], lam)
            k3 = rhs_a(a + 0.5 * h * k2[0], b + 0.5 * h * k2[1], lam)
            k4 = rhs_a(a + h * k3[0], b + h * k3[1], lam)
            a += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            b += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        elif opt == 'gd':
            k1 = rhs_a(a, b, lam)
            a += h * k1[0]
            b += h * k1[1]
            # scalar identity (expand the step): Phi_{k+1} = (1 - lambda eta^2 (s e_k)^2) Phi_k
            pred = Pk * (1 - lam * h * h * SA * SA * ek * ek)
            id_res = max(id_res, abs(b * b / lam - a * a - pred) / scale)
        else:
            ga, gb = SA * ek * b, SA * ek * a
            ma = AB1 * ma + (1 - AB1) * ga
            mb = AB1 * mb + (1 - AB1) * gb
            va = AB2 * va + (1 - AB2) * ga * ga
            vb = AB2 * vb + (1 - AB2) * gb * gb
            c1, c2 = 1 - AB1 ** k, 1 - AB2 ** k
            a -= h * (ma / c1) / (math.sqrt(va / c2) + AEPS)
            b -= lam * h * (mb / c1) / (math.sqrt(vb / c2) + AEPS)
        if not (abs(a) < 1e6 and abs(b) < 1e6):
            raise RuntimeError('diverged')
        ts.append(k * h)
        As.append(a)
        Bs.append(b)
    t, A, B = np.array(ts), np.array(As), np.array(Bs)
    return dict(t=t, a=A, b=B, phi=B * B / lam - A * A, id_res=id_res, loss=0.5 * (SA * A[-1] * B[-1] - DA) ** 2)


def hyperbola(Phi, lam, n=400):
    """Branches of b^2/lam - a^2 = Phi inside the window (lists of (a, b) arrays)."""
    out = []
    if Phi < 0:
        bb = np.linspace(-LIM * 1.2, LIM * 1.2, n)
        for sg in (1, -1):
            out.append((sg * np.sqrt(bb * bb / lam - Phi), bb))
    elif Phi > 0:
        aa = np.linspace(-LIM * 1.2, LIM * 1.2, n)
        for sg in (1, -1):
            out.append((aa, sg * np.sqrt(lam * (Phi + aa * aa))))
    else:
        aa = np.array([-LIM * 1.2, LIM * 1.2])
        out += [(aa, math.sqrt(lam) * aa), (aa, -math.sqrt(lam) * aa)]
    return out


# ===============================================================================================================
# (b) matrix LoRA, m = n = 6, r = 2
# ===============================================================================================================
SEED_B = 20210718
ST = [2.0, 0.3]                   # singular values of the target update Delta*
SR = [1.2, 0.6]                   # (drawn for the site's PiSSA option; kept so that the seeded draws match)
S_B, C_B, ETA_A, ETA_B = 1.0, 1.0, 1.0, 1.0
XLO, XHI = 0.003, 30.0
MID_REL, MID_STIFF = 0.01, 4.0
STEP_CAP, T_CAP, REST_TOL, FRAME_GROW = 60000, 4000.0, 1e-8, 1.025
RES_FLOOR = 1e-6


def orth62(rnd):
    """Two orthonormal columns in R^6 (6x2), Gram-Schmidt applied twice, in the site's draw order (row-major)."""
    Q = np.array([rnd.normal() for _ in range(12)]).reshape(6, 2)
    for k in range(2):
        for _ in range(2):
            for p in range(k):
                Q[:, k] -= (Q[:, k] @ Q[:, p]) * Q[:, p]
        Q[:, k] /= np.linalg.norm(Q[:, k])
    return Q


def setup_b():
    rnd = Rng(SEED_B)
    D = dict(Ut=orth62(rnd), Vt=orth62(rnd), Q=orth62(rnd), QB=orth62(rnd), Ur=orth62(rnd), Vr=orth62(rnd))
    D['Zt'] = D['Ut'] @ np.diag(ST) @ D['Vt'].T
    return D


def fgk(x, kap):
    """f_kappa(x) = sqrt(x + kappa^2/4) - kappa/2 and g_kappa = f_kappa + kappa, without cancellation."""
    root = np.sqrt(x + kap * kap / 4)
    ak = abs(kap) / 2
    small, large = x / (root + ak), root + ak
    if kap == 0:
        return root, root
    return (small, large) if kap > 0 else (large, small)


def svd_factors(B, A, s):
    """Compact SVD of Z = s B A (rank 2) from the factors: Z = s QB (RB RA^T) QA^T, then a 2x2 SVD."""
    QB, RB = np.linalg.qr(B)
    QA, RA = np.linalg.qr(A.T)
    u, sv, vt = np.linalg.svd(s * RB @ RA.T)
    return sv, QB @ u, QA @ vt.T


def sim_matrix(D):
    s, c, eA, eB = S_B, C_B, ETA_A, ETA_B
    Zs = D['Zt']
    B = np.zeros((6, 2))
    A = math.sqrt(c) * D['Q'].T                     # A0 A0^T = c I exactly
    kappa = c / eA
    mu = s * math.sqrt(eA * eB)
    sstar = mu * abs(kappa) / 2

    def unpack(y):
        return y[:12].reshape(6, 2), y[12:].reshape(2, 6)

    def rhs(y):
        Bm, Am = unpack(y)
        G = s * Bm @ Am - Zs
        return np.concatenate([(-eB * s * G @ Am.T).ravel(), (-eA * s * Bm.T @ G).ravel()]), G

    I6, I2 = np.eye(6), np.eye(2)

    def jac(y, G):
        """Analytic Jacobian of the flow (row-major vec: vec(P X Q) = kron(P, Q^T) vec(X))."""
        Bm, Am = unpack(y)
        J = np.zeros((24, 24))
        J[:12, :12] = -eB * s * s * np.kron(I6, Am @ Am.T)
        T2 = np.zeros((6, 2, 2, 6))
        for k in range(2):
            T2[:, k, k, :] = G
        J[:12, 12:] = -eB * s * (s * np.kron(Bm, Am) + T2.reshape(12, 12))
        T3 = np.zeros((2, 6, 6, 2))
        for k in range(2):
            T3[k, :, :, k] = G.T
        J[12:, :12] = -eA * s * (T3.reshape(12, 12) + s * np.kron(Bm.T, Am.T))
        J[12:, 12:] = -eA * s * s * np.kron(Bm.T @ Bm, I6)
        return J

    y = np.concatenate([B.ravel(), A.ravel()])
    F = dict(t=[], s1=[], s2=[], f1=[], f2=[], g1=[], g2=[], res=[], drift=[], wn=[])
    st = dict(maxRes=0.0, maxDrift=0.0, nRes=0, wmax=0.0)

    def measure(t, y):
        f, G = rhs(y)
        Bm, Am = unpack(y)
        dB, dA = unpack(f)
        Wf = s * (dB @ Am + Bm @ dA)
        wn = np.linalg.norm(Wf)
        st['wmax'] = max(st['wmax'], wn)
        Phi = Bm.T @ Bm / eB - Am @ Am.T / eA
        drift = np.linalg.norm(Phi + kappa * I2)
        st['maxDrift'] = max(st['maxDrift'], drift)
        sv = fv = gv = (math.nan, math.nan)
        res = math.nan
        if np.linalg.norm(Bm) > 0:
            svals, U, V = svd_factors(Bm, Am, s)
            sv = tuple(svals)
            fv = tuple(np.sum((Bm.T @ U) ** 2, axis=0) / eB)       # u_k^T (B B^T / eta_B) u_k
            gv = tuple(np.sum((Am @ V) ** 2, axis=0) / eA)         # v_k^T (A^T A / eta_A) v_k
            if svals[1] > 1e-9 and svals[1] > 1e-7 * svals[0] and wn > 0:
                fc, gc = fgk(svals ** 2 / mu ** 2, kappa)
                Wc = -mu * mu * (G @ V @ np.diag(gc) @ V.T + U @ np.diag(fc) @ U.T @ G)
                res = np.linalg.norm(Wc - Wf) / wn
                if wn >= RES_FLOOR * st['wmax']:
                    st['maxRes'] = max(st['maxRes'], res)
                    st['nRes'] += 1
        for key, v in (('t', t), ('s1', sv[0]), ('s2', sv[1]), ('f1', fv[0]), ('f2', fv[1]), ('g1', gv[0]),
                       ('g2', gv[1]), ('res', res), ('drift', drift), ('wn', wn)):
            F[key].append(v)
        return np.linalg.norm(f) / max(np.linalg.norm(y), 1e-300)

    def mid_step(y0, f0, hh):
        """One implicit-midpoint step y1 = y0 + h f((y0 + y1)/2), Newton-solved; None if Newton fails."""
        h2 = hh / 2
        yb = y0 + h2 * f0
        prev = math.inf
        for it in range(12):
            fb, G = rhs(yb)
            R = y0 + h2 * fb - yb
            sc = max(np.abs(yb).max(), h2 * np.abs(fb).max())
            Mm = np.eye(24) - h2 * jac(yb, G)
            try:
                dy = np.linalg.solve(Mm, R)
            except np.linalg.LinAlgError:
                return None
            yb = yb + dy
            dn = np.abs(dy).max()
            if dn <= 4e-16 * sc:
                break
            if it >= 2 and dn > 0.5 * prev and dn <= 1e-13 * sc:
                break
            prev = dn
            if it == 11:
                return None
        return 2 * yb - y0

    t, steps, stop = 0.0, 0, 'cap'
    measure(0.0, y)
    f0, G = rhs(y)
    t_next = MID_REL * math.sqrt((y @ y) / max(f0 @ f0, 1e-300))
    while steps < STEP_CAP:
        Bm, Am = unpack(y)
        Lst = s * s * (eB * np.sum(Am ** 2) + eA * np.sum(Bm ** 2)) + s * math.sqrt(eA * eB) * np.linalg.norm(G)
        hh = min(MID_STIFF / Lst, MID_REL * math.sqrt((y @ y) / max(f0 @ f0, 1e-300)), t_next - t)
        if not hh > 0:
            hh = t_next - t
        y1 = mid_step(y, f0, hh)
        tries = 0
        while y1 is None:
            hh *= 0.5
            tries += 1
            if tries > 30:
                raise RuntimeError('Newton failed')
            y1 = mid_step(y, f0, hh)
        y = y1
        t += hh
        steps += 1
        f0, G = rhs(y)
        if t >= t_next * (1 - 1e-12):
            rel = measure(t, y)
            t_next = t * FRAME_GROW
            if rel * t <= REST_TOL:
                stop = 'rest'
                break
            if t >= T_CAP:
                stop = 'tcap'
                break
    if F['t'][-1] != t:
        measure(t, y)
    out = {k: np.array(v, dtype=float) for k, v in F.items()}
    out.update(st, T=t, steps=steps, stop=stop, kappa=kappa, mu=mu, sstar=sstar, loss=0.5 * np.sum(G ** 2))
    return out


# regimes of Thm III.9 on the two gains of one direction: one-sided when one gain is over ten times the other,
# balanced when they are within a factor 2 (as on the site)
Q_ONE, Q_BAL = 0.1, 0.5


def sig_of_ratio(q):
    """sigma / sigma* at which min(f, g) / max(f, g) = q."""
    y = q / (1 - q)
    return 2 * math.sqrt(y * y + y)


# ===============================================================================================================
# compute and report
# ===============================================================================================================
def compute():
    RA = dict(flow=[sim_scalar(a0, b0, LAM, 'flow') for a0, b0 in STARTS])
    a0, b0 = STARTS[0]
    for opt in ('gd', 'adam'):
        for eta in ETAS:
            RA[opt, eta] = sim_scalar(a0, b0, LAM, opt, eta)
    print('(a) scalar LoRA, lambda = eta_B/eta_A = %g, T = %g' % (LAM, T_A))
    for (a0, b0), r in zip(STARTS, RA['flow']):
        print('   flow from (%5.2f, %5.2f): Phi0 = %+.4f  max|Phi - Phi0| = %.1e  end (%.3f, %.3f)  loss %.1e'
              % (a0, b0, r['phi'][0], np.abs(r['phi'] - r['phi'][0]).max(), r['a'][-1], r['b'][-1], r['loss']))
        assert np.abs(r['phi'] - r['phi'][0]).max() < 1e-9 and r['loss'] < 1e-20
    for opt in ('gd', 'adam'):
        for eta in ETAS:
            r = RA[opt, eta]
            print('   %-4s eta = %-6g from (1.2, 0): Phi(T) = %+.4f  max|Phi - Phi0| = %.3f  end (%.3f, %.3f)%s'
                  % (opt, eta, r['phi'][-1], np.abs(r['phi'] - r['phi'][0]).max(), r['a'][-1], r['b'][-1],
                     '  GD identity residual %.1e' % r['id_res'] if opt == 'gd' else ''))
    assert max(RA['gd', e]['id_res'] for e in ETAS) < 1e-14
    # drift over the fixed horizon against the step size, from the zero-B start
    RA['sweep'] = ETA_SWEEP
    for opt in ('gd', 'adam'):
        RA['drift', opt] = []
        RA['step', opt] = []
        for eta in ETA_SWEEP:
            r = sim_scalar(*STARTS[0], LAM, opt, eta)
            RA['drift', opt].append(np.abs(r['phi'] - r['phi'][0]).max())
            RA['step', opt].append(np.abs(np.diff(r['phi'])).max())
            if opt == 'gd':
                assert r['id_res'] < 1e-14
        RA['drift', opt] = np.array(RA['drift', opt])
        RA['step', opt] = np.array(RA['step', opt])
    RA['flow_drift'] = np.abs(RA['flow'][0]['phi'] - RA['flow'][0]['phi'][0]).max()
    print('   step-size sweep (zero-B start, T = %g): eta, max_t |Phi - Phi0| for GD and Adam, max per-step change' % T_A)
    for i, eta in enumerate(ETA_SWEEP):
        print('     eta = %.5f   GD %.3e  Adam %.3e   per step: GD %.3e (= lambda eta^2 e0^2 |Phi0| = %.3e)  Adam %.3e'
              % (eta, RA['drift', 'gd'][i], RA['drift', 'adam'][i], RA['step', 'gd'][i],
                 LAM * eta ** 2 * DA ** 2 * abs(RA['flow'][0]['phi'][0]), RA['step', 'adam'][i]))
    sl = np.polyfit(np.log(ETA_SWEEP), np.log(RA['drift', 'gd']), 1)[0]
    print('     GD drift slope in log-log: %.3f;  flow (RK4) drift %.1e' % (sl, RA['flow_drift']))

    D = setup_b()
    RB = sim_matrix(D)
    print('(b) matrix LoRA m = n = 6, r = 2: kappa = %g, mu = %g, sigma* = %g' % (RB['kappa'], RB['mu'], RB['sstar']))
    print('   %d implicit-midpoint steps, stop = %s at t = %.4g, %d checkpoints, final loss %.2e'
          % (RB['steps'], RB['stop'], RB['T'], len(RB['t']), RB['loss']))
    print('   final singular values %.6f, %.6f (target %g, %g)' % (RB['s1'][-1], RB['s2'][-1], *ST))
    print('   max ||Phi + kappa I|| = %.1e;  max relative residual of the closed form = %.1e over %d checkpoints'
          % (RB['maxDrift'], RB['maxRes'], RB['nRes']))
    print('   regimes: one-sided below %.3f sigma* = %.3f, balanced above %.3f sigma* = %.3f'
          % (sig_of_ratio(Q_ONE), sig_of_ratio(Q_ONE) * RB['sstar'], sig_of_ratio(Q_BAL), sig_of_ratio(Q_BAL) * RB['sstar']))
    assert RB['maxDrift'] < 1e-12 and RB['maxRes'] < 1e-9
    return RA, RB


# ===============================================================================================================
# drawing
# ===============================================================================================================
def fit_width(fig, width, iters=5):
    """Rescale the figure so that its tight bounding box (what save() writes) is exactly `width` inches wide."""
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w = bb.width + 2 * plt.rcParams['savefig.pad_inches']
        if abs(w - width) < 1e-3:
            break
        fw, fh = fig.get_size_inches()
        fig.set_size_inches(fw * width / w, fh * width / w)


def panel_label(ax, s, x, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='medium', color=C['ink'], va='bottom', ha='left')


def log_ticks(ax, axis, lo, hi):
    from matplotlib.ticker import FixedLocator, NullLocator
    dec = [10.0 ** e for e in range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)
           if lo <= 10.0 ** e <= hi]

    def lab(v):
        e = int(round(math.log10(v)))
        return {0: '1', 1: '10', -1: '0.1', -2: '0.01'}.get(e, r'$10^{%d}$' % e)
    getattr(ax, 'set_%sticks' % axis)(dec)
    getattr(ax, 'set_%sticklabels' % axis)([lab(v) for v in dec])
    getattr(ax, '%saxis' % axis).set_minor_locator(NullLocator())


def draw(RA, RB):
    apply_style()
    fig = plt.figure(figsize=(W1, 3.45))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.18], height_ratios=[1.62, 1], wspace=0.26, hspace=0.34,
                          left=0.07, right=0.985, bottom=0.11, top=0.95)
    axP = fig.add_subplot(gs[0, 0])
    axM = fig.add_subplot(gs[1, 0])
    axG = fig.add_subplot(gs[0, 1])
    axS = fig.add_subplot(gs[1, 1], sharex=axG)

    # ---------------- (a) the (a, b) plane ----------------
    ax = axP
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect('equal')
    for v in (0,):
        ax.axhline(v, color=C['rule'], lw=0.5, zorder=0)
        ax.axvline(v, color=C['rule'], lw=0.5, zorder=0)
    # the slice Phi = 0 (asymptotes b = +- sqrt(lambda) a)
    for aa, bb in hyperbola(0.0, LAM):
        ax.plot(aa, bb, color=C['moss'], lw=0.8, ls=(0, (4, 3)), zorder=1)
    # the fibre over w*
    for sg in (1, -1):
        aa = np.geomspace(DA / (SA * LIM * 1.2), LIM * 1.2, 300)
        ax.plot(sg * aa, sg * DA / (SA * aa), color=C['ochre'], lw=2.4, solid_capstyle='round', zorder=2)
    # each start's own charge hyperbola (the branch it lies on) and the flow trajectory along it
    for (a0, b0), r in zip(STARTS, RA['flow']):
        Phi0 = r['phi'][0]
        if abs(Phi0) < 1e-12:
            continue
        for aa, bb in hyperbola(Phi0, LAM):
            if (Phi0 < 0 and np.sign(aa[0]) == np.sign(a0)) or (Phi0 > 0 and np.sign(bb[0]) == np.sign(b0)):
                ax.plot(aa, bb, color=C['ink3'], lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    ad = RA['adam', ETAS[0]]
    ax.plot(ad['a'], ad['b'], color=C['seal'], lw=1.0, zorder=3)
    for (a0, b0), r in zip(STARTS, RA['flow']):
        ax.plot(r['a'], r['b'], color=C['tide'], lw=1.7, solid_capstyle='round', zorder=4)
        ax.plot([a0], [b0], 'o', ms=3.6, mfc=C['tide'], mec='white', mew=0.6, zorder=5)
        ax.plot([r['a'][-1]], [r['b'][-1]], 'o', ms=3.4, mfc='white', mec=C['tide'], mew=0.9, zorder=5)
    ax.plot([ad['a'][-1]], [ad['b'][-1]], 'o', ms=3.4, mfc='white', mec=C['seal'], mew=0.9, zorder=5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(['\u22122', '\u22121', '0', '1', '2'])
    ax.set_yticklabels(['\u22122', '\u22121', '0', '1', '2'])
    ax.set_xlabel('$a$', labelpad=1)
    ax.set_ylabel('$b$', labelpad=1, rotation=0)
    halo = [pe.withStroke(linewidth=2.2, foreground='white')]
    # to the right of the upper branch of the fibre, clear of the thick curve itself
    ax.text(0.68, 2.2, r'$s\,ab=w^\star-w_0$', color=C['ochre'], fontsize=7.5, ha='left', va='center', path_effects=halo,
            zorder=6)
    ax.text(1.1, -0.1, r'$b_0=0$', color=C['ink2'], fontsize=7, ha='right', va='top', path_effects=halo, zorder=6)
    ax.text(-0.08, 1.2, r'$a_0=0$', color=C['ink2'], fontsize=7, ha='right', va='center', path_effects=halo, zorder=6)
    ax.text(-2.3, -1.72, r'$\Phi=0$', color=C['moss'], fontsize=7, ha='left', va='center', path_effects=halo, zorder=6)
    ax.text(1.6, 1.24, 'Adam', color=C['seal'], fontsize=7, ha='left', va='center', path_effects=halo, zorder=6)
    panel_label(ax, '(a)', -0.2, 1.0)

    # ---------------- (a) the charge drift over the run against the step size ----------------
    ax = axM
    ex = RA['sweep']
    ax.plot(ex, RA['drift', 'gd'], '-o', color=C['ochre'], lw=1.1, ms=3.2, mfc=C['ochre'], mec='white', mew=0.5)
    ax.plot(ex, RA['drift', 'adam'], '-o', color=C['seal'], lw=1.1, ms=3.2, mfc=C['seal'], mec='white', mew=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(ex[0] / 1.35, ex[-1] * 1.35)
    ax.set_ylim(1.5e-5, 6)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels([r'$10^{-4}$', r'$10^{-3}$', '0.01', '0.1', '1'])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks([0.003, 0.01, 0.03, 0.1])
    ax.set_xticklabels(['0.003', '0.01', '0.03', '0.1'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r'step size $\eta$', labelpad=1)
    ax.set_ylabel(r'drift of $\Phi$', labelpad=2)
    ax.text(ex[0], RA['drift', 'adam'][0] * 1.5, 'Adam', color=C['seal'], fontsize=7, ha='left', va='bottom')
    ax.text(ex[0], RA['drift', 'gd'][0] / 1.7, r'GD ($\propto\eta$)', color=C['ochre'], fontsize=7, ha='left',
            va='top')
    m, e = ('%.1e' % RA['flow_drift']).split('e')
    ax.text(ex[-1] * 1.3, 2.4e-5, r'gradient flow: $%s\times10^{%d}$' % (m, int(e)), color=C['tide'], fontsize=7,
            ha='right', va='bottom')

    # ---------------- (b) gains f, g against sigma ----------------
    kap, mu, ss = RB['kappa'], RB['mu'], RB['sstar']
    xone, xbal = ss * sig_of_ratio(Q_ONE), ss * sig_of_ratio(Q_BAL)
    ax = axG
    ghi = fgk(XHI ** 2 / mu ** 2, kap)
    yhi = max(ghi) * 1.6
    flo = fgk(XLO ** 2 / mu ** 2, kap)
    ylo = max(min(flo), yhi * 1e-5)
    for a_ in (axG, axS):
        a_.axvspan(XLO, xone, color=C['tide'], alpha=0.10, lw=0, zorder=0)
        a_.axvspan(xone, xbal, color=C['tide'], alpha=0.04, lw=0, zorder=0)
        a_.axvspan(xbal, XHI, color=C['paper2'], alpha=0.9, lw=0, zorder=0)
        a_.axvline(ss, color=C['ochre'], lw=1.1, ls=(0, (4, 2.5)), zorder=2)
    sg = np.geomspace(XLO, XHI, 400)
    fv, gv = fgk(sg ** 2 / mu ** 2, kap)
    ax.plot(sg, np.full_like(sg, abs(kap)), color=C['ink3'], lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.plot(sg, sg / mu, color=C['ink3'], lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.plot(sg, gv, color=C['tide'], lw=1.7, zorder=3)
    ax.plot(sg, fv, color=C['purple'], lw=1.7, zorder=3)
    every = max(1, int(round(len(RB['t']) / 45)))
    for i in range(1, len(RB['t']), every):
        for sk, fk, col in (('s1', 'f1', C['purple']), ('s2', 'f2', C['purple']), ('s1', 'g1', C['tide']), ('s2', 'g2', C['tide'])):
            sv, gvv = RB[sk][i], RB[fk][i]
            if sv > 0 and gvv > 0:
                ax.plot([sv], [gvv], 'o', ms=2.6, mfc='white', mec=col, mew=0.6, zorder=4, alpha=0.9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(ylo, yhi)
    log_ticks(ax, 'y', ylo, yhi)
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r'gain $f_\kappa,\ g_\kappa$', labelpad=2)
    ax.text(ss * 1.08, yhi * 0.75, r'$\sigma^*=\mu|\kappa|/2$', color=C['ochre'], fontsize=7.5, ha='left', va='top')
    ax.text(XLO * 1.25, abs(kap) * 1.35, r'$g_\kappa\to|\kappa|$ ($B$-update)', color=C['tide'], fontsize=7.5,
            ha='left', va='bottom')
    xl = 0.012
    ax.text(0.11, 2.2e-3, r'$f_\kappa$ ($A$-update)', color=C['purple'], fontsize=7.5, ha='left', va='center')
    ax.text(XLO * 1.25, yhi * 0.75, 'one-sided', color=C['tide'], fontsize=7, ha='left', va='top')
    ax.text(XHI * 0.85, ylo * 1.6, 'balanced', color=C['ink2'], fontsize=7, ha='right', va='bottom')
    ax.text(XHI * 0.85, 9.0, r'$\sigma/\mu$', color=C['ink2'], fontsize=7, ha='right', va='top')
    panel_label(ax, '(b)', -0.16, 1.0)

    # ---------------- (b) singular values along the flow, time running down ----------------
    ax = axS
    tt = RB['t'][1:]
    ax.plot(np.maximum(RB['s1'][1:], XLO * 0.5), tt, color=C['ink'], lw=1.3, zorder=3)
    ax.plot(np.maximum(RB['s2'][1:], XLO * 0.5), tt, color=C['ink2'], lw=1.3, ls=(0, (4, 2)), zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(RB['T'], tt[0])
    log_ticks(ax, 'y', tt[0], RB['T'])
    log_ticks(ax, 'x', XLO, XHI)
    ax.set_xlabel(r'singular value $\sigma$ of $\Delta W$', labelpad=1)
    ax.set_ylabel(r'time $t$', labelpad=2)
    for key, lab in (('s1', r'$\sigma_1$'), ('s2', r'$\sigma_2$')):
        ax.text(RB[key][-1] * 1.1, RB['T'] * 0.5, lab, color=C['ink'], fontsize=7.5, ha='left', va='bottom')
    fit_width(fig, W1)
    return fig


if __name__ == '__main__':
    RA, RB = compute()
    fig = draw(RA, RB)
    print('wrote', save(fig, 'noether'))
