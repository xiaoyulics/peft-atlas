"""Figure "apex" (Expose II.A): the apex and the smooth point.

(a), (b) The symmetric 2x2 slice S = [[x+y, z], [z, x-y]] <-> (x, y, z), in which
        ||S||_F^2 = 2 (x^2 + y^2 + z^2),  det S = x^2 - y^2 - z^2,
    so rank <= 1 is the double cone x^2 = y^2 + z^2. The model rho(c, phi) = W_res + c v v^T, v = (cos phi, sin phi),
    plays LoRA (c for B, v for A); d rho = [v v^T, c (v vdot^T + vdot v^T)].
      (a) zero init:  q0 = (0, phi0), W_res = theta0          -> S_1 = R v0 v0^T, one ruling; deficit 1: apex.
      (b) split init: q0 = (c0, phi0), W_res = theta0 - P,  P = c0 v0 v0^T
                                                            -> S_1 = tangent plane of the cone at theta0; deficit 0.
    dim S_1 and d are Jacobian ranks of this model, computed below.
(c) dim S_1 against d(M) at (m, n, r) = (4096, 4096, 16) for zero-init LoRA (stratum T_A), a split init (stratum S)
    and HRA's paired init (Thm II.2(b), Thm II.7(b)); the closed forms are first checked against Jacobian ranks at a
    small size, as in theory/framework_checks.py.
A port of site/js/fig-apex.js. Run from paper/figures/src:  python3 fig_apex.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

from style import apply_style, C, W1, save

DEG = np.pi / 180
RNG = np.random.default_rng(20241)


# ---------------------------------------------------------------------------------------------------------------
# numerical rank and the Jacobian checks
# ---------------------------------------------------------------------------------------------------------------
def nrank(J, tol=1e-9):
    s = np.linalg.svd(J, compute_uv=False)
    return int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0


def lora_jac(B, A):
    """Jacobian of (B, A) -> B A at (B, A), as an (m n) x (r (m + n)) matrix."""
    m, r = B.shape
    n = A.shape[1]
    cols = []
    for i in range(m):
        for j in range(r):
            E = np.zeros((m, r)); E[i, j] = 1
            cols.append((E @ A).ravel())
    for i in range(r):
        for j in range(n):
            E = np.zeros((r, n)); E[i, j] = 1
            cols.append((B @ E).ravel())
    return np.array(cols).T


def householder(u):
    return np.eye(len(u)) - 2 * np.outer(u, u) / (u @ u)


def hra_jac(W0, U):
    """Jacobian of (u_1, ..., u_r) -> W0 H_{u_1} ... H_{u_r} at U (rows u_i)."""
    r, n = U.shape
    Hs = [householder(u) for u in U]
    cols = []
    for i in range(r):
        L = np.eye(n); R = np.eye(n)
        for t in range(i):
            L = L @ Hs[t]
        for t in range(i + 1, r):
            R = R @ Hs[t]
        u = U[i]; nn = u @ u
        for j in range(n):
            ej = np.zeros(n); ej[j] = 1
            D = -2 * (np.outer(ej, u) + np.outer(u, ej)) / nn + 4 * u[j] * np.outer(u, u) / nn ** 2
            cols.append((W0 @ L @ D @ R).ravel())
    return np.array(cols).T


def lora_counts(m, n, r):
    d = r * (m + n - r)
    return {'zero': (m * r, d), 'split': (d, d)}


def hra_counts(m, n, r):
    k = r // 2
    return k * n - k * (k + 1) // 2, r * n - r * (r + 1) // 2


def check_closed_forms(m=11, n=9, r=4):
    """Jacobian ranks at a small size against the closed forms of Thm II.2(b) and Thm II.7(b)."""
    A0 = RNG.standard_normal((r, n)); B0 = RNG.standard_normal((m, r))
    Bg, Ag = RNG.standard_normal((m, r)), RNG.standard_normal((r, n))
    got = {
        'zero': (nrank(lora_jac(np.zeros((m, r)), A0)), nrank(lora_jac(Bg, Ag))),
        'split': (nrank(lora_jac(B0, A0)), nrank(lora_jac(Bg, Ag))),
    }
    W0 = RNG.standard_normal((m, n))                      # injective since m >= n
    k = r // 2
    base = RNG.standard_normal((k, n))
    paired = np.repeat(base, 2, axis=0)                   # u_{2i-1} = u_{2i} = w_i (HF's repeat_interleave)
    got['hra'] = (nrank(hra_jac(W0, paired)), nrank(hra_jac(W0, RNG.standard_normal((r, n)))))
    want = dict(lora_counts(m, n, r)); want['hra'] = hra_counts(m, n, r)
    for key in got:
        print(f'  check ({m},{n},{r}) {key:5s}: Jacobian ranks (dim S1, d) = {got[key]}, closed form = {want[key]}')
        assert got[key] == tuple(want[key]), key


# ---------------------------------------------------------------------------------------------------------------
# the symmetric 2x2 slice model
# ---------------------------------------------------------------------------------------------------------------
W0S = np.array([0.62, 0.36, 0.88])        # theta0 = [[0.62, 0.36], [0.36, 0.88]] stored as [a, b, d]
PHI0, C0 = 45 * DEG, 0.8                   # base direction v0 and the split scale c0
H = 0.62                                   # drawn half-height of the cone (t = c/2 in [-H, H])
AZ, EL = 30 * DEG, 20 * DEG                # camera


def vvT(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([c * c, c * s, s * s])


def sym_outer(v, w):                       # v w^T + w v^T
    return np.array([2 * v[0] * w[0], v[0] * w[1] + v[1] * w[0], 2 * v[1] * w[1]])


def vec4(S):
    return np.array([S[0], S[1], S[1], S[2]])


def xyz(S):
    return np.array([(S[0] + S[2]) / 2, (S[0] - S[2]) / 2, S[1]])


def slice_jac(c, phi):
    v, vd = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
    return np.stack([vec4(vvT(phi)), vec4(c * sym_outer(v, vd))], 1)


def slice_model(mode):
    c = C0 if mode == 'split' else 0.0
    P = C0 * vvT(PHI0)
    vertex = W0S - P if mode == 'split' else W0S.copy()
    dimS1 = nrank(slice_jac(c, PHI0))
    d = max(nrank(slice_jac(cc, pp)) for cc, pp in RNG.uniform([-1, 0], [1, np.pi], (8, 2)))
    det = lambda S: S[0] * S[2] - S[1] ** 2
    # theta0 lies on the image (vertex + rank <= 1): det(theta0 - vertex) = 0
    assert abs(det(W0S - vertex)) < 1e-12
    return dict(start=xyz(W0S - vertex), dimS1=dimS1, d=d, P=P)


def project(p):
    """Orthographic camera of the site: returns screen (u, v) and depth (positive = towards the viewer)."""
    ca, sa, ce, se = np.cos(AZ), np.sin(AZ), np.cos(EL), np.sin(EL)
    p = np.atleast_2d(p)
    u = p[:, 1] * ca + p[:, 2] * sa
    w = -p[:, 1] * sa + p[:, 2] * ca
    return np.stack([u, p[:, 0] * ce - w * se], 1), p[:, 0] * se + w * ce


def hull(pts):
    pts = sorted(map(tuple, pts))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def draw_cone(ax):
    """The double cone x^2 = y^2 + z^2 (vertex at the origin): rulings, latitude circles, light nappe fill."""
    ink = C['ink3']
    segs, cols = [], []
    for k in range(24):
        psi = k * 15 * DEG
        for sg in (1, -1):
            end = np.array([sg * H, sg * H * np.cos(psi), sg * H * np.sin(psi)])
            (o, e), _ = project(np.stack([0 * end, end]))
            _, dm = project(end / 2)
            segs.append([o, e]); cols.append(to_rgba(ink, 0.55 if dm[0] >= 0 else 0.18))
    for j in (1, 2, 3):
        for sg in (1, -1):
            t = sg * H * j / 3
            ps = np.linspace(0, 2 * np.pi, 145)
            P3 = np.stack([np.full_like(ps, t), abs(t) * np.cos(ps), abs(t) * np.sin(ps)], 1)
            P2, dep = project(P3)
            front = (-P3[:, 1] * np.sin(AZ) + P3[:, 2] * np.cos(AZ)) >= 0
            op = 1.0 if j == 3 else 0.7
            for q in range(1, len(ps)):
                f = front[q] and front[q - 1]
                segs.append([P2[q - 1], P2[q]]); cols.append(to_rgba(ink, (0.6 if f else 0.2) * op))
            if j == 3:
                hp = hull(np.vstack([P2, [[0, 0]]]))
                ax.add_patch(Polygon(hp, closed=True, fc=to_rgba(C['ink3'], 0.07), ec='none', zorder=1))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.5, zorder=2))
    a1, a2 = project(np.array([[H * 1.14, 0, 0], [-H * 1.14, 0, 0]]))[0]
    ax.plot(*np.stack([a2, a1]).T, color=C['ink3'], lw=0.5, ls=(0, (1, 2)), zorder=1.5)


def scene(ax, mode, R):
    ax.set_aspect('equal'); ax.axis('off')
    draw_cone(ax)
    halo = [pe.withStroke(linewidth=2.4, foreground='white')]
    start = R['start']
    psi0 = 2 * PHI0
    t1 = np.array([1, np.cos(psi0), np.sin(psi0)]) / np.sqrt(2)      # unit ruling direction through theta0
    t2 = np.array([0, -np.sin(psi0), np.cos(psi0)])                   # unit circumferential direction
    at = lambda a, b: start + a * t1 + b * t2
    (S0,), _ = project(start)
    (O,), _ = project(np.zeros(3))
    if R['dimS1'] == 1:
        L1 = H * np.sqrt(2) * 1.16
        e1, e2 = project(np.stack([at(L1, 0), at(-L1, 0)]))[0]
        ax.plot(*np.stack([e2, e1]).T, color=C['tide'], lw=1.5, solid_capstyle='butt', zorder=5)
        (lp,), _ = project(at(0.30, 0))
        ax.annotate(r'$S_1$', lp, xytext=(5, -4), textcoords='offset points', fontsize=9, color=C['tide'],
                    ha='left', va='center', zorder=9, path_effects=halo)
        ax.annotate('a line (one ruling)', lp, xytext=(5, -13), textcoords='offset points', fontsize=7,
                    color=C['tide'], ha='left', va='center', zorder=9, path_effects=halo)
    else:
        dv = np.linalg.norm(start)
        a0, a1, b = -(dv + 0.08), 0.36, 0.36
        cs = project(np.stack([at(a0, -b), at(a1, -b), at(a1, b), at(a0, b)]))[0]
        ax.add_patch(Polygon(cs, closed=True, fc=to_rgba(C['tide'], 0.16), ec=to_rgba(C['tide'], 0.9), lw=0.8,
                             zorder=4))
        r1, r2 = project(np.stack([at(-dv, 0), at(a1, 0)]))[0]
        ax.plot(*np.stack([r1, r2]).T, color=C['tide'], lw=1.0, zorder=5)
        corner = cs[np.argmin(cs[:, 1] - 0.5 * cs[:, 0])]                 # the plane's lower right corner
        ax.annotate(r'$S_1$', corner, xytext=(5, -4), textcoords='offset points', fontsize=9, color=C['tide'],
                    ha='left', va='center', zorder=9, path_effects=halo)
        ax.annotate('tangent plane', corner, xytext=(5, -13), textcoords='offset points', fontsize=7,
                    color=C['tide'], ha='left', va='center', zorder=9, path_effects=halo)
    # vertex (diamond) and base point theta0 (dot)
    if mode == 'split':
        ax.plot(*O, marker='D', ms=3.6, mfc='white', mec=C['ink'], mew=0.8, zorder=7)
        ax.annotate(r'$\theta_0-P$', O, xytext=(-6, -2), textcoords='offset points', fontsize=8.5,
                    ha='right', va='top', color=C['ink'], zorder=9, path_effects=halo)
        ax.plot(*S0, 'o', ms=4.2, color=C['ink'], mec='white', mew=0.6, zorder=8)
        ax.annotate(r'$\theta_0$', S0, xytext=(-7, 2), textcoords='offset points', fontsize=9.5,
                    ha='right', va='bottom', color=C['ink'], zorder=9, path_effects=halo)
    else:
        ax.plot(*O, 'o', ms=4.2, color=C['ink'], mec='white', mew=0.6, zorder=8)
        ax.annotate(r'$\theta_0$', O, xytext=(-7, 2), textcoords='offset points', fontsize=9.5,
                    ha='right', va='bottom', color=C['ink'], zorder=9, path_effects=halo)



def fit_width(fig, width, iters=4):
    """Rescale the figure so that its tight bounding box (what save() writes) is exactly `width` inches wide:
    text keeps its point size at print size when the PDF is included at \\linewidth."""
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w = bb.width + 2 * plt.rcParams['savefig.pad_inches']
        if abs(w - width) < 1e-3:
            break
        fw, fh = fig.get_size_inches()
        fig.set_size_inches(fw * width / w, fh * width / w)

def main():
    apply_style()
    print('closed forms against Jacobian ranks:')
    check_closed_forms()
    RA, RB = slice_model('zero'), slice_model('split')
    print('slice model: zero init dim S1 =', RA['dimS1'], 'd =', RA['d'], '| split dim S1 =', RB['dimS1'], 'd =', RB['d'])
    assert (RA['dimS1'], RA['d'], RB['dimS1'], RB['d']) == (1, 2, 2, 2)

    m, n, r = 4096, 4096, 16
    lc = lora_counts(m, n, r)
    rows = [('Zero-init LoRA', r'$B_0=0$, stratum $\mathrm{T}_A$', *lc['zero']),
            ('Split init', r'PiSSA-type, stratum $\mathrm{S}$', *lc['split']),
            ('HRA, paired init', r'$u_{2i-1}=u_{2i}$', *hra_counts(m, n, r))]
    for name, _, s1, d in rows:
        print(f'  {name:18s} dim S1 = {s1:7,d}   d = {d:7,d}   deficit = {d - s1:7,d}')

    fig = plt.figure(figsize=(W1, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.06, left=0.0, right=1.0, bottom=0.02, top=0.98)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    tags = {'zero': 'zero init', 'split': 'split init'}
    imlab = {'zero': r'$\mathrm{Im}=\theta_0+\mathcal{M}_{\leq 1}$', 'split': r'$\mathrm{Im}=\theta_0-P+\mathcal{M}_{\leq 1}$'}
    for ax, mode, R in ((axA, 'zero', RA), (axB, 'split', RB)):
        scene(ax, mode, R)
        ax.set_xlim(-0.80, 1.02); ax.set_ylim(-1.22, 0.92)
        ok = R['dimS1'] == R['d']
        ax.text(0.0, 0.085, imlab[mode] + rf', $d={R["d"]}$', transform=ax.transAxes, ha='left', va='bottom',
                fontsize=7.5, color=C['ink2'])
        ax.text(0.0, 0.0, rf'$\dim S_1={R["dimS1"]}$: ' + ('deficit 0, a smooth point' if ok else
                f'deficit {R["d"] - R["dimS1"]}, an apex'), transform=ax.transAxes, ha='left', va='bottom',
                fontsize=7.5, color=C['moss'] if ok else C['seal'])
        ax.text(0.115, 1.0, tags[mode], transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color=C['ink2'])

    # (c) pairs of bars: d(M) (grey) over dim S_1 (teal); the gap between them is the first-order deficit (seal)
    ax = axC
    bh, gap = 0.34, 1.32
    dmax = max(d for *_, d in rows)
    for i, (name, sub, s1, d) in enumerate(rows):
        y0 = -i * gap
        y1 = y0 - bh - 0.05
        ax.barh(y0, d, height=bh, color=to_rgba(C['ink3'], 0.30), lw=0, zorder=2)
        ax.barh(y1, s1, height=bh, color=C['tide'], lw=0, zorder=2)
        # baseline for this pair of bars only, so it does not run through the row's name above it
        ax.plot([0, 0], [y1 - bh / 2, y0 + bh / 2], color=C['ink3'], lw=0.6, zorder=3, solid_capstyle='butt')
        tn = ax.text(0, y0 + bh / 2 + 0.09, name, fontsize=8, fontweight='medium', color=C['ink'], va='bottom')
        ax.annotate(sub, xy=(1, 0), xycoords=tn, xytext=(4, 0), textcoords='offset points', fontsize=7,
                    color=C['ink2'], va='bottom')
        ax.text(0.02 * dmax, y0, f'{d:,}', fontsize=7, color=C['ink'], va='center', zorder=4)
        ax.text(0.02 * dmax, y1, f'{s1:,}', fontsize=7, color='white', va='center', fontweight='medium', zorder=4)
        if d > s1:
            ax.barh(y1, d - s1, left=s1, height=bh, color=to_rgba(C['seal'], 0.10), ec=C['seal'], lw=0.6,
                    ls=(0, (2, 1.4)), zorder=2)
            txt = f'deficit {d - s1:,}'
            if (d - s1) > 0.42 * dmax:
                ax.text(s1 + 0.03 * dmax, y1, txt, fontsize=7, color=C['seal'], va='center', zorder=4)
            else:
                ax.text(d + 0.03 * dmax, y1, txt, fontsize=7, color=C['seal'], va='center', zorder=4)
        else:
            ax.text(d + 0.03 * dmax, y1, 'deficit 0', fontsize=7, color=C['moss'], va='center', zorder=4)
    ax.set_ylim(-(len(rows) - 1) * gap - bh - 0.30, 0.80)
    ax.set_xlim(0, dmax * 1.36)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ('bottom', 'left'):
        ax.spines[sp].set_visible(False)
    # key, in the empty upper-right corner; the matrix size as the panel's tag
    kx, ky = 0.785, 0.80
    for j, (fc, ec, ls, txt, tc) in enumerate([
            (to_rgba(C['ink3'], 0.30), 'none', '-', r'$d(M)$', C['ink2']),
            (C['tide'], 'none', '-', r'$\dim S_1$', C['tide']),
            (to_rgba(C['seal'], 0.10), C['seal'], (0, (2, 1.4)), 'deficit', C['seal'])]):
        yy = ky - j * 0.07
        ax.add_patch(plt.Rectangle((kx, yy - 0.02), 0.05, 0.04, transform=ax.transAxes, fc=fc, ec=ec, lw=0.6,
                                   ls=ls, clip_on=False))
        ax.text(kx + 0.07, yy, txt, transform=ax.transAxes, va='center', fontsize=7, color=tc)
    ax.text(0.095, 1.0, rf'$m=n={m}$, $r={r}$', transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
            color=C['ink2'])

    for ax, lab in ((axA, '(a)'), (axB, '(b)'), (axC, '(c)')):
        ax.text(0.0, 1.0, lab, transform=ax.transAxes, ha='left', va='top', fontsize=8.5, fontweight='semibold',
                color=C['ink'], family='sans-serif')
    fit_width(fig, W1)
    print('wrote', save(fig, 'apex'))


if __name__ == '__main__':
    main()
