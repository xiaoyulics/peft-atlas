"""Figure "teaser" (Expose 0): fine-tuning moves a point.

Weight space Theta around the pretrained point theta0, with the image germs of four methods through it and two
dashed non-examples. A port of the still frame of site/js/fig-hero.js (Expose II / VII of theory/framework.md):

  LoRA     theta0 + M_{<=r}: a cone with apex at theta0, drawn as the rank <= 1 double cone x^2 = y^2 + z^2 of the
           symmetric 2x2 slice, in perspective (rulings and rims).
  OFT      theta0 R^T: the orbit of theta0 under rotations. For one weight row it is the circle of radius |theta0|
           about the zero weight 0, so the arc through theta0 is centred at 0.
  (IA)^3   diag(l) theta0: the torus orbit R^x theta0, whose closure is the line through 0 and theta0.
  BitFit   a translation of the bias block: a coordinate line, parallel to the graticule.
  Prefix   (dashed) an extension, unpointed: a curve that never passes through theta0.
  QLoRA    (dashed) LoRA pointed at kappa(theta0), kappa = rounding to the quantiser grid through 0 (the graticule):
           the same cone translated by the defect e = kappa(theta0) - theta0 (Prop V.2).

Every derived point (the grid node kappa(theta0), the defect e, the orbit centre and radius, the gap from the prefix
curve to theta0) is computed below from the chart's design parameters. Method names and modification kinds come from
site/data/atlas-data.js. Run from paper/figures/src:  python3 fig_teaser.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

from style import apply_style, C, KIND, W1, save

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(HERE, '..', '..', '..', 'site', 'data', 'atlas-data.js')
DEG = np.pi / 180


def load_methods():
    txt = open(ATLAS, encoding='utf-8').read()
    data = json.loads(txt[txt.index('{'):txt.rstrip().rstrip(';').rindex('}') + 1])
    return {m['id']: m for m in data['methods']}


# ---------------------------------------------------------------------------------------------------------------
# Chart geometry in units of the composition radius. theta0 sits at the world origin; y points up.
# ---------------------------------------------------------------------------------------------------------------
THETA0 = np.zeros(2)
GAMMA = -42 * DEG                                   # direction from theta0 to the zero weight 0 in Theta
RHO = 1.3                                           # |theta0 - 0|: the OFT orbit radius
ZERO = RHO * np.array([np.cos(GAMMA), np.sin(GAMMA)])
QSTEP = 0.40                                        # quantiser grid spacing: the graticule is 0 + QSTEP Z^2


def kappa(p):
    """Round every coordinate to the grid through 0 (round half up, as the site's Math.round)."""
    return ZERO + QSTEP * np.floor((np.asarray(p) - ZERO) / QSTEP + 0.5)


KT0 = kappa(THETA0)                                 # kappa(theta0)
E = KT0 - THETA0                                    # the pointing defect e = kappa(theta0) - theta0
BETA0 = np.arctan2(THETA0[1] - ZERO[1], THETA0[0] - ZERO[0])   # angle of theta0 seen from 0

# LoRA cone: half-angle 45 deg (|radial| = |axial|, i.e. x^2 = y^2 + z^2, det S = 0 on Sym_2), axis tilted away from
# the viewer and drawn in perspective about its apex, which lies on the chart plane.
CONE = dict(h=0.31, axis=93 * DEG, tilt=21 * DEG, f=5.5, n=18, nq=10)
_a, _p = CONE['axis'], CONE['tilt']
AX = np.array([np.cos(_a) * np.cos(_p), np.sin(_a) * np.cos(_p), -np.sin(_p)])
UX = np.array([AX[1], -AX[0], 0.0]) / np.hypot(AX[0], AX[1])
VX = np.cross(AX, UX)


def cone_world(t, phi, apex):
    """Point at height t in [-1, 1] (nappe sign = sign t) and angle phi on the cone with apex `apex` (chart coords);
    returns (x, y, depth) with depth in [-1, 1] (positive = towards the viewer)."""
    P = CONE['h'] * t * (AX + np.cos(phi) * UX + np.sin(phi) * VX)
    k = CONE['f'] / (CONE['f'] - P[2])
    return np.array([apex[0] + P[0] * k, apex[1] + P[1] * k, P[2] / CONE['h']])


# Prefix tuning: a smooth curve that misses theta0 (Catmull-Rom through these points).
PREFIX_CTRL = np.array([[-0.66, 0.66], [-0.40, 0.31], [-0.26, -0.02], [-0.37, -0.36], [-0.60, -0.66]])


def catmull_rom(P, per=24):
    out = []
    for k in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(0, k - 1)], P[k], P[k + 1], P[min(len(P) - 1, k + 2)]
        for j in range(per):
            t = j / per
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1])
    return np.array(out)


PREFIX = catmull_rom(PREFIX_CTRL)
_gap = np.hypot(PREFIX[:, 0], PREFIX[:, 1])
PREFIX_GAP_PT, PREFIX_GAP = PREFIX[np.argmin(_gap)], _gap.min()   # closest point of the prefix curve to theta0


def oft_pt(beta):
    return ZERO + RHO * np.stack([np.cos(beta), np.sin(beta)], -1)


def ia3_pt(l):
    """0 + l (theta0 - 0): l = 1 is theta0, l = 0 is the zero weight."""
    l = np.asarray(l)[..., None]
    return ZERO + l * (THETA0 - ZERO)


# ---------------------------------------------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------------------------------------------
def germ(ax, pts, color, lw, fade_r, floor=0.0, alpha=1.0, dashes=None, z=3):
    """A polyline whose opacity fades with distance from theta0: it is drawn as a germ at theta0."""
    seg = np.stack([pts[:-1], pts[1:]], 1)
    mid = seg.mean(1)
    d = np.hypot(mid[:, 0], mid[:, 1]) / fade_r
    f = np.clip(np.maximum(floor, 1 - d * d), 0, 1)
    rgba = np.array([to_rgba(color, alpha * fi) for fi in f])
    lc = LineCollection(seg, colors=rgba, linewidths=lw, capstyle='butt', zorder=z)
    if dashes is not None:
        # one continuous dash pattern: draw the dashed path once, with per-segment colours via the collection
        lc.set_linestyle((0, dashes))
    ax.add_collection(lc)
    return lc


def draw_cone(ax, apex, color, alpha, dashed=False, n=18, spin=0.0, z=4):
    segs, cols = [], []
    for sign in (-1, 1):
        for i in range(n):
            phi = 2 * np.pi * i / n + spin
            rim = cone_world(sign, phi, apex)
            front = 0.35 + 0.65 * np.clip((rim[2] + 0.45) / 0.9, 0, 1)
            pts = [cone_world(sign * k / 6, phi, apex) for k in range(7)]
            for k in range(1, 7):
                ramp = min(1.0, (k - 0.5) / 3)
                segs.append([pts[k - 1][:2], pts[k][:2]])
                cols.append(to_rgba(color, alpha * front * (0.18 + 0.82 * ramp)))
        ring = [cone_world(sign, 2 * np.pi * k / 96, apex) for k in range(97)]
        for k in range(1, 97):
            zz = 0.5 * (ring[k][2] + ring[k - 1][2])
            segs.append([ring[k - 1][:2], ring[k][:2]])
            cols.append(to_rgba(color, alpha * (0.45 + 0.55 * np.clip((zz + 0.45) / 0.9, 0, 1))))
    lc = LineCollection(segs, colors=cols, linewidths=0.6 if dashed else 0.65, capstyle='butt', zorder=z)
    if dashed:
        lc.set_linestyle((0, (2.2, 1.8)))
    ax.add_collection(lc)


def label(ax, xy, name, desc, color, ha='left', va='baseline', dx=0, dy=0, desc_below=True):
    """Two-line direct label: method name in its kind colour, descriptor in ink-2 below it."""
    t1 = ax.annotate(name, xy, xytext=(dx, dy), textcoords='offset points', ha=ha, va='baseline',
                     fontsize=8, fontweight='medium', color=color, zorder=10)
    if desc:
        ax.annotate(desc, xy, xytext=(dx, dy - 8.6), textcoords='offset points', ha=ha, va='baseline',
                    fontsize=7, color=C['ink2'], zorder=10)
    return t1



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
    meth = load_methods()
    name = {k: meth[k]['name'] for k in ('lora', 'oft', 'ia3', 'bitfit', 'prefix-tuning', 'qlora')}
    col = {k: KIND[meth[k]['modification_kind']] for k in name}
    name['prefix-tuning'] = 'Prefix tuning'

    # numbers the figure shows, for the record
    print('zero weight 0       =', np.round(ZERO, 4), ' |theta0 - 0| =', RHO)
    print('kappa(theta0)       =', np.round(KT0, 4), ' defect e =', np.round(E, 4), ' |e| =', round(float(np.hypot(*E)), 4))
    print('prefix gap to theta0 =', round(float(PREFIX_GAP), 4), '(> 0: unpointed)')
    assert PREFIX_GAP > 0.15 and np.hypot(*E) > 1e-6

    xlim, ylim = (-1.62, 1.98), (-1.02, 0.80)
    fig = plt.figure(figsize=(W1, W1 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')

    # graticule = the quantiser's grid 0 + QSTEP Z^2, and its nodes (the representable weights)
    gx = ZERO[0] + QSTEP * np.arange(np.ceil((xlim[0] - ZERO[0]) / QSTEP), np.floor((xlim[1] - ZERO[0]) / QSTEP) + 1)
    gy = ZERO[1] + QSTEP * np.arange(np.ceil((ylim[0] - ZERO[1]) / QSTEP), np.floor((ylim[1] - ZERO[1]) / QSTEP) + 1)
    for x in gx:
        ax.plot([x, x], ylim, color=C['rule'], lw=0.35, zorder=0)
    for y in gy:
        ax.plot(xlim, [y, y], color=C['rule'], lw=0.35, zorder=0)
    GX, GY = np.meshgrid(gx, gy)
    ax.scatter(GX.ravel(), GY.ravel(), s=1.6, color=C['ink3'], alpha=0.45, lw=0, zorder=0)

    # 0 in Theta: centre of the OFT orbit and the point the (IA)^3 line runs through
    ax.plot(*ZERO, marker='+', ms=6, mew=0.8, color=C['ink2'], zorder=5)
    # the zero weight is the number 0, set as in the caption (not the dotted monospace zero)
    ax.annotate(r'$0$', ZERO, xytext=(4, -9), textcoords='offset points', fontsize=7.5, color=C['ink2'],
                zorder=10)

    # OFT: the whole orbit (faint) and the germ arc through theta0
    beta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(*oft_pt(beta).T, color=col['oft'], lw=0.6, alpha=0.45, ls=(0, (0.8, 2.2)), zorder=2)
    germ(ax, oft_pt(np.linspace(BETA0 - 1.0, BETA0 + 1.0, 160)), col['oft'], 1.3, fade_r=1.15)

    # (IA)^3: the line through 0 and theta0
    germ(ax, ia3_pt(np.linspace(-0.22, 1.62, 120)), col['ia3'], 1.3, fade_r=1.6, floor=0.18)

    # BitFit: a coordinate line through theta0, parallel to the graticule
    xs = np.linspace(-1.3, 1.3, 120)
    germ(ax, np.stack([xs, 0 * xs], 1), col['bitfit'], 1.3, fade_r=1.35)

    # prefix tuning (dashed): a curve that misses theta0; dotted seal segment = its gap to theta0
    ax.plot(*PREFIX.T, color=col['prefix-tuning'], lw=1.2, ls=(0, (4.0, 2.6)), zorder=3)
    ax.plot([0, PREFIX_GAP_PT[0]], [0, PREFIX_GAP_PT[1]], color=C['seal'], lw=0.8, ls=(0, (1.0, 1.6)), zorder=3)

    # cones: LoRA (apex at theta0) and QLoRA (dashed, apex at kappa(theta0))
    draw_cone(ax, KT0, col['qlora'], 0.62, dashed=True, n=CONE['nq'], spin=np.pi / CONE['nq'], z=3.5)
    draw_cone(ax, THETA0, col['lora'], 0.95, n=CONE['n'], z=4)

    # the defect e = kappa(theta0) - theta0 and the grid node kappa(theta0)
    ax.add_patch(FancyArrowPatch(THETA0, KT0, arrowstyle='-|>,head_length=3.2,head_width=1.6', lw=0.9,
                                 color=C['seal'], shrinkA=2.2, shrinkB=1.6, zorder=6))
    ax.plot(*KT0, marker='o', ms=3.4, mfc='white', mec=col['qlora'], mew=0.8, zorder=7)
    nrm = E / np.hypot(*E)
    halo = [pe.withStroke(linewidth=2.2, foreground='white')]
    ax.annotate(r'$e$', 0.5 * E + 0.065 * np.array([nrm[1], -nrm[0]]), ha='center', va='center',
                fontsize=9, color=C['seal'], zorder=10, path_effects=halo)
    ax.annotate(r'$\kappa\theta_0$', KT0, xytext=(5, -10), textcoords='offset points', fontsize=8.5,
                color=C['ink'], zorder=10, path_effects=halo)

    # theta0
    ax.plot(0, 0, 'o', ms=4.2, color=C['ink'], mec='white', mew=0.6, zorder=8)
    ax.annotate(r'$\theta_0$', (0, 0), xytext=(-11, 3.5), textcoords='offset points', fontsize=10,
                color=C['ink'], ha='center', zorder=10, path_effects=halo)

    # direct labels, anchored on the geometry (as on the site)
    top = max((cone_world(1, 2 * np.pi * k / 96, THETA0) for k in range(96)), key=lambda p: p[1])
    label(ax, top[:2], name['lora'], r'cone, apex at $\theta_0$', col['lora'], ha='center', dy=14)
    otop = ZERO + np.array([0.0, RHO])                                  # top of the orbit circle
    label(ax, otop, name['oft'], r'orbit of $\theta_0$ under rotations', col['oft'], ha='left', dx=-4, dy=14)
    label(ax, ia3_pt(0.42), name['ia3'], r'torus orbit: line through $0$ and $\theta_0$', col['ia3'], dx=15, dy=1)
    label(ax, np.array([1.30, 0.0]), name['bitfit'], 'coordinate line', col['bitfit'], dx=4, dy=6)
    label(ax, PREFIX[0], name['prefix-tuning'], r'unpointed: misses $\theta_0$', col['prefix-tuning'],
          ha='right', dx=-4, dy=2)
    bot = min((cone_world(-1, 2 * np.pi * k / 96, KT0) for k in range(96)), key=lambda p: p[1])
    label(ax, bot[:2], name['qlora'], r'apex moved to $\kappa\theta_0$ by $e=\kappa\theta_0-\theta_0$',
          col['qlora'], ha='center', dx=12, dy=-12)

    # key, in the empty lower-left corner
    kx, ky = xlim[0] + 0.06, ylim[0] + 0.30
    ax.plot([kx, kx + 0.16], [ky, ky], color=C['ink2'], lw=1.2, zorder=9)
    ax.annotate(r'pointed at $\theta_0$', (kx + 0.21, ky), va='center', fontsize=7, color=C['ink2'])
    ax.plot([kx, kx + 0.16], [ky - 0.11, ky - 0.11], color=C['ink2'], lw=1.2, ls=(0, (3.0, 2.0)), zorder=9)
    ax.annotate(r'not pointed at $\theta_0$', (kx + 0.21, ky - 0.11), va='center', fontsize=7, color=C['ink2'])
    ax.plot([kx + 0.08], [ky - 0.22], marker='+', ms=5, mew=0.7, color=C['ink2'])
    ax.annotate(r'zero weight; grid = quantiser $\kappa$', (kx + 0.21, ky - 0.22), va='center', fontsize=7,
                color=C['ink2'])
    ax.annotate(r'$\Theta$', (xlim[1] - 0.04, ylim[1] - 0.04), ha='right', va='top', fontsize=11, color=C['ink2'])

    fit_width(fig, W1)
    print('wrote', save(fig, 'teaser'))


if __name__ == '__main__':
    main()
