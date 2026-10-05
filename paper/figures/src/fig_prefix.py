"""Figure "prefix" (Expose VI): a prefix is a gate.

One query q in R^4 attends to six content tokens C and to l prefix tokens P (seeded synthetic tokens, key scale
1/sqrt(4)); a port of the still frame of site/js/fig-prefix.js. With Z_q(S) = sum_{s in S} exp(<q,k_s>/2):

  joint softmax (prefix tuning, Thm VI.5(b),(c)):
      Attn_q(P u C) = (1 - lambda_q) Attn_q(C) + lambda_q Attn_q(P),   lambda_q = Z_q(P) / (Z_q(P) + Z_q(C)),
      so every content weight is multiplied by 1 - lambda_q (ratios and order unchanged) and the output lies on the
      segment from Attn_q(C) to Attn_q(P), a point of the prefix value hull conv V_P.
  separate softmax (LLaMA-Adapter, Thm VI.5(d)):
      out_q = Attn_q(C) + g Attn_q(P),
      so the content weights are exactly the no-prefix ones and the output leaves the segment.

(a) prefix tuning and (b) LLaMA-Adapter: attention weights over [P ; C] with the no-prefix content weights as
outlines, and below them the ratio after / before for every content token. (c) the value space, projected on the
two leading principal axes of the value vectors shown.

Everything printed in the figure is computed below; the joint softmax is evaluated directly over P u C and then
checked against the identity. Run from paper/figures/src:  python3 fig_prefix.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe

from style import apply_style, C, W1, save

DK, NC, LMAX = 4, 6, 8          # key dimension, content tokens, prefix tokens drawn (the first ELL are used)
RHO = 2.4                       # |q|
SEED, PHI, ELL, S_PREF, G = 4, 195.0, 4, 1.2, 0.5   # draw, query direction (deg), prefix length, prefix key scale, gate
SQ = np.sqrt(DK)


# ---------------------------------------------------------------------------------------------------------------
# data: seeded synthetic tokens (the generator of site/js/fig-prefix.js, with numpy's PCG64 in place of the site's RNG)
# ---------------------------------------------------------------------------------------------------------------
def gen(seed):
    R = np.random.default_rng(1000 + 7919 * seed)
    Q, _ = np.linalg.qr(R.standard_normal((DK, DK)))
    u = Q.T                                                 # orthonormal rows u0..u3

    def mk(a, th, o2, o3):
        return a * np.cos(th) * u[0] + a * np.sin(th) * u[1] + o2 * u[2] + o3 * u[3]

    base = 2 * np.pi * R.random()
    kc = np.array([mk(1.4 + 0.75 * R.random(), base + 2 * np.pi * j / NC + 0.5 * (R.random() - 0.5),
                      0.6 * R.standard_normal(), 0.6 * R.standard_normal()) for j in range(NC)])
    thP = base + np.pi * (0.75 + 0.5 * R.random())
    kp = np.array([mk(1.05 + 0.6 * R.random(), thP + 1.7 * (R.random() - 0.5),
                      0.5 * R.standard_normal(), 0.5 * R.standard_normal()) for _ in range(LMAX)])
    vc = 0.85 * R.standard_normal((NC, DK))
    cP = R.standard_normal(DK)
    cP *= 2.5 / np.linalg.norm(cP)
    vp = cP + 0.72 * R.standard_normal((LMAX, DK))
    return dict(u=u, kc=kc, kp=kp, vc=vc, vp=vp)


def pca_plane(V):
    """Mean and the two leading principal axes of the rows of V; the share of variance they keep."""
    mu = V.mean(0)
    w, E = np.linalg.eigh(np.cov((V - mu).T, bias=True))
    idx = np.argsort(w)[::-1]
    return mu, E[:, idx[:2]], w[idx[:2]].sum() / np.clip(w, 0, None).sum()


def farthest_order(P2):
    """Farthest-point order of 2-D points, starting from the one farthest from their centroid (as the site does,
    so that the first few prefix tokens already span a visible hull)."""
    c = P2.mean(0)
    order = [int(np.argmax(np.linalg.norm(P2 - c, axis=1)))]
    left = [k for k in range(len(P2)) if k != order[0]]
    while left:
        dmin = [min(np.linalg.norm(P2[k] - P2[o]) for o in order) for k in left]
        order.append(left.pop(int(np.argmax(dmin))))
    return order


# ---------------------------------------------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------------------------------------------
def attend(q, K, V, scale=1.0):
    """Softmax attention of q over the rows of (K, V), logits scale * <q,k> / sqrt(d), stable log-sum-exp."""
    lg = scale * (K @ q) / SQ
    m = lg.max()
    e = np.exp(lg - m)
    w = e / e.sum()
    return dict(w=w, out=w @ V, logZ=m + np.log(e.sum()))


def hull(P):
    """Convex hull of 2-D points (monotone chain), counter-clockwise."""
    P = sorted(map(tuple, P))
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def compute():
    M = gen(SEED)
    # order the prefix pairs by farthest point of their projected values (keys travel with their values)
    mu14, E14, _ = pca_plane(np.vstack([M['vc'], M['vp']]))
    order = farthest_order((M['vp'] - mu14) @ E14)
    kp, vp = M['kp'][order][:ELL], M['vp'][order][:ELL]
    kc, vc = M['kc'], M['vc']
    phi = np.deg2rad(PHI)
    q = RHO * (np.cos(phi) * M['u'][0] + np.sin(phi) * M['u'][1])

    Cc = attend(q, kc, vc)                                  # no prefix
    Pp = attend(q, kp, vp, S_PREF)                          # prefix alone
    # joint softmax over P u C, computed directly
    K = np.vstack([kp, kc]); V = np.vstack([vp, vc])
    scl = np.r_[np.full(ELL, S_PREF), np.ones(NC)]
    lg = scl * (K @ q) / SQ
    wJ = np.exp(lg - lg.max()); wJ /= wJ.sum()
    outJ = wJ @ V
    lam = 1.0 / (1.0 + np.exp(Cc['logZ'] - Pp['logZ']))
    prefJ, contJ = wJ[:ELL], wJ[ELL:]
    ratioJ = contJ / Cc['w']
    ident = (1 - lam) * Cc['out'] + lam * Pp['out']
    # position of the output on the segment [Attn(C), Attn(P)] in R^4
    ab = Pp['out'] - Cc['out']
    tJ = (outJ - Cc['out']) @ ab / (ab @ ab)
    offJ = np.linalg.norm(outJ - (Cc['out'] + tJ * ab))
    # LLaMA-Adapter: content softmax on its own, prefix softmax scaled by g
    C2 = attend(q, kc, vc)
    prefL = G * Pp['w']
    outL = C2['out'] + G * Pp['out']
    ratioL = C2['w'] / Cc['w']
    tL = np.clip((outL - Cc['out']) @ ab / (ab @ ab), 0, 1)
    offL = np.linalg.norm(outL - (Cc['out'] + tL * ab))

    mu, E, frac = pca_plane(np.vstack([vc, vp]))
    proj = lambda v: (np.atleast_2d(v) - mu) @ E
    # orient the axes so that the prefix values sit up and to the right
    sgn = np.sign((vp.mean(0) - vc.mean(0)) @ E)
    sgn[sgn == 0] = 1
    E = E * sgn

    R = dict(q=q, Cw=Cc['w'], Pw=Pp['w'], prefJ=prefJ, contJ=contJ, ratioJ=ratioJ, lam=lam,
             ZP=np.exp(Pp['logZ']), ZC=np.exp(Cc['logZ']),
             residJ=np.linalg.norm(outJ - ident), spreadJ=np.abs(ratioJ - (1 - lam)).max(),
             orderJ=bool(np.all(np.argsort(-contJ, kind='stable') == np.argsort(-Cc['w'], kind='stable'))),
             tJ=tJ, offJ=offJ, prefL=prefL, contL=C2['w'], ratioL=ratioL, devL=np.abs(C2['w'] - Cc['w']).max(),
             tL=tL, offL=offL, pushL=np.linalg.norm(outL - Cc['out']), gnormP=G * np.linalg.norm(Pp['out']),
             vc2=proj(vc), vp2=proj(vp), AC2=proj(Cc['out'])[0], AP2=proj(Pp['out'])[0], out2=proj(outJ)[0],
             outL2=proj(outL)[0], frac=frac)
    return R


# ---------------------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------------------
def bars(ax, axr, R, mode, ymax):
    joint = mode == 'joint'
    pref = R['prefJ'] if joint else R['prefL']
    cont = R['contJ'] if joint else R['contL']
    ratio = R['ratioJ'] if joint else R['ratioL']
    gap = 0.9
    xp = np.arange(ELL)
    xc = ELL + gap + np.arange(NC)
    bw = 0.62
    ax.bar(xp, pref, width=bw, color=C['ochre'], lw=0, zorder=3)
    # no-prefix content weights: outlines; with the prefix: filled, slightly narrower so both read
    ax.bar(xc, R['Cw'], width=bw + 0.12, facecolor=(0.055, 0.451, 0.522, 0.10), edgecolor=C['tide'], lw=0.6,
           ls=(0, (2.2, 1.4)), zorder=2)
    ax.bar(xc, cont, width=bw - 0.12, color=C['tide'], lw=0, zorder=3)
    ax.set_xlim(-0.6, xc[-1] + 0.6)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.2, 0.4] if ymax > 0.4 else [0, 0.1, 0.2, 0.3])
    ax.set_xticks(np.r_[xp, xc])
    ax.set_xticklabels([f'$p_{i + 1}$' for i in range(ELL)] + [f'$c_{j + 1}$' for j in range(NC)], fontsize=7.5)
    for t in ax.get_xticklabels()[:ELL]:
        t.set_color(C['ochre'])
    for t in ax.get_xticklabels()[ELL:]:
        t.set_color(C['tide'])
    ax.tick_params(axis='x', length=0, pad=2)
    # group masses
    mp = R['lam'] if joint else G
    mc = 1 - R['lam'] if joint else 1.0
    # left-aligned just inside the axis, so the label never runs onto the y spine
    ax.text(-0.4, ymax * 0.985, (r'$\Sigma=\lambda_q=$' if joint else r'$\Sigma=g=$') + f'{mp:.2f}',
            ha='left', va='top', fontsize=7.5, color=C['ochre'])
    ax.text(xc.mean(), ymax * 0.985, (r'$\Sigma=1-\lambda_q=$' + f'{mc:.2f}') if joint else r'$\Sigma=1$',
            ha='center', va='top', fontsize=7.5, color=C['tide'])

    # ratio strip: after / before for every content token
    axr.set_xlim(ax.get_xlim())
    axr.set_ylim(0, 1.3)
    axr.axhline(1, color=C['ink3'], lw=0.5, ls=(0, (1.5, 2)), zorder=1)
    axr.plot(xc, ratio, color=C['ink'], lw=1.0, zorder=3)
    axr.plot(xc, ratio, 'o', ms=3.4, mfc='white', mec=C['ink'], mew=0.9, zorder=4)
    axr.set_yticks([0, 1])
    axr.set_xticks([])
    axr.spines['bottom'].set_visible(False)
    lab = ('after $\\div$ before\n' + (r'$=1-\lambda_q$' if joint else '$=1$ exactly'))
    axr.text(xp[0] - 0.4, 0.08, lab, ha='left', va='bottom', fontsize=7.5, color=C['ink'], linespacing=1.15)


def value_space(ax, R):
    vc2, vp2 = R['vc2'], R['vp2']
    hc = hull(vc2)
    ax.add_patch(Polygon(hc, closed=True, fc=(0.055, 0.451, 0.522, 0.07), ec=C['tide'], lw=0.6,
                         ls=(0, (2.5, 2)), zorder=1))
    hp = hull(vp2)
    ax.add_patch(Polygon(hp, closed=True, fc=(0.659, 0.447, 0.047, 0.16), ec=C['ochre'], lw=0.8, zorder=1))
    ax.plot(vc2[:, 0], vc2[:, 1], 'o', ms=3.2, color=C['tide'], mew=0, zorder=3)
    ax.plot(vp2[:, 0], vp2[:, 1], 'D', ms=3.0, color=C['ochre'], mew=0, zorder=3)
    AC, AP, out, outL = R['AC2'], R['AP2'], R['out2'], R['outL2']
    ax.plot([AC[0], AP[0]], [AC[1], AP[1]], color=C['ink3'], lw=0.7, zorder=2)
    ax.annotate('', xy=out, xytext=AC, arrowprops=dict(arrowstyle='-|>', color=C['ink'], lw=1.1,
                                                       shrinkA=2.5, shrinkB=2.5, mutation_scale=7), zorder=4)
    ax.annotate('', xy=outL, xytext=AC, arrowprops=dict(arrowstyle='-|>', color=C['ink3'], lw=0.9,
                                                        ls=(0, (2, 1.5)), shrinkA=2.5, shrinkB=3, mutation_scale=6),
                zorder=4)
    ax.plot(*AC, 'o', ms=5.2, mfc='white', mec=C['tide'], mew=1.2, zorder=5)
    ax.plot(*AP, 'D', ms=4.8, mfc='white', mec=C['ochre'], mew=1.2, zorder=5)
    ax.plot(*out, 'o', ms=4.6, color=C['ink'], zorder=6)
    ax.plot(*outL, 'o', ms=4.4, mfc='white', mec=C['ink3'], mew=1.0, zorder=6)
    return AC, AP, out, outL


def fit_width(fig, width, iters=5):
    """Rescale the figure so that its tight bounding box (what save() writes) is `width` inches wide: text keeps its
    point size at print size when the PDF is included at \\linewidth."""
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
    R = compute()
    # the identities of Thm VI.5(b)-(d), to machine precision
    assert R['residJ'] < 1e-12 and R['spreadJ'] < 1e-12 and R['orderJ']
    assert R['devL'] == 0.0 and abs(R['tJ'] - R['lam']) < 1e-12 and R['offJ'] < 1e-12
    print(f"seed {SEED}, phi {PHI:.0f} deg, |q| = {RHO}, l = {ELL}, s = {S_PREF}, g = {G}")
    print(f"Z(P) = {R['ZP']:.4f}, Z(C) = {R['ZC']:.4f}, lambda_q = {R['lam']:.4f}, 1 - lambda_q = {1 - R['lam']:.4f}")
    print('content weights, no prefix  :', np.round(R['Cw'], 4))
    print('content weights, joint      :', np.round(R['contJ'], 4))
    print('ratios (joint)              :', np.round(R['ratioJ'], 12))
    print('prefix weights, joint       :', np.round(R['prefJ'], 4), ' sum', round(R['prefJ'].sum(), 4))
    print('prefix weights, LLaMA (g*w) :', np.round(R['prefL'], 4), ' sum', round(R['prefL'].sum(), 4))
    print(f"identity residual {R['residJ']:.2e}, ratio spread {R['spreadJ']:.2e}, order kept {R['orderJ']}")
    print(f"output at t = {R['tJ']:.4f} of the way from Attn(C) to Attn(P) (lambda_q = {R['lam']:.4f}), "
          f"off the segment by {R['offJ']:.1e}")
    print(f"LLaMA-Adapter: max |a'_j - a_j| = {R['devL']:.1e}, |out - Attn(C)| = {R['pushL']:.4f} = g|Attn(P)| = "
          f"{R['gnormP']:.4f}, distance from the segment {R['offL']:.4f}")
    print(f"PCA plane keeps {100 * R['frac']:.1f}% of the value variance")

    fig = plt.figure(figsize=(W1, 2.3))
    # (a), (b): bars with a ratio strip under each; (c): value space
    L, Wb, gapb = 0.06, 0.245, 0.04
    yb, hb, yr, hr = 0.34, 0.6, 0.05, 0.19
    xC, WC = L + 2 * Wb + gapb + 0.035, 0.36
    axA = fig.add_axes([L, yb, Wb, hb]); axAr = fig.add_axes([L, yr, Wb, hr])
    axB = fig.add_axes([L + Wb + gapb, yb, Wb, hb]); axBr = fig.add_axes([L + Wb + gapb, yr, Wb, hr])
    axC = fig.add_axes([xC, yr, WC, yb + hb - yr])
    top = max(R['Cw'].max(), R['prefJ'].max(), R['prefL'].max())
    ymax = 0.5 if top < 0.44 else 0.6
    bars(axA, axAr, R, 'joint', ymax)
    bars(axB, axBr, R, 'llama', ymax)
    axB.set_yticklabels([])
    axBr.set_yticklabels([])
    axA.set_ylabel('attention weight', labelpad=2)

    AC, AP, out, outL = value_space(axC, R)
    pts = np.vstack([R['vc2'], R['vp2'], R['outL2'][None]])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.1 * (hi - lo).max()
    axC.set_xlim(lo[0] - pad, hi[0] + pad)
    axC.set_ylim(lo[1] - pad, hi[1] + 1.6 * pad)
    axC.set_aspect('equal', adjustable='datalim')
    axC.set_xticks([]); axC.set_yticks([])
    for sp in axC.spines.values():
        sp.set_visible(False)
    axC.text(1.0, 1.0, f'value space, PCA plane\n({100 * R["frac"]:.0f}% of the variance)', transform=axC.transAxes,
             ha='right', va='top', fontsize=7, color=C['ink2'], linespacing=1.1)
    # labels for the key points, offset in points
    def lab(xy, txt, dx, dy, ha, va, col, fs=7.5, **kw):
        axC.annotate(txt, xy=xy, xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, color=col, fontsize=fs,
                     path_effects=[pe.withStroke(linewidth=2.2, foreground='white')], zorder=7, **kw)
    lab(AC, r'$\mathrm{Attn}_q(C)$', 12, -5, 'center', 'top', C['tide'])
    lab(AP, r'$\mathrm{Attn}_q(P)$', 0, -5, 'center', 'top', C['ochre'])
    lab(out, 'out', 0, 5, 'center', 'bottom', C['ink'])
    lab(outL, 'LLaMA-Adapter', 0, 5, 'center', 'bottom', C['ink2'], fs=7)
    vp2, vc2 = R['vp2'], R['vc2']
    lab(vp2[np.argmax(vp2[:, 1])], r'$\mathrm{conv}\,V_P$', 0, 4, 'center', 'bottom', C['ochre'])
    lab(vc2[np.argmax(vc2[:, 1])], r'$\mathrm{conv}\,V_C$', 0, 4, 'center', 'bottom', C['tide'])

    ytop = yb + hb + 0.035
    for x_, lab_ in ((0.0, '(a)'), (L + Wb + gapb - 0.005, '(b)'), (xC, '(c)')):
        fig.text(x_, ytop, lab_, ha='left', va='bottom', fontsize=8.5, fontweight='semibold', color=C['ink'],
                 family='sans-serif')
    fit_width(fig, W1)
    print('wrote', save(fig, 'prefix'))
    return R


if __name__ == '__main__':
    main()
