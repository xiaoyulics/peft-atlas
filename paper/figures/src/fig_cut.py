"""Figure "cut" (Expose II.C): cut the tensor.

Thm II.11 (Kronecker incomparability) and Prop II.13 (sum_{i<=k} C_i (x) D_i = R^{-1}_* LoRA_k) of theory/framework.md;
theory/visuals.md section 8. A port of site/js/fig-cut.js (4 (x) 4 split, presets uv^T, I_16 and C (x) D).

Delta in R^{16x16} is read as a 4-leg tensor Delta[(u1,u2),(v1,v2)] with R^16 = R^4 (x) R^4 on both sides, in the
row-major Kronecker convention i = 4 u1 + u2, j = 4 v1 + v2, so that (C (x) D)[i, j] = C[u1, v1] D[u2, v2]
(numpy.kron). The three 2|2 cuts of the four legs are reshapes of the same 256 numbers:
    cut 1  {U1U2 | V1V2}   Delta itself                       16 x 16    rank               (LoRA factors here)
    cut 2  {U1V1 | U2V2}   R Delta [(u1,v1),(u2,v2)]          16 x 16    Kronecker rank     (Van Loan-Pitsianis)
    cut 3  {U1V2 | U2V1}   R(Delta Pi^T) [(u1,v2),(u2,v1)]    16 x 16    (Pi swaps V1 and V2)
Each reshape permutes the entries, so sum_i sigma_i^2 = ||Delta||_F^2 on every cut (Delta is normalised to 1). The
singular values across a cut are the operator-Schmidt coefficients for that grouping of legs. Numerical rank =
#{sigma_i > 1e-10 ||Delta||_F}. A bond of dimension k across any of the three cuts costs k (4*4 + 4*4) = 32 k
parameters on this split.
Expected (Thm II.11(b), (c)): generic uv^T has ranks (1, kappa_1 = 16, 16); I_16 = I_4 (x) I_4 and C (x) D have
ranks (16, 1, 16).

Run from paper/figures/src:  python3 fig_cut.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from style import apply_style, C, W1, save

N, S = 16, 4                 # Delta in R^{16x16}, split 16 = 4 * 4 on both sides
m1 = m2 = n1 = n2 = S
TOL = 1e-10
FLOOR = 10 ** -17.0          # values below it, exact zeros included, are drawn at the floor
RNG = np.random.default_rng(20260211)


def cut_matrices(D):
    T = D.reshape(m1, m2, n1, n2)                              # T[u1, u2, v1, v2]
    return [D,
            T.transpose(0, 2, 1, 3).reshape(m1 * n1, m2 * n2),     # rows (u1, v1), cols (u2, v2)
            T.transpose(0, 3, 1, 2).reshape(m1 * n2, m2 * n1)]     # rows (u1, v2), cols (u2, v1)


def spectra(D):
    D = D / np.linalg.norm(D)
    out = []
    for X in cut_matrices(D):
        s = np.linalg.svd(X, compute_uv=False)
        out.append(dict(s=s, rank=int((s > TOL).sum()), energy=float((s ** 2).sum())))
    return out


def updates():
    u, v = RNG.standard_normal(N), RNG.standard_normal(N)
    Cm, Dm = RNG.standard_normal((m1, n1)), RNG.standard_normal((m2, n2))
    return [
        ('uv', np.outer(u, v)),
        ('id', np.eye(N)),
        ('cd', np.kron(Cm, Dm)),
    ]


EXPECT = {'uv': (1, min(m1, m2) * min(n1, n2), 16), 'id': (16, 1, 16), 'cd': (16, 1, 16)}
CUTCOL = [C['tide'], C['ochre'], C['ink2']]
CUTNAME = ['ordinary cut', 'Kronecker cut', 'third cut']
CUTLEGS = [r'$U_1U_2\,|\,V_1V_2$', r'$U_1V_1\,|\,U_2V_2$', r'$U_1V_2\,|\,U_2V_1$']
BOXES = [('B', 'A', ('U_1', 'U_2'), ('V_1', 'V_2')),
         ('C', 'D', ('U_1', 'V_1'), ('U_2', 'V_2')),
         ('E', 'F', ('U_1', 'V_2'), ('U_2', 'V_1'))]
WHO = ['LoRA\n' + r'$\Delta=BA$', 'Kronecker sum\n' + r'$\Delta=\sum C_i\otimes D_i$',
       r'after $V_1\leftrightarrow V_2$' + '\n' + r'$\Delta\Pi^{\top}=\sum E_i\otimes F_i$']
ROWS = {'uv': ('(a)', r'$uv^{\top}$', 'random,\nrank one'),
        'id': ('(b)', r'$I_{16}$', r'$=I_4\otimes I_4$'),
        'cd': ('(c)', r'$C\otimes D$', 'random\nfactors')}


def glyph(ax, c, col):
    """Two boxes joined by the bond that crosses cut c; each box carries its two legs (equal-aspect units)."""
    left, right, lleg, rleg = BOXES[c]
    ax.set_xlim(0, 7.2); ax.set_ylim(0, 3.0); ax.set_aspect('equal'); ax.axis('off')
    yc, xl, xr, rad = 1.5, 2.6, 4.6, 0.5
    for (x, name) in ((xl, left), (xr, right)):
        ax.add_patch(Circle((x, yc), rad, fc='white', ec=C['ink'], lw=0.8, zorder=3))
        ax.text(x, yc - 0.02, f'${name}$', ha='center', va='center', fontsize=7.5, color=C['ink'], zorder=4)
    ax.plot([xl + rad, xr - rad], [yc, yc], color=col, lw=1.8, solid_capstyle='butt', zorder=2)
    for sgn, x0, labs, ha in ((-1, xl, lleg, 'right'), (1, xr, rleg, 'left')):
        for dy, lab in zip((1, -1), labs):
            a = np.array([x0 + sgn * rad * 0.71, yc + dy * rad * 0.71])
            b = np.array([x0 + sgn * 1.25, yc + dy * 1.05])
            ax.plot([a[0], b[0]], [a[1], b[1]], color=C['ink'], lw=0.8)
            ax.text(b[0] + sgn * 0.12, b[1], f'${lab}$', ha=ha, va='center', fontsize=7.5, color=C['ink'])
    ax.plot([3.6, 3.6], [yc - 0.62, yc + 0.62], color=C['ink2'], lw=0.8, ls=(0, (2, 1.4)))


def draw(data):
    H = 3.80
    fig = plt.figure(figsize=(W1, H))
    left, right = 0.170, 0.995
    colw_gap = 0.035
    cw = (right - left - 2 * colw_gap) / 3
    head_h, row_h, row_gap, bottom = 0.72 / H, 0.78 / H, 0.13 / H, 0.30 / H
    xs = [left + c * (cw + colw_gap) for c in range(3)]
    top_rows = 1 - head_h

    # column headers: name, legs, the factorisation picture across the cut
    for c in range(3):
        fig.text(xs[c] + cw / 2, 1 - 0.02 / H, CUTNAME[c] + '   ' + CUTLEGS[c], ha='center', va='top', fontsize=8,
                 color=CUTCOL[c], fontweight='semibold')
        gh = 0.42                                            # glyph height in inches, width from the aspect
        gax = fig.add_axes([xs[c] - 0.01, 1 - 0.66 / H, gh * 7.2 / 3.0 / W1, gh / H])
        glyph(gax, c, CUTCOL[c])
        fig.text(xs[c] + cw * 0.60, 1 - 0.45 / H, WHO[c], ha='left', va='center', fontsize=7, color=C['ink2'],
                 linespacing=1.25)

    ylo, yhi = 10 ** -18.2, 10 ** 1.0
    for r, (key, D) in enumerate(data):
        y0 = top_rows - (r + 1) * row_h - r * row_gap
        lab, sym, sub = ROWS[key]
        fig.text(0.004, y0 + row_h * 0.86, lab, ha='left', va='top', fontsize=8.5, fontweight='semibold', color=C['ink'])
        fig.text(0.004, y0 + row_h * 0.62, sym, ha='left', va='top', fontsize=9, color=C['ink'])
        fig.text(0.004, y0 + row_h * 0.30, sub, ha='left', va='top', fontsize=7, color=C['ink2'], linespacing=1.1)
        sp = D['spectra']
        for c in range(3):
            ax = fig.add_axes([xs[c], y0, cw, row_h])
            s = sp[c]['s']
            on = s > TOL
            idx = np.arange(1, len(s) + 1)
            sv = np.maximum(s, FLOOR)
            ax.plot(idx[on], sv[on], 'o', ms=3.4, color=CUTCOL[c], mec='white', mew=0.4, zorder=3)
            ax.plot(idx[~on], sv[~on], 'o', ms=2.6, mfc='none', mec=C['ink3'], mew=0.6, zorder=3)
            ax.axhline(TOL, color=C['ink3'], lw=0.5, ls=(0, (3, 2)), zorder=0)
            ax.set_yscale('log')
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(0.2, 16.8)
            ax.set_yticks([1, 1e-8, 1e-16])
            ax.yaxis.set_minor_locator(plt.NullLocator())
            if c == 0:
                ax.set_yticklabels(['1', r'$10^{-8}$', r'$10^{-16}$'])
            else:
                ax.set_yticklabels([])
            ax.set_xticks([1, 4, 8, 12, 16])
            if r == 2:
                ax.set_xticklabels(['1', '4', '8', '12', '16'])
            else:
                ax.set_xticklabels([])
            rk = sp[c]['rank']
            # the rank sits in the empty band between the numerically zero values and the nonzero spectrum
            ax.text(16.6, 10 ** -5.5, f'rank {rk}', ha='right', va='center', fontsize=8, fontweight='semibold',
                    color=CUTCOL[c], zorder=5)
            ax.text(16.6, 10 ** -8.1, f'{32 * rk} parameters', ha='right', va='center', fontsize=7,
                    color=C['ink2'], zorder=5)
            if r == 0 and c == 0:
                ax.text(2.0, TOL * 10 ** 0.3, 'threshold', ha='left', va='bottom', fontsize=7, color=C['ink3'])
    fig.text(xs[0] - 0.066, top_rows - 1.5 * row_h - row_gap, r'$\sigma_i\,/\,\|\Delta\|_F$', rotation=90,
             ha='center', va='center', fontsize=8.5, color=C['ink'])
    fig.text(xs[1] + cw / 2, 0.012, r'index $i$ of the singular value across the cut', ha='center', va='bottom',
             fontsize=8, color=C['ink'])
    return fig


def main():
    data = []
    ok = True
    print('-- operator-Schmidt spectra across the three cuts of a 16 x 16 matrix, 4 (x) 4 split --')
    for key, D in updates():
        sp = spectra(D)
        ranks = tuple(x['rank'] for x in sp)
        good = ranks == EXPECT[key]
        ok &= good
        print(f'  {key}: ranks {ranks} expected {EXPECT[key]} {"ok" if good else "MISMATCH"};  '
              f'sum sigma^2 on each cut {[round(x["energy"], 15) for x in sp]};  '
              f'largest numerically-zero value {max([x["s"][x["rank"]] if x["rank"] < 16 else 0 for x in sp]):.1e};  '
              f'smallest nonzero {min(x["s"][x["rank"] - 1] for x in sp):.1e}')
        data.append((key, dict(spectra=sp)))
    apply_style()
    fig = draw(data)
    print('wrote', save(fig, 'cut'))
    if not ok:
        raise SystemExit('ranks differ from Thm II.11')


if __name__ == '__main__':
    main()
