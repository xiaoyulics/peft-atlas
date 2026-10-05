"""Figure "rankvol" (Expose II.C): rank is not volume.

Prop II.12 (gauge census: rank is a cut, dimension is a volume) and Thm II.10 (cut-rank bound) of theory/framework.md;
HiRA through Prop II.9(a) and Prop II.13 (transport of structure); KronA through Thm II.11.
A port of site/js/fig-rankvol.js, extended by KronA and MoRA.

One 4096 x 4096 layer and the budget |M| = R(m+n) of LoRA_R, R = 8, so |M| = 65,536. Each method is placed by
    x = the maximum rank of Delta W (a cut, Thm II.10), log scale,
    y = d(M) / (mn), its expressive dimension as a fraction of the layer (a volume, Prop II.12).
Census at this budget (Prop II.12(b)):
    LoRA_R           d = R(m+n-R)                                  rank R
    LoHa_{r,r}       d <= F = 2 r(m+n-r) - (m+n-1), r = R/2         rank r^2        (F a proved upper bound; equality [Num])
    GraLoRA_k        d = R(m+n-R)                                  rank min(kR, m, n)
    HiRA_R           d = R(m+n-R)  (W0 without zero entries)       rank min(R rk W0, m, n) = min(m, n) for generic W0
    KronA A (x) B    d = m1 n1 + m2 n2 - 1  (gauge R^x)            rank min(m1, n1) min(m2, n2)
                     its budget is fixed by the factor shapes; the shapes closest to |M| are used (no exact match
                     exists on 4096 x 4096 for R = 8)
    MoRA, r_hat      Delta W = I_{n / r_hat} (x) M, r_hat^2 = |M|; linear and injective: d = |M|, rank min(m, n)
    FourierFT, n_c   n_c = |M| frequencies; d = n_c - #(conjugate pairs); rank <= min(2 n_c, m, n) (Thm II.10)
                     the frequencies are HF PEFT's default draw: torch.randperm(m n, seed 777)[:n_c]
    C3A, b           b = mn / |M| (b must divide m and n); b x b circulant blocks, injective: d = |M|, rank min(m, n)
Every formula is first checked by a float64 Jacobian rank (exact central differences: every map here is affine in each
single coordinate) and by the rank of Delta W at a seeded random point, on a 64 x 64 copy with the same R, the same
GraLoRA k, the analogous KronA shape and MoRA / C3A sizes (|M| = 1024 = 32^2 there).

Run from paper/figures/src:  python3 fig_rankvol.py   (--no-check skips the small-copy Jacobian check)
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import apply_style, C, W1, save

SEED_FT = 777   # HF PEFT's default fourierft_random_loc_seed
RNG = np.random.default_rng(20260412)


# ---------------------------------------------------------------------------------------------------------------
# FourierFT frequencies and conjugate pairs
# ---------------------------------------------------------------------------------------------------------------
def fourier_indices(m, n, nc, seed=SEED_FT):
    """HF PEFT's draw: torch.randperm(m*n, generator=manual_seed(seed))[:nc]; numpy fallback without torch."""
    try:
        import torch
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(m * n, generator=g)[:nc].numpy().astype(np.int64), 'torch.randperm'
    except ImportError:  # pragma: no cover
        return np.random.default_rng(seed).choice(m * n, nc, replace=False).astype(np.int64), 'numpy'


def conjugate_pairs(idx, m, n):
    """#{p, -p} both drawn with p != -p, and #{p = -p} (self-conjugate), frequencies taken mod (m, n)."""
    u, v = idx // n, idx % n
    conj = ((-u) % m) * n + ((-v) % n)
    s = set(idx.tolist())
    selfc = int((conj == idx).sum())
    pairs = sum(1 for a, c in zip(idx.tolist(), conj.tolist()) if a < c and c in s)
    return pairs, selfc


# ---------------------------------------------------------------------------------------------------------------
# the census (Prop II.12(b)) at one budget
# ---------------------------------------------------------------------------------------------------------------
def divisors(x):
    return [d for d in range(1, x + 1) if x % d == 0]


def krona_shape(m, n, P):
    """KronA factor shapes (m1 x n1) (x) (m2 x n2) with budget closest to P; ties: larger max rank."""
    best = None
    for m1 in divisors(m):
        for n1 in divisors(n):
            m2, n2 = m // m1, n // n1
            if min(m1, n1, m2, n2) < 2:
                continue
            bud = m1 * n1 + m2 * n2
            rk = min(m1, n1) * min(m2, n2)
            key = (abs(bud - P), -rk, -m1)
            if best is None or key < best[0]:
                best = (key, (m1, n1, m2, n2), bud, rk)
    return best[1], best[2], best[3]


def census(m, n, R, k):
    P, s, mn, mi = R * (m + n), m + n, m * n, min(m, n)
    L = {}
    L['lora'] = dict(P=P, d=R * (s - R), rank=R, bound=False, gauge=R * R)
    r = R // 2
    F = 2 * r * (s - r) - (s - 1)
    L['loha'] = dict(P=P, d=min(mn, F), rank=min(r * r, m, n), bound=False, r=r, dle=True,
                     gauge=2 * r * r + s - 1)
    assert m % k == 0 and n % k == 0 and R % k == 0
    L['gralora'] = dict(P=P, d=R * (s - R), rank=min(k * R, m, n), bound=False, k=k, gauge=R * R)
    L['hira'] = dict(P=P, d=R * (s - R), rank=mi, bound=False, gauge=R * R)
    shp, bud, rk = krona_shape(m, n, P)
    L['krona'] = dict(P=bud, d=bud - 1, rank=rk, bound=False, shape=shp, gauge=1)
    rh = math.isqrt(P)
    assert rh * rh == P and m % rh == 0 and n % rh == 0, 'MoRA needs r_hat^2 = |M| with r_hat | m, n'
    L['mora'] = dict(P=P, d=P, rank=mi, bound=False, rhat=rh, gauge=0)
    idx, src = fourier_indices(m, n, P)
    pairs, selfc = conjugate_pairs(idx, m, n)
    L['fourier'] = dict(P=P, d=P - pairs, rank=min(2 * P, m, n), bound=True, pairs=pairs, selfc=selfc, idx=idx,
                        src=src, gauge=pairs)
    b = mn // P
    assert mn % P == 0 and m % b == 0 and n % b == 0, 'C3A needs b = mn/|M| dividing m and n'
    L['c3a'] = dict(P=P, d=P, rank=mi, bound=False, b=b, gauge=0)
    return dict(m=m, n=n, R=R, P=P, mn=mn, L=L)


# ---------------------------------------------------------------------------------------------------------------
# parametrisations (for the Jacobian check on a small copy)
# ---------------------------------------------------------------------------------------------------------------
def map_lora(m, n, R):
    def f(q):
        return q[:m * R].reshape(m, R) @ q[m * R:].reshape(R, n)
    return f, R * (m + n)


def map_loha(m, n, r):
    fl, p = map_lora(m, n, r)
    return (lambda q: fl(q[:p]) * fl(q[p:])), 2 * p


def map_gralora(m, n, R, k):
    bm, bn, br = m // k, n // k, R // k

    def f(q):
        out, i = np.zeros((m, n)), 0
        for I in range(k):
            for J in range(k):
                B = q[i:i + bm * br].reshape(bm, br); i += bm * br
                A = q[i:i + br * bn].reshape(br, bn); i += br * bn
                out[I * bm:(I + 1) * bm, J * bn:(J + 1) * bn] = B @ A
        return out
    return f, R * (m + n)


def map_hira(m, n, R, W0):
    fl, p = map_lora(m, n, R)
    return (lambda q: W0 * fl(q)), p


def map_krona(m1, n1, m2, n2):
    def f(q):
        return np.kron(q[:m1 * n1].reshape(m1, n1), q[m1 * n1:].reshape(m2, n2))
    return f, m1 * n1 + m2 * n2


def map_mora(m, n, rh):
    assert m == n
    return (lambda q: np.kron(np.eye(n // rh), q.reshape(rh, rh))), rh * rh


def map_fourier(m, n, idx):
    u, v = idx // n, idx % n
    I, J = np.arange(m)[:, None], np.arange(n)[None, :]
    pats = np.stack([np.cos(2 * np.pi * (uu * I / m + vv * J / n)) / (m * n) for uu, vv in zip(u, v)])
    return (lambda q: np.tensordot(q, pats, axes=1)), len(idx)


def map_c3a(m, n, b):
    gm, gn = m // b, n // b
    rr, cc = np.meshgrid(np.arange(b), np.arange(b), indexing='ij')
    shift = (cc - rr) % b   # circ(w)[r, c] = w[(c - r) mod b]

    def f(q):
        out, off = np.zeros((m, n)), 0
        for I in range(gm):
            for J in range(gn):
                out[I * b:(I + 1) * b, J * b:(J + 1) * b] = q[off:off + b][shift]
                off += b
        return out
    return f, gm * gn * b


def jacobian(f, q):
    """Exact for maps affine in each single coordinate: J e_i = (f(q + e_i) - f(q - e_i)) / 2."""
    cols, e = [], np.zeros_like(q)
    for i in range(q.size):
        e[i] = 1.0
        cols.append(0.5 * (f(q + e) - f(q - e)).ravel())
        e[i] = 0.0
    return np.array(cols).T


def nrank(X, tol=1e-9):
    s = np.linalg.svd(X, compute_uv=False)
    return int((s > tol * s[0]).sum())


def certify(m=64, n=64, R=8, k=2):
    """Jacobian ranks and max ranks on a small copy against the census formulas."""
    Cs = census(m, n, R, k)
    L = Cs['L']
    W0 = RNG.standard_normal((m, n))
    maps = {
        'lora': map_lora(m, n, R), 'loha': map_loha(m, n, R // 2), 'gralora': map_gralora(m, n, R, k),
        'hira': map_hira(m, n, R, W0), 'krona': map_krona(*L['krona']['shape']),
        'mora': map_mora(m, n, L['mora']['rhat']), 'fourier': map_fourier(m, n, L['fourier']['idx']),
        'c3a': map_c3a(m, n, L['c3a']['b']),
    }
    ok = True
    print(f'-- Jacobian check on a {m} x {n} copy, R = {R}, |M| = {Cs["P"]} --')
    # numpy on Apple Accelerate raises spurious floating-point warnings in small matmuls; the products are finite
    np.seterr(all='ignore')
    for key, (f, p) in maps.items():
        assert p == L[key]['P'], (key, p, L[key]['P'])
        q = RNG.standard_normal(p)
        d = nrank(jacobian(f, q))
        rk = nrank(f(q))
        good = d == L[key]['d'] and rk == L[key]['rank']
        ok &= good
        print(f'  {key:8s} |M| {p:5d}  d: Jacobian {d:5d} formula {L[key]["d"]:5d}   '
              f'rank: at a random point {rk:3d} formula {L[key]["rank"]:3d}   {"ok" if good else "MISMATCH"}')
    print('  all formulas confirmed' if ok else '  MISMATCH: see above')
    return ok


# ---------------------------------------------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------------------------------------------
NAME = {'lora': 'LoRA₈', 'loha': 'LoHa₄,₄', 'gralora': 'GraLoRA', 'hira': 'HiRA₈', 'krona': 'KronA',
        'mora': 'MoRA', 'fourier': 'FourierFT', 'c3a': 'C3A'}
TIDE_TINT = '#7FB3BD'   # tide mixed half with white: GraLoRA at other block counts
# (|M| is typeset with the sans bar: STIX's mathtext bar is too short at 7 pt)
COL = {'lora': C['ink'], 'loha': C['seal'], 'gralora': C['tide'], 'hira': C['purple'], 'krona': C['blue'],
       'mora': C['vermillion'], 'fourier': C['ochre'], 'c3a': C['moss']}


def draw(Cs):
    m, n, R, P, mn, L = Cs['m'], Cs['n'], Cs['R'], Cs['P'], Cs['mn'], Cs['L']
    sc = 1e3 / mn                                   # y in units of 10^-3 of mn
    y = {k: v['d'] * sc for k, v in L.items()}
    dL, ceil = L['lora']['d'] * sc, P * sc

    fig = plt.figure(figsize=(W1, 2.85))
    # two bands of one y axis, at the same data scale (units per inch), with a break between them
    top_lo, top_hi = (L['fourier']['d'] - 100) * sc, (L['krona']['d'] + 55) * sc
    k_per_in = (top_hi - top_lo) / 1.62
    bot_c = L['loha']['d'] * sc
    bot_lo, bot_hi = bot_c - 0.35 * k_per_in, bot_c + 0.35 * k_per_in
    left, right, bottom = 0.105, 0.985, 0.155
    H = 2.85
    hb, ht, gap = 0.70 / H, 1.62 / H, 0.10 / H
    axb = fig.add_axes([left, bottom, right - left, hb])
    axt = fig.add_axes([left, bottom + hb + gap, right - left, ht], sharex=axb)
    axt.set_ylim(top_lo, top_hi)
    axb.set_ylim(bot_lo, bot_hi)
    xlo, xhi = 5.2, 6300
    axb.set_xscale('log', base=2)
    axb.set_xlim(xlo, xhi)

    for ax in (axt, axb):
        ax.tick_params(axis='y', length=2.5)
    axt.spines['bottom'].set_visible(False)
    axt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ticks = [2 ** e for e in range(3, 13)]
    axb.set_xticks(ticks)
    axb.set_xticklabels([str(t) for t in ticks])
    axb.xaxis.set_minor_locator(plt.NullLocator())
    axt.set_yticks([3.90, 3.91, 3.92])
    axt.set_yticklabels(['3.90', '3.91', '3.92'])
    axb.set_yticks([3.41, 3.42])
    axb.set_yticklabels(['3.41', '3.42'])
    # break marks on the y axis
    for ax, yy in ((axt, 0.0), (axb, 1.0)):
        ax.plot([-0.012, 0.012], [yy - 0.02 / (ax.get_position().height * H), yy + 0.02 / (ax.get_position().height * H)],
                transform=ax.transAxes, color=C['ink3'], lw=0.6, clip_on=False)
    axb.set_xlabel(r'maximum rank of $\Delta W$  (a cut; log scale)', labelpad=3)
    fig.text(0.012, bottom + (hb + gap + ht) / 2, r'$d(M)\,/\,mn$   ($\times10^{-3}$)', rotation=90, va='center', ha='left',
             fontsize=8.5, color=C['ink'])

    # reference lines: full rank, |M| (no gauge), d(LoRA_8)
    for ax in (axt, axb):
        ax.axvline(min(m, n), color=C['ink3'], lw=0.7, ls=(0, (3, 2.5)), zorder=1)
    axb.text(min(m, n) * 1.09, bot_c, 'full rank', rotation=90, ha='left', va='center', fontsize=7, color=C['ink2'])
    axt.axhline(ceil, color=C['ink2'], lw=0.7, ls=(0, (5, 2.5)), zorder=1)
    axt.text(xlo * 1.04, ceil + 0.0011, f'$d=$|$M$| = {P:,}: no gauge', fontsize=7, color=C['ink2'], va='bottom')
    axt.axhline(dL, color=C['ink'], lw=0.6, ls=(0, (1, 2)), zorder=1)
    axt.text(92, dL - 0.0011, f'$d=$|$M$|$\\,-\\,8^2$ = {L["lora"]["d"]:,}', fontsize=7, color=C['ink2'], va='top')

    # GraLoRA at other block counts k (same dimension, higher rank)
    for kk in (4, 8):
        xr = min(kk * R, m, n)
        axt.plot([xr], [dL], 'o', ms=4.6, mfc=TIDE_TINT, mec='white', mew=0.6, zorder=4)   # exact ranks: filled
        axt.text(xr, dL - 0.0012, f'$k$={kk}', fontsize=7, color=C['tide'], ha='center', va='top')

    # points; MoRA and C3A coincide and are dodged to either side of the full-rank line
    dodge = {'mora': 2 ** -0.13, 'c3a': 2 ** 0.13}
    for key in ['lora', 'gralora', 'hira', 'krona', 'mora', 'c3a', 'fourier', 'loha']:
        E = L[key]
        ax = axb if key == 'loha' else axt
        xx = E['rank'] * dodge.get(key, 1.0)
        if E['bound']:
            ax.plot([xx], [y[key]], 'o', ms=5.2, mfc='white', mec=COL[key], mew=1.4, zorder=5)
        else:
            ax.plot([xx], [y[key]], 'o', ms=5.4, mfc=COL[key], mec='white', mew=0.7, zorder=5)

    lab = dict(fontsize=7.5, fontweight='medium', zorder=6)
    axt.text(L['lora']['rank'] * 0.93, dL + 0.0012, NAME['lora'], color=COL['lora'], ha='center', va='bottom', **lab)
    axt.text(L['gralora']['rank'] * 1.0, dL + 0.0012, 'GraLoRA $k$=2', color=COL['gralora'], ha='left', va='bottom', **lab)
    axt.text(L['hira']['rank'] * 0.86, dL, NAME['hira'], color=COL['hira'], ha='right', va='center',
             bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'), **lab)
    axt.text(L['mora']['rank'] * 0.84, ceil + 0.0006, NAME['mora'], color=COL['mora'], ha='right', va='bottom', **lab)
    axt.text(L['c3a']['rank'] * 1.2, ceil + 0.0006, NAME['c3a'], color=COL['c3a'], ha='left', va='bottom', **lab)
    F = L['fourier']
    axt.text(F['rank'] * 0.84, y['fourier'], f'FourierFT  {F["d"]:,}', color=COL['fourier'], ha='right', va='center', **lab)
    K = L['krona']
    m1, n1, m2, n2 = K['shape']
    axt.text(K['rank'] * 0.84, y['krona'] + 0.0004, f'KronA  {K["d"]:,}', color=COL['krona'], ha='right', va='bottom', **lab)
    axt.text(K['rank'] * 0.84, y['krona'] - 0.0004, f'|$M$| = {K["P"]:,}  ({m1}×{n1} $\\otimes$ {m2}×{n2})',
             color=C['ink2'], ha='right', va='top', fontsize=7, zorder=6)

    # LoHa: same maximum rank as GraLoRA k=2, at least 8,159 fewer dimensions; drawn across the break
    E = L['loha']
    gapd = L['lora']['d'] - E['d']
    axb.annotate('', xy=(E['rank'], bot_hi), xytext=(E['rank'], y['loha'] + 0.0009),
                 arrowprops=dict(arrowstyle='-', color=C['seal'], lw=0.8, ls=(0, (1, 1.6))), annotation_clip=False)
    axt.annotate('', xy=(E['rank'], dL - 0.0009), xytext=(E['rank'], top_lo),
                 arrowprops=dict(arrowstyle='-', color=C['seal'], lw=0.8, ls=(0, (1, 1.6))), annotation_clip=False)
    axb.text(E['rank'] * 1.12, y['loha'], f'{NAME["loha"]}  ≤ {E["d"]:,}', color=COL['loha'], ha='left', va='center', **lab)
    axb.text(E['rank'] * 1.12, y['loha'] + 0.0011, f'at least {gapd:,} fewer dimensions than LoRA₈, at rank {E["rank"]}',
             color=COL['loha'], ha='left', va='bottom', fontsize=7)

    # one volume, many ranks
    ya = dL - 0.0056
    x0, x1 = 92, 2600
    axt.annotate('', xy=(x1, ya), xytext=(x0, ya),
                 arrowprops=dict(arrowstyle='-|>', color=C['ink3'], lw=0.7, mutation_scale=7))
    axt.text(math.sqrt(x0 * x1), ya - 0.0007, f'same $d(M)$, maximum rank 8 → {min(m, n):,}', fontsize=7,
             color=C['ink2'], ha='center', va='top')

    # key: open marker = rank bound
    h = [Line2D([], [], ls='none', marker='o', ms=5.2, mfc='white', mec=C['ink2'], mew=1.2)]
    axb.legend(h, ['open marker: maximum rank is an upper bound (Thm II.10)'], loc='lower right', fontsize=7,
               handletextpad=0.3, borderaxespad=0.2, bbox_to_anchor=(0.9, 0.0))
    return fig


def main():
    import sys
    ok = True if '--no-check' in sys.argv else certify()
    m = n = 4096
    R, k = 8, 2
    Cs = census(m, n, R, k)
    L = Cs['L']
    print(f'-- census at {m} x {n}, budget |M| = R(m+n) = {Cs["P"]:,} (R = {R}), mn = {Cs["mn"]:,} --')
    for key, E in L.items():
        extra = ''
        if key == 'krona':
            extra = f'  shape {E["shape"]}, budget {E["P"]:,} (= |M| + {E["P"] - Cs["P"]})'
        if key == 'fourier':
            extra = f'  {E["pairs"]} conjugate pairs, {E["selfc"]} self-conjugate ({E["src"]}, seed {SEED_FT})'
        if key == 'mora':
            extra = f'  r_hat = {E["rhat"]}'
        if key == 'c3a':
            extra = f'  b = {E["b"]}'
        print(f'  {key:8s} d {"<=" if E.get("dle") else "= "} {E["d"]:7,}  d/mn = {E["d"] / Cs["mn"]:.6e}  '
              f'max rank {"<=" if E["bound"] else "= "} {E["rank"]:5d}  |M| - d = {E["P"] - E["d"]:,}{extra}')
    print(f'  LoHa gap to LoRA_8: {L["lora"]["d"] - L["loha"]["d"]:,} = (m+n-1) - 2r^2 = {m + n - 1 - 2 * 16}')
    apply_style()
    fig = draw(Cs)
    print('wrote', save(fig, 'rankvol'))
    if not ok:
        raise SystemExit('census formulas not confirmed on the small copy')


if __name__ == '__main__':
    main()
