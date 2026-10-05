"""Figure "timeline" (Expose VIII): the catalogue in time.

One dot per catalogued method of site/data/atlas-data.js, at the month of its paper's first arXiv submission,
t = Y + (M - 1/2)/12 with year Y and month M read from the YYMM prefix of the arXiv identifier, or at mid-year,
t = Y + 1/2, when the entry has no identifier. A port of the horizontal layout of site/js/fig-timeline.js:

  * deterministic centred beeswarm (dodge): points in time order (ties by arXiv sequence number, then kind, then id)
    each take the offset of least |offset| that keeps them clear of every dot already placed; ties between +q and -q
    alternate with the processing index, so the swarm stays centred;
  * colour = modification kind (style.KIND, the site's legend order), ring = ships in Hugging Face PEFT;
  * entries before 2017 sit in a compressed band left of an axis break;
  * the landmark methods of the site, plus the newest entry, are labelled; the number under each year counts its
    entries; a dashed line marks the catalogue's build date (meta.built).

Run from paper/figures/src:  python3 fig_timeline.py
"""
import json
import os
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe

from style import apply_style, C, KIND, W1, save

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(HERE, '..', '..', '..', 'site', 'data', 'atlas-data.js')
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
KIND_NAME = {'additive-weight': 'Additive', 'multiplicative-weight': 'Multiplicative',
             'architectural-insertion': 'Architectural', 'input-or-activation-augmentation': 'Learned state',
             'selective': 'Selective', 'optimizer-or-training-procedure': 'Optimizer-side',
             'composition-or-routing': 'Composition', 'hybrid': 'Hybrid'}
KORDER = list(KIND)
T0 = 2017                      # the axis proper starts here
LANDMARKS = ['houlsby-adapter', 'prefix-tuning', 'lora', 'ia3', 'qlora', 'dora', 'galore']

# geometry, in points (the plot is drawn in a point-for-point data frame)
PAD = 0.02                     # savefig.pad_inches of the style: the saved PDF is exactly W1 wide
WPT = (W1 - 2 * PAD) * 72      # plot width in points
R_DOT = 1.6                    # dot radius
RING = 0.9                     # ring radius above the dot radius, for HF PEFT entries
GAP = 0.4                      # air between neighbouring dots
GUT = 22.0                     # width of the pre-2017 band
X0, X1 = GUT + 12.0, WPT - 3.0


def load():
    txt = open(ATLAS, encoding='utf-8').read()
    return json.loads(txt[txt.index('{'):txt.rstrip().rstrip(';').rindex('}') + 1])


def parse_arxiv(raw):
    s = str(raw or '').strip()
    s = re.sub(r'^https?://(www\.)?arxiv\.org/(abs|pdf)/', '', s, flags=re.I)
    s = re.sub(r'\.pdf$', '', s, flags=re.I)
    s = re.sub(r'^arxiv:\s*', '', s, flags=re.I)
    m = re.match(r'^(\d{2})(\d{2})\.(\d{4,5})(v\d+)?$', s)
    if m:
        yy, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and yy >= 7:
            return 2000 + yy, mo, int(m.group(3))
    m = re.match(r'^[a-z][a-z\-]*(?:\.[a-z\-]+)?/(\d{2})(\d{2})(\d{3})(v\d+)?$', s, flags=re.I)
    if m:
        y2, m2 = int(m.group(1)), int(m.group(2))
        if 1 <= m2 <= 12:
            return (1900 if y2 >= 91 else 2000) + y2, m2, int(m.group(3))
    return None


def items_from(data):
    items, undated = [], []
    for m in data['methods']:
        ax = parse_arxiv(m.get('arxiv'))
        if ax:
            y, mo, seq = ax
            it = dict(t=y + (mo - 0.5) / 12, year=y, month=mo, exact=True, seq=seq)
        elif str(m.get('year', '')).isdigit() and int(m['year']) >= 1950:
            it = dict(t=int(m['year']) + 0.5, year=int(m['year']), month=None, exact=False, seq=10 ** 9)
        else:
            undated.append(m)
            continue
        kind = m.get('modification_kind') or 'hybrid'
        it.update(id=m['id'], name=m['name'], kind=kind, ki=KORDER.index(kind) if kind in KORDER else 99,
                  hf=bool(m.get('hf_peft')))
        items.append(it)
    # within a month, spread the dots evenly in the order of their arXiv numbers (each stays inside its month)
    groups = {}
    for it in items:
        if it['exact']:
            groups.setdefault((it['year'], it['month']), []).append(it)
    for (y, mo), g in groups.items():
        g.sort(key=lambda it: (it['seq'], it['ki'], it['id']))
        for k, it in enumerate(g):
            it['t'] = y + (mo - 1 + (k + 0.5) / len(g)) / 12
    items.sort(key=lambda it: (it['t'], it['seq'], it['ki'], it['id']))
    return items, undated


def dodge(items, pos, rad, gap):
    """Centred beeswarm: the site's deterministic dodge layout."""
    placed, out = [], {}
    for i, it in enumerate(items):
        p, R = pos(it), rad(it)
        ivs, cands = [], [0.0]
        for (po, qo, Ro) in placed:
            dp, Dm = abs(p - po), R + Ro + gap
            if dp < Dm:
                hh = np.sqrt(Dm * Dm - dp * dp)
                ivs.append((qo - hh, qo + hh))
                cands += [qo - hh, qo + hh]
        pref = 1 if i % 2 else -1
        cands.sort(key=lambda a: (round(abs(a), 9), -a * pref))
        q = 0.0
        for y in cands:
            if all(not (lo + 1e-7 < y < hi - 1e-7) for lo, hi in ivs):
                q = y
                break
        placed.append((p, q, R))
        out[it['id']] = q
    return out


def darker(c, f=0.72):
    r, g, b = to_rgb(c)
    return (r * f, g * f, b * f)


_MFIG = None


def text_w(txt, fs, weight='normal'):
    """Width in points of a string set in the figure face (measured with the Agg renderer at 72 dpi)."""
    global _MFIG
    if _MFIG is None:
        _MFIG = plt.figure(figsize=(4, 1), dpi=72)
    t = _MFIG.text(0, 0, txt, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(_MFIG.canvas.get_renderer()).width
    t.remove()
    return w


def clash(c, o):
    """Two label boxes (or a label box and an obstacle) overlap, or a leader line crosses the other box."""
    if not (c['b1'] + 4 < o['b0'] or c['b0'] - 4 > o['b1']) and not (c['y1'] + 1.5 < o['y0'] or c['y0'] - 1.5 > o['y1']):
        return True
    if 'x' in c and o['b0'] - 2 < c['x'] < o['b1'] + 2 and c['la'] < o['y1'] + 1 and c['lb'] > o['y0'] - 1:
        return True
    if 'x' in o and c['b0'] - 2 < o['x'] < c['b1'] + 2 and o['la'] < c['y1'] + 1 and o['lb'] > c['y0'] - 1:
        return True
    return False


def main():
    apply_style()
    data = load()
    items, undated = items_from(data)
    by_id = {it['id']: it for it in items}
    n = len(items)
    assert n + len(undated) == len(data['methods'])
    T1 = max(T0 + 1, int(np.floor(items[-1]['t'])) + 1)
    early = [it for it in items if it['t'] < T0]
    years = Counter('early' if it['t'] < T0 else it['year'] for it in items)
    kinds = Counter(it['kind'] for it in items)
    n_hf = sum(it['hf'] for it in items)
    n_inexact = sum(not it['exact'] for it in items)
    built = data['meta']['built']
    by_, bmo, bd = map(int, built.split('-'))
    dim = [31, 29 if by_ % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][bmo - 1]
    t_built = by_ + (bmo - 1 + (bd - 1) / dim) / 12

    def xmain(t):
        return X0 + (t - T0) / (T1 - T0) * (X1 - X0)

    def pos(it):
        return GUT / 2 if it['t'] < T0 else xmain(it['t'])

    def rad(it):
        return R_DOT + RING + 0.3 if it['hf'] else R_DOT + 0.15

    q = dodge(items, pos, rad, GAP)            # offsets from the spine, y growing downward
    qs = np.array([q[it['id']] for it in items])
    xs = np.array([pos(it) for it in items])
    lo_sw = min(q[it['id']] - rad(it) for it in items)
    hi_sw = max(q[it['id']] + rad(it) for it in items)

    print(f'{n} methods placed ({n_inexact} at mid-year without an arXiv identifier, {len(undated)} undated); '
          f'{len(early)} before {T0}; {n_hf} in HF PEFT; axis {T0}-{T1}')
    print('per year:', {k: years[k] for k in sorted(years, key=str)})
    print('per kind:', {KIND_NAME[k]: kinds[k] for k in KORDER})
    per_month = Counter((it['year'], it['month']) for it in items if it['exact'])
    (bym, bmm), bmn = per_month.most_common(1)[0]
    print(f'busiest month: {MONTHS[bmm - 1]} {bym} with {bmn} entries; mid-year entries:',
          sorted(Counter(it['year'] for it in items if not it['exact']).items()))
    print(f"before {T0}: {[(it['name'], it['year'], it['month']) for it in early]}; newest: {items[-1]['name']} "
          f"({MONTHS[items[-1]['month'] - 1]} {items[-1]['year']})")
    print(f'swarm extent {lo_sw:.1f} .. {hi_sw:.1f} pt about the spine; built {built} (t = {t_built:.3f})')

    # ------------------------------------------------------------------ legend (an obstacle for the labels)
    FS_L = 7.0
    entries = [(KIND_NAME[k], kinds[k], KIND[k]) for k in KORDER if kinds[k]] + [('in HF PEFT', n_hf, None)]
    rows = 5
    dy = 8.4
    colw = [max(text_w(e[0], FS_L) for e in entries[c * rows:(c + 1) * rows]) + 2 * R_DOT + 4 + 22 for c in (0, 1)]
    lx = 3.0
    ly0 = hi_sw - (rows - 1) * dy - 3.5         # q of the first row: the legend sits at the bottom left
    legend_box = dict(b0=lx - 2, b1=lx + colw[0] + colw[1] + 2, y0=ly0 - 5, y1=ly0 + (rows - 1) * dy + 5)

    # ------------------------------------------------------------------ landmark labels
    lms = [by_id[i] for i in LANDMARKS if i in by_id]
    if items[-1] not in lms:
        lms.append(items[-1])
    FS_N, FS_D, LH = 7.5, 7.0, 9.0

    def date_of(it):
        return f"{MONTHS[it['month'] - 1]} {it['year']}" if it['exact'] else f"mid-{it['year']}"

    dots = [(pos(it), q[it['id']], rad(it)) for it in items]

    def envelope(b0, b1):
        top, bot = 0.0, 0.0
        for (x, y, r) in dots:
            if x + r > b0 - 1.5 and x - r < b1 + 1.5:
                top, bot = min(top, y - r), max(bot, y + r)
        return top, bot

    # candidate placements: side x anchor x level; cost = leader length + 6 per dot the leader crosses
    # + 1.5 x the height added to the swarm; then branch and bound over all labels (no overlaps, no leader through
    # another label), as the site does
    def crossed(x, ya, yb, own):
        return sum(1 for (dx, dy, dr), it2 in zip(dots, items)
                   if it2 is not own and abs(dx - x) < dr + 0.3 and dy + dr > ya and dy - dr < yb)

    cands = []
    for it in lms:
        x, y0, r0 = pos(it), q[it['id']], rad(it)
        wn, wd = text_w(it['name'], FS_N, 'semibold'), text_w(date_of(it), FS_D)
        w = wn + 2.5 + wd
        opts = []
        for side in ('above', 'below'):
            for anchor in ('start', 'end'):
                b0, b1 = (x - 1.5, x + 1.5 + w) if anchor == 'start' else (x - 1.5 - w, x + 1.5)
                if b0 < 0 or b1 > WPT:
                    continue
                top, bot = envelope(b0, b1)
                base = top - 3 if side == 'above' else bot + 3
                opts.append(dict(side=side, anchor=anchor, b0=b0, b1=b1, base=base))
        cands.append(dict(it=it, x=x, y0=y0, r0=r0, wn=wn, wd=wd, opts=opts))

    def make(cd, o, lv):
        side = o['side']
        y0b, y1b = (lv - LH, lv) if side == 'above' else (lv, lv + LH)
        la, lb = (lv, cd['y0'] - cd['r0']) if side == 'above' else (cd['y0'] + cd['r0'], lv)
        cost = (lb - la) + 6 * crossed(cd['x'], la, lb, cd['it']) + (0 if o['anchor'] == 'start' else 1)
        return dict(it=cd['it'], side=side, anchor=o['anchor'], b0=o['b0'], b1=o['b1'], y0=y0b, y1=y1b,
                    x=cd['x'], wn=cd['wn'], wd=cd['wd'], la=la, lb=lb, cost=cost)

    best, best_cost, nodes = None, [np.inf], [0]
    chosen = []

    def rec(i, cost, top, bot):
        total = cost + 1.5 * (lo_sw - top) + 1.5 * (bot - hi_sw)
        nodes[0] += 1
        if total >= best_cost[0] or nodes[0] > 400000:
            return
        if i == len(cands):
            best_cost[0] = total
            best[:] = list(chosen)
            return
        cd = cands[i]
        tries = []
        for o in cd['opts']:
            levels = [o['base']]
            for p_ in chosen + [legend_box]:
                if o['side'] == 'above' and p_['y0'] - 2 < o['base']:
                    levels.append(p_['y0'] - 2)
                if o['side'] == 'below' and p_['y1'] + 2 > o['base']:
                    levels.append(p_['y1'] + 2)
            tries += [make(cd, o, lv) for lv in levels]
        tries.sort(key=lambda c: c['cost'])
        for c in tries:
            if any(clash(c, p_) for p_ in chosen + [legend_box]):
                continue
            chosen.append(c)
            rec(i + 1, cost + c['cost'], min(top, c['y0']), max(bot, c['y1']))
            chosen.pop()

    best = []
    rec(0, 0.0, lo_sw, hi_sw)
    assert len(best) == len(cands), 'no label placement found'
    placed = best
    print(f'labels placed (cost {best_cost[0]:.1f}, {nodes[0]} nodes):',
          ', '.join(f"{p['it']['name']} {p['side']}" for p in placed))

    top_ext = min([lo_sw] + [p['y0'] for p in placed])
    bot_ext = max([hi_sw, legend_box['y1']] + [p['y1'] for p in placed])

    # ------------------------------------------------------------------ figure, in a point-for-point frame (y up = -q)
    y_axis = -(bot_ext + 5.0)
    y_year = y_axis - 9.5
    y_count = y_year - 8.5
    y_min = y_count - 2.5
    y_max = -top_ext + 1.5
    HPT = y_max - y_min
    fig = plt.figure(figsize=(WPT / 72, HPT / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WPT)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()

    # zebra year bands and the pre-2017 band
    for y in range(T0, T1):
        if (y - T0) % 2:
            ax.fill_between([xmain(y), xmain(y + 1)], y_axis, y_max, color=C['paper2'], lw=0, zorder=0)
    if early:
        ax.fill_between([0, GUT], y_axis, y_max, color=C['paper2'], lw=0, zorder=0)
    # axis with month ticks, year labels and per-year counts
    ax.plot([X0, X1], [y_axis, y_axis], color=C['ink3'], lw=0.6, zorder=1, solid_capstyle='butt')
    for mi in range((T1 - T0) * 12 + 1):
        xm = xmain(T0 + mi / 12)
        yr = mi % 12 == 0
        ax.plot([xm, xm], [y_axis, y_axis - (3.0 if yr else 1.4)], color=C['ink3'] if yr else C['rule'],
                lw=0.5, zorder=1)
    for y in range(T0, T1):
        xc = (xmain(y) + xmain(y + 1)) / 2
        ax.text(xc, y_year, str(y), ha='center', va='baseline', fontsize=7.5, color=C['ink2'])
        ax.text(xc, y_count, str(years.get(y, 0)), ha='center', va='baseline', fontsize=7, color=C['ink3'])
    if early:
        ax.plot([1, GUT - 1], [y_axis, y_axis], color=C['ink3'], lw=0.6, zorder=1, solid_capstyle='butt')
        bx = (GUT + X0) / 2
        for dx in (-1.5, 1.5):
            ax.plot([bx + dx - 1.3, bx + dx + 1.3], [y_axis - 2.3, y_axis + 2.3], color=C['ink3'], lw=0.6)
        ey = sorted({it['year'] for it in early})
        lab = str(ey[0]) if len(ey) == 1 else f'{ey[0]}\u2013{str(T0 - 1)[2:]}'
        ax.text(GUT / 2, y_year, lab, ha='center', va='baseline', fontsize=7.5, color=C['ink2'])
        ax.text(GUT / 2, y_count, str(years['early']), ha='center', va='baseline', fontsize=7, color=C['ink3'])

    # build date: a dotted line with its label set along it, right of the newest dots
    if T0 <= t_built <= T1:
        xb = xmain(t_built)
        ax.plot([xb, xb], [y_axis, y_max - 2], color=C['ink3'], lw=0.5, ls=(0, (1.2, 1.8)), zorder=1)
        ax.text(xb + 1.5, y_axis + 2, f'built {bd} {MONTHS[bmo - 1]} {by_}', rotation=90, ha='left', va='bottom',
                fontsize=7, color=C['ink2'], zorder=2)

    # dots and rings
    cols = [KIND.get(it['kind'], KIND['hybrid']) for it in items]
    ax.scatter(xs, -qs, s=(2 * R_DOT) ** 2, c=cols, edgecolors=[darker(c) for c in cols], linewidths=0.3,
               zorder=3)
    hf = np.array([it['hf'] for it in items])
    ax.scatter(xs[hf], -qs[hf], s=(2 * (R_DOT + RING)) ** 2, facecolors='none', edgecolors=C['ink'],
               linewidths=0.5, zorder=4)

    # landmark labels with leaders: name (semibold) then month of first submission
    for p in placed:
        it = p['it']
        # leaders run over the dots (with a thin paper halo) so that each ends visibly on its own dot
        ax.plot([p['x'], p['x']], [-p['la'], -p['lb']], color=C['ink2'], lw=0.5, zorder=4.5, solid_capstyle='butt',
                path_effects=[pe.withStroke(linewidth=1.3, foreground='white')])
        ax.scatter([p['x']], [-q[it['id']]], s=(2 * R_DOT) ** 2, c=[KIND.get(it['kind'], KIND['hybrid'])],
                   edgecolors=[C['ink']], linewidths=0.6, zorder=4.6)
        yb = -p['y1'] + 2.0
        xn = p['x'] - 1.0 if p['anchor'] == 'start' else p['x'] + 1.0 - p['wn'] - 2.5 - p['wd']
        ax.text(xn, yb, it['name'], ha='left', va='baseline', fontsize=FS_N, fontweight='semibold', color=C['ink'],
                zorder=5)
        ax.text(xn + p['wn'] + 2.5, yb, date_of(it), ha='left', va='baseline', fontsize=FS_D, color=C['ink2'],
                zorder=5)

    # legend
    for i, (name, cnt, col) in enumerate(entries):
        c_, r_ = divmod(i, rows)
        cx = lx + sum(colw[:c_])
        cy = -(ly0 + r_ * dy)
        if col is None:
            ax.scatter([cx + R_DOT + 0.6], [cy], s=(2 * R_DOT) ** 2, c=[C['paper2']], edgecolors=[C['ink3']],
                       linewidths=0.3, zorder=5)
            ax.scatter([cx + R_DOT + 0.6], [cy], s=(2 * (R_DOT + RING)) ** 2, facecolors='none',
                       edgecolors=C['ink'], linewidths=0.5, zorder=5)
        else:
            ax.scatter([cx + R_DOT + 0.6], [cy], s=(2 * R_DOT) ** 2, c=[col], edgecolors=[darker(col)],
                       linewidths=0.3, zorder=5)
        ax.text(cx + 2 * R_DOT + 4, cy, name, ha='left', va='center', fontsize=FS_L, color=C['ink'])
        ax.text(cx + colw[c_] - 7, cy, str(cnt), ha='right', va='center', fontsize=FS_L, color=C['ink3'])

    print('wrote', save(fig, 'timeline'))
    return dict(n=n, years=years, kinds=kinds, n_hf=n_hf, placed=placed)


if __name__ == '__main__':
    main()
