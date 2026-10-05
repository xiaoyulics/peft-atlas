"""Figure `graph`: the category of core methods as a layered Hasse diagram (Expose VIII.C).

Objects are the core methods of site/data/atlas-data.js; arrows are every A2-A5 arrow recorded in
ATLAS_DATA.edges and method.arrows between two core methods (deduplicated by source, target, rule and h),
exactly as the site's figure module site/js/fig-graph.js builds them. Frozen (initial) sits at the bottom,
Full FT (terminal) at the top; the A1 arrows are drawn as the interval frame.

Layout is a deterministic Sugiyama pipeline ported from fig-graph.js and tuned for print:
  1. strongly connected components (mutually simulating families) are collapsed to one rung;
  2. longest-path layering of the quotient DAG, then non-sinks promoted to sit just below their lowest
     successor;
  3. the leaf feeders of a hub (one quotient arrow, into the hub, nothing below them) are packed as a comb;
  4. dummy vertices on long arrows, barycentric ordering sweeps with adjacent swaps per weakly connected
     component, keeping the order with fewest crossings;
  5. levels wider than the text width are split into sub-rows (still one Hasse level) and ordered again;
  6. x by iterated neighbour barycentres under order-preserving separation constraints.
Nothing is random. Every count printed in the figure is computed here.

Run from paper/figures/src:  python3 fig_graph.py   ->  paper/figures/graph.pdf
"""
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb

from style import apply_style, C, KIND, W1, save

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(ROOT, 'site', 'data', 'atlas-data.js')

RULE_ORDER = ['A2', 'A3', 'A4', 'A5']
RULES = {
    'A2': dict(name='freezing (sub-object)', col=C['tide'], dash=None, hook=True),
    'A3': dict(name='factorisation through $h$', col=C['ink3'], dash=None, hook=False),
    'A4': dict(name='projection of a product', col=C['moss'], dash=(3.0, 1.6), hook=False),
    'A5': dict(name='summand or stage', col=C['ochre'], dash=(0.9, 1.3), hook=False),
}
KIND_ORDER = list(KIND.keys())
KIND_LABEL = {   # the companion site's kind names (site/js/core.js, Atlas.KINDS), as in the timeline figure
    'additive-weight': 'additive', 'multiplicative-weight': 'multiplicative',
    'architectural-insertion': 'architectural', 'input-or-activation-augmentation': 'learned state',
    'selective': 'selective', 'optimizer-or-training-procedure': 'optimizer-side',
    'composition-or-routing': 'composition', 'hybrid': 'hybrid',
}
SHORT = {
    'houlsby-adapter': 'Houlsby', 'pfeiffer-adapter': 'Pfeiffer', 'miss': 'MiSS', 'super-tuning': 'Super-Tuning',
    'intruder-dimension-reduction': 'Intruder-dim.', 'polytropon': 'Poly', 'lora-soups': 'LoRA Soups',
    'riemannian-preconditioned-lora': 'Riemannian LoRA', 'text-to-lora': 'Text-to-LoRA',
    'parallel-adapter': 'Parallel Adapter',
}
HUB = 4        # in-degree from which incoming arrows are bundled into one trunk
COMB_MIN = 6   # leaf feeders from which a hub's leaves are drawn as a comb
SWEEPS = 24    # barycentric sweeps per ordering pass
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]   # white halo under labels that sit on lines


def cmp_key(s):
    return s.lower()


def load_data():
    s = open(DATA, encoding='utf-8').read()
    s = s[s.index('{'):s.rstrip().rstrip(';').rindex('}') + 1]
    return json.loads(s)


def short_name(m):
    if m['id'] in SHORT:
        return SHORT[m['id']]
    return re.sub(r'\s*\((?!IA\))[^)]*\)\s*$', '', m.get('name') or m['id'])


# ================================================================ model
def build_model(D):
    ms = [m for m in D['methods'] if m.get('core')]
    by = {m['id']: m for m in ms}
    order = {m['id']: i for i, m in enumerate(ms)}
    seen, arrows = set(), []

    def add(s, a):
        if not a or s not in by or a.get('target') not in by or a.get('rule') not in RULES:
            return
        key = (s, a['target'], a['rule'], a.get('h') or '')
        if key in seen:
            return
        seen.add(key)
        arrows.append(dict(s=s, t=a['target'], rule=a['rule'], h=a.get('h') or ''))

    for e in D.get('edges') or []:
        add(e.get('source'), e)
    for m in ms:
        for a in m.get('arrows') or []:
            add(m['id'], a)
    arrows.sort(key=lambda a: (order[a['s']], order[a['t']], a['rule'], a['h']))
    for i, a in enumerate(arrows):
        a['i'] = i
        a['self'] = a['s'] == a['t']
        a['inGroup'] = False
    out = {m['id']: [] for m in ms}
    inn = {m['id']: [] for m in ms}
    for a in arrows:
        if not a['self']:
            out[a['s']].append(a)
            inn[a['t']].append(a)

    # Tarjan SCC (iterative)
    index, idx, low, onst, stack, comps = [0], {}, {}, set(), [], []

    def strong(v0):
        work = [(v0, 0)]
        idx[v0] = low[v0] = index[0]; index[0] += 1; stack.append(v0); onst.add(v0)
        while work:
            v, k = work[-1]
            if k < len(out[v]):
                work[-1] = (v, k + 1)
                w = out[v][k]['t']
                if w not in idx:
                    idx[w] = low[w] = index[0]; index[0] += 1; stack.append(w); onst.add(w)
                    work.append((w, 0))
                elif w in onst:
                    low[v] = min(low[v], idx[w])
            else:
                work.pop()
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
                if low[v] == idx[v]:
                    c = []
                    while True:
                        w = stack.pop(); onst.discard(w); c.append(w)
                        if w == v:
                            break
                    comps.append(c)

    for m in ms:
        if m['id'] not in idx:
            strong(m['id'])

    nodes = {}
    for m in ms:
        code = (m.get('coords') or {}).get('kind') or 'R'
        nodes[m['id']] = dict(id=m['id'], label=short_name(m), kind=m.get('modification_kind') or 'hybrid',
                              code=code, ext=code != 'R', indeg=len(inn[m['id']]),
                              selfs=[a for a in arrows if a['self'] and a['s'] == m['id']])

    groups, gof = [], {}
    for c in comps:
        c.sort(key=lambda i: order[i])
        g = dict(members=list(c), size=len(c))
        if len(c) > 2:
            deg = {i: 0 for i in c}
            for a in arrows:
                if not a['self'] and a['s'] in deg and a['t'] in deg:
                    deg[a['s']] += 1; deg[a['t']] += 1
            hub = sorted(c, key=lambda i: (-deg[i], -nodes[i]['indeg'], order[i]))[0]
            rest = sorted([i for i in c if i != hub], key=lambda i: cmp_key(nodes[i]['label']))
            left = [x for k, x in enumerate(rest) if k % 2 == 0]
            right = [x for k, x in enumerate(rest) if k % 2 == 1]
            g['members'] = left[::-1] + [hub] + right
        elif len(c) == 2:
            g['members'].sort(key=lambda i: cmp_key(nodes[i]['label']))
        groups.append(g)
    groups.sort(key=lambda g: min(order[i] for i in g['members']))
    for i, g in enumerate(groups):
        g['i'] = i
        g['out'], g['inn'] = [], []
        for mid in g['members']:
            gof[mid] = g

    ce_map, ces = {}, []
    for a in arrows:
        if a['self']:
            continue
        ga, gb = gof[a['s']], gof[a['t']]
        if ga is gb:
            a['inGroup'] = True
            continue
        k = (ga['i'], gb['i'])
        if k not in ce_map:
            ce = dict(a=ga, b=gb, arrows=[], i=len(ces))
            ce_map[k] = ce; ces.append(ce); ga['out'].append(ce); gb['inn'].append(ce)
        ce_map[k]['arrows'].append(a)
        a['ce'] = ce_map[k]
    for g in groups:
        g['isolated'] = g['size'] == 1 and not g['out'] and not g['inn']

    # topological order (Kahn, smallest index first)
    indeg = {g['i']: len(g['inn']) for g in groups}
    queue = [g for g in groups if not indeg[g['i']]]
    topo = []
    while queue:
        queue.sort(key=lambda g: g['i'])
        g0 = queue.pop(0)
        topo.append(g0)
        for ce in g0['out']:
            indeg[ce['b']['i']] -= 1
            if not indeg[ce['b']['i']]:
                queue.append(ce['b'])
    assert len(topo) == len(groups), 'quotient is not a DAG'
    # longest-path layering, then promotion of non-sinks under their lowest successor
    for g in topo:
        g['rank'] = max([ce['a']['rank'] + 1 for ce in g['inn']] or [0])
    for g in reversed(topo):
        if g['out']:
            g['rank'] = min(ce['b']['rank'] for ce in g['out']) - 1
    max_rank = max([g['rank'] for g in groups if not g['isolated']] or [0])

    # composites: a quotient arrow a->b is composite if b is reachable from a through another successor
    for ce in ces:
        st = [o['b'] for o in ce['a']['out'] if o['b'] is not ce['b']]
        seen_g, found = set(), False
        while st and not found:
            x = st.pop()
            if x is ce['b']:
                found = True
                break
            if x['i'] in seen_g:
                continue
            seen_g.add(x['i'])
            st.extend(o['b'] for o in x['out'])
        ce['composite'] = found
        for a in ce['arrows']:
            a['composite'] = found

    stats = dict(objects=len(ms), arrows=0, byRule={r: 0 for r in RULE_ORDER}, selfs=0, composite=0,
                 cycles=0, cycMembers=0, isolated=0, R=sum(1 for m in ms if nodes[m['id']]['code'] == 'R'))
    for a in arrows:
        if a['self']:
            stats['selfs'] += 1
            continue
        stats['arrows'] += 1
        stats['byRule'][a['rule']] += 1
        if a.get('composite'):
            stats['composite'] += 1
    for g in groups:
        if g['size'] > 1:
            stats['cycles'] += 1; stats['cycMembers'] += g['size']
        if g['isolated']:
            stats['isolated'] += 1
    return dict(methods=ms, nodes=nodes, arrows=arrows, out=out, inn=inn, groups=groups, gof=gof, ces=ces,
                topo=topo, maxRank=max_rank, stats=stats, by=by)




# ================================================================ layout (all lengths in points at print size)
FS = 7.0          # label size of an ordinary object
PADX = 2.3        # pill horizontal padding
PILL_PAD = 3.6    # pill height = font size + PILL_PAD
GAP = 3.4         # gap between neighbouring objects in a row
GAP_IN = 11.0     # gap between members of one mutually simulating class (room for the double arrow)
DGAP = 1.6        # separation between two dummy vertices
DSEP = 2.6        # extra clearance between a dummy and an object
SUB_GAP = 5.8     # between sub-rows of one level
RANK_GAP = 12.0   # between levels
MARGIN = 10.0     # left/right margin (the A1 frame runs inside it)
END_FS = 8.5      # Frozen / Full FT
END_H = 13.0
END_SUB = 8.0     # the line under Full FT / over Frozen
COMB_JOIN = 9.6   # arrows through a comb join its spine this far below it (under the comb's caption)


class Item:
    """A vertex of the layered graph: a group ('g'), a comb of leaf feeders ('c') or a dummy ('d')."""
    __slots__ = ('t', 'id', 'w', 'h', 'ups', 'dns', 'L', 'pos', 'x', 'y', 'key', 'g', 'offs', 'ce', 'lane', 'bc',
                 'hub', 'leaves', 'rank', 'cells', 'rails', 'rowH', 'comb', 'row', 'shelf')

    def __init__(self, t, id_, w=0.0, h=0.0, **kw):
        self.t, self.id, self.w, self.h = t, id_, w, h
        self.ups, self.dns = [], []
        self.L = self.pos = 0
        self.x = self.y = 0.0
        self.key = 0.0
        self.g = self.offs = self.ce = self.hub = self.leaves = self.cells = self.rails = self.comb = None
        self.lane = False
        self.bc = 0.0
        self.rank = self.row = 0
        self.rowH = 0.0
        self.shelf = False
        for k, v in kw.items():
            setattr(self, k, v)


def link(lo_, hi_, lo_off, hi_off):
    lo_.ups.append((hi_, lo_off, hi_off))   # (other, own offset, other's offset)
    hi_.dns.append((lo_, hi_off, lo_off))


def crossings(layers):
    total = 0
    for L in range(len(layers) - 1):
        pairs = [(it.pos, n[0].pos) for it in layers[L] for n in it.ups]
        for i in range(len(pairs)):
            a0, a1 = pairs[i]
            for j in range(i + 1, len(pairs)):
                d0, d1 = a0 - pairs[j][0], a1 - pairs[j][1]
                if (d0 < 0 < d1) or (d0 > 0 > d1):
                    total += 1
    return total


def sweep(layers, direction):
    rng = range(1, len(layers)) if direction > 0 else range(len(layers) - 2, -1, -1)
    for L in rng:
        lay = layers[L]
        out, with_nb = [None] * len(lay), []
        for it in lay:
            nb = it.dns if direction > 0 else it.ups
            if nb:
                it.bc = sum(n[0].pos for n in nb) / len(nb)
                with_nb.append(it)
            else:
                out[it.pos] = it   # no neighbour on that side: keep its slot
        with_nb.sort(key=lambda it: (it.bc, it.pos))
        j = 0
        for i in range(len(out)):
            if out[i] is None:
                out[i] = with_nb[j]; j += 1
        for k, it in enumerate(out):
            it.pos = k
        layers[L] = out


def pair_cross(u, v):
    c = 0
    for a in u.ups:
        for b in v.ups:
            if a[0].pos > b[0].pos:
                c += 1
    for a in u.dns:
        for b in v.dns:
            if a[0].pos > b[0].pos:
                c += 1
    return c


def transpose(layers):
    for _ in range(6):
        improved = False
        for lay in layers:
            for i in range(len(lay) - 1):
                u, v = lay[i], lay[i + 1]
                if pair_cross(v, u) < pair_cross(u, v):
                    lay[i], lay[i + 1] = v, u
                    v.pos, u.pos = i, i + 1
                    improved = True
        if not improved:
            break


def order_layers(layers, sweeps):
    for lay in layers:
        for i, it in enumerate(lay):
            it.pos = i
    best_c, best_o = crossings(layers), [list(l) for l in layers]
    for s in range(sweeps):
        sweep(layers, -1 if s % 2 else 1)
        transpose(layers)
        c = crossings(layers)
        if c < best_c:
            best_c, best_o = c, [list(l) for l in layers]
    for L in range(len(layers)):
        layers[L] = best_o[L]
        for i, it in enumerate(layers[L]):
            it.pos = i
    return best_c


def order_by_components(layers, sweeps):
    """Order each weakly connected component on its own, then concatenate, so unrelated families never cross."""
    par, allit = {}, []

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for lay in layers:
        for it in lay:
            par[it.id] = it.id
            allit.append(it)
    for it in allit:
        for n in it.ups:
            a, b = find(it.id), find(n[0].id)
            if a != b:
                par[a] = b
    comps, order = {}, []
    for it in allit:
        r = find(it.id)
        if r not in comps:
            comps[r] = []
            order.append(r)
        comps[r].append(it)
    order.sort(key=lambda r: (-sum(1 for it in comps[r] if it.t != 'd'), r))
    total, subs = 0, []
    for r in order:
        mine = {it.id for it in comps[r]}
        ls = [[it for it in lay if it.id in mine] for lay in layers]
        total += order_layers(ls, sweeps)
        subs.append(ls)
    for L in range(len(layers)):
        row = []
        for ls in subs:
            row += ls[L]
        for i, it in enumerate(row):
            it.pos = i
        layers[L] = row
    return total, len(order)


def make_measure():
    fig = plt.figure(figsize=(2, 2), dpi=72)
    rend = fig.canvas.get_renderer()
    cache = {}

    def measure(txt, fs, weight='regular'):
        k = (txt, fs, weight)
        if k not in cache:
            prop = FontProperties(family='IBM Plex Sans', size=fs, weight=weight)
            w, h, d = rend.get_text_width_height_descent(txt, prop, ismath=False)
            cache[k] = w   # pixels at 72 dpi = points
        return cache[k]
    return measure


def label_size(n):
    """Hubs get slightly larger labels: FS + up to 2 pt, growing with log in-degree (as on the site)."""
    k = math.log2(n['indeg'] + 1) - 2.6
    return FS + max(0.0, min(2.0, k * 0.62))


def build_comb(c, items, max_w, KI, nodes):
    """A hub's leaf feeders as one compound object: chips in rows split by a central spine; each chip ticks up
    to a rail above its row, the rails join the spine and the spine rises to the hub."""
    AISLE, CG = 9.0, GAP
    items.sort(key=lambda it: (KI.get(nodes[it.g['members'][0]]['kind'], 99),
                               cmp_key(nodes[it.g['members'][0]]['label'])))
    n = len(items)
    row_h = max(it.h for it in items)

    def arrange(per):
        rows = math.ceil(n / per)
        bal = math.ceil(n / rows)
        out, wmax = [], 0.0
        for r in range(rows):
            row = items[r * bal:(r + 1) * bal]
            nl = math.ceil(len(row) / 2)
            left, right = row[:nl], row[nl:]
            lw = sum(it.w for it in left) + CG * max(0, len(left) - 1)
            rw = sum(it.w for it in right) + CG * max(0, len(right) - 1)
            wmax = max(wmax, 2 * max(lw, rw) + AISLE)
            out.append((left, right))
        return out, wmax

    best = arrange(2)
    per = 4
    while per <= min(n, 16):
        a = arrange(per)
        if a[1] <= max_w:
            best = a
        else:
            break
        per += 2
    rows, w = best
    PITCH, TOP = row_h + 5.4, 3.6
    c.w = w
    c.h = TOP + len(rows) * PITCH - 1.8
    c.rowH = row_h
    c.cells, c.rails = [], []
    for r, (left, right) in enumerate(rows):
        cy = -c.h / 2 + TOP + r * PITCH + row_h / 2
        x = -AISLE / 2
        for it in reversed(left):
            c.cells.append((it, x - it.w / 2, cy, r)); x -= it.w + CG
        xl = x + CG
        x = AISLE / 2
        for it in right:
            c.cells.append((it, x + it.w / 2, cy, r)); x += it.w + CG
        c.rails.append((cy - row_h / 2 - 2.4, min(xl, -AISLE / 2), max(x - CG, AISLE / 2)))


def layout(model, W, measure, reserve=(0.0, 0.0), legend_h=0.0):
    """reserve = widths kept free in the upper left and right corners for the legends, legend_h their height."""
    nodes, lo, hi = model['nodes'], MARGIN, W - MARGIN
    avail = hi - lo
    KI = {k: i for i, k in enumerate(KIND_ORDER)}
    pill = {}
    for m in model['methods']:
        n = nodes[m['id']]
        fs = label_size(n)
        pill[m['id']] = dict(fs=fs, w=math.ceil(measure(n['label'], fs) + 2 * PADX), h=round(fs + PILL_PAD, 2))
    g_item = {}
    for g in model['groups']:
        w = sum(pill[i]['w'] for i in g['members']) + GAP_IN * (len(g['members']) - 1)
        h = max(pill[i]['h'] for i in g['members'])
        x, offs = -w / 2, {}
        for i in g['members']:
            offs[i] = x + pill[i]['w'] / 2
            x += pill[i]['w'] + GAP_IN
        g_item[g['i']] = Item('g', 'g:' + '='.join(nodes[i]['label'] for i in g['members']), w, h, g=g, offs=offs)

    def ce_offsets(ce):
        a, b = g_item[ce['a']['i']], g_item[ce['b']['i']]
        so = sum(a.offs[x['s']] for x in ce['arrows']) / len(ce['arrows'])
        to = sum(b.offs[x['t']] for x in ce['arrows']) / len(ce['arrows'])
        return so, to

    # hubs: a member with HUB or more incoming arrows from outside its class
    hub_g, hub_off = {}, {}
    for g in model['groups']:
        best = -1
        for i in g['members']:
            ext = len([a for a in model['inn'][i] if not a['inGroup']])
            if ext >= HUB and ext > best:
                best = ext
                hub_g[g['i']] = True
                hub_off[g['i']] = g_item[g['i']].offs[i]
    # combs of leaf feeders
    comb_of, combs = {}, []
    for b in model['groups']:
        if not hub_g.get(b['i']):
            continue
        leaves = [ce['a'] for ce in b['inn'] if ce['a']['size'] == 1 and not ce['a']['inn'] and len(ce['a']['out']) == 1]
        if len(leaves) < COMB_MIN:
            continue
        c = Item('c', 'c:%d' % b['i'], hub=b, leaves=leaves, rank=b['rank'] - 1)
        build_comb(c, [g_item[g['i']] for g in leaves], min(avail * 0.80, 700), KI, nodes)
        combs.append(c)
        for g in leaves:
            comb_of[g['i']] = c
    connected = [g for g in model['groups'] if not g['isolated'] and g['i'] not in comb_of]
    R = model['maxRank'] + 1

    def items_of_rank():
        by = [[] for _ in range(R)]
        for g in connected:
            by[g['rank']].append(g_item[g['i']])
        for c in combs:
            by[c.rank].append(c)
        return by

    def item_of(g):
        return comb_of.get(g['i']) or g_item[g['i']]

    # ---- coarse ordering on levels, with dummies
    coarse = items_of_rank()
    for r, lay in enumerate(coarse):
        for it in lay:
            it.L, it.ups, it.dns = r, [], []
    linked_c = set()
    for ce in model['ces']:
        A0, B0 = item_of(ce['a']), item_of(ce['b'])
        off = ce_offsets(ce)
        if A0.t == 'c':
            kk = A0.id + '>' + B0.id
            if kk in linked_c:
                continue
            linked_c.add(kk)
            link(A0, B0, 0.0, hub_off[ce['b']['i']])
            continue
        prev, prev_off = A0, off[0]
        for L in range(ce['a']['rank'] + 1, ce['b']['rank']):
            d = Item('d', 'd%d_%d' % (ce['i'], L), ce=ce, L=L)
            coarse[L].append(d)
            link(prev, d, prev_off, 0.0)
            prev, prev_off = d, 0.0
        link(prev, B0, prev_off, off[1])
    # initial order: depth-first discovery from the busiest objects, top level first
    starts = [it for lay in coarse for it in lay if it.t != 'd']
    starts.sort(key=lambda it: (-it.L, -(len(it.ups) + len(it.dns)), it.id))
    seen, init = set(), [[] for _ in coarse]
    for s0 in starts:
        st = [s0]
        while st:
            it = st.pop()
            if it.id in seen:
                continue
            seen.add(it.id)
            init[it.L].append(it)
            nb = sorted([n[0] for n in it.ups + it.dns], key=lambda x: x.id, reverse=True)
            for x in nb:
                if x.id not in seen:
                    st.append(x)
    coarse = init
    order_by_components(coarse, SWEEPS)
    key_of = {}
    for lay in coarse:
        for i, it in enumerate(lay):
            key_of[it.id] = i / (len(lay) - 1) if len(lay) > 1 else 0.5
    coarse_dns = {it.id: len(it.dns) for lay in coarse for it in lay}

    # ---- refinement: levels wider than the text are split into sub-rows (one Hasse level each)
    def sep(a, b):
        if a.t == 'd' and b.t == 'd':
            return DGAP
        if a.t == 'd' or b.t == 'd':
            return (a.w + b.w) / 2 + DSEP
        return (a.w + b.w) / 2 + GAP

    def row_width(lay):
        tot = 0.0
        for i, it in enumerate(lay):
            tot += it.w + (sep(lay[i - 1], it) - (lay[i - 1].w + it.w) / 2 if i else 0.0)
        return tot

    def resolve(lay, des, band=None):
        """Place a row as close to the desired centres as the order and separations allow. `band` maps an item to
        its admissible interval (the top levels keep clear of the legends in the upper corners)."""
        n = len(lay)
        if not n:
            return
        lb = [(band(it) if band else (lo, hi))[0] + it.w / 2 for it in lay]
        ub = [(band(it) if band else (lo, hi))[1] - it.w / 2 for it in lay]
        Lx, Rx = [0.0] * n, [0.0] * n
        for i in range(n):
            Lx[i] = max(des[i], Lx[i - 1] + sep(lay[i - 1], lay[i]) if i else -1e9, lb[i])
        for i in range(n - 1, -1, -1):
            Rx[i] = min(des[i], Rx[i + 1] - sep(lay[i], lay[i + 1]) if i < n - 1 else 1e9, ub[i])
        for i in range(n):
            lay[i].x = (Lx[i] + Rx[i]) / 2
        if row_width(lay) <= hi - lo:
            for i in range(n):
                lay[i].x = max(lay[i].x, lb[i], lay[i - 1].x + sep(lay[i - 1], lay[i]) if i else -1e9)
            for i in range(n - 1, -1, -1):
                lay[i].x = min(lay[i].x, ub[i], lay[i + 1].x - sep(lay[i], lay[i + 1]) if i < n - 1 else 1e9)
        else:
            x = (lo + hi) / 2 - row_width(lay) / 2
            for k, it in enumerate(lay):
                if k:
                    x += sep(lay[k - 1], it) - (lay[k - 1].w + it.w) / 2
                it.x = x + it.w / 2
                x += it.w

    def desired(it, mode):
        nb = it.ups if mode == 'up' else it.dns if mode == 'dn' else it.ups + it.dns
        if not nb:
            return it.x
        s = ws = 0.0
        for (o, so, oo) in nb:
            w = 2.0 if (it.t != 'g' or o.t != 'g') else 1.0
            s += w * (o.x + oo - so)
            ws += w
        return s / ws

    def refine(K):
        base, nL = [], 0
        for r in range(R):
            base.append(nL)
            nL += K[r]

        def Lidx(rk, s):
            return base[rk] + (K[rk] - 1 - s)
        rank_of = {}
        for r in range(R):
            for q in range(K[r]):
                rank_of[base[r] + q] = r
        # sub-rows: a comb takes the top sub-row alone; objects reached from below fill the lower sub-rows,
        # sources the upper ones, at most ceil(n / rows) per sub-row
        sub = {}
        for r in range(R):
            k = K[r]
            objs = [it for it in coarse[r] if it.t != 'd']
            comb = [it for it in objs if it.t == 'c']
            rest = [it for it in objs if it.t == 'g']
            for c in comb:
                sub[c.id] = 0
            s0 = 1 if comb and k > 1 else 0
            cap = math.ceil(len(rest) / max(1, k - s0))
            cnt = defaultdict(int)
            with_in = [it for it in rest if coarse_dns[it.id] > 0]
            srcs = [it for it in rest if not coarse_dns[it.id]]
            s = k - 1
            for it in with_in:
                while s > s0 and cnt[s] >= cap:
                    s -= 1
                sub[it.id] = s
                cnt[s] += 1
            for it in srcs:
                t = s0
                while t < k - 1 and cnt[t] >= cap:
                    t += 1
                sub[it.id] = t
                cnt[t] += 1
        layers = [[] for _ in range(nL)]
        for rk, lay in enumerate(items_of_rank()):
            for it in lay:
                it.ups, it.dns = [], []
                it.L = Lidx(rk, sub[it.id])
                it.key = key_of[it.id]
                layers[it.L].append(it)
        lanes, linked, dummies = {}, set(), []

        def link_once(a, b, ao, bo):
            kk = a.id + '>' + b.id
            if kk in linked:
                return
            linked.add(kk)
            link(a, b, ao, bo)
        for c in combs:
            lanes['l%d_%d' % (c.hub['i'], c.L)] = c
        for ce in model['ces']:
            a, b = item_of(ce['a']), item_of(ce['b'])
            off = ce_offsets(ce)
            ce['dum'] = []
            if a.t == 'c':
                if b.L - a.L == 1:
                    link_once(a, b, 0.0, hub_off[ce['b']['i']])
                continue
            if b.L - a.L <= 1:
                link(a, b, off[0], off[1])
                continue
            if hub_g.get(ce['b']['i']):
                for Lc in range(a.L + 1, b.L):
                    lk = 'l%d_%d' % (ce['b']['i'], Lc)
                    if lk not in lanes:
                        lanes[lk] = Item('d', lk, lane=True, L=Lc, key=b.key)
                        layers[Lc].append(lanes[lk])
                        dummies.append(lanes[lk])
                    ce['dum'].append(lanes[lk])
                link(a, ce['dum'][0], off[0], 0.0)
                for m in range(len(ce['dum']) - 1):
                    link_once(ce['dum'][m], ce['dum'][m + 1], 0.0, 0.0)
                link_once(ce['dum'][-1], b, 0.0, hub_off[ce['b']['i']])
                continue
            prev, prev_off = a, off[0]
            for Lc in range(a.L + 1, b.L):
                f = (Lc - a.L) / (b.L - a.L)
                d = Item('d', 'd%d_%d' % (ce['i'], Lc), ce=ce, L=Lc, key=a.key + f * (b.key - a.key))
                layers[Lc].append(d)
                dummies.append(d)
                ce['dum'].append(d)
                link(prev, d, prev_off, 0.0)
                prev, prev_off = d, 0.0
            link(prev, b, prev_off, off[1])
        for lay in layers:
            lay.sort(key=lambda it: (it.key, it.id))
        n_cross, n_comp = order_by_components(layers, SWEEPS)
        # x: iterated barycentres under order-preserving separation
        for lay in layers:
            x = (lo + hi) / 2 - row_width(lay) / 2
            for i, it in enumerate(lay):
                if i:
                    x += sep(lay[i - 1], it) - (lay[i - 1].w + it.w) / 2
                it.x = x + it.w / 2
                x += it.w
        top = len(layers) - 1

        def band_for(L):
            # the two top levels, and the dummies on the level below them, stay between the corner legends
            r = rank_of[L]
            if r >= R - 2:
                return lambda it: (lo + reserve[0], hi - reserve[1])
            if r == R - 3:
                return lambda it: (lo + reserve[0], hi - reserve[1]) if it.t == 'd' else (lo, hi)
            return None
        for p in range(12):
            if p % 2 == 0:
                for L1 in range(top - 1, -1, -1):
                    resolve(layers[L1], [desired(it, 'up') for it in layers[L1]], band_for(L1))
            else:
                for L1 in range(1, top + 1):
                    resolve(layers[L1], [desired(it, 'dn') for it in layers[L1]], band_for(L1))
        for p in range(4):
            for L1 in range(top + 1):
                resolve(layers[L1], [desired(it, 'both') for it in layers[L1]], band_for(L1))
        over = {rank_of[Lq] for Lq, lay in enumerate(layers) if row_width(lay) > hi - lo + 0.5}
        return dict(layers=layers, nL=nL, Lidx=Lidx, over=over, nCross=n_cross, comps=n_comp, dummies=dummies)

    # first guess from the objects alone, then add sub-rows to any level whose rows overflow
    K = []
    for r in range(R):
        rw = 0.0
        nr = 0
        has_c = False
        for it in coarse[r]:
            if it.t == 'g':
                rw += it.w + (GAP if nr else 0)
                nr += 1
            if it.t == 'c':
                has_c = True
        k0 = 1 if nr else 0
        while nr and k0 < 10 and rw / k0 > avail * 0.84:
            k0 += 1
        K.append(max(1, k0 + (1 if has_c and nr else 0)))
    RF = None
    for _ in range(10):
        RF = refine(K)
        grew = False
        for rk in sorted(RF['over']):
            if K[rk] < 12:
                K[rk] += 1
                grew = True
        if not grew:
            break
    layers, nL, Lidx = RF['layers'], RF['nL'], RF['Lidx']
    # a comb's spine rises straight into its hub's trunk: centre it under the hub, keeping its row's order
    for c in combs:
        hx = g_item[c.hub['i']].x + hub_off[c.hub['i']]
        lay = layers[c.L]
        for _ in range(8):   # each pass splits the difference with the neighbours, which yield as far as they can
            resolve(lay, [hx if it is c else it.x for it in lay])

    # ---- y (downwards)
    layer_y = [0.0] * nL
    full_y = 2.0 + END_H / 2
    legend_top = full_y + 5.0
    legend_bot = legend_top + legend_h

    def place_rows(extra):
        # extra[r]: additional space above level r (used to lower the two top levels past the legends)
        y = full_y + END_H / 2 + END_SUB + RANK_GAP * 0.7     # room for the 'terminal' line under Full FT
        bottom = {}
        for r in range(R - 1, -1, -1):
            y += extra.get(r, 0.0)
            for s1 in range(K[r]):
                Lr = Lidx(r, s1)
                row_h = max([it.h for it in layers[Lr] if it.t != 'd'] or [8.0])
                comb = any(it.t == 'c' for it in layers[Lr])
                layer_y[Lr] = y + row_h / 2
                bottom[r] = y + row_h
                y += row_h + (SUB_GAP if s1 < K[r] - 1 else RANK_GAP) + (7.5 if comb else 0.0)
        return y, bottom
    y, bottom = place_rows({})
    # arrows from the third level into the second must not cross the legends: the second level from the top
    # ends below them, the gap spread over the two top levels
    deficit = legend_bot + 2.0 - bottom.get(R - 2, legend_bot + 2.0)
    if deficit > 0 and R >= 3:
        y, bottom = place_rows({R - 1: deficit / 2, R - 2: deficit / 2})
    # shelf of objects with no recorded arrow besides A1
    shelf = [g_item[g['i']] for g in model['groups'] if g['isolated']]
    shelf.sort(key=lambda it: (KI.get(nodes[it.g['members'][0]]['kind'], 99), cmp_key(nodes[it.g['members'][0]]['label'])))
    shelf_box = None
    if shelf:
        y += 2.0 - RANK_GAP * 0.35
        shelf_top = y
        rows, rw2 = [[]], 0.0
        for it in shelf:
            add = it.w + (GAP if rows[-1] else 0.0)
            if rw2 + add > avail - 12 and rows[-1]:
                rows.append([])
                rw2, add = 0.0, it.w
            rows[-1].append(it)
            rw2 += add
        y += 9.0
        for ri, row in enumerate(rows):
            tot = sum(it.w for it in row) + GAP * (len(row) - 1)
            rh = max(it.h for it in row)
            x0 = (lo + hi) / 2 - tot / 2
            for it in row:
                it.x, it.y, it.shelf, it.row = x0 + it.w / 2, y + rh / 2, True, ri
                x0 += it.w + GAP
            y += rh + 3.4
        shelf_box = (lo - 3, hi + 3, shelf_top, y - 2)
        y += 2.0
    y += END_SUB                                          # room for the 'initial' line over Frozen
    frozen_y = y + END_H / 2
    H = frozen_y + END_H / 2 + 2.0
    for L, lay in enumerate(layers):
        for it in lay:
            it.y = layer_y[L]

    # ---- positions of every method
    pos = {}

    def place(g, it):
        for i in g['members']:
            P = pill[i]
            pos[i] = dict(x=it.x + it.offs[i], y=it.y, w=P['w'], h=P['h'], fs=P['fs'], g=g, item=it, comb=None, row=0)
    for g in connected:
        place(g, g_item[g['i']])
    for it in shelf:
        place(it.g, it)
    for c in combs:
        for (it, dx, dy, row) in c.cells:
            it.x, it.y, it.comb, it.row = c.x + dx, c.y + dy, c, row
            place(it.g, it)
            pos[it.g['members'][0]]['comb'] = c
            pos[it.g['members'][0]]['row'] = row
    ends = dict(frozen=dict(x=(lo + hi) / 2, y=frozen_y, h=END_H, label='Frozen'),
                full=dict(x=(lo + hi) / 2, y=full_y, h=END_H, label='Full FT'))
    for e in ends.values():
        e['w'] = math.ceil(measure(e['label'], END_FS, 'medium') + 2 * PADX + 4)
    return dict(W=W, H=H, lo=lo, hi=hi, legend_top=legend_top, layers=layers, g_item=g_item, pos=pos, ends=ends, shelf=shelf,
                shelf_box=shelf_box, combs=combs, comb_of=comb_of, K=K, R=R, crossings=RF['nCross'],
                comps=RF['comps'], dummies=RF['dummies'], hub_g=hub_g, pill=pill)


# ================================================================ drawing
def mix(c, t, base='#FFFFFF'):
    a, b = to_rgb(c), to_rgb(base)
    return tuple(b[k] + t * (a[k] - b[k]) for k in range(3))


def kind_colours(kind):
    """(fill, edge) of a chip: a tint of the KIND colour and the colour itself; the Wong yellow is darkened for the
    outline only, so that it reads on white paper."""
    col = KIND.get(kind, C['gray'])
    if kind == 'composition-or-routing':
        return mix(col, 0.42), mix(col, 0.80, '#000000')
    return mix(col, 0.20), to_rgb(col)


def monotone_path(pts):
    """d3.curveMonotoneY through points with increasing y: Steffen tangents, cubic Bezier segments."""
    u = [p[1] for p in pts]
    v = [p[0] for p in pts]
    n = len(pts)
    if n == 2:
        (x0, y0), (x1, y1) = pts
        return [(x0, y0), (x0, y0 + (y1 - y0) / 3), (x1, y1 - (y1 - y0) / 3), (x1, y1)], \
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    def sgn(z):
        return (z > 0) - (z < 0)
    t = [0.0] * n
    for i in range(1, n - 1):
        h0, h1 = u[i] - u[i - 1], u[i + 1] - u[i]
        s0 = (v[i] - v[i - 1]) / h0 if h0 else 0.0
        s1 = (v[i + 1] - v[i]) / h1 if h1 else 0.0
        p = (s0 * h1 + s1 * h0) / (h0 + h1) if (h0 + h1) else 0.0
        t[i] = (sgn(s0) + sgn(s1)) * min(abs(s0), abs(s1), 0.5 * abs(p))
    h = u[1] - u[0]
    t[0] = (3 * (v[1] - v[0]) / h - t[1]) / 2 if h else t[1]
    h = u[-1] - u[-2]
    t[-1] = (3 * (v[-1] - v[-2]) / h - t[-2]) / 2 if h else t[-2]
    verts, codes = [(v[0], u[0])], [Path.MOVETO]
    for i in range(n - 1):
        du = (u[i + 1] - u[i]) / 3
        verts += [(v[i] + du * t[i], u[i] + du), (v[i + 1] - du * t[i + 1], u[i + 1] - du), (v[i + 1], u[i + 1])]
        codes += [Path.CURVE4] * 3
    return verts, codes


def chevron(ax, x, y, dx, dy, col, s=2.3, lw=0.55, z=3, alpha=1.0):
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    b = s * 0.72
    ax.plot([x - ux * s + px * b, x, x - ux * s - px * b], [y - uy * s + py * b, y, y - uy * s - py * b],
            color=col, lw=lw, solid_joinstyle='miter', solid_capstyle='butt', zorder=z, alpha=alpha)


def draw_path(ax, verts, codes, col, lw=0.55, dash=None, z=2, alpha=1.0, cap='round'):
    pp = PathPatch(Path(verts, codes), fill=False, edgecolor=col, lw=lw, zorder=z, alpha=alpha, capstyle=cap,
                   joinstyle='round')
    if dash:
        pp.set_linestyle((0, dash))
    ax.add_patch(pp)
    return pp


def rule_sample(ax, x, y, rule, w=13.0):
    """A short legend arrow for a rule, from (x, y) to (x + w, y)."""
    R0 = RULES[rule]
    draw_path(ax, [(x, y), (x + w, y)], [Path.MOVETO, Path.LINETO], R0['col'], lw=0.75, dash=R0['dash'], z=6)
    chevron(ax, x + w, y, 1, 0, R0['col'], s=2.3, lw=0.7, z=6)
    if R0['hook']:
        hook(ax, x, y, R0['col'], horizontal=True, z=6)


def hook(ax, x, y, col, horizontal=False, z=3, lw=0.55, r=1.25):
    """The tail of an inclusion arrow (A2: freezing gives a sub-object), a half circle."""
    th = [math.pi * k / 16 for k in range(17)]
    if horizontal:   # tail of a rightward arrow: half circle behind the start, curling over the shaft (as in ↪)
        xs = [x - r * math.sin(a) for a in th]
        ys = [y - r * (1 - math.cos(a)) for a in th]
    else:            # tail of an upward arrow (y down): half circle below the start, opening upwards on the left
        xs = [x - r * (1 - math.cos(a)) for a in th]
        ys = [y + r * math.sin(a) for a in th]
    ax.plot(xs, ys, color=col, lw=lw, zorder=z, solid_capstyle='round')


# ================================================================ legends (upper corners, inside the A1 frame)
LEG_FS = 7.0
LEG_LINE = 7.9
LEG_SAMPLE = 14.0   # width of a sample arrow / swatch column


def legend_rows(model):
    """Left block: what an arrow is and the rules; right block: chip colour = modification kind, dashed chip."""
    left = [('head', r'$M\to N$: $N$ simulates $M$'),
            ('A1', 'A1 interval, the frame'), ('A2', 'A2 freezing: a sub-object'), ('A3', 'A3 factorisation via $h$'),
            ('A4', 'A4 projection of a product'), ('A5', 'A5 summand or stage'),
            ('loop', 'loop: within one family'), ('pair', 'mutual simulation (one rung)')]
    counts = defaultdict(int)
    for m in model['methods']:
        counts[model['nodes'][m['id']]['kind']] += 1
    right = [('kind:' + k, KIND_LABEL[k], str(counts[k])) for k in KIND_ORDER if counts[k]]
    n_ext = sum(1 for m in model['methods'] if model['nodes'][m['id']]['ext'])
    right.append(('ext', 'not a weight-space object', str(n_ext)))
    return left, right


def legend_size(model, measure):
    left, right = legend_rows(model)
    wl = max(measure(re.sub(r'\$[^$]*\$', 'M to N', r[1]), LEG_FS) for r in left) + LEG_SAMPLE + 3
    wr = max(measure(r[1] + '  ' + r[2], LEG_FS) for r in right) + 10 + 4
    return (wl + 10, wr + 12), max(len(left), len(right)) * LEG_LINE


def draw_legends(ax, model, G):
    left, right = legend_rows(model)
    ink, ink2, ink3 = C['ink'], C['ink2'], C['ink3']
    y0 = G['legend_top'] + LEG_LINE / 2
    x = G['lo'] + 3.0
    for k, row in enumerate(left):
        y = y0 + k * LEG_LINE
        kind, text = row
        tx = x + LEG_SAMPLE + 3
        if kind == 'head':
            ax.text(x, y, text, ha='left', va='center_baseline', fontsize=LEG_FS, color=ink, zorder=7)
            continue
        if kind == 'A1':
            draw_path(ax, [(x, y), (x + LEG_SAMPLE - 1, y)], [Path.MOVETO, Path.LINETO], ink3, lw=0.5,
                      dash=(2.0, 2.2), z=7)
            chevron(ax, x + LEG_SAMPLE - 1, y, 1, 0, ink3, s=2.3, lw=0.55, z=7)
        elif kind in RULES:
            rule_sample(ax, x + 1.5, y, kind, w=LEG_SAMPLE - 2.5)
        elif kind == 'loop':
            bx, bw = x + 0.5, LEG_SAMPLE - 1.0
            ax.add_patch(FancyBboxPatch((bx, y - 0.6), bw, 4.2, boxstyle='round,pad=0,rounding_size=1.4',
                                        fc=mix(C['gray'], 0.12), ec=ink3, lw=0.5, zorder=6))
            cx, cy = bx + bw - 2.6, y - 0.6
            verts = [(cx - 2.0, cy - 0.2), (cx - 2.5, cy - 5.6), (cx + 4.2, cy - 5.2), (cx + 1.9, cy - 0.6)]
            draw_path(ax, verts, [Path.MOVETO] + [Path.CURVE4] * 3, ink2, lw=0.55, z=7)
            chevron(ax, cx + 1.9, cy - 0.6, -0.45, 1, ink2, s=1.6, lw=0.5, z=7)
        elif kind == 'pair':
            for d, yy in ((1, y - 1.5), (-1, y + 1.5)):
                xa, xb = (x + 1.5, x + LEG_SAMPLE - 1) if d > 0 else (x + LEG_SAMPLE - 1, x + 1.5)
                draw_path(ax, [(xa, yy), (xb, yy)], [Path.MOVETO, Path.LINETO], ink2, lw=0.6, z=7, cap='butt')
                chevron(ax, xb, yy, d, 0, ink2, s=1.9, lw=0.55, z=7)
        ax.text(tx, y, text, ha='left', va='center_baseline', fontsize=LEG_FS, color=ink, zorder=7)
    # right block, right-aligned to the frame
    wr = G['reserve'][1] - 12
    x = G['hi'] - 3.0 - wr
    for k, (kind, text, cnt) in enumerate(right):
        y = y0 + k * LEG_LINE
        if kind == 'ext':
            fc, ec, ls = 'white', ink3, (0, (2.0, 1.1))
        else:
            fc, ec = kind_colours(kind[5:])
            ls = 'solid'
        sw = FancyBboxPatch((x + 0.5, y - 2.6), 9.0, 5.2, boxstyle='round,pad=0,rounding_size=1.4', fc=fc, ec=ec,
                            lw=0.6, zorder=7)
        sw.set_linestyle(ls)
        ax.add_patch(sw)
        t = ax.text(x + 13.0, y, text, ha='left', va='center_baseline', fontsize=LEG_FS, color=ink, zorder=7)
        ax.text(x + wr, y, cnt, ha='right', va='center_baseline', fontsize=LEG_FS, color=ink2, zorder=7)



def draw(model, G):
    W, H = G['W'], G['H']
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    pos, ends, nodes = G['pos'], G['ends'], model['nodes']
    ink, ink2, ink3 = C['ink'], C['ink2'], C['ink3']
    fs_ann = 7.0

    # --- A1: the interval frame from Frozen out to the margins and up into Full FT
    fz, ff = ends['frozen'], ends['full']
    xl, xr, rr = G['lo'] - 4.6, G['hi'] + 4.6, 6.0
    for s, xm in ((-1, xl), (1, xr)):
        x0, x1 = fz['x'] + s * fz['w'] / 2, ff['x'] + s * ff['w'] / 2
        verts = [(x0, fz['y']), (xm - s * rr, fz['y']), (xm, fz['y']), (xm, fz['y'] - rr), (xm, ff['y'] + rr),
                 (xm, ff['y']), (xm - s * rr, ff['y']), (x1 + s * 1.6, ff['y'])]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO]
        draw_path(ax, verts, codes, ink3, lw=0.5, dash=(2.0, 2.2), z=1)
        chevron(ax, x1 + s * 1.6, ff['y'], -s, 0, ink3, s=2.6, lw=0.55, z=1)
    ax.text(xl, (fz['y'] + ff['y']) / 2, 'A1', rotation=90, ha='center', va='center', fontsize=fs_ann, color=ink2,
            fontweight='medium', zorder=2, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax.text(xr, (fz['y'] + ff['y']) / 2, 'A1', rotation=-90, ha='center', va='center', fontsize=fs_ann, color=ink2,
            fontweight='medium', zorder=2, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    for key, sub in (('full', r'terminal $(\Theta,\theta_0,\mathrm{id})$'), ('frozen', r'initial $(\mathrm{pt},\ast,\theta_0)$')):
        e = ends[key]
        p = FancyBboxPatch((e['x'] - e['w'] / 2, e['y'] - e['h'] / 2), e['w'], e['h'],
                           boxstyle='round,pad=0,rounding_size=2.4', fc='white', ec=ink, lw=0.8, zorder=5)
        ax.add_patch(p)
        ax.text(e['x'], e['y'], e['label'], ha='center', va='center_baseline', fontsize=END_FS, color=ink,
                fontweight='medium', zorder=6)
        dy = (e['h'] / 2 + END_SUB / 2 + 0.6) * (1 if key == 'full' else -1)
        ax.text(e['x'], e['y'] + dy, sub, ha='center', va='center_baseline', fontsize=fs_ann, color=ink2, zorder=6)

    # --- shelf frame
    if G['shelf_box']:
        x0, x1, y0, y1 = G['shelf_box']
        draw_path(ax, [(x0 + 4, y0 + 3), (x1 - 4, y0 + 3)], [Path.MOVETO, Path.LINETO], C['rule'], lw=0.6,
                  dash=(2.0, 2.0), z=1)
        n = len(G['shelf'])
        ax.text((x0 + x1) / 2, y0 + 3, '%d objects with no A2–A5 arrow to or from another object' % n, ha='center',
                va='center_baseline', fontsize=fs_ann, color=ink2, zorder=2,
                bbox=dict(boxstyle='square,pad=0.25', fc='white', ec='none'))

    # --- hubs: incoming arrows bundled into one trunk
    hub_of = {}
    for m in model['methods']:
        ext = [a for a in model['inn'][m['id']] if not a['inGroup']]
        p = pos[m['id']]
        if len(ext) >= HUB:
            hub_of[m['id']] = dict(n=len(ext), cx=p['x'], cy=p['y'] + p['h'] / 2 + 7.0, top=p['y'] + p['h'] / 2 + 0.6)

    # --- inter-level arrows: source top -> dummies -> target bottom (or trunk)
    drawn = [a for a in model['arrows'] if not a['self'] and not a['inGroup']]

    def first_x(a):
        d = a['ce'].get('dum')
        return d[0].x if d else pos[a['t']]['x']

    def last_x(a):
        d = a['ce'].get('dum')
        return d[-1].x if d else pos[a['s']]['x']
    out_p, in_p = {}, {}
    for m in model['methods']:
        o = sorted([a for a in drawn if a['s'] == m['id']], key=lambda a: (first_x(a), a['i']))
        w = pos[m['id']]['w']
        sp = min(2.6, (w - 5) / (len(o) - 1)) if len(o) > 1 else 0.0
        for k, a in enumerate(o):
            out_p[a['i']] = (k - (len(o) - 1) / 2) * sp
        if m['id'] not in hub_of:
            ii = sorted([a for a in drawn if a['t'] == m['id']], key=lambda a: (last_x(a), a['i']))
            sp2 = min(2.6, (w - 5) / (len(ii) - 1)) if len(ii) > 1 else 0.0
            for k, a in enumerate(ii):
                in_p[a['i']] = (k - (len(ii) - 1) / 2) * sp2
    LW_E = 0.55
    for a in drawn:
        ps, pt, R0 = pos[a['s']], pos[a['t']], RULES[a['rule']]
        hub = hub_of.get(a['t'])
        tx = hub['cx'] if hub else pt['x'] + in_p.get(a['i'], 0.0)
        ty = hub['cy'] if hub else pt['y'] + pt['h'] / 2 + 0.9
        alpha = 0.55 if a.get('composite') else 0.95
        if ps['comb'] is not None:
            # a leaf of a comb: tick up to the rail of its row; the rail and spine are drawn once, in ink
            c = ps['comb']
            ry = c.y + c.rails[ps['row']][0]
            sx = ps['x'] + out_p.get(a['i'], 0.0)
            sy = ps['y'] - ps['h'] / 2 - (2.4 if R0['hook'] else 0.5)
            draw_path(ax, [(sx, sy), (sx, ry)], [Path.MOVETO, Path.LINETO], R0['col'], lw=0.7, dash=R0['dash'],
                      z=2, cap='butt')
            if R0['hook']:
                hook(ax, sx, sy, R0['col'], lw=0.6)
            if not hub:
                # into a member of the hub's class that is not the hub: from the top of the spine to the target
                verts, codes = monotone_path([(tx, ty), (tx, ty + 2.6), (c.x, c.y - c.h / 2)])
                draw_path(ax, verts, codes, R0['col'], lw=LW_E, dash=R0['dash'], z=2, alpha=alpha)
                chevron(ax, tx, ty, 0, -1, R0['col'], lw=LW_E, z=3, alpha=alpha)
            continue
        sx = ps['x'] + out_p.get(a['i'], 0.0)
        sy = ps['y'] - ps['h'] / 2 - (2.4 if R0['hook'] else 0.5)
        pts = [(tx, ty), (tx, ty + (3.5 if hub else 2.6))]
        for dm in reversed(a['ce'].get('dum') or []):
            if dm.t == 'c':
                pts += [(dm.x, dm.y - dm.h / 2 - 1.0), (dm.x, dm.y + dm.h / 2 + COMB_JOIN)]
            else:
                pts += [(dm.x, dm.y - 1.6), (dm.x, dm.y + 1.6)]
        pts += [(sx, sy - 2.6), (sx, sy)]
        verts, codes = monotone_path(pts)
        draw_path(ax, verts, codes, R0['col'], lw=LW_E, dash=R0['dash'], z=2, alpha=alpha)
        if not hub:
            chevron(ax, tx, ty, 0, -1, R0['col'], lw=LW_E, z=3, alpha=alpha)
        if R0['hook']:
            hook(ax, sx, sy, R0['col'], lw=LW_E)

    # --- comb buses: rails and spine, drawn once in ink over the ticks
    for c in G['combs']:
        hub = None
        for mid, h0 in hub_of.items():
            if model['gof'][mid] is c.hub and (hub is None or h0['n'] > hub['n']):
                hub, hub_id = h0, mid
        through = any(c in (ce.get('dum') or []) for ce in model['ces'])
        top_y = c.y - c.h / 2
        bot_y = c.y + c.h / 2 + COMB_JOIN if through else c.y + c.rails[-1][0]
        for (ry, rx0, rx1) in c.rails:
            ax.plot([c.x + rx0, c.x + rx1], [c.y + ry, c.y + ry], color=ink2, lw=0.75, zorder=3,
                    solid_capstyle='round')
        ax.plot([c.x, c.x], [bot_y, top_y], color=ink2, lw=0.95, zorder=3, solid_capstyle='round')
        if hub:
            verts, codes = monotone_path([(hub['cx'], hub['cy']), (c.x, top_y)])
            draw_path(ax, verts, codes, ink2, lw=0.95, z=3)
        single = all(len(model['out'][g['members'][0]]) == 1 and model['out'][g['members'][0]][0]['t'] == hub_id
                     for g in c.leaves)
        lab = ('%d objects whose only arrow goes into %s' % (len(c.leaves), nodes[hub_id]['label']) if single else
               '%d objects with arrows only into the %s class' % (len(c.leaves), nodes[hub_id]['label']))
        ax.text(c.x + 3.0, c.y + c.h / 2 + 4.6, lab, ha='left', va='center_baseline', fontsize=fs_ann, color=ink2,
                zorder=4, path_effects=HALO)
    for mid, h0 in hub_of.items():
        ax.plot([h0['cx'], h0['cx']], [h0['cy'] + 0.3, h0['top']], color=ink2, lw=0.95, zorder=3,
                solid_capstyle='butt')
        chevron(ax, h0['cx'], h0['top'], 0, -1, ink2, s=2.8, lw=0.8, z=3)
        if h0['n'] >= 12:
            ax.text(h0['cx'] - 3.0, h0['cy'] - 0.4, '%d arrows' % h0['n'], ha='right',
                    va='center', fontsize=fs_ann, color=ink2, zorder=4, path_effects=HALO)

    # --- classes of mutually simulating families: a dashed frame, double arrows between neighbours
    for g in model['groups']:
        if g['size'] < 2:
            continue
        it = G['g_item'][g['i']]
        p = FancyBboxPatch((it.x - it.w / 2 - 2.6, it.y - it.h / 2 - 2.2), it.w + 5.2, it.h + 4.4,
                           boxstyle='round,pad=0,rounding_size=3', fc=C['paper2'], ec=ink3, lw=0.45, zorder=1.5,
                           alpha=0.9)
        p.set_linestyle((0, (1.6, 1.4)))
        ax.add_patch(p)
    for a in model['arrows']:
        if not a['inGroup']:
            continue
        g = model['gof'][a['s']]
        ia, ib = g['members'].index(a['s']), g['members'].index(a['t'])
        ps, pt, R0 = pos[a['s']], pos[a['t']], RULES[a['rule']]
        if abs(ia - ib) == 1:
            d = 1 if ib > ia else -1
            yy = ps['y'] + (-1.6 if d > 0 else 1.6)
            x0, x1 = ps['x'] + d * (ps['w'] / 2 + 1.3), pt['x'] - d * (pt['w'] / 2 + 1.3)
            draw_path(ax, [(x0, yy), (x1, yy)], [Path.MOVETO, Path.LINETO], R0['col'], lw=0.6, dash=R0['dash'], z=4,
                      cap='butt')
            chevron(ax, x1, yy, d, 0, R0['col'], s=1.9, lw=0.55, z=4)
        else:
            yt, lift = ps['y'] - ps['h'] / 2 - 0.8, 7.0
            verts = [(ps['x'], yt), (ps['x'], yt - lift), (pt['x'], yt - lift), (pt['x'], yt)]
            draw_path(ax, verts, [Path.MOVETO] + [Path.CURVE4] * 3, R0['col'], lw=0.6, dash=R0['dash'], z=4)
            chevron(ax, pt['x'], yt, 0, 1, R0['col'], s=1.9, lw=0.55, z=4)

    # --- objects
    for m in model['methods']:
        n, p = nodes[m['id']], pos[m['id']]
        fc, ec = kind_colours(n['kind'])
        box = FancyBboxPatch((p['x'] - p['w'] / 2, p['y'] - p['h'] / 2), p['w'], p['h'],
                             boxstyle='round,pad=0,rounding_size=2.0', fc=fc, ec=ec, lw=0.6, zorder=5)
        if n['ext']:
            box.set_linestyle((0, (2.0, 1.1)))
        ax.add_patch(box)
        ax.text(p['x'], p['y'], n['label'], ha='center', va='center_baseline', fontsize=p['fs'], color=ink,
                zorder=6, fontweight='medium' if p['fs'] > FS + 0.5 else 'regular')
        # loops: arrows from a family to itself (another instance of the family)
        for k, a in enumerate(n['selfs']):
            R0 = RULES[a['rule']]
            cx, cy = p['x'] + p['w'] / 2 - 2.6 - k * 4.4, p['y'] - p['h'] / 2
            verts = [(cx - 2.0, cy - 0.2), (cx - 2.5, cy - 5.6), (cx + 4.2, cy - 5.2), (cx + 1.9, cy - 0.6)]
            draw_path(ax, verts, [Path.MOVETO] + [Path.CURVE4] * 3, R0['col'], lw=0.55, dash=R0['dash'], z=4.5)
            chevron(ax, cx + 1.9, cy - 0.6, -0.45, 1, R0['col'], s=1.6, lw=0.5, z=4.5)
    return fig, ax


def main():
    apply_style()
    D = load_data()
    model = build_model(D)
    measure = make_measure()
    W = W1 * 72
    reserve, legend_h = legend_size(model, measure)
    G = layout(model, W, measure, reserve, legend_h)
    G['reserve'] = reserve
    fig, ax = draw(model, G)
    draw_legends(ax, model, G)
    # the page is exactly the text width (no tight crop), so the figure is placed at scale 1 and 7 pt stays 7 pt
    matplotlib.rcParams['savefig.bbox'] = 'standard'
    path = save(fig, 'graph')
    st = model['stats']
    print('wrote', path)
    print('objects %d (%d of kind R), arrows %d (%s), loops %d, classes %d (%d methods), composites %d, isolated %d'
          % (st['objects'], st['R'], st['arrows'], ', '.join('%s %d' % (k, st['byRule'][k]) for k in RULE_ORDER),
             st['selfs'], st['cycles'], st['cycMembers'], st['composite'], st['isolated']))
    print('levels %d, rows %d, crossings %d, size %.2f x %.2f in' % (G['R'], sum(G['K']), G['crossings'], W / 72, G['H'] / 72))
    allr = defaultdict(int)
    for a in model['arrows']:
        allr[a['rule']] += 1
    trunks = {mid: len([a for a in model['inn'][mid] if not a['inGroup']]) for mid in model['nodes']}
    print('recorded arrows between core methods, loops included: %d (%s)' % (
        len(model['arrows']), ', '.join('%s %d' % (k, allr[k]) for k in RULE_ORDER)))
    print('trunks (>= %d arrows from outside the class): %s; comb leaves: %s' % (
        HUB, ', '.join('%s %d' % (k, v) for k, v in trunks.items() if v >= HUB),
        ', '.join(str(len(c.leaves)) for c in G['combs'])))
    return model, G


if __name__ == '__main__':
    main()
