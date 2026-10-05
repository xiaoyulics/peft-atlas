#!/usr/bin/env python3
"""Merge the per-family verified catalogs into one deduplicated catalog.

Methods dedupe on canonical id only, plus an explicit alias table (from the completeness critic).
Name-based matching is deliberately NOT used: it folds distinct methods together (LoRA+ -> LoRA, O-LoRA -> OLoRA).
Several arXiv ids legitimately carry two methods (Pfeiffer adapter / AdapterFusion, ...), so no arXiv dedupe for methods.
Papers dedupe on arXiv id, else normalised title. On conflict, prefer verified, then confidence, then non-empty fields.
"""
import json, re, sys, glob, os, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
ALIASES = {'loraplus': 'lora-plus', 'riemannian-lora': 'riemannian-preconditioned-lora', 'shadowpeft': 'shadow-peft',
           'reft': 'loreft', 'adapter-houlsby': 'houlsby-adapter'}
PAPER_ALIASES = {}
SRC = sorted(glob.glob(os.path.join(ROOT, 'verified', '*.json')))
CONF = {'high': 3, 'medium': 2, 'low': 1}

def norm_name(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower().replace('³', '3').replace('^3', '3'))

def score(m):
    return (1 if m.get('verified') else 0, CONF.get(m.get('confidence', 'low'), 0), sum(1 for v in m.values() if v not in ('', None, [], {})))

def merge_two(a, b):
    """a is preferred; fill gaps from b; union list fields."""
    out = dict(a)
    for k, v in b.items():
        if k in ('relations', 'sources', 'corrections'):
            seen, merged = set(), []
            for item in (a.get(k) or []) + (v or []):
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key); merged.append(item)
            out[k] = merged
        elif out.get(k) in ('', None, [], {}) and v not in ('', None, [], {}):
            out[k] = v
    out['families'] = sorted(set((a.get('families') or []) + (b.get('families') or [])))
    return out

def main():
    methods, papers, notes = [], [], {}
    for path in SRC:
        fam = os.path.splitext(os.path.basename(path))[0]
        try:
            data = json.load(open(path))
        except Exception as e:
            print('SKIP (bad json)', path, e, file=sys.stderr); continue
        notes[fam] = {'notes': data.get('notes', ''), 'verification_summary': data.get('verification_summary', '')}
        for m in data.get('methods', []):
            m = dict(m); m['families'] = [data.get('family', fam)]
            methods.append(m)
        for p in data.get('papers', []):
            p = dict(p); p['families'] = [data.get('family', fam)]
            papers.append(p)

    # --- dedupe methods (id + explicit aliases only) ---
    by_key, order = {}, []
    canon = lambda x: ALIASES.get((x or '').strip().lower(), (x or '').strip().lower())
    for m in methods:
        key = canon(m.get('id'))
        if key in by_key:
            a, b = (by_key[key], m) if score(by_key[key]) >= score(m) else (m, by_key[key])
            merged = merge_two(a, b); merged['id'] = key
            by_key[key] = merged
        else:
            m['id'] = key; by_key[key] = m; order.append(key)
    merged_methods = [by_key[k] for k in order]
    ids = set(order)
    dangling = collections.Counter()
    for m in merged_methods:
        rels, seen = [], set()
        for r in m.get('relations') or []:
            t = canon(r.get('target'))
            if t == m['id'] or (t, r.get('type')) in seen:
                continue
            seen.add((t, r.get('type')))
            if t not in ids:
                dangling[t] += 1
            rels.append(dict(r, target=t))
        m['relations'] = rels

    # --- dedupe papers ---
    pk, merged_papers = {}, []
    for p in papers:
        k = (p.get('arxiv') or '').strip() or norm_name(p.get('title'))
        if k in pk:
            i = pk[k]
            a, b = (merged_papers[i], p) if score(merged_papers[i]) >= score(p) else (p, merged_papers[i])
            merged_papers[i] = merge_two(a, b)
        else:
            pk[k] = len(merged_papers); merged_papers.append(p)

    # link papers that are the source of a catalogued method
    by_arxiv = collections.defaultdict(list)
    for m in merged_methods:
        if m.get('arxiv'): by_arxiv[m['arxiv'].strip()].append(m['id'])
    for p in merged_papers:
        p['method_ids'] = by_arxiv.get((p.get('arxiv') or '').strip(), [])
    out = {'methods': merged_methods, 'papers': merged_papers, 'family_notes': notes,
           'stats': {'methods': len(merged_methods), 'papers': len(merged_papers), 'raw_methods': len(methods),
                     'verified_methods': sum(1 for m in merged_methods if m.get('verified')),
                     'hf_peft': sum(1 for m in merged_methods if m.get('hf_peft')),
                     'dangling_relation_targets': dict(dangling.most_common())}}
    json.dump(out, open(os.path.join(ROOT, 'catalog.json'), 'w'), ensure_ascii=False, indent=1)
    s = out['stats']
    print(f"methods {s['methods']} (raw {s['raw_methods']}, verified {s['verified_methods']}, hf {s['hf_peft']}); papers {s['papers']}")
    print('kinds', collections.Counter(m.get('modification_kind') for m in merged_methods).most_common())
    print('years', sorted(collections.Counter(m.get('year') for m in merged_methods).items(), key=lambda kv: (kv[0] is None, kv[0])))
    print('dangling', list(dangling.items())[:40])

if __name__ == '__main__':
    main()
