#!/usr/bin/env python3
"""Compile research + theory + classification into site/data/atlas-data.js (window.ATLAS_DATA).

Inputs
  research/catalog.json            merged, verified method + paper catalog
  research/arxiv/meta.json         authoritative arXiv metadata (full author lists)
  research/classified/*.json       per-method coordinates and arrows (optional until classification runs)
  theory/axes.json, theory/rosetta.json, theory/propositions.json (optional)
"""
import json, os, re, glob, unicodedata, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
ROOT = os.path.dirname(SITE)
R = lambda *p: os.path.join(ROOT, *p)

def load(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default

cat = load(R('research', 'catalog.json'))
meta = load(R('research', 'arxiv', 'meta.json'), {})
axes = load(R('theory', 'axes.json'), {})
rosetta = load(R('theory', 'rosetta.json'), [])
props = load(R('theory', 'propositions.json'), [])

classified = {}
for p in sorted(glob.glob(R('research', 'classified', 'batch-*.json'))):
    d = load(p, {})
    for m in d.get('methods', []):
        classified[m['id']] = m
edges_file = load(R('research', 'classified', 'edges.json'), {})

# display overrides: prose fields rewritten with explicit \( \) math delimiters (research/display/*.json)
display = {}
for p in sorted(glob.glob(R('research', 'display', '*.json'))):
    for k, v in (load(p, {}) or {}).items():
        display.setdefault(k, {}).update(v)

# ---------------- BibTeX ----------------
def ascii_fold(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()

STOP = {'a', 'an', 'the', 'on', 'of', 'for', 'to', 'in', 'is', 'and', 'with', 'via', 'towards', 'toward', 'are', 'from', 'by', 'at', 'when', 'what', 'how', 'do', 'does'}

def bib_key(authors, year, title):
    sur = ascii_fold((authors[0] if authors else 'anon').split()[-1]).lower()
    sur = re.sub(r'[^a-z]', '', sur) or 'anon'
    words = [w for w in re.findall(r'[A-Za-z0-9]+', ascii_fold(title or '')) if w.lower() not in STOP]
    return f"{sur}{year or ''}{(words[0].lower() if words else 'x')}"

VENUE_MAP = [
    (r'^ICLR', 'International Conference on Learning Representations (ICLR)'),
    (r'^NeurIPS|^NIPS', 'Advances in Neural Information Processing Systems (NeurIPS)'),
    (r'^ICML', 'Proceedings of the International Conference on Machine Learning (ICML)'),
    (r'^COLM', 'Conference on Language Modeling (COLM)'),
    (r'Findings', 'Findings of the Association for Computational Linguistics'),
    (r'^ACL', 'Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)'),
    (r'^EMNLP', 'Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)'),
    (r'^NAACL', 'Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)'),
    (r'^EACL', 'Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL)'),
    (r'^COLING', 'Proceedings of the International Conference on Computational Linguistics (COLING)'),
    (r'^CVPR', 'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)'),
    (r'^ICCV', 'Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)'),
    (r'^ECCV', 'European Conference on Computer Vision (ECCV)'),
    (r'^AAAI', 'Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)'),
    (r'^IJCAI', 'Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)'),
    (r'^AISTATS', 'International Conference on Artificial Intelligence and Statistics (AISTATS)'),
    (r'^UAI', 'Conference on Uncertainty in Artificial Intelligence (UAI)'),
    (r'^KDD', 'Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)'),
    (r'^MLSys', 'Proceedings of Machine Learning and Systems (MLSys)'),
    (r'^ESOP', 'European Symposium on Programming (ESOP)'),
    (r'^LICS', 'Symposium on Logic in Computer Science (LICS)'),
    (r'^SIGGRAPH', 'ACM SIGGRAPH Conference Proceedings'),
    (r'^WACV', 'IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)'),
    (r'^ICASSP', 'IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)'),
    (r'^INTERSPEECH|^Interspeech', 'Interspeech'),
    (r'^CoLLAs', 'Conference on Lifelong Learning Agents (CoLLAs)'),
    (r'^ACT|^Applied Category Theory', 'Applied Category Theory (ACT)'),
]
JOURNALS = ['TMLR', 'Transactions on Machine Learning Research', 'Nature', 'JMLR', 'Journal of', 'TPAMI', 'Transactions', 'AI Open', 'Neural Networks', 'Science China', 'Proceedings of the Royal', 'Communications']

def tex_escape(s):
    return (s or '').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#').replace('_', r'\_')

def bibtex(entry, title, authors_full, year, venue, arxiv, url, first=None):
    key = bib_key(first or authors_full or [entry.get('authors', 'anon')], year, title)
    if authors_full:
        authors = ' and '.join(authors_full)
    else:  # catalogue short form: "Surname et al.", "A and B", "A & B"
        a = (entry.get('authors') or '').strip().rstrip('.')
        a = re.sub(r'\s*&\s*', ' and ', a)
        a = re.sub(r',?\s+et al$', ' and others', a)
        authors = a
    fields = [('title', '{' + tex_escape(title) + '}'), ('author', tex_escape(authors)), ('year', str(year or ''))]
    v = (venue or '').strip()
    typ = 'misc'
    if v and not re.match(r'^arxiv', v, re.I) and 'blog' not in v.lower() and 'software' not in v.lower():
        if any(j.lower() in v.lower() for j in JOURNALS):
            typ = 'article'; fields.append(('journal', tex_escape(re.sub(r'\s*\d{4}\s*$', '', v))))
        else:
            book = None
            for pat, name in VENUE_MAP:
                if re.search(pat, v):
                    book = name; break
            if book:
                if 'Findings' in v:
                    m = re.search(r'(ACL|EMNLP|NAACL|EACL|AACL|IJCNLP)[^0-9]*(\d{4})', v)
                    if m: book = f'Findings of the Association for Computational Linguistics: {m.group(1)} {m.group(2)}'
                typ = 'inproceedings'; fields.append(('booktitle', book))
            else:
                typ = 'inproceedings'; fields.append(('booktitle', tex_escape(re.sub(r'\s*\d{4}\s*$', '', v))))
    if arxiv:
        fields += [('eprint', arxiv), ('archivePrefix', 'arXiv')]
        if typ == 'misc':
            typ = 'article'; fields.append(('journal', f'arXiv preprint arXiv:{arxiv}'))
    if url:
        fields.append(('url', url))
    if typ == 'misc' and v:
        fields.append(('howpublished', tex_escape(v)))
    body = ',\n'.join(f'  {k} = {{{val}}}' if not val.startswith('{') else f'  {k} = {{{val}}}' for k, val in fields)
    return key, f'@{typ}{{{key},\n{body}\n}}'

def enrich(x, is_method):
    a = (x.get('arxiv') or '').strip()
    m = meta.get(a, {})
    title = (x.get('paper_title') if is_method else x.get('title')) or m.get('title') or x.get('name') or ''
    authors_full = m.get('authors') or []
    first = authors_full[:1] or [re.split(r'\s+et al\.?|\s+and\s+|\s*&\s*', (x.get('authors') or 'anon').strip())[0]]
    key, bib = bibtex(x, title, authors_full, x.get('year'), x.get('venue'), a, x.get('url'), first)
    x['authors_full'] = authors_full
    x['bibkey'] = key
    x['bibtex'] = bib
    return x

DROP = {'corrections', 'verification_notes', 'sources_checked'}
methods = []
seen_keys = {}
for m in cat['methods']:
    m = {k: v for k, v in m.items() if k not in DROP}
    enrich(m, True)
    c = classified.get(m['id'])
    if c:
        for k in ('coords', 'coord_notes', 'categorical_reading', 'entry_type', 'core', 'arrows', 'obstructions', 'table_row', 'table_col'):
            if k in c:
                m[k] = c[k]
    for k, v in display.get(m['id'], {}).items():
        if k in m and isinstance(v, str):
            m[k] = v
        elif k == 'coord_notes' and isinstance(v, dict) and isinstance(m.get('coord_notes'), dict):
            m['coord_notes'] = dict(m['coord_notes'], **v)
    methods.append(m)
papers = []
for p in cat['papers']:
    p = {k: v for k, v in p.items() if k not in DROP}
    for k, v in display.get('paper:' + p.get('id', ''), {}).items():
        if k in p and isinstance(v, str):
            p[k] = v
    enrich(p, False)
    papers.append(p)
# disambiguate duplicate bib keys
for coll in (methods, papers):
    for x in coll:
        k = x['bibkey']
        if k in seen_keys and seen_keys[k] != (x.get('arxiv') or x.get('url')):
            n = 2
            while f'{k}{chr(96 + n)}' in seen_keys: n += 1
            nk = f'{k}{chr(96 + n)}'
            x['bibtex'] = x['bibtex'].replace('{' + k + ',', '{' + nk + ',', 1)
            x['bibkey'] = nk; k = nk
        seen_keys[k] = x.get('arxiv') or x.get('url')

data = {
    'meta': {
        'title': 'A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges',
        'built': datetime.date.today().isoformat(),
        'counts': {'methods': len(methods), 'papers': len(papers), 'hf_peft': sum(1 for m in methods if m.get('hf_peft')),
                   'classified': sum(1 for m in methods if m.get('coords')), 'propositions': len(props) if isinstance(props, list) else 0},
    },
    'methods': methods,
    'papers': papers,
    'axes': axes.get('axes', []),
    'arrow_rules': axes.get('arrow_rules', []),
    'edges': edges_file.get('edges', []),
    'rosetta': rosetta if isinstance(rosetta, list) else rosetta.get('rows', []),
    'propositions': props if isinstance(props, list) else props.get('propositions', []),
}
os.makedirs(os.path.join(SITE, 'data'), exist_ok=True)
js = 'window.ATLAS_DATA = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n'
open(os.path.join(SITE, 'data', 'atlas-data.js'), 'w').write(js)
print('wrote atlas-data.js', round(len(js) / 1024), 'KB', data['meta']['counts'])
