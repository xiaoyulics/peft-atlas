#!/usr/bin/env python3
"""paper/bib/refs.bib and paper/bib/keys.tsv from the verified catalogue (site/data/atlas-data.js), plus bib/extra/*.bib."""
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE); ROOT = os.path.dirname(PAPER)
src = open(os.path.join(ROOT, 'site', 'data', 'atlas-data.js')).read(); d = json.loads(src[src.index('{'):src.rstrip().rindex('}') + 1])
REP = {'³': '$^3$', '²': '$^2$', '¹': '$^1$', '–': '--', '—': '---', '’': "'", '‘': '`', '“': '``', '”': "''", '×': '$\\times$', '≤': '$\\le$', '≥': '$\\ge$',
       '→': '$\\to$', 'α': '$\\alpha$', 'β': '$\\beta$', 'λ': '$\\lambda$', 'μ': '$\\mu$', '∞': '$\\infty$', '·': '$\\cdot$', '⊗': '$\\otimes$', '√': '$\\surd$',
       '∈': '$\\in$', 'ε': '$\\varepsilon$', 'θ': '$\\theta$', 'Δ': '$\\Delta$', 'π': '$\\pi$', 'σ': '$\\sigma$', 'ρ': '$\\rho$', 'γ': '$\\gamma$', 'δ': '$\\delta$',
       'ℓ': '$\\ell$', 'Σ': '$\\Sigma$', 'Φ': '$\\Phi$', 'Ω': '$\\Omega$', '∘': '$\\circ$', '≈': '$\\approx$', ' ': ' ', ' ': ' ', '​': ''}
def clean(b):
    for a, c in REP.items(): b = b.replace(a, c)
    return b
seen, rows = {}, []
for kind, coll in (('method', d['methods']), ('paper', d['papers'])):
    for x in coll:
        k, b = x.get('bibkey'), x.get('bibtex')
        if not k or not b: continue
        seen.setdefault(k, b)
        rows.append('\t'.join([k, kind, x.get('id', ''), (x.get('name') or x.get('title', '')).replace('\t', ' '), str(x.get('year', '')), x.get('arxiv', ''), (x.get('paper_title') or x.get('title') or '').replace('\t', ' ')]))
# bib/overrides.bib: corrected entries that replace catalogue entries with the same key
import re
ovp = os.path.join(PAPER, 'bib', 'overrides.bib')
if os.path.exists(ovp):
    for m in re.finditer(r'@\w+\{([^,\s]+),.*?\n\}', open(ovp).read(), re.S):
        if m.group(1) in seen: seen[m.group(1)] = m.group(0)
        else: print('override for unknown key', m.group(1))
# catalogue entries that name a journal or a thesis in `booktitle`: retype them, so the reference list prints
# "Journal" instead of "In Journal" (the year stays that of the catalogue, the first public version)
JOURNALS = {'IJCV': 'International Journal of Computer Vision', 'ACM Computing Surveys': 'ACM Computing Surveys',
            'APL Machine Learning': 'APL Machine Learning', 'TACL': 'Transactions of the Association for Computational Linguistics',
            'IEEE/ACM TASLP': 'IEEE/ACM Transactions on Audio, Speech, and Language Processing',
            'Frontiers of Computer Science': 'Frontiers of Computer Science', 'Computational Statistics': 'Computational Statistics',
            'Foundations and Trends in Machine Learning 9(4-5):249-429': 'Foundations and Trends in Machine Learning'}
def retype(b):
    m = re.search(r'\n(\s*)booktitle\s*=\s*\{([^{}]*)\},?', b)
    if not m or not b.lstrip().lower().startswith('@inproceedings'): return b
    bt = m.group(2)
    if bt in JOURNALS:
        extra_f = ''
        if bt.startswith('Foundations and Trends'):
            extra_f = ',\n' + m.group(1) + 'volume = {9},\n' + m.group(1) + 'number = {4--5},\n' + m.group(1) + 'pages = {249--429}'
        b = re.sub(r'^\s*@inproceedings', '@article', b, flags=re.I)
        return b.replace(m.group(0), '\n' + m.group(1) + 'journal = {' + JOURNALS[bt] + '}' + extra_f + (',' if m.group(0).endswith(',') else ''))
    if bt.startswith('PhD thesis, '):
        b = re.sub(r'^\s*@inproceedings', '@phdthesis', b, flags=re.I)
        return b.replace(m.group(0), '\n' + m.group(1) + 'school = {' + bt[len('PhD thesis, '):] + '}' + (',' if m.group(0).endswith(',') else ''))
    return b
seen = {k: retype(b) for k, b in seen.items()}
extra = []
for p in sorted(glob.glob(os.path.join(PAPER, 'bib', 'extra', '*.bib'))):
    extra.append('% from ' + os.path.basename(p) + '\n' + open(p).read())
open(os.path.join(PAPER, 'bib', 'refs.bib'), 'w').write('\n\n'.join(clean(b) for b in seen.values()) + '\n\n' + '\n\n'.join(extra) + '\n')
open(os.path.join(PAPER, 'bib', 'keys.tsv'), 'w').write('bibkey\tkind\tid\tname\tyear\tarxiv\ttitle\n' + '\n'.join(rows) + '\n')
print(len(seen), 'catalogue entries;', len(extra), 'extra files')
