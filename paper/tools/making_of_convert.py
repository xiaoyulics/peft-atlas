#!/usr/bin/env python3
"""Convert paper/_reviews/making_of.md (a restricted Markdown) into the site section and the paper appendix."""
import re, html, os
HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE); ROOT = os.path.dirname(PAPER)
src = open(os.path.join(PAPER, '_reviews', 'making_of.md')).read().split('\n')
src = [l for l in src if not l.startswith('# ')]   # drop the document title

def blocks(lines):
    """yield ('h', text) | ('p', text) | ('ul', items) | ('ol', items) | ('table', rows); items are (level, text)"""
    i = 0; para = []
    def flush():
        nonlocal para
        if para: out.append(('p', ' '.join(para))); para = []
    out = []
    while i < len(lines):
        l = lines[i]
        if l.startswith('## '): flush(); out.append(('h', l[3:].strip())); i += 1; continue
        if l.startswith('|'):
            flush(); rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'-+', c) for c in cells): rows.append(cells)
                i += 1
            out.append(('table', rows)); continue
        m = re.match(r'^( *)([-*]|\d+\.) (.*)$', l)
        if m:
            flush(); kind = 'ol' if m.group(2)[0].isdigit() else 'ul'; items = []
            while i < len(lines):
                m2 = re.match(r'^( *)([-*]|\d+\.) (.*)$', lines[i])
                if not m2: break
                items.append((len(m2.group(1)) // 2, m2.group(3))); i += 1
            out.append((kind, items)); continue
        if not l.strip(): flush(); i += 1; continue
        para.append(l.strip()); i += 1
    flush(); return out

B = blocks(src)

# ---------------------------------------------------------------- HTML
VARIANT = re.compile(r'\{\{(.*?)\|\|(.*?)\}\}')   # {{site text||paper text}}
CITE = re.compile(r'\s*\[(@[^\]]+)\]')             # [@key; @key2] -> \citep{key,key2} / linked author-year on the site
CITE_HTML = {    # key -> (label, url) for every key the section cites
    'xgrothendieck2022recoltes': ('Grothendieck 2022', 'https://catalogue.bnf.fr/ark:/12148/cb46976953t'),
    'xmclarty2007rising': ('McLarty 2007', 'https://doi.org/10.1090/hmath/032/14'),
}
def cite_keys(m): return [k.strip().lstrip('@') for k in m.group(1).split(';')]
def cite_html(m):
    parts = []
    for k in cite_keys(m):
        label, url = CITE_HTML[k]
        parts.append('<a href="%s">%s</a>' % (url, html.escape(label)) if url else html.escape(label))
    return ' (' + '; '.join(parts) + ')'

def ih(t):
    t = VARIANT.sub(r'\1', t)
    cites = []
    t = CITE.sub(lambda m: cites.append(cite_html(m)) or '\x00%d\x00' % (len(cites) - 1), t)
    t = html.escape(t, quote=False)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = t.replace('ρ_N∘h = ρ_M', '\\(\\rho_N\\circ h=\\rho_M\\)').replace('GL(r)', '\\(GL_r\\)')
    t = re.sub('\x00(\\d+)\x00', lambda m: cites[int(m.group(1))], t)
    return t
def list_html(kind, items):
    out = []; stack = []
    for lvl, text in items:
        while len(stack) > lvl + 1: out.append('</li></%s>' % stack.pop())
        if len(stack) == lvl + 1: out.append('</li>')
        while len(stack) < lvl + 1:
            k = kind if not stack else 'ul'; out.append('<%s>' % k); stack.append(k)
        out.append('<li>' + ih(text))
    while stack: out.append('</li></%s>' % stack.pop())
    return ''.join(out)
H = []
for kind, x in B:
    if kind == 'h': H.append('<h3 id="making-%s">%s</h3>' % (re.sub(r'[^a-z]+', '-', x.lower()).strip('-'), ih(x)))
    elif kind == 'p': H.append('<p>' + ih(x) + '</p>')
    elif kind in ('ul', 'ol'): H.append(list_html(kind, x))
    elif kind == 'table':
        head, *body = x
        H.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th%s>%s</th>' % (' class="num"' if j > 0 and any(c.isdigit() for r in body for c in r[j:j+1]) and head[0] == 'Phase' else '', ih(c)) for j, c in enumerate(head)) + '</tr></thead><tbody>'
                 + ''.join('<tr>' + ''.join('<td%s>%s</td>' % (' class="num"' if j > 0 and head[0] == 'Phase' else '', ih(c)) for j, c in enumerate(r)) + '</tr>' for r in body) + '</tbody></table></div>')
section = ('<section class="expose" id="making-of" data-toc="∗" data-toc-title="How this was made">\n'
           '  <header class="expose-head"><span class="label">Afterword</span><h2>How this atlas was made</h2>'
           '<p class="summary">Who did what, how the material was found, how the results were generalised, unified and checked, and what it cost.</p></header>\n'
           '  <div class="prose">\n' + '\n'.join(H) + '\n  </div>\n</section>\n')
open(os.path.join(ROOT, 'site', 'sections', '15-making-of.html'), 'w').write(section)

# ---------------------------------------------------------------- LaTeX
def il(t):
    t = VARIANT.sub(r'\2', t)
    cites = []
    t = CITE.sub(lambda m: cites.append('~\\citep{%s}' % ','.join(cite_keys(m))) or '\x00%d\x00' % (len(cites) - 1), t)
    t = t.replace('this section', 'this appendix').replace('This section', 'This appendix')
    t = t.replace('\\', '\\textbackslash{}')
    for a, b in (('&', '\\&'), ('%', '\\%'), ('#', '\\#'), ('_', '\\_'), ('$', '\\$')):
        t = t.replace(a, b)
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\\emph{\1}', t)
    t = re.sub(r'`([^`]+)`', lambda m: '\\texttt{' + m.group(1) + '}', t)
    t = re.sub(r'"([^"]+)"', r"``\1''", t)
    t = t.replace('ρ\\_N∘h = ρ\\_M', '$\\rho_N\\circ h=\\rho_M$').replace('GL(r)', '$\\GL_r$')
    t = t.replace('–', '--').replace(' – ', ' -- ')
    t = re.sub('\x00(\\d+)\x00', lambda m: cites[int(m.group(1))], t)
    return t
def list_tex(kind, items):
    env = {'ul': 'itemize', 'ol': 'enumerate'}
    out = []; stack = []
    for lvl, text in items:
        while len(stack) > lvl + 1: out.append('\\end{%s}' % stack.pop())
        while len(stack) < lvl + 1:
            e = env[kind] if not stack else 'itemize'; out.append('\\begin{%s}' % e); stack.append(e)
        out.append('\\item ' + il(text))
    while stack: out.append('\\end{%s}' % stack.pop())
    return '\n'.join(out)
L = ['\\section{How this paper and its website were made}\\label{app:making-of}',
     'This appendix records how the paper and its companion site were produced: who did what, how the material was found, how the results were generalised, unified and checked, and what it cost.']
for kind, x in B:
    if kind == 'h': L.append('\\subsection{%s}' % il(x))
    elif kind == 'p': L.append(il(x))
    elif kind in ('ul', 'ol'): L.append(list_tex(kind, x))
    elif kind == 'table':
        head, *body = x; n = len(head)
        spec = 'p{3.6cm}p{11.2cm}' if n == 2 else 'p{5.4cm}' + 'r' * (n - 1)
        cap = ('Timeline of the work, in Australian Eastern Standard Time.' if n == 2 else
               'Agent runs and tokens by phase. Fresh tokens are prompts written to the cache plus generated text; re-reads from the prompt cache are not counted.')
        lab = 'tab:making-timeline' if n == 2 else 'tab:making-cost'
        if L and L[-1].startswith('\\subsection'):   # a float may drift; give the heading a sentence that points to it
            L.append('The main steps are listed in Table~\\ref{%s}.' % lab if n == 2 else
                     'Table~\\ref{%s} breaks the agent runs and tokens down by phase.' % lab)
        L.append('\\begin{table}[htbp]\\centering\\caption{%s}\\label{%s}\\footnotesize\n\\begin{tabular}{@{}%s@{}}\\toprule\\headrow %s\\\\\\midrule\n%s\\\\\\bottomrule\\end{tabular}\\end{table}' % (
            cap, lab, spec, ' & '.join('\\textbf{%s}' % il(c) for c in head), '\\\\\n'.join(' & '.join(il(c) for c in r) for r in body)))
open(os.path.join(PAPER, 'sections', 'F-making-of.tex'), 'w').write('\n\n'.join(L) + '\n')
print('wrote site/sections/15-making-of.html and paper/sections/F-making-of.tex;', len(B), 'blocks')
