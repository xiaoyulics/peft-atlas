#!/usr/bin/env python3
"""Assemble the site.

  site/sections/NN-*.html  ->  site/index.html          (multi-file: GitHub Pages / local)
                           ->  dist/artifact.html       (single file, everything inlined: claude.ai Artifact)

Section 00 is the hero and sits outside <main>. Every other section is
<section class="expose" id=... data-toc=... data-toc-title=...>.
"""
import glob, html, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
ROOT = os.path.dirname(SITE)
DIST = os.path.join(ROOT, 'dist')

TITLE = 'A Categorical Atlas of Parameter-Efficient Fine-Tuning'
# X and most link unfurlers only fetch an absolute og:image. Set ATLAS_SITE_URL to the published root
# (for example https://<user>.github.io/<repo>/) before the launch build; without it the card path stays relative.
SITE_URL = os.environ.get('ATLAS_SITE_URL', '').strip()
CARD_ALT = ('A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges. Every PEFT method is a morphism into the pretrained point. A chart of weight space '
            'with the pretrained point, the LoRA cone at its apex, the OFT orbit and the (IA)3 line through it.')
# favicon: the pretrained point with a cone through it, on chart navy
FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Crect width='32' height='32' rx='6' fill='%230a121c'/%3E"
           "%3Cpath d='M9 7h14L9 25h14z' fill='none' stroke='%234cc3cf' stroke-width='1.6' stroke-linejoin='round'/%3E"
           "%3Ccircle cx='16' cy='16' r='2.6' fill='%23e3e9ee'/%3E%3C/svg%3E")
DESC = ('A categorical atlas of parameter-efficient fine-tuning that reads a PEFT method as a morphism into the pretrained point. '
        'Its image is what it can say, its fibres are how it learns, and base change is how it travels. '
        'Interactive figures, 41 numbered results and a checked catalogue of 382 methods.')
FONTS = ('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300..700;1,8..60,300..700'
         '&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap')
D3 = 'https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js'
MATHJAX = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js'
MJ_CONFIG = r"""window.MathJax = {
  tex: { inlineMath: [['\\(', '\\)'], ['$', '$']], displayMath: [['\\[', '\\]'], ['$$', '$$']], processEscapes: true,
         macros: { R: '\\mathbb{R}', Meth: '\\mathbf{Meth}', Diff: '\\mathbf{Diff}', Para: '\\mathbf{Para}', id: '\\mathrm{id}', Im: '\\operatorname{Im}', rk: '\\operatorname{rk}' } },
  svg: { fontCache: 'global' },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
  startup: { ready: function () { MathJax.startup.defaultReady(); var go = function () { if (window.Atlas && Atlas.glueMath) Atlas.glueMath(document); document.dispatchEvent(new CustomEvent('atlas:mathjax')); }; MathJax.startup.promise.then(go, go); } }
};"""

# Top-bar label for each exposé, keyed by its section id. Every link points at a <section class="expose">, so the
# scroll spy can mark it; a section left out here (the abstract) is still in the side table of exposés.
NAV = [('idea', 'Idea'), ('image', 'Image'), ('erlangen', 'Erlangen'), ('rank', 'Rank'), ('fibres', 'Fibres'),
       ('lens', 'Lens'), ('base', 'Base change'), ('merge', 'Merging'), ('atlas', 'Atlas'), ('rosetta', 'Rosetta'),
       ('predictions', 'Predictions'), ('appendix', 'Appendix'), ('cite', 'Cite'), ('making-of', 'Making of')]

# Side-table titles that should follow the exposé's current heading rather than its data-toc-title.
TOC_TITLE = {'image': 'Image and initialisation', 'rank': 'Rank and dimension', 'predictions': 'Predictions and open problems'}

SPY = r"""(function () {
  function openTo(id) {
    var el = id && document.getElementById(id); if (!el) return;
    for (var d = el.closest('details'); d; d = d.parentElement && d.parentElement.closest('details')) d.open = true;
  }
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#"]'); if (a) openTo(decodeURIComponent(a.getAttribute('href').slice(1)));
  }, true);
  window.addEventListener('hashchange', function () { openTo(location.hash.slice(1)); var el = document.getElementById(location.hash.slice(1)); if (el) el.scrollIntoView(); });
  if (location.hash) { openTo(location.hash.slice(1)); }
  var toc = document.querySelector('.toc'), hero = document.querySelector('.hero');
  if (toc && hero && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (es) { toc.classList.toggle('show', !es[0].isIntersecting); }, { rootMargin: '-80px 0px 0px 0px' }).observe(hero);
  } else if (toc) { toc.classList.add('show'); }
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a, .topbar nav a'));
  var nav = document.querySelector('.topbar nav');
  if (!('IntersectionObserver' in window)) return;
  var current = null;
  function reveal() {                       // keep the active exposé visible in a scrolling top bar (phones)
    var a = nav && nav.querySelector('a.active');
    if (!a || nav.scrollWidth <= nav.clientWidth + 1) return;
    var nr = nav.getBoundingClientRect(), ar = a.getBoundingClientRect();
    if (ar.left >= nr.left + 16 && ar.right <= nr.right - 28) return;
    var left = nav.scrollLeft + (ar.left - nr.left) - (nr.width - ar.width) / 2;
    try { nav.scrollTo({ left: left, behavior: 'smooth' }); } catch (e) { nav.scrollLeft = left; }
  }
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { if (e.isIntersecting) current = e.target.id; });
    links.forEach(function (a) {
      var on = a.getAttribute('href') === '#' + current;
      a.classList.toggle('active', on);
      if (on) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
    });
    reveal();
  }, { rootMargin: '-35% 0px -60% 0px' });
  document.querySelectorAll('section.expose').forEach(function (s) { io.observe(s); });
})();"""


def read(p):
    return open(p, encoding='utf-8').read()


def sections():
    files = sorted(glob.glob(os.path.join(SITE, 'sections', '[0-9][0-9]-*.html')))
    if not files:
        sys.exit('no sections found')
    hero, body = '', []
    for f in files:
        txt = read(f).strip()
        if os.path.basename(f).startswith('00-'):
            hero = txt
        else:
            body.append(txt)
    return hero, body


def toc(body):
    items = []
    for s in body:
        m = re.search(r'<section[^>]*\bid="([^"]+)"[^>]*>', s)
        if not m:
            continue
        tag = m.group(0)
        num = re.search(r'data-toc="([^"]*)"', tag)
        tt = re.search(r'data-toc-title="([^"]*)"', tag)
        if not tt:
            continue
        title = TOC_TITLE.get(m.group(1), tt.group(1))
        items.append(f'<li><a href="#{m.group(1)}"><span class="n">{num.group(1) if num else ""}</span><span>{title}</span></a></li>')
    return '<nav class="toc" aria-label="Exposés"><ol>' + ''.join(items) + '</ol></nav>'


def figure_scripts():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(SITE, 'js', 'fig-*.js')))


def topbar(ids):
    missing = [i for i in ids if i not in dict(NAV) and i != 'abstract']
    if missing:
        print('note: exposés without a top-bar label: ' + ', '.join(sorted(missing)))
    links = ''.join(f'<a href="#{i}">{t}</a>' for i, t in NAV if i in ids)
    return (f'<a class="skip-link" href="#main">Skip to the exposés</a>'
            f'<header class="topbar"><a class="brand" href="#top">A Categorical Atlas of <b>PEFT</b></a>'
            f'<nav aria-label="Sections">{links}</nav>'
            f'<button class="theme-toggle" type="button" data-theme-toggle>Dark</button></header>')


FOOT = ('<footer class="site-foot"><div class="inner">'
        '<span>A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges · Xiaoyu Li, Zhizhou Sha, Chiwun Yang, Dai Shi · 2026</span>'
        '<span>A static page. Every number in a figure is computed in your browser.</span>'
        '</div></footer>')


def build():
    hero, body = sections()
    ids = set(re.findall(r'<section[^>]*\bid="([^"]+)"', '\n'.join(body)))   # top-bar links go to sections only
    main = '\n'.join(body)
    figs = figure_scripts()
    page_body = (topbar(ids) + '\n' + toc(body) + '\n' + hero + '\n<main class="page" id="main"><div class="atlas">\n'
                 + main + '\n</div></main>\n' + FOOT)

    # ---------- multi-file index.html ----------
    root = SITE_URL.rstrip('/') + '/' if SITE_URL else ''
    card = root + 'assets/card.png'
    url_meta = f'<meta property="og:url" content="{html.escape(root)}">\n<link rel="canonical" href="{html.escape(root)}">\n' if root else ''
    if not root:
        print('note: og:image is relative (assets/card.png); set ATLAS_SITE_URL for the launch build so X can fetch the card')
    head = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{TITLE}</title>
<meta name="description" content="{html.escape(DESC)}">
<meta name="author" content="Xiaoyu Li, Zhizhou Sha, Chiwun Yang, Dai Shi">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{html.escape(DESC)}">
<meta property="og:type" content="website">
{url_meta}<meta property="og:image" content="{card}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{html.escape(CARD_ALT)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{html.escape(DESC)}">
<meta name="twitter:image" content="{card}">
<meta name="twitter:image:alt" content="{html.escape(CARD_ALT)}">
<meta name="theme-color" content="#f2f4f1" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0a121c" media="(prefers-color-scheme: dark)">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="css/atlas.css">
<script>{MJ_CONFIG}</script>
<script src="{MATHJAX}" defer></script>
<script src="{D3}"></script>"""
    scripts = '\n'.join(['<script src="data/atlas-data.js"></script>', '<script src="js/core.js"></script>']
                        + [f'<script src="js/{f}"></script>' for f in figs] + [f'<script>{SPY}</script>'])
    index = f'<!doctype html>\n<html lang="en">\n<head>\n{head}\n</head>\n<body id="top">\n{page_body}\n{scripts}\n</body>\n</html>\n'
    open(os.path.join(SITE, 'index.html'), 'w', encoding='utf-8').write(index)

    # ---------- single-file artifact ----------
    css = read(os.path.join(SITE, 'css', 'atlas.css'))
    inline_js = [read(os.path.join(SITE, 'data', 'atlas-data.js')), read(os.path.join(SITE, 'js', 'core.js'))]
    inline_js += [read(os.path.join(SITE, 'js', f)) for f in figs]
    def safe(js):
        return js.replace('</script', '<\\/script')
    art = (f'<title>{TITLE}</title>\n<link rel="stylesheet" href="{FONTS}">\n<style>\n{css}\n</style>\n'
           f'<script>{MJ_CONFIG}</script>\n<script src="{MATHJAX}" defer></script>\n<script src="{D3}"></script>\n'
           f'<div id="top"></div>\n{page_body}\n'
           + '\n'.join(f'<script>\n{safe(j)}\n</script>' for j in inline_js) + f'\n<script>{SPY}</script>\n')
    os.makedirs(DIST, exist_ok=True)
    open(os.path.join(DIST, 'artifact.html'), 'w', encoding='utf-8').write(art)
    print(f'index.html {len(index)//1024} KB; artifact.html {len(art)//1024} KB; {len(body)} sections; {len(figs)} figure scripts: {", ".join(figs)}')


if __name__ == '__main__':
    build()
