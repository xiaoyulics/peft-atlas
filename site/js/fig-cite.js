/* fig-cite.js — the unnumbered figure "Cite this atlas".
   (1) The BibTeX entry for the atlas itself (plus a plain-text form), with a Copy button.
   (2) Ready-to-paste bibliographies for every method and every paper the atlas catalogues, built from the
       `bibtex` field of window.ATLAS_DATA. They can be narrowed by a search, by HF PEFT availability, by year
       (click the chart) and by modification kind (click the legend); the button labels carry live counts.
       Records that share a citation key (several methods introduced by one paper, or a reading-list paper that
       also introduces a method) are written once, so the pasted file compiles without "Repeated entry" errors.
   (3) A stats strip. Every count is computed from ATLAS_DATA at mount time (arrays first, cross-checked
       against ATLAS_DATA.meta.counts; a mismatch is reported on the console).
   It also gives every <pre class="bib"> elsewhere on the page a Copy button.
   Clipboard: navigator.clipboard.writeText is called inside the click handler; if it rejects (or never settles,
   as in some sandboxes) the text is placed, selected, in a visible textarea. No file links. */
(function () {
  'use strict';

  var CSS_ID = 'css-cite';
  var F = '[data-figure="cite"] ';
  var TITLE = 'A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges';
  var NOTE = 'Technical report';
  var PH_AUTHORS = '[Authors]', PH_URL = '[URL]';

  var CSS = [
    /* header */
    F + '.ct-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem;margin:0 0 .2rem}',
    F + '.ct-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.55rem,3.6vw,2.15rem);line-height:1.08;letter-spacing:-.006em;margin:0;color:var(--ink)}',
    F + '.ct-title em{font-style:italic;color:var(--ochre)}',
    F + '.ct-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);font-variant-numeric:tabular-nums}',
    F + '.ct-kicker span{white-space:nowrap}',
    F + '.ct-instr{font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);margin:.25rem 0 .9rem;max-width:56rem}',
    /* layout */
    F + '.ct-top{display:grid;gap:.75rem;grid-template-columns:minmax(0,1fr);margin:0 0 .75rem}',
    '@media (min-width:860px){' + F + '.ct-top{grid-template-columns:minmax(0,1.3fr) minmax(0,1fr)}' + F + '.ct-stat{align-content:center}}',
    F + '.ct-panel{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.75rem .85rem;min-width:0}',
    F + '.ct-ph{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.45rem .8rem;margin:0 0 .6rem}',
    F + '.label{letter-spacing:.06em;color:var(--ink-2)}',
    F + '.ct-ph .label{letter-spacing:.06em}',
    F + '.ct-ph .label b{font-weight:600;color:var(--ink)}',
    F + '.ct-ph .label code{background:none;padding:0;font-size:1em;font-weight:400;text-transform:none;letter-spacing:0;color:var(--ink)}',
    F + '.ct-acts{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center}',
    F + '.ct-acts .seg button{padding:.32rem .66rem}',
    '@media (max-width:520px){' + F + '.stage{padding:.9rem .8rem .85rem}' + F + '.ct-panel{padding:.6rem .55rem}' + F + '.ct-lsep{display:none}}',
    /* bibtex blocks (hanging indent so wrapped lines stay readable at phone width) */
    F + '.ct-bib{margin:0;padding:.7rem .8rem;background:var(--paper-2);border:1px solid var(--rule);border-radius:4px;font-family:var(--f-mono);font-size:.8rem;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--ink)}',
    F + '.ct-bib .bl{display:block;padding-left:4ch;text-indent:-4ch}',
    F + '.ct-bib .bl.gap{height:.8em}',
    F + '.ct-bib .t{color:var(--tide)}',
    F + '.ct-bib .k{color:var(--ink);font-weight:500}',
    F + '.ct-bib .f{color:var(--ink-2)}',
    F + '.ct-bib .p{color:var(--ink-3)}',
    F + '.ct-bib .cmd{color:var(--tide)}',
    F + '.ct-bib .c{color:var(--ink-2);font-style:italic}',
    F + '.ct-bib mark.ph{background:var(--ochre-soft);color:var(--ochre-ink);border-radius:2px;padding:0 .12em;box-shadow:inset 0 -1px 0 currentColor}',
    F + '.ct-bib.ct-txt{font-family:var(--f-body);font-size:.98rem;line-height:1.55;padding:.85rem .95rem}',
    F + '.ct-bib.ct-txt i{font-style:italic}',
    F + '.ct-self .ct-bib{font-size:.84rem}',
    F + '.ct-foot{margin:.5rem 0 0;font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
    F + '.ct-foot mark{background:var(--ochre-soft);color:var(--ochre-ink);font-family:var(--f-mono);font-size:.78rem;padding:0 .15em;border-radius:2px}',
    /* stats */
    F + '.ct-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);min-width:0;margin:0;padding:0;list-style:none}',
    F + '.ct-stat{padding:.8rem .9rem .75rem;display:grid;gap:.12rem;align-content:start;min-width:0;border:0 solid var(--rule);margin:0}',
    F + '.ct-stat:nth-child(odd){border-right-width:1px}',
    F + '.ct-stat:nth-child(-n+2){border-bottom-width:1px}',
    F + '.ct-stat .v{font-family:var(--f-mono);font-weight:500;font-size:clamp(1.6rem,3.2vw,2.15rem);line-height:1.05;letter-spacing:-.03em;color:var(--ink);font-variant-numeric:tabular-nums}',
    F + '.ct-stat .k{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    F + '.ct-stat .s{font-family:var(--f-ui);font-size:.8rem;line-height:1.4;color:var(--ink-2);font-variant-numeric:tabular-nums;margin-top:.15rem}',
    F + '.ct-stat .s a{color:inherit;text-decoration-color:var(--rule)}',
    F + '.ct-stat .s a:hover{color:var(--tide)}',
    /* filters */
    F + '.ct-filters{display:flex;flex-wrap:wrap;gap:.6rem 1.1rem;align-items:flex-end;margin:0 0 .75rem}',
    F + '.ct-fld{display:grid;gap:.28rem;min-width:0}',
    F + '.ct-fld.grow{flex:1 1 15rem;max-width:26rem}',
    F + '.ct-fld input[type="search"]{width:100%;font-family:var(--f-ui);font-size:.875rem;padding:.4rem .6rem;text-overflow:ellipsis}',
    F + '.ct-fld .seg button{padding:.36rem .7rem}',
    F + '.ct-clear{padding:.38rem .7rem}',
    F + '.ct-clear:disabled{opacity:.4;cursor:default}',
    F + '.ct-clear:disabled:hover{border-color:var(--ink-3);color:var(--ink)}',
    /* chart */
    F + '.ct-cap{display:flex;flex-wrap:wrap;justify-content:space-between;gap:.2rem 1rem;font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2);margin:0 0 .2rem}',
    F + '.ct-cap .label{letter-spacing:.06em}',
    F + '.ct-chart{position:relative;margin:0 0 .3rem;min-width:0}',
    F + '.ct-chart svg{display:block;width:100%;height:auto;overflow:visible}',
    F + '.ct-chart svg text{font-family:var(--f-mono);font-size:12px;font-variant-numeric:tabular-nums}',
    F + '.ct-bin{cursor:pointer;outline:none}',
    F + '.ct-bin .hit{fill:transparent;stroke:none}',
    F + '.ct-bin:hover .hit{fill:var(--tide-soft)}',
    F + '.ct-bin:focus-visible .hit{stroke:var(--ochre);stroke-width:1.5px}',
    F + '.ct-bin .ysel{fill:transparent}',
    F + '.ct-bin.on .ysel{fill:var(--ochre-soft)}',
    F + '.ct-yl{fill:var(--ink-2)}',
    F + '.ct-bin.on .ct-yl{fill:var(--ochre-ink);font-weight:500}',
    F + '.ct-cl{fill:var(--ink)}',
    F + '.ct-cl.zero{fill:var(--ink-2)}',
    F + '.ct-axis{stroke:var(--ink-3);stroke-width:.8px}',
    /* legend */
    F + '.ct-legend{display:flex;flex-wrap:wrap;gap:.3rem .35rem;align-items:center;margin:0 0 .95rem}',
    F + '.ct-kind{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--f-ui);font-weight:500;font-size:.78rem;letter-spacing:.01em;color:var(--ink-2);background:transparent;border:1px solid var(--rule);border-radius:999px;padding:.2rem .55rem .2rem .45rem;cursor:pointer;line-height:1.3}',
    F + '.ct-kind:hover{border-color:var(--ink-3);color:var(--ink)}',
    F + '.ct-kind .n{font-family:var(--f-mono);font-weight:400;font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
    F + '.ct-kind[aria-pressed="true"]{border-color:var(--ink-2);color:var(--ink);background:var(--paper-2)}',
    F + '.ct-legend.picking .ct-kind[aria-pressed="false"]{opacity:.6}',
    F + '.ct-sw{display:inline-block;width:.75rem;height:.75rem;border-radius:2px;flex:none}',
    F + '.ct-sw.hatch{background:repeating-linear-gradient(45deg,var(--ink-2) 0 1px,transparent 1px 3.4px);box-shadow:inset 0 0 0 1px var(--ink-2)}',
    F + '.ct-sw.pale{background:color-mix(in srgb,var(--ink-3) 24%,transparent)}',
    F + '.ct-sw.ysw{background:var(--ochre-soft);box-shadow:inset 0 -2px 0 var(--ochre)}',
    F + '.ct-li{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--f-ui);font-weight:500;font-size:.78rem;color:var(--ink-2);padding:.2rem .35rem;white-space:nowrap}',
    F + '.ct-li[hidden]{display:none}',
    F + '.ct-lsep{width:1px;align-self:stretch;background:var(--rule);margin:0 .25rem}',
    /* selection summary + export rows */
    F + '.ct-sum{font-family:var(--f-ui);font-size:.8rem;line-height:1.45;color:var(--ink-2);font-variant-numeric:tabular-nums;margin:0 0 .55rem;padding:.4rem 0 0;border-top:1px solid var(--rule)}',
    F + '.ct-sum b{font-weight:600;color:var(--ink)}',
    F + '.ct-exports{display:grid;gap:.5rem;margin:0 0 .55rem}',
    F + '.ct-row{display:grid;grid-template-columns:minmax(0,1fr);gap:.28rem .95rem;align-items:center}',
    '@media (min-width:700px){' + F + '.ct-row{grid-template-columns:minmax(17.5rem,21.5rem) minmax(0,1fr)}}',
    F + '.ct-go{text-transform:none;letter-spacing:.01em;font-size:.8rem;text-align:left;padding:.55rem .8rem;width:100%;line-height:1.35}',
    F + '.ct-go:disabled{opacity:.45;cursor:not-allowed}',
    F + '.ct-go:disabled:hover{border-color:var(--ink-3);color:var(--ink)}',
    F + '.ct-go.primary:disabled:hover{color:var(--paper);border-color:var(--tide)}',
    F + '.ct-ro{font-family:var(--f-ui);font-size:.8rem;line-height:1.45;color:var(--ink-2);font-variant-numeric:tabular-nums;min-width:0}',
    F + '.ct-ro b{font-weight:600;color:var(--ink)}',
    F + '.ct-note{font-family:var(--f-body);font-size:.84rem;line-height:1.5;color:var(--ink-2);margin:.15rem 0 0}',
    F + '.ct-note code{font-size:.9em}',
    /* preview */
    F + '.ct-prev-h{display:flex;flex-wrap:wrap;gap:.45rem .8rem;align-items:center;justify-content:space-between;margin:1rem 0 .45rem;padding-top:.75rem;border-top:1px solid var(--rule)}',
    F + '.ct-prev-h .ct-ro{flex:1 1 auto;text-align:right}',
    F + '.ct-prev{max-height:17rem;overflow:auto;font-size:.75rem;line-height:1.55}',
    '@media (max-width:520px){' + F + '.ct-prev-h .ct-ro{text-align:left;flex-basis:100%}' + F + '.ct-prev{max-height:14rem}}',
    /* copied feedback + clipboard fallback (also used by enhanced <pre class="bib"> outside the figure) */
    '.btn.ct-cp.is-copied,.btn.ct-cp.is-copied:hover{border-color:var(--moss);color:var(--moss);background:var(--moss-soft)}',
    '.btn.ct-cp.primary.is-copied,.btn.ct-cp.primary.is-copied:hover{border-color:var(--moss);color:var(--paper);background:var(--moss)}',
    '.btn.ct-cp.is-selected{border-color:var(--ochre);color:var(--ochre)}',
    '.ct-fb{display:grid;gap:.45rem;margin:.6rem 0;padding:.6rem .7rem;border:1px solid var(--ochre);background:var(--ochre-soft);border-radius:var(--radius)}',
    '.ct-fb[hidden]{display:none}',
    '.ct-fb-row{display:flex;gap:.6rem;align-items:flex-start;justify-content:space-between}',
    '.ct-fb-msg{margin:0;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.45;color:var(--ink)}',
    '.ct-fb-ta{display:block;width:100%;min-height:8.5rem;font-family:var(--f-mono);font-size:.78rem;line-height:1.5;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:4px;padding:.5rem;resize:vertical}',
    '.cite-box .ct-fb .btn{position:static;background:transparent}',
    '.cite-box.ct-enh>.btn{z-index:1}',
    '.cite-box.ct-enh>pre.bib{padding-right:5.8rem}',
    '@media (max-width:520px){.cite-box.ct-enh>pre.bib{padding-right:1.1rem;padding-top:2.8rem}}'
  ].join('\n');

  function injectCSS() {
    if (document.getElementById(CSS_ID)) return;
    var s = document.createElement('style'); s.id = CSS_ID; s.textContent = CSS; document.head.appendChild(s);
  }

  /* ---------------- pure helpers ---------------- */
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function pad(s, n) { s = String(s); while (s.length < n) s += ' '; return s; }
  function utf8Len(s) {
    if (window.TextEncoder) return new TextEncoder().encode(s).length;
    return unescape(encodeURIComponent(s)).length;
  }
  function fmtKB(bytes) { return (bytes / 1024).toFixed(1) + ' KB'; }
  function fold(s) {
    s = String(s == null ? '' : s).toLowerCase();
    return s.normalize ? s.normalize('NFD').replace(/[̀-ͯ]/g, '') : s;
  }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }
  function keyOf(x) {
    if (x && x.bibkey) return String(x.bibkey);
    var m = /^@\w+\s*\{\s*([^,\s]+)/.exec(String((x && x.bibtex) || ''));
    return m ? m[1] : String((x && x.bibtex) || '');
  }
  function bibOf(x) { return String((x && x.bibtex) || '').trim(); }
  function cmpKey(a, b) { var x = a.toLowerCase(), y = b.toLowerCase(); return x < y ? -1 : x > y ? 1 : 0; }

  /* De-duplicate by citation key: the first record with a key wins (callers pass methods before papers),
     then entries are sorted by key. `dropped` = records whose key had already been written. */
  function bundle(list) {
    var seen = {}, rows = [], groups = {}, n = 0, missing = 0;
    list.forEach(function (x) {
      var b = bibOf(x);
      if (!b) { missing++; return; }
      n++;
      var k = keyOf(x);
      (groups[k] = groups[k] || []).push(x);
      if (seen[k]) return;
      seen[k] = 1; rows.push({ k: k, b: b });
    });
    rows.sort(function (a, b) { return cmpKey(a.k, b.k); });
    var shared = rows.filter(function (r) { return groups[r.k].length > 1; }).map(function (r) { return r.k; });
    return { entries: rows.map(function (r) { return r.b; }), keys: rows.map(function (r) { return r.k; }), n: n, missing: missing, dropped: n - rows.length, shared: shared, groups: groups };
  }

  /* ---------------- the atlas entry ---------------- */
  /* The address to cite: a canonical link or og:url if the page declares one, else the page's own
     address when it is served from a public host. Not from file://, a local server, or inside a frame
     (an embedded copy's address is not the page's), where the placeholder stays for the author to fill. */
  function pageURL() {
    try {
      var c = document.querySelector('link[rel="canonical"]');
      if (c && /^https?:/.test(c.href)) return c.href;
      var og = document.querySelector('meta[property="og:url"]');
      if (og && /^https?:/.test(og.content || '')) return og.content;
      var loc = window.location, host = loc.hostname || '';
      if (!/^https?:$/.test(loc.protocol) || window.top !== window.self) return '';
      if (!host || /^(localhost|127\.|0\.0\.0\.0|\[::1\]|::1$)/.test(host) || /\.(localhost|test|local)$/.test(host)) return '';
      return loc.origin + loc.pathname.replace(/index\.html?$/, '');
    } catch (e) { return ''; }
  }
  function atlasCfg(el, D) {
    var meta = (D && D.meta) || {};
    var m = /^(\d{4})/.exec(String(meta.built || ''));
    var year = m ? m[1] : '2026';
    var au = el.getAttribute('data-authors') || meta.authors || PH_AUTHORS;
    if (Array.isArray(au)) au = au.join(' and ');
    var url = el.getAttribute('data-url') || meta.url || pageURL() || PH_URL;
    return { key: 'li' + year + 'categorical', year: year, authors: String(au), url: String(url) };
  }
  function atlasBib(c) {
    /* title in double braces so BibTeX styles keep its capitalisation (as every entry in the catalogue does) */
    var f = [['title', '{' + TITLE + '}'], ['author', c.authors], ['year', c.year], ['howpublished', '\\url{' + c.url + '}'], ['note', NOTE]];
    var w = 0; f.forEach(function (p) { w = Math.max(w, p[0].length); });
    return '@misc{' + c.key + ',\n' + f.map(function (p, i) {
      return '  ' + pad(p[0], w) + ' = {' + p[1] + '}' + (i < f.length - 1 ? ',' : '');
    }).join('\n') + '\n}\n';
  }
  function atlasText(c) { return c.authors + ' (' + c.year + '). ' + TITLE + '. ' + NOTE + '. ' + c.url; }

  /* ---------------- BibTeX highlighter (escapes everything it does not wrap) ---------------- */
  var TOK = /(\[Authors\]|\[URL\]|\\[A-Za-z]+|\\.|[{}])/g;
  function hlValue(v, ph) {
    var out = '', last = 0, m;
    TOK.lastIndex = 0;
    while ((m = TOK.exec(v))) {
      out += esc(v.slice(last, m.index));
      var t = m[0];
      if (t === PH_AUTHORS || t === PH_URL) out += ph ? '<mark class="ph" title="placeholder">' + esc(t) + '</mark>' : esc(t);
      else if (t === '{' || t === '}') out += '<span class="p">' + t + '</span>';
      else out += '<span class="cmd">' + esc(t) + '</span>';
      last = m.index + t.length;
    }
    return out + esc(v.slice(last));
  }
  function hlBib(text, ph) {
    return String(text).replace(/\n+$/, '').split('\n').map(function (l) {
      var m;
      if (!l.trim()) return '<span class="bl gap"></span>';
      if (l.charAt(0) === '%') return '<span class="bl c">' + esc(l) + '</span>';
      if ((m = /^@(\w+)(\s*\{\s*)([^,]*)(,?)(.*)$/.exec(l))) {
        return '<span class="bl"><span class="t">@' + esc(m[1]) + '</span><span class="p">' + esc(m[2]) + '</span><span class="k">' + esc(m[3]) + '</span><span class="p">' + esc(m[4]) + '</span>' + esc(m[5]) + '</span>';
      }
      if ((m = /^(\s+)([\w-]+)(\s*=\s*)(.*)$/.exec(l))) {
        return '<span class="bl">' + m[1] + '<span class="f">' + esc(m[2]) + '</span><span class="p">' + esc(m[3]) + '</span>' + hlValue(m[4], ph) + '</span>';
      }
      return '<span class="bl">' + hlValue(l, ph) + '</span>';
    }).join('');
  }

  /* ---------------- clipboard ---------------- */
  /* Must be called synchronously from the click handler (user activation). `ok` on success; `fail` if the
     Clipboard API is missing, throws, rejects, or does not settle within 1.5 s (some sandboxes). */
  function copyText(text, ok, fail) {
    var done = false, timer = null;
    function win() { if (done) return; done = true; clearTimeout(timer); ok(); }
    function lose() { if (done) return; done = true; clearTimeout(timer); fail(); }
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        timer = setTimeout(lose, 1500);
        navigator.clipboard.writeText(text).then(win, lose);
        return;
      }
    } catch (e) { /* fall through */ }
    lose();
  }
  function setLabel(btn, s) { btn.__label = s; if (!btn.__flash) btn.textContent = s; }
  function flash(btn, msg, cls, ms) {
    if (btn.__label == null) btn.__label = btn.textContent;
    btn.__flash = true;
    btn.classList.remove('is-copied', 'is-selected');
    btn.classList.add(cls || 'is-copied');
    btn.textContent = msg;
    clearTimeout(btn.__t);
    btn.__t = setTimeout(function () {
      btn.__flash = false; btn.classList.remove('is-copied', 'is-selected'); btn.textContent = btn.__label;
    }, ms || 1800);
  }
  /* a visible, selected textarea; `show` returns true when execCommand('copy') also succeeded */
  function makeFallback(H) {
    var msg = H('p', { class: 'ct-fb-msg', role: 'status' });
    var done = H('button', { type: 'button', class: 'btn', text: 'Done' });
    var ta = H('textarea', { class: 'ct-fb-ta', readonly: 'readonly', rows: '8', spellcheck: 'false', 'aria-label': 'Text to copy' });
    var box = H('div', { class: 'ct-fb', hidden: 'hidden' }, [H('div', { class: 'ct-fb-row' }, [msg, done]), ta]);
    done.addEventListener('click', function () { box.hidden = true; if (box.__ret && box.__ret.focus) box.__ret.focus(); });
    return {
      el: box,
      show: function (text, what, ret) {
        ta.value = text;
        box.hidden = false;
        box.__ret = ret;
        try { ta.focus({ preventScroll: false }); } catch (e) { ta.focus(); }
        ta.select();
        try { ta.setSelectionRange(0, text.length); } catch (e) { /* ignore */ }
        var ok = false;
        try { ok = !!(document.execCommand && document.execCommand('copy')); } catch (e) { ok = false; }
        msg.textContent = ok
          ? 'Copied ' + what + ' through the selection below, because the clipboard API was unavailable.'
          : 'The clipboard is blocked here, so the text (' + what + ') is selected below: press ⌘C or Ctrl+C.';
        return ok;
      },
      hide: function () { box.hidden = true; }
    };
  }
  /* copy with feedback on `btn`, falling back to `fb` (a makeFallback instance) */
  function copyWithFeedback(text, btn, fb, what, announce, okMsg) {
    okMsg = okMsg || (btn.__short ? 'Copied' : 'Copied ✓');
    copyText(text, function () {
      fb.hide();
      flash(btn, okMsg, 'is-copied');
      if (announce) announce('Copied ' + what + ' to the clipboard.');
    }, function () {
      var ok = fb.show(text, what, btn);
      if (ok) { flash(btn, okMsg, 'is-copied'); if (announce) announce('Copied ' + what + '.'); }
      else { flash(btn, btn.__short ? 'Selected' : 'Selected: press ⌘C', 'is-selected', 3200); if (announce) announce('Clipboard blocked; ' + what + ' selected in a text box.'); }
    });
  }

  /* ---------------- Copy buttons on every <pre class="bib"> elsewhere on the page ---------------- */
  function enhanceBibs() {
    var A = window.Atlas;
    if (!A || !A.h) return;
    var pres = document.querySelectorAll('pre.bib');
    if (!pres.length) return;
    injectCSS();
    Array.prototype.forEach.call(pres, function (pre) {
      if (pre.__ctEnh || (pre.closest && pre.closest('[data-figure="cite"]'))) return;
      pre.__ctEnh = true;
      var box = pre.parentNode;
      if (!box) return;
      if (!(box.classList && box.classList.contains('cite-box'))) {
        var wrap = A.h('div', { class: 'cite-box' });
        box.insertBefore(wrap, pre); wrap.appendChild(pre); box = wrap;
      }
      var existing = null;
      Array.prototype.forEach.call(box.querySelectorAll('button'), function (b) { if (!existing) existing = b; });
      if (existing && !existing.hasAttribute('data-cite-copy')) return;   /* the page already wires its own button */
      box.classList.add('ct-enh');
      var key = (/^\s*@\w+\s*\{\s*([^,\s]+)/.exec(pre.textContent) || [])[1] || 'this entry';
      var btn = existing || A.h('button', { type: 'button', class: 'btn' });
      btn.classList.add('ct-cp');
      btn.__short = true;
      if (!existing) { btn.textContent = 'Copy'; box.insertBefore(btn, pre); }
      btn.__label = btn.textContent;
      btn.setAttribute('aria-label', 'Copy the BibTeX for ' + key);
      var fb = null;
      btn.addEventListener('click', function () {
        if (!fb) { fb = makeFallback(A.h); box.appendChild(fb.el); }
        copyWithFeedback(pre.textContent.replace(/\s+$/, '') + '\n', btn, fb, 'the BibTeX for ' + key);
      });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', enhanceBibs);
  else enhanceBibs();
  document.addEventListener('atlas:ready', enhanceBibs);

  /* ================================ the figure ================================ */
  Atlas.register('cite', function (el, A) {
    injectCSS();
    enhanceBibs();
    var H = A.h;
    var D = A.data() || {};
    var M = D.methods || [], P = D.papers || [], PR = D.propositions || [];
    var meta = D.meta || {}, mc = meta.counts || {};
    var built = String(meta.built || '');
    var cfg = atlasCfg(el, D);
    var hasPH = cfg.authors === PH_AUTHORS || cfg.url === PH_URL;

    /* ---------- counts: from the arrays, cross-checked against meta.counts ---------- */
    function isHF(m) { return m.hf_peft === true; }
    var C = {
      methods: M.length ? M.length : (+mc.methods || 0),
      papers: P.length ? P.length : (+mc.papers || 0),
      hf: M.length ? M.filter(isHF).length : (+mc.hf_peft || 0),
      props: PR.length ? PR.length : (+mc.propositions || 0)
    };
    [['methods', 'methods'], ['papers', 'papers'], ['hf', 'hf_peft'], ['props', 'propositions']].forEach(function (p) {
      if (mc[p[1]] != null && +mc[p[1]] !== C[p[0]]) console.warn('[cite] meta.counts.' + p[1] + ' = ' + mc[p[1]] + ' but the data hold ' + C[p[0]] + '; showing the data count.');
    });
    function span(list) {
      var ys = list.map(function (x) { return +x.year; }).filter(function (y) { return isFinite(y) && y > 0; });
      if (!ys.length) return '';
      var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
      return lo === hi ? String(lo) : lo + '–' + hi;
    }
    var statusCount = {};
    PR.forEach(function (p) { var s = String(p.status || 'unlabelled'); statusCount[s] = (statusCount[s] || 0) + 1; });

    /* ---------- year bins: the ten latest years, plus one bin for everything earlier ---------- */
    var allYears = M.concat(P).map(function (x) { return +x.year; }).filter(function (y) { return isFinite(y) && y > 0; });
    var maxY = allYears.length ? Math.max.apply(null, allYears) : 2026;
    var minY = allYears.length ? Math.min.apply(null, allYears) : maxY;
    var loY = maxY - 9;
    var BINS = [];
    if (minY < loY) BINS.push({ key: 'lt', lab: '≤' + (loY - 1), short: '≤’' + String(loY - 1).slice(2), long: (minY === loY - 1 ? String(minY) : minY + '–' + (loY - 1)) });
    for (var yy = loY; yy <= maxY; yy++) BINS.push({ key: String(yy), lab: String(yy), short: '’' + String(yy).slice(2), long: String(yy) });
    var binIdx = {}; BINS.forEach(function (b, i) { binIdx[b.key] = i; });
    function binKey(y) { y = +y; if (!isFinite(y) || y <= 0) return null; if (y < loY) return BINS[0] && BINS[0].key === 'lt' ? 'lt' : null; return String(y); }

    var KINDS = A.KINDS.map(function (k) { return k.key; });
    var kindPos = {}; KINDS.forEach(function (k, i) { kindPos[k] = i; });
    function kindOf(m) { return kindPos[m.modification_kind] != null ? m.modification_kind : 'hybrid'; }

    var MX = M.map(function (m) {
      return { x: m, bin: binKey(m.year), kind: kindOf(m), hf: isHF(m),
        hay: fold([m.name, m.full_name, m.id, m.authors, (m.authors_full || []).join(' '), m.paper_title, m.bibkey, m.year, m.venue, (m.families || []).join(' '), m.arxiv].join(' | ')) };
    });
    var PX = P.map(function (p) {
      return { x: p, bin: binKey(p.year),
        hay: fold([p.title, p.id, p.authors, (p.authors_full || []).join(' '), p.bibkey, p.year, p.venue, (p.families || []).join(' '), p.arxiv].join(' | ')) };
    });
    var totM = BINS.map(function () { return 0; }), totP = BINS.map(function () { return 0; });
    MX.forEach(function (r) { if (r.bin != null) totM[binIdx[r.bin]]++; });
    PX.forEach(function (r) { if (r.bin != null) totP[binIdx[r.bin]]++; });

    /* ---------- state ---------- */
    var st = { q: '', hf: 'all', kinds: {}, years: {}, fmt: 'bibtex', prev: 'methods' };
    function anyKinds() { for (var k in st.kinds) if (st.kinds[k]) return true; return false; }
    function anyYears() { for (var k in st.years) if (st.years[k]) return true; return false; }
    function toks() { return fold(st.q).split(/\s+/).filter(Boolean); }
    function hit(hay, t) { for (var i = 0; i < t.length; i++) if (hay.indexOf(t[i]) < 0) return false; return true; }
    function mFilterOn() { return toks().length > 0 || st.hf === 'hf' || anyKinds() || anyYears(); }
    function pFilterOn() { return toks().length > 0 || anyYears(); }
    var S = null;   /* current selection */
    function select() {
      var t = toks(), ky = anyKinds(), yr = anyYears();
      var m = [], mNoKind = [], p = [];
      MX.forEach(function (r) {
        if (st.hf === 'hf' && !r.hf) return;
        if (yr && !(r.bin && st.years[r.bin])) return;
        if (t.length && !hit(r.hay, t)) return;
        mNoKind.push(r);
        if (ky && !st.kinds[r.kind]) return;
        m.push(r);
      });
      PX.forEach(function (r) {
        if (yr && !(r.bin && st.years[r.bin])) return;
        if (t.length && !hit(r.hay, t)) return;
        p.push(r);
      });
      var selM = BINS.map(function () { return KINDS.map(function () { return 0; }); }), selP = BINS.map(function () { return 0; });
      m.forEach(function (r) { if (r.bin != null) selM[binIdx[r.bin]][kindPos[r.kind]]++; });
      p.forEach(function (r) { if (r.bin != null) selP[binIdx[r.bin]]++; });
      var kindN = {}; mNoKind.forEach(function (r) { kindN[r.kind] = (kindN[r.kind] || 0) + 1; });
      var ms = m.map(function (r) { return r.x; }), ps = p.map(function (r) { return r.x; });
      return { m: ms, p: ps, kindN: kindN, selM: selM, selP: selP,
        bm: bundle(ms), bp: bundle(ps), ball: bundle(ms.concat(ps)) };
    }
    function describe(scope) {
      var parts = [], t = st.q.trim(), methodsToo = scope !== 'papers';
      if (t) parts.push('matching "' + t.replace(/[@{}%\\"]/g, '') + '"');
      if (methodsToo && st.hf === 'hf') parts.push('in HF PEFT');
      if (methodsToo && anyKinds()) parts.push('kind: ' + KINDS.filter(function (k) { return st.kinds[k]; }).map(function (k) { return A.kindName(k); }).join(', '));
      if (anyYears()) parts.push('year: ' + BINS.filter(function (b) { return st.years[b.key]; }).map(function (b) { return b.long; }).join(', '));
      return parts.join('; ');
    }
    function scopeText(sel, total, noun) { return (sel === total ? 'all ' + total : sel + ' of ' + total) + ' ' + noun; }
    function fileText(which) {
      var b = which === 'methods' ? S.bm : which === 'papers' ? S.bp : S.ball;
      if (!b.entries.length) return '';
      var what = which === 'methods' ? scopeText(S.m.length, C.methods, 'methods')
        : which === 'papers' ? scopeText(S.p.length, C.papers, 'papers')
        : scopeText(S.m.length, C.methods, 'methods') + ' and ' + scopeText(S.p.length, C.papers, 'papers');
      var desc = describe(which);
      var head = '% A Categorical Atlas of Parameter-Efficient Fine-Tuning' + (built ? ' (atlas data built ' + built + ')' : '') + '\n'
        + '% BibTeX for ' + what + (desc ? ' (' + desc + ')' : '') + ': ' + plural(b.entries.length, 'entry', 'entries')
        + (b.dropped ? ', ' + plural(b.dropped, 'repeated key') + ' merged' : '') + '.\n\n';
      return head + b.entries.join('\n\n') + '\n';
    }

    /* ================= DOM ================= */
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Cite this atlas. The BibTeX entry for the atlas, and copyable bibliographies of the ' + C.methods + ' methods and ' + C.papers + ' papers it catalogues.');
    var stage = H('div', { class: 'stage mathjax_ignore' });   /* no mathematics here; keep MathJax out */
    el.appendChild(stage);
    var live = H('div', { class: 'sr-only', 'aria-live': 'polite' });
    function announce(s) { live.textContent = ''; setTimeout(function () { live.textContent = s; }, 30); }

    stage.appendChild(H('div', { class: 'ct-head' }, [
      H('h3', { class: 'ct-title', html: 'Cite this <em>atlas</em>' }),
      H('span', { class: 'ct-kicker' }, [H('span', { text: 'BibTeX' }), built ? ' · ' : null, built ? H('span', { text: 'data built ' + built }) : null])
    ]));
    stage.appendChild(H('p', { class: 'ct-instr', text: 'Copy the entry for this atlas, or a ready-to-paste bibliography of the ' + C.methods + ' methods and ' + C.papers + ' papers it maps. Search, or pick a year or a kind to narrow it; each button says how many entries it copies.' }));

    /* ----- (1) the atlas entry ----- */
    var top = H('div', { class: 'ct-top' });
    stage.appendChild(top);
    var selfId = 'ct-self-' + Math.random().toString(36).slice(2, 7);
    var selfPanel = H('section', { class: 'ct-panel ct-self', 'aria-labelledby': selfId });
    var selfPre = H('pre', { class: 'ct-bib mathjax_ignore', tabindex: '0', 'aria-label': 'Citation for this atlas' });
    var selfBtn = H('button', { type: 'button', class: 'btn primary ct-cp', text: 'Copy', 'aria-label': 'Copy the citation for this atlas' });
    selfBtn.__label = 'Copy';
    var fmtSeg = A.seg([{ value: 'bibtex', label: 'BibTeX' }, { value: 'text', label: 'Plain text' }], st.fmt, function (v) { st.fmt = v; renderSelf(); }, 'Citation format');
    selfPanel.appendChild(H('div', { class: 'ct-ph' }, [
      H('span', { class: 'label', id: selfId, html: 'This atlas · <code>' + esc(cfg.key) + '</code>' }),
      H('div', { class: 'ct-acts' }, [fmtSeg.el, selfBtn])
    ]));
    selfPanel.appendChild(selfPre);
    if (hasPH) selfPanel.appendChild(H('p', { class: 'ct-foot', html: 'Fields marked like <mark>' + esc(PH_AUTHORS) + '</mark> are placeholders.' }));
    var selfFb = makeFallback(H);
    selfPanel.appendChild(selfFb.el);
    top.appendChild(selfPanel);
    function selfText() { return st.fmt === 'bibtex' ? atlasBib(cfg) : atlasText(cfg) + '\n'; }
    function renderSelf() {
      if (st.fmt === 'bibtex') { selfPre.classList.remove('ct-txt'); selfPre.innerHTML = hlBib(atlasBib(cfg), true); }
      else {
        selfPre.classList.add('ct-txt');
        var phm = function (s) { return (s === PH_AUTHORS || s === PH_URL) ? '<mark class="ph" title="placeholder">' + esc(s) + '</mark>' : esc(s); };
        selfPre.innerHTML = phm(cfg.authors) + ' (' + esc(cfg.year) + '). <i>' + esc(TITLE) + '</i>. ' + esc(NOTE) + '. ' + phm(cfg.url);
      }
    }
    selfBtn.addEventListener('click', function () {
      copyWithFeedback(selfText(), selfBtn, selfFb, st.fmt === 'bibtex' ? 'the BibTeX entry for this atlas' : 'the citation for this atlas', announce);
    });

    /* ----- (3) stats ----- */
    var hasLegend = !!document.getElementById('status-legend');
    var statusTxt = Object.keys(statusCount).sort(function (a, b) { return statusCount[a] - statusCount[b]; })
      .map(function (s) { return statusCount[s] + ' ' + s; }).join(' · ');
    var kindsUsed = {}; M.forEach(function (m) { kindsUsed[kindOf(m)] = 1; });
    var nKinds = Object.keys(kindsUsed).length;
    function stat(v, k, s) { return H('li', { class: 'ct-stat' }, [H('span', { class: 'v', text: String(v) }), H('span', { class: 'k', text: k }), s ? H('span', { class: 's', html: s }) : null]); }
    var stats = H('ul', { class: 'ct-stats', 'aria-label': 'What the atlas contains' }, [
      stat(C.methods, 'methods', esc(span(M)) + (nKinds ? ' · ' + plural(nKinds, 'kind') : '')),
      stat(C.papers, 'papers', esc(span(P))),
      stat(C.hf, 'in HF PEFT', C.methods ? (100 * C.hf / C.methods).toFixed(1) + '% of the methods' : ''),
      stat(C.props, 'numbered results', statusTxt ? (hasLegend ? '<a href="#status-legend">' + esc(statusTxt) + '</a>' : esc(statusTxt)) : '')
    ]);
    top.appendChild(stats);

    /* ----- (2) the bibliography ----- */
    var lit = H('section', { class: 'ct-panel ct-lit', 'aria-label': 'Bibliography of the catalogued methods and papers' });
    stage.appendChild(lit);
    lit.appendChild(H('div', { class: 'ct-ph' }, [
      H('span', { class: 'label', html: 'The literature it maps · <b>' + C.methods + ' methods, ' + C.papers + ' papers</b>' })
    ]));
    /* filters */
    var qId = 'ct-q-' + Math.random().toString(36).slice(2, 7);
    var qIn = H('input', { type: 'search', id: qId, placeholder: 'name, author, year or key', autocomplete: 'off', spellcheck: 'false' });
    var hfSeg = A.seg([{ value: 'all', label: 'All methods' }, { value: 'hf', label: 'In HF PEFT' }], st.hf, function (v) { st.hf = v; update(); }, 'Which methods');
    var clearBtn = H('button', { type: 'button', class: 'btn ct-clear', text: 'Clear filters' });
    lit.appendChild(H('div', { class: 'ct-filters' }, [
      H('div', { class: 'ct-fld grow' }, [H('label', { class: 'label', for: qId, text: 'Search methods and papers' }), qIn]),
      H('div', { class: 'ct-fld' }, [H('span', { class: 'label', text: 'Methods' }), hfSeg.el]),
      clearBtn
    ]));
    var debounced = A.debounce(function () { st.q = qIn.value; update(); }, 140);
    qIn.addEventListener('input', debounced);
    qIn.addEventListener('search', function () { st.q = qIn.value; update(); });
    clearBtn.addEventListener('click', function () {
      st.q = ''; qIn.value = ''; st.hf = 'all'; hfSeg.set('all'); st.kinds = {}; st.years = {};
      update(); announce('Filters cleared.');
      qIn.focus();
    });

    /* chart */
    lit.appendChild(H('div', { class: 'ct-cap' }, [
      H('span', { text: 'Entries by year of publication: methods above the line, stacked by kind; papers hatched below.' }),
      H('span', { class: 'label', text: 'click a year to filter' })
    ]));
    var chartBox = H('div', { class: 'ct-chart' });
    lit.appendChild(chartBox);
    var NS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('role', 'group');
    svg.setAttribute('aria-label', 'Catalogue entries per year; each year is a toggle that filters the bibliography');
    chartBox.appendChild(svg);
    var hatchId = 'ct-hatch-' + Math.random().toString(36).slice(2, 7);
    var focusIdx = BINS.length - 1, refocus = false;

    /* legend (kinds are toggles) */
    var legend = H('div', { class: 'ct-legend', role: 'group', 'aria-label': 'Filter methods by modification kind' });
    lit.appendChild(legend);
    var kindBtns = {};
    A.KINDS.forEach(function (k) {
      var n = H('span', { class: 'n' });
      var b = H('button', { type: 'button', class: 'ct-kind', 'aria-pressed': 'false', 'data-kind': k.key },
        [H('span', { class: 'ct-sw', style: 'background: var(' + k.token + ')' }), k.name + ' ', n]);
      b.__n = n;
      b.addEventListener('click', function () { st.kinds[k.key] = !st.kinds[k.key]; if (!st.kinds[k.key]) delete st.kinds[k.key]; update(); });
      kindBtns[k.key] = b;
      legend.appendChild(b);
    });
    legend.appendChild(H('span', { class: 'ct-lsep', 'aria-hidden': 'true' }));
    legend.appendChild(H('span', { class: 'ct-li' }, [H('span', { class: 'ct-sw hatch' }), 'papers']));
    var paleLi = H('span', { class: 'ct-li', hidden: 'hidden' }, [H('span', { class: 'ct-sw pale' }), 'filtered out']);
    var yearLi = H('span', { class: 'ct-li', hidden: 'hidden' }, [H('span', { class: 'ct-sw ysw' }), 'chosen year']);
    legend.appendChild(paleLi);
    legend.appendChild(yearLi);

    /* summary + export rows */
    var sum = H('p', { class: 'ct-sum', 'aria-live': 'polite' });
    lit.appendChild(sum);
    var exports = H('div', { class: 'ct-exports' });
    lit.appendChild(exports);
    function exportRow(which, primary) {
      var roId = 'ct-ro-' + which + '-' + Math.random().toString(36).slice(2, 7);
      var btn = H('button', { type: 'button', class: 'btn ct-go ct-cp' + (primary ? ' primary' : ''), 'aria-describedby': roId });
      var ro = H('div', { class: 'ct-ro', id: roId });
      exports.appendChild(H('div', { class: 'ct-row' }, [btn, ro]));
      btn.addEventListener('click', function () {
        var text = fileText(which);
        if (!text) return;
        var b = which === 'methods' ? S.bm : which === 'papers' ? S.bp : S.ball;
        if (st.prev !== which) { st.prev = which; prevSeg.set(which); renderPreview(); }
        copyWithFeedback(text, btn, litFb, plural(b.entries.length, 'BibTeX entry', 'BibTeX entries') + ', ' + fmtKB(utf8Len(text)), announce,
          'Copied ' + plural(b.entries.length, 'entry', 'entries') + ' ✓');
      });
      return { btn: btn, ro: ro };
    }
    var rowM = exportRow('methods', true), rowP = exportRow('papers', false), rowA = exportRow('all', false);
    var note = H('p', { class: 'ct-note' });
    lit.appendChild(note);
    var litFb = makeFallback(H);
    lit.appendChild(litFb.el);

    /* preview */
    var prevSeg = A.seg([{ value: 'methods', label: 'Methods' }, { value: 'papers', label: 'Papers' }, { value: 'all', label: 'Combined' }], st.prev,
      function (v) { st.prev = v; renderPreview(); }, 'Which bibliography to preview');
    var prevRo = H('span', { class: 'ct-ro' });
    lit.appendChild(H('div', { class: 'ct-prev-h' }, [H('span', { class: 'label', text: 'Preview · exactly what is copied' }), prevSeg.el, prevRo]));
    var prevPre = H('pre', { class: 'ct-bib ct-prev mathjax_ignore', tabindex: '0', 'aria-label': 'Preview of the bibliography that the buttons copy' });
    lit.appendChild(prevPre);
    stage.appendChild(live);

    /* ================= render ================= */
    function drawChart() {
      var w = Math.max(240, chartBox.getBoundingClientRect().width || 640);   /* fractional: the 1:1 viewBox keeps the labels at their set size */
      var nb = BINS.length;
      if (!nb) { svg.innerHTML = ''; return; }
      var narrow = w < 560;
      var band = w / nb;
      var bw = Math.max(6, Math.min(44, band * (narrow ? 0.64 : 0.56)));
      var maxM = Math.max.apply(null, totM.concat([1])), maxP = Math.max.apply(null, totP.concat([0]));
      var hTop = narrow ? 100 : 124;
      var k = hTop / Math.max(maxM, maxP, 1);          /* one scale for both halves: bar length = entries */
      var y0 = 16 + maxM * k, bandH = 20, y1 = y0 + bandH;
      var Ht = Math.ceil(y1 + maxP * k + 19);
      var fOn = mFilterOn(), pOn = pFilterOn(), useShort = band < 40;
      var ink2 = A.css('--ink-2'), ink3 = A.css('--ink-3');
      var kc = KINDS.map(function (kk) { return A.kindColor(kk); });
      var s = '<defs><pattern id="' + hatchId + '" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        + '<rect width="4" height="4" fill="' + ink2 + '" fill-opacity="0.1"></rect><line x1="0.6" y1="0" x2="0.6" y2="4" stroke="' + ink2 + '" stroke-width="1.2"></line></pattern></defs>';
      for (var i = 0; i < nb; i++) {
        var b = BINS[i], cx = band * (i + 0.5), x = cx - bw / 2, on = !!st.years[b.key];
        var sm = S.selM[i].reduce(function (a, c) { return a + c; }, 0), sp = S.selP[i];
        var lab = b.long + ': ' + plural(totM[i], 'method') + ' and ' + plural(totP[i], 'paper')
          + ((fOn || pOn) ? '; ' + sm + ' and ' + sp + ' selected' : '') + '. ' + (on ? 'Year filter on.' : 'Activate to filter by this year.');
        s += '<g class="ct-bin' + (on ? ' on' : '') + '" data-i="' + i + '" role="button" tabindex="' + (i === focusIdx ? 0 : -1) + '" aria-pressed="' + on + '" aria-label="' + esc(lab) + '">';
        s += '<rect class="ysel" x="' + (band * i + 1).toFixed(1) + '" y="0" width="' + (band - 2).toFixed(1) + '" height="' + Ht + '" rx="3"></rect>';
        s += '<rect class="hit" x="' + (band * i + 1).toFixed(1) + '" y="0" width="' + (band - 2).toFixed(1) + '" height="' + Ht + '" rx="3"></rect>';
        /* methods: selected part stacked by kind from the line up, the filtered-out remainder pale on top */
        var yc = y0;
        for (var j = 0; j < KINDS.length; j++) {
          var c = S.selM[i][j];
          if (!c) continue;
          s += '<rect x="' + x.toFixed(1) + '" y="' + (yc - c * k).toFixed(2) + '" width="' + bw.toFixed(1) + '" height="' + (c * k).toFixed(2) + '" fill="' + kc[j] + '"></rect>';
          yc -= c * k;
        }
        if (totM[i] > sm) s += '<rect x="' + x.toFixed(1) + '" y="' + (yc - (totM[i] - sm) * k).toFixed(2) + '" width="' + bw.toFixed(1) + '" height="' + ((totM[i] - sm) * k).toFixed(2) + '" fill="' + ink3 + '" fill-opacity="0.24"></rect>';
        var shownM = fOn ? sm : totM[i];
        s += '<text class="ct-cl' + (shownM ? '' : ' zero') + '" x="' + cx.toFixed(1) + '" y="' + (y0 - totM[i] * k - 4).toFixed(1) + '" text-anchor="middle">' + shownM + '</text>';
        /* papers: hang below the second line */
        if (sp) s += '<rect x="' + x.toFixed(1) + '" y="' + y1.toFixed(2) + '" width="' + bw.toFixed(1) + '" height="' + (sp * k).toFixed(2) + '" fill="url(#' + hatchId + ')" stroke="' + ink2 + '" stroke-width="0.8"></rect>';
        if (totP[i] > sp) s += '<rect x="' + x.toFixed(1) + '" y="' + (y1 + sp * k).toFixed(2) + '" width="' + bw.toFixed(1) + '" height="' + ((totP[i] - sp) * k).toFixed(2) + '" fill="' + ink3 + '" fill-opacity="0.24"></rect>';
        var shownP = pOn ? sp : totP[i];
        s += '<text class="ct-cl' + (shownP ? '' : ' zero') + '" x="' + cx.toFixed(1) + '" y="' + (y1 + totP[i] * k + 13.5).toFixed(1) + '" text-anchor="middle">' + shownP + '</text>';
        s += '<text class="ct-yl" x="' + cx.toFixed(1) + '" y="' + (y0 + bandH / 2 + 4.2).toFixed(1) + '" text-anchor="middle">' + esc(useShort ? b.short : b.lab) + '</text>';
        s += '</g>';
      }
      s += '<line class="ct-axis" x1="0" x2="' + w + '" y1="' + y0.toFixed(1) + '" y2="' + y0.toFixed(1) + '"></line>';
      s += '<line class="ct-axis" x1="0" x2="' + w + '" y1="' + y1.toFixed(1) + '" y2="' + y1.toFixed(1) + '"></line>';
      svg.setAttribute('viewBox', '0 0 ' + w + ' ' + Ht);
      svg.setAttribute('width', w);
      svg.setAttribute('height', Ht);
      svg.innerHTML = s;
      if (refocus) {
        refocus = false;
        var g = svg.querySelector('.ct-bin[data-i="' + focusIdx + '"]');
        if (g && g.focus) g.focus();
      }
    }
    function binTip(i) {
      var b = BINS[i], sm = S.selM[i].reduce(function (a, c) { return a + c; }, 0), sp = S.selP[i];
      var fOn = mFilterOn() || pFilterOn();
      var rows = KINDS.map(function (kk, j) { return { k: kk, n: S.selM[i][j] }; }).filter(function (r) { return r.n; })
        .map(function (r) { return '<span class="swatch" style="background:var(' + A.KINDS[kindPos[r.k]].token + ')"></span> ' + esc(A.kindName(r.k)) + ' <span class="num mono">' + r.n + '</span>'; });
      return '<div class="t">' + esc(b.long) + '</div>'
        + '<div class="num mono" style="font-size:.78rem">' + (fOn ? sm + ' of ' : '') + plural(totM[i], 'method') + ' · ' + (fOn ? sp + ' of ' : '') + plural(totP[i], 'paper') + '</div>'
        + (rows.length ? '<div style="font-size:.8rem;line-height:1.5;margin-top:.25rem">' + rows.join('<br>') + '</div>' : '')
        + '<div style="font-size:.76rem;color:var(--ink-2);margin-top:.25rem">' + (st.years[b.key] ? 'Click to drop this year' : 'Click to keep only the chosen years') + '</div>';
    }
    svg.addEventListener('mousemove', function (e) {
      var g = e.target.closest ? e.target.closest('.ct-bin') : null;
      if (!g) { A.tip.hide(); return; }
      A.tip.show(binTip(+g.getAttribute('data-i')), e);
    });
    svg.addEventListener('mouseleave', function () { A.tip.hide(); });
    function toggleYear(i, kb) {
      var key = BINS[i].key;
      if (st.years[key]) delete st.years[key]; else st.years[key] = true;
      focusIdx = i; refocus = !!kb;
      update();
    }
    svg.addEventListener('click', function (e) {
      var g = e.target.closest ? e.target.closest('.ct-bin') : null;
      if (!g) return;
      toggleYear(+g.getAttribute('data-i'), false);
      if (e.pointerType === 'mouse') A.tip.show(binTip(+g.getAttribute('data-i')), e); else A.tip.hide();
    });
    svg.addEventListener('keydown', function (e) {
      var g = e.target.closest ? e.target.closest('.ct-bin') : null;
      if (!g) return;
      var i = +g.getAttribute('data-i'), n = BINS.length, j = i;
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); toggleYear(i, true); return; }
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(n - 1, i + 1);
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
      else if (e.key === 'Home') j = 0;
      else if (e.key === 'End') j = n - 1;
      else return;
      e.preventDefault();
      focusIdx = j;
      Array.prototype.forEach.call(svg.querySelectorAll('.ct-bin'), function (x) { x.setAttribute('tabindex', +x.getAttribute('data-i') === j ? '0' : '-1'); });
      var t = svg.querySelector('.ct-bin[data-i="' + j + '"]');
      if (t) t.focus();
    });

    function renderPreview() {
      var text = fileText(st.prev);
      var b = st.prev === 'methods' ? S.bm : st.prev === 'papers' ? S.bp : S.ball;
      if (!text) {
        prevPre.innerHTML = '<span class="bl c">% Nothing matches the current filters.</span>';
        prevRo.innerHTML = '0 entries';
        return;
      }
      prevPre.innerHTML = hlBib(text, false);
      prevRo.innerHTML = '<b>' + b.entries.length + '</b> ' + (b.entries.length === 1 ? 'entry' : 'entries') + ' · ' + fmtKB(utf8Len(text));
    }

    function update() {
      S = select();
      var fM = mFilterOn(), fP = pFilterOn(), any = fM || fP;
      /* buttons with live counts */
      setLabel(rowM.btn, S.m.length ? 'Copy BibTeX for ' + scopeText(S.m.length, C.methods, 'methods') : 'No methods match');
      setLabel(rowP.btn, S.p.length ? 'Copy BibTeX for ' + scopeText(S.p.length, C.papers, 'papers') : 'No papers match');
      setLabel(rowA.btn, S.ball.entries.length ? 'Copy methods and papers together' : 'Nothing matches');
      rowM.btn.disabled = !S.bm.entries.length;
      rowP.btn.disabled = !S.bp.entries.length;
      rowA.btn.disabled = !S.ball.entries.length;
      function ro(b, extra) {
        if (!b.entries.length) return '<span>' + (b.n ? '' : 'nothing to copy') + '</span>';
        var t = fileText(b === S.bm ? 'methods' : b === S.bp ? 'papers' : 'all');
        return '<b>' + b.entries.length + '</b> ' + (b.entries.length === 1 ? 'entry' : 'entries') + ' · ' + fmtKB(utf8Len(t))
          + (b.dropped ? ' · ' + plural(b.dropped, 'repeated key') + ' merged' : '') + (extra || '')
          + (b.missing ? ' · ' + b.missing + ' without BibTeX' : '');
      }
      var cross = S.bm.entries.length + S.bp.entries.length - S.ball.entries.length;
      rowM.ro.innerHTML = ro(S.bm);
      rowP.ro.innerHTML = ro(S.bp);
      rowA.ro.innerHTML = ro(S.ball, '');
      /* footnote on merged keys, computed for the current selection */
      var bits = [];
      if (S.bm.shared.length) {
        var k0 = S.bm.shared[0], g0 = S.bm.groups[k0].map(function (m) { return m.name || m.id; });
        bits.push('One paper can introduce several methods: ' + esc(g0.slice(0, -1).join(', ')) + ' and ' + esc(g0[g0.length - 1]) + ' share <code>' + esc(k0) + '</code>'
          + (S.bm.shared.length > 1 ? ', and ' + plural(S.bm.shared.length - 1, 'more key is', 'more keys are') + ' shared like this' : '') + '. Each key is written once.');
      }
      if (cross > 0) bits.push(plural(cross, 'reading-list paper') + ' also ' + (cross === 1 ? 'introduces' : 'introduce') + ' a catalogued method; the combined file keeps one entry for each (the method’s record).');
      note.innerHTML = bits.join(' ');
      note.hidden = !bits.length;
      /* selection summary */
      sum.innerHTML = any
        ? 'Selected <b>' + S.m.length + '</b> of ' + C.methods + ' methods and <b>' + S.p.length + '</b> of ' + C.papers + ' papers · ' + esc(describe())
        : 'No filters: <b>' + C.methods + '</b> methods and <b>' + C.papers + '</b> papers, ' + S.ball.entries.length + ' distinct citation keys in all.';
      clearBtn.disabled = !any;
      /* legend */
      var picking = anyKinds();
      legend.classList.toggle('picking', picking);
      KINDS.forEach(function (kk) {
        var b = kindBtns[kk]; if (!b) return;
        b.setAttribute('aria-pressed', String(!!st.kinds[kk]));
        b.__n.textContent = String(S.kindN[kk] || 0);
        b.setAttribute('aria-label', A.kindName(kk) + ': ' + plural(S.kindN[kk] || 0, 'method') + '. ' + (st.kinds[kk] ? 'Kind filter on.' : 'Activate to keep only the chosen kinds.'));
      });
      paleLi.hidden = !any;
      yearLi.hidden = !anyYears();
      drawChart();
      renderPreview();
    }

    renderSelf();
    update();
    A.onTheme(function () { drawChart(); });
    var lastW = chartBox.clientWidth;
    var onResize = A.debounce(function () { var w = chartBox.clientWidth; if (w !== lastW) { lastW = w; drawChart(); } }, 80);
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(chartBox);
    else window.addEventListener('resize', onResize);

    /* expose computed numbers for the headless self-test (read-only snapshot) */
    el.__citeState = function () {
      return { counts: C, status: statusCount, bins: BINS.map(function (b, i) { return [b.lab, totM[i], totP[i]]; }),
        methods: { records: S.bm.n, entries: S.bm.entries.length, dropped: S.bm.dropped, shared: S.bm.shared.length, bytes: utf8Len(fileText('methods')) },
        papers: { records: S.bp.n, entries: S.bp.entries.length, dropped: S.bp.dropped, bytes: utf8Len(fileText('papers')) },
        all: { records: S.ball.n, entries: S.ball.entries.length, dropped: S.ball.dropped, bytes: utf8Len(fileText('all')) },
        selM: S.m.length, selP: S.p.length, atlas: atlasBib(cfg) };
    };
    el.__citeText = function (which) { return which === 'atlas' ? atlasBib(cfg) : fileText(which); };
  });
})();
