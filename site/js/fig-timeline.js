/* fig-timeline.js — Timeline of the PEFT literature.
   One dot per catalogued method (ATLAS_DATA.methods), placed at the month of its first arXiv
   submission (YYMM prefix of the identifier) or at mid-year when there is no identifier.
   Deterministic centred beeswarm (dodge layout), coloured by modification_kind, ringed when the
   method ships in Hugging Face PEFT. Hover/focus shows the paper and its catalogued relations;
   click selects it for the catalogue. Horizontal on wide stages, vertical on phones. */
(function () {
  'use strict';

  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var T0 = 2017; // the axis proper starts here; earlier entries go to a compressed gutter
  var GAP = 0.6; // px of air between neighbouring dots
  var instances = 0; // for unique element ids (no randomness anywhere in the figure)
  var LANDMARKS = [
    { id: 'houlsby-adapter', note: 'bottleneck adapters' },
    { id: 'prefix-tuning', note: 'trainable prefix vectors' },
    { id: 'lora', note: 'ΔW = BA, rank ≤ r' },
    { id: 'ia3', note: 'learned rescaling vectors' },
    { id: 'qlora', note: 'LoRA on a 4-bit base' },
    { id: 'dora', note: 'magnitude × direction' },
    { id: 'galore', note: 'low-rank gradient projection' }
  ];
  var STRUCTURAL = { 'builds-on': 1, generalizes: 1, 'special-case-of': 1, 'equivalent-to': 1, combines: 1, 'dual-to': 1 };

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }

  /* arXiv identifier -> {id, year, month, seq} (new style YYMM.NNNNN, or old style archive/YYMMNNN) */
  function parseArxiv(raw) {
    if (raw == null) return null;
    var s = String(raw).trim();
    if (!s) return null;
    s = s.replace(/^https?:\/\/(www\.)?arxiv\.org\/(abs|pdf)\//i, '').replace(/\.pdf$/i, '').replace(/^arxiv:\s*/i, '');
    var mm = s.match(/^(\d{2})(\d{2})\.(\d{4,5})(v\d+)?$/);
    if (mm) {
      var yy = +mm[1], mo = +mm[2];
      if (mo < 1 || mo > 12 || yy < 7) return null;
      return { id: mm[1] + mm[2] + '.' + mm[3], year: 2000 + yy, month: mo, seq: +mm[3] };
    }
    mm = s.match(/^[a-z][a-z\-]*(?:\.[a-z\-]+)?\/(\d{2})(\d{2})(\d{3})(v\d+)?$/i);
    if (mm) {
      var y2 = +mm[1], m2 = +mm[2];
      if (m2 < 1 || m2 > 12) return null;
      return { id: s.replace(/v\d+$/, ''), year: (y2 >= 91 ? 1900 : 2000) + y2, month: m2, seq: +mm[3] };
    }
    return null;
  }

  function dateLabel(it) { return it.exact ? MONTHS[it.month - 1] + ' ' + it.year : 'mid-' + it.year; }
  function venueYear(m) {
    var v = String(m.venue || '').trim();
    if (!v) return m.year ? String(m.year) : '';
    return /\b(19|20)\d{2}\b/.test(v) ? v : v + ' · ' + m.year;
  }

  /* deterministic dodge: points processed in the given order; each takes the offset q of least |q|
     that keeps it from overlapping every point already placed (circles of radius rad(it)).
     Ties between +q and -q alternate with the processing index, so the swarm stays centred. */
  function dodge(list, pos, rad, gap) {
    var out = {}, placed = [], lo = 0, hi = 0;
    for (var i = 0; i < list.length; i++) {
      var it = list[i], p = pos(it), R = rad(it), ivs = [], cands = [0];
      for (var j = 0; j < placed.length; j++) {
        var o = placed[j], dp = Math.abs(p - o.p), Dm = R + o.R + gap;
        if (dp < Dm) {
          var hh = Math.sqrt(Dm * Dm - dp * dp);
          ivs.push(o.q - hh, o.q + hh);
          cands.push(o.q - hh, o.q + hh);
        }
      }
      var pref = i % 2 ? 1 : -1;
      cands.sort(function (a, b) {
        var d = Math.abs(a) - Math.abs(b);
        if (Math.abs(d) > 1e-9) return d;
        return b * pref - a * pref;
      });
      var q = 0;
      for (var c = 0; c < cands.length; c++) {
        var y = cands[c], ok = true;
        for (var k = 0; k < ivs.length; k += 2) if (y > ivs[k] + 1e-7 && y < ivs[k + 1] - 1e-7) { ok = false; break; }
        if (ok) { q = y; break; }
      }
      placed.push({ p: p, q: q, R: R });
      out[it.id] = q;
      if (q - R < lo) lo = q - R;
      if (q + R > hi) hi = q + R;
    }
    return { q: out, lo: lo, hi: hi };
  }

  function injectCSS() {
    if (document.getElementById('css-timeline')) return;
    var s = document.createElement('style');
    s.id = 'css-timeline';
    s.textContent = [
      '[data-figure="timeline"] .tl-stage{padding:1.15rem 1.15rem .9rem}',
      /* type: display serif for the title, body serif for prose, Plex Sans for UI words, Plex Mono for numbers */
      '[data-figure="timeline"] .tl-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide);font-variant-numeric:tabular-nums}',
      '[data-figure="timeline"] .tl-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.3rem,3.4vw,1.85rem);line-height:1.12;letter-spacing:-.005em;margin:.25rem 0 0;color:var(--ink)}',
      '[data-figure="timeline"] .tl-instr{font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);margin:.35rem 0 .85rem;max-width:50rem}',
      '[data-figure="timeline"] .tl-controls{display:flex;flex-wrap:wrap;gap:.4rem .4rem;align-items:center;margin:0 0 .6rem}',
      '[data-figure="timeline"] .tl-key{display:inline-flex;align-items:center;gap:.42rem;font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;line-height:1.2;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:999px;padding:.3rem .62rem .3rem .45rem;cursor:pointer;white-space:nowrap}',
      '[data-figure="timeline"] .tl-key:hover{border-color:var(--ink-3)}',
      '[data-figure="timeline"] .tl-key .sw{display:inline-block;width:.74rem;height:.74rem;border-radius:50%;flex:none}',
      '[data-figure="timeline"] .tl-key .n{font-family:var(--f-mono);font-weight:400;font-size:.75rem;letter-spacing:0;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      '[data-figure="timeline"] .tl-key[aria-pressed="false"]{color:var(--ink-2);background:transparent;border-style:dashed}',
      '[data-figure="timeline"] .tl-key svg{display:block;flex:none}',
      '[data-figure="timeline"] .tl-toggle[aria-pressed="false"]{color:var(--ink);background:var(--paper);border-style:solid}',
      '[data-figure="timeline"] .tl-toggle[aria-pressed="true"]{color:var(--paper);background:var(--ink);border-color:var(--ink)}',
      '[data-figure="timeline"] .tl-toggle[aria-pressed="true"] .n{color:var(--paper-3)}',
      '[data-figure="timeline"] .tl-sep{width:1px;height:1.4rem;background:var(--rule);margin:0 .15rem}',
      '[data-figure="timeline"] .tl-reset{padding:.32rem .68rem;font-size:.75rem;font-weight:500;letter-spacing:.06em}',
      '[data-figure="timeline"] .tl-readout{display:flex;flex-wrap:wrap;gap:.15rem 1.2rem;font-family:var(--f-ui);font-size:.8rem;line-height:1.5;color:var(--ink-2);font-variant-numeric:tabular-nums;margin:0 0 .35rem;padding-top:.5rem;border-top:1px solid var(--rule)}',
      '[data-figure="timeline"] .tl-readout b{font-family:var(--f-mono);font-size:.78rem;color:var(--ink);font-weight:500}',
      '[data-figure="timeline"] .tl-plot{position:relative;min-height:120px}',
      '[data-figure="timeline"] .tl-plot svg{display:block;width:100%;height:auto;overflow:visible;touch-action:manipulation;-webkit-tap-highlight-color:transparent}',
      '[data-figure="timeline"] .tl-plot svg:focus{outline:none}',
      '[data-figure="timeline"] .tl-plot svg:focus-visible{outline:2px solid var(--ochre);outline-offset:4px;border-radius:2px}',
      '[data-figure="timeline"] .tl-foot{font-family:var(--f-body);font-size:.86rem;line-height:1.5;color:var(--ink-2);margin:.5rem 0 0;max-width:46rem}',
      '[data-figure="timeline"] .tl-foot mjx-container{color:var(--ink-2)}',
      '[data-figure="timeline"] .tl-empty{font-family:var(--f-body);font-style:italic;font-size:var(--fs-sm);color:var(--ink-2);padding:2rem 0;text-align:center}',
      '@media (max-width:560px){[data-figure="timeline"] .tl-stage{padding:.95rem .85rem .8rem}[data-figure="timeline"] .tl-controls{gap:.3rem}[data-figure="timeline"] .tl-key{font-size:.78rem;padding:.26rem .52rem .26rem .4rem;gap:.34rem}[data-figure="timeline"] .tl-key .sw{width:.66rem;height:.66rem}[data-figure="timeline"] .tl-sep{display:none}[data-figure="timeline"] .tl-reset{padding:.28rem .6rem;font-size:.75rem}}',
      '.atlas-tip .tl-tip{display:grid;gap:.28rem}',
      '.atlas-tip .tl-tip .t{line-height:1.15}',
      '.atlas-tip .tl-tip-full{font-style:italic;color:var(--ink-2);line-height:1.3}',
      '.atlas-tip .tl-tip-meta{font-family:var(--f-ui);font-size:.8rem;color:var(--ink-2);line-height:1.45;font-variant-numeric:tabular-nums}',
      '.atlas-tip .tl-tip-meta .id{font-family:var(--f-mono);font-size:.76rem}',
      '.atlas-tip .tl-tip-row{display:flex;flex-wrap:wrap;align-items:center;gap:.25rem .5rem;font-family:var(--f-ui);font-weight:500;font-size:.8rem;color:var(--ink-2)}',
      '.atlas-tip .tl-tip-sw{display:inline-block;width:.65rem;height:.65rem;border-radius:50%;margin-right:.3rem;vertical-align:-1px}',
      '.atlas-tip .tl-tip-hf{color:var(--ochre-ink)}',
      '.atlas-tip .tl-tip-idea{line-height:1.42;display:-webkit-box;-webkit-line-clamp:6;-webkit-box-orient:vertical;overflow:hidden}',
      '.atlas-tip .tl-tip-cls,.atlas-tip .tl-tip-hint{font-family:var(--f-ui);font-size:.78rem;line-height:1.45;color:var(--ink-2);font-variant-numeric:tabular-nums}'
    ].join('\n');
    document.head.appendChild(s);
  }

  Atlas.register('timeline', function (el, A) {
    injectCSS();
    var D = A.data();
    var KI = {};
    A.KINDS.forEach(function (k, i) { KI[k.key] = i; });

    /* ---------- items (one per catalogued method) ---------- */
    var items = [], undated = [];
    (D.methods || []).forEach(function (m) {
      if (!m || !m.id) return;
      var ax = parseArxiv(m.arxiv), it;
      if (ax) it = { t: ax.year + (ax.month - 0.5) / 12, year: ax.year, month: ax.month, exact: true, arxiv: ax.id, seq: ax.seq };
      else if (isFinite(+m.year) && +m.year >= 1950) it = { t: +m.year + 0.5, year: +m.year, month: null, exact: false, arxiv: null, seq: 1e9 };
      else { undated.push(m); return; }
      it.id = m.id; it.m = m; it.kind = m.modification_kind || 'hybrid';
      it.ki = KI[it.kind] != null ? KI[it.kind] : 99; it.hf = !!m.hf_peft;
      items.push(it);
    });
    items.sort(function (a, b) {
      return a.t - b.t || a.seq - b.seq || a.ki - b.ki || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
    });
    var byId = {};
    items.forEach(function (it, i) { byId[it.id] = it; it.order = i; });

    /* undirected relation graph restricted to catalogued methods */
    var nbr = {};
    items.forEach(function (it) { nbr[it.id] = {}; });
    items.forEach(function (it) {
      (it.m.relations || []).forEach(function (r) {
        if (!r || !byId[r.target] || r.target === it.id) return;
        var st = !!STRUCTURAL[r.type];
        nbr[it.id][r.target] = nbr[it.id][r.target] || st;
        nbr[r.target][it.id] = nbr[r.target][it.id] || st;
      });
    });

    /* legend kinds (fixed order from Atlas.KINDS, then any unknown kinds) */
    var kc = {};
    items.forEach(function (it) { kc[it.kind] = (kc[it.kind] || 0) + 1; });
    var kinds = [];
    A.KINDS.forEach(function (k) { if (kc[k.key]) kinds.push({ key: k.key, name: k.name, n: kc[k.key] }); });
    Object.keys(kc).forEach(function (k) { if (KI[k] == null) kinds.push({ key: k, name: A.kindName(k), n: kc[k] }); });
    var nHF = items.filter(function (it) { return it.hf; }).length;
    var nInexact = items.filter(function (it) { return !it.exact; }).length;
    var nClassified = items.filter(function (it) { return it.m.coords && it.m.coords.kind; }).length;

    var maxT = items.length ? items[items.length - 1].t : T0 + 1;
    var T1 = Math.max(T0 + 1, Math.floor(maxT) + 1);
    var early = items.filter(function (it) { return it.t < T0; });
    var hasEarly = early.length > 0;
    var earlyMin = hasEarly ? d3.min(early, function (it) { return it.year; }) : T0;
    var earlyLabel = !hasEarly ? '' : earlyMin === T0 - 1 ? String(T0 - 1) : early.every(function (it) { return it.year === earlyMin; }) ? String(earlyMin) : earlyMin + '–' + String(T0 - 1).slice(2);
    var firstYear = items.length ? items[0].year : T0;
    var lastYear = items.length ? items[items.length - 1].year : T0;

    /* catalogue build date (meta.built, ISO) as a right-hand marker */
    var builtT = null, builtLabel = '';
    var bm = D.meta && typeof D.meta.built === 'string' ? D.meta.built.match(/^(\d{4})-(\d{2})-(\d{2})/) : null;
    if (bm) {
      var by = +bm[1], bmo = +bm[2], bd = +bm[3], dim = new Date(Date.UTC(by, bmo, 0)).getUTCDate();
      builtT = by + (bmo - 1 + (bd - 1) / dim) / 12;
      builtLabel = 'catalogue built ' + bd + ' ' + MONTHS[bmo - 1] + ' ' + by;
      if (builtT < T0 || builtT > T1) builtT = null;
    }

    /* landmarks, by id when present, plus the newest entry */
    var landmarks = [];
    LANDMARKS.forEach(function (L) { if (byId[L.id]) landmarks.push({ it: byId[L.id], note: L.note }); });
    if (items.length) {
      var newest = items[items.length - 1], dup = null;
      landmarks.forEach(function (L) { if (L.it === newest) dup = L; });
      if (dup) dup.note += ' · newest'; else landmarks.push({ it: newest, note: 'newest entry' });
    }
    landmarks.sort(function (a, b) { return a.it.order - b.it.order; });
    var isLandmark = {};
    landmarks.forEach(function (L) { isLandmark[L.it.id] = L; });

    /* ---------- state ---------- */
    var hidden = {}, hfOnly = false, hoverId = null, focusId = null, selectedId = null, lastPointer = 'mouse';
    var geoCache = {}, G = null, P = {}, R = {}, prev = null, delaunay = null, visList = [], mounted = false, lastW = 0;
    function isVis(it) { return !hidden[it.kind] && (!hfOnly || it.hf); }

    /* ---------- DOM ---------- */
    var coarse = window.matchMedia && window.matchMedia('(hover: none)').matches;
    var uid = 'tl-fig-' + (++instances);
    var stage = A.h('div', { class: 'stage tl-stage' });
    el.appendChild(stage);
    stage.appendChild(A.h('div', { class: 'tl-kicker', text: 'Timeline · ' + plural(items.length, 'method') + ' · ' + firstYear + '–' + lastYear }));
    stage.appendChild(A.h('h3', { class: 'tl-title', text: 'The PEFT literature, month by month' }));
    stage.appendChild(A.h('p', {
      class: 'tl-instr', id: uid + '-instr',
      text: 'One dot per catalogued method, at the month its paper first appeared on arXiv. ' +
        (coarse ? 'Tap a dot for the paper and its relations, tap again to open it in the catalogue. Tap a kind to hide it.' :
          'Hover a dot for the paper and its relations, click to open it in the catalogue. Click a kind to hide it, double-click to isolate it.')
    }));

    var controls = A.h('div', { class: 'tl-controls', role: 'group', 'aria-label': 'Filter methods by modification kind' });
    stage.appendChild(controls);
    var keyBtns = kinds.map(function (k) {
      var sw = A.h('span', { class: 'sw', 'aria-hidden': 'true' });
      var b = A.h('button', { type: 'button', class: 'tl-key', 'aria-pressed': 'true', title: 'Click to hide or show; double-click or Shift+Enter to show only ' + k.name },
        [sw, A.h('span', { text: k.name }), A.h('span', { class: 'n', text: String(k.n) })]);
      b.addEventListener('click', function () { hidden[k.key] = !hidden[k.key]; changed(); });
      b.addEventListener('dblclick', function (e) {
        e.preventDefault();
        kinds.forEach(function (o) { hidden[o.key] = o.key !== k.key; });
        changed();
      });
      b.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); kinds.forEach(function (o) { hidden[o.key] = o.key !== k.key; }); changed(); }
      });
      controls.appendChild(b);
      return { k: k, b: b, sw: sw };
    });
    controls.appendChild(A.h('span', { class: 'tl-sep', 'aria-hidden': 'true' }));
    var ringIcon = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="3.2" class="tl-ri-dot"/><circle cx="7" cy="7" r="5.6" fill="none" class="tl-ri-ring" stroke-width="1.2"/></svg>';
    var hfBtn = A.h('button', { type: 'button', class: 'tl-key tl-toggle', 'aria-pressed': 'false', title: 'Ringed dots ship in Hugging Face PEFT. Press to show only those.', html: ringIcon });
    hfBtn.appendChild(A.h('span', { text: 'in HF PEFT only' }));
    hfBtn.appendChild(A.h('span', { class: 'n', text: String(nHF) }));
    hfBtn.addEventListener('click', function () { hfOnly = !hfOnly; changed(); });
    controls.appendChild(hfBtn);
    var resetBtn = A.h('button', { type: 'button', class: 'btn tl-reset', text: 'Show all' });
    resetBtn.addEventListener('click', function () { hidden = {}; hfOnly = false; changed(); });
    controls.appendChild(resetBtn);

    var readout = A.h('div', { class: 'tl-readout', 'aria-live': 'polite' });
    stage.appendChild(readout);
    var plot = A.h('div', { class: 'tl-plot' });
    stage.appendChild(plot);
    var live = A.h('div', { class: 'sr-only', 'aria-live': 'polite' });
    stage.appendChild(live);

    var foot = A.h('p', { class: 'tl-foot' });
    var footParts = [
      'A dot sits at \\(t = Y + (M - \\tfrac12)/12\\), with year \\(Y\\) and month \\(M\\) read from the YYMM prefix of the paper’s arXiv identifier (its month of first submission).'
    ];
    if (nInexact) footParts.push(plural(nInexact, 'entry', 'entries') + ' without an arXiv identifier ' + (nInexact === 1 ? 'sits' : 'sit') + ' at mid-year, \\(t = Y + \\tfrac12\\).');
    if (hasEarly) footParts.push(plural(early.length, 'entry', 'entries') + ' before ' + T0 + ' ' + (early.length === 1 ? 'is' : 'are') + ' drawn in a compressed band before the axis break.');
    if (undated.length) footParts.push(plural(undated.length, 'undated entry', 'undated entries') + ' cannot be placed and ' + (undated.length === 1 ? 'is' : 'are') + ' omitted.');
    footParts.push('Ringed dots (ink-outlined on narrow screens) ship in Hugging Face PEFT. Hovering draws the catalogue’s typed relations as arcs: solid for structural links (builds on, generalizes, special case, equivalent, combines, dual), dashed for competes-with. ' +
      (nClassified ? nClassified + ' of ' + items.length + ' entries carry an Erlangen classification.' : 'Erlangen classification pending.'));
    foot.innerHTML = A.esc(footParts.join(' '));
    stage.appendChild(foot);
    A.typeset(foot);

    if (!items.length) {
      plot.appendChild(A.h('div', { class: 'tl-empty', text: 'No dated methods in the catalogue yet.' }));
      return;
    }

    var svg = d3.select(plot).append('svg')
      .attr('tabindex', 0)
      .attr('role', 'group')
      .attr('aria-roledescription', 'timeline')
      .attr('aria-describedby', uid + '-instr')
      .attr('aria-label', 'Timeline of ' + items.length + ' PEFT methods by month of first arXiv submission. Use the arrow keys to step through methods in time order, Enter to open one in the catalogue.');
    var gBands = svg.append('g'), gBuilt = svg.append('g'), gAxis = svg.append('g'), gBars = svg.append('g');
    /* leaders sit under the dots so a line crossing a dense year never cuts through the data */
    var gLead = svg.append('g'), gArcs = svg.append('g').attr('fill', 'none'), gDots = svg.append('g'), gRings = svg.append('g');
    var gHi = svg.append('g').attr('pointer-events', 'none'), gLabels = svg.append('g').attr('pointer-events', 'none');

    /* ---------- text measurement ---------- */
    var mctx = document.createElement('canvas').getContext('2d');
    function measure(txt, font, ls) { mctx.font = font; return mctx.measureText(txt).width + (ls || 0) * txt.length; }

    /* ---------- geometry ---------- */
    function radius(g) {
      return function (it) { return it.hf ? (g.ring ? g.r + g.ring + 0.75 : g.r + 0.7) : g.r + 0.45; };
    }
    function geomH(W) {
      var g = { vertical: false, W: W, ring: 1.9, gap: GAP };
      var monoF = '400 12px ' + A.css('--f-mono');
      g.gut = hasEarly ? Math.max(30, Math.ceil(measure(earlyLabel, monoF)) + 8) : 0;
      g.x0 = g.gut ? g.gut + 16 : 4;
      g.x1 = W - 6;
      var span = g.x1 - g.x0;
      g.main = function (t) { return g.x0 + (t - T0) / (T1 - T0) * span; };
      g.pos = function (it) { return it.t < T0 ? g.gut / 2 : g.main(it.t); };
      g.r = clamp(span / 205, 2.4, 4.4);
      for (;;) {
        g.full = dodge(items, g.pos, radius(g), g.gap);
        if (g.full.hi - g.full.lo <= 340 || g.r <= 2.2) break;
        g.r -= 0.2;
      }
      /* landmark labels float just outside the local swarm envelope, above or below it (full layout,
         spine = 0, y grows downward). Branch-and-bound over side x anchor x level: labels never overlap,
         no leader line crosses a label; cost = total leader length + 1.5 x height added to the swarm. */
      var fName = '600 14px ' + A.css('--f-display'), fDate = '400 12px ' + A.css('--f-mono'), fNote = 'italic 400 12.5px ' + A.css('--f-body');
      var LH = 31, rad = radius(g), lo0 = g.full.lo, hi0 = g.full.hi;
      var dots = items.map(function (it) { return { x: g.pos(it), q: g.full.q[it.id], R: rad(it) }; });
      function env(b0, b1) {
        var top = 0, bot = 0;
        for (var i = 0; i < dots.length; i++) {
          var d = dots[i];
          if (d.x + d.R > b0 - 3 && d.x - d.R < b1 + 3) { if (d.q - d.R < top) top = d.q - d.R; if (d.q + d.R > bot) bot = d.q + d.R; }
        }
        return [top, bot];
      }
      var cands = landmarks.map(function (L) {
        var dt = dateLabel(L.it);
        L.date = dt;
        var w = Math.max(measure(L.it.m.name, fName) + 6 + measure(dt, fDate), measure(L.note, fNote)) + 4;
        var x = g.pos(L.it), q = g.full.q[L.it.id], r0 = rad(L.it), opts = [];
        ['start', 'end'].forEach(function (anchor, ai) {
          var b0 = anchor === 'start' ? x - 1 : x - 4 - w, b1 = anchor === 'start' ? x + 4 + w : x + 1;
          if (b0 < 0 || b1 > W) return;
          var e = env(b0, b1);
          opts.push({ side: 'above', anchor: anchor, b0: b0, b1: b1, base: e[0] - 6, pen: ai * 4 });
          opts.push({ side: 'below', anchor: anchor, b0: b0, b1: b1, base: e[1] + 6, pen: ai * 4 + 2 });
        });
        return { L: L, x: x, dotTop: q - r0, dotBot: q + r0, opts: opts };
      });
      function mk(cd, o, lvl) {
        var c = { side: o.side, anchor: o.anchor, b0: o.b0, b1: o.b1, x: cd.x, pen: o.pen };
        if (o.side === 'above') { c.y1 = lvl; c.y0 = lvl - LH; c.la = lvl; c.lb = cd.dotTop; c.len = cd.dotTop - lvl; }
        else { c.y0 = lvl; c.y1 = lvl + LH; c.la = cd.dotBot; c.lb = lvl; c.len = lvl - cd.dotBot; }
        return c;
      }
      function clash(c, o) {
        if (!(c.b1 + 8 < o.b0 || c.b0 - 8 > o.b1) && !(c.y1 + 4 < o.y0 || c.y0 - 4 > o.y1)) return true; // boxes overlap
        if (c.x > o.b0 - 3 && c.x < o.b1 + 3 && c.la < o.y1 + 2 && c.lb > o.y0 - 2) return true;         // my leader crosses o
        if (o.x > c.b0 - 3 && o.x < c.b1 + 3 && o.la < c.y1 + 2 && o.lb > c.y0 - 2) return true;         // o's leader crosses me
        return false;
      }
      var placed = [], best = null, bestCost = Infinity, nodes = 0;
      (function rec(i, lead, top, bot) {
        var cost = lead + 1.5 * (lo0 - top) + 1.5 * (bot - hi0);
        if (cost >= bestCost || ++nodes > 250000) return;
        if (i === cands.length) { bestCost = cost; best = placed.slice(); return; }
        var cd = cands[i], tries = [];
        cd.opts.forEach(function (o) {
          var levels = [o.base];
          placed.forEach(function (p) {
            if (o.side === 'above' && p.y0 - 6 < o.base) levels.push(p.y0 - 6);
            if (o.side === 'below' && p.y1 + 6 > o.base) levels.push(p.y1 + 6);
          });
          levels.forEach(function (lv) { tries.push(mk(cd, o, lv)); });
        });
        tries.sort(function (a, b) { return (a.len + a.pen) - (b.len + b.pen); });
        for (var k = 0; k < tries.length; k++) {
          var c = tries[k];
          if (placed.some(function (p) { return clash(c, p); })) continue;
          placed.push(c);
          rec(i + 1, lead + c.len + c.pen, Math.min(top, c.y0), Math.max(bot, c.y1));
          placed.pop();
        }
      })(0, 0, lo0, hi0);
      if (!best) { // fall back: stack in rows above the swarm
        best = cands.map(function (cd, i) {
          var o = cd.opts[0] || { side: 'above', anchor: 'start', b0: cd.x, b1: cd.x, pen: 0 };
          return mk(cd, o, lo0 - 6 - i * (LH + 4));
        });
      }
      cands.forEach(function (cd, i) { cd.L.h = best[i]; });
      var topExt = Math.min(lo0, d3.min(best, function (b) { return b.y0; }));
      var botExt = Math.max(hi0, d3.max(best, function (b) { return b.y1; }));
      g.top = 6;
      g.spine = g.top + 4 - topExt;
      g.swarmTop = g.spine + lo0;
      g.swarmBot = g.spine + botExt;
      g.axisY = g.swarmBot + (builtT != null ? 21 : 16); /* room for the build-date label above the axis */
      g.yearY = g.axisY + 19;
      g.BAR = 44;
      g.barBase = g.axisY + 32 + 13 + g.BAR;
      g.capY = g.barBase + 17;
      /* the axis title is one line where it fits and breaks at its middle dot where it does not */
      var capFull = 'Entries per year, by first arXiv submission · outline = whole catalogue';
      g.cap = measure(capFull, '500 12px ' + A.css('--f-ui')) <= W - g.x0 - 4 ? [capFull] : ['Entries per year, by first arXiv submission', 'outline = whole catalogue'];
      g.H = g.capY + 6 + (g.cap.length - 1) * 16;
      return g;
    }
    function geomV(W) {
      /* phones: time runs down; HF PEFT is an ink outline on the dot instead of a separate ring */
      var g = { vertical: true, W: W, ring: 0, gap: 0.35 };
      g.left = 46;
      g.lw = clamp(Math.round(W * 0.3), 92, 170);
      g.avail = W - g.left - g.lw - 10;
      g.ppy = clamp(Math.round(W * 0.25), 72, 96);
      g.top = 6;
      g.earlyH = hasEarly ? 50 : 0;
      g.main = function (t) { return g.top + g.earlyH + (t - T0) * g.ppy; };
      g.pos = function (it) { return it.t < T0 ? g.top + g.earlyH / 2 : g.main(it.t); };
      g.r = 4;
      for (;;) {
        g.full = dodge(items, g.pos, radius(g), g.gap);
        if (g.full.hi - g.full.lo <= g.avail || g.r <= 1.8) break;
        g.r -= 0.1;
      }
      g.spine = g.left + 6 + Math.max(0, (g.avail - (g.full.hi - g.full.lo)) / 2) - g.full.lo;
      g.endY = g.main(T1);
      g.H = g.endY + 44; /* two lines of axis title under the last band */
      /* labels in the right-hand column; a long name wraps at its parenthesis rather than losing it */
      var lx = W - g.lw + 8, maxW = W - lx - 2, fd = A.css('--f-display');
      var fDate = '400 12px ' + A.css('--f-mono');
      g.lx = lx;
      landmarks.forEach(function (L) {
        var fs = 13, lines = [L.it.m.name];
        function widest() { return d3.max(lines, function (t) { return measure(t, '600 ' + fs + 'px ' + fd); }); }
        while (fs > 12 && widest() > maxW) fs -= 0.5;
        var pm = L.it.m.name.match(/^(.*\S)\s*(\([^)]*\))\s*$/);
        if (widest() > maxW && pm) { lines = [pm[1], pm[2]]; fs = 13; while (fs > 12 && widest() > maxW) fs -= 0.5; }
        lines = lines.map(function (t) {
          while (measure(t, '600 ' + fs + 'px ' + fd) > maxW && t.length > 4) t = t.slice(0, -2).replace(/\s+$/, '') + '…';
          return t;
        });
        var dt = dateLabel(L.it), line2 = dt + ' · ' + L.note;
        if (measure(line2, fDate) > maxW) line2 = dt;
        L.v = { fs: fs, lines: lines, lh: fs + 1, line2: line2, y: g.pos(L.it) + 4 };
      });
      function span(v) { return (v.lines.length - 1) * v.lh + 32; } // first baseline to next label's first baseline
      for (var i = 1; i < landmarks.length; i++) {
        var a = landmarks[i - 1].v, b = landmarks[i].v;
        if (b.y < a.y + span(a)) b.y = a.y + span(a);
      }
      var lastL = landmarks[landmarks.length - 1];
      if (lastL) {
        var maxY = g.H - 18 - (lastL.v.lines.length - 1) * lastL.v.lh;
        if (lastL.v.y > maxY) {
          lastL.v.y = maxY;
          for (var j = landmarks.length - 2; j >= 0; j--) {
            if (landmarks[j].v.y > landmarks[j + 1].v.y - span(landmarks[j].v)) landmarks[j].v.y = landmarks[j + 1].v.y - span(landmarks[j].v);
          }
        }
      }
      return g;
    }

    /* ---------- bins ---------- */
    function bins(list) {
      var b = {};
      list.forEach(function (it) {
        var key = it.t < T0 ? 'early' : String(it.year);
        if (!b[key]) b[key] = { n: 0, k: {} };
        b[key].n++;
        b[key].k[it.kind] = (b[key].k[it.kind] || 0) + 1;
      });
      return b;
    }
    var allBins = bins(items);
    var binKeys = (hasEarly ? ['early'] : []).concat(d3.range(T0, T1).map(String));
    var maxBin = d3.max(binKeys, function (k) { return allBins[k] ? allBins[k].n : 0; }) || 1;

    /* ---------- render ---------- */
    function col(name) { return A.css(name); }
    function txt(sel, o) {
      sel.style('font-family', o.ff || A.css('--f-mono')).style('font-size', (o.fs || 11) + 'px').style('fill', o.fill || col('--ink-2'));
      if (o.fw) sel.style('font-weight', o.fw);
      if (o.ls) sel.style('letter-spacing', o.ls);
      if (o.fst) sel.style('font-style', o.fst);
      if (o.vn !== false) sel.style('font-variant-numeric', 'tabular-nums');
      if (o.halo) sel.style('paint-order', 'stroke').style('stroke', o.halo).style('stroke-width', '3px').style('stroke-linejoin', 'round');
      return sel;
    }

    function render(mode) { // mode: 'intro' | 'filter' | 'static'
      var W = Math.max(280, Math.floor(plot.getBoundingClientRect().width || plot.clientWidth || 320));
      lastW = W;
      /* geometry (scale, radius, full-swarm extent, label placement) depends only on the width: cache it */
      if (!geoCache[W] && Object.keys(geoCache).length >= 8) geoCache = {}; // a long drag-resize must not grow the cache without bound
      var g = (G = geoCache[W] || (geoCache[W] = W < 600 ? geomV(W) : geomH(W)));
      var reduce = A.reducedMotion();
      var animate = !reduce && (mode === 'intro' || (mode === 'filter' && prev && prev.W === W && prev.vertical === g.vertical));
      var vis = items.filter(isVis);
      visList = vis;
      var lay = dodge(vis, g.pos, radius(g), g.gap);
      var rad = radius(g);
      P = {}; R = {};
      vis.forEach(function (it) {
        var q = lay.q[it.id], p = g.pos(it);
        P[it.id] = g.vertical ? [g.spine + q, p] : [p, g.spine + q];
        R[it.id] = rad(it);
      });
      svg.attr('viewBox', '0 0 ' + W + ' ' + g.H).attr('width', W).attr('height', g.H);

      var ink = col('--ink'), ink2 = col('--ink-2'), ink3 = col('--ink-3'), rule = col('--rule'), paper2 = col('--paper-2'), paper3 = col('--paper-3');
      var fMono = A.css('--f-mono'), fDisp = A.css('--f-display'), fBody = A.css('--f-body'), fUi = A.css('--f-ui');
      [gBands, gBuilt, gAxis, gBars, gArcs, gLead, gDots, gRings, gHi, gLabels].forEach(function (s) { s.selectAll('*').remove(); });
      var vb = bins(vis), newBars = {};

      if (!g.vertical) {
        /* zebra year bands + month ruler */
        d3.range(T0, T1).forEach(function (y) {
          if ((y - T0) % 2) gBands.append('rect').attr('x', g.main(y)).attr('y', g.top).attr('width', g.main(y + 1) - g.main(y)).attr('height', g.axisY - g.top).attr('fill', paper3).attr('opacity', 0.55);
        });
        if (hasEarly) gBands.append('rect').attr('x', 0).attr('y', g.top).attr('width', g.gut).attr('height', g.axisY - g.top).attr('fill', paper3).attr('opacity', 0.55);
        gAxis.append('line').attr('x1', g.x0).attr('x2', g.x1).attr('y1', g.axisY).attr('y2', g.axisY).attr('stroke', ink3).attr('stroke-width', 1);
        for (var mi = 0; mi <= (T1 - T0) * 12; mi++) {
          var xm = g.main(T0 + mi / 12), yr = mi % 12 === 0;
          gAxis.append('line').attr('x1', xm).attr('x2', xm).attr('y1', g.axisY).attr('y2', g.axisY + (yr ? 7 : 3)).attr('stroke', yr ? ink3 : rule).attr('stroke-width', 1);
        }
        d3.range(T0, T1).forEach(function (y) {
          txt(gAxis.append('text').attr('x', (g.main(y) + g.main(y + 1)) / 2).attr('y', g.yearY).attr('text-anchor', 'middle').text(y), { fs: 12, fill: ink2 });
        });
        if (hasEarly) {
          gAxis.append('line').attr('x1', 2).attr('x2', g.gut - 2).attr('y1', g.axisY).attr('y2', g.axisY).attr('stroke', ink3);
          var bx = (g.gut + g.x0) / 2;
          [-2.5, 2.5].forEach(function (dx) {
            gAxis.append('line').attr('x1', bx + dx - 2.5).attr('x2', bx + dx + 2.5).attr('y1', g.axisY + 4).attr('y2', g.axisY - 4).attr('stroke', ink3).attr('stroke-width', 1);
          });
          txt(gAxis.append('text').attr('x', g.gut / 2).attr('y', g.yearY).attr('text-anchor', 'middle').text(earlyLabel), { fs: 12, fill: ink2 });
        }
        /* per-year count bars (stacked by kind for what is visible, ghost outline = whole catalogue) */
        var bw = Math.min(26, (g.main(T0 + 1) - g.main(T0)) * 0.42);
        binKeys.forEach(function (k) {
          var cx = k === 'early' ? g.gut / 2 : (g.main(+k) + g.main(+k + 1)) / 2;
          var w = k === 'early' ? Math.min(bw, g.gut * 0.6) : bw;
          var tot = allBins[k] ? allBins[k].n : 0, vn = vb[k] ? vb[k].n : 0;
          var hTot = tot / maxBin * g.BAR;
          if (tot) gBars.append('rect').attr('x', cx - w / 2 + 0.5).attr('y', g.barBase - hTot + 0.5).attr('width', w - 1).attr('height', Math.max(0, hTot - 1)).attr('fill', 'none').attr('stroke', rule).attr('stroke-width', 1);
          var y = g.barBase;
          kinds.forEach(function (kd) {
            var c = vb[k] && vb[k].k[kd.key];
            if (!c) return;
            var hh = c / maxBin * g.BAR;
            var r = gBars.append('rect').attr('x', cx - w / 2).attr('width', w).attr('fill', A.kindColor(kd.key));
            var pk = prev && prev.bars && prev.bars[k + '|' + kd.key];
            if (animate && mode === 'filter' && pk) {
              r.attr('y', pk[0]).attr('height', pk[1]).transition().duration(480).ease(d3.easeCubicInOut).attr('y', y - hh).attr('height', hh);
            } else if (animate && mode === 'intro') {
              r.attr('y', g.barBase).attr('height', 0).transition().delay(250 + (cx / W) * 700).duration(420).ease(d3.easeCubicOut).attr('y', y - hh).attr('height', hh);
            } else r.attr('y', y - hh).attr('height', hh);
            newBars[k + '|' + kd.key] = [y - hh, hh];
            y -= hh;
          });
          txt(gBars.append('text').attr('x', cx).attr('y', g.barBase - Math.max(hTot, vn / maxBin * g.BAR) - 4).attr('text-anchor', 'middle').text(vn), { fs: 12, fill: vn ? ink : ink2, fw: 500 });
        });
        gBars.append('line').attr('x1', hasEarly ? 2 : g.x0).attr('x2', g.x1).attr('y1', g.barBase + 0.5).attr('y2', g.barBase + 0.5).attr('stroke', rule);
        g.cap.forEach(function (line, li) {
          txt(gBars.append('text').attr('x', g.x0).attr('y', g.capY + li * 16).text(line), { ff: fUi, fs: 12, fw: 500, fill: ink2 });
        });
        /* build marker */
        if (builtT != null) {
          var xb = g.main(builtT);
          gBuilt.append('line').attr('x1', xb).attr('x2', xb).attr('y1', g.swarmTop - 6).attr('y2', g.axisY).attr('stroke', ink3).attr('stroke-dasharray', '2 3').attr('stroke-width', 1);
          txt(gBuilt.append('text').attr('x', xb - 5).attr('y', g.axisY - 5).attr('text-anchor', 'end').text(builtLabel), { ff: fUi, fs: 12, fw: 500, fill: ink2, halo: paper2 });
        }
      } else {
        /* vertical: year bands across, year/count column on the left */
        var swL = g.left + 2, swR = g.W - g.lw - 2;
        d3.range(T0, T1).forEach(function (y) {
          if ((y - T0) % 2) gBands.append('rect').attr('x', 0).attr('y', g.main(y)).attr('width', swR).attr('height', g.ppy).attr('fill', paper3).attr('opacity', 0.55);
          gBands.append('line').attr('x1', 0).attr('x2', swR).attr('y1', g.main(y) + 0.5).attr('y2', g.main(y) + 0.5).attr('stroke', rule);
        });
        gBands.append('line').attr('x1', 0).attr('x2', swR).attr('y1', g.endY + 0.5).attr('y2', g.endY + 0.5).attr('stroke', rule);
        if (hasEarly) {
          gBands.append('rect').attr('x', 0).attr('y', g.top).attr('width', swR).attr('height', g.earlyH - 4).attr('fill', paper3).attr('opacity', 0.55);
          gBands.append('line').attr('x1', 0).attr('x2', swR).attr('y1', g.top + g.earlyH - 3.5).attr('y2', g.top + g.earlyH - 3.5).attr('stroke', rule);
        }
        var barMax = g.left - 16;
        binKeys.forEach(function (k) {
          var top = k === 'early' ? g.top : g.main(+k);
          var label = k === 'early' ? earlyLabel : k;
          txt(gAxis.append('text').attr('x', 0).attr('y', top + 14).text(label), { fs: 12, fill: ink, fw: 500 });
          var tot = allBins[k] ? allBins[k].n : 0, vn = vb[k] ? vb[k].n : 0;
          if (tot) gBars.append('rect').attr('x', 0.5).attr('y', top + 20.5).attr('width', Math.max(0, tot / maxBin * barMax - 1)).attr('height', 5).attr('fill', 'none').attr('stroke', rule);
          var x = 0;
          kinds.forEach(function (kd) {
            var c = vb[k] && vb[k].k[kd.key];
            if (!c) return;
            var ww = c / maxBin * barMax;
            var r = gBars.append('rect').attr('y', top + 20).attr('height', 6).attr('fill', A.kindColor(kd.key));
            var pk = prev && prev.bars && prev.bars[k + '|' + kd.key];
            if (animate && mode === 'filter' && pk) r.attr('x', pk[0]).attr('width', pk[1]).transition().duration(480).ease(d3.easeCubicInOut).attr('x', x).attr('width', ww);
            else if (animate && mode === 'intro') r.attr('x', 0).attr('width', 0).transition().delay(250 + (top / g.H) * 700).duration(420).attr('x', x).attr('width', ww);
            else r.attr('x', x).attr('width', ww);
            newBars[k + '|' + kd.key] = [x, ww];
            x += ww;
          });
          txt(gBars.append('text').attr('x', 0).attr('y', top + 41).text(vn), { fs: 12, fill: ink2 });
        });
        ['Bars: entries per year,', 'by first arXiv submission'].forEach(function (line, li) {
          txt(gBars.append('text').attr('x', 0).attr('y', g.endY + 19 + li * 16).text(line), { ff: fUi, fs: 12, fw: 500, fill: ink2 });
        });
        if (builtT != null) {
          var yb = g.main(builtT);
          gBuilt.append('line').attr('x1', swL).attr('x2', swR).attr('y1', yb).attr('y2', yb).attr('stroke', ink3).attr('stroke-dasharray', '2 3');
          /* under the line, where no dot can be (nothing is newer than the build); above it if the band is too short */
          var ybl = g.endY - yb >= 17 ? yb + 14 : yb - 5;
          txt(gBuilt.append('text').attr('x', swR - 2).attr('y', ybl).attr('text-anchor', 'end').text(builtLabel.replace('catalogue built', 'built')), { ff: fUi, fs: 12, fw: 500, fill: ink2, halo: paper2 });
        }
      }

      /* dots */
      var dur = 520, ease = d3.easeCubicInOut;
      if (animate && mode === 'filter') {
        Object.keys(prev.P).forEach(function (id) {
          if (P[id]) return;
          var it = byId[id], pp = prev.P[id];
          gDots.append('circle').attr('cx', pp[0]).attr('cy', pp[1]).attr('r', G.r).attr('fill', A.kindColor(it.kind)).attr('pointer-events', 'none')
            .transition().duration(300).attr('r', 0).attr('opacity', 0).remove();
        });
      }
      vis.forEach(function (it) {
        var p = P[it.id], c = A.kindColor(it.kind);
        var outline = it.hf && !g.ring;
        var dot = gDots.append('circle').attr('data-id', it.id).attr('fill', c).attr('stroke', outline ? ink : paper2).attr('stroke-width', outline ? 1.2 : 0.75);
        var ring = it.hf && g.ring ? gRings.append('circle').attr('data-id', it.id).attr('fill', 'none').attr('stroke', ink).attr('stroke-width', 1.05) : null;
        var pp = animate && mode === 'filter' ? prev.P[it.id] : null;
        if (pp) {
          dot.attr('cx', pp[0]).attr('cy', pp[1]).attr('r', g.r).transition().duration(dur).ease(ease).attr('cx', p[0]).attr('cy', p[1]);
          if (ring) ring.attr('cx', pp[0]).attr('cy', pp[1]).attr('r', g.r + g.ring).transition().duration(dur).ease(ease).attr('cx', p[0]).attr('cy', p[1]);
        } else if (animate) {
          var delay = mode === 'intro' ? 60 + ((it.t < T0 ? T0 - 0.5 : it.t) - T0 + 0.5) / (T1 - T0 + 0.5) * 900 : 220;
          dot.attr('cx', p[0]).attr('cy', p[1]).attr('r', 0).transition().delay(delay).duration(380).ease(d3.easeBackOut.overshoot(1.6)).attr('r', g.r);
          if (ring) ring.attr('cx', p[0]).attr('cy', p[1]).attr('r', 0).attr('opacity', 0).transition().delay(delay + 120).duration(320).attr('r', g.r + g.ring).attr('opacity', 1);
        } else {
          dot.attr('cx', p[0]).attr('cy', p[1]).attr('r', g.r);
          if (ring) ring.attr('cx', p[0]).attr('cy', p[1]).attr('r', g.r + g.ring);
        }
      });

      /* landmarks */
      function leader(sel, d) {
        sel.append('path').attr('d', d).attr('fill', 'none').attr('stroke', paper2).attr('stroke-width', 3).attr('stroke-linecap', 'round');
        sel.append('path').attr('d', d).attr('fill', 'none').attr('stroke', ink3).attr('stroke-width', 1);
      }
      var labDelay = animate ? (mode === 'intro' ? 900 : dur) : 0;
      landmarks.forEach(function (L) {
        var it = L.it, p = P[it.id];
        if (!p) return;
        var gl = gLabels.append('g'), lead = gLead.append('g');
        if (!g.vertical) {
          var h = L.h, above = h.side === 'above';
          var base1 = above ? g.spine + h.y1 - 19 : g.spine + h.y0 + 12, base2 = above ? g.spine + h.y1 - 4 : g.spine + h.y0 + 27;
          var ly0 = above ? g.spine + h.y1 + 1 : g.spine + h.y0 - 1, ly1 = above ? p[1] - R[it.id] - 1.5 : p[1] + R[it.id] + 1.5;
          if (Math.abs(ly1 - ly0) > 1 && (above ? ly1 > ly0 : ly1 < ly0)) leader(lead, 'M' + h.x + ',' + ly0 + 'V' + ly1);
          lead.append('circle').attr('cx', h.x).attr('cy', ly0).attr('r', 1.5).attr('fill', ink3);
          var tx = h.anchor === 'start' ? h.x + 4 : h.x - 4;
          var t1 = txt(gl.append('text').attr('x', tx).attr('y', base1).attr('text-anchor', h.anchor), { halo: paper2 });
          txt(t1.append('tspan').text(it.m.name), { ff: fDisp, fs: 14, fill: ink, fw: 600, vn: false });
          txt(t1.append('tspan').attr('dx', 6).text(L.date), { ff: fMono, fs: 12, fill: ink2 });
          txt(gl.append('text').attr('x', tx).attr('y', base2).attr('text-anchor', h.anchor).text(L.note), { ff: fBody, fs: 12.5, fill: ink2, fst: 'italic', vn: false, halo: paper2 });
        } else {
          var v = L.v, ex = g.lx - 12;
          leader(lead, 'M' + (p[0] + R[it.id] + 1.5) + ',' + p[1] + 'H' + ex + 'L' + (g.lx - 4) + ',' + (v.y - 4));
          v.lines.forEach(function (t, li) {
            txt(gl.append('text').attr('x', g.lx).attr('y', v.y + li * v.lh).text(t), { ff: fDisp, fs: v.fs, fill: ink, fw: 600, vn: false, halo: paper2 });
          });
          txt(gl.append('text').attr('x', g.lx).attr('y', v.y + (v.lines.length - 1) * v.lh + 15).text(v.line2), { ff: fMono, fs: 12, fill: ink2, halo: paper2 });
        }
        /* the landmark dot is redrawn above its leader so the line visibly ends on it */
        lead.append('circle').attr('class', 'tl-lm').attr('data-id', it.id).attr('cx', p[0]).attr('cy', p[1]).attr('r', g.r)
          .attr('fill', A.kindColor(it.kind)).attr('stroke', it.hf && !g.ring ? ink : paper2).attr('stroke-width', it.hf && !g.ring ? 1.2 : 0.75);
        if (it.hf && g.ring) lead.append('circle').attr('class', 'tl-lm').attr('data-id', it.id).attr('cx', p[0]).attr('cy', p[1]).attr('r', g.r + g.ring)
          .attr('fill', 'none').attr('stroke', ink).attr('stroke-width', 1.05);
        if (labDelay) {
          gl.attr('opacity', 0).transition().delay(labDelay).duration(300).attr('opacity', 1);
          lead.attr('opacity', 0).transition().delay(labDelay).duration(300).attr('opacity', 1);
        }
      });

      if (!vis.length) {
        var ex = g.vertical ? (g.left + g.W - g.lw) / 2 : (g.x0 + g.x1) / 2;
        var ey = g.vertical ? g.main(T0 + 0.5) : g.spine;
        txt(gLabels.append('text').attr('x', ex).attr('y', ey).attr('text-anchor', 'middle').text('No methods match these filters'), { ff: fBody, fs: 15, fill: ink2, fst: 'italic', vn: false, halo: paper2 });
        txt(gLabels.append('text').attr('x', ex).attr('y', ey + 21).attr('text-anchor', 'middle').text('Show all brings every kind back'), { ff: fUi, fs: 12, fw: 500, fill: ink2, halo: paper2 });
      }

      prev = { W: W, vertical: g.vertical, P: P, bars: newBars };
      var pts = vis.map(function (it) { return P[it.id]; });
      delaunay = pts.length ? d3.Delaunay.from(pts) : null;
      updateLegend();
      updateReadout(vis);
      drawHighlight();
    }

    /* ---------- legend + readout ---------- */
    function updateLegend() {
      keyBtns.forEach(function (o) {
        var on = !hidden[o.k.key];
        o.b.setAttribute('aria-pressed', String(on));
        o.b.setAttribute('aria-label', o.k.name + ', ' + o.k.n + ' methods, ' + (on ? 'shown' : 'hidden') + '. Shift+Enter shows only this kind.');
        var c = A.kindColor(o.k.key);
        o.sw.style.background = on ? c : 'transparent';
        o.sw.style.boxShadow = on ? 'none' : 'inset 0 0 0 1.5px ' + c;
      });
      hfBtn.setAttribute('aria-pressed', String(hfOnly));
      var dot = hfBtn.querySelector('.tl-ri-dot'), ring = hfBtn.querySelector('.tl-ri-ring');
      if (dot) dot.setAttribute('fill', A.css(hfOnly ? '--paper-3' : '--ink-3'));
      if (ring) ring.setAttribute('stroke', A.css(hfOnly ? '--paper' : '--ink'));
    }
    function updateReadout(vis) {
      var n = vis.length;
      if (!n) { readout.innerHTML = '<span>No methods match these filters.</span>'; return; }
      var med = vis[Math.floor((n - 1) / 2)];
      var months = {}, bestM = null, years = {}, bestY = null;
      vis.forEach(function (it) {
        if (it.exact) {
          var k = it.year * 12 + it.month - 1;
          months[k] = (months[k] || 0) + 1;
          if (bestM == null || months[k] > months[bestM] || (months[k] === months[bestM] && k < bestM)) bestM = k;
        }
        if (it.t >= T0) {
          years[it.year] = (years[it.year] || 0) + 1;
          if (bestY == null || years[it.year] > years[bestY] || (years[it.year] === years[bestY] && it.year < bestY)) bestY = it.year;
        }
      });
      var nh = vis.filter(function (it) { return it.hf; }).length;
      var parts = ['<span>showing <b>' + n + '</b> of ' + items.length + '</span>'];
      parts.push('<span>median entry <b>' + dateLabel(med) + '</b></span>');
      if (bestM != null) parts.push('<span>busiest month <b>' + MONTHS[bestM % 12] + ' ' + Math.floor(bestM / 12) + '</b> (' + months[bestM] + ')</span>');
      if (bestY != null) parts.push('<span>busiest year <b>' + bestY + '</b> (' + years[bestY] + ')</span>');
      parts.push('<span>in HF PEFT <b>' + nh + '</b></span>');
      readout.innerHTML = parts.join('');
    }

    /* ---------- hover / focus / selection overlay ---------- */
    function activeId() { return hoverId || focusId; }
    function drawHighlight() {
      gArcs.selectAll('*').remove();
      gHi.selectAll('*').remove();
      var g = G, id = activeId(), ink2 = A.css('--ink-2'), ochre = A.css('--ochre'), ink = A.css('--ink');
      var related = {};
      if (id && P[id]) {
        var a = P[id], nb = nbr[id] || {};
        var nArcs = Object.keys(nb).filter(function (o) { return P[o]; }).length;
        var op = clamp(2.6 / Math.sqrt(Math.max(1, nArcs)), 0.26, 0.85); // a 170-arc fan stays a veil, a 3-arc fan stays crisp
        Object.keys(nb).forEach(function (o) {
          if (!P[o]) return;
          related[o] = true;
          var b = P[o], path;
          if (!g.vertical) {
            var dx = Math.abs(b[0] - a[0]), cy = Math.max(4, Math.min(a[1], b[1]) - clamp(dx * 0.32, 18, 150));
            path = 'M' + a[0] + ',' + a[1] + 'Q' + (a[0] + b[0]) / 2 + ',' + cy + ' ' + b[0] + ',' + b[1];
          } else {
            var dy = Math.abs(b[1] - a[1]), cx = Math.min(g.W - 4, Math.max(a[0], b[0]) + clamp(dy * 0.3, 16, 110));
            path = 'M' + a[0] + ',' + a[1] + 'Q' + cx + ',' + (a[1] + b[1]) / 2 + ' ' + b[0] + ',' + b[1];
          }
          gArcs.append('path').attr('d', path).attr('stroke', ink2).attr('stroke-width', nb[o] ? 1.1 : 0.9)
            .attr('stroke-dasharray', nb[o] ? null : '3 2.5').attr('opacity', nb[o] ? op : op * 0.75);
        });
        gDots.selectAll('circle').attr('opacity', function () { var d = this.getAttribute('data-id'); return !d || d === id || related[d] ? 1 : 0.28; });
        gRings.selectAll('circle').attr('opacity', function () { var d = this.getAttribute('data-id'); return d === id || related[d] ? 1 : 0.28; });
        gLead.selectAll('circle.tl-lm').attr('opacity', function () { var d = this.getAttribute('data-id'); return d === id || related[d] ? 1 : 0.28; });
        gHi.append('circle').attr('cx', a[0]).attr('cy', a[1]).attr('r', R[id] + 2.6).attr('fill', 'none').attr('stroke', ink).attr('stroke-width', 1.6);
      } else {
        gDots.selectAll('circle').attr('opacity', null);
        gRings.selectAll('circle').attr('opacity', null);
        gLead.selectAll('circle.tl-lm').attr('opacity', null);
      }
      if (selectedId && P[selectedId]) {
        var s = P[selectedId];
        gHi.append('circle').attr('cx', s[0]).attr('cy', s[1]).attr('r', R[selectedId] + 4.2).attr('fill', 'none').attr('stroke', ochre).attr('stroke-width', 2);
      }
    }
    function tipHTML(it) {
      var m = it.m, c = A.kindColor(it.kind), nb = nbr[it.id] || {};
      var nRel = 0, nStruct = 0;
      Object.keys(nb).forEach(function (o) { nRel++; if (nb[o]) nStruct++; });
      var nShown = Object.keys(nb).filter(function (o) { return P[o]; }).length;
      var cls = 'Erlangen kind: classification pending';
      if (m.coords && m.coords.kind) {
        var ax = (D.axes || []).filter(function (a) { return a.key === 'kind'; })[0], vname = '';
        if (ax && ax.values) ax.values.forEach(function (v) { if (v.key === m.coords.kind) vname = v.name; });
        cls = 'Erlangen kind: ' + A.esc(m.coords.kind) + (vname ? ' · ' + A.esc(vname) : '');
      }
      var h = '<div class="tl-tip"><div class="t">' + A.esc(m.name) + '</div>';
      if (m.full_name && m.full_name !== m.name) h += '<div class="tl-tip-full">' + A.esc(m.full_name) + '</div>';
      h += '<div class="tl-tip-meta">' + (m.authors ? A.esc(m.authors) + ' · ' : '') + A.esc(venueYear(m)) + '<br>' +
        (it.exact ? 'arXiv <span class="id">' + A.esc(it.arxiv) + '</span> · first submitted ' + dateLabel(it) : 'no arXiv identifier · placed at mid-' + it.year) + '</div>';
      h += '<div class="tl-tip-row"><span><span class="tl-tip-sw" style="background:' + c + '"></span>' + A.esc(A.kindName(it.kind)) + '</span>' +
        (it.hf ? '<span class="tl-tip-hf">HF PEFT' + (m.hf_peft_config ? ' · ' + A.esc(m.hf_peft_config) : '') + '</span>' : '') + '</div>';
      if (m.key_idea) h += '<div class="tl-tip-idea">' + A.esc(m.key_idea) + '</div>';
      h += '<div class="tl-tip-cls">' + plural(nRel, 'catalogued relation') + (nRel ? ' (' + nStruct + ' structural' + (nShown < nRel ? ', ' + nShown + ' visible' : '') + ')' : '') + '<br>' + cls + '</div>';
      h += '<div class="tl-tip-hint">' + (lastPointer === 'touch' ? 'Tap again' : 'Click or press Enter') + ' to open in the catalogue</div></div>';
      return h;
    }
    function clientEvt(id) {
      var p = P[id], rect = svg.node().getBoundingClientRect(), s = rect.width / (G.W || rect.width);
      return { clientX: rect.left + p[0] * s, clientY: rect.top + p[1] * s };
    }
    function setHover(id, evt) {
      if (id === hoverId && id) { if (evt) A.tip.move(evt); return; }
      hoverId = id;
      drawHighlight();
      var a = activeId();
      if (a && P[a]) A.tip.show(tipHTML(byId[a]), hoverId ? (evt || clientEvt(a)) : clientEvt(a));
      else A.tip.hide();
    }
    function setFocus(id) {
      focusId = id;
      hoverId = null;
      drawHighlight();
      if (id && P[id]) {
        var it = byId[id];
        A.tip.show(tipHTML(it), clientEvt(id));
        live.textContent = it.m.name + ', ' + dateLabel(it) + ', ' + A.kindName(it.kind) + (it.hf ? ', in Hugging Face PEFT' : '') + ', ' + venueYear(it.m) + '. ' + (visList.indexOf(it) + 1) + ' of ' + visList.length + '.';
      } else A.tip.hide();
    }
    function select(id) {
      if (!byId[id]) return;
      selectedId = id;
      drawHighlight();
      document.dispatchEvent(new CustomEvent('atlas:select-method', { detail: { id: id } }));
      var cat = document.getElementById('catalog');
      if (cat) {
        A.tip.hide();
        cat.scrollIntoView({ behavior: A.reducedMotion() ? 'auto' : 'smooth', block: 'start' });
      }
    }
    document.addEventListener('atlas:select-method', function (e) {
      var id = e && e.detail && e.detail.id;
      if (id && byId[id] && id !== selectedId) { selectedId = id; drawHighlight(); }
    });

    function hit(ev, generous) {
      if (!delaunay || !visList.length) return null;
      var rect = svg.node().getBoundingClientRect(), s = (G.W || rect.width) / rect.width;
      var x = (ev.clientX - rect.left) * s, y = (ev.clientY - rect.top) * s;
      var i = delaunay.find(x, y), it = visList[i];
      if (!it) return null;
      var p = P[it.id], d = Math.hypot(p[0] - x, p[1] - y);
      return d <= (generous ? 22 : Math.max(12, G.r * 3)) ? it.id : null;
    }
    svg.on('pointerdown', function (ev) { lastPointer = ev.pointerType || 'mouse'; });
    svg.on('pointermove', function (ev) {
      if (ev.pointerType === 'touch') return;
      lastPointer = ev.pointerType || 'mouse';
      var id = hit(ev);
      if (id) svg.style('cursor', 'pointer'); else svg.style('cursor', null);
      if (id !== hoverId || id) setHover(id, ev);
    });
    svg.on('pointerleave', function (ev) { if (ev.pointerType === 'touch') return; svg.style('cursor', null); setHover(null); });
    svg.on('click', function (ev) {
      var touch = lastPointer === 'touch' || lastPointer === 'pen';
      var id = hit(ev, touch);
      if (!id) { if (touch) setHover(null); return; }
      if (touch && hoverId !== id) { setHover(id, ev); return; }
      select(id);
    });
    var lastFocus = null;
    svg.on('focus', function () {
      var kb = true;
      try { kb = svg.node().matches(':focus-visible'); } catch (e) {}
      if (!kb) return;
      var start = (lastFocus && P[lastFocus] && lastFocus) || (selectedId && P[selectedId] && selectedId) || (P.lora && 'lora') || (visList[0] && visList[0].id);
      setFocus(start || null);
    });
    svg.on('blur', function () { if (focusId) lastFocus = focusId; focusId = null; hoverId = null; drawHighlight(); A.tip.hide(); });
    document.addEventListener('pointerdown', function (e) { if (hoverId && !svg.node().contains(e.target)) setHover(null); });
    window.addEventListener('scroll', function () { if (hoverId && lastPointer !== 'mouse') setHover(null); }, { passive: true });
    svg.on('keydown', function (ev) {
      if (!visList.length) return;
      var cur = focusId && byId[focusId] ? visList.indexOf(byId[focusId]) : -1, n = visList.length, nx = null;
      switch (ev.key) {
        case 'ArrowRight': case 'ArrowDown': nx = cur < 0 ? 0 : Math.min(n - 1, cur + 1); break;
        case 'ArrowLeft': case 'ArrowUp': nx = cur < 0 ? 0 : Math.max(0, cur - 1); break;
        case 'PageDown': nx = Math.min(n - 1, Math.max(0, cur) + 10); break;
        case 'PageUp': nx = Math.max(0, cur - 10); break;
        case 'Home': nx = 0; break;
        case 'End': nx = n - 1; break;
        case 'Enter': case ' ': if (focusId) { ev.preventDefault(); select(focusId); } return;
        case 'Escape': focusId = null; drawHighlight(); A.tip.hide(); return;
        default: return;
      }
      ev.preventDefault();
      setFocus(visList[nx].id);
    });

    function changed() {
      hoverId = null;
      if (focusId && !isVis(byId[focusId])) focusId = null;
      A.tip.hide();
      render('filter');
    }

    render('intro');
    mounted = true;
    A.onTheme(function () { render('static'); });
    var onResize = A.debounce(function () {
      var w = Math.floor(plot.getBoundingClientRect().width);
      if (w && Math.abs(w - lastW) >= 1) render('static');
    }, 120);
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(plot); else window.addEventListener('resize', onResize);
    /* label widths are measured with the web fonts; re-measure once if they were still loading */
    if (document.fonts && document.fonts.status !== 'loaded' && document.fonts.ready) document.fonts.ready.then(function () { if (mounted) { geoCache = {}; render('static'); } });
  });
})();
