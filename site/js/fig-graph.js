/* fig-graph.js — Figure 15: the category Meth(Θ, θ₀), drawn as a layered Hasse diagram.
   Objects are the core methods of ATLAS_DATA; arrows are every A2–A5 arrow recorded in
   ATLAS_DATA.edges and method.arrows between two core methods. Frozen (initial) sits at the
   bottom and Full FT (terminal) at the top; their A1 arrows are drawn as the interval frame and,
   for the method in focus, individually.
   Layout is a deterministic Sugiyama pipeline written here:
     1. strongly connected components (mutually simulating families) are collapsed;
     2. longest-path layering of the quotient DAG, then non-sinks promoted to sit just below
        their lowest successor (shorter arrows, no change of order);
     3. dummy vertices on long arrows, barycentric ordering sweeps with adjacent swaps,
        keeping the order with fewest crossings;
     4. ranks wider than the stage are split into sub-rows (still one Hasse level), and the
        sweeps are run again on the refined layering;
     5. x by iterated neighbour barycentres under order-preserving separation constraints.
   Nothing is random. Hover a node: its down-set (what it simulates) and up-set (what simulates it)
   light up; hover an arrow: its map h in a MathJax card. Click a node: pin it and fire
   atlas:select-method so the catalogue opens its card. */
(function () {
  'use strict';

  var RULES = {
    A2: { key: 'A2', name: 'freezing', long: 'A2 · freezing gives a sub-object', tok: '--tide', dash: null, hook: true },
    A3: { key: 'A3', name: 'factorisation', long: 'A3 · factorisation through h', tok: '--ink-2', dash: null },
    A4: { key: 'A4', name: 'product', long: 'A4 · projection of a product', tok: '--moss', dash: '5 3' },
    A5: { key: 'A5', name: 'sum / stage', long: 'A5 · summand or stage', tok: '--ochre', dash: '1.6 2.6' }
  };
  var RULE_ORDER = ['A2', 'A3', 'A4', 'A5'];
  var CODES = {
    R: { label: 'R', name: 'reparametrisation', inMeth: true, gloss: 'an object of the pointed slice, so Frozen \\(\\to\\) it \\(\\to\\) Full FT (A1).' },
    E: { label: 'E', name: 'pointed extension', gloss: 'lives on an extension of the architecture, not inside the interval [Frozen, Full FT]; its arrows are those of the extended slice.' },
    E0: { label: 'E°', name: 'unpointed extension', gloss: 'an extension with no neutral point, outside the weight-space slice; its arrows are those of the extended slice.' },
    B: { label: 'B', name: 'backward lens or schedule', gloss: 'acts on the update or the schedule; its arrows compare stages or periods.' },
    P: { label: 'P', name: 'post-hoc operation', gloss: 'operates on already-trained points; its arrows compare the operations.' },
    F_T: { label: 'F', name: 'family over a base', gloss: 'a family over tasks, inputs or tenants; its arrows are fibrewise.' }
  };
  var SHORT = {
    'houlsby-adapter': 'Houlsby', 'pfeiffer-adapter': 'Pfeiffer', 'miss': 'MiSS', 'super-tuning': 'Super-Tuning',
    'intruder-dimension-reduction': 'Intruder-dim.', 'polytropon': 'Poly', 'lora-soups': 'LoRA Soups',
    'riemannian-preconditioned-lora': 'Riemannian LoRA', 'text-to-lora': 'Text-to-LoRA', 'parallel-adapter': 'Parallel Adapter'
  };

  /* geometry (px at scale 1) */
  var FS = 12.5;          // base label size (display serif)
  var PADX = 7.5;         // pill horizontal padding
  var GAP = 11;           // gap between neighbouring objects in a row
  var GAP_IN = 26;        // gap between members of one mutually-simulating class (room for ⇄)
  var DGAP = 5;           // separation between two dummy vertices
  var DSEP = 8;           // extra clearance between a dummy and an object
  var SUB_GAP = 21;       // between sub-rows of one rank
  var RANK_GAP = 42;      // between ranks
  var MARGIN = 28;        // left/right margin (the A1 frame runs inside it)
  var HUB = 4;            // in-degree from which incoming arrows are bundled into a trunk
  var COMB_MIN = 6;       // leaf feeders from which a hub's leaves are drawn as a comb
  var OB_MAX = 12;        // obstruction lines drawn for a pinned method
  var SWEEPS = 24;        // barycentric sweeps per ordering pass
  var NARROW = 680;       // below this stage width: fixed canvas in a horizontal scroller (so the svg is never drawn below scale 1)
  var NARROW_W = 760;
  var instances = 0;

  function cmp(a, b) { return a < b ? -1 : a > b ? 1 : 0; }
  /* notes in the data sometimes name the build file they came from; the reader sees the table instead */
  function clean(s) {
    return String(s == null ? '' : s)
      .replace(/\(axes\.json ([AO]\d) examples?\)/g, function (m0, k) { return '(an example of ' + (k.charAt(0) === 'A' ? 'rule ' : 'test ') + k + ')'; })
      .replace(/^axes\.json ([AO]\d) example;\s*/, function (m0, k) { return 'The example of ' + (k.charAt(0) === 'A' ? 'rule ' : 'test ') + k + '; '; })
      .replace(/\b(the )?axes\.json's\b/g, 'the axis table\u2019s').replace(/\b(the )?axes\.json/g, 'the axis table')
      .replace(/(^|[.!?]\s+)the axis table/g, '$1The axis table');
  }
  /* svg text with Greek letters in the body serif (in the mono face Θ and θ are nearly one glyph) */
  function mixText(sel, str) {
    String(str).split(/([ΘθρΦ])/).forEach(function (part, i) {
      if (!part) return;
      var t = sel.append('tspan').text(part);
      if (i % 2) t.attr('class', 'gk');
    });
    return sel;
  }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }
  function shortName(m) {
    if (SHORT[m.id]) return SHORT[m.id];
    return String(m.name || m.id).replace(/\s*\((?!IA\))[^)]*\)\s*$/, '');
  }
  function kindTok(kind) {
    for (var i = 0; i < Atlas.KINDS.length; i++) if (Atlas.KINDS[i].key === kind) return Atlas.KINDS[i].token;
    return '--c-hybrid';
  }

  function injectCSS() {
    if (document.getElementById('css-graph')) return;
    var F = '[data-figure="graph"] ';
    var s = document.createElement('style');
    s.id = 'css-graph';
    s.textContent = [
      F + '.gr-stage{padding:1.15rem 1.15rem .9rem}',
      /* type: display serif for the title and the objects, body serif for prose, Plex Sans for UI words, Plex Mono for numbers */
      F + '.gr-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide);font-variant-numeric:tabular-nums}',
      F + '.gr-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.3rem,3.4vw,1.85rem);line-height:1.12;letter-spacing:-.005em;margin:.25rem 0 0;color:var(--ink)}',
      F + '.gr-title b{font-weight:700;letter-spacing:.01em}',
      F + '.gr-title .nw{white-space:nowrap}',
      F + 'svg tspan.gk{font-family:var(--f-body);font-style:italic;font-size:1.18em;letter-spacing:0}',
      F + '.gr-foot .nw{white-space:nowrap}',
      F + '.gr-wrap.narrow{display:flex;flex-direction:column}',
      F + '.gr-wrap.narrow .gr-zoom{position:static;order:-1;align-self:flex-end;margin:0 0 .4rem}',
      F + '.gr-instr{font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);margin:.35rem 0 .8rem;max-width:52rem}',
      F + '.gr-row{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:0 0 .5rem}',
      F + '.gr-key{display:inline-flex;align-items:center;gap:.42rem;font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;line-height:1.2;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:999px;padding:.3rem .62rem .3rem .45rem;cursor:pointer;white-space:nowrap}',
      F + '.gr-key:hover{border-color:var(--ink-3)}',
      F + '.gr-key .sw{display:inline-block;width:.74rem;height:.74rem;border-radius:3px;flex:none;border:1.5px solid currentColor}',
      F + '.gr-key .n{font-family:var(--f-mono);font-weight:400;font-size:.75rem;letter-spacing:0;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      F + '.gr-key[aria-pressed="false"]{color:var(--ink-2);background:transparent;border-style:dashed}',
      F + '.gr-key[aria-pressed="false"] .sw{opacity:.35}',
      F + '.gr-sep{width:1px;height:1.4rem;background:var(--rule);margin:0 .15rem}',
      F + '.gr-small{padding:.32rem .66rem;font-size:.75rem;font-weight:500}',
      F + '.gr-tools{display:flex;flex-wrap:wrap;gap:.45rem .7rem;align-items:center;margin:.15rem 0 .55rem}',
      F + '.gr-find{display:inline-flex;align-items:center;gap:.45rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.gr-find input{width:13.5rem;max-width:100%;font-family:var(--f-ui);font-weight:400;font-size:.875rem;letter-spacing:0;text-transform:none;padding:.3rem .55rem}',
      F + '.gr-wrap{position:relative}',
      F + '.gr-zoom{position:absolute;z-index:5;top:.35rem;right:.1rem;display:inline-flex;gap:.3rem}',
      F + '.gr-zoom .btn{padding:.26rem .58rem;font-size:.75rem;font-weight:500;min-width:1.9rem;background:var(--paper)}',
      F + '.gr-grp{display:inline-flex;flex-wrap:nowrap;align-items:center;gap:.45rem;max-width:100%}',
      F + '.gr-seg-label{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.gr-legend{display:flex;flex-wrap:wrap;gap:.3rem 1.05rem;align-items:center;font-family:var(--f-ui);font-size:.8rem;line-height:1.3;color:var(--ink-2);margin:0 0 .45rem}',
      F + '.gr-legend span.it{display:inline-flex;align-items:center;gap:.38rem;white-space:nowrap}',
      F + '.gr-legend svg{display:block;flex:none;overflow:visible}',
      F + '.gr-readout{display:flex;flex-wrap:wrap;gap:.15rem 1.15rem;font-family:var(--f-ui);font-size:.8rem;line-height:1.5;color:var(--ink-2);font-variant-numeric:tabular-nums;margin:0;padding:.5rem 0 .35rem;border-top:1px solid var(--rule)}',
      F + '.gr-readout b{font-family:var(--f-mono);font-size:.78rem;color:var(--ink);font-weight:500}',
      F + '.gr-readout button{font:inherit;color:var(--tide);background:none;border:0;padding:0;cursor:pointer;text-decoration:underline;text-underline-offset:.18em;text-decoration-thickness:.06em}',
      F + '.gr-readout button[aria-pressed="true"]{color:var(--ochre)}',
      F + '.gr-view{position:relative;margin:0 -.2rem}',
      F + '.gr-view>svg{display:block;width:100%;height:auto;overflow:hidden;touch-action:manipulation;-webkit-tap-highlight-color:transparent;cursor:grab;user-select:none;-webkit-user-select:none}',
      F + '.gr-view>svg:active{cursor:grabbing}',
      F + '.gr-view>svg:focus{outline:none}',
      F + '.gr-view>svg:focus-visible{outline:2px solid var(--ochre);outline-offset:3px;border-radius:3px}',
      F + '.gr-view.narrow{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;overscroll-behavior-x:contain;margin:0 -.85rem;padding:0 .85rem}',
      F + '.gr-view.narrow>svg{width:auto;max-width:none;touch-action:pan-x pan-y}',
      F + '.gr-hint{font-family:var(--f-body);font-size:.86rem;line-height:1.45;color:var(--ink-2);margin:.3rem 0 0}',
      /* svg */
      F + 'svg text.gr-lbl{font-family:var(--f-display);font-weight:500;fill:var(--ink);font-size:12.5px;pointer-events:none}',
      F + 'svg text.gr-end{font-family:var(--f-display);font-weight:var(--w-head);fill:var(--ink);pointer-events:none}',
      F + 'svg text.gr-sub{font-family:var(--f-ui);font-size:12px;letter-spacing:.02em;fill:var(--ink-2);pointer-events:none}',
      F + 'svg text.gr-ann{font-family:var(--f-ui);font-weight:500;font-size:12px;letter-spacing:.01em;fill:var(--ink-2);pointer-events:none}',
      F + 'svg text.gr-ann-i{font-family:var(--f-body);font-style:italic;font-size:12px;fill:var(--ink-2);pointer-events:none}',
      F + '.gr-n{cursor:pointer;transition:opacity .16s}',
      F + '.gr-n .bg{fill:var(--paper)}',
      F + '.gr-n .tn{fill-opacity:.15;stroke-width:1.15}',
      F + '.gr-n.ext .tn{stroke-dasharray:3.2 2.2}',
      F + '.gr-n .ring{fill:none;stroke:none;stroke-width:1.6}',
      F + '.gr-n.cur .ring{stroke:var(--ochre)}',
      F + '.gr-n.match .ring{stroke:var(--ochre);stroke-dasharray:3 2}',
      F + '.gr-n.cur .tn,' + F + '.gr-n:hover .tn{stroke-width:2;fill-opacity:.26}',
      F + '.gr-n.kfocus .ring{stroke:var(--ochre);stroke-width:2.4}',
      F + '.gr-e{fill:none;stroke-width:1.15;stroke-linecap:round;stroke-linejoin:round;opacity:.6;transition:opacity .16s}',
      F + '.gr-e .hd{stroke-dasharray:none}',
      F + '.gr-e.comp{opacity:.42}',
      F + '.gr-trunk{fill:none;stroke:var(--ink-2);stroke-width:1.5;stroke-linecap:round;opacity:.75;transition:opacity .16s}',
      F + '.gr-bus{stroke-width:1.25;opacity:.62}',
      F + 'svg.has-focus .gr-bus{opacity:.07}',
      F + '.gr-gframe{fill:var(--paper-3);fill-opacity:.55;stroke:var(--ink-3);stroke-width:.9;stroke-dasharray:2 2.5}',
      F + '.gr-a1{fill:none;stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 4;opacity:.7}',
      F + '.gr-a1hd{fill:none;stroke:var(--ink-3);stroke-width:1;opacity:.8}',
      F + '.gr-hit{fill:none;stroke:transparent;stroke-width:9;pointer-events:stroke;cursor:help}',
      F + '.gr-shelf{fill:none;stroke:var(--rule);stroke-width:1;stroke-dasharray:3 3}',
      F + '.gr-fx{fill:none;pointer-events:none}',
      F + '.gr-fx .a1{stroke:var(--ink-2);stroke-width:1.1;stroke-dasharray:2.5 3.5;opacity:.85}',
      F + '.gr-fx .ob{stroke:var(--seal);stroke-width:1.2;stroke-dasharray:1.5 3;opacity:.9}',
      F + '.gr-fx .obx{stroke:var(--seal);stroke-width:1.6;stroke-linecap:round}',
      F + '.gr-fx .obbg{fill:var(--paper);stroke:var(--seal);stroke-width:1}',
      F + 'svg.has-focus .gr-e{opacity:.06}',
      F + 'svg.has-focus .gr-e.hot{opacity:1}',
      F + 'svg.has-focus .gr-e.hot .ln{stroke-width:1.7}',
      F + 'svg.has-focus .gr-trunk{opacity:.08}',
      F + 'svg.has-focus .gr-trunk.hot{opacity:1}',
      F + 'svg.has-focus .gr-n{opacity:.2}',
      F + 'svg.has-focus .gr-n.hot{opacity:1}',
      F + 'svg.has-focus .gr-gframe{opacity:.35}',
      F + '.gr-n.off,' + F + '.gr-e.off{opacity:.07!important;pointer-events:none}',
      F + '.gr-hit.off{pointer-events:none}',
      /* card */
      F + '.gr-card{position:absolute;z-index:6;left:0;top:0;width:max-content;max-width:min(25rem,calc(100% - 16px));background:var(--paper);color:var(--ink);border:1px solid var(--rule);border-radius:6px;box-shadow:var(--shadow);padding:.6rem .75rem .65rem;font-size:var(--fs-sm);line-height:1.45;pointer-events:none;opacity:0;visibility:hidden;transition:opacity .12s,visibility .12s}',
      F + '.gr-card.on{opacity:1;visibility:visible}',
      F + '.gr-card.pinned{pointer-events:auto}',
      F + '.gr-panel{display:none}',
      F + '.gr-panel.on{display:block;position:fixed;z-index:45;left:16px;right:16px;bottom:10px;max-height:52vh;overflow-y:auto;overscroll-behavior:contain;background:var(--paper);border:1px solid var(--rule);border-radius:6px;box-shadow:var(--shadow);padding:.6rem .75rem .65rem;font-size:var(--fs-sm);line-height:1.45}',
      F + '.gr-panel.on.away{display:none}',
      F + '.gr-panel .gr-x{position:sticky;top:0;float:right;margin:-.2rem -.3rem 0 .4rem;font:inherit;font-family:var(--f-ui);font-size:.9rem;line-height:1;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);border-radius:999px;width:1.7rem;height:1.7rem;cursor:pointer}',
      F + '.gr-c-h{display:flex;align-items:baseline;gap:.45rem;flex-wrap:wrap}',
      F + '.gr-c-h .t{font-family:var(--f-display);font-size:1.25rem;font-weight:var(--w-head);line-height:1.15}',
      F + '.gr-c-h .arr{font-family:var(--f-display);color:var(--ink-2)}',
      F + '.gr-c-sw{display:inline-block;width:.7rem;height:.7rem;border-radius:3px;border:1.5px solid currentColor;align-self:center}',
      F + '.gr-c-full{font-style:italic;color:var(--ink-2);line-height:1.3;margin-top:.1rem}',
      F + '.gr-c-meta{font-family:var(--f-ui);font-size:.8rem;color:var(--ink-2);line-height:1.5;margin-top:.3rem;font-variant-numeric:tabular-nums}',
      F + '.gr-c-meta b{color:var(--ink);font-weight:500}',
      F + '.gr-c-rule{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase}',
      F + '.gr-c-sets{margin-top:.35rem;font-size:.84rem;color:var(--ink-2)}',
      F + '.gr-c-sets b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}',
      F + '.gr-c-sec{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:.5rem 0 .15rem}',
      F + '.gr-c-list{list-style:none;margin:0;padding:0;display:grid;gap:.22rem}',
      F + '.gr-c-list li{min-width:0}',
      F + '.gr-c-list .k{font-family:var(--f-ui);font-size:.8rem;color:var(--ink-2)}',
      F + '.gr-c-list .k b{font-weight:600}',
      F + '.gr-c-list .m{overflow-x:auto;overflow-y:hidden;font-size:.86rem;padding:.05rem 0 .1rem .1rem;scrollbar-width:thin}',
      F + '.gr-c-list .why{font-size:.8rem;color:var(--ink-2);line-height:1.4}',
      F + '.gr-c-math{overflow-x:auto;overflow-y:hidden;margin:.35rem 0 .2rem;font-size:1.08rem}',
      F + '.gr-c-math mjx-container[display="true"]{margin:.2em 0!important;text-align:left!important}',
      F + '.gr-c-note{font-size:.82rem;color:var(--ink-2);line-height:1.45}',
      F + '.gr-c-hint{font-family:var(--f-ui);font-size:.78rem;color:var(--ink-2);margin-top:.45rem}',
      F + '.gr-c-acts{display:flex;gap:.4rem;margin-top:.55rem;flex-wrap:wrap}',
      F + '.gr-c-acts .btn{padding:.32rem .64rem;font-size:.75rem;font-weight:500}',
      F + '.gr-foot{font-family:var(--f-body);font-size:.86rem;line-height:1.5;color:var(--ink-2);margin:.55rem 0 0;max-width:52rem}',
      F + '.gr-foot mjx-container{color:var(--ink-2)}',
      '@media (max-width:560px){' + F + '.gr-stage{padding:.95rem .85rem .8rem}' + F + '.gr-key{font-size:.78rem;padding:.26rem .52rem .26rem .4rem;gap:.34rem}' + F + '.gr-key .sw{width:.62rem;height:.62rem}' + F + '.gr-sep{display:none}' + F + '.gr-find{width:100%}' + F + '.gr-find input{flex:1;width:auto}' + F + '.gr-legend{font-size:.78rem;gap:.25rem .8rem}}'
    ].join('\n');
    document.head.appendChild(s);
  }

  /* ============================================================ model ============================================================ */
  function buildModel(D, scope) {
    var ms = (D.methods || []).filter(function (m) {
      return m && m.core && (scope !== 'R' || (m.coords && m.coords.kind === 'R'));
    });
    var by = {}, ord = {};
    ms.forEach(function (m, i) { by[m.id] = m; ord[m.id] = i; });

    /* every recorded A2–A5 arrow between two objects in scope: edges ∪ method.arrows, deduplicated */
    var seen = {}, arrows = [];
    function add(s, a) {
      if (!a || !by[s] || !by[a.target] || !RULES[a.rule]) return;
      var key = s + '|' + a.target + '|' + a.rule + '|' + (a.h || '');
      if (seen[key]) return;
      seen[key] = 1;
      arrows.push({ s: s, t: a.target, rule: a.rule, h: a.h || '', note: a.note || '' });
    }
    (D.edges || []).forEach(function (e) { add(e.source, e); });
    ms.forEach(function (m) { (m.arrows || []).forEach(function (a) { add(m.id, a); }); });
    arrows.sort(function (a, b) { return ord[a.s] - ord[b.s] || ord[a.t] - ord[b.t] || cmp(a.rule, b.rule) || cmp(a.h, b.h); });
    arrows.forEach(function (a, i) { a.i = i; a.self = a.s === a.t; });

    var out = {}, inn = {};
    ms.forEach(function (m) { out[m.id] = []; inn[m.id] = []; });
    arrows.forEach(function (a) { if (!a.self) { out[a.s].push(a); inn[a.t].push(a); } });

    /* Tarjan SCC (recursion depth is bounded by the 139 objects) */
    var index = 0, stack = [], onSt = {}, idx = {}, low = {}, comps = [];
    function strong(v) {
      idx[v] = low[v] = index++; stack.push(v); onSt[v] = true;
      out[v].forEach(function (a) {
        var w = a.t;
        if (idx[w] == null) { strong(w); low[v] = Math.min(low[v], low[w]); }
        else if (onSt[w]) low[v] = Math.min(low[v], idx[w]);
      });
      if (low[v] === idx[v]) {
        var c = [], w;
        do { w = stack.pop(); onSt[w] = false; c.push(w); } while (w !== v);
        comps.push(c);
      }
    }
    ms.forEach(function (m) { if (idx[m.id] == null) strong(m.id); });

    var nodes = {};
    ms.forEach(function (m) {
      var code = (m.coords && m.coords.kind) || 'R';
      nodes[m.id] = {
        id: m.id, m: m, label: shortName(m), kind: m.modification_kind || 'hybrid', code: code,
        ext: code !== 'R', indeg: inn[m.id].length, selfs: arrows.filter(function (a) { return a.self && a.s === m.id; })
      };
    });

    var groups = [], gOf = {};
    comps.forEach(function (c) {
      c.sort(function (a, b) { return ord[a] - ord[b]; });
      var g = { members: c.slice(), size: c.length };
      if (c.length > 2) {
        /* hub (most arrows inside the class) in the middle, the others alternately left and right */
        var deg = {};
        c.forEach(function (id) { deg[id] = 0; });
        arrows.forEach(function (a) { if (!a.self && deg[a.s] != null && deg[a.t] != null) { deg[a.s]++; deg[a.t]++; } });
        var hub = c.slice().sort(function (a, b) { return deg[b] - deg[a] || nodes[b].indeg - nodes[a].indeg || ord[a] - ord[b]; })[0];
        var rest = c.filter(function (id) { return id !== hub; }).sort(function (a, b) { return cmp(nodes[a].label.toLowerCase(), nodes[b].label.toLowerCase()); });
        var left = [], right = [];
        rest.forEach(function (id, i) { (i % 2 ? right : left).push(id); });
        g.members = left.reverse().concat([hub], right);
      } else if (c.length === 2) {
        g.members.sort(function (a, b) { return cmp(nodes[a].label.toLowerCase(), nodes[b].label.toLowerCase()); });
      }
      groups.push(g);
    });
    /* deterministic group order: by the first member's position in the data */
    groups.sort(function (a, b) {
      var ma = Math.min.apply(null, a.members.map(function (id) { return ord[id]; }));
      var mb = Math.min.apply(null, b.members.map(function (id) { return ord[id]; }));
      return ma - mb;
    });
    groups.forEach(function (g, i) { g.i = i; g.id = 'g' + i; g.members.forEach(function (id) { gOf[id] = g; }); g.out = []; g.inn = []; });

    /* quotient DAG */
    var ceMap = {}, ces = [];
    arrows.forEach(function (a) {
      if (a.self) return;
      var ga = gOf[a.s], gb = gOf[a.t];
      if (ga === gb) { a.inGroup = true; return; }
      var k = ga.i + '>' + gb.i;
      if (!ceMap[k]) { ceMap[k] = { a: ga, b: gb, arrows: [], i: ces.length }; ces.push(ceMap[k]); ga.out.push(ceMap[k]); gb.inn.push(ceMap[k]); }
      ceMap[k].arrows.push(a);
      a.ce = ceMap[k];
    });
    groups.forEach(function (g) { g.isolated = g.size === 1 && !g.out.length && !g.inn.length; });

    /* topological order (Kahn, smallest index first) */
    var indeg = groups.map(function (g) { return g.inn.length; }), queue = [], topo = [];
    groups.forEach(function (g) { if (!indeg[g.i]) queue.push(g); });
    while (queue.length) {
      queue.sort(function (a, b) { return a.i - b.i; });
      var g0 = queue.shift();
      topo.push(g0);
      g0.out.forEach(function (ce) { if (!--indeg[ce.b.i]) queue.push(ce.b); });
    }
    /* longest-path layering from the initial object, then promotion of non-sinks */
    topo.forEach(function (g) { g.asap = 0; g.inn.forEach(function (ce) { g.asap = Math.max(g.asap, ce.a.asap + 1); }); g.rank = g.asap; });
    for (var t = topo.length - 1; t >= 0; t--) {
      var g1 = topo[t];
      if (g1.out.length) g1.rank = Math.min.apply(null, g1.out.map(function (ce) { return ce.b.rank; })) - 1;
    }
    var maxRank = 0;
    groups.forEach(function (g) { if (!g.isolated) maxRank = Math.max(maxRank, g.rank); });

    /* covering relation: a quotient arrow a→b is a composite if b is reachable from a through another successor */
    ces.forEach(function (ce) {
      var seenG = {}, st = [];
      ce.a.out.forEach(function (o) { if (o.b !== ce.b) st.push(o.b); });
      var found = false;
      while (st.length && !found) {
        var x = st.pop();
        if (x === ce.b) { found = true; break; }
        if (seenG[x.i]) continue;
        seenG[x.i] = 1;
        x.out.forEach(function (o) { st.push(o.b); });
      }
      ce.composite = found;
      ce.arrows.forEach(function (a) { a.composite = found; });
    });

    /* longest chain in the quotient (deterministic: earliest group wins ties) */
    var best = {}, prevG = {};
    topo.forEach(function (g) {
      best[g.i] = 0; prevG[g.i] = null;
      g.inn.forEach(function (ce) {
        var v = best[ce.a.i] + 1;
        if (v > best[g.i] || (v === best[g.i] && prevG[g.i] && ce.a.i < prevG[g.i].a.i)) { best[g.i] = v; prevG[g.i] = ce; }
      });
    });
    var endG = null;
    topo.forEach(function (g) { if (endG === null || best[g.i] > best[endG.i]) endG = g; });
    var chainCes = [];
    for (var cur = endG; cur && prevG[cur.i]; cur = prevG[cur.i].a) chainCes.unshift(prevG[cur.i]);
    var chain = { ces: chainCes, arrows: [], ids: [] };
    chainCes.forEach(function (ce, k) {
      /* prefer an arrow whose source is the previous arrow's target (an honest path of methods) */
      var prevT = k ? chain.arrows[k - 1].t : null, pick = null;
      ce.arrows.forEach(function (a) { if (!pick || (prevT && a.s === prevT && pick.s !== prevT)) pick = a; });
      chain.arrows.push(pick);
    });
    chain.arrows.forEach(function (a, k) {
      if (!k) chain.ids.push(a.s);
      else if (chain.ids[chain.ids.length - 1] !== a.s) chain.ids.push(a.s);
      chain.ids.push(a.t);
    });
    /* bridge a step inside a class (e.g. arrive at LoRA-FA, leave from Flora) by its in-class arrow */
    chain.inner = [];
    chain.arrows.forEach(function (a, k) {
      if (!k) return;
      var p = chain.arrows[k - 1];
      if (p.t !== a.s) arrows.forEach(function (x) { if (x.inGroup && x.s === p.t && x.t === a.s) chain.inner.push(x); });
    });

    var stats = { objects: ms.length, arrows: 0, byRule: { A2: 0, A3: 0, A4: 0, A5: 0 }, selfs: 0, composite: 0, cycles: 0, cycMembers: 0, isolated: 0, chain: chainCes.length };
    arrows.forEach(function (a) {
      if (a.self) { stats.selfs++; return; }
      stats.arrows++; stats.byRule[a.rule]++;
      if (a.composite) stats.composite++;
    });
    groups.forEach(function (g) { if (g.size > 1) { stats.cycles++; stats.cycMembers += g.size; } if (g.isolated) stats.isolated++; });

    /* obstructions recorded between two objects in scope */
    var obst = [];
    ms.forEach(function (m) {
      (m.obstructions || []).forEach(function (o) { if (o && by[o.target] && o.target !== m.id) obst.push({ s: m.id, t: o.target, test: o.test || '', reason: o.reason || '' }); });
    });

    return { scope: scope, methods: ms, nodes: nodes, arrows: arrows, out: out, inn: inn, groups: groups, gOf: gOf, ces: ces, topo: topo,
      maxRank: maxRank, chain: chain, stats: stats, obst: obst, by: by };
  }

  /* reachability sets on the method level (cycles allowed) */
  function reach(adj, key, start) {
    var seen = {}, st = [start], res = [];
    while (st.length) {
      var v = st.pop();
      (adj[v] || []).forEach(function (a) { var w = a[key]; if (!seen[w]) { seen[w] = 1; res.push(w); st.push(w); } });
    }
    return seen;
  }

  /* ============================================================ layout ============================================================ */
  function makeMeasure() {
    var cv = document.createElement('canvas'), ctx = cv.getContext('2d');
    var fam = (Atlas.css('--f-display') || 'Georgia, serif');
    return function (txt, fs, wt) { ctx.font = (wt || 500) + ' ' + fs + 'px ' + fam; return ctx.measureText(txt).width; };
  }
  function labelSize(n) {
    var k = Math.log2(n.indeg + 1) - 2.6;
    return FS + Math.max(0, Math.min(5, k * 1.6));
  }

  /* order the items of a layered graph (layers[L] arrays; items carry ups/dns) */
  function crossings(layers) {
    var total = 0;
    for (var L = 0; L < layers.length - 1; L++) {
      var pairs = [];
      layers[L].forEach(function (it) { it.ups.forEach(function (n) { pairs.push(it.pos, n.it.pos); }); });
      for (var i = 0; i < pairs.length; i += 2) for (var j = i + 2; j < pairs.length; j += 2) {
        var d0 = pairs[i] - pairs[j], d1 = pairs[i + 1] - pairs[j + 1];
        if ((d0 < 0 && d1 > 0) || (d0 > 0 && d1 < 0)) total++;
      }
    }
    return total;
  }
  function sweep(layers, dir) {
    var start = dir > 0 ? 1 : layers.length - 2;
    for (var L = start; L >= 0 && L < layers.length; L += dir) {
      var lay = layers[L], withNb = [], out = new Array(lay.length);
      lay.forEach(function (it) {
        var nb = dir > 0 ? it.dns : it.ups;
        if (nb.length) {
          var s = 0;
          nb.forEach(function (n) { s += n.it.pos; });
          it._bc = s / nb.length; withNb.push(it);
        } else out[it.pos] = it; /* no neighbour on that side: keep its slot */
      });
      withNb.sort(function (a, b) { return a._bc - b._bc || a.pos - b.pos; });
      var j = 0;
      for (var i = 0; i < out.length; i++) if (!out[i]) out[i] = withNb[j++];
      out.forEach(function (it, k) { it.pos = k; });
      layers[L] = out;
    }
  }
  function pairCross(u, v) { /* crossings among u's and v's arrows when u is left of v */
    var c = 0;
    u.ups.forEach(function (a) { v.ups.forEach(function (b) { if (a.it.pos > b.it.pos) c++; }); });
    u.dns.forEach(function (a) { v.dns.forEach(function (b) { if (a.it.pos > b.it.pos) c++; }); });
    return c;
  }
  function transpose(layers) {
    for (var round = 0; round < 6; round++) {
      var improved = false;
      layers.forEach(function (lay) {
        for (var i = 0; i < lay.length - 1; i++) {
          var u = lay[i], v = lay[i + 1];
          if (pairCross(v, u) < pairCross(u, v)) {
            lay[i] = v; lay[i + 1] = u; v.pos = i; u.pos = i + 1; improved = true;
          }
        }
      });
      if (!improved) break;
    }
  }
  function orderLayers(layers, sweeps) {
    layers.forEach(function (lay) { lay.forEach(function (it, i) { it.pos = i; }); });
    var bestC = crossings(layers), bestO = layers.map(function (l) { return l.slice(); });
    for (var s = 0; s < sweeps; s++) {
      sweep(layers, s % 2 ? -1 : 1);
      transpose(layers);
      var c = crossings(layers);
      if (c < bestC) { bestC = c; bestO = layers.map(function (l) { return l.slice(); }); }
    }
    for (var L = 0; L < layers.length; L++) { layers[L] = bestO[L]; layers[L].forEach(function (it, i) { it.pos = i; }); }
    return bestC;
  }
  function link(lo, hi, loOff, hiOff) {
    lo.ups.push({ it: hi, so: loOff, oo: hiOff });
    hi.dns.push({ it: lo, so: hiOff, oo: loOff });
  }

  /* order the layers of each weakly connected component on its own, then concatenate them,
     so arrows of unrelated families never cross */
  function orderByComponents(layers, sweeps) {
    var par = {}, all = [];
    function find(x) { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    layers.forEach(function (lay) { lay.forEach(function (it) { par[it.id] = it.id; all.push(it); }); });
    all.forEach(function (it) { it.ups.forEach(function (n) { var a = find(it.id), b = find(n.it.id); if (a !== b) par[a] = b; }); });
    var comps = {}, order = [];
    all.forEach(function (it) { var r = find(it.id); if (!comps[r]) { comps[r] = []; order.push(r); } comps[r].push(it); });
    order.sort(function (a, b) {
      var wa = 0, wb = 0;
      comps[a].forEach(function (it) { wa += it.t === 'd' ? 0 : 1; });
      comps[b].forEach(function (it) { wb += it.t === 'd' ? 0 : 1; });
      return wb - wa || cmp(a, b);
    });
    var total = 0, sub = order.map(function (r) {
      var mine = {};
      comps[r].forEach(function (it) { mine[it.id] = 1; });
      var ls = layers.map(function (lay) { return lay.filter(function (it) { return mine[it.id]; }); });
      total += orderLayers(ls, sweeps);
      return ls;
    });
    for (var L = 0; L < layers.length; L++) {
      var row = [];
      sub.forEach(function (ls) { row = row.concat(ls[L]); });
      row.forEach(function (it, i) { it.pos = i; });
      layers[L] = row;
    }
    return { crossings: total, components: order.length };
  }

  /* a hub's leaf feeders as one compound object: chips in rows, split by a central spine;
     each chip ticks up to a rail above its row, rails join the spine, the spine rises to the hub */
  function buildComb(c, items, maxW, KI, nodes) {
    var AISLE = 22, CG = 9;
    items.sort(function (a, b) {
      var na = nodes[a.g.members[0]], nb = nodes[b.g.members[0]];
      return (KI[na.kind] == null ? 99 : KI[na.kind]) - (KI[nb.kind] == null ? 99 : KI[nb.kind]) || cmp(na.label.toLowerCase(), nb.label.toLowerCase());
    });
    var n = items.length, rowH = 0;
    items.forEach(function (it) { rowH = Math.max(rowH, it.h); });
    function arrange(per) {
      var rows = Math.ceil(n / per), bal = Math.ceil(n / rows), out = [], wmax = 0;
      for (var r = 0; r < rows; r++) {
        var row = items.slice(r * bal, (r + 1) * bal), nl = Math.ceil(row.length / 2);
        var left = row.slice(0, nl), right = row.slice(nl), lw = 0, rw = 0;
        left.forEach(function (it, i) { lw += it.w + (i ? CG : 0); });
        right.forEach(function (it, i) { rw += it.w + (i ? CG : 0); });
        wmax = Math.max(wmax, 2 * Math.max(lw, rw) + AISLE);
        out.push({ left: left, right: right, lw: lw, rw: rw });
      }
      return { rows: out, w: wmax };
    }
    var best = arrange(2);
    for (var per = 4; per <= Math.min(n, 16); per += 2) { var a = arrange(per); if (a.w <= maxW) best = a; else break; }
    var PITCH = rowH + 15, TOP = 9;
    c.w = best.w; c.h = TOP + best.rows.length * PITCH - 4; c.rowH = rowH; c.cells = []; c.rails = [];
    best.rows.forEach(function (row, r) {
      var cy = -c.h / 2 + TOP + r * PITCH + rowH / 2, x = -AISLE / 2;
      for (var i = row.left.length - 1; i >= 0; i--) { var it = row.left[i]; c.cells.push({ it: it, dx: x - it.w / 2, dy: cy, row: r }); x -= it.w + CG; }
      var xl = x + CG;
      x = AISLE / 2;
      row.right.forEach(function (it2) { c.cells.push({ it: it2, dx: x + it2.w / 2, dy: cy, row: r }); x += it2.w + CG; });
      c.rails.push({ y: cy - rowH / 2 - 6, x0: Math.min(xl, -AISLE / 2) , x1: Math.max(x - CG, AISLE / 2) });
    });
  }

  function layout(model, W, measure) {
    var nodes = model.nodes, lo = MARGIN, hi = W - MARGIN, avail = hi - lo;
    var KI = {};
    Atlas.KINDS.forEach(function (k, i) { KI[k.key] = i; });
    /* object sizes */
    var pill = {};
    model.methods.forEach(function (m) {
      var n = nodes[m.id], fs = labelSize(n);
      pill[m.id] = { fs: fs, w: Math.ceil(measure(n.label, fs) + 2 * PADX), h: Math.round(fs + 9) };
    });
    var gItem = {};
    model.groups.forEach(function (g) {
      var w = 0, h = 0, offs = {};
      g.members.forEach(function (id, k) { w += pill[id].w + (k ? GAP_IN : 0); h = Math.max(h, pill[id].h); });
      var x = -w / 2;
      g.members.forEach(function (id) { offs[id] = x + pill[id].w / 2; x += pill[id].w + GAP_IN; });
      gItem[g.i] = { t: 'g', g: g, w: w, h: h, offs: offs, id: 'g:' + g.members.map(function (id) { return nodes[id].label; }).join('='), ups: [], dns: [] };
    });
    function ceOffsets(ce) {
      var a = gItem[ce.a.i], b = gItem[ce.b.i], so = 0, to = 0;
      ce.arrows.forEach(function (x) { so += a.offs[x.s]; to += b.offs[x.t]; });
      return [so / ce.arrows.length, to / ce.arrows.length];
    }
    /* hubs: a member with HUB or more incoming arrows from outside its class */
    var hubG = {}, hubOff = {};
    model.groups.forEach(function (g) {
      var best = -1;
      g.members.forEach(function (id) {
        var ext = model.inn[id].filter(function (a) { return !a.inGroup; }).length;
        if (ext >= HUB && ext > best) { best = ext; hubG[g.i] = true; hubOff[g.i] = gItem[g.i].offs[id]; }
      });
    });
    /* combs of leaf feeders */
    var combOf = {}, combs = [];
    model.groups.forEach(function (b) {
      if (!hubG[b.i]) return;
      var leaves = b.inn.filter(function (ce) { return ce.a.size === 1 && !ce.a.inn.length && ce.a.out.length === 1; }).map(function (ce) { return ce.a; });
      if (leaves.length < COMB_MIN) return;
      var c = { t: 'c', hub: b, leaves: leaves, id: 'c:' + b.i, ups: [], dns: [], rank: b.rank - 1, key: 0 };
      buildComb(c, leaves.map(function (g) { return gItem[g.i]; }), Math.min(avail * 0.74, 700), KI, nodes);
      combs.push(c);
      leaves.forEach(function (g) { combOf[g.i] = c; });
    });
    var connected = model.groups.filter(function (g) { return !g.isolated && !combOf[g.i]; });
    var R = model.maxRank + 1;
    function itemsOfRank() {
      var by = [];
      for (var r = 0; r < R; r++) by.push([]);
      connected.forEach(function (g) { by[g.rank].push(gItem[g.i]); });
      combs.forEach(function (c) { by[c.rank].push(c); });
      return by;
    }
    function itemOf(g) { return combOf[g.i] || gItem[g.i]; }

    /* ---- coarse ordering on ranks, with dummies ---- */
    var coarse = itemsOfRank();
    coarse.forEach(function (lay, r) { lay.forEach(function (it) { it.L = r; it.ups = []; it.dns = []; }); });
    var linkedC = {};
    model.ces.forEach(function (ce) {
      var A0 = itemOf(ce.a), B0 = itemOf(ce.b), off = ceOffsets(ce);
      if (A0.t === 'c') { var kk = A0.id + '>' + B0.id; if (linkedC[kk]) return; linkedC[kk] = 1; link(A0, B0, 0, hubOff[ce.b.i]); return; }
      var prev = A0, prevOff = off[0];
      for (var L = ce.a.rank + 1; L < ce.b.rank; L++) {
        var d = { t: 'd', ce: ce, L: L, w: 0, h: 0, id: 'd' + ce.i + '_' + L, ups: [], dns: [] };
        coarse[L].push(d); link(prev, d, prevOff, 0); prev = d; prevOff = 0;
      }
      link(prev, B0, prevOff, off[1]);
    });
    /* initial order: depth-first discovery from the busiest objects, top rank first */
    var starts = [];
    coarse.forEach(function (lay) { lay.forEach(function (it) { if (it.t !== 'd') starts.push(it); }); });
    starts.sort(function (a, b) { return b.L - a.L || (b.ups.length + b.dns.length) - (a.ups.length + a.dns.length) || cmp(a.id, b.id); });
    var seen = {}, init = coarse.map(function () { return []; });
    starts.forEach(function (s0) {
      var st = [s0];
      while (st.length) {
        var it = st.pop();
        if (seen[it.id]) continue;
        seen[it.id] = 1; init[it.L].push(it);
        var nb = it.ups.concat(it.dns).map(function (n) { return n.it; }).sort(function (a, b) { return cmp(b.id, a.id); });
        nb.forEach(function (x) { if (!seen[x.id]) st.push(x); });
      }
    });
    coarse = init;
    orderByComponents(coarse, SWEEPS);
    var keyOf = {};
    coarse.forEach(function (lay) { lay.forEach(function (it, i) { keyOf[it.id] = lay.length > 1 ? i / (lay.length - 1) : 0.5; }); });
    var coarseDns = {};
    coarse.forEach(function (lay) { lay.forEach(function (it) { coarseDns[it.id] = it.dns.length; }); });

    /* ---- refinement: ranks wider than the stage are split into sub-rows (one Hasse level each) ---- */
    function sep(a, b) {
      if (a.t === 'd' && b.t === 'd') return DGAP;
      if (a.t === 'd' || b.t === 'd') return (a.w + b.w) / 2 + DSEP;
      return (a.w + b.w) / 2 + GAP;
    }
    function rowWidth(lay) {
      var tot = 0;
      lay.forEach(function (it, i) { tot += it.w + (i ? sep(lay[i - 1], it) - (lay[i - 1].w + it.w) / 2 : 0); });
      return tot;
    }
    function resolve(lay, des) {
      var n = lay.length, Lx = new Array(n), Rx = new Array(n), i;
      for (i = 0; i < n; i++) Lx[i] = Math.max(des[i], i ? Lx[i - 1] + sep(lay[i - 1], lay[i]) : -1e9, lo + lay[i].w / 2);
      for (i = n - 1; i >= 0; i--) Rx[i] = Math.min(des[i], i < n - 1 ? Rx[i + 1] - sep(lay[i], lay[i + 1]) : 1e9, hi - lay[i].w / 2);
      for (i = 0; i < n; i++) lay[i].x = (Lx[i] + Rx[i]) / 2;
      if (!n) return;
      /* project onto the feasible set: forward (lower bound, spacing), then backward (upper bound, spacing);
         feasible whenever the packed row fits between lo and hi */
      if (rowWidth(lay) <= hi - lo) {
        for (i = 0; i < n; i++) lay[i].x = Math.max(lay[i].x, lo + lay[i].w / 2, i ? lay[i - 1].x + sep(lay[i - 1], lay[i]) : -1e9);
        for (i = n - 1; i >= 0; i--) lay[i].x = Math.min(lay[i].x, hi - lay[i].w / 2, i < n - 1 ? lay[i + 1].x - sep(lay[i], lay[i + 1]) : 1e9);
      } else {
        var x = (lo + hi) / 2 - rowWidth(lay) / 2;
        lay.forEach(function (it, k) { if (k) x += sep(lay[k - 1], it) - (lay[k - 1].w + it.w) / 2; it.x = x + it.w / 2; x += it.w; });
      }
    }
    function desired(it, mode) {
      var nb = mode === 'up' ? it.ups : mode === 'dn' ? it.dns : it.ups.concat(it.dns);
      if (!nb.length) return it.x;
      var s = 0, ws = 0;
      nb.forEach(function (n) { var w = (it.t !== 'g' || n.it.t !== 'g') ? 2 : 1; s += w * (n.it.x + n.oo - n.so); ws += w; });
      return s / ws;
    }
    function refine(K) {
      var base = [], nL = 0, r, L;
      for (r = 0; r < R; r++) { base.push(nL); nL += K[r]; }
      var Lidx = function (rk, s) { return base[rk] + (K[rk] - 1 - s); };
      var rankOf = [];
      for (r = 0; r < R; r++) for (var q = 0; q < K[r]; q++) rankOf[base[r] + q] = r;
      /* sub-rows: a comb takes the top sub-row alone; objects reached by arrows from below fill the
         lower sub-rows, sources the upper ones, at most ceil(n / rows) per sub-row */
      var sub = {};
      for (r = 0; r < R; r++) {
        var k = K[r], objs = coarse[r].filter(function (it) { return it.t !== 'd'; });
        var comb = objs.filter(function (it) { return it.t === 'c'; }), rest = objs.filter(function (it) { return it.t === 'g'; });
        comb.forEach(function (c) { sub[c.id] = 0; });
        var s0 = comb.length && k > 1 ? 1 : 0, nsub = k - s0, cap = Math.ceil(rest.length / Math.max(1, nsub)), cnt = {};
        var withIn = rest.filter(function (it) { return coarseDns[it.id] > 0; }), srcs = rest.filter(function (it) { return !coarseDns[it.id]; });
        var s = k - 1;
        withIn.forEach(function (it) {
          while (s > s0 && (cnt[s] || 0) >= cap) s--;
          sub[it.id] = s; cnt[s] = (cnt[s] || 0) + 1;
        });
        srcs.forEach(function (it) {
          var t = s0;
          while (t < k - 1 && (cnt[t] || 0) >= cap) t++;
          sub[it.id] = t; cnt[t] = (cnt[t] || 0) + 1;
        });
      }
      var layers = [];
      for (L = 0; L < nL; L++) layers.push([]);
      itemsOfRank().forEach(function (lay, rk) {
        lay.forEach(function (it) { it.ups = []; it.dns = []; it.L = Lidx(rk, sub[it.id]); it.key = keyOf[it.id]; layers[it.L].push(it); });
      });
      var lanes = {}, linked = {}, dummies = [];
      function linkOnce(a, b, ao, bo) { var kk = a.id + '>' + b.id; if (linked[kk]) return; linked[kk] = 1; link(a, b, ao, bo); }
      combs.forEach(function (c) { lanes['l' + c.hub.i + '_' + c.L] = c; });
      model.ces.forEach(function (ce) {
        var a = itemOf(ce.a), b = itemOf(ce.b), off = ceOffsets(ce), Lc;
        ce.dum = [];
        if (a.t === 'c') { if (b.L - a.L === 1) linkOnce(a, b, 0, hubOff[ce.b.i]); return; }
        if (b.L - a.L <= 1) { link(a, b, off[0], off[1]); return; }
        if (hubG[ce.b.i]) {
          for (Lc = a.L + 1; Lc < b.L; Lc++) {
            var lk = 'l' + ce.b.i + '_' + Lc;
            if (!lanes[lk]) { lanes[lk] = { t: 'd', lane: true, L: Lc, w: 0, h: 0, id: lk, key: b.key, ups: [], dns: [] }; layers[Lc].push(lanes[lk]); dummies.push(lanes[lk]); }
            ce.dum.push(lanes[lk]);
          }
          link(a, ce.dum[0], off[0], 0);
          for (var m = 0; m < ce.dum.length - 1; m++) linkOnce(ce.dum[m], ce.dum[m + 1], 0, 0);
          linkOnce(ce.dum[ce.dum.length - 1], b, 0, hubOff[ce.b.i]);
          return;
        }
        var prev = a, prevOff = off[0];
        for (Lc = a.L + 1; Lc < b.L; Lc++) {
          var f = (Lc - a.L) / (b.L - a.L);
          var d = { t: 'd', ce: ce, L: Lc, w: 0, h: 0, id: 'd' + ce.i + '_' + Lc, key: a.key + f * (b.key - a.key), ups: [], dns: [] };
          layers[Lc].push(d); dummies.push(d); ce.dum.push(d);
          link(prev, d, prevOff, 0); prev = d; prevOff = 0;
        }
        link(prev, b, prevOff, off[1]);
      });
      layers.forEach(function (lay) { lay.sort(function (x, y) { return x.key - y.key || cmp(x.id, y.id); }); });
      var oc = orderByComponents(layers, SWEEPS);

      /* x: iterated barycentres under order-preserving separation */
      layers.forEach(function (lay) {
        var x = (lo + hi) / 2 - rowWidth(lay) / 2;
        lay.forEach(function (it, i) { if (i) x += sep(lay[i - 1], it) - (lay[i - 1].w + it.w) / 2; it.x = x + it.w / 2; x += it.w; });
      });
      var top = layers.length - 1, p, L1;
      for (p = 0; p < 12; p++) {
        if (p % 2 === 0) for (L1 = top - 1; L1 >= 0; L1--) resolve(layers[L1], layers[L1].map(function (it) { return desired(it, 'up'); }));
        else for (L1 = 1; L1 <= top; L1++) resolve(layers[L1], layers[L1].map(function (it) { return desired(it, 'dn'); }));
      }
      for (p = 0; p < 4; p++) for (L1 = 0; L1 <= top; L1++) resolve(layers[L1], layers[L1].map(function (it) { return desired(it, 'both'); }));

      var over = {};
      layers.forEach(function (lay, Lq) { if (rowWidth(lay) > hi - lo + 0.5) over[rankOf[Lq]] = true; });
      return { layers: layers, nL: nL, Lidx: Lidx, over: over, nCross: oc.crossings, comps: oc.components, dummies: dummies, lanes: Object.keys(lanes).length - combs.length };
    }
    /* first guess from the objects alone, then add sub-rows to any rank whose rows overflow */
    var K = [], r;
    for (r = 0; r < R; r++) {
      var rw = 0, nr = 0, hasC = false;
      coarse[r].forEach(function (it) { if (it.t === 'g') { rw += it.w + (nr ? GAP : 0); nr++; } if (it.t === 'c') hasC = true; });
      var k0 = nr ? 1 : 0;
      while (nr && k0 < 10 && rw / k0 > avail * 0.84) k0++;
      K.push(Math.max(1, k0 + (hasC && nr ? 1 : 0)));
    }
    var RF;
    for (var iter = 0; iter < 10; iter++) {
      RF = refine(K);
      var grew = false;
      Object.keys(RF.over).forEach(function (rk) { if (K[rk] < 12) { K[rk]++; grew = true; } });
      if (!grew) break;
    }
    var layers = RF.layers, nL = RF.nL, Lidx = RF.Lidx;

    /* ---- y ---- */
    var layerY = new Array(nL), y = 20;
    var endH = 28, endFs = 17;
    var fullY = y + endH / 2 + 12; /* room for the 'terminal' caption */
    y = fullY + endH / 2 + RANK_GAP;
    for (r = R - 1; r >= 0; r--) {
      for (var s1 = 0; s1 < K[r]; s1++) {
        var Lr = Lidx(r, s1), rowH = 0, comb = false;
        layers[Lr].forEach(function (it) { if (it.t !== 'd') rowH = Math.max(rowH, it.h); if (it.t === 'c') comb = true; });
        rowH = rowH || 12;
        layerY[Lr] = y + rowH / 2;
        /* a comb carries a one-line caption under it: give the gap below it room for that line */
        y += rowH + (s1 < K[r] - 1 ? SUB_GAP : RANK_GAP) + (comb ? 12 : 0);
      }
    }
    /* shelf of objects with no recorded arrow besides A1 */
    var shelf = model.groups.filter(function (g) { return g.isolated; }).map(function (g) { return gItem[g.i]; });
    shelf.sort(function (a, b) {
      var na = nodes[a.g.members[0]], nb = nodes[b.g.members[0]];
      return (KI[na.kind] == null ? 99 : KI[na.kind]) - (KI[nb.kind] == null ? 99 : KI[nb.kind]) || cmp(na.label.toLowerCase(), nb.label.toLowerCase());
    });
    var shelfBox = null;
    if (shelf.length) {
      y += 6;
      var shelfTop = y, rows = [[]], rw2 = 0;
      shelf.forEach(function (it) {
        var add = it.w + (rows[rows.length - 1].length ? GAP : 0);
        if (rw2 + add > avail - 24 && rows[rows.length - 1].length) { rows.push([]); rw2 = 0; add = it.w; }
        rows[rows.length - 1].push(it); rw2 += add;
      });
      y += 22;
      rows.forEach(function (row, ri) {
        var tot = 0, rh = 0;
        row.forEach(function (it, i) { tot += it.w + (i ? GAP : 0); rh = Math.max(rh, it.h); });
        var x0 = (lo + hi) / 2 - tot / 2;
        row.forEach(function (it) { it.x = x0 + it.w / 2; it.y = y + rh / 2; it.shelf = true; it.row = ri; x0 += it.w + GAP; });
        y += rh + 12;
      });
      shelfBox = { x0: lo - 6, x1: hi + 6, y0: shelfTop, y1: y - 4 };
      y += RANK_GAP - 12;
    }
    var frozenY = y + endH / 2;
    var H = frozenY + endH / 2 + 34;
    layers.forEach(function (lay, L) { lay.forEach(function (it) { it.y = layerY[L]; }); });

    /* ---- positions of every method ---- */
    var pos = {};
    function place(g, it) {
      g.members.forEach(function (id) {
        var P = pill[id];
        pos[id] = { x: it.x + it.offs[id], y: it.y, w: P.w, h: P.h, fs: P.fs, g: g, item: it };
      });
    }
    connected.forEach(function (g) { place(g, gItem[g.i]); });
    shelf.forEach(function (it) { place(it.g, it); });
    combs.forEach(function (c) {
      c.cells.forEach(function (cell) {
        var it = cell.it;
        it.x = c.x + cell.dx; it.y = c.y + cell.dy; it.comb = c; it.row = cell.row;
        place(it.g, it);
        pos[it.g.members[0]].comb = c;
        pos[it.g.members[0]].row = cell.row;
      });
    });
    var ends = {
      frozen: { x: (lo + hi) / 2, y: frozenY, h: endH, fs: endFs, label: 'Frozen' },
      full: { x: (lo + hi) / 2, y: fullY, h: endH, fs: endFs, label: 'Full FT' }
    };
    ends.frozen.w = Math.ceil(measure('Frozen', endFs, 600) + 2 * PADX + 4);
    ends.full.w = Math.ceil(measure('Full FT', endFs, 600) + 2 * PADX + 4);

    return { W: W, H: H, lo: lo, hi: hi, layers: layers, gItem: gItem, pos: pos, ends: ends, shelf: shelf, shelfBox: shelfBox, combs: combs, combOf: combOf,
      K: K, R: R, crossings: RF.nCross, comps: RF.comps, dummies: RF.dummies, lanes: RF.lanes, sweeps: SWEEPS, hubG: hubG };
  }

  /* ============================================================ figure ============================================================ */
  Atlas.register('graph', function (el, A) {
    injectCSS();
    if (!window.d3) { el.appendChild(A.h('p', { class: 'muted', text: 'This figure needs D3, which did not load.' })); return; }
    var D = A.data();
    var uid = 'gr-' + (++instances);
    var coarsePtr = window.matchMedia && window.matchMedia('(hover: none)').matches;

    /* ---------- state ---------- */
    var S = { scope: 'all', covering: false, hiddenKinds: {}, hover: null, pinned: null, kfocus: null, chain: false, query: '', matches: [] };
    var model = null, G = null, svgSel = null, root = null, zoom = null, narrow = false, lastW = 0, mounted = false;
    var down = {}, up = {};
    var lastPointer = 'mouse';

    /* ---------- DOM ---------- */
    var stage = A.h('div', { class: 'stage gr-stage' });
    el.appendChild(stage);
    var kicker = A.h('div', { class: 'gr-kicker' });
    stage.appendChild(kicker);
    var title = A.h('h3', { class: 'gr-title' });
    stage.appendChild(title);
    function setTitle() {
      /* with every core method drawn, extensions, lenses and families sit beside the weight-space objects */
      title.innerHTML = S.scope === 'R' ? 'The category <span class="nw"><b>Meth</b>(Θ, θ<sub>0</sub>)</span>, as a Hasse diagram' : 'The category of methods, as a Hasse diagram';
    }
    stage.appendChild(A.h('p', {
      class: 'gr-instr', id: uid + '-instr',
      text: 'Arrows point up: an arrow M → N means N simulates M. ' +
        (coarsePtr ? 'Tap a method to light everything it simulates (below) and everything that simulates it (above); tap an arrow for its map h. '
          : 'Hover a method to light everything it simulates (below) and everything that simulates it (above); hover an arrow for its map h. Click a method to pin it and open it in the catalogue. ') +
        'Chip colour is the modification kind, as everywhere on this site; click a kind to dim it, double-click or Shift-click to isolate it.'
    }));

    var kindRow = A.h('div', { class: 'gr-row', role: 'group', 'aria-label': 'Modification kind legend and filter' });
    stage.appendChild(kindRow);
    var tools = A.h('div', { class: 'gr-tools' });
    stage.appendChild(tools);
    var listId = uid + '-names';
    var findIn = A.h('input', { type: 'search', id: uid + '-find', list: listId, placeholder: 'LoRA, VeRA, OFT…', autocomplete: 'off', spellcheck: 'false' });
    var dl = A.h('datalist', { id: listId });
    tools.appendChild(A.h('label', { class: 'gr-find', for: uid + '-find' }, ['Find', findIn]));
    tools.appendChild(dl);
    var scopeSeg = A.seg([{ value: 'all', label: 'all core' }, { value: 'R', label: 'weight-space (R)' }], 'all', function (v) { S.scope = v; S.pinned = null; S.hover = null; S.kfocus = null; S.chain = false; rebuild(); }, 'Which objects to draw');
    var arrSeg = A.seg([{ value: 'all', label: 'all' }, { value: 'cover', label: 'covering only' }], 'all', function (v) { S.covering = v === 'cover'; applyState(); renderReadout(); }, 'Which arrows to draw');
    /* each label wraps together with its control */
    tools.appendChild(A.h('span', { class: 'gr-grp' }, [A.h('span', { class: 'gr-seg-label', text: 'Objects' }), scopeSeg.el]));
    tools.appendChild(A.h('span', { class: 'gr-grp' }, [A.h('span', { class: 'gr-seg-label', text: 'Arrows' }), arrSeg.el]));
    var zoomBox = A.h('span', { class: 'gr-zoom', role: 'group', 'aria-label': 'Zoom' });
    var zIn = A.h('button', { type: 'button', class: 'btn', 'aria-label': 'Zoom in', text: '+' });
    var zOut = A.h('button', { type: 'button', class: 'btn', 'aria-label': 'Zoom out', text: '−' });
    var zReset = A.h('button', { type: 'button', class: 'btn', text: 'Reset view' });
    zoomBox.appendChild(zIn); zoomBox.appendChild(zOut); zoomBox.appendChild(zReset);

    var legend = A.h('div', { class: 'gr-legend', 'aria-label': 'Arrow legend' });
    stage.appendChild(legend);
    var readout = A.h('div', { class: 'gr-readout' });
    stage.appendChild(readout);
    var wrap = A.h('div', { class: 'gr-wrap' });
    stage.appendChild(wrap);
    var view = A.h('div', { class: 'gr-view' });
    wrap.appendChild(view);
    wrap.appendChild(zoomBox);
    var card = A.h('div', { class: 'gr-card', role: 'status', 'aria-live': 'off' });
    view.appendChild(card);
    var panel = A.h('div', { class: 'gr-panel' });
    stage.appendChild(panel);
    var hint = A.h('p', { class: 'gr-hint' });
    stage.appendChild(hint);
    var live = A.h('div', { class: 'sr-only', 'aria-live': 'polite' });
    stage.appendChild(live);
    var foot = A.h('p', { class: 'gr-foot' });
    stage.appendChild(foot);

    /* legend for rules (inline svg samples coloured by token) */
    function ruleSample(rule) {
      var R0 = RULES[rule], col = 'var(' + R0.tok + ')';
      var dash = R0.dash ? ' stroke-dasharray="' + R0.dash + '"' : '';
      var hook = R0.hook ? '<path d="M5 9 A2.4 2.4 0 0 1 0.2 9" fill="none" stroke="' + col + '" stroke-width="1.2"/>' : '';
      return '<svg width="30" height="12" viewBox="0 0 30 12" aria-hidden="true" style="color:' + col + '">' + hook +
        '<path d="M5 9 C 12 9, 16 3, 27 3" fill="none" stroke="' + col + '" stroke-width="1.3" stroke-linecap="round"' + dash + '/>' +
        '<path d="M23.5 0.6 L27 3 L23.5 5.4" fill="none" stroke="' + col + '" stroke-width="1.2" stroke-linejoin="round"/></svg>';
    }
    function buildLegend() {
      var h = '';
      RULE_ORDER.forEach(function (k) { h += '<span class="it">' + ruleSample(k) + A.esc(RULES[k].long) + '</span>'; });
      h += '<span class="it"><svg width="22" height="14" viewBox="0 0 22 14" aria-hidden="true"><rect x="1" y="3" width="13" height="10" rx="3" fill="none" stroke="var(--ink-3)" stroke-width="1"/><path d="M10 3 A5 5 0 1 1 15 8" fill="none" stroke="var(--ink-2)" stroke-width="1.1"/><path d="M15.6 5.6 L15 8.2 L12.6 7.4" fill="none" stroke="var(--ink-2)" stroke-width="1"/></svg>loop · arrow inside one family</span>';
      h += '<span class="it"><svg width="26" height="12" viewBox="0 0 26 12" aria-hidden="true"><path d="M3 4 H22 M19 1.8 L22 4 L19 6.2 M23 8.5 H4 M7 6.3 L4 8.5 L7 10.7" fill="none" stroke="var(--ink-2)" stroke-width="1.1"/></svg>mutual simulation (one rung)</span>';
      h += '<span class="it"><svg width="22" height="12" viewBox="0 0 22 12" aria-hidden="true"><rect x="1" y="1" width="20" height="10" rx="3" fill="none" stroke="var(--ink-3)" stroke-width="1.1" stroke-dasharray="3 2"/></svg>dashed chip · not a weight-space object</span>';
      h += '<span class="it"><svg width="26" height="12" viewBox="0 0 26 12" aria-hidden="true"><path d="M2 6 H24" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="1.5 3"/><path d="M10.5 3.5 L15.5 8.5 M15.5 3.5 L10.5 8.5" stroke="var(--seal)" stroke-width="1.5" stroke-linecap="round"/></svg>no arrow, by O1–O3 (when pinned)</span>';
      legend.innerHTML = h;
    }
    buildLegend();

    /* kind chips (legend + filter), counts for the current scope */
    var kindBtns = [];
    function buildKinds() {
      kindRow.innerHTML = '';
      var kc = {};
      model.methods.forEach(function (m) { var k = m.modification_kind || 'hybrid'; kc[k] = (kc[k] || 0) + 1; });
      kindBtns = [];
      A.KINDS.forEach(function (k) {
        if (!kc[k.key]) return;
        var sw = A.h('span', { class: 'sw', 'aria-hidden': 'true', style: 'color:var(' + k.token + ');background:color-mix(in srgb, var(' + k.token + ') 22%, transparent)' });
        var b = A.h('button', { type: 'button', class: 'gr-key', 'aria-pressed': String(!S.hiddenKinds[k.key]), title: 'Click to dim or restore; double-click or Shift-click to show only ' + k.name },
          [sw, A.h('span', { text: k.name }), A.h('span', { class: 'n', text: String(kc[k.key]) })]);
        b.addEventListener('click', function (e) {
          if (e.shiftKey) { A.KINDS.forEach(function (o) { S.hiddenKinds[o.key] = o.key !== k.key; }); }
          else S.hiddenKinds[k.key] = !S.hiddenKinds[k.key];
          syncKinds(); applyState();
        });
        b.addEventListener('dblclick', function (e) {
          e.preventDefault();
          A.KINDS.forEach(function (o) { S.hiddenKinds[o.key] = o.key !== k.key; });
          syncKinds(); applyState();
        });
        kindRow.appendChild(b);
        kindBtns.push({ k: k, b: b });
      });
      kindRow.appendChild(A.h('span', { class: 'gr-sep', 'aria-hidden': 'true' }));
      var all = A.h('button', { type: 'button', class: 'btn gr-small', text: 'All kinds' });
      all.addEventListener('click', function () { S.hiddenKinds = {}; syncKinds(); applyState(); });
      kindRow.appendChild(all);
    }
    function syncKinds() { kindBtns.forEach(function (o) { o.b.setAttribute('aria-pressed', String(!S.hiddenKinds[o.k.key])); }); }

    /* ---------- model + layout ---------- */
    function rebuild() {
      zt = d3.zoomIdentity;
      model = buildModel(D, S.scope);
      down = {}; up = {};
      dl.innerHTML = '';
      model.methods.slice().sort(function (a, b) { return cmp(shortName(a).toLowerCase(), shortName(b).toLowerCase()); }).forEach(function (m) {
        dl.appendChild(A.h('option', { value: shortName(m) }));
      });
      buildKinds();
      syncKinds();
      setTitle();
      S.matches = matchIds(S.query);
      render();
    }
    function sets(id) {
      if (!down[id]) {
        var d = reach(model.inn, 's', id), u = reach(model.out, 't', id);
        delete d[id]; delete u[id];
        down[id] = d; up[id] = u;
      }
      return { down: down[id], up: up[id] };
    }

    var measure = makeMeasure();

    function render() {
      if (!model) return;
      var vw = Math.floor(view.getBoundingClientRect().width) || 900;
      lastW = vw;
      narrow = vw < NARROW;
      view.classList.toggle('narrow', narrow);
      view.classList.toggle('scroll-x', narrow); /* a horizontal scroller by design, as the site marks them */
      wrap.classList.toggle('narrow', narrow);
      var W = narrow ? NARROW_W : Math.max(NARROW_W - 80, vw);
      measure = makeMeasure();
      G = layout(model, W, measure);
      draw();
      renderReadout();
      renderFoot();
      var fn = el.closest && el.closest('figure'), fnTxt = fn && fn.querySelector('.fig-n') ? fn.querySelector('.fig-n').textContent.trim() : '';
      kicker.textContent = (/^Figure\s*\d+$/.test(fnTxt) ? fnTxt : 'Exposé VIII.C') + ' · ' + plural(model.stats.objects, 'object') + ' · ' + plural(model.stats.arrows, 'arrow') + ' · ' + plural(model.stats.selfs, 'loop');
      hint.textContent = narrow ? 'Swipe sideways inside the diagram · tap a method or an arrow · + / − to zoom'
        : 'Drag to pan · ⌘/Ctrl + scroll or + / − to zoom · arrow keys move between methods, Enter pins';
      applyState();
      if (narrow && !mounted) {
        var p0 = G.pos.lora || null;
        view.scrollLeft = p0 ? Math.max(0, p0.x - view.clientWidth / 2) : (W - view.clientWidth) / 2;
      }
    }

    /* ---------- drawing ---------- */
    var line = d3.line().curve(d3.curveMonotoneY);
    function col(tok) { return 'var(' + tok + ')'; }
    function headPath(x, y, dx, dy, s) {
      /* open chevron at (x,y) pointing along (dx,dy) */
      var L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, px = -uy, py = ux, a = s || 4.2, b = a * 0.72;
      return 'M' + (x - ux * a + px * b).toFixed(2) + ',' + (y - uy * a + py * b).toFixed(2) + 'L' + x.toFixed(2) + ',' + y.toFixed(2) +
        'L' + (x - ux * a - px * b).toFixed(2) + ',' + (y - uy * a - py * b).toFixed(2);
    }

    var E = {}; /* per arrow: {g (d3 sel), hit, pts} */
    var hubOf = {};
    function draw() {
      Array.prototype.slice.call(view.children).forEach(function (n) { if (n.tagName && n.tagName.toLowerCase() === 'svg') n.remove(); });
      var W = G.W, H = G.H;
      svgSel = d3.select(view).insert('svg', ':first-child')
        .attr('viewBox', '0 0 ' + W + ' ' + H)
        .attr('width', narrow ? W : null)
        .attr('height', narrow ? H : null)
        .attr('role', 'application')
        .attr('aria-roledescription', 'diagram')
        .attr('aria-label', 'Hasse diagram of ' + model.stats.objects + ' methods and ' + model.stats.arrows + ' simulation arrows, Frozen at the bottom and Full fine-tuning at the top. Use the arrow keys to move between methods; Enter pins a method and opens it in the catalogue.')
        .attr('aria-describedby', uid + '-instr')
        .attr('tabindex', 0);
      root = svgSel.append('g').attr('class', 'gr-root');
      var gA1 = root.append('g'), gF = root.append('g'), gE = root.append('g'), gT = root.append('g'), gHit = root.append('g'), gN = root.append('g'), gFx = root.append('g').attr('class', 'gr-fx');
      E = {}; hubOf = {};
      var pos = G.pos, ends = G.ends;

      /* --- A1: the interval frame from Frozen out to the margins and up into Full FT --- */
      var fz = ends.frozen, ff = ends.full, xl = G.lo - 14, xr = G.hi + 14, rr = 16;
      [[-1, xl], [1, xr]].forEach(function (sd) {
        var s = sd[0], xm = sd[1];
        var x0 = fz.x + s * fz.w / 2, x1 = ff.x + s * ff.w / 2;
        var d = 'M' + x0 + ',' + fz.y + ' H' + (xm - s * rr) + ' Q' + xm + ',' + fz.y + ' ' + xm + ',' + (fz.y - rr) +
          ' V' + (ff.y + rr) + ' Q' + xm + ',' + ff.y + ' ' + (xm - s * rr) + ',' + ff.y + ' H' + (x1 + s * 3);
        gA1.append('path').attr('class', 'gr-a1').attr('d', d);
        gA1.append('path').attr('class', 'gr-a1hd').attr('d', headPath(x1 + s * 3, ff.y, -s, 0, 5));
      });
      var midY = (fz.y + ff.y) / 2;
      gA1.append('text').attr('class', 'gr-ann').attr('transform', 'translate(' + (xl - 4) + ',' + midY + ') rotate(-90)').attr('text-anchor', 'middle').attr('dy', '-0.1em')
        .text('A1 · Frozen → M → Full FT for every weight-space M');
      var tR = gA1.append('text').attr('class', 'gr-ann').attr('transform', 'translate(' + (xr + 4) + ',' + midY + ') rotate(90)').attr('text-anchor', 'middle').attr('dy', '-0.1em');
      mixText(tR, 'h = const at the bottom · h = ρ');
      tR.append('tspan').attr('dy', '0.3em').style('font-size', '10px').text('M');
      tR.append('tspan').attr('dy', '-0.3em').text(' (merge) at the top');

      /* ends */
      [['full', ff, 'terminal · (Θ, θ₀, id)', -1], ['frozen', fz, 'initial · (pt, ∗, θ₀)', 1]].forEach(function (e) {
        var p = e[1], g = gN.append('g').attr('class', 'gr-end-g');
        g.append('rect').attr('x', p.x - p.w / 2).attr('y', p.y - p.h / 2).attr('width', p.w).attr('height', p.h).attr('rx', 5)
          .style('fill', 'var(--paper)').style('stroke', 'var(--ink)').style('stroke-width', 1.2);
        g.append('text').attr('class', 'gr-end').attr('x', p.x).attr('y', p.y).attr('dy', '0.35em').attr('text-anchor', 'middle').style('font-size', p.fs + 'px').text(p.label);
        mixText(g.append('text').attr('class', 'gr-sub').attr('x', p.x).attr('y', p.y + e[3] * (p.h / 2 + 9)).attr('dy', e[3] > 0 ? '0.55em' : '0').attr('text-anchor', 'middle'), e[2]);
      });

      /* --- shelf frame --- */
      if (G.shelfBox) {
        var sb = G.shelfBox;
        gA1.append('path').attr('class', 'gr-shelf').attr('d', 'M' + sb.x0 + ',' + (sb.y0 + 8) + ' H' + sb.x1);
        gA1.append('text').attr('class', 'gr-ann').attr('x', (sb.x0 + sb.x1) / 2).attr('y', sb.y0 + 8).attr('dy', '0.35em').attr('text-anchor', 'middle')
          .style('paint-order', 'stroke').style('stroke', 'var(--paper-2)').style('stroke-width', 6)
          .text(plural(G.shelf.length, 'object') + ' with no recorded A2–A5 arrow');
      }

      /* --- hubs: incoming arrows bundled into one trunk --- */
      model.methods.forEach(function (m) {
        var n = model.nodes[m.id];
        var ext = model.inn[m.id].filter(function (a) { return !a.inGroup; });
        if (ext.length >= HUB && pos[m.id]) hubOf[m.id] = { n: ext.length, cx: pos[m.id].x, cy: pos[m.id].y + pos[m.id].h / 2 + 16, top: pos[m.id].y + pos[m.id].h / 2 + 1.5 };
      });

      /* --- inter-rank arrows: source top → dummies → target bottom (or trunk) --- */
      var drawn = model.arrows.filter(function (a) { return !a.self && !a.inGroup; });
      /* port offsets: spread several arrows leaving / entering one pill */
      function firstX(a) { var d = a.ce.dum; return d && d.length ? d[0].x : pos[a.t].x; }
      function lastX(a) { var d = a.ce.dum; return d && d.length ? d[d.length - 1].x : pos[a.s].x; }
      var outP = {}, inP = {};
      model.methods.forEach(function (m) {
        var o = drawn.filter(function (a) { return a.s === m.id; }).sort(function (a, b) { return firstX(a) - firstX(b) || a.i - b.i; });
        var w = pos[m.id].w, sp = o.length > 1 ? Math.min(6, (w - 12) / (o.length - 1)) : 0;
        o.forEach(function (a, k) { outP[a.i] = (k - (o.length - 1) / 2) * sp; });
        if (!hubOf[m.id]) {
          var ii = drawn.filter(function (a) { return a.t === m.id; }).sort(function (a, b) { return lastX(a) - lastX(b) || a.i - b.i; });
          var sp2 = ii.length > 1 ? Math.min(6, (w - 12) / (ii.length - 1)) : 0;
          ii.forEach(function (a, k) { inP[a.i] = (k - (ii.length - 1) / 2) * sp2; });
        }
      });
      function f2(v) { return Math.round(v * 100) / 100; }
      drawn.forEach(function (a) {
        var ps = pos[a.s], pt = pos[a.t], R0 = RULES[a.rule], d;
        var hub = hubOf[a.t];
        var tx = hub ? hub.cx : pt.x + (inP[a.i] || 0), ty = hub ? hub.cy : pt.y + pt.h / 2 + 2;
        var sx, sy;
        if (ps.comb) {
          /* leaf of a comb: tick up to the rail of its row, along the rail to the spine, up the spine */
          var c = ps.comb, rail = c.rails[ps.row], ry = c.y + rail.y, spX = c.x, topY = c.y - c.h / 2, rr = 4;
          sx = ps.x; sy = ps.y - ps.h / 2 - (R0.hook ? 4.5 : 1.5);
          var dir = spX >= sx ? 1 : -1;
          d = 'M' + f2(sx) + ',' + f2(sy) + 'V' + f2(ry + rr) + 'Q' + f2(sx) + ',' + f2(ry) + ' ' + f2(sx + dir * rr) + ',' + f2(ry) +
            'H' + f2(spX - dir * rr) + 'Q' + f2(spX) + ',' + f2(ry) + ' ' + f2(spX) + ',' + f2(ry - rr) + 'V' + f2(topY);
          var my = (topY + ty) / 2;
          d += 'C' + f2(spX) + ',' + f2(my) + ' ' + f2(tx) + ',' + f2(my) + ' ' + f2(tx) + ',' + f2(ty);
        } else {
          sx = ps.x + (outP[a.i] || 0); sy = ps.y - ps.h / 2 - (R0.hook ? 5 : 1.5);
          var pts = [[tx, ty], [tx, ty + (hub ? 9 : 7)]];
          (a.ce.dum || []).slice().reverse().forEach(function (dm) {
            if (dm.t === 'c') pts.push([dm.x, dm.y - dm.h / 2 - 2], [dm.x, dm.y + dm.h / 2 + 2]);
            else pts.push([dm.x, dm.y - 4], [dm.x, dm.y + 4]);
          });
          pts.push([sx, sy - 7], [sx, sy]);
          d = line(pts);
        }
        var g = gE.append('g').attr('class', 'gr-e' + (a.composite ? ' comp' : '') + (ps.comb ? ' leaf' : '')).style('stroke', col(R0.tok));
        g.append('path').attr('class', 'ln').attr('d', d).attr('stroke-dasharray', R0.dash);
        if (!hub) g.append('path').attr('class', 'hd').attr('d', headPath(tx, ty, 0, -1));
        if (R0.hook) g.append('path').attr('class', 'hk').attr('d', 'M' + sx + ',' + sy + ' A2.5,2.5 0 0 1 ' + (sx - 5) + ',' + sy);
        var hit = gHit.append('path').attr('class', 'gr-hit').attr('d', d);
        E[a.i] = { g: g, hit: hit, a: a };
        bindEdge(hit, a);
      });
      /* comb buses: the shared rails and spine, drawn once in ink over the individual arrows */
      (G.combs || []).forEach(function (c) {
        var hub = null;
        Object.keys(hubOf).forEach(function (id) { if (model.gOf[id] === c.hub && (!hub || hubOf[id].n > hub.n)) hub = hubOf[id]; });
        var through = model.ces.some(function (ce) { return (ce.dum || []).indexOf(c) >= 0; });
        var topY = c.y - c.h / 2, botY = through ? c.y + c.h / 2 + 2 : c.y + c.rails[c.rails.length - 1].y, dd = '';
        c.rails.forEach(function (rl) { dd += 'M' + f2(c.x + rl.x0) + ',' + f2(c.y + rl.y) + 'H' + f2(c.x + rl.x1); });
        dd += 'M' + f2(c.x) + ',' + f2(botY) + 'V' + f2(topY);
        if (hub) { var my = (topY + hub.cy) / 2; dd += 'C' + f2(c.x) + ',' + f2(my) + ' ' + f2(hub.cx) + ',' + f2(my) + ' ' + f2(hub.cx) + ',' + f2(hub.cy); }
        gT.append('path').attr('class', 'gr-trunk gr-bus').attr('d', dd);
        /* the caption states exactly what the data say about these leaves */
        var hubId = null, single = true;
        Object.keys(hubOf).forEach(function (id) { if (hubOf[id] === hub) hubId = id; });
        c.leaves.forEach(function (g) { var o = model.out[g.members[0]]; if (o.length !== 1 || o[0].t !== hubId) single = false; });
        var lab = single && hubId ? c.leaves.length + ' objects whose only arrow goes into ' + model.nodes[hubId].label
          : c.leaves.length + ' objects whose arrows all land in ' + c.hub.members.map(function (id) { return model.nodes[id].label; }).join(' ⇄ ');
        gT.append('text').attr('class', 'gr-ann').attr('x', c.x).attr('y', c.y + c.h / 2 + 14).attr('dy', '0.35em').attr('text-anchor', 'middle')
          .style('paint-order', 'stroke').style('stroke', 'var(--paper-2)').style('stroke-width', 5).text(lab);
      });
      Object.keys(hubOf).forEach(function (id) {
        var h0 = hubOf[id];
        h0.sel = gT.append('path').attr('class', 'gr-trunk').attr('d', 'M' + h0.cx + ',' + (h0.cy + 0.5) + ' V' + (h0.top + 1) + ' ' + headPath(h0.cx, h0.top + 1, 0, -1, 4.8));
        if (h0.n >= 12) {
          gT.append('text').attr('class', 'gr-ann').attr('x', h0.cx + 7).attr('y', (h0.cy + h0.top) / 2 + 4).attr('dy', '0.35em')
            .style('paint-order', 'stroke').style('stroke', 'var(--paper-2)').style('stroke-width', 4)
            .text(h0.n + ' arrows from below');
        }
      });

      /* --- objects --- */
      var groupsDrawn = model.groups.map(function (g) { return G.gItem[g.i]; });
      groupsDrawn.forEach(function (it) {
        if (it.g.size < 2) return;
        gF.append('rect').attr('class', 'gr-gframe').attr('x', it.x - it.w / 2 - 6).attr('y', it.y - it.h / 2 - 5)
          .attr('width', it.w + 12).attr('height', it.h + 10).attr('rx', 7);
      });
      /* arrows inside a class: ⇄ between neighbours, arcs otherwise */
      model.arrows.filter(function (a) { return a.inGroup; }).forEach(function (a) {
        var g = model.gOf[a.s], ia = g.members.indexOf(a.s), ib = g.members.indexOf(a.t), ps = pos[a.s], pt = pos[a.t], R0 = RULES[a.rule], d, hd;
        if (Math.abs(ia - ib) === 1) {
          var dir = ib > ia ? 1 : -1, yy = ps.y + (dir > 0 ? -4 : 4);
          var x0 = ps.x + dir * (ps.w / 2 + 3), x1 = pt.x - dir * (pt.w / 2 + 3);
          d = 'M' + x0 + ',' + yy + ' H' + x1;
          hd = headPath(x1, yy, dir, 0, 3.8);
        } else {
          var yt = ps.y - ps.h / 2 - 2, lift = 15;
          d = 'M' + ps.x + ',' + yt + ' C' + ps.x + ',' + (yt - lift) + ' ' + pt.x + ',' + (yt - lift) + ' ' + pt.x + ',' + yt;
          hd = headPath(pt.x, yt, 0, 1, 3.8);
        }
        var gg = gE.append('g').attr('class', 'gr-e in').style('stroke', col(R0.tok));
        gg.append('path').attr('class', 'ln').attr('d', d).attr('stroke-dasharray', R0.dash);
        gg.append('path').attr('class', 'hd').attr('d', hd);
        var hit = gHit.append('path').attr('class', 'gr-hit').attr('d', d).style('stroke-width', 7);
        E[a.i] = { g: gg, hit: hit, a: a };
        bindEdge(hit, a);
      });
      /* nodes */
      model.methods.forEach(function (m) {
        var n = model.nodes[m.id], p = pos[m.id], tok = kindTok(n.kind);
        var g = gN.append('g').attr('class', 'gr-n' + (n.ext ? ' ext' : '')).attr('data-id', m.id);
        var x = p.x - p.w / 2, y = p.y - p.h / 2;
        g.append('rect').attr('class', 'ring').attr('x', x - 3.5).attr('y', y - 3.5).attr('width', p.w + 7).attr('height', p.h + 7).attr('rx', 7);
        g.append('rect').attr('class', 'bg').attr('x', x).attr('y', y).attr('width', p.w).attr('height', p.h).attr('rx', 4.5);
        g.append('rect').attr('class', 'tn').attr('x', x).attr('y', y).attr('width', p.w).attr('height', p.h).attr('rx', 4.5)
          .style('fill', col(tok)).style('stroke', col(tok));
        g.append('text').attr('class', 'gr-lbl').attr('x', p.x).attr('y', p.y).attr('dy', '0.36em').attr('text-anchor', 'middle')
          .style('font-size', p.fs + 'px').text(n.label);
        /* loops: arrows from a family to itself */
        n.selfs.forEach(function (a, k) {
          var R0 = RULES[a.rule], cx = p.x + p.w / 2 - 6 - k * 9, cy = y;
          var d = 'M' + (cx - 5) + ',' + (cy - 0.5) + ' C' + (cx - 6) + ',' + (cy - 13) + ' ' + (cx + 10) + ',' + (cy - 12) + ' ' + (cx + 4.5) + ',' + (cy - 1.5);
          var gg = gE.append('g').attr('class', 'gr-e self').style('stroke', col(R0.tok));
          gg.append('path').attr('class', 'ln').attr('d', d).attr('stroke-dasharray', R0.dash);
          gg.append('path').attr('class', 'hd').attr('d', headPath(cx + 4.5, cy - 1.5, -0.45, 1, 3.4));
          var hit = gHit.append('path').attr('class', 'gr-hit').attr('d', d).style('stroke-width', 7);
          E[a.i] = { g: gg, hit: hit, a: a, mid: [cx + 2, cy - 10] };
          bindEdge(hit, a);
        });
        p.sel = g;
        bindNode(g, m.id);
      });
      svgSel.on('click', function (ev) {
        if (ev.defaultPrevented) return;
        if (ev.target === svgSel.node() || ev.target.closest('.gr-end-g') || ev.target.classList.contains('gr-gframe')) { S.pinned = null; S.hover = null; applyState(); }
      });
      svgSel.on('keydown', onKey);
      svgSel.on('focus', onFocus);
      svgSel.on('blur', function () { S.kfocus = null; applyState(); });
      svgSel.on('pointerdown.ptr', function (ev) { lastPointer = ev.pointerType || 'mouse'; });
      G.fx = gFx;
      setupZoom();
    }

    /* ---------- zoom / pan ---------- */
    var zt = d3.zoomIdentity;
    function setupZoom() {
      var W = G.W, H = G.H;
      if (narrow) { zoom = null; zt = d3.zoomIdentity; if (nk !== 1) svgSel.attr('width', Math.round(W * nk)).attr('height', Math.round(H * nk)); return; }
      zoom = d3.zoom()
        .scaleExtent([0.6, 4])
        .extent([[0, 0], [W, H]])
        .translateExtent([[-40, -40], [W + 40, H + 40]])
        .filter(function (ev) {
          if (ev.type === 'wheel') return ev.ctrlKey || ev.metaKey;
          if (ev.type === 'dblclick') return false;
          if (ev.type === 'touchstart') return ev.touches && ev.touches.length > 1;
          return !ev.button;
        })
        .on('zoom', function (ev) { zt = ev.transform; root.attr('transform', zt); if (card.classList.contains('on')) placeCard(); });
      svgSel.call(zoom);
      if (zt && zt.k !== 1) svgSel.call(zoom.transform, zt);
    }
    function dur() { return A.reducedMotion() ? 0 : 280; }
    /* phones: zoom resizes the svg and the native scroller pans it (no touch capture) */
    var nk = 1;
    function narrowZoom(k) {
      var cx = (view.scrollLeft + view.clientWidth / 2) / nk;
      nk = Math.max(0.5, Math.min(2.5, k));
      svgSel.attr('width', Math.round(G.W * nk)).attr('height', Math.round(G.H * nk));
      view.scrollLeft = Math.max(0, cx * nk - view.clientWidth / 2);
    }
    zIn.addEventListener('click', function () { if (narrow) narrowZoom(nk * 1.3); else if (zoom) svgSel.transition().duration(dur()).call(zoom.scaleBy, 1.4); });
    zOut.addEventListener('click', function () { if (narrow) narrowZoom(nk / 1.3); else if (zoom) svgSel.transition().duration(dur()).call(zoom.scaleBy, 1 / 1.4); });
    zReset.addEventListener('click', function () {
      if (narrow) {
        narrowZoom(1);
        if (G.pos.lora) view.scrollTo({ left: Math.max(0, G.pos.lora.x - view.clientWidth / 2), behavior: A.reducedMotion() ? 'auto' : 'smooth' });
      } else if (zoom) svgSel.transition().duration(dur()).call(zoom.transform, d3.zoomIdentity);
    });

    /* ---------- highlight state ---------- */
    function active() { return S.hover || (S.kfocus ? { type: 'node', id: S.kfocus } : null) || S.pinned; }
    function isOffNode(id) { var n = model.nodes[id]; return !!S.hiddenKinds[n.kind]; }
    function applyState() {
      if (!G || !svgSel) return;
      var act = active(), hotN = null, hotE = null, curId = null, chainOn = S.chain && !act;
      if (act && act.type === 'node') {
        var st = sets(act.id);
        curId = act.id;
        hotN = {}; hotE = {};
        hotN[act.id] = 1;
        Object.keys(st.down).forEach(function (k) { hotN[k] = 1; });
        Object.keys(st.up).forEach(function (k) { hotN[k] = 1; });
        model.arrows.forEach(function (a) {
          if (a.self) { if (a.s === act.id) hotE[a.i] = 1; return; }
          if ((st.down[a.s] && (st.down[a.t] || a.t === act.id)) || ((st.up[a.s] || a.s === act.id) && st.up[a.t])) hotE[a.i] = 1;
        });
      } else if (act && act.type === 'edge') {
        var a0 = model.arrows[act.i];
        hotN = {}; hotE = {};
        hotN[a0.s] = 1; hotN[a0.t] = 1; hotE[a0.i] = 1;
      } else if (chainOn) {
        hotN = {}; hotE = {};
        model.chain.arrows.concat(model.chain.inner).forEach(function (a) { hotN[a.s] = 1; hotN[a.t] = 1; hotE[a.i] = 1; });
      } else if (S.matches.length) {
        hotN = {}; hotE = {};
        S.matches.forEach(function (id) { hotN[id] = 1; });
      }
      svgSel.classed('has-focus', !!hotN);
      var match = {};
      S.matches.forEach(function (id) { match[id] = 1; });
      model.methods.forEach(function (m) {
        var p = G.pos[m.id];
        if (!p || !p.sel) return;
        p.sel.classed('hot', !!(hotN && hotN[m.id]))
          .classed('cur', m.id === curId || (S.pinned && S.pinned.type === 'node' && S.pinned.id === m.id))
          .classed('kfocus', m.id === S.kfocus)
          .classed('match', !!match[m.id] && m.id !== curId)
          .classed('off', isOffNode(m.id));
      });
      Object.keys(E).forEach(function (k) {
        var e = E[k], a = e.a;
        var off = isOffNode(a.s) || isOffNode(a.t) || (S.covering && a.composite);
        e.g.classed('hot', !!(hotE && hotE[a.i])).classed('off', off);
        e.g.style('display', S.covering && a.composite ? 'none' : null);
        e.hit.classed('off', off);
      });
      Object.keys(hubOf).forEach(function (id) {
        var h0 = hubOf[id], hot = false;
        if (hotE) model.inn[id].forEach(function (a) { if (hotE[a.i] && !a.inGroup) hot = true; });
        h0.sel.classed('hot', hot);
      });
      drawFx(act);
      renderCard(act);
    }

    /* A1 arrows and obstructions for the object in focus */
    function drawFx(act) {
      var g = G.fx;
      g.selectAll('*').remove();
      if (!act || act.type !== 'node') return;
      var id = act.id, p = G.pos[id], n = model.nodes[id];
      if (!p) return;
      if (!n.ext) {
        var fz = G.ends.frozen, ff = G.ends.full;
        var up1 = [[ff.x, ff.y + ff.h / 2 + 2], [ff.x, ff.y + ff.h / 2 + 10], [p.x, p.y - p.h / 2 - 12], [p.x, p.y - p.h / 2 - 3]];
        var dn1 = [[p.x, p.y + p.h / 2 + 3], [p.x, p.y + p.h / 2 + 12], [fz.x, fz.y - fz.h / 2 - 10], [fz.x, fz.y - fz.h / 2 - 2]];
        g.append('path').attr('class', 'a1').attr('d', line(up1));
        g.append('path').attr('class', 'a1').attr('d', headPath(ff.x, ff.y + ff.h / 2 + 2, 0, -1, 4.6)).style('stroke-dasharray', 'none');
        g.append('path').attr('class', 'a1').attr('d', line(dn1));
        g.append('path').attr('class', 'a1').attr('d', headPath(p.x, p.y + p.h / 2 + 3, 0, -1, 4.2)).style('stroke-dasharray', 'none');
      }
      /* obstructions only for a pinned method: those recorded at it first, then the others, at most OB_MAX lines */
      if (!(S.pinned && S.pinned.type === 'node' && S.pinned.id === id)) return;
      var obs = model.obst.filter(function (o) { return o.s === id; }).concat(model.obst.filter(function (o) { return o.t === id; })).slice(0, OB_MAX);
      obs.forEach(function (o) {
        var a = G.pos[o.s], b = G.pos[o.t];
        if (!a || !b) return;
        g.append('path').attr('class', 'ob').attr('d', 'M' + a.x + ',' + a.y + ' L' + b.x + ',' + b.y);
        var mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        g.append('circle').attr('class', 'obbg').attr('cx', mx).attr('cy', my).attr('r', 5.2);
        g.append('path').attr('class', 'obx').attr('d', 'M' + (mx - 2.4) + ',' + (my - 2.4) + ' L' + (mx + 2.4) + ',' + (my + 2.4) + ' M' + (mx + 2.4) + ',' + (my - 2.4) + ' L' + (mx - 2.4) + ',' + (my + 2.4));
      });
    }

    /* ---------- card ---------- */
    var tsq = Promise.resolve(), cardKey = null, cache = {};
    /* forget MathJax's records of math that is about to be replaced, so repeated hovers do not pile them up */
    function clearTS(node) { try { if (window.MathJax && MathJax.typesetClear) MathJax.typesetClear([node]); } catch (e) { /* ignore */ } }
    function mathText(s) { return A.esc(s); } /* notes carry their own \( \) delimiters */
    function nodeCard(id, pinned) {
      var n = model.nodes[id], m = n.m, st = sets(id), code = CODES[n.code] || CODES.R, tok = kindTok(n.kind);
      var nd = Object.keys(st.down).length, nu = Object.keys(st.up).length;
      var h = '<div class="gr-c-h"><span class="gr-c-sw" style="color:var(' + tok + ');background:color-mix(in srgb, var(' + tok + ') 22%, transparent)"></span><span class="t">' + A.esc(n.label) + '</span></div>';
      if (m.full_name && m.full_name !== n.label) h += '<div class="gr-c-full">' + A.esc(m.full_name) + '</div>';
      h += '<div class="gr-c-meta"><b>' + A.esc(A.kindName(n.kind)) + '</b> · kind <b>' + A.esc(code.label) + '</b> (' + A.esc(code.name) + ')' + (m.year ? ' · ' + A.esc(m.year) : '') + (m.hf_peft ? ' · in HF PEFT' : '') + '</div>';
      h += '<div class="gr-c-sets">Simulates <b>' + nd + '</b> ' + (nd === 1 ? 'object' : 'objects') + ' below it; <b>' + nu + '</b> above it ' + (nu === 1 ? 'simulates' : 'simulate') + ' it.' +
        (n.ext ? ' <span class="gr-c-note">Not a weight-space object: it ' + code.gloss + '</span>' : ' <span class="gr-c-note">Kind R: ' + code.gloss + '</span>') + '</div>';
      var outs = model.out[id].concat(n.selfs);
      if (outs.length) {
        h += '<div class="gr-c-sec">Arrows out · the map h</div><ul class="gr-c-list">';
        outs.slice(0, 5).forEach(function (a) {
          var R0 = RULES[a.rule];
          h += '<li><div class="k"><b style="color:var(' + R0.tok + ')">' + a.rule + '</b> → ' + A.esc(model.nodes[a.t].label) + (a.self ? ' (same family)' : '') + (a.composite ? ' · composite' : '') + '</div>' +
            '<div class="m">\\(' + A.esc(a.h) + '\\)</div></li>';
        });
        if (outs.length > 5) h += '<li class="k">+ ' + (outs.length - 5) + ' more on the arrows themselves</li>';
        h += '</ul>';
      }
      var ins = model.inn[id].length;
      var verb = coarsePtr || lastPointer === 'touch' ? 'tap' : 'hover';
      if (ins) h += '<div class="gr-c-meta">' + plural(ins, 'arrow') + ' in' + (ins > 1 ? '; ' + verb + ' one for its h' : ': ' + verb + ' it for its h') + '.</div>';
      var obs = model.obst.filter(function (o) { return o.s === id; }).concat(model.obst.filter(function (o) { return o.t === id; }));
      if (obs.length) {
        h += '<div class="gr-c-sec" style="color:var(--seal)">No arrow · ' + obs.length + ' recorded ' + (obs.length === 1 ? 'test' : 'tests') + '</div><ul class="gr-c-list">';
        var lim = pinned ? 4 : 3;
        obs.slice(0, lim).forEach(function (o) {
          h += '<li><div class="k"><b style="color:var(--seal)">' + A.esc(o.test) + '</b> ' + A.esc(model.nodes[o.s].label) + ' ↛ ' + A.esc(model.nodes[o.t].label) + '</div>' +
            (pinned ? '<div class="why">' + mathText(clean(o.reason)) + (model.out[o.s].some(function (a) { return a.t === o.t; }) ? ' <i>An arrow between other members of these families is also recorded; this test concerns the case named here.</i>' : '') + '</div>' : '') + '</li>';
        });
        if (obs.length > lim) h += '<li class="k">+ ' + (obs.length - lim) + ' more in the catalogue' + (pinned ? '; dotted lines mark the first ' + Math.min(obs.length, OB_MAX) : '') + '</li>';
        else if (pinned) h += '<li class="k">Dotted lines in the diagram mark these pairs.</li>';
        h += '</ul>';
      }
      if (pinned) h += '<div class="gr-c-acts"><button type="button" class="btn primary" data-act="cat">Open in catalogue ↓</button><button type="button" class="btn" data-act="clear">Unpin</button></div>';
      else h += '<div class="gr-c-hint">' + (coarsePtr || lastPointer === 'touch' ? 'Tap again to pin it and open it in the catalogue' : 'Click to pin it and open it in the catalogue') + '</div>';
      return h;
    }
    function edgeCard(a, pinned) {
      var R0 = RULES[a.rule], s = model.nodes[a.s], t = model.nodes[a.t];
      var h = '<div class="gr-c-rule" style="color:var(' + R0.tok + ')">' + A.esc(R0.long) + (a.composite ? ' · composite' : '') + '</div>';
      h += '<div class="gr-c-h"><span class="t">' + A.esc(s.label) + '</span><span class="arr">→</span><span class="t">' + A.esc(t.label) + '</span></div>';
      h += '<div class="gr-c-math">\\[' + A.esc(a.h) + '\\]</div>';
      if (a.note) h += '<div class="gr-c-note">' + mathText(clean(a.note)) + '</div>';
      var extra = [];
      if (a.self) extra.push('A loop: both ends are members of one family, at different hyperparameters.');
      else if (a.inGroup) extra.push('Inside a class of mutually simulating families: arrows run both ways, so the Hasse diagram draws them on one rung.');
      if (a.composite) extra.push('A longer path of arrows also reaches it, so the covering view hides it.');
      extra.push('Read \\(\\rho_N\\circ h=\\rho_M\\) with \\(M\\) = ' + A.esc(s.label) + ', \\(N\\) = ' + A.esc(t.label) + ': ' + A.esc(t.label) + ' simulates ' + A.esc(s.label) + '.');
      h += '<div class="gr-c-meta">' + extra.join(' ') + '</div>';
      if (pinned) h += '<div class="gr-c-acts"><button type="button" class="btn" data-act="clear">Close</button></div>';
      return h;
    }
    var anchor = null;
    function renderCard(act) {
      var target = narrow ? panel : card;
      if (!act) {
        card.classList.remove('on', 'pinned'); panel.classList.remove('on'); cardKey = null; return;
      }
      var pinned = !!(S.pinned && act === S.pinned);
      var key = (act.type === 'node' ? 'n:' + act.id : 'e:' + act.i) + (pinned ? ':p' : '') + ':' + S.scope + (narrow ? ':n' : '');
      (narrow ? card : panel).classList.remove('on');
      if (key !== cardKey) {
        cardKey = key;
        clearTS(target);
        if (cache[key]) target.innerHTML = cache[key];
        else {
          target.innerHTML = act.type === 'node' ? nodeCard(act.id, pinned) : edgeCard(model.arrows[act.i], pinned);
          var k0 = key;
          tsq = tsq.then(function () { return A.typeset(target); }).then(function () {
            if (cardKey === k0 && target.querySelector('mjx-container')) cache[k0] = target.innerHTML;
            if (cardKey === k0 && !narrow) placeCard();
          });
        }
      }
      if (narrow && !target.querySelector('.gr-x')) target.insertBefore(A.h('button', { type: 'button', class: 'gr-x', 'data-act': 'clear', 'aria-label': 'Close', text: '×' }), target.firstChild);
      target.classList.add('on');
      target.classList.toggle('pinned', pinned);
      anchor = act;
      if (!narrow) placeCard();
    }
    var lastPt = null;
    function placeCard() {
      if (narrow || !anchor) return;
      var vr = view.getBoundingClientRect(), bx, by, bw = 0, bh = 0;
      if (anchor.type === 'node') {
        var p = G.pos[anchor.id];
        if (!p || !p.sel) return;
        var r = p.sel.node().getBoundingClientRect();
        bx = r.left - vr.left; by = r.top - vr.top; bw = r.width; bh = r.height;
      } else if (lastPt && S.hover === anchor) { bx = lastPt[0] - vr.left; by = lastPt[1] - vr.top; }
      else {
        var e = E[anchor.i];
        if (!e) return;
        var rr = e.g.node().getBoundingClientRect();
        bx = rr.left - vr.left + rr.width / 2; by = rr.top - vr.top + rr.height / 2;
      }
      var cw = card.offsetWidth, ch = card.offsetHeight, VW = view.clientWidth, VH = view.clientHeight, gap = 14, left, top;
      if (anchor === S.pinned && anchor.type === 'node' && !S.hover) {
        /* a pinned card docks on the far side, so the lit up- and down-sets stay visible */
        left = bx + bw / 2 < VW / 2 ? VW - cw - 6 : 6;
        top = Math.max(4, Math.min(VH - ch - 4, by - 40));
        card.style.transform = 'translate(' + Math.round(left) + 'px,' + Math.round(top) + 'px)';
        return;
      }
      if (bx + bw + gap + cw <= VW - 4) left = bx + bw + gap;
      else if (bx - gap - cw >= 4) left = bx - gap - cw;
      if (left != null) top = by + bh / 2 - ch / 2;
      else {
        left = Math.max(4, Math.min(VW - cw - 4, bx + bw / 2 - cw / 2));
        top = by + bh + gap;
        if (top + ch > VH - 4) top = by - ch - gap;
      }
      top = Math.max(4, Math.min(VH - ch - 4, top));
      card.style.transform = 'translate(' + Math.round(left) + 'px,' + Math.round(top) + 'px)';
    }
    function cardClick(ev) {
      var b = ev.target.closest('button[data-act]');
      if (!b) return;
      if (b.getAttribute('data-act') === 'clear') { S.pinned = null; S.hover = null; applyState(); svgSel && svgSel.node().focus({ preventScroll: true }); }
      if (b.getAttribute('data-act') === 'cat') {
        var cat = document.getElementById('catalog');
        if (cat) cat.scrollIntoView({ behavior: A.reducedMotion() ? 'auto' : 'smooth', block: 'start' });
      }
    }
    card.addEventListener('click', cardClick);
    /* the phone sheet is fixed to the viewport: show it only while the diagram is on screen */
    if ('IntersectionObserver' in window) new IntersectionObserver(function (es) { es.forEach(function (e) { panel.classList.toggle('away', !e.isIntersecting); }); }).observe(view);
    panel.addEventListener('click', cardClick);

    /* ---------- interaction ---------- */
    function announce(id) {
      var n = model.nodes[id], st = sets(id);
      live.textContent = n.label + ', ' + A.kindName(n.kind) + ', kind ' + (CODES[n.code] || CODES.R).label + '. Simulates ' + Object.keys(st.down).length + ', simulated by ' + Object.keys(st.up).length + '. ' +
        model.out[id].length + ' arrows out, ' + model.inn[id].length + ' in.';
    }
    function select(id) {
      S.pinned = { type: 'node', id: id };
      S.hover = null;
      applyState();
      document.dispatchEvent(new CustomEvent('atlas:select-method', { detail: { id: id, source: 'graph' } }));
    }
    function bindNode(g, id) {
      g.on('pointerenter', function (ev) {
        lastPointer = ev.pointerType || 'mouse';
        if (ev.pointerType === 'touch') return;
        S.hover = { type: 'node', id: id }; applyState();
      });
      g.on('pointerleave', function (ev) { if (ev.pointerType === 'touch') return; if (S.hover && S.hover.type === 'node' && S.hover.id === id) { S.hover = null; applyState(); } });
      g.on('click', function (ev) {
        ev.stopPropagation();
        var touch = lastPointer === 'touch' || lastPointer === 'pen';
        if (touch && !(S.pinned && S.pinned.type === 'node' && S.pinned.id === id)) {
          S.pinned = { type: 'node', id: id }; S.hover = null; applyState();
          document.dispatchEvent(new CustomEvent('atlas:select-method', { detail: { id: id, source: 'graph' } }));
          return;
        }
        select(id);
      });
    }
    function bindEdge(hit, a) {
      hit.on('pointerenter', function (ev) {
        lastPointer = ev.pointerType || 'mouse';
        if (ev.pointerType === 'touch') return;
        lastPt = [ev.clientX, ev.clientY];
        S.hover = { type: 'edge', i: a.i }; applyState();
      });
      hit.on('pointermove', function (ev) { if (ev.pointerType === 'touch') return; lastPt = [ev.clientX, ev.clientY]; placeCard(); });
      hit.on('pointerleave', function (ev) { if (ev.pointerType === 'touch') return; if (S.hover && S.hover.type === 'edge' && S.hover.i === a.i) { S.hover = null; applyState(); } });
      hit.on('click', function (ev) {
        ev.stopPropagation();
        lastPt = [ev.clientX, ev.clientY];
        S.pinned = { type: 'edge', i: a.i }; S.hover = null; applyState();
      });
    }

    /* keyboard: spatial navigation between objects */
    function navList() {
      return model.methods.filter(function (m) { return G.pos[m.id] && !isOffNode(m.id); }).map(function (m) { var p = G.pos[m.id]; return { id: m.id, x: p.x, y: p.y }; });
    }
    function onFocus() {
      var kb = true;
      try { kb = svgSel.node().matches(':focus-visible'); } catch (e) {}
      if (!kb) return;
      var start = (S.pinned && S.pinned.type === 'node' && S.pinned.id) || (G.pos.lora && !isOffNode('lora') ? 'lora' : (navList()[0] || {}).id);
      if (start) { S.kfocus = start; applyState(); ensureVisible(start); announce(start); }
    }
    function onKey(ev) {
      var list = navList();
      if (!list.length) return;
      var cur = S.kfocus && G.pos[S.kfocus] ? G.pos[S.kfocus] : null, nx = null;
      if (ev.key === 'Escape') { S.kfocus = null; S.pinned = null; S.hover = null; applyState(); return; }
      if ((ev.key === 'Enter' || ev.key === ' ') && S.kfocus) { ev.preventDefault(); select(S.kfocus); return; }
      if (!cur) { if (/^Arrow/.test(ev.key)) { ev.preventDefault(); S.kfocus = 'lora' in G.pos ? 'lora' : list[0].id; applyState(); } return; }
      var same = function (o) { return Math.abs(o.y - cur.y) < 2; };
      if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') {
        var s = ev.key === 'ArrowRight' ? 1 : -1;
        var row = list.filter(same).filter(function (o) { return s * (o.x - cur.x) > 1; }).sort(function (a, b) { return s * (a.x - b.x); });
        if (row.length) nx = row[0].id;
      } else if (ev.key === 'ArrowUp' || ev.key === 'ArrowDown') {
        var s2 = ev.key === 'ArrowUp' ? -1 : 1;
        var cand = list.filter(function (o) { return s2 * (o.y - cur.y) > 2; });
        if (cand.length) {
          var ny = cand.reduce(function (b, o) { return s2 * (o.y - b) < 0 ? o.y : b; }, cand[0].y);
          var rowN = cand.filter(function (o) { return Math.abs(o.y - ny) < 2; }).sort(function (a, b) { return Math.abs(a.x - cur.x) - Math.abs(b.x - cur.x); });
          nx = rowN[0].id;
        }
      } else if (ev.key === 'Home' || ev.key === 'End') {
        var r2 = list.filter(same).sort(function (a, b) { return a.x - b.x; });
        nx = ev.key === 'Home' ? r2[0].id : r2[r2.length - 1].id;
      } else return;
      ev.preventDefault();
      if (nx) { S.kfocus = nx; applyState(); ensureVisible(nx); announce(nx); }
    }
    function ensureVisible(id) {
      var p = G.pos[id];
      if (!p) return;
      if (narrow) {
        var x = p.x * nk, l = view.scrollLeft, w = view.clientWidth;
        if (x < l + 40 || x > l + w - 40) view.scrollTo({ left: Math.max(0, x - w / 2), behavior: A.reducedMotion() ? 'auto' : 'smooth' });
      } else if (zt.k > 1.001 && zoom) {
        var sx = p.x * zt.k + zt.x, sy = p.y * zt.k + zt.y;
        if (sx < 30 || sx > G.W - 30 || sy < 30 || sy > G.H - 30) svgSel.transition().duration(dur()).call(zoom.translateTo, p.x, p.y);
      }
    }

    /* search */
    function matchIds(q) {
      q = String(q || '').trim().toLowerCase();
      if (!q || !model) return [];
      var exact = [], pre = [], sub = [];
      model.methods.forEach(function (m) {
        var lab = shortName(m).toLowerCase(), nm = String(m.name || '').toLowerCase(), fl = String(m.full_name || '').toLowerCase();
        if (lab === q || nm === q || m.id === q) exact.push(m.id);
        else if (lab.indexOf(q) === 0 || nm.indexOf(q) === 0) pre.push(m.id);
        else if (lab.indexOf(q) >= 0 || nm.indexOf(q) >= 0 || fl.indexOf(q) >= 0 || m.id.indexOf(q) >= 0) sub.push(m.id);
      });
      return exact.concat(pre, sub);
    }
    function locate(id) {
      if (!G.pos[id]) return;
      S.pinned = { type: 'node', id: id }; S.hover = null;
      applyState();
      var p = G.pos[id];
      if (narrow) view.scrollTo({ left: Math.max(0, p.x * nk - view.clientWidth / 2), behavior: A.reducedMotion() ? 'auto' : 'smooth' });
      else if (zt.k > 1.001 && zoom) svgSel.transition().duration(dur()).call(zoom.translateTo, p.x, p.y);
      var r = p.sel.node().getBoundingClientRect();
      if (r.top < 60 || r.bottom > window.innerHeight - 20) p.sel.node().scrollIntoView({ block: 'center', behavior: A.reducedMotion() ? 'auto' : 'smooth' });
      announce(id);
    }
    findIn.addEventListener('input', function () {
      S.query = findIn.value; S.matches = matchIds(S.query);
      var ex = S.matches.length && model.methods.some(function (m) { return shortName(m) === findIn.value.trim() && m.id === S.matches[0]; });
      if (ex) locate(S.matches[0]); else applyState();
    });
    findIn.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); if (S.matches.length) locate(S.matches[0]); }
      if (e.key === 'Escape') { findIn.value = ''; S.query = ''; S.matches = []; applyState(); }
    });

    document.addEventListener('atlas:select-method', function (e) {
      var d = e && e.detail, id = d && (typeof d === 'string' ? d : d.id);
      if (!id || !model || (d && d.source === 'graph')) return;
      if (G && G.pos[id]) { S.pinned = { type: 'node', id: id }; applyState(); }
    });
    document.addEventListener('pointerdown', function (e) {
      if (S.hover && svgSel && !svgSel.node().contains(e.target) && !card.contains(e.target)) { S.hover = null; applyState(); }
    });
    document.addEventListener('atlas:mathjax', function () { cache = {}; cardKey = null; if (model) { applyState(); A.typeset(foot); } });

    /* ---------- readout & foot (all numbers computed) ---------- */
    var chainBtn = null;
    function renderReadout() {
      var st = model.stats, br = st.byRule;
      var shown = S.covering ? st.arrows - st.composite : st.arrows;
      readout.innerHTML = '';
      var add = function (html) { readout.appendChild(A.h('span', { html: html })); };
      var nR = model.methods.filter(function (m) { return !model.nodes[m.id].ext; }).length;
      add('<b>' + st.objects + '</b> objects' + (nR < st.objects ? ' (' + nR + ' of kind R)' : ''));
      add('<b>' + shown + '</b> ' + (S.covering ? 'covering ' : '') + 'arrows · ' + RULE_ORDER.map(function (k) { return k + ' ' + br[k]; }).join(' · '));
      add('<b>' + st.selfs + '</b> loops');
      add('<b>' + st.cycles + '</b> ⇄ classes (' + st.cycMembers + ' methods)');
      add('<b>' + st.composite + '</b> composites');
      var names = model.chain.ids.map(function (id) { return model.nodes[id].label; });
      chainBtn = A.h('button', { type: 'button', 'aria-pressed': String(S.chain), text: 'show' });
      chainBtn.addEventListener('click', function () { S.chain = !S.chain; S.pinned = null; S.hover = null; chainBtn.setAttribute('aria-pressed', String(S.chain)); applyState(); });
      var sp = A.h('span', { html: 'longest chain <b>' + st.chain + '</b> arrows: ' + A.esc(names[0] || '') + ' → … → ' + A.esc(names[names.length - 1] || '') + ' ' });
      sp.appendChild(chainBtn);
      readout.appendChild(sp);
    }
    function renderFoot() {
      var st = model.stats, R = G.R, subRows = G.K.reduce(function (s, k) { return s + k; }, 0);
      var nwh = '<span class="nw">\\(h\\).</span>';
      clearTS(foot);
      foot.innerHTML =
        A.esc('An arrow \\(h:M\\to N\\) is a pointed smooth map with \\(\\rho_N\\circ h=\\rho_M\\) (Def I.2). Every arrow drawn is an A2–A5 arrow recorded in the atlas, with its map ') + nwh + ' ' +
        A.esc('An object stands for a family of methods over ranks and pointings, and each arrow lands on the instance its \\(h\\) picks. So a loop (such as \\(\\mathrm{LoRA}_r\\hookrightarrow\\mathrm{LoRA}_{r+1}\\)) and a cycle need not be identities or isomorphisms; the ' + st.cycles + ' classes of mutually simulating families are drawn on one rung, joined by ⇄. ' +
        'Layout: longest-path layering of the quotient (' + R + ' levels, non-sinks promoted under their lowest successor), wide levels split into sub-rows (' + subRows + ' rows in all), dummy vertices on long arrows, then ' + G.sweeps + ' barycentric sweeps with adjacent swaps per connected component, keeping the order with fewest crossings (' + G.crossings + ' remain). ' +
        'Arrows into an object with ' + HUB + ' or more incoming arrows share one lane per row and end in one trunk' + (G.combs.length ? '; ' + G.combs.map(function (c) { return c.leaves.length; }).join(' and ') + ' sources whose arrows all go to one hub are drawn as a comb beneath it' : '') + '. Nothing is random, so the same data give the same picture. ' +
        st.composite + ' arrows are also reached by a longer path of arrows; “covering only” hides them and leaves the covering arrows, the Hasse diagram proper.');
      A.typeset(foot);
    }

    /* ---------- boot ---------- */
    /* test hook (headless checks drive states through it) */
    el.__graph = {
      S: S, model: function () { return model; }, G: function () { return G; },
      hover: function (h) { S.hover = h; applyState(); }, pin: function (id) { select(id); }, locate: function (id) { locate(id); },
      scope: function (v) { scopeSeg.set(v); S.scope = v; S.pinned = null; S.hover = null; rebuild(); },
      cover: function (on) { arrSeg.set(on ? 'cover' : 'all'); S.covering = !!on; applyState(); renderReadout(); },
      find: function (q) { findIn.value = q; findIn.dispatchEvent(new Event('input')); }
    };
    rebuild();
    mounted = true;
    A.onTheme(function () { cache = {}; cardKey = null; render(); });
    var onResize = A.debounce(function () {
      var w = Math.floor(view.getBoundingClientRect().width);
      if (w && Math.abs(w - lastW) >= 2) { cardKey = null; render(); }
    }, 140);
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(view); else window.addEventListener('resize', onResize);
    if (document.fonts && document.fonts.status !== 'loaded' && document.fonts.ready) document.fonts.ready.then(function () { render(); });
  });
})();
