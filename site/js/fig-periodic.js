/* fig-periodic.js — Figure 14, "The Periodic Table of PEFT" (Exposé VIII.B; theory/visuals.md §12).
   Everything is read from window.ATLAS_DATA:
     row     = method.table_row  (the kind coordinate, with reparametrisations split by how they touch θ0)
     column  = method.table_col  (the shape of the image germ; "–" = no germ, drawn as an off-table strip)
     filters = the other five coordinates (method.coords), OR within an axis and AND across axes
     arrows  = ATLAS_DATA.edges ∪ method.arrows (deduplicated), each with its rule and its map h
     no-arrows = method.obstructions (M ↛ N) and the obstructions of other methods that target M (K ↛ M)
   Hover or focus a chip: its seven coordinates, formula and budget (MathJax, cached).
   Select a chip: a mini-Hasse panel opens under its row (simulators above, simulated below, the A1
   interval Frozen → M → Full FT as a dashed spine for reparametrisations), listing every arrow with its
   h and every obstruction in seal, and the event atlas:select-method is sent so the catalogue follows.
   Below 600px of width the table becomes a filterable list grouped by row and by shape. */
(function () {
  'use strict';

  var SC = '[data-figure="periodic"]';
  var NONE = '__none';
  var instances = 0;
  var FAX = ['base_point', 'gauge', 'covariance', 'merge', 'rebasing'];
  var AX7 = ['kind', 'shape', 'base_point', 'gauge', 'covariance', 'merge', 'rebasing'];
  var AXIS_SHORT = { kind: 'Kind', shape: 'Shape', base_point: 'Base point', gauge: 'Gauge', covariance: 'Covariance', merge: 'Merge', rebasing: 'Rebasing' };

  /* rows: the kind coordinate; reparametrisations split by their coupling to θ0 (Exposé VIII.B) */
  var ROWS = [
    { key: 'R-additive', sym: 'R', grp: 'R', name: 'additive', desc: 'θ<sub>0</sub> + δ(q)', tip: 'An additive reparametrisation: ρ(q) = θ₀ + δ(q) with δ(q₀) = 0 (Def I.2).' },
    { key: 'R-action', sym: 'R', grp: 'R', name: 'action', desc: 'γ(q) · θ<sub>0</sub>', tip: 'An action-type reparametrisation: ρ(q) = γ(q)·θ₀ for a map γ into a group or monoid acting on Θ (Def I.2).' },
    { key: 'R-Hadamard', sym: 'R', grp: 'R', name: 'Hadamard', desc: 'θ<sub>0</sub> ⊙ (…)', tip: 'A reparametrisation that couples to θ₀ through an entrywise product, as HiRA does: W₀ ⊙ (𝟙𝟙ᵀ + BA).' },
    { key: 'R-selective', sym: 'R', grp: 'R', name: 'selective', desc: 'a block of θ<sub>0</sub>', tip: 'A reparametrisation that trains a subset of the coordinates of θ₀ itself: biases, norms, a head or a mask.' },
    { key: 'E', sym: 'E', name: 'pointed ext.', desc: 'a neutral point', kind: 'E' },
    { key: 'E0', sym: 'E°', name: 'unpointed ext.', desc: 'no neutral point', kind: 'E0' },
    { key: 'B', sym: 'B', name: 'backward lens', desc: 'forward map id<sub>Θ</sub>', kind: 'B' },
    { key: 'P', sym: 'P', name: 'post-hoc', desc: 'Θ<sup>N</sup> → Θ', kind: 'P' },
    { key: 'F_T', sym: 'F<sub>T</sub>', name: 'family over T', desc: 'T → Q → Θ', kind: 'F_T' }
  ];
  /* columns: the shape coordinate (axes.json order) */
  var COLS = [
    { key: 'Lin', name: 'flat' },
    { key: 'Cone', name: 'vertex θ₀' },
    { key: 'Cone*', name: 'split' },
    { key: 'Orb', name: 'orbit' },
    { key: 'Sat', name: 'torus·cone' },
    { key: 'Meet', name: 'cone ∩ orbit' },
    { key: 'Fun', name: 'function' }
  ];
  var KD = { E0: 'E°', F_T: 'F<sub>T</sub>', 'M1->': 'M1→', Minf: 'M∞', GLxGL: 'GL×GL', OxO: 'CO×CO', GLxCO: 'GL×CO', MonxGL: 'Mon×GL', MonxMon: 'Mon×Mon', Pi: 'Π<sub>X</sub>', GLxTorus: 'GL×torus', na: 'n/a', span: '→span', group: '→group', nonlinear: 'non-linear' };
  var KT = { E0: 'E°', F_T: 'F_T', 'M1->': 'M1→', Minf: 'M∞', GLxGL: 'GL×GL', OxO: 'CO×CO', GLxCO: 'GL×CO', MonxGL: 'Mon×GL', MonxMon: 'Mon×Mon', Pi: 'Π_X', GLxTorus: 'GL×torus', na: 'n/a', span: '→span', group: '→group', nonlinear: 'non-linear' };
  var GLOSS = {
    kind: { R: 'reparametrisation', E: 'pointed extension', E0: 'unpointed extension', B: 'backward lens / schedule', P: 'post-hoc operation', F_T: 'family over a base T' },
    shape: { Lin: 'flat: ρ affine', Cone: 'cone, vertex at θ<sub>0</sub>', 'Cone*': 'split: θ<sub>0</sub> off the vertex', Orb: 'group orbit', Sat: 'torus acting on a cone', Meet: 'cone ∩ orbit', Fun: 'function level only' },
    base_point: { regular: 'first-order deficit 0', apex: 'first-order deficit > 0', neutral: 'neutral point of an extension', defect: 'ρ(q<sub>0</sub>) ≠ θ<sub>0</sub>', unpointed: 'no value gives back f' },
    gauge: { none: 'ρ injective near generic points', scalar: 'one scalar slide', torus: 'a diagonal torus', GL: 'GL on internal wires', GLxTorus: 'GL with a Hadamard torus', nonlinear: 'fibre not a group orbit' },
    covariance: { GLxGL: 'basis-free', OxO: 'spectral frame', GLxCO: 'input-side orthogonal', MonxGL: 'one-sided torus', MonxMon: 'coordinate geometry', Pi: 'imposed structure', frame: 'random or data frame, in law' },
    merge: { M1: 'fuses into its own box', 'M1->': 'fuses into a neighbour', Mq: 'leaves the quantised type', Minf: 'no box-local merge', na: 'writes the full weight' },
    rebasing: { idempotent: 'restarts gain nothing', span: 'restarts reach the span', group: 'restarts reach a generated group', schedule: 'is a rebasing scheme' }
  };
  var RULE_SHORT = { A1: 'interval', A2: 'freezing', A3: 'factorisation', A4: 'product', A5: 'sum', T: 'transport', O1: 'image', O2: 'size', O3: 'first order' };
  var SHORT = {
    'MiSS (formerly Bone)': 'MiSS', 'Super-Tuning (Super / Supra)': 'Super-Tuning', 'Text-to-LoRA (T2L)': 'Text-to-LoRA',
    'Doc-to-LoRA (D2L)': 'Doc-to-LoRA', 'Polytropon (Poly)': 'Polytropon', 'Adapter (Houlsby)': 'Houlsby adapter',
    'LoRA Switch / LoRA Composite': 'LoRA Switch/Composite', 'LoRA layer replication (DUS)': 'LoRA replication (DUS)'
  };
  /* figures elsewhere on the page that already model the method */
  var ALSO = [
    { anchor: 'fig-workbench', name: 'Workbench', ids: ['lora', 'lora-fa', 'vera', 'lora-xs', 'miss', 'gralora', 'loha', 'hira', 'lokr', 'krona', 'mora', 'dora', 'ia3', 'houlsby-adapter', 'parallel-adapter'] },
    { anchor: 'fig-gram', name: 'Gram Lab', ids: ['oft', 'boft', 'hra', 'ia3', 'road', 'dora', 'hira', 'lora', 'qgoft'] },
    { anchor: 'fig-apex', name: 'Apex', ids: ['lora', 'pissa', 'qlora', 'hra'] },
    { anchor: 'fig-graph', name: 'Category graph', ids: null, coreOnly: true }
  ];

  var mctx = null;
  function tw(text, font) {
    if (!mctx) mctx = document.createElement('canvas').getContext('2d');
    mctx.font = font;
    return mctx.measureText(text).width;
  }
  /* Greek letters are mathematics: set them in the body serif italic, whatever face the label around them uses */
  function gk(html) { return String(html).replace(/[ΘθρδγΦ]/g, '<span class="gk">$&</span>'); }
  /* notes in the data sometimes name the build file they came from; the reader sees the table instead */
  function clean(s) {
    return String(s == null ? '' : s)
      .replace(/\(axes\.json ([AO]\d) examples?\)/g, function (m0, k) { return '(an example of ' + (k.charAt(0) === 'A' ? 'rule ' : 'test ') + k + ')'; })
      .replace(/^axes\.json ([AO]\d) example;\s*/, function (m0, k) { return 'The example of ' + (k.charAt(0) === 'A' ? 'rule ' : 'test ') + k + '; '; })
      .replace(/\b(the )?axes\.json's\b/g, 'the axis table’s').replace(/\b(the )?axes\.json/g, 'the axis table')
      .replace(/(^|[.!?]\s+)the axis table/g, '$1The axis table');
  }
  /* axis values as the referees accept them: the data's wording is kept except where it says more */
  var AXFIX = {
    merge: {
      Minf: { name: 'No box-local merge', add: ' The merge rules are sufficient conditions, so M∞ records a box-local obstruction; whether a whole network admits some other merge is left open.' },
      'M1->': { re: [[/never under GELU/g, 'generically not under GELU']] }
    },
    shape: { Meet: { re: [[/\(a categorical product, which exists only when the set-level fibre product is a manifold\)/, '(a categorical product, which exists when the fibre product exists in Diff, for instance as an embedded submanifold, and can fail to exist)']] } },
    rebasing: { group: { re: [[/R_infinity is the orbit of the connected Lie subgroup generated/, 'R_infinity is the orbit of the generated monoid, a connected Lie subgroup when one cycle reaches a neighbourhood of the identity']] } }
  };
  function axFixed(ax, v) {
    if (!v) return v;
    var f = AXFIX[ax] && AXFIX[ax][v.key];
    if (!f) return v;
    var d = String(v.definition || '');
    (f.re || []).forEach(function (p) { d = d.replace(p[0], p[1]); });
    return { key: v.key, name: f.name || v.name, definition: d + (f.add || '') };
  }
  function norm(s) { return String(s == null ? '' : s).toLowerCase().replace(/[\s\-_()·.,/]+/g, ''); }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  function injectCSS() {
    if (document.getElementById('css-periodic')) return;
    var s = document.createElement('style');
    s.id = 'css-periodic';
    var c = [
      'SC .pt-stage{padding:1.15rem 1.15rem 1rem}',
      'SC [hidden]{display:none!important}',
      'SC .gk{font-family:var(--f-body);font-style:italic;font-weight:400;text-transform:none;letter-spacing:0;font-size:1.14em;line-height:1}',
      'SC .pt-chk-a{color:inherit;text-decoration:underline;text-decoration-color:var(--rule);text-underline-offset:.15em}',
      'SC .pt-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide);font-variant-numeric:tabular-nums}',
      'SC .pt-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.35rem,3.4vw,1.9rem);line-height:1.12;letter-spacing:-.005em;margin:.25rem 0 0;color:var(--ink)}',
      'SC .pt-instr{font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);margin:.35rem 0 .8rem;max-width:54rem}',
      'SC .pt-bar{display:flex;flex-wrap:wrap;gap:.5rem .7rem;align-items:center;margin:0 0 .6rem}',
      'SC .pt-bar .seg button{padding:.34rem .7rem}',
      'SC .pt-search{display:inline-flex;align-items:center}',
      'SC .pt-search input{width:12.5rem;max-width:100%;font-family:var(--f-ui);font-size:.875rem;padding:.3rem .6rem;border-radius:999px}',
      'SC .pt-try{display:inline-flex;flex-wrap:wrap;align-items:center;gap:.3rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      'SC .pt-pre{font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;text-transform:none;border:1px dashed var(--ochre);color:var(--ochre-ink);background:transparent;border-radius:999px;padding:.22rem .62rem;cursor:pointer}',
      'SC .pt-pre:hover{background:var(--ochre-soft)}',
      'SC .pt-pre[aria-pressed="true"]{border-style:solid;background:var(--ochre-soft)}',
      'SC .pt-reset{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;border:1px solid var(--rule);background:transparent;color:var(--ink-2);border-radius:999px;padding:.24rem .62rem;cursor:pointer}',
      'SC .pt-reset:hover:not(:disabled){color:var(--ink);border-color:var(--ink-3)}',
      'SC .pt-reset:disabled{opacity:.45;cursor:default}',
      'SC .pt-fdet{margin:0 0 .5rem}',
      'SC .pt-fdet>summary{cursor:pointer;font-family:var(--f-ui);font-weight:500;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);padding:.3rem 0;list-style:none}',
      'SC .pt-fdet>summary::-webkit-details-marker{display:none}',
      'SC .pt-fdet>summary::before{content:"▸ ";color:var(--ink-2)}',
      'SC .pt-fdet[open]>summary::before{content:"▾ "}',
      'SC .pt-fdet.wide>summary{display:none}',
      'SC .pt-filters{display:flex;flex-wrap:wrap;gap:.45rem 1.5rem}',
      'SC .pt-ax{display:flex;flex-wrap:wrap;align-items:center;gap:.26rem}',
      'SC .pt-ax-l{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin-right:.2rem;min-width:0}',
      'SC .pt-f{display:inline-flex;align-items:center;gap:.34rem;font-family:var(--f-ui);font-weight:500;font-size:.8rem;line-height:1.2;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);border-radius:999px;padding:.18rem .52rem;cursor:pointer;white-space:nowrap}',
      'SC .pt-f:hover{border-color:var(--ink-3);color:var(--ink)}',
      'SC .pt-f .n{font-family:var(--f-mono);font-weight:400;color:var(--ink-2);font-variant-numeric:tabular-nums;font-size:.75rem}',
      'SC .pt-f[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--paper)}',
      'SC .pt-f[aria-pressed="true"] .n{color:var(--paper-3)}',
      'SC .pt-f.zero:not([aria-pressed="true"]){opacity:.42}',
      'SC .pt-readout{display:flex;flex-wrap:wrap;align-items:baseline;gap:.2rem 1rem;font-family:var(--f-ui);font-size:.8rem;line-height:1.5;color:var(--ink-2);font-variant-numeric:tabular-nums;margin:.55rem 0 .45rem;padding-top:.5rem;border-top:1px solid var(--rule)}',
      'SC .pt-readout b{font-family:var(--f-mono);font-size:.78rem;color:var(--ink);font-weight:500}',
      'SC .pt-expr{color:var(--ochre-ink);font-weight:500}',
      'SC .pt-legend{display:flex;flex-wrap:wrap;align-items:center;gap:.25rem .85rem;font-family:var(--f-ui);font-size:.8rem;color:var(--ink-2);margin:0 0 .5rem}',
      'SC .pt-legend .lt{font-weight:500;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);font-size:.75rem}',
      'SC .pt-legend .it{display:inline-flex;align-items:center;gap:.3rem;white-space:nowrap}',
      'SC .pt-legend .sw{display:inline-block;width:.72rem;height:.72rem;border-radius:2px;background:color-mix(in srgb,var(--k) 22%,var(--paper));border:1px solid var(--k);border-left-width:3px}',
      'SC .pt-legend .n{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      /* table */
      'SC .pt-wrap{overflow-x:auto;padding:2px 1px 2px}',
      'SC .pt-grid{display:grid;gap:4px;align-items:stretch}',
      'SC .pt-band{font-family:var(--f-ui);font-size:.8rem;letter-spacing:0;color:var(--ink-2);padding:.15rem 0 .1rem;line-height:1.5}',
      'SC .pt-band b{font-family:var(--f-display);font-size:1.05rem;letter-spacing:0;font-weight:var(--w-head);color:var(--tide)}',
      'SC .pt-band b sub{font-size:.7em}',
      'SC .pt-band span{color:var(--ink-2)}',
      'SC .pt-band.second{margin-top:.9rem}',
      'SC .pt-band.second b{color:var(--ink)}',
      'SC .pt-corner{align-self:end;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-2);line-height:1.4;padding:0 0 .4rem .1rem}',
      'SC .pt-ch{display:flex;flex-direction:column;justify-content:flex-end;gap:1px;padding:0 .25rem .38rem;border-bottom:1.5px solid var(--ink-3);min-width:0;cursor:default}',
      'SC .pt-ch .k{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.05rem;line-height:1.05;color:var(--ink);display:flex;align-items:baseline;gap:.3rem}',
      'SC .pt-ch .nm{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:0;color:var(--ink-2);line-height:1.25}',
      'SC .pt-ch .ct,SC .pt-rh .ct{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      'SC .pt-gl{display:block;overflow:visible}',
      'SC .pt-gl .ln{fill:none;stroke:var(--ink-3);stroke-width:1.15;stroke-linejoin:round;stroke-linecap:round}',
      'SC .pt-gl .fl{fill:var(--tide-soft)}',
      'SC .pt-gl .dsh{stroke-dasharray:2 2}',
      'SC .pt-gl .p0{fill:var(--ochre)}',
      'SC .pt-gl .vx{fill:var(--paper);stroke:var(--ink-3);stroke-width:1.15}',
      'SC .pt-rh{position:relative;display:grid;grid-template-columns:1.45rem minmax(0,1fr);align-content:start;gap:0 .25rem;padding:.3rem .2rem .3rem 0;border-top:1px solid var(--rule)}',
      'SC .pt-rh .sym{grid-row:1/span 3;font-family:var(--f-display);font-size:1.5rem;line-height:1;color:var(--ink);text-align:center;padding-top:.05rem}',
      'SC .pt-rh .sym sub{font-size:.55em}',
      'SC .pt-rh .nm{font-family:var(--f-ui);font-size:.8rem;line-height:1.2;color:var(--ink);font-weight:500}',
      'SC .pt-rh .ds{font-family:var(--f-ui);font-size:.75rem;line-height:1.3;color:var(--ink-2)}',
      'SC .pt-rh.gR .sym{color:var(--tide)}',
      'SC .pt-rh.gR::after{content:"";position:absolute;left:.7rem;width:0;border-left:1.5px solid var(--tide);opacity:.55}',
      'SC .pt-rh.gR.first::after{top:1.95rem;bottom:-5px}',
      'SC .pt-rh.gR.mid::after{top:-5px;bottom:-5px}',
      'SC .pt-rh.gR.last::after{top:-5px;bottom:.55rem}',
      'SC .pt-rh.gR.last::before{content:"";position:absolute;left:.7rem;bottom:.55rem;width:.4rem;border-top:1.5px solid var(--tide);opacity:.55}',
      'SC .pt-rh.gR:not(.first) .sym{visibility:hidden}',
      'SC .pt-cell{position:relative;min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:4px;padding:4px;display:flex;flex-wrap:wrap;align-content:flex-start;gap:3px}',
      'SC .pt-cell.empty{background:transparent;border-style:dashed;border-color:var(--rule)}',
      'SC .pt-cell.def{background:repeating-linear-gradient(135deg,transparent 0 7px,var(--rule) 7px 8px);border-color:transparent;opacity:.38}',
      'SCF .pt-cell.zero{background:transparent}',
      'SC .pt-cn{margin-left:auto;align-self:flex-end;font-family:var(--f-mono);font-size:.75rem;line-height:1;color:var(--ink-2);font-variant-numeric:tabular-nums;padding:0 1px 2px 4px;pointer-events:none}',
      'SCF .pt-cn.hit{color:var(--ochre-ink)}',
      'SC .pt-chip{display:inline-flex;align-items:center;max-width:100%;min-height:20px;font-family:var(--f-display);font-size:.78rem;font-weight:500;line-height:1.12;text-align:left;color:var(--ink);background:color-mix(in srgb,var(--k) 13%,var(--paper));border:1px solid color-mix(in srgb,var(--k) 40%,transparent);border-left:3px solid var(--k);border-radius:3px;padding:2px 5px 2px 4px;cursor:pointer;transition:opacity .16s,filter .16s;-webkit-tap-highlight-color:transparent}',
      'SC .pt-chip sup{font-size:.75em;line-height:0}',
      /* a chip wider than its cell wraps at spaces, then hyphenates a long word (the page is lang="en"), then breaks it */
      'SC .pt-chip{-webkit-hyphens:auto;hyphens:auto;-webkit-hyphenate-limit-before:4;-webkit-hyphenate-limit-after:3;hyphenate-limit-chars:8 4 3;overflow-wrap:anywhere}',
      'SC .pt-chip:hover{background:color-mix(in srgb,var(--k) 26%,var(--paper))}',
      'SC .pt-chip:focus-visible{outline:2px solid var(--ochre);outline-offset:1px}',
      'SC .pt-chip[aria-expanded="true"]{outline:2px solid var(--ink);outline-offset:1px;background:color-mix(in srgb,var(--k) 30%,var(--paper))}',
      'SC .pt-chip.ext{box-shadow:0 0 0 2px var(--paper),0 0 0 3.5px var(--tide)}',
      'SCF .pt-chip.off{opacity:.24;filter:saturate(.15)}',
      'SCF .pt-chip.on{background:color-mix(in srgb,var(--k) 30%,var(--paper));border-color:var(--k)}',
      'SC .pt-tight .pt-cell{padding:3px}',
      'SC .pt-tight .pt-chip{font-size:.75rem;min-height:18px;padding:1px 4px 1px 3px}',
      'SC .pt-dense .pt-chip{font-size:.75rem;min-height:18px;padding:1px 4px 1px 3px}',
      'SC .pt-off{display:grid;grid-template-columns:var(--lw,108px) minmax(0,1fr);gap:4px;margin-top:6px}',
      'SC .pt-off .pt-rh{border-top:1px dashed var(--rule)}',
      'SC .pt-off .pt-cell{border-style:dashed}',
      /* checks */
      'SC .pt-checks{display:grid;gap:.3rem;margin:.85rem 0 0;padding-top:.6rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
      'SC .pt-checks .hd{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      'SC .pt-ck{display:grid;grid-template-columns:1.1rem minmax(0,1fr) auto;gap:.1rem .45rem;align-items:baseline}',
      'SC .pt-ck .ic{font-family:var(--f-mono);font-weight:600;text-align:center}',
      'SC .pt-ck.ok .ic{color:var(--moss)}',
      'SC .pt-ck.bad .ic{color:var(--seal)}',
      'SC .pt-ck b{color:var(--ink);font-weight:600}',
      'SC .pt-ck button{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;border:1px solid var(--rule);background:transparent;color:var(--ink-2);border-radius:999px;padding:.14rem .55rem;cursor:pointer;white-space:nowrap}',
      'SC .pt-ck button:hover{border-color:var(--ochre);color:var(--ochre-ink)}',
      /* list (narrow) */
      'SC .pt-list{display:grid;gap:.55rem}',
      'SC .pt-lsec{border-top:1px solid var(--rule);padding-top:.45rem}',
      'SC .pt-lh{display:flex;align-items:baseline;gap:.45rem;margin-bottom:.3rem}',
      'SC .pt-lh .sym{font-family:var(--f-display);font-size:1.25rem;line-height:1;color:var(--ink);min-width:1.3rem}',
      'SC .pt-lh .sym.gR{color:var(--tide)}',
      'SC .pt-lh .sym sub{font-size:.55em}',
      'SC .pt-lh .nm{font-family:var(--f-ui);font-size:.875rem;font-weight:500;color:var(--ink)}',
      'SC .pt-lh .ds{font-family:var(--f-ui);font-size:.78rem;color:var(--ink-2)}',
      'SC .pt-lh .ct{margin-left:auto;font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      'SC .pt-lsub{display:grid;grid-template-columns:4.4rem minmax(0,1fr);gap:.4rem;align-items:start;padding:.22rem 0}',
      'SC .pt-lsub-h{display:flex;flex-direction:column;align-items:flex-start;gap:1px;font-family:var(--f-display);font-weight:var(--w-head);font-size:.95rem;line-height:1.05;color:var(--ink)}',
      'SC .pt-lsub-h .ct{font-family:var(--f-mono);font-weight:400;font-size:.75rem;color:var(--ink-2)}',
      'SC .pt-lchips{display:flex;flex-wrap:wrap;gap:4px}',
      'SC .pt-list .pt-chip{min-height:26px;font-size:.84rem;padding:3px 8px 3px 6px}',
      'SC .pt-none{font-family:var(--f-body);font-style:italic;font-size:var(--fs-sm);color:var(--ink-2);padding:1.2rem 0;text-align:center}',
      /* panel */
      'SC .pt-panel{position:relative;margin:5px 0 7px;background:var(--paper);border:1px solid var(--rule);border-top:2px solid var(--k);border-radius:6px;padding:.85rem 1rem 1rem;min-width:0;animation:pt-in .2s ease-out}',
      '@keyframes pt-in{from{opacity:0;transform:translateY(-3px)}to{opacity:1;transform:none}}',
      'SC .pt-panel::before{content:"";position:absolute;top:-8px;left:calc(var(--nx,48px) - 7px);width:12px;height:12px;background:var(--paper);border-left:2px solid var(--k);border-top:2px solid var(--k);transform:rotate(45deg);border-top-left-radius:2px}',
      'SC .pt-ph{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;gap:.5rem 1rem}',
      'SC .pt-ph-l{min-width:0;flex:1 1 18rem}',
      'SC .pt-meta{display:flex;flex-wrap:wrap;align-items:center;gap:.15rem .55rem;font-family:var(--f-ui);font-size:.8rem;letter-spacing:0;color:var(--ink-2);line-height:1.5;font-variant-numeric:tabular-nums}',
      'SC .pt-meta .sw{display:inline-block;width:.7rem;height:.7rem;border-radius:2px;background:var(--k);vertical-align:-1px;margin-right:.3rem}',
      'SC .pt-meta .hf{color:var(--ochre-ink);font-weight:500}',
      'SC .pt-ph h4{font-family:var(--f-display);font-size:1.6rem;font-weight:var(--w-head);line-height:1.1;margin:.12rem 0 0;color:var(--ink)}',
      'SC .pt-ph h4:focus{outline:none}',
      'SC .pt-ph h4:focus-visible{outline:2px solid var(--ochre);outline-offset:3px}',
      'SC .pt-ph .full{font-family:var(--f-body);font-style:italic;color:var(--ink-2);font-size:.92rem;line-height:1.35}',
      'SC .pt-ph-r{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center;justify-content:flex-end}',
      'SC .pt-ph-r a,SC .pt-ph-r button.lk{font-family:var(--f-ui);font-weight:500;font-size:.78rem;letter-spacing:.01em;text-decoration:none;border:1px solid var(--rule);color:var(--tide);background:transparent;border-radius:999px;padding:.2rem .6rem;cursor:pointer;white-space:nowrap}',
      'SC .pt-ph-r a:hover,SC .pt-ph-r button.lk:hover{border-color:var(--tide)}',
      'SC .pt-ph{padding-right:2.3rem}',
      'SC .pt-x{position:absolute;top:.65rem;right:.65rem;font-family:var(--f-ui);font-size:1.05rem;line-height:1;border:1px solid var(--rule);background:var(--paper-2);color:var(--ink-2);border-radius:999px;width:1.9rem;height:1.9rem;cursor:pointer}',
      'SC .pt-x:hover{color:var(--ink);border-color:var(--ink-3)}',
      'SC .pt-coords{display:grid;grid-template-columns:repeat(auto-fill,minmax(6.75rem,1fr));gap:.3rem;margin:.7rem 0 .2rem}',
      'SC .pt-co{display:grid;gap:1px;padding:.28rem .45rem;border:1px solid var(--rule);border-radius:4px;background:var(--paper-2);min-width:0}',
      'SC .pt-co .ax{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.01em;color:var(--ink-2)}',
      'SC .pt-co .v{font-family:var(--f-mono);font-size:.8rem;color:var(--ink);font-variant-numeric:tabular-nums}',
      'SC .pt-co .gl{font-family:var(--f-body);font-size:.8rem;line-height:1.25;color:var(--ink-2);-webkit-hyphens:auto;hyphens:auto;overflow-wrap:anywhere}',
      'SC .pt-co.hit{border-color:var(--ochre);background:var(--ochre-soft)}',
      'SC button.pt-co{font:inherit;text-align:left;color:inherit;cursor:pointer}',
      'SC button.pt-co:hover{border-color:var(--ink-3)}',
      'SC button.pt-co[aria-expanded="true"]{border-color:var(--tide);box-shadow:inset 0 -2px 0 var(--tide)}',
      'SC .pt-co .why{float:right;font-size:.75rem;letter-spacing:0;color:var(--tide);text-transform:lowercase}',
      'SC .pt-conote{margin:.35rem 0 .1rem;padding:.45rem .65rem;border-left:3px solid var(--tide);background:var(--paper-2);border-radius:0 4px 4px 0;font-family:var(--f-body);font-size:.84rem;line-height:1.5;color:var(--ink-2)}',
      'SC .pt-conote .lb{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--tide);margin-right:.45rem}',
      'SC .pt-pf{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:.2rem 1.2rem;margin:.55rem 0 .1rem}',
      'SC .pt-pf.st{grid-template-columns:minmax(0,1fr)}',
      'SC .pt-sec{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:.2rem 0 .1rem}',
      'SC .pt-fl,.pt-card .pt-fl{overflow-x:auto;overflow-y:hidden;color:var(--ink)}',
      'SC .pt-pf .pt-fl{font-size:.92rem}',
      'SC .pt-fl svg{max-width:none}',
      'SC .pt-ai .hm svg{max-width:none}',
      'SC .pt-fl mjx-container[jax="SVG"][display="true"],.pt-card .pt-fl mjx-container[jax="SVG"][display="true"]{margin:.12em 0!important;text-align:left!important;padding:0}',
      'SC .pt-pb{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:.9rem 1.3rem;margin-top:.7rem;padding-top:.7rem;border-top:1px solid var(--rule)}',
      'SC .pt-panel.nr .pt-pb{grid-template-columns:minmax(0,1fr)}',
      'SC .pt-panel.nr .pt-pf{grid-template-columns:minmax(0,1fr)}',
      'SC .pt-hd{min-width:0;position:sticky;top:4.4rem;align-self:start}',
      'SC .pt-panel.nr .pt-hd{position:static}',
      'SC .pt-hcap{font-family:var(--f-body);font-size:.8rem;line-height:1.45;color:var(--ink-2);margin-top:.35rem}',
      'SC .pt-hasse svg{display:block;width:100%;height:auto;overflow:visible}',
      'SC .pt-hasse .e path.v{fill:none;stroke:var(--ink-2);stroke-width:1.25}',
      'SC .pt-hasse .e path.ht{fill:none;stroke:transparent;stroke-width:12;pointer-events:stroke}',
      'SC .pt-hasse .e.sp path.v{stroke:var(--ink-3);stroke-dasharray:4 3}',
      'SC .pt-hasse .e.ob path.v{stroke:var(--seal);stroke-dasharray:5 3}',
      'SC .pt-hasse .e.ob path.sl{stroke:var(--seal);stroke-width:1.6}',
      'SC .pt-hasse .e.mo path.v{stroke:var(--ink-3);stroke-dasharray:2 3}',
      'SC .pt-hasse .e.hl path.v{stroke:var(--tide);stroke-width:2.1}',
      'SC .pt-hasse .e.ob.hl path.v{stroke:var(--seal);stroke-width:2.1}',
      'SC .pt-hasse .tg rect{fill:var(--paper);stroke:var(--rule);stroke-width:1}',
      'SC .pt-hasse .tg text{font-family:var(--f-ui);font-weight:500;font-size:12px;fill:var(--ink-2);font-variant-numeric:tabular-nums}',
      'SC .pt-hasse .tg tspan.gk{font-family:var(--f-body);font-style:italic;font-weight:400;font-size:13px}',
      'SC .pt-hasse .tg.ob rect{stroke:var(--seal);fill:var(--paper)}',
      'SC .pt-hasse .tg.ob text{fill:var(--seal-ink)}',
      'SC .pt-hasse .tg.sp text{fill:var(--ink-2)}',
      'SC .pt-hasse .tg.hl rect{stroke:var(--tide)}',
      'SC .pt-hasse .tg.hl text{fill:var(--tide)}',
      'SC .pt-hasse .n{cursor:pointer}',
      'SC .pt-hasse .n:focus{outline:none}',
      'SC .pt-hasse .n rect.b{stroke-width:1}',
      'SC .pt-hasse .n text,SC .pt-hasse .c text{font-family:var(--f-body);font-size:12px;font-weight:500;fill:var(--ink)}',
      'SC .pt-hasse .n:focus-visible rect.b,SC .pt-hasse .n:hover rect.b,SC .pt-hasse .n.hl rect.b{stroke-width:2}',
      'SC .pt-hasse .n:focus-visible rect.b{stroke:var(--ochre)}',
      'SC .pt-hasse .n.nc rect.b{stroke-dasharray:3 2}',
      'SC .pt-hasse .n.ob rect.b{fill:var(--seal-soft);stroke:var(--seal);stroke-dasharray:4 2}',
      'SC .pt-hasse .n.mr rect.b{fill:var(--paper-2);stroke:var(--ink-3);stroke-dasharray:3 2}',
      'SC .pt-hasse .n.mr text{fill:var(--ink-2);font-family:var(--f-ui);font-weight:500;font-size:12px}',
      'SC .pt-hasse .c text{font-family:var(--f-display);font-weight:var(--w-head);font-size:17px}',
      'SC .pt-hasse .ax rect{fill:none;stroke:var(--ink-3);stroke-dasharray:3 2}',
      'SC .pt-hasse .ax text{font-family:var(--f-body);font-style:italic;font-size:12px;fill:var(--ink-2)}',
      'SC .pt-hasse .ax text.s{font-family:var(--f-ui);font-style:normal;font-weight:500;font-size:12px;letter-spacing:.05em;fill:var(--ink-2)}',
      'SC .pt-al{min-width:0;display:grid;gap:.55rem;align-content:start}',
      'SC .pt-als h5{margin:0 0 .2rem;font-family:var(--f-ui);font-size:.75rem;font-weight:500;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2)}',
      'SC .pt-als h5 b{font-family:var(--f-mono);color:var(--ink);font-weight:500}',
      'SC .pt-als.ob h5{color:var(--seal-ink)}',
      'SC .pt-als ul{list-style:none;margin:0;padding:0;display:grid;gap:.15rem}',
      'SC .pt-ai{padding:.28rem .45rem .3rem;border-left:2px solid var(--rule);border-radius:0 4px 4px 0;font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2);min-width:0}',
      'SC .pt-ai.hl{background:var(--tide-soft);border-left-color:var(--tide)}',
      'SC .pt-ai.ob{border-left-color:var(--seal)}',
      'SC .pt-ai.ob.hl{background:var(--seal-soft)}',
      'SC .pt-ai .hh{display:flex;flex-wrap:wrap;align-items:center;gap:.2rem .4rem}',
      'SC .pt-ai .dir{font-family:var(--f-ui);font-size:.9rem;color:var(--ink-2)}',
      'SC .pt-ai.ob .dir{color:var(--seal-ink)}',
      'SC .pt-nb{font-family:var(--f-display);font-size:.84rem;font-weight:500;color:var(--ink);background:color-mix(in srgb,var(--k) 13%,var(--paper));border:1px solid color-mix(in srgb,var(--k) 40%,transparent);border-left:3px solid var(--k);border-radius:3px;padding:0 5px 0 4px;cursor:pointer;line-height:1.35}',
      'SC .pt-nb:hover{background:color-mix(in srgb,var(--k) 26%,var(--paper))}',
      'SC .pt-nb.self{cursor:default}',
      'SC .pt-rule{font-family:var(--f-ui);font-weight:500;font-size:.75rem;padding:0 .32rem;border:1px solid var(--ink-3);border-radius:3px;color:var(--ink-2);line-height:1.45;cursor:help;font-variant-numeric:tabular-nums}',
      'SC .pt-rule.ob{color:var(--seal-ink);border-color:var(--seal);background:var(--seal-soft)}',
      'SC .pt-rn{font-family:var(--f-ui);font-size:.75rem;color:var(--ink-2)}',
      'SC .pt-ai .hm{overflow-x:auto;overflow-y:hidden;color:var(--ink);padding:.1rem 0 0}',
      'SC .pt-ai .hm .lb{font-family:var(--f-body);font-style:italic;font-size:.84rem;color:var(--ink-2);margin-right:.3rem}',
      'SC .pt-ai .rs{color:var(--ink-2);padding-top:.1rem}',
      'SC .pt-ai .both{font-size:.8rem;font-style:italic;color:var(--ink-2);padding-top:.15rem;line-height:1.4}',
      'SC .pt-ai details{margin-top:.1rem}',
      'SC .pt-ai summary{cursor:pointer;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);width:max-content}',
      'SC .pt-ai summary:hover{color:var(--ink)}',
      'SC .pt-ai details>div{font-size:.8rem;line-height:1.45;color:var(--ink-2);padding:.15rem 0 .1rem}',
      'SC .pt-more{font-family:var(--f-ui);font-weight:500;font-size:.75rem;border:1px dashed var(--ink-3);background:transparent;color:var(--ink-2);border-radius:999px;padding:.16rem .62rem;cursor:pointer;margin:.2rem 0 0 .45rem}',
      'SC .pt-more:hover{color:var(--ink);border-style:solid}',
      'SC .pt-empty{font-family:var(--f-body);font-size:.84rem;color:var(--ink-2);font-style:italic}',
      'SC .pt-read{margin-top:.75rem;padding:.55rem .7rem;border-left:3px solid var(--tide);background:var(--paper-2);border-radius:0 4px 4px 0;font-family:var(--f-body);font-size:.86rem;line-height:1.5;color:var(--ink-2)}',
      'SC .pt-read .lb{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--tide);margin-right:.45rem}',
      'SC .pt-pending .pt-tx{visibility:hidden}',
      'SC .pt-panel.nr .pt-ph h4{font-size:1.35rem}',
      'SC .pt-panel.nr{padding:.75rem .7rem .85rem}',
      '@media (max-width:600px){SC .pt-stage{padding:.95rem .8rem .85rem}SC .pt-search input{width:100%}SC .pt-search{flex:1 1 10rem}SC .pt-filters{gap:.5rem}SC .pt-ax{gap:.22rem}SC .pt-f{font-size:.78rem;padding:.24rem .52rem}SC .pt-readout{font-size:.78rem}SC .pt-ck{grid-template-columns:1.1rem minmax(0,1fr)}SC .pt-ck button{grid-column:2;justify-self:start}}',
      /* hover card (lives on document.body) */
      '.pt-card{position:fixed;z-index:62;left:0;top:0;width:min(23.5rem,calc(100vw - 16px));background:var(--paper);color:var(--ink);border:1px solid var(--rule);border-top:3px solid var(--k);border-radius:6px;box-shadow:var(--shadow);padding:.6rem .75rem .6rem;font-family:var(--f-body);font-size:.84rem;line-height:1.4;pointer-events:none;opacity:0;visibility:hidden;transform:translateY(3px);transition:opacity .12s,transform .12s,visibility 0s linear .12s}',
      '.pt-card.on{opacity:1;visibility:visible;transform:none;transition:opacity .12s,transform .12s,visibility 0s}',
      '.pt-card .tp{display:flex;flex-wrap:wrap;align-items:center;gap:.1rem .5rem;font-family:var(--f-ui);font-size:.78rem;color:var(--ink-2);line-height:1.5;font-variant-numeric:tabular-nums}',
      '.pt-card .tp .sw{display:inline-block;width:.66rem;height:.66rem;border-radius:2px;background:var(--k);margin-right:.25rem;vertical-align:-1px}',
      '.pt-card .tp .hf{color:var(--ochre-ink);font-weight:500}',
      '.pt-card .nm{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.3rem;line-height:1.1;margin-top:.05rem}',
      '.pt-card .fu{font-style:italic;color:var(--ink-2);line-height:1.3}',
      '.pt-card .co{display:grid;grid-template-columns:max-content max-content minmax(0,1fr);gap:.06rem .5rem;margin:.45rem 0 .35rem;padding:.35rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);align-items:baseline}',
      '.pt-card .co .a{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-2)}',
      '.pt-card .co .v{font-family:var(--f-mono);font-size:.78rem;color:var(--ink);font-variant-numeric:tabular-nums}',
      '.pt-card .co .g{font-size:.8rem;color:var(--ink-2);line-height:1.25}',
      '.pt-card .co .hit{color:var(--ochre-ink)}',
      '.pt-card .sc{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin-top:.15rem}',
      '.pt-card .fx{max-height:7.4em;overflow:hidden}',
      '.pt-card .fx .pt-fl:nth-child(n+4){display:none}',
      '.pt-card .pt-fl{overflow:hidden}',
      '.pt-card .pt-fl svg{max-width:100%!important;height:auto}',
      '.pt-card .wait{font-family:var(--f-ui);font-size:.78rem;color:var(--ink-2);padding:.2rem 0}',
      '.pt-card .ft{margin-top:.35rem;font-family:var(--f-ui);font-size:.78rem;color:var(--ink-2);line-height:1.45;font-variant-numeric:tabular-nums}',
      '.pt-card .ft b{font-family:var(--f-mono);color:var(--ink);font-weight:500}',
      '.pt-card .ft .ob{color:var(--seal-ink)}',
      '.pt-mjstage{position:absolute;left:-9999px;top:0;width:24rem;visibility:hidden;pointer-events:none}'
    ];
    s.textContent = c.join('\n').replace(/SCF /g, SC + '.is-f ').replace(/SC /g, SC + ' ');
    document.head.appendChild(s);
  }

  /* shape glyphs (viewBox 44 x 28); the ochre dot is θ0 */
  function glyph(key, w, hgt) {
    var cone = '<path class="ln" d="M22 23 L9 6 M22 23 L35 6"/><ellipse class="ln fl" cx="22" cy="6" rx="13" ry="3.1"/>';
    var s = '<svg class="pt-gl" viewBox="0 0 44 28" width="' + w + '" height="' + hgt + '" aria-hidden="true" focusable="false">';
    switch (key) {
      case 'Lin': s += '<path class="ln fl" d="M3 21 L14 7 L41 7 L30 21 Z"/><circle class="p0" cx="22" cy="14" r="2.6"/>'; break;
      case 'Cone': s += cone + '<circle class="p0" cx="22" cy="23" r="2.6"/>'; break;
      case 'Cone*': s += cone + '<circle class="vx" cx="22" cy="23" r="2.1"/><circle class="p0" cx="15.3" cy="14.2" r="2.6"/>'; break;
      case 'Orb': s += '<ellipse class="ln fl" cx="21" cy="14" rx="15" ry="8.5"/><path class="ln" d="M12.5 6.4 l3.4 -0.9 l-1.6 3"/><circle class="p0" cx="36" cy="14" r="2.6"/>'; break;
      case 'Sat': s += cone + '<ellipse class="ln dsh" cx="22" cy="14.5" rx="10.5" ry="2.7"/><circle class="p0" cx="22" cy="23" r="2.6"/>'; break;
      case 'Meet': s += '<path class="ln" d="M22 23 L12 4 M22 23 L32 4"/><ellipse class="ln dsh" cx="22" cy="15" rx="17" ry="8"/><circle class="p0" cx="22" cy="23" r="2.6"/>'; break;
      case 'Fun': s += '<path class="ln" d="M3 19 C 9 3, 15 3, 21 14 S 33 25, 41 8"/>'; break;
      default: s += '<circle class="vx" cx="11" cy="18" r="2.1"/><circle class="vx" cx="23" cy="8.5" r="2.1"/><circle class="vx" cx="34" cy="19" r="2.1"/>';
    }
    return s + '</svg>';
  }

  /* break a long TeX formula into display lines: cut at top-level clause separators (";", "\quad",
     ",\ "), never inside braces, environments, \left…\right or brackets (where ";" stacks rows),
     then pack the clauses greedily into lines of about `budget` visible characters */
  /* MathJax does not run macros inside \text{…}; an escaped underscore there would print its backslash */
  function fixTeX(t) { return String(t || '').replace(/\\(text|textrm|mbox)\{([^{}]*)\}/g, function (m0, c, inner) { return '\\' + c + '{' + inner.replace(/\\_/g, '_') + '}'; }); }
  function vlen(t) {
    return t.replace(/\\(text|textrm|mathrm|operatorname|mathbb|mathcal|mathbf|boldsymbol|tfrac|dfrac|frac|big|Big|bigg|Bigg|left|right|quad|qquad|textstyle|displaystyle)\b/g, '')
      .replace(/\\[A-Za-z]+/g, 'x').replace(/\\./g, '').replace(/[{}_^]/g, '').replace(/\s+/g, ' ').length;
  }
  function splitTeX(src, budget) {
    var s = fixTeX(src), parts = [], depth = 0, env = 0, lr = 0, par = 0, start = 0, i = 0;
    function cut(a, b, sep) { parts.push({ t: s.slice(start, a), sep: sep }); start = b; }
    function top() { return depth === 0 && env === 0 && lr === 0 && par === 0; }
    while (i < s.length) {
      var ch = s.charAt(i);
      if (ch === '\\') {
        var m = /^\\([A-Za-z]+|.)/.exec(s.slice(i)), nm = m ? m[1] : '';
        var len = m ? m[0].length : 1;
        if (nm === 'begin') env++;
        else if (nm === 'end') env = Math.max(0, env - 1);
        else if (nm === 'left') lr++;
        else if (nm === 'right') lr = Math.max(0, lr - 1);
        else if (nm === '{' && depth === 0) par++;
        else if (nm === '}' && depth === 0) par = Math.max(0, par - 1);
        else if ((nm === 'quad' || nm === 'qquad') && top()) { cut(i, i + len, '\\quad '); }
        i += len; continue;
      }
      if (ch === '{') depth++;
      else if (ch === '}') depth = Math.max(0, depth - 1);
      else if (depth === 0 && (ch === '(' || ch === '[')) par++;
      else if (depth === 0 && (ch === ')' || ch === ']')) par = Math.max(0, par - 1);
      else if (ch === ';' && top()) { cut(i, i + 1, ';\\ '); }
      else if (ch === ',' && top() && /^,\s*\\(\s|,|;|quad|qquad)/.test(s.slice(i))) { cut(i, i + 1, ',\\ '); }
      i++;
    }
    parts.push({ t: s.slice(start), sep: '' });
    var clean = parts.map(function (p) {
      var t = p.t, prev;
      do { prev = t; t = t.replace(/^\s+|\s+$/g, '').replace(/^(\\[ ,;:!]|\\quad|\\qquad|,)+/, '').replace(/(\\[ ,;:!]|\\quad|\\qquad|,)+$/, ''); } while (t !== prev);
      return { t: t, sep: p.sep };
    }).filter(function (p) { return p.t.length > 0; });
    if (!clean.length) return [s];
    var lines = [], cur = clean[0].t, curLen = vlen(clean[0].t), sep = clean[0].sep;
    for (var k = 1; k < clean.length; k++) {
      var L = vlen(clean[k].t);
      if (curLen + 2 + L <= (budget || 60)) { cur += (sep === '\\quad ' ? ',\\quad ' : sep) + clean[k].t; curLen += 2 + L; }
      else { lines.push(cur); cur = clean[k].t; curLen = L; }
      sep = clean[k].sep;
    }
    lines.push(cur);
    var out = [];
    lines.forEach(function (l) { if (vlen(l) > (budget || 60) * 1.15) out = out.concat(wrapClause(l, budget || 60)); else out.push(l); });
    return out;
  }
  /* a clause still too long: cut between top-level math and \text{…} runs, and inside long text runs at spaces */
  function wrapClause(t, budget) {
    if (/\\(begin|left|right)\b/.test(t)) return [t];
    var toks = [], i = 0, start = 0, depth = 0;
    while (i < t.length) {
      var m = depth === 0 ? /^\\(text|textrm|mbox)\s*\{/.exec(t.slice(i)) : null;
      if (m) {
        if (i > start) toks.push({ k: 'm', s: t.slice(start, i) });
        var j = i + m[0].length, d = 1;
        while (j < t.length && d > 0) { var c = t.charAt(j); if (c === '\\') { j += 2; continue; } if (c === '{') d++; else if (c === '}') d--; j++; }
        toks.push({ k: 't', cmd: m[1], s: t.slice(i + m[0].length, j - 1) });
        i = j; start = j; continue;
      }
      var ch = t.charAt(i);
      if (ch === '\\') { i += 2; continue; }
      if (ch === '{') depth++; else if (ch === '}') depth = Math.max(0, depth - 1);
      i++;
    }
    if (start < t.length) toks.push({ k: 'm', s: t.slice(start) });
    var lines = [], cur = '', len = 0;
    function flush() { if (cur.replace(/\\[ ,;]|\s/g, '').length) lines.push(cur); cur = ''; len = 0; }
    toks.forEach(function (tk) {
      if (tk.k === 'm') {
        var L = vlen(tk.s);
        if (len && len + L > budget) flush();
        cur += tk.s; len += L;
        return;
      }
      var words = [], w = '', dd = 0;
      for (var q = 0; q < tk.s.length; q++) {
        var c2 = tk.s.charAt(q);
        if (c2 === '{') dd++; else if (c2 === '}') dd--;
        if (c2 === ' ' && dd === 0) { words.push(w); w = ''; } else w += c2;
      }
      words.push(w);
      var run = '';
      words.forEach(function (wd, k) {
        var piece = (k ? ' ' : '') + wd;
        if (len + vlen(piece) > budget && len > 0) {
          if (run) cur += '\\' + tk.cmd + '{' + run + '}';
          flush(); run = wd; len = vlen(wd);
        } else { run += piece; len += vlen(piece); }
      });
      if (run) cur += '\\' + tk.cmd + '{' + run + '}';
    });
    flush();
    return lines.length ? lines : [t];
  }

  Atlas.register('periodic', function (el, A) {
    injectCSS();
    var uid = 'pt' + (++instances);
    var D = A.data();
    var methods = (D.methods || []).filter(function (m) { return m && m.id; });
    var byId = {};
    methods.forEach(function (m) { byId[m.id] = m; });

    /* ---------- axes, rules ---------- */
    var AXV = {};
    (D.axes || []).forEach(function (a) { AXV[a.key] = {}; (a.values || []).forEach(function (v) { AXV[a.key][v.key] = axFixed(a.key, v); }); AXV[a.key].__order = (a.values || []).map(function (v) { return v.key; }); AXV[a.key].__name = a.name; });
    var RULES = {};
    (D.arrow_rules || []).forEach(function (r) { RULES[r.key] = r; });

    /* ---------- arrows (edges ∪ method.arrows, deduplicated) and obstructions ---------- */
    var OUT = {}, IN = {}, seen = {};
    function addArrow(s, t, rule, hh, note) {
      if (!s || !t) return;
      var k = s + '>' + t + '|' + (rule || '') + '|' + (hh || '');
      if (seen[k]) return;
      seen[k] = 1;
      var a = { s: s, t: t, rule: rule || '', h: hh || '', note: note || '' };
      (OUT[s] = OUT[s] || []).push(a);
      (IN[t] = IN[t] || []).push(a);
    }
    (D.edges || []).forEach(function (e) { addArrow(e.source, e.target, e.rule, e.h, e.note); });
    methods.forEach(function (m) { (m.arrows || []).forEach(function (a) { if (a) addArrow(m.id, a.target, a.rule, a.h, a.note); }); });
    var OBS_IN = {};
    methods.forEach(function (m) { (m.obstructions || []).forEach(function (o) { if (o && o.target) (OBS_IN[o.target] = OBS_IN[o.target] || []).push({ s: m.id, t: o.target, test: o.test, reason: o.reason }); }); });
    var DEG = {};
    methods.forEach(function (m) { DEG[m.id] = (OUT[m.id] || []).length + (IN[m.id] || []).length; });

    /* ---------- coordinates ---------- */
    function cv(m, ax) { var v = m.coords ? m.coords[ax] : null; return (v == null || v === '') ? NONE : String(v); }
    function keyHTML(k) { return k === NONE ? '–' : (KD[k] || A.esc(k)); }
    function keyText(k) { return k === NONE ? 'n/a' : (KT[k] || k); }
    function gloss(ax, k) {
      if (k === NONE) return 'not applicable';
      if (GLOSS[ax] && GLOSS[ax][k]) return GLOSS[ax][k];
      var v = AXV[ax] && AXV[ax][k];
      return v ? pretty(v.name) : '';
    }
    function pretty(s) {
      return A.esc(s).replace(/\btheta0'/g, 'θ<sub>0</sub>′').replace(/\btheta0\b/g, 'θ<sub>0</sub>').replace(/\btheta_res\b/g, 'θ<sub>res</sub>').replace(/\bTheta\b/g, 'Θ')
        .replace(/\brho\b/g, 'ρ').replace(/\bPi_X\b/g, 'Π<sub>X</sub>').replace(/\bS1\b/g, 'S<sub>1</sub>').replace(/\bq0\b/g, 'q<sub>0</sub>')
        .replace(/\bR_infinity\b/g, 'R<sub>∞</sub>').replace(/-&gt;/g, '→').replace(/ x (?=[A-Z(])/g, ' × ');
    }
    var ROWBY = {};
    ROWS.forEach(function (r) { ROWBY[r.key] = r; });
    function rowOf(m) {
      if (m.table_row) return m.table_row;
      var k = cv(m, 'kind');
      return k === 'R' ? 'R-additive' : (k === NONE ? '?' : k);
    }
    function colOf(m) {
      var c = m.table_col != null && m.table_col !== '' ? m.table_col : (m.coords ? m.coords.shape : null);
      if (c == null || c === '' || c === '–' || c === '-') return null;
      return String(c);
    }
    function isR(m) { return cv(m, 'kind') === 'R' || /^R-/.test(rowOf(m)); }
    /* rows/columns present in the data, in theory order, unknown keys appended */
    var rowKeys = ROWS.map(function (r) { return r.key; }), colKeys = COLS.map(function (c) { return c.key; });
    methods.forEach(function (m) {
      var r = rowOf(m), c = colOf(m);
      if (rowKeys.indexOf(r) < 0) { rowKeys.push(r); ROWBY[r] = { key: r, sym: A.esc(r), name: '', desc: '' }; }
      if (c && colKeys.indexOf(c) < 0) { colKeys.push(c); COLS.push({ key: c, name: '' }); }
    });
    var COLBY = {};
    COLS.forEach(function (c) { COLBY[c.key] = c; });
    ROWS.forEach(function (r) {
      if (!r.tip && r.kind && AXV.kind && AXV.kind[r.kind]) r.tip = AXV.kind[r.kind].name + '. ' + AXV.kind[r.kind].definition;
    });

    var hasCore = methods.some(function (m) { return m.core === true; });
    var nCore = methods.filter(function (m) { return m.core === true; }).length;

    /* ---------- labels ---------- */
    function shortName(m) { return SHORT[m.name] || m.name; }
    /* a long camel-case name may break between its parts (HyperDream·Booth) when a narrow cell needs it */
    function camelParts(w) { return w.length < 11 ? [w] : w.replace(/([a-z])(?=[A-Z])/g, function (m0, c, off) { return off + 1 >= 4 && w.length - off - 1 >= 4 ? c + '\u0000' : c; }).split('\u0000'); }
    function labelHTML(name) { return A.esc(name).replace(/\^([A-Za-z0-9])/g, '<sup>$1</sup>').replace(/[A-Za-z]{11,}/g, function (w) { return camelParts(w).join('<wbr>'); }); }
    /* the pieces a chip label can wrap into: at spaces, after hyphens, between camel-case parts */
    function labelParts(label) {
      var out = [];
      label.split(/\s+/).forEach(function (wd) { wd.replace(/-/g, '-\u0000').split('\u0000').forEach(function (p) { if (p) out = out.concat(camelParts(p)); }); });
      return out;
    }
    function plainLabel(m) { return shortName(m).replace(/\^R/g, 'ᴿ').replace(/\^A/g, 'ᴬ').replace(/\^([A-Za-z0-9])/g, '$1'); }
    function nodeLabel(id) { return byId[id] ? plainLabel(byId[id]) : id; }
    function kcol(m) { return A.kindColor(m ? m.modification_kind : null); }
    function kindTok(kind) { for (var i = 0; i < A.KINDS.length; i++) if (A.KINDS[i].key === kind) return A.KINDS[i].token; return '--c-hybrid'; }
    function rowLabelText(r) { var R = ROWBY[r]; if (!R) return r; return (R.grp === 'R' ? 'R, ' : (R.sym || '').replace(/<[^>]+>/g, '') + ', ') + R.name; }
    function yearVenue(m) {
      var v = String(m.venue || '').trim();
      if (!v) return m.year ? String(m.year) : '';
      return /\b(19|20)\d{2}\b/.test(v) ? v : (m.year ? m.year + ' · ' + v : v);
    }
    function ariaOf(m) {
      var c = colOf(m);
      return m.name + '. ' + A.kindName(m.modification_kind) + '. Row ' + rowLabelText(rowOf(m)) + ', column ' + (c ? keyText(c) : 'n/a') +
        '. Base point ' + keyText(cv(m, 'base_point')) + ', gauge ' + keyText(cv(m, 'gauge')) + ', covariance ' + keyText(cv(m, 'covariance')) +
        ', merge ' + keyText(cv(m, 'merge')) + ', rebasing ' + keyText(cv(m, 'rebasing')) + '.';
    }
    function rank(a, b) {
      var ma = byId[a], mb = byId[b];
      var ca = ma && ma.core ? 1 : 0, cb = mb && mb.core ? 1 : 0;
      if (ca !== cb) return cb - ca;
      var da = DEG[a] || 0, db = DEG[b] || 0;
      if (da !== db) return db - da;
      return nodeLabel(a).localeCompare(nodeLabel(b));
    }

    /* ---------- state ---------- */
    var st = { all: !hasCore, f: {}, q: '', sel: null, ext: null, narrow: null };
    FAX.forEach(function (ax) { st.f[ax] = {}; });
    var G = {}; /* current body geometry and elements */

    function popList() { return methods.filter(function (m) { return st.all || m.core === true; }); }
    function nSel(ax) { var n = 0; for (var k in st.f[ax]) if (st.f[ax][k]) n++; return n; }
    function anyFilter() { return FAX.some(function (ax) { return nSel(ax) > 0; }); }
    function isActive() { return anyFilter() || !!st.q; }
    function passQ(m) { if (!st.q) return true; return (m.__q || (m.__q = norm(m.name + ' ' + (m.full_name || '') + ' ' + m.id + ' ' + shortName(m)))).indexOf(st.q) >= 0; }
    function passAx(m, ax) { return nSel(ax) === 0 || !!st.f[ax][cv(m, ax)]; }
    function isLit(m, skip) {
      for (var i = 0; i < FAX.length; i++) if (FAX[i] !== skip && !passAx(m, FAX[i])) return false;
      return passQ(m);
    }

    /* ---------- MathJax: one queue, so two typesets never overlap ---------- */
    var mjReady = null, tq = Promise.resolve();
    function whenMJ() {
      if (mjReady) return mjReady;
      mjReady = new Promise(function (res) {
        var t0 = Date.now();
        (function poll() {
          var MJ = window.MathJax;
          if (MJ && MJ.typesetPromise) {
            (MJ.startup && MJ.startup.promise ? MJ.startup.promise : Promise.resolve()).then(function () { res(true); }, function () { res(true); });
            return;
          }
          if (Date.now() - t0 > 20000) { res(false); return; }
          setTimeout(poll, 150);
        })();
      });
      return mjReady;
    }
    function typesetQ(node) {
      tq = tq.then(whenMJ).then(function (ok) {
        if (!ok || !node || !node.isConnected) return false;
        return window.MathJax.typesetPromise([node]).then(function () { return true; }, function (e) { console.warn('[periodic] MathJax', e); return false; });
      });
      return tq;
    }
    function clearTS(node) { try { if (node && window.MathJax && MathJax.typesetClear) MathJax.typesetClear([node]); } catch (e) { /* ignore */ } }
    var mjStage = A.h('div', { class: 'pt-mjstage', 'aria-hidden': 'true' });
    document.body.appendChild(mjStage);
    var mjCache = {};
    function texLines(src, cls, budget) {
      var box = A.h('div', { class: cls || 'fx' });
      splitTeX(src, budget).forEach(function (p) { box.appendChild(A.h('div', { class: 'pt-fl', text: '\\[' + p + '\\]' })); });
      return box;
    }
    function mjFor(m) {
      var c = mjCache[m.id];
      if (c) return c;
      c = mjCache[m.id] = { ready: false, cbs: [] };
      c.wrap = A.h('div');
      c.f = m.formula_latex ? texLines(m.formula_latex, 'fx', 44) : null;
      c.b = m.trainable_params ? texLines(m.trainable_params, 'fx', 44) : null;
      if (c.f) c.wrap.appendChild(c.f);
      if (c.b) c.wrap.appendChild(c.b);
      mjStage.appendChild(c.wrap);
      typesetQ(c.wrap).then(function () { c.ready = true; var cb = c.cbs; c.cbs = []; cb.forEach(function (fn) { fn(); }); });
      return c;
    }

    /* ---------- DOM skeleton ---------- */
    el.innerHTML = '';
    el.classList.add('pt-root');
    var stage = A.h('div', { class: 'stage pt-stage' });
    el.appendChild(stage);
    var head = A.h('div', { class: 'pt-head' });
    head.appendChild(A.h('div', { class: 'pt-kicker', text: 'Exposé VIII.B · seven coordinates, two drawn' }));
    head.appendChild(A.h('h3', { class: 'pt-title', text: 'The periodic table of PEFT' }));
    var instr = A.h('p', { class: 'pt-instr' });
    head.appendChild(instr);
    stage.appendChild(head);

    var bar = A.h('div', { class: 'pt-bar' });
    var seg = A.seg([{ value: false, label: 'Core · ' + nCore }, { value: true, label: 'All · ' + methods.length }], st.all, function (v) {
      st.all = v; hideCard(); rebuild(); }, 'Methods shown in the table');
    if (hasCore) bar.appendChild(seg.el);
    var sid = uid + '-q';
    var search = A.h('input', { type: 'search', id: sid, placeholder: 'Find a method', autocomplete: 'off', spellcheck: 'false' });
    bar.appendChild(A.h('span', { class: 'pt-search' }, [A.h('label', { for: sid, class: 'sr-only', text: 'Find a method by name' }), search]));
    var PRESETS = [
      { label: 'apex ∧ M1', f: { base_point: ['apex'], merge: ['M1'] } },
      { label: 'regular ∧ idempotent', f: { base_point: ['regular'], rebasing: ['idempotent'] } }
    ];
    var tryBox = A.h('span', { class: 'pt-try' }, [A.h('span', { text: 'Try' })]);
    var preBtns = PRESETS.map(function (p) {
      var b = A.h('button', { type: 'button', class: 'pt-pre', 'aria-pressed': 'false', text: p.label });
      b.addEventListener('click', function () { applyFilter(p.f); });
      tryBox.appendChild(b);
      return b;
    });
    bar.appendChild(tryBox);
    var resetBtn = A.h('button', { type: 'button', class: 'pt-reset', text: 'Reset' });
    resetBtn.addEventListener('click', function () { FAX.forEach(function (ax) { st.f[ax] = {}; }); st.q = ''; search.value = ''; update(); });
    bar.appendChild(resetBtn);
    stage.appendChild(bar);

    /* filters */
    var fdet = A.h('details', { class: 'pt-fdet' });
    var fsum = A.h('summary', { text: 'Filter by coordinates' });
    fdet.appendChild(fsum);
    var fbox = A.h('div', { class: 'pt-filters' });
    fdet.appendChild(fbox);
    stage.appendChild(fdet);
    var fBtns = {};
    FAX.forEach(function (ax) {
      var g = A.h('div', { class: 'pt-ax', role: 'group', 'aria-label': AXIS_SHORT[ax] });
      g.appendChild(A.h('span', { class: 'pt-ax-l', text: AXIS_SHORT[ax] }));
      var vals = (AXV[ax] && AXV[ax].__order ? AXV[ax].__order.slice() : []);
      methods.forEach(function (m) { var k = cv(m, ax); if (k !== NONE && vals.indexOf(k) < 0) vals.push(k); });
      if (methods.some(function (m) { return cv(m, ax) === NONE; })) vals.push(NONE);
      fBtns[ax] = {};
      vals.forEach(function (k) {
        var b = A.h('button', { type: 'button', class: 'pt-f', 'aria-pressed': 'false', 'data-ax': ax, 'data-v': k, html: '<span class="v">' + keyHTML(k) + '</span><span class="n">0</span>' });
        b.addEventListener('click', function () { st.f[ax][k] = !st.f[ax][k]; update(); });
        var tip = function (e) {
          var v = AXV[ax] && AXV[ax][k];
          var html = '<div class="t">' + A.esc(AXIS_SHORT[ax]) + ': ' + keyHTML(k) + '</div>' + (v ? '<div>' + pretty(v.name) + '. ' + pretty(v.definition || '') + '</div>' : '<div>' + (k === NONE ? 'No value on this axis (– in Table VIII.B): the axis does not apply.' : '') + '</div>');
          A.tip.show(html, evtFor(b, e));
        };
        b.addEventListener('mouseenter', tip);
        b.addEventListener('focus', tip);
        b.addEventListener('mouseleave', A.tip.hide);
        b.addEventListener('blur', A.tip.hide);
        fBtns[ax][k] = b;
        g.appendChild(b);
      });
      fbox.appendChild(g);
    });
    function evtFor(node, e) {
      if (e && e.clientX != null && e.type !== 'focus') return e;
      var r = node.getBoundingClientRect();
      return { clientX: r.left + 4, clientY: r.bottom - 6 };
    }

    var readout = A.h('div', { class: 'pt-readout', 'aria-hidden': 'true' });
    stage.appendChild(readout);
    var legend = A.h('div', { class: 'pt-legend' });
    stage.appendChild(legend);
    var body = A.h('div', { class: 'pt-body' });
    stage.appendChild(body);
    var checks = A.h('div', { class: 'pt-checks' });
    stage.appendChild(checks);
    var live = A.h('div', { class: 'sr-only', 'aria-live': 'polite' });
    stage.appendChild(live);

    /* ---------- hover card ---------- */
    var card = A.h('div', { class: 'pt-card', role: 'tooltip', id: uid + '-card' });
    document.body.appendChild(card);
    var cardFor = null, cardTimer = 0, hideTimer = 0;
    function coordRows(m) {
      return AX7.map(function (ax) {
        var k = cv(m, ax);
        var hit = FAX.indexOf(ax) >= 0 && st.f[ax][k];
        var v = ax === 'kind' ? keyHTML(k) + (isR(m) && ROWBY[rowOf(m)] ? ' · ' + ROWBY[rowOf(m)].name : '') : keyHTML(k);
        return '<span class="a">' + AXIS_SHORT[ax] + '</span><span class="v' + (hit ? ' hit' : '') + '">' + v + '</span><span class="g">' + gloss(ax, k) + '</span>';
      }).join('');
    }
    function showCard(chipEl) {
      var m = byId[chipEl.getAttribute('data-id')];
      if (!m) return;
      clearTimeout(hideTimer);
      cardFor = m.id;
      card.style.setProperty('--k', kcol(m));
      var nOut = (OUT[m.id] || []).length, nIn = (IN[m.id] || []).length, nOb = (m.obstructions || []).length;
      card.innerHTML = '<div class="tp"><span><span class="sw"></span>' + A.esc(A.kindName(m.modification_kind)) + '</span><span>' + A.esc(yearVenue(m)) + '</span>' +
        (m.hf_peft ? '<span class="hf">HF PEFT</span>' : '') + (m.entry_type && m.entry_type !== 'method' ? '<span>' + A.esc(m.entry_type) + '</span>' : '') + '</div>' +
        '<div class="nm">' + labelHTML(m.name) + '</div>' + (m.full_name && norm(m.full_name) !== norm(m.name) ? '<div class="fu">' + A.esc(m.full_name) + '</div>' : '') +
        '<div class="co">' + coordRows(m) + '</div>';
      var c = mjFor(m);
      function sec(label, node) {
        card.appendChild(A.h('div', { class: 'sc', text: label }));
        if (!node) { card.appendChild(A.h('div', { class: 'wait', text: '—' })); return; }
        if (c.ready) card.appendChild(node.cloneNode(true));
        else {
          var w = A.h('div', { class: 'wait', text: 'typesetting…' });
          card.appendChild(w);
          c.cbs.push(function () { if (cardFor === m.id && w.isConnected) { w.replaceWith(node.cloneNode(true)); placeCard(chipEl); } });
        }
      }
      sec('Formula', c.f);
      sec('Budget |M| = dim Q', c.b);
      var more = c.f && c.f.children.length > 3 ? ' · formula continues in the panel' : '';
      card.appendChild(A.h('div', { class: 'ft', html: '<b>' + nOut + '</b> ' + (nOut === 1 ? 'arrow' : 'arrows') + ' out · <b>' + nIn + '</b> in · <span class="ob">' + nOb + ' ruled out</span>' + more + '<br>' + (st.sel === m.id ? 'Its panel is open below.' : nOut + nIn ? 'Select it for its arrows, each with its map h.' : nOb ? 'Select it for its obstructions.' : 'Select it for its full card.') }));
      card.classList.add('on');
      chipEl.setAttribute('aria-describedby', card.id);
      placeCard(chipEl);
    }
    function placeCard(chipEl) {
      if (!chipEl.isConnected) return;
      var r = chipEl.getBoundingClientRect(), cw = card.offsetWidth, ch = card.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
      var x, y;
      if (r.right + 12 + cw <= vw - 8) x = r.right + 12;
      else if (r.left - 12 - cw >= 8) x = r.left - 12 - cw;
      else x = clamp(r.left, 8, vw - cw - 8);
      if (x === r.right + 12 || x === r.left - 12 - cw) y = clamp(r.top - 10, 8, vh - ch - 8);
      else y = r.bottom + 8 + ch <= vh - 8 ? r.bottom + 8 : Math.max(8, r.top - 8 - ch);
      card.style.left = Math.round(x) + 'px';
      card.style.top = Math.round(y) + 'px';
    }
    function hideCard() {
      clearTimeout(cardTimer);
      card.classList.remove('on');
      if (cardFor) { var b = chipById(cardFor); if (b) b.removeAttribute('aria-describedby'); }
      cardFor = null;
    }
    window.addEventListener('scroll', function () {
      if (!cardFor) return;
      var c = chipById(cardFor);
      if (c && document.activeElement === c) placeCard(c); else hideCard();
    }, { passive: true });

    /* ---------- body: table or list ---------- */
    var chipEls = {};
    function chipById(id) { return chipEls[id] || null; }
    function makeChip(m) {
      var b = A.h('button', { type: 'button', class: 'pt-chip', 'data-id': m.id, tabindex: '-1', 'aria-label': ariaOf(m), 'aria-expanded': 'false', html: labelHTML(shortName(m)) });
      b.style.setProperty('--k', kcol(m));
      chipEls[m.id] = b;
      return b;
    }
    function sortCell(list) {
      return list.sort(function (a, b) { return (a.year || 0) - (b.year || 0) || plainLabel(a).localeCompare(plainLabel(b)); });
    }
    function rowHead(rk, i, cls) {
      var R = ROWBY[rk] || { sym: rk, name: '', desc: '' };
      var c = 'pt-rh' + (cls ? ' ' + cls : '');
      if (R.grp === 'R') {
        var rIdx = ROWS.filter(function (x) { return x.grp === 'R'; }).map(function (x) { return x.key; });
        var j = rIdx.indexOf(rk);
        c += ' gR ' + (j === 0 ? 'first' : j === rIdx.length - 1 ? 'last' : 'mid');
      }
      var d = A.h('div', { class: c, 'data-row': rk });
      d.innerHTML = '<span class="sym">' + R.sym + '</span><span class="nm">' + A.esc(R.name) + '</span><span class="ds">' + gk(R.desc || '') + '</span><span class="ct" data-ct></span>';
      if (R.tip) d.title = R.tip;
      return d;
    }

    /* Two aligned blocks. Block I holds the reparametrisations, the objects of Meth(Θ, θ0): their
       germs live in weight space, so their columns are Lin … Meet and Fun is empty by definition.
       Block II holds extensions, lenses, post-hoc operations and families: the shapes that occur
       there (from the data) share the Lin and Cone columns with block I, and Fun takes the rest. */
    function blockPlan(pop) {
      var rI = rowKeys.filter(function (r) { return /^R-/.test(r) && pop.some(function (m) { return rowOf(m) === r; }); });
      var rII = rowKeys.filter(function (r) { return !/^R-/.test(r) && pop.some(function (m) { return rowOf(m) === r; }); });
      function shapesOf(rows, all) {
        var s = {};
        methods.forEach(function (m) { if (rows.indexOf(rowOf(m)) >= 0 || (all && all(m))) { var c = colOf(m); if (c) s[c] = 1; } });
        return s;
      }
      var isRrow = function (m) { return /^R-/.test(rowOf(m)); };
      var sI = shapesOf([], isRrow), sII = shapesOf([], function (m) { return !isRrow(m); });
      var colsI = ['Lin', 'Cone', 'Cone*', 'Orb', 'Sat', 'Meet'].concat(colKeys.filter(function (c) { return sI[c] && ['Lin', 'Cone', 'Cone*', 'Orb', 'Sat', 'Meet'].indexOf(c) < 0; }));
      var colsII = colKeys.filter(function (c) { return sII[c] || c === 'Lin' || c === 'Cone' || c === 'Fun'; });
      var nGrid = colsI.length;
      function spans(cols) {
        var out = [], used = {}, tail = [];
        cols.forEach(function (c) { var j = colsI.indexOf(c); if (j >= 0 && !used[j]) { used[j] = 1; out.push({ key: c, start: j, span: 1 }); } else tail.push(c); });
        var free = [];
        for (var j = 0; j < nGrid; j++) if (!used[j]) free.push(j);
        var lastShared = out.length ? Math.max.apply(null, out.map(function (o) { return o.start; })) : -1;
        free = free.filter(function (j) { return j > lastShared; });
        if (tail.length) {
          if (free.length < tail.length) { while (free.length < tail.length) { free.push(nGrid); nGrid++; } }
          var per = Math.floor(free.length / tail.length), extra = free.length - per * tail.length, k = 0;
          tail.forEach(function (c, i) { var sp = per + (i < extra ? 1 : 0); out.push({ key: c, start: free[k], span: sp }); k += sp; });
        }
        return out.sort(function (a, b) { return a.start - b.start; });
      }
      var bI = { key: 'I', rows: rI, cols: spans(colsI) }, bII = { key: 'II', rows: rII, cols: spans(colsII) };
      var unocc = ['Cone*', 'Orb', 'Sat', 'Meet'].filter(function (c) { return !sII[c]; });
      bI.title = '<b>R</b> · reparametrisations, the objects of Meth(Θ, θ<sub>0</sub>)' + (sI.Fun ? '' : ' <span>· ρ lands in Θ, so no Fun column: empty by definition</span>');
      bII.title = '<b>E · E° · B · P · F<sub>T</sub></b> · extensions, lenses, post-hoc operations, families' + (unocc.length ? ' <span>· ' + unocc.join(', ') + ' unoccupied here</span>' : '');
      return { blocks: [bI, bII].filter(function (b) { return b.rows.length; }), nGrid: nGrid };
    }

    function colHead(ck, count) {
      var C = COLBY[ck] || { key: ck, name: '' }, v = AXV.shape && AXV.shape[ck];
      var hd = A.h('div', { class: 'pt-ch', 'data-col': ck, title: v ? (v.name + '. ' + v.definition).replace(/theta0/g, 'θ₀').replace(/\brho\b/g, 'ρ') : '' });
      hd.innerHTML = glyph(ck, 40, 25) + '<span class="k">' + A.esc(ck) + '</span><span class="nm">' + gk(A.esc(C.name)).replace(/·/g, '·<wbr>') + '</span><span class="ct" data-ct></span>';
      return hd;
    }
    function buildTable() {
      var pop = popList();
      var cells = {}, off = [];
      pop.forEach(function (m) {
        var r = rowOf(m), c = colOf(m);
        if (!c) { off.push(m); return; }
        (cells[r + '|' + c] = cells[r + '|' + c] || []).push(m);
      });
      for (var k in cells) sortCell(cells[k]);
      sortCell(off);
      var plan = blockPlan(pop);
      G = { mode: 'table', pop: pop, cells: cells, off: off, blocks: plan.blocks, nGrid: plan.nGrid, rows: [], cellEls: {}, rowEls: {}, rowBlock: {} };
      var wrap = A.h('div', { class: 'pt-wrap' });
      var grid = A.h('div', { class: 'pt-grid' + (st.all ? ' pt-dense' : ''), role: 'group', 'aria-label': 'Periodic table of ' + pop.length + ' methods: kinds in rows, shapes of the image germ in columns' });
      G.wrap = wrap; G.grid = grid;
      G.blocks.forEach(function (B, bi) {
        B.band = A.h('div', { class: 'pt-band' + (bi ? ' second' : ''), html: gk(B.title) });
        B.band.style.gridColumn = '1 / -1';
        grid.appendChild(B.band);
        B.corner = A.h('div', { class: 'pt-corner', html: 'kind ↓<br>shape →' });
        B.corner.style.gridColumn = '1';
        grid.appendChild(B.corner);
        B.headEls = {};
        B.cols.forEach(function (c) {
          var hd = colHead(c.key);
          hd.style.gridColumn = (c.start + 2) + ' / span ' + c.span;
          B.headEls[c.key] = hd;
          grid.appendChild(hd);
        });
        B.rows.forEach(function (rk) {
          G.rows.push(rk);
          G.rowBlock[rk] = B;
          var els = [];
          var rh = rowHead(rk);
          rh.style.gridColumn = '1';
          els.push(rh);
          B.cols.forEach(function (c) {
            var ck = c.key, list = cells[rk + '|' + ck] || [];
            var cell = A.h('div', { class: 'pt-cell' + (list.length ? '' : ' empty'), role: 'group', 'data-row': rk, 'data-col': ck,
              'aria-label': rowLabelText(rk) + ' by ' + ck + ': ' + plural(list.length, 'method') });
            list.forEach(function (m) { cell.appendChild(makeChip(m)); });
            if (list.length) cell.appendChild(A.h('span', { class: 'pt-cn', 'aria-hidden': 'true', text: String(list.length) }));
            cell.style.gridColumn = (c.start + 2) + ' / span ' + c.span;
            G.cellEls[rk + '|' + ck] = cell;
            els.push(cell);
          });
          G.rowEls[rk] = els;
          els.forEach(function (e) { grid.appendChild(e); });
        });
      });
      wrap.appendChild(grid);
      body.appendChild(wrap);
      if (off.length) {
        var strip = A.h('div', { class: 'pt-off' });
        var sh = A.h('div', { class: 'pt-rh', title: 'Shape n/a: no continuous coefficients, so no germ at θ₀ (– in Table VIII.B).' });
        sh.innerHTML = '<span class="sym">' + glyph('none', 26, 17) + '</span><span class="nm">off the table</span><span class="ds">shape n/a: no germ</span><span class="ct" data-ct></span>';
        var oc = A.h('div', { class: 'pt-cell', role: 'group', 'aria-label': 'Off the table, shape not applicable: ' + plural(off.length, 'method') });
        off.forEach(function (m) { oc.appendChild(makeChip(m)); });
        oc.appendChild(A.h('span', { class: 'pt-cn', 'aria-hidden': 'true', text: String(off.length) }));
        strip.appendChild(sh); strip.appendChild(oc);
        G.offEl = strip; G.offCell = oc; G.offHead = sh;
        body.appendChild(strip);
      }
      placeGrid();
      layoutColumns();
    }
    function placeGrid() {
      if (G.mode !== 'table') return;
      var r = 1;
      G.blocks.forEach(function (B) {
        B.band.style.gridRow = String(r++);
        B.corner.style.gridRow = String(r);
        for (var k in B.headEls) B.headEls[k].style.gridRow = String(r);
        r++;
        B.rows.forEach(function (rk) {
          G.rowEls[rk].forEach(function (e) { e.style.gridRow = String(r); });
          r++;
          if (G.panel && G.panelRow === rk && G.panel.parentNode === G.grid) { G.panel.style.gridRow = String(r); G.panel.style.gridColumn = '1 / -1'; r++; }
        });
      });
    }

    /* column widths: minimise the table's height for the chips it actually holds (both blocks at once) */
    function chipFont() {
      var any = G.grid && G.grid.querySelector('.pt-chip');
      var f = any ? getComputedStyle(any).font : '';
      return f || ('500 12px ' + A.css('--f-body'));
    }
    function chipPad() {
      var any = G.grid && G.grid.querySelector('.pt-chip');
      if (!any) return 14;
      var cs = getComputedStyle(any);
      return (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0) + (parseFloat(cs.borderLeftWidth) || 0) + (parseFloat(cs.borderRightWidth) || 0);
    }
    function layoutColumns() {
      if (G.mode !== 'table' || !G.grid) return;
      var W = G.wrap.clientWidth;
      if (!W) return;
      G.grid.classList.toggle('pt-tight', W < 700);
      var LW = W < 760 ? 96 : 108, gap = 4, nG = G.nGrid; /* row heads: kind names in Plex Sans at 12.8px */
      var avail = W - LW - gap * nG - 2;
      var font = chipFont(), dense = st.all;
      var pad = chipPad(), pitch = (dense || W < 700) ? 21 : 23, rowMin = 52, headMin = W < 760 ? 46 : 50;
      var cap = Math.min(dense ? 84 : 100, Math.max(62, Math.floor(avail / nG)));
      var badge = isActive() ? 44 : 24; /* the cell count, "n" or "lit/n", in 12px mono */
      var cellsSpec = [], mins = [];
      for (var j = 0; j < nG; j++) mins[j] = headMin;
      /* a column is never narrower than the longest word of its header */
      var hfUi = '500 12px ' + A.css('--f-ui'), hfKey = '600 16.8px ' + A.css('--f-display');
      G.blocks.forEach(function (B) {
        B.cols.forEach(function (c) {
          if (c.span !== 1) return;
          var C = COLBY[c.key] || { name: '' }, wmax = tw(c.key, hfKey);
          String(C.name || '').split(/\s+|·/).forEach(function (wd) { wmax = Math.max(wmax, tw(wd + '·', hfUi)); });
          mins[c.start] = Math.max(mins[c.start], Math.ceil(wmax) + 10);
        });
      });
      /* and, where the stage allows, never narrower than the longest unbreakable word of a chip in it
         (a chip wraps at spaces and hyphens; a word that still does not fit breaks inside the chip) */
      var wordMin = mins.slice(), cellEl = G.grid.querySelector('.pt-cell'), ccs = cellEl ? getComputedStyle(cellEl) : null;
      var cellPad = ccs ? (parseFloat(ccs.paddingLeft) || 0) + (parseFloat(ccs.paddingRight) || 0) + (parseFloat(ccs.borderLeftWidth) || 0) + (parseFloat(ccs.borderRightWidth) || 0) : 10;
      G.blocks.forEach(function (B) {
        B.rows.forEach(function (rk) {
          var spec = { rk: rk, cells: [] };
          B.cols.forEach(function (c) {
            var list = G.cells[rk + '|' + c.key] || [];
            var arr = list.map(function (m) { return Math.ceil(tw(plainLabel(m), font) + pad) + 1; });
            if (arr.length) arr.push(badge);
            spec.cells.push({ start: c.start, span: c.span, ws: arr });
            if (c.span === 1) {
              arr.forEach(function (w) { mins[c.start] = Math.max(mins[c.start], Math.min(w + 10, cap)); });
              list.forEach(function (m) {
                labelParts(plainLabel(m)).forEach(function (part) { wordMin[c.start] = Math.max(wordMin[c.start], Math.ceil(tw(part, font) + pad) + cellPad + 1); });
              });
            }
          });
          cellsSpec.push(spec);
        });
      });
      /* when the stage cannot give every column its longest word, settle the small shortfalls first:
         one long word left to break reads better than several */
      var budget = avail - mins.reduce(function (a, b) { return a + b; }, 0);
      mins.map(function (m0, k) { return { k: k, need: Math.max(0, wordMin[k] - m0) }; })
        .filter(function (o) { return o.need > 0; }).sort(function (a, b) { return a.need - b.need || a.k - b.k; })
        .forEach(function (o) { var add = Math.max(0, Math.min(o.need, budget)); mins[o.k] += add; budget -= add; });
      function lines(arr, inner) {
        if (!arr.length) return 0;
        var n = 1, x = 0;
        for (var i = 0; i < arr.length; i++) {
          var w = arr[i];
          if (w > inner) { if (x > 0) n++; n++; x = inner; continue; }
          if (x > 0 && x + 3 + w > inner) { n++; x = w; } else x += (x > 0 ? 3 : 0) + w;
        }
        return n;
      }
      function rowH(spec, wd) {
        var mxl = 0;
        spec.cells.forEach(function (c) {
          var w = 0;
          for (var j = c.start; j < c.start + c.span; j++) w += wd[j];
          w += gap * (c.span - 1);
          var l = lines(c.ws, w - 10);
          if (l > mxl) mxl = l;
        });
        return Math.max(rowMin, mxl * pitch + 10);
      }
      function totalH(wd) { var t = 0; cellsSpec.forEach(function (s) { t += rowH(s, wd); }); return t; }
      var widths = mins.slice(), sum = mins.reduce(function (a, b) { return a + b; }, 0);
      var tpl;
      if (sum >= avail) {
        tpl = LW + 'px ' + widths.map(function (w) { return w + 'px'; }).join(' ');
      } else {
        /* start where the chips are, then move width between columns while the table gets shorter */
        var rem = avail - sum, need = [];
        for (var q = 0; q < nG; q++) need[q] = 1;
        cellsSpec.forEach(function (s) { s.cells.forEach(function (c) { var tot = 0; c.ws.forEach(function (w) { tot += w; }); for (var j = c.start; j < c.start + c.span; j++) need[j] += tot / c.span; }); });
        need = need.map(Math.sqrt);
        var ns = need.reduce(function (x, y) { return x + y; }, 0);
        widths = widths.map(function (w, j) { return w + rem * need[j] / ns; });
        var cur = totalH(widths), guard = 0;
        [128, 64, 32, 16, 8, 4, 2].forEach(function (dl) {
          var improved = true;
          while (improved && guard++ < 600) {
            improved = false;
            for (var f = 0; f < nG; f++) for (var t2 = 0; t2 < nG; t2++) {
              if (f === t2 || widths[f] - dl < mins[f]) continue;
              widths[f] -= dl; widths[t2] += dl;
              var t = totalH(widths);
              if (t < cur - 0.5) { cur = t; improved = true; } else { widths[f] += dl; widths[t2] -= dl; }
            }
          }
        });
        tpl = LW + 'px ' + widths.map(function (w) { return 'minmax(0,' + w.toFixed(1) + 'fr)'; }).join(' ');
      }
      G.grid.style.gridTemplateColumns = tpl;
      G.layoutInfo = { W: W, avail: avail, mins: mins, widths: widths.map(function (w) { return Math.round(w); }), predicted: totalH(widths) };
      if (G.offEl) G.offEl.style.setProperty('--lw', LW + 'px');
      G.lw = LW;
    }

    function buildList() {
      var pop = popList();
      var groups = {};
      pop.forEach(function (m) { var r = rowOf(m), c = colOf(m) || '–'; (groups[r] = groups[r] || {}); (groups[r][c] = groups[r][c] || []).push(m); });
      var rows = rowKeys.filter(function (r) { return groups[r]; });
      G = { mode: 'list', pop: pop, rows: rows, secEls: {}, subEls: {} };
      var list = A.h('div', { class: 'pt-list', role: 'group', 'aria-label': 'Methods grouped by kind and by shape' });
      rows.forEach(function (rk) {
        var R = ROWBY[rk] || { sym: rk, name: '', desc: '' };
        var sec = A.h('section', { class: 'pt-lsec', 'data-row': rk });
        var hd = A.h('div', { class: 'pt-lh' });
        hd.innerHTML = '<span class="sym' + (R.grp === 'R' ? ' gR' : '') + '">' + R.sym + '</span><span class="nm">' + A.esc(R.name) + '</span><span class="ds">' + gk(R.desc || '') + '</span><span class="ct" data-ct></span>';
        sec.appendChild(hd);
        colKeys.concat(['–']).forEach(function (ck) {
          var ms = groups[rk][ck];
          if (!ms) return;
          sortCell(ms);
          var sub = A.h('div', { class: 'pt-lsub', 'data-row': rk, 'data-col': ck });
          var sh = A.h('div', { class: 'pt-lsub-h' });
          sh.innerHTML = glyph(ck === '–' ? 'none' : ck, 30, 19) + '<span>' + A.esc(ck === '–' ? 'n/a' : ck) + '</span><span class="ct" data-ct></span>';
          var chips = A.h('div', { class: 'pt-lchips' });
          ms.forEach(function (m) { chips.appendChild(makeChip(m)); });
          sub.appendChild(sh); sub.appendChild(chips);
          sec.appendChild(sub);
          G.subEls[rk + '|' + ck] = { el: sub, ms: ms };
        });
        G.secEls[rk] = sec;
        list.appendChild(sec);
      });
      G.none = A.h('p', { class: 'pt-none', text: 'No method matches. Reset or relax a filter.' });
      G.none.hidden = true;
      list.appendChild(G.none);
      body.appendChild(list);
      G.list = list;
    }

    function rebuild() {
      var keepSel = st.sel;
      closePanel(true);
      body.innerHTML = '';
      chipEls = {};
      st.narrow = el.clientWidth < 600;
      fdet.classList.toggle('wide', !st.narrow);
      if (!st.narrow) fdet.open = true;
      else if (!fdet.__touched) fdet.open = false;
      if (st.narrow) buildList(); else buildTable();
      setRoving(null);
      buildLegend();
      buildChecks();
      update();
      if (keepSel && byId[keepSel]) openPanel(keepSel, { silent: true, noScroll: true });
    }
    fdet.addEventListener('toggle', function () { if (st.narrow) fdet.__touched = true; });

    /* ---------- legend, checks ---------- */
    function buildLegend() {
      var pop = G.pop, cnt = {};
      pop.forEach(function (m) { cnt[m.modification_kind] = (cnt[m.modification_kind] || 0) + 1; });
      legend.innerHTML = '<span class="lt">Chip colour · modification kind</span>' + A.KINDS.filter(function (k) { return cnt[k.key]; }).map(function (k) {
        return '<span class="it"><span class="sw" style="--k:' + A.css(k.token) + '"></span>' + A.esc(k.name) + ' <span class="n">' + cnt[k.key] + '</span></span>';
      }).join('');
    }
    function buildChecks() {
      var pop = G.pop;
      var apex = pop.filter(function (m) { return cv(m, 'base_point') === 'apex'; });
      var linApex = apex.filter(function (m) { return colOf(m) === 'Lin'; });
      var idem = pop.filter(function (m) { return cv(m, 'rebasing') === 'idempotent'; });
      var idCols = {};
      idem.forEach(function (m) { var c = colOf(m) || 'n/a'; idCols[c] = (idCols[c] || 0) + 1; });
      var idBad = idem.filter(function (m) { var c = colOf(m); return c !== 'Lin' && c !== 'Orb'; });
      var rFun = pop.filter(function (m) { return /^R-/.test(rowOf(m)) && colOf(m) === 'Fun'; });
      var nR = pop.filter(function (m) { return /^R-/.test(rowOf(m)); }).length;
      function names(list) { return list.slice(0, 6).map(function (m) { return A.esc(m.name); }).join(', ') + (list.length > 6 ? '…' : ''); }
      /* a citation links to its statement when the page carries it */
      function ref(label, anchor) { return document.getElementById(anchor) ? '<a class="pt-chk-a" href="#' + anchor + '">' + label + '</a>' : label; }
      var colList = ['Lin', 'Orb'].concat(Object.keys(idCols).filter(function (c) { return c !== 'Lin' && c !== 'Orb'; })).filter(function (c) { return idCols[c]; })
        .map(function (c) { return idCols[c] + ' ' + A.esc(c); });
      var colText = colList.length > 1 ? colList.slice(0, -1).join(', ') + ' and ' + colList[colList.length - 1] : colList.join('');
      var items = [
        { ok: linApex.length === 0, html: linApex.length === 0 ? '<b>No apex in the Lin column.</b> None of the ' + apex.length + ' apex methods is linear. An affine ρ has the same differential everywhere, so its first-order deficit is 0 (' + ref('Def I.5', 'thm-I-5') + ').' : '<b>Apex in the Lin column:</b> ' + names(linApex) + '.', f: { base_point: ['apex'] }, btn: 'show apex' },
        { ok: idBad.length === 0 && idem.length > 0, html: (idBad.length === 0 ? '<b>Idempotent rebasing stays in Lin ∪ Orb.</b> The ' + idem.length + ' idempotent methods are ' + colText + '. ' : '<b>Idempotent rebasing leaves Lin ∪ Orb:</b> ' + names(idBad) + '. ') + 'A shape that is a fixed subspace or subgroup gains nothing from merge-and-restart (' + ref('Thm V.3', 'thm-V-3') + ', ' + ref('Corollary 2', 'base-rebase-corollaries') + ').', f: { rebasing: ['idempotent'] }, btn: 'show them' },
        { ok: rFun.length === 0, html: rFun.length === 0 ? '<b>R × Fun is empty.</b> None of the ' + nR + ' reparametrisations is function-level. Each ρ lands in Θ, so its germ lives in weight space (' + ref('Def I.2', 'thm-I-2') + '), and the first block has no Fun column.' : '<b>R × Fun is occupied:</b> ' + names(rFun) + '.', f: null }
      ];
      checks.innerHTML = '<div class="hd">Checks on the table, computed from the coordinates</div>';
      items.forEach(function (it) {
        var row = A.h('div', { class: 'pt-ck ' + (it.ok ? 'ok' : 'bad') });
        row.innerHTML = '<span class="ic" aria-hidden="true">' + (it.ok ? '✓' : '✗') + '</span><span><span class="sr-only">' + (it.ok ? 'Check passes. ' : 'Check fails. ') + '</span>' + it.html + '</span>';
        if (it.f) { var b = A.h('button', { type: 'button', text: it.btn }); b.addEventListener('click', function () { applyFilter(it.f); }); row.appendChild(b); }
        else row.appendChild(A.h('span'));
        checks.appendChild(row);
      });
    }

    function applyFilter(f) {
      FAX.forEach(function (ax) { st.f[ax] = {}; });
      for (var ax in f) f[ax].forEach(function (v) { st.f[ax][v] = true; });
      update();
    }

    /* ---------- update: lit state, counts, readout ---------- */
    function exprText() {
      var parts = [];
      FAX.forEach(function (ax) {
        var ks = Object.keys(st.f[ax]).filter(function (k) { return st.f[ax][k]; });
        if (!ks.length) return;
        var s = ks.map(keyHTML).join(' ∨ ');
        parts.push(ks.length > 1 ? '(' + s + ')' : s);
      });
      if (st.q) parts.push('“' + A.esc(search.value.trim()) + '”');
      return parts.join(' ∧ ');
    }
    function update() {
      var pop = G.pop || [];
      var act = isActive();
      el.classList.toggle('is-f', act);
      var lit = {}, nLit = 0;
      pop.forEach(function (m) { var on = !act || isLit(m); lit[m.id] = on; if (on) nLit++; });
      for (var id in chipEls) {
        var b = chipEls[id], on = lit[id] !== false;
        b.classList.toggle('on', act && on);
        b.classList.toggle('off', act && !on);
        if (G.mode === 'list') b.hidden = act && !on;
      }
      var occ = 0, occLit = 0;
      if (G.mode === 'table') {
        var rc = {}, rcl = {};
        G.blocks.forEach(function (B) {
          var cc = {}, ccl = {};
          B.rows.forEach(function (rk) {
            B.cols.forEach(function (c) {
              var ck = c.key, list = G.cells[rk + '|' + ck] || [], cell = G.cellEls[rk + '|' + ck];
              var n = list.length, nl = list.filter(function (m) { return lit[m.id]; }).length;
              rc[rk] = (rc[rk] || 0) + n; rcl[rk] = (rcl[rk] || 0) + nl; cc[ck] = (cc[ck] || 0) + n; ccl[ck] = (ccl[ck] || 0) + nl;
              if (n) { occ++; if (nl) occLit++; }
              cell.classList.toggle('zero', act && n > 0 && nl === 0);
              var cn = cell.querySelector('.pt-cn');
              if (cn) { cn.textContent = act ? nl + '/' + n : String(n); cn.classList.toggle('hit', act && nl > 0); }
            });
          });
          B.cols.forEach(function (c) { var e = B.headEls[c.key].querySelector('[data-ct]'); e.textContent = act ? (ccl[c.key] || 0) + ' / ' + (cc[c.key] || 0) : String(cc[c.key] || 0); });
        });
        G.rows.forEach(function (rk) { var e = G.rowEls[rk][0].querySelector('[data-ct]'); e.textContent = act ? rcl[rk] + ' / ' + rc[rk] : String(rc[rk]); });
        if (G.offCell) {
          var no = G.off.length, nol = G.off.filter(function (m) { return lit[m.id]; }).length;
          var cn2 = G.offCell.querySelector('.pt-cn');
          cn2.textContent = act ? nol + '/' + no : String(no);
          if (no) { occ++; if (nol) occLit++; }
          G.offHead.querySelector('[data-ct]').textContent = act ? nol + ' / ' + no : String(no);
        }
      } else if (G.mode === 'list') {
        var anyShown = false;
        G.rows.forEach(function (rk) {
          var sec = G.secEls[rk], tot = 0, tl = 0;
          Object.keys(G.subEls).forEach(function (key) {
            if (key.split('|')[0] !== rk) return;
            var s = G.subEls[key], n = s.ms.length, nl = s.ms.filter(function (m) { return lit[m.id]; }).length;
            tot += n; tl += nl; occ++; if (nl) occLit++;
            var hasPanel = G.panel && G.panel.parentNode === sec && G.panelSub === key;
            s.el.hidden = act && nl === 0 && !hasPanel;
            s.el.querySelector('[data-ct]').textContent = act ? nl + '/' + n : String(n);
          });
          var keep = G.panel && G.panel.parentNode === sec;
          sec.hidden = act && tl === 0 && !keep;
          if (!sec.hidden) anyShown = true;
          sec.querySelector('.pt-lh [data-ct]').textContent = act ? tl + ' / ' + tot : String(tot);
        });
        G.none.hidden = anyShown;
      }
      /* faceted counts on the filter chips */
      FAX.forEach(function (ax) {
        var cnt = {};
        pop.forEach(function (m) { if (isLit(m, ax)) { var k = cv(m, ax); cnt[k] = (cnt[k] || 0) + 1; } });
        for (var k in fBtns[ax]) {
          var b = fBtns[ax][k], on = !!st.f[ax][k];
          b.setAttribute('aria-pressed', String(on));
          b.querySelector('.n').textContent = cnt[k] || 0;
          b.classList.toggle('zero', !cnt[k]);
          b.setAttribute('aria-label', AXIS_SHORT[ax] + ' ' + keyText(k) + ', ' + plural(cnt[k] || 0, 'method'));
        }
      });
      PRESETS.forEach(function (p, i) {
        var same = !st.q && FAX.every(function (ax) {
          var want = (p.f[ax] || []).slice().sort().join(','), have = Object.keys(st.f[ax]).filter(function (k) { return st.f[ax][k]; }).sort().join(',');
          return want === have;
        });
        preBtns[i].setAttribute('aria-pressed', String(same));
      });
      resetBtn.disabled = !act;
      var nf = FAX.reduce(function (s, ax) { return s + nSel(ax); }, 0);
      fsum.textContent = 'Filter by coordinates' + (nf ? ' · ' + nf + ' active' : '');
      var popName = st.all ? 'catalogued' : 'core';
      if (act) {
        readout.innerHTML = '<span class="pt-expr">' + exprText() + '</span><span>→ <b>' + nLit + '</b> of ' + pop.length + ' ' + popName + ' methods lit, in <b>' + occLit + '</b> ' + (G.mode === 'table' ? 'cells' : 'groups') + '</span>';
      } else {
        readout.innerHTML = '<span><b>' + pop.length + '</b> ' + popName + ' methods' + ' in <b>' + occ + '</b> occupied ' + (G.mode === 'table' ? 'cells' : 'groups') + '</span><span>values on one axis combine with ∨, axes with ∧</span>';
      }
      instr.textContent = (st.narrow ? 'Methods grouped by kind, then by the shape of the image germ. ' : 'Kind down the side, the shape of the image germ across the top, and the other five coordinates as filters. ') +
        (st.narrow ? 'Tap a method for its coordinates, formula, budget, arrows and obstructions.' : 'Hover or focus a method for its seven coordinates, formula and budget; select it for its arrows, each with its map h, and its obstructions. Arrow keys move between methods and Escape closes.');
      announce(act ? nLit + ' of ' + pop.length + ' methods match.' : pop.length + ' methods shown.');
      ensureRoving();
      if (G.mode === 'table' && act !== G.lastAct) { G.lastAct = act; layoutColumns(); }
    }
    var announce = A.debounce(function (t) { live.textContent = t; }, 400);

    /* ---------- keyboard: one tab stop, arrow keys move between chips ---------- */
    var roving = null;
    function visibleChips() { return Array.prototype.filter.call(body.querySelectorAll('.pt-chip'), function (b) { return !b.hidden && b.offsetParent !== null; }); }
    function setRoving(b) {
      if (roving && roving !== b) roving.tabIndex = -1;
      roving = b || null;
      if (!roving) { var v = visibleChips(); roving = (st.sel && chipById(st.sel) && !chipById(st.sel).hidden ? chipById(st.sel) : v[0]) || null; }
      if (roving) roving.tabIndex = 0;
    }
    function ensureRoving() { if (!roving || !roving.isConnected || roving.hidden) { roving = null; setRoving(null); } }
    function moveFocus(from, key) {
      var v = visibleChips(), i = v.indexOf(from), to = null;
      if (key === 'ArrowRight') to = v[i + 1];
      else if (key === 'ArrowLeft') to = v[i - 1];
      else if (key === 'Home') to = v[0];
      else if (key === 'End') to = v[v.length - 1];
      else {
        var r = from.getBoundingClientRect(), cy = r.top + r.height / 2, cx = r.left + r.width / 2, best = Infinity;
        v.forEach(function (b) {
          if (b === from) return;
          var q = b.getBoundingClientRect(), qy = q.top + q.height / 2, dy = key === 'ArrowDown' ? qy - cy : cy - qy;
          if (dy < 6) return;
          var s = dy * 2.5 + Math.abs(q.left + q.width / 2 - cx);
          if (s < best) { best = s; to = b; }
        });
      }
      if (to) { setRoving(to); to.focus(); }
    }

    body.addEventListener('keydown', function (e) {
      var b = e.target.closest && e.target.closest('.pt-chip');
      if (b && ['ArrowRight', 'ArrowLeft', 'ArrowUp', 'ArrowDown', 'Home', 'End'].indexOf(e.key) >= 0) { e.preventDefault(); moveFocus(b, e.key); return; }
      if (e.key === 'Escape') {
        if (cardFor) { hideCard(); e.preventDefault(); return; }
        if (st.sel) { var id = st.sel; closePanel(); var c = chipById(id); if (c) { setRoving(c); c.focus(); } e.preventDefault(); }
      }
    });
    body.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('.pt-chip');
      if (!b || !body.contains(b)) return;
      hideCard();
      setRoving(b);
      var id = b.getAttribute('data-id');
      if (st.sel === id) closePanel();
      else openPanel(id, {});
    });
    body.addEventListener('pointerover', function (e) {
      var b = e.target.closest && e.target.closest('.pt-chip');
      if (!b || e.pointerType === 'touch') return;
      clearTimeout(hideTimer);
      clearTimeout(cardTimer);
      if (cardFor === b.getAttribute('data-id')) return;
      cardTimer = setTimeout(function () { showCard(b); }, cardFor ? 30 : 110);
    });
    body.addEventListener('pointerout', function (e) {
      var b = e.target.closest && e.target.closest('.pt-chip');
      if (!b) return;
      if (e.relatedTarget && b.contains(e.relatedTarget)) return;
      clearTimeout(cardTimer);
      hideTimer = setTimeout(hideCard, 90);
    });
    body.addEventListener('focusin', function (e) {
      var b = e.target.closest && e.target.closest('.pt-chip');
      if (!b) return;
      setRoving(b);
      var fv = false;
      try { fv = b.matches(':focus-visible'); } catch (x) { fv = true; }
      if (fv) showCard(b);
    });
    body.addEventListener('focusout', function (e) {
      var b = e.target.closest && e.target.closest('.pt-chip');
      if (b && cardFor === b.getAttribute('data-id')) hideCard();
    });
    search.addEventListener('input', function () { st.q = norm(search.value); update(); });
    search.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter') return;
      var hits = (G.pop || []).filter(function (m) { return isLit(m); });
      if (hits.length === 1) { e.preventDefault(); openPanel(hits[0].id, {}); var c = chipById(hits[0].id); if (c) { setRoving(c); } }
    });

    /* ---------- the panel ---------- */
    function closePanel(silent) {
      if (G.panel) {
        clearTS(G.panel);
        if (G.panel.parentNode) G.panel.parentNode.removeChild(G.panel);
      }
      var prev = st.sel;
      G.panel = null; G.panelRow = null; G.panelSub = null;
      st.sel = null;
      if (prev && chipById(prev)) chipById(prev).setAttribute('aria-expanded', 'false');
      if (prev && chipById(prev)) chipById(prev).removeAttribute('aria-controls');
      placeGrid();
      if (!silent) update();
    }
    function openPanel(id, opts) {
      var m = byId[id];
      if (!m) return;
      opts = opts || {};
      closePanel(true);
      st.sel = id;
      st.ext = id;
      var p = buildPanel(m);
      var rk = rowOf(m), ck = colOf(m);
      G.panel = p; G.panelRow = rk;
      if (G.mode === 'table') {
        if (ck && G.rowEls[rk]) {
          var els = G.rowEls[rk], last = els[els.length - 1];
          G.grid.insertBefore(p, last.nextSibling);
        } else if (G.offEl) G.offEl.parentNode.insertBefore(p, G.offEl.nextSibling);
        else body.appendChild(p);
        placeGrid();
      } else {
        var key = rk + '|' + (ck || '–'), sub = G.subEls[key];
        G.panelSub = key;
        if (sub) sub.el.parentNode.insertBefore(p, sub.el.nextSibling);
        else if (G.secEls[rk]) G.secEls[rk].appendChild(p);
        else G.list.insertBefore(p, G.list.firstChild);
      }
      var c = chipById(id);
      for (var k in chipEls) chipEls[k].classList.toggle('ext', k === id && false);
      if (c) { c.setAttribute('aria-expanded', 'true'); c.setAttribute('aria-controls', p.id); }
      update();
      fillPanel(p, m);
      if (!opts.silent) document.dispatchEvent(new CustomEvent('atlas:select-method', { detail: { id: id, source: 'periodic' } }));
      if (!opts.noScroll) {
        var r = p.getBoundingClientRect(), vh = window.innerHeight, beh = A.reducedMotion() ? 'auto' : 'smooth';
        if (opts.fromPanel) { if (r.top < 70 || r.top > vh * 0.6) window.scrollBy({ top: r.top - 84, behavior: beh }); }
        else if (r.top > vh - 140) window.scrollBy({ top: r.top - vh * 0.42, behavior: beh });
      }
      if (opts.fromPanel) { var h4 = p.querySelector('h4'); if (h4) h4.focus({ preventScroll: true }); }
    }
    function notch(p, m) {
      var anchor = chipById(m.id);
      if (!anchor || anchor.hidden || !anchor.offsetParent) {
        var ck = colOf(m);
        anchor = G.mode === 'table' ? (ck ? G.cellEls[rowOf(m) + '|' + ck] : G.offCell) : null;
      }
      if (!anchor) { p.style.setProperty('--nx', '48px'); return; }
      var a = anchor.getBoundingClientRect(), pr = p.getBoundingClientRect();
      p.style.setProperty('--nx', clamp(a.left + a.width / 2 - pr.left, 20, pr.width - 20).toFixed(0) + 'px');
    }

    function linkTo(anchor) { return document.getElementById(anchor); }
    function buildPanel(m) {
      var p = A.h('section', { class: 'pt-panel pt-pending', id: uid + '-panel', 'aria-label': 'Arrows and obstructions of ' + m.name });
      p.style.setProperty('--k', kcol(m));
      var ph = A.h('div', { class: 'pt-ph' });
      var L = A.h('div', { class: 'pt-ph-l' });
      L.innerHTML = '<div class="pt-meta"><span><span class="sw"></span>' + A.esc(A.kindName(m.modification_kind)) + '</span><span>' + A.esc(yearVenue(m)) + '</span>' +
        (m.hf_peft ? '<span class="hf">HF PEFT</span>' : '') + (m.entry_type && m.entry_type !== 'method' ? '<span>' + A.esc(m.entry_type) + '</span>' : '') +
        (hasCore && !m.core ? '<span>not in the core table</span>' : '') + '</div>' +
        '<h4 tabindex="-1">' + labelHTML(m.name) + '</h4>' + (m.full_name && norm(m.full_name) !== norm(m.name) ? '<div class="full">' + A.esc(m.full_name) + '</div>' : '');
      var R = A.h('div', { class: 'pt-ph-r' });
      if (linkTo('catalog')) {
        var cb = A.h('button', { type: 'button', class: 'lk', text: 'Catalogue card ↓' });
        cb.addEventListener('click', function () {
          document.dispatchEvent(new CustomEvent('atlas:select-method', { detail: { id: m.id, source: 'periodic' } }));
          linkTo('catalog').scrollIntoView({ behavior: A.reducedMotion() ? 'auto' : 'smooth', block: 'start' });
        });
        R.appendChild(cb);
      }
      ALSO.forEach(function (x) {
        var t = linkTo(x.anchor);
        if (!t || (x.ids && x.ids.indexOf(m.id) < 0) || (x.coreOnly && hasCore && !m.core)) return;
        var n = t.querySelector('.fig-n');
        R.appendChild(A.h('a', { href: '#' + x.anchor, text: x.name + (n ? ' · ' + n.textContent.replace(/Figure\s*/i, 'Fig. ') : '') + ' ↗' }));
      });
      if (hasCore && !m.core && !st.all) {
        var sa = A.h('button', { type: 'button', class: 'lk', text: 'Show all ' + methods.length });
        sa.addEventListener('click', function () { st.all = true; seg.set(true); rebuild(); });
        R.appendChild(sa);
      }
      var x = A.h('button', { type: 'button', class: 'pt-x', 'aria-label': 'Close the panel of ' + m.name, html: '×' });
      x.addEventListener('click', function () { var id = m.id; closePanel(); var c = chipById(id); if (c) { setRoving(c); c.focus(); } });
      R.appendChild(x);
      ph.appendChild(L); ph.appendChild(R);
      p.appendChild(ph);
      /* seven coordinates; each one with a note opens it below the strip */
      var co = A.h('div', { class: 'pt-coords', role: 'group', 'aria-label': 'Seven coordinates' });
      var notes = A.h('div', { class: 'pt-conotes' });
      var noteEls = {}, coBtns = [];
      AX7.forEach(function (ax) {
        var k = cv(m, ax), hit = FAX.indexOf(ax) >= 0 && st.f[ax][k];
        var v = ax === 'kind' ? keyHTML(k) + (isR(m) && ROWBY[rowOf(m)] ? ' · ' + A.esc(ROWBY[rowOf(m)].name) : '') : keyHTML(k);
        var note = m.coord_notes && m.coord_notes[ax];
        var d = A.h(note ? 'button' : 'div', { class: 'pt-co' + (hit ? ' hit' : '') + (note ? ' has' : '') });
        if (note) { d.type = 'button'; d.setAttribute('aria-expanded', 'false'); d.setAttribute('aria-label', AXIS_SHORT[ax] + ' ' + keyText(k) + ': show why'); }
        d.innerHTML = '<span class="ax">' + AXIS_SHORT[ax] + (note ? ' <span class="why" aria-hidden="true">why</span>' : '') + '</span><span class="v">' + v + '</span><span class="gl">' + gloss(ax, k) + '</span>';
        if (note) {
          var ne = A.h('div', { class: 'pt-conote pt-tx', hidden: 'hidden', html: '<span class="lb">' + AXIS_SHORT[ax] + ' · ' + keyHTML(k) + '</span>' + A.esc(clean(note)) });
          noteEls[ax] = ne;
          notes.appendChild(ne);
          d.addEventListener('click', function () {
            var open = d.getAttribute('aria-expanded') !== 'true';
            coBtns.forEach(function (x) { x.b.setAttribute('aria-expanded', 'false'); x.n.hidden = true; });
            if (open) { d.setAttribute('aria-expanded', 'true'); ne.hidden = false; }
          });
          coBtns.push({ b: d, n: ne });
        }
        co.appendChild(d);
      });
      p.appendChild(co);
      p.appendChild(notes);
      /* formula and budget */
      var pf = A.h('div', { class: 'pt-pf' + (m.formula_latex && m.trainable_params ? '' : ' st') });
      if (m.formula_latex) { var fd = A.h('div', { class: 'pt-tx' }); fd.appendChild(A.h('div', { class: 'pt-sec', text: 'Formula' })); fd.appendChild(texLines(m.formula_latex, 'fxp', st.narrow ? 38 : 66)); pf.appendChild(fd); }
      if (m.trainable_params) { var bd = A.h('div', { class: 'pt-tx' }); bd.appendChild(A.h('div', { class: 'pt-sec', text: 'Budget |M| = dim Q' })); bd.appendChild(texLines(m.trainable_params, 'fxp', st.narrow ? 38 : 34)); pf.appendChild(bd); }
      p.appendChild(pf);
      /* diagram + list */
      var pb = A.h('div', { class: 'pt-pb' });
      var hd = A.h('div', { class: 'pt-hd' });
      var hhost = A.h('div', { class: 'pt-hasse' });
      hd.appendChild(hhost);
      hd.appendChild(A.h('div', { class: 'pt-hcap' }));
      var al = A.h('div', { class: 'pt-al' });
      pb.appendChild(hd); pb.appendChild(al);
      p.appendChild(pb);
      if (m.categorical_reading) p.appendChild(A.h('div', { class: 'pt-read pt-tx', html: '<span class="lb">Categorical reading</span>' + A.esc(clean(m.categorical_reading)) }));
      p.__m = m; p.__host = hhost; p.__list = al;
      return p;
    }
    function fillPanel(p, m) {
      p.classList.toggle('nr', p.clientWidth < 700);
      notch(p, m);
      buildArrowList(p, m);
      drawHasse(p.__host, m, p);
      typesetQ(p).then(function () { p.classList.remove('pt-pending'); });
      setTimeout(function () { p.classList.remove('pt-pending'); }, 6000);
    }

    /* the list of arrows and obstructions */
    var LIMIT = 5;
    function ruleTag(key, ob) {
      var r = RULES[key];
      var s = A.h('span', { class: 'pt-rule' + (ob ? ' ob' : ''), tabindex: '0', text: key || '?' });
      var tip = function (e) { A.tip.show('<div class="t">' + A.esc((key || '') + (r ? ' · ' + r.name : '')) + '</div>' + (r ? '<div>' + pretty(r.rule) + '</div>' : ''), evtFor(s, e)); };
      s.addEventListener('mouseenter', tip); s.addEventListener('focus', tip);
      s.addEventListener('mouseleave', A.tip.hide); s.addEventListener('blur', A.tip.hide);
      return s;
    }
    function nbButton(nid, m) {
      var n = byId[nid];
      if (!n) return A.h('span', { class: 'pt-nb self', text: nid });
      var b = A.h('button', { type: 'button', class: 'pt-nb' + (nid === m.id ? ' self' : ''), 'data-id': nid, html: labelHTML(shortName(n)), 'aria-label': nid === m.id ? n.name : 'Open the arrows of ' + n.name });
      b.style.setProperty('--k', kcol(n));
      if (nid !== m.id) b.addEventListener('click', function () { openPanel(nid, { fromPanel: true }); });
      return b;
    }
    function itemEl(kind, a, m) {
      var nid = (kind === 'o' || kind === 'x') ? a.t : (kind === 'i' || kind === 'y') ? a.s : m.id;
      var ob = kind === 'x' || kind === 'y';
      var li = A.h('li', { class: 'pt-ai' + (ob ? ' ob' : ''), 'data-k': kind + ':' + nid });
      var hh = A.h('div', { class: 'hh' });
      var dir = { o: '→', i: '→', s: '↻', x: '↛', y: '↛' }[kind];
      if (kind === 'o' || kind === 'x') { hh.appendChild(A.h('span', { class: 'dir', 'aria-hidden': 'true', text: dir })); hh.appendChild(nbButton(nid, m)); }
      else if (kind === 'i' || kind === 'y') { hh.appendChild(nbButton(nid, m)); hh.appendChild(A.h('span', { class: 'dir', 'aria-hidden': 'true', text: dir })); }
      else { hh.appendChild(A.h('span', { class: 'dir', 'aria-hidden': 'true', text: dir })); }
      var key = ob ? a.test : a.rule;
      hh.appendChild(ruleTag(key, ob));
      if (RULE_SHORT[key]) hh.appendChild(A.h('span', { class: 'pt-rn', text: RULE_SHORT[key] }));
      li.appendChild(hh);
      if (!ob && a.h) {
        var hm = A.h('div', { class: 'hm pt-tx' });
        var hl = splitTeX(a.h, st.narrow ? 34 : 46);
        hl.forEach(function (ln, k) {
          var row = A.h('div');
          if (k === 0 && !/^\s*h\b/.test(a.h)) row.appendChild(A.h('span', { class: 'lb', text: 'h :' }));
          row.appendChild(document.createTextNode('\\(' + (k ? '\\quad ' : '') + ln + '\\)'));
          hm.appendChild(row);
        });
        li.appendChild(hm);
      }
      if (ob && a.reason) li.appendChild(A.h('div', { class: 'rs pt-tx', html: A.esc(clean(a.reason)) }));
      if (ob && (OUT[a.s] || []).some(function (b) { return b.t === a.t; })) {
        li.appendChild(A.h('div', { class: 'both', text: 'An arrow ' + nodeLabel(a.s) + ' → ' + nodeLabel(a.t) + ' is also recorded, so the two concern different instances of ' + nodeLabel(a.t) + ' (another rank, pointing or frame); its note says which.' }));
      }
      if (!ob && a.note) {
        var det = A.h('details', { class: 'pt-tx' });
        det.appendChild(A.h('summary', { text: 'note' }));
        det.appendChild(A.h('div', { html: A.esc(clean(a.note)) }));
        li.appendChild(det);
      }
      li.addEventListener('mouseenter', function () { highlight([kind + ':' + nid], p0(li)); });
      li.addEventListener('mouseleave', function () { highlight([], p0(li)); });
      li.addEventListener('focusin', function () { highlight([kind + ':' + nid], p0(li)); });
      return li;
    }
    function p0(node) { return node.closest('.pt-panel'); }
    function buildArrowList(p, m) {
      var al = p.__list, name = m.name;
      al.innerHTML = '';
      var outs = (OUT[m.id] || []).filter(function (a) { return a.t !== m.id; }).sort(function (a, b) { return rank(a.t, b.t); });
      var ins = (IN[m.id] || []).filter(function (a) { return a.s !== m.id; }).sort(function (a, b) { return rank(a.s, b.s); });
      var selfs = (OUT[m.id] || []).filter(function (a) { return a.t === m.id; });
      var obo = (m.obstructions || []).filter(function (o) { return o && o.target; }).map(function (o) { return { s: m.id, t: o.target, test: o.test, reason: o.reason }; }).sort(function (a, b) { return rank(a.t, b.t); });
      var obi = (OBS_IN[m.id] || []).filter(function (o) { return o.s !== m.id; }).sort(function (a, b) { return rank(a.s, b.s); });
      function section(key, title, list, kind, cls, collapsed) {
        if (!list.length) return;
        var sec = A.h('div', { class: 'pt-als' + (cls ? ' ' + cls : ''), 'data-sec': key });
        var h5 = A.h('h5', { html: title + ' · <b>' + list.length + '</b>' });
        var ul = A.h('ul');
        var lim = collapsed ? 0 : LIMIT;
        list.slice(0, lim).forEach(function (a) { ul.appendChild(itemEl(kind, a, m)); });
        sec.appendChild(h5); sec.appendChild(ul);
        if (list.length > lim) {
          var mb = A.h('button', { type: 'button', class: 'pt-more', 'aria-expanded': 'false', text: (lim ? 'Show all ' + list.length : 'Show the ' + list.length) });
          mb.addEventListener('click', function () {
            list.slice(lim).forEach(function (a) { var li = itemEl(kind, a, m); li.classList.add('pt-tx'); ul.appendChild(li); });
            mb.remove();
            ul.classList.add('pt-pending');
            typesetQ(ul).then(function () { ul.classList.remove('pt-pending'); });
            setTimeout(function () { ul.classList.remove('pt-pending'); }, 6000);
          });
          sec.appendChild(mb);
          sec.__more = mb;
        }
        al.appendChild(sec);
      }
      section('o', 'Arrows out · ' + A.esc(name) + ' → N, N simulates it', outs, 'o');
      section('i', 'Arrows in · K → ' + A.esc(name) + ', it simulates K', ins, 'i');
      section('s', 'Arrows to itself · another size or stage', selfs, 's');
      section('x', 'No arrow ' + A.esc(name) + ' → N', obo, 'x', 'ob');
      section('y', 'No arrow K → ' + A.esc(name), obi, 'y', 'ob', obi.length > 4);
      if (!outs.length && !ins.length && !selfs.length && !obo.length && !obi.length) al.appendChild(A.h('p', { class: 'pt-empty', text: 'No arrows or obstructions are recorded for this method yet.' }));
      else if (!outs.length && !ins.length && !selfs.length) al.insertBefore(A.h('p', { class: 'pt-empty', text: 'No simulation arrows are recorded beyond those of rule A1.' }), al.firstChild);
    }
    function highlight(keys, p) {
      if (!p) return;
      var set = {};
      keys.forEach(function (k) { set[k] = 1; });
      function hit(ks) { return String(ks || '').split(' ').some(function (k) { return set[k]; }); }
      Array.prototype.forEach.call(p.querySelectorAll('.pt-hasse [data-k]'), function (e) {
        var on = hit(e.getAttribute('data-k'));
        e.classList.toggle('hl', on);
        var v = e.querySelector('path.v');
        if (v && v.__mk) v.setAttribute('marker-end', 'url(#' + (on ? v.__mk.hl : v.__mk.base) + ')');
      });
      Array.prototype.forEach.call(p.querySelectorAll('.pt-al [data-k]'), function (e) { e.classList.toggle('hl', hit(e.getAttribute('data-k'))); });
    }

    /* ---------- the mini-Hasse diagram ---------- */
    function cssFont(tok) { return A.css(tok) || 'serif'; }
    function bz(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t) {
      var u = 1 - t;
      return [u * u * u * p0x + 3 * u * u * t * p1x + 3 * u * t * t * p2x + t * t * t * p3x, u * u * u * p0y + 3 * u * u * t * p1y + 3 * u * t * t * p2y + t * t * t * p3y];
    }
    function drawHasse(host, m, p) {
      var W = Math.max(250, Math.floor(host.clientWidth || 300));
      host.__w = W;
      host.innerHTML = '';
      var id = m.id;
      var outs = (OUT[id] || []).filter(function (a) { return a.t !== id; });
      var ins = (IN[id] || []).filter(function (a) { return a.s !== id; });
      var selfs = (OUT[id] || []).filter(function (a) { return a.t === id; });
      function grp(list, f) { var g = {}; list.forEach(function (a) { (g[a[f]] = g[a[f]] || []).push(a); }); return g; }
      var up = grp(outs, 't'), dn = grp(ins, 's');
      var mutual = Object.keys(up).filter(function (k) { return dn[k]; }).sort(rank);
      var upIds = Object.keys(up).filter(function (k) { return !dn[k]; }).sort(rank);
      var dnIds = Object.keys(dn).filter(function (k) { return !up[k]; }).sort(rank);
      var obsG = grp((m.obstructions || []).filter(function (o) { return o && o.target; }).map(function (o) { return { t: o.target, test: o.test }; }), 't');
      var obsIds = Object.keys(obsG).sort(rank);
      var spine = isR(m);
      function rulesOfA(list, f) { var o = []; (list || []).forEach(function (a) { if (o.indexOf(a[f]) < 0) o.push(a[f]); }); return o.join('·'); }
      var NF = '500 12px ' + cssFont('--f-body'), MF = '600 17px ' + cssFont('--f-display'), TF = '500 12px ' + cssFont('--f-ui');
      var NH = 22, MH = 34, GAP = 9, PITCH = 46, SIDE = 46, MARG = 4, TH = 17;
      /* parameter at which the arrow curve (control points at mid-height) reaches height yT: the rule tag
         of an arrow from an outer row sits in the middle of the gap between the rows, clear of both */
      function tAtY(sy, ey, yT) {
        var f = (yT - sy) / (ey - sy), lo = 0, hi = 1;
        for (var i = 0; i < 28; i++) { var t = (lo + hi) / 2; if (1.5 * t * (1 - t) + t * t * t < f) lo = t; else hi = t; }
        return (lo + hi) / 2;
      }
      function nw(nid) { return Math.ceil(tw(nodeLabel(nid), NF)) + 20; }
      var Mw = Math.ceil(tw(plainLabel(m), MF)) + 30;
      if (selfs.length) Mw = Math.max(Mw, 124);
      var cx = W / 2;
      var mwL = mutual.reduce(function (a, k) { return Math.max(a, nw(k)); }, 0), mwR = obsIds.reduce(function (a, k) { return Math.max(a, nw(k)); }, 0);
      /* place M (and the spine) off-centre when the side columns need room; obstructions first */
      function mutTag(k) { return rulesOfA(up[k], 'rule') + ' ⇄ ' + rulesOfA(dn[k], 'rule'); }
      var tagL = mutual.reduce(function (a, k) { return Math.max(a, Math.ceil(tw(mutTag(k), TF)) + 14); }, 0);
      /* the obstruction column sits far enough out for its test tag to clear the no-arrow slash */
      function obsText(k) { return obsG[k].map(function (o) { return o.test; }).join('·'); }
      var SIDE_R = obsIds.reduce(function (a, k) { return Math.max(a, Math.ceil(tw(obsText(k), TF)) + 8 + 50); }, SIDE);
      var needR = obsIds.length ? Mw / 2 + SIDE_R + mwR + MARG : Mw / 2 + MARG;
      var obsSide = !obsIds.length || (obsIds.length <= 4 && needR + Mw / 2 + MARG <= W);
      if (!obsSide) needR = Mw / 2 + MARG;
      var needL = mutual.length ? Mw / 2 + SIDE + tagL + mwL + MARG : Mw / 2 + MARG;
      var mutSide = !mutual.length || (mutual.length <= 4 && needL + needR <= W);
      if (!mutSide) needL = Mw / 2 + MARG;
      cx = clamp(W / 2, needL, Math.max(needL, W - needR));
      var sideOK = obsSide;
      var mutualUp = [];
      if (!mutSide) { mutualUp = mutual; mutual = []; }
      var obsShown = obsSide ? obsIds : [];
      var upAll = mutualUp.concat(upIds);
      var spineGap = spine ? 30 : GAP;
      function packNodes(nodes, maxRows) {
        var halfL = cx - MARG - spineGap / 2, halfR = W - cx - MARG - spineGap / 2, half = Math.max(halfL, halfR), rows = [], cur = { L: [], R: [], lw: 0, rw: 0 }, placed = 0;
        function fits(side, w) { var u = side === 'L' ? cur.lw : cur.rw; return u + (u ? GAP : 0) + w <= (side === 'L' ? halfL : halfR); }
        function put(side, n) { if (side === 'L') { cur.lw += (cur.lw ? GAP : 0) + n.w; cur.L.push(n); } else { cur.rw += (cur.rw ? GAP : 0) + n.w; cur.R.push(n); } }
        for (var i = 0; i < nodes.length; i++) {
          var n = nodes[i], side = cur.lw <= cur.rw ? 'L' : 'R';
          if (!fits(side, n.w)) side = side === 'L' ? 'R' : 'L';
          if (!fits(side, n.w)) {
            rows.push(cur);
            if (rows.length >= maxRows) { cur = null; break; }
            cur = { L: [], R: [], lw: 0, rw: 0 };
            side = 'L';
            if (!fits(side, n.w)) { n.w = Math.floor(half); }
          }
          put(side, n); placed++;
        }
        if (cur && (cur.L.length || cur.R.length)) rows.push(cur);
        return { rows: rows, placed: placed };
      }
      function tier(ids, maxRows) {
        var nodes = ids.map(function (nid) { return { id: nid, w: nw(nid) }; });
        var res = packNodes(nodes, maxRows);
        if (res.placed < nodes.length) {
          for (var k = res.placed - 1; k >= 0; k--) {
            var extra = nodes.length - k;
            var more = { id: '__more', more: extra, w: Math.ceil(tw('+' + extra + ' more', TF)) + 18 };
            var r2 = packNodes(nodes.slice(0, k).map(function (n) { return { id: n.id, w: nw(n.id) }; }).concat([more]), maxRows);
            if (r2.placed === k + 1) { res = r2; break; }
          }
        }
        return res.rows;
      }
      var maxRows = W < 420 ? 3 : 2;
      var upRows = tier(upAll, maxRows), dnRows = tier(dnIds, maxRows);
      function positionRow(row, y) {
        var x = cx - spineGap / 2, all = [];
        row.L.forEach(function (n) { n.x = x - n.w / 2; x -= n.w + GAP; n.y = y; all.push(n); });
        x = cx + spineGap / 2;
        row.R.forEach(function (n) { n.x = x + n.w / 2; x += n.w + GAP; n.y = y; all.push(n); });
        if (!spine && all.length) {
          var lo = Math.min.apply(null, all.map(function (n) { return n.x - n.w / 2; })), hi = Math.max.apply(null, all.map(function (n) { return n.x + n.w / 2; }));
          var sh = cx - (lo + hi) / 2;
          if (lo + sh < MARG) sh = MARG - lo;
          if (hi + sh > W - MARG) sh = W - MARG - hi;
          all.forEach(function (n) { n.x += sh; });
        }
        return all;
      }
      var stackN = Math.max(mutual.length, obsShown.length);
      var stackH = stackN ? (stackN - 1) * 30 + NH : 0;
      var dU = Math.max(selfs.length ? 74 : 62, stackH / 2 + 36), dD = Math.max(62, stackH / 2 + 36);
      var upNodes = [], dnNodes = [];
      upRows.forEach(function (row, k) { upNodes = upNodes.concat(positionRow(row, -(dU + k * PITCH)).map(function (n) { n.row = k; return n; })); });
      dnRows.forEach(function (row, k) { dnNodes = dnNodes.concat(positionRow(row, dD + k * PITCH).map(function (n) { n.row = k; return n; })); });
      var yF = spine ? (upRows.length ? -(dU + upRows.length * PITCH) : -Math.max(selfs.length ? 70 : 58, stackH / 2 + 32)) : null;
      var yZ = spine ? (dnRows.length ? dD + dnRows.length * PITCH : Math.max(58, stackH / 2 + 32)) : null;
      var sideL = mutual.map(function (nid, i) { var w = nw(nid); return { id: nid, w: w, x: cx - Mw / 2 - SIDE - tagL - w / 2, y: (i - (mutual.length - 1) / 2) * 30 }; });
      var sideR = obsShown.map(function (nid, i) { var w = nw(nid); return { id: nid, w: w, x: cx + Mw / 2 + SIDE_R + w / 2, y: (i - (obsShown.length - 1) / 2) * 30 }; });
      var ys = [-MH / 2, MH / 2];
      upNodes.concat(dnNodes, sideL, sideR).forEach(function (n) { ys.push(n.y - NH / 2, n.y + NH / 2); });
      if (spine) ys.push(yF - NH / 2 - 2, yZ + NH / 2 + 2);
      if (selfs.length) ys.push(-MH / 2 - 30);
      var y0 = Math.min.apply(null, ys) - 6, y1 = Math.max.apply(null, ys) + 6, H = Math.ceil(y1 - y0);

      var svg = d3.select(host).append('svg').attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H)
        .attr('role', 'group').attr('aria-label', 'Mini-Hasse diagram of ' + m.name + ': ' + plural(Object.keys(up).length, 'method') + ' simulating it above, ' + plural(Object.keys(dn).length, 'method') + ' it simulates below' + (spine ? ', between Frozen and Full FT' : '') + '.');
      var defs = svg.append('defs');
      var MK = { base: uid + '-ah', hl: uid + '-ahh', sp: uid + '-ahs', ob: uid + '-aho' };
      [['base', 'var(--ink-2)'], ['hl', 'var(--tide)'], ['sp', 'var(--ink-3)'], ['ob', 'var(--seal)']].forEach(function (d) {
        defs.append('marker').attr('id', MK[d[0]]).attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 8).attr('markerHeight', 8)
          .attr('markerUnits', 'userSpaceOnUse').attr('orient', 'auto')
          .append('path').attr('d', 'M1.5,1.5 L9,5 L1.5,8.5').attr('fill', 'none').attr('style', 'stroke:' + d[1] + ';stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round');
      });
      var root = svg.append('g').attr('transform', 'translate(0,' + (-y0) + ')');
      var gE = root.append('g'), gT = root.append('g'), gN = root.append('g');
      function edge(d, cls, keys, mk, hlMk) {
        var g = gE.append('g').attr('class', 'e ' + cls).attr('data-k', keys);
        var v = g.append('path').attr('class', 'v').attr('d', d).attr('marker-end', 'url(#' + mk + ')');
        v.node().__mk = { base: mk, hl: hlMk || mk };
        g.append('path').attr('class', 'ht').attr('d', d);
        return g;
      }
      function tag(x, y, text, cls, keys) {
        var w = Math.ceil(tw(text, TF)) + 8;
        var g = gT.append('g').attr('class', 'tg ' + (cls || '')).attr('transform', 'translate(' + x.toFixed(1) + ',' + y.toFixed(1) + ')');
        if (keys) g.attr('data-k', keys);
        g.append('rect').attr('x', -w / 2).attr('y', -TH / 2).attr('width', w).attr('height', TH).attr('rx', 3);
        var tx = g.append('text').attr('x', 0).attr('y', 4.3).attr('text-anchor', 'middle');
        /* Greek in the body serif, as everywhere in the figure */
        String(text).split(/([ΘθρΦ])/).forEach(function (part, i) { if (part) { var t = tx.append('tspan').text(part); if (i % 2) t.attr('class', 'gk'); } });
        return g;
      }
      function rulesOf(list, f) { return rulesOfA(list, f); }
      var kxHiUp = selfs.length ? Mw / 2 - 38 : Mw / 2 - 10;
      function kx(x, upward) { return cx + clamp((x - cx) * 0.3, -(Mw / 2 - 10), upward ? kxHiUp : Mw / 2 - 10); }
      /* interval spine (A1) */
      if (spine) {
        edge('M' + cx + ',' + (-MH / 2) + ' L' + cx + ',' + (yF + NH / 2), 'sp', 'a1', MK.sp);
        edge('M' + cx + ',' + (yZ - NH / 2) + ' L' + cx + ',' + (MH / 2), 'sp', 'a1', MK.sp);
        var wF = Math.ceil(tw('Full FT', 'italic 400 12px ' + cssFont('--f-body'))) + 18, wZ = Math.ceil(tw('Frozen', 'italic 400 12px ' + cssFont('--f-body'))) + 18;
        var tF = 'A1 · h = ρ', tZ = 'A1 · h = const';
        tag(cx - wF / 2 - 8 - (Math.ceil(tw(tF, TF)) + 8) / 2, yF, tF, 'sp');
        tag(cx - wZ / 2 - 8 - (Math.ceil(tw(tZ, TF)) + 8) / 2, yZ, tZ, 'sp');
      }
      /* arrows out (M → N, N above) */
      upNodes.forEach(function (n) {
        var sx = kx(n.x, true), sy = -MH / 2, ex = n.x, ey = n.y + NH / 2, my = (sy + ey) / 2;
        var keys = n.id === '__more' ? 'more-o' : 'o:' + n.id + (dn[n.id] ? ' i:' + n.id : '');
        edge('M' + sx + ',' + sy + ' C' + sx + ',' + my + ' ' + ex + ',' + my + ' ' + ex + ',' + ey, n.id === '__more' ? 'mo' : '', keys, MK.base, MK.hl);
        if (dn[n.id]) { /* mutual, drawn in the up tier on narrow stages: add the return arrow */
          edge('M' + (ex + 6) + ',' + ey + ' C' + (ex + 6) + ',' + my + ' ' + (sx + 8) + ',' + my + ' ' + (sx + 8) + ',' + sy, '', 'i:' + n.id + ' o:' + n.id, MK.base, MK.hl);
        }
        var t = bz(sx, sy, sx, my, ex, my, ex, ey, n.row ? tAtY(sy, ey, ey + (PITCH - NH) / 2) : 0.74);
        var label = n.id === '__more' ? n.more + '×' : rulesOf(up[n.id], 'rule') + (dn[n.id] ? ' ⇄ ' + rulesOf(dn[n.id], 'rule') : '');
        tag(t[0], t[1], label, '', keys);
      });
      /* arrows in (K → M, K below) */
      dnNodes.forEach(function (n) {
        var sx = n.x, sy = n.y - NH / 2, ex = kx(n.x), ey = MH / 2, my = (sy + ey) / 2;
        var keys = n.id === '__more' ? 'more-i' : 'i:' + n.id;
        edge('M' + sx + ',' + sy + ' C' + sx + ',' + my + ' ' + ex + ',' + my + ' ' + ex + ',' + ey, n.id === '__more' ? 'mo' : '', keys, MK.base, MK.hl);
        var t = bz(sx, sy, sx, my, ex, my, ex, ey, n.row ? tAtY(sy, ey, sy - (PITCH - NH) / 2) : 0.26);
        tag(t[0], t[1], n.id === '__more' ? n.more + '×' : rulesOf(dn[n.id], 'rule'), '', keys);
      });
      /* mutual arrows (left) */
      sideL.forEach(function (n) {
        var ax = cx - Mw / 2, bx = n.x + n.w / 2, keys = 'o:' + n.id + ' i:' + n.id;
        var c1 = (ax + bx) / 2;
        edge('M' + ax + ',' + (-5) + ' C' + c1 + ',' + (-5) + ' ' + c1 + ',' + (n.y - 5) + ' ' + bx + ',' + (n.y - 5), '', keys, MK.base, MK.hl);
        edge('M' + bx + ',' + (n.y + 5) + ' C' + c1 + ',' + (n.y + 5) + ' ' + c1 + ',' + 5 + ' ' + ax + ',' + 5, '', keys, MK.base, MK.hl);
        var lab = mutTag(n.id);
        tag(bx + 6 + (Math.ceil(tw(lab, TF)) + 8) / 2, n.y, lab, '', keys);
      });
      /* obstructions (right): no arrow M → N */
      sideR.forEach(function (n) {
        var ax = cx + Mw / 2, bx = n.x - n.w / 2, keys = 'x:' + n.id, c1 = (ax + bx) / 2;
        var g = edge('M' + ax + ',0 C' + c1 + ',0 ' + c1 + ',' + n.y + ' ' + bx + ',' + n.y, 'ob', keys, MK.ob, MK.ob);
        /* the slash just before the test tag, and the tag just short of the node it rules out (same curve law in x as in y) */
        var otext = obsText(n.id), wT = Math.ceil(tw(otext, TF)) + 8, xS = Math.max(ax + 8, bx - 14 - wT);
        var mid = bz(ax, 0, c1, 0, c1, n.y, bx, n.y, tAtY(ax, bx, xS));
        g.append('path').attr('class', 'sl').attr('d', 'M' + (mid[0] - 3.5) + ',' + (mid[1] + 6) + ' L' + (mid[0] + 3.5) + ',' + (mid[1] - 6));
        var nr = bz(ax, 0, c1, 0, c1, n.y, bx, n.y, tAtY(ax, bx, bx - 4 - wT / 2));
        tag(nr[0], nr[1], otext, 'ob', keys);
      });
      /* arrows to itself */
      if (selfs.length) {
        var lx = cx + Mw / 2 - 30, ly = -MH / 2, rx = cx + Mw / 2 - 8;
        edge('M' + lx + ',' + ly + ' C' + (lx - 4) + ',' + (ly - 28) + ' ' + (rx + 4) + ',' + (ly - 28) + ' ' + rx + ',' + (ly - 1), '', 's:' + id, MK.base, MK.hl);
      }
      /* nodes */
      function node(n, cls) {
        var isMore = n.id === '__more', nm = byId[n.id];
        var g = gN.append('g').attr('class', 'n ' + (cls || '') + (isMore ? ' mr' : '') + (nm && hasCore && !nm.core ? ' nc' : ''))
          .attr('transform', 'translate(' + n.x.toFixed(1) + ',' + n.y.toFixed(1) + ')').attr('tabindex', 0).attr('role', 'button');
        var col = nm ? kcol(nm) : A.css('--ink-3');
        if (!isMore && cls !== 'ob') {
          g.append('rect').attr('class', 'b').attr('x', -n.w / 2).attr('y', -NH / 2).attr('width', n.w).attr('height', NH).attr('rx', 4)
            .attr('style', 'fill:color-mix(in srgb,' + col + ' 15%,var(--paper));stroke:' + col);
          g.append('rect').attr('x', -n.w / 2 + 0.5).attr('y', -NH / 2 + 0.5).attr('width', 3).attr('height', NH - 1).attr('rx', 1.5).attr('style', 'fill:' + col);
        } else {
          g.append('rect').attr('class', 'b').attr('x', -n.w / 2).attr('y', -NH / 2).attr('width', n.w).attr('height', NH).attr('rx', 4);
        }
        g.append('text').attr('x', isMore ? 0 : 1.5).attr('y', 4.2).attr('text-anchor', 'middle').text(isMore ? '+' + n.more + ' more' : nodeLabel(n.id));
        var keys = isMore ? [n.dir === 'up' ? 'more-o' : 'more-i'] : ['o:' + n.id, 'i:' + n.id, 'x:' + n.id];
        var label = isMore ? 'Show all ' + (n.dir === 'up' ? 'arrows out' : 'arrows in') : (nm ? 'Open the arrows of ' + nm.name : n.id) + (cls === 'ob' ? ' (no arrow: ' + obsG[n.id].map(function (o) { return o.test; }).join(', ') + ')' : '');
        g.attr('aria-label', label).attr('data-k', keys.join(' ')).attr('data-id', n.id);
        g.on('mouseenter', function (e) {
          highlight(keys, p);
          if (!isMore) A.tip.show('<div class="t">' + A.esc(nm ? nm.name : n.id) + '</div><div>' + edgeSummary(n.id, cls) + '</div>', e);
        }).on('mousemove', function (e) { A.tip.move(e); })
          .on('mouseleave', function () { highlight([], p); A.tip.hide(); })
          .on('focus', function () { highlight(keys, p); })
          .on('blur', function () { highlight([], p); })
          .on('click', function () { A.tip.hide(); act(); })
          .on('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
        function act() {
          if (isMore) {
            var sec = p.querySelector('.pt-als[data-sec="' + (n.dir === 'up' ? 'o' : 'i') + '"]');
            if (sec && sec.__more) sec.__more.click();
            if (sec) sec.scrollIntoView({ block: 'nearest', behavior: A.reducedMotion() ? 'auto' : 'smooth' });
          } else if (nm) openPanel(n.id, { fromPanel: true });
        }
      }
      function edgeSummary(nid, cls) {
        var s = [];
        (up[nid] || []).forEach(function (a) { s.push(A.esc(m.name) + ' → ' + A.esc(nodeLabel(nid)) + ' · ' + a.rule + ' ' + (RULE_SHORT[a.rule] || '')); });
        (dn[nid] || []).forEach(function (a) { s.push(A.esc(nodeLabel(nid)) + ' → ' + A.esc(m.name) + ' · ' + a.rule + ' ' + (RULE_SHORT[a.rule] || '')); });
        (obsG[nid] || []).forEach(function (o) { s.push('no arrow ' + A.esc(m.name) + ' → ' + A.esc(nodeLabel(nid)) + ' · ' + o.test + ' ' + (RULE_SHORT[o.test] || '')); });
        return s.join('<br>') + '<br><span class="muted">Select to re-centre on it.</span>';
      }
      upNodes.forEach(function (n) { if (n.id === '__more') n.dir = 'up'; node(n, ''); });
      dnNodes.forEach(function (n) { if (n.id === '__more') n.dir = 'down'; node(n, ''); });
      sideL.forEach(function (n) { node(n, ''); });
      sideR.forEach(function (n) { node(n, 'ob'); });
      if (spine) {
        [['Full FT', yF, 'TERMINAL'], ['Frozen', yZ, 'INITIAL']].forEach(function (d) {
          var w = Math.ceil(tw(d[0], 'italic 400 12px ' + cssFont('--f-body'))) + 18;
          var g = gN.append('g').attr('class', 'ax').attr('transform', 'translate(' + cx + ',' + d[1] + ')');
          g.append('rect').attr('x', -w / 2).attr('y', -NH / 2).attr('width', w).attr('height', NH).attr('rx', 4);
          g.append('text').attr('x', 0).attr('y', 4).attr('text-anchor', 'middle').text(d[0]);
          g.append('text').attr('class', 's').attr('x', w / 2 + 6).attr('y', 3).text(d[2]);
        });
      }
      var col0 = kcol(m);
      var gc = gN.append('g').attr('class', 'c').attr('transform', 'translate(' + cx + ',0)');
      gc.append('rect').attr('x', -Mw / 2).attr('y', -MH / 2).attr('width', Mw).attr('height', MH).attr('rx', 5)
        .attr('style', 'fill:color-mix(in srgb,' + col0 + ' 22%,var(--paper));stroke:' + col0 + ';stroke-width:1.6');
      gc.append('text').attr('x', 0).attr('y', 6).attr('text-anchor', 'middle').text(plainLabel(m));
      /* caption under the diagram */
      var cap = host.parentNode.querySelector('.pt-hcap');
      var hidden = [];
      if (!sideOK && obsIds.length) hidden.push('obstructions are listed only');
      var kindNote = spine ? 'Dashed: the interval Frozen → ' + A.esc(plainLabel(m)) + ' → Full FT of rule A1, which every weight-space method has (Obs I.4).'
        : ({ E: 'A pointed extension: it lives on a larger architecture and is compared through the realisation Φ<sub>f</sub>, outside Meth(Θ, θ<sub>0</sub>), so the A1 interval is not drawn.',
          E0: 'An unpointed extension: no parameter value gives back f, so it is not an object of Meth(Θ, θ<sub>0</sub>) and rule A1 does not apply.',
          B: 'A backward lens or schedule: the forward map is id<sub>Θ</sub>, so the A1 interval of reparametrisations is not drawn.',
          P: 'A post-hoc operation Θ<sup>N</sup> → Θ on trained points, not an object of Meth(Θ, θ<sub>0</sub>); rule A1 is not drawn.',
          F_T: 'A family T → Q of a method; its arrows compare the family pointwise, and rule A1 is not drawn.' }[cv(m, 'kind')] || '');
      cap.innerHTML = 'Up: methods that simulate ' + A.esc(plainLabel(m)) + '. Down: methods it simulates. ' + (obsShown.length ? 'Right, in seal: no arrow. ' : '') + (mutual.length ? 'Left: arrows both ways. ' : '') + (selfs.length ? 'The loop: an arrow to itself (' + rulesOf(selfs, 'rule') + ', another size or stage). ' : '') + kindNote + (hidden.length ? ' On this width the ' + hidden.join(', ') + '.' : '');
    }

    /* ---------- external selection (catalogue, timeline, graph) ---------- */
    document.addEventListener('atlas:select-method', function (e) {
      var d = e && e.detail, id = d && (typeof d === 'string' ? d : d.id);
      if (!id || !byId[id] || (d && d.source === 'periodic') || id === st.sel) return;
      st.ext = id;
      for (var k in chipEls) chipEls[k].classList.toggle('ext', k === id);
    });

    /* ---------- theme, resize, fonts ---------- */
    function repaint() {
      for (var id in chipEls) chipEls[id].style.setProperty('--k', kcol(byId[id]));
      Array.prototype.forEach.call(body.querySelectorAll('.pt-nb[data-id]'), function (b) { b.style.setProperty('--k', kcol(byId[b.getAttribute('data-id')])); });
      buildLegend();
      if (G.panel) { G.panel.style.setProperty('--k', kcol(G.panel.__m)); drawHasse(G.panel.__host, G.panel.__m, G.panel); }
      if (cardFor && byId[cardFor]) card.style.setProperty('--k', kcol(byId[cardFor]));
    }
    A.onTheme(repaint);
    var lastW = 0;
    function onResize() {
      var w = el.clientWidth;
      if (!w || w === lastW) return;
      lastW = w;
      if ((w < 600) !== st.narrow) { rebuild(); return; }
      layoutColumns();
      if (G.panel) {
        var p = G.panel;
        p.classList.toggle('nr', p.clientWidth < 700);
        if (Math.abs((p.__host.__w || 0) - p.__host.clientWidth) > 2) drawHasse(p.__host, p.__m, p);
        notch(p, p.__m);
      }
    }
    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(onResize, 80)).observe(el);
    else window.addEventListener('resize', A.debounce(onResize, 120));
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { layoutColumns(); if (G.panel) { drawHasse(G.panel.__host, G.panel.__m, G.panel); notch(G.panel, G.panel.__m); } });

    /* ---------- complete at rest ---------- */
    lastW = el.clientWidth;
    rebuild();

    /* hook for headless state tests */
    el.__periodic = {
      state: st, open: function (id) { openPanel(id, { silent: true, noScroll: true }); }, close: function () { closePanel(); },
      filter: applyFilter, all: function (v) { st.all = !!v; seg.set(!!v); rebuild(); },
      hover: function (id) { var c = chipById(id); if (c) showCard(c); },
      search: function (q) { search.value = q; st.q = norm(q); update(); },
      info: function () { return G.layoutInfo; },
      split: splitTeX,
      relayout: function () { layoutColumns(); return G.layoutInfo; },
      lit: function () { return (G.pop || []).filter(function (m) { return !isActive() || isLit(m); }).map(function (m) { return m.id; }); }
    };
  });
})();
