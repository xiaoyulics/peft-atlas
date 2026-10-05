/* Fig. "interval" — Figure 1: every weight-space method lives between doing nothing and doing everything.
   Exposé I of theory/framework.md: Def I.2 (methods and simulations), Remark I.3, Obs I.4(a) (the frozen model
   is initial, full fine-tuning is terminal, and the unique arrow M -> Full FT is rho_M itself), and the arrow rules
   A1-A5 / obstruction tests O1-O3 of Exposé VIII.C.

   Panel A draws the commutative triangle
        Frozen = ({*}, *)  --const_{q0}-->  M = (Q_M, q0)  --rho_M (merge)-->  Full FT = (Theta, theta0)
   whose composite is the inclusion {*} -> Theta, * |-> theta0.  The selected method's catalogue data decide how
   each arrow is drawn (nothing is hand-set per method):
     - const_{q0} exists exactly when rho_M(q0) = theta0: base point regular / apex / neutral.  A pointing defect
       (rho(q0) = theta0 + e) leaves only a weak (unpointed) arrow, so the triangle commutes only up to e; an
       unpointed method (no q0 realises f) has no such arrow at all.
     - rho_M is the canonical map to the terminal object when the merge type is M1 / M1-> (and for backward lenses,
       whose forward map is id_Theta); Mq keeps rho_M but the exact merge leaves the quantised type; M-infinity
       means there is no map back into Theta (extensions, input-dependent routing).
   |Q_M| is evaluated from the entry's catalogued trainable-parameter formula at d_in = d_out = d = 4096, r = 16
   (one 4096 x 4096 linear layer, |Theta| = 4096^2).  The evaluator accepts only +,-,*,/,^, parentheses, \frac and
   the symbols r, d_in, d_out, d (and r_init, r_max, r_l read as r); a formula with any other symbol (b, n, k, L,
   ...) is shown, not evaluated.  Whole-model budgets are not reported per layer.
   The sketch inside Theta shows the shape class of the image germ (the "shape" coordinate, Exposé VIII.A); the
   sketch inside Q_M shows the scalar slice (b, a) with the fibres {ba = c} when the gauge is multiplicative
   (scalar / torus / GL), where q0 sits on the singular fibre {ba = 0} exactly at an apex.  Both are schematic.

   Panel B lists the outgoing simulations of the source method (A1 into Full FT, plus every catalogued arrow
   M -> N with its rule and explicit h), the incoming ones, and the obstructions (no arrow M -> N, with the failed
   test).  Selecting one draws the triangle Q_M --h--> Q_N over Theta with rho_N o h = rho_M.

   Chips are coloured by Atlas.kindColor(modification_kind), the site-wide convention.  Clicking a method dispatches
   atlas:select-method so the catalogue opens its card. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var D_IN = 4096, D_OUT = 4096, RANK = 16, THETA = D_IN * D_OUT;
  var PER1 = 7600, PER2 = 5600;          // animation periods (ms)

  /* ======================= budget evaluator (pure) ======================= */
  function matchBrace(s, i) {
    var d = 0;
    for (var j = i; j < s.length; j++) {
      var c = s[j];
      if (c === '\\') { j++; continue; }
      if (c === '{') d++; else if (c === '}') { d--; if (d === 0) return j; }
    }
    return -1;
  }
  /* the leading clause: stop at a top-level ; , = < > or \quad, \approx, \le ... (outside braces and parentheses) */
  function firstClause(s) {
    var d = 0, pd = 0;
    for (var i = 0; i < s.length; i++) {
      var c = s[i];
      if (c === '\\') {
        if (d === 0 && pd <= 0 && /^\\(quad|qquad|approx|le|leq|ge|geq|in|sim|Rightarrow)(?![A-Za-z])/.test(s.slice(i))) return s.slice(0, i);
        i++; continue;
      }
      if (c === '{') d++; else if (c === '}') d--;
      else if (c === '(') pd++; else if (c === ')') pd--;
      else if (d === 0 && pd <= 0 && (c === ';' || c === ',' || c === '=' || c === '<' || c === '>')) return s.slice(0, i);
    }
    return s;
  }
  /* split at the first top-level \text{...}: the math before it, the qualifier, and what follows */
  function splitText(s) {
    var d = 0;
    for (var i = 0; i < s.length; i++) {
      var c = s[i];
      if (c === '\\') {
        if (d === 0) {
          var m = /^\\text(?:rm|it|tt|bf|sf)?\s*\{/.exec(s.slice(i));
          if (m) { var b0 = i + m[0].length - 1, b1 = matchBrace(s, b0); return { pre: s.slice(0, i), qual: b1 > 0 ? s.slice(b0 + 1, b1) : '', rest: b1 > 0 ? s.slice(b1 + 1) : '' }; }
        }
        i++; continue;
      }
      if (c === '{') d++; else if (c === '}') d--;
    }
    return { pre: s, qual: '', rest: '' };
  }
  function texToExpr(t) {
    var s = ' ' + t + ' ', alias = [];
    s = s.replace(/\\(?:left|right|bigg|Bigg|big|Big)[lr]?/g, ' ').replace(/\\(?:textstyle|displaystyle|scriptstyle)/g, ' ');
    s = s.replace(/\\[,;:! ]/g, ' ').replace(/~/g, ' ');
    for (var g = 0; g < 12; g++) {
      var m = /\\[dt]?frac\s*\{/.exec(s);
      if (!m) break;
      var a0 = m.index + m[0].length - 1, a1 = matchBrace(s, a0);
      if (a1 < 0) return null;
      var b0 = a1 + 1;
      while (s[b0] === ' ') b0++;
      if (s[b0] !== '{') return null;
      var b1 = matchBrace(s, b0);
      if (b1 < 0) return null;
      s = s.slice(0, m.index) + '((' + s.slice(a0 + 1, a1) + ')/(' + s.slice(b0 + 1, b1) + '))' + s.slice(b1 + 1);
    }
    s = s.replace(/\\cdot|\\times/g, '*');
    s = s.replace(/d_\{\s*(?:\\mathrm\{in\}|\\rm in|in)\s*\}/g, ' I ').replace(/d_\{\s*(?:\\mathrm\{out\}|\\rm out|out)\s*\}/g, ' O ');
    s = s.replace(/r_\{\s*(init|max|\\ell)\s*\}|r_\\ell/g, function (mm, k) { alias.push(k === 'init' ? 'init' : k === 'max' ? 'max' : '\\ell'); return ' R '; });
    s = s.replace(/\{/g, '(').replace(/\}/g, ')');
    if (/[\\_|']/.test(s)) return null;
    return { expr: s, alias: alias };
  }
  function evalExpr(s, env) {
    var toks = [], i = 0, used = {};
    while (i < s.length) {
      var c = s[i];
      if (c === ' ') { i++; continue; }
      if (/\d/.test(c)) { var j = i; while (j < s.length && /[\d.]/.test(s[j])) j++; toks.push({ num: parseFloat(s.slice(i, j)) }); i = j; continue; }
      if (/[A-Za-z]/.test(c)) { if (!(c in env)) return null; toks.push({ id: c }); i++; continue; }
      if ('+-*/^()'.indexOf(c) >= 0) { toks.push({ op: c }); i++; continue; }
      return null;
    }
    var p = 0;
    function pk() { return toks[p]; }
    function expr() { var v = term(); while (pk() && (pk().op === '+' || pk().op === '-')) { var o = toks[p++].op, w = term(); v = o === '+' ? v + w : v - w; } return v; }
    function term() {
      var v = factor();
      for (;;) {
        var t = pk();
        if (!t) break;
        if (t.op === '*' || t.op === '/') { p++; var w = factor(); v = t.op === '*' ? v * w : v / w; }
        else if (t.num != null || t.id || t.op === '(') v = v * factor();
        else break;
      }
      return v;
    }
    function factor() { var b = unary(); if (pk() && pk().op === '^') { p++; return Math.pow(b, factor()); } return b; }
    function unary() { var t = pk(); if (t && t.op === '-') { p++; return -unary(); } if (t && t.op === '+') { p++; return unary(); } return prim(); }
    function prim() {
      var t = toks[p++];
      if (!t) throw 0;
      if (t.num != null) return t.num;
      if (t.id) { used[t.id] = 1; return env[t.id]; }
      if (t.op === '(') { var v = expr(), q = toks[p++]; if (!q || q.op !== ')') throw 0; return v; }
      throw 0;
    }
    try { if (!toks.length) return null; var v = expr(); if (p !== toks.length) return null; return { v: v, used: used }; } catch (e) { return null; }
  }
  function parenBal(t) { return (t.match(/\(/g) || []).length - (t.match(/\)/g) || []).length; }
  function stripSp(t) { return t.replace(/^(?:\s|\\[,;:! ]|~)+/, ''); }
  /* after the qualifier: swallow a parenthetical (it may span several \text groups); a trailing
     "+ \text{...}" becomes a stated extra term; anything else means the count continues -> not evaluated */
  function tailRest(qual, rest) {
    var bal = Math.max(0, parenBal(qual)), more = false, r = stripSp(rest), guard = 0;
    while (r && guard++ < 600 && (bal > 0 || r[0] === '(')) {
      more = true;
      var m = /^\\text(?:rm|it|tt|bf|sf)?\s*\{/.exec(r);
      if (m) { var b1 = matchBrace(r, m[0].length - 1); if (b1 < 0) return null; bal += parenBal(r.slice(m[0].length, b1)); r = r.slice(b1 + 1); }
      else { if (r[0] === '(') bal++; else if (r[0] === ')') bal--; r = r.slice(r[0] === '\\' ? 2 : 1); }
      if (bal <= 0) { bal = 0; r = stripSp(r); }
    }
    r = stripSp(r).replace(/^[.\s]+$/, '');
    if (!r) return { plus: '', more: more };
    var pm = /^\+\s*(?:\\[,;:! ]\s*)*\\text\s*\{([^{}]*)\}\s*$/.exec(r);
    if (pm) return { plus: pm[1].trim(), more: true };
    return null;
  }
  var ENV = { I: D_IN, O: D_OUT, d: D_IN, r: RANK, R: RANK };
  function budget(tp) {
    if (!tp) return { ok: false, why: 'none' };
    var cl = firstClause(tp), sp = splitText(cl), te = texToExpr(sp.pre);
    if (!te || !sp.pre.trim()) return { ok: false, why: 'params' };
    if (/[-+*/^]\s*$/.test(te.expr)) return { ok: false, why: 'params' };
    var tail = tailRest(sp.qual, sp.rest);
    if (!tail) return { ok: false, why: 'params' };
    var r = evalExpr(te.expr, ENV);
    if (!r || !isFinite(r.v) || r.v < 0 || Math.abs(r.v - Math.round(r.v)) > 1e-9) return { ok: false, why: 'params' };
    if (/whole model|in total/i.test(sp.qual)) return { ok: false, why: 'global' };
    var q = sp.qual.trim().replace(/^\(/, '').replace(/[\s;,(]+$/, '');
    q = q.split(/\s\(|;/)[0].replace(/\)$/, '').trim();
    if (r.v === 0) q = '';                       // "0 (one scalar λ …)": the qualifier is cut mid-phrase; the catalogued line says it
    return {
      ok: true, value: Math.round(r.v), tex: sp.pre.replace(/(?:\\[,;:! ]|\s)+$/, '').trim(), qual: q, plus: tail.plus,
      used: r.used, alias: te.alias, more: tail.more || tp.trim() !== cl.trim() || q !== sp.qual.trim()
    };
  }
  /* the budget a figure shows: backward lenses without a merge step have forward map id_Theta, so Q_M = Theta */
  function budgetOf(m) {
    var c = m.coords || {};
    if (c.kind === 'B' && c.merge === 'na') return { ok: true, value: THETA, tex: '\\lvert\\Theta\\rvert=d_{out}d_{in}', qual: 'forward map is the identity of Θ', plus: '', used: { I: 1, O: 1 }, alias: [], more: true, theta: true };
    return budget(m.trainable_params);
  }

  /* ======================= labels ======================= */
  var AX = [
    { key: 'kind', short: 'Kind' }, { key: 'shape', short: 'Shape' }, { key: 'base_point', short: 'Base point' },
    { key: 'gauge', short: 'Gauge' }, { key: 'covariance', short: 'Covariance' }, { key: 'merge', short: 'Merge' }, { key: 'rebasing', short: 'Rebasing' }
  ];
  var SHOW = {
    kind: { E0: 'E°', F_T: 'F_T' },
    covariance: { GLxGL: 'GL × GL', OxO: 'CO × CO', GLxCO: 'GL × CO', MonxGL: 'Mon × GL', MonxMon: 'Mon × Mon', Pi: 'Π', frame: 'frame' },
    gauge: { GLxTorus: 'GL × torus', nonlinear: 'non-linear' },
    merge: { 'M1->': 'M1→', Minf: 'M∞', na: 'n/a' },
    rebasing: { span: '→ span', group: '→ group' }
  };
  function pretty(t) {
    return String(t == null ? '' : t).replace(/\btheta0'/g, 'θ₀′').replace(/\btheta0\b/g, 'θ₀').replace(/\bTheta\b/g, 'Θ').replace(/\btheta\b/g, 'θ')
      .replace(/\brho\b/g, 'ρ').replace(/\bq0\b/g, 'q₀').replace(/\bS1\b/g, 'S₁').replace(/->/g, '→').replace(/\bf~/g, 'f̃').replace(/\bTheta'/g, 'Θ′');
  }
  function prettyHTML(t, esc) {
    return esc(pretty(t)).replace(/\brho_([A-Za-z])\b/g, 'ρ<sub>$1</sub>').replace(/(<\/sub>) o /g, '$1 ∘ ');
  }
  /* Unicode subscript digits -> <sub> outside TeX, code and tags (the body serif draws U+2080 like a small "o") */
  function subs(html) {
    return String(html).split(/(\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|<code>[\s\S]*?<\/code>|<[^>]+>)/).map(function (seg, i) {
      return i % 2 ? seg : seg.replace(/[₀-₉]+/g, function (m) { return '<sub>' + m.replace(/[₀-₉]/g, function (ch) { return String(ch.charCodeAt(0) - 8320); }) + '</sub>'; });
    }).join('');
  }
  /* catalogue notes: escaped TeX, with markdown code spans rendered as code */
  function noteHTML(t, esc) {
    return esc(t).replace(/`([^`]+)`/g, function (m0, code) { return '<code>' + code.replace(/\\(&quot;|")/g, '$1') + '</code>'; });
  }
  /* split a catalogue formula at top-level \quad / \qquad into stacked display lines */
  function splitQuad(t) {
    var out = [], d = 0, last = 0, i, m;
    for (i = 0; i < t.length; i++) {
      var c = t[i];
      if (c === '{') d++; else if (c === '}') d--;
      else if (d === 0 && (c === ',' || c === ';') && /^[,;]\\ \s*\\ /.test(t.slice(i))) {
        out.push(t.slice(last, i)); m = /^[,;]\\ \s*\\ /.exec(t.slice(i)); last = i + m[0].length; i = last - 1;
      }
      else if (c === '\\') {
        m = d === 0 && /^\\q?quad(?![A-Za-z])/.exec(t.slice(i));
        if (m) { out.push(t.slice(last, i)); last = i + m[0].length; i = last - 1; } else i++;
      }
    }
    out.push(t.slice(last));
    return out.map(function (x) { return x.replace(/^(?:\s|\\[,;:! ]|~)+|(?:[\s,;]|\\[,;:! ]|~)+$/g, ''); }).filter(function (x) { return x.length; });
  }
  /* \text{...} is verbatim in MathJax without textmacros: unescape \_ \& \% \# there */
  function texFix(t) {
    return String(t == null ? '' : t).replace(/\\text(?:rm|tt|it|bf|sf)?\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g, function (m0, c) { return m0.slice(0, m0.length - c.length - 1) + c.replace(/\\([_&%#])/g, '$1') + '}'; });
  }
  function showVal(axis, v) { if (v == null || v === '') return '—'; var m = SHOW[axis]; return (m && m[v]) || v; }
  var ROWS = [
    ['R-additive', 'R · additive reparametrisations'], ['R-action', 'R · action type'], ['R-Hadamard', 'R · Hadamard'],
    ['R-selective', 'R · selective'], ['B', 'B · backward lenses and schedules'], ['P', 'P · post-hoc operations'],
    ['F_T', 'F_T · families over a base'], ['E', 'E · pointed extensions'], ['E0', 'E° · unpointed extensions']
  ];
  var FEATURED = ['lora', 'dora', 'pissa', 'qlora', 'lora-xs', 'vera', 'hra', 'oft', 'ia3', 'bitfit', 'galore', 'task-arithmetic', 'houlsby-adapter', 'prefix-tuning'];
  var SHAPE_NAME = { Lin: 'linear', Cone: 'cone, vertex θ₀', 'Cone*': 'translated cone', Orb: 'group orbit', Sat: 'torus saturation', Meet: 'cone ∩ orbit', Fun: 'function-level' };
  var KIND_TXT = {
    R: 'A pointed map \\(\\rho_M:(Q_M,q_0)\\to(\\Theta,\\theta_0)\\), so an object of \\(\\mathsf{Meth}(\\Theta,\\theta_0)\\). The frozen model is initial, full fine-tuning is terminal, and the arrow into Full FT is \\(\\rho_M\\) itself (Obs I.4).',
    R_defect: 'A map \\(\\rho_M:Q_M\\to\\Theta\\) into the existing weights whose initial value lands at \\(\\theta_0+e\\). It is pointed at \\(\\theta_0+e\\), so it is an object of \\(\\mathsf{Meth}(\\Theta,\\theta_0+e)\\) rather than of \\(\\mathsf{Meth}(\\Theta,\\theta_0)\\). Merging is still \\(\\rho_M\\).',
    R_unpointed: 'A map \\(\\rho_M:Q_M\\to\\Theta\\) into the existing weights, but no parameter value lands on \\(\\theta_0\\), so it is not an object of \\(\\mathsf{Meth}(\\Theta,\\theta_0)\\). Merging is still \\(\\rho_M\\).',
    E: 'A pointed extension: it first enlarges the network to \\(\\tilde f\\) on \\(\\Theta\\times\\Theta\'\\), with a neutral point \\(\\theta_0\'\\) that reproduces \\(f\\). Its \\(\\rho\\) lands in the larger space, so it is not an object of \\(\\mathsf{Meth}(\\Theta,\\theta_0)\\); returning to \\(\\Theta\\) is a lifting problem (Exposé VI).',
    E0: 'An unpointed extension: no parameter value reproduces the frozen function, so not even \\(\\mathrm{const}_{q_0}\\) lands on the frozen model. It stands outside the interval (Exposé VI).',
    B: 'A backward lens or schedule: the forward map is \\(\\mathrm{id}_\\Theta\\), so as an object it is Full FT itself; what changes is how gradients are compressed or when the base is re-pointed (Exposé IV).',
    B_stage: 'A schedule of merge-and-restart stages: each stage is an object of the slice, merged into \\(\\Theta\\) by its \\(\\rho\\) before the next one starts (Exposé V).',
    P: 'A post-hoc operation \\(\\Theta^N\\to\\Theta\\) on already-trained points. With those points frozen, its coefficients form a map into \\(\\Theta\\), pointed when some coefficient returns \\(\\theta_0\\).',
    F_T: 'A family over an auxiliary base \\(T\\): a \\(T\\)-point \\(a:T\\to Q\\) gives \\(\\rho\\circ a:T\\to\\Theta\\). When \\(T\\) is a set of tasks or tenants, each fibre is an ordinary object; when \\(T\\) is the input, nothing fixed fuses into the weights.'
  };
  var BP_TXT = {
    regular: '\\(\\rho_M(q_0)=\\theta_0\\), and \\(\\theta_0\\) is a regular point of the image, with no first-order deficit.',
    apex: '\\(\\rho_M(q_0)=\\theta_0\\) at an apex, where the first-order image \\(S_1\\) is smaller than the image (Def I.5).',
    neutral: 'The neutral point \\(q_0\\) reproduces the frozen function in the extended network.',
    defect: 'A pointing defect, \\(\\rho_M(q_0)=\\theta_0+e\\neq\\theta_0\\). The route through \\(M\\) ends at \\(\\theta_0+e\\) and the direct route at \\(\\theta_0\\), so the triangle commutes only up to \\(e\\).',
    unpointed: 'No initial value realises the frozen model, so there is no pointed arrow from Frozen.',
    none: 'It has no trainable coordinates or base point of its own.'
  };
  var MG_TXT = {
    M1: 'Merge type M1: it fuses into its own box, and merging is \\(\\rho_M\\).',
    'M1->': 'Merge type M1→: it fuses into a neighbouring linear box, under side conditions.',
    Mq: 'Merge type Mq: \\(\\rho_M\\) exists, but an exact merge leaves the quantised type, and a merge that stays on the grid is lossy.',
    Minf: 'Merge type M∞: the edit depends non-affinely on the input or acts only at selected positions, so it does not fold into the existing weights box by box (Exposé VI).',
    na: 'Merge n/a: the update is written into \\(\\Theta\\) directly.'
  };

  /* what the data say about the three arrows of the triangle */
  function status(m) {
    var c = m.coords || {}, k = c.kind, bp = c.base_point, mg = c.merge;
    var ext = k === 'E' || k === 'E0';
    var cst = (bp === 'regular' || bp === 'apex' || bp === 'neutral') ? 'ok' : bp === 'defect' ? 'cond' : bp === 'unpointed' ? 'no' : 'na';
    var rho;
    if (mg === 'Minf') rho = 'no';
    else if (mg === 'M1' || mg === 'M1->') rho = ext ? 'cond' : 'ok';
    else if (mg === 'Mq') rho = 'cond';
    else if (mg === 'na') rho = k === 'B' ? 'ok' : 'na';
    else rho = 'na';
    var tone, head;
    if (cst === 'no' || rho === 'no') {
      tone = 'no';
      head = cst === 'no' ? (ext ? 'Outside the interval: no neutral point' : 'Outside the interval: no q₀ with ρ(q₀) = θ₀') : ext ? 'Outside the interval: it extends the network' : 'Outside the interval: no merge back into Θ';
    } else if (cst === 'na') { tone = 'na'; head = 'No base point of its own'; }
    else if (cst === 'cond') { tone = 'cond'; head = 'Commutes only up to the defect e'; }
    else if (rho === 'cond') { tone = 'cond'; head = ext ? 'Commutes after merging the extension (M1)' : 'Commutes; the exact merge leaves the quantised type'; }
    else {
      tone = 'ok';
      head = (k === 'B' && mg === 'na') ? 'Commutes, with M = Full FT as an object' : k === 'F_T' ? 'Commutes on every task fibre' : 'The triangle commutes';
    }
    var idRho = k === 'B' && mg === 'na';
    return { cst: cst, rho: rho, tone: tone, head: head, ext: ext, idRho: idRho, kind: k, bp: bp, mg: mg };
  }

  /* ======================= scoped CSS ======================= */
  function injectCSS() {
    if (document.getElementById('css-interval')) return;
    var F = '[data-figure="interval"] ';
    var css = [
      '[data-figure="interval"]{position:relative}',
      F + '.iv-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.iv-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.iv-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.1rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.iv-title i{color:var(--tide)}',
      F + '.iv-h3{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.25rem,1rem + 1.2cqi,1.6rem);line-height:1.1;margin:0;color:var(--ink)}',
      F + '.iv-h3 i{color:var(--tide)}',
      F + '.iv-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.iv-instr{margin:.45rem 0 .85rem;font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:72ch}',
      F + '.iv-instr b{color:var(--ink);font-weight:600}',
      F + '.iv-pick{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem .8rem;margin:0 0 .6rem}',
      F + '.iv-pick label{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.iv-sel{font-family:var(--f-ui);font-size:var(--fs-sm);padding:.3rem .45rem;background:var(--paper);color:var(--ink);border:1px solid var(--rule);border-radius:4px;max-width:100%;min-width:0}',
      F + '.iv-ctl{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin-left:auto}',
      F + '.iv-mini{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.02em;line-height:1.1;padding:.36rem .6rem;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);border-radius:4px;cursor:pointer;white-space:nowrap}',
      F + '.iv-mini:hover{border-color:var(--tide);color:var(--tide)}',
      /* quick picks wrap on wide stages; on narrow ones they form one scrolling row that fades out at the right */
      F + '.iv-chips{display:flex;gap:.35rem;flex-wrap:wrap;padding:.1rem;margin:0 0 .6rem}',
      F + '.iv-chip{flex:none;display:inline-flex;align-items:center;gap:.38rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.01em;padding:.22rem .62rem .22rem .46rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap}',
      F + '.iv-chip .sw{width:.62rem;height:.62rem;border-radius:50%;flex:none}',
      F + '.iv-chip:hover{color:var(--ink);border-color:var(--ink-3)}',
      F + '.iv-chip[aria-pressed="true"]{color:var(--ink);border-color:currentColor;box-shadow:inset 0 0 0 1px var(--kc,var(--tide));background:var(--paper-2)}',
      F + '.iv-dia{position:relative;width:100%;min-width:0}',
      F + '.iv-dia > svg{display:block;width:100%;height:auto;overflow:visible}',
      F + '.iv-dia svg text{font-family:var(--f-ui);font-size:12px}',
      F + '.iv-dia svg .num{font-family:var(--f-mono);font-variant-numeric:tabular-nums}',
      F + '.iv-dia svg .ui{font-weight:500}',
      F + '.iv-dia svg .disp{font-family:var(--f-display);font-weight:var(--w-head)}',
      F + '.iv-dia svg .ser{font-family:var(--f-body)}',
      F + '.iv-dia svg .it{font-style:italic}',
      F + '.iv-dia svg .halo{paint-order:stroke;stroke:var(--paper);stroke-width:3px;stroke-linejoin:round}',
      F + '.iv-dia svg [tabindex]{cursor:pointer;outline:none}',
      F + '.iv-dia svg [tabindex]:focus-visible .iv-focus{stroke:var(--ochre);stroke-width:2}',
      F + '.iv-p1{display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem;margin-bottom:.9rem}',
      F + '.iv-info{display:grid;grid-template-columns:minmax(0,1fr);gap:.75rem;align-items:start}',
      F + '.iv-banner{display:grid;gap:.25rem;padding:.6rem .75rem .65rem;border-radius:var(--radius);border:1px solid currentColor;min-width:0}',
      F + '.iv-banner .bt{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.22rem;line-height:1.15}',
      F + '.iv-banner .bs{font-size:.86rem;line-height:1.5;color:var(--ink-2)}',
      F + '.iv-banner.ok{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.iv-banner.cond{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + '.iv-banner.no{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.iv-banner.na{color:var(--ink-2);background:var(--paper)}',
      F + '.iv-box{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.6rem .75rem .65rem;min-width:0}',
      F + '.iv-lab{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);display:flex;flex-wrap:wrap;gap:.2rem .7rem;align-items:baseline}',
      F + '.iv-lab .nt{text-transform:none;letter-spacing:.01em}',
      F + '.iv-fx{overflow-x:auto;overflow-y:hidden;font-size:.92rem;padding:.25rem 0 .1rem;color:var(--ink)}',
      F + '.iv-fx mjx-container{margin:.15em 0 !important}',
      F + '.iv-fxl mjx-container[display="true"]{text-align:left !important;margin:.1em 0 .2em !important}',
      F + '.iv-fx mjx-container svg,' + F + '.iv-sub mjx-container svg{max-width:none}',
      F + '.iv-sub mjx-container{white-space:nowrap}',
      F + '.iv-cnote code,' + F + '.iv-dn code{font-size:.9em;background:var(--paper-2)}',
      F + 'sub{font-size:.72em;line-height:0;font-style:normal;font-variant-numeric:lining-nums}',
      F + '.iv-bud{display:flex;flex-wrap:wrap;align-items:baseline;gap:.15rem .7rem;margin-top:.15rem}',
      F + '.iv-bud .big{font-family:var(--f-mono);font-size:1.32rem;color:var(--ink);font-variant-numeric:tabular-nums;letter-spacing:-.01em}',
      F + '.sym{font-family:var(--f-body);letter-spacing:0}',
      F + '.iv-bud .q{font-size:.84rem;color:var(--ink-2)}',
      F + '.iv-bud .pc{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:var(--fs-xs);color:var(--ink-2)}',
      F + '.iv-sub{font-size:.82rem;color:var(--ink-2);line-height:1.45;overflow-x:auto;overflow-y:hidden}',
      F + '.iv-sub .k{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin-right:.4rem}',
      F + '.iv-tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.4rem;margin:0 0 .5rem}',
      F + '.iv-tile{display:grid;gap:.08rem;text-align:left;font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.4rem .55rem .45rem;cursor:pointer;min-width:0}',
      F + '.iv-tile:hover{border-color:var(--ink-3)}',
      F + '.iv-tile[aria-pressed="true"]{border-color:var(--tide);box-shadow:inset 0 -2px 0 var(--tide)}',
      F + '.iv-tile .ax{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.iv-tile .v{font-family:var(--f-display);font-size:1.16rem;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      F + '.iv-tile .n{font-size:var(--fs-xs);line-height:1.3;color:var(--ink-2);overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow-wrap:break-word}',
      F + '.iv-tile.ok .v{color:var(--moss-ink)}', F + '.iv-tile.cond .v{color:var(--ochre-ink)}', F + '.iv-tile.no .v{color:var(--seal-ink)}',
      F + '.iv-cnote{font-size:.86rem;line-height:1.55;color:var(--ink-2);padding:.5rem .7rem;border-left:2px solid var(--tide);background:var(--paper);border-radius:0 var(--radius) var(--radius) 0;margin:0 0 1rem;overflow-x:auto;overflow-y:hidden}',
      F + '.iv-cnote b{color:var(--ink);font-weight:600}',
      F + '.iv-cnote .def{display:block;color:var(--ink-2);font-size:.8rem;margin-top:.2rem}',
      F + '.iv-hr{border:0;border-top:1px solid var(--rule);margin:1.1rem 0 1rem}',
      F + '.iv-p2{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem}',
      F + '.iv-p2 > .iv-det{order:-1}',
      F + '.iv-lists{display:grid;gap:.85rem;align-content:start;min-width:0}',
      F + '.iv-lh{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:0 0 .3rem;display:flex;justify-content:space-between;gap:.5rem}',
      F + '.iv-list{display:grid;gap:.3rem;margin:0;padding:0;list-style:none}',
      F + '.iv-row{display:grid;grid-template-columns:auto auto minmax(0,1fr);column-gap:.45rem;row-gap:.1rem;align-items:baseline;width:100%;text-align:left;font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.4rem .55rem .45rem;cursor:pointer;min-width:0}',
      F + '.iv-row:hover{border-color:var(--ink-3)}',
      F + '.iv-row[aria-pressed="true"]{border-color:var(--tide);box-shadow:inset 3px 0 0 var(--tide)}',
      F + '.iv-row.ob[aria-pressed="true"]{border-color:var(--seal);box-shadow:inset 3px 0 0 var(--seal)}',
      F + '.iv-badge{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;font-variant-numeric:tabular-nums;padding:.02rem .34rem;border-radius:3px;border:1px solid var(--tide);color:var(--tide);line-height:1.3}',
      F + '.iv-badge.s{border-color:var(--seal);color:var(--seal-ink)}',
      F + '.iv-arr{font-family:var(--f-body);color:var(--ink-2);font-size:.95rem}',
      F + '.iv-arr.s{color:var(--seal)}',
      F + '.iv-tn{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--f-display);font-size:1.02rem;line-height:1.2;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
      F + '.iv-tn .sw{width:.58rem;height:.58rem;border-radius:50%;flex:none}',
      F + '.iv-rn{grid-column:1 / -1;font-size:.8rem;line-height:1.42;color:var(--ink-2);overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}',
      F + '.iv-empty{font-size:.84rem;color:var(--ink-2);font-style:italic;margin:0}',
      F + '.iv-inc{display:flex;flex-wrap:wrap;gap:.3rem}',
      F + '.iv-inc .iv-chip{font-size:.75rem;padding:.16rem .52rem .16rem .4rem}',
      F + '.iv-det{display:grid;gap:.55rem;align-content:start;min-width:0}',
      F + '.iv-dh{display:flex;flex-wrap:wrap;align-items:baseline;gap:.2rem .6rem}',
      F + '.iv-dh .t{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.25rem;line-height:1.15;color:var(--ink)}',
      F + '.iv-dh .t .x{color:var(--seal)}',
      F + '.iv-dh .r{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.02em;color:var(--ink-2)}',
      F + '.iv-dn{font-size:.84rem;line-height:1.55;color:var(--ink-2);overflow-x:auto;overflow-y:hidden}',
      F + '.iv-acts{display:flex;flex-wrap:wrap;gap:.4rem}',
      F + '.iv-legend{display:flex;flex-wrap:wrap;gap:.25rem .9rem;margin:1rem 0 0;padding-top:.7rem;border-top:1px solid var(--rule);font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);color:var(--ink-2);align-items:center}',
      F + '.iv-legend .lt{color:var(--ink-2);letter-spacing:.06em;text-transform:uppercase}',
      F + '.iv-legend span.li{display:inline-flex;align-items:center;gap:.3rem;white-space:nowrap}',
      F + '.iv-legend .sw{width:.58rem;height:.58rem;border-radius:50%}',
      F + '.iv-legend .ln{display:inline-block;width:1.3rem;height:0;border-top:1.5px solid currentColor;vertical-align:middle}',
      F + '.iv-legend .ln.d{border-top-style:dashed}',
      F + '.iv-foot{margin-top:.7rem;font-family:var(--f-body);font-size:.84rem;line-height:1.55;color:var(--ink-2)}',
      F + '.iv-card{position:absolute;z-index:30;width:min(21rem,calc(100% - 4px));background:var(--paper);color:var(--ink);border:1px solid var(--rule);border-radius:var(--radius);box-shadow:var(--shadow);padding:.6rem .75rem .65rem;font-size:.82rem;line-height:1.45;pointer-events:none}',
      F + '.iv-card .ch{display:flex;align-items:baseline;gap:.45rem;flex-wrap:wrap}',
      F + '.iv-card .ch .sw{width:.62rem;height:.62rem;border-radius:50%;align-self:center}',
      F + '.iv-card .cn{font-family:var(--f-display);font-weight:var(--w-head);font-size:1.22rem;line-height:1.1}',
      F + '.iv-card .cy{font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--ink-2)}',
      F + '.iv-card .cf{font-style:italic;color:var(--ink-2)}',
      F + '.iv-card .ck{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;color:var(--ink-2);text-transform:uppercase;margin-top:.1rem}',
      F + '.iv-card .ci{margin:.35rem 0 .4rem;color:var(--ink-2)}',
      F + '.iv-card .cc{display:grid;grid-template-columns:auto minmax(0,1fr) auto minmax(0,1fr);gap:.12rem .5rem;font-family:var(--f-ui);font-size:.75rem;align-items:baseline}',
      F + '.iv-card .cc .a{font-weight:500;color:var(--ink-2)}',
      F + '.iv-card .cc .b{color:var(--ink)}',
      F + '.iv-card .cb{margin-top:.4rem;font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:.75rem;color:var(--ink-2)}',
      F + '.iv-card .hint{margin-top:.3rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;color:var(--tide);letter-spacing:.02em}',
      '@container (min-width: 500px){' + F + '.iv-tiles{grid-template-columns:repeat(4,minmax(0,1fr))}' + '}',
      /* seven tiles in one row only where each keeps a readable width; the wide figure (about 850px inside) takes 4 + 3 */
      '@container (min-width: 980px){' + F + '.iv-tiles{grid-template-columns:repeat(7,minmax(0,1fr))}' + '}',
      '@container (min-width: 760px){' +
        F + '.iv-info{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr)}' +
        F + '.iv-p2{grid-template-columns:minmax(0,.92fr) minmax(0,1.08fr);gap:1.4rem}' +
        F + '.iv-p2 > .iv-det{order:0}' +
      '}',
      '@container (max-width: 520px){' + F + '.iv-ctl{margin-left:0}' + F + '.iv-sel{flex:1 1 12rem}' + '}',
      '@container (max-width: 460px){' + F + '.iv-chips{flex-wrap:nowrap;overflow-x:auto;padding:.1rem 1.6rem .45rem .1rem;scrollbar-width:thin;' +
        '-webkit-mask-image:linear-gradient(to right,#000 calc(100% - 2.2rem),transparent);mask-image:linear-gradient(to right,#000 calc(100% - 2.2rem),transparent)}' + '}'
    ].join('\n');
    var s = document.createElement('style');
    s.id = 'css-interval';
    s.textContent = css;
    document.head.appendChild(s);
  }

  /* ======================= figure ======================= */
  Atlas.register('interval', function (el, A) {
    injectCSS();
    var h = A.h, D = A.data(), methods = D.methods || [], edges = D.edges || [];
    var uid = 'iv' + Math.random().toString(36).slice(2, 7);
    var byId = {};
    methods.forEach(function (m) { byId[m.id] = m; });
    var axisVal = {};
    (D.axes || []).forEach(function (a) { var o = {}; (a.values || []).forEach(function (v) { o[v.key] = v; }); axisVal[a.key] = { name: a.name, values: o }; });
    var ruleBy = {};
    (D.arrow_rules || []).forEach(function (r) { ruleBy[r.key] = r; });
    var outE = {}, inE = {};
    edges.forEach(function (e) { (outE[e.source] = outE[e.source] || []).push(e); (inE[e.target] = inE[e.target] || []).push(e); });
    var core = methods.filter(function (m) { return m.core; });

    var S0 = { src: byId.lora ? 'lora' : (core[0] && core[0].id), tile: 'base_point', arrow: null, showAllIn: false, paused: false };
    var reduced = A.reducedMotion();

    /* ---------- helpers ---------- */
    /* text styling goes inline: the site stylesheet's ".figure svg text {fill; font-size}" outranks attributes */
    var STY = { 'font-size': 1, fill: 1, 'font-style': 1, 'letter-spacing': 1, 'font-weight': 1, 'font-family': 1 };
    function S(tag, attrs, parent) {
      var e = document.createElementNS(NS, tag), st = '', tx = tag === 'text' || tag === 'tspan';
      if (attrs) for (var k in attrs) if (attrs[k] != null) {
        if (tx && STY[k]) st += k + ':' + attrs[k] + (k === 'font-size' && typeof attrs[k] === 'number' ? 'px' : '') + ';';
        else e.setAttribute(k, attrs[k]);
      }
      if (st) e.setAttribute('style', st);
      if (parent) parent.appendChild(e);
      return e;
    }
    function T(parent, x, y, txt, attrs) {
      /* Unicode subscript digits are set as lowered tspans: the UI and mono faces have no glyphs for them */
      if (/[₀-₉]/.test(txt)) {
        var parts = [];
        String(txt).split(/([₀-₉]+)/).forEach(function (p, i) { if (p) parts.push(i % 2 ? [p.replace(/[₀-₉]/g, function (ch) { return String(ch.charCodeAt(0) - 8320); }), 'sub'] : [p]); });
        return RT(parent, x, y, parts, attrs);
      }
      var a = attrs || {}; a.x = x; a.y = y; var t = S('text', a, parent); t.textContent = txt; return t;
    }
    /* text with subscripts: parts = [[str, 'sub'|'it'|undefined], ...] */
    function RT(parent, x, y, parts, attrs) {
      var t = S('text', Object.assign({ x: x, y: y }, attrs || {}), parent), down = false;
      parts.forEach(function (p) {
        var sub = p[1] && p[1].indexOf('sub') >= 0, it = p[1] && p[1].indexOf('it') >= 0, ser = p[1] && p[1].indexOf('ser') >= 0;
        var a = {};
        if (ser) a['font-family'] = 'var(--f-body)';     // a Greek symbol inside a mono readout (Plex Mono's Θ reads as 0)
        if (sub && !down) { a.dy = '0.32em'; down = true; } else if (!sub && down) { a.dy = '-0.32em'; down = false; if (!/^\s/.test(p[0])) a.dx = '0.1em'; }   // a hair after a subscript keeps the next glyph's halo off it
        if (sub) a['font-size'] = '0.78em';
        if (it) a['font-style'] = 'italic';
        var ts = S('tspan', a, t);
        ts.textContent = p[0];
      });
      return t;
    }
    function fit(t, maxW, minPx) {
      try {
        var len = t.getComputedTextLength();
        if (!len || len <= maxW) return;
        var fs = parseFloat(t.style.fontSize) || 14, nf = Math.max(minPx || 10, fs * maxW / len);
        t.style.fontSize = nf.toFixed(1) + 'px';
        if (t.getComputedTextLength() > maxW) {
          var s = t.textContent;
          while (s.length > 3 && t.getComputedTextLength() > maxW) { s = s.slice(0, -1); t.textContent = s.replace(/\s+$/, '') + '…'; }
        }
      } catch (e) {}
    }
    /* shift a label horizontally so its box stays within [x0, x1] (labels near the figure's edges) */
    function keepIn(t, x0, x1) {
      try {
        var b = t.getBBox(), dx = b.x < x0 ? x0 - b.x : b.x + b.width > x1 ? x1 - b.x - b.width : 0;
        if (dx) t.setAttribute('x', (parseFloat(t.getAttribute('x')) + dx).toFixed(1));
      } catch (e) {}
      return t;
    }
    function fmt(n) { return Math.round(n).toLocaleString('en-US'); }
    function C(tok) { return A.css(tok); }
    function toneCol(t) { return t === 'ok' ? C('--ink-2') : t === 'cond' ? C('--ochre') : t === 'no' ? C('--seal') : C('--ink-3'); }
    function toneInk(t) { return t === 'cond' ? C('--ochre-ink') : t === 'no' ? C('--seal-ink') : C('--ink-2'); }   // label text in an arrow's tone
    function semCol(t) { return t === 'ok' ? C('--moss') : t === 'cond' ? C('--ochre') : t === 'no' ? C('--seal') : C('--ink-3'); }
    /* the same tones for text (the -ink variants keep small labels at AA contrast) */
    function semInk(t) { return t === 'ok' ? C('--moss-ink') : t === 'cond' ? C('--ochre-ink') : t === 'no' ? C('--seal-ink') : C('--ink-2'); }
    function cur() { return byId[S0.src]; }
    /* a pointing defect that moves the base itself (quantised or truncated base), read off the catalogue's notes */
    function vshiftOf(m) {
      var n = (m && m.coord_notes) || {};
      return !!m && (m.coords || {}).base_point === 'defect' && /kappa|κ|translated|W_world|quantis/i.test((n.shape || '') + ' ' + (n.base_point || ''));
    }
    function nameOf(id) { return id === '__full' ? 'Full FT' : (byId[id] ? byId[id].name : id); }
    function kcol(id) { return id === '__full' ? C('--ink-2') : byId[id] ? A.kindColor(byId[id].modification_kind) : C('--ink-3'); }
    var tq = Promise.resolve();
    function typeset(node) { tq = tq.then(function () { return A.typeset(node); }).catch(function () {}); return tq; }
    function setTeX(node, html) {
      var MJ = window.MathJax;
      if (MJ && MJ.typesetClear) { try { MJ.typesetClear([node]); } catch (e) {} }
      node.innerHTML = subs(html);
      if (/\\\(|\\\[/.test(html)) typeset(node).then(function () { fitMath(node); });
    }
    /* a formula wider than its box is scaled down (to 70% at most) before it falls back to scrolling */
    function fitMath(node) {
      if (!node || !node.querySelectorAll) return;
      node.querySelectorAll('.iv-fx mjx-container, .iv-sub mjx-container').forEach(function (mc) {
        var svg = mc.querySelector('svg'), box = mc.closest('.iv-fx, .iv-sub');
        if (!svg || !box) return;
        mc.style.fontSize = '';
        var room = box.clientWidth - 2, pct = 100;
        for (var k = 0; k < 4 && room > 0; k++) {
          var need = svg.getBoundingClientRect().width;
          if (need <= room || pct <= 70) break;
          pct = Math.max(70, Math.floor(pct * room / need) - 1);
          mc.style.fontSize = pct + '%';
        }
      });
    }
    function dispatch(id) {
      if (!id || id === '__full' || !byId[id]) return;
      selfFire = true;
      try { document.dispatchEvent(new CustomEvent('atlas:select-method', { detail: { id: id } })); } finally { selfFire = false; }
    }
    var selfFire = false;

    /* ---------- DOM skeleton ---------- */
    var stage = h('div', { class: 'stage iv-stage' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'iv-head' }, [
      h('h3', { class: 'iv-title', html: 'Between <i>nothing</i> and <i>everything</i>' }),
      h('span', { class: 'iv-kicker', text: 'Exposé I · Def I.2 · Obs I.4 · VIII.C' })
    ]));
    stage.appendChild(h('p', { class: 'iv-instr', html: 'Pick a method \\(M\\). When \\(M\\) starts at \\(\\theta_0\\), the frozen model maps into it by \\(\\mathrm{const}_{q_0}\\). \\(M\\) maps into full fine-tuning by its own \\(\\rho_M\\), and that arrow <b>is the merge</b>. Both routes from the frozen model end at \\(\\theta_0\\), so the triangle commutes. The moving dot carries a trained \\(q\\) along \\(\\rho_M\\) to the weight \\(\\rho_M(q)\\) it merges into.' }));

    var selId = uid + '-sel';
    var sel = h('select', { id: selId, class: 'iv-sel', 'aria-label': 'Method M' });
    var pauseBtn = h('button', { type: 'button', class: 'iv-mini', text: 'Pause motion' });
    stage.appendChild(h('div', { class: 'iv-pick' }, [h('label', { for: selId, text: 'Method M' }), sel, h('div', { class: 'iv-ctl' }, [pauseBtn])]));
    var chipsWrap = h('div', { class: 'iv-chips scroll-x', role: 'group', 'aria-label': 'Quick picks: featured methods' });
    stage.appendChild(chipsWrap);

    var p1 = h('div', { class: 'iv-p1' });
    stage.appendChild(p1);
    var dia1 = h('div', { class: 'iv-dia' });
    p1.appendChild(dia1);
    var info = h('div', { class: 'iv-info' });
    p1.appendChild(info);
    var infoL = h('div', { style: 'display:grid;gap:.6rem;min-width:0' }), infoR = h('div', { style: 'display:grid;gap:.6rem;min-width:0' });
    info.appendChild(infoL); info.appendChild(infoR);
    var fxBox = h('div', { class: 'iv-box' });
    var budBox = h('div', { class: 'iv-box' });
    infoL.appendChild(fxBox); infoL.appendChild(budBox);
    var banner = h('div', { class: 'iv-banner', role: 'status' });
    infoR.appendChild(banner);

    var tiles = h('div', { class: 'iv-tiles', role: 'group', 'aria-label': 'The seven coordinates of M' });
    stage.appendChild(tiles);
    var cnote = h('div', { class: 'iv-cnote', 'aria-live': 'polite' });
    stage.appendChild(cnote);

    stage.appendChild(h('hr', { class: 'iv-hr' }));
    stage.appendChild(h('div', { class: 'iv-head' }, [
      h('h4', { class: 'iv-h3', html: 'Arrows <i>between</i> methods' }),
      h('span', { class: 'iv-kicker', text: 'Def I.2 · rules A1–A5 · tests O1–O3' })
    ]));
    stage.appendChild(h('p', { class: 'iv-instr', html: 'An arrow \\(h:M\\to N\\) is a pointed smooth map with \\(\\rho_N\\circ h=\\rho_M\\). Then <b>\\(N\\) simulates \\(M\\)</b>, since \\(h\\) carries every parameter path of \\(M\\) to one of \\(N\\) with the same weights. Pick an arrow to draw its triangle over \\(\\Theta\\). Obstructions, in red, are pairs with no such \\(h\\), shown with the test that fails.' }));
    var sel2Id = uid + '-sel2';
    var sel2 = h('select', { id: sel2Id, class: 'iv-sel', 'aria-label': 'Source method M for the arrows' });
    stage.appendChild(h('div', { class: 'iv-pick' }, [h('label', { for: sel2Id, text: 'Source M' }), sel2]));
    var p2 = h('div', { class: 'iv-p2' });
    stage.appendChild(p2);
    var lists = h('div', { class: 'iv-lists' });
    var right = h('div', { class: 'iv-det' });
    p2.appendChild(lists); p2.appendChild(right);
    var dia2 = h('div', { class: 'iv-dia' });
    right.appendChild(dia2);
    var det = h('div', { class: 'iv-det' });
    right.appendChild(det);

    var legend = h('div', { class: 'iv-legend' });
    stage.appendChild(legend);
    stage.appendChild(h('p', { class: 'iv-foot', html: 'Setup: one linear layer with \\(d_{in}=d_{out}=d=4096\\) and rank \\(r=16\\), so \\(\\lvert\\Theta\\rvert=4096^2=16{,}777{,}216\\). Each \\(\\lvert Q_M\\rvert\\) is evaluated from the catalogued trainable-parameter formula; a formula that needs any other hyper-parameter is shown, not evaluated. The sketches inside the boxes are schematic: \\(\\Theta\\) shows the shape class of \\(\\mathrm{Im}\\,\\rho_M\\) (Exposé VIII.A), and \\(Q_M\\) shows the scalar slice \\((b,a)\\) with fibres \\(\\{ba=c\\}\\) when the gauge is multiplicative. Arrows, rules and obstructions are the atlas data (Exposé VIII.C).' }));

    var card = h('div', { class: 'iv-card', role: 'tooltip', id: uid + '-card' });
    card.hidden = true;
    el.appendChild(card);
    var live = h('div', { class: 'sr-only', 'aria-live': 'polite' });
    el.appendChild(live);

    /* ---------- selector ---------- */
    function buildSelect() { fillSelect(sel); fillSelect(sel2); }
    function fillSelect(sel) {
      sel.innerHTML = '';
      var m0 = cur();
      if (m0 && !m0.core) {
        var g0 = h('optgroup', { label: 'Current (outside the core list)' });
        g0.appendChild(h('option', { value: m0.id, text: m0.name }));
        sel.appendChild(g0);
      }
      ROWS.forEach(function (rw) {
        var ms = core.filter(function (m) { return m.table_row === rw[0]; }).sort(function (a, b) { return a.name.localeCompare(b.name); });
        if (!ms.length) return;
        var g = h('optgroup', { label: rw[1] + ' (' + ms.length + ')' });
        ms.forEach(function (m) { g.appendChild(h('option', { value: m.id, text: m.name })); });
        sel.appendChild(g);
      });
      var rest = core.filter(function (m) { return !ROWS.some(function (rw) { return rw[0] === m.table_row; }); });
      if (rest.length) {
        var g2 = h('optgroup', { label: 'Other' });
        rest.forEach(function (m) { g2.appendChild(h('option', { value: m.id, text: m.name })); });
        sel.appendChild(g2);
      }
      sel.value = S0.src;
    }
    sel.addEventListener('change', function () { setSource(sel.value, true); });
    sel2.addEventListener('change', function () { setSource(sel2.value, true); });

    var chipEls = [];
    function buildChips() {
      chipsWrap.innerHTML = '';
      chipEls = [];
      FEATURED.filter(function (id) { return byId[id]; }).forEach(function (id) {
        var m = byId[id], col = A.kindColor(m.modification_kind);
        var b = h('button', { type: 'button', class: 'iv-chip', 'aria-pressed': String(id === S0.src), 'data-mid': id, style: '--kc:' + col });
        b.appendChild(h('span', { class: 'sw', style: 'background:' + col }));
        b.appendChild(document.createTextNode(m.name));
        b.addEventListener('click', function () { setSource(id, true); });
        bindCard(b, id);
        chipsWrap.appendChild(b);
        chipEls.push(b);
      });
    }
    function syncChips() { chipEls.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-mid') === S0.src)); }); }

    pauseBtn.addEventListener('click', function () {
      S0.paused = !S0.paused;
      pauseBtn.textContent = S0.paused ? 'Play motion' : 'Pause motion';
      schedule();
    });
    if (reduced) { pauseBtn.hidden = true; }

    /* ---------- hover card ---------- */
    var cardFor = null;
    function cardHTML(id) {
      if (id === '__full') {
        return '<div class="ch"><span class="cn">Full FT</span><span class="cy">terminal object</span></div>' +
          '<div class="cf">Full fine-tuning: \\((\\Theta,\\theta_0,\\mathrm{id})\\)</div>' +
          '<p class="ci">Every weight-space method maps into it by its own \\(\\rho\\); that arrow is the merge.</p>' +
          '<div class="cb">|<span class="sym">Θ</span>| = ' + fmt(THETA) + ' at 4096 × 4096</div>';
      }
      var m = byId[id];
      if (!m) return '';
      var c = m.coords || {}, col = A.kindColor(m.modification_kind), b = budgetOf(m);
      var cc = AX.map(function (a) { return '<span class="a">' + A.esc(a.short) + '</span><span class="b">' + A.esc(showVal(a.key, c[a.key])) + '</span>'; }).join('');
      return '<div class="ch"><span class="sw" style="background:' + col + '"></span><span class="cn">' + A.esc(m.name) + '</span><span class="cy">' + A.esc([m.year, m.venue].filter(Boolean).join(' · ')) + '</span></div>' +
        (m.full_name && m.full_name !== m.name ? '<div class="cf">' + A.esc(m.full_name) + '</div>' : '') +
        '<div class="ck">' + A.esc(A.kindName(m.modification_kind)) + (m.hf_peft ? ' · HF PEFT' : '') + (m.core ? '' : ' · extended catalogue') + '</div>' +
        (m.key_idea ? '<p class="ci">' + A.esc(m.key_idea) + '</p>' : '') +
        '<div class="cc">' + cc + '</div>' +
        '<div class="cb">|Q| at 4096², r = 16: ' + (b.ok ? fmt(b.value) + (b.qual ? ' ' + A.esc(b.qual) : '') : 'not evaluated') + '</div>' +
        '<div class="hint">Click: open in the catalogue</div>';
    }
    function showCard(node, id) {
      if (cardFor !== id) { card.innerHTML = subs(cardHTML(id)); cardFor = id; if (id === '__full') typeset(card); }
      card.hidden = false;
      var r = el.getBoundingClientRect(), t = node.getBoundingClientRect();
      var cw = card.offsetWidth, ch = card.offsetHeight;
      var x = t.left - r.left + t.width / 2 - cw / 2;
      x = Math.max(2, Math.min(r.width - cw - 2, x));
      var y = t.bottom - r.top + 8;
      if (t.bottom + 8 + ch > window.innerHeight && t.top - ch - 8 > 0) y = t.top - r.top - ch - 8;
      card.style.left = x + 'px';
      card.style.top = y + 'px';
    }
    function hideCard() { clearTimeout(cardT); card.hidden = true; }
    var cardT = 0;
    function bindCard(node, id) {
      node.addEventListener('pointerenter', function (e) {
        if (e.pointerType !== 'mouse') return;
        clearTimeout(cardT);
        if (!card.hidden) showCard(node, id); else cardT = setTimeout(function () { showCard(node, id); }, 180);
      });
      node.addEventListener('pointerleave', function () { clearTimeout(cardT); hideCard(); });
      node.addEventListener('focus', function () { var fv = false; try { fv = node.matches(':focus-visible'); } catch (e) {} if (fv) showCard(node, id); });
      node.addEventListener('blur', hideCard);
    }
    el.addEventListener('keydown', function (e) { if (e.key === 'Escape') hideCard(); });

    /* ======================= sketches ======================= */
    /* image germ of rho inside Theta, centred at theta0 = (ox, oy), half-size s; returns the trained path t -> point */
    function drawImage(g, shape, ox, oy, s, col, o) {
      o = o || {};
      var fa = o.fa != null ? o.fa : 0.15, sw = o.sw || 1.3, faint = o.faint;
      function P(u, v) { return [ox + u * s, oy + v * s]; }
      function d(pts, close) { return 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L') + (close ? 'Z' : ''); }
      function ell(cx, cy, rx, ry, attrs) { return S('ellipse', Object.assign({ cx: ox + cx * s, cy: oy + cy * s, rx: Math.abs(rx * s), ry: Math.abs(ry * s) }, attrs), g); }
      var fill = { fill: col, 'fill-opacity': faint ? fa * 0.55 : fa, stroke: col, 'stroke-opacity': faint ? 0.45 : 0.85, 'stroke-width': sw, 'stroke-linejoin': 'round' };
      var line = { fill: 'none', stroke: col, 'stroke-opacity': faint ? 0.4 : 0.55, 'stroke-width': 0.8 };
      var K = 0.55, E = 0.3;
      function cone(ax, ay, hu, hd, wide) {
        var k = wide || K;
        S('path', Object.assign({ d: d([P(ax - k * hu, ay - hu), P(ax, ay), P(ax + k * hu, ay - hu)], false) + ' A' + (k * hu * s).toFixed(1) + ',' + (E * k * hu * s).toFixed(1) + ' 0 0 0 ' + P(ax - k * hu, ay - hu).map(function (v) { return v.toFixed(1); }).join(',') }, fill), g);
        ell(ax, ay - hu, k * hu, E * k * hu, { fill: 'none', stroke: col, 'stroke-opacity': faint ? 0.45 : 0.85, 'stroke-width': sw });
        if (hd > 0) {
          S('path', Object.assign({ d: d([P(ax - k * hd, ay + hd), P(ax, ay), P(ax + k * hd, ay + hd)], false) + ' A' + (k * hd * s).toFixed(1) + ',' + (E * k * hd * s).toFixed(1) + ' 0 0 1 ' + P(ax - k * hd, ay + hd).map(function (v) { return v.toFixed(1); }).join(',') }, Object.assign({}, fill, { 'fill-opacity': (faint ? fa * 0.55 : fa) * 0.7 })), g);
          ell(ax, ay + hd, k * hd, E * k * hd, { fill: 'none', stroke: col, 'stroke-opacity': faint ? 0.35 : 0.6, 'stroke-width': sw * 0.9 });
        }
        if (!faint) for (var i = 1; i < 5; i++) {
          var ph = -Math.PI / 2 + i * Math.PI / 5;
          S('line', Object.assign({ x1: P(ax, ay)[0], y1: P(ax, ay)[1], x2: P(ax + k * hu * Math.cos(ph), ay - hu + E * k * hu * Math.sin(ph))[0], y2: P(ax + k * hu * Math.cos(ph), ay - hu + E * k * hu * Math.sin(ph))[1] }, line), g);
        }
      }
      function onCone(ax, ay, k, hh, ph) { return P(ax + k * hh * Math.sin(ph), ay - hh + E * k * hh * Math.cos(ph)); }
      switch (shape) {
        case 'Lin': {
          S('path', Object.assign({ d: d([P(-0.95, 0.3), P(0.42, 0.62), P(0.95, -0.3), P(-0.42, -0.62)], true) }, fill), g);
          S('line', Object.assign({ x1: P(-0.685, -0.16)[0], y1: P(-0.685, -0.16)[1], x2: P(0.685, 0.16)[0], y2: P(0.685, 0.16)[1] }, line), g);
          S('line', Object.assign({ x1: P(-0.25, 0.46)[0], y1: P(-0.25, 0.46)[1], x2: P(0.25, -0.46)[0], y2: P(0.25, -0.46)[1] }, line), g);
          return { at: function (t) { return P(0.6 * t, -0.3 * t + 0.12 * Math.sin(Math.PI * t)); }, start: P(0, 0) };
        }
        case 'Cone': {
          cone(0, 0, 0.92, 0.62);
          return { at: function (t) { return onCone(0, 0, K, 0.8 * t, -0.9 + 1.5 * t); }, start: P(0, 0) };
        }
        case 'Cone*': {
          var h0 = 0.5, ph0 = 0.55;
          var ax = -K * h0 * Math.sin(ph0), ay = h0 - E * K * h0 * Math.cos(ph0);
          cone(ax, ay, 1.38, 0.34);
          S('circle', { cx: P(ax, ay)[0], cy: P(ax, ay)[1], r: 1.8, fill: col, 'fill-opacity': 0.8 }, g);
          return { at: function (t) { return onCone(ax, ay, K, h0 + 0.38 * t, ph0 + 1.0 * t); }, start: P(0, 0), apex: P(ax, ay) };
        }
        case 'Orb': {
          ell(0, -0.42, 0.8, 0.42, { fill: col, 'fill-opacity': faint ? 0.03 : 0.05, stroke: col, 'stroke-opacity': faint ? 0.45 : 0.9, 'stroke-width': sw * 1.25 });
          ell(0, -0.42, 0.8, 0.42, { fill: 'none', stroke: col, 'stroke-opacity': 0.25, 'stroke-width': 5 });
          return { at: function (t) { var a = 1.7 * t; return P(0.8 * Math.sin(a), -0.42 + 0.42 * Math.cos(a)); }, start: P(0, 0) };
        }
        case 'Sat': {
          cone(0, 0, 0.92, 0.6, 0.95);
          cone(0, 0, 0.92, 0, K);
          return { at: function (t) { return onCone(0, 0, 0.8, 0.8 * t, -1.2 + 2.0 * t); }, start: P(0, 0) };
        }
        case 'Meet': {
          var save = faint; faint = true; cone(0, 0, 0.92, 0.5); faint = save;
          var rot = -58 * Math.PI / 180, rx = 0.62, ry = 0.46;
          var mp = function (a) { var u = rx * Math.sin(a), v = -ry + ry * Math.cos(a); return P(u * Math.cos(rot) - v * Math.sin(rot), u * Math.sin(rot) + v * Math.cos(rot)); };
          var pts = []; for (var j = 0; j <= 48; j++) pts.push(mp(j / 48 * 2 * Math.PI));
          S('path', { d: d(pts, true), fill: 'none', stroke: col, 'stroke-opacity': 0.4, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, g);
          var arc = []; for (j = 0; j <= 24; j++) arc.push(mp(-0.15 + j / 24 * 1.25));
          S('path', { d: d(arc, false), fill: 'none', stroke: col, 'stroke-opacity': faint ? 0.5 : 0.95, 'stroke-width': sw * 2, 'stroke-linecap': 'round' }, g);
          return { at: function (t) { return mp(1.05 * t); }, start: P(0, 0) };
        }
        case 'Fun': {
          var bl = [];
          for (var q = 0; q <= 40; q++) { var a = q / 40 * 2 * Math.PI, rr = 0.62 + 0.09 * Math.sin(3 * a) + 0.05 * Math.cos(5 * a); bl.push(P(0.25 + rr * Math.cos(a) * 1.05, -0.12 + rr * Math.sin(a) * 0.72)); }
          S('path', { d: d(bl, true), fill: col, 'fill-opacity': 0.06, stroke: col, 'stroke-opacity': 0.7, 'stroke-width': 1.1, 'stroke-dasharray': '4 3' }, g);
          return { at: null, start: P(0, 0), fun: P(0.25, -0.12) };
        }
        case 'Θ':
          return { at: function (t) { return P(0.62 * t, -0.42 * t + 0.1 * Math.sin(Math.PI * t)); }, start: P(0, 0) };
        default:
          return { at: null, start: P(0, 0) };
      }
    }
    /* parameter space Q, centred at (cx, cy), half sizes (hw, hh); returns q0 and the trained path */
    function drawQ(g, cx, cy, hw, hh, m, col) {
      var c = (m && m.coords) || {}, gauge = c.gauge, bp = c.base_point;
      var mult = gauge === 'GL' || gauge === 'scalar' || gauge === 'torus' || gauge === 'GLxTorus';
      var sx = hw / 1.2, sy = hh / 1.0;
      function P(b, a) { return [cx + b * sx, cy - a * sy]; }
      var faint = { fill: 'none', stroke: C('--ink-3'), 'stroke-opacity': 0.35, 'stroke-width': 0.8 };
      var q0, path;
      if (mult) {
        /* fibres of the scalar slice (b, a) -> ba: hyperbolas, and the singular fibre {ba = 0} (the axes) */
        S('line', Object.assign({ x1: P(-1.2, 0)[0], y1: P(-1.2, 0)[1], x2: P(1.2, 0)[0], y2: P(1.2, 0)[1], 'stroke-dasharray': '2 3' }, faint, { 'stroke-opacity': 0.6 }), g);
        S('line', Object.assign({ x1: P(0, -1)[0], y1: P(0, -1)[1], x2: P(0, 1)[0], y2: P(0, 1)[1], 'stroke-dasharray': '2 3' }, faint, { 'stroke-opacity': 0.6 }), g);
        [0.12, 0.3, 0.55].forEach(function (cc) {
          [1, -1].forEach(function (sg) {
            [1, -1].forEach(function (sb) {
              var pts = [];
              for (var i = 0; i <= 30; i++) { var b = cc / 1.0 + (1.2 - cc) * i / 30; var a = sg * cc / b; if (Math.abs(a) <= 1) pts.push(P(sb * b, sb * a)); }
              if (pts.length > 1) S('path', Object.assign({ d: 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L') }, faint), g);
            });
          });
        });
        /* q0 on the singular fibre {ba = 0} exactly when rho(q0) is the vertex of a cone-shaped image */
        if (bp === 'apex' || bp === 'neutral' || !bp || (bp === 'defect' && c.shape === 'Cone' && vshiftOf(m))) { q0 = [0, 0.55]; path = function (t) { return [0.85 * t, 0.55 + 0.18 * Math.sin(Math.PI * t * 0.8)]; }; }
        else { q0 = [-0.55, 0.55]; path = function (t) { return [-0.55 + 1.35 * t, 0.55 - 0.25 * Math.sin(Math.PI * t * 0.9)]; }; }
      } else if (gauge === 'nonlinear') {
        for (var i = 0; i < 5; i++) {
          var pts = [];
          for (var j = 0; j <= 30; j++) { var b = -1.2 + 2.4 * j / 30; pts.push(P(b, -0.8 + i * 0.4 + 0.14 * Math.sin(2.6 * b + i))); }
          S('path', Object.assign({ d: 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L') }, faint), g);
        }
        q0 = [-0.6, 0.0]; path = function (t) { return [-0.6 + 1.4 * t, 0.42 * Math.sin(Math.PI * t * 0.9)]; };
      } else {
        for (var k = -2; k <= 2; k++) {
          S('line', Object.assign({ x1: P(k * 0.5, -1)[0], y1: P(k * 0.5, -1)[1], x2: P(k * 0.5, 1)[0], y2: P(k * 0.5, 1)[1] }, faint), g);
          if (Math.abs(k) < 2) S('line', Object.assign({ x1: P(-1.2, k * 0.5)[0], y1: P(-1.2, k * 0.5)[1], x2: P(1.2, k * 0.5)[0], y2: P(1.2, k * 0.5)[1] }, faint), g);
        }
        q0 = [-0.5, -0.25]; path = function (t) { return [-0.5 + 1.3 * t, -0.25 + 0.62 * t - 0.1 * Math.sin(Math.PI * t)]; };
      }
      return { q0: P(q0[0], q0[1]), at: function (t) { var v = path(t); return P(v[0], v[1]); } };
    }

    /* straight arrow between two points with an open head; tone sets colour; cut draws the seal ⊘ */
    function arrow(g, p, q, o) {
      o = o || {};
      var col = o.col || toneCol(o.tone), dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
      var dash = o.tone === 'no' || o.tone === 'na' || o.dash ? '5 4' : null;
      S('line', { x1: p[0].toFixed(1), y1: p[1].toFixed(1), x2: q[0].toFixed(1), y2: q[1].toFixed(1), stroke: col, 'stroke-width': o.w || 1.3, 'stroke-dasharray': dash, 'stroke-linecap': 'round' }, g);
      var a = Math.atan2(dy, dx), hs = o.hs || 8.5;
      S('polyline', { points: [q[0] - hs * Math.cos(a - 0.42), q[1] - hs * Math.sin(a - 0.42)].map(function (v) { return v.toFixed(1); }).join(',') + ' ' + q[0].toFixed(1) + ',' + q[1].toFixed(1) + ' ' + [q[0] - hs * Math.cos(a + 0.42), q[1] - hs * Math.sin(a + 0.42)].map(function (v) { return v.toFixed(1); }).join(','), fill: 'none', stroke: col, 'stroke-width': o.w || 1.3, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, g);
      if (o.hook) {
        var nx = -uy * (o.hookSide || 1), ny = ux * (o.hookSide || 1), r = 4;
        S('path', { d: 'M' + (p[0] + nx * 2 * r).toFixed(1) + ',' + (p[1] + ny * 2 * r).toFixed(1) + ' C' + (p[0] + nx * 2 * r - ux * 1.4 * r).toFixed(1) + ',' + (p[1] + ny * 2 * r - uy * 1.4 * r).toFixed(1) + ' ' + (p[0] - ux * 1.4 * r).toFixed(1) + ',' + (p[1] - uy * 1.4 * r).toFixed(1) + ' ' + p[0].toFixed(1) + ',' + p[1].toFixed(1), fill: 'none', stroke: col, 'stroke-width': o.w || 1.3, 'stroke-linecap': 'round' }, g);
      }
      var cut = null;
      if (o.cut != null) {
        var cx = p[0] + dx * o.cut, cy = p[1] + dy * o.cut;
        S('circle', { cx: cx, cy: cy, r: 8, fill: C('--paper-2'), stroke: C('--seal'), 'stroke-width': 1.4 }, g);
        S('line', { x1: cx - 5.4, y1: cy + 5.4, x2: cx + 5.4, y2: cy - 5.4, stroke: C('--seal'), 'stroke-width': 1.4, 'stroke-linecap': 'round' }, g);
        cut = [cx, cy];
      }
      return { p: p, q: q, len: len, ux: ux, uy: uy, cut: cut };
    }
    /* point where the ray from c towards t leaves the rectangle (c, hw, hh), plus a gap */
    function edgePt(c, hw, hh, t, gap) {
      var dx = t[0] - c[0], dy = t[1] - c[1];
      var s = Math.min(dx ? hw / Math.abs(dx) : Infinity, dy ? hh / Math.abs(dy) : Infinity);
      var len = Math.hypot(dx, dy) || 1;
      return [c[0] + dx * s + dx / len * gap, c[1] + dy * s + dy / len * gap];
    }
    /* label placed at the midpoint of a segment, on the side away from a reference point */
    function sideLabel(a, ref, off) {
      var mx = (a.p[0] + a.q[0]) / 2, my = (a.p[1] + a.q[1]) / 2, nx = -a.uy, ny = a.ux;
      if ((mx - ref[0]) * nx + (my - ref[1]) * ny < 0) { nx = -nx; ny = -ny; }
      var x = mx + nx * off, y = my + ny * off;
      var anchor = nx > 0.35 ? 'start' : nx < -0.35 ? 'end' : 'middle';
      return { x: x, y: y + (ny > 0.35 ? 9 : ny < -0.35 ? -2 : 4), anchor: anchor, nx: nx, ny: ny };
    }
    function boxRect(g, b, attrs) {
      return S('rect', Object.assign({ x: (b.cx - b.w / 2).toFixed(1), y: (b.cy - b.h / 2).toFixed(1), width: b.w.toFixed(1), height: b.h.toFixed(1), rx: 5, fill: C('--paper'), stroke: C('--rule'), 'stroke-width': 1 }, attrs || {}), g);
    }
    function clipFor(svg, b, id) {
      var defs = svg.querySelector('defs') || S('defs', null, svg);
      var cp = S('clipPath', { id: id }, defs);
      S('rect', { x: b.cx - b.w / 2 + 1, y: b.cy - b.h / 2 + 1, width: b.w - 2, height: b.h - 2, rx: 4 }, cp);
      return 'url(#' + id + ')';
    }
    function dot(g, p, r, fill, stroke) { return S('circle', { cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: r, fill: fill, stroke: stroke || 'none', 'stroke-width': stroke ? 1.4 : null }, g); }
    function comet(g, col) {
      var gg = S('g', { opacity: 0, 'pointer-events': 'none' }, g), cs = [];
      for (var i = 4; i >= 0; i--) cs.push(S('circle', { r: i === 0 ? 4.2 : 3.2 - i * 0.45, fill: col, 'fill-opacity': i === 0 ? 1 : 0.5 - i * 0.09 }, gg));
      cs.reverse();
      return { g: gg, cs: cs };
    }
    /* polyline arc-length parametrisation */
    function poly(pts) {
      var L = [0];
      for (var i = 1; i < pts.length; i++) L.push(L[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      var tot = L[L.length - 1] || 1;
      return {
        len: tot,
        at: function (s) {
          var d = Math.max(0, Math.min(1, s)) * tot;
          for (var i = 1; i < pts.length; i++) if (d <= L[i] || i === pts.length - 1) {
            var seg = L[i] - L[i - 1] || 1, u = (d - L[i - 1]) / seg;
            return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * u, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * u];
          }
          return pts[pts.length - 1];
        },
        frac: function (k) { return L[k] / tot; }
      };
    }
    function placeComet(cm, path, s, alpha) {
      if (s == null || alpha <= 0) { cm.g.setAttribute('opacity', 0); return; }
      cm.g.setAttribute('opacity', alpha.toFixed(3));
      for (var i = 0; i < cm.cs.length; i++) {
        var pt = path.at(Math.max(0, s - i * 0.022));
        cm.cs[i].setAttribute('cx', pt[0].toFixed(1));
        cm.cs[i].setAttribute('cy', pt[1].toFixed(1));
      }
    }
    function ramp(p, a, b) { return p <= a ? 0 : p >= b ? 1 : (p - a) / (b - a); }
    function ease(x) { return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2; }
    function trailPath(g, fn, n, col, w) {
      var pts = [];
      for (var i = 0; i <= n; i++) pts.push(fn(i / n));
      var dd = 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L');
      var pe = S('path', { d: dd, fill: 'none', stroke: col, 'stroke-width': w || 1.6, 'stroke-linecap': 'round', 'stroke-dasharray': '2 3.2' }, g);
      return pe;
    }

    /* ======================= panel A: the interval triangle ======================= */
    var A1 = null;     // animation handles
    function lay1(W) {
      var c = W < 700, pad = c ? 4 : 14, H, baseY;
      var F = { w: c ? (W > 480 ? 46 : 40) : 58, h: c ? (W > 480 ? 46 : 40) : 58 };
      var M = { w: c ? Math.min(200, W * 0.45) : Math.min(236, W * 0.28), h: c ? Math.min(112, Math.round(0.28 * W)) : 124 };
      var T = { w: c ? Math.min(224, W * 0.46) : Math.min(270, W * 0.31), h: c ? Math.min(152, Math.round(0.38 * W)) : 158 };
      if (c) {
        M.cx = W * 0.52; M.cy = 40 + M.h / 2;
        baseY = Math.round(M.cy + M.h / 2 + 112 + T.h / 2);
        H = baseY + T.h / 2 + 116;
      } else {
        H = Math.round(Math.max(384, Math.min(444, W * 0.47 + 4)));
        baseY = H - 118;
      }
      F.cx = pad + F.w / 2 + (c ? 4 : Math.max(30, W * 0.05)); F.cy = baseY;
      T.cx = W - pad - T.w / 2 - (c ? 0 : Math.max(10, W * 0.03)); T.cy = baseY;
      if (!c) { M.cx = (F.cx + T.cx) / 2 - W * 0.015; M.cy = 108; }
      return { W: W, H: H, c: c, F: F, M: M, T: T };
    }
    function renderTri() {
      var m = cur();
      if (!m) return;
      var st = status(m), col = A.kindColor(m.modification_kind), bud = budgetOf(m);
      var W = Math.max(280, dia1.getBoundingClientRect().width || dia1.clientWidth || 640), L = lay1(W), c = L.c;   // exact width: the SVG draws 1:1, so a 12px label renders at 12px
      dia1.innerHTML = '';
      var ink = C('--ink'), ink2 = C('--ink-2'), ink3 = C('--ink-3'), moss = C('--moss'), seal = C('--seal'), ochre = C('--ochre');
      var sealI = C('--seal-ink'), ochreI = C('--ochre-ink');
      var desc = 'Commutative triangle. Frozen maps to ' + m.name + ' by const q0' + (st.cst === 'ok' ? '' : st.cst === 'cond' ? ' only up to a pointing defect' : st.cst === 'no' ? ', which does not exist' : ', not applicable') +
        '; ' + m.name + ' maps to Full FT by rho, the merge' + (st.rho === 'ok' ? '' : st.rho === 'cond' ? ', with a condition' : st.rho === 'no' ? ', which is obstructed' : ', not applicable') +
        (st.cst === 'ok' && st.rho !== 'no' && st.rho !== 'na' ? '; the composite is the inclusion of theta0. ' : st.cst === 'cond' ? '; the composite lands at theta0 + e, not at theta0. ' : '. ') + st.head + '.';
      var svg = S('svg', { width: W, height: L.H, viewBox: '0 0 ' + W + ' ' + L.H, role: 'img', 'aria-label': desc }, dia1);
      var gA = S('g', null, svg), gB = S('g', null, svg), gL = S('g', null, svg), gN = S('g', null, svg), gX = S('g', null, svg);
      var F = L.F, M = L.M, Tb = L.T;
      var cen = [(F.cx + M.cx + Tb.cx) / 3, (F.cy + M.cy + Tb.cy) / 3 + (c ? 4 : 10)];

      /* --- boxes --- */
      boxRect(gB, F);
      var star = [F.cx, F.cy];
      dot(gB, star, c ? 3.6 : 4.2, ink);
      T(gB, F.cx + (c ? 6 : 8), F.cy - (c ? 6 : 8), '∗', { class: 'ser', 'font-size': c ? 13 : 14, fill: ink2 });

      var mg = S('g', { tabindex: 0, role: 'button', 'aria-label': m.name + ': open in the catalogue', 'data-mid': m.id }, gB);
      boxRect(mg, M, { class: 'iv-focus', stroke: col, 'stroke-opacity': 0.75, 'stroke-width': 1.3 });
      var qg = S('g', { 'clip-path': clipFor(svg, M, uid + '-cq') }, mg);
      var Q = drawQ(qg, M.cx, M.cy, M.w / 2 - 6, M.h / 2 - 6, m, col);
      mg.addEventListener('click', function () { dispatch(m.id); });
      mg.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); dispatch(m.id); } });
      bindCard(mg, m.id);

      boxRect(gB, Tb);
      var tg = S('g', { 'clip-path': clipFor(svg, Tb, uid + '-ct') }, gB);
      /* where θ₀ and ρ(q₀) sit relative to the image (rq0 = ρ(q₀), th0 = θ₀):
         - pointed: both at the image's base point;
         - defect from a quantised or truncated base (QLoRA, LoftQ, PEQA, KaSA, …; read off the catalogue's notes): the
           image is anchored at ρ(q₀) = θ₀ + e (a cone has its vertex there) and θ₀ is drawn off it, the generic case
           (QLoRA reaches θ₀ only if rank e ≤ r, Prop V.2);
         - any other defect (uni-LoRA, LoRA-One, AdaMerging, …): the image still has its vertex or base at θ₀, and the
           start θ₀ + e is another point of it;
         - unpointed (ETHER, prefixes): no parameter value gives θ₀, so θ₀ is drawn off the image. */
      var shape = (m.coords || {}).shape, sImg = Math.min(Tb.w, Tb.h) * (shape === 'Sat' ? 0.4 : 0.44);
      var vshift = vshiftOf(m);
      /* a translated cone that still contains θ₀ at a smooth point (KaSA: e = −[W₀]_tail has rank ≤ r), read off the notes */
      var thOnCone = vshift && shape === 'Cone' && st.cst === 'cond' && /lies in the image/i.test(((m.coord_notes || {}).shape || '') + ' ' + ((m.coord_notes || {}).base_point || ''));
      var rq0 = [Tb.cx - Tb.w * 0.08, Tb.cy + Tb.h * 0.1], th0 = rq0;
      var offImg = (st.cst === 'cond' && vshift && !thOnCone) || st.cst === 'no', onImgDefect = st.cst === 'cond' && !vshift;
      if (offImg) { rq0 = [Tb.cx + Tb.w * 0.05, Tb.cy - Tb.h * 0.02]; th0 = [rq0[0] - 0.46 * sImg, rq0[1] + 0.56 * sImg]; }
      /* on the far side of the cone from the trained path: hh = 0.6, phase -1.2 in drawImage's cone coordinates */
      if (thOnCone) th0 = [rq0[0] + sImg * 0.55 * 0.6 * Math.sin(-1.2), rq0[1] + sImg * (-0.6 + 0.3 * 0.55 * 0.6 * Math.cos(-1.2))];
      var imgOK = st.rho !== 'no' && st.rho !== 'na' && !st.ext;
      if (st.idRho) S('rect', { x: Tb.cx - Tb.w / 2, y: Tb.cy - Tb.h / 2, width: Tb.w, height: Tb.h, fill: col, 'fill-opacity': 0.1 }, tg);
      var IMG = drawImage(tg, st.idRho ? 'Θ' : shape, rq0[0], rq0[1], sImg, col, { faint: !imgOK && shape !== 'Fun' });
      var imgAt = IMG.at;
      if (onImgDefect) {
        if (IMG.at) { rq0 = IMG.at(0.5); imgAt = function (t) { return IMG.at(0.5 + 0.5 * t); }; }
        else rq0 = IMG.fun ? IMG.fun : [th0[0] + 0.3 * sImg, th0[1] - 0.25 * sImg];
      }

      /* --- labels of objects --- */
      /* Frozen's labels sit under its box, clear of the arrow const_q0 */
      var fx0 = c ? F.cx - F.w / 2 : F.cx, fa = c ? 'start' : 'middle';
      T(gL, fx0, F.cy + F.h / 2 + (c ? 18 : 21), 'Frozen', { class: 'disp', 'font-size': c ? 16 : 19, fill: ink, 'text-anchor': fa });
      RT(gL, fx0, F.cy + F.h / 2 + (c ? 33 : 38), [['({∗}, ∗)', 'it']], { class: 'ser', 'font-size': c ? 13 : 14, fill: ink2, 'text-anchor': fa });
      T(gL, fx0, F.cy + F.h / 2 + (c ? 48 : 54), '|Q| = 0', { class: 'num', 'font-size': 12, fill: ink2, 'text-anchor': fa });

      var tM = T(gL, M.cx, M.cy - M.h / 2 - (c ? 8 : 10), m.name, { class: 'disp', 'font-size': c ? 17 : 22, fill: col, 'text-anchor': 'middle' });
      fit(tM, Math.max(M.w + (c ? 40 : 90), 120), 12);
      var bt = bud.ok ? '|Q| = ' + fmt(bud.value) : '|Q|: see formula', btCls = bud.ok ? 'num' : null;
      if (c) {
        /* left of the box: (Q_M, q0) over the budget; a budget wider than the margin breaks after its "=" or ":" */
        var lx = M.cx - M.w / 2 - 7;
        RT(gL, lx, M.cy - 6, [['(Q', 'it'], ['M', 'sub it'], [', q', 'it'], ['0', 'sub'], [')', 'it']], { class: 'ser', 'font-size': 13, fill: ink2, 'text-anchor': 'end' });
        var bt1 = T(gL, lx, M.cy + 12, bt, { class: btCls, 'font-size': 12, fill: ink2, 'text-anchor': 'end' });
        var btLen = 0;
        try { btLen = bt1.getComputedTextLength(); } catch (e) {}
        var cut = bt.search(/[=:] /);
        if (btLen > lx - 4 && cut > 0) {
          bt1.textContent = bt.slice(0, cut + 1);
          fit(T(gL, lx, M.cy + 27, bt.slice(cut + 2), { class: btCls, 'font-size': 12, fill: ink2, 'text-anchor': 'end' }), lx - 4, 11);
        } else fit(bt1, lx - 4, 11);
      } else {
        RT(gL, M.cx, M.cy + M.h / 2 + 19, [['(Q', 'it'], ['M', 'sub it'], [', q', 'it'], ['0', 'sub'], [')', 'it']], { class: 'ser', 'font-size': 14, fill: ink2, 'text-anchor': 'middle' });
        T(gL, M.cx, M.cy + M.h / 2 + 35, bt, { class: btCls, 'font-size': 12, fill: ink2, 'text-anchor': 'middle' });
      }

      T(gL, Tb.cx + Tb.w / 2, Tb.cy - Tb.h / 2 - (c ? 8 : 10), 'Full FT', { class: 'disp', 'font-size': c ? 16 : 19, fill: ink, 'text-anchor': 'end' });
      RT(gL, Tb.cx, Tb.cy + Tb.h / 2 + (c ? 16 : 19), [['(Θ, θ', 'it'], ['0', 'sub'], [')', 'it']], { class: 'ser', 'font-size': c ? 13 : 14, fill: ink2, 'text-anchor': 'middle' });
      keepIn(RT(gL, Tb.cx, Tb.cy + Tb.h / 2 + (c ? 32 : 35), [['|'], ['Θ', 'ser'], ['| = ' + fmt(THETA)]], { class: 'num', 'font-size': 12, fill: ink2, 'text-anchor': 'middle' }), 2, W - 2);
      var shapeTxt = st.idRho ? 'Im ρ = Θ' : st.ext ? 'Im ρ: in the extended net' : shape ? 'Im ρ: ' + (shape === 'Cone' && st.cst === 'cond' && vshift ? 'cone, vertex θ₀ + e' : SHAPE_NAME[shape] || shape) : '';
      if (shapeTxt && c) keepIn(T(gL, Tb.cx, Tb.cy + Tb.h / 2 + 46, shapeTxt, { 'font-size': 12, fill: ink2, 'text-anchor': 'middle' }), 2, W - 2);
      else if (shapeTxt) {
        var sl = shapeTxt.split(': ');
        T(gL, Tb.cx + Tb.w / 2 - 7, Tb.cy + Tb.h / 2 - (sl[1] ? 23 : 8), sl[0], { 'font-size': 12, fill: ink2, 'text-anchor': 'end' });
        if (sl[1]) T(gL, Tb.cx + Tb.w / 2 - 7, Tb.cy + Tb.h / 2 - 8, sl[1], { 'font-size': 12, fill: ink2, 'text-anchor': 'end' });
      }

      /* --- arrows --- */
      var aC = arrow(gA, edgePt([F.cx, F.cy], F.w / 2, F.h / 2, [M.cx, M.cy], 6), edgePt([M.cx, M.cy], M.w / 2, M.h / 2, [F.cx, F.cy], 7), { tone: st.cst === 'ok' ? 'ok' : st.cst, cut: st.cst === 'no' ? 0.64 : null });
      var aR = arrow(gA, edgePt([M.cx, M.cy], M.w / 2, M.h / 2, [Tb.cx, Tb.cy], 6), edgePt([Tb.cx, Tb.cy], Tb.w / 2, Tb.h / 2, [M.cx, M.cy], 7), { tone: st.rho, cut: st.rho === 'no' ? 0.36 : null, w: st.rho === 'ok' ? 1.6 : 1.3, col: st.rho === 'ok' ? col : null });
      var aB = arrow(gA, edgePt([F.cx, F.cy], F.w / 2, F.h / 2, [Tb.cx, Tb.cy], 6), edgePt([Tb.cx, Tb.cy], Tb.w / 2, Tb.h / 2, [F.cx, F.cy], 7), { tone: 'ok' });

      var fsL = c ? 14 : 16;
      var lc = sideLabel(aC, cen, c ? 10 : 14);
      RT(gL, lc.x, lc.y, [['const', ''], ['q', 'sub it'], ['0', 'sub']], { class: 'ser', 'font-size': fsL, fill: toneInk(st.cst === 'ok' ? 'ok' : st.cst), 'text-anchor': lc.anchor });
      /* the note under an arrow label follows the arrow's slope, so it does not run into the arrow below the label */
      var under = function (lab, arr, dy) { var sh = Math.abs(arr.uy) > 0.2 ? Math.max(-30, Math.min(30, dy * arr.ux / arr.uy)) : 0; return (lab.anchor === 'end' && sh < 0) || (lab.anchor === 'start' && sh > 0) ? lab.x + sh : lab.x; };
      if (st.cst === 'cond') keepIn(T(gL, under(lc, aC, 15), lc.y + 15, 'lands at θ₀ + e', { 'font-size': 12, fill: ochreI, 'text-anchor': lc.anchor }), 2, W - 2);
      else if (st.cst === 'no') keepIn(T(gL, under(lc, aC, 15), lc.y + 15, st.ext ? 'no neutral point' : 'no q₀ lands on θ₀', { 'font-size': 12, fill: sealI, 'text-anchor': lc.anchor }), 2, W - 2);
      else if (st.cst === 'na') keepIn(T(gL, under(lc, aC, 15), lc.y + 15, 'no base point', { 'font-size': 12, fill: ink2, 'text-anchor': lc.anchor }), 2, W - 2);

      var lr = sideLabel(aR, cen, c ? 10 : 14);
      var rparts = st.idRho ? [['id', ''], ['Θ', 'sub']] : [['ρ', 'it'], ['M', 'sub it']];
      RT(gL, lr.x, lr.y, rparts, { class: 'ser', 'font-size': fsL + 2, fill: st.rho === 'ok' ? col : toneInk(st.rho), 'text-anchor': lr.anchor });
      var mtxt = st.rho === 'no' ? 'merge: M∞' : st.rho === 'cond' ? (st.mg === 'Mq' ? 'merge: Mq' : 'merge, once extended') : st.idRho ? 'Q = Θ' : 'merge';
      keepIn(T(gL, under(lr, aR, 18), lr.y + 18, mtxt, { class: 'ui', 'font-size': 12, fill: st.rho === 'no' ? sealI : st.rho === 'cond' ? ochreI : ink2, 'text-anchor': lr.anchor, 'letter-spacing': '0.02em' }), 2, W - 2);

      var lb = { x: (aB.p[0] + aB.q[0]) / 2, y: aB.p[1] + 20 };
      /* the composite equals the direct arrow only when both legs exist and land in Θ */
      var eqOK = st.cst === 'ok' && st.rho !== 'no' && st.rho !== 'na';
      var comp = eqOK ? [['θ', 'it'], ['0', 'sub'], ['  =  ρ', 'it'], ['M', 'sub it'], [' ∘ const', ''], ['q', 'sub it'], ['0', 'sub']]
        : st.cst === 'cond' ? [['ρ', 'it'], ['M', 'sub it'], [' ∘ const', ''], ['q', 'sub it'], ['0', 'sub'], ['  =  θ', 'it'], ['0', 'sub'], [' + e  ≠  θ', 'it'], ['0', 'sub']]
        : [['θ', 'it'], ['0', 'sub']];
      var compNote = eqOK ? 'inclusion of the base point' : st.cst === 'cond' ? 'the direct arrow is the inclusion of θ₀' :
        st.cst === 'no' ? 'inclusion of the base point; no arrow Frozen → M' : st.cst === 'na' ? 'inclusion of the base point; M has no base point' :
        'inclusion of the base point; the route through M leaves Θ';
      if (c) {
        /* the composite and its note sit under the triangle; a note wider than the figure breaks at its ";" */
        RT(gL, lb.x, aB.p[1] - 7, [['θ', 'it'], ['0', 'sub']], { class: 'ser', 'font-size': 14, fill: ink2, 'text-anchor': 'middle' });
        RT(gL, W / 2, L.H - 40, comp, { class: 'ser', 'font-size': 13, fill: st.cst === 'cond' ? ochreI : ink2, 'text-anchor': 'middle' });
        var cn1 = T(gL, W / 2, L.H - 22, compNote, { 'font-size': 12, fill: ink2, 'text-anchor': 'middle' }), cnLen = 0;
        try { cnLen = cn1.getComputedTextLength(); } catch (e) {}
        if (cnLen > W - 8 && compNote.indexOf('; ') > 0) {
          cn1.textContent = compNote.slice(0, compNote.indexOf('; ') + 1);
          fit(T(gL, W / 2, L.H - 7, compNote.slice(compNote.indexOf('; ') + 2), { 'font-size': 12, fill: ink2, 'text-anchor': 'middle' }), W - 8, 11);
        } else fit(cn1, W - 8, 11);
      } else {
        RT(gL, lb.x, lb.y, comp, { class: 'ser', 'font-size': 14, fill: st.cst === 'cond' ? ochreI : ink2, 'text-anchor': 'middle' });
        T(gL, lb.x, lb.y + 17, compNote, { 'font-size': 12, fill: ink2, 'text-anchor': 'middle' });
      }

      /* --- commutation mark --- */
      var mk = S('g', { transform: 'translate(' + cen[0].toFixed(1) + ',' + cen[1].toFixed(1) + ')' }, gN);
      var mcol = semCol(st.tone), mcolT = semInk(st.tone), rr = c ? 11 : 14;
      if (st.tone === 'no') {
        S('circle', { r: rr, fill: 'none', stroke: mcol, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, mk);
        S('line', { x1: -rr * 0.6, y1: rr * 0.6, x2: rr * 0.6, y2: -rr * 0.6, stroke: mcol, 'stroke-width': 1.4, 'stroke-linecap': 'round' }, mk);
      } else {
        var a0 = -Math.PI * 0.85, a1 = Math.PI * 0.55;
        S('path', { d: 'M' + (rr * Math.cos(a0)).toFixed(2) + ',' + (rr * Math.sin(a0)).toFixed(2) + ' A' + rr + ',' + rr + ' 0 1 1 ' + (rr * Math.cos(a1)).toFixed(2) + ',' + (rr * Math.sin(a1)).toFixed(2), fill: 'none', stroke: mcol, 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-dasharray': st.tone === 'cond' ? '3 2.5' : null }, mk);
        var ex = rr * Math.cos(a1), ey = rr * Math.sin(a1), ta = a1 + Math.PI / 2;
        S('polyline', { points: [ex - 5 * Math.cos(ta - 0.5), ey - 5 * Math.sin(ta - 0.5)].join(',') + ' ' + ex + ',' + ey + ' ' + [ex - 5 * Math.cos(ta + 0.5), ey - 5 * Math.sin(ta + 0.5)].join(','), fill: 'none', stroke: mcol, 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, mk);
      }
      var mword = st.tone === 'ok' ? 'commutes' : st.tone === 'cond' ? (st.cst === 'cond' ? 'commutes up to e' : 'commutes, with a condition') : st.tone === 'no' ? 'outside Meth(Θ, θ₀)' : 'no base point';
      /* narrow layouts set a long verdict on two lines, kept clear of the Full FT box */
      var mlines = c && mword.length > 12 ? [mword.slice(0, mword.indexOf(' ')), mword.slice(mword.indexOf(' ') + 1)] : [mword];
      mlines.forEach(function (ln, i) {
        var mt = T(mk, 0, rr + (c ? 15 : 17) + i * 14, ln, { class: 'ui', 'font-size': 12, fill: mcolT, 'text-anchor': 'middle', 'letter-spacing': '0.02em' });
        try {
          var mlen = mt.getComputedTextLength(), roomR = Tb.cx - Tb.w / 2 - 6 - cen[0], roomL = 4 - cen[0];
          if (mlen / 2 > roomR) mt.setAttribute('x', Math.max(roomL + mlen / 2, roomR - mlen / 2).toFixed(1));
        } catch (e) {}
      });
      var pulse = S('circle', { r: rr, fill: 'none', stroke: mcol, 'stroke-width': 1.2, opacity: 0 }, mk);

      /* --- points and the moving pieces --- */
      dot(gX, Q.q0, c ? 3.4 : 4, ink);
      RT(gX, Q.q0[0] - (c ? 5 : 6), Q.q0[1] - (c ? 5 : 6), [['q', 'it'], ['0', 'sub']], { class: 'ser halo', 'font-size': 13, fill: ink2, 'text-anchor': 'end' });
      trailPath(gX, Q.at, 40, col, 1.2).setAttribute('opacity', 0.3);
      if (imgAt && imgOK) trailPath(gX, imgAt, 40, col, 1.3).setAttribute('opacity', 0.3);
      var qTrail = trailPath(gX, Q.at, 40, col, 1.5);
      var tTrail = imgAt && imgOK ? trailPath(gX, imgAt, 40, col, 1.7) : null;
      dot(gX, th0, c ? 3.4 : 4, ink);
      RT(gX, th0[0] - (c ? 6 : 7), th0[1] + (c ? 14 : 15), [['θ', 'it'], ['0', 'sub']], { class: 'ser halo', 'font-size': 13, fill: ink2, 'text-anchor': 'end' });
      if (st.cst === 'cond') {
        arrow(gX, th0, [rq0[0] - (rq0[0] - th0[0]) * 0.12, rq0[1] - (rq0[1] - th0[1]) * 0.12], { col: ochre, w: 1.2, hs: 6 });
        dot(gX, rq0, c ? 3.2 : 3.8, C('--paper'), ochre);
        T(gX, (th0[0] + rq0[0]) / 2 + 7, (th0[1] + rq0[1]) / 2 + 9, 'e', { class: 'ser it halo', 'font-size': c ? 13 : 14, fill: ochreI });
        /* on the image the trained point moves off to the right, so this label goes left */
        RT(gX, rq0[0] + (onImgDefect ? -7 : 7), rq0[1] - 6, [['ρ', 'it'], ['M', 'sub it'], ['(q', 'it'], ['0', 'sub'], [')', 'it']], { class: 'ser halo', 'font-size': c ? 12.5 : 13, fill: ochreI, 'text-anchor': onImgDefect ? 'end' : 'start' });
      } else if (offImg && !st.ext) {
        dot(gX, rq0, c ? 3.2 : 3.8, C('--paper'), seal);
        RT(gX, rq0[0] + 7, rq0[1] - 6, [['ρ', 'it'], ['M', 'sub it'], ['(q', 'it'], ['0', 'sub'], [')', 'it']], { class: 'ser halo', 'font-size': c ? 12.5 : 13, fill: sealI });
      }
      if (IMG.apex && imgOK) T(gX, IMG.apex[0] - 5, IMG.apex[1] + 13, 'θ₀ − P', { class: 'ser halo', 'font-size': 12.5, fill: ink2, 'text-anchor': 'end' });
      var qDot = dot(gX, Q.at(1), c ? 3.6 : 4.4, col, C('--paper'));
      var tDot = tTrail ? dot(gX, imgAt(1), c ? 3.8 : 4.6, col, C('--paper')) : null;
      var qLab = RT(gX, 0, 0, [['q', 'it']], { class: 'ser halo', 'font-size': 13, fill: col });
      var tLab = tDot ? RT(gX, 0, 0, st.idRho ? [['q', 'it']] : [['ρ', 'it'], ['M', 'sub it'], ['(q)', 'it']], { class: 'ser halo', 'font-size': 13, fill: col }) : null;
      var ring = S('circle', { cx: th0[0], cy: th0[1], r: 4, fill: 'none', stroke: st.cst === 'cond' ? ochre : moss, 'stroke-width': 1.6, opacity: 0 }, gX);
      var ringM = tDot ? S('circle', { r: 4, fill: 'none', stroke: col, 'stroke-width': 1.6, opacity: 0 }, gX) : null;

      var cmA = comet(gX, ink), cmB = comet(gX, ink), cmC = comet(gX, col);
      /* comet routes */
      var routeC = poly([star, aC.p, aC.cut || aC.q].concat(aC.cut ? [] : [Q.q0]));
      var routeR0 = poly([Q.q0, aR.p, aR.cut || aR.q].concat(aR.cut ? [] : [rq0]));
      var routeB = poly([star, aB.p, aB.q, th0]);
      var qEnd = Q.at(1), tEnd = tDot ? [+tDot.getAttribute('cx'), +tDot.getAttribute('cy')] : null;
      var routeM = poly([qEnd, aR.p, aR.cut || aR.q].concat(aR.cut ? [] : (tEnd ? [tEnd] : [])));
      var qLen = 0, tLen = 0;
      try { qLen = qTrail.getTotalLength(); if (tTrail) tLen = tTrail.getTotalLength(); } catch (e) {}

      function frame(p) {
        var stat = p == null;
        /* phase 1: the two routes from Frozen to Full FT arrive together at theta0 */
        if (stat) { placeComet(cmA, routeC, null, 0); placeComet(cmB, routeB, null, 0); placeComet(cmC, routeM, null, 0); }
        else {
          var s1 = ramp(p, 0.02, 0.17), s2 = ramp(p, 0.17, 0.36), sb = ramp(p, 0.02, 0.36);
          var fadeA = p < 0.40 ? 1 : 0;
          if (st.cst === 'na') placeComet(cmA, routeC, null, 0);
          else if (st.cst === 'no') { placeComet(cmA, routeC, ease(s1), p < 0.24 ? 1 - ramp(p, 0.17, 0.24) : 0); }
          else if (p < 0.17) placeComet(cmA, routeC, ease(s1), 1);
          else if (st.rho === 'no' || st.rho === 'na') placeComet(cmA, routeR0, ease(s2), p < 0.40 ? 1 - ramp(p, 0.33, 0.40) : 0);
          else placeComet(cmA, routeR0, ease(s2), fadeA * (1 - ramp(p, 0.36, 0.40)));
          placeComet(cmB, routeB, ease(sb), p < 0.40 ? 1 - ramp(p, 0.36, 0.40) : 0);
          /* merge comet: a trained q rides rho_M to rho_M(q) */
          var sm = ramp(p, 0.80, 0.93);
          placeComet(cmC, routeM, ease(sm), p >= 0.80 && p < 0.97 ? (aR.cut ? 1 - ramp(p, 0.88, 0.95) : 1 - ramp(p, 0.93, 0.97)) : 0);
        }
        /* ring where the routes meet */
        var rp = stat ? -1 : ramp(p, 0.36, 0.52);
        if (rp > 0 && rp < 1 && st.cst !== 'no' && st.rho !== 'no' && st.rho !== 'na') {
          ring.setAttribute('cx', rq0[0]); ring.setAttribute('cy', rq0[1]);
          ring.setAttribute('r', (4 + 14 * rp).toFixed(1)); ring.setAttribute('opacity', (0.9 * (1 - rp)).toFixed(3));
          pulse.setAttribute('r', (rr + 10 * rp).toFixed(1)); pulse.setAttribute('opacity', (0.7 * (1 - rp)).toFixed(3));
        } else { ring.setAttribute('opacity', 0); pulse.setAttribute('opacity', 0); }
        /* training: q moves in Q_M, rho(q) in Theta */
        var tr = stat ? 1 : ease(ramp(p, 0.42, 0.78)), vis = stat ? 1 : (p < 0.42 ? 0 : 1 - ramp(p, 0.95, 1));
        var dv = stat ? 1 : Math.max(0.35, vis);
        if (qLen) { qTrail.setAttribute('stroke-dasharray', stat ? '2 3.2' : (qLen * tr).toFixed(1) + ' ' + (qLen + 10).toFixed(1)); qTrail.setAttribute('opacity', (0.85 * vis).toFixed(3)); }
        if (tTrail && tLen) { tTrail.setAttribute('stroke-dasharray', stat ? '2 3.2' : (tLen * tr).toFixed(1) + ' ' + (tLen + 10).toFixed(1)); tTrail.setAttribute('opacity', (0.9 * vis).toFixed(3)); }
        var qp = Q.at(tr);
        if (!stat && p < 0.42) qp = Q.at(1);
        qDot.setAttribute('cx', qp[0].toFixed(1)); qDot.setAttribute('cy', qp[1].toFixed(1)); qDot.setAttribute('opacity', dv.toFixed(3));
        qLab.setAttribute('x', (qp[0] + 7).toFixed(1)); qLab.setAttribute('y', (qp[1] - 6).toFixed(1)); qLab.setAttribute('opacity', dv.toFixed(3));
        if (tDot) {
          var tp = imgAt(!stat && p < 0.42 ? 1 : tr);
          tDot.setAttribute('cx', tp[0].toFixed(1)); tDot.setAttribute('cy', tp[1].toFixed(1)); tDot.setAttribute('opacity', dv.toFixed(3));
          tLab.setAttribute('x', (tp[0] + 7).toFixed(1)); tLab.setAttribute('y', (tp[1] - 7).toFixed(1)); tLab.setAttribute('opacity', dv.toFixed(3));
          var mp = stat ? -1 : ramp(p, 0.93, 1.0);
          if (mp > 0 && mp < 1) { ringM.setAttribute('cx', tp[0].toFixed(1)); ringM.setAttribute('cy', tp[1].toFixed(1)); ringM.setAttribute('r', (4.5 + 12 * mp).toFixed(1)); ringM.setAttribute('opacity', (0.9 * (1 - mp)).toFixed(3)); }
          else ringM.setAttribute('opacity', 0);
        }
      }
      A1 = { frame: frame };
      frame(null);
      schedule();
    }

    /* ---------- panel A: text blocks ---------- */
    function renderInfo() {
      var m = cur(), st = status(m), bud = budgetOf(m), c = m.coords || {};
      setTeX(fxBox, '<div class="iv-lab"><span class="nt">ρ<sub>M</sub></span> the map into <span class="nt">Θ</span> <span class="nt">' + A.esc(m.full_name && m.full_name !== m.name ? m.full_name : '') + '</span></div>' +
        (m.formula_latex ? '<div class="iv-fx iv-fxl">' + splitQuad(texFix(m.formula_latex)).map(function (t) { return '\\[' + A.esc(t) + '\\]'; }).join('') + '</div>' : '<p class="iv-empty">No closed-form ρ is catalogued.</p>'));
      var bh = '<div class="iv-lab">Budget \\(\\lvert Q_M\\rvert\\) <span class="nt">one 4096 × 4096 layer, r = 16</span></div>';
      if (bud.ok) {
        /* a percentage of one linear layer only makes sense for a per-matrix count */
        var showPct = !st.ext && bud.value > 0 && (!bud.qual || /matrix|linear|projection|identity/i.test(bud.qual));
        var subs = [];
        if (bud.used.r || bud.used.R) subs.push((bud.alias.length ? 'r_{' + bud.alias[0] + '}' : 'r') + '=16');
        if (bud.used.I && bud.used.O) subs.push('d_{in}=d_{out}=4096'); else if (bud.used.I) subs.push('d_{in}=4096'); else if (bud.used.O) subs.push('d_{out}=4096');
        if (bud.used.d) subs.push('d=4096');
        bh += '<div class="iv-bud"><span class="big">' + fmt(bud.value) + '</span>' + (bud.qual ? '<span class="q">' + A.esc(bud.qual) + (bud.plus ? ' + ' + A.esc(bud.plus) : '') + '</span>' : bud.plus ? '<span class="q">+ ' + A.esc(bud.plus) + '</span>' : '') +
          (showPct ? '<span class="pc">' + A.fmtPct(bud.value / THETA) + ' of |<span class="sym">Θ</span>|</span>' : '') + '</div>' +
          (subs.length ? '<div class="iv-sub"><span class="k">evaluated</span>\\(' + A.esc(bud.tex) + '\\) at \\(' + subs.join(',\\ ') + '\\)</div>' : '');
        if (bud.more && m.trainable_params && !bud.theta) bh += '<div class="iv-sub"><span class="k">catalogued</span>\\(' + A.esc(texFix(m.trainable_params)) + '\\)</div>';
        if (bud.theta && m.trainable_params) bh += '<div class="iv-sub"><span class="k">catalogued</span>\\(' + A.esc(texFix(m.trainable_params)) + '\\)</div>';
      } else {
        bh += '<div class="iv-bud"><span class="q">' + (bud.why === 'global' ? 'A whole-model budget, not a per-layer count.' : 'Not evaluated: the catalogued count needs more than \\(d_{in},d_{out},r\\), or is not a single number.') + '</span></div>' +
          (m.trainable_params ? '<div class="iv-sub"><span class="k">catalogued</span>\\(' + A.esc(texFix(m.trainable_params)) + '\\)</div>' : '');
      }
      setTeX(budBox, bh);
      banner.className = 'iv-banner ' + st.tone;
      var ktxt = c.kind === 'B' && c.merge === 'M1' ? KIND_TXT.B_stage : c.kind === 'R' && c.base_point === 'defect' ? KIND_TXT.R_defect :
        c.kind === 'R' && c.base_point === 'unpointed' ? KIND_TXT.R_unpointed : (KIND_TXT[c.kind] || '');
      var bpt = BP_TXT[c.base_point] || BP_TXT.none;
      var mgt = MG_TXT[c.merge] || '';
      setTeX(banner, '<div class="bt">' + A.esc(st.head) + '</div><div class="bs">' + ktxt + '</div><div class="bs">' + bpt + ' ' + mgt + '</div>');
    }

    /* ---------- coordinate tiles ---------- */
    function tileTone(axis, v) {
      if (axis === 'merge') return v === 'M1' || v === 'M1->' ? 'ok' : v === 'Mq' ? 'cond' : v === 'Minf' ? 'no' : '';
      if (axis === 'base_point') return v === 'regular' || v === 'neutral' ? 'ok' : v === 'apex' || v === 'defect' ? 'cond' : v === 'unpointed' ? 'no' : '';
      return '';
    }
    /* axis value names; a cone over a quantised or truncated base has its vertex at θ₀ + e, not at θ₀ */
    function valName(axis, v, av, c) {
      if (axis === 'shape' && v === 'Cone' && c.base_point === 'defect' && vshiftOf(cur())) return 'Cone with vertex at θ₀ + e';
      return pretty(av.name);
    }
    function renderTiles() {
      var m = cur(), c = m.coords || {};
      tiles.innerHTML = '';
      AX.forEach(function (a) {
        var v = c[a.key], av = axisVal[a.key] && axisVal[a.key].values[v];
        var b = h('button', { type: 'button', class: 'iv-tile ' + tileTone(a.key, v), 'aria-pressed': String(S0.tile === a.key) });
        b.appendChild(h('span', { class: 'ax', text: a.short }));
        b.appendChild(h('span', { class: 'v', text: showVal(a.key, v) }));
        b.appendChild(h('span', { class: 'n', html: subs(A.esc(av ? valName(a.key, v, av, c) : (v == null ? 'not applicable' : ''))) }));
        b.addEventListener('click', function () { S0.tile = a.key; renderTiles(); });
        tiles.appendChild(b);
      });
      var a0 = S0.tile, v0 = c[a0], av0 = axisVal[a0] && axisVal[a0].values[v0];
      var note = (m.coord_notes || {})[a0];
      var axName = axisVal[a0] ? axisVal[a0].name : a0;
      setTeX(cnote, '<b>' + A.esc(pretty(axName)) + ': ' + A.esc(showVal(a0, v0)) + (av0 ? ' (' + A.esc(valName(a0, v0, av0, c)) + ')' : '') + '.</b> ' +
        (note ? noteHTML(note, A.esc) : '<span style="color:var(--ink-2)">No method-specific note; the value follows the definition.</span>') +
        (av0 && av0.definition ? '<span class="def">' + A.esc(pretty(av0.definition)) + '</span>' : ''));
    }

    /* ======================= panel B: arrows ======================= */
    function arrowsOf(id) {
      var m = byId[id], st = status(m), out = [];
      var c = m.coords || {};
      var a1ok = (st.rho === 'ok' || st.rho === 'cond') && !st.ext;
      if (a1ok) out.push({ key: 'A1', kind: 'out', source: id, target: '__full', rule: 'A1', h: st.idRho ? 'h=\\mathrm{id}_\\Theta' : 'h=\\rho_M', note: (st.cst === 'ok' ? 'The terminal arrow. Every object maps into Full FT by its own \\(\\rho\\) (Obs I.4(a)), and this arrow is the merge.' : 'Here \\(\\rho_M\\) is only a weak morphism into Full FT, one that does not preserve base points, because ' + (c.base_point === 'defect' ? '\\(\\rho_M(q_0)=\\theta_0+e\\).' : c.base_point === 'unpointed' ? 'no initial value maps to \\(\\theta_0\\).' : 'the method has no base point of its own.')) });
      (outE[id] || []).forEach(function (e, i) { out.push({ key: 'o' + i, kind: 'out', source: id, target: e.target, rule: e.rule, h: e.h, note: e.note }); });
      var obs = (m.obstructions || []).map(function (o, i) { return { key: 'x' + i, kind: 'obs', source: id, target: o.target, rule: o.test, reason: o.reason }; });
      var inc = (inE[id] || []).map(function (e, i) { return { key: 'i' + i, kind: 'in', source: e.source, target: id, rule: e.rule, h: e.h, note: e.note }; });
      return { out: out, obs: obs, inc: inc };
    }
    function defaultArrow(L) {
      var o = L.out.filter(function (a) { return a.target !== '__full' && a.target !== a.source; });
      return (o[0] || L.out.filter(function (a) { return a.target !== '__full'; })[0] || L.out[0] || L.obs[0] || L.inc[0] || null);
    }
    var curL = null;
    function findArrow(key) {
      if (!curL || !key) return null;
      var all = curL.out.concat(curL.obs, curL.inc);
      for (var i = 0; i < all.length; i++) if (all[i].key === key) return all[i];
      return null;
    }
    function rowEl(a) {
      var other = a.kind === 'in' ? a.source : a.target, isObs = a.kind === 'obs';
      var b = h('button', { type: 'button', class: 'iv-row' + (isObs ? ' ob' : ''), 'aria-pressed': String(S0.arrow === a.key), 'data-mid': other });
      b.appendChild(h('span', { class: 'iv-badge' + (isObs ? ' s' : ''), text: a.rule }));
      b.appendChild(h('span', { class: 'iv-arr' + (isObs ? ' s' : ''), text: a.kind === 'in' ? '←' : isObs ? '↛' : '→' }));
      var tn = h('span', { class: 'iv-tn' });
      tn.appendChild(h('span', { class: 'sw', style: 'background:' + kcol(other) }));
      tn.appendChild(document.createTextNode(nameOf(other) + (a.source === a.target ? '′' : '')));
      b.appendChild(tn);
      var txt = isObs ? a.reason : a.note;
      if (txt) b.appendChild(h('span', { class: 'iv-rn', html: subs(A.esc(txt)) }));
      b.setAttribute('aria-label', (a.kind === 'in' ? nameOf(a.source) + ' to ' + nameOf(a.target) : nameOf(a.source) + (isObs ? ' has no arrow to ' : ' to ') + nameOf(a.target)) + ', ' + (isObs ? 'test ' : 'rule ') + a.rule);
      b.addEventListener('click', function () { S0.arrow = a.key; renderLists(); renderArrow(); dispatch(other); revealArrow(); });
      if (other !== '__full') bindCard(b, other); else bindCard(b, '__full');
      return b;
    }
    function renderLists() {
      var m = cur();
      lists.innerHTML = '';
      var sec = function (title, count, extra) {
        var w = h('div');
        w.appendChild(h('div', { class: 'iv-lh' }, [h('span', { text: title }), h('span', { text: count })].concat(extra ? [extra] : [])));
        lists.appendChild(w);
        return w;
      };
      var w1 = sec('Out of ' + m.name + ' · each target simulates it', String(curL.out.length));
      if (curL.out.length) { var ul = h('div', { class: 'iv-list' }); curL.out.forEach(function (a) { ul.appendChild(rowEl(a)); }); w1.appendChild(ul); }
      else w1.appendChild(h('p', { class: 'iv-empty', text: 'No arrow out of ' + m.name + ' is recorded' + (status(m).ext ? '; an extension has no ρ into Θ.' : '.') }));
      var w2 = sec('Obstructions · no arrow out of ' + m.name, String(curL.obs.length));
      if (curL.obs.length) { var ul2 = h('div', { class: 'iv-list' }); curL.obs.forEach(function (a) { ul2.appendChild(rowEl(a)); }); w2.appendChild(ul2); }
      else w2.appendChild(h('p', { class: 'iv-empty', text: 'None recorded.' }));
      var w3 = sec('Into ' + m.name + ' · it simulates each source', String(curL.inc.length));
      if (curL.inc.length) {
        var lim = S0.showAllIn ? curL.inc.length : 14, wrap = h('div', { class: 'iv-inc' });
        curL.inc.slice(0, lim).forEach(function (a) {
          var col = kcol(a.source);
          var b = h('button', { type: 'button', class: 'iv-chip', 'aria-pressed': String(S0.arrow === a.key), 'data-mid': a.source, style: '--kc:' + col, 'aria-label': nameOf(a.source) + ' to ' + m.name + ', rule ' + a.rule });
          b.appendChild(h('span', { class: 'sw', style: 'background:' + col }));
          b.appendChild(document.createTextNode(nameOf(a.source) + (a.source === a.target ? '′' : '')));
          b.addEventListener('click', function () { S0.arrow = a.key; renderLists(); renderArrow(); dispatch(a.source); revealArrow(); });
          bindCard(b, a.source);
          wrap.appendChild(b);
        });
        if (curL.inc.length > 14) {
          var more = h('button', { type: 'button', class: 'iv-mini', text: S0.showAllIn ? 'Show fewer' : 'Show all ' + curL.inc.length });
          more.addEventListener('click', function () { S0.showAllIn = !S0.showAllIn; renderLists(); });
          wrap.appendChild(more);
        }
        w3.appendChild(wrap);
      } else w3.appendChild(h('p', { class: 'iv-empty', text: 'None recorded.' }));
      typeset(lists);
    }

    function revealArrow() {
      try {
        var cols = getComputedStyle(p2).gridTemplateColumns.split(' ').length;
        if (cols < 2) dia2.scrollIntoView({ block: 'nearest', behavior: A.reducedMotion() ? 'auto' : 'smooth' });
      } catch (e) {}
    }
    var A2 = null;
    function lay2(W) {
      var c = W < 440, H = c ? 306 : 328, pad = c ? 4 : 10;
      var bw = c ? Math.min(118, W * 0.36) : Math.min(156, W * 0.33), bh = c ? 78 : 88;
      var Q1 = { w: bw, h: bh, cx: pad + bw / 2, cy: c ? 66 : 70 };
      var Q2 = { w: bw, h: bh, cx: W - pad - bw / 2, cy: Q1.cy };
      var Tt = { w: c ? Math.min(176, W * 0.54) : Math.min(212, W * 0.46), h: c ? 100 : 112 };
      Tt.cx = W / 2; Tt.cy = H - Tt.h / 2 - (c ? 38 : 42);
      return { W: W, H: H, c: c, Q1: Q1, Q2: Q2, T: Tt };
    }
    function renderArrow() {
      var a = findArrow(S0.arrow);
      dia2.innerHTML = '';
      var W = Math.max(280, dia2.getBoundingClientRect().width || dia2.clientWidth || 480), L = lay2(W), c = L.c;
      var ink = C('--ink'), ink2 = C('--ink-2'), ink3 = C('--ink-3'), seal = C('--seal'), moss = C('--moss'), sealI = C('--seal-ink'), mossI = C('--moss-ink');
      if (!a) {
        var sv0 = S('svg', { width: W, height: 80, viewBox: '0 0 ' + W + ' 80', role: 'img', 'aria-label': 'No arrows recorded' }, dia2);
        fit(T(sv0, W / 2, 44, 'No arrows or obstructions are recorded for ' + cur().name + '.', { 'text-anchor': 'middle', fill: ink2, 'font-size': 12 }), W - 8, 11);
        setTeX(det, '');
        A2 = null;
        return;
      }
      var M1 = byId[a.source], N1 = a.target === '__full' ? null : byId[a.target], isObs = a.kind === 'obs';
      var cM = kcol(a.source), cN = kcol(a.target);
      var svg = S('svg', { width: W, height: L.H, viewBox: '0 0 ' + W + ' ' + L.H, role: 'img' }, dia2);
      svg.setAttribute('aria-label', isObs ? 'No arrow from ' + nameOf(a.source) + ' to ' + nameOf(a.target) + ': test ' + a.rule + ' fails.' : 'Triangle: h from Q of ' + nameOf(a.source) + ' to Q of ' + nameOf(a.target) + ', both mapping to Theta, with rho_N after h equal to rho_M.');
      var gA = S('g', null, svg), gB = S('g', null, svg), gL = S('g', null, svg), gX = S('g', null, svg);
      var Q1 = L.Q1, Q2 = L.Q2, Tt = L.T;
      var cen = [(Q1.cx + Q2.cx + Tt.cx) / 3, (Q1.cy + Q2.cy + Tt.cy) / 3 + 4];

      function qbox(b, id, mm, colr) {
        var g = S('g', { tabindex: 0, role: 'button', 'data-mid': id, 'aria-label': nameOf(id) + ': open in the catalogue' }, gB);
        boxRect(g, b, { class: 'iv-focus', stroke: colr, 'stroke-opacity': 0.75, 'stroke-width': 1.2 });
        var cg = S('g', { 'clip-path': clipFor(svg, b, uid + '-c' + id.replace(/[^a-z0-9]/gi, '') + Math.random().toString(36).slice(2, 5)) }, g);
        var res;
        if (mm) res = drawQ(cg, b.cx, b.cy, b.w / 2 - 5, b.h / 2 - 5, mm, colr);
        else {
          S('rect', { x: b.cx - b.w / 2, y: b.cy - b.h / 2, width: b.w, height: b.h, fill: colr, 'fill-opacity': 0.08 }, cg);
          res = { q0: [b.cx - b.w * 0.1, b.cy + b.h * 0.1], at: function (t) { return [b.cx - b.w * 0.1 + b.w * 0.42 * t, b.cy + b.h * 0.1 - b.h * 0.3 * t]; } };
        }
        g.addEventListener('click', function () { dispatch(id); });
        g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); dispatch(id); } });
        bindCard(g, id);
        var tt = T(gL, b.cx, b.cy - b.h / 2 - 8, nameOf(id) + (a.source === a.target && b === Q2 ? '′' : ''), { class: 'disp', 'font-size': c ? 15 : 17, fill: colr, 'text-anchor': 'middle' });
        fit(tt, b.w + 14, 12);
        return res;
      }
      var QA = qbox(Q1, a.source, M1, cM), QB = qbox(Q2, a.target, N1, cN);
      RT(gL, Q1.cx, Q1.cy + Q1.h / 2 + 16, [['Q', 'it'], ['M', 'sub it']], { class: 'ser', 'font-size': 13, fill: ink2, 'text-anchor': 'middle' });
      RT(gL, Q2.cx, Q2.cy + Q2.h / 2 + 16, N1 ? [['Q', 'it'], ['N', 'sub it']] : [['Θ', 'it']], { class: 'ser', 'font-size': 13, fill: ink2, 'text-anchor': 'middle' });

      boxRect(gB, Tt);
      var tg = S('g', { 'clip-path': clipFor(svg, Tt, uid + '-c2t') }, gB);
      var th0 = [Tt.cx - Tt.w * 0.08, Tt.cy + Tt.h * 0.14], s = Math.min(Tt.w, Tt.h) * 0.44;
      var stM = status(M1), shM = (M1.coords || {}).shape;
      if (N1) {
        var shN = (N1.coords || {}).shape;
        if (status(N1).idRho) S('rect', { x: Tt.cx - Tt.w / 2, y: Tt.cy - Tt.h / 2, width: Tt.w, height: Tt.h, fill: cN, 'fill-opacity': 0.07 }, tg);
        else drawImage(tg, shN, th0[0], th0[1], s * 0.98, cN, { fa: 0.1, faint: true });
      } else S('rect', { x: Tt.cx - Tt.w / 2, y: Tt.cy - Tt.h / 2, width: Tt.w, height: Tt.h, fill: cN, 'fill-opacity': 0.07 }, tg);
      var IM = drawImage(tg, stM.idRho ? 'Θ' : shM, th0[0], th0[1], s * 0.74, cM, { fa: 0.18 });
      if (stM.idRho) S('rect', { x: Tt.cx - Tt.w / 2, y: Tt.cy - Tt.h / 2, width: Tt.w, height: Tt.h, fill: cM, 'fill-opacity': 0.08 }, tg);
      dot(gX, th0, 3.4, ink);
      RT(gX, th0[0] - 6, th0[1] + 14, [['θ', 'it'], ['0', 'sub']], { class: 'ser halo', 'font-size': 13, fill: ink2, 'text-anchor': 'end' });
      RT(gL, Tt.cx, Tt.cy + Tt.h / 2 + 16, [['(Θ, θ', 'it'], ['0', 'sub'], [')', 'it']], { class: 'ser', 'font-size': 13, fill: ink2, 'text-anchor': 'middle' });
      fit(T(gL, Tt.cx, Tt.cy + Tt.h / 2 + 32, isObs ? (a.rule === 'O1' ? 'Im M ⊄ Im N' : a.rule === 'O2' ? 'd(M) > d(N), or a larger max rank' : 'S₁(M) ⊄ S₁(N), or another stratum') : 'Im M ⊆ Im N', { 'font-size': 12, fill: isObs ? sealI : ink2, 'text-anchor': 'middle' }), W - 8, 11);

      /* arrows */
      var hP = edgePt([Q1.cx, Q1.cy], Q1.w / 2, Q1.h / 2, [Q2.cx, Q2.cy], 6), hQ = edgePt([Q2.cx, Q2.cy], Q2.w / 2, Q2.h / 2, [Q1.cx, Q1.cy], 7);
      var aH = arrow(gA, hP, hQ, { tone: isObs ? 'no' : 'ok', cut: isObs ? 0.5 : null, hook: !isObs && a.rule === 'A2', w: 1.5, col: isObs ? null : ink });
      var aM = arrow(gA, edgePt([Q1.cx, Q1.cy], Q1.w / 2, Q1.h / 2, [Tt.cx, Tt.cy], 6), edgePt([Tt.cx, Tt.cy], Tt.w / 2, Tt.h / 2, [Q1.cx, Q1.cy], 7), { tone: 'ok', col: cM });
      var aN = arrow(gA, edgePt([Q2.cx, Q2.cy], Q2.w / 2, Q2.h / 2, [Tt.cx, Tt.cy], 6), edgePt([Tt.cx, Tt.cy], Tt.w / 2, Tt.h / 2, [Q2.cx, Q2.cy], 7), { tone: 'ok', col: N1 ? cN : ink2 });
      var hy = (aH.p[1] + aH.q[1]) / 2;
      if (isObs) {
        RT(gL, (aH.p[0] + aH.q[0]) / 2, hy - 13, [['no ', ''], ['h', 'it']], { class: 'ser', 'font-size': c ? 14 : 15, fill: sealI, 'text-anchor': 'middle' });
      } else {
        RT(gL, (aH.p[0] + aH.q[0]) / 2, hy - 9, [['h', 'it']], { class: 'ser', 'font-size': c ? 15 : 17, fill: ink, 'text-anchor': 'middle' });
      }
      var bx = (aH.p[0] + aH.q[0]) / 2;
      var bg = S('g', null, gL);
      var bwid = 28, by = hy + (isObs ? 12 : 6);          // below the ⊘ mark when there is one
      S('rect', { x: bx - bwid / 2, y: by, width: bwid, height: 17, rx: 3, fill: C('--paper-2'), stroke: isObs ? seal : C('--tide'), 'stroke-width': 1 }, bg);
      T(bg, bx, by + 12.6, a.rule, { class: 'ui', 'font-size': 12, fill: isObs ? sealI : C('--tide'), 'text-anchor': 'middle', 'letter-spacing': '0.02em' });
      var lm = sideLabel(aM, cen, 9), ln = sideLabel(aN, cen, 9);
      RT(gL, lm.x, lm.y, [['ρ', 'it'], ['M', 'sub it']], { class: 'ser', 'font-size': c ? 13.5 : 15.5, fill: cM, 'text-anchor': lm.anchor });
      RT(gL, ln.x, ln.y, N1 ? [['ρ', 'it'], ['N', 'sub it']] : [['id', ''], ['Θ', 'sub']], { class: 'ser', 'font-size': c ? 13.5 : 15.5, fill: N1 ? cN : ink2, 'text-anchor': ln.anchor });

      /* commutation mark */
      var mk = S('g', { transform: 'translate(' + cen[0].toFixed(1) + ',' + cen[1].toFixed(1) + ')' }, gL);
      var rr = c ? 10 : 12, mcol = isObs ? seal : moss;
      if (isObs) {
        S('circle', { r: rr, fill: 'none', stroke: mcol, 'stroke-width': 1.3, 'stroke-dasharray': '3 3' }, mk);
        S('line', { x1: -rr * 0.6, y1: rr * 0.6, x2: rr * 0.6, y2: -rr * 0.6, stroke: mcol, 'stroke-width': 1.3, 'stroke-linecap': 'round' }, mk);
      } else {
        var a0 = -Math.PI * 0.85, a1 = Math.PI * 0.55;
        S('path', { d: 'M' + (rr * Math.cos(a0)).toFixed(2) + ',' + (rr * Math.sin(a0)).toFixed(2) + ' A' + rr + ',' + rr + ' 0 1 1 ' + (rr * Math.cos(a1)).toFixed(2) + ',' + (rr * Math.sin(a1)).toFixed(2), fill: 'none', stroke: mcol, 'stroke-width': 1.5, 'stroke-linecap': 'round' }, mk);
        var ex = rr * Math.cos(a1), ey = rr * Math.sin(a1), ta = a1 + Math.PI / 2;
        S('polyline', { points: [ex - 5 * Math.cos(ta - 0.5), ey - 5 * Math.sin(ta - 0.5)].join(',') + ' ' + ex + ',' + ey + ' ' + [ex - 5 * Math.cos(ta + 0.5), ey - 5 * Math.sin(ta + 0.5)].join(','), fill: 'none', stroke: mcol, 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, mk);
      }
      var eq = isObs ? [['∄ ', ''], ['h', 'it'], [' : ρ', 'it'], ['N', 'sub it'], [' ∘ ', ''], ['h', 'it'], [' = ρ', 'it'], ['M', 'sub it']] : [['ρ', 'it'], ['N', 'sub it'], [' ∘ ', ''], ['h', 'it'], [' = ρ', 'it'], ['M', 'sub it']];
      RT(mk, 0, rr + (c ? 15 : 17), eq, { class: 'ser', 'font-size': 13, fill: isObs ? sealI : mossI, 'text-anchor': 'middle' });
      var pulse = S('circle', { r: rr, fill: 'none', stroke: mcol, 'stroke-width': 1.2, opacity: 0 }, mk);

      /* moving points: q in Q_M, h(q) in Q_N, p = rho_M(q) = rho_N(h(q)) in Theta */
      var tq0 = 0.7;
      var qpt = QA.at(tq0), hq = QB.at(tq0);
      var ppt = IM.at ? IM.at(0.75) : (IM.fun || th0);
      dot(gX, qpt, 3.8, cM, C('--paper'));
      RT(gX, qpt[0] + 6, qpt[1] - 6, [['q', 'it']], { class: 'ser halo', 'font-size': 13, fill: cM });
      if (!isObs) {
        dot(gX, hq, 3.8, N1 ? cN : ink2, C('--paper'));
        keepIn(RT(gX, hq[0] + 6, hq[1] - 6, [['h', 'it'], ['(q)', 'it']], { class: 'ser halo', 'font-size': 13, fill: N1 ? cN : ink2 }), 2, W - 2);
      }
      dot(gX, ppt, 4, cM, C('--paper'));
      var ring = S('circle', { cx: ppt[0], cy: ppt[1], r: 4, fill: 'none', stroke: moss, 'stroke-width': 1.6, opacity: 0 }, gX);
      var cmH = comet(gX, ink), cmN = comet(gX, N1 ? cN : ink2), cmM = comet(gX, cM);
      var rH = poly([qpt, aH.p, aH.cut || aH.q].concat(aH.cut ? [] : [hq]));
      var rN = poly([hq, aN.p, aN.q, ppt]);
      var rM = poly([qpt, aM.p, aM.q, ppt]);
      function frame(p) {
        if (p == null) { placeComet(cmH, rH, null, 0); placeComet(cmN, rN, null, 0); placeComet(cmM, rM, null, 0); ring.setAttribute('opacity', 0); pulse.setAttribute('opacity', 0); return; }
        var s1 = ramp(p, 0.04, 0.30), s2 = ramp(p, 0.30, 0.58), sm = ramp(p, 0.04, 0.58);
        if (isObs) placeComet(cmH, rH, ease(s1), p < 0.36 ? 1 - ramp(p, 0.28, 0.36) : 0);
        else if (p < 0.30) placeComet(cmH, rH, ease(s1), 1);
        else placeComet(cmH, rH, null, 0);
        if (!isObs) placeComet(cmN, rN, ease(s2), p >= 0.30 && p < 0.62 ? 1 - ramp(p, 0.58, 0.62) : 0);
        else placeComet(cmN, rN, null, 0);
        placeComet(cmM, rM, ease(sm), p < 0.62 ? 1 - ramp(p, 0.58, 0.62) : 0);
        var rp = ramp(p, 0.58, 0.78);
        if (!isObs && rp > 0 && rp < 1) {
          ring.setAttribute('r', (4 + 13 * rp).toFixed(1)); ring.setAttribute('opacity', (0.9 * (1 - rp)).toFixed(3));
          pulse.setAttribute('r', (rr + 9 * rp).toFixed(1)); pulse.setAttribute('opacity', (0.7 * (1 - rp)).toFixed(3));
        } else { ring.setAttribute('opacity', 0); pulse.setAttribute('opacity', 0); }
      }
      A2 = { frame: frame };
      frame(null);
      renderDetail(a);
      schedule();
    }
    function renderDetail(a) {
      var isObs = a.kind === 'obs', rule = ruleBy[a.rule] || {}, other = a.kind === 'in' ? a.source : a.target;
      var head = '<div class="iv-dh"><span class="t">' + A.esc(nameOf(a.source)) + (isObs ? ' <span class="x">↛</span> ' : ' → ') + A.esc(nameOf(a.target)) + (a.source === a.target ? '′' : '') + '</span>' +
        '<span class="r">' + A.esc(a.rule + (rule.name ? ' · ' + rule.name : '')) + '</span></div>';
      var body = '';
      if (isObs) {
        body += '<div class="iv-box"><div class="iv-lab">Failed test</div><div class="iv-dn">' + prettyHTML(rule.rule || '', A.esc) + '</div></div>';
        body += '<div class="iv-dn">' + noteHTML(a.reason || '', A.esc) + '</div>';
      } else {
        body += '<div class="iv-box"><div class="iv-lab">The map <span class="nt"><i>h</i></span> <span class="nt">' + (a.kind === 'in' ? A.esc(nameOf(a.target)) + ' simulates ' + A.esc(nameOf(a.source)) : A.esc(nameOf(a.target)) + ' simulates ' + A.esc(nameOf(a.source))) + '</span></div><div class="iv-fx">\\[' + A.esc(texFix(a.h)) + '\\]</div></div>';
        if (a.note) body += '<div class="iv-dn">' + noteHTML(a.note, A.esc) + '</div>';
        /* A1 holds for weight-space methods only (Obs I.4); the catalogue's wording says "every method" */
        var rtext = a.rule === 'A1' ? 'For every weight-space method M: Frozen -> M -> Full FT.' : rule.rule;
        if (rtext) body += '<div class="iv-dn" style="font-size:.8rem;color:var(--ink-2)">' + prettyHTML(a.rule + ': ' + rtext, A.esc) + '</div>';
      }
      setTeX(det, head + body);
      var acts = h('div', { class: 'iv-acts' });
      var walkTo = a.kind === 'in' ? a.source : a.target;
      if (walkTo !== '__full' && byId[walkTo] && walkTo !== S0.src) {
        var b1 = h('button', { type: 'button', class: 'iv-mini', text: 'Make ' + nameOf(walkTo) + ' the method M' });
        b1.addEventListener('click', function () { setSource(walkTo, true); });
        acts.appendChild(b1);
      }
      if (other !== '__full' && byId[other]) {
        var b2 = h('button', { type: 'button', class: 'iv-mini', text: 'Open ' + nameOf(other) + ' in the catalogue' });
        b2.addEventListener('click', function () { dispatch(other); });
        acts.appendChild(b2);
      }
      if (acts.childNodes.length) det.appendChild(acts);
    }

    /* ---------- legend ---------- */
    function renderLegend() {
      legend.innerHTML = '';
      legend.appendChild(h('span', { class: 'lt', text: 'Colour = modification kind' }));
      A.KINDS.forEach(function (k) {
        var s = h('span', { class: 'li' });
        s.appendChild(h('span', { class: 'sw', style: 'background:' + A.css(k.token) }));
        s.appendChild(document.createTextNode(k.name));
        legend.appendChild(s);
      });
      var s1 = h('span', { class: 'li', style: 'color:var(--moss-ink)' }); s1.appendChild(h('span', { class: 'ln' })); s1.appendChild(document.createTextNode('commutes')); legend.appendChild(s1);
      var s2 = h('span', { class: 'li', style: 'color:var(--ochre-ink)' }); s2.appendChild(h('span', { class: 'ln' })); s2.appendChild(document.createTextNode('condition')); legend.appendChild(s2);
      var s3 = h('span', { class: 'li', style: 'color:var(--seal-ink)' }); s3.appendChild(h('span', { class: 'ln d' })); s3.appendChild(document.createTextNode('obstruction ⊘')); legend.appendChild(s3);
    }

    /* ======================= state ======================= */
    function setSource(id, fire) {
      if (!byId[id]) return;
      var changed = id !== S0.src;
      S0.src = id;
      curL = arrowsOf(id);
      var d0 = defaultArrow(curL);
      S0.arrow = d0 ? d0.key : null;
      S0.showAllIn = false;
      if (!byId[id].core || sel.querySelector('optgroup[label^="Current"]')) buildSelect();
      sel.value = id; sel2.value = id;
      syncChips();
      hideCard();
      renderTri(); renderInfo(); renderTiles(); renderLists(); renderArrow();
      if (changed) live.textContent = byId[id].name + ' selected. ' + status(byId[id]).head + '.';
      if (fire) dispatch(id);
    }
    document.addEventListener('atlas:select-method', function (e) {
      if (selfFire) return;
      var id = e && e.detail && (e.detail.id || e.detail);
      if (typeof id === 'string' && byId[id] && id !== S0.src) setSource(id, false);
    });

    /* ---------- animation loop ---------- */
    var raf = 0, visible = true, t0 = performance.now();
    function running() { return !reduced && !S0.paused && visible && !document.hidden; }
    function tick(now) {
      raf = 0;
      if (!running()) return;
      var e = now - t0;
      if (A1) A1.frame((e % PER1) / PER1);
      if (A2) A2.frame(((e + 900) % PER2) / PER2);
      raf = requestAnimationFrame(tick);
    }
    function schedule() {
      if (running()) { if (!raf) raf = requestAnimationFrame(tick); }
      else { if (raf) { cancelAnimationFrame(raf); raf = 0; } if (A1) A1.frame(null); if (A2) A2.frame(null); }
    }
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) { es.forEach(function (en) { visible = en.isIntersecting; }); schedule(); }, { rootMargin: '120px 0px' }).observe(el);
    }
    document.addEventListener('visibilitychange', schedule);
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-reduced-motion: reduce)');
      var onMq = function () { reduced = mq.matches; pauseBtn.hidden = reduced; schedule(); };
      if (mq.addEventListener) mq.addEventListener('change', onMq); else if (mq.addListener) mq.addListener(onMq);
    }

    /* ---------- theme, resize, MathJax ---------- */
    A.onTheme(function () { buildChips(); renderLegend(); renderTri(); renderLists(); renderArrow(); });
    var lastW1 = 0, lastW2 = 0;
    var onResize = A.debounce(function () {
      var w1 = Math.round(dia1.clientWidth), w2 = Math.round(dia2.clientWidth);
      if (w1 && w1 !== lastW1) { lastW1 = w1; renderTri(); }
      if (w2 && w2 !== lastW2) { lastW2 = w2; renderArrow(); }
      fitMath(fxBox); fitMath(budBox); fitMath(det);
    }, 90);
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(stage); else window.addEventListener('resize', onResize);
    document.addEventListener('atlas:mathjax', function () { typeset(stage); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { renderTri(); renderArrow(); });

    /* ---------- initial state: complete at rest ---------- */
    buildSelect();
    buildChips();
    renderLegend();
    typeset(stage);
    setSource(S0.src, false);
    lastW1 = Math.round(dia1.clientWidth); lastW2 = Math.round(dia2.clientWidth);

    /* headless hook for tests */
    el.__interval = {
      setSource: setSource, budget: budget, status: status, state: S0,
      select: function (key) { S0.arrow = key; renderLists(); renderArrow(); },
      freeze: function (p1, p2) { S0.paused = true; schedule(); if (A1) A1.frame(p1); if (A2) A2.frame(p2); }
    };
  });
  Atlas._intervalBudget = budget;
})();
