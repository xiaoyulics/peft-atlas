/* Fig. "catalog" — The Catalogue: every method, searchable, citable.
   A three-pane explorer (filter rail · result list · method card) over window.ATLAS_DATA.
   Every count on screen is computed from the data at render time. Works with or without the
   classification fields (coords, arrows, obstructions, categorical_reading, ...). */
(function () {
  'use strict';
  var FIG = 'catalog';
  var SCOPE = '[data-figure="catalog"]';

  /* =====================================================================
     1. TeX → Unicode HTML (texHTML).  Used for math islands in compact places (list rows,
        tooltips), for names ("Spectral Adapter^R"), and as a fallback for a prose field that
        still holds raw TeX commands without delimiters. Card prose with explicit \( … \)
        islands, formula_latex and trainable_params are typeset by MathJax (see prose()).
     ===================================================================== */
  var GREEK = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ϵ', varepsilon: 'ε', zeta: 'ζ', eta: 'η',
    theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π',
    varpi: 'ϖ', rho: 'ρ', varrho: 'ϱ', sigma: 'σ', varsigma: 'ς', tau: 'τ', upsilon: 'υ', phi: 'ϕ',
    varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω', Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ',
    Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω'
  };
  var SYM = {
    times: '×', cdot: '·', cdotp: '·', circ: '∘', bullet: '•', ast: '∗', star: '⋆', otimes: '⊗', oplus: '⊕',
    odot: '⊙', ominus: '⊖', oslash: '⊘', boxtimes: '⊠', boxplus: '⊞', bigotimes: '⨂', bigoplus: '⨁',
    bigodot: '⨀', div: '÷', pm: '±', mp: '∓', setminus: '∖', backslash: '\\', wr: '≀', rtimes: '⋊',
    ltimes: '⋉', bowtie: '⋈', sqcup: '⊔', sqcap: '⊓', bigsqcup: '⨆', uplus: '⊎', cap: '∩', cup: '∪',
    bigcap: '⋂', bigcup: '⋃', wedge: '∧', land: '∧', vee: '∨', lor: '∨', bigwedge: '⋀', bigvee: '⋁',
    neg: '¬', lnot: '¬', in: '∈', notin: '∉', ni: '∋', subset: '⊂', subseteq: '⊆', subsetneq: '⊊',
    supset: '⊃', supseteq: '⊇', emptyset: '∅', varnothing: '∅', le: '≤', leq: '≤', ge: '≥', geq: '≥',
    leqslant: '⩽', geqslant: '⩾', ll: '≪', gg: '≫', ne: '≠', neq: '≠', approx: '≈', sim: '∼', simeq: '≃',
    cong: '≅', equiv: '≡', propto: '∝', asymp: '≍', doteq: '≐', triangleq: '≜', coloneqq: '≔',
    coloneq: '≔', prec: '≺', succ: '≻', preceq: '⪯', succeq: '⪰', lesssim: '≲', gtrsim: '≳',
    models: '⊨', vdash: '⊢', dashv: '⊣', perp: '⊥', parallel: '∥', mid: '∣', nmid: '∤',
    to: '→', rightarrow: '→', leftarrow: '←', gets: '←', leftrightarrow: '↔', Rightarrow: '⇒',
    Leftarrow: '⇐', Leftrightarrow: '⇔', iff: '⟺', implies: '⟹', impliedby: '⟸', longrightarrow: '⟶',
    Longrightarrow: '⟹', longleftarrow: '⟵', longmapsto: '⟼', mapsto: '↦', hookrightarrow: '↪',
    hookleftarrow: '↩', twoheadrightarrow: '↠', rightrightarrows: '⇉', rightleftharpoons: '⇌',
    uparrow: '↑', downarrow: '↓', nearrow: '↗', searrow: '↘', leadsto: '⇝', rightsquigarrow: '⇝',
    nrightarrow: '↛', infty: '∞', partial: '∂', nabla: '∇', sum: '∑', prod: '∏', coprod: '∐', int: '∫',
    iint: '∬', oint: '∮', forall: '∀', exists: '∃', nexists: '∄', top: '⊤', bot: '⊥', dagger: '†',
    ddagger: '‡', dag: '†', ell: 'ℓ', hbar: 'ℏ', imath: 'ı', jmath: 'ȷ', aleph: 'ℵ', wp: '℘',
    Re: 'Re', Im: 'Im', angle: '∠', triangle: '△', square: '□', Box: '□', diamond: '⋄', checkmark: '✓',
    prime: '′', S: '§', ldots: '…', cdots: '⋯', dots: '…', dotsc: '…', dotsb: '⋯', vdots: '⋮',
    ddots: '⋱', langle: '⟨', rangle: '⟩', lceil: '⌈', rceil: '⌉', lfloor: '⌊', rfloor: '⌋', lbrace: '{',
    rbrace: '}', lbrack: '[', rbrack: ']', vert: '|', lvert: '|', rvert: '|', Vert: '‖', lVert: '‖',
    rVert: '‖', colon: ':', sharp: '♯', flat: '♭', natural: '♮'
  };
  var BINREL = {};
  ('times otimes oplus odot circ in notin subset subseteq subsetneq supset supseteq le leq ge geq ne neq approx sim simeq cong equiv propto to rightarrow leftarrow gets mapsto Rightarrow implies iff hookrightarrow twoheadrightarrow longrightarrow cap cup setminus pm sqcup rtimes ltimes ll gg lesssim gtrsim models vdash perp')
    .split(' ').forEach(function (k) { BINREL[k] = 1; });
  var SPACE = { quad: '\u2003', qquad: '\u2003\u2003', enspace: '\u2002', thinspace: '\u2009', medspace: '\u205f', thickspace: '\u2005', space: ' ', nobreakspace: '\u00a0' };
  var FUNCS = {};
  ('log ln lg exp sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh coth max min arg argmin argmax sup inf lim liminf limsup det dim ker deg rank tr Tr diag softmax sign sgn mod bmod gcd lcm Pr vol span Hom End Aut id rk erf')
    .split(' ').forEach(function (f) { FUNCS[f] = 1; });
  var ACCENT = { hat: '\u0302', widehat: '\u0302', bar: '\u0304', overline: '\u0305', tilde: '\u0303', widetilde: '\u0303', vec: '\u20d7', dot: '\u0307', ddot: '\u0308', check: '\u030c', breve: '\u0306', acute: '\u0301', grave: '\u0300', mathring: '\u030a' };
  var TEXTCMD = { text: 1, textrm: 1, textup: 1, textnormal: 1, mathrm: 1, operatorname: 1, mathop: 1, mbox: 1, hbox: 1, textsf: 1, mathsf: 1, mathfrak: 1, mathnormal: 1, textsc: 1, textmd: 1 };
  var BOLD = { mathbf: 1, textbf: 1, boldsymbol: 1, bm: 1, pmb: 1 };
  var ITAL = { mathit: 1, textit: 1, emph: 1, textsl: 1 };
  var MONO = { mathtt: 1, texttt: 1 };
  var DROP = { left: 1, right: 1, big: 1, Big: 1, bigg: 1, Bigg: 1, bigl: 1, bigr: 1, Bigl: 1, Bigr: 1, biggl: 1, biggr: 1, Biggl: 1, Biggr: 1, middle: 1, displaystyle: 1, textstyle: 1, scriptstyle: 1, scriptscriptstyle: 1, limits: 1, nolimits: 1, nobreak: 1, allowbreak: 1, relax: 1, mathstrut: 1, strut: 1 };
  var SKIPARG = { hspace: ' ', vspace: '', color: '', label: '', tag: '', phantom: '', hphantom: '', vphantom: '' };
  var NOT = { '∈': '∉', '⊆': '⊈', '⊂': '⊄', '=': '≠', '∼': '≁', '≡': '≢', '⊇': '⊉', '⊃': '⊅', '∣': '∤', '|': '∤', '≤': '≰', '≥': '≱' };
  var BB = { A: '𝔸', B: '𝔹', C: 'ℂ', D: '𝔻', E: '𝔼', F: '𝔽', G: '𝔾', H: 'ℍ', I: '𝕀', J: '𝕁', K: '𝕂', L: '𝕃', M: '𝕄', N: 'ℕ', O: '𝕆', P: 'ℙ', Q: 'ℚ', R: 'ℝ', S: '𝕊', T: '𝕋', U: '𝕌', V: '𝕍', W: '𝕎', X: '𝕏', Y: '𝕐', Z: 'ℤ', '1': '𝟙' };
  var CAL = { A: '𝒜', B: 'ℬ', C: '𝒞', D: '𝒟', E: 'ℰ', F: 'ℱ', G: '𝒢', H: 'ℋ', I: 'ℐ', J: '𝒥', K: '𝒦', L: 'ℒ', M: 'ℳ', N: '𝒩', O: '𝒪', P: '𝒫', Q: '𝒬', R: 'ℛ', S: '𝒮', T: '𝒯', U: '𝒰', V: '𝒱', W: '𝒲', X: '𝒳', Y: '𝒴', Z: '𝒵' };

  function escC(c) { return c === '&' ? '&amp;' : c === '<' ? '&lt;' : c === '>' ? '&gt;' : c === '"' ? '&quot;' : c; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function plainOf(h) { return String(h).replace(/<[^>]*>/g, '').replace(/&[a-z#0-9]+;/g, 'x'); }
  function cpLen(s) { return Array.from ? Array.from(s).length : s.length; }

  function texHTML(src) {
    src = String(src == null ? '' : src);
    if (!/[\\^_{}]/.test(src)) return esc(src);
    var p = { s: src, i: 0, tex: /\\[A-Za-z]/.test(src), se: -1 };
    return seq(p, '');
  }
  function seq(p, stop) {
    var out = '', s = p.s;
    while (p.i < s.length) {
      var c = s.charAt(p.i);
      if (stop && c === stop) { p.i++; return out; }
      if (c === '\\') {
        p.bin = false;
        var cm = cmd(p);
        if (p.bin) { out = out.replace(/[ \u2009]+$/, ''); skipSp(p); out += '\u2009' + cm + '\u2009'; }
        else {
          /* "\Delta W" -> "ΔW": a Greek letter followed by a single-letter variable */
          if (p.greek && /^ [A-Za-z](?![A-Za-z])/.test(s.slice(p.i, p.i + 3))) p.i++;
          out += cm;
        }
        p.greek = false;
        continue;
      }
      if (c === '^' || c === '_') { var r = script(p); if (r !== null) { out += r; continue; } }
      if (p.tex && c === '{') { p.i++; out += seq(p, '}'); continue; }
      if (p.tex && c === '}') { p.i++; continue; }
      out += escC(c); p.i++;
    }
    return out;
  }
  function skipSp(p) { while (p.s.charAt(p.i) === ' ') p.i++; }
  /* is the character run before ^ or _ a mathematical base (a single symbol), not a word like lora_A? */
  function baseOK(p) {
    var s = p.s, j = p.i - 1;
    if (j < 0) return false;
    if (j === p.se - 1) return true; /* stacked scripts: W_q^2 */
    var c = s.charAt(j), k;
    if (/[)\]}'*′]/.test(c)) return true;
    if (/[\u0370-\u03ff]/.test(c)) return true;
    if (/[0-9]/.test(c)) {
      k = j; while (k >= 0 && k >= p.se && /[0-9]/.test(s.charAt(k))) k--;
      return k < 0 || k < p.se || !/[A-Za-z]/.test(s.charAt(k));
    }
    if (/[A-Za-z]/.test(c)) {
      k = j; while (k >= 0 && k >= p.se && /[A-Za-z]/.test(s.charAt(k))) k--;
      var run = s.slice(k + 1, j + 1);
      if (k >= 0 && s.charAt(k) === '\\') return true;
      if (run.length === 1) return true;
      return /[A-Z]/.test(run) && run.length <= 7;
    }
    return false;
  }
  function script(p) {
    if (!baseOK(p)) return null;
    var s = p.s, start = p.i, tag = s.charAt(p.i) === '^' ? 'sup' : 'sub', arg = null, m;
    p.i++;
    var c = s.charAt(p.i);
    if (c === '{') { p.i++; arg = seq(p, '}'); }
    else if (c === '\\') arg = cmd(p);
    else if (/[0-9]/.test(c)) { m = /^[0-9]+/.exec(s.slice(p.i))[0]; p.i += m.length; arg = m; }
    else if ((c === '-' || c === '+') && /[0-9A-Za-z]/.test(s.charAt(p.i + 1))) {
      p.i++; m = /^([0-9]+|[A-Za-z])/.exec(s.slice(p.i))[0]; p.i += m.length; arg = (c === '-' ? '\u2212' : '+') + m;
    } else if (/[A-Za-z]/.test(c)) {
      var w = /^[A-Za-z]+/.exec(s.slice(p.i))[0], nx = s.charAt(p.i + w.length);
      if (w.length > 1 && w.length <= 4 && /^[a-z]+$/.test(w) && !/[0-9_^]/.test(nx)) { p.i += w.length; arg = w; }
      else { p.i++; arg = c; }
    } else if (c === '*' || c === "'" || c === '′') { p.i++; arg = c === "'" ? '′' : c; }
    else if (/[\u0370-\u03ff]/.test(c)) { p.i++; arg = c; }
    if (arg === null) { p.i = start; return null; }
    p.se = p.i;
    arg = arg.replace(/^[\u2009 ]+|[\u2009 ]+$/g, '').replace(/-(?![^<]*>)/g, '\u2212');
    return '<' + tag + '>' + arg + '</' + tag + '>';
  }
  function arg(p, raw) {
    skipSp(p);
    var s = p.s, c = s.charAt(p.i);
    if (c === '{') {
      if (raw) {
        var depth = 0, j = p.i;
        for (; j < s.length; j++) { if (s.charAt(j) === '{') depth++; else if (s.charAt(j) === '}') { depth--; if (!depth) break; } }
        var t = s.slice(p.i + 1, j); p.i = j + 1; return t;
      }
      p.i++; return seq(p, '}');
    }
    if (c === '\\') {
      if (raw) { var mm = /^\\([A-Za-z]+|.)/.exec(s.slice(p.i)); p.i += mm[0].length; return mm[1]; }
      return cmd(p);
    }
    if (!c) return '';
    p.i++; return raw ? c : escC(c);
  }
  function par(h) {
    var t = plainOf(h);
    return cpLen(t) <= 1 || /^[A-Za-z0-9\u0370-\u03ff.′]+$/.test(t) ? h : '(' + h + ')';
  }
  function mapChars(raw, map) { return Array.from(raw).map(function (ch) { return map[ch] || escC(ch); }).join(''); }
  function accent(a, mark, name) {
    var t = plainOf(a);
    if (cpLen(t) === 1 && a.indexOf('<') < 0) return a + mark;
    if (name === 'bar' || name === 'overline') return '<span class="ov">' + a + '</span>';
    return a + mark;
  }
  function cmd(p) {
    var s = p.s; p.i++;
    var c = s.charAt(p.i);
    if (!c) return '';
    if (!/[A-Za-z]/.test(c)) {
      p.i++;
      switch (c) {
        case ',': return '\u2009';
        case ';': case ':': case '>': return '\u2005';
        case '!': return '';
        case ' ': case '\\': return ' ';
        case '|': return '‖';
        case '(': case ')': case '[': case ']': return '';
        default: return escC(c);
      }
    }
    var name = /^[A-Za-z]+/.exec(s.slice(p.i))[0];
    p.i += name.length;
    if (s.charAt(p.i) === '*' && (TEXTCMD[name] || name === 'operatorname')) p.i++;
    if (GREEK[name]) { p.greek = true; return GREEK[name]; }
    if (SYM[name] !== undefined) { if (BINREL[name]) p.bin = true; return SYM[name]; }
    if (SPACE[name] !== undefined) return SPACE[name];
    if (FUNCS[name]) return name + (/[A-Za-z\\]/.test(s.charAt(p.i)) ? '\u2009' : '');
    if (name in SKIPARG) { arg(p, true); return SKIPARG[name]; }
    if (DROP[name]) { if (s.charAt(p.i) === '.') p.i++; return ''; }
    if (ACCENT[name]) return accent(arg(p), ACCENT[name], name);
    if (name === 'mathbb' || name === 'mathds' || name === 'mathbbm') return mapChars(arg(p, true), BB);
    if (name === 'mathcal' || name === 'mathscr') return mapChars(arg(p, true), CAL);
    if (TEXTCMD[name]) {
      var tx = arg(p);
      if ((name === 'operatorname' || name === 'mathop' || (name === 'mathrm' && FUNCS[plainOf(tx)])) && /[A-Za-z\\]/.test(s.charAt(p.i))) tx += '\u2009';
      return tx;
    }
    if (BOLD[name]) return '<b>' + arg(p) + '</b>';
    if (ITAL[name]) return '<i>' + arg(p) + '</i>';
    if (MONO[name]) return '<span class="tt">' + arg(p) + '</span>';
    if (name === 'frac' || name === 'dfrac' || name === 'tfrac' || name === 'cfrac') { var a1 = arg(p), b1 = arg(p); return par(a1) + '/' + par(b1); }
    if (name === 'sqrt') {
      var idx = ''; skipSp(p);
      if (s.charAt(p.i) === '[') { var e = s.indexOf(']', p.i); if (e > p.i) { idx = texHTML(s.slice(p.i + 1, e)); p.i = e + 1; } }
      var r = arg(p); return (idx ? '<sup>' + idx + '</sup>' : '') + '√' + par(r);
    }
    if (name === 'binom' || name === 'tbinom' || name === 'dbinom') { var n1 = arg(p), k1 = arg(p); return '(' + n1 + ' choose ' + k1 + ')'; }
    if (name === 'underbrace' || name === 'overbrace') { var u = arg(p); skipSp(p); if (/[_^]/.test(s.charAt(p.i))) { p.i++; arg(p); } return u; }
    if (name === 'underline') return '<u>' + arg(p) + '</u>';
    if (name === 'overset' || name === 'stackrel') { var top = arg(p), base = arg(p); return base + '<sup>' + top + '</sup>'; }
    if (name === 'underset') { var bot = arg(p), base2 = arg(p); return base2 + '<sub>' + bot + '</sub>'; }
    if (name === 'xrightarrow' || name === 'xleftarrow') {
      skipSp(p); if (s.charAt(p.i) === '[') { var e2 = s.indexOf(']', p.i); if (e2 > p.i) p.i = e2 + 1; }
      return (name === 'xrightarrow' ? '→' : '←') + '<sup>' + arg(p) + '</sup>';
    }
    if (name === 'not') { skipSp(p); var nx = s.charAt(p.i) === '\\' ? cmd(p) : escC(s.charAt(p.i++)); return NOT[nx] || nx + '\u0338'; }
    if (name === 'begin' || name === 'end') { arg(p, true); return ' '; }
    return esc(name);
  }

  /* split a TeX formula into display lines at top-level clause separators (\quad, \qquad, ";\ ").
     Only splits where braces, \left…\right, \begin…\end and ( ) [ ] are all balanced, so every
     line is valid TeX on its own and the mathematics is unchanged; only the line breaks move. */
  function splitClauses(tex) {
    var s = String(tex || ''), out = [], depth = 0, env = 0, lr = 0, par0 = 0, last = 0, i = 0, m;
    function cut(at, skipTo) {
      var piece = s.slice(last, at).replace(/^(\s|\\[ ,;:!])+/, '').replace(/(\s|\\[ ,;:!])+$/, '');
      if (piece) out.push(piece);
      last = skipTo;
    }
    while (i < s.length) {
      var c = s.charAt(i);
      if (c === '\\') {
        m = /^\\([A-Za-z]+|.)/.exec(s.slice(i));
        var nm = m ? m[1] : '';
        var top = depth === 0 && env === 0 && lr === 0 && par0 === 0;
        if (nm === 'begin') env++;
        else if (nm === 'end') env = Math.max(0, env - 1);
        else if (nm === 'left') lr++;
        else if (nm === 'right') lr = Math.max(0, lr - 1);
        else if ((nm === 'quad' || nm === 'qquad') && top) { cut(i, i + m[0].length); i += m[0].length; continue; }
        i += m ? m[0].length : 1;
        continue;
      }
      if (c === '{') depth++;
      else if (c === '}') depth = Math.max(0, depth - 1);
      else if (c === '(' || c === '[') { if (depth === 0) par0++; }
      else if (c === ')' || c === ']') { if (depth === 0) par0 = Math.max(0, par0 - 1); }
      else if (c === ';' && depth === 0 && env === 0 && lr === 0 && par0 === 0 && /^;(\\[ ,;:]|\s)/.test(s.slice(i, i + 3))) {
        cut(i + 1, i + 1); i++; continue;
      }
      i++;
    }
    cut(s.length, s.length);
    return out.length ? out : [s];
  }

  /* ---------- prose with explicit math delimiters ----------
     Prose fields carry their mathematics as explicit \( … \) (or \[ … \]) islands. In the card the
     islands stay as TeX for MathJax (A.typeset) and the text between them is plain text. In compact
     places (list rows, tooltips, notes) the islands go through texHTML instead. A field without
     delimiters is plain text; only when it still holds a raw TeX command (\alpha, \operatorname …)
     does the TeX-to-Unicode converter step in, so a reader never sees a backslash. */
  function splitMath(s) {
    var out = [], i = 0, n = s.length, start = 0, open = -1, closer = '';
    while (i < n) {
      if (s.charAt(i) !== '\\' || i + 1 >= n) { i++; continue; }
      var d = s.charAt(i + 1);
      if (open < 0 && (d === '(' || d === '[')) {
        if (i > start) out.push({ t: s.slice(start, i) });
        open = i; start = i; closer = d === '(' ? ')' : ']'; i += 2; continue;
      }
      if (open >= 0 && d === closer) {
        out.push({ m: s.slice(open + 2, i), d: closer === ']' });
        open = -1; i += 2; start = i; continue;
      }
      i += 2; /* an escaped pair (\\, \{, \alpha's backslash …) is never a delimiter */
    }
    if (start < n) out.push({ t: open >= 0 ? s.slice(start).replace(/^\\([(\[])/, '$1') : s.slice(start) }); /* an unclosed \( reads as a bracket */
    return out;
  }
  /* MathJax 3.2 skips elements classed mathjax_ignore and enters mathjax_process ones inside them
     (the tex2jax_* names of MathJax 2 are not recognised). The stage and card are mathjax_ignore;
     every container that holds a math island is mathjax_process. */
  function isMathPart(p) { return p.m !== undefined; }
  function hasMath(s) { return s != null && /\\[(\[]/.test(String(s)) && splitMath(String(s)).some(isMathPart); }
  function looksTeX(s) { return /\\[A-Za-z]/.test(s); }
  /* `$` in plain text must not pair up into $…$ math once MathJax scans the container */
  function plainTxt(t) { return esc(t).replace(/\$/g, '<span>$</span>'); }
  /* card prose: { h: html, mj: true when the container needs MathJax } */
  function prose(s) {
    s = String(s == null ? '' : s);
    if (!hasMath(s)) return { h: looksTeX(s) ? texHTML(s) : esc(s), mj: false };
    return {
      h: splitMath(s).map(function (p) { return isMathPart(p) ? esc(p.d ? '\\[' + p.m + '\\]' : '\\(' + p.m + '\\)') : plainTxt(p.t); }).join(''),
      mj: true
    };
  }
  /* one math island as Unicode HTML; a leading script (LoRA\(_r\)) attaches to the text before it */
  function uniMath(m) {
    var lead = /^\s*[_^]/.test(m), h = texHTML(lead ? 'x' + m.replace(/^\s+/, '') : m);
    return lead ? h.replace(/^x/, '') : h;
  }
  /* compact prose: math islands as Unicode, everything else plain text */
  function proseU(s) {
    s = String(s == null ? '' : s);
    if (!hasMath(s)) return looksTeX(s) ? texHTML(s) : esc(s);
    return splitMath(s).map(function (p) { return isMathPart(p) ? uniMath(p.m) : esc(p.t); }).join('');
  }
  /* names keep the light converter even without delimiters ("Spectral Adapter^R") */
  function nameU(s) { return hasMath(s) ? proseU(s) : texHTML(s); }
  function textOf(html) { var d = document.createElement('div'); d.innerHTML = html; return d.textContent.replace(/\s+/g, ' ').trim(); }
  /* an arrow's map h as one inline TeX formula */
  function hTeX(h) { h = String(h || '').trim(); return /^h(?![A-Za-z])/.test(h) ? h : 'h\\colon ' + h; }

  /* ---------- axis value display ---------- */
  var KEY_DISPLAY = {
    E0: 'E°', F_T: 'F<sub>T</sub>', 'M1->': 'M1→', Minf: 'M∞', GLxGL: 'GL×GL', OxO: 'O×O', GLxCO: 'GL×CO',
    MonxGL: 'Mon×GL', MonxMon: 'Mon×Mon', Pi: 'Π<sub>X</sub>', GLxTorus: 'GL×torus', na: 'n/a',
    span: '→span', group: '→group', Lin: 'Lin', Cone: 'Cone', 'Cone*': 'Cone*', GL: 'GL'
  };
  function keyHTML(k) { return KEY_DISPLAY[k] || esc(String(k)).replace(/-&gt;/g, '→'); }
  /* axes.json and the arrow rules are written in ASCII ("rho:(Q,q0)->(Theta,theta0)"); set them in symbols */
  function prettyHTML(h) {
    return h.replace(/\btheta0\b/g, 'θ<sub>0</sub>').replace(/\btheta_res\b/g, 'θ<sub>res</sub>')
      .replace(/\bPi_X\b/g, 'Π<sub>X</sub>').replace(/\bid_Theta\b/g, 'id<sub>Θ</sub>').replace(/\brho_([A-Z])\b/g, 'ρ<sub>$1</sub>').replace(/ o (?=h\b)/g, ' ∘ ').replace(/\bTheta\b/g, 'Θ').replace(/\btheta\b/g, 'θ').replace(/\brho\b/g, 'ρ').replace(/\beta\b/g, 'η')
      .replace(/\b([A-Z][A-Za-z]*(?:\([a-z]\))?) x (?=[A-Z])/g, '$1 × ')
      .replace(/ -\/-&gt; /g, ' ↛ ').replace(/-&gt;/g, '→').replace(/&lt;-(?!-)/g, '←').replace(/&lt;=/g, '≤').replace(/&gt;=/g, '≥')
      .replace(/\b([qW])0\b/g, '$1<sub>0</sub>').replace(/\bS1\b/g, 'S<sub>1</sub>');
  }
  function prettyAxis(s) { return prettyHTML(esc(s)); }
  /* longer definitions also carry light TeX (GL_r, g^{-1}, Theta^N) */
  function prettyDef(s) {
    return prettyHTML(texHTML(String(s == null ? '' : s).replace(/_infinity\b/g, '_{∞}').replace(/\binfinity\b/g, '∞')));
  }
  var AXIS_SHORT = { kind: 'Kind', shape: 'Shape', base_point: 'Base point', gauge: 'Gauge', covariance: 'Covariance', merge: 'Merge', rebasing: 'Rebasing' };
  var AXIS_ORDER = ['kind', 'shape', 'base_point', 'gauge', 'covariance', 'merge', 'rebasing'];

  var REL_ORDER = ['equivalent-to', 'special-case-of', 'generalizes', 'builds-on', 'dual-to', 'combines', 'competes-with'];
  var REL_OUT = { 'equivalent-to': 'equivalent to', 'special-case-of': 'special case of', generalizes: 'generalises', 'builds-on': 'builds on', 'dual-to': 'dual to', combines: 'combines with', 'competes-with': 'competes with' };
  var REL_IN = { 'equivalent-to': 'equivalent to it', 'special-case-of': 'special cases of it', generalizes: 'generalise it', 'builds-on': 'build on it', 'dual-to': 'dual to it', combines: 'combine with it', 'competes-with': 'compete with it' };
  var MERGE_LABEL = { yes: 'mergeable', conditional: 'conditional', no: 'not mergeable' };
  var MERGE_SHORT = { yes: 'merges', conditional: 'cond.', no: 'no merge' };
  var MERGE_CLASS = { yes: 'yes', conditional: 'cond', no: 'no' };

  /* ---------- deep links & events that arrive before the figure mounts ---------- */
  var live = [];           /* mounted instances */
  var early = null;        /* id from an atlas:select-method fired before mount */
  var deepTarget = null;   /* () => the element a deep link should bring to the top (set on mount) */
  function idOf(d) { if (!d) return null; if (typeof d === 'string') return d; return d.id || d.method || d.methodId || null; }
  function parseHash() {
    var raw = window.location.hash || '';
    try { raw = decodeURIComponent(raw); } catch (e) {}
    var m = /^#([mp])-([A-Za-z0-9._~+-]+)$/.exec(raw);
    return m ? { tab: m[1] === 'm' ? 'methods' : 'papers', id: m[2] } : null;
  }
  function hashKnown(h) {
    var D = window.ATLAS_DATA || {}, coll = h && (h.tab === 'methods' ? D.methods : D.papers);
    return !!(coll && coll.some(function (x) { return x.id === h.id; }));
  }
  document.addEventListener('atlas:select-method', function (e) { if (!live.length) { var id = idOf(e.detail); if (id) early = id; } });
  function figTarget() {
    if (deepTarget) return deepTarget();
    var fig = document.querySelector(SCOPE);
    return fig ? (fig.closest('figure') || fig) : null;
  }
  /* jump without the page's smooth scrolling; scroll-padding-top keeps it clear of the top bar */
  function jumpTo(t) {
    var de = document.documentElement, prev = de.style.scrollBehavior;
    de.style.scrollBehavior = 'auto';
    try { t.scrollIntoView({ block: 'start' }); } catch (e) {}
    de.style.scrollBehavior = prev;
  }
  /* Keep the target in place while lazy figures mount and MathJax re-flows the page above it,
     until the reader scrolls, taps or types, or `ms` pass. */
  var anchorTok = null, lastInput = 0;
  function stopAnchor() { anchorTok = null; lastInput = Date.now(); }
  ['wheel', 'touchstart', 'mousedown', 'keydown'].forEach(function (t) {
    window.addEventListener(t, stopAnchor, { passive: true, capture: true });
  });
  function anchorTo(ms) {
    var tok = {}, until = Date.now() + (ms || 7000), de = document.documentElement;
    anchorTok = tok;
    (function tick() {
      if (anchorTok !== tok) return;
      if (Date.now() > until) { anchorTok = null; return; }
      var t = figTarget();
      if (t) {
        var pad = parseFloat(getComputedStyle(de).scrollPaddingTop) || 0;
        var top = t.getBoundingClientRect().top, canDown = window.scrollY + window.innerHeight < de.scrollHeight - 1;
        if (top < pad - 2 || (top > pad + 2 && canDown)) jumpTo(t);
      }
      setTimeout(tick, 150);
    })();
  }
  function scrollForHash() { if (hashKnown(parseHash())) anchorTo(9000); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', scrollForHash); else setTimeout(scrollForHash, 0);
  /* an in-page link to #m-<id> before the figure has mounted: bring it into view (mounting it) */
  window.addEventListener('hashchange', function () { if (!live.length) scrollForHash(); });

  /* ---------- scoped stylesheet ---------- */
  function injectCSS() {
    if (document.getElementById('css-catalog')) return;
    var css = [
      '& .cat-stage{padding:1.1rem 1.1rem 1rem;--pane-h:clamp(30rem,76vh,44rem)}',
      '& .cat-head{margin-bottom:.95rem}',
      '& .cat-sub{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:.7rem 1.5rem;margin-top:.45rem}',
      '& .cat-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide);margin-bottom:.25rem}',
      '& .cat-title{font-family:var(--f-display);font-size:clamp(1.65rem,3.2vw,2.15rem);font-weight:var(--w-head);letter-spacing:-.01em;line-height:1.05;margin:0;color:var(--ink)}',
      '& .cat-title em{font-style:italic;color:var(--tide)}',
      '& .cat-instr{margin:0;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.45;color:var(--ink-2);font-style:italic;max-width:27rem;flex:1 1 16rem}',
      '& .cat-stats{display:grid;grid-template-columns:repeat(var(--n,4),auto);gap:0;border:1px solid var(--rule);border-radius:var(--radius);background:var(--paper);overflow:hidden}',
      '& .cat-stat{padding:.42rem .75rem .4rem;border-right:1px solid var(--rule);min-width:0}',
      '& .cat-stat:last-child{border-right:0}',
      '& .cat-stat b{display:block;font-family:var(--f-mono);font-weight:500;font-size:1.22rem;line-height:1.15;font-variant-numeric:tabular-nums;color:var(--ink)}',
      '& .cat-stat span{display:block;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-2);white-space:nowrap}',
      /* toolbar */
      '& .cat-tools{display:flex;flex-wrap:wrap;gap:.55rem .7rem;align-items:center;margin-bottom:.75rem}',
      '& .cat-tabs{flex:none}',
      '& .cat-tabs button{display:inline-flex;gap:.4rem;align-items:baseline}',
      '& .cat-tabs button[aria-selected="true"]{background:var(--ink);color:var(--paper)}',
      '& .cat-tabs button .n{font-variant-numeric:tabular-nums;opacity:.72}',
      '& .cat-search{flex:1 1 15rem;min-width:0;position:relative}',
      '& .cat-search svg{position:absolute;left:.62rem;top:50%;transform:translateY(-50%);width:15px;height:15px;stroke:var(--ink-3);fill:none;stroke-width:1.6;pointer-events:none}',
      '& .cat-search input{width:100%;padding:.48rem .6rem .48rem 2rem;font-family:var(--f-body);font-size:1rem;text-overflow:ellipsis;border-radius:999px;background:var(--paper);border:1px solid var(--rule)}',
      '& .cat-search input:focus{border-color:var(--tide);outline:none;box-shadow:0 0 0 3px var(--tide-soft)}',
      '& .cat-sort{display:flex;align-items:center;gap:.4rem;flex:none}',
      '& .cat-sort label{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      '& .cat-sort select{font-family:var(--f-ui);font-weight:500;font-size:.8rem;padding:.36rem .45rem;border-radius:4px}',
      /* kind strip (legend + filter) */
      '& .label{letter-spacing:.06em;color:var(--ink-2)}',
      '& .cat-strip{margin-bottom:.9rem}',
      '& .cat-strip-h{display:flex;justify-content:space-between;gap:.6rem;align-items:baseline;margin-bottom:.35rem}',
      '& .cat-bar{display:flex;height:9px;border-radius:2px;overflow:hidden;background:var(--paper-3);margin-bottom:.5rem;gap:1px}',
      '& .cat-bar i{display:block;height:100%;background:var(--k);transition:opacity .15s,flex-grow .25s}',
      '& .cat-bar i.off{opacity:.18}',
      '& .cat-kinds{display:flex;flex-wrap:wrap;gap:.3rem .35rem}',
      '& .cat-fchip{display:inline-flex;align-items:center;gap:.38rem;font-family:var(--f-ui);font-weight:500;font-size:.78rem;letter-spacing:.01em;line-height:1.2;padding:.26rem .55rem .26rem .45rem;border-radius:999px;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);cursor:pointer;white-space:nowrap}',
      '& .cat-fchip:hover{border-color:var(--ink-3);color:var(--ink)}',
      '& .cat-fchip .sw{width:.62rem;height:.62rem;border-radius:2px;background:var(--k);flex:none}',
      '& .cat-fchip .dot{width:.5rem;height:.5rem;border-radius:50%;background:var(--k);flex:none}',
      '& .cat-fchip .n{font-family:var(--f-mono);font-weight:400;font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      '& .cat-fchip[aria-pressed="true"]{border-color:var(--k,var(--tide));color:var(--ink);background:color-mix(in srgb,var(--k,var(--tide)) 14%,var(--paper));box-shadow:inset 0 0 0 1px var(--k,var(--tide))}',
      '& .cat-fchip[aria-pressed="true"] .n{color:var(--ink-2)}',
      '& .cat-fchip.zero{opacity:.45}',
      '& .cat-fchip sub,& .cat-fchip sup{font-size:.94em;line-height:0}',
      /* body grid */
      '& .cat-body{display:grid;gap:.85rem;grid-template-columns:minmax(0,1fr)}',
      '& .cat-stage[data-layout="wide"] .cat-body{grid-template-columns:12.2rem minmax(0,.92fr) minmax(0,1.32fr);align-items:start}',
      '& .cat-stage[data-layout="mid"] .cat-body{grid-template-columns:minmax(0,.9fr) minmax(0,1.2fr);align-items:start}',
      '& .cat-stage[data-layout="mid"] .cat-rail{grid-column:1/-1}',
      /* rail */
      '& .cat-rail{min-width:0}',
      '& .cat-rail>summary{list-style:none;cursor:pointer;display:flex;justify-content:space-between;align-items:center;gap:.6rem;padding:.5rem .75rem;border:1px solid var(--rule);border-radius:var(--radius);background:var(--paper);font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      '& .cat-rail>summary::-webkit-details-marker{display:none}',
      '& .cat-rail>summary .chev{transition:transform .15s;display:inline-block}',
      '& .cat-rail[open]>summary .chev{transform:rotate(180deg)}',
      '& .cat-rail[open]>summary{margin-bottom:.7rem}',
      '& .cat-stage[data-layout="wide"] .cat-rail>summary{display:none}',
      '& .cat-stage[data-layout="wide"] .cat-rail-in{height:var(--pane-h);overflow:auto;padding-right:.35rem;scrollbar-width:thin;scrollbar-color:var(--rule) transparent}',
      '& .cat-stage[data-layout="mid"] .cat-rail-in{display:grid;grid-template-columns:repeat(auto-fit,minmax(14rem,1fr));gap:0 1.2rem}',
      '& .cat-grp{padding:0 0 .85rem;margin:0 0 .85rem;border-bottom:1px solid var(--rule)}',
      '& .cat-grp:last-child{border-bottom:0;margin-bottom:0}',
      '& .cat-grp-h{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;margin-bottom:.45rem}',
      '& .cat-grp-h .label{color:var(--ink-2)}',
      '& .cat-grp .cat-kinds{gap:.28rem}',
      '& .cat-seg{display:flex;border:1px solid var(--rule);border-radius:6px;overflow:hidden;background:var(--paper)}',
      '& .cat-seg button{flex:1 1 0;min-width:0;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:0;border:0;border-right:1px solid var(--rule);background:transparent;color:var(--ink-2);padding:.36rem .3rem;cursor:pointer;line-height:1.25}',
      '& .cat-seg button:last-child{border-right:0}',
      '& .cat-seg button .n{display:block;font-family:var(--f-mono);font-weight:400;font-variant-numeric:tabular-nums;color:var(--ink-2);font-size:.75rem}',
      '& .cat-seg button[aria-pressed="true"]{background:var(--tide);color:var(--paper)}',
      '& .cat-seg button[aria-pressed="true"] .n{color:var(--paper);opacity:.9}',
      '& .cat-merge{display:grid;gap:.28rem}',
      '& .cat-merge .cat-fchip{justify-content:flex-start}',
      '& .cat-merge .cat-fchip .n{margin-left:auto}',
      '& .cat-hist svg{display:block;width:100%;height:46px}',
      '& .cat-hist .hb{fill:var(--paper-3)}',
      '& .cat-hist .hf{fill:var(--tide)}',
      '& .cat-hist .hf.out{fill:var(--ink-3);opacity:.35}',
      '& .cat-hist .hit{fill:transparent;cursor:pointer}',
      '& .cat-hist .hit:hover{fill:var(--tide-soft)}',
      '& .cat-hist-ax{display:flex;justify-content:space-between;font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums;margin:.15rem 0 .35rem;border-top:1px solid var(--rule);padding-top:.15rem}',
      '& .cat-hist-ax b{font-weight:500;color:var(--ink)}',
      '& .cat-yr{display:grid;grid-template-columns:1fr 1fr;gap:.2rem .7rem}',
      '& .cat-yr label{display:flex;justify-content:space-between;align-items:baseline;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      '& .cat-yr output{font-family:var(--f-mono);font-weight:400;letter-spacing:0;color:var(--ink);font-variant-numeric:tabular-nums}',
      '& .cat-yr input{margin:0}',
      '& .cat-axis{margin:0 0 .35rem}',
      '& .cat-axis>summary{cursor:pointer;list-style:none;font-family:var(--f-ui);font-weight:500;font-size:.8rem;letter-spacing:.01em;color:var(--ink-2);padding:.18rem 0;display:flex;align-items:baseline;gap:.4rem}',
      '& .cat-axis>summary::-webkit-details-marker{display:none}',
      '& .cat-axis>summary::before{content:"\\25B8";color:var(--ink-2);font-size:.75rem;transition:transform .15s;display:inline-block}',
      '& .cat-axis[open]>summary::before{transform:rotate(90deg)}',
      '& .cat-axis>summary .n{margin-left:auto}',
      '& .cat-axis>summary:hover{color:var(--ink)}',
      '& .cat-axis>summary .n{font-family:var(--f-mono);font-weight:400;font-variant-numeric:tabular-nums;color:var(--tide)}',
      '& .cat-axis[open]>summary{margin-bottom:.3rem}',
      '& .cat-axvals{display:grid;gap:.22rem;padding-left:.1rem}',
      '& .cat-fchip.cat-axv{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:.3rem;border-radius:5px;text-align:left;white-space:normal;padding:.24rem .42rem}',
      '& .cat-fchip.cat-axv{align-items:baseline}',
      '& .cat-axv .k{font-family:var(--f-mono);font-weight:400;font-size:.8125rem;color:var(--ink);min-width:1.3rem}',
      '& .cat-axv .nmv{font-family:var(--f-body);font-weight:400;font-size:.8rem;hyphens:auto;-webkit-hyphens:auto;overflow-wrap:anywhere;line-height:1.2;letter-spacing:0;white-space:normal;color:var(--ink-2)}',
      '& .cat-axv sub{font-size:.94em;line-height:0}',
      '& .cat-pending{border:1px dashed var(--rule);border-radius:var(--radius);padding:.55rem .65rem;font-family:var(--f-body);font-size:.8rem;line-height:1.45;color:var(--ink-2)}',
      '& .cat-pending ol{margin:.35rem 0 0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:.2rem}',
      '& .cat-pending li{font-family:var(--f-ui);font-weight:500;font-size:.75rem;padding:.06rem .42rem;border:1px solid var(--rule);border-radius:999px;color:var(--ink-2)}',
      '& .cat-note{font-family:var(--f-body);font-size:.8rem;line-height:1.45;color:var(--ink-2);font-style:italic}',
      /* list pane */
      '& .cat-listpane{display:flex;flex-direction:column;min-width:0;border:1px solid var(--rule);border-radius:var(--radius);background:var(--paper);overflow:hidden}',
      '& .cat-stage[data-layout="wide"] .cat-listpane,& .cat-stage[data-layout="mid"] .cat-listpane{height:var(--pane-h)}',
      '& .cat-lhead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:.35rem .6rem;padding:.5rem .65rem;border-bottom:1px solid var(--rule);background:var(--paper)}',
      '& .cat-count{font-family:var(--f-ui);font-size:.8rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      '& .cat-count b{font-family:var(--f-mono);color:var(--ink);font-weight:500}',
      '& .cat-lhead .btn{font-size:.75rem;letter-spacing:.03em;padding:.26rem .55rem;white-space:nowrap}',
      '& .cat-lhead .btn:disabled{opacity:.45;cursor:default}',
      '& .cat-lbtns{display:flex;gap:.35rem;flex-wrap:wrap}',
      '& .cat-hidden{padding:.4rem .65rem;font-family:var(--f-body);font-size:.8rem;color:var(--ink-2);background:var(--ochre-soft);border-bottom:1px solid var(--rule)}',
      '& .cat-hidden button,& .cat-empty button{font:inherit;color:var(--tide);background:none;border:0;padding:0;text-decoration:underline;cursor:pointer}',
      '& .cat-scroll{position:relative;flex:1 1 auto;min-height:0;overflow:auto;scrollbar-width:thin;scrollbar-color:var(--rule) transparent}',
      '& .cat-stage[data-layout="narrow"] .cat-scroll{max-height:23.5rem}',
      '& .cat-list:focus{outline:none}',
      '& .cat-yh{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:.5rem;padding:.24rem .65rem;font-family:var(--f-mono);font-size:.75rem;letter-spacing:.02em;font-variant-numeric:tabular-nums;color:var(--ink-2);background:var(--paper-2);border-bottom:1px solid var(--rule)}',
      '& .cat-yh b{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums}',
      '& .cat-yh i{flex:1;height:1px;background:var(--rule)}',
      '& .cat-row{display:grid;grid-template-columns:.62rem minmax(0,1fr) auto;column-gap:.5rem;row-gap:.08rem;align-items:center;line-height:1.25;padding:.4rem .65rem .42rem .55rem;border-bottom:1px solid var(--rule);cursor:pointer;position:relative;outline:none}',
      '& .cat-row:hover{background:var(--tide-soft)}',
      '& .cat-row[aria-selected="true"]{background:color-mix(in srgb,var(--tide) 9%,var(--paper));box-shadow:inset 3px 0 0 var(--tide)}',
      '& .cat-row:focus-visible{box-shadow:inset 0 0 0 2px var(--ochre)}',
      '& .cat-row .sw{width:.62rem;height:.62rem;border-radius:2px;background:var(--k);align-self:center}',
      '& .cat-row .nm{font-family:var(--f-display);font-size:1.03rem;line-height:1.25;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}',
      '& .cat-row .nm sub,& .cat-row .nm sup{font-size:.76em;line-height:0}',
      '& .cat-row .rt{display:flex;gap:.3rem;align-items:center;justify-self:end}',
      '& .cat-row .yr{font-family:var(--f-mono);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      '& .cat-row .ln2{grid-column:2/4;display:flex;gap:.45rem;align-items:center;min-width:0}',
      '& .cat-row .vn{flex:none;max-width:44%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--f-ui);font-size:.75rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      '& .cat-row .fn{flex:1 1 auto;min-width:0;font-family:var(--f-body);font-style:italic;color:var(--ink-2);font-size:.82rem;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '& .cat-row .ch{flex:none}',
      '& .cat-row .fn sub,& .cat-row .fn sup{font-size:.94em;line-height:0}',
      '& .cat-row .chip{font-size:.75rem;line-height:1.3;padding:.02rem .38rem;letter-spacing:.01em}',
      '& .cat-row .ch .chip{border-color:transparent;background:none;padding:0;gap:.28rem}',
      '& .cat-row .ch .chip.yes::before,& .cat-row .ch .chip.cond::before,& .cat-row .ch .chip.no::before{content:"";width:.42rem;height:.42rem;border-radius:50%;background:currentColor}',
      '& .cat-row .ch .chip.cnt{color:var(--ink-2)}',
      '& .cat-row .ch .chip.cond::before{background:none;box-shadow:inset 0 0 0 1.5px currentColor}',
      '& .cat-row .ch .chip.no::before{background:none;box-shadow:inset 0 0 0 1.5px currentColor}',
      '& .cat-row.paper{grid-template-columns:minmax(0,1fr) auto}',
      '& .cat-row.paper .nm{white-space:normal;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;font-size:.98rem;line-height:1.25}',
      '& .cat-row.paper .ln2{grid-column:1/3}',
      '& .cat-row mark,& .cat-card mark{background:var(--ochre-soft);color:inherit;border-radius:2px;padding:0 .05em;box-shadow:0 1px 0 var(--ochre)}',
      '& .cat-empty{padding:1.4rem .9rem;text-align:center;color:var(--ink-2);font-size:var(--fs-sm)}',
      /* card pane */
      '& .cat-cardpane{min-width:0;border:1px solid var(--rule);border-radius:var(--radius);background:var(--paper);scrollbar-width:thin;scrollbar-color:var(--rule) transparent}',
      '& .cat-stage[data-layout="wide"] .cat-cardpane,& .cat-stage[data-layout="mid"] .cat-cardpane{height:var(--pane-h);overflow:auto}',
      '& .cat-card{padding:1rem 1.1rem 1.15rem;gap:.7rem}',
      '& .cat-card .cc-top{display:flex;flex-wrap:wrap;gap:.35rem .45rem;align-items:center}',
      '& .cat-card .cc-kind{display:inline-flex;align-items:center;gap:.4rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--k)}',
      '& .cat-card .cc-kind .sw{width:.7rem;height:.7rem;border-radius:2px;background:var(--k)}',
      '& .cat-card .cc-year{margin-left:auto;font-family:var(--f-mono);font-size:1.05rem;color:var(--ink-2);font-variant-numeric:tabular-nums;letter-spacing:.02em}',
      '& .cat-card h4{font-size:clamp(1.55rem,3vw,1.95rem);line-height:1.08;letter-spacing:-.005em;margin-top:.05rem}',
      '& .cat-card h4 sub,& .cat-card h4 sup{font-size:.6em}',
      '& .cat-card .full{font-size:var(--fs-md);line-height:1.35;margin-top:-.25rem}',
      '& .cat-card .cc-paper{font-size:.85rem;color:var(--ink-2)}',
      '& .cat-card .cc-auth{font-size:.88rem;line-height:1.45;color:var(--ink)}',
      '& .cat-card .cc-auth button{font:inherit;font-size:.8rem;color:var(--tide);background:none;border:0;padding:0 .1rem;cursor:pointer;text-decoration:underline;text-underline-offset:.15em}',
      '& .cat-card .cc-meta{display:flex;flex-wrap:wrap;gap:.3rem .5rem;align-items:center;font-family:var(--f-ui);font-size:.8rem;color:var(--ink-2);font-variant-numeric:tabular-nums}',
      '& .cat-card .cc-venue{color:var(--ink);font-weight:500}',
      '& .cat-card .cc-meta a{display:inline-flex;align-items:center;gap:.2rem;text-decoration:none;border:1px solid var(--rule);border-radius:4px;padding:.12rem .42rem;color:var(--tide);overflow-wrap:anywhere}',
      '& .cat-card .cc-meta a:hover{border-color:var(--tide)}',
      '& .cat-card .cc-formula{font-size:1.04rem;background:var(--paper-2);border:1px solid var(--rule);border-radius:var(--radius);padding:.35rem .75rem;overflow-x:auto;color:var(--ink)}',
      '& .cat-card .cc-formula .ln{overflow-x:auto;overflow-y:hidden;padding:.12rem 0;line-height:1.5;white-space:nowrap}',
      '& .cat-card .cc-formula .ln+.ln{border-top:1px dotted var(--rule)}',
      '& .cat-card .cc-formula mjx-container[display="true"]{text-align:left!important;margin:.15em 0!important;overflow:visible}',
      '& .cat-card .cc-math{overflow-x:auto;overflow-y:hidden;white-space:nowrap;padding:.05rem 0}',
      '& .cat-card .cc-math+.cc-math{margin-top:.1rem}',
      '& .cat-card .cc-formula svg,& .cat-card .cc-math svg{max-width:none}',
      '& .cat-card .ovf-r{-webkit-mask-image:linear-gradient(to right,currentColor 84%,transparent);mask-image:linear-gradient(to right,currentColor 84%,transparent)}',
      '& .cat-card .ovf-l{-webkit-mask-image:linear-gradient(to left,currentColor 84%,transparent);mask-image:linear-gradient(to left,currentColor 84%,transparent)}',
      '& .cat-card .ovf-lr{-webkit-mask-image:linear-gradient(to right,transparent,currentColor 12%,currentColor 88%,transparent);mask-image:linear-gradient(to right,transparent,currentColor 12%,currentColor 88%,transparent)}',
      '& .cat-card .cc-raw{font-family:var(--f-mono);font-size:.8rem;white-space:pre-wrap;overflow-wrap:anywhere}',
      '& .cat-card dl{gap:.42rem .9rem}',
      '& .cat-card dt{font-weight:500;color:var(--ink-2)}',
      '& .cat-card dd{line-height:1.5}',
      '& .cat-card dd sub,& .cat-card dd sup,& .cat-card .cc-sec sub,& .cat-card .cc-sec sup{font-size:.86em;line-height:0}',
      '& .cat-card .ov{text-decoration:overline}',
      '& .cat-card .tt{font-family:var(--f-mono);font-size:.92em}',
      '& .cat-card dd code{display:block;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.78rem;line-height:1.45;padding:.3rem .45rem;margin-top:.1rem}',
      '& .cat-card dd .chip{margin-right:.35rem;vertical-align:.08em}',
      '& .cat-stage[data-layout="narrow"] .cat-card dl{grid-template-columns:minmax(0,1fr);gap:.1rem}',
      '& .cat-stage[data-layout="narrow"] .cat-card dd{margin-bottom:.45rem}',
      '& .cat-card .cc-sec{border-top:1px solid var(--rule);padding-top:.7rem;display:grid;gap:.45rem;min-width:0}',
      '& .cat-card .cc-sec-h{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.15rem .5rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--tide)}',
      '& .cat-card .cc-sec-h small{font-weight:400;font-size:.75rem;letter-spacing:0;text-transform:none;color:var(--ink-2)}',
      '& .cat-card .cc-sec-h>span{white-space:nowrap}',
      '& .cat-card .cc-coords{display:grid;grid-template-columns:repeat(auto-fill,minmax(8.4rem,1fr));gap:.35rem}',
      '& .cat-card .cc-co{border:1px solid var(--rule);border-radius:5px;padding:.32rem .5rem .36rem;background:var(--paper-2);min-width:0}',
      '& .cat-card .cc-co .ax{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2)}',
      '& .cat-card .cc-co .v{font-family:var(--f-mono);font-size:.92rem;color:var(--ink);line-height:1.25}',
      '& .cat-card .cc-co .vn{font-family:var(--f-body);font-size:.8rem;line-height:1.3;color:var(--ink-2)}',
      '& .cat-card .cc-co.pend{background:transparent;border-style:dashed}',
      '& .cat-card .cc-co.pend .v{color:var(--ink-2)}',
      '& .cat-card .cc-co.has-note{cursor:help}',
      '& .cat-card .cc-co.has-note .ax::after{content:" ·";color:var(--tide)}',
      '& .cat-card .cc-pendnote{font-size:.8rem;color:var(--ink-2);font-style:italic}',
      '& .cat-card .cc-read{border-left:3px solid var(--tide);padding:.15rem 0 .15rem .75rem;font-size:.9rem;line-height:1.5}',
      '& .cat-card .cc-rg{display:flex;flex-wrap:wrap;gap:.28rem .32rem;align-items:baseline}',
      '& .cat-card .cc-rt{font-family:var(--f-body);font-style:italic;color:var(--ink-2);font-size:.82rem;margin-right:.15rem;min-width:6.6rem}',
      '& .cat-card .cc-link{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--f-body);font-size:.84rem;line-height:1.2;padding:.12rem .45rem;border-radius:999px;border:1px solid var(--rule);color:var(--ink);text-decoration:none;background:var(--paper)}',
      '& .cat-card a.cc-link:hover,& .cat-card a.cc-link:focus-visible{border-color:var(--tide);color:var(--tide)}',
      '& .cat-card .cc-link .sw{width:.5rem;height:.5rem;border-radius:2px;background:var(--k)}',
      '& .cat-card .cc-link.missing{color:var(--ink-2);border-style:dashed}',
      '& .cat-card .cc-link.paper{border-radius:4px;font-size:.8rem}',
      '& .cat-card details.cc-in>summary{cursor:pointer;font-size:.84rem;color:var(--tide);padding:.15rem 0}',
      '& .cat-card details.cc-in>summary:hover{text-decoration:underline}',
      '& .cat-card details.cc-in[open]>summary{margin-bottom:.4rem}',
      '& .cat-card details.cc-in .cc-rg+.cc-rg{margin-top:.3rem}',
      '& .cat-card .cc-arrow{display:grid;grid-template-columns:1rem auto minmax(0,1fr);gap:.18rem .45rem;align-items:center;font-size:.84rem;padding:.42rem 0;border-top:1px dotted var(--rule)}',
      '& .cat-card .cc-agh+.cc-arrow,& .cat-card .cc-lazy>.cc-arrow:first-child{border-top:0;padding-top:.1rem}',
      '& .cat-card .cc-arrow .ar{font-family:var(--f-mono);color:var(--ink-2);text-align:center}',
      '& .cat-card .cc-arrow .cc-link{justify-self:start;max-width:100%}',
      '& .cat-card .cc-arrow .cc-h{grid-column:2/-1;font-size:.95rem;color:var(--ink)}',
      '& .cat-card .cc-arrow .cc-an{grid-column:2/-1;color:var(--ink-2);font-size:.84rem;line-height:1.5;min-width:0;overflow-wrap:anywhere}',
      '& .cat-card .cc-arrow.ob .ar{color:var(--seal)}',
      '& .cat-card .cc-ag{display:grid;min-width:0}',
      '& .cat-card .cc-agh{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2);margin:.25rem 0 .15rem}',
      '& .cat-card .cc-agh b{font-weight:600;color:var(--ink);text-transform:none;letter-spacing:0}',
      '& .cat-card .cc-lede{font-size:.84rem;line-height:1.5;color:var(--ink-2)}',
      '& .cat-card details.cc-more>summary{cursor:pointer;font-size:.84rem;color:var(--tide);padding:.15rem 0}',
      '& .cat-card details.cc-more>summary:hover{text-decoration:underline}',
      '& .cat-card details.cc-more[open]>summary{margin-bottom:.3rem}',
      '& .cat-card .cc-cn{display:grid;grid-template-columns:minmax(0,1fr);gap:.1rem;margin:0}',
      '& .cat-card .cc-cn dt{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2);margin-top:.35rem}',
      '& .cat-card .cc-cn dd{margin:0;font-size:.86rem;line-height:1.5;color:var(--ink);min-width:0;overflow-wrap:anywhere}',
      /* MathJax's hidden assistive MathML is as wide as the formula; keep it inside its container */
      '& mjx-assistive-mml{right:0!important;max-width:100%!important}',
      '& .cat-card mjx-container.cc-wide{display:block!important;max-width:100%;overflow-x:auto;overflow-y:hidden;padding:.1rem 0}',
      '& .cat-card .chip.rule{cursor:help;color:var(--tide);border-color:currentColor}',
      '& .cat-card .chip.rule.ob{color:var(--seal)}',
      '& .cat-card .chip.core{color:var(--tide);border-color:currentColor;background:var(--tide-soft)}',
      '& .cat-card .chip.ok{color:var(--moss-ink);border-color:currentColor}',
      '& .chip{font-size:.75rem;letter-spacing:.02em}',
      '& .cat-row .chip{font-size:.75rem;letter-spacing:.01em}',
      '& .cat-card .cc-cite-h{align-items:center}',
      '& .cat-card .cc-cite-h small{margin-left:.45rem}',
      '& .cat-card .cc-cite-h .btn{font-size:.75rem;letter-spacing:.04em;white-space:nowrap;padding:.28rem .55rem;color:var(--ink);text-transform:uppercase}',
      '& .cat-card .cc-bib{margin:0;max-height:15rem;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.78rem;line-height:1.5;color:var(--ink)}',
      '& .cat-card .cc-foot{font-family:var(--f-ui);font-size:.75rem;color:var(--ink-2);display:flex;flex-wrap:wrap;gap:.3rem .9rem;justify-content:space-between}',
      '& .cat-card .cc-foot code{background:none;padding:0;font-size:1em;color:var(--ink)}',
      '& .cat-fallback{margin:0 0 .8rem;border:1px solid var(--ochre);border-radius:var(--radius);background:var(--paper);padding:.6rem .7rem;display:grid;gap:.45rem}',
      '& .cat-fallback[hidden]{display:none}',
      '& .cat-fallback .fb-h{display:flex;justify-content:space-between;align-items:center;gap:.6rem;font-family:var(--f-body);font-size:.84rem;color:var(--ink-2)}',
      '& .cat-fallback textarea{width:100%;min-height:9rem;font-family:var(--f-mono);font-size:.78rem;line-height:1.45;color:var(--ink);background:var(--paper-2);border:1px solid var(--rule);border-radius:4px;padding:.45rem;resize:vertical}',
      '& .cat-legend-note{font-family:var(--f-ui);font-size:.75rem;letter-spacing:.01em;color:var(--ink-2)}',
      '@media (max-width:520px){& .cat-stats{grid-template-columns:repeat(2,minmax(0,1fr));width:100%}& .cat-stat:nth-child(2n){border-right:0}& .cat-stat:nth-child(-n+2){border-bottom:1px solid var(--rule)}& .cat-stat:nth-child(5){grid-column:1/-1;border-top:1px solid var(--rule)}& .cat-stage{padding:.9rem .8rem .85rem}& .cat-card{padding:.85rem .8rem 1rem}& .cat-search input{padding-right:.5rem}}'
    ].join('\n').replace(/&/g, SCOPE);
    var st = document.createElement('style');
    st.id = 'css-catalog';
    st.textContent = css;
    document.head.appendChild(st);
  }

  /* ---------- search normalisation ---------- */
  function norm(s) {
    s = String(s == null ? '' : s);
    try { s = s.normalize('NFD').replace(/[\u0300-\u036f]/g, ''); } catch (e) {}
    return s.toLowerCase().replace(/\\[a-z]+/g, ' ').replace(/[{}^$\\]/g, '');
  }
  function compact(s) { return norm(s).replace(/[^a-z0-9]/g, ''); }
  function hostOf(u) { var m = /^https?:\/\/(?:www\.)?([^\/?#]+)(\/[^?#]*)?/.exec(u || ''); if (!m) return u || ''; var path = (m[2] || '').replace(/\/$/, ''); var parts = path.split('/').filter(Boolean).slice(0, 2); return m[1] + (parts.length ? '/' + parts.join('/') : ''); }
  function nkeys(o) { for (var k in o) if (o[k]) return true; return false; }
  function countKeys(o) { var n = 0; for (var k in o) if (o[k]) n++; return n; }

  /* =====================================================================
     2. The figure
     ===================================================================== */
  Atlas.register(FIG, function (el, A) {
    injectCSS();
    var D = A.data();
    var methods = D.methods || [], papers = D.papers || [], axes = D.axes || [], rules = D.arrow_rules || [];
    var byId = {}, pById = {};
    methods.forEach(function (m) { byId[m.id] = m; });
    papers.forEach(function (p) { pById[p.id] = p; });
    var axisByKey = {};
    axes.forEach(function (a) { axisByKey[a.key] = a; });
    var axisKeys = AXIS_ORDER.filter(function (k) { return axisByKey[k]; }).concat(axes.map(function (a) { return a.key; }).filter(function (k) { return AXIS_ORDER.indexOf(k) < 0; }));
    var ruleByKey = {};
    rules.forEach(function (r) { ruleByKey[r.key] = r; });

    function kindTok(kind) { for (var i = 0; i < A.KINDS.length; i++) if (A.KINDS[i].key === kind) return A.KINDS[i].token; return '--c-hybrid'; }
    function coordVals(m, ax) {
      var c = m.coords && m.coords[ax];
      if (c == null || c === '') return [];
      if (Array.isArray(c)) return c.filter(function (x) { return x != null && x !== ''; }).map(function (x) { return typeof x === 'object' ? String(x.value || x.key || '') : String(x); });
      if (typeof c === 'object') return [String(c.value || c.key || '')].filter(Boolean);
      return [String(c)];
    }
    function hasCoords(m) { return !!(m.coords && axisKeys.some(function (k) { return coordVals(m, k).length; })); }
    function axisValue(ax, key) {
      var a = axisByKey[ax]; if (!a) return null;
      for (var i = 0; i < (a.values || []).length; i++) if (a.values[i].key === key || a.values[i].name === key) return a.values[i];
      return null;
    }

    /* ---------- derived indexes (computed once) ---------- */
    var nClassified = methods.filter(hasCoords).length;
    var anyCore = methods.some(function (m) { return m.core === true; });
    var axisExtra = {}; /* values found in data but not listed in axes[] */
    axisKeys.forEach(function (ax) {
      var seen = {};
      methods.forEach(function (m) { coordVals(m, ax).forEach(function (v) { if (!axisValue(ax, v)) seen[v] = 1; }); });
      axisExtra[ax] = Object.keys(seen).sort();
    });
    var inRel = {};
    methods.forEach(function (m) {
      (m.relations || []).forEach(function (r) { if (!r || !r.target) return; (inRel[r.target] = inRel[r.target] || []).push({ source: m.id, type: r.type, note: r.note }); });
    });
    /* arrows: union of top-level edges and per-method arrows, de-duplicated */
    var arrowsOut = {}, arrowsIn = {}, seenArrow = {};
    function addArrow(a) {
      if (!a || !a.source || !a.target) return;
      var k = a.source + '>' + a.target + '>' + (a.rule || '');
      if (seenArrow[k]) return; seenArrow[k] = 1;
      (arrowsOut[a.source] = arrowsOut[a.source] || []).push(a);
      (arrowsIn[a.target] = arrowsIn[a.target] || []).push(a);
    }
    methods.forEach(function (m) { (m.arrows || []).forEach(function (a) { addArrow({ source: m.id, target: a.target, rule: a.rule, h: a.h, note: a.note }); }); });
    (D.edges || []).forEach(addArrow);
    var papersFor = {};
    papers.forEach(function (p) { (p.method_ids || []).forEach(function (id) { (papersFor[id] = papersFor[id] || []).push(p.id); }); });

    var items = methods.map(function (m) {
      return {
        m: m, id: m.id, year: +m.year,
        nName: norm(m.name), cName: compact(m.name), nId: norm(m.id), nFull: norm(m.full_name),
        nAuth: norm((m.authors || '') + ' ' + (m.authors_full || []).join(' ')), nTitle: norm(m.paper_title),
        nIdea: norm(m.key_idea), nStruct: norm(m.structure), score: 0, row: null
      };
    });
    var pitems = papers.map(function (p) {
      return {
        m: p, id: p.id, year: +p.year, paper: true,
        nName: norm(p.title), cName: compact(p.title), nId: norm(p.id), nFull: norm(p.venue),
        nAuth: norm((p.authors || '') + ' ' + (p.authors_full || []).join(' ')), nTitle: '',
        nIdea: norm(p.claim), nStruct: norm(p.relevance), score: 0, row: null
      };
    });
    function yearsOf(arr) {
      var lo = Infinity, hi = -Infinity;
      arr.forEach(function (it) { if (isFinite(it.year)) { lo = Math.min(lo, it.year); hi = Math.max(hi, it.year); } });
      if (!isFinite(lo)) { lo = hi = new Date().getFullYear(); }
      return [lo, hi];
    }
    var MY = yearsOf(items), PY = yearsOf(pitems);

    /* ---------- state ---------- */
    var S = {
      tab: 'methods', q: '', toks: [], sort: 'year-asc', sortAuto: false, sortTouched: false,
      kinds: {}, hf: 'any', merge: {}, coords: {}, core: false,
      yM: MY.slice(), yP: PY.slice(),
      selM: byId.lora ? 'lora' : (methods[0] && methods[0].id), selP: papers[0] && papers[0].id,
      authorsOpen: false, notesOpen: false, inOpen: false
    };
    axisKeys.forEach(function (k) { S.coords[k] = {}; });
    var visible = [];
    var layout = null;

    /* ---------- DOM skeleton ---------- */
    var h = A.h;
    var stage = h('div', { class: 'stage cat-stage mathjax_ignore', role: 'group', 'aria-label': 'The Catalogue: searchable explorer of every PEFT method and paper' });
    el.appendChild(stage);

    var nHF = methods.filter(function (m) { return m.hf_peft; }).length;
    var nMergeYes = methods.filter(function (m) { return m.mergeable === 'yes'; }).length;
    var stats = [[methods.length, 'methods'], [papers.length, 'papers'], [nHF, 'in HF PEFT'], [nMergeYes, 'mergeable']];
    /* "classified" only says something while classification is incomplete */
    if (nClassified > 0 && nClassified < methods.length) stats.push([nClassified, 'classified']);
    var head = h('div', { class: 'cat-head' }, [
      h('div', { class: 'cat-kicker', text: 'Catalogue · ' + (MY[0] === MY[1] ? MY[0] : MY[0] + '–' + MY[1]) }),
      h('h3', { class: 'cat-title', html: 'Every method, <em>searchable</em> and citable' }),
      h('div', { class: 'cat-sub' }, [
        h('p', { class: 'cat-instr', text: 'Search, or tap a colour to filter by kind. Open a method to read its formula, coordinates, arrows and BibTeX; the address bar then links to it.' }),
        h('div', { class: 'cat-stats', style: '--n:' + stats.length, role: 'list', 'aria-label': 'Catalogue totals' }, stats.map(function (s) {
          return h('div', { class: 'cat-stat', role: 'listitem' }, [h('b', { text: String(s[0]) }), h('span', { text: s[1] })]);
        }))
      ])
    ]);
    stage.appendChild(head);

    /* toolbar: tabs · search · sort */
    var tabs = h('div', { class: 'seg cat-tabs', role: 'tablist', 'aria-label': 'Collection' });
    var tabBtns = {};
    [['methods', 'Methods', methods.length], ['papers', 'Papers', papers.length]].forEach(function (t) {
      var b = h('button', { type: 'button', role: 'tab', 'aria-selected': String(t[0] === S.tab), tabindex: t[0] === S.tab ? '0' : '-1', 'aria-controls': 'cat-list-' + FIG }, [h('span', { text: t[1] }), h('span', { class: 'n', text: String(t[2]) })]);
      b.addEventListener('click', function () { setTab(t[0]); });
      tabs.appendChild(b); tabBtns[t[0]] = b;
    });
    tabs.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight' && e.key !== 'Home' && e.key !== 'End') return;
      e.preventDefault();
      var nt = (e.key === 'ArrowLeft' || e.key === 'Home') ? 'methods' : 'papers';
      setTab(nt); tabBtns[nt].focus();
    });
    var searchIn = h('input', { type: 'search', id: 'cat-q', autocomplete: 'off', spellcheck: 'false', 'aria-label': 'Search the catalogue' });
    var searchWrap = h('div', { class: 'cat-search', role: 'search' }, [searchIn]);
    searchWrap.insertAdjacentHTML('afterbegin', '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="6.8" cy="6.8" r="4.6"/><path d="M10.3 10.3 14 14" stroke-linecap="round"/></svg>');
    var sortSel = h('select', { id: 'cat-sort' });
    [['year-asc', 'Oldest first'], ['year-desc', 'Newest first'], ['az', 'A–Z'], ['match', 'Best match']].forEach(function (o) { sortSel.appendChild(h('option', { value: o[0], text: o[1] })); });
    sortSel.value = S.sort;
    var tools = h('div', { class: 'cat-tools' }, [tabs, searchWrap, h('div', { class: 'cat-sort' }, [h('label', { for: 'cat-sort', text: 'Sort' }), sortSel])]);
    stage.appendChild(tools);

    /* clipboard fallback panel (shown only when the Clipboard API is refused) */
    var fbText = h('textarea', { readonly: 'readonly', 'aria-label': 'BibTeX text to copy' });
    var fbClose = h('button', { type: 'button', class: 'btn', text: 'Close' });
    var fbMsg = h('span');
    var fallback = h('div', { class: 'cat-fallback', hidden: 'hidden' }, [h('div', { class: 'fb-h' }, [fbMsg, fbClose]), fbText]);
    fbClose.addEventListener('click', function () { fallback.hidden = true; });
    stage.appendChild(fallback);

    /* kind strip: proportional bar + chips (the colour legend doubles as the kind filter) */
    var strip = h('div', { class: 'cat-strip' });
    var stripHead = h('div', { class: 'cat-strip-h' }, [h('span', { class: 'label', text: 'Modification kind' }), h('span', { class: 'cat-legend-note', text: 'colour key · tap to filter' })]);
    var bar = h('div', { class: 'cat-bar', 'aria-hidden': 'true' });
    var kindWrap = h('div', { class: 'cat-kinds', role: 'group', 'aria-label': 'Filter by modification kind' });
    var kindBtns = {}, barSegs = {};
    A.KINDS.forEach(function (k) {
      var seg = h('i', { style: '--k: var(' + k.token + ')' }); bar.appendChild(seg); barSegs[k.key] = seg;
      var b = h('button', { type: 'button', class: 'cat-fchip', 'aria-pressed': 'false', style: '--k: var(' + k.token + ')' }, [h('span', { class: 'sw' }), h('span', { text: k.name }), h('span', { class: 'n', text: '0' })]);
      b.addEventListener('click', function () { S.kinds[k.key] = !S.kinds[k.key]; update(); });
      kindWrap.appendChild(b); kindBtns[k.key] = b;
    });
    strip.appendChild(stripHead); strip.appendChild(bar); strip.appendChild(kindWrap);
    stage.appendChild(strip);

    /* body */
    var body = h('div', { class: 'cat-body' });
    stage.appendChild(body);

    /* ---- rail ---- */
    var rail = h('details', { class: 'cat-rail', open: 'open' });
    var railSum = h('summary', {}, [h('span', { class: 'rs-t', text: 'Filters' }), h('span', { class: 'chev', 'aria-hidden': 'true', text: '▾' })]);
    var railIn = h('div', { class: 'cat-rail-in' });
    rail.appendChild(railSum); rail.appendChild(railIn);
    body.appendChild(rail);

    function grp(title, extra) {
      var g = h('div', { class: 'cat-grp' });
      var hd = h('div', { class: 'cat-grp-h' }, [h('span', { class: 'label', text: title })]);
      if (extra) hd.appendChild(extra);
      g.appendChild(hd);
      railIn.appendChild(g);
      return g;
    }
    /* HF PEFT */
    var gHF = grp('Hugging Face PEFT');
    var hfSeg = h('div', { class: 'cat-seg', role: 'group', 'aria-label': 'Filter by Hugging Face PEFT support' });
    var hfBtns = {};
    [['any', 'Any'], ['yes', 'In PEFT'], ['no', 'Not in']].forEach(function (o) {
      var b = h('button', { type: 'button', 'aria-pressed': String(o[0] === S.hf) }, [h('span', { text: o[1] }), h('span', { class: 'n', text: '' })]);
      b.addEventListener('click', function () { S.hf = o[0]; update(); });
      hfSeg.appendChild(b); hfBtns[o[0]] = b;
    });
    gHF.appendChild(hfSeg);
    /* mergeable */
    var gMerge = grp('Mergeable into W');
    var mergeWrap = h('div', { class: 'cat-merge', role: 'group', 'aria-label': 'Filter by mergeability' });
    var mergeBtns = {};
    [['yes', 'mergeable', '--moss'], ['conditional', 'conditional', '--ochre'], ['no', 'not mergeable', '--seal']].forEach(function (o) {
      var b = h('button', { type: 'button', class: 'cat-fchip', 'aria-pressed': 'false', style: '--k: var(' + o[2] + ')' }, [h('span', { class: 'dot' }), h('span', { text: o[1] }), h('span', { class: 'n', text: '0' })]);
      b.addEventListener('click', function () { S.merge[o[0]] = !S.merge[o[0]]; update(); });
      mergeWrap.appendChild(b); mergeBtns[o[0]] = b;
    });
    gMerge.appendChild(mergeWrap);
    /* core */
    var coreBtn = null;
    if (anyCore) {
      var gCore = grp('Scope');
      coreBtn = h('button', { type: 'button', class: 'cat-fchip', 'aria-pressed': 'false', style: '--k: var(--tide)' }, [h('span', { class: 'dot' }), h('span', { text: 'core methods only' }), h('span', { class: 'n', text: '0' })]);
      coreBtn.addEventListener('click', function () { S.core = !S.core; update(); });
      gCore.appendChild(coreBtn);
    }
    /* year */
    var yrRead = h('span', { class: 'cat-count' });
    var gYear = grp('Year', yrRead);
    var hist = h('div', { class: 'cat-hist' });
    var histAx = h('div', { class: 'cat-hist-ax', 'aria-hidden': 'true' });
    var y0In = h('input', { type: 'range', id: 'cat-y0', step: 1 });
    var y1In = h('input', { type: 'range', id: 'cat-y1', step: 1 });
    var y0Out = h('output', { for: 'cat-y0' }), y1Out = h('output', { for: 'cat-y1' });
    var yr = h('div', { class: 'cat-yr' }, [
      h('div', {}, [h('label', { for: 'cat-y0' }, [h('span', { text: 'From' }), y0Out]), y0In]),
      h('div', {}, [h('label', { for: 'cat-y1' }, [h('span', { text: 'To' }), y1Out]), y1In])
    ]);
    gYear.appendChild(hist); gYear.appendChild(histAx); gYear.appendChild(yr);
    function yState() { return S.tab === 'methods' ? S.yM : S.yP; }
    function yBounds() { return S.tab === 'methods' ? MY : PY; }
    function syncYearInputs() {
      var b = yBounds(), y = yState();
      [y0In, y1In].forEach(function (inp) { inp.min = b[0]; inp.max = b[1]; });
      y0In.value = y[0]; y1In.value = y[1];
      y0Out.textContent = y[0]; y1Out.textContent = y[1];
    }
    var onYear = A.debounce(update, 40);
    y0In.addEventListener('input', function () { var y = yState(); y[0] = +y0In.value; if (y[0] > y[1]) { y[1] = y[0]; y1In.value = y[1]; } y0Out.textContent = y[0]; y1Out.textContent = y[1]; onYear(); });
    y1In.addEventListener('input', function () { var y = yState(); y[1] = +y1In.value; if (y[1] < y[0]) { y[0] = y[1]; y0In.value = y[0]; } y0Out.textContent = y[0]; y1Out.textContent = y[1]; onYear(); });
    hist.addEventListener('click', function (e) {
      var t = e.target.closest && e.target.closest('[data-y]'); if (!t) return;
      var yy = +t.getAttribute('data-y'), y = yState(), b = yBounds();
      if (y[0] === yy && y[1] === yy) { y[0] = b[0]; y[1] = b[1]; } else { y[0] = yy; y[1] = yy; }
      syncYearInputs(); update();
    });
    /* coordinates */
    var gCo = grp('Coordinates', h('span', { class: 'cat-count', text: nClassified + '/' + methods.length }));
    var axisChips = {}, axisSums = {};
    if (nClassified > 0) {
      axisKeys.forEach(function (ax, ai) {
        var a = axisByKey[ax];
        var det = h('details', { class: 'cat-axis' });
        if (ai < 1) det.open = true;
        var sumN = h('span', { class: 'n' });
        det.appendChild(h('summary', { title: a.question || a.name }, [h('span', { text: AXIS_SHORT[ax] || a.name }), sumN]));
        var wrap = h('div', { class: 'cat-axvals', role: 'group', 'aria-label': 'Filter by ' + (AXIS_SHORT[ax] || a.name) });
        axisChips[ax] = {}; axisSums[ax] = sumN;
        var vals = (a.values || []).map(function (v) { return { key: v.key, name: v.name }; }).concat(axisExtra[ax].map(function (k) { return { key: k, name: k + ' (unlisted value)' }; }));
        vals.forEach(function (v) {
          var b = h('button', { type: 'button', class: 'cat-fchip cat-axv', 'aria-pressed': 'false', title: v.key + ': ' + v.name });
          b.innerHTML = '<span class="k">' + keyHTML(v.key) + '</span><span class="nmv">' + prettyAxis(v.name) + '</span><span class="n">0</span>';
          b.__label = (AXIS_SHORT[ax] || ax) + ': ' + v.name;
          b.addEventListener('click', function () { S.coords[ax][v.key] = !S.coords[ax][v.key]; update(); });
          wrap.appendChild(b); axisChips[ax][v.key] = b;
        });
        det.appendChild(wrap);
        gCo.appendChild(det);
      });
    } else {
      var pend = h('div', { class: 'cat-pending' });
      pend.innerHTML = '<b>Classification pending.</b> ' + nClassified + ' of ' + methods.length + ' methods carry coordinates so far; these filters appear when they land.<ol>' +
        axisKeys.map(function (k) { return '<li>' + esc(AXIS_SHORT[k] || k) + '</li>'; }).join('') + '</ol>';
      gCo.appendChild(pend);
    }
    var gPaperNote = h('div', { class: 'cat-grp' }, [h('p', { class: 'cat-note', text: 'Papers are background reading. Each paper card links the catalogued methods it discusses; method filters do not apply here.' })]);
    railIn.appendChild(gPaperNote);

    /* ---- list ---- */
    var listPane = h('div', { class: 'cat-listpane' });
    var countEl = h('div', { class: 'cat-count', 'aria-live': 'polite' });
    var clearBtn = h('button', { type: 'button', class: 'btn', text: 'Clear' });
    var copyAllBtn = h('button', { type: 'button', class: 'btn', text: 'Copy BibTeX' });
    var lhead = h('div', { class: 'cat-lhead' }, [countEl, h('div', { class: 'cat-lbtns' }, [clearBtn, copyAllBtn])]);
    var hiddenNote = h('div', { class: 'cat-hidden', hidden: 'hidden' });
    var scroll = h('div', { class: 'cat-scroll' });
    var list = h('div', { class: 'cat-list', id: 'cat-list-' + FIG, role: 'listbox', 'aria-label': 'Methods' });
    scroll.appendChild(list);
    listPane.appendChild(lhead); listPane.appendChild(hiddenNote); listPane.appendChild(scroll);
    body.appendChild(listPane);

    /* ---- card ---- */
    var cardPane = h('div', { class: 'cat-cardpane', role: 'region', 'aria-label': 'Details', tabindex: '-1' });
    var card = h('article', { class: 'method-card cat-card mathjax_ignore' });
    cardPane.appendChild(card);
    body.appendChild(cardPane);

    /* ---------- tooltips (notes on hover / focus) ---------- */
    var tips = [];
    function tipAttr(html) { tips.push(html); return ' data-tip="' + (tips.length - 1) + '"'; }
    function showTip(t, evt) {
      var i = t.getAttribute('data-tip'); if (i == null || !tips[+i]) return;
      if (!evt) { var r = t.getBoundingClientRect(); evt = { clientX: r.left, clientY: r.bottom - 10 }; }
      A.tip.show(tips[+i], evt);
    }
    card.addEventListener('mouseover', function (e) { var t = e.target.closest && e.target.closest('[data-tip]'); if (t && card.contains(t)) showTip(t, e); });
    card.addEventListener('mousemove', function (e) { if (e.target.closest && e.target.closest('[data-tip]')) A.tip.move(e); });
    card.addEventListener('mouseout', function (e) { var t = e.target.closest && e.target.closest('[data-tip]'); if (t && (!e.relatedTarget || !t.contains(e.relatedTarget))) A.tip.hide(); });
    card.addEventListener('focusin', function (e) { var t = e.target.closest && e.target.closest('[data-tip]'); if (t) showTip(t, null); });
    card.addEventListener('focusout', function () { A.tip.hide(); });

    /* ---------- filtering ---------- */
    function scoreOf(it) {
      var toks = S.toks; if (!toks.length) return 1;
      var total = 0;
      for (var i = 0; i < toks.length; i++) {
        var t = toks[i], tc = t.replace(/[^a-z0-9]/g, ''), s = 0;
        if (it.nName === t || (tc && it.cName === tc)) s = 60;
        else if (it.nName.indexOf(t) === 0 || (tc && it.cName.indexOf(tc) === 0)) s = 36;
        else if (it.nName.indexOf(t) >= 0 || (tc.length > 1 && it.cName.indexOf(tc) >= 0)) s = 24;
        else if (it.nId.indexOf(t) >= 0) s = 18;
        else if (it.nFull.indexOf(t) >= 0) s = 12;
        else if (it.nAuth.indexOf(t) >= 0) s = 9;
        else if (it.nTitle.indexOf(t) >= 0) s = 7;
        else if (it.nIdea.indexOf(t) >= 0) s = 4;
        else if (it.nStruct.indexOf(t) >= 0) s = 3;
        else return 0;
        total += s;
      }
      var qn = norm(S.q).trim();
      if (qn && (it.nName === qn || it.cName === compact(S.q))) total += 100;
      return total;
    }
    function passM(it, skip) {
      var m = it.m;
      if (S.toks.length && !(it.score > 0)) return false;
      if (skip !== 'kind' && nkeys(S.kinds) && !S.kinds[m.modification_kind]) return false;
      if (skip !== 'hf' && S.hf !== 'any' && (S.hf === 'yes') !== !!m.hf_peft) return false;
      if (skip !== 'merge' && nkeys(S.merge) && !S.merge[m.mergeable]) return false;
      if (skip !== 'year' && isFinite(it.year) && (it.year < S.yM[0] || it.year > S.yM[1])) return false;
      if (skip !== 'core' && S.core && m.core !== true) return false;
      for (var ax in S.coords) {
        if (skip === 'ax:' + ax || !nkeys(S.coords[ax])) continue;
        var vals = coordVals(m, ax), ok = false;
        for (var j = 0; j < vals.length; j++) if (S.coords[ax][vals[j]]) { ok = true; break; }
        if (!ok) return false;
      }
      return true;
    }
    function passP(it, skip) {
      if (S.toks.length && !(it.score > 0)) return false;
      if (skip !== 'year' && isFinite(it.year) && (it.year < S.yP[0] || it.year > S.yP[1])) return false;
      return true;
    }
    function activeFilterCount() {
      if (S.tab === 'papers') return (S.q ? 1 : 0) + (S.yP[0] !== PY[0] || S.yP[1] !== PY[1] ? 1 : 0);
      var n = (S.q ? 1 : 0) + countKeys(S.kinds) + (S.hf !== 'any' ? 1 : 0) + countKeys(S.merge) + (S.core ? 1 : 0) + (S.yM[0] !== MY[0] || S.yM[1] !== MY[1] ? 1 : 0);
      for (var ax in S.coords) n += countKeys(S.coords[ax]);
      return n;
    }
    function cmpName(a, b) { return String(a.m.name || a.m.title).localeCompare(String(b.m.name || b.m.title), undefined, { sensitivity: 'base', numeric: true }); }
    function sorter(mode) {
      if (mode === 'year-desc') return function (a, b) { return (b.year || 0) - (a.year || 0) || cmpName(a, b); };
      if (mode === 'az') return cmpName;
      if (mode === 'match') return function (a, b) { return b.score - a.score || (a.year || 0) - (b.year || 0) || cmpName(a, b); };
      return function (a, b) { return (a.year || 0) - (b.year || 0) || cmpName(a, b); };
    }

    /* ---------- rows ---------- */
    function hl(text, toks) {
      text = String(text == null ? '' : text);
      if (!toks.length || /[\^_\\]/.test(text)) return nameU(text);
      var low = text.toLowerCase(), ranges = [];
      toks.forEach(function (t) { if (t.length < 2 && toks.length > 1) return; var i = low.indexOf(t); while (i >= 0) { ranges.push([i, i + t.length]); i = low.indexOf(t, i + t.length); } });
      if (!ranges.length) return esc(text);
      ranges.sort(function (a, b) { return a[0] - b[0]; });
      var merged = [ranges[0]];
      for (var k = 1; k < ranges.length; k++) { var last = merged[merged.length - 1]; if (ranges[k][0] <= last[1]) last[1] = Math.max(last[1], ranges[k][1]); else merged.push(ranges[k]); }
      var out = '', pos = 0;
      merged.forEach(function (r) { out += esc(text.slice(pos, r[0])) + '<mark>' + esc(text.slice(r[0], r[1])) + '</mark>'; pos = r[1]; });
      return out + esc(text.slice(pos));
    }
    function mergeChip(v, short) {
      var cls = MERGE_CLASS[v]; if (!cls) return '';
      return '<span class="chip ' + cls + '">' + esc(short ? MERGE_SHORT[v] : MERGE_LABEL[v]) + '</span>';
    }
    function buildRow(it) {
      var r = h('div', { class: 'cat-row' + (it.paper ? ' paper' : ''), role: 'option', 'aria-selected': 'false', tabindex: '-1', id: 'cat-opt-' + (it.paper ? 'p-' : 'm-') + it.id, 'data-id': it.id });
      it.row = r;
      paintRow(it);
      return r;
    }
    function paintRow(it) {
      var m = it.m, r = it.row, toks = S.toks;
      if (it.paper) {
        var nm = (m.method_ids || []).length;
        r.innerHTML = '<div class="nm">' + hl(m.title, toks) + '</div><div class="rt"><span class="yr">' + esc(m.year) + '</span></div>' +
          '<div class="ln2"><span class="vn">' + esc(m.venue || '') + '</span><span class="fn">' + hl(m.authors || '', toks) + '</span>' +
          (nm ? '<span class="ch"><span class="chip cnt">' + nm + (nm === 1 ? ' method' : ' methods') + '</span></span>' : '') + '</div>';
        r.setAttribute('aria-label', textOf(nameU(m.title)) + ', ' + (m.authors || '') + ', ' + m.year);
        return;
      }
      r.innerHTML = '<span class="sw" style="--k: var(' + kindTok(m.modification_kind) + ')"></span>' +
        '<div class="nm">' + hl(m.name, toks) + '</div>' +
        '<div class="rt">' + (m.hf_peft ? '<span class="chip hf" title="In Hugging Face PEFT">HF</span>' : '') + '<span class="yr">' + esc(m.year) + '</span></div>' +
        '<div class="ln2"><span class="vn">' + esc(m.venue || '') + '</span><span class="fn">' + hl(m.full_name, toks) + '</span><span class="ch">' + mergeChip(m.mergeable, true) + '</span></div>';
      r.setAttribute('aria-label', textOf(nameU(m.name)) + ', ' + A.kindName(m.modification_kind) + ', ' + m.year + (m.venue ? ', ' + m.venue : '') + (m.hf_peft ? ', in HF PEFT' : '') + (MERGE_LABEL[m.mergeable] ? ', ' + MERGE_LABEL[m.mergeable] : ''));
    }
    var paintedToks = { methods: null, papers: null };

    /* ---------- the render pipeline ---------- */
    function update() {
      var isM = S.tab === 'methods';
      var coll = isM ? items : pitems;
      coll.forEach(function (it) { it.score = scoreOf(it); });
      var pass = isM ? passM : passP;
      visible = coll.filter(function (it) { return pass(it, null); });
      var mode = S.sort === 'match' && !S.toks.length ? 'year-asc' : S.sort;
      visible.sort(sorter(mode));

      /* facet counts: each facet counts over items passing every *other* filter */
      if (isM) {
        var kc = {}, kTot = 0;
        items.forEach(function (it) { if (passM(it, 'kind')) { kc[it.m.modification_kind] = (kc[it.m.modification_kind] || 0) + 1; kTot++; } });
        var anyK = nkeys(S.kinds);
        A.KINDS.forEach(function (k) {
          var n = kc[k.key] || 0, b = kindBtns[k.key];
          b.querySelector('.n').textContent = n;
          b.setAttribute('aria-pressed', String(!!S.kinds[k.key]));
          b.classList.toggle('zero', n === 0);
          b.setAttribute('aria-label', k.name + ': ' + n + ' methods' + (S.kinds[k.key] ? ', filter on' : ''));
          var seg = barSegs[k.key];
          seg.style.flexGrow = String(n);
          seg.style.display = n ? '' : 'none';
          seg.classList.toggle('off', anyK && !S.kinds[k.key]);
          seg.title = k.name + ': ' + n;
        });
        bar.style.opacity = kTot ? '' : '.3';
        var hc = { any: 0, yes: 0, no: 0 };
        items.forEach(function (it) { if (passM(it, 'hf')) { hc.any++; hc[it.m.hf_peft ? 'yes' : 'no']++; } });
        ['any', 'yes', 'no'].forEach(function (k) { hfBtns[k].querySelector('.n').textContent = hc[k]; hfBtns[k].setAttribute('aria-pressed', String(S.hf === k)); });
        var mc = { yes: 0, conditional: 0, no: 0 };
        items.forEach(function (it) { if (passM(it, 'merge') && it.m.mergeable in mc) mc[it.m.mergeable]++; });
        Object.keys(mergeBtns).forEach(function (k) { var b = mergeBtns[k]; b.querySelector('.n').textContent = mc[k]; b.setAttribute('aria-pressed', String(!!S.merge[k])); b.classList.toggle('zero', !mc[k]); });
        if (coreBtn) { var cc = 0; items.forEach(function (it) { if (passM(it, 'core') && it.m.core === true) cc++; }); coreBtn.querySelector('.n').textContent = cc; coreBtn.setAttribute('aria-pressed', String(S.core)); }
        if (nClassified > 0) axisKeys.forEach(function (ax) {
          var c = {};
          items.forEach(function (it) { if (passM(it, 'ax:' + ax)) coordVals(it.m, ax).forEach(function (v) { c[v] = (c[v] || 0) + 1; }); });
          var nSel = 0;
          Object.keys(axisChips[ax]).forEach(function (v) { var b = axisChips[ax][v]; b.querySelector('.n').textContent = c[v] || 0; b.setAttribute('aria-label', b.__label + ', ' + (c[v] || 0) + ' methods'); b.setAttribute('aria-pressed', String(!!S.coords[ax][v])); b.classList.toggle('zero', !c[v]); if (S.coords[ax][v]) nSel++; });
          axisSums[ax].textContent = nSel ? nSel + ' on' : '';
          if (nSel && !axisSums[ax].parentNode.parentNode.open) axisSums[ax].parentNode.parentNode.open = true;
        });
      }
      /* year histogram (facet over year) */
      var yc = {}, yt = {}, b = yBounds(), y = yState();
      coll.forEach(function (it) { if (!isFinite(it.year)) return; yt[it.year] = (yt[it.year] || 0) + 1; if (pass(it, 'year')) yc[it.year] = (yc[it.year] || 0) + 1; });
      drawHist(yc, yt, b, y);
      yrRead.textContent = (y[0] === y[1] ? y[0] : y[0] + '–' + y[1]);

      /* list */
      renderList(isM);
      var total = coll.length, noun = isM ? 'methods' : 'papers';
      countEl.innerHTML = '<b>' + visible.length + '</b> of ' + total + ' ' + noun;
      var nUnique = uniqueBib(visible).length;
      copyAllBtn.textContent = 'Copy ' + nUnique + ' BibTeX';
      copyAllBtn.disabled = !nUnique;
      copyAllBtn.setAttribute('aria-label', 'Copy BibTeX for all ' + visible.length + ' filtered ' + noun + ' (' + nUnique + ' unique entries)');
      copyAllBtn.title = nUnique < visible.length ? nUnique + ' distinct entries: ' + noun + ' that share a paper share one entry' : '';
      var nAct = activeFilterCount();
      clearBtn.hidden = !nAct;
      railSum.querySelector('.rs-t').textContent = 'Filters' + (nAct ? ' · ' + nAct + ' active' : '');
    }

    function drawHist(yc, yt, b, y) {
      var n = b[1] - b[0] + 1, W = Math.max(1, n) * 10, H = 44, max = 1, yy;
      for (yy = b[0]; yy <= b[1]; yy++) max = Math.max(max, yt[yy] || 0);
      var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" role="img" aria-label="Distribution by year">';
      for (yy = b[0]; yy <= b[1]; yy++) {
        var x = (yy - b[0]) * 10 + 1, t = yt[yy] || 0, f = yc[yy] || 0;
        var th = t ? Math.max(1.5, t / max * (H - 2)) : 0, fh = f ? Math.max(1.5, f / max * (H - 2)) : 0;
        var out = yy < y[0] || yy > y[1];
        if (th) s += '<rect class="hb" x="' + x + '" y="' + (H - th) + '" width="8" height="' + th + '"/>';
        if (fh) s += '<rect class="hf' + (out ? ' out' : '') + '" x="' + x + '" y="' + (H - fh) + '" width="8" height="' + fh + '"/>';
        s += '<rect class="hit" data-y="' + yy + '" x="' + (x - 1) + '" y="0" width="10" height="' + H + '"><title>' + yy + ': ' + f + ' of ' + t + (S.tab === 'methods' ? ' methods' : ' papers') + ' (click to isolate)</title></rect>';
      }
      hist.innerHTML = s + '</svg>';
      histAx.innerHTML = '<span>' + b[0] + '</span><span>peak <b>' + max + '</b>/yr</span><span>' + b[1] + '</span>';
    }

    function renderList(isM) {
      var key = isM ? 'methods' : 'papers', tk = S.toks.join(' ');
      var coll = isM ? items : pitems;
      if (paintedToks[key] !== tk) { coll.forEach(function (it) { if (it.row) paintRow(it); }); paintedToks[key] = tk; }
      var frag = document.createDocumentFragment();
      var mode = S.sort === 'match' && !S.toks.length ? 'year-asc' : S.sort;
      var yearHeads = mode === 'year-asc' || mode === 'year-desc';
      var perYear = {};
      function yKey(it) { return isFinite(it.year) ? String(it.year) : '—'; }   /* NaN !== NaN: key undated entries */
      if (yearHeads) visible.forEach(function (it) { perYear[yKey(it)] = (perYear[yKey(it)] || 0) + 1; });
      var lastY = null, sel = isM ? S.selM : S.selP, selVisible = false;
      visible.forEach(function (it) {
        if (yearHeads && yKey(it) !== lastY) {
          lastY = yKey(it);
          var yh = document.createElement('div');
          yh.className = 'cat-yh'; yh.setAttribute('aria-hidden', 'true');
          yh.innerHTML = '<b>' + esc(lastY) + '</b><i></i><span>' + perYear[lastY] + '</span>';
          frag.appendChild(yh);
        }
        var r = it.row || buildRow(it);
        var on = it.id === sel;
        if (on) selVisible = true;
        r.setAttribute('aria-selected', String(on));
        r.tabIndex = -1;
        frag.appendChild(r);
      });
      list.textContent = '';
      if (!visible.length) {
        var em = document.createElement('div');
        em.className = 'cat-empty';
        em.innerHTML = 'Nothing matches these filters. <button type="button" data-clear>Clear filters</button>';
        list.appendChild(em);
      } else list.appendChild(frag);
      list.setAttribute('aria-label', isM ? 'Methods' : 'Papers');
      /* roving tab stop: the selected row, else the first */
      var stop = selVisible ? (coll.filter(function (it) { return it.id === sel; })[0] || {}).row : (visible[0] && visible[0].row);
      if (stop) stop.tabIndex = 0;
      /* note when the open card is outside the filters */
      var selItem = sel && (isM ? byId[sel] : pById[sel]);
      if (selItem && !selVisible && visible.length >= 0 && activeFilterCount()) {
        hiddenNote.hidden = false;
        hiddenNote.innerHTML = '<b>' + (isM ? nameU(selItem.name) : 'This paper') + '</b> (open in the card) is outside the current filters. <button type="button" data-clear>Clear filters</button>';
      } else hiddenNote.hidden = true;
    }

    function ensureVisible(row) {
      if (!row || !row.parentNode) return;
      var c = scroll, top = row.offsetTop, bot = top + row.offsetHeight, pad = 30;
      if (top - pad < c.scrollTop) c.scrollTop = Math.max(0, top - pad);
      else if (bot > c.scrollTop + c.clientHeight) c.scrollTop = bot - c.clientHeight + 4;
    }

    /* ---------- selection ---------- */
    var renderCardSoon = A.debounce(function () { renderCard(); }, 70);
    function select(id, opts) {
      opts = opts || {};
      var isM = S.tab === 'methods';
      if (isM ? !byId[id] : !pById[id]) return;
      if (isM) S.selM = id; else S.selP = id;
      S.authorsOpen = false;
      var coll = isM ? items : pitems;
      coll.forEach(function (it) { if (it.row) { var on = it.id === id; it.row.setAttribute('aria-selected', String(on)); it.row.tabIndex = on ? 0 : -1; } });
      var it = coll.filter(function (x) { return x.id === id; })[0];
      if (it && it.row && it.row.parentNode) {
        if (opts.focus) it.row.focus({ preventScroll: true });
        ensureVisible(it.row);
      }
      if (opts.hash) setHash((isM ? 'm-' : 'p-') + id);
      if (opts.defer) renderCardSoon(); else renderCard();
      if (opts.reveal) revealCard();
      if (!visible.some(function (x) { return x.id === id; })) renderList(isM);
      else hiddenNote.hidden = true;
    }
    function revealCard() {
      cardPane.scrollTop = 0;
      if (layout === 'narrow') {
        var r = cardPane.getBoundingClientRect();
        if (r.top < 0 || r.top > window.innerHeight * 0.6) cardPane.scrollIntoView({ block: 'start', behavior: A.reducedMotion() ? 'auto' : 'smooth' });
      }
    }
    var selfHash = false;
    function setHash(tok) {
      if (('#' + tok) === window.location.hash) return;
      try { history.replaceState(history.state, '', '#' + tok); }
      catch (e) { try { selfHash = true; window.location.hash = tok; } catch (e2) {} }
    }

    /* ---------- cards ---------- */
    function link(id, note, extraTip) {
      var t = byId[id];
      if (!t) return '<span class="cc-link missing" title="Not in the catalogue">' + esc(id) + '</span>';
      var tipHtml = '<div class="t">' + nameU(t.name) + '</div>' + (extraTip ? '<div class="label">' + extraTip + '</div>' : '') + (note ? '<div>' + proseU(note) + '</div>' : '<div class="muted">' + nameU(t.full_name || '') + '</div>');
      return '<a class="cc-link" href="#m-' + esc(id) + '" data-goto="' + esc(id) + '"' + tipAttr(tipHtml) + '><span class="sw" style="--k: var(' + kindTok(t.modification_kind) + ')"></span>' + nameU(t.name) + '</a>';
    }
    function paperLink(pid) {
      var p = pById[pid]; if (!p) return '';
      return '<a class="cc-link paper" href="#p-' + esc(pid) + '" data-gopaper="' + esc(pid) + '"' + tipAttr('<div class="t">' + nameU(p.title) + '</div><div class="muted">' + esc((p.authors || '') + ' · ' + (p.venue || p.year)) + '</div>') + '>' + esc(p.authors || p.title) + ' <span class="muted">' + esc(p.year) + '</span></a>';
    }
    function authorsHTML(full, short) {
      if (!full || !full.length) return short ? '<div class="cc-auth">' + esc(short) + '</div>' : '';
      var LIM = 6;
      if (full.length <= LIM) return '<div class="cc-auth">' + esc(full.join(', ')) + '</div>';
      if (S.authorsOpen) return '<div class="cc-auth">' + esc(full.join(', ')) + ' <button type="button" data-authors aria-expanded="true">show fewer</button></div>';
      return '<div class="cc-auth">' + esc(full.slice(0, LIM).join(', ')) + ', et al. <button type="button" data-authors aria-expanded="false" aria-label="Show all ' + full.length + ' authors">+' + (full.length - LIM) + ' more</button></div>';
    }
    function linksHTML(x) {
      var s = '';
      var abs = x.arxiv ? 'https://arxiv.org/abs/' + x.arxiv : '';
      if (x.arxiv) s += '<a href="' + esc(abs) + '" target="_blank" rel="noopener">arXiv:' + esc(x.arxiv) + ' ↗</a>';
      var u = x.url || '';
      if (u && u.replace(/^http:/, 'https:').replace(/v\d+$/, '').replace(/\/$/, '') !== abs) s += '<a href="' + esc(u) + '" target="_blank" rel="noopener">' + esc(hostOf(u)) + ' ↗</a>';
      if (x.code_url) s += '<a href="' + esc(x.code_url) + '" target="_blank" rel="noopener">code · ' + esc(hostOf(x.code_url).replace(/^github\.com\//, 'gh/')) + ' ↗</a>';
      return s;
    }
    function venueHTML(x) {
      var v = x.venue || '';
      return v ? '<span class="cc-venue">' + esc(v) + '</span>' : '';
    }
    function mathBlock(tex, display) {
      var lines = splitClauses(tex);
      return lines.map(function (ln) {
        return '<div class="' + (display ? 'ln' : 'cc-math') + ' mathjax_process">' + esc(display ? '\\[' + ln + '\\]' : '\\(' + ln + '\\)') + '</div>';
      }).join('');
    }
    /* a prose field as an element whose class asks MathJax in only when it holds math */
    function proseEl(tag, cls, s) {
      var r = prose(s), c = (cls || '') + (r.mj ? ' mathjax_process' : '');
      return '<' + tag + (c.trim() ? ' class="' + c.trim() + '"' : '') + '>' + r.h + '</' + tag + '>';
    }
    function paramsHTML(tp) {
      tp = String(tp || '').trim();
      if (!tp) return '';
      if (hasMath(tp)) return prose(tp);
      if (!/\\/.test(tp) && /[a-z]{4,}/i.test(tp.replace(/_\{[^}]*\}/g, ''))) return { h: texHTML(tp), mj: false };
      return { h: mathBlock(tp, false), mj: false };
    }
    function coordsHTML(m) {
      var classified = hasCoords(m);
      var s = '<div class="cc-sec"><div class="cc-sec-h"><span>Coordinates</span><small>seven axes · Exposé VIII</small></div>';
      if (!classified) {
        s += '<div class="cc-pendnote">Classification pending for this method.</div><div class="cc-coords">';
        axisKeys.forEach(function (ax) {
          var a = axisByKey[ax];
          s += '<div class="cc-co pend"' + tipAttr('<div class="t">' + prettyAxis(a.name) + '</div><div>' + prettyDef(a.question || '') + '</div>') + ' tabindex="0"><div class="ax">' + esc(AXIS_SHORT[ax] || a.name) + '</div><div class="v">—</div></div>';
        });
        return s + '</div></div>';
      }
      s += '<div class="cc-coords">';
      var notes = [];
      axisKeys.forEach(function (ax) {
        var a = axisByKey[ax], vals = coordVals(m, ax), note = m.coord_notes && m.coord_notes[ax];
        if (note) notes.push([AXIS_SHORT[ax] || a.name, note]);
        var vHTML = vals.length ? vals.map(keyHTML).join(' · ') : '—';
        var names = vals.map(function (v) { var av = axisValue(ax, v); return av ? prettyAxis(av.name) : (v ? 'unlisted value' : ''); }).filter(Boolean).join('; ');
        var tipH = '<div class="t">' + prettyAxis(a.name) + '</div>' + (note ? '<div>' + proseU(note) + '</div>' : '') +
          vals.map(function (v) { var av = axisValue(ax, v); return av && av.definition ? '<div class="muted" style="margin-top:.3rem">' + keyHTML(v) + ': ' + prettyDef(av.definition) + '</div>' : ''; }).join('');
        s += '<div class="cc-co' + (note ? ' has-note' : '') + (vals.length ? '' : ' pend') + '"' + tipAttr(tipH) + ' tabindex="0"><div class="ax">' + esc(AXIS_SHORT[ax] || a.name) + '</div><div class="v">' + vHTML + '</div>' + (names ? '<div class="vn">' + names + '</div>' : '') + '</div>';
      });
      s += '</div>';
      if (notes.length) {
        s += '<details class="cc-more" data-keep="notes"' + (S.notesOpen ? ' open' : '') + '><summary>Why these values · ' + notes.length + ' note' + (notes.length === 1 ? '' : 's') + '</summary><dl class="cc-cn">' +
          notes.map(function (n) { return '<dt>' + esc(n[0]) + '</dt>' + proseEl('dd', '', n[1]); }).join('') + '</dl></details>';
      }
      if (m.table_row || m.table_col) s += '<div class="cc-pendnote">Periodic table cell: row ' + texHTML(m.table_row || '—') + ', column ' + texHTML(m.table_col || '—') + '.</div>';
      return s + '</div>';
    }
    function relationsHTML(m) {
      var rels = (m.relations || []).filter(function (r) { return r && r.target; });
      var inb = inRel[m.id] || [];
      if (!rels.length && !inb.length) return '';
      var s = '<div class="cc-sec"><div class="cc-sec-h"><span>Relations</span><small>as published · ' + rels.length + ' out · ' + inb.length + ' in</small></div>';
      function groups(list, labels, key) {
        var g = {}, order = REL_ORDER.slice();
        list.forEach(function (r) { var t = r.type || 'related'; if (order.indexOf(t) < 0) order.push(t); (g[t] = g[t] || []).push(r); });
        var out = '';
        order.forEach(function (t) {
          if (!g[t]) return;
          out += '<div class="cc-rg"><span class="cc-rt">' + esc(labels[t] || t.replace(/-/g, ' ')) + '</span>' +
            g[t].map(function (r) { return link(r[key], r.note, esc(labels[t] || t)); }).join('') + '</div>';
        });
        return out;
      }
      if (rels.length) s += groups(rels, REL_OUT, 'target');
      if (inb.length) {
        var nSrc = {}; inb.forEach(function (r) { nSrc[r.source] = 1; });
        s += '<details class="cc-in"><summary>Named in the relations of ' + Object.keys(nSrc).length + ' other method' + (Object.keys(nSrc).length === 1 ? '' : 's') + '</summary>' + groups(inb, REL_IN, 'source') + '</details>';
      }
      return s + '</div>';
    }
    function ruleChip(key, ob) {
      if (!key) return '<span></span>';
      var r = ruleByKey[key];
      var tipH = r ? '<div class="t">' + esc(r.key + ' · ' + r.name) + '</div><div>' + prettyDef(r.rule) + '</div>' : '<div>' + esc(key) + '</div>';
      return '<span class="chip rule' + (ob ? ' ob' : '') + '" tabindex="0"' + tipAttr(tipH) + '>' + esc(key) + '</span>';
    }
    /* one arrow: glyph, rule chip, the other method, its map h and the note, each on its own line */
    function arrowRow(sym, rule, ob, other, hh, note) {
      var s = '<div class="cc-arrow' + (ob ? ' ob' : '') + '"><span class="ar" aria-hidden="true">' + sym + '</span>' + ruleChip(rule, ob) + link(other, null);
      if (hh) s += hasMath(hh) ? proseEl('div', 'cc-h cc-an', hh) : '<div class="cc-math cc-h mathjax_process">' + esc('\\(' + hTeX(hh) + '\\)') + '</div>';
      if (note) s += proseEl('div', 'cc-an', note);
      return s + '</div>';
    }
    var IN_INLINE = 4;   /* more incoming arrows than this go behind a disclosure, rendered on demand */
    function inArrowsHTML(m) {
      return (arrowsIn[m.id] || []).map(function (a) { return arrowRow('←', a.rule, false, a.source, a.h, a.note); }).join('');
    }
    function arrowsHTML(m) {
      var out = arrowsOut[m.id] || [], inn = arrowsIn[m.id] || [], obs = (m.obstructions || []).filter(function (o) { return o && o.target; });
      if (!out.length && !inn.length && !obs.length) return '';
      var nm = nameU(m.name);
      var s = '<div class="cc-sec"><div class="cc-sec-h"><span>Arrows</span><small>' + out.length + ' out · ' + inn.length + ' in · ' + obs.length + ' ruled out</small></div>';
      s += '<p class="cc-lede mathjax_process">An arrow \\(M\\to N\\) means \\(N\\) simulates \\(M\\), through a pointed map \\(h\\) with \\(\\rho_N\\circ h=\\rho_M\\). Each chip names the rule behind an arrow, or the test that rules one out; hover or tap it to read it.</p>';
      if (out.length) s += '<div class="cc-ag"><div class="cc-agh">Simulated by · <b>' + nm + ' → N</b></div>' + out.map(function (a) { return arrowRow('→', a.rule, false, a.target, a.h, a.note); }).join('') + '</div>';
      if (inn.length) {
        var head = 'Simulates · <b>N → ' + nm + '</b>';
        if (inn.length <= IN_INLINE) s += '<div class="cc-ag"><div class="cc-agh">' + head + '</div>' + inArrowsHTML(m) + '</div>';
        else s += '<details class="cc-more cc-ag" data-keep="in" data-lazy="in"' + (S.inOpen ? ' open' : '') + '><summary>' + nm + ' simulates ' + inn.length + ' methods</summary><div class="cc-agh">' + head + '</div><div class="cc-lazy mathjax_ignore">' + (S.inOpen ? inArrowsHTML(m) : '') + '</div></details>';
      }
      if (obs.length) s += '<div class="cc-ag"><div class="cc-agh">Ruled out by a test · <b>' + nm + ' ↛ N</b></div>' + obs.map(function (o) { return arrowRow('↛', o.test, true, o.target, null, o.reason); }).join('') + '</div>';
      return s + '</div>';
    }
    function bibHTML(x, label) {
      if (!x.bibtex) return '';
      return '<div class="cc-sec"><div class="cc-sec-h cc-cite-h"><span>Cite <small>' + esc(x.bibkey || '') + '</small></span><button type="button" class="btn" data-copy aria-label="Copy the BibTeX entry for ' + esc(label) + '">Copy BibTeX</button></div><pre class="cc-bib" tabindex="0" aria-label="BibTeX for ' + esc(label) + '">' + esc(x.bibtex) + '</pre></div>';
    }
    function dlHTML(dl) {
      return '<dl>' + dl.map(function (d) { return '<dt>' + d[0] + '</dt><dd' + (d[2] ? ' class="mathjax_process"' : '') + '>' + d[1] + '</dd>'; }).join('') + '</dl>';
    }
    function proseRow(label, s, pre) { var r = prose(s); return [label, (pre || '') + r.h, r.mj]; }
    function methodCardHTML(m) {
      var k = kindTok(m.modification_kind), s = '';
      s += '<div class="cc-top"><span class="cc-kind" style="--k: var(' + k + ')"><span class="sw"></span>' + esc(A.kindName(m.modification_kind)) + '</span>';
      if (m.entry_type) s += '<span class="chip">' + esc(String(m.entry_type).replace(/[-_]/g, ' ')) + '</span>';
      if (m.core === true) s += '<span class="chip core" title="A core method, drawn in the category of methods">core</span>';
      if (m.hf_peft) s += '<span class="chip hf" title="In Hugging Face PEFT">HF PEFT</span>';
      s += '<span class="cc-year" title="Year of first arXiv version">' + esc(m.year) + '</span></div>';
      s += hasMath(m.name) ? proseEl('h4', '', m.name) : '<h4>' + texHTML(m.name) + '</h4>';
      if (m.full_name && norm(m.full_name) !== norm(m.name)) s += hasMath(m.full_name) ? proseEl('div', 'full', m.full_name) : '<div class="full">' + texHTML(m.full_name) + '</div>';
      if (m.paper_title && norm(m.paper_title) !== norm(m.full_name)) s += '<div class="cc-paper">“' + esc(m.paper_title) + '”</div>';
      s += authorsHTML(m.authors_full, m.authors);
      s += '<div class="cc-meta">' + venueHTML(m) + linksHTML(m) + '</div>';
      if (m.formula_latex) s += '<div class="cc-formula" role="math" aria-label="Formula: ' + esc(m.formula_latex) + '">' + mathBlock(m.formula_latex, true) + '</div>';
      var dl = [];
      if (m.trainable_params) { var tp = paramsHTML(m.trainable_params); dl.push(['Trainable', tp.h, tp.mj]); }
      if (m.locus) dl.push(proseRow('Locus', m.locus));
      if (m.structure) dl.push(proseRow('Structure', m.structure));
      if (m.mergeable) dl.push(proseRow('Merge', m.mergeable_reason || '', mergeChip(m.mergeable, false) || '<span class="chip">' + esc(m.mergeable) + '</span>'));
      if (m.identity_at_init) dl.push(proseRow('At init', m.identity_at_init));
      if (m.key_idea) dl.push(proseRow('Key idea', m.key_idea));
      if (m.results_note) dl.push(proseRow('Results', m.results_note));
      if (m.hf_peft && m.hf_peft_config) dl.push(['HF PEFT', '<code>' + esc(m.hf_peft_config) + '</code>', false]);
      s += dlHTML(dl);
      s += coordsHTML(m);
      if (m.categorical_reading) s += '<div class="cc-sec"><div class="cc-sec-h"><span>Categorical reading</span><small>under the hypotheses of the results it cites</small></div>' + proseEl('div', 'cc-read', m.categorical_reading) + '</div>';
      s += arrowsHTML(m);
      s += relationsHTML(m);
      var pp = papersFor[m.id] || [];
      if (pp.length) s += '<div class="cc-sec"><div class="cc-sec-h"><span>Discussed in</span><small>' + pp.length + ' paper' + (pp.length === 1 ? '' : 's') + '</small></div><div class="cc-rg">' + pp.map(paperLink).join('') + '</div></div>';
      s += bibHTML(m, textOf(nameU(m.name)));
      s += '<div class="cc-foot"><span>permalink <code>#m-' + esc(m.id) + '</code></span><span>confidence ' + esc(m.confidence || '—') + (m.verified ? ' · verified against sources' : '') + '</span></div>';
      return s;
    }
    function paperCardHTML(p) {
      var s = '';
      s += '<div class="cc-top"><span class="cc-kind" style="--k: var(--ink-2)"><span class="sw"></span>Paper</span><span class="cc-year">' + esc(p.year) + '</span></div>';
      s += hasMath(p.title) ? proseEl('h4', '', p.title) : '<h4>' + esc(p.title) + '</h4>';
      s += authorsHTML(p.authors_full, p.authors);
      s += '<div class="cc-meta">' + venueHTML(p) + linksHTML(p) + '</div>';
      var dl = [];
      if (p.claim) dl.push(proseRow('Claim', p.claim));
      if (p.relevance) dl.push(proseRow('Relevance', p.relevance));
      s += dlHTML(dl);
      var ms = (p.method_ids || []);
      if (ms.length) s += '<div class="cc-sec"><div class="cc-sec-h"><span>Methods discussed</span><small>' + ms.length + '</small></div><div class="cc-rg">' + ms.map(function (id) { return link(id, null); }).join('') + '</div></div>';
      s += bibHTML(p, textOf(nameU(p.title)));
      s += '<div class="cc-foot"><span>permalink <code>#p-' + esc(p.id) + '</code></span><span>confidence ' + esc(p.confidence || '—') + (p.verified ? ' · verified against sources' : '') + '</span></div>';
      return s;
    }
    /* MathJax runs one typeset at a time (a queue), and a typeset for a card that has since been
       replaced is skipped (cardGen), so fast keyboard browsing never typesets stale nodes. */
    var tsq = Promise.resolve(), cardGen = 0;
    function typesetNode(node, gen) {
      tsq = tsq.then(function () {
        if (gen !== cardGen || !node.isConnected || !node.querySelector('.mathjax_process')) return;
        return A.typeset(node);
      }).then(function () { if (gen === cardGen) markOverflow(); }, function () {});
      return tsq;
    }
    function renderCard() {
      var isM = S.tab === 'methods';
      var x = isM ? byId[S.selM] : pById[S.selP];
      var gen = ++cardGen;
      try { if (window.MathJax && MathJax.typesetClear) MathJax.typesetClear([card]); } catch (e) {}
      tips = [];
      A.tip.hide();
      if (!x) { card.innerHTML = '<p class="cat-note">Nothing selected.</p>'; return; }
      card.innerHTML = isM ? methodCardHTML(x) : paperCardHTML(x);
      cardPane.setAttribute('aria-label', (isM ? 'Method: ' : 'Paper: ') + textOf(nameU(x.name || x.title)));
      markOverflow();
      typesetNode(card, gen);
    }
    /* long formula lines keep their type size and scroll sideways; fade the edge that has more.
       An inline formula wider than its line becomes its own scrolling line. */
    function markOverflow() {
      Array.prototype.forEach.call(card.querySelectorAll('.ln, .cc-math'), function (n) {
        var o = n.scrollWidth > n.clientWidth + 2, atL = n.scrollLeft <= 1, atR = n.scrollLeft + n.clientWidth >= n.scrollWidth - 1;
        n.classList.toggle('ovf-r', o && atL);
        n.classList.toggle('ovf-l', o && atR && !atL);
        n.classList.toggle('ovf-lr', o && !atL && !atR);
        if (o && !n.hasAttribute('tabindex')) { n.setAttribute('tabindex', '0'); n.setAttribute('aria-label', 'Formula line, scrolls sideways'); }
      });
      Array.prototype.forEach.call(card.querySelectorAll('mjx-container:not([display="true"])'), function (mj) {
        if (mj.closest('.ln, .cc-math')) return;
        var host = mj.parentElement; if (!host || !host.clientWidth) return;
        var svg = mj.firstElementChild, w = svg && svg.getBoundingClientRect ? svg.getBoundingClientRect().width : 0;
        if (w > host.clientWidth + 1) { mj.classList.add('cc-wide'); if (!mj.hasAttribute('tabindex')) mj.setAttribute('tabindex', '0'); }
        else if (mj.classList.contains('cc-wide') && w <= host.clientWidth - 40) mj.classList.remove('cc-wide');
      });
    }
    card.addEventListener('scroll', function (e) { var t = e.target; if (t && t.classList && (t.classList.contains('ln') || t.classList.contains('cc-math'))) markOverflow(); }, true);
    /* disclosures: remember open/closed across cards, and fill the long ones only when opened */
    card.addEventListener('toggle', function (e) {
      var d = e.target; if (!d || d.tagName !== 'DETAILS') return;
      var keep = d.getAttribute('data-keep');
      if (keep === 'notes') S.notesOpen = d.open;
      if (keep === 'in') S.inOpen = d.open;
      if (d.open) {
        var lazy = d.querySelector('.cc-lazy');
        if (lazy && !lazy.firstChild && d.getAttribute('data-lazy') === 'in') {
          var x = byId[S.selM];
          if (x) { lazy.innerHTML = inArrowsHTML(x); typesetNode(lazy, cardGen); }
        }
        markOverflow();
      }
    }, true);

    /* card interactions (delegated) */
    card.addEventListener('click', function (e) {
      var t = e.target.closest ? e.target : null; if (!t) return;
      var go = t.closest('[data-goto]');
      if (go) {
        e.preventDefault();
        if (S.tab !== 'methods') setTab('methods', true);
        select(go.getAttribute('data-goto'), { hash: true, reveal: true });
        return;
      }
      var gp = t.closest('[data-gopaper]');
      if (gp) { e.preventDefault(); setTab('papers', true); select(gp.getAttribute('data-gopaper'), { hash: true, reveal: true }); return; }
      if (t.closest('[data-authors]')) { S.authorsOpen = !S.authorsOpen; var x = S.tab === 'methods' ? byId[S.selM] : pById[S.selP]; var au = card.querySelector('.cc-auth'); if (au && x) { au.outerHTML = authorsHTML(x.authors_full, x.authors); var nb = card.querySelector('[data-authors]'); if (nb) nb.focus(); } return; }
      var cp = t.closest('[data-copy]');
      if (cp) {
        var pre = card.querySelector('.cc-bib'); if (!pre) return;
        copyText(pre.textContent, function () { flash(cp, 'Copied', 'Copy BibTeX'); }, function () { selectNode(pre); flash(cp, 'Selected · ⌘/Ctrl C', 'Copy BibTeX', 2600); });
      }
    });

    /* ---------- clipboard ---------- */
    /* Clipboard API inside the click handler; on rejection (or a promise that never settles,
       as in some sandboxes) fall back to selecting the text for a manual copy. */
    function copyText(text, ok, fail) {
      var done = false, timer = null;
      function win() { if (done) return; done = true; clearTimeout(timer); ok(); }
      function lose() { if (done) return; done = true; clearTimeout(timer); fail(); }
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          timer = setTimeout(lose, 900);
          navigator.clipboard.writeText(text).then(win, lose);
          return;
        }
      } catch (e) {}
      lose();
    }
    function selectNode(node) {
      try { var r = document.createRange(); r.selectNodeContents(node); var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r); } catch (e) {}
    }
    function flash(btn, msg, back, ms) {
      btn.textContent = msg;
      clearTimeout(btn.__t);
      btn.__t = setTimeout(function () { btn.textContent = back; }, ms || 1600);
    }
    function uniqueBib(arr) {
      var seen = {}, out = [];
      arr.forEach(function (it) { var x = it.m; if (!x.bibtex) return; var k = x.bibkey || x.bibtex; if (seen[k]) return; seen[k] = 1; out.push(x.bibtex); });
      return out;
    }
    copyAllBtn.addEventListener('click', function () {
      var bibs = uniqueBib(visible); if (!bibs.length) return;
      var text = bibs.join('\n\n') + '\n';
      var label = copyAllBtn.textContent;
      copyText(text, function () {
        fallback.hidden = true;
        flash(copyAllBtn, 'Copied ' + bibs.length, label);
      }, function () {
        fbText.value = text;
        fbMsg.textContent = 'The clipboard is blocked here. ' + bibs.length + ' entries are selected below; press ⌘/Ctrl C.';
        fallback.hidden = false;
        fbText.focus(); fbText.select();
      });
    });

    /* ---------- list interactions ---------- */
    list.addEventListener('click', function (e) {
      if (e.target.closest('[data-clear]')) { clearAll(); return; }
      var r = e.target.closest('.cat-row'); if (!r) return;
      select(r.getAttribute('data-id'), { hash: true, reveal: true, focus: true });
    });
    hiddenNote.addEventListener('click', function (e) { if (e.target.closest('[data-clear]')) clearAll(); });
    list.addEventListener('keydown', function (e) {
      var steps = { ArrowDown: 1, ArrowUp: -1, PageDown: 8, PageUp: -8, Home: -1e9, End: 1e9 };
      if (!(e.key in steps)) {
        if (e.key === 'Enter' || e.key === ' ') { var r = e.target.closest && e.target.closest('.cat-row'); if (r) { e.preventDefault(); select(r.getAttribute('data-id'), { hash: true, reveal: true }); } }
        return;
      }
      if (!visible.length) return;
      e.preventDefault();
      var sel = S.tab === 'methods' ? S.selM : S.selP, idx = -1;
      for (var i = 0; i < visible.length; i++) if (visible[i].id === sel) { idx = i; break; }
      var ni = idx < 0 ? 0 : Math.max(0, Math.min(visible.length - 1, idx + steps[e.key]));
      select(visible[ni].id, { focus: true, defer: true, hash: true });
    });

    /* ---------- controls ---------- */
    var onSearch = A.debounce(function () {
      var q = searchIn.value.trim();
      S.q = q;
      S.toks = norm(q).split(/\s+/).filter(Boolean);
      if (S.toks.length && (S.sort === 'year-asc' && !S.sortTouched)) { S.sort = 'match'; S.sortAuto = true; sortSel.value = 'match'; }
      else if (!S.toks.length && S.sortAuto) { S.sort = 'year-asc'; S.sortAuto = false; sortSel.value = 'year-asc'; }
      update();
      scroll.scrollTop = 0;
    }, 90);
    searchIn.addEventListener('input', onSearch);
    searchIn.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown' && visible.length) { e.preventDefault(); select(visible[0].id, { focus: true, hash: true }); }
      if (e.key === 'Enter' && visible.length) { e.preventDefault(); select(visible[0].id, { hash: true, reveal: true }); }
    });
    sortSel.addEventListener('change', function () { S.sort = sortSel.value; S.sortTouched = true; S.sortAuto = false; update(); });
    clearBtn.addEventListener('click', clearAll);
    function clearAll() {
      searchIn.value = ''; S.q = ''; S.toks = [];
      if (S.sortAuto) { S.sort = 'year-asc'; S.sortAuto = false; sortSel.value = 'year-asc'; }
      if (S.tab === 'methods') {
        S.kinds = {}; S.hf = 'any'; S.merge = {}; S.core = false; S.yM = MY.slice();
        axisKeys.forEach(function (k) { S.coords[k] = {}; });
      } else S.yP = PY.slice();
      syncYearInputs();
      update();
      var it = (S.tab === 'methods' ? items : pitems).filter(function (x) { return x.id === (S.tab === 'methods' ? S.selM : S.selP); })[0];
      if (it) ensureVisible(it.row);
    }
    function setTab(t, silent) {
      if (S.tab === t) return;
      S.tab = t;
      Object.keys(tabBtns).forEach(function (k) { tabBtns[k].setAttribute('aria-selected', String(k === t)); tabBtns[k].tabIndex = k === t ? 0 : -1; });
      var isM = t === 'methods';
      strip.hidden = !isM;
      [gHF, gMerge, gCo].forEach(function (g) { g.hidden = !isM; });
      if (coreBtn) coreBtn.parentNode.hidden = !isM;
      gPaperNote.hidden = isM;
      searchIn.placeholder = isM ? 'Search ' + methods.length + ' methods: name, author, idea' : 'Search ' + papers.length + ' papers: title, author, claim';
      syncYearInputs();
      update();
      if (!silent) renderCard();
      scroll.scrollTop = 0;
      var it = (isM ? items : pitems).filter(function (x) { return x.id === (isM ? S.selM : S.selP); })[0];
      if (it && it.row && it.row.parentNode) ensureVisible(it.row);
    }

    /* ---------- responsive layout ---------- */
    function doLayout() {
      var w = stage.clientWidth - 2;
      var L = w >= 860 ? 'wide' : w >= 600 ? 'mid' : 'narrow';
      if (L === layout) return;
      var prev = layout;
      layout = L;
      stage.setAttribute('data-layout', L);
      if (L === 'wide') rail.open = true;
      else if (prev === null || prev === 'wide') rail.open = false;
    }
    doLayout();
    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(function () { doLayout(); markOverflow(); }, 60)).observe(stage);
    else window.addEventListener('resize', A.debounce(doLayout, 100));

    /* ---------- external selection ---------- */
    function openExternal(id, tab) {
      tab = tab || 'methods';
      if (tab === 'methods' ? !byId[id] : !pById[id]) return false;
      if (S.tab !== tab) setTab(tab, true);
      select(id, {});
      return true;
    }
    document.addEventListener('atlas:select-method', function (e) { var id = idOf(e.detail); if (id) openExternal(id, 'methods'); });
    /* a deep link lands on the card on a phone (the list would otherwise fill the screen), else on the figure */
    function linkTarget() { return layout === 'narrow' ? cardPane : (el.closest('figure') || el); }
    deepTarget = linkTarget;
    window.addEventListener('hashchange', function () {
      if (selfHash) { selfHash = false; return; }
      var hh = parseHash(); if (!hh) return;
      if (!openExternal(hh.id, hh.tab)) return;
      /* A short trip glides; a long one jumps, since lazy figures mounting on the way would move
         the target under a smooth scroll. Either way the target is then held in place briefly. */
      var t = linkTarget(), far = Math.abs(t.getBoundingClientRect().top) > 1.5 * window.innerHeight;
      if (far || A.reducedMotion()) anchorTo(4000);
      else {
        var t0 = Date.now();
        anchorTok = null;
        try { t.scrollIntoView({ block: 'start', behavior: 'smooth' }); } catch (e) { t.scrollIntoView(true); }
        setTimeout(function () { if (lastInput <= t0) anchorTo(2500); }, 700);
      }
    });
    A.onTheme(function () { update(); });
    /* MathJax loads deferred: once it is ready, typeset whatever card is showing */
    document.addEventListener('atlas:mathjax', function () { typesetNode(card, cardGen); });
    live.push(el);

    /* ---------- initial state: complete at rest ---------- */
    searchIn.placeholder = 'Search ' + methods.length + ' methods: name, author, idea';
    gPaperNote.hidden = true;
    syncYearInputs();
    var hh0 = parseHash();
    if (hh0 && hh0.tab === 'papers' && pById[hh0.id]) { S.selP = hh0.id; S.tab = 'methods'; setTab('papers', true); }
    else if (hh0 && hh0.tab === 'methods' && byId[hh0.id]) S.selM = hh0.id;
    else if (early && byId[early]) S.selM = early;
    update();
    renderCard();
    var it0 = (S.tab === 'methods' ? items : pitems).filter(function (x) { return x.id === (S.tab === 'methods' ? S.selM : S.selP); })[0];
    if (it0 && it0.row) {
      /* centre the open row in the list without moving the page */
      requestAnimationFrame(function () { scroll.scrollTop = Math.max(0, it0.row.offsetTop - scroll.clientHeight * 0.35); });
    }
    /* read-only hook for the headless checks */
    el.__catalog = { open: openExternal, typeset: function () { return tsq; }, state: function () { return { tab: S.tab, selM: S.selM, selP: S.selP, layout: layout, visible: visible.length }; } };
  });

  /* exposed for headless checks of the prose converter */
  Atlas._catalogTex = { texHTML: texHTML, splitClauses: splitClauses, splitMath: splitMath, prose: prose, proseU: proseU };
})();
