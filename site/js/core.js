/* Atlas core: figure registry, theme, MathJax, tooltip, colours, seeded RNG, small linear algebra.
   Classic script (no modules) so the site also works from file:// and inside a sandboxed artifact. */
(function () {
  'use strict';
  var Atlas = (window.Atlas = window.Atlas || {});
  var registry = {};

  /* ---------- figure registry ----------
     A figure is any element with data-figure="name". Modules call Atlas.register(name, init),
     init(el, Atlas) runs once when the element is first near the viewport (lazy). */
  Atlas.register = function (name, init) {
    registry[name] = init;
    if (Atlas._booted) mountAll();
  };
  function mount(el) {
    if (el.__atlasMounted) return;
    var init = registry[el.getAttribute('data-figure')];
    if (!init) return;
    el.__atlasMounted = true;
    try { init(el, Atlas); } catch (err) { console.error('[atlas] figure failed:', el.getAttribute('data-figure'), err); }
  }
  function mountAll() {
    var els = document.querySelectorAll('[data-figure]');
    if (!('IntersectionObserver' in window)) { els.forEach(mount); return; }
    if (!Atlas._io) {
      Atlas._io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (e.isIntersecting) { mount(e.target); Atlas._io.unobserve(e.target); } });
      }, { rootMargin: '600px 0px' });
    }
    els.forEach(function (el) { if (!el.__atlasMounted) Atlas._io.observe(el); });
  }

  /* ---------- in-page links (2026-10-05) ----------
     A smooth scroll down this long page passes dozens of figures that mount on the way and change their heights, so
     the browser's landing point goes stale. Jump instantly, then re-aim while figures near the target settle, and
     leave room for the sticky top bar. */
  function barOffset() {
    var bar = document.querySelector('.topbar');
    return (bar ? bar.getBoundingClientRect().height : 0) + 12;
  }
  function settleOn(id) {
    var el = document.getElementById(id);
    if (!el) return;
    var tries = 0, still = 0;
    (function aim() {
      var off = barOffset();
      var y = el.getBoundingClientRect().top + window.pageYOffset - off;
      window.scrollTo({ top: Math.max(0, y), behavior: 'instant' });
      setTimeout(function () {
        var miss = Math.abs(el.getBoundingClientRect().top - barOffset());
        still = miss <= 2 ? still + 1 : 0;
        if (++tries < 16 && still < 3) aim();
      }, 110);
    })();
  }
  document.addEventListener('click', function (ev) {
    var a = ev.target.closest && ev.target.closest('a[href^="#"]');
    if (!a || ev.defaultPrevented || ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
    var id = decodeURIComponent(a.getAttribute('href').slice(1));
    if (!id || !document.getElementById(id)) return;
    ev.preventDefault();
    if (location.hash !== '#' + id) history.pushState(null, '', '#' + id);
    settleOn(id);
  });
  window.addEventListener('popstate', function () {
    if (location.hash) settleOn(decodeURIComponent(location.hash.slice(1)));
  });
  if (location.hash) window.addEventListener('load', function () { settleOn(decodeURIComponent(location.hash.slice(1))); });

  /* ---------- theme ---------- */
  var THEME_KEY = 'peft-atlas-theme';
  function storedTheme() { try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; } }
  function storeTheme(v) { try { if (v) localStorage.setItem(THEME_KEY, v); else localStorage.removeItem(THEME_KEY); } catch (e) {} }
  Atlas.isDark = function () {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  };
  function applyTheme(v) {
    if (v === 'dark' || v === 'light') document.documentElement.setAttribute('data-theme', v);
    else document.documentElement.removeAttribute('data-theme');
    Atlas._cssCache = {};
    document.dispatchEvent(new CustomEvent('atlas:theme', { detail: { dark: Atlas.isDark() } }));
  }
  Atlas.onTheme = function (fn) { document.addEventListener('atlas:theme', fn); };
  var st = storedTheme();
  if (st) applyTheme(st);
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var h = function () { if (!document.documentElement.getAttribute('data-theme')) applyTheme(null); };
    if (mq.addEventListener) mq.addEventListener('change', h); else if (mq.addListener) mq.addListener(h);
  }
  function wireThemeToggle() {
    document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
      var label = function () { btn.textContent = Atlas.isDark() ? 'Light' : 'Dark'; btn.setAttribute('aria-label', 'Switch to ' + (Atlas.isDark() ? 'light' : 'dark') + ' theme'); };
      label();
      btn.addEventListener('click', function () { var next = Atlas.isDark() ? 'light' : 'dark'; storeTheme(next); applyTheme(next); label(); });
      Atlas.onTheme(label);
    });
  }

  /* ---------- CSS tokens ---------- */
  Atlas._cssCache = {};
  Atlas.css = function (name) {
    if (Atlas._cssCache[name]) return Atlas._cssCache[name];
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    Atlas._cssCache[name] = v;
    return v;
  };
  /* modification-kind -> colour token (series colours are fixed per kind across every figure) */
  Atlas.KINDS = [
    { key: 'additive-weight', name: 'Additive', token: '--c-additive' },
    { key: 'multiplicative-weight', name: 'Multiplicative', token: '--c-multiplicative' },
    { key: 'architectural-insertion', name: 'Architectural', token: '--c-architectural' },
    { key: 'input-or-activation-augmentation', name: 'Learned state', token: '--c-augmentation' },
    { key: 'selective', name: 'Selective', token: '--c-selective' },
    { key: 'optimizer-or-training-procedure', name: 'Optimizer-side', token: '--c-optimizer' },
    { key: 'composition-or-routing', name: 'Composition', token: '--c-composition' },
    { key: 'hybrid', name: 'Hybrid', token: '--c-hybrid' },
  ];
  Atlas.kindColor = function (kind) {
    for (var i = 0; i < Atlas.KINDS.length; i++) if (Atlas.KINDS[i].key === kind) return Atlas.css(Atlas.KINDS[i].token);
    return Atlas.css('--c-hybrid');
  };
  Atlas.kindName = function (kind) {
    for (var i = 0; i < Atlas.KINDS.length; i++) if (Atlas.KINDS[i].key === kind) return Atlas.KINDS[i].name;
    return kind;
  };

  /* ---------- MathJax ---------- */
  Atlas.typeset = function (el) {
    var MJ = window.MathJax;
    if (MJ && MJ.typesetPromise) {
      /* typeset even when the startup pass failed (for example, an extension that did not load):
         the document exists by then, and only the failing formula is lost */
      var run = function () { return MJ.typesetPromise(el ? [el] : undefined); };
      return (MJ.startup && MJ.startup.promise ? MJ.startup.promise : Promise.resolve())
        .then(run, run)
        .then(function () { Atlas.glueMath(el || document); })
        .catch(function (e) { console.warn('[atlas] MathJax', e); });
    }
    Atlas._pendingTypeset = (Atlas._pendingTypeset || []).concat(el ? [el] : []);
    return Promise.resolve();
  };

  /* keep punctuation that follows inline math, and a word that touches it from the left (as in "(IA)\(^3\)"
     or "LoRA\(_{16}\)"), on the same line as the formula */
  Atlas.glueMath = function (root) {
    var scope = root || document;
    var list = scope.querySelectorAll ? scope.querySelectorAll('mjx-container:not([display="true"])') : [];
    Array.prototype.forEach.call(list, function (mj) {
      if (mj.parentNode && mj.parentNode.classList && mj.parentNode.classList.contains('mj-glue')) return;
      var next = mj.nextSibling, prev = mj.previousSibling;
      var m = next && next.nodeType === 3 ? /^[,.;:!?)\]’”'"]+/.exec(next.nodeValue) : null;
      var p = prev && prev.nodeType === 3 ? /\S{1,24}$/.exec(prev.nodeValue) : null;
      if (!m && !p) return;
      var wrap = document.createElement('span');
      wrap.className = 'mj-glue';
      mj.parentNode.insertBefore(wrap, mj);
      if (p) { wrap.appendChild(document.createTextNode(p[0])); prev.nodeValue = prev.nodeValue.slice(0, prev.nodeValue.length - p[0].length); }
      wrap.appendChild(mj);
      if (m) { wrap.appendChild(document.createTextNode(m[0])); next.nodeValue = next.nodeValue.slice(m[0].length); }
    });
  };
  /* render one TeX string to an SVG element (for use inside D3 figures) */
  Atlas.tex = function (tex, display) {
    var MJ = window.MathJax;
    if (MJ && MJ.tex2svg) {
      try { return MJ.tex2svg(tex, { display: !!display }); } catch (e) {}
    }
    var span = document.createElement('span');
    span.textContent = tex;
    return span;
  };

  /* ---------- tooltip ---------- */
  var tipEl = null;
  Atlas.tip = {
    show: function (html, evt) {
      if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'atlas-tip'; tipEl.setAttribute('role', 'tooltip'); document.body.appendChild(tipEl); }
      tipEl.innerHTML = html;
      tipEl.classList.add('on');
      Atlas.tip.move(evt);
    },
    move: function (evt) {
      if (!tipEl || !evt) return;
      var x = evt.clientX, y = evt.clientY, w = tipEl.offsetWidth, hgt = tipEl.offsetHeight;
      var left = Math.min(window.innerWidth - w - 12, x + 14), top = y + 16;
      if (top + hgt > window.innerHeight - 8) top = y - hgt - 12;
      tipEl.style.left = Math.max(8, left) + 'px';
      tipEl.style.top = Math.max(8, top) + 'px';
    },
    hide: function () { if (tipEl) tipEl.classList.remove('on'); },
  };

  /* ---------- small helpers ---------- */
  Atlas.h = function (tag, attrs, children) {
    var el = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'class') el.className = attrs[k];
      else if (k === 'html') el.innerHTML = attrs[k];
      else if (k === 'text') el.textContent = attrs[k];
      else if (k.slice(0, 2) === 'on' && typeof attrs[k] === 'function') el.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== undefined && attrs[k] !== null && attrs[k] !== false) el.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(function (c) { if (c == null) return; el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return el;
  };
  Atlas.esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };
  Atlas.fmtCount = function (n) {
    if (!isFinite(n)) return '—';
    var a = Math.abs(n);
    if (a >= 1e9) return (n / 1e9).toFixed(a >= 1e10 ? 1 : 2) + 'B';
    if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 1 : 2) + 'M';
    if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e4 ? 1 : 2) + 'K';
    return String(Math.round(n));
  };
  Atlas.fmtPct = function (x) {
    if (!isFinite(x)) return '—';
    if (x === 0) return '0%';
    if (x < 0.0001) return (x * 100).toExponential(1) + '%';
    if (x < 0.01) return (x * 100).toFixed(3) + '%';
    return (x * 100).toFixed(2) + '%';
  };
  /* segmented button group: options [{value,label}], returns {el, value(), set(v)} */
  Atlas.seg = function (options, initial, onChange, ariaLabel) {
    var wrap = Atlas.h('div', { class: 'seg', role: 'group', 'aria-label': ariaLabel || 'options' });
    var cur = initial;
    var btns = options.map(function (o) {
      var b = Atlas.h('button', { type: 'button', 'aria-pressed': String(o.value === cur), text: o.label });
      b.addEventListener('click', function () { set(o.value); if (onChange) onChange(o.value); });
      wrap.appendChild(b);
      return b;
    });
    function set(v) { cur = v; btns.forEach(function (b, i) { b.setAttribute('aria-pressed', String(options[i].value === v)); }); }
    return { el: wrap, value: function () { return cur; }, set: set };
  };
  /* labelled range slider: returns {el, input, value()} */
  Atlas.slider = function (opts) {
    var id = opts.id || ('s-' + Math.random().toString(36).slice(2, 8));
    var input = Atlas.h('input', { type: 'range', id: id, min: opts.min, max: opts.max, step: opts.step || 1, value: opts.value });
    var out = Atlas.h('output', { for: id });
    var fmt = opts.format || function (v) { return v; };
    var update = function () { out.textContent = fmt(+input.value); };
    input.addEventListener('input', function () { update(); if (opts.onInput) opts.onInput(+input.value); });
    update();
    var el = Atlas.h('div', { class: 'control' }, [Atlas.h('label', { for: id, text: opts.label }), input, out]);
    return { el: el, input: input, value: function () { return +input.value; }, refresh: update };
  };
  Atlas.debounce = function (fn, ms) { var t; return function () { var a = arguments, s = this; clearTimeout(t); t = setTimeout(function () { fn.apply(s, a); }, ms || 120); }; };
  Atlas.reducedMotion = function () { return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches; };
  /* resize-aware canvas with devicePixelRatio */
  Atlas.canvas = function (parent, heightPx, draw) {
    var c = Atlas.h('canvas'); parent.appendChild(c);
    var ctx = c.getContext('2d');
    function resize() {
      var w = parent.clientWidth, hh = typeof heightPx === 'function' ? heightPx(w) : heightPx, dpr = Math.min(2, window.devicePixelRatio || 1);
      c.style.width = w + 'px'; c.style.height = hh + 'px'; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (draw) draw(ctx, w, hh);
    }
    if ('ResizeObserver' in window) new ResizeObserver(Atlas.debounce(resize, 60)).observe(parent); else window.addEventListener('resize', Atlas.debounce(resize, 100));
    resize();
    return { canvas: c, ctx: ctx, resize: resize };
  };

  /* ---------- seeded RNG ---------- */
  Atlas.rng = function (seed) {
    var a = seed >>> 0 || 1;
    var next = function () { a |= 0; a = (a + 0x6d2b79f5) | 0; var t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    next.normal = function () { var u = 0, v = 0; while (u === 0) u = next(); while (v === 0) v = next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return next;
  };

  /* ---------- small dense linear algebra (row-major arrays of arrays) ---------- */
  var LA = (Atlas.LA = {});
  LA.zeros = function (m, n) { var A = new Array(m); for (var i = 0; i < m; i++) { A[i] = new Float64Array(n); } return A; };
  LA.eye = function (n) { var I = LA.zeros(n, n); for (var i = 0; i < n; i++) I[i][i] = 1; return I; };
  LA.randn = function (m, n, rand, scale) { var A = LA.zeros(m, n), s = scale == null ? 1 : scale; for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) A[i][j] = rand.normal() * s; return A; };
  LA.shape = function (A) { return [A.length, A.length ? A[0].length : 0]; };
  LA.T = function (A) { var m = A.length, n = A[0].length, B = LA.zeros(n, m); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) B[j][i] = A[i][j]; return B; };
  LA.mul = function (A, B) {
    var m = A.length, k = B.length, n = B[0].length, C = LA.zeros(m, n);
    for (var i = 0; i < m; i++) { var Ai = A[i], Ci = C[i]; for (var p = 0; p < k; p++) { var a = Ai[p]; if (a === 0) continue; var Bp = B[p]; for (var j = 0; j < n; j++) Ci[j] += a * Bp[j]; } }
    return C;
  };
  LA.add = function (A, B, beta) { var b = beta == null ? 1 : beta, m = A.length, n = A[0].length, C = LA.zeros(m, n); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) C[i][j] = A[i][j] + b * B[i][j]; return C; };
  LA.scale = function (A, s) { return A.map(function (r) { return r.map(function (v) { return v * s; }); }); };
  LA.hadamard = function (A, B) { var m = A.length, n = A[0].length, C = LA.zeros(m, n); for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) C[i][j] = A[i][j] * B[i][j]; return C; };
  LA.kron = function (A, B) {
    var ma = A.length, na = A[0].length, mb = B.length, nb = B[0].length, C = LA.zeros(ma * mb, na * nb);
    for (var i = 0; i < ma; i++) for (var j = 0; j < na; j++) { var a = A[i][j]; for (var p = 0; p < mb; p++) for (var q = 0; q < nb; q++) C[i * mb + p][j * nb + q] = a * B[p][q]; }
    return C;
  };
  LA.diag = function (v) { var n = v.length, D = LA.zeros(n, n); for (var i = 0; i < n; i++) D[i][i] = v[i]; return D; };
  LA.frob = function (A) { var s = 0; for (var i = 0; i < A.length; i++) for (var j = 0; j < A[i].length; j++) s += A[i][j] * A[i][j]; return Math.sqrt(s); };
  /* symmetric eigenvalues by cyclic Jacobi (fine for n <= ~96) */
  LA.symEig = function (S) {
    var n = S.length, A = S.map(function (r) { return Float64Array.from(r); }), V = LA.eye(n);
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0;
      for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-22) break;
      for (p = 0; p < n; p++) for (q = p + 1; q < n; q++) {
        var apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        var theta = (A[q][q] - A[p][p]) / (2 * apq);
        var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        var c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (var k = 0; k < n; k++) { var akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (k = 0; k < n; k++) { var apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (k = 0; k < n; k++) { var vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    var vals = []; for (var i = 0; i < n; i++) vals.push(A[i][i]);
    return { values: vals, vectors: V };
  };
  /* singular values, descending */
  LA.svals = function (A) {
    var m = A.length, n = A[0].length;
    var G = m >= n ? LA.mul(LA.T(A), A) : LA.mul(A, LA.T(A));
    return LA.symEig(G).values.map(function (v) { return Math.sqrt(Math.max(0, v)); }).sort(function (a, b) { return b - a; });
  };
  LA.numRank = function (sv, rtol) { var tol = (rtol || 1e-6) * (sv[0] || 0); var r = 0; sv.forEach(function (s) { if (s > tol) r++; }); return r; };
  /* random orthogonal matrix via Gram-Schmidt */
  LA.randOrth = function (n, rand) {
    var Q = LA.randn(n, n, rand);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < i; j++) { var d = 0; for (var k = 0; k < n; k++) d += Q[i][k] * Q[j][k]; for (k = 0; k < n; k++) Q[i][k] -= d * Q[j][k]; }
      var nr = 0; for (k = 0; k < n; k++) nr += Q[i][k] * Q[i][k]; nr = Math.sqrt(nr); for (k = 0; k < n; k++) Q[i][k] /= nr;
    }
    return Q;
  };

  /* ---------- data access ---------- */
  Atlas.data = function () { return window.ATLAS_DATA || { methods: [], papers: [], edges: [], axes: [] }; };
  Atlas.method = function (id) { var ms = Atlas.data().methods || []; for (var i = 0; i < ms.length; i++) if (ms[i].id === id) return ms[i]; return null; };

  /* ---------- wide tables (2026-10-05) ----------
     On screens with a margin column, a table wider than the text borrows the margin instead of scrolling sideways.
     Margin notes placed just before such a table move to just after it, so they don't collide. A table keeps its
     sideways scroll when it would still not fit, leave the window, or be pushed below an earlier note. */
  function widenTables() {
    if (!window.matchMedia('(min-width: 1100px)').matches) return;
    var vw = document.documentElement.clientWidth;
    document.querySelectorAll('.table-wrap').forEach(function (w) {
      if (w.__wide !== undefined || w.offsetParent === null) return;      // done already, or hidden (closed proof)
      if (w.scrollWidth - w.clientWidth <= 4) { w.__wide = false; return; }
      var notes = [];
      for (var p = w.previousElementSibling; p && p.matches && p.matches('aside.note'); p = p.previousElementSibling) notes.unshift(p);
      var top0 = w.getBoundingClientRect().top, after = w;
      notes.forEach(function (n) { after.after(n); after = n; });
      w.classList.add('wide');
      var r = w.getBoundingClientRect();
      var ok = w.scrollWidth - w.clientWidth <= 4 && r.right <= vw - 12 && r.top - top0 < 40;
      if (!ok) { w.classList.remove('wide'); notes.forEach(function (n) { w.before(n); }); }
      w.__wide = ok;
    });
  }
  function widenWhenTypeset() {
    var MJ = window.MathJax;
    var typeset = MJ && MJ.startup && MJ.startup.promise ? MJ.startup.promise : Promise.resolve();
    typeset.then(function () { return document.fonts ? document.fonts.ready : null; })
      .then(function () { setTimeout(widenTables, 60); }, function () { setTimeout(widenTables, 60); });
  }
  if (document.readyState === 'complete') widenWhenTypeset(); else window.addEventListener('load', widenWhenTypeset);
  document.addEventListener('toggle', function (ev) { if (ev.target.open) setTimeout(widenTables, 60); }, true);

  /* ---------- the film: chapter links seek the player; "Watch the film" starts it ---------- */
  function wireFilm() {
    var v = document.querySelector('#film video');
    if (!v) return;
    var links = Array.prototype.slice.call(document.querySelectorAll('#film [data-t]'));
    function play() { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
    function seek(t) {
      play();
      if (v.readyState >= 1) v.currentTime = t; else v.addEventListener('loadedmetadata', function () { v.currentTime = t; }, { once: true });
    }
    links.forEach(function (a) {
      a.addEventListener('click', function (e) { e.preventDefault(); seek(+a.getAttribute('data-t')); });
    });
    document.querySelectorAll('[data-play-film]').forEach(function (a) { a.addEventListener('click', play); });
    v.addEventListener('timeupdate', function () {
      var cur = null;
      links.forEach(function (a) { if (v.currentTime + 0.5 >= +a.getAttribute('data-t')) cur = a; });
      links.forEach(function (a) { a.classList.toggle('on', a === cur); });
    });
  }

  /* ---------- boot ---------- */
  function boot() {
    Atlas._booted = true;
    wireThemeToggle();
    wireFilm();
    mountAll();
    document.dispatchEvent(new CustomEvent('atlas:ready'));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
