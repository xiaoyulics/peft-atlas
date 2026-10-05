/* Hero canvas: "Fine-tuning moves a point."
   Weight space Θ drawn as a calm nautical chart around the pretrained point θ₀.
   Through θ₀ pass the image germs of several methods (Exposé II / VII of theory/framework.md):
     LoRA      W₀ + M_{≤r}: a cone with apex at θ₀; drawn as the rank ≤ 1 cone x² = y² + z² of the
               symmetric 2×2 slice (rulings + rims, in perspective).
     OFT       W₀Rᵀ: the orbit of θ₀ under rotations. For one weight row it is the circle of radius |θ₀|
               centred at 0 ∈ Θ, so the arc through θ₀ is centred at the chart's "0".
     (IA)³     diag(ℓ)W₀: the torus orbit ℝ^× θ₀; its closure is the line through 0 and θ₀.
     BitFit    a translation of the bias block: a coordinate line, parallel to the graticule.
     Full FT   all of Θ; its point follows the gradient flow of the bathymetric "loss" L drawn as contours.
   Non-examples (dashed):
     Prefix    unpointed: a curve that never passes through θ₀.
     QLoRA     LoRA pointed at κθ₀ (κ = round to the graticule, which is the quantiser's grid through 0):
               the same cone translated by the defect e = κθ₀ − θ₀ (additive families commute with translation).
   Everything shown (the grid node κθ₀, the arrow e, the flow line, the orbit radius) is computed from the
   geometry below; nothing is placed by hand except the chart's design parameters. */
(function () {
  'use strict';

  var TAU = Math.PI * 2;
  var DEG = Math.PI / 180;

  /* ------------------------------------------------------------------------------------------
     Chart geometry in units of R (the composition radius). θ₀ sits at the world origin; y points up.
     ------------------------------------------------------------------------------------------ */
  var GAMMA = -42 * DEG;                       // direction from θ₀ to the zero weight 0 ∈ Θ
  var RHO = 1.3;                               // |θ₀ − 0|: the OFT orbit radius
  var ZERO = { x: RHO * Math.cos(GAMMA), y: RHO * Math.sin(GAMMA) };
  var QSTEP = 0.40;                            // quantiser grid spacing; the graticule is 0 + QSTEP·ℤ²
  function kappa(p) {                          // κ: round every coordinate to the grid through 0
    return {
      x: ZERO.x + QSTEP * Math.round((p.x - ZERO.x) / QSTEP),
      y: ZERO.y + QSTEP * Math.round((p.y - ZERO.y) / QSTEP),
    };
  }
  var KT0 = kappa({ x: 0, y: 0 });             // κθ₀; the defect is e = κθ₀ − θ₀ = KT0 (θ₀ = 0 here)
  var BETA0 = Math.atan2(-ZERO.y, -ZERO.x);    // angle of θ₀ seen from 0 (on the OFT circle)

  /* LoRA cone: half-angle 45° (|radial| = |axial|, i.e. x² = y² + z², det S = 0 on Sym₂). */
  var CONE = { h: 0.31, axis: 93 * DEG, tilt: 21 * DEG, f: 5.5, n: 18, nq: 10, spinPeriod: 150 };
  var BASIS = (function () {
    var a = CONE.axis, p = CONE.tilt;
    var A = [Math.cos(a) * Math.cos(p), Math.sin(a) * Math.cos(p), -Math.sin(p)];  // top tips away from the viewer
    var n = Math.hypot(A[0], A[1]);
    var U = [A[1] / n, -A[0] / n, 0];                                                // U = A × ẑ, normalised
    var V = [A[1] * U[2] - A[2] * U[1], A[2] * U[0] - A[0] * U[2], A[0] * U[1] - A[1] * U[0]]; // V = A × U
    return { A: A, U: U, V: V };
  })();
  function cone3(t, phi) {
    var c = Math.cos(phi), s = Math.sin(phi), h = CONE.h * t, B = BASIS;
    return [h * (B.A[0] + c * B.U[0] + s * B.V[0]), h * (B.A[1] + c * B.U[1] + s * B.V[1]), h * (B.A[2] + c * B.U[2] + s * B.V[2])];
  }
  function coneWorld(t, phi, apex) {          // perspective about the apex, which lies on the chart plane (Z = 0)
    var P = cone3(t, phi), k = CONE.f / (CONE.f - P[2]);
    return { x: apex.x + P[0] * k, y: apex.y + P[1] * k, z: P[2] / CONE.h };
  }
  /* the ruling a travelling point rides: the one whose rim point heads closest to a target screen bearing */
  function pickRuling(sign, bearing) {
    var best = 0, bd = 1e9;
    for (var i = 0; i < 360; i++) {
      var phi = i * DEG, p = coneWorld(sign, phi, { x: 0, y: 0 });
      var d = Math.abs(Math.atan2(Math.sin(Math.atan2(p.y, p.x) - bearing), Math.cos(Math.atan2(p.y, p.x) - bearing)));
      if (p.z > -0.2 && d < bd) { bd = d; best = phi; }
    }
    return best;
  }
  var PHI_LORA = pickRuling(1, 58 * DEG);
  var PHI_QLORA = pickRuling(-1, -64 * DEG);

  /* Bathymetry: a smooth field L on the chart (a stand-in loss). Contours are its level sets; the full
     fine-tuning point follows its normalised gradient flow from θ₀. */
  var BUMPS = [ // [cx, cy, amplitude, sigma_u, sigma_v, rotation°]: anisotropic Gaussians
    [1.10, 0.32, -0.80, 0.70, 0.42, 18],   // the deep the full fine-tuning point drains into
    [0.30, -0.42, 0.40, 0.36, 0.28, -30],
    [-0.55, 0.85, 0.28, 0.45, 0.35, 40],
    [1.85, -0.60, 0.38, 0.60, 0.45, -20],
    [-1.9, -0.75, -0.70, 0.85, 0.55, 25],
    [-3.0, 0.75, -0.55, 0.90, 0.60, -35],
    [0.15, 1.30, -0.32, 0.60, 0.35, 5],
    [2.4, 0.9, -0.30, 0.50, 0.70, 0],
  ].map(function (b) { var r = b[5] * DEG; return { x: b[0], y: b[1], a: b[2], iu: 1 / (b[3] * b[3]), iv: 1 / (b[4] * b[4]), c: Math.cos(r), s: Math.sin(r) }; });
  var WAVES = [[1.7, 0.6, 1.3, 0.05], [-0.9, 2.1, 0.4, 0.045], [2.6, -1.4, 2.2, 0.03], [0.5, 3.1, 0.9, 0.02]]; // [kx, ky, phase, amp]
  function Lf(x, y) {
    var v = 0, i, b, w, dx, dy, u, q;
    for (i = 0; i < BUMPS.length; i++) {
      b = BUMPS[i]; dx = x - b.x; dy = y - b.y; u = b.c * dx + b.s * dy; q = -b.s * dx + b.c * dy;
      v += b.a * Math.exp(-0.5 * (u * u * b.iu + q * q * b.iv));
    }
    for (i = 0; i < WAVES.length; i++) { w = WAVES[i]; v += w[3] * Math.sin(w[0] * x + w[1] * y + w[2]); }
    return v;
  }
  function gradL(x, y) {                       // analytic gradient of Lf
    var gx = 0, gy = 0, i, b, w, dx, dy, u, q, e, du, dq, c;
    for (i = 0; i < BUMPS.length; i++) {
      b = BUMPS[i]; dx = x - b.x; dy = y - b.y; u = b.c * dx + b.s * dy; q = -b.s * dx + b.c * dy;
      e = b.a * Math.exp(-0.5 * (u * u * b.iu + q * q * b.iv)); du = -u * b.iu * e; dq = -q * b.iv * e;
      gx += du * b.c - dq * b.s; gy += du * b.s + dq * b.c;
    }
    for (i = 0; i < WAVES.length; i++) { w = WAVES[i]; c = w[3] * Math.cos(w[0] * x + w[1] * y + w[2]); gx += c * w[0]; gy += c * w[1]; }
    return [gx, gy];
  }
  var FLOW = (function () {                    // arc-length parametrised descent line from θ₀
    var p = { x: 0, y: 0 }, pts = [p], h = 0.01, len = 0;
    for (var i = 0; i < 400 && len < 1.6; i++) {
      var g = gradL(p.x, p.y), n = Math.hypot(g[0], g[1]);
      if (n < 1e-5) break;
      var q = { x: p.x - h * g[0] / n, y: p.y - h * g[1] / n };
      if (Lf(q.x, q.y) > Lf(p.x, p.y) - 1e-7) break;
      pts.push(q); p = q; len += h;
    }
    return pts;
  })();
  var LEVEL_STEP = 0.075;                      // contour interval of L
  var RISE_PERIOD = 26;                        // seconds for the "sea" to rise by one contour interval

  /* Prefix tuning: a smooth curve that misses θ₀ (Catmull–Rom through these points). */
  var PREFIX_CTRL = [[-0.66, 0.66], [-0.40, 0.31], [-0.26, -0.02], [-0.37, -0.36], [-0.60, -0.66]];
  var PREFIX = (function () {
    var P = PREFIX_CTRL, out = [];
    for (var k = 0; k < P.length - 1; k++) {
      var p0 = P[Math.max(0, k - 1)], p1 = P[k], p2 = P[k + 1], p3 = P[Math.min(P.length - 1, k + 2)];
      for (var j = 0; j < 16; j++) {
        var t = j / 16, t2 = t * t, t3 = t2 * t;
        out.push({
          x: 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
          y: 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
        });
      }
    }
    out.push({ x: P[P.length - 1][0], y: P[P.length - 1][1] });
    return out;
  })();
  var PREFIX_GAP = (function () {              // closest point of the prefix curve to θ₀ (its distance is > 0)
    var best = PREFIX[0], bd = 1e9;
    PREFIX.forEach(function (p) { var d = Math.hypot(p.x, p.y); if (d < bd) { bd = d; best = p; } });
    return { p: best, d: bd };
  })();

  /* polyline helpers in world units */
  function arcLengths(pts) { var acc = [0]; for (var i = 1; i < pts.length; i++) acc.push(acc[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)); return acc; }
  function atLength(pts, acc, s) {             // point at fraction s of the arc length
    var L = acc[acc.length - 1] * Math.max(0, Math.min(1, s)), i = 1;
    while (i < acc.length - 1 && acc[i] < L) i++;
    var seg = acc[i] - acc[i - 1] || 1, u = (L - acc[i - 1]) / seg;
    return { x: pts[i - 1].x + u * (pts[i].x - pts[i - 1].x), y: pts[i - 1].y + u * (pts[i].y - pts[i - 1].y) };
  }
  var FLOW_ACC = arcLengths(FLOW), PREFIX_ACC = arcLengths(PREFIX);

  function ease(u) { u = Math.max(0, Math.min(1, u)); return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  function rgba(col, a) {
    var m, s = String(col || '').trim();
    if ((m = /^#([0-9a-f]{3})$/i.exec(s))) {
      var r = parseInt(m[1][0] + m[1][0], 16), g = parseInt(m[1][1] + m[1][1], 16), b = parseInt(m[1][2] + m[1][2], 16);
      return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
    }
    if ((m = /^#([0-9a-f]{6})/i.exec(s))) { var n = parseInt(m[1], 16); return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')'; }
    if ((m = /^rgba?\(([^)]+)\)$/i.exec(s))) { var p = m[1].split(/[\s,\/]+/).filter(Boolean); return 'rgba(' + p[0] + ',' + p[1] + ',' + p[2] + ',' + a + ')'; }
    return s;
  }
  function rgbOf(col) {
    var c = rgba(col, 1), m = /rgba\((\d+),(\d+),(\d+)/.exec(c);
    return m ? [+m[1], +m[2], +m[3]] : [14, 115, 133];
  }

  function injectCss() {
    if (document.getElementById('css-hero')) return;
    var st = document.createElement('style');
    st.id = 'css-hero';
    st.textContent = [
      '.hero > .sea-ctl{position:absolute;right:16px;bottom:12px;z-index:3;display:flex;align-items:center;gap:.7rem;',
      'max-width:calc(100% - 32px);font-family:var(--f-ui);font-size:12px;letter-spacing:0;color:var(--ink-2);pointer-events:none}',
      '.hero > .sea-ctl .sea-now{display:block;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;',
      'padding:.1rem .5rem;border-radius:999px;background:color-mix(in srgb,var(--paper) 72%,transparent)}',
      '.hero > .sea-ctl .sea-now .cap{display:none}',
      '.hero > .sea-ctl .sea-now .cap.on{display:inline}',
      '.hero > .sea-ctl .sea-now .long,.hero > .sea-ctl .sea-now .short{font-family:var(--f-body);font-size:13.5px;color:var(--ink-2)}',
      '.hero > .sea-ctl .sea-now mjx-container{font-size:112%!important;margin:0 .05em}',
      '.hero > .sea-ctl .sea-now .sw{width:.55rem;height:.55rem;border-radius:50%;display:inline-block;margin-right:.45rem;vertical-align:-.02em}',
      '.hero > .sea-ctl .sea-now .sw.dash{background:transparent!important;border:1.5px dashed currentColor}',
      '.hero > .sea-ctl .sea-now b{font-weight:500;color:var(--ink);letter-spacing:.06em;text-transform:uppercase;margin-right:.15em}',
      '.hero > .sea-ctl button{pointer-events:auto;flex:none;font:inherit;font-weight:500;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);',
      'background:color-mix(in srgb,var(--paper) 82%,transparent);border:1px solid var(--rule);border-radius:999px;padding:.22rem .65rem;cursor:pointer}',
      '.hero > .sea-ctl button:hover{color:var(--ink);border-color:var(--ink-3)}',
      '.hero > .sea-ctl .short{display:none}',
      '@media (max-width:640px){.hero > .sea-ctl .long{display:none}.hero > .sea-ctl .short{display:inline}}',
      /* below 980px the text fills the width, so the chart gets its own band under the buttons instead of sitting behind them */
      '@media (max-width:979.98px){.hero.hero-live > .hero-inner{padding-bottom:calc(clamp(3rem,7vw,5.5rem) + 128px)}}',
      '[data-figure="hero"].hero-fallback{position:relative;min-height:440px;overflow:hidden;border:1px solid var(--rule);border-radius:var(--radius)}',
    ].join('');
    document.head.appendChild(st);
  }

  Atlas.register('hero', function (el, A) {
    injectCss();

    /* ---- host and canvas: el is normally the <canvas class="sea">; tolerate a plain container ---- */
    var canvas, host;
    if (el.tagName === 'CANVAS') { canvas = el; host = el.parentNode || el; }
    else {
      host = el; el.classList.add('hero-fallback');
      canvas = A.h('canvas', { class: 'sea', 'aria-hidden': 'true' });
      canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
      el.appendChild(canvas);
    }
    if (!canvas.getContext) return;
    var ctx = canvas.getContext('2d');
    if (host !== el && host.classList) host.classList.add('hero-live');

    /* ---- test hooks: #hero-t=12.5 freezes the clock at 12.5 s; #hero-still forces the reduced-motion frame ---- */
    var hash = String(location.hash || '');
    var frozenT = /hero-t=([\d.]+)/.exec(hash); frozenT = frozenT ? +frozenT[1] : null;
    var forceStill = /hero-still/.test(hash);
    var mqReduce = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
    function still() { return forceStill || A.reducedMotion(); }

    /* ---- data: names and kinds from ATLAS_DATA when present ---- */
    function name(id, fb) { var m = id && A.method(id); return (m && m.name) || fb; }
    function kind(id, fb) { var m = id && A.method(id); return (m && m.modification_kind) || fb; }

    /* ---- layout ---- */
    var LAY = { W: 0, H: 0, dpr: 1, fx: 0, fy: 0, R: 100, G: 1, phone: false, fs: 13, fsm: 10.5 };
    function toS(p) { return { x: LAY.fx + LAY.R * p.x, y: LAY.fy - LAY.R * p.y }; }
    function toW(x, y) { return { x: (x - LAY.fx) / LAY.R, y: (LAY.fy - y) / LAY.R }; }

    /* ---- colour tokens (re-read on every theme change) ---- */
    var T = {};
    /* font stacks come from the type tokens at draw time (cached by A.css until the theme changes), so the canvas
       follows the page fonts */
    function readFonts() {
      T.ui = A.css('--f-ui') || 'sans-serif';
      T.mono = A.css('--f-mono') || 'monospace';
      T.display = A.css('--f-display') || 'serif';
      T.body = A.css('--f-body') || 'serif';
    }
    function readTokens() {
      A._cssCache = {};
      T.paper = A.css('--paper') || '#f2f4f1';
      T.ink = A.css('--ink') || '#13233a';
      T.ink2 = A.css('--ink-2') || '#46566a';
      T.ink3 = A.css('--ink-3') || '#6f7d8c';
      T.rule = A.css('--rule') || '#c7d0cc';
      T.tide = A.css('--tide') || '#0e7385';
      T.seal = A.css('--seal') || '#a3392b';
      readFonts();
      T.dark = A.isDark();
      GERMS.forEach(function (g) { g.color = g.colorToken ? (A.css(g.colorToken) || T.ink2) : A.kindColor(g.kind); });
    }

    /* ---- drawing primitives ---- */
    function setDash(d) { ctx.setLineDash(d || []); }
    /* stroke a world polyline whose opacity fades with distance from θ₀ (it is a germ at θ₀) */
    function strokeGerm(pts, color, alpha, width, dash, fadeR, floor) {
      if (alpha <= 0.002) return;
      var S = pts.map(toS), acc = 0, t0 = toS({ x: 0, y: 0 });
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'butt'; setDash(dash);
      for (var i = 1; i < S.length; i++) {
        var a = S[i - 1], b = S[i], len = Math.hypot(b.x - a.x, b.y - a.y);
        var d = Math.hypot((a.x + b.x) / 2 - t0.x, (a.y + b.y) / 2 - t0.y) / (LAY.R * (fadeR || 1.2));
        var f = Math.max(floor || 0, 1 - d * d);
        if (f > 0.004) {
          ctx.globalAlpha = alpha * f * LAY.G;
          if (dash) ctx.lineDashOffset = acc;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
        acc += len;
      }
      ctx.globalAlpha = 1; setDash(null); ctx.lineDashOffset = 0;
    }
    function strokePlain(pts, color, alpha, width, dash) {
      if (alpha <= 0.002 || pts.length < 2) return;
      var S = pts.map(toS);
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; setDash(dash);
      ctx.globalAlpha = alpha * LAY.G;
      ctx.beginPath(); ctx.moveTo(S[0].x, S[0].y);
      for (var i = 1; i < S.length; i++) ctx.lineTo(S[i].x, S[i].y);
      ctx.stroke(); ctx.globalAlpha = 1; setDash(null);
    }
    function strokePlainSeg(a, b, color, alpha, width, dash, offset) {
      if (alpha <= 0.002) return;
      var A2 = toS(a), B2 = toS(b);
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'butt'; setDash(dash);
      if (dash) ctx.lineDashOffset = offset || 0;
      ctx.globalAlpha = alpha * LAY.G;
      ctx.beginPath(); ctx.moveTo(A2.x, A2.y); ctx.lineTo(B2.x, B2.y); ctx.stroke();
      ctx.globalAlpha = 1; setDash(null); ctx.lineDashOffset = 0;
    }
    function text(str, x, y, font, color, alpha, align, track, halo) {
      if (alpha <= 0.002) return;
      ctx.font = font; ctx.textAlign = align || 'left'; ctx.textBaseline = 'alphabetic';
      if ('letterSpacing' in ctx) ctx.letterSpacing = (track || 0) + 'px';
      if (halo) {
        ctx.globalAlpha = Math.min(1, alpha * 1.25) * LAY.G * 0.9;
        ctx.strokeStyle = T.paper; ctx.lineWidth = halo; ctx.lineJoin = 'round';
        ctx.strokeText(str, x, y);
      }
      ctx.globalAlpha = alpha * LAY.G; ctx.fillStyle = color;
      ctx.fillText(str, x, y);
      ctx.globalAlpha = 1;
      if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
    }
    /* descriptor strings: UI-face caps, with $…$ segments set as serif-italic math ('_x' = subscript x),
       so the symbols match the page's math rather than the label face. */
    function parseRich(str) {
      var parts = [];
      String(str).split('$').forEach(function (seg, i) {
        if (!seg) return;
        if (i % 2 === 0) { parts.push({ t: seg, k: 'ui' }); return; }
        var re = /_(.)/g, last = 0, m;
        while ((m = re.exec(seg))) {
          if (m.index > last) parts.push({ t: seg.slice(last, m.index), k: 'math' });
          parts.push({ t: m[1], k: 'sub' }); last = re.lastIndex;
        }
        if (last < seg.length) parts.push({ t: seg.slice(last), k: 'math' });
      });
      return parts;
    }
    function richFont(k, fs) {
      if (k === 'ui') return '500 ' + fs + 'px ' + T.ui;
      if (k === 'sub') return '400 ' + Math.round(fs * 0.95) + 'px ' + T.body;
      return 'italic 400 ' + Math.round(fs * 1.4) + 'px ' + T.body;
    }
    function richAdvance(p, fs, track) {
      if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
      ctx.font = richFont(p.k, fs);
      return ctx.measureText(p.t).width + (p.k === 'ui' ? track * p.t.length : p.k === 'math' ? 1 : 0.5);
    }
    function richWidth(str, fs, track) { return parseRich(str).reduce(function (w, p) { return w + richAdvance(p, fs, track); }, 0); }
    function richText(str, x, y, fs, color, alpha, align, track) {
      var parts = parseRich(str), w = richWidth(str, fs, track);
      var cx = align === 'right' ? x - w : align === 'center' ? x - w / 2 : x;
      parts.forEach(function (p) {
        var adv = richAdvance(p, fs, track);
        text(p.t, cx, y + (p.k === 'sub' ? Math.round(fs * 0.32) : 0), richFont(p.k, fs), color, alpha, 'left', p.k === 'ui' ? track : 0, 3);
        cx += adv;
      });
    }
    /* the fade of drawMask() at a point, evaluated analytically (radial falloff times the fade toward the text) */
    function maskAt(x, y) {
      var R = LAY.R, cx = LAY.fx + 0.1 * R, r0 = R * (LAY.band ? 0.9 : 1.15), r1 = R * (LAY.band ? 2.6 : 3.0);
      var d = Math.hypot(x - cx, y - LAY.fy), rad = d <= r0 ? 1 : d >= r1 ? 0.22 : 1 - 0.78 * (d - r0) / (r1 - r0), lin;
      if (LAY.band) { var y1 = LAY.top != null ? LAY.top + 6 : LAY.H * 0.62, y0 = Math.max(0, y1 - 150); lin = y >= y1 ? 1 : y <= y0 ? 0.14 : 0.14 + 0.86 * (y - y0) / (y1 - y0); }
      else { var x0 = LAY.W * 0.30, x1 = LAY.W * 0.545; lin = x >= x1 ? 1 : x <= x0 ? 0.1 : 0.1 + 0.9 * (x - x0) / (x1 - x0); }
      return rad * lin;
    }
    var LBOX = [];                                     // label boxes placed in the current frame
    function label(g, anchor, align, alpha, dy) {
      var x = anchor.x, y = anchor.y + (dy || 0);
      /* names: UI face, medium, natural case; descriptors: UI caps at 12px with 0.06em tracking */
      var fs = LAY.fs, fd = Math.max(12, fs - 1), trN = +(0.01 * fs).toFixed(2), trD = +(0.06 * fd).toFixed(2), nameFont = '500 ' + fs + 'px ' + T.ui;
      if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
      ctx.font = nameFont;
      var wName = ctx.measureText(g.label).width + trN * g.label.length;
      var wDesc = LAY.band ? 0 : richWidth(g.desc, fd, trD);
      var w = Math.max(wName, wDesc) + 4;
      if (align === 'left') x = Math.min(x, LAY.W - 10 - w);
      else if (align === 'right') x = Math.max(x, 10 + w);
      else x = Math.max(10 + w / 2, Math.min(LAY.W - 10 - w / 2, x));
      var yLo = LAY.band ? LAY.top + fs + 6 : fs + 8, yHi = LAY.band ? LAY.H - 48 : LAY.H - fs - 14;   // in the band: above the strip
      y = Math.max(yLo, Math.min(yHi, y));
      /* keep clear of the labels already set this frame: try a line below, then above, then two lines away */
      var left = align === 'left' ? x : align === 'right' ? x - w : x - w / 2, below = LAY.band ? 4 : fd + 8;
      var boxAt = function (yy) { return { l: left - 3, r: left + w + 3, t: yy - fs - 1, b: yy + below }; };
      var hits = function (b) { return LBOX.some(function (o) { return b.l < o.r && b.r > o.l && b.t < o.b && b.b > o.t; }); };
      /* in the band a label may rise above the text's bottom edge where no text line or button sits beside it */
      var fits = function (yy) {
        if (yy > yHi || yy < fs + 8) return false;
        if (yy >= yLo) return true;
        if (!LAY.band) return false;
        var bb = boxAt(yy), tr = textRight(bb.t, bb.b);
        return tr == null || tr + 8 < bb.l;
      };
      if (hits(boxAt(y))) {
        var step = fs + below + 3, tries = [step, -step, 2 * step, -2 * step, 3 * step, -3 * step];
        for (var k = 0; k < tries.length; k++) { var yy = y + tries[k]; if (fits(yy) && !hits(boxAt(yy))) { y = yy; break; } }
      }
      LBOX.push(boxAt(y));
      alpha = Math.min(1 / LAY.G, alpha * Math.max(0.9, maskAt(left + w / 2, y - fs / 2)) / LAY.G);   // labels are exempt from the band's overall dimming
      text(g.label, x, y, nameFont, g.color, alpha, align, trN, 3);
      if (!LAY.band) richText(g.desc, x, y + fd + 4, fd, T.ink2, alpha * 0.92, align, trD);
    }
    function dot(p, r, color, alpha, ring) {
      var s = toS(p);
      ctx.globalAlpha = alpha * LAY.G;
      if (ring) { ctx.beginPath(); ctx.arc(s.x, s.y, r + 1.6, 0, TAU); ctx.fillStyle = T.paper; ctx.fill(); }
      ctx.beginPath(); ctx.arc(s.x, s.y, r, 0, TAU); ctx.fillStyle = color; ctx.fill();
      ctx.globalAlpha = 1;
    }
    function arrow(from, to, color, alpha, width) {
      var a = toS(from), b = toS(to), ang = Math.atan2(b.y - a.y, b.x - a.x), hl = 6.5;
      ctx.globalAlpha = alpha * LAY.G; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; setDash(null);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x - Math.cos(ang) * 1.2, b.y - Math.sin(ang) * 1.2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(b.x - hl * Math.cos(ang - 0.42), b.y - hl * Math.sin(ang - 0.42));
      ctx.lineTo(b.x, b.y);
      ctx.lineTo(b.x - hl * Math.cos(ang + 0.42), b.y - hl * Math.sin(ang + 0.42));
      ctx.stroke(); ctx.globalAlpha = 1;
    }

    /* ---- the germs ---- */
    function line(f, a, b, n) { var out = []; for (var i = 0; i <= n; i++) out.push(f(a + (b - a) * i / n)); return out; }
    function oftPt(beta) { return { x: ZERO.x + RHO * Math.cos(beta), y: ZERO.y + RHO * Math.sin(beta) }; }
    function ia3Pt(l) { return { x: ZERO.x * (1 - l), y: ZERO.y * (1 - l) }; }   // 0 + ℓ(θ₀ − 0)

    function drawCone(apex, alpha, dashed, spin, nRulings) {
      var col = this.color, n = nRulings, i, k, phi, pts, front;
      for (var sign = -1; sign <= 1; sign += 2) {
        for (i = 0; i < n; i++) {
          phi = TAU * i / n + spin;
          var rim = coneWorld(sign, phi, apex);
          front = 0.35 + 0.65 * clamp01((rim.z + 0.45) / 0.9);
          var prev = coneWorld(0, phi, apex), acc = 0;
          for (k = 1; k <= 6; k++) {
            var cur = coneWorld(sign * k / 6, phi, apex), ramp = Math.min(1, (k - 0.5) / 3);
            strokePlainSeg(prev, cur, col, alpha * front * (0.18 + 0.82 * ramp), dashed ? 0.9 : 0.95, dashed ? [3, 4] : null, acc);
            acc += Math.hypot(cur.x - prev.x, cur.y - prev.y) * LAY.R; prev = cur;
          }
        }
        /* rim: front half brighter */
        var rimPts = [];
        for (k = 0; k <= 72; k++) rimPts.push(coneWorld(sign, TAU * k / 72, apex));
        for (k = 1; k < rimPts.length; k++) {
          var z = (rimPts[k].z + rimPts[k - 1].z) / 2;
          strokePlain([rimPts[k - 1], rimPts[k]], col, alpha * (0.45 + 0.55 * clamp01((z + 0.45) / 0.9)), dashed ? 0.9 : 1.1, dashed ? [3, 4] : null);
        }
      }
    }

    var GERMS = [
      {
        id: 'lora', dataId: 'lora', fb: 'LoRA', kindFb: 'additive-weight',
        desc: 'CONE · APEX AT $θ_0$', sentence: 'a cone with its apex at \\(\\theta_0\\)', short: 'cone, apex at \\(\\theta_0\\)',
        draw: function (alpha, clock) { drawCone.call(this, { x: 0, y: 0 }, alpha, false, spinAt(clock), CONE.n); },
        path: function (s) { return coneWorld(s, PHI_LORA, { x: 0, y: 0 }); },
        anchor: function () {
          var top = null;
          for (var k = 0; k < 72; k++) { var p = coneWorld(1, TAU * k / 72, { x: 0, y: 0 }); if (!top || p.y > top.y) top = p; }
          return { p: top, align: 'center', dy: -21 };
        },
      },
      {
        id: 'oft', dataId: 'oft', fb: 'OFT', kindFb: 'multiplicative-weight',
        desc: 'ORBIT OF $θ_0$', sentence: 'the orbit of \\(\\theta_0\\) under rotations, smooth at \\(\\theta_0\\)', short: 'orbit of \\(\\theta_0\\)',
        draw: function (alpha) {
          /* the whole orbit (a circle about 0), faint, and the germ arc through θ₀ */
          strokePlain(line(oftPt, 0, TAU, 160), this.color, alpha * 0.22, 0.9, [1.5, 5]);
          strokeGerm(line(oftPt, BETA0 - 1.0, BETA0 + 1.0, 80), this.color, alpha, 1.25, null, 1.15, 0);
        },
        path: function (s) { return oftPt(BETA0 - 0.46 * s); },
        anchor: function () { return { p: oftPt(BETA0 - 0.56), align: 'left', dx: 10, dy: -2 }; },
      },
      {
        id: 'ia3', dataId: 'ia3', fb: '(IA)³', kindFb: 'multiplicative-weight',
        desc: 'TORUS ORBIT · LINE THROUGH 0', sentence: 'a torus orbit whose closure is the line through \\(0\\) and \\(\\theta_0\\)', short: 'torus orbit, a line',
        draw: function (alpha) { strokeGerm(line(ia3Pt, -0.22, 1.62, 60), this.color, alpha, 1.25, null, 1.6, 0.18); },
        path: function (s) { return ia3Pt(1 - 0.5 * s); },
        anchor: function () { return { p: ia3Pt(0.36), align: 'left', dx: 12, dy: 2 }; },
      },
      {
        id: 'bitfit', dataId: 'bitfit', fb: 'BitFit', kindFb: 'selective',
        desc: 'COORDINATE LINE', sentence: 'only the bias coordinates move, along a coordinate line', short: 'a coordinate line',
        draw: function (alpha) { strokeGerm(line(function (x) { return { x: x, y: 0 }; }, -0.9, 1.4, 50), this.color, alpha, 1.25, null, 1.35, 0); },
        path: function (s) { return { x: 0.82 * s, y: 0 }; },
        anchor: function () { return { p: { x: 1.02, y: 0 }, align: 'left', dx: 8, dy: -6 }; },
      },
      {
        id: 'full', dataId: null, fb: 'Full FT', colorToken: '--ink-2', kindFb: null,
        desc: 'ALL OF $Θ$', sentence: 'all of \\(\\Theta\\), here following the gradient flow of the contoured loss', short: 'all of \\(\\Theta\\)',
        draw: function (alpha, clock, emph) {
          if (emph > 0.01) strokeGerm(FLOW, this.color, alpha * emph * 0.55, 1, [2, 4], 2.2, 0.5);
        },
        path: function (s) { return atLength(FLOW, FLOW_ACC, s); },
        anchor: function () { var e = FLOW[FLOW.length - 1]; return { p: { x: e.x + 0.06, y: e.y + 0.16 }, align: 'left', dx: 4, dy: 0 }; },
      },
      {
        id: 'prefix', dataId: 'prefix-tuning', fb: 'Prefix', kindFb: 'input-or-activation-augmentation', dashed: true,
        desc: 'UNPOINTED · MISSES $θ_0$', sentence: 'unpointed, since generically no prefix gives back the frozen model', short: 'unpointed',
        draw: function (alpha, clock, emph) {
          strokeGerm(PREFIX, this.color, alpha, 1.2, [5, 4], 1.4, 0.2);
          if (emph > 0.01) strokePlain([{ x: 0, y: 0 }, PREFIX_GAP.p], T.seal, emph * 0.7, 1, [1.5, 3]);
        },
        path: function (s) { return atLength(PREFIX, PREFIX_ACC, s * 0.92); },
        anchor: function () { return { p: PREFIX[0], align: 'right', dx: -6, dy: -4 }; },
      },
      {
        id: 'qlora', dataId: 'qlora', fb: 'QLoRA', kindFb: 'additive-weight', dashed: true,
        desc: 'APEX AT $κθ_0$ · DEFECT $e$', sentence: 'the same cone with its apex moved to \\(\\kappa\\theta_0\\), a defect \\(e=\\kappa\\theta_0-\\theta_0\\)', short: 'apex moved to \\(\\kappa\\theta_0\\)',
        draw: function (alpha, clock, emph) {
          drawCone.call(this, KT0, alpha * (0.45 + 0.55 * emph), true, spinAt(clock) + Math.PI / CONE.nq, CONE.nq);
        },
        path: function (s) {
          if (s < 0.28) { var u = s / 0.28; return { x: KT0.x * u, y: KT0.y * u }; }
          return coneWorld(-(s - 0.28) / 0.72, PHI_QLORA, KT0);
        },
        anchor: function () { return { p: { x: KT0.x + 0.14, y: KT0.y - 0.035 }, align: 'left', dx: 0, dy: 0 }; },
      },
    ];
    GERMS.forEach(function (g) {
      g.name = name(g.dataId, g.fb);
      g.kind = kind(g.dataId, g.kindFb);
      g.label = g.name;
    });
    var ORDER = ['lora', 'oft', 'ia3', 'bitfit', 'full', 'prefix', 'qlora'];
    var BY = {}; GERMS.forEach(function (g) { BY[g.id] = g; });
    var PHASE = 5.4;                                   // seconds per method
    function spinAt(clock) { return TAU * clock / CONE.spinPeriod; }
    function phaseAt(clock) {
      var k = Math.floor(clock / PHASE), u = (clock - k * PHASE) / PHASE;
      return { id: ORDER[((k % ORDER.length) + ORDER.length) % ORDER.length], u: u };
    }
    function emphOf(u) { return clamp01(Math.min(u / 0.09, (1 - u) / 0.13)); }
    function travelOf(u) { return ease((u - 0.07) / 0.7); }

    /* ---- cached layers: bathymetric shading + graticule (static), contours (slowly rising), mask ---- */
    var layStatic = document.createElement('canvas'), layContour = document.createElement('canvas'), layMask = document.createElement('canvas');
    var field = null, lastContourAt = -1;

    function buildField() {
      var c = LAY.phone ? 6 : 7, nx = Math.ceil(LAY.W / c) + 2, ny = Math.ceil(LAY.H / c) + 2;
      var F = new Float32Array(nx * ny);
      for (var j = 0; j < ny; j++) for (var i = 0; i < nx; i++) { var w = toW(i * c, j * c); F[j * nx + i] = Lf(w.x, w.y); }
      field = { c: c, nx: nx, ny: ny, F: F };
    }

    function drawStatic() {
      var g = layStatic.getContext('2d'), W = LAY.W, H = LAY.H, dpr = LAY.dpr;
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, layStatic.width, layStatic.height);
      /* bathymetric shading: deeper water (lower L) tinted with --tide */
      var c = 6, nx = Math.ceil(W / c) + 1, ny = Math.ceil(H / c) + 1;
      var small = document.createElement('canvas'); small.width = nx; small.height = ny;
      var sg = small.getContext('2d'), img = sg.createImageData(nx, ny), rgb = rgbOf(T.tide), maxA = T.dark ? 0.10 : 0.075;
      for (var j = 0; j < ny; j++) for (var i = 0; i < nx; i++) {
        var w = toW(i * c, j * c), depth = clamp01((-Lf(w.x, w.y) - 0.12) / 0.85), o = (j * nx + i) * 4;
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = Math.round(255 * maxA * depth * LAY.G);
      }
      sg.putImageData(img, 0, 0);
      g.imageSmoothingEnabled = true;
      g.drawImage(small, 0, 0, nx, ny, 0, 0, nx * c * dpr, ny * c * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      /* graticule = the quantiser's grid 0 + QSTEP·ℤ² */
      var step = QSTEP * LAY.R, x0 = toS(ZERO).x, y0 = toS(ZERO).y, x, y;
      g.strokeStyle = T.rule; g.lineWidth = 1; g.globalAlpha = (T.dark ? 0.85 : 0.95) * LAY.G;
      g.beginPath();
      for (x = x0 - Math.ceil(x0 / step) * step; x <= W; x += step) { g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, H); }
      for (y = y0 - Math.ceil(y0 / step) * step; y <= H; y += step) { g.moveTo(0, Math.round(y) + 0.5); g.lineTo(W, Math.round(y) + 0.5); }
      g.stroke();
      /* grid nodes: the representable (quantised) weights */
      g.fillStyle = T.ink3; g.globalAlpha = (T.dark ? 0.38 : 0.32) * LAY.G;
      for (x = x0 - Math.ceil(x0 / step) * step; x <= W; x += step) for (y = y0 - Math.ceil(y0 / step) * step; y <= H; y += step) {
        g.fillRect(Math.round(x) - 0.5, Math.round(y) - 0.5, 2, 2);
      }
      /* minute ticks along the chart's top and bottom edges (a neatline, as on a sea chart) */
      g.strokeStyle = T.ink3; g.globalAlpha = 0.35 * LAY.G; g.beginPath();
      var minor = step / 5;
      for (x = x0 - Math.ceil(x0 / minor) * minor; x <= W; x += minor) {
        var major = Math.abs(((x - x0) / step) - Math.round((x - x0) / step)) < 0.01, len = major ? 7 : 3.5;
        g.moveTo(Math.round(x) + 0.5, H); g.lineTo(Math.round(x) + 0.5, H - len);
        g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, len);
      }
      for (y = y0 - Math.ceil(y0 / minor) * minor; y <= H; y += minor) {
        var majorY = Math.abs(((y - y0) / step) - Math.round((y - y0) / step)) < 0.01, lenY = majorY ? 7 : 3.5;
        g.moveTo(W, Math.round(y) + 0.5); g.lineTo(W - lenY, Math.round(y) + 0.5);
      }
      g.stroke();
      g.globalAlpha = 1;
    }

    function drawContours(clock) {
      var g = layContour.getContext('2d'), f = field, c = f.c, nx = f.nx, F = f.F;
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, layContour.width, layContour.height);
      g.setTransform(LAY.dpr, 0, 0, LAY.dpr, 0, 0);
      var s = clock / RISE_PERIOD, d = LEVEL_STEP;
      s = s - Math.floor(s / 4) * 4;                    // keep the index-contour cycle (every 4th) exact
      var pN = new Path2D(), pI = new Path2D();
      for (var j = 0; j < f.ny - 1; j++) {
        var y0 = j * c, y1 = y0 + c;
        for (var i = 0; i < nx - 1; i++) {
          var a = F[j * nx + i], b = F[j * nx + i + 1], cc = F[(j + 1) * nx + i + 1], dd = F[(j + 1) * nx + i];
          var lo = Math.min(a, b, cc, dd), hi = Math.max(a, b, cc, dd);
          var n0 = Math.ceil(lo / d - s), n1 = Math.floor(hi / d - s);
          if (n1 < n0) continue;
          var x0 = i * c, x1 = x0 + c;
          for (var n = n0; n <= n1; n++) {
            var v = d * (n + s), P = (((n % 4) + 4) % 4 === 0) ? pI : pN;
            var idx = (a > v ? 8 : 0) | (b > v ? 4 : 0) | (cc > v ? 2 : 0) | (dd > v ? 1 : 0);
            if (idx === 0 || idx === 15) continue;
            var tx = x0 + c * (v - a) / (b - a), ty = y0;           // top edge
            var rx = x1, ry = y0 + c * (v - b) / (cc - b);           // right edge
            var bx = x0 + c * (v - dd) / (cc - dd), by = y1;         // bottom edge
            var lx = x0, ly = y0 + c * (v - a) / (dd - a);           // left edge
            switch (idx) {
              case 1: case 14: P.moveTo(lx, ly); P.lineTo(bx, by); break;
              case 2: case 13: P.moveTo(bx, by); P.lineTo(rx, ry); break;
              case 3: case 12: P.moveTo(lx, ly); P.lineTo(rx, ry); break;
              case 4: case 11: P.moveTo(tx, ty); P.lineTo(rx, ry); break;
              case 6: case 9: P.moveTo(tx, ty); P.lineTo(bx, by); break;
              case 7: case 8: P.moveTo(tx, ty); P.lineTo(lx, ly); break;
              case 5: P.moveTo(tx, ty); P.lineTo(rx, ry); P.moveTo(lx, ly); P.lineTo(bx, by); break;
              case 10: P.moveTo(tx, ty); P.lineTo(lx, ly); P.moveTo(rx, ry); P.lineTo(bx, by); break;
            }
          }
        }
      }
      g.strokeStyle = T.tide; g.lineJoin = 'round';
      g.globalAlpha = (T.dark ? 0.15 : 0.17) * LAY.G; g.lineWidth = 0.8; g.stroke(pN);
      g.globalAlpha = (T.dark ? 0.25 : 0.27) * LAY.G; g.lineWidth = 1.1; g.stroke(pI);
      g.globalAlpha = 1;
    }

    function drawMask() {
      var g = layMask.getContext('2d'), W = LAY.W, H = LAY.H, R = LAY.R;
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, layMask.width, layMask.height);
      g.setTransform(LAY.dpr, 0, 0, LAY.dpr, 0, 0);
      g.globalCompositeOperation = 'source-over';
      var rad = g.createRadialGradient(LAY.fx + 0.1 * R, LAY.fy, R * (LAY.band ? 0.9 : 1.15), LAY.fx + 0.1 * R, LAY.fy, R * (LAY.band ? 2.6 : 3.0));
      rad.addColorStop(0, 'rgba(0,0,0,1)'); rad.addColorStop(1, 'rgba(0,0,0,0.22)');
      g.fillStyle = rad; g.fillRect(0, 0, W, H);
      g.globalCompositeOperation = 'destination-in';
      var lin;
      if (LAY.band) {                                   // fade out above the band, where the text is
        var y1 = LAY.top != null ? LAY.top + 6 : H * 0.62, y0 = Math.max(0, y1 - 150);
        lin = g.createLinearGradient(0, y0, 0, y1); lin.addColorStop(0, 'rgba(0,0,0,0.14)'); lin.addColorStop(1, 'rgba(0,0,0,1)');
      }
      else { lin = g.createLinearGradient(W * 0.30, 0, W * 0.545, 0); lin.addColorStop(0, 'rgba(0,0,0,0.1)'); lin.addColorStop(1, 'rgba(0,0,0,1)'); }
      g.fillStyle = lin; g.fillRect(0, 0, W, H);
      g.globalCompositeOperation = 'source-over';
    }

    /* bottom edge of the hero's text block (kicker … buttons), in canvas pixels; null without the real markup */
    function textBottom() {
      var inner = host && host !== el && host.querySelector ? host.querySelector('.hero-inner') : null;
      if (!inner) return null;
      var top = canvas.getBoundingClientRect().top, b = 0;
      for (var i = 0; i < inner.children.length; i++) { var r = inner.children[i].getBoundingClientRect(); if (r.height) b = Math.max(b, r.bottom - top); }
      return b > 0 ? b : null;
    }
    /* right edge of the hero's text lines and buttons that meet the canvas rows [y0, y1]; null without the real markup */
    function textRight(y0, y1) {
      var inner = host && host !== el && host.querySelector ? host.querySelector('.hero-inner') : null;
      if (!inner || !document.createRange || !document.createTreeWalker) return null;
      var cr = canvas.getBoundingClientRect(), right = null, rg = document.createRange();
      function take(r) { var t = r.top - cr.top, b = r.bottom - cr.top; if (r.width && b >= y0 && t <= y1) right = Math.max(right == null ? 0 : right, r.right - cr.left); }
      var tw = document.createTreeWalker(inner, NodeFilter.SHOW_TEXT, null), n;
      while ((n = tw.nextNode())) {
        if (!/\S/.test(n.nodeValue)) continue;
        rg.selectNodeContents(n);
        var rs = rg.getClientRects();
        for (var i = 0; i < rs.length; i++) take(rs[i]);
      }
      Array.prototype.forEach.call(inner.querySelectorAll('a, button'), function (b) { take(b.getBoundingClientRect()); });
      return right;
    }
    function resize() {
      var W = canvas.clientWidth, H = canvas.clientHeight;
      if (!W || !H) return false;
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      LAY.W = W; LAY.H = H; LAY.dpr = dpr;
      LAY.phone = W < 640; LAY.tablet = W >= 640 && W < 980;
      /* same test as the CSS that opens the band (viewport width, scrollbar included) */
      LAY.band = host !== el && window.matchMedia ? window.matchMedia('(max-width: 979.98px)').matches : W < 640;
      if (LAY.band) {
        /* below 980px the text block spans the width, so θ₀ sits in the free band between the buttons and the
           control strip (the band comes from the .hero-live padding); without the real markup, fall back to
           the lower part of the canvas. No method labels here: the strip names the moving germ. */
        var tb = textBottom(), stripTop = H - 40;
        if (tb == null || tb > stripTop - 90) tb = Math.min(0.62 * H, stripTop - 90);
        var band = stripTop - tb;
        LAY.top = tb; LAY.fx = (LAY.phone ? 0.6 : 0.62) * W; LAY.fy = tb + 0.5 * band;
        LAY.R = Math.max(56, Math.min((LAY.phone ? 0.3 : 0.22) * W, 0.98 * band)); LAY.G = LAY.phone ? 0.8 : 0.88; LAY.fs = 12.5; LAY.fsm = LAY.phone ? 9 : 10;
      }
      else {
        LAY.fx = 0.725 * W; LAY.fy = 0.5 * H; LAY.R = Math.min(0.18 * W, 0.38 * H, 280); LAY.G = 1; LAY.fs = 13; LAY.fsm = 10.5;
        /* keep the cone and the θ₀ label clear of the text lines at its height (the dek reaches far right at ~1000px) */
        var tr = textRight(LAY.fy - 0.5 * LAY.R, LAY.fy + 0.5 * LAY.R);
        if (tr != null) LAY.fx = Math.min(W - 1.1 * LAY.R, Math.max(LAY.fx, tr + 0.36 * LAY.R + 30));
      }
      [canvas, layStatic, layContour, layMask].forEach(function (c) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); });
      buildField(); drawStatic(); drawMask(); lastContourAt = -1;
      return true;
    }

    /* ---- one frame ---- */
    function render(clock) {
      if (!LAY.W) return;
      readFonts();
      var W = LAY.W, H = LAY.H, dpr = LAY.dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (lastContourAt < 0 || Math.abs(clock - lastContourAt) > 0.1) { drawContours(clock); lastContourAt = clock; }
      ctx.drawImage(layStatic, 0, 0);
      ctx.drawImage(layContour, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var ph = still() ? null : phaseAt(clock);
      var active = ph ? BY[ph.id] : null, emph = ph ? emphOf(ph.u) : 0;
      var rest = still() ? 0.62 : 0.4;

      /* 0 ∈ Θ: centre of the OFT orbit and the point the (IA)³ line runs through */
      var z = toS(ZERO);
      if (z.y < LAY.H - 46) {                            // not under the control strip
        ctx.globalAlpha = 0.6 * LAY.G; ctx.strokeStyle = T.ink3; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(z.x - 4, z.y); ctx.lineTo(z.x + 4, z.y); ctx.moveTo(z.x, z.y - 4); ctx.lineTo(z.x, z.y + 4); ctx.stroke();
        ctx.globalAlpha = 1;
        text('0', z.x + 6, z.y + 13, '400 12px ' + T.mono, T.ink2, 0.85, 'left', 0, 3);
      }

      /* germs: inactive first, the active one on top */
      GERMS.forEach(function (g) {
        if (g === active) return;
        var a = rest;
        if (g.id === 'lora' && active && active.id === 'qlora') a = rest * (1 - 0.6 * emph);
        g.draw(a, clock, 0);
      });
      if (active) active.draw(rest + (0.95 - rest) * emph, clock, emph);

      /* defect arrow e = κθ₀ − θ₀ and the grid node κθ₀ */
      var qe = active && active.id === 'qlora' ? emph : 0;
      arrow({ x: 0, y: 0 }, KT0, T.seal, (still() ? 0.75 : 0.55) + 0.4 * qe, 1.3);
      var k = toS(KT0);
      ctx.globalAlpha = (0.55 + 0.45 * qe) * LAY.G; ctx.strokeStyle = BY.qlora.color; ctx.lineWidth = 1.1;
      ctx.beginPath(); ctx.arc(k.x, k.y, 3, 0, TAU); ctx.stroke(); ctx.globalAlpha = 1;
      var em = { x: KT0.x / 2, y: KT0.y / 2 }, nrm = Math.hypot(KT0.x, KT0.y) || 1, es = toS({ x: em.x + 0.07 * KT0.y / nrm, y: em.y - 0.07 * KT0.x / nrm });
      text('e', es.x - 3, es.y + 4, 'italic 500 ' + (LAY.fsm + 4) + 'px ' + T.body, T.seal, 0.8 + 0.2 * qe, 'left', 0, 3);

      /* travelling point: the fine-tuned model θ(t) moving along the active germ */
      if (active && !still()) {
        var s = travelOf(ph.u), trail = [], N = 48;
        for (var i = 0; i <= N; i++) trail.push(active.path(s * i / N));
        strokePlain(trail, active.color, 0.9 * emph, 2, active.dashed ? [5, 4] : null);
        var p = active.path(s), ps = toS(p);
        var glow = ctx.createRadialGradient(ps.x, ps.y, 0, ps.x, ps.y, 16);
        glow.addColorStop(0, rgba(active.color, 0.35 * emph * LAY.G)); glow.addColorStop(1, rgba(active.color, 0));
        ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(ps.x, ps.y, 16, 0, TAU); ctx.fill();
        dot(p, LAY.phone ? 2.8 : 3.4, active.color, emph, true);
        var pa = toS(active.path(Math.max(0, s - 0.03))), pb = toS(active.path(Math.min(1, s + 0.03)));
        var nx = -(pb.y - pa.y), ny = pb.x - pa.x, nn = Math.hypot(nx, ny) || 1, la = active.anchor(), ls = toS(la.p);
        nx /= nn; ny /= nn;
        if (nx * (ls.x - ps.x) + ny * (ls.y - ps.y) > 0) { nx = -nx; ny = -ny; }
        if (nn < 0.5) { nx = 0.7; ny = 0.7; }
        var away = clamp01((Math.hypot(ps.x - LAY.fx, ps.y - LAY.fy) - 14) / 22);   // show the label once the point has left θ₀
        text('θ', ps.x + 13 * nx, ps.y + 13 * ny + 5, 'italic 500 ' + (LAY.fsm + 5) + 'px ' + T.display, T.ink, 0.85 * emph * away, 'center', 0, 3);
      }

      /* fade everything toward the headline and the chart's edges */
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = 'destination-in';
      ctx.drawImage(layMask, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      /* labels, after the fade: they take only a floored share of it, so a label stays readable wherever it lands.
         In the narrow band layout the strip names the moving germ, so only the still frame gets (name-only) labels. */
      LBOX = [];
      if (!LAY.band || still()) GERMS.forEach(function (g) {
        var an = g.anchor(), sp = toS(an.p);
        var a = still() ? 0.95 : (g === active ? 0.78 + 0.22 * emph : 0.78);
        if (g.id === 'full' && !still() && g !== active) a = 0.7;
        label(g, { x: sp.x + (an.dx || 0), y: sp.y }, an.align, a, an.dy);
      });

      /* θ₀ on top: the focus of the chart, pulsing gently */
      var t0 = toS({ x: 0, y: 0 });
      if (!still()) {
        var q = (clock / 2.8) % 1;
        ctx.globalAlpha = 0.5 * (1 - q) * (1 - q) * LAY.G; ctx.strokeStyle = T.tide; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(t0.x, t0.y, 5 + 15 * q, 0, TAU); ctx.stroke();
      }
      ctx.globalAlpha = 0.75 * LAY.G; ctx.strokeStyle = T.tide; ctx.lineWidth = 1.1;
      ctx.beginPath(); ctx.arc(t0.x, t0.y, 7, 0, TAU); ctx.stroke(); ctx.globalAlpha = 1;
      dot({ x: 0, y: 0 }, LAY.phone ? 3 : 3.6, T.ink, 0.95, true);
      var fsT = LAY.phone ? 16 : 20;
      text('0', t0.x - 11, t0.y + 25, '500 ' + Math.round(fsT * 0.62) + 'px ' + T.display, T.ink, 0.95, 'right', 0, 3);
      text('θ', t0.x - 11 - Math.round(fsT * 0.38), t0.y + 20, 'italic 500 ' + fsT + 'px ' + T.display, T.ink, 0.95, 'right', 0, 4);
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      updateCaption(active);
    }

    /* ---- small control strip (pause + what is moving), only when animated ---- */
    var strip = null, nowEl = null, btn = null, paused = false, lastCaptionId = null;
    function buildStrip() {
      if (strip || el.tagName !== 'CANVAS' || !host || host === el || !host.appendChild) return;
      strip = A.h('div', { class: 'sea-ctl' });
      nowEl = A.h('span', { class: 'sea-now', 'aria-hidden': 'true' });
      btn = A.h('button', { type: 'button', 'aria-label': 'Pause the background animation', text: 'Pause' });
      btn.addEventListener('click', function () {
        paused = !paused;
        btn.textContent = paused ? 'Play' : 'Pause';
        btn.setAttribute('aria-label', (paused ? 'Play' : 'Pause') + ' the background animation');
        update();
      });
      strip.appendChild(nowEl); strip.appendChild(btn);
      host.appendChild(strip);
      buildCaptions();
    }
    /* every caption is built once and typeset once by MathJax, then only shown or hidden (no TeX flashes) */
    function plainMath(str) {                          // fallback when MathJax is absent
      return str.replace(/\\\(([\s\S]*?)\\\)/g, function (m, t) {
        return t.replace(/\\theta/g, 'θ').replace(/\\Theta/g, 'Θ').replace(/\\kappa/g, 'κ').replace(/_0/g, '₀').replace(/-/g, ' − ').replace(/=/g, ' = ');
      });
    }
    function buildCaptions() {
      var html = GERMS.map(function (g) {
        var tok = g.colorToken || (function () { for (var i = 0; i < A.KINDS.length; i++) if (A.KINDS[i].key === g.kind) return A.KINDS[i].token; return '--c-hybrid'; })();
        return '<span class="cap" data-k="' + g.id + '"><span class="sw' + (g.dashed ? ' dash' : '') + '" style="background:var(' + tok + ');color:var(' + tok + ')"></span>' +
          '<b>' + A.esc(g.name) + '</b> <span class="long">' + g.sentence +
          '</span><span class="short">' + g.short + '</span></span>';
      }).join('') +
        '<span class="cap" data-k="__still"><b>Germs at \\(\\theta_0\\)</b> <span class="long">solid: pointed at \\(\\theta_0\\) · dashed: not pointed at \\(\\theta_0\\)</span><span class="short">dashed: not at \\(\\theta_0\\)</span></span>';
      var MJ = window.MathJax;
      nowEl.innerHTML = MJ && (MJ.typesetPromise || MJ.startup) ? html : plainMath(html);
      if (MJ && (MJ.typesetPromise || MJ.startup)) A.typeset(nowEl);
    }
    function updateCaption(active) {
      if (!nowEl) return;
      var id = still() ? '__still' : (active ? active.id : null);
      if (!id || id === lastCaptionId) return;
      lastCaptionId = id;
      var caps = nowEl.querySelectorAll('.cap');
      for (var i = 0; i < caps.length; i++) caps[i].classList.toggle('on', caps[i].getAttribute('data-k') === id);
    }

    /* ---- animation loop: rAF, ~30 fps, paused off-screen / hidden / by the user / under reduced motion ---- */
    var clock = frozenT != null ? frozenT : 0, raf = 0, running = false, last = 0, onScreen = true;
    function frame(now) {
      raf = requestAnimationFrame(frame);
      if (!last) { last = now; return; }
      var dt = (now - last) / 1000;
      if (dt < 1 / 31) return;
      last = now;
      clock += Math.min(dt, 0.1);
      render(clock);
    }
    function start() { if (running) return; running = true; last = 0; raf = requestAnimationFrame(frame); }
    function stop() { running = false; cancelAnimationFrame(raf); }
    function animated() { return !still() && frozenT == null; }
    function update() {
      buildStrip();
      if (btn) btn.hidden = !animated();
      if (animated() && onScreen && !document.hidden && !paused) start();
      else { stop(); render(clock); }
    }

    canvas.__hero = { render: function (t) { render(t == null ? clock : t); }, clock: function () { return clock; } }; // debugging / perf probes
    readTokens();
    if (resize()) render(clock);
    update();

    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(function () { if (resize()) render(clock); }, 80)).observe(canvas);
    else window.addEventListener('resize', A.debounce(function () { if (resize()) render(clock); }, 100));
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) { onScreen = es[es.length - 1].isIntersecting; update(); }).observe(canvas);
    }
    document.addEventListener('visibilitychange', update);
    if (mqReduce) { var h = function () { lastCaptionId = null; update(); }; if (mqReduce.addEventListener) mqReduce.addEventListener('change', h); else if (mqReduce.addListener) mqReduce.addListener(h); }
    A.onTheme(function () { readTokens(); if (LAY.W) { drawStatic(); lastContourAt = -1; render(clock); } });
    /* web fonts reflow the text block, which moves the phone band: lay out again */
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { if (resize()) render(clock); });
  });
})();
