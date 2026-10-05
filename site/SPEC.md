# Site engineering spec (for every builder agent)

The site is a static, dependency-light academic explorable. It must work in three places:
1. GitHub Pages (or any static host),
2. opened directly from disk (`file://`),
3. a sandboxed claude.ai Artifact (strict CSP).

## Hard constraints
- **Classic scripts only** (no ES modules, no bundler, no `import`). Every module is an IIFE that calls `Atlas.register(name, init)`.
- **Data is JavaScript, not JSON fetched at runtime**: `data/atlas-data.js` sets `window.ATLAS_DATA = {...}`. Never `fetch()`.
- **External scripts only** from `https://cdnjs.cloudflare.com` (preferred) or `https://cdn.jsdelivr.net/npm/`, pinned exact versions. Allowed libraries already loaded by `index.html`:
  - D3 v7: `https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js` (global `d3`)
  - MathJax 3 SVG: `https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js` (global `MathJax`); inline math `\( … \)` or `$…$`, display `\[ … \]` or `$$…$$`.
  Do not add any other library. No external stylesheets except Google Fonts (already linked). No images from other hosts; draw with SVG/Canvas.
- No `alert/confirm/prompt`, no `window.print`, no downloads via `<a download>`, no iframes, no `localStorage` without try/catch.
- **Theme**: never hard-code colours. Read tokens with `Atlas.css('--tide')` etc. and re-render on `Atlas.onTheme(fn)`. In SVG prefer CSS classes or `style="fill: var(--ink-2)"`.
- **Responsive**: works at 380px width; no horizontal page scroll. Wide SVGs live inside `.scroll-x` or scale via `viewBox` + `width:100%`.
- **Reduced motion**: check `Atlas.reducedMotion()`; animations must have a meaningful static frame.
- **Complete at rest**: every figure renders a meaningful default state immediately on mount (no blank canvas waiting for input).
- **Accessibility**: real `<button>`/`<input>` with labels; focus visible; figures have an `aria-label` or caption.
- **Correctness**: every number a figure shows must be computed, not invented. Parameter-count formulas must match the method's paper. Math in captions must be true.

## Files
```
site/
  index.html            page shell + all prose (exposés); figures are <div data-figure="name" class="..."></div>
  css/atlas.css         design tokens + components (read it before styling; extend with a module-scoped block only if needed)
  js/core.js            Atlas namespace: register, css, onTheme, typeset, tex, tip, h, esc, fmtCount, fmtPct, seg, slider,
                        canvas, rng (seeded, .normal()), LA (zeros, eye, randn, T, mul, add, scale, hadamard, kron, diag,
                        frob, symEig, svals, numRank, randOrth), data(), method(id), KINDS, kindColor, kindName, reducedMotion, debounce
  js/fig-<name>.js      one file per interactive figure
  data/atlas-data.js    window.ATLAS_DATA (generated; do not hand-edit)
```

## Figure module template
```js
(function () {
  'use strict';
  Atlas.register('gauge', function (el, A) {
    var stage = A.h('div', { class: 'stage' });
    el.appendChild(stage);
    function draw() { /* read A.css('--tide') etc. here, every time */ }
    draw();
    A.onTheme(draw);
  });
})();
```
CSS that is specific to one figure goes in a `<style>` block injected once by that module (`if (!document.getElementById('css-gauge')) {...}`), scoped under `[data-figure="gauge"]`.

## Visual language
- Palette tokens: `--paper --paper-2 --paper-3 --ink --ink-2 --ink-3 --rule --tide --tide-soft --ochre --ochre-soft --seal --seal-soft --moss --moss-soft`.
- Series colours by modification kind: `--c-additive --c-multiplicative --c-architectural --c-augmentation --c-selective --c-optimizer --c-composition --c-hybrid` (use `Atlas.kindColor(kind)`).
- Semantic: `--moss` = holds / mergeable / commutes; `--seal` = obstruction / not mergeable; `--ochre` = highlight / conditional.
- Type: `--f-display` (Source Serif 4, weight `var(--w-head)` = 600) for figure titles and node labels of objects; `--f-body` (Source Serif 4, weight `var(--w-body)`) for prose, instructions and captions; `--f-ui` (IBM Plex Sans, weight 500, letter-spacing at most 0.06em) for labels, controls, tabs, chips, legends, axis titles and uppercase eyebrows; `--f-mono` (IBM Plex Mono, `font-variant-numeric: tabular-nums`) for numbers, readouts, matrices, code and tick values; `--f-brand` (Bodoni Moda) for the wordmark and hero title only. SVG text defaults to `--f-ui` 12px; add class `num` or `mono` for numbers. Rendered text a reader must read is at least 12px (11px for dense ticks) at 1280px and at 390px, after viewBox scaling. Canvas fonts are built from the tokens at draw time and redrawn on theme change. Readable text uses `--ink` or `--ink-2`; `--ink-3` is for decoration only.
- Commutative diagrams: objects in display serif, arrows 1.25px `--ink-2` with small open arrowheads, 2-cells as double arrows, labels in italic serif via MathJax when math is needed.
- Thin rules, generous whitespace, no drop shadows on figures, radius 6px max.

## Self-test (required before you finish)
1. `node --check js/fig-<name>.js` (syntax).
2. `bash test/figtest.sh <name>` renders only your figure headless (Chrome) at 1280px and 390px, light and dark, and prints
   console output. Then **look at the PNGs** with the Read tool (`test/fig-<name>-light-1280.png`, `-dark-390.png`, ...).
   Fix anything clipped, overlapping, unreadable in one theme, blank, or erroring. Iterate until clean.
   (The harness uses `--virtual-time-budget=8000`, so animations will have advanced; that is expected.)
3. Data lives in `data/atlas-data.js` (regenerate with `python3 build/build_data.py` only if told to; never hand-edit).
