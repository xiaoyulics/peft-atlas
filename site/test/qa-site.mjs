// Whole-site QA through the Chrome DevTools Protocol.
// node test/qa-site.mjs [page=index.html] [outdir=test/qa]
// For each (theme, width): load the page, scroll through it so every lazy figure mounts, then report
// console errors/exceptions, unmounted figures, horizontal overflow, and save per-section screenshots.
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const page = path.resolve(process.argv[2] || 'index.html');
const out = path.resolve(process.argv[3] || 'test/qa');
mkdirSync(out, { recursive: true });
const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9800 + Math.floor(Math.random() * 150);
const prof = mkdtempSync(path.join(process.env.TMPDIR || tmpdir(), 'qacdp-'));
const chrome = spawn(CH, ['--headless=new', '--disable-gpu', '--no-first-run', '--hide-scrollbars', '--allow-file-access-from-files',
  `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, '--window-size=1440,1000', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws, id = 0; const pend = new Map(); let logs = [];
async function connect() {
  for (let i = 0; i < 300; i++) {
    try { const l = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); const pg = l.find((t) => t.type === 'page'); if (pg) return pg.webSocketDebuggerUrl; } catch (e) {}
    await sleep(200);
  }
  throw new Error('no chrome');
}
function send(method, params = {}) { return new Promise((res) => { const i = ++id; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); }); }
async function ev(expr) { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) return 'EXC ' + JSON.stringify(r.exceptionDetails).slice(0, 300); return r.result && r.result.value; }

const CONFIGS = (process.env.QA_CONFIGS || 'light-1440,dark-1440,light-390,dark-390').split(',');
const SHOTS = process.env.QA_SHOTS !== '0';
try {
  ws = new WebSocket(await connect());
  await new Promise((r) => ws.addEventListener('open', r));
  ws.addEventListener('message', (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pend.has(d.id)) { pend.get(d.id)(d.result || d); pend.delete(d.id); }
    else if (d.method === 'Runtime.consoleAPICalled' && /error|warn|assert/.test(d.params.type)) logs.push(d.params.type + ': ' + d.params.args.map((a) => a.value !== undefined ? a.value : a.description).join(' ').slice(0, 300));
    else if (d.method === 'Runtime.exceptionThrown') logs.push('EXC ' + String((d.params.exceptionDetails.exception && d.params.exceptionDetails.exception.description) || d.params.exceptionDetails.text).slice(0, 400));
    else if (d.method === 'Log.entryAdded' && d.params.entry.level === 'error') logs.push('LOG ' + d.params.entry.text.slice(0, 200) + ' ' + (d.params.entry.url || ''));
  });
  await send('Runtime.enable'); await send('Page.enable'); await send('Log.enable');
  for (const cfg of CONFIGS) {
    const [theme, wStr] = cfg.split('-'); const w = +wStr; const mobile = w < 700;
    logs = [];
    await send('Emulation.setDeviceMetricsOverride', { width: w, height: 900, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: theme }] });
    await send('Page.navigate', { url: 'file://' + page });
    for (let i = 0; i < 80; i++) { await sleep(250); if (await ev(`document.readyState==='complete' && !!(window.MathJax&&MathJax.startup&&MathJax.startup.document)`)) break; }
    await sleep(1500);
    // scroll through to mount every lazy figure
    for (let y = 0, guard = 0; guard < 2000; guard++, y += 800) {
      await ev(`window.scrollTo(0, ${y})`); await sleep(90);
      if (y > (await ev('document.documentElement.scrollHeight')) + 1000) break;
    }
    await sleep(2500);
    await ev('window.scrollTo(0,0)'); await sleep(500);
    const report = await ev(`(function(){
      var vw = innerWidth, d = document.documentElement, figs = [], ovf = [], clip = [];
      /* content cut at an SVG viewport or at an ancestor that clips within the viewport cannot widen the page */
      function clippedBy(e) {
        if (e.ownerSVGElement && getComputedStyle(e.ownerSVGElement).overflow !== 'visible' && e.ownerSVGElement.getBoundingClientRect().right <= vw + 1) return true;
        for (var a = e.parentElement; a && a !== document.body; a = a.parentElement) {
          var ox = getComputedStyle(a).overflowX;
          if (ox !== 'visible' && a.getBoundingClientRect().right <= vw + 1) return true;
        }
        return false;
      }
      document.querySelectorAll('[data-figure]').forEach(function (el) {
        var n = el.getAttribute('data-figure');
        var mounted = el.__atlasMounted, empty = el.tagName !== 'CANVAS' && el.children.length === 0;
        if (!mounted || empty) figs.push(n + (mounted ? ':empty' : ':unmounted'));
      });
      document.querySelectorAll('body *').forEach(function (e) {
        var r = e.getBoundingClientRect(); if (!r.width) return;
        if (r.right > vw + 1 && !e.closest('.scroll-x,.table-wrap,mjx-container,pre,.toc,.topbar nav')) {
          var p = e.parentElement; var pr = p && p.getBoundingClientRect();
          if (!pr || pr.right <= vw + 1) (clippedBy(e) ? clip : ovf).push((clippedBy(e) && e.parentElement ? '{' + String(e.parentElement.className && e.parentElement.className.baseVal !== undefined ? e.parentElement.className.baseVal : e.parentElement.className).slice(0, 24) + ': ' + (e.parentElement.textContent || '').trim().slice(0, 40) + '} ' : '') + (e.closest('[data-figure]') ? '[' + e.closest('[data-figure]').getAttribute('data-figure') + '] ' : (e.closest('section') ? '#' + e.closest('section').id + ' ' : '')) + e.tagName.toLowerCase() + '.' + String(e.className && e.className.baseVal !== undefined ? e.className.baseVal : e.className).slice(0, 30) + ' right=' + Math.round(r.right));
        }
      });
      var mjErr = document.querySelectorAll('mjx-merror, .mjx-merror, [data-mjx-error]').length;
      var rawTex = 0; document.querySelectorAll('.prose p, .prose li, .env, figcaption').forEach(function (p) { if (/\\\\(\\(|\\[)|\\\\(frac|mathbb|mathrm|operatorname)\\b/.test(p.textContent) && !p.closest('pre,code')) rawTex++; });
      var D = window.ATLAS_DATA || {}, known = {};
      (D.methods || []).forEach(function (x) { known['m-' + x.id] = 1; }); (D.papers || []).forEach(function (x) { known['p-' + x.id] = 1; });
      /* #m-<id> and #p-<id> are catalogue deep links routed by fig-catalog.js, fine when the id is catalogued */
      var badLinks = []; document.querySelectorAll('a[href^="#"]').forEach(function (a) { var h = a.getAttribute('href').slice(1); try { h = decodeURIComponent(h); } catch (e) {} if (h && !document.getElementById(h) && !known[h]) badLinks.push(h); });
      return JSON.stringify({ vw: vw, scrollW: d.scrollWidth, height: d.scrollHeight, figuresWithProblems: figs, overflow: ovf.slice(0, 25), overflowCount: ovf.length, clippedOverflow: clip.slice(0, 8), clippedCount: clip.length, mathjaxErrors: mjErr, paragraphsWithRawTex: rawTex, brokenAnchors: Array.from(new Set(badLinks)).slice(0, 40), brokenAnchorCount: new Set(badLinks).size });
    })()`);
    console.log('=====', cfg, report);
    const uniq = Array.from(new Set(logs));
    if (uniq.length) console.log('  LOGS (' + uniq.length + ')\n   ' + uniq.slice(0, 30).join('\n   '));
    if (SHOTS) {
      const secs = JSON.parse(await ev(`JSON.stringify(Array.from(document.querySelectorAll('header.hero, section.expose')).map(function (s) { var r = s.getBoundingClientRect(); return { id: s.id || 'hero', y: r.top + scrollY, h: r.height }; }))`));
      for (const s of secs) {
        let y0 = s.y, part = 0;
        while (y0 < s.y + s.h && part < 12) {
          const hh = Math.min(3000, s.y + s.h - y0);
          await ev(`window.scrollTo(0, ${y0})`); await sleep(350);
          const r = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: y0, width: w, height: hh, scale: mobile ? 0.5 : 0.6 } });
          if (r.data) writeFileSync(path.join(out, `${cfg}-${s.id}-${part}.png`), Buffer.from(r.data, 'base64'));
          y0 += hh; part++;
        }
      }
    }
  }
} finally { try { ws && ws.close(); } catch (e) {} chrome.kill(); }
