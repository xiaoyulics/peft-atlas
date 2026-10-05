/* Fig. 12 — The Merge Game (Thm VI.3, the merge calculus; Lemma VI.2; Thm VI.5(b,f) for prefixes).
   A pre-LN transformer block (d = 8, d_ff = 16, one causal head, 6 tokens) drawn as a string diagram.
   A PEFT edit is dropped on a wire or a box; the rule engine (R1–R7 of the repaired Thm VI.3) slides it toward
   the weight the reader chose, and either fuses it there (moss) or stops at the box whose side condition fails
   (seal), naming the failed rule. Side conditions are exactly those of the repaired statement:
     R1 fusion (affine boxes compose; shifts need a bias slot), R3 needs ℓ ≥ 0 and a positively homogeneous σ,
     R4 spider slide through the GLU Hadamard product for every ℓ, R5 under RoPE needs ℓ constant on rotary pairs,
     R6 norm absorption (the shift Wβ needs a bias slot), R7 copy slide to every consumer (pre-LN).
   Every rewrite is checked numerically in float64 on seeded random weights:
     licensed merge  ‖y_adapted − y_merged‖_F / ‖y_adapted‖_F   (≈ 1e−16 when the rewrite is legal),
     forced merge    the same norm after applying the refused rule anyway (diagonal edits), or after the best
                     least-squares refit of the target box (non-affine edits: adapters, prefixes).
   Only box-local merges (Def VI.1) are claimed; nothing here says a weight elsewhere could not absorb an edit. */
(function () {
  'use strict';

  var CSS_ID = 'css-merge';
  var F = '[data-figure="merge"] ';
  var CSS = [
    F + '.mg-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem;margin:0 0 .2rem}',
    F + '.mg-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.55rem,3.6vw,2.15rem);line-height:1.08;letter-spacing:-.006em;margin:0;color:var(--ink)}',
    F + '.mg-title em{font-style:italic;color:var(--ochre)}',
    F + '.mg-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    F + '.mg-instr{font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);margin:.2rem 0 .85rem;max-width:54rem}',
    F + '.label{letter-spacing:.06em;color:var(--ink-2)}',
    F + '.mg-controls{display:grid;gap:.75rem;margin:0 0 .7rem;padding:.7rem .8rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius)}',
    F + '.mg-row{display:flex;flex-wrap:wrap;gap:.65rem 1.15rem;align-items:flex-end}',
    F + '.mg-grp{display:grid;gap:.28rem;min-width:0;max-width:100%}',
    F + '.mg-grp .seg{max-width:100%;justify-self:start;border-radius:15px}',
    F + '.mg-grp .seg button{font-size:var(--fs-xs);font-weight:500;letter-spacing:.02em;padding:.36rem .64rem}',
    F + '.mg-grp .seg button i{font-family:var(--f-body);font-size:1.12em}',
    F + '.mg-grp .seg button:disabled{cursor:not-allowed}',
    F + '.mg-grp.is-off{opacity:.45}',
    F + '.mg-grp .label i{font-family:var(--f-body);text-transform:none;letter-spacing:0;font-size:1.15em}',
    F + '.mg-btns{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin-left:auto}',
    F + '.mg-btns .btn{padding:.38rem .7rem;letter-spacing:.05em}',
    F + '.mg-btns .btn:disabled{opacity:.45;cursor:not-allowed}',
    F + '.mg-btns .btn.force{border-color:var(--seal);color:var(--seal)}',
    F + '.mg-btns .btn.force:hover:not(:disabled){background:var(--seal-soft)}',
    F + '.mg-btns .btn.force[aria-pressed="true"]{background:var(--seal);color:var(--paper)}',
    F + '.mg-legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:0;color:var(--ink-2);margin:0 0 .6rem}',
    F + '.mg-legend span{display:inline-flex;align-items:center;gap:.35rem;white-space:nowrap}',
    F + '.mg-legend svg{display:inline-block;overflow:visible;flex:none}',
    F + '.mg-main{display:grid;gap:.75rem;grid-template-columns:minmax(0,1fr)}',
    /* two columns only when the stage itself is wide enough for the diagram to keep its labels at 12px or more */
    F + '.mg-two .mg-main{grid-template-columns:minmax(0,1fr) minmax(0,1.08fr)}',
    F + '.mg-panel{background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.6rem .7rem;min-width:0}',
    '@media (max-width:520px){' + F + '.stage{padding:.8rem .6rem}' + F + '.mg-panel{padding:.5rem .4rem}' + F + '.mg-controls{padding:.6rem .5rem}}',
    F + '.mg-side{display:grid;gap:.75rem;align-content:start;min-width:0}',
    F + '.mg-ph{display:flex;justify-content:space-between;align-items:baseline;gap:.15rem .6rem;flex-wrap:wrap;font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:0 0 .45rem}',
    F + '.mg-ph b{font-weight:600;color:var(--ink)}',
    F + '.mg-ph .thm{font-family:var(--f-body);font-weight:var(--w-body);text-transform:none;letter-spacing:0;font-size:.88rem;color:var(--ink-2)}',
    F + '.mg-ph .num{font-weight:400;letter-spacing:0;text-transform:none;font-variant-numeric:tabular-nums}',
    F + '.mg-svg svg{display:block;width:100%;height:auto;overflow:visible}',
    F + '.mg-hyp{margin-top:.45rem;padding-top:.45rem;border-top:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:.3rem .35rem;align-items:center;font-size:.8rem;line-height:1.45;color:var(--ink-2)}',
    F + '.mg-hyp .label{margin-right:.2rem}',
    F + '.chip{font-size:var(--fs-xs);font-weight:500;letter-spacing:.02em}',
    F + '.mg-hyp .chip{white-space:normal}',
    F + '.mg-hyp p{margin:.15rem 0 0;flex-basis:100%;font-family:var(--f-body);font-size:.84rem}',
    F + '.mg-log ol{list-style:none;margin:0;padding:0;display:grid;gap:.5rem}',
    F + '.mg-step{display:grid;grid-template-columns:3.9rem minmax(0,1fr);gap:.55rem;align-items:start;transition:opacity .3s}',
    F + '.mg-step.pending{opacity:.16}',
    F + '.mg-rule{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.02em;text-align:center;padding:.12rem .15rem;border-radius:4px;border:1px solid currentColor;color:var(--ochre-ink);background:var(--ochre-soft);margin-top:.12rem;white-space:nowrap}',
    F + '.mg-step.bad .mg-rule{color:var(--seal-ink);background:var(--seal-soft)}',
    F + '.mg-step.edit .mg-rule{color:var(--ink-2);background:transparent;border-style:dashed}',
    F + '.mg-step.forced .mg-rule{color:var(--paper);background:var(--seal);border-color:var(--seal)}',
    F + '.mg-sname{font-family:var(--f-body);font-weight:600;font-size:.97rem;color:var(--ink);line-height:1.3}',
    F + '.mg-sname i,.mg-snote i,.mg-verdict i,.mg-cell i,.mg-foot i{font-family:var(--f-body)}',
    F + '.mg-smath{color:var(--ink);font-size:.92rem;overflow-x:auto;overflow-y:hidden;padding:.12rem 0 .05rem;max-width:100%}',
    F + '.mg-snote{font-family:var(--f-body);font-size:.86rem;line-height:1.45;color:var(--ink-2)}',
    F + '.mg-snote .mono{font-size:.92em;font-variant-numeric:tabular-nums}',
    F + '.mg-verdict{margin-top:.65rem;padding:.5rem .65rem;border-radius:4px;border:1px solid currentColor;font-family:var(--f-body);font-size:.92rem;line-height:1.42;transition:opacity .3s}',
    F + '.mg-verdict.ok{color:var(--moss-ink);background:var(--moss-soft)}',
    F + '.mg-verdict.bad{color:var(--seal-ink);background:var(--seal-soft)}',
    F + '.mg-verdict .vt{font-family:var(--f-display);font-size:1.15rem;font-weight:var(--w-head)}',
    F + '.mg-verdict .vb{color:var(--ink)}',
    F + '.mg-verdict .vr{display:block;font-family:var(--f-body);font-size:.82rem;line-height:1.4;color:var(--ink-2);margin-top:.25rem}',
    F + '.mg-verdict.pending{opacity:0}',
    /* sub- and superscripts carry identity (W_down vs W_up): never below 12px */
    F + 'sub,' + F + 'sup{font-size:max(.78em,12px);line-height:0}',
    F + '.mg-cells{display:grid;gap:.55rem .9rem;grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr))}',
    F + '.mg-cell .nw{white-space:nowrap}',
    F + '.mg-cell .k .nt{text-transform:none;letter-spacing:0;font-family:var(--f-body);font-size:1.12em}',
    F + '.mg-cell{border-top:1px solid var(--rule);padding-top:.38rem;min-width:0}',
    F + '.mg-cell .k{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);line-height:1.35;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
    F + '.mg-cell .v{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:1.02rem;line-height:1.3;color:var(--ink);overflow-wrap:anywhere}',
    F + '.mg-cell .v.big{font-size:1.6rem;line-height:1.15}',
    F + '.mg-cell .v.ok{color:var(--moss-ink)}',
    F + '.mg-cell .v.bad{color:var(--seal-ink)}',
    F + '.mg-cell .v.mut{font-family:var(--f-ui);font-weight:500;color:var(--ink-2);font-size:1.02rem}',
    F + '.mg-cell .s{font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2);overflow-wrap:anywhere}',
    F + '.mg-cell .s .mono{font-size:.9em;font-variant-numeric:tabular-nums}',
    F + '.mg-mat table{width:100%;border-collapse:separate;border-spacing:5px;font-size:.8rem;margin:-5px;width:calc(100% + 10px);table-layout:fixed}',
    F + '.mg-mat th{position:static;background:none;padding:.1rem .2rem;border:0;font-family:var(--f-ui);font-size:var(--fs-xs);letter-spacing:.02em;text-transform:none;color:var(--ink);font-weight:500;text-align:center;vertical-align:bottom}',
    F + '.mg-mat th i{font-family:var(--f-body);font-size:1.2em}',
    F + '.mg-mat th.rh{text-align:left;width:4rem;vertical-align:middle;color:var(--ink-2)}',
    F + '.mg-mat td{padding:0;border:0;vertical-align:middle}',
    F + '.mg-mat .cell{width:100%;display:grid;gap:.1rem;justify-items:center;text-align:center;padding:.32rem .15rem;border-radius:4px;border:1px solid var(--rule);background:var(--paper-2);font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:var(--fs-xs);color:var(--ink-2);cursor:pointer;line-height:1.25}',
    F + '.mg-mat .cell .ic{font-size:.98rem;line-height:1}',
    F + '.mg-mat .cell .fw,' + F + '.mg-mat .cell .nt{font-family:var(--f-ui);font-weight:500;letter-spacing:.01em}',
    F + '.mg-mat .cell .nt{white-space:nowrap}',
    F + '.mg-mat .cell.ok{border-color:var(--moss);background:var(--moss-soft);color:var(--moss-ink)}',
    F + '.mg-mat .cell.bad{border-color:var(--seal);background:var(--seal-soft);color:var(--seal-ink)}',
    F + '.mg-mat .cell.na{cursor:default;border-style:dashed;background:transparent;color:var(--ink-2)}',
    F + '.mg-mat .cell.na .ic{opacity:.6}',
    F + '.mg-mat .cell.cur{outline:2px solid var(--ochre);outline-offset:1px}',
    F + '.mg-mat .cell:hover:not(.na){filter:brightness(1.04)}',
    F + '.mg-mat .cap{margin:.45rem 0 0;font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--ink-2)}',
    F + '.mg-mat .cap .mono{font-size:.9em}',
    F + '.mg-foot{margin-top:.85rem;padding-top:.55rem;border-top:1px solid var(--rule);display:grid;gap:.4rem;font-family:var(--f-body);font-size:.84rem;line-height:1.5;color:var(--ink-2)}',
    F + '.mg-foot a{color:var(--ink)}',
    F + '.mg-foot .mg-meth{display:inline-flex;flex-wrap:wrap;gap:.3rem;align-items:center}',
    F + '.mg-foot .mg-meth b{font-family:var(--f-display);font-size:.98rem;font-weight:var(--w-head);color:var(--ink);margin:0 .15rem 0 .4rem}',
    F + '.mg-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
  ].join('\n');

  /* ---------------- the toy block ---------------- */
  var D = 8, DFF = 16, T = 6, NSEQ = 6, NLS = 12, RK = 2, NPRE = 2, ROPE_BASE = 10000;

  /* ---------------- float64 helpers ---------------- */
  function vz(n) { return new Float64Array(n); }
  function vadd(a, b) { var r = vz(a.length); for (var i = 0; i < a.length; i++) r[i] = a[i] + b[i]; return r; }
  function vmul(a, b) { var r = vz(a.length); for (var i = 0; i < a.length; i++) r[i] = a[i] * b[i]; return r; }
  function vmap(a, f) { var r = vz(a.length); for (var i = 0; i < a.length; i++) r[i] = f(a[i]); return r; }
  function affd(g, x, b) { var r = vz(x.length); for (var i = 0; i < x.length; i++) r[i] = g[i] * x[i] + b[i]; return r; }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function mv(W, x) { var m = W.length, n = x.length, r = vz(m); for (var i = 0; i < m; i++) { var s = 0, Wi = W[i]; for (var j = 0; j < n; j++) s += Wi[j] * x[j]; r[i] = s; } return r; }
  function dL(l, M) { return M.map(function (row, i) { var r = vz(row.length); for (var j = 0; j < row.length; j++) r[j] = l[i] * row[j]; return r; }); }
  function dR(M, l) { return M.map(function (row) { var r = vz(row.length); for (var j = 0; j < row.length; j++) r[j] = row[j] * l[j]; return r; }); }
  function madd(A, B) { return A.map(function (row, i) { return vadd(row, B[i]); }); }
  function mmul(A, B) { var m = A.length, k = B.length, n = B[0].length, C = []; for (var i = 0; i < m; i++) { var r = vz(n); for (var p = 0; p < k; p++) { var a = A[i][p]; for (var j = 0; j < n; j++) r[j] += a * B[p][j]; } C.push(r); } return C; }
  function mT(A) { var m = A.length, n = A[0].length, B = []; for (var j = 0; j < n; j++) { var r = vz(m); for (var i = 0; i < m; i++) r[i] = A[i][j]; B.push(r); } return B; }
  function cp(o) { var r = {}; for (var k in o) r[k] = o[k]; return r; }
  function gelu(t) { return 0.5 * t * (1 + Math.tanh(0.7978845608028654 * (t + 0.044715 * t * t * t))); }
  function relu(t) { return t > 0 ? t : 0; }
  function silu(t) { return t / (1 + Math.exp(-t)); }
  function actFn(a) { return a === 'relu' ? relu : a === 'gelu' ? gelu : silu; }
  function lnorm(x, g, b) {
    var n = x.length, mu = 0, v = 0, i, r = vz(n);
    for (i = 0; i < n; i++) mu += x[i];
    mu /= n;
    for (i = 0; i < n; i++) v += (x[i] - mu) * (x[i] - mu);
    v /= n;
    var s = 1 / Math.sqrt(v + 1e-5);
    for (i = 0; i < n; i++) r[i] = g[i] * (x[i] - mu) * s + b[i];
    return r;
  }
  /* RoPE on pairs (2i, 2i+1) with angle pos·base^(−2i/d) */
  function rope(x, pos) {
    var r = Float64Array.from(x);
    for (var i = 0; 2 * i + 1 < x.length; i++) {
      var th = pos * Math.pow(ROPE_BASE, -2 * i / x.length), c = Math.cos(th), s = Math.sin(th), a = x[2 * i], b = x[2 * i + 1];
      r[2 * i] = c * a - s * b; r[2 * i + 1] = s * a + c * b;
    }
    return r;
  }
  /* Gaussian elimination with partial pivoting: solve G Z = H (G p×p, H p×q), returns Z */
  function solve(G, H) {
    var p = G.length, q = H[0].length, A = [], i, j, k;
    for (i = 0; i < p; i++) { var row = vz(p + q); for (j = 0; j < p; j++) row[j] = G[i][j]; for (j = 0; j < q; j++) row[p + j] = H[i][j]; A.push(row); }
    for (k = 0; k < p; k++) {
      var piv = k, best = Math.abs(A[k][k]);
      for (i = k + 1; i < p; i++) if (Math.abs(A[i][k]) > best) { best = Math.abs(A[i][k]); piv = i; }
      if (piv !== k) { var tmp = A[k]; A[k] = A[piv]; A[piv] = tmp; }
      var d = A[k][k];
      if (Math.abs(d) < 1e-300) continue;
      for (i = k + 1; i < p; i++) { var f = A[i][k] / d; if (f === 0) continue; for (j = k; j < p + q; j++) A[i][j] -= f * A[k][j]; }
    }
    var Z = []; for (i = 0; i < p; i++) Z.push(vz(q));
    for (i = p - 1; i >= 0; i--) for (j = 0; j < q; j++) {
      var s = A[i][p + j];
      for (k = i + 1; k < p; k++) s -= A[i][k] * Z[k][j];
      Z[i][j] = Math.abs(A[i][i]) < 1e-300 ? 0 : s / A[i][i];
    }
    return Z;
  }
  /* least squares: B (q×p) minimising Σ‖B x_n − y_n‖² (a 1e−13 relative ridge only guards singular designs) */
  function lstsq(Xr, Yr) {
    var p = Xr[0].length, q = Yr[0].length, G = [], H = [], i, j, n;
    for (i = 0; i < p; i++) { G.push(vz(p)); H.push(vz(q)); }
    for (n = 0; n < Xr.length; n++) {
      var x = Xr[n], y = Yr[n];
      for (i = 0; i < p; i++) { var xi = x[i]; if (xi === 0) continue; var Gi = G[i], Hi = H[i]; for (j = 0; j < p; j++) Gi[j] += xi * x[j]; for (j = 0; j < q; j++) Hi[j] += xi * y[j]; }
    }
    var tr = 0; for (i = 0; i < p; i++) tr += G[i][i];
    for (i = 0; i < p; i++) G[i][i] += 1e-13 * tr / p;
    return mT(solve(G, H));
  }
  function relErr(Y1, Y2) {
    var num = 0, den = 0;
    for (var s = 0; s < Y1.length; s++) for (var t = 0; t < Y1[s].length; t++) for (var i = 0; i < D; i++) { var d = Y1[s][t][i] - Y2[s][t][i]; num += d * d; den += Y1[s][t][i] * Y1[s][t][i]; }
    return Math.sqrt(num / den);
  }

  /* ---------------- seeded weights, method parameters and probes ---------------- */
  function gen(seed, A) {
    var LA = A.LA, R = A.rng(20261001 + 7919 * seed), R2 = A.rng(424242 + 104729 * seed), R3 = A.rng(777 + 31 * seed);
    function v(r, n, s, c) { var x = vz(n); for (var i = 0; i < n; i++) x[i] = (c || 0) + s * r.normal(); return x; }
    var s8 = 1 / Math.sqrt(D), s16 = 1 / Math.sqrt(DFF);
    var W = {
      g1: v(R, D, 0.15, 1), b1: v(R, D, 0.15),
      Wq: LA.randn(D, D, R, s8), Wk: LA.randn(D, D, R, s8), Wv: LA.randn(D, D, R, s8), Wo: LA.randn(D, D, R, s8),
      bq: v(R, D, 0.1), bk: v(R, D, 0.1), bv: v(R, D, 0.1), bo: v(R, D, 0.1),
      g2: v(R, D, 0.15, 1), b2: v(R, D, 0.15),
      Wup: LA.randn(DFF, D, R, s8), Wgate: LA.randn(DFF, D, R, s8), Wdown: LA.randn(D, DFF, R, s16),
      bup: v(R, DFF, 0.1), bgate: v(R, DFF, 0.1), bdown: v(R, D, 0.1),
    };
    /* (IA)^3 vectors: a draw around the identity point ℓ = 1. HF does not constrain the sign, so the FFN vector
       is a generic point of R^16; if the draw happens to be all-positive, the smallest entry is negated so that
       the sign-sensitive rule R3 has something to act on. "ℓ ≥ 0" uses |ℓ|. */
    var ellF = v(R2, DFF, 0.9, 1), hasNeg = false, iMin = 0, i;
    for (i = 0; i < DFF; i++) { if (ellF[i] < 0) hasNeg = true; if (ellF[i] < ellF[iMin]) iMin = i; }
    if (!hasNeg) ellF[iMin] = -Math.abs(ellF[iMin]);
    /* OFT: block-diagonal exact Cayley rotation, block size 4 */
    function cayley(b) {
      var Q = LA.zeros(b, b), a, c;
      for (a = 0; a < b; a++) for (c = a + 1; c < b; c++) { var z = 0.35 * R2.normal(); Q[a][c] = z; Q[c][a] = -z; }
      var IpQ = LA.zeros(b, b), ImQ = LA.zeros(b, b);
      for (a = 0; a < b; a++) for (c = 0; c < b; c++) { IpQ[a][c] = (a === c ? 1 : 0) + Q[a][c]; ImQ[a][c] = (a === c ? 1 : 0) - Q[a][c]; }
      var I = LA.eye(b);
      return mmul(IpQ, solve(ImQ, I));
    }
    var Rb = [cayley(4), cayley(4)], Rm = LA.zeros(D, D);
    for (var bk = 0; bk < 2; bk++) for (var a = 0; a < 4; a++) for (var c = 0; c < 4; c++) Rm[4 * bk + a][4 * bk + c] = Rb[bk][a][c];
    var RtR = mmul(mT(Rm), Rm), orth = 0;
    for (a = 0; a < D; a++) for (c = 0; c < D; c++) orth = Math.max(orth, Math.abs(RtR[a][c] - (a === c ? 1 : 0)));
    var M = {
      ellF: ellF, ellK: v(R2, D, 0.6, 1), ellV: v(R2, D, 0.6, 1),
      ssfG: v(R2, D, 0.3, 1), ssfB: v(R2, D, 0.3),
      lnG1: vadd(W.g1, v(R2, D, 0.3)), lnB1: vadd(W.b1, v(R2, D, 0.3)),
      lnG2: vadd(W.g2, v(R2, D, 0.3)), lnB2: vadd(W.b2, v(R2, D, 0.3)),
      loraB: { q: LA.randn(D, RK, R2, 0.5), v: LA.randn(D, RK, R2, 0.5), up: LA.randn(DFF, RK, R2, 0.5) },
      loraA: { q: LA.randn(RK, D, R2, 0.5), v: LA.randn(RK, D, R2, 0.5), up: LA.randn(RK, D, R2, 0.5) },
      oftRt: mT(Rm), oftOrth: orth,
      adD: LA.randn(RK, D, R2, 0.6), adU: LA.randn(D, RK, R2, 0.8), adbD: v(R2, RK, 0.2), adbU: v(R2, D, 0.1),
      Pk: LA.randn(NPRE, D, R2, 1.2), Pv: LA.randn(NPRE, D, R2, 1),
    };
    function seqs(n) { var out = []; for (var s = 0; s < n; s++) { var q = []; for (var t = 0; t < T; t++) q.push(v(R3, D, 1)); out.push(q); } return out; }
    return { W: W, M: M, X: seqs(NSEQ), XL: seqs(NLS), seed: seed };
  }

  /* ---------------- the block forward pass, with hooks for unmerged (adapted) methods ---------------- */
  function linb(W, name, x, cfg) { var y = mv(W['W' + name], x); return cfg.bias ? vadd(y, W['b' + name]) : y; }
  function forward(W, X, cfg, hk) {
    hk = hk || {};
    var pre = hk.pre || {}, post = hk.post || {}, sq = Math.sqrt(D);
    var Y = [], lam = [], oIn = [], oOut = [], dIn = [], dOut = [];
    var sig = actFn(cfg.act === 'swiglu' ? 'silu' : cfg.act);
    function lin(name, x) {
      var xin = pre[name] ? pre[name](x) : x;
      var y = linb(W, name, xin, cfg);
      return post[name] ? post[name](y, x) : y;
    }
    for (var s = 0; s < X.length; s++) {
      var xs = X[s], n = xs.length, i, j, d, q = [], k = [], v = [];
      for (i = 0; i < n; i++) {
        var z = lnorm(xs[i], W.g1, W.b1);
        if (hk.ln1) z = hk.ln1(z);
        var qi = lin('q', z), ki = lin('k', z);
        if (cfg.rope) { qi = rope(qi, i); ki = rope(ki, i); }
        q.push(qi); k.push(ki); v.push(lin('v', z));
      }
      var ys = [];
      for (i = 0; i < n; i++) {
        var sc = [], vals = [], npre = 0;
        for (j = 0; j <= i; j++) { sc.push(dot(q[i], k[j]) / sq); vals.push(v[j]); }
        if (hk.prefix) for (var p = 0; p < hk.prefix.K.length; p++) { sc.push(dot(q[i], hk.prefix.K[p]) / sq); vals.push(hk.prefix.V[p]); npre++; }
        var mx = -Infinity; for (j = 0; j < sc.length; j++) if (sc[j] > mx) mx = sc[j];
        var e = [], Z = 0; for (j = 0; j < sc.length; j++) { e.push(Math.exp(sc[j] - mx)); Z += e[j]; }
        var o = vz(D); for (j = 0; j < sc.length; j++) { var w = e[j] / Z; for (d = 0; d < D; d++) o[d] += w * vals[j][d]; }
        if (npre) { var lp = 0; for (j = sc.length - npre; j < sc.length; j++) lp += e[j] / Z; lam.push(lp); }
        var a = lin('o', o); oIn.push(o); oOut.push(a);
        var h = vadd(xs[i], a);
        var z2 = lnorm(h, W.g2, W.b2);
        if (hk.ln2) z2 = hk.ln2(z2);
        var hid;
        if (cfg.act === 'swiglu') { var g = lin('gate', z2), up = lin('up', z2); hid = vz(DFF); for (d = 0; d < DFF; d++) hid[d] = silu(g[d]) * up[d]; }
        else hid = vmap(lin('up', z2), sig);
        var f = lin('down', hid); dIn.push(hid); dOut.push(f);
        ys.push(vadd(h, f));
      }
      Y.push(ys);
    }
    return { Y: Y, lam: lam, oIn: oIn, oOut: oOut, dIn: dIn, dOut: dOut };
  }

  /* ℓ for (IA)^3 at a site, in the chosen parameter domain */
  function ellAt(M, site, st) {
    var l = Float64Array.from(site === 'ffn' ? M.ellF : site === 'k' ? M.ellK : M.ellV), i;
    if (site === 'k' && st.tie) for (i = 0; i + 1 < l.length; i += 2) l[i + 1] = l[i];
    if (st.sign === 'pos') for (i = 0; i < l.length; i++) l[i] = Math.abs(l[i]);
    return l;
  }
  /* the adapted (unmerged) network: base weights plus hooks that run the PEFT module as trained */
  function adapted(st, WM) {
    var M = WM.M, W = WM.W, hk = { pre: {}, post: {} }, Wa = W, m = st.method, s = st.site;
    var sig = actFn(st.act === 'swiglu' ? 'silu' : st.act);
    if (m === 'ia3') {
      var l = ellAt(M, s, st);
      if (s === 'k') hk.post.k = function (y) { return vmul(l, y); };
      if (s === 'v') hk.post.v = function (y) { return vmul(l, y); };
      if (s === 'ffn') hk.pre.down = function (h) { return vmul(l, h); };
    } else if (m === 'ssf') {
      if (s === 'o') hk.post.o = function (y) { return affd(M.ssfG, y, M.ssfB); };
      else hk.ln1 = function (z) { return affd(M.ssfG, z, M.ssfB); };
    } else if (m === 'ln') {
      Wa = cp(W);
      if (s === 'ln1') { Wa.g1 = M.lnG1; Wa.b1 = M.lnB1; } else { Wa.g2 = M.lnG2; Wa.b2 = M.lnB2; }
    } else if (m === 'lora') {
      var B = M.loraB[s], Am = M.loraA[s];
      hk.post[s] = function (y, x) { return vadd(y, mv(B, mv(Am, x))); };
    } else if (m === 'oft') {
      hk.pre[s] = function (x) { return mv(M.oftRt, x); };
    } else if (m === 'adapter') {
      hk.post[s === 'o' ? 'o' : 'down'] = function (y) { return vadd(vadd(y, mv(M.adU, vmap(vadd(mv(M.adD, y), M.adbD), sig))), M.adbU); };
    } else if (m === 'prefix') {
      hk.prefix = { K: M.Pk, V: M.Pv };
    }
    return { W: Wa, hk: hk };
  }
  /* best affine (or linear, without bias slots) refit of one box from its inputs to the edited outputs */
  function boxRefit(WM, cfg, inputs, targets, box) {
    var rows = inputs.map(function (x) { if (!cfg.bias) return x; var r = vz(x.length + 1); r.set(x); r[x.length] = 1; return r; });
    var B = lstsq(rows, targets), n = inputs[0].length, W2 = cp(WM.W);
    W2['W' + box] = B.map(function (r) { return Float64Array.from(r.subarray(0, n)); });
    if (cfg.bias) W2['b' + box] = Float64Array.from(B.map(function (r) { return r[n]; }));
    return W2;
  }
  /* prefix: refit (W_V, b_V) of the base head so that its content-only attention best matches the prefixed head */
  function prefixRefit(WM, cfg, A1) {
    var W = WM.W, rows = [], sq = Math.sqrt(D);
    WM.X.forEach(function (xs) {
      var u = xs.map(function (x) { return lnorm(x, W.g1, W.b1); });
      var q = u.map(function (z, i) { var r = linb(W, 'q', z, cfg); return cfg.rope ? rope(r, i) : r; });
      var k = u.map(function (z, i) { var r = linb(W, 'k', z, cfg); return cfg.rope ? rope(r, i) : r; });
      for (var i = 0; i < xs.length; i++) {
        var sc = [], j, mx = -Infinity, Z = 0;
        for (j = 0; j <= i; j++) { sc.push(dot(q[i], k[j]) / sq); if (sc[j] > mx) mx = sc[j]; }
        var e = sc.map(function (c) { var t = Math.exp(c - mx); Z += t; return t; });
        var r = vz(cfg.bias ? D + 1 : D);
        for (j = 0; j <= i; j++) { for (var d = 0; d < D; d++) r[d] += e[j] / Z * u[j][d]; if (cfg.bias) r[D] += e[j] / Z; }
        rows.push(r);
      }
    });
    var B = lstsq(rows, A1.oIn), W2 = cp(W);
    W2.Wv = B.map(function (r) { return Float64Array.from(r.subarray(0, D)); });
    if (cfg.bias) W2.bv = Float64Array.from(B.map(function (r) { return r[D]; }));
    return W2;
  }
  /* RoPE: best W_Q (and b_Q) by least squares on the causal attention logits, against keys scaled by ℓ */
  function ropeLS(WM, cfg, l) {
    var W = WM.W, Phi = [], tg = [];
    WM.XL.forEach(function (xs) {
      var u = xs.map(function (x) { return lnorm(x, W.g1, W.b1); });
      var q = u.map(function (z) { return linb(W, 'q', z, cfg); }), k = u.map(function (z) { return linb(W, 'k', z, cfg); });
      for (var i = 0; i < xs.length; i++) for (var j = 0; j <= i; j++) {
        var w = rope(k[j], j - i), f = vz(cfg.bias ? D * D + D : D * D), a, b;
        for (a = 0; a < D; a++) for (b = 0; b < D; b++) f[a * D + b] = w[a] * u[i][b];
        if (cfg.bias) for (a = 0; a < D; a++) f[D * D + a] = w[a];
        Phi.push(f); tg.push([dot(rope(q[i], i), rope(vmul(l, k[j]), j))]);
      }
    });
    var B = lstsq(Phi, tg)[0], num = 0, den = 0;
    Phi.forEach(function (f, n) { var r = dot(B, f) - tg[n][0]; num += r * r; den += tg[n][0] * tg[n][0]; });
    return { res: Math.sqrt(num / den), neq: Phi.length, nunk: Phi[0].length };
  }

  /* ---------------- diagram geometry (viewBox 420 × 716) ---------------- */
  var VBX = 14, VBW = 400, VBH = 742;
  var XR = 30, XC = 230, XQ = 130, XK = 230, XV = 330, XG = 165, XU = 295, XS = 180, XPF = 388;
  var Y = { xin: 15, sp0: 34, ln1: 58, ssf: 81, sp1: 96, oftq: 108, proj: 132, kv: 157, rope: 178, cup: 214, smax: 250, mix: 288, wo: 326, so: 357, add1: 386, sp2: 402, ln2: 424, sp3: 449, up: 490, act: 528, had: 558, hid: 580, down: 608, sd: 640, add2: 670 };
  function geo(sw, rope) {
    return {
      ln1: { x: XC, y: Y.ln1, w: 66, h: 24, k: 'ln', lab: [['LN', 'r'], ['1', 's']] },
      wq: { x: XQ, y: Y.proj, w: 54, h: 28, k: 'lin', lab: [['W', 'i'], ['Q', 'si']] },
      wk: { x: XK, y: Y.proj, w: 54, h: 28, k: 'lin', lab: [['W', 'i'], ['K', 'si']] },
      wv: { x: XV, y: Y.proj, w: 54, h: 28, k: 'lin', lab: [['W', 'i'], ['V', 'si']] },
      ropeq: rope ? { x: XQ, y: Y.rope, w: 46, h: 24, k: 'rope', lab: [['R', 'i'], ['m', 'si']] } : null,
      ropek: rope ? { x: XK, y: Y.rope, w: 46, h: 24, k: 'rope', lab: [['R', 'i'], ['n', 'si']] } : null,
      cup: { x: XS, y: Y.cup, w: 128, h: 24, k: 'nl', lab: [['⟨', 'r'], ['q', 'i'], [', ', 'r'], ['k', 'i'], ['⟩ / √', 'r'], ['d', 'i']] },
      smax: { x: XS, y: Y.smax, w: 86, h: 24, k: 'nl', lab: [['softmax', 'r']] },
      mix: { x: 255, y: Y.mix, w: 178, h: 24, k: 'mix', lab: [['Σ ', 'r'], ['a', 'i'], ['mn', 'si'], [' v', 'i'], ['n', 'si']] },
      wo: { x: XC, y: Y.wo, w: 54, h: 28, k: 'lin', lab: [['W', 'i'], ['O', 'si']] },
      add1: { x: XC, y: Y.add1, r: 8, k: 'add' },
      ln2: { x: XC, y: Y.ln2, w: 66, h: 24, k: 'ln', lab: [['LN', 'r'], ['2', 's']] },
      wgate: sw ? { x: XG, y: Y.up, w: 58, h: 28, k: 'lin', lab: [['W', 'i'], ['gate', 's']] } : null,
      wup: { x: sw ? XU : XC, y: Y.up, w: 56, h: 28, k: 'lin', lab: [['W', 'i'], ['up', 's']] },
      act: { x: sw ? XG : XC, y: Y.act, w: sw ? 58 : 66, h: 22, k: 'nl', lab: null },
      had: sw ? { x: XC, y: Y.had, r: 8, k: 'had' } : null,
      wdown: { x: XC, y: Y.down, w: 66, h: 28, k: 'lin', lab: [['W', 'i'], ['down', 's']] },
      add2: { x: XC, y: Y.add2, r: 8, k: 'add' },
      adO: { x: XC, y: Y.so, w: 106, h: 20, k: 'adapter', lab: [['+ ', 'r'], ['U', 'i'], ['σ', 'i'], ['(', 'r'], ['D', 'i'], ['·)', 'r']] },
      adD: { x: XC, y: Y.sd, w: 106, h: 20, k: 'adapter', lab: [['+ ', 'r'], ['U', 'i'], ['σ', 'i'], ['(', 'r'], ['D', 'i'], ['·)', 'r']] },
      prefix: { x: XPF, y: 232, w: 40, h: 44, k: 'prefix', lab: [['P', 'i']] },
    };
  }
  function wiresFor(sw, rope) {
    var w = [
      { p: [[XC, 22], [XC, Y.ln1 - 12]], a: 1 },
      { p: [[XC, Y.sp0], [XR, Y.sp0], [XR, Y.add1], [XC - 8, Y.add1]], a: 1 },
      { p: [[XC, Y.ln1 + 12], [XC, Y.sp1]] },
      { p: [[XC, Y.sp1], [XQ, Y.sp1], [XQ, Y.proj - 14]], a: 1 },
      { p: [[XC, Y.sp1], [XK, Y.proj - 14]], a: 1 },
      { p: [[XC, Y.sp1], [XV, Y.sp1], [XV, Y.proj - 14]], a: 1 },
      { p: [[XQ, Y.proj + 14], [XQ, Y.cup - 12]], a: 1 },
      { p: [[XK, Y.proj + 14], [XK, Y.cup - 12]], a: 1 },
      { p: [[XV, Y.proj + 14], [XV, Y.mix - 12]], a: 1 },
      { p: [[XS, Y.cup + 12], [XS, Y.smax - 12]], a: 1 },
      { p: [[XS, Y.smax + 12], [XS, Y.mix - 12]], a: 1 },
      { p: [[XC, Y.mix + 12], [XC, Y.wo - 14]], a: 1 },
      { p: [[XC, Y.wo + 14], [XC, Y.add1 - 8]], a: 1 },
      { p: [[XC, Y.add1 + 8], [XC, Y.ln2 - 12]], a: 1 },
      { p: [[XC, Y.sp2], [XR, Y.sp2], [XR, Y.add2], [XC - 8, Y.add2]], a: 1 },
      { p: [[XC, Y.down + 14], [XC, Y.add2 - 8]], a: 1 },
      { p: [[XC, Y.add2 + 8], [XC, 692]], a: 1 },
    ];
    if (sw) {
      w.push({ p: [[XC, Y.ln2 + 12], [XC, Y.sp3]] });
      w.push({ p: [[XC, Y.sp3], [XG, Y.sp3], [XG, Y.up - 14]], a: 1 });
      w.push({ p: [[XC, Y.sp3], [XU, Y.sp3], [XU, Y.up - 14]], a: 1 });
      w.push({ p: [[XG, Y.up + 14], [XG, Y.act - 11]], a: 1 });
      w.push({ p: [[XG, Y.act + 11], [XG, Y.had], [XC - 8, Y.had]], a: 1 });
      w.push({ p: [[XU, Y.up + 14], [XU, Y.had], [XC + 8, Y.had]], a: 1 });
      w.push({ p: [[XC, Y.had + 8], [XC, Y.down - 14]], a: 1 });
    } else {
      w.push({ p: [[XC, Y.ln2 + 12], [XC, Y.up - 14]], a: 1 });
      w.push({ p: [[XC, Y.up + 14], [XC, Y.act - 11]], a: 1 });
      w.push({ p: [[XC, Y.act + 11], [XC, Y.down - 14]], a: 1 });
    }
    return w;
  }
  function backOff(p, d) {
    var L = polyLen(p), want = Math.max(0, L - d), acc = 0;
    for (var i = 1; i < p.length; i++) { var sl = Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); if (acc + sl >= want) { var f = sl ? (want - acc) / sl : 0; return [p[i - 1][0] + f * (p[i][0] - p[i - 1][0]), p[i - 1][1] + f * (p[i][1] - p[i - 1][1])]; } acc += sl; }
    return p[0];
  }
  function polyLen(p) { var L = 0; for (var i = 1; i < p.length; i++) L += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); return L; }
  function rpath(p, r) {
    var d = 'M' + p[0][0] + ',' + p[0][1];
    for (var i = 1; i < p.length - 1; i++) {
      var a = p[i - 1], b = p[i], c = p[i + 1];
      var d1 = Math.hypot(b[0] - a[0], b[1] - a[1]), d2 = Math.hypot(c[0] - b[0], c[1] - b[1]);
      if (!d1 || !d2) continue;
      var rr = Math.min(r, d1 / 2, d2 / 2);
      var p1 = [b[0] + (a[0] - b[0]) * rr / d1, b[1] + (a[1] - b[1]) * rr / d1], p2 = [b[0] + (c[0] - b[0]) * rr / d2, b[1] + (c[1] - b[1]) * rr / d2];
      d += 'L' + p1[0].toFixed(2) + ',' + p1[1].toFixed(2) + 'Q' + b[0] + ',' + b[1] + ' ' + p2[0].toFixed(2) + ',' + p2[1].toFixed(2);
    }
    var z = p[p.length - 1];
    return d + 'L' + z[0] + ',' + z[1];
  }

  /* ---------------- method catalogue ---------------- */
  var Wh = function (s, it) { return '<i>W</i><sub>' + (it ? '<i>' + s + '</i>' : s) + '</sub>'; };
  var METHODS = [
    { id: 'ia3', label: '(IA)³', data: 'ia3', sites: [{ v: 'k', h: 'keys <i>k</i>' }, { v: 'v', h: 'values <i>v</i>' }, { v: 'ffn', h: 'FFN hidden <i>h</i>' }] },
    { id: 'ssf', label: 'SSF', data: 'ssf', sites: [{ v: 'ln1', h: 'after LN<sub>1</sub>' }, { v: 'o', h: 'after ' + Wh('O', 1) }] },
    { id: 'ln', label: 'LN-tuning', data: 'ln-tuning', sites: [{ v: 'ln1', h: 'LN<sub>1</sub>' }, { v: 'ln2', h: 'LN<sub>2</sub>' }] },
    { id: 'lora', label: 'LoRA', data: 'lora', sites: [{ v: 'q', h: Wh('Q', 1) }, { v: 'v', h: Wh('V', 1) }, { v: 'up', h: Wh('up') }] },
    { id: 'oft', label: 'OFT', data: 'oft', sites: [{ v: 'q', h: Wh('Q', 1) }, { v: 'up', h: Wh('up') }] },
    { id: 'adapter', label: 'Adapter', data: 'houlsby-adapter', sites: [{ v: 'o', h: 'after ' + Wh('O', 1) }, { v: 'down', h: 'after ' + Wh('down') }] },
    { id: 'prefix', label: 'Prefix', data: 'prefix-tuning', sites: [{ v: 'attn', h: 'attention' }] },
  ];
  function methodDef(id) { for (var i = 0; i < METHODS.length; i++) if (METHODS[i].id === id) return METHODS[i]; return METHODS[0]; }
  var TLAB = {
    wq: Wh('Q', 1), wk: Wh('K', 1), wv: Wh('V', 1), wo: Wh('O', 1), wup: Wh('up'), wgate: Wh('gate'), wdown: Wh('down'),
    ln1: 'LN<sub>1</sub> itself', inplace: 'in place', fold1: 'fold into ' + Wh('Q,K,V', 1), fold2: 'fold into FFN', head: 'the head',
  };
  function targetsFor(m, s, act) {
    var sw = act === 'swiglu';
    var k = m + '/' + s;
    var L = {
      'ia3/k': ['wk', 'wq'], 'ia3/v': ['wv'], 'ia3/ffn': sw ? ['wdown', 'wup', 'wgate'] : ['wdown', 'wup'],
      'ssf/o': ['wo'], 'ssf/ln1': ['ln1'], 'ln/ln1': ['inplace', 'fold1'], 'ln/ln2': ['inplace', 'fold2'],
      'lora/q': ['wq'], 'lora/v': ['wv'], 'lora/up': ['wup'], 'oft/q': ['wq'], 'oft/up': ['wup'],
      'adapter/o': ['wo'], 'adapter/down': ['wdown'], 'prefix/attn': ['head'],
    }[k] || ['wq'];
    return L;
  }
  var DEFAULT_TARGET = { 'ia3/k': 'wq', 'ia3/ffn': 'wup', 'ln/ln1': 'fold1', 'ln/ln2': 'fold2' };
  var BOXNAME = { wq: 'q', wk: 'k', wv: 'v', wo: 'o', wup: 'up', wgate: 'gate', wdown: 'down' };

  /* ---------------- the rule engine ----------------
     plan(st, M) returns the rewrite sequence for (method, site, target) under (σ, RoPE, bias slots, ℓ domain):
     steps with the rule used, its statement, the side condition, and the diagram path the edit travels;
     ok/blocked; the merged weights for a licensed rewrite, and what "force merge" does when it is refused. */
  function plan(st, M) {
    var m = st.method, s = st.site, t = st.target, sw = st.act === 'swiglu', bias = st.bias, rp = st.rope;
    var xu = sw ? XU : XC, aN = st.act === 'gelu' ? 'GELU' : st.act === 'relu' ? 'ReLU' : 'SiLU';
    var o = { steps: [], ok: true, kind: 'fuse', tgt: [], block: null, fpaths: null, tok: null, edit: null, ref: '', vt: '', merged: null, forced: null, fdesc: '', ex: null };
    function S(rule, name, tex, note, paths, chip) { var x = { rule: rule, name: name, tex: tex, note: note, paths: paths || [], chip: chip }; o.steps.push(x); return x; }
    function X(rule, name, tex, note, paths, chip, at) { var x = S(rule, name, tex, note, paths, chip); x.bad = true; o.ok = false; o.kind = 'blocked'; o.block = at; return x; }
    function rows(box, l) { return function (WM) { var W2 = cp(WM.W); W2['W' + box] = dL(l, WM.W['W' + box]); W2['b' + box] = vmul(l, WM.W['b' + box]); return W2; }; }
    var bTex = function (box) { return bias ? '+b_{' + box + '}' : ''; };

    if (m === 'ia3') {
      var l = ellAt(M, s, st);
      o.ex = { l: l };
      if (s === 'ffn') {
        o.tok = { x: XC, y: Y.hid, lab: [['ℓ', 'i'], [' ⊙', 'r']], w: 36 };
        o.edit = { tex: 'h\\mapsto \\ell\\odot h,\\qquad \\ell\\in\\mathbb R^{16}', note: '(IA)³ rescales the FFN hidden units, i.e. the input of ' + Wh('down') + ' (as HF PEFT does).' };
        if (t === 'wdown') {
          S('R1', 'fusion', 'W_{\\rm down}(\\ell\\odot h)=(W_{\\rm down}\\,\\mathrm{diag}\\,\\ell)\\,h', 'Holds for every σ and every ℓ.', [[[XC, Y.hid], [XC, Y.down]]], [290, 586]);
          o.tgt = ['wdown']; o.merged = function (WM) { var W2 = cp(WM.W); W2.Wdown = dR(WM.W.Wdown, l); return W2; };
          o.ref = 'Thm VI.3, corollary: (IA)³ on the FFN fuses into W<sub>down</sub> for every σ';
        } else if (t === 'wup') {
          o.tgt = ['wup'];
          o.merged = rows('up', l);
          var r1 = function (y0) { S('R1', 'fusion', '\\mathrm{diag}(\\ell)\\,(W_{\\rm up}x' + bTex('\\rm up') + ')=(\\mathrm{diag}(\\ell)\\,W_{\\rm up})\\,x' + (bias ? '+\\ell\\odot b_{\\rm up}' : ''), 'The scaling lands in ' + Wh('up') + (bias ? ' and its bias.' : '.'), [[[xu, y0], [xu, Y.up]]], sw ? [XU + 34, 514] : [292, 509]); };
          if (sw) {
            S('R4', 'spider slide', '\\mathrm{diag}(\\ell)\\big(\\mathrm{silu}(g)\\odot u\\big)=\\mathrm{silu}(g)\\odot\\big(\\mathrm{diag}(\\ell)\\,u\\big)', 'The Hadamard product lets ℓ onto the linear branch, for every ℓ of either sign.', [[[XC, Y.hid], [XC, Y.had], [XU, Y.had], [XU, 517]]], [263, 572]);
            r1(517);
            o.ref = 'Thm VI.3, corollary: under SwiGLU, (IA)³ fuses into W<sub>up</sub> for every ℓ (R4; AWQ migrates scales the same way)';
          } else if (st.act === 'relu' && st.sign === 'pos') {
            S('R3', 'positive-diagonal slide', '\\mathrm{diag}(\\ell)\\,\\mathrm{ReLU}(z)=\\mathrm{ReLU}(\\mathrm{diag}(\\ell)\\,z)', 'ReLU is positively homogeneous and here no ℓ<sub>i</sub> is negative.', [[[XC, Y.hid], [XC, Y.act], [XC, 509]]], [292, 528]);
            r1(509);
            o.ref = 'Thm VI.3, corollary: under ReLU, (IA)³ fuses into W<sub>up</sub> when ℓ ≥ 0 (R3)';
          } else if (st.act === 'relu') {
            X('R3', 'positive-diagonal slide', '\\ell_i\\,\\mathrm{ReLU}(t)\\neq\\mathrm{ReLU}(\\ell_i t)\\quad\\text{if }\\ell_i<0', 'R3 needs ℓ ≥ 0, but (IA)³ trains ℓ over all of ℝ<sup>16</sup>, and this ℓ has {negs}. Thm VI.3(b) asks the slide to work for every ℓ in the parameter domain; on ℓ ≥ 0 it does. A unit with ℓ<sub>i</sub> &lt; 0 outputs ℓ<sub>i</sub>ReLU(t) &lt; 0 wherever it is active, while a ReLU never outputs a negative value, so ' + Wh('up') + ' cannot absorb it.', [[[XC, Y.hid], [XC, 542]]], [298, 528], 'act');
            o.forced = rows('up', l); o.fpaths = [[[XC, 542], [XC, Y.up]]];
            o.fdesc = 'slide diag(ℓ) through ReLU anyway: ' + Wh('up') + '′ = diag(ℓ)' + Wh('up');
            o.ref = 'Thm VI.3(b); R3 is licensed only on the restricted domain ℓ ≥ 0';
          } else {
            X('R3', 'positive-diagonal slide', 'c\\,\\mathrm{GELU}(t)=\\mathrm{GELU}(ct)\\ \\ \\forall t\\iff c\\in\\{0,1\\}', 'GELU is not positively homogeneous (compare Taylor coefficients at 0, Lemma VI.2(c)). Here {off01s} outside {0, 1}.', [[[XC, Y.hid], [XC, 542]]], [298, 528], 'act');
            o.forced = rows('up', l); o.fpaths = [[[XC, 542], [XC, Y.up]]];
            o.fdesc = 'slide diag(ℓ) through GELU anyway: ' + Wh('up') + '′ = diag(ℓ)' + Wh('up');
            o.ref = 'Thm VI.3, corollary: under GELU, (IA)³ does not fuse into W<sub>up</sub>; for generic weights only entries ℓ<sub>i</sub> ∈ {0, 1} fold';
          }
        } else { /* wgate, SwiGLU only */
          o.tgt = ['wgate'];
          S('R4', 'spider slide', '\\mathrm{diag}(\\ell)\\,(a\\odot b)=(\\mathrm{diag}(\\ell)\\,a)\\odot b', 'Onto the gate branch: still an identity.', [[[XC, Y.hid], [XC, Y.had], [XG, Y.had], [XG, 542]]], [197, 572]);
          X('R3', 'positive-diagonal slide', 'c\\,\\mathrm{silu}(t)=\\mathrm{silu}(ct)\\ \\ \\forall t\\iff c\\in\\{0,1\\}', 'SiLU is not positively homogeneous, so ℓ cannot cross it ({off01s} outside {0, 1}). The linear branch ' + Wh('up') + ' is the one that absorbs ℓ.', [], [XG - 62, 528], 'act');
          o.forced = rows('gate', l); o.fpaths = [[[XG, 542], [XG, Y.up]]];
          o.fdesc = 'slide diag(ℓ) through SiLU anyway: ' + Wh('gate') + '′ = diag(ℓ)' + Wh('gate');
          o.ref = 'Thm VI.3, R3 and Lemma VI.2(c): SiLU fixes c = 1';
        }
      } else if (s === 'k') {
        o.tok = { x: XK, y: Y.kv, lab: [['ℓ', 'i'], [' ⊙', 'r']], w: 36 };
        o.edit = { tex: 'k\\mapsto \\ell\\odot k,\\qquad \\ell\\in\\mathbb R^{8}', note: '(IA)³ rescales the keys: the output of ' + Wh('K', 1) + ', before RoPE.' };
        if (t === 'wk') {
          S('R1', 'fusion', '\\ell\\odot(W_Kx' + bTex('K') + ')=(\\mathrm{diag}(\\ell)\\,W_K)\\,x' + (bias ? '+\\ell\\odot b_K' : ''), 'Every σ, every ℓ. The scaling sits before RoPE, so RoPE plays no role.', [[[XK, Y.kv], [XK, Y.proj]]], [270, Y.kv]);
          o.tgt = ['wk']; o.merged = rows('k', l);
          o.ref = 'Thm VI.3, corollary: (IA)³ on keys is an output scaling of W<sub>K</sub> and fuses into it';
        } else {
          o.tgt = ['wq'];
          var cupPath = [[XK, Y.kv], [XK, Y.cup], [XQ, Y.cup], [XQ, 150]];
          if (!rp) {
            S('R5', 'cup slide', '\\langle q,\\ \\mathrm{diag}(\\ell)\\,k\\rangle=\\langle \\mathrm{diag}(\\ell)\\,q,\\ k\\rangle', 'The diagonal crosses the attention score to the query side.', [cupPath], [XS, 192]);
          } else if (st.tie) {
            S('R5', 'cup slide under RoPE', '\\langle R_mq,\\ R_n\\,\\mathrm{diag}(\\ell)\\,k\\rangle=\\langle R_m\\,\\mathrm{diag}(\\ell)\\,q,\\ R_nk\\rangle', 'ℓ is constant on each rotary pair, so diag(ℓ) commutes with every rotation R<sub>n</sub>.', [cupPath], [XS, 192]);
          } else {
            X('R5', 'cup slide under RoPE', 'R_n\\,\\mathrm{diag}(\\ell)\\neq\\mathrm{diag}(\\ell)\\,R_n', 'ℓ differs inside rotary pairs, so the scaling does not commute with the rotations; Thm VI.3(b) needs ℓ constant on each pair. Best ' + Wh('Q', 1) + ' by least squares on the logits still misses by {ls}.', [[[XK, Y.kv], [XK, 167]]], [XK + 46, Y.rope], 'ropek');
            o.forced = rows('q', l); o.fpaths = [[[XK, 167], [XK, Y.cup], [XQ, Y.cup], [XQ, Y.proj]]];
            o.fdesc = 'slide diag(ℓ) past the rotations anyway: ' + Wh('Q', 1) + '′ = diag(ℓ)' + Wh('Q', 1);
            o.ref = 'Thm VI.3(b): under RoPE, ℓ must be constant on each rotary pair';
          }
          if (o.ok) {
            S('R1', 'fusion', '\\mathrm{diag}(\\ell)\\,(W_Qx' + bTex('Q') + ')=(\\mathrm{diag}(\\ell)\\,W_Q)\\,x' + (bias ? '+\\ell\\odot b_Q' : ''), 'The query weight absorbs the key scaling.', [[[XQ, 150], [XQ, Y.proj]]], [XQ - 30, 152]);
            o.merged = rows('q', l);
            o.ref = rp ? 'Thm VI.3(b), R5 under RoPE with ℓ pair-constant' : 'Thm VI.3, R5 + R1 (no RoPE)';
          }
        }
      } else { /* v */
        o.tok = { x: XV, y: Y.kv, lab: [['ℓ', 'i'], [' ⊙', 'r']], w: 36 };
        o.edit = { tex: 'v\\mapsto \\ell\\odot v,\\qquad \\ell\\in\\mathbb R^{8}', note: '(IA)³ rescales the values: the output of ' + Wh('V', 1) + '.' };
        S('R1', 'fusion', '\\ell\\odot(W_Vx' + bTex('V') + ')=(\\mathrm{diag}(\\ell)\\,W_V)\\,x' + (bias ? '+\\ell\\odot b_V' : ''), 'Every σ, every ℓ.', [[[XV, Y.kv], [XV, Y.proj]]], [370, Y.kv]);
        o.tgt = ['wv']; o.merged = rows('v', l);
        o.ref = 'Thm VI.3, corollary: (IA)³ on values fuses into W<sub>V</sub>';
      }
    } else if (m === 'ssf') {
      if (s === 'o') {
        o.tok = { x: XC, y: Y.so, lab: [['γ', 'i'], [' ⊙ · + ', 'r'], ['β', 'i']], w: 64 };
        o.edit = { tex: 'y\\mapsto \\gamma\\odot y+\\beta', note: 'SSF scales and shifts the attention output, after ' + Wh('O', 1) + '.' };
        o.tgt = ['wo'];
        var ssfO = function (WM) { var W2 = cp(WM.W); W2.Wo = dL(WM.M.ssfG, WM.W.Wo); W2.bo = vadd(vmul(WM.M.ssfG, WM.W.bo), WM.M.ssfB); return W2; };
        if (bias) {
          S('R1', 'fusion', '\\gamma\\odot(W_Oa+b_O)+\\beta=(\\mathrm{diag}(\\gamma)\\,W_O)\\,a+(\\gamma\\odot b_O+\\beta)', 'The shift lands in the bias slot of ' + Wh('O', 1) + '.', [[[XC, Y.so], [XC, Y.wo]]], [292, Y.so]);
          o.merged = ssfO;
          o.ref = 'Thm VI.3(a): SSF fuses into the preceding box when it has a bias slot';
        } else {
          X('R1', 'fusion', 'a\\mapsto\\gamma\\odot(W_Oa)+\\beta\\ \\notin\\ \\{a\\mapsto Wa\\}\\quad(\\beta\\neq0)', Wh('O', 1) + ' is bias-free, so the shift β has no slot (Thm VI.3(a)). Merging would need a new bias, i.e. an extension of the architecture.', [[[XC, Y.so], [XC, 343]]], [294, Y.wo], 'wo');
          o.forced = ssfO; o.fpaths = [[[XC, 343], [XC, Y.wo]]];
          o.fdesc = 'fold γ into ' + Wh('O', 1) + ' and drop β (there is no slot for it)';
          o.ref = 'Thm VI.3(a): a shift needs a bias slot (bias-free, LLaMA-type layers)';
        }
      } else {
        o.tok = { x: XC, y: Y.ssf, lab: [['γ', 'i'], [' ⊙ · + ', 'r'], ['β', 'i']], w: 64 };
        o.edit = { tex: 'z\\mapsto \\gamma\\odot z+\\beta', note: 'SSF scales and shifts the output of LN<sub>1</sub>.' };
        S('R1', 'fusion', '\\gamma\\odot(\\gamma_1\\odot n(x)+\\beta_1)+\\beta=(\\gamma\\odot\\gamma_1)\\odot n(x)+(\\gamma\\odot\\beta_1+\\beta)', 'The gain and shift of LN<sub>1</sub> absorb it, whether or not the linear boxes have bias slots.', [[[XC, Y.ssf], [XC, Y.ln1]]], [298, Y.ssf]);
        o.tgt = ['ln1'];
        o.merged = function (WM) { var W2 = cp(WM.W); W2.g1 = vmul(WM.M.ssfG, WM.W.g1); W2.b1 = vadd(vmul(WM.M.ssfG, WM.W.b1), WM.M.ssfB); return W2; };
        o.ref = 'Thm VI.3, R1: diagonal affine maps compose (SSF re-parameterisation)';
      }
    } else if (m === 'ln') {
      var one = s === 'ln1', ly = one ? Y.ln1 : Y.ln2, nm = one ? '1' : '2';
      o.tok = { x: XC + 68, y: ly, lab: [['γ', 'i'], ['′', 'r'], [', ', 'r'], ['β', 'i'], ['′', 'r']], w: 50, onBox: one ? 'ln1' : 'ln2' };
      o.edit = { tex: '(\\gamma_' + nm + ',\\beta_' + nm + ')\\mapsto(\\gamma_' + nm + '\',\\beta_' + nm + '\')', note: 'LN-tuning retrains the gain and shift of LN<sub>' + nm + '</sub>; every other weight stays frozen.' };
      var gK = one ? 'lnG1' : 'lnG2', bK = one ? 'lnB1' : 'lnB2';
      if (t === 'inplace') {
        S('in place', 'nothing to rewrite', '\\mathrm{LN}_' + nm + '\'(x)=\\gamma_' + nm + '\'\\odot n(x)+\\beta_' + nm + '\'', 'The trained parameters already belong to a box of the base network, so the merge is the identity.', [], [XC + 68, ly + 22]);
        o.kind = 'inplace'; o.tgt = [one ? 'ln1' : 'ln2'];
        o.merged = function (WM) { var W2 = cp(WM.W); if (one) { W2.g1 = WM.M.lnG1; W2.b1 = WM.M.lnB1; } else { W2.g2 = WM.M.lnG2; W2.b2 = WM.M.lnB2; } return W2; };
        o.ref = 'Thm VI.3, corollary: LN-tuning edits existing parameters, trivially mergeable';
      } else {
        var cons = one ? ['q', 'k', 'v'] : (sw ? ['gate', 'up'] : ['up']);
        var fold = function (WM) {
          var W2 = cp(WM.W), g = WM.M[gK], b = WM.M[bK];
          if (one) { W2.g1 = vz(D).fill(1); W2.b1 = vz(D); } else { W2.g2 = vz(D).fill(1); W2.b2 = vz(D); }
          cons.forEach(function (c) { W2['W' + c] = dR(WM.W['W' + c], g); W2['b' + c] = vadd(WM.W['b' + c], mv(WM.W['W' + c], b)); });
          return W2;
        };
        var r6end = one ? [[XC + 68, ly + 12], [XC + 68, 84], [XC, 84], [XC, Y.sp1]] : (sw ? [[XC + 68, ly + 12], [XC + 68, 441], [XC, 441], [XC, Y.sp3]] : [[XC + 68, ly + 12], [XC + 68, 446], [XC, 446], [XC, 462]]);
        S('R6', 'norm absorption', 'W(\\gamma\'\\odot n+\\beta\')=(W\\,\\mathrm{diag}\\,\\gamma\')\\,n+W\\beta\'', 'LN<sub>' + nm + '</sub> is left with γ = 1, β = 0; the affine part now rides on the wire.', [r6end], one ? [XC + 116, 72] : [XC + 116, ly]);
        var tops = one ? [[XQ, 108], [XK, 108], [XV, 108]] : (sw ? [[XG, 466], [XU, 466]] : [[XC, 462]]);
        var cps = one ? [[[XC, Y.sp1], [XQ, Y.sp1], [XQ, 108]], [[XC, Y.sp1], [XK, 108]], [[XC, Y.sp1], [XV, Y.sp1], [XV, 108]]] : (sw ? [[[XC, Y.sp3], [XG, Y.sp3], [XG, 466]], [[XC, Y.sp3], [XU, Y.sp3], [XU, 466]]] : null);
        if (cps) S('R7', 'copy slide', one ? '\\Delta_3\\circ E=(E\\otimes E\\otimes E)\\circ\\Delta_3' : '\\Delta_2\\circ E=(E\\otimes E)\\circ\\Delta_2', 'Pre-LN: LN<sub>' + nm + '</sub> feeds only ' + (one ? Wh('Q', 1) + ', ' + Wh('K', 1) + ', ' + Wh('V', 1) : Wh('gate') + ' and ' + Wh('up')) + ' (the residual bypasses it), so every copy reaches a trainable box.', cps, one ? [XV + 44, Y.sp1] : [XU + 40, Y.sp3]);
        var ids = cons.map(function (c) { return 'w' + c; });
        var r1paths = tops.map(function (p) { return [p, [p[0], one ? Y.proj : Y.up]]; });
        o.tgt = ids;
        if (bias) {
          S('R1', 'fusion', 'W\'=W\\,\\mathrm{diag}\\,\\gamma\',\\qquad b\'=b+W\\beta\'', 'For each consumer; the shift lands in its bias slot.', r1paths, one ? [XV + 46, 126] : (sw ? [XU + 46, 470] : [XC + 46, 466]));
          o.merged = fold;
          o.ref = 'Thm VI.3, R6 + R7 + R1: pre-LN block with bias slots';
        } else {
          X('R1', 'fusion', 'W\\beta\'\\neq0', 'The shift Wβ′ needs a bias slot (R6) and the consumers are bias-free. Folding β′ forward needs a bias slot in every consumer (a gain alone, as in RMSNorm, folds without one); in place it is always fine.', tops.map(function (p) { return [p, [p[0], p[1] + 2]]; }), one ? [XV + 53, 126] : (sw ? [XU + 46, 470] : [XC + 46, 466]), ids[0]);
          o.blockAll = ids;
          o.forced = fold; o.fpaths = r1paths;
          o.fdesc = 'fold γ′ into the consumers and drop Wβ′ (no bias slot)';
          o.ref = 'Thm VI.3, R6: the shift Wβ needs a bias slot';
        }
      }
    } else if (m === 'lora' || m === 'oft') {
      var box = s === 'q' ? 'wq' : s === 'v' ? 'wv' : 'wup', g0 = geo(sw, rp)[box], bn = BOXNAME[box];
      var BT = { wq: 'W_Q', wv: 'W_V', wup: 'W_{\\rm up}' }[box], BH = { wq: Wh('Q', 1), wv: Wh('V', 1), wup: Wh('up') }[box];
      o.tgt = [box];
      if (m === 'lora') {
        var bx = box === 'wq' ? g0.x - g0.w / 2 - 29 : g0.x + g0.w / 2 + 29;
        o.tok = { x: bx, y: g0.y, lab: [['+ ', 'r'], ['sBA', 'i']], w: 50 };
        o.edit = { tex: BT + 'x\\mapsto ' + BT + 'x+sB(Ax),\\qquad r=2', note: 'LoRA adds a parallel rank-2 branch to ' + BH + '.' };
        S('R1', 'fusion', BT + 'x+sB(Ax)=(' + BT + '+sBA)\\,x', 'Linear boxes add (merge_and_unload).', [[[bx, g0.y], [g0.x, g0.y]]], [bx, g0.y - 21]);
        o.merged = function (WM) { var W2 = cp(WM.W); W2['W' + bn] = madd(WM.W['W' + bn], mmul(WM.M.loraB[s], WM.M.loraA[s])); return W2; };
        o.ref = 'Thm VI.3, R1; LoRA (Hu et al.) merges by linearity';
      } else {
        var oy = box === 'wq' ? Y.oftq : (sw ? 466 : 455);
        o.tok = { x: g0.x, y: oy, lab: [['R', 'i'], ['⊤', 'p']], w: 32 };
        o.edit = { tex: BT + 'x\\mapsto ' + BT + '(R^{\\top}x),\\quad R=\\mathrm{blockdiag}(R_1,R_2)', note: 'OFT rotates the input of ' + BH + ' (block size 4, exact Cayley map).' };
        S('R1', 'fusion', BT + '(R^{\\top}x)=(' + BT + 'R^{\\top})\\,x', 'The Cayley map is a parameter-side nonlinearity, irrelevant to merging; the fold is exact even for HF’s approximately orthogonal Cayley–Neumann R. Here \\(\\lVert R^{\\top}R-I\\rVert_{\\max}\\) = {orth}.', [[[g0.x, oy], [g0.x, g0.y]]], [g0.x - 34, oy]);
        o.merged = function (WM) { var W2 = cp(WM.W); W2['W' + bn] = mmul(WM.W['W' + bn], WM.M.oftRt); return W2; };
        o.ref = 'Thm VI.3, corollary: reparametrisation methods merge (their nonlinearity is parameter-side)';
      }
    } else if (m === 'adapter') {
      var atO = s === 'o', node = atO ? 'adO' : 'adD', tb = atO ? 'wo' : 'wdown', ty = atO ? Y.wo : Y.down, ay = atO ? Y.so : Y.sd;
      o.tok = { node: node };
      o.edit = { tex: 'y\\mapsto y+U\\,\\sigma(Dy+b_D)+b_U,\\qquad r=2', note: 'A Houlsby adapter after ' + (atO ? Wh('O', 1) : Wh('down')) + ', with σ = ' + aN + '.' };
      o.tgt = [tb];
      X('R1', 'fusion', 'y\\mapsto y+U\\,\\sigma(Dy+b_D)+b_U\\ \\text{ is not affine}', 'Thm VI.3(c): a non-affine edit of a linear box is not box-locally mergeable; no rule moves σ out of the adapter.', [], [XC + 88, ay], node);
      o.forced = function (WM, cfg, A1) { return atO ? boxRefit(WM, cfg, A1.oIn, A1.oOut, 'o') : boxRefit(WM, cfg, A1.dIn, A1.dOut, 'down'); };
      o.fpaths = [[[XC, ay - 10], [XC, ty]]];
      o.fdesc = 'best ' + (bias ? 'affine' : 'linear') + ' refit of ' + (atO ? Wh('O', 1) : Wh('down')) + ' by least squares over the 36 probe tokens';
      o.ref = 'Thm VI.3(c): nonlinear Houlsby adapters are not box-locally mergeable';
    } else { /* prefix */
      o.tok = { node: 'prefix' };
      o.edit = { tex: '\\mathrm{Attn}_q(C)\\mapsto \\mathrm{Attn}_q(P\\uplus C),\\qquad |P|=2', note: 'Prefix tuning adds two trained key–value pairs to every query’s softmax.' };
      o.tgt = ['wv'];
      S('VI.5b', 'prefix = gated adapter', '\\mathrm{Attn}_q(P\\uplus C)=(1-\\lambda_q)\\,\\mathrm{Attn}_q(C)+\\lambda_q\\,\\mathrm{Attn}_q(P)', 'An identity (Thm VI.5(b), after He et al. 2022). It is not yet a merge.', [], [300, 202]);
      X('VI.5f', 'the gate depends on the input', '\\lambda_q=Z_q(P)\\,/\\,\\big(Z_q(P)+Z_q(C)\\big)', 'On one-token contexts (each probe token alone) λ<sub>q</sub> ranges from {lamlo} to {lamhi}, and the best affine fit of the prefixed head misses by {aff1}. Every weight setting of a plain head is affine there, while a generic prefixed head is not (Thm VI.5(f)); so prefix tuning is not box-locally mergeable (Thm VI.3(c)).', [], [XS + 76, Y.smax], 'smax');
      o.forced = function (WM, cfg, A1) { return prefixRefit(WM, cfg, A1); };
      o.fpaths = [[[XPF, 210], [XPF, Y.proj], [XV + 27, Y.proj]]];
      o.fdesc = 'best refit of ' + Wh('V', 1) + (bias ? ', b<sub><i>V</i></sub>' : '') + ' by least squares over the 36 probe queries';
      o.ref = 'Thm VI.5(b, f); Thm VI.3(c): prefix tuning is not box-locally mergeable';
    }
    return o;
  }

  /* run the plan numerically: adapted network vs merged (licensed) or forced weights */
  function evaluate(st, WM) {
    var cfg = { act: st.act, rope: st.rope, bias: st.bias };
    var pl = plan(st, WM.M), ad = adapted(st, WM);
    var A1 = forward(ad.W, WM.X, cfg, ad.hk);
    var r = { plan: pl, cfg: cfg, delta: null, forced: null, ex: {} };
    if (pl.ok) r.delta = relErr(A1.Y, forward(pl.merged(WM, cfg, A1), WM.X, cfg, null).Y);
    else r.forced = relErr(A1.Y, forward(pl.forced(WM, cfg, A1), WM.X, cfg, null).Y);
    if (pl.ex && pl.ex.l) {
      var l = pl.ex.l, neg = 0, off = 0;
      for (var i = 0; i < l.length; i++) { if (l[i] < 0) neg++; if (Math.abs(l[i]) > 1e-12 && Math.abs(l[i] - 1) > 1e-12) off++; }
      r.ex.neg = neg; r.ex.off01 = off; r.ex.n = l.length;
    }
    if (A1.lam.length) {
      /* Thm VI.5(f) is about one-token contexts: rerun every probe token alone and read its gate */
      var X1 = []; WM.X.forEach(function (q) { q.forEach(function (x) { X1.push([x]); }); });
      var L1 = forward(ad.W, X1, cfg, ad.hk).lam;
      r.ex.lamlo = Math.min.apply(null, L1); r.ex.lamhi = Math.max.apply(null, L1); r.ex.lamn = L1.length;
      /* non-affinity witness: best affine fit of the one-token head (input x̂ = LN₁ output, output = mix) */
      var F1 = forward(ad.W, X1, cfg, ad.hk), xin = X1.map(function (q) { var z = lnorm(q[0], ad.W.g1, ad.W.b1), r1 = vz(D + 1); r1.set(z); r1[D] = 1; return r1; });
      var Bf = lstsq(xin, F1.oIn), num = 0, den = 0;
      xin.forEach(function (z, n) { var pr = mv(Bf, z); for (var d = 0; d < D; d++) { var e = pr[d] - F1.oIn[n][d]; num += e * e; den += F1.oIn[n][d] * F1.oIn[n][d]; } });
      r.ex.aff1 = Math.sqrt(num / den);
    }
    if (st.method === 'ia3' && st.site === 'k' && st.target === 'wq' && st.rope && !st.tie) r.ex.ls = ropeLS(WM, cfg, pl.ex.l);
    if (st.method === 'oft') r.ex.orth = WM.M.oftOrth;
    return r;
  }

  /* ---------------- formatting ---------------- */
  function fe(x) {
    if (!isFinite(x)) return '—';
    if (x === 0) return '0';
    var e = Math.floor(Math.log10(Math.abs(x))), mt = x / Math.pow(10, e), s = mt.toFixed(1);
    if (s === '10.0') { s = '1.0'; e += 1; }
    return s + 'e' + (e < 0 ? '−' : '+') + Math.abs(e);
  }
  function fd(x) { if (x === null || x === undefined) return '—'; return x < 1e-9 ? fe(x) : (x < 0.01 ? x.toFixed(4) : x.toPrecision(3)); }
  function pct(x) { return (100 * x).toFixed(x < 0.1 ? 2 : 1) + '%'; }

  /* ---------------- the figure ---------------- */
  Atlas.register('merge', function (el, A) {
    if (!document.getElementById(CSS_ID)) { var stl = document.createElement('style'); stl.id = CSS_ID; stl.textContent = CSS; document.head.appendChild(stl); }
    var H = A.h, d3 = window.d3;
    var UID = 'mg' + Math.random().toString(36).slice(2, 7);
    var DEF = { method: 'ia3', site: 'ffn', target: 'wup', act: 'swiglu', rope: true, bias: false, sign: 'any', tie: false, seed: 7, view: 'rewrite' };
    var st = cp(DEF);
    var WM = gen(st.seed, A), cur = null, mat = null, animToken = 0;

    el.setAttribute('aria-label', 'The Merge Game: a pre-LN transformer block in which a PEFT edit is rewritten by the rules of Thm VI.3 until a weight absorbs it or a box blocks it.');
    var stage = H('div', { class: 'stage' });
    el.appendChild(stage);

    /* header */
    stage.appendChild(H('div', { class: 'mg-head' }, [
      H('h3', { class: 'mg-title', html: 'The Merge <em>Game</em>' }),
      H('span', { class: 'mg-kicker', text: 'Fig. 12 · Thm VI.3 · Lemma VI.2' }),
    ]));
    stage.appendChild(H('p', { class: 'mg-instr', html: 'Pick a method, where it attaches, and the frozen weight that should absorb it; then press <b>Rewrite</b>. The edit slides through the block by the rules R1–R7 until a weight absorbs it exactly (moss) or a box stops it (seal). Change σ, RoPE or bias slots and watch the verdict move.' }));

    /* controls */
    var ctr = H('div', { class: 'mg-controls' });
    stage.appendChild(ctr);
    function segH(opts, onPick, aria) {
      var wrap = H('div', { class: 'seg', role: 'group', 'aria-label': aria });
      var btns = [], sig = null;
      function build(list) {
        var key = list.map(function (o) { return String(o.v) + '|' + o.h; }).join('#');
        if (key === sig) return;   /* unchanged: keep the buttons (and keyboard focus) */
        sig = key;
        wrap.innerHTML = ''; btns = [];
        list.forEach(function (o) {
          var b = H('button', { type: 'button', 'aria-pressed': 'false', html: o.h });
          if (o.title) b.title = o.title;
          b.addEventListener('click', function () { if (!b.disabled) onPick(o.v); });
          b._v = o.v; wrap.appendChild(b); btns.push(b);
        });
      }
      build(opts);
      return {
        el: wrap, build: build,
        set: function (v) { btns.forEach(function (b) { b.setAttribute('aria-pressed', String(b._v === v)); }); },
        off: function (yes) { btns.forEach(function (b) { b.disabled = !!yes; }); },
      };
    }
    function grp(label, seg) { var g = H('div', { class: 'mg-grp' }, [H('span', { class: 'label', html: label }), seg.el]); return g; }
    var sgMethod = segH(METHODS.map(function (m) { return { v: m.id, h: m.label }; }), function (v) {
      if (v === st.method) return;
      st.method = v; st.site = methodDef(v).sites[0].v; st.target = defTarget(); update();
    }, 'PEFT method');
    var sgSite = segH([], function (v) { if (v === st.site) return; st.site = v; st.target = defTarget(); update(); }, 'Where the method attaches');
    var sgTarget = segH([], function (v) { st.target = v; update(); }, 'Weight that should absorb the edit');
    var sgAct = segH([{ v: 'gelu', h: 'GELU' }, { v: 'relu', h: 'ReLU' }, { v: 'swiglu', h: 'SwiGLU' }], function (v) { st.act = v; fixTarget(); update(); }, 'FFN activation');
    var sgRope = segH([{ v: false, h: 'off' }, { v: true, h: 'on' }], function (v) { st.rope = v; update(); }, 'RoPE on queries and keys');
    var sgBias = segH([{ v: true, h: 'yes' }, { v: false, h: 'bias-free' }], function (v) { st.bias = v; update(); }, 'Bias slots on linear boxes');
    var sgSign = segH([{ v: 'any', h: '<i>ℓ</i> ∈ ℝ' }, { v: 'pos', h: '<i>ℓ</i> ≥ 0' }], function (v) { st.sign = v; update(); }, 'Parameter domain of ell: sign');
    var sgTie = segH([{ v: false, h: 'free' }, { v: true, h: 'pair-tied' }], function (v) { st.tie = v; update(); }, 'Parameter domain of ell on rotary pairs');
    var gSign = grp('(IA)³ domain', sgSign), gTie = grp('<i>ℓ</i> on rotary pairs', sgTie);
    var bRewrite = H('button', { type: 'button', class: 'btn primary', text: 'Rewrite ▸' });
    var bForce = H('button', { type: 'button', class: 'btn force', text: 'Force merge', 'aria-pressed': 'false' });
    var bSeed = H('button', { type: 'button', class: 'btn', text: 'New weights' });
    var bReset = H('button', { type: 'button', class: 'btn', text: 'Reset' });
    ctr.appendChild(H('div', { class: 'mg-row' }, [grp('Method', sgMethod), grp('Attached at', sgSite)]));
    ctr.appendChild(H('div', { class: 'mg-row' }, [grp('Try to merge into', sgTarget), grp('FFN activation <i>σ</i>', sgAct), grp('RoPE', sgRope), grp('Bias slots', sgBias)]));
    ctr.appendChild(H('div', { class: 'mg-row' }, [gSign, gTie, H('div', { class: 'mg-btns' }, [bRewrite, bForce, bSeed, bReset])]));
    bRewrite.addEventListener('click', function () { st.view = 'rewrite'; play(false); });
    bForce.addEventListener('click', function () { if (cur && !cur.plan.ok) { st.view = 'forced'; play(true); } });
    bSeed.addEventListener('click', function () { st.seed += 1; WM = gen(st.seed, A); update(); });
    bReset.addEventListener('click', function () { st = cp(DEF); WM = gen(st.seed, A); update(); });

    /* legend */
    var legend = H('div', { class: 'mg-legend', 'aria-hidden': 'true' });
    stage.appendChild(legend);

    /* main grid */
    var main = H('div', { class: 'mg-main' });
    stage.appendChild(main);
    var pDiag = H('div', { class: 'mg-panel mg-diag' });
    var diagHead = H('div', { class: 'mg-ph' });
    var svgBox = H('div', { class: 'mg-svg' });
    var hyp = H('div', { class: 'mg-hyp' });
    pDiag.appendChild(diagHead); pDiag.appendChild(svgBox); pDiag.appendChild(hyp);
    var side = H('div', { class: 'mg-side' });
    var pLog = H('div', { class: 'mg-panel mg-log' }), pNum = H('div', { class: 'mg-panel mg-numc' }), pMat = H('div', { class: 'mg-panel mg-mat' });
    side.appendChild(pLog); side.appendChild(pNum); side.appendChild(pMat);
    main.appendChild(pDiag); main.appendChild(side);
    var live = H('div', { class: 'mg-sr', 'aria-live': 'polite' });
    stage.appendChild(live);
    var foot = H('div', { class: 'mg-foot' });
    stage.appendChild(foot);

    function layout() { stage.classList.toggle('mg-two', stage.clientWidth >= 860); }
    layout();
    if ('ResizeObserver' in window) new ResizeObserver(layout).observe(stage); else window.addEventListener('resize', layout);
    var svg = d3.select(svgBox).append('svg').attr('viewBox', VBX + ' 0 ' + VBW + ' ' + VBH).attr('role', 'img');
    var KS = 1, lastW = 0;
    if ('ResizeObserver' in window) new ResizeObserver(A.debounce(function () { var w = svgBox.clientWidth; if (Math.abs(w - lastW) > 8 && cur) { lastW = w; settle(); } }, 120)).observe(svgBox);

    function defTarget() {
      var k = st.method + '/' + st.site, L = targetsFor(st.method, st.site, st.act);
      return DEFAULT_TARGET[k] && L.indexOf(DEFAULT_TARGET[k]) >= 0 ? DEFAULT_TARGET[k] : L[0];
    }
    function fixTarget() {
      var md = methodDef(st.method), okSite = md.sites.some(function (x) { return x.v === st.site; });
      if (!okSite) st.site = md.sites[0].v;
      var L = targetsFor(st.method, st.site, st.act);
      if (L.indexOf(st.target) < 0) st.target = st.target === 'wgate' && L.indexOf('wup') >= 0 ? 'wup' : defTarget();
    }

    function compute() {
      cur = evaluate(st, WM);
      var cols = st.method === 'ia3' && st.site === 'ffn' ? ['wdown', 'wup', 'wgate'] : targetsFor(st.method, st.site, 'swiglu');
      if (st.method === 'ln' && st.site === 'ln2') cols = ['inplace', 'fold2'];
      mat = { cols: cols, rows: ['gelu', 'relu', 'swiglu'].map(function (a) {
        return cols.map(function (t) {
          if (targetsFor(st.method, st.site, a).indexOf(t) < 0) return null;
          var s2 = cp(st); s2.act = a; s2.target = t;
          return evaluate(s2, WM);
        });
      }) };
    }

    /* ---------------- rendering: controls ---------------- */
    function renderControls() {
      var md = methodDef(st.method);
      sgMethod.set(st.method);
      sgSite.build(md.sites); sgSite.set(st.site);
      sgTarget.build(targetsFor(st.method, st.site, st.act).map(function (t) { return { v: t, h: TLAB[t] }; })); sgTarget.set(st.target);
      sgAct.set(st.act); sgRope.set(st.rope); sgBias.set(st.bias); sgSign.set(st.sign); sgTie.set(st.tie);
      var signOn = st.method === 'ia3', tieOn = st.method === 'ia3' && st.site === 'k';
      sgSign.off(!signOn); gSign.classList.toggle('is-off', !signOn);
      sgTie.off(!tieOn); gTie.classList.toggle('is-off', !tieOn);
      var canForce = cur && !cur.plan.ok;
      bForce.disabled = !canForce;
      bForce.title = canForce ? 'Apply the refused rule anyway and measure the damage' : 'Nothing to force: the rewrite is licensed';
      bForce.setAttribute('aria-pressed', String(st.view === 'forced' && canForce));
    }

    /* ---------------- rendering: legend, hypotheses, header ---------------- */
    function sw(col, kind) {
      if (kind === 'badge') return '<svg width="22" height="12" viewBox="0 0 22 12"><rect x="1" y="1" width="20" height="10" rx="3" style="fill:var(--paper);stroke:var(--ochre);stroke-width:1.3"/></svg>';
      if (kind === 'path') return '<svg width="22" height="12" viewBox="0 0 22 12"><path d="M1 6H21" style="stroke:var(--ochre);stroke-width:2.2;stroke-linecap:round"/></svg>';
      if (kind === 'aim') return '<svg width="22" height="12" viewBox="0 0 22 12"><rect x="1" y="1" width="20" height="10" rx="2" style="fill:none;stroke:var(--ink-3);stroke-width:1.2;stroke-dasharray:4 3"/></svg>';
      if (kind === 'force') return '<svg width="22" height="12" viewBox="0 0 22 12"><path d="M1 6H21" style="stroke:var(--seal);stroke-width:2.2;stroke-linecap:round;stroke-dasharray:5 4"/></svg>';
      if (kind === 'pill') return '<svg width="22" height="12" viewBox="0 0 22 12"><rect x="1" y="1" width="20" height="10" rx="5" style="fill:var(--paper);stroke:var(--ink-3);stroke-width:1.1"/></svg>';
      return '<svg width="22" height="12" viewBox="0 0 22 12"><rect x="1" y="1" width="20" height="10" rx="2" style="fill:var(--' + col + '-soft);stroke:var(--' + col + ');stroke-width:1.8"/></svg>';
    }
    function renderLegend() {
      legend.innerHTML = '<span>' + sw(0, 'badge') + 'the edit</span><span>' + sw(0, 'path') + 'its rewrite path</span>' +
        '<span>' + sw('moss') + 'absorbing weight (exact merge)</span><span>' + sw('seal') + 'blocking box</span>' +
        '<span>' + sw(0, 'aim') + 'weight it was aimed at</span><span>' + sw(0, 'force') + 'forced merge</span>' +
        '<span>' + sw(0, 'pill') + 'nonlinear box</span><span><svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="3" style="fill:var(--ink-2)"/></svg>copy spider</span>';
    }
    function renderHyp() {
      diagHead.innerHTML = '<b>Pre-LN block</b><span class="num"><i>d</i> = 8 · <i>d</i><sub>ff</sub> = 16 · 1 head · causal</span>';
      hyp.innerHTML = '<span class="label">Thm VI.3 hypotheses</span>' +
        '<span class="chip">pre-LN</span><span class="chip">no weight tying, no GQA</span><span class="chip">edit acts at every position</span>' +
        '<span class="chip ' + (st.bias ? 'yes' : 'cond') + '">bias slots ' + (st.bias ? 'on' : 'off') + '</span>' +
        '<span class="chip ' + (st.rope ? 'cond' : 'yes') + '">RoPE ' + (st.rope ? 'on' : 'off') + '</span>' +
        '<p>LN<sub>1</sub> feeds only ' + Wh('Q', 1) + ', ' + Wh('K', 1) + ', ' + Wh('V', 1) + ' and LN<sub>2</sub> only the FFN; the residual stream bypasses both. Box-local merges only (Def VI.1).</p>';
    }

    /* ---------------- rendering: rewrite log ---------------- */
    function fill(note) {
      var ex = cur.ex;
      return note.replace('{neg}', String(ex.neg)).replace('{off01}', String(ex.off01))
        .replace('{lamlo}', ex.lamlo !== undefined ? ex.lamlo.toFixed(3) : '—').replace('{lamhi}', ex.lamhi !== undefined ? ex.lamhi.toFixed(3) : '—')
        .replace('{orth}', ex.orth !== undefined ? fe(ex.orth) : '—').replace('{ls}', ex.ls ? pct(ex.ls.res) : '—')
        .replace('{aff1}', ex.aff1 !== undefined ? pct(ex.aff1) : '—')
        .replace('{negs}', ex.neg === 1 ? '1 negative entry' : ex.neg + ' negative entries')
        .replace('{off01s}', ex.off01 === ex.n ? 'all ' + ex.n + ' entries of ℓ lie' : ex.off01 + ' of the ' + ex.n + ' entries of ℓ ' + (ex.off01 === 1 ? 'lies' : 'lie'));
    }
    function subjT(id) { return { fold1: 'They', fold2: 'They', head: 'The head' }[id] || tlab(id); }
    function tlab(id) { return id === 'inplace' ? 'LN<sub>' + (st.site === 'ln1' ? '1' : '2') + '</sub>' : TLAB[id] || id; }
    function nodeName(id) {
      return { act: st.act === 'swiglu' ? 'SiLU' : (st.act === 'gelu' ? 'GELU' : 'ReLU'), ropek: 'the rotation <i>R<sub>n</sub></i>', smax: 'the softmax gate', adO: 'the adapter’s σ', adD: 'the adapter’s σ',
        wo: Wh('O', 1) + ' (no bias slot)', wq: 'the bias-free consumers', wup: 'the bias-free consumers', wgate: 'the bias-free consumers' }[id] || id;
    }
    function altTargets() {
      return targetsFor(st.method, st.site, st.act).filter(function (t) { if (t === st.target) return false; var s2 = cp(st); s2.target = t; return plan(s2, WM.M).ok; });
    }
    function renderLog(pending) {
      var pl = cur.plan, forced = st.view === 'forced' && !pl.ok;
      var h = '<div class="mg-ph"><b>Rewrite</b><span class="thm">' + methodDef(st.method).label + ' at ' + siteName() + ' → ' + tlab(st.target) + '</span></div><ol>';
      h += '<li class="mg-step edit"><span class="mg-rule">edit</span><div><div class="mg-smath">\\(' + pl.edit.tex + '\\)</div><div class="mg-snote">' + pl.edit.note + '</div></div></li>';
      pl.steps.forEach(function (s, i) {
        h += '<li class="mg-step' + (s.bad ? ' bad' : '') + (pending ? ' pending' : '') + '" data-i="' + i + '"><span class="mg-rule">' + (s.bad ? '✗ ' : '') + s.rule + '</span><div>' +
          '<div class="mg-sname">' + s.name + '</div><div class="mg-smath">\\(' + s.tex + '\\)</div><div class="mg-snote">' + fill(s.note) + '</div></div></li>';
      });
      if (forced) h += '<li class="mg-step bad forced' + (pending ? ' pending' : '') + '" data-i="' + pl.steps.length + '"><span class="mg-rule">force</span><div><div class="mg-sname">merge anyway</div><div class="mg-snote">' + pl.fdesc + '. The merged block no longer computes the adapted function: <span class="mono">‖Δ‖/‖y‖ = ' + fd(cur.forced) + '</span>.</div></div></li>';
      h += '</ol>';
      var alt = altTargets();
      if (pl.ok) {
        h += '<div class="mg-verdict ok' + (pending ? ' pending' : '') + '"><span class="vt">' + (pl.kind === 'inplace' ? 'Mergeable in place' : 'Fuses into ' + pl.tgt.map(function (t) { return TLAB[t]; }).join(', ')) + '.</span> <span class="vb">No added inference latency; ' + (cur.delta === 0 ? 'the merged weights are the trained ones, so the check gives exactly 0.' : 'the numerical check agrees to ' + fd(cur.delta) + '.') + '</span><span class="vr">' + pl.ref + '</span></div>';
      } else {
        h += '<div class="mg-verdict bad' + (pending ? ' pending' : '') + '"><span class="vt">Blocked at ' + nodeName(pl.block) + '.</span> <span class="vb">' + subjT(st.target) + ' cannot absorb this edit' +
          (alt.length ? '; ' + alt.map(function (t) { return t === 'inplace' ? tlab(t) + ' itself, in place,' : tlab(t); }).join(', ') + ' can.' : (st.method === 'adapter' || st.method === 'prefix' ? '; the edit is not box-locally mergeable.' : '.')) + '</span><span class="vr">' + pl.ref + '</span></div>';
      }
      pLog.innerHTML = h;
      A.typeset(pLog);
    }
    function siteName() { var md = methodDef(st.method); for (var i = 0; i < md.sites.length; i++) if (md.sites[i].v === st.site) return md.sites[i].h; return st.site; }

    /* ---------------- rendering: numbers ---------------- */
    function renderNums() {
      var pl = cur.plan, ex = cur.ex;
      var h = '<div class="mg-ph"><b>Numerical check</b><span class="num">seed ' + st.seed + ' · 6 × 6 tokens · float64</span></div><div class="mg-cells">';
      if (pl.ok) {
        h += '<div class="mg-cell"><div class="k">adapted vs merged weights</div><div class="v big ok">' + fd(cur.delta) + '</div><div class="s">\\(\\lVert y_{\\rm adapted}-y_{\\rm merged}\\rVert_F\\) \\(/\\,\\lVert y_{\\rm adapted}\\rVert_F\\)' + (cur.delta === 0 ? ' (identical weights)' : ', machine precision') + '</div></div>';
        h += '<div class="mg-cell"><div class="k">force merge</div><div class="v mut">not needed</div><div class="s">every side condition holds</div></div>';
      } else {
        h += '<div class="mg-cell"><div class="k">licensed merge</div><div class="v mut">none</div><div class="s">the rewrite stops at ' + nodeName(pl.block) + '</div></div>';
        h += '<div class="mg-cell"><div class="k">forced merge</div><div class="v big bad">' + fd(cur.forced) + '</div><div class="s">' + pl.fdesc + '</div></div>';
      }
      var x = extraCell();
      if (x) h += x;
      h += '</div>';
      pNum.innerHTML = h;
      A.typeset(pNum);
    }
    function extraCell() {
      var ex = cur.ex, pl = cur.plan;
      if (st.method === 'ia3' && st.site === 'ffn' && st.target !== 'wdown') {
        if (st.act === 'relu') return '<div class="mg-cell"><div class="k">units that block R3</div><div class="v' + (ex.neg ? ' bad' : ' ok') + '">' + ex.neg + ' / ' + ex.n + '</div><div class="s">ℓ<sub>i</sub> &lt; 0; ReLU commutes with ℓ<sub>i</sub> ≥ 0 only</div></div>';
        if (st.target === 'wgate' || st.act === 'gelu') return '<div class="mg-cell"><div class="k">units that block R3</div><div class="v">' + ex.off01 + ' / ' + ex.n + '</div><div class="s">units with ℓ<sub>i</sub> ∉ {0, 1}; only entries 0 and 1 cross ' + (st.target === 'wgate' ? 'SiLU' : 'GELU') + '</div></div>';
        return '<div class="mg-cell"><div class="k">units that block R4</div><div class="v">0 / ' + ex.n + '</div><div class="s">R4 holds for every ℓ, ' + ex.neg + ' negative entr' + (ex.neg === 1 ? 'y' : 'ies') + ' included</div></div>';
      }
      if (ex.ls) return '<div class="mg-cell"><div class="k">best <span class="nt">' + Wh('Q', 1) + '</span>, least squares</div><div class="v bad">' + pct(ex.ls.res) + '</div><div class="s">relative residual on ' + ex.ls.neq + ' causal logits, ' + ex.ls.nunk + ' unknowns</div></div>';
      if (ex.lamlo !== undefined) return '<div class="mg-cell"><div class="k">gate <span class="nt"><i>λ<sub>q</sub></i></span>, one-token contexts</div><div class="v">' + ex.lamlo.toFixed(3) + ' – ' + ex.lamhi.toFixed(3) + '</div><div class="s">over ' + ex.lamn + ' tokens; best affine fit of the head misses by ' + pct(ex.aff1) + ' (a plain head: 0)</div></div>';
      if (ex.orth !== undefined) return '<div class="mg-cell"><div class="k"><span class="nt">\\(\\lVert R^{\\top}R-I\\rVert_{\\max}\\)</span></div><div class="v">' + fe(ex.orth) + '</div><div class="s">exact Cayley, block size 4</div></div>';
      return '';
    }

    /* ---------------- rendering: matrix ---------------- */
    function renderMatrix() {
      var h = '<div class="mg-ph"><b>Which weight absorbs it?</b><span class="thm">all three σ, current settings</span></div><table><thead><tr><th class="rh" scope="col"><span class="mg-sr">activation</span></th>';
      mat.cols.forEach(function (c) { h += '<th scope="col">' + tlab(c) + '</th>'; });
      h += '</tr></thead><tbody>';
      ['gelu', 'relu', 'swiglu'].forEach(function (a, ri) {
        var an = a === 'gelu' ? 'GELU' : a === 'relu' ? 'ReLU' : 'SwiGLU';
        h += '<tr><th class="rh" scope="row">' + an + '</th>';
        mat.rows[ri].forEach(function (r, ci) {
          var c = mat.cols[ci], isCur = a === st.act && c === st.target;
          if (!r) { h += '<td><span class="cell na" aria-label="' + an + ': no gate branch"><span class="ic">–</span><span class="nt">no gate</span></span></td>'; return; }
          var ok = r.plan.ok, val = ok ? fd(r.delta) : fd(r.forced);
          var note = '';
          if (!ok && a === 'relu' && st.method === 'ia3' && st.site === 'ffn' && c === 'wup') note = '✓ if ℓ ≥ 0';
          h += '<td><button type="button" class="cell ' + (ok ? 'ok' : 'bad') + (isCur ? ' cur' : '') + '" data-a="' + a + '" data-t="' + c + '" aria-label="' + an + ', ' + c + ': ' + (ok ? 'merges, error ' : 'blocked, forced error ') + val + '">' +
            '<span class="ic">' + (ok ? '✓' : '✗') + '</span><span class="fv">' + (ok ? '' : '<span class="fw">forced</span> ') + '<span>' + val + '</span></span>' + (note ? '<span class="nt">' + note + '</span>' : '') + '</button></td>';
        });
        h += '</tr>';
      });
      h += '</tbody></table><p class="cap">Each cell runs the rule engine and the numerical check: ✓ shows <span class="mono">‖Δ‖/‖y‖</span> of the licensed merge, ✗ that of the forced one. Click a cell to inspect it.</p>';
      pMat.innerHTML = h;
      pMat.querySelectorAll('button.cell').forEach(function (b) {
        b.addEventListener('click', function () { st.act = b.getAttribute('data-a'); st.target = b.getAttribute('data-t'); update(); });
      });
    }

    /* ---------------- rendering: footer ---------------- */
    function renderFoot() {
      var md = methodDef(st.method), m = A.method(md.data);
      var meth = m ? '<span class="mg-meth"><span class="label">In the atlas</span><b>' + A.esc(m.name) + '</b>' + (m.url ? '<a href="' + A.esc(m.url) + '" target="_blank" rel="noopener">' + A.esc(m.authors || '') + ' ' + A.esc(String(m.year || '')) + '</a>' : '') +
        ' <span class="chip ' + (m.mergeable === 'yes' ? 'yes' : m.mergeable === 'no' ? 'no' : 'cond') + '">mergeable: ' + A.esc(m.mergeable || '?') + '</span></span>' : '';
      foot.innerHTML = '<div>Rules R1–R7 and their side conditions follow the repaired Thm VI.3; the σ-dependence comes from Lemma VI.2, the prefix obstruction from Thm VI.5(f). The rules certify merges and name box-local obstructions. A stop means the chosen weight cannot absorb the edit; it does not rule out a rewrite of the whole network. Seeded synthetic weights; GELU is the tanh approximation, SiLU is exact, RoPE uses base 10 000 on four rotary pairs. Every number is computed in your browser.</div>' + meth;
    }

    /* ---------------- SVG ---------------- */
    function colors() {
      return { ink: A.css('--ink'), ink2: A.css('--ink-2'), ink3: A.css('--ink-3'), rule: A.css('--rule'), paper: A.css('--paper'), paper2: A.css('--paper-2'), paper3: A.css('--paper-3'),
        ochre: A.css('--ochre'), ochreS: A.css('--ochre-soft'), seal: A.css('--seal'), sealS: A.css('--seal-soft'), moss: A.css('--moss'), mossS: A.css('--moss-soft'), tide: A.css('--tide'),
        ochreI: A.css('--ochre-ink'), sealI: A.css('--seal-ink'), mossI: A.css('--moss-ink') };
    }
    function rich(sel, parts, size, fam, col) {
      var t = sel.append('text').attr('text-anchor', 'middle').attr('dominant-baseline', 'central').style('font-family', fam || 'var(--f-display)').style('font-size', size + 'px').style('fill', col);
      var shift = 0;
      parts.forEach(function (p) {
        var s = p[0], k = p[1], ts = t.append('tspan').text(s);
        var want = (k === 's' || k === 'si') ? size * 0.26 : (k === 'p' ? -size * 0.38 : 0);
        if (want !== shift) { ts.attr('dy', (want - shift).toFixed(2)); shift = want; }
        if (k === 'i' || k === 'si') ts.style('font-style', 'italic');
        /* scripts at 72% of the label, but never under 13.5 viewBox px (12px rendered at the narrowest diagram) */
        if (k === 's' || k === 'si' || k === 'p') ts.style('font-size', Math.max(size * 0.72, 13.5).toFixed(1) + 'px');
      });
      return t;
    }
    function draw(anim) {
      var C = colors(), pl = cur.plan, sw = st.act === 'swiglu', G = geo(sw, st.rope), forced = st.view === 'forced' && !pl.ok;
      var pw = svgBox.clientWidth || 400; KS = Math.max(1, Math.min(1.35, 400 / pw));
      svg.interrupt(); svg.selectAll('*').interrupt();
      svg.selectAll('*').remove();
      var defs = svg.append('defs');
      defs.append('marker').attr('id', UID + 'ar').attr('viewBox', '0 0 8 8').attr('refX', 7).attr('refY', 4).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
        .append('path').attr('d', 'M1,1 L7,4 L1,7').attr('fill', 'none').attr('stroke', C.ink3).attr('stroke-width', 1.2);
      defs.append('marker').attr('id', UID + 'ao').attr('viewBox', '0 0 8 8').attr('refX', 7).attr('refY', 4).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
        .append('path').attr('d', 'M1,1 L7,4 L1,7').attr('fill', 'none').attr('stroke', C.ochre).attr('stroke-width', 1.2);
      var gW = svg.append('g'), gT = svg.append('g'), gN = svg.append('g'), gK = svg.append('g'), gC = svg.append('g');

      /* side labels */
      [['ATTENTION', 228], ['FEED-FORWARD', 545]].forEach(function (s) {
        gW.append('text').attr('transform', 'translate(' + (XR + 16) + ',' + s[1] + ') rotate(-90)').attr('text-anchor', 'middle')
          .style('font-family', 'var(--f-ui)').style('font-weight', '500').style('font-size', '14px').style('letter-spacing', '.06em').style('fill', C.ink2).text(s[0]);
      });
      /* input / output */
      rich(gW.append('g').attr('transform', 'translate(' + XC + ',' + Y.xin + ')'), [['x', 'i']], 19, 'var(--f-display)', C.ink);
      rich(gW.append('g').attr('transform', 'translate(' + XC + ',704)'), [['y', 'i']], 19, 'var(--f-display)', C.ink);
      /* wires */
      wiresFor(sw, st.rope).forEach(function (w) {
        gW.append('path').attr('d', rpath(w.p, 7)).attr('fill', 'none').attr('stroke', C.ink3).attr('stroke-width', 1.25).attr('marker-end', w.a ? 'url(#' + UID + 'ar)' : null);
      });
      [[XC, Y.sp0], [XC, Y.sp1], [XC, Y.sp2]].concat(sw ? [[XC, Y.sp3]] : []).forEach(function (p) {
        gW.append('circle').attr('cx', p[0]).attr('cy', p[1]).attr('r', 3.4).attr('fill', C.ink2);
      });
      /* prefix wiring (part of the edit) */
      if (st.method === 'prefix') {
        gW.append('path').attr('d', rpath([[XPF - 20, Y.cup], [XS + 64, Y.cup]], 6)).attr('fill', 'none').attr('stroke', C.ochre).attr('stroke-width', 1.4).attr('marker-end', 'url(#' + UID + 'ao)');
        gW.append('path').attr('d', rpath([[XPF, 254], [XPF, Y.mix], [344, Y.mix]], 6)).attr('fill', 'none').attr('stroke', C.ochre).attr('stroke-width', 1.4).attr('marker-end', 'url(#' + UID + 'ao)');
        gW.append('text').attr('x', XPF - 30).attr('y', Y.cup - 6).attr('text-anchor', 'middle').style('font-family', 'var(--f-body)').style('font-style', 'italic').style('font-size', '15px').style('fill', C.ochreI).text('k');
        gW.append('text').attr('x', XPF + 9).attr('y', 274).attr('text-anchor', 'start').style('font-family', 'var(--f-body)').style('font-style', 'italic').style('font-size', '15px').style('fill', C.ochreI).text('v');
      }

      /* node states */
      var state = {};
      if (pl.ok) pl.tgt.forEach(function (id) { state[id] = 'ok'; });
      else {
        pl.tgt.forEach(function (id) { state[id] = forced ? 'fbad' : 'aim'; });
        (pl.blockAll || [pl.block]).forEach(function (id) { if (!forced || state[id] !== 'fbad') state[id] = 'bad'; });
        if (forced) pl.tgt.forEach(function (id) { state[id] = 'fbad'; });
      }
      if (anim) state = {};
      var editNode = pl.tok && pl.tok.node, editBox = pl.tok && pl.tok.onBox;

      function nodeEl(id) {
        var n = G[id];
        if (!n) return;
        if (id === 'adO' && !(st.method === 'adapter' && st.site === 'o')) return;
        if (id === 'adD' && !(st.method === 'adapter' && st.site === 'down')) return;
        if (id === 'prefix' && st.method !== 'prefix') return;
        var g = gN.append('g').attr('transform', 'translate(' + n.x + ',' + n.y + ')').attr('data-node', id);
        var s = state[id], isEdit = id === editNode;
        var strokeC = s === 'ok' ? C.moss : (s === 'bad' || s === 'fbad') ? C.seal : isEdit || id === editBox ? C.ochre : C.ink3;
        var sw2 = s === 'ok' || s === 'bad' || s === 'fbad' ? 2 : (isEdit || id === editBox ? 1.5 : 1.1);
        var fillS = s === 'ok' ? C.mossS : (s === 'bad' || s === 'fbad') ? C.sealS : (isEdit ? C.ochreS : null);
        if (n.k === 'add' || n.k === 'had') {
          g.append('circle').attr('r', n.r).attr('fill', C.paper).attr('stroke', C.ink3).attr('stroke-width', 1.1);
          rich(g, [[n.k === 'add' ? '+' : '⊙', 'r']], 14, 'var(--f-body)', C.ink2).attr('y', n.k === 'add' ? 0 : 0.5);
          return;
        }
        var rx = n.k === 'nl' ? n.h / 2 : n.k === 'rope' ? 10 : 3;
        g.append('rect').attr('x', -n.w / 2).attr('y', -n.h / 2).attr('width', n.w).attr('height', n.h).attr('rx', rx).attr('fill', C.paper);
        if (fillS) g.append('rect').attr('x', -n.w / 2).attr('y', -n.h / 2).attr('width', n.w).attr('height', n.h).attr('rx', rx).attr('fill', fillS);
        g.append('rect').attr('x', -n.w / 2).attr('y', -n.h / 2).attr('width', n.w).attr('height', n.h).attr('rx', rx).attr('fill', 'none')
          .attr('stroke', strokeC).attr('stroke-width', sw2).attr('stroke-dasharray', s === 'aim' ? '4 3' : (id === editBox ? '3 2' : null));
        var lab = n.lab, prime = s === 'ok' || s === 'fbad';
        if (id === 'act') lab = [[sw ? 'SiLU' : (st.act === 'gelu' ? 'GELU' : 'ReLU'), 'r']];
        if (!lab) return;
        if (prime && id !== 'adO' && id !== 'adD') { lab = lab.slice(); lab.splice(1, 0, ['′', 'r']); }
        var fam = n.k === 'nl' || n.k === 'mix' || n.k === 'adapter' ? 'var(--f-body)' : 'var(--f-display)';
        var size = n.k === 'lin' ? 18 : n.k === 'ln' ? 16 : n.k === 'rope' ? 15.5 : n.k === 'prefix' ? 20 : 15;
        var lc = s === 'ok' ? C.mossI : (s === 'bad' || s === 'fbad') ? C.sealI : (isEdit ? C.ochreI : C.ink);
        rich(g, lab, size, fam, lc).attr('y', n.k === 'prefix' ? -7 : 0.5);
        if (n.k === 'prefix') g.append('text').attr('y', 16).attr('text-anchor', 'middle').style('font-family', 'var(--f-body)').style('font-style', 'italic').style('font-size', '14px').style('fill', C.ochreI).text('k, v');
      }
      ['ln1', 'wq', 'wk', 'wv', 'ropeq', 'ropek', 'cup', 'smax', 'mix', 'wo', 'adO', 'add1', 'ln2', 'wgate', 'wup', 'act', 'had', 'wdown', 'adD', 'add2', 'prefix'].forEach(nodeEl);

      /* the edit token: ghost at its site; a live token where the rewrite left it */
      function badge(parent, tk, col, dashed, op) {
        var g = parent.append('g').attr('transform', 'translate(' + tk.x + ',' + tk.y + ') scale(' + KS + ')');
        g.append('rect').attr('x', -tk.w / 2).attr('y', -10).attr('width', tk.w).attr('height', 20).attr('rx', 4).attr('fill', C.paper);
        var fr = g.append('g');
        if (!dashed) fr.append('rect').attr('x', -tk.w / 2).attr('y', -10).attr('width', tk.w).attr('height', 20).attr('rx', 4).attr('fill', col === C.seal ? C.sealS : col === C.moss ? C.mossS : C.ochreS);
        fr.append('rect').attr('x', -tk.w / 2).attr('y', -10).attr('width', tk.w).attr('height', 20).attr('rx', 4).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 1.5).attr('stroke-dasharray', dashed ? '3 2' : null);
        rich(fr, tk.lab, 14, 'var(--f-body)', col === C.ochre ? C.ochreI : col === C.moss ? C.mossI : col === C.seal ? C.sealI : col).attr('y', 0.5);
        if (op !== undefined) fr.style('opacity', op);
        return g;
      }
      var tk = pl.tok, liveTok = null;
      if (tk && !tk.node) {
        var absorbed = (pl.ok && pl.kind !== 'inplace') || forced;
        if (anim) {
          badge(gK, tk, C.ochre, true, 0.55);
          liveTok = badge(gK, tk, C.ochre, false);
        } else if (absorbed) {
          badge(gK, tk, C.ochre, true, 0.6);
        } else if (pl.kind === 'inplace') {
          badge(gK, tk, C.moss, false);
        } else {
          badge(gK, tk, C.ochre, true, 0.55);
          /* the live token stops one half-height short of the blocking box, so it never hides it */
          var ends = [];
          pl.steps.forEach(function (s) { if (s.paths.length) ends = s.paths.map(function (p) { return backOff(p, 13); }); });
          if (!ends.length) ends = [[tk.x, tk.y]];
          ends.forEach(function (e) { badge(gK, { x: e[0], y: e[1], lab: tk.lab, w: tk.w }, C.seal, false); });
        }
      }
      /* trails */
      var trails = [];
      pl.steps.forEach(function (s, i) {
        var els = s.paths.map(function (p) {
          return gT.append('path').attr('d', rpath(p, 6)).attr('fill', 'none').attr('stroke', C.ochre).attr('stroke-width', 2.4).attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round').attr('opacity', 0.9);
        });
        trails.push(els);
      });
      var ftrails = [];
      if ((forced || anim === 'force') && pl.fpaths) {
        ftrails = pl.fpaths.map(function (p) {
          return gT.append('path').attr('d', rpath(p, 6)).attr('fill', 'none').attr('stroke', C.seal).attr('stroke-width', 2.2).attr('stroke-linecap', 'round').attr('stroke-dasharray', anim ? null : '5 4');
        });
      }
      /* chips */
      var chips = pl.steps.map(function (s) {
        var pos = s.chip;
        if (!pos && s.paths.length) { var p = s.paths[0], L = polyLen(p) / 2, acc = 0; for (var i = 1; i < p.length; i++) { var sl = Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); if (acc + sl >= L) { var f = (L - acc) / sl; pos = [p[i - 1][0] + f * (p[i][0] - p[i - 1][0]) + 20, p[i - 1][1] + f * (p[i][1] - p[i - 1][1])]; break; } acc += sl; } }
        if (!pos) return null;
        var txt = (s.bad ? '✗ ' : '') + s.rule, col = s.bad ? C.seal : C.ochre;
        var g = gC.append('g').attr('transform', 'translate(' + pos[0] + ',' + pos[1] + ') scale(' + KS + ')');
        var box = g.append('rect');
        var lt = g.append('text').attr('text-anchor', 'middle').attr('dominant-baseline', 'central').attr('y', 0.5)
          .style('font-family', 'var(--f-ui)').style('font-weight', '500').style('font-size', '12.5px').style('fill', s.bad ? C.sealI : C.ochreI).text(txt);
        /* size the chip to the label as set in the page font (an estimate until the node is laid out) */
        var tw = 0; try { tw = lt.node().getComputedTextLength(); } catch (e) { tw = 0; }
        var w = (tw > 0 ? tw : 7.1 * txt.length) + 11;
        box.attr('x', -w / 2).attr('y', -9).attr('width', w).attr('height', 18).attr('rx', 3).attr('fill', C.paper).attr('stroke', col).attr('stroke-width', 1.1);
        return g;
      });
      /* output badge */
      var outG = gC.append('g').attr('transform', 'translate(' + XC + ',728) scale(' + KS + ')');
      if (pl.ok || forced) {
        var v = pl.ok ? cur.delta : cur.forced, col2 = pl.ok ? C.mossI : C.sealI;
        var ot = outG.append('text').attr('text-anchor', 'middle').attr('dominant-baseline', 'central').style('font-family', 'var(--f-ui)').style('font-weight', '500').style('font-size', '13px').style('fill', col2);
        ot.append('tspan').text(pl.ok ? 'merged · ' : 'forced · ');
        ot.append('tspan').style('font-family', 'var(--f-mono)').style('font-weight', '400').style('font-variant-numeric', 'tabular-nums').text('‖Δ‖/‖y‖ = ' + fd(v));
      } else {
        outG.append('text').attr('text-anchor', 'middle').attr('dominant-baseline', 'central').style('font-family', 'var(--f-ui)').style('font-weight', '500').style('font-size', '13px').style('fill', C.sealI).text('no exact merge into ' + plainT(st.target));
      }
      svg.attr('aria-label', svgSummary());
      return { trails: trails, ftrails: ftrails, chips: chips, tok: liveTok, outG: outG };
    }
    function plainT(t) { return { wq: 'W_Q', wk: 'W_K', wv: 'W_V', wo: 'W_O', wup: 'W_up', wgate: 'W_gate', wdown: 'W_down', ln1: 'LN1', inplace: 'LN', fold1: 'W_Q,K,V', fold2: 'the FFN', head: 'the head' }[t] || t; }
    function svgSummary() {
      var pl = cur.plan;
      return 'Pre-LN block, ' + (st.act === 'swiglu' ? 'SwiGLU' : st.act.toUpperCase()) + ' FFN, RoPE ' + (st.rope ? 'on' : 'off') + ', bias slots ' + (st.bias ? 'on' : 'off') + '. ' +
        methodDef(st.method).label + ' toward ' + plainT(st.target) + ': ' + (pl.ok ? 'merges; relative error ' + fd(cur.delta) : 'blocked by rule ' + pl.steps[pl.steps.length - 1].rule + (st.view === 'forced' ? '; forced merge error ' + fd(cur.forced) : '')) + '.';
    }

    /* ---------------- animation ---------------- */
    function revealLog(i) { var li = pLog.querySelector('li[data-i="' + i + '"]'); if (li) li.classList.remove('pending'); }
    function play(forced) {
      var myTok = ++animToken;
      renderControls();
      renderLog(true); renderNums();
      if (A.reducedMotion() || !d3.transition) { draw(); renderLog(false); return; }
      var ctx = draw(forced ? 'force' : 'rewrite'), pl = cur.plan;
      ctx.chips.forEach(function (c) { if (c) c.style('opacity', 0); });
      ctx.outG.style('opacity', 0);
      var all = [];
      ctx.trails.forEach(function (els) { els.forEach(function (e) { all.push(e); }); });
      ctx.ftrails.forEach(function (e) { all.push(e); });
      all.forEach(function (e) { var L = e.node().getTotalLength(); e.attr('stroke-dasharray', L + ' ' + L).attr('stroke-dashoffset', L); });
      var toks = ctx.tok ? [ctx.tok] : [];
      var phases = pl.steps.map(function (s, i) { return { els: ctx.trails[i], chip: ctx.chips[i], i: i, bad: s.bad }; });
      if (forced && pl.fpaths) phases.push({ els: ctx.ftrails, chip: null, i: pl.steps.length, force: true });
      var k = 0;
      function alive() { return myTok === animToken; }
      function next() {
        if (!alive()) return;
        if (k >= phases.length) return finish();
        var ph = phases[k++];
        if (!ph.els || !ph.els.length) { setTimeout(function () { if (!alive()) return; if (ph.chip) ph.chip.transition().duration(220).style('opacity', 1); revealLog(ph.i); setTimeout(next, ph.bad ? 650 : 380); }, 260); return; }
        var L0 = ph.els[0].node().getTotalLength(), dur = Math.max(480, Math.min(1100, L0 * 7));
        while (toks.length && toks.length < ph.els.length) { var c = toks[0].clone(true); toks.push(c); }
        if (ph.force && toks[0]) toks.forEach(function (t) { t.selectAll('rect').attr('stroke', A.css('--seal')); t.selectAll('text').style('fill', A.css('--seal')); });
        var done = 0;
        ph.els.forEach(function (e, j) {
          var L = e.node().getTotalLength();
          e.transition().duration(dur).ease(d3.easeCubicInOut).attr('stroke-dashoffset', 0);
          var t = toks[j];
          if (t) t.transition().duration(dur).ease(d3.easeCubicInOut).attrTween('transform', function () { return function (u) { var p = e.node().getPointAtLength(u * L); return 'translate(' + p.x + ',' + p.y + ') scale(' + KS + ')'; }; });
          setTimeout(function () {
            if (!alive()) return;
            if (++done < ph.els.length) return;
            if (ph.chip) ph.chip.transition().duration(220).style('opacity', 1);
            revealLog(ph.i);
            if (ph.bad && toks[0]) {
              toks.forEach(function (t2) { t2.selectAll('rect').attr('stroke', A.css('--seal')); t2.selectAll('text').style('fill', A.css('--seal')); });
              var tr = toks[0].attr('transform');
              toks[0].transition().duration(70).attr('transform', tr + ' translate(-4,0)').transition().duration(70).attr('transform', tr + ' translate(4,0)').transition().duration(70).attr('transform', tr + ' translate(-3,0)').transition().duration(70).attr('transform', tr);
              setTimeout(next, 700);
            } else setTimeout(next, 200);
          }, dur + 20);
        });
      }
      function finish() {
        setTimeout(function () {
          if (!alive()) return;
          draw();
          renderLog(false);
          live.textContent = svgSummary();
        }, 260);
      }
      setTimeout(next, 250);
    }

    /* ---------------- update ---------------- */
    function render() {
      renderControls(); renderLegend(); renderHyp(); renderLog(false); renderNums(); renderMatrix(); renderFoot(); draw();
      live.textContent = svgSummary();
    }
    /* stop any animation and show the final frame (theme change, resize) */
    function settle() { animToken++; draw(); pLog.querySelectorAll('.pending').forEach(function (n) { n.classList.remove('pending'); }); }
    var update = function () { animToken++; st.view = 'rewrite'; fixTarget(); compute(); render(); };
    compute();
    render();
    A.onTheme(settle);
    /* chip widths are measured in the page font: redraw once it has loaded, unless the reader has started a rewrite */
    var mountTok = animToken;
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { if (cur && animToken === mountTok) settle(); });
    el.__merge = { get st() { return st; }, get WM() { return WM; }, evaluate: evaluate, plan: plan, gen: gen, forward: forward, adapted: adapted, ropeLS: ropeLS, cur: function () { return cur; }, set: function (o) { for (var k in o) st[k] = o[k]; update(); }, force: function () { st.view = 'forced'; renderControls(); renderLog(false); renderNums(); draw(); }, reseed: function (n) { st.seed = n; WM = gen(n, A); update(); } };
  });
})();
