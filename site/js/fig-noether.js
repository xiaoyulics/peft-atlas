/* Fig. "noether" — Noether hyperbolas and the crossover.
   theory/framework.md: Thm III.5 (charges for any cotangent signal under gradient flow, after Zhao et al. 2023),
   Cor III.6(b) (the two weight-decay conventions), Prop III.7(a) (the unweighted balanced slice mu = 0) and
   Thm III.9 (closed-form isotropic-charge dynamics and the crossover scale; Tarmoun et al. 2021, Min et al. 2021).

   Panel A, scalar LoRA. w = w0 + s b a, loss l = (w - w*)^2 / 2, rates eta_A = 1, eta_B = lambda (time is measured
   in units of 1/eta_A). The charge Phi = b^2/eta_B - a^2/eta_A is integrated, never assumed:
     gradient flow  RK4 with h = 1e-3;   GD  Euler steps of size eta (learning rates eta*eta_A, eta*eta_B);
     Adam           beta = (0.9, 0.999), eps = 1e-8, bias-corrected, learning rates eta*eta_A, eta*eta_B.
   Weight decay: per-time  (a,b)' += -wd (a,b);  learning-rate-coupled  a' += -eta_A wd a, b' += -eta_B wd b
   (PyTorch SGD weight_decay; the decay term of PyTorch AdamW). Under Adam the decay is applied decoupled from the
   moments, as AdamW does.

   Panel B, matrix LoRA with m = n = 6, r = 2, W = W_res + s B A, loss ||W - W*||_F^2 / 2, rates eta_A = 1,
   eta_B = lambda. Initialisations: LoRA (B0 = 0, A0 A0^T = c I exactly, kappa = c/eta_A), Init[B] (A0 = 0,
   B0^T B0 = c I, kappa = -c/eta_B), PiSSA (B0 = U_r (S_r/s)^{1/2}, A0 = (S_r/s)^{1/2} V_r^T, kappa = 0 iff
   eta_A = eta_B). The flow is integrated by the implicit midpoint rule (Newton-solved; see simMatrix); at log-spaced times we take the SVD of Z = s B A, read
   f_i = u_i^T (B B^T/eta_B) u_i and g_i = v_i^T (A^T A/eta_A) v_i off the factors, and compare the closed-form
   W-velocity of Thm III.9 with s(dB A + B dA). */
(function () {
  'use strict';

  /* ---------- Panel A constants ---------- */
  var SA = 1, DA = 1;                     // s and delta = w* - w0, so the fibre over w* is {s a b = delta}
  var LIM = 2.4;                          // half-width of the (a, b) window
  var H_FLOW = 1e-3;                      // RK4 step (theory/visuals.md, section 3)
  var T_PLAIN = 10, T_DECAY = 40;         // horizons without / with weight decay
  var AB1 = 0.9, AB2 = 0.999, AEPS = 1e-8;
  var LAMS_A = [1, 2, 4, 8];
  var LEVELS = [-4, -2.25, -1, -0.25, 0.25, 1, 2.25, 4];
  var START = { lora: [1.2, 0], initb: [0, 1.2] };

  /* ---------- Panel B constants ---------- */
  var SEED_B = 20210718;                  // fixed seed (Tarmoun et al. appeared at ICML 2021)
  var ST = [2.0, 0.3];                    // singular values of the target update Delta*
  var SR = [1.2, 0.6];                    // top-2 singular values of the seeded W0 used by PiSSA
  var XLO = 0.003, XHI = 30;              // shared sigma axis of Panel B
  var MID_REL = 0.01, MID_STIFF = 4;      // implicit midpoint: relative change <= 1% per step, and h <= 4 / (stiffness bound)
  var STEP_CAP = 60000, T_CAP = 4000, REST_TOL = 1e-8, FRAME_GROW = 1.025;
  var RES_FLOOR = 1e-6;                   // residuals are counted while ||W'|| >= RES_FLOOR * max ||W'|| (see measure())
  var RANGE_B = { ks: [-2, 1], kc: [-3, 0], kl: [0, 4] };

  /* ---------- formatting ---------- */
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sup(n) { return String(n).split('').map(function (ch) { return SUP[ch] || ch; }).join(''); }
  function minus(s) { return String(s).replace(/^-/, '−'); }
  function sci(x, d) {
    if (x === 0) return '0';
    if (!isFinite(x)) return x > 0 ? '∞' : '−∞';
    var p = x.toExponential(d == null ? 1 : d).split('e'), e = parseInt(p[1], 10);
    if (e === 0) return minus(p[0]);
    return minus(p[0]) + ' × 10' + sup(e);
  }
  function num(x, p) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '—';
    var a = Math.abs(x);
    if (a >= 1e-3 && a < 1e5) return minus(String(+x.toPrecision(p || 4)));
    return sci(x, (p || 4) - 2);
  }
  /* the same numbers for HTML: a real superscript instead of Unicode digits (Plex Mono has no superscript minus) */
  function sciH(x, d) {
    if (x === 0) return '0';
    if (!isFinite(x)) return x > 0 ? '∞' : '−∞';
    var p = x.toExponential(d == null ? 1 : d).split('e'), e = parseInt(p[1], 10);
    if (e === 0) return minus(p[0]);
    return minus(p[0]) + '<span class="x">×</span>10<sup>' + minus(String(e)) + '</sup>';
  }
  function numH(x, p) {
    if (x === 0) return '0';
    if (!isFinite(x)) return '—';
    var a = Math.abs(x);
    if (a >= 1e-3 && a < 1e5) return minus(String(+x.toPrecision(p || 4)));
    return sciH(x, (p || 4) - 2);
  }
  function fx(x, d) { var v = x.toFixed(d); if (/^-0\.?0*$/.test(v)) v = v.slice(1); return minus(v); }
  function pow2s(k) { return k >= 0 ? String(Math.pow(2, k)) : '1/' + Math.pow(2, -k); }

  /* ======================================================================
     Panel A numerics: scalar LoRA
     ====================================================================== */
  function rhsA(a, b, lam, conv, wd, out) {
    var e = SA * a * b - DA, ga = SA * e * b, gb = SA * e * a;
    if (conv === 'pertime') { out[0] = -ga - wd * a; out[1] = -lam * gb - wd * b; }
    else if (conv === 'coupled') { out[0] = -(ga + wd * a); out[1] = -lam * (gb + wd * b); }
    else { out[0] = -ga; out[1] = -lam * gb; }
  }
  function simScalar(p) {
    var lam = p.lam, conv = p.conv, wd = conv === 'none' ? 0 : p.wd, opt = p.opt;
    var T = conv === 'none' ? T_PLAIN : T_DECAY;
    var h = opt === 'flow' ? H_FLOW : p.eta, N = Math.round(T / h);
    var K = Math.max(1, Math.ceil(N / 900)), M = Math.floor(N / K) + 2;
    var ts = new Float64Array(M), as = new Float64Array(M), bs = new Float64Array(M);
    var ph = new Float64Array(M), mus = new Float64Array(M), lw = new Float64Array(M);
    var a = p.a0, b = p.b0, Phi0 = b * b / lam - a * a, rate = 2 * wd;
    if (Math.abs(Phi0) < 1e-13 * (a * a + b * b / lam)) Phi0 = 0;      // a start placed on Phi = 0 (preset) up to round-off
    var n = 0, maxDev = 0, idRes = 0, diverged = false, k;
    var k1 = [0, 0], k2 = [0, 0], k3 = [0, 0], k4 = [0, 0];
    var ma = 0, mb = 0, va = 0, vb = 0;
    function rec(t) {
      ts[n] = t; as[n] = a; bs[n] = b; ph[n] = b * b / lam - a * a; mus[n] = b * b - a * a; lw[n] = Phi0 * Math.exp(-rate * t); n++;
    }
    rec(0);
    for (k = 1; k <= N; k++) {
      var Pk = b * b / lam - a * a, ek = SA * a * b - DA, scale = a * a + b * b / lam;
      if (opt === 'flow') {
        rhsA(a, b, lam, conv, wd, k1);
        rhsA(a + 0.5 * h * k1[0], b + 0.5 * h * k1[1], lam, conv, wd, k2);
        rhsA(a + 0.5 * h * k2[0], b + 0.5 * h * k2[1], lam, conv, wd, k3);
        rhsA(a + h * k3[0], b + h * k3[1], lam, conv, wd, k4);
        a += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        b += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      } else if (opt === 'gd') {
        rhsA(a, b, lam, conv, wd, k1);
        a += h * k1[0]; b += h * k1[1];
        if (conv === 'none' && scale > 0) {
          /* scalar identity (expand the step): Phi_{k+1} = (1 - lambda eta^2 (s e_k)^2) Phi_k */
          var pred = Pk * (1 - lam * h * h * SA * SA * ek * ek);
          idRes = Math.max(idRes, Math.abs(b * b / lam - a * a - pred) / scale);
        }
      } else {
        var ga = SA * ek * b, gb = SA * ek * a;
        if (conv === 'pertime') { a -= h * wd * a; b -= h * wd * b; }
        else if (conv === 'coupled') { a -= h * wd * a; b -= lam * h * wd * b; }
        ma = AB1 * ma + (1 - AB1) * ga; mb = AB1 * mb + (1 - AB1) * gb;
        va = AB2 * va + (1 - AB2) * ga * ga; vb = AB2 * vb + (1 - AB2) * gb * gb;
        var c1 = 1 - Math.pow(AB1, k), c2 = 1 - Math.pow(AB2, k);
        a -= h * (ma / c1) / (Math.sqrt(va / c2) + AEPS);
        b -= lam * h * (mb / c1) / (Math.sqrt(vb / c2) + AEPS);
      }
      if (!(Math.abs(a) < 1e6 && Math.abs(b) < 1e6)) { diverged = true; break; }
      var t = k * h;
      maxDev = Math.max(maxDev, Math.abs(b * b / lam - a * a - Phi0 * Math.exp(-rate * t)));
      if (k % K === 0 || k === N) rec(t);
    }
    var last = n - 1;
    return {
      t: ts.subarray(0, n), a: as.subarray(0, n), b: bs.subarray(0, n), phi: ph.subarray(0, n), mu: mus.subarray(0, n), law: lw.subarray(0, n),
      n: n, T: T, steps: N, h: h, Phi0: Phi0, PhiT: ph[last], muT: mus[last], mu0: p.b0 * p.b0 - p.a0 * p.a0, lawT: lw[last], tEnd: ts[last],
      maxDev: maxDev, idRes: idRes, diverged: diverged, loss: 0.5 * Math.pow(SA * as[last] * bs[last] - DA, 2), scale: p.a0 * p.a0 + p.b0 * p.b0 / lam
    };
  }

  /* ======================================================================
     Panel B numerics: matrix LoRA, m = n = 6, r = 2
     ====================================================================== */
  /* two orthonormal columns in R^6 (stored row-major 6x2: [i*2 + k]), Gram-Schmidt applied twice */
  function orth62(rnd) {
    var Q = new Float64Array(12), i, k, p, pass;
    for (i = 0; i < 12; i++) Q[i] = rnd.normal();
    for (k = 0; k < 2; k++) {
      for (pass = 0; pass < 2; pass++) for (p = 0; p < k; p++) {
        var d = 0; for (i = 0; i < 6; i++) d += Q[i * 2 + k] * Q[i * 2 + p];
        for (i = 0; i < 6; i++) Q[i * 2 + k] -= d * Q[i * 2 + p];
      }
      var nr = 0; for (i = 0; i < 6; i++) nr += Q[i * 2 + k] * Q[i * 2 + k];
      nr = Math.sqrt(nr); for (i = 0; i < 6; i++) Q[i * 2 + k] /= nr;
    }
    return Q;
  }
  function lowRank(U, S, V) {          // U diag(S) V^T, 6x6 row-major
    var Z = new Float64Array(36);
    for (var i = 0; i < 6; i++) for (var j = 0; j < 6; j++) Z[i * 6 + j] = S[0] * U[i * 2] * V[j * 2] + S[1] * U[i * 2 + 1] * V[j * 2 + 1];
    return Z;
  }
  function setupB(A) {
    var rnd = A.rng(SEED_B);
    var D = { Ut: orth62(rnd), Vt: orth62(rnd), Q: orth62(rnd), QB: orth62(rnd), Ur: orth62(rnd), Vr: orth62(rnd), St: ST, Sr: SR };
    D.Zt = lowRank(D.Ut, ST, D.Vt);                    // Delta* (rank 2)
    var P = lowRank(D.Ur, SR, D.Vr);                   // the PiSSA adapter's starting product s B0 A0
    D.ZP = new Float64Array(36);
    for (var i = 0; i < 36; i++) D.ZP[i] = D.Zt[i] + P[i];
    return D;
  }
  /* f_kappa(x) = sqrt(x + kappa^2/4) - kappa/2 and g_kappa = f_kappa + kappa, without cancellation */
  function fgk(x, kap) {
    var root = Math.sqrt(x + kap * kap / 4), ak = Math.abs(kap) / 2;
    var small = x / (root + ak), large = root + ak;
    if (kap === 0) return [root, root];
    return kap > 0 ? [small, large] : [large, small];
  }
  /* QR of the two columns col(k) in R^6 by Gram-Schmidt applied twice */
  function qr2(get) {
    var q0 = new Float64Array(6), q1 = new Float64Array(6), i, r00 = 0, r01 = 0, r11 = 0, d;
    for (i = 0; i < 6; i++) { q0[i] = get(i, 0); r00 += q0[i] * q0[i]; }
    r00 = Math.sqrt(r00);
    if (!(r00 > 0)) return null;
    for (i = 0; i < 6; i++) { q0[i] /= r00; q1[i] = get(i, 1); }
    for (var pass = 0; pass < 2; pass++) {
      d = 0; for (i = 0; i < 6; i++) d += q0[i] * q1[i];
      r01 += d; for (i = 0; i < 6; i++) q1[i] -= d * q0[i];
    }
    for (i = 0; i < 6; i++) r11 += q1[i] * q1[i];
    r11 = Math.sqrt(r11);
    if (!(r11 > 1e-300)) return null;
    for (i = 0; i < 6; i++) q1[i] /= r11;
    return { q0: q0, q1: q1, r00: r00, r01: r01, r11: r11 };
  }
  /* compact SVD of Z = s B A (rank 2) from the factors: Z = s QB (RB RA^T) QA^T, then a 2x2 SVD */
  function svdFactors(Bm, Am, s) {
    var qb = qr2(function (i, k) { return Bm[i * 2 + k]; });
    var qa = qr2(function (j, k) { return Am[k * 6 + j]; });
    if (!qb || !qa) return null;
    var m00 = s * (qb.r00 * qa.r00 + qb.r01 * qa.r01), m01 = s * qb.r01 * qa.r11, m10 = s * qb.r11 * qa.r01, m11 = s * qb.r11 * qa.r11;
    var p = m00 * m00 + m10 * m10, q = m00 * m01 + m10 * m11, r = m01 * m01 + m11 * m11;
    var th = 0.5 * Math.atan2(2 * q, p - r), c = Math.cos(th), sn = Math.sin(th);
    var v1 = [c, sn], v2 = [-sn, c];
    var Mv1 = [m00 * v1[0] + m01 * v1[1], m10 * v1[0] + m11 * v1[1]];
    var s1 = Math.hypot(Mv1[0], Mv1[1]);
    if (!(s1 > 0)) return null;
    var u1 = [Mv1[0] / s1, Mv1[1] / s1], s2 = Math.abs(m00 * m11 - m01 * m10) / s1;
    var u2 = [-u1[1], u1[0]];
    var Mv2 = [m00 * v2[0] + m01 * v2[1], m10 * v2[0] + m11 * v2[1]];
    if (u2[0] * Mv2[0] + u2[1] * Mv2[1] < 0) { u2[0] = -u2[0]; u2[1] = -u2[1]; }
    var U = new Float64Array(12), V = new Float64Array(12);
    for (var i = 0; i < 6; i++) {
      U[i * 2] = qb.q0[i] * u1[0] + qb.q1[i] * u1[1]; U[i * 2 + 1] = qb.q0[i] * u2[0] + qb.q1[i] * u2[1];
      V[i * 2] = qa.q0[i] * v1[0] + qa.q1[i] * v1[1]; V[i * 2 + 1] = qa.q0[i] * v2[0] + qa.q1[i] * v2[1];
    }
    return { s1: s1, s2: s2, U: U, V: V };
  }
  /* dense solve of the 24 x 24 system M x = r by Gaussian elimination with partial pivoting (M is overwritten) */
  function solve24(M, r) {
    var n = 24, i, j, k, piv, mx, t, f;
    for (k = 0; k < n; k++) {
      piv = k; mx = Math.abs(M[k * n + k]);
      for (i = k + 1; i < n; i++) { t = Math.abs(M[i * n + k]); if (t > mx) { mx = t; piv = i; } }
      if (!(mx > 0)) return false;
      if (piv !== k) {
        for (j = k; j < n; j++) { t = M[k * n + j]; M[k * n + j] = M[piv * n + j]; M[piv * n + j] = t; }
        t = r[k]; r[k] = r[piv]; r[piv] = t;
      }
      var d = M[k * n + k];
      for (i = k + 1; i < n; i++) {
        f = M[i * n + k] / d;
        if (f === 0) continue;
        for (j = k + 1; j < n; j++) M[i * n + j] -= f * M[k * n + j];
        r[i] -= f * r[k];
      }
    }
    for (i = n - 1; i >= 0; i--) {
      t = r[i];
      for (j = i + 1; j < n; j++) t -= M[i * n + j] * r[j];
      r[i] = t / M[i * n + i];
    }
    return true;
  }
  /* Matrix LoRA flow  B' = -eta_B s G A^T,  A' = -eta_A s B^T G,  G = s B A - Z*  (state y = [B (6x2), A (2x6)], row-major).
     Integrator: the implicit midpoint rule y1 = y0 + h f((y0 + y1)/2), solved by Newton with the analytic Jacobian.
     It is the one-stage Gauss-Legendre method; for any quadratic Phi it gives Phi(y1) - Phi(y0) = h dPhi(ybar) f(ybar),
     so the charge stays at -kappa I up to round-off exactly when dPhi . f = 0, i.e. when Thm III.5 holds. It is A-stable:
     the step is set by accuracy (relative change <= MID_REL), capped at MID_STIFF / (stiffness bound) so that the fast
     B-mode stays damped (midpoint amplification (1 - z/2)/(1 + z/2) >= -1/3). Rest: ||y'|| t <= REST_TOL ||y||, which a
     saddle plateau never meets (its escape is exponential in t). */
  function simMatrix(D, p) {
    var s = p.s, c = p.c, eA = 1, eB = p.lam, init = p.init, i, j, k;
    var Y = new Float64Array(24), Bm = Y.subarray(0, 12), Am = Y.subarray(12, 24);
    var Zs = init === 'pissa' ? D.ZP : D.Zt;
    var kappa, iso = true;
    if (init === 'lora') { for (k = 0; k < 2; k++) for (j = 0; j < 6; j++) Am[k * 6 + j] = Math.sqrt(c) * D.Q[j * 2 + k]; kappa = c / eA; }
    else if (init === 'initb') { for (i = 0; i < 12; i++) Bm[i] = Math.sqrt(c) * D.QB[i]; kappa = -c / eB; }
    else {
      for (i = 0; i < 6; i++) for (k = 0; k < 2; k++) Bm[i * 2 + k] = D.Ur[i * 2 + k] * Math.sqrt(D.Sr[k] / s);
      for (k = 0; k < 2; k++) for (j = 0; j < 6; j++) Am[k * 6 + j] = Math.sqrt(D.Sr[k] / s) * D.Vr[j * 2 + k];
      iso = eA === eB; kappa = iso ? 0 : NaN;      // Phi0 = (S_r/s)(1/eta_B - 1/eta_A): isotropic iff eta_A = eta_B
    }
    var mu = s * Math.sqrt(eA * eB), sstar = iso ? mu * Math.abs(kappa) / 2 : NaN;
    var Gb = new Float64Array(36);
    /* f(y) into out; leaves G(y) in Gb; returns ||G||_F^2 */
    function rhs(y, out) {
      var g2 = 0, ii, jj, gg;
      for (ii = 0; ii < 6; ii++) {
        var b0 = y[ii * 2], b1 = y[ii * 2 + 1];
        for (jj = 0; jj < 6; jj++) { gg = s * (b0 * y[12 + jj] + b1 * y[18 + jj]) - Zs[ii * 6 + jj]; Gb[ii * 6 + jj] = gg; g2 += gg * gg; }
      }
      for (ii = 0; ii < 6; ii++) {
        var s0 = 0, s1 = 0;
        for (jj = 0; jj < 6; jj++) { gg = Gb[ii * 6 + jj]; s0 += gg * y[12 + jj]; s1 += gg * y[18 + jj]; }
        out[ii * 2] = -eB * s * s0; out[ii * 2 + 1] = -eB * s * s1;
      }
      for (jj = 0; jj < 6; jj++) {
        var t0 = 0, t1 = 0;
        for (ii = 0; ii < 6; ii++) { gg = Gb[ii * 6 + jj]; t0 += y[ii * 2] * gg; t1 += y[ii * 2 + 1] * gg; }
        out[12 + jj] = -eA * s * t0; out[18 + jj] = -eA * s * t1;
      }
      return g2;
    }
    /* M = I - h2 * Df(y), Df analytic; Gb must hold G(y) */
    var M = new Float64Array(576);
    function newtonMatrix(y, h2) {
      var sB = eB * s, sA = eA * s, ii, jj, kk, q, l, pp;
      M.fill(0);
      var a00 = 0, a01 = 0, a11 = 0, b00 = 0, b01 = 0, b11 = 0;
      for (jj = 0; jj < 6; jj++) { var x0 = y[12 + jj], x1 = y[18 + jj]; a00 += x0 * x0; a01 += x0 * x1; a11 += x1 * x1; }
      for (ii = 0; ii < 6; ii++) { var z0 = y[ii * 2], z1 = y[ii * 2 + 1]; b00 += z0 * z0; b01 += z0 * z1; b11 += z1 * z1; }
      var AA = [[a00, a01], [a01, a11]], BB = [[b00, b01], [b01, b11]];
      for (ii = 0; ii < 6; ii++) for (kk = 0; kk < 2; kk++) {
        var row = (ii * 2 + kk) * 24;
        for (q = 0; q < 2; q++) {
          M[row + ii * 2 + q] = -sB * s * AA[q][kk];                                   // dB'/dB = -eta_B s^2 (. A A^T)
          for (l = 0; l < 6; l++) M[row + 12 + q * 6 + l] = -sB * (s * y[ii * 2 + q] * y[12 + kk * 6 + l] + (kk === q ? Gb[ii * 6 + l] : 0));
        }
      }
      for (kk = 0; kk < 2; kk++) for (jj = 0; jj < 6; jj++) {
        var rw = (12 + kk * 6 + jj) * 24;
        for (pp = 0; pp < 6; pp++) for (q = 0; q < 2; q++) M[rw + pp * 2 + q] = -sA * ((kk === q ? Gb[pp * 6 + jj] : 0) + s * y[pp * 2 + kk] * y[12 + q * 6 + jj]);
        for (q = 0; q < 2; q++) M[rw + 12 + q * 6 + jj] = -sA * s * BB[kk][q];        // dA'/dA = -eta_A s^2 (B^T B .)
      }
      for (pp = 0; pp < 576; pp++) M[pp] *= -h2;
      for (pp = 0; pp < 24; pp++) M[pp * 25] += 1;
    }
    var Y0 = new Float64Array(24), Yb = new Float64Array(24), F0 = new Float64Array(24), Fb = new Float64Array(24), Rv = new Float64Array(24);
    var newtonIts = 0, rejects = 0;
    /* one implicit-midpoint step of size hh from Y (F0 = f(Y) on entry); false if Newton did not converge */
    function midStep(hh) {
      var h2 = hh / 2, it, pp, sc, dn, prev = Infinity;
      Y0.set(Y);
      for (pp = 0; pp < 24; pp++) Yb[pp] = Y0[pp] + h2 * F0[pp];
      for (it = 0; it < 12; it++) {
        rhs(Yb, Fb);
        sc = 0;
        for (pp = 0; pp < 24; pp++) { Rv[pp] = Y0[pp] + h2 * Fb[pp] - Yb[pp]; sc = Math.max(sc, Math.abs(Yb[pp]), h2 * Math.abs(Fb[pp])); }
        newtonMatrix(Yb, h2);
        if (!solve24(M, Rv)) return false;
        dn = 0;
        for (pp = 0; pp < 24; pp++) { Yb[pp] += Rv[pp]; dn = Math.max(dn, Math.abs(Rv[pp])); }
        newtonIts++;
        if (dn <= 4e-16 * sc) break;                        // converged to round-off
        if (it >= 2 && dn > 0.5 * prev && dn <= 1e-13 * sc) break;   // stagnated at round-off
        prev = dn;
        if (it === 11) return false;
      }
      for (pp = 0; pp < 24; pp++) Y[pp] = 2 * Yb[pp] - Y0[pp];
      return true;
    }
    var K1 = new Float64Array(24), K1B = K1.subarray(0, 12), K1A = K1.subarray(12, 24);
    var Wf = new Float64Array(36), Wc = new Float64Array(36);
    var F = { t: [], s1: [], s2: [], f1: [], f2: [], g1: [], g2: [], res: [], drift: [], wn: [] };
    var maxRes = 0, maxDrift = 0, nRes = 0, wmax = 0;
    function measure(t) {
      rhs(Y, K1);
      var wn = 0, ii, jj, kk;
      for (ii = 0; ii < 6; ii++) for (jj = 0; jj < 6; jj++) {
        var v = s * (K1B[ii * 2] * Am[jj] + K1B[ii * 2 + 1] * Am[6 + jj] + Bm[ii * 2] * K1A[jj] + Bm[ii * 2 + 1] * K1A[6 + jj]);
        Wf[ii * 6 + jj] = v; wn += v * v;
      }
      wn = Math.sqrt(wn); wmax = Math.max(wmax, wn);
      /* the charge Phi = B^T B/eta_B - A A^T/eta_A, measured */
      var P00 = 0, P01 = 0, P11 = 0;
      for (ii = 0; ii < 6; ii++) { P00 += Bm[ii * 2] * Bm[ii * 2] / eB; P01 += Bm[ii * 2] * Bm[ii * 2 + 1] / eB; P11 += Bm[ii * 2 + 1] * Bm[ii * 2 + 1] / eB; }
      for (jj = 0; jj < 6; jj++) { P00 -= Am[jj] * Am[jj] / eA; P01 -= Am[jj] * Am[6 + jj] / eA; P11 -= Am[6 + jj] * Am[6 + jj] / eA; }
      var drift = iso ? Math.sqrt((P00 + kappa) * (P00 + kappa) + 2 * P01 * P01 + (P11 + kappa) * (P11 + kappa)) : NaN;
      var sv = svdFactors(Bm, Am, s), s1 = 0, s2 = 0, f = [NaN, NaN], g = [NaN, NaN], res = NaN;
      if (sv) {
        s1 = sv.s1; s2 = sv.s2;
        for (kk = 0; kk < 2; kk++) {
          var bu0 = 0, bu1 = 0, av0 = 0, av1 = 0;
          for (ii = 0; ii < 6; ii++) { bu0 += Bm[ii * 2] * sv.U[ii * 2 + kk]; bu1 += Bm[ii * 2 + 1] * sv.U[ii * 2 + kk]; }
          for (jj = 0; jj < 6; jj++) { av0 += Am[jj] * sv.V[jj * 2 + kk]; av1 += Am[6 + jj] * sv.V[jj * 2 + kk]; }
          f[kk] = (bu0 * bu0 + bu1 * bu1) / eB;       // u_k^T (B B^T / eta_B) u_k
          g[kk] = (av0 * av0 + av1 * av1) / eA;       // v_k^T (A^T A / eta_A) v_k
        }
        if (iso && s2 > 1e-9 && s2 > 1e-7 * s1 && wn > 0) {
          var fc = [fgk(s1 * s1 / (mu * mu), kappa), fgk(s2 * s2 / (mu * mu), kappa)];
          var GV = new Float64Array(12), UtG = new Float64Array(12);
          for (ii = 0; ii < 6; ii++) for (kk = 0; kk < 2; kk++) { var acc = 0; for (jj = 0; jj < 6; jj++) acc += Gb[ii * 6 + jj] * sv.V[jj * 2 + kk]; GV[ii * 2 + kk] = acc; }
          for (kk = 0; kk < 2; kk++) for (jj = 0; jj < 6; jj++) { var ac2 = 0; for (ii = 0; ii < 6; ii++) ac2 += sv.U[ii * 2 + kk] * Gb[ii * 6 + jj]; UtG[kk * 6 + jj] = ac2; }
          var dn = 0;
          for (ii = 0; ii < 6; ii++) for (jj = 0; jj < 6; jj++) {
            var w = 0;
            for (kk = 0; kk < 2; kk++) w += fc[kk][1] * GV[ii * 2 + kk] * sv.V[jj * 2 + kk] + fc[kk][0] * sv.U[ii * 2 + kk] * UtG[kk * 6 + jj];
            w *= -mu * mu;
            Wc[ii * 6 + jj] = w; dn += (w - Wf[ii * 6 + jj]) * (w - Wf[ii * 6 + jj]);
          }
          res = Math.sqrt(dn) / wn;
          /* near rest W' is a difference of two O(1) terms (PiSSA's rank-4 target is not reachable), so the relative
             residual is set by cancellation, not by the theorem: only checkpoints with ||W'|| >= RES_FLOOR max ||W'|| count */
          if (wn >= RES_FLOOR * wmax) { maxRes = Math.max(maxRes, res); nRes++; }
        }
      }
      if (iso) maxDrift = Math.max(maxDrift, drift);
      F.t.push(t); F.s1.push(s1); F.s2.push(s2); F.f1.push(f[0]); F.f2.push(f[1]); F.g1.push(g[0]); F.g2.push(g[1]);
      F.res.push(res); F.drift.push(drift); F.wn.push(wn);
      var fy = 0, ny = 0;
      for (ii = 0; ii < 24; ii++) { fy += K1[ii] * K1[ii]; ny += Y[ii] * Y[ii]; }
      return Math.sqrt(fy / Math.max(ny, 1e-300));       // relative speed ||y'|| / ||y||
    }
    var t = 0, steps = 0, stop = 'cap', g2, hh, tNext, relSpeed;
    measure(0);
    g2 = rhs(Y, F0);
    /* first checkpoint: a small fraction of the initial time scale */
    var nrm0 = 0, sp0 = 0;
    for (i = 0; i < 24; i++) { nrm0 += Y[i] * Y[i]; sp0 += F0[i] * F0[i]; }
    tNext = MID_REL * Math.sqrt(nrm0 / Math.max(sp0, 1e-300));
    while (steps < STEP_CAP) {
      var na = 0, nb = 0, ny = 0, fy = 0;
      for (i = 0; i < 12; i++) { na += Am[i] * Am[i]; nb += Bm[i] * Bm[i]; }
      for (i = 0; i < 24; i++) { ny += Y[i] * Y[i]; fy += F0[i] * F0[i]; }
      var Lst = s * s * (eB * na + eA * nb) + s * Math.sqrt(eA * eB) * Math.sqrt(g2);   // bound on the stiffness
      hh = Math.min(MID_STIFF / Lst, MID_REL * Math.sqrt(ny / Math.max(fy, 1e-300)), tNext - t);
      if (!(hh > 0)) hh = tNext - t;
      var tries = 0;
      while (!midStep(hh)) { Y.set(Y0); hh *= 0.5; rejects++; if (++tries > 30) { stop = 'nan'; break; } }
      if (stop === 'nan') break;
      t += hh; steps++;
      if (!isFinite(Y[0] + Y[12])) { stop = 'nan'; break; }
      g2 = rhs(Y, F0);
      if (t >= tNext * (1 - 1e-12)) {
        relSpeed = measure(t);
        tNext = t * FRAME_GROW;
        if (relSpeed * t <= REST_TOL) { stop = 'rest'; break; }
        if (t >= T_CAP) { stop = 'tcap'; break; }
      }
    }
    if (F.t[F.t.length - 1] !== t) measure(t);
    var n = F.t.length;
    return {
      F: F, n: n, T: t, steps: steps, stop: stop, kappa: kappa, iso: iso, mu: mu, sstar: sstar, eA: eA, eB: eB, s: s, c: c, init: init,
      maxRes: maxRes, nRes: nRes, maxDrift: maxDrift, s1T: F.s1[n - 1], s2T: F.s2[n - 1], newtonIts: newtonIts, rejects: rejects,
      loss: 0.5 * g2, phi0: init === 'pissa' ? [D.Sr[0] / s * (1 / eB - 1 / eA), D.Sr[1] / s * (1 / eB - 1 / eA)] : null
    };
  }

  /* ======================================================================
     scoped CSS
     ====================================================================== */
  function injectCSS() {
    if (document.getElementById('css-noether')) return;
    var F = '[data-figure="noether"] ';
    var css = [
      F + '.nh-stage{container-type:inline-size;padding:clamp(.85rem,2.2vw,1.35rem)}',
      F + '.nh-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.15rem 1rem}',
      F + '.nh-title{font-family:var(--f-display);font-weight:var(--w-head);font-size:clamp(1.45rem,1.05rem + 1.9cqi,2.05rem);line-height:1.08;letter-spacing:-.012em;margin:0;color:var(--ink)}',
      F + '.nh-title i{color:var(--tide)}',
      F + '.nh-kicker{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.nh-instr{margin:.45rem 0 1rem;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2);max-width:70ch}',
      F + '.nh-instr b{color:var(--ink);font-weight:600}',
      F + '.nh-panel{border-top:1px solid var(--ink-3);padding-top:.75rem;margin-top:.4rem}',
      F + '.nh-panel + .nh-panel{margin-top:1.6rem}',
      F + '.nh-phead{display:flex;flex-wrap:wrap;align-items:baseline;gap:.2rem .7rem;margin-bottom:.5rem}',
      F + '.nh-badge{font-family:var(--f-ui);font-weight:600;font-size:.78rem;letter-spacing:.06em;color:var(--paper);background:var(--ink-2);border-radius:3px;padding:.05rem .38rem;align-self:center}',
      F + '.nh-ptitle{font-family:var(--f-display);font-size:clamp(1.12rem,.95rem + .8cqi,1.38rem);font-weight:var(--w-head);color:var(--ink);line-height:1.2}',
      F + '.nh-ptitle i{font-family:var(--f-body)}',
      F + '.nh-pdim{font-family:var(--f-body);font-weight:var(--w-body);font-size:.8em;color:var(--ink-2);white-space:nowrap}',
      F + '.nh-group > .seg{justify-self:start;max-width:100%}',
      F + '.nh-pthm{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.nh-formula{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem 1.4rem;padding:.45rem .75rem;margin:0 0 .85rem;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);font-size:.88rem;line-height:1.5;color:var(--ink);overflow-x:auto}',
      F + '.nh-formula > span{display:inline-flex;flex-wrap:nowrap;align-items:center;gap:.15rem .45rem;min-width:0}',
      F + '.nh-formula mjx-container{white-space:nowrap}',
      F + '.nh-formula mjx-container svg{max-width:none}',
      F + '.nh-tag{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}',
      F + '.nh-controls{display:grid;grid-template-columns:minmax(0,1fr);gap:.7rem 1.4rem;margin-bottom:1rem}',
      F + '.nh-group{min-width:0;border-top:1px solid var(--rule);padding-top:.45rem;display:grid;gap:.4rem;align-content:start}',
      F + '.nh-group > .nh-tag{color:var(--ink-2)}',
      F + '.nh-row{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem .55rem}',
      F + '.nh-sl{display:grid;gap:.05rem;min-width:0}',
      F + '.nh-sl-top{display:flex;justify-content:space-between;align-items:baseline;gap:0 .5rem;min-width:0}',
      F + '.nh-sl label{font-family:var(--f-body);font-size:var(--fs-sm);color:var(--ink);white-space:nowrap}',
      F + '.nh-sl label .nm{font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);margin-left:.35rem}',
      F + '.nh-sl output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}',
      F + '.nh-sl input[type=range]{margin:0;height:1.25rem}',
      F + '.nh-sl.is-off{opacity:.45}',
      F + '.seg button:disabled{opacity:.4;cursor:not-allowed}',
      F + '.nh-btn{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.02em;padding:.32rem .6rem;border-radius:4px;border:1px solid var(--rule);background:var(--paper);color:var(--ink);cursor:pointer;white-space:nowrap}',
      F + '.nh-btn:hover:not(:disabled){border-color:var(--tide);color:var(--tide)}',
      F + '.nh-btn:disabled{opacity:.4;cursor:not-allowed}',
      F + '.nh-btn.play{min-width:4.6rem;background:var(--tide);border-color:var(--tide);color:var(--paper)}',
      F + '.nh-btn.play:hover{filter:brightness(1.08);color:var(--paper)}',
      F + '.nh-hint{font-family:var(--f-body);font-size:.8rem;color:var(--ink-2);line-height:1.45}',
      F + '.nh-body{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem 1.5rem;align-items:start}',
      F + '.nh-plot{min-width:0}',
      F + '.nh-plot > svg,.nh-chart > svg{display:block;width:100%;height:auto;overflow:visible}',
      F + '.nh-plot > svg{touch-action:pan-y pinch-zoom;cursor:crosshair;max-width:520px;margin-inline:auto}',
      F + '.nt{text-transform:none;letter-spacing:0}',
      F + '.nh-handle{cursor:grab;touch-action:none}',
      F + '.nh-handle:focus{outline:none}',
      F + '.nh-handle:focus-visible .ring{stroke:var(--ochre);stroke-width:2.5px}',
      F + '.nh-legend{display:flex;flex-wrap:wrap;gap:.25rem 1rem;margin:.5rem 0 0;font-family:var(--f-ui);font-size:var(--fs-xs);line-height:1.35;color:var(--ink-2)}',
      F + '.nh-li{display:inline-flex;align-items:center;gap:.4rem;white-space:nowrap}',
      F + '.nh-li svg{flex:none;overflow:visible}',
      F + '.nh-li i{font-family:var(--f-body);font-size:.9rem}',
      F + '.nh-meter{display:grid;gap:.5rem;align-content:start;min-width:0;background:var(--paper);border:1px solid var(--rule);border-radius:var(--radius);padding:.8rem .9rem .85rem}',
      F + '.nh-big{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-weight:500;font-size:clamp(1.7rem,1.25rem + 2cqi,2.35rem);line-height:1;letter-spacing:-.02em;color:var(--ink);white-space:nowrap}',
      F + '.nh-big sup{font-size:.52em;vertical-align:.85em;line-height:0;letter-spacing:0;margin-left:.05em}',
      F + '.nh-big .x{margin:0 .18em;font-size:.82em}',
      F + 'table.nh-tab .x{margin:0 .12em;font-size:.95em}',
      F + 'table.nh-tab sup{font-size:.72em;vertical-align:.45em;line-height:0}',
      F + '.nh-biglab{font-family:var(--f-body);font-size:.95rem;color:var(--ink-2)}',
      F + '.nh-biglab i{font-family:var(--f-body)}',
      F + '.nh-chips{display:flex;flex-wrap:wrap;gap:.35rem}',
      F + '.nh-chip{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--f-ui);font-weight:500;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;padding:.14rem .5rem;border-radius:999px;border:1px solid currentColor;white-space:normal;line-height:1.3}',
      F + '.nh-chip.ok{color:var(--moss-ink);background:var(--moss-soft)}',
      F + '.nh-chip.bad{color:var(--seal-ink);background:var(--seal-soft)}',
      F + '.nh-chip.mid{color:var(--ochre-ink);background:var(--ochre-soft)}',
      F + 'table.nh-tab{width:100%;border-collapse:collapse;font-size:.8rem;line-height:1.35}',
      F + 'table.nh-tab th,' + F + 'table.nh-tab td{position:static;background:none;padding:.26rem .25rem;border-bottom:1px solid var(--rule);vertical-align:baseline;text-transform:none;letter-spacing:0}',
      F + 'table.nh-tab tr:last-child th,' + F + 'table.nh-tab tr:last-child td{border-bottom:0}',
      F + 'table.nh-tab thead th{font-family:var(--f-ui);font-size:.75rem;letter-spacing:.02em;color:var(--ink-2);font-weight:500;text-align:right}',
      F + 'table.nh-tab thead th:first-child{text-align:left}',
      F + 'table.nh-tab tbody th{font-family:var(--f-body);font-weight:400;color:var(--ink);text-align:left;white-space:nowrap;font-size:.86rem}',
      F + 'table.nh-tab td{font-family:var(--f-mono);font-variant-numeric:tabular-nums;text-align:right;color:var(--ink);white-space:nowrap}',
      F + 'table.nh-tab td.l{text-align:left;white-space:normal;font-family:var(--f-body);color:var(--ink-2);font-size:.8rem}',
      F + 'table.nh-tab td.ok{color:var(--moss-ink)}',
      F + 'table.nh-tab td.bad{color:var(--seal-ink)}',
      F + 'table.nh-tab td.mid{color:var(--ochre-ink)}',
      F + '.nh-why{margin:.15rem 0 0;font-family:var(--f-body);font-size:var(--fs-sm);line-height:1.5;color:var(--ink-2)}',
      F + '.nh-why i{font-family:var(--f-body)}',
      F + '.nh-why .lead{font-family:var(--f-ui);font-weight:500;font-size:var(--fs-xs);letter-spacing:.06em;text-transform:uppercase;margin-right:.4rem;color:var(--ink-2)}',
      F + '.nh-sep{border:0;border-top:1px solid var(--rule);margin:.15rem 0}',
      F + '.nh-warn{font-family:var(--f-body);font-size:.84rem;line-height:1.45;color:var(--seal-ink);background:var(--seal-soft);border:1px solid currentColor;border-radius:4px;padding:.4rem .55rem}',
      F + '.nh-warn i{font-family:var(--f-body)}',
      F + '.nh-time{display:flex;align-items:center;gap:.6rem;margin:.1rem 0 .5rem}',
      F + '.nh-time input[type=range]{flex:1;min-width:0;margin:0}',
      F + '.nh-time output{font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap;min-width:6.5rem;text-align:right}',
      F + '.nh-foot{margin-top:1.2rem;padding-top:.55rem;border-top:1px solid var(--rule);font-family:var(--f-body);font-size:.8rem;line-height:1.55;color:var(--ink-2)}',
      F + '.nh-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
      '@container (min-width: 600px){',
      F + '.nh-controls{grid-template-columns:repeat(2,minmax(0,1fr))}',
      '}',
      '@container (min-width: 780px){',
      F + '.nh-controls{grid-template-columns:minmax(0,1fr) minmax(0,.9fr) minmax(0,1.2fr)}',
      F + '.nh-body.a{grid-template-columns:minmax(0,1.05fr) minmax(0,1fr)}',
      F + '.nh-body.b{grid-template-columns:minmax(0,1.55fr) minmax(0,1fr)}',
      '}'
    ].join('\n');
    var st = document.createElement('style');
    st.id = 'css-noether';
    st.textContent = css;
    document.head.appendChild(st);
  }

  /* ======================================================================
     the figure
     ====================================================================== */
  Atlas.register('noether', function (el, A) {
    injectCSS();
    var d3 = window.d3, h = A.h;
    var uid = 'nh' + Math.random().toString(36).slice(2, 7);
    var SVGNS = 'http://www.w3.org/2000/svg';
    var motion = !A.reducedMotion();

    /* ---------- state ---------- */
    var stA = { a0: START.lora[0], b0: START.lora[1], lam: 1, opt: 'flow', conv: 'none', wd: 0.1, eta: 0.05 };
    var stB = { init: 'lora', ks: 0, kc: 0, kl: 0, frame: -1 };
    var D = setupB(A);
    var RA = null, RB = null;
    var cacheB = {}, cacheKeys = [];
    function pB() { return { init: stB.init, s: Math.pow(2, stB.ks), c: Math.pow(2, stB.kc), lam: Math.pow(2, stB.kl) }; }
    function runB() {
      var p = pB(), key = p.init + '|' + stB.ks + '|' + (p.init === 'pissa' ? 'x' : stB.kc) + '|' + stB.kl;
      if (cacheB[key]) return cacheB[key];
      var r = simMatrix(D, p);
      cacheB[key] = r; cacheKeys.push(key);
      if (cacheKeys.length > 30) delete cacheB[cacheKeys.shift()];
      return r;
    }

    /* ---------- DOM: head ---------- */
    var stage = h('div', { class: 'stage nh-stage', role: 'group', 'aria-label': 'Noether hyperbolas and the crossover. Panel A: scalar LoRA trajectories ride the level sets of the charge Phi under gradient flow; a meter shows how Phi behaves under GD, Adam and two weight-decay conventions. Panel B: matrix LoRA, the closed-form gains f and g of Thm III.9 against the singular value sigma, with the crossover scale sigma star and singular values from the integrated flow.' });
    el.appendChild(stage);
    stage.appendChild(h('div', { class: 'nh-head' }, [
      h('h3', { class: 'nh-title', html: 'Noether hyperbolas and the <i>crossover</i>' }),
      h('span', { class: 'nh-kicker', html: 'Thm&nbsp;III.5 · Cor&nbsp;III.6 · Prop&nbsp;III.7 · Thm&nbsp;III.9' })
    ]));
    stage.appendChild(h('p', { class: 'nh-instr', html: '<b>Drag the start point</b> in A and watch gradient flow keep to its charge hyperbola on the way to the fibre of minimisers. Switch the optimizer or the weight decay and read the charge meter. In B, move <i>s</i>, <i>c</i> and <i>λ</i> to slide the crossover scale <i>σ</i>* past the singular values of Δ<i>W</i>, and press <b>Play</b> to replay the flow.' }));

    function tex(node, html) { node.innerHTML = html; A.typeset(node); }
    function sliderEl(id, labelHtml, nm, min, max, val) {
      var input = h('input', { type: 'range', id: id, min: min, max: max, step: 1, value: val });
      var out = h('output', { for: id });
      var wrap = h('div', { class: 'nh-sl' }, [h('div', { class: 'nh-sl-top' }, [h('label', { for: id, html: labelHtml + (nm ? '<span class="nm">' + nm + '</span>' : '') }), out]), input]);
      return { el: wrap, input: input, out: out };
    }
    function rangeEl(id, labelHtml, nm, min, max, step, val) {
      var input = h('input', { type: 'range', id: id, min: min, max: max, step: step, value: val });
      var out = h('output', { for: id });
      var wrap = h('div', { class: 'nh-sl' }, [h('div', { class: 'nh-sl-top' }, [h('label', { for: id, html: labelHtml + (nm ? '<span class="nm">' + nm + '</span>' : '') }), out]), input]);
      return { el: wrap, input: input, out: out };
    }

    /* ---------- DOM: panel A ---------- */
    var panA = h('section', { class: 'nh-panel', 'aria-label': 'Panel A: scalar LoRA' });
    stage.appendChild(panA);
    panA.appendChild(h('div', { class: 'nh-phead' }, [
      h('span', { class: 'nh-badge', text: 'A' }),
      h('span', { class: 'nh-ptitle', html: 'Scalar LoRA rides its charge' }),
      h('span', { class: 'nh-pthm', html: 'Thm III.5 · Cor III.6<span class="nt">(b)</span> · Prop III.7<span class="nt">(a)</span>' })
    ]));
    var formA = h('div', { class: 'nh-formula' });
    panA.appendChild(formA);
    tex(formA,
      '<span><span class="nh-tag">model</span>\\(w=w_0+s\\,b\\,a,\\ \\ \\ell=\\tfrac12(w-w^\\star)^2\\)</span>' +
      '<span><span class="nh-tag">charge</span>\\(\\Phi=\\dfrac{b^2}{\\eta_B}-\\dfrac{a^2}{\\eta_A}\\)</span>' +
      '<span><span class="nh-tag">unweighted</span>\\(\\mu=b^2-a^2\\)</span>' +
      '<span><span class="nh-tag">rates</span>\\(\\eta_A=1,\\ \\eta_B=\\lambda\\)</span>');

    var ctlA = h('div', { class: 'nh-controls' });
    panA.appendChild(ctlA);
    var optSeg = A.seg([{ value: 'flow', label: 'Flow · RK4' }, { value: 'gd', label: 'GD' }, { value: 'adam', label: 'Adam' }], stA.opt, function (v) { stA.opt = v; changedA(true); }, 'Optimizer');
    var etaSl = rangeEl(uid + '-eta', '<i>η</i>', 'step size', 0.01, 0.2, 0.01, stA.eta);
    etaSl.input.addEventListener('input', function () { stA.eta = +etaSl.input.value; changedA(true); });
    ctlA.appendChild(h('div', { class: 'nh-group' }, [h('span', { class: 'nh-tag', text: 'Optimizer' }), optSeg.el, etaSl.el]));
    var lamSegA = A.seg(LAMS_A.map(function (l) { return { value: l, label: String(l) }; }), stA.lam, function (v) { stA.lam = v; changedA(true); }, 'Rate ratio lambda = eta_B / eta_A');
    var presetRow = h('div', { class: 'nh-row' });
    [['lora', 'LoRA · b₀=0'], ['initb', 'Init[B] · a₀=0'], ['bal', 'Φ₀ = 0']].forEach(function (pr) {
      var bt = h('button', { type: 'button', class: 'nh-btn', text: pr[1] });
      bt.addEventListener('click', function () {
        if (pr[0] === 'bal') { stA.a0 = 0.35; stA.b0 = 0.35 * Math.sqrt(stA.lam); }   // on Phi = 0, below the fibre for every lambda <= 8
        else { stA.a0 = START[pr[0]][0]; stA.b0 = START[pr[0]][1]; }
        changedA(true);
      });
      presetRow.appendChild(bt);
    });
    ctlA.appendChild(h('div', { class: 'nh-group' }, [h('span', { class: 'nh-tag', html: 'Rate ratio <span class="nt">λ = η<sub>B</sub>/η<sub>A</sub></span>' }), lamSegA.el, h('span', { class: 'nh-tag', html: 'Start <span class="nt">(a₀, b₀)</span> · or drag the dot' }), presetRow]));
    var convSeg = A.seg([{ value: 'none', label: 'none' }, { value: 'pertime', label: 'per-time' }, { value: 'coupled', label: 'lr-coupled' }], stA.conv, function (v) { stA.conv = v; changedA(true); }, 'Weight-decay convention');
    var wdSl = rangeEl(uid + '-wd', 'wd', 'decay coefficient', 0.02, 0.3, 0.01, stA.wd);
    wdSl.input.addEventListener('input', function () { stA.wd = +wdSl.input.value; changedA(true); });
    var convHint = h('div', { class: 'nh-hint' });
    ctlA.appendChild(h('div', { class: 'nh-group' }, [h('span', { class: 'nh-tag', text: 'Weight decay' }), convSeg.el, wdSl.el, convHint]));

    var bodyA = h('div', { class: 'nh-body a' });
    panA.appendChild(bodyA);
    var plotWrap = h('div', { class: 'nh-plot' });
    var plotSvg = document.createElementNS(SVGNS, 'svg');
    plotSvg.setAttribute('role', 'img');
    plotWrap.appendChild(plotSvg);
    var legendA = h('div', { class: 'nh-legend' });
    plotWrap.appendChild(legendA);
    bodyA.appendChild(plotWrap);
    var meter = h('div', { class: 'nh-meter' });
    bodyA.appendChild(meter);
    meter.appendChild(h('span', { class: 'nh-tag', text: 'Charge meter · measured along the run' }));
    var meterChart = h('div', { class: 'nh-chart' });
    var meterLegend = h('div', { class: 'nh-legend', style: 'margin:0' });
    meter.appendChild(meterLegend);
    meter.appendChild(meterChart);
    var bigLab = h('div', { class: 'nh-biglab' });
    var big = h('div', { class: 'nh-big' });
    var chipsA = h('div', { class: 'nh-chips' });
    var tabA = h('table', { class: 'nh-tab' });
    var whyA = h('p', { class: 'nh-why', 'aria-live': 'polite' });
    meter.appendChild(bigLab); meter.appendChild(big); meter.appendChild(chipsA); meter.appendChild(tabA);
    meter.appendChild(h('hr', { class: 'nh-sep' }));
    meter.appendChild(whyA);

    /* ---------- DOM: panel B ---------- */
    var panB = h('section', { class: 'nh-panel', 'aria-label': 'Panel B: matrix LoRA and the crossover' });
    stage.appendChild(panB);
    panB.appendChild(h('div', { class: 'nh-phead' }, [
      h('span', { class: 'nh-badge', text: 'B' }),
      h('span', { class: 'nh-ptitle', html: 'The crossover <i>σ</i>* in matrix LoRA <span class="nh-pdim">(<i>m</i> = <i>n</i> = 6, <i>r</i> = 2)</span>' }),
      h('span', { class: 'nh-pthm', html: 'Thm III.9 · Cor III.6<span class="nt">(a)</span>' })
    ]));
    var formB = h('div', { class: 'nh-formula' });
    panB.appendChild(formB);
    tex(formB,
      '<span>\\(\\dot W=-\\mu^2\\big[G\\,V g_\\kappa V^{\\top}+U f_\\kappa U^{\\top}G\\big]\\)</span>' +
      '<span>\\(f_\\kappa=\\sqrt{x+\\kappa^2/4}-\\kappa/2,\\ \\ g_\\kappa=f_\\kappa+\\kappa\\)</span>' +
      '<span>\\(x=\\Sigma^2/\\mu^2,\\ \\ \\Phi=-\\kappa I,\\ \\ \\mu=s\\sqrt{\\eta_A\\eta_B}\\)</span>' +
      '<span class="nh-star">\\(\\sigma^*=\\mu|\\kappa|/2\\)</span>');

    var ctlB = h('div', { class: 'nh-controls' });
    panB.appendChild(ctlB);
    var initSeg = A.seg([{ value: 'lora', label: 'LoRA' }, { value: 'initb', label: 'Init[B]' }, { value: 'pissa', label: 'PiSSA' }], stB.init, function (v) {
      stB.init = v; if (v === 'pissa') { stB.kl = 0; slL.input.value = 0; }
      changedB(true);
    }, 'Initialisation');
    var initHint = h('div', { class: 'nh-hint' });
    ctlB.appendChild(h('div', { class: 'nh-group' }, [h('span', { class: 'nh-tag', text: 'Initialisation' }), initSeg.el, initHint]));
    var slS = sliderEl(uid + '-s', '<i>s</i> = α/<i>r</i>', 'scale', RANGE_B.ks[0], RANGE_B.ks[1], stB.ks);
    var slC = sliderEl(uid + '-c', '<i>c</i>', 'A₀A₀ᵀ = cI', RANGE_B.kc[0], RANGE_B.kc[1], stB.kc);
    slS.input.addEventListener('input', function () { stB.ks = +slS.input.value; changedB(true); });
    slC.input.addEventListener('input', function () { stB.kc = +slC.input.value; changedB(true); });
    ctlB.appendChild(h('div', { class: 'nh-group' }, [h('span', { class: 'nh-tag', text: 'Scale and init size' }), slS.el, slC.el]));
    var slL = sliderEl(uid + '-l', '<i>λ</i> = η<sub>B</sub>/η<sub>A</sub>', 'LoRA+ ratio', RANGE_B.kl[0], RANGE_B.kl[1], stB.kl);
    slL.input.addEventListener('input', function () { stB.kl = +slL.input.value; changedB(true); });
    var presetB = h('div', { class: 'nh-row' });
    [['Reset', function () { stB.init = 'lora'; stB.ks = 0; stB.kc = 0; stB.kl = 0; }]].forEach(function (pr) {
      var bt = h('button', { type: 'button', class: 'nh-btn', text: pr[0] });
      bt.addEventListener('click', function () { pr[1](); initSeg.set(stB.init); slS.input.value = stB.ks; slC.input.value = stB.kc; slL.input.value = stB.kl; changedB(true); });
      presetB.appendChild(bt);
    });
    ctlB.appendChild(h('div', { class: 'nh-group' }, [h('span', { class: 'nh-tag', text: 'Learning rates' }), slL.el, presetB]));

    var bodyB = h('div', { class: 'nh-body b' });
    panB.appendChild(bodyB);
    var chartsB = h('div', { style: 'min-width:0' });
    var legendB = h('div', { class: 'nh-legend', style: 'margin:0 0 .35rem' });
    var playBtn = h('button', { type: 'button', class: 'nh-btn play', text: '▶ Play' });
    var timeIn = h('input', { type: 'range', id: uid + '-t', min: 0, max: 1, step: 1, value: 1, 'aria-label': 'Time along the integrated flow' });
    var timeOut = h('output', { for: uid + '-t' });
    var timeRow = h('div', { class: 'nh-time' }, [playBtn, timeIn, timeOut]);
    var gainBox = h('div', { class: 'nh-chart' });
    var pathBox = h('div', { class: 'nh-chart' });
    chartsB.appendChild(legendB); chartsB.appendChild(timeRow); chartsB.appendChild(gainBox); chartsB.appendChild(pathBox);
    bodyB.appendChild(chartsB);
    var readB = h('div', { class: 'nh-meter' });
    bodyB.appendChild(readB);

    var foot = h('div', { class: 'nh-foot' });
    stage.appendChild(foot);
    var live = h('div', { class: 'nh-live', 'aria-live': 'polite' });
    stage.appendChild(live);

    if (!d3) { stage.appendChild(h('p', { class: 'muted', text: 'This figure needs D3 v7, which did not load.' })); return; }

    /* ---------- shared drawing helpers ---------- */
    function C(name) { return A.css(name); }
    function lineSw(color, w, dash, op) {
      return '<svg width="24" height="10" aria-hidden="true"><line x1="1" y1="5" x2="23" y2="5" stroke="' + color + '" stroke-width="' + w + '" stroke-linecap="round"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + (op ? ' stroke-opacity="' + op + '"' : '') + '/></svg>';
    }
    function dotSw(color, r) { return '<svg width="12" height="10" aria-hidden="true"><circle cx="6" cy="5" r="' + (r || 3.5) + '" fill="' + color + '" stroke="' + C('--paper') + '" stroke-width="1.2"/></svg>'; }
    function ringSw(color) { return '<svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="2.6" fill="none" stroke="' + color + '" stroke-width="1.1"/></svg>'; }
    function boxSw(color) { return '<svg width="14" height="10" aria-hidden="true"><rect x="1" y="1" width="12" height="8" rx="2" fill="' + color + '"/></svg>'; }
    function halo(sel, w, col) { return sel.style('paint-order', 'stroke').style('stroke', col || C('--paper')).style('stroke-width', (w || 3.5) + 'px').style('stroke-linejoin', 'round'); }
    function mixSoft(token, pct) { return 'color-mix(in srgb, ' + C(token) + ' ' + pct + '%, transparent)'; }

    /* ======================================================================
       Panel A rendering
       ====================================================================== */
    var animA = null;          // { start, dur, idx }
    var beadIdx = -1;          // index into RA arrays currently highlighted (-1 = end)
    var dragging = false, refocusHandle = false;
    var geomA = null;

    function computeA() { RA = simScalar(stA); }

    function hyperbolaPaths(Phi, lam, x, y) {
      var out = [], i, n = 160, pts;
      if (Phi < 0) {
        [1, -1].forEach(function (sg) {
          pts = [];
          for (i = 0; i <= n; i++) { var b = -LIM * 1.15 + 2.3 * LIM * i / n; pts.push([sg * Math.sqrt(b * b / lam - Phi), b]); }
          out.push(pts);
        });
      } else if (Phi > 0) {
        [1, -1].forEach(function (sg) {
          pts = [];
          for (i = 0; i <= n; i++) { var a = -LIM * 1.15 + 2.3 * LIM * i / n; pts.push([a, sg * Math.sqrt(lam * (Phi + a * a))]); }
          out.push(pts);
        });
      } else {
        out.push([[-LIM * 1.2, -Math.sqrt(lam) * LIM * 1.2], [LIM * 1.2, Math.sqrt(lam) * LIM * 1.2]]);
        out.push([[-LIM * 1.2, Math.sqrt(lam) * LIM * 1.2], [LIM * 1.2, -Math.sqrt(lam) * LIM * 1.2]]);
      }
      var line = d3.line().x(function (d) { return x(d[0]); }).y(function (d) { return y(d[1]); });
      return out.map(function (p) { return line(p); });
    }
    function fibrePaths(x, y) {
      var out = [], n = 160, line = d3.line().x(function (d) { return x(d[0]); }).y(function (d) { return y(d[1]); });
      [1, -1].forEach(function (sg) {
        var pts = [], lo = DA / (SA * LIM * 1.2), hi = LIM * 1.2;
        for (var i = 0; i <= n; i++) { var a = lo * Math.pow(hi / lo, i / n); pts.push([sg * a, sg * DA / (SA * a)]); }
        out.push(line(pts));
      });
      return out;
    }

    function drawPlotA() {
      var W = Math.max(260, Math.min(520, plotWrap.clientWidth || 400));
      var ml = 30, mr = 12, mt = 12, mb = 28, inner = W - ml - mr, H = inner + mt + mb;
      var x = d3.scaleLinear().domain([-LIM, LIM]).range([ml, ml + inner]);
      var y = d3.scaleLinear().domain([-LIM, LIM]).range([mt + inner, mt]);
      geomA = { x: x, y: y, W: W, H: H, ml: ml, mt: mt, inner: inner };
      var svg = d3.select(plotSvg).attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H);
      svg.selectAll('*').remove();
      var ink = C('--ink'), ink2 = C('--ink-2'), ink3 = C('--ink-3'), rule = C('--rule'), tide = C('--tide'), ochre = C('--ochre'), moss = C('--moss'), violet = C('--c-multiplicative'), paper = C('--paper'), seal = C('--seal');
      var lam = stA.lam;
      var cid = uid + '-clipA';
      svg.append('defs').append('clipPath').attr('id', cid).append('rect').attr('x', ml).attr('y', mt).attr('width', inner).attr('height', inner);
      svg.append('rect').attr('x', ml).attr('y', mt).attr('width', inner).attr('height', inner).attr('fill', paper).attr('stroke', rule);
      var g = svg.append('g').attr('clip-path', 'url(#' + cid + ')');
      // grid
      d3.range(-2, 3).forEach(function (v) {
        g.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', mt).attr('y2', mt + inner).attr('stroke', rule).attr('stroke-width', v === 0 ? 1 : 0.6).attr('stroke-opacity', v === 0 ? 1 : 0.7);
        g.append('line').attr('y1', y(v)).attr('y2', y(v)).attr('x1', ml).attr('x2', ml + inner).attr('stroke', rule).attr('stroke-width', v === 0 ? 1 : 0.6).attr('stroke-opacity', v === 0 ? 1 : 0.7);
      });
      // charge levels
      LEVELS.forEach(function (Lv) {
        hyperbolaPaths(Lv, lam, x, y).forEach(function (d) { g.append('path').attr('d', d).attr('fill', 'none').attr('stroke', ink3).attr('stroke-opacity', 0.32).attr('stroke-width', 0.9); });
      });
      // slices: mu = 0 (unweighted) and Phi = 0 (rate-weighted)
      if (lam !== 1) hyperbolaPaths(0, 1, x, y).forEach(function (d) { g.append('path').attr('d', d).attr('fill', 'none').attr('stroke', violet).attr('stroke-width', 1.5).attr('stroke-dasharray', '1.5 3.5').attr('stroke-linecap', 'round'); });
      hyperbolaPaths(0, lam, x, y).forEach(function (d) { g.append('path').attr('d', d).attr('fill', 'none').attr('stroke', moss).attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4'); });
      // the fibre over w*
      fibrePaths(x, y).forEach(function (d) { g.append('path').attr('d', d).attr('fill', 'none').attr('stroke', ochre).attr('stroke-width', 2.6).attr('stroke-linecap', 'round'); });
      // the start's own charge hyperbola
      var Phi0 = RA.Phi0;
      if (Math.abs(Phi0) > 1e-12) hyperbolaPaths(Phi0, lam, x, y).forEach(function (d) { g.append('path').attr('d', d).attr('fill', 'none').attr('stroke', ink2).attr('stroke-width', 1.3).attr('stroke-dasharray', '4 3'); });
      // trajectory
      var pts = [];
      for (var i = 0; i < RA.n; i++) pts.push([x(RA.a[i]), y(RA.b[i])]);
      var line = d3.line();
      var trajBase = g.append('path').attr('d', line(pts)).attr('fill', 'none').attr('stroke', RA.diverged ? seal : tide).attr('stroke-width', 2.4).attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round');
      var trajLen = 0;
      try { trajLen = trajBase.node().getTotalLength(); } catch (e) { trajLen = 0; }
      geomA.traj = trajBase; geomA.trajLen = trajLen; geomA.pts = pts;
      // cumulative arclength in screen units (for the bead)
      var cum = new Float64Array(pts.length);
      for (i = 1; i < pts.length; i++) cum[i] = cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      geomA.cum = cum;
      // end point
      var last = RA.n - 1;
      geomA.endDot = g.append('circle').attr('cx', pts[last][0]).attr('cy', pts[last][1]).attr('r', 4.5).attr('fill', paper).attr('stroke', RA.diverged ? seal : tide).attr('stroke-width', 2);
      geomA.bead = g.append('circle').attr('r', 5).attr('fill', tide).attr('stroke', paper).attr('stroke-width', 1.5).style('display', 'none');
      // labels (outside clip for legibility)
      var lab = svg.append('g');
      // fibre label at upper right
      var fb = 2.1, fa = DA / (SA * fb);
      var fibLab = halo(lab.append('text').attr('x', x(fa) + 9).attr('y', y(fb) + 4).attr('class', 'serif').style('font-size', '13.5px').style('font-style', 'italic').attr('fill', C('--ochre-ink')).text('ρ⁻¹(w★)'), 4, paper);
      // Phi0 label: on the start's own branch, in the quadrant the fibre s a b = w* - w0 > 0 never enters
      if (Math.abs(Phi0) > 1e-12) {
        var bl, al, sgA = stA.a0 < 0 ? -1 : 1, sgB = stA.b0 < 0 ? -1 : 1;
        if (Phi0 < 0) {                                    // branches a = +-sqrt(b^2/lam - Phi0): label at the far end in b
          bl = -sgA * 2.12; al = sgA * Math.sqrt(bl * bl / lam - Phi0);
          if (Math.abs(al) > LIM - 0.1) { al = sgA * (LIM - 0.25); bl = -sgA * Math.sqrt(lam * (al * al + Phi0)); }
        } else {                                           // branches b = +-sqrt(lam (Phi0 + a^2)): label on the other side in a
          al = -sgB * 1.55; bl = sgB * Math.sqrt(lam * (Phi0 + al * al));
          if (Math.abs(bl) > LIM - 0.1) { al = -sgB * Math.sqrt(Math.max(0, (LIM - 0.25) * (LIM - 0.25) / lam - Phi0)); bl = sgB * (LIM - 0.25); }
        }
        if (isFinite(bl) && isFinite(al) && Math.abs(bl) < LIM - 0.1 && Math.abs(al) < LIM) {
          var pl = lab.append('text').attr('x', x(al) + (al >= 0 ? -7 : 7)).attr('y', y(bl) + (bl >= 0 ? 4 : 12)).attr('text-anchor', al >= 0 ? 'end' : 'start').attr('fill', ink2).style('font-size', '12px').style('font-variant-numeric', 'tabular-nums').text('Φ = ' + num(Phi0, 3));
          try { var pb = pl.node().getBBox(); lab.insert('rect', function () { return pl.node(); }).attr('x', pb.x - 3).attr('y', pb.y - 1).attr('width', pb.width + 6).attr('height', pb.height + 2).attr('rx', 3).attr('fill', paper).attr('fill-opacity', 0.9); } catch (er) { halo(pl); }
        }
      }
      // axis labels and ticks
      [-2, -1, 1, 2].forEach(function (v) {
        lab.append('text').attr('x', x(v)).attr('y', mt + inner + 14).attr('text-anchor', 'middle').attr('class', 'num').attr('fill', ink2).style('font-size', '11px').text(minus(String(v)));
        lab.append('text').attr('x', ml - 6).attr('y', y(v) + 4).attr('text-anchor', 'end').attr('class', 'num').attr('fill', ink2).style('font-size', '11px').text(minus(String(v)));
      });
      lab.append('text').attr('x', ml + inner).attr('y', mt + inner + 24).attr('text-anchor', 'end').attr('class', 'serif').style('font-size', '13px').style('font-style', 'italic').attr('fill', ink2).text('a →');
      lab.append('text').attr('x', ml - 8).attr('y', mt + 4).attr('text-anchor', 'end').attr('class', 'serif').style('font-size', '13px').style('font-style', 'italic').attr('fill', ink2).attr('dominant-baseline', 'hanging').text('b');
      // start handle
      var hd = svg.append('g').attr('class', 'nh-handle').attr('tabindex', 0).attr('role', 'slider')
        .attr('aria-roledescription', 'draggable start point').attr('aria-valuemin', -LIM).attr('aria-valuemax', LIM).attr('aria-valuenow', stA.a0)
        .attr('aria-label', 'Start point (a0, b0). Arrow keys move it by 0.05, with Shift by 0.25.')
        .attr('aria-valuetext', 'a0 = ' + fx(stA.a0, 2) + ', b0 = ' + fx(stA.b0, 2) + ', Phi0 = ' + num(Phi0, 3))
        .attr('transform', 'translate(' + x(stA.a0) + ',' + y(stA.b0) + ')');
      hd.append('circle').attr('r', 16).attr('fill', 'transparent');
      hd.append('circle').attr('class', 'ring').attr('r', 8.5).attr('fill', 'none').attr('stroke', tide).attr('stroke-width', 1.5).attr('stroke-opacity', 0.55);
      hd.append('circle').attr('r', 5.5).attr('fill', tide).attr('stroke', paper).attr('stroke-width', 1.5);
      /* coordinate label: on the side the trajectory leaves from, and clear of the fibre label */
      var hx = x(stA.a0), hy = y(stA.b0), dvx = 0, dvy = 0;
      for (i = 1; i < pts.length; i++) { dvx = pts[i][0] - hx; dvy = pts[i][1] - hy; if (dvx * dvx + dvy * dvy > 900) break; }
      var fbb = null;
      try { var fbn = fibLab.node().getBBox(); fbb = { x0: fbn.x - 4, y0: fbn.y - 2, x1: fbn.x + fbn.width + 4, y1: fbn.y + fbn.height + 2 }; } catch (er) { fbb = null; }
      var cands = [[11, -10, 'start'], [-11, -10, 'end'], [11, 18, 'start'], [-11, 18, 'end']], best = null, bestScore = Infinity;
      cands.forEach(function (cd) {
        var w = 88, bx0 = hx + (cd[2] === 'start' ? cd[0] : cd[0] - w), by0 = hy + cd[1] - 11;
        var cx0 = (cd[2] === 'start' ? 1 : -1), cy0 = cd[1] < 0 ? -1 : 1;
        var score = (cx0 * dvx + cy0 * dvy > 0 ? 2 : 0);                       // facing the trajectory
        if (fbb && bx0 < fbb.x1 && bx0 + w > fbb.x0 && by0 < fbb.y1 && by0 + 13 > fbb.y0) score += 3;
        if (bx0 < ml || bx0 + w > ml + inner || by0 < mt || by0 + 14 > mt + inner) score += 1;
        if (score < bestScore) { bestScore = score; best = cd; }
      });
      halo(hd.append('text').attr('x', best[0]).attr('y', best[1]).attr('text-anchor', best[2]).attr('class', 'num').attr('fill', ink).style('font-size', '12px').text('(' + fx(stA.a0, 2) + ', ' + fx(stA.b0, 2) + ')'));
      geomA.handle = hd;
      plotSvg.setAttribute('aria-label', 'Plane of LoRA factors (a, b) with charge hyperbolas, the fibre s a b = w* - w0 and the trajectory from (' + fx(stA.a0, 2) + ', ' + fx(stA.b0, 2) + ').');
      hd.on('keydown', function (ev) {
        var st = ev.shiftKey ? 0.25 : 0.05, done = true;
        if (ev.key === 'ArrowLeft') stA.a0 -= st;
        else if (ev.key === 'ArrowRight') stA.a0 += st;
        else if (ev.key === 'ArrowUp') stA.b0 += st;
        else if (ev.key === 'ArrowDown') stA.b0 -= st;
        else done = false;
        if (done) { ev.preventDefault(); clampStart(); refocusHandle = true; changedA(true, true); }
      });
      hd.on('pointerdown', function (ev) { if (ev.button !== 0 && ev.pointerType === 'mouse') return; ev.stopPropagation(); ev.preventDefault(); startDrag(ev); });
    }
    function clampStart() {
      stA.a0 = Math.max(-LIM + 0.05, Math.min(LIM - 0.05, Math.round(stA.a0 * 100) / 100));
      stA.b0 = Math.max(-LIM + 0.05, Math.min(LIM - 0.05, Math.round(stA.b0 * 100) / 100));
    }
    function eventToAB(ev) {
      var r = plotSvg.getBoundingClientRect();
      var sx = geomA.W / r.width, sy = geomA.H / r.height;
      var px = (ev.clientX - r.left) * sx, py = (ev.clientY - r.top) * sy;
      return [geomA.x.invert(px), geomA.y.invert(py), px, py];
    }
    var dragPid = null;
    function startDrag(ev) {
      dragging = true; dragPid = ev.pointerId;
      try { plotSvg.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
      moveTo(ev);
    }
    function moveTo(ev) {
      var p = eventToAB(ev);
      stA.a0 = p[0]; stA.b0 = p[1]; clampStart();
      changedA(false, true);
    }
    plotSvg.addEventListener('pointerdown', function (ev) {
      if (dragging || !geomA) return;
      if (ev.pointerType !== 'mouse') return;        // touch: tap to place (click), drag the handle itself
      if (ev.button !== 0) return;
      var p = eventToAB(ev);
      if (p[2] < geomA.ml || p[2] > geomA.ml + geomA.inner || p[3] < geomA.mt || p[3] > geomA.mt + geomA.inner) return;
      ev.preventDefault(); startDrag(ev);
    });
    plotSvg.addEventListener('pointermove', function (ev) { if (dragging && ev.pointerId === dragPid) moveTo(ev); });
    function endDrag(ev) { if (!dragging || (ev && ev.pointerId !== dragPid)) return; dragging = false; dragPid = null; changedA(true, true); }
    plotSvg.addEventListener('pointerup', endDrag);
    plotSvg.addEventListener('pointercancel', endDrag);
    plotSvg.addEventListener('click', function (ev) {
      if (!geomA || ev.pointerType === 'mouse' || dragging) return;
      var p = eventToAB(ev);
      if (p[2] < geomA.ml || p[2] > geomA.ml + geomA.inner || p[3] < geomA.mt || p[3] > geomA.mt + geomA.inner) return;
      stA.a0 = p[0]; stA.b0 = p[1]; clampStart(); changedA(true, true);
    });

    function renderLegendA() {
      var tide = C('--tide'), ink2 = C('--ink-2'), ink3 = C('--ink-3'), ochre = C('--ochre'), moss = C('--moss'), violet = C('--c-multiplicative');
      var items = [
        lineSw(tide, 2.4) + '<span>trajectory</span>',
        lineSw(ochre, 2.6) + '<span>fibre ρ⁻¹(<i>w</i>★): s·a·b = w★ − w₀</span>',
        lineSw(ink2, 1.3, '4 3') + '<span>its charge level Φ = Φ₀</span>',
        lineSw(ink3, 0.9, '', 0.5) + '<span>other levels Φ = ±¼, ±1, ±9/4, ±4</span>',
        lineSw(moss, 1.4, '6 4') + '<span>Φ = 0 (rate-weighted slice)</span>'
      ];
      items.push(stA.lam !== 1 ? lineSw(violet, 1.5, '1.5 3.5') + '<span>μ = 0 (unweighted slice)</span>' : '<span style="color:var(--ink-2);font-style:italic">μ = 0 coincides with Φ = 0 at λ = 1</span>');
      legendA.innerHTML = items.map(function (s) { return '<span class="nh-li">' + s + '</span>'; }).join('');
    }

    function drawMeterA(idx) {
      var W = Math.max(240, meterChart.clientWidth || 320), H = 158, ml = 46, mr = 10, mt = 8, mb = 31;
      var svg = d3.select(meterChart).selectAll('svg').data([0]).join('svg');
      svg.attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', W).attr('height', H).attr('role', 'img')
        .attr('aria-label', 'Charge Phi, its closed-form law and the unweighted mu against time.');
      svg.selectAll('*').remove();
      var tide = C('--tide'), ink3 = C('--ink-3'), ink2 = C('--ink-2'), rule = C('--rule'), violet = C('--c-multiplicative'), ochre = C('--ochre');
      var lo = Infinity, hi = -Infinity, i;
      for (i = 0; i < RA.n; i++) { lo = Math.min(lo, RA.phi[i], RA.law[i], RA.mu[i]); hi = Math.max(hi, RA.phi[i], RA.law[i], RA.mu[i]); }
      lo = Math.min(lo, 0); hi = Math.max(hi, 0);
      if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
      var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
      var x = d3.scaleLinear().domain([0, RA.T]).range([ml, W - mr]);
      var y = d3.scaleLinear().domain([lo, hi]).range([H - mb, mt]).nice(4);
      var g = svg.append('g');
      y.ticks(4).forEach(function (v) {
        g.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y(v)).attr('y2', y(v)).attr('stroke', rule).attr('stroke-width', v === 0 ? 1 : 0.6);
        g.append('text').attr('x', ml - 5).attr('y', y(v) + 4).attr('text-anchor', 'end').attr('class', 'num').attr('fill', ink2).style('font-size', '11px').text(minus(String(+v.toPrecision(3))));
      });
      x.ticks(5).forEach(function (v) {
        g.append('text').attr('x', x(v)).attr('y', H - mb + 14).attr('text-anchor', 'middle').attr('class', 'num').attr('fill', ink2).style('font-size', '11px').text(String(v));
      });
      g.append('text').attr('x', W - mr).attr('y', H - 2).attr('text-anchor', 'end').attr('fill', ink2).style('font-size', '12px').style('font-weight', 500).text(stA.opt === 'flow' ? 'time t' : 'time t = step × η');
      var ln = function (arr) { return d3.line().x(function (d, j) { return x(RA.t[j]); }).y(function (d) { return y(d); })(Array.prototype.slice.call(arr)); };
      g.append('path').attr('d', ln(RA.mu)).attr('fill', 'none').attr('stroke', violet).attr('stroke-width', 1.6).attr('stroke-opacity', stA.lam === 1 ? 0.6 : 1);
      g.append('path').attr('d', ln(RA.law)).attr('fill', 'none').attr('stroke', ink2).attr('stroke-width', 1.3).attr('stroke-dasharray', '5 3.5');
      g.append('path').attr('d', ln(RA.phi)).attr('fill', 'none').attr('stroke', tide).attr('stroke-width', 2.2).attr('stroke-linejoin', 'round');
      if (idx != null && idx >= 0 && idx < RA.n) {
        var xx = x(RA.t[idx]);
        g.append('line').attr('x1', xx).attr('x2', xx).attr('y1', mt).attr('y2', H - mb).attr('stroke', ochre).attr('stroke-width', 1);
        g.append('circle').attr('cx', xx).attr('cy', y(RA.phi[idx])).attr('r', 3.5).attr('fill', tide);
      }
    }

    function lawName() {
      if (stA.conv === 'none') return { tex: 'Φ₀', thm: 'Thm III.5' };
      if (stA.conv === 'pertime') return { tex: 'e<sup>−2wd·t</sup>Φ₀', thm: 'Cor III.6(b)' };
      return { tex: 'e<sup>−2η wd·t</sup>Φ₀', thm: 'Cor III.6(b), which needs η<sub>A</sub> = η<sub>B</sub> = η; here η = η<sub>A</sub> = 1' };
    }

    function renderReadA() {
      var R = RA, ln = lawName();
      var tol = 1e-9 * Math.max(R.scale, 1e-6);
      var holds = !R.diverged && R.maxDev <= tol;
      var lawLabel = stA.conv === 'none' ? 'Φ₀ (conserved)' : ln.tex;
      bigLab.innerHTML = 'max<sub><i>t</i></sub> |Φ(<i>t</i>) − law(<i>t</i>)|, law = ' + lawLabel;
      big.innerHTML = R.diverged ? 'diverged' : R.maxDev >= 1e-3 ? numH(R.maxDev, 2) : sciH(R.maxDev, 1);
      big.style.color = R.diverged ? C('--seal') : holds ? C('--moss') : (stA.opt === 'gd' && !(stA.conv === 'coupled' && stA.lam !== 1) ? C('--ochre') : C('--seal'));
      var chips = [];
      /* coupled decay at unequal rates has no exponential law, even for the flow (Cor III.6(b)) */
      var noLaw = stA.conv === 'coupled' && stA.lam !== 1;
      if (R.diverged) chips.push(['bad', 'diverged · lower <span class="nt">η</span>']);
      else if (holds) chips.push(['ok', (stA.conv === 'none' ? 'Φ conserved · Thm III.5' : 'decay law holds · Cor III.6<span class="nt">(b)</span>')]);
      else if (stA.opt === 'adam') chips.push(['bad', 'Adam drifts · outside Thm III.5']);
      else if (noLaw) chips.push(['bad', 'no exponential law at <span class="nt">λ ≠ 1</span> · Cor III.6<span class="nt">(b)</span>']);
      else if (stA.opt === 'gd') chips.push(['mid', 'discrete steps · <span class="nt">O(η²)</span> per step']);
      else chips.push(['bad', 'law fails']);
      if (!R.diverged && stA.conv !== 'none') {
        var muRel = Math.abs(R.muT) / Math.max(1e-12, Math.abs(R.mu0), Math.abs(R.Phi0));
        var phRel = Math.abs(R.PhiT) / Math.max(1e-12, Math.abs(R.Phi0), Math.abs(R.mu0));
        if (stA.conv === 'coupled' && stA.lam !== 1 && muRel < 1e-2) chips.push(['ok', '<span class="nt">μ → 0</span> · unweighted slice']);
        if (stA.conv === 'pertime' && phRel < 1e-2) chips.push(['ok', 'Φ → 0 · rate-weighted slice']);
      }
      chipsA.innerHTML = chips.map(function (c) { return '<span class="nh-chip ' + c[0] + '">' + c[1] + '</span>'; }).join('');
      var rows = [
        ['Φ₀', numH(R.Phi0, 4), ''],
        ['Φ(<i>T</i>)', R.diverged ? '—' : numH(R.PhiT, 4), 'T = ' + num(R.tEnd, 3)],
        ['law(<i>T</i>)', numH(R.lawT, 4), ln.thm],
        ['μ(<i>T</i>)', R.diverged ? '—' : numH(R.muT, 4), 'μ₀ = ' + num(R.mu0, 3)],
        ['loss ℓ(<i>T</i>)', R.diverged ? '—' : (R.loss < 1e-300 ? '0' : sciH(R.loss, 1)), stA.opt === 'flow' ? R.steps.toLocaleString('en-US') + ' RK4 steps, h = 10⁻³' : R.steps.toLocaleString('en-US') + ' steps, η = ' + stA.eta]
      ];
      if (stA.opt === 'gd' && stA.conv === 'none' && !R.diverged) rows.push(['GD identity', sciH(R.idRes, 1), 'Φₖ₊₁ = (1 − λη²(s·eₖ)²) Φₖ, eₖ = wₖ − w★']);
      tabA.innerHTML = '<tbody>' + rows.map(function (r) { return '<tr><th scope="row">' + r[0] + '</th><td>' + r[1] + '</td><td class="l">' + r[2] + '</td></tr>'; }).join('') + '</tbody>';
      var why;
      var coupledTxt = 'Coupled decay gives dΦ/dt = −2wd·μ, with μ = b² − a². Since μ is not a function of Φ when η<sub>A</sub> ≠ η<sub>B</sub>, Φ does not follow an exponential law in general. The equilibria lie on the unweighted slice μ = 0, the balanced slice of Prop III.7(a), which differs from Φ = 0 here.';
      if (R.diverged) why = 'GD with these rates is unstable (step × curvature too large). The run is cut where |a| or |b| exceeds 10⁶.';
      else if (stA.opt === 'adam') why = 'Adam divides each coordinate by its own √v̂, so its update is not of the form −ΛDρᵀγ with a constant Λ. Thm III.5 does not apply, and the measured charge drifts. Neither decay law describes Adam training either.';
      else if (stA.opt === 'gd' && stA.conv === 'none') why = 'Each GD step multiplies Φ by exactly 1 − λη²(s·e)², where e = w − w★ is the residual. That is a relative change of order η² per step, and it shrinks |Φ| while λη²(s·e)² &lt; 2. The flow (η → 0) keeps Φ fixed.';
      else if (stA.opt === 'gd' && noLaw) why = coupledTxt + ' GD adds its own O(η²) error per step on top.';
      else if (stA.opt === 'gd') why = 'GD is the Euler discretisation of the same flow, so it follows the decay law only up to O(η²) per step.';
      else if (stA.conv === 'none') {
        why = 'The loss sees only w = w₀ + s·b·a, which is invariant under (a, b) ↦ (ta, b/t). Gradient flow therefore conserves the matching charge Φ = b²/η<sub>B</sub> − a²/η<sub>A</sub>, and the run slides along its hyperbola.';
        why += R.loss < 1e-6 ? ' Here it reaches the fibre of minimisers.'
          : stA.a0 === 0 && stA.b0 === 0 ? ' This start is the saddle (0, 0) itself, where both gradients vanish, so the run never moves.'
          : Math.abs(R.Phi0) < 1e-9 && stA.a0 * stA.b0 < 0 ? ' This start lies on the branch of Φ = 0 that never meets the fibre, so the run stalls at the saddle (0, 0).'
            : ' It has not reached the fibre by T = ' + T_PLAIN + ', because its path passes near the saddle at the origin, where the flow is slow.';
      }
      else if (stA.conv === 'pertime') why = 'Per-time decay adds −wd·(a, b) to the velocity of both factors, so dΦ/dt = −2wd·Φ exactly and Φ decays as e<sup>−2wd·t</sup> toward the rate-weighted slice Φ = 0.' + (stA.lam !== 1 ? ' The unweighted μ = b² − a² does not follow it to 0.' : '');
      else if (stA.lam === 1) why = 'At η<sub>A</sub> = η<sub>B</sub> the two conventions coincide and Φ(t) = e<sup>−2η wd·t</sup>Φ₀ (here η = 1).';
      else why = coupledTxt;
      whyA.innerHTML = '<span class="lead">why</span>' + why;
      // controls state
      etaSl.out.textContent = String(stA.eta);
      etaSl.el.classList.toggle('is-off', stA.opt === 'flow');
      etaSl.input.disabled = stA.opt === 'flow';
      wdSl.out.textContent = stA.wd.toFixed(2);
      wdSl.el.classList.toggle('is-off', stA.conv === 'none');
      wdSl.input.disabled = stA.conv === 'none';
      convHint.innerHTML = stA.conv === 'pertime' ? 'ȧ = −∂ₐℓ − wd·a,&nbsp; ḃ = −λ∂ᵦℓ − wd·b' : stA.conv === 'coupled' ? 'ȧ = −(∂ₐℓ + wd·a),&nbsp; ḃ = −λ(∂ᵦℓ + wd·b)' + (stA.opt === 'adam' ? ' · AdamW-style' : '') : 'no decay: ȧ = −∂ₐℓ,&nbsp; ḃ = −λ∂ᵦℓ';
      // meter legend
      var tide = C('--tide'), ink2 = C('--ink-2'), violet = C('--c-multiplicative');
      meterLegend.innerHTML = '<span class="nh-li">' + lineSw(tide, 2.2) + 'Φ(t) measured</span><span class="nh-li">' + lineSw(ink2, 1.3, '5 3.5') + 'law ' + (stA.conv === 'none' ? 'Φ₀' : ln.tex) + '</span><span class="nh-li">' + lineSw(violet, 1.6) + 'μ(t) = b² − a²</span>';
    }

    function frameA() {
      // bead + progress, synced to the meter playhead
      if (!geomA || !geomA.pts) return;
      var idx = beadIdx;
      if (idx < 0 || idx >= RA.n) { geomA.bead.style('display', 'none'); if (geomA.trajLen) geomA.traj.attr('stroke-dasharray', null).attr('stroke-dashoffset', null); drawMeterA(null); return; }
      geomA.bead.style('display', null).attr('cx', geomA.pts[idx][0]).attr('cy', geomA.pts[idx][1]);
      if (geomA.trajLen) {
        var frac = geomA.cum[geomA.cum.length - 1] > 0 ? geomA.cum[idx] / geomA.cum[geomA.cum.length - 1] : 1;
        var L = geomA.trajLen;
        geomA.traj.attr('stroke-dasharray', (frac * L) + ' ' + (L + 1)).attr('stroke-dashoffset', 0);
      }
      drawMeterA(idx);
    }
    var rafA = 0;
    function playA() {
      if (rafA) cancelAnimationFrame(rafA);
      if (!motion || !geomA || !geomA.cum || RA.n < 3) { beadIdx = -1; frameA(); return; }
      var total = geomA.cum[geomA.cum.length - 1];
      var dur = 1700, t0 = null;
      var tick = function (now) {
        if (t0 === null) t0 = now;
        var u = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - u, 2.2);
        var target = e * total, lo = 0, hi = geomA.cum.length - 1;
        while (lo < hi) { var mid = (lo + hi) >> 1; if (geomA.cum[mid] < target) lo = mid + 1; else hi = mid; }
        beadIdx = u >= 1 ? -1 : lo;
        frameA();
        if (u < 1) rafA = requestAnimationFrame(tick); else rafA = 0;
      };
      rafA = requestAnimationFrame(tick);
    }

    var pendA = false, pendAnim = false;
    function changedA(animate, fromDrag) {
      if (animate) pendAnim = true;
      if (pendA) return;
      pendA = true;
      var run = function () {
        if (!pendA) return;
        pendA = false;
        computeA();
        lamSegA.set(stA.lam); optSeg.set(stA.opt); convSeg.set(stA.conv);
        drawPlotA(); renderLegendA(); renderReadA();
        if (geomA && geomA.handle && refocusHandle && !dragging) { /* the handle is redrawn: keep keyboard focus on it */ refocusHandle = false; try { geomA.handle.node().focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
        if (pendAnim && !dragging) { pendAnim = false; playA(); } else { if (rafA) { cancelAnimationFrame(rafA); rafA = 0; } beadIdx = -1; frameA(); }
        live.textContent = 'Panel A: Phi0 = ' + num(RA.Phi0, 3) + ', max deviation from the law ' + sci(RA.maxDev, 1) + '.';
      };
      if (window.requestAnimationFrame) requestAnimationFrame(run);
      setTimeout(run, 40);
    }

    /* ======================================================================
       Panel B rendering
       ====================================================================== */
    var geomB = null, rafB = 0, playingB = false;
    function curF(i) { return i < 0 || i >= RB.n ? RB.n - 1 : i; }

    /* regimes of Thm III.9, made operational on the two gains of one singular direction: one-sided when one gain
       is over ten times the other (sigma < 0.70 sigma*), balanced when they are within a factor 2 (sigma > 2.83 sigma*) */
    var Q_ONE = 0.1, Q_BAL = 0.5;
    function sigOfRatio(q) { var y = q / (1 - q); return 2 * Math.sqrt(y * y + y); }   // sigma / sigma* where min(f,g)/max(f,g) = q
    function regimeOf(f, g) {
      if (!RB.iso || !(f > 0) || !(g > 0)) return '—';
      var q = Math.min(f, g) / Math.max(f, g);
      return q < Q_ONE ? 'one-sided' : q >= Q_BAL ? 'balanced' : 'crossover';
    }

    function renderControlsB() {
      var p = pB();
      slS.out.textContent = pow2s(stB.ks);
      slC.out.textContent = stB.init === 'pissa' ? 'n/a' : pow2s(stB.kc);
      slL.out.textContent = pow2s(stB.kl);
      slS.input.setAttribute('aria-valuetext', 's = ' + pow2s(stB.ks));
      slC.input.setAttribute('aria-valuetext', 'c = ' + pow2s(stB.kc));
      slL.input.setAttribute('aria-valuetext', 'lambda = ' + pow2s(stB.kl));
      slC.input.disabled = stB.init === 'pissa';
      slC.el.classList.toggle('is-off', stB.init === 'pissa');
      var lab = slC.el.querySelector('label .nm');
      if (lab) lab.textContent = stB.init === 'initb' ? 'B₀ᵀB₀ = cI' : 'A₀A₀ᵀ = cI';
      initSeg.set(stB.init);
      initHint.innerHTML = stB.init === 'lora' ? 'B₀ = 0, A₀ with orthonormal rows × √c (EVA-type): Φ₀ = −(c/η<sub>A</sub>) I, κ = c/η<sub>A</sub>'
        : stB.init === 'initb' ? 'A₀ = 0, B₀ with orthonormal columns × √c: Φ₀ = (c/η<sub>B</sub>) I, κ = −c/η<sub>B</sub>'
          : 'B₀ = U<sub>r</sub>(S<sub>r</sub>/s)<sup>½</sup>, A₀ = (S<sub>r</sub>/s)<sup>½</sup>V<sub>r</sub>ᵀ (HF PiSSA): μ₀ = 0 at every rate, Φ₀ = (S<sub>r</sub>/s)(1/η<sub>B</sub> − 1/η<sub>A</sub>), isotropic only at λ = 1';
    }

    function drawB() {
      var W = Math.max(280, chartsB.clientWidth || 520);
      var narrow = W < 460;
      var ml = 46, mr = 14;
      var tide = C('--tide'), violet = C('--c-multiplicative'), ochre = C('--ochre'), ink = C('--ink'), ink2 = C('--ink-2'), ink3 = C('--ink-3'), rule = C('--rule'), paper = C('--paper');
      var x = d3.scaleLog().domain([XLO, XHI]).range([ml, W - mr]).clamp(false);
      var R = RB, kap = R.kappa, mu = R.mu, ss = R.sstar;
      var hasStar = R.iso && kap !== 0 && ss >= XLO * 0.999 && ss <= XHI;
      var xs = hasStar ? x(ss) : (R.iso && kap !== 0 ? (ss < XLO ? ml : W - mr) : ml);
      var oneColor = kap > 0 ? tide : violet;

      /* ---- top: gains f, g ---- */
      var H1 = narrow ? 226 : 266, mt1 = 38, mb1 = 22;
      var svg1 = d3.select(gainBox).selectAll('svg').data([0]).join('svg');
      svg1.attr('viewBox', '0 0 ' + W + ' ' + H1).attr('width', W).attr('height', H1).attr('role', 'img');
      svg1.selectAll('*').remove();
      var yHi, yLo;
      if (R.iso) {
        var gHi = fgk(XHI * XHI / (mu * mu), kap);
        yHi = Math.max(gHi[0], gHi[1]) * 1.6;
        var fLo = fgk(XLO * XLO / (mu * mu), kap);
        yLo = Math.max(Math.min(fLo[0], fLo[1]), yHi * 1e-5);
      } else {
        var mx = 0, mn = Infinity;
        for (var q = 1; q < R.n; q++) { [R.F.f1[q], R.F.f2[q], R.F.g1[q], R.F.g2[q]].forEach(function (v) { if (v > 0 && isFinite(v)) { mx = Math.max(mx, v); mn = Math.min(mn, v); } }); }
        yHi = mx * 2; yLo = Math.max(mn / 2, yHi * 1e-5);
      }
      var y1 = d3.scaleLog().domain([yLo, yHi]).range([H1 - mb1, mt1]);
      var cid = uid + '-clipB1';
      svg1.append('defs').append('clipPath').attr('id', cid).append('rect').attr('x', ml).attr('y', mt1).attr('width', W - mr - ml).attr('height', H1 - mt1 - mb1);
      var g1 = svg1.append('g');
      // bands: one-sided (gains differ over 10x), crossover, balanced (within 2x)
      var clampX = function (v) { return Math.max(ml, Math.min(W - mr, v)); };
      var xOne = R.iso && kap !== 0 ? clampX(x(ss * sigOfRatio(Q_ONE))) : ml, xBal = R.iso && kap !== 0 ? clampX(x(ss * sigOfRatio(Q_BAL))) : ml;
      var bandsB = function (gg, y0, hh) {
        gg.append('rect').attr('x', ml).attr('y', y0).attr('width', Math.max(0, xOne - ml)).attr('height', hh).attr('fill', mixSoft(kap > 0 ? '--tide' : '--c-multiplicative', 15));
        gg.append('rect').attr('x', xOne).attr('y', y0).attr('width', Math.max(0, xBal - xOne)).attr('height', hh).attr('fill', mixSoft(kap > 0 ? '--tide' : '--c-multiplicative', 6));
        gg.append('rect').attr('x', xBal).attr('y', y0).attr('width', Math.max(0, W - mr - xBal)).attr('height', hh).attr('fill', C('--paper-3')).attr('fill-opacity', 0.55);
      };
      if (R.iso && kap !== 0) {
        bandsB(g1, mt1, H1 - mt1 - mb1);
      } else {
        g1.append('rect').attr('x', ml).attr('y', mt1).attr('width', W - mr - ml).attr('height', H1 - mt1 - mb1).attr('fill', C('--paper-3')).attr('fill-opacity', 0.55);
      }
      g1.append('rect').attr('x', ml).attr('y', mt1).attr('width', W - mr - ml).attr('height', H1 - mt1 - mb1).attr('fill', 'none').attr('stroke', rule);
      // grid: decades
      var dec = [];
      for (var e = -3; e <= 2; e++) dec.push(Math.pow(10, e));
      dec.forEach(function (v) {
        if (v < XLO || v > XHI) return;
        g1.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', mt1).attr('y2', H1 - mb1).attr('stroke', rule).attr('stroke-width', 0.6);
      });
      var ydec = [];
      for (e = Math.ceil(Math.log10(yLo)); e <= Math.floor(Math.log10(yHi)); e++) ydec.push(Math.pow(10, e));
      var ystep = Math.max(1, Math.ceil(ydec.length / (narrow ? 4 : 6)));
      ydec.forEach(function (v, j) {
        g1.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y1(v)).attr('y2', y1(v)).attr('stroke', rule).attr('stroke-width', 0.6);
        if (j % ystep === 0) decT(g1.append('text').attr('x', ml - 5).attr('y', y1(v) + 4).attr('text-anchor', 'end').attr('class', 'num').attr('fill', ink2).style('font-size', '11px'), v, 11);
      });
      var gc = svg1.append('g').attr('clip-path', 'url(#' + cid + ')');
      // curves
      if (R.iso) {
        var n = 240, fpts = [], gpts = [];
        for (var i = 0; i <= n; i++) {
          var sg = XLO * Math.pow(XHI / XLO, i / n), v2 = fgk(sg * sg / (mu * mu), kap);
          fpts.push([x(sg), y1(Math.max(v2[0], yLo * 1e-3))]); gpts.push([x(sg), y1(Math.max(v2[1], yLo * 1e-3))]);
        }
        // asymptotes: plateau |kappa| and the balanced line sigma/mu
        if (kap !== 0) gc.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y1(Math.abs(kap))).attr('y2', y1(Math.abs(kap))).attr('stroke', ink3).attr('stroke-width', 1).attr('stroke-dasharray', '1.5 3');
        gc.append('line').attr('x1', x(XLO)).attr('x2', x(XHI)).attr('y1', y1(XLO / mu)).attr('y2', y1(XHI / mu)).attr('stroke', ink3).attr('stroke-width', 1).attr('stroke-dasharray', '1.5 3');
        gc.append('path').attr('d', d3.line()(gpts)).attr('fill', 'none').attr('stroke', tide).attr('stroke-width', 2.4);
        gc.append('path').attr('d', d3.line()(fpts)).attr('fill', 'none').attr('stroke', violet).attr('stroke-width', 2.4).attr('stroke-dasharray', kap === 0 ? '7 5' : null);
      }
      // sigma* line
      if (hasStar) {
        svg1.append('line').attr('x1', xs).attr('x2', xs).attr('y1', mt1 - 4).attr('y2', H1 - mb1).attr('stroke', ochre).attr('stroke-width', 1.6).attr('stroke-dasharray', '5 3');
        halo(svg1.append('text').attr('x', xs).attr('y', mt1 - 9).attr('text-anchor', xs > W - mr - 60 ? 'end' : xs < ml + 60 ? 'start' : 'middle').attr('fill', C('--ochre-ink')).style('font-size', '12px').style('font-weight', 500).style('font-variant-numeric', 'tabular-nums').text('σ* = ' + num(ss, 3)));
      } else if (R.iso && kap !== 0) {
        halo(svg1.append('text').attr('x', ss < XLO ? ml + 4 : W - mr - 4).attr('y', mt1 - 9).attr('text-anchor', ss < XLO ? 'start' : 'end').attr('fill', C('--ochre-ink')).style('font-size', '12px').style('font-weight', 500).style('font-variant-numeric', 'tabular-nums').text((ss < XLO ? '← ' : '') + 'σ* = ' + num(ss, 3) + (ss < XLO ? '' : ' →')));
      }
      // band labels
      var bandY = mt1 + 15, bandYb = H1 - mb1 - 7;   // one-sided label top-left, balanced label bottom-right: both clear of the curves
      /* a band label takes the longest wording that fits its band, measured in the page font */
      var bandLabel = function (xx, yy, anchor, fill, texts, maxW) {
        var t = g1.append('text').attr('x', xx).attr('y', yy).attr('text-anchor', anchor).attr('fill', fill).style('font-size', '12px').style('font-weight', 500).style('letter-spacing', '.02em');
        for (var k = 0; k < texts.length; k++) {
          t.text(texts[k]);
          var tw = 0; try { tw = t.node().getComputedTextLength(); } catch (er) { tw = texts[k].length * 7; }
          if (tw <= maxW) { halo(t, 3); return; }
        }
        t.remove();
      };
      if (R.iso && kap !== 0) {
        var leftW = xOne - ml, rightW = W - mr - xBal;
        bandLabel(ml + 6, bandY, 'start', oneColor, [kap > 0 ? 'one-sided · LoRA-FA' : 'one-sided · Init[B]', 'one-sided'], leftW - 12);
        bandLabel(W - mr - 6, bandYb, 'end', ink2, ['balanced · Arora–Cohen–Hazan', 'balanced'], rightW - 12);
      } else if (R.iso) {
        bandLabel(W - mr - 6, bandYb, 'end', ink2, ['κ = 0: balanced at every scale', 'κ = 0: balanced'], W - mr - ml - 12);
      } else {
        bandLabel(W - mr - 6, bandY, 'end', C('--seal-ink'), ['Φ ≠ −κI: no closed form'], W - mr - ml - 12);
      }
      // curve labels: parts starting with '_' are subscripts (dy shifts, so they sit at a true subscript height)
      var subT = function (sel, parts, fs) {
        var shifted = false;
        parts.forEach(function (pt) {
          if (pt.charAt(0) === '_') { sel.append('tspan').attr('dy', fs * 0.28).style('font-size', (fs * 0.76) + 'px').text(pt.slice(1)); shifted = true; }
          else { var tt = sel.append('tspan').text(pt); if (shifted) { tt.attr('dy', -fs * 0.28); shifted = false; } }
        });
        return sel;
      };
      var avoid = [];
      var boxOf = function (sel) { try { var bb = sel.node().getBBox(); return { x0: bb.x - 3, y0: bb.y - 2, x1: bb.x + bb.width + 3, y1: bb.y + bb.height + 2 }; } catch (er) { return null; } };
      if (R.iso) {
        if (kap === 0) {
          var lx0 = 0.012, t0 = gc.append('text').attr('x', x(lx0)).attr('y', y1(lx0 / mu) - 12).attr('fill', ink).style('font-size', '12px').style('font-weight', 500);
          subT(t0, ['f', '_κ', ' = g', '_κ', ' = σ/μ  (κ = 0)'], 12);
          halo(t0, 3.5); avoid.push(boxOf(t0));
        } else {
          var plG = kap > 0, ak = Math.abs(kap);
          var yPl = y1(ak) - 8 - 12 < bandY + 4 ? y1(ak) + 17 : y1(ak) - 8;   // below the plateau if the band label is in the way
          var t1 = gc.append('text').attr('x', ml + 7).attr('y', yPl).attr('fill', plG ? tide : violet).style('font-size', '12px').style('font-weight', 500);
          subT(t1, [plG ? 'g' : 'f', '_κ', ' → |κ|' + (xs - ml >= 205 ? ' · ' + (plG ? 'B-update' : 'A-update') : '')], 12);
          halo(t1, 3.5); avoid.push(boxOf(t1));
          var tgt = Math.sqrt(yLo * ak), xx = tgt * tgt + tgt * ak, sgl = mu * Math.sqrt(xx), px = x(sgl), py = y1(tgt);
          var t2 = gc.append('text').attr('y', py + 4).attr('fill', plG ? violet : tide).style('font-size', '12px').style('font-weight', 500);
          if (px - ml > 180) t2.attr('x', px - 10).attr('text-anchor', 'end'); else t2.attr('x', px + 10);
          subT(t2, [plG ? 'f' : 'g', '_κ', ' · ' + (plG ? 'A-update' : 'B-update')], 12);
          halo(t2, 3.5); avoid.push(boxOf(t2));
        }
      }
      var cap = svg1.append('text').attr('x', ml).attr('y', 12).attr('fill', ink2).style('font-size', '12px').style('font-weight', 500);
      subT(cap, ['f', '_κ', ', g', '_κ', '(σ²/μ²) on each singular direction (log)'], 12);
      try { if (cap.node().getComputedTextLength() > W - ml - 4) { cap.selectAll('*').remove(); subT(cap, ['f', '_κ', ', g', '_κ', '(σ²/μ²) per direction (log)'], 12); } } catch (er) { /* not rendered */ }
      // history (measured on the integrated flow) and current dots
      var hist = gc.append('g'), cur = svg1.append('g').attr('clip-path', 'url(#' + cid + ')');
      var every = Math.max(1, Math.round(R.n / 70));
      for (var fI = 1; fI < R.n; fI += every) {
        [[R.F.s1[fI], R.F.f1[fI], violet], [R.F.s2[fI], R.F.f2[fI], violet], [R.F.s1[fI], R.F.g1[fI], tide], [R.F.s2[fI], R.F.g2[fI], tide]].forEach(function (d) {
          if (!(d[0] > 0) || !(d[1] > 0)) return;
          hist.append('circle').attr('cx', x(d[0])).attr('cy', y1(d[1])).attr('r', 1.7).attr('fill', 'none').attr('stroke', d[2]).attr('stroke-opacity', 0.55).attr('stroke-width', 0.9);
        });
      }
      svg1.attr('aria-label', 'Gains f and g of Thm III.9 against the singular value sigma on log axes' + (hasStar ? ', crossover at sigma star = ' + num(ss, 3) : '') + '. Hollow circles are measured on the integrated flow.');

      /* ---- bottom: sigma_i(t), time running down ---- */
      var H2 = narrow ? 190 : 210, mt2 = 6, mb2 = 34;
      var svg2 = d3.select(pathBox).selectAll('svg').data([0]).join('svg');
      svg2.attr('viewBox', '0 0 ' + W + ' ' + H2).attr('width', W).attr('height', H2).attr('role', 'img')
        .attr('aria-label', 'Singular values of Delta W along the integrated flow; time runs downward on a log scale.');
      svg2.selectAll('*').remove();
      var tLo = R.F.t[1] || 1e-3, tHi = R.T;
      var y2 = d3.scaleLog().domain([tLo, tHi]).range([mt2, H2 - mb2]);
      var g2 = svg2.append('g');
      var cid2 = uid + '-clipB2';
      svg2.append('defs').append('clipPath').attr('id', cid2).append('rect').attr('x', ml).attr('y', mt2).attr('width', W - mr - ml).attr('height', H2 - mt2 - mb2);
      if (R.iso && kap !== 0) bandsB(g2, mt2, H2 - mt2 - mb2);
      else g2.append('rect').attr('x', ml).attr('y', mt2).attr('width', W - mr - ml).attr('height', H2 - mt2 - mb2).attr('fill', C('--paper-3')).attr('fill-opacity', 0.55);
      g2.append('rect').attr('x', ml).attr('y', mt2).attr('width', W - mr - ml).attr('height', H2 - mt2 - mb2).attr('fill', 'none').attr('stroke', rule);
      dec.forEach(function (v) {
        if (v < XLO || v > XHI) return;
        g2.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', mt2).attr('y2', H2 - mb2).attr('stroke', rule).attr('stroke-width', 0.6);
        decT(g2.append('text').attr('x', x(v)).attr('y', H2 - mb2 + 14).attr('text-anchor', 'middle').attr('class', 'num').attr('fill', ink2).style('font-size', '11px'), v, 11);
      });
      var tdec = [];
      for (e = Math.ceil(Math.log10(tLo)); e <= Math.floor(Math.log10(tHi)); e++) tdec.push(Math.pow(10, e));
      tdec.forEach(function (v) {
        g2.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y2(v)).attr('y2', y2(v)).attr('stroke', rule).attr('stroke-width', 0.6);
        decT(g2.append('text').attr('x', ml - 5).attr('y', y2(v) + 4).attr('text-anchor', 'end').attr('class', 'num').attr('fill', ink2).style('font-size', '11px'), v, 11);
      });
      g2.append('text').attr('x', W - mr).attr('y', H2 - 3).attr('text-anchor', 'end').attr('fill', ink2).style('font-size', '12px').style('font-weight', 500).text('singular value σ of ΔW (log) →');
      g2.append('text').attr('x', ml).attr('y', H2 - 3).attr('fill', ink2).style('font-size', '12px').style('font-weight', 500).text('t ↓ (log)');
      if (hasStar) svg2.append('line').attr('x1', xs).attr('x2', xs).attr('y1', mt2).attr('y2', H2 - mb2).attr('stroke', ochre).attr('stroke-width', 1.6).attr('stroke-dasharray', '5 3');
      var gp = svg2.append('g').attr('clip-path', 'url(#' + cid2 + ')');
      ['s1', 's2'].forEach(function (key, kk) {
        var pts = [];
        for (var j = 1; j < R.n; j++) { var sv = R.F[key][j]; if (sv > 0) pts.push([x(Math.max(sv, XLO * 0.5)), y2(R.F.t[j])]); }
        gp.append('path').attr('d', d3.line()(pts)).attr('fill', 'none').attr('stroke', kk === 0 ? ink : ink2).attr('stroke-width', 1.8).attr('stroke-dasharray', kk === 0 ? null : '6 3');
      });
      // final-value labels
      [['σ₁', R.s1T], ['σ₂', R.s2T]].forEach(function (d) {
        if (d[1] > XLO && d[1] < XHI) halo(svg2.append('text').attr('x', x(d[1]) + 5).attr('y', H2 - mb2 - 5).attr('fill', ink).style('font-size', '12px').text(d[0]), 3);
      });
      geomB = { avoid: avoid.filter(Boolean), x: x, y1: y1, y2: y2, W: W, ml: ml, mr: mr, cur: cur, svg2: svg2, H2: H2, mt2: mt2, mb2: mb2, tide: tide, violet: violet, ink: ink, ink2: ink2, paper: paper, ochre: ochre };
      drawFrameB();
    }
    /* a decade tick as SVG text; large and small decades get a real superscript (Plex Mono has none) */
    function decT(sel, v, fs) {
      var e = Math.round(Math.log10(v));
      if (e >= -2 && e <= 3) return sel.text(decLabel(v));
      sel.text('10'); sel.append('tspan').attr('dy', -0.42 * fs).style('font-size', (0.74 * fs).toFixed(1) + 'px').text(minus(String(e)));
      return sel;
    }
    function decLabel(v) {
      var e = Math.round(Math.log10(v));
      if (e >= 0 && e <= 3) return String(Math.pow(10, e));
      if (e === -1) return '0.1';
      if (e === -2) return '0.01';
      return '10' + sup(e);
    }
    function drawFrameB() {
      if (!geomB) return;
      var R = RB, i = curF(stB.frame), G = geomB;
      G.cur.selectAll('*').remove();
      G.svg2.selectAll('.nh-ph').remove();
      var dots = [[R.F.s1[i], R.F.g1[i], G.tide, 'σ₁'], [R.F.s2[i], R.F.g2[i], G.tide, 'σ₂'], [R.F.s1[i], R.F.f1[i], G.violet, ''], [R.F.s2[i], R.F.f2[i], G.violet, '']];
      var placed = [];
      /* draw all dots first, labels after, so a label never hides under a later dot */
      dots.slice().reverse().forEach(function (d) {
        if (!(d[0] > 0) || !(d[1] > 0)) return;
        placed.push({ x0: G.x(d[0]) - 6, y0: G.y1(d[1]) - 6, x1: G.x(d[0]) + 6, y1: G.y1(d[1]) + 6 });
        G.cur.append('circle').attr('cx', G.x(d[0])).attr('cy', G.y1(d[1])).attr('r', 5.2).attr('fill', d[2]).attr('stroke', G.paper).attr('stroke-width', 1.6);
      });
      dots.forEach(function (d) {
        if (!(d[0] > 0) || !(d[1] > 0)) return;
        var cx = G.x(d[0]), cy = G.y1(d[1]);
        if (!d[3]) return;
        /* first free spot among above-right, below-right, above-left (labels are ~16 x 12 px) */
        var spots = [[cx + 7, cy - 7, 'start'], [cx + 7, cy + 15, 'start'], [cx - 7, cy - 7, 'end'], [cx - 7, cy + 15, 'end']], pick = spots[0];
        for (var q = 0; q < spots.length; q++) {
          var sp = spots[q], bx0 = sp[2] === 'start' ? sp[0] : sp[0] - 19, r = { x0: bx0, y0: sp[1] - 11, x1: bx0 + 19, y1: sp[1] + 3 };
          var hit = (G.avoid || []).concat(placed).some(function (o) { return r.x0 < o.x1 && r.x1 > o.x0 && r.y0 < o.y1 && r.y1 > o.y0; });
          if (!hit) { pick = sp; placed.push(r); break; }
        }
        halo(G.cur.append('text').attr('x', pick[0]).attr('y', pick[1]).attr('text-anchor', pick[2]).attr('fill', G.ink).style('font-size', '12px').text(d[3]), 3);
      });
      if (i > 0) {
        var yy = G.y2(R.F.t[i]);
        G.svg2.append('line').attr('class', 'nh-ph').attr('x1', G.ml).attr('x2', G.W - G.mr).attr('y1', yy).attr('y2', yy).attr('stroke', G.ochre).attr('stroke-width', 1).attr('stroke-opacity', 0.9);
        [R.F.s1[i], R.F.s2[i]].forEach(function (sv, kk) {
          if (!(sv > XLO)) return;
          G.svg2.append('circle').attr('class', 'nh-ph').attr('cx', G.x(sv)).attr('cy', yy).attr('r', 4).attr('fill', kk === 0 ? G.ink : G.ink2).attr('stroke', G.paper).attr('stroke-width', 1.4);
        });
      }
      timeIn.value = i;
      timeOut.textContent = 't = ' + num(R.F.t[i], 3);
      timeIn.setAttribute('aria-valuetext', 't = ' + num(R.F.t[i], 3) + ', sigma1 = ' + num(R.F.s1[i], 3) + ', sigma2 = ' + num(R.F.s2[i], 3));
      renderReadB(i);
    }

    function renderReadB(i) {
      var R = RB, kap = R.kappa;
      var html = '<span class="nh-tag">Computed from <span class="nt">s, c, λ (η<sub>A</sub> = 1, η<sub>B</sub> = λ)</span></span>';
      var rows = [];
      if (R.iso) {
        rows.push(['κ', numH(kap, 4), stB.init === 'lora' ? 'c/η<sub>A</sub>' : stB.init === 'initb' ? '−c/η<sub>B</sub>' : 'Φ₀ = 0 at η<sub>A</sub> = η<sub>B</sub>']);
        rows.push(['μ', numH(R.mu, 4), 's√(η<sub>A</sub>η<sub>B</sub>)']);
        rows.push(['σ*', kap === 0 ? '0' : numH(R.sstar, 4), 'μ|κ|/2' + (stB.init === 'lora' ? ' = (sc/2)√λ' : stB.init === 'initb' ? ' = sc/(2√λ)' : '')]);
      } else {
        rows.push(['Φ₀', 'diag(' + numH(R.phi0[0], 3) + ', ' + numH(R.phi0[1], 3) + ')', 'not a multiple of I']);
        rows.push(['μ', numH(R.mu, 4), 's√(η<sub>A</sub>η<sub>B</sub>)']);
      }
      html += '<table class="nh-tab"><tbody>' + rows.map(function (r) { return '<tr><th scope="row">' + r[0] + '</th><td>' + r[1] + '</td><td class="l">' + r[2] + '</td></tr>'; }).join('') + '</tbody></table>';
      // per-direction table at the current time
      var sh = function (f, g) { return f > 0 && g > 0 ? fx(f / (f + g), 3) : '—'; };
      var dirRow = function (name, sv, f, g) {
        var reg = regimeOf(f, g), cls = reg === 'one-sided' ? 'mid' : '';
        var ratio = R.iso && kap !== 0 && sv > 0 ? (sv / R.sstar >= 99.5 ? fx(sv / R.sstar, 0) : sv / R.sstar >= 0.01 ? fx(sv / R.sstar, 2) : numH(sv / R.sstar, 2)) + 'σ*' : '—';
        return '<tr><th scope="row">' + name + '</th><td>' + (sv > 0 ? numH(sv, 4) : '0') + '</td><td>' + ratio + '</td><td>' + sh(f, g) + '</td><td class="l ' + cls + '">' + reg + '</td></tr>';
      };
      html += '<hr class="nh-sep"><span class="nh-tag">At <span class="nt">t = ' + num(R.F.t[i], 3) + '</span> · measured on the factors</span>';
      html += '<table class="nh-tab"><thead><tr><th></th><th>σ</th><th>σ/σ*</th><th title="share of the growth of sigma_i contributed by the A-update, f/(f+g)">A-share</th><th style="text-align:left">regime</th></tr></thead><tbody>' +
        dirRow('σ₁', R.F.s1[i], R.F.f1[i], R.F.g1[i]) + dirRow('σ₂', R.F.s2[i], R.F.f2[i], R.F.g2[i]) + '</tbody></table>';
      var atStar = !R.iso ? '' : kap > 0 ? ' At σ = σ* it is (√2−1)/(2√2) ≈ 0.146, and g is about 5.8 times f.' : kap < 0 ? ' At σ = σ* it is (√2+1)/(2√2) ≈ 0.854, and f is about 5.8 times g.' : ' At κ = 0, f = g, so it is 1/2 at every scale.';
      var regTxt = R.iso ? ' Regime: one-sided when one gain is over 10 times the other (σ ≲ 0.70σ*), balanced when they are within a factor 2 (σ ≳ 2.83σ*), crossover in between.' : '';
      html += '<div class="nh-hint">A-share = f/(f+g), the part of dσᵢ/dt = −μ²(fᵢ+gᵢ)·uᵢᵀGvᵢ that comes from the A-update. f and g are read off the factors as uᵢᵀ(BBᵀ/η<sub>B</sub>)uᵢ and vᵢᵀ(AᵀA/η<sub>A</sub>)vᵢ.' + atStar + regTxt + '</div>';
      // residual meter
      html += '<hr class="nh-sep"><span class="nh-tag">Closed form vs integrated flow · Thm III.9</span>';
      if (R.iso && R.nRes > 0) {
        var good = R.maxRes < 1e-8;
        html += '<div class="nh-biglab">max<sub><i>t</i></sub> ‖<i>Ẇ</i><sub>closed</sub> − <i>Ẇ</i><sub>flow</sub>‖ / ‖<i>Ẇ</i><sub>flow</sub>‖</div>';
        html += '<div class="nh-big" style="color:' + (good ? C('--moss') : C('--ochre')) + '">' + (R.maxRes >= 1e-3 ? numH(R.maxRes, 2) : sciH(R.maxRes, 1)) + '</div>';
        html += '<div class="nh-chips"><span class="nh-chip ' + (good ? 'ok' : 'mid') + '">' + (good ? 'closed form holds · round-off level' : 'above round-off · check the run') + '</span></div>';
        var stopTxt = R.stop === 'rest' ? 'at rest' : R.stop === 'tcap' ? 'time cap' : R.stop === 'cap' ? 'step budget reached' : 'stopped';
        html += '<table class="nh-tab"><tbody>' +
          '<tr><th scope="row">charge</th><td>' + sciH(R.maxDrift, 1) + '</td><td class="l">max<sub>t</sub> ‖Φ(t) + κI‖<sub>F</sub>; the midpoint rule keeps Φ only because dΦ·f = 0 (Thm III.5)</td></tr>' +
          '<tr><th scope="row">run</th><td>' + R.steps.toLocaleString('en-US') + '</td><td class="l">implicit-midpoint steps to t = ' + num(R.T, 3) + ' (' + stopTxt + ') · ' + R.nRes + ' checkpoints while ‖Ẇ‖ ≥ 10⁻⁶ max‖Ẇ‖</td></tr></tbody></table>';
      } else if (R.iso) {
        html += '<div class="nh-hint">No checkpoint on the rank-2 locus yet.</div>';
      } else {
        html += '<div class="nh-warn">The charge Φ₀ = diag(' + numH(R.phi0[0], 3) + ', ' + numH(R.phi0[1], 3) + ') is not a multiple of I, so Thm III.9 does not apply. PiSSA starts with μ₀ = B₀ᵀB₀ − A₀A₀ᵀ = 0 at every rate, but its charge Φ₀ is isotropic only at equal rates (Cor III.6(a)). The flow is still integrated, and its measured gains no longer lie on one pair of curves.</div>';
      }
      readB.innerHTML = html;
    }

    function renderLegendB() {
      var tide = C('--tide'), violet = C('--c-multiplicative'), ochre = C('--ochre'), ink = C('--ink'), ink2 = C('--ink-2'), ink3 = C('--ink-3');
      var iso = RB.iso, kap = RB.kappa, it = [];
      if (iso) {
        it.push(lineSw(tide, 2.4) + '<span><i>g</i><sub>κ</sub>(σ²/μ²) · B-update</span>');
        it.push(lineSw(violet, 2.4, kap === 0 ? '7 5' : '') + '<span><i>f</i><sub>κ</sub>(σ²/μ²) · A-update</span>');
      }
      it.push(ringSw(tide) + ringSw(violet) + '<span>' + (iso ? 'g, f measured on the flow' : 'g, f measured on the flow (no closed form)') + '</span>');
      it.push(dotSw(tide) + dotSw(violet) + '<span>at the time t shown</span>');
      if (iso && kap !== 0) it.push(lineSw(ochre, 1.6, '5 3') + '<span>σ*</span>');
      if (iso) it.push(lineSw(ink3, 1, '1.5 3') + '<span>' + (kap !== 0 ? 'asymptotes |κ| and σ/μ' : 'asymptote σ/μ') + '</span>');
      it.push(lineSw(ink, 1.8) + lineSw(ink2, 1.8, '6 3') + '<span>σ₁(t), σ₂(t)</span>');
      if (iso && kap !== 0) {
        var tok = kap > 0 ? '--tide' : '--c-multiplicative';
        it.push(boxSw(mixSoft(tok, 30)) + '<span>one-sided</span>' + boxSw(mixSoft(tok, 13)) + '<span>crossover</span>' + boxSw(C('--paper-3')) + '<span>balanced</span>');
      }
      legendB.innerHTML = it.map(function (s) { return '<span class="nh-li">' + s + '</span>'; }).join('');
    }


    function stopB() { playingB = false; if (rafB) cancelAnimationFrame(rafB); rafB = 0; playBtn.textContent = '▶ Play'; playBtn.setAttribute('aria-label', 'Play the flow'); }
    function playB() {
      if (!motion) { stB.frame = -1; drawFrameB(); return; }
      stopB();
      playingB = true; playBtn.textContent = '❚❚ Pause'; playBtn.setAttribute('aria-label', 'Pause');
      var n = RB.n, start = curF(stB.frame) >= n - 1 ? 1 : curF(stB.frame), t0 = null, dur = 5200 * (n - 1 - start) / Math.max(1, n - 2);
      var tick = function (now) {
        if (t0 === null) t0 = now;
        var u = dur > 0 ? Math.min(1, (now - t0) / dur) : 1;
        stB.frame = Math.round(start + u * (n - 1 - start));
        if (u >= 1) stB.frame = -1;
        drawFrameB();
        if (u < 1 && playingB) rafB = requestAnimationFrame(tick); else stopB();
      };
      rafB = requestAnimationFrame(tick);
    }
    playBtn.addEventListener('click', function () { if (playingB) stopB(); else playB(); });
    timeIn.addEventListener('input', function () { stopB(); stB.frame = +timeIn.value; drawFrameB(); });

    var pendB = false, timerB = 0, autoplayB = false;
    function changedB(animate) {
      renderControlsB();
      if (animate) autoplayB = true;
      clearTimeout(timerB);
      timerB = setTimeout(function () {
        stopB();
        RB = runB();
        timeIn.max = RB.n - 1;
        stB.frame = -1;
        renderLegendB();
        drawB();
        renderFoot();
        live.textContent = 'Panel B: sigma star = ' + (RB.iso ? num(RB.sstar, 3) : 'undefined') + ', final singular values ' + num(RB.s1T, 3) + ' and ' + num(RB.s2T, 3) + '.';
        if (autoplayB) { autoplayB = false; playB(); }
      }, 70);
    }

    function renderFoot() {
      foot.innerHTML = 'A: w₀ = 0, w★ = 1, s = 1, η<sub>A</sub> = 1; flow by RK4 with h = 10⁻³; GD and Adam (β = 0.9, 0.999; ε = 10⁻⁸; bias-corrected) at learning rates η·η<sub>A</sub>, η·η<sub>B</sub>, time t = steps × η; horizon T = ' + T_PLAIN + ' without decay, ' + T_DECAY + ' with. ' +
        'B: W = W<sub>res</sub> + sBA, ℓ = ½‖W − W★‖²<sub>F</sub>, ΔW★ of rank 2 with σ = (' + ST.join(', ') + '), seed ' + SEED_B + '; PiSSA splits a W₀ with top singular values (' + SR.join(', ') + '), so its adapter targets ΔW★ + U<sub>r</sub>S<sub>r</sub>V<sub>r</sub>ᵀ (rank 4, not reachable at r = 2). ' +
        'Flow integrated by the implicit midpoint rule (Newton to round-off, ≤ 1% relative change per step), stopped when ‖ẏ‖·t ≤ 10⁻⁸‖y‖. ' +
        'Charges after Zhao et al. 2023; closed form and crossover after Tarmoun et al. 2021 and Min et al. 2021; balanced regime after Arora, Cohen and Hazan 2018. Both panels test gradient-flow statements. Under SGD the switch at σ* is a prediction (X.7), and Adam is not covered.';
    }

    /* ---------- resize + theme ---------- */
    var lastW = 0;
    function redrawAll() {
      if (RA) { drawPlotA(); renderLegendA(); renderReadA(); beadIdx = -1; frameA(); }
      if (RB) { renderLegendB(); drawB(); }
    }
    if ('ResizeObserver' in window) {
      new ResizeObserver(A.debounce(function () {
        var w = stage.clientWidth;
        if (Math.abs(w - lastW) < 2) return;
        lastW = w; redrawAll();
      }, 80)).observe(stage);
    }
    A.onTheme(function () { A._cssCache = {}; redrawAll(); });

    /* ---------- test hook (read-only numerics) ---------- */
    el.__noether = { simScalar: simScalar, simMatrix: function (p) { return simMatrix(D, p); }, D: D, state: function () { return { A: stA, B: stB, RA: RA, RB: RB }; } };

    /* ---------- first render: complete at rest, then play once ---------- */
    computeA();
    drawPlotA(); renderLegendA(); renderReadA(); frameA();
    renderControlsB();
    RB = runB();
    timeIn.max = RB.n - 1;
    renderLegendB(); drawB(); renderFoot();
    lastW = stage.clientWidth;
  });
})();
