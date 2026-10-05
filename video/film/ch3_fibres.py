import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *
# Chapter III · Fibres — how a method learns (beats III.0 to III.4 of video/script/SCRIPT.md).
#
# Every number on screen is read from a simulation output:
#   video/sim/out/fibres_export.npz   video/sim/fibres_export.py (the setup of video/sim/fibres.py): III.1 to III.3
#   video/sim/out/lora_plus.npz       video/sim/lora_plus.py: III.4
#
#   PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -qh --fps 30 --disable_caching \
#       --media_dir video/build/Fibres video/film/ch3_fibres.py Fibres

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim" / "out"
FX = np.load(SIM / "fibres_export.npz")
LP = np.load(SIM / "lora_plus.npz")

CM_PER_UNIT_30 = 1 / 0.886          # a LaTeX \parbox of 1 cm is 0.886 frame units wide at font_size 30


# ---------------------------------------------------------------- numbers
def sci_parts(x, digits=2, force=None):
    """('6.7', '−16') for 6.66e-16; ('0.35', None) for 0.345; ('0.0', None) for an exact zero."""
    x = float(x)
    if x == 0:
        return "0.0", None
    e = int(np.floor(np.log10(abs(x))))
    mant = round(x / 10 ** e, digits - 1)
    if abs(mant) >= 10:
        e += 1
        mant = round(x / 10 ** e, digits - 1)
    sci = force if force is not None else not (-2 <= e <= 2)
    if not sci:
        return f"{x:.{max(digits - 1 - e, 0)}f}".replace("-", "−"), None
    return f"{mant:.{digits - 1}f}".replace("-", "−"), str(e).replace("-", "−")


def value_text(label, x, size=22, color=INK, digits=2, force=None, label_color=INK3):
    """A monospace readout 'label 6.7 × 10⁻¹⁶' with a raised exponent. Element 0 is the main line."""
    mant, ex = sci_parts(x, digits, force)
    body = (f"{label} " if label else "") + mant + (" × 10" if ex else "")
    t = Text(body, font=MONO, font_size=size, color=color, t2c={label: label_color} if label else {})
    g = VGroup(t)
    if ex:
        last = t[-1]
        sup = Text(ex, font=MONO, font_size=size * 0.64, color=color)
        sup.next_to(last, RIGHT, buff=0.035 * size / 22).align_to(last, DOWN).shift(last.height * 0.55 * UP)
        g.add(sup)
    return g


def place(g, anchor, edge=LEFT):
    """Shift g so that the `edge` point of its main line (element 0) sits at `anchor`."""
    return g.shift(np.array(anchor, dtype=float) - g[0].get_critical_point(edge))


def live(make, anchor, edge=LEFT):
    """A readout rebuilt every frame by make() (a time-based updater, so waits keep rendering)."""
    m = place(make(), anchor, edge)
    m.add_updater(lambda mob, dt: mob.become(place(make(), anchor, edge)))
    return m


# ---------------------------------------------------------------- cards and pieces
def card(label, body, credit=None, width=8.0, body_size=28, credit_size=20):
    """A paper card (the look of look.paper_card) whose credit wraps to the same width as its body."""
    lab = Text(label.upper(), font=SANS, weight=MEDIUM, font_size=15, color=PTIDE)
    txt = Tex(r"\parbox{%.2fcm}{\raggedright %s}" % (width * CM_PER_UNIT_30 * 30 / body_size, body),
              tex_template=TEX, font_size=body_size, color=PINK)
    parts = [lab, txt]
    if credit:
        parts.append(Tex(r"\parbox{%.2fcm}{\raggedright\itshape %s}" % (width * CM_PER_UNIT_30 * 30 / credit_size,
                                                                       credit),
                         tex_template=TEX, font_size=credit_size, color=PINK2))
    col = VGroup(*parts).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
    box = RoundedRectangle(corner_radius=0.10, width=col.width + 0.7, height=col.height + 0.6, fill_color=PAPER,
                           fill_opacity=1, stroke_color=PRULE, stroke_width=1)
    shadow = box.copy().set_fill(BLACK, 0.35).set_stroke(width=0).shift(0.06 * RIGHT + 0.08 * DOWN)
    box.move_to(col)
    return VGroup(shadow, box, col)


def tile(M, cell, vmax, pos=TIDE, neg=CORAL, fmt="{:.2f}", size=15):
    """A small heat map with its entries printed in the cells (dark ink on bright cells)."""
    M = np.asarray(M, dtype=float)
    cells = heatmap(M, cell, vmax, pos, neg)
    txt = VGroup(*[mono(fmt.format(v).replace("-", "−"), size, BG if abs(v) / vmax > 0.6 else INK)
                   .move_to(sq) for sq, v in zip(cells, M.ravel())])
    return VGroup(cells, txt)


def frame_box(w, h, center, fill=BG2):
    return RoundedRectangle(corner_radius=0.08, width=w, height=h, fill_color=fill, fill_opacity=1,
                            stroke_color=RULE, stroke_width=1).move_to(center)


def path_fitter(paths, box, margin=0.22):
    """A map from 2-D data points to the frame that fits all `paths` into `box`, one scale for both axes."""
    P = np.concatenate(paths)
    lo, hi = P.min(0), P.max(0)
    span = np.maximum(hi - lo, 1e-9)
    s = min((box.width - 2 * margin) / span[0], (box.height - 2 * margin) / span[1])
    mid, c = (lo + hi) / 2, box.get_center()
    return lambda Q: np.column_stack([c[0] + s * (np.asarray(Q)[:, 0] - mid[0]),
                                      c[1] + s * (np.asarray(Q)[:, 1] - mid[1]), np.zeros(len(Q))])


def polyline(pts, color, width, opacity=1.0):
    return VMobject().set_points_as_corners(pts).set_stroke(color, width, opacity)


def dashed(vm, dash=0.11):
    return DashedVMobject(vm, num_dashes=max(4, int(vm.get_arc_length() / dash)), dashed_ratio=0.55)


def scope_line(items, size=17, color=INK2):
    """'Adam · ε = 0 · no decay · …' as one line of sans text."""
    return Text("  ·  ".join(items), font=SANS, font_size=size, color=color)


# ---------------------------------------------------------------- a schematic torus drawn in 2-D
class Torus2D:
    """A torus (major radius R, tube radius a) tilted towards the viewer and drawn flat: shaded facets in
    painter's order, a hidden-line test by ray marching, and curves on the surface split into seen and hidden parts."""

    def __init__(self, R=1.55, a=0.62, tilt=60 * DEGREES, center=ORIGIN):
        self.R, self.a, self.t, self.c = R, a, tilt, np.array(center, dtype=float)

    def xyz(self, u, v):
        R, a, t = self.R, self.a, self.t
        X = (R + a * np.cos(v)) * np.cos(u)
        Y = (R + a * np.cos(v)) * np.sin(u)
        Z = a * np.sin(v)
        return np.stack([X, Y * np.cos(t) + Z * np.sin(t), -Y * np.sin(t) + Z * np.cos(t)], -1)

    def normal(self, u, v):
        t = self.t
        nx, ny, nz = np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)
        return np.stack([nx, ny * np.cos(t) + nz * np.sin(t), -ny * np.sin(t) + nz * np.cos(t)], -1)

    def pt(self, u, v):
        p = self.xyz(u, v)
        return np.array([p[0], p[1], 0.0]) + self.c

    def front_z(self, x, y):
        """Depth of the surface nearest the viewer along the ray through screen point (x, y); -inf if none."""
        R, a, t = self.R, self.a, self.t
        zs = np.linspace(-(R + a) - 0.1, R + a + 0.1, 1400)
        out = np.empty(len(x))
        for k in range(0, len(x), 1500):
            X = x[k:k + 1500, None]
            yy = y[k:k + 1500, None]
            Yl = yy * np.cos(t) - zs[None] * np.sin(t)
            Zl = yy * np.sin(t) + zs[None] * np.cos(t)
            inside = (np.sqrt(X ** 2 + Yl ** 2) - R) ** 2 + Zl ** 2 <= a * a
            idx = np.where(inside.any(1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], 1), -1)
            out[k:k + 1500] = np.where(idx >= 0, zs[np.maximum(idx, 0)], -np.inf)
        return out

    def visible(self, P):
        return P[:, 2] >= self.front_z(P[:, 0], P[:, 1]) - 0.03

    def body(self, nu=72, nv=30, lo="#0E1824", hi="#2D4058"):
        us, vs = np.linspace(0, TAU, nu + 1), np.linspace(0, TAU, nv + 1)
        L = np.array([-0.45, 0.55, 0.70])
        L /= np.linalg.norm(L)
        quads = []
        for i in range(nu):
            for j in range(nv):
                N = self.normal((us[i] + us[i + 1]) / 2, (vs[j] + vs[j + 1]) / 2)
                if N[2] <= 0:
                    continue
                cs = [self.xyz(us[i], vs[j]), self.xyz(us[i + 1], vs[j]), self.xyz(us[i + 1], vs[j + 1]),
                      self.xyz(us[i], vs[j + 1])]
                col = interpolate_color(ManimColor(lo), ManimColor(hi), 0.22 + 0.78 * max(0.0, float(N @ L)))
                quads.append((np.mean([p[2] for p in cs]),
                              Polygon(*[np.array([p[0], p[1], 0]) + self.c for p in cs], fill_color=col,
                                      fill_opacity=1, stroke_color=col, stroke_width=0.6)))
        quads.sort(key=lambda q: q[0])
        return VGroup(*[q[1] for q in quads])

    def curve(self, fn, t0, t1, color, width, n=420, hidden_opacity=0.0):
        """Curve fn(s), s in [t0, t1], on the surface: seen parts solid; hidden parts dashed at hidden_opacity."""
        ts = np.linspace(t0, t1, n)
        P = np.array([fn(s) for s in ts])
        vis = self.visible(P)
        seen, hidden, start = VGroup(), VGroup(), 0
        for k in range(1, n + 1):
            if k == n or vis[k] != vis[start]:
                seg = P[start:min(k + 1, n)]
                if len(seg) >= 2:
                    pts = [np.array([p[0], p[1], 0]) + self.c for p in seg]
                    vm = VMobject().set_points_smoothly(pts) if len(pts) > 3 else VMobject().set_points_as_corners(pts)
                    if vis[start]:
                        seen.add(vm.set_stroke(color, width, 1.0))
                    elif hidden_opacity > 0:
                        hidden.add(dashed(vm.set_stroke(color, width * 0.7, hidden_opacity), 0.12))
                start = k
        return VGroup(seen, hidden)

    def wire(self, n_u=18, n_v=8, color="#3B506A", width=1.0):
        g = VGroup()
        for i in range(n_u):
            g.add(self.curve(lambda v, u=TAU * i / n_u: self.xyz(u, v), 0, TAU, color, width, n=160)[0])
        for j in range(n_v):
            g.add(self.curve(lambda u, v=TAU * j / n_v: self.xyz(u, v), 0, TAU, color, width, n=320)[0])
        return g


# ---------------------------------------------------------------- the chapter
class Fibres(Chapter):
    BEATS = ["III.0", "III.1", "III.2", "III.3", "III.4"]
    EXTRA = {"III.2": 1.5, "III.3": 4.0, "III.4": 5.0}

    # -------------------------------------------------------------- staging
    def tick(self):
        """An invisible time-based updater, so that waits keep rendering frames (live readouts, the sea)."""
        if getattr(self, "_ticker", None) is None or self._ticker not in self.mobjects:
            self._ticker = Mobject()
            self._ticker.add_updater(lambda m, dt: None)
            self.add(self._ticker)

    def go(self, b, phrase, offset=0.0, end=False):
        self.at(self.cue(b, phrase, offset, end))

    def fade_all(self, keep=(), run_time=0.5):
        """Fade out everything but `keep`, the chapter label, the legend and the ticker."""
        keep = set(keep) | {self._ticker} | set(getattr(self, "_chrome", ()))
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def chrome(self):
        lab = self.chapter_label("III", "fibres")
        leg = self.legend("fibres")
        self._chrome = (lab, leg)
        for m in self._chrome:
            m.set_opacity(0)
        return [m.animate.set_opacity(1) for m in self._chrome]

    def construct(self):
        self.tick()
        self.chapter_card("III.0", "III", "Fibres", "how a method learns", notch=3)
        self.beat_gauge()
        self.beat_charge()
        self.beat_sees()
        self.beat_torus()
        self.finish()

    # -------------------------------------------------------------- III.1 the gauge
    def beat_gauge(self):
        b = "III.1"
        B1, A1, BA1 = FX["B1"], FX["A1"], FX["BA1"]
        Bg, gA, prod, dBA, G1 = FX["Bg"], FX["gA"], FX["prod"], FX["dBA"], FX["G1"]
        detg = np.linalg.det(G1)
        K = len(Bg) - 1
        vB, vA, vBA, vg = 1.3, 1.1, float(np.abs(BA1).max()), 1.8
        cell, y0 = 0.4, 0.55

        hB = heatmap(B1, cell, vB).move_to([-5.55, y0, 0])
        hA = heatmap(A1, cell, vA).move_to([-3.0, y0, 0])
        hBA = heatmap(BA1, cell, vBA).move_to([0.45, y0, 0])
        dot = math(r"\cdot", 50, INK2).move_to([-4.6, y0, 0])
        eq = math("=", 42, INK2).move_to([-1.27, y0, 0])
        ylab, ydim = -1.15, -1.62
        lB = math("B", 36).move_to([-5.55, ylab, 0])
        lA = math("A", 36).move_to([-3.0, ylab, 0])
        lBA = math("BA", 36).move_to([0.45, ylab, 0])
        dims = VGroup(*[sans(t, 15, INK3).move_to([x, ydim, 0])
                        for t, x in (("m × r", -5.55), ("r × n", -3.0), ("m × n", 0.45))])
        k_w = kicker("weights", INK3, 15).move_to([0.45, 2.05, 0])
        k_p = kicker("parameters", INK3, 15).move_to([-4.25, 2.05, 0])
        f1 = MathTex(r"\Delta W \;=\;", r"B\,A", tex_template=TEX, font_size=40, color=INK)
        f2 = MathTex(r"\Delta W \;=\;", r"B\,A", r"\;=\; (B\,g^{-1})\,(g\,A)", tex_template=TEX, font_size=40,
                     color=INK)
        f1.move_to([-2.15, 2.62, 0])
        f2.move_to([-2.15, 2.62, 0])
        lBg = math(r"B\,g^{-1}", 36).move_to(lB)
        lgA = math(r"g\,A", 36).move_to(lA)

        # the gauge element g on the right, with a slider along its path
        gx = 4.55
        g_lab = math(r"g\in\mathrm{GL}_2", 36).move_to([gx, 2.35, 0])
        g_tile = tile(G1[0], 0.8, vg, GOLD, CORAL).move_to([gx, 1.05, 0])
        det0 = value_text("det g =", detg[0], 18, INK, digits=3)
        place(det0, [gx - 0.95, 0.02, 0])
        track = Line([gx - 1.15, -0.62, 0], [gx + 1.15, -0.62, 0], stroke_color=RULE, stroke_width=4)
        knob = Dot(track.get_start(), radius=0.09, color=GOLD)
        trail = Line(track.get_start(), track.get_start() + 0.001 * RIGHT, stroke_color=GOLD, stroke_width=4)
        cap = VGroup(sans("a path in", 15, INK3), math(r"\mathrm{GL}_2", 22, INK3),
                     sans("from g = I", 15, INK3)).arrange(RIGHT, buff=0.1).move_to([gx, -1.0, 0])

        # 0.33 the weights; 3.2 the parameters
        self.go(b, "Images say")
        self.play(*self.chrome(), FadeIn(hBA), FadeIn(lBA), FadeIn(dims[2]), FadeIn(k_w), run_time=1.0)
        self.go(b, "Fibres say")
        self.play(FadeIn(hB), FadeIn(hA), FadeIn(dot), FadeIn(eq), FadeIn(lB), FadeIn(lA), FadeIn(dims[:2]),
                  FadeIn(k_p), run_time=1.0)
        self.go(b, "Take a LoRA update")
        self.play(Write(f1), run_time=1.0)
        self.go(b, "slip any invertible")
        self.play(FadeIn(g_lab), FadeIn(g_tile), FadeIn(det0), Create(track), FadeIn(knob), FadeIn(cap),
                  run_time=1.0)
        self.go(b, "B times g-inverse")
        self.play(TransformMatchingTex(f1, f2), FadeTransform(lB, lBg), FadeTransform(lA, lgA), run_time=1.2)

        # 16.14 the factors change, the product does not
        ts, te = self.cue(b, "The factors change"), self.cue(b, "gauge") - 0.4
        kf = lambda: int(round(smooth(np.clip((self.renderer.time - ts) / (te - ts), 0, 1)) * K))
        hB.add_updater(lambda m, dt: recolor_heatmap(m, Bg[kf()], vB))
        hA.add_updater(lambda m, dt: recolor_heatmap(m, gA[kf()], vA))
        hBA.add_updater(lambda m, dt: recolor_heatmap(m, prod[kf()], vBA))
        g_tile.add_updater(lambda m, dt: m.become(tile(G1[kf()], 0.8, vg, GOLD, CORAL).move_to([gx, 1.05, 0])))
        det = live(lambda: value_text("det g =", detg[kf()], 18, INK, digits=3), [gx - 0.95, 0.02, 0])
        self.remove(det0)
        self.add(det)
        knob.add_updater(lambda m, dt: m.move_to(track.point_from_proportion(kf() / K)))
        trail.add_updater(lambda m, dt: m.put_start_and_end_on(track.get_start(),
                                                               track.point_from_proportion(max(kf() / K, 1e-3))))
        self.add(trail, knob)

        self.go(b, "Their product")
        ring = SurroundingRectangle(hBA, buff=0.1, color=TIDE, stroke_width=2.5, corner_radius=0.06)
        ro = live(lambda: value_text("max |Δ(BA)| =", dBA[kf()], 22, TIDE), [-1.65, -2.17, 0])
        self.play(Create(ring), FadeIn(ro), run_time=0.8)
        self.play(FadeOut(ring), run_time=0.7)

        self.go(b, "All of these settings")
        fib = serif("same weights, different parameters — a fibre", 30, INK).move_to([-2.0, -2.9, 0])
        self.play(FadeIn(fib, shift=0.15 * UP), run_time=1.0)
        self.go(b, "When the product has full rank")
        note = VGroup(sans("full rank: the fibre is the", 18, INK2), math(r"\mathrm{GL}_r", 26, INK2),
                      sans("orbit", 18, INK2)).arrange(RIGHT, buff=0.1).move_to([-2.0, -3.45, 0])
        self.play(FadeIn(note), run_time=0.8)
        self.go(b, "and the group of all such g")
        gauge = kicker("LoRA's gauge", GOLD, 16).next_to(g_lab, UP, buff=0.16)
        self.play(FadeIn(gauge, shift=0.1 * UP), run_time=0.8)

        self.at(self.END(b))
        self.fade_all()

    # -------------------------------------------------------------- III.2 a conserved charge
    def beat_charge(self):
        b = "III.2"
        ft, fB, fA, fphi, fdrift = FX["flow_t"], FX["flow_B"], FX["flow_A"], FX["flow_phi"], FX["flow_drift"]
        etas, d_sgd, d_adam = FX["etas"], FX["d_sgd"], FX["d_adam"]
        vB, vA = float(np.abs(fB).max()), float(np.abs(fA).max())
        vphi = float(np.abs(fphi).max()) * 1.15
        y0 = 0.5

        ts = self.cue(b, "Under gradient flow")
        te = self.cue(b, "each divided", end=True) + 0.6
        def kf():
            tau = 2.0 * np.clip((self.renderer.time - ts) / (te - ts), 0, 1)
            return int(np.searchsorted(ft, tau + 1e-12) - 1)

        hB = heatmap(fB[0], 0.32, vB).move_to([-6.05, y0, 0])
        hA = heatmap(fA[0], 0.32, vA).move_to([-4.15, y0, 0])
        lB = math("B", 32).move_to([-6.05, -0.75, 0])
        lA = math("A", 32).move_to([-4.15, -0.75, 0])
        k_flow = kicker("gradient flow", INK3, 15).move_to([-5.0, 1.75, 0])
        tread = live(lambda: VGroup(Text(f"t = {ft[kf()]:.2f}", font=MONO, font_size=18, color=INK2,
                                         t2c={"t =": INK3})), [-5.6, -1.25, 0])
        phi_tile = tile(fphi[0], 1.0, vphi, fmt="{:.4f}", size=15).move_to([-1.45, y0, 0])
        l_phi = math(r"\Phi", 36).move_to([-1.45, -0.75, 0])
        form = math(r"\Phi \;=\; B^{\top}\!B/\eta_B \;-\; AA^{\top}\!/\eta_A", 40).move_to([-3.55, 2.62, 0])
        form_note = sans("equal block rates in this run", 15, INK3).next_to(form, DOWN, buff=0.14)

        # the drift meter: |Φ(t) − Φ(0)| against flow time, on a linear scale up to 0.01
        ax = plain_axes([0, 2], [0, 0.01], width=5.0, height=0.9).move_to([-3.75, -2.25, 0])
        m_title = sans("drift of Φ", 16, INK2).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        y_top = math(r"0.01", 18, INK3).next_to(ax.c2p(0, 0.01), LEFT, buff=0.1)
        y_bot = math("0", 18, INK3).next_to(ax.c2p(0, 0), LEFT, buff=0.1)
        x_lab = math("t", 22, INK3).next_to(ax.c2p(2, 0), RIGHT, buff=0.1)

        def meter_line():
            k = max(kf(), 1)
            return polyline([ax.c2p(ft[i], min(fdrift[i], 0.01)) for i in range(k + 1)], TIDE, 3)
        trace = meter_line()
        trace.add_updater(lambda m, dt: m.become(meter_line()))
        dread = live(lambda: value_text("max |Φ(t) − Φ(0)| =", fdrift[kf()], 20, TIDE), [-6.25, -3.2, 0])

        # 0.0 B and A train under gradient flow
        self.at(self.T(b))
        self.play(FadeIn(hB), FadeIn(hA), FadeIn(lB), FadeIn(lA), FadeIn(k_flow), run_time=1.0)
        self.go(b, "Under gradient flow")
        hB.add_updater(lambda m, dt: recolor_heatmap(m, fB[kf()], vB))
        hA.add_updater(lambda m, dt: recolor_heatmap(m, fA[kf()], vA))
        self.add(tread)
        self.wait(0.2)
        self.go(b, "one quantity never changes")
        self.play(FadeIn(phi_tile), FadeIn(l_phi), Create(ax), FadeIn(m_title), FadeIn(y_top), FadeIn(y_bot),
                  FadeIn(x_lab), run_time=1.0)
        phi_tile.add_updater(lambda m, dt: m.become(tile(fphi[kf()], 1.0, vphi, fmt="{:.4f}", size=15)
                                                      .move_to([-1.45, y0, 0])))
        self.add(trace, dread)
        self.go(b, "B-transpose-B")
        self.play(Write(form), run_time=1.4)
        self.play(FadeIn(form_note), run_time=0.5)

        # 18.15 the credit card
        self.go(b, "It's a conserved charge")
        cd = card("Theorem III.5 · a conserved charge (known)",
                  r"Under gradient flow, $\Phi=B^{\top}B/\eta_B-AA^{\top}/\eta_A$ is conserved, for any loss and any "
                  r"data.",
                  credit=r"Du, Hu \& Lee 2018; Arora, Cohen \& Hazan 2018; Kunin et al.\ 2020; Zhao et al.\ 2022",
                  width=5.3, body_size=26, credit_size=19).move_to([3.75, 0.55, 0])
        self.play(FadeIn(cd, shift=0.15 * UP), run_time=0.8)

        # 25.81 one step from the same state: SGD falls 4x per halving, Adam 2x (log scale)
        self.go(b, "With plain SGD", offset=-0.45)
        self.play(FadeOut(cd), run_time=0.45)
        x0, per = 2.45, 0.55                                  # 1e-6 at x0; 0.55 frame units per decade
        X = lambda v: x0 + per * (np.log10(v) + 6)
        ys_sgd, ys_adam, bh = [1.55, 1.1, 0.65], [-0.3, -0.75, -1.2], 0.3
        axis = Line([x0, -1.62, 0], [X(1e-1), -1.62, 0], stroke_color=RULE, stroke_width=1.5)
        ticks = VGroup(*[VGroup(Line([X(10.0 ** e), -1.62, 0], [X(10.0 ** e), -1.7, 0], stroke_color=RULE,
                                     stroke_width=1.5),
                                math(r"10^{%d}" % e, 18, INK3).move_to([X(10.0 ** e), -1.98, 0]))
                         for e in range(-6, 0)])
        cap = sans("drift of Φ after one step from the same state", 16, INK2).move_to([3.7, -2.45, 0])
        title = kicker("one step, three step sizes", INK3, 15).move_to([3.7, 2.2, 0])

        def rows(vals, ys, color, name):
            bars_, vals_, labs_ = VGroup(), VGroup(), VGroup()
            for e, v, y in zip(etas, vals, ys):
                bars_.add(Rectangle(width=X(v) - x0, height=bh, fill_color=color, fill_opacity=0.9, stroke_width=0)
                          .move_to([(x0 + X(v)) / 2, y, 0]))
                vals_.add(place(value_text("", v, 16, INK, force=True), [X(v) + 0.12, y, 0]))
                labs_.add(place(VGroup(mono(f"η = {e:g}", 15, INK3)), [x0 - 0.12, y, 0], RIGHT))
            head = sans(name, 21, color, MEDIUM).move_to([0.55, np.mean(ys), 0])
            return bars_, vals_, labs_, head

        sb, sv, sl, sh = rows(d_sgd, ys_sgd, TIDE, "SGD")
        ab, av, al, ah = rows(d_adam, ys_adam, CORAL, "Adam")
        self.play(FadeIn(title), Create(axis), FadeIn(ticks), FadeIn(cap), FadeIn(sh), FadeIn(sl),
                  LaggedStart(*[GrowFromEdge(r, LEFT) for r in sb], lag_ratio=0.25), run_time=1.4)
        self.play(FadeIn(sv), run_time=0.5)

        def ratio_marks(vals_mob, ys, label, color, x):
            g = VGroup()
            for y1, y2 in zip(ys, ys[1:]):
                br = VGroup(Line([x, y1 - 0.06, 0], [x, y2 + 0.06, 0], stroke_color=color, stroke_width=1.6),
                            Line([x - 0.07, y1 - 0.06, 0], [x, y1 - 0.06, 0], stroke_color=color, stroke_width=1.6),
                            Line([x - 0.07, y2 + 0.06, 0], [x, y2 + 0.06, 0], stroke_color=color, stroke_width=1.6))
                g.add(VGroup(br, mono(label, 17, color).move_to([x + 0.32, (y1 + y2) / 2, 0])))
            return g

        self.go(b, "but only at second order")
        x_sgd = max(m.get_right()[0] for m in sv) + 0.22
        q4 = ratio_marks(sv, ys_sgd, "÷4", TIDE, x_sgd)
        self.play(FadeIn(q4), run_time=0.8)

        self.go(b, "Under Adam")
        self.play(FadeIn(ah), FadeIn(al), LaggedStart(*[GrowFromEdge(r, LEFT) for r in ab], lag_ratio=0.2),
                  run_time=0.9)
        self.go(b, "isn't conserved")
        x_adam = max(m.get_right()[0] for m in av) + 0.22
        q2 = ratio_marks(av, ys_adam, "÷2", CORAL, x_adam)
        self.play(FadeIn(av), FadeIn(q2), run_time=0.7)

        self.at(self.END(b))
        self.fade_all()

    # -------------------------------------------------------------- III.3 what each optimizer sees
    def beat_sees(self):
        b = "III.3"
        # nested groups on the left: signed permutations ⊂ O(r) ⊂ GL_r
        cx = -4.55
        outer = RoundedRectangle(corner_radius=0.25, width=4.3, height=6.05, stroke_color=INK3, stroke_width=1.6,
                                 fill_color=BG2, fill_opacity=0.6).move_to([cx, -0.12, 0])
        mid = RoundedRectangle(corner_radius=0.22, width=3.6, height=4.55, stroke_color=GOLD, stroke_width=1.6,
                               fill_color=BG2, fill_opacity=0.9).move_to([cx, -0.62, 0])
        inner = RoundedRectangle(corner_radius=0.2, width=2.95, height=2.6, stroke_color=CORAL, stroke_width=1.6,
                                 fill_color=BG3, fill_opacity=0.9).move_to([cx, -1.12, 0])
        l_out = VGroup(math(r"\mathrm{GL}_r", 32, INK), sans("the gauge", 16, INK3)).arrange(RIGHT, buff=0.16)
        l_out.next_to(outer.get_top(), DOWN, buff=0.22)
        t_out = sans("undamped scaled GD, LoRA-RITE", 15, INK3).next_to(l_out, DOWN, buff=0.1)
        l_mid = VGroup(math(r"\mathrm{O}(r)", 30, GOLD), sans("rotations, reflections", 16, INK2)
                       ).arrange(RIGHT, buff=0.16).next_to(mid.get_top(), DOWN, buff=0.2)
        t_mid = sans("SGD", 18, GOLD, MEDIUM).next_to(l_mid, DOWN, buff=0.12)
        l_in = sans("signed permutations", 16, INK2).next_to(inner.get_top(), DOWN, buff=0.18)
        t_in = sans("Adam", 18, CORAL, MEDIUM).next_to(inner.get_bottom(), UP, buff=0.16)
        perms = []
        for P in ([[1, 0], [0, 1]], [[0, 1], [1, 0]]):
            for s in ([1, 1], [1, -1], [-1, 1], [-1, -1]):
                perms.append(np.diag(s) @ np.array(P, dtype=float))
        sp = FX["sperm"]
        tiles = VGroup()
        for k, P in enumerate(perms):
            t = heatmap(P, 0.17, 1.0, TIDE, CORAL)
            if np.array_equal(P, sp):
                t = VGroup(t, SurroundingRectangle(t, buff=0.04, color=GOLD, stroke_width=1.6, corner_radius=0.03))
            tiles.add(t)
        tiles.arrange_in_grid(2, 4, buff=(0.32, 0.22)).move_to([cx, -1.05, 0])

        # three rows on the right
        yrows = [2.0, -0.05, -2.1]
        boxes = [frame_box(3.3, 1.62, [2.55, y, 0]) for y in yrows]
        rot = FX["rot"]
        rot_tex = r"\begin{pmatrix}%.2f&%.2f\\%.2f&%.2f\end{pmatrix}" % (rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1])
        sp_tex = r"\begin{pmatrix}%d&%d\\%d&%d\end{pmatrix}" % tuple(int(v) for v in sp.ravel())
        heads = []
        for name, gname, gtex, y in (("SGD", "rotation", rot_tex, yrows[0]), ("Adam", "rotation", rot_tex, yrows[1]),
                                     ("Adam", "signed permutation", sp_tex, yrows[2])):
            gm = math("g=" + gtex, 22, INK2)
            heads.append(VGroup(sans(name, 22, INK, MEDIUM), sans("g = " + gname, 16, INK2), gm)
                         .arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-0.5, y, 0]).align_to([-1.75, 0, 0], LEFT))
        legend = VGroup(Line(ORIGIN, 0.45 * RIGHT, stroke_color=TIDE, stroke_width=6), sans("from A₀", 15, INK2),
                        Line(ORIGIN, 0.45 * RIGHT, stroke_color=GOLD, stroke_width=2), sans("from g A₀", 15, INK2)
                        ).arrange(RIGHT, buff=0.12)
        legend[2].shift(0.18 * RIGHT)
        legend[3].shift(0.18 * RIGHT)
        cap = VGroup(sans("two zero-B starts; W(t) in a 2-D projection", 15, INK3), legend).arrange(RIGHT, buff=0.35)
        cap.move_to([2.4, 3.05, 0])

        runs = [(FX["P_sgd_base"], FX["P_sgd_rot"], FX["run_sgd_rot"], TIDE),
                (FX["P_adam_base"], FX["P_adam_rot"], FX["run_adam_rot"], CORAL),
                (FX["P_adam_base"], FX["P_adam_perm"], FX["run_adam_perm"], TIDE)]
        fit_sgd = path_fitter([runs[0][0], runs[0][1]], boxes[0])
        fit_adam = path_fitter([runs[1][0], runs[1][1]], boxes[1])

        def row_mobs(i):
            base, other, run, col = runs[i]
            fit = fit_sgd if i == 0 else (lambda Q, i=i: fit_adam(Q) + np.array([0, yrows[i] - yrows[1], 0]))
            pb, po = fit(base), fit(other)
            l_base = polyline(pb, TIDE, 6.0)
            l_other = polyline(po, GOLD, 2.0)
            start = Dot(pb[0], radius=0.05, color=INK)
            return l_base, l_other, start, run, col

        def draw_row(i, run_time):
            l_base, l_other, start, run, col = row_mobs(i)
            clock = {"t0": None}
            n = len(run) - 1
            k_of = lambda: 0 if clock["t0"] is None else \
                int(round(np.clip((self.renderer.time - clock["t0"]) / run_time, 0, 1) * n))
            lab = mono("max |ΔW|", 15, INK3).move_to([5.55, yrows[i] + 0.3, 0])
            val = live(lambda: value_text("", run[k_of()], 22, col, digits=3 if col == CORAL else 2),
                       [5.55, yrows[i] - 0.12, 0], ORIGIN)
            self.play(FadeIn(boxes[i]), FadeIn(lab), FadeIn(val), FadeIn(heads[i]), FadeIn(start), run_time=0.4)
            clock["t0"] = self.renderer.time
            self.play(Create(l_base), Create(l_other), run_time=run_time, rate_func=linear)
            val.clear_updaters()
            final = place(value_text("", run[-1], 22, col, digits=3 if col == CORAL else 2), [5.55, yrows[i] - 0.12, 0],
                          ORIGIN)
            self.remove(val)
            self.add(final)
            return VGroup(boxes[i], lab, final, heads[i], start, l_base, l_other)

        self.at(self.T(b))
        self.go(b, "The gauge is a whole group")
        self.play(FadeIn(outer), FadeIn(l_out), run_time=0.9)
        self.go(b, "but the usual optimizers")
        self.play(FadeIn(mid), FadeIn(l_mid), FadeIn(inner), FadeIn(l_in), FadeIn(t_out), run_time=1.0)
        self.play(FadeIn(cap), run_time=0.6)
        self.go(b, "Gradient descent respects")
        self.play(FadeIn(t_mid), mid.animate.set_stroke(width=3.0), run_time=0.6)
        r1 = draw_row(0, 2.6)
        self.go(b, "Adam respects far less")
        self.play(FadeIn(t_in), inner.animate.set_stroke(width=3.0), mid.animate.set_stroke(width=1.6), run_time=0.6)
        self.go(b, "which shuffle")
        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in tiles], lag_ratio=0.12), run_time=1.4)
        self.go(b, "Start Adam")
        r2 = draw_row(1, 4.6)
        self.go(b, "Relate them")
        r3 = draw_row(2, 2.6)

        self.go(b, "The optimizer sees more")
        cd = card("Theorem III.12 · Corollary III.13 · what the optimizer sees",
                  r"On the full-rank locus, with hyperparameters shared by the rank channels, the largest part of the "
                  r"gauge $\mathrm{GL}_r$ that an update rule respects is $\mathrm{O}(r)$ for SGD, the signed "
                  r"permutations $B_r$ for Adam, and all of $\mathrm{GL}_r$ for undamped scaled GD and LoRA-RITE. "
                  r"From $B_0=0$, the image remembers only $\operatorname{row}A_0$; SGD remembers $A_0^{\top}A_0$; "
                  r"Adam remembers $A_0$ up to signed permutations of its rows.",
                  width=9.6, body_size=27).move_to([0.2, 0.0, 0])
        dim = [m for m in self.mobjects if m not in (self._ticker, *self._chrome)]
        self.play(*[m.animate.fade(0.82) for m in dim], FadeIn(cd, shift=0.15 * UP), run_time=0.9)
        self.at(self.END(b))
        self.fade_all()

    # -------------------------------------------------------------- III.4 the torus; nut 2 dissolves
    def beat_torus(self):
        b = "III.4"
        # five knobs of LoRA (schematic positions): A's start and rate, B's start and rate, the scale
        xs = [-4.3, -3.2, -1.3, -0.2, 1.7]
        names = [r"\sigma_A", r"\eta_A", r"\sigma_B", r"\eta_B", r"\alpha"]
        descs = ["A's start", "A's rate", "B's start", "B's rate", "scale"]
        pos0 = [0.15, -0.1, -1.0, -0.1, 0.35]                 # knob heights in [-1.1, 1.1] on the tracks
        yc, half = 0.55, 1.15
        tracks = VGroup(*[Line([x, yc - half, 0], [x, yc + half, 0], stroke_color=RULE, stroke_width=4) for x in xs])
        knobs = VGroup(*[Dot([x, yc + p, 0], radius=0.1, color=TIDE if k < 4 else GOLD)
                         for k, (x, p) in enumerate(zip(xs, pos0))])
        knobs[2].set_color(INK3)
        names_m = VGroup(*[math(s, 32).move_to([x, yc - half - 0.42, 0]) for s, x in zip(names, xs)])
        descs_m = VGroup(*[sans(d, 15, INK3).move_to([x, yc - half - 0.85, 0]) for d, x in zip(descs, xs)])
        zero = sans("B₀ = 0", 15, INK3).next_to(knobs[2], RIGHT, buff=0.12)
        brA = VGroup(Line([xs[0] - 0.25, yc + half + 0.3, 0], [xs[1] + 0.25, yc + half + 0.3, 0], stroke_color=INK3,
                          stroke_width=1.5), sans("block A", 16, INK2).move_to([(xs[0] + xs[1]) / 2, yc + half + 0.58, 0]))
        brB = VGroup(Line([xs[2] - 0.25, yc + half + 0.3, 0], [xs[3] + 0.25, yc + half + 0.3, 0], stroke_color=INK3,
                          stroke_width=1.5), sans("block B", 16, INK2).move_to([(xs[2] + xs[3]) / 2, yc + half + 0.58, 0]))
        panel = VGroup(tracks, knobs, names_m, descs_m, zero, brA, brB).shift(0.9 * LEFT)
        act = math(r"(\sigma_A,\ \eta_A,\ \alpha)\ \longmapsto\ (c\,\sigma_A,\ c\,\eta_A,\ \alpha/c)", 36)
        act.move_to([3.9, 1.0, 0])
        act_note = sans("rescale block A by c", 16, INK3).next_to(act, UP, buff=0.18)
        adam_tag = sans("under Adam, with ε, decay and clipping off", 18, INK2).next_to(act, DOWN, buff=0.45)
        same = serif("the same weights W(t), step for step", 24, INK).next_to(adam_tag, DOWN, buff=0.5)
        mul = [math(r"\times c", 26, TIDE).next_to(knobs[k], RIGHT, buff=0.12) for k in (0, 1)]
        div = math(r"\div c", 26, GOLD).next_to(knobs[4], RIGHT, buff=0.12)

        self.at(self.T(b))
        self.play(FadeIn(panel), run_time=1.0)
        self.go(b, "Rescale a block")
        self.play(brA[0].animate.set_stroke(TIDE, 2.5), brA[1].animate.set_color(TIDE), FadeIn(act_note),
                  run_time=0.7)
        self.go(b, "and its learning rate")
        self.play(knobs[0].animate.shift(0.55 * UP), knobs[1].animate.shift(0.55 * UP), run_time=1.2)
        for m_ in mul:
            m_.shift(0.55 * UP)
        self.play(FadeIn(mul[0]), FadeIn(mul[1]), run_time=0.5)
        self.go(b, "adjust alpha")
        div.shift(0.55 * DOWN)
        self.play(knobs[4].animate.shift(0.55 * DOWN), FadeIn(div, shift=0.55 * DOWN), Write(act), run_time=1.1)
        self.go(b, "and under Adam", offset=0.3)
        self.play(FadeIn(adam_tag), run_time=0.8)
        self.go(b, "the weights follow")
        self.play(FadeIn(same, shift=0.12 * UP), run_time=1.0)
        self.go(b, "These rescalings form a torus", offset=-0.55)
        self.fade_all(run_time=0.5)

        # 18.3 the torus of rescalings (schematic)
        T = Torus2D(R=1.55, a=0.62, tilt=60 * DEGREES, center=[-3.35, -0.05, 0])
        body = T.body()
        wire = T.wire()
        vB_, uL, uP = 1.05, -0.78 * PI, -0.27 * PI
        circ_B = T.curve(lambda u: T.xyz(u, vB_), 0, TAU, GOLD, 2.6, hidden_opacity=0.4)
        circ_A = T.curve(lambda v: T.xyz(uL, v), 0, TAU, GOLD, 2.6, hidden_opacity=0.4)
        joinarc = T.curve(lambda u: T.xyz(u, vB_), uL, uP, TIDE, 5.0, n=200)[0]
        pL, pP = T.pt(uL, vB_), T.pt(uP, vB_)
        dL = Dot(pL, radius=0.085, color=INK)
        dP = Dot(pP, radius=0.085, color=TIDE)
        labL = sans("LoRA", 20, INK, MEDIUM).next_to(pL, UL, buff=0.08)
        labP = VGroup(sans("LoRA+,", 20, TIDE, MEDIUM), math(r"\lambda", 28, TIDE)).arrange(RIGHT, buff=0.08)
        labP.next_to(pP, UR, buff=0.06)
        top_pt = T.pt(0.5 * PI, vB_)
        lab_B = sans("rescale B: its start and its rate together", 18, GOLD).move_to([-3.35, 2.35, 0])
        lead_B = DashedLine(lab_B.get_bottom() + 0.05 * DOWN, top_pt + 0.06 * UP, color=GOLD, stroke_width=1.2,
                            dash_length=0.05)
        a_pt = T.pt(uL, -0.55 * PI)
        lab_A = sans("rescale A: its start and its rate together", 18, GOLD).move_to([-3.35, -2.55, 0])
        lead_A = DashedLine(lab_A.get_top() + 0.05 * UP, T.pt(uL, -0.3 * PI) + 0.04 * DOWN, color=GOLD,
                            stroke_width=1.2, dash_length=0.05)
        tag = schematic_tag().move_to([-1.0, -1.95, 0])
        group_note = VGroup(sans("the group is", 15, INK3), math(r"(\mathbb{R}_{>0})^2", 22, INK3),
                            sans("one factor per block", 15, INK3)).arrange(RIGHT, buff=0.1).move_to([-3.35, -3.1, 0])

        self.go(b, "These rescalings form a torus")
        self.play(FadeIn(body), FadeIn(wire), run_time=1.0)
        self.play(Create(circ_B[0]), FadeIn(circ_B[1]), Create(circ_A[0]), FadeIn(circ_A[1]), FadeIn(lab_B),
                  FadeIn(lab_A), Create(lead_B), Create(lead_A), FadeIn(tag), FadeIn(group_note), run_time=1.2)

        # 21.55 LoRA+ is a point on the torus, joined to LoRA along the B circle
        r1 = math(r"\text{LoRA+}:\quad \eta_B=\lambda\,\eta_A,\quad \text{scale }\alpha", 30, TIDE)
        r2 = math(r"\text{LoRA}:\quad \eta_B=\eta_A,\quad \text{scale }\lambda\alpha", 30, INK)
        arr = VGroup(Arrow(UP * 0.35, DOWN * 0.35, buff=0, stroke_width=2.5, color=GOLD,
                           max_tip_length_to_length_ratio=0.25),
                     sans("rescale B by 1/λ (its start stays 0)", 16, GOLD))
        arr[1].next_to(arr[0], RIGHT, buff=0.15)
        hyper = VGroup(r1, arr, r2).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([3.7, 0.5, 0])
        self.go(b, "LoRA-plus, which gives B")
        self.play(FadeIn(dL, scale=0.5), FadeIn(labL), FadeIn(r1), run_time=0.9)
        self.play(FadeIn(dP, scale=0.5), FadeIn(labP), run_time=0.8)
        self.go(b, "is just a point")
        self.play(Create(joinarc), FadeIn(arr), FadeIn(r2), run_time=1.3)

        # 30.42 the opening simulation returns, with its exact settings; the control peels away
        self.go(b, "That's the coincidence")
        self.play(FadeOut(hyper), run_time=0.5)
        P_plus, P_alpha, P_plain = LP["P_plus"], LP["P_alpha"], LP["P_plain"]
        W_plus, W_alpha, W_plain = LP["W_plus"], LP["W_alpha"], LP["W_plain"]
        run_same = np.maximum.accumulate(np.abs(W_plus - W_alpha).max(axis=(1, 2)))
        run_ctrl = np.maximum.accumulate(np.abs(W_plus - W_plain).max(axis=(1, 2)))
        lam = float(LP["lam"])
        box = frame_box(4.9, 2.75, [3.6, 0.72, 0])
        fit = path_fitter([P_plus, P_plain], box, margin=0.26)
        q_plus, q_alpha, q_plain = fit(P_plus), fit(P_alpha), fit(P_plain)
        l_plus = polyline(q_plus, TIDE, 6.0)
        l_alpha = polyline(q_alpha, GOLD, 2.0)
        l_plain = polyline(q_plain, CORAL, 2.6)
        start = Dot(q_plus[0], radius=0.05, color=INK)
        key = VGroup(
            VGroup(Line(ORIGIN, 0.4 * RIGHT, stroke_color=TIDE, stroke_width=6),
                   sans(f"LoRA+, λ = {lam:g}", 16, INK2)).arrange(RIGHT, buff=0.12),
            VGroup(Line(ORIGIN, 0.4 * RIGHT, stroke_color=GOLD, stroke_width=2),
                   sans(f"LoRA, α × {lam:g}", 16, INK2)).arrange(RIGHT, buff=0.12),
            VGroup(Line(ORIGIN, 0.4 * RIGHT, stroke_color=CORAL, stroke_width=2.6),
                   sans("control: LoRA, α unchanged", 16, INK2)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(box, UP, buff=0.14).align_to(box, LEFT)
        key[2].set_opacity(0)
        settings = VGroup(scope_line(["Adam", "ε = 0", "no decay", "no clipping"], 17, INK2),
                          scope_line(["B₀ = 0", "shared schedule", f"λ = {lam:g}"], 17, INK2)
                          ).arrange(DOWN, buff=0.12).next_to(box, DOWN, buff=0.2)
        n_steps = len(P_plus) - 1
        clock = {"same": None, "ctrl": None}
        k_of = lambda key_, dur_: 0 if clock[key_] is None else \
            int(round(np.clip((self.renderer.time - clock[key_]) / dur_, 0, 1) * n_steps))
        dur, dur2 = 5.0, 3.6
        x_ro = box.get_left()[0]
        ro_same = live(lambda: value_text("max |W_LoRA+ − W_LoRA(α×16)| =", run_same[k_of("same", dur)], 18, TIDE),
                       [x_ro, -2.25, 0])
        ro_ctrl = live(lambda: value_text("control:", run_ctrl[k_of("ctrl", dur2)], 18, CORAL, digits=3),
                       [x_ro, -2.72, 0])
        self.play(FadeIn(box), FadeIn(key[:2]), FadeIn(settings), FadeIn(start), FadeIn(ro_same),
                  run_time=0.6)
        clock["same"] = self.renderer.time
        self.play(Create(l_plus), Create(l_alpha), run_time=dur, rate_func=linear)

        self.go(b, "LoRA-plus and plain LoRA", offset=0.2)
        self.play(key[2].animate.set_opacity(1), FadeIn(ro_ctrl), run_time=0.4)
        clock["ctrl"] = self.renderer.time
        self.play(Create(l_plain), run_time=dur2, rate_func=linear)
        ro_same.clear_updaters()
        ro_ctrl.clear_updaters()
        self.go(b, "take the same path")
        self.play(Indicate(ro_same, color=TIDE, scale_factor=1.06), run_time=1.0)

        # 44.17 the paper card
        self.go(b, "Schulman and colleagues")
        cd = card("Theorem III.10 · Corollary III.11 · the hyperparameter torus",
                  r"Rescale a block's starting size and its learning rate together, and the scale to match: the "
                  r"weights follow the same trajectory (rates $\times c$ under Adam, $\times c^2$ under SGD). So "
                  r"under Adam with $\varepsilon=0$, no decay, no clipping and a shared schedule, LoRA+ with ratio "
                  r"$\lambda$ from $B_0=0$ is plain LoRA with $\alpha\times\lambda$; under SGD, with "
                  r"$\alpha\times\sqrt{\lambda}$.",
                  credit=r"LoRA and Adam, $\varepsilon$ included: Schulman \& Thinking Machines Lab 2025 $\cdot$ "
                         r"new here: SGD (factor $\sqrt{\lambda}$) and every multihomogeneous method",
                  width=9.4, body_size=27, credit_size=20).move_to([0.2, 0.15, 0])
        dim = [m for m in self.mobjects if m not in (self._ticker, *self._chrome)]
        self.play(*[m.animate.fade(0.85) for m in dim], FadeIn(cd, shift=0.15 * UP), run_time=0.9)

        # 56.8 cut to the shore: nut 2 dissolves
        self.shore_cut(b, 2)

    def shore_cut(self, b, k):
        """The nut-k shore cut in the EXTRA time after beat b's narration, timed on absolute scene times."""
        end = self.END(b)
        self.at(self.T(b) + self.plan[b]["audio"])
        for m in self.mobjects:
            if m is not self._ticker:
                m.clear_updaters()
        self.clear_stage(run_time=0.6, keep=(self._ticker,))
        s = self.shore(LEVELS[k], dissolved=set(range(1, k)))
        self.at(end - 4.0)
        self.dissolve_nut(s, k, LEVELS[k + 1])
        self.at(end - 0.6)
        for m in self.mobjects:
            if m is not self._ticker:
                m.clear_updaters()
        self.clear_stage(run_time=0.6, keep=(self._ticker,))
