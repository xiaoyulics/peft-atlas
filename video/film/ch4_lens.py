import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *
# Chapter IV · The lens — when a method compresses the gradient (beats IV.0 and IV.1 of video/script/SCRIPT.md).
#
# Every number on screen is read from video/sim/out/galore.npz, written by video/sim/galore.py: GaLore with Adam
# against one-sided LoRA W_k + P_k C (Prop IV.2), and a control whose LoRA side resets its optimizer state.
#
#   PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -qh --fps 30 --disable_caching \
#       --media_dir video/build/Lens video/film/ch4_lens.py Lens

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim" / "out"
GL = np.load(SIM / "galore.npz")

CM_PER_UNIT_30 = 1 / 0.886          # a LaTeX \parbox of 1 cm is 0.886 frame units wide at font_size 30


# ---------------------------------------------------------------- numbers (as in ch3_fibres.py)
def sci_parts(x, digits=2, force=None):
    """('6.1', '−16') for 6.11e-16; ('0.115', None) for 0.115 at three digits; ('0.0', None) for an exact zero."""
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
    """A monospace readout 'label 6.1 × 10⁻¹⁶' with a raised exponent. Element 0 is the main line."""
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
    return g.shift(np.array(anchor, dtype=float) - g[0].get_critical_point(edge))


def live(make, anchor, edge=LEFT):
    """A readout rebuilt every frame by make() (a time-based updater, so waits keep rendering)."""
    m = place(make(), anchor, edge)
    m.add_updater(lambda mob, dt: mob.become(place(make(), anchor, edge)))
    return m


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


def polyline(pts, color, width, opacity=1.0):
    return VMobject().set_points_as_corners(pts).set_stroke(color, width, opacity)


def scope_line(items, size=17, color=INK2):
    return Text("  ·  ".join(items), font=SANS, font_size=size, color=color)


def arrow(a, b, color=INK3, width=2.2):
    return Arrow(a, b, buff=0, stroke_width=width, color=color, max_tip_length_to_length_ratio=0.3,
                 max_stroke_width_to_length_ratio=12)


def lens_box(center, w, h, top, bottom, top_color=TIDE, bottom_color=GOLD, size=26):
    """A lens drawn as a box with two lanes: a forward arrow on top, a backward arrow below.
    `top` and `bottom` are (left, right) LaTeX labels at the ends of each lane, or a single centred label."""
    c = np.array(center, dtype=float)
    box = RoundedRectangle(corner_radius=0.22, width=w, height=h, stroke_color=INK3, stroke_width=1.4,
                           fill_color=BG2, fill_opacity=1).move_to(c)
    yt, yb = c[1] + h * 0.22, c[1] - h * 0.22
    fwd = arrow([c[0] - w * 0.18, yt, 0], [c[0] + w * 0.18, yt, 0], top_color, 2.6)
    bwd = arrow([c[0] + w * 0.18, yb, 0], [c[0] - w * 0.18, yb, 0], bottom_color, 2.6)
    tl = math(top[0], size).next_to(fwd, LEFT, buff=0.14)
    tr = math(top[1], size).next_to(fwd, RIGHT, buff=0.14)
    bl = math(bottom[0], size).next_to(bwd, LEFT, buff=0.14)
    br = math(bottom[1], size).next_to(bwd, RIGHT, buff=0.14)
    return VGroup(box, fwd, bwd, tl, tr, bl, br)


# ---------------------------------------------------------------- the chapter
class Lens(Chapter):
    BEATS = ["IV.0", "IV.1"]
    EXTRA = {"IV.1": 5.0}

    def tick(self):
        """An invisible time-based updater, so that waits keep rendering frames (live readouts, the sea)."""
        if getattr(self, "_ticker", None) is None or self._ticker not in self.mobjects:
            self._ticker = Mobject()
            self._ticker.add_updater(lambda m, dt: None)
            self.add(self._ticker)

    def go(self, b, phrase, offset=0.0, end=False):
        self.at(self.cue(b, phrase, offset, end))

    def fade_all(self, keep=(), run_time=0.5):
        keep = set(keep) | {self._ticker} | set(getattr(self, "_chrome", ()))
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def chrome(self):
        lab = self.chapter_label("IV", "the lens")
        leg = self.legend("fibres")
        self._chrome = (lab, leg)
        for m in self._chrome:
            m.set_opacity(0)
        return [m.animate.set_opacity(1) for m in self._chrome]

    def construct(self):
        self.tick()
        self.chapter_card("IV.0", "IV", "The lens", "when a method compresses the gradient", notch=4)
        self.beat_lens()
        self.finish()

    def beat_lens(self):
        b = "IV.1"
        lr = float(GL["lr"])
        T, K = int(GL["T"]), int(GL["K"])

        # ---------------- 0.3 no adapter: GaLore trains W directly and compresses its gradient
        W0, G0, P0, R0, U0, dW0 = GL["W0"], GL["G0"], GL["P0"], GL["R0"], GL["U0"], GL["dW0"]
        cell, y0 = 0.17, 0.3
        xW, xG, xR, xU, xD = -5.6, -2.85, 0.0, 2.75, 5.45
        hW = heatmap(W0, cell).move_to([xW, y0, 0])
        hG = heatmap(G0, cell).move_to([xG, y0, 0])
        hR = heatmap(R0, cell).move_to([xR, y0, 0])
        hU = heatmap(U0, cell).move_to([xU, y0, 0])
        hD = heatmap(dW0, cell).move_to([xD, y0, 0])
        ylab = 1.48
        lW = math("W", 28).move_to([xW, ylab, 0])
        lG = math(r"G=\nabla L(W)", 26).move_to([xG, ylab, 0])
        lR = math(r"P^{\top}G", 26).move_to([xR, ylab, 0])
        lU = math(r"\mathrm{Adam}(P^{\top}G)", 26).move_to([xU, ylab, 0])
        lD = math(r"-\eta\,P\,\mathrm{Adam}(P^{\top}G)", 24).move_to([xD, ylab, 0])
        dims = VGroup(*[sans(t, 15, INK3).move_to([x, -0.82, 0])
                        for t, x in (("10 × 12", xW), ("10 × 12", xG), ("2 × 12", xR), ("2 × 12", xU),
                                     ("10 × 12", xD))])
        a_WG = arrow([xW + 1.08, y0, 0], [xG - 1.08, y0, 0])
        a_GR = arrow([xG + 1.08, y0, 0], [xR - 1.08, y0, 0], GOLD)
        a_RU = arrow([xR + 1.08, y0, 0], [xU - 1.08, y0, 0])
        a_UD = arrow([xU + 1.08, y0, 0], [xD - 1.08, y0, 0], GOLD)
        t_GR = math(r"P^{\top}", 24, GOLD).next_to(a_GR, UP, buff=0.08)
        t_UD = math("P", 24, GOLD).next_to(a_UD, UP, buff=0.08)
        t_RU = sans("Adam", 15, INK2).next_to(a_RU, UP, buff=0.1)
        p_note = sans("P: the top-2 left singular vectors of G, refreshed each period", 15, INK3)
        p_note.move_to([0.0, -1.3, 0])
        loop = VMobject().set_points_as_corners([[xD, -1.75, 0], [xD, -2.1, 0], [xW, -2.1, 0], [xW, -1.75, 0]])
        loop.set_stroke(INK3, 1.8)
        tip = Triangle(fill_color=INK3, fill_opacity=1, stroke_width=0).scale(0.07).move_to([xW, -1.72, 0])
        upd = math(r"W \;\leftarrow\; W-\eta\,P\,\mathrm{Adam}(P^{\top}G)", 32).move_to([0, -2.6, 0])
        no_ad = math(r"W_0 \;+\; B\,A", 40).move_to([0, 2.55, 0])
        strike = Line(no_ad[0][3].get_left() + 0.08 * LEFT, no_ad.get_right() + 0.08 * RIGHT, stroke_color=CORAL,
                      stroke_width=4)
        k_gal = kicker("GaLore", GOLD, 20).move_to([0, 2.55, 0])

        self.at(self.T(b))
        self.go(b, "Some methods add")
        self.play(*self.chrome(), FadeIn(no_ad), run_time=0.8)
        self.play(Create(strike), run_time=0.7)
        self.go(b, "they train the weights")
        self.play(FadeIn(hW), FadeIn(lW), FadeIn(dims[0]), run_time=0.9)
        self.go(b, "but compress the gradient")
        self.play(GrowArrow(a_WG), FadeIn(hG), FadeIn(lG), FadeIn(dims[1]), run_time=0.8)
        self.play(GrowArrow(a_GR), FadeIn(t_GR), FadeIn(hR), FadeIn(lR), FadeIn(dims[2]), run_time=0.8)
        self.go(b, "GaLore projects")
        self.play(FadeOut(no_ad), FadeOut(strike), FadeIn(k_gal), run_time=0.8)
        self.go(b, "onto a small subspace")
        self.play(FadeIn(p_note), Indicate(hR, color=GOLD, scale_factor=1.08), run_time=1.0)
        self.go(b, "and runs Adam there")
        self.play(GrowArrow(a_RU), FadeIn(t_RU), FadeIn(hU), FadeIn(lU), FadeIn(dims[3]), run_time=0.6)
        self.play(GrowArrow(a_UD), FadeIn(t_UD), FadeIn(hD), FadeIn(lD), FadeIn(dims[4]), run_time=0.6)
        self.play(Create(loop), FadeIn(tip), FadeIn(upd), run_time=0.7)

        # ---------------- 13.2 backpropagation is a lens
        self.go(b, "Backpropagation has", offset=-0.25)
        self.fade_all(run_time=0.5)
        gen = lens_box([0, 0.85, 0], 6.2, 2.3, (r"\text{weights}", r"\text{loss}"),
                       (r"\text{gradient}", r"dL=1"), size=28)
        f_lab = sans("forward", 16, TIDE).next_to(gen[1], UP, buff=0.1)
        b_lab = sans("backward", 16, GOLD).next_to(gen[2], DOWN, buff=0.1)
        k_lens = kicker("backpropagation is a lens", INK3, 16).next_to(gen[0], UP, buff=0.22)
        self.play(FadeIn(gen[0]), FadeIn(k_lens), run_time=0.7)
        self.go(b, "a forward map")
        self.play(GrowArrow(gen[1]), FadeIn(gen[3]), FadeIn(gen[4]), FadeIn(f_lab), run_time=0.8)
        self.go(b, "paired with a backward one")
        self.play(GrowArrow(gen[2]), FadeIn(gen[5]), FadeIn(gen[6]), FadeIn(b_lab), run_time=0.8)

        # ---------------- 19.3 two views of the same thing
        ad = lens_box([-3.45, -1.25, 0], 5.6, 1.8, (r"C", r"W_k+P\,C"), (r"P^{\top}G", r"G"), size=28)
        cg = lens_box([3.45, -1.25, 0], 5.6, 1.8, (r"W", r"W"), (r"P^{\top}G", r"G"), size=28)
        t_ad = VGroup(sans("a linear adapter", 18, TIDE, MEDIUM), sans("one-sided LoRA", 15, INK3)
                      ).arrange(RIGHT, buff=0.18).next_to(ad[0], UP, buff=0.14)
        t_cg = VGroup(sans("a compressed gradient", 18, GOLD, MEDIUM), sans("GaLore", 15, INK3)
                      ).arrange(RIGHT, buff=0.18).next_to(cg[0], UP, buff=0.14)
        top_grp = VGroup(gen, f_lab, b_lab, k_lens)
        self.go(b, "Through that lens")
        self.play(top_grp.animate.scale(0.72).move_to([0, 2.15, 0]), run_time=0.9)
        self.go(b, "a linear adapter", offset=-0.1)
        self.play(FadeIn(ad), FadeIn(t_ad), run_time=0.8)
        self.go(b, "and a compressed gradient")
        self.play(FadeIn(cg), FadeIn(t_cg), run_time=0.8)
        self.go(b, "two views")
        rings = VGroup(*[SurroundingRectangle(m, buff=0.08, color=GOLD, stroke_width=2, corner_radius=0.05)
                         for m in (ad[5], cg[5])])
        same = sans("the optimizer is fed the same compressed gradient", 20, INK2).move_to([0, -2.8, 0])
        self.play(Create(rings), FadeIn(same), run_time=0.9)

        # ---------------- 25.9 GaLore is one-sided LoRA: three runs on the same data
        self.go(b, "So the third nut", offset=-0.3)
        self.fade_all(run_time=0.5)
        Wg, Wl, Wr = GL["W_gal"], GL["W_lora"], GL["W_reset"]
        d_same, d_reset = GL["d_same"], GL["d_reset"]
        run_same, run_reset = np.maximum.accumulate(d_same), np.maximum.accumulate(d_reset)
        n = len(Wg) - 1
        vmax = float(max(np.abs(Wg - W0).max(), np.abs(Wr - W0).max()))
        cell2, ym = 0.2, 1.05
        xs = [-4.6, -1.35, 3.95]
        maps = [heatmap(np.zeros_like(W0), cell2, vmax).move_to([x, ym, 0]) for x in xs]
        frozen = VGroup(sans("frozen factor", 15, INK3), math(r"P_k", 22, INK3), sans("= GaLore's projector", 15, INK3)
                        ).arrange(RIGHT, buff=0.08)
        titles = [
            VGroup(sans("GaLore", 21, INK, MEDIUM), sans("project, Adam, map back", 15, INK3)),
            VGroup(VGroup(sans("one-sided LoRA", 21, INK, MEDIUM), math(r"W_k+P_k\,C", 24, INK2)
                          ).arrange(RIGHT, buff=0.14),
                   frozen, sans("merged every period, state carried over", 15, INK3)),
            VGroup(sans("control: LoRA side resets", 21, CORAL, MEDIUM),
                   sans("its optimizer state each period", 15, INK3)),
        ]
        for t, x in zip(titles, xs):
            t.arrange(DOWN, buff=0.07)
            t.move_to([x, 0, 0]).align_to([0, 3.12, 0], UP)
        eqs = math("=", 40, INK2).move_to([(xs[0] + xs[1]) / 2, ym, 0])
        neq = math(r"\neq", 40, CORAL).move_to([(xs[1] + xs[2]) / 2 + 0.25, ym, 0])

        # the gap chart: log scale, steps on x, merges marked
        cx0, cx1, cy0, cy1 = -5.25, 5.35, -3.05, -1.4
        X = lambda k: cx0 + (cx1 - cx0) * k / n
        Y = lambda v: cy0 + (cy1 - cy0) * (np.log10(max(float(v), 1e-17)) + 17) / 17
        axes = VGroup(Line([cx0, cy0, 0], [cx1, cy0, 0], stroke_color=RULE, stroke_width=1.5),
                      Line([cx0, cy0, 0], [cx0, cy1, 0], stroke_color=RULE, stroke_width=1.5))
        yt = VGroup(*[math(s, 17, INK3).next_to([cx0, Y(v), 0], LEFT, buff=0.1)
                      for s, v in ((r"10^{-16}", 1e-16), (r"10^{-8}", 1e-8), (r"1", 1.0))])
        merges = VGroup(*[DashedLine([X(T * j), cy0, 0], [X(T * j), cy1, 0], color=GOLD, stroke_width=1.2,
                                     dash_length=0.05, stroke_opacity=0.7) for j in range(1, K)])
        m_labs = VGroup(*[sans("merge", 15, GOLD).move_to([X(T * j), cy0 - 0.22, 0]) for j in range(1, K)])
        x_lab = sans("step", 15, INK3).next_to([cx1, cy0, 0], DOWN, buff=0.12).align_to([cx1, 0, 0], RIGHT)
        p_labs = VGroup(*[sans(f"period {j + 1}", 15, INK3).move_to([X(T * j + T / 2), cy0 - 0.22, 0])
                          for j in range(K)])
        c_title = sans("max |W_GaLore − W| so far, log scale", 15, INK2).next_to([cx0, cy1, 0], UP, buff=0.12).align_to(
            [cx0, 0, 0], LEFT)

        clock = {"t0": None}
        dur = 9.4
        kf = lambda: 0 if clock["t0"] is None else int(round(np.clip((self.renderer.time - clock["t0"]) / dur, 0,
                                                                     1) * n))

        def curve(vals, color, width):
            k = max(kf(), 1)
            return polyline([[X(i), Y(vals[i]), 0] for i in range(k + 1)], color, width)

        c_same = curve(d_same, TIDE, 2.6)
        c_reset = curve(d_reset, CORAL, 2.6)
        cursor = Line([X(0), cy0, 0], [X(0), cy1, 0], stroke_color=INK3, stroke_width=1)
        ro_same = live(lambda: value_text("difference ≈", run_same[kf()], 20, TIDE),
                       [(xs[0] + xs[1]) / 2 - 1.75, -0.32, 0])
        ro_reset = live(lambda: value_text("difference =", run_reset[kf()], 20, CORAL, digits=3),
                        [xs[2] - 1.5, -0.32, 0])
        step_ro = live(lambda: VGroup(Text(f"step {kf():3d} of {n}", font=MONO, font_size=15, color=INK3)),
                       [cx1, cy1 + 0.22, 0], RIGHT)
        scope = scope_line(["Adam", "no weight decay", "no clipping", "the same ε on both sides"], 16, INK2)
        scope.move_to([0, -0.8, 0])

        self.play(*[FadeIn(m) for m in maps], *[FadeIn(t) for t in titles], FadeIn(axes), FadeIn(yt),
                  FadeIn(merges), FadeIn(m_labs), FadeIn(x_lab), FadeIn(p_labs), FadeIn(c_title), FadeIn(ro_same),
                  FadeIn(ro_reset), FadeIn(step_ro), FadeIn(cursor), run_time=0.9)
        self.go(b, "without weight decay")
        self.play(FadeIn(scope), run_time=0.7)
        self.go(b, "GaLore is exactly")
        for h, Wt in zip(maps, (Wg, Wl, Wr)):
            h.add_updater(lambda m, dt, Wt=Wt: recolor_heatmap(m, Wt[kf()] - W0, vmax))
        c_same.add_updater(lambda m, dt: m.become(curve(d_same, TIDE, 2.6)))
        c_reset.add_updater(lambda m, dt: m.become(curve(d_reset, CORAL, 2.6)))
        cursor.add_updater(lambda m, dt: m.put_start_and_end_on([X(kf()), cy0, 0], [X(kf()), cy1, 0]))
        flash = SurroundingRectangle(maps[1], buff=0.08, color=GOLD, stroke_width=2.5, corner_radius=0.05)
        flash.add_updater(lambda m, dt: m.set_stroke(opacity=max(
            [np.exp(-((kf() - T * j) / 2.2) ** 2) if kf() >= T * j - 1 else 0.0 for j in range(1, K + 1)])))
        self.add(c_same, c_reset, flash)
        clock["t0"] = self.renderer.time
        mp = sans("(machine precision)", 15, INK3).move_to([0, -0.32, 0]).align_to([-0.66, 0, 0], LEFT)
        self.play(FadeIn(eqs), FadeIn(mp), run_time=0.6)
        self.go(b, "with GaLore's projector")
        self.play(frozen.animate.set_color(GOLD), run_time=0.6)
        self.go(b, "merged and restarted")
        self.play(FadeIn(neq), run_time=0.6)
        self.at(clock["t0"] + dur + 0.1)
        for m in (*maps, c_same, c_reset, cursor, flash, ro_same, ro_reset, step_ro):
            m.clear_updaters()

        # ---------------- 41.2 the paper card
        self.go(b, "Torroba-Hennigen")
        cd = card("Proposition IV.2 · the degree-one collapse (known)",
                  r"Let $\rho_k(C)=W_k+\mathcal{L}C$ with $\mathcal{L}$ linear and $C_0=0$, and let the optimizer read "
                  r"only the gradients it is fed (SGD, momentum, Adam; no weight decay, no clipping). Training $C$ and "
                  r"merging gives the same weights as the backward method with compression $\mathcal{L}^{*}$ and "
                  r"anchor $\mathcal{L}$, by induction over the steps. So GaLore with projector $P_k$ is one-sided "
                  r"LoRA $W_k+P_kC$, merged and restarted every period, its optimizer state carried over.",
                  credit=r"random projections: Hao, Cao \& Mou 2024 $\cdot$ GaLore: Torroba-Hennigen, Lang, Guo \& "
                         r"Kim 2025 $\cdot$ backprop as a lens: Fong, Spivak \& Tuyéras 2017; Cruttwell et al.\ 2021",
                  width=9.6, body_size=27, credit_size=20).move_to([0.1, 0.1, 0])
        dim = [m for m in self.mobjects if m not in (self._ticker, *self._chrome)]
        self.play(*[m.animate.fade(0.85) for m in dim], FadeIn(cd, shift=0.15 * UP), run_time=0.9)

        # ---------------- 50.2 cut to the shore: nut 3 dissolves
        self.shore_cut(b, 3)

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
