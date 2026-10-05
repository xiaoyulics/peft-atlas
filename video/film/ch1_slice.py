"""Chapter I · The slice — fine-tuning moves a point (beats I.0-I.4)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from story import *  # noqa: E402,F401

P0 = np.array([2.3, -0.3, 0])          # theta_0 on screen


def double_cone_sketch(P, h=1.25, w=0.85, color=TIDE):
    """A flat drawing of the double cone with its tip at P (two triangles and two rims)."""
    top = VGroup(Line(P, P + [-w, h, 0]), Line(P, P + [w, h, 0]),
                 Ellipse(width=2 * w, height=0.42, fill_color=color, fill_opacity=0).move_to(P + [0, h, 0]))
    bot = VGroup(Line(P, P + [-w, -h, 0]), Line(P, P + [w, -h, 0]),
                 Ellipse(width=2 * w, height=0.42, fill_color=color, fill_opacity=0).move_to(P + [0, -h, 0]))
    g = VGroup(top, bot).set_stroke(color, 2.2)
    fill = VGroup(Polygon(P, P + [-w, h, 0], P + [w, h, 0]), Polygon(P, P + [-w, -h, 0], P + [w, -h, 0]))
    fill.set_fill(color, 0.08).set_stroke(width=0)
    return VGroup(fill, g)


class Slice(Chapter):
    BEATS = ["I.0", "I.1", "I.2", "I.3", "I.4"]
    EXTRA = {"I.2": 3.0, "I.3": 4.5, "I.4": 2.2}

    def construct(self):
        self.chapter_card("I.0", "I", "The slice", "fine-tuning moves a point", notch=1)
        self.beat_point()
        self.beat_map()
        self.beat_slice()
        self.beat_questions()
        self.finish()

    # -------------------------------------------------------------- I.1 one point
    def beat_point(self):
        self.at(self.T("I.1"))
        self.label = self.chapter_label("I", "the slice")
        rng = np.random.default_rng(3)
        layers = VGroup(*[heatmap(rng.normal(size=(7, 11)), cell=0.17) for _ in range(3)])
        for k, h in enumerate(layers):
            h.move_to([-3.3 + 0.32 * k, 0.45 - 0.32 * k, 0])
        cap = sans("pretrained weights", 18, INK3).next_to(layers, UP, buff=0.35).align_to(layers, LEFT)
        self.play(LaggedStart(*[FadeIn(h, shift=0.1 * UP) for h in layers], lag_ratio=0.25), FadeIn(cap),
                  run_time=1.4)
        self.at(self.cue("I.1", "a single point"))
        dot, halo, lab = self.theta0(P0, label=False)
        self.remove(halo)
        self.play(*[h.animate.scale(0.04).move_to(P0).set_opacity(0.2) for h in layers], FadeOut(cap),
                  run_time=1.1, rate_func=smooth)
        self.remove(*layers)
        self.add(dot)
        self.add_fixed_orientation_mobjects(halo)
        self.play(FadeIn(halo, scale=0.6), run_time=0.5)
        space = sans("weight space Θ", 18, INK3).move_to(P0 + [2.7, 2.6, 0])
        ring = Circle(radius=0.3, stroke_color=TIDE, stroke_width=1.5).move_to(P0)
        self.play(ring.animate.scale(9).set_stroke(opacity=0), FadeIn(space), run_time=1.4)
        self.remove(ring)
        self.at(self.cue("I.1", "Call it theta-zero"))
        th = math(r"\theta_0", 44, INK).next_to(P0, LEFT, buff=0.3).shift(0.1 * UP)
        cap = serif("the pretrained model", 26, INK2, italic=True).next_to(th, DOWN, buff=0.18).align_to(th, RIGHT)
        self.play(Write(th), FadeIn(cap, shift=0.1 * UP), run_time=0.9)
        self.at(self.cue("I.1", "Fine-tuning moves that point"))
        path = CubicBezier(P0, P0 + [0.9, 1.3, 0], P0 + [2.1, 1.0, 0], P0 + [2.6, 2.0, 0]).set_stroke(INK2, 2)
        mover = Dot(P0, radius=0.06, color=INK2)
        self.play(Create(path), MoveAlongPath(mover, path), run_time=1.6, rate_func=smooth)
        self.at(self.cue("I.1", "Full fine-tuning can move it anywhere"))
        fan = VGroup()
        for a in np.linspace(-1.9, 1.9, 9):
            end = P0 + 2.4 * np.array([np.cos(a) * 1.3, np.sin(a), 0]) * rng.uniform(0.75, 1.1)
            mid = (P0 + end) / 2 + 0.5 * np.array([-np.sin(a), np.cos(a), 0])
            fan.add(CubicBezier(P0, mid, mid, end).set_stroke(INK3, 1.4, opacity=0.8))
        tips = VGroup(*[Dot(c.get_end(), radius=0.04, color=INK3) for c in fan])
        self.play(LaggedStart(*[Create(c) for c in fan], lag_ratio=0.08), FadeIn(tips, lag_ratio=0.08),
                  run_time=1.8)
        self.p0 = VGroup(dot, th)
        self.halo = halo
        self.space = space
        self.i1_rest = VGroup(cap, path, mover, fan, tips)

    # -------------------------------------------------------------- I.2 a method is a map
    def beat_map(self):
        self.at(self.T("I.2"))
        self.play(FadeOut(self.i1_rest), run_time=0.7)
        Qc = np.array([-4.3, -0.4, 0])
        disc = Circle(radius=1.15, fill_color=BG2, fill_opacity=1, stroke_color=INK3, stroke_width=1.6).move_to(Qc)
        qlab = math("Q", 40, INK).next_to(disc, UP, buff=0.18)
        qcap = sans("a few trainable numbers", 18, INK3).next_to(disc, DOWN, buff=0.25)
        self.play(FadeIn(disc), Write(qlab), FadeIn(qcap), run_time=1.0)
        self.at(self.cue("I.2", "So think of it as a map"))
        arrow = CurvedArrow(Qc + [1.25, 0.5, 0], P0 + [-0.35, 0.35, 0], angle=-0.55, color=TIDE, stroke_width=3,
                            tip_length=0.22)
        rho = math(r"\rho", 44, TIDE).move_to((Qc + P0) / 2 + [0, 1.75, 0])
        self.play(Create(arrow), Write(rho), run_time=1.2)
        self.at(self.cue("I.2", "into weight space"))
        self.play(Indicate(self.space, color=INK, scale_factor=1.15), run_time=1.0)
        self.at(self.cue("I.2", "with one rule"))
        q0 = Dot(Qc, radius=0.07, color=INK)
        q0l = math("q_0", 34, INK).next_to(q0, DOWN, buff=0.12)
        self.play(FadeIn(q0, scale=0.5), Write(q0l), run_time=0.7)
        self.at(self.cue("I.2", "land exactly on theta-zero"))
        traveller = Dot(Qc, radius=0.06, color=TIDE)
        trip = CubicBezier(Qc, Qc + [2.2, 2.4, 0], P0 + [-2.2, 2.0, 0], P0)
        self.play(MoveAlongPath(traveller, trip), run_time=1.3, rate_func=smooth)
        ring = Circle(radius=0.12, stroke_color=TIDE, stroke_width=3).move_to(P0)
        formula = math(r"\rho:(Q,q_0)\longrightarrow(\Theta,\theta_0),\qquad \rho(q_0)=\theta_0", 40, INK)
        formula.to_corner(DL, buff=0.7)
        self.play(ring.animate.scale(4).set_stroke(opacity=0), FadeOut(traveller), Write(formula), run_time=1.3)
        self.remove(ring)
        # picture-only: a probe wanders in Q and its image traces a curve through theta_0
        t = ValueTracker(0.0)
        probe = always_redraw(lambda: Dot(Qc + 0.7 * np.array([np.sin(t.get_value()),
                                                                0.6 * np.sin(2 * t.get_value()), 0]),
                                          radius=0.05, color=GOLD))

        def image_point():
            u, v = np.sin(t.get_value()), 0.6 * np.sin(2 * t.get_value())
            return P0 + np.array([1.5 * u * abs(u) ** 0.2 - 0.3 * v, 1.7 * u * v, 0])
        img = always_redraw(lambda: Dot(image_point(), radius=0.05, color=GOLD))
        trace = TracedPath(image_point, stroke_color=GOLD, stroke_width=2, stroke_opacity=0.8)
        self.add(trace, probe, img)
        self.play(t.animate.set_value(TAU), run_time=2.6, rate_func=linear)
        self.i2 = VGroup(disc, qlab, qcap, arrow, rho, q0, q0l, probe, img, trace)
        self.formula = formula

    # -------------------------------------------------------------- I.3 the slice
    def beat_slice(self):
        self.at(self.T("I.3"))
        self.play(FadeOut(self.i2), FadeOut(self.p0), FadeOut(self.halo), FadeOut(self.space),
                  self.formula.animate.scale(0.7).to_corner(DR, buff=0.5), run_time=0.9)
        meth = math(r"\mathsf{Meth}(\Theta,\theta_0)", 46, INK).move_to([0, 2.75, 0])
        sub = sans("the slice of methods pointed at θ₀", 18, INK3).next_to(meth, DOWN, buff=0.18)
        self.play(Write(meth), FadeIn(sub), run_time=1.2)

        self.at(self.cue("I.3", "An arrow from one method to another"))
        M, N = math("M", 42).move_to([-1.6, 1.15, 0]), math("N", 42).move_to([1.6, 1.15, 0])
        Th = math(r"\Theta", 42).move_to([0, -0.75, 0])
        h = Arrow(M.get_right(), N.get_left(), buff=0.2, color=TIDE, stroke_width=3, max_tip_length_to_length_ratio=0.12)
        hl = math("h", 34, TIDE).next_to(h, UP, buff=0.08)
        rm = Arrow(M.get_bottom(), Th.get_left() + [0, 0.1, 0], buff=0.15, color=INK2, stroke_width=2.4)
        rn = Arrow(N.get_bottom(), Th.get_right() + [0, 0.1, 0], buff=0.15, color=INK2, stroke_width=2.4)
        rml = math(r"\rho_M", 30, INK2).next_to(rm.get_center(), LEFT, buff=0.15)
        rnl = math(r"\rho_N", 30, INK2).next_to(rn.get_center(), RIGHT, buff=0.15)
        tri = VGroup(M, N, Th, h, hl, rm, rn, rml, rnl)
        self.play(FadeIn(M), FadeIn(N), FadeIn(Th), run_time=0.6)
        self.play(GrowArrow(h), Write(hl), GrowArrow(rm), GrowArrow(rn), Write(rml), Write(rnl), run_time=1.3)
        self.at(self.cue("I.3", "the second can simulate the first"))
        says = VGroup(serif("N simulates M", 28, INK), math(r"\rho_N\circ h=\rho_M", 32, INK2)).arrange(RIGHT, buff=0.4)
        says.next_to(Th, DOWN, buff=0.45)
        self.play(FadeIn(says, shift=0.1 * UP), run_time=0.8)
        self.at(self.cue("I.3", "wherever the first can take theta-zero"))
        follow = serif("wherever M can take θ₀, N can follow", 24, TIDE, italic=True).next_to(says, DOWN, buff=0.3)
        self.play(FadeIn(follow, shift=0.1 * UP), Indicate(h, color=TIDE), run_time=1.0)

        self.at(self.cue("I.3", "At one end sits the frozen model"))
        self.play(FadeOut(VGroup(tri, says, follow)), run_time=0.6)
        y = -1.1
        frozen = VGroup(Dot([-5.4, y, 0], radius=0.09, color=INK2))
        fl = VGroup(serif("frozen", 26, INK), sans("can't move", 16, INK3)).arrange(DOWN, buff=0.08)
        fl.next_to(frozen, DOWN, buff=0.25)
        self.play(FadeIn(frozen, scale=0.5), FadeIn(fl), run_time=0.8)
        self.at(self.cue("I.3", "At the other sits full fine-tuning"))
        full = VGroup(Dot([5.4, y, 0], radius=0.11, color=INK))
        ful = VGroup(serif("full fine-tuning", 26, INK), sans("simulates everything", 16, INK3)).arrange(DOWN, buff=0.08)
        ful.next_to(full, DOWN, buff=0.25)
        self.play(FadeIn(full, scale=0.5), FadeIn(ful), run_time=0.8)
        self.at(self.cue("I.3", "Every method that starts exactly at theta-zero"))
        names = [("(IA)³", [-2.9, 0.25]), ("BitFit", [-3.0, -2.35]), ("LoRA, zero init", [-0.7, 0.65]),
                 ("LoRA, split init", [-0.4, -2.55]), ("OFT", [1.9, 0.4]), ("VeRA", [1.7, -1.25]),
                 ("DoRA", [3.1, -2.1])]
        nodes, links = VGroup(), VGroup()
        for name, (x, yy) in names:
            nd = VGroup(Dot([x, yy, 0], radius=0.05, color=TIDE),
                        sans(name, 17, INK2).next_to([x, yy, 0], UP, buff=0.12))
            nodes.add(nd)
            links.add(Line(frozen.get_center(), [x, yy, 0], stroke_color=RULE, stroke_width=1.3, buff=0.12))
            links.add(Line([x, yy, 0], full.get_center(), stroke_color=RULE, stroke_width=1.3, buff=0.14))
        self.play(LaggedStart(*[FadeIn(n, scale=0.8) for n in nodes], lag_ratio=0.12), Create(links),
                  run_time=2.0)
        za, sp = nodes[2][0].get_center(), nodes[3][0].get_center()
        noarrow = DashedLine(za + [0.05, -0.15, 0], sp + [0, 0.32, 0], color=CORAL, stroke_width=1.6, dash_length=0.08)
        nlab = sans("no arrow either way", 14, CORAL).next_to(noarrow, LEFT, buff=0.12)
        tag = schematic_tag().to_corner(DL, buff=0.5)
        self.play(Create(noarrow), FadeIn(nlab), FadeIn(tag), run_time=1.0)
        # picture-only: the card
        self.at(self.END("I.3") - 4.3)
        card = paper_card("Observation I.4 · universal properties",
                          r"In the slice of weight-space methods pointed at $\theta_0$, the frozen model is "
                          r"\emph{initial} and full fine-tuning is \emph{terminal}: the unique map to full "
                          r"fine-tuning is $\rho$ itself, that is, merging.", width_cm=10.5)
        card.move_to([0, 2.45, 0])
        self.play(FadeOut(VGroup(meth, sub)), FadeIn(card, shift=0.15 * UP), run_time=0.8)
        self.i3 = VGroup(card, frozen, fl, full, ful, nodes, links, noarrow, nlab, tag)

    # -------------------------------------------------------------- I.4 three questions
    def beat_questions(self):
        self.at(self.T("I.4"))
        self.play(FadeOut(self.i3), FadeOut(self.formula), run_time=0.7)
        P = np.array([0.6, -0.2, 0])
        params = sans("how many parameters?", 24, INK3).move_to([0.6, 1.6, 0])
        self.play(FadeIn(params), run_time=0.6)
        strike = Line(params.get_left(), params.get_right(), color=CORAL, stroke_width=2.5)
        self.play(Create(strike), run_time=0.6)
        self.at(self.cue("I.4", "three questions about its map"))
        dot, halo, lab = self.theta0(P, label=True, label_dir=LEFT)
        self.play(FadeOut(VGroup(params, strike)), FadeIn(dot, scale=0.5), FadeIn(halo), Write(lab), run_time=0.8)

        self.at(self.cue("I.4", "What is its image"))
        cone2d = double_cone_sketch(P, color=TIDE)
        q1 = VGroup(serif("image", 30, TIDE, SEMIBOLD), serif("where can it go?", 22, INK2, italic=True))
        q1.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([-4.6, 2.2, 0], aligned_edge=LEFT)
        self.play(FadeIn(cone2d), FadeIn(q1, shift=0.1 * RIGHT), run_time=1.0)

        self.at(self.cue("I.4", "What are its fibres"))
        target = P + np.array([0.5, 0.85, 0])
        srcs = [np.array([-3.4 + 0.45 * k, -1.6 + 0.35 * np.sin(1.3 * k), 0]) for k in range(5)]
        fibre = VMobject().set_points_smoothly(srcs).set_stroke(GOLD, 2)
        pts = VGroup(*[Dot(s, radius=0.05, color=GOLD) for s in srcs])
        rays = VGroup(*[Arrow(s, target, buff=0.08, stroke_width=1.6, color=GOLD, max_tip_length_to_length_ratio=0.05,
                              stroke_opacity=0.7) for s in srcs])
        tgt = Dot(target, radius=0.06, color=GOLD)
        q2 = VGroup(serif("fibres", 30, GOLD, SEMIBOLD),
                    serif("same weights; how training moves", 22, INK2, italic=True))
        q2.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([-4.6, 0.75, 0], aligned_edge=LEFT)
        self.play(Create(fibre), FadeIn(pts), FadeIn(q2, shift=0.1 * RIGHT), run_time=1.0)
        self.play(LaggedStart(*[GrowArrow(r) for r in rays], lag_ratio=0.1), FadeIn(tgt), run_time=1.4)

        self.at(self.cue("I.4", "And what happens when the base point itself moves"))
        P2 = P + np.array([2.6, -0.9, 0])
        hop = Arrow(P, P2, buff=0.15, color=LEAF, stroke_width=2.6)
        q3 = VGroup(serif("base change", 30, LEAF, SEMIBOLD), serif("when the base point moves", 22, INK2, italic=True))
        q3.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([-4.6, -2.65, 0], aligned_edge=LEFT)
        ghost = cone2d.copy()
        ghost[1].set_stroke(opacity=0.3)
        ghost[0].set_fill(opacity=0.03)
        self.add(ghost)
        self.play(GrowArrow(hop), cone2d.animate.shift(P2 - P), FadeIn(q3, shift=0.1 * RIGHT), run_time=1.4)

        self.at(self.cue("I.4", "Its image is what it can say"))
        say = [serif("Its image is what it can say.", 30, INK), serif("Its fibres are how it learns.", 30, INK),
               serif("Base change is how it travels.", 30, INK)]
        col = VGroup(*say).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0.2, 0.2, 0], aligned_edge=LEFT)
        picture = VGroup(cone2d, ghost, fibre, pts, rays, tgt, hop, dot, lab)
        self.play(FadeOut(picture), FadeOut(halo), FadeIn(say[0], shift=0.1 * UP), run_time=0.9)
        self.at(self.cue("I.4", "Its fibres are how it learns"))
        self.play(FadeIn(say[1], shift=0.1 * UP), run_time=0.7)
        self.at(self.cue("I.4", "Base change is how it travels"))
        self.play(FadeIn(say[2], shift=0.1 * UP), run_time=0.7)
        # picture-only: the three words become the legend that stays in the corner for the rest of the film
        self.at(self.END("I.4") - 1.9)
        leg = self.legend(None)
        leg.set_opacity(0)
        targets = [leg[0], leg[2], leg[4]]
        self.play(*[ReplacementTransform(q[0], t.copy().set_opacity(1)) for q, t in zip((q1, q2, q3), targets)],
                  FadeOut(VGroup(q1[1], q2[1], q3[1])), FadeOut(col), leg.animate.set_opacity(1),
                  run_time=1.3)
        self.clear_stage(run_time=0.5)
