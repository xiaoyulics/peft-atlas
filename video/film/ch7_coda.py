import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *
# Coda: beats C.1 (three questions as the sea covers the shore) and C.2 (end card) of video/script/SCRIPT.md.
# No numbers on screen. The end card reuses the social-card parts of ch0_opening.py (contours, neatline, motif, title).
#
#   PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -qh --fps 30 --disable_caching \
#       --media_dir video/build/Coda video/film/ch7_coda.py Coda
from ch0_opening import card_glow, contours, cpx, keyed, motif, neatline, title_block


class Coda(Chapter):
    BEATS = ["C.1", "C.2"]
    EXTRA = {"C.2": 6.0}

    def construct(self):
        self.three_questions()
        self.end_card()
        self.finish()

    # ============================================================ C.1 · Three questions
    def three_questions(self):
        cue = lambda p, end=False: self.cue("C.1", p, end=end)
        t0 = self.T("C.1")
        t_settle = cue("Often, the answer is already under the water.")
        s = self.shore(LEVELS[6], dissolved={1, 2, 3, 4})
        sea, land = s["sea"], s["land"]
        rise = keyed([(t0 + 0.4, LEVELS[6]), (cue("Often, the answer is already under the water.", end=True), COVERED)])
        alpha = ValueTracker(0.0)

        def sea_upd(m, dt):                                 # time-based, so the waves also move during waits
            a = alpha.get_value()
            m.become(wave_polygon(rise(self.renderer.time), 1.2 * self.renderer.time, 0.22 * a))
            m.set_stroke(opacity=0.75 * a)

        sea.clear_updaters()
        sea.add_updater(sea_upd)
        sea_upd(sea, 0)
        self.at(t0)
        self.play(FadeIn(land), alpha.animate.set_value(1.0), run_time=1.2)

        # the legend of I.4 returns as three questions, one at a time, above the water
        rows = [("image", "What can its map reach?", "What can its map reach?"),
                ("fibres", "How do its fibres meet the optimizer?", "How do its fibres meet the optimizer?"),
                ("base change", "What happens when the base point moves?", "And what happens when the base point moves?")]
        qs = [serif(q, 32, INK) for _, q, _ in rows]
        tags = [sans(w, 16, TIDE, MEDIUM) for w, _, _ in rows]
        TAG_X, Q_X, Y0, DY, SETTLE = -5.05, -3.05, 3.32, 0.56, 0.06
        for k, (q, tg) in enumerate(zip(qs, tags)):
            q.move_to([Q_X, Y0 - k * DY, 0], aligned_edge=LEFT)
            tg.move_to([TAG_X, q[0].get_center()[1], 0], aligned_edge=LEFT)
        block = VGroup(*qs, *tags)
        block.shift((-block.get_center()[0]) * RIGHT)
        assert min(q.get_bottom()[1] for q in qs) - SETTLE > COVERED + 0.15, "a question would touch the water"
        self.hud(*qs, *tags)
        for m in (*qs, *tags):
            m.set_opacity(0)
        for k, (_, _, said) in enumerate(rows):
            self.at(cue(said))
            qs[k].shift(0.14 * DOWN)
            anims = [qs[k].animate.set_opacity(1).shift(0.14 * UP), tags[k].animate.set_opacity(1)]
            if k:
                anims += [qs[k - 1].animate.set_color(INK2), tags[k - 1].animate.set_color(INK3)]
            self.play(*anims, run_time=0.9)

        # 'Often, the answer is already under the water.': the questions settle
        self.at(t_settle)
        self.play(*[q.animate.set_color(INK2).shift(SETTLE * DOWN) for q in qs],
                  *[t.animate.set_color(INK3).shift(SETTLE * DOWN) for t in tags], run_time=1.6, rate_func=smooth)
        self.coda_sea = (sea, land, qs, tags)

    # ============================================================ C.2 · End card
    def end_card(self):
        cue = lambda p, end=False: self.cue("C.2", p, end=end)
        t0, t1 = self.T("C.2"), self.END("C.2")
        old = list(self.mobjects)
        glow, cont, neat = card_glow(), contours(), neatline()
        for m in (glow, cont, neat):
            m.set_z_index(-1)
        mo = motif()
        X0 = cpx(80, 0)[0]
        title = title_block(40).move_to([X0, 1.55, 0], aligned_edge=LEFT)
        authors = sans("Xiaoyu Li · Zhizhou Sha · Chiwun Yang · Dai Shi", 24, INK).move_to([X0, -0.45, 0],
                                                                                           aligned_edge=LEFT)
        credit = VGroup(serif("with Claude Opus 5.5 (Anthropic),", 22, INK2, italic=True),
                        serif("in the spirit of Grothendieck's rising sea", 22, INK2, italic=True))
        credit.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([X0, -1.25, 0], aligned_edge=LEFT)
        url = sans("xiaoyulics.com/peft-atlas", 30, TIDE, MEDIUM).move_to([X0, -2.45, 0], aligned_edge=LEFT)
        under = Line(url.get_corner(DL) + 0.1 * DOWN, url.get_corner(DR) + 0.1 * DOWN, stroke_color=TIDE,
                     stroke_width=1.6, stroke_opacity=0.7)
        assert credit.get_right()[0] < 2.3 and title.get_right()[0] < 1.6, "end-card text runs into the motif"

        # the contours return
        self.at(t0)
        self.play(*[FadeOut(m) for m in old], FadeIn(glow), FadeIn(cont), FadeIn(neat), run_time=1.6)
        for m in old:
            m.clear_updaters()
        dot, halo, lab = self.theta0(mo["T"], label=True, label_dir=LEFT)
        lab.shift(0.05 * LEFT)
        self.play(LaggedStart(*[FadeIn(m, shift=0.1 * UP) for m in (*title[0], title[1])], lag_ratio=0.2),
                  LaggedStart(*[Create(r) for r in mo["cone"][1]], lag_ratio=0.02), FadeIn(mo["cone"][0]),
                  Create(mo["cone"][2]), Create(mo["orbit"]), Create(mo["line"]), FadeIn(mo["zero"]),
                  FadeIn(dot), FadeIn(halo), FadeIn(lab), run_time=2.0)
        self.at(cue("are in the paper"))
        self.play(FadeIn(authors, shift=0.08 * UP), run_time=1.0)
        self.at(cue("companion site"))
        self.play(FadeIn(credit, shift=0.08 * UP), run_time=1.0)
        self.at(cue("The link is below."))
        self.play(FadeIn(url, shift=0.08 * UP), Create(under), run_time=1.0)

        # hold, then fade to the background
        self.at(t1 - 1.6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
