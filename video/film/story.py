"""Narrative devices and the Chapter base class.

A chapter is one Manim scene covering several beats of video/script/SCRIPT.md. Its narration is laid out by
clock.plan(); the scene times its animations with self.at(t), self.T(beat) and self.cue(beat, phrase), and ends with
self.finish(). Rendering writes video/build/timeline/<Scene>.json, which tools/assemble.py uses to lay the audio.

    class ChapterII(Chapter):
        BEATS = ["II.0", "II.1", ...]
        EXTRA = {"II.4": 4.5}           # visual-only seconds after a beat's narration (e.g. a shore cut)
        PRE = {"0.1": 1.5}              # visual-only seconds before a beat's narration starts
        def construct(self):
            self.chapter_card("II.0", "II", "The image", "what a method can say", notch=2)
            self.at(self.cue("II.1", "They form a cone"))
            ...
            self.finish()

The sea: notch levels in LEVELS; chapter cards raise it one notch; the shore shows the four nuts.
"""
import sys

import numpy as np
from manim import *

from look import *
import core as clock  # after the star imports: manim exports a module named core

# ---------------------------------------------------------------- the sea and the shore
SHORE_SLOPE, SHORE_Y0 = 0.20, -3.60                     # shoreline y = SHORE_Y0 + SLOPE * (x + 7.2)
NUT_X = [-5.0, -1.7, 1.6, 4.9]                          # nut icons on the shore, nut 1 lowest
CARD_X = [-5.25, -1.75, 1.75, 5.25]                       # their cards along the top


def shore_y(x):
    return SHORE_Y0 + SHORE_SLOPE * (x + 7.2)


def _nut_level(k):
    return shore_y(NUT_X[k - 1]) + 0.38                 # water just over nut k


# water surface for notch 0 (empty) .. 6 (atlas); notch n >= 2 just covers nut n - 1
LEVELS = [-4.45, -3.45, _nut_level(1), _nut_level(2), _nut_level(3), _nut_level(4), -0.15]
COVERED = 1.6                                           # the coda: the sea over everything on the shore

NUTS = {
    1: ("No exact initialisation lets LoRA reach an update of rank above 2r.",
        "any exact initialisation"),
    2: ("LoRA+ is LoRA with α rescaled.",
        "Adam, ε = 0 (0/0 := 0), no decay, no clipping, B₀ = 0, shared schedule"),
    3: ("GaLore is one-sided LoRA with a moving frame.",
        "gradient-only optimizer (Adam or SGD), no weight decay, no clipping"),
    4: ("Exactly orthogonal methods never change singular values, however often you merge.",
        "exact arithmetic; one orthogonal factor per weight"),
}


def wave_polygon(level, phase, opacity=0.13, color=TIDE, x0=-7.6, x1=7.6, bottom=-4.6, alpha=1.0):
    xs = np.linspace(x0, x1, 96)
    top = [np.array([x, level + 0.06 * np.sin(1.3 * x + phase) + 0.035 * np.sin(2.9 * x - 1.6 * phase), 0])
           for x in xs]
    return Polygon(*top, np.array([x1, bottom, 0]), np.array([x0, bottom, 0]), stroke_color=color, stroke_width=2,
                   stroke_opacity=0.75 * alpha, fill_color=color, fill_opacity=opacity * alpha)


def nut_icon(r=0.24):
    shell, dark = "#B98B52", "#7E5C33"
    body = Ellipse(width=2.3 * r, height=2 * r, fill_color=shell, fill_opacity=1, stroke_color=dark, stroke_width=2)
    seam = Line(body.get_top(), body.get_bottom(), stroke_color=dark, stroke_width=2)
    wr = VGroup(*[Arc(radius=r * k, start_angle=PI / 2 + s * 0.5, angle=s * 1.2, stroke_color=dark, stroke_width=1.2)
                  .move_arc_center_to(body.get_center() + s * r * 0.35 * RIGHT) for k in (0.5, 0.8) for s in (1, -1)])
    return VGroup(body, seam, wr)


def nut_card(k, width=3.3):
    claim, scope = NUTS[k]
    badge = VGroup(Circle(0.16, stroke_color=TIDE, stroke_width=1.6), sans(str(k), 16, TIDE, MEDIUM))
    head = VGroup(badge, kicker("nut", INK3, 14)).arrange(RIGHT, buff=0.12)
    body = paragraph(claim, 25, 18, INK)
    tag = paragraph(scope, 38, 11, INK3, font=SANS)
    col = VGroup(head, body, tag).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    box = RoundedRectangle(corner_radius=0.08, width=width, height=col.height + 0.42, fill_color=BG2,
                           fill_opacity=1, stroke_color=RULE, stroke_width=1.2)
    col.move_to(box).align_to(box, LEFT).shift(0.2 * RIGHT)
    return VGroup(box, col)


def land():
    pts = [np.array([x, shore_y(x), 0]) for x in np.linspace(-7.6, 7.6, 40)]
    return Polygon(*pts, np.array([7.6, -4.6, 0]), np.array([-7.6, -4.6, 0]), fill_color=BG3, fill_opacity=1,
                   stroke_color=RULE, stroke_width=1.5)


# ---------------------------------------------------------------- the chapter scene
class Chapter(ThreeDScene):
    BEATS: list = []
    EXTRA: dict = {}
    WORDLESS: dict = {}
    PRE: dict = {}

    def setup(self):
        super().setup()
        self.camera.background_color = BG
        self.plan, self.total = clock.plan(self.BEATS, self.EXTRA, self.WORDLESS, pre=self.PRE)
        clock.write_timeline(type(self).__name__, self.plan, self.total)

    # -------------------------------------------------------------- time
    def T(self, bid):
        """Start of beat bid (seconds into this scene)."""
        return self.plan[bid]["start"]

    def END(self, bid):
        """End of beat bid, including its EXTRA time."""
        return self.plan[bid]["end"]

    def cue(self, bid, phrase, offset=0.0, end=False):
        """Scene time at which the narration of bid reaches `phrase` (its start, or its end with end=True)."""
        return self.T(bid) + clock.phrase_time(bid, phrase, end) + offset

    def now(self):
        return self.renderer.time

    def at(self, t):
        """Wait until scene time t. Warns when the scene is already late (narration would drift)."""
        dt = t - self.renderer.time
        if dt < -0.3:
            print(f"[timing] {type(self).__name__}: {-dt:.2f} s late for t = {t:.2f}", file=sys.stderr)
        if dt > 1 / 30:
            self.wait(dt)

    def finish(self):
        self.at(self.total)

    # -------------------------------------------------------------- staging helpers
    def hud(self, *mobs):
        """Pin mobjects to the frame (needed for text and 2-D overlays in 3-D shots). Returns them."""
        self.add_fixed_in_frame_mobjects(*mobs)
        return mobs[0] if len(mobs) == 1 else mobs

    def clear_stage(self, run_time=0.6, keep=()):
        """Fade out everything (live seas fade through their alpha tracker, which FadeOut cannot do)."""
        mobs = [m for m in self.mobjects if m not in keep and not isinstance(m, ValueTracker)]
        seas = [m for m in mobs if hasattr(m, "alpha")]
        rest = [m for m in mobs if not hasattr(m, "alpha")]
        anims = [FadeOut(m) for m in rest] + [m.alpha.animate.set_value(0) for m in seas]
        if anims:
            self.play(*anims, run_time=run_time)
        self.remove(*seas)
        self.stop_ambient_camera_rotation()

    def chapter_label(self, numeral, name):
        """Small upper-left label for the chapter, e.g. 'II · THE IMAGE'."""
        lab = kicker(f"{numeral} · {name}", INK3, 15).to_corner(UL, buff=0.42)
        return self.hud(lab)

    def legend(self, active):
        """Upper-right wayfinding: image · fibres · base change, the active one in tide."""
        words = [("image", "image"), ("fibres", "fibres"), ("base change", "base")]
        parts = []
        for k, (w, key) in enumerate(words):
            parts.append(sans(w, 15, TIDE if key == active else INK3, MEDIUM if key == active else NORMAL))
            if k < 2:
                parts.append(sans("·", 15, INK3))
        g = VGroup(*parts).arrange(RIGHT, buff=0.12).to_corner(UR, buff=0.42)
        return self.hud(g)

    def theta0(self, P=ORIGIN, label=True, label_dir=LEFT, color=INK):
        """theta_0 with its halo (fixed orientation, so it faces the camera in 3-D). Returns (dot, halo, label)."""
        dot = Dot3D(P, radius=0.075, color=color, resolution=(10, 10))
        halo = VGroup(*[Circle(radius=r, stroke_width=0, fill_color=TIDE, fill_opacity=o)
                        for r, o in ((0.30, 0.08), (0.20, 0.12), (0.13, 0.18))]).move_to(P)
        lab = math(r"\theta_0", 40, color).move_to(P + 0.5 * label_dir + 0.08 * UP) if label else VGroup()
        self.add_fixed_orientation_mobjects(halo, lab)
        return dot, halo, lab

    def to_3d(self, phi=72, theta=-62, run_time=0.0, **kw):
        if run_time:
            self.move_camera(phi=phi * DEGREES, theta=theta * DEGREES, run_time=run_time, **kw)
        else:
            self.set_camera_orientation(phi=phi * DEGREES, theta=theta * DEGREES, **kw)

    def to_2d(self, run_time=0.0):
        self.to_3d(0, -90, run_time)

    # -------------------------------------------------------------- the sea
    def sea(self, level=LEVELS[0], opacity=0.13, alpha=1.0):
        """A live sea pinned to the frame. Returns (mobject, level ValueTracker); animate the tracker to move it.
        Fade it with poly.alpha (a ValueTracker, 0..1), never with FadeIn/FadeOut: its updater rebuilds it each
        frame, so opacity animations on the polygon itself do nothing. clear_stage() handles this."""
        tracker, fade = ValueTracker(level), ValueTracker(alpha)
        poly = wave_polygon(level, 0, opacity, alpha=alpha)
        poly.alpha = fade
        # dt in the signature keeps Manim from treating waits as static frames, so the waves keep moving
        poly.add_updater(lambda m, dt: m.become(wave_polygon(tracker.get_value(), 1.2 * self.renderer.time,
                                                             opacity, alpha=fade.get_value())))
        self.add_fixed_in_frame_mobjects(poly)
        return poly, tracker

    def chapter_card(self, bid, numeral, title, subtitle, notch):
        """The whole wordless beat bid: card in, sea rises from notch-1 to notch, hold, both out. Stage must be clear."""
        self.at(self.T(bid))
        dur = self.plan[bid]["dur"]
        poly, level = self.sea(LEVELS[max(notch - 1, 0)], alpha=0.0)
        card = VGroup(kicker(f"Exposé {numeral}" if numeral else "", TIDE, 20),
                      serif(title, 66, INK, SEMIBOLD), serif(subtitle, 34, INK2, italic=True))
        card.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(2.4 * LEFT + 1.0 * UP)
        self.hud(card)
        card.set_opacity(0)
        self.play(card.animate.set_opacity(1), poly.alpha.animate.set_value(1), run_time=0.8)
        self.play(level.animate.set_value(LEVELS[notch]), run_time=2.2, rate_func=smooth)
        self.wait(max(dur - 0.8 - 2.2 - 0.7, 0.05))
        self.play(FadeOut(card), poly.alpha.animate.set_value(0), run_time=0.7)
        self.remove(poly)

    def shore(self, level, dissolved=()):
        """Build the shore: land, nut icons and nut cards (minus dissolved ones), and a live sea at `level`.
        Returns dict(land, icons, cards, links, sea, level) with icons/cards/links keyed by nut number."""
        ground = land()
        icons, cards, links = {}, {}, {}
        for k in (1, 2, 3, 4):
            if k in dissolved:
                continue
            ic = nut_icon().move_to([NUT_X[k - 1], shore_y(NUT_X[k - 1]) + 0.2, 0])
            cd = nut_card(k).move_to([CARD_X[k - 1], 0, 0])
            cd.shift((3.3 - cd.get_top()[1]) * UP)     # tops aligned
            ln = DashedLine(cd.get_bottom(), ic.get_top() + 0.08 * UP, color=RULE, stroke_width=1.2,
                            dash_length=0.06)
            icons[k], cards[k], links[k] = ic, cd, ln
        self.hud(ground, *icons.values(), *links.values(), *cards.values())
        poly, tracker = self.sea(level, opacity=0.22)
        return dict(land=ground, icons=icons, cards=cards, links=links, sea=poly, level=tracker)

    def dissolve_nut(self, s, k, rise_to, run_time=3.0):
        """Raise the sea to `rise_to` and dissolve nut k (icon sinks in bubbles, card fades upward)."""
        ic, cd, ln = s["icons"][k], s["cards"][k], s["links"][k]
        bubbles = VGroup(*[Circle(radius=r, stroke_color=TIDE, stroke_width=1.2, stroke_opacity=0.8)
                           .move_to(ic.get_center() + np.array([dx, 0, 0]))
                           for r, dx in ((0.05, -0.15), (0.035, 0.05), (0.06, 0.18), (0.03, -0.02))])
        self.play(s["level"].animate.set_value(rise_to), run_time=run_time * 0.55, rate_func=smooth)
        self.hud(bubbles)
        self.play(ic.animate.scale(0.2).set_opacity(0), FadeOut(ln),
                  cd.animate.shift(0.25 * UP).set_opacity(0.0),
                  LaggedStart(*[b.animate.shift(0.9 * UP).set_stroke(opacity=0) for b in bubbles], lag_ratio=0.15),
                  run_time=run_time * 0.45)
        self.remove(ic, cd, ln, bubbles)
