"""One 24-second beat of the explainer in three visual styles, timed to video/audio/style-demo.wav.

  PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -qh --fps 30 --media_dir video/build \
      video/styles/style_demo.py Chalk Atlas Sea
"""
import os

import manimpango
import numpy as np
from manim import *

TEXFONTS = "/usr/local/texlive/2026/texmf-dist/fonts/opentype"
for f in ("adobe/sourceserif/SourceSerif4-Regular.otf", "adobe/sourceserif/SourceSerif4-Semibold.otf",
          "adobe/sourceserif/SourceSerif4-RegularIt.otf", "ibm/plex/IBMPlexSans-Regular.otf",
          "ibm/plex/IBMPlexSans-Medium.otf", "ibm/plex/IBMPlexMono-Regular.otf"):
    if os.path.exists(os.path.join(TEXFONTS, f)):
        manimpango.register_font(os.path.join(TEXFONTS, f))

# Sentence onsets in style-demo.wav, read off ffmpeg silencedetect (seconds).
CUE = dict(title=0.27, point=2.80, theta=3.56, pretrained=4.87, method=7.04,
           lora=10.02, apex=12.9, oft=15.04, ia3=18.59, end=22.44, out=24.0)

STYLES = {
    # 3Blue1Brown-like: charcoal board, Computer Modern, primary accents
    "Chalk": dict(bg="#16181D", ink="#ECE8DF", dim="#8C919B", faint="#2A2E36",
                  lora="#58C4DD", oft="#F4D35E", ia3="#83C167", hot="#FC6255",
                  serif=False, grid=False, sea=False, contours=False, glow=False, halo=False, card=False),
    # the site and the paper: paper, ink, tide; Source Serif 4 and IBM Plex
    "Atlas": dict(bg="#F2F4F1", ink="#13233A", dim="#5E6C7B", faint="#D3DBD4",
                  lora="#0E7385", oft="#B07F22", ia3="#5E6C7B", hot="#A3392B",
                  serif=True, grid=True, sea=False, contours=False, glow=False, halo=False, card=False),
    # the social card: deep navy, contours, glowing tide and gold, a rising sea
    "Sea": dict(bg="#0A121C", ink="#E3E9EE", dim="#8B98A6", faint="#1A2A3F",
                lora="#4CC3CF", oft="#E2B04C", ia3="#A3B0BD", hot="#EC7A66",
                serif=True, grid=True, sea=True, contours=True, glow=True, halo=True, card=False),
    # C, quiet: the same palette with no wallpaper; the sea appears only between chapters
    "Quiet": dict(bg="#0A121C", ink="#E3E9EE", dim="#8B98A6", faint="#1A2A3F",
                  lora="#4CC3CF", oft="#E2B04C", ia3="#A3B0BD", hot="#EC7A66",
                  serif=True, grid=False, sea=False, contours=False, glow=False, halo=True, card=True),
}

PHI, THETA = 72 * DEGREES, -62 * DEGREES
RIGHT_S = np.array([-np.sin(THETA), np.cos(THETA), 0.0])   # screen-right at the opening camera angle
Z = OUT
P0 = 2.2 * RIGHT_S + 0.2 * Z                               # theta_0, right of centre on screen
P0_SCREEN = np.array([2.2, 0.2 * np.sin(PHI), 0.0])
H, R = 1.7, 1.15                                           # cone half-height and rim radius
ZERO = P0 + 1.9 * RIGHT_S - 1.6 * Z                        # the zero weight, lower right of theta_0
RAD = np.linalg.norm(P0 - ZERO)
U = (P0 - ZERO) / RAD                                      # from 0 towards theta_0
_t = RIGHT_S + 0.9 * Z
TAN = (_t - np.dot(_t, U) * U) / np.linalg.norm(_t - np.dot(_t, U) * U)


def orbit(a):
    return ZERO + RAD * (np.cos(a) * U + np.sin(a) * TAN)


def ia3(s):
    return ZERO + s * (P0 - ZERO)


class StyleDemo(ThreeDScene):
    STYLE = "Chalk"

    # ---------------------------------------------------------------- helpers
    def at(self, t):
        dt = t - self.renderer.time
        if dt > 1e-3:
            self.wait(dt)

    def tex_template(self):
        tpl = TexTemplate()
        if self.s["serif"]:
            tpl.add_to_preamble(r"\usepackage{sourceserif}\usepackage[varbb,smallerops,cmintegrals]{newtxmath}")
        return tpl

    def math(self, src, color, size=40):
        return MathTex(src, color=color, font_size=size, tex_template=self.tpl)

    def words(self, txt, color, size=26, weight=NORMAL, ui=True):
        if not self.s["serif"]:
            txt = txt.replace("³", r"$^{3}$")
            body = r"\textbf{%s}" % txt if weight != NORMAL else txt
            return Tex(body, color=color, font_size=size * 1.25, tex_template=self.tpl)
        return Text(txt, color=color, font_size=size, weight=weight,
                    font="IBM Plex Sans" if ui else "Source Serif 4")

    def label(self, name, rest, color):
        parts = [self.words(name, color, 26, MEDIUM)]
        for piece in rest:
            parts.append(self.math(piece[1:-1], self.s["dim"], 30) if piece.startswith("$")
                         else self.words(piece, self.s["dim"], 22))
        return VGroup(*parts).arrange(RIGHT, buff=0.14, aligned_edge=DOWN)

    def glowing(self, mob, color, width):
        if not self.s["glow"]:
            return mob
        halo = mob.copy().set_stroke(color, width * 4.5, opacity=0.14)
        return VGroup(halo, mob)

    def fade_in_fixed(self, mob, run_time=0.8):
        mob.set_opacity(0)
        self.add_fixed_orientation_mobjects(mob)
        self.play(mob.animate.set_opacity(1), run_time=run_time)

    # ---------------------------------------------------------------- backdrop
    def backdrop(self):
        s, layers = self.s, VGroup()
        if s["contours"]:
            for k in range(1, 9):
                e = Ellipse(width=2.2 * k, height=1.15 * k, color=s["faint"], stroke_width=1.3)
                e.apply_function(lambda p, k=k: p + 0.06 * k * np.array([np.sin(1.7 * p[1]), np.cos(1.3 * p[0]), 0]))
                layers.add(e.move_to(P0_SCREEN + 0.4 * DOWN))
        if s["grid"]:
            step = 0.8
            for x in np.arange(-7.2, 7.3, step):
                layers.add(Line([x, -4.2, 0], [x, 4.2, 0], color=s["faint"], stroke_width=0.8,
                                stroke_opacity=0.55 if s["sea"] else 0.9))
            for y in np.arange(-4.0, 4.1, step):
                layers.add(Line([-7.3, y, 0], [7.3, y, 0], color=s["faint"], stroke_width=0.8,
                                stroke_opacity=0.55 if s["sea"] else 0.9))
        return layers

    def sea(self):
        s = self.s

        def wave(level, phase):
            xs = np.linspace(-7.4, 7.4, 90)
            top = [np.array([x, level + 0.07 * np.sin(1.3 * x + phase) + 0.04 * np.sin(2.9 * x - 1.6 * phase), 0])
                   for x in xs]
            pts = top + [np.array([7.4, -4.3, 0]), np.array([-7.4, -4.3, 0])]
            return Polygon(*pts, stroke_color=s["lora"], stroke_width=2, stroke_opacity=0.7,
                           fill_color=s["lora"], fill_opacity=0.13)

        band = wave(-4.25, 0)
        band.add_updater(lambda m: m.become(wave(-4.25 + 1.0 * min(1, self.renderer.time / CUE["out"]),
                                                1.4 * self.renderer.time)))
        return band

    # ---------------------------------------------------------------- the shot
    def construct(self):
        s = self.s = STYLES[self.STYLE]
        self.tpl = self.tex_template()
        self.camera.background_color = s["bg"]
        self.set_camera_orientation(phi=PHI, theta=THETA)

        back = self.backdrop()
        if len(back):
            self.add_fixed_in_frame_mobjects(back)
        if s["sea"]:
            self.add_fixed_in_frame_mobjects(self.sea())

        # heads-up column on the left
        title = self.words("Fine-tuning moves a point.", s["ink"], 42, SEMIBOLD, ui=False) if s["serif"] \
            else Tex("Fine-tuning moves a point.", color=s["ink"], font_size=54)
        title.to_corner(UL, buff=0.75).shift(0.25 * DOWN)
        hud = [title]
        if s["serif"]:
            kicker = self.words("EXPOSÉ I  ·  THE SLICE", s["lora"], 18, MEDIUM)
            kicker.next_to(title, UP, buff=0.22, aligned_edge=LEFT)
            hud.append(kicker)
        caption = VGroup(self.math(r"\theta_0", s["lora"], 40),
                         self.words("the pretrained model", s["dim"], 26, ui=False)
                         if s["serif"] else Tex("the pretrained model", color=s["dim"], font_size=36))
        caption.arrange(RIGHT, buff=0.22).next_to(title, DOWN, buff=0.55, aligned_edge=LEFT)
        formula = self.math(r"\rho:(Q,q_0)\longrightarrow(\Theta,\theta_0),\qquad \rho(q_0)=\theta_0", s["ink"], 40)
        formula.to_corner(DL, buff=0.75).shift(1.05 * UP)

        self.at(CUE["title"])
        self.add_fixed_in_frame_mobjects(*hud)
        self.play(*[Write(m) for m in hud], run_time=1.6)

        # theta_0
        self.at(CUE["point"])
        p0 = Dot3D(P0, radius=0.075, color=s["ink"], resolution=(10, 10))
        halo = VGroup(*[Circle(radius=r, stroke_width=0, fill_color=s["lora"], fill_opacity=o)
                        for r, o in ((0.30, 0.08), (0.20, 0.12), (0.13, 0.18))]).move_to(P0) if s["halo"] else VGroup()
        self.add_fixed_orientation_mobjects(halo)
        self.play(FadeIn(p0, scale=0.3), FadeIn(halo), run_time=0.7)
        th = self.math(r"\theta_0", s["ink"], 44).move_to(P0 - 0.48 * RIGHT_S + 0.1 * Z)
        self.at(CUE["theta"])
        self.fade_in_fixed(th, 0.6)
        self.at(CUE["pretrained"])
        self.add_fixed_in_frame_mobjects(caption)
        self.play(FadeIn(caption, shift=0.15 * UP), run_time=0.8)

        # a method is a pointed map
        self.at(CUE["method"])
        self.add_fixed_in_frame_mobjects(formula)
        self.play(Write(formula), run_time=1.8)

        # LoRA: the cone theta_0 + M_r, apex at theta_0
        self.at(CUE["lora"])
        angles = np.linspace(0, TAU, 14, endpoint=False)
        rulings = VGroup(*[Line(P0, P0 + np.array([R * np.cos(a), R * np.sin(a), sgn * H]),
                                color=s["lora"], stroke_width=1.3, stroke_opacity=0.75)
                           for sgn in (1, -1) for a in angles])
        rims = VGroup(*[ParametricFunction(lambda a, z=z: P0 + np.array([R * np.cos(a), R * np.sin(a), z]),
                                           t_range=[0, TAU], color=s["lora"], stroke_width=2.4)
                        for z in (H, -H)])
        skin = Surface(lambda u, v: P0 + np.array([u * R / H * np.cos(v), u * R / H * np.sin(v), u]),
                       u_range=[-H, H], v_range=[0, TAU], resolution=(6, 28), checkerboard_colors=False,
                       fill_color=s["lora"], fill_opacity=0.08 if s["bg"] != "#F2F4F1" else 0.06, stroke_width=0)
        lora_lab = self.label("LoRA", ["cone, apex at", "$\\theta_0$"], s["lora"]).move_to(P0 + 0.35 * RIGHT_S + (H + 0.62) * Z)
        self.play(LaggedStart(*[Create(l) for l in rulings], lag_ratio=0.03), run_time=1.6)
        self.play(Create(self.glowing(rims, s["lora"], 2.4)), FadeIn(skin), run_time=1.0)
        rulings.add_updater(lambda m, dt: m.rotate(0.16 * dt, axis=Z, about_point=P0))
        self.fade_in_fixed(lora_lab, 0.6)
        self.at(CUE["apex"])
        ring = Circle(radius=0.12, color=s["hot"], stroke_width=3.5).move_to(P0)
        self.add_fixed_orientation_mobjects(ring)
        self.play(ring.animate.scale(4.2).set_stroke(opacity=0), run_time=1.0)
        self.remove(ring)

        # OFT: the orbit of theta_0, an arc of the circle about 0
        self.at(CUE["oft"])
        arc = ParametricFunction(orbit, t_range=[-0.85, 0.8], color=s["oft"], stroke_width=3)
        rest = DashedVMobject(ParametricFunction(orbit, t_range=[0.8, TAU - 0.85]), num_dashes=70, dashed_ratio=0.45)
        rest.set_stroke(s["oft"], 1.6, opacity=0.4)
        oft_lab = self.label("OFT", ["orbit of", "$\\theta_0$"], s["oft"]).move_to(orbit(0.95) + 0.45 * Z + 0.4 * RIGHT_S)
        self.play(Create(self.glowing(arc, s["oft"], 3)), Create(rest), run_time=1.6)
        self.fade_in_fixed(oft_lab, 0.5)
        rider = Dot3D(P0, radius=0.065, color=s["oft"], resolution=(8, 8))
        self.play(MoveAlongPath(rider, ParametricFunction(orbit, t_range=[0, 0.6])), run_time=1.1, rate_func=smooth)

        # (IA)^3: the line through 0 and theta_0
        self.at(CUE["ia3"])
        line = Line(ia3(-0.04), ia3(1.55), color=s["ia3"], stroke_width=2.4)
        plus = self.math("+", s["dim"], 40).move_to(ZERO)
        zero = self.math("0", s["dim"], 34).move_to(ZERO + 0.35 * RIGHT_S - 0.15 * Z)
        ia3_lab = self.label("(IA)³", ["line through", "$\\theta_0$"], s["ia3"]).move_to(ia3(1.55) - 1.55 * RIGHT_S + 0.12 * Z)
        self.play(Create(self.glowing(line, s["ia3"], 2.4)), run_time=1.4)
        self.add_fixed_orientation_mobjects(plus, zero)
        self.play(FadeIn(plus), FadeIn(zero), run_time=0.4)
        self.fade_in_fixed(ia3_lab, 0.5)
        slider = Dot3D(P0, radius=0.065, color=s["ia3"], resolution=(8, 8))
        self.play(FadeOut(rider), MoveAlongPath(slider, Line(P0, ia3(0.62))), run_time=1.2, rate_func=smooth)

        self.at(CUE["out"])
        if s["card"]:
            self.chapter_card()

    def chapter_card(self):
        """Between chapters the stage clears and the sea rises one notch."""
        s = self.s
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.8)
        level = ValueTracker(-4.7)

        def wave():
            lv, ph = level.get_value(), 1.2 * self.renderer.time
            top = [np.array([x, lv + 0.06 * np.sin(1.3 * x + ph) + 0.035 * np.sin(2.9 * x - 1.6 * ph), 0])
                   for x in np.linspace(-7.4, 7.4, 90)]
            return Polygon(*top, np.array([7.4, -4.4, 0]), np.array([-7.4, -4.4, 0]),
                           stroke_color=s["lora"], stroke_width=2, stroke_opacity=0.75,
                           fill_color=s["lora"], fill_opacity=0.12)

        sea = wave()
        sea.add_updater(lambda m: m.become(wave()))
        kicker = self.words("EXPOSÉ II", s["lora"], 20, MEDIUM)
        title = self.words("The image", s["ink"], 66, SEMIBOLD, ui=False)
        sub = Text("what a method can say", font="Source Serif 4", slant=ITALIC, color=s["dim"], font_size=34)
        card = VGroup(kicker, title, sub).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(2.6 * LEFT + 0.7 * UP)
        self.add_fixed_in_frame_mobjects(sea, card)
        self.play(FadeIn(card, shift=0.2 * UP), level.animate.set_value(-2.95), run_time=2.6, rate_func=smooth)
        self.wait(1.6)


class Chalk(StyleDemo):
    STYLE = "Chalk"


class Atlas(StyleDemo):
    STYLE = "Atlas"


class Sea(StyleDemo):
    STYLE = "Sea"


class Quiet(StyleDemo):
    STYLE = "Quiet"
