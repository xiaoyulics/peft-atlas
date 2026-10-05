import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *

"""Chapter II · The image: what a method can say (beats II.0 to II.5).

  PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -ql --disable_caching --media_dir video/build/TheImage \
      video/film/ch2_image.py TheImage                       # preview; -qh --fps 30 for the final
  CH2_ONLY=II.3,II.4 ...                                     # render only these beats (the clock still runs)

Every number on screen is read from video/sim/out/*.json, written by first_order.py (II.3), repoint_rank.py (II.4)
and cut_volume.py (II.5), which run when this file is imported.
"""
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim"


def _sim(name):
    subprocess.run([sys.executable, str(SIM / f"{name}.py")], check=True, capture_output=True)
    return json.loads((SIM / "out" / f"{name}.json").read_text())


FO, RR, CV = _sim("first_order"), _sim("repoint_rank"), _sim("cut_volume")

# ---------------------------------------------------------------- the 3-D shot (camera of the Quiet style sample)
PHI, THETA = 72 * DEGREES, -62 * DEGREES
RS = np.array([-np.sin(THETA), np.cos(THETA), 0.0])     # screen-right at this camera
Z, EX = OUT, RIGHT                                       # the cone's axis; the coordinate axis e_1

# II.1: cone, orbit, line and BitFit around theta_0 (the Quiet shot, moved left to make room for labels)
P0 = 0.7 * RS + 0.45 * Z
H1, R1 = 1.7, 1.15
ZERO = P0 + 1.9 * RS - 1.6 * Z                           # the zero weight
RAD = np.linalg.norm(P0 - ZERO)
U_ = (P0 - ZERO) / RAD
_t = RS + 0.9 * Z
TAN = (_t - np.dot(_t, U_) * U_) / np.linalg.norm(_t - np.dot(_t, U_) * U_)

# II.3 and II.4: one cone, right of centre
PB = 2.3 * RS + 0.32 * Z
H, R = 1.75, 1.2
A_S, H_S = -45 * DEGREES, 0.6 * 1.75                     # the smooth point S (later the split initialisation)
B_TIP = -150 * DEGREES                                   # the ruling that carries S_1 at the tip (left silhouette)
PLANE_U, PLANE_V = (-0.4, 0.9), (-0.35, 1.15)            # the part of S's tangent plane that is drawn


def orbit(a):
    return ZERO + RAD * (np.cos(a) * U_ + np.sin(a) * TAN)


def ia3(s):
    return ZERO + s * (P0 - ZERO)


def cone_pt(P, h, a, HH=H, RR_=R):
    """The point of the double cone with apex P at signed height h and azimuth a."""
    return P + np.array([RR_ / HH * abs(h) * np.cos(a), RR_ / HH * abs(h) * np.sin(a), h])


def ruling(a, HH=H, RR_=R):
    v = np.array([RR_ / HH * np.cos(a), RR_ / HH * np.sin(a), 1.0])
    return v / np.linalg.norm(v)


def ghost_cone(V, n=10, opacity=0.32, color=TIDE, HH=H, RR_=R):
    """A translated copy of the cone (vertex V): rulings and rims only, faint."""
    angles = np.linspace(0, TAU, n, endpoint=False) + 0.2
    rul = VGroup(*[Line(V, V + np.array([RR_ * np.cos(a), RR_ * np.sin(a), s * HH]), color=color, stroke_width=1.0,
                        stroke_opacity=opacity) for s in (1, -1) for a in angles])
    rims = VGroup(*[ParametricFunction(lambda a, z=z: V + np.array([RR_ * np.cos(a), RR_ * np.sin(a), z]),
                                       t_range=[0, TAU], color=color, stroke_width=1.4, stroke_opacity=opacity * 1.6)
                    for z in (HH, -HH)])
    return VGroup(rul, rims)


# ---------------------------------------------------------------- 2-D parts
def label2(name, rest, formula=None, color=INK, align=LEFT):
    """'Name  descriptor' over an optional formula line (the 3-D labels of II.1)."""
    top = [sans(name, 24, color, MEDIUM)]
    for piece in rest:
        top.append(math(piece[1:-1], 28, INK2) if piece.startswith("$") else sans(piece, 20, INK2))
    l1 = VGroup(*top).arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
    if formula is None:
        return VGroup(l1)
    l2 = formula if isinstance(formula, Mobject) else math(formula, 28, INK2)
    return VGroup(l1, l2).arrange(DOWN, aligned_edge=align, buff=0.12)


def namecard(title, sub=None, width=3.6, height=1.45, tcolor=INK, tsize=40):
    box = RoundedRectangle(corner_radius=0.12, width=width, height=height, fill_color=BG2, fill_opacity=1,
                           stroke_color=RULE, stroke_width=1.4)
    t = serif(title, tsize, tcolor, SEMIBOLD)
    g = VGroup(t) if sub is None else VGroup(t, sub).arrange(DOWN, buff=0.16)
    g.move_to(box)
    return VGroup(box, g)


def wire(a, b, w=4.0, color=INK2):
    return Line(a, b, color=color, stroke_width=w)


def box(label, center, w=0.95, h=0.78, color=INK):
    r = RoundedRectangle(corner_radius=0.08, width=w, height=h, fill_color=BG2, fill_opacity=1, stroke_color=color,
                         stroke_width=2.0).move_to(center)
    return VGroup(r, math(label, 34, color).move_to(center))


# ---------------------------------------------------------------- the paper cards (statements: theory/propositions.json)
CARD = {
    "II.8": dict(
        label="Proposition II.8 · DoRA is LoRA followed by an output scale",
        body=r"For a linear layer whose frozen weight $W_0$ has no zero rows, and $r\le\min(m,n)$, DoRA reaches "
             r"exactly the same merged weights as rank-$r$ LoRA followed by a free gain per output neuron, "
             r"$\{\operatorname{diag}(\ell)\,(W_0+X):\operatorname{rank}X\le r\}$. Any difference from that baseline "
             r"comes from training dynamics.",
        credit=r"Scope: DoRA as implemented in HF PEFT (one magnitude per output neuron); $W_0$ with no zero rows; "
               r"merged weights."),
    "II.2": dict(
        label="Theorem II.2 · three strata of LoRA initialisations",
        body=r"Zero-factor initialisations of LoRA ($B_0=0$ or $A_0=0$) sit at the apex of "
             r"$\theta_0+\mathcal M_{\le r}$, with $r(n-r)$ or $r(m-r)$ fewer first-order directions. Split "
             r"initialisations sit at a smooth point. Neither image contains the other.",
        credit=r"For $r<\min(m,n)$ and full-rank factors; set-level, in exact arithmetic."),
    "II.7": dict(
        label="Theorem II.7 · HRA starts at an apex",
        body=r"At Hugging Face's default initialisation (reflections in equal pairs), with $r\le n-2$ even and "
             r"$W_0$ of full column rank, $W_0$ is a singular vertex of HRA's image: an apex, as for zero-init LoRA.",
        credit=r"HRA: Yuan, Liu \& Xu 2024. The paired initialisation is HF's default and is not stated in the HRA "
               r"paper."),
    "II.1": dict(
        label="Theorem II.1 · pointings of an additive method",
        body=r"Start an additive method with shape $C$ at any $q$ and subtract $\delta(q)$ from the frozen weights: "
             r"it is still pointed at $\theta_0$, and together these images cover $\theta_0+(C-C)$. For "
             r"LoRA$_r$, $C-C$ is the matrices of rank at most $\min(2r,m,n)$, so no exact initialisation, zero or "
             r"split, reaches an update of rank above $2r$. Split initialisations attain $2r$ when "
             r"$2r\le\min(m,n)$.",
        credit=r"known in practice: converting a PiSSA adapter to plain LoRA takes rank $2r$ (Meng, Wang \& Zhang "
               r"2024; HF PEFT) $\cdot$ new here: $2r$ is needed in general, and the argument covers every "
               r"additive method"),
    "II.10": dict(
        label="Theorem II.10 (standard) · cut-rank bound",
        body=r"For a tensor network evaluated in finite-dimensional vector spaces, the rank of the map it computes is "
             r"at most the smallest product of wire dimensions across any cut that separates inputs from outputs.",
        credit=r"The bound refers to a chosen presentation of the network."),
    "II.12": dict(
        label="Proposition II.12 · rank is a cut, dimension is a volume",
        body=r"At equal budget $2r(m+n)$, LoHa$_{r,r}$ has dimension at most $2r(m+n-r)-(m+n-1)$, against "
             r"$2r(m+n-2r)$ for LoRA$_{2r}$. When $2r^2<m+n-1$, LoHa reaches at least $(m+n-1)-2r^2$ fewer "
             r"dimensions: a trade of dimension for rank (up to $r^2$) when $r\ge3$, and strictly dominated when "
             r"$r\le2$.",
        credit=r"For $2r\le\min(m,n)$ and $r^2\le\min(m,n)$. When $2r^2\ge m+n-1$ the bound no longer separates "
               r"the two."),
}


def card(key, width_cm=8.0, body_size=26):
    c = CARD[key]
    return paper_card(c["label"], c["body"], credit=c["credit"], width_cm=width_cm, body_size=body_size)


# ---------------------------------------------------------------- the chapter
class TheImage(Chapter):
    BEATS = ["II.0", "II.1", "II.2", "II.3", "II.4", "II.5"]
    EXTRA = {"II.2": 2.0, "II.3": 1.5, "II.4": 4.5, "II.5": 2.5}

    # -------------------------------------------------------------- helpers
    def section(self, bid):
        only = os.environ.get("CH2_ONLY")
        skip = bool(only) and bid not in only.split(",")
        self.next_section(bid, skip_animations=skip)
        if not skip:
            print(f"[section] {bid} renders from scene time {self.renderer.time:.3f}", file=sys.stderr)

    def scr(self, p):
        """Screen position (frame units) of a 3-D point at the current camera."""
        return self.renderer.camera.project_point(np.array(p, dtype=float)) * np.array([1.0, 1.0, 0.0])

    def cam3d(self):
        self.to_3d(PHI / DEGREES, THETA / DEGREES)
        self.renderer.camera.reset_rotation_matrix()

    def cam2d(self):
        self.to_2d()
        self.renderer.camera.reset_rotation_matrix()

    def show(self, *mobs, run_time=0.6, shift=None, lag=0.0):
        """Pin 2-D overlays to the frame and fade them in."""
        self.hud(*mobs)
        anims = [FadeIn(m, shift=shift) if shift is not None else FadeIn(m) for m in mobs]
        self.play(LaggedStart(*anims, lag_ratio=lag) if lag else AnimationGroup(*anims), run_time=run_time)

    def pin(self, *mobs):
        """Fixed-orientation (faces the camera) 3-D markers such as dots and rings."""
        self.add_fixed_orientation_mobjects(*mobs)
        return mobs[0] if len(mobs) == 1 else mobs

    def theta_mark(self, P):
        """theta_0's halo (from the library) and a flat dot drawn above the lines, as one mobject in the scene."""
        _, halo, _ = self.theta0(P, label=False)
        cap = self.pin(Dot(P, radius=0.075, color=INK))
        self.remove(halo, cap)
        g = VGroup(halo, cap)
        self.add(g)
        return g

    def pulse(self, P, color=CORAL):
        ring = self.pin(Circle(radius=0.12, color=color, stroke_width=3.5).move_to(P))
        self.play(ring.animate.scale(4.2).set_stroke(opacity=0), run_time=1.0)
        self.remove(ring)

    # -------------------------------------------------------------- the film
    def construct(self):
        self.section("II.0")
        self.chapter_card("II.0", "II", "The image", "what a method can say", notch=2)
        self.ii1()
        self.ii2()
        self.ii3()
        self.ii4()
        self.ii5()
        self.finish()

    # ============================================================== II.1 cone, orbit, line
    def ii1(self):
        b = "II.1"
        self.section(b)
        self.at(self.T(b))
        self.cam3d()
        scr = self.scr
        self.at(self.cue(b, "Let's look at images."))
        self.lab = self.chapter_label("II", "the image")
        self.leg = self.legend("image")
        th0 = self.theta_mark(P0)
        th_lab = self.hud(math(r"\theta_0", 40).move_to(scr(P0) + np.array([0.72, 0.25, 0])))
        self.play(FadeIn(self.lab), FadeIn(self.leg), FadeIn(th0), FadeIn(th_lab), run_time=0.9)

        # LoRA: the formula, then the cone
        f1, f2 = math(r"W_0+BA,", 28, INK2), math(r"\operatorname{rank}BA\le r", 28, INK2)
        lora_lab = label2("LoRA", ["cone, apex at", "$\\theta_0$"], VGroup(f1, f2).arrange(RIGHT, buff=0.2), TIDE)
        lora_lab.move_to(scr(P0 + (H1 + 0.1) * Z) + np.array([0.05, 0.85, 0]))
        self.at(self.cue(b, "LoRA adds a low-rank matrix"))
        self.show(f1, shift=0.1 * UP, run_time=0.7)
        self.at(self.cue(b, "the matrices of rank at most r"))
        self.show(f2, shift=0.1 * UP, run_time=0.7)
        self.at(self.cue(b, "They form a cone"))
        rulings, rims, skin = cone(P0, H1, R1)
        self.play(Create(rulings, lag_ratio=0.03), run_time=1.4)
        self.play(Create(rims), FadeIn(skin), run_time=0.8)
        self.bring_to_front(th0)
        rulings.add_updater(lambda m, dt: m.rotate(0.16 * dt, axis=Z, about_point=P0))   # the cone turns slowly
        self.show(lora_lab[0], run_time=0.6)
        self.at(self.cue(b, "its tip sits right at theta-zero"))
        self.pulse(P0)

        # OFT: an arc of the circle about 0 through theta_0
        self.at(self.cue(b, "OFT rotates the weights"))
        arc = ParametricFunction(orbit, t_range=[-0.85, 0.8], color=GOLD, stroke_width=3)
        rest = DashedVMobject(ParametricFunction(orbit, t_range=[0.8, TAU - 0.85]), num_dashes=70, dashed_ratio=0.45)
        rest.set_stroke(GOLD, 1.6, opacity=0.4)
        oft_lab = label2("OFT", ["orbit of", "$\\theta_0$"], r"W_0R,\ \ R\ \text{a rotation}", GOLD)
        oft_lab.next_to(scr(orbit(0.8)), RIGHT, buff=0.25).shift(0.45 * UP)
        self.play(Create(arc), Create(rest), run_time=1.6)
        self.bring_to_front(th0)
        self.show(oft_lab, run_time=0.5)
        self.at(self.cue(b, "so it moves theta-zero along an orbit"))
        rider = self.pin(Dot(P0, radius=0.065, color=GOLD))
        self.play(MoveAlongPath(rider, ParametricFunction(orbit, t_range=[0, 0.62])), run_time=1.6, rate_func=smooth)

        # (IA)^3: the line through 0 and theta_0 (schematic)
        self.at(self.cue(b, "I-A-cubed rescales neurons"))
        line = Line(ia3(-0.04), ia3(1.55), color=INK2, stroke_width=2.4)
        zero = math("0", 30, INK2).move_to(scr(ZERO) + np.array([-0.3, -0.26, 0]))
        ia3_lab = label2("(IA)³", ["line through 0 and", "$\\theta_0$"], r"\operatorname{diag}(\ell)\,W_0", INK2,
                         align=RIGHT)
        ia3_lab.add(sans("schematic: in general an m-dimensional flat through 0", 16, INK3))
        ia3_lab.arrange(DOWN, aligned_edge=RIGHT, buff=0.1)
        ia3_lab.next_to(scr(ia3(1.55)), LEFT, buff=0.2).shift(0.2 * UP)
        self.play(Create(line), FadeOut(rider), run_time=1.2)
        self.bring_to_front(th0)
        self.show(zero, ia3_lab, run_time=0.6)
        self.at(self.cue(b, "it slides along a line through the origin"))
        slider = self.pin(Dot(P0, radius=0.065, color=INK2))
        self.play(MoveAlongPath(slider, Line(P0, ia3(0.6))), run_time=1.6, rate_func=smooth)

        # BitFit: parallel to a coordinate axis (the axes drawn at 0)
        self.at(self.cue(b, "And BitFit"))
        axes = VGroup(Line(ZERO, ZERO + 0.9 * EX, color=LEAF, stroke_width=2.4),
                      Line(ZERO, ZERO + 0.9 * UP, color=INK3, stroke_width=1.8),
                      Line(ZERO, ZERO + 0.9 * Z, color=INK3, stroke_width=1.8))
        bit = Line(P0 - 1.3 * EX, P0 + 1.3 * EX, color=LEAF, stroke_width=2.6)
        bit_lab = label2("BitFit", ["parallel to a coordinate axis"], r"\text{biases only: }b_0+\beta", LEAF,
                         align=RIGHT)
        bit_lab.next_to(scr(P0 - 1.3 * EX), LEFT, buff=0.2).shift(0.42 * DOWN)
        self.play(FadeOut(slider), LaggedStart(*[Create(a) for a in axes], lag_ratio=0.25), run_time=0.9)
        self.play(Create(bit), run_time=1.0)
        self.bring_to_front(th0)
        self.show(bit_lab, run_time=0.6)
        self.at(self.cue(b, "moves parallel to a few coordinate axes"))
        ghost = axes[0].copy()
        self.play(ghost.animate.shift(P0 - ZERO - 0.45 * EX), run_time=1.1)
        self.play(FadeOut(ghost), run_time=0.3)
        self.remove(ghost)
        mover = self.pin(Dot(P0, radius=0.065, color=LEAF))
        self.play(MoveAlongPath(mover, Line(P0, P0 + 1.0 * EX)), run_time=1.3, rate_func=smooth)

        self.at(self.END(b))
        self.clear_stage(run_time=0.5, keep=(self.lab, self.leg))
        self.cam2d()

    # ============================================================== II.2 disguises
    def ii2(self):
        b = "II.2"
        self.section(b)
        y = 1.85
        self.at(self.cue(b, "Images also expose disguises."))
        D = namecard("DoRA", sans("weight-decomposed low-rank adaptation", 18, INK3), width=5.4, tsize=46)
        D.move_to(y * UP)
        self.play(FadeIn(D, shift=0.2 * UP), run_time=0.8)

        self.at(self.cue(b, "which splits each weight"))
        M = namecard("magnitude", math(r"\operatorname{diag}(\mu)", 30, INK2), width=3.7).move_to([-2.45, y, 0])
        Dr = namecard("direction", math(r"N(W_0+BA)", 30, INK2), width=3.7).move_to([2.45, y, 0])
        dot = math(r"\times", 34, INK2).move_to([0, y, 0])
        note = sans("N scales each row to unit length · one magnitude per output neuron", 16, INK3)
        note.move_to([0, y - 1.05, 0])
        self.play(ReplacementTransform(D[0].copy(), M[0]), ReplacementTransform(D[0], Dr[0]), FadeOut(D[1]),
                  run_time=1.1)
        self.play(FadeIn(M[1]), FadeIn(Dr[1]), FadeIn(dot), FadeIn(note), run_time=0.7)

        # reassemble as LoRA followed by one gain per output neuron
        self.at(self.cue(b, "reaches exactly the same weights", offset=0.5))
        left_t = VGroup(serif("LoRA", 36, TIDE, SEMIBOLD), math(r"W_0+BA", 30, INK2)).arrange(DOWN, buff=0.14)
        right_t = VGroup(serif("one gain per output neuron", 30, INK, SEMIBOLD),
                         math(r"\operatorname{diag}(\ell)", 30, INK2)).arrange(DOWN, buff=0.14)
        arr = Arrow(LEFT * 0.45, RIGHT * 0.45, buff=0, color=INK2, stroke_width=3, max_tip_length_to_length_ratio=0.3)
        row = VGroup(left_t, arr, right_t).arrange(RIGHT, buff=0.45)
        fbox = RoundedRectangle(corner_radius=0.12, width=row.width + 1.0, height=1.45, fill_color=BG2,
                                fill_opacity=1, stroke_color=RULE, stroke_width=1.4)
        F = VGroup(fbox, row)
        row.move_to(fbox)
        F.move_to(y * UP)
        fbox2 = fbox.copy()
        self.play(ReplacementTransform(M[0], fbox2), ReplacementTransform(Dr[0], fbox),
                  FadeTransform(Dr[1], left_t, path_arc=-0.6 * PI), FadeTransform(M[1], right_t, path_arc=-0.6 * PI),
                  FadeOut(dot), FadeOut(note), run_time=1.6)
        self.remove(fbox2)
        self.play(GrowArrow(arr), run_time=0.5)
        comp = math(r"\operatorname{diag}(\ell)\,(W_0+BA)", 32, INK).next_to(F, DOWN, buff=0.32)
        self.play(FadeIn(comp, shift=0.1 * UP), run_time=0.6)

        self.at(self.cue(b, "one gain per output neuron", offset=-0.4))
        pc = card("II.8", width_cm=12.0, body_size=26).move_to([0, -1.88, 0])
        self.play(FadeIn(pc, shift=0.2 * UP), run_time=0.8)
        self.at(self.cue(b, "Beyond those gains"))
        cap = serif("beyond the gains: training, not reach", 26, INK2, italic=True).next_to(comp, DOWN, buff=0.22)
        self.play(FadeIn(cap), Indicate(right_t[0], color=INK, scale_factor=1.05), run_time=1.0)

        self.at(self.END(b))
        self.clear_stage(run_time=0.5, keep=(self.lab, self.leg))
        self.cam3d()

    # ============================================================== II.3 tips and smooth points
    def ii3(self):
        b = "II.3"
        self.section(b)
        scr = self.scr
        self.at(self.cue(b, "Where you start on an image matters."))
        rulings, rims, skin = cone(PB, H, R)
        self.play(Create(rulings, lag_ratio=0.03), run_time=1.3)
        th0 = self.theta_mark(PB)
        t_xy = scr(PB)
        th_lab = self.hud(math(r"\theta_0", 40).move_to(t_xy + np.array([-0.55, -0.25, 0])))
        self.play(Create(rims), FadeIn(skin), FadeIn(th0), FadeIn(th_lab), run_time=0.8)
        self.bring_to_front(th0)
        self.cone_parts = VGroup(rulings, rims, skin)
        self.th0, self.th_lab = th0, th_lab

        # a smooth point and its tangent plane
        S = cone_pt(PB, H_S, A_S)
        u = ruling(A_S)
        v = np.array([-np.sin(A_S), np.cos(A_S), 0.0])
        (u0, u1), (v0, v1) = PLANE_U, PLANE_V
        corners = [S + u0 * u + v0 * v, S + u1 * u + v0 * v, S + u1 * u + v1 * v, S + u0 * u + v1 * v]
        plane = Polygon(*corners, fill_color=LEAF, fill_opacity=0.18, stroke_color=LEAF, stroke_width=1.6)
        sdot = self.pin(Dot(S, radius=0.065, color=INK))
        sdot.set_opacity(0)
        s_xy = scr(S)
        x_lab = max(scr(c)[0] for c in corners) + 0.18
        s_name = sans("a smooth point", 20, INK)
        s_dir = VGroup(sans("first-order directions:", 18, LEAF), sans("a flat plane", 18, LEAF))
        s_dir.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        s_lab = VGroup(s_name, s_dir).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        s_lab.move_to([x_lab, s_xy[1] + 0.35, 0], aligned_edge=LEFT)
        self.at(self.cue(b, "At a smooth point"))
        self.play(sdot.animate.set_opacity(1), run_time=0.5)
        self.show(s_name, run_time=0.5)
        self.at(self.cue(b, "form a flat plane"))
        self.play(FadeIn(plane), run_time=1.2)
        self.bring_to_front(sdot, th0)
        self.show(s_dir, run_time=0.5)

        # the tip: a smaller set of first-order directions (one ruling in this picture)
        w = ruling(B_TIP)
        s1 = Line(PB - 1.35 * w, PB + 1.35 * w, color=LEAF, stroke_width=4)
        self.at(self.cue(b, "At the tip of a cone"))
        self.play(plane.animate.set_fill(opacity=0.06).set_stroke(opacity=0.3), s_dir.animate.set_opacity(0.4),
                  Create(s1), run_time=1.0)
        self.bring_to_front(sdot, th0)
        tip_lab = VGroup(sans("zero init:", 20, INK), math(r"B_0=0", 30, INK)).arrange(RIGHT, buff=0.14)
        tip_lab.move_to([t_xy[0] - 1.45, t_xy[1] - 0.62, 0], aligned_edge=RIGHT)
        lead = Line([t_xy[0] - 1.38, t_xy[1] - 0.6, 0], t_xy + np.array([-0.13, -0.1, 0]), color=RULE,
                    stroke_width=1.4)
        self.at(self.cue(b, "Zero-initialised LoRA"))
        self.show(tip_lab, lead, run_time=0.6)
        self.at(self.cue(b, "starts exactly at the tip"))
        self.pulse(PB)

        # B = 0: nudging A changes nothing, so only B moves the weights at first
        eq1 = math(r"\delta(BA)=\delta B\,A+B\,\delta A", 36)
        eq1.to_corner(UL, buff=0.75).shift(0.8 * DOWN)
        term = eq1[0][10:13]                                # "B \delta A"
        strike = Line(term.get_corner(DL) + 0.04 * DOWN, term.get_corner(UR) + 0.04 * UP, color=CORAL,
                      stroke_width=3)
        why = sans("zero when B = 0", 18, CORAL).next_to(term, DOWN, buff=0.16)
        self.at(self.cue(b, "With B at zero"))
        self.show(eq1, run_time=0.7)
        self.hud(strike, why)
        self.play(Create(strike), FadeIn(why), run_time=0.6)
        self.at(self.cue(b, "so at first only B can move the weights"))
        mover = self.pin(Dot(PB, radius=0.06, color=LEAF))
        self.play(eq1[0][6:9].animate.set_color(LEAF), run_time=0.5)
        self.play(MoveAlongPath(mover, Line(PB, PB + 1.1 * w)), run_time=1.0, rate_func=smooth)
        self.play(MoveAlongPath(mover, Line(PB + 1.1 * w, PB - 0.9 * w)), run_time=1.0, rate_func=smooth)
        self.play(FadeOut(mover), run_time=0.3)

        # counts at the tip, and the computed values (video/sim/first_order.py)
        d1 = FO["dimS1"]
        tip_cnt = VGroup(VGroup(sans("first-order directions:", 20, LEAF), math(r"mr", 30, LEAF)),
                         VGroup(sans("dimension:", 20, INK2), math(r"r(m+n-r)", 30, INK2)))
        for g in tip_cnt:
            g.arrange(RIGHT, buff=0.12)
        tip_cnt.arrange(DOWN, aligned_edge=RIGHT, buff=0.14)
        tip_cnt.next_to(tip_lab, DOWN, buff=0.18, aligned_edge=RIGHT)
        tab_head = VGroup(sans("first-order directions, computed", 18, INK2),
                          sans("Jacobian ranks at (m, n, r) = (%d, %d, %d)" % (FO["m"], FO["n"], FO["r"]), 16, INK3))
        tab_head.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rows = [("zero-B init", str(d1["zero_B"]), "= mr", LEAF),
                ("zero-A init", str(d1["zero_A"]), "= nr", LEAF),
                ("split init", str(d1["split"]), "= r(m + n − r)", LEAF),
                ("image dimension", str(FO["image_dim_generic_jacobian_rank"]), "= r(m + n − r)", INK2)]
        tab = VGroup()
        for i, (name, val, form, col) in enumerate(rows):
            yy = -0.44 * i
            tab.add(VGroup(sans(name, 20, INK2).move_to([0, yy, 0], aligned_edge=LEFT),
                           mono(val, 24, col).move_to([2.65, yy, 0], aligned_edge=RIGHT),
                           sans(form, 18, INK3).move_to([2.85, yy, 0], aligned_edge=LEFT)))
        table = VGroup(tab_head, tab).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        table.to_corner(DL, buff=0.75).shift(0.1 * UP)
        self.at(self.cue(b, "and fewer directions are open"))
        self.show(tip_cnt, run_time=0.7)
        self.show(tab_head, tab[0], tab[1], tab[3], run_time=0.7, lag=0.15)

        # the split initialisation: a point on the cone's side
        self.at(self.cue(b, "Split initialisations"))
        s_name2 = VGroup(sans("split init", 20, INK, MEDIUM), sans("a smooth point", 18, INK2))
        s_name2.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(s_name, aligned_edge=UL)
        s_dir2 = VGroup(sans("first-order directions:", 18, LEAF), math(r"r(m+n-r)", 28, LEAF))
        s_dir2.arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(s_name2, DOWN, buff=0.14, aligned_edge=LEFT)
        self.hud(s_name2)
        self.play(FadeOut(s_name), FadeIn(s_name2), FadeOut(s_dir),
                  plane.animate.set_fill(opacity=0.18).set_stroke(opacity=1.0), run_time=0.8)
        self.bring_to_front(sdot, th0)
        eq2 = math(r"W_0=(W_0-B_0A_0)+B_0A_0", 34)
        eq2.next_to(eq1, DOWN, buff=0.8, aligned_edge=LEFT)
        braces = VGroup(sans("frozen", 16, INK3).next_to(eq2[0][3:12], DOWN, buff=0.12),
                        sans("adapter at start", 16, INK3).next_to(eq2[0][13:], DOWN, buff=0.12))
        self.at(self.cue(b, "which carve a low-rank piece"))
        self.show(eq2, run_time=0.7)
        self.show(braces, run_time=0.4)
        self.at(self.cue(b, "start at a smooth point instead"))
        self.show(s_dir2, run_time=0.6)
        self.show(tab[2], run_time=0.5)

        # the paper cards: Theorem II.2, and a mention of Theorem II.7 (HRA)
        self.at(self.cue(b, "start at a smooth point instead", end=True, offset=0.3))
        left = VGroup(eq1, strike, why, eq2, braces, table, tip_lab, lead, tip_cnt)
        self.play(*[FadeOut(m) for m in left], run_time=0.6)
        c2 = card("II.2", width_cm=7.6, body_size=24).to_corner(UL, buff=0.5).shift(0.42 * DOWN)
        self.show(c2, shift=0.15 * UP, run_time=0.8)
        self.at(self.cue(b, "HRA, a method built from reflections"))
        c7 = card("II.7", width_cm=7.6, body_size=24).next_to(c2, DOWN, buff=0.3, aligned_edge=LEFT)
        self.show(c7, shift=0.15 * UP, run_time=0.8)
        self.ii3_keep = dict(S=S, sdot=sdot, plane=plane, s1=s1, s_lab=VGroup(s_name2, s_dir2), cards=VGroup(c2, c7))

        self.at(self.END(b))
        k = self.ii3_keep
        self.play(FadeOut(k["cards"]), FadeOut(k["s_lab"]), FadeOut(k["plane"]), FadeOut(k["s1"]), run_time=0.5)

    # ============================================================== II.4 nut 1 dissolves
    def ii4(self):
        b = "II.4"
        self.section(b)
        scr = self.scr
        k = self.ii3_keep
        S, sdot = k["S"], k["sdot"]
        rulings, rims, skin = self.cone_parts

        self.at(self.cue(b, "Now the first nut."))
        badge = VGroup(Circle(0.16, stroke_color=TIDE, stroke_width=1.6), sans("1", 16, TIDE, MEDIUM))
        head = VGroup(badge, kicker("nut", INK3, 15)).arrange(RIGHT, buff=0.12)
        claim = paragraph("No exact initialisation lets LoRA reach an update of rank above 2r.", 36, 24, INK)
        nut = VGroup(head, claim).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_corner(UL, buff=0.75)
        nut.shift(0.28 * DOWN)
        self.show(nut, run_time=0.8)

        # the adapter's starting value B0 A0: the arrow from the tip to S
        self.at(self.cue(b, "Split initialisations start with a nonzero adapter"))
        def b0a0_arrow(tail, head):
            return Arrow(scr(tail), scr(head), buff=0.06, color=INK, stroke_width=3, tip_length=0.16,
                         max_tip_length_to_length_ratio=0.2)
        arrow = b0a0_arrow(PB, S)
        mid = scr(0.5 * (PB + S))
        a_lab = math(r"B_0A_0", 32, INK).move_to(mid + np.array([0.45, -0.22, 0]))
        self.hud(arrow)
        self.play(GrowArrow(arrow), run_time=0.8)
        self.show(a_lab, run_time=0.5)
        eqa = VGroup(sans("adapter at start", 18, INK3), math(r"B_0A_0\neq0", 32)).arrange(RIGHT, buff=0.2)
        eqb = VGroup(sans("frozen weights", 18, INK3), math(r"W_0-B_0A_0", 32)).arrange(RIGHT, buff=0.2)
        eqc = VGroup(sans("model at start", 18, INK3), math(r"(W_0-B_0A_0)+B_0A_0=\theta_0", 32))
        eqc.arrange(RIGHT, buff=0.2)
        eqs = VGroup(eqa, eqb, eqc).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        eqs.next_to(nut, DOWN, buff=0.6, aligned_edge=LEFT)
        self.show(eqa, run_time=0.6)

        # subtract it from the frozen weights: the cone slides until it passes through theta_0 again
        self.at(self.cue(b, "and subtract that piece from the frozen weights"))
        d = PB - S
        dscr = scr(PB + d) - scr(PB)
        self.show(eqb, run_time=0.6)
        self.play(rulings.animate.shift(d), rims.animate.shift(d), skin.animate.shift(d), sdot.animate.move_to(PB),
                  Transform(arrow, b0a0_arrow(PB + d, PB)), a_lab.animate.shift(dscr),
                  self.th_lab.animate.move_to(scr(PB) + np.array([0.55, 0.22, 0])), run_time=2.4, rate_func=smooth)
        self.bring_to_front(self.th0)
        self.at(self.cue(b, "so the model still begins at theta-zero"))
        self.show(eqc, run_time=0.7)
        self.pulse(PB, color=TIDE)

        # the new vertex theta_0 - B0 A0
        V = PB + d
        self.at(self.cue(b, "that's sliding the whole cone"))
        vx = self.pin(Square(0.13, color=INK2, stroke_width=2).rotate(PI / 4).move_to(V))
        v_lab = VGroup(sans("vertex", 18, INK2), math(r"\theta_0-B_0A_0", 28, INK2)).arrange(RIGHT, buff=0.12)
        v_lab.move_to(scr(V) + np.array([0.5, -0.36, 0]), aligned_edge=LEFT)
        self.play(FadeIn(vx), run_time=0.5)
        self.show(v_lab, run_time=0.5)

        # every way: translated copies through theta_0, whose union is theta_0 + (C - C)
        self.at(self.cue(b, "Slide it every possible way", offset=-0.6))
        self.play(FadeOut(arrow), FadeOut(a_lab), FadeOut(vx), FadeOut(v_lab), run_time=0.6)
        ps = [cone_pt(ORIGIN, s * 0.4 * H, a) for s in (1, -1)
              for a in np.linspace(0, TAU, 6, endpoint=False) + (0.3 if s > 0 else 0.8)]
        copies = VGroup(*[ghost_cone(PB - p, HH=0.8 * H, RR_=0.8 * R) for p in ps])
        anims = []
        for p, cp in zip(ps, copies):
            anims.append(FadeIn(cp, shift=(PB - p) - V))
        self.play(LaggedStart(*anims, lag_ratio=0.22), FadeOut(VGroup(rulings, rims, skin)), run_time=4.6)
        self.bring_to_front(self.th0)
        ext_h = 1.2 * H * np.sin(PHI) + 0.2
        ext_w = 1.2 * R + 0.22
        region = Ellipse(width=2 * ext_w, height=2 * ext_h, fill_color=TIDE, fill_opacity=0.06, stroke_width=0)
        outline = DashedVMobject(Ellipse(width=2 * ext_w, height=2 * ext_h), num_dashes=64, dashed_ratio=0.5)
        outline.set_stroke(TIDE, 1.6, opacity=0.8)
        reg = VGroup(region, outline).move_to(scr(PB))
        r_lab = VGroup(math(r"\theta_0+(C-C)", 34, TIDE), sans("all differences of", 16, INK2),
                       sans("two points of the cone", 16, INK2))
        r_lab.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        r_lab.next_to(reg, RIGHT, buff=0.2).shift(0.9 * UP)
        self.at(self.cue(b, "and together the copies cover", offset=2.6))
        self.hud(reg)
        self.play(FadeIn(reg), run_time=1.0)
        self.show(r_lab, run_time=0.6)
        self.play(*[cp.animate.set_stroke(opacity=0.12) for cp in copies], run_time=0.8)

        self.at(self.cue(b, "The difference of two matrices"))
        eqd = math(r"\operatorname{rank}(X-Y)\le\operatorname{rank}X+\operatorname{rank}Y\le 2r", 32)
        eqd.next_to(eqs, DOWN, buff=0.55, aligned_edge=LEFT)
        self.show(eqd, run_time=0.8)
        self.play(Indicate(eqd, color=TIDE, scale_factor=1.03), run_time=1.0)

        # the histogram, from video/sim/repoint_rank.py
        self.at(self.cue(b, "So no exact initialisation"))
        gone = [m for m in self.mobjects if m not in (self.lab, self.leg, nut)]
        self.play(*[FadeOut(m) for m in gone], run_time=0.6)
        self.cam2d()
        counts = RR["counts"]
        two_r = 2 * RR["r"]
        nb = len(counts)
        X0, Y0, W_ax, top = -6.36, -2.3, 5.4, 2.0
        unit = W_ax / nb
        ax = plain_axes([0, nb, 1], [0, 1, 1], width=W_ax, height=top + 0.4)
        ax.shift(np.array([X0, Y0, 0]) - ax.c2p(0, 0))
        vmax = max(counts)
        bars_ = VGroup()
        for q, c in enumerate(counts):
            h = top * c / vmax
            bars_.add(Rectangle(width=unit * 0.7, height=max(h, 1e-3), fill_color=TIDE, fill_opacity=0.9,
                                stroke_width=0).move_to([X0 + (q + 0.5) * unit, Y0 + h / 2, 0]))
        ticks = VGroup(*[mono(str(q), 20, INK2).move_to([X0 + (q + 0.5) * unit, Y0 - 0.27, 0]) for q in range(nb)])
        xl = sans("rank of the update θ − θ₀", 18, INK2).next_to(ticks, DOWN, buff=0.16)
        vals = VGroup(*[mono(str(c), 16, INK).next_to(bars_[q], UP, buff=0.08) for q, c in enumerate(counts) if c])
        n_samp = f'{RR["samples"]:,}'.replace(",", " ")
        title = VGroup(sans("Rank of θ − θ₀ over %s reachable points" % n_samp, 20, INK),
                       sans("m = n = %d, r = %d · %s random split initialisations"
                            % (RR["m"], RR["r"], f'{RR["inits"]:,}'.replace(",", " ")), 16, INK3),
                       sans("%d random reachable points each (rank BA = 0, 1 or 2)" % RR["per_init"], 16, INK3))
        title.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([X0, 0.98, 0], aligned_edge=LEFT)
        xc = X0 + (two_r + 1) * unit
        ceil_ = DashedLine([xc, Y0, 0], [xc, Y0 + top + 0.4, 0], color=CORAL, stroke_width=2, dash_length=0.08)
        ceil_lab = math(r"2r=%d" % two_r, 28, CORAL).next_to(ceil_.get_end(), RIGHT, buff=0.1)
        none = VGroup(sans("above 2r:", 18, INK2), mono("%d of %s" % (RR["above_2r"], n_samp), 20, INK))
        none.arrange(RIGHT, buff=0.12).move_to([xc + 0.15, Y0 + 1.05, 0], aligned_edge=LEFT)
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), FadeIn(title), run_time=0.8)
        self.at(self.cue(b, "lets LoRA reach an update of rank above two r", offset=-1.4))
        self.play(LaggedStart(*[GrowFromEdge(br, DOWN) for br in bars_], lag_ratio=0.08), run_time=1.4)
        self.play(FadeIn(vals), run_time=0.5)
        self.at(self.cue(b, "rank above two r"))
        self.play(Create(ceil_), FadeIn(ceil_lab), run_time=0.7)
        self.play(FadeIn(none), run_time=0.5)

        self.at(self.cue(b, "Practitioners already knew this ceiling"))
        c1 = card("II.1", width_cm=7.0, body_size=24).move_to([3.45, -0.25, 0])
        self.play(FadeIn(c1, shift=0.2 * UP), run_time=0.8)
        self.at(self.cue(b, "The picture adds that two r is really needed"))
        top = bars_[two_r]
        mark = SurroundingRectangle(VGroup(top, vals[-1]), color=TIDE, buff=0.08, stroke_width=2)
        self.play(Create(mark), run_time=0.7)

        # the shore: nut 1 dissolves
        self.at(self.T(b) + self.plan[b]["audio"])
        self.clear_stage(run_time=0.6)
        s = self.shore(LEVELS[1])
        self.wait(0.3)
        self.dissolve_nut(s, 1, LEVELS[2])
        self.wait(0.2)
        self.clear_stage(run_time=0.6)

    # ============================================================== II.5 rank is a cut, dimension is a volume
    def ii5(self):
        b = "II.5"
        self.section(b)
        self.at(self.cue(b, "Rank has a picture of its own."))
        self.lab = self.chapter_label("II", "the image")
        self.leg = self.legend("image")
        yA, yH = 1.95, -0.15
        xs = dict(i=-4.45, A=-1.55, B=1.55, o=4.6)
        xcut = 0.5                                          # where the cuts stop: on the r-wires, clear of the labels
        lora_name = VGroup(sans("LoRA", 24, TIDE, MEDIUM), math(r"BA", 30, INK2)).arrange(DOWN, buff=0.1)
        lora_name.move_to([-5.75, yA, 0])
        self.play(FadeIn(self.lab), FadeIn(self.leg), FadeIn(lora_name), run_time=0.8)

        # LoRA as a tensor diagram
        wn = wire([xs["i"], yA, 0], [xs["A"] - 0.48, yA, 0], 5)
        bA = box("A", [xs["A"], yA, 0])
        wr = wire([xs["A"] + 0.48, yA, 0], [xs["B"] - 0.48, yA, 0], 2)
        bB = box("B", [xs["B"], yA, 0])
        wm = wire([xs["B"] + 0.48, yA, 0], [xs["o"], yA, 0], 5)
        lab_n = math("n", 30, INK2).next_to(wn, UP, buff=0.12)
        lab_r = math("r", 30, INK2).next_to(wr, UP, buff=0.12)
        lab_m = math("m", 30, INK2).next_to(wm, UP, buff=0.12)
        io = VGroup(sans("input", 16, INK3).next_to(wn.get_start(), DOWN, buff=0.14),
                    sans("output", 16, INK3).next_to(wm.get_end(), DOWN, buff=0.14))
        self.at(self.cue(b, "Draw an adapter as boxes joined by wires."))
        self.play(LaggedStart(Create(wn), FadeIn(bA), Create(wr), FadeIn(bB), Create(wm), lag_ratio=0.35),
                  run_time=2.2)
        self.play(FadeIn(lab_n), FadeIn(lab_r), FadeIn(lab_m), FadeIn(io), run_time=0.6)

        # a cut sweeps across and stops on the r-wire
        cut_x = ValueTracker(xs["i"] + 0.9)
        cut = DashedLine([0, yA - 0.62, 0], [0, yA + 0.62, 0], color=CORAL, stroke_width=2.6, dash_length=0.09)
        cut.move_to([cut_x.get_value(), yA, 0])
        cut.add_updater(lambda m: m.move_to([cut_x.get_value(), yA, 0]))
        reads = {"n": math(r"\operatorname{rank}\le n", 30, CORAL), "r": math(r"\operatorname{rank}\le r", 30, CORAL),
                 "m": math(r"\operatorname{rank}\le m", 30, CORAL)}
        spans = {"n": (xs["i"], xs["A"] - 0.48), "r": (xs["A"] + 0.48, xs["B"] - 0.48), "m": (xs["B"] + 0.48, xs["o"])}

        def reader(key):
            def upd(m):
                x = cut_x.get_value()
                lo, hi = spans[key]
                inside = lo + 0.05 < x < hi - 0.05
                m.move_to([x, yA + 0.88, 0]).set_opacity(1.0 if inside else 0.0)
            return upd
        for key, m in reads.items():
            m.add_updater(reader(key))
        self.at(self.cue(b, "Cut the diagram anywhere"))
        self.add(*reads.values())
        self.play(Create(cut), run_time=0.6)
        self.at(self.cue(b, "and the rank can't exceed"))
        self.play(cut_x.animate.set_value(xs["o"] - 0.9), run_time=4.0, rate_func=linear)
        c10 = card("II.10", width_cm=15.0, body_size=24).move_to([0, -2.47, 0])
        self.play(FadeIn(c10, shift=0.2 * UP), run_time=0.8)
        self.at(self.cue(b, "For LoRA, the narrowest cut"))
        self.play(cut_x.animate.set_value(xcut), run_time=1.6, rate_func=smooth)
        self.play(wr.animate.set_color(CORAL), Indicate(reads["r"], color=CORAL, scale_factor=1.15), run_time=0.9)
        for m in (cut, *reads.values()):
            m.clear_updaters()

        # LoHa: two chains joined by an entrywise product
        yu, yd = yH + 0.62, yH - 0.62
        loha_name = VGroup(sans("LoHa", 24, VIOLET, MEDIUM), math(r"B_1A_1\odot B_2A_2", 26, INK2))
        loha_name.arrange(DOWN, buff=0.1).move_to([-5.75, yH, 0])
        xin, xcp, xpr = xs["i"], -3.2, 3.2
        w_in = wire([xin, yH, 0], [xcp, yH, 0], 5)
        cp = Dot([xcp, yH, 0], radius=0.09, color=INK)
        legs_in = VGroup(*[wire([xcp, yH, 0], [xs["A"] - 0.48, y, 0], 5) for y in (yu, yd)])
        bA1, bA2 = box("A_1", [xs["A"], yu, 0], 1.0, 0.68), box("A_2", [xs["A"], yd, 0], 1.0, 0.68)
        wr1 = wire([xs["A"] + 0.5, yu, 0], [xs["B"] - 0.5, yu, 0], 2)
        wr2 = wire([xs["A"] + 0.5, yd, 0], [xs["B"] - 0.5, yd, 0], 2)
        bB1, bB2 = box("B_1", [xs["B"], yu, 0], 1.0, 0.68), box("B_2", [xs["B"], yd, 0], 1.0, 0.68)
        legs_out = VGroup(*[wire([xs["B"] + 0.5, y, 0], [xpr - 0.2, yH, 0], 5) for y in (yu, yd)])
        prod = VGroup(Circle(0.2, color=INK, stroke_width=2, fill_color=BG2, fill_opacity=1),
                      math(r"\odot", 30, INK)).move_to([xpr, yH, 0])
        w_out = wire([xpr + 0.2, yH, 0], [xs["o"], yH, 0], 5)
        h_lab = VGroup(math("n", 30, INK2).next_to(w_in, UP, buff=0.12),
                       math("r", 30, INK2).next_to(wr1, UP, buff=0.1), math("r", 30, INK2).next_to(wr2, DOWN, buff=0.1),
                       math("m", 30, INK2).next_to(w_out, UP, buff=0.12))
        self.at(self.cue(b, "LoHa multiplies two low-rank factors"))
        self.play(FadeIn(loha_name), run_time=0.5)
        self.play(LaggedStart(Create(w_in), FadeIn(cp), Create(legs_in), FadeIn(bA1), FadeIn(bA2), Create(wr1),
                              Create(wr2), FadeIn(bB1), FadeIn(bB2), Create(legs_out), FadeIn(prod), Create(w_out),
                              lag_ratio=0.18), run_time=2.6)
        self.play(FadeIn(h_lab), run_time=0.5)
        self.at(self.cue(b, "and its narrowest cut is r times r"))
        cut2 = DashedLine([xcut, yd - 0.4, 0], [xcut, yu + 0.4, 0], color=CORAL, stroke_width=2.6, dash_length=0.09)
        read2 = math(r"\operatorname{rank}\le r\cdot r", 30, CORAL).next_to(cut2, UP, buff=0.08)
        self.play(Create(cut2), run_time=0.7)
        self.play(wr1.animate.set_color(CORAL), wr2.animate.set_color(CORAL), FadeIn(read2), run_time=0.7)


        # equal budget: rank and dimension, computed by video/sim/cut_volume.py
        self.at(self.cue(b, "is a volume"))
        gone = [m for m in self.mobjects if m not in (self.lab, self.leg)]
        self.play(*[FadeOut(m) for m in gone], run_time=0.6)
        case = CV["cases"]["3"]
        r_ = case["r"]
        pair_w, bw = 2.3, 0.62
        base_y, top_h = -1.9, 2.8

        def pair(xc_, vals_, vmax_, head, sub, budget=None, names=("LoHa", "LoRA")):
            g = VGroup()
            for i, (v, col) in enumerate(zip(vals_, (VIOLET, TIDE))):
                h = top_h * v / vmax_
                g.add(Rectangle(width=bw, height=h, fill_color=col, fill_opacity=0.9, stroke_width=0)
                      .move_to([xc_ + (i - 0.5) * 0.95, base_y + h / 2, 0]))
            base = Line([xc_ - pair_w / 2, base_y, 0], [xc_ + pair_w / 2, base_y, 0], color=RULE, stroke_width=1.5)
            nums = VGroup(*[mono(str(v), 20, BG).move_to(g[i].get_top() + 0.2 * DOWN) for i, v in enumerate(vals_)])
            hd = VGroup(sans(head, 22, INK, MEDIUM), sans(sub, 18, INK3)).arrange(DOWN, buff=0.08)
            hd.move_to([xc_, base_y + top_h + 0.95, 0])
            parts = dict(bars=g, base=base, nums=nums, head=hd)
            if budget is not None:
                yb = base_y + top_h * budget / vmax_
                parts["budget"] = VGroup(DashedLine([xc_ - pair_w / 2, yb, 0], [xc_ + pair_w / 2, yb, 0],
                                                    color=INK3, stroke_width=1.5, dash_length=0.07),
                                         mono("budget %d" % budget, 16, INK3)
                                         .move_to([xc_ - pair_w / 2, yb + 0.18, 0], aligned_edge=LEFT))
            return parts

        def names_row(xc_, a, b_):
            return VGroup(math(a, 26, VIOLET).move_to([xc_ - 0.475, base_y - 0.32, 0]),
                          math(b_, 26, TIDE).move_to([xc_ + 0.475, base_y - 0.32, 0]))
        xr, xd = -5.0, -2.05
        ranks = pair(xr, [case["loha"]["rank"], case["lora"]["rank"]], 10.0, "rank", "a cut")
        dims = pair(xd, [case["loha"]["dim"], case["lora"]["dim"]], 800.0, "dimension", "a volume",
                    budget=case["budget_lora"])
        n1 = names_row(xr, r"\mathrm{LoHa}_{%d,%d}" % (r_, r_), r"\mathrm{LoRA}_{%d}" % (2 * r_))
        n2 = names_row(xd, r"\mathrm{LoHa}_{%d,%d}" % (r_, r_), r"\mathrm{LoRA}_{%d}" % (2 * r_))
        case_lab = VGroup(sans("m = n = %d, r = %d" % (CV["m"], r_), 20, INK),
                          sans("equal budget 2r(m + n) = %d numbers each" % case["budget_lora"], 16, INK3))
        case_lab.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([xr - pair_w / 2, 2.85, 0], aligned_edge=LEFT)
        self.play(FadeIn(ranks["head"]), FadeIn(dims["head"]), Create(ranks["base"]), Create(dims["base"]),
                  run_time=0.6)
        self.at(self.cue(b, "At the same budget as LoRA of rank two r"))
        self.play(FadeIn(case_lab), FadeIn(n1), FadeIn(n2), FadeIn(dims["budget"]), run_time=0.7)
        c12 = card("II.12", width_cm=7.3, body_size=24).move_to([3.28, 0.0, 0])
        self.play(FadeIn(c12, shift=0.2 * UP), run_time=0.8)
        self.at(self.cue(b, "LoHa reaches rank up to r squared"))
        self.play(LaggedStart(*[GrowFromEdge(x, DOWN) for x in ranks["bars"]], lag_ratio=0.3), run_time=1.0)
        self.play(FadeIn(ranks["nums"]), run_time=0.4)
        self.at(self.cue(b, "but on large layers"))
        self.play(LaggedStart(*[GrowFromEdge(x, DOWN) for x in dims["bars"]], lag_ratio=0.3), run_time=1.0)
        self.play(FadeIn(dims["nums"]), run_time=0.4)
        cap_y = base_y - 0.95
        cap1 = VGroup(sans("r ≥ 3:", 20, INK, MEDIUM), sans("LoHa trades dimension for rank", 20, INK2))
        cap1.arrange(RIGHT, buff=0.14).move_to([xr - pair_w / 2, cap_y, 0], aligned_edge=LEFT)
        self.at(self.cue(b, "a trade of dimension for rank"))
        self.play(FadeIn(cap1), run_time=0.6)

        # r = 2: the same comparison, where LoHa gains no rank
        self.at(self.cue(b, "and a pure loss", offset=-0.3))
        c2_ = CV["cases"]["2"]
        r2 = c2_["r"]
        ranks2 = pair(xr, [c2_["loha"]["rank"], c2_["lora"]["rank"]], 10.0, "rank", "a cut")
        dims2 = pair(xd, [c2_["loha"]["dim"], c2_["lora"]["dim"]], 800.0, "dimension", "a volume",
                     budget=c2_["budget_lora"])
        n1b = names_row(xr, r"\mathrm{LoHa}_{%d,%d}" % (r2, r2), r"\mathrm{LoRA}_{%d}" % (2 * r2))
        n2b = names_row(xd, r"\mathrm{LoHa}_{%d,%d}" % (r2, r2), r"\mathrm{LoRA}_{%d}" % (2 * r2))
        case2 = VGroup(sans("m = n = %d, r = %d" % (CV["m"], r2), 20, INK),
                       sans("equal budget 2r(m + n) = %d numbers each" % c2_["budget_lora"], 16, INK3))
        case2.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(case_lab, aligned_edge=LEFT)
        cap2 = VGroup(sans("r ≤ 2:", 20, INK, MEDIUM), sans("no rank gained, a pure loss", 20, INK2))
        cap2.arrange(RIGHT, buff=0.14).move_to([xr - pair_w / 2, cap_y - 0.45, 0], aligned_edge=LEFT)
        self.play(*[Transform(x, y) for x, y in zip(ranks["bars"], ranks2["bars"])],
                  *[Transform(x, y) for x, y in zip(dims["bars"], dims2["bars"])],
                  Transform(ranks["nums"], ranks2["nums"]), Transform(dims["nums"], dims2["nums"]),
                  Transform(dims["budget"], dims2["budget"]), Transform(n1, n1b), Transform(n2, n2b),
                  Transform(case_lab, case2), cap1.animate.set_opacity(0.45), run_time=1.2)
        self.play(FadeIn(cap2), run_time=0.6)
