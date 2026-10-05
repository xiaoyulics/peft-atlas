import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *
# Cold open and prologue: beats 0.1, 0.2, 0.3, P.1 and P.2 of video/script/SCRIPT.md.
#
# Numbers on screen come from
#   video/sim/lora_plus.py      0.1: both trajectories, their losses and max |W_LoRA+ - W_LoRA(alpha*16)| (exactly 0.0)
#   video/sim/lora_plus_eps.py  0.1 footnote: stock Adam with eps on A and 16 eps on B is still bit-exact (0.0)
#   video/sim/atlas_names.py    0.2: the method names in site/data/atlas-data.js and their count (382)
#
#   PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -qh --fps 30 --disable_caching \
#       --media_dir video/build/Opening video/film/ch0_opening.py Opening
import json

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim" / "out"


# ---------------------------------------------------------------- small parts
class Ticker(VGroup):
    """A right-aligned monospace integer that can change every frame. Each digit slot holds the ten glyphs and shows
    one of them by opacity, so the mobject family never changes (Manim keeps drawing replaced submobjects that were
    in its moving-mobject list). `anchor` is the right edge of the last slot, at the digits' vertical centre."""

    def __init__(self, size, color, anchor, ndig=3):
        super().__init__()
        ref = Text("0123456789", font=MONO, font_size=size, color=color)
        self.adv = (ref[9].get_center()[0] - ref[0].get_center()[0]) / 9
        slot0 = ref[0].get_center()
        self.ndig, self.value = ndig, None
        anchor = np.array(anchor, dtype=float)
        for j in range(ndig):
            for d in range(10):
                g = ref[d].copy().shift(-(slot0 + d * self.adv * RIGHT))
                g.shift(anchor + (j - ndig + 0.5) * self.adv * RIGHT)
                self.add(g.set_fill(opacity=0))
        self.set_value(0)

    def set_value(self, v):
        v = int(v)
        if v != self.value:
            s = str(v).rjust(self.ndig)
            for j, ch in enumerate(s):
                for d in range(10):
                    self[10 * j + d].set_fill(opacity=1.0 if ch == str(d) else 0.0)
            self.value = v
        return self


def markup(parts, size, font=SANS):
    """One line of text in several colours on a common baseline: parts = [(text, colour), ...]."""
    body = "".join(f'<span foreground="{c}">{t.replace("&", "&amp;").replace("<", "&lt;")}</span>' for t, c in parts)
    return MarkupText(body, font=font, font_size=size)


# ---------------------------------------------------------------- the social card (0.3 here, C.2 in ch7_coda.py)
CARD_K = 14.22 / 1200          # site/assets/card.html is 1200 x 630 px: its width maps onto the frame's
NEAT = (6.70, 3.60)            # outer neatline (half-width, half-height): 0.4 margin inside the frame
NEAT_GAP = 0.07                # to the inner neatline
INNER = (-NEAT[0] + NEAT_GAP, NEAT[0] - NEAT_GAP, -NEAT[1] + NEAT_GAP, NEAT[1] - NEAT_GAP)   # x0, x1, y0, y1


def cpx(X, Y):
    """A point of card.html (px, y down) in frame units."""
    return np.array([(X - 600) * CARD_K, (315 - Y) * CARD_K, 0.0])


def _clip_segment(a, b, box):
    """Liang-Barsky: the part of segment a-b inside box = (x0, x1, y0, y1), or None."""
    x0, x1, y0, y1 = box
    t0, t1 = 0.0, 1.0
    d = b - a
    for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            t0 = max(t0, q / p)
        else:
            t1 = min(t1, q / p)
    return None if t0 > t1 else (a + t0 * d, a + t1 * d)


def clip_polyline(pts, box):
    """The runs of a polyline that lie inside box, each an array of points."""
    runs, cur = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = _clip_segment(a, b, box)
        if seg is None:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        p, q = seg
        if not cur or np.linalg.norm(cur[-1] - p) > 1e-9:
            if len(cur) > 1:
                runs.append(cur)
            cur = [p]
        cur.append(q)
        if np.linalg.norm(q - b) > 1e-9:          # left the box
            runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return [np.array(r) for r in runs]


def polyline(pts, color, width, opacity=1.0):
    return VMobject().set_points_as_corners(np.asarray(pts)).set_stroke(color, width, opacity)


def card_glow():
    """The faint radial glow of card.html, behind the motif."""
    c = cpx(912, 315)
    return VGroup(*[Ellipse(width=2 * 720 * CARD_K * f, height=2 * 378 * CARD_K * f, fill_color="#12233a",
                            fill_opacity=0.12, stroke_width=0).move_to(c) for f in np.linspace(1.0, 0.18, 7)])


def contours():
    """The isobaths of card.html (11 closed curves about a deep point), clipped to the neatline, fading to the left
    as under the card's mask. Allowed only on the title card (0.3) and the end card (C.2)."""
    deep = (960.0, 250.0)
    stops_x, stops_o = [-7.11, 0.0, 3.13], [0.18, 0.45, 1.0]
    g = VGroup()
    for k in range(11):
        r0 = 46 + 52 * k
        a = np.linspace(0, TAU, 321)
        r = r0 * (1 + 0.09 * np.sin(2 * a + 0.6 + 0.13 * k) + 0.05 * np.sin(3 * a + 1.7 - 0.21 * k)
                  + 0.025 * np.sin(5 * a + k))
        pts = np.array([cpx(deep[0] + 1.35 * rr * np.cos(aa), deep[1] + 0.82 * rr * np.sin(aa)) for rr, aa in zip(r, a)])
        color, width = ("#2a3d55", 1.35) if k % 5 == 4 else ("#1d2e44", 1.1)
        for run in clip_polyline(pts, INNER):
            for i in range(0, len(run) - 1, 8):
                chunk = run[i:i + 9]
                if len(chunk) > 1:
                    g.add(polyline(chunk, color, width, float(np.interp(chunk[:, 0].mean(), stops_x, stops_o))))
    return g


def neatline():
    """The graduated double rule around the card."""
    X, Y = NEAT
    g = VGroup(*[Rectangle(width=2 * (X - d), height=2 * (Y - d), stroke_color="#2a3d55", stroke_width=1.2,
                           fill_opacity=0) for d in (0, NEAT_GAP)])
    seg, band = 0.45, NEAT_GAP - 0.012
    xs = np.arange(-X + NEAT_GAP, X - NEAT_GAP, seg)
    ys = np.arange(-Y + NEAT_GAP, Y - NEAT_GAP, seg)
    for k, x in enumerate(xs):
        if k % 2 == 0:
            w = min(seg, X - NEAT_GAP - x)
            for yc in (Y - NEAT_GAP / 2, -Y + NEAT_GAP / 2):
                g.add(Rectangle(width=w, height=band, fill_color="#1f3149", fill_opacity=1, stroke_width=0)
                      .move_to([x + w / 2, yc, 0]))
    for k, y in enumerate(ys):
        if k % 2 == 0:
            h = min(seg, Y - NEAT_GAP - y)
            for xc in (X - NEAT_GAP / 2, -X + NEAT_GAP / 2):
                g.add(Rectangle(width=band, height=h, fill_color="#1f3149", fill_opacity=1, stroke_width=0)
                      .move_to([xc, y + h / 2, 0]))
    return g


def motif():
    """The card's geometry (card.html): LoRA's cone with apex theta_0, OFT's orbit of theta_0 (a circle about 0),
    and (IA)^3's line through 0 and theta_0. Returns dict(cone, orbit, line, zero, T) with T = theta_0."""
    D = DEGREES
    O, R, GAM = np.array([1074.0, 488.0]), 236.0, 222 * D
    T = O + R * np.array([np.cos(GAM), np.sin(GAM)])
    radial = np.array([np.cos(GAM), np.sin(GAM)])
    AX, h, a, b = -97 * D, 128.0, 84.0, 22.0
    u = np.array([np.cos(AX), np.sin(AX)])
    v = np.array([-u[1], u[0]])

    def rim(sg, phi):
        c = T + sg * h * u
        return cpx(*(c + a * np.cos(phi) * v + b * np.sin(phi) * u))

    phiT = np.arcsin(b / h)
    fills, rulings, lines = VGroup(), VGroup(), VGroup()
    Tp = cpx(*T)
    for sg in (1, -1):
        ring = [rim(sg, p) for p in np.linspace(0, TAU, 97)]
        tA, tB = rim(sg, -sg * phiT), rim(sg, PI + sg * phiT)
        fills.add(Polygon(*ring[:-1], stroke_width=0, fill_color=TIDE, fill_opacity=0.07))
        fills.add(Polygon(tA, Tp, tB, stroke_width=0, fill_color=TIDE, fill_opacity=0.06))
        for q in range(14):
            rulings.add(Line(Tp, rim(sg, q / 14 * TAU + 0.11), stroke_color=TIDE, stroke_width=0.9,
                             stroke_opacity=0.42))
        lines.add(polyline([tA, Tp, tB], TIDE, 1.8))
        if sg > 0:
            lines.add(polyline(ring, TIDE, 1.8))
        else:
            near = [rim(sg, phiT - n / 64 * (PI + 2 * phiT)) for n in range(65)]
            far = [rim(sg, phiT + n / 48 * (PI - 2 * phiT)) for n in range(49)]
            lines.add(polyline(near, TIDE, 1.8))
            lines.add(DashedVMobject(polyline(far, TIDE, 1.3, 0.6), num_dashes=16, dashed_ratio=0.45))
    cone = VGroup(fills, rulings, lines)

    circ = [cpx(*(O + R * np.array([np.cos(t), np.sin(t)]))) for t in np.linspace(0, TAU, 721)]
    orbit = VGroup(*[polyline(run, GOLD, 1.8, 0.95) for run in clip_polyline(np.array(circ), INNER)])
    ca = GAM + 0.36
    p = O + R * np.array([np.cos(ca), np.sin(ca)])
    t, n = np.array([-np.sin(ca), np.cos(ca)]), np.array([np.cos(ca), np.sin(ca)])
    chev = polyline([cpx(*(p - 9 * t + 5 * n)), cpx(*p), cpx(*(p - 9 * t - 5 * n))], GOLD, 1.7)
    orbit.add(chev)

    lineA, lineB = T + radial * 170, O - radial * 70
    line = Line(cpx(*lineA), cpx(*lineB), stroke_color=INK2, stroke_width=1.5)
    Op = cpx(*O)
    zero = VGroup(Line(Op + 0.071 * LEFT, Op + 0.071 * RIGHT, stroke_color=INK2, stroke_width=1.5),
                  Line(Op + 0.071 * DOWN, Op + 0.071 * UP, stroke_color=INK2, stroke_width=1.5),
                  serif("0", 26, INK2, italic=True).move_to(cpx(O[0] + 17, O[1] + 13)))
    return dict(cone=cone, orbit=orbit, line=line, zero=zero, T=Tp)


def title_block(size=45.0, subtitle=True):
    """'A Categorical Atlas of / Parameter-Efficient / Fine-Tuning / Cones, Orbits, and Gauges', as on the card."""
    head = VGroup(serif("A Categorical Atlas of", size, INK, SEMIBOLD),
                  serif("Parameter-Efficient", size, TIDE, SEMIBOLD),
                  serif("Fine-Tuning", size, TIDE, SEMIBOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15 * size / 45)
    if not subtitle:
        return VGroup(head)
    sub = serif("Cones, Orbits, and Gauges", size * 0.68, TIDE, italic=True)
    return VGroup(head, sub).arrange(DOWN, aligned_edge=LEFT, buff=0.30 * size / 45)


# ---------------------------------------------------------------- the nut and its tools (P.1)
SHELL, SHELL_DARK, SHELL_SOFT, SHELL_SOFT_DARK = "#B98B52", "#7E5C33", "#CDAB82", "#8F7050"
KERNEL, KERNEL_DARK = "#E6D3AA", "#B99A6B"
CRACK = "#3A2814"
METAL, METAL_DARK, HANDLE = "#A3B0BD", "#5E6C7B", "#7D8B99"
TERRAIN = "#111B28"


def _split_at_seam(pts, cx):
    """Runs of a polyline on the left (x < cx) and on the right of the seam."""
    left, right, cur, side = [], [], [], None
    for p in pts:
        s = p[0] < cx
        if side is not None and s != side and len(cur) > 1:
            (left if side else right).append(cur)
            cur = [cur[-1]]
        cur.append(p)
        side = s
    if len(cur) > 1:
        (left if side else right).append(cur)
    return left, right


def big_nut(center, r):
    """The shore's nut icon (story.nut_icon) drawn large: an ellipse with a seam and nested pointed arches, plus a
    little light and shade. It is built as two halves that can part along the seam, with a kernel inside.
    Returns dict(left, right, kernel, shadow, ax, ay, center); each half is VGroup(shell, lines, light)."""
    c = np.array(center, dtype=float)
    ax, ay = 1.15 * r, r
    arch_l, arch_r = [], []
    for k in (0.45, 0.74, 1.03):
        for s in (1, -1):
            th = np.linspace(PI / 2 + s * 0.5, PI / 2 + s * 1.7, 50)
            pts = [c + s * r * 0.35 * RIGHT + r * k * np.array([np.cos(t), np.sin(t), 0]) for t in th]
            lft, rgt = _split_at_seam(pts, c[0])
            arch_l += lft
            arch_r += rgt

    def half(sign):
        th = np.linspace(PI / 2, 3 * PI / 2, 64) if sign < 0 else np.linspace(-PI / 2, PI / 2, 64)
        rim = [c + np.array([ax * np.cos(t), ay * np.sin(t), 0]) for t in th]
        shell = VMobject().set_points_smoothly(rim)
        shell.add_line_to(rim[0])
        shell.set_fill(SHELL, 1).set_stroke(SHELL_DARK, 2.4)
        lines = VGroup(*[polyline(p, SHELL_DARK, 1.9, 0.9) for p in (arch_l if sign < 0 else arch_r)])
        if sign < 0:
            light = VMobject().set_points_smoothly([c + np.array([0.80 * ax * np.cos(t), 0.80 * ay * np.sin(t), 0])
                                                    for t in np.linspace(1.95, 2.75, 12)]).set_stroke("#DDB57E", 5, 0.45)
        else:
            light = VMobject().set_points_smoothly([c + np.array([0.86 * ax * np.cos(t), 0.86 * ay * np.sin(t), 0])
                                                    for t in np.linspace(-1.35, -0.25, 14)]).set_stroke("#9A7140", 7, 0.5)
        return VGroup(shell, lines, light)

    lobe = lambda s: Ellipse(width=0.86 * ax, height=1.46 * ay, fill_color=KERNEL, fill_opacity=1,
                             stroke_color=KERNEL_DARK, stroke_width=1.6).move_to(c + s * 0.40 * ax * RIGHT)
    kernel = VGroup(lobe(-1), lobe(1))
    for s in (-1, 1):
        kernel.add(VMobject().set_points_smoothly([c + np.array([s * (0.40 + 0.12 * np.sin(4 * t)) * ax, 0.55 * ay * t, 0])
                                                    for t in np.linspace(-0.9, 0.9, 12)]).set_stroke(KERNEL_DARK, 1.4, 0.8))
    shadow = Ellipse(width=2.1 * ax, height=0.22 * ay, fill_color="#04080D", fill_opacity=0.6,
                     stroke_width=0).move_to(c + 0.97 * ay * DOWN)
    return dict(left=half(-1), right=half(1), kernel=kernel, shadow=shadow, ax=ax, ay=ay, center=c)


def chisel(L=1.6):
    """A cold chisel lying along +x with its tip at the origin (the butt at x = -L)."""
    bl = 0.45
    blade = Polygon([0, 0, 0], [-bl, 0.09, 0], [-bl, -0.09, 0], fill_color=METAL, fill_opacity=1,
                    stroke_color=METAL_DARK, stroke_width=1.4)
    shaft = Rectangle(width=L - bl, height=0.18, fill_color=METAL, fill_opacity=1, stroke_color=METAL_DARK,
                      stroke_width=1.4).move_to([-bl - (L - bl) / 2, 0, 0])
    cap = RoundedRectangle(corner_radius=0.04, width=0.11, height=0.25, fill_color=METAL, fill_opacity=1,
                           stroke_color=METAL_DARK, stroke_width=1.4).move_to([-L + 0.02, 0, 0])
    return VGroup(shaft, blade, cap)


def hammer(face_x, handle=1.7):
    """A hammer whose striking face is at x = face_x (it strikes along +x), handle pointing down (-y).
    Returns (group, pivot point at the end of the handle)."""
    head = RoundedRectangle(corner_radius=0.06, width=0.85, height=0.38, fill_color=METAL, fill_opacity=1,
                            stroke_color=METAL_DARK, stroke_width=1.5).move_to([face_x - 0.425, 0, 0])
    face = Line([face_x, 0.16, 0], [face_x, -0.16, 0], stroke_color=METAL_DARK, stroke_width=2.4)
    hx = face_x - 0.47
    shaft = RoundedRectangle(corner_radius=0.06, width=0.14, height=handle, fill_color=HANDLE, fill_opacity=1,
                             stroke_color=METAL_DARK, stroke_width=1.3).move_to([hx, -0.18 - handle / 2, 0])
    return VGroup(shaft, head, face), np.array([hx, -0.18 - handle + 0.1, 0])


def finger():
    """A pointing finger seen from the front, tip at the origin, entering from above (line art)."""
    w, top = 0.25, 6.5
    pts = [np.array([-w, top, 0]), np.array([-w * 1.02, 1.6, 0]), np.array([-w * 0.98, 0.55, 0]),
           np.array([-w * 0.80, 0.16, 0]), np.array([-w * 0.42, 0.02, 0]), np.array([0, 0, 0]),
           np.array([w * 0.42, 0.02, 0]), np.array([w * 0.80, 0.16, 0]), np.array([w * 0.98, 0.55, 0]),
           np.array([w * 1.04, 1.6, 0]), np.array([w, top, 0])]
    body = VMobject().set_points_smoothly(pts)
    body.add_line_to(pts[0])
    body.set_fill(BG2, 1).set_stroke(INK2, 1.8)
    nail = VMobject().set_points_smoothly([np.array([-0.14, 0.66, 0]), np.array([-0.145, 0.32, 0]),
                                           np.array([-0.075, 0.13, 0]), np.array([0, 0.10, 0]),
                                           np.array([0.075, 0.13, 0]), np.array([0.145, 0.32, 0]),
                                           np.array([0.14, 0.66, 0])]).set_stroke(INK3, 1.3)
    creases = VGroup(*[Arc(radius=0.17, start_angle=PI * 1.2, angle=PI * 0.6, stroke_color=INK3, stroke_width=1.2)
                       .move_to([0, y, 0]) for y in (1.10, 1.22, 2.15)])
    return VGroup(body, nail, creases)


def boulder(cx, top, hw, base, seed, fill=BG3):
    """A rounded boulder from (cx - hw, base) over a broad top at height `top`."""
    rng = np.random.default_rng(seed)
    H = top - base
    prof = [(-1.00, 0.00), (-0.98, 0.42), (-0.84, 0.80), (-0.50, 0.97), (0.0, 1.0), (0.48, 0.96), (0.82, 0.80),
            (0.97, 0.45), (1.00, 0.00)]
    pts = []
    for k, (x, y) in enumerate(prof):
        j = 0 if k in (0, len(prof) - 1) else 1
        pts.append(np.array([cx + hw * (x + j * 0.04 * rng.normal()), base + H * min(1.0, y + j * 0.03 * rng.normal()), 0]))
    m = VMobject().set_points_smoothly(pts)
    m.add_line_to(pts[0])
    m.set_fill(fill, 1).set_stroke(RULE, 1.6)
    facet = VMobject().set_points_smoothly([np.array([cx - 0.72 * hw, base + 0.80 * H, 0]),
                                            np.array([cx - 0.40 * hw, base + 0.93 * H, 0]),
                                            np.array([cx + 0.05 * hw, base + 0.95 * H, 0])]).set_stroke("#24364E", 2.2, 0.9)
    return VGroup(m, facet)


def small_nut(top_pt, r):
    """A shore nut icon resting on a point."""
    return nut_icon(r).move_to(top_pt + np.array([0, r * 0.95, 0]))


def keyed(keys):
    """A function of time through (t, value) keys, eased (smooth) between consecutive keys, constant outside."""
    def f(t):
        if t <= keys[0][0]:
            return keys[0][1]
        for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
            if t <= t1:
                return v0 + (v1 - v0) * smooth((t - t0) / max(t1 - t0, 1e-6))
        return keys[-1][1]
    return f


# ---------------------------------------------------------------- the chapter
class Opening(Chapter):
    BEATS = ["0.1", "0.2", "0.3", "P.1", "P.2"]
    PRE = {"0.1": 1.5}
    EXTRA = {"0.2": 1.0, "P.2": 2.0}

    def construct(self):
        self.cold_open()
        self.names()
        self.title_card()
        self.hammer_or_sea()
        self.four_nuts()
        self.finish()

    # ============================================================ 0.1 · Two runs, one path
    def cold_open(self):
        d = np.load(SIM / "lora_plus.npz")
        P, L = (d["P_plus"], d["P_alpha"]), (d["L_plus"], d["L_alpha"])
        N, lam = len(P[0]), float(d["lam"])
        gap = np.maximum.accumulate(np.abs(d["W_plus"] - d["W_alpha"]).reshape(N, -1).max(axis=1))
        assert gap[-1] == float(d["same"]), "running max disagrees with lora_plus.npz['same']"
        eps = json.loads((SIM / "lora_plus_eps.json").read_text())
        assert eps["same"] == 0.0 and eps["lam"] == lam and eps["steps"] == N, "footnote claim not reproduced"
        lam_s = f"{lam:g}"

        # step index as a function of scene time: drawn from 'Train both with Adam' to the end of 'step after step'
        t_go = self.cue("0.1", "Train both with Adam") + 1.0
        t_end = self.cue("0.1", "step after step", end=True) - 0.25

        def k_now():
            u = np.clip((self.renderer.time - t_go) / (t_end - t_go), 0.0, 1.0)
            return (N - 1) * u ** 1.7

        # panel geometry
        PW, PH, CY = 6.0, 5.0, 0.85
        anchors = [ValueTracker(-3.25), ValueTracker(3.25)]
        center = lambda j: np.array([anchors[j].get_value(), CY, 0.0])
        allP = np.vstack(P)
        lo, hi = allP.min(0), allP.max(0)
        sig = min(5.0 / (hi[0] - lo[0]), 2.4 / (hi[1] - lo[1]))
        mid = (lo + hi) / 2
        PATH_C, LOSS_C, LW, LH = np.array([0.0, 0.42, 0.0]), np.array([0.0, -1.68, 0.0]), 5.0, 0.6
        Lmin, Lmax = min(l.min() for l in L), max(l.max() for l in L)

        def to_screen(c, pts2):
            out = np.zeros((len(pts2), 3))
            out[:, :2] = (pts2 - mid) * sig
            return out + c + PATH_C

        def loss_screen(c, idx, vals):
            x = -LW / 2 + LW * np.asarray(idx) / (N - 1)
            y = -LH / 2 + LH * (np.asarray(vals) - Lmin) / (Lmax - Lmin)
            return np.stack([x, y, 0 * x], 1) + c + LOSS_C

        STY = [dict(color=TIDE, w=5.0, op=0.9, dot=0.085, lw=3.2),
               dict(color=INK, w=1.8, op=1.0, dot=0.042, lw=1.3)]

        def panel_static(j):
            c = center(j)
            frame = RoundedRectangle(corner_radius=0.12, width=PW, height=PH, fill_color=BG2, fill_opacity=1,
                                     stroke_color=RULE, stroke_width=1.2).move_to(c)
            start = Circle(radius=0.07, stroke_color=INK3, stroke_width=1.5).move_to(to_screen(c, np.zeros((1, 2)))[0])
            cap = sans("weights W(t), 2-D projection", 15, INK3)
            cap.move_to(c + PATH_C + np.array([LW / 2, -1.27, 0]), aligned_edge=RIGHT)
            base = Line(c + LOSS_C + np.array([-LW / 2, -LH / 2 - 0.05, 0]), c + LOSS_C + np.array([LW / 2, -LH / 2 - 0.05, 0]),
                        stroke_color=RULE, stroke_width=1.2)
            lab = sans("loss", 15, INK3).move_to(c + LOSS_C + np.array([LW / 2, LH / 2 + 0.05, 0]), aligned_edge=RIGHT)
            frame.set_z_index(0)
            for m in (start, cap, base, lab):
                m.set_z_index(1)
            return VGroup(frame, start, cap, base, lab)

        def title(j):
            name, rest = (("LoRA+", f", λ = {lam_s}"), ("LoRA", f", α × {lam_s}"))[j]
            t = markup([(name, (TIDE, INK)[j]), (rest, INK2)], 24)
            sw = Line(ORIGIN, 0.38 * RIGHT).set_stroke(STY[j]["color"], STY[j]["w"], STY[j]["op"])
            g = VGroup(sw, t).arrange(RIGHT, buff=0.16)
            g.move_to(center(j) + np.array([-PW / 2 + 0.36, PH / 2 - 0.42, 0]), aligned_edge=LEFT)
            return g.set_z_index(6)

        def path_mob(j):
            st = STY[j]
            m = VMobject().set_stroke(st["color"], st["w"], st["op"]).set_z_index(2 + j)

            def upd(m, dt):
                k = k_now()
                i = int(k)
                f = k - i
                pts = P[j][: i + 1]
                if i + 1 < N and f > 1e-6:
                    pts = np.vstack([pts, P[j][i] + f * (P[j][i + 1] - P[j][i])])
                if len(pts) < 2:
                    pts = np.vstack([pts, pts + 1e-4])
                m.set_points_as_corners(to_screen(center(j), pts))
            m.add_updater(upd)
            upd(m, 0)
            return m

        def dot_mob(j):
            st = STY[j]
            m = Dot(radius=st["dot"], color=st["color"]).set_z_index(4 + j)

            def upd(m, dt):
                k = k_now()
                i = min(int(k), N - 2)
                f = k - i
                m.move_to(to_screen(center(j), (P[j][i] + f * (P[j][i + 1] - P[j][i]))[None])[0])
            m.add_updater(upd)
            upd(m, 0)
            return m

        def loss_mob(j):
            st = STY[j]
            m = VMobject().set_stroke(st["color"], st["lw"], st["op"]).set_z_index(2 + j)

            def upd(m, dt):
                k = k_now()
                i = int(k)
                idx = np.arange(i + 1, dtype=float)
                vals = L[j][: i + 1]
                if i + 1 < N and k - i > 1e-6:
                    idx = np.append(idx, k)
                    vals = np.append(vals, L[j][i] + (k - i) * (L[j][i + 1] - L[j][i]))
                if len(idx) < 2:
                    idx, vals = np.append(idx, idx[-1] + 1e-3), np.append(vals, vals[-1])
                m.set_points_as_corners(loss_screen(center(j), idx, vals))
            m.add_updater(upd)
            upd(m, 0)
            return m

        statics = [panel_static(0), panel_static(1)]
        titles = [title(0), title(1)]

        # readouts under the panels: [step ### / N]   [max |W_LoRA+ − W_LoRA(α×16)| = value]
        ROW = -2.32
        ro_label = f"max |W_LoRA+ − W_LoRA(α×{lam_s})| = "
        step_cache, ro_cache = {}, {}

        def step_text(i):
            if i not in step_cache:
                step_cache[i] = markup([("step ", INK3), (f"{i:>3}", INK2), (f" / {N}", INK3)], 20, MONO)
            return step_cache[i]

        def ro_text(s):
            if s not in ro_cache:
                ro_cache[s] = markup([(ro_label, INK3), (s, TIDE)], 22, MONO)
            return ro_cache[s]

        step_mob = step_text(0).copy()
        ro_mob = ro_text(repr(float(gap[0]))).copy()
        row = VGroup(step_mob, ro_mob).arrange(RIGHT, buff=0.9)
        row.move_to([0, ROW, 0])
        step_left, ro_left = step_mob.get_left(), ro_mob.get_left()
        shown = dict(step=0, ro=repr(float(gap[0])))

        def step_upd(m, dt):
            i = int(k_now()) + 1 if self.renderer.time >= t_go else 0
            if i != shown["step"]:
                shown["step"] = i
                m.become(step_text(i).copy().move_to(step_left, aligned_edge=LEFT))

        def ro_upd(m, dt):
            s = repr(float(gap[int(k_now())]))            # the running max up to the current step
            if s != shown["ro"]:
                shown["ro"] = s
                m.become(ro_text(s).copy().move_to(ro_left, aligned_edge=LEFT))

        settings = sans(f"Adam, ε = 0 · no weight decay · no clipping · B₀ = 0 · shared schedule · {N} steps", 16, INK3)
        settings.move_to([0, -2.92, 0])
        foot = credit_line(f"ε = 0 with the convention 0/0 := 0 (stock Adam: ε on A and {lam_s}ε on B, still bit-exact)"
                           f" · after Schulman et al. 2025", 16)
        foot.move_to([0, -3.45, 0])

        # ---- timeline
        self.at(self.T("0.1"))
        self.play(FadeIn(statics[0]), FadeIn(statics[1]), run_time=1.2)
        self.at(self.cue("0.1", "One is LoRA-plus"))
        self.play(FadeIn(titles[0], shift=0.12 * UP), run_time=0.8)
        self.at(self.cue("0.1", "The other is plain LoRA"))
        self.play(FadeIn(titles[1], shift=0.12 * UP), run_time=0.8)
        self.at(self.cue("0.1", "Train both with Adam"))
        self.play(FadeIn(settings), FadeIn(row), run_time=0.7)
        self.at(t_go - 0.2)
        movers = [path_mob(0), loss_mob(0), path_mob(1), loss_mob(1)]
        dots = [dot_mob(0), dot_mob(1)]
        step_mob.add_updater(step_upd)
        ro_mob.add_updater(ro_upd)
        self.add(*movers)
        self.play(FadeIn(dots[0]), FadeIn(dots[1]), run_time=0.4)
        self.at(self.cue("0.1", "with epsilon"))
        self.play(FadeIn(foot), run_time=0.8)

        # the panels slide together and overlay
        self.at(self.cue("0.1", "They're identical"))
        dx = 3.25
        tgt = titles[0].copy().shift(dx * RIGHT)
        right_target = titles[1].copy().next_to(tgt, RIGHT, buff=0.4).align_to(tgt, DOWN)
        self.play(anchors[0].animate.set_value(0.0), anchors[1].animate.set_value(0.0),
                  statics[0].animate.shift(dx * RIGHT), titles[0].animate.shift(dx * RIGHT),
                  FadeOut(statics[1], shift=dx * LEFT), titles[1].animate.move_to(right_target),
                  run_time=1.5, rate_func=smooth)
        self.at(self.cue("0.1", "Not close"))
        val = ro_mob[-len(shown["ro"]):]
        under = Line(val.get_corner(DL) + 0.09 * DOWN, val.get_corner(DR) + 0.09 * DOWN,
                     stroke_color=TIDE, stroke_width=2.2)
        self.play(Create(under), run_time=0.6)
        self.at(t_end + 0.3)

    # ============================================================ 0.2 · 382 names
    def names(self):
        data = json.loads((SIM / "atlas_names.json").read_text(encoding="utf-8"))
        names, count = data["names"], data["count"]
        assert count == len(names) == 382, f"atlas count is {count}; the narration says 382"

        # the overlaid paths fade
        self.at(self.cue("0.2", "That isn't luck") + 0.1)
        old = list(self.mobjects)
        if old:
            self.play(*[FadeOut(m) for m in old], run_time=1.2)
            for m in old:
                m.clear_updaters()

        famous = ["LoRA", "DoRA", "OFT", "BOFT", "VeRA", "(IA)³", "BitFit", "QLoRA", "GaLore", "HRA", "LoHa", "LoKr",
                  "AdaLoRA", "Prefix-Tuning"]
        last = "LoRA+"
        assert all(f in names for f in famous + [last])
        rng = np.random.default_rng(382)
        rest = [n for n in names if n not in famous and n != last]
        rng.shuffle(rest)
        order = famous + rest + [last]
        assert len(order) == count

        def card(txt, size, fg, stroke):
            t = Text(txt, font=SANS, font_size=size, color=fg)
            s = size / 15
            box = RoundedRectangle(corner_radius=0.045 * s, width=t.width + 0.22 * s, height=0.32 * s, fill_color=BG2,
                                   fill_opacity=1, stroke_color=stroke, stroke_width=1.0)
            t.move_to(box)
            return VGroup(box, t)

        cards = []
        for i, n in enumerate(order):
            if n in famous or n == last:
                cards.append(card(n, 20, INK, "#30455F"))
            else:
                cards.append(card(n, 15, (INK2, INK3)[i % 2], RULE))

        # ---- where each card lands: a heap built on a height map
        GROUND, X0, X1, DXC = -3.62, -6.75, 6.75, 0.04
        cols = np.arange(X0, X1 + DXC, DXC)
        H = np.full(len(cols), GROUND)
        fx = list(np.linspace(-5.4, 5.4, len(famous)))
        rng.shuffle(fx)
        land_xy, land_ang = [], []
        for i, (n, c) in enumerate(zip(order, cards)):
            w, h = c.width, c.height
            if n in famous:
                x = fx[famous.index(n)]
            elif n == last:
                x = 0.0
            else:
                x = float(np.clip(rng.normal(0.0, 3.2), X0 + w / 2 + 0.05, X1 - w / 2 - 0.05))
            span = (cols >= x - w / 2) & (cols <= x + w / 2)
            yb = H[span].max()
            land_xy.append(np.array([x, yb + h / 2, 0.0]))
            land_ang.append(0.0 if n == last else float(rng.uniform(-10, 10) * DEGREES))
            H[span] = yb + (0.03 if n not in famous else 0.042)
            # let the heap settle a little (no towers)
            for _ in range(2):
                H[1:-1] = np.maximum(H[1:-1], 0.5 * (H[:-2] + H[2:]) - 0.02)
        land_xy = np.array(land_xy)

        # ---- when each card falls
        tf0 = self.cue("0.2", "it isn't the only")
        t_last = self.cue("0.2", "eighty-two")
        nf, nr = len(famous), len(rest)
        spawn = np.zeros(count)
        dur = np.zeros(count)
        for i in range(nf):
            spawn[i], dur[i] = tf0 + 0.36 * i, 1.7
        ts0, ts1 = tf0 + 0.36 * nf - 1.2, t_last - 1.45
        for j in range(nr):
            spawn[nf + j] = ts0 + (ts1 - ts0) * ((j + 1) / nr) ** 0.62
            dur[nf + j] = rng.uniform(0.95, 1.25)
        spawn[-1], dur[-1] = t_last - 1.3, 1.3
        landing = spawn + dur
        assert landing[:-1].max() < landing[-1], "LoRA+ must land last"

        COUNTER_BOX = (-7.0, -2.3, 2.55, 4.1)                       # keep the rain out of the counter
        start_xy = []
        for i, c in enumerate(cards):
            x = land_xy[i][0]
            y0 = 4.45 + 0.35 * rng.random()
            if x + c.width / 2 > COUNTER_BOX[0] and x - c.width / 2 < COUNTER_BOX[1]:
                y0 = 2.35
            start_xy.append(np.array([x, y0, 0.0]))
        start_ang = rng.uniform(-25, 25, count) * DEGREES
        state = ["wait"] * count
        cur_ang = np.zeros(count)
        for i, c in enumerate(cards):
            c.move_to(start_xy[i])

        heap = VGroup()
        landed, falling = [], []

        def heap_upd(m, dt):
            t = self.renderer.time
            changed = False
            for i in range(count):
                if state[i] == "done":
                    continue
                if state[i] == "wait":
                    if t < spawn[i]:
                        continue
                    state[i] = "fall"
                    falling.append(i)
                    changed = True
                c = cards[i]
                p = min(max((t - spawn[i]) / dur[i], 0.0), 1.0)
                ang = start_ang[i] + (land_ang[i] - start_ang[i]) * p
                pos = start_xy[i] + (land_xy[i] - start_xy[i]) * p ** 1.6
                c.rotate(ang - cur_ang[i])
                cur_ang[i] = ang
                c.move_to(pos)
                if start_xy[i][1] < 4.0:                   # cards that start inside the frame fade in
                    c.set_opacity(min(1.0, p / 0.12))
                if p >= 1.0:
                    state[i] = "done"
                    c[0].set_fill(BG2, 1).set_stroke(opacity=1)
                    c[1].set_fill(opacity=1)
                    falling.remove(i)
                    landed.append(i)
                    changed = True
            if changed:
                m.submobjects = [cards[i] for i in landed] + [cards[i] for i in falling]

        # the counter
        cnt = Ticker(60, INK, [-6.70 + 3 * 0.5, 3.25, 0])
        cnt.shift((-6.70 - cnt.get_left()[0]) * RIGHT)    # the hundreds slot starts at the margin
        cnt_lab = kicker("methods in the atlas", INK3, 16).move_to([-6.70, 2.72, 0], aligned_edge=LEFT)

        def cnt_upd(m, dt):
            m.set_value(int(np.sum(landing <= self.renderer.time)))

        self.at(tf0 - 0.5)
        self.play(FadeIn(cnt), FadeIn(cnt_lab), run_time=0.6)
        heap.add_updater(heap_upd)
        cnt.add_updater(cnt_upd)
        self.add(heap)
        self.at(landing[-1] + 0.05)
        heap_upd(heap, 0)
        heap.clear_updaters()
        cnt.clear_updaters()
        cnt.set_value(int(np.sum(landing <= landing[-1] + 1e-9)))
        assert cnt.value == count
        self.play(cnt.animate.set_color(TIDE), run_time=0.5)

        # one card flips: two empty slots (schematic)
        self.at(self.cue("0.2", "We usually compare"))
        veil = Rectangle(width=15, height=9, fill_color=BG, fill_opacity=0.0, stroke_width=0).set_z_index(1)
        CW, CH, CC = 5.3, 2.5, np.array([0.0, 1.45, 0.0])
        front_box = RoundedRectangle(corner_radius=0.12, width=CW, height=CH, fill_color=BG2, fill_opacity=1,
                                     stroke_color="#30455F", stroke_width=1.6).move_to(CC)
        front_txt = Text(last, font=SANS, font_size=60, color=INK).move_to(CC)
        front = VGroup(front_box, front_txt).set_z_index(2)
        small = cards[-1]
        heap.remove(small)
        small.set_z_index(2)
        self.add(small)
        self.play(veil.animate.set_fill(opacity=0.66), Transform(small[0], front_box), FadeTransform(small[1], front_txt),
                  run_time=1.2)
        self.remove(small, *small, front_txt)          # FadeTransform split `small` into top-level parts
        self.add(front)

        back_box = front_box.copy()
        back_kick = kicker(last, INK3, 15).move_to(CC + np.array([-CW / 2 + 0.35, CH / 2 - 0.32, 0]), aligned_edge=LEFT)
        rows, slots = VGroup(), []
        for k, lab in enumerate(("trainable parameters", "benchmark score")):
            y = CC[1] + 0.32 - 0.72 * k
            t = sans(lab, 22, INK2).move_to([CC[0] - CW / 2 + 0.35, y, 0], aligned_edge=LEFT)
            slot = RoundedRectangle(corner_radius=0.06, width=1.45, height=0.48, stroke_color=INK3, stroke_width=1.4,
                                    fill_opacity=0).move_to([CC[0] + CW / 2 - 0.35 - 0.725, y, 0])
            dslot = DashedVMobject(slot, num_dashes=26, dashed_ratio=0.55)
            rows.add(VGroup(t, dslot))
            slots.append((t, dslot))
        tag = schematic_tag().move_to(CC + np.array([CW / 2 - 0.35, -CH / 2 + 0.25, 0]), aligned_edge=RIGHT)
        back = VGroup(back_box, back_kick, rows, tag).set_z_index(2)

        self.at(self.cue("0.2", "two numbers") - 0.15)
        self.play(front.animate.stretch(0.03, 0), run_time=0.3, rate_func=rate_functions.ease_in_sine)
        self.remove(front)
        back.stretch(0.03, 0)
        self.add(back)
        self.play(back.animate.stretch(1 / 0.03, 0), run_time=0.3, rate_func=rate_functions.ease_out_sine)

        for k, cue in enumerate(("how many parameters", "how well they score")):
            self.at(self.cue("0.2", cue))
            t, ds = slots[k]
            self.play(t.animate.set_color(INK), ds.animate.set_stroke(TIDE, 2.0), run_time=0.6)
        self.at(self.cue("0.2", "Those numbers"))
        self.play(*[t.animate.set_color(INK2) for t, _ in slots], *[ds.animate.set_stroke(INK3, 1.4) for _, ds in slots],
                  run_time=0.8)

        # compare by their shape: the heap goes, the cone of the title card appears
        self.at(self.cue("0.2", "To see it"))
        self.play(FadeOut(heap), FadeOut(cnt), FadeOut(cnt_lab), veil.animate.set_fill(opacity=0), run_time=1.4)
        self.remove(veil)
        self.at(self.cue("0.2", "by their shape"))
        mo = motif()
        dot, halo, lab = self.theta0(mo["T"], label=True, label_dir=LEFT)
        lab.shift(0.05 * LEFT)
        self.play(FadeOut(back, shift=0.3 * LEFT), run_time=0.6)
        self.play(LaggedStart(*[Create(r) for r in mo["cone"][1]], lag_ratio=0.02), FadeIn(mo["cone"][0]),
                  Create(mo["cone"][2]), FadeIn(dot), FadeIn(halo), FadeIn(lab), run_time=1.5)
        self.title_motif = (mo, dot, halo, lab)

    # ============================================================ 0.3 · Title (wordless)
    def title_card(self):
        t0, t1 = self.T("0.3"), self.END("0.3")
        if not hasattr(self, "title_motif"):
            mo = motif()
            dot, halo, lab = self.theta0(mo["T"], label=True, label_dir=LEFT)
            lab.shift(0.05 * LEFT)
            self.add(*mo["cone"], dot, halo, lab)
            self.title_motif = (mo, dot, halo, lab)
        mo, dot, halo, lab = self.title_motif
        glow, cont, neat = card_glow(), contours(), neatline()
        for m in (glow, cont, neat):
            m.set_z_index(-1)
        tb = title_block(45)
        tb.move_to([cpx(80, 0)[0], 0.45, 0], aligned_edge=LEFT)
        self.at(t0)
        self.play(FadeIn(glow), FadeIn(cont), FadeIn(neat),
                  LaggedStart(*[FadeIn(m, shift=0.1 * UP) for m in (*tb[0], tb[1])], lag_ratio=0.25),
                  Create(mo["orbit"]), Create(mo["line"]), FadeIn(mo["zero"]), run_time=2.2)
        self.at(t1 - 1.3)
        everything = [m for m in self.mobjects]
        self.play(*[FadeOut(m) for m in everything], run_time=1.2)

    # ============================================================ P.1 · Hammer or sea
    def hammer_or_sea(self):
        cue = lambda p, end=False: self.cue("P.1", p, end=end)
        cam = self.camera
        cam.set_zoom(1.0)
        cam.frame_center = ORIGIN

        # ---- the world: at zoom 1 the frame shows the close-up; the landscape lies around it, out of frame
        NC = np.array([0.0, -0.45, 0.0])
        nut = big_nut(NC, 1.25)
        ax, ay = nut["ax"], nut["ay"]
        big_rock = boulder(0.15, NC[1] - ay + 0.04, 3.8, -7.4, 1)
        tx = np.linspace(-38, 38, 160)                          # a valley: low under the close-up, rising outward
        ty = -7.0 + 0.4 * np.maximum(0, np.abs(tx) - 7) + 0.35 * np.sin(0.31 * tx + 0.4) + 0.2 * np.sin(0.71 * tx + 1.3)
        tpts = [np.array([x, y, 0]) for x, y in zip(tx, ty)]
        terrain = VMobject().set_points_smoothly(tpts)
        for p in (np.array([38, -40, 0]), np.array([-38, -40, 0]), tpts[0]):
            terrain.add_line_to(p)
        terrain.set_fill(TERRAIN, 1).set_stroke(RULE, 1.2)
        specs = [(-26.0, 2.6, 2.6, 0.72), (-19.5, 3.0, 3.0, 0.80), (-13.0, 3.2, 2.6, 0.72), (-6.4, 1.3, 2.2, 0.66),
                 (6.6, 1.2, 2.0, 0.66), (10.6, 3.4, 2.4, 0.78), (16.0, 3.0, 3.0, 0.72), (21.5, 3.0, 2.8, 0.80),
                 (27.0, 2.4, 2.4, 0.70)]        # (x, height above the ground, half-width, nut radius); none in the close-up
        rocks, nuts = VGroup(), VGroup()
        for k, (x, hgt, hw, r) in enumerate(specs):
            ground = float(np.interp(x, tx, ty))
            top = ground + hgt
            assert abs(x) - hw > 7.3 or top + 2 * r < -4.1, "landscape rock visible in the close-up"
            rocks.add(boulder(x, top, hw, ground - 0.8, 10 + k))
            nuts.add(small_nut(np.array([x + 0.15 * hw * np.sin(2.3 * k), top - 0.04, 0]), r))
        for m, z in ((terrain, 0), (rocks, 1), (nuts, 2), (big_rock, 1), (nut["shadow"], 2), (nut["kernel"], 2),
                     (nut["left"], 3), (nut["right"], 3)):
            m.set_z_index(z)

        # ---- the water and the softening shell are functions of time, so they run under everything else
        t_soak, t_press = cue("soak it"), cue("pressure of a hand") + 0.3
        t_large, t_rise, t_diss = cue("On a larger scale"), cue("rising sea"), cue("dissolves into a theory")
        level_f = keyed([(t_soak, -5.2), (t_press, -0.55), (t_large, -0.42), (t_large + 3.5, -0.10),
                         (t_rise, -0.05), (t_diss - 0.15, 4.2)])
        soft_f = keyed([(t_soak + 0.5, 0.0), (t_press, 1.0)])

        def surface_y(xs, t):
            return level_f(t) + 0.06 * np.sin(1.3 * xs + 1.1 * t) + 0.035 * np.sin(2.9 * xs - 1.76 * t)

        def water_poly():
            z, fc, t = cam.get_zoom(), cam.frame_center, self.renderer.time
            xs = np.linspace(fc[0] - 7.8 / z, fc[0] + 7.8 / z, 120)
            top = [np.array([x, y, 0]) for x, y in zip(xs, surface_y(xs, t))]
            return Polygon(*top, np.array([xs[-1], fc[1] - 6 / z, 0]), np.array([xs[0], fc[1] - 6 / z, 0]),
                           stroke_color=TIDE, stroke_width=2, stroke_opacity=0.7, fill_color=TIDE, fill_opacity=0.16)

        water = water_poly().set_z_index(5)
        water.add_updater(lambda m, dt: m.become(water_poly()))
        marks = VGroup()                                       # the hammer's cracks, which soften away

        def soften(m, dt):
            s = soft_f(self.renderer.time)
            for h in (nut["left"], nut["right"]):
                h[0].set_fill(interpolate_color(ManimColor(SHELL), ManimColor(SHELL_SOFT), s), 1)
                h[0].set_stroke(interpolate_color(ManimColor(SHELL_DARK), ManimColor(SHELL_SOFT_DARK), s), 2.4)
                h[1].set_stroke(opacity=0.9 - 0.45 * s)
            marks.set_stroke(opacity=1.0 - s)

        # ---- a nut on a rock
        self.at(self.T("P.1"))
        self.add(terrain, rocks, nuts)
        self.play(FadeIn(big_rock), FadeIn(nut["shadow"]), FadeIn(nut["left"]), FadeIn(nut["right"]), run_time=1.8)

        # ---- hammer and chisel, point after point
        def surface(theta):
            p = NC + np.array([ax * np.cos(theta), ay * np.sin(theta), 0])
            nrm = np.array([np.cos(theta) / ax, np.sin(theta) / ay, 0])
            return p, nrm / np.linalg.norm(nrm)

        angles = [155 * DEGREES, 137 * DEGREES, 119 * DEGREES]
        tool_c = chisel(1.6)
        tool_h, pivot0 = hammer(-1.62)
        piv = Dot(pivot0, radius=0.001).set_opacity(0)
        tip = Dot(ORIGIN, radius=0.001).set_opacity(0)
        tool_h.add(piv)
        rig = VGroup(tool_c, tool_h, tip).set_z_index(8)
        rig_ang = [0.0]

        def set_rig(theta):
            p, nrm = surface(theta)
            ang = np.arctan2(-nrm[1], -nrm[0])                 # the rig's +x points into the nut
            rig.rotate(ang - rig_ang[0], about_point=tip.get_center())
            rig.shift(p - tip.get_center())
            rig_ang[0] = ang

        set_rig(angles[0])
        crng = np.random.default_rng(5)

        def crack(theta, length):
            p, nrm = surface(theta)
            inward, perp = -nrm, np.array([-nrm[1], nrm[0], 0])
            pts, q = [p + 0.01 * inward], p + 0.01 * inward
            for _ in range(5):
                q = q + inward * length / 5 + perp * crng.uniform(-0.08, 0.08)
                pts.append(q)
            b0 = pts[2]
            br = [b0, b0 + 0.14 * inward + 0.13 * perp, b0 + 0.24 * inward + 0.17 * perp]
            return VGroup(polyline(pts, CRACK, 1.8), polyline(br, CRACK, 1.4)).set_z_index(4)

        self.at(cue("hammer and chisel"))
        self.play(FadeIn(rig, shift=0.25 * (UP + LEFT)), run_time=0.7)
        impacts = [cue("and strike", end=True) - 0.2, cue("point after point", end=True) - 0.3,
                   cue("until the shell cracks") + 0.75]
        swing = 0.30
        for k, theta in enumerate(angles):
            if k:
                tr = ValueTracker(angles[k - 1])
                mover = lambda m, dt, tr=tr: set_rig(tr.get_value())
                rig.add_updater(mover)
                self.play(tr.animate.set_value(theta), run_time=0.3, rate_func=smooth)
                rig.remove_updater(mover)
            _, nrm = surface(theta)
            v = tool_h[1].get_center() - piv.get_center()
            sd = 1.0 if np.dot(np.array([-v[1], v[0], 0]), nrm) > 0 else -1.0
            self.at(impacts[k] - 0.42)
            self.play(Rotate(tool_h, sd * swing, about_point=piv.get_center()), run_time=0.3, rate_func=smooth)
            self.play(Rotate(tool_h, -sd * swing, about_point=piv.get_center()), run_time=0.12,
                      rate_func=rate_functions.ease_in_quad)
            ck = crack(theta, 0.55 + 0.08 * k)
            marks.add(ck)
            self.play(Create(ck), run_time=0.22)
        pts = [surface(t)[0] - 0.32 * surface(t)[1] for t in angles]
        joins = VGroup()
        for a, b in zip(pts[:-1], pts[1:]):
            n2 = np.array([-(b - a)[1], (b - a)[0], 0])
            joins.add(polyline([a + (b - a) * s + crng.uniform(-0.05, 0.05) * n2 for s in np.linspace(0, 1, 6)],
                               CRACK, 1.5))
        joins.set_z_index(4)
        marks.add(joins)
        self.at(cue("cracks.") + 0.05)
        self.play(Create(joins), run_time=0.45)
        self.play(FadeOut(rig, shift=0.4 * (UP + LEFT)), run_time=0.6)
        self.remove(rig)
        self.add(marks)

        # ---- soak it: the water rises, very slowly; the shell softens
        self.at(t_soak - 0.1)
        self.add(water)                                     # its updater takes dt, so waits keep the waves moving
        driver = Mobject()
        driver.add_updater(soften)
        self.add(driver)

        # ---- the pressure of a hand
        tip_rest = NC + np.array([0.05, ay + 0.004, 0])
        fing = finger().rotate(8 * DEGREES, about_point=ORIGIN).shift(tip_rest + 4.3 * UP).set_z_index(9)
        self.at(cue("until the pressure"))
        self.add(fing)
        self.play(fing.animate.shift(4.3 * DOWN), run_time=1.3, rate_func=smooth)
        self.at(t_press)
        lv = level_f(t_press)
        rings = VGroup(*[Ellipse(width=3.1, height=0.26, stroke_color=TIDE, stroke_width=1.5, stroke_opacity=0.75)
                         .move_to([NC[0], lv, 0]) for _ in range(2)]).set_z_index(6)
        self.add(rings)
        self.play(fing.animate.shift(0.05 * DOWN), rings[0].animate.scale(1.7).set_stroke(opacity=0),
                  rings[1].animate.scale(2.5).set_stroke(opacity=0), run_time=1.0)
        self.remove(rings)
        self.at(cue("hand is enough") + 0.1)
        driver.clear_updaters()
        self.remove(driver, marks)
        self.add(nut["kernel"])                              # hidden behind the shell until the halves part
        bl, br = NC + np.array([-0.06, -0.96 * ay, 0]), NC + np.array([0.06, -0.96 * ay, 0])
        self.play(Rotate(nut["left"], 16 * DEGREES, about_point=bl), Rotate(nut["right"], -16 * DEGREES, about_point=br),
                  fing.animate.shift(0.3 * UP), run_time=1.4, rate_func=smooth)
        self.play(fing.animate.shift(2.2 * UP).set_opacity(0), run_time=0.6)
        self.remove(fing)

        # ---- a larger scale: pull back; a calm sea rises around a landscape of rocks and small nuts
        self.at(t_large)
        Z1, S0, S1 = 0.26, NC[1], -0.9                      # zoom, and the nut's height on screen before and after
        u = ValueTracker(0.0)

        def camdrive(m, dt):
            s = u.get_value()
            z = float(Z1 ** s)
            cam.set_zoom(z)
            cam.frame_center = NC - np.array([0.0, S0 + (S1 - S0) * s, 0.0]) / z
        cdrv = Mobject()
        cdrv.add_updater(camdrive)
        self.add(cdrv)
        self.play(u.animate.set_value(1.0), run_time=3.5, rate_func=smooth)
        cdrv.clear_updaters()
        self.remove(cdrv)

        # ---- it dissolves into a theory: the shapes melt into the water's surface
        self.at(t_diss)
        lv = level_f(self.renderer.time)
        shapes = [terrain, *rocks, *nuts, big_rock, nut["shadow"], nut["kernel"], nut["left"], nut["right"]]

        def melt(m):
            line = np.array([m.get_center()[0], min(lv, m.get_top()[1]), 0])
            return m.animate.stretch(0.04, 1, about_point=line).set_opacity(0)
        shapes.sort(key=lambda m: abs(m.get_center()[0]))
        self.play(LaggedStart(*[melt(m) for m in shapes], lag_ratio=0.05), run_time=1.85)
        self.remove(*shapes, terrain, rocks, nuts)

        # ... a theory that reaches far beyond it: a light runs out along the surface
        reach = ValueTracker(0.0)

        def glint():
            z, fc, t = cam.get_zoom(), cam.frame_center, self.renderer.time
            half = max(reach.get_value(), 1e-3) * 7.9 / z
            xs = np.linspace(fc[0] - half, fc[0] + half, 100)
            return polyline([np.array([x, y, 0]) for x, y in zip(xs, surface_y(xs, t))], INK, 2.2, 0.55)

        light = glint().set_z_index(7)
        light.add_updater(lambda m, dt: m.become(glint()))
        self.at(cue("reaches far beyond"))
        self.add(light)
        self.play(reach.animate.set_value(1.0), run_time=1.0, rate_func=smooth)
        self.at(self.END("P.1") - 0.7)
        self.play(FadeOut(light), FadeOut(water), run_time=0.6)
        for m in (light, water):
            m.clear_updaters()
            self.remove(m)
        cam.set_zoom(1.0)
        cam.frame_center = ORIGIN

    # ============================================================ P.2 · Four nuts on the shore
    def four_nuts(self):
        self.at(self.T("P.2"))
        s = self.shore(LEVELS[0])
        sea, lvl = s["sea"], s["level"]
        sea.clear_updaters()
        sea.add_updater(lambda m, dt: m.become(wave_polygon(lvl.get_value(), 1.2 * self.renderer.time, 0.22)))
        parts = [s["land"], *s["icons"].values(), *s["links"].values(), *s["cards"].values()]
        self.play(*[FadeIn(p) for p in parts], run_time=1.4)

        def outline(k, color=TIDE, width=2.2, buff=0.07):
            box = s["cards"][k][0]
            return RoundedRectangle(corner_radius=0.08 + buff, width=box.width + 2 * buff, height=box.height + 2 * buff,
                                    stroke_color=color, stroke_width=width, fill_opacity=0).move_to(box)

        def veil(k):
            box = s["cards"][k][0]
            return RoundedRectangle(corner_radius=0.08, width=box.width + 0.04, height=box.height + 0.04,
                                    fill_color=BG, fill_opacity=0.0, stroke_width=0).move_to(box)

        # card 2 glows: "You've already met one"
        self.at(self.cue("P.2", "You've already met one"))
        glow = VGroup(outline(2, TIDE, 14, 0.07).set_stroke(opacity=0.08), outline(2, TIDE, 7, 0.07).set_stroke(opacity=0.16),
                      outline(2, TIDE, 2.4, 0.07))
        ring = Circle(radius=0.42, stroke_color=TIDE, stroke_width=2, stroke_opacity=0.9).move_to(s["icons"][2])
        self.hud(glow, ring)
        glow.set_opacity(0)
        ring.set_stroke(opacity=0)
        base_ops = [0.08, 0.16, 1.0]
        for rep in range(2):
            self.play(*[g.animate.set_stroke(opacity=o) for g, o in zip(glow, base_ops)],
                      ring.animate.set_stroke(opacity=0.9), run_time=0.55)
            self.play(*[g.animate.set_stroke(opacity=o * 0.25) for g, o in zip(glow, base_ops)],
                      ring.animate.set_stroke(opacity=0.2), run_time=0.75)
        self.play(FadeOut(glow), FadeOut(ring), run_time=0.5)

        # each card is highlighted as it is named
        veils = {k: veil(k) for k in (1, 2, 3, 4)}
        self.hud(*veils.values())
        cues = {1: "No exact initialisation", 2: "LoRA-plus is LoRA", 3: "GaLore is one-sided",
                4: "And exactly orthogonal"}
        cur = None
        for k in (1, 2, 3, 4):
            self.at(self.cue("P.2", cues[k]))
            ol = outline(k)
            self.hud(ol)
            ol.set_stroke(opacity=0)
            anims = [ol.animate.set_stroke(opacity=1)]
            for j in (1, 2, 3, 4):
                anims.append(veils[j].animate.set_fill(opacity=0.0 if j == k else 0.55))
            if cur is not None:
                anims.append(cur.animate.set_stroke(opacity=0))
            self.play(*anims, run_time=0.6)
            if cur is not None:
                self.remove(cur)
            cur = ol
        self.at(self.cue("P.2", "Each sounds like"))
        self.play(cur.animate.set_stroke(opacity=0), *[v.animate.set_fill(opacity=0) for v in veils.values()],
                  run_time=0.8)
        self.remove(cur, *veils.values())

        # we'll raise the sea: a gentle ripple
        self.at(self.cue("P.2", "We'll raise the sea"))
        RIP = -3.62                                          # just over the lowest corner of the shore
        rip_c = np.array([-5.4, RIP, 0])
        rips = VGroup(*[Ellipse(width=0.4, height=0.06, stroke_color=TIDE, stroke_width=1.6, stroke_opacity=0.85)
                        .move_to(rip_c) for _ in range(3)])
        self.play(lvl.animate.set_value(RIP), run_time=1.5, rate_func=smooth)
        self.hud(rips)
        self.play(LaggedStart(*[r.animate.stretch_to_fit_width(4.6).stretch_to_fit_height(0.36).set_stroke(opacity=0)
                                for r in rips], lag_ratio=0.32), run_time=2.6)
        self.remove(rips)
        self.at(self.END("P.2") - 0.7)
        self.clear_stage(run_time=0.6)
        sea.clear_updaters()
