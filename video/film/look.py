"""The look of the film ("quiet C"): palette, type, and reusable visual parts.

Rules (see video/README.md): deep navy background, no background texture while mathematics is on screen, a halo only
on theta_0, tide for LoRA and the main accent, gold for OFT and orbits, coral for warnings and obstructions, leaf
green sparingly. Statements and credits go on light paper cards. Every number shown comes from a script in video/sim.
"""
import os
import textwrap

import manimpango
import numpy as np
from manim import *

# ---------------------------------------------------------------- type (Source Serif 4 and IBM Plex from TeX Live)
TEXFONTS = "/usr/local/texlive/2026/texmf-dist/fonts/opentype"
for _f in ("adobe/sourceserif/SourceSerif4-Regular.otf", "adobe/sourceserif/SourceSerif4-Semibold.otf",
           "adobe/sourceserif/SourceSerif4-RegularIt.otf", "adobe/sourceserif/SourceSerif4-SemiboldIt.otf",
           "ibm/plex/IBMPlexSans-Regular.otf", "ibm/plex/IBMPlexSans-Medium.otf",
           "ibm/plex/IBMPlexSans-SemiBold.otf", "ibm/plex/IBMPlexMono-Regular.otf",
           "ibm/plex/IBMPlexMono-Medium.otf"):
    if os.path.exists(os.path.join(TEXFONTS, _f)):
        manimpango.register_font(os.path.join(TEXFONTS, _f))
SERIF, SANS, MONO = "Source Serif 4", "IBM Plex Sans", "IBM Plex Mono"

# ---------------------------------------------------------------- palette
BG, BG2, BG3 = "#0A121C", "#101B28", "#172435"        # background, raised panel, land / track
RULE = "#24344A"                                       # hairlines, axes
INK, INK2, INK3 = "#E3E9EE", "#A3B0BD", "#7D8B99"      # text: primary, secondary, faint
TIDE, GOLD, CORAL, LEAF = "#4CC3CF", "#E2B04C", "#EC7A66", "#7CC276"
VIOLET = "#A99BE6"                                     # a fourth series colour, rarely
# light paper cards (the atlas look)
PAPER, PAPER2, PRULE = "#F2F4F1", "#E8ECE7", "#C7D0CC"
PINK, PINK2, PTIDE, PSEAL = "#13233A", "#46566A", "#0E7385", "#A3392B"

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{sourceserif}\usepackage[varbb,smallerops,cmintegrals]{newtxmath}")


# ---------------------------------------------------------------- text
def serif(txt, size=36, color=INK, weight=NORMAL, italic=False):
    return Text(txt, font=SERIF, font_size=size, color=color, weight=weight, slant=ITALIC if italic else NORMAL)


def sans(txt, size=22, color=INK2, weight=NORMAL):
    return Text(txt, font=SANS, font_size=size, color=color, weight=weight)


def mono(txt, size=22, color=INK2):
    return Text(txt, font=MONO, font_size=size, color=color)


def kicker(txt, color=TIDE, size=18):
    """Small upper-case label, e.g. 'EXPOSÉ II · THE IMAGE'."""
    return Text(txt.upper(), font=SANS, font_size=size, color=color, weight=MEDIUM)


def math(src, size=40, color=INK):
    """Display or inline mathematics in the paper's faces (Source Serif + newtxmath)."""
    return MathTex(src, tex_template=TEX, font_size=size, color=color)


def tex(src, size=34, color=INK):
    """A line of text that may contain $...$ mathematics, typeset by LaTeX in the paper's faces."""
    return Tex(src, tex_template=TEX, font_size=size, color=color)


def paragraph(txt, width_chars=34, size=26, color=INK, font=SERIF, line_spacing=0.32, weight=NORMAL):
    """Left-aligned wrapped text (Pango), one Text per line, for card bodies without mathematics."""
    lines = [Text(l, font=font, font_size=size, color=color, weight=weight) for l in textwrap.wrap(txt, width_chars)]
    g = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=line_spacing * size / 26 * 0.5)
    return g


# ---------------------------------------------------------------- cards
def paper_card(label, body, credit=None, width_cm=11.0, body_size=30):
    """A light card in the atlas style.

    label  -- e.g. "Theorem II.1 · pointings of an additive method" (set in small caps style)
    body   -- LaTeX text; may contain $...$
    credit -- optional small italic line (prior work, versions, scope)
    Returns a VGroup(card, label, body[, credit]); position it with .move_to / .to_edge.
    """
    lab = Text(label.upper(), font=SANS, weight=MEDIUM, font_size=15, color=PTIDE)
    txt = Tex(r"\parbox{%.1fcm}{\raggedright %s}" % (width_cm, body), tex_template=TEX, font_size=body_size, color=PINK)
    parts = [lab, txt]
    if credit:
        # the credit is set smaller, so its box is widened to keep the same width on screen as the body
        parts.append(Tex(r"\parbox{%.1fcm}{\raggedright\itshape %s}" % (width_cm / 0.72, credit), tex_template=TEX,
                         font_size=body_size * 0.72, color=PINK2))
    col = VGroup(*parts).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    card = RoundedRectangle(corner_radius=0.10, width=col.width + 0.7, height=col.height + 0.6,
                            fill_color=PAPER, fill_opacity=1, stroke_color=PRULE, stroke_width=1)
    shadow = card.copy().set_fill(BLACK, 0.35).set_stroke(width=0).shift(0.06 * RIGHT + 0.08 * DOWN)
    card.move_to(col)
    return VGroup(shadow, card, col)


def schematic_tag(color=INK3):
    return sans("schematic", 14, color)


def readout(label, value, color=INK, size=22):
    """A monospace readout such as 'max |ΔW| = 0.0'. Returns VGroup(label, value); update value with .become."""
    a = mono(label, size, INK3)
    b = mono(value, size, color)
    return VGroup(a, b).arrange(RIGHT, buff=0.18)


def credit_line(txt, size=16):
    return Text(txt, font=SERIF, slant=ITALIC, font_size=size, color=INK3)


# ---------------------------------------------------------------- data graphics
def heatmap(M, cell=0.28, vmax=None, pos=TIDE, neg=CORAL, mid=BG3, stroke=BG):
    """Matrix as coloured squares (row 0 at top). Diverging: positive -> pos, negative -> neg, zero -> mid."""
    M = np.asarray(M, dtype=float)
    vmax = vmax or (np.abs(M).max() or 1.0)
    cells = VGroup()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = np.clip(M[i, j] / vmax, -1, 1)
            c = interpolate_color(ManimColor(mid), ManimColor(pos if v >= 0 else neg), abs(v) ** 0.8)
            sq = Square(cell, fill_color=c, fill_opacity=1, stroke_color=stroke, stroke_width=1)
            sq.move_to([j * cell, -i * cell, 0])
            cells.add(sq)
    return cells.move_to(ORIGIN)


def recolor_heatmap(cells, M, vmax, pos=TIDE, neg=CORAL, mid=BG3):
    """Recolour an existing heatmap in place (for updaters)."""
    M = np.asarray(M, dtype=float)
    for k, sq in enumerate(cells):
        i, j = divmod(k, M.shape[1])
        v = np.clip(M[i, j] / vmax, -1, 1)
        sq.set_fill(interpolate_color(ManimColor(mid), ManimColor(pos if v >= 0 else neg), abs(v) ** 0.8), 1)
    return cells


def plain_axes(x_range, y_range, width=5, height=3, color=RULE):
    """Quiet axes: hairlines, no ticks, no numbers. Label them yourself with sans() if needed."""
    return Axes(x_range=x_range, y_range=y_range, x_length=width, y_length=height, tips=False,
                axis_config=dict(color=color, stroke_width=1.5, include_ticks=False))


def bars(values, width=0.32, gap=0.12, max_height=2.5, vmax=None, color=TIDE):
    """A row of vertical bars sitting on y = 0 (bottom-aligned)."""
    vmax = vmax or max(values)
    g = VGroup()
    for k, v in enumerate(values):
        h = max(1e-3, max_height * v / vmax)
        g.add(Rectangle(width=width, height=h, fill_color=color, fill_opacity=0.9, stroke_width=0)
              .move_to([k * (width + gap), h / 2, 0]))
    return g


# ---------------------------------------------------------------- geometry in weight space (3-D)
def cone(P, H=1.7, R=1.15, color=TIDE, n=14, skin_opacity=0.08):
    """LoRA's double cone with its tip at P: rulings, two rims and a faint skin. Returns (rulings, rims, skin)."""
    angles = np.linspace(0, TAU, n, endpoint=False)
    rulings = VGroup(*[Line(P, P + np.array([R * np.cos(a), R * np.sin(a), s * H]), color=color, stroke_width=1.3,
                            stroke_opacity=0.75) for s in (1, -1) for a in angles])
    rims = VGroup(*[ParametricFunction(lambda a, z=z: P + np.array([R * np.cos(a), R * np.sin(a), z]),
                                       t_range=[0, TAU], color=color, stroke_width=2.4) for z in (H, -H)])
    skin = Surface(lambda u, v: P + np.array([u * R / H * np.cos(v), u * R / H * np.sin(v), u]),
                   u_range=[-H, H], v_range=[0, TAU], resolution=(6, 28), checkerboard_colors=False,
                   fill_color=color, fill_opacity=skin_opacity, stroke_width=0)
    return rulings, rims, skin


def circle_through(center, point, tangent_hint):
    """Parametrised circle about `center` through `point`, in the plane of (point - center) and tangent_hint."""
    U = point - center
    rad = np.linalg.norm(U)
    U = U / rad
    t = tangent_hint - np.dot(tangent_hint, U) * U
    t = t / np.linalg.norm(t)
    return lambda a: center + rad * (np.cos(a) * U + np.sin(a) * t)
