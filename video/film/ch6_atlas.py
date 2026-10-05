"""Chapters VIII-XI · The atlas — every method, seven coordinates (beats A.0-A.2).

Every count on screen comes from video/sim/atlas.py, which reads site/data/atlas-data.js.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *  # noqa: E401,E702,F403

import json

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim" / "out"


def mix(a, b, t):
    return interpolate_color(ManimColor(a), ManimColor(b), t)


# a muted categorical palette made from the look's colours
SEA, AMBER, ROSE, SLATE, DUSK = mix(TIDE, LEAF, 0.5), mix(GOLD, CORAL, 0.5), mix(VIOLET, CORAL, 0.45), INK2, INK3
HUE = {
    "kind": dict(R=TIDE, E=VIOLET, E0=ROSE, B=GOLD, P=LEAF, F_T=SLATE),
    "shape": {"Lin": SLATE, "Cone": TIDE, "Cone*": SEA, "Orb": GOLD, "Sat": LEAF, "Meet": VIOLET, "Fun": DUSK},
    "base_point": dict(regular=LEAF, apex=TIDE, neutral=SLATE, defect=CORAL, unpointed=VIOLET),
    "gauge": dict(none=SLATE, scalar=LEAF, torus=GOLD, GL=TIDE, GLxTorus=SEA, nonlinear=VIOLET),
    "covariance": dict(GLxGL=TIDE, OxO=GOLD, GLxCO=AMBER, MonxGL=LEAF, MonxMon=SEA, Pi=VIOLET, frame=SLATE),
    "merge": {"M1": TIDE, "M1->": LEAF, "Mq": GOLD, "Minf": CORAL},
    "rebasing": dict(idempotent=SLATE, span=TIDE, group=GOLD, schedule=VIOLET),
}
MUTE = 0.3                                     # how far each hue is pulled towards the background
NULL = RULE                                    # unset or not applicable
SHORT = {                                      # cell labels, shortened from the value names in theory/axes.json
    "kind": dict(R="reparam.", E="extension", E0="unpointed", B="lens", P="post hoc", F_T="family"),
    "shape": {"Lin": "linear", "Cone": "cone", "Cone*": "split cone", "Orb": "orbit", "Sat": "saturation",
              "Meet": "meet", "Fun": "function"},
    "base_point": dict(regular="regular", apex="apex", neutral="neutral", defect="defect", unpointed="unpointed"),
    "gauge": dict(none="trivial", scalar="scalar", torus="torus", GL="GL(r)", GLxTorus="GL × torus",
                  nonlinear="non-linear"),
    "covariance": dict(GLxGL="GL × GL", OxO="CO × CO", GLxCO="GL × CO(n)", MonxGL="Mon × GL", MonxMon="Mon × Mon",
                       Pi="imposed", frame="frame"),
    "merge": {"M1": "fuses", "M1->": "neighbour", "Mq": "leaves type", "Minf": "obstructed"},
    "rebasing": dict(idempotent="idempotent", span="span", group="group", schedule="schedule"),
}
HEAD = [("kind",), ("shape of", "the image"), ("base point",), ("gauge",), ("covariance", "group"),
        ("merge", "type"), ("rebasing", "closure")]


def strut(txt, size, color, weight=NORMAL):
    """Sans text led by an invisible 'Hg', so that lines of different letters share one baseline."""
    t = Text("Hg" + txt, font=SANS, font_size=size, color=color, weight=weight)
    t[0].set_opacity(0)
    t[1].set_opacity(0)
    return t


def put(t, x, y):
    """Centre the visible part of a strut text on x, and the whole line on y."""
    return t.shift(np.array([x - t[2:].get_center()[0], y - t.get_center()[1], 0]))


def cell_color(axis, key):
    return NULL if key is None else mix(HUE[axis][key], BG, MUTE)


class Atlas(Chapter):
    BEATS = ["A.0", "A.1", "A.2"]
    EXTRA = {"A.1": 1.0, "A.2": 1.2}

    def chapter_card(self, bid, numeral, title, subtitle, notch):
        """story.Chapter.chapter_card with the same layout and timing, but with working fades.

        The library animates the sea polygon itself (poly.animate.set_opacity, FadeOut(poly)). Its become() updater
        is copied onto the animation's start and target copies, which Manim keeps updating, so the sea pops in on
        the first frame and out on the last. Here a tracker drives the sea's opacity instead.
        """
        self.at(self.T(bid))
        dur = self.plan[bid]["dur"]
        level, alpha = ValueTracker(LEVELS[max(notch - 1, 0)]), ValueTracker(0.0)

        def draw(m, dt):                                # dt: waits stay live frames, so the waves keep moving
            a = alpha.get_value()
            m.become(wave_polygon(level.get_value(), 1.2 * self.renderer.time))
            m.set_fill(opacity=0.13 * a).set_stroke(opacity=0.75 * a)

        poly = wave_polygon(level.get_value(), 0)
        poly.add_updater(draw)
        self.add_fixed_in_frame_mobjects(poly)
        card = VGroup(kicker(f"Exposé {numeral}" if numeral else "", TIDE, 20),
                      serif(title, 66, INK, SEMIBOLD), serif(subtitle, 34, INK2, italic=True))
        card.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(2.4 * LEFT + 1.0 * UP)
        self.hud(card)
        self.remove(card)
        self.play(FadeIn(card), alpha.animate.set_value(1), run_time=0.8)
        self.play(level.animate.set_value(LEVELS[notch]), run_time=2.2, rate_func=smooth)
        self.wait(max(dur - 0.8 - 2.2 - 0.7, 0.05))
        self.play(FadeOut(card), alpha.animate.set_value(0), run_time=0.7)
        poly.clear_updaters()
        self.remove(poly)

    def construct(self):
        self.chapter_card("A.0", "VIII–XI", "The atlas", "every method, seven coordinates", notch=6)
        self.beat_table()
        self.beat_limits()
        self.finish()

    # -------------------------------------------------------------- A.1 seven coordinates
    def beat_table(self):
        b = "A.1"
        cue = lambda p, **k: self.cue(b, p, **k)          # noqa: E731
        D = json.loads((SIM / "atlas.json").read_text())
        rows, axes, values, counts = D["rows"], D["axes"], D["values"], D["counts"]
        N = len(rows)
        self.at(self.T(b))
        self.lab = self.chapter_label("VIII–XI", "the atlas")
        self.leg = self.legend(None)
        self.play(FadeIn(self.lab), FadeIn(self.leg), run_time=0.4)

        # geometry of the mosaic: one thin row of seven cells per method
        y_top, y_bot, cw, mx0 = 2.95, -3.55, 0.22, -6.3
        rh = (y_top - y_bot) / N
        mx1 = mx0 + 7 * cw

        def ry(i):                                        # centre of row i
            return y_top - (i + 0.5) * rh

        def key(i, k):
            c = rows[i]["codes"][k]
            return None if c < 0 else values[axes[k]][c]

        # the names return
        rng = np.random.default_rng(7)
        names = [sans(rw["name"], 15, INK3) for rw in rows]
        lines, cur, x = [], [], 0.0
        for i in rng.permutation(N):                    # flow the names into lines, in shuffled order
            if cur and x + names[i].width > 13.1:
                lines.append(cur)
                cur, x = [], 0.0
            cur.append((i, x))
            x += names[i].width + 0.17
        lines.append(cur)
        dy = min(0.22, 6.45 / (len(lines) - 1))
        for li, line in enumerate(lines):
            last, lx = line[-1]
            off = -(lx + names[last].width) / 2
            for i, lx in line:
                names[i].move_to([off + lx + names[i].width / 2, 3.08 - li * dy, 0])
        for t in names:
            t.set_opacity(0.8)
        self.at(cue("Do this for every method"))
        self.play(LaggedStart(*[FadeIn(VGroup(*[names[i] for i, _ in line])) for line in lines], lag_ratio=0.06),
                  run_time=1.6)

        # they settle into a table
        ex_ids = sorted(D["examples"].values())             # example rows, top to bottom
        skeleton = [Rectangle(width=7 * cw, height=rh, fill_color=NULL, fill_opacity=1, stroke_width=0)
                    .move_to([(mx0 + mx1) / 2, ry(i), 0]) for i in range(N)]
        tx_name, tx0 = -2.62, -1.62                         # the enlarged rows
        chgt, gap = 0.44, 0.045
        need = []                                           # each column as wide as its widest label, plus air
        for k in range(7):
            ws = [strut(SHORT[axes[k]][key(i, k)] if key(i, k) is not None else "n/a", 15, INK)[2:].width
                  for i in ex_ids]
            ws += [strut(w, 16, INK)[2:].width for w in HEAD[k]]
            need.append(max(ws) + 0.18)
        extra = (6.7 - tx0 - 6 * gap - sum(need)) / 7
        assert extra > 0, extra
        col_w, col_x, x = [n + extra for n in need], [], tx0
        for w in col_w:
            col_x.append(x + w / 2)
            x += w + gap
        row_y = [2.2 - 0.55 * j for j in range(len(ex_ids))]
        big = {}
        for j, i in enumerate(ex_ids):
            big[i] = sans(rows[i]["name"], 21, INK, MEDIUM).move_to([0, row_y[j], 0]).align_to([tx_name, 0, 0], LEFT)
        anims = []
        for i in range(N):
            if i in big:
                anims.append(AnimationGroup(FadeIn(skeleton[i]), Transform(names[i], big[i])))
            else:
                anims.append(AnimationGroup(FadeIn(skeleton[i]),
                                            names[i].animate.scale(0.06).move_to(skeleton[i]).set_opacity(0)))
        self.at(cue("you get an atlas"))
        self.play(LaggedStart(*anims, lag_ratio=0.006), run_time=1.7)
        self.remove(*[names[i] for i in range(N) if i not in big])

        # the table frame: column numbers, empty cells, leaders from the mosaic
        nums = VGroup(*[sans(str(k + 1), 15, INK3).move_to([mx0 + (k + 0.5) * cw, y_top + 0.17, 0]) for k in range(7)])
        cells = {}
        frame = VGroup()
        for j, i in enumerate(ex_ids):
            for k in range(7):
                c = RoundedRectangle(corner_radius=0.05, width=col_w[k], height=chgt, fill_color=BG2, fill_opacity=1,
                                     stroke_color=RULE, stroke_width=1).move_to([col_x[k], row_y[j], 0])
                cells[i, k] = c
                frame.add(c)
        leaders = VGroup(*[VGroup(Dot([mx1 + 0.03, ry(i), 0], radius=0.025, color=INK2),
                                  Line([mx1 + 0.05, ry(i), 0], [tx_name - 0.1, row_y[j], 0], stroke_color=INK3,
                                       stroke_width=1, stroke_opacity=0.7))
                           for j, i in enumerate(ex_ids)])
        heads = VGroup()
        for k in range(7):
            xc = col_x[k]
            y = row_y[0] + chgt / 2 + 0.17               # centre line of the lowest header line
            words = VGroup()
            for w in reversed(HEAD[k]):
                words.add(put(strut(w, 16, INK2), xc, y))
                y += 0.24
            num = sans(str(k + 1), 15, INK3).move_to([xc, y + 0.02, 0])
            heads.add(VGroup(num, words))
        self.at(cue("The paper reads seven coordinates"))
        self.play(FadeIn(nums), FadeIn(frame, lag_ratio=0.01), Create(leaders), run_time=1.0)

        # one coordinate at a time: a column of the mosaic, a column of the table
        col_runs = []
        for k in range(7):
            runs, start = VGroup(), 0
            for i in range(1, N + 1):
                if i == N or key(i, k) != key(start, k):
                    if key(start, k) is not None:
                        h = (i - start) * rh
                        runs.add(Rectangle(width=cw, height=h, fill_color=cell_color(axes[k], key(start, k)),
                                           fill_opacity=1, stroke_width=0)
                                 .move_to([mx0 + (k + 0.5) * cw, y_top - start * rh - h / 2, 0]))
                    start = i
            col_runs.append(runs)
        phrases = ["where it acts", "the shape of its image", "where it starts", "its gauge",
                   "which changes of basis it respects", "whether it can be merged",
                   "and what merge-and-restart can reach"]
        for k, ph in enumerate(phrases):
            self.at(cue(ph))
            fills, labels = [], []
            for i in ex_ids:
                kk = key(i, k)
                fills.append(cells[i, k].animate.set_fill(cell_color(axes[k], kk), 1).set_stroke(opacity=0))
                txt = SHORT[axes[k]][kk] if kk is not None else "n/a"
                c = cells[i, k].get_center()
                labels.append(put(strut(txt, 15, BG if kk is not None else INK3), c[0], c[1]))
            self.play(LaggedStart(*[FadeIn(r) for r in col_runs[k]], lag_ratio=0.5 / max(len(col_runs[k]), 1)),
                      FadeIn(heads[k], shift=0.06 * DOWN), *fills,
                      LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.15), nums[k].animate.set_color(INK),
                      run_time=0.9)
        key_note = sans("cells coloured by value  ·  grey: unset or not applicable", 15, INK3)
        key_note.move_to([0, row_y[-1] - chgt / 2 - 0.28, 0]).align_to([tx_name, 0, 0], LEFT)
        self.play(FadeIn(key_note), run_time=0.6)

        # counts, computed from the data
        cx = [-1.85, 0.3, 2.75, 5.35]
        cy = -1.75

        def counter(k, value, color, caption):
            tr = ValueTracker(0)
            num = mono("0", 44, color).move_to([cx[k], cy, 0])
            num.add_updater(lambda m: m.become(mono(str(int(round(tr.get_value()))), 44, color).move_to([cx[k], cy, 0])))
            cap = put(strut(caption[0], 18, INK2), cx[k], cy - 0.55)
            return tr, num, cap

        specs = [(counts["methods"], INK, ["methods"]), (counts["papers"], INK, ["papers"]),
                 (counts["arrows"], TIDE, ["simulation arrows"]), (counts["obstructions"], CORAL, ["obstructions"])]
        made = [counter(k, *sp) for k, sp in enumerate(specs)]
        for k, ph in ((0, "Three hundred and eighty-two methods"), (1, "from a hundred and thirty-eight papers")):
            tr, num, cap = made[k]
            self.at(cue(ph))
            self.add(num)
            self.play(tr.animate.set_value(specs[k][0]), FadeIn(cap), run_time=1.0, rate_func=smooth)
            num.clear_updaters()

        # the simulation arrows as a network beside the mosaic
        xa = mx1 + 0.04
        bulge = lambda span: 0.11 * np.sqrt(max(span, 0.6))  # noqa: E731
        arcs = []
        for a_, b_, _ in sorted(D["arrows"], key=lambda e: abs(e[0] - e[1])):
            ya, yb = ry(a_), ry(b_)
            if a_ == b_:
                ya, yb = ya + 0.03, ya - 0.03
            w = bulge(abs(a_ - b_)) * 4 / 3
            arcs.append(CubicBezier([xa, ya, 0], [xa + w, ya, 0], [xa + w, yb, 0], [xa, yb, 0])
                        .set_stroke(TIDE, 1.1, opacity=0.5))
        tr, num, cap = made[2]
        self.at(cue("Four hundred and three arrows"))
        self.play(FadeOut(leaders), run_time=0.4)
        self.add(num)
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.012), tr.animate.set_value(len(arcs)),
                  FadeIn(cap), run_time=2.6)
        num.clear_updaters()
        assert len(arcs) == specs[2][0]
        self.at(cue("each with an explicit map"))
        more = put(strut("each with an explicit map", 16, INK3), cx[2], cy - 0.83)
        self.play(FadeIn(more), run_time=0.6)

        # the obstructions: a short coral mark where an arrow cannot be drawn
        marks = []
        for a_, b_, _ in D["obstructions"]:
            x = xa + bulge(abs(a_ - b_))
            y = (ry(a_) + ry(b_)) / 2
            marks.append(Line([x - 0.04, y, 0], [x + 0.04, y, 0], stroke_color=CORAL, stroke_width=2.2))
        tr, num, cap = made[3]
        self.at(cue("and four hundred and eighty-five obstructions"))
        self.add(num)
        self.play(LaggedStart(*[FadeIn(m) for m in marks], lag_ratio=0.01), tr.animate.set_value(len(marks)),
                  FadeIn(cap), run_time=2.2)
        num.clear_updaters()
        assert len(marks) == specs[3][0]
        self.at(cue("saying which can't"))
        more2 = put(strut("each with a named test", 16, INK3), cx[3], cy - 0.83)
        self.play(FadeIn(more2), run_time=0.6)

    # -------------------------------------------------------------- A.2 predictions and limits
    def beat_limits(self):
        b = "A.2"
        cue = lambda p, **k: self.cue(b, p, **k)          # noqa: E731
        self.at(self.T(b))
        self.clear_stage(run_time=0.7, keep=(self.lab, self.leg))

        k1 = kicker("predictions · Exposé X", TIDE, 16).to_edge(LEFT, buff=0.6).set_y(3.12)
        self.at(cue("so the atlas makes predictions"))
        self.play(FadeIn(k1, shift=0.08 * UP), run_time=0.7)

        x1 = paper_card("Prediction X.1 · a fair LoRA+ baseline",
                        r"A fair LoRA+ baseline multiplies $\alpha$ by the ratio $\lambda$. Under Adam, with "
                        r"$\varepsilon$, decay and per-block clipping thresholds rescaled and one shared schedule, the "
                        r"two runs give identical weights.",
                        credit=r"after Schulman \& Thinking Machines Lab 2025", width_cm=8.8, body_size=22)
        x8 = paper_card("Prediction X.8 · drift of the charge",
                        r"LoRA's charge $\Phi=B^{\top}B/\eta_B-AA^{\top}/\eta_A$ is constant under gradient flow and, "
                        r"when a step size $\eta$ multiplies fixed block rates, moves by $O(\eta^2)$ per SGD step: a "
                        r"cheap diagnostic of what the optimizer adds.",
                        credit=r"the conservation law: after Zhao et al.\ 2022", width_cm=8.8, body_size=22)
        top = 2.88
        x1.to_edge(LEFT, buff=0.55).align_to([0, top, 0], UP)
        x8.to_edge(RIGHT, buff=0.55).align_to([0, top, 0], UP)
        self.at(cue("Under Adam, a fair"))
        self.play(FadeIn(x1, shift=0.12 * UP), run_time=0.9)
        self.at(cue("LoRA's charge should drift"))
        self.play(FadeIn(x8, shift=0.12 * UP), run_time=0.9)

        k2 = kicker("limits · Conclusion and Exposé XI", CORAL, 16).to_edge(LEFT, buff=0.6)
        k2.set_y(min(x1.get_bottom()[1], x8.get_bottom()[1]) - 0.32)
        l1 = paper_card("Conclusion · learnability",
                        r"Expressivity is not learnability. An image says where a method can go, not whether "
                        r"training gets there. Only Expos\'e III is dynamical, and its conservation laws hold under "
                        r"gradient flow, not under Adam or momentum.", width_cm=5.5, body_size=22)
        l2 = paper_card("Conclusion · exact arithmetic",
                        r"Most results are about images: what can be reached at all, as sets and in exact "
                        r"arithmetic. The classification of initialisations is set-level, and low-precision merges "
                        r"add rounding drift.", width_cm=5.5, body_size=22)
        l3 = paper_card("Exposé XI · analogies",
                        r"Noether, Erlangen, gauge and moduli are analogies beyond what is proved, and Expos\'e XI "
                        r"says where each one stops: no Lagrangian behind the charge, no connection or curvature "
                        r"behind the gauge.", width_cm=5.5, body_size=22)
        lim = VGroup(l1, l2, l3).arrange(RIGHT, buff=0.3, aligned_edge=UP)
        lim.move_to([0, 0, 0]).align_to([0, k2.get_bottom()[1] - 0.2, 0], UP)
        self.at(cue("And it marks its own limits"))
        self.play(FadeIn(k2, shift=0.08 * UP), run_time=0.7)
        for card, ph in ((l1, "Reaching a weight"), (l2, "Most results are about images"),
                         (l3, "And words like gauge")):
            self.at(cue(ph))
            self.play(FadeIn(card, shift=0.12 * UP), run_time=0.9)

        self.at(self.END(b) - 0.7)
        self.clear_stage(run_time=0.7)
