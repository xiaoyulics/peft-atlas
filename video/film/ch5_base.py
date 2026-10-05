"""Chapter V · Base change — how a method travels (beats V.0-V.3).

Numbers on screen come from video/sim/relora.py (V.2) and video/sim/oft_merge.py (V.3).
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *  # noqa: E401,E702,F403

import json
import os

SIM = pathlib.Path(__file__).resolve().parents[1] / "sim" / "out"
ONLY = os.environ.get("CH5_BEATS")            # development only: build just these beats (comma list)

PHI, THETA = 72, -62                          # the Quiet camera
_t = THETA * DEGREES
RS = np.array([-np.sin(_t), np.cos(_t), 0.0])  # screen-right at this camera
ZZ = np.array([0.0, 0.0, 1.0])                 # up
DEP = np.cross(ZZ, RS)                         # horizontal, perpendicular to RS


def plate(mob, pad=0.07):
    """A background-coloured plate behind a label, so grid lines never run through text."""
    r = Rectangle(width=mob.width + 2 * pad, height=mob.height + 2 * pad, fill_color=BG, fill_opacity=1,
                  stroke_width=0).move_to(mob)
    return VGroup(r, mob)


def fmt_pct(v):
    p = 100 * v
    return f"{p:.2f}%" if p < 10 else f"{p:.1f}%"


class BarPanel(VGroup):
    """A row of bars for singular values, with ghost ticks at the starting values and a baseline."""

    def __init__(self, sig0, vmax, color, width=4.0, height=1.55, gap_ratio=0.32):
        super().__init__()
        n = len(sig0)
        step = width / n
        self.bw, self.step, self.h, self.vmax = step * (1 - gap_ratio), step, height, vmax
        self.x0 = -width / 2 + step / 2
        self.base = Line([-width / 2 - 0.05, 0, 0], [width / 2 + 0.05, 0, 0], stroke_color=RULE, stroke_width=1.5)
        self.bars = VGroup(*[Rectangle(width=self.bw, height=0.01, fill_color=color, fill_opacity=0.9, stroke_width=0)
                             for _ in range(n)])
        self.ticks = VGroup(*[Line([self.x0 + k * step - self.bw * 0.8, self.y_of(sig0[k]), 0],
                                   [self.x0 + k * step + self.bw * 0.8, self.y_of(sig0[k]), 0],
                                   stroke_color=INK, stroke_width=1.4, stroke_opacity=0.55) for k in range(n)])
        self.add(self.base, self.bars, self.ticks)
        self.set_values(sig0)

    def y_of(self, v):
        return self.h * v / self.vmax

    def set_values(self, sig):
        o = self.base.get_center()
        for k, b in enumerate(self.bars):
            x, y = o[0] + self.x0 + k * self.step, o[1]
            hgt = max(self.y_of(sig[k]), 0.004)
            b.set_points_as_corners([[x - self.bw / 2, y, 0], [x + self.bw / 2, y, 0], [x + self.bw / 2, y + hgt, 0],
                                     [x - self.bw / 2, y + hgt, 0], [x - self.bw / 2, y, 0]])
        return self


def interp_rows(A, c):
    """Row c of A, linearly interpolated for fractional c."""
    c = float(np.clip(c, 0, len(A) - 1))
    i = int(np.floor(c))
    j = min(i + 1, len(A) - 1)
    return A[i] + (c - i) * (A[j] - A[i])


class BaseChange(Chapter):
    BEATS = ["V.0", "V.1", "V.2", "V.3"]
    EXTRA = {"V.1": 1.0, "V.2": 1.0, "V.3": 6.0}

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

    def want(self, bid):
        return ONLY is None or bid in ONLY.split(",")

    def construct(self):
        if self.want("V.0"):
            self.chapter_card("V.0", "V", "Base change", "how a method travels", notch=5)
        self.at(self.T("V.1"))
        self.lab = self.chapter_label("V", "base change")
        self.leg = self.legend("base")
        self.play(FadeIn(self.lab), FadeIn(self.leg), run_time=0.25)
        if self.want("V.1"):
            self.beat_quantise()
        if self.want("V.2"):
            self.beat_relora()
        if self.want("V.3"):
            self.beat_orthogonal()
        self.finish()

    # -------------------------------------------------------------- helpers for the 3-D shots
    def scr(self, p):
        self.camera.reset_rotation_matrix()           # the matrix is otherwise stale until the next frame
        q = self.camera.project_point(np.asarray(p, dtype=float))
        return np.array([q[0], q[1], 0.0])

    def pin(self, mob, p3, offset=ORIGIN):
        """A flat label pinned to the frame at the projection of p3 (plus a screen offset). Not yet shown."""
        mob.move_to(self.scr(p3) + offset)
        self.hud(mob)
        self.remove(mob)
        return mob

    def show_theta0(self, P):
        dot, halo, _ = self.theta0(P, label=False)
        self.remove(halo)
        return dot, halo

    # -------------------------------------------------------------- V.1 quantisation moves the base point
    def beat_quantise(self):
        b = "V.1"
        cue = lambda p, **k: self.cue(b, p, **k)          # noqa: E731
        self.to_3d(PHI, THETA)
        P0 = 2.75 * RS + 0.05 * ZZ                         # theta_0
        s = 1.25                                           # grid spacing (schematic)
        K = P0 + (-0.44 * RS + 0.28 * ZZ) * s              # its nearest node, kappa theta_0
        u0, z0 = float(K @ RS), float(K @ ZZ)

        # theta_0 and LoRA's cone, as in Exposé II
        dot, halo = self.show_theta0(P0)
        th_lab = self.pin(math(r"\theta_0", 40, INK), P0, 0.44 * RIGHT)
        rulings, rims, skin = cone(P0)
        self.apex = P0.copy()
        rulings.add_updater(lambda m, dt: m.rotate(0.12 * dt, axis=ZZ, about_point=self.apex))
        lora_lab = plate(VGroup(sans("LoRA", 26, TIDE, MEDIUM), sans("cone, apex at θ₀", 20, INK2))
                         .arrange(RIGHT, buff=0.16, aligned_edge=DOWN))
        self.pin(lora_lab, P0 + 1.95 * ZZ, 1.95 * RIGHT + 0.55 * UP)
        self.at(cue("So far theta-zero"))
        self.add_fixed_orientation_mobjects(halo)
        self.play(FadeIn(dot, scale=0.4), FadeIn(halo, scale=0.6), FadeIn(th_lab), run_time=0.8)
        self.play(LaggedStart(*[Create(l) for l in rulings], lag_ratio=0.02), run_time=1.2)
        self.add(rulings)
        self.play(Create(rims), FadeIn(skin), FadeIn(lora_lab), run_time=1.0)

        # the grid of a 4-bit quantiser (schematic)
        self.at(cue("Quantise the model"))
        u_lo, u_hi, z_lo, z_hi = -0.5, 6.55, -2.6, 2.9
        us = [u0 + i * s for i in range(-6, 7) if u_lo + 0.1 < u0 + i * s < u_hi - 0.1]
        zs = [z0 + j * s for j in range(-6, 7) if z_lo + 0.1 < z0 + j * s < z_hi - 0.1]
        lines = VGroup(*[Line(u * RS + z_lo * ZZ, u * RS + z_hi * ZZ) for u in us],
                       *[Line(u_lo * RS + z * ZZ, u_hi * RS + z * ZZ) for z in zs])
        lines.set_stroke(RULE, 1.8, opacity=1)
        nodes = VGroup(*[Dot(self.scr(u * RS + z * ZZ), radius=0.034, color=INK3) for u in us for z in zs])
        nodes.set_opacity(0.8)
        grid_lab = sans("4-bit quantiser grid", 18, INK3)
        grid_lab.move_to(self.scr(u_lo * RS + z_hi * ZZ) + np.array([grid_lab.width / 2 + 0.02, 0.28, 0]))
        note = VGroup(sans("schematic", 15, INK3), sans("·  NF4's levels are not evenly spaced", 15, INK3))
        note.arrange(RIGHT, buff=0.1)
        note.move_to(self.scr(u_hi * RS + z_lo * ZZ) + np.array([-note.width / 2, -0.25, 0]))
        self.hud(nodes, grid_lab, note)
        self.remove(nodes, grid_lab, note)
        self.add(lines)
        self.bring_to_front(rulings, rims, halo, th_lab, lora_lab)
        self.play(Create(lines, lag_ratio=0.05), FadeIn(nodes), FadeIn(grid_lab), FadeIn(note), run_time=1.8)
        self.bring_to_front(rulings, rims, halo, th_lab, lora_lab)

        # theta_0 snaps to its nearest node
        self.at(cue("snap to a grid"))
        kdot = Dot(self.scr(P0), radius=0.075, color=INK)
        self.hud(kdot)
        k_lab = self.pin(math(r"\kappa\theta_0", 38, INK), K, np.array([-0.74, 0.24, 0]))
        self.play(kdot.animate.move_to(self.scr(K)), run_time=0.8, rate_func=smooth)
        self.play(FadeIn(k_lab), run_time=0.5)

        # QLoRA is LoRA pointed at kappa theta_0: the cone moves with it
        self.at(cue("Q-LoRA is ordinary LoRA"))
        q_lab = plate(VGroup(sans("QLoRA", 26, TIDE, MEDIUM), sans("LoRA pointed at κθ₀", 20, INK2))
                      .arrange(RIGHT, buff=0.16, aligned_edge=DOWN))
        q_lab.move_to(lora_lab.get_center() + (self.scr(K) - self.scr(P0)))
        self.hud(q_lab)
        self.remove(q_lab)
        rulings.suspend_updating()
        e = K - P0
        self.play(skin.animate.shift(e), rulings.animate.shift(e), rims.animate.shift(e),
                  FadeOut(lora_lab), FadeIn(q_lab), run_time=1.4, rate_func=smooth)
        self.apex = K.copy()
        rulings.resume_updating()
        self.bring_to_front(kdot, k_lab)

        # Proposition V.2
        self.at(cue("pointed at the snapped") + 0.6)
        card = paper_card("Proposition V.2 · quantisation and splitting",
                          r"QLoRA is LoRA pointed at the quantised weights $\kappa\theta_0$. It misses $\theta_0$ by "
                          r"$e=\kappa\theta_0-\theta_0$, which is generically of full rank, so it can reach $\theta_0$ "
                          r"only if $\operatorname{rank}e\le r$.", width_cm=6.8, body_size=25)
        card.to_edge(LEFT, buff=0.45).shift(0.15 * DOWN)
        self.hud(card)
        self.remove(card)
        self.play(FadeIn(card, shift=0.15 * UP), run_time=0.9)

        # the miss: e = kappa theta_0 - theta_0
        self.at(cue("starts off target"))
        a0, a1 = self.scr(P0), self.scr(K)
        arrow = Arrow(a0, a1, buff=0.0, color=CORAL, stroke_width=4, max_tip_length_to_length_ratio=0.3,
                      tip_length=0.15)
        arrow.put_start_and_end_on(a0 + 0.11 * normalize(a1 - a0), a1 - 0.08 * normalize(a1 - a0))
        self.hud(arrow)
        self.remove(arrow)
        e_lab = plate(math(r"e=\kappa\theta_0-\theta_0", 34, CORAL))
        e_lab.move_to(a0 + np.array([e_lab.width / 2 + 0.62, -0.42, 0]))
        self.hud(e_lab)
        self.remove(e_lab)
        self.play(GrowArrow(arrow), run_time=0.8)
        self.play(FadeIn(e_lab), run_time=0.6)

        self.at(cue("generically full rank"))
        full = plate(sans("generically full rank", 22, INK2))
        full.next_to(e_lab, DOWN, buff=0.1, aligned_edge=LEFT)
        self.hud(full)
        self.remove(full)
        self.play(FadeIn(full, shift=0.08 * UP), run_time=0.7)

        self.at(cue("and it can only get back"))
        only = plate(tex(r"reaches $\theta_0$ only if $\operatorname{rank}e\le r$", 30, INK))
        only.move_to(self.scr(K - 1.7 * ZZ) + np.array([0.25, -0.75, 0]))
        self.hud(only)
        self.remove(only)
        self.play(FadeIn(only, shift=0.08 * UP), run_time=0.8)

        # out
        self.at(self.END(b) - 0.6)
        rulings.clear_updaters()
        self.clear_stage(run_time=0.6, keep=(self.lab, self.leg))

    # -------------------------------------------------------------- V.2 merge and restart
    def beat_relora(self):
        b = "V.2"
        cue = lambda p, **k: self.cue(b, p, **k)          # noqa: E731
        self.at(self.T(b))
        self.to_3d(PHI, THETA)
        data = json.loads((SIM / "relora.json").read_text())
        ranks, r, m, n = data["largest_rank"], data["r"], data["m"], data["n"]
        H2, R2, t = 1.45, 0.98, 0.86
        base = [-0.05 * RS - 0.62 * ZZ]
        for k in range(4):
            sgn = 1 if k % 2 == 0 else -1
            base.append(base[-1] + t * (R2 * RS + sgn * H2 * ZZ))

        def surface_path(P, k):
            """A training path on the cone at P, from its apex to the next base point."""
            sgn = 1 if k % 2 == 0 else -1

            def f(a):
                ang = -0.6 * (1 - a) ** 1.3
                return P + a * t * (R2 * (np.cos(ang) * RS + np.sin(ang) * DEP) + sgn * H2 * ZZ)
            return ParametricFunction(f, t_range=[0, 1], stroke_color=TIDE, stroke_width=3.2)

        # theta_0 and its cone
        dot, halo = self.show_theta0(base[0])
        th_lab = self.pin(math(r"\theta_0", 40, INK), base[0], 0.42 * LEFT + 0.14 * UP)
        rulings, rims, skin = cone(base[0], H=H2, R=R2)
        cones = [VGroup(skin, rulings, rims)]
        self.at(cue("Merging moves"))
        self.add_fixed_orientation_mobjects(halo)
        self.play(FadeIn(dot, scale=0.4), FadeIn(halo, scale=0.6), FadeIn(th_lab), run_time=0.7)
        self.play(LaggedStart(*[Create(l) for l in rulings], lag_ratio=0.02), Create(rims), FadeIn(skin),
                  run_time=1.2)

        # the readout (video/sim/relora.py)
        head = kicker("largest reachable rank", INK3, 16)
        sub = math(r"m=n=%d,\ \ r=%d" % (m, r), 30, INK2)
        rows = VGroup()
        for k, rk in enumerate(ranks):
            K = k + 1
            right = []
            if K * r == rk:
                right.append(math(r"=r" if K == 1 else r"=%dr" % K, 28, INK3))
            if rk == min(m, n):
                right.append(sans("full" if K * r != rk else "· full", 18, INK3))
            row = VGroup(math(r"K=%d" % K, 30, INK2), mono(str(rk), 30, TIDE), VGroup(*right))
            row[1].move_to(row[0].get_center() + 1.45 * RIGHT)
            row[2].arrange(RIGHT, buff=0.12).next_to(row[1], RIGHT, buff=0.32)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        panel = VGroup(head, sub, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        panel.to_edge(LEFT, buff=0.75).shift(0.75 * UP)
        self.hud(panel)
        self.remove(panel)
        self.play(FadeIn(head), FadeIn(sub), run_time=0.7)

        words = []

        def word(txt, p3, off):
            w = self.pin(sans(txt, 19, INK2), p3, off)
            words.append(w)
            return w

        def cycle(k, t_train, t_merge, t_grow, slow):
            """Train an adapter on cone k, merge it (theta_0 hops), grow a fresh cone at the new point."""
            Pk, Pn = base[k], base[k + 1]
            path = surface_path(Pk, k)
            mover = Dot3D(Pk, radius=0.06, color=TIDE, resolution=(8, 8))
            self.at(t_train)
            anims = [Create(path), MoveAlongPath(mover, path)]
            if slow:
                anims.append(FadeIn(word("train", Pk + 0.45 * t * (R2 * RS + H2 * ZZ), 0.85 * RIGHT + 0.12 * DOWN)))
            self.play(*anims, run_time=1.2 if slow else 0.7, rate_func=smooth)
            self.at(t_merge)
            old = cones[-1]
            ghost = Dot(self.scr(Pk), radius=0.04, color=INK3)
            self.hud(ghost)
            anims = [dot.animate.move_to(Pn), halo.animate.move_to(Pn),
                     th_lab.animate.move_to(self.scr(Pn) + 0.42 * LEFT + 0.14 * UP),
                     FadeOut(mover), path.animate.set_stroke(opacity=0.45),
                     old[0].animate.set_fill(opacity=0.0), old[1].animate.set_stroke(opacity=0.10),
                     old[2].animate.set_stroke(opacity=0.22)]
            if slow:
                anims += [FadeIn(word("merge", Pn, 0.9 * RIGHT + 0.1 * DOWN)), FadeOut(words[0])]
            self.play(*anims, run_time=0.9 if slow else 0.45, rate_func=smooth)
            self.at(t_grow)
            ru, ri, sk = cone(Pn, H=H2, R=R2)
            cones.append(VGroup(sk, ru, ri))
            anims = [LaggedStart(*[Create(l) for l in ru], lag_ratio=0.02), Create(ri), FadeIn(sk)]
            if slow:
                anims += [FadeIn(word("restart", Pn + 0.9 * H2 * ZZ, 1.55 * RIGHT)), FadeOut(words[1])]
            self.play(*anims, run_time=1.2 if slow else 0.6)
            self.bring_to_front(halo, th_lab)
            self.play(FadeIn(rows[k + 1], shift=0.06 * RIGHT), run_time=0.4)

        self.at(cue("Train an adapter") - 0.5)
        self.play(FadeIn(rows[0], shift=0.06 * RIGHT), run_time=0.4)
        cycle(0, cue("Train an adapter"), cue("merge it into the weights"), cue("and start a fresh one"), True)
        self.at(cue("Every cycle re-points") - 0.45)
        self.play(FadeOut(words[2]), run_time=0.4)
        t0 = cue("Every cycle re-points")
        cycle(1, t0, t0 + 0.75, t0 + 1.25, False)
        t0 = cue("and the reach grows") - 0.3
        cycle(2, t0, t0 + 0.75, t0 + 1.25, False)
        t0 = cue("after K cycles") + 0.25
        cycle(3, t0, t0 + 0.75, t0 + 1.25, False)

        # the theorem's statement
        self.at(cue("Re-LoRA reaches exactly") + 0.9)
        thm = VGroup(sans("after K cycles, ReLoRA reaches exactly", 18, INK3),
                     math(r"W_0+\{\Delta:\ \operatorname{rank}\Delta\le\min(Kr,m,n)\}", 30, INK))
        thm.arrange(DOWN, aligned_edge=LEFT, buff=0.16).next_to(panel, DOWN, buff=0.5, aligned_edge=LEFT)
        self.hud(thm)
        self.remove(thm)
        self.play(FadeIn(thm, shift=0.1 * UP), run_time=0.8)
        self.at(cue("rank at most K times r"))
        self.play(Indicate(thm[1], color=TIDE, scale_factor=1.06), run_time=1.2)
        self.at(self.END(b) - 0.6)
        self.clear_stage(run_time=0.6, keep=(self.lab, self.leg))

    # -------------------------------------------------------------- V.3 nut 4 dissolves
    def beat_orthogonal(self):
        b = "V.3"
        cue = lambda p, **k: self.cue(b, p, **k)          # noqa: E731
        self.at(self.T(b))
        self.to_2d()
        D = np.load(SIM / "oft_merge.npz")
        sig0, sig_a, run_a, sig_b, run_b = D["sig0"], D["sig_a"], D["run_a"], D["sig_b"], D["run_b"]
        sig_c, dK_c, marks, dK_marks = D["sig_c"], D["dK_c"], D["marks"], D["dK_marks"]
        n_w, draws = int(D["meta"][0]), int(D["meta"][3])
        top_y = 2.0                                        # centre line of the upper band

        # the nut, as a reminder, in the slot the paper cards will use
        nut = nut_card(4, width=4.4).to_edge(RIGHT, buff=0.9).align_to(np.array([0, 3.2, 0]), UP)
        self.play(FadeIn(nut, shift=0.1 * UP), run_time=0.7)

        # neurons are the rows of W; an input-side rotation turns them all together
        O = np.array([-5.65, 1.1, 0])
        angs, lens = [10, 46, 92], [1.85, 1.4, 1.6]
        rot = ValueTracker(0.0)

        def direction(k):
            a = (angs[k] + rot.get_value()) * DEGREES
            return np.array([np.cos(a), np.sin(a), 0])

        def mk_arrow(k):
            return Arrow(O, O + lens[k] * direction(k), buff=0, color=INK2, stroke_width=3, tip_length=0.16,
                         max_tip_length_to_length_ratio=0.12)

        arrows = VGroup(*[always_redraw(lambda k=k: mk_arrow(k)) for k in range(3)])
        wl = VGroup(*[always_redraw(lambda k=k: math(r"w_%d" % (k + 1), 30, INK2).move_to(
            O + (lens[k] + 0.3) * direction(k))) for k in range(3)])

        def arc(k, rad):
            a0 = (angs[k] + rot.get_value()) * DEGREES
            return Arc(radius=rad, start_angle=a0, angle=(angs[k + 1] - angs[k]) * DEGREES, arc_center=O,
                       stroke_color=GOLD, stroke_width=2.6)
        arcs = VGroup(always_redraw(lambda: arc(0, 0.6)), always_redraw(lambda: arc(1, 0.85)))
        cap = sans("neurons: the rows of W", 17, INK3).move_to(O + np.array([0.75, -0.5, 0]))
        self.at(cue("A method that rotates"))
        tmp = [mk_arrow(k) for k in range(3)]
        self.play(LaggedStart(*[GrowArrow(a) for a in tmp], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(w) for w in wl], lag_ratio=0.2), FadeIn(cap), run_time=1.2)
        self.remove(*tmp)
        self.add(arrows, wl)
        f1 = math(r"W\ \leftarrow\ WR", 34, INK)
        f2 = math(r"(WR)(WR)^{\top}=WW^{\top}", 34, INK)
        f3 = math(r"R_2R_1\ \text{is still a rotation}", 30, INK2)
        fblock = VGroup(f1, f2, f3).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        fblock.move_to([0, top_y + 0.15, 0]).align_to(np.array([-3.25, 0, 0]), LEFT)
        self.at(cue("from the input side"))
        self.play(Write(f1), rot.animate.set_value(-16), run_time=1.6, rate_func=smooth)
        self.at(cue("never changes the inner products"))
        tmp = [arc(0, 0.6), arc(1, 0.85)]
        self.play(*[Create(a) for a in tmp], Write(f2), run_time=1.3)
        self.remove(*tmp)
        self.add(arcs)
        self.play(rot.animate.set_value(6), run_time=1.6, rate_func=smooth)

        # Corollary II.6
        self.at(cue("That is what OFT"))
        cor = paper_card("Corollary II.6 · exactly orthogonal methods",
                         r"If every weight a method produces, merges included, is $W_0R$ with $R$ exactly orthogonal "
                         r"on the input side, then in exact arithmetic $WW^{\top}$ never changes: neuron norms, angles "
                         r"between neurons and singular values all stay fixed.",
                         credit=r"OFT was built to keep these angles: Qiu et al.\ 2023, after Liu et al.\ 2018",
                         width_cm=8.9, body_size=22)
        cor.to_edge(RIGHT, buff=0.42).align_to(np.array([0, 3.2, 0]), UP)
        self.play(FadeOut(nut, shift=0.1 * UP), FadeIn(cor, shift=0.1 * UP), run_time=0.9)

        # three bar panels: the singular values of one weight
        vmax = max(sig0.max(), sig_b.max()) * 1.02
        xs = [-4.55, 0.0, 4.55]
        base_y = -2.25
        pa, pb, pc = (BarPanel(sig0, vmax, col) for col in (GOLD, TIDE, GOLD))
        for p, x in zip((pa, pb, pc), xs):
            p.shift(np.array([x, base_y, 0]) - p.base.get_center())
            p.set_values(sig0)
        tag = VGroup(RoundedRectangle(corner_radius=0.06, width=1.28, height=0.34, stroke_color=CORAL,
                                      stroke_width=1.4, fill_opacity=0), sans("PEFT ≥ 0.18", 15, CORAL))
        tag[1].move_to(tag[0])
        ttl = [VGroup(sans("exact Cayley OFT", 19, INK, MEDIUM), math(r"R=(I-Q)^{-1}(I+Q)", 26, INK2)),
               VGroup(sans("LoRA merges", 19, INK, MEDIUM),
                      math(r"W\leftarrow W+BA,\ \ \operatorname{rank}BA=%d" % int(D["meta"][4]), 26, INK2)),
               VGroup(VGroup(sans("HF PEFT default OFT", 19, INK, MEDIUM), tag).arrange(RIGHT, buff=0.18),
                      sans("Cayley–Neumann", 17, INK2))]
        for t_, x in zip(ttl, xs):
            t_.arrange(DOWN, buff=0.13).move_to([x, base_y + 2.12, 0])
        ttl[2][1].align_to(ttl[2][0], LEFT)
        ttl[2][0].align_to(ttl[2][1], LEFT)

        self.at(cue("So it never changes a single"))
        self.play(FadeIn(pa, lag_ratio=0.02), FadeIn(ttl[0]), FadeIn(pb, lag_ratio=0.02), FadeIn(ttl[1]), run_time=1.2)
        sv = sans(f"singular values of one weight W ({n_w} × {n_w}), largest first", 17, INK3)
        sv.move_to([-2.27, base_y - 0.36, 0])
        self.at(cue("singular value") - 0.3)
        self.play(FadeIn(sv), run_time=0.6)

        # readouts (computed in video/sim/oft_merge.py)
        def ro(label, color, size=24):
            g = VGroup(math(label, size, INK3), mono("0", 22, color)).arrange(RIGHT, buff=0.18)
            g.value_color = color
            return g

        ra_m, rb_m = ro(r"\text{merges}", INK), ro(r"\text{merges}", INK)
        ra_d, rb_d = ro(r"\max\,\lvert\Delta\sigma\rvert", GOLD), ro(r"\max\,\lvert\Delta\sigma\rvert", TIDE)
        for g, x in ((VGroup(ra_m, ra_d), xs[0]), (VGroup(rb_m, rb_d), xs[1])):
            g.arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([x, base_y - 0.62, 0])
        cache = {}

        def set_num(g, txt, key):
            if cache.get(key) != txt:
                cache[key] = txt
                g[1].become(mono(txt, 22, g.value_color).next_to(g[0], RIGHT, buff=0.18))

        cyc = ValueTracker(0.0)

        def upd_ab(_):
            c = cyc.get_value()
            pa.set_values(interp_rows(sig_a, c))
            pb.set_values(interp_rows(sig_b, c))
            ci = int(round(c))
            set_num(ra_m, f"{ci}", "am")
            set_num(rb_m, f"{ci}", "bm")
            set_num(ra_d, "0" if ci == 0 else f"{run_a[ci]:.1e}", "ad")
            set_num(rb_d, "0" if ci == 0 else f"{run_b[ci]:.3f}", "bd")

        self.at(cue("and in exact arithmetic"))
        self.play(FadeOut(sv), FadeIn(VGroup(ra_m, ra_d, rb_m, rb_d)), run_time=0.5)
        holder = Mobject()
        holder.add_updater(upd_ab)
        self.add(holder)
        self.play(cyc.animate.set_value(len(sig_a) - 1), run_time=cue("a rotation times a rotation") - 0.15 - self.now(),
                  rate_func=linear)
        holder.clear_updaters()
        self.remove(holder)

        self.at(cue("a rotation times a rotation"))
        self.play(Write(f3), rot.animate.set_value(-14), run_time=1.6, rate_func=smooth)

        # the caution: HF PEFT's default Cayley-Neumann OFT
        self.at(cue("One caution"))
        for mob in (*arrows, *wl, *arcs):
            mob.clear_updaters()
        self.play(FadeOut(VGroup(arrows, wl, arcs, cap, fblock)), FadeIn(pc, lag_ratio=0.02), FadeIn(ttl[2]),
                  run_time=1.0)
        hd = kicker("the default OFT in Hugging Face's PEFT", CORAL, 16)
        g1 = math(r"R=(I+Q)(I+Q+Q^2+Q^3)", 36, INK)
        g2 = math(r"R^{\top}R=(I-Q^4)^2\ \ne\ I", 36, INK)
        gblock = VGroup(hd, g1, g2).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        gblock.to_edge(LEFT, buff=0.55).align_to(np.array([0, 3.12, 0]), UP)
        self.at(cue("The default OFT"))
        self.play(FadeIn(hd, shift=0.1 * UP), Indicate(tag, color=CORAL, scale_factor=1.12), run_time=1.0)
        self.at(cue("uses an approximation"))
        self.play(Write(g1), run_time=1.2)
        self.at(cue("isn't exactly orthogonal"))
        self.play(Write(g2), run_time=1.0)

        # Theorem V.3, and the bars sink
        self.at(cue("Each merge shrinks"))
        thm = paper_card("Theorem V.3 · rebasing closure",
                         r"Exactly orthogonal input-side methods keep $WW^{\top}$ and the spectrum through any number "
                         r"of merges. HF PEFT's default Cayley--Neumann OFT is not orthogonal: with small generators "
                         r"each merge is a contraction, so repeated merging can only shrink neuron norms and singular "
                         r"values.", width_cm=8.9, body_size=22)
        thm.move_to(cor.get_center()).align_to(cor, UP)
        rc_m = ro(r"\text{merges}", INK)
        rc_d = ro(r"\lVert\Delta K_{\mathrm{out}}\rVert_F/\lVert K_{\mathrm{out}}\rVert_F", CORAL)
        VGroup(rc_m, rc_d).arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([xs[2], base_y - 0.62, 0])
        self.play(FadeOut(cor), FadeIn(thm), FadeIn(VGroup(rc_m, rc_d)), run_time=0.9)
        cyc_c = ValueTracker(0.0)

        def upd_c(_):
            c = cyc_c.get_value()
            pc.set_values(interp_rows(sig_c, c))
            ci = int(round(c))
            set_num(rc_m, f"{ci}", "cm")
            set_num(rc_d, "0" if ci == 0 else fmt_pct(dK_c[ci]), "cd")

        holder = Mobject()
        holder.add_updater(upd_c)
        self.add(holder)
        self.play(cyc_c.animate.set_value(len(sig_c) - 1), run_time=cue("two hundred merges", end=True) - self.now(),
                  rate_func=linear)
        holder.clear_updaters()
        self.remove(holder)

        # the paper's checkpoints, from our run
        lab1 = math(r"\text{merges}", 24, INK3)
        lab2 = math(r"\lVert\Delta K_{\mathrm{out}}\rVert_F/\lVert K_{\mathrm{out}}\rVert_F", 24, INK3)
        cols = [VGroup(mono(f"{int(k)}", 18, INK2), mono(fmt_pct(v), 18, CORAL)) for k, v in zip(marks, dK_marks)]
        tab = VGroup(VGroup(lab1, lab2).arrange(DOWN, buff=0.2, aligned_edge=RIGHT))
        for c_ in cols:
            c_[0].move_to([0, lab1.get_center()[1], 0])
            c_[1].move_to([0, lab2.get_center()[1], 0])
            tab.add(c_)
        x = tab[0].get_right()[0] + 0.38
        for c_ in cols:
            c_.shift((x + c_.width / 2 - c_.get_center()[0]) * RIGHT)
            x += c_.width + 0.28
        block, q_std = int(D["meta"][1]), float(D["meta"][2])
        foot = sans(f"mean of {draws} draws · block size {block} · skew entries ≈ {q_std:g}", 15, INK3)
        foot.move_to([0, base_y - 1.2, 0]).to_edge(RIGHT, buff=0.5)
        tab.next_to(gblock, DOWN, buff=0.36, aligned_edge=LEFT)
        self.play(FadeIn(tab, shift=0.1 * UP), FadeIn(foot), run_time=0.8)
        self.at(cue("by about a fifth"))
        self.play(Indicate(rc_d[1], color=CORAL, scale_factor=1.15), Indicate(cols[-1][1], color=CORAL,
                                                                             scale_factor=1.15), run_time=1.0)

        # cut to the shore: nut 4 dissolves, and all four nuts are gone
        self.at(self.T(b) + self.plan[b]["audio"] + 0.2)
        self.clear_stage(run_time=0.6)
        s = self.shore(LEVELS[4], dissolved={1, 2, 3})
        self.wait(0.5)
        self.dissolve_nut(s, 4, LEVELS[5])
        self.at(self.END(b) - 0.6)
        # the shore's sea has a become() updater, so FadeOut cannot fade it (see chapter_card): use a tracker
        fade, sea = ValueTracker(1.0), s["sea"]
        sea.clear_updaters()
        sea.add_updater(lambda m, dt: m.become(wave_polygon(s["level"].get_value(), 1.2 * self.renderer.time, 0.22))
                        .set_fill(opacity=0.22 * fade.get_value()).set_stroke(opacity=0.75 * fade.get_value()))
        self.play(*[FadeOut(m) for m in self.mobjects if m is not sea], fade.animate.set_value(0), run_time=0.6)
        sea.clear_updaters()
        self.remove(sea)
