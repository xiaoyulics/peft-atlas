"""Smoke test for the shared film library (not part of the film)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from story import *  # noqa: E402,F401


class LibTest(Chapter):
    BEATS = ["II.0", "II.1"]

    def construct(self):
        self.chapter_card("II.0", "II", "The image", "what a method can say", notch=2)
        self.at(self.T("II.1"))
        s = self.shore(LEVELS[1])
        self.wait(0.5)
        self.dissolve_nut(s, 1, LEVELS[2])
        self.wait(0.5)
        self.clear_stage()
        self.chapter_label("II", "the image")
        self.legend("image")
        card = paper_card("Theorem II.1 · pointings of an additive method",
                          r"The union of all re-pointed images is $\theta_0+(C-C)$. For $\mathrm{LoRA}_r$, "
                          r"$C-C$ is the matrices of rank at most $\min(2r,m,n)$.",
                          credit=r"Known in practice: Meng, Wang \& Zhang 2024 (PiSSA $\to$ LoRA conversion).")
        card.to_edge(RIGHT, buff=0.6).shift(0.6 * UP)
        hm = heatmap(np.random.default_rng(0).normal(size=(6, 4))).to_edge(LEFT, buff=1.0)
        ro = readout("max |ΔW| =", "0.0", TIDE).next_to(hm, DOWN, buff=0.5)
        self.play(FadeIn(card, shift=0.2 * UP), FadeIn(hm), FadeIn(ro), run_time=1.0)
        self.at(self.cue("II.1", "They form a cone"))
        self.play(Indicate(ro), run_time=1.0)
        self.wait(2)
