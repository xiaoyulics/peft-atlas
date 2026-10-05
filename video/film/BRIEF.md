# Brief for chapter builders

You are building chapters of a ~17-minute narrated explainer film for the paper *A Categorical Atlas of
Parameter-Efficient Fine-Tuning*. The narration is finished, in `video/audio/beats/<id>.wav`. Your job is the picture:
one Manim scene per chapter, timed to that narration. Paths below are relative to the repository root.

## Read first
1. `video/script/SCRIPT.md`: the script. For each of your beats, follow **On screen** and the narration (lines starting
   with `>`, read verbatim by the voice). Credits, scope tags and the word "schematic" are required where the script
   asks for them. The four nut cards and their scope tags are in the table at the top.
2. `video/film/look.py`, `video/film/story.py`, `video/film/core.py`: the shared library (API below). `video/film/_libtest.py`
   is a small worked example.
3. `video/styles/style_demo.py`, class `Quiet`: the reference 3-D shot (cone, orbit and line around θ₀) and the look.
   Its stills are `video/samples/sea-vs-quiet.png` (middle and bottom panels).
4. `video/README.md`: the decisions.

## Look: "quiet C"
- Deep navy background `BG`. Text: `INK`, `INK2`, `INK3`. Accents: `TIDE` (LoRA and the main accent), `GOLD` (OFT,
  orbits), `CORAL` (controls, warnings, obstructions, defects), `LEAF` and `VIOLET` sparingly.
- Fonts: `serif()` for prose and headlines, `sans()` and `kicker()` for labels, `mono()` for numeric readouts,
  `math()` and `tex()` for mathematics in the paper's faces.
- **Never put background texture behind mathematics.** No grids, contours or water, with three exceptions. The sea
  appears only through `chapter_card()` and the shore helpers. The quantiser grid appears only in beat V.1. Contours
  appear only on the title card (0.3) and the end card (C.2).
- The halo belongs to θ₀ only (`self.theta0()`).
- Statements of theorems and credits to prior work go on **paper cards** (`paper_card()`), held long enough to read:
  at least 3 s, longer for long ones. Credit lines use author names and years as in `paper/bib/refs.bib`.
- In content beats, show `self.chapter_label(numeral, name)` at the upper left and `self.legend(active)` at the upper
  right. Use "image" for II, "fibres" for III and IV, "base" for V. Leave both off chapter cards and shore cuts.
- Calm motion: run_time of 0.6–2 s for most moves; no bouncing; the camera moves slowly, if at all.
- Frame: 14.22 × 8 units. Keep about 0.4 units of margin. Nothing may overlap unintentionally. Text sizes: readable at
  1080p on a laptop (body text ≥ 22, labels ≥ 15).

## Timing
- Subclass `Chapter` and set `BEATS` (in order) and optionally `EXTRA` (picture-only seconds after a beat's
  narration) and `PRE` (picture-only seconds before it). The narration of beat b starts at `self.T(b)`.
- `self.cue(b, phrase)` gives the scene time at which the narration of b reaches `phrase`. `phrase` must be an exact
  substring of the narration with the `<...>` tags removed; `core.script_beats()[b]["text"]` gives the text. Make each
  visual event land on the words that name it.
- Use `self.at(t)` to wait until a time. If an animation runs past a time you wait for, the scene prints
  `[timing] ... late` to stderr. Fix every such warning, because the narration would drift.
- End `construct()` with `self.finish()`. Rendering writes `video/build/timeline/<Scene>.json`, which assembly uses to
  place the audio.
- Chapter cards: `self.chapter_card(bid, numeral, title, subtitle, notch)` plays the whole wordless beat. The stage
  must be clear first.
- Nut k dissolves at the end of its beat, in about 4.5 s of `EXTRA`: `self.clear_stage()`, then
  `s = self.shore(LEVELS[k], dissolved=set(range(1, k)))`, a short hold, `self.dissolve_nut(s, k, LEVELS[k + 1])`, a
  short hold, then `self.clear_stage()`. The P.2 shore uses `LEVELS[0]`. The coda uses `COVERED`.

## Numbers
Every number on screen must be computed. Write the computation as a deterministic script in
`video/sim/<name>.py` that saves to `video/sim/out/<name>.npz` or `.json`, and read that output in the scene.
`video/sim/lora_plus.py` and `video/sim/fibres.py` already exist. Use them, and do not change what they print. Where
the script quotes the paper's number ([Num] notes), compare it with yours. If they disagree, show your computed number
and report the disagreement; never adjust a number to match.

## Rules
- Create only your own chapter file(s) in `video/film/` and new sim scripts. Do not edit `look.py`, `story.py`,
  `core.py`, `SCRIPT.md`, the audio, or another builder's files. If you need a helper, define it in your own file. If
  the library has a bug, work around it and report it.
- Begin each chapter file with
  `import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from story import *`.
- Render with your own media directory. Manim processes that share one collide on LaTeX files.
  - Preview: `PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -ql --disable_caching --media_dir video/build/<Scene> video/film/<file>.py <Scene>`
  - Final: the same command with `-qh --fps 30` in place of `-ql`.
  - Always pass `--disable_caching`. On a re-render, cached partial movies advance the clock by unrounded durations
    and the picture drifts from the narration.
- Check your work visually. Extract frames at the cue times from the preview MP4 with
  `ffmpeg -ss <t> -i <mp4> -frames:v 1 <png>` (do not use the drawtext filter's `fontfile` option; it fails here), then
  look at them with the Read tool. Iterate until every beat looks right: no overlaps, nothing off-frame, labels
  legible, visuals landing on their words.
- The system Python is 3.9. Use `video/.venv/bin/python`, which has numpy, scipy and manim.

## Report when done
- The final MP4 path, its duration, and the `total` in your timeline JSON. They must agree within 0.1 s.
- Each number shown on screen, with the script that computed it, and any disagreement with the paper.
- Any departure from SCRIPT.md, and why.
- Any library problems you worked around.
