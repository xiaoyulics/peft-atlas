# Explainer video

A narrated, animated explainer of about 15 minutes for *A Categorical Atlas of Parameter-Efficient Fine-Tuning*.

## Decisions (2026-10-05)
- **Language:** English. **Length:** about 17 minutes, at Sulafat's natural pace (no trimming).
- **Voice:** Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`), prebuilt voice **Sulafat**. The style direction is in
  `tools/tts.py` (`DEFAULT_STYLE`).
- **Look:** quiet C. Deep navy (`#0A121C`), ink `#E3E9EE`, tide `#4CC3CF`, gold `#E2B04C`, coral `#EC7A66`; Source Serif 4
  and IBM Plex, taken from TeX Live; a halo on θ₀ only. No background texture during the mathematics. The sea appears
  only on chapter cards (one notch per chapter) and when a nut dissolves. The quantiser grid appears only in the QLoRA
  scene. Contours appear only on the title and end cards. Statements and credits go on paper cards in the light atlas
  style.
- **Emphasis:** 3-D geometry, real numerical simulations (every number on screen is computed), narrative devices
  (the four nuts and the rising sea).
- **Review gates:** script → voice table read → one finished chapter → full film.

## Layout
- `script/SCRIPT.md`: the script. Lines starting with `>` are narration, read verbatim. Each beat lists what its
  claims were checked against.
- `tools/tts.py`: one Gemini TTS call. The key is read from `$GEMINI_API_KEY` or `~/.config/gemini/api_key`, and never
  printed.
- `tools/narrate.py`: synthesises every beat to `audio/beats/<id>.wav`, records durations and pauses (cue times) in
  `<id>.json`, and builds `audio/table-read.m4a`. Unchanged beats are cached by hash.
- `tools/mux.sh`: lays narration under a rendered clip, with loudness at -16 LUFS.
- `sim/`: the computations behind the `[SIM]` beats (`lora_plus.py`, `fibres.py`, …).
- `styles/style_demo.py`: the style samples (classes Chalk, Atlas, Sea, Quiet). `samples/` holds their renders.
- `.venv/`: Python 3.12 with Manim 0.21 and SciPy, made with `uv`. The system Python is 3.9.

## The film
- `film/`: one Manim scene per chapter, on a shared library (`look.py` for the look, `story.py` for chapter cards,
  the sea, the shore, the nuts and the Chapter base class, `core.py` for beat timing and phrase cues). `BRIEF.md` is
  the builders' brief.
- Chapters, with their scenes: `ch0_opening.py` Opening (0.1–P.2), `ch1_slice.py` Slice (I), `ch2_image.py` TheImage (II),
  `ch3_fibres.py` Fibres (III), `ch4_lens.py` Lens (IV), `ch5_base.py` BaseChange (V), `ch6_atlas.py` Atlas
  (VIII–XI), `ch7_coda.py` Coda.
- Every on-screen number comes from a script in `sim/`, with its output in `sim/out/`.
- `tools/assemble.py` joins the chapters, lays each beat's narration at its timeline position and normalises loudness
  to -16 LUFS. It adds a quiet pad under cards, shore cuts and the coda; `--no-music` leaves it out. It also writes
  subtitles, chapter marks and a thumbnail to `out/`.
- `tools/review.py <Scene>` makes a contact sheet of a rendered chapter.

## Commands
```bash
python3 video/tools/narrate.py
PATH=/Library/TeX/texbin:$PATH video/.venv/bin/manim -qh --fps 30 --disable_caching --media_dir video/build/Slice video/film/ch1_slice.py Slice
python3 video/tools/assemble.py
python3 video/tools/assemble.py --no-music
video/.venv/bin/python video/sim/lora_plus.py
```
Do not render several scenes that share LaTeX at the same time with separate `manim` processes. They share
`build/Tex` and can collide on a temporary SVG.

## Costs and limits
A key without billing is on the free tier: 10 requests a day and 3 a minute (seen 2026-10-05). That is too few for
the 25 narrated beats, so the key's project needs billing enabled.
Gemini 3.8 Flash TTS bills 25 audio tokens per second at $9 per million until 31 December 2026, and $18 per million
after that. A full 15-minute take costs about $0.20.
