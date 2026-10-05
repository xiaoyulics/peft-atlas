#!/usr/bin/env python3
"""Contact sheet of one rendered chapter: frames near the start, middle and end of every beat.

  python3 video/tools/review.py TheImage            # final render if present, else the 480p preview
  python3 video/tools/review.py TheImage --cols 3

Writes video/out/review/<Scene>.png (frames labelled with beat id and time, 3 per row by default).
"""
import argparse
import json
import pathlib
import subprocess
import sys

VIDEO = pathlib.Path(__file__).resolve().parents[1]
STEMS = {"Opening": "ch0_opening", "Slice": "ch1_slice", "TheImage": "ch2_image", "Fibres": "ch3_fibres",
         "Lens": "ch4_lens", "BaseChange": "ch5_base", "Atlas": "ch6_atlas", "Coda": "ch7_coda"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()
    stem = STEMS[a.scene]
    for q in ("1080p30", "480p15"):
        mp4 = VIDEO / "build" / a.scene / "videos" / stem / q / f"{a.scene}.mp4"
        if mp4.exists():
            break
    else:
        sys.exit(f"no render for {a.scene}")
    tl = json.loads((VIDEO / "build" / "timeline" / f"{a.scene}.json").read_text())
    shots = []
    for bid, b in tl["beats"].items():
        for frac in (0.15, 0.55, 0.95):
            shots.append((bid, b["start"] + frac * b["dur"]))
    tmp = VIDEO / "out" / "review" / a.scene
    tmp.mkdir(parents=True, exist_ok=True)
    files = []
    for k, (bid, t) in enumerate(shots):
        f = tmp / f"{k:03d}.png"
        label = f"{bid}  t={t:.1f}s".replace(":", "\\:")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(mp4),
                        "-frames:v", "1", "-vf", f"scale=640:360,drawtext=text='{label}':x=8:y=8:fontsize=18:"
                        "fontcolor=white:box=1:boxcolor=black@0.6", str(f)], stderr=subprocess.DEVNULL)
        files.append(f)
    while len(files) % a.cols:
        files.append(files[-1])
    rows = [files[i:i + a.cols] for i in range(0, len(files), a.cols)]
    inputs, filt, labels = [], "", []
    for r, row in enumerate(rows):
        for f in row:
            inputs += ["-i", str(f)]
    n = 0
    for r, row in enumerate(rows):
        ins = "".join(f"[{n + j}]" for j in range(len(row)))
        filt += f"{ins}hstack=inputs={len(row)}[r{r}];"
        n += len(row)
    filt += "".join(f"[r{r}]" for r in range(len(rows))) + f"vstack=inputs={len(rows)}"
    out = VIDEO / "out" / "review" / f"{a.scene}.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", filt, str(out)],
                   check=True)
    print(out, f"({mp4.parent.name}, {len(shots)} frames)")


if __name__ == "__main__":
    main()
