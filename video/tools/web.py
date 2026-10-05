"""Export the film's web assets into site/film/ after assembling it (tools/assemble.py).

  video/.venv/bin/python video/tools/web.py

Writes the poster, a 1200x630 share card, English captions (WebVTT and SRT) and a chapter track. The MP4 itself is
not tracked in git: it is attached to a GitHub release and fetched when the site is built (.github/workflows/pages.yml).
For a local preview, site/film/categorical-atlas-explainer.mp4 is linked to video/out/.
"""
import json
import pathlib
import re
import subprocess
import sys

from PIL import Image

VIDEO = pathlib.Path(__file__).resolve().parents[1]
ROOT = VIDEO.parent
sys.path.insert(0, str(VIDEO / "film"))
sys.path.insert(0, str(VIDEO / "tools"))
import core  # noqa: E402
import assemble  # noqa: E402

OUT = VIDEO / "out"
FILM = OUT / f"{assemble.NAME}.mp4"
SITE = ROOT / "site" / "film"
NAVY = (10, 18, 28)


def vtt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def chapters():
    """[(film time, title)] at each chapter's first beat, as tools/assemble.py marks them, plus the film's length."""
    marks, t = [], 0.0
    for scene, _stem, scene_marks in assemble.CHAPTERS:
        tl = json.loads((core.TIMELINE / f"{scene}.json").read_text())
        for bid, title in scene_marks:
            if bid in tl["beats"]:
                marks.append((0.0 if bid == "0.1" else t + tl["beats"][bid]["start"], title))
        t += assemble.duration(OUT / "work" / f"{scene}.mp4")
    return sorted(marks), t


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    marks, total = chapters()

    # poster: the title card, where tools/assemble.py takes the thumbnail
    opening = json.loads((core.TIMELINE / "Opening.json").read_text())["beats"]["0.3"]
    at = (opening["start"] + opening["end"]) / 2
    png = SITE / "_poster.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at:.2f}", "-i", str(FILM),
                    "-frames:v", "1", str(png)], check=True)
    poster = Image.open(png).convert("RGB")
    png.unlink()
    poster.save(SITE / "poster.jpg", quality=88, optimize=True, progressive=True)
    card = Image.new("RGB", (1200, 630), NAVY)       # share card: the poster, letterboxed to 1.91:1
    small = poster.resize((1120, 630), Image.LANCZOS)
    card.paste(small, (40, 0))
    card.save(SITE / "card.jpg", quality=90, optimize=True)

    # captions: the film's SRT as WebVTT (and the SRT itself, for download)
    srt = (OUT / f"{assemble.NAME}.en.srt").read_text()
    (SITE / f"{assemble.NAME}.en.srt").write_text(srt)
    vtt = re.sub(r"(\d\d:\d\d:\d\d),(\d\d\d)", r"\1.\2", srt)
    (SITE / f"{assemble.NAME}.en.vtt").write_text("WEBVTT\n\n" + vtt)

    # chapters, for the player and for the lists on the pages
    lines = ["WEBVTT", ""]
    for k, (t, title) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else total
        lines += [f"{k + 1}", f"{vtt_time(t)} --> {vtt_time(end)}", title, ""]
    (SITE / "chapters.vtt").write_text("\n".join(lines))

    link = SITE / FILM.name
    if not link.exists():
        link.symlink_to(FILM)
    mb = FILM.stat().st_size / 1e6
    print(f"site/film: poster.jpg, card.jpg, captions, {len(marks)} chapters; film {total / 60:.2f} min, {mb:.0f} MB")
    for t, title in marks:
        print(f"  {int(t) // 60}:{int(t) % 60:02d}  {title}")


if __name__ == "__main__":
    main()
