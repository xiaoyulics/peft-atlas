"""Export the film's web assets into site/film/ after assembling it (tools/assemble.py).

  video/.venv/bin/python video/tools/web.py

Writes the poster, a 1200x630 share card and the English captions (SRT). The film plays from YouTube; the MP4 is
not tracked in git but attached to the GitHub release. The script prints the chapter times for the pages and YouTube.
"""
import json
import pathlib
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

    # captions, for download
    (SITE / f"{assemble.NAME}.en.srt").write_text((OUT / f"{assemble.NAME}.en.srt").read_text())

    mb = FILM.stat().st_size / 1e6
    print(f"site/film: poster.jpg, card.jpg, captions; film {total / 60:.2f} min, {mb:.0f} MB; chapters:")
    for t, title in marks:
        print(f"  {int(t) // 60}:{int(t) % 60:02d}  {title}")


if __name__ == "__main__":
    main()
