#!/usr/bin/env python3
"""Assemble the film: chapter videos end to end, narration laid at each beat's start, subtitles and chapter marks.

Chapter offsets come from the measured lengths of the rendered videos, and beat starts from each scene's timeline
(video/build/timeline/<Scene>.json), so the narration lands where the picture expects it.

  python3 video/tools/assemble.py            # all chapters that have a final render; warns about missing ones
  python3 video/tools/assemble.py --draft    # use 480p previews where finals are missing (for a quick check)

Outputs in video/out/: the film (.mp4), English subtitles (.srt) and chapter marks for YouTube (.txt).
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import textwrap
import wave

import numpy as np

VIDEO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO / "film"))
import core  # noqa: E402

OUT = VIDEO / "out"
RATE = 24000
NAME = "categorical-atlas-explainer"
CHAPTERS = [  # (scene, file stem, chapter mark title, first beat for the mark; extra marks inside the scene)
    ("Opening", "ch0_opening", [("0.1", "Two runs, one path"), ("P.1", "The nut and the sea")]),
    ("Slice", "ch1_slice", [("I.0", "I · Fine-tuning moves a point")]),
    ("TheImage", "ch2_image", [("II.0", "II · The image: what a method can say")]),
    ("Fibres", "ch3_fibres", [("III.0", "III · Fibres: how a method learns")]),
    ("Lens", "ch4_lens", [("IV.0", "IV · The lens: GaLore")]),
    ("BaseChange", "ch5_base", [("V.0", "V · Base change: how a method travels")]),
    ("Atlas", "ch6_atlas", [("A.0", "The atlas: seven coordinates, 382 methods")]),
    ("Coda", "ch7_coda", [("C.1", "Three questions")]),
]
PRINT_FORMS = [  # narration spelling -> subtitle spelling
    ("B-transpose-B minus A-A-transpose", "BᵀB − AAᵀ"),
    ("theta-zero", "θ₀"), ("LoRA-plus", "LoRA+"), ("I-A-cubed", "(IA)³"),
    ("Q-LoRA", "QLoRA"), ("Re-LoRA", "ReLoRA"),
]


def run(cmd):
    subprocess.run(cmd, check=True)


def duration(mp4):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mp4), "-map", "0:v:0", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    frames = int(re.findall(r"frame=\s*(\d+)", out)[-1])
    fps = float(re.search(r"(\d+(?:\.\d+)?) fps", out).group(1))
    return frames / fps


def find_render(scene, stem, draft):
    final = VIDEO / "build" / scene / "videos" / stem / "1080p30" / f"{scene}.mp4"
    if final.exists():
        return final, "1080p30"
    if draft:
        prev = VIDEO / "build" / scene / "videos" / stem / "480p15" / f"{scene}.mp4"
        if prev.exists():
            return prev, "480p15"
    return None, None


# ---------------------------------------------------------------- a quiet pad under cards, shore cuts and the coda
CHORDS = [  # Hz: Dmaj9, Bm9, Gmaj7(#11), Em9 -- soft, open voicings
    [146.83, 220.00, 277.18, 329.63, 369.99],
    [123.47, 185.00, 220.00, 277.18, 293.66],
    [98.00, 146.83, 185.00, 220.00, 277.18],
    [164.81, 246.94, 293.66, 369.99, 392.00],
]


def pad_segment(seconds, chord, rate=RATE, attack=1.6, release=2.2):
    t = np.arange(int(seconds * rate)) / rate
    sig = np.zeros_like(t)
    for k, f in enumerate(chord):
        for detune in (-0.12, 0.12):
            ph = 2 * np.pi * (f + detune) * t + k
            sig += np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    sig *= 1 + 0.08 * np.sin(2 * np.pi * 0.11 * t)            # slow breathing
    env = np.minimum(1, t / attack) * np.minimum(1, (seconds - t) / release).clip(0, 1)
    sig *= env / (len(chord) * 2 * 1.3)
    # one-pole low-pass at about 1.6 kHz to soften the upper partials
    a = np.exp(-2 * np.pi * 1600 / rate)
    out = np.empty_like(sig)
    acc = 0.0
    for i, x in enumerate(sig):
        acc = (1 - a) * x + a * acc
        out[i] = acc
    return out


def music_bed(total, windows):
    """windows: list of (start, end, level) with level a linear amplitude (1.0 = full scale)."""
    bed = np.zeros(int(round(total * RATE)) + RATE)
    for k, (s, e, level) in enumerate(windows):
        seg = pad_segment(e - s, CHORDS[k % len(CHORDS)]) * level
        i = int(round(s * RATE))
        if i < 0:                       # a window that starts before the film: drop its head
            seg, i = seg[-i:], 0
        seg = seg[:max(0, len(bed) - i)]
        bed[i:i + len(seg)] += seg
    return bed


def srt_time(t):
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}".replace(",1000", ",999")


def printed(text):
    for a, b in PRINT_FORMS:
        text = text.replace(a, b)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--no-music", action="store_true", help="narration only")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    work = OUT / "work"
    work.mkdir(exist_ok=True)

    parts, offset, audio, cues, marks, windows, thumb_at = [], 0.0, [], [], [], [], None
    for scene, stem, chapter_marks in CHAPTERS:
        mp4, quality = find_render(scene, stem, a.draft)
        tl_file = core.TIMELINE / f"{scene}.json"
        if not mp4 or not tl_file.exists():
            print(f"  missing: {scene} (no render or timeline) - skipped")
            continue
        tl = json.loads(tl_file.read_text())
        dur = duration(mp4)
        drift = dur - tl["total"]
        print(f"  {scene:10s} {quality:7s} {dur:7.2f} s  (timeline {tl['total']:.2f}, {drift:+.2f})")
        if abs(drift) > 0.2:
            print(f"    warning: {scene} differs from its timeline by {drift:+.2f} s")
        # normalise every part to 1080p30 so the concat demuxer can copy streams
        norm = work / f"{scene}.mp4"
        if quality == "1080p30":
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-c", "copy", "-an", str(norm)])
        else:
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                 "scale=1920:1080:flags=lanczos,fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                 "-an", str(norm)])
            dur = duration(norm)
        parts.append(norm)
        for bid, b in tl["beats"].items():
            if b["audio"] > 0:
                audio.append((offset + b["start"], VIDEO / "audio" / "beats" / f"{bid}.wav"))
                for s, e, txt in core.sentences(bid):
                    cues.append((offset + b["start"] + s, offset + b["start"] + e, printed(txt)))
        # music: under wordless beats (title, chapter cards), shore cuts (picture-only tails) and the coda
        for bid, b in tl["beats"].items():
            if b["audio"] == 0:
                windows.append((offset + b["start"] - 0.4, offset + b["end"] + 0.8, 0.13))
            elif b["dur"] - b["audio"] > 3.0:
                windows.append((offset + b["start"] + b["audio"] - 0.5, offset + b["end"] + 0.6, 0.10))
        if scene == "Coda":
            windows.append((offset, offset + dur, 0.04))
        if scene == "Opening" and "0.1" in tl["beats"]:
            windows.append((offset, offset + tl["beats"]["0.1"]["start"] + 1.0, 0.08))
        if scene == "Opening" and "0.3" in tl["beats"]:
            thumb_at = offset + (tl["beats"]["0.3"]["start"] + tl["beats"]["0.3"]["end"]) / 2
        for bid, title in chapter_marks:
            if bid in tl["beats"]:
                marks.append((offset + tl["beats"][bid]["start"] - (0 if bid != "0.1" else tl["beats"][bid]["start"]),
                              title))
        offset += dur
    if not parts:
        sys.exit("nothing to assemble")

    # picture: concatenate the parts
    lst = work / "parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    picture = work / "picture.mp4"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", str(picture)])

    # sound: every narrated beat at its place on one track
    track = np.zeros(int(round(offset * RATE)) + RATE, dtype=np.int16)
    for t0, wav in audio:
        with wave.open(str(wav)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        i = int(round(t0 * RATE))
        track[i:i + len(pcm)] = pcm[:max(0, len(track) - i)]
    raw = work / "narration.wav"
    mix = track.astype(np.float64) / 32768
    if not a.no_music:
        mix = mix + music_bed(offset, windows)[:len(mix)]
    mix = np.clip(mix, -1, 1)
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    # two-pass loudness normalisation to -16 LUFS
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(raw), "-af",
                            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    m = json.loads(probe[probe.rindex("{"):probe.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    film = OUT / (f"{NAME}-no-music.mp4" if a.no_music else f"{NAME}.mp4")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(picture), "-i", str(raw),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", ln + ",aresample=48000", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", str(film)])

    # subtitles: two lines of at most 42 characters, no overlaps, at least 1 s each
    cues.sort()
    blocks = []
    for k, (s, e, txt) in enumerate(cues):
        nxt = cues[k + 1][0] if k + 1 < len(cues) else e + 2
        e = min(max(e + 0.25, s + 1.0), nxt - 0.05)
        blocks.append(f"{len(blocks) + 1}\n{srt_time(s)} --> {srt_time(e)}\n" + "\n".join(textwrap.wrap(txt, 42)) + "\n")
    (OUT / f"{NAME}.en.srt").write_text("\n".join(blocks))

    if thumb_at is not None and not a.no_music:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{thumb_at:.2f}", "-i", str(picture),
             "-frames:v", "1", "-vf", "scale=1280:720:flags=lanczos", str(OUT / f"{NAME}.thumbnail.png")])
    lines = []
    for t, title in sorted(marks):
        mm, ss = divmod(int(t), 60)
        lines.append(f"{mm}:{ss:02d} {title}")
    (OUT / f"{NAME}.chapters.txt").write_text("\n".join(lines) + "\n")
    print(f"\n{film}  {offset / 60:.2f} min\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
