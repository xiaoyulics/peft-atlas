#!/usr/bin/env python3
"""Narrate video/script/SCRIPT.md beat by beat with Gemini TTS, then build a table read.

A beat may carry a line "**Delivery.** ..." that is appended to the default style for that beat only.
Each beat's narration (its lines starting with "> ") becomes video/audio/beats/<id>.wav, with <id>.json holding the
duration, the pauses found by silence detection (cue times for the animation) and a hash of the text, voice and style,
so unchanged beats are not synthesised again.

  python3 video/tools/narrate.py              # synthesise new or changed beats, then build the table read
  python3 video/tools/narrate.py --only II.3  # just these beats (comma-separated)
  python3 video/tools/narrate.py --force      # synthesise every beat again
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import wave

HERE = pathlib.Path(__file__).resolve().parent
VIDEO = HERE.parent
sys.path.insert(0, str(HERE))
import tts  # noqa: E402

SCRIPT = VIDEO / "script" / "SCRIPT.md"
BEATS = VIDEO / "audio" / "beats"
RATE = 24000
GAP = 0.5  # seconds of silence between beats in the table read


def parse(path):
    beats = []
    for block in re.split(r"^### ", path.read_text(), flags=re.M)[1:]:
        head = block.splitlines()[0]
        m = re.search(r"⏱ (\d+):(\d+)", head)
        beats.append(dict(
            id=head.split(" · ")[0].strip(),
            title=head.split(" · ")[1].split("⏱")[0].strip() if " · " in head else "",
            budget=int(m.group(1)) * 60 + int(m.group(2)) if m else 0,
            text=" ".join(l[2:].strip() for l in block.splitlines() if l.startswith("> ")),
            delivery=(re.search(r"^\*\*Delivery\.\*\* (.+)$", block, flags=re.M) or [None, ""])[1].strip()))
    return beats


def pauses(wav):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(wav), "-af", "silencedetect=noise=-38dB:d=0.25",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]
    return [[round(s, 3), round(e, 3)] for s, e in zip(starts, ends)]


def synth(beat, force):
    wav, meta = BEATS / f"{beat['id']}.wav", BEATS / f"{beat['id']}.json"
    style = tts.DEFAULT_STYLE + (f"; {beat['delivery']}" if beat["delivery"] else "")
    key = hashlib.sha256("\n".join([tts.DEFAULT_MODEL, tts.DEFAULT_VOICE, style, beat["text"]]).encode())
    key = key.hexdigest()[:16]
    if not force and wav.exists() and meta.exists() and json.loads(meta.read_text()).get("hash") == key:
        return "cached"
    data = tts.synthesize(beat["text"], tts.DEFAULT_VOICE, style, tts.DEFAULT_MODEL)
    wav.write_bytes(data)
    meta.write_text(json.dumps(dict(hash=key, seconds=round(tts.wav_seconds(data), 3), pauses=pauses(wav),
                                    text=beat["text"]), indent=1))
    return "new"


def pcm(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def silence(sec):
    return b"\x00\x00" * int(round(sec * RATE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated beat ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests (keep low on low rate limits)")
    a = ap.parse_args()

    beats = parse(SCRIPT)
    BEATS.mkdir(parents=True, exist_ok=True)
    todo = [b for b in beats if b["text"] and (not a.only or b["id"] in a.only.split(","))]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(synth, b, a.force): b["id"] for b in todo}
        for f in cf.as_completed(futs):
            try:
                status = f.result()
            except BaseException as e:  # tts.synthesize exits with the API's message
                status = f"FAILED: {e}"
            print(f"  {futs[f]:6s} {status}", flush=True)

    # table read: narrated beats in order, wordless beats as silence of their budgeted length
    parts, clock, rows, chapter = [], 0.0, [], None
    for b in beats:
        ch = b["id"].split(".")[0]
        if ch != chapter:
            rows.append(f"\n{clock / 60:5.2f} min  chapter {ch}")
            chapter = ch
        if b["text"]:
            wav = BEATS / f"{b['id']}.wav"
            if not wav.exists():
                print(f"missing audio for {b['id']}; table read not built"); return
            sec = json.loads((BEATS / f"{b['id']}.json").read_text())["seconds"]
            parts.append(pcm(wav))
        else:
            sec = b["budget"]
            parts.append(silence(sec))
        parts.append(silence(GAP))
        rows.append(f"   {b['id']:6s} {sec:6.1f} s  (budget {b['budget']:3d} s{', over' if sec > b['budget'] + 0.5 else ''})  {b['title']}")
        clock += sec + GAP

    raw = VIDEO / "audio" / "table-read.wav"
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(b"".join(parts))
    out = VIDEO / "audio" / "table-read.m4a"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "128k", str(out)], check=True)
    raw.unlink()
    report = "\n".join(rows) + f"\n\ntotal {clock / 60:.2f} min\n"
    (VIDEO / "audio" / "table-read.txt").write_text(report)
    print(report)
    print(out)


if __name__ == "__main__":
    main()
