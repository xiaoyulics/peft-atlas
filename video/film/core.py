"""Timing for the film: beat audio, phrase cues and the per-chapter timeline.

Every narrated beat has video/audio/beats/<id>.wav and <id>.json (duration, silences found by ffmpeg silencedetect,
the narration text). `phrase_time(bid, phrase)` returns when, inside that beat's audio, the phrase starts: detected
pauses are matched in order to punctuation and <pause> tags in the text, and the time between matched pauses is shared
out in proportion to the characters spoken. Accurate to a few tenths of a second, which is what visual cues need.

A chapter scene lists its beats; `plan()` lays them end to end (narration plus optional visual-only time, then a
gap) and the scene writes the result to video/build/timeline/<Scene>.json, which assemble.py reads to place the audio.
"""
import functools
import json
import pathlib
import re

VIDEO = pathlib.Path(__file__).resolve().parents[1]
BEATS = VIDEO / "audio" / "beats"
SCRIPT = VIDEO / "script" / "SCRIPT.md"
TIMELINE = VIDEO / "build" / "timeline"
GAP = 0.5   # seconds of silence between consecutive beats


# ---------------------------------------------------------------- script and beat audio
@functools.lru_cache(None)
def script_beats():
    """{id: dict(title, budget, text)} in script order."""
    out = {}
    for block in re.split(r"^### ", SCRIPT.read_text(), flags=re.M)[1:]:
        head = block.splitlines()[0]
        m = re.search(r"⏱ (\d+):(\d+)", head)
        bid = head.split(" · ")[0].strip()
        out[bid] = dict(title=head.split(" · ")[1].split("⏱")[0].strip() if " · " in head else "",
                        budget=int(m.group(1)) * 60 + int(m.group(2)) if m else 0,
                        text=" ".join(l[2:].strip() for l in block.splitlines() if l.startswith("> ")))
    return out


@functools.lru_cache(None)
def beat_audio(bid):
    """dict(seconds, pauses, text) for a narrated beat, or None for a wordless one."""
    f = BEATS / f"{bid}.json"
    if not script_beats()[bid]["text"]:
        return None
    if not f.exists():
        raise FileNotFoundError(f"no narration for {bid}; run python3 video/tools/narrate.py")
    return json.loads(f.read_text())


# ---------------------------------------------------------------- phrase cues
def _clean(text):
    """Text without tags, whitespace collapsed, and the clean positions (start of the next word) where tags stood."""
    raw = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "\x00", text)).strip()
    clean, tags = "", []
    for ch in raw:
        if ch == "\x00":
            tags.append(len(clean))
            continue
        if ch == " " and (not clean or clean.endswith(" ")):
            continue
        clean += ch
    return clean.rstrip(), tags


@functools.lru_cache(None)
def _mapping(bid):
    a = beat_audio(bid)
    clean, tag_pos = _clean(a["text"])
    C, T = len(clean), a["seconds"]
    pauses = [tuple(p) for p in a["pauses"]]
    lead = pauses[0][1] if pauses and pauses[0][0] < 0.05 else 0.0
    tail = pauses[-1][0] if pauses and pauses[-1][1] > T - 0.05 else T
    inner = [p for p in pauses if p[0] > lead + 0.05 and p[1] < tail - 0.05]

    # candidate boundaries: (char index where the next words start, strength)
    cands = {}
    for m in re.finditer(r"[.?!:;,—](?=\s)", clean):
        strength = 3 if m.group() in ".?!" else 2 if m.group() in ":;—" else 1
        k = m.end() + 1
        cands[k] = max(cands.get(k, 0), strength)
    for k in tag_pos:
        cands[k] = max(cands.get(k, 0), 4)
    cands = sorted((k, s) for k, s in cands.items() if 0 < k < C)

    # expected character position of each inner pause, from speech time spoken before it
    speech = (tail - lead) - sum(e - s for s, e in inner)
    done, prev_end, est = 0.0, lead, []
    for s, e in inner:
        done += s - prev_end
        est.append(C * done / max(speech, 1e-6))
        prev_end = e

    # dynamic programme: match pauses to boundaries in order; a pause may stay unmatched at a cost
    INF, SKIP = 1e9, 0.6
    K, M = len(inner), len(cands)
    best = [[INF] * (M + 1) for _ in range(K + 1)]
    back = [[None] * (M + 1) for _ in range(K + 1)]
    for j in range(M + 1):
        best[0][j] = 0.0
    for i in range(1, K + 1):
        for j in range(M + 1):
            # pause i unmatched
            if best[i - 1][j] + SKIP < best[i][j]:
                best[i][j], back[i][j] = best[i - 1][j] + SKIP, ("skip", j)
            if j:
                # pause i matched to boundary j (1-based), previous pauses used boundaries < j
                pos, strength = cands[j - 1]
                cost = 12 * abs(pos - est[i - 1]) / C + 0.12 * (4 - strength)
                cand = min(best[i - 1][jj] for jj in range(j)) + cost
                if cand < best[i][j]:
                    jj = min(range(j), key=lambda x: best[i - 1][x])
                    best[i][j], back[i][j] = cand, ("match", jj)
    j = min(range(M + 1), key=lambda x: best[K][x])
    anchors, i = [], K
    while i > 0:
        kind, jj = back[i][j]
        if kind == "match":
            anchors.append((cands[j - 1][0], inner[i - 1]))
        i, j = i - 1, (jj if kind == "match" else j)
    anchors.sort()

    # piecewise-linear map from character index to time
    knots = [(0, lead)]
    for pos, (s, e) in anchors:
        knots.append((pos - 1e-6, s))
        knots.append((pos, e))
    knots.append((C, tail))

    def char_time(c):
        for (c0, t0), (c1, t1) in zip(knots, knots[1:]):
            if c0 <= c <= c1:
                return t0 + (t1 - t0) * (c - c0) / (c1 - c0) if c1 > c0 else t1
        return tail
    return clean, char_time


def phrase_time(bid, phrase, end=False):
    """Seconds into beat bid's audio at which `phrase` starts (or ends, with end=True)."""
    clean, char_time = _mapping(bid)
    k = clean.find(phrase)
    if k < 0:
        raise KeyError(f"phrase {phrase!r} not found in the narration of {bid}")
    return char_time(k + len(phrase) if end else k)


def sentences(bid, max_chars=84):
    """Subtitle chunks [(start, end, text)] in beat time: sentences, split at commas when longer than max_chars."""
    clean, char_time = _mapping(bid)
    chunks, start = [], 0
    for m in re.finditer(r"(?<=[.?!])\s+", clean + " "):
        sent = clean[start:m.start()].strip()
        if sent:
            pieces, cur = [], ""
            for part in re.split(r"(?<=[,;:—])\s+", sent):
                if cur and len(cur) + 1 + len(part) > max_chars:
                    pieces.append(cur); cur = part
                else:
                    cur = f"{cur} {part}".strip()
            pieces.append(cur)
            pos = start
            for p in pieces:
                k = clean.find(p, pos)
                chunks.append((char_time(k), char_time(k + len(p)), p))
                pos = k + len(p)
        start = m.end()
    return chunks


# ---------------------------------------------------------------- chapter timeline
def plan(beats, extra=None, wordless=None, gap=GAP, pre=None):
    """Lay beats end to end. pre[bid] seconds of picture come before beat bid starts (its narration begins at
    start); narrated beats last their audio plus extra[bid]; wordless beats last wordless[bid] or their budget in
    SCRIPT.md, plus extra[bid]. Returns (dict id -> dict(start, audio, dur, end), total)."""
    extra, wordless, pre, t, out = extra or {}, wordless or {}, pre or {}, 0.0, {}
    for bid in beats:
        t += pre.get(bid, 0.0)
        a = beat_audio(bid)
        audio = a["seconds"] if a else 0.0
        dur = (audio if a else wordless.get(bid, script_beats()[bid]["budget"])) + extra.get(bid, 0.0)
        out[bid] = dict(start=round(t, 3), audio=round(audio, 3), dur=round(dur, 3), end=round(t + dur, 3))
        t += dur + gap
    return out, round(t - gap, 3)


def write_timeline(scene, beats_plan, total):
    TIMELINE.mkdir(parents=True, exist_ok=True)
    (TIMELINE / f"{scene}.json").write_text(json.dumps(dict(scene=scene, total=total, beats=beats_plan), indent=1))
