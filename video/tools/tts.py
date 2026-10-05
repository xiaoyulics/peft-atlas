#!/usr/bin/env python3
"""Narration with Gemini 3.8 Flash TTS (Interactions API), standard library only.

The API key is read from $GEMINI_API_KEY, or else from ~/.config/gemini/api_key.
It is sent only in the x-goog-api-key header to generativelanguage.googleapis.com and is never printed.

  python3 video/tools/tts.py --check
  python3 video/tools/tts.py --text "Fine-tuning moves a point." --out video/audio/test.wav
  python3 video/tools/tts.py --file video/script/01-open.txt --out video/audio/01.wav
"""
import argparse
import base64
import json
import os
import pathlib
import re
import struct
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = "gemini-3.8-flash-tts"
DEFAULT_VOICE = "sulafat"  # chosen narrator (2026-10-05): warm, inviting, comforting; en-US
DEFAULT_STYLE = ("gentle, warm and soft-spoken; an unhurried young woman explaining mathematics "
                 "to a friend, with a light smile in her voice")
AUDIO_TOKENS_PER_SEC = 25          # Gemini pricing page, September 2026
USD_PER_M_AUDIO_TOKENS = 9.00      # gemini-3.8-flash-tts, through 31 December 2026


class TTSError(RuntimeError):
    pass


def api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        f = pathlib.Path.home() / ".config" / "gemini" / "api_key"
        if f.exists():
            key = f.read_text().strip()
    if not key:
        raise TTSError("No API key: set GEMINI_API_KEY or write it to ~/.config/gemini/api_key")
    return key


def find_audio(node, found):
    """Collect base64 audio payloads ({"type": "audio", "data": ...}) anywhere in the response."""
    if isinstance(node, dict):
        if node.get("type") == "audio" and isinstance(node.get("data"), str):
            found.append(node["data"])
        for v in node.values():
            find_audio(v, found)
    elif isinstance(node, list):
        for v in node:
            find_audio(v, found)
    return found


def as_wav(raw, rate=24000):
    """The unary default is a RIFF WAV; wrap headerless 16-bit mono PCM if that is what came back."""
    if raw[:4] == b"RIFF":
        return raw
    header = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(raw))
    return header + raw


def wav_seconds(wav):
    channels, rate = struct.unpack("<HI", wav[22:28])
    bits = struct.unpack("<H", wav[34:36])[0]
    i = wav.find(b"data")
    size = struct.unpack("<I", wav[i + 4:i + 8])[0] if i >= 0 else len(wav) - 44
    return size / (rate * channels * bits / 8)


def synthesize(text, voice, style, model):
    content = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]
    body = {
        "model": model,
        "input": [{"type": "user_input", "content": [content]}],
        "response_format": {"type": "audio", "mime_type": "audio/wav"},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(), method="POST",
        headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 429:
                # per-minute limits: wait as told and retry; per-day limits: stop with the API's message
                m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg)
                wait = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3)) if m else 30.0
                if "per day" not in msg and wait <= 120 and attempt < 7:
                    time.sleep(wait + 1)
                    continue
                limit = re.search(r"limit: [^.)]*", msg)
                raise TTSError(f"rate limit ({limit.group(0) if limit else 'HTTP 429'}); retry in {wait / 3600:.1f} h"
                               " or enable billing for the key's project")
            raise TTSError(f"HTTP {e.code}: {msg[:1500]}")
    chunks = find_audio(resp, [])
    if not chunks:
        raise TTSError("No audio in the response:\n" + json.dumps(resp)[:2000])
    return as_wav(base64.b64decode(chunks[-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file", type=pathlib.Path)
    src.add_argument("--check", action="store_true", help="synthesise one short test sentence")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("video/audio/check.wav"))
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--style", default=DEFAULT_STYLE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    a = p.parse_args()
    text = ("Fine-tuning moves a point. <short pause> This is a test of the narration voice."
            if a.check else a.text if a.text is not None else a.file.read_text())
    try:
        wav = synthesize(text.strip(), a.voice, a.style, a.model)
    except TTSError as e:
        sys.exit(str(e))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(wav)
    sec = wav_seconds(wav)
    cost = sec * AUDIO_TOKENS_PER_SEC * USD_PER_M_AUDIO_TOKENS / 1e6
    print(f"{a.out}  {sec:.1f} s  voice={a.voice}  model={a.model}  ~${cost:.4f} audio output")


if __name__ == "__main__":
    main()
