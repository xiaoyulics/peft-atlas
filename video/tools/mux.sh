#!/bin/bash
# Lay a narration track under a rendered clip: loudness to -16 LUFS, video stream copied as is.
#   bash video/tools/mux.sh <video.mp4> <narration.wav> <out.mp4>
set -euo pipefail
ffmpeg -hide_banner -loglevel error -y -i "$1" -i "$2" \
  -map 0:v -map 1:a -c:v copy \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000" -c:a aac -b:a 192k \
  -movflags +faststart "$3"
echo "$3"
