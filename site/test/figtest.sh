#!/bin/bash
# Usage: test/figtest.sh <figure-name> [extra-class]
# Builds test/fig-<name>.html that mounts only that figure, then renders it headless at
# desktop (1280) and phone (390) widths in light and dark, writing PNGs + a console log.
set -u
NAME="$1"; CLS="${2:-figure wide}"
DIR="$(cd "$(dirname "$0")" && pwd)"; SITE="$(dirname "$DIR")"
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for THEME in light dark; do
cat > "$DIR/fig-$NAME-$THEME.html" <<HTML
<!doctype html><html data-theme="$THEME"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..600;1,6..96,400..600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300..700;1,8..60,300..700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="../css/atlas.css">
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)'],['\$','\$']],displayMath:[['\\\\[','\\\\]'],['\$\$','\$\$']]},svg:{fontCache:'global'}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
</head><body><main class="page"><div class="atlas"><div class="prose">
<figure class="$CLS"><div data-figure="$NAME"></div><figcaption><span class="fig-n">Test</span>$NAME ($THEME)</figcaption></figure>
</div></div></main>
<script src="../data/atlas-data.js"></script>
<script src="../js/core.js"></script>
<script src="../js/fig-$NAME.js"></script>
<script>window.addEventListener('error',function(e){console.error('UNCAUGHT '+e.message+' @'+e.filename+':'+e.lineno)});</script>
</body></html>
HTML
done
: > "$DIR/fig-$NAME.console.txt"
for THEME in light dark; do
  for W in 1280 390; do
    H=1800
    timeout 90 "$CH" --headless=new --disable-gpu --no-first-run --hide-scrollbars --allow-file-access-from-files \
      --enable-logging=stderr --v=0 --window-size=$W,$H --virtual-time-budget=8000 \
      --screenshot="$DIR/fig-$NAME-$THEME-$W.png" "file://$DIR/fig-$NAME-$THEME.html" 2>&1 \
      | grep -E "CONSOLE|Uncaught|ERROR:.*(js|JavaScript)" | grep -v cv_display_link >> "$DIR/fig-$NAME.console.txt"
  done
done
echo "PNGs: $(ls "$DIR"/fig-$NAME-*.png | tr '\n' ' ')"
echo "--- console (errors/warnings/logs) ---"; sort -u "$DIR/fig-$NAME.console.txt" | head -40
