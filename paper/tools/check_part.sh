#!/bin/bash
# Compile one section file in isolation with the paper's style, bibliography and tables.
# Usage: tools/check_part.sh sections/03a-fibres.tex [section-number]   (section-number: e.g. 3 for Exposé III)
# Prints LaTeX errors and undefined control sequences; undefined references/citations to other parts are expected.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"; F="$1"; N="${2:-0}"; NAME="$(basename "$F" .tex)"
W="$P/_parts/$NAME"; mkdir -p "$W"; cp "$P/fibres.sty" "$P/fibresnat.bst" "$W/"
# figures are found through a link in the work folder: \graphicspath with the absolute path failed on the non-ASCII
# folder name of the project, while relative paths such as figures/rebase work
ln -sfn "$P/figures" "$W/figures"
cat > "$W/doc.tex" <<TEX
\documentclass[11pt]{article}
\usepackage{fibres}
\begin{document}
\input{$P/tables/counts}
\setcounter{section}{$((N>0 ? N-1 : -1))}
\input{$P/$F}
\bibliographystyle{fibresnat}
\bibliography{$P/bib/refs}
\end{document}
TEX
cd "$W" && latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet doc.tex >/dev/null 2>&1
echo "== $F  (pages: $(pdfinfo doc.pdf 2>/dev/null | awk '/Pages/{print $2}'))"
grep -nE "^! " -A4 doc.log | head -40
grep -E "Undefined control sequence|Missing \\$|Runaway|Emergency" doc.log | sort | uniq -c | head
grep -cE "Overfull \\\\hbox" doc.log | sed 's/^/overfull hboxes: /'
echo "PDF: $W/doc.pdf"
