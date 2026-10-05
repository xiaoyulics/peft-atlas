#!/bin/bash
# Build the arXiv submission package and test it the way arXiv compiles (pdfLaTeX only, no BibTeX run).
# Output: paper/arxiv/ (the tree that is uploaded), paper/categorical-atlas-peft-arxiv.tar.gz, paper/arxiv-test/ (clean compile)
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P"

echo "== 1. fresh full build (with BibTeX) to get an up-to-date main.bbl"
latexmk -pdf -interaction=nonstopmode -quiet main.tex >/dev/null 2>&1 || true
grep -E "^! " main.log && { echo "LaTeX errors in main.log"; exit 1; } || true
test -s main.bbl || { echo "main.bbl missing"; exit 1; }

echo "== 2. assemble the upload tree"
rm -rf arxiv && mkdir -p arxiv/sections arxiv/tables arxiv/figures arxiv/anc
cp main.tex main.bbl fibres.sty fibresnat.bst arxiv/
cp sections/*.tex arxiv/sections/
cp tables/*.tex arxiv/tables/
# only the figures the paper includes
grep -ho 'includegraphics\(\[[^]]*\]\)\?{[^}]*}' main.tex sections/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u | while read f; do
  src="$f"; [ -f "$src" ] || src="$f.pdf"
  [ -f "$src" ] && cp "$src" "arxiv/figures/$(basename "$src")" || echo "  missing figure: $f"
done
cp anc/* arxiv/anc/
# arXiv does not need the .bib once main.bbl is present; keep the bib out of the upload to avoid a stale rebuild
echo "  files: $(find arxiv -type f | wc -l | tr -d ' '), size: $(du -sh arxiv | cut -f1)"

echo "== 3. sanity checks"
head -5 arxiv/main.tex | grep -q 'pdfoutput=1' && echo "  pdfoutput=1 present" || echo "  WARNING: add \\pdfoutput=1 to the first lines of main.tex"
grep -rn --include=*.tex -E '\\graphicspath|/Users/|\\write18|\\immediate\\write' arxiv && echo "  WARNING: absolute paths or shell escapes above" || echo "  no absolute paths or shell escapes"
LC_ALL=C find arxiv -name '*[^ -~]*' | grep . && echo "  WARNING: non-ASCII file names" || echo "  file names are ASCII"

echo "== 4. clean compile as arXiv does (pdflatex x3, no bibtex)"
rm -rf arxiv-test && cp -R arxiv arxiv-test && cd arxiv-test
for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true; done
grep -E "^! " main.log && { echo "  ERRORS in clean compile"; exit 1; } || echo "  no errors"
echo "  undefined refs/cites: $(grep -cE 'undefined' main.log)"
echo "  pages: $(pdfinfo main.pdf | awk '/Pages/{print $2}')"
pdffonts main.pdf | awk 'NR>2 && $(NF-4)!="yes"{print "  NOT EMBEDDED: "$0}' | head
cd "$P"

echo "== 5. tarball"
rm -f categorical-atlas-peft-arxiv.tar.gz
tar -czf categorical-atlas-peft-arxiv.tar.gz -C arxiv .
ls -la categorical-atlas-peft-arxiv.tar.gz
cp arxiv-test/main.pdf categorical-atlas-peft.pdf
echo "PDF: $P/categorical-atlas-peft.pdf"
