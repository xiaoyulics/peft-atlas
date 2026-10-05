# A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges

A categorical atlas of parameter-efficient fine-tuning (PEFT): every weight-space PEFT method read as a pointed
map into the pretrained point θ₀, studied through its image (what it can express), its fibres and metric (how it
trains), and base change (initialisation, quantisation, restarts, merging). 382 verified methods, 138 papers,
41 numbered results (each attacked by independent referees in two rounds), 20 interactive figures.

## Layout
- `site/` — the static website. Open `site/index.html` directly, or serve the folder (GitHub Pages: publish `site/`).
  - `sections/NN-*.html` — the exposés (prose); `js/fig-*.js` — one file per interactive figure; `css/atlas.css` — design tokens.
  - `data/atlas-data.js` — generated; do not edit by hand.
- `research/` — the catalogue: `catalog.json` (merged, verified), `arxiv/meta.json` (authoritative metadata),
  `classified/` (seven coordinates, arrows, obstructions), `display/` (math-delimited display text).
- `theory/` — `framework.md` (full exposition), `propositions.json` (statements, proofs, review status),
  `round2_verdicts.json`, `axes.json`, `rosetta.json`, `framework_checks.py` (re-runs every numerical claim).

## Build
```bash
python3 site/build/build_data.py      # research + theory -> site/data/atlas-data.js
ATLAS_SITE_URL=https://xiaoyulics.com/peft-atlas/ python3 site/build/assemble.py   # -> site/index.html and dist/artifact.html
python3 theory/framework_checks.py     # numerical checks behind the [Num] claims
```
Set `ATLAS_SITE_URL` to the published root so the social card (`site/assets/card.png`) is an absolute `og:image` that X can fetch.

**Publishing.** Every push to `main` that touches `site/`, `research/` or `theory/` rebuilds the site with
`.github/workflows/pages.yml` and publishes it with GitHub Pages at <https://xiaoyulics.com/peft-atlas/>. You can
also run the workflow by hand from the Actions tab.

## QA
`bash site/test/figtest.sh <figure>` renders one figure headless; `node site/test/qa-site.mjs index.html <outdir>`
(run inside `site/`) checks the whole page for console errors, unmounted figures, overflow and broken anchors.

## Explainer film
`video/` holds a narrated explainer of about 17 minutes:
- the script, with each claim checked against the paper (`video/script/SCRIPT.md`);
- the Manim scenes (`video/film/`);
- the computations behind every number on screen (`video/sim/`);
- the narration and assembly tools (`video/tools/`).

See `video/README.md`. The repository does not track renders, the Python environment or the synthesised audio. The
film plays on the companion site at <https://xiaoyulics.com/peft-atlas/film/> and in its overview. The MP4 is an asset
of the latest GitHub release, which the site build fetches; its poster, captions and chapters are in `site/film/`.

## License
This repository is dual-licensed.
- **Code: MIT** (`LICENSE`). This covers the website's scripts, styles and build and test tools (`site/js/`,
  `site/css/`, `site/build/`, `site/test/`), the numerical checks (`theory/*.py`), the catalogue scripts
  (`research/*.py`), the paper's build and figure scripts (`paper/tools/`, `paper/figures/src/*.py`), and the
  film's code (`video/film/`, `video/sim/`, `video/tools/`, `video/styles/`).
- **Content: CC BY 4.0** (`LICENSE-CONTENT.md`). This covers the paper (sources, figures and PDF), the website's
  prose, the catalogue and theory data, and the film's script and narration. You may reuse and adapt it for any
  purpose, provided you credit the authors.
- **Third party.** The fonts in `paper/figures/src/fonts/` are under the SIL Open Font License 1.1, with their
  license texts beside them. Libraries the website loads from CDNs (MathJax, d3, web fonts) keep their own licenses.

## How to cite
```bibtex
@misc{li2026categorical,
  title        = {{A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges}},
  author       = {Xiaoyu Li and Zhizhou Sha and Chiwun Yang and Dai Shi},
  year         = {2026},
  howpublished = {\url{https://xiaoyulics.com/peft-atlas/}},
  note         = {Technical report}
}
```
`CITATION.cff` has the same information for GitHub's "Cite this repository" button.
