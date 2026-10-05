# Writing guide for the paper "A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges" (read fully before writing)

The paper is the arXiv version of the website in `../site/` (the "companion site"). It is a long, complete research
paper: nothing in the theory is cut. Every agent writes LaTeX into its own file(s) under `paper/sections/` and must
compile them with `tools/check_part.sh sections/<file>.tex <exposé number>`.

## Sources, in order of authority
1. `../theory/propositions.json`: every numbered result with `statement`, `proof`, `status`, `round2`, `public_statement`
   and referee notes. **The paper never states a result more strongly than its `public_statement`** (two rounds of
   adversarial review narrowed many results; their hypotheses must appear).
2. `../theory/framework.md`: the complete exposition (≈35k words, Markdown + `$…$` maths), already synced with both
   review rounds. It is the base text for content, order and numbering. Convert it fully; do not summarise.
3. `../site/sections/*.html`: the website's refereed text. It contains later corrections and additions that you must
   carry over (e.g. the LOFT example in `04-erlangen.html#erlangen-loft`, notes labelled "2026" about methods from the
   2026 literature sweep, caption-level corrections). Where the site and framework.md disagree on a mathematical claim,
   the version that agrees with `propositions.json` (`public_statement`, `round2`) wins.
4. `../theory/axes.json`, `../theory/rosetta.json`, `../research/classified/report.md`, `tables/*.tex` (generated data).

## Numbering and labels (must match the site exactly)
- Sections are exposés with Roman numerals: 0 Thesis, I The pointed slice, II The image germ (II.A strata, II.B Erlangen,
  II.C cut and volume), III Fibres and metrics, IV The lens, V Base change, VI Extensions and mergeability,
  VII Worked examples, VIII The classification (VIII.A–C), IX Rosetta stone, X Falsifiable predictions,
  XI Analogy and open problems, XII Related work, XIII Conclusion. Appendices A–E.
- Section labels: `sec:thesis, sec:slice, sec:image, sec:strata, sec:erlangen, sec:cut, sec:fibres, sec:lens, sec:base,
  sec:merge, sec:examples, sec:atlas, sec:rosetta, sec:predictions, sec:analogy, sec:related, sec:conclusion`;
  appendices `app:background, app:auxiliary, app:proofs, app:catalogue, app:repro`.
- Every numbered item keeps its framework number. Use the template environments, which take the number and a name:
  `\begin{theorem}{II.7}{HRA is the meet of LoRA with the rotation orbit} … \end{theorem}`.
  Environments: `theorem, proposition, lemma, corollary, observation, conjecture` (italic body, teal rule),
  `definition` (ochre rule), `remark, example, prediction` (upright), `analogy` (dashed rule), unnumbered
  `example*{name}` and `remark*{name}`. Each creates the label `res:<number>`; cite with `\thmref{Theorem}{II.7}`.
- "What this explains" paragraphs go in `\begin{explains} … \end{explains}`.
- Appendix C restates each result with `\begin{restate}{Theorem}{II.7}{name} … \end{restate}` (no label) followed by
  `\begin{proof} … \end{proof}`, one `\subsection{Theorem II.7}\label{proof:II.7}` per result, in framework order.
- Appendix A entries keep the site's numbers (A.1 … D.34): `\begin{bgentry}{C.13}{name}` / `\begin{bgfact}{D.12}{name}`,
  cited with `\bgref{C.13}`. Appendix B: `\begin{auxresult}{B.3}{name}{\citep{key}} … \end{auxresult}`, `\auxref{B.3}`.

## Proofs
- No proofs and no proof sketches in the main text. Exposé 0 states once that complete proofs of all numbered results are
  in Appendix C, in the same order. Numerical checks marked [Num] are reported where the framework reports them,
  with a pointer to Appendix E.
- Appendix C proofs are complete. Use the repaired proof from `propositions.json` and the framework; incorporate the
  round-2 `suggested_fix`es (`../theory/round2_verdicts.json`). Quote external theorems through Appendix B.

## Mathematics and notation
- Keep the site's notation: `W_0 \in \R^{m\times n}`, `y = Wx`, `\theta_0`, `\Theta`, `\rho:(Q,q_0)\to(\Theta,\theta_0)`,
  `S_1`, `d(M)`, `|M|`, `K = D\rho\,\Lambda\,D\rho^\top`, `\Meth(\Theta,\theta_0)`. Matrices are plain italic capitals as on
  the site (a deliberate choice so readers can move between site and paper).
- Macros from `fibres.sty`: `\R \N \Meth \Diff \Para \Set \FinVect \Lens \Mr` (= `\mathcal M_{\le r}`) `\MM \Im \rk \tr \Gr
  \skewop \symop \diag \vect \spanop \id \gl \so \Sym \GL \SO \Ort \CO \T` (transpose). Use `\operatorname{…}` for others.
- Important formulas go in single-line display equations. No `\!`. Escape nothing MathJax-specific: write `<`, `>`
  (never `\lt`, `\gt`).

## Citations
- `\citep{key}` / `\citet{key}` with keys from `bib/keys.tsv` (502 verified entries: every catalogued method and paper).
  Find keys by grepping that file (by name, arXiv id or title).
- A reference that is not there (textbooks, classical mathematics: Klein 1872, Scherk 1950, Kempf–Ness 1979, Van Loan–
  Pitsianis 1993, Yamabe 1950, Lee's *Introduction to Smooth Manifolds*, Mac Lane, Riehl, Leinster, …): add a complete,
  correct BibTeX entry to `bib/extra/<your-file-stem>.bib` with a key starting with `x` (e.g. `xkempf1979length`).
  Verify every field against the publisher, DOI or arXiv page; never invent an entry. Run `python3 tools/make_bib.py`
  afterwards so `bib/refs.bib` includes it.
- Credit prior work exactly where the framework marks a result as Known (e.g. "[Known: Tarmoun et al. 2021]"): write it
  into the result's name or the sentence after it, e.g. `\begin{theorem}{III.9}{closed-form isotropic-charge dynamics; known, \citet{…}}`.

## Figures and tables
- Figures are vector PDFs in `figures/` made by scripts in `figures/src/` (style module `figures/src/style.py`). Include them
  as `\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/<name>}\caption{…}\label{fig:<name>}\end{figure}`.
  Captions go below figures and explain what the reader sees. The figure list and what each one shows is in the
  workflow prompt; another agent draws them in parallel, so describe exactly what the spec says.
- Commutative and string diagrams: draw inline with `tikz-cd` / TikZ (colours `tide`, `ochre`, `seal`, `moss`, `inkgray`).
- Tables: booktabs, `\headrow` after `\toprule`, bold header cells, caption above, `\small` or `\footnotesize`.
  Generated tables: `\input{tables/periodic}`, `\input{tables/catalogue}`, `\input{tables/arrows}`, `\input{tables/obstructions}`;
  counts as macros from `tables/counts.tex` (`\nMethods \nPapers \nHF \nCore \nArrows \nObstructions \nPropositions`).

## Voice
Plain, precise, confident and warm, as a mathematician writing for ML researchers. Short paragraphs. No hype, no
rhetorical questions, minimal em-dashes, no "not X, but Y", no colon-then-reveal. State limits as conditions next to the
claim ("for exact Cayley maps", "when W_0 is injective"), not as apologies. Mark analogies as analogies.
