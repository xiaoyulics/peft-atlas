# arXiv submission sheet: A Categorical Atlas of Parameter-Efficient Fine-Tuning

## Files
- Upload: `categorical-atlas-peft-arxiv.tar.gz` (source, `main.bbl`, template, figures, and `anc/`).
  It compiles the way arXiv compiles it (pdfLaTeX three times, no BibTeX), giving 402 pages with no errors and no
  undefined references. arXiv treats the top-level `anc/` folder as ancillary files.
- Reference PDF: `categorical-atlas-peft.pdf`.

## Metadata (copy into the form)
- **Title:** A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges
- **Authors:** Xiaoyu Li (University of New South Wales), Zhizhou Sha (University of Texas at Austin), Chiwun Yang (City University of Hong Kong), Dai Shi (University of Cambridge)
- **Abstract** (plain text, 1870 characters, under the 1,920 limit):

Parameter-efficient fine-tuning has hundreds of methods, usually compared by parameter count and benchmark score. We compare them as maps. A pretrained checkpoint is a base point in weight space, and a weight-space method is a smooth map from a small parameter space that starts at that point. These maps form a slice category in which the frozen model is initial and full fine-tuning terminal. Three invariants of the map do most of the work. Its image near the base point governs expressivity, through the first-order image, the dimension, a min-cut bound on rank and whether that point is smooth or an apex. Its fibres, with the optimizer's metric, govern training dynamics. Base change moves the base point and covers splitting initialisations, quantisation, restarts and merging. We prove, for example, that zero-initialised LoRA, and Householder reflection adaptation at its default start under stated rank and parity conditions, begin at the apex of a cone; that no exact initialisation lets LoRA reach an update of rank above twice its own; that exactly orthogonal methods never change a weight's singular values in exact arithmetic, however often they are merged; that under Adam with no epsilon, weight decay or clipping and one shared schedule, LoRA+ from the standard zero initialisation follows exactly the trajectory of plain LoRA with its scale multiplied by the learning-rate ratio; that Adam with shared hyperparameters respects only the signed permutations inside LoRA's continuous symmetry group of full-rank factors; and that the activation function decides when a scaling adapter folds into preceding weights. We classify 382 methods from 138 papers by seven coordinates read off their formulas, with 403 simulation arrows and 485 obstructions. Every numbered result is proved in full, and results due to others are credited where we re-derive them.

- **Comments:** Technical report. 402 pages, 17 figures, 41 tables. Interactive companion: https://xiaoyulics.com/peft-atlas/. Ancillary files: the catalogue of 382 PEFT methods with seven coordinates, 403 simulation arrows with explicit maps, 485 obstructions, every numbered result with its proof and review status, and the numerical checks.
- **Primary category:** cs.LG (Machine Learning).
- **Cross-lists (suggested):** math.CT (Category Theory), stat.ML (Machine Learning); optionally cs.CL.
- **MSC class (optional):** 68T07; 18M05; 53C30; 22E70.
- **ACM class (optional):** I.2.6; G.1.6.
- **License:** your choice. CC BY 4.0 maximises reuse of the figures and catalogue.

## Before you press Submit
1. Authors, affiliations and emails: done (2026-10-03).
2. Companion URL: done (2026-10-06, https://xiaoyulics.com/peft-atlas/).
3. Write the acknowledgements in `sections/ack.tex`. The current text is a placeholder. AI use is documented in Appendix F, so no separate statement is needed.
4. Re-run `tools/make_arxiv.sh`, which rebuilds the PDF, the `.bbl` and the tarball, and tests a clean compile.
5. Check arXiv's current policy for the CS category. Since late 2025, arXiv CS has asked that review articles and
   position papers be peer reviewed before they are posted. This paper is original research (new theorems with full
   proofs), and its abstract and Thesis present it that way. The catalogue is a supporting artefact.
