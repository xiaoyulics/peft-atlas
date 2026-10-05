# Classification consolidation report (2026-10-01)

## Totals
- **Methods classified:** 320 of 320 in the catalog.
- **Core methods:** 139. That is every method named in the Exposé VIII periodic table plus 46 others.
- **Arrows:** 334 in `edges.json`, deduplicated (A3 263, A2 58, A5 11, A4 2).
- **Obstructions:** 395, deduplicated (O1 231, O2 110, O3 54).
- **Targets:** every arrow and obstruction target is a catalog id, and every rule and test is a valid key.

## Counts per axis value
Counts are all methods / core methods. "–" means `null`: the axis does not apply, as in the periodic table's "–".
- **kind:** R 177/82, E 31/10, E0 27/13, B 23/6, P 24/11, F_T 38/17
- **shape:** Lin 90/45, Cone 112/43, Cone* 13/8, Orb 10/6, Sat 5/3, Meet 1/1, Fun 82/32, – 7/1
- **base_point:** regular 93/54, apex 96/39, neutral 32/13, defect 45/13, unpointed 52/18, – 2/2
- **gauge:** none 118/61, scalar 8/6, torus 36/9, GL 123/49, GLxTorus 2/1, nonlinear 19/10, – 14/3
- **covariance:** GLxGL 138/53, OxO 37/18, GLxCO 7/4, MonxGL 20/10, MonxMon 44/18, Pi 42/19, frame 16/10, – 16/7
- **merge:** M1 201/95, M1-> 10/5, Mq 8/2, Minf 84/32, na 17/5
- **rebasing:** idempotent 80/42, span 109/44, group 11/6, schedule 22/9, – 98/38

## Fixes made in the batch files
Every edit is logged in the method's `check_notes` under "Consolidation (2026-10-01)"; 135 methods were touched. A backup of the original batches is in the scratchpad at `classified-backup/`.

1. **Invalid `"n/a"` values:** 137 coordinates used `"n/a"`, which is not a value in `axes.json`; the merge axis has its own `na`.
   - 136 became `null`, matching the table's "–". The site already handles null.
   - GenKnowSub's base point became `unpointed`, by the axis definition and to match the other parameterless P operations.
   - The 7 methods with a null shape now have `table_col` "–".
2. **`table_row` labels:** batch 1's `R`, `R (Type II)`, `R (Type I)` and `R at κθ0` became `R-additive` (31 methods), and Lily's `F_X` became `F_T`.
3. **LISA and BAdam covariance changed from Pi to GLxGL.** This matches HiFT, surgical FT and BitFit. Each stage frees whole boxes, and the layer partition breaks no neuron permutation. HiFT's note was updated to match.
4. **Four obstruction reasons now state which instance they mean:** DSEE→LoRA, GaLore→LoRA, Spectral Adapter^A→LoRA, MoV→LoRA.

## Contradiction checks: no genuine contradictions remain
- **Same pair with both an arrow and an obstruction (22 cases):** each pair concerns different instances of the target: a different rank, pointing, hyperparameter regime or variant. Both are kept.
- **Obstruction versus a path of arrows (16 cases):** all qualified by rank or frame; the path ends in a higher-rank LoRA than the one the obstruction is about.
- **Self-loops (8):** all are inclusions between instances of one family, such as LoRA_r ↪ LoRA_{r+1} and ReLoRA^K ↪ ReLoRA^{K+1}. They are kept.
- **Cycles (13 groups):**
  - Genuine isomorphisms: DoRA≅BiDoRA, ReLoRA≅COLA, AsyLoRA≅LoRA-FA, AdaptFormer≅Parallel adapter, S-LoRA≅Punica, EFT≅Proxy-Tuning, LoRAMoE≅MoLoRA, AdaMerging≅CAT (differs only in pointing).
  - Equal images but not isomorphic: prompt tuning with MPT, prompt tuning with P-tuning, QLoRA with IR-QLoRA.
  - Cycles that only go up in rank or stage count, so they are not isomorphisms: LoRA↔ReLoRA, LoRA↔PEANuT, LoRA-FA↔Flora, surgical FT↔HiFT, KronA↔LoKr.
  - Initial-state tuning ↔ prefix tuning holds on S4 state-space layers only.
- **Arrows against O1/O2/O3 using coordinates:** none violates an obstruction once its stated pointing or rank is taken into account.

## Unresolved doubts
1. **Periodic table and `axes.json` need edits.** The batch values below were kept because the formulas support them, so `framework.md` and `axes.json` now disagree with the data:

| Method | Table / axes.json says | Classified as |
|---|---|---|
| LST | E | E0, unpointed |
| UniPELT, MAM | E | E0 |
| MPT | gauge R^× | torus (when P* is trainable) |
| Compacter | gauge GL on PHM index | torus |
| Houlsby adapter | ReLU torus | none (uses GELU/swish) |
| DeLoRA, AdaLoRA | torus | nonlinear |
| LoRETTA | apex | Cone*, regular |
| LoRA-SB | CO×CO | MonxMon |
| Trainable Tokens | Mon×Mon | MonxGL |
| VB-LoRA | fr | Pi |
| LISA | Π (blocks) | GLxGL |
| HyperFormer | merge M1 | Minf |
| S-LoRA, Punica | "unmerged by design" | M1 |

   Some table entries have no matching key, so they were coded to the nearest one: OLoRA CO×Borel→Pi, LoReFT/PoLAR O(r)→GL, OFT "stabiliser"→none.
2. **Null versus explicit "not applicable".** If you want a filterable value instead of null, add an `na` value to shape, base_point, gauge, covariance and rebasing in `axes.json`.
3. **Covariance of function-level methods is split.** About 66 carry a function-level value, while 16 routed families and P compositions are "–" (LoRAMoE, MixLoRA, HydraLoRA, ICEdit, MoV, Lily, MoLE, X-LoRA, PHATGOOSE, Arrow, LoraRetriever, LoRA-Flow, SMEAR, GOAT, AdapterSoup, LoRA Switch). One convention should be chosen.
4. **Instance qualifiers live only in free text.** The `edges.json` schema has no rank or pointing fields, so an id-level checker will flag the 22 + 16 + 8 cases above. Optional instance fields would fix this.
5. **Some ids cover more than one variant:**
   - `mov` covers both MoV and MoLoRA.
   - `child-tuning` covers both its D and F variants.
   - The Concept Sliders → Task Arithmetic arrow starts from the deployed object, not the classified training object.
6. **Two extensions break the rebasing convention.** Initial-state tuning and state-offset tuning are Minf yet have rebasing `idempotent`; every other non-mergeable extension has null.
7. **P operations treat coefficients unevenly.** Task Arithmetic, TIES and DARE count λ as a trainable coordinate; Fisher Merging, RegMean and Model Soups are treated as having no coefficient family.
8. **`core` has no written definition.** Nine HF PEFT methods are not core.
9. **Obstruction contents were not re-derived here.** I checked them only against arrows and coordinates.

Files:
- /Users/z5236444/Documents/范畴论_PEFT/research/classified/batch-0.json … batch-9.json
- /Users/z5236444/Documents/范畴论_PEFT/research/classified/edges.json
- /Users/z5236444/Documents/范畴论_PEFT/site/data/atlas-data.js
- /Users/z5236444/Documents/范畴论_PEFT/research/classified/report.md (not written)