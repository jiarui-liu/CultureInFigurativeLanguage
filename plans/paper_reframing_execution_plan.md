# Paper reframing and execution plan

## Objective

Rewrite the paper around one precise question: **What does a language model learn from a culture's idioms?** The answer will distinguish three levels of acquisition:

1. **Idiom meaning:** whether the model learns the meanings of expressions encountered during continued pretraining.
2. **Figurative generalization:** whether learning transfers to figurative expressions absent from the knowledge base or training annotations.
3. **Cultural symbolism:** whether the model learns culture-specific associations between concrete entities and abstract meanings, rather than defaulting to English associations.

The paper will make only claims supported by comparisons against the token-matched Random-CPT control. Results against the untrained base model remain diagnostic. Broad culture benchmarks will be used to establish construct boundaries: they mostly test named facts and practices, while idioms predominantly express symbolic and evaluative knowledge.

## Target story

The paper has a single arc:

1. Idioms pair abstract meanings with concrete imagery.
2. IdiomAtlas makes those pairings measurable across English, Chinese, Hindi, and Arabic.
3. The pairings are culture-specific in both directions: shared entities carry different associations, and shared meanings use different entities.
4. Controlled continued pretraining shows what models acquire from idiom-bearing text: meanings of encountered idioms, some transfer to unseen figurative language, and less English-centered cultural symbolism.
5. The source of learning matters. Running text supports transfer; dictionary glosses strongly teach listed meanings but can narrow generalization.
6. Existing culture benchmarks cover the relevant symbolic layer poorly, motivating the new symbolism probe and explaining why factual benchmark suites under-measure what idioms teach.

The title, abstract, introduction, results, discussion, conclusion, figures, and tables must all instantiate this same arc. The paper will not use a failed broad-transfer hypothesis as its written research question.

## Evidence standard

- Primary contrasts are paired, token-matched Idiom-CPT versus Random-CPT comparisons.
- Separate exposure from annotation using idiom-bearing text without glosses.
- Preserve the material adverse result that gloss-appended training lowers unseen Arabic idiom accuracy; present it as evidence about the mechanism and a design lesson.
- Do not manufacture cross-language symmetry. For findings currently supported in only one language, diagnose differences in benchmark coverage, exposure, power, headroom, contamination, and response format; run principled analyses or experiments where feasible.
- Fix the analysis decision before inspecting a new result whenever a new metric, split, or prompt is introduced. Record all attempted cross-language extensions in a research log, even if only the supported findings enter the main narrative.
- Do not claim a language-wide effect from one benchmark. State the evaluated construct and dataset.

## Phase 1: audit and claim ledger

- Inventory all paper sources, result tables, experiment scripts, logs, checkpoints, and generated artifacts.
- Build a claim ledger with, for every headline number: source file, exact contrast, sample size, uncertainty/significance, training condition, and whether it is seen or held out.
- Recompute headline deltas from raw predictions where available.
- Check consistency of every number across abstract, main text, tables, figures, appendix, and conclusion.
- Identify text-heavy or overwide tables. Apply `\\resizebox` to any retained table that exceeds the column width.

Deliverables: `plans/paper_claim_ledger.md`, `plans/research_log.md`, and reproducible validation output.

## Phase 2: cross-language diagnosis and targeted research

### Figurative generalization

- Verify the Arabic AR-Figurative overlap calculation and the no-gloss/gloss contrasts.
- Locate genuinely held-out Hindi and Chinese figurative benchmarks or construct matched held-out splits from existing data without reusing training idioms.
- Match evaluation format and scoring across languages where data permits.
- Quantify exposure, baseline headroom, item count, and confidence intervals before interpreting differences.

### Cultural symbolism

- Audit symbolism-probe generation, held-out criteria, answer ordering, English-association definition, and per-language sample sizes.
- Recompute both accuracy and English-association selection rates with paired uncertainty tests.
- Test whether Chinese and Arabic results change under predeclared high-confidence, adequately exposed, and culture-divergent subsets.
- Check whether the no-idiom culture-text ablation and symbolism-statement experiment isolate idiom content or introduce a selection confound.
- If compute and checkpoints are available, run the smallest decisive extension before launching new large training runs.

### Retention and cost

- Compare English capability and target-language language modeling losses against Random-CPT, with uncertainty where predictions exist.
- Support a “no additional capability cost over matched in-language training” claim only at the tested scales and tasks.

Deliverables: scripts/configs, raw results, statistical summaries, and an updated claim ledger.

## Phase 3: visual redesign

Prefer figures in the main paper for relationships and mechanisms:

1. **Conceptual overview:** idiom → literal imagery/entity → figurative meaning → culturally specific association; then training conditions and three evaluated outcomes.
2. **Cross-cultural mapping figure:** same entity/different meanings and same meaning/different entities, with one memorable multilingual example and aggregate statistics.
3. **Learning figure:** Random-CPT-referenced gains for seen meaning, unseen figurative inference, culture benchmark, and symbolism bias, showing text-only and gloss-appended conditions separately.
4. **Mechanism figure:** training signal versus outcome, emphasizing that running context supports generalization while glosses concentrate gains on listed idioms.
5. **Measurement-coverage figure:** composition of idiom tasks, symbolism probes, and existing culture benchmarks by knowledge type.

Keep compact numeric tables for dataset statistics and exact results. Move exhaustive benchmark grids, robustness checks, and secondary conditions to the appendix.

## Phase 4: full rewrite

- Rewrite title and abstract around acquisition from idioms.
- Rewrite the introduction with one research question and three testable outcomes.
- Tighten related work around figurative language as cultural knowledge, culturally grounded evaluation, and continued pretraining.
- Reorganize data/methods so every annotation field has a role in a later analysis or experiment.
- Present controlled training and evaluation before secondary diagnostics.
- Write results in claim–evidence–interpretation order.
- Use the benchmark content audit to define measurement scope, without making broad benchmark nulls the headline.
- Rewrite the conclusion to answer the central question directly.
- Keep limitations accurate and proportional; remove argumentative repetition and discussion better placed in methods or appendix.
- Edit the appendix for correspondence with the rewritten main paper.

## Phase 5: verification

- Compile from a clean LaTeX build and resolve errors, missing references, overfull boxes, and unreadable figures.
- Run automated checks for inconsistent headline numbers, undefined references/citations, TODOs, duplicated prose, malformed LaTeX, and tables wider than their target.
- Inspect the rendered PDF page by page for figure legibility, float placement, spacing, and narrative continuity.
- Perform a final reviewer-style audit: novelty, construct validity, control validity, statistical support, reproducibility, and claim calibration.

## Completion criteria

- The full paper, including appendices, compiles cleanly.
- Every headline claim is traceable to a validated result and token-matched control.
- The abstract, introduction, figures, results, and conclusion tell the same three-level acquisition story.
- Main-text displays emphasize interpretable figures; exact values remain accessible in compact tables or appendices.
- Cross-language asymmetries are either explained by evidence or stated at the level actually supported.
- No material result needed to interpret a headline claim is concealed.

## Completion record (2026-10-11)

- All five phases are complete. The claim ledger links every headline result to a generated artifact.
- The cross-language symbolism analysis and independent evidence-only audit are reproducible from `src/culture/analysis/v3/`.
- The main paper contains four figures and one compact dataset table; full numerical grids are in the appendix.
- The paper builds from a clean LaTeX state to 39 pages. The conclusion begins on main-text page 8, the appendix begins after references on page 12, and the final log contains no undefined references, undefined citations, overfull boxes, or LaTeX errors.
- `paper_writing/code/validate_paper_claims.py` passes 17 numerical invariants and nine headline-text checks.
- The rendered PDF was inspected page by page, including the two new result figures at print resolution.
