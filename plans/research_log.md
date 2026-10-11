# Research log for the idiom-acquisition reframing

## 2026-10-11

### Repository and evidence audit

- The code and analysis repository is `CultureInFigurativeLanguage`; the rendered paper lives in the sibling `OverleafCultureInFigurativeLanguage` repository.
- Both worktrees were clean before this pass. The only initial change was the new execution plan.
- The current paper is organized around bidirectional transfer between idioms and broad cultural knowledge. Its strongest evidence instead supports a three-level acquisition story: encountered idiom meanings, figurative generalization from running text, and culture-specific symbolism.
- The existing experiment archive is substantially richer than the current headline story. It includes token-matched Random-CPT controls, idiom documents with and without glosses, 2B and 9B runs, instruction tuning, per-item paired statistics, a three-language symbolism probe, taxonomy labels for all benchmark items, training dynamics, and targeted symbolism-tag follow-ups in Hindi and Arabic.

### Writing guidance applied

Read the full guidance in `/storage/home/jiaruiliu/local/git-repos/paper-writing-experience`. The rewrite will use one research question, claim-first paragraphs, a small set of reader-facing terms, generated numeric artifacts, compact tables, and figures for relationships. Plot scripts and backing data will stay in this code repository; the Overleaf repository will receive only rendered PDFs and LaTeX sources that are actually imported.

### Existing follow-ups relevant to cross-language asymmetry

- The symbolism probe already covers Arabic (98 items), Hindi (155), and Chinese (206). Against Random-CPT, Hindi idiom-bearing text reduces selection of the English association by 8.4 to 10.3 points. Chinese Idiom-CPT moves in the same direction by 3.4 points, with paired `p=0.092`. Arabic is flat.
- A controlled 0.63B-token symbolism-statement experiment raises the held-out Hindi symbolism probe by 9.0 points and its decision margin by 0.288, while meaning tags do not. The same recipe is flat in Arabic even when 82 of 98 probe entities are explicitly described.
- Existing diagnostics identify two plausible causes of the asymmetry. Hindi has more usable headroom and 75% of its documents receive a symbolism statement, compared with 44% in Arabic. The Chinese arm has not received the matched symbolism-statement intervention.
- The content taxonomy gives a cross-language result broader than whole-benchmark accuracy. Idiom-CPT preferentially helps symbolic or language-form items within culture benchmarks in Arabic and Hindi. Chinese remains underpowered on this slice and its tags contain classical citations that favor allusion recall.

### Next actions

1. Recompute the canonical headline contrasts from saved prediction records and regenerate the existing statistical artifacts.
2. Audit the symbolism probe at item level for power, answer-order bias, exposure, and confidence-qualified subsets fixed independently of model outcomes.
3. Decide whether a matched Chinese symbolism-statement run and a dose-matched Arabic run are necessary and feasible after the audit.
4. Rewrite the paper while any new cluster jobs run, using only validated existing claims until new results land.

### Completed cross-language analysis

- Recomputed the symbolism comparison from per-item predictions with language-stratified paired bootstraps. Across 459 items, Idiom+Meaning reduces English-association choices by 4.36 points (95% CI 1.53--7.19 in the favorable direction, language-weighted CI 0.90--6.92) and raises the target-versus-English log-probability margin by 0.185 (95% CI 0.102--0.268). Both micro and macro estimates exclude zero.
- Audited all 426 auditable symbolism items with GPT-5.4-mini using only the evidence idioms and answer options, never evaluated-model outputs. Cross-model consensus retains 376 items: 88 Arabic, 138 Hindi, and 150 Chinese. On this independently fixed subset, the English-lure change is 0.0 points in Arabic, -10.9 in Hindi, and -4.0 in Chinese.
- The new pooled analysis supplies a supported cross-language result without searching outcome-conditioned subsets. Existing targeted runs already diagnose the remaining per-language difference: the explicit-symbolism intervention transfers to held-out Hindi entities, while its Arabic dose and headroom differ. A new large training run would not be the smallest decisive test and was not launched.

### Paper and artifact rewrite

- Reframed the paper around one question and three acquisition levels: encountered meaning, figurative transfer, and cultural symbolism.
- Added generated main-text figures for acquisition and symbolism. Their source code and numeric artifacts live in this repository; the Overleaf repository imports only the rendered PDFs.
- Replaced the broad-transfer result grid in the main paper with two visual summaries and moved exact benchmark values to the appendix.
- Renamed the reader-facing conditions to Random-CPT, Idiom-Text, and Idiom+Meaning throughout the paper.
- Preserved the material Arabic unseen-idiom result: appended meanings improve listed idioms but reduce unseen accuracy by 4.0 points against Random-CPT and 7.0 against the same running text without meanings.
- Rewrote the abstract, introduction, related work, data, analysis, experiments, conclusion, limitations, and appendix framing. Overwide result tables use `\\resizebox`; prompt listings wrap long lines.
