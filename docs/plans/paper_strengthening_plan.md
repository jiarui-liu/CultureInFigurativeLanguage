# Paper-Strengthening Plan (toward ACL/EMNLP)

**Goal:** turn the CPT-only study into an acceptable ACL/EMNLP paper without doing
instruction tuning. Reframe around one well-ablated question + shore up rigor.
Owner: automated run. Started 2026-08-24.

## Reframe (story)
Drop the "two-stage pipeline" framing (IT stage is untested → Future Work only).
Center on: *does meaning-tagged CPT inject figurative/cultural knowledge, and how
much is the idiom curation vs. generic in-language exposure?* across 3 languages
(hi/zh/ar), each with a matched-size **unfiltered** control, and now a
**filtered-untagged** control (point 6) that decomposes the gap into
document-selection vs. meaning-tag effects.

## Work items (this plan executes 1–6)

**1. Significance / bootstrap CIs (no retraining).** Every eval task stores
per-item `records` (gold/pred/correct/logprobs) under
`/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/{hi,zh,ar}/{base,cpt,unfiltered,untagged}/<task>.json`.
Compute per-run acc with item-level bootstrap 95% CIs, and paired tests
(CPT vs base, CPT vs unfiltered, CPT vs untagged) via paired bootstrap +
McNemar/Wilcoxon on per-item correctness. Script:
`src/culture/evaluation/compute_cis.py` → writes `eval/<lang>/ci_report.json` + a
LaTeX-ready table. Mark every delta significant / n.s.

**2. Related Work section.** Add to the paper: continued pretraining / domain-
adaptive pretraining; knowledge injection & entity/definition augmentation;
multilingual & low-resource cultural adaptation; cultural/figurative benchmarks.
Position the contribution as the *controlled multilingual measurement* of when
meaning-tag CPT helps.

**3. Knowledge-injection vs. comprehension analysis (RQ3).** Split idiom tasks
into (a) answer-string-injected / meaning-lookup (kinayat_meaning, Chengyu-Bench
connotation) vs. (b) clean cloze/comprehension (kinayat_cloze, ChID, MABL) vs.
(c) generalization to unseen idioms (ar_figurative). Show the curation wins on
(a)+(c) but ties on (b). Frame as a headline finding, backed by point-6.

**4. Fill eval gap: Arabic English-retention.** Run the language-agnostic English
retention suite (MMLU/BoolQ/GSM8K/HumanEval/WikiText) on ar base + ar-cpt +
ar-unfiltered so all three languages report Dim-1 symmetrically.

**5. Qualitative examples + error analysis.** A tagged-document example per
language; 2–3 win/loss cases per idiom task from the stored `records`.

**6. Method baseline = filtered-untagged CPT, ALL THREE LANGUAGES.** A 4th
variant per language: the SAME idiom-bearing filtered documents as the augmented
run but with the appended meaning block STRIPPED. Isolates the meaning-tag effect
(augmented vs filtered-untagged) from the document-selection effect
(filtered-untagged vs unfiltered).
- Build by stripping tags from the existing tagged shards (no re-filtering):
  hi via `text[:original_text_chars]`; zh split on `\n\n【成语注释】`; ar split on
  `\n\nالمعاني الاصطلاحية للتعابير الواردة في النص:`.
- Token-match to the augmented run by training with `max_steps` = augmented
  `global_step` (hi 2100 / zh 11157 / ar 1608), same recipe (packing cycles the
  corpus so max_steps fixes the token budget exactly).
- Then eval the new checkpoint on the target + control benchmarks and fold into
  the CI + analysis.

## Parallelization
- Strip jobs (3 langs) run concurrently on cpu,all.
- CPT untagged runs (3 langs) submitted concurrently (afterok strip), 4 nodes each.
- Arabic English-retention eval (point 4) runs immediately (independent).
- Point-1 CIs computed now on existing base/cpt/unfiltered; re-run when untagged evals land.
- Points 2/3/5 (writing) drafted from existing data; numbers refreshed after point-6.

## Compute note
zh untagged = 11,157 steps (~30h, 4 nodes); hi/ar ~5h each. Monitored by cron.
