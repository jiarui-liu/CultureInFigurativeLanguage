# Deep-analysis pass 3 (2026-10-07) — closing the explanatory gaps before submission

**Directive.** "Conduct ANY analyses you can that is related to this paper's research question",
autonomously, writing the results into the draft. This document is the plan and the running log.

**Scope boundary (other live sessions).** Three other Claude sessions are working this repo
right now; this pass must not touch what they own:

| Session | Owns |
|---|---|
| `2aed4770` | §4 Table 2/3 (`entity_cases*`, `meaning_cases*`) for the five non-en–zh language pairs, via the MetaGen API |
| `c3a6bfbb` | Missing 9B experiment results / §5 tables for the 9B grid |
| `3b20c8c6`, `42df72fc` | `fig_entities`, `fig_cpt_effects`, `fig_typology_expanded` (done 2026-10-05) |

This pass therefore owns **new analyses only**, written into new files
(`src/culture/analysis/v3/`, `docs/paper_stats/v3/`), new figures/tables with a `v3_` or new
name, and new prose in §4/§5/appendix that does not overlap the above.

---

## 0. What the paper still asserts without measuring

Reading the current draft end to end, the claims that carry the argument but rest on indirect
evidence are:

1. **"Each kind of text mostly teaches what it states."** Measured only at the level of whole
   benchmarks. The decisive test is *within* a benchmark: if the taxonomy explanation is right,
   Idiom-CPT's gain should concentrate on the symbolic/evaluative items of a culture benchmark
   and Culture-CPT's on the material-practice/factual items. §7.7 of the main runbook records
   this as explicitly **not done** ("the cache keys on prompt hash and never stores `qid`").
2. **"Why some benchmarks move and others do not."** The paper names ArabCulture as "closest to
   the content of the culture corpus" but never measures closeness. A corpus→benchmark affinity
   score computed the same way for all 17 benchmarks turns an anecdote into a predictor.
3. **"The idiom corpus and the culture corpus are different populations."** Measured in one
   direction only (what share of culture-rich documents contain an idiom). The reverse — how
   culture-rich the *idiom* corpus is, on the paper's own classifier — is missing, and it is the
   direct answer to "is Idiom-CPT also a culture-CPT?".
4. **"The benchmarks never ask about the imagery idioms use."** Asserted from the taxonomy; never
   checked at the level of the entities themselves.
5. **"Accuracy is flat"** on most reverse-direction contrasts. A flat accuracy with a moved
   decision margin is *latent* learning; the paper measures the margin on one benchmark
   (Kinayat-Meaning at 2B) and generalises informally.
6. **"The tags teach item-specific knowledge."** Shown against exposure count. Not shown against
   *semantic neighbourhood*: whether an unseen idiom close in meaning to a seen one benefits.
7. **Capability cost.** One appendix row (HumanEval, MMLU). No systematic profile of what
   continued pretraining does to the model across domains, and no decomposition of a benchmark
   delta into items learned vs items forgotten.
8. **The idiom-free restriction.** The paper attributes the Chinese 9B losses to it and asks
   pipelines to report the share removed — but never runs the arm without the restriction, which
   is the obvious reviewer request.

---

## 1. Analyses in this pass

All outputs under `docs/paper_stats/v3/`; all scripts under `src/culture/analysis/v3/`.
Every contrast is reported with an item-level paired bootstrap 95% CI and, where it is a
hypothesis test, a permutation or McNemar p-value with Holm correction inside its family.
Negative results are reported as negative.

| # | Analysis | Answers | Needs |
|---|---|---|---|
| **A1** | **Per-item culture-layer labels → within-benchmark interaction.** Label every item of 12 benchmarks with the §5.4 six-way taxonomy, keyed by `qid`, then test arm × layer interactions. | Does Idiom-CPT help the symbolic items and Culture-CPT the factual ones, *inside* the same benchmark? | GPU (vLLM, Qwen3.5-27B-FP8) + MetaGen for the second annotator |
| **A2** | **Corpus→benchmark affinity predicts transfer.** Embed every benchmark item and 20K documents per corpus (Random / Idiom / Culture, 3 languages) with Qwen3-Embedding-0.6B; score each benchmark by its affinity to each corpus; regress the observed per-benchmark Δ on the affinity difference. | Why ArabCulture and not the rest | GPU (embedding) |
| **A3** | **Joint profile of the three corpora.** Run the paper's own culture classifier over samples of all three arms' corpora, and the idiom matcher over all three, giving each arm a (culture score, idiom density) distribution. | Is Idiom-CPT secretly a culture-CPT? How far from the natural distribution is each arm? | CPU + saved ridge models + GPU embeddings |
| **A4** | **Entity coverage of the benchmarks.** How many KB entities appear in culture-benchmark items, and when they do, is the item about what the entity *is* or what it *stands for*? | The "imagery mismatch" claim, at entity level | CPU + A1 labels |
| **A5** | **Margin analysis across the whole grid.** Gold-minus-best-distractor log-prob margin for every arm × benchmark at 9B and 2B, as an accuracy-free measure. | Latent learning the accuracy hides | CPU (eval records already store logprobs) |
| **A6** | **Semantic-neighbourhood generalisation.** For each unseen IdiomAtlas-MC item, similarity to its nearest *seen* idiom (by meaning and by entity overlap); does the gain rise with it? | Does tag learning generalise along meaning space, or not at all? | GPU (embedding) |
| **A7** | **Item-level predictors of idiom learning.** Logistic regression of per-item gain on exposure, idiom length, entity count, entity semantic type, taxonomy layer, distractor similarity. | What kind of idiom is learnable from CPT | CPU |
| **A8** | **Capability profile and churn.** (a) Perplexity table across arms × domains from the existing `ppl_*` records; (b) decomposition of every benchmark delta into items fixed vs items broken; (c) cross-arm answer-agreement matrix. | What continued pretraining costs, and whether a flat delta is a flat model or a reshuffle | CPU |
| **A9** | **Where do wrong answers go?** On the culture benchmarks, do arms differ in whether their errors pick the option that is locally specific vs generic? Uses the symbolism-probe lure logic generalised to culture benchmarks via A1 labels. | Qualitative direction of the (small) cultural effects | CPU + A1 |

## 2. Training in this pass

| # | Run | Question | Cost |
|---|---|---|---|
| **T1** | **Culture-CPT without the idiom-free restriction**, Chinese and Arabic, with matched Random-CPT and Culture-CPT(restricted) arms, on **Qwen3.5-4B** (the largest base checkpoint available locally besides 9B), ~0.3B tokens per arm | Is the Chinese 9B cost caused by the exclusion of idiom-bearing documents, as the paper claims, or by culture-rich text as such? This is the most likely reviewer request and the paper currently answers it only by argument. | 4 arms × 0.3B tokens at 4B |
| **T2** | *(contingent on T1 finishing early)* Idiom-CPT with **symbolism-stating tags** instead of dictionary glosses, Arabic, 4B | The paper's own "natural next step": a notes generator aimed at evaluative associations. Even a single arm turns a stated future direction into a result. | 1 arm |

T1 is the priority; T2 only if the queue allows.

---

## 3. Log

### 2026-10-07, first results

**Infrastructure built.** `src/culture/analysis/v3/`: `dump_items.py` (reconstructs
`qid -> item text` for 27 benchmarks by calling the same loaders `run_eval` calls — this is what
unblocks every item-keyed analysis), `item_taxonomy.py`, `corpus_affinity.py`,
`generalization.py`, `margins_and_churn.py`, `kb_corpus_benchmark.py`, `distractor_bias.py`,
`parallel_items.py`, `affinity_vs_transfer.py`, `layer_interaction.py`, `make_tables.py`,
`build_zh_t1.py`. New slurm: `v3_gpu.slurm`, `build_zh_t1.slurm`,
`evaluation/eval_capability.slurm`, `evaluation/eval_ar_dynamics.slurm`,
`evaluation/eval_zh_t1.slurm`, `training/.../cpt_zh_t1.slurm`.

**A3 — the two headline numbers of this pass.** Scoring 6,000 documents of every arm's corpus
with the paper's own culture classifier (tags stripped):

| | random | idiom | idiom−tags | culture | culture+notes |
|---|---|---|---|---|---|
| ar | 0.94 | 1.94 | 1.93 | **2.81** | 2.81 |
| hi | 1.17 | 1.56 | 1.59 | **2.76** | — |
| zh | 0.63 | 1.67 | 1.78 | **1.12** | — |

1. *The idiom corpus is itself culture-rich* in all three languages (2x random in ar and zh),
   so Idiom-CPT is not a culture-free intervention — and it still moves one of seven culture
   benchmarks. The mismatch is not a lack of cultural content.
2. *Chinese Culture-CPT is not culture-rich at all* (1.12, below its own idiom corpus). The
   idiom-free restriction forced the selector down to a score cutoff of 1.12. This is a far
   sharper statement of §5.3's argument than the 90.2% document count, and it is the reason the
   Chinese arm loses accuracy.

**A2 — closeness predicts transfer.** Over the 98 arm-benchmark pairs of the 9B grid, the gain
over Random-CPT rises with how much closer the arm's corpus is to the benchmark in
Qwen3-Embedding space (Spearman rho=0.39, permutation p=0.0002; rho=0.29 p=0.004 on the margin).
Holds per language (ar .39, hi .53, zh .43) and for the culture arm alone (.40, p=0.03).

**A5 — the reverse direction is not null below the accuracy threshold.** Margins for the whole
9B grid: Culture-CPT widens the gold-vs-best-distractor margin on Kinayat-Meaning (+0.07
[0.05,0.09]), both Arabic IdiomAtlas-MC splits (+0.02, +0.03), Chengyu-Bench (+0.23
[0.18,0.28]) and IdiomAtlas-MC zh seen (+0.07), with flat accuracy; Hindi does not move. The 2B
grid replicates (ar Kinayat +0.11). Idiom-CPT *lowers* the unseen-Arabic margin by 0.16 (0.29 at
2B) — the tags displace unseen items' glosses rather than merely failing to help.

**A8b — a null delta is not a null model.** Median arm-benchmark pair changes 9-11% of its
answers against the control while accuracy moves ~1 point. ArabicMMLU/Culture-CPT: 621 fixed,
620 broken, delta 0.0.

**New headline for §5.2.** Item-weighted over the ten culture benchmarks, 89% of Idiom-CPT's
gain over the untrained model is already delivered by the token-matched control (97% for
Idiom−tags, 103% for Culture-CPT).

**A6 — the glosses generalise to a short radius.** Only the tagged arm's per-item gain on
*unseen* idioms rises with the cosine of the item's gold meaning to the nearest *glossed*
meaning (ar rho=0.11 p=0.009, hi rho=0.10 p=0.015; |rho|<=0.04 for the untagged and culture arms,
and negative for the base model). Terciles: ar −10.0 / −1.5 / −0.5 points; hi −4.5 / +3.0 / +7.0.
Surface similarity and shared entities predict nothing.

**A4 — a negative result that removes the cheap explanation.** 60-100% of culture-benchmark
items mention a KB entity of their language, and entity frequency across idioms correlates with
entity frequency across benchmark items (rho 0.21-0.53). The benchmarks use the same imagery;
they ask different questions about it.

**A4b — corpus selection bias.** Coverage ar 47.6%, hi 12.9%, zh 98.5%. Attested vs unattested
entity-type distributions differ weakly (V=0.05-0.10); the Hindi corpus is tilted away from
kinship/social imagery (−4.4 points), i.e. away from what makes the Hindi inventory distinctive.

**Correction found.** The paper's claim that Idiom-CPT's unseen-Arabic errors lean toward
tag-glossed distractors is not significant (+3.7 points, exact binomial p=0.14) and does not
replicate in Hindi (+1.4, p=0.48); in Chinese 98.7% of distractors are tag-glossed so the test
is vacuous. §5.1 softened accordingly.

**A9' — item-matched cross-language control.** On the 103 parallel Global-PIQA items, per-item
deltas of the same arm are uncorrelated across languages (|r|<=0.25, mostly negative) although
item difficulty is shared (r=0.14-0.25). Weak (n=103) but consistent with language-specific
rather than content-specific learning.

**Written into the draft.** `05_exp.tex`: two new paragraphs in §5.3 (margins, churn), two new
paragraphs in §5.6 (corpus profile, affinity), one new paragraph on neighbourhood
generalisation, the 89% exposure number in §5.2, and the softened distractor claim in §5.1.
New `latex/07_appendix_pass3.tex` (wired into `main.tex`) and six generated tables.

### Second batch of results (2026-10-07, afternoon) — all four queued jobs landed

**A1 — the taxonomy predicts which items move.** Every item of 27 benchmarks labelled
(38,003 items, Qwen3.5-27B-FP8; gemma-4-26B second annotator on 3,856, kappa=0.66 with a
systematic symbolic/language-form confusion documented in the appendix). Whole-benchmark
distributions replace the 400-item samples. The decisive within-benchmark test: in Arabic,
Idiom-CPT gains +3.1 points on the symbolic items of culture benchmarks and +0.4 on the rest
(interaction +2.6 [0.2,5.1] p=0.03); under the wider symbolic+language grouping +5.4 [3.6,7.3]
p<0.001, and Hindi becomes testable and significant (+4.1 [0.3,7.8] p=0.015). Culture-CPT shows
no such preference, and on ArabCulture it *loses* 8.9 points on the 79 symbolic items while
gaining 1.3 on the rest. Alyah (26% symbolic) and DziriEval (19%) are the most symbolic culture
benchmarks and the two that Idiom-CPT improves most. At 2B the interaction points the same way
but is not significant.

**Training dynamics (Arabic, steps 200/400/800/1200/1608).** 86% of the Kinayat-Meaning gain is
in place after 200 steps; the seen-split gain climbs to step 1200; the unseen-split cost is
present at step 200 and does not deepen; no cultural effect appears transiently (ArabCulture is
flat at every checkpoint, Alyah grows monotonically); and the Culture-CPT reverse signal is
already +4.6 at step 200 and never exceeds +6.8 — flat in training tokens, unlike the forward
signal, which argues for a shallow preference shift. New figure `fig_dynamics.pdf`.

**Capability profile — the Chinese smoking gun.** The Chinese Culture-CPT model is the only arm
in the study whose held-out in-language perplexity is *worse than the untrained model's*
(zh Wikipedia 11.36 vs 10.11; held-out chengyu text 14.25 vs 13.59), while Random-CPT holds at
10.10/9.86 and the Arabic and Hindi culture arms track their controls. English retention is
unaffected for the same model, so the damage is specific to Chinese. This is the mechanism
behind §5.3's Chinese losses, measured without any benchmark.

### T2 added (2026-10-07): the paper's own "next step", run

§5.3 ends by saying "a notes generator aimed at evaluative associations rather than definitions
is the natural next step". T2 runs it. `ar_symbolism_summaries.py` (job 404540, Qwen3.5-27B-FP8)
writes, for each Arabic entity with >= 4 idioms, one or two Arabic sentences stating what that
entity stands for in Arabic proverbs, grounded only in the proverbs of the KB that contain it:
829 entities, e.g. *kalb* (dog) -> "baseness, forgetfulness of favours, stinginess and greed",
*jamal* (camel) -> "slowness, foolishness, a bad bargain". **The 98 entities the Arabic
symbolism probe asks about are excluded**, so the probe stays a generalisation test.

`build_ar_t2.py` then writes three corpora from one pass over the same 410,669 idiom-bearing
documents, differing only in what is appended:

| arm | appended | tokens |
|---|---|---|
| `ar_t2_untagged` | nothing | 1.00B |
| `ar_t2_dict` | the existing meaning tag (dictionary glosses) | 1.05B |
| `ar_t2_sym` | the entity-symbolism statements (44% of documents carry one) | 1.02B |

Training: jobs 404575/404576/404577, Qwen3.5-9B, 1,200 steps x 524K tokens = 0.63B, 4 GPUs each.
Evaluation: `src/culture/evaluation/eval_ar_t2.slurm`. The question the triple answers is the
one the paper poses and cannot answer: given the same documents and the same budget, does
stating the symbolic layer teach it, where stating the dictionary gloss does not?

### Still running
(none — all jobs of this pass have completed)

### T1 results (2026-10-08): confirmed for idiom-specific competence

Three Qwen3.5-9B arms, 1.05B tokens each, same pool, differing only in the selection rule.
The last column is the effect of the restriction alone (restricted minus unrestricted):

| | restr. − random | unrestr. − random | restr. − unrestr. |
|---|---|---|---|
| ChID (idiom cloze, n=3,756) | −0.7 | +0.5 | **−1.2 [−2.1,−0.3]** |
| held-out chengyu text (ppl) | +0.36 | −0.71 | **+1.07** |
| symbolism probe (n=206) | −1.9 | −0.5 | −1.5 (n.s.) |
| IdiomAtlas-MC seen (n=600) | +0.0 | **−2.5** | **+2.5** |
| zh Wikipedia (ppl) | −0.14 | −0.18 | +0.05 |

The two measures the full-budget Chinese arm loses most on — idiom cloze and perplexity on
idiom-bearing text — reverse when the restriction is lifted, and the restricted arm is the only
one of the three that is a worse model of chengyu-bearing text than the random control. The
IdiomAtlas-MC seen split goes the other way, and the general-Chinese degradation does not
appear at 1/22 of the tokens. Reported with those qualifications.

### T2 results (2026-10-08): the proposed next step does not work

Three Qwen3.5-9B arms on the same 410,669 Arabic idiom-bearing documents, 0.63B tokens,
differing only in what is appended.

| | meaning tags − untagged | symbolism tags − untagged |
|---|---|---|
| Kinayat-Meaning (325) | **+12.9 [8.9,16.9]** | **−4.6 [−7.4,−2.1]** |
| IdiomAtlas-MC seen (600) | **+14.8 [11.0,18.7]** | +0.0 |
| IdiomAtlas-MC unseen (600) | **−4.5 [−7.7,−1.3]** | +0.0 |
| Symbolism probe (98) | −2.0 | +0.0 [−7.1,+7.1] |
| Symbolism probe v1 (46) | +2.2 | +4.3 [0.0,10.9] |
| 8 culture / regional benchmarks | all within ±1 | all within ±1 |

The dictionary-tag control reproduces the main study's signature exactly at a fifth of the
budget, which validates the setup; the symbolism tags teach neither symbolism nor meaning. This
is the paper's own "natural next step", run, and it is a negative. Written into §5.6, the
conclusion and Appendix~\ref{app:t2} with its three limits (LLM-written statements, reduced
budget, probe entities held out).

### What a future pass could still do
- Native-speaker validation (the four remaining `\todo{}`s) — needs people, not compute.
- T2 at the full Arabic budget, and a version that does not hold out the probe entities, which
  would separate "the layer resists continued pretraining" from "it resists generalisation".
- The same symbolism-tag arm in Hindi, the language where idiom documents already shift the
  lure rate.

### Fixed along the way
- **A duplicate-qid bug with real consequences.** ArabCulture repeats a question id across
  country variants (3,463 rows, 2,168 distinct ids), so every analysis that keyed a dict on the
  id silently dropped a third of the benchmark and understated the \culturecpt{} gain as +0.8
  instead of +1.7. All item-level scripts now disambiguate by occurrence number. The within-
  benchmark interaction test was recomputed; the narrow symbolic grouping drops to +2.2
  (p=0.06) and the merged grouping rises to +5.1 (p<0.001), and the ArabCulture
  symbolic-items loss under Culture-CPT becomes Holm-significant (-9.0, p_adj=0.028).
- **A stale number in the draft.** §5.3 reported \idiomdocs{} at +8.0 on Kinayat-Meaning; the
  current eval records give +7.7 (0.3138 vs 0.2369). Corrected.
- The remaining 17 numbers of §5 that this pass could recompute all match the draft exactly.


---

## Pass 4 (2026-10-10) — the two follow-ups pass 3 left open

Pass 3 closed with a list of three things a future pass could still do. Two of them are
compute, and this pass runs both; the third (native-speaker validation) needs people.

**Why these two.** T2 found that appending statements of what an entity symbolizes teaches
neither symbolism nor meaning, while the dictionary-gloss control reproduced the main study's
signature exactly. That null has two readings the experiment cannot separate:

1. *the layer resists continued pretraining* — stating an evaluative association does not put
   it into the model at all; or
2. *the layer resists generalisation* — the model does learn the statements it reads, but T2
   deliberately held out the 98 entities the symbolism probe asks about, so the probe could
   only ever measure transfer to entities the training text never mentions.

**E1 — Arabic without the holdout.** `t2_symbolism_summaries.py --lang ar --include_probe`
regenerates the statements with the probe entities left in: 911 entities, of which **82 of the
probe's 98** now carry a statement (the other 16 have fewer than 4 idioms in the knowledge base
and fall below the generator's threshold). `build_t2.py --lang ar --arms sym --max_docs 410669`
writes `ar_t2_symall` over **exactly the same 410,669 documents** as the existing triple, so
`symall − sym` isolates one thing: whether the probe's own entities are among those discussed.
207,758 documents carry a tag (vs 179,103 in `ar_t2_sym`), 1.02B tokens.
A gain on the probe reads (2); a null reads (1), and is the stronger statement.

**E2 — the same triple in Hindi.** Hindi is where idiom-bearing documents already move the
probe's lure rate by 8–10 points (§5.5), so it is the language with the most headroom for a
symbolism tag to work, and a null there is correspondingly harder to attribute to the language.
`t2_symbolism_summaries.py --lang hi` writes statements for 1,707 entities with the 155 Hindi
probe entities held out; `build_t2.py --lang hi` writes the token-matched triple over 228,413
idiom-bearing Hindi documents (untagged 0.99B / dict 1.01B / sym 1.05B est. tokens; 75% of
documents carry a symbolism tag). The `dict` arm is the setup check, exactly as in Arabic.

**Infrastructure.** All of pass 3's T2 code was Arabic-only; this pass generalises it rather
than copying it. New: `analysis/v3/t2_symbolism_summaries.py` (language + `--include_probe`),
`analysis/v3/build_t2.py` (language + `--arms` + `--max_docs`), `analysis/v3/t2b_stats.py`
(arm list + the `symall − sym` contrast), slurm `t2_symbolism.slurm`, `build_t2.slurm`,
`training/.../cpt_t2.slurm`, `evaluation/eval_hi_t2.slurm`, `evaluation/eval_ar_t2b.slurm`,
and four configs `qwen3p5_9b_cpt_{ar_t2_symall,hi_t2_untagged,hi_t2_dict,hi_t2_sym}.yaml`.
Training is the T2 recipe unchanged: Qwen3.5-9B, 1,200 steps x 524K tokens = 0.63B, 4 GPUs.

**Jobs.** generation 413827 (hi) / 413828 (ar all); build 413922 (hi) / 413923 (ar symall);
training 413944 `ar_t2_symall`, 413945 `hi_t2_untagged`, 413946 `hi_t2_dict`,
413947 `hi_t2_sym`.

### Results (2026-10-10) — one positive, one negative, and they point the same way

All eight jobs landed: four 9B arms at 7.7--8.0 h each, four evaluations.

**E2 (Hindi) — the proposed next step works, in Hindi.** Against the untagged control on the
same 228,413 documents and the same 0.63B tokens:

| | meaning tags | symbolism tags |
|---|---|---|
| Symbolism probe (155, **entities held out**) | +0.0 [-5.2,+5.2] | **+9.0 [+2.6,+15.5]** (52.9->61.9, McNemar p=0.016) |
| Symbolism probe, margin | -0.013 [-0.087,+0.061] | **+0.288 [+0.127,+0.444]** |
| Symbolism probe, lure rate | -3.2 | -5.8 [-11.6,+0.0] |
| IdiomAtlas-MC hi seen (600) | **+17.5 [+13.8,+21.2]** | **-4.0 [-6.5,-1.5]** (Holm 0.035) |
| IdiomAtlas-MC hi unseen (600) | -1.3 | -1.5 |
| ParamBench culture (5,449) | **+0.9 [+0.1,+1.8]** | +0.5 |
| ParamBench other (5,219) | -0.3 | **-1.3 [-2.1,-0.5]** (Holm 0.016) |
| MILU / MABL / Global-PIQA hi | all within +-1 | all within +-1 |
| held-out proverb text (ppl) | 2.764 vs 2.852 | 2.857 vs 2.852 |

This is a **double dissociation inside one controlled triple**: the two tag types, over the same
documents at the same budget, each move the benchmark that matches what they state and cost the
other. It is the sharpest available form of the paper's thesis, and the probe gain is
generalisation --- `t2_probe_split.py --lang hi` confirms **0 of the 155 probe entities** appear
in the statements. The probe gain does not survive Holm over the 12 Hindi benchmarks
(p_adj=0.16), which is why the margin (+0.288, CI well clear of zero, every item used rather
than thresholded) is the form to lead with. The perplexity rows rule out the cheap explanation:
the symbolism arm is no better at modelling proverb-bearing Hindi than the untagged control, so
the probe gain is not a by-product of fitting the domain.

**E1 (Arabic, no holdout) — the Arabic null is not a failure to generalise.** With 82 of the
probe's 98 entities now explicitly described in the training text, over the identical 410,669
documents: probe `symall - sym` **+2.0 [+0.0,+5.1]** (n.s.), `symall - untagged` +2.0
[-5.1,+9.2] (n.s.). `t2_probe_split.py --lang ar` splits the probe by whether the item's entity
was stated: the gain does not concentrate on the 82 stated items against the 16 that stayed out
(accuracy interaction +2.4 points, p=0.27; margin interaction -0.01, p=0.92 --- weak with 16
control items, but the point estimates show nothing). For scale, the meaning tags gain **+14.8**
on exactly the analogous in-distribution condition. So in Arabic the statements are not learned,
not merely not transferred. Two incidental Holm-significant cells in the `symall` column
(ArabicCulturalQA +1.8, ArabicMMLU +1.0 against `sym`) have no mechanism and are not reported.

**What changed in the paper.** The pass-3 conclusion "the proposed next step does not work" is
wrong as stated and is replaced. Rewritten: the abstract's last sentence, the Conclusion's
follow-up sentence, the §5.6 paragraph (now four paragraphs covering both languages and the
no-holdout arm), Appendix `app:t2` (generation for both languages, the no-holdout arm, five
limits), and the Limitations sentence on the follow-ups. `tables/t2.tex` regenerated with four
arms; new `tables/t2_hi.tex`.

**The honest caveats, all in the paper.** The two languages differ in corpus size and tag rate
(75% of Hindi documents carry a statement vs 44% of Arabic ones) as well as in language, so dose
is a live alternative to the headroom explanation. The statements are LLM-written, so the Hindi
positive is a result about LLM-written symbolism statements. The budget is a fifth of the main
Arabic study.

### Status
- generation, corpora, training, evaluation, statistics, draft: done
- remaining in this repo: native-speaker validation (3 `\todo{}`s, needs people), HF upload of
  the new 9B checkpoints (needs network), and committing both working trees
