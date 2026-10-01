# IdiomCulture: testing culture-trained models on the culture that idioms encode

Status: design + code ready (2026-10-01). API stages to be run on the API server; evaluation runs
on babel, where the 2B checkpoints live.

## 1. Why

The 2B reverse-direction study (Culture-CPT, Culture-CPT+notes vs Random-CPT) found no gain on idiom
benchmarks, while the same culture data did improve the culture benchmark closest to its content
(ArabCulture +2.1 / +2.8, Holm-significant). Two readings are possible:

1. culture-rich text teaches cultural knowledge, but not the knowledge idioms rely on; or
2. it does teach that knowledge, but idiom-*meaning* benchmarks cannot show it, because answering
   them also requires knowing the idiom's conventional meaning.

The existing culture benchmarks (ArabCulture, Alyah, MILU, CCPM, ...) cannot separate these readings:
they cover festivals, food, geography, and exam knowledge, not the specific cultural knowledge
that idioms presuppose (what an entity symbolizes, which behaviour is valued, how a social relation
works). Section 4 of the paper shows that this layer is culture-specific (entity typology,
divergence index 0.60).

IdiomCulture extracts that layer from IdiomAtlas and asks about it **without the idiom**. It gives a
third bridge between the two directions:

| | idiom benchmarks | IdiomCulture (culture *in* idioms) | generic culture benchmarks |
|---|---|---|---|
| Idiom-CPT (forward) | + (seen items only) | ? does reading idioms teach the culture they encode? | ≈ 0 |
| Culture-CPT (reverse) | ≈ 0 | ? does culture text teach the culture idioms encode? | + (ArabCulture) |

Predictions:
- If Culture-CPT gains on IdiomCulture but not on idiom meaning, reading (2) holds: the knowledge
  transfers, but meaning tests need lexical knowledge of the idiom as well.
- If Idiom-CPT gains on IdiomCulture items whose evidence idioms were **seen**, idioms teach
  culture in a form that is usable without the idiom (a cleaner forward test than ArabCulture).
- The **unseen** split (evidence idioms in no training document) and the **universal** split
  (aspects judged not culture-specific) are controls.

## 2. Design

### 2.1 Unit: a culture aspect
A self-contained factual statement about the culture that an idiom presupposes or encodes,
written without quoting or paraphrasing the idiom, and not a restatement of its meaning.
Example: *In traditional Chinese culture, the crane symbolizes longevity.* (not: *the idiom means
one should be patient*). Each aspect carries a coarse type (value/norm, belief/symbolism,
practice/custom, social relation, material culture, history/religion, environment/livelihood),
a specificity label (culture-specific vs universal), and its evidence idioms.

### 2.2 Pipeline (`src/culture/idiomculture/build_idiomculture.py`)

| stage | where | what |
|---|---|---|
| `exposure` | babel, **done** | label every IdiomAtlas idiom seen / unseen / other with the same corpus scans as IdiomAtlas-MC → `culture/data/idiomculture/exposure_{zh,hi,ar}.json` |
| `extract` | API server | generator LLM, one call per sampled idiom → 0–3 aspects (JSON) |
| `consolidate` | API server (CPU ok) | Qwen3-Embedding-0.6B, greedy clustering at cosine ≥ 0.85 → one aspect per cluster with all its evidence idioms; exposure = seen if any evidence idiom is seen |
| `questions` | API server | generator LLM writes one 4-option MC question per aspect in the target language: gold, an Anglophone *lure*, two plausible wrong options |
| `verify` | API server | verifier LLM from a **different family**: (a) answers blind, (b) judges six criteria |
| `export` | API server | filters + MC jsonl (letter format and continuation format) + human sheet |

Sampling in `extract` (defaults): all-or-2,000 unseen idioms, 2,000 seen, 1,000 other per language
(zh has only 167 unseen idioms; all are used). Exposure counts: zh 20,440 seen / 167 unseen /
6,690 other; hi 2,124 / 14,465 / 0; ar 4,944 / 5,442 / 0.

### 2.3 Quality control (in `verify` + `export`)
An item is kept only if
1. the verifier judges it answerable without the idiom, gold correct, single correct answer,
   no form giveaway, and fluent;
2. the verifier answers it correctly blind (`--require_verifier 1`; report the count with 0 too);
3. no IdiomAtlas idiom occurs in the question or options (the corpus matcher of `build_pool.py`);
4. no option shares more than 50% of its words (Chinese: characters) with an evidence idiom.

`export` prints the longest-option baseline, which should be near 25%; report it in the paper.
`culture_specific` is not a filter but a split (`meta.specificity`), so universal items serve as a
control. A random 100 items per language go to native speakers (`human_sheet.jsonl`: is the claimed
answer right in your culture? is it specific to your culture?).

### 2.4 Models
- Generator: a frontier model strong in all three languages (default `gemini-3.8-pro` via
  `--gen_provider gemini`; Claude via `--gen_provider anthropic` also works).
- Verifier: a different family (e.g. via OpenRouter, `--ver_provider openrouter --ver_model <id>`).
  Avoid the Qwen family for generation and verification, since the evaluated models are Qwen3.5.
- Expected API volume per language: ~5K extract calls, ~4–6K question calls, 2× that for verify.
  All responses are cached (`LLM_CACHE_DIR`), so reruns are free.

## 3. Commands (API server)

```bash
cd CultureInFigurativeLanguage
export PYTHONPATH=src LLM_CACHE_DIR=$PWD/../llm_cache
# keys: gemini_api_key_1..N, OPEN_ROUTER_API_KEY, or ANTHROPIC_API_KEY
pip install sentence-transformers  # for consolidate (CPU is fine)
W=$PWD/../idiomculture_work; OUT=$PWD/../idiomculture_mc
for L in zh hi ar; do
  python -m culture.idiomculture.build_idiomculture --stage extract     --lang $L --work $W --gen_provider gemini --gen_model gemini-3.8-pro
  python -m culture.idiomculture.build_idiomculture --stage consolidate --lang $L --work $W
  python -m culture.idiomculture.build_idiomculture --stage questions   --lang $L --work $W --gen_provider gemini --gen_model gemini-3.8-pro
  python -m culture.idiomculture.build_idiomculture --stage verify      --lang $L --work $W --ver_provider openrouter --ver_model <verifier id>
  python -m culture.idiomculture.build_idiomculture --stage export      --lang $L --work $W --out_dir $OUT
done
```

Pilot first: run `extract` with `--n_unseen 50 --n_seen 50 --n_other 0` for one language, read
20 aspects and 20 questions by hand, adjust prompts, then run in full (the cache keeps the pilot).
Things to check in the pilot: aspects that merely restate the idiom meaning; questions that ask
about an expression; lures that are actually correct; options that name the culture only in the gold.

Output to bring back to babel: `$OUT/idiomculture_{zh,hi,ar}{,_letter}.jsonl` (small) and
`$W/<lang>/verified.jsonl` (for the paper's statistics).

## 4. Evaluation (babel)

```bash
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
cp idiomculture_mc/*.jsonl $B/eval_data/mc/
cd CultureInFigurativeLanguage/src/culture/bidirectional
for L in ar hi zh; do for run in $(ls $B/ckpt/$L); do
  sbatch --partition=preempt --qos=preempt_qos --requeue \
    --export=ALL,MODEL=$B/ckpt/$L/$run,OUT=$B/eval2b/$L/$run,TASKS=jsonl:idiomculture_${L}_letter+jsonl:idiomculture_${L},BS=4 eval.slurm
done; done
# base checkpoints (no training): MODEL=Qwen/Qwen3.5-2B -> eval2b/$L/i_base, Qwen/Qwen3.5-2B-Base -> eval2b/$L/base
```

Analysis: add `idiomculture_<L>_letter` to `GROUP` in `aggregate.py` (new group `idiom_culture`)
and run `aggregate.py` and `margin_analysis.py`; then split by `meta.exposure`, `meta.specificity`
and `meta.type` (paired bootstrap vs Random). 9B checkpoints: same task names via the §4 commands
of `bidirectional_transfer_plan.md`.

## 5. What goes in the paper
- Benchmark description (one paragraph + appendix: prompts, filters, counts, human validation).
- Results: one row per condition for IdiomCulture overall, seen vs unseen, culture-specific vs
  universal; the transfer-matrix figure gains an *IdiomCulture* column.
- If the reverse direction is positive here, the framing becomes: culture-rich text teaches the
  culture behind idioms but not idioms' conventional meanings; idioms teach their meanings but
  little of the culture behind them, beyond what they are paired with.

## 6. Related control to run on the API server: style-matched Kinayat-Meaning

Finding on babel (2026-10-01, `docs/paper_stats/v2/kinayat_exposure.json`): on Kinayat-Meaning the
culture arms raise the gold-vs-distractor log-probability margin over Random-CPT by about +0.11
(Culture-CPT +0.109 [0.093, 0.125] on the 287 items whose expression never occurs in the culture
corpus; +0.101 on the 38 that do), while accuracy changes are not significant. So the gain is not
exposure to the expressions. Remaining confound: the gold options are the classical dictionary
glosses ("يقولون…", "كناية عن…", "يريدون به"), the distractors are LLM-written in modern MSA, and
culture-rich text may simply make the model prefer the classical register.

Control: rewrite each gold option into the register of its distractor (same content, modern MSA,
similar length, no "كناية/يقولون" framing), with an API model, and re-evaluate all Arabic 2B
checkpoints on the rewritten set. If the culture-arm margin gain survives, it reflects knowledge of
the meaning; if it vanishes, it was register. Input: the Kinayat-Meaning items (qid, idiom, options,
gold) from `$B/eval2b/ar/i_random/kinayat_meaning.json`; output: a `jsonl:kinayat_meaning_stylematched`
task in the MC jsonl format of §2.2 (continuation format, two options, `gold` index unchanged).
Prompt sketch: "Rewrite the following Arabic explanation of an expression's meaning so that it has
the same content but the style of the second text (modern standard Arabic, plain explanatory
sentence, about N words). Do not add or remove information. Do not start with كناية or يقولون."
Check 30 rewrites by hand for meaning preservation before running the evaluation.
