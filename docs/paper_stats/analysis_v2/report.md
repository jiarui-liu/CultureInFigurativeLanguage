# Section-4 analyses (v2) — report

Code: `src/culture/analysis/v2/` (`run_final.sh` reproduces every number from cached LLM outputs).
Outputs: this directory.

## 0. Deviations and model roles
- API quotas were exhausted, so all LLM steps use local models: **Qwen3.5-27B-FP8** (vLLM, greedy, thinking off) is the primary annotator (typology, entity translation, English paraphrases of meanings, pair judge); second annotators: Qwen3.5-9B (same family), aya-expanse-8b (weaker), nemotron-3-ultra-550b (OpenRouter free tier, different vendor). Embeddings: Qwen3-Embedding-0.6B (same size as the original analysis).
- Arabic entities: taken from the enriched KB on HF (`Jerry9999/CultureInFigurativeLanguage`, `data/idioms/ar`; the paper's GPT-5.4 entities), re-grounded with `normalize_ar` (39 dropped); written to `/data/group_data/r3lit_culture_pretrain/culture/bidir/ar_entities.jsonl`.
- Bug found: `fig_entities` Arabic normalization turns الله into "له" (fixed in `code/make_figures.py`).
- tab1 has 516 rows; stroke/笔画 is empty, so 515 is correct.

## 1. Entity semantic typology (4 languages)
All distinct entities typed (100% of mentions). Stability on 300 entities: kappa 0.805 (nemotron), 0.695 (Qwen3.5-9B), 0.537 (aya).
Shares of mentions (%), adjusted residual in parentheses:

| type | en | zh | hi | ar |
|---|---|---|---|---|
| animal | 7.2 | 8.5 | 8.1 | **12.0 (+17)** |
| body/mind | 16.0 | 15.4 | 10.4 | 14.7 |
| nature & cosmos | 10.4 | **17.4 (+22)** | 14.0 | 13.0 |
| food & drink | 4.1 | 2.3 | **8.2** | 7.2 |
| kinship & social roles | 6.5 | 8.3 | **22.9 (+62)** | 18.6 |
| religion & supernatural | 2.3 | 1.6 | **6.2** | 2.6 |
| occupation & economy | 3.8 | 1.7 | **6.5** | 3.7 |
| artefact & household | **19.6 (+29)** | 12.1 | 11.8 | 11.3 |
| abstract/other | 30.1 | 32.6 | 11.9 | 16.9 |

chi2(24)=15,835, p<1e-300, Cramér's V=0.205 (idiom-bootstrap CI [0.202, 0.209]); pairwise V en–zh 0.152 (most similar), zh–hi 0.366 (most different).
Weakens the draft: English is not more "body"-based than Chinese (16.0 vs 15.4%); its distinctive type is artefacts/household. Arabic "led by God" holds for entity ranks only (religion type 2.6%).

## 2. Prompt-independent divergence, same entity (en–zh)
Meanings paraphrased to one English sentence (same prompt both languages), embedded; size-matched comparisons (5 vs 5 idioms, 30 draws). 327/515 entities have ≥5 idioms per side.
Centroid distance: same entity same language (split halves) 0.16; same entity across languages 0.225 [0.221, 0.229]; random different entities across languages 0.266. Divergence index DI=0.60 (energy distance 0.58). Same entity closer than a random entity for 89% of entities. Spearman with LLM shared share rho=-0.30.
LLM counts: 497/515 report exactly 2–3 shared items; recomputed 2.57 shared vs 5.89 unique (2.73 vs 6.10 on the 327 subset).
Caveats: the most divergent entities are mostly polysemy/translation mismatches (pound/磅, fall/秋, suit/诉讼, character/字, sake/故); embeddings are insensitive to valence (dog is among the least divergent).
Case-table idiom counts (en/zh): dog 20/20 (KB 80/149), heart 20/20, moon 15/20, red 4/20, turtle 3/20, dragon 2/20, phoenix 1/20 — the last four are below the ≥5 threshold.

## 3. Same meaning, different entity (en–zh)
Reproduced: shared-entity rate 5.5% (1,107/20,068), 7.5% at cos≥0.75; rises 4.8% → 6.0% → 7.2% → 12.2% (cos≥0.80).
Clusters: 82.5% strictly one-to-one; single idiom per side en 90.3%, zh 91.3%; only 34/3,494 (1.0%) have ≥2 idioms on both sides; 2.2% have ≥5 idioms in total.
Pair precision (LLM judge, not human; 200 pairs, 50/bin): strict 0.16 / 0.36 / 0.52 / 0.68 for [0.70,0.72) / [0.72,0.75) / [0.75,0.80) / ≥0.80; lenient 0.64 / 0.76 / 0.88 / 0.92; weighted strict 0.27, lenient 0.71. nemotron strict 0.10/0.40/0.54/0.68, kappa 0.55. ⇒ "same meaning" should read "related meaning".

## 4. en–hi / en–ar (embedding divergence only)
| pair | entities | DI | same < random | median percentile of true translation |
|---|---|---|---|---|
| en–zh | 156 | 0.58 | 92% | 0.11 |
| en–hi | 128 | 0.62 | 92% | 0.14 |
| en–ar | 129 | 0.57 | 88% | 0.16 |
Similar across pairs; do not rank pairs. Top ranks again contain translation artefacts.
