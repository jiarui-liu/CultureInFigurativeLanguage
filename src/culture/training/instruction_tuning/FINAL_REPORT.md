# SFT FINAL REPORT

Eval = the SAME pipeline used for the CPT checkpoints, run on the 4 SFT checkpoints.
`base`/`cpt` columns are the pre-SFT CPT-eval numbers for reference (higher = better,
except ppl_* which are BPB/perplexity where lower = better).

### HI

| task | base(CPTeval) | cpt(CPTeval) | hi-base-sft | hi-cpt-sft |
|---|---|---|---|---|
| en:boolq | 0.8850 | 0.8306 | 0.8954 | 0.8887 |
| en:mmlu | 0.7852 | 0.7767 | 0.7479 | 0.7528 |
| en_gen:gsm8k | 0.8749 | 0.8590 | 0.8848 | 0.8931 |
| en_gen:humaneval | 0.7195 | 0.5915 | 0.7317 | 0.7256 |
| global_piqa | 0.6000 | 0.7200 | 0.6000 | 0.6900 |
| idiomce | 2.0575 | 2.4825 | — | — |
| mabl | 0.5310 | 0.5920 | 0.5410 | 0.5630 |
| milu | 0.5323 | 0.6081 | 0.5674 | 0.5827 |
| ppl_hi_proverbs | 14.5639 | 2.4767 | 4.4848 | 3.0119 |
| ppl_hi_samanantar | 65.1219 | 3.4329 | 4.9514 | 3.9474 |
| ppl_wikitext | 13.2580 | 13.7842 | — | — |

### AR

| task | base(CPTeval) | cpt(CPTeval) | ar-base-sft | ar-cpt-sft |
|---|---|---|---|---|
| alyah | 0.3339 | 0.3809 | 0.3083 | 0.3365 |
| ar_figurative | 0.2930 | 0.3344 | 0.2484 | 0.3089 |
| arabculture | 0.4583 | 0.4909 | 0.4404 | 0.4623 |
| arabic_cultural_qa | 0.7079 | 0.7069 | 0.6663 | 0.6678 |
| arabicmmlu | 0.6730 | 0.6794 | 0.6448 | 0.6614 |
| dzirieval | 0.3263 | 0.3526 | 0.3095 | 0.2979 |
| en:boolq | — | 0.8422 | — | — |
| en:mmlu | — | 0.7827 | — | — |
| en_gen:gsm8k | — | 0.8650 | — | — |
| en_gen:humaneval | — | 0.6463 | — | — |
| global_piqa_ar | 0.5811 | 0.5920 | 0.5665 | 0.5811 |
| global_piqa_ar_parallel | 0.2977 | 0.2816 | 0.2913 | 0.3010 |
| kinayat_cloze | 0.5000 | 0.5267 | 0.5267 | 0.5600 |
| kinayat_meaning | 0.2215 | 0.4523 | 0.1046 | 0.1908 |
| ppl_fineweb2_heldout | 12.4320 | 9.7214 | 14.6512 | 12.2032 |
| ppl_wiki_heldout | 7.8197 | 7.2398 | 9.1221 | 8.3793 |

### Mixture manifests
- **hi**: total=300000 target=180000 english=120000 realized_target_ratio=0.6 seed=42
- **ar**: total=300000 target=180000 english=120000 realized_target_ratio=0.6 seed=42
