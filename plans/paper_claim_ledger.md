# Paper claim ledger

This ledger records the canonical source for every planned headline claim. Values are percentage-point changes against token-matched Random-CPT unless stated otherwise.

| Planned claim | Canonical value | Evidence source | Status |
|---|---:|---|---|
| IdiomAtlas contains 75,578 entries in four languages | en 21,278; zh 27,297; hi 16,617; ar 10,386 | `OverleafCultureInFigurativeLanguage/latex/tables/dataset.tex`; construction scripts under `src/culture/data_processing/` | validated |
| Shared meanings usually use different imagery | 5.5% (1,107/20,068) of English–Chinese matched-meaning pairs share an entity | `docs/paper_stats/analysis_v2/report.md`; `pairing_threshold_sensitivity.json` | validated |
| Shared entities carry language-specific meanings | all six language pairs; English–Chinese shared 2.6 meanings/entity and 5.9 language-specific | `docs/paper_stats/analysis_v2/entity_divergence_cross.json`; `entity_divergence_en_zh.json` | validated |
| Meaning tags teach encountered idioms at 9B | ar +24.7 [20.3,29.0]; hi +25.5 [21.8,29.2]; zh +10.7 [8.2,13.3] | `docs/paper_stats/v2/idiomatlas_9b.json` | validated artifact |
| Arabic Kinayat meaning improves | +21.5 [16.6,26.5] | `docs/paper_stats/v2/stats_9b.json` | validated artifact |
| Idiom-bearing text transfers to unseen Arabic idioms | text without tags +3.0 [0.2,5.8] on IdiomAtlas unseen | `docs/paper_stats/v2/idiomatlas_9b.json` | validated artifact |
| Figurative inference transfers beyond the KB | Idiom-CPT +5.4 [1.6,9.2]; text without tags +3.8 [0.6,7.0] on AR-Figurative; about 99% of expressions absent from KB | `docs/paper_stats/reframe_stats_9b.json`; overlap analysis in evaluation builders | validated |
| Emirati cultural reasoning improves | Alyah +2.9 [1.2,4.6], Holm-adjusted p=0.0207; +2.3 after removing figurative items | `docs/paper_stats/v2/stats_9b.json` | validated artifact |
| Symbolism becomes less English-centered across languages | pooled lure rate −4.4 [−7.2,−1.5], p=.0054; target-vs-English margin +0.185 [0.102,0.268], p=.00003 | `docs/paper_stats/v3/cross_language_symbolism.json` | validated primary artifact |
| Hindi symbolism becomes less English-centered | lure rate −8.4 for Idiom+Meaning and −10.3 for Idiom-Text | `docs/paper_stats/v3/cross_language_symbolism.json` | validated |
| Chinese symbolism moves in the same direction | lure rate −3.4, paired p=.092; decision margin +0.245 [0.139,0.348] | `docs/paper_stats/v3/cross_language_symbolism.json` | validated |
| Independent symbolism audit | 376/426 auditable items pass cross-model consensus; Hindi lure shift −10.9, Chinese −4.0 | `docs/paper_stats/v3/symbolism_probe_audit.json` | validated, item selection independent of predictions |
| Glosses narrow unseen Arabic generalization | Idiom-CPT −4.0 [−7.7,−0.3] against Random-CPT; −7.0 against same documents without tags | `docs/paper_stats/v2/idiomatlas_9b.json` | required mechanism result |
| Instruction tuning preserves acquired Arabic meaning | +15.1 for Idiom+Meaning-IT against Random-CPT-IT | `OverleafCultureInFigurativeLanguage/latex/tables/it_results.tex` and raw instruction-tuning evaluations | validated table value |
| Gains track exposure | seen-item gain rises with occurrence count in Arabic and Hindi; Arabic exposure bins reach 54% to 100% accuracy | `docs/paper_stats/analysis_v2/exposure_dose_response.json`; `docs/paper_stats/v2/kinayat_exposure.json` | validated diagnostic |
| Idiom documents are culture-rich | classifier mean 1.94 vs 0.94 random in Arabic; 1.56 vs 1.17 Hindi; 1.67 vs 0.63 Chinese | `latex/tables/corpus_profile.tex`; `docs/paper_stats/v3/kb_corpus_benchmark.json` | validated artifact |
| Idioms and current culture benchmarks cover different constructs | IdiomAtlas inventories are 62–84% symbolic/evaluative; evaluated idiom/figurative tasks are 62–100% symbolic-or-language under the reliability audit | `docs/paper_stats/v3/item_taxonomy.json`; `OverleafCultureInFigurativeLanguage/latex/tables/item_taxonomy.tex` | validated with scope stated explicitly |
| Symbolic-item interaction appears across languages | merged symbolic+language interaction +5.1 Arabic and +4.1 Hindi; Chinese null | `docs/paper_stats/v3/layer_interaction_culture_merged.json` | secondary mechanism claim |
| Symbolism statements can transfer to held-out entities | Hindi +9.0 [2.6,15.5], margin +0.288 [0.127,0.444] | `docs/paper_stats/v3/t2_hi_stats.json`; `t2_hi_probe_split.json` | controlled follow-up |
| English and general-capability cost matches Random-CPT | MMLU and English task changes are comparable to Random-CPT; held-out idiom text perplexity improves | `OverleafCultureInFigurativeLanguage/latex/tables/retention.tex`; `capability.tex` | validated at tested tasks and scales |

All abstract and main-text numerical claims now resolve to a validated artifact above. Per-language symbolism estimates remain visible beside the pooled estimate so the aggregation does not conceal heterogeneity.
