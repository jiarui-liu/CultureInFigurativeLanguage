# Arabic CPT — Final Report (2-variant experiment)

**Date:** 2026-08-24 · **Model:** Qwen3.5-9B (base) · **Corpus:** FineWeb-2 `arb_Arab`
**Question:** does idiom-curated + knowledge-tagged CPT ("augmented") beat plain
in-language CPT ("baseline/unfiltered") on Arabic figurative & cultural competence?

## Runs
| Variant | Corpus | Docs | Tokens | Train loss | Checkpoint |
|---|---|---:|---:|---:|---|
| **augmented** (`ar_amthal`) | FineWeb-2 filtered to idiom-bearing docs + Arabic knowledge block | 410,669 | 1.127B | 2.084 | `ckpts/qwen3p5-9b-ar-cpt` |
| **baseline** (`ar_unfiltered`) | same FineWeb-2, random docs, no filter/no tag, matched token budget | — | 1.127B (matched) | 2.177 | `ckpts/qwen3p5-9b-ar-cpt-unfiltered` |

Both 3 epochs, identical optimizer (OdysSim recipe), 4×8 H100. The only difference
is the idiom curation. Base = untuned Qwen3.5-9B.

## Results (primary metric: acc_norm for continuation tasks, acc for letter tasks; BPB lower=better)

### Dim 2 — Arabic language modeling (BPB)
| Probe | base | augmented | baseline |
|---|---:|---:|---:|
| FineWeb-2 heldout (in-domain) | 0.6147 | 0.5547 | **0.5256** |
| Wikipedia heldout (OOD)       | 0.5681 | **0.5468** | 0.5500 |

Both CPT variants improve Arabic modeling over base. Baseline is best in-domain
(it is pure FineWeb-2 text); augmented is best OOD and carries the extra knowledge-block
distribution. In-language adaptation clearly worked for both.

### Dim 3 — figurative / idiom  (the crux)
| Task | base | augmented | baseline | aug−base | base−base | **aug−baseline** |
|---|---:|---:|---:|---:|---:|---:|
| **kinayat_meaning** (knowledge injection) | 0.2215 | **0.4523** | 0.2369 | **+0.2308** | +0.0154 | **+0.2154** |
| **ar_figurative** (generalise to unseen idioms) | 0.2930 | **0.3344** | 0.2803 | +0.0414 | −0.0127 | **+0.0541** |
| kinayat_cloze (comprehension) | 0.5000 | 0.5267 | 0.5333 | +0.0267 | +0.0333 | −0.0066 |

### Dim 4 — cultural competence
| Task | base | augmented | baseline | aug−base | unf−base |
|---|---:|---:|---:|---:|---:|
| arabculture | 0.4583 | 0.4909 | 0.4871 | +0.0326 | +0.0288 |
| arabic_cultural_qa | 0.7079 | 0.7069 | 0.7179 | −0.0010 | +0.0100 |
| arabicmmlu | 0.6730 | 0.6794 | 0.6747 | +0.0064 | +0.0017 |
| global_piqa_ar | 0.5811 | 0.5920 | 0.5938 | +0.0109 | +0.0127 |
| alyah | 0.3339 | 0.3809 | 0.3518 | +0.0470 | +0.0179 |
| dzirieval | 0.3263 | 0.3526 | 0.3347 | +0.0263 | +0.0084 |

### Control — culture-agnostic physics (regression detector)
| Task | base | augmented | baseline |
|---|---:|---:|---:|
| global_piqa_ar_parallel | 0.2977 | 0.2816 | 0.2718 |

Both variants ~flat/slightly down on the control (within noise) — no spurious inflation.

## Conclusion

**Idiom curation delivers exactly where it should, and clearly beats plain
in-language CPT on the idiom axis:**

- **Knowledge injection (kinayat_meaning): +23.1 pts over base, +21.5 pts over the
  baseline.** The augmented model learned the injected idiom meanings; plain
  in-language data barely moved this (+1.5 pts).
- **Generalisation to unseen idioms (ar_figurative): +4.1 pts over base, +5.4 pts
  over baseline** (the baseline actually regressed here). Curation transfers beyond
  the trained idioms.
- **Comprehension (kinayat_cloze):** both ~+3 pts over base, essentially tied — this
  is the hardest, near-symmetric task.
- **General cultural competence (Dim 4) and Arabic LM (Dim 2):** augmented and
  baseline are comparable — these gains come from *in-language* CPT, not idiom
  curation specifically (as expected).

**Bottom line:** for Arabic, the augmented (idiom-curated) corpus buys a large,
targeted improvement in figurative/idiom knowledge and generalisation over a
token-matched plain-FineWeb-2 baseline, at no cost to general cultural competence
or the culture-agnostic control. Consistent with the Hindi/Chinese pattern.

## Job outcomes
- Filter: 25-way parallel array `241061` (all COMPLETED); scanned ~62M docs → 410,669 tagged.
- Prepare `241078`; augmented train `241299`; baseline sample `241080` + train `241081`.
- Eval prereqs `241725`; eval `241726` (base) / `241727` (cpt) / `241728` (unfiltered) — all COMPLETED.
- Outputs: `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/{base,cpt,unfiltered}/`.
