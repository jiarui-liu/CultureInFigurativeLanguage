# Point 6 (filtered-untagged decomposition) — DONE (2026-08-26)

All three untagged CPT checkpoints trained, evaluated with the SAME protocol as
base/cpt/unfiltered, CIs refreshed, and the paper Analysis section updated with the
document-selection vs. meaning-tag decomposition.

## cpt_vs_untagged (MEANING-TAG effect = Idiom-CPT - Untagged), key tasks

Accuracy points; * = McNemar p<0.05 with bootstrap 95% CI excluding 0.

AR:
- kinayat_meaning: +13.5*  (p<1e-4)   [total cpt-unfilt +21.5*; doc-sel +8.0]
- ar_figurative:   +1.6    (p=0.36)   [total +5.4*; doc-sel +3.8]
- kinayat_cloze:   +2.0    (ns)       [total -0.7; doc-sel -2.7]
- alyah:           +0.9    (ns)       [total +2.9*; doc-sel +2.0]

ZH:
- chengyu_bench:   +2.6*   (p=0.003)  [total +6.8*; doc-sel +4.2]
- chid:            -0.2    (ns)       [total -0.4; doc-sel -0.2]
- cmmlu:           +0.4    (ns)       [total +0.1; doc-sel -0.3]
- ccpm:            +0.2    (ns)

HI:
- mabl:            -0.6    (ns)       [total +0.7; doc-sel +1.3]
- global_piqa:     +3.0    (ns)       [total +7.0; doc-sel +4.0]
- milu:            -0.1    (ns)       [total -0.6; doc-sel -0.5]

## Conclusion
The meaning-tag component is significant ONLY on the two pure meaning-lookup tasks
(ar kinayat_meaning, zh chengyu_bench) — exactly the knowledge-injection targets the
tags were designed for. All broader gains (alyah, ar_figurative, hi global_piqa) come
mainly from DOCUMENT SELECTION (idiom-rich corpus), with a small non-significant tag
residual. Hindi shows no significant tag effect on any task.

## Artifacts
- Evals: /lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/{hi,ar,zh}/untagged/
- CIs: CIFL/docs/paper_stats/ci_report.json (refreshed with untagged for all 3 langs)
- Paper: OverleafCultureInFigurativeLanguage/colm2026_conference.tex, tab:analysis + sec:analysis
  (added Delta_sel / Delta_tag / Delta_tot columns + "Document selection vs. meaning tags" paragraph)
- Point 4 (ar English-retention): eval/ar/{cpt,unfiltered}/en,en_gen (done earlier)

Monitor cron deleted after this file was written.
