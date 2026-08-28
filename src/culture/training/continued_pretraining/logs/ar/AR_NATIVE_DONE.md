# Arabic NATIVE SFT (Quora-Arabic-GPT4 + CIDAR) — COMPLETE (2026-08-27)

Second Arabic instruction-tuning run finished; paper updated & verified.

## Recipe
Native-Arabic SFT: Quora-Arabic-GPT4 (42,862) + CIDAR (9,578) + SmolTalk2 English (35,367)
= 87,807 examples, 60/40, 1x each (no upsampling). SmolKalam dropped (translated).
Trained ar-cpt-native-sft + ar-base-native-sft (1372 steps, 2 epochs). Full eval on all
Arabic benchmarks (jobs 248432-436; amthal re-run 248604/605 after a filename-bug fix).

## HEADLINE: native data recovers Arabic cultural grounding lost by translated SmolKalam
Under matched Idiom-CPT init, native vs SmolKalam-only +IT:
- Kinayat-meaning 19.1 -> 35.1 (+16.0)
- ArabCulture 46.2 -> 49.8 ; ArabicCulturalQA 66.8 -> 70.1 ; ArabicMMLU 66.1 -> 67.7
- Alyah 33.7 -> 37.5 ; DziriEval 29.8 -> 34.7 ; Ar-Figurative 30.9 -> 34.4 ; GlobalPIQA-ar 58.1 -> 60.9
- Kinayat-cloze 56.0 -> 54.7 (~tie) ; amthal in-domain PPL 9.6 -> 8.5 / BPB 0.605 -> 0.574
- English retained/better: MMLU 77.1, BoolQ 89.2, GSM8K 89.8, HumanEval(chat) 71.3 -> 85.4, WikiText 14.0/0.878
Cross-init survival (native cpt vs base) holds on EVERY idiom/cultural task (e.g. Kinayat-meaning
35.1 vs 18.8, amthal BPB 0.574 vs 0.726).

## Paper changes (colm2026_conference.tex)
Arabic +IT column swapped SmolKalam -> native (all rows). §IT: setup notes Arabic native mixture
(~88K, Quora+CIDAR); cross-init numbers updated to native; new paragraph "Native instruction data
lifts Arabic cultural grounding" (translated->native deltas); caveat 45.2->35.1 (native) vs ->19.1
(translated); Arabic caption notes +IT recipe. Verified 0 non-ASCII, 5/5 tabulars, cols match, refs
resolve. NOT pushed (auto-commit hook). Bug fixed: eval_ar_sft_full.slurm amthal filename
(ar_amthal -> ar_amthal_heldout, output ppl_ar_amthal).
