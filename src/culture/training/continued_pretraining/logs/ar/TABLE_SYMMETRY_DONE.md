# Table symmetry + chat-template eval — COMPLETE (2026-08-27)

All eval jobs finished; paper tables assembled & verified (0 non-ASCII, 5/5 tabulars,
all column counts match spec, refs resolve).

## What was assembled
- **Arabic table**: added category (1) English retention (MMLU/BoolQ/GSM8K/HumanEval/
  WikiText PPL+BPB) and category (2) amthal in-domain PPL/BPB (new idiom-specific probe,
  analog of hi-proverbs/zh-chengyu). Amthal shows Idiom-CPT best (PPL 7.1, BPB 0.526).
- **Chinese +IT column** filled: MC loglik (Chengyu-Bench 95.2, ChID 63.8, CMMLU 79.2,
  CCPM 81.6, MMLU 75.3, BoolQ 88.5) + chat generative (GSM8K 88.5, HumanEval 67.7).
  LM/WikiText +IT stay dashed (not collected for SFT ckpt; noted in caption).
- **Hindi +IT generative** switched to chat: IdiomCE 1.70, GSM8K 89.6, HumanEval 68.9.
- **Appendix** table now has Docs|Tokens per language (done earlier).
- Captions + §IT/abstract/conclusion prose updated: +IT generative = native chat
  template (nothink); +IT multiple-choice = shared base-loglik protocol.

## KEY FINDING (chat vs raw for +IT generative)
Mixed, and it VALIDATES using chat: Chinese HumanEval raw 57.3 -> chat 67.7 (+10.4,
raw badly under-measured the instruct model); hi/ar HumanEval chat slightly BELOW raw
(hi 72.6->68.9, ar 73.8->71.3, completion-friendly). IdiomCE hi: chat 1.70 vs raw 1.82
(both agree IT reduces idiomatic rendering to ~9% -> a real effect, not artifact).

## zh IT status
zh SFT done & in table (+IT). Cross-init survival holds on the idiom axis
(chengyu_bench cpt-init 95.2 vs base-init 90.0) but NOT general-cultural (cmmlu/ccpm
favor base-init slightly) -> abstract/conclusion keep the strong "every benchmark"
survival claim scoped to hi/ar; zh reported without over-claiming.

## New assets (CIFL src/culture/evaluation/)
scorer.py _apply_chat_template(enable_thinking=False) + generate(chat=); run_eval.py
--chat; run_english_gen.py --chat + _extract_code (HumanEval fence extraction);
eval_gen_chat.slurm; eval_idiomce_chat.slurm; eval_ar_lm_backfill.slurm;
data/eval/ar/ar_amthal_heldout.jsonl. Chat outputs in eval/<lang>/<run>/{en_gen_chat,idiomce_chat}.
