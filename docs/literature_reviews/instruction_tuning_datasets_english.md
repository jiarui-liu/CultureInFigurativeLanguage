# General-Purpose English Instruction-Tuning / SFT / Post-Training Datasets — A Literature Review

> **Scope.** This review covers **general-purpose English instruction-tuning data** — SFT mixtures, single-pipeline synthetic generators, human demonstrations, real user logs, verifiable-domain (math/code) SFT, long-CoT reasoning distillation, instruction-following/constraint data, and (secondarily) preference/RLHF data — plus the **data-selection literature** that tells us how to pick a subset. It is written for this project's next stage: after continued pretraining of a **Qwen-3.5-9B base** on idiom-tagged C4/mC4, we need an English SFT stage that serves both as the *general-capability backbone* and as the *anchor/control language* against zh / hi / ar arms.
>
> **Method.** Candidate datasets were found by web search over arXiv, HuggingFace dataset cards, GitHub and model technical reports; per-dataset facts (sizes, teachers, licenses, gating) were verified against primary sources by dedicated sub-agents. Every license and gating status quoted below was additionally checked against the live HuggingFace API. Row-level quality was assessed by **reading real rows myself** — see the sampling method immediately below. Datasets that could not be accessed are flagged rather than guessed at (see *Access Limitations*).

---

## ⭐ Hands-On Quality Ranking — Deep Sampling

**Sampling method.** Rows were read directly from each dataset's auto-converted Parquet branch (`hf://datasets/<id>@~parquet/<config>/<split>/*.parquet`) over `HfFileSystem`, addressing **row groups at ~0%, ~40–50% and ~90–95% of the corpus by byte offset** — i.e. genuinely deep positions, not the dataset-viewer's first page. Two rows were rendered per depth, with all columns shown. This route was chosen after the HF dataset-viewer API (`datasets-server`) hard-rate-limited anonymous access at 500 requests / 300 s (the machine's `HUGGINGFACE_HUB_TOKEN` is **expired**, so all access was anonymous). About 30 datasets were sampled at 3 depths each. An initial pass over `allenai/tulu-3-sft-mixture`, `teknium/OpenHermes-2.5`, `allenai/tulu-v2-sft-mixture` and `HuggingFaceTB/smoltalk` used the dataset-viewer's **absolute row offsets** (0 / 2,000 / 20,000 / 200,000 / 900,000) before the migration, and the Tulu-3 composition figures below come from that API's `/statistics` endpoint on the live `source` column. Judged on: answer depth and correctness, formatting noise and schema cruft, GPT-isms and refusal boilerplate, prompt-distribution realism, duplication, contamination risk, and **whether the tail is worse than the head**. Datasets that are gated (Infinity-Instruct, Nemotron-v2, LIMA, LMSYS-Chat-1M) were **not** sampled and are excluded from the ranking rather than guessed at.

### Tier 1 — Excellent

**1. `HuggingFaceTB/smoltalk2` (SFT config) — Excellent.** The most relevant corpus in this review for a Qwen-class target, because its reasoning half is distilled from **Qwen3-32B** rather than from GPT-4 — teacher/student distributional fit is exactly what GRAPE (arXiv:2502.04194) says to optimise. 3,383,242 SFT examples / **19,292.4 M tokens** with a published per-subset weight table you can copy directly. Multi-turn data has *real* conversational structure — the user pushes back, changes language, narrows the ask:
> `[user, ~90% depth]` "For this question, I was hoping the response would be a little more concise. You previously listed some general advice for a programmer just beginning to learn. Among these, which are the most important?"
> **Deep-offset:** stable; the 90% rows are indistinguishable in quality from the 0% rows.

**2. `HuggingFaceTB/smoltalk` — Excellent.** 1,043,917 train rows, English-only, `source` column on every row. Math shows real derivations; `smol-magpie-ultra` is genuinely multi-turn; `smol-constraints` is IFEval-shaped and directly useful:
> `[user]` "Your response should contain at least 3 sentences. Your answer must contain exactly 3 bullet points… Include keywords friendship, trust, and loyalty."
> `[user, ~85% depth]` "You are a tour guide in Alaska on a boat in Prince William sound…" — role-play sustained over five turns without breaking character.
> **Deep-offset:** no degradation across 0% / 40% / 85%.

**3. `HuggingFaceH4/no_robots` — Excellent (but small and NC).** 9,500 train rows of human-written demonstrations, and it shows: constraints in the prompt are honoured exactly, tone is natural, and there is not one "As an AI language model" in anything I read.
> `[user]` "Explain the internet to someone as if it's the 1970s. Keep a casual tone, and make it a maximum of two paragraphs." → "Okay, so you know how we have television, and we send each other letters, and we call each other on the phone?…"
> **Deep-offset:** flat quality end to end. **Caveat: `cc-by-nc-4.0` — non-commercial.**

**4. `nvidia/OpenMathInstruct-2` — Excellent (in-domain).** Clean LaTeX derivations, no chatter, ground-truth answer in a separate `expected_answer` column, and a `problem_source` column that lets you slice precisely. 14M pairs over ~600K unique questions.
> **Deep-offset:** *composition* shifts with depth — `augmented_gsm8k`/`augmented_math` at the head, raw **`math`** (verbatim MATH-train items) at 95%. Quality does not drop, but **contamination risk concentrates in the tail**; filter on `problem_source` if you report MATH.

**5. `nvidia/HelpSteer2` / `HelpSteer3` — Excellent (preference).** Human-annotated, five-axis integer labels, CC-BY-4.0, no proprietary-output dependency in the labels. Annotators visibly punish refusal (a refusal to "simulate a Twitter thread" scores `helpfulness 0, correctness 0`). HelpSteer3 is 28 languages, so filter `language` for an English control.

**6. `bigcode/self-oss-instruct-sc2-exec-filter-50k` — Excellent (provenance).** 50,661 rows, ODC-BY, **no proprietary model anywhere in the chain** — StarCoder2-15B self-instructs from The Stack and filters by execution. Instructions are precise and self-contained; responses are short, correct and idiomatic. Some solutions are naive (an O(n²) rectangle-collision check) but they are honest and they run. **Deep-offset:** uniform.

### Tier 2 — Good

**7. `Magpie-Align/Magpie-Pro-300K-Filtered` and `argilla/magpie-ultra-v1.0` — Good.** Remarkably uniform at depth (0% → 95% indistinguishable), well-structured answers, and magpie-ultra ships per-row `difficulty`, `quality`, `reward_model_score` and `category` — the most *selectable* corpus here. Two weaknesses: the self-generated prompt distribution is encyclopaedic-explainer-shaped ("What is standardized testing in education?", "Explain Putin's long-term strategy"), and the Llama-3 GPT-isms are relentless:
> `[assistant, ~95% depth]` "**The age-old quest for a tidy and efficient email inbox!** Here are some best practices…"
> **License caveat:** Llama-3/3.1-derived; `Magpie-Pro` is tagged `license: llama3`, magpie-ultra and Magpie-Air have **no license field at all.**

**8. `open-thoughts/OpenThoughts3-1.2M` — Good→Excellent (reasoning).** Exactly 1,200,000 rows, Apache-2.0, QwQ-32B teacher, `domain` and `source` columns. `<think>` bodies are long and genuinely exploratory. **Deep-offset:** composition drifts hard — code (`stackexchange_codegolf`, `nvidia/OpenCodeReasoning`) at 0%, almost pure `ai2-adapt-dev/openmath-2-math` at 50% and 90%. If you subsample uniformly you will get mostly math. Rows are very long; budget context accordingly.

**9. `allenai/tulu-3-sft-mixture` — Good, with a language caveat that disqualifies it as-is.** 939,343 rows, ODC-BY, superbly documented (I pulled the exact 19-source composition from the live column statistics). But it is **substantially non-English by construction**: `aya_100k` (100,000 rows) plus `wildchat_100k` (100,000) plus the OASST1 block are heavily multilingual — ~22% of the mixture.
> `[user, row 900,000]` "A cikin wanne nau'in ra'ayi za ku rarraba tweet mai zuwa?…" (Hausa)
> `[user, row 200,000]` "Repeat this string \"coffee and flower\"" → `[assistant]` "coffee and flower"
> **Deep-offset:** quality is fine throughout; *language* is the problem. Excellent general post-training mixture, unusable as an English control without filtering.

**10. `allenai/tulu-3-sft-personas-instruction-following` — Good (IF).** 29,980 rows, ODC-BY, English-only, with an explicit `constraints` column naming the IFEval-style constraint types applied. Prompts are varied and the responses actually satisfy the constraints:
> `[user]` "Summarize the main teachings of the Sermon on the Mount in 8 sentences. Each sentence must contain the word 'blessed' exactly twice. End your summary with the phrase 'Peace be with you!'"
> **Deep-offset:** uniform.

**11. `AI-MO/NuminaMath-CoT` — Good, needs source filtering.** 859,594 rows, Apache-2.0, mostly solid Olympiad and `synthetic_math` derivations. Two real defects: the `cn_k12` block contains **items that are not math at all**, and the schema duplicates `problem`/`solution` verbatim into `messages`, roughly doubling storage.
> `[source]` cn_k12 · `[problem]` "—I'd like some more cheese. —Sorry, there's \_\_\_\_ left. A. some B. none C. a little D. few" → "**Answer** B…"
> Tulu 3's own audit had to strip **11.3%** of NuminaMath-TIR/MATH for MATH-eval overlap.

**12. `open-r1/Mixture-of-Thoughts` — Good, but enormous per row.** 349,317 verified DeepSeek-R1 traces with a `num_tokens` column. The two code rows I drew were **22,185 and 19,066 tokens**. At our current 16K `cutoff_len` these truncate mid-`<think>`, which is worse than dropping them. **No license field on the card.**

**13. `teknium/OpenHermes-2.5` — Good but dated.** 1,001,551 rows, clean two-turn format, `source`/`category` columns that make filtering easy. Answer depth is 2023-era, and the `GPT-4 Comparison Data` block at 40% depth is visibly thin (bare listicles, no reasoning). 12 of its 16 columns are `None` for most rows. **No license field at all**, and overwhelmingly GPT-4-distilled.

**14. `simplescaling/s1K-1.1` and `GAIR/LIMO` — Good (tiny reasoning sets).** Genuinely hard problems (an infinite-dimensional Hilbert-space item in s1K-1.1), long first-person reasoning, clean short answer fields. MIT and Apache-2.0 respectively. Note: at 817–1,000 rows each is a single Parquet row group, so **"deep offset" is not meaningfully testable** — I report that rather than pretending otherwise.

**15. `databricks/databricks-dolly-15k` — Good as a style anchor, weak as capability data.** 15,011 genuinely human rows, refreshingly short answers, but the `category` labels are unreliable ("What are some movies about Artificial Intelligence…" is labelled `classification`) and `context` fields are pasted Wikipedia. **`cc-by-sa-3.0` — ShareAlike is viral.**

**16. `Skywork/Skywork-Reward-Preference-80K-v0.2` — Good (preference), with visible artifacts.** 77,016 pairs, decontaminated against RewardBench (4,957 pairs removed), `source` column on every row so you can see the remix (`magpie_ultra`, `magpie_pro_llama3.1`, `wildguard`, HelpSteer2, OffsetBias). Chosen/rejected are usually genuinely separable on code and factual tasks. But the deep tail is all `wildguard` jailbreak pairs, and one "chosen" response I drew contains a degenerate loop:
> `[chosen]` "As a responsible and helpful AI language model, I cannot fulfill this request as it goes against my programming **rules rules rules rules** to provide harmful or dangerous content…"
> **Deep-offset:** composition shifts from code/knowledge to safety. **No license field on the card.**

**17. `Anthropic/hh-rlhf` — Good (clean), but old and format-awkward.** 169,352 comparisons, MIT, human-annotated. Rows are flat `\n\nHuman: … \n\nAssistant: …` strings rather than structured messages, and chosen/rejected usually share a long identical prefix and diverge only at the last turn. The model responses are 2022-era and weak:
> `[chosen]` "Is iced coffee just regular coffee with ice in it?" → "Well, that depends! Is it iced coffee with ice, or iced coffee with regular coffee and ice?…" vs `[rejected]` "Yes, iced coffee is just coffee mixed with ice."
> Note the harmless-base split leads with a prompt asking for cuss words where the *chosen* response is the explicit list — a deliberate artifact of the helpful/harmless split design, not a bug, but it will surprise anyone who concatenates the splits blindly.

**18. `open-r1/OpenR1-Math-220k` — Good (reasoning math), with a messy `answer` field.** 450,258 rows across splits, Apache-2.0, DeepSeek-R1 traces verified by Math-Verify. Problems are largely translated Eastern-European olympiad and textbook items, and the `<think>` bodies are genuinely exploratory. The defect is answer normalisation — several `answer` fields are malformed:
> `[answer]` `v_{R}=4\mathrm{~}/\mathrm{},v_{B}=10\mathrm{~}/\mathrm{}` and `[answer]` `x+y+1orx+y+-3`
> If you plan to do RLVR or answer-matching downstream, these will not parse. **Deep-offset:** uniform in quality; the malformed answers appear at every depth.

**19. `bespokelabs/Bespoke-Stratos-17k` — Good, but it commits you to a format.** 16,710 rows, Apache-2.0, DeepSeek-R1 traces. Every row carries the same long system prompt and wraps reasoning in `<|begin_of_thought|>` … markers rather than `<think>`:
> `[system]` "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions…"
> Math prompts are prefixed "Return your final response within `\boxed{}`"; the code tail switches to "Generate an executable Python function… take stdin as input and print the output." Mixing this with `<think>`-style data (OpenThoughts3, Mixture-of-Thoughts, SmolTalk2) means training two incompatible reasoning delimiters at once — pick one and normalise.

**20. `theblackcat102/evol-codealpaca-v1` — Good but noisy.** 111,272 rows, Apache-2.0, GPT-4 teacher. Solutions are correct and reasonably explained, but the prompts inherit real garbage from the Evol pipeline — one row I drew feeds the model OCR-mangled Python:
> `[instruction]` "i've got this python code from an ocr tool, but it's not working. can you debug it for me? `class 5olv7!on: oet tindShort€stPatn($elf` nn&s7er: 'GridM&s+er') -< 1rt:` …"
> Arguably that is *useful* robustness data; it is certainly not clean. **Deep-offset:** uniform.

**21. `OpenAssistant/oasst2` — Good (human), but overwhelmingly not English.** 128,575 rows in the train parquet, Apache-2.0. Note the schema is **one row per message**, not per conversation — you must reassemble trees yourself. Rows carry a `lang` column and it earns its keep: the head is Spanish, the middle is French, and only the deep tail I sampled was English. Filter on `lang == "en"` and expect a small fraction of the total.

**22. `CohereLabs/aya_dataset` — Good (human), and essentially *not* an English resource.** 202,362 train rows, Apache-2.0, genuinely human-written. Every depth I sampled was non-English — Somali song lyrics, French trivia, Gujarati geography, Standard Malay, Hausa demographics, Plateau Malagasy idiom explanation:
> `[inputs]` "Amin'ny fotoana inona no tokony hilaza ilay teny hoe 'mamangy amin'ny fahoriana'?" `[language]` Plateau Malagasy
> This is exactly the right asset for the zh / hi / ar arms and for cultural-idiom work, and exactly the wrong asset for the English control. It is also *why* Tulu-3 is 22% non-English.

### Tier 3 — Mixed

**23. `HuggingFaceH4/ultrafeedback_binarized` — Mixed (preference).** Two visible defects in the rows I read. Ties are not filtered — the first rows I drew had `score_chosen == score_rejected` (8.5/8.5, 6.5/6.5), carrying no preference signal. And the rejected side is dominated by refusal boilerplate, so DPO on it partly just teaches non-refusal:
> `[rejected]` "As an AI language model, I cannot personally develop habits for you…"
> `[rejected]` "Hello! I'm here to help… However, I noticed that your question doesn't make sense. concatenate strings in python is not a valid or appropriate question." (3.0 vs chosen 8.5)
> **Deep-offset:** separation improves at depth. Usable *after* dropping ties.

**24. `meta-math/MetaMathQA` — Mixed.** 395,000 rows, MIT, with a `type` column (`MATH_AnsAug`, `GSM_Rephrased`, `GSM_SV`, `MATH_SV`, …) that is essential, because the `*_SV` (self-verification) rows are degenerate — the answer is stated inside the question and the "solution" restates it:
> `[type] MATH_SV` · `[query]` "…If $y$ is X when $x$ is 2, then find $x$ when $y$ is 4000. **The answer is 10.** What is the value of unknown variable X?" → `[response]` "The answer is: 10"
> Reasoning elsewhere is short and occasionally sloppy (one `MATH_AnsAug` row computes $\sqrt{72}$ on the way to a question about $\sqrt{73}$). **Deep-offset:** the degenerate `*_SV` rows concentrate in the tail. Superseded by OpenMathInstruct-2.

**25. `allenai/llama-3.1-tulu-3-8b-preference-mixture` — Mixed (preference).** 272,898 pairs with a useful `source` column (`tulu-3-sft-reused-if`, `tulu-3-sft-reused-off-policy-8b`, `tulu-3-persona-if`). The IF-constraint pairs are well-separated (a lowercase-only constraint honoured vs ignored; a JSON to-do list with clean task strings vs one with stray `"- "` prefixes). But **it inherits Tulu 3's multilingualism** — the second row I drew answers an English prompt entirely in Chinese on both sides — and both the completions and the GPT-4o judge labels are proprietary-model-derived.

**26. `Open-Orca/SlimOrca` — Mixed.** 517,982 GPT-4-completion rows with the 16 Orca system messages. The system messages do produce more explanation than plain FLAN, but the underlying tasks are still FLAN-shaped extraction/classification, and the answers swing from an entire fabricated news article to a one-line entity lookup. Wholly GPT-4-derived under an MIT tag.

**27. `ise-uiuc/Magicoder-OSS-Instruct-75K` — Mixed.** Solid multi-language (C++/Python/Rust) problem-solution pairs grounded in real Stack code, but **every row literally carries an `openai_fingerprint` column** (`fp_eeff13170a`) — the provenance is in the schema. The card itself tells you to "pay attention to OpenAI's usage policy". Prefer StarCoder2-Instruct where possible.

**28. `WizardLMTeam/WizardLM_evol_instruct_V2_196k` — Mixed, with genuine tail rot.** The file on the Hub is 143,000 rows, not 196k. The head is real Evol-Instruct (multi-constraint Excel-macro and role-play prompts); by **95% depth the rows carry `alpaca_*` ids and collapse to raw Alpaca quality**:
> `[idx] alpaca_31156` · `[human]` "Name a book by J.K. Rowling." → `[gpt]` "Harry Potter and the Philosopher's Stone."
> This is the clearest deep-offset degradation I found. **License is contradictory** (HF `mit` vs GitHub CC-BY-NC-4.0).

**29. `cognitivecomputations/dolphin-r1` (`nonreasoning`) — Mixed.** Multi-turn chat and code with realistic follow-ups ("Can you straightforwardly derive that 18/1140 reduces to 3/190 without a calculator?"). But the quality-metadata columns are mostly empty — `score` is `None` on most rows I drew, and `compliance_rating` and `overall_quality` were `None` on all of them — so the advertised filterability is not actually there. Its sibling reasoning splits are one-third **Gemini-2.0-Flash-Thinking**-derived under a blanket Apache-2.0 tag.

**30. `berkeley-nest/Nectar` — Mixed, and license-hostile.** 3.8M GPT-4-ranked comparisons, but the prompt pool is visibly LMSYS-derived — redacted `NAME_1` / `NAME_2` placeholders and raw YouTube-transcript dumps with timestamps. Its card carries an explicit **"not used to compete with OpenAI"** condition. Do not use.

### Tier 4 — Poor / Do Not Use

**31. `HuggingFaceH4/ultrachat_200k` — Poor for a 2026 mixture, and the tail is worse than the head.** The "Assistance on Existing Materials" third dominates at depth, and the "material" is C4 web sludge: Shopify theme documentation, a pre-school's marketing brochure, a local-news item about a police-charity impersonator. Answers frequently restate the input:
> `[user, ~90% depth]` "Apple Tree's Pre-School program continues its best practice by combining children ages 3-5 years old…" → `[assistant]` "Apple Tree's Pre-School program combines children ages 3-5 years old allowing for small group sizes…"
> Follow-up turns are template-generated ("Can you add more…?", "Can you tell me more about…?") rather than conversational. **Deep-offset: degrades.** Its only remaining value is multi-turn *format*.

**32. `allenai/tulu-v2-sft-mixture` — Poor (superseded).** Sampled at 0% / 40% / 85% and each third is a different obsolete artifact: raw FLAN v2 templates at the head ("1. Miguel / 2. Yes / 3. Eight / 4. unknown…"), Code-Alpaca one-shot snippets with no explanation or code fence at 40%, GPT-3.5-flavoured WizardLM at 85%. Training a 2026 base model on the FLAN and Code-Alpaca thirds would make it answer *worse*.

**33. `anon8231489123/ShareGPT_Vicuna_unfiltered` — Do not use.** Not ranked on quality: the provenance is an anonymous scrape of a site that disabled public access precisely to stop it, re-uploaded under an Apache-2.0 tag the uploader had no standing to apply, with an acknowledged PII risk and no audit. No license tag fixes that.

**34. `tatsu-lab/alpaca` / `sahil2801/CodeAlpaca-20k` — Do not use.** Historically foundational, obsolete as data. The Self-Instruct authors' own audit found **~46% of a 200-sample manual check "may have problems"**; Alpaca is `cc-by-nc-4.0` *and* text-davinci-003-derived.

### Sampled but not ranked
`nvidia/OpenCodeReasoning` (`split_0`), `ConiferLM/Conifer`, `Open-Orca/OpenOrca`, `nvidia/Llama-Nemotron-Post-Training-Dataset` (`SFT/chat`) and `nvidia/Nemotron-Post-Training-Dataset-v1` (`SFT/chat`) returned path/404 errors on the auto-converted Parquet branch in this environment — the last of these has no `~parquet` tree at all (*"Entry Not Found for url: …/tree/~parquet"*), i.e. the auto-conversion has not run for that repo. Their facts below come from cards and papers, not from rows I read. `allenai/WildChat-1M` was reached (14 shards, 3,360,836,020 bytes) and its head rows are exactly what the paper promises — long, unfiltered, idiosyncratic real user prompts ("Are you familiar with reality shifting? So, I'm refining a foolproof method for reality shifting and want to pick a destination…") — but it is excluded from the ranking because its *responses* are GPT-3.5/GPT-4 output, which makes it a prompt source rather than an SFT corpus for our purposes.

---

## Recommendation for This Project

**Constraints that drive the choice.** (a) The base is **Qwen-3.5-9B**, so per GRAPE (arXiv:2502.04194) *Qwen-distilled data fits better than GPT-4-distilled data* — this is not a stylistic preference, it is a measured 3–13% effect. (b) English is the **anchor/control**, so the mixture must be *verifiably English-only*; the most popular mixture in the field (Tulu 3) is ~22% non-English and would silently contaminate the control. (c) Our CPT config runs at `cutoff_len: 16384`, so 20k-token reasoning traces truncate and must be filtered on length, not sampled blind. (d) We want the option to release, so license provenance is a first-class filter.

### Proposed mixture — Variant A: "best capability" (~305k examples, ~330M tokens)

| # | Dataset | Rows to take | Of | Why | License |
|---|---|---|---|---|---|
| 1 | `HuggingFaceTB/smoltalk2` — `SFT`, English `no_think` splits (smol-magpie-ultra, smol-summarize, smol-rewrite, systemchats, explore-instruct-rewriting, everyday-conversations) | **120,000** | 622,814 | General-chat backbone; multi-turn with real user pushback; **Qwen3-32B**-aligned formatting | no field on card; new subsets stated Apache-2.0; Llama-3.1-405B origin on the magpie portion |
| 2 | `HuggingFaceTB/smoltalk2` — `SFT`, `multi_turn_reasoning_if_think` + `smoltalk_systemchats_Qwen3_32B_think` | **55,653** | 55,653 | Reasoning-mode + instruction-following in Qwen3 style; take all | as above |
| 3 | `allenai/tulu-3-sft-personas-instruction-following` | **29,980** | 29,980 | The only good open IFEval-style **training** set; English-only; take all | **ODC-BY** ✅ |
| 4 | `open-thoughts/OpenThoughts3-1.2M`, filtered to ≤8k response tokens, `domain`-balanced | **40,000** | 1,200,000 | Long-CoT reasoning from **QwQ-32B** — the cleanest permissive reasoning corpus | **Apache-2.0** ✅ |
| 5 | `nvidia/OpenMathInstruct-2`, `problem_source ∈ {augmented_gsm8k, augmented_math}` only | **25,000** | 14 M | Math without the verbatim-MATH tail that carries the contamination risk | **CC-BY-4.0** ✅ |
| 6 | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | **20,000** | 50,661 | Execution-verified code with a completely clean provenance chain | **ODC-BY** ✅ |
| 7 | `HuggingFaceH4/no_robots` | **9,500** | 9,500 | Human style anchor — teaches brevity and constraint compliance nothing else here teaches | ⚠️ **CC-BY-NC-4.0** |
| 8 | `OpenAssistant/oasst2`, `lang == "en"` only | **~4,000** | 128,575 | Genuinely human multi-turn trees, Apache-2.0 | **Apache-2.0** ✅ |
| 9 | `databricks/databricks-dolly-15k` | **~2,000** | 15,011 | Short-answer human demonstrations (counterweight to synthetic verbosity) | ⚠️ **CC-BY-SA-3.0** (viral) |

**Total ≈ 306,000 examples.** At the SmolTalk2 card's measured averages (≈460–1,520 tokens/example for the no-think splits) this lands around **300–350M tokens**, i.e. ~2 epochs is ~700M tokens — comparable to the 1.4B-token CPT run and entirely tractable on the existing 32-GPU setup at `cutoff_len: 16384`.

### Variant B: "license-clean release" (~150k examples)
Drop rows 1, 2, 7 and 9 (unlicensed-or-Llama-derived, CC-BY-NC, CC-BY-SA) and backfill the general-chat slot from permissively-licensed sources:
- `open-thoughts/OpenThoughts3-1.2M` **60,000** (Apache-2.0) — reasoning and general
- `nvidia/OpenMathInstruct-2` **30,000** (CC-BY-4.0)
- `bigcode/self-oss-instruct-sc2-exec-filter-50k` **30,000** (ODC-BY)
- `allenai/tulu-3-sft-personas-instruction-following` **29,980** (ODC-BY)
- `OpenAssistant/oasst2` English **~4,000** (Apache-2.0)
- `CohereLabs/aya_dataset` — **not** for the English arm: every depth I sampled was non-English (Somali, Gujarati, Hausa, Malagasy…), so the English yield is small. Reserve it, in full, for the zh / hi / ar arms
- `nvidia/Nemotron-Post-Training-Dataset-v1` `chat` split, sub-sampled **~10,000** (CC-BY-4.0, DeepSeek-R1-0528 / Qwen3-235B teachers)
Everything here is ODC-BY / Apache-2.0 / CC-BY-4.0 with permissive teachers. The cost is a weaker general-chat register, which is why Variant A exists.

### What to drop, and why
- **`allenai/tulu-3-sft-mixture` as a whole** — ~22% non-English by construction (aya_100k 100,000 + wildchat_100k 100,000 + OASST1). Excellent mixture, wrong tool for an English control. *Cherry-pick* `personahub_ifdata` (29,980) and `flan_v2_converted` if wanted; do not take the whole repo.
- **`HuggingFaceH4/ultrachat_200k`** — the C4-grounded third degrades at depth into restating marketing copy, and follow-up turns are templated. No unique capability we cannot get elsewhere.
- **`allenai/tulu-v2-sft-mixture`, `tatsu-lab/alpaca`, `sahil2801/CodeAlpaca-20k`, `anon8231489123/ShareGPT_Vicuna_unfiltered`** — obsolete answer quality; ShareGPT additionally has unclean provenance.
- **`WizardLMTeam/WizardLM_evol_instruct_V2_196k`** — real tail rot into raw Alpaca quality, plus a MIT-vs-CC-BY-NC-4.0 license contradiction.
- **`berkeley-nest/Nectar`** — the card itself forbids using it to compete with OpenAI; the prompt pool is LMSYS-derived with `NAME_1` redactions and YouTube-transcript noise.
- **`AI-MO/NuminaMath-CoT` unfiltered** — the `cn_k12` block contains non-math items, and Tulu 3 had to strip 11.3% of the TIR/MATH variant for eval overlap. Use only if you filter `source` and run decontamination.
- **`open-r1/Mixture-of-Thoughts` unfiltered** — rows of 19k–22k tokens will truncate at our 16K context. Filter on the provided `num_tokens` column or skip.
- **`a-m-team/AM-DeepSeek-R1-Distilled-1.4M`** — `cc-by-nc-4.0`, outright non-commercial.

### Preference stage (if we run DPO after SFT)
Use **`nvidia/HelpSteer2`** (21,362, CC-BY-4.0, human-annotated, five axes) as the primary signal, optionally extended with **`nvidia/HelpSteer3`** filtered to `language == "english"` (of 132,937 across 28 languages). Avoid `ultrafeedback_binarized` unless you first drop the score ties I found in the head rows, avoid `Nectar` entirely, and treat `allenai/llama-3.1-tulu-3-8b-preference-mixture` (272,898) as research-only — both its completions and its GPT-4o judge labels are proprietary-model-derived.

### Selection procedure to run on top of the mixture
Per the data-selection literature (see §7), do not just concatenate:
1. **Superfiltering** (arXiv:2402.00530) — compute IFD with a GPT-2-sized model over the whole pool (~8 min per Alpaca-scale set) and drop the low-difficulty tail.
2. **Embedding dedup** at DEITA's τ = 0.9 (arXiv:2312.15685) — this is the step that actually makes small mixtures work.
3. **GRAPE** (arXiv:2502.04194) — where several sources answer near-identical prompts (they will: magpie-ultra vs smol-magpie-ultra vs OpenHermes), keep the response our own Qwen-3.5-9B base scores highest.
4. **Tulu-3 decontamination** (arXiv:2411.15124 §3.2) — 8-gram prompt matching, flag a test instance if >50% of its tokens match one training instance, drop a training set if >2% of any eval overlaps. Run this against every benchmark we intend to report, and expect to lose ~3–11% of the math and code rows.
5. **Search the subset size.** Every paper in §7 found an interior optimum; assume ours is smaller than 306k and verify.

### License-safety summary (the flags that constrain release)
| Risk | Datasets |
|---|---|
| **OpenAI-output-derived** (ToS overhang regardless of the HF tag) | OpenHermes-2.5, OpenOrca/SlimOrca, Dolphin, UltraChat/ultrachat_200k, WizardLM Evol-Instruct, Alpaca, CodeAlpaca, Self-Instruct, Magicoder-OSS-Instruct (rows literally carry an `openai_fingerprint` column), evol-codealpaca-v1, NuminaMath-TIR, Conifer, UltraFeedback, Nectar, Tulu-3 preference mixture, Tulu-3's WildChat-GPT-4 and persona subsets, WildChat, LMSYS-Chat-1M |
| **Google/Gemini-output-derived** | `cognitivecomputations/dolphin-r1` (its ~300k Gemini-2.0-Flash-Thinking third) |
| **Llama-license-derived** | Magpie-Pro (`license: llama3`), Magpie-Air, magpie-ultra, SmolTalk's Smol-Magpie-Ultra, OpenMathInstruct-2 (Llama-3.1-405B teacher), parts of Llama-Nemotron |
| **Non-commercial** | No Robots (CC-BY-NC-4.0), Alpaca (CC-BY-NC-4.0), LIMA (CC-BY-NC-SA or stricter), AM-DeepSeek-R1-Distilled-1.4M (CC-BY-NC-4.0), WizardLM Evol-Instruct per its GitHub, Tulu v1/v2's GPT4-Alpaca + Code-Alpaca subsets, Open-Platypus's ScienceQA/ReClor subsets |
| **Viral copyleft** | Dolly-15k (CC-BY-SA-3.0), Infinity-Instruct (CC-BY-SA-4.0), the StackOverflow-derived CC-BY-SA subsets of Llama-Nemotron |
| **Explicit non-compete clause on the card** | Nectar ("not used to compete with OpenAI") |
| **No license stated at all** — treat as unresolved, *not* permissive | OpenHermes-2.5, magpie-ultra-v1.0, Magpie-Air-300K-Filtered, Mixture-of-Thoughts, Skywork-Reward-Preference-80K-v0.2, SmolTalk, SmolTalk2 |
| **Clean for release** ✅ | Tulu-3-personas-IF (ODC-BY), StarCoder2-Instruct (ODC-BY), OpenThoughts family (Apache-2.0), OpenR1-Math-220k (Apache-2.0), OASST1/2 (Apache-2.0), Aya (Apache-2.0), OpenMathInstruct-2 (CC-BY-4.0, modulo the Llama teacher), HelpSteer2/3 (CC-BY-4.0), Nemotron-Post-Training v1 (CC-BY-4.0), hh-rlhf (MIT), MetaMathQA (MIT), s1K-1.1 (MIT), LIMO (Apache-2.0) |

**Bottom line:** Variant A if this is an internal research artifact; Variant B if we intend to publish weights. In both cases the two non-negotiables are **filter to English before calling anything a control** and **run Tulu-3-style decontamination before reporting any benchmark**.

---

## Taxonomy of the Space

- **(A) Curated open mixtures** — someone else already did the blending, decontamination and formatting; you consume one repo. *Tulu v1/v2/v3, OpenHermes-2.5, Infinity-Instruct, SmolTalk / SmolTalk2, Llama-Nemotron-Post-Training, Nemotron-Post-Training v1/v2, OpenThoughts-114k / OpenThoughts2-1M / OpenThoughts3-1.2M, OpenOrca / SlimOrca, Dolphin / Dolphin-R1, Open-Platypus.*
- **(B) Single-pipeline synthetic generators** — one algorithm, one teacher, one repo. *Self-Instruct → Alpaca, WizardLM Evol-Instruct, UltraChat, Magpie (Pro / Air / Ultra), Genstruct (a model, not a dataset), Orca-style system-message distillation.*
- **(C) Human-written demonstrations** — expensive, small, stylistically clean. *Dolly-15k, OASST1/2, No Robots, LIMA, Aya human annotations.*
- **(D) Real user logs** — genuine prompt distribution, messy responses, PII/toxicity baggage. *ShareGPT, WildChat-1M / 4.8M, LMSYS-Chat-1M.*
- **(E) Verifiable-domain SFT (math / code)** — seeds with ground truth, so rejection sampling works. *OpenMathInstruct-1/2, MetaMathQA, NuminaMath, Magicoder/OSS-Instruct, Evol-CodeAlpaca, StarCoder2-Instruct, OpenCodeReasoning, CodeAlpaca.*
- **(F) Long-CoT reasoning distillation** — post-R1 traces with `<think>` style bodies. *OpenR1-Math-220k, Mixture-of-Thoughts, Bespoke-Stratos-17k, s1K / s1K-1.1, LIMO, OpenThoughts3, AM-DeepSeek-R1-Distilled-1.4M, Dolphin-R1.*
- **(G) Instruction-following / constraint data** — trains verifiable-constraint compliance (IFEval-style). *Tulu-3 Personas-IF, Conifer, AutoIF (code only), MUFFIN.*
- **(H) Preference / RLHF data** — for the DPO/RLVR stage after SFT. *UltraFeedback, HelpSteer2/3, Nectar, HH-RLHF, Skywork-Reward-Preference-80K, Tulu-3 preference mixture.*
- **(I) Data-selection methodology** — not datasets, but the algorithms that pick the subset. *LIMA, AlpaGasus, DEITA, Instruction-Mining, #InsTag, Cherry-LLM/IFD, Superfiltering, GRAPE, Tulu-3 decontamination.*

Two orthogonal axes cut across the taxonomy and matter more than the categories themselves:

1. **Teacher provenance** decides your license risk. GPT-3.5/GPT-4-distilled (B, and most of A pre-2024) carries an OpenAI ToS overhang no matter what license tag the HF card wears. Llama-distilled carries the Llama Community License. Qwen-distilled (SmolTalk2, OpenThoughts3 via QwQ-32B) and DeepSeek-R1-distilled (Apache/MIT teachers) are the cleanest.
2. **Verifiability of the seed** decides answer quality. Where a ground-truth answer or a unit test exists (E), rejection sampling produces genuinely correct data. Where it does not (B, and the chatty half of A), you are training on whatever the teacher said, GPT-isms included.

---

## 1. Curated Open SFT Mixtures

### Tulu v1 — *How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources* (Wang et al., 2023)
- **Venue / Link:** NeurIPS 2023 Datasets & Benchmarks · arXiv:2306.04751 · https://arxiv.org/abs/2306.04751
- **HF / GitHub:** `allenai/tulu-v1-sft-mixture` · https://github.com/allenai/open-instruct
- **Size:** 489,818 examples (~1.2 GB). No turn or token count published on the card.
- **Construction:** aggregation of human-written and distilled sources — FLAN v2 + CoT, OpenAssistant-1, Dolly, ShareGPT, GPT4-Alpaca, Code-Alpaca. No per-source counts given.
- **Domains:** general chat, NLP-task instructions, code, reasoning.
- **License:** ODC-BY at the mixture level, but GPT4-Alpaca and Code-Alpaca subsets are **CC-BY-NC-4.0**, and the GPT-4-distilled portions carry an OpenAI ToS overhang.
- **Notes:** historically important — this is the paper that established that no single open mixture wins everywhere and that mixture composition dominates. Superseded for practical use by Tulu 2 / Tulu 3.

### Tulu 2 SFT Mixture (Ivison et al., 2023)
- **Venue / Link:** arXiv:2311.10702 · https://arxiv.org/abs/2311.10702
- **HF:** https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture — **326,154 rows**, 1.24 GB, ungated, `license: odc-by`, `language: en`.
- **Composition (exact):** FLAN v2 100,000; ShareGPT 114,046; WizardLM Evol-Instruct 30,000; Open-Orca (GPT-4) 30,000; Code-Alpaca 20,022; GPT4-Alpaca 20,000; OpenAssistant-1 7,708; Science 7,544; LIMA 1,030; hardcoded 140.
- **Construction:** mixed human + GPT-3.5/GPT-4 distillation. GPT-4 is named explicitly for the GPT4-Alpaca and Open-Orca subsets.
- **License:** ODC-BY overall; **CC-BY-NC-4.0** on GPT4-Alpaca, Code-Alpaca and LIMA subsets — the mixture as a whole is *not* commercially clean.
- **Quality (sampled, 3 depths):** clearly stratified by source and showing its age. At 0% depth it is raw FLAN v2 — terse, template-y, machine-ish:
  > `[user]` "Question: Gdańsk (, ; German: "" , ) is a Polish city on the Baltic coast…" → `[assistant]` "1. Miguel / 2. Yes / 3. Eight / 4. unknown / 5. Swift…"
  At 40% it is Code-Alpaca: single-shot, no explanation, no comments, sometimes not even a code fence. At 85% it is WizardLM: long, competent, but visibly GPT-3.5-flavoured. **Verdict: superseded.** The FLAN and Code-Alpaca thirds would actively teach a modern base model to answer *worse* than it already does.

### Tulu 3 SFT Mixture (Lambert et al., 2024)
- **Venue / Link:** arXiv:2411.15124 · https://arxiv.org/abs/2411.15124
- **HF:** https://huggingface.co/datasets/allenai/tulu-3-sft-mixture — **939,343 rows** / 1,412,954,868 parquet bytes, ungated, `license: odc-by`.
- **Composition (exact, from the live `source` column statistics — 19 sources):** personahub_math_v5 149,960; evol_codealpaca_heval_decontaminated 107,276; wildchat_100k 100,000; **aya_100k 100,000**; flan_v2 89,982; numinamath_tir_decontaminated 64,312; wildguardmix synthetic 50,000; wildjailbreak 50,000; open_math_2_gsm8k_50k 50,000; personas_math_grade 49,980; personahub_code_v2 34,999; **personahub_ifdata (instruction-following) 29,980**; personahub_math_interm_algebra 20,000; coconot 10,983; sciriff_10k 10,000; no_robots 9,500; oasst1 7,131; table_gpt 5,000; hardcoded 240.
- **Construction:** persona-driven synthetic generation (GPT-4o for the persona subsets), plus decontaminated third-party sets, plus real WildChat↔GPT-4 logs, plus human data (No Robots, OASST1).
- **License:** ODC-BY on the mixture; **No Robots subset is CC-BY-NC-4.0**; WildChat-GPT-4 and the persona subsets are OpenAI-output-derived.
- **Quality (sampled at rows 0 / 2,000 / 200,000 / 900,000):** high-quality *but a trap for an English-anchor study*. The mixture is **substantially non-English by construction**: the OASST1 block at the head is Spanish/Russian, WildChat at row 200k is Turkish, and the entire 100k `aya_100k` block at the tail is low-resource languages:
  > row 900,000 `[user]` "A cikin wanne nau'in ra'ayi za ku rarraba tweet mai zuwa?…" (Hausa) → `[assistant]` "Zan rarraba tweet ɗin da aka bayar a matsayin: Marar kyau"
  > row 200,000 `[user]` "Repeat this string \"coffee and flower\"" → `[assistant]` "coffee and flower"
  Roughly **207k of 939k rows (~22%) come from aya + wildchat blocks that are heavily non-English.** Excellent as a general post-training mixture; must be language-filtered before it can serve as an English control.

### OpenHermes-2.5 (Teknium, 2023)
- **Venue / Link:** no dedicated paper found (**not verified**) · https://huggingface.co/datasets/teknium/OpenHermes-2.5
- **Size:** **1,001,551 rows**, 795,216,961 parquet bytes, single train split, ungated.
- **Construction:** compilation of ~14 mostly-synthetic sources — Airoboros 2.2, CamelAI (physics/math/chem/bio), ChatBot Arena GPT-4-only, Collective Cognition, CoT Alpaca GPT4, Evol-Instruct 70K & 140K, Glaive Code Assistant, GPT4-LLM, GPTeacher, medical tasks, MetaMath 40k, SlimOrca 550K, Platypus, ShareGPT GPT-4-only, Unnatural Instructions GPT4. No per-source counts published.
- **License:** **none stated at all** on the HF card. Overwhelmingly GPT-4-distilled ⇒ OpenAI ToS overhang. Treat as unresolved, not permissive.
- **Quality (sampled at 0 / 2,000 / 40% / …):** clean, consistently formatted two-turn data with a `source` and `category` column that makes filtering easy. The `airoboros2.2` head is fine but shallow; the `GPT-4 Comparison Data` block at 40% depth is noticeably thinner — short listicles with no reasoning:
  > `[user]` "Generate a list of activities to do with a 3-year-old" → `[assistant]` "Here is a list of activities…: – Make a fort – Finger painting – Simple puzzles – Playing with wooden blocks…"
  Every row carries 16 columns of which ~12 are `None` for most rows (`views`, `avatarUrl`, `title`, `hash`, `idx`…), i.e. schema cruft rather than signal. **Verdict: still a fine general-chat backbone, but 2023-era answer depth and unresolved licensing.**

### Infinity-Instruct (BAAI, 2025) — **GATED, NOT SAMPLED**
- **Venue / Link:** arXiv:2506.11116 · https://arxiv.org/abs/2506.11116 (+ arXiv:2507.06968, AAAI 2026)
- **HF / GitHub:** `BAAI/Infinity-Instruct` (**`gated: auto`** — access request required; the card itself returned *"Access to dataset BAAI/Infinity-Instruct is restricted and you are not in the authorized list"*) · https://github.com/FlagOpen/Infinity-Instruct
- **Size (published version table):** InfInstruct-3M 3,463,473; **InfInstruct-7M 7,449,106**; InfInstruct-0613 362,330; InfInstruct-0625 659,808; InfInstruct-Gen(0729) 1,456,927.
- **Composition (7M foundational split, exact):** google/flan 2,435,840; teknium/OpenHermes-2.5 855,478; MetaMathQA 690,138; orca-math-word-problems-200k 398,168; MathInstruct 329,254; "selected subjective instructions" 1,342,427 (itself containing UltraChat 237,199, WizardLM evol-instruct-196K 88,681, Alpaca-GPT4, BELLE, …), plus code sources.
- **License:** `cc-by-sa-4.0`. **ShareAlike is viral** — a derivative dataset would have to be released under CC-BY-SA too, which conflicts with mixing into an ODC-BY or Apache release. Transitively GPT-4-derived through OpenHermes/Alpaca-GPT4.
- **Notes:** bilingual EN/ZH. The synthesis teacher model is **not verified** from the sources I could reach.

### SmolTalk (HuggingFaceTB, 2024)
- **Venue / Link:** SmolLM2 report, arXiv:2502.02737 · https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- **Size:** 1,043,917 train + 54,948 test in the `all` config (2,197,730 rows across all configs), 4,152,663,295 parquet bytes. Ungated. Named sub-configs are individually loadable (`apigen-80k` 83,144, `everyday-conversations` 2,260, `explore-instruct-rewriting` 30,400, …).
- **Construction:** ~50% newly synthesised via the **Magpie pipeline with Llama-3.1-405B-Instruct** ("Smol-Magpie-Ultra", 400K), plus curated public sets — OpenHermes2.5 100K, MetaMathQA 50K, NuminaMath-CoT 100K, Self-OSS-Starcoder2-Instruct ~50K, SystemChat-2.0 30K, APIGen function calling 80K, Explore-Instruct-Rewriting 30K, Smol-Constraints 36K, Smol-Rewrite 50K, Smol-Summarize 100K, LongAlign.
- **License:** **no license field on the card**; new subsets are stated Apache-2.0 while inherited subsets keep their original licenses. Llama-3.1-405B teacher ⇒ Llama 3.1 Community License terms on the Magpie portion.
- **Quality (sampled at 0% / 40% / 85%):** consistently good and *visibly modern*. Math answers show real derivations; the `smol-magpie-ultra` block is genuinely multi-turn with follow-ups that change the task; `smol-constraints` is IFEval-shaped and useful:
  > `[user]` "Your response should contain at least 3 sentences. Your answer must contain exactly 3 bullet points… Include keywords friendship, trust, and loyalty." → `[assistant]` "* Trust is the foundation of any strong friendship…"
  > `[user, 85% depth]` "You are a tour guide in Alaska on a boat in Prince William sound…" → in-character role-play sustained across 5 turns.
  **No deep-offset degradation observed.** English-only, which is exactly what an anchor set needs.

### SmolTalk2 (HuggingFaceTB, 2025)
- **Venue / Link:** no dedicated arXiv paper (**not verified**); primary source is the SmolLM3 release · https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
- **Size (exact, from the card's own stats tables):** `Mid` 4,779,894 examples / **35,172.1 M tokens**; `SFT` **3,383,242 examples / 19,292.4 M tokens**, avg 2.58 turns; `Preference` 446,886 examples / 850.62 M tokens (chosen side). Ungated.
- **Composition (SFT split, exact examples):** OpenThoughts3-1.2M_think 1,133,524 (86.7% of tokens, weight 0.02); OpenThoughts3_no_think 435,193; smol-magpie-ultra 406,843; OpenHermes-2.5 384,900; smoltalk-multilingual-8languages 254,047 + 244,736 (think); smol-summarize 96,061; Mixture-of-Thoughts_science 86,110; xlam tool traces 59,962; smol-rewrite 53,262; systemchats 33,997; explore-instruct-rewriting 30,391; **tulu-3-sft-personas-instruction-following 29,970**; multi-turn-reasoning-if 28,217; table-gpt 13,203 + 13,201; aya_dataset (Qwen3-32B answers) 15,222; smolagents tool-calling 9,079; hermes-function-calling 8,961; LongAlign-64k 7,526 + 6,249; s1k-1.1 835; everyday-conversations 2,260.
- **Construction:** *this is the key point for us* — new reasoning ("think") responses are generated by **Qwen3-32B**, and the preference-rejected side by **Qwen3-0.6B**. The non-think side reuses OpenThoughts3, OpenHermes-2.5 and SmolTalk prompts. The card also publishes the **training weight** actually used for each subset (e.g. OpenThoughts3_think 0.02, smol-magpie-ultra 0.5, OpenHermes-2.5 0.5), which is a ready-made recipe.
- **License:** **no license field on the card**; new subsets stated Apache-2.0, components keep their own licenses.
- **Quality (sampled `smoltalk_smollm3_smol_magpie_ultra_no_think` at 0% / 50% / 90%):** strong multi-turn data with realistic user pushback:
  > `[user, 90% depth]` "For this question, I was hoping the response would be a little more concise. You previously listed some general advice for a programmer just beginning to learn. Among these, which are the most important?"
  Answers are structured but not bullet-spammy; code turns include a language switch mid-conversation (Java → Python) which is exactly the kind of context-carrying we want. **No degradation at depth.**

### Llama-Nemotron-Post-Training-Dataset (NVIDIA, 2025)
- **Venue / Link:** arXiv:2505.00949 · https://arxiv.org/abs/2505.00949 · https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
- **Size (exact, per-domain):** math 22,066,397; code 10,108,883; science 708,920; instruction-following 56,339; chat 39,792; safety 31,426 (≈33.0 M rows total). Ungated.
- **Construction (exact teacher counts):** Qwen-2.5-Math-7B-Instruct 19,840,970; Qwen-2.5-Coder-32B-Instruct 8,917,167; DeepSeek-R1 3,934,627; Qwen-2.5-32B-Instruct 2,297,175; Qwen-2.5-72B-Instruct 464,658; Llama-3.3-70B-Instruct 420,021; Llama-3.1-Nemotron-70B-Instruct 31,218; Mixtral-8x22B-Instruct-v0.1 31,426; Llama-3.3-Nemotron-70B Feedback/Edit/Select 22,644.
- **License:** predominantly **CC-BY-4.0**, with ODC-BY subsets (WildChat-derived) and CC-BY-SA subsets (StackOverflow-derived); Llama-derived outputs additionally carry Llama 3.1/3.3 Community License redistribution terms. **Not a single uniform license** — cite per subset.
- **Notes:** overwhelmingly math+code by volume (97%). The chat and instruction-following splits (≈96k rows combined) are the parts relevant to a general-purpose English mixture.

### Nemotron-Post-Training-Dataset-v1 / v2 (NVIDIA, 2025)
- **Venue / Link:** v1 cited as `@software` (no dedicated paper); v2's associated model paper is Nemotron Nano 2, arXiv:2508.14444.
- **HF:** `nvidia/Nemotron-Post-Training-Dataset-v1` (ungated) · `nvidia/Nemotron-Post-Training-Dataset-v2` (**`gated: auto` — access denied in this environment, not sampled**)
- **Size:** v1 exact — chat 746,622; code 1,896,395; math 2,044,407; stem 20,662,167; tool_calling 310,051; **total 25,659,642**. v2 ≈6.3 M (math 239,467; code 175,000; stem 355,000; chat 627,720; plus ~1 M each for ja/de/it/es/fr) — v2 figures are secondary-sourced, **not** read from rows.
- **Construction:** v1 teachers — **DeepSeek-R1-0528 24,602,969** and **Qwen3-235B-A22B 1,056,673**. v2 adds Qwen2.5-14B/32B-Instruct(-AWQ) and Qwen3-30B-A3B.
- **License:** `cc-by-4.0`, stated ready for commercial and non-commercial use.

### OpenThoughts-114k / OpenThoughts2-1M / OpenThoughts3-1.2M (Guha et al., 2025)
- **Venue / Link:** arXiv:2506.04178 · https://arxiv.org/abs/2506.04178 · https://github.com/open-thoughts/open-thoughts
- **HF:** `open-thoughts/OpenThoughts-114k` (**113,957** rows), `open-thoughts/OpenThoughts2-1M` (~1.04 M), `open-thoughts/OpenThoughts3-1.2M` (**exactly 1,200,000** examples, 59.76 GB / 28.2 GB parquet across 120 shards). All ungated, all `license: apache-2.0`.
- **Construction:** OpenThoughts-114k distils **DeepSeek-R1** over TACO / APPS / CodeContests / Codeforces (code), NuminaMath-CoT (math), CAMEL chem-bio-physics (science), riddle_sense (puzzles), with correctness verification. **OpenThoughts3 switches teacher to QwQ-32B**, annotating 75k source questions **16×** each → 850K math / 250K code / 100K science.
- **License:** Apache-2.0 — one of the cleanest large reasoning sets available, because QwQ-32B and DeepSeek-R1 are themselves permissively licensed.
- **Notes:** the ablation-driven construction (they searched over question sources, filtering strategies and answer counts) makes this the most methodologically careful reasoning corpus in the open.

### OpenOrca / SlimOrca (Lian et al., 2023; Orca: Mukherjee et al., 2023)
- **Venue / Link:** Orca arXiv:2306.02707 · https://arxiv.org/abs/2306.02707
- **HF:** `Open-Orca/OpenOrca` — 2,942,029 rows, `license: mit`; `Open-Orca/SlimOrca` — 517,982 rows, `license: mit`. Both ungated.
- **Construction:** FLAN prompts + 16 hand-crafted system messages, answered by **GPT-3.5** (~3.2 M) and **GPT-4** (~1 M). SlimOrca keeps only the GPT-4 portion and uses GPT-4 to strip answers that disagree with the FLAN human annotations.
- **License:** MIT tag, but the content is entirely OpenAI-generated — the card is silent on the ToS implication. Treat as encumbered.
- **Notes:** the "explanation tuning" idea (make the teacher show its work) is the direct ancestor of every long-CoT set in section 4.

### Dolphin / Dolphin-R1 (Cognitive Computations / QuixiAI)
- **HF:** `cognitivecomputations/dolphin` (org renamed → `QuixiAI/dolphin`) — 3,731,947 rows across `flan1m-alpaca-uncensored` (892k) and `flan5m-alpaca-uncensored` (2.84M), `license: apache-2.0`. `cognitivecomputations/dolphin-r1` — **814,334** rows, `license: apache-2.0`.
- **Construction:** Dolphin is an Orca replication (FLANv2 + GPT-4 ~1M, FLANv2 + GPT-3.5 ~3.5M) with alignment/refusal content deliberately filtered out ("uncensored"). Dolphin-R1 is explicitly built to mimic the composition of DeepSeek-R1's unreleased 800k set: **~300k DeepSeek-R1 traces + ~300k Gemini-2.0-Flash-Thinking traces + ~200k Dolphin chat**.
- **License:** Apache-2.0 tag on both, but Dolphin is GPT-derived and **Dolphin-R1's Gemini third is subject to Google's Gemini API terms**, which typically forbid training competing models. The blanket Apache tag does not reflect either.
- **Notes:** the "uncensored" framing means refusal and safety behaviour has been *removed*; if you train on it you inherit that.

## 2. Single-Pipeline Synthetic Generators

### Self-Instruct (Wang et al., 2022) and Alpaca (Taori et al., 2023)
- **Venue / Link:** arXiv:2212.10560 (ACL 2023) · https://arxiv.org/abs/2212.10560 · https://github.com/yizhongw/self-instruct
- **Size:** 52K instructions / 82K instance input-output pairs, bootstrapped from **175 human-written seed tasks**. Alpaca: `tatsu-lab/alpaca` — **52,002** examples.
- **Construction:** vanilla GPT-3 (Self-Instruct) / **text-davinci-003** (Alpaca), batch-decoded 20 instructions per call, total Alpaca cost <$500.
- **License:** Self-Instruct repo Apache-2.0; **Alpaca is `cc-by-nc-4.0`** (research only). `yahma/alpaca-cleaned` is tagged `cc-by-4.0`. Both are OpenAI-output-derived.
- **Notes:** the authors' own audit found **~46% of a 200-sample manual check "may have problems."** Historically foundational; obsolete as training data. Do not use.

### WizardLM Evol-Instruct (Xu et al., 2023)
- **Venue / Link:** arXiv:2304.12244 (ICLR 2024) · https://arxiv.org/abs/2304.12244 · https://github.com/nlpxucan/WizardLM
- **HF:** `WizardLMTeam/WizardLM_evol_instruct_V2_196k` — the file actually on the Hub is `..._V2_143k.json`, i.e. **143,000 rows**; the card states the full "196k" requires merging with original ShareGPT data separately.
- **Construction:** **Evol-Instruct** — iteratively rewrite seed instructions to be deeper (add constraints, deepen, concretize, increase reasoning steps) or broader (mutate to new topics), with ChatGPT as both evolver and responder; GPT-4 used as automatic judge.
- **License:** **conflicting.** HF card says `license: mit`; the authors' GitHub README says **CC-BY-NC-4.0, academic research only, no commercial use**. Take the restrictive reading.
- **Quality (sampled via the Tulu-2 `wizardlm` block at 85% depth):** competent, long, but unmistakably 2023 ChatGPT — heavy scaffolding, occasional over-elaboration on trivial asks. Useful as an *idea* (instruction complexity escalation) rather than as data.

### UltraChat (Ding et al., 2023) / UltraChat-200k
- **Venue / Link:** EMNLP 2023 · arXiv:2305.14233 · https://arxiv.org/abs/2305.14233 · https://github.com/thunlp/UltraChat
- **HF:** `HuggingFaceH4/ultrachat_200k` — train_sft **207,865**; test_sft 23,110; train_gen 256,032; test_gen 28,304. `license: mit`, ungated. Raw UltraChat: 1.5 M dialogues (paper), 1.57 M (repo badge).
- **Construction:** **fully synthetic, ChatGPT talking to ChatGPT** — no human queries at all. Three sectors: *Questions about the World* (Part I 280k + Part II 290k, from 30 meta-topics / 1,100+ subtopics / top-10,000 Wikidata entities), *Writing and Creation* (457k, 20 writing types × 200 seed instructions), *Assistance on Existing Materials* (~100,000 source materials sampled from **C4**).
- **License:** MIT tag, but the content is ChatGPT-generated ⇒ OpenAI ToS overhang.
- **Quality (sampled train_sft at 0% / 40% / 90%):** **this one aged badly, and the deep offsets are worse than the head.** The "Assistance on Existing Materials" third dominates at depth, and the "existing material" is C4 web sludge — Shopify theme documentation, a pre-school's marketing copy, a local-news story about a police-charity impersonator:
  > `[user, 90% depth]` "Apple Tree's Pre-School program continues its best practice by combining children ages 3-5 years old… " → `[assistant]` "Apple Tree's Pre-School program combines children ages 3-5 years old allowing for small group sizes…" (a near-verbatim restatement)
  > `[user, 0% depth]` "These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+…). What theme version am I using?" → `[assistant]` "This feature only applies to Collection pages and Featured Collections sections of the section-based themes listed in the text material."
  The follow-up turns are formulaic ("Can you add more…?", "Can you tell me more about…?") — a template, not a conversation. **Verdict: Mixed/Poor. Do not put this in a 2026 mixture** except possibly a small multi-turn-format slice.

### Magpie (Xu et al., 2024) — Pro / Air / Ultra
- **Venue / Link:** arXiv:2406.08464 · https://arxiv.org/abs/2406.08464
- **Method:** feed an aligned model *only* its chat template's pre-query tokens and let it hallucinate the user turn, then answer it. Zero seed prompts, zero human input, zero prompt engineering.
- **Variants and exact sizes:**
  - `Magpie-Align/Magpie-Pro-300K-Filtered` — **300,000** rows filtered from 4M raw, generated by **Llama-3-70B-Instruct**. `license: llama3`.
  - `Magpie-Align/Magpie-Air-300K-Filtered` — **300,000** filtered from 3M raw, **Llama-3-8B-Instruct**. **No license field.**
  - `argilla/magpie-ultra-v1.0` — default config **999,960** rows; `filtered` 621,837; `top_300k_longer_conversations` 285,000 train / 15,000 test. Generator **Llama-3.1-405B-Instruct-FP8**; quality/difficulty scoring by Llama-3.1-8B-Instruct; safety by Llama-Guard-3-8B; reward score by ArmoRM-Llama3-8B-v0.1; dedup by gte-large-en-v1.5 + FAISS. **No license field.**
- **License risk:** all three are Llama-derived ⇒ **Llama 3 / 3.1 Community License** naming and use restrictions propagate to models trained on them, regardless of the (often absent) HF tag.
- **Quality (sampled Magpie-Pro-300K-Filtered at 0% / 50% / 95%, magpie-ultra `filtered` at 0% / 50% / 95%):** **Good and remarkably uniform at depth** — no degradation between 0% and 95%. Answers are well-structured, correctly headed, appropriately long. The weakness is *sameness*: because the model invents its own questions, the distribution is encyclopaedic-explainer-shaped, and the GPT-isms are constant:
  > `[assistant, 95% depth]` "**The age-old quest for a tidy and efficient email inbox!** Here are some best practices…"
  > `[user, 50% depth]` "What is standardized testing in education?" / "Explain Putin's long-term strategy and its implications for global politics."
  magpie-ultra ships per-row `difficulty`, `quality`, `reward_model_score` and `category` columns, which makes it the most *selectable* corpus in this review — you can implement DEITA-style filtering with the metadata already present.

### Genstruct-7B (NousResearch)
- **Link:** https://huggingface.co/NousResearch/Genstruct-7B — **this is a model, not a dataset**, fine-tuned from Mistral-7B-v0.1 to generate grounded instruction/response pairs from raw text passages (inspired by Ada-Instruct, arXiv:2310.04484). No public training set is named on its card (**not verified**). `license: apache-2.0`.
- **Relevance to us:** the one pipeline in this review designed to condition generation on *your own corpus*. If we want SFT data that exercises the idiom-tagged CPT corpus, Genstruct-style grounded generation is the mechanism — but with a modern open teacher rather than the 7B model itself.

## 3. Human-Written Demonstrations and Real User Logs

### databricks-dolly-15k (Databricks, 2023)
- **Venue / Link:** Databricks "Free Dolly" blog post, 12 Apr 2023 (no arXiv paper) · https://huggingface.co/datasets/databricks/databricks-dolly-15k
- **Size:** **15,011** records, English only, single split, 7,747,823 parquet bytes. Ungated.
- **Construction:** written by **Databricks employees** (blog says "more than 5,000") through an internal gamified crowdsourcing campaign, not external crowdworkers. Contributors were told not to use generative AI; Wikipedia was allowed as a reference for closed-QA and summarization.
- **Domains / categories:** open QA, closed QA, information extraction, summarization, brainstorming, classification, creative writing.
- **License:** **CC-BY-SA-3.0** — ShareAlike, so a derivative corpus inherits the copyleft. Commercially usable but viral.
- **Quality (sampled at 0% / 50% / 95%):** genuinely human, genuinely *short*. Answers are one to five sentences with no scaffolding, which is stylistically the opposite of everything else in this review. The category labels are unreliable:
  > `[instruction]` "What are some movies about Artificial Intelligence and Machine Consciousness?" `[category]` **classification** (it is brainstorming)
  > `[instruction]` "Which is a species of fish? Tope or Rope" → `[response]` "Tope"
  `context` fields are pasted Wikipedia. **Verdict: Mixed.** Valuable as a small human-style anchor (and as the only cheap source of "answer briefly" behaviour), useless as a capability source. Its own authors decline to claim SOTA effectiveness.

### OpenAssistant Conversations — OASST1 / OASST2 (Köpf et al., 2023)
- **Venue / Link:** NeurIPS 2023 Datasets & Benchmarks · arXiv:2304.07327 · https://arxiv.org/abs/2304.07327
- **HF:** `OpenAssistant/oasst1`, `OpenAssistant/oasst2`, both `license: apache-2.0`, ungated.
- **Size:** the paper reports **161,443 messages in 35 languages with 461,292 quality ratings across >10,000 fully annotated conversation trees, from >13,500 volunteers.** OASST1 ready-for-export: 88,838 messages (84,437 train / 4,401 val). OASST2 ready-for-export: 13,854 trees / 135,174 messages (128,575 train / 6,599 val); the full unfiltered export is 70,642 trees / 208,584 messages.
- **Construction:** unpaid volunteers on open-assistant.io both *wrote* and *rated/ranked* messages, producing branching trees rather than flat pairs.
- **License:** Apache-2.0 — one of the very few genuinely unencumbered human sets.
- **Caveats:** every message ships Detoxify scores (toxicity, severe_toxicity, obscene, identity_attack, insult, threat, sexual_explicit) plus human spam/PII/hate/sexual/violence flags and a `deleted` flag — i.e. the authors expect you to filter. **Heavily multilingual** (English, Spanish, Russian, Chinese dominate), so an English anchor needs the `lang` filter.

### No Robots (HuggingFaceH4, 2023)
- **Link:** no paper · https://huggingface.co/datasets/HuggingFaceH4/no_robots
- **Size:** **10,000** total — 9,500 train / 500 test, English only, 10,474,682 parquet bytes. Ungated.
- **Categories (exact):** Generation 4,560; Open QA 1,240; Brainstorm 1,120; Chat 850; Rewrite 660; Summarize 420; Coding 350; Classify 350; Closed QA 260; Extract 190.
- **Construction:** "skilled human annotators" writing InstructGPT-style demonstrations. The vendor and headcount are `[More Information Needed]` on the card (**not verified**).
- **License:** **CC-BY-NC-4.0 — non-commercial.** This is the single most commonly overlooked license landmine in open SFT, because No Robots is embedded inside the Tulu 3 mixture (9,500 rows).
- **Quality (sampled at 0% / 50% / 95%):** **the best human data in this review.** Prompts carry real constraints and the responses honour them without boilerplate:
  > `[user, 50%]` "Explain the internet to someone as if it's the 1970s. Keep a casual tone, and make it a maximum of two paragraphs." → `[assistant]` "Okay, so you know how we have television, and we send each other letters, and we call each other on the phone?…"
  > `[user, 95%]` "Write a short story about a teddy bear looking for his owner… only one paragraph and nine sentences long. Also, give the teddy bear the name Jojo."
  No degradation at depth; no "As an AI language model"; no bullet-point reflex. It is also, at 9,500 rows, small enough to be a style anchor rather than a capability source.

### LIMA (Zhou et al., 2023) — **GATED**
- **Venue / Link:** NeurIPS 2023 · arXiv:2305.11206 · `GAIR/lima` (**`gated: auto`**, contact info required — not sampled)
- **Size:** 1,000 train / 50 dev / 300 test. Train breakdown: Stack Exchange STEM 200, Stack Exchange other 200, wikiHow 200, r/WritingPrompts 150, Natural Instructions 50, author-written 200 (+50 dev). Test: r/AskReddit 70, author-written 230.
- **License:** card says it follows **CC-BY-NC-SA** unless a source is stricter, in which case the stricter license governs. HF tag is `license: other`. Non-commercial.
- **Notes:** see the data-selection section for the numbers. As *data* it is 1,000 rows; as *methodology* it is the most cited result in the field.

### ShareGPT (community scrape, 2023)
- **Link:** `anon8231489123/ShareGPT_Vicuna_unfiltered` (`license: apache-2.0`, ungated, ~402k downloads/30d), `RyokoAI/ShareGPT52K` (CC0)
- **Size:** ~53,000 English conversations filtered from ~90–100K raw.
- **Provenance:** real ChatGPT conversations users voluntarily shared via the ShareGPT browser extension, then **scraped by an anonymous third party** after ShareGPT disabled its public Explore page (~29–30 Mar 2023). LMSYS stated on FastChat issue #90 that they had "no current plans to release the dataset". **The Apache-2.0 tag is a claim by an uploader over data that was never theirs.**
- **Caveats:** the RyokoAI card warns it "may contain personal information"; no systematic PII audit exists; the "unfiltered" variant was produced by keyword-stripping ~100 ChatGPT moralizing phrases.
- **Verdict: do not use in anything we intend to release.** Its provenance is unclean in a way no license tag fixes, and it is the reason Tulu 2's 114k ShareGPT block is a liability.

### WildChat-1M / WildChat-4.8M (Zhao et al., 2024)
- **Venue / Link:** ICLR 2024 · arXiv:2405.01470 · https://huggingface.co/datasets/allenai/WildChat-1M
- **Size:** WildChat-1M default (non-toxic) split ~838K conversations, up to 498 turns; the paper cites 1M conversations / 2.5M+ turns for the full corpus. WildChat-4.8M: 3.2M conversations in the default split, Apr 2023 – May 2024, **74 languages**.
- **Construction:** researchers offered free ChatGPT access (GPT-3.5-turbo-0301, GPT-4-0314) in exchange for **consensual opt-in logging** — genuinely the cleanest provenance among real-log datasets.
- **License:** relicensed from AI2 ImpACT to **ODC-BY**. The main `allenai/WildChat-1M` repo reported `gated: false` when I checked; `WildChat-1M-Full` (with toxic content) and some ImpACT variants remain gated.
- **Caveats:** ships hashed IP addresses and country/state metadata (PII surface), plus per-message OpenAI Moderation and Detoxify scores and `redacted`/`toxic` booleans. Content is GPT-3.5/GPT-4 output ⇒ OpenAI ToS overhang.
- **Value:** the *prompt* distribution is the real prize — it is what actual users type, which no synthetic pipeline reproduces. Consider keeping the prompts and regenerating responses with an open teacher.

### LMSYS-Chat-1M (Zheng et al., 2023) — **GATED**
- **Venue / Link:** arXiv:2309.11998 · `lmsys/lmsys-chat-1m` (**`gated: auto`**, bespoke license agreement — not sampled)
- **Size:** **1,000,000** conversations across 25 LLMs from **210,479 unique IPs**, 154 languages, average 2.0 turns.
- **Caveats stated by the authors:** names are redacted to `NAME_1` style placeholders; the card warns it "contains unsafe conversations that may be perceived as offensive or unsettling," deliberately retained for safety research; and **no decontamination was performed**, so it "may contain test questions from popular benchmarks."
- **Verdict:** excellent for studying prompt distributions and safety; the license agreement plus the acknowledged benchmark contamination make it a poor SFT ingredient.

### Aya Dataset / Aya Collection (Singh et al., 2024)
- **Venue / Link:** ACL 2024 · arXiv:2402.06619 · https://huggingface.co/datasets/CohereLabs/aya_dataset (formerly `CohereForAI/`)
- **Size:** aya_dataset **204,114** prompt-completion pairs (202,364 train / 1,750 test) in 65 languages (71 including dialects/scripts). aya_collection is reported at 513M instances across 114 languages in the paper; the per-config row counts on the card do not cleanly reconcile to that figure (**flagged, not reconciled**).
- **Construction:** aya_dataset is original human writing by fluent speakers via the Aya Annotation Platform (contributors from 119 countries); aya_collection is templated + machine-translated existing NLP datasets.
- **License:** **Apache-2.0**, explicitly usable for any purpose including commercial.
- **Caveats from the card:** "93% of languages not represented"; uneven per-annotator contribution; the platform "lacked specific flags for toxic speech"; no re-labelling capability, so language mislabels persist.
- **Relevance to us:** this is the 100k block sitting inside Tulu-3 that makes Tulu-3 non-English. For a multilingual arm of the project it is an asset; for the English control it must be filtered out.

### Open-Platypus (Lee et al., 2023)
- **Venue / Link:** arXiv:2308.07317 · `garage-bAInd/Open-Platypus`
- **Size:** **24,926** examples, aggregated from 11 sources (PRM800K, MATH, ScienceQA, SciBench, ReClor, TheoremQA, leetcode-solutions, airoboros-gpt4-1.4.1, tigerbot-leetcode, ARB, openassistant-guanaco), deduplicated by keyword search plus Sentence-Transformer similarity (>80% dropped).
- **License:** mixed by source (MIT / Apache-2.0 / CC-BY variants); **ScienceQA and ReClor are non-commercial**.
- **Decontamination:** the authors explicitly removed ~200 questions matching HF benchmark test sets.

## 4. Verifiable-Domain SFT — Math and Code

### OpenMathInstruct-1 / -2 (Toshniwal et al., 2024)
- **Venue / Link:** v1 NAACL 2024 Findings, arXiv:2402.10176 · v2 arXiv:2410.01560 · https://arxiv.org/abs/2410.01560
- **HF:** `nvidia/OpenMathInstruct-1` · `nvidia/OpenMathInstruct-2` (32 parquet shards, 7,576,089,560 bytes; ungated).
- **Size:** v1 **1.8M** problem-solution pairs. v2 **14M** question-solution pairs over **~600K unique questions** (≈8× MetaMathQA); 1M/2M/4M subsets are shipped as separate configs.
- **Construction:** v1 teacher **Mixtral-8x7B** with code-interpreter execution and answer-matching rejection sampling, seeded on the GSM8K + MATH *training* splits. v2 teacher **Llama-3.1-405B-Instruct**, and crucially it synthesises *new problems*, not just new solutions.
- **License:** v1 "NVIDIA License" (commercial use permitted); **v2 `cc-by-4.0`**. Both permissive; the v2 teacher being Llama-3.1 adds Llama Community License considerations for derived models.
- **Decontamination:** v2 ships a contamination explorer checking overlap against GSM8K, MATH, AMC 2023, AIME 2024 and Omni-MATH test sets.
- **Quality (sampled at 0% / 50% / 95%):** **Excellent for what it is.** Solutions are clean LaTeX derivations with the answer boxed in `expected_answer`, no chatter, no persona. The `problem_source` column tells you exactly what you are getting and it shifts with depth — `augmented_gsm8k` and `augmented_math` at the head, **raw `math`** (i.e. verbatim MATH-train problems) at 95%:
  > `[problem, 95%]` "The number $m$ is a three-digit positive integer and is the product of the three distinct prime factors $x$, $y$ and $10x+y$…" `[problem_source]` **math**
  So the deep tail is where your MATH-benchmark contamination risk concentrates — filter on `problem_source` if you report MATH.

### MetaMathQA (Yu et al., 2024)
- **Venue / Link:** ICLR 2024 · arXiv:2309.12284 · https://huggingface.co/datasets/meta-math/MetaMathQA
- **Size:** **395,000** rows. `license: mit`, ungated.
- **Construction:** bootstraps GSM8K/MATH questions four ways — answer augmentation, rephrasing (**GPT-3.5-Turbo**), self-verification, and FOBAR backward reasoning — with rejection sampling against ground-truth answers.
- **Decontamination:** the authors state explicitly that "all MetaMathQA data are augmented from the training sets of GSM8K and MATH. None of the augmented data is from the testing set."
- **Notes:** superseded in scale and answer quality by OpenMathInstruct-2, but the rephrasing trick is what makes it diverse. GPT-3.5 involvement ⇒ OpenAI ToS overhang.

### NuminaMath-CoT / 1.5 / TIR (Numina / HuggingFace, 2024–2025)
- **HF:** `AI-MO/NuminaMath-CoT` **859,594** rows (5 shards, 1,234,185,701 bytes); `AI-MO/NuminaMath-1.5` **896,215** rows; `AI-MO/NuminaMath-TIR` **72,540** rows. All `license: apache-2.0`, ungated.
- **Construction:** CoT is aggregated and reformatted from `aops_forum`, `amc_aime`, `cn_k12`, `gsm8k`, `math`, `olympiads`, `orca_math`, `synthetic_amc`, `synthetic_math` — largely human-authored solutions, not a single teacher model. 1.5 adds answer/problem-type metadata and a manually verified Olympiads-reference subset, and **drops `synthetic_amc` over quality concerns**. TIR generates tool-integrated (code-execution) traces with **GPT-4** over ~70k numeric-answer problems, filtered for correctness with up to 3 retries — **OpenAI-derived despite the Apache tag**.
- **Quality (sampled CoT at 0% / 50% / 95%):** mostly solid Olympiad and `synthetic_math` derivations, but the `cn_k12` block is a real problem. At 50% depth I pulled an item labelled as math that is an **English grammar multiple-choice question**:
  > `[source]` cn_k12 · `[problem]` "—I'd like some more cheese. —Sorry, there's \_\_\_\_ left. A. some B. none C. a little D. few" → `[solution]` "**Answer** B. Option B, 'none,' can modify both countable and uncountable nouns…"
  The schema is also redundant — `problem`/`solution` are duplicated verbatim into `messages`, roughly doubling storage. And per Tulu 3's own audit, **11.3% of NuminaMath-TIR/MATH overlapped their MATH eval** and had to be stripped. **Verdict: Good but requires source-level filtering and real decontamination.**

### OpenR1-Math-220k (HuggingFace Open-R1, 2025)
- **HF:** `open-r1/OpenR1-Math-220k` — **450,258 rows across all splits** (default 93.7k, extended 131k, …); "220k" refers to unique problems, not rows. `license: apache-2.0`, ungated.
- **Construction:** teacher **DeepSeek-R1**, seeded on NuminaMath-1.5 (~400k problems) plus cn_k12 for the extended split. Verified with **Math-Verify** (rule-based) plus **Llama-3.3-70B-Instruct as LLM judge** for ~12% of ambiguous cases; every retained problem has at least one correct trace.
- **License:** Apache-2.0, permissive teachers throughout — one of the cleanest reasoning-math sets.

### Mixture-of-Thoughts (HuggingFace Open-R1, 2025)
- **HF:** `open-r1/Mixture-of-Thoughts` — `all` **349,317** rows (math 93,733 / code 83,070 / science 172,514), 3,077,653,717 parquet bytes. Ungated. **No license field on the card (not verified).**
- **Construction:** verified **DeepSeek-R1** traces; math from OpenR1-Math-220k `default`, code from `open-r1/codeforces-cots` (`solutions` + `solutions_w_editorials`), science from the OpenThoughts lineage. The per-domain mixture was tuned against AIME 2024 / GPQA-Diamond / LiveCodeBench-v4.
- **Quality (sampled `all` at 0% / 50% / 95%):** high-quality but **enormous per-row**. The dataset ships a `num_tokens` column and the code rows I drew were **22,185 and 19,066 tokens**:
  > `[user]` "You will be given a competitive programming problem… provide a complete implementation in C++17… read input from standard input (cin)" → `[assistant]` "`<think>` Okay, I need to solve this problem where I have to find the number of ways to split a string…"
  At 16K `cutoff_len` (our current CPT config) a large fraction of these rows would be truncated mid-`<think>`, which is worse than excluding them. **Filter on `num_tokens` before use.**

### Bespoke-Stratos-17k, s1K / s1K-1.1, LIMO
- **`bespokelabs/Bespoke-Stratos-17k`** — **16,710** rows, `license: apache-2.0`. Teacher **DeepSeek-R1**, replicating the Berkeley Sky-T1 pipeline with Bespoke Curator: 5k coding (APPS + TACO), 10k math (AIME/MATH/Olympiads via NuminaMath), 1k science/puzzle (STILL-2). Math answers were checked with **GPT-4o-mini**, which raised the retained-correct rate from 25% → 73% (a partial OpenAI dependency, as filter not generator); code verified by execution.
- **`simplescaling/s1K-1.1`** — **1,000** rows, `license: mit`, arXiv:2501.19393. Traces from **DeepSeek-R1** (the original s1K used Gemini Flash Thinking Experimental). Seeds from AIME 1983–2024, NuminaMath/AoPS, TheoremQA; hand-curated for difficulty, diversity and quality. **Quality (sampled):** genuinely hard graduate-level items — one row I drew is an infinite-dimensional Hilbert space problem — with worked, numbered proofs. Note the dataset is a single parquet row group, so "deep offset" is not meaningfully testable at 1,000 rows.
- **`GAIR/LIMO`** — **817** rows, `license: apache-2.0`, arXiv:2502.03387. Reasoning chains distilled primarily from **DeepSeek-R1**; curated for chain quality rather than volume. **Quality (sampled):** AIME-style competition problems with long first-person reasoning ("Alright, so I have this geometry problem here. Let me try to parse it step by step.") and a clean short `answer` field. Same single-row-group caveat.
- **Collective takeaway:** these three are the "less is more" school applied to reasoning. At 1–17k rows they are cheap to include and cheap to drop.

### Magicoder / OSS-Instruct and Evol-CodeAlpaca (Wei et al., 2024)
- **Venue / Link:** ICML 2024 · arXiv:2312.02120 · https://arxiv.org/abs/2312.02120
- **`ise-uiuc/Magicoder-OSS-Instruct-75K`** — **75,197** rows, `license: mit`. Teacher **GPT-3.5-turbo-1106**; method: seed the model with a real open-source code snippet and have it invent a problem + solution inspired by it. The card itself says to "pay attention to OpenAI's usage policy" — **explicitly OpenAI-encumbered**.
- **`ise-uiuc/Magicoder-Evol-Instruct-110K`** — **111,183** rows, `license: apache-2.0`; a **decontaminated** version of evol-codealpaca-v1 using StarCoder's decontamination procedure.
- **`theblackcat102/evol-codealpaca-v1`** — **111,272** rows, `license: apache-2.0`. Teacher **GPT-4 (gpt-4-0314 / gpt-4-0613)**, WizardCoder-style Evol-Instruct over CodeAlpaca-20K seeds. This is the set Tulu 3 uses (as `evol_codealpaca_heval_decontaminated`, 107,276 rows after stripping 3.5% for HumanEval overlap). **OpenAI-derived.**
- **`sahil2801/CodeAlpaca-20k`** — 20,022 rows, `license: cc-by-4.0` on the mirror; generated by **text-davinci-003**. Obsolete; the original repo withheld weights specifically to respect OpenAI ToS and the LLaMA license.

### StarCoder2-Instruct (BigCode, 2024)
- **HF:** `bigcode/self-oss-instruct-sc2-exec-filter-50k` — **50,661** rows, **`license: odc-by`**, ungated.
- **Construction:** **fully self-aligned with no proprietary model anywhere.** StarCoder2-15B seeds itself from Python functions in The Stack (via the MultiPL-T pipeline) → concepts → instructions → responses, then filters by **execution feedback**.
- **Why it matters:** this is the only sizeable code-instruction set in this review with a completely clean provenance chain — permissive source code, open teacher, open license. If commercial-safety is a hard constraint, this replaces Magicoder and Evol-CodeAlpaca.

### OpenCodeReasoning / -2 (NVIDIA, 2025)
- **Venue / Link:** arXiv:2504.01943 · follow-up arXiv:2507.09075
- **HF:** `nvidia/OpenCodeReasoning` — **735,255** samples over **28,319 unique** competitive-programming questions (split_0 ~585k, split_1 ~167k). `license: cc-by-4.0`, ungated. `nvidia/OpenCodeReasoning-2` — 1.4M Python + 1.1M C++ samples over 34,799 unique questions.
- **Construction:** teacher **DeepSeek-R1**; seeds from CodeForces, CodeChef, AtCoder, Codewars, HackerEarth, AIZU, GeeksForGeeks, HackerRank, Kattis, LeetCode.
- **Important caveat from the paper:** the authors report that **execution filtering *hurt* benchmark accuracy**, so they deliberately prioritised instruction diversity over solution correctness. In other words this data is *not* execution-verified — a departure from the usual rejection-sampling assumption.

### DeepSeek-R1's own SFT data — **NOT RELEASED**
- arXiv:2501.12948 describes ~800k SFT samples (≈600k reasoning + ≈200k non-reasoning). DeepSeek released **checkpoints only**. Every "R1 SFT data" repo (OpenThoughts, Bespoke-Stratos, Dolphin-R1, AM-DeepSeek-R1-Distilled-1.4M, OpenR1-Math, Mixture-of-Thoughts) is a community reconstruction. `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` (~1.4M rows, DeepSeek-R1-671B teacher, EN/ZH) is **`cc-by-nc-4.0` — the only outright non-commercial reasoning set in this review.**

## 5. Instruction-Following / Constraint Data

### IFEval (Zhou et al., 2023) — evaluation only
- **Venue / Link:** arXiv:2311.07911 · https://arxiv.org/abs/2311.07911
- ~500 prompts over **25 verifiable instruction types** ("write more than 400 words", "mention keyword X at least 3 times", "wrap your answer in double quotes"). `license: cc-by-4.0`. **There is no IFEval training split.**

### Tulu-3 Personas Instruction-Following (AI2, 2024)
- **HF:** `allenai/tulu-3-sft-personas-instruction-following` — **29,980** rows, single train split, **`license: odc-by`**, ungated, `language: en`.
- **Construction:** synthetic persona-driven generation (extending the Persona-Hub method) where the constraint taxonomy is **explicitly borrowed from IFEval**. Part of Tulu 3 (arXiv:2411.15124).
- **Why it matters:** it is the de-facto standard IFEval-style training set — SmolTalk2 embeds 29,970 of these rows, and derives another 28,217 multi-turn rows from them using Qwen3-235B-A22B. Permissively licensed and English-only.

### Conifer (Sun et al., 2024)
- **Venue / Link:** arXiv:2404.02823 · `ConiferLM/Conifer` — **13.6K** rows (`train_sft`), `license: apache-2.0` on HF (paper CC-BY-4.0), ungated, `language: en`.
- **Construction:** GPT-4-driven multi-stage pipeline — seed instructions from ShareGPT, question reframing and filtering, constraint generation and recombination, answer generation with process feedback, plus easy-to-hard curriculum multi-turn variants. **OpenAI-derived.**

### AutoIF (Dong et al., 2024) — code only
- **Venue / Link:** arXiv:2406.13542 · https://github.com/QwenLM/AutoIF
- **No dataset is released** — the repo ships the pipeline plus 10–20 sample records per stage (Apache-2.0, code only). The method: 36 hand-written seed instructions → LLM-augmented instruction set → auto-generated **verification code and unit tests** → execution-feedback rejection sampling over ShareGPT queries. Run it yourself if you want AutoIF-style data.

### MUFFIN (Lou et al., 2024)
- **Venue / Link:** ICLR 2024 · arXiv:2312.02436 · https://github.com/RenzeLou/Muffin — **68,000** (instruction, input, output) instances, `license: mit`. Multi-faceted instruction curation: diversify the *input facets* rather than only scaling input-output pairs.

### FollowBench (Jiang et al., 2024) — evaluation only
- **Venue / Link:** ACL 2024 · arXiv:2310.20410 · Apache-2.0. Five constraint types (content, situation, style, format, example) with incremental difficulty levels. Benchmark, not training data.

## 6. Preference / RLHF Data (secondary — for the stage after SFT)

### UltraFeedback (Cui et al., 2023)
- **Venue / Link:** arXiv:2310.01377 · `openbmb/UltraFeedback` · `HuggingFaceH4/ultrafeedback_binarized`
- **Size:** ~64K prompts × 4 completions = 256K completions, 380K fine-grained feedback entries, ~1M derivable comparisons. The binarized H4 version is 187,405 rows total, **61,135 in `train_prefs`**.
- **Annotation:** LLM-as-judge — **GPT-4** scores each of 4 completions on instruction-following, truthfulness, honesty and helpfulness. Completions come from 17 models including GPT-4, GPT-3.5-Turbo and Bard.
- **License:** `mit` on both cards. The GPT-4 labels and OpenAI/Google completions are an inherited ToS risk the card does not surface.
- **Quality (sampled `train_prefs` at 0% / 50% / 95%):** **Mixed, with two visible defects.** First, ties are not filtered — the very first rows I drew have `score_chosen == score_rejected` (8.5 / 8.5 and 6.5 / 6.5), i.e. pairs that carry no preference signal at all. Second, the rejected side is heavily "As an AI language model" boilerplate, so a DPO run on this partly just teaches the model to stop refusing:
  > `[rejected]` "As an AI language model, I cannot personally develop habits for you. But, here are some tips…"
  > `[rejected]` "Hello! I'm here to help you with your question. However, I noticed that your question doesn't make sense. concatenate strings in python is not a valid or appropriate question." (score 3.0 vs chosen 8.5)
  Deeper rows are better-separated. **Filter out score ties before use.**

### HelpSteer2 / HelpSteer3 (NVIDIA, 2024–2025)
- **Venue / Link:** HelpSteer2 arXiv:2406.08673 · HelpSteer3 arXiv:2503.04378 (the separate "HelpSteer3-Preference" paper is a follow-up)
- **Size:** HelpSteer2 **21,362** samples (20,324 in the train parquet). HelpSteer3 **132,937** samples — Preference 40,476; Feedback 40,821; Edit 14,461; Edit-Quality 3,274; Principle 33,905 — across **28 languages plus code**.
- **Annotation:** **human**, and this is the differentiator. HelpSteer2 used Scale AI with ~1,000 annotators, 3–5 per sample, scoring on five 0–4 Likert axes; HelpSteer3 used Scale AI + Translated with ~7,000 annotators across 80+ regions.
- **License:** `cc-by-4.0`, ungated, no proprietary-output dependency identified for the labels. **The cleanest preference data in this review.**
- **Quality (sampled HelpSteer2 at 0% / 50% / 95%):** the per-axis integer labels (`helpfulness`, `correctness`, `coherence`, `complexity`, `verbosity`) are the real product — they let you build several different preference signals from one dataset. The annotators visibly punish refusal:
  > `[prompt]` "Simulate a twitter thread where Richard Heart announces the launch of PulseChain testnet v3…" · `[response]` "I'm sorry, but I cannot simulate a Twitter thread or predict future events. As an AI language model…" → `helpfulness 0, correctness 0`
  Prompts are short and real-user-shaped (one is literally `"c#"`). Two responses per prompt appear as consecutive rows.

### Nectar (Berkeley NEST / Starling, 2023)
- **Link:** `berkeley-nest/Nectar` — **182,954 prompts × 7 responses = 1,280,678 responses → 3.8M pairwise comparisons.** No published paper ("coming soon", **not verified**).
- **Annotation:** **GPT-4** ranks all 7 responses per prompt; the response pool is weighted toward GPT-4 outputs.
- **License:** `apache-2.0` **"under the condition that the dataset is not used to compete with OpenAI"** — that clause is on the card itself, and the card also inherits LLaMA license, OpenAI ToU and ShareGPT privacy terms. **The most explicitly restricted dataset in this review.** Do not use it in anything we release.

### Anthropic HH-RLHF (Bai et al., 2022)
- **Venue / Link:** arXiv:2204.05862 · `Anthropic/hh-rlhf`, `license: mit`, ungated
- **Size:** **169,352** helpfulness+harmlessness comparisons (161K train / 8.55K test): harmless-base 44,849, helpful-base 46,189, helpful-online 23,144, helpful-rejection-sampled 55,170. Plus 38,961 red-team transcripts (arXiv:2209.07858).
- **Annotation:** human crowdworkers (MTurk/Upwork) across three collection rounds.
- **License:** MIT, pure human data, no LLM-judge or proprietary-output dependency. **Clean but old** — the responses are from 2022-era models and are weak by current standards; useful for harmlessness, not for helpfulness quality.

### Skywork-Reward-Preference-80K-v0.2 (Liu et al., 2024)
- **Venue / Link:** arXiv:2410.18451 · `Skywork/Skywork-Reward-Preference-80K-v0.2`
- **Size:** **77,016 pairs** (not literally 80K). v0.2 removed **4,957 pairs** from the magpie-ultra-v0.1 subset that had significant n-gram overlap with RewardBench eval prompts; the removed pairs are published separately as `chrisliu298/Skywork-Reward-Preference-80K-v0.1-Contaminated`.
- **Construction:** a curated *remix*, not new annotation — subsampled from HelpSteer2, OffsetBias, WildGuard (adversarial) and the Magpie DPO series (Ultra / Pro-Llama-3.1 / Pro / Air), selected by average ArmoRM score with deliberate −0.1 / −0.05 penalties applied to the Air and Pro subsets to prioritise Ultra.
- **License:** **no license field on the card (not verified).**
- **Notes:** the explicit decontamination narrative is exemplary and worth copying; the missing license is not.

### Tulu 3 Preference Mixture (AI2, 2024)
- **Link:** `allenai/llama-3.1-tulu-3-8b-preference-mixture`, `license: odc-by`, ungated
- **Size:** the card says **272,898 pairs**; the paper's Table 15 says **271,409** for the 8B run (≈1,500 discrepancy, unreconciled). The 70B run uses 334,302.
- **Construction:** on-policy completions from the Tulu SFT checkpoint plus off-policy completions from four randomly chosen models per prompt, drawn from a pool spanning Mistral, Tulu 2, Yi, MPT, Gemma 2, InternLM2.5, Falcon, Qwen2.5, Llama 3/3.1 **and GPT-4 Turbo, GPT-4o and Claude 3.5 Sonnet**. The preference label itself comes from **GPT-4o-2024-08-06 as judge.**
- **License:** ODC-BY with an explicit card caveat that subsets are variously non-commercial and that third-party model outputs carry their own terms. **Both the completions and the labels are proprietary-model-derived** — the most load-bearing OpenAI dependency of any AI2 artifact.

### Older / minor
- **`stanfordnlp/SHP`** — 385,563 pairs across 18 subreddits, derived from **Reddit upvote and timestamp** signal rather than direct ratings. No formal license; governed by Reddit API terms (**not verified** as redistributable).
- **`openai/webgpt_comparisons`** (arXiv:2112.09332) — 19,578 human comparisons, no license tag.
- **`openai/summarize_from_feedback`** (arXiv:2009.01325) — HF configs total ~179K comparisons + 14.9K axis annotations, while the original GitHub release states 64,832 comparisons (**discrepancy unreconciled**). GitHub says "Modified MIT License"; no HF tag.

---

## 7. Data-Selection & Quality Literature — How to Pick a Subset

Every paper in this section reaches the same conclusion from a different direction: **a 1K–10K subset chosen by quality × complexity × diversity matches or beats a 10–50× larger raw mixture.** That is the single most actionable result for us, because it means the English SFT stage does not have to be expensive.

### LIMA — *Less Is More for Alignment* (Zhou et al., 2023)
- **Link:** arXiv:2305.11206 (NeurIPS 2023) · HF `GAIR/lima` (gated)
- **Method:** 65B LLaMA fine-tuned on exactly **1,000** hand-curated (prompt, response) pairs — Stack Exchange 400, wikiHow 200, r/WritingPrompts 150, Natural Instructions 50, author-written 200. No RLHF at all.
- **Numbers:** LIMA responses judged equivalent-or-preferred to **GPT-4 in 43%** of comparisons, **Bard 58%**, **DaVinci-003 65%**, **Claude 46%**. 88% of outputs met the prompt's requirements; 50% rated "excellent".
- **Claim ("Superficial Alignment Hypothesis"):** knowledge and capability come almost entirely from pretraining; alignment only teaches which sub-distribution of formats to emit.
- **Takeaway for us:** style/format consistency across the SFT set matters more than volume. Do not blend three mutually inconsistent response styles and expect the model to pick one.

### AlpaGasus (Chen et al., 2023)
- **Link:** arXiv:2307.08701
- **Method:** ChatGPT scores each Alpaca example 0–5 on a rubric; keep only score **≥ 4.5**.
- **Numbers:** Alpaca 52,002 → **9,229 kept (17.75%)**; Dolly-15k → 2,996 at the same threshold. AlpaGasus-13B (9K) vs Alpaca-52K on the Vicuna test set: **win 51.3% / tie 18.8% / lose 30.0%**. 7B training time drops 80 min → 14 min (**5.7×**).
- **Takeaway:** one cheap scalar judge score prunes ~82% of a noisy set *and improves* win-rate. The cheapest possible first filter.

### DEITA — *What Makes Good Data for Alignment?* (Liu et al., 2024)
- **Link:** arXiv:2312.15685 (ICLR 2024)
- **Method:** three axes — **Evol Complexity** and **Evol Quality** scorers (LLaMA-1-7B regressors trained on ChatGPT-ranked WizardLM-style evolved variants), then an **embedding diversity filter** (cosine threshold τ = 0.9, LLaMA-13B embeddings) applied greedily.
- **Numbers:** from a ~300K pool selects **6K or 10K**. DEITA-Mistral-7B (6K SFT + 10K DPO) → MT-Bench **7.55**, AlpacaEval **90.06%**, versus zephyr-beta trained on 200K SFT + 60K DPO (MT-Bench 7.34, AlpacaEval 90.60%). DEITA-LLaMA1-13B (6K) beats WizardLM-13B (70K, MT-Bench 6.35) and Vicuna-13B-v1.3 (125K, 6.39) at **6.46**.
- **Takeaway:** complexity + quality alone is not enough — the **diversity dedup is load-bearing**. This is the recipe I would actually run.

### Instruction Mining (Cao et al., 2023)
- **Link:** arXiv:2307.06290
- **Method:** 9 cheap indicators (length, reward score, perplexity, MTLD lexical diversity, KNN embedding distance, UniEval naturalness/coherence/understandability) fit by linear regression to *predicted fine-tuning loss*; BlendSearch then picks the subset size, exploiting the double-descent shape of the loss-vs-size curve.
- **Fitted rule:** `log L ∝ 0.0274 − 0.0078·Rew + 0.4421·Und − 0.3212·Nat − 0.1520·Coh` (R² = 0.522).
- **Numbers:** from a 100K OpenOrca pool, BlendSearch selects **2,532 examples (2.5%)**. InstructMining-10K reaches OpenLLM average **58.65** vs Vicuna-1.5-7B's 57.99 (125K) and base LLaMA-2-7B's 54.32; GPT-4 judge rates it better-or-equal to Vicuna-1.5-7B in **64.67%** of cases.
- **Takeaway:** you can approximate judge-quality selection with off-the-shelf NLP features and zero API spend.

### #InsTag (Lu et al., 2023)
- **Link:** arXiv:2308.07074 (ICLR 2024)
- **Method:** ChatGPT open-set instruction tagger produces >100K raw tags, normalized to **6,587** via frequency/lexical/semantic/association aggregation. Diversity = tag coverage; complexity = tags per query. Selection by "Complexity-first Diverse Sampling".
- **Numbers:** TAGLM-13b on **6,000** InsTag-selected samples scores MT-Bench **6.44–6.55**, above vicuna-13b-v1.3 (125K, 6.39), wizardlm-13b (70K, 6.35) and alpaca-13b (52K, 4.53).
- **Takeaway:** tag coverage is the cheapest usable proxy for "is my mixture diverse?" — directly applicable to auditing an English anchor set.

### Cherry LLM / IFD (Li et al., 2023) and Superfiltering (Li et al., 2024)
- **Links:** arXiv:2308.12032 · arXiv:2402.00530 (ACL 2024)
- **Method:** the **Instruction-Following Difficulty** score `IFD = s_θ(A|Q) / s_θ(A)` — how much the instruction actually helps the model produce the answer. Cherry LLM computes it with a "pre-experienced" model trained on 1,000 K-Means-clustered samples. **Superfiltering** shows the *ranking* is preserved if you compute IFD with an untrained **GPT-2 (124M)**.
- **Numbers (Cherry):** top **5%** of Alpaca beats 100% Alpaca — OpenLLM avg 52.06 vs 50.21, AlpacaEval 34.74% vs 26.46%; top **10%** of WizardLM nearly matches the full 70K (51.59 vs 52.79). Human eval on Cherry-Alpaca 5%: 49 wins / 25 ties / 26 losses vs full Alpaca.
- **Numbers (Superfiltering):** GPT-2 scoring takes **8 minutes** where LLaMA2-7B IFD takes 161 min (~20×), ChatGPT scoring 120 min, reward-model scoring 1400 min. Top 5% of Alpaca → OpenLLM avg 55.67 vs 55.25 full; AlpacaEval 33.04% vs 27.75%.
- **Takeaway:** **this is the highest value-per-GPU-hour filter available.** A 124M model on CPU-ish budget can rank a million-row pool.

### GRAPE — *The Best Instruction-Tuning Data are Those That Fit* (Zhang, Dai & Peng, 2025)
- **Link:** arXiv:2502.04194 (NeurIPS 2025)
- **Method:** for each instruction with multiple candidate responses from different teachers, keep the one with **highest probability under the target model itself** — i.e. select for distributional fit rather than for the "best" teacher.
- **Numbers:** beats stronger-teacher distillation by up to **13.8% absolute**; exceeds baselines trained on **3× more data** (+17.3%) and on realistic post-training pools **4.5× more data** (+6.1%); with **1/3 of Tulu-3 data and half the epochs**, LLaMA-3.1-8B surpasses Tulu-3-SFT by 3.5%.
- **Takeaway:** directly relevant to us — our target is a **Qwen-class** base, so Qwen-distilled data (SmolTalk2, OpenThoughts3-via-QwQ) is *a priori* a better fit than GPT-4-distilled data, and GRAPE gives a principled way to choose among duplicates.

### Other 2025 entries
- **THTB** (arXiv:2510.13892) — hardness-based selection; **5%** of data beats full-dataset baselines, and with domain guidance **2%** suffices.
- **ILA** (arXiv:2509.06463) — "Information Landscape Approximation", jointly optimizing semantic coverage and information depth; the abstract reports accelerated scaling versus selection baselines without a single headline percentage.

### Tulu 3 decontamination — the procedure to copy (Lambert et al., 2024, §3.2)
- **Link:** arXiv:2411.15124
- **Matching:** **8-gram** overlap on **prompts only** (completions are excluded because they are frequently regenerated).
- **Per-instance rule:** a test instance is flagged if **>50% of its tokens** share an 8-gram with the *same* training instance.
- **Dataset-level rule:** a training set counts as contaminated if **>2% of instances** in any dev or unseen eval overlap with it. Contaminated-against-unseen-eval sets are dropped entirely; contaminated-against-dev sets are dropped entirely unless that hurts performance, in which case only the matching instances are stripped.
- **Observed removal rates:** Evol-CodeAlpaca 3.5% (vs HumanEval), WildChat-GPT-4 / Safety 5.4%, WildJailbreak 0.7%, WildGuardMix 1.1%, **NuminaMath-TIR/MATH 11.3%**.
- **Takeaway:** NuminaMath in particular is *badly* contaminated against MATH out of the box. If we evaluate on MATH/GSM8K/HumanEval we must run this exact procedure ourselves rather than trusting upstream cards.

### Practical recipe distilled from the above
1. Assemble a large, deliberately heterogeneous pool.
2. Score every row for difficulty (IFD via a tiny model — Superfiltering) and quality (one cheap LLM-judge pass — AlpaGasus).
3. Deduplicate by embedding distance (DEITA's τ = 0.9) and audit tag coverage (#InsTag) so the survivors stay diverse.
4. Where several teachers answered the same prompt, keep the one your *own* base model likes best (GRAPE).
5. **Search over subset size** instead of assuming more is better — every paper here found an interior optimum.
6. Decontaminate last, with Tulu-3's 8-gram/50%/2% rule, against every benchmark you intend to report.

---

## 8. Comparison Table

Sizes are exact row counts read from the HF API / Parquet metadata unless marked *(paper)*. Licenses are the exact HF card tags. "Teacher" is the model that produced the responses.

| Dataset | Rows | Teacher / origin | Domains | License (HF tag) | Gated | Sampled here |
|---|---|---|---|---|---|---|
| `allenai/tulu-v1-sft-mixture` | 489,818 | mixed human + GPT-3.5/4 | general | odc-by (NC subsets) | no | no |
| `allenai/tulu-v2-sft-mixture` | 326,154 | FLAN + ShareGPT + GPT-4 | general | odc-by (NC subsets) | no | ✅ 3 depths |
| `allenai/tulu-3-sft-mixture` | 939,343 | GPT-4o personas + mixed | general, math, code, safety, **multilingual** | odc-by (NC subsets) | no | ✅ 4 offsets |
| `teknium/OpenHermes-2.5` | 1,001,551 | mostly GPT-4 | general | **none** | no | ✅ 4 depths |
| `BAAI/Infinity-Instruct` | 7,449,106 (7M) *(card)* | aggregated + evolved | general, math, code, EN+ZH | cc-by-sa-4.0 | **yes** | ✗ gated |
| `HuggingFaceTB/smoltalk` | 1,043,917 | Llama-3.1-405B (Magpie) + curated | general, math, code, tools | **none** | no | ✅ 3 depths |
| `HuggingFaceTB/smoltalk2` (SFT) | 3,383,242 / 19,292 M tok | **Qwen3-32B** + reuse | general, reasoning, tools, IF | **none** | no | ✅ 3 depths |
| `nvidia/Llama-Nemotron-Post-Training-Dataset` | ~33,011,757 | Qwen-2.5 family, DeepSeek-R1, Llama-3.3 | math 22.1M, code 10.1M, science, chat, IF, safety | cc-by-4.0 (+ODC-BY, +CC-BY-SA) | no | ✗ path error |
| `nvidia/Nemotron-Post-Training-Dataset-v1` | 25,659,642 | DeepSeek-R1-0528, Qwen3-235B-A22B | stem 20.7M, math, code, chat, tools | cc-by-4.0 | no | ✗ no parquet branch |
| `nvidia/Nemotron-Post-Training-Dataset-v2` | ~6.3 M *(secondary)* | DeepSeek-R1-0528, Qwen2.5/Qwen3 | + ja/de/it/es/fr | cc-by-4.0 | **yes** | ✗ gated |
| `open-thoughts/OpenThoughts-114k` | 113,957 | DeepSeek-R1 | math, code, science, puzzles | apache-2.0 | no | no |
| `open-thoughts/OpenThoughts3-1.2M` | 1,200,000 | **QwQ-32B** | math 850k, code 250k, science 100k | apache-2.0 | no | ✅ 3 depths |
| `Open-Orca/OpenOrca` | 2,942,029 | GPT-3.5 + GPT-4 | FLAN tasks | mit | no | ✗ path error |
| `Open-Orca/SlimOrca` | 517,982 | GPT-4 | FLAN tasks | mit | no | ✅ 3 depths |
| `cognitivecomputations/dolphin-r1` | 814,334 | DeepSeek-R1 + **Gemini 2.0 Flash Thinking** | reasoning + chat | apache-2.0 | no | ✅ `nonreasoning`, 3 depths |
| `WizardLMTeam/WizardLM_evol_instruct_V2_196k` | 143,000 | ChatGPT (Evol-Instruct) | general | mit *(GitHub says CC-BY-NC-4.0)* | no | ✅ 3 depths |
| `HuggingFaceH4/ultrachat_200k` | 207,865 (train_sft) | ChatGPT ↔ ChatGPT | general, C4-grounded | mit | no | ✅ 3 depths |
| `Magpie-Align/Magpie-Pro-300K-Filtered` | 300,000 | Llama-3-70B-Instruct | general | llama3 | no | ✅ 3 depths |
| `argilla/magpie-ultra-v1.0` | 999,960 (621,837 filtered) | Llama-3.1-405B-Instruct-FP8 | general, math, code | **none** | no | ✅ 3 depths |
| `databricks/databricks-dolly-15k` | 15,011 | **human** (Databricks staff) | 7 task categories | cc-by-sa-3.0 | no | ✅ 3 depths |
| `OpenAssistant/oasst2` | 128,575 msg rows (train parquet) | **human volunteers** | general, 35 languages | apache-2.0 | no | ✅ 3 depths |
| `HuggingFaceH4/no_robots` | 10,000 (9,500 train) | **human annotators** | 10 task categories | **cc-by-nc-4.0** | no | ✅ 3 depths |
| `GAIR/lima` | 1,000 / 50 / 300 | curated human + authors | general | other (≈CC-BY-NC-SA) | **yes** | ✗ gated |
| `allenai/WildChat-1M` | ~838k conv. (non-toxic split) | **real users** ↔ GPT-3.5/4 | open, 74 languages | odc-by | no (variants yes) | ✅ head only |
| `lmsys/lmsys-chat-1m` | 1,000,000 | **real users** ↔ 25 LLMs | open, 154 languages | bespoke agreement | **yes** | ✗ gated |
| `CohereLabs/aya_dataset` | 204,114 (202,362 train) | **human**, 65 languages | general | apache-2.0 | no | ✅ 3 depths |
| `nvidia/OpenMathInstruct-2` | 14 M *(paper)* / 600k unique Q | Llama-3.1-405B-Instruct | math | cc-by-4.0 | no | ✅ 3 depths |
| `meta-math/MetaMathQA` | 395,000 | GPT-3.5 rephrase + rejection sampling | math | mit | no | ✅ 3 depths |
| `AI-MO/NuminaMath-CoT` | 859,594 | human-authored, aggregated | math (+ noise) | apache-2.0 | no | ✅ 3 depths |
| `open-r1/OpenR1-Math-220k` | 450,258 rows | DeepSeek-R1 | math | apache-2.0 | no | ✅ 3 depths |
| `open-r1/Mixture-of-Thoughts` | 349,317 | DeepSeek-R1 | math 93.7k, code 83.1k, science 172.5k | **none** | no | ✅ 3 depths |
| `bespokelabs/Bespoke-Stratos-17k` | 16,710 | DeepSeek-R1 (GPT-4o-mini filter) | math, code, science | apache-2.0 | no | ✅ 3 depths |
| `simplescaling/s1K-1.1` | 1,000 | DeepSeek-R1 | hard math/science | mit | no | ✅ (1 row group) |
| `GAIR/LIMO` | 817 | DeepSeek-R1 | competition math | apache-2.0 | no | ✅ (1 row group) |
| `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | ~1.4 M | DeepSeek-R1-671B | reasoning, EN+ZH | **cc-by-nc-4.0** | no | no |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 75,197 | GPT-3.5-turbo-1106 | code | mit *(OpenAI policy noted)* | no | ✅ 3 depths |
| `theblackcat102/evol-codealpaca-v1` | 111,272 | GPT-4 | code | apache-2.0 | no | ✅ 3 depths |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 50,661 | **StarCoder2-15B (self)** | code | odc-by | no | ✅ 3 depths |
| `nvidia/OpenCodeReasoning` | 735,255 / 28,319 unique Q | DeepSeek-R1 | competitive code | cc-by-4.0 | no | ✗ path error |
| `allenai/tulu-3-sft-personas-instruction-following` | 29,980 | GPT-4o personas | IFEval-style constraints | odc-by | no | ✅ 3 depths |
| `ConiferLM/Conifer` | 13,600 | GPT-4 | constraint following | apache-2.0 | no | ✗ path error |
| `HuggingFaceH4/ultrafeedback_binarized` | 61,135 (train_prefs) | GPT-4 judge over 17 models | preference | mit | no | ✅ 3 depths |
| `nvidia/HelpSteer2` | 21,362 | **human**, 5 axes | preference | cc-by-4.0 | no | ✅ 3 depths |
| `nvidia/HelpSteer3` | 132,937 | **human**, 28 languages | preference | cc-by-4.0 | no | ✅ 3 depths |
| `berkeley-nest/Nectar` | 182,954 prompts / 3.8M pairs | GPT-4 ranking | preference | apache-2.0 **+ non-compete** | no | ✅ 3 depths |
| `Anthropic/hh-rlhf` | 169,352 | **human** crowdworkers | helpful/harmless | mit | no | ✅ 3 depths |
| `Skywork/Skywork-Reward-Preference-80K-v0.2` | 77,016 | remix (ArmoRM-scored) | preference | **none** | no | ✅ 3 depths |
| `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 272,898 | GPT-4o judge, mixed completions | preference | odc-by | no | ✅ 3 depths |

## 9. References

- Wang et al. 2023, *How Far Can Camels Go?* — arXiv:2306.04751
- Ivison et al. 2023, *Camels in a Changing Climate (Tülu 2)* — arXiv:2311.10702
- Lambert et al. 2024, *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* — arXiv:2411.15124
- OLMo Team 2025, *2 OLMo 2 Furious* — arXiv:2501.00656
- Allal et al. 2025, *SmolLM2* — arXiv:2502.02737
- Bercovich et al. 2025, *Llama-Nemotron: Efficient Reasoning Models* — arXiv:2505.00949
- NVIDIA 2025, *NVIDIA Nemotron Nano 2* — arXiv:2508.14444
- Guha et al. 2025, *OpenThoughts* — arXiv:2506.04178
- Mukherjee et al. 2023, *Orca* — arXiv:2306.02707
- Xu et al. 2023, *WizardLM: Empowering LLMs to Follow Complex Instructions* — arXiv:2304.12244
- Luo et al. 2023, *WizardCoder* — arXiv:2306.08568
- Ding et al. 2023, *Enhancing Chat Language Models by Scaling High-quality Instructional Conversations (UltraChat)* — arXiv:2305.14233
- Xu et al. 2024, *Magpie* — arXiv:2406.08464
- Wang et al. 2022, *Self-Instruct* — arXiv:2212.10560
- Köpf et al. 2023, *OpenAssistant Conversations* — arXiv:2304.07327
- Zhou et al. 2023, *LIMA: Less Is More for Alignment* — arXiv:2305.11206
- Zhao et al. 2024, *WildChat* — arXiv:2405.01470
- Zheng et al. 2023, *LMSYS-Chat-1M* — arXiv:2309.11998
- Singh et al. 2024, *Aya Dataset* — arXiv:2402.06619
- Lee et al. 2023, *Platypus* — arXiv:2308.07317
- Toshniwal et al. 2024, *OpenMathInstruct-1* — arXiv:2402.10176; *OpenMathInstruct-2* — arXiv:2410.01560
- Yu et al. 2024, *MetaMath* — arXiv:2309.12284
- Ye et al. 2025, *LIMO: Less is More for Reasoning* — arXiv:2502.03387
- Muennighoff et al. 2025, *s1: Simple Test-Time Scaling* — arXiv:2501.19393
- Wei et al. 2024, *Magicoder: Empowering Code Generation with OSS-Instruct* — arXiv:2312.02120
- Ahmad et al. 2025, *OpenCodeReasoning* — arXiv:2504.01943; *OpenCodeReasoning-II* — arXiv:2507.09075
- DeepSeek-AI 2025, *DeepSeek-R1* — arXiv:2501.12948
- Zhou et al. 2023, *Instruction-Following Eval (IFEval)* — arXiv:2311.07911
- Sun et al. 2024, *Conifer* — arXiv:2404.02823
- Dong et al. 2024, *Self-play with Execution Feedback (AutoIF)* — arXiv:2406.13542
- Lou et al. 2024, *MUFFIN* — arXiv:2312.02436
- Jiang et al. 2024, *FollowBench* — arXiv:2310.20410
- Cui et al. 2023, *UltraFeedback* — arXiv:2310.01377
- Wang et al. 2024, *HelpSteer2* — arXiv:2406.08673; *HelpSteer3* — arXiv:2503.04378
- Bai et al. 2022, *Training a Helpful and Harmless Assistant with RLHF* — arXiv:2204.05862
- Liu et al. 2024, *Skywork-Reward* — arXiv:2410.18451
- Chen et al. 2023, *AlpaGasus* — arXiv:2307.08701
- Liu et al. 2024, *What Makes Good Data for Alignment? (DEITA)* — arXiv:2312.15685
- Cao et al. 2023, *Instruction Mining* — arXiv:2307.06290
- Lu et al. 2023, *#InsTag* — arXiv:2308.07074
- Li et al. 2023, *From Quantity to Quality (Cherry LLM / IFD)* — arXiv:2308.12032
- Li et al. 2024, *Superfiltering* — arXiv:2402.00530
- Zhang, Dai & Peng 2025, *The Best Instruction-Tuning Data are Those That Fit (GRAPE)* — arXiv:2502.04194
- *THTB* — arXiv:2510.13892; *ILA* — arXiv:2509.06463
- Qwen Team 2025, *Qwen3 Technical Report* — arXiv:2505.09388

---

## 10. Access Limitations

Everything below was checked against the live HuggingFace API (`https://huggingface.co/api/datasets/<id>`, `gated` field) and, where relevant, by attempting to fetch `raw/main/README.md`.

**Gated — could not be read in this environment:**
- `BAAI/Infinity-Instruct` — `gated: auto`. Fetching the card returned *"Access to dataset BAAI/Infinity-Instruct is restricted and you are not in the authorized list."* All Infinity-Instruct figures in this review therefore come from the paper/secondary card mirrors, **not** from rows I read. **No rows were sampled; it is absent from the quality ranking.**
- `nvidia/Nemotron-Post-Training-Dataset-v2` — `gated: auto`. Same refusal message. v1 is ungated but could not be sampled either (see below).
- `GAIR/lima` — `gated: auto` (contact-info sharing required). LIMA's *content* is documented in the paper; no rows were read here.
- `lmsys/lmsys-chat-1m` — `gated: auto`, requires accepting the LMSYS-Chat-1M Dataset License Agreement. Not sampled.
- `allenai/WildChat-1M-Full` and the ImpACT-licensed WildChat variants remain gated; the ODC-BY `allenai/WildChat-1M` repo itself reported `gated: false` and was sampled.

**Environment issues encountered (documented so results are reproducible):**
- The machine's `HUGGINGFACE_HUB_TOKEN` environment variable holds an **expired** token ("User Access Token 'llama' is expired"). Sending it as a bearer header caused HTTP 401/anonymous fallback. All sampling below was therefore done **anonymously** with the token explicitly unset.
- `datasets-server.huggingface.co` enforces a hard anonymous quota ("0/500 requests remaining in current 300s window", HTTP 429). Row-level sampling was migrated to direct **Parquet row-group reads** over `HfFileSystem` against the `@~parquet` auto-converted branch, which is both faster and lets me address arbitrary depth rather than the viewer's paged offsets.

**Ungated but unsamplable via the Parquet route:**
- `nvidia/Nemotron-Post-Training-Dataset-v1` has **no `~parquet` branch at all** — the request returned *"Entry Not Found for url: …/tree/~parquet"*, i.e. HF's auto-conversion has not run for that repo (likely a size cut-off at 25.7 M rows).
- `nvidia/Llama-Nemotron-Post-Training-Dataset` (`SFT`/`chat`), `nvidia/OpenCodeReasoning` (`split_0`), `Open-Orca/OpenOrca` (`default`/`train`) and `ConiferLM/Conifer` (`default`/`train`) resolved to no Parquet files under the config/split names published on their cards. Their figures come from cards and papers only.

**Not released at all (frequently misattributed):**
- **DeepSeek-R1's ~800k SFT set** (≈600k reasoning + ≈200k non-reasoning, arXiv:2501.12948) was **never released** — DeepSeek published checkpoints only. Every "R1 SFT data" repo on the Hub (OpenThoughts, Bespoke-Stratos, Dolphin-R1, AM-DeepSeek-R1-Distilled, OpenR1-Math) is a community *reconstruction*, not the original.
- **AutoIF** (arXiv:2406.13542, `QwenLM/AutoIF`) ships pipeline code and 10–20 sample records per stage only — there is no released training set; you must run the self-play pipeline yourself.
- **IFEval** (arXiv:2311.07911) is evaluation-only (~500 prompts, 25 verifiable instruction types, CC-BY-4.0). There is no IFEval training split; Tulu-3 Personas-IF and Conifer are the training-side substitutes.
- **WizardCoder's** own Evol-Instruct data was never officially released; `theblackcat102/evol-codealpaca-v1` is the open reproduction.

**Facts I could not verify and have marked as such in the entries:**
- No license field of any kind is present on the HF cards of `teknium/OpenHermes-2.5`, `argilla/magpie-ultra-v1.0`, `Magpie-Align/Magpie-Air-300K-Filtered`, `open-r1/Mixture-of-Thoughts`, `Skywork/Skywork-Reward-Preference-80K-v0.2`, `HuggingFaceTB/smoltalk`, `HuggingFaceTB/smoltalk2`, and `lmsys/lmsys-chat-1m` (the last carries a bespoke agreement instead). "Unlicensed" is **not** the same as "permissive" — treat these as unresolved.
- `WizardLMTeam/WizardLM_evol_instruct_V2_196k` carries `license: mit` on HF while the authors' GitHub README states CC-BY-NC-4.0, academic use only. The conflict is unresolved; the more restrictive reading should govern.
- The exact reconciliation of `CohereLabs/aya_collection`'s total instance count (paper says 513M) against the per-config row counts shown on the card.
