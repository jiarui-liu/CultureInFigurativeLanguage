# Related work for the bidirectional (idiom <-> culture) CPT study

Compiled 2026-09-27. Candidate BibTeX: `OverleafCultureInFigurativeLanguage/new_refs_candidates.bib` (47 entries, no key collisions with `references.bib` / `custom.bib`).

Verification: every entry below was checked against its arXiv abs page and/or ACL Anthology `.bib` (ACL entries are copied verbatim from the Anthology with renamed keys). Venue fields for non-ACL venues (NeurIPS/ICML/ICLR/COLM/WWW) come from the arXiv "Comments"/"Journal-ref" field or the venue page. Items marked **[VENUE-PENDING]** have an accepted venue on arXiv but no proceedings entry yet; update them before camera-ready.

Already cited in the paper (not repeated): `yang2025qwen3`, `yang2025synthetic` (Synthetic CPT), `penedo2025fineweb2`, `myung2024blend`, `chiu2024culturalbench`, `shi2024culturebank`, `liu2024maps`, `li2024idiomkb`, `kabra2023multi`, `chengyubench2025`, `magdy2025jawaher`, `kinayat2025`, `gururangan2020dont`, `adilazuarda2024towards`, `opencsg_fineweb_zh`.

---

## 0. Novelty check (read first)

**Closest prior work: `attia2026figurative`** (Attia, Diab, Solorio; arXiv 2608.18361, Aug 2026). It tests transfer between figurative and cultural knowledge in *both directions*, in Arabic only. Setup: LoRA (r=4) **task fine-tuning** on about 1k QA-formatted benchmark items (culture: ArabCulture, Palm; figurative: FannOrFlop poetry, Jawaher proverbs; control: ArabicMMLU), with evaluation on Jawaher, Kinayat and AraDiCE. Models: ALLaM-7B, Fanar-1-9B, Qwen3-8B, Llama-3.1-8B. Findings:
- poetry -> idioms: +2.33 (p=0.021), the only significant effect;
- culture -> figurative: about 0;
- Palm fine-tuning *hurts* proverb accuracy for Arabic-centric models.

The authors conclude that the link "is not straightforwardly captured through fine-tuning." Their related-work section states that whether the two kinds of knowledge support each other "remains untested."

**Consequences for our claims:**
- We can **no longer claim to be the first to ask whether idiom knowledge and cultural knowledge transfer to each other in LLMs.** We must cite and contrast with Attia et al.
- Novelty still holds on these axes (none are covered by Attia et al. or anything else we found):
  1. **Continued pretraining on raw web documents** (not SFT on benchmark QA items), at a far larger token scale.
  2. A **token-matched random-document control** for both directions, which isolates content from extra-training effects. Attia et al. use an ArabicMMLU SFT control.
  3. **Four languages/cultures** (en/zh/hi/ar), not Arabic only.
  4. **Two kinds of intervention**: *selection* (a classifier picks idiom-bearing or culture-relevant documents) and *augmentation* (appended meaning tags or LLM-written cultural notes).
  5. The **symbolism probe** as a mechanistic bridge between the two sides (what an entity symbolizes in each culture's idioms).
- Their null or negative culture -> figurative result under SFT is a useful point of comparison. Whatever we find under CPT, whether it agrees or differs, is informative.
- We found **no work that runs idiom -> culture or culture -> idiom transfer through (continued) pretraining.** The pretraining-side idiom studies (`kunz2026idiomatic`, `mi2026decomposability`) look at *when/how idioms are acquired* during pretraining, not transfer to culture. The pretraining-side culture studies (`zhang2025crosslingualculture`, `li2025attributing`, `mukherjee2026maple`) look at cultural knowledge alone, not idioms.
- **Symbolism probe:** we did not find a text-only benchmark that asks what an entity (dog, dragon, red, ...) symbolizes in each culture's idioms or proverbs. The nearest are CUNIT (`li2024cunit`, cross-cultural concept matching for clothing/food), CULTURE-GEN and MEMOED (`li2024culturegen`, `li2025attributing`, "cultural symbols" = culture-marked entities generated for food/clothing), MultiMM (`yang2025multimm`, Chinese/English multimodal metaphor), and SAGE (arXiv 2512.07075, cross-cultural concept alignment; not added to the bib). Treat the novelty as "we did not find one" (unverified negative), not as a proven absence.

---

## (a) Culture-aware training data / culture-oriented (continued) pretraining

| key | one-line summary | relation to us |
|---|---|---|
| `li2024culturellm` | CultureLLM (NeurIPS'24): augments World Values Survey seeds with semantic data augmentation and fine-tunes culture-specific LLMs. | Culture injection via synthetic **SFT**; we inject via **CPT on web text** and test idioms, which they do not test. |
| `li2024culturepark` | CulturePark (NeurIPS'24): multi-agent LLM "cross-cultural dialogue" generates data for cultural fine-tuning. | Same contrast (synthetic dialogue SFT vs our CPT); a candidate baseline for culture-augmentation. |
| `elmekki2025nilechat` | NileChat (EMNLP'25): Egyptian/Moroccan 3B LLM built with controlled synthetic culture/persona data and retrieval-augmented pretraining. | Closest "culture-augmented CPT" precedent (LLM-written culturally grounded text in pretraining). It evaluates cultural/value alignment, **not idioms**. |
| `mukherjee2026maple` **[VENUE-PENDING: EMNLP 2026]** | MAPLE: pretrains 1B/3B LMs with geographic metadata (URL, country, continent) prepended; improves locale-dependent QA. | Controlled pretraining with appended culture/locale signal, matching our "append tags/notes" design. Their target is locale factual QA, not figurative language. |
| `sahu2026culturefunnel` | The Culture Funnel (arXiv'26): tags cultural markers across pretraining, SFT, alignment and reasoning datasets; cultural signal drops sharply after pretraining; releases 5.6M culturally tagged samples. | Motivates a culture-relevance classifier over pretraining data. Its tagged data could serve as seeds or validation for our classifier. |
| `li2025attributing` | MEMOED (ICLR'25): attributes culture-conditioned generations (food/clothing, 110 cultures) to memorized pretraining documents; high-frequency cultures produce more memorized "symbols". | Pretraining-frequency view of cultural knowledge; supports the idea that CPT data composition changes cultural behaviour. |
| `zhang2025crosslingualculture` | ACL'25 short: interpretable CPT/language-adaptation framework shows asymmetric cross-lingual transfer of cultural knowledge (frequency hypothesis). | **Closest CPT-based culture-transfer study**: transfer *across languages*, whereas ours is *across knowledge types* (idiom <-> culture). Useful method precedent for CPT with transparent data. |
| `alkhamissi2024investigating` | ACL'24: survey-simulation measure of cultural alignment; prompting in the native language and pretraining-language mix matter; proposes anthropological prompting. | Shows that pretraining language composition drives cultural alignment. Background for why CPT data matters. |
| `lertvittayakumjorn2025geocultural` | ACL'25 short: KB- and search-grounding help propositional culture MCQs but not open-ended cultural fluency. | Knowledge-injection-at-inference contrast; supports the separation between cultural propositional knowledge and fluency. |
| `nguyen2024culturax` | CulturaX (LREC-COLING'24): cleaned 6.3T-token corpus in 167 languages. | A possible source corpus for zh/hi/ar web documents (we use FineWeb-2 / FineWeb-Edu-zh). |

## (b) Classifier-based (FineWeb-Edu style) data selection

| key | summary | relation |
|---|---|---|
| `penedo2024fineweb` | FineWeb and **FineWeb-Edu** (NeurIPS'24 D&B): Llama-3-70B annotates educational value, a small regressor is trained on those labels, and it filters 15T tokens. | **Direct template** for our culture-relevance classifier (LLM annotation, distilled classifier, threshold). |
| `li2024datacomplm` | DCLM (NeurIPS'24 D&B): controlled benchmark for data curation; a fastText quality classifier is the best filter. | Justifies classifier filtering and the "fixed-token, vary-data" controlled comparison we follow. |
| `wettig2024qurating` | QuRating (ICML'24): LLM pairwise judgments of quality criteria are distilled into raters used to select data. | Alternative LLM-rater paradigm; cite alongside FineWeb-Edu. |
| `messmer2025multilingual` | NeurIPS'25 D&B: model-based (FastText/transformer) selection for multilingual pretraining (FineWeb-2 languages). | Evidence that classifier selection works for non-English (zh/ar) data. |
| `gunasekar2023textbooks` | phi-1: "textbook-quality" filtered and synthetic data. | Origin of the educational-value filtering idea; optional. |

## (c) Knowledge-augmented / rephrased / metadata-conditioned pretraining

| key | summary | relation |
|---|---|---|
| `maini2024rephrasing` | WRAP (ACL'24): rephrases web documents with an instruction LLM into styles (Wikipedia, QA), speeding pretraining and improving results. | Precedent for LLM-rewritten pretraining text; our cultural *notes* are appended, not a rewrite. |
| `su2025nemotroncc` | Nemotron-CC (ACL'25): classifier ensembles plus synthetic rephrasing of Common Crawl for long-horizon pretraining. | Combines (b) and (c), as our culture-filtered plus culture-augmented arms do. |
| `cheng2024instruction` | Instruction Pre-Training (EMNLP'24): augments raw corpora with synthesized instruction-response pairs during pretraining. | Precedent for appending generated knowledge to raw documents. Closest in form to our meaning tags and cultural notes. |
| `gao2025metadata` | MeCo (ICML'25): prepends URL metadata during pretraining, then cools down without it; accelerates pretraining. | Theory and precedent for appended "tags" in pretraining. Relevant to the design choice of meaning tags. |
| `allenzhu2024physics31` | Physics of LMs 3.1 (ICML'24): knowledge becomes extractable only with *augmented* (diverse and rephrased) pretraining data. | Mechanistic motivation for why augmentation (tags/notes) may be needed for transfer. |
| (cited) `yang2025synthetic` | Synthetic CPT (EntiGraph). | Already in custom.bib; keep citing. |

## (d) Language-adaptive continued pretraining (ar / zh / hi)

| key | summary | relation |
|---|---|---|
| `huang2024acegpt` | AceGPT (NAACL'24): Arabic localization of LLaMA-2 via CPT, SFT and RLAIF with culturally aligned rewards. | Arabic CPT with an explicit cultural goal. |
| `sengupta2023jais` | Jais (arXiv'23): Arabic-centric 13B model trained from scratch plus instruction tuning. | Arabic LLM background; not CPT. |
| `cui2023chinesellama` | Chinese-LLaMA/Alpaca (arXiv'23): vocabulary extension plus Chinese CPT and SFT of LLaMA. | Canonical Chinese CPT reference. |
| `gala2024airavata` | Airavata (arXiv'24): Hindi instruction-tuned model built on OpenHathi (itself a Hindi-CPT LLaMA-2). | Hindi adaptation reference (mostly SFT). |
| `choudhury2025nanda` | Nanda (arXiv'25): Hindi-centric Llama-3 with 65B-token Hindi CPT using block expansion (Llama-Pro). | Hindi CPT reference. |
| `nguyen2024seallms` | SeaLLMs (ACL'24 demo): regional CPT and vocabulary extension for Southeast Asian languages. | General regional-CPT precedent (optional). |

Positioning: these works adapt *language* broadly. We keep the language mix fixed and vary *content type* (idiom-bearing or culture-relevant vs random), which is a narrower, controlled intervention.

## (e) Idioms, figurative language and culture in LLMs

| key | summary | relation |
|---|---|---|
| `attia2026figurative` | See section 0. Bidirectional figurative <-> culture transfer via LoRA SFT, Arabic only. | **Must cite; closest prior work.** |
| `almheiri2026midi` | MIDI (ACL'26): native-speaker idioms in 18 languages (high/medium/low resource), sentence and conversation contexts, figurative vs literal. | Recent multilingual idiom benchmark; possible extra idiom eval (check en/zh/hi/ar coverage). |
| `kunz2026idiomatic` **[VENUE-PENDING: TACL]** | Tracks idiom preference during pretraining (Swedish): idioms are acquired slowly and forgotten quickly under further training. | **Pretraining-dynamics precedent for idioms**. Warns that CPT could erase idiom knowledge; supports our reverse-direction check. |
| `mi2026decomposability` **[VENUE-PENDING: ACL 2026]** | Idiom representations across pretraining checkpoints: surprisal and decomposability explain acquisition better than frequency. | Pretraining-side idiom acquisition; supports controlling for frequency. |
| `mi2025dice` | DICE (ACL'25): contrastive idiomatic vs literal context dataset; LLMs fail at context-dependent idiom disambiguation; frequency helps but does not guarantee success. | Idiom evaluation background. |
| `azime2025proverbeval` | ProverbEval (Findings NAACL'25): Ethiopian-language and English proverb MCQ/cloze/generation; large sensitivity to option order. | Culture-specific proverb benchmark; methodological caution for MCQ option order. |
| `yang2025chineseidiomtranslation` | COLM'25: evaluates LLM Chinese idiom (chengyu) translation with a taxonomy of errors. | zh idiom background next to `chengyubench2025`. |
| `liu2023crossing` | EMNLP'23: idiomatic MT via retrieval augmentation and loss weighting; multilingual idiom sets. | Idiom knowledge injection (training-side) in MT. |
| `zeng2022bart` | TACL'22: adapter trained on idiomatic sentences as a "non-compositional expert" for BART. | Early training-time idiom injection. |
| `yang2025multimm` | MultiMM (ACL'25): Chinese/English multimodal metaphor dataset from advertisements, studying cultural differences in metaphor. | Cross-cultural metaphor data; related to the symbolism probe. |

## (f) Cultural benchmarks and cultural symbols

| key | summary | relation |
|---|---|---|
| `rao2025normad` | NormAd (NAACL'25): 2.6k etiquette stories across 75 countries; tests norm adaptation at country/value/rule-of-thumb granularity. | Candidate culture eval (norm reasoning). |
| `nguyen2023candle` | CANDLE (WWW'23): 1.1M cultural commonsense assertions extracted from the web. | Cultural KB; possible source for culture-relevance seeds. |
| `fung2024cultureatlas` | CultureAtlas (arXiv'24): Wikipedia-derived multicultural knowledge acquisition and benchmark. | Culture eval / seed source. |
| `li2024culturegen` | CULTURE-GEN (COLM'24): culture-conditioned generations for 110 cultures; extracts "culture symbols" and measures markedness/diversity. | Closest to "symbols" terminology; entity-level, not idiom-based. |
| `li2024cunit` | CUNIT (COLM'24): contrastive cross-cultural concept matching (clothing, food). | Cross-cultural concept alignment; contrast with our idiom-grounded symbolism probe. |
| (cited) BLEnD, CulturalBench, CultureBank, ArabCulture, etc. | Already cited. | |

## (g) Base model citation

- There is **no Qwen3.5 text-LLM technical report** as of 2026-09. The only Qwen3.5 arXiv report is **Qwen3.5-Omni** (arXiv 2604.15804), a different (omni) model, so do not use it for Qwen3.5-9B.
- The Qwen3.5-9B Hugging Face model card recommends the blog citation, added as `qwen2026qwen35` ("Qwen3.5: Towards Native Multimodal Agents", Qwen Team, Feb 2026, https://qwen.ai/blog?id=qwen3.5).
- Recommended: cite `qwen2026qwen35` together with the existing `yang2025qwen3` (Qwen3 technical report, arXiv 2505.09388) for architecture lineage.

## (h) Statistics

| key | use |
|---|---|
| `dror2018hitchhiker` | ACL'18 guide to significance testing in NLP; justifies test choice. |
| `bergkirkpatrick2012empirical` | EMNLP'12 empirical study of paired bootstrap in NLP. |
| `efron1993bootstrap` | Bootstrap confidence intervals (book). |
| `mcnemar1947note` | Paired binary-outcome test for per-item accuracy of CPT vs control. |
| `holm1979simple` | Holm-Bonferroni correction across benchmarks and languages. |

---

## All new bib keys (47)

ACL Anthology (19): `dror2018hitchhiker`, `bergkirkpatrick2012empirical`, `almheiri2026midi`, `maini2024rephrasing`, `huang2024acegpt`, `nguyen2024seallms`, `su2025nemotroncc`, `cheng2024instruction`, `alkhamissi2024investigating`, `nguyen2024culturax`, `liu2023crossing`, `azime2025proverbeval`, `yang2025multimm`, `elmekki2025nilechat`, `zhang2025crosslingualculture`, `mi2025dice`, `lertvittayakumjorn2025geocultural`, `rao2025normad`, `zeng2022bart`

Hand-assembled (28): `li2024culturellm`, `li2024culturepark`, `mukherjee2026maple`, `sahu2026culturefunnel`, `li2025attributing`, `penedo2024fineweb`, `li2024datacomplm`, `wettig2024qurating`, `messmer2025multilingual`, `gunasekar2023textbooks`, `gao2025metadata`, `allenzhu2024physics31`, `sengupta2023jais`, `cui2023chinesellama`, `gala2024airavata`, `choudhury2025nanda`, `attia2026figurative`, `kunz2026idiomatic`, `mi2026decomposability`, `yang2025chineseidiomtranslation`, `nguyen2023candle`, `fung2024cultureatlas`, `li2024culturegen`, `li2024cunit`, `qwen2026qwen35`, `holm1979simple`, `mcnemar1947note`, `efron1993bootstrap`

## Unverified / caveats
- `kunz2026idiomatic`: TACL acceptance per arXiv. Volume/pages not verified.
- `mukherjee2026maple` (EMNLP 2026) and `mi2026decomposability` (ACL 2026): venue per arXiv comments. No Anthology bib pulled yet (MIDI's ACL 2026 Anthology entry does exist, so check the Anthology for `mi2026decomposability`).
- NeurIPS/ICML/ICLR/COLM/WWW entries: venue from arXiv comments or venue pages. Page numbers omitted.
- Holm 1979 / McNemar 1947 / Efron & Tibshirani 1993: standard bibliographic data from memory of well-known references (not web-fetched). The McNemar DOI should be double-checked.
- The absence of a prior idiom-symbolism benchmark and of prior CPT-based idiom <-> culture transfer studies is a result of our searches (about 20 queries), not a proof.
