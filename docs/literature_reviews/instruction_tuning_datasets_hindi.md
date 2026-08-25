# General-Purpose Hindi Instruction-Tuning / SFT / Post-Training Datasets — A Literature Review
# सामान्य-प्रयोजन हिन्दी इंस्ट्रक्शन-ट्यूनिंग / SFT / पोस्ट-ट्रेनिंग डेटासेट — एक साहित्य समीक्षा

> **Companion doc / सहयोगी दस्तावेज़:** this review covers the **post-training (SFT / preference) stage** for Hindi.
> For the **pretraining & continued-pretraining corpora** (Sangraha, FineWeb-2, IndicCorp, CulturaX …) see
> [`continued_pretraining_corpus_hindi.md`](continued_pretraining_corpus_hindi.md).
> यह समीक्षा हिन्दी के लिए **पोस्ट-ट्रेनिंग (SFT / अभिरुचि) चरण** को कवर करती है; **प्री-ट्रेनिंग व निरंतर-प्री-ट्रेनिंग
> कॉर्पोरा** के लिए [`continued_pretraining_corpus_hindi.md`](continued_pretraining_corpus_hindi.md) देखें।

---

## ⭐ Dataset Quality Ranking — Hands-On Deep Sampling / डेटासेट गुणवत्ता रैंकिंग — प्रत्यक्ष गहन नमूनाकरण

*Method / विधि:* the HF `datasets-server` **`/rows` endpoint returned HTTP 429 (rate-limited) throughout this session**,
so every sample below was obtained by reading the **auto-converted Parquet shards directly over HTTP byte-range
requests** (`refs/convert/parquet` + `pyarrow.ParquetFile` on a custom range reader). For each dataset/config: **(a)** a
*profile* over **8 randomly selected row-groups × up to 50 rows (≈400 real rows)** measuring script mix
(Devanagari-dominant / mixed / romanized-Hinglish / Latin-only), character-length quantiles, and a near-duplicate rate
(MD5 of the first 300 characters); **(b)** *verbatim reads at explicit deep offsets* — row 0, row 2,000, and one deep
offset between 5,000 and 500,000 depending on size. **All row counts below are exact `num_rows` values read from Parquet
footers / the HF `/size` API — not dataset-card claims.** "Devanagari-dominant" = ≥ 60 % of letter characters in the
Devanagari Unicode block. / इस सत्र में HF `datasets-server` का `/rows` एंडपॉइंट HTTP 429 दे रहा था, अतः सारा नमूनाकरण
सीधे Parquet शार्ड्स से HTTP बाइट-रेंज पठन द्वारा किया गया। हर डेटासेट के लिए ~400 वास्तविक पंक्तियों का प्रोफ़ाइल
(लिपि-मिश्रण, लंबाई-चतुर्थक, निकट-डुप्लिकेट दर) तथा पंक्ति 0 / 2,000 / गहरे ऑफ़सेट पर शब्दशः पठन। **नीचे दी गई सभी
पंक्ति-संख्याएँ Parquet फ़ुटर से पढ़ी गई वास्तविक संख्याएँ हैं, कार्ड के दावे नहीं।**

**The single most important finding / सबसे महत्वपूर्ण निष्कर्ष:** virtually every *large* "Hindi SFT dataset" on the Hub
is **English instruction data pushed through IndicTrans2, NLLB, SeamlessM4T, Google Translate or GPT-3.5**, and the
damage is visible in real rows at every depth — calqued entity names, broken MCQ option letters, degenerate token loops,
`<unk>` corruption, and half-translated sentences. Genuinely **native-authored** Hindi assistant data exists only at the
**~7 k – 100 k** scale (wikiHow-hi, Anudesh prompts, Samvaad-hi, Aya-hi = **1,153 rows**). Plan the mixture around that
fact rather than around raw row counts. / हब पर लगभग हर *बड़ा* "हिन्दी SFT डेटासेट" वस्तुतः अनूदित अंग्रेज़ी डेटा है, और
अनुवाद-दोष हर गहराई पर वास्तविक पंक्तियों में दिखते हैं। वास्तव में **मूल रूप से हिन्दी में लिखा गया** सहायक-डेटा केवल
~7 हज़ार–1 लाख के पैमाने पर है (Aya का हिन्दी मानव-लिखित हिस्सा तो मात्र **1,153** पंक्तियाँ है)। मिश्रण की योजना कच्ची
पंक्ति-संख्या के बजाय इसी तथ्य के इर्द-गिर्द बनाएँ।

---

### Tier A — use these first / श्रेणी अ — इन्हें पहले उपयोग करें

**A1. `ai4bharat/indic-align` → the `hin_Deva` column** (Wiki-Conv **141,435** · Wiki-Chat **198,254** · WikiHow
**20,313** · Indic-ShareLlama **21,171**; plus Anudesh **36,820** and Toxic-Matrix **90,352**) — **CC-BY-4.0.**
The only *large* Hindi SFT pool built **exclusively from open, license-friendly teacher models** (Llama-2-70B-Chat +
Mixtral-8x7B-v0.1), so it is the only one that is cleanly usable commercially. Sampled `hin_Deva` at rows 0 / 2,000 /
20,000 in five configs: **100 % Devanagari, 0.0–0.8 % near-dup, no depth degradation**. Wiki-Conv (median 960 chars) is
short grounded factual multi-turn; Wiki-Chat (median 4,399) is long open conversation; WikiHow-T (median 3,364) is
procedural; Indic-ShareLlama (median 4,242) is real ShareGPT first-turn prompts answered by Llama-2-70B. | यह एकमात्र
*बड़ा* हिन्दी SFT स्रोत है जो पूर्णतः खुले, लाइसेंस-अनुकूल शिक्षक मॉडलों से बना है, अतः व्यावसायिक उपयोग के लिए स्वच्छ है।
पाँचों कॉन्फ़िग में 100 % देवनागरी, गहराई पर कोई क्षरण नहीं।
> `मेघालय की राजधानी क्या है? || मेघालय की राजधानी शिलोंग है।` — *EN: "What is the capital of Meghalaya? — The capital of Meghalaya is Shillong."* (Wiki-Conv, row 0)
> **Verdict / निर्णय: SYNTHETIC (Llama-2-70B / Mixtral, Wikipedia-grounded) → IndicTrans2-translated — but by a wide margin the best-behaved translation in this survey.** Residual artefact: IndicTrans2's acronym spacing, e.g. `आई. ए. टी. ए. कोड ... वी. जी. ए. है` for "IATA code … is VGA". **Deep-offset: robust.**
> ⚠️ **Two structural traps.** (i) Most configs are **n-way parallel with one column per language-script**
> (`eng_Latn, hin_Deva, hin_Latn, mar_Deva, npi_Deva, san_Deva, urd_Arab …`) — you must **project the `hin_Deva`
> column**, otherwise you ingest 15–20 languages. (ii) The **`Anudesh` and `IndoWordNet` configs do NOT have per-language
> columns**; `Anudesh` is raw multi-language crowd interactions (sampled profile: **54 % Latin-only, 36 % Devanagari**,
> and the Devanagari includes **Marathi** — row 20,000 is Marathi, not Hindi), while `IndoWordNet` carries a separate
> `language` field. | ⚠️ दो संरचनात्मक जाल: अधिकांश कॉन्फ़िग भाषा-वार स्तंभों वाले हैं (`hin_Deva` चुनें), पर `Anudesh` और
> `IndoWordNet` ऐसे नहीं हैं — `Anudesh` में मराठी आदि मिश्रित हैं।

**A2. `sarvamai/samvaad-hi-v1` — 101,476 conversations — Apache-2.0.**
The best openly-licensed *India-grounded chat* set. Profile: **45 % Devanagari-dominant, 25 % mixed, 24 % Latin-only,
6 % romanized-Hinglish; median 2,105 chars; 0.0 % near-dup**; deep offsets (2 k / 20 k / 80 k) stay on-topic and varied
(politics, cinema, railways, festivals, colonial history). Many prompts are English-asking-for-Hindi (*"… please answer
in Hindi"*), which is *useful cross-lingual instruction data* but means **fewer than half the rows produce Hindi
output** — filter if you want a pure-Hindi mixture. | खुले लाइसेंस वाला सर्वश्रेष्ठ भारत-केंद्रित चैट सेट; परंतु आधे से
कम पंक्तियाँ ही हिन्दी-आउटपुट देती हैं, अतः छाँटना आवश्यक।
> `भारतीय त्योहार किस तरह से घर की सजावट और डिजाइन को प्रभावित करते हैं?` — *EN: "How do Indian festivals influence home decoration and design?"* (row ≈ 2,000)
> **Verdict / निर्णय: MODEL-GENERATED over Indic source text; light-to-moderate translationese.** Artefacts at depth: `पचंडी ट्रेन स्थानक` (Marathi-flavoured *sthānak* where Hindi uses *स्टेशन*), `अपनी M.A.B डिग्री` (untranslated degree token). **Deep-offset: stable.**

**A3. `ai4bharat/indic-instruct-data-v0.1` → `wikihow/hi` — 6,055 — CC-0.**
The **only genuinely native-authored, human-moderated long-form Hindi** in the survey: scraped from `hi.wikihow.com`, so
it is real colloquial written Hindi with naturally nativised English loans in Devanagari (`ट्रीट`, `कम्फ़र्टेबल`,
`इंग्रेडिएंट्स`, `सब्स्टीट्यूट`) rather than the Sanskritised register that MT produces. **97 % Devanagari, median
11,986 chars, 2.0 % near-dup.** | सर्वेक्षण में एकमात्र वास्तव में मूल-लिखित, मानव-संपादित दीर्घ हिन्दी सामग्री —
स्वाभाविक बोलचाल की हिन्दी, अनुवाद की संस्कृतनिष्ठ शैली नहीं।
> `अगर आपने पहले कभी एक केक नहीं बनाया है या फिर आप एक ऐसा ट्रीट तैयार करना चाहते हैं, जो ज्यादा फ़ैन्सी न हो, तो फिर एक प्लेन केक (plain cake) बेक करके देखें।` — *EN: "If you have never baked a cake before, or you want to make a treat that isn't too fancy, try baking a plain cake."*
> ⚠️ **Shipping bug, verified at rows 0 and 2,000:** the `messages` user turn embeds the **raw URL-percent-escaped title**, e.g. `कैसे %e0%a4%8f%e0%a4%95 %e0%a4%aa%e0%a5%8d%e0%a4%b2%e0%a5%87%e0%a4%a8 ... (make a plain cake)?`. Do **not** train on `messages` as shipped — rebuild the prompt by URL-decoding `url`/`title`, and take the answer from `intro` + `steps`. | ⚠️ `messages` का यूज़र-टर्न URL-एन्कोडेड कचरा है; प्रॉम्प्ट को `url`/`title` को डिकोड करके पुनर्निर्मित करें।
> **Verdict / निर्णय: NATIVE — human-written Hindi, not translated.**

**A4. `ai4bharat/indic-instruct-data-v0.1` → `anudesh/hi` — 7,577 — CC-BY-4.0.**
**100 % Devanagari, median 1,941 chars, 1.8 % near-dup.** The *prompts* are natively written by Indian crowdworkers under
intent/domain/language guidelines, so the prompt distribution is genuinely Indian (Kolkata pollution essays, poems about
the Tamil month *Aippasi*, ML-interview study plans). The *responses* are Llama-2-70B outputs pushed through
IndicTrans2. | प्रॉम्प्ट भारतीय क्राउडवर्करों द्वारा मूल रूप से लिखे गए हैं (अतः प्रश्न-वितरण वास्तव में भारतीय है);
उत्तर Llama-2-70B के IndicTrans2-अनूदित आउटपुट हैं।
> `कोलकाता में प्रदूषण के बारे में निबंध लिखने में मेरी सहायता करें` — *EN: "Help me write an essay about pollution in Kolkata."*
> **Verdict / निर्णय: NATIVE prompts + MACHINE-TRANSLATED synthetic responses.** Response-side artefacts: `कंप्यूटर दृष्टि` (calque for "computer vision"), `एम. एल.` acronym spacing, mangled clock times (`8: 00 AM-8:30 AM`, `देर सुबह 5 बजे`). **Deep-offset: stable.**

**A5. `BhabhaAI/Hi-Instruct-v0` — 9,969 — licence unspecified.**
**100 % Devanagari, median 1,894 chars.** Seed-word-conditioned synthetic Hindi Q&A that reads as *directly generated in
Hindi*: no calques, correct Devanagari markdown lists, modern register, genuinely open-ended prompts. Downsides:
**7.2 % near-dup** (seed words recur) and only ~10 k rows. | बीज-शब्द आधारित सिंथेटिक हिन्दी; सीधे हिन्दी में उत्पन्न
प्रतीत होता है, अनूदित नहीं। कमी: 7.2 % निकट-डुप्लिकेट और केवल ~10 हज़ार पंक्तियाँ।
> `यदि आप किसी ऐसे ग्रह पर पैदा होते जहाँ गुरुत्वाकर्षण पृथ्वी से दोगुना होता, तो आपकी शारीरिक संरचना और जीवनशैली में क्या परिवर्तन होते?` — *EN: "If you were born on a planet with twice Earth's gravity, what changes would occur in your physiology and lifestyle?"*
> **Verdict / निर्णय: DIRECTLY-GENERATED Hindi (teacher model undocumented — treat licence/provenance as unknown).**

**A6. `CohereLabs/aya_dataset` filtered to `language == "Hindi"` — 1,153 rows — Apache-2.0.**
Tiny, but it is **human-written by fluent Hindi speakers** and culturally grounded in a way nothing else here is.
Profiling the 84 Hindi rows we could reach across 12 spaced row-groups: **93 % Devanagari-dominant, median 341 chars.**
It contains genuine Hindi-classroom grammar tasks and Indian everyday knowledge that no translated set produces. |
अत्यंत छोटा, पर **धाराप्रवाह हिन्दी वक्ताओं द्वारा स्वयं लिखा गया** और सांस्कृतिक रूप से आधारित — ऐसा और कुछ नहीं है।
> `दाल मखनी कैसे बनाते हैं?` — *EN: "How do you make dal makhani?"*
> `पान कहीं आगे खा लेंगे। (कर्मवाच्य में बदलिए) || पान कहीं आगे से खा लेंगे।` — *EN: "We'll eat paan somewhere ahead. (Convert to passive voice)"* — a Hindi-grammar exercise that could only be written by a Hindi speaker.
> **Verdict / निर्णय: NATIVE, human-written. Upsample it heavily; there is nothing else of this quality.**

---

### Tier B — good for bulk, with caveats / श्रेणी ब — मात्रा के लिए उपयोगी, चेतावनी सहित

**B1. `BhabhaAI/openhermes-2.5-hindi` — 620,211 — licence unspecified (OpenHermes-2.5 is GPT-4-derived).**
Profiling the `conversations` (Hindi) column only: **93 % Devanagari, median 1,314 chars, 1.5 % near-dup**, stable to row
300,000. The single largest fluent Hindi assistant corpus that is *not* WordNet templates. | टेम्पलेट-रहित सबसे बड़ा
धाराप्रवाह हिन्दी सहायक-कॉर्पस।
> ⚠️ **MCQ option letters are half-transliterated, which silently breaks every multiple-choice row:** `... उ. यह नमूना तैयार करने ... / B. यह ... / ग. यह ... / D. यह ...` — options A and C became Devanagari `उ.` and `ग.`, B and D stayed Latin.
> Other artefact: `गैर-छलांग वाले वर्ष` for "non-leap year" (literally "non-jump year"). **Deep-offset: stable but the MCQ bug is pervasive.**

**B2. `apurvagup/ultrachat_hindi_seamless` — 185,542 (`train_sft`) — licence unspecified.**
**100 % Devanagari, median 4,989 chars, 0.0 % near-dup**, stable to row 100,000. UltraChat translated with SeamlessM4T;
long, well-structured multi-turn assistant answers — the best *long-form format* teacher in the survey. | UltraChat का
SeamlessM4T-अनुवाद; दीर्घ, सुसंरचित बहु-वार्ता उत्तर।
> `एक आपूर्तिकर्ता को कम से कम 150 शब्दों का एक पेशेवर ईमेल लिखें जो विशिष्ट उत्पादों की कीमतों और उपलब्धता के बारे में पूछताछ करता है।` — *EN: "Write a professional email of at least 150 words to a supplier enquiring about prices and availability of specific products."*
> **Verdict / निर्णय: MACHINE-TRANSLATED (SeamlessM4T) synthetic — but cleanly so. Deep-offset: robust.**

**B3. `shreyas18/Hindi_instruct_1_5M_v1` — 1,488,730 · `atharvanighot/Hindi-Instruct-500K` — 508,609 — both licence-less.**
**100 % Devanagari; median 1,846 / 2,135 chars; 0.5 % / 0.0 % near-dup**, stable to rows 500,000 / 300,000. **These two are
the same upstream data** — rows 0 and 2,000 are byte-identical across both repos, so **deduplicate before mixing**. The
answer style (`यहाँ ... कुछ महत्वपूर्ण अंतर हैं:` + `•` bullets) points to a translated Claude/HH-style corpus. | ये दोनों
एक ही स्रोत से हैं (पंक्ति 0 व 2,000 बाइट-दर-बाइट समान) — मिलाने से पहले डुप्लिकेट हटाएँ।
> Entity mistranslation, row ≈ 2,000 (identical in both): `लोकप्रिय विकल्पों में स्लैक, **प्रवचन**, मेलचिम्प, आदि शामिल हैं।` — the forum software **Discourse** was translated to `प्रवचन` ("a religious discourse/sermon").
> Also `1।  2।  3।` — Latin digits followed by a Devanagari *danda* instead of a period, a systematic MT artefact.
> **Verdict / निर्णय: MACHINE-TRANSLATED, undocumented provenance. Deep-offset: stable but entity-lossy.**

**B4. `BhabhaAI/orca-math-word-problems-200k-hindi-filtered` — 188,943 — MIT.**
**100 % Devanagari, median 980 chars, 0.0 % near-dup.** The largest clean Hindi math-reasoning pool available.
Model-output-derived upstream (Microsoft Orca-Math is GPT-4-generated). | उपलब्ध सबसे बड़ा स्वच्छ हिन्दी गणित-तर्क संग्रह।

**B5. `GenVRadmin/Aryabhatta-Orca-Maths-Hindi` — 200,000 — MIT.**
**100 % Devanagari, median 846, 0.0 % near-dup.** Same upstream lineage as B4 — **use one, not both**. | B4 जैसा ही स्रोत; दोनों में से एक ही लें।

**B6. `MBZUAI/Bactrian-X` config `hi` — 67,017 — CC-BY-NC-4.0.**
**96 % Devanagari, median 428, 0.5 % near-dup**, stable to row 60,000, fluent. But prompts are Google-Translate of
Alpaca + Dolly and responses are `gpt-3.5-turbo` → **non-commercial licence *and* OpenAI-output-derived**. | धाराप्रवाह
और गहराई तक स्थिर, परंतु ग़ैर-व्यावसायिक लाइसेंस + OpenAI-व्युत्पन्न।
> Task destroyed by transliteration, row 1: `पहचानें कि कौन सा वाद्य यंत्र स्ट्रिंग या पर्क्यूशन है: वॉबल बोर्ड, शीथोल्ट` — "Wobble board, Sheetholt" left as opaque transliterations, so the question is unanswerable in Hindi.

**B7. `FreedomIntelligence/alpaca-gpt4-hindi` — 49,969 · `evol-instruct-hindi` — 59,022 · `sharegpt-hindi` — 3,142 (Apache-2.0).**
alpaca-gpt4-hindi: **99 % Devanagari, median 553, 0.0 % near-dup** — clean. evol-instruct-hindi: **78 % Devanagari, 18 %
mixed, median 1,730** — the "mixed" rows are where translation **broke code and English answers leaked**. sharegpt-hindi:
**90 % Devanagari but 7.8 % near-dup** on only 3 k rows. | alpaca संस्करण स्वच्छ; evol संस्करण में कोड टूटा और अंग्रेज़ी उत्तर लीक हुए।
> Language leak, row ≈ 2,000 of `evol-instruct-hindi`: prompt in Hindi (`... ट्वीट बनाएं ...`), answer entirely in English (`"Excited to announce the formation of the new American Football team ..."`). **Deep-offset: degrades.**
> Code corruption, row 0: `df <- data.frame(नाम=c("जॉन", "मैरी", "पीटर"), आयु=c(25, 30, 35), वेतन=c(50000, 60000, 70000))` — R identifiers translated into Devanagari.

**B8. `iamshnoo/alpaca-cleaned-hindi` — 51,760 · `saillab/alpaca-hindi-cleaned` — 41,601 — licences unspecified.**
97–100 % Devanagari, median 553–630, 0.0 % near-dup. Clean but **Alpaca-shallow**, with visible MT repetition. | स्वच्छ पर उथला; अनुवाद-पुनरावृत्ति दिखती है।
> Verbatim self-duplication inside one answer, row 1: `प्रकाश के लिए उपयोग किए जाने वाले योजक रंग प्रणाली में, प्राथमिक रंग लाल, हरे और नीले (आरजीबी) हैं। प्रकाश के लिए उपयोग किए जाने वाले योजक रंग प्रणाली में, प्राथमिक रंग लाल, हरे और नीले (आरजीबी) हैं।` (the identical sentence twice).

**B9. `bingbangboom/gsm8k-hindi` — 7,473 (`train_main`) / 7,473 (`train_socratic`) — MIT.**
**100 % Devanagari, median 518, 1.5 % near-dup.** Small but clean; pairs naturally with B4/B5. | छोटा पर स्वच्छ।

**B10. `CohereLabs/aya_collection_language_split` config `hindi` — 3,772,864 train (+283,272 val / 325,548 test) — Apache-2.0.**
**84 % Devanagari, 15 % mixed, median 301 chars, 0.0 % near-dup.** Enormous and permissively licensed, but overwhelmingly
**templated NLP tasks NLLB-translated from FLAN**. Use as a *task-diversity garnish*, never as the backbone. | विशाल एवं
उदार-लाइसेंस, पर मुख्यतः FLAN के NLLB-अनूदित टेम्पलेट कार्य; केवल विविधता हेतु।
> Half-translated sentence, row 1: `श्री रॉबर्ट Vyner, गंभीरता से, के रूप में वह एक में reclined डेक-कुर्सी पर` — *"Mr Robert Vyner, gravely, as he reclined in a deck-chair"*, with `Vyner` and `reclined` left in English.
> `<unk>` corruption + numeral chaos in one target: `... 484<unk>425 ईसा पूर्व ९. अनातोलियाई तुर्क **दस।** तुर्की 11. हाँ हाँ` — the en-dash became `<unk>`, and list item **"10."** was translated into the *word* `दस।` ("ten."). **Deep-offset: uniformly noisy.**

---

### Tier C — usable only after heavy filtering / श्रेणी स — भारी छँटाई के बाद ही उपयोगी

**C1. `ai4bharat/indic-instruct-data-v0.1` → `flan_v2/hi` (67,463) and `lm_sys/hi` (50,000).**
The two biggest slices of IndicInstruct and the two worst. The dataset helpfully ships `backtranslated_*` fields and
`quality_metrics`, which **prove the paper's chrF++ ≥ 50 filter is far too lenient**. | ये IndicInstruct के दो सबसे बड़े
और सबसे ख़राब हिस्से हैं; इनके अपने `quality_metrics` सिद्ध करते हैं कि chrF++ ≥ 50 की छँटाई अत्यंत ढीली है।
> Degenerate MT that **still scored `chrF++ = 94.33`** (`flan_v2-2002`): `समस्याः रिक्त स्थान जोड़ें-यदि अतिरिक्त हो तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-तो-...` — the FLAN `word_segment` task ("Add spaces: …") is *semantically void* once translated, and the model looped on `तो`.
> Literal translation of a product name (`lm_sys/hi`, row 0): `कृपया मेरी बीमार बूढ़ी दादी का अभिनय करें जो मुझे सोने के लिए **खिड़की 11** की कुंजी पढ़ती थीं।` — "Windows 11" became `खिड़की 11` ("window 11").
> Corroborating the *Aya* post-editing study, the classic Dolly fish question survives as nonsense in **both** AI4Bharat releases: `मछली की एक प्रजाति कौन सी है? **टोप या रस्सी**` (IndicInstruct `dolly/hi`, "cap or rope") and — worse — `मछली की एक प्रजाति कौन सी है? **रस्सी या रस्सी**` (IndicAlign `Dolly_T` `hin_Deva`, "**rope or rope**"). The original was "Tope or Rope".
> ⚠️ `lm_sys/hi` also inherits the **LMSYS-Chat-1M Dataset License Agreement**, which is *not* a standard open licence.

**C2. `ai4bharat/indic-instruct-data-v0.1` → `dolly/hi` (15,011) / `ai4bharat/indic-align` → `Dolly_T` (15,011).**
Both are clean Devanagari (median 949 / 269 chars) and *mostly* fine, but **named entities are systematically
destroyed** — a real risk for a culture/knowledge-focused project. | दोनों स्वच्छ देवनागरी हैं, पर नामित-इकाइयाँ
व्यवस्थित रूप से नष्ट होती हैं — संस्कृति/ज्ञान-केंद्रित परियोजना के लिए वास्तविक जोखिम।
> `मकई के गोमांस` for **corned beef** (literally "beef of maize"); `हज़ार द्वीप ड्रेसिंग` for Thousand Island dressing; `अल्फ्रेड स्कीइंग` for **Alfred Scheuing**; `रूबेन के स्वादिष्ट व्यंजन` for **Reuben's Delicatessen**; `झरने और झरने` for **"springs and falls"** (both rendered "waterfalls"); `लाल चट्टानों का एम्फीथिएटर` for **Red Rocks Amphitheatre**; `अल्पाइन घाटी संगीत रंगमंच` for **Alpine Valley Music Theatre**.

**C3. `pranjalchitale/indicsft` — 25,030,167 rows, 43 GB — no dataset card, no licence.**
Profile: **74 % Latin-only, 16 % Devanagari, 10 % mixed; median 1,275; 3.5 % near-dup.** English system prompts +
native-script task instances across many Indic languages, **not language-split**. A sampled Hindi row's system prompt
declares the task *"text-simplification"* while the user turn asks to `इस वाक्य का एक अधिक जटिल संस्करण उत्पन्न करें`
("generate a **more complex** version of this sentence") — task/label inconsistency. Valuable as a scale reference;
unusable as-is without provenance or a licence. | विशाल, पर बिना कार्ड/लाइसेंस, भाषा-वार अविभाजित, और कार्य-लेबल असंगत।

**C4. `zicsx/indic-align-hindi` — 13,310,858 rows — licence unspecified.**
A **Devanagari-filtered** re-cut of IndicAlign. Rows 0 and 2,000 are genuine natural Hindi (wedding-gift advice, Hindu
vs. *Sanātanī* terminology). **But at offset 200,000 the rows are Bodo, not Hindi** — the filter was *script*-based, so
Bodo / Marathi / Nepali / Sanskrit leak in, and the median length collapses to 261 chars because IndoWordNet templates
dominate the tail. This is exactly the failure that shallow sampling misses. | देवनागरी-आधारित (भाषा-आधारित नहीं) छँटाई;
गहराई पर बोडो/मराठी/संस्कृत घुसे हुए हैं।
> Bodo row at offset 200,000: `(सोदोब) नि थाखाय थार बुंफुरलुखौ सायख ': a) ('माडागास्करनि राजथावनि') ...` — Devanagari script, Bodo language.
> **Deep-offset: severe degradation.**

**C5. `manishiitg/aditi-syn-v2` — 55,450 — Apache-2.0.**
**38 % Latin-only, 36 % Devanagari, 18 % romanized-Hinglish, 9 % mixed; median 1,974.** One of the very few sets that
*deliberately* trains Hindi ↔ Hinglish ↔ English mode-switching via system prompts (`Answer in mix of hindi and
hinglish`, `Answer in hinglish only`). Quality is uneven. | हिन्दी↔हिंग्लिश↔अंग्रेज़ी मोड-स्विचिंग सिखाने वाले गिने-चुने
सेटों में से एक; गुणवत्ता असमान।
> Mistranslated system persona, seen repeatedly: `आप एक सहायक सहायक हैं.` — "You are a helpful assistant" rendered as "You are an **assistant assistant**".
> Degenerate creative task, row ≈ 2,000: `टीमेटिक ट्रांसिटियन, ट्रांसपारेंट टाइम, / ट्रांसमिशन ट्रांसमाइट, ट्रांसलेट टाइटल, ...` — a "every word starts with T" sonnet, produced as transliterated English nonsense.

**C6. `BhabhaAI/Cross-Hindi-Hinglish-chat` (19,254) · `NebulaByte/alpaca-gpt4-hindi-hinglish` (49,969) · `ai4bharat/indic-align` `hin_Latn` columns.**
All three label a romanized column *Hinglish*, but it is **mechanical character-level transliteration of formal Hindi
(IndicXlit or similar)**, not the code-mixing real speakers produce. Do not train romanized Hindi on these. | तीनों का
"हिंग्लिश" वस्तुतः औपचारिक हिन्दी का यांत्रिक लिप्यंतरण है, वास्तविक कोड-मिक्सिंग नहीं।
> `emily dickinson kii style main ganit kii sundarataa ke bare main ek kavita likhen.` — a real speaker writes *"ke baare mein"*, not *"ke bare main"*; `main` is a transliteration of `में` that collides with English "main".
> `Kuchh ek risaikling abhiyan ke lie ek nara sujhav den.` (NebulaByte) — `ke lie` for `के लिये`, `risaikal` for "recycle".
> IndicAlign `hin_Latn` is the worst of the three: `prithvi parr paani vibhinn sroton seey aataa hainn jinamen shamil hainh /  / **thair.** saur nihaarikaah ...` — `parr`/`seey`/`hainn`/`hainh`/`merrie`/`curr rhaa huun`, and the list marker **"1."** became `thair.` (a phonetic mangling of *`थीं`*/`1.`).
> Cross-Hindi-Hinglish is additionally **62 % Latin-only** because most *answers* are plain English (it is really a cross-lingual Hinglish→English set).

**C7. `guneetsk99/hindi_instruction_set_187K` — 187,525 — CC-BY-NC-**ND**-4.0.**
Rows at offsets 0 and 2,000 are **byte-identical to `MBZUAI/Bactrian-X` config `hi`** (same `dolly-10834`, `dolly-10835`,
`dolly-10505` ids and outputs) — it is Bactrian-X-hi repackaged, plus bare MT bitext that dominates the tail (offset
100,000 is `Please translate the given English sentence to Hindi`). **CC-BY-NC-ND forbids derivative works, i.e. forbids
remixing it into a training set at all.** | यह Bactrian-X-hi की पुनर्पैकेजिंग है; ND लाइसेंस व्युत्पन्न कार्य ही निषिद्ध करता है। **Deep-offset: degrades to bare bitext.**

**C8. `fhai50032/Hindi-Instruct-HQ` — 27,999 — licence unspecified.**
**59 % mixed, 40 % Devanagari, median 3,100 chars, 0.5 % near-dup.** English persona + English user turn, **Hindi
assistant answer** — a genuine cross-lingual format. But the `modelId` column reads `mistral-large-2402`, i.e. it is
**derived from a proprietary Mistral API model**, which carries output-use restrictions. | अंग्रेज़ी प्रॉम्प्ट + हिन्दी उत्तर;
पर `mistral-large-2402` (प्रोप्रायटरी API) से व्युत्पन्न।

---

### Tier D — do not train on / श्रेणी द — प्रशिक्षण हेतु अनुपयुक्त

**D1. `GenVRadmin/Samvaad-Mixed-Language-3` — 25,920 — MIT label, but `backend = openai|gpt-3.5-turbo-0125`.**
**67 % Latin-only, only 5 % Devanagari-dominant.** The retained `prompt` column exposes the generation instruction
verbatim: *"Try to answer in **positive and respectful manner towards India**."* — **deliberate bias injection** — and the
resulting Hindi Q&A is frequently **factually wrong**. The MIT label also cannot cure the OpenAI output-use terms. |
जनरेशन-प्रॉम्प्ट में जान-बूझकर पक्षपात डाला गया; उत्पन्न तथ्य प्रायः ग़लत; MIT लेबल OpenAI की शर्तों को रद्द नहीं करता।
> `Q: कश्मीर की किस रानी को 'नेली' कहा जाता है? A: रानी नेली` (circular non-answer); `Q: कश्मीर के शिकारे किसे कहा जाता है? A: हांगुल` (a *shikara* is a boat; *hangul* is a deer).

**D2. `equal-ai/conversational_hindi` — NOT an instruction dataset.** Despite the name and 6.3 GB of "conversational
Hindi", the schema is `['audio','sentence','segment_id','episode_id','channel','speaker_id','gender','start_time',
'end_time','duration','language']` — **47,107 rows of ASR audio segments**. Listed so nobody else wastes the download. |
नाम भ्रामक है; यह ASR ऑडियो डेटासेट है, इंस्ट्रक्शन डेटा नहीं।

**D3. `pfin123/hindi-aggregated` — 745,066 — Apache-2.0 — NOT an instruction dataset.** Schema `['text','timestamp','url']`;
the rows are **raw Hindi news web pages** (Oneindia, Patrika, Amar Ujala) with navigation boilerplate and embedded JSON.
It is a *pretraining* corpus (and belongs in the companion doc), not SFT data. | यह कच्चा हिन्दी समाचार-वेब टेक्स्ट है —
प्री-ट्रेनिंग कॉर्पस, SFT डेटा नहीं।

**D4. `OdiaGenAI/instruction_set_hindi_1035` — ~1,035 rows, **25.9 % near-duplicate**, median 160 chars.** Too small and too repetitive to matter.

---

## 🎯 Recommendation — what to actually train on / अनुशंसा — वास्तव में किस पर प्रशिक्षण करें

**Short version / संक्षेप में:** for a Qwen-class model that has already been continued-pretrained on Hindi, **do not
build the SFT stage out of translated Hindi data by volume.** NVIDIA's Nemotron-Mini-Hindi ablation (arXiv:2410.14815)
reports in text that adding back-translation-filtered translated Hindi SFT data to an English SFT set produced *"no
improvements with this addition"*, and their Table 5 bears this out at the margin: on the Hindi-pretrained base,
**English SFT + English DPO = 3.81** SubjectiveEval, **+ translated Hindi SFT = 4.28**, but **+ synthetic Hindi DPO
alone = 4.30 (their best cell)** and **doing both = 4.25** — i.e. Hindi *preference* data buys the same gain as Hindi
SFT data, and the two do not stack. Google's
"Just a Pinch of Multilinguality" (ACL Findings 2024) independently found that replacing **only ~1 % (40 examples)** of an
English tuning set with multilingual examples substantially improves multilingual instruction-following. The lever that
matters for Hindi is therefore **(i) a small, genuinely native Hindi core, (ii) Hindi-language *preference* data, and
(iii) aggressive translation-quality filtering — not more translated SFT rows.** / एक ऐसे Qwen-श्रेणी मॉडल के लिए जिस पर
पहले ही हिन्दी में निरंतर-प्री-ट्रेनिंग हो चुकी है, **SFT चरण को अनूदित हिन्दी डेटा की मात्रा पर मत बनाइए।** NVIDIA के
Nemotron-Mini-Hindi के परीक्षण में हिन्दी-प्री-ट्रेन्ड बेस पर अनूदित हिन्दी SFT जोड़ने से विश्वसनीय लाभ नहीं हुआ, जबकि
DPO चरण में हिन्दी अभिरुचि-डेटा जोड़ने से सर्वाधिक लाभ हुआ। निर्णायक कारक हैं: **छोटा पर वास्तविक मूल हिन्दी कोर,
हिन्दी अभिरुचि-डेटा, और कठोर अनुवाद-गुणवत्ता छँटाई — न कि और अधिक अनूदित पंक्तियाँ।**

**Concrete proposed SFT mixture (≈ 400 k examples, Hindi ≈ 40 %) / प्रस्तावित ठोस SFT मिश्रण (≈ 4 लाख उदाहरण, हिन्दी ≈ 40 %):**

| Bucket / श्रेणी | Source / स्रोत | Examples / उदाहरण | Why | कारण |
|---|---|---|---|---|
| **Hindi native core** (upsample ×3) | `indic-instruct-data-v0.1 wikihow/hi` (6,055, prompts rebuilt) + `anudesh/hi` (7,577) + `BhabhaAI/Hi-Instruct-v0` (9,969) + **`aya_dataset` Hindi (1,153, human-written)** | ~25 k → ~75 k | Only truly native / directly-generated Hindi register. | केवल यही वास्तविक हिन्दी शैली सिखाता है। |
| **Hindi grounded chat** | `sarvamai/samvaad-hi-v1` filtered to Hindi-output rows (≈ 45–70 % of 101,476) | ~50 k | India-grounded, Apache-2.0, multi-turn. | भारत-केंद्रित, अपाचे-2.0, बहु-वार्ता। |
| **Hindi open-license bulk** | `ai4bharat/indic-align` `hin_Deva` from Wiki-Conv + Wiki-Chat + WikiHow (sample from 141 k + 198 k + 20 k) | ~60 k | CC-BY-4.0, no proprietary outputs, clean at depth. | CC-BY-4.0, प्रोप्रायटरी-मुक्त, गहराई पर स्वच्छ। |
| **Hindi long-form format** | `apurvagup/ultrachat_hindi_seamless` (185,542) — sample | ~30 k | Teaches long structured answers; 0 % dup. | दीर्घ, सुसंरचित उत्तर सिखाता है। |
| **Hindi math / reasoning** | `BhabhaAI/orca-math-...-hindi-filtered` **or** `Aryabhatta-Orca-Maths-Hindi` (not both) + `bingbangboom/gsm8k-hindi` | ~40 k | Sarvam-M found the base model "lacked basic understanding of whole numbers and arithmetic" *in Hindi*. | सर्वम-M ने पाया कि बेस मॉडल को हिन्दी में बुनियादी अंकगणित तक नहीं आता था। |
| **Romanized / code-mixed Hindi** | `smangrul/hindi_instruct_v1` (20,215, real Hinglish + a `Transliteration and Code Mixing` category) + `manishiitg/aditi-syn-v2` Hinglish rows; then **generate more yourself** from the Hindi core with your own CPT'd model (Sarvam-M recipe) | ~20 k | **Every public romanized Hindi set sampled here is mechanical IndicXlit output.** | सार्वजनिक रोमनीकृत सेट यांत्रिक लिप्यंतरण हैं — स्वयं उत्पन्न करें। |
| **Hindi safety / refusal** | `ai4bharat/indic-align` `Toxic_Matrix` `hin_Deva` (90,352) — sample **and paraphrase the refusal opener** | ~5 k | 100 % of refusals begin `मैं इस संकेत का जवाब नहीं दे सकता क्योंकि यह प्रकृति में संभावित रूप से विषाक्त है।` — training on it verbatim installs a canned refusal. | सभी अस्वीकृतियाँ एक ही वाक्य से शुरू होती हैं; ज्यों-का-त्यों प्रशिक्षण रटा-रटाया इनकार सिखाएगा। |
| **English backbone** | your existing English SFT mixture (Tulu-3-style) | ~200 k | Cross-lingual transfer of instruction-following is strong; Airavata showed **full fine-tuning degraded English while LoRA did not**. | निर्देश-पालन का क्रॉस-लिंग्वल अंतरण प्रबल है; Airavata में पूर्ण फ़ाइन-ट्यूनिंग ने अंग्रेज़ी बिगाड़ी, LoRA ने नहीं। |
| **Cross-lingual glue** | `indic-instruct-data-v0.1 nmt-seed/hi` (50 k En↔Hi bitext) — sample; plus "answer in Hindi" prompts | ~15 k | AI4Bharat added bitext explicitly "to enable better cross-lingual transfer". | AI4Bharat ने क्रॉस-लिंग्वल अंतरण हेतु यही जोड़ा था। |

**Then a preference stage — this is where the Hindi gain actually lands / फिर अभिरुचि चरण — असली हिन्दी लाभ यहीं मिलता है:**
`aaditya/orca_dpo_pairs-Hindi` (10,305 **code-mixed** pairs) + `manishiitg/aditi-dpo-prompts` (48,745 prompts, 44 %
Devanagari / 42 % Latin) + **self-generated on-policy pairs** scored by a Hindi-capable judge. Sarvam AI's key reported
finding is that a **"real-value" reward score** (probability-weighted over the 0–9 score token's log-probs) lifted
reward-model accuracy from **72.85 % → 85.53 %** across 11 Indian languages versus plain generative scoring — cheap to
copy and directly relevant. | सर्वम AI का मुख्य निष्कर्ष: "रियल-वैल्यू" स्कोरिंग से 11 भारतीय भाषाओं में रिवॉर्ड-मॉडल
सटीकता 72.85 % → 85.53 % हो गई — इसे अपनाना सस्ता और सीधे प्रासंगिक है।

**Filtering rules to apply before any of the above / ऊपर कुछ भी उपयोग करने से पहले लागू करने योग्य छँटाई-नियम:**
1. **Do not reuse Airavata's `chrF++ ≥ 50` threshold.** We observed a fully degenerate row scoring **chrF++ = 94.33**.
   The Hindi-benchmark team of arXiv:2508.19831 used **chrF++ ≥ 90** for accepting back-translated data. Use ≥ 85–90,
   *and* add a degeneration check (max repeated-token run) — back-translation similarity alone does not catch loops. |
   `chrF++ ≥ 50` कदापि न दोहराएँ; ≥ 85–90 रखें और पुनरावृत्ति-जाँच जोड़ें।
2. **Devanagari ≠ Hindi.** Run a family-aware LID (Hindi vs Marathi / Nepali / Sanskrit / Bodo / Maithili) — `zicsx/indic-align-hindi`
   is Bodo at depth and `indic-align`'s `Anudesh` config is Marathi at depth. | देवनागरी का अर्थ हिन्दी नहीं; भाषा-परिवार-सचेत LID चलाएँ।
3. **Entity-preservation check.** Reject rows where a source named entity or product name was translated
   (`Windows 11 → खिड़की 11`, `Discourse → प्रवचन`, `corned beef → मकई के गोमांस`). A cheap proxy: flag rows where the
   back-translation loses a capitalised source token. | नामित-इकाई संरक्षण जाँच जोड़ें।
4. **Repair or drop structural markers.** MCQ option letters (`A/B/C/D → उ./B./ग./D.` in `openhermes-2.5-hindi`), list
   numbering (`10. → दस।` in Aya-hi; `1. → 1।` in the 1.5 M sets), and URL-escaped prompts (wikiHow-hi). | संरचनात्मक
   चिह्नों (विकल्प-अक्षर, सूची-क्रमांक, URL-एन्कोडिंग) की मरम्मत करें या उन पंक्तियों को हटाएँ।
5. **Deduplicate across repos, not just within.** `shreyas18/Hindi_instruct_1_5M_v1` ⊃ `atharvanighot/Hindi-Instruct-500K`;
   `guneetsk99/hindi_instruction_set_187K` ⊃ `MBZUAI/Bactrian-X hi`; `Aryabhatta-Orca-Maths-Hindi` ≈ `orca-math-...-hindi-filtered`; `smangrul/hindi_instruct_v1` ≡ `justinj92/hinglish_sharegpt_v0.1`. |
   रेपो-के-बीच भी डुप्लिकेट हटाएँ।
6. **Licence hygiene.** If the output must be commercially usable, the *only* large clean Hindi pool is
   **`ai4bharat/indic-align` (CC-BY-4.0, open teacher models)**; `samvaad-hi-v1` (Apache-2.0) and the Aya sets
   (Apache-2.0) are also safe. Exclude `Bactrian-X` (CC-BY-NC), `M2Lingual` (CC-BY-NC-SA), `hindi_instruction_set_187K`
   (CC-BY-NC-**ND**), `rishiraj/hindichat` (CC-BY-NC), `lm_sys/hi` (LMSYS agreement), and anything GPT-/Mistral-derived. |
   व्यावसायिक उपयोग हेतु केवल `indic-align`, `samvaad-hi-v1` और Aya सुरक्षित हैं।

**Language-ratio guidance / भाषा-अनुपात मार्गदर्शन:** Mantra-14B (arXiv:2504.09753) tuned **Qwen-2.5-14B** and **Phi-4**
on the same 485 k En+Hi mixture and reports that **Qwen did best with a Hindi share above 50 %, while Phi-4 did best
below 50 %** — i.e. the optimum is base-model-specific, so run a small ratio sweep before the full run. Sarvam-M chose
**28 % Hindi within its Indic portion** and split each Indic prompt **50 % native script / 25 % code-mixed / 25 %
romanized**. | Mantra-14B के अनुसार Qwen के लिए 50 % से अधिक हिन्दी सर्वोत्तम रहा, Phi-4 के लिए 50 % से कम — अतः पूर्ण रन
से पहले छोटा अनुपात-स्वीप चलाएँ।

**Evaluation to gate on / मूल्यांकन:** prefer **natively-authored** Hindi benchmarks over translated ones —
`ai4bharat/MILU` (built from 1,500+ Indian competitive exams; GPT-4o tops it at **74 %**), `ai4bharat/IndicIFEval`
(both `indicifeval-ground/hi` = natively grounded and `indicifeval-trans/hi` = translated, so you can measure the gap),
`sarvamai/gsm8k-indic` configs `hi` **and `hi_roman`** (romanized Hindi is where Sarvam-M's biggest gain, **+86 %**,
appeared), and the IFEval-Hi / MT-Bench-Hi suite of arXiv:2508.19831. | अनूदित के बजाय **मूल रूप से रचित** हिन्दी बेंचमार्क
प्राथमिकता दें: MILU, IndicIFEval (grounded बनाम translated दोनों), gsm8k-indic (`hi` तथा `hi_roman`)।

> ⚠️ **One more trap, verified by direct sampling:** `ai4bharat/indic-align` config **`IndoWordNet` (96,843,950 rows on
> the Hub; 74.3 M claimed in the paper)** is **100 % Hindi in the shard we scanned (5 row-groups × 1,000 rows, all
> `language == "hi"`)** and is **extreme template repetition** — the paper states 100 templates were sampled per word, and
> we observed the identical question repeated with only the answer wording paraphrased:
> `'अन्तराभिमुखी' शब्द का सामान्य अर्थ क्या है?` → *"'अन्तराभिमुखी' शब्द का सामान्य अर्थ हैः …"* / *"जब हम 'अन्तराभिमुखी' कहते हैं, तो यह आमतौर पर संदर्भित करता हैः …"* / *"अगर हम 'अन्तराभिमुखी' के बारे में बात करते हैं, तो इसका आम तौर पर मतलब हैः …"* / *"'अन्तराभिमुखी' की मानक परिभाषा हैः …"* — four rows, one fact. **Exclude IndoWordNet from any SFT mixture** (it dwarfs everything else and would teach nothing but a lexicon-lookup template). | ⚠️ `IndoWordNet` कॉन्फ़िग (9.68 करोड़+ पंक्तियाँ) पूर्णतः हिन्दी है पर अत्यधिक टेम्पलेट-पुनरावृत्ति है — एक ही प्रश्न के 100 पुनर्वाक्यांशित उत्तर। **किसी भी SFT मिश्रण से इसे बाहर रखें।**

---

## Taxonomy / वर्गीकरण

- **(A) Flagship open Hindi/Indic SFT collections / प्रमुख खुले हिन्दी-भारतीय SFT संग्रह:** IndicInstruct (Airavata),
  IndicAlign (IndicLLMSuite), Samvaad-Hi-v1, Aya Dataset / Aya Collection (Hindi split).
- **(B) Translated English-SFT derivatives / अनूदित अंग्रेज़ी-SFT व्युत्पन्न:** Bactrian-X-hi, alpaca-cleaned-hindi,
  alpaca-hindi-cleaned, alpaca-gpt4-hindi, evol-instruct-hindi, sharegpt-hindi, openhermes-2.5-hindi,
  ultrachat_hindi_seamless, Hindi_instruct_1_5M / Hindi-Instruct-500K, hindi_instruction_set_187K,
  indic-instruct-data-v0.2-filtered.
- **(C) Directly-generated / distilled Hindi / सीधे उत्पन्न या आसवित हिन्दी:** BhabhaAI Hi-Instruct-v0, aditi-syn-v1/v2,
  Samvaad-Mixed-Language-*, Hindi-Instruct-HQ, indicsft.
- **(D) Romanized Hindi & Hinglish / रोमनीकृत हिन्दी व हिंग्लिश:** CMU Hinglish DoG, Hinglish-TOP, Hinglish-Everyday-
  Conversations-1M, Cross-Hindi-Hinglish-chat, alpaca-gpt4-hindi-hinglish, `hin_Latn` columns of IndicAlign, RomanSetu
  (method), Sarvam-M's three-script scheme (method).
- **(E) Reasoning / math / code in Hindi / हिन्दी में तर्क-गणित-कोड:** orca-math-...-hindi-filtered,
  Aryabhatta-Orca-Maths-Hindi, gsm8k-hindi, sarvamai/gsm8k-indic (eval).
- **(F) Massively-multilingual mixtures with a Hindi share / हिन्दी अंश वाले बहुभाषी मिश्रण:** Aya Collection/Dataset,
  Bactrian-X, M2Lingual, xP3x, Tulu-3-SFT (via its Aya subset).
- **(G) Preference / RLHF data / अभिरुचि-डेटा:** orca_dpo_pairs-Hindi, aditi-dpo-prompts, IndicAlign-Toxic
  (HH-RLHF-T + Toxic-Matrix), PARIKSHA human preferences, Cohere's ML-23-230K (unreleased).
- **(H) Closed industrial post-training recipes (methodology only, no data release) / बंद औद्योगिक पोस्ट-ट्रेनिंग विधियाँ:**
  Sarvam-M, Sarvam-1/OpenHathi, Krutrim-1/2, Llama-3-Nanda-10B-Chat, Nemotron-Mini-Hindi-4B, PARAM-1, Mantra-14B.
- **(I) Methodology & critique of translated SFT / अनूदित SFT की पद्धति व आलोचना:** Aya post-editing study,
  "Just a Pinch of Multilinguality", Nemotron-Mini-Hindi ablation, Benchmarking Hindi LLMs, MILU, RomanSetu.
- **(J) Native Hindi evaluation resources needed to gate the above / मूल्यांकन संसाधन:** MILU, IndicIFEval,
  IFEval-Hi / MT-Bench-Hi / GSM8K-Hi / ChatRAG-Hi / BFCL-Hi, gsm8k-indic, Airavata human-eval prompts.

---

## 1. Flagship Open Hindi / Indic SFT Collections / एक. प्रमुख खुले हिन्दी–भारतीय SFT संग्रह

### IndicInstruct — Airavata: Introducing Hindi Instruction-tuned LLM (Gala et al., 2024)
- **Venue / Link:** arXiv preprint (AI4Bharat), Jan 2024 — https://arxiv.org/abs/2401.15006 · code https://github.com/AI4Bharat/IndicInstruct
- **Data / Link:** https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1
- **Motivation / अभिप्रेरणा:** open multilingual LLMs underperform on Hindi and there was no open, reusable Hindi instruction-tuning mixture; the authors release both a Hindi assistant (Airavata) and the underlying data. | खुले बहुभाषी मॉडल हिन्दी में कमज़ोर थे और कोई पुनःप्रयोज्य हिन्दी इंस्ट्रक्शन-मिश्रण उपलब्ध नहीं था; लेखक मॉडल और डेटा दोनों जारी करते हैं।
- **Size (verified from Parquet footers):** **404,287 rows total.** Per config/split — `flan_v2` en 67,463 / hi 67,463; `lm_sys` 50,000 / 50,000; `nmt-seed` hi 50,000; `oasst1` en 19,945 / hi 20,128; `dolly` 15,011 / 15,011; `wikihow` en 20,400 / hi 6,055; `anudesh` en 5,234 / hi 7,577; `hh-rlhf` 5,000 / 5,000. The **paper's Table 1** (post-filter) gives Hindi counts of FLAN-v2 65,228 · HHH 4,911 · Dolly 14,880 · OpenAssistant 16,384 · LMSys 37,422 · WikiHow 6,055 · Anudesh 7,577 → **English 183,053 + Hindi 152,457 = 335,510**, and the paper's stated "**385 k** examples used for instruction tuning" matches that total once the 50 k NMT bitext is added (335,510 + 50,000 = 385,510) — this reconciliation is our inference, not stated in the paper.
- **Construction / निर्माण:** English sets translated with **IndicTrans2**; Hindi kept only when the **back-translation chrF++ ≥ 50**. Two *native* Hindi sets were created: **wikiHow** (scraped `hi.wikihow.com`, human-moderated) and **Anudesh** (crowd-written prompts + **Llama-2-70B** responses, translated). The authors explicitly rejected ChatGPT for translation: *"this is not cost-effective, and the translation quality of ChatGPT is lower than IndicTrans2, and its generation quality in Hindi might not be up to the mark."* | अंग्रेज़ी सेट IndicTrans2 से अनूदित; back-translation chrF++ ≥ 50 पर ही रखे गए। दो *मूल* हिन्दी सेट: wikiHow और Anudesh।
- **Script / dialect:** Devanagari only (no romanized column). | केवल देवनागरी।
- **Licence (per subset, from the paper):** FLAN-v2 Apache-2.0 · Anthropic-HHH MIT · Dolly CC-BY-SA-3.0 · OpenAssistant Apache-2.0 · **LMSYS-Chat-1M under the LMSYS-Chat-1M Dataset License Agreement** · NMT/BPCC-Human CC-BY-4.0 · **wikiHow CC-0** · **Anudesh CC-BY-4.0**. The repo itself carries **no top-level licence tag** — treat per-subset. | प्रत्येक उप-सेट का अलग लाइसेंस; रेपो पर कोई शीर्ष-स्तरीय लाइसेंस नहीं।
- **Recipe / विधि:** Airavata = **OpenHathi (LLaMA-2 7B + Hindi CPT & vocab expansion by Sarvam AI)** fine-tuned with **LoRA** (rank 16, α 32, dropout 0.05, 4 epochs, lr 5e-4, batch 128, bf16), loss on output tokens only, plus checkpoint averaging (interpolation 0.6 between epoch-3 and epoch-4). **LoRA was chosen over full fine-tuning because full FT degraded English.**
- **Key findings / मुख्य निष्कर्ष:** *"while Airavata still trails significantly behind GPT-4 in terms of its ability to follow instructions and the quality of its content, it performs relatively better when generating **natural-sounding Hindi** content compared to both GPT-4 and ChatGPT."* It also beats BactrianX-LLaMA-7B by a wide margin, which the authors attribute to Bactrian-X being *"trained on a lower-quality dataset for instruction tuning that was completely generated using ChatGPT"* plus its lack of Hindi vocabulary expansion. Weakness: *"the model struggles with tasks demanding creative language usage, as our SFT data lacks components emphasizing creativity."* | Airavata GPT-4 से निर्देश-पालन में पीछे है पर **अधिक स्वाभाविक हिन्दी** लिखता है; रचनात्मक कार्यों में कमज़ोर क्योंकि SFT डेटा में रचनात्मकता का अंश नहीं।
- **Quality notes from our sampling / हमारे नमूनाकरण से गुणवत्ता टिप्पणी:** see Tier A3/A4 and Tier C1/C2 above. The `flan_v2` and `lm_sys` slices carry `quality_metrics` proving chrF++ ≥ 50 is too permissive; the `wikihow/hi` `messages` field is URL-escape-corrupted.

### IndicAlign — IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages (Khan et al., 2024)
- **Venue / Link:** ACL 2024 (Outstanding Paper) — https://arxiv.org/abs/2403.06350
- **Data / Link:** https://huggingface.co/datasets/ai4bharat/indic-align — **CC-BY-4.0**
- **Motivation / अभिप्रेरणा:** *"Majority of the high-quality synthetic supervised fine-tuning data released has been created with proprietary models like ChatGPT and GPT-4, which renders them unusable in commercial settings. We therefore consider only license-friendly datasets and models."* This is the design constraint that makes IndicAlign uniquely valuable. | अधिकांश उच्च-गुणवत्ता सिंथेटिक SFT डेटा प्रोप्रायटरी मॉडलों से बना है और व्यावसायिक रूप से अनुपयोगी है; अतः यहाँ केवल लाइसेंस-अनुकूल मॉडल/डेटा प्रयुक्त हुए।
- **Size (Hub row counts, verified):** IndoWordNet **96,843,950** · Wiki-Chat **198,254** · Wiki-Conv **141,435** · Toxic-Matrix **90,352** · Anudesh **36,820** · HHRLHF-T **32,669** · Indic-ShareLlama **21,171** · WikiHow **20,313** · OpenAssistant-T **19,909** · Dolly-T **15,011** → **97,419,884 rows**. The paper's Table 7 reports the same components as **74.7 M** pairs (IndoWordNet 74,272.2 k) — the Hub figure is larger; both numbers are recorded here rather than reconciled. Paper also reports avg. turns (Wiki-Conv 9.14, Wiki-Chat 2.8, OpenAssistant-T 2.98, Anudesh 1.58) and MTLD lexical diversity (Toxic-Matrix 86.57, HH-RLHF-T 79.00, Wiki-Chat 56.67, WikiHow 23.87).
- **Construction / निर्माण:** prompt/response provenance per component (H = human, M = model): Indic-ShareLlama H/M (ShareGPT first turns → **Llama-2-70B-Chat**, non-English/code/math prompts excluded); Dolly-T, OpenAssistant-T, WikiHow H/H (translated + transliterated); IndoWordNet H/H (**original**, templated from IndoWordNet via `pyiwn`, ~100 templates per word); Anudesh H/M (crowd prompts, Llama-2-70B-Chat); Wiki-Conv M/M (Llama-2-70B-Chat over India-centric Wikipedia passages + infoboxes); Wiki-Chat M/M (four-agent simulation — Intent LLM, Init-User LLM, Assistant LLM, Next-User LLM — using **Llama-2-70B-Chat and Mixtral-8x7B-v0.1**, grounded in Wikipedia); HH-RLHF-T H/M; Toxic-Matrix M/M (Mistral-7B-Instruct + Llama-2). Translation/transliteration via the **Setu-Translate (IndicTrans2)** and **Setu-Transliterate (IndicXlit)** pipelines into 14 Indian languages. | घटक-वार मानव/मॉडल स्रोत; अनुवाद व लिप्यंतरण Setu पाइपलाइनों द्वारा 14 भारतीय भाषाओं में।
- **Script / dialect:** **n-way parallel columns** `eng_Latn, hin_Deva, hin_Latn, asm_Beng, ben_Beng, guj_Gujr, kan_Knda, mal_Mlym, mar_Deva, npi_Deva, ory_Orya, pan_Guru, san_Deva, tam_Taml, tel_Telu, urd_Arab` (+ `*_Latn` romanizations) for most configs; `Anudesh` and `IndoWordNet` instead carry raw/`language`-tagged rows. | अधिकांश कॉन्फ़िग में भाषा-वार समानांतर स्तंभ; `Anudesh`/`IndoWordNet` अलग संरचना।
- **Quality notes / गुणवत्ता टिप्पणी:** `hin_Deva` is the best-behaved MT in this survey (Tier A1); `hin_Latn` is unusable (Tier C6); `IndoWordNet` is ~9.7 M near-duplicate template rows per shard and must be excluded; `Anudesh` mixes 20 languages with no per-row language column. | `hin_Deva` उत्कृष्ट, `hin_Latn` अनुपयोगी, `IndoWordNet` अत्यधिक दोहरावपूर्ण, `Anudesh` भाषा-मिश्रित।

### Samvaad-Hi-v1 (Sarvam AI, 2024)
- **Data / Link:** https://huggingface.co/datasets/sarvamai/samvaad-hi-v1 — **Apache-2.0**, 69 likes, 12+ downstream models
- **Motivation / अभिप्रेरणा:** card describes *"100k high-quality conversations in English, Hindi, and Hinglish curated exclusively with an Indic context"* — i.e. the goal is India-grounded chat rather than translated generic assistant data. | उद्देश्य: अनूदित सामान्य सहायक-डेटा के बजाय भारत-केंद्रित संवाद।
- **Size:** **101,476 conversations** (single `train` split, `messages` column, 203 MB).
- **Construction / निर्माण:** **not documented in the card** beyond "curated"; our sampling shows model-generated multi-turn answers over Indic reference material (Wikipedia-style biographies, stations, films, festivals) — i.e. **synthetic-over-Indic-source**, not human-written and not a straight translation of an English set. Teacher model **UNVERIFIED**. | कार्ड में निर्माण-विधि अनुल्लिखित; नमूनाकरण से भारतीय स्रोत-सामग्री पर मॉडल-जनित उत्तर प्रतीत होते हैं। शिक्षक मॉडल अज्ञात।
- **Script / dialect:** Devanagari + English + a small romanized-Hinglish slice (sampled mix 45 / 24 / 6 %, plus 25 % mixed).
- **Quality notes / गुणवत्ता टिप्पणी:** see Tier A2 — best openly-licensed India-grounded chat set; light-to-moderate translationese; filter for Hindi-output rows.

### The Aya Dataset & Aya Collection (Singh et al., 2024)
- **Venue / Link:** ACL 2024 — https://arxiv.org/abs/2402.06619
- **Data / Links:** https://huggingface.co/datasets/CohereLabs/aya_dataset · https://huggingface.co/datasets/CohereLabs/aya_collection_language_split — both **Apache-2.0**
- **Motivation / अभिप्रेरणा:** build a genuinely human-annotated multilingual instruction resource rather than another machine-translated one, via a participatory annotation platform. | मशीन-अनूदित के बजाय वास्तव में मानव-लिखित बहुभाषी निर्देश-संसाधन बनाना।
- **Size — Hindi specifically (verified by a full scan of the `language` column):** **Aya Dataset train = 202,362 rows total, of which Hindi = 1,153.** For comparison, Tamil has 14,133, Telugu 8,439, Punjabi 6,385, Nepali 4,002, Gujarati 3,989, Marathi 3,545. **Hindi is one of the *worst*-represented Indian languages in Aya's human-written portion.** Aya Collection `hindi` split = **3,772,864 train / 283,272 validation / 325,548 test** (templated + translated, not human-written). Paper totals: 204,114 human annotations in 65 languages; 513 M instances in 114 languages. | **Aya के मानव-लिखित हिस्से में हिन्दी केवल 1,153 पंक्तियाँ है** — तमिल (14,133) से बहुत कम। Aya Collection का हिन्दी हिस्सा 37.7 लाख पंक्तियाँ है, पर वह टेम्पलेट/अनूदित है।
- **Construction / निर्माण:** Aya Dataset = original prompt-completion pairs written by fluent speakers on the Aya Annotation Platform. Aya Collection = existing NLP datasets re-templated by fluent speakers + machine translation (NLLB) of English data.
- **The load-bearing MT-quality datapoint / निर्णायक अनुवाद-गुणवत्ता आँकड़ा:** the authors professionally post-edited the machine-translated Dolly test prompts in six languages and report (Table 7) that for **Hindi, 60.0 % of prompts required editing** (HTER 6.16, HChrF 95.00) versus 41.0 % for Arabic and 86.5 % for Russian — *"We find that editors edited at least 41 % of prompts in all languages, a surprisingly high number. This indicates that translation errors in the dolly-machine-translated test set are quite common."* Their worked example is exactly the fish question that survives corrupted in both AI4Bharat releases. | लेखकों ने पाया कि **हिन्दी में 60 % मशीन-अनूदित प्रॉम्प्ट को मानव-संपादन की आवश्यकता पड़ी** — अनुवाद-त्रुटियाँ बेहद सामान्य हैं।
- **Quality notes / गुणवत्ता टिप्पणी:** see Tier B10 — the Collection's Hindi split is FLAN-templated NLLB output with `<unk>` corruption, half-translated sentences and numeral-word confusion.

---

## 2. Translated English-SFT Derivatives / दो. अनूदित अंग्रेज़ी-SFT व्युत्पन्न

### Bactrian-X (Li et al., 2023)
- **Venue / Link:** arXiv:2305.15011 — https://arxiv.org/abs/2305.15011 · https://huggingface.co/datasets/MBZUAI/Bactrian-X
- **Motivation / अभिप्रेरणा:** create a parallel 52-language instruction set cheaply, by translating prompts and generating fresh responses in-language rather than translating responses. | प्रॉम्प्ट अनूदित कर उत्तर उसी भाषा में उत्पन्न करके सस्ते में 52-भाषी समानांतर सेट बनाना।
- **Size:** **67,017 per language × 52 languages = 3,484,884 rows** (verified); **Hindi = 67,017**.
- **Construction / निर्माण:** Alpaca (52 k, itself GPT-3.5 self-instruct) + Dolly (15 k, human) prompts translated by the **Google Translate API** (~US$10,000), programming-related prompts excluded by keyword matching; responses generated by **`gpt-3.5-turbo`** (~US$3,000), 16–21 April 2023. The authors note they deliberately did *not* translate responses because *"potential issues such as 'translationese' and non-native answer styles may arise from relying solely on translated responses."* Quality check: 100 back-translated instances per language scored BLEU / chrF++ / COMET, worst BLEU 28.0 (Mongolian), most > 40.
- **Licence / लाइसेंस:** **CC-BY-NC-4.0** *and* OpenAI-output-derived → **research use only**. | केवल शोध हेतु।
- **Quality notes / गुणवत्ता टिप्पणी:** Tier B6. Also note Airavata's human evaluation attributes Bactrian-X-LLaMA's poor Hindi partly to *"a lower-quality dataset for instruction tuning that was completely generated using ChatGPT."*

### The Alpaca-Hindi family
`iamshnoo/alpaca-cleaned-hindi` (**51,760**), `saillab/alpaca-hindi-cleaned` (**41,601** train + 10,401 test), `FreedomIntelligence/alpaca-gpt4-hindi` (**49,969**), `NebulaByte/alpaca-gpt4-hindi-hinglish` (**49,969**, adds `input_hinglish`/`output_hinglish` transliteration columns), `smangrul/hindi_instruct_v1` (**20,215**, MIT).
- **Motivation / अभिप्रेरणा:** the cheapest possible Hindi SFT bootstrap — translate the 52 k Alpaca instruction set. | न्यूनतम लागत पर हिन्दी SFT शुरू करना — 52 हज़ार Alpaca निर्देशों का अनुवाद।
- **Construction:** MT of Alpaca / Alpaca-GPT4; MT system **UNVERIFIED** per repo (no cards). Licences mostly **unspecified**; the GPT-4 lineage carries OpenAI output-use terms.
- **Quality notes / गुणवत्ता टिप्पणी:** Tier B8 / C6 — clean Devanagari but shallow, with self-duplicating answers and mechanical "Hinglish".

### FreedomIntelligence multilingual SFT ports
`evol-instruct-hindi` (**59,022**), `alpaca-gpt4-hindi` (**49,969**), `sharegpt-hindi` (**3,142**, Apache-2.0).
- **Motivation:** port the standard English SFT trio (Evol-Instruct / Alpaca-GPT4 / ShareGPT) into many languages for the Phoenix / MultilingualSIFT line of work. | मानक अंग्रेज़ी SFT त्रयी को अनेक भाषाओं में ले जाना।
- **Quality notes:** Tier B7 — evol-instruct-hindi leaks English answers and translates code identifiers into Devanagari; sharegpt-hindi is only 3 k rows with 7.8 % near-dup.

### BhabhaAI translated bulk
`openhermes-2.5-hindi` (**620,211**), `orca-math-word-problems-200k-hindi-filtered` (**188,943**, MIT), `indic-instruct-data-v0.1-filtered` / `-v0.2-filtered` (a re-filtered IndicInstruct — e.g. `lm_sys/hi` drops **50,000 → 24,505**, but our profile of that slice shows **9.6 % near-dup**, worse than the original), `Cross-Hindi-Hinglish-chat` (**19,254**), `hindi-RAG-20k`, `news-summary`, `alpaca-gpt4-hindi-trans`, `Hi-Instruct-v0` (**9,969**).
- **Motivation / अभिप्रेरणा:** the GenVR/BhabhaAI "Gaja" model line needed Hindi-scale SFT data, so they translated the strongest English mixtures (OpenHermes-2.5, Orca-Math) wholesale and re-filtered AI4Bharat's data. | Gaja मॉडल-शृंखला हेतु सबसे मज़बूत अंग्रेज़ी मिश्रणों (OpenHermes-2.5, Orca-Math) का थोक अनुवाद।
- **Licences:** mostly **unspecified**; upstreams are GPT-4-derived (OpenHermes-2.5, Orca-Math).
- **Quality notes:** Tier B1 / B4 / C6 — best bulk fluency available, but the MCQ option-letter corruption in `openhermes-2.5-hindi` must be repaired.

### Undocumented large repackagings
`shreyas18/Hindi_instruct_1_5M_v1` (**1,488,730**) ⊃ `atharvanighot/Hindi-Instruct-500K` (**508,609**);
`guneetsk99/hindi_instruction_set_187K` (**187,525**, CC-BY-NC-**ND**-4.0) ⊃ `MBZUAI/Bactrian-X hi`;
`apurvagup/ultrachat_hindi_seamless` (**185,542** train_sft — UltraChat via SeamlessM4T);
`pranjalchitale/indicsft` (**25,030,167**, 43 GB, no card).
- **Quality notes:** Tier B2 / B3 / C3 / C7. These are the only route to million-scale Hindi SFT, but all lack cards and licences; deduplicate across repos before use. | करोड़-पैमाने पर हिन्दी SFT का एकमात्र रास्ता, पर सभी बिना कार्ड/लाइसेंस के; रेपो-के-बीच डुप्लिकेट हटाना अनिवार्य।

---

## 3. Directly-Generated / Distilled Hindi / तीन. सीधे उत्पन्न या आसवित हिन्दी

### BhabhaAI/Hi-Instruct-v0
**9,969** rows, schema `seed / question / answer`, 100 % Devanagari. Seed-word conditioning produces genuinely open-ended
Hindi prompts. Teacher model undocumented. Tier A5. | बीज-शब्द से उत्पन्न वास्तव में खुले-अंत वाले हिन्दी प्रश्न; शिक्षक मॉडल अज्ञात।

### manishiitg "Aditi" line
`aditi-syn-v1`, `aditi-syn-v2` (**55,450**, Apache-2.0), `aditi-dpo-prompts` (**48,745**), `chat-instruct-hi-v3/v4`,
`indic-synthetic-roleplay`, `indic-synthetic-rag-complex`, `indic-agent`, plus the `open-aditi-hi-v1…v4` models.
- **Motivation / अभिप्रेरणा:** an individual-scale effort to give Hindi models explicit **script-mode control** (Devanagari / Hinglish / English) through system prompts — a capability no other public Hindi set trains. | हिन्दी मॉडलों को सिस्टम-प्रॉम्प्ट द्वारा **लिपि-मोड नियंत्रण** सिखाने का प्रयास — यह क्षमता अन्य किसी सार्वजनिक हिन्दी सेट में नहीं।
- **Quality notes:** Tier C5 — valuable idea, uneven execution (`आप एक सहायक सहायक हैं.`, degenerate alliteration tasks).

### GenVRadmin Samvaad-Mixed-Language-1/2/3 · Samvaad-Indic-Positive · Samvaad-Tamil-Mixtral · Aryabhatta-Orca-Maths-Hindi
`Samvaad-Mixed-Language-3` = **25,920** (MIT label, `backend = openai|gpt-3.5-turbo-0125`); `Aryabhatta-Orca-Maths-Hindi` = **200,000** (MIT).
- **Quality notes:** the maths set is Tier B5 and fine; the Samvaad-Mixed-Language sets are **Tier D** because the retained generation prompt shows explicit bias injection and the generated facts are wrong. | गणित-सेट ठीक है; Samvaad-Mixed-Language सेट पक्षपात-अंतःक्षेपण और तथ्यात्मक त्रुटियों के कारण अनुपयुक्त।

### fhai50032/Hindi-Instruct-HQ
**27,999** rows; English persona + English user turn + **Hindi assistant answer**; `modelId = mistral-large-2402`
(proprietary API → output-use restrictions). Tier C8. | अंग्रेज़ी प्रॉम्प्ट + हिन्दी उत्तर; प्रोप्रायटरी Mistral API से व्युत्पन्न।

---

## 4. Romanized Hindi, Hinglish & Code-Mixing / चार. रोमनीकृत हिन्दी, हिंग्लिश व कोड-मिक्सिंग

### Why this matters / यह क्यों महत्वपूर्ण है
Sarvam-M's largest single reported gain was **+86 % on a romanized Indian-language GSM-8K benchmark**, and its
`GSM-8K-IN-R` score is **0.82 vs 0.44** for the Mistral-Small base — romanized Hindi is where off-the-shelf models fail
hardest. RomanSetu (ACL 2024) independently shows romanized text has **2×–4× lower token fertility** than Devanagari and
that instruction tuning on romanized data *"matches or outperforms"* native-script tuning. | रोमनीकृत हिन्दी वही क्षेत्र है
जहाँ सामान्य मॉडल सबसे अधिक विफल होते हैं; RomanSetu के अनुसार रोमनीकृत पाठ की टोकन-फ़र्टिलिटी 2–4 गुना कम है।

### RomanSetu: Efficiently Unlocking Multilingual Capabilities of LLMs via Romanization (Husain et al., 2024)
- **Venue / Link:** ACL 2024 / arXiv:2401.14280 — https://arxiv.org/abs/2401.14280
- **Method / विधि:** continue-pretrain LLaMA-2 on ~500 M words per language of web text **romanized with IndicXlit**, then instruction-tune on **FLAN (65 k) + Dolly** translated with **IndicTrans2** and then romanized — **120 k IFT examples per language**. Compares IFT-N (native script) vs IFT-R (romanized).
- **Finding / निष्कर्ष:** romanized text *"not only reduces token fertility by 2x-4x but also matches or outperforms native-script representation"* on NLU and NLG. | रोमनीकरण टोकन-फ़र्टिलिटी 2–4 गुना घटाता है और मूल-लिपि निरूपण के बराबर या बेहतर प्रदर्शन देता है।

### CMU Hinglish DoG (`festvox/cmu_hinglish_dog`)
**8,060 train** / 960 test / 942 validation rows; licence **CC-BY-SA-3.0 + GFDL**. Human code-mixed Hindi-English
document-grounded dialogue. Our profile: **82 % Latin-only, 18 % romanized-Hinglish, median 238 chars, 1.5 % near-dup** —
i.e. genuinely code-mixed but short and small; usable as a *style* reference, not as SFT bulk. | वास्तविक मानव कोड-मिक्स्ड
संवाद, पर छोटा और संक्षिप्त — शैली-संदर्भ के रूप में उपयोगी, थोक SFT के लिए नहीं।

### Hinglish-TOP (`WillHeld/hinglish_top`)
**2,993 train / 1,390 validation / 6,513 test** (verified — the test split is larger than train). Task-oriented **semantic-parsing** pairs (`en_query`, `cs_query`,
`en_parse`, `cs_parse`, `domain`). Profile: **82 % Latin, 18 % Hinglish, 5.5 % near-dup**. Narrow (TOP parses), not
general chat. | कार्य-उन्मुख सिमैंटिक-पार्सिंग; सामान्य संवाद नहीं।

### Abhishekcr448/Hinglish-Everyday-Conversations-1M
**1,001,323** rows, **MIT**, schema `input / output`. Profile: **100 % romanized-Hinglish, median only 168 chars, 0.0 %
near-dup.** The largest romanized Hinglish resource by far, but the turns are extremely short small-talk — good for
*register*, useless for instruction-following. | अब तक का सबसे बड़ा रोमनीकृत हिंग्लिश संसाधन, पर वार्ताएँ अत्यंत छोटी —
शैली के लिए उपयोगी, निर्देश-पालन के लिए नहीं।

### `smangrul/hindi_instruct_v1` (MIT) ≡ `justinj92/hinglish_sharegpt_v0.1` — **20,215** rows each
**Verified by direct comparison: these two repos hold the same `train` data in different serialisations** (both 20,215 train rows, byte-identical first rows; `smangrul` additionally ships a 7,788-row `test` split; `smangrul` uses `category` + `messages`, `justinj92` uses `conversations`). Profile:
**37–38 % Devanagari, 24–28 % Latin, 15–19 % romanized-Hinglish, 19–20 % mixed; median 321–340 chars.** Genuinely
script-mixed, which is rare — and `smangrul`'s `category` column includes an explicit **`Transliteration and Code
Mixing`** class, e.g. `Translate the following from English into Hinglish. / I need to know the duration of time it
will take to get from Atlanta to Philadelphia?` → `Muje pata hone chahiye ki Atlanta se Philadelphia tak pohochne me
kitna time lagega?` — that is *real* Hinglish, unlike the transliteration traps in C6. **Deduplicate: do not load both
repos.** | **सत्यापित: ये दोनों रेपो एक ही डेटा हैं**, केवल क्रमांकन भिन्न है। `smangrul` में स्पष्ट `Transliteration and
Code Mixing` श्रेणी है और उसकी हिंग्लिश *वास्तविक* है। दोनों को एक साथ न लोड करें।

### Mechanical-transliteration traps / यांत्रिक लिप्यंतरण के जाल
`BhabhaAI/Cross-Hindi-Hinglish-chat`, `NebulaByte/alpaca-gpt4-hindi-hinglish`, and every `*_Latn` column of
`ai4bharat/indic-align` are IndicXlit-style character transliterations, **not** code-mixing. See Tier C6 for verbatim
examples. **Do not train romanized Hindi on these.** | ये तीनों वास्तविक कोड-मिक्सिंग नहीं, यांत्रिक लिप्यंतरण हैं।

### Sample-Efficient Language Model for Hinglish Conversational AI (Singh et al., 2025)
arXiv:2504.19070 — a Hinglish-focused conversational LM; listed as the most recent dedicated Hinglish modelling effort.
Data details **UNVERIFIED** here (full text not fetched). | नवीनतम समर्पित हिंग्लिश मॉडलिंग प्रयास; डेटा-विवरण यहाँ असत्यापित।

---

## 5. Reasoning, Math & Code in Hindi / पाँच. हिन्दी में तर्क, गणित व कोड

- **`BhabhaAI/orca-math-word-problems-200k-hindi-filtered`** — **188,943**, MIT, 100 % Devanagari, median 980 chars, 0 % near-dup. Translated Microsoft Orca-Math (GPT-4-generated upstream).
- **`GenVRadmin/Aryabhatta-Orca-Maths-Hindi`** — **200,000**, MIT, 100 % Devanagari, median 846. Same lineage; pick one.
- **`bingbangboom/gsm8k-hindi`** — **7,473** `train_main` + **7,473** `train_socratic` (+ 1,319 each for `test_main` / `test_socratic`), MIT, 100 % Devanagari.
- **`sarvamai/gsm8k-indic`** — *evaluation only*: `hi/test` **and `hi_roman/test`**, 11 languages × native + romanized. This is the single most useful public gauge of romanized-Hindi reasoning.
- **Sarvam-M's RLVR maths curriculum (method, no data release):** *"Our final dataset comprised 40 % English data, 40 % data in Indian languages using native scripts, and 20 % data in Indian languages using romanized script. Of the Indian language content, 28 % was in Hindi."* They also report that forcing a **fixed answer format** beat few-shot prompting *"especially for Indian language prompts."* | सर्वम-M का RLVR गणित-पाठ्यक्रम: 40 % अंग्रेज़ी / 40 % मूल-लिपि / 20 % रोमनीकृत; भारतीय-भाषा अंश में 28 % हिन्दी।
- **Missing / अनुपलब्ध:** there is **no public Hindi code-SFT dataset** of any scale. Bactrian-X explicitly excluded programming prompts; IndicAlign explicitly excluded coding and math prompts from Indic-ShareLlama. Hindi coding ability has to come from English transfer or from data you generate. | **किसी भी पैमाने का सार्वजनिक हिन्दी कोड-SFT डेटासेट मौजूद नहीं है।**

---

## 6. Massively-Multilingual Mixtures with a Hindi Share / छह. हिन्दी अंश वाले बहुभाषी मिश्रण

### M2Lingual (Maheshwary et al., ServiceNow, 2024)
- **Link:** https://huggingface.co/datasets/ServiceNow-AI/M2Lingual — **CC-BY-NC-SA-4.0**, arXiv:2406.16783
- **Size — verified by a full scan of the `language` column:** `full_data/train` = **173,672 rows total**, of which
  **Hindi = 2,788** (the language distribution is near-uniform: Moroccan Arabic 2,804, Serbian 2,789, **Hindi 2,788**,
  Urdu 2,771, English 2,764 …). Also ships `seed_data` and `seed_evol_data` configs. | पूर्ण स्कैन से सत्यापित: कुल
  173,672 पंक्तियाँ, जिनमें **हिन्दी = 2,788**।
- **Paper's own framing / पेपर का दावा:** *"the first fully synthetic, multi-turn multilingual dataset having **175K conversations across 70 languages**"*, covering **17+ NLP tasks**, built with a *"novel two-step Evol prompt taxonomy"*; the authors position it explicitly against machine translation (*"existing approaches still rely on machine translation to improve multilingual performance"*). Our footer count for `full_data/train` is **173,672**.
- **Construction / निर्माण:** multilingual **Evol-Instruct** — seed prompts are evolved along typed axes (`task_evol_type`
  e.g. `DIALECT`, `multiturn_evol_type` e.g. `RECALL_INFORMATION`) and the evolution prompts are retained in the data,
  which makes it unusually auditable. Teacher model **UNVERIFIED** from our sampling. | बहुभाषी Evol-Instruct; विकास-प्रॉम्प्ट
  डेटा में संरक्षित हैं, जो इसे असामान्य रूप से अंकेक्षणीय बनाता है।
- **Verdict / निर्णय:** **too small a Hindi share (2,788) to matter as bulk, and non-commercial.** Useful as a source of
  *multi-turn evolution templates* to run on your own Hindi prompts. | हिन्दी अंश बहुत छोटा और ग़ैर-व्यावसायिक; इसका मूल्य
  बहु-वार्ता विकास-टेम्पलेट के रूप में है।

### Others
- **`allenai/tulu-3-sft-mixture`** (ODC-BY) lists `hin` among its language tags (it absorbs an Aya subset), but Hindi is a negligible fraction; treat it as the English backbone, not a Hindi source. | Tulu-3 में हिन्दी नगण्य है; इसे अंग्रेज़ी आधार मानें।
- **`nvidia/Nemotron-Post-Training-Dataset-v2`** (CC-BY-4.0, gated) covers **en, de, it, fr, es, ja — no Hindi.** | इसमें हिन्दी नहीं है।
- **Magpie:** we found **no Hindi Magpie dataset** on the `Magpie-Align` org; `Magpie-Qwen2.5-Pro-1M-v0.1` and siblings are English/Chinese. Magpie-style self-synthesis remains an *unused* option for Hindi — and, given how bad the translated pool is, an attractive one. | Magpie का कोई हिन्दी संस्करण नहीं मिला; Magpie-शैली स्व-संश्लेषण हिन्दी के लिए अप्रयुक्त और आकर्षक विकल्प है।
- **`CohereLabs/xP3x`**, `Global-MMLU`, `include-base-44`, `kaleidoscope` — evaluation/templating resources with Hindi coverage, not SFT bulk.

---

## 7. Preference / DPO / RLHF Data with Hindi / सात. हिन्दी अभिरुचि, DPO व RLHF डेटा

This is the **thinnest** part of the ecosystem — and, per the Nemotron ablation, the part with the highest marginal
value. | यह पारिस्थितिकी का **सबसे पतला** हिस्सा है — और Nemotron के परीक्षण के अनुसार सबसे अधिक सीमांत मूल्य वाला।

- **`aaditya/orca_dpo_pairs-Hindi` — 10,305 pairs.** Schema `codemix_{system,question,chosen,rejected}` + parallel
  `en_*` columns. Profile: **84 % romanized-Hinglish, 14 % Latin, median 4,698 chars, 3.2 % near-dup.** This is the only
  public **code-mixed Hindi preference set** we found, and the Hinglish reads naturally
  (`Iss movie plot ke liye ek movie title suggest kijiye: …`; system prompt `Aap ek AI assistant hain. Aapko ek task
  diya jayega. Aapko ek detailed aur long answer generate karna hai.`). Derived from Orca DPO pairs (GPT-4 lineage);
  licence unspecified. | एकमात्र सार्वजनिक **कोड-मिक्स्ड हिन्दी अभिरुचि-सेट**; हिंग्लिश स्वाभाविक लगती है। लाइसेंस अनुल्लिखित।
- **`manishiitg/aditi-dpo-prompts` — 48,745.** Profile: **44 % Devanagari, 42 % Latin, 14 % mixed; median 2,976; 0 % near-dup.** Prompt bank for Hindi/Hinglish preference generation.
- **`ai4bharat/indic-align` → `HHRLHF_T` (32,669) and `Toxic_Matrix` (90,352), CC-BY-4.0.** The `hin_Deva` column of
  Toxic-Matrix is clean 100 % Devanagari safety data (median 788 chars, 0 % near-dup), but **every refusal opens with the
  identical sentence** `मैं इस संकेत का जवाब नहीं दे सकता क्योंकि यह प्रकृति में संभावित रूप से विषाक्त है।` — paraphrase
  the opener before training or you install a canned refusal. | सुरक्षा-डेटा स्वच्छ है, पर हर अस्वीकृति एक ही वाक्य से
  शुरू होती है — प्रशिक्षण से पहले उसे पुनर्वाक्यांशित करें।
- **`aaditya/orca_dpo_pairs-Hindi_`, `damerajee/hindi-dpo`, `damerajee/Dpo-hindi-clean`, `dhanushreddy29/hindi_orca_dpo_pairs`** — small community re-cuts of the same Orca-DPO lineage.
- **RLHF Can Speak Many Languages (Dang et al., 2024, arXiv:2407.02552)** — the strongest published evidence that Hindi
  needs *its own* preference data: *"increasing the number of languages in preference optimization training data
  consistently improves multilingual performance compared to English-only training data, increasing win-rates by up to
  7.0 % from 46.4 % to 53.4 % when all languages are included"*, and *"RLOO as an online method achieves better overall
  performance than DPO by a maximum 10.6 % difference in their average win-rates (54.4 % vs 43.8 %)"*. Hindi is among the
  23 languages. **The ML-23-230K preference mixture described in the paper is NOT on the CohereLabs Hub org** — we
  enumerated it and found only Aya SFT/eval/red-teaming sets. | बहुभाषी अभिरुचि-डेटा जोड़ने से जीत-दर 46.4 % → 53.4 %;
  ऑनलाइन RLOO, DPO से 10.6 % बेहतर। पर वह अभिरुचि-मिश्रण सार्वजनिक रूप से जारी नहीं हुआ।
- **`CohereLabs/aya_redteaming`** — multilingual red-teaming prompts; **not samplable** (the HF dataset viewer is disabled for this repo), so its Hindi coverage is **UNVERIFIED**.
- **PARIKSHA (Watts et al., 2024)** — cited by MILU as having collected *"more than 90,000 human preferences"* across
  25+ Indic models; a potential source of *human* Indic preference signal. Availability **UNVERIFIED**. | 90,000+ मानव
  अभिरुचियाँ एकत्र कीं; उपलब्धता असत्यापित।

---

## 8. Closed Industrial Post-Training Recipes (no data release) / आठ. बंद औद्योगिक पोस्ट-ट्रेनिंग विधियाँ

### Sarvam-M (Sarvam AI, May 2025) — the most detailed public Hindi post-training recipe
- **Link:** https://www.sarvam.ai/blogs/sarvam-m (27-min technical blog; model on HF, **data not released**)
- **Base:** Mistral-Small 24B (Apache-2.0). Stages: SFT → RLVR (GRPO) → inference optimisation.
- **Prompt curation / प्रॉम्प्ट चयन:** **11.5 M prompts** collected from selected HF fine-tuning datasets → **~7 M** after
  min-hash + fuzzy dedup → **~5.2 M English** after lang-ID + Gemma-2-9B filtering → classified for quality, hardness and
  16 categories by **Llama-3.3-70B** → embedded with **gte-Qwen2-7B**, clustered into **100,000 faiss clusters**,
  semantically deduplicated within cluster at cosine 0.8 → **3.7 M** final prompts (quality: 61.31 % "excellent",
  32.98 % "good"; difficulty: 6.11 % "very hard", 44.45 % "hard"). | 1.15 करोड़ प्रॉम्प्ट → 37 लाख, गुणवत्ता/कठिनाई
  वर्गीकरण और 1 लाख क्लस्टरों में सिमैंटिक डुप्लिकेट-हटाव के बाद।
- **Indic conversion / भारतीय-भाषा रूपांतरण:** *"we convert about 30 % of the coding, math, and reasoning prompts, and
  50 % of other prompts to Indian languages. We chose to have a **28 % representation of Hindi** and 8 % representation
  each for 9 other languages … **50 % of the translations are in the native script, while 25 % each are in code-mixed and
  romanised scripts.**"* Translation is done by **Llama-3.1-8B models fine-tuned for the task with expert oversight**,
  not an off-the-shelf MT system. | अनुवाद कार्य-विशिष्ट रूप से फ़ाइन-ट्यून किए गए Llama-3.1-8B मॉडलों से, तैयार MT
  प्रणाली से नहीं।
- **Completions / उत्तर-निर्माण:** a custom **generative reward scorer** (Llama-3.3-70B fine-tuned on 120 K
  Gemini-1.5-Pro-judged responses) with **"real-value scoring"** (probability-weighted expectation over the 0–9 score
  token) — accuracy **72.85 % → 85.53 %** across 11 languages. Best completions came from **DeepSeek-R1 with English
  thinking tokens and Indic-language output in the non-thinking tokens** (avg > 8/9 across all 10 Indic languages). For
  code-mixed and romanized prompts *"none of the models generated good results"*, so those were produced by converting
  formal-Indic outputs with their in-house transliteration models. | सर्वोत्तम उत्तर DeepSeek-R1 से मिले (अंग्रेज़ी में
  सोच, भारतीय भाषा में उत्तर); कोड-मिक्स्ड/रोमनीकृत के लिए कोई भी मॉडल अच्छा नहीं था।
- **Character training / चरित्र-प्रशिक्षण:** **0.5 %** of pairs flagged for political/geographical/cultural bias were
  regenerated with a debiased model (Perplexity R1-1776); **~5 %** identified as needing Indian cultural relevance were
  regenerated with a culture-inducing prompt. | 0.5 % पक्षपाती जोड़े पुनर्जनित; ~5 % को भारतीय सांस्कृतिक प्रासंगिकता हेतु
  पुनर्जनित किया गया।
- **Results / परिणाम:** +20 % avg on Indian-language benchmarks, +21.6 % maths, +17.6 % programming; **GSM-8K-IN-R
  0.82 vs 0.44** for the base (a +86 % relative gain on romanized Indian-language GSM-8K); MILU-IN 0.75 vs 0.59.
- **Relevance to this project / प्रासंगिकता:** this is the recipe to copy. Nothing in it depends on proprietary data —
  only on a scorer, a curriculum, and a script-form split. | नकल करने योग्य विधि; इसमें कुछ भी प्रोप्रायटरी डेटा पर निर्भर नहीं।

### Llama-3-Nanda-10B-Chat (Choudhury et al., 2025)
- **Link:** arXiv:2504.06011 — CC-BY-NC-SA-4.0
- **SFT data (not released) / SFT डेटा (जारी नहीं):** **~81 K instructions total** — **~39 K English** (of which ~20 K
  maths; 7.7 M prompt + 9 M response tokens ≈ 17 M), **~22 K Hindi** (3.8 M prompt + 10 M response tokens ≈ 14 M), and
  **20 K safety** examples sampled from a Hindi-specific safety collection covering 8 attack types and 100+ categories.
- **The Hindi 22 K is split by *register*, which is unusual and worth copying:** *"**Formal Hindi** – The translated
  instances are written in Devanagari script with a style of writing consistent with official documents"* (**~13.5 K**)
  vs *"**Casual Hindi** – Generated translations contain Hindi (and some English) words using a mix of Devanagari and
  Latin scripts"* (**~8.5 K**). Produced by machine translation, then *"several Hindi language experts ensure the quality
  of translations by manually verifying a sample."* | हिन्दी हिस्सा **औपचारिक (≈13.5 हज़ार)** और **अनौपचारिक (≈8.5 हज़ार)**
  में बँटा है — यह विभाजन नक़ल करने योग्य है।
- **Training / प्रशिक्षण:** instructions oversampled to **300 %**, giving SFT over ~100 M tokens (47 M Hindi + 53 M English); loss on answer tokens only.

### Nemotron-Mini-Hindi-4B (Joshi et al., NVIDIA, 2024)
- **Link:** arXiv:2410.14815
- **The most decision-relevant ablation in this whole review / इस पूरी समीक्षा का सबसे निर्णय-प्रासंगिक परीक्षण:**
  *"Due to the lack of a high-quality Hindi SFT corpus, we leverage English-only data for SFT. We also experimented with
  translated English data (filtered using back-translation-based methods) for SFT, but **did not observe any improvements
  with this addition**. We found that using the English-only SFT corpus enhances instruction-following capabilities in
  Hindi, highlighting the cross-lingual transferability of these skills."* For the DPO stage they used **~200 K English +
  60 K synthetic Hindi** pairs and *"observe that incorporating synthetic Hindi samples during this stage improves the
  overall performance."*
- **Table 5 (SubjectiveEval / IndicQuest-Hi / IndicQuest-En) on the Hindi-pretrained base:** SFT(En)+DPO(En) **3.81 /
  4.12 / 4.02**; SFT(En)+DPO(En+Hi) **4.30 / 4.10 / 4.03**; SFT(En+Hi)+DPO(En) **4.28 / 4.06 / 4.02**;
  SFT(En+Hi)+DPO(En+Hi) **4.25 / 4.13 / 4.04**. Hindi SFT and Hindi DPO each give ≈ +0.5 on SubjectiveEval **and do not
  stack**. | हिन्दी SFT और हिन्दी DPO दोनों अलग-अलग ≈ +0.5 देते हैं, पर मिलकर जुड़ते नहीं।
- **General SFT corpus:** ~200 K examples (English), 1 epoch, batch 1024, lr 5e-6→9e-7 cosine. Ablation used ~70 K
  high-quality Hindi examples selected from a pool of 200 K translated samples.

### Mantra-14B (Kadiyala et al., 2025) — directly relevant to a Qwen-class project
- **Link:** arXiv:2504.09753 · models/datasets at https://huggingface.co/1-800-LLMs (component datasets are public;
  the merged Hindi instruct sets `1024m/Qwen-2.5-14B-Hindi-Instruct-Data`, `1024m/PHI-4-Hindi-Instruct-Data`,
  `1024m/Hindi-Gemma-Post-Training` are **gated**).
- **Data / डेटा:** *"the collected dataset had **3.12 M samples** with a nearly 50:50 ratio of English and Hindi data.
  Around **90 K** samples … cover localized and cultural knowledge … After filtering the training data, we had around
  **485 K** samples, of which **20 % are of localized domain and cultural knowledge**."* Translation of Big-Bench-Hard,
  XNLI and XL-Sum used **GPT-4o-mini via Azure**; Indian FAQ tables (legal, UPSC, tax, medicines, cuisines, travel) were
  converted to instruction pairs by GPT-4o-mini and then *"manually verified by multiple annotators"*; Aya translation /
  simplification / summarization subsets were also mixed in. | 31.2 लाख नमूने → छँटाई के बाद 4.85 लाख, जिनमें 20 %
  स्थानीय/सांस्कृतिक ज्ञान।
- **Ratio finding / अनुपात-निष्कर्ष:** *"in case of Qwen, the best results were obtained when ratio of Hindi is higher
  than 50 %, but for Phi-4, the results were better with ratio of Hindi less than 50 %."* Reported ~3 % average gain over
  the base models. | **Qwen के लिए 50 % से अधिक हिन्दी सर्वोत्तम; Phi-4 के लिए 50 % से कम।**

### Krutrim-1 / Krutrim-2 (Ola Krutrim, 2025)
arXiv:2502.09642. Describes India-centric SFT across ~10 task families plus a second-stage SFT to reduce factual
hallucination (base instruction-tuned model *"hallucinated around 33 %"* on factual questions) and a DPO alignment stage.
**No SFT/DPO dataset is released**; the `krutrim-ai-labs` HF org publishes benchmarks (`IndicVisionBench`, `VoiceAgentBench`,
`IndicST`, `MUTANT`) and the **`BhashaKritika`** synthetic *pretraining* set, not post-training data. | कोई SFT/DPO डेटासेट
जारी नहीं; संगठन केवल बेंचमार्क और प्री-ट्रेनिंग डेटा प्रकाशित करता है।

### Others (methodology only)
**OpenHathi** (Sarvam AI, 2023 — blog-only LLaMA-2 7B Hindi CPT base for Airavata), **Sarvam-1** (2B),
**PARAM-1 / BharatGen** (arXiv:2507.13390), **Project Indus**, **Navarasa 2.0** (Telugu-LLM-Labs Indic-Gemma —
*"approximately 650 K instruction samples across 18 datasets covering 15 Indian languages plus English"*; its Hindi
component is `ravithejads/samvaad-hi-filtered` + `HydraIndicLM/hindi_alpaca_dolly_67k`, i.e. Samvaad + translated
Alpaca/Dolly; base licence `gemma-terms-of-use`), **Tamil-Llama / Ambari / Tensoic** (same translate-then-LoRA template
applied to other Indic languages).

---

## 9. Methodology & Critique of Translated SFT / नौ. अनूदित SFT की पद्धति व आलोचना

- **Aya (arXiv:2402.06619)** — professional post-editors had to edit **60.0 % of NLLB-translated Hindi Dolly prompts**
  (HTER 6.16, HChrF 95.00). Crucially, **HChrF was 95** while 60 % of prompts still needed fixing — *chrF-family metrics
  do not detect the errors that matter for instruction data.* | **HChrF 95 होने के बावजूद 60 % प्रॉम्प्ट संपादन-योग्य थे** —
  chrF-श्रेणी के मेट्रिक निर्देश-डेटा की महत्वपूर्ण त्रुटियाँ नहीं पकड़ते।
- **Multilingual Instruction Tuning With Just a Pinch of Multilinguality (Shaham et al., ACL Findings 2024,
  arXiv:2401.01854)** — *"only 40 multilingual examples integrated in an English tuning set substantially improve
  multilingual instruction-following, both in seen and unseen languages"*; *"diversifying the instruction tuning set with
  even just 2-4 languages significantly improves cross-lingual generalization."* Hindi is one of their 12 languages. The
  authors flag their own limitation: the data was Google-Translate-produced, *"not originally sourced by native speakers."*
  | अंग्रेज़ी सेट में मात्र **40 बहुभाषी उदाहरण** जोड़ने से बहुभाषी निर्देश-पालन में उल्लेखनीय सुधार होता है।
- **Nemotron-Mini-Hindi (arXiv:2410.14815)** — translated Hindi SFT gave *no* improvement over English-only SFT on a
  Hindi-pretrained base; Hindi DPO did. See §8.
- **Airavata (arXiv:2401.15006)** — the human evaluation (50 prompts, single annotator, rubrics IFA / CNS = "Closeness to
  Native Speaker" / CQ) found Airavata produces **more natural-sounding Hindi than GPT-4 and ChatGPT** while trailing on
  instruction-following and content quality. The authors also caution that the evaluation *"is not robust and thorough."*
  | Airavata GPT-4/ChatGPT से **अधिक स्वाभाविक हिन्दी** लिखता है पर निर्देश-पालन व विषय-गुणवत्ता में पीछे है।
- **Benchmarking Hindi LLMs (Kamath et al., NVIDIA; arXiv:2508.19831, v2 Oct 2025)** — releases **IFEval-Hi (848), MT-Bench-Hi (200),
  GSM8K-Hi (1,319), ChatRAG-Hi (5,948), BFCL-Hi (2,251)** built by *"from-scratch human annotation"* plus a
  *"translate-and-verify"* workflow. Their acceptance rule for machine-translated content was **chrF++ ≥ 90** on the
  back-translation — nearly double Airavata's threshold. They argue translated benchmarks *"often test a model's ability
  to comprehend translated English rather than its native fluency and instruction fidelity."* | अनूदित सामग्री स्वीकारने
  हेतु उन्होंने **chrF++ ≥ 90** रखा — Airavata की सीमा से लगभग दुगना।
- **MILU (arXiv:2411.02538, AI4Bharat)** — a natively-authored Indic benchmark from **1,500+ Indian competitive exams**
  across 8 domains, 41 subjects, 11 languages, precisely because *"translating existing English benchmarks into Indian
  languages fails to capture this knowledge."* GPT-4o tops it at **74 %**. Use MILU, not translated MMLU, to gate Hindi
  progress. | अनूदित बेंचमार्क भारतीय ज्ञान नहीं पकड़ते; MILU 1,500+ भारतीय परीक्षाओं से बना है।
- **IndicTrans2 (Gala et al., arXiv:2305.16307)** — the MT engine behind IndicInstruct, IndicAlign and RomanSetu; its
  Hindi bitext pool is the largest in BPCC (473.2 M sentence pairs mined, 27.1 M extracted). For reference, Airavata's own
  Table 5 reports En→Hi **chrF++ 55.41 (Flores) / 54.23 (IN22-Gen)** for OpenHathi — i.e. even a strong system leaves
  substantial surface divergence. | IndicTrans2 ही इन सभी सेटों का अनुवाद-इंजन है; फिर भी En→Hi chrF++ ≈ 54–55 ही रहता है।
- **RomanSetu (arXiv:2401.14280)** — romanization halves-to-quarters token fertility and matches or beats native-script
  instruction tuning; relevant if tokenizer efficiency is a constraint for your Qwen base. | रोमनीकरण टोकन-फ़र्टिलिटी
  घटाता है और मूल-लिपि ट्यूनिंग के बराबर या बेहतर है।

---

## 10. Comparison Table — Hindi-usable SFT/preference data actually sampled / दस. तुलना तालिका

| Dataset | Hindi rows (verified) | Origin | Script | Licence | Sampled Deva-dominant | Near-dup | Deep-offset | Tier |
|---|---|---|---|---|---|---|---|---|
| `ai4bharat/indic-align` `hin_Deva` (Wiki-Conv) | 141,435 | synth (Llama-2-70B) → IndicTrans2 | Deva | CC-BY-4.0 | 100 % | 0.8 % | robust | **A** |
| `ai4bharat/indic-align` `hin_Deva` (Wiki-Chat) | 198,254 | synth (Llama-2-70B + Mixtral) → IT2 | Deva | CC-BY-4.0 | 100 % | 0.8 % | robust | **A** |
| `ai4bharat/indic-align` `hin_Deva` (WikiHow) | 20,313 | human EN wikiHow → IT2 | Deva | CC-BY-4.0 | 100 % | 0.0 % | robust | **A** |
| `ai4bharat/indic-align` `hin_Deva` (Indic-ShareLlama) | 21,171 | ShareGPT prompts + Llama-2-70B → IT2 | Deva | CC-BY-4.0 | 100 % | 2.5 % | robust | **A** |
| `sarvamai/samvaad-hi-v1` | 101,476 conv. | model-gen over Indic sources | Deva/En/Hing | Apache-2.0 | 45 % | 0.0 % | stable | **A** |
| `indic-instruct-data-v0.1` `wikihow/hi` | 6,055 | **NATIVE** hi.wikihow.com | Deva | CC-0 | 97 % | 2.0 % | stable (prompt bug) | **A** |
| `indic-instruct-data-v0.1` `anudesh/hi` | 7,577 | **NATIVE prompts** + Llama-2-70B → IT2 | Deva | CC-BY-4.0 | 100 % | 1.8 % | stable | **A** |
| `BhabhaAI/Hi-Instruct-v0` | 9,969 | directly generated in Hindi | Deva | unspecified | 100 % | 7.2 % | stable | **A** |
| `BhabhaAI/openhermes-2.5-hindi` | 620,211 | OpenHermes-2.5 (GPT-4) translated | Deva | unspecified | 93 % | 1.5 % | stable (MCQ bug) | **B** |
| `apurvagup/ultrachat_hindi_seamless` | 185,542 | UltraChat → SeamlessM4T | Deva | unspecified | 100 % | 0.0 % | robust | **B** |
| `shreyas18/Hindi_instruct_1_5M_v1` | 1,488,730 | undocumented MT | Deva | none | 100 % | 0.5 % | stable | **B** |
| `atharvanighot/Hindi-Instruct-500K` | 508,609 | ⊂ the above | Deva | none | 100 % | 0.0 % | stable | **B** |
| `BhabhaAI/orca-math-...-hindi-filtered` | 188,943 | Orca-Math (GPT-4) translated | Deva | MIT | 100 % | 0.0 % | robust | **B** |
| `GenVRadmin/Aryabhatta-Orca-Maths-Hindi` | 200,000 | same lineage | Deva | MIT | 100 % | 0.0 % | robust | **B** |
| `MBZUAI/Bactrian-X` `hi` | 67,017 | GTranslate prompts + gpt-3.5 | Deva | **CC-BY-NC-4.0** | 96 % | 0.5 % | stable | **B** |
| `FreedomIntelligence/alpaca-gpt4-hindi` | 49,969 | Alpaca-GPT4 translated | Deva | unspecified | 99 % | 0.0 % | stable | **B** |
| `FreedomIntelligence/evol-instruct-hindi` | 59,022 | Evol-Instruct translated | Deva | unspecified | 78 % | 0.0 % | degrades | **B** |
| `iamshnoo/alpaca-cleaned-hindi` | 51,760 | Alpaca translated | Deva | unspecified | 97 % | 0.0 % | stable | **B** |
| `saillab/alpaca-hindi-cleaned` | 41,601 | Alpaca translated | Deva | unspecified | 100 % | 0.0 % | stable | **B** |
| `bingbangboom/gsm8k-hindi` | 7,473 (+7,473 socratic) | GSM8K translated | Deva | MIT | 100 % | 1.5 % | robust | **B** |
| `CohereLabs/aya_collection_language_split` `hindi` | 3,772,864 | FLAN etc. → NLLB | Deva | Apache-2.0 | 84 % | 0.0 % | noisy throughout | **B−** |
| `CohereLabs/aya_dataset` (Hindi) | **1,153** | **human-written by fluent speakers** | Deva | Apache-2.0 | 93 % | 0.0 % | n/a (tiny) | **A (tiny)** |
| `smangrul/hindi_instruct_v1` ≡ `justinj92/hinglish_sharegpt_v0.1` | 20,215 | mixed-script instruct incl. real Hinglish | Deva+Hing | MIT / unspecified | 37 % | 1.2 % | stable | **B (Hinglish)** |
| `indic-instruct-data-v0.1` `flan_v2/hi` | 67,463 | FLAN → IT2 | Deva | Apache-2.0 | 98 % "mixed"* | 0.5 % | broken tasks | **C** |
| `indic-instruct-data-v0.1` `lm_sys/hi` | 50,000 | LMSYS → IT2 | Deva | **LMSYS agreement** | 100 % "mixed"* | 0.0 % | stable but lossy | **C** |
| `pranjalchitale/indicsft` | 25,030,167 (all langs) | undocumented | mixed | **none** | 16 % | 3.5 % | inconsistent | **C** |
| `zicsx/indic-align-hindi` | 13,310,858 | IndicAlign, script-filtered | Deva | unspecified | 100 % | 3.2 % | **Bodo at 200 k** | **C** |
| `manishiitg/aditi-syn-v2` | 55,450 | synthetic Hi/Hinglish | Deva+Hing | Apache-2.0 | 36 % | 1.2 % | uneven | **C** |
| `guneetsk99/hindi_instruction_set_187K` | 187,525 | ⊃ Bactrian-X-hi | Deva | **CC-BY-NC-ND-4.0** | 48 % | 0.0 % | → bitext | **C** |
| `fhai50032/Hindi-Instruct-HQ` | 27,999 | mistral-large-2402 | En→Deva | unspecified | 40 % | 0.5 % | stable | **C** |
| `aaditya/orca_dpo_pairs-Hindi` (DPO) | 10,305 pairs | Orca-DPO code-mixed | Hinglish | unspecified | 0 % (84 % Hing) | 3.2 % | stable | **B (DPO)** |
| `manishiitg/aditi-dpo-prompts` (DPO prompts) | 48,745 | synthetic | Deva+Latn | unspecified | 44 % | 0.0 % | stable | **B (DPO)** |
| `ai4bharat/indic-align` `Toxic_Matrix` `hin_Deva` | 90,352 | Mistral-7B + Llama-2 | Deva | CC-BY-4.0 | 100 % | 0.0 % | robust (templated refusal) | **B (safety)** |
| `ServiceNow-AI/M2Lingual` (Hindi) | **2,788** | multilingual Evol-Instruct | Deva | **CC-BY-NC-SA-4.0** | — | 0.2 % | n/a | **C** |
| `Abhishekcr448/Hinglish-Everyday-Conversations-1M` | 1,001,323 | synthetic small-talk | Hinglish | MIT | 0 % (100 % Hing) | 0.0 % | stable, very short | **C (style)** |
| `festvox/cmu_hinglish_dog` | 8,060 | **human code-mixed** | Hinglish | CC-BY-SA-3.0 + GFDL | 0 % | 1.5 % | n/a | **B (style)** |
| `WillHeld/hinglish_top` | 2,993 train | semantic parses | Hinglish | unspecified | 0 % | 5.5 % | n/a | **C** |
| `ai4bharat/indic-align` `IndoWordNet` | 96,843,950 | IndoWordNet templates | Deva | CC-BY-4.0 | 100 % | ~template-repeat | **exclude** | **D** |
| `GenVRadmin/Samvaad-Mixed-Language-3` | 25,920 | gpt-3.5-turbo, bias-primed | mixed | MIT label | 5 % | 0.5 % | factually wrong | **D** |
| `pfin123/hindi-aggregated` | 745,066 | raw news web text | Deva | Apache-2.0 | 88 % | 0.0 % | **not SFT data** | **D** |
| `equal-ai/conversational_hindi` | 47,107 | ASR audio segments | — | unspecified | — | — | **not SFT data** | **D** |

\* "mixed" here is an artefact of these configs storing the Hindi text *and* its English back-translation in the same row;
the Hindi fields themselves are ~100 % Devanagari. | * इन कॉन्फ़िग में हिन्दी पाठ और उसका अंग्रेज़ी बैक-ट्रांसलेशन एक ही
पंक्ति में हैं, इसीलिए "mixed" दिखता है।

---

## 11. Access Limitations & Corrections / ग्यारह. पहुँच-सीमाएँ व सुधार

- **HF `datasets-server` `/rows` was rate-limited (HTTP 429) for this environment throughout the session**, both with and
  without an auth token. All sampling was therefore done by reading `refs/convert/parquet` shards over HTTP byte ranges.
  The `/parquet`, `/splits` and `/size` endpoints worked. | `/rows` एंडपॉइंट पूरे सत्र में 429 देता रहा; अतः Parquet
  शार्ड्स सीधे पढ़े गए।
- **Gated datasets (not sampled) / गेटेड डेटासेट (नमूना नहीं लिया गया):** `1024m/Qwen-2.5-14B-Hindi-Instruct-Data`
  (gated: auto), `1024m/PHI-4-Hindi-Instruct-Data` (auto), `1024m/Hindi-Gemma-Post-Training` (manual) — the merged
  Mantra-14B training sets; `ai4bharat/MILU` (auto); `nvidia/Nemotron-Post-Training-Dataset-v2` (auto).
  Their sizes are reported from card metadata only. | ये गेटेड हैं; इनके आकार केवल कार्ड-मेटाडेटा से लिए गए।
- **`CohereLabs/aya_redteaming`:** the HF dataset viewer is **disabled** for this repo (`"Not supported: dataset viewer is disabled."`), so it could not be sampled and its Hindi coverage is **UNVERIFIED**. | इस रेपो का डेटासेट-व्यूअर बंद है; अतः नमूना नहीं लिया जा सका और इसकी हिन्दी उपस्थिति असत्यापित है।
- **`www.sarvam.ai/blogs/sarvam-m` returned HTTP 403 to the fetch tool**; the content quoted above was retrieved with a
  direct `curl` + HTML-to-text extraction. All Sarvam-M numbers are from that page's own text. | Sarvam-M ब्लॉग fetch-टूल
  को 403 देता है; उद्धरण सीधे curl से प्राप्त किए गए।
- **Number discrepancy, `ai4bharat/indic-align`:** the Hub reports **97,419,884** rows across configs while the ACL paper's
  Table 7 sums to **~74.7 M** (IndoWordNet 74,272.2 k vs **96,843,950** on the Hub). Both figures are recorded above; not
  reconciled. Likewise Wiki-Chat 202 k (paper) vs **198,254** (Hub) and Wiki-Conv 144 k vs **141,435**. | पेपर और हब की
  संख्याओं में अंतर है; दोनों दर्ज हैं, समाधान नहीं किया गया।
- **`ai4bharat/indic-instruct-data-v0.1` has no top-level licence tag** on the Hub; per-subset licences are taken from
  the Airavata paper's own text. | हब पर कोई शीर्ष-स्तरीय लाइसेंस नहीं; उप-सेट लाइसेंस पेपर से लिए गए।
- **Naming traps corrected / नाम-भ्रम सुधार:** `equal-ai/conversational_hindi` is an **ASR audio** dataset;
  `pfin123/hindi-aggregated` is a **raw web-text pretraining** corpus; neither is instruction data despite topping
  keyword searches. | ये दोनों निर्देश-डेटा नहीं हैं।
- **Not found / नहीं मिला:** no Hindi **Magpie** dataset on `Magpie-Align`; no released **Cohere ML-23-230K** multilingual
  preference mixture; no **Sarvam-M / Krutrim / Nanda / Nemotron-Hindi / PARAM-1** SFT or DPO data release; **no public
  Hindi code-SFT dataset** of any scale. | Magpie का हिन्दी संस्करण, Cohere का ML-23-230K, तथा किसी भी औद्योगिक मॉडल का
  SFT/DPO डेटा सार्वजनिक नहीं; **हिन्दी कोड-SFT डेटासेट भी नहीं**।
- **Language-ID caveat we applied / भाषा-पहचान चेतावनी:** every "Hindi" claim above was checked *by reading the text*, not
  by trusting the config name — which is how the Bodo rows in `zicsx/indic-align-hindi` and the Marathi rows in
  `indic-align/Anudesh` were caught. | हर "हिन्दी" दावा पाठ पढ़कर जाँचा गया, कॉन्फ़िग-नाम पर भरोसा करके नहीं।
- **`ServiceNow-AI/M2Lingual` and `CohereLabs/aya_dataset` Hindi counts** were obtained by scanning the **entire**
  `language` column (173,672 and 202,362 rows respectively), not by sampling. | ये दोनों गणनाएँ पूर्ण स्तंभ-स्कैन से हैं।

---

## 12. References / संदर्भ

1. Gala et al. *Airavata: Introducing Hindi Instruction-tuned LLM.* arXiv:2401.15006. https://arxiv.org/abs/2401.15006
2. Khan et al. *IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages.* ACL 2024. https://arxiv.org/abs/2403.06350
3. Singh et al. *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.* ACL 2024. https://arxiv.org/abs/2402.06619
4. Li et al. *Bactrian-X: Multilingual Replicable Instruction-Following Models with Low-Rank Adaptation.* arXiv:2305.15011. https://arxiv.org/abs/2305.15011
5. Maheshwary, Yadav, Nguyen, Mahajan, Madhusudhan (ServiceNow). *M2Lingual: Enhancing Multilingual, Multi-Turn Instruction Alignment in Large Language Models.* arXiv:2406.16783. https://arxiv.org/abs/2406.16783
6. Shaham et al. *Multilingual Instruction Tuning With Just a Pinch of Multilinguality.* Findings of ACL 2024 / arXiv:2401.01854. https://arxiv.org/abs/2401.01854
7. Joshi et al. *Adapting Multilingual LLMs to Low-Resource Languages using Continued Pre-training and Synthetic Corpus (Nemotron-Mini-Hindi 4B).* arXiv:2410.14815. https://arxiv.org/abs/2410.14815
8. Choudhury et al. *Llama-3-Nanda-10B-Chat: An Open Generative LLM for Hindi.* arXiv:2504.06011. https://arxiv.org/abs/2504.06011
9. Kadiyala et al. *Improving Multilingual Capabilities with Cultural and Local Knowledge in LLMs While Enhancing Native Performance (Mantra-14B).* arXiv:2504.09753. https://arxiv.org/abs/2504.09753
10. Kallappa et al. *Krutrim LLM: Multilingual Foundational Model for over a Billion People.* arXiv:2502.09642. https://arxiv.org/abs/2502.09642
11. Sarvam AI. *Sarvam-M: Explorations in Post Training and Inferencing Optimizations for a Hybrid Indic LLM.* 23 May 2025. https://www.sarvam.ai/blogs/sarvam-m
12. Husain et al. *RomanSetu: Efficiently Unlocking Multilingual Capabilities of LLMs via Romanization.* ACL 2024 / arXiv:2401.14280. https://arxiv.org/abs/2401.14280
13. Gala et al. *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages.* TMLR / arXiv:2305.16307. https://arxiv.org/abs/2305.16307
14. Verma, Khan, Kumar, Murthy, Sen (AI4Bharat + IBM Research India). *MILU: A Multi-task Indic Language Understanding Benchmark.* arXiv:2411.02538. https://arxiv.org/abs/2411.02538
15. Kamath, Singla, Paul, Joshi, Vaidya, Chauhan, Wartikar (NVIDIA). *Benchmarking Hindi LLMs: A New Suite of Datasets and a Comparative Analysis.* arXiv:2508.19831. https://arxiv.org/abs/2508.19831
16. Dang et al. *RLHF Can Speak Many Languages: Unlocking Multilingual Preference Optimization for LLMs.* arXiv:2407.02552. https://arxiv.org/abs/2407.02552
17. Singh et al. *Sample-Efficient Language Model for Hinglish Conversational AI.* arXiv:2504.19070. https://arxiv.org/abs/2504.19070
18. Pundalik et al. *PARAM-1: BharatGen 2.9B Model.* arXiv:2507.13390. https://arxiv.org/abs/2507.13390

**Primary data links / प्रमुख डेटा लिंक:**
`ai4bharat/indic-instruct-data-v0.1` · `ai4bharat/indic-align` · `sarvamai/samvaad-hi-v1` · `CohereLabs/aya_dataset` ·
`CohereLabs/aya_collection_language_split` · `MBZUAI/Bactrian-X` · `ServiceNow-AI/M2Lingual` ·
`BhabhaAI/{openhermes-2.5-hindi, orca-math-word-problems-200k-hindi-filtered, Hi-Instruct-v0, Cross-Hindi-Hinglish-chat, indic-instruct-data-v0.2-filtered}` ·
`GenVRadmin/{Aryabhatta-Orca-Maths-Hindi, Samvaad-Mixed-Language-3}` · `manishiitg/{aditi-syn-v2, aditi-dpo-prompts}` ·
`FreedomIntelligence/{alpaca-gpt4-hindi, evol-instruct-hindi, sharegpt-hindi}` ·
`{iamshnoo/alpaca-cleaned-hindi, saillab/alpaca-hindi-cleaned, smangrul/hindi_instruct_v1, NebulaByte/alpaca-gpt4-hindi-hinglish}` ·
`{apurvagup/ultrachat_hindi_seamless, shreyas18/Hindi_instruct_1_5M_v1, atharvanighot/Hindi-Instruct-500K, guneetsk99/hindi_instruction_set_187K, pranjalchitale/indicsft, zicsx/indic-align-hindi, fhai50032/Hindi-Instruct-HQ}` ·
`{aaditya/orca_dpo_pairs-Hindi, bingbangboom/gsm8k-hindi}` ·
`{festvox/cmu_hinglish_dog, WillHeld/hinglish_top, Abhishekcr448/Hinglish-Everyday-Conversations-1M, justinj92/hinglish_sharegpt_v0.1}` ·
eval: `ai4bharat/{MILU, IndicIFEval}` · `sarvamai/gsm8k-indic`
