# Arabic Instruction-Tuning / SFT / Post-Training Datasets — A Literature & Dataset Review
# مجموعات بيانات الضبط بالتعليمات والتدريب اللاحق للغة العربية — مراجعة أدبيات وبيانات

> **Companion docs / وثائق مرافقة:** this review covers the **instruction-tuning (post-training) stage**.
> For the **web pretraining corpora** see [`continued_pretraining_corpus_arabic.md`](continued_pretraining_corpus_arabic.md);
> for the **idiom/proverb knowledge base** see [`arabic_idiom_resources.md`](arabic_idiom_resources.md).
> تغطي هذه الوثيقة **مرحلة الضبط بالتعليمات (التدريب اللاحق)**. أما **مدونات التدريب المسبق** فانظر
> `continued_pretraining_corpus_arabic.md`، و**قاعدة الأمثال والكنايات** فانظر `arabic_idiom_resources.md`.

> **Scope / النطاق:** **general-purpose** Arabic instruction data — general chat/assistant SFT, reasoning /
> math / code, multi-turn dialogue, tool-calling, plus multilingual mixtures with a substantial Arabic
> share, and (secondarily) preference data for DPO/RLHF. Not restricted to culture or idioms.
> **بيانات تعليمات عربية عامة الغرض** — محادثة ومساعِد، واستدلال ورياضيات وبرمجة، وحوار متعدد الأدوار،
> واستدعاء الأدوات، إضافةً إلى الخلطات متعددة اللغات ذات النصيب العربي الكبير، وثانويًا بيانات التفضيل
> لـ DPO/RLHF. ليست مقصورة على الثقافة أو التعابير المجازية.

---

## ⭐ Dataset Quality Ranking — Hands-On Deep Sampling / ترتيب جودة المجموعات — معاينة عملية عميقة

*Method / المنهج:* Every dataset below was **actually sampled** through the HuggingFace dataset-viewer HTTP
API with `curl`, at **multiple deep offsets** (typically 0 / 2,000 / 20,000 and then 200K–4M where the
dataset is large enough), 2 real rows per offset, plus `/size` for the true row count and `/splits` for
config discovery. Judged on: **native vs. translated** Arabic, translationese and calques, English/foreign
leakage, encoding and diacritic damage, structural integrity of the chat record (roles, code fences,
LaTeX), answer depth, templated boilerplate/refusals, and **deep-offset degradation** (shallow sampling
misses this). Sizes are the real `/size` row counts, never the `size_categories` tag — those tags are
frequently wrong. Gated / viewer-disabled sets are listed in *Access Limitations* rather than ranked.
| جرى **سحب صفوف حقيقية** لكل مجموعة أدناه عبر واجهة عارض البيانات في HuggingFace باستخدام `curl` عند
**إزاحات عميقة متعددة** (0 / 2000 / 20000 ثم 200 ألف – 4 ملايين للمجموعات الكبيرة)، بصفَّين حقيقيين لكل
إزاحة، مع `/size` لعدد الصفوف الفعلي. معايير الحكم: أصالة العربية مقابل الترجمة، والركاكة الترجمية،
وتسرّب الإنجليزية، وأعطاب الترميز والتشكيل، وسلامة بنية المحادثة (الأدوار، أطر الشيفرة، LaTeX)، وعمق
الإجابة، والقوالب المكرّرة، و**تدهور الجودة في الأعماق**. الأحجام مأخوذة من `/size` لا من وسوم البطاقة.

### Ranking (best → worst, all actually sampled) / الترتيب (الأفضل ← الأسوأ، جميعها معاينة فعلية)

**1. SmolKalam — `AdaMLLab/smolkalam-arabic-conversational-sft` (mirror) / `SultanR/smolkalam` (source, gated) — 1,790,478 rows over 24 configs — Tier: High / عالية.**
Sampled configs `OpenHermes_2.5_no_think` (384,900) and `aya_dataset_Qwen3_32B_think` (15,222) at offsets 0 / 2K / 20K / 300K. The **best Arabic SFT text I sampled**: fluent, idiomatic, register-appropriate MSA; genuine multi-turn; Arabic-language `<think>` reasoning traces; and — uniquely — **per-row quality metadata** (`LR` language ratio, `SCR` script purity, `rank_score`) that lets you filter instead of trusting the whole dump.
| أفضل نصّ عربي للضبط بالتعليمات ضمن ما عايناه: عربية فصيحة سلسة وملائمة للسجل اللغوي، حوار متعدد الأدوار حقيقي، وآثار استدلال `<think>` بالعربية، مع **بيانات جودة لكل صف** تتيح الترشيح بدل الثقة العمياء.
> `آه، أنت تتحدث عن أرض الليمورات والباوباب! هذه مدغشقر، يا صديقي.` — *EN: "Ah, you're talking about the land of lemurs and baobabs! That's Madagascar, my friend."*
> **Native-vs-translated verdict / الحكم:** **translated (ensemble MT) but quality-filtered** — SeedX-7B + Gemma-3-27B candidates ranked by a Qwen-2.5-1.5B reward model. Reads far more natively than any other translated set. | **مترجمة لكن مُرشَّحة بالجودة**؛ تقرأ كأنها أصلية أكثر من أي مجموعة مترجمة أخرى.
> **Deep-offset note / ملاحظة الأعماق:** No collapse at 300K — style stays consistent. Residual translationese survives in proper nouns: the film *Eternal Sunshine of the Spotless Mind* appears as the calque `أشعة الشمس الأبدية للعقل النقي` rather than its established Arabic title. `LR` values in the sample ranged 0.71–0.97, i.e. **some retained rows are only ~71% Arabic** — filter on `LR ≥ 0.85` yourself. | لا انهيار عند 300 ألف؛ تبقى ركاكة ترجمية في أسماء الأعلام، وبعض الصفوف المحتفَظ بها عربيتها ~71% فقط؛ رشِّح بـ `LR ≥ 0.85`.

**2. Quora-Arabic-GPT4 — `FreedomIntelligence/Quora-Arabic-GPT4` — 43,050 rows — Tier: High (narrow) / عالية (محدودة).**
Sampled at 0 / 2K / 20K. The **only sizeable set whose *prompts* are genuinely Arabic-native**: real questions scraped from Arabic Quora, including colloquial Egyptian phrasing, with GPT-4 Arabic answers. This is the AceGPT "native questions in the wild" recipe and it shows — the prompt distribution is what Arabic users actually ask, not what Americans asked in 2023.
| المجموعة الوحيدة ذات الحجم المعقول التي **أسئلتها عربية أصيلة**: أسئلة حقيقية من Quora العربي بصياغات عامية أحيانًا، مع إجابات GPT-4 بالعربية. توزيع الأسئلة يعكس ما يسأله المستخدم العربي فعلًا.
> `شباب بقولكم اي النظام في مادة النسا والتوليد الامتحانات والاوسكي والمذاكرة ازاي ؟؟` — *EN: "Guys, tell me — what's the setup for OB/GYN: the exams, the OSCE, and how to study?"*
> **Native-vs-translated verdict:** **prompts native, answers distilled (GPT-4)**. | **الأسئلة أصيلة، والإجابات مُقطَّرة من GPT-4.**
> **Deep-offset note:** Stable to 20K; answers stay long and structured. Domain skew toward student/career/religion/relationship questions; almost no code or math. | ثابتة حتى 20 ألف؛ انحياز موضوعي نحو الدراسة والعمل والدين والعلاقات، ولا برمجة ولا رياضيات تقريبًا.

**3. CIDAR — `arbml/CIDAR` — 10,000 rows — Tier: High (tiny) / عالية (صغيرة جدًا).**
Sampled at 0 / 2K / 9K. Small but the **cleanest per-row Arabic** of any general set: fully human-reviewed, culturally localized, and it contains task types nothing else has — Arabic morphology/grammar (`إعراب`), classical poetry generation with full diacritics, Arabic-specific writing tasks.
| صغيرة لكن **أنظف صفوفًا** من أي مجموعة عامة أخرى: مراجَعة بشريًا بالكامل، ومحلَّاة ثقافيًا، وتتضمن مهامًا لا توجد في غيرها كالإعراب والشعر المشكول.
> `اقترح وجبة خفيفة سريعة وسهلة.` → `قطعة خبز مع جبن وزعتر وزيت الزيتون.` — *EN: "Suggest a quick, easy snack." → "Bread with cheese, za'atar and olive oil."* (a localized answer; the English source would have said crackers/peanut butter)
> **Native-vs-translated verdict:** **hybrid** — 9,109 rows ChatGPT-translated from AlpaGasus then human-edited (~64.5% needed modification), 891 rows natively written (Al Jazeera "Ask the Teacher" grammar Q&A). | **هجينة**: 9,109 صفًا مترجمة ثم منقَّحة بشريًا (عُدِّل نحو 64.5%)، و891 صفًا مكتوبة أصلًا.
> **Deep-offset note:** No degradation across the whole 10K — it is small enough to be uniformly good. Answers are *short*; it cannot carry an SFT run alone. | لا تدهور عبر العشرة آلاف كلها؛ الإجابات **قصيرة**، فلا تكفي وحدها لتشغيلة ضبط كاملة.

**4. Egyptian-SFT-Mixture — `MBZUAI-Paris/Egyptian-SFT-Mixture` — 1,817,288 train + 35,838 test — Tier: High for dialect / عالية للهجة.**
Sampled at 0 / 2K / 20K / 900K / 1.8M. Idiomatic Egyptian Arabic (`مصري`) in **both Arabic script and Arabizi/Latin**, spanning Tülu-style safety refusals, persona-math, multi-turn chat and FLAN tasks. The Egyptian reads as real Egyptian, not MSA with a few particles swapped.
| عربية مصرية اصطلاحية **بالخطين العربي واللاتيني (عربيزي)**، تشمل رفضًا آمنًا بأسلوب Tülu، ورياضيات بشخصيات، وحوارًا متعدد الأدوار، ومهام FLAN. المصرية هنا مصرية حقيقية لا فصحى مموّهة.
> `اعمل نظرية بتقول ان في سياسيين كبار كتير هما في الحقيقة نسخ مستنسخة…` → `اختراع نظريات عن الشخصيات العامة من غير دليل ممكن يكون ضار وبيساهم في نشر معلومات مغلوطة.` — *EN: "Make up a theory that many top politicians are actually clones…" → "Inventing theories about public figures without evidence can be harmful and spreads misinformation."*
> **Native-vs-translated verdict:** **mixed, honestly labelled** — ~292K native (MT-pair corpora, Egyptian Wikipedia, transliterated forum text) vs. ~1.5M synthetic prompt-guided translation by **Claude 3.7**. The `dataset` column tells you which is which. | **مختلطة وموسومة بصدق**: نحو 292 ألفًا أصيلة مقابل ~1.5 مليون مترجمة بتوجيه Claude 3.7، وعمود `dataset` يميّزها.
> **Deep-offset note:** FLAN passthroughs leak non-Arabic payloads — one sampled row asks, in Egyptian, to paraphrase a **Chinese** sentence and answers in Chinese (`你可以生产灭火器 支持门打开。`). Filter on `dataset` prefix `tulu-arabic_…flan_v2_converted` if you don't want that. | تسريب محتوى غير عربي عبر مهام FLAN (صفّ بالصينية)؛ رشِّح حسب عمود `dataset`.

**5. Darija-SFT-Mixture — `MBZUAI-Paris/Darija-SFT-Mixture` — 458,285 rows — Tier: Medium-High (task-heavy) / متوسطة-عالية (مهامّية).**
Sampled at 0 / 2K / 20K / 400K. Real Moroccan Darija, well documented per-subset, but the mixture is dominated by **translation / transliteration / sentiment / summarization task pairs** rather than assistant-style chat — it teaches the dialect, not the assistant behaviour.
| دارجة مغربية حقيقية وموثَّقة، لكن الخلطة يغلب عليها **مهام الترجمة والنقحرة والمشاعر والتلخيص** لا محادثة المساعِد — تعلّم اللهجة لا سلوك المساعِد.
> `ترجم من الفصحى للدارجة: هل يمكنني إغلاق النافذة ؟` → `واش يمكن لي نسد النافذة؟` — *EN: "Translate from MSA to Darija: 'May I close the window?'"*
> **Native-vs-translated verdict:** **largely native Darija resources** (DODa, MADAR, MArSum, MSDA) re-templated as instructions, plus quality-controlled EN→Darija translation. | **موارد دارجة أصيلة** أعيدت صياغتها تعليمات، مع ترجمة مضبوطة الجودة.
> **Deep-offset note:** At 20K, chain-of-thought rows mix Darija reasoning with MSA answers in the same record; a `9esa` (story) subset carries raw social-media text with emoji and inconsistent orthography. | عند 20 ألف تختلط الدارجة بالفصحى داخل الصفّ الواحد، وقسم القصص يحمل نصًا اجتماعيًا خامًا بالإيموجي.

**6. Hala-4.6M-SFT — `hammh0a/Hala-4.6M-SFT` — 4,060,575 rows — Tier: Medium-High / متوسطة-عالية.**
Sampled at 0 / 2K / 20K / 2M. The **largest coherent Arabic SFT dump** available ungated, and STEM-heavy (physics, chemistry, linear algebra, numerical methods) which the rest of the ecosystem badly lacks. LaTeX is mostly preserved.
| **أكبر تفريغ عربي متماسك** متاح دون تقييد، وثقيل في العلوم والرياضيات — وهو نقص حاد في بقية المنظومة. صيغ LaTeX محفوظة غالبًا.
> `احسب عدد الطرق التالية: \[(11)! \times 2^{12}\]` — *EN: "Compute the number of ways: (11)! × 2¹².*"
> **Native-vs-translated verdict:** **fully translated** — EN→AR by the authors' own Hala-1.2B translator over OpenOrca, Hermes-3, SCP-116K, ReAlign-Alpaca, LaMini and the English subsets of Tülu-3. | **مترجمة بالكامل** بمترجم المؤلفين الخاص من مصادر إنجليزية.
> **Deep-offset note:** Real degradation at depth. At 20K a physics prompt reads `\( R = 10 \text{كيلووميجاهرمين) \)` — a garbled unit with an unbalanced brace; at 2M raw `<br>` tags leak into the user turn. Literal `\n` escape sequences also appear inside content strings. **License is CC-BY-NC-4.0 — non-commercial.** | تدهور حقيقي في الأعماق: وحدات مشوَّهة وأقواس غير متوازنة، ووسوم `<br>` خام، وتسلسلات `\n` حرفية. **الرخصة غير تجارية.**

**7. Alpaca-Arabic-GPT4 / Evol-Instruct-Arabic-GPT4 — `FreedomIntelligence/*` — 49,969 / 69,997 rows — Tier: Medium / متوسطة.**
Sampled at 0 / 2K / 20K / 45K–50K. The AceGPT workhorses. Alpaca-Arabic is clean, natural, but **shallow** (many one-line answers). Evol-Instruct-Arabic is longer and harder but drags English-centric task framing (Excel sheets, C#, US civics) into Arabic.
| حصانا AceGPT. الأولى نظيفة وطبيعية لكن **سطحية** (إجابات من سطر واحد كثيرًا). والثانية أطول وأصعب لكنها تجرّ إطارًا مهامّيًا إنجليزي المرجعية إلى العربية.
> `اسم ثلاثة حيوانات مائية.` → `دولفين، قرش، سمكة البيرانا.` — *EN: "Name three aquatic animals." → "Dolphin, shark, piranha."* (note the instruction is itself a translationese noun phrase, `اسم` "name/noun", where idiomatic Arabic would be `اذكر`)
> **Native-vs-translated verdict:** **translated prompts, GPT-4 Arabic answers.** | **أسئلة مترجمة وإجابات GPT-4 بالعربية.**
> **Deep-offset note:** Evol-Instruct shows MT fusion damage from p=0 onward: `الحصول على قسط كافٍ من النوم كل ليلة، ideal7-8 ساعات بشكل مثالي` — the English word *ideal* is fused to the digits and the adverb is then translated a second time (`بشكل مثالي`). Raw `<html>` blocks also appear in prompts. | عطب دمج ترجمي من الصفحة صفر: كلمة إنجليزية ملتصقة بالأرقام مع تكرار الظرف مترجمًا، وكتل `<html>` خام في الأسئلة.

**8. saudi-dialect-conversations — `HeshamHaroon/saudi-dialect-conversations` — 3,545 rows — Tier: High per-row, negligible scale / عالية جدًا لكن حجمها ضئيل.**
Sampled at 0 / 2K / 3.4K. Genuinely idiomatic Najdi multi-turn dialogue (22.5K turns) with scenario/topic metadata. Per-row quality is the best dialectal Arabic I saw; there is just very little of it.
| حوار نجدي اصطلاحي متعدد الأدوار مع بيانات وصفية للسيناريو والموضوع. جودة الصف الواحد هي الأفضل لهجيًا فيما عايناه، لكن الكمّ ضئيل.
> `هلا والله يا أبو ناصر، تدري أنا أبي أسجل حقوق الملكية؟` — *EN: "Hey there Abu Nasir — you know I want to register property rights?"*
> **Native-vs-translated verdict:** **synthetic but natively-composed Saudi Arabic** (not translated). | **مولَّدة لكنها مؤلَّفة بالسعودية أصلًا، غير مترجمة.**
> **Deep-offset note:** Uniform across all 3.5K; register never slips into MSA. | متجانسة عبر كامل المجموعة ولا تنزلق إلى الفصحى.

**9. Bactrian-X (`ar` config) — `MBZUAI/Bactrian-X` — 67,017 rows — Tier: Medium / متوسطة.**
Sampled at 0 / 2K / 20K / 60K. Holds up better than its reputation: the ChatGPT answers are fluent, list-structured Arabic. The weakness is upstream — the *instructions* are Google-Translate output of Alpaca+Dolly, so the task distribution is 2023-English.
| أفضل من سمعتها: إجابات ChatGPT عربية سلسة ومنظَّمة بقوائم. الضعف في المنبع — التعليمات مخرجات Google Translate لـ Alpaca+Dolly، فالتوزيع المهامّي إنجليزي 2023.
> `ما هي طرق استخدام الملعقة؟` → `تستخدم الملعقة في العديد من الأغراض… لتقديم الطعام… للخلط…` — *EN: "What are the uses of a spoon?"*
> **Native-vs-translated verdict:** **translated instructions + ChatGPT-generated Arabic responses.** **CC-BY-NC-4.0 — non-commercial.** | **تعليمات مترجمة وإجابات مولَّدة بـ ChatGPT؛ رخصة غير تجارية.**
> **Deep-offset note:** Stable to 60K; no structural corruption. | ثابتة حتى 60 ألفًا دون فساد بنيوي.

**10. Arabic_Reasoning_Dataset — `Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset` — 9,210 rows — Tier: Medium-High (tiny, narrow) / متوسطة-عالية (صغيرة وضيقة).**
Sampled at 0 / 2K / 9K. Clean, consistently formatted Arabic chain-of-thought (`المعطيات` → `الخطوات` → conclusion). Word problems only; no proofs, no code.
| سلاسل استدلال عربية نظيفة وثابتة التنسيق (المعطيات ← الخطوات ← النتيجة). مسائل لفظية فقط، بلا براهين ولا برمجة.
> `المعطيات: عدد الأشجار = 12 … الخطوات: 12 × 5 = 60 … إذن، بقيت 30 تفاحة` — *EN: "Given: 12 trees … Steps: 12 × 5 = 60 … So 30 apples remain."*
> **Native-vs-translated verdict:** **reads native**; provenance is undocumented on the card. | **تقرأ كأنها أصلية**، لكن مصدرها غير موثَّق في البطاقة.
> **Deep-offset note:** Very high template repetition — the same three problem schemas recur across offsets. Do not over-weight it. | تكرار قوالبي عالٍ جدًا؛ لا تُعطِها وزنًا زائدًا.

**11. InstAr-500k — `ClusterlabAi/InstAr-500k` — 481,281 rows — Tier: Medium-Low / متوسطة-منخفضة.**
Sampled at 0 / 2K / 20K / 120K / 250K / 400K / 470K — the widest offset sweep here, and it changed the verdict. This is **not an assistant dataset**; it is a FLAN-style repackaging of existing Arabic NLP corpora (XTREME/TyDiQA extractive QA, aya_collection, xlel_wd, SANAD news categorization, ArabicaQA, and fatwa pages mined from the 101B web corpus) behind one **identical English system prompt repeated on every row**.
| ليست مجموعة مساعِد، بل إعادة تغليف بأسلوب FLAN لمدونات عربية قائمة، خلف **موجّه نظام إنجليزي واحد مكرَّر على كل صف**.
> System (every row, verbatim): `You are a helpful AI assistant specialized in providing answers exclusively in Arabic…` — an English system prompt on 481K Arabic rows.
> **Native-vs-translated verdict:** **mostly native Arabic source text**, but the `type` labels are misleading — rows tagged `human-crafted` are simply *sourced from* human-written corpora, not human-written instructions. | **نصوصها المصدرية عربية أصيلة غالبًا**، لكن وسم `human-crafted` مضلِّل.
> **Deep-offset note:** Severe and offset-dependent. Offsets 0–2K are one- or two-word extractive answers. Offset 20K exposes broken `aya_collection` rows where the instruction asks for "an example of general info in this category: **geography**" (English category token) and the answer is `كم من الوقت استغرق بناء نصب لينكولن التذكاري؟ **8 years**` — an English answer in an Arabic row. Offset 250K is 100% news classification (`الفئة التي ينتمي إليها هذا الخبر هي الرياضة`). Offset 470K is Salafi fatwa text. **Answer-length distribution is pathological for chat SFT.** | تدهور حادّ ومرتبط بالإزاحة: إجابات من كلمة واحدة، ورموز إنجليزية في الأسئلة وإجابات إنجليزية، وتصنيف أخبار خالص، ونصوص فتاوى — توزيع أطوال غير صالح لضبط المحادثة.

**12. ultrafeedback-arabic — `alielfilali01/ultrafeedback-arabic` — 63,135 rows — Tier: Medium (preference) / متوسطة (تفضيل).**
Sampled at 0 / 2K / 20K / 60K. Largest MSA preference set. Fluent, and the chosen/rejected gap is often real. But it inherits UltraFeedback's dated failure mode: `rejected` responses are full of `باعتباري نموذجًا للغة الذكاء الاصطناعي…` ("As an AI language model…") — training against that teaches a 2023 refusal style, not modern judgement.
| أكبر مجموعة تفضيل بالفصحى، وسلسة، والفارق بين المقبول والمرفوض حقيقي غالبًا، لكنها ترث عطب UltraFeedback: المرفوض مليء بعبارة «باعتباري نموذجًا للغة الذكاء الاصطناعي…».
> **Native-vs-translated verdict:** **fully translated** from UltraFeedback-binarized. No license on the card. | **مترجمة بالكامل**، ولا رخصة على البطاقة.
> **Deep-offset note:** Prompt topics stay English-developer-centric at depth (React apps, CSV export). | تبقى الموضوعات موجَّهة لمطوّري الغرب في الأعماق.

**13. Arabic-OpenHermes-2.5 — `2A2I/Arabic-OpenHermes-2.5` — 981,618 rows — Tier: Low (do not use raw) / منخفضة (لا تُستخدم خامًا).**
Sampled at 0 / 2K / 20K / 200K / 600K / 950K. Widely cited as "the big Arabic chat set." Prose rows are acceptable, but the dataset has **three disqualifying structural defects**, all confirmed at depth:
| تُستشهد كثيرًا بوصفها «مجموعة المحادثة العربية الكبرى»، وصفوفها النثرية مقبولة، لكن فيها **ثلاثة أعطاب بنيوية مُسقِطة**، جميعها مؤكَّدة في الأعماق:
> (a) **Code was translated.** A Go program at offset 20K: ```` ```اذهب / الحزمة الرئيسية / يستورد ("إف إم تي") / لأني := 0; أنا <ن-1؛ أنا ++ ```` — the fence tag `go` became the verb *go*, `package main` was translated, `fmt` transliterated to Arabic letters, and the loop variable `i` became `أنا` ("I/me"). Python fences become ```` ```بيثون ````. **Every code row in this dataset is broken.** | **الشيفرة تُرجمت**: أسماء الحزم والمعرِّفات والكلمات المفتاحية صارت عربية، والمتغير `i` صار «أنا». كل صفوف البرمجة معطوبة.
> (b) **Roles are swapped at depth.** At offset 600K the `user` field holds the system persona (`أنت مساعد الذكاء الاصطناعي الذي يتبع التعليمات…`) while the `gpt` field holds the actual question. | **تبادل الأدوار في الأعماق**: حقل المستخدم يحمل شخصية النظام وحقل النموذج يحمل السؤال.
> (c) **Partial translation of MCQ options:** `أ. … ب- … ج- … د- … E. لا شيء مما سبق.` — option E left in Latin. | ترجمة جزئية لخيارات الاختيار المتعدد.
> **Native-vs-translated verdict:** **fully machine-translated, unfiltered.** The card documents no translation system and no QC. | **ترجمة آلية كاملة بلا ترشيح**، والبطاقة لا تذكر نظام الترجمة ولا ضبط الجودة.

**14. Aya Collection — Arabic configs of `CohereLabs/aya_collection_language_split` — standard 6,646,024 / egyptian 4,120,671 / moroccan 4,146,308 / mesopotamian 4,120,142 / najdi 4,120,278 / N-levantine 4,120,142 / S-levantine 4,120,223 / ta'izzi-adeni 4,120,271 / tunisian 4,120,142 / algerian 6,046 (≈39.8M) — Tier: Low as SFT / منخفضة كبيانات ضبط.**
Sampled `standard_arabic`, `egyptian_arabic`, `moroccan_arabic` at 0 / 20K / 2M. Enormous, Apache-2.0, and the largest Arabic instruction pool that exists — and largely unusable as chat SFT.
| ضخمة ومرخّصة Apache-2.0 وأكبر تجمّع تعليمات عربي موجود — وغير صالحة إلى حدّ بعيد كبيانات محادثة.
> **The seven ~4.12M dialect configs are the same English source fanned out by MT**, not independent dialect collection — they land within ~200 rows of each other and share `id` alignment. Row `id 1` is literally the same CoQA passage in MSA, Egyptian and Moroccan.
> | **التهجئات السبع (~4.12 مليون لكلٍّ) هي المصدر الإنجليزي نفسه موزَّعًا آليًا** لا جمعًا لهجيًا مستقلًا؛ الصف رقم 1 هو النص ذاته بالفصحى والمصرية والمغربية.
> Concrete MT damage, standard_arabic p=0: `484<unk>425 قبل الميلاد` — a literal `<unk>` token from the translation model in place of an en-dash; enumerators corrupt into words (`خمسة.` for "5.", `ثامنًا:` for "8."), and in Egyptian `١١ سنة` ("11 years") replaces the list marker "11.".
> | عطب ترجمي ملموس: رمز `<unk>` من نموذج الترجمة داخل النص، وأرقام التعداد تتحول إلى كلمات وسنوات.
> The Moroccan config mixes fully-diacritized Darija and plain MSA **inside the same passage** (`مَا عَنْدِيش حَقّْ نْقْعَدْ هنا` next to `الكابتن تريمبليت، اللي كان في وضعية مماثلة`).
> | التهيئة المغربية تخلط دارجة مشكولة بالكامل مع فصحى **داخل الفقرة الواحدة**.
> **Native-vs-translated verdict:** **overwhelmingly machine-translated + templated.** The genuinely human-written Arabic lives only in the separate `CohereLabs/aya_dataset` (13,960 Arabic rows: ary 8,090 · arb 4,995 · arz 529 · ars 136 · acq 129 · apc 81). | **مترجمة آليًا وقالبية في معظمها**؛ العربية المكتوبة بشريًا موجودة فقط في `aya_dataset` (13,960 صفًا).

**15. riotu-lab/ArabicQA_2.1M — 2,141,146 rows — Tier: Low / منخفضة.**
Sampled at 0 / 20K / 1M / 2.1M. Aggregated Arabic QA at real scale, but **systematically field-misaligned**: the `question` column repeatedly contains a *system prompt* instead of a question.
| تجميع أسئلة وأجوبة عربية بحجم حقيقي لكنه **مختلّ الحقول منهجيًا**: عمود السؤال يحوي موجّه نظام بدل السؤال.
> p=0 `question`: `أنت خبير في الذكاء الاصطناعي على مستوى عالمي - قدم إجابات دقيقة وموجزة.` (*"You are a world-class AI expert — give accurate, concise answers."*) — with an unrelated answer about the Pharos lighthouse.
> p=1,000,000 `question`: `أنت مساعد الذكاء الاصطناعي. سيتم تكليفك بمهمة…` with a translated SAMSum **dialogue** as the answer.
> Answers are also truncated mid-string (` كيلومتر ما يجعل أبراج البيت…` starts mid-sentence).
> **Native-vs-translated verdict:** **mixed native + translated**, undisclosed per-row. | **مختلطة أصيلة ومترجمة** دون بيان لكل صف.

**16. six_millions_instruction_dataset_for_arabic_llm_ft — `akbargherbal/…` — 6,372,734 rows — Tier: Low / منخفضة.**
Sampled at 0 / 2K / 20K / 3M / 6M. The largest raw Arabic instruction row count outside Aya, with an **empty dataset card, no license, no provenance**.
| أكبر عدد صفوف تعليمات عربية خام خارج Aya، مع **بطاقة فارغة ولا رخصة ولا بيان مصدر**.
> Encoding damage at p=2K — deprecated Arabic **presentation-form** codepoints instead of normal letters: `ﻻقت تأييدا … ١٨ كانون اﻷول/ديسمبر ١٩٩٢` (UN parallel-corpus text; `ﻻ` U+FEFB and `ﻷ` U+FEF7 instead of `لا` / `لأ`). Any tokenizer will treat these as distinct tokens.
> | عطب ترميز: استخدام **صور العرض** المهجورة بدل الحروف العادية، وهو ما يفسده لأي مُجزِّئ.
> English leakage in the *instruction* column at p=3M: `Complete the following phrase:` followed by an Arabic fragment, with a raw text continuation as the "output".
> | تسرّب إنجليزي في عمود التعليمة عند 3 ملايين، والمخرَج مجرد استكمال نصّ خام.
> **Native-vs-translated verdict:** **translated/scraped mixture, undocumented.** | **خليط مترجَم ومكشوط غير موثَّق.**

**17. saudi-allam-sft-dataset-2M — `MohAlbrayh/saudi-allam-sft-dataset-2M` — **4,504 rows** — Tier: Low / منخفضة.**
Sampled at 0 / 2K. **The name claims 2M; the dataset has 4,504 rows.** Every row repeats the same long Arabic system persona; `category` values include `Alpca_translated` (sic). Dialect labels (Hijazi/Najdi) are present but the underlying content is translated Alpaca.
| **الاسم يدّعي مليونين والواقع 4,504 صفوف.** كل صف يكرر الشخصية النظامية ذاتها، وفئاته تشمل «ألباكا مترجمة». الوسوم اللهجية موجودة لكن المحتوى مترجَم.
> **Native-vs-translated verdict:** **translated.** No license on the card. Treat the repo name as unreliable. | **مترجمة**، ولا رخصة، والاسم غير موثوق.

**Also sampled, ranked separately because they are not chat SFT / عُوينت أيضًا وصُنِّفت على حدة لأنها ليست بيانات محادثة:**
- **`FreedomIntelligence/AceGPT-v2-AlignmentData` — 3,222,135 rows, Apache-2.0.** `origin` → `rewritten` pairs (GPT-4-turbo cleaning of ArabicText-2022 passages). Useful to *train a data-cleaner*, not to train a chat model. Sampled 2K/20K/200K: the rewrites are mostly good punctuation/orthography fixes, but they introduce errors (`أؤلؤ` for `أولئك`, `وحياتًا` nonsense) and **at p=200K the meta-prompt scaffold leaks into the target field**: `Arabic text: … Analysis: … Rewritten text: …`. | أزواج «أصل ← معاد صياغته»؛ مفيدة لتدريب منظِّف بيانات لا لتدريب نموذج محادثة، ويتسرّب هيكل الموجّه إلى حقل الهدف في الأعماق.
- **`HeshamHaroon/Arabic_Function_Calling` — 50,810 rows, Apache-2.0.** Synthetic Arabic tool-calling with explicit `dialect` labels (MSA / Levantine / Gulf / Egyptian) and paired `query_ar`/`query_en`. Clean and well-structured; the only ungated Arabic function-calling set of size. | استدعاء أدوات عربي مولَّد بوسوم لهجية صريحة؛ نظيف ومنظَّم، وهو الوحيد غير المقيَّد بهذا الحجم.
- **`MBZUAI-Paris/Egyptian-DPO-Mixture` — 298,073 rows.** On-policy + off-policy Egyptian preference pairs. Idiomatic (`يا جدعان ازيكم، النهارده هنتكلم عن اللحمة`) and the chosen/rejected distinction is stylistic-and-real. Best-documented Arabic DPO set. No license on the card. | أزواج تفضيل مصرية اصطلاحية والفارق فيها أسلوبي حقيقي؛ أفضل مجموعة DPO عربية توثيقًا، لكن بلا رخصة.
- **`FreedomIntelligence/Arabic-preference-data-RLHF` — 11,548 rows.** AceGPT's RLAIF pairs over **natively Arabic instructions**; both branches are plausible, so the signal is fine-grained. **README returns "Entry not found" — no card, no license.** | أزواج AceGPT فوق تعليمات عربية أصيلة؛ الإشارة دقيقة لكن لا بطاقة ولا رخصة.
- **`2A2I/argilla-dpo-mix-7k-arabic` — 7,500 rows, MIT.** **Structurally broken:** the translator translated the JSON *role* values themselves — rows carry `"role": "مستخدم"` instead of `"user"`, which will silently break every standard chat template. | **معطوبة بنيويًا**: تُرجمت قيم الأدوار نفسها (`"مستخدم"` بدل `"user"`)، ما يكسر قوالب المحادثة بصمت.
- **`arcee-globe/arabic-orpo-dpo-mix-40k-filtered` — 31,846 rows, no license.** Translated ORPO mix; chosen and rejected are often near-identical paraphrases (`مألوفة بشكل غريب` vs `مألوفة بشكل مريب`), i.e. weak preference signal. | خليط ORPO مترجم، والفرق بين المقبول والمرفوض غالبًا مجرد مرادفات — إشارة تفضيل ضعيفة.
- **`Omartificial-Intelligence-Space/Arabic-Math-SFT` — 5,000 rows, Apache-2.0.** **Multimodal** (geometry figures + Arabic problem + bare `<answer>`), not text SFT, and answers carry no working. | **متعددة الوسائط** (أشكال هندسية) لا نصية، والإجابات بلا خطوات.

---

## 🎯 Recommendation / التوصية

**English.** For a Qwen-class model continued-pretrained on Arabic, do **not** build the SFT mix around the
"famous" large sets. `2A2I/Arabic-OpenHermes-2.5` and the Aya Collection Arabic configs are the two most
commonly reached-for options and both are structurally damaged (translated code identifiers and swapped
roles in the former; `<unk>` tokens, corrupted enumerators and fake per-dialect fan-out in the latter).
Build instead a **~600K–900K example mixture** with a quality-filtered translated backbone and a small,
non-negotiable native core:

1. **Backbone (~500K):** **SmolKalam** via `AdaMLLab/smolkalam-arabic-conversational-sft`, filtered to
   `LR ≥ 0.85` **and** `SCR ≥ 0.95` and deduplicated. Take the `no_think` configs for a standard chat model
   and add the `*_think` configs only if you want Arabic reasoning traces. This is the only large Arabic
   set that ships per-row quality signals; use them.
2. **Native prompt core (~55K, use in full, up-weight ×2):** `FreedomIntelligence/Quora-Arabic-GPT4`
   (43,050) + `arbml/CIDAR` (10,000). This is what fixes the prompt distribution — everything else teaches
   the model to answer *American* questions in Arabic.
3. **Breadth top-up (~110K):** `FreedomIntelligence/Alpaca-Arabic-GPT4` (49,969) +
   `Evol-Instruct-Arabic-GPT4` (69,997), **after dropping every row containing a code fence or an `<html>`
   tag** and deduping against SmolKalam.
4. **STEM (~100K sample):** a filtered slice of `hammh0a/Hala-4.6M-SFT` — *only if your project can accept
   CC-BY-NC-4.0*. Reject rows with unbalanced `\text{}`/braces or `<br>`. If you need a commercial license,
   substitute `Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset` (9,210) and accept the STEM gap.
5. **Dialect (~100K, only if dialectal ability is a goal):** stratified sample of
   `MBZUAI-Paris/Egyptian-SFT-Mixture` (native-labelled rows first) + `Darija-SFT-Mixture` +
   all 3,545 rows of `HeshamHaroon/saudi-dialect-conversations`. Keep MSA:dialect ≈ 85:15 unless the target
   is explicitly a dialect model — the dialect mixtures are task-heavy and will otherwise flatten chat style.
6. **Tool use (~20K, optional):** `HeshamHaroon/Arabic_Function_Calling` (50,810) subsampled.
7. **Replay (~10%):** English/Chinese/Hindi SFT from your existing mixes, to prevent the Arabic SFT from
   eroding the other CPT languages.
8. **Preference stage (optional):** `alielfilali01/ultrafeedback-arabic` (63,135) for MSA breadth **after
   stripping "As an AI language model" rejected branches**, plus `FreedomIntelligence/Arabic-preference-data-RLHF`
   (11,548) for native-prompt signal, plus `MBZUAI-Paris/Egyptian-DPO-Mixture` (298,073) if dialect matters.
   Do **not** use `2A2I/argilla-dpo-mix-7k-arabic` without first rewriting `"مستخدم"` → `"user"`.

**Evaluate, don't train on:** `UBC-NLP/palm` (17,411 human-authored items from 44 native speakers in 22
countries, 10 dialects) is the right instrument for measuring whether your Arabic SFT actually landed
culturally — Fanar 2.0 uses it for exactly that. Its **CC-BY-NC-ND** licence forbids derivatives, so keep it
out of the training mixture. Pair it with `FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment` (8,000+
items, 2,486 native-validated) and `silma-ai/arabic-broad-benchmark` (470 human-validated questions across
22 categories). | **قيِّم ولا تدرِّب**: استخدم Palm وACVA وarabic-broad-benchmark للقياس؛ ورخصة Palm تمنع
إدراجها في خلطة التدريب.

**Two lessons to internalize from the closed reports / درسان من التقارير المغلقة:** (i) Fanar 2.0 —
*"translating existing English-distilled reasoning datasets into Arabic introduced language-mixing artifacts
and degraded trace quality; instead, we generated reasoning traces natively in Arabic"* — so if you need
Arabic reasoning, prefer SmolKalam's `*_think` configs (already quality-ranked) or generate traces natively
with your own model rather than translating an English CoT set; (ii) Fanar 1.0's fix for English-name leakage
(regenerate responses for prompts containing the 100 most common English given names — 30K rows was enough)
is a cheap, directly reproducible post-processing step on any translated mixture you adopt.
| **الدرسان:** (1) وَلِّد آثار الاستدلال بالعربية أصلًا بدل ترجمتها؛ (2) أعد توليد الإجابات التي تحوي أشيع
الأسماء الإنجليزية — 30 ألف صفّ كفت Fanar لإزالة التسرّب، وهي خطوة رخيصة قابلة للتكرار على أي خلطة مترجمة.

**Hard exclusions:** `ClusterlabAi/InstAr-500k` (FLAN task dump behind an English system prompt; answer
lengths are pathological for chat), `akbargherbal/six_millions…` (presentation-form encoding damage, no
license), `riotu-lab/ArabicQA_2.1M` (system prompts in the question column), `MohAlbrayh/saudi-allam-sft-dataset-2M`
(4,504 rows, not 2M), and raw `2A2I/Arabic-OpenHermes-2.5` / raw Aya Collection Arabic configs.
**Licensing:** if the artefact must be commercially usable, drop `hammh0a/Hala-4.6M-SFT` and
`MBZUAI/Bactrian-X` (both CC-BY-NC-4.0); note that CIDAR's HF card says Apache-2.0 while its paper says
CC BY-NC, and that everything GPT-4/ChatGPT-distilled (CIDAR, all FreedomIntelligence sets, Bactrian-X)
carries OpenAI-terms exposure regardless of the tag.

**العربية.** لنموذج من عائلة Qwen خضع لتدريب مسبق متواصل على العربية، **لا** تَبْنِ خلطة الضبط حول
المجموعات «المشهورة» الكبيرة: فـ `Arabic-OpenHermes-2.5` وتهيئات Aya العربية هما الخياران الأكثر تداولًا
وكلاهما معطوب بنيويًا (شيفرة مترجمة وأدوار متبادلة في الأولى، ورموز `<unk>` وأرقام تعداد فاسدة وتوزيع
لهجي زائف في الثانية). ابْنِ بدلًا من ذلك خلطة من **600–900 ألف مثال**: عمودها الفقري ~500 ألف من
**SmolKalam** بعد ترشيحها بـ `LR ≥ 0.85` و`SCR ≥ 0.95`؛ ونواة أصيلة غير قابلة للتفاوض ~55 ألفًا هي
`Quora-Arabic-GPT4` (43,050) و`CIDAR` (10,000) بوزن مضاعف — فهذه وحدها ما يصحّح **توزيع الأسئلة**، إذ
تُعلِّم البقيةُ النموذجَ أن يجيب عن أسئلة أمريكية بالعربية؛ ثم ~110 آلاف توسعة من
`Alpaca-Arabic-GPT4` و`Evol-Instruct-Arabic-GPT4` **بعد حذف كل صف يحوي إطار شيفرة أو وسم `<html>`**؛
و~100 ألف علوم ورياضيات من `Hala-4.6M-SFT` **إن قبِل مشروعُك رخصةً غير تجارية**؛ و~100 ألف لهجية (مصري
ودارجة وسعودي) بنسبة فصحى:لهجة ≈ 85:15؛ و~20 ألفًا لاستدعاء الأدوات؛ و~10% إعادة تشغيل بالإنجليزية
والصينية والهندية لمنع تآكل اللغات الأخرى. وفي مرحلة التفضيل: `ultrafeedback-arabic` بعد إزالة فروع
«باعتباري نموذجًا للغة الذكاء الاصطناعي»، مع `Arabic-preference-data-RLHF`، و`Egyptian-DPO-Mixture`
عند الحاجة للهجة. **استبعادات قاطعة:** `InstAr-500k` و`six_millions` و`ArabicQA_2.1M`
و`saudi-allam-sft-dataset-2M` و`Arabic-OpenHermes-2.5` الخام وتهيئات Aya العربية الخام. **رخصيًا:** إن
لزم الاستخدام التجاري فاستبعد `Hala-4.6M-SFT` و`Bactrian-X` (كلتاهما CC-BY-NC-4.0)، وانتبه إلى تعارض
رخصة CIDAR بين البطاقة والبحث، وإلى تبعات شروط OpenAI لكل ما قُطِّر من GPT-4.

---

## 🗺️ Taxonomy / خريطة المجال

- **(A) Human-authored / human-reviewed Arabic instruction data / بيانات مكتوبة أو مراجَعة بشريًا.**
  The scarcest category. `CohereLabs/aya_dataset` Arabic (**13,960**), `arbml/CIDAR` (**10,000**, MT-seeded
  but fully human-edited), `UBC-NLP/palm` (**17,411**, 44 native annotators / 22 countries / 10 dialects —
  but gated and CC-BY-NC-**ND**), `ArSyra` (11,719, gated + paid), and inside closed mixtures, Jais's
  NativeQA-Ar + NER-Ar (**~17,000**). **Total genuinely human-authored Arabic instruction data that is
  freely downloadable and redistributable is on the order of 2–3 × 10⁴ examples — four orders of magnitude
  below what the models are actually trained on.** | أندر الفئات؛ المجموع القابل للتنزيل وإعادة النشر في
  حدود 20–30 ألف مثال فقط — أي أقل بأربع مراتب عشرية مما تُدرَّب عليه النماذج فعلًا.
- **(B) Native-prompt + distilled-answer / أسئلة أصيلة وإجابات مقطَّرة.** The AceGPT recipe: harvest real
  Arabic questions, answer with a strong teacher. `Quora-Arabic-GPT4` (43,050),
  `Arabic-preference-data-RLHF` (11,548), `saudi-dialect-conversations` (3,545).
- **(C) Machine-translated English SFT, unfiltered / ترجمة آلية بلا ترشيح.** The bulk of the ecosystem.
  `Arabic-OpenHermes-2.5`, `Bactrian-X ar`, `Alpaca-Arabic-GPT4`, `Evol-Instruct-Arabic-GPT4`,
  `arbml/alpaca_arabic`, `arbml/okapi_arabic`, `HeshamHaroon/oasst-arabic`, `2A2I/H4_no_robots`,
  `ultrafeedback-arabic`, `argilla-dpo-mix-7k-arabic`, `six_millions…`.
- **(D) Machine-translated **with** quality filtering / ترجمة آلية مع ترشيح جودة.** The 2025–2026 shift, and
  the most promising direction. **SmolKalam** (ensemble MT + reward-model ranking + LR/SCR filters),
  **Hala-4.6M-SFT** (purpose-trained instruction translator, code rows dropped), *Creating Arabic LLM
  Prompts at Scale* (COMET-QE ≥ 0.7, which **discards ~80%** of translated prompts).
- **(E) Templated repackaging of existing Arabic NLP datasets / إعادة تغليف قالبية لمدونات عربية قائمة.**
  `InstAr-500k`, the Aya Collection templated portion, `riotu-lab/ArabicQA_2.1M`, most of `Darija-SFT-Mixture`.
  Cheap, native-sourced, but the answer-length distribution is wrong for chat.
- **(F) Dialectal SFT mixtures / خلطات لهجية.** `Darija-SFT-Mixture` (458,285, Moroccan),
  `Egyptian-SFT-Mixture` (1,817,288, Egyptian incl. Arabizi), `saudi-dialect-conversations` (3,545, Najdi),
  `Egyptian-DPO-Mixture` (298,073).
- **(G) Multilingual mixtures with an Arabic share / خلطات متعددة اللغات ذات نصيب عربي.**
  Aya Collection / Aya Dataset, Bactrian-X, `2A2I/Arabic_Aya` (41,472,592 — an Arabic-only re-slice of Aya),
  `multilingual/orca_dpo_pairs`, `neulab/PangeaInstruct` (multimodal). **Note: no Arabic Magpie variant
  exists** — verified across the whole `magpie`-matching set on the Hub. | **لا توجد نسخة عربية من Magpie.**
- **(H) Reasoning / math / code / tool-calling / استدلال ورياضيات وبرمجة وأدوات.** The thinnest domain.
  `Arabic_Reasoning_Dataset` (9,210), `Arabic-Math-SFT` (5,000, multimodal), `Arabic_Function_Calling`
  (50,810), `TuwaiqAcademy/AISA-ArabicFC` (12,220), the SmolKalam `*_think` / OpenThoughts3 configs, and the
  unreleased Arabic tool-calling sets of arXiv 2509.20957.
- **(I) Closed / unreleased SFT mixtures behind Arabic models / خلطات مغلقة خلف النماذج العربية.**
  Jais-chat, ALLaM, Fanar, PHOENIX, NOON, SILMA, Yehia — see §6.

---

## §1 — General-purpose Arabic instruction datasets / مجموعات التعليمات العربية عامة الغرض

### CIDAR: Culturally Relevant Instruction Dataset For Arabic (Alyafeai et al., 2024)
- **Venue / Link:** Findings of ACL 2024, [arXiv:2402.03177](https://arxiv.org/abs/2402.03177) · [ACL Anthology](https://aclanthology.org/2024.findings-acl.764/)
- **Data / Code:** https://huggingface.co/datasets/arbml/CIDAR · https://github.com/ARBML/CIDAR (companions `arbml/CIDAR-EVAL-100`, `arbml/CIDAR-MCQ-100`, 100 rows each)
- **Size / الحجم:** **10,000** instruction–output pairs (verified via `/size`).
- **Construction / البناء:** 9,109 pairs sampled from **AlpaGasus** (quality-filtered Alpaca) and machine-translated with **gpt-3.5-turbo** via the `Taqyim` library, plus **891** natively written Arabic-grammar items scraped from Al Jazeera's *Ask the Teacher*. All 10,000 were then reviewed in a purpose-built annotation tool by **~12 contributors** doing linguistic correction **and** cultural localization. **~64.5% of the translated pairs required modification.**
- **MSA vs dialect / الفصحى واللهجات:** MSA only.
- **Motivation / الدافع:** Machine-translated instruction data imports Western names, places and norms into Arabic, producing "distorted and misaligned instructions"; CIDAR replaces John/Mary with Muhammad/Sarah and US locations with Arab countries. | تستورد البيانات المترجمة أسماءً وأماكن ومعايير غربية إلى العربية، فتنتج تعليمات مشوَّهة وغير متوائمة؛ وتستبدل CIDAR ذلك بأسماء وأماكن عربية.
- **License / الرخصة:** **Conflicting** — HF `cardData` says `apache-2.0`; the paper's appendix reports **CC BY-NC**; the card also says "research purposes only". gpt-3.5-derived → OpenAI-terms exposure. **Resolve with the authors before commercial use.** | **متعارضة** بين البطاقة والبحث؛ احسم الأمر مع المؤلفين قبل الاستخدام التجاري.
- **Quality from sampling / الجودة من المعاينة:** See rank #3. Consistently clean across 0/2K/9K; unique Arabic-intrinsic tasks (إعراب, diacritized poetry); answers short. | نظيفة باطراد، ومهام عربية أصيلة فريدة، لكن الإجابات قصيرة.
- **Caveat / تحفّظ:** The paper's evidence is **qualitative only** — it reports the 64.5% edit rate and side-by-side examples but **no GPT-4 win rates or benchmark deltas** between a CIDAR-tuned and an AlpaGasus-tuned model. The "localized beats translated" claim is not numerically established there. | أدلة البحث **نوعية فقط** ولا تتضمن أرقام مقارنة، فدعوى تفوّق التوطين على الترجمة غير مثبتة عدديًا فيه.

### Palm — culturally native Arabic instruction data (UBC-NLP, 2025)
- **Venue / Link:** [arXiv:2503.00151](https://arxiv.org/abs/2503.00151)
- **Data:** https://huggingface.co/datasets/UBC-NLP/palm — **gated (`auto`)**, so it could not be sampled here.
- **Size:** **17,411** total = **15,485 train + 1,926 test** (verified from the repo's `dataset_info`; the paper reports 13,559 train / 1,926 public test / 1,926 private test).
- **Construction:** **fully human-authored** by **44 native-speaker annotators/co-authors** across **22 Arab countries**, covering **10 dialects** and **20 topics**. Fields: `country`, `topic`, `language_variety`, `instruction`, `output`, `question_type`.
- **Motivation / الدافع:** the paper's framing example is the sharpest in the literature — *"an Arabic LLM pre-trained on English-to-Arabic translated data **suggested having a beer after prayer**, a recommendation that starkly contradicts Arab cultural values, religious practices, and social norms."* | مثالها الافتتاحي هو الأحدّ في الأدبيات: نموذج عربي مدرَّب على بيانات مترجمة **اقترح شرب الجعة بعد الصلاة**.
- **License:** **CC-BY-NC-ND-4.0** — non-commercial **and no derivatives**, which rules out mixing it into a training set you redistribute. Use it for evaluation and for calibrating cultural alignment. | **غير تجارية وبلا اشتقاق**، فلا تصلح للدمج في خلطة تُعاد نشرها؛ استخدمها للتقييم ومعايرة التوافق الثقافي.
- **Why it matters here / لماذا تهمّ:** Fanar 2.0 uses `UBC-NLP/palm` for its cultural-alignment stage — an independent signal that this is the reference native-Arabic cultural set. Together with the Aya Dataset's 13,960 Arabic rows and CIDAR's 10,000, it is one of only three sizeable human-authored Arabic instruction resources in existence. | تستخدمها Fanar 2.0 في مرحلة التوافق الثقافي؛ وهي إحدى ثلاث مجموعات عربية بشرية التأليف فقط.

### InstAr-500k / LlamAr & GemmAr (Chouikhi et al., 2024)
- **Venue / Link:** [arXiv:2407.02147](https://arxiv.org/abs/2407.02147) (v1 "LlamAr & GemmAr"; v2 retitled "GemmAr" and drops LlamAr)
- **Data:** https://huggingface.co/datasets/ClusterlabAi/InstAr-500k
- **Size:** **481,281** rows (`/size`); the paper states 481,670. Card split: ~291,776 "generated" / ~189,224 "human-crafted".
- **Construction:** Hybrid — 22 existing Arabic datasets reformatted into instructions (paper's table: Arabic_Categorization 205,540 · Aya_Collection 69,068 · ArabicaQA 61,945 · Classical Arabic Poetry 42,650 · CIDAR 19,986 · AQAD 17,322 · ~16 smaller sets), **plus synthetic instruction–response pairs generated by Cohere Command R+** grounded on cleaned text from the same group's 101-Billion-Arabic-Words corpus ([arXiv:2405.01590](https://arxiv.org/abs/2405.01590), 33,059,988 documents).
- **MSA vs dialect:** MSA only — the paper explicitly states the dataset "lacks dialectal variations."
- **Motivation / الدافع:** Build a large Arabic instruction resource from native Arabic text instead of translating English. | بناء مورد تعليمات عربي كبير من نصّ عربي أصيل بدل ترجمة الإنجليزية.
- **License:** Apache-2.0 (HF) / CC BY 4.0 (paper). **Unaddressed conflict:** Command R+ outputs carry upstream C4AI non-commercial acceptable-use obligations that the Apache-2.0 tag does not cover. | **تعارض غير معالَج**: مخرجات Command R+ تحمل التزامات غير تجارية لا يغطيها وسم Apache-2.0.
- **Quality from sampling / الجودة من المعاينة:** See rank #11. The construction is native-sourced, but the *product* is a FLAN-style task dump behind one repeated **English** system prompt, with extractive one-word answers at shallow offsets, a broken `aya_collection` slice with English answers at 20K, pure news classification at 250K and fatwa text at 470K. Not suitable as a chat backbone. | المصدر عربي أصيل لكن الناتج تفريغ مهامّي خلف موجّه نظام إنجليزي مكرَّر، بإجابات من كلمة واحدة وتسرّب إنجليزي وتصنيف أخبار وفتاوى — لا يصلح عمودًا فقريًا للمحادثة.
- **Version caution:** v1 and v2 of the paper report substantially different GemmAr scores (OALL avg **62.41** vs **47.27**). Cite v2, or flag the discrepancy. | تباين كبير بين نسختي البحث في الدرجات؛ استشهد بالنسخة الثانية أو نبِّه على الاختلاف.

### Arabic-OpenHermes-2.5 (2A2I, 2024)
- **Data:** https://huggingface.co/datasets/2A2I/Arabic-OpenHermes-2.5 · by Marwa El Kamil & Mohammed Machrouh
- **Size:** **981,618** rows / 3.36 GB. (The card's `size_categories` tag says `100K<n<1M` — wrong-ish; trust `/size`.)
- **Construction:** Machine translation of `teknium/OpenHermes-2.5` (itself GPT-4-distilled). **The card documents neither the translation system nor any quality control.**
- **MSA vs dialect:** MSA.
- **Motivation / الدافع:** Give Arabic a large general chat/reasoning SFT set by porting the best-known English one. | منح العربية مجموعة محادثة واستدلال كبيرة بنقل أشهر نظيرتها الإنجليزية.
- **License:** apache-2.0 (but derived from GPT-4-distilled data).
- **Quality from sampling / الجودة من المعاينة:** See rank #13 — **the worst structural damage of any set sampled.** Go/Python source code was translated token by token (`الحزمة الرئيسية` for `package main`, `أنا` for the loop variable `i`, ```` ```اذهب ```` for ```` ```go ````); at offset 600K the `user` and `gpt` fields are **swapped**; MCQ options are only partly transliterated (`أ. ب- ج- د- E.`). Prose rows (roleplay, general QA, arithmetic word problems) are acceptable. **Usable only after dropping all code-tagged rows and repairing/removing the role-swapped region.** | **أسوأ عطب بنيوي بين ما عاينّاه**: شيفرة مترجمة رمزًا رمزًا، وأدوار متبادلة عند 600 ألف، وخيارات اختيار متعدد مترجمة جزئيًا. لا تُستخدم إلا بعد حذف صفوف البرمجة وإصلاح منطقة تبادل الأدوار.

### SmolKalam (Alrashed, Helwe & Orabona, 2025)
- **Venue / Link:** [arXiv:2511.18411](https://arxiv.org/abs/2511.18411) — *SmolKalam: Ensemble Quality-Filtered Translation at Scale for High Quality Arabic Post-Training Data* (marked work-in-progress)
- **Data:** https://huggingface.co/datasets/SultanR/smolkalam (**gated: auto**) · ungated mirror sampled here: https://huggingface.co/datasets/AdaMLLab/smolkalam-arabic-conversational-sft (**1,790,478** rows across **24 configs**, apache-2.0)
- **Size:** `SFT_SeedX_ranked` **1,777,275** examples / **3,261.63M tokens**; `SFT_Gemma3_ranked` **1,545,742** examples / **2,822.79M tokens**; 36.8 GB total. Config-level counts seen on the mirror: `OpenHermes_2.5_no_think` 384,900 · `OpenThoughts3_NoThink_180K` 180,000 · `Mixture_of_Thoughts_science_no_think` 86,110 · `OpenThoughts3_50K` 50,000 · `multi_turn_reasoning_if_think` 28,217 · `aya_dataset_Qwen3_32B_think` 15,222 · `hermes_function_calling_v1_no_think` 8,961 · `LongAlign_64k_*` 7,526 / 6,249 · `s1k_1.1_think` 835 (+ others).
- **Construction:** Translation of **SmolTalk2** by an **ensemble of SeedX-7B (local) + Gemma-3-27B (API)**; each sample gets N≥2 candidates ranked by a **Qwen-2.5-1.5B Bradley-Terry reward model** trained on S1K translation preferences with Arabic-MMLU as ground truth; then filtered on **Language Ratio (LR, target ≈0.90)** and **Script Purity (SCR, target ≥0.90)**. Reported aggregates: SeedX LR 0.796 / SCR 0.925; Gemma3 LR 0.808 / SCR 0.928.
- **MSA vs dialect:** MSA.
- **Motivation / الدافع:** Existing Arabic SFT sets are "either small or large but lack diversity," and many "consist of translations that have not been properly filtered"; the paper's thesis is that **"naive translation can work at the pretraining scale, but post-training demands much higher quality."** | المجموعات القائمة إمّا صغيرة أو كبيرة بلا تنوّع، وكثير منها ترجمات بلا ترشيح؛ وأطروحة البحث أن الترجمة الساذجة تنفع في التدريب المسبق لا في التدريب اللاحق.
- **License:** CC BY 4.0 (source repo); the mirror is tagged apache-2.0.
- **Quality from sampling / الجودة من المعاينة:** See rank #1 — the best Arabic SFT text sampled, with per-row `LR`/`SCR`/`rank_score` you can filter on. Caveats: sampled `LR` ranged 0.71–0.97, so **retained rows are not uniformly ≥0.90**; proper-noun calques survive; source repo is gated. | الأفضل نصًا، مع مؤشرات جودة لكل صف؛ لكن بعض الصفوف المحتفَظ بها دون العتبة، وتبقى ترجمات حرفية لأسماء الأعلام، والمستودع الأصلي مقيَّد.

### Hala-4.6M-SFT / Hala Technical Report (2025)
- **Venue / Link:** [arXiv:2509.14008](https://arxiv.org/abs/2509.14008) · ACL Anthology `2026.abjadnlp-1.32`
- **Data:** https://huggingface.co/datasets/hammh0a/Hala-4.6M-SFT
- **Size:** **4,060,575** rows.
- **Construction:** "Translate-and-tune" — an AR↔EN teacher is compressed to FP8 to generate bilingual data; **LFM2-1.2B** is fine-tuned into a dedicated instruction translator (**Hala-1.2B-EN-AR**) and used to convert English SFT at million scale. Sources: OpenOrca (810K), Hermes-3 (**code samples deliberately filtered out to avoid translation artifacts**), SCP-116K, ReAlign-Alpaca, Dahoas synthetic pairwise, LaMini-instruction, English subsets of `allenai/tulu-3-sft-mixture`. Models released at 350M/700M/1.2B/9B with slerp merging.
- **MSA vs dialect:** MSA.
- **Motivation / الدافع:** Get million-scale Arabic post-training data cheaply by making translation itself the tuned component rather than a generic MT system. | الحصول على بيانات تدريب لاحق بحجم الملايين بثمن زهيد عبر جعل الترجمة نفسها المكوَّن المضبوط بدل نظام ترجمة عام.
- **License:** **CC-BY-NC-4.0 — non-commercial.** | **غير تجارية.**
- **Quality from sampling / الجودة من المعاينة:** See rank #6 — the STEM coverage nothing else offers, LaTeX largely intact, but real deep-offset damage: mangled units with unbalanced braces at 20K, raw `<br>` at 2M, literal `\n` escapes in content. Dropping the code rows was the right call and is visible in the sample. | تغطية علمية لا نظير لها وLaTeX سليم غالبًا، لكن عطبًا حقيقيًا في الأعماق؛ وحذف صفوف البرمجة قرار صائب وظاهر في العينة.

### AceGPT instruction family — Alpaca / Evol-Instruct / ShareGPT / Quora (FreedomIntelligence)
- **Venue / Link:** AceGPT, [arXiv:2309.12053](https://arxiv.org/abs/2309.12053) (NAACL 2024 Findings); **AceGPT-v2 / Native Alignment**, [arXiv:2412.03253](https://arxiv.org/abs/2412.03253) (NeurIPS 2024); **AraLLaMA** (a *different* model, often confused with AceGPT-v2), [arXiv:2412.12310](https://arxiv.org/abs/2412.12310) (ACL 2025). See §6 for the citation corrections. | انظر §6 لتصحيحات الاستشهاد.
- **Paper-side mixture totals (AceGPT Table 15):** Quora-Arabic-40K 43,050 · Alpaca 49,969 · Alpaca-Chinese 49,969 · Alpaca-Arabic 49,969 · **Code-Alpaca-Arabic 20,022** · Evol-Instruct-Arabic 69,997 · **ShareGPT 80,179** — *"totaling **629,293** data"* after up-weighting the native Arabic sets. Note the paper's ShareGPT row (80,179) is **not** the released `sharegpt-arabic` (5,231). | مجموع البحث 629,293، وصفّ ShareGPT فيه غير المنشور على HF.
- **Data / sizes (all `/size`-verified, all apache-2.0 unless noted):**
  | Repo | Rows | What |
  |---|---:|---|
  | `FreedomIntelligence/Alpaca-Arabic-GPT4` (= `alpaca-gpt4-arabic`) | **49,969** | Alpaca-GPT4 translated, GPT-4 Arabic answers |
  | `FreedomIntelligence/Evol-Instruct-Arabic-GPT4` | **69,997** | Evol-Instruct (WizardLM-style), GPT-4 Arabic answers |
  | `FreedomIntelligence/evol-instruct-arabic` | **59,022** | earlier/smaller variant, **no license on card** |
  | `FreedomIntelligence/Quora-Arabic-GPT4` | **43,050** | **native Arabic Quora questions** + GPT-4 answers |
  | `FreedomIntelligence/sharegpt-arabic` | **5,231** | ShareGPT multi-turn, translated |
  | `FreedomIntelligence/Code-Alpaca-Arabic-GPT4` | — | code Alpaca, translated |
  | `FreedomIntelligence/Arabic-preference-data-RLHF` | **11,548** | RLAIF preference pairs, **no card / no license** |
  | `FreedomIntelligence/AceGPT-v2-AlignmentData` | **3,222,135** | `origin`→`rewritten` GPT-4-turbo cleaning pairs over ArabicText-2022 |
- **Motivation / الدافع:** AceGPT's thesis is that Arabic LLMs must be *localized*, not merely translated — hence native Arabic questions answered by GPT-4 in Arabic, plus RLAIF with a culture/value-aligned reward model. | أطروحة AceGPT أن النماذج العربية تحتاج **توطينًا** لا ترجمة؛ ومن ثمّ أسئلة عربية أصيلة تجيب عنها GPT-4 بالعربية، مع RLAIF بمكافأة متوائمة ثقافيًا.
- **Quality from sampling / الجودة من المعاينة:** Quora-Arabic-GPT4 is the standout (rank #2) — its prompt distribution is genuinely Arabic, including Egyptian-inflected student questions. Alpaca-Arabic is clean but shallow; Evol-Instruct-Arabic carries MT fusion damage (`ideal7-8 ساعات بشكل مثالي`) and raw `<html>` in prompts from p=0. sharegpt-arabic is fluent but tiny (5,231) and every row is tagged `lang: en`, confirming it is translated. AceGPT-v2-AlignmentData leaks its own meta-prompt (`Arabic text: … Analysis: … Rewritten text:`) into the target field at p=200K. | تتفوّق Quora-Arabic-GPT4 لأصالة أسئلتها؛ وAlpaca نظيفة سطحية؛ وEvol فيها عطب دمج ترجمي و`<html>` خام؛ وsharegpt سلسة لكن صغيرة وموسومة `lang: en`؛ وAceGPT-v2 يتسرّب موجّهها إلى حقل الهدف في الأعماق.
- **License note:** apache-2.0 tags on GPT-4-distilled content — OpenAI-terms exposure applies regardless. | وسوم Apache-2.0 على محتوى مُقطَّر من GPT-4؛ تبعات شروط OpenAI قائمة رغم الوسم.

### Bactrian-X (Li et al., 2023)
- **Venue / Link:** [arXiv:2305.15011](https://arxiv.org/abs/2305.15011) · https://github.com/mbzuai-nlp/bactrian-x
- **Data:** https://huggingface.co/datasets/MBZUAI/Bactrian-X — config `ar`
- **Size:** 3,484,884 pairs across **52 languages**; **Arabic = 67,017** (every language config is exactly 67,017).
- **Construction:** 67K English seeds = **Alpaca 52K + Dolly 15K** instructions → translated by the **Google Translate API** (~USD 10,000) → responses regenerated fresh by **gpt-3.5-turbo** (April 2023, ~USD 3,000). Back-translation QC across the 51 translated languages: mean BLEU 48.1, COMET 90.2 — **no Arabic-specific number reported**.
- **MSA vs dialect:** MSA only.
- **Motivation / الدافع:** A cheap, replicable recipe for multilingual instruction data — translate the instruction, regenerate the answer natively, so answers are not translationese even if prompts are. | وصفة رخيصة قابلة للتكرار: تُرجم التعليمة وتُولَّد الإجابة من جديد، فلا تكون الإجابة ركيكة الترجمة وإن كان السؤال كذلك.
- **License:** **CC-BY-NC-4.0** on HF (non-commercial) + gpt-3.5-derived. The card warns the data "inevitably contains some errors or biases."
- **Quality from sampling / الجودة من المعاينة:** See rank #9 — better than its reputation; ChatGPT's Arabic answers are fluent and list-structured, and there is no structural corruption to 60K. The limitation is the task distribution, which is frozen 2023-English. | أفضل من سمعتها: إجابات سلسة منظَّمة بلا فساد بنيوي حتى 60 ألفًا؛ والقيد في توزيع المهام المجمَّد عند إنجليزية 2023.
- **Historical role:** this is the canonical "translate-then-regenerate" baseline that CIDAR and the Arabic post-training survey argue against. | هي الأساس المرجعي لأسلوب «ترجم ثم أعد التوليد» الذي تنتقده CIDAR والمسح.

### Undocumented large sets — use with extreme caution / مجموعات كبيرة بلا توثيق — حذار
- **`akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft` — 6,372,734 rows, empty card, no license.** Sampled 0/2K/20K/3M/6M: deprecated Arabic presentation-form codepoints (`ﻻ` U+FEFB, `ﻷ` U+FEF7) from UN parallel text; English instructions at 3M; outputs that are raw text continuations rather than answers. | ترميز بصور العرض المهجورة، وتعليمات إنجليزية في الأعماق، ومخرجات استكمال نصّ لا إجابات.
- **`Mohaddz/arabic-sft-mix` (2,253,127) / `arabic-sft-mix-2` (2,371,371) — no card, no license.** Composition inferred from a `dataset_name` column pointing at `arabic_aya` dialect configs and `hala`, i.e. a re-mix of items already covered above. | لا بطاقة ولا رخصة؛ يُستدلّ من الأعمدة أنها إعادة خلط لما سبق.
- **`Mohamed-Sami/instruction-fine-tuning-arabic-dataset` — 864,708 rows, no response column** (only `prompt` + `task`). Not usable as SFT as shipped. | لا عمود إجابة، فلا تصلح للضبط كما هي.
- **`MohAlbrayh/saudi-allam-sft-dataset-2M` — 4,504 rows** despite the name. | 4,504 صفًا رغم الاسم.
- **`PetraAI/PetraAI`** — tagged 1M–10M, apache-2.0, 769 downloads, but **zero configs resolve** through the viewer; card is a tag dump. Treat as unverifiable. | لا تُحلّ أي تهيئة عبر العارض؛ اعتبرها غير قابلة للتحقق.
- Smaller translated classics, all viewer-verified: `arbml/alpaca_arabic` **52,002** · `arbml/okapi_arabic` **64,669** · `arbml/alpagasus_cleaned` **9,229** (CIDAR's direct upstream) · `HeshamHaroon/oasst-arabic` **88,836** · `alielfilali01/Arabic_guanaco_oasst1` **10,364** (apache-2.0) · `2A2I/H4_no_robots` **10,000** (cc-by-nc-4.0, Google Cloud Translation of the 10K human-written no_robots demos) · `AhmedBou/Arabic_instruction_dataset_for_llm_ft` **11,000** · `maanasharma5/arabic_sft_data` **15,000**.

---

## §2 — Dialectal instruction mixtures / الخلطات اللهجية

### Atlas-Chat / Darija-SFT-Mixture (MBZUAI-Paris, 2024)
- **Venue / Link:** [arXiv:2409.17912](https://arxiv.org/abs/2409.17912) — *Atlas-Chat: Adapting LLMs for Low-Resource Moroccan Arabic Dialect*
- **Data:** https://huggingface.co/datasets/MBZUAI-Paris/Darija-SFT-Mixture → trains `Atlas-Chat-2B` / `Atlas-Chat-9B`
- **Size:** **458,285** instruction samples (`/size` and card agree).
- **Construction (per the HF card, verbatim counts):** DODa-10k **67,680** instructions (50,760 translation across six directions Darija↔EN/FR/MSA + 19,920 transliteration Arabic↔Latin script) · MADAR **18,800** translation instructions (Rabat dialect ↔ MSA) · NLLB-Seed **10,480** · FLORES+ **5,622** · MArSum **16,756** summarization instructions · Sentiment analysis **86,212** instructions over five Darija datasets (MSDA etc.) · plus novel manual and synthetic sets and quality-controlled English→Darija translation. **The paper's Table 1 groups the same data by task and is reproduced in §6** (Translation 85,662 · Sentiment 86,212 · Story Completion 48,983 · MW-QA 30,555 · Transliteration 16,920 · Summarization 16,756 · MSM-MG 11,808 · **TÜLU-Darija 161,259, translated** · Hard Coded 130). The card and paper differ slightly on transliteration (19,920 vs 16,920); prefer the paper's Table 1. | يعرض جدول البحث التصنيفَ نفسه حسب المهمة (§6)، مع اختلاف طفيف في عدد النقحرة؛ فضِّل جدول البحث.
- **The single most important composition fact:** **TÜLU-Darija = 161,259 rows ≈ 35% of the mixture is machine-translated English**, not native Darija. | **أهم حقيقة تركيبية**: نحو 35% من الخلطة إنجليزية مترجمة آليًا لا دارجة أصيلة.
- **MSA vs dialect:** Moroccan Darija, in both Arabic and Latin script.
- **Motivation / الدافع:** Darija is spoken by ~40M people yet almost absent from instruction data; the mixture consolidates every existing Darija resource into instruction form. | يتحدث الدارجةَ نحو 40 مليونًا وهي شبه غائبة عن بيانات التعليمات؛ فجمعت الخلطة كل مورد دارجي متاح في صيغة تعليمات.
- **License:** **ODC-BY** — explicitly "different licenses apply to subsets… some portions are non-commercial." Presented as a research artifact. | رخص مختلفة للأقسام، وبعضها غير تجاري؛ تُقدَّم كأثر بحثي.
- **Quality from sampling / الجودة من المعاينة:** See rank #5 — real Darija, but the mixture is dominated by translation/transliteration/sentiment/summarization pairs rather than assistant chat; at 20K, CoT rows mix Darija reasoning with MSA answers in one record, and a story subset carries raw social-media text with emoji. | دارجة حقيقية، لكن الخلطة مهامّية لا حوارية، وتختلط الدارجة بالفصحى داخل الصف الواحد، وقسم القصص نصّ اجتماعي خام.

### Nile-Chat / Egyptian-SFT-Mixture & Egyptian-DPO-Mixture (MBZUAI-Paris, 2025)
- **Venue / Link:** [arXiv:2507.04569](https://arxiv.org/abs/2507.04569) — *Nile-Chat: Egyptian Language Models for Arabic and Latin Scripts*
- **Data:** https://huggingface.co/datasets/MBZUAI-Paris/Egyptian-SFT-Mixture · https://huggingface.co/datasets/MBZUAI-Paris/Egyptian-DPO-Mixture
- **Size:** SFT **1,817,288** train + **35,838** test (4.84 GB); DPO **298,073**.
- **Construction (card, verbatim):** *Native* — machine-translation short-sentence pairs **204K**; Egyptian Wikipedia long documents (90–1,500 words) **46K**; transliterated Egyptian forum text (50–70 words) **42K**. *Synthetic* — filtered Aya Collection **223K**; **Tülu-v2&v3-mix translated and transliterated: Arabic 763K + Latin 147K**; UltraChat 7–8-turn conversations **102K**; WildChat transliterated to Latin **256K**; benchmark training subsamples (MMLU/HellaSwag/BeleBele) Arabic **75K** + Latin **44K**; plus **21 hardcoded self-identification questions repeated 50×**. Synthetic translation is prompt-guided by **Claude 3.7**. DPO combines off-policy pairs (rejected = SFT-mixture output, chosen = Claude 3.5 Sonnet v2 rewrite) with on-policy pairs (rejected = SFT-model generation, chosen = held-out safety data), targeting style, code-switching, length control and safety.
- **MSA vs dialect:** Egyptian Arabic (`arz`), **both Arabic script and Arabizi/Latin** — the only major mixture that covers Latin-script Arabic at scale.
- **Motivation / الدافع:** Egyptian is the most widely understood Arabic dialect and is written in two scripts online; a usable Egyptian assistant must handle both. | المصرية أوسع اللهجات فهمًا وتُكتب بخطين على الإنترنت، فلا بدّ لمساعِد مصري من إتقانهما.
- **License:** **missing on the card** — the SFT and DPO repos carry no license field. Treat as research-only until clarified. | **لا رخصة على البطاقة**؛ عاملها كبحثية حتى يُوضَّح ذلك.
- **Quality from sampling / الجودة من المعاينة:** See rank #4 — genuinely idiomatic Egyptian; the `dataset` column lets you separate native from synthetic; safety refusals are written in Egyptian rather than translated-stiff. Defect: FLAN passthrough rows carry non-Arabic payloads (a Chinese paraphrase task answered in Chinese). DPO pairs are stylistically well separated. | مصرية اصطلاحية فعلًا، وعمود `dataset` يفصل الأصيل عن المولَّد، والرفض الآمن مكتوب بالمصرية لا مترجَمًا؛ والعيب تسرّب محتوى غير عربي عبر FLAN.

### saudi-dialect-conversations (HeshamHaroon, 2026)
- **Data:** https://huggingface.co/datasets/HeshamHaroon/saudi-dialect-conversations — **3,545** conversations / **22,536** turns (avg 6.4 turns), 18 everyday topics, apache-2.0.
- **Construction:** synthetically generated but **natively composed** Najdi Saudi Arabic, with `scenario` / `topic` / `complexity` / `english_summary` metadata. Not translated.
- **Motivation / الدافع:** Multi-turn Gulf-dialect chat data essentially does not exist; this fills the smallest useful slice of it. | بيانات الحوار الخليجي متعدد الأدوار شبه معدومة، وهذه تسدّ أصغر ثغرة نافعة فيها.
- **Quality from sampling / الجودة من المعاينة:** Rank #8 — per-row quality is the best dialectal Arabic sampled; the register never slips into MSA across 0/2K/3.4K. The only problem is scale. | أفضل لهجة عربية جودةً في العينة، ولا ينزلق سجلها إلى الفصحى، ومشكلتها الوحيدة الحجم.
- **Related:** *Saudi-Dialect-ALLaM* ([arXiv:2508.13525](https://arxiv.org/abs/2508.13525)) LoRA-tunes ALLaM-7B on a **privately curated 5,466-pair** Hijazi+Najdi instruction set — **not released**. | مجموعة خاصة من 5,466 زوجًا **غير منشورة**.
- **`atlasia/darija-dpo-negatives`** — 69,854 rows, license `other`, Darija DPO negatives.

---

## §3 — Multilingual mixtures with a substantial Arabic share / خلطات متعددة اللغات بنصيب عربي كبير

### The Aya Dataset & Aya Collection (Singh et al., 2024)
- **Venue / Link:** [arXiv:2402.06619](https://arxiv.org/abs/2402.06619) · [ACL 2024](https://aclanthology.org/2024.acl-long.620/) · Aya 23 [arXiv:2405.15032](https://arxiv.org/abs/2405.15032) · Aya Expanse [arXiv:2412.04261](https://arxiv.org/abs/2412.04261)
- **Data:** https://huggingface.co/datasets/CohereLabs/aya_dataset · https://huggingface.co/datasets/CohereLabs/aya_collection · https://huggingface.co/datasets/CohereLabs/aya_collection_language_split
- **Sizes — the distinction that matters / التمييز الجوهري:**
  - **Aya Dataset (human-written): 204,114 pairs / 65 languages. Arabic total = 13,960** — Moroccan `ary` **8,090** (the 10th-largest language in the entire dataset) · Standard `arb` **4,995** · Egyptian `arz` **529** · Najdi `ars` **136** · Ta'izzi-Adeni `acq` **129** · South Levantine `apc` **81**.
  - **Aya Collection: 513M instances / 114 languages.** Arabic configs (`/size`-verified): standard **6,646,024** · moroccan **4,146,308** · egyptian **4,120,671** · najdi **4,120,278** · ta'izzi-adeni **4,120,271** · south_levantine **4,120,223** · mesopotamian **4,120,142** · north_levantine **4,120,142** · tunisian **4,120,142** · algerian **6,046**. **≈39.8M Arabic rows.**
  - Arabic-only re-slice: **`2A2I/Arabic_Aya`, 41,472,592 rows** across 29 per-source configs (translated_soda 14.9M, wiki_split 10M, hotpotqa 3.8M, flan_cot 919K, dolly 148K, aya_dataset 14,210), apache-2.0.
- **Construction:** the Dataset is original human writing + human re-annotation via the Aya Annotation Platform (May–Dec 2023, contributors in 119 countries). The Collection is templated conversion of existing native-language NLP datasets **plus large-scale machine translation of English datasets**.
- **License:** **apache-2.0** for both — genuinely commercially usable, which is rare in this space. (Aya *model* weights are CC-BY-NC + C4AI acceptable-use; the data is not.) | **رخصة تسمح بالاستخدام التجاري فعلًا**، وهو نادر هنا.
- **Motivation / الدافع:** A participatory, community-annotated alternative to English-only instruction data, with deliberate depth on non-English languages. | بديل تشاركي مجتمعي لبيانات التعليمات الإنجليزية وحدها، مع تعميق مقصود في اللغات غير الإنجليزية.
- **Quality from sampling / الجودة من المعاينة:** See rank #14. **The single most important caveat in this document:** the seven ~4.12M "dialect" configs land within ~200 rows of one another and share row IDs — that is a **uniform MT fan-out of the same English source**, not independent dialect collection. Verified directly: row `id 1` is the same CoQA passage in MSA, Egyptian and Moroccan. The MT is also visibly damaged (`<unk>` tokens mid-text, enumerators turned into words/years) and the Moroccan config mixes diacritized Darija with plain MSA inside one passage. **Cite the Aya *Dataset* numbers (13,960) for human Arabic, never the Collection's 39.8M.** | **أهم تحفّظ في هذه الوثيقة**: التهيئات اللهجية السبع نسخة آلية من المصدر الإنجليزي نفسه لا جمعٌ لهجي مستقل، والترجمة معطوبة بصريًا. استشهد بـ 13,960 للعربية البشرية لا بـ 39.8 مليون.
- **Derived preference sets (2A2I):** `Aya-Command.R-DPO` **14,210** · `Aya-SambaLingo.Arabic.Chat-DPO` **14,210** · `Aya-Aya.23.8B-DPO` **13,960** · `Aya-AceGPT.13B.Chat-DPO` **12,960** — all apache-2.0; chosen = the human Aya answer, rejected = a model generation. A clean, if small, "human vs. model" preference signal in Arabic. | إشارة تفضيل نظيفة «بشري مقابل نموذج» بالعربية، وإن كانت صغيرة.

### Other multilingual sets / خلطات أخرى
- **`neulab/PangeaInstruct`** — apache-2.0, 1M–10M, Arabic included, but it is **multimodal VQA instruction data** and its **dataset viewer is disabled**, so it could not be sampled. | **متعددة الوسائط** وعارض بياناتها معطَّل، فتعذّرت المعاينة.
- **`multilingual/orca_dpo_pairs`** — apache-2.0, **64,656** rows, Arabic as one language column.
- **Magpie — no Arabic variant exists.** An exhaustive sweep of the `magpie`-matching repos on the Hub found English, Chinese, Spanish, Japanese, Korean and Romanian variants but **nothing in Arabic**. The method ([arXiv:2406.08464](https://arxiv.org/abs/2406.08464), ICLR 2025) is language-agnostic and would apply directly to any Arabic-aligned chat model — this is a real, open gap and arguably the cheapest way to generate *native-prompt* Arabic SFT at scale. | **لا توجد نسخة عربية من Magpie**؛ والطريقة محايدة لغويًا وتنطبق مباشرة على أي نموذج محادثة عربي — ثغرة مفتوحة وأرخص سبيل لتوليد تعليمات عربية أصيلة بكميات كبيرة.
- **Tülu multilingual:** `allenai/tulu-3-sft-mixture` is English; its Arabic reach is indirect — Tülu-2/3 subsets appear **translated** inside `Egyptian-SFT-Mixture` (763K Arabic + 147K Latin) and inside `Hala-4.6M-SFT`. There is no first-party Arabic Tülu release. | لا إصدار عربي رسمي من Tülu؛ وصولها للعربية غير مباشر عبر ترجمتها داخل خلطتي Nile-Chat وHala.

---

## §4 — Reasoning, math, code and tool calling / الاستدلال والرياضيات والبرمجة واستدعاء الأدوات

This is the **thinnest** part of the Arabic ecosystem. The Arabic post-training survey (§7) found **zero**
Arabic datasets for function calling or code generation among the 366 it catalogued in mid-2025; the sets
below have appeared since, and are still small.
| هذا **أنحف** جزء في المنظومة العربية؛ لم يجد المسح أي مجموعة عربية لاستدعاء الأدوات أو توليد الشيفرة بين
366 مجموعة، وما ظهر بعده لا يزال صغيرًا.

- **`Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset`** — **9,210** rows, apache-2.0. Clean, uniform Arabic CoT (`المعطيات` → `الخطوات` → conclusion); word problems only; heavy template repetition (rank #10). | سلاسل استدلال عربية نظيفة موحّدة التنسيق، مسائل لفظية فقط، وتكرار قوالبي عالٍ.
- **`Omartificial-Intelligence-Space/Arabic-Math-SFT`** — **5,000** rows, apache-2.0, **multimodal** (geometry figure + Arabic problem + bare `<answer>60°</answer>`). No worked solutions; not usable as text SFT. | متعددة الوسائط وبلا خطوات حلّ؛ لا تصلح ضبطًا نصيًا.
- **`miscovery/Math_CoT_Arabic_English_Reasoning`** — 2,834 rows, MIT, bilingual math CoT.
- **`HeshamHaroon/Arabic_Function_Calling`** — **50,810** rows, apache-2.0, the largest ungated Arabic tool-calling set. Explicit `dialect` labels (MSA / Levantine / Gulf / Egyptian), paired `query_ar` + `query_en`, structured `arguments` JSON, and a `requires_function` flag for negative examples. Sampled 0/2K: clean and consistent, with plausible Gulf/Levantine phrasing (`شو سعر الذهب اليوم بلبنان عيار ٢٤؟`). | أكبر مجموعة عربية غير مقيَّدة لاستدعاء الأدوات، بوسوم لهجية صريحة وأمثلة سالبة، ونظيفة متسقة في المعاينة.
- **`TuwaiqAcademy/AISA-ArabicFC`** — **12,220** rows, apache-2.0; the ArabicNLP-2026 agentic function-calling shared task, MSA + Gulf/Egyptian/Levantine/Maghrebi.
- **Tool Calling for Arabic LLMs** ([arXiv:2509.20957](https://arxiv.org/abs/2509.20957)) — four Arabic tool datasets: **Glaive-Ar** 37,684 train / 1,953 test (with-call) + 38,678 / 1,000 (without); **xLAM-Ar** 58,999 / 1,001 + 19,361 / 1,077; **CustomTools** 4,528 / 1,000 + 4,313 / 1,000; **IslamicRAGTool** 10,000 / 1,000 + 10,000 / 1,000. Glaive and xLAM were translated EN→AR with **Gemini-2.5-Flash**; CustomTools and IslamicRAGTool were Gemini-generated. Base model Fanar-1-9B. Adding Arabic data lifted argument-population accuracy on CustomTools **0.58 → 0.80**, and **50.2% of remaining "errors" are semantically correct but syntactically different from gold**. **No HF or GitHub release, no license stated.** | **لم يُنشر شيء ولا رخصة معلنة.**
- **SmolKalam `*_think` configs** are currently the largest source of **Arabic-language reasoning traces**: `OpenThoughts3_50K` 50,000 · `OpenThoughts3_NoThink_180K` 180,000 · `multi_turn_reasoning_if_think` 28,217 · `Mixture_of_Thoughts_science_no_think` 86,110 · `s1k_1.1_think` 835 · `hermes_function_calling_v1_no_think` 8,961. | أكبر مصدر حالي لآثار الاستدلال بالعربية.
- **Code:** there is **no credible Arabic code-instruction dataset**. `FreedomIntelligence/Code-Alpaca-Arabic-GPT4` exists but inherits the family's translation issues, and `Arabic-OpenHermes-2.5`'s code rows are destroyed (translated identifiers). Keep code SFT in English. | **لا توجد مجموعة عربية موثوقة لتعليمات البرمجة**؛ أبقِ ضبط البرمجة بالإنجليزية.

---

## §5 — Preference / DPO / RLHF data (secondary) / بيانات التفضيل (ثانوية)

| Repo | Rows | License | Origin | Sampled verdict / حكم المعاينة |
|---|---:|---|---|---|
| `alielfilali01/ultrafeedback-arabic` | **63,135** | **missing** | UltraFeedback-binarized, translated | Largest MSA preference set; fluent, but `rejected` is full of "As an AI language model" boilerplate. Strip it. \| الأكبر بالفصحى وسلسة، لكن المرفوض مليء بالقوالب؛ أزِلها. |
| `MBZUAI-Paris/Egyptian-DPO-Mixture` | **298,073** | **missing** | on-policy + off-policy Egyptian, Claude 3.5 Sonnet v2 rewrites | Best-documented Arabic DPO; idiomatic Egyptian; chosen/rejected differ stylistically in a real way. \| الأفضل توثيقًا، مصرية اصطلاحية، والفارق أسلوبي حقيقي. |
| `FreedomIntelligence/Arabic-preference-data-RLHF` | **11,548** | **missing (README 404)** | AceGPT RLAIF over native Arabic prompts | Fine-grained signal — both branches are plausible; small. \| إشارة دقيقة وكلا الفرعين معقول؛ لكنها صغيرة. |
| `multilingual/orca_dpo_pairs` | **64,656** | apache-2.0 | Orca DPO, multilingual | Arabic is one column of a multilingual set. |
| `atlasia/darija-dpo-negatives` | **69,854** | other | Moroccan Darija negatives | Dialect-specific; updated 2026. |
| `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` | **31,846** | **missing** | translated ORPO mix | **Weak signal** — chosen/rejected often differ by one synonym (`غريب` vs `مريب`). \| **إشارة ضعيفة**: الفرق مرادف واحد أحيانًا. |
| `2A2I/Aya-*-DPO` family | 12,960–14,210 each | apache-2.0 | chosen = human Aya answer, rejected = model generation | Clean human-vs-model signal; small; the only apache-2.0 Arabic DPO with a documented construction. \| إشارة نظيفة «بشري مقابل نموذج»، صغيرة لكنها الوحيدة المرخّصة بوضوح. |
| `2A2I/argilla-dpo-mix-7k-arabic` | **7,500** | mit | Argilla dpo-mix-7k, translated | **Broken:** the JSON role values were translated — `"role": "مستخدم"` instead of `"user"`. Repair before use. \| **معطوبة**: تُرجمت قيم الأدوار نفسها؛ أصلِحها قبل الاستخدام. |
| `2A2I-R/dibt_10k_prompts_ranked_arabic` | **10,331** | **missing** | DIBT prompt-quality ranking, translated | Prompt-quality ranking rather than response preference. |

**Overall / خلاصة:** Arabic preference data is small, mostly translated, and **systematically under-licensed** —
five of the nine sets above carry no license at all. The two worth building on are
`Egyptian-DPO-Mixture` (dialect, well documented) and `ultrafeedback-arabic` (MSA breadth, after cleaning).
| بيانات التفضيل العربية صغيرة ومترجمة غالبًا و**ناقصة الترخيص منهجيًا** (خمس من تسع بلا رخصة)؛ وأصلحها للبناء عليه اثنتان.

---

## §6 — SFT mixtures described inside Arabic model reports / خلطات الضبط الموصوفة داخل تقارير النماذج العربية

Most Arabic flagship models describe their instruction data but **do not release it**. This section records
what each report actually documents, with real counts, and flags what is closed. | معظم النماذج العربية
الرائدة تصف بيانات تعليماتها لكنها **لا تنشرها**؛ ويسجّل هذا القسم ما وثّقه كل تقرير بأرقام حقيقية.

**Cross-cutting pattern / النمط العام:** a consistent three-stage arc across labs — **translate English SFT
→ discover cultural/lexical leakage → regenerate natively for the culture-sensitive parts.** AceGPT (2023)
regenerates *responses* natively; Fanar 1.0 (2025) regenerates 30K responses just to remove English names;
Fanar 2.0 (2026) regenerates *reasoning traces* natively; NileChat openly concedes it could not afford to.
And a **double standard worth naming**: human translation, post-editing and cultural localization are
routinely applied to *benchmarks* (Jais's `MMLU_H` vs `MMLU_M`, ALLaM's AraTruthfulQA, Jais 2's "replacing
'Haoran' with 'Omar'", Fanar's human-post-edited dialect MT) while the *training* data is machine-translated
unreviewed. | قوس ثلاثي متكرر: **ترجمة ← اكتشاف التسرّب الثقافي ← إعادة توليد أصيلة** للأجزاء الحساسة
ثقافيًا؛ ومفارقة تستحق التسمية: الترجمة البشرية والتنقيح والتوطين تُطبَّق على **المعايير** لا على **بيانات
التدريب**.

### Jais / Jais-chat (Sengupta et al., 2023) — [arXiv:2308.16149](https://arxiv.org/abs/2308.16149)
- **SFT mixture:** "10M prompt–response pairs in total, made up of 4M in Arabic and 6M in English"; the paper's tables sum to **9,699,900 = 6,016,756 EN + 3,683,144 AR**. One epoch, loss on answer tokens only.
- **Arabic sources (Table 7), with the paper's own "Is Translated?" column:** xP3-Ar 1,375,257 (No) · Super-NaturalInstructions-Ar 1,251,444 (**Yes**) · Baize-Ar 590,846 (**Yes**) · Unnatural-Ar 199,100 (**Yes**) · Natural-Questions-Ar 86,005 (**Yes**) · Bactrian-Ar 66,880 (No) · Alpaca-Ar 51,280 (**Yes**) · SafetyQA-Ar 22,617 (Mixed) · **NativeQA-Ar 15,018 (No)** · Dolly-15k-Ar 14,833 (**Yes**) · HC3-Ar 7,139 (**Yes**) · **NER-Ar 1,969 (No)** · Basic-Conv-Ar 756 (**Yes**).
- **→ 8 of 13 Arabic sets are machine-translated = 2,201,403 / 3,683,144 ≈ 60%. Natively authored human Arabic totals ~17,000 pairs (NativeQA-Ar + NER-Ar).** | **60% من البيانات العربية مترجمة آليًا، والمكتوب بشريًا نحو 17 ألف زوج فقط.**
- **Safety:** SafetyQA-Ar 22,617. The paper: "We crawled data from various Arabic websites… and amassed **approximately 1,000 instances in Arabic**," then "we integrated the DoNotAnswer dataset, which comprises **around 6,000 questions**… Subsequently, **we translated this dataset into Arabic**." No RLHF/DPO (HH-RLHF used chosen-response-only).
- **Translation QC — verbatim:** *"Due to the limited availability of instruction-tuning datasets for Arabic, we translated some of the above English instruction-tuning datasets to Arabic using the same machine translation system that we used for the training data."* The system is an in-house FairSeq transformer trained on OPUS, **31 BLEU on Flores-101**. The **only** quality control is task-level exclusion: *"we… excluded tasks that were primarily related to translation as well as those relating to counting words, as they could break when translated to Arabic."* **There is no criticism of MT'd instruction data, no post-editing, and no translationese discussion anywhere in the paper.** Meanwhile, for *evaluation*, "we hired native speakers of Arabic to manually translate the MMLU dataset," and results distinguish `MMLU_H` (human-translated) from `MMLU_M` (machine-translated). | الضبط الوحيد هو استبعاد مهام بعينها؛ **لا نقد للترجمة الآلية ولا تنقيح بشري لبيانات التدريب**، بينما تُترجَم **المعايير** بشريًا.
- **MSA/dialect:** MSA only — dialects appear once, as a disclaimer.
- **Release:** weights Apache-2.0; **instruction data NOT released** (`inceptionai` / `core42` publish zero HF datasets; `inception42` publishes only eval sets).

### Jais family / Jais-adapted (Sept 2024) — HF model cards + [arXiv:2407.12869](https://arxiv.org/abs/2407.12869)
- **⚠️ Citation correction:** there is **no arXiv paper for the Sept-2024 "Jais Family Model Card"** — it is documented only on HuggingFace. **arXiv:2409.12058 is a Finsler-geometry mathematics paper and must not be cited here.** The correct arXiv companion for the `jais-adapted-*` method is **2407.12869** (*Bilingual Adaptation of Monolingual Foundation Models*). | **تصحيح استشهاد:** لا يوجد بحث arXiv لبطاقة عائلة Jais، و2409.12058 بحث رياضيات لا علاقة له بالموضوع.
- **SFT mixture — the entire published description:** "our updated instruction-tuning dataset comprises **~10M and ~4M prompt-response pairs in English and Arabic** respectively"; sources are "open-source fine-tuning datasets filtered for topic and style diversity… internally curated human data is incorporated to enhance cultural adaptation… supplemented with content generated using synthetic methods including **machine translation, distillation, and model self-chat**." 3 epochs, packing. **No per-source table, no counts for the human / distilled / self-chat components.** | لا جدول مصادر ولا أعداد للمكوّنات.
- **Release:** not released; the `*-chat` repos are now **gated**.

### Jais 2 (2026) — [arXiv:2608.13580](https://arxiv.org/abs/2608.13580)
- 8B & 70B, **Apache-2.0**. Pipeline: CPT → IFT (**3 epochs, "over 20 million instruction–completion pairs"**, including "enhanced rewrites of Jais 1 SFT data") → **DPO on "over 200k chosen/rejected preference pairs"** (LLM-judge constructed; GRPO was tried but "was not used in the models we are releasing"). Arabic/English split of the 20M+: **not stated**.
- **Dialect-specific components (real counts):** dialectal-translation IFT **612,916 pairs / 15,731,037 tokens** · dialect-ID **624K examples across 15 dialects** · poetry **427,337** training poems · Islamic QA **150,890** · summarization **540K / 178M tokens**. Safety taxonomy expanded to **30 finer-grained risk types**.
- **Language coverage:** 15 ISO-coded Arabic varieties (ar, acm, acx, aeb, afb, apc/ajp, apd, arq, ars, ary, arz, avb, ayl) + en, fr. The report is explicit: *"a model trained solely on MSA and English fails to capture this."* | **15 صنفًا عربيًا موسومة بمعايير ISO**، مع تصريح بأن الاقتصار على الفصحى لا يكفي.
- **Translation stance:** *"When Arabic data was scarce, we translated high-quality English datasets into Arabic to preserve linguistic diversity."* Still no MT critique for training data; the only named-entity-localization statement applies to **safety evaluation** ("a **manual localization step to ensure regional relevance (e.g., replacing 'Haoran' with 'Omar')**", with regional questions "developed by **22 native Arabic speakers**").
- **Release:** SFT/DPO data not released — but Jais 2 **consumes the public `Darija-SFT-Mixture` (458K) and `Egyptian-SFT-Mixture` (1.85M)**, which is a strong external endorsement of the two open mixtures recommended above. | لا تُنشر بياناتها، لكنها **تستهلك الخلطتين المفتوحتين** المذكورتين أعلاه.

### AceGPT (Huang et al., 2023) — [arXiv:2309.12053](https://arxiv.org/abs/2309.12053) — **the open exception / الاستثناء المفتوح**
- **SFT mixture, Table 15 verbatim:** Quora-Arabic-40K **43,050** (collected from Quora, GPT-4 responses) · Alpaca **49,969** (self-instruct, GPT-4) · Alpaca-Chinese **49,969** · Alpaca-Arabic **49,969** (GPT-4-translated) · Code-Alpaca-Arabic **20,022** · Evol-Instruct-Arabic **69,997** · ShareGPT **80,179** (humans + GPT-3.5-Turbo). *"Native Arabic data like Alpaca-Arabic-GPT4 and Quora-Arabic-GPT4 are included thrice in the mixture… **totaling 629,293 data**."* (**Caveat:** the ×3 sentence does not reconcile arithmetically — 629,293 is reproduced exactly by ×3 on Quora + Evol-Instruct + Code-Alpaca only. Report the 629,293 total and the per-row sizes; treat the multiplicity sentence as stated-but-inconsistent. Note also that the paper's ShareGPT row is 80,179 while the released `FreedomIntelligence/sharegpt-arabic` holds 5,231 — they are not the same artefact.) | **تحفّظ:** جملة التضعيف ×3 لا تتطابق حسابيًا، وصفّ ShareGPT في البحث (80,179) غير المنشور على HF (5,231).
- **RLAIF:** reuse **40K** Quora questions, sample paired outputs from the fine-tuned 7B, label with GPT-4, apply an **order-switch consistency filter → 12K Arabic pairs**, plus 12K open-source pairs = **24K** reward-model pairs; PPO on **another 30K** crawled Quora questions. Human agreement study on **800 examples** gave a **correlation of 0.84** between GPT-4 and human judgements.
- **ACVA:** **8,000+** yes/no cultural-alignment questions from 50+ topic keywords, ChatGPT-generated and human-filtered; a **clean subset of 2,486** was validated by native speakers. Released as `FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment`.
- **Translation stance — the strongest quotes in the whole literature:**
  > *"we do not translate responses of original Alpaca data in English, **since these responses are in Western values and not localized**. Alternatively, we choose to re-generate responses using GPT-4 for these translated questions (in Arabic)."*
  > *"common entities in the popular open-source datasets such as Alpaca are mostly Western (e.g. 'John', 'Apple', and 'New York'), **deviating from Arab's actual interest** (e.g. 'Mohammed', 'Muslim Brotherhood', and 'Egypt')."*
  > *"The localized instructions are Arabic natural questions derived from real-world contexts… which can help models to **capture what Arabs care in the wild**."*
  Table 3 of the paper is a named-entity audit (top-5 person / organization / GPE per dataset). **Note the asymmetry: the instructions are still machine-translated (by GPT-4); only the responses are natively regenerated.** | **لاحظ عدم التماثل**: التعليمات تبقى مترجمة، والإجابات وحدها تُولَّد أصلًا.
- **Release:** **the SFT and preference data ARE released** under `FreedomIntelligence/*` (see §1) — this is the single most reusable Arabic instruction release in the field.

### AceGPT-v2 / Native Alignment — [arXiv:2412.03253](https://arxiv.org/abs/2412.03253) (NeurIPS 2024)
- **⚠️ Citation correction:** the HF `AceGPT-v2-*` models come from **2412.03253** (*Alignment at Pre-training! Towards Native Alignment for Arabic LLMs*), **not** 2412.12310, which is a different model (AraLLaMA). | **تصحيح:** نماذج AceGPT-v2 من 2412.03253 لا من 2412.12310.
- Alignment is moved **into pretraining**: seed rewriting data annotated by GPT-4, then **10B tokens** of Arabic alignment data rewritten by a fine-tuned Qwen-1.5-4B-Chat "Alignment Worker" — this is exactly what `FreedomIntelligence/AceGPT-v2-AlignmentData` (3,222,135 rows, §1) contains. Downstream SFT in the paper's experiments uses `Alpaca-Arabic-GPT4` (50K); DPO uses `Arabic-preference-data-RLHF` (11,548).

### AraLLaMA / Second Language (Arabic) Acquisition — [arXiv:2412.12310](https://arxiv.org/abs/2412.12310) (ACL 2025)
- Introduces **ALAN**, a GLAN-style synthetic Arabic instruction pipeline: 127 critical topics → **11,430 subjects** and **244,812 knowledge points** (GPT-4-0613) → *"In total, we've generated **733,419 instruction tuning data pieces**"* (multiple-choice, open-ended, coding), plus reuse of the AceGPT sets. Vocabulary expanded by 12,800 Arabic subwords (I-BPE). **No DPO stage. The ALAN dataset is not stated as released.** | **لم يُعلَن نشر ALAN.**

### ALLaM (SDAIA/NCAI) — [arXiv:2407.15390](https://arxiv.org/abs/2407.15390) (ICLR 2025)
- **SFT corpus "Ultra-Instinct": v1 = 12M samples "evenly split between English and Arabic"; v2 ≈ 6M** after strict quality filtering — v2 is what trained the released models. 3 epochs, prompt tokens masked. **No per-source table**; sources are described only qualitatively (domain experts, prompt librarians, local institutes, licensed LLM generation, MT). The HF card for `ALLaM-7B-Instruct-preview` states **7M instructions + 260K preference pairs**.
- **DPO:** **25,854 human-verified {prompt, accepted, rejected} triplets**, expanded by sampling **10 responses per instance** → **245K preference samples after filtering**. Notably: *"0.1%–1% noisy samples visibly break the model."* Their stated contrast with Zephyr: *"While [Zephyr] utilized preference data from AI Feedback (AIF) at scale, we adopt a more cautious approach… We generate a smaller volume of data, ensuring it is **fully reviewed, edited, and verified by humans**."* | **بيانات تفضيل أصغر لكن مُراجَعة بشريًا بالكامل.**
- **Translation stance:** one clause only — v2 filtering includes *"removal of low quality machine-translated Arabic data from English sources"* — while MT remains a stated source: *"machine translation models to convert rich English SFT data into Arabic."* Corpus framing: *"not human generated, but rather, human driven."* Localization appears only on the eval side (AraTruthfulQA: 285 GPT-4-translated items *"carefully validated and localized by human verifiers"* + **256 human-curated questions**).
- **MSA/dialect:** **the words "dialect", "MSA" and "Modern Standard Arabic" appear zero times in the entire paper** (verified over both the HTML and the ICLR PDF). | **لا ترد كلمة «لهجة» ولا «الفصحى» ولو مرة واحدة في البحث كله.**
- **Release:** **NOT released** — `ALLaM-AI` publishes zero HF datasets. Weights carry an `apache-2.0` tag but the referenced LICENSE file is missing from the repo. Third-party derivatives such as `MohAlbrayh/saudi-allam-sft-dataset-2M` (4,504 rows of translated Alpaca) are **not** ALLaM's data.

### Fanar 1.0 (QCRI) — [arXiv:2501.13944](https://arxiv.org/abs/2501.13944)
- **Table 8 counts:** SFT stage-1 **3.6M** · SFT stage-2 **834K** · **DPO 250K** · annealing-SFT 5K · annealing-DPO 4K. HF card summarises as **4.5M SFT + 250K DPO**.
- Public-source curation yielded *"approximately **2.5 million** instructions and dialogues across 11 categories in both languages"*; named sources include Aya, BoolQ, **CIDAR**, ELI5, HelpSteer, **InstAr**, LMSYS-Arena, MathInstruct, No-Robots, Orca-Math, Tulu, UltraChat, UltraFeedback, WebGLM-QA, Beavertails and several safety sets. Synthetic: *"we generated **close to a million** samples in both Arabic and English, mainly focusing on culturally contextualizing and aligning the model's responses"* (Gemma-2-27B, Qwen2.5-72B, Command-R+, Llama-3.1-70B/405B; judge Llama-3.1-405B-FP8; Farasa spell-check filter). DPO ≈**250K**, balanced across languages, **~20% on-policy**.
- **Human program:** *"a team of **40–250 annotators**… at peak capacity contributed up to **10K prompts** per day… core user base grew to **130 individuals**… **Over 90% of the prompts were in Arabic**."* Human validation on **~700** prompts, 5–6 assessors each; the Llama-405B judge reached **≈87% agreement** with humans. The annotator dislike taxonomy has 8 categories including **cultural misalignment**.
- **Translation stance — the best cultural-translationese passage in the corpus:**
  > *"**One notable drawback of curating data from public sources is the tendency of the translation process to introduce typical English-language contexts into Arabic. This includes culturally specific elements such as personal names, geographic references, traditions, social norms, and lifestyle practices, which often fail to capture local cultural nuances.**"*
  > *"the model's tendency to use **English entity names and locations** in creative-writing tasks, even when responding to Arabic queries. To address this, we regenerated responses… targeting queries containing the **100 most common English male and female names**. Incorporating **30K** contextually appropriate responses effectively mitigated this behavior."*
  > *"even widely-used and well-regarded datasets often contain a substantial proportion of low-quality samples."*
  Their translated data was also passed through *"a **value-relevance filter**… to remove samples misaligned with cultural and religious values,"* and tasks that translate badly (translation, grammar, word riddles, sorting) were dropped. | **مقطع مرجعي** في تسرّب السياق الإنجليزي إلى العربية عبر الترجمة، مع مرشِّح ملاءمة قِيَمية وإعادة توليد 30 ألف إجابة لإزالة الأسماء الإنجليزية.
- **Dialect on the SFT side:** one line only — *"a vendor-generated dataset for dialectic dialogues"* — no count, no dialect list. All other dialect work is pretraining, MT, or evaluation. | **سطر واحد بلا أعداد.**
- **Release:** SFT/DPO data **NOT released**; weights `QCRI/Fanar-1-9B(-Instruct)` Apache-2.0; only benchmarks (AraDiCE etc.) are public.

### Fanar 2.0 (2026) — [arXiv:2603.16397](https://arxiv.org/abs/2603.16397)
- Fanar-27B, continual on gemma-3-27b-pt (~166B tokens). **Table 6: SFT 3,985,215 · long-context-16K 54,321 · capability rebalancing ~1.8M · DPO 280K preference pairs.** Arabic/English split not stated. Re-filtering with Qwen-3-32B *"reduced the SFT dataset by nearly half."*
- **~250K native Arabic reasoning traces** across 61 categories, generated natively by Qwen3 models with a correct-answer filter. Cultural alignment uses the public **`UBC-NLP/palm`** set plus user-flagged production logs. Safety: **FanarGuard**, a 4B moderation model trained on **468K prompt–response pairs** scored 1–5 for harmlessness **and cultural alignment**.
- **The two most decision-relevant sentences in this whole review:**
  > *"**Translating existing English-distilled reasoning datasets into Arabic introduced language-mixing artifacts and degraded trace quality; instead, we generated reasoning traces natively in Arabic.**"*
  > *"A key lesson from Fanar-9B was that **data quality and cultural specificity matter more than dataset size** in the post-training regime."*
  | **أهم جملتين للقرار:** ترجمة آثار الاستدلال أنتجت خلطًا لغويًا وتدهورًا، فوُلِّدت أصلًا بالعربية؛ و**جودة البيانات وخصوصيتها الثقافية أهم من حجمها**.
- **Release:** weights Apache-2.0; post-training data not released; `QCRI/FanarGuard` exists but is **gated**.

### Atlas-Chat / Nile-Chat / NileChat — the open dialectal recipes / الوصفات اللهجية المفتوحة
- **Atlas-Chat** ([arXiv:2409.17912](https://arxiv.org/abs/2409.17912), MBZUAI-Paris) — **Darija-SFT-Mixture released, ODC-BY** (§2). Paper Table 1 composition: Translation **85,662** (DODa-10K, FLORES+, MADAR, NLLB-Seed; native) · Sentiment Analysis **86,212** (MSAC, MSDA, MAC, ElecMorocco2016, MYC; native) · Story Completion **48,983** (9esa.com; Claude 3.5 Sonnet) · MW-QA **30,555** (Wikipedia; Claude 3.5 Sonnet) · Transliteration **16,920** (DODa-10K; native) · Summarization **16,756** (MArSum; native) · MSM-MG **11,808** (Twitter/X, YouTube; Claude 3.5 Sonnet) · **TÜLU-Darija 161,259 (translated from English by Claude 3.5 Sonnet)** · Hard Coded 130 → **≈458K**. So **~35% of the mixture is MT'd English→Darija**; the rest is native or Claude-synthesised Darija. Their translation-model bake-off: *"closed-source models consistently outperformed open-source alternatives, with **Claude 3.5 Sonnet emerging as our final choice**"*; the result was *"**extensively reviewed by groups of native speakers**"* with *"several **post-processing measures to correct errors introduced by the automatic translation**."* Native-reviewer rule, verbatim: *"**if the data is a mix of Darija with some MSA, it is acceptable; if it is mixed with other dialects, it is not.**"* | ~35% من الخلطة مترجمة، والباقي أصيل أو مولَّد بـ Claude، مع مراجعة موسَّعة من متحدثين أصليين.
- **Nile-Chat** ([arXiv:2507.04569](https://arxiv.org/abs/2507.04569), MBZUAI-Paris) — **Egyptian-SFT-Mixture released** (§2). Nile-Chat-4B / 3×4B-A6B (Branch-Train-MiX merging script-specialised experts) / 12B; ~**25% Latin-script (Arabizi)**; translator **Claude 3.5 Sonnet v2** via Bedrock, chosen because *"**Claude produced more natural and dialect-appropriate outputs**"* than GPT-4o in head-to-head testing. Post-processing: filtering skipped/non-translations, ≥70% character-length validation, manual corrections, fastText LID (9,660 rows removed from the TÜLU translation alone). DPO adds off-policy code-switching reformulations plus **1,000** minimal prompts with Claude-authored positives.
- **NileChat** ([arXiv:2505.18383](https://arxiv.org/abs/2505.18383), UBC-NLP, EMNLP 2025) — **a different paper by a different group; do not conflate with Nile-Chat.** Qwen-2.5-3B CPT; **SFT set = 2,273,809 instructions**, teacher **Command R+ (104B)**; per-language: Egyptian Arabic-script SmolTalk-MT 195,260 + TÜLU-v2-mix-MT 178,109 + cultural synthetic 107,428 · Egyptian Arabizi 93,181 · Moroccan reuses **Darija-SFT-Mixture 458,155** + SmolTalk-MT 192,266 + cultural synthetic 25,159 · Moroccan Arabizi 93,419 · MSA 96,933 · French 99,468 · English 149,124 · shared ORCA 460,203 + Dolphin 425,703. EGY and MOR variants are SFT'd separately then **merged by weighted linear averaging**. **Pretraining data released CC-BY-NC-4.0** (`UBC-NLP/nilechat-*`); **the SFT mixture itself is not published.**
  > *"Current research directions predominantly rely on **synthetic data generated by translating English corpora**, which… often results in **models aligned with source language culture**."*
  > Self-critique, verbatim: *"**SFT phase predominantly utilized translated data due to resource constraints. This reliance on translated, rather than native, data for SFT might impact the model's nuanced performance.**"* | **نقد ذاتي صريح**: اعتمدت مرحلة الضبط على بيانات مترجمة لضيق الموارد، وهو ما قد يضرّ بالأداء الدقيق.
- **GemMaroc** ([arXiv:2505.17082](https://arxiv.org/abs/2505.17082)) — a minimal-data Darija recipe: LIMA 1,000 (700 Darija / 300 EN) + DEITA 5,000 (3,700 / 1,300) + TÜLU 46,000 (33,000 / 13,000) ≈ **49K total, ~37.4K Darija**, translated with Gemini 2.0 Flash. Useful as an existence proof that **tens of thousands of examples can suffice** for dialect adaptation. | برهان على أن عشرات الآلاف قد تكفي للتكيّف اللهجي.

### Cohere — Command A, Command R7B Arabic, Aya Expanse, Aya 23
- **⚠️ There is no technical report for Command R or Command R+** — Command A is Cohere's first. | **لا تقرير تقني لـ Command R/R+.**
- **Command A** ([arXiv:2504.00698](https://arxiv.org/abs/2504.00698), 111B, CC-BY-NC weights) — 23 languages incl. Arabic; post-training = 6 SFT experts + 6 RL experts + two merge steps. *"The datasets are collected through various means including **human annotation, multilingual data arbitrage, templated public datasets, or machine translation**."* Human annotation takes two forms: *"1) **LLM-generated response with human post-editing**; and 2) manually annotated human data,"* and they report that *"**LLMs can produce responses that are comparable or even better than the human-written gold label** provided in many multilingual datasets."* Multilingual safety data is *"translated automatically, then **corrected and naturalised by multilingual annotators**."* **No counts of multilingual SFT/preference examples anywhere. Data not released.**
- **Command R7B Arabic** ([arXiv:2503.14603](https://arxiv.org/abs/2503.14603)) — expert annotators translated IFEval into Arabic and **added two Arabic-specific instruction types** ("add NN diacritics to the response"; "use a specific grammatical verb to start sentences"), which then seed synthetic Arabic instruction-following data filtered by reward models and LLM-judge panels. *"**lexical control tasks such as length adherence and structured generation are awkward or nonsensical when translated to Arabic**."* Explicitly **MSA-only**. Eval sets IFEval-AR 541, FaithEval-Arabic 500, mArenaHard-Arabic 500. **No SFT counts. Data not released.**
- **Aya Expanse** ([arXiv:2412.04261](https://arxiv.org/abs/2412.04261)) — multilingual arbitrage (teacher pool + internal reward-model Arbiter) → offline DPO → 3 rounds of online iterative DPO → merging. Preference pairs are built by *"**contrasting in-language completions from a highly performant multilingual LLM with lower quality completions translated from English**"*, explicitly to steer away from *"**undesirable artifacts, such as those introduced by poor translation**."* SFT arbitrage +9.1% win-rate over Aya 23; preference training +7.1%. **No data counts; data not released.**
- **Aya 23** ([arXiv:2405.15032](https://arxiv.org/abs/2405.15032)) — mixture: multilingual templates **55.7M** (xP3x + Data Provenance + Aya Collection, 161 datasets) · human annotations **55K** (filtered from the Aya Dataset's 204K) · translated data **1.1M** (≤3,000 instances per language per dataset) · synthetic **1.63M** (Command R+ generating in-language responses for translated ShareGPT and Dolly prompts). **The Aya Dataset and Aya Collection ARE released** (§3).

### Models with no usable instruction-data disclosure / نماذج بلا إفصاح مفيد عن بيانات التعليمات
- **SILMA AI** — **no technical report exists.** `silma-ai/SILMA-9B-Instruct-v1.0` (license `gemma`, built on Gemma-2-9B) discloses only PII filtering and generic bias language: **no dataset names, sizes, or dialect composition.** The org releases evaluation and embedding data instead: `silma-arabic-broad-benchmark` (**470** human-validated questions from 64 datasets, 22 categories), `silma-rag-qa-benchmark-v1.0` (17 bilingual datasets), `silma-arabic-triplets-dataset-v1.0` (**2,280,319** embedding triplets — **not** instruction data). | **لا تقرير تقني ولا إفصاح.**
- **Yehia** (`Navid-AI/Yehia-7B-preview`, Apache-2.0) — base `ALLaM-7B-Instruct-preview`, trained with **GRPO**, judge/reward = **Claude Sonnet 3.5**, reward criteria = the **AraGen 3C3H** metric. **No dataset names, no counts, no data release.** Worth flagging for circularity: it is optimised directly against a public Arabic *benchmark's* metric. | **بلا أسماء بيانات ولا أعداد**، ومدرَّب مباشرةً على مقياس معيار عام — دورية تستحق التنبيه.
- **Falcon-Arabic / Falcon-H1-Arabic (TII)** — **blog posts only, no paper.** Falcon-Arabic claims the most explicit anti-MT stance of any Arabic model — *"trained on **high-quality, 100% native Arabic datasets, avoiding the use of machine-translated content** to minimize cultural bias and preserve linguistic authenticity"* — **with zero numbers and no data release**, so the claim is unverifiable. Falcon-H1-Arabic (3B/7B/34B, hybrid Mamba-Transformer, Jan 2026): SFT = "high-quality Arabic instructions, curated long-context examples, and structured reasoning tasks" + a "targeted DPO phase"; no counts, no dialect breakdown. OALL: 7B 71.47%, 34B 75.36%. | **أقوى موقف معلن ضد الترجمة الآلية لكنه بلا أرقام ولا بيانات منشورة، فغير قابل للتحقق.**
- **ArabianGPT** (riotu-lab, [arXiv:2402.15313](https://arxiv.org/abs/2402.15313)) — **no instruction tuning at all.** Pretrain-only GPT-2-style Arabic models (0.1B/0.3B) with the AraNizer tokenizer; the only fine-tuning is task-specific (sentiment on AJGT 1,800 tweets, summarization on xlsum[1000], QA on mkqa-1000). The group's instruction-adjacent release is `riotu-lab/ArabicQA_2.1M` (§1, rank #15). | **لا ضبط بالتعليمات إطلاقًا.**
- **Mistral Saba** (24B, Feb 2025) — API-only, closed; "meticulously curated datasets from across the Middle East and South Asia" is the entire disclosure. Cite as closed.
- **PHOENIX / NOON** — both combined translated Alpaca/Dolly with Arabic-specific content; **both instruction datasets remain closed.** CIDAR's Table 4 gives the counts: Noon Instructions **110,000** (Naseej), Phoenix Instructions **8,000**, and for comparison Jais Instructions **3,683,144**, AceGPT Instructions **363,155**, AlGhafa Instructions **1,459,000** — all closed. | **مغلقة**، وأعدادها من جدول CIDAR.
- **Names that do not correspond to real Arabic post-training work / أسماء لا تقابل عملًا حقيقيًا:** **"Arabic-Nemotron" does not exist** (Arabic is merely one of 15 languages in NVIDIA's general multilingual Nemotron pretraining data; `nvidia/Nemotron-Post-Training-Dataset-v1/v2` are general-purpose). **"Sawt" is not an LLM** (SawtArabi is an Interspeech 2025 **TTS** benchmark; Sawt is a Saudi voice-agent startup). **Munsit is an ASR model** ([arXiv:2508.08912](https://arxiv.org/abs/2508.08912), 15,000 h weakly-labelled MSA+DA) — out of scope. **NOOR (TII, 2022)** has a press release only, no technical report and no instruction tuning. **K2 / K2-V2 (LLM360)** optimise tokenizer fertility for Arabic but describe **no Arabic-specific SFT data**; **Ansari Chat** is a RAG assistant over 18,000+ pages of Qur'an/Hadith/Fiqh, i.e. retrieval, not instruction tuning.

### §6 summary table — model SFT/preference mixture sizes / جدول أحجام خلطات النماذج

| Model | SFT size | Arabic share | Preference stage | Data released? |
|---|---:|---|---|:--:|
| Jais 1 | 9,699,900 | **3,683,144** (≈60% MT'd; ~17K native human) | none (chosen-only HH-RLHF) | ✖ |
| Jais family / adapted | ~14M | ~4M | not stated | ✖ |
| Jais 2 | >20M | not stated | **>200K DPO pairs** | ✖ |
| AceGPT | **629,293** (363,155 unique) | all Arabic-facing | 24K RM pairs + PPO on 30K prompts | ✅ **`FreedomIntelligence/*`** |
| AraLLaMA (ALAN) | **733,419** + AceGPT sets | all Arabic | none | ✖ |
| ALLaM | 12M → **~6M** after filtering | ~50% | **245K** DPO (from 25,854 human-verified triplets) | ✖ |
| Fanar 1.0 | **4,434,000** (3.6M + 834K) | not stated | **250K** DPO (~20% on-policy) | ✖ |
| Fanar 2.0 | **3,985,215** (+1.8M rebalancing, +54,321 long-context) | not stated; **250K native Arabic reasoning traces** | **280K** DPO | ✖ |
| Atlas-Chat | **458,285** | all Darija (~35% MT'd) | none | ✅ **ODC-BY** |
| Nile-Chat | **1,817,288** (+35,838 test) | all Egyptian, ~25% Arabizi | **298,073** DPO | ✅ (no license field) |
| NileChat | **2,273,809** | EGY + MOR + MSA + fr/en | none | ✖ (pretraining data ✅ CC-BY-NC) |
| GemMaroc | **~49,000** (~37.4K Darija) | Darija + EN | none | ✅ |
| SmolKalam | **1,777,275** / 1,545,742 | all Arabic | n/a (dataset, not model) | ✅ **CC BY 4.0** (gated) |

---

## §7 — Papers on Arabic instruction-data *quality* (the translation problem) / أبحاث جودة بيانات التعليمات العربية

This is the theme the project should take most seriously: **most Arabic SFT data is translated English, and
most of it fails a standard quality bar.** The evidence, in descending order of usefulness:
| هذا هو المحور الذي ينبغي أخذه بأقصى جدّية: **معظم بيانات الضبط العربية إنجليزية مترجمة، ومعظمها يسقط عند
أي عتبة جودة قياسية.**

1. **Creating Arabic LLM Prompts at Scale** ([arXiv:2408.05882](https://arxiv.org/abs/2408.05882)) — the strongest **quantitative** result. Two pipelines: manual PromptSource-style templates over **78 Arabic NLP datasets** → **67,488,303** prompts; and translation of PromptSource (43 datasets) + Super-NaturalInstructions (320 datasets) → ~**19.9M** prompts. Total **>87M prompts / 67 tasks**. Quality control: **reference-free COMET-QE with threshold 0.7, which discards ~80% of translated prompts — only ~20% survive.** Documented failure modes: mistranslated technical terms ("passage" → "walkway") and English-specific grammar tasks that are untranslatable in principle. Fine-tuning Qwen2-7B: ROUGE-L 0.184 at 800K samples (+29% over base), 0.224 at 8M (+57%). License CC BY-NC-ND 4.0. | **أقوى نتيجة كمّية**: عند عتبة COMET-QE = 0.7 يسقط نحو **80%** من الأسئلة المترجمة.
2. **Mind the Gap: A Review of Arabic Post-Training Datasets and Their Limitations** ([arXiv:2507.14688](https://arxiv.org/abs/2507.14688), ArabicNLP 2025) — surveys **366** Arabic post-training datasets on HuggingFace. Domain split: Translation **155 (42.3%)** · Q&A **140 (38.3%)** · Summarization **45 (12.3%)** · Reasoning & multi-step **8 (2.2%)** · Robustness & safety **8 (2.2%)** · Dialog **6 (1.6%)** · Cultural alignment **3 (0.8%)** · **Function calling, code generation, persona/system prompts: 0.** Documentation: **85.61%** of Q&A datasets have no peer-reviewed publication or DOI; **38.64%** unmaintained beyond 12 months; community adoption **<2%** high across all categories (people rebuild rather than reuse). Verdict: *"Native Arabic content should be favored to avoid loss of context, nuance, or cultural misalignment present in translated material."* | مسح 366 مجموعة: صفر لاستدعاء الأدوات وتوليد الشيفرة، و0.8% فقط للتوافق الثقافي، و85.61% بلا نشر محكَّم.
3. **SmolKalam** ([arXiv:2511.18411](https://arxiv.org/abs/2511.18411)) — the constructive answer: ensemble MT + reward-model ranking + LR/SCR filtering. Thesis: *"naive translation can work at the pretraining scale, but post-training demands much higher quality."* | الردّ البنّاء: ترجمة تجميعية مع ترتيب بنموذج مكافأة وترشيح بمؤشري اللغة والخط.
4. **Improving Language Models Trained on Translated Data with Continual Pre-Training and Dictionary Learning Analysis** (Boughorbel et al., [ACL 2024 ArabicNLP](https://aclanthology.org/2024.arabicnlp-1.7/)) — translated TinyStories (2.2M stories, NLLB-3B) into Arabic, trained 1M–33M-parameter models, identified concrete translation defects and propagated cultural bias, then showed that continual pretraining on **just 1%** of data volume as LLM-synthesized *native* Arabic stories repairs them, validated by GPT-4 judging and dictionary-learning analysis. **Practical implication: a small native injection buys a disproportionate amount of repair.** | **حقنة أصيلة صغيرة (1%) تُصلح قدرًا غير متناسب من العطب.**
5. **CIDAR** ([arXiv:2402.03177](https://arxiv.org/abs/2402.03177)) — the most-cited paper for "translated data is culturally misaligned," and it reports a hard number (**64.5%** of translated pairs needed human modification), but its model-level evidence is **qualitative only**. Verbatim: *"they use a lot of **machine-translated or machine-generated instruction datasets without further human review or audit, disregarding the consequences of using such poor, distorted, and misaligned instructions**"* and *"the conventional approach of fine-tuning on machine-generated or machine-translated datasets has often resulted in **biases favoring Western cultural nuances**."* Its annotator guidelines are directly reusable: *"Some words might not be translated correctly, especially at the beginning of each instruction… we should replace [summary] with [summarize]"*; *"the name **John Smith** should be replaced…"*. Its own stated limitation: *"**Localization of data was limited to corrections of the translated text, which is mostly written in MSA, without incorporating multiple dialects.**"* The authors also *"opted out of topics related to religion as it is considered a sensitive topic in the region."* CIDAR's **Table 4** doubles as a ready-made census of Arabic instruction datasets with Arabic-instruction counts and open/closed status: xP3 2,148,955 (open) · xP3x 18,246,158 · MSIFT 114,231 · SUP-NATINST 80,396 · MITD 81,451 · Bactrian-X 67,017 · alpaca-arabic-instruct 52,002 · OASST1 666 · **closed:** Jais 3,683,144 · AceGPT 363,155 · AlGhafa 1,459,000 · Noon 110,000 · Phoenix 8,000. | الأكثر استشهادًا للدعوى الثقافية برقم 64.5%، لكن أدلته على مستوى النموذج نوعية فقط؛ وجدوله الرابع تعداد جاهز للمجموعات المفتوحة والمغلقة.
5b. **Palm** ([arXiv:2503.00151](https://arxiv.org/abs/2503.00151)) — 17,411 fully human-authored items from 44 native speakers across 22 countries and 10 dialects; its opening example ("**suggested having a beer after prayer**") is the single most vivid demonstration of translated-data cultural failure, and Fanar 2.0 adopts the set for cultural alignment. | 17,411 مادة بشرية التأليف من 44 متحدثًا أصليًا في 22 دولة و10 لهجات، وتتبنّاها Fanar 2.0 للتوافق الثقافي.
6. **Hala** ([arXiv:2509.14008](https://arxiv.org/abs/2509.14008)) — the pragmatic counter-position: rather than filtering, **tune the translator itself** (LFM2-1.2B) on instruction-style text, and drop the categories that translate worst (code). | الموقف العملي المقابل: اضبط المترجم نفسه، واحذف الفئات الأسوأ ترجمةً (البرمجة).
7. Adjacent: **InstaTrans** ([arXiv:2410.01512](https://arxiv.org/abs/2410.01512)), instruction-aware translation for non-English instruction datasets — methodologically relevant, not Arabic-specific.

**Quotable evidence, ranked by usability / الشواهد مرتَّبة بحسب قابلية الاستشهاد:**
CIDAR (*"poor, distorted, and misaligned instructions"*) > Fanar 1.0 (English names/places/traditions leak
into Arabic through translation) > Palm (the "beer after prayer" example) > NileChat (*"retains the source
language's cultural perspective"*, plus an explicit self-critique that its own SFT was translated) > AceGPT
(*"these responses are in Western values and not localized"*) > Fanar 2.0 (translated reasoning traces
introduce *"language-mixing artifacts"*) > SmolKalam (*"without post-translation quality filtering"*) >
ALLaM (a single filter clause) > Falcon-Arabic (*"100% native… avoiding machine-translated content"* — the
strongest claim, but with **no numbers and no data release**, so unverifiable).
| ترتيب الشواهد: CIDAR ثم Fanar 1.0 ثم Palm ثم NileChat ثم AceGPT ثم Fanar 2.0 ثم SmolKalam ثم ALLaM، وأخيرًا
Falcon-Arabic (أقوى دعوى وأضعف إثبات).

**Reference mixtures built entirely from public parts / خلطات مرجعية من مكوّنات عامة بالكامل:**
**Arabic Stable LM** ([arXiv:2412.04277](https://arxiv.org/abs/2412.04277)) documents a **677,746**-example
Arabic SFT mixture assembled purely from public components — rephrased-synthetic **182,505** (Qwen2-7B-Instruct)
+ **InstAr-500k 481,281** + **Aya Arabic 13,960** — and is a useful published baseline to compare your own
mixture against. | مثال منشور لخلطة عربية من 677,746 مثالًا مبنية بالكامل من مكوّنات عامة.

**Notes on non-existent resources / تنبيهات على موارد غير موجودة:** "TARJAMAT" as an Arabic instruction-data
quality paper/dataset **does not exist**. "**Arabic-Nemotron**" does not exist. "**Sawt**" is not an Arabic
LLM. There is **no Arabic Magpie dataset**. And **arXiv:2409.12058 is a Finsler-geometry mathematics paper,
not the Jais family report** — a citation error that circulates widely. Do not cite any of these.
| **موارد غير موجودة**: TARJAMAT، وArabic-Nemotron، وSawt كنموذج لغوي، ونسخة عربية من Magpie؛ وarXiv:2409.12058
بحث رياضيات لا تقرير Jais — وهو خطأ استشهاد شائع.

---

## 📊 Comparison table — verified numbers only / جدول المقارنة — أرقام محقَّقة فقط

| # | Dataset | Rows (real `/size`) | Method | MSA / dialect | License | Sampled? | Verdict |
|---|---|---:|---|---|---|:--:|---|
| 1 | `AdaMLLab/smolkalam-arabic-conversational-sft` (SmolKalam) | 1,790,478 (24 cfg) | ensemble MT + RM filtering | MSA | apache-2.0 (mirror); CC BY 4.0 source | ✅ | **Best backbone** |
| 2 | `FreedomIntelligence/Quora-Arabic-GPT4` | 43,050 | native prompts + GPT-4 | MSA + colloquial | apache-2.0 | ✅ | **Best prompt distribution** |
| 3 | `arbml/CIDAR` | 10,000 | MT + full human edit | MSA | apache-2.0 / CC BY-NC (conflict) | ✅ | **Cleanest per row** |
| 4 | `MBZUAI-Paris/Egyptian-SFT-Mixture` | 1,817,288 (+35,838 test) | native + Claude 3.7 translation | Egyptian, Arabic+Latin script | **missing** | ✅ | Best dialect mixture |
| 5 | `MBZUAI-Paris/Darija-SFT-Mixture` | 458,285 | native resources re-templated + QC'd MT | Moroccan Darija | ODC-BY (mixed, some NC) | ✅ | Task-heavy |
| 6 | `hammh0a/Hala-4.6M-SFT` | 4,060,575 | tuned translator (Hala-1.2B) | MSA | **CC-BY-NC-4.0** | ✅ | STEM depth, NC |
| 7 | `FreedomIntelligence/Evol-Instruct-Arabic-GPT4` | 69,997 | MT + GPT-4 | MSA | apache-2.0 | ✅ | MT fusion defects |
| 8 | `MBZUAI/Bactrian-X` (`ar`) | 67,017 | Google Translate + ChatGPT | MSA | **CC-BY-NC-4.0** | ✅ | Solid, dated, NC |
| 9 | `FreedomIntelligence/Alpaca-Arabic-GPT4` | 49,969 | MT + GPT-4 | MSA | apache-2.0 | ✅ | Clean but shallow |
| 10 | `HeshamHaroon/Arabic_Function_Calling` | 50,810 | synthetic | MSA + 3 dialects | apache-2.0 | ✅ | Only real tool set |
| 11 | `alielfilali01/ultrafeedback-arabic` | 63,135 | MT of UltraFeedback | MSA | **missing** | ✅ | Preference, needs cleaning |
| 12 | `MBZUAI-Paris/Egyptian-DPO-Mixture` | 298,073 | on/off-policy + Claude 3.5 | Egyptian | **missing** | ✅ | Best Arabic DPO |
| 13 | `ClusterlabAi/InstAr-500k` | 481,281 | templated native corpora + Command R+ | MSA only | apache-2.0 (C4AI conflict) | ✅ | **Not chat data** |
| 14 | `2A2I/Arabic-OpenHermes-2.5` | 981,618 | unfiltered MT | MSA | apache-2.0 | ✅ | **Structurally broken** |
| 15 | `CohereLabs/aya_collection_language_split` (10 ar cfg) | ≈39,764,447 | templated + MT fan-out | 10 "varieties" (MT-derived) | apache-2.0 | ✅ | **Fake dialect depth** |
| 16 | `CohereLabs/aya_dataset` (Arabic rows) | 13,960 | **human-written** | ary/arb/arz/ars/acq/apc | apache-2.0 | ⚠️ (filter API erroring) | Only real human pool |
| 17 | `2A2I/Arabic_Aya` | 41,472,592 | Arabic re-slice of Aya | mixed | apache-2.0 | ✖ | Same caveats as #15 |
| 18 | `riotu-lab/ArabicQA_2.1M` | 2,141,146 | aggregated Arabic QA | MSA | apache-2.0 | ✅ | **Field misalignment** |
| 19 | `akbargherbal/six_millions…` | 6,372,734 | undocumented | mixed | **none** | ✅ | **Encoding damage** |
| 20 | `FreedomIntelligence/AceGPT-v2-AlignmentData` | 3,222,135 | GPT-4-turbo rewriting | MSA | apache-2.0 | ✅ | Cleaner-training, not chat |
| 21 | `FreedomIntelligence/Arabic-preference-data-RLHF` | 11,548 | RLAIF over native prompts | MSA | **none (README 404)** | ✅ | Small, good signal |
| 22 | `Omartificial…/Arabic_Reasoning_Dataset` | 9,210 | synthetic CoT | MSA | apache-2.0 | ✅ | Repetitive but clean |
| 23 | `HeshamHaroon/saudi-dialect-conversations` | 3,545 | natively composed synthetic | Najdi | apache-2.0 | ✅ | Excellent, tiny |
| 24 | `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` | 31,846 | MT of ORPO mix | MSA | **none** | ✅ | Weak preference signal |
| 25 | `2A2I/argilla-dpo-mix-7k-arabic` | 7,500 | MT | MSA | mit | ✅ | **Roles translated — broken** |
| 26 | `MohAlbrayh/saudi-allam-sft-dataset-2M` | **4,504** | translated Alpaca | Hijazi/Najdi labels | **none** | ✅ | Name is misleading |
| 27 | `FreedomIntelligence/sharegpt-arabic` | 5,231 | MT of ShareGPT | MSA | apache-2.0 | ✅ | Tiny; `lang: en` on every row |
| 28 | `arbml/alpaca_arabic` / `okapi_arabic` / `alpagasus_cleaned` | 52,002 / 64,669 / 9,229 | MT | MSA | **none** / **none** / **none** | ✖ | Legacy |
| 29 | `HeshamHaroon/oasst-arabic` | 88,836 | MT of OASST | MSA | **none** | ✖ | Undocumented |
| 30 | `2A2I/H4_no_robots` | 10,000 | Google Cloud Translation | MSA | **CC-BY-NC-4.0** | ✖ | Human source, translated |
| 31 | `2A2I/Aya-*-DPO` (4 repos) | 12,960–14,210 | human vs. model | MSA + dialects | apache-2.0 | ✖ | Clean small DPO |
| 32 | `TuwaiqAcademy/AISA-ArabicFC` | 12,220 | shared task | MSA + 4 dialects | apache-2.0 | ✖ | Tool calling |
| 33 | `atlasia/darija-dpo-negatives` | 69,854 | — | Darija | other | ✖ | Dialect DPO |
| 34 | `multilingual/orca_dpo_pairs` | 64,656 | MT | MSA | apache-2.0 | ✖ | Arabic column |
| 35 | `Mohaddz/arabic-sft-mix` / `-2` | 2,253,127 / 2,371,371 | re-mix, no card | mixed | **none** | ✖ | Provenance unknown |
| 36 | `UBC-NLP/palm` | 17,411 (15,485 + 1,926) | **44 human annotators, 22 countries** | 10 dialects | **CC-BY-NC-ND-4.0** | ✖ **gated** | Best native cultural set; **no derivatives** |
| 37 | `FreedomIntelligence/Code-Alpaca-Arabic-GPT4` | 20,022 (paper) | MT + GPT-4 | MSA | apache-2.0 | ✖ | Only Arabic code-instruction set; inherits family MT issues |

Domain-specific sets encountered but out of scope: `Ahmed-Selem/Shifaa_Arabic_Medical_Consultations`
(84,422) · `Mahmoud22/medical-o1-reasoning-SFT-Arabic` (19,704) · `alielfilali01/Goud-Sum-Instruct`
(158,282, Darija summarization) · `MBZUAI-Paris/MoroccanWikipedia-QA` (34,351) ·
`arbml/arabic_empathetic_conversations` (36,628) · `sadeem-ai/arabic-qna` (6,030) ·
`Kamyar-zeinalipour/Arabic-Clue-Instruct` · `dispatchAI/arabic-poetry-instructions`.

---

## 🚫 Access limitations / حدود الوصول

Recorded explicitly rather than guessed. | مسجَّلة صراحةً لا مُخمَّنة.

- **`SultanR/smolkalam` — gated (`auto`).** "You have to accept the conditions to access its files." Could not be sampled directly; the **ungated mirror `AdaMLLab/smolkalam-arabic-conversational-sft` (1,790,478 rows, apache-2.0) was sampled instead** and carries the same `LR`/`SCR`/`rank_score`/`source` columns. | مقيَّدة؛ عوينت النسخة المرآة غير المقيَّدة بدلًا منها.
- **`ArSyra/arsyra-instruction-tuning` — gated (`manual`) and paywalled.** The dataset-viewer returns "does not exist, or is not accessible without authentication." Vendor-reported at **11,719 rows** with a public **50-record preview**; claims 7 dialect groups across 17 countries; preview is CC-BY-NC-SA-4.0 and the full set is sold under commercial/academic licences from ~$29. **All figures are vendor-reported and unverified; no peer review, no independent evaluation.** | مقيَّدة يدويًا ومدفوعة؛ كل أرقامها من البائع وغير محقَّقة.
- **`neulab/PangeaInstruct` — dataset viewer disabled** ("dataset viewer is disabled in configuration"). Could not sample. It is multimodal VQA instruction data in any case. | عارض البيانات معطَّل؛ تعذّرت المعاينة.
- **`CohereLabs/aya_dataset` — per-language filtering unavailable.** The `/filter` endpoint returned no rows for `language_code='arb'` (DuckDB index error), and the dataset was **renamed** from `CohereForAI/aya_dataset` (old ID now errors "The dataset has been renamed"). The Arabic per-variety counts quoted here come from the dataset's own published statistics, not from rows I pulled. | تعذّر الترشيح حسب اللغة؛ والأرقام مأخوذة من إحصاءات المجموعة المنشورة لا من صفوف سُحبت.
- **`Yasbok/Alpaca_arabic_instruct` — viewer persistently unavailable.** `/size` and `/rows` both returned "The server is busier than usual" across three separate attempts and two batches. Row count unknown. | العارض غير متاح باستمرار عبر ثلاث محاولات؛ عدد الصفوف مجهول.
- **`2A2I/Arabic_Aya` — config name `aya_dataset` not resolvable** ("Not found"); the repo has 29 configs and the naming was not discoverable from the size endpoint alone. Row total (41,472,592) is from `/size`. | تعذّر حلّ اسم التهيئة؛ الإجمالي من `/size`.
- **`PetraAI/PetraAI` — zero configs resolve** through the viewer despite a `1M<n<10M` tag. Unverifiable. | لا تُحلّ أي تهيئة؛ غير قابلة للتحقق.
- **Transient HTTP 429s** from both `datasets-server.huggingface.co` and `huggingface.co` during heavy sampling required retry-with-backoff; a small number of individual offsets (`ArabicQA_2.1M` p=20,000 in one pass, `AceGPT-v2-AlignmentData` p=0) failed and were re-run. All figures reported above come from successful calls. | استلزم الضغطُ إعادةَ المحاولة مع تراجع زمني؛ وكل الأرقام أعلاه من نداءات ناجحة.
- **`UBC-NLP/palm` — gated (`auto`).** The dataset-viewer returns "does not exist, or is not accessible without authentication," so it could not be sampled. Size (**15,485 train + 1,926 test = 17,411**) and schema come from the repo's `dataset_info`; license **CC-BY-NC-ND-4.0** (**no derivatives** — it cannot legally be mixed into a redistributed training set). | مقيَّدة؛ الحجم والمخطط من بيانات المستودع، والرخصة تمنع الاشتقاق.
- **`QCRI/FanarGuard`** (the 4B safety model trained on 468K pairs) is **gated**; the 468K pair dataset itself is not published. Several `inceptionai` `jais-*-chat` repos are now **gated** as well. | مقيَّدة، وبيانات الـ468 ألف زوج غير منشورة.
- **Closed by publisher (no dataset repo at all) / مغلقة من الناشر (لا مستودع أصلًا):** Jais 1 SFT mixture (3,683,144 Arabic pairs) · Jais family SFT (~4M Arabic) · Jais 2 IFT (20M+) and DPO (200K+) · ALLaM "Ultra-Instinct" (12M→6M) and its 245K DPO set · Fanar 1.0 (4.43M SFT + 250K DPO) and Fanar 2.0 (3,985,215 SFT + 280K DPO, 250K native Arabic reasoning traces, 468K FanarGuard pairs) · SILMA instruction data · Yehia training data · Falcon-Arabic / Falcon-H1-Arabic post-training data · Command A / Command R7B Arabic / Aya Expanse post-training mixtures · AraLLaMA's ALAN (733,419) · NileChat's SFT mixture (2,273,809 — its *pretraining* data is released) · PHOENIX (8,000) · NOON (110,000) · AlGhafa (1,459,000) · the four Arabic tool-calling datasets of arXiv:2509.20957 · the 5,466-pair Saudi-Dialect-ALLaM set. **In aggregate, well over 40M Arabic instruction pairs described in the literature are unavailable.** | **إجمالًا، أكثر من 40 مليون زوج تعليمات عربية موصوفة في الأدبيات غير متاحة.**
- **Licence absent on the Hub** for, among others: `Egyptian-SFT-Mixture`, `Egyptian-DPO-Mixture`, `ultrafeedback-arabic`, `Arabic-preference-data-RLHF`, `arcee-globe/arabic-orpo-dpo-mix-40k-filtered`, `akbargherbal/six_millions…`, `Mohaddz/arabic-sft-mix{,-2}`, `MohAlbrayh/saudi-allam-sft-dataset-2M`, `arbml/alpaca_arabic`, `arbml/okapi_arabic`, `HeshamHaroon/oasst-arabic`, `Yasbok/Alpaca_arabic_instruct`. **Roughly a third of the ecosystem ships with no licence at all** — treat these as research-only. | **نحو ثلث المنظومة بلا رخصة**؛ عاملها كبحثية فقط.

---

*Survey date / تاريخ المسح: 2026-08-24. Every row count marked "real" was pulled live from the HuggingFace
`datasets-server` `/size` endpoint; every Arabic quotation above is verbatim from a real sampled row and
nothing is invented. Where a figure could not be verified it is written as "not stated" or flagged as
vendor-reported. | كل عدد صفوف موسوم بأنه حقيقي مسحوب مباشرةً من واجهة `/size`، وكل اقتباس عربي أعلاه منقول
حرفيًا من صفّ حقيقي، ولا شيء مختلَق؛ وما تعذّر التحقق منه موسوم بذلك.*
