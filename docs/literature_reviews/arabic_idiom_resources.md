# Arabic Idiom & Proverb (أمثال / كنايات) Resources — A Literature & Dataset Survey
# 阿拉伯语习语与谚语（أمثال / كنايات）资源综述

> **Purpose / 目的:** identify sources for an Arabic idiom knowledge base to drive the
> continued-pretraining pipeline, matching what already exists for English (~21K idioms),
> Chinese (~28K chengyu) and Hindi (16,617 proverbs).
> 为持续预训练流水线寻找阿拉伯语习语知识库来源，对标已有的英语（约 21K）、中文（约 28K 成语）与印地语（16,617 条谚语）。

> **Target schema (acceptance criterion) / 目标字段（验收标准）:**
> `idiom | entities | literal_meanings | figurative_meanings | examples | source/derivation`
> Any field with no valid value is written **`NAN`**. / 无有效值的字段一律写 **`NAN`**。

*Survey date / 调研日期: 2026-07-23. Four parallel sub-agents; every dataset marked "verified"
below was actually downloaded and its rows counted, not read off a card. Arabic text is quoted
verbatim; nothing is invented. / 四个子代理并行调研；下文标注"实测"的数据集均为真实下载并逐行统计，
非摘自数据卡。阿拉伯语原文均为逐字引用，无任何编造。*

---

## ⭐ Headline findings / 核心结论

1. **No multilingual idiom KB contains Arabic.** IdiomKB (en/zh/ja only), MABL (hi/id/jv/kn/su/sw/yo),
   MAPS (en/de/ru/bn/zh/id) — all verified to exclude Arabic. There is nothing to extend.
   | **没有任何多语习语知识库包含阿语**：IdiomKB、MABL、MAPS 均实测确认无阿语，无现成基础可扩展。
2. **The publishing gap is severe.** The largest *published* Arabic idiom resource (AIPSeLEX, 32,785
   collected) was **never released** — the promise dates to 2015. Jawaher publishes 10,037 proverbs but
   releases **1,017**. Five separate resources say "available upon acceptance."
   | **"论文有、数据无"极其普遍**：最大的已发表资源 AIPSeLEX（32,785 条）**从未发布**（2015 年至今）；
   Jawaher 论文 10,037 条、公开仅 **1,017** 条；共五个资源承诺"录用后发布"。
3. **Realistic downloadable total ≈ 11K rows, < 10K after dedup** — roughly *half* our Hindi coverage
   and *a third* of Chinese. Modern NLP datasets alone **cannot** reach parity.
   | **今天能下载的合计约 1.1 万条，去重后不足 1 万** —— 约为印地语的一半、中文的三分之一。仅靠现代 NLP
   数据集**无法**达到与其他语言同等规模。
4. **But the classical layer closes the gap, and needs NO OCR.** `AuthenticIlm/Shamela4_Full_DB` ships
   born-digital (not scanned) text of the canonical *amthal* compendia — **≈10,400 numbered proverb
   entries**, each already carrying an origin story and a `يضرب...` figurative gloss.
   | **但古典层能补齐缺口，且完全无需 OCR**：`AuthenticIlm/Shamela4_Full_DB` 提供四大 amthal 汇编的
   **原生数字文本**（非扫描件），**约 10,400 条编号词条**，且每条自带典故来源与 `يضرب...` 比喻义。
5. **Three schema columns must be built by us.** Across *every* Arabic resource found:
   `examples` ≈97% missing, `literal_meanings` ≈98% missing, `entities` **100% missing**.
   | **三列必须我们自建**：所有阿语资源中 `examples` 约 97% 缺失、`literal_meanings` 约 98% 缺失、
   `entities` **100% 缺失**。

---

## 📊 Field coverage against the target schema / 目标字段覆盖对照表

Verified counts. ✅ present · ⚠️ partial/derivable · ❌ absent (→ `NAN`)
/ 实测统计。✅ 有 · ⚠️ 部分或可推导 · ❌ 无（记 `NAN`）

| Resource | n | idiom | literal | figurative | examples | entities | source | Variety |
|---|---:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| `tahaalselwii` classical | 4,543 | ✅ | ❌ | ✅ (ar) | ❌ | ❌ | ⚠️ file-level | Classical/MSA |
| `tahaalselwii` colloquial | 3,187 | ✅ | ❌ | ✅ (ar) | ❌ | ❌ | ⚠️ file-level | Egyptian |
| **Jawaher** (public slice) | 1,017 | ✅ | ❌* | ✅ **ar+en** | ❌ | ❌ | ❌ | **20 varieties** |
| Tunisian proverbs | 999 | ✅ | ⚠️ MT-corrupt | ✅ (ar) | ❌ | ❌ | ❌ | Tunisian |
| Absher (proverbs+phrases) | 561 | ✅ | ⚠️ inline | ✅ (ar) | ⚠️ in MCQ | ❌ | ❌ | Saudi ×5 |
| **Kinayat** | 325 | ✅ | ❌ | ✅ (ar) | ✅ **150/325** | ❌ | ❌ | Egyptian |
| Hassaniya amthal | 319 | ✅ | ❌ | ✅ (ar) | ❌ | ❌ | ⚠️ book | Mauritanian |
| Wiktextract `ar` (strict idioms) | ~140 | ✅ | ⚠️ per-sense | ✅ | ✅ ~17 | ❌ | ❌ | MSA+dialects |
| Wiktextract `ar` (MWE set) | 1,828 | ✅ | ✅ many | ⚠️ | ✅ 133 | ❌ | ❌ | MSA+dialects |
| **Shamela classical (extract)** | ≈10,400 | ✅ | ❌ | ✅ `يضرب` ~4,350 | ❌ | ❌ | ✅ **per-row story** | Classical |

\* The Jawaher **paper** reports 2,205 literal translations, but the public HF release **drops that
column entirely**. / Jawaher 论文称有 2,205 条字面翻译，但公开的 HF 版本**完全没有该列**。

**Nothing anywhere provides `entities`** — it is the genuinely novel column and must be extracted
(NER over the explanation text, which is rich in names: `براقش`, `عَمْرو بن الأهتم`, `الزِّبْرِقَان بن بدر`).
/ **所有资源都没有 `entities`** —— 这是需要我们从释义文本中抽取的全新列（释义中人名地名密集）。

---

## 1. Downloadable Arabic idiom/proverb datasets / 一、可下载的阿语习语谚语数据集

### 1.1 `tahaalselwii/arabic-proverbs-collection` — largest downloadable inventory / 现存最大可下载库
- **Link:** https://huggingface.co/datasets/tahaalselwii/arabic-proverbs-collection · **CC BY 4.0**, not gated
- **Size (verified) / 规模（实测）:** **8,305** rows = classical **4,543** + colloquial **3,187** + popular **575**
- **Provenance / 来源:** classical ← **مجمع الأمثال** (al-Maydani, d. 1124); colloquial ← **الأمثال العامية**
  (Aḥmad Taymūr Pasha, 1949, Egyptian). | 古典部分源自迈达尼《谚语集成》，方言部分源自泰穆尔《埃及俗谚》。
- **Fields / 字段:** only `Arabic Proverb`, `Explanation` (Arabic prose, median 135–212 chars).
- **⚠️ Discard `popular_arabic_proverbs.csv` (575) — AI-generated**; the card itself warns entries "may
  contain inaccuracies… or statements that are not commonly recognized as authentic Arabic proverbs."
  Usable human-authored total = **7,730**. | **须丢弃 popular 一档（575 条，AI 生成）**，数据卡自承可能
  包含不实内容；可用的人工撰写条目为 **7,730** 条。
- **Verbatim / 实例:**
  `إنَّ دَوَاءَ الشَّقِّ أنْ تَحُوصَهُ` → `الْحَوْصُ:الخياطةُ يضرب في رَتْق الفَتْق وإطفاء النائرة`
  *(“the cure for a tear is to stitch it” — used for mending a rift; note the embedded literal gloss `الحوص=الخياطة`)*
- **Verdict / 结论:** **USABLE, needs enrichment.** Split long classical explanations into figurative
  meaning + origin story; generate literal/examples/entities.

### 1.2 `UBC-NLP/Jawaher-benchmark` — best per-row quality / 单条质量最佳
- **Link:** https://huggingface.co/datasets/UBC-NLP/Jawaher-benchmark · paper arXiv:2503.00231 (NAACL 2025)
- **Size (verified) / 规模（实测）:** `Jawaher_train_fixed.jsonl` **817** + `Jawaher_test.jsonl` **200** = **1,017**
  - **⚠️ Use `_fixed`.** The stale `Jawaher_train.jsonl` has 619 rows and **no `Variety` column**.
    | **务必用 `_fixed`**；旧的 `Jawaher_train.jsonl` 仅 619 行且缺 `Variety` 列。
- **Fields:** `Proverbs, Ar_Explanation, En_Explanation, En_Equivalent, Variety` — figurative meaning in
  **both Arabic and English**, plus an **English idiomatic equivalent**.
- **Varieties (verified, train) / 方言分布（实测）:** MSA 108, EGY 98, UAE 71, ALG 59, SAU 52, TUN 52,
  SUD 47, KUW 39, PAL 37, SYR 33, YEM 31, LIB 30, QAT 26, OMA 26, JOR 25, MAU 25, IRQ 25, BAH 24, LEB 9
  (+MOR in test) = **20 varieties**.
- **🚩 The 90% gap / 90% 的缺口:** the paper reports **10,037** proverbs and **2,205 literal
  translations**; the public release is **1,017** rows **with no literal column**. The remainder is
  claimed at `github.com/UBC-NLP/jawaher` (unverified — GitHub blocked).
  **Fetching that repo by hand is the single highest-value action on this list.**
  | 论文 10,037 条、含 2,205 条字面翻译；公开仅 1,017 条且无字面列。其余据称在 GitHub 仓库（未核实）。
  **手动拉取该仓库是本清单中收益最高的一步。**
- **⚠️ No license declared on the HF repo** — legal review needed. | HF 仓库**未声明许可证**，需法务确认。
- **Verbatim / 实例:**
  `إنك لا تجني من الشوك العنب` → Ar: `يضرب هذا المثل لمن يصنع المعروف في غير أهله…` ·
  En_Equivalent: `You can't make a silk purse out of a sow's ear.`

### 1.3 `menaattia/Kinayat` — the only source with real usage sentences / 唯一自带真实例句
- **Link:** https://huggingface.co/datasets/menaattia/Kinayat · **CC BY 4.0** · paper arXiv:2510.23828 (EACL 2026)
- **Size (verified per column) / 规模（逐列实测）:** **325** rows; `Idiom` 325 · `Ar_Explanation` 325 ·
  `Incorrect_Explanation` 325 · **`full_sentence` 150** · `sentence` (cloze) 150.
  **⚠️ Only 150/325 carry a usage sentence** — an earlier reading that all 325 did was wrong.
  | **仅 150/325 条带例句**，此前"全部带例句"的说法有误。
- **Provenance:** mined from **الكنايات العامية** (Aḥmad Taymūr Pasha, 1949), public domain via Hindawi.
- **⚠️ `Incorrect_Explanation` is a deliberate distractor — never ingest as truth** (it is free
  hard-negative supervision for eval/DPO). | **`Incorrect_Explanation` 是刻意的干扰项，切勿当作正确释义摄入**。
- **Verbatim / 实例:**
  `خَبَرَ أبْيَضْ` → `كناية عن الخبر السيئ، وهو من الأضداد، يريدون به الخبر الأسود.`
  usage: `يا خَبَرَ أبْيَضْ! ايه اللي هو عمله ده.`
- **Verdict:** **USABLE AS-IS** and, more importantly, it is the **gold template** — use its 150 rows as
  few-shot exemplars when LLM-generating `examples` for everything else.
  | 可直接使用；更重要的是它是**黄金模板**——用这 150 条作为为其他资源生成例句时的 few-shot 范例。

### 1.4 Dialect-specific sets / 方言专项集
| Dataset | n | Fields | Variety | License |
|---|---:|---|---|---|
| `HabibaAbderrahim/Tunisian-Proverbs-with-Image-Associations` | **999** | `tunisan_proverb`, `proverb_arabic_explaination`, `context`, `dynamic` (En equivalent) | Tunisian Derja | CC BY 4.0 |
| `Renad10/Absher-Benchmark` | **83 proverbs + 478 phrases** (+2,533 words) | `Term`, `Meaning_of_term`, `Dialect type`, 6 MCQ task files | Saudi (Central/Western/Southern/Northern/Eastern) | CC BY 4.0 |
| `ahmed02mk/amthal-hassaniya` | **319** | Alpaca `input`=proverb / `output`=meaning | Hassaniya (Mauritania) | CC BY 4.0 |

- Tunisian: **discard `caption_formal` / `prompt`** — machine-translated garbage ("graffiti", "Kawai").
  Keep `dynamic` (English functional equivalent). | 突尼斯集须丢弃机翻列，保留 `dynamic`。
- Absher: **dedupe the 6× MCQ replication** down to `Term`+`Meaning`+`Dialect`. Some meanings embed
  **word-level literal glosses** (`أدنى: أقل. حمول: ...`) — a rare literal-meaning signal. The
  *Contextual_Usage* task file yields extractable example sentences.
  | Absher 需把 6 套 MCQ 去重；部分释义内嵌词级字面注解（罕见的字面义信号）；Contextual_Usage 可抽例句。
  **⚠️ Discrepancy:** the paper says data will be released "upon acceptance," yet the HF repo is live and
  was downloaded. | **矛盾点**：论文称"录用后发布"，但 HF 仓库已上线且实测可下载。
- Verbatim / 实例: Hassaniya `ألْبَلْ تبرك على أكبارها` → `يضرب لأهمية الكبار في مجتمعهم وحتمية التبعية لهم`

### 1.5 Wiktionary / Wiktextract — structured, example-bearing, but low idiom yield / 结构化且带例句，但习语产出少
Two verified machine-readable routes (both mirror kaikki.org's Wiktextract):
- **`DataDock/wiktextract`** — raw kaikki JSONL, monthly snapshots. Verified download URL:
  `https://huggingface.co/datasets/DataDock/wiktextract/resolve/main/2026-08-05.jsonl.gz` (2.83 GB gz)
- **`jake-anto/wiktionary`** — same dump flattened to parquet (12.9M senses / 1.5M examples), CC BY-SA 4.0,
  DuckDB-queryable in place over `hf://`; `categories` preserved so you can filter Wiktionary's
  *Arabic idioms* / *Arabic proverbs* categories across `ar`, `arz`, `ary`, `afb`, `apc`, `ajp`.

**Measured Arabic yield (agent streamed all 10,806,865 entries) / 阿语实测产出:**

| metric | count |
|---|---:|
| Arabic entries (`lang_code=="ar"`) | **77,339** |
| Arabic senses | 100,111 |
| Entries with ≥1 usage example | 5,364 |
| Senses tagged `idiomatic` | 105 |
| Category *Arabic idioms* / tagged idiomatic | **107** |
| Category *Arabic proverbs* / `pos=="proverb"` | **33** |
| MWE set (`pos` ∈ phrase/proverb/prep_phrase) | **1,828** (1,827 glossed, only **133** with an example) |

- **Verdict:** the **format is exactly our target schema** (gloss + romanisation + example + English
  translation per sense), but the strict-idiom inventory is tiny (~140). Best used as (a) a
  **structure template** and (b) a reusable **example-sentence pool** (5,364 example-bearing entries).
  | 格式与目标 schema 完全一致，但严格习语仅约 140 条；宜作为**结构模板**与**例句池**使用。
- **Verbatim / 实例:**
  `عدو عدوي صديقي` → *the enemy of my enemy is my friend* · `أهل مكة أدرى بشعابها` → *locals know their own territory best*
  `استجار من الرمضاء بالنار` → *out of the frying pan into the fire*

---

## 2. The classical layer — scale without OCR / 二、古典层：无需 OCR 的规模来源

**This is how Arabic reaches parity with Hindi/Chinese.** `AuthenticIlm/Shamela4_Full_DB` (MIT, 8,604
books) ships **born-digital Shamela text** with full tashkīl and `<span data-type="title">` structural
markup — **not** scanned OCR. Path pattern:
`https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB/resolve/main/<path>/pages.jsonl`

| Book / 书名 | Author (d. AH) | Pages | Numbered entries / 编号词条 | `يضرب` gloss |
|---|---|---:|---:|---:|
| **مجمع الأمثال** Majmaʿ al-Amthāl | al-Maydānī (518) | 5,115 | **4,965** | 2,340 |
| **المستقصى في أمثال العرب** | al-Zamakhsharī (538) | 875 | **3,439** | 1,474 |
| **جمهرة الأمثال** | Abū Hilāl al-ʿAskarī (395) | 1,019 | **1,965** | 541 |
| **فصل المقال في شرح كتاب الأمثال** | al-Bakrī (487) | 530 | 199 | 56 |
| **الأمثال** | Abū ʿUbayd b. Sallām (224) | 357 | 0 (530 `قولهم` anchors) | 91 |
| **الأمثال المولدة** | al-Khwārizmī (383) | 607 | 878 | 9 |
| **زهر الأكم في الأمثال والحكم** | al-Yūsī (1102), Maghrebi | 902 | — | 280 |
| **معجم تيمور الكبير في الألفاظ العامية** | Aḥmad Taymūr (1930) | 1,771 | 4,249 lemma spans | — |

**Top three alone ≈ 10,400 numbered entries** (~4,350 with an explicit `يضرب` figurative gloss).
/ **仅前三部即约 10,400 条编号词条**（约 4,350 条带显式 `يضرب` 比喻义）。

**Verified sample — already `{idiom, origin, figurative meaning}` / 实测样例，天然三段结构:**
```
<span data-type="title" id=toc-115>١١٣- إنْ كُنْتَ رِيحاً فَقَدْ لاَقَيْتَ إِعْصارا</span>
قال أبو عبيدة: الإعصار ريحٌ تهبّ شديدة فيما بين السماء والأرض.        ← origin / 典故
يضرب مثلا للمُدِلّ بنفسه إذا صُلِىَ بمن هو أدهى منه وأشدّ.              ← figurative / 比喻义
```

**Parsing notes / 解析要点:**
- Maydani is near-mechanical: one `<span data-type="title">` per proverb. | 迈达尼近乎机械可解析。
- **Strip Arabic diacritics BEFORE regexing** — `يضرب` is written `يضْرب`; naive matching undercounts
  by ~100×. | **正则前必须先去变音符**，否则漏匹配约 100 倍。
- al-Bakrī / Abū ʿUbayd are continuous prose with no numbering → need an LLM segmenter (Tier 2).
- **Only this layer provides per-row `source/derivation`** (the story behind the proverb) at scale.
  | **只有这一层能大规模提供逐条 `source/derivation`**（典故来源）。

**⚠️ Copyright / 版权:** the medieval *texts* are public domain, but these are 20th-century **critical
editions** (Dār al-Maʿrifa / Muḥyī al-Dīn ʿAbd al-Ḥamīd for Maydani; Iḥsān ʿAbbās 1971 for al-Bakrī;
DKI 1987 for al-Zamakhsharī). The editor's apparatus, vocalization and pagination carry rights; the HF
repo's MIT tag does **not** clear the underlying editions.
| 中世纪原文属公有领域，但这些是 20 世纪校勘本，编者的校勘/注音/分页有版权；HF 的 MIT 标注并不清除底层版本权利。

**Alternative (OCR, lower quality) / 备选（OCR，质量较低）:** `ieasybooks-org/shamela-waqfeya-library`
(MIT, 4,661 books, Google Document AI TXT). **Two-column critical editions come out interleaved** —
the Maydani OCR mixes entry 334's proverb with entry 340's commentary. Single-column dialect books are
clean (e.g. Jayakar's Omani proverbs, which uniquely contains `معناه الحرفى:` = **literal meaning**).
| 双栏校勘本 OCR 会打乱阅读顺序；单栏方言书干净，其中 Jayakar 阿曼谚语集**罕见地自带字面义字段**。

---

## 3. 🚩 Published but NOT downloadable / 三、已发表但拿不到的资源

**This is the defining feature of the Arabic landscape.** / **这是阿语资源版图最突出的特征。**

| Resource | Claimed size | Status |
|---|---|---|
| **AIPSeLEX** (Ibrahim et al., 2015, arXiv:1506.01906) | **32,785 collected / 3,632 annotated** | Never released in 11 years. Fields would have been idiom + English translation + Buckwalter + sentiment. |
| **Jawaher remainder** | 10,037 − 1,017 = **~9,020** | GitHub unverified; public slice also **drops the 2,205 literal translations** |
| **Absher** (arXiv:2507.10216) | 3,094 items | Paper says "upon acceptance" — **but an HF release is live** (see §1.4) |
| **CAMMAR** metaphor set (arXiv:2607.15847) | unstated | "will release upon acceptance" |
| **Wisdom in Unity** (arXiv:2608.08090) | 149 Arabic concepts / 226 instances | "upon publication"; builds on a 25,300-pair multilingual proverb set (OSF, blocked) |
| **Kinayat examples** | 325 idioms, **150** sentences | partial release |
| Al-Mawrid / al-Ghani / al-Waseet / Contemporary Arabic Dictionary | tens of thousands each | privately licensed; SALMA authors state they "obtained a license" |
| Almaany, Lisaan Masry | large | no API, no dump, ToS-restricted |

---

## 4. ⚠️ Traps — look usable, are not / 四、陷阱：看着能用其实不能

1. **`aymansharara/IdiomX`** (MIT, ~175K rows, 12.8K idioms) — advertises Arabic. **The idioms are
   ENGLISH**; Arabic appears only as GPT-translated meanings. Verified: `example_language` is `en` for
   **100%** of rows; `is_generated_example` **True for all**; only ~729 idioms have any Arabic text.
   Sample Arabic gloss even contains a duplication artefact (`صحة أو صحة`). **Ingesting it unfiltered
   would poison the KB.** Its *field design* is however the closest existing template to our schema.
   | 习语本体是英语，阿语只是 GPT 翻译的释义；例句 100% 为英语、100% 为生成。**不可未过滤摄入**；
   但其字段设计可作 schema 模板参考。
2. **PARSEME `EGY` split = Ancient Egyptian hieroglyphs**, not Egyptian Arabic. Verified rows contain
   `Hiero=𓅃`, `LEMMA=ꜥḥꜥ`. | PARSEME 的 `EGY` 是**古埃及语象形文字**，不是埃及阿拉伯语。
   > **⚠️ CORRECTION (2026-08-22):** an earlier version of this doc concluded "there is no Arabic in any
   > PARSEME release." That is wrong. The *shared-task 2.0* release has no Arabic, but **PARSEME corpus
   > release 1.3 does contain `parseme-ar`** — 7,483 sentences / 311,743 tokens / **4,749 verbal MWEs**,
   > CC-BY-4.0, built on PADT (Hadj Mohamed et al., LREC 2022 `2022.lrec-1.196`). It has **spans, not
   > glosses**, so it is useless as a meaning KB — but it is a **gold set for measuring our matcher's
   > precision**. Key statistic: only **42.17% of Arabic VMWE occurrences are contiguous**; 17.3% have
   > gaps >3 tokens, making Arabic the second most discontinuous language in PARSEME after German.
   > | **更正**：此前"PARSEME 无任何阿语数据"的结论有误。共享任务 2.0 版确无阿语，但**语料 1.3 版包含
   > `parseme-ar`**（4,749 条动词性 MWE，CC-BY-4.0）。它只有跨度标注、无释义，不能当释义库，但**可用作
   > 评估我们匹配器精确率的黄金集**。关键数字：阿语 VMWE 仅 **42.17% 是连续的**。
3. **`ArSyra/*`** — card claims 23,254 records with an excellent schema (`msa_text`, `context`), but the
   dataset is **gated + commercially sold**; HF hosts only a 50-record preview that also returns
   "access restricted." | 声称 23,254 条且 schema 优秀，但**受限且商业售卖**，仅 50 条预览且不可访问。
4. **Qabas** (58,171 lemmas, the flagship Arabic lexicographic DB) — verbatim: *"multi-word lemmas are
   ignored at this phase."* **Zero idioms by design.** | 旗舰阿语词库，但**设计上排除多词词条，零习语**。
5. **Arabic WordNet 3.0/4.0** — the 2024 revision adds glosses + examples to all synsets but the paper
   never mentions idioms; its new "phrasets" are explicitly **free (compositional)** word combinations —
   the opposite of idioms. | 新增释义与例句，但"phrasets"明确是**自由（可组合）**词组，与习语相反。
6. **Irony/sarcasm corpora** (ArSarcasm 10,547 tweets, iSarcasmEval, IDAT) — tweet+label only, **no
   expression inventory, no meanings**. Rule out. | 仅推文+标签，无表达清单与释义，排除。
7. **`Tamazight-NLP/1002-Amazigh-Proverbs`** — 1,002 rows with the **richest schema found anywhere**
   (11 fields incl. Arabic translation *and* Arabic explanation *and* literal translation), but the
   proverbs are **Amazigh (Tifinagh), not Arabic**. Use as a **schema template** / North-African annex.
   | schema 最丰富，但谚语本体是**柏柏尔语**而非阿语；可作 schema 模板与北非附录。

---

## 5. Multilingual resources — Arabic coverage / 五、多语资源中的阿语覆盖（负面结论）

| Resource | Arabic? | Note |
|---|---|---|
| **IdiomKB** (AAAI 2024) | ❌ | en 3,990 / zh 8,643 / ja 270 only |
| **MABL** (ACL 2023) | ❌ | hi/id/jv/kn/su/sw/yo — confirmed absent |
| **MAPS** (NAACL 2024) | ❌ | en/de/ru/bn/zh/id |
| **BLEnD** | ✅ but ❌ idioms | Arabic via the **Algeria** subset; everyday cultural commonsense QA, no figurative content |
| **Global PIQA** | ✅ but ❌ idioms | 12 Arabic variants (`arb/arz/ary/arq/ars/acm/aeb/afb/apc×4`), 103 items each; physical commonsense. Useful as a **dialect-style reference** and to seed `entities` |
| **Tatoeba / OPUS** | ✅ sentences, ❌ idiom tags | tags are not carried into the released bitext |
| **AlphaMWE-Arabic** | ✅ 750 sentences | MSA + Egyptian + Tunisian vMWE **spans, no glosses** — detection only |
| **`islamlab/arabic-lexicons`** | ✅ **197,731 entries / 136 classical dictionaries** | incl. Taymūr's colloquial lexicon (13,298). Definitions as free article text. **CC BY-NC-SA** — check compatibility |
| **`mysamai/m3ajim`** | ✅ 40 dictionaries / 207,010 entries | no idiom dictionary among them; useful as a **literal-gloss lookup** for archaic words during enrichment. CC BY-NC-4.0 |

---

## 6. Recommended build plan / 六、建议的构建方案

**Stage 1 — assemble the seed KB today (~10.5–11K rows, all CC-BY-4.0 except where noted)**
/ **第一阶段：今天即可组装种子库（约 1.05–1.1 万条）**

| # | Source | n | Effort |
|---|---|---:|---|
| 1 | `tahaalselwii` (classical + colloquial, **drop `popular`**) | 7,730 | low |
| 2 | `UBC-NLP/Jawaher-benchmark` (`_fixed` + test) | 1,017 | very low — **check license** |
| 3 | Tunisian proverbs (drop MT columns) | 999 | low |
| 4 | Absher (dedupe 6× MCQ) | 561 | low |
| 5 | Kinayat (**gold template**) | 325 | very low |
| 6 | Hassaniya amthal (reshape from Alpaca) | 319 | very low |
| | **deduped subtotal** | **< 10K** | Kinayat / `tahaalselwii`-colloquial / Jawaher-EGY all derive from Taymūr — **expect real overlap** |

**Stage 2 — reach parity via the classical layer (+≈10K)** / **第二阶段：用古典层补到对标规模**
Parse `AuthenticIlm/Shamela4_Full_DB` → Maydani (4,965, near-mechanical) → al-Mustaqṣā (3,439) →
Jamhara (1,965). This is also the **only** source of per-row `source/derivation`.

**Stage 3 — build the three missing columns** / **第三阶段：补建三个缺失列**
- `examples` — LLM-generate, few-shot from Kinayat's 150 human-verified sentences; optionally mine
  attested usages by string-matching the idiom list against Arabic Tatoeba/OPUS/FineWeb-2 `arb_Arab`.
- `literal_meanings` — LLM-generate, grounded in `mysamai/m3ajim` / `islamlab/arabic-lexicons` for
  archaic vocabulary (many classical proverbs contain words no longer in use).
- `entities` — NER over the Arabic explanations (dense in names: `براقش`, `الزِّبْرِقَان بن بدر`).
- **Everything still missing after this is written `NAN`.** / **此后仍缺失的一律写 `NAN`。**

**Manual follow-ups (blocked from this environment) / 需人工跟进（本环境网络受限）**
1. **`github.com/UBC-NLP/jawaher`** — the other ~9,020 proverbs + the 2,205 literal translations.
   *Highest value action on this list.* | **收益最高的一步。**
2. **`pip install sinatools`** — Birzeit's SinaTools states it bundles *"a dictionary of [multi-word]
   expressions… with its glosses"* collected from 150 lexicons. If it ships as a data file, it is a
   ready-made Arabic MWE→gloss lexicon. **Highest-upside unverified lead.**
3. **Re-mine Hindawi's public-domain Taymūr volumes** (الأمثال العامية, الكنايات العامية) directly —
   both Kinayat (325) and `tahaalselwii`-colloquial (3,187) are *partial* extracts of full PD books.
4. `github.com/saleml/arabic-dialect-hub` — 552 phrases with genuine **literal English translations**
   (the scarcest field), MIT.

---

## 7. Morphological variation & the matching recipe / 七、形态变异与匹配方案

*Measured 2026-08-22 on **real data**: 60,000 FineWeb-2 `arb_Arab` documents (222.7M Arabic
chars) × the 7,730-entry human-authored inventory, with `camel-tools==1.6.0` read and executed
to verify every API claim. / 在真实数据上实测：FineWeb-2 阿语 6 万篇 × 7,730 条人工词条。*

### 7.1 The problem is real and severe / 问题真实且严重

**100% of dictionary entries are vocalized; only 1.50% of FineWeb-2 Arabic letters carry a
diacritic.** The two sides are in different orthographies.
| **词典 100% 带音符，而语料中仅 1.50% 的字母带音符**——两边根本不在同一套正字法上。

| Normalization level | docs matched / 60k | distinct idioms | vs baseline |
|---|---:|---:|---:|
| **L0 — raw citation form (the pipeline before this fix)** | **10** | 8 | 1× |
| L1 — dediacritize + tatweel + NFKC | 417 | 271 | 41.7× |
| L2 — + alef unification | 439 | 282 | 43.9× |
| **L3 — + ى→ي + ة→ه (what `normalize.py` implements)** | **477** | **308** | **47.7×** |

Dediacritization alone is ~97% of the win; the alef/maksura/teh-marbuta folds add **+14% overall
and +42% on the colloquial (Egyptian) list**, where ة↔ه alternation is pervasive.
| 去音符贡献约 97%；其余三项整体再加 14%，在埃及方言表上加 **42%**。

### 7.2 Two bugs this surfaced in our own code / 由此发现的两个自身缺陷

1. **🐛 Guillemets — was a total-recall killer.** **100% of the 3,187 Taymūr colloquial proverbs
   are wrapped in `«…»`**, which never appears around the proverb in running text. Verified:
   those entries had **0/2 recall** before the fix, **2/2** after.
   Fixed by `strip_quote_furniture()`. | **3,187 条方言谚语 100% 被 «» 包裹**，修复前**零召回**。
2. **🐛 Perso-Urdu letters leak into Arabic web text** — `ی` (U+06CC) in 307 docs, `ک` (U+06A9)
   in 154 docs of the 60k sample. Now folded to `ي`/`ك`.

### 7.3 Counterintuitive: do NOT add word boundaries / 反直觉：不要加词边界

Of 692 raw hits, **134 were "glued" on the left** (proclitic و/ف/ب/ال on the first word) and 10 on
the right. **Enforcing token boundaries would destroy 21.4% of all hits**, and every left-glued hit
inspected was a true positive. Aho-Corasick's substring semantics gives clitic tolerance at both
edges *for free* — only the idiom's **interior** is a problem.
| 强制词边界会**摧毁 21.4% 的命中**，且这些几乎全是真阳性。子串匹配天然容忍首尾词缀，**只有中间会出问题**。

### 7.4 Amthal are more frozen than kinayat / 谚语比习语更"冻结"

Fraction of occurrences 100% surface-identical to the citation form:

| Inventory | occurrences | fully identical |
|---|---:|---:|
| Classical *amthal* (Maydānī) | 499 | **73.9%** |
| Colloquial *amthal* (Taymūr) | 79 | **78.5%** |
| Egyptian *kināyāt* (Kinayat, in their own usage sentences) | 148 | **67.6%** |

External corroboration — **SAMER** (Al-Badrashiny et al. 2016) matched 4,000 Arabic verbal MWEs
against ATB+Gigaword: **20 distinct surface forms per MWE type on average**, only **15.7% of types
fully fixed**, and verbs fixed in only **17.7%** of types. The failure is concentrated in the
**pronoun suffix**, not the verb: `عينه فيه`→`عينها فيها`, `باله`→`بالك`, `إيده`→`إيدهم`.

**⇒ Split the inventory**: route frozen *amthal* through normalization only; route verbal *kināyāt*
through stem matching. Cheap heuristic: does the entry start with a perfect-tense verb, or contain a
3ms suffix `ـه`? | **应拆分词库**：冻结型走归一化，动词型走词干匹配。

### 7.5 Ranked recipe / 分级方案

| Rank | Action | Effort | Recall | FP risk | Status |
|---|---|---|---|---|---|
| 1 | **Normalize both sides before Aho-Corasick** | ~30 lines | **47.7×** | none | ✅ **implemented** (`normalize.py`) |
| 2 | **Keep substring semantics (no word boundaries)** | 0 | +21% of hits | negligible | ✅ implemented |
| 3 | **Strip guillemets/punctuation; drop patterns <10 chars** | 5 lines | prevents 0-recall on 3,187 entries | *negative* (cuts FPs) | ✅ implemented |
| 4 | Stem-token second pass, **min stem length 2** | ~150 lines | **+33% docs / +24% idioms** | ~5% | ⬜ not yet |
| 5 | Interior-clitic variant expansion | low | +~4% | low | ⬜ (skip if #4 done) |
| 6 | Gapped / bag-of-lemmas matching | high | +8% | **~50%** | ⛔ **do not do globally** |
| 7 | Full morphological lemmatization of the corpus | very high | unknown | — | ⛔ not worth it |

**On #6/#7:** the literature agrees. SAMER's best global setting is *max gap 2, no reordering*;
Hawwari 2012 says free gaps give "a large number of false positives"; Hadj Mohamed 2024's unbounded
bag-of-lemmas scores **P = 0.41**, recovered only by a trained literal/figurative BERT filter. Our
own measurement of gapped matching: **~50% precision**. MADAMIRA ≈10⁴ core-hours and Stanza ≈10⁶
GPU-hours for 100M docs; Farasa is research-license-only.
| 加间隔匹配实测精确率仅约 50%，文献结论一致；全量形态分析算力不可行。

**⚠️ Gotcha for #4:** set the light stemmer's **minimum stem length to 2, not 3**. A 3-char guard
refuses to strip `ها` from `فيها`, which alone caused most residual misses in the gold set.

### 7.6 Toolchain / 工具链

**Vendor CAMeL Tools' five util files (MIT), do not `pip install camel-tools`** — the package hard-
requires `torch>=2.0` + `transformers>=4.44` and downloads its catalogue from a blocked host, but
`utils/{normalize,dediac,charmap,charsets,stringutils}.py` import only `re`/`unicodedata`/`six`/`emoji`.
Verified semantics: `dediac_ar` does **not** remove tatweel (U+0640 is in `AR_LETTERS_CHARSET`) and
leaves Quranic marks — **our `normalize.py` removes both**. There is no combined "normalize
everything" function and no `lemmatize` in the package.

Measured throughput (single core): full normalize pipeline **15.7 MB/s**, Aho-Corasick over 7,592
patterns **40.4 MB/s**, end-to-end **11.3 MB/s ⇒ ~24.6 core-hours/TB**. Automaton build: 0.06 s.
**100M documents is a few hundred core-hours — not a constraint.** The bottleneck was
`normalize_unicode` NFKC at 4.4 MB/s, which `normalize.py` avoids by running NFKC only when a
presentation-form character is present and otherwise using a guarded `is_normalized("NFC")` check.

**Rejected:** `pyarabic` (GPL, and its `normalize_alef` maps ى→ا, silently destroying `على`/`إلى`),
`tashaphyne.get_root()` (returned `كوب` for `الكتاب`), Farasa (research-license-only), MADAMIRA/Stanza
(compute-prohibitive in the hot loop).

### 7.7 We would be the first / 我们会是第一个

**No prior work matches an Arabic idiom inventory against a web corpus at scale.** Jawaher and
Kinayat do no corpus matching; AIPSeLEX works on tweets with cosine+Levenshtein; Hawwari 2012
(250M tokens) and SAMER (~900M) are the largest published runs but use *clean newswire with gold
morphology* and never fed the output into pretraining. The corollary: **no published precision
baseline exists**, so budget for a literal/figurative disambiguation stage — lexicon projection
alone scores **R 0.79 / P 0.41** (Hadj Mohamed 2024).

---

## 8. Access limitations / 八、访问限制说明

Only **huggingface.co** and **arxiv.org** were reachable. **Hard-blocked:** github.com (web/API/raw),
ACL Anthology, HAL, Springer, MDPI, Zenodo, Kaggle, LDC/ELRA, kaikki.org, all Wikimedia hosts,
archive.org, shamela.ws, al-maktaba.org, hindawi.org, qdl.qa, `datasets-server.huggingface.co`.
Facts from blocked hosts are marked **[UNVERIFIED]** in-line and rest on search snippets only; no
figure or Arabic string below was invented. Two exact-size figures remain open: FineWeb-2 `arb_Arab`
row count and FinePDFs `arb_Arab` size (HF size API unreachable).
| 仅可访问 HF 与 arXiv；上述站点全部被封锁。来自被封锁站点的信息均标注 **[UNVERIFIED]**，仅基于搜索摘要，
未编造任何数字或阿语文本。

**Known discrepancies left unresolved / 未解决的矛盾:**
- `tahaalselwii` row total: 8,304 vs 8,305 across counts (arithmetic: 4,543+3,187+575 = **8,305**).
- Kinayat: paper text says "150 items", released CSV has **325** rows (150 with sentences).
- IdiomX: `dataset_statistics.json` says 174,956/12,823; the data card says 190K+/12K+ **and**
  124,411/8,806 elsewhere — internally inconsistent, another reason to distrust it.
- Absher: paper says "released upon acceptance", yet the HF repo is live and downloadable.
