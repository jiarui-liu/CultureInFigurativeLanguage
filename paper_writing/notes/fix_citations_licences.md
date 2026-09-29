# Citation and licence fixes for the IdiomAtlas draft

Prepared 2026-09-27. No `.tex` file was edited. The BibTeX entries below have also been appended to
`new_refs_candidates.bib`, in section "(i) IdiomAtlas sources". None of the new keys duplicate a key in
`references.bib`, `custom.bib`, or `new_refs_candidates.bib`. To use an entry, move it into `custom.bib`,
because `main.tex` only loads `\bibliography{references,custom}`.

Not legal advice. The licence notes below record what each source states and what that implies for
redistribution. The Hindi source and the scraped Chinese sources should be checked by CMU's library or
legal office before anything is released.

---

## 1. Citation fixes

### 1(a) Chengyu cloze-test collection (03_data.tex, line 12)

Checked against the ACL Anthology page W18-0516. It is "Chengyu Cloze Test", by Jiang, Zhang, Huang and Ji,
published at BEA 2018 (the 13th Workshop on Innovative Use of NLP for Building Educational Applications),
pp. 154--158. Our local copy comes from `github.com/bazingagin/chengyu_data`, whose README links to this paper.

```bibtex
@inproceedings{jiang2018chengyu,
    title = "Chengyu Cloze Test",
    author = "Jiang, Zhiying  and Zhang, Boliang  and Huang, Lifu  and Ji, Heng",
    editor = "Tetreault, Joel  and Burstein, Jill  and Kochmar, Ekaterina  and Leacock, Claudia  and Yannakoudakis, Helen",
    booktitle = "Proceedings of the Thirteenth Workshop on Innovative Use of {NLP} for Building Educational Applications",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-0516/",
    doi = "10.18653/v1/W18-0516",
    pages = "154--158"
}
```
Replace the todo with `the Chengyu cloze-test collection \citep{jiang2018chengyu}`.

### 1(b) FuxiBench (03_data.tex, line 12)

FuxiBench is Fùxì (负屃), a 21-task benchmark on ancient Chinese understanding and generation, from
`github.com/cubenlp/FuxiBench`. Our pipeline reads only `test_data/idiom_exp.json`, the "Idiom Explanation" task
(1,236 items; input = idiom, output = explanation). The code that reads it is
`src/culture/data_processing/zh_idioms/reformat_idiom_from_sources.{py,sh}`.

The citation below is the one given in the repo README. I checked it against arXiv:2503.15837: v1 is dated
20 March 2025, with the comment "working in progress", and I found no published venue.
```bibtex
@misc{zhao2025fuxi,
    title = {F\`ux\`i: A Benchmark for Evaluating Language Models on Ancient {C}hinese Text Understanding and Generation},
    author = {Zhao, Shangqing and Zhou, Yuhao and Ren, Yupei and Chen, Zhe and Jia, Chenghao and Zhe, Fang and Long, Zhaogaung and Liu, Shu and Lan, Man},
    year = {2025}, eprint = {2503.15837}, archivePrefix = {arXiv}, primaryClass = {cs.CL},
    url = {https://arxiv.org/abs/2503.15837}
}
```
Replace the todo with `the idiom-explanation task of the F\`ux\`i benchmark \citep{zhao2025fuxi}`.
"Zhaogaung" is spelled that way on arXiv. Keep it.

Two warnings:
- `idiom_exp.json` is a **test split** of a published benchmark. Putting it in the KB, and therefore in the
  meaning tags, contaminates FuxiBench-IE for anyone who evaluates on it later. We do not evaluate on
  FuxiBench, so our results are unaffected, but the release notes should say this.
- Neither the paper nor the README says where the idiom explanations come from, and the repo has **no licence**
  (see §3).

### 1(c) Citation keys in the .tex files

I extracted every `\cite*{...}` key from `main.tex`, `latex/*.tex` and `latex/tables/*.tex`: 40 unique keys.
**All 40 are defined in `references.bib` or `custom.bib`, so no key is missing.** No key is defined in both files.

Sources that 03_data.tex names but does not cite (optional entries, all in `new_refs_candidates.bib`):

| Mention in 03_data.tex | Suggested key | Status |
|---|---|---|
| `englishidioms` dictionary | `zaghloul2023englishidioms` (package) and/or `spears2005mcgraw` (underlying book, copyright 2005 McGraw-Hill) | verified |
| Wiktionary *English idioms* category | `wiktionary_en_idioms` (or a footnote URL) | crawled Dec 2025, per file timestamps |
| open Xinhua dictionary | `pwxcoo2018xinhua` | GitHub repo, commit fe6d6c2 |
| Hindi print dictionary | `tiwari_lokokti` | **year not verified, see §2.3** |
| Absher (one of the "three benchmarks with explanations") | `almonef2026absher` | verified via Crossref |
| Tunisian proverbs | `abderrahim2025tunisian` | from dataset card (HF DOI) |
| Kinayat | `kinayat2025` | already in custom.bib, currently cited only in 05_exp |

Neither al-Maydānī's *Majmaʿ al-Amthāl* nor Taymūr's *al-Amthāl al-ʿĀmmiyya* has a bib entry. They are
reached through the HF dataset `tahaalselwii/arabic-proverbs-collection`, which has no citation. A footnote URL
is probably enough.

### 1(d) Qwen3.5 (05_exp.tex, line 8)

The Qwen3.5-9B-Base model card (checked today) recommends
`@misc{qwen3.5, title={{Qwen3.5}: Towards Native Multimodal Agents}, author={{Qwen Team}}, month={February}, year={2026}, url={https://qwen.ai/blog?id=qwen3.5}}`.
This matches `qwen2026qwen35` in `new_refs_candidates.bib`. Move that entry to `custom.bib` and write:

```latex
We continue pretraining Qwen3.5-9B-Base \citep{qwen2026qwen35,yang2025qwen3} ...
```
Then delete the `\todo`. The model is Apache-2.0, so releasing the continued-pretraining checkpoints raises no
licence problem on the model side.

---

## 2. Provenance notes

### 2.1 English
Merged by `src/culture/data_processing/en_idioms/merge_en_idioms.sh` from three inputs:
IdiomKB `reformatted_en_idiom_meaning.jsonl`, Wiktionary `english_idioms_reformatted.jsonl`, and englishidioms
`phrases_reformatted.jsonl`. The file `culture/data/en_idiom_oxford_dictionary.pdf`, and its OCR, is **not** an
input to the merge.
- englishidioms (`github.com/zaghloul404/englishidioms`): the code is MIT. Its README says it parses "all
  dictionary entries in the *McGraw-Hill Dictionary of American Idioms and Phrasal Verbs*" (22,209 entries), and
  adds: "copyrighted material ... not endorsed or authorized by The McGraw-Hill Companies ... permitted to use
  this package for personal, non-commercial purposes only. Any commercial application or distribution of this
  package's output may require the prior written consent of the publisher." We take the idiom, the definitions
  and the examples, and several definitions carry literal/figurative labels.
- Wiktionary: crawled through the MediaWiki API from `Category:English_idioms`. Wiktionary's copyright page
  says entry texts are "dual-licensed ... under both the Creative Commons Attribution-ShareAlike 4.0
  International License and the GNU Free Documentation License".
- IdiomKB (`github.com/lishuang-w/IdiomKB`): the repo has **no licence file**. Its README says: "The data in this
  repository was generated by OpenAI's gpt-3.5-turbo-0613 model". So the IdiomKB meanings are **LLM-generated**.
  That conflicts with the rule stated for Arabic ("Collections generated by language models are excluded"),
  and 03_data.tex should either acknowledge it or drop IdiomKB for consistency.

### 2.2 Chinese
- chinese-xinhua (`github.com/pwxcoo/chinese-xinhua`): the repo is MIT. Its README "Copyright" section says the
  data was "collected from the web ... scraped from various websites", with "no commercial purpose; will delete
  on infringement". The MIT grant therefore comes from someone who does not claim to hold rights in the
  underlying definitions. We use `word`, `explanation`, `derivation` and `example`.
- Chengyu Cloze Test (`github.com/bazingagin/chengyu_data`): **no licence**. The paper (§4.1) says the sentences
  were crawled from zaojv.com and the definitions from cy.5156edu.com, and promises release "for research
  purposes". We use both the definitions and the example sentences.
- FuxiBench: **no licence** and no stated source. See 1(b).
- IdiomKB zh: same as the English portion (no licence, GPT-3.5-generated).

### 2.3 Hindi — the book is almost certainly still in copyright
We use the digitized copy at archive.org/details/in.ernet.dli.2015.464150 (Digital Library of India item
2015.464150). Local copies are `culture/data/hi_idioms/raw/proverbs_djvu.txt` and `page_numbers.json`, and the
identifier appears in the JSON. The title page and archive metadata show:
- Title: *Vṛhat/Bṛhat Hindī Lokokti Koś* (A Comprehensive Dictionary of Hindi Proverbs).
- Editor: Dr. Bholanath Tiwari (भोलानाथ तिवारी). Co-editors: Nur Nabi Abbasi and Dr. Kiran Bala.
- Publisher: Shabdakar, 2203 Gali Dakautan, Turkman Gate, Delhi. 1,127 scanned pages.
- **Publication year: not verified.** No year survives in the OCR, and none of the catalogues I could reach
  (NITI Aayog, Lucknow Digital Library, Open Library, Google Books) gives one. A second DLI scan
  (in.ernet.dli.2015.444182) is dated 1960, but that cannot be right: the preface ("दो शब्द") says the collection
  was edited until July 1978, damaged in the August 1978 Delhi flood, recopied over about 1.5 years, retyped,
  and then revised by Abbasi. The first edition is therefore **after 1978**, probably in the 1980s.
- Tiwari lived from 4 Nov 1923 to 25 Oct 1989 (Hindi Wikipedia). India's term for literary works is life + 60
  years, so the book stays protected until at least 1 Jan 2050, and later if a co-editor outlived him.
- The DLI metadata says `dc.rights: In Public Domain`. That label is **not reliable**: DLI applied it in bulk,
  including to many in-copyright books. It does not license redistribution.
- Individual folk proverbs are traditional sayings with no single author. Tiwari's selection, his Hindi
  explanations, the cross-references and the comparative material are his copyrighted work. Our Hindi KB
  (16.6K proverbs) comes entirely from this book, via OCR plus LLM repair.

### 2.4 Arabic
Sources are listed in `SOURCES` in `src/culture/data_processing/ar_idioms/build_ar_idioms.py`. Licences are the
`license` field returned by `https://huggingface.co/api/datasets/<repo>` today.

| # | Name in code | HF repo (file) | Upstream content | HF licence |
|---|---|---|---|---|
| 1 | tahaalselwii_classical | `tahaalselwii/arabic-proverbs-collection` (`classical_arabic_proverbs.csv`) | al-Maydānī, *Majmaʿ al-Amthāl* (d. 1124). The text carries Shamela page markers (⦗٢٣٨⦘), so it was digitized from a modern printed edition | CC BY 4.0 |
| 2 | tahaalselwii_colloquial | same repo (`colloquial_arabic_proverbs.csv`) | Aḥmad Taymūr Pasha, *al-Amthāl al-ʿĀmmiyya*, published 1949 (the author died in 1930) | CC BY 4.0 |
| 3 | jawaher | `UBC-NLP/Jawaher-benchmark` (`Jawaher_train_fixed.jsonl`, `Jawaher_test.jsonl`) | NAACL 2025 benchmark; 1,017 of 10,037 proverbs public | **none declared** (empty card, no GitHub repo found) |
| 4 | kinayat | `menaattia/Kinayat` | from Taymūr, *al-Kināyāt al-ʿĀmmiyya* (1949); usage sentences GPT-4.1-generated and then human-reviewed | CC BY 4.0 |
| 5 | tunisian | `HabibaAbderrahim/Tunisian-Proverbs-with-Image-Associations-...` | Tunisian proverbs with Arabic explanations; MT columns dropped | CC BY 4.0 |
| 6 | absher | `Renad10/Absher-Benchmark` (Proverbs/Phrases `*_Meaning_questions.csv`) | scraped from the Moajam crowd dictionary | CC BY 4.0 |
| 7 | hassaniya | `ahmed02mk/amthal-hassaniya` | from the book *Mawsūʿat al-Amthāl al-Ḥassāniyya* by Bakkār wuld Aḥmadū (the card's "المصدر") | CC BY 4.0 |

The code excludes the following and never fetches them: `tahaalselwii/...popular_arabic_proverbs.csv`
(AI-generated), `aymansharara/IdiomX`, `kenantang/IdiomTranslate30`, and `ArSyra/*`.

The todo's "two Arabic critical editions" are presumably #1 and #2. A CC BY 4.0 tag from the uploader does not
clear rights in the edition behind the digitization. For al-Maydānī, the medieval text is public domain, but a
20th-century editor's vocalization, apparatus and pagination may be protected; our internal note
`docs/literature_reviews/arabic_idiom_resources.md` identifies the Shamela text as Muḥyī al-Dīn ʿAbd al-Ḥamīd's
edition, which I have not verified. For Taymūr, the 1949 book is probably public domain in Egypt, where the term
is life + 50 years, but whoever digitized it (and from which reprint) is unknown. #7 is a modern
Mauritanian book, so it is unclear whether the uploader could license it under CC BY.

### 2.5 Web corpora
| Corpus | HF repo checked | Licence |
|---|---|---|
| mC4 | `allenai/c4` (the code loads `allenai/c4`/`allenai/mc4`) | ODC-BY; Common Crawl terms of use also apply |
| FineWeb-2 | `HuggingFaceFW/fineweb-2` | ODC-BY; Common Crawl terms of use also apply |
| Fineweb-Edu-Chinese-V2.1 | `opencsg/Fineweb-Edu-Chinese-V2.1` | Apache-2.0 tag, but the card also requires "adherence to the OpenCSG Community License"; commercial use needs email permission |
| IndicCorp v2 | `ai4bharat/IndicCorpV2` | no licence field on the HF card; the card text and the IndicBERT README say datasets are released under CC0 |

### 2.6 Evaluation benchmarks (only the easy checks; the rest are unchecked)
| Benchmark | Source checked | Licence |
|---|---|---|
| ArabCulture | HF `MBZUAI/ArabCulture` | CC BY-NC-SA 4.0 |
| ArabicMMLU | HF `MBZUAI/ArabicMMLU` | CC BY-NC 4.0 |
| MILU | HF `ai4bharat/MILU` (gated) | CC BY 4.0 |
| Global-PIQA | HF `mrlbenchmarks/global-piqa-{parallel,nonparallel}` | CC BY-SA 4.0 |
| ChID | GitHub `chujiezheng/ChID-Dataset` | Apache-2.0 |
| CMMLU | GitHub `haonan-li/CMMLU` README | CC BY-NC-SA 4.0 |
| Kinayat | HF `menaattia/Kinayat` | CC BY 4.0 |
| Chengyu-Bench, CCPM, MABL, ArabicCulturalQA, Alyah, DziriEval | — | **not checked** |

We evaluate only, so none of these licences restricts the paper. **AR-Figurative**, the 314-item set we build
from Alyah and DziriEval, can be released only if the Alyah and DziriEval licences allow it. Check them before
promising a release.

---

## 3. Licence table (IdiomAtlas sources)

"Release" means redistributing KB entries (idiom string, meanings, entities, examples).

| Source | What we use | Licence (as stated) | Redistribution implication |
|---|---|---|---|
| **en** englishidioms / McGraw-Hill dictionary | idioms, definitions (some literal/figurative labelled), examples | package code MIT; **content © 2005 McGraw-Hill**; README: personal, non-commercial use only | **Exclude** definitions and examples from the release. Release a rebuild script that reads the user's own install of the package. Releasing even the headword list is doubtful because it reproduces the dictionary's selection; leave it out to be safe. |
| **en** Wiktionary *English idioms* | headwords, glosses, examples | CC BY-SA 4.0 + GFDL | **Can release** with attribution. Derived entries must also be **CC BY-SA 4.0** (ShareAlike), which forces the English release, or at least this subset, to be CC BY-SA. |
| **en/zh** IdiomKB | idioms, English/Chinese meanings | **no licence** (all rights reserved by default); meanings generated by gpt-3.5-turbo-0613 | **Do not redistribute**; point to the repo and provide a rebuild script. The KB also becomes partly LLM-generated, so disclose this. |
| **zh** chinese-xinhua | idioms, explanation, derivation, examples | MIT on the repo, but the data was scraped from websites and the maintainer disclaims rights | **Ambiguous.** MIT allows release with the notice included, but the upstream rights are unclear. Moderate risk: either release with the MIT notice plus a provenance caveat, or release idiom and entity only. |
| **zh** Chengyu Cloze Test | definitions and cloze sentences | **no licence**; crawled from zaojv.com and cy.5156edu.com; "for research purposes" | **Do not redistribute** definitions or sentences; rebuild script only. |
| **zh** FuxiBench Idiom Explanation | 1,236 idiom explanations (test split) | **no licence**; source of explanations not stated | **Do not redistribute**; rebuild script. Warn about test-set contamination. |
| **hi** *Bṛhat Hindī Lokokti Koś* (Tiwari, Abbasi, Bala; Shabdakar, Delhi; year unverified, after 1978) | proverbs, meanings (OCR + LLM repair) | **In copyright** (Tiwari d. 1989, India life+60, so at least until 2050); DLI "In Public Domain" label unreliable | **Do not release meanings or the OCR text.** At most release proverb strings (traditional sayings) plus our own entity annotations, and even that reproduces the book's selection, so clear it with legal/the library first. Otherwise release the OCR-repair pipeline and the archive.org identifier. Asking the publisher (Shabdakar) for permission is the only clean way to release the full Hindi KB. |
| **ar** tahaalselwii classical (al-Maydānī) | proverbs and explanations | CC BY 4.0 (HF); medieval text PD, digitized from a modern edition | **Can release** under CC BY 4.0 with attribution. Caveat: rights in the editor's apparatus are not cleared by the uploader's tag. Low-moderate risk. |
| **ar** tahaalselwii colloquial (Taymūr 1949) | proverbs and explanations | CC BY 4.0 (HF); author d. 1930 | **Can release** under CC BY 4.0. The underlying work is probably PD in Egypt. Low risk. |
| **ar** Jawaher | 1,017 proverbs, Arabic explanations | **no licence declared** | **Exclude** from the release (or pointer + script) until the authors grant a licence. Easy fix: email them. |
| **ar** Kinayat | idioms and meanings (+ reviewed LLM example sentences) | CC BY 4.0 | **Can release** with attribution; mark the examples as LLM-generated and human-reviewed. |
| **ar** Tunisian proverbs | proverbs and Arabic explanations | CC BY 4.0 | **Can release** with attribution. |
| **ar** Absher | proverbs/phrases and meanings | CC BY 4.0; scraped from the Moajam crowd dictionary | **Can release** with attribution; the Moajam terms are unknown. |
| **ar** amthal-hassaniya | proverbs and meanings | CC BY 4.0 (HF), taken from a modern book | **Can release** as licensed, with a caveat that the uploader's rights are unclear. |
| Our additions (entity lists, literal readings generated for Arabic, register/variety labels, cross-lingual analyses) | — | ours | Release under CC BY 4.0 (or CC BY-SA where attached to Wiktionary content). Keep generated fields marked as generated. |
| Web corpora (mC4, FineWeb-2, Fineweb-Edu-Zh, IndicCorp v2) | idiom-matched documents + meaning tags | ODC-BY / ODC-BY / Apache-2.0 + OpenCSG Community License / CC0 | Release **document IDs, idiom offsets and filtering scripts**, not re-hosted text, or re-host under the upstream terms. Meaning tags inherit the restrictions of the KB entries they quote. |

Overall: the entries that can be released as-is are Wiktionary (English) and the six CC BY Arabic sources
(about 10K entries). The Hindi KB, the McGraw-Hill-derived English entries, IdiomKB, Chengyu Cloze, FuxiBench and
Jawaher need either rebuild scripts or permission. Xinhua is a judgement call.

---

## 4. Proposed LaTeX (appendix "Licences and release", plus one sentence in §3)

**Replacement for the licence `\todo` in 03_data.tex:**
```latex
Appendix~\ref{app:licences} lists the licence of every source and what we release.
```

**Appendix text:**
```latex
\section{Licences and Release}
\label{app:licences}

\dataname{} is built from sources whose licences differ, and several of them do not permit redistribution.
Table~\ref{tab:licences} lists each source, the licence it states, and what we release.

\paragraph{Openly licensed sources.}
The Wiktionary entries are available under CC BY-SA 4.0 and the GFDL; entries derived from them are released
under CC BY-SA 4.0. Six of the seven Arabic collections are distributed on Hugging Face under CC BY 4.0
(al-Mayd\={a}n\={\i}'s \textit{Majma\textsuperscript{c} al-Amth\={a}l} and Taym\={u}r's colloquial proverbs, Kinayat,
the Tunisian and Hassaniya proverb sets, and Absher \citep{kinayat2025,abderrahim2025tunisian,almonef2026absher}).
We release these entries with attribution. The classical and Hassaniya collections are digitizations of printed
editions, and the uploaders' licences may not cover rights held by the editors of those editions.
The open Xinhua data \citep{pwxcoo2018xinhua} is MIT-licensed, but its maintainer states that it was collected
from websites; we release its entries with the MIT notice and this caveat.

\paragraph{Sources we do not redistribute.}
The English dictionary distributed with the \texttt{englishidioms} package is derived from a copyrighted
dictionary \citep{spears2005mcgraw}, and the package restricts use of its content to personal, non-commercial
purposes. IdiomKB \citep{li2024idiomkb}, the Chengyu cloze-test data \citep{jiang2018chengyu}, the idiom-explanation
task of F\`ux\`i \citep{zhao2025fuxi}, and the public portion of Jawaher \citep{magdy2025jawaher} declare no licence.
IdiomKB's meanings were generated by \texttt{gpt-3.5-turbo}, and F\`ux\`i's idiom explanations are a benchmark test
split. The Hindi dictionary \citep{tiwari_lokokti} is in copyright, even though the digitized copy we used is
labelled public domain by the Digital Library of India. For all of these sources we release only the scripts that
rebuild the corresponding entries from a copy the user obtains, together with source identifiers. For Hindi this
means the OCR-repair pipeline and the archive identifier of the scan, but none of the dictionary's text.

\paragraph{Corpora and models.}
The web corpora are available under ODC-BY (mC4, FineWeb-2), CC0 (IndicCorp v2), and Apache-2.0 with the OpenCSG
Community License (Fineweb-Edu-Chinese). Instead of re-hosting documents, we release document identifiers, idiom
match offsets, and the filtering code. Meaning tags inherit the terms of the entries they quote. The continued-pretraining
checkpoints are derived from Qwen3.5-9B-Base (Apache-2.0). We use all evaluation benchmarks under their
own licences, for evaluation only.
```
(`tab:licences` would be a condensed version of the table in §3, with columns Source / Licence / Released.)

**Things to settle before this paragraph is true:**
1. Fill in the Hindi publication year from a library catalogue (WorldCat or the National Library of India).
2. Decide on Xinhua: release with a caveat, or idiom and entity only.
3. Email the Jawaher authors about a licence. If they grant one, move Jawaher into the open group.
4. Check the Alyah and DziriEval licences before promising to release AR-Figurative.
5. Rewrite the IdiomKB sentence in 03_data.tex, or drop IdiomKB, because it contradicts "Collections generated by
   language models are excluded" (currently stated for Arabic only).

---

## 5. Side observation (not a licence issue; please check)

In `culture/data/idioms/zh/idioms_merged_llm_formatted.jsonl`, 22,945 of 31,155 entries have a non-empty
`literal_meanings`, and **14,516** of those contain a book-title quotation (《…》). Examples: 一丁不识 →
"《旧唐书·张弘靖传》…" and 一不做，二不休 → "唐·赵元一《奉天录》…". These look like the Xinhua `derivation` field
(the source text where the idiom first appears), not a gloss of the literal scene. If the paper's number for zh
(71.5% with a literal meaning) is computed from this field, it is probably inflated, and the claim that "Chinese
dictionaries usually gloss the literal scene" needs rechecking. I checked only the file on disk. The paper may
use a later, repaired version.

---

## Sources checked
- ACL Anthology W18-0516: https://aclanthology.org/W18-0516/
- arXiv 2503.15837 (Fùxì): https://arxiv.org/abs/2503.15837; FuxiBench README (local clone of github.com/cubenlp/FuxiBench, commit 41069dc)
- GitHub API licence fields: pwxcoo/chinese-xinhua (MIT), zaghloul404/englishidioms (MIT), cubenlp/FuxiBench (none), bazingagin/chengyu_data (none), lishuang-w/IdiomKB (none)
- Wiktionary copyrights: https://en.wiktionary.org/wiki/Wiktionary:Copyrights
- HF dataset API (`/api/datasets/<repo>`) for all Arabic sources, the corpora and the benchmarks listed above; HF model API for Qwen/Qwen3.5-9B-Base
- archive.org metadata: in.ernet.dli.2015.464150 and in.ernet.dli.2015.444182
- Bholanath Tiwari dates: https://hi.wikipedia.org/wiki/भोलानाथ_तिवारी
- Crossref: 10.1016/j.aej.2025.12.066 (Absher)
- McGraw-Hill dictionary: Open Library record (Spears) and publisher copyright notice 2005
