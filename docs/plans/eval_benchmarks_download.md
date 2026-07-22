# Downloading the Hindi evaluation benchmarks

Exact download instructions for the four Hindi language/culture/idiom benchmarks
used by `src/culture/evaluation/`. Each section gives the verified command, the
expected schema, and where the eval loader expects the file.

Legend: **[VERIFIED]** confirmed against the live source; **[INSPECT]** confirm
by looking at the file/features after download (gated or column-order caveat).

Default local layout (matches `run_eval.sh`, `DATA_DIR=data/eval/hi`):

```
data/eval/hi/
├── mabl_hi.csv          # MABL Hindi
├── global_piqa_hi.tsv   # Global PIQA Hindi (Devanagari)
└── idiomce_hi.jsonl     # IdiomCE (you assemble this — see §4)
# MILU is pulled from HuggingFace at run time (gated), no local file needed.
```

Prereqs:
```bash
pip install "huggingface_hub[cli]" datasets
huggingface-cli login          # needed for MILU (gated)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # run_eval reads HF_TOKEN
```

---

## 1. MABL (figurative-meaning inference) — `mabl`

**Source:** `github.com/simran-khanuja/Multilingual-Fig-QA` (Kabra et al., ACL
Findings 2023). **No HuggingFace mirror of the Hindi split** — the `SEACrowd/mabl`
HF dataset contains only Indonesian/Javanese/Sundanese.

**Hindi file:** `langdata/hi.csv` (1,000 items, test-only). **[VERIFIED]** path
via the SEACrowd loader (`_URL = ".../main/langdata/" + "{iso}.csv"`, Hindi ISO = `hi`).

```bash
# just the Hindi file:
curl -L -o data/eval/hi/mabl_hi.csv \
  https://raw.githubusercontent.com/simran-khanuja/Multilingual-Fig-QA/main/langdata/hi.csv
# or clone the repo: git clone https://github.com/simran-khanuja/Multilingual-Fig-QA.git
#   -> Multilingual-Fig-QA/langdata/hi.csv
```

**Schema:** logical fields `startphrase`, `ending1`, `ending2`, `labels` (0/1 =
index of the correct ending).

> **[INSPECT] Column order varies by language** in this repo (e.g. Indonesian is
> `ending1,ending2,labels,startphrase`). Check the header of `hi.csv` before use:
> ```bash
> head -1 data/eval/hi/mabl_hi.csv
> ```
> The loader (`tasks.load_mabl`) reads by **column name** (with aliases), so any
> order works as long as the header names match; if `hi.csv` has no header or odd
> names, rename columns to `startphrase,ending1,ending2,labels`.

---

## 2. MILU (cultural-knowledge exam QA) — `milu`

**Source:** `ai4bharat/MILU` on HuggingFace. **Gated** (accept terms on the page +
HF token). License CC BY 4.0. **[VERIFIED]** from the dataset card.

- **Config for Hindi:** `"Hindi"` (full English name, **not** `"hi"`), passed as `data_dir`.
- **Splits:** `test` (evaluate) and `validation` (few-shot pool). No train split.

The eval loader pulls it automatically at run time (no local file needed) — just
authenticate. To pre-download / inspect:

```python
from datasets import load_dataset
test = load_dataset("ai4bharat/MILU", data_dir="Hindi", split="test", token=True)
val  = load_dataset("ai4bharat/MILU", data_dir="Hindi", split="validation", token=True)
print(test.features)          # <-- confirm field names (see INSPECT below)
```
```bash
huggingface-cli download ai4bharat/MILU --repo-type dataset --include "Hindi/*"
```

> **[INSPECT] Exact field names are behind the gate** — the card shows no example
> row and the schema endpoint requires auth. After loading, run `test.features`
> and confirm they match what `tasks.load_milu` expects: a question field
> (`question`), options (a list under `options`/`choices`, or `option1..option4`),
> and an answer (`answer`/`target`/`label` as letter, index, or option text). The
> loader is defensive across these, but verify once and adjust the template/aliases
> in `tasks.py` if MILU uses different names.

To use MILU's validation split for leak-free few-shot, dump it to a local file and
pass `--milu_fewshot_path`:
```python
val.to_json("data/eval/hi/milu_hi_val.jsonl", force_ascii=False, lines=True)
```

---

## 3. Global PIQA (cultural physical commonsense) — `global_piqa`

**Source:** `mrlbenchmarks/global-piqa-nonparallel` on HuggingFace. Openly
downloadable (CC BY-SA 4.0). **[VERIFIED]** — headers + rows read directly.

- **Hindi Devanagari subset:** config `hin_deva`, file `data/nonparallel_hin_deva.tsv`
  (100 rows). (`hin_latn` = romanized.)
- **Columns:** `prompt`, `solution0`, `solution1`, `label` (0/1), plus
  `language, eng_translated0/1, approx_cultural_score, llm_used, example_id, supplement`.

```python
from datasets import load_dataset
ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", "hin_deva")["test"]
```
```bash
# as a local TSV for the loader:
huggingface-cli download mrlbenchmarks/global-piqa-nonparallel \
  --repo-type dataset --include "data/nonparallel_hin_deva.tsv" \
  --local-dir data/eval/hi/_gpiqa
cp data/eval/hi/_gpiqa/data/nonparallel_hin_deva.tsv data/eval/hi/global_piqa_hi.tsv
# raw URL: https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/resolve/main/data/nonparallel_hin_deva.tsv
```

> **License note:** the card restricts use to **LLM evaluation only — no training**.
> Fine for this eval; do not fold it into the CPT corpus.
>
> A 4-option **parallel** split also exists (`mrlbenchmarks/global-piqa-parallel`,
> `data/parallel_hin_deva.tsv`) with columns `solution0..solution3`. The loader
> handles 2 options; if you switch to the parallel split, it reads `solution0/1`
> only — use the non-parallel file above unless you extend the loader.

---

## 4. IdiomCE (En→Hi idiomatic translation) — `idiomce`

**No public data or code release. [VERIFIED]** (arXiv 2505.21937 has no repo link;
nothing on GitHub/HF). You must **assemble the eval JSONL yourself** from source
idiom resources. Target format (one line per item):

```json
{"source": "Don't count your chickens before they hatch.", "idiom_en": "count your chickens before they hatch", "reference": "<optional Hindi>"}
```
Only `source` is required; `reference`/`idiom_en` make the OpenAI judge
reference-guided (see the eval README).

**Where to get English idiom sentences (+ Hindi references):**

| Resource | Status | Get it |
|---|---|---|
| **Samanantar** (Indic parallel corpus) | **[VERIFIED] public** | `load_dataset("ai4bharat/samanantar", "hi")` — filter en side for idioms |
| **MAGPIE** (English potentially-idiomatic, 56K) | public | `github.com/hslh/magpie-corpus` (via ACL `2020.lrec-1.35`); 400 used by IdiomCE |
| Agrawal et al. 2018 (LREC `L18-1048`) en↔7 Indian-lang idioms | **[INSPECT]** no confirmed download | read PDF `aclanthology.org/L18-1048.pdf` for availability, or contact IIIT-H LTRC |
| Thakre et al. 2018 (Hi→En idiomatic sentences) | **[INSPECT]** no public dataset found | small journal paper; no repo located |

Practical path: pull idiom-bearing English sentences from **MAGPIE** (or your own
English idiom KB under `culture/data/idioms/en/`), optionally attach Hindi
references from **Samanantar**, and write them to `data/eval/hi/idiomce_hi.jsonl`.

> **Do not use** `github.com/amazon-science/idiom-mt` — it is a German–English
> idiom set (Fadaee et al. 2018), unrelated to IdiomCE.

---

## Environment caveats (for reproducing this doc's verification)

Verified against reachable sources (HuggingFace card/files, arXiv, the SEACrowd
loader script). GitHub, `raw.githubusercontent`, and ACL Anthology were network-
blocked at verification time, so MABL's exact column order, MILU's gated field
names, and the Agrawal/Thakre download links are marked **[INSPECT]** rather than
asserted — confirm those after you can reach the sources / accept the gate.
