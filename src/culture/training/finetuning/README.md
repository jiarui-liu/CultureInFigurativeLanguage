# Cultural Instruction Tuning Dataset Generation

Generate instruction tuning data for cultural alignment of language models by integrating CultureBank contexts with idiom data and cross-lingual cultural analysis.

## Overview

The pipeline creates diverse QA pairs about **cultural behaviors, norms, and values**. Idioms and their figurative/literal meanings serve as a pool of cultural knowledge injected into the LLM prompt — the generated questions focus entirely on cultural aspects.

For each CultureBank context (e.g., "Americans tip servers in restaurants"):

1. **Draws N idioms** from a shuffled pool (each idiom used at most once — guaranteed unique)
2. **Provides all N idioms** with their figurative and literal meanings to the LLM as cultural knowledge
3. **LLM generates a QA pair** about cultural behavior/values/norms, enriched by that knowledge

English prompts for English idioms; Chinese prompts for Chinese idioms.

## Data Sources

| Source | Path | Size | Key Fields |
|--------|------|------|------------|
| CultureBank (Reddit) | `culture/data/CultureBank/culturebank_reddit.csv` | ~11K | cultural_group, context, topic, actor_behavior, eval_whole_desc |
| CultureBank (TikTok) | `culture/data/CultureBank/culturebank_tiktok.csv` | ~12K | Same as above |
| English Idioms | `culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl` | ~21K | idiom, entities, figurative_meanings, literal_meanings |
| Chinese Idioms | `culture/data/idioms/zh/idioms_merged_llm_formatted.jsonl` | ~28K | Same as above |
| Cultural Analysis | `culture/data/idioms/cross_lingual_analysis/cultural_analysis_clusters.json` | 516 clusters | english_unique_aspects, chinese_unique_aspects, cultural_explanation, summary |

## Answer Types

All types focus on cultural understanding:

| Type | Question Style | Answer Focus |
|------|---------------|--------------|
| `cultural_behavior_explanation` | "I noticed people do X, why?" | Cultural logic, social roots, norms driving the behavior |
| `cross_cultural_advice` | "How should I handle this cultural situation?" | Actionable guidance, common pitfalls, expectation management |
| `cultural_value_analysis` | "What deeper values explain this pattern?" | Core values, philosophical traditions, cross-cultural contrast |
| `cultural_norm_reasoning` | Scenario + "What is expected here?" | Unwritten rules, social consequences, appropriate responses |

## Usage

```bash
# English, 5000 samples via GPT-4o, 5 idioms per context
python generate_cultural_instruction_dataset.py \
    --language en \
    --num_samples 5000 \
    --model gpt-4o \
    --provider openai \
    --batch_size 20

# Chinese, 10000 samples
python generate_cultural_instruction_dataset.py \
    --language zh \
    --num_samples 10000 \
    --model gpt-4o \
    --provider openai

# Both languages, 3 idioms per context
python generate_cultural_instruction_dataset.py \
    --language both \
    --num_samples 10000 \
    --model gpt-4o \
    --provider openai \
    --candidates_per_context 3
```

## Idiom Selection Strategy

No keyword matching or heuristic scoring. Instead:

1. **Shuffle** all idioms into a random pool
2. **Draw** `candidates_per_context` idioms per CultureBank context (default: 5)
3. **Provide all** to the LLM with their figurative/literal meanings as cultural knowledge
4. **Consumed idioms are never reused** — each idiom appears at most once across the dataset

This guarantees maximum coverage of the idiom corpus. With 5 idioms per context, you can generate up to 4,255 EN examples or 5,633 ZH examples from the full idiom pool.

## Output Format

Each line in the output JSONL:

```json
{
    "instruction": "You are an expert in cross-cultural communication...",
    "input": "I recently moved to Japan for work and noticed that my colleagues always bow...",
    "output": "The bowing practice in Japanese professional settings reflects...",
    "metadata": {
        "idioms_provided": ["save face", "keep up appearances", ...],
        "language": "en",
        "answer_type": "cultural_behavior_explanation",
        "cultural_group": "Japanese",
        "topic": "Social Norms and Etiquette",
        "has_cultural_analysis": true,
        "source": "reddit"
    }
}
```

## Architecture

```
CultureBank (23K entries, shuffled)
        │
        ▼
  ┌─────────────────────────────┐
  │ Shuffled Idiom Pool         │   21K EN / 28K ZH idioms
  │ Draw N per context          │   Each used at most once
  └────────┬────────────────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │ Cultural Analysis Lookup    │   516 clusters, 2319 entities
  └────────┬────────────────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │ Answer Type Selection       │   Round-robin + randomness
  └────────┬────────────────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │ LLM Prompt with:           │
  │  - CultureBank context     │
  │  - N idioms + meanings     │   Cultural knowledge,
  │  - Cultural analysis       │   not idiom focus
  └────────┬────────────────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │ Batch Async LLM Generation  │
  │ + Validation                │
  └────────┬────────────────────┘
           │
           ▼
  JSONL output
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--language` | `zh` | `en`, `zh`, or `both` |
| `--num_samples` | `1000` | CultureBank contexts to sample per language |
| `--output_dir` | `culture/data/training/` | Output directory |
| `--output_file` | auto-generated | Override output path (single language only) |
| `--model` | `gpt-4o` | LLM model name |
| `--provider` | `openai` | LLM provider (`openai`, `togetherai`, `bedrock`, `huggingface`) |
| `--batch_size` | `20` | Concurrent LLM requests per batch |
| `--candidates_per_context` | `5` | Number of idioms (with meanings) provided per context |
| `--seed` | `42` | Random seed |

## Dependencies

- `pandas` — CultureBank CSV loading
- `tqdm` — Progress bars
- `culture.models.llm_utils.ChatModel` — LLM generation (required)
