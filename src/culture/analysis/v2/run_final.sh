#!/bin/bash
# Final pass: every LLM output is cached by now, so this only re-parses, embeds and aggregates.
set -x
source /data/group_data/r3lit_culture_pretrain/envs/bidir/bin/activate
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONUNBUFFERED=1
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
export PYTHONPATH=$REPO/src:$REPO/src/culture/analysis/v2
cd $REPO/src/culture/analysis/v2
python entity_typology_validate.py --only Qwen3.5-9B --skip_openrouter
python entity_typology_validate.py --only aya --skip_openrouter
python entity_divergence_multi.py
python entity_divergence_en_zh.py
python meaning_clusters.py --judges both --or_judge
