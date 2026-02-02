#!/bin/bash

# Cross-lingual entity analysis for idioms
# This script analyzes how the same entities convey different meanings
# across English and Chinese idioms.

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

EN_IDIOMS="$PROJECT_ROOT/culture/data/idioms/en/idioms_merged_llm_formatted.jsonl"
ZH_IDIOMS="$PROJECT_ROOT/culture/data/idioms/zh/idioms_merged_llm_formatted.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/culture/data/idioms/cross_lingual_analysis"

# Default parameters
TOP_K=200
MODEL="gpt-5.2-chat-latest"
PROVIDER="openai"
MAX_IDIOMS=20
BATCH_SIZE=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --max_idioms)
            MAX_IDIOMS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Cross-lingual Entity Analysis for Idioms"
echo "============================================"
echo "English idioms: $EN_IDIOMS"
echo "Chinese idioms: $ZH_IDIOMS"
echo "Output directory: $OUTPUT_DIR"
echo "Top-K entities: $TOP_K"
echo "Model: $MODEL"
echo "Provider: $PROVIDER"
echo "Max idioms per entity: $MAX_IDIOMS"
echo "Batch size: $BATCH_SIZE"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the analysis
python3 "$SCRIPT_DIR/cross_lingual_entity_analysis.py" \
    --en_idioms "$EN_IDIOMS" \
    --zh_idioms "$ZH_IDIOMS" \
    --output_dir "$OUTPUT_DIR" \
    --top_k "$TOP_K" \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --max_idioms "$MAX_IDIOMS" \
    --batch_size "$BATCH_SIZE"

echo "============================================"
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
