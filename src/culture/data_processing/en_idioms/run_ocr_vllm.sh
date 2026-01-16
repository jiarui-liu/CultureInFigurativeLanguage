#!/usr/bin/env bash

export hf_model_path=$CULTURE_ROOT/culture/models/DotsOCR  # Path to your downloaded model weights, Please use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) for the model save path. This is a temporary workaround pending our integration with Transformers.
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
grep -q "^from DotsOCR import modeling_dots_ocr_vllm$" "$(which vllm)" || \
sed -i "/^from vllm\.entrypoints\.cli\.main import main$/a from DotsOCR import modeling_dots_ocr_vllm" "$(which vllm)"


# launch vllm server
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.95  --chat-template-content-format string --served-model-name model --trust-remote-code
# If you get a ModuleNotFoundError: No module named 'DotsOCR', please check the note above on the saved model directory name.

# vllm api demo
python3 $CULTURE_ROOT/src/dots.ocr/demo/demo_vllm.py --prompt_mode prompt_layout_all_en
