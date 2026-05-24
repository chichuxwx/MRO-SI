#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# End-to-end MRO-SI pipeline:
#   1. Build masked-route data from OpenThoughts.
#   2. Run MRO-SI post-training and evaluation.

BASE_MODEL="${BASE_MODEL:-../models/Qwen3-1.7B}"
MASK_GENERATOR_MODEL="${MASK_GENERATOR_MODEL:-$BASE_MODEL}"
RAW_DATASET="${RAW_DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
RAW_DATASET_SPLIT="${RAW_DATASET_SPLIT:-train}"
MASKED_DATASET="${MASKED_DATASET:-data/openthoughts_math_30k_masked.jsonl}"
BUILD_MASKED_DATASET="${BUILD_MASKED_DATASET:-auto}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${NUM_PROCESSES:-}" ]]; then
    IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    NUM_PROCESSES="${#_visible_gpus[@]}"
fi

mkdir -p data logs

echo "================================================================================"
echo "MRO-SI full pipeline"
echo "raw dataset       : $RAW_DATASET"
echo "raw split         : $RAW_DATASET_SPLIT"
echo "masked dataset    : $MASKED_DATASET"
echo "base model        : $BASE_MODEL"
echo "mask generator    : $MASK_GENERATOR_MODEL"
echo "train/eval GPUs   : $CUDA_VISIBLE_DEVICES (processes=$NUM_PROCESSES)"
echo "================================================================================"

if [[ "$BUILD_MASKED_DATASET" == "true" || ( "$BUILD_MASKED_DATASET" == "auto" && ! -s "$MASKED_DATASET" ) ]]; then
    echo
    echo "================================================================================"
    echo "STEP 1/2: Build masked-route data"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python scripts/build_masked_derivations.py \
        --dataset_name_or_path "$RAW_DATASET" \
        --dataset_split "$RAW_DATASET_SPLIT" \
        --generation_backend vllm \
        --model_name_or_path "$MASK_GENERATOR_MODEL" \
        --output_path "$MASKED_DATASET" \
        --tensor_parallel_size "$NUM_PROCESSES" \
        --batch_size 16 \
        --max_new_tokens 1024 \
        --num_candidates 3 \
        --resume \
        2>&1 | tee "logs/full_pipeline_mask_build_$(date +%Y%m%d_%H%M%S).log"
else
    echo "STEP 1/2: Reuse masked-route data: $MASKED_DATASET"
fi

if [[ ! -s "$MASKED_DATASET" ]]; then
    echo "ERROR: masked dataset was not produced: $MASKED_DATASET" >&2
    exit 1
fi

echo
echo "================================================================================"
echo "STEP 2/2: Run MRO-SI post-training and evaluation"
echo "================================================================================"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NUM_PROCESSES="$NUM_PROCESSES" \
BASE_MODEL="$BASE_MODEL" \
MASKED_DATASET="$MASKED_DATASET" \
bash scripts/run_mrosi_train_eval.sh

echo
echo "Full pipeline complete."
echo "Masked dataset: $MASKED_DATASET"
echo "Checkpoints   : ${OUTPUT_ROOT:-outputs/mrosi}"
echo "Eval results  : ${EVAL_OUTPUT_ROOT:-eval_results}"
