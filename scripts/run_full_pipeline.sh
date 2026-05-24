#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# End-to-end MRO-SI pipeline:
#   1. Build masked-derivation training data.
#   2. Run MRO-SI post-training.
#   3. Evaluate saved checkpoints.
#
# The raw training file must contain at least problem and solution columns.

BASE_MODEL="${BASE_MODEL:-../models/Qwen3-1.7B}"
MASK_GENERATOR_MODEL="${MASK_GENERATOR_MODEL:-$BASE_MODEL}"
RAW_DATASET="${RAW_DATASET:-data/train.jsonl}"
MASKED_DATASET="${MASKED_DATASET:-data/train_masked.jsonl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${NUM_PROCESSES:-}" ]]; then
    IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    NUM_PROCESSES="${#_visible_gpus[@]}"
fi

BUILD_MASKED_DATASET="${BUILD_MASKED_DATASET:-auto}"
PREPARE_RAW_DATASET="${PREPARE_RAW_DATASET:-false}"
MASK_GENERATION_BACKEND="${MASK_GENERATION_BACKEND:-vllm}"
MASK_BUILD_CUDA_VISIBLE_DEVICES="${MASK_BUILD_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
MASK_BUILD_TENSOR_PARALLEL_SIZE="${MASK_BUILD_TENSOR_PARALLEL_SIZE:-$NUM_PROCESSES}"
MASK_BUILD_GPU_MEMORY_UTILIZATION="${MASK_BUILD_GPU_MEMORY_UTILIZATION:-0.75}"
MASK_BUILD_BATCH_SIZE="${MASK_BUILD_BATCH_SIZE:-16}"
MASK_BUILD_MAX_NEW_TOKENS="${MASK_BUILD_MAX_NEW_TOKENS:-1024}"
MASK_BUILD_LIMIT="${MASK_BUILD_LIMIT:-0}"
MASK_BUILD_NUM_CANDIDATES="${MASK_BUILD_NUM_CANDIDATES:-3}"
MASK_BUILD_RETRIES="${MASK_BUILD_RETRIES:-1}"
MASKED_FAIL_BELOW_PASS_RATE="${MASKED_FAIL_BELOW_PASS_RATE:-0.20}"

PROBLEM_COLUMN="${PROBLEM_COLUMN:-problem}"
SOLUTION_COLUMN="${SOLUTION_COLUMN:-solution}"
ANSWER_COLUMN="${ANSWER_COLUMN:-Answer}"

mkdir -p data logs

if [[ ! -s "$RAW_DATASET" ]]; then
    if [[ "$PREPARE_RAW_DATASET" == "true" || "$PREPARE_RAW_DATASET" == "1" ]]; then
        echo "Preparing raw training data: $RAW_DATASET"
        python scripts/prepare_math_train_minus_math500.py \
            --output_path "$RAW_DATASET"
    else
        echo "ERROR: raw dataset not found or empty: $RAW_DATASET" >&2
        echo "Provide RAW_DATASET=/path/to/train.jsonl, or set PREPARE_RAW_DATASET=true to build MATH train-minus-MATH500." >&2
        exit 1
    fi
fi

should_build_masked=false
case "$BUILD_MASKED_DATASET" in
    true|1|yes|y)
        should_build_masked=true
        ;;
    false|0|no|n)
        should_build_masked=false
        ;;
    auto)
        if [[ ! -s "$MASKED_DATASET" ]]; then
            should_build_masked=true
        fi
        ;;
    *)
        echo "ERROR: BUILD_MASKED_DATASET must be auto, true, or false; got $BUILD_MASKED_DATASET" >&2
        exit 1
        ;;
esac

echo "================================================================================"
echo "MRO-SI full pipeline"
echo "raw dataset       : $RAW_DATASET"
echo "masked dataset    : $MASKED_DATASET"
echo "base model        : $BASE_MODEL"
echo "mask generator    : $MASK_GENERATOR_MODEL"
echo "mask backend      : $MASK_GENERATION_BACKEND"
echo "build masked data : $should_build_masked"
echo "train/eval GPUs   : $CUDA_VISIBLE_DEVICES (processes=$NUM_PROCESSES)"
echo "================================================================================"

if [[ "$should_build_masked" == "true" ]]; then
    echo
    echo "================================================================================"
    echo "STEP 1/3: Building masked derivation dataset"
    echo "================================================================================"

    if [[ "$MASK_GENERATION_BACKEND" == "vllm" ]]; then
        CUDA_VISIBLE_DEVICES="$MASK_BUILD_CUDA_VISIBLE_DEVICES" \
        python scripts/build_masked_derivations.py \
            --dataset_name_or_path "$RAW_DATASET" \
            --dataset_split train \
            --generation_backend vllm \
            --model_name_or_path "$MASK_GENERATOR_MODEL" \
            --output_path "$MASKED_DATASET" \
            --problem_column "$PROBLEM_COLUMN" \
            --solution_column "$SOLUTION_COLUMN" \
            --answer_column "$ANSWER_COLUMN" \
            --tensor_parallel_size "$MASK_BUILD_TENSOR_PARALLEL_SIZE" \
            --gpu_memory_utilization "$MASK_BUILD_GPU_MEMORY_UTILIZATION" \
            --batch_size "$MASK_BUILD_BATCH_SIZE" \
            --max_new_tokens "$MASK_BUILD_MAX_NEW_TOKENS" \
            --num_candidates "$MASK_BUILD_NUM_CANDIDATES" \
            --retries "$MASK_BUILD_RETRIES" \
            --limit "$MASK_BUILD_LIMIT" \
            --fail_below_pass_rate "$MASKED_FAIL_BELOW_PASS_RATE" \
            --resume \
            2>&1 | tee "logs/full_pipeline_mask_build_$(date +%Y%m%d_%H%M%S).log"
    elif [[ "$MASK_GENERATION_BACKEND" == "openai_api" ]]; then
        python scripts/build_masked_derivations.py \
            --dataset_name_or_path "$RAW_DATASET" \
            --dataset_split train \
            --generation_backend openai_api \
            --output_path "$MASKED_DATASET" \
            --problem_column "$PROBLEM_COLUMN" \
            --solution_column "$SOLUTION_COLUMN" \
            --answer_column "$ANSWER_COLUMN" \
            --batch_size "$MASK_BUILD_BATCH_SIZE" \
            --max_new_tokens "$MASK_BUILD_MAX_NEW_TOKENS" \
            --num_candidates "$MASK_BUILD_NUM_CANDIDATES" \
            --retries "$MASK_BUILD_RETRIES" \
            --limit "$MASK_BUILD_LIMIT" \
            --fail_below_pass_rate "$MASKED_FAIL_BELOW_PASS_RATE" \
            --resume \
            2>&1 | tee "logs/full_pipeline_mask_build_$(date +%Y%m%d_%H%M%S).log"
    else
        echo "ERROR: MASK_GENERATION_BACKEND must be vllm or openai_api; got $MASK_GENERATION_BACKEND" >&2
        exit 1
    fi
else
    echo "STEP 1/3: Reusing existing masked dataset: $MASKED_DATASET"
fi

if [[ ! -s "$MASKED_DATASET" ]]; then
    echo "ERROR: masked dataset was not produced: $MASKED_DATASET" >&2
    exit 1
fi

echo
echo "================================================================================"
echo "STEP 2/3 and 3/3: Running MRO-SI post-training and evaluation"
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
