#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MROSI_ROOT="${MROSI_ROOT:-$ROOT_DIR/..}"
MROSI_ROOT="$(cd "$MROSI_ROOT" && pwd)"
BASE_MODEL="${BASE_MODEL:-$MROSI_ROOT/../models/Qwen3-1.7B}"
MASKED_DATASET="${MASKED_DATASET:-$MROSI_ROOT/openthoughts_math_30k_masked.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aomp_opsd}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_results}"
WANDB_PROJECT="${WANDB_PROJECT:-AOMP-OPSD}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ROOT_DIR/configs/accelerate_8gpu_ddp.yaml}"

MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-25}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
TOP_K_LOSS="${TOP_K_LOSS:-200}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.05}"

AOMP_OPSD_VARIANT="${AOMP_OPSD_VARIANT:-full_aomp_opsd}"
AOMP_OPSD_ENABLED="${AOMP_OPSD_ENABLED:-true}"
AOMP_OPSD_STEP_SIZE="${AOMP_OPSD_STEP_SIZE:-1.0}"
AOMP_OPSD_LOOKAHEAD_LR="${AOMP_OPSD_LOOKAHEAD_LR:-1.0e-5}"
AOMP_OPSD_GROUP_SIZE="${AOMP_OPSD_GROUP_SIZE:-2}"
AOMP_OPSD_AUDIT_SOURCE="${AOMP_OPSD_AUDIT_SOURCE:-auto}"
AOMP_OPSD_AUDIT_TEMPERATURE="${AOMP_OPSD_AUDIT_TEMPERATURE:-1.0}"
AOMP_OPSD_POSITIVE_PATH_WEIGHT="${AOMP_OPSD_POSITIVE_PATH_WEIGHT:-1.0}"
AOMP_OPSD_NEGATIVE_PATH_WEIGHT="${AOMP_OPSD_NEGATIVE_PATH_WEIGHT:-1.0}"
AOMP_OPSD_TEACHER_KL_WEIGHT="${AOMP_OPSD_TEACHER_KL_WEIGHT:-1.0}"
AOMP_OPSD_SELF_DISTILL_WEIGHT="${AOMP_OPSD_SELF_DISTILL_WEIGHT:-1.0}"
AOMP_OPSD_TOKEN_WEIGHT_NORMALIZATION="${AOMP_OPSD_TOKEN_WEIGHT_NORMALIZATION:-sequence_sum}"
AOMP_OPSD_USE_TEACHER_RELIABILITY="${AOMP_OPSD_USE_TEACHER_RELIABILITY:-true}"
AOMP_OPSD_USE_STUDENT_UNCERTAINTY="${AOMP_OPSD_USE_STUDENT_UNCERTAINTY:-true}"
AOMP_OPSD_USE_PREFIX_RELIABILITY="${AOMP_OPSD_USE_PREFIX_RELIABILITY:-true}"
AOMP_OPSD_PREFIX_DECAY_LAMBDA="${AOMP_OPSD_PREFIX_DECAY_LAMBDA:-1.0}"
AOMP_OPSD_MAX_TOKEN_WEIGHT="${AOMP_OPSD_MAX_TOKEN_WEIGHT:-5.0}"
AOMP_OPSD_MIN_TOKEN_WEIGHT="${AOMP_OPSD_MIN_TOKEN_WEIGHT:-0.0}"
AOMP_OPSD_AUDIT_BUDGET="${AOMP_OPSD_AUDIT_BUDGET:-0}"
AOMP_OPSD_AUDIT_START_STEP="${AOMP_OPSD_AUDIT_START_STEP:-0}"
AOMP_OPSD_AUDIT_EVERY_N_STEPS="${AOMP_OPSD_AUDIT_EVERY_N_STEPS:-2}"
AOMP_OPSD_VLLM_APPROX_LOOKAHEAD="${AOMP_OPSD_VLLM_APPROX_LOOKAHEAD:-false}"
AOMP_OPSD_LOG_DIAGNOSTICS="${AOMP_OPSD_LOG_DIAGNOSTICS:-true}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ -z "${NUM_PROCESSES:-}" ]]; then
    IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    NUM_PROCESSES="${#_visible_gpus[@]}"
fi
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_MAIN_PROCESS_PORT="${TRAIN_MAIN_PROCESS_PORT:-12959}"
TRAIN_VLLM_GPU_MEMORY_UTILIZATION="${TRAIN_VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
TRAIN_USE_VLLM="${TRAIN_USE_VLLM:-true}"
TRAIN_VLLM_MODE="${TRAIN_VLLM_MODE:-colocate}"
TRAIN_VLLM_TENSOR_PARALLEL_SIZE="${TRAIN_VLLM_TENSOR_PARALLEL_SIZE:-1}"
TRAIN_VLLM_SYNC_FREQUENCY="${TRAIN_VLLM_SYNC_FREQUENCY:-1}"
SHUFFLE_DATASET="${SHUFFLE_DATASET:-true}"

RUN_CONFIG="${RUN_CONFIG:-qwen3_1p7b_${AOMP_OPSD_VARIANT}_${MAX_STEPS}step}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-false}"

EVAL_DATASETS="${EVAL_DATASETS:-aime24 aime25 hmmt25}"
EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-$(seq "$SAVE_STEPS" "$SAVE_STEPS" "$MAX_STEPS" | tr '\n' ' ')}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-$NUM_PROCESSES}"
EVAL_VAL_N="${EVAL_VAL_N:-12}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-38912}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-true}"

TARGET_DIR="$OUTPUT_ROOT/$RUN_CONFIG"
mkdir -p logs "$OUTPUT_ROOT" "$EVAL_OUTPUT_ROOT"

if [[ ! -d "$MROSI_ROOT" ]]; then
    echo "ERROR: MRO-SI root not found: $MROSI_ROOT" >&2
    exit 1
fi
if [[ ! -s "$MASKED_DATASET" ]]; then
    echo "ERROR: masked dataset not found or empty: $MASKED_DATASET" >&2
    exit 1
fi
if [[ "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "1" ]]; then
    if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
        echo "ERROR: accelerate config not found: $ACCELERATE_CONFIG" >&2
        exit 1
    fi
    if [[ -e "$TARGET_DIR" && "$ALLOW_EXISTING_RUN" != "true" && "$ALLOW_EXISTING_RUN" != "1" ]]; then
        echo "ERROR: target run already exists: $TARGET_DIR" >&2
        echo "Use RUN_CONFIG=... or ALLOW_EXISTING_RUN=true." >&2
        exit 1
    fi
fi

echo "================================================================================"
echo "AOMP-OPSD run"
echo "run config       : $RUN_CONFIG"
echo "variant          : $AOMP_OPSD_VARIANT"
echo "MRO-SI root      : $MROSI_ROOT"
echo "base model       : $BASE_MODEL"
echo "masked dataset   : $MASKED_DATASET"
echo "output dir       : $TARGET_DIR"
echo "max/save steps   : $MAX_STEPS / $SAVE_STEPS"
echo "train GPUs       : $CUDA_VISIBLE_DEVICES (processes=$NUM_PROCESSES, grad_accum=$TRAIN_GRAD_ACCUM)"
echo "================================================================================"

if [[ "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "1" ]]; then
    TRAIN_VLLM_ARGS=()
    if [[ "$TRAIN_USE_VLLM" == "true" || "$TRAIN_USE_VLLM" == "1" ]]; then
        TRAIN_VLLM_ARGS=(
            --use_vllm
            --vllm_mode "$TRAIN_VLLM_MODE"
            --vllm_gpu_memory_utilization "$TRAIN_VLLM_GPU_MEMORY_UTILIZATION"
            --vllm_tensor_parallel_size "$TRAIN_VLLM_TENSOR_PARALLEL_SIZE"
            --vllm_sync_frequency "$TRAIN_VLLM_SYNC_FREQUENCY"
        )
    fi

    MROSI_ROOT="$MROSI_ROOT" \
    PYTHONPATH="$ROOT_DIR:$MROSI_ROOT:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    python -m accelerate.commands.launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_PROCESSES" \
        --gradient_accumulation_steps "$TRAIN_GRAD_ACCUM" \
        --main_process_port "$TRAIN_MAIN_PROCESS_PORT" \
        aomp_opsd/train.py \
        --model_name_or_path "$BASE_MODEL" \
        --dataset_name_or_path "$MASKED_DATASET" \
        --masked_derivation_column masked_derivation \
        --skip_missing_masked_derivation true \
        --mrosi_self_imitation_weight 0.0 \
        --learning_rate "$LEARNING_RATE" \
        --max_grad_norm 0.1 \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --ddp_find_unused_parameters false \
        --gradient_accumulation_steps "$TRAIN_GRAD_ACCUM" \
        --output_dir "$OUTPUT_ROOT" \
        --run_config "$RUN_CONFIG" \
        --num_train_epochs 30 \
        --max_steps "$MAX_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --logging_steps 1 \
        --max_length 20000 \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --shuffle_dataset "$SHUFFLE_DATASET" \
        --attn_implementation flash_attention_2 \
        --torch_dtype bfloat16 \
        --beta 0 \
        --use_peft \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --temperature 1.1 \
        --top_p 0.95 \
        --top_k 20 \
        --lmbda 1 \
        --top_k_loss "$TOP_K_LOSS" \
        --jsd_token_clip "$JSD_TOKEN_CLIP" \
        --wandb_project "$WANDB_PROJECT" \
        "${TRAIN_VLLM_ARGS[@]}" \
        --aomp_opsd_enabled "$AOMP_OPSD_ENABLED" \
        --variant "$AOMP_OPSD_VARIANT" \
        --step_size "$AOMP_OPSD_STEP_SIZE" \
        --lookahead_lr "$AOMP_OPSD_LOOKAHEAD_LR" \
        --group_size "$AOMP_OPSD_GROUP_SIZE" \
        --audit_source "$AOMP_OPSD_AUDIT_SOURCE" \
        --audit_temperature "$AOMP_OPSD_AUDIT_TEMPERATURE" \
        --positive_path_weight "$AOMP_OPSD_POSITIVE_PATH_WEIGHT" \
        --negative_path_weight "$AOMP_OPSD_NEGATIVE_PATH_WEIGHT" \
        --teacher_kl_weight "$AOMP_OPSD_TEACHER_KL_WEIGHT" \
        --self_distill_weight "$AOMP_OPSD_SELF_DISTILL_WEIGHT" \
        --token_weight_normalization "$AOMP_OPSD_TOKEN_WEIGHT_NORMALIZATION" \
        --use_teacher_reliability "$AOMP_OPSD_USE_TEACHER_RELIABILITY" \
        --use_student_uncertainty "$AOMP_OPSD_USE_STUDENT_UNCERTAINTY" \
        --use_prefix_reliability "$AOMP_OPSD_USE_PREFIX_RELIABILITY" \
        --prefix_decay_lambda "$AOMP_OPSD_PREFIX_DECAY_LAMBDA" \
        --max_token_weight "$AOMP_OPSD_MAX_TOKEN_WEIGHT" \
        --min_token_weight "$AOMP_OPSD_MIN_TOKEN_WEIGHT" \
        --audit_budget "$AOMP_OPSD_AUDIT_BUDGET" \
        --audit_start_step "$AOMP_OPSD_AUDIT_START_STEP" \
        --audit_every_n_steps "$AOMP_OPSD_AUDIT_EVERY_N_STEPS" \
        --vllm_approx_lookahead "$AOMP_OPSD_VLLM_APPROX_LOOKAHEAD" \
        --log_diagnostics "$AOMP_OPSD_LOG_DIAGNOSTICS" \
        2>&1 | tee "logs/${RUN_CONFIG}_train_$(date +%Y%m%d_%H%M%S).log"
fi

eval_one() {
    local checkpoint_step="$1"
    local dataset="$2"
    local checkpoint_dir="$TARGET_DIR/checkpoint-$checkpoint_step"
    local output_file="$EVAL_OUTPUT_ROOT/${RUN_CONFIG}_checkpoint-${checkpoint_step}_${dataset}_valn${EVAL_VAL_N}.json"

    if [[ ! -d "$checkpoint_dir" ]]; then
        echo "Skipping missing checkpoint: $checkpoint_dir" >&2
        return
    fi
    if [[ -s "$output_file" && ( "$SKIP_EXISTING_EVAL" == "true" || "$SKIP_EXISTING_EVAL" == "1" ) ]]; then
        echo "Skipping existing eval: $output_file"
        return
    fi

    MROSI_ROOT="$MROSI_ROOT" \
    PYTHONPATH="$ROOT_DIR:$MROSI_ROOT:${PYTHONPATH:-}" \
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
    python "$MROSI_ROOT/eval/evaluate_math.py" \
        --base_model "$BASE_MODEL" \
        --checkpoint_dir "$checkpoint_dir" \
        --dataset "$dataset" \
        --val_n "$EVAL_VAL_N" \
        --temperature "$EVAL_TEMPERATURE" \
        --tensor_parallel_size "$EVAL_TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
        --output_file "$output_file" \
        2>&1 | tee "logs/${RUN_CONFIG}_checkpoint-${checkpoint_step}_${dataset}_valn${EVAL_VAL_N}_$(date +%Y%m%d_%H%M%S).log"
}

if [[ "$RUN_EVAL" == "true" || "$RUN_EVAL" == "1" ]]; then
    for checkpoint_step in $EVAL_CHECKPOINTS; do
        for dataset in $EVAL_DATASETS; do
            eval_one "$checkpoint_step" "$dataset"
        done
    done
fi

echo "Done."
