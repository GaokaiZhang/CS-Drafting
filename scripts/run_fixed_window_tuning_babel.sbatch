#!/bin/bash
#SBATCH --job-name=fixed_window_tune
#SBATCH --output=/home/%u/logs/sbatch/fixed_window_tune_%A.out
#SBATCH --error=/home/%u/logs/sbatch/fixed_window_tune_%A.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00

set -euo pipefail

RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
DATASET="${DATASET:-gsm8k}"
MS_NAME="${MS_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MM_NAME="${MM_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}"
N_SAMPLES="${N_SAMPLES:-20}"
TRACE_SAMPLES="${TRACE_SAMPLES:-0}"
SMALL_WINDOW="${SMALL_WINDOW:-3}"
SMALL_WINDOWS="${SMALL_WINDOWS:-}"
MIDDLE_WINDOWS="${MIDDLE_WINDOWS:-6:9:12:15}"
HIERARCHICAL_VARIANTS="${HIERARCHICAL_VARIANTS:-double_layer}"
WINDOW_POLICIES="${WINDOW_POLICIES:-fixed:utility}"
DYNAMIC_MIDDLE_WINDOW_MIN="${DYNAMIC_MIDDLE_WINDOW_MIN:-3}"
DYNAMIC_MIDDLE_WINDOW_MAX="${DYNAMIC_MIDDLE_WINDOW_MAX:-18}"
DYNAMIC_ACCEPTANCE_LOW="${DYNAMIC_ACCEPTANCE_LOW:-0.6}"
DYNAMIC_ACCEPTANCE_HIGH="${DYNAMIC_ACCEPTANCE_HIGH:-0.8}"
DYNAMIC_WINDOW_STEP="${DYNAMIC_WINDOW_STEP:-1}"
DYNAMIC_UTILITY_MARGIN="${DYNAMIC_UTILITY_MARGIN:-0.0}"
MAX_LENGTH="${MAX_LENGTH:-200}"
OUTPUT_SLUG="${OUTPUT_SLUG:-qwen_0p5b_1p5b_14b_tuning}"
RESULT_DIR="${RESULT_DIR:-$RUN_ROOT/results}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_JSON:-$RESULT_DIR/fixed_window_tuning_${DATASET}_${OUTPUT_SLUG}.json}}"

mkdir -p "/home/$USER/logs/sbatch"

source "$RUN_ROOT/scripts/babel_hf_env.sh"
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=8

eval "$(conda shell.bash hook)"
conda activate mlsys-fw

cd "$RUN_ROOT"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [ -z "${SMALL_DEVICE:-}" ] || [ -z "${MIDDLE_DEVICE:-}" ] || [ -z "${LARGE_DEVICE:-}" ]; then
  if [ "${GPU_COUNT}" -ge 3 ]; then
    SMALL_DEVICE="${SMALL_DEVICE:-cuda:0}"
    MIDDLE_DEVICE="${MIDDLE_DEVICE:-cuda:1}"
    LARGE_DEVICE="${LARGE_DEVICE:-cuda:2}"
  else
    SMALL_DEVICE="${SMALL_DEVICE:-cuda:0}"
    MIDDLE_DEVICE="${MIDDLE_DEVICE:-cuda:0}"
    LARGE_DEVICE="${LARGE_DEVICE:-cuda:1}"
  fi
fi

echo "Resolved devices: small=${SMALL_DEVICE} middle=${MIDDLE_DEVICE} large=${LARGE_DEVICE}"

EXTRA_ARGS=()
if [ -n "$SMALL_WINDOWS" ]; then
  EXTRA_ARGS+=(--small_windows "$SMALL_WINDOWS")
fi

python -u main_fixed_window_tuning.py \
  --dataset "$DATASET" \
  --ms_name "$MS_NAME" \
  --mm_name "$MM_NAME" \
  --ml_name "$ML_NAME" \
  --small_device "$SMALL_DEVICE" \
  --middle_device "$MIDDLE_DEVICE" \
  --large_device "$LARGE_DEVICE" \
  --n_samples "$N_SAMPLES" \
  --small_window "$SMALL_WINDOW" \
  --middle_windows "$MIDDLE_WINDOWS" \
  --hierarchical_variants "$HIERARCHICAL_VARIANTS" \
  --window_policies "$WINDOW_POLICIES" \
  --dynamic_middle_window_min "$DYNAMIC_MIDDLE_WINDOW_MIN" \
  --dynamic_middle_window_max "$DYNAMIC_MIDDLE_WINDOW_MAX" \
  --dynamic_acceptance_low "$DYNAMIC_ACCEPTANCE_LOW" \
  --dynamic_acceptance_high "$DYNAMIC_ACCEPTANCE_HIGH" \
  --dynamic_window_step "$DYNAMIC_WINDOW_STEP" \
  --dynamic_utility_margin "$DYNAMIC_UTILITY_MARGIN" \
  --max_length "$MAX_LENGTH" \
  --trace_samples "$TRACE_SAMPLES" \
  --output "$RESULT_PATH" \
  "${EXTRA_ARGS[@]}"
