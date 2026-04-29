#!/bin/bash
#SBATCH --job-name=acsd_compare
#SBATCH --output=/home/%u/logs/sbatch/acsd_compare_%A.out
#SBATCH --error=/home/%u/logs/sbatch/acsd_compare_%A.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail

DATASET="${DATASET:-${1:-mmlu}}"
RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
BRANCH_ROOT="${BRANCH_ROOT:-$RUN_ROOT}"
MS_NAME="${MS_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MM_NAME="${MM_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}"
N_SAMPLES="${N_SAMPLES:-100}"
NUM_SHARDS="${NUM_SHARDS:-1}"
if [ -n "${SHARD_INDEX:-}" ]; then
  EFFECTIVE_SHARD_INDEX="$SHARD_INDEX"
else
  EFFECTIVE_SHARD_INDEX="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${SHARD_OFFSET:-0} ))"
fi
TRACE_SAMPLES="${TRACE_SAMPLES:-4}"
MODE="${MODE:-compare_all}"
K_S="${K_S:-5}"
K_M="${K_M:-4}"
LENIENCY="${LENIENCY:-1}"
TAU="${TAU:-0.4}"
WINDOW_SIZE="${WINDOW_SIZE:-20}"
MAX_LENGTH="${MAX_LENGTH:-200}"
MIDDLE_REFILL="${MIDDLE_REFILL:-1}"
SHADOW_MIDDLE="${SHADOW_MIDDLE:-0}"
DRAFT_WINDOW_POLICY="${DRAFT_WINDOW_POLICY:-fixed}"
DYNAMIC_K_S_MIN="${DYNAMIC_K_S_MIN:-2}"
DYNAMIC_K_S_MAX="${DYNAMIC_K_S_MAX:-8}"
DYNAMIC_K_M_MIN="${DYNAMIC_K_M_MIN:-2}"
DYNAMIC_K_M_MAX="${DYNAMIC_K_M_MAX:-8}"
DYNAMIC_ACCEPTANCE_LOW="${DYNAMIC_ACCEPTANCE_LOW:-0.5}"
DYNAMIC_ACCEPTANCE_HIGH="${DYNAMIC_ACCEPTANCE_HIGH:-0.8}"
DYNAMIC_WINDOW_STEP="${DYNAMIC_WINDOW_STEP:-1}"
OUTPUT_SLUG="${OUTPUT_SLUG:-acsd_clean_100}"
RESULT_DIR="${RESULT_DIR:-$RUN_ROOT/results}"
if [ "$NUM_SHARDS" -gt 1 ]; then
  SHARD_SUFFIX="_shard${EFFECTIVE_SHARD_INDEX}of${NUM_SHARDS}"
else
  SHARD_SUFFIX=""
fi
RESULT_PATH="${RESULT_PATH:-$RESULT_DIR/acsd_compare_${DATASET}_${OUTPUT_SLUG}${SHARD_SUFFIX}.json}"

mkdir -p "/home/$USER/logs/sbatch"

source "$RUN_ROOT/scripts/babel_hf_env.sh"
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=8
LOCAL_MODEL_ROOT="${HF_LOCAL_MODEL_ROOT:-/data/user_data/$USER/hf_models}"

resolve_snapshot() {
  local model_name="$1"
  if [ -z "$model_name" ] || [ -d "$model_name" ] || [[ "$model_name" != */* ]]; then
    printf '%s\n' "$model_name"
    return
  fi

  local namespace="${model_name%%/*}"
  local repo="${model_name#*/}"
  local local_dir="${LOCAL_MODEL_ROOT}/${namespace}/${repo}"
  local repo_cache
  local latest_snapshot
  if [ -d "$local_dir" ]; then
    printf '%s\n' "$local_dir"
    return
  fi
  for root in "${HF_HUB_CACHE:-}" "${HF_HOME:-}/hub" "/data/hf_cache/hub" "/mnt/data/xuandong/hf_models"; do
    if [ -z "$root" ]; then
      continue
    fi
    repo_cache="$root/models--${namespace}--${repo}"
    if [ ! -d "$repo_cache" ]; then
      continue
    fi
    if [ -f "$repo_cache/refs/main" ]; then
      local revision
      revision="$(<"$repo_cache/refs/main")"
      if [ -d "$repo_cache/snapshots/$revision" ]; then
        printf '%s\n' "$repo_cache/snapshots/$revision"
        return
      fi
    fi
    latest_snapshot="$(ls -d "$repo_cache"/snapshots/* 2>/dev/null | sort | tail -n 1 || true)"
    if [ -n "$latest_snapshot" ]; then
      printf '%s\n' "$latest_snapshot"
      return
    fi
  done

  printf '%s\n' "$model_name"
}

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

if [ "${GPU_COUNT}" -ge 3 ]; then
  SMALL_DEVICE="${SMALL_DEVICE:-cuda:0}"
  MIDDLE_DEVICE="${MIDDLE_DEVICE:-cuda:1}"
  LARGE_DEVICE="${LARGE_DEVICE:-cuda:2}"
else
  SMALL_DEVICE="${SMALL_DEVICE:-cuda:0}"
  MIDDLE_DEVICE="${MIDDLE_DEVICE:-cuda:0}"
  LARGE_DEVICE="${LARGE_DEVICE:-cuda:1}"
fi

echo "Resolved devices: small=${SMALL_DEVICE} middle=${MIDDLE_DEVICE} large=${LARGE_DEVICE}"

MS_NAME_RESOLVED="$(resolve_snapshot "$MS_NAME")"
MM_NAME_RESOLVED="$(resolve_snapshot "$MM_NAME")"
ML_NAME_RESOLVED="$(resolve_snapshot "$ML_NAME")"
echo "Resolved models: small=${MS_NAME_RESOLVED} middle=${MM_NAME_RESOLVED} large=${ML_NAME_RESOLVED}"

ABLATION_ARGS=()
if [ "$MIDDLE_REFILL" = "0" ]; then
  ABLATION_ARGS+=(--no_middle_refill)
else
  ABLATION_ARGS+=(--middle_refill)
fi
if [ "$SHADOW_MIDDLE" = "1" ]; then
  ABLATION_ARGS+=(--shadow_middle)
fi

python -u main_acsd_compare.py \
  --branch_root "$BRANCH_ROOT" \
  --mode "$MODE" \
  --dataset "$DATASET" \
  --ms_name "$MS_NAME_RESOLVED" \
  --mm_name "$MM_NAME_RESOLVED" \
  --ml_name "$ML_NAME_RESOLVED" \
  --small_device "$SMALL_DEVICE" \
  --middle_device "$MIDDLE_DEVICE" \
  --large_device "$LARGE_DEVICE" \
  --n_samples "$N_SAMPLES" \
  --num_shards "$NUM_SHARDS" \
  --shard_index "$EFFECTIVE_SHARD_INDEX" \
  --k_s "$K_S" \
  --k_m "$K_M" \
  --leniency "$LENIENCY" \
  --tau "$TAU" \
  --window_size "$WINDOW_SIZE" \
  --max_length "$MAX_LENGTH" \
  --trace_samples "$TRACE_SAMPLES" \
  "${ABLATION_ARGS[@]}" \
  --draft_window_policy "$DRAFT_WINDOW_POLICY" \
  --dynamic_k_s_min "$DYNAMIC_K_S_MIN" \
  --dynamic_k_s_max "$DYNAMIC_K_S_MAX" \
  --dynamic_k_m_min "$DYNAMIC_K_M_MIN" \
  --dynamic_k_m_max "$DYNAMIC_K_M_MAX" \
  --dynamic_acceptance_low "$DYNAMIC_ACCEPTANCE_LOW" \
  --dynamic_acceptance_high "$DYNAMIC_ACCEPTANCE_HIGH" \
  --dynamic_window_step "$DYNAMIC_WINDOW_STEP" \
  --output "$RESULT_PATH"
