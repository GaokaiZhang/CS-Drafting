#!/bin/bash
#SBATCH --job-name=fixed_window_compare
#SBATCH --output=/home/%u/logs/sbatch/fixed_window_%A_%a.out
#SBATCH --error=/home/%u/logs/sbatch/fixed_window_%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=144G
#SBATCH --time=12:00:00

set -euo pipefail

DATASET="${DATASET:-${1:-mmlu}}"
RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
MS_NAME="${MS_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MM_NAME="${MM_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}"
N_SAMPLES="${N_SAMPLES:-100}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$NUM_SHARDS" -gt 1 ]; then
  SHARD_INDEX="$((SHARD_OFFSET + SLURM_ARRAY_TASK_ID))"
fi
TRACE_SAMPLES="${TRACE_SAMPLES:-8}"
SMALL_WINDOW="${SMALL_WINDOW:-3}"
MIDDLE_WINDOW="${MIDDLE_WINDOW:-9}"
HIERARCHICAL_VARIANT="${HIERARCHICAL_VARIANT:-window_fill}"
PROXY_THRESHOLD="${PROXY_THRESHOLD:-}"
MAVG_WINDOW="${MAVG_WINDOW:-5}"
WINDOW_POLICY="${WINDOW_POLICY:-fixed}"
ADAPT_SMALL_WINDOW="${ADAPT_SMALL_WINDOW:-0}"
DYNAMIC_SMALL_WINDOW_MIN="${DYNAMIC_SMALL_WINDOW_MIN:-}"
DYNAMIC_SMALL_WINDOW_MAX="${DYNAMIC_SMALL_WINDOW_MAX:-}"
DYNAMIC_MIDDLE_WINDOW_MIN="${DYNAMIC_MIDDLE_WINDOW_MIN:-}"
DYNAMIC_MIDDLE_WINDOW_MAX="${DYNAMIC_MIDDLE_WINDOW_MAX:-}"
DYNAMIC_ACCEPTANCE_LOW="${DYNAMIC_ACCEPTANCE_LOW:-0.7}"
DYNAMIC_ACCEPTANCE_HIGH="${DYNAMIC_ACCEPTANCE_HIGH:-0.9}"
DYNAMIC_WINDOW_STEP="${DYNAMIC_WINDOW_STEP:-1}"
DYNAMIC_UTILITY_MARGIN="${DYNAMIC_UTILITY_MARGIN:-0.0}"
MAX_LENGTH="${MAX_LENGTH:-200}"
OUTPUT_SLUG="${OUTPUT_SLUG:-qwen_1p5b_7b_14b}"
RESULT_DIR="${RESULT_DIR:-$RUN_ROOT/results}"
SHARD_SUFFIX=""
if [ "$NUM_SHARDS" -gt 1 ]; then
  SHARD_SUFFIX="_shard${SHARD_INDEX}of${NUM_SHARDS}"
fi
if [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
  echo "Invalid shard index ${SHARD_INDEX} for NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
fi
RESULT_PATH="${RESULT_PATH:-${OUTPUT_JSON:-$RESULT_DIR/fixed_window_compare_${DATASET}_${OUTPUT_SLUG}${SHARD_SUFFIX}.json}}"

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

for attempt in $(seq 1 6); do
  if python - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 2
print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
PY
  then
    break
  fi
  if [ "$attempt" -eq 6 ]; then
    echo "CUDA preflight failed after ${attempt} attempts" >&2
    exit 1
  fi
  sleep 10
done

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
echo "Shard settings: offset=${SHARD_OFFSET} index=${SHARD_INDEX} num_shards=${NUM_SHARDS}"

MS_NAME_RESOLVED="$(resolve_snapshot "$MS_NAME")"
MM_NAME_RESOLVED="$(resolve_snapshot "$MM_NAME")"
ML_NAME_RESOLVED="$(resolve_snapshot "$ML_NAME")"
echo "Resolved models: small=${MS_NAME_RESOLVED} middle=${MM_NAME_RESOLVED} large=${ML_NAME_RESOLVED}"

EXTRA_ARGS=(
  --hierarchical_variant "$HIERARCHICAL_VARIANT"
  --mavg_window "$MAVG_WINDOW"
  --window_policy "$WINDOW_POLICY"
  --dynamic_acceptance_low "$DYNAMIC_ACCEPTANCE_LOW"
  --dynamic_acceptance_high "$DYNAMIC_ACCEPTANCE_HIGH"
  --dynamic_window_step "$DYNAMIC_WINDOW_STEP"
  --dynamic_utility_margin "$DYNAMIC_UTILITY_MARGIN"
)
if [ "$ADAPT_SMALL_WINDOW" = "1" ] || [ "$ADAPT_SMALL_WINDOW" = "true" ] || [ "$ADAPT_SMALL_WINDOW" = "TRUE" ]; then
  EXTRA_ARGS+=(--adapt_small_window)
fi
if [ -n "$PROXY_THRESHOLD" ]; then
  EXTRA_ARGS+=(--proxy_threshold "$PROXY_THRESHOLD")
fi
if [ -n "$DYNAMIC_SMALL_WINDOW_MIN" ]; then
  EXTRA_ARGS+=(--dynamic_small_window_min "$DYNAMIC_SMALL_WINDOW_MIN")
fi
if [ -n "$DYNAMIC_SMALL_WINDOW_MAX" ]; then
  EXTRA_ARGS+=(--dynamic_small_window_max "$DYNAMIC_SMALL_WINDOW_MAX")
fi
if [ -n "$DYNAMIC_MIDDLE_WINDOW_MIN" ]; then
  EXTRA_ARGS+=(--dynamic_middle_window_min "$DYNAMIC_MIDDLE_WINDOW_MIN")
fi
if [ -n "$DYNAMIC_MIDDLE_WINDOW_MAX" ]; then
  EXTRA_ARGS+=(--dynamic_middle_window_max "$DYNAMIC_MIDDLE_WINDOW_MAX")
fi

python -u main_fixed_window.py \
  --mode compare \
  --dataset "$DATASET" \
  --ms_name "$MS_NAME_RESOLVED" \
  --mm_name "$MM_NAME_RESOLVED" \
  --ml_name "$ML_NAME_RESOLVED" \
  --small_device "$SMALL_DEVICE" \
  --middle_device "$MIDDLE_DEVICE" \
  --large_device "$LARGE_DEVICE" \
  --n_samples "$N_SAMPLES" \
  --num_shards "$NUM_SHARDS" \
  --shard_index "$SHARD_INDEX" \
  --small_window "$SMALL_WINDOW" \
  --middle_window "$MIDDLE_WINDOW" \
  --max_length "$MAX_LENGTH" \
  --trace_samples "$TRACE_SAMPLES" \
  --device cuda:0 \
  --output "$RESULT_PATH" \
  "${EXTRA_ARGS[@]}"
