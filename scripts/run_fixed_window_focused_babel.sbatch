#!/bin/bash
#SBATCH --job-name=fixed_window_focus
#SBATCH --output=/home/%u/logs/sbatch/fixed_window_focus_%A_%a.out
#SBATCH --error=/home/%u/logs/sbatch/fixed_window_focus_%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00

set -euo pipefail

RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
DATASETS_RAW="${DATASETS:-${DATASET:-gsm8k}}"
MS_NAME="${MS_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MM_NAME="${MM_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}"
N_SAMPLES="${N_SAMPLES:-50}"
TRACE_SAMPLES="${TRACE_SAMPLES:-0}"
CONFIG_SPECS="${CONFIG_SPECS:-baseline_sw3|baseline|3|-|-|-|;fixed_cascade_sw3_mw6|hierarchical|3|6|double_layer|fixed|baseline_sw3;adaptive_refill_sw3_mw4|hierarchical|3|4|double_layer_refill_on_full_accept|utility|baseline_sw3;selective_refill_sw3_mw4|hierarchical|3|4|selective_route_refill_on_full_accept|utility|baseline_sw3;cost_aware_selective_sw3_mw6|hierarchical|3|6|cost_aware_selective_route_refill_on_full_accept|utility|baseline_sw3;baseline_sw5|baseline|5|-|-|-|}"
DYNAMIC_SMALL_WINDOW_MIN="${DYNAMIC_SMALL_WINDOW_MIN:-}"
DYNAMIC_SMALL_WINDOW_MAX="${DYNAMIC_SMALL_WINDOW_MAX:-}"
DYNAMIC_MIDDLE_WINDOW_MIN="${DYNAMIC_MIDDLE_WINDOW_MIN:-3}"
DYNAMIC_MIDDLE_WINDOW_MAX="${DYNAMIC_MIDDLE_WINDOW_MAX:-18}"
DYNAMIC_ACCEPTANCE_LOW="${DYNAMIC_ACCEPTANCE_LOW:-0.6}"
DYNAMIC_ACCEPTANCE_HIGH="${DYNAMIC_ACCEPTANCE_HIGH:-0.8}"
DYNAMIC_WINDOW_STEP="${DYNAMIC_WINDOW_STEP:-1}"
DYNAMIC_UTILITY_MARGIN="${DYNAMIC_UTILITY_MARGIN:-0.0}"
SELECTIVE_ROUTE_WARMUP_BLOCKS="${SELECTIVE_ROUTE_WARMUP_BLOCKS:-1}"
SELECTIVE_ROUTE_HISTORY_WINDOW="${SELECTIVE_ROUTE_HISTORY_WINDOW:-4}"
SELECTIVE_ROUTE_UTILITY_MARGIN="${SELECTIVE_ROUTE_UTILITY_MARGIN:-0.0}"
SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_LOW="${SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_LOW:-0.7}"
SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_HIGH="${SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_HIGH:-0.85}"
SELECTIVE_ROUTE_MIDDLE_ACCEPTANCE_LOW="${SELECTIVE_ROUTE_MIDDLE_ACCEPTANCE_LOW:-}"
SELECTIVE_ROUTE_PROBE_INTERVAL="${SELECTIVE_ROUTE_PROBE_INTERVAL:-}"
ADAPT_SMALL_WINDOW="${ADAPT_SMALL_WINDOW:-0}"
MAX_LENGTH="${MAX_LENGTH:-200}"
OUTPUT_SLUG="${OUTPUT_SLUG:-qwen_0p5b_1p5b_14b_focus}"
RESULT_DIR="${RESULT_DIR:-$RUN_ROOT/results}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$NUM_SHARDS" -gt 1 ]; then
  SHARD_INDEX="$((SHARD_OFFSET + SLURM_ARRAY_TASK_ID))"
fi
SHARD_SUFFIX=""
if [ "$NUM_SHARDS" -gt 1 ]; then
  SHARD_SUFFIX="_shard${SHARD_INDEX}of${NUM_SHARDS}"
fi
if [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
  echo "Invalid shard index ${SHARD_INDEX} for NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
fi

mkdir -p "/home/$USER/logs/sbatch"

source "$RUN_ROOT/scripts/babel_hf_env.sh"
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=8
LOCAL_MODEL_ROOT="${HF_LOCAL_MODEL_ROOT:-/data/user_data/$USER/hf_models}"

eval "$(conda shell.bash hook)"
conda activate mlsys-fw

cd "$RUN_ROOT"

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
  elif [ "${GPU_COUNT}" -ge 2 ]; then
    SMALL_DEVICE="${SMALL_DEVICE:-cuda:0}"
    MIDDLE_DEVICE="${MIDDLE_DEVICE:-cuda:0}"
    LARGE_DEVICE="${LARGE_DEVICE:-cuda:1}"
  else
    SMALL_DEVICE="${SMALL_DEVICE:-cuda:0}"
    MIDDLE_DEVICE="${MIDDLE_DEVICE:-cuda:0}"
    LARGE_DEVICE="${LARGE_DEVICE:-cuda:0}"
  fi
fi

echo "Resolved devices: small=${SMALL_DEVICE} middle=${MIDDLE_DEVICE} large=${LARGE_DEVICE}"
echo "Datasets: ${DATASETS_RAW}"
echo "Shard settings: offset=${SHARD_OFFSET} index=${SHARD_INDEX} num_shards=${NUM_SHARDS}"

MS_NAME_RESOLVED="$(resolve_snapshot "$MS_NAME")"
MM_NAME_RESOLVED="$(resolve_snapshot "$MM_NAME")"
ML_NAME_RESOLVED="$(resolve_snapshot "$ML_NAME")"
echo "Resolved models: small=${MS_NAME_RESOLVED} middle=${MM_NAME_RESOLVED} large=${ML_NAME_RESOLVED}"

EXTRA_ARGS=()
if [ "$ADAPT_SMALL_WINDOW" = "1" ] || [ "$ADAPT_SMALL_WINDOW" = "true" ] || [ "$ADAPT_SMALL_WINDOW" = "TRUE" ]; then
  EXTRA_ARGS+=(--adapt_small_window)
fi
if [ -n "$DYNAMIC_SMALL_WINDOW_MIN" ]; then
  EXTRA_ARGS+=(--dynamic_small_window_min "$DYNAMIC_SMALL_WINDOW_MIN")
fi
if [ -n "$DYNAMIC_SMALL_WINDOW_MAX" ]; then
  EXTRA_ARGS+=(--dynamic_small_window_max "$DYNAMIC_SMALL_WINDOW_MAX")
fi
if [ -n "$SELECTIVE_ROUTE_MIDDLE_ACCEPTANCE_LOW" ]; then
  EXTRA_ARGS+=(--selective_route_middle_acceptance_low "$SELECTIVE_ROUTE_MIDDLE_ACCEPTANCE_LOW")
fi
if [ -n "$SELECTIVE_ROUTE_PROBE_INTERVAL" ]; then
  EXTRA_ARGS+=(--selective_route_probe_interval "$SELECTIVE_ROUTE_PROBE_INTERVAL")
fi
if [ "${SKIP_MISSING_BASELINE_COMPARISONS:-0}" = "1" ] || [ "${SKIP_MISSING_BASELINE_COMPARISONS:-0}" = "true" ] || [ "${SKIP_MISSING_BASELINE_COMPARISONS:-0}" = "TRUE" ]; then
  EXTRA_ARGS+=(--skip_missing_baseline_comparisons)
fi

IFS=',:' read -r -a DATASET_LIST <<< "$DATASETS_RAW"

for CURRENT_DATASET in "${DATASET_LIST[@]}"; do
  if [ -z "$CURRENT_DATASET" ]; then
    continue
  fi

  RESULT_PATH="${RESULT_DIR}/fixed_window_focus_${CURRENT_DATASET}_${OUTPUT_SLUG}${SHARD_SUFFIX}.json"
  echo "Running dataset=${CURRENT_DATASET} output=${RESULT_PATH}"

  python -u main_fixed_window_focused_compare.py \
    --dataset "$CURRENT_DATASET" \
    --ms_name "$MS_NAME_RESOLVED" \
    --mm_name "$MM_NAME_RESOLVED" \
    --ml_name "$ML_NAME_RESOLVED" \
    --small_device "$SMALL_DEVICE" \
    --middle_device "$MIDDLE_DEVICE" \
    --large_device "$LARGE_DEVICE" \
    --n_samples "$N_SAMPLES" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$SHARD_INDEX" \
    --config_specs "$CONFIG_SPECS" \
    --dynamic_middle_window_min "$DYNAMIC_MIDDLE_WINDOW_MIN" \
    --dynamic_middle_window_max "$DYNAMIC_MIDDLE_WINDOW_MAX" \
    --dynamic_acceptance_low "$DYNAMIC_ACCEPTANCE_LOW" \
    --dynamic_acceptance_high "$DYNAMIC_ACCEPTANCE_HIGH" \
    --dynamic_window_step "$DYNAMIC_WINDOW_STEP" \
    --dynamic_utility_margin "$DYNAMIC_UTILITY_MARGIN" \
    --selective_route_warmup_blocks "$SELECTIVE_ROUTE_WARMUP_BLOCKS" \
    --selective_route_history_window "$SELECTIVE_ROUTE_HISTORY_WINDOW" \
    --selective_route_utility_margin "$SELECTIVE_ROUTE_UTILITY_MARGIN" \
    --selective_route_direct_acceptance_low "$SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_LOW" \
    --selective_route_direct_acceptance_high "$SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_HIGH" \
    --max_length "$MAX_LENGTH" \
    --trace_samples "$TRACE_SAMPLES" \
    --output "$RESULT_PATH" \
    "${EXTRA_ARGS[@]}"
done
