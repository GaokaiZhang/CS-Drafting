#!/bin/bash
#SBATCH --job-name=acsd_repair
#SBATCH --output=/home/%u/logs/sbatch/acsd_repair_%A.out
#SBATCH --error=/home/%u/logs/sbatch/acsd_repair_%A.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

DATASET="${1:?dataset required}"
OUTPUT_SLUG="${2:?output slug required}"
NUM_SHARDS="${3:?num_shards required}"
RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
SHARD_DIR="${RESULT_DIR:-$RUN_ROOT/results/shards/${DATASET}_${OUTPUT_SLUG}}"
RERUN_PARTITION="${RERUN_PARTITION:-general}"
RERUN_GPU_SPEC="${RERUN_GPU_SPEC:-gpu:L40S:2}"
RERUN_CPUS_PER_TASK="${RERUN_CPUS_PER_TASK:-8}"
RERUN_MEMORY="${RERUN_MEMORY:-96G}"
RERUN_TIME_LIMIT="${RERUN_TIME_LIMIT:-08:00:00}"
MS_NAME="${MS_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MM_NAME="${MM_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}"
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
BRANCH_ROOT="${BRANCH_ROOT:-$RUN_ROOT}"

missing=()
for ((idx=0; idx<NUM_SHARDS; idx++)); do
  shard_path="$SHARD_DIR/acsd_compare_${DATASET}_${OUTPUT_SLUG}_shard${idx}of${NUM_SHARDS}.json"
  if [ ! -f "$shard_path" ]; then
    missing+=("$idx")
  fi
done

echo "Shard dir: $SHARD_DIR"
echo "Missing shards: ${missing[*]:-none}"

rerun_deps=()
for idx in "${missing[@]}"; do
  job_id="$(
    sbatch --parsable \
      -p "$RERUN_PARTITION" \
      --gres="$RERUN_GPU_SPEC" \
      --cpus-per-task="$RERUN_CPUS_PER_TASK" \
      --mem="$RERUN_MEMORY" \
      --time="$RERUN_TIME_LIMIT" \
      --export=ALL,N_SAMPLES="${N_SAMPLES:-100}",NUM_SHARDS="$NUM_SHARDS",SHARD_INDEX="$idx",TRACE_SAMPLES="${TRACE_SAMPLES:-2}",OUTPUT_SLUG="$OUTPUT_SLUG",RESULT_DIR="$SHARD_DIR",SHARD_OFFSET=0,BRANCH_ROOT="$BRANCH_ROOT",MODE="$MODE",MS_NAME="$MS_NAME",MM_NAME="$MM_NAME",ML_NAME="$ML_NAME",K_S="$K_S",K_M="$K_M",LENIENCY="$LENIENCY",TAU="$TAU",WINDOW_SIZE="$WINDOW_SIZE",MAX_LENGTH="$MAX_LENGTH",MIDDLE_REFILL="$MIDDLE_REFILL",SHADOW_MIDDLE="$SHADOW_MIDDLE",DRAFT_WINDOW_POLICY="$DRAFT_WINDOW_POLICY",DYNAMIC_K_S_MIN="$DYNAMIC_K_S_MIN",DYNAMIC_K_S_MAX="$DYNAMIC_K_S_MAX",DYNAMIC_K_M_MIN="$DYNAMIC_K_M_MIN",DYNAMIC_K_M_MAX="$DYNAMIC_K_M_MAX",DYNAMIC_ACCEPTANCE_LOW="$DYNAMIC_ACCEPTANCE_LOW",DYNAMIC_ACCEPTANCE_HIGH="$DYNAMIC_ACCEPTANCE_HIGH",DYNAMIC_WINDOW_STEP="$DYNAMIC_WINDOW_STEP" \
      scripts/run_acsd_compare_babel.sbatch "$DATASET"
  )"
  rerun_deps+=("$job_id")
  echo "Submitted rerun shard job $job_id for shard $idx"
done

merge_dependency=()
if [ "${#rerun_deps[@]}" -gt 0 ]; then
  dep_spec="$(IFS=:; echo "${rerun_deps[*]}")"
  merge_dependency+=(--dependency="afterok:${dep_spec}")
fi

merge_job_id="$(
  sbatch --parsable \
    "${merge_dependency[@]}" \
    --export=ALL,RESULT_DIR="$SHARD_DIR" \
    scripts/merge_acsd_shards.sbatch "$DATASET" "$OUTPUT_SLUG" "$NUM_SHARDS"
)"

echo "Submitted merge job: $merge_job_id"
