#!/bin/bash
#SBATCH --job-name=fixed_window_repair
#SBATCH --output=/home/%u/logs/sbatch/fixed_window_repair_%A.out
#SBATCH --error=/home/%u/logs/sbatch/fixed_window_repair_%A.err
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
RERUN_TIME_LIMIT="${RERUN_TIME_LIMIT:-02:00:00}"
MS_NAME="${MS_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MM_NAME="${MM_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}"
SMALL_WINDOW="${SMALL_WINDOW:-3}"
MIDDLE_WINDOW="${MIDDLE_WINDOW:-9}"
HIERARCHICAL_VARIANT="${HIERARCHICAL_VARIANT:-window_fill}"
PROXY_THRESHOLD="${PROXY_THRESHOLD:-}"
MAVG_WINDOW="${MAVG_WINDOW:-5}"
MAX_LENGTH="${MAX_LENGTH:-200}"
# The current mlsys-fw environment uses a PyTorch build without sm_120 support,
# so reruns should avoid Blackwell RTX PRO 6000 nodes by default.
RERUN_EXCLUDE="${RERUN_EXCLUDE:-babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28}"

missing=()
for ((idx=0; idx<NUM_SHARDS; idx++)); do
  shard_path="$SHARD_DIR/fixed_window_compare_${DATASET}_${OUTPUT_SLUG}_shard${idx}of${NUM_SHARDS}.json"
  if [ ! -f "$shard_path" ]; then
    missing+=("$idx")
  fi
done

echo "Shard dir: $SHARD_DIR"
echo "Missing shards: ${missing[*]:-none}"

rerun_deps=()
for idx in "${missing[@]}"; do
  rerun_args=()
  if [ -n "$RERUN_EXCLUDE" ]; then
    rerun_args+=(--exclude="$RERUN_EXCLUDE")
  fi
  job_id="$(
    sbatch --parsable \
      -p "$RERUN_PARTITION" \
      --gres="$RERUN_GPU_SPEC" \
      --cpus-per-task="$RERUN_CPUS_PER_TASK" \
      --mem="$RERUN_MEMORY" \
      --time="$RERUN_TIME_LIMIT" \
      "${rerun_args[@]}" \
      --export=ALL,N_SAMPLES="${N_SAMPLES:-100}",NUM_SHARDS="$NUM_SHARDS",SHARD_INDEX="$idx",TRACE_SAMPLES="${TRACE_SAMPLES:-2}",OUTPUT_SLUG="$OUTPUT_SLUG",RESULT_DIR="$SHARD_DIR",SHARD_OFFSET=0,MS_NAME="$MS_NAME",MM_NAME="$MM_NAME",ML_NAME="$ML_NAME",SMALL_WINDOW="$SMALL_WINDOW",MIDDLE_WINDOW="$MIDDLE_WINDOW",HIERARCHICAL_VARIANT="$HIERARCHICAL_VARIANT",PROXY_THRESHOLD="$PROXY_THRESHOLD",MAVG_WINDOW="$MAVG_WINDOW",MAX_LENGTH="$MAX_LENGTH" \
      scripts/run_fixed_window_babel.sbatch "$DATASET"
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
    scripts/merge_fixed_window_shards.sbatch "$DATASET" "$OUTPUT_SLUG" "$NUM_SHARDS"
)"

echo "Submitted merge job: $merge_job_id"
