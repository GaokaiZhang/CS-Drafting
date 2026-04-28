#!/bin/bash
#SBATCH --job-name=fixed_window_focus_repair
#SBATCH --output=/home/%u/logs/sbatch/fixed_window_focus_repair_%A.out
#SBATCH --error=/home/%u/logs/sbatch/fixed_window_focus_repair_%A.err
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
RERUN_GPU_SPEC="${RERUN_GPU_SPEC:-gpu:A6000:1}"
RERUN_CPUS_PER_TASK="${RERUN_CPUS_PER_TASK:-12}"
RERUN_MEMORY="${RERUN_MEMORY:-96G}"
RERUN_TIME_LIMIT="${RERUN_TIME_LIMIT:-06:00:00}"

is_complete_shard() {
  local shard_path="$1"
  python - "$shard_path" "${CONFIG_SPECS:-}" <<'PY'
import json
import sys

path = sys.argv[1]
config_specs = sys.argv[2]

try:
    with open(path) as handle:
        payload = json.load(handle)
except Exception as exc:
    print(f"{path}: could not load JSON: {exc}", file=sys.stderr)
    sys.exit(1)

runs = payload.get("runs")
if not isinstance(runs, dict) or not runs:
    print(f"{path}: missing runs block", file=sys.stderr)
    sys.exit(1)

if payload.get("specs") is not None:
    expected = [spec["label"] for spec in payload["specs"]]
    for spec in payload["specs"]:
        baseline_label = spec.get("baseline_label")
        if baseline_label and baseline_label not in expected:
            expected.append(baseline_label)
elif config_specs:
    expected = [entry.split("|", 1)[0] for entry in config_specs.split(";") if entry]
else:
    expected = list(runs)

missing = [label for label in expected if label not in runs]
if missing:
    print(f"{path}: missing completed runs {missing}", file=sys.stderr)
    sys.exit(1)

selected_n_samples = payload.get("selection", {}).get("selected_n_samples")
for label in expected:
    run_data = runs[label]
    for field in ("summary", "config", "sample_metrics", "aggregate_counters", "merge_stats"):
        if field not in run_data:
            print(f"{path}: run {label} missing {field}", file=sys.stderr)
            sys.exit(1)
    sample_metrics = run_data.get("sample_metrics")
    if selected_n_samples is not None and sample_metrics is not None:
        if len(sample_metrics) != int(selected_n_samples):
            print(
                f"{path}: run {label} has {len(sample_metrics)} sample metrics "
                f"for {selected_n_samples} selected samples",
                file=sys.stderr,
            )
            sys.exit(1)
PY
}

missing=()
for ((idx=0; idx<NUM_SHARDS; idx++)); do
  shard_path="$SHARD_DIR/fixed_window_focus_${DATASET}_${OUTPUT_SLUG}_shard${idx}of${NUM_SHARDS}.json"
  if [ ! -f "$shard_path" ] || ! is_complete_shard "$shard_path"; then
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
      --export=ALL,DATASET="$DATASET",N_SAMPLES="${N_SAMPLES:-0}",NUM_SHARDS="$NUM_SHARDS",SHARD_INDEX="$idx",TRACE_SAMPLES="${TRACE_SAMPLES:-0}",OUTPUT_SLUG="$OUTPUT_SLUG",RESULT_DIR="$SHARD_DIR",SHARD_OFFSET=0,MS_NAME="${MS_NAME:-Qwen/Qwen2.5-0.5B-Instruct}",MM_NAME="${MM_NAME:-Qwen/Qwen2.5-1.5B-Instruct}",ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}",CONFIG_SPECS="${CONFIG_SPECS:-}",DYNAMIC_SMALL_WINDOW_MIN="${DYNAMIC_SMALL_WINDOW_MIN:-}",DYNAMIC_SMALL_WINDOW_MAX="${DYNAMIC_SMALL_WINDOW_MAX:-}",DYNAMIC_MIDDLE_WINDOW_MIN="${DYNAMIC_MIDDLE_WINDOW_MIN:-3}",DYNAMIC_MIDDLE_WINDOW_MAX="${DYNAMIC_MIDDLE_WINDOW_MAX:-18}",DYNAMIC_ACCEPTANCE_LOW="${DYNAMIC_ACCEPTANCE_LOW:-0.6}",DYNAMIC_ACCEPTANCE_HIGH="${DYNAMIC_ACCEPTANCE_HIGH:-0.8}",DYNAMIC_WINDOW_STEP="${DYNAMIC_WINDOW_STEP:-1}",DYNAMIC_UTILITY_MARGIN="${DYNAMIC_UTILITY_MARGIN:-0.0}",SELECTIVE_ROUTE_WARMUP_BLOCKS="${SELECTIVE_ROUTE_WARMUP_BLOCKS:-1}",SELECTIVE_ROUTE_HISTORY_WINDOW="${SELECTIVE_ROUTE_HISTORY_WINDOW:-4}",SELECTIVE_ROUTE_UTILITY_MARGIN="${SELECTIVE_ROUTE_UTILITY_MARGIN:-0.0}",SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_LOW="${SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_LOW:-0.7}",SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_HIGH="${SELECTIVE_ROUTE_DIRECT_ACCEPTANCE_HIGH:-0.85}",SELECTIVE_ROUTE_MIDDLE_ACCEPTANCE_LOW="${SELECTIVE_ROUTE_MIDDLE_ACCEPTANCE_LOW:-}",SELECTIVE_ROUTE_PROBE_INTERVAL="${SELECTIVE_ROUTE_PROBE_INTERVAL:-}",ADAPT_SMALL_WINDOW="${ADAPT_SMALL_WINDOW:-0}",MAX_LENGTH="${MAX_LENGTH:-200}" \
      scripts/run_fixed_window_focused_babel.sbatch
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
    scripts/merge_fixed_window_focus_shards.sbatch "$DATASET" "$OUTPUT_SLUG" "$NUM_SHARDS"
)"

echo "Submitted merge job: $merge_job_id"
