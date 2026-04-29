#!/bin/bash
#SBATCH --job-name=fixed_window_merge
#SBATCH --output=/home/%u/logs/sbatch/fixed_window_merge_%A.out
#SBATCH --error=/home/%u/logs/sbatch/fixed_window_merge_%A.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

DATASET="${1:?dataset required}"
OUTPUT_SLUG="${2:?output slug required}"
NUM_SHARDS="${3:?num_shards required}"
RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
SHARD_DIR="$RUN_ROOT/results/shards/${DATASET}_${OUTPUT_SLUG}"
FINAL_OUTPUT="$RUN_ROOT/results/fixed_window_compare_${DATASET}_${OUTPUT_SLUG}.json"

eval "$(conda shell.bash hook)"
conda activate mlsys-fw

cd "$RUN_ROOT"

python merge_fixed_window_shards.py \
  --output "$FINAL_OUTPUT" \
  --inputs "$SHARD_DIR"/fixed_window_compare_"${DATASET}"_"${OUTPUT_SLUG}"_shard*of"${NUM_SHARDS}".json

echo "Merged shard outputs into $FINAL_OUTPUT"
