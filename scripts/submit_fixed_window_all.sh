#!/bin/bash
set -euo pipefail

RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
PARTITION="${PARTITION:-general}"
GPU_SPEC="${GPU_SPEC:-gpu:L40S:3}"

cd "$RUN_ROOT"

sbatch -p "$PARTITION" --gres="$GPU_SPEC" scripts/run_fixed_window_babel.sbatch mmlu
sbatch -p "$PARTITION" --gres="$GPU_SPEC" scripts/run_fixed_window_babel.sbatch gsm8k
