#!/bin/bash
#SBATCH --job-name=hf_prefetch
#SBATCH --output=/home/%u/logs/sbatch/hf_prefetch_%A.out
#SBATCH --error=/home/%u/logs/sbatch/hf_prefetch_%A.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"
MS_NAME="${MS_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MM_NAME="${MM_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ML_NAME="${ML_NAME:-Qwen/Qwen2.5-14B-Instruct}"

mkdir -p "/home/$USER/logs/sbatch"

source "$RUN_ROOT/scripts/babel_hf_env.sh"
export HF_HUB_OFFLINE=0

eval "$(conda shell.bash hook)"
conda activate mlsys-fw

cd "$RUN_ROOT"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

local_root = os.environ["HF_LOCAL_MODEL_ROOT"]
model_ids = []
for key in ("MS_NAME", "MM_NAME", "ML_NAME"):
    value = os.environ.get(key)
    if value and value not in model_ids:
        model_ids.append(value)

print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
print(f"HF_LOCAL_MODEL_ROOT={local_root}")

for repo_id in model_ids:
    if "/" not in repo_id:
        raise ValueError(f"Expected repo id like namespace/repo, got: {repo_id}")
    namespace, repo = repo_id.split("/", 1)
    local_dir = os.path.join(local_root, namespace, repo)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Prefetching {repo_id} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.onnx", "*.gguf", "*.h5", "*.msgpack"],
    )
    print(f"Completed {repo_id}", flush=True)
PY
