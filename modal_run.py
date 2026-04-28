"""
Modal deployment for double-layer speculative decoding experiments.

Setup (one-time)
----------------
1. Install modal:          pip install modal
2. Authenticate:          modal setup
3. Create HF token secret (needed for Llama-2):
      modal secret create huggingface-secret HF_TOKEN=<your-token>
4. Download models to Modal volume (one-time, ~42 GB):
      modal run modal_run.py::download_models
5. Run all experiments:
      modal run modal_run.py            (spawns all jobs in parallel, detached)

Fetching results
----------------
      modal volume ls   acsd-results
      modal volume get  acsd-results /  ./results/   # sync everything locally

Monitoring
----------
      modal app list                  # list running apps
      modal app logs <app-id>         # stream logs of a specific run
"""

import modal

# ── infrastructure ─────────────────────────────────────────────────────────────

MODEL_VOLUME   = modal.Volume.from_name("acsd-models",  create_if_missing=True)
RESULTS_VOLUME = modal.Volume.from_name("acsd-results", create_if_missing=True)

# Mount the local CS-Drafting source tree into the container
CODE_MOUNT = modal.Mount.from_local_dir(
    local_path=".",          # CS-Drafting directory
    remote_path="/acsd",
    condition=lambda p: not any(
        p.endswith(ext) for ext in ('.json', '.log', '.pyc', '__pycache__')
    ),
)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "transformers>=4.45",
        "datasets>=2.20",
        "accelerate",
        "sentencepiece",
        "tqdm",
        "huggingface-hub",
    )
)

app = modal.App("acsd-double-layer", image=IMAGE)

# HuggingFace model IDs used in containers (ungated / NousResearch wrappers)
_MS = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_MM = "NousResearch/Llama-2-7b-hf"
_ML = "NousResearch/Llama-2-13b-hf"

# Local paths inside the container (written by download_models, read by run_experiment)
_MODEL_ROOT = "/models"
_MS_PATH    = f"{_MODEL_ROOT}/TinyLlama-1.1B"
_MM_PATH    = f"{_MODEL_ROOT}/Llama-2-7b-hf"
_ML_PATH    = f"{_MODEL_ROOT}/Llama-2-13b-hf"


# ── one-time model download ────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    volumes={_MODEL_ROOT: MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    cpu=4,
)
def download_models():
    """Download all three models to the Modal volume. Run once."""
    from huggingface_hub import snapshot_download
    import os

    hf_token = os.environ.get("HF_TOKEN")

    for hf_id, local_dir in [
        (_MS, _MS_PATH),
        (_MM, _MM_PATH),
        (_ML, _ML_PATH),
    ]:
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            print(f"[skip] {hf_id} already at {local_dir}")
            continue
        print(f"Downloading {hf_id} → {local_dir}")
        snapshot_download(
            repo_id=hf_id,
            local_dir=local_dir,
            token=hf_token,
            ignore_patterns=["*.bin"],      # prefer safetensors
        )

    MODEL_VOLUME.commit()
    print("All models downloaded and committed to volume.")


# ── single experiment function ─────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100-80GB",           # ~42 GB for all three models in fp16
    volumes={
        _MODEL_ROOT:  MODEL_VOLUME,
        "/results":   RESULTS_VOLUME,
    },
    mounts=[CODE_MOUNT],
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,              # 2 h upper bound; typically ~15 min per experiment
)
def run_experiment(mode: str, dataset: str,
                   k_s: int = 5, k_m: int = 10,
                   proxy_threshold: float | None = None,
                   mavg_window: int = 5,
                   n_samples: int = 100):
    """Run one (mode, dataset) combination and write JSON to the results volume."""
    import subprocess, sys, os

    tag = f"{mode}_{dataset}_ks{k_s}_km{k_m}"
    if proxy_threshold is not None:
        tag += f"_thr{proxy_threshold}"
    output_path = f"/results/{tag}.json"

    cmd = [
        sys.executable, "/acsd/main_acsd.py",
        "--mode",      mode,
        "--dataset",   dataset,
        "--k_s",       str(k_s),
        "--k_m",       str(k_m),
        "--n_samples", str(n_samples),
        "--device",    "cuda:0",
        "--ms_name",   _MS_PATH,
        "--mm_name",   _MM_PATH,
        "--ml_name",   _ML_PATH,
        "--output",    output_path,
    ]
    if proxy_threshold is not None:
        cmd += ["--proxy_threshold", str(proxy_threshold)]
    cmd += ["--mavg_window", str(mavg_window)]

    print(f"[{tag}] starting …")
    proc = subprocess.run(cmd, capture_output=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"[{tag}] FAILED (exit {proc.returncode})")

    RESULTS_VOLUME.commit()
    print(f"[{tag}] done → {output_path}")
    return tag


# ── local entrypoint — spawns all jobs ────────────────────────────────────────

@app.local_entrypoint()
def main():
    """
    Spawn all experiments in parallel (detached from this terminal).
    Results land in the 'acsd-results' Modal volume.
    Retrieve with:  modal volume get acsd-results / ./results/
    """
    experiments = [
        # base method
        dict(mode="double_layer",   dataset="mmlu",  k_s=5, k_m=10),
        dict(mode="double_layer",   dataset="gsm8k", k_s=5, k_m=10),
        # proxy — top-1 log-prob
        dict(mode="proxy_top1",     dataset="mmlu",  k_s=5, k_m=10),
        dict(mode="proxy_top1",     dataset="gsm8k", k_s=5, k_m=10),
        # proxy — entropy
        dict(mode="proxy_entropy",  dataset="mmlu",  k_s=5, k_m=10),
        dict(mode="proxy_entropy",  dataset="gsm8k", k_s=5, k_m=10),
        # proxy — top1-top2 margin
        dict(mode="proxy_margin",   dataset="mmlu",  k_s=5, k_m=10),
        dict(mode="proxy_margin",   dataset="gsm8k", k_s=5, k_m=10),
        # proxy — moving-average log-prob
        dict(mode="proxy_mavg",     dataset="mmlu",  k_s=5, k_m=10),
        dict(mode="proxy_mavg",     dataset="gsm8k", k_s=5, k_m=10),
    ]

    handles = []
    for kwargs in experiments:
        h = run_experiment.spawn(**kwargs)
        print(f"Spawned: {kwargs['mode']} / {kwargs['dataset']}")
        handles.append((kwargs, h))

    print(f"\n{len(handles)} jobs running on Modal (A100-80GB each).")
    print("Monitor:  modal app list")
    print("Fetch:    modal volume get acsd-results / ./results/")
