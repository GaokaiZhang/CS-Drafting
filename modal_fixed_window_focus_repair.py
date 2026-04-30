"""Modal repair runner for a single fixed-window focused MMLU shard.

This script runs the focused comparison for the 128-way subshards that together
cover original shard 10/16. It returns compressed JSON
payloads to the local machine, writes them under results/shards, splices the
merged runs into the original shard-10 artifact, and runs the final 16-shard
merge locally.
"""

from __future__ import annotations

import base64
import gzip
import os
import subprocess
import sys
from pathlib import Path

import modal


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_SLUG = "qwen_0p5b_1p5b_14b_costaware_focus_full_20260426_l40s2_v1"
DATASET = "mmlu"
ORIGINAL_SHARD_DIR = RUN_ROOT / "results" / "shards" / f"{DATASET}_{OUTPUT_SLUG}"
SPLIT_SHARD_DIR = (
    RUN_ROOT
    / "results"
    / "shards"
    / f"{DATASET}_{OUTPUT_SLUG}_shard10_split128_modal"
)
TEMP_MERGED = (
    SPLIT_SHARD_DIR
    / f"fixed_window_focus_{DATASET}_{OUTPUT_SLUG}_shard10of16_merged_from128.json"
)
FINAL_OUTPUT = RUN_ROOT / "results" / f"fixed_window_focus_{DATASET}_{OUTPUT_SLUG}.json"

CONFIG_SPECS = ";".join(
    [
        "baseline_sw5|baseline|5|-|-|-|",
        "fixed_cascade_sw4_mw6|hierarchical|4|6|double_layer|fixed|baseline_sw5",
        "adaptive_sw4_mw6|hierarchical|4|6|double_layer|utility|baseline_sw5",
        "selective_sw4_mw6|hierarchical|4|6|selective_route|utility|baseline_sw5",
        "cost_aware_selective_sw4_mw6|hierarchical|4|6|cost_aware_selective_route|utility|baseline_sw5",
    ]
)
RUN_LABELS = [
    "baseline_sw5",
    "fixed_cascade_sw4_mw6",
    "adaptive_sw4_mw6",
    "selective_sw4_mw6",
    "cost_aware_selective_sw4_mw6",
]


def _ignore_local(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts & {".git", "__pycache__", ".pytest_cache", "results"}
        or path.suffix in {".pyc", ".log", ".out", ".err", ".pdf"}
    )


MODEL_VOLUME = modal.Volume.from_name("csdrafting-qwen-cache", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "transformers>=4.45,<5",
        "datasets>=2.20,<4",
        "accelerate>=0.26,<2",
        "fsspec<=2025.3.0,>=2023.1.0",
        "sentencepiece",
        "tqdm",
        "huggingface-hub[hf_transfer]",
    )
    .add_local_dir(RUN_ROOT, remote_path="/repo", copy=True, ignore=_ignore_local)
)

app = modal.App("csdrafting-fixed-window-repair", image=IMAGE)


@app.function(
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={"/models": MODEL_VOLUME},
)
def run_costaware_subshard(shard_index: int, num_shards: int = 64) -> dict[str, str | int]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/models/hf",
            "HF_HUB_CACHE": "/models/hf/hub",
            "HF_DATASETS_CACHE": "/models/hf/datasets",
            "HF_ENABLE_PARALLEL_LOADING": "true",
            "HF_PARALLEL_LOADING_WORKERS": "8",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    output = (
        f"/tmp/fixed_window_focus_{DATASET}_{OUTPUT_SLUG}_shard"
        f"{shard_index}of{num_shards}.json"
    )
    cmd = [
        sys.executable,
        "/repo/main_fixed_window_focused_compare.py",
        "--dataset",
        DATASET,
        "--ms_name",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--mm_name",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "--ml_name",
        "Qwen/Qwen2.5-14B-Instruct",
        "--small_device",
        "cuda:0",
        "--middle_device",
        "cuda:0",
        "--large_device",
        "cuda:0",
        "--n_samples",
        "0",
        "--num_shards",
        str(num_shards),
        "--shard_index",
        str(shard_index),
        "--config_specs",
        CONFIG_SPECS,
        "--dynamic_middle_window_min",
        "4",
        "--dynamic_middle_window_max",
        "8",
        "--dynamic_small_window_min",
        "3",
        "--dynamic_small_window_max",
        "5",
        "--dynamic_acceptance_low",
        "0.55",
        "--dynamic_acceptance_high",
        "0.75",
        "--dynamic_window_step",
        "1",
        "--dynamic_utility_margin",
        "0.02",
        "--selective_route_warmup_blocks",
        "1",
        "--selective_route_history_window",
        "2",
        "--selective_route_utility_margin",
        "0.02",
        "--selective_route_direct_acceptance_low",
        "0.55",
        "--selective_route_direct_acceptance_high",
        "0.7",
        "--selective_route_middle_acceptance_low",
        "0.55",
        "--selective_route_probe_interval",
        "0",
        "--adapt_small_window",
        "--trace_samples",
        "2",
        "--max_length",
        "200",
        "--skip_missing_baseline_comparisons",
        "--output",
        output,
    ]
    subprocess.run(cmd, cwd="/repo", env=env, check=True)
    MODEL_VOLUME.commit()

    raw = Path(output).read_bytes()
    return {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "payload": base64.b64encode(gzip.compress(raw, compresslevel=6)).decode("ascii"),
    }


def _run_local(command: str) -> None:
    subprocess.run(
        [
            "bash",
            "-lc",
            f'eval "$(conda shell.bash hook)" && conda activate mlsys-fw && {command}',
        ],
        cwd=RUN_ROOT,
        check=True,
    )


@app.local_entrypoint()
def repair(shards: str = "10,26,42,58,74,90,106,122", num_shards: int = 128) -> None:
    shard_indices = [int(value) for value in shards.replace(":", ",").split(",") if value]
    SPLIT_SHARD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running Modal cost-aware subshards: {shard_indices}")
    calls = [run_costaware_subshard.spawn(index, num_shards) for index in shard_indices]
    for call in calls:
        result = call.get()
        shard_index = int(result["shard_index"])
        result_num_shards = int(result["num_shards"])
        out_path = (
            SPLIT_SHARD_DIR
            / f"fixed_window_focus_{DATASET}_{OUTPUT_SLUG}_shard{shard_index}of{result_num_shards}.json"
        )
        out_path.write_bytes(gzip.decompress(base64.b64decode(str(result["payload"]))))
        print(f"Wrote {out_path}")

    subshard_inputs = " ".join(
        str(
            SPLIT_SHARD_DIR
            / f"fixed_window_focus_{DATASET}_{OUTPUT_SLUG}_shard{index}of{num_shards}.json"
        )
        for index in shard_indices
    )
    _run_local(
        "python merge_fixed_window_shards.py "
        "--skip_missing_baseline_comparisons "
        f"--output {TEMP_MERGED} "
        f"--inputs {subshard_inputs}"
    )
    _run_local(
        "python scripts/splice_focused_run.py "
        f"--source {TEMP_MERGED} "
        f"--target {ORIGINAL_SHARD_DIR / f'fixed_window_focus_{DATASET}_{OUTPUT_SLUG}_shard10of16.json'} "
        + " ".join(f"--label {label}" for label in RUN_LABELS)
    )
    _run_local(
        "python merge_fixed_window_shards.py "
        f"--output {FINAL_OUTPUT} "
        f"--inputs {ORIGINAL_SHARD_DIR}/fixed_window_focus_{DATASET}_{OUTPUT_SLUG}_shard*of16.json"
    )
    print(f"Final merged result: {FINAL_OUTPUT}")
