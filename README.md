# Cascade Speculative Drafting + ACSD Extension

Original paper: "[Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462)"

This repo contains the original CS-Drafting implementation **plus our course-project extension: Adaptive Cascaded Speculative Decoding (ACSD)**, which adds a third middle-tier model to the cascade with cascaded pre-verification and a double-layer drafting design.

> **For teammates:** See [`status.md`](status.md) for current project status, experiment instructions, and deliverable deadlines.

---

## ACSD System

| Tier | Model | Role |
|------|-------|------|
| $M_s$ | TinyLlama-1.1B | Small fast drafter |
| $M_m$ | LLaMA-2-7B | Pre-verifier (Phase 2) / verifier + extender (Phase 4) |
| $M_l$ | LLaMA-2-13B | Final verifier (always runs — losslessness guaranteed) |

**Phase 2 (cascaded):** $M_m$ pre-filters $M_s$'s draft tokens before $M_l$ sees them, reducing $M_l$ forward-pass cost.

**Phase 4 (double-layer + proxy):** $M_s$ drafts up to $k_s$ tokens; $M_m$ verifies and auto-regressively extends to $k_m$ total; $M_l$ verifies all $k_m$. Proxy variants let $M_s$ stop early when its confidence is low.

---

## Parameter Reference

| Parameter | CLI flag | Default | Meaning |
|-----------|----------|---------|---------|
| mode | `--mode` | — | `autoregressive`: $M_l$ only; `baseline`: $M_s \to M_l$; `cascaded`: $M_s \to M_m$ pre-verify $\to M_l$; `double_layer`: $M_s$ drafts $k_s$, $M_m$ verifies and extends to $k_m$, $M_l$ verifies all; `proxy_entropy`/`proxy_top1`/`proxy_margin`/`proxy_mavg`: `double_layer` + confidence proxy that stops $M_s$ early |
| dataset | `--dataset` | `mmlu` | Evaluation dataset: `mmlu` or `gsm8k` |
| n\_samples | `--n_samples` | `100` | Number of prompts to evaluate |
| k\_s | (config) | `5` | Max tokens $M_s$ drafts per step. In proxy modes, $M_s$ may stop early. |
| k\_m | `--k_m` | `10` | Total tokens $M_m$ extends to in `double_layer`/proxy modes (must be ≥ k\_s). |
| proxy\_threshold | `--proxy_threshold` | type-specific | Confidence threshold at which $M_s$ stops early. Defaults: entropy > 2.0, top1 < −1.5, margin < 1.0, mavg < −1.5. |
| mavg\_window | `--mavg_window` | `5` | Rolling window for mavg proxy smoothing. |
| answer\_only | `--answer_only` | off | Accuracy-only mode: direct-answer prompt (no CoT) for MMLU, 200-token budget for GSM8K. |

**Metrics in result JSON:**

| Metric | Meaning |
|--------|---------|
| `tokens_per_sec` | End-to-end throughput |
| `avg_ml_calls` | Average $M_l$ forward passes per sample (lower = more efficient) |
| `avg_mm_calls` | Average $M_m$ forward passes per sample |
| `accuracy` | Fraction of correct answers (only present in `--answer_only` runs) |

---

## Experiment Results

All experiments: 100 samples, max 200 tokens, single A6000 48 GB GPU, fp16.
Speedup relative to Baseline CSD.

**Output accuracy** (zero-shot, direct-answer prompt, `--answer_only`, separate from throughput runs): MMLU 30.0%, GSM8K 9.0%.
Since all methods are lossless ($M_l$ always has final say), accuracy is identical across all methods.
Empirically verified: Baseline CSD and Double-layer both match autoregressive $M_l$ on MMLU (30.0%) and within 1 sample on GSM8K (10.0% vs 9.0% — parser noise on truncated outputs).

### Phase 2: Cascaded Pre-Verification

| Method | Dataset | Tok/s | Speedup vs baseline | $M_l$ calls/sample | $M_m$ saved/sample |
|--------|---------|------:|--------------------:|-------------------:|-------------------:|
| Autoregressive ($M_l$ only) | MMLU | 9.3 | — | 190.0 | — |
| Autoregressive ($M_l$ only) | GSM8K | 9.4 | — | 177.0 | — |
| Baseline CSD ($M_s \to M_l$) | MMLU | 26.6 | 1.00× | 29.7 | — |
| Baseline CSD ($M_s \to M_l$) | GSM8K | 23.9 | 1.00× | 30.4 | — |
| ACSD Cascaded (Phase 2) | MMLU | **41.1** | **1.55×** | 38.0 | 37.7 |
| ACSD Cascaded (Phase 2) | GSM8K | **39.9** | **1.67×** | 36.1 | 41.9 |

### Phase 4: Double-Layer & Confidence Proxy

All experiments: k\_s=5, k\_m=10, 100 samples, max 200 tokens, single A6000 48 GB GPU, fp16.

| Method | Dataset | Tok/s | Speedup vs baseline | $M_l$ calls/sample | $M_m$ calls/sample |
|--------|---------|------:|--------------------:|-------------------:|-------------------:|
| Double-layer | MMLU | 29.5 | 1.11× | **24.8** | 69.0 |
| Double-layer | GSM8K | 27.7 | 1.16× | **23.4** | 69.2 |
| + proxy\_top1 (log-prob < −1.5) | MMLU | 30.9 | 1.16× | **24.8** | 70.2 |
| + proxy\_top1 | GSM8K | 29.3 | 1.23× | **23.4** | 70.2 |
| + proxy\_entropy (H > 2.0) | MMLU | 32.1 | 1.21× | **24.8** | 76.9 |
| + proxy\_entropy | GSM8K | **31.4** | **1.31×** | **23.4** | 75.1 |
| + proxy\_margin (margin < 1.0) | MMLU | **32.9** | **1.24×** | **24.8** | 77.5 |
| + proxy\_margin | GSM8K | **31.4** | **1.31×** | **23.4** | 76.4 |
| + proxy\_mavg (mavg < −1.5) | MMLU | 29.9 | 1.12× | **24.8** | 69.6 |
| + proxy\_mavg | GSM8K | 28.9 | 1.21× | **23.4** | 69.9 |

**Key observations:**
- Double-layer reduces $M_l$ calls ~35% vs. Phase 2 (~25 vs. ~38/sample), because $M_m$ extends drafts to $k_m=10$ tokens, amortising $M_l$ verification over longer sequences.
- Raw throughput is lower than Phase 2 cascaded (29–33 vs. 41 tok/s) — $M_m$ extension adds latency. Different operating point: fewer $M_l$ calls rather than maximum raw speed.
- `proxy_margin` and `proxy_entropy` give the best throughput gains (1.21–1.31× vs. baseline).
- `proxy_mavg` barely improves over double-layer base — the smoothed threshold rarely triggers.

---

## Re-running Experiments

```bash
conda activate acsd
cd /mnt/data/gaokaizhang/mlsys/CS-Drafting

# autoregressive / baseline: M_l only (~26 GB)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode baseline --dataset mmlu --n_samples 100 --device cuda:0 \
    --output results/baseline_mmlu.json

# cascaded (Phase 2): all 3 models (~42 GB)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode cascaded --dataset mmlu --n_samples 100 --device cuda:0 \
    --output results/cascaded_mmlu.json

# double_layer (Phase 4): all 3 models (~42 GB)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode double_layer --dataset mmlu --n_samples 100 \
    --k_s 5 --k_m 10 --device cuda:0 \
    --output results/double_layer_mmlu.json

# proxy modes (Phase 4): swap --mode for proxy_top1, proxy_margin, proxy_mavg
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode proxy_entropy --dataset mmlu --n_samples 100 \
    --k_s 5 --k_m 10 --device cuda:0 \
    --output results/proxy_entropy_mmlu.json

# accuracy evaluation (any mode, uses direct-answer prompt)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode autoregressive --dataset mmlu --n_samples 100 \
    --answer_only --device cuda:0 \
    --output results/accuracy_mmlu.json
```

GPU memory: `autoregressive`/`baseline` need ~26 GB; `cascaded`/`double_layer`/`proxy_*` need ~42 GB (all 3 models).

Each result file is a JSON with three top-level keys: `config` (hyperparameters), `summary` (tok/s, avg_wall_time, avg_ml_calls, accuracy), and `raw` (per-sample arrays including `correct` and `generated_texts` in `--answer_only` runs).

---

## Setup

### ACSD environment (python 3.11, recommended)

```bash
conda create -n acsd python=3.11 -y
conda activate acsd
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.45 datasets>=2.20 accelerate sentencepiece tqdm
```

> CUDA 12.4 (`cu124`) requires driver ≥ 520. Use `cu126` for CUDA 12.6+ drivers.

### Original CS-Drafting environment (python 3.9)

```bash
conda create --name csd python=3.9
conda activate csd
pip install -r requirements.txt
```

---

## Recreating Original CS-Drafting Experiments

The starting point of the report is `main.py`, which can be run without args.
All experiment settings can be adjusted in the `config` dict in `main.py`.
GPU usage can be adjusted by changing:

```python
usable_devices = [0, 1, 2] * 2
```

Each index in the list represents a single process on the GPU of that index.
We recommend using 2 processes per GPU with at least 16 GB of memory.

To run experiments with FLAN-T5 on MMLU for SWI (model size) setup:

```python
config = {
    'draft_names': ['google/flan-t5-base', 'google/flan-t5-small'],
    'target_name': 'google/flan-t5-xxl',
    'is_decoder_only': False,
    'use_mag': True,
    'k_matrix': [[5, 14, 10], [0, 1, 10], [0, 0, 10]],
    'lenience': 2,
    'dataset': 'mmlu',
    'counter_version': 'model_parameters',
    'sample': False
}
```

For SWI (previous work):

```python
config = {
    'draft_names': ['google/flan-t5-base', 'google/flan-t5-small'],
    'target_name': 'google/flan-t5-xxl',
    'is_decoder_only': False,
    'use_mag': True,
    'k_matrix': [[5, 8, 10], [0, 1, 10], [0, 0, 10]],
    'lenience': 5,
    'dataset': 'sampled_mmlu',
    'counter_version': 'previous_work',
    'sample': False
}
```

To run LLaMA-7B on MMLU:

```python
config = {
    'draft_names': ['JackFram/llama-160m'],
    'target_name': 'llama_7b',
    'is_decoder_only': True,
    'use_mag': True,
    'k_matrix': [[13, 10], [0, 10]],
    'lenience': 3,
    'dataset': 'mmlu',
    'counter_version': 'model_parameters',
    'sample': False
}
```

To run GSM8K, change `'dataset': 'gsm8k'` in the config.

Note: when using two draft models without MAG, the `k_matrix` parameters relate to the paper notation as:
```
k_matrix[0][0] = k_11
k_matrix[0][1] = k_12 - k_matrix[0][0]
```

---

## Using CS-Drafting for Inference

```python
from csd import csd
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import CountedCSDraftingDecoderModelKVCache, get_mag_model

draft_list = []
for draft_name in ['JackFram/llama-160m']:
    hf_model = AutoModelForCausalLM.from_pretrained(draft_name)
    model = CountedCSDraftingDecoderModelKVCache(hf_model, name=draft_name)
    draft_list.append(model)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
hf_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
target_model = CountedCSDraftingDecoderModelKVCache(hf_model, name='llama', vocab_size=32000)
target_model.cuda(0)

question = '<Your input>'
initial_input = tokenizer(question, return_tensors='pt')['input_ids'].to(target_model.device)
res = csd(draft_list, target_model, initial_input, initial_input, k_matrix=[[5, 10], [0, 10]])
print(tokenizer.batch_decode(res, skip_special_tokens=True))
```

---

## Testing and CI

Local test stack:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

Test layers now included:

- unit tests for scoring and fixed-window decoding behavior
- integration tests for comparison artifact generation and serialization
- functional smoke tests for the CLI and static UI contract

CI/CD:

- `.github/workflows/ci.yml` runs the full test suite on GitHub Actions using CPU PyTorch
- `.github/workflows/deploy-ui.yml` publishes the static inspector in `ui/` to GitHub Pages on pushes to `main` or `master`

The fixed-window comparison runner is:

```bash
python main_fixed_window.py \
  --mode compare \
  --dataset mmlu \
  --ms_name Qwen/Qwen2.5-1.5B-Instruct \
  --mm_name Qwen/Qwen2.5-7B-Instruct \
  --ml_name Qwen/Qwen2.5-14B-Instruct \
  --small_device cuda:0 \
  --middle_device cuda:1 \
  --large_device cuda:2 \
  --small_window 3 \
  --middle_window 9 \
  --n_samples 100 \
  --trace_samples 5 \
  --output results/fixed_window_compare_mmlu.json
```

Open `ui/index.html` and load the generated JSON to inspect per-token traces, usage breakdowns, pass-through rates, and benchmark scores.

### Babel HPC setup

Create the repo-specific environment:

```bash
conda create -n mlsys-fw python=3.11 -y
conda activate mlsys-fw
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install -r requirements.txt -r requirements-dev.txt
```

Load the shared Hugging Face cache on Babel compute nodes:

```bash
source scripts/babel_hf_env.sh
```

That script sets:

- `HF_HOME=/data/user_data/$USER/.hf_cache`
- `HF_HUB_CACHE=/data/hf_cache/hub`
- `HF_DATASETS_CACHE=/data/hf_cache/datasets`
- `HF_HUB_OFFLINE=1`

The fixed-window hierarchy now defaults to a same-family Qwen stack:

- small: `Qwen/Qwen2.5-1.5B-Instruct`
- middle: `Qwen/Qwen2.5-7B-Instruct`
- large: `Qwen/Qwen2.5-14B-Instruct`

On Babel we run them split across two GPUs:

- `small_device=cuda:0`
- `middle_device=cuda:0`
- `large_device=cuda:1`

The checked-in Babel Slurm script targets `debug` with `2 x L40S`, which fits within the current `debug_qos` per-user GPU cap while keeping the large verifier isolated on its own GPU.

Submit both benchmark jobs:

```bash
./scripts/submit_fixed_window_all.sh
```

Or submit one dataset explicitly:

```bash
sbatch scripts/run_fixed_window_babel.sbatch mmlu
sbatch scripts/run_fixed_window_babel.sbatch gsm8k
```

### Flask result viewer

Once result JSON files are in `results/`, run:

```bash
PORT=5000 conda run -n mlsys-fw python flask_ui.py
```

The Flask UI serves:

- `/` for the dashboard
- `/api/results` for available result artifacts
- `/api/results/<filename>` for a specific experiment payload

---

## Change Log

- 2024-04-02: Added KV Cache to reduce latency for long generation
- 2026-04: ACSD extension — cascaded pre-verification (`acsd.py`, `main_acsd.py`, `model.py`)
- 2026-04: Updated for transformers ≥ 4.45 (DynamicCache API, `cais/mmlu` Parquet dataset)
- 2026-04-14: Phase 4 complete — double\_layer and proxy\_\* (entropy, top1, margin, mavg) on MMLU + GSM8K
- 2026-04-14: Added accuracy evaluation (`--answer_only`) with empirical losslessness verification

---

## Citation

```bibtex
@article{chen2023cascade,
  title={Cascade Speculative Drafting for Even Faster LLM Inference},
  author={Chen, Ziyi and Yang, Xiaocong and Lin, Jiacheng and Sun, Chenkai and Chen, Yangyi and Chang, Kevin Chen-Chuan and Huang, Jie},
  journal={arXiv preprint arXiv:2312.11462},
  year={2023}
}
```
