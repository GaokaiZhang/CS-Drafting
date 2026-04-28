# Cascade Speculative Drafting + ACSD Extension

Original paper: "[Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462)"

This repo contains the original CS-Drafting implementation **plus our course-project extension: Adaptive Cascaded Speculative Decoding (ACSD)**, which adds a third middle-tier model to the cascade with cascaded pre-verification and adaptive role-switching.

> **For teammates:** See [`status.md`](status.md) for current project status, experiment instructions, and deliverable deadlines.

---

## ACSD System

| Tier | Model | Role |
|------|-------|------|
| $M_s$ | TinyLlama-1.1B | Small fast drafter |
| $M_m$ | LLaMA-2-7B | Pre-verifier **or** promoted drafter |
| $M_l$ | LLaMA-2-13B | Final verifier (always runs — losslessness guaranteed) |

**Phase 2 (cascaded):** $M_m$ pre-filters $M_s$'s draft tokens before $M_l$ sees them, reducing $M_l$ forward-pass cost.

**Phase 3 (adaptive):** When $M_s$'s rolling acceptance rate $\alpha$ drops below threshold $\tau$, $M_m$ is promoted to drafter, bypassing $M_s$ entirely. $M_l$ always performs final verification.

---

## Parameter Reference

| Parameter | CLI flag | Default | Meaning |
|-----------|----------|---------|---------|
| mode | `--mode` | `adaptive` | `autoregressive`: $M_l$ only (no drafting); `baseline`: $M_s \to M_l$; `cascaded`: $M_s \to M_m \to M_l$ pre-verify; `adaptive`: cascaded + dynamic drafter switching |
| dataset | `--dataset` | `mmlu` | Evaluation dataset: `mmlu` or `gsm8k` |
| n\_samples | `--n_samples` | `100` | Number of prompts to evaluate |
| τ (tau) | `--tau` | `0.4` | Switching threshold. $M_m$ becomes drafter when $M_s$ rolling acceptance $\bar{\alpha} < \tau$. Higher $\tau$ = more aggressive switching. Only used in `adaptive` mode. |
| W (window\_size) | `--window_size` | `20` | Rolling window length in drafting steps for computing $\bar{\alpha}$. Each step corresponds to one $M_s$ proposal of $k_s$ tokens. Larger $W$ = smoother, slower-reacting signal. |
| k\_s | (config) | `5` | Tokens $M_s$ drafts per step (fixed). |
| k\_m | `--k_m` | `4` | Tokens $M_m$ drafts per step when promoted to drafter. Only active in `adaptive` mode when $\bar{\alpha} < \tau$. |

**Metrics in result JSON:**

| Metric | Meaning |
|--------|---------|
| `tokens_per_sec` | End-to-end throughput |
| `avg_ml_calls` | Average $M_l$ forward passes per sample (lower = more efficient) |
| `mm_saved_positions` | Token positions $M_m$ filtered before $M_l$ per sample (higher = more $M_l$ savings) |
| `switch_counts` | Transitions across $\tau$ boundary in the *final* $\alpha$ window at end of sample |

---

## Experiment Results

All experiments: 100 samples, max 200 tokens, single A6000 48 GB GPU, fp16.
Speedup relative to Baseline CSD.

### Main Results

| Method | Dataset | Tok/s | Speedup vs baseline | $M_l$ calls/sample | $M_m$ saved/sample |
|--------|---------|------:|--------------------:|-------------------:|-------------------:|
| Autoregressive ($M_l$ only) | MMLU | 9.3 | — | 190.0 | — |
| Autoregressive ($M_l$ only) | GSM8K | 9.4 | — | 177.0 | — |
| Baseline CSD ($M_s \to M_l$) | MMLU | 26.6 | 1.00× | 29.7 | — |
| Baseline CSD ($M_s \to M_l$) | GSM8K | 23.9 | 1.00× | 30.4 | — |
| ACSD Cascaded (Phase 2) | MMLU | **41.1** | **1.55×** | 38.0 | 37.7 |
| ACSD Cascaded (Phase 2) | GSM8K | **39.9** | **1.67×** | 36.1 | 41.9 |
| ACSD Adaptive τ=0.4 (Phase 3) | MMLU | 39.3 | 1.48× | 38.2 | 36.2 |
| ACSD Adaptive τ=0.4 (Phase 3) | GSM8K | 26.3 | 1.10× | 38.8 | 20.5 |

### τ Ablation — MMLU (W=20, k\_m=4)

| τ | Tok/s | $M_l$ calls/sample | $M_m$ saved/sample | Switches in final window |
|---|------:|-------------------:|-------------------:|-------------------------:|
| 0.2 | 40.8 | 38.0 | 37.7 | 0.0 |
| 0.3 | **40.9** | 38.0 | 37.7 | 1.30 |
| 0.4 | 39.3 | 38.2 | 36.2 | 1.26 |
| 0.5 | 32.5 | 39.6 | 28.7 | 2.96 |

At τ=0.2 the rolling α never falls below 0.2, so no switching occurs (degenerates to cascaded). τ=0.3 is optimal for MMLU.

### τ Ablation — GSM8K (W=20, k\_m=4)

| τ | Tok/s | $M_l$ calls/sample | $M_m$ saved/sample | Switches in final window |
|---|------:|-------------------:|-------------------:|-------------------------:|
| 0.2 | **39.7** | 36.1 | 41.9 | 0.0 |
| 0.3 | 28.7 | 37.9 | 28.2 | 0.93 |
| 0.4 | 26.3 | 38.8 | 20.5 | 0.87 |
| 0.5 | 21.8 | 40.2 | 8.4 | 0.70 |

On GSM8K, any switching to $M_m$ drafting degrades throughput. τ=0.2 (no switching, degenerates to cascaded) is optimal.

### Rolling Window Ablation — MMLU (τ=0.4, k\_m=4)

| W | Tok/s | $M_l$ calls/sample | $M_m$ saved/sample |
|---|------:|-------------------:|-------------------:|
| 10 | 39.4 | 38.2 | 36.2 |
| 20 | 39.3 | 38.2 | 36.2 |
| 50 | 39.4 | 38.2 | 36.2 |

Window size has negligible effect on throughput or $M_l$ cost for MMLU.

### k\_m Ablation — MMLU (τ=0.4, W=20)

| k\_m | Tok/s | $M_l$ calls/sample | $M_m$ saved/sample |
|------|------:|-------------------:|-------------------:|
| 2 | 33.9 | 39.2 | 36.2 |
| 4 | **39.3** | 38.2 | 36.2 |
| 6 | 33.3 | 37.8 | 36.2 |

k\_m=4 is optimal. Smaller k\_m underutilises $M_m$ promotion steps; larger k\_m keeps $M_m$ in drafter mode too long, increasing per-step latency.

---

## Re-running Experiments

```bash
conda activate acsd
cd /mnt/data/gaokaizhang/mlsys/CS-Drafting

# autoregressive / baseline: M_l only (~26 GB)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode baseline --dataset mmlu --n_samples 100 --device cuda:0 \
    --output results/baseline_mmlu.json

# cascaded / adaptive: all 3 models (~42 GB)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode adaptive --dataset mmlu --n_samples 100 \
    --tau 0.4 --window_size 20 --k_m 4 --device cuda:0 \
    --output results/adaptive_mmlu_tau04.json
```

GPU memory: `autoregressive`/`baseline` need ~26 GB; `cascaded`/`adaptive` need ~42 GB (all 3 models).

Each result file is a JSON with three top-level keys: `config` (hyperparameters), `summary` (tok/s, avg_wall_time, avg_ml_calls), and `raw` (per-sample arrays).

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
- 2026-04: ACSD extension — cascaded pre-verification and adaptive role-switching (`acsd.py`, `main_acsd.py`, `model.py`)
- 2026-04: Updated for transformers ≥ 4.45 (DynamicCache API, `cais/mmlu` Parquet dataset)
- 2026-04-08: All main experiments and ablations complete; results in `results/`

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
