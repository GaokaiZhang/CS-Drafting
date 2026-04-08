# Cascade Speculative Drafting + ACSD Extension

Original paper: "[Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462)"

This repo contains the original CS-Drafting implementation **plus our course-project extension: Adaptive Cascaded Speculative Decoding (ACSD)**, which adds a third middle-tier model to the cascade with cascaded pre-verification and adaptive role-switching.

> **For teammates:** See [`status.md`](status.md) for current project status, experiment instructions, and deliverable deadlines.

## ACSD System

| Tier | Model | Role |
|------|-------|------|
| $M_s$ | TinyLlama-1.1B | Small drafter |
| $M_m$ | LLaMA-2-7B | Pre-verifier or promoted drafter |
| $M_l$ | LLaMA-2-13B | Final verifier (always runs — losslessness guaranteed) |

**Phase 2 (cascaded):** $M_m$ filters $M_s$'s draft tokens before $M_l$ sees them, reducing $M_l$ forward-pass cost.

**Phase 3 (adaptive):** When $M_s$'s rolling acceptance rate drops below threshold $\tau$, $M_m$ is promoted to drafter, bypassing $M_s$ entirely.

## Experiment Results

Main results (100 samples, max 200 tokens, single A6000 48 GB GPU, fp16):

| Method | Dataset | Tok/s | $M_l$ calls/sample | $M_m$ saved/sample |
|--------|---------|-------|---------------------|--------------------|
| Autoregressive ($M_l$ only) | MMLU | 9.3 | 190.0 | — |
| Autoregressive ($M_l$ only) | GSM8K | 9.4 | 177.0 | — |
| Baseline CSD ($M_s \to M_l$) | MMLU | 26.6 | 29.7 | — |
| Baseline CSD ($M_s \to M_l$) | GSM8K | 23.9 | 30.4 | — |
| ACSD Cascaded (Phase 2) | MMLU | **41.1** (1.55×) | 38.0 | 37.7 |
| ACSD Cascaded (Phase 2) | GSM8K | **39.9** (1.67×) | 36.1 | 41.9 |
| ACSD Adaptive τ=0.4 (Phase 3) | MMLU | 39.3 (1.48×) | 38.2 | 36.2 |
| ACSD Adaptive τ=0.4 (Phase 3) | GSM8K | 26.3 (1.10×) | 38.8 | 20.5 |

Speedup is relative to Baseline CSD.

**τ ablation (MMLU, W=20):** τ=0.2→40.7, τ=0.3→**40.9**, τ=0.4→39.3, τ=0.5→32.5 tok/s.

**Window size ablation (MMLU, τ=0.4):** W=10→39.4, W=20→39.3, W=50→39.4 tok/s; throughput is negligibly affected by window size.

## Re-running Experiments

```bash
conda activate acsd
cd /mnt/data/gaokaizhang/mlsys/CS-Drafting

# baseline/autoregressive: M_l only (~26 GB)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode baseline --dataset mmlu --n_samples 100 --device cuda:0 \
    --output results/baseline_mmlu.json

# cascaded/adaptive: all 3 models (~42 GB)
env CUDA_VISIBLE_DEVICES=<GPU> python main_acsd.py \
    --mode adaptive --dataset mmlu --n_samples 100 \
    --tau 0.4 --window_size 20 --device cuda:0 \
    --output results/adaptive_mmlu_tau04.json
```

Result files are JSON in `results/`. Each contains `config`, `summary` (tok/s, avg_wall_time, avg_ml_calls), and `raw` arrays.

## Change Log
- 2024-04-02: Added KV Cache to reduce latency when generation length is long
- 2026-04: ACSD extension: cascaded pre-verification and adaptive role-switching (`acsd.py`, `main_acsd.py`, `model.py`)
- 2026-04: Updated for transformers ≥ 4.45 (DynamicCache API, `cais/mmlu` Parquet dataset)
- 2026-04-07: All main experiments complete; results in `results/`


## Setup

### Original environment (python 3.9)

```
conda create --name csd python=3.9
conda activate csd
pip install -r requirements.txt
```

### ACSD environment (python 3.11, recommended)

Required for running `main_acsd.py` with modern LLaMA models:

```
conda create -n acsd python=3.11 -y
conda activate acsd
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.45 datasets>=2.20 accelerate sentencepiece tqdm
```

> **Note:** CUDA 12.4 (`cu124`) requires driver ≥ 520. If your driver supports CUDA 12.6+ use `cu126` instead.
> CUDA_VISIBLE_DEVICES may need to be set explicitly via `env CUDA_VISIBLE_DEVICES=<id> python ...` depending on your cluster configuration.


## ACSD Experiments

`main_acsd.py` is the entry point for the three-tier ACSD system (TinyLlama-1.1B → LLaMA-2-7B → LLaMA-2-13B).
All settings are in the `config` dict at the top of the file.

```python
config = {
    'ms_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',   # or local path
    'mm_name': 'meta-llama/Llama-2-7b-hf',
    'ml_name': 'meta-llama/Llama-2-13b-hf',
    'mode':    'adaptive',   # 'baseline' | 'cascaded' | 'adaptive'
    'dataset': 'mmlu',       # 'mmlu' | 'gsm8k'
    'n_samples': 100,
    ...
}
```

Run:
```
env CUDA_VISIBLE_DEVICES=0 python main_acsd.py
```

All three models in fp16 fit on a single 48 GB GPU (~41 GB total).

**MMLU note:** The original `lukaemon/mmlu` dataset uses a loading script no longer supported by `datasets ≥ 2.20`. The code now uses `cais/mmlu` (Parquet format, identical content).


## Recreating Original CS-Drafting Experiments

The starting point of the report is main.py which can be run without args for maximum hackability.
All experiment setting can be adjusted in the config diction in main.py.
GPU usage can be adjusted by changing the following line in main.py

```
usable_devices = [0, 1, 2] * 2
```

Each index in the list representing a single process on gpu of the index.
Note that target model is cached in ./cache, so running each process will cost less than 8GB of memory.
We recommend using 2 process for each GPU with at least 16gb of memory for higher GPU utiliization.

To run experiments with FLAN-T5 on mmlu for SWI (model size) setup, change the config to the following:

```
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

For SWI (previous work)

```
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

To run LLAMA-7B on mmlu

```
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

To run gsm8k, you can change the dataset field in the config to 

```
'dataset': 'gsm8k'
```
Note that when using two draft models other than mag, the parameter
in k_matrix is different from the one in the paper. Their relations are the following:
```
k_matrix[0][0] = k<sub>11</sub>
k_matrix[0][1] = k<sub>12</sub> - k_matrix[0][0]
```



## Using CS Drafting for Inference

To run csd on your own inputs

```
from csd import csd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from model import CountedCSDraftingDecoderModelKVCache, get_mag_model


draft_list = []
draft_names = ['JackFram/llama-160m']
for draft_name in draft_names:
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(draft_name)
    model = CountedCSDraftingDecoderModelKVCache(hf_model, name=draft_name, counter_version=config['counter_version'])
    draft_list.append(model)

_BIGRAM_DIR = './bigram_models/'
bi_gram_path = _BIGRAM_DIR + 'wiki_bigram_naive_bayers_greedy_llama_next_token.json'
mag_model = get_mag_model(bi_gram_path, config['is_decoder_only'])
draft_list.append(mag_model)

LLAMA_HF_PATH = LLAMA_PATH + 'hf_7b_chat'
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = <your hugginface llama tokenizer>
hf_model = <your hugginface llama model>

target_model = CountedCSDraftingDecoderModelKVCache(hf_model, name='llama', vocab_size=32000)
target_model.cuda(device)

question = '<Your inputs>'
initial_input = tokenizer(question, truncation=True, padding=False, return_tensors="pt")['input_ids'].to(target_model.device)
input_ids = initial_input
res = csd(draft_list, target_model, initial_input, input_ids, k_matrix, max_length=200)
generated_text = tokenizer.batch_decode(res, skip_special_tokens=True)
```

 

## Citation

The details of this repo are described in the following paper:

```
@article{chen2023cascade,
  title={Cascade Speculative Drafting for Even Faster LLM Inference},
  author={Chen, Ziyi and Yang, Xiaocong and Lin, Jiacheng and Sun, Chenkai and Chen, Yangyi and Chang, Kevin Chen-Chuan and Huang, Jie},
  journal={arXiv preprint arXiv:2312.11462},
  year={2023}
}
```

