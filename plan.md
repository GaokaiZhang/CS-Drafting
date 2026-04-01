# Adaptive Cascaded Speculative Decoding (ACSD) — Implementation Plan

## Overview

Extend CS-Drafting to implement the three-tier speculative decoding system proposed in
"Adaptive Cascaded Speculative Decoding for Efficient LLM Inference" (Yin, Wu, Zhang).

### Model Tiers

| Role     | Model              | Size  | Notes                                 |
|----------|--------------------|-------|---------------------------------------|
| M_s      | TinyLLaMA-1.1B     | 1.1B  | Fast drafter; may be demoted          |
| M_m      | LLaMA-2-7B (or 3B) | 7B    | Dual role: pre-verifier OR drafter    |
| M_l      | LLaMA-2-13B        | 13B   | Final verifier; always runs last      |

M_l always performs final verification — losslessness is preserved by design.

---

## Step 0: Environment & Dependency Upgrade

### Problem
The original repo pins:
- `torch==2.0.1` (Feb 2023)
- `transformers==4.34.1` (Aug 2023)
- `fschat` (superseded; only used for model loading)

These block loading modern LLaMA models (3.x requires transformers ≥ 4.40).

### Fix: new conda environment

```bash
conda create -n acsd python=3.11 -y
conda activate acsd
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.45 datasets>=2.20 accelerate sentencepiece
```

### Changes to `requirements.txt`
- Remove `fschat`
- Remove pinned `torch==2.0.1`
- Add `accelerate`, `sentencepiece`
- Pin ranges instead of exact versions

### Remove `fschat` dependency
`fschat` is only used for model path resolution in `main.py`. Replace with plain
`AutoModelForCausalLM.from_pretrained(name)` from HuggingFace.

### Fix module-level side-effect in `model.py`
Line 405 runs `AutoTokenizer.from_pretrained('JackFram/llama-160m')` at import time.
This must be removed or moved into the class that needs it.

---

## Step 1: Baseline — Three-Tier CSD (no new algorithm)

**Goal:** Reproduce standard cascade speculative decoding with three tiers using
modern models, verifying the existing `csd.py` loop works end-to-end.

**Config (new `main_acsd.py`):**
```python
config = {
    'draft_names': ['TinyLlama/TinyLlama-1.1B-Chat-v1.0'],
    'target_name': 'meta-llama/Llama-2-7b-hf',   # M_m as target for baseline
    'is_decoder_only': True,
    'use_mag': False,
    'k_matrix': [[5, 10], [0, 10]],
    'lenience': 1,
    'dataset': 'mmlu',
}
```

**No algorithm changes.** Purely verifies the updated stack works.

**Deliverable:** Wall-clock time and token acceptance rate for (M_s → M_l) baseline.

---

## Step 2: Cascaded Pre-Verification (RQ1)

**Goal:** Insert M_m as a cheap filter between M_s drafts and M_l final verification.

### Mechanism
```
M_s drafts k tokens
    → M_m.pre_verify() filters the draft sequence
        → M_l.review() only sees the filtered (shorter) sequence
```

M_m rejects low-quality draft tokens early, reducing the length of the sequence M_l
must verify. Each token M_m rejects = one fewer position M_l has to process.

### New class: `ACSDMiddleTierModel` in `model.py`

Extends `CSDraftingDecoderModelKVCache` with:
- `pre_verify(initial_input, input_ids, probs, review_index, leniency)`:
  wraps `review()`, additionally tracks how many tokens were accepted/rejected
- `acceptance_history: list[float]` — per-step acceptance ratio
- `saved_ml_positions: int` — cumulative count of positions saved from M_l

### New file: `acsd.py`

Core function:
```python
def acsd_cascaded(m_s, m_m, m_l, initial_input, input_ids, k_s, leniency, max_length):
    """Phase 2: M_s drafts, M_m pre-verifies, M_l final-verifies."""
```

Internal step:
1. Call `m_s.propose(initial_input, input_ids, k_s)` → draft tokens
2. Call `m_m.pre_verify(...)` → filtered shorter sequence + m_m probs
3. Call `m_l.review(...)` on the filtered sequence → accepted tokens

**Losslessness:** M_l still calls `review()` on every step. M_m only reduces the
number of draft positions M_l sees, never replaces M_l's decision.

**Deliverable:** Compare M_l forward-pass count and wall-clock time vs. Step 1 baseline.

---

## Step 3: Adaptive Role-Switching (RQ2 & RQ3)

**Goal:** When M_s's rolling acceptance rate α falls below threshold τ, promote M_m
from pre-verifier to drafter, bypassing M_s entirely.

### Mechanism
```
if rolling_alpha(M_s, window=W) >= τ:
    active_drafter = M_s  (fast path, Phase 2 logic)
else:
    active_drafter = M_m  (M_m drafts directly → M_l verifies)
```

### New class: `AdaptiveCSDState`

```python
@dataclass
class AdaptiveCSDState:
    alpha_window: deque          # rolling acceptance rates of M_s
    window_size: int = 20
    tau: float = 0.4             # switching threshold (tune via ablation)
    current_drafter: str = 'ms'  # 'ms' or 'mm'
```

### Rolling acceptance rate computation

After each `review()` call, compute:
```
alpha = tokens_accepted / tokens_drafted
```
This is derivable from `(output_len - review_index)` vs `(input_len - review_index)`.
Appended to `alpha_window`; rolling mean triggers switch when below τ.

### KV cache invalidation on switch

When `current_drafter` changes from M_s to M_m mid-sequence:
- M_m's `past_key_values` may have stale entries
- The existing `longest_common_prefix` logic in `CSDraftingDecoderModelKVCache`
  handles partial reuse; ensure M_m's cache is invalidated (`self.past_key_values = None`)
  at the moment of role switch

### Hyperparameters to ablate (RQ3)
- `tau` ∈ {0.2, 0.3, 0.4, 0.5}
- `window_size` ∈ {10, 20, 50}
- `k_m_draft` (tokens M_m drafts when promoted) ∈ {3, 4, 5}

**Deliverable:** Speedup curve over τ values; acceptance-rate trace over generation.

---

## File Changes Summary

| File | Action | What changes |
|------|--------|--------------|
| `requirements.txt` | Modify | Remove fschat, unpin torch, add accelerate |
| `model.py` | Modify | Remove line-405 side-effect; add `ACSDMiddleTierModel` |
| `csd.py` | No change | Reused as-is for baseline |
| `acsd.py` | New | Phase 2 & 3 generation loops |
| `main_acsd.py` | New | Entry point with 3-model config + ACSD hyperparams |
| `csd_datasets.py` | Modify | Add MT-Bench and HumanEval loaders alongside MMLU/GSM8K |

---

## Metrics to Track

| Metric | How | Where |
|--------|-----|-------|
| Wall-clock speedup | `time.time()` around generation loop | `main_acsd.py` |
| M_l forward passes | `forward_count` in `CountedCSDraftingDecoderModelKVCache` | `model.py` |
| M_l positions saved by M_m | `saved_ml_positions` in `ACSDMiddleTierModel` | `model.py` |
| M_s rolling acceptance rate α | `AdaptiveCSDState.alpha_window` | `acsd.py` |
| Output quality | Token-level match vs. M_l greedy baseline | `main_acsd.py` |

---

## Hardware

All three models (TinyLLaMA-1.1B + LLaMA-2-7B + LLaMA-2-13B) in fp16 ≈ 42 GB total.
Fits on a single RTX A6000 (48 GB). No H100/A100 needed.
Node has 8× A6000 (7 free), 128-core AMD EPYC, 1 TB RAM.

---

## Work Order

1. [ ] **Env setup** — create `acsd` conda env, install deps
2. [ ] **Dep upgrade** — update `requirements.txt`, fix `model.py` line 405, remove fschat from `main.py`
3. [ ] **Step 1** — run baseline with TinyLLaMA → LLaMA-13B, verify numbers
4. [ ] **Step 2** — implement `ACSDMiddleTierModel` + `acsd.py` Phase 2, measure M_l savings
5. [ ] **Step 3** — implement `AdaptiveCSDState`, adaptive switching loop, ablate τ
6. [ ] **Eval** — run on MMLU, GSM8K, MT-Bench; collect speedup and quality tables
