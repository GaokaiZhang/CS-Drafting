"""
Experiment runner for double-layer speculative decoding.

Modes
-----
  autoregressive  Pure M_l autoregressive (reference only, no M_s/M_m)
  baseline        Standard CSD: M_s → M_l
  double_layer    Base method:  M_s/M_m inner cascade × k_m → M_l
  proxy_entropy   Full method:  double_layer + entropy proxy on M_s
  proxy_top1      Full method:  double_layer + top-1 log-prob proxy on M_s
  proxy_margin    Full method:  double_layer + top1-top2 margin proxy on M_s
  proxy_mavg      Full method:  double_layer + moving-avg log-prob proxy on M_s

Key hyperparameters
-------------------
  --k_s            M_s draft window (max tokens per inner step)
  --k_m            Outer batch size fed to M_l  (k_m > k_s)
  --proxy_threshold Override default confidence threshold for proxy modes
  --mavg_window    Window length for proxy_mavg (default 5)
  --ms_name / --mm_name / --ml_name   Override HuggingFace model IDs or local paths
"""

import argparse
import json
import os
import re
import time
import torch
from tqdm import tqdm
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer

from csd import csd
from acsd import acsd_double_layer, acsd_proxy
from model import (
    CountedCSDraftingDecoderModel,
    CountedCSDraftingDecoderModelKVCache,
    ACSDMiddleTierModel,
)
from csd_datasets import get_test_set, format_initial_input, format_accuracy_input


# ── mode constants ─────────────────────────────────────────────────────────────

_PROXY_MODES = {'proxy_entropy', 'proxy_top1', 'proxy_margin', 'proxy_mavg'}
_DOUBLE_LAYER_MODES = {'double_layer'} | _PROXY_MODES
_ALL_MODES = ['autoregressive', 'baseline', 'double_layer',
              'proxy_entropy', 'proxy_top1', 'proxy_margin', 'proxy_mavg']

# Default thresholds for each proxy type
_PROXY_DEFAULTS = {
    'entropy': 2.0,   # stop if H(p) > 2.0 nats
    'top1':   -1.5,   # stop if log p_max < -1.5  (max prob ~22%)
    'margin':  1.0,   # stop if log p_1 - log p_2 < 1.0
    'mavg':   -1.5,   # stop if moving-avg log p_max < -1.5
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',             type=str, choices=_ALL_MODES, required=True)
    p.add_argument('--dataset',          type=str, choices=['mmlu', 'gsm8k'], required=True)
    p.add_argument('--k_s',             type=int,   default=5)
    p.add_argument('--k_m',             type=int,   default=10)
    p.add_argument('--n_samples',        type=int,   default=100)
    p.add_argument('--device',           type=str,   default='cuda:0')
    p.add_argument('--proxy_threshold',  type=float, default=None,
                   help='Override default confidence threshold for proxy modes')
    p.add_argument('--mavg_window',      type=int,   default=5)
    p.add_argument('--leniency',         type=int,   default=1)
    p.add_argument('--max_length',       type=int,   default=200)
    p.add_argument('--ms_name',          type=str,   default=None,
                   help='M_s HuggingFace model ID or local path')
    p.add_argument('--mm_name',          type=str,   default=None,
                   help='M_m HuggingFace model ID or local path')
    p.add_argument('--ml_name',          type=str,   default=None,
                   help='M_l HuggingFace model ID or local path')
    p.add_argument('--output',           type=str,   default=None,
                   help='Path to write JSON results')
    p.add_argument('--answer_only',      action='store_true',
                   help='Accuracy-only mode: direct-answer prompt (no CoT) for MMLU; '
                        'longer budget for GSM8K. Does not affect throughput measurement.')
    return p.parse_args()


# ── default model paths (local server) ────────────────────────────────────────

_HF_MODELS = '/mnt/data/xuandong/hf_models'
_DEFAULT_MS = (f'{_HF_MODELS}/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0'
               '/snapshots/77e23968eed12d195bd46c519aa679cc22a27ddc')
_DEFAULT_MM = (f'{_HF_MODELS}/models--NousResearch--Llama-2-7b-hf'
               '/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc')
_DEFAULT_ML = (f'{_HF_MODELS}/models--NousResearch--Llama-2-13b-hf'
               '/snapshots/b0491461253755d8c60bf22f0d696b9e337c6375')


# ── model loading ──────────────────────────────────────────────────────────────

def load_models(args):
    ms_name = args.ms_name or _DEFAULT_MS
    mm_name = args.mm_name or _DEFAULT_MM
    ml_name = args.ml_name or _DEFAULT_ML
    device  = args.device
    dtype   = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(ms_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'autoregressive':
        print(f"Loading M_l: {ml_name}")
        ml_hf = AutoModelForCausalLM.from_pretrained(ml_name, torch_dtype=dtype, device_map=device)
        m_l = CountedCSDraftingDecoderModelKVCache(ml_hf, name=ml_name, vocab_size=32000)
        return None, None, m_l, tokenizer, ml_name

    if args.mode == 'baseline':
        print(f"Loading M_s: {ms_name}")
        ms_hf = AutoModelForCausalLM.from_pretrained(ms_name, torch_dtype=dtype, device_map=device)
        m_s = CountedCSDraftingDecoderModel(ms_hf, name=ms_name, vocab_size=32000)
        print(f"Loading M_l: {ml_name}")
        ml_hf = AutoModelForCausalLM.from_pretrained(ml_name, torch_dtype=dtype, device_map=device)
        m_l = CountedCSDraftingDecoderModelKVCache(ml_hf, name=ml_name, vocab_size=32000)
        return m_s, None, m_l, tokenizer, ml_name

    # double_layer / proxy_* — need all three models
    print(f"Loading M_s: {ms_name}")
    ms_hf = AutoModelForCausalLM.from_pretrained(ms_name, torch_dtype=dtype, device_map=device)
    m_s = CountedCSDraftingDecoderModel(ms_hf, name=ms_name, vocab_size=32000)
    print(f"Loading M_m: {mm_name}")
    mm_hf = AutoModelForCausalLM.from_pretrained(mm_name, torch_dtype=dtype, device_map=device)
    m_m = ACSDMiddleTierModel(mm_hf, name=mm_name, vocab_size=32000)
    print(f"Loading M_l: {ml_name}")
    ml_hf = AutoModelForCausalLM.from_pretrained(ml_name, torch_dtype=dtype, device_map=device)
    m_l = CountedCSDraftingDecoderModelKVCache(ml_hf, name=ml_name, vocab_size=32000)
    return m_s, m_m, m_l, tokenizer, ml_name


# ── accuracy helpers ───────────────────────────────────────────────────────────

def _parse_mmlu_answer(text, answer_only=False):
    """Return predicted choice index (0–3) or None.

    answer_only=True: prompt ended with 'The answer is (' so the very first
    A/B/C/D character in the output is the answer.
    answer_only=False: CoT output — scan for explicit answer phrases.
    """
    if answer_only:
        m = re.match(r'\s*([A-D])', text, re.IGNORECASE)
        return ord(m.group(1).upper()) - ord('A') if m else None

    # CoT mode: look for explicit answer phrases.
    for pattern in [
        r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+\(?([A-D])\)?',
        r'answer[:\s]+\(?([A-D])\)?',
        r'\(?([A-D])\)?\s+is\s+(?:correct|right|the\s+answer)',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return ord(m.group(1).upper()) - ord('A')
    # Last-resort: last standalone A/B/C/D
    hits = list(re.finditer(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text))
    return ord(hits[-1].group(1).upper()) - ord('A') if hits else None


def _parse_gsm8k_answer(text):
    """Extract final numeric answer from model output.

    Priority:
      1. '#### N' (fine-tuned format)
      2. 'the answer is N'
      3. First '= $N' (dollar amount — appears before any looping noise)
      4. Last standalone number in text
    """
    m = re.search(r'####\s*(-?[\d,]+)', text)
    if m:
        return float(m.group(1).replace(',', ''))
    m = re.search(r'the\s+answer\s+is[:\s]+\$?(-?[\d,]+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))
    # First dollar-amount result (= $N) — most reliable before repetition loops.
    m = re.search(r'=\s*\$(-?[\d,]+)', text)
    if m:
        return float(m.group(1).replace(',', ''))
    nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    return float(nums[-1].replace(',', '')) if nums else None


def _gt_gsm8k(item):
    """Extract ground-truth number from GSM8K answer string (after ####)."""
    m = re.search(r'####\s*([\d,]+)', item['answer'])
    if not m:
        nums = re.findall(r'\d[\d,]*', item['answer'])
        return float(nums[-1].replace(',', '')) if nums else None
    return float(m.group(1).replace(',', ''))


def _is_correct(generated_text, item, dataset_name, answer_only=False):
    if dataset_name == 'mmlu':
        pred = _parse_mmlu_answer(generated_text, answer_only=answer_only)
        return int(pred == item['answer']) if pred is not None else 0
    elif dataset_name == 'gsm8k':
        pred = _parse_gsm8k_answer(generated_text)
        gt   = _gt_gsm8k(item)
        if pred is None or gt is None:
            return 0
        return int(abs(pred - gt) < 0.5)
    return 0


# ── evaluation loop ────────────────────────────────────────────────────────────

def run_eval(args, m_s, m_m, m_l, tokenizer, test_set):
    results = {
        'wall_times':         [],
        'tokens_generated':   [],
        'ml_forward_calls':   [],
        'mm_forward_calls':   [],
        'ms_forward_calls':   [],
        'mm_rejected_by_ms':  [],   # M_s tokens M_m rejected (inner loop)
        'correct':            [],   # 1 if answer correct, 0 otherwise
        'generated_texts':    [],   # decoded output (for debugging / offline eval)
    }

    proxy_type = args.mode.split('_', 1)[1] if args.mode in _PROXY_MODES else None
    threshold = args.proxy_threshold
    if proxy_type is not None and threshold is None:
        threshold = _PROXY_DEFAULTS[proxy_type]

    answer_only = getattr(args, 'answer_only', False)
    # In answer_only mode: MMLU uses a direct-answer prompt (5 tokens enough);
    # GSM8K uses extra budget to let the model finish its reasoning.
    max_length = args.max_length
    if answer_only:
        max_length = 5 if args.dataset == 'mmlu' else 200

    for item in tqdm(test_set[:args.n_samples], desc=args.mode, smoothing=0):
        text = (format_accuracy_input(item, args.dataset)
                if answer_only else format_initial_input(item, args.dataset))
        initial_input = tokenizer(
            text, truncation=True, padding=False, return_tensors='pt'
        )['input_ids'].to(args.device)
        input_ids = initial_input.clone()

        m_l.forward_count = 0
        m_l.wall_time = []
        if m_s is not None:
            m_s.forward_count = 0
        if m_m is not None:
            m_m.forward_count = 0
            m_m.acceptance_history = []
            m_m.saved_ml_positions = 0

        t0 = time.time()

        if args.mode == 'autoregressive':
            cur = input_ids.clone()
            init_len = cur.shape[-1]
            with torch.no_grad():
                while cur.shape[-1] - init_len < max_length:
                    out = m_l.model(cur, use_cache=False)
                    tok = torch.argmax(out.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                    cur = torch.cat([cur, tok], dim=1)
                    m_l.forward_count += 1
                    if tok.item() == 2:
                        break
            output_ids = cur

        elif args.mode == 'baseline':
            k_mat = torch.tensor([[args.k_s, max_length],
                                  [0,         max_length]])
            output_ids = csd(
                [m_s], m_l, initial_input, input_ids,
                k_mat, max_length=max_length, leniency=args.leniency,
            )

        elif args.mode == 'double_layer':
            output_ids = acsd_double_layer(
                m_s, m_m, m_l, initial_input, input_ids,
                k_s=args.k_s, k_m=args.k_m,
                leniency=args.leniency, max_length=max_length,
            )

        else:  # proxy_*
            output_ids = acsd_proxy(
                m_s, m_m, m_l, initial_input, input_ids,
                k_s=args.k_s, k_m=args.k_m,
                proxy_type=proxy_type,
                threshold=threshold,
                mavg_window=args.mavg_window,
                leniency=args.leniency, max_length=max_length,
            )

        wall = time.time() - t0
        n_gen = output_ids.shape[-1] - initial_input.shape[-1]

        generated_text = tokenizer.decode(
            output_ids[0, initial_input.shape[-1]:], skip_special_tokens=True
        )

        results['wall_times'].append(wall)
        results['tokens_generated'].append(n_gen)
        results['ml_forward_calls'].append(m_l.forward_count)
        results['mm_forward_calls'].append(m_m.forward_count if m_m else 0)
        results['ms_forward_calls'].append(m_s.forward_count if m_s else 0)
        results['mm_rejected_by_ms'].append(m_m.saved_ml_positions if m_m else 0)
        results['correct'].append(_is_correct(generated_text, item, args.dataset,
                                              answer_only=answer_only))
        results['generated_texts'].append(generated_text)

    return results


# ── summary ────────────────────────────────────────────────────────────────────

def summarise(results, args, ml_name, proxy_type=None, threshold=None):
    n              = len(results['wall_times'])
    total_tok      = sum(results['tokens_generated'])
    total_wall     = sum(results['wall_times'])
    tok_per_sec    = total_tok / total_wall if total_wall > 0 else 0
    avg_ml         = sum(results['ml_forward_calls']) / n
    avg_mm         = sum(results['mm_forward_calls']) / n
    avg_ms         = sum(results['ms_forward_calls']) / n
    avg_mm_rej     = sum(results['mm_rejected_by_ms']) / n
    accuracy       = sum(results['correct']) / n if results['correct'] else None

    print('\n' + '='*60)
    print(f"Mode:            {args.mode}")
    print(f"Dataset:         {args.dataset}  (n={n})")
    print(f"M_l:             {ml_name.split('/')[-1]}")
    if args.mode in _DOUBLE_LAYER_MODES:
        print(f"k_s={args.k_s}  k_m={args.k_m}")
    if proxy_type:
        print(f"Proxy:           {proxy_type}  threshold={threshold}"
              f"  mavg_window={args.mavg_window}")
    print(f"Tokens/sec:      {tok_per_sec:.1f}")
    print(f"Avg wall/sample: {total_wall/n:.2f}s")
    print(f"Avg M_l calls:   {avg_ml:.1f}")
    if avg_mm > 0:
        print(f"Avg M_m calls:   {avg_mm:.1f}")
    if avg_ms > 0:
        print(f"Avg M_s calls:   {avg_ms:.1f}")
    if avg_mm_rej > 0:
        print(f"Avg M_s toks rejected by M_m: {avg_mm_rej:.1f}")
    if accuracy is not None:
        print(f"Accuracy:        {accuracy:.3f}  ({sum(results['correct'])}/{n})")
    print('='*60)

    return {
        'tokens_per_sec': tok_per_sec,
        'avg_wall_time':  total_wall / n,
        'avg_ml_calls':   avg_ml,
        'avg_mm_calls':   avg_mm,
        'avg_ms_calls':   avg_ms,
        'avg_mm_rejected_by_ms': avg_mm_rej,
        'accuracy':       accuracy,
    }


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    proxy_type = args.mode.split('_', 1)[1] if args.mode in _PROXY_MODES else None
    threshold  = args.proxy_threshold
    if proxy_type and threshold is None:
        threshold = _PROXY_DEFAULTS[proxy_type]

    pprint(vars(args))

    m_s, m_m, m_l, tokenizer, ml_name = load_models(args)
    test_set = get_test_set(args.dataset)

    results = run_eval(args, m_s, m_m, m_l, tokenizer, test_set)
    summary = summarise(results, args, ml_name, proxy_type, threshold)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        out = {
            'config': {
                'mode': args.mode, 'dataset': args.dataset,
                'k_s': args.k_s, 'k_m': args.k_m,
                'n_samples': args.n_samples,
                'proxy_type': proxy_type, 'proxy_threshold': threshold,
                'mavg_window': args.mavg_window, 'leniency': args.leniency,
                'max_length': args.max_length,
            },
            'summary': summary,
            'raw': {
                'wall_times':        results['wall_times'],
                'tokens_generated':  results['tokens_generated'],
                'ml_forward_calls':  results['ml_forward_calls'],
                'mm_forward_calls':  results['mm_forward_calls'],
                'ms_forward_calls':  results['ms_forward_calls'],
                'mm_rejected_by_ms': results['mm_rejected_by_ms'],
                'correct':           results['correct'],
                'generated_texts':   results['generated_texts'],
            },
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Results → {args.output}')
