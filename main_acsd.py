"""
Entry point for ACSD experiments.

Usage:
    python main_acsd.py [--mode MODE] [--dataset DS] [--tau TAU]
                        [--window_size W] [--n_samples N] [--device DEV]
                        [--output PATH]

Edit the `config` dict below to choose defaults, or override via CLI.
  - mode:    'autoregressive' — pure M_l token-by-token (no speculation)
             'baseline'   — standard CSD with M_s → M_l (no M_m)
             'cascaded'   — Phase 2: M_s → M_m pre-verify → M_l
             'adaptive'   — Phase 3: adaptive switching based on rolling alpha
  - dataset: 'mmlu' | 'gsm8k'
  - tau, window_size, k_s, k_m: ACSD hyperparameters
  - output:  path to write JSON results (optional)
"""

import argparse
import json
import os
import time
import torch
from tqdm import tqdm
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer

from csd import csd
from acsd import acsd_cascaded, acsd_adaptive
from model import (
    CountedCSDraftingDecoderModel,
    CountedCSDraftingDecoderModelKVCache,
    ACSDMiddleTierModel,
)
from csd_datasets import get_test_set, format_initial_input


# ── CLI args ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',        type=str, choices=['autoregressive', 'baseline', 'cascaded', 'adaptive'])
    p.add_argument('--dataset',     type=str, choices=['mmlu', 'gsm8k'])
    p.add_argument('--tau',         type=float)
    p.add_argument('--window_size', type=int)
    p.add_argument('--k_s',         type=int)
    p.add_argument('--k_m',         type=int)
    p.add_argument('--n_samples',   type=int)
    p.add_argument('--device',      type=str)
    p.add_argument('--output',      type=str, help='path to write JSON results')
    return p.parse_args()


# ── config ─────────────────────────────────────────────────────────────────────

_HF_MODELS = '/mnt/data/xuandong/hf_models'

config = {
    # Local model paths
    'ms_name':  f'{_HF_MODELS}/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/77e23968eed12d195bd46c519aa679cc22a27ddc',
    'mm_name':  f'{_HF_MODELS}/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc',
    'ml_name':  f'{_HF_MODELS}/models--NousResearch--Llama-2-13b-hf/snapshots/b0491461253755d8c60bf22f0d696b9e337c6375',

    # Run mode
    'mode': 'adaptive',     # 'baseline' | 'cascaded' | 'adaptive'

    # Dataset
    'dataset': 'mmlu',      # 'mmlu' | 'gsm8k'
    'n_samples': 20,        # number of test examples to evaluate

    # ACSD hyperparameters
    'k_s': 5,               # tokens M_s drafts per step
    'k_m': 4,               # tokens M_m drafts when promoted (adaptive only)
    'leniency': 1,          # acceptance leniency (1 = exact greedy match)
    'tau': 0.4,             # rolling-alpha threshold for switching M_s → M_m
    'window_size': 20,      # rolling window length for alpha computation

    # Generation
    'max_length': 200,

    # Device (single GPU)
    'device': 'cuda:0',

    # Model dtype
    'dtype': torch.float16,
}


# ── model loading ──────────────────────────────────────────────────────────────

def load_models(cfg):
    device = cfg['device']
    dtype = cfg['dtype']

    print(f"Loading M_s: {cfg['ms_name']}")
    ms_hf = AutoModelForCausalLM.from_pretrained(
        cfg['ms_name'], torch_dtype=dtype, device_map=device
    )
    m_s = CountedCSDraftingDecoderModel(ms_hf, name=cfg['ms_name'], vocab_size=32000)

    tokenizer = AutoTokenizer.from_pretrained(cfg['ms_name'])
    tokenizer.pad_token = tokenizer.eos_token

    if cfg['mode'] in ('autoregressive', 'baseline'):
        print(f"Loading M_l: {cfg['ml_name']}")
        ml_hf = AutoModelForCausalLM.from_pretrained(
            cfg['ml_name'], torch_dtype=dtype, device_map=device
        )
        m_l = CountedCSDraftingDecoderModelKVCache(ml_hf, name=cfg['ml_name'], vocab_size=32000)
        if cfg['mode'] == 'autoregressive':
            return None, None, m_l, tokenizer
        return m_s, None, m_l, tokenizer

    print(f"Loading M_m: {cfg['mm_name']}")
    mm_hf = AutoModelForCausalLM.from_pretrained(
        cfg['mm_name'], torch_dtype=dtype, device_map=device
    )
    m_m = ACSDMiddleTierModel(mm_hf, name=cfg['mm_name'], vocab_size=32000)

    print(f"Loading M_l: {cfg['ml_name']}")
    ml_hf = AutoModelForCausalLM.from_pretrained(
        cfg['ml_name'], torch_dtype=dtype, device_map=device
    )
    m_l = CountedCSDraftingDecoderModelKVCache(ml_hf, name=cfg['ml_name'], vocab_size=32000)

    return m_s, m_m, m_l, tokenizer


# ── evaluation ─────────────────────────────────────────────────────────────────

def run_eval(cfg, m_s, m_m, m_l, tokenizer, test_set):
    device = cfg['device']
    results = {
        'wall_times': [],
        'tokens_generated': [],
        'ml_forward_calls': [],
        'mm_saved_positions': [],
        'alpha_traces': [],          # only for adaptive mode
        'switch_counts': [],         # only for adaptive mode
    }

    for item in tqdm(test_set[:cfg['n_samples']], desc=cfg['mode']):
        text_input = format_initial_input(item, cfg['dataset'])
        initial_input = tokenizer(
            text_input, truncation=True, padding=False, return_tensors='pt'
        )['input_ids'].to(device)
        input_ids = initial_input.clone()

        # Reset forward counters
        m_l.forward_count = 0
        m_l.wall_time = []
        if m_m is not None:
            m_m.forward_count = 0
            m_m.acceptance_history = []
            m_m.saved_ml_positions = 0

        t0 = time.time()

        if cfg['mode'] == 'autoregressive':
            cur_ids = input_ids.clone()
            initial_len = cur_ids.shape[-1]
            with torch.no_grad():
                while cur_ids.shape[-1] - initial_len < cfg['max_length']:
                    out = m_l.model(cur_ids, use_cache=False)
                    next_tok = torch.argmax(out.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                    cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                    m_l.forward_count += 1
                    if next_tok.item() == 2:  # EOS
                        break
            output_ids = cur_ids
            state = None

        elif cfg['mode'] == 'baseline':
            k_matrix = torch.tensor([[cfg['k_s'], cfg['max_length']],
                                     [0,           cfg['max_length']]])
            output_ids = csd(
                [m_s], m_l, initial_input, input_ids,
                k_matrix, max_length=cfg['max_length'], leniency=cfg['leniency']
            )
            state = None

        elif cfg['mode'] == 'cascaded':
            output_ids = acsd_cascaded(
                m_s, m_m, m_l, initial_input, input_ids,
                k_s=cfg['k_s'], leniency=cfg['leniency'],
                max_length=cfg['max_length'],
            )
            state = None

        else:  # adaptive
            output_ids, state = acsd_adaptive(
                m_s, m_m, m_l, initial_input, input_ids,
                k_s=cfg['k_s'], k_m=cfg['k_m'],
                leniency=cfg['leniency'], tau=cfg['tau'],
                window_size=cfg['window_size'],
                max_length=cfg['max_length'],
            )

        wall = time.time() - t0
        n_generated = output_ids.shape[-1] - initial_input.shape[-1]

        results['wall_times'].append(wall)
        results['tokens_generated'].append(n_generated)
        results['ml_forward_calls'].append(m_l.forward_count)
        if m_m is not None:
            results['mm_saved_positions'].append(m_m.saved_ml_positions)
        if state is not None:
            results['alpha_traces'].append(list(state.alpha_window))
            n_switches = sum(
                1 for i in range(1, len(state.alpha_window))
                if (state.alpha_window[i-1] >= cfg['tau']) != (state.alpha_window[i] >= cfg['tau'])
            )
            results['switch_counts'].append(n_switches)

    return results


def summarise(results, cfg):
    n = len(results['wall_times'])
    total_tokens = sum(results['tokens_generated'])
    total_wall   = sum(results['wall_times'])
    avg_ml_calls = sum(results['ml_forward_calls']) / n
    tokens_per_sec = total_tokens / total_wall if total_wall > 0 else 0

    print('\n' + '='*60)
    print(f"Mode:            {cfg['mode']}")
    print(f"Dataset:         {cfg['dataset']}  (n={n})")
    print(f"Models:          M_s={cfg['ms_name'].split('/')[-1]}")
    if cfg['mode'] != 'baseline':
        print(f"                 M_m={cfg['mm_name'].split('/')[-1]}")
    print(f"                 M_l={cfg['ml_name'].split('/')[-1]}")
    print(f"Avg wall time:   {total_wall/n:.2f}s per sample")
    print(f"Tokens/sec:      {tokens_per_sec:.1f}")
    print(f"Avg M_l calls:   {avg_ml_calls:.1f} per sample")
    if results['mm_saved_positions']:
        avg_saved = sum(results['mm_saved_positions']) / n
        print(f"Avg M_l positions saved by M_m:  {avg_saved:.1f} per sample")
    if results['switch_counts']:
        avg_sw = sum(results['switch_counts']) / n
        print(f"Avg drafter switches (tau={cfg['tau']}):  {avg_sw:.1f} per sample")
    print('='*60)
    return {
        'tokens_per_sec': tokens_per_sec,
        'avg_wall_time':  total_wall / n,
        'avg_ml_calls':   avg_ml_calls,
    }


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    # Override config with any CLI args provided
    for key in ('mode', 'dataset', 'tau', 'window_size', 'k_s', 'k_m', 'n_samples', 'device'):
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    print('Config:')
    pprint(config)

    m_s, m_m, m_l, tokenizer = load_models(config)
    test_set = get_test_set(config['dataset'])

    results = run_eval(config, m_s, m_m, m_l, tokenizer, test_set)
    summary = summarise(results, config)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        out = {
            'config': {k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                       for k, v in config.items()},
            'summary': summary,
            'raw': {
                'wall_times':         results['wall_times'],
                'tokens_generated':   results['tokens_generated'],
                'ml_forward_calls':   results['ml_forward_calls'],
                'mm_saved_positions': results['mm_saved_positions'],
                'switch_counts':      results['switch_counts'],
            },
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nResults saved to {args.output}')
