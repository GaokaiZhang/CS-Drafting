"""
Run a compact fixed-window tuning sweep from one model load.

The baseline depends on the small draft window, so this runner evaluates one
baseline per `small_window`, then evaluates hierarchical configurations against
the matching baseline.
"""

import argparse
import copy
import re
from pprint import pprint

import torch

from csd_datasets import get_test_set
from fixed_window import comparison_summary
from main_fixed_window import (
    config as default_config,
    load_models,
    run_eval,
    save_result,
    select_eval_items,
    serialize_config,
)


def _csv_values(raw, cast=str):
    return [cast(value.strip()) for value in re.split(r"[:,]", raw) if value.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k"], default="gsm8k")
    parser.add_argument("--ms_name", required=True)
    parser.add_argument("--mm_name", required=True)
    parser.add_argument("--ml_name", required=True)
    parser.add_argument("--small_device", default="cuda:0")
    parser.add_argument("--middle_device", default="cuda:0")
    parser.add_argument("--large_device", default="cuda:1")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--small_window", type=int, default=3)
    parser.add_argument("--small_windows", default=None)
    parser.add_argument("--middle_windows", default="6,9,12,15")
    parser.add_argument("--hierarchical_variants", default="double_layer")
    parser.add_argument("--window_policies", default="fixed,utility")
    parser.add_argument("--dynamic_middle_window_min", type=int, default=3)
    parser.add_argument("--dynamic_middle_window_max", type=int, default=18)
    parser.add_argument("--dynamic_acceptance_low", type=float, default=0.6)
    parser.add_argument("--dynamic_acceptance_high", type=float, default=0.8)
    parser.add_argument("--dynamic_window_step", type=int, default=1)
    parser.add_argument("--dynamic_utility_margin", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--trace_samples", type=int, default=0)
    parser.add_argument("--disable_stop_on_answer", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _base_config(args):
    cfg = copy.deepcopy(default_config)
    cfg.update(
        {
            "mode": "compare",
            "dataset": args.dataset,
            "ms_name": args.ms_name,
            "mm_name": args.mm_name,
            "ml_name": args.ml_name,
            "small_device": args.small_device,
            "middle_device": args.middle_device,
            "large_device": args.large_device,
            "n_samples": args.n_samples,
            "num_shards": 1,
            "shard_index": 0,
            "small_window": args.small_window,
            "middle_window": _csv_values(args.middle_windows, int)[0],
            "hierarchical_variant": _csv_values(args.hierarchical_variants)[0],
            "window_policy": _csv_values(args.window_policies)[0],
            "adapt_small_window": False,
            "dynamic_middle_window_min": args.dynamic_middle_window_min,
            "dynamic_middle_window_max": args.dynamic_middle_window_max,
            "dynamic_acceptance_low": args.dynamic_acceptance_low,
            "dynamic_acceptance_high": args.dynamic_acceptance_high,
            "dynamic_window_step": args.dynamic_window_step,
            "dynamic_utility_margin": args.dynamic_utility_margin,
            "max_length": args.max_length,
            "trace_samples": args.trace_samples,
            "stop_on_answer": not args.disable_stop_on_answer,
            "device": args.small_device,
            "output": args.output,
            "dtype": torch.float16,
        }
    )
    return cfg


def main():
    args = parse_args()
    cfg = _base_config(args)
    small_windows = (
        _csv_values(args.small_windows, int)
        if args.small_windows
        else [int(args.small_window)]
    )
    middle_windows = _csv_values(args.middle_windows, int)
    variants = _csv_values(args.hierarchical_variants)
    policies = _csv_values(args.window_policies)

    print("Base config:")
    pprint(serialize_config(cfg))
    print(
        "Tuning "
        f"small_windows={small_windows} "
        f"middle_windows={middle_windows} "
        f"variants={variants} "
        f"policies={policies}"
    )

    m_s, m_m, m_l, tokenizer = load_models(cfg, need_middle=True)
    test_set = get_test_set(cfg["dataset"])
    selected_items = select_eval_items(test_set, cfg)

    result = {
        "config": serialize_config(cfg),
        "selection": {
            "requested_n_samples": cfg["n_samples"],
            "selected_n_samples": len(selected_items),
            "selected_sample_indices": [sample_index for sample_index, _ in selected_items],
        },
        "runs": {},
        "comparisons": {},
        "comparison_baselines": {},
    }

    best_label = None
    best_tps_delta = None

    for small_window in small_windows:
        baseline_cfg = copy.deepcopy(cfg)
        baseline_cfg["small_window"] = small_window
        baseline_label = (
            "baseline" if len(small_windows) == 1 else f"baseline_sw{small_window}"
        )
        print(f"\nRunning baseline {baseline_label}")
        baseline = run_eval(
            "baseline",
            baseline_cfg,
            selected_items,
            tokenizer,
            m_s,
            m_m,
            m_l,
        )
        result["runs"][baseline_label] = baseline

        for variant in variants:
            for policy in policies:
                for middle_window in middle_windows:
                    candidate_cfg = copy.deepcopy(cfg)
                    candidate_cfg.update(
                        {
                            "small_window": small_window,
                            "hierarchical_variant": variant,
                            "window_policy": policy,
                            "middle_window": middle_window,
                        }
                    )
                    label = f"{variant}_sw{small_window}_mw{middle_window}_{policy}"
                    print(f"\nRunning candidate {label}")
                    candidate = run_eval(
                        "hierarchical",
                        candidate_cfg,
                        selected_items,
                        tokenizer,
                        m_s,
                        m_m,
                        m_l,
                    )
                    result["runs"][label] = candidate
                    result["comparisons"][label] = comparison_summary(
                        candidate["summary"],
                        baseline["summary"],
                    )
                    result["comparison_baselines"][label] = baseline_label

                    tps_delta = (
                        candidate["summary"]["tokens_per_sec"]
                        - baseline["summary"]["tokens_per_sec"]
                    )
                    if best_tps_delta is None or tps_delta > best_tps_delta:
                        best_tps_delta = tps_delta
                        best_label = label
                    save_result(result, args.output)

    if best_label is not None:
        best_summary = result["runs"][best_label]["summary"]
        best_baseline = result["runs"][result["comparison_baselines"][best_label]]["summary"]
        print("\nBest candidate:")
        pprint(
            {
                "label": best_label,
                "tokens_per_sec": best_summary["tokens_per_sec"],
                "benchmark_score": best_summary["benchmark_score"],
                "baseline_tokens_per_sec": best_baseline["tokens_per_sec"],
                "baseline_benchmark_score": best_baseline["benchmark_score"],
                "tokens_per_sec_gain_pct": (
                    (best_summary["tokens_per_sec"] / best_baseline["tokens_per_sec"] - 1.0)
                    * 100.0
                ),
            }
        )

    save_result(result, args.output)
    print(f"\nSaved tuning results to {args.output}")


if __name__ == "__main__":
    main()
