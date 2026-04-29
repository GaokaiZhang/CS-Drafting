"""
Run an explicit fixed-window comparison set from one model load.

This is meant for presentation-oriented reruns where we want a small, curated
set of baselines and hierarchical configurations rather than a full grid.
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


def _normalize_optional(raw):
    if raw is None:
        return None
    value = str(raw).strip()
    if value in ("", "-", "_", "none", "None", "null"):
        return None
    return value


def parse_config_specs(raw_specs):
    specs = []
    for index, raw_spec in enumerate(part for part in raw_specs.split(";") if part.strip()):
        fields = [field.strip() for field in raw_spec.split("|")]
        if len(fields) not in (6, 7):
            raise ValueError(
                "Each config spec must have 6 or 7 pipe-separated fields: "
                "label|run_type|small_window|middle_window|variant|policy|baseline_label"
            )

        if len(fields) == 6:
            fields.append("")

        label, run_type, small_window, middle_window, variant, policy, baseline_label = fields
        run_type = run_type.lower()
        if run_type not in ("baseline", "hierarchical"):
            raise ValueError(f"Unsupported run_type '{run_type}' in spec '{raw_spec}'")

        spec = {
            "label": label,
            "run_type": run_type,
            "small_window": int(small_window),
            "middle_window": (
                int(_normalize_optional(middle_window))
                if _normalize_optional(middle_window) is not None
                else None
            ),
            "hierarchical_variant": _normalize_optional(variant),
            "window_policy": _normalize_optional(policy),
            "baseline_label": _normalize_optional(baseline_label),
            "order": index,
        }

        if run_type == "baseline":
            spec["baseline_label"] = None
        else:
            if spec["middle_window"] is None:
                raise ValueError(f"Hierarchical spec '{raw_spec}' requires a middle_window")
            if spec["hierarchical_variant"] is None:
                raise ValueError(
                    f"Hierarchical spec '{raw_spec}' requires a hierarchical_variant"
                )
            if spec["window_policy"] is None:
                raise ValueError(f"Hierarchical spec '{raw_spec}' requires a window_policy")

        specs.append(spec)

    if not specs:
        raise ValueError("At least one config spec is required")

    labels = [spec["label"] for spec in specs]
    duplicates = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        raise ValueError(f"Duplicate config labels are not allowed: {sorted(duplicates)}")
    return specs


def _default_baseline_label(spec):
    return f"baseline_sw{spec['small_window']}"


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
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--config_specs", required=True)
    parser.add_argument("--dynamic_small_window_min", type=int)
    parser.add_argument("--dynamic_small_window_max", type=int)
    parser.add_argument("--dynamic_middle_window_min", type=int, default=3)
    parser.add_argument("--dynamic_middle_window_max", type=int, default=18)
    parser.add_argument("--dynamic_acceptance_low", type=float, default=0.6)
    parser.add_argument("--dynamic_acceptance_high", type=float, default=0.8)
    parser.add_argument("--dynamic_window_step", type=int, default=1)
    parser.add_argument("--dynamic_utility_margin", type=float, default=0.0)
    parser.add_argument("--selective_route_warmup_blocks", type=int, default=1)
    parser.add_argument("--selective_route_history_window", type=int, default=4)
    parser.add_argument("--selective_route_utility_margin", type=float, default=0.0)
    parser.add_argument("--selective_route_direct_acceptance_low", type=float, default=0.7)
    parser.add_argument("--selective_route_direct_acceptance_high", type=float, default=0.85)
    parser.add_argument("--selective_route_middle_acceptance_low", type=float)
    parser.add_argument("--selective_route_probe_interval", type=int)
    parser.add_argument("--adapt_small_window", action="store_true")
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--trace_samples", type=int, default=0)
    parser.add_argument("--disable_stop_on_answer", action="store_true")
    parser.add_argument("--skip_missing_baseline_comparisons", action="store_true")
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
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "small_window": 3,
            "middle_window": 6,
            "hierarchical_variant": "double_layer",
            "window_policy": "fixed",
            "adapt_small_window": args.adapt_small_window,
            "dynamic_small_window_min": args.dynamic_small_window_min,
            "dynamic_small_window_max": args.dynamic_small_window_max,
            "dynamic_middle_window_min": args.dynamic_middle_window_min,
            "dynamic_middle_window_max": args.dynamic_middle_window_max,
            "dynamic_acceptance_low": args.dynamic_acceptance_low,
            "dynamic_acceptance_high": args.dynamic_acceptance_high,
            "dynamic_window_step": args.dynamic_window_step,
            "dynamic_utility_margin": args.dynamic_utility_margin,
            "selective_route_warmup_blocks": args.selective_route_warmup_blocks,
            "selective_route_history_window": args.selective_route_history_window,
            "selective_route_utility_margin": args.selective_route_utility_margin,
            "selective_route_direct_acceptance_low": (
                args.selective_route_direct_acceptance_low
            ),
            "selective_route_direct_acceptance_high": (
                args.selective_route_direct_acceptance_high
            ),
            "selective_route_middle_acceptance_low": (
                args.selective_route_middle_acceptance_low
            ),
            "selective_route_probe_interval": args.selective_route_probe_interval,
            "max_length": args.max_length,
            "trace_samples": args.trace_samples,
            "stop_on_answer": not args.disable_stop_on_answer,
            "device": args.small_device,
            "output": args.output,
            "dtype": torch.float16,
        }
    )
    return cfg


def _run_config(spec, base_cfg, selected_items, tokenizer, m_s, m_m, m_l):
    run_cfg = copy.deepcopy(base_cfg)
    run_cfg["run_type"] = spec["run_type"]
    run_cfg["small_window"] = spec["small_window"]
    if spec["run_type"] == "hierarchical":
        run_cfg.update(
            {
                "middle_window": spec["middle_window"],
                "hierarchical_variant": spec["hierarchical_variant"],
                "window_policy": spec["window_policy"],
            }
        )
    return run_cfg, run_eval(
        spec["run_type"],
        run_cfg,
        selected_items,
        tokenizer,
        m_s,
        m_m,
        m_l,
    )


def main():
    args = parse_args()
    specs = parse_config_specs(args.config_specs)
    base_cfg = _base_config(args)

    print("Base config:")
    pprint(serialize_config(base_cfg))
    print("Focused specs:")
    pprint(specs)

    m_s, m_m, m_l, tokenizer = load_models(base_cfg, need_middle=True)
    test_set = get_test_set(base_cfg["dataset"])
    selected_items = select_eval_items(test_set, base_cfg)
    requested_n_samples = base_cfg.get("n_samples")
    if requested_n_samples is None or int(requested_n_samples) <= 0:
        requested_n_samples = len(test_set)

    result = {
        "config": serialize_config(base_cfg),
        "selection": {
            "requested_n_samples": requested_n_samples,
            "selected_n_samples": len(selected_items),
            "num_shards": base_cfg["num_shards"],
            "shard_index": base_cfg["shard_index"],
            "selected_sample_indices": [sample_index for sample_index, _ in selected_items],
        },
        "specs": [
            {
                key: value
                for key, value in spec.items()
                if key != "order"
            }
            for spec in specs
        ],
        "runs": {},
        "comparisons": {},
        "comparison_baselines": {},
    }

    ordered_specs = sorted(specs, key=lambda spec: (spec["run_type"] != "baseline", spec["order"]))
    for spec in ordered_specs:
        print(f"\nRunning {spec['label']}")
        run_cfg, run_result = _run_config(
            spec,
            base_cfg,
            selected_items,
            tokenizer,
            m_s,
            m_m,
            m_l,
        )
        run_result["config"] = serialize_config(run_cfg)
        result["runs"][spec["label"]] = run_result
        save_result(result, args.output)

    best_label = None
    best_tps_delta = None
    for spec in specs:
        if spec["run_type"] != "hierarchical":
            continue
        baseline_label = spec["baseline_label"] or _default_baseline_label(spec)
        if baseline_label not in result["runs"]:
            if args.skip_missing_baseline_comparisons:
                continue
            raise ValueError(
                f"Baseline label '{baseline_label}' for '{spec['label']}' was not run"
            )
        baseline_summary = result["runs"][baseline_label]["summary"]
        candidate_summary = result["runs"][spec["label"]]["summary"]
        result["comparisons"][spec["label"]] = comparison_summary(
            candidate_summary,
            baseline_summary,
        )
        result["comparison_baselines"][spec["label"]] = baseline_label

        tps_delta = (
            candidate_summary["tokens_per_sec"] - baseline_summary["tokens_per_sec"]
        )
        if best_tps_delta is None or tps_delta > best_tps_delta:
            best_tps_delta = tps_delta
            best_label = spec["label"]

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
    print(f"\nSaved focused results to {args.output}")


if __name__ == "__main__":
    main()
