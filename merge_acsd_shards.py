import argparse
import json
from copy import deepcopy
from pathlib import Path

from main_acsd_compare import (
    COUNTER_KEYS,
    DRAFT_WINDOW_MODEL_KEYS,
    comparison_summary,
    new_run_stats,
    summary_from_merge_stats,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    return parser.parse_args()


def _load_result(path):
    with open(path) as handle:
        return json.load(handle)


def _normalize_inputs(input_paths):
    return sorted(str(Path(path).resolve()) for path in input_paths)


def _merge_counter_block(dst, src):
    for counter_key in COUNTER_KEYS:
        for model_key, value in src[counter_key].items():
            dst[counter_key][model_key] += value
    for edge_key, counters in src["edge_pass"].items():
        dst["edge_pass"][edge_key]["accepted"] += counters["accepted"]
        dst["edge_pass"][edge_key]["proposed"] += counters["proposed"]


def _new_merge_stats():
    return {
        "n_samples": 0,
        "total_tokens": 0,
        "total_wall_time": 0.0,
        "total_score": 0.0,
        "benchmark_metric": "score",
        "ml_forward_call_sum": 0.0,
        "mm_forward_call_sum": 0.0,
        "switch_count_sum": 0.0,
        "mm_saved_positions_sum": 0.0,
        "drafter_step_sums": {
            "small": 0.0,
            "middle": 0.0,
        },
        "draft_window_policy": None,
        "draft_window_totals": {
            model_key: {"sum": 0.0, "count": 0}
            for model_key in DRAFT_WINDOW_MODEL_KEYS
        },
        "draft_window_change_count_sum": 0.0,
        "runtime_totals": {
            model_key: {
                "propose_calls": 0,
                "review_calls": 0,
                "propose_wall_time": 0.0,
                "review_wall_time": 0.0,
                "total_wall_time": 0.0,
            }
            for model_key in ("small", "middle", "large")
        },
    }


def _merge_stats_block(dst, src):
    dst["n_samples"] += src["n_samples"]
    dst["total_tokens"] += src["total_tokens"]
    dst["total_wall_time"] += src["total_wall_time"]
    dst["total_score"] += src["total_score"]
    dst["benchmark_metric"] = src["benchmark_metric"]
    dst["ml_forward_call_sum"] += src.get("ml_forward_call_sum", 0.0)
    dst["mm_forward_call_sum"] += src.get("mm_forward_call_sum", 0.0)
    dst["switch_count_sum"] += src.get("switch_count_sum", 0.0)
    dst["mm_saved_positions_sum"] += src.get("mm_saved_positions_sum", 0.0)
    dst["drafter_step_sums"]["small"] += src.get("drafter_step_sums", {}).get(
        "small", 0.0
    )
    dst["drafter_step_sums"]["middle"] += src.get("drafter_step_sums", {}).get(
        "middle", 0.0
    )
    if dst.get("draft_window_policy") is None:
        dst["draft_window_policy"] = src.get("draft_window_policy")
    for model_key in DRAFT_WINDOW_MODEL_KEYS:
        dst_bucket = dst["draft_window_totals"][model_key]
        src_bucket = src.get("draft_window_totals", {}).get(model_key, {})
        dst_bucket["sum"] += src_bucket.get("sum", 0.0)
        dst_bucket["count"] += src_bucket.get("count", 0)
    dst["draft_window_change_count_sum"] += src.get(
        "draft_window_change_count_sum",
        0.0,
    )
    for model_key, runtime in src["runtime_totals"].items():
        bucket = dst["runtime_totals"][model_key]
        for field in bucket:
            bucket[field] += runtime.get(field, 0)


def merge_results(input_paths):
    input_paths = _normalize_inputs(input_paths)
    if not input_paths:
        raise ValueError("No input shard files were provided.")

    loaded = [_load_result(path) for path in input_paths]
    comparable_fields = (
        "mode",
        "dataset",
        "ms_name",
        "mm_name",
        "ml_name",
        "k_s",
        "k_m",
        "tau",
        "window_size",
        "leniency",
        "max_length",
        "draft_window_policy",
        "dynamic_k_s_min",
        "dynamic_k_s_max",
        "dynamic_k_m_min",
        "dynamic_k_m_max",
        "dynamic_acceptance_low",
        "dynamic_acceptance_high",
        "dynamic_window_step",
        "n_samples",
    )
    for payload in loaded[1:]:
        for field in comparable_fields:
            if payload["config"].get(field) != loaded[0]["config"].get(field):
                raise ValueError(
                    f"Shard configs do not match for field '{field}': "
                    f"{payload['config'].get(field)} != {loaded[0]['config'].get(field)}"
                )

    base_config = deepcopy(loaded[0]["config"])
    base_config["num_shards"] = 1
    base_config["shard_index"] = 0
    base_config["trace_samples"] = max(
        payload["config"].get("trace_samples", 0) for payload in loaded
    )

    merged = {
        "config": base_config,
        "selection": {
            "requested_n_samples": loaded[0].get("selection", {}).get(
                "requested_n_samples",
                loaded[0]["config"].get("n_samples"),
            ),
            "selected_n_samples": 0,
            "num_shards": len(input_paths),
            "shard_index": None,
            "selected_sample_indices": [],
            "merged_inputs": input_paths,
        },
        "runs": {},
    }

    merged_runs = {}
    for payload in loaded:
        merged["selection"]["selected_sample_indices"].extend(
            payload.get("selection", {}).get("selected_sample_indices", [])
        )
        for run_name, run_data in payload["runs"].items():
            if run_name not in merged_runs:
                merged_runs[run_name] = {
                    "aggregate_counters": new_run_stats(),
                    "merge_stats": _new_merge_stats(),
                    "sample_metrics": [],
                    "trace_samples": [],
                }
            bucket = merged_runs[run_name]
            _merge_counter_block(
                bucket["aggregate_counters"], run_data["aggregate_counters"]
            )
            _merge_stats_block(bucket["merge_stats"], run_data["merge_stats"])
            bucket["sample_metrics"].extend(run_data.get("sample_metrics", []))
            bucket["trace_samples"].extend(run_data.get("trace_samples", []))

    merged["selection"]["selected_sample_indices"] = sorted(
        merged["selection"]["selected_sample_indices"]
    )
    merged["selection"]["selected_n_samples"] = len(
        merged["selection"]["selected_sample_indices"]
    )

    for run_name, bucket in merged_runs.items():
        bucket["sample_metrics"] = sorted(
            bucket["sample_metrics"], key=lambda sample: sample["sample_index"]
        )
        bucket["trace_samples"] = sorted(
            bucket["trace_samples"], key=lambda sample: sample["sample_index"]
        )[: merged["config"]["trace_samples"]]
        summary = summary_from_merge_stats(
            run_name,
            merged["config"],
            bucket["aggregate_counters"],
            bucket["merge_stats"],
        )
        merged["runs"][run_name] = {
            "summary": summary,
            "aggregate_counters": bucket["aggregate_counters"],
            "merge_stats": bucket["merge_stats"],
            "sample_metrics": bucket["sample_metrics"],
            "trace_samples": bucket["trace_samples"],
        }

    if "baseline" in merged["runs"]:
        baseline_summary = merged["runs"]["baseline"]["summary"]
        comparisons = {}
        for candidate_name in ("cascaded", "adaptive"):
            if candidate_name in merged["runs"]:
                comparisons[f"{candidate_name}_vs_baseline"] = comparison_summary(
                    merged["runs"][candidate_name]["summary"],
                    baseline_summary,
                )
        if comparisons:
            merged["comparisons"] = comparisons

    return merged


def save_result(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    args = parse_args()
    merged = merge_results(args.inputs)
    save_result(merged, args.output)
