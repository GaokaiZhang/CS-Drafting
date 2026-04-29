import argparse
import json
import math
import re
from copy import deepcopy
from pathlib import Path

from fixed_window import comparison_summary, new_run_stats, summary_from_merge_stats


COUNTER_KEYS = (
    "draft_generated_counts",
    "final_source_counts",
    "verification_positions",
    "verification_calls",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--skip_missing_baseline_comparisons", action="store_true")
    return parser.parse_args()


def _load_result(path):
    with open(path) as handle:
        return json.load(handle)


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
        "draft_window_totals": {
            "small": {"sum": 0, "count": 0},
            "middle": {"sum": 0, "count": 0},
        },
        "window_change_count": 0,
        "routing_totals": {"middle": 0, "large": 0},
        "routing_change_count": 0,
        "routing_seen": False,
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
    for model_key, totals in src.get("draft_window_totals", {}).items():
        if model_key not in dst["draft_window_totals"]:
            continue
        dst["draft_window_totals"][model_key]["sum"] += totals.get("sum", 0)
        dst["draft_window_totals"][model_key]["count"] += totals.get("count", 0)
    dst["window_change_count"] += src.get("window_change_count", 0)
    for route_key, value in src.get("routing_totals", {}).items():
        if route_key in dst["routing_totals"]:
            dst["routing_totals"][route_key] += value
    dst["routing_change_count"] += src.get("routing_change_count", 0)
    dst["routing_seen"] = dst["routing_seen"] or src.get("routing_seen", False)
    for model_key, runtime in src["runtime_totals"].items():
        bucket = dst["runtime_totals"][model_key]
        for field in bucket:
            bucket[field] += runtime.get(field, 0)


def _default_baseline_label(spec):
    return f"baseline_sw{spec['small_window']}"


def _normalize_inputs(input_paths):
    return sorted(str(Path(path).resolve()) for path in input_paths)


def _canonicalize_model_ref(value):
    if not isinstance(value, str):
        return value

    match = re.search(
        r"/models--(?P<namespace>[^/]+)--(?P<repo>[^/]+)(?:/snapshots/[^/]+)?/?$",
        value,
    )
    if match:
        return f"{match.group('namespace')}/{match.group('repo')}"

    match = re.search(
        r"/hf_models/(?P<namespace>[^/]+)/(?P<repo>[^/]+?)/?$",
        value,
    )
    if match:
        return f"{match.group('namespace')}/{match.group('repo')}"

    return value


def _normalize_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _normalize_json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(child) for child in value]
    return value


def _normalize_config(config):
    normalized = _normalize_json_value(deepcopy(config))
    for key in ("ms_name", "mm_name", "ml_name"):
        if key in normalized:
            normalized[key] = _canonicalize_model_ref(normalized[key])
    return normalized


def _normalize_run_config(config):
    normalized = _normalize_config(config)
    for optional_key in (
        "dynamic_small_window_min",
        "dynamic_small_window_max",
        "selective_route_middle_acceptance_low",
        "selective_route_probe_interval",
    ):
        normalized.setdefault(optional_key, None)
    normalized["num_shards"] = 1
    normalized["shard_index"] = 0
    normalized.pop("output", None)
    for key in [key for key in normalized if key.startswith("_")]:
        normalized.pop(key, None)
    return normalized


def _expected_run_labels(payload):
    specs = payload.get("specs")
    if specs is not None:
        labels = [spec["label"] for spec in specs]
        for spec in specs:
            baseline_label = spec.get("baseline_label")
            if baseline_label and baseline_label not in labels:
                labels.append(baseline_label)
        if not labels:
            raise ValueError("Focused shard specs did not include any run labels.")
        return labels
    labels = list(payload.get("runs", {}))
    if not labels:
        raise ValueError("Shard does not contain any runs.")
    return labels


def _validate_complete_runs(payload, expected_labels, path):
    runs = payload.get("runs", {})
    missing = [label for label in expected_labels if label not in runs]
    extra = [label for label in runs if label not in expected_labels]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing runs: {missing}")
        if extra:
            details.append(f"unexpected runs: {extra}")
        raise ValueError(f"Shard '{path}' is incomplete ({'; '.join(details)}).")

    selected_n_samples = payload.get("selection", {}).get("selected_n_samples")
    for label in expected_labels:
        run_data = runs[label]
        for field in ("summary", "aggregate_counters", "merge_stats"):
            if field not in run_data:
                raise ValueError(
                    f"Shard '{path}' run '{label}' is incomplete: missing '{field}'."
                )
        if selected_n_samples is not None:
            sample_metrics = run_data.get("sample_metrics")
            if sample_metrics is not None and len(sample_metrics) != int(selected_n_samples):
                raise ValueError(
                    f"Shard '{path}' run '{label}' has {len(sample_metrics)} sample "
                    f"metrics for {selected_n_samples} selected samples."
                )


def merge_results(input_paths, skip_missing_baseline_comparisons=False):
    input_paths = _normalize_inputs(input_paths)
    if not input_paths:
        raise ValueError("No input shard files were provided.")

    loaded = [_load_result(path) for path in input_paths]
    normalized_loaded = []
    for payload in loaded:
        normalized_payload = deepcopy(payload)
        normalized_payload["config"] = _normalize_config(payload["config"])
        normalized_loaded.append(normalized_payload)
    comparable_fields = (
        "mode",
        "dataset",
        "ms_name",
        "mm_name",
        "ml_name",
        "small_window",
        "middle_window",
        "max_length",
        "n_samples",
        "stop_on_answer",
    )
    for payload in normalized_loaded[1:]:
        for field in comparable_fields:
            if payload["config"].get(field) != normalized_loaded[0]["config"].get(field):
                raise ValueError(
                    f"Shard configs do not match for field '{field}': "
                    f"{payload['config'].get(field)} != {normalized_loaded[0]['config'].get(field)}"
                )
        if payload.get("specs") != normalized_loaded[0].get("specs"):
            raise ValueError("Shard focused specs do not match.")

    expected_run_labels = _expected_run_labels(normalized_loaded[0])
    for path, payload in zip(input_paths, normalized_loaded):
        _validate_complete_runs(payload, expected_run_labels, path)

    base_config = deepcopy(normalized_loaded[0]["config"])
    base_config["num_shards"] = 1
    base_config["shard_index"] = 0
    base_config["trace_samples"] = max(
        payload["config"].get("trace_samples", 0) for payload in normalized_loaded
    )

    merged = {
        "config": base_config,
        "selection": {
            "requested_n_samples": loaded[0].get("selection", {}).get(
                "requested_n_samples",
                normalized_loaded[0]["config"].get("n_samples"),
            ),
            "selected_n_samples": 0,
            "num_shards": len(input_paths),
            "shard_index": None,
            "selected_sample_indices": [],
            "merged_inputs": input_paths,
        },
        "runs": {},
    }
    if normalized_loaded[0].get("specs") is not None:
        merged["specs"] = deepcopy(normalized_loaded[0]["specs"])
        merged["comparisons"] = {}
        merged["comparison_baselines"] = {}

    merged_runs = {}
    for payload in normalized_loaded:
        merged["selection"]["selected_sample_indices"].extend(
            payload.get("selection", {}).get("selected_sample_indices", [])
        )
        for run_name, run_data in payload["runs"].items():
            if run_name not in merged_runs:
                merged_runs[run_name] = {
                    "config": _normalize_run_config(
                        run_data.get("config", payload["config"])
                    ),
                    "aggregate_counters": new_run_stats(),
                    "merge_stats": _new_merge_stats(),
                    "sample_metrics": [],
                    "trace_samples": [],
                }
            bucket = merged_runs[run_name]
            run_config = _normalize_run_config(run_data.get("config", payload["config"]))
            if run_config != bucket["config"]:
                raise ValueError(f"Shard run configs do not match for run '{run_name}'.")
            _merge_counter_block(bucket["aggregate_counters"], run_data["aggregate_counters"])
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
            bucket["config"],
            bucket["aggregate_counters"],
            bucket["merge_stats"],
        )
        merged["runs"][run_name] = {
            "config": bucket["config"],
            "summary": summary,
            "aggregate_counters": bucket["aggregate_counters"],
            "merge_stats": bucket["merge_stats"],
            "sample_metrics": bucket["sample_metrics"],
            "trace_samples": bucket["trace_samples"],
        }

    if merged.get("specs"):
        for spec in merged["specs"]:
            if spec.get("run_type") != "hierarchical":
                continue
            label = spec["label"]
            baseline_label = spec.get("baseline_label") or _default_baseline_label(spec)
            if label not in merged["runs"] or baseline_label not in merged["runs"]:
                if skip_missing_baseline_comparisons:
                    continue
                raise ValueError(
                    f"Missing merged runs for comparison: {label} vs {baseline_label}"
                )
            merged["comparisons"][label] = comparison_summary(
                merged["runs"][label]["summary"],
                merged["runs"][baseline_label]["summary"],
            )
            merged["comparison_baselines"][label] = baseline_label
    elif merged["config"]["mode"] == "compare":
        merged["comparison"] = comparison_summary(
            merged["runs"]["hierarchical"]["summary"],
            merged["runs"]["baseline"]["summary"],
        )

    return merged


def save_result(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    args = parse_args()
    merged = merge_results(
        args.inputs,
        skip_missing_baseline_comparisons=args.skip_missing_baseline_comparisons,
    )
    save_result(merged, args.output)
