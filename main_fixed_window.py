"""
Fixed-window speculative decoding comparison runner.

This implements:
  - baseline:      small -> large with a fixed small draft window
  - hierarchical:  small -> middle every small window, then large every middle window
  - compare:       run both and write a UI-friendly comparison artifact
"""

import argparse
import glob
import json
import os
import re
import time
from pprint import pprint
from types import MethodType

import torch
from tqdm import tqdm

from csd_datasets import format_initial_input, get_test_set
from fixed_window import (
    comparison_summary,
    run_baseline_sample,
    run_double_layer_sample,
    run_hierarchical_sample,
    summarize_run,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "hierarchical", "compare"], default="compare")
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k"], default="mmlu")
    parser.add_argument("--ms_name", type=str)
    parser.add_argument("--mm_name", type=str)
    parser.add_argument("--ml_name", type=str)
    parser.add_argument("--small_device", type=str)
    parser.add_argument("--middle_device", type=str)
    parser.add_argument("--large_device", type=str)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--small_window", type=int, default=3)
    parser.add_argument("--middle_window", type=int, default=9)
    parser.add_argument(
        "--hierarchical_variant",
        choices=[
            "window_fill",
            "double_layer",
            "filter_only",
            "double_layer_filter_only",
            "refill_on_full_accept",
            "double_layer_refill_on_full_accept",
            "confidence_refill",
            "double_layer_confidence_refill",
            "selective_route",
            "selective_route_refill_on_full_accept",
            "cost_aware_selective_route",
            "cost_aware_selective_route_refill_on_full_accept",
            "proxy_entropy",
            "proxy_top1",
            "proxy_margin",
            "proxy_mavg",
        ],
        default="window_fill",
    )
    parser.add_argument("--proxy_threshold", type=float)
    parser.add_argument("--mavg_window", type=int, default=5)
    parser.add_argument(
        "--window_policy",
        choices=["fixed", "adaptive", "acceptance", "utility"],
        default="fixed",
    )
    parser.add_argument(
        "--adapt_small_window",
        "--adaptive_small_window",
        action="store_true",
        dest="adapt_small_window",
        help="Allow adaptive policies to change the small-model draft chunk size.",
    )
    parser.add_argument("--dynamic_small_window_min", type=int)
    parser.add_argument("--dynamic_small_window_max", type=int)
    parser.add_argument("--dynamic_middle_window_min", type=int)
    parser.add_argument("--dynamic_middle_window_max", type=int)
    parser.add_argument("--dynamic_acceptance_low", type=float, default=0.7)
    parser.add_argument("--dynamic_acceptance_high", type=float, default=0.9)
    parser.add_argument("--dynamic_window_step", type=int, default=1)
    parser.add_argument("--dynamic_utility_margin", type=float, default=0.0)
    parser.add_argument("--selective_route_warmup_blocks", type=int, default=1)
    parser.add_argument("--selective_route_history_window", type=int, default=4)
    parser.add_argument("--selective_route_utility_margin", type=float, default=0.0)
    parser.add_argument("--selective_route_direct_acceptance_low", type=float, default=0.7)
    parser.add_argument("--selective_route_direct_acceptance_high", type=float, default=0.85)
    parser.add_argument("--selective_route_middle_acceptance_low", type=float)
    parser.add_argument("--selective_route_probe_interval", type=int)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--trace_samples", type=int, default=0)
    parser.add_argument("--disable_stop_on_answer", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="results/fixed_window_compare.json")
    return parser.parse_args()


_HF_MODELS = "/mnt/data/xuandong/hf_models"

config = {
    "ms_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "mm_name": "Qwen/Qwen2.5-7B-Instruct",
    "ml_name": "Qwen/Qwen2.5-14B-Instruct",
    "mode": "compare",
    "dataset": "mmlu",
    "small_device": "cuda:0",
    "middle_device": "cuda:1",
    "large_device": "cuda:2",
    "n_samples": 20,
    "num_shards": 1,
    "shard_index": 0,
    "small_window": 3,
    "middle_window": 9,
    "hierarchical_variant": "window_fill",
    "proxy_threshold": None,
    "mavg_window": 5,
    "window_policy": "fixed",
    "adapt_small_window": False,
    "dynamic_small_window_min": None,
    "dynamic_small_window_max": None,
    "dynamic_middle_window_min": None,
    "dynamic_middle_window_max": None,
    "dynamic_acceptance_low": 0.7,
    "dynamic_acceptance_high": 0.9,
    "dynamic_window_step": 1,
    "dynamic_utility_margin": 0.0,
    "selective_route_warmup_blocks": 1,
    "selective_route_history_window": 4,
    "selective_route_utility_margin": 0.0,
    "selective_route_direct_acceptance_low": 0.7,
    "selective_route_direct_acceptance_high": 0.85,
    "selective_route_middle_acceptance_low": 0.55,
    "selective_route_probe_interval": 0,
    "max_length": 200,
    "trace_samples": 0,
    "stop_on_answer": True,
    "device": "cuda:0",
    "dtype": torch.float16,
}


def _align_probs_last_dim(probs, vocab_size):
    if probs is None:
        return None
    current = probs.shape[-1]
    if current == vocab_size:
        return probs
    if current > vocab_size:
        return probs[..., :vocab_size]
    pad = probs.new_zeros(*probs.shape[:-1], vocab_size - current)
    return torch.cat([probs, pad], dim=-1)


def _move_tensor_to_device(value, device):
    if value is None or device is None:
        return value
    target = torch.device(device)
    if (
        torch.is_tensor(value)
        and value.device.type == "cuda"
        and target.type == "cuda"
        and value.device != target
        and not torch.is_floating_point(value)
    ):
        return value.detach().cpu().to(target)
    return value.to(device)


def _harmonize_model_vocab_sizes(models):
    active_models = [model for model in models if model is not None]
    if not active_models:
        return

    native_vocab_sizes = [int(model.vocab_size) for model in active_models]
    compare_vocab_size = max(native_vocab_sizes)
    generation_vocab_size = min(native_vocab_sizes)

    for model in active_models:
        native_vocab_size = int(getattr(model, "native_vocab_size", model.vocab_size))
        model.native_vocab_size = native_vocab_size
        model.compare_vocab_size = compare_vocab_size
        model.generation_vocab_size = generation_vocab_size
        model.vocab_size = compare_vocab_size

        original_review = model.review

        def compat_review(
            self,
            initial_input,
            input_ids,
            probs,
            review_index,
            leniency=1,
            _orig=original_review,
        ):
            review_device = getattr(
                getattr(self, "model", None),
                "device",
                getattr(self, "device", None),
            )
            native_probs = _align_probs_last_dim(probs, self.native_vocab_size)
            result_ids, result_probs = _orig(
                _move_tensor_to_device(initial_input, review_device),
                _move_tensor_to_device(input_ids, review_device),
                _move_tensor_to_device(native_probs, review_device),
                review_index,
                leniency=leniency,
            )
            return result_ids, _align_probs_last_dim(result_probs, self.compare_vocab_size)

        model.review = MethodType(compat_review, model)


def _resolve_device(cfg, key):
    return cfg.get(key) or cfg["device"]


def _device_map(device):
    return {"": device} if device.startswith("cuda") else {"": "cpu"}


def _select_tokenizer_path(small_path, large_path):
    # Shared-vocab generation is bounded by the smallest model vocab, so use the
    # small-model tokenizer to keep prompt ids inside that native vocab as well.
    return small_path or large_path


def _candidate_local_model_roots():
    user = os.environ.get("USER")
    roots = []
    for root in (
        os.environ.get("HF_LOCAL_MODEL_ROOT"),
        f"/data/user_data/{user}/hf_models" if user else None,
    ):
        if root and root not in roots:
            roots.append(root)
    return roots


def _resolve_local_snapshot(model_name):
    if not model_name:
        return model_name

    if os.path.isabs(model_name):
        return model_name

    repo_id = model_name
    if os.path.isabs(model_name):
        match = re.search(
            r"/models--(?P<namespace>[^/]+)--(?P<repo>[^/]+)/snapshots/[^/]+/?$",
            model_name,
        )
        if not match:
            return model_name
        repo_id = f"{match.group('namespace')}/{match.group('repo')}"
    elif "/" not in model_name:
        return model_name

    namespace, repo = repo_id.split("/", 1)

    for root in _candidate_local_model_roots():
        local_dir = os.path.join(root, namespace, repo)
        if os.path.isdir(local_dir):
            return local_dir

    repo_key = f"models--{namespace}--{repo}"

    candidate_roots = []
    for root in (
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
        "/data/hf_cache/hub",
        _HF_MODELS,
    ):
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    def _latest_snapshot(repo_cache):
        if not os.path.isdir(repo_cache):
            return None

        ref_path = os.path.join(repo_cache, "refs", "main")
        if os.path.exists(ref_path):
            with open(ref_path) as handle:
                revision = handle.read().strip()
            snapshot_path = os.path.join(repo_cache, "snapshots", revision)
            if os.path.isdir(snapshot_path):
                return snapshot_path

        snapshots = sorted(glob.glob(os.path.join(repo_cache, "snapshots", "*")))
        return snapshots[-1] if snapshots else None

    for attempt in range(3):
        for root in candidate_roots:
            snapshot_path = _latest_snapshot(os.path.join(root, repo_key))
            if snapshot_path:
                return snapshot_path
        if attempt < 2 and candidate_roots:
            time.sleep(1)

    return repo_id


def load_models(cfg, need_middle):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model import ACSDMiddleTierModel, CountedCSDraftingDecoderModelKVCache

    dtype = cfg["dtype"]
    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    small_device = _resolve_device(cfg, "small_device")
    middle_device = _resolve_device(cfg, "middle_device")
    large_device = _resolve_device(cfg, "large_device")
    small_path = _resolve_local_snapshot(cfg["ms_name"])
    middle_path = _resolve_local_snapshot(cfg["mm_name"]) if need_middle else None
    large_path = _resolve_local_snapshot(cfg["ml_name"])

    print(f"Loading small model: {cfg['ms_name']} -> {small_path}")
    ms_hf = AutoModelForCausalLM.from_pretrained(
        small_path,
        dtype=dtype,
        device_map=_device_map(small_device),
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    m_s = CountedCSDraftingDecoderModelKVCache(
        ms_hf,
        name=cfg["ms_name"],
        vocab_size=ms_hf.config.vocab_size,
    )

    tokenizer_path = _select_tokenizer_path(small_path, large_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=local_files_only,
    )
    tokenizer.pad_token = tokenizer.eos_token

    m_m = None
    if need_middle:
        print(f"Loading middle model: {cfg['mm_name']} -> {middle_path}")
        mm_hf = AutoModelForCausalLM.from_pretrained(
            middle_path,
            dtype=dtype,
            device_map=_device_map(middle_device),
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        m_m = ACSDMiddleTierModel(
            mm_hf,
            name=cfg["mm_name"],
            vocab_size=mm_hf.config.vocab_size,
        )

    print(f"Loading large model: {cfg['ml_name']} -> {large_path}")
    ml_hf = AutoModelForCausalLM.from_pretrained(
        large_path,
        dtype=dtype,
        device_map=_device_map(large_device),
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    m_l = CountedCSDraftingDecoderModelKVCache(
        ml_hf,
        name=cfg["ml_name"],
        vocab_size=ml_hf.config.vocab_size,
    )
    _harmonize_model_vocab_sizes([m_s, m_m, m_l])
    m_l.assume_prefix_cache_match = True
    return m_s, m_m, m_l, tokenizer


def _compact_sample(sample):
    return {
        "sample_index": sample["sample_index"],
        "wall_time": sample["wall_time"],
        "tokens_generated": sample["tokens_generated"],
        "benchmark_score": sample["score"]["score"],
        "correct": sample["score"]["correct"],
        "prediction": sample["score"]["prediction"],
        "gold": sample["score"]["gold"],
        "usage": sample["usage"],
        "model_runtime": sample.get("model_runtime", {}),
        "ml_forward_calls": sample.get("ml_forward_calls", 0),
        "mm_forward_calls": sample.get("mm_forward_calls", 0),
    }


def select_eval_items(test_set, cfg):
    if cfg["num_shards"] < 1:
        raise ValueError("num_shards must be >= 1")
    if cfg["shard_index"] < 0 or cfg["shard_index"] >= cfg["num_shards"]:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    n_samples = cfg.get("n_samples")
    if n_samples is None or int(n_samples) <= 0:
        requested = test_set
    else:
        requested = test_set[: int(n_samples)]
    if cfg["num_shards"] == 1:
        return list(enumerate(requested))
    return [
        (sample_index, item)
        for sample_index, item in enumerate(requested)
        if sample_index % cfg["num_shards"] == cfg["shard_index"]
    ]


def _runtime_totals_from_samples(sample_results):
    totals = {
        model_key: {
            "propose_calls": 0,
            "review_calls": 0,
            "propose_wall_time": 0.0,
            "review_wall_time": 0.0,
            "total_wall_time": 0.0,
        }
        for model_key in ("small", "middle", "large")
    }
    for sample in sample_results:
        for model_key, runtime in sample.get("model_runtime", {}).items():
            bucket = totals[model_key]
            for field in bucket:
                bucket[field] += runtime.get(field, 0)
    return totals


def _merge_stats_from_samples(sample_results):
    draft_window_totals = {
        "small": {"sum": 0, "count": 0},
        "middle": {"sum": 0, "count": 0},
    }
    window_change_count = 0
    routing_totals = {"middle": 0, "large": 0}
    routing_change_count = 0
    routing_seen = False

    for sample in sample_results:
        draft_window = sample.get("draft_window") or {}
        window_change_count += draft_window.get("change_count", 0)
        for model_key, totals in (draft_window.get("totals") or {}).items():
            if model_key not in draft_window_totals:
                continue
            draft_window_totals[model_key]["sum"] += totals.get("sum", 0)
            draft_window_totals[model_key]["count"] += totals.get("count", 0)

        routing = sample.get("routing") or {}
        if routing:
            routing_seen = True
            routing_change_count += routing.get("change_count", 0)
            for route_key, value in (routing.get("counts") or {}).items():
                if route_key in routing_totals:
                    routing_totals[route_key] += value

    return {
        "n_samples": len(sample_results),
        "total_tokens": sum(sample["tokens_generated"] for sample in sample_results),
        "total_wall_time": sum(sample["wall_time"] for sample in sample_results),
        "total_score": sum(sample["score"]["score"] for sample in sample_results),
        "benchmark_metric": (
            sample_results[0]["score"]["metric_name"] if sample_results else "score"
        ),
        "ml_forward_call_sum": sum(
            sample.get("ml_forward_calls", 0) for sample in sample_results
        ),
        "mm_forward_call_sum": sum(
            sample.get("mm_forward_calls", 0) for sample in sample_results
        ),
        "runtime_totals": _runtime_totals_from_samples(sample_results),
        "draft_window_totals": draft_window_totals,
        "window_change_count": window_change_count,
        "routing_totals": routing_totals,
        "routing_change_count": routing_change_count,
        "routing_seen": routing_seen,
    }


def run_eval(run_name, cfg, test_items, tokenizer, m_s, m_m, m_l):
    sample_results = []
    iterator = tqdm(test_items, desc=run_name)
    for sample_index, item in iterator:
        prompt = format_initial_input(item, cfg["dataset"])
        initial_input = tokenizer(
            prompt, truncation=True, padding=False, return_tensors="pt"
        )["input_ids"].to(m_l.device)
        try:
            if run_name == "baseline":
                sample = run_baseline_sample(
                    cfg=cfg,
                    item=item,
                    initial_input=initial_input,
                    tokenizer=tokenizer,
                    m_s=m_s,
                    m_l=m_l,
                    sample_index=sample_index,
                    capture_trace=len(sample_results) < cfg["trace_samples"],
                )
            else:
                if cfg.get("hierarchical_variant", "window_fill") == "window_fill":
                    sample = run_hierarchical_sample(
                        cfg=cfg,
                        item=item,
                        initial_input=initial_input,
                        tokenizer=tokenizer,
                        m_s=m_s,
                        m_m=m_m,
                        m_l=m_l,
                        sample_index=sample_index,
                        capture_trace=len(sample_results) < cfg["trace_samples"],
                    )
                else:
                    sample = run_double_layer_sample(
                        cfg=cfg,
                        item=item,
                        initial_input=initial_input,
                        tokenizer=tokenizer,
                        m_s=m_s,
                        m_m=m_m,
                        m_l=m_l,
                        sample_index=sample_index,
                        capture_trace=len(sample_results) < cfg["trace_samples"],
                    )
        except Exception as exc:
            raise RuntimeError(
                f"{run_name} failed for sample_index={sample_index} on dataset={cfg['dataset']}"
            ) from exc
        sample_results.append(sample)

    summary, aggregate = summarize_run(sample_results, run_name, cfg)
    merge_stats = _merge_stats_from_samples(sample_results)
    print(f"\n{run_name} summary")
    pprint(summary)
    return {
        "summary": summary,
        "aggregate_counters": aggregate,
        "merge_stats": merge_stats,
        "sample_metrics": [_compact_sample(sample) for sample in sample_results],
        "trace_samples": sample_results[: cfg["trace_samples"]],
    }


def serialize_config(cfg):
    out = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def build_result(cfg, test_set, tokenizer, m_s, m_m, m_l):
    selected_items = select_eval_items(test_set, cfg)
    requested_n_samples = cfg.get("n_samples")
    if requested_n_samples is None or int(requested_n_samples) <= 0:
        requested_n_samples = len(test_set)
    result = {
        "config": serialize_config(cfg),
        "selection": {
            "requested_n_samples": requested_n_samples,
            "selected_n_samples": len(selected_items),
            "num_shards": cfg["num_shards"],
            "shard_index": cfg["shard_index"],
            "selected_sample_indices": [sample_index for sample_index, _ in selected_items],
        },
        "runs": {},
    }

    if cfg["mode"] in ("baseline", "compare"):
        result["runs"]["baseline"] = run_eval(
            "baseline", cfg, selected_items, tokenizer, m_s, m_m, m_l
        )
    if cfg["mode"] in ("hierarchical", "compare"):
        if m_m is None:
            raise RuntimeError("Middle model is required for hierarchical mode.")
        result["runs"]["hierarchical"] = run_eval(
            "hierarchical", cfg, selected_items, tokenizer, m_s, m_m, m_l
        )

    if cfg["mode"] == "compare":
        result["comparison"] = comparison_summary(
            result["runs"]["hierarchical"]["summary"],
            result["runs"]["baseline"]["summary"],
        )

    return result


def save_result(result, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    args = parse_args()
    for key in (
        "mode",
        "dataset",
        "ms_name",
        "mm_name",
        "ml_name",
        "small_device",
        "middle_device",
        "large_device",
        "n_samples",
        "num_shards",
        "shard_index",
        "small_window",
        "middle_window",
        "hierarchical_variant",
        "proxy_threshold",
        "mavg_window",
        "window_policy",
        "adapt_small_window",
        "dynamic_small_window_min",
        "dynamic_small_window_max",
        "dynamic_middle_window_min",
        "dynamic_middle_window_max",
        "dynamic_acceptance_low",
        "dynamic_acceptance_high",
        "dynamic_window_step",
        "dynamic_utility_margin",
        "selective_route_warmup_blocks",
        "selective_route_history_window",
        "selective_route_utility_margin",
        "selective_route_direct_acceptance_low",
        "selective_route_direct_acceptance_high",
        "selective_route_middle_acceptance_low",
        "selective_route_probe_interval",
        "max_length",
        "trace_samples",
        "disable_stop_on_answer",
        "device",
        "output",
    ):
        config[key] = getattr(args, key)
    config["stop_on_answer"] = not config.pop("disable_stop_on_answer")

    print("Config:")
    pprint(config)

    need_middle = config["mode"] in ("hierarchical", "compare")
    m_s, m_m, m_l, tokenizer = load_models(config, need_middle=need_middle)
    test_set = get_test_set(config["dataset"])
    result = build_result(config, test_set, tokenizer, m_s, m_m, m_l)
    save_result(result, config["output"])

    print(f"\nSaved results to {config['output']}")
