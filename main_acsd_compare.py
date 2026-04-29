"""
ACSD evaluation runner with UI-oriented traces and baseline comparison.

This runner is meant to evaluate the branch currently under test, emit
token-level provenance for traced samples, and aggregate run-level metrics that
make the middle tier and adaptive switching behavior inspectable in the Flask UI.
"""

import argparse
import copy
import glob
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from types import MethodType
from pprint import pprint

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import score_sample


CURRENT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BRANCH_ROOT = CURRENT_ROOT
HF_MODELS = "/mnt/data/xuandong/hf_models"
MODEL_KEYS = ("small", "middle", "large")
DRAFT_WINDOW_MODEL_KEYS = ("small", "middle")
EDGE_KEYS = ("small_to_middle", "middle_to_large", "small_to_large")
COUNTER_KEYS = (
    "draft_generated_counts",
    "final_source_counts",
    "verification_positions",
    "verification_calls",
)

DEFAULT_CONFIG = {
    "branch_root": DEFAULT_BRANCH_ROOT,
    "mode": "compare_all",
    "dataset": "mmlu",
    "ms_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "mm_name": "Qwen/Qwen2.5-7B-Instruct",
    "ml_name": "Qwen/Qwen2.5-14B-Instruct",
    "small_device": "cuda:0",
    "middle_device": "cuda:0",
    "large_device": "cuda:1",
    "device": "cuda:0",
    "n_samples": 100,
    "num_shards": 1,
    "shard_index": 0,
    "k_s": 5,
    "k_m": 4,
    "leniency": 1,
    "tau": 0.4,
    "window_size": 20,
    "max_length": 200,
    "trace_samples": 5,
    "middle_refill": True,
    "shadow_middle": False,
    "draft_window_policy": "fixed",
    "dynamic_k_s_min": 2,
    "dynamic_k_s_max": 8,
    "dynamic_k_m_min": 2,
    "dynamic_k_m_max": 8,
    "dynamic_acceptance_low": 0.5,
    "dynamic_acceptance_high": 0.8,
    "dynamic_window_step": 1,
    "dtype": torch.float16,
    "output": "results/acsd_compare.json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "cascaded",
            "adaptive",
            "compare",
            "compare_all",
        ],
        default=DEFAULT_CONFIG["mode"],
    )
    parser.add_argument("--branch_root", type=str, default=DEFAULT_CONFIG["branch_root"])
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k"], default=DEFAULT_CONFIG["dataset"])
    parser.add_argument("--ms_name", type=str, default=DEFAULT_CONFIG["ms_name"])
    parser.add_argument("--mm_name", type=str, default=DEFAULT_CONFIG["mm_name"])
    parser.add_argument("--ml_name", type=str, default=DEFAULT_CONFIG["ml_name"])
    parser.add_argument("--small_device", type=str, default=DEFAULT_CONFIG["small_device"])
    parser.add_argument("--middle_device", type=str, default=DEFAULT_CONFIG["middle_device"])
    parser.add_argument("--large_device", type=str, default=DEFAULT_CONFIG["large_device"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--n_samples", type=int, default=DEFAULT_CONFIG["n_samples"])
    parser.add_argument("--num_shards", type=int, default=DEFAULT_CONFIG["num_shards"])
    parser.add_argument("--shard_index", type=int, default=DEFAULT_CONFIG["shard_index"])
    parser.add_argument("--k_s", type=int, default=DEFAULT_CONFIG["k_s"])
    parser.add_argument("--k_m", type=int, default=DEFAULT_CONFIG["k_m"])
    parser.add_argument("--leniency", type=int, default=DEFAULT_CONFIG["leniency"])
    parser.add_argument("--tau", type=float, default=DEFAULT_CONFIG["tau"])
    parser.add_argument("--window_size", type=int, default=DEFAULT_CONFIG["window_size"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--trace_samples", type=int, default=DEFAULT_CONFIG["trace_samples"])
    parser.add_argument(
        "--middle_refill",
        dest="middle_refill",
        action="store_true",
        default=DEFAULT_CONFIG["middle_refill"],
        help="Allow the middle model to append its replacement token after rejecting a small draft.",
    )
    parser.add_argument(
        "--no_middle_refill",
        dest="middle_refill",
        action="store_false",
        help="Use the middle model as a filter only; large model performs the refill.",
    )
    parser.add_argument(
        "--shadow_middle",
        action="store_true",
        default=DEFAULT_CONFIG["shadow_middle"],
        help="Run middle verification for metrics but send the original small draft to the large model.",
    )
    parser.add_argument(
        "--draft_window_policy",
        choices=["fixed", "acceptance"],
        default=DEFAULT_CONFIG["draft_window_policy"],
        help="Use fixed k_s/k_m, or adapt draft windows from recent acceptance.",
    )
    parser.add_argument("--dynamic_k_s_min", type=int, default=DEFAULT_CONFIG["dynamic_k_s_min"])
    parser.add_argument("--dynamic_k_s_max", type=int, default=DEFAULT_CONFIG["dynamic_k_s_max"])
    parser.add_argument("--dynamic_k_m_min", type=int, default=DEFAULT_CONFIG["dynamic_k_m_min"])
    parser.add_argument("--dynamic_k_m_max", type=int, default=DEFAULT_CONFIG["dynamic_k_m_max"])
    parser.add_argument("--dynamic_acceptance_low", type=float, default=DEFAULT_CONFIG["dynamic_acceptance_low"])
    parser.add_argument("--dynamic_acceptance_high", type=float, default=DEFAULT_CONFIG["dynamic_acceptance_high"])
    parser.add_argument("--dynamic_window_step", type=int, default=DEFAULT_CONFIG["dynamic_window_step"])
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG["output"])
    return parser.parse_args()


def _resolve_device(cfg, key):
    return cfg.get(key) or cfg["device"]


def _device_map(device):
    return {"": device} if device.startswith("cuda") else {"": "cpu"}


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

        def compat_review(self, initial_input, input_ids, probs, review_index, leniency=1, _orig=original_review):
            review_device = getattr(getattr(self, "model", None), "device", getattr(self, "device", None))
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

    if os.path.isdir(model_name):
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
        HF_MODELS,
    ):
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    def latest_snapshot(repo_cache):
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
            snapshot = latest_snapshot(os.path.join(root, repo_key))
            if snapshot:
                return snapshot
        if attempt < 2 and candidate_roots:
            time.sleep(1)

    return repo_id


def load_branch_modules(branch_root):
    branch_root = os.path.abspath(branch_root)
    if not os.path.isdir(branch_root):
        raise FileNotFoundError(f"Branch root does not exist: {branch_root}")

    modules = {}
    module_paths = {
        "acsd": os.path.join(branch_root, "acsd.py"),
        "csd": os.path.join(branch_root, "csd.py"),
        "model": os.path.join(CURRENT_ROOT, "model.py"),
        "csd_datasets": os.path.join(CURRENT_ROOT, "csd_datasets.py"),
    }
    for name, module_path in module_paths.items():
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module {name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def load_models(cfg, modules, run_names):
    model_mod = modules["model"]
    need_middle = any(name in ("cascaded", "adaptive") for name in run_names)
    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    dtype = cfg["dtype"]

    small_path = _resolve_local_snapshot(cfg["ms_name"])
    middle_path = _resolve_local_snapshot(cfg["mm_name"]) if need_middle else None
    large_path = _resolve_local_snapshot(cfg["ml_name"])

    small_device = _resolve_device(cfg, "small_device")
    middle_device = _resolve_device(cfg, "middle_device")
    large_device = _resolve_device(cfg, "large_device")

    print(f"Loading M_s: {cfg['ms_name']} -> {small_path}")
    ms_hf = AutoModelForCausalLM.from_pretrained(
        small_path,
        torch_dtype=dtype,
        device_map=_device_map(small_device),
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    m_s = model_mod.CountedCSDraftingDecoderModel(
        ms_hf,
        name=cfg["ms_name"],
        vocab_size=ms_hf.config.vocab_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        small_path,
        local_files_only=local_files_only,
    )
    tokenizer.pad_token = tokenizer.eos_token

    m_m = None
    if need_middle:
        print(f"Loading M_m: {cfg['mm_name']} -> {middle_path}")
        mm_hf = AutoModelForCausalLM.from_pretrained(
            middle_path,
            torch_dtype=dtype,
            device_map=_device_map(middle_device),
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        m_m = model_mod.ACSDMiddleTierModel(
            mm_hf,
            name=cfg["mm_name"],
            vocab_size=mm_hf.config.vocab_size,
        )

    print(f"Loading M_l: {cfg['ml_name']} -> {large_path}")
    ml_hf = AutoModelForCausalLM.from_pretrained(
        large_path,
        torch_dtype=dtype,
        device_map=_device_map(large_device),
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    m_l = model_mod.CountedCSDraftingDecoderModelKVCache(
        ml_hf,
        name=cfg["ml_name"],
        vocab_size=ml_hf.config.vocab_size,
    )
    _harmonize_model_vocab_sizes([m_s, m_m, m_l])

    return tokenizer, m_s, m_m, m_l


def select_eval_items(test_set, cfg):
    if cfg["num_shards"] < 1:
        raise ValueError("num_shards must be >= 1")
    if cfg["shard_index"] < 0 or cfg["shard_index"] >= cfg["num_shards"]:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    requested = test_set[: cfg["n_samples"]]
    if cfg["num_shards"] == 1:
        return list(enumerate(requested))
    return [
        (sample_index, item)
        for sample_index, item in enumerate(requested)
        if sample_index % cfg["num_shards"] == cfg["shard_index"]
    ]


def reset_model_state(model):
    if model is None:
        return
    if hasattr(model, "past_key_values"):
        model.past_key_values = None
    if hasattr(model, "past_ids"):
        model.past_ids = None
    if hasattr(model, "forward_count"):
        model.forward_count = 0
    if hasattr(model, "propose_count"):
        model.propose_count = 0
    if hasattr(model, "review_count"):
        model.review_count = 0
    if hasattr(model, "propose_wall_time"):
        model.propose_wall_time = []
    if hasattr(model, "review_wall_time"):
        model.review_wall_time = []
    if hasattr(model, "wall_time"):
        model.wall_time = []
    if hasattr(model, "acceptance_history"):
        model.acceptance_history = []
    if hasattr(model, "saved_ml_positions"):
        model.saved_ml_positions = 0


def model_runtime_snapshot(model):
    if model is None:
        return {
            "propose_calls": 0,
            "review_calls": 0,
            "propose_wall_time": 0.0,
            "review_wall_time": 0.0,
            "total_wall_time": 0.0,
        }

    propose_wall = float(sum(getattr(model, "propose_wall_time", []) or []))
    review_wall = float(sum(getattr(model, "review_wall_time", []) or []))
    review_calls = int(
        getattr(model, "review_count", getattr(model, "forward_count", 0))
    )
    return {
        "propose_calls": int(getattr(model, "propose_count", 0)),
        "review_calls": review_calls,
        "propose_wall_time": propose_wall,
        "review_wall_time": review_wall,
        "total_wall_time": propose_wall + review_wall,
    }


def _eos_token(model):
    return 2 if "t5" not in model.name.lower() else 1


def _make_one_hot_probs(token_ids, vocab_size, device):
    return (
        torch.nn.functional.one_hot(
            token_ids.squeeze(0)[1:], num_classes=vocab_size
        )
        .float()
        .unsqueeze(0)
        .to(device)
    )


def _first_unreviewable_index(candidate_records, verifier):
    native_vocab_size = int(
        getattr(verifier, "native_vocab_size", getattr(verifier, "vocab_size", 0))
    )
    if native_vocab_size <= 0:
        return None
    for index, record in enumerate(candidate_records):
        token_id = int(record["token_id"])
        if token_id < 0 or token_id >= native_vocab_size:
            return index
    return None


def _new_counter():
    return {key: 0 for key in MODEL_KEYS}


def _new_edge_counter():
    return {key: {"accepted": 0, "proposed": 0} for key in EDGE_KEYS}


def new_run_stats():
    return {
        "draft_generated_counts": _new_counter(),
        "final_source_counts": _new_counter(),
        "verification_positions": _new_counter(),
        "verification_calls": _new_counter(),
        "edge_pass": _new_edge_counter(),
    }


def _per_sample_usage(stats, total_tokens):
    usage = {}
    for bucket_name in (
        "draft_generated_counts",
        "final_source_counts",
        "verification_positions",
        "verification_calls",
    ):
        usage[bucket_name] = {}
        bucket_total = sum(stats[bucket_name].values())
        for model_key in MODEL_KEYS:
            count = stats[bucket_name][model_key]
            usage[bucket_name][model_key] = {
                "count": count,
                "pct": (count / bucket_total) if bucket_total else 0.0,
            }

    usage["edge_pass_rates"] = {}
    for edge_key in EDGE_KEYS:
        proposed = stats["edge_pass"][edge_key]["proposed"]
        usage["edge_pass_rates"][edge_key] = (
            stats["edge_pass"][edge_key]["accepted"] / proposed if proposed else None
        )
    usage["total_tokens"] = total_tokens
    return usage


def _runtime_totals_from_samples(sample_results):
    totals = {
        model_key: {
            "propose_calls": 0,
            "review_calls": 0,
            "propose_wall_time": 0.0,
            "review_wall_time": 0.0,
            "total_wall_time": 0.0,
        }
        for model_key in MODEL_KEYS
    }
    for sample in sample_results:
        for model_key, runtime in sample.get("model_runtime", {}).items():
            bucket = totals[model_key]
            for field in bucket:
                bucket[field] += runtime.get(field, 0)
    return totals


def _aggregate_counters_from_samples(sample_results):
    aggregate = new_run_stats()
    for sample in sample_results:
        raw_counters = sample.get("raw_counters") or new_run_stats()
        for bucket_name in COUNTER_KEYS:
            for model_key, value in raw_counters[bucket_name].items():
                aggregate[bucket_name][model_key] += value
        for edge_key, counters in raw_counters["edge_pass"].items():
            aggregate["edge_pass"][edge_key]["accepted"] += counters["accepted"]
            aggregate["edge_pass"][edge_key]["proposed"] += counters["proposed"]
    return aggregate


def _merge_stats_from_samples(sample_results):
    draft_window_totals, draft_window_change_count_sum = _draft_window_totals_from_samples(
        sample_results
    )
    draft_window_policy = (
        sample_results[0].get("draft_window", {}).get("policy")
        if sample_results
        else None
    )
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
        "switch_count_sum": sum(
            sample.get("switch_count", 0) for sample in sample_results
        ),
        "mm_saved_positions_sum": sum(
            sample.get("mm_saved_positions", 0) for sample in sample_results
        ),
        "drafter_step_sums": {
            "small": sum(
                sample.get("drafter_steps", {}).get("small", 0)
                for sample in sample_results
            ),
            "middle": sum(
                sample.get("drafter_steps", {}).get("middle", 0)
                for sample in sample_results
            ),
        },
        "draft_window_policy": draft_window_policy,
        "draft_window_totals": draft_window_totals,
        "draft_window_change_count_sum": draft_window_change_count_sum,
        "runtime_totals": _runtime_totals_from_samples(sample_results),
    }


def _draft_window_totals_from_samples(sample_results):
    totals = {
        model_key: {"sum": 0.0, "count": 0}
        for model_key in DRAFT_WINDOW_MODEL_KEYS
    }
    change_count_sum = 0
    for sample in sample_results:
        draft_window = sample.get("draft_window") or {}
        change_count_sum += draft_window.get("change_count", 0)
        for model_key in DRAFT_WINDOW_MODEL_KEYS:
            bucket = (draft_window.get("totals") or {}).get(model_key, {})
            totals[model_key]["sum"] += bucket.get("sum", 0.0)
            totals[model_key]["count"] += bucket.get("count", 0)
    return totals, change_count_sum


def _avg_draft_windows(draft_window_totals):
    return {
        model_key: (
            draft_window_totals.get(model_key, {}).get("sum", 0.0)
            / draft_window_totals.get(model_key, {}).get("count", 0)
            if draft_window_totals.get(model_key, {}).get("count", 0)
            else 0.0
        )
        for model_key in DRAFT_WINDOW_MODEL_KEYS
    }


def decode_token(tokenizer, token_id):
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def ids_from_records(records, device):
    if not records:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return torch.tensor(
        [[int(record["token_id"]) for record in records]],
        dtype=torch.long,
        device=device,
    )


def make_records(token_ids, tokenizer, source_model, positions, step_index):
    return [
        {
            "position": int(position),
            "token_id": int(token_id),
            "token_text": decode_token(tokenizer, int(token_id)),
            "source_model": source_model,
            "verified_by_middle": False,
            "verified_by_large": False,
            "step_index": step_index,
        }
        for position, token_id in zip(positions, token_ids)
    ]


def _clone_record(record):
    return copy.deepcopy(record)


def runtime_snapshots(m_s, m_m, m_l):
    return {
        "small": model_runtime_snapshot(m_s),
        "middle": model_runtime_snapshot(m_m),
        "large": model_runtime_snapshot(m_l),
    }


def runtime_delta(before, after):
    delta = {}
    for model_key in MODEL_KEYS:
        delta[model_key] = {}
        before_bucket = before.get(model_key, {})
        after_bucket = after.get(model_key, {})
        for field in (
            "propose_calls",
            "review_calls",
            "propose_wall_time",
            "review_wall_time",
            "total_wall_time",
        ):
            delta[model_key][field] = after_bucket.get(field, 0) - before_bucket.get(
                field, 0
            )
    return delta


def review_candidate_records(
    verifier,
    verifier_key,
    edge_key,
    initial_input,
    prefix_ids,
    candidate_records,
    tokenizer,
    stats,
    step_index,
    probs=None,
    leniency=1,
    allow_refill=True,
    empty_refill_position=None,
):
    candidate_len = len(candidate_records)
    if candidate_len == 0:
        generated_tokens = []
        if allow_refill:
            verifier_device = verifier.device
            review_index = prefix_ids.shape[-1]
            proposed_ids = verifier.propose(
                initial_input.to(verifier_device),
                prefix_ids.to(verifier_device),
                1,
            )
            generated_tokens = proposed_ids[:, review_index:].detach().cpu().tolist()[0]
            first_position = (
                empty_refill_position
                if empty_refill_position is not None
                else prefix_ids.shape[-1]
            )
            positions = [
                first_position + idx
                for idx in range(len(generated_tokens))
            ]
            result_records = make_records(
                generated_tokens,
                tokenizer=tokenizer,
                source_model=verifier_key,
                positions=positions,
                step_index=step_index,
            )
            stats["draft_generated_counts"][verifier_key] += len(generated_tokens)
            return result_records, proposed_ids, None, {
                "accepted_count": 0,
                "generated_count": len(generated_tokens),
                "candidate_len": 0,
            }
        return [], None, None, {
            "accepted_count": 0,
            "generated_count": 0,
            "candidate_len": 0,
        }

    verifier_device = verifier.device
    review_index = prefix_ids.shape[-1]
    stats["verification_calls"][verifier_key] += 1
    stats["verification_positions"][verifier_key] += candidate_len
    stats["edge_pass"][edge_key]["proposed"] += candidate_len

    cutoff = _first_unreviewable_index(candidate_records, verifier)
    if cutoff == 0:
        if not allow_refill:
            return [], prefix_ids.to(verifier_device), None, {
                "accepted_count": 0,
                "generated_count": 0,
                "candidate_len": candidate_len,
            }
        proposed_ids = verifier.propose(
            initial_input.to(verifier_device),
            prefix_ids.to(verifier_device),
            1,
        )
        generated_tokens = proposed_ids[:, review_index:].detach().cpu().tolist()[0]
        positions = [candidate_records[0]["position"] + idx for idx in range(len(generated_tokens))]
        result_records = make_records(
            generated_tokens,
            tokenizer=tokenizer,
            source_model=verifier_key,
            positions=positions,
            step_index=step_index,
        )
        stats["draft_generated_counts"][verifier_key] += len(generated_tokens)
        return result_records, None, None, {
            "accepted_count": 0,
            "generated_count": len(generated_tokens),
            "candidate_len": candidate_len,
        }

    reviewable_records = candidate_records if cutoff is None else candidate_records[:cutoff]
    candidate_ids = ids_from_records(reviewable_records, verifier_device)
    review_ids = torch.cat([prefix_ids.to(verifier_device), candidate_ids], dim=1)

    reviewed_ids, reviewed_probs = verifier.review(
        initial_input.to(verifier_device),
        review_ids,
        _move_tensor_to_device(probs, verifier_device),
        review_index,
        leniency=leniency,
    )
    reviewed_ext = reviewed_ids[:, review_index:]
    reviewable_len = len(reviewable_records)
    accepted_count = min(reviewable_len, max(0, reviewed_ext.shape[-1] - 1))
    stats["edge_pass"][edge_key]["accepted"] += accepted_count

    result_records = []
    for record in candidate_records[:accepted_count]:
        accepted_record = _clone_record(record)
        if verifier_key == "middle":
            accepted_record["verified_by_middle"] = True
        if verifier_key == "large":
            accepted_record["verified_by_large"] = True
        result_records.append(accepted_record)

    generated_tokens = []
    if allow_refill and reviewed_ext.shape[-1] > accepted_count:
        generated_tokens = reviewed_ext[0, accepted_count:].detach().cpu().tolist()
        if accepted_count < candidate_len:
            first_position = candidate_records[accepted_count]["position"]
        else:
            first_position = candidate_records[-1]["position"] + 1
        positions = [first_position + idx for idx in range(len(generated_tokens))]
        result_records.extend(
            make_records(
                generated_tokens,
                tokenizer=tokenizer,
                source_model=verifier_key,
                positions=positions,
                step_index=step_index,
            )
        )
        stats["draft_generated_counts"][verifier_key] += len(generated_tokens)

    if not allow_refill:
        accepted_ids = ids_from_records(result_records, verifier_device)
        reviewed_ids = torch.cat([prefix_ids.to(verifier_device), accepted_ids], dim=1)
        reviewed_probs = None

    return result_records, reviewed_ids, reviewed_probs, {
        "accepted_count": accepted_count,
        "generated_count": len(generated_tokens),
        "candidate_len": candidate_len,
    }


def finalize_sample_payload(
    sample_index,
    prompt,
    item,
    wall_time,
    initial_input,
    output_ids,
    tokenizer,
    m_s,
    m_m,
    m_l,
    stats,
    final_records,
    capture_trace,
    steps=None,
    switch_count=0,
    alpha_trace=None,
    acceptance_trace=None,
    drafter_steps=None,
    draft_window=None,
):
    generated_ids = output_ids[0, initial_input.shape[-1] :].detach().cpu().tolist()
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    score = score_sample(item["dataset_name"], item["raw_item"], generated_text)

    for record in final_records:
        stats["final_source_counts"][record["source_model"]] += 1

    sample_usage = _per_sample_usage(stats, len(generated_ids))
    sample = {
        "sample_index": sample_index,
        "prompt": prompt,
        "generated_text": generated_text,
        "score": score,
        "gold_answer": item["raw_item"].get("answer"),
        "wall_time": wall_time,
        "tokens_generated": len(generated_ids),
        "usage": sample_usage,
        "raw_counters": stats,
        "ml_forward_calls": int(getattr(m_l, "forward_count", 0)),
        "mm_forward_calls": int(getattr(m_m, "forward_count", 0)) if m_m else 0,
        "mm_saved_positions": (
            stats["edge_pass"]["small_to_middle"]["proposed"]
            - stats["edge_pass"]["small_to_middle"]["accepted"]
        ),
        "switch_count": switch_count,
        "alpha_trace": alpha_trace or [],
        "acceptance_trace": acceptance_trace or [],
        "drafter_steps": drafter_steps or {"small": 0, "middle": 0},
        "draft_window": draft_window or {},
        "model_runtime": {
            "small": model_runtime_snapshot(m_s),
            "middle": model_runtime_snapshot(m_m),
            "large": model_runtime_snapshot(m_l),
        },
    }
    if capture_trace:
        sample["final_tokens"] = final_records
        sample["steps"] = steps or []
    return sample


def serialize_config(cfg):
    out = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def sample_metric_payload(sample):
    return {
        "sample_index": sample["sample_index"],
        "wall_time": sample["wall_time"],
        "tokens_generated": sample["tokens_generated"],
        "benchmark_score": sample["score"]["score"],
        "correct": sample["score"]["correct"],
        "prediction": sample["score"]["prediction"],
        "gold": sample["score"]["gold"],
        "usage": sample["usage"],
        "ml_forward_calls": sample["ml_forward_calls"],
        "mm_forward_calls": sample["mm_forward_calls"],
        "mm_saved_positions": sample["mm_saved_positions"],
        "switch_count": sample["switch_count"],
        "drafter_steps": sample["drafter_steps"],
        "alpha_trace": sample["alpha_trace"],
        "draft_window": sample.get("draft_window", {}),
        "model_runtime": sample["model_runtime"],
    }


def _init_draft_window_state(cfg):
    return {
        "current": {
            "small": int(cfg["k_s"]),
            "middle": int(cfg.get("k_m", cfg["k_s"])),
        },
        "totals": {
            "small": {"sum": 0, "count": 0},
            "middle": {"sum": 0, "count": 0},
        },
        "trace": [],
        "change_count": 0,
    }


def _draft_window_bounds(cfg, model_key):
    if model_key == "middle":
        min_k = int(cfg["dynamic_k_m_min"])
        max_k = int(cfg["dynamic_k_m_max"])
    else:
        min_k = int(cfg["dynamic_k_s_min"])
        max_k = int(cfg["dynamic_k_s_max"])
    min_k = max(1, min_k)
    max_k = max(min_k, max_k)
    return min_k, max_k


def _draft_window_request(cfg, window_state, model_key, remaining, dynamic_enabled=True):
    if not dynamic_enabled or cfg["draft_window_policy"] == "fixed":
        base = cfg["k_m"] if model_key == "middle" else cfg["k_s"]
    else:
        min_k, max_k = _draft_window_bounds(cfg, model_key)
        window_state["current"][model_key] = max(
            min_k,
            min(int(window_state["current"][model_key]), max_k),
        )
        base = window_state["current"][model_key]
    requested = max(1, min(int(base), int(remaining)))
    totals = window_state["totals"][model_key]
    totals["sum"] += requested
    totals["count"] += 1
    return requested


def _update_draft_window(cfg, window_state, model_key, acceptance_ratio, step_index):
    if window_state.get("policy") != "acceptance":
        return

    low, high = float(cfg["dynamic_acceptance_low"]), float(cfg["dynamic_acceptance_high"])
    step = max(1, int(cfg["dynamic_window_step"]))
    min_k, max_k = _draft_window_bounds(cfg, model_key)
    previous = max(min_k, min(int(window_state["current"][model_key]), max_k))
    updated = previous
    if acceptance_ratio >= high:
        updated = min(max_k, previous + step)
    elif acceptance_ratio <= low:
        updated = max(min_k, previous - step)
    window_state["current"][model_key] = updated
    if updated != previous:
        window_state["change_count"] += 1
    window_state["trace"].append(
        {
            "step_index": step_index,
            "model": model_key,
            "acceptance_ratio": acceptance_ratio,
            "previous_window": previous,
            "next_window": updated,
            "changed": updated != previous,
        }
    )


def _draft_window_payload(window_state):
    averages = {}
    for model_key, totals in window_state["totals"].items():
        count = totals["count"]
        averages[model_key] = (totals["sum"] / count) if count else 0.0
    return {
        "policy": window_state.get("policy", "fixed"),
        "averages": averages,
        "totals": window_state["totals"],
        "change_count": window_state["change_count"],
        "trace": window_state["trace"],
    }


def run_sample(
    run_name,
    cfg,
    modules,
    tokenizer,
    m_s,
    m_m,
    m_l,
    item,
    sample_index,
    capture_trace=False,
):
    reset_model_state(m_s)
    reset_model_state(m_m)
    reset_model_state(m_l)

    dataset_mod = modules["csd_datasets"]
    acsd_mod = modules["acsd"]
    prompt = dataset_mod.format_initial_input(item, cfg["dataset"])
    initial_input = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )["input_ids"].to(m_l.device)
    output_ids = initial_input.clone().to(m_l.device)
    initial_len = output_ids.shape[-1]
    final_records = []
    steps = [] if capture_trace else None
    stats = new_run_stats()
    eos_id = _eos_token(m_l)
    drafter_steps = {"small": 0, "middle": 0}
    alpha_trace = []
    acceptance_trace = []
    switch_count = 0
    draft_window_state = _init_draft_window_state(cfg)
    dynamic_windows_enabled = (
        cfg["draft_window_policy"] == "acceptance"
        and run_name in ("cascaded", "adaptive")
    )
    draft_window_state["policy"] = (
        cfg["draft_window_policy"] if dynamic_windows_enabled else "fixed"
    )
    sample_item = {"dataset_name": cfg["dataset"], "raw_item": item}

    t0 = time.time()
    with torch.no_grad():
        if run_name == "baseline":
            while output_ids.shape[-1] - initial_len < cfg["max_length"]:
                step_index = len(steps) if steps is not None else len(final_records)
                remaining = cfg["max_length"] - (output_ids.shape[-1] - initial_len)
                requested = _draft_window_request(
                    cfg,
                    draft_window_state,
                    "small",
                    remaining,
                    dynamic_enabled=False,
                )
                before_runtime = runtime_snapshots(m_s, m_m, m_l)

                draft_ids = m_s.propose(
                    initial_input.to(m_s.device),
                    output_ids.to(m_s.device),
                    requested,
                )
                drafted_tokens = draft_ids[:, output_ids.shape[-1] :].detach().cpu().tolist()[0]
                stats["draft_generated_counts"]["small"] += len(drafted_tokens)
                drafter_steps["small"] += 1
                positions = range(
                    output_ids.shape[-1] - initial_len,
                    output_ids.shape[-1] - initial_len + len(drafted_tokens),
                )
                small_records = make_records(
                    drafted_tokens,
                    tokenizer=tokenizer,
                    source_model="small",
                    positions=positions,
                    step_index=step_index,
                )
                final_step_records, _, _, large_meta = review_candidate_records(
                    m_l,
                    "large",
                    "small_to_large",
                    initial_input,
                    output_ids,
                    small_records,
                    tokenizer,
                    stats,
                    step_index=step_index,
                    leniency=1,
                )
                after_runtime = runtime_snapshots(m_s, m_m, m_l)
                output_ids = torch.cat(
                    [output_ids.to(m_l.device), ids_from_records(final_step_records, m_l.device)],
                    dim=1,
                )
                final_records.extend(final_step_records)

                if steps is not None:
                    steps.append(
                        {
                            "step_index": step_index,
                            "drafter": "small",
                            "requested_tokens": requested,
                            "small_draft": small_records,
                            "candidate_to_large": small_records,
                            "final_step": final_step_records,
                            "large_accepted_count": large_meta["accepted_count"],
                            "large_generated_count": large_meta["generated_count"],
                            "model_runtime_delta": runtime_delta(before_runtime, after_runtime),
                        }
                    )

                if not final_step_records or any(
                    record["token_id"] == eos_id for record in final_step_records
                ):
                    break

        elif run_name == "cascaded":
            while output_ids.shape[-1] - initial_len < cfg["max_length"]:
                step_index = len(steps) if steps is not None else len(final_records)
                remaining = cfg["max_length"] - (output_ids.shape[-1] - initial_len)
                requested = _draft_window_request(
                    cfg,
                    draft_window_state,
                    "small",
                    remaining,
                    dynamic_enabled=dynamic_windows_enabled,
                )
                before_runtime = runtime_snapshots(m_s, m_m, m_l)

                draft_ids = m_s.propose(
                    initial_input.to(m_s.device),
                    output_ids.to(m_s.device),
                    requested,
                )
                drafted_tokens = draft_ids[:, output_ids.shape[-1] :].detach().cpu().tolist()[0]
                stats["draft_generated_counts"]["small"] += len(drafted_tokens)
                drafter_steps["small"] += 1
                positions = range(
                    output_ids.shape[-1] - initial_len,
                    output_ids.shape[-1] - initial_len + len(drafted_tokens),
                )
                small_records = make_records(
                    drafted_tokens,
                    tokenizer=tokenizer,
                    source_model="small",
                    positions=positions,
                    step_index=step_index,
                )
                ms_probs = _make_one_hot_probs(draft_ids, m_m.vocab_size, m_m.device)
                middle_records, middle_ids, middle_probs, middle_meta = review_candidate_records(
                    m_m,
                    "middle",
                    "small_to_middle",
                    initial_input,
                    output_ids,
                    small_records,
                    tokenizer,
                    stats,
                    step_index=step_index,
                    probs=ms_probs,
                    leniency=cfg["leniency"],
                    allow_refill=cfg["middle_refill"] and not cfg["shadow_middle"],
                )
                candidate_to_large = small_records if cfg["shadow_middle"] else middle_records
                if cfg["shadow_middle"]:
                    probs_to_large = _make_one_hot_probs(
                        draft_ids,
                        m_l.vocab_size,
                        m_l.device,
                    )
                    large_edge_key = "small_to_large"
                else:
                    probs_to_large = middle_probs
                    if probs_to_large is None and middle_ids is not None:
                        probs_to_large = _make_one_hot_probs(
                            middle_ids,
                            m_l.vocab_size,
                            m_l.device,
                        )
                    large_edge_key = "middle_to_large"
                final_step_records, _, _, large_meta = review_candidate_records(
                    m_l,
                    "large",
                    large_edge_key,
                    initial_input,
                    output_ids,
                    candidate_to_large,
                    tokenizer,
                    stats,
                    step_index=step_index,
                    probs=probs_to_large,
                    leniency=1,
                    empty_refill_position=output_ids.shape[-1] - initial_len,
                )
                dynamic_ratio = (
                    large_meta["accepted_count"] / len(small_records)
                    if cfg["shadow_middle"] and small_records
                    else middle_meta["accepted_count"] / len(small_records)
                    if small_records
                    else 1.0
                )
                _update_draft_window(
                    cfg,
                    draft_window_state,
                    "small",
                    dynamic_ratio,
                    step_index,
                )
                after_runtime = runtime_snapshots(m_s, m_m, m_l)
                output_ids = torch.cat(
                    [output_ids.to(m_l.device), ids_from_records(final_step_records, m_l.device)],
                    dim=1,
                )
                final_records.extend(final_step_records)

                if steps is not None:
                    steps.append(
                        {
                            "step_index": step_index,
                            "drafter": "small",
                            "requested_tokens": requested,
                            "small_draft": small_records,
                            "middle_result": middle_records,
                            "candidate_to_large": candidate_to_large,
                            "middle_refill": cfg["middle_refill"],
                            "shadow_middle": cfg["shadow_middle"],
                            "draft_window_policy": draft_window_state["policy"],
                            "draft_window_acceptance_ratio": dynamic_ratio,
                            "final_step": final_step_records,
                            "middle_accepted_count": middle_meta["accepted_count"],
                            "middle_generated_count": middle_meta["generated_count"],
                            "middle_saved_positions": len(small_records)
                            - middle_meta["accepted_count"],
                            "large_accepted_count": large_meta["accepted_count"],
                            "large_generated_count": large_meta["generated_count"],
                            "model_runtime_delta": runtime_delta(before_runtime, after_runtime),
                        }
                    )

                if not final_step_records or any(
                    record["token_id"] == eos_id for record in final_step_records
                ):
                    break

        elif run_name == "adaptive":
            state = acsd_mod.AdaptiveCSDState(
                tau=cfg["tau"],
                window_size=cfg["window_size"],
            )
            mm_as_verifier = True

            while output_ids.shape[-1] - initial_len < cfg["max_length"]:
                step_index = len(steps) if steps is not None else len(final_records)
                remaining = cfg["max_length"] - (output_ids.shape[-1] - initial_len)
                before_runtime = runtime_snapshots(m_s, m_m, m_l)
                drafter_before = state.current_drafter
                alpha_before = state.rolling_alpha

                if drafter_before == "ms":
                    requested = _draft_window_request(
                        cfg,
                        draft_window_state,
                        "small",
                        remaining,
                        dynamic_enabled=dynamic_windows_enabled,
                    )
                    draft_ids = m_s.propose(
                        initial_input.to(m_s.device),
                        output_ids.to(m_s.device),
                        requested,
                    )
                    drafted_tokens = draft_ids[:, output_ids.shape[-1] :].detach().cpu().tolist()[0]
                    stats["draft_generated_counts"]["small"] += len(drafted_tokens)
                    drafter_steps["small"] += 1
                    positions = range(
                        output_ids.shape[-1] - initial_len,
                        output_ids.shape[-1] - initial_len + len(drafted_tokens),
                    )
                    small_records = make_records(
                        drafted_tokens,
                        tokenizer=tokenizer,
                        source_model="small",
                        positions=positions,
                        step_index=step_index,
                    )
                    ms_probs = _make_one_hot_probs(draft_ids, m_m.vocab_size, m_m.device)
                    middle_records, middle_ids, middle_probs, middle_meta = review_candidate_records(
                        m_m,
                        "middle",
                        "small_to_middle",
                        initial_input,
                        output_ids,
                        small_records,
                        tokenizer,
                        stats,
                        step_index=step_index,
                        probs=ms_probs,
                        leniency=cfg["leniency"],
                        allow_refill=cfg["middle_refill"] and not cfg["shadow_middle"],
                    )
                    mm_as_verifier = True
                    candidate_to_large = small_records if cfg["shadow_middle"] else middle_records
                    if cfg["shadow_middle"]:
                        probs_to_large = _make_one_hot_probs(
                            draft_ids,
                            m_l.vocab_size,
                            m_l.device,
                        )
                        large_edge_key = "small_to_large"
                    else:
                        probs_to_large = middle_probs
                        if probs_to_large is None and middle_ids is not None:
                            probs_to_large = _make_one_hot_probs(
                                middle_ids,
                                m_l.vocab_size,
                                m_l.device,
                            )
                        large_edge_key = "middle_to_large"
                    final_step_records, _, _, large_meta = review_candidate_records(
                        m_l,
                        "large",
                        large_edge_key,
                        initial_input,
                        output_ids,
                        candidate_to_large,
                        tokenizer,
                        stats,
                        step_index=step_index,
                        probs=probs_to_large,
                        leniency=1,
                        empty_refill_position=output_ids.shape[-1] - initial_len,
                    )
                    acceptance_ratio = (
                        len(final_step_records) / len(drafted_tokens)
                        if drafted_tokens
                        else 1.0
                    )
                    acceptance_trace.append(acceptance_ratio)
                    state.update(len(final_step_records), len(drafted_tokens))
                    dynamic_ratio = (
                        large_meta["accepted_count"] / len(small_records)
                        if cfg["shadow_middle"] and small_records
                        else middle_meta["accepted_count"] / len(small_records)
                        if small_records
                        else 1.0
                    )
                    _update_draft_window(
                        cfg,
                        draft_window_state,
                        "small",
                        dynamic_ratio,
                        step_index,
                    )
                    middle_trace_records = middle_records
                    large_trace_records = candidate_to_large
                    middle_meta_for_trace = middle_meta
                else:
                    if mm_as_verifier:
                        m_m.past_key_values = None
                        m_m.past_ids = None
                        mm_as_verifier = False
                    requested = _draft_window_request(
                        cfg,
                        draft_window_state,
                        "middle",
                        remaining,
                        dynamic_enabled=dynamic_windows_enabled,
                    )
                    draft_ids = m_m.propose(
                        initial_input.to(m_m.device),
                        output_ids.to(m_m.device),
                        requested,
                    )
                    drafted_tokens = draft_ids[:, output_ids.shape[-1] :].detach().cpu().tolist()[0]
                    stats["draft_generated_counts"]["middle"] += len(drafted_tokens)
                    drafter_steps["middle"] += 1
                    positions = range(
                        output_ids.shape[-1] - initial_len,
                        output_ids.shape[-1] - initial_len + len(drafted_tokens),
                    )
                    middle_trace_records = make_records(
                        drafted_tokens,
                        tokenizer=tokenizer,
                        source_model="middle",
                        positions=positions,
                        step_index=step_index,
                    )
                    mm_probs = _make_one_hot_probs(draft_ids, m_l.vocab_size, m_l.device)
                    final_step_records, _, _, large_meta = review_candidate_records(
                        m_l,
                        "large",
                        "middle_to_large",
                        initial_input,
                        output_ids,
                        middle_trace_records,
                        tokenizer,
                        stats,
                        step_index=step_index,
                        probs=mm_probs,
                        leniency=1,
                    )
                    middle_meta_for_trace = {
                        "accepted_count": 0,
                        "generated_count": len(middle_trace_records),
                    }
                    large_trace_records = middle_trace_records
                    dynamic_ratio = (
                        large_meta["accepted_count"] / len(middle_trace_records)
                        if middle_trace_records
                        else 1.0
                    )
                    _update_draft_window(
                        cfg,
                        draft_window_state,
                        "middle",
                        dynamic_ratio,
                        step_index,
                    )

                output_ids = torch.cat(
                    [output_ids.to(m_l.device), ids_from_records(final_step_records, m_l.device)],
                    dim=1,
                )
                final_records.extend(final_step_records)

                previous_drafter = state.current_drafter
                state.maybe_switch()
                switched = previous_drafter != state.current_drafter
                switch_count += int(switched)
                alpha_after = state.rolling_alpha
                alpha_trace.append(alpha_after)

                after_runtime = runtime_snapshots(m_s, m_m, m_l)
                if steps is not None:
                    step_payload = {
                        "step_index": step_index,
                        "drafter": drafter_before,
                        "next_drafter": state.current_drafter,
                        "requested_tokens": requested,
                        "candidate_to_large": large_trace_records,
                        "final_step": final_step_records,
                        "large_accepted_count": large_meta["accepted_count"],
                        "large_generated_count": large_meta["generated_count"],
                        "alpha_before": alpha_before,
                        "alpha_after": alpha_after,
                        "switched": switched,
                        "draft_window_policy": draft_window_state["policy"],
                        "draft_window_acceptance_ratio": dynamic_ratio,
                        "model_runtime_delta": runtime_delta(before_runtime, after_runtime),
                    }
                    if drafter_before == "ms":
                        step_payload["small_draft"] = small_records
                        step_payload["middle_result"] = middle_trace_records
                        step_payload["middle_refill"] = cfg["middle_refill"]
                        step_payload["shadow_middle"] = cfg["shadow_middle"]
                        step_payload["middle_accepted_count"] = middle_meta_for_trace[
                            "accepted_count"
                        ]
                        step_payload["middle_generated_count"] = middle_meta_for_trace[
                            "generated_count"
                        ]
                        step_payload["middle_saved_positions"] = len(small_records) - middle_meta_for_trace[
                            "accepted_count"
                        ]
                    else:
                        step_payload["middle_draft"] = middle_trace_records
                    steps.append(step_payload)

                if not final_step_records or any(
                    record["token_id"] == eos_id for record in final_step_records
                ):
                    break
        else:
            raise ValueError(f"Unsupported run name: {run_name}")
    wall_time = time.time() - t0

    return finalize_sample_payload(
        sample_index=sample_index,
        prompt=prompt,
        item=sample_item,
        wall_time=wall_time,
        initial_input=initial_input,
        output_ids=output_ids,
        tokenizer=tokenizer,
        m_s=m_s,
        m_m=m_m,
        m_l=m_l,
        stats=stats,
        final_records=final_records,
        capture_trace=capture_trace,
        steps=steps,
        switch_count=switch_count,
        alpha_trace=alpha_trace,
        acceptance_trace=acceptance_trace,
        drafter_steps=drafter_steps,
        draft_window=_draft_window_payload(draft_window_state),
    )


def summarize_run(sample_results, run_name, cfg):
    n = len(sample_results)
    total_tokens = sum(sample["tokens_generated"] for sample in sample_results)
    total_wall = sum(sample["wall_time"] for sample in sample_results)
    total_score = sum(sample["score"]["score"] for sample in sample_results)
    benchmark_metric = (
        sample_results[0]["score"]["metric_name"] if sample_results else "score"
    )

    runtime_totals = _runtime_totals_from_samples(sample_results)
    aggregate = _aggregate_counters_from_samples(sample_results)
    draft_window_totals, draft_window_change_count_sum = _draft_window_totals_from_samples(
        sample_results
    )
    draft_window_policy = (
        sample_results[0].get("draft_window", {}).get(
            "policy",
            cfg.get("draft_window_policy", "fixed"),
        )
        if sample_results
        else cfg.get("draft_window_policy", "fixed")
    )

    total_runtime_wall = sum(
        runtime_totals[model_key]["total_wall_time"] for model_key in MODEL_KEYS
    )
    avg_runtime = {}
    for model_key in MODEL_KEYS:
        bucket = runtime_totals[model_key]
        avg_runtime[model_key] = {
            "propose_calls": (bucket["propose_calls"] / n) if n else 0.0,
            "review_calls": (bucket["review_calls"] / n) if n else 0.0,
            "propose_wall_time": (bucket["propose_wall_time"] / n) if n else 0.0,
            "review_wall_time": (bucket["review_wall_time"] / n) if n else 0.0,
            "total_wall_time": (bucket["total_wall_time"] / n) if n else 0.0,
            "share": (
                bucket["total_wall_time"] / total_runtime_wall
                if total_runtime_wall
                else 0.0
            ),
        }

    summary = {
        "run_name": run_name,
        "tokens_per_sec": (total_tokens / total_wall) if total_wall else 0.0,
        "avg_wall_time": (total_wall / n) if n else 0.0,
        "avg_tokens_generated": (total_tokens / n) if n else 0.0,
        "benchmark_metric": benchmark_metric,
        "benchmark_score": (total_score / n) if n else 0.0,
        "avg_ml_forward_calls": (
            sum(sample["ml_forward_calls"] for sample in sample_results) / n if n else 0.0
        ),
        "avg_mm_saved_positions": (
            sum(sample["mm_saved_positions"] for sample in sample_results) / n if n else 0.0
        ),
        "avg_switch_count": (
            sum(sample["switch_count"] for sample in sample_results) / n if n else 0.0
        ),
        "avg_mm_forward_calls": (
            sum(sample.get("mm_forward_calls", 0) for sample in sample_results) / n
            if n
            else 0.0
        ),
        "avg_drafter_steps": {
            "small": (
                sum(
                    sample.get("drafter_steps", {}).get("small", 0)
                    for sample in sample_results
                )
                / n
                if n
                else 0.0
            ),
            "middle": (
                sum(
                    sample.get("drafter_steps", {}).get("middle", 0)
                    for sample in sample_results
                )
                / n
                if n
                else 0.0
            ),
        },
        "draft_window_policy": draft_window_policy,
        "avg_draft_window": _avg_draft_windows(draft_window_totals),
        "avg_draft_window_changes": (
            draft_window_change_count_sum / n if n else 0.0
        ),
        "avg_draft_generated_tokens": {
            model_key: (aggregate["draft_generated_counts"][model_key] / n) if n else 0.0
            for model_key in MODEL_KEYS
        },
        "avg_final_source_tokens": {
            model_key: (aggregate["final_source_counts"][model_key] / n) if n else 0.0
            for model_key in MODEL_KEYS
        },
        "avg_middle_review_positions": (
            aggregate["verification_positions"]["middle"] / n if n else 0.0
        ),
        "avg_large_review_positions": (
            aggregate["verification_positions"]["large"] / n if n else 0.0
        ),
        "usage": _per_sample_usage(aggregate, total_tokens),
        "avg_model_runtime": avg_runtime,
        "n_samples": n,
        "k_s": cfg["k_s"],
        "k_m": cfg.get("k_m"),
        "tau": cfg.get("tau"),
        "window_size": cfg.get("window_size"),
        "leniency": cfg["leniency"],
        "max_length": cfg["max_length"],
    }
    return summary


def summary_from_merge_stats(run_name, cfg, aggregate_counters, merge_stats):
    n = merge_stats["n_samples"]
    total_tokens = merge_stats["total_tokens"]
    total_wall = merge_stats["total_wall_time"]
    total_score = merge_stats["total_score"]
    runtime_totals = merge_stats["runtime_totals"]
    total_runtime_wall = sum(
        runtime_totals[model_key]["total_wall_time"] for model_key in MODEL_KEYS
    )

    avg_runtime = {}
    for model_key in MODEL_KEYS:
        bucket = runtime_totals[model_key]
        avg_runtime[model_key] = {
            "propose_calls": (bucket["propose_calls"] / n) if n else 0.0,
            "review_calls": (bucket["review_calls"] / n) if n else 0.0,
            "propose_wall_time": (bucket["propose_wall_time"] / n) if n else 0.0,
            "review_wall_time": (bucket["review_wall_time"] / n) if n else 0.0,
            "total_wall_time": (bucket["total_wall_time"] / n) if n else 0.0,
            "share": (
                bucket["total_wall_time"] / total_runtime_wall
                if total_runtime_wall
                else 0.0
            ),
        }

    return {
        "run_name": run_name,
        "tokens_per_sec": (total_tokens / total_wall) if total_wall else 0.0,
        "avg_wall_time": (total_wall / n) if n else 0.0,
        "avg_tokens_generated": (total_tokens / n) if n else 0.0,
        "benchmark_metric": merge_stats["benchmark_metric"],
        "benchmark_score": (total_score / n) if n else 0.0,
        "avg_ml_forward_calls": (
            merge_stats["ml_forward_call_sum"] / n if n else 0.0
        ),
        "avg_mm_saved_positions": (
            merge_stats["mm_saved_positions_sum"] / n if n else 0.0
        ),
        "avg_switch_count": (
            merge_stats["switch_count_sum"] / n if n else 0.0
        ),
        "avg_mm_forward_calls": (
            merge_stats["mm_forward_call_sum"] / n if n else 0.0
        ),
        "avg_drafter_steps": {
            "small": (
                merge_stats["drafter_step_sums"]["small"] / n if n else 0.0
            ),
            "middle": (
                merge_stats["drafter_step_sums"]["middle"] / n if n else 0.0
            ),
        },
        "draft_window_policy": merge_stats.get(
            "draft_window_policy",
            cfg.get("draft_window_policy", "fixed"),
        ),
        "avg_draft_window": _avg_draft_windows(
            merge_stats.get("draft_window_totals", {})
        ),
        "avg_draft_window_changes": (
            merge_stats.get("draft_window_change_count_sum", 0.0) / n
            if n
            else 0.0
        ),
        "avg_draft_generated_tokens": {
            model_key: (aggregate_counters["draft_generated_counts"][model_key] / n)
            if n
            else 0.0
            for model_key in MODEL_KEYS
        },
        "avg_final_source_tokens": {
            model_key: (aggregate_counters["final_source_counts"][model_key] / n)
            if n
            else 0.0
            for model_key in MODEL_KEYS
        },
        "avg_middle_review_positions": (
            aggregate_counters["verification_positions"]["middle"] / n if n else 0.0
        ),
        "avg_large_review_positions": (
            aggregate_counters["verification_positions"]["large"] / n if n else 0.0
        ),
        "usage": _per_sample_usage(aggregate_counters, total_tokens),
        "avg_model_runtime": avg_runtime,
        "n_samples": n,
        "k_s": cfg["k_s"],
        "k_m": cfg.get("k_m"),
        "tau": cfg.get("tau"),
        "window_size": cfg.get("window_size"),
        "leniency": cfg["leniency"],
        "max_length": cfg["max_length"],
    }


def comparison_summary(candidate_summary, baseline_summary):
    throughput_speedup = None
    baseline_tps = baseline_summary["tokens_per_sec"]
    if baseline_tps:
        throughput_speedup = candidate_summary["tokens_per_sec"] / baseline_tps

    return {
        "throughput_delta": (
            candidate_summary["tokens_per_sec"] - baseline_summary["tokens_per_sec"]
        ),
        "throughput_speedup": throughput_speedup,
        "benchmark_delta": (
            candidate_summary["benchmark_score"] - baseline_summary["benchmark_score"]
        ),
        "avg_wall_time_delta": (
            candidate_summary["avg_wall_time"] - baseline_summary["avg_wall_time"]
        ),
        "avg_ml_forward_calls_delta": (
            candidate_summary["avg_ml_forward_calls"]
            - baseline_summary["avg_ml_forward_calls"]
        ),
        "avg_mm_saved_positions_delta": (
            candidate_summary["avg_mm_saved_positions"]
            - baseline_summary["avg_mm_saved_positions"]
        ),
        "avg_switch_count_delta": (
            candidate_summary["avg_switch_count"]
            - baseline_summary["avg_switch_count"]
        ),
        "avg_draft_window": candidate_summary.get("avg_draft_window", {}),
        "baseline_avg_draft_window": baseline_summary.get("avg_draft_window", {}),
        "avg_draft_window_changes": candidate_summary.get(
            "avg_draft_window_changes",
            0.0,
        ),
        "candidate_pass_rates": candidate_summary.get("usage", {}).get(
            "edge_pass_rates", {}
        ),
        "baseline_pass_rates": baseline_summary.get("usage", {}).get(
            "edge_pass_rates", {}
        ),
    }


def run_names_for_mode(mode):
    if mode == "baseline":
        return ["baseline"]
    if mode == "cascaded":
        return ["cascaded"]
    if mode == "adaptive":
        return ["adaptive"]
    if mode == "compare":
        return ["baseline", "cascaded"]
    if mode == "compare_all":
        return ["baseline", "cascaded", "adaptive"]
    raise ValueError(f"Unsupported mode: {mode}")


def build_result(cfg, modules, tokenizer, m_s, m_m, m_l):
    test_set = modules["csd_datasets"].get_test_set(cfg["dataset"])
    selected_items = select_eval_items(test_set, cfg)
    run_names = run_names_for_mode(cfg["mode"])

    result = {
        "config": serialize_config(cfg),
        "selection": {
            "requested_n_samples": cfg["n_samples"],
            "selected_n_samples": len(selected_items),
            "num_shards": cfg["num_shards"],
            "shard_index": cfg["shard_index"],
            "selected_sample_indices": [sample_index for sample_index, _ in selected_items],
        },
        "runs": {},
    }

    for run_name in run_names:
        sample_results = []
        iterator = tqdm(selected_items, desc=run_name)
        for sample_index, item in iterator:
            sample = run_sample(
                run_name,
                cfg,
                modules,
                tokenizer,
                m_s,
                m_m,
                m_l,
                item,
                sample_index,
                capture_trace=len(sample_results) < cfg["trace_samples"],
            )
            sample_results.append(sample)

        summary = summarize_run(sample_results, run_name, cfg)
        aggregate_counters = _aggregate_counters_from_samples(sample_results)
        merge_stats = _merge_stats_from_samples(sample_results)
        print(f"\n{run_name} summary")
        pprint(summary)
        result["runs"][run_name] = {
            "summary": summary,
            "aggregate_counters": aggregate_counters,
            "merge_stats": merge_stats,
            "sample_metrics": [sample_metric_payload(sample) for sample in sample_results],
            "trace_samples": sample_results[: cfg["trace_samples"]],
        }

    comparisons = {}
    if "baseline" in result["runs"]:
        baseline_summary = result["runs"]["baseline"]["summary"]
        for candidate_name in ("cascaded", "adaptive"):
            if candidate_name in result["runs"]:
                comparisons[f"{candidate_name}_vs_baseline"] = comparison_summary(
                    result["runs"][candidate_name]["summary"],
                    baseline_summary,
                )
    if comparisons:
        result["comparisons"] = comparisons

    return result


def save_result(result, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(DEFAULT_CONFIG)
    for key in (
        "mode",
        "branch_root",
        "dataset",
        "ms_name",
        "mm_name",
        "ml_name",
        "small_device",
        "middle_device",
        "large_device",
        "device",
        "n_samples",
        "num_shards",
        "shard_index",
        "k_s",
        "k_m",
        "leniency",
        "tau",
        "window_size",
        "max_length",
        "trace_samples",
        "middle_refill",
        "shadow_middle",
        "draft_window_policy",
        "dynamic_k_s_min",
        "dynamic_k_s_max",
        "dynamic_k_m_min",
        "dynamic_k_m_max",
        "dynamic_acceptance_low",
        "dynamic_acceptance_high",
        "dynamic_window_step",
        "output",
    ):
        cfg[key] = getattr(args, key)

    print("Config:")
    pprint(cfg)
    modules = load_branch_modules(cfg["branch_root"])
    run_names = run_names_for_mode(cfg["mode"])
    tokenizer, m_s, m_m, m_l = load_models(cfg, modules, run_names)
    result = build_result(cfg, modules, tokenizer, m_s, m_m, m_l)
    save_result(result, cfg["output"])
    print(f"\nSaved results to {cfg['output']}")
