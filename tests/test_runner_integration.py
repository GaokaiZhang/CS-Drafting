import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from merge_fixed_window_shards import merge_results, save_result as save_merged_result
from fixed_window import comparison_summary
from main_fixed_window import build_result, run_eval, save_result, select_eval_items, serialize_config
from tests.fakes import FakeModel, FakeTokenizer


pytestmark = pytest.mark.integration


@pytest.fixture
def fake_cfg(tmp_path):
    return {
        "mode": "compare",
        "dataset": "mmlu",
        "n_samples": 1,
        "num_shards": 1,
        "shard_index": 0,
        "small_window": 3,
        "middle_window": 6,
        "max_length": 6,
        "trace_samples": 1,
        "stop_on_answer": True,
        "output": str(tmp_path / "compare.json"),
        "device": "cpu",
        "dtype": torch.float16,
    }


@pytest.fixture
def fake_models():
    target = [11, 12, 13, 14, 15, 16]
    small_path = [11, 12, 13, 18, 19, 20]
    tokenizer = FakeTokenizer()
    m_s = FakeModel(small_path, small_path)
    m_m = FakeModel(target, target)
    m_l = FakeModel(target, target)
    return tokenizer, m_s, m_m, m_l


def test_build_result_compare_shape(fake_cfg, fake_models):
    tokenizer, m_s, m_m, m_l = fake_models
    test_set = [{
        "question": "Which verifier runs first?",
        "choices": ["small", "middle", "large", "none"],
        "answer": 1,
    }]
    result = build_result(fake_cfg, test_set, tokenizer, m_s, m_m, m_l)

    assert set(result["runs"]) == {"baseline", "hierarchical"}
    assert "comparison" in result
    assert result["runs"]["baseline"]["summary"]["run_name"] == "baseline"
    assert result["runs"]["hierarchical"]["summary"]["run_name"] == "hierarchical"
    assert result["runs"]["hierarchical"]["summary"]["usage"]["edge_pass_rates"]["small_to_middle"] == pytest.approx(0.5)
    assert result["runs"]["hierarchical"]["summary"]["avg_model_runtime"]["middle"]["review_calls"] == pytest.approx(2.0)
    assert len(result["runs"]["hierarchical"]["trace_samples"]) == 1
    assert result["runs"]["hierarchical"]["trace_samples"][0]["score"]["correct"] is True


def test_build_result_double_layer_variant_uses_teammate_style(fake_cfg, fake_models):
    tokenizer, m_s, m_m, m_l = fake_models
    cfg = dict(fake_cfg)
    cfg["hierarchical_variant"] = "double_layer"
    test_set = [{
        "question": "Which verifier runs first?",
        "choices": ["small", "middle", "large", "none"],
        "answer": 1,
    }]

    result = build_result(cfg, test_set, tokenizer, m_s, m_m, m_l)

    hierarchical_trace = result["runs"]["hierarchical"]["trace_samples"][0]
    first_middle_result = hierarchical_trace["blocks"][0]["middle_cycles"][0]["middle_result"]
    assert [token["source_model"] for token in first_middle_result] == [
        "small",
        "small",
        "small",
        "middle",
    ]
    assert result["runs"]["hierarchical"]["summary"]["avg_model_runtime"]["middle"]["review_calls"] == pytest.approx(3.0)


def test_save_result_roundtrip(fake_cfg, fake_models):
    tokenizer, m_s, m_m, m_l = fake_models
    test_set = [{
        "question": "Which verifier runs first?",
        "choices": ["small", "middle", "large", "none"],
        "answer": 1,
    }]
    result = build_result(fake_cfg, test_set, tokenizer, m_s, m_m, m_l)
    save_result(result, fake_cfg["output"])

    with open(fake_cfg["output"]) as handle:
        loaded = json.load(handle)

    assert loaded["config"]["mode"] == "compare"
    assert loaded["runs"]["baseline"]["summary"]["benchmark_score"] == pytest.approx(1.0)
    assert loaded["runs"]["hierarchical"]["sample_metrics"][0]["prediction"] == "B"


def test_merge_sharded_results_matches_unsharded(tmp_path):
    test_set = [
        {
            "question": "Which verifier runs first?",
            "choices": ["small", "middle", "large", "none"],
            "answer": 1,
        },
        {
            "question": "Which verifier runs second?",
            "choices": ["small", "middle", "large", "none"],
            "answer": 1,
        },
    ]

    def make_models():
        target = [11, 12, 13, 14, 15, 16]
        small_path = [11, 12, 13, 18, 19, 20]
        return FakeTokenizer(), FakeModel(small_path, small_path), FakeModel(target, target), FakeModel(target, target)

    full_cfg = {
        "mode": "compare",
        "dataset": "mmlu",
        "n_samples": 2,
        "num_shards": 1,
        "shard_index": 0,
        "small_window": 3,
        "middle_window": 6,
        "max_length": 6,
        "trace_samples": 2,
        "stop_on_answer": True,
        "output": str(tmp_path / "full.json"),
        "device": "cpu",
        "dtype": torch.float16,
    }
    tokenizer, m_s, m_m, m_l = make_models()
    full_result = build_result(full_cfg, test_set, tokenizer, m_s, m_m, m_l)

    shard_paths = []
    for shard_index in (0, 1):
        shard_cfg = dict(full_cfg)
        shard_cfg["num_shards"] = 2
        shard_cfg["shard_index"] = shard_index
        shard_cfg["trace_samples"] = 1
        shard_cfg["output"] = str(tmp_path / f"shard_{shard_index}.json")
        tokenizer, m_s, m_m, m_l = make_models()
        shard_result = build_result(shard_cfg, test_set, tokenizer, m_s, m_m, m_l)
        save_result(shard_result, shard_cfg["output"])
        shard_paths.append(shard_cfg["output"])

    merged = merge_results(shard_paths)
    save_merged_result(merged, tmp_path / "merged.json")

    assert merged["selection"]["selected_n_samples"] == 2
    assert merged["selection"]["selected_sample_indices"] == [0, 1]
    assert merged["runs"]["baseline"]["summary"]["benchmark_score"] == pytest.approx(
        full_result["runs"]["baseline"]["summary"]["benchmark_score"]
    )
    assert merged["runs"]["hierarchical"]["summary"]["benchmark_score"] == pytest.approx(
        full_result["runs"]["hierarchical"]["summary"]["benchmark_score"]
    )
    assert merged["runs"]["baseline"]["summary"]["usage"]["edge_pass_rates"]["small_to_large"] == pytest.approx(
        full_result["runs"]["baseline"]["summary"]["usage"]["edge_pass_rates"]["small_to_large"]
    )
    assert [sample["sample_index"] for sample in merged["runs"]["baseline"]["sample_metrics"]] == [0, 1]


def test_merge_sharded_results_normalizes_equivalent_model_paths(tmp_path):
    base = {
        "config": {
            "mode": "compare",
            "dataset": "gsm8k",
            "ms_name": "/data/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/rev-a",
            "mm_name": "/data/user_data/zichuwu/hf_models/Qwen/Qwen2.5-1.5B-Instruct",
            "ml_name": "Qwen/Qwen2.5-14B-Instruct",
            "small_window": 3,
            "middle_window": 6,
            "max_length": 200,
            "n_samples": 10,
            "stop_on_answer": True,
            "trace_samples": 0,
            "num_shards": 2,
            "shard_index": 0,
        },
        "selection": {
            "requested_n_samples": 10,
            "selected_n_samples": 1,
            "num_shards": 2,
            "shard_index": 0,
            "selected_sample_indices": [0],
        },
        "runs": {
            "baseline": {
                "config": {
                    "mode": "compare",
                    "dataset": "gsm8k",
                    "ms_name": "/data/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/rev-a",
                    "mm_name": "/data/user_data/zichuwu/hf_models/Qwen/Qwen2.5-1.5B-Instruct",
                    "ml_name": "Qwen/Qwen2.5-14B-Instruct",
                    "small_window": 3,
                    "middle_window": 6,
                    "max_length": 200,
                    "n_samples": 10,
                    "stop_on_answer": True,
                    "trace_samples": 0,
                    "num_shards": 2,
                    "shard_index": 0,
                },
                "summary": {
                    "run_name": "baseline",
                    "benchmark_score": 1.0,
                    "tokens_per_sec": 10.0,
                    "avg_wall_time": 1.0,
                    "avg_tokens_generated": 10.0,
                    "benchmark_metric": "exact_match",
                    "avg_ml_forward_calls": 1.0,
                    "avg_mm_forward_calls": 0.0,
                },
                "aggregate_counters": {
                    "draft_generated_counts": {"small": 1, "middle": 0, "large": 0},
                    "final_source_counts": {"small": 1, "middle": 0, "large": 0},
                    "verification_positions": {"small": 0, "middle": 0, "large": 1},
                    "verification_calls": {"small": 0, "middle": 0, "large": 1},
                    "edge_pass": {
                        "small_to_middle": {"accepted": 0, "proposed": 0},
                        "middle_to_large": {"accepted": 0, "proposed": 0},
                        "small_to_large": {"accepted": 1, "proposed": 1},
                    },
                },
                "merge_stats": {
                    "n_samples": 1,
                    "total_tokens": 10,
                    "total_wall_time": 1.0,
                    "total_score": 1.0,
                    "benchmark_metric": "exact_match",
                    "ml_forward_call_sum": 1.0,
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
                        model: {
                            "propose_calls": 0,
                            "review_calls": 0,
                            "propose_wall_time": 0.0,
                            "review_wall_time": 0.0,
                            "total_wall_time": 0.0,
                        }
                        for model in ("small", "middle", "large")
                    },
                },
                "sample_metrics": [{"sample_index": 0}],
                "trace_samples": [],
            },
            "hierarchical": {
                "config": {
                    "mode": "compare",
                    "dataset": "gsm8k",
                    "ms_name": "/data/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/rev-a",
                    "mm_name": "/data/user_data/zichuwu/hf_models/Qwen/Qwen2.5-1.5B-Instruct",
                    "ml_name": "Qwen/Qwen2.5-14B-Instruct",
                    "small_window": 3,
                    "middle_window": 6,
                    "max_length": 200,
                    "n_samples": 10,
                    "stop_on_answer": True,
                    "trace_samples": 0,
                    "num_shards": 2,
                    "shard_index": 0,
                },
                "summary": {
                    "run_name": "hierarchical",
                    "benchmark_score": 1.0,
                    "tokens_per_sec": 11.0,
                    "avg_wall_time": 1.0,
                    "avg_tokens_generated": 10.0,
                    "benchmark_metric": "exact_match",
                    "avg_ml_forward_calls": 1.0,
                    "avg_mm_forward_calls": 1.0,
                },
                "aggregate_counters": {
                    "draft_generated_counts": {"small": 1, "middle": 0, "large": 0},
                    "final_source_counts": {"small": 1, "middle": 0, "large": 0},
                    "verification_positions": {"small": 0, "middle": 1, "large": 1},
                    "verification_calls": {"small": 0, "middle": 1, "large": 1},
                    "edge_pass": {
                        "small_to_middle": {"accepted": 1, "proposed": 1},
                        "middle_to_large": {"accepted": 1, "proposed": 1},
                        "small_to_large": {"accepted": 0, "proposed": 0},
                    },
                },
                "merge_stats": {
                    "n_samples": 1,
                    "total_tokens": 10,
                    "total_wall_time": 1.0,
                    "total_score": 1.0,
                    "benchmark_metric": "exact_match",
                    "ml_forward_call_sum": 1.0,
                    "mm_forward_call_sum": 1.0,
                    "draft_window_totals": {
                        "small": {"sum": 0, "count": 0},
                        "middle": {"sum": 0, "count": 0},
                    },
                    "window_change_count": 0,
                    "routing_totals": {"middle": 0, "large": 0},
                    "routing_change_count": 0,
                    "routing_seen": False,
                    "runtime_totals": {
                        model: {
                            "propose_calls": 0,
                            "review_calls": 0,
                            "propose_wall_time": 0.0,
                            "review_wall_time": 0.0,
                            "total_wall_time": 0.0,
                        }
                        for model in ("small", "middle", "large")
                    },
                },
                "sample_metrics": [{"sample_index": 0}],
                "trace_samples": [],
            },
        },
    }

    alt = json.loads(json.dumps(base))
    alt["config"]["ms_name"] = "/data/user_data/zichuwu/hf_models/Qwen/Qwen2.5-0.5B-Instruct"
    alt["config"]["shard_index"] = 1
    alt["selection"]["shard_index"] = 1
    alt["selection"]["selected_sample_indices"] = [1]
    for run in alt["runs"].values():
        run["config"]["ms_name"] = "/data/user_data/zichuwu/hf_models/Qwen/Qwen2.5-0.5B-Instruct"
        run["config"]["shard_index"] = 1
        run["sample_metrics"] = [{"sample_index": 1}]

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(base))
    second.write_text(json.dumps(alt))

    merged = merge_results([str(first), str(second)])

    assert merged["config"]["ms_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert merged["selection"]["selected_sample_indices"] == [0, 1]


def test_merge_sharded_results_ignores_runtime_only_run_state(tmp_path):
    base = {
        "config": {
            "mode": "compare",
            "dataset": "gsm8k",
            "ms_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "mm_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "ml_name": "Qwen/Qwen2.5-14B-Instruct",
            "small_window": 4,
            "middle_window": 6,
            "max_length": 200,
            "n_samples": 10,
            "stop_on_answer": True,
            "trace_samples": 0,
            "num_shards": 2,
            "shard_index": 0,
        },
        "selection": {
            "requested_n_samples": 10,
            "selected_n_samples": 1,
            "num_shards": 2,
            "shard_index": 0,
            "selected_sample_indices": [0],
        },
        "specs": [
            {
                "label": "adaptive_sw4_mw6",
                "run_type": "hierarchical",
                "small_window": 4,
                "middle_window": 6,
                "hierarchical_variant": "double_layer",
                "window_policy": "utility",
                "baseline_label": "baseline_sw5",
                "order": 0,
            }
        ],
        "runs": {
            "baseline_sw5": {
                "config": {
                    "mode": "compare",
                    "dataset": "gsm8k",
                    "ms_name": "Qwen/Qwen2.5-0.5B-Instruct",
                    "mm_name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "ml_name": "Qwen/Qwen2.5-14B-Instruct",
                    "small_window": 5,
                    "middle_window": 6,
                    "max_length": 200,
                    "n_samples": 10,
                    "stop_on_answer": True,
                    "trace_samples": 0,
                    "num_shards": 2,
                    "shard_index": 0,
                },
                "summary": {
                    "run_name": "baseline_sw5",
                    "benchmark_score": 1.0,
                    "tokens_per_sec": 10.0,
                    "avg_wall_time": 1.0,
                    "avg_tokens_generated": 10.0,
                    "benchmark_metric": "exact_match",
                    "avg_ml_forward_calls": 1.0,
                    "avg_mm_forward_calls": 0.0,
                },
                "aggregate_counters": {
                    "draft_generated_counts": {"small": 1, "middle": 0, "large": 0},
                    "final_source_counts": {"small": 1, "middle": 0, "large": 0},
                    "verification_positions": {"small": 0, "middle": 0, "large": 1},
                    "verification_calls": {"small": 0, "middle": 0, "large": 1},
                    "edge_pass": {
                        "small_to_middle": {"accepted": 0, "proposed": 0},
                        "middle_to_large": {"accepted": 0, "proposed": 0},
                        "small_to_large": {"accepted": 1, "proposed": 1},
                    },
                },
                "merge_stats": {
                    "n_samples": 1,
                    "total_tokens": 10,
                    "total_wall_time": 1.0,
                    "total_score": 1.0,
                    "benchmark_metric": "exact_match",
                    "ml_forward_call_sum": 1.0,
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
                        model: {
                            "propose_calls": 0,
                            "review_calls": 0,
                            "propose_wall_time": 0.0,
                            "review_wall_time": 0.0,
                            "total_wall_time": 0.0,
                        }
                        for model in ("small", "middle", "large")
                    },
                },
                "sample_metrics": [{"sample_index": 0}],
                "trace_samples": [],
            },
            "adaptive_sw4_mw6": {
                "config": {
                    "mode": "compare",
                    "dataset": "gsm8k",
                    "ms_name": "Qwen/Qwen2.5-0.5B-Instruct",
                    "mm_name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "ml_name": "Qwen/Qwen2.5-14B-Instruct",
                    "small_window": 4,
                    "middle_window": 6,
                    "max_length": 200,
                    "n_samples": 10,
                    "stop_on_answer": True,
                    "trace_samples": 0,
                    "num_shards": 2,
                    "shard_index": 0,
                    "hierarchical_variant": "double_layer",
                    "window_policy": "utility",
                    "_global_window_state": {
                        "current": {"small": 4, "middle": 5},
                    },
                },
                "summary": {
                    "run_name": "adaptive_sw4_mw6",
                    "benchmark_score": 1.1,
                    "tokens_per_sec": 11.0,
                    "avg_wall_time": 0.9,
                    "avg_tokens_generated": 10.0,
                    "benchmark_metric": "exact_match",
                    "avg_ml_forward_calls": 0.9,
                    "avg_mm_forward_calls": 0.5,
                },
                "aggregate_counters": {
                    "draft_generated_counts": {"small": 1, "middle": 0, "large": 0},
                    "final_source_counts": {"small": 1, "middle": 0, "large": 0},
                    "verification_positions": {"small": 0, "middle": 0, "large": 1},
                    "verification_calls": {"small": 0, "middle": 0, "large": 1},
                    "edge_pass": {
                        "small_to_middle": {"accepted": 0, "proposed": 0},
                        "middle_to_large": {"accepted": 0, "proposed": 0},
                        "small_to_large": {"accepted": 1, "proposed": 1},
                    },
                },
                "merge_stats": {
                    "n_samples": 1,
                    "total_tokens": 10,
                    "total_wall_time": 0.9,
                    "total_score": 1.1,
                    "benchmark_metric": "exact_match",
                    "ml_forward_call_sum": 0.9,
                    "mm_forward_call_sum": 0.5,
                    "draft_window_totals": {
                        "small": {"sum": 4, "count": 1},
                        "middle": {"sum": 5, "count": 1},
                    },
                    "window_change_count": 1,
                    "routing_totals": {"middle": 1, "large": 0},
                    "routing_change_count": 0,
                    "routing_seen": False,
                    "runtime_totals": {
                        model: {
                            "propose_calls": 0,
                            "review_calls": 0,
                            "propose_wall_time": 0.0,
                            "review_wall_time": 0.0,
                            "total_wall_time": 0.0,
                        }
                        for model in ("small", "middle", "large")
                    },
                },
                "sample_metrics": [{"sample_index": 0}],
                "trace_samples": [],
            },
        },
    }

    alt = json.loads(json.dumps(base))
    alt["config"]["shard_index"] = 1
    alt["selection"]["shard_index"] = 1
    alt["selection"]["selected_sample_indices"] = [1]
    for run in alt["runs"].values():
        run["config"]["shard_index"] = 1
        run["sample_metrics"] = [{"sample_index": 1}]
    alt["runs"]["adaptive_sw4_mw6"]["config"]["_global_window_state"]["current"]["middle"] = 7

    first = tmp_path / "first_runtime_state.json"
    second = tmp_path / "second_runtime_state.json"
    first.write_text(json.dumps(base))
    second.write_text(json.dumps(alt))

    merged = merge_results([str(first), str(second)])

    assert merged["selection"]["selected_sample_indices"] == [0, 1]
    assert "_global_window_state" not in merged["runs"]["adaptive_sw4_mw6"]["config"]


def test_select_eval_items_uses_full_dataset_when_n_samples_nonpositive():
    test_set = ["a", "b", "c"]
    cfg = {"n_samples": 0, "num_shards": 1, "shard_index": 0}

    selected = select_eval_items(test_set, cfg)

    assert [index for index, _ in selected] == [0, 1, 2]


def test_merge_sharded_focused_results_preserves_comparisons_and_routing(tmp_path):
    test_set = [
        {
            "question": "Which verifier runs first?",
            "choices": ["small", "middle", "large", "none"],
            "answer": 1,
        },
        {
            "question": "Which verifier runs second?",
            "choices": ["small", "middle", "large", "none"],
            "answer": 1,
        },
    ]

    specs = [
        {
            "label": "baseline_sw3",
            "run_type": "baseline",
            "small_window": 3,
            "middle_window": None,
            "hierarchical_variant": None,
            "window_policy": None,
            "baseline_label": None,
            "order": 0,
        },
        {
            "label": "selective_refill_sw3_mw6",
            "run_type": "hierarchical",
            "small_window": 3,
            "middle_window": 6,
            "hierarchical_variant": "selective_route_refill_on_full_accept",
            "window_policy": "utility",
            "baseline_label": "baseline_sw3",
            "order": 1,
        },
    ]

    def make_models():
        target = [11, 12, 13, 14, 15, 16, 17, 18]
        small_path = [11, 12, 13, 18, 19, 20, 21, 22]
        return (
            FakeTokenizer(),
            FakeModel(small_path, small_path),
            FakeModel(target, target),
            FakeModel(target, target),
        )

    def build_focused_result(cfg):
        tokenizer, m_s, m_m, m_l = make_models()
        selected_items = select_eval_items(test_set, cfg)
        result = {
            "config": serialize_config(cfg),
            "selection": {
                "requested_n_samples": cfg["n_samples"],
                "selected_n_samples": len(selected_items),
                "num_shards": cfg["num_shards"],
                "shard_index": cfg["shard_index"],
                "selected_sample_indices": [sample_index for sample_index, _ in selected_items],
            },
            "specs": [
                {key: value for key, value in spec.items() if key != "order"}
                for spec in specs
            ],
            "runs": {},
            "comparisons": {},
            "comparison_baselines": {},
        }

        for spec in specs:
            run_cfg = dict(cfg)
            run_cfg["run_type"] = spec["run_type"]
            run_cfg["small_window"] = spec["small_window"]
            if spec["run_type"] == "hierarchical":
                run_cfg["middle_window"] = spec["middle_window"]
                run_cfg["hierarchical_variant"] = spec["hierarchical_variant"]
                run_cfg["window_policy"] = spec["window_policy"]
            run_data = run_eval(
                spec["run_type"],
                run_cfg,
                selected_items,
                tokenizer,
                m_s,
                m_m,
                m_l,
            )
            run_data["config"] = serialize_config(run_cfg)
            result["runs"][spec["label"]] = run_data

        result["comparisons"]["selective_refill_sw3_mw6"] = comparison_summary(
            result["runs"]["selective_refill_sw3_mw6"]["summary"],
            result["runs"]["baseline_sw3"]["summary"],
        )
        result["comparison_baselines"]["selective_refill_sw3_mw6"] = "baseline_sw3"
        return result

    full_cfg = {
        "mode": "compare",
        "dataset": "mmlu",
        "n_samples": 2,
        "num_shards": 1,
        "shard_index": 0,
        "small_window": 3,
        "middle_window": 6,
        "hierarchical_variant": "double_layer",
        "window_policy": "fixed",
        "dynamic_utility_margin": 0.0,
        "max_length": 8,
        "trace_samples": 1,
        "stop_on_answer": False,
        "output": str(tmp_path / "focused_full.json"),
        "device": "cpu",
        "dtype": torch.float16,
    }

    full_result = build_focused_result(full_cfg)

    shard_paths = []
    for shard_index in (0, 1):
        shard_cfg = dict(full_cfg)
        shard_cfg["num_shards"] = 2
        shard_cfg["shard_index"] = shard_index
        shard_cfg["trace_samples"] = 0
        shard_cfg["output"] = str(tmp_path / f"focused_shard_{shard_index}.json")
        shard_result = build_focused_result(shard_cfg)
        save_result(shard_result, shard_cfg["output"])
        shard_paths.append(shard_cfg["output"])

    merged = merge_results(shard_paths)

    assert merged["comparison_baselines"]["selective_refill_sw3_mw6"] == "baseline_sw3"
    assert merged["comparisons"]["selective_refill_sw3_mw6"]["throughput_delta"] == pytest.approx(
        merged["runs"]["selective_refill_sw3_mw6"]["summary"]["tokens_per_sec"]
        - merged["runs"]["baseline_sw3"]["summary"]["tokens_per_sec"]
    )
    assert merged["comparisons"]["selective_refill_sw3_mw6"]["benchmark_delta"] == pytest.approx(
        full_result["comparisons"]["selective_refill_sw3_mw6"]["benchmark_delta"]
    )
    assert merged["runs"]["selective_refill_sw3_mw6"]["summary"]["routing"]["counts"]["large"] >= 0
    assert merged["runs"]["selective_refill_sw3_mw6"]["summary"]["avg_draft_window"]["middle"] == pytest.approx(
        full_result["runs"]["selective_refill_sw3_mw6"]["summary"]["avg_draft_window"]["middle"]
    )


def test_merge_rejects_incomplete_focused_shard(tmp_path):
    shard = {
        "config": {
            "mode": "compare",
            "dataset": "mmlu",
            "ms_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "mm_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "ml_name": "Qwen/Qwen2.5-14B-Instruct",
            "small_window": 3,
            "middle_window": 6,
            "max_length": 8,
            "n_samples": 2,
            "stop_on_answer": False,
            "num_shards": 2,
            "shard_index": 0,
        },
        "selection": {
            "selected_n_samples": 1,
            "selected_sample_indices": [0],
        },
        "specs": [
            {
                "label": "baseline_sw3",
                "run_type": "baseline",
                "small_window": 3,
                "middle_window": None,
                "hierarchical_variant": None,
                "window_policy": None,
                "baseline_label": None,
            },
            {
                "label": "selective_refill_sw3_mw6",
                "run_type": "hierarchical",
                "small_window": 3,
                "middle_window": 6,
                "hierarchical_variant": "selective_route_refill_on_full_accept",
                "window_policy": "utility",
                "baseline_label": "baseline_sw3",
            },
        ],
        "runs": {
            "baseline_sw3": {},
        },
    }
    path = tmp_path / "partial_shard.json"
    path.write_text(json.dumps(shard))

    with pytest.raises(ValueError, match="missing runs"):
        merge_results([str(path)])
