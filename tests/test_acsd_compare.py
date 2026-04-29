import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_acsd_compare import (
    _first_unreviewable_index,
    _align_probs_last_dim,
    _draft_window_payload,
    _draft_window_request,
    _harmonize_model_vocab_sizes,
    _init_draft_window_state,
    _merge_stats_from_samples,
    _resolve_local_snapshot,
    _update_draft_window,
    comparison_summary,
    new_run_stats,
    review_candidate_records,
    summary_from_merge_stats,
    summarize_run,
)


pytestmark = pytest.mark.unit


def _runtime(total, propose_calls=0, review_calls=0):
    return {
        "propose_calls": propose_calls,
        "review_calls": review_calls,
        "propose_wall_time": total / 2,
        "review_wall_time": total / 2,
        "total_wall_time": total,
    }


def _raw_counters(
    *,
    draft_small=0,
    draft_middle=0,
    draft_large=0,
    final_small=0,
    final_middle=0,
    final_large=0,
    verify_middle=0,
    verify_large=0,
    calls_middle=0,
    calls_large=0,
    s_to_m=(0, 0),
    m_to_l=(0, 0),
    s_to_l=(0, 0),
):
    counters = new_run_stats()
    counters["draft_generated_counts"]["small"] = draft_small
    counters["draft_generated_counts"]["middle"] = draft_middle
    counters["draft_generated_counts"]["large"] = draft_large
    counters["final_source_counts"]["small"] = final_small
    counters["final_source_counts"]["middle"] = final_middle
    counters["final_source_counts"]["large"] = final_large
    counters["verification_positions"]["middle"] = verify_middle
    counters["verification_positions"]["large"] = verify_large
    counters["verification_calls"]["middle"] = calls_middle
    counters["verification_calls"]["large"] = calls_large
    counters["edge_pass"]["small_to_middle"]["accepted"] = s_to_m[0]
    counters["edge_pass"]["small_to_middle"]["proposed"] = s_to_m[1]
    counters["edge_pass"]["middle_to_large"]["accepted"] = m_to_l[0]
    counters["edge_pass"]["middle_to_large"]["proposed"] = m_to_l[1]
    counters["edge_pass"]["small_to_large"]["accepted"] = s_to_l[0]
    counters["edge_pass"]["small_to_large"]["proposed"] = s_to_l[1]
    return counters


def test_summarize_run_aggregates_branch_metrics():
    cfg = {
        "k_s": 5,
        "k_m": 4,
        "tau": 0.4,
        "window_size": 20,
        "leniency": 1,
        "max_length": 200,
    }
    sample_results = [
        {
            "tokens_generated": 10,
            "wall_time": 2.0,
            "score": {"metric_name": "accuracy", "score": 1.0},
            "ml_forward_calls": 3,
            "mm_forward_calls": 1,
            "mm_saved_positions": 4,
            "switch_count": 1,
            "drafter_steps": {"small": 2, "middle": 1},
            "raw_counters": _raw_counters(
                draft_small=8,
                draft_middle=2,
                final_small=6,
                final_middle=2,
                verify_middle=8,
                verify_large=8,
                calls_middle=2,
                calls_large=2,
                s_to_m=(6, 8),
                m_to_l=(7, 8),
            ),
            "model_runtime": {
                "small": _runtime(0.8, propose_calls=1),
                "middle": _runtime(0.4, review_calls=1),
                "large": _runtime(0.8, review_calls=1),
            },
        },
        {
            "tokens_generated": 14,
            "wall_time": 3.0,
            "score": {"metric_name": "accuracy", "score": 0.0},
            "ml_forward_calls": 5,
            "mm_forward_calls": 2,
            "mm_saved_positions": 2,
            "switch_count": 3,
            "drafter_steps": {"small": 1, "middle": 2},
            "raw_counters": _raw_counters(
                draft_small=9,
                draft_middle=4,
                draft_large=1,
                final_small=4,
                final_middle=5,
                final_large=1,
                verify_middle=9,
                verify_large=10,
                calls_middle=2,
                calls_large=2,
                s_to_m=(7, 9),
                m_to_l=(9, 10),
            ),
            "model_runtime": {
                "small": _runtime(1.0, propose_calls=2),
                "middle": _runtime(0.6, review_calls=2),
                "large": _runtime(1.4, review_calls=2),
            },
        },
    ]

    summary = summarize_run(sample_results, "cascaded", cfg)

    assert summary["run_name"] == "cascaded"
    assert summary["benchmark_metric"] == "accuracy"
    assert summary["benchmark_score"] == pytest.approx(0.5)
    assert summary["tokens_per_sec"] == pytest.approx(24 / 5)
    assert summary["avg_wall_time"] == pytest.approx(2.5)
    assert summary["avg_ml_forward_calls"] == pytest.approx(4.0)
    assert summary["avg_mm_saved_positions"] == pytest.approx(3.0)
    assert summary["avg_switch_count"] == pytest.approx(2.0)
    assert summary["avg_mm_forward_calls"] == pytest.approx(1.5)
    assert summary["avg_drafter_steps"]["middle"] == pytest.approx(1.5)
    assert summary["usage"]["edge_pass_rates"]["small_to_middle"] == pytest.approx(13 / 17)
    assert summary["usage"]["edge_pass_rates"]["middle_to_large"] == pytest.approx(16 / 18)
    assert summary["avg_model_runtime"]["middle"]["review_calls"] == pytest.approx(1.5)


def test_comparison_summary_reports_deltas_and_speedup():
    baseline = {
        "tokens_per_sec": 20.0,
        "benchmark_score": 0.30,
        "avg_wall_time": 5.0,
        "avg_ml_forward_calls": 30.0,
        "avg_mm_saved_positions": 0.0,
        "avg_switch_count": 0.0,
    }
    candidate = {
        "tokens_per_sec": 30.0,
        "benchmark_score": 0.35,
        "avg_wall_time": 4.0,
        "avg_ml_forward_calls": 24.0,
        "avg_mm_saved_positions": 8.0,
        "avg_switch_count": 1.0,
    }

    comparison = comparison_summary(candidate, baseline)

    assert comparison["throughput_delta"] == pytest.approx(10.0)
    assert comparison["throughput_speedup"] == pytest.approx(1.5)
    assert comparison["benchmark_delta"] == pytest.approx(0.05)
    assert comparison["avg_wall_time_delta"] == pytest.approx(-1.0)
    assert comparison["avg_ml_forward_calls_delta"] == pytest.approx(-6.0)
    assert comparison["avg_mm_saved_positions_delta"] == pytest.approx(8.0)


def test_acceptance_draft_window_updates_and_summarizes_requests():
    cfg = {
        "k_s": 5,
        "k_m": 4,
        "draft_window_policy": "acceptance",
        "dynamic_k_s_min": 2,
        "dynamic_k_s_max": 7,
        "dynamic_k_m_min": 2,
        "dynamic_k_m_max": 6,
        "dynamic_acceptance_low": 0.5,
        "dynamic_acceptance_high": 0.8,
        "dynamic_window_step": 1,
    }
    state = _init_draft_window_state(cfg)
    state["policy"] = "acceptance"

    assert _draft_window_request(cfg, state, "small", remaining=20) == 5
    _update_draft_window(cfg, state, "small", acceptance_ratio=0.9, step_index=0)
    assert _draft_window_request(cfg, state, "small", remaining=20) == 6
    _update_draft_window(cfg, state, "small", acceptance_ratio=0.2, step_index=1)

    payload = _draft_window_payload(state)
    assert payload["averages"]["small"] == pytest.approx(5.5)
    assert payload["change_count"] == 2
    assert payload["trace"][0]["previous_window"] == 5
    assert payload["trace"][0]["next_window"] == 6

    merge_stats = _merge_stats_from_samples(
        [
            {
                "tokens_generated": 1,
                "wall_time": 1.0,
                "score": {"metric_name": "accuracy", "score": 1.0},
                "ml_forward_calls": 0,
                "mm_forward_calls": 0,
                "switch_count": 0,
                "mm_saved_positions": 0,
                "drafter_steps": {"small": 1, "middle": 0},
                "model_runtime": {
                    "small": _runtime(0.0),
                    "middle": _runtime(0.0),
                    "large": _runtime(0.0),
                },
                "draft_window": payload,
            }
        ]
    )
    assert merge_stats["draft_window_totals"]["small"]["sum"] == 11
    assert merge_stats["draft_window_change_count_sum"] == 2


def test_resolve_local_snapshot_prefers_user_data_model_root(tmp_path, monkeypatch):
    local_root = tmp_path / "hf_models"
    model_dir = local_root / "Qwen" / "Qwen2.5-1.5B-Instruct"
    model_dir.mkdir(parents=True)

    monkeypatch.setenv("HF_LOCAL_MODEL_ROOT", str(local_root))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    assert _resolve_local_snapshot("Qwen/Qwen2.5-1.5B-Instruct") == str(model_dir)


def test_align_probs_last_dim_pads_and_truncates():
    probs = torch.ones(1, 2, 3)
    padded = _align_probs_last_dim(probs, 5)
    assert padded.shape == (1, 2, 5)
    assert torch.equal(padded[..., :3], probs)
    assert torch.count_nonzero(padded[..., 3:]) == 0

    truncated = _align_probs_last_dim(probs, 2)
    assert truncated.shape == (1, 2, 2)
    assert torch.equal(truncated, probs[..., :2])


def test_harmonize_model_vocab_sizes_wraps_review_to_common_vocab():
    class DummyModel:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.seen_vocab_sizes = []

        def review(self, initial_input, input_ids, probs, review_index, leniency=1):
            self.seen_vocab_sizes.append(probs.shape[-1])
            return input_ids, probs

    small = DummyModel(4)
    large = DummyModel(6)
    _harmonize_model_vocab_sizes([small, large])

    input_ids = torch.tensor([[1, 2, 3]])
    probs = torch.ones(1, 3, 6)
    _, small_probs = small.review(None, input_ids, probs, 1)
    _, large_probs = large.review(None, input_ids, probs, 1)

    assert small.native_vocab_size == 4
    assert small.vocab_size == 6
    assert small.generation_vocab_size == 4
    assert small.seen_vocab_sizes == [4]
    assert small_probs.shape[-1] == 6

    assert large.native_vocab_size == 6
    assert large.vocab_size == 6
    assert large.generation_vocab_size == 4
    assert large.seen_vocab_sizes == [6]
    assert large_probs.shape[-1] == 6


def test_first_unreviewable_index_detects_out_of_vocab_token():
    verifier = type("Verifier", (), {"native_vocab_size": 5})()
    candidate_records = [
        {"token_id": 1},
        {"token_id": 4},
        {"token_id": 5},
    ]

    assert _first_unreviewable_index(candidate_records, verifier) == 2


def test_review_candidate_records_falls_back_to_verifier_when_first_token_is_invalid():
    class DummyVerifier:
        def __init__(self):
            self.device = torch.device("cpu")
            self.native_vocab_size = 4
            self.vocab_size = 4
            self.propose_calls = 0

        def propose(self, initial_input, input_ids, k):
            self.propose_calls += 1
            return torch.tensor([[11, 3]], dtype=torch.long)

        def review(self, initial_input, input_ids, probs, review_index, leniency=1):
            raise AssertionError("review should not be called for an invalid first token")

    stats = new_run_stats()
    verifier = DummyVerifier()
    candidate_records = [
        {
            "position": 0,
            "token_id": 9,
            "token_text": "<bad>",
            "source_model": "small",
            "verified_by_middle": False,
            "verified_by_large": False,
            "step_index": 0,
        }
    ]

    result_records, reviewed_ids, reviewed_probs, meta = review_candidate_records(
        verifier,
        "large",
        "small_to_large",
        initial_input=torch.tensor([[11]], dtype=torch.long),
        prefix_ids=torch.tensor([[11]], dtype=torch.long),
        candidate_records=candidate_records,
        tokenizer=type("Tokenizer", (), {"decode": staticmethod(lambda ids, **_: f"tok{ids[0]}")})(),
        stats=stats,
        step_index=0,
        probs=None,
        leniency=1,
    )

    assert verifier.propose_calls == 1
    assert reviewed_ids is None
    assert reviewed_probs is None
    assert meta == {"accepted_count": 0, "generated_count": 1, "candidate_len": 1}
    assert result_records[0]["token_id"] == 3
    assert result_records[0]["source_model"] == "large"
    assert stats["edge_pass"]["small_to_large"]["proposed"] == 1
    assert stats["edge_pass"]["small_to_large"]["accepted"] == 0
    assert stats["draft_generated_counts"]["large"] == 1


def test_summary_from_merge_stats_matches_direct_summary():
    cfg = {
        "k_s": 5,
        "k_m": 4,
        "tau": 0.4,
        "window_size": 20,
        "leniency": 1,
        "max_length": 200,
    }
    sample_results = [
        {
            "tokens_generated": 10,
            "wall_time": 2.0,
            "score": {"metric_name": "accuracy", "score": 1.0},
            "ml_forward_calls": 3,
            "mm_forward_calls": 1,
            "mm_saved_positions": 4,
            "switch_count": 1,
            "drafter_steps": {"small": 2, "middle": 1},
            "raw_counters": _raw_counters(
                draft_small=8,
                draft_middle=2,
                final_small=6,
                final_middle=2,
                verify_middle=8,
                verify_large=8,
                calls_middle=2,
                calls_large=2,
                s_to_m=(6, 8),
                m_to_l=(7, 8),
            ),
            "model_runtime": {
                "small": _runtime(0.8, propose_calls=1),
                "middle": _runtime(0.4, review_calls=1),
                "large": _runtime(0.8, review_calls=1),
            },
        },
        {
            "tokens_generated": 14,
            "wall_time": 3.0,
            "score": {"metric_name": "accuracy", "score": 0.0},
            "ml_forward_calls": 5,
            "mm_forward_calls": 2,
            "mm_saved_positions": 2,
            "switch_count": 3,
            "drafter_steps": {"small": 1, "middle": 2},
            "raw_counters": _raw_counters(
                draft_small=9,
                draft_middle=4,
                draft_large=1,
                final_small=4,
                final_middle=5,
                final_large=1,
                verify_middle=9,
                verify_large=10,
                calls_middle=2,
                calls_large=2,
                s_to_m=(7, 9),
                m_to_l=(9, 10),
            ),
            "model_runtime": {
                "small": _runtime(1.0, propose_calls=2),
                "middle": _runtime(0.6, review_calls=2),
                "large": _runtime(1.4, review_calls=2),
            },
        },
    ]
    direct = summarize_run(sample_results, "adaptive", cfg)
    aggregate = _raw_counters(
        draft_small=17,
        draft_middle=6,
        draft_large=1,
        final_small=10,
        final_middle=7,
        final_large=1,
        verify_middle=17,
        verify_large=18,
        calls_middle=4,
        calls_large=4,
        s_to_m=(13, 17),
        m_to_l=(16, 18),
    )
    merge_stats = {
        "n_samples": 2,
        "total_tokens": 24,
        "total_wall_time": 5.0,
        "total_score": 1.0,
        "benchmark_metric": "accuracy",
        "ml_forward_call_sum": 8,
        "mm_forward_call_sum": 3,
        "switch_count_sum": 4,
        "mm_saved_positions_sum": 6,
        "drafter_step_sums": {"small": 3, "middle": 3},
        "runtime_totals": {
            "small": {
                "propose_calls": 3,
                "review_calls": 0,
                "propose_wall_time": 0.9,
                "review_wall_time": 0.9,
                "total_wall_time": 1.8,
            },
            "middle": {
                "propose_calls": 0,
                "review_calls": 3,
                "propose_wall_time": 0.5,
                "review_wall_time": 0.5,
                "total_wall_time": 1.0,
            },
            "large": {
                "propose_calls": 0,
                "review_calls": 3,
                "propose_wall_time": 1.1,
                "review_wall_time": 1.1,
                "total_wall_time": 2.2,
            },
        },
    }

    merged = summary_from_merge_stats("adaptive", cfg, aggregate, merge_stats)

    assert merged["tokens_per_sec"] == pytest.approx(direct["tokens_per_sec"])
    assert merged["benchmark_score"] == pytest.approx(direct["benchmark_score"])
    assert merged["avg_mm_saved_positions"] == pytest.approx(direct["avg_mm_saved_positions"])
    assert merged["usage"]["edge_pass_rates"]["small_to_middle"] == pytest.approx(
        direct["usage"]["edge_pass_rates"]["small_to_middle"]
    )
