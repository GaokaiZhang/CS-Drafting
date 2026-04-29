import os
import sys
import unittest

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark import (
    extract_gsm8k_answer,
    extract_gsm8k_answer_strict,
    extract_mmlu_choice,
    extract_mmlu_choice_strict,
    has_final_answer_signal,
)
from fixed_window import (
    run_baseline_sample,
    run_double_layer_sample,
    run_hierarchical_sample,
)
from tests.fakes import FakeModel, FakeTokenizer


pytestmark = pytest.mark.unit


class TimedFakeModel(FakeModel):
    def __init__(self, proposal_sequence, review_sequence, name="llama", propose_time=0.0, review_time=0.0):
        super().__init__(proposal_sequence, review_sequence, name=name)
        self._fixed_propose_time = propose_time
        self._fixed_review_time = review_time

    def propose(self, initial_input, input_ids, k):
        result = super().propose(initial_input, input_ids, k)
        self.propose_wall_time.append(self._fixed_propose_time)
        return result

    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        result = super().review(initial_input, input_ids, probs, review_index, leniency=leniency)
        self.review_wall_time.append(self._fixed_review_time)
        return result


class FixedWindowTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.initial_input = torch.tensor([[100]], dtype=torch.long)
        self.item = {"answer": 1}
        self.cfg = {
            "dataset": "mmlu",
            "max_length": 10,
            "small_window": 3,
            "middle_window": 6,
            "stop_on_answer": True,
        }
        self.target = [11, 12, 13, 14, 15, 16]
        self.small_path = [11, 12, 13, 18, 19, 20]

    class CountingTokenizer(FakeTokenizer):
        def __init__(self):
            self.decode_calls = 0

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            self.decode_calls += 1
            return super().decode(
                ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

    def test_baseline_fixed_window_tracks_large_replacements(self):
        m_s = FakeModel(self.small_path, self.small_path)
        m_l = FakeModel(self.target, self.target)
        result = run_baseline_sample(
            cfg=self.cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_l=m_l,
            sample_index=0,
        )
        self.assertEqual([token["source_model"] for token in result["final_tokens"]], ["small", "small", "small", "large", "large", "large"])
        self.assertEqual(result["score"]["prediction"], "B")
        self.assertTrue(result["score"]["correct"])
        self.assertAlmostEqual(result["usage"]["edge_pass_rates"]["small_to_large"], 0.5)
        self.assertEqual(result["model_runtime"]["small"]["propose_calls"], 2)
        self.assertEqual(result["model_runtime"]["large"]["review_calls"], 2)

    def test_hierarchical_fixed_window_tracks_middle_replacements(self):
        m_s = FakeModel(self.small_path, self.small_path)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        result = run_hierarchical_sample(
            cfg=self.cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
        )
        self.assertEqual([token["source_model"] for token in result["final_tokens"]], ["small", "small", "small", "middle", "middle", "middle"])
        self.assertTrue(all(token["verified_by_large"] for token in result["final_tokens"]))
        self.assertAlmostEqual(result["usage"]["edge_pass_rates"]["small_to_middle"], 0.5)
        self.assertAlmostEqual(result["usage"]["edge_pass_rates"]["middle_to_large"], 1.0)
        self.assertEqual(result["mm_forward_calls"], 2)
        self.assertEqual(result["model_runtime"]["middle"]["review_calls"], 2)

    def test_double_layer_extends_with_middle_token(self):
        m_s = FakeModel(self.target, self.target)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        result = run_double_layer_sample(
            cfg=self.cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
        )
        first_cycle = result["blocks"][0]["middle_cycles"][0]["middle_result"]
        self.assertEqual(
            [token["source_model"] for token in first_cycle],
            ["small", "small", "small", "middle"],
        )
        self.assertEqual(first_cycle[-1]["token_id"], 14)
        self.assertIn(result["final_tokens"][-1]["source_model"], {"large", "middle"})

    def test_double_layer_filter_only_does_not_emit_middle_refill(self):
        cfg = {
            **self.cfg,
            "hierarchical_variant": "filter_only",
        }
        m_s = FakeModel(self.target, self.target)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        result = run_double_layer_sample(
            cfg=cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
        )
        first_cycle = result["blocks"][0]["middle_cycles"][0]["middle_result"]
        self.assertEqual(
            [token["source_model"] for token in first_cycle],
            ["small", "small", "small"],
        )
        self.assertEqual(result["middle_refill_policy"], "never")

    def test_double_layer_adaptive_window_keeps_small_fixed_by_default(self):
        cfg = {
            **self.cfg,
            "hierarchical_variant": "filter_only",
            "window_policy": "adaptive",
            "dynamic_small_window_min": 2,
            "dynamic_small_window_max": 4,
            "dynamic_middle_window_min": 4,
            "dynamic_middle_window_max": 8,
            "dynamic_acceptance_low": 0.6,
            "dynamic_acceptance_high": 0.8,
            "dynamic_window_step": 1,
        }
        m_s = FakeModel(self.small_path, self.small_path)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        result = run_double_layer_sample(
            cfg=cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
        )
        trace_models = {event["model"] for event in result["draft_window"]["trace"]}
        self.assertNotIn("small", trace_models)
        self.assertIn("middle", trace_models)
        self.assertFalse(result["draft_window"]["adaptive_models"]["small"])
        self.assertTrue(result["draft_window"]["adaptive_models"]["middle"])
        self.assertGreater(result["draft_window"]["totals"]["small"]["count"], 0)
        self.assertGreater(result["draft_window"]["totals"]["middle"]["count"], 0)
        self.assertEqual(result["draft_window"]["averages"]["small"], self.cfg["small_window"])

    def test_double_layer_can_opt_into_adaptive_small_window(self):
        cfg = {
            **self.cfg,
            "hierarchical_variant": "filter_only",
            "window_policy": "adaptive",
            "adapt_small_window": True,
            "dynamic_small_window_min": 2,
            "dynamic_small_window_max": 4,
            "dynamic_middle_window_min": 4,
            "dynamic_middle_window_max": 8,
            "dynamic_acceptance_low": 0.6,
            "dynamic_acceptance_high": 0.8,
            "dynamic_window_step": 1,
        }
        m_s = FakeModel(self.small_path, self.small_path)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        result = run_double_layer_sample(
            cfg=cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
        )
        trace_models = {event["model"] for event in result["draft_window"]["trace"]}
        self.assertIn("small", trace_models)
        self.assertIn("middle", trace_models)
        self.assertTrue(result["draft_window"]["adaptive_models"]["small"])
        self.assertTrue(result["draft_window"]["adaptive_models"]["middle"])

    def test_adaptive_middle_window_does_not_collapse_to_small_window(self):
        cfg = {
            **self.cfg,
            "hierarchical_variant": "filter_only",
            "window_policy": "adaptive",
            "dynamic_middle_window_min": 2,
            "dynamic_middle_window_max": 8,
            "dynamic_acceptance_low": 0.6,
            "dynamic_acceptance_high": 0.8,
            "dynamic_window_step": 1,
        }
        m_s = FakeModel(self.small_path, self.small_path)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        result = run_double_layer_sample(
            cfg=cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
        )
        middle_events = [
            event for event in result["draft_window"]["trace"] if event["model"] == "middle"
        ]
        self.assertTrue(middle_events)
        self.assertGreaterEqual(
            min(event["next_window"] for event in middle_events),
            cfg["small_window"] + 1,
        )

    def test_selective_route_can_bypass_middle_after_warmup(self):
        cfg = {
            **self.cfg,
            "hierarchical_variant": "selective_route",
            "window_policy": "utility",
            "stop_on_answer": False,
            "max_length": 12,
            "small_window": 3,
            "middle_window": 6,
            "selective_route_warmup_blocks": 1,
        }
        long_target = [18, 19, 20, 18, 19, 20, 18, 19, 20, 18, 19, 20]
        m_s = FakeModel(long_target, long_target)
        m_m = FakeModel(long_target, long_target)
        m_l = FakeModel(long_target, long_target)
        result = run_double_layer_sample(
            cfg=cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
            capture_trace=False,
        )
        self.assertEqual(result["routing"]["counts"]["middle"], 1)
        self.assertGreaterEqual(result["routing"]["counts"]["large"], 1)
        self.assertIsNotNone(result["usage"]["edge_pass_rates"]["small_to_large"])
        self.assertIsNotNone(result["usage"]["edge_pass_rates"]["small_to_middle"])

    def test_cost_aware_selective_route_does_not_chase_negative_middle_utility(self):
        cfg = {
            **self.cfg,
            "hierarchical_variant": "cost_aware_selective_route",
            "window_policy": "utility",
            "stop_on_answer": False,
            "max_length": 14,
            "small_window": 3,
            "middle_window": 4,
            "selective_route_warmup_blocks": 1,
            "selective_route_direct_acceptance_low": 0.7,
            "selective_route_middle_acceptance_low": 0.55,
        }
        target = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 11, 12, 13, 14]
        small_path = [11, 12, 13, 14, 18, 19, 20, 18, 19, 20, 18, 19, 20, 18]
        m_s = TimedFakeModel(small_path, small_path, propose_time=0.001)
        m_m = TimedFakeModel(target, target, review_time=0.5)
        m_l = TimedFakeModel(target, target, propose_time=0.001, review_time=0.01)
        result = run_double_layer_sample(
            cfg=cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
            capture_trace=False,
        )

        self.assertEqual(result["routing"]["counts"]["middle"], 1)
        self.assertGreaterEqual(result["routing"]["counts"]["large"], 2)
        self.assertIn(
            "middle_utility_negative",
            {event["reason"] for event in result["routing"]["trace"]},
        )

    def test_non_trace_samples_skip_trace_payload(self):
        m_s = FakeModel(self.small_path, self.small_path)
        m_l = FakeModel(self.target, self.target)
        result = run_baseline_sample(
            cfg=self.cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_l=m_l,
            sample_index=0,
            capture_trace=False,
        )
        self.assertNotIn("prompt", result)
        self.assertNotIn("final_tokens", result)
        self.assertNotIn("chunks", result)
        self.assertTrue(result["score"]["correct"])

    def test_non_trace_baseline_avoids_per_token_decoding(self):
        tokenizer = self.CountingTokenizer()
        cfg = {
            **self.cfg,
            "stop_on_answer": False,
        }
        m_s = FakeModel(self.small_path, self.small_path)
        m_l = FakeModel(self.target, self.target)
        result = run_baseline_sample(
            cfg=cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=tokenizer,
            m_s=m_s,
            m_l=m_l,
            sample_index=0,
            capture_trace=False,
        )
        self.assertEqual(tokenizer.decode_calls, 1)
        self.assertEqual(result["tokens_generated"], 6)
        self.assertTrue(result["score"]["correct"])

    def test_non_trace_hierarchical_matches_trace_summary(self):
        m_s = FakeModel(self.small_path, self.small_path)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        traced = run_hierarchical_sample(
            cfg=self.cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
            capture_trace=True,
        )

        m_s = FakeModel(self.small_path, self.small_path)
        m_m = FakeModel(self.target, self.target)
        m_l = FakeModel(self.target, self.target)
        fast = run_hierarchical_sample(
            cfg=self.cfg,
            item=self.item,
            initial_input=self.initial_input,
            tokenizer=self.tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=0,
            capture_trace=False,
        )

        self.assertEqual(fast["tokens_generated"], traced["tokens_generated"])
        self.assertEqual(fast["score"]["prediction"], traced["score"]["prediction"])
        self.assertEqual(fast["usage"]["final_source_counts"], traced["usage"]["final_source_counts"])
        self.assertEqual(fast["usage"]["edge_pass_rates"], traced["usage"]["edge_pass_rates"])

    def test_answer_extractors(self):
        self.assertEqual(extract_mmlu_choice("Reasoning... The answer is (C)."), "C")
        self.assertEqual(extract_mmlu_choice_strict("Reasoning... The answer is (C)."), "C")
        self.assertEqual(extract_gsm8k_answer("work ... #### 1,234"), "1234")
        self.assertEqual(extract_gsm8k_answer_strict("work ... #### 1,234"), "1234")
        self.assertTrue(has_final_answer_signal("mmlu", "Reasoning... The answer is (C)."))


if __name__ == "__main__":
    unittest.main()
