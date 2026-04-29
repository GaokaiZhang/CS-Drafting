window.DEMO_COMPARISON = {
  config: {
    mode: "compare_all",
    dataset: "mmlu",
    n_samples: 2,
    k_s: 3,
    k_m: 5,
    tau: 0.4,
    window_size: 20,
    max_length: 24,
    trace_samples: 1
  },
  runs: {
    baseline: {
      summary: {
        run_name: "baseline",
        tokens_per_sec: 18.4,
        avg_wall_time: 1.72,
        avg_tokens_generated: 12,
        benchmark_metric: "accuracy",
        benchmark_score: 0.5,
        avg_ml_forward_calls: 4.0,
        avg_mm_saved_positions: 0.0,
        avg_switch_count: 0.0,
        avg_mm_forward_calls: 0.0,
        avg_drafter_steps: { small: 4.0, middle: 0.0 },
        avg_draft_generated_tokens: { small: 12, middle: 0, large: 4 },
        avg_final_source_tokens: { small: 8, middle: 0, large: 4 },
        avg_middle_review_positions: 0,
        avg_large_review_positions: 12,
        avg_model_runtime: {
          small: { propose_calls: 4, review_calls: 0, propose_wall_time: 0.42, review_wall_time: 0, total_wall_time: 0.42, share: 0.24 },
          middle: { propose_calls: 0, review_calls: 0, propose_wall_time: 0, review_wall_time: 0, total_wall_time: 0, share: 0 },
          large: { propose_calls: 0, review_calls: 4, propose_wall_time: 0, review_wall_time: 1.30, total_wall_time: 1.30, share: 0.76 }
        },
        usage: {
          draft_generated_counts: {
            small: { count: 24, pct: 0.75 },
            middle: { count: 0, pct: 0.0 },
            large: { count: 8, pct: 0.25 }
          },
          final_source_counts: {
            small: { count: 16, pct: 0.67 },
            middle: { count: 0, pct: 0.0 },
            large: { count: 8, pct: 0.33 }
          },
          verification_positions: {
            small: { count: 0, pct: 0.0 },
            middle: { count: 0, pct: 0.0 },
            large: { count: 24, pct: 1.0 }
          },
          verification_calls: {
            small: { count: 0, pct: 0.0 },
            middle: { count: 0, pct: 0.0 },
            large: { count: 8, pct: 1.0 }
          },
          edge_pass_rates: {
            small_to_middle: null,
            middle_to_large: null,
            small_to_large: 0.67
          }
        }
      },
      sample_metrics: [
        {
          sample_index: 0,
          wall_time: 1.72,
          tokens_generated: 12,
          benchmark_score: 1.0,
          correct: true,
          prediction: "B",
          gold: "B"
        }
      ],
      trace_samples: [
        {
          sample_index: 0,
          prompt: "Question: Which model performs final verification in ACSD?\n(A) small (B) middle (C) large (D) none\nAnswer:\n",
          generated_text: " The answer is (C).",
          score: {
            metric_name: "accuracy",
            prediction: "C",
            gold: "C",
            correct: true,
            score: 1.0
          },
          wall_time: 1.72,
          tokens_generated: 7,
          ml_forward_calls: 3,
          mm_saved_positions: 0,
          switch_count: 0,
          alpha_trace: [],
          acceptance_trace: [],
          drafter_steps: { small: 3, middle: 0 },
          usage: {
            draft_generated_counts: {
              small: { count: 9, pct: 0.69 },
              middle: { count: 0, pct: 0.0 },
              large: { count: 4, pct: 0.31 }
            },
            final_source_counts: {
              small: { count: 5, pct: 0.71 },
              middle: { count: 0, pct: 0.0 },
              large: { count: 2, pct: 0.29 }
            },
            verification_positions: {
              small: { count: 0, pct: 0.0 },
              middle: { count: 0, pct: 0.0 },
              large: { count: 9, pct: 1.0 }
            },
            verification_calls: {
              small: { count: 0, pct: 0.0 },
              middle: { count: 0, pct: 0.0 },
              large: { count: 3, pct: 1.0 }
            },
            edge_pass_rates: {
              small_to_middle: null,
              middle_to_large: null,
              small_to_large: 0.67
            }
          },
          final_tokens: [
            { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 0 },
            { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 0 },
            { position: 2, token_id: 3, token_text: " is", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 0 },
            { position: 3, token_id: 4, token_text: " (", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 1 },
            { position: 4, token_id: 5, token_text: "C", source_model: "large", verified_by_middle: false, verified_by_large: false, step_index: 1 },
            { position: 5, token_id: 6, token_text: ")", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 2 },
            { position: 6, token_id: 7, token_text: ".", source_model: "large", verified_by_middle: false, verified_by_large: false, step_index: 2 }
          ],
          steps: [
            {
              step_index: 0,
              drafter: "small",
              next_drafter: "small",
              requested_tokens: 3,
              candidate_to_large: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              small_draft: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              final_step: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "small", verified_by_middle: false, verified_by_large: true, step_index: 0 }
              ],
              large_accepted_count: 3,
              large_generated_count: 0,
              model_runtime_delta: {
                small: { total_wall_time: 0.12 },
                middle: { total_wall_time: 0 },
                large: { total_wall_time: 0.41 }
              }
            }
          ]
        }
      ]
    },
    cascaded: {
      summary: {
        run_name: "cascaded",
        tokens_per_sec: 21.8,
        avg_wall_time: 1.44,
        avg_tokens_generated: 12,
        benchmark_metric: "accuracy",
        benchmark_score: 0.5,
        avg_ml_forward_calls: 3.0,
        avg_mm_saved_positions: 2.5,
        avg_switch_count: 0.0,
        avg_mm_forward_calls: 3.0,
        avg_drafter_steps: { small: 4.0, middle: 0.0 },
        avg_draft_generated_tokens: { small: 24, middle: 6, large: 4 },
        avg_final_source_tokens: { small: 14, middle: 6, large: 4 },
        avg_middle_review_positions: 12,
        avg_large_review_positions: 10,
        avg_model_runtime: {
          small: { propose_calls: 4, review_calls: 0, propose_wall_time: 0.42, review_wall_time: 0, total_wall_time: 0.42, share: 0.22 },
          middle: { propose_calls: 0, review_calls: 4, propose_wall_time: 0, review_wall_time: 0.58, total_wall_time: 0.58, share: 0.30 },
          large: { propose_calls: 0, review_calls: 4, propose_wall_time: 0, review_wall_time: 0.92, total_wall_time: 0.92, share: 0.48 }
        },
        usage: {
          draft_generated_counts: {
            small: { count: 24, pct: 0.71 },
            middle: { count: 6, pct: 0.18 },
            large: { count: 4, pct: 0.12 }
          },
          final_source_counts: {
            small: { count: 14, pct: 0.58 },
            middle: { count: 6, pct: 0.25 },
            large: { count: 4, pct: 0.17 }
          },
          verification_positions: {
            small: { count: 0, pct: 0.0 },
            middle: { count: 24, pct: 0.55 },
            large: { count: 20, pct: 0.45 }
          },
          verification_calls: {
            small: { count: 0, pct: 0.0 },
            middle: { count: 8, pct: 0.5 },
            large: { count: 8, pct: 0.5 }
          },
          edge_pass_rates: {
            small_to_middle: 0.75,
            middle_to_large: 0.90,
            small_to_large: null
          }
        }
      },
      sample_metrics: [
        {
          sample_index: 0,
          wall_time: 1.44,
          tokens_generated: 12,
          benchmark_score: 1.0,
          correct: true,
          prediction: "C",
          gold: "C"
        }
      ],
      trace_samples: [
        {
          sample_index: 0,
          prompt: "Question: Which model performs final verification in ACSD?\n(A) small (B) middle (C) large (D) none\nAnswer:\n",
          generated_text: " The answer is (C).",
          score: {
            metric_name: "accuracy",
            prediction: "C",
            gold: "C",
            correct: true,
            score: 1.0
          },
          wall_time: 1.44,
          tokens_generated: 7,
          ml_forward_calls: 2,
          mm_saved_positions: 2,
          switch_count: 0,
          alpha_trace: [],
          acceptance_trace: [],
          drafter_steps: { small: 2, middle: 0 },
          usage: {
            draft_generated_counts: {
              small: { count: 6, pct: 0.55 },
              middle: { count: 3, pct: 0.27 },
              large: { count: 2, pct: 0.18 }
            },
            final_source_counts: {
              small: { count: 3, pct: 0.43 },
              middle: { count: 3, pct: 0.43 },
              large: { count: 1, pct: 0.14 }
            },
            verification_positions: {
              small: { count: 0, pct: 0.0 },
              middle: { count: 6, pct: 0.55 },
              large: { count: 5, pct: 0.45 }
            },
            verification_calls: {
              small: { count: 0, pct: 0.0 },
              middle: { count: 2, pct: 0.5 },
              large: { count: 2, pct: 0.5 }
            },
            edge_pass_rates: {
              small_to_middle: 0.67,
              middle_to_large: 0.80,
              small_to_large: null
            }
          },
          final_tokens: [
            { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
            { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
            { position: 2, token_id: 3, token_text: " is", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
            { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 0 },
            { position: 4, token_id: 5, token_text: "C", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
            { position: 5, token_id: 6, token_text: ")", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
            { position: 6, token_id: 7, token_text: ".", source_model: "large", verified_by_middle: false, verified_by_large: false, step_index: 1 }
          ],
          steps: [
            {
              step_index: 0,
              drafter: "small",
              next_drafter: "small",
              requested_tokens: 3,
              small_draft: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 8, token_text: " maybe", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              middle_result: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: true, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              candidate_to_large: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: true, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              final_step: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
                { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 0 }
              ],
              middle_accepted_count: 2,
              middle_generated_count: 2,
              middle_saved_positions: 1,
              large_accepted_count: 4,
              large_generated_count: 0,
              model_runtime_delta: {
                small: { total_wall_time: 0.10 },
                middle: { total_wall_time: 0.15 },
                large: { total_wall_time: 0.23 }
              }
            }
          ]
        }
      ]
    },
    adaptive: {
      summary: {
        run_name: "adaptive",
        tokens_per_sec: 24.9,
        avg_wall_time: 1.26,
        avg_tokens_generated: 12,
        benchmark_metric: "accuracy",
        benchmark_score: 0.5,
        avg_ml_forward_calls: 2.5,
        avg_mm_saved_positions: 1.0,
        avg_switch_count: 1.0,
        avg_mm_forward_calls: 4.0,
        avg_drafter_steps: { small: 2.0, middle: 2.0 },
        avg_draft_generated_tokens: { small: 12, middle: 10, large: 4 },
        avg_final_source_tokens: { small: 6, middle: 10, large: 4 },
        avg_middle_review_positions: 6,
        avg_large_review_positions: 12,
        avg_model_runtime: {
          small: { propose_calls: 2, review_calls: 0, propose_wall_time: 0.21, review_wall_time: 0, total_wall_time: 0.21, share: 0.12 },
          middle: { propose_calls: 2, review_calls: 2, propose_wall_time: 0.44, review_wall_time: 0.27, total_wall_time: 0.71, share: 0.41 },
          large: { propose_calls: 0, review_calls: 4, propose_wall_time: 0, review_wall_time: 0.81, total_wall_time: 0.81, share: 0.47 }
        },
        usage: {
          draft_generated_counts: {
            small: { count: 12, pct: 0.46 },
            middle: { count: 10, pct: 0.38 },
            large: { count: 4, pct: 0.15 }
          },
          final_source_counts: {
            small: { count: 6, pct: 0.30 },
            middle: { count: 10, pct: 0.50 },
            large: { count: 4, pct: 0.20 }
          },
          verification_positions: {
            small: { count: 0, pct: 0.0 },
            middle: { count: 6, pct: 0.33 },
            large: { count: 12, pct: 0.67 }
          },
          verification_calls: {
            small: { count: 0, pct: 0.0 },
            middle: { count: 2, pct: 0.33 },
            large: { count: 4, pct: 0.67 }
          },
          edge_pass_rates: {
            small_to_middle: 0.67,
            middle_to_large: 0.92,
            small_to_large: null
          }
        }
      },
      sample_metrics: [
        {
          sample_index: 0,
          wall_time: 1.26,
          tokens_generated: 12,
          benchmark_score: 1.0,
          correct: true,
          prediction: "C",
          gold: "C"
        }
      ],
      trace_samples: [
        {
          sample_index: 0,
          prompt: "Question: Which model performs final verification in ACSD?\n(A) small (B) middle (C) large (D) none\nAnswer:\n",
          generated_text: " The answer is (C).",
          score: {
            metric_name: "accuracy",
            prediction: "C",
            gold: "C",
            correct: true,
            score: 1.0
          },
          wall_time: 1.26,
          tokens_generated: 7,
          ml_forward_calls: 2,
          mm_saved_positions: 1,
          switch_count: 1,
          alpha_trace: [0.33, 0.33],
          acceptance_trace: [0.33],
          drafter_steps: { small: 1, middle: 1 },
          usage: {
            draft_generated_counts: {
              small: { count: 3, pct: 0.27 },
              middle: { count: 6, pct: 0.55 },
              large: { count: 2, pct: 0.18 }
            },
            final_source_counts: {
              small: { count: 1, pct: 0.14 },
              middle: { count: 5, pct: 0.71 },
              large: { count: 1, pct: 0.14 }
            },
            verification_positions: {
              small: { count: 0, pct: 0.0 },
              middle: { count: 3, pct: 0.27 },
              large: { count: 8, pct: 0.73 }
            },
            verification_calls: {
              small: { count: 0, pct: 0.0 },
              middle: { count: 1, pct: 0.33 },
              large: { count: 2, pct: 0.67 }
            },
            edge_pass_rates: {
              small_to_middle: 0.33,
              middle_to_large: 0.88,
              small_to_large: null
            }
          },
          final_tokens: [
            { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
            { position: 1, token_id: 2, token_text: " answer", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 0 },
            { position: 2, token_id: 3, token_text: " is", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 0 },
            { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
            { position: 4, token_id: 5, token_text: "C", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
            { position: 5, token_id: 6, token_text: ")", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
            { position: 6, token_id: 7, token_text: ".", source_model: "large", verified_by_middle: false, verified_by_large: false, step_index: 1 }
          ],
          steps: [
            {
              step_index: 0,
              drafter: "small",
              next_drafter: "middle",
              requested_tokens: 3,
              small_draft: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 8, token_text: " maybe", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 9, token_text: "?", source_model: "small", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              middle_result: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              candidate_to_large: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: false, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 0 }
              ],
              final_step: [
                { position: 0, token_id: 1, token_text: " The", source_model: "small", verified_by_middle: true, verified_by_large: true, step_index: 0 },
                { position: 1, token_id: 2, token_text: " answer", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 0 },
                { position: 2, token_id: 3, token_text: " is", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 0 }
              ],
              middle_accepted_count: 1,
              middle_generated_count: 2,
              middle_saved_positions: 2,
              large_accepted_count: 3,
              large_generated_count: 0,
              alpha_before: 1.0,
              alpha_after: 0.33,
              switched: true,
              model_runtime_delta: {
                small: { total_wall_time: 0.10 },
                middle: { total_wall_time: 0.16 },
                large: { total_wall_time: 0.22 }
              }
            },
            {
              step_index: 1,
              drafter: "middle",
              next_drafter: "middle",
              requested_tokens: 3,
              middle_draft: [
                { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 1 },
                { position: 4, token_id: 5, token_text: "C", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 1 },
                { position: 5, token_id: 6, token_text: ")", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 1 }
              ],
              candidate_to_large: [
                { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 1 },
                { position: 4, token_id: 5, token_text: "C", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 1 },
                { position: 5, token_id: 6, token_text: ")", source_model: "middle", verified_by_middle: false, verified_by_large: false, step_index: 1 }
              ],
              final_step: [
                { position: 3, token_id: 4, token_text: " (", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
                { position: 4, token_id: 5, token_text: "C", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
                { position: 5, token_id: 6, token_text: ")", source_model: "middle", verified_by_middle: false, verified_by_large: true, step_index: 1 },
                { position: 6, token_id: 7, token_text: ".", source_model: "large", verified_by_middle: false, verified_by_large: false, step_index: 1 }
              ],
              large_accepted_count: 3,
              large_generated_count: 1,
              alpha_before: 0.33,
              alpha_after: 0.33,
              switched: false,
              model_runtime_delta: {
                small: { total_wall_time: 0.00 },
                middle: { total_wall_time: 0.20 },
                large: { total_wall_time: 0.24 }
              }
            }
          ]
        }
      ]
    }
  },
  comparisons: {
    cascaded_vs_baseline: {
      throughput_delta: 3.4,
      throughput_speedup: 1.185,
      benchmark_delta: 0.0,
      avg_wall_time_delta: -0.28,
      avg_ml_forward_calls_delta: -1.0,
      avg_mm_saved_positions_delta: 2.5,
      avg_switch_count_delta: 0.0,
      candidate_pass_rates: {
        small_to_middle: 0.75,
        middle_to_large: 0.90,
        small_to_large: null
      },
      baseline_pass_rates: {
        small_to_middle: null,
        middle_to_large: null,
        small_to_large: 0.67
      }
    },
    adaptive_vs_baseline: {
      throughput_delta: 6.5,
      throughput_speedup: 1.353,
      benchmark_delta: 0.0,
      avg_wall_time_delta: -0.46,
      avg_ml_forward_calls_delta: -1.5,
      avg_mm_saved_positions_delta: 1.0,
      avg_switch_count_delta: 1.0,
      candidate_pass_rates: {
        small_to_middle: 0.67,
        middle_to_large: 0.92,
        small_to_large: null
      },
      baseline_pass_rates: {
        small_to_middle: null,
        middle_to_large: null,
        small_to_large: 0.67
      }
    }
  }
};
