# Claude Design Poster Brief: Three-Tier Speculative Decoding

## Project Metadata

Title: Middle-Model Verification for Three-Tier Speculative Decoding

Subtitle: A Systems Evaluation on Reasoning-Oriented Generation

Course: MLSys Course Project, April 2026

Team: Zichu Wu, Gaokai Zhang, Melody Yin

## One-Sentence Thesis

A middle verifier can improve reasoning quality, but it is not a free speedup: on GSM8K it improves exact-match accuracy, while on MMLU it mostly adds overhead because the task is short and multiple-choice.

## Main Claim To Emphasize

Three-tier speculative decoding should be presented as a routing and quality-filtering policy, not as a universal throughput improvement.

Use this exact wording prominently:

> Middle-layer verification improves output quality on reasoning-heavy generation, and selective routing is the clearest completed way to make that benefit practical.

## Visual Story

The poster should tell a before/after systems story:

1. Standard speculative decoding uses a small model to draft and a large model to verify.
2. We insert a middle model: Qwen2.5-0.5B -> Qwen2.5-1.5B -> Qwen2.5-14B.
3. The middle model filters weak drafts before final large-model verification.
4. This helps on GSM8K because reasoning trajectories matter.
5. It does not help much on MMLU because answers are short and constrained.
6. Cost-aware routing can reduce middle-model calls, but if it bypasses too aggressively it removes the GSM8K quality gain.

## Recommended Layout

Use a 3-column scientific poster layout.

Top band:
- Title and subtitle.
- Team names: Zichu Wu, Gaokai Zhang, Melody Yin.
- One-sentence takeaway in a highlighted callout.

Left column:
- Motivation.
- Two-layer vs three-layer decoding diagram.
- Model stack table.
- Method variants.

Middle column:
- GSM8K result table.
- Cost-aware routing diagnostic.
- Explanation for why GSM8K benefits.

Right column:
- MMLU boundary result table.
- Throughput caveat for Modal shard repair.
- System/reliability contribution.
- Final conclusion.

## Core Diagram

Create a clean pipeline diagram:

```text
Prompt / accepted prefix
        |
        v
Qwen2.5-0.5B small drafter
        |
        v
Qwen2.5-1.5B middle verifier / router
        |
        v
Qwen2.5-14B large final verifier
        |
        v
accepted tokens + corrective token
```

Next to it, include the two-layer baseline:

```text
small drafter -> large verifier
```

## Method Details To Include

Model stack:

| Tier | Model | Role |
| --- | --- | --- |
| Small | Qwen2.5-0.5B-Instruct | Draft candidate tokens |
| Middle | Qwen2.5-1.5B-Instruct | Verify, filter, and route drafts |
| Large | Qwen2.5-14B-Instruct | Final verifier and answer source |

Final methods:

| Method | Meaning |
| --- | --- |
| `baseline_sw5` | Tuned two-layer baseline with small window 5 |
| `fixed_cascade_sw4_mw6` | Always use middle verification, small window 4, middle window 6 |
| `adaptive_sw4_mw6` | Utility-based dynamic draft windows |
| `selective_sw4_mw6` | Route through the middle only when recent utility favors it |
| `cost_aware_selective_sw4_mw6` | Bypass middle when measured utility or acceptance is too low |

## GSM8K Result

Use this as the main positive result.

| Method | Samples | Tokens/sec | Exact match | Score gain |
| --- | ---: | ---: | ---: | ---: |
| `baseline_sw5` | 1319 | 33.716 | 0.3541 | reference |
| `fixed_cascade_sw4_mw6` | 1319 | 30.993 | 0.3874 | +0.0334 |
| `adaptive_sw4_mw6` | 1319 | 31.881 | 0.3723 | +0.0182 |
| `selective_sw4_mw6` | 1319 | 32.347 | 0.3753 | +0.0212 |
| `cost_aware_selective_sw4_mw6` | 1319 | 31.331 | 0.3541 | +0.0000 |

Caption:

> On GSM8K, fixed, adaptive, and selective middle verification improve exact-match accuracy over the tuned two-layer baseline. Selective routing is the best completed speed-quality compromise among methods that improve score.

## Paired GSM8K Result

This is useful as a secondary callout:

| Method | Samples | Tokens/sec | Exact match |
| --- | ---: | ---: | ---: |
| Two-layer baseline | 1319 | 29.677 | 0.3010 |
| Three-layer hierarchy | 1319 | 30.052 | 0.3874 |

Caption:

> In an earlier paired full-dataset run, the hierarchy improved GSM8K exact match by 8.64 percentage points while preserving throughput. After tuning the two-layer baseline, the final story becomes a quality-speed tradeoff rather than a free speedup.

## Cost-Aware Routing Result

Use this as a diagnostic, not as the headline positive result.

| Dataset | Method | Samples | Tokens/sec | Score | Middle route |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | `selective_sw4_mw6` | 1319 | 32.347 | 0.3753 | 93.6% |
| GSM8K | `cost_aware_selective_sw4_mw6` | 1319 | 31.331 | 0.3541 | 3.3% |
| MMLU | `selective_sw4_mw6` | 14042 | 19.946 | 0.3257 | 92.9% |
| MMLU | `cost_aware_selective_sw4_mw6` | 14042 | 23.552 | 0.3242 | 7.5% |

Caption:

> Cost-aware routing can remove middle-model work, but calibration matters. On GSM8K it bypassed the middle verifier too often and removed the quality gain. On MMLU it recovered throughput, but there was little quality gain to preserve.

## MMLU Boundary Result

Use this as the main boundary/counterexample.

| Method | Samples | Tokens/sec | Accuracy | Accuracy gain |
| --- | ---: | ---: | ---: | ---: |
| `baseline_sw5` | 14042 | 24.864 | 0.3243 | reference |
| `fixed_cascade_sw4_mw6` | 14042 | 18.692 | 0.3256 | +0.0013 |
| `adaptive_sw4_mw6` | 14042 | 19.655 | 0.3255 | +0.0011 |
| `selective_sw4_mw6` | 14042 | 19.946 | 0.3257 | +0.0014 |
| `cost_aware_selective_sw4_mw6` | 14042 | 23.552 | 0.3242 | -0.0001 |

Caption:

> MMLU is short and multiple-choice. The middle verifier has little generation trajectory to repair, so accuracy is effectively unchanged. Cost-aware routing recovers much of the throughput loss by bypassing the middle model.

## Required Throughput Caveat

Include a small caution box near the MMLU table:

> Throughput caveat: the final MMLU aggregate includes one Modal-repaired shard. That shard ran on a single A100-80GB worker with all models on one GPU, while the original Slurm shards used two L40S GPUs. The MMLU accuracy is full-dataset and validated; the MMLU throughput should be interpreted with this mixed-hardware caveat.

Optional sensitivity note:

> Excluding only the Modal-repaired shard, MMLU throughput is 25.598 tok/s for `baseline_sw5`, 20.878 tok/s for `selective_sw4_mw6`, and 24.226 tok/s for `cost_aware_selective_sw4_mw6`. The conclusion does not change.

## System Contributions

Include a compact system contribution box:

| Workstream | Contribution |
| --- | --- |
| Reproduction | Re-ran CS-Drafting baselines and inspected cost counters |
| Scoring | Added GSM8K numeric extraction and MMLU choice extraction |
| ACSD prototype | Inserted a middle verifier and tested proxy stopping |
| Fixed-window harness | Built final Qwen 0.5B -> 1.5B -> 14B benchmark |
| Routing | Implemented adaptive, selective, and cost-aware policies |
| Reliability | Added strict shard validation, repair jobs, and Modal subshards |
| Debugging | Fixed vocab-size mismatch, token-id validation, and cross-GPU tensor movement |

## What Not To Claim

Do not claim:
- Three-tier speculative decoding is universally faster.
- Cost-aware routing solved the speed-quality tradeoff.
- MMLU shows a meaningful accuracy gain.
- MMLU throughput is a perfectly homogeneous two-L40S measurement.

Do claim:
- Middle verification improves GSM8K answer quality.
- Selective routing is the strongest completed practical variant.
- Cost-aware routing needs calibration.
- The middle tier is most compelling when generation trajectory affects the final answer.

## Copy-Ready Final Conclusion

> A middle model can improve reasoning quality, but it must be scheduled carefully. In our Qwen 0.5B -> 1.5B -> 14B stack, fixed, adaptive, and selective middle verification improve GSM8K exact match over the tuned two-layer baseline. MMLU provides the boundary case: the answer format is short, so middle verification adds overhead without meaningful accuracy gain. The practical lesson is to treat the middle model as a selective quality filter, not an always-on verifier.

## Suggested Visual Style

Use a clean systems-poster style:
- White or very light background.
- Dark text.
- One accent color for the middle-verifier path.
- Use arrows and model-size labels prominently.
- Highlight GSM8K positive result in green or blue.
- Highlight MMLU/cost-aware caveat in amber or neutral gray.
- Avoid making the poster look like a marketing page; it should read like an MLSys systems result.
