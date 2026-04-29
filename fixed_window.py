import copy
import math
import time

import torch

from benchmark import has_final_answer_signal, score_sample


MODEL_KEYS = ("small", "middle", "large")
EDGE_KEYS = ("small_to_middle", "middle_to_large", "small_to_large")
DOUBLE_LAYER_PROXY_VARIANTS = (
    "proxy_entropy",
    "proxy_top1",
    "proxy_margin",
    "proxy_mavg",
)
SELECTIVE_ROUTING_VARIANTS = (
    "selective_route",
    "selective_route_refill_on_full_accept",
)
COST_AWARE_SELECTIVE_ROUTING_VARIANTS = (
    "cost_aware_selective_route",
    "cost_aware_selective_route_refill_on_full_accept",
)
DOUBLE_LAYER_VARIANTS = (
    "double_layer",
    "filter_only",
    "double_layer_filter_only",
    "refill_on_full_accept",
    "double_layer_refill_on_full_accept",
    "confidence_refill",
    "double_layer_confidence_refill",
    *DOUBLE_LAYER_PROXY_VARIANTS,
    *SELECTIVE_ROUTING_VARIANTS,
    *COST_AWARE_SELECTIVE_ROUTING_VARIANTS,
)


def _eos_token(model):
    return 2 if "t5" not in model.name.lower() else 1


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


def reset_model_cache(model):
    if hasattr(model, "past_key_values"):
        model.past_key_values = None
    if hasattr(model, "past_ids"):
        model.past_ids = None


def reset_model_runtime(model):
    if hasattr(model, "forward_count"):
        model.forward_count = 0
    if hasattr(model, "propose_count"):
        model.propose_count = 0
    if hasattr(model, "review_count"):
        model.review_count = 0
    if hasattr(model, "wall_time"):
        model.wall_time = []
    if hasattr(model, "propose_wall_time"):
        model.propose_wall_time = []
    if hasattr(model, "review_wall_time"):
        model.review_wall_time = []
    reset_model_cache(model)


def _double_layer_refill_policy(cfg):
    variant = cfg.get("hierarchical_variant", "double_layer")
    if variant in ("filter_only", "double_layer_filter_only"):
        return "never"
    if variant in ("refill_on_full_accept", "double_layer_refill_on_full_accept"):
        return "on_full_accept"
    if variant == "selective_route_refill_on_full_accept":
        return "on_full_accept"
    if variant == "cost_aware_selective_route_refill_on_full_accept":
        return "on_full_accept"
    if variant in ("confidence_refill", "double_layer_confidence_refill"):
        # Until logits/probs are plumbed through the middle review, use full
        # acceptance as the conservative confidence gate.
        return "on_full_accept"
    return "always"


def _selective_routing_enabled(cfg):
    return cfg.get("hierarchical_variant", "double_layer") in (
        *SELECTIVE_ROUTING_VARIANTS,
        *COST_AWARE_SELECTIVE_ROUTING_VARIANTS,
    )


def _cost_aware_selective_routing_enabled(cfg):
    return (
        cfg.get("hierarchical_variant", "double_layer")
        in COST_AWARE_SELECTIVE_ROUTING_VARIANTS
    )


def _window_policy(cfg):
    return cfg.get("window_policy", cfg.get("draft_window_policy", "fixed"))


def _adaptive_windows_enabled(cfg):
    return _window_policy(cfg) in ("adaptive", "acceptance", "utility")


def _adaptive_model_enabled(cfg, model_key):
    if not _adaptive_windows_enabled(cfg):
        return False
    if model_key == "small":
        return bool(cfg.get("adapt_small_window", False))
    return model_key == "middle"


def _window_bounds(cfg, model_key):
    if model_key == "middle":
        base = int(cfg["middle_window"])
        min_key = "dynamic_middle_window_min"
        max_key = "dynamic_middle_window_max"
    else:
        base = int(cfg["small_window"])
        min_key = "dynamic_small_window_min"
        max_key = "dynamic_small_window_max"
    min_raw = cfg.get(min_key)
    max_raw = cfg.get(max_key)
    min_value = int(min_raw) if min_raw is not None else max(1, base // 2)
    max_value = int(max_raw) if max_raw is not None else max(base, base * 2)
    if model_key == "middle":
        min_value = max(min_value, int(cfg.get("small_window", 1)) + 1)
    min_value = max(1, min_value)
    max_value = max(min_value, max_value)
    return min_value, max_value


def _init_window_state(cfg):
    global_window_state = cfg.setdefault(
        "_global_window_state",
        {
            "current": {
                "small": int(cfg["small_window"]),
                "middle": int(cfg["middle_window"]),
            },
            "utility": {
                "small": {},
                "middle": {},
            },
        },
    )
    return {
        "policy": _window_policy(cfg) if _adaptive_windows_enabled(cfg) else "fixed",
        "current": {
            "small": int(global_window_state["current"]["small"]),
            "middle": int(global_window_state["current"]["middle"]),
        },
        "totals": {
            "small": {"sum": 0, "count": 0},
            "middle": {"sum": 0, "count": 0},
        },
        "trace": [],
        "change_count": 0,
        "global": global_window_state,
    }


def _request_window(cfg, window_state, model_key, remaining):
    if _adaptive_model_enabled(cfg, model_key):
        min_value, max_value = _window_bounds(cfg, model_key)
        window_state["current"][model_key] = max(
            min_value,
            min(int(window_state["current"][model_key]), max_value),
        )
        base = window_state["current"][model_key]
    else:
        base = cfg["middle_window"] if model_key == "middle" else cfg["small_window"]
    requested = max(1, min(int(base), int(remaining)))
    totals = window_state["totals"][model_key]
    totals["sum"] += requested
    totals["count"] += 1
    return requested


def _update_window(cfg, window_state, model_key, acceptance_ratio, step_index):
    if not _adaptive_model_enabled(cfg, model_key):
        return
    low = float(cfg.get("dynamic_acceptance_low", 0.7))
    high = float(cfg.get("dynamic_acceptance_high", 0.9))
    step = max(1, int(cfg.get("dynamic_window_step", 1)))
    min_value, max_value = _window_bounds(cfg, model_key)
    previous = max(min_value, min(int(window_state["current"][model_key]), max_value))
    updated = previous
    policy = _window_policy(cfg)
    utility_score = cfg.get("_last_window_utility")
    utility_margin = float(cfg.get("dynamic_utility_margin", 0.0))
    global_state = window_state.get("global")
    global_utility = (
        global_state.get("utility", {}).setdefault(model_key, {})
        if global_state is not None
        else None
    )
    if policy == "utility" and model_key == "middle" and utility_score is not None:
        if global_utility is not None:
            stats = global_utility.setdefault(
                str(previous),
                {"sum": 0.0, "count": 0},
            )
            stats["sum"] += float(utility_score)
            stats["count"] += 1
        if acceptance_ratio <= low or utility_score < -utility_margin:
            updated = max(min_value, previous - step)
        elif acceptance_ratio >= high and utility_score > utility_margin:
            updated = min(max_value, previous + step)
        if global_utility:
            best_window = None
            best_avg = None
            best_count = None
            for window_str, stats in global_utility.items():
                count = int(stats.get("count", 0))
                if count < 3:
                    continue
                window = int(window_str)
                if window < min_value or window > max_value:
                    continue
                avg = float(stats["sum"]) / count
                if (
                    best_avg is None
                    or avg > best_avg
                    or (avg == best_avg and count > best_count)
                ):
                    best_window = window
                    best_avg = avg
                    best_count = count
            if best_window is not None and best_avg is not None and best_avg > utility_margin:
                updated = best_window
    elif acceptance_ratio >= high:
        updated = min(max_value, previous + step)
    elif acceptance_ratio <= low:
        updated = max(min_value, previous - step)
    window_state["current"][model_key] = updated
    if global_state is not None:
        global_state["current"][model_key] = updated
    if updated != previous:
        window_state["change_count"] += 1
    window_state["trace"].append(
        {
            "step_index": int(step_index),
            "model": model_key,
            "acceptance_ratio": float(acceptance_ratio),
            "previous_window": int(previous),
            "next_window": int(updated),
            "changed": updated != previous,
            "utility_score": (
                float(utility_score)
                if policy == "utility" and model_key == "middle" and utility_score is not None
                else None
            ),
        }
    )


def _window_payload(window_state):
    averages = {}
    for model_key, totals in window_state["totals"].items():
        count = totals["count"]
        averages[model_key] = totals["sum"] / count if count else 0.0
    return {
        "policy": window_state["policy"],
        "adaptive_models": {
            model_key: any(event["model"] == model_key for event in window_state["trace"])
            for model_key in ("small", "middle")
        },
        "averages": averages,
        "totals": window_state["totals"],
        "change_count": window_state["change_count"],
        "trace": window_state["trace"],
    }


def _init_route_state(cfg):
    if not _selective_routing_enabled(cfg):
        return None
    return {
        "counts": {"middle": 0, "large": 0},
        "change_count": 0,
        "last_route": None,
        "history": {
            "middle_utility": [],
            "middle_acceptance": [],
            "direct_acceptance": [],
        },
        "trace": [],
    }


def _recent_average(values, window):
    if not values:
        return None
    recent = values[-max(1, int(window)) :]
    return sum(recent) / len(recent)


def _select_route(cfg, route_state, block_index):
    if route_state is None:
        return "middle", "disabled"

    warmup_blocks = int(cfg.get("selective_route_warmup_blocks", 1))
    history_window = int(cfg.get("selective_route_history_window", 4))
    utility_margin = float(
        cfg.get("selective_route_utility_margin", cfg.get("dynamic_utility_margin", 0.0))
    )
    direct_acceptance_low = float(cfg.get("selective_route_direct_acceptance_low", 0.7))
    direct_acceptance_high = float(cfg.get("selective_route_direct_acceptance_high", 0.85))
    middle_acceptance_low_cfg = cfg.get("selective_route_middle_acceptance_low")
    middle_acceptance_low = (
        0.55 if middle_acceptance_low_cfg is None else float(middle_acceptance_low_cfg)
    )
    probe_interval = int(cfg.get("selective_route_probe_interval", 0) or 0)

    if block_index < warmup_blocks:
        return "middle", "warmup"

    middle_utility = _recent_average(
        route_state["history"]["middle_utility"],
        history_window,
    )
    middle_acceptance = _recent_average(
        route_state["history"]["middle_acceptance"],
        history_window,
    )
    direct_acceptance = _recent_average(
        route_state["history"]["direct_acceptance"],
        history_window,
    )

    if _cost_aware_selective_routing_enabled(cfg):
        if (
            probe_interval > 0
            and route_state["counts"]["large"] > 0
            and (block_index - warmup_blocks) % probe_interval == 0
        ):
            return "middle", "periodic_middle_probe"

        if middle_utility is None:
            return "middle", "no_middle_history"

        if middle_utility < -utility_margin:
            if direct_acceptance is None:
                return "large", "measure_direct_after_negative_middle_utility"
            return "large", "middle_utility_negative"

        if (
            middle_acceptance is not None
            and middle_acceptance < middle_acceptance_low
        ):
            if direct_acceptance is None:
                return "large", "measure_direct_after_low_middle_acceptance"
            return "large", "middle_acceptance_low"

        if direct_acceptance is not None and direct_acceptance >= direct_acceptance_high:
            return "large", "direct_acceptance_high"

        if middle_utility > utility_margin:
            return "middle", "middle_utility_positive"

        if direct_acceptance is not None and direct_acceptance < direct_acceptance_low:
            return "middle", "direct_acceptance_low_with_nonnegative_middle_utility"

        return "large", "middle_utility_nonpositive"

    if direct_acceptance is not None and direct_acceptance < direct_acceptance_low:
        return "middle", "direct_acceptance_low"

    if middle_utility is None:
        return "middle", "no_middle_history"

    if middle_utility > utility_margin:
        return "middle", "middle_utility_positive"

    if direct_acceptance is None or direct_acceptance >= direct_acceptance_low:
        if direct_acceptance is not None and direct_acceptance >= direct_acceptance_high:
            return "large", "direct_acceptance_high"
        return "large", "middle_utility_nonpositive"

    return "middle", "fallback_middle"


def _record_route_outcome(
    route_state,
    block_index,
    route,
    reason,
    acceptance_ratio,
    utility_score,
    tokens_generated,
):
    if route_state is None:
        return
    last_route = route_state.get("last_route")
    if last_route is not None and last_route != route:
        route_state["change_count"] += 1
    route_state["last_route"] = route
    route_state["counts"][route] += 1
    if route == "middle":
        route_state["history"]["middle_acceptance"].append(float(acceptance_ratio))
        if utility_score is not None:
            route_state["history"]["middle_utility"].append(float(utility_score))
    else:
        route_state["history"]["direct_acceptance"].append(float(acceptance_ratio))
    route_state["trace"].append(
        {
            "block_index": int(block_index),
            "route": route,
            "reason": reason,
            "acceptance_ratio": float(acceptance_ratio),
            "utility_score": float(utility_score) if utility_score is not None else None,
            "tokens_generated": int(tokens_generated),
        }
    )


def _route_payload(route_state):
    if route_state is None:
        return None
    total = sum(route_state["counts"].values())
    return {
        "counts": dict(route_state["counts"]),
        "pct": {
            key: (value / total) if total else 0.0
            for key, value in route_state["counts"].items()
        },
        "change_count": int(route_state["change_count"]),
        "trace": list(route_state["trace"]),
    }


def _run_direct_large_block_fast(
    cfg,
    window_state,
    initial_input,
    current_small_ids,
    current_large_ids,
    m_s,
    m_l,
    stats,
    block_target,
    eos_token,
    proxy_type=None,
    proxy_threshold=None,
    mavg_window=5,
    block_index=0,
):
    block_final_ids = empty_ids(m_l.device)
    block_sources = []
    cycle_index = 0
    accepted_total = 0
    proposed_total = 0
    while block_final_ids.shape[-1] < block_target:
        remaining_in_block = block_target - block_final_ids.shape[-1]
        chunk_target = _request_window(
            cfg,
            window_state,
            "small",
            remaining_in_block,
        )
        small_prefix = _concat_ids(current_small_ids, block_final_ids, m_s.device)
        if proxy_type is None:
            small_ids, small_sources = propose_ids(
                proposer=m_s,
                initial_input=initial_input,
                prefix_ids=small_prefix,
                k=chunk_target,
                source_model="small",
                stats=stats,
            )
        else:
            small_ids, small_sources = propose_ids_with_proxy(
                proposer=m_s,
                initial_input=initial_input,
                prefix_ids=small_prefix,
                k=chunk_target,
                proxy_type=proxy_type,
                threshold=proxy_threshold,
                mavg_window=mavg_window,
                source_model="small",
                stats=stats,
            )

        large_prefix = _concat_ids(current_large_ids, block_final_ids, m_l.device)
        final_chunk_ids, final_chunk_sources, large_meta = verify_fixed_window_ids(
            verifier=m_l,
            verifier_key="large",
            edge_key="small_to_large",
            initial_input=initial_input,
            prefix_ids=large_prefix,
            candidate_ids=small_ids,
            candidate_sources=small_sources,
            stats=stats,
        )
        direct_acceptance_ratio = (
            large_meta["accepted_count"] / large_meta["candidate_len"]
            if large_meta["candidate_len"]
            else 1.0
        )
        _update_window(
            cfg,
            window_state,
            "small",
            direct_acceptance_ratio,
            block_index * 1000 + cycle_index,
        )
        accepted_total += large_meta["accepted_count"]
        proposed_total += large_meta["candidate_len"]
        final_chunk_ids = final_chunk_ids[:, :remaining_in_block]
        final_chunk_sources = _take_segments_prefix(final_chunk_sources, remaining_in_block)
        block_final_ids = _append_ids(block_final_ids, final_chunk_ids)
        _extend_segments(block_sources, final_chunk_sources)
        if final_chunk_ids.shape[-1] == 0:
            break
        if int(final_chunk_ids[0, -1].item()) == eos_token:
            break
        cycle_index += 1

    return block_final_ids, block_sources, {
        "accepted_count": accepted_total,
        "candidate_len": proposed_total,
        "acceptance_ratio": (accepted_total / proposed_total) if proposed_total else 1.0,
    }


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


def _runtime_sum(model, attr):
    values = getattr(model, attr, None)
    if not values:
        return 0.0
    return float(sum(values))


def _runtime_count(model, attr):
    values = getattr(model, attr, None)
    return len(values) if values else 0


def _runtime_checkpoint(m_s, m_m, m_l):
    return {
        "small": {
            "propose_wall_time": _runtime_sum(m_s, "propose_wall_time"),
            "propose_calls": getattr(m_s, "propose_count", 0),
        },
        "middle": {
            "review_wall_time": _runtime_sum(m_m, "review_wall_time"),
            "review_calls": getattr(m_m, "review_count", 0),
        },
        "large": {
            "review_wall_time": _runtime_sum(m_l, "review_wall_time"),
            "review_calls": getattr(m_l, "review_count", 0),
        },
    }


def _avg_runtime(model, attr):
    count = _runtime_count(model, attr)
    return _runtime_sum(model, attr) / count if count else 0.0


def _estimate_middle_window_utility(cfg, checkpoint, m_s, m_m, m_l, block_candidate_len):
    if block_candidate_len <= 0:
        return 0.0

    small_window = max(1, int(cfg["small_window"]))
    baseline_large_reviews = max(1, math.ceil(block_candidate_len / small_window))
    baseline_small_calls = max(1, math.ceil(block_candidate_len / small_window))

    small_time = _runtime_sum(m_s, "propose_wall_time") - checkpoint["small"]["propose_wall_time"]
    small_calls = getattr(m_s, "propose_count", 0) - checkpoint["small"]["propose_calls"]
    middle_time = _runtime_sum(m_m, "review_wall_time") - checkpoint["middle"]["review_wall_time"]
    large_time = _runtime_sum(m_l, "review_wall_time") - checkpoint["large"]["review_wall_time"]

    small_call_time = _avg_runtime(m_s, "propose_wall_time")
    large_review_time = _avg_runtime(m_l, "review_wall_time")
    if small_call_time <= 0.0 and small_calls > 0:
        small_call_time = small_time / small_calls
    if large_review_time <= 0.0:
        large_review_calls = getattr(m_l, "review_count", 0) - checkpoint["large"]["review_calls"]
        if large_review_calls > 0:
            large_review_time = large_time / large_review_calls

    expected_small_time = baseline_small_calls * small_call_time
    added_small_time = max(0.0, small_time - expected_small_time)
    saved_large_time = max(0.0, baseline_large_reviews * large_review_time - large_time)
    return saved_large_time - (added_small_time + middle_time)


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


def empty_ids(device):
    return torch.empty((1, 0), dtype=torch.long, device=device)


def _move_token_ids(token_ids, device):
    target = torch.device(device)
    if (
        torch.is_tensor(token_ids)
        and token_ids.device.type == "cuda"
        and target.type == "cuda"
        and token_ids.device != target
        and not torch.is_floating_point(token_ids)
    ):
        return token_ids.detach().cpu().to(target)
    return token_ids.to(target)


def make_records(token_ids, tokenizer, source_model, positions, block_index, cycle_index=None):
    records = []
    for idx, token_id in enumerate(token_ids):
        records.append(
            {
                "position": int(positions[idx]),
                "token_id": int(token_id),
                "token_text": decode_token(tokenizer, int(token_id)),
                "source_model": source_model,
                "verified_by_middle": False,
                "verified_by_large": False,
                "large_block_index": block_index,
                "middle_cycle_index": cycle_index,
            }
        )
    return records


def _current_sequence_ids(prefix_ids, pending_records, device):
    pending_ids = ids_from_records(pending_records, device)
    return torch.cat([_move_token_ids(prefix_ids, device), pending_ids], dim=1)


def _concat_ids(prefix_ids, suffix_ids, device):
    if suffix_ids is None or suffix_ids.shape[-1] == 0:
        return _move_token_ids(prefix_ids, device)
    return torch.cat(
        [_move_token_ids(prefix_ids, device), _move_token_ids(suffix_ids, device)],
        dim=1,
    )


def _append_ids(prefix_ids, suffix_ids):
    if suffix_ids is None or suffix_ids.shape[-1] == 0:
        return prefix_ids
    return torch.cat([prefix_ids, _move_token_ids(suffix_ids, prefix_ids.device)], dim=1)


def _append_segment(segments, source_model, count):
    if count <= 0:
        return
    if segments and segments[-1][0] == source_model:
        segments[-1] = (source_model, segments[-1][1] + count)
    else:
        segments.append((source_model, count))


def _extend_segments(dst, src):
    for source_model, count in src:
        _append_segment(dst, source_model, count)


def _take_segments_prefix(segments, limit):
    out = []
    remaining = limit
    for source_model, count in segments:
        if remaining <= 0:
            break
        take = min(count, remaining)
        _append_segment(out, source_model, take)
        remaining -= take
    return out


def _segment_counts(segments):
    counts = _new_counter()
    for source_model, count in segments:
        counts[source_model] += count
    return counts


def _add_counts(counter, updates):
    for key, value in updates.items():
        counter[key] += value


def _score_and_finalize_ids(dataset_name, item, tokenizer, generated_ids):
    generated_text = tokenizer.decode(
        generated_ids[0].detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    score = score_sample(dataset_name, item, generated_text)
    return generated_text, score


def should_stop_on_ids(cfg, tokenizer, generated_ids):
    if not cfg.get("stop_on_answer", True) or generated_ids.shape[-1] == 0:
        return False
    generated_text = tokenizer.decode(
        generated_ids[0].detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return has_final_answer_signal(cfg["dataset"], generated_text)


def propose_records(
    proposer,
    initial_input,
    prefix_ids,
    k,
    tokenizer,
    source_model,
    positions,
    block_index,
    cycle_index,
    stats,
):
    if k <= 0:
        return []
    start_len = prefix_ids.shape[-1]
    proposed_ids = proposer.propose(
        _move_token_ids(initial_input, proposer.device),
        _move_token_ids(prefix_ids, proposer.device),
        k,
    )
    new_tokens = proposed_ids[:, start_len:].detach().cpu().tolist()[0]
    stats["draft_generated_counts"][source_model] += len(new_tokens)
    return make_records(
        token_ids=new_tokens,
        tokenizer=tokenizer,
        source_model=source_model,
        positions=positions,
        block_index=block_index,
        cycle_index=cycle_index,
    )


def propose_ids(
    proposer,
    initial_input,
    prefix_ids,
    k,
    source_model,
    stats,
):
    if k <= 0:
        return empty_ids(proposer.device), []
    start_len = prefix_ids.shape[-1]
    proposed_ids = proposer.propose(
        _move_token_ids(initial_input, proposer.device),
        _move_token_ids(prefix_ids, proposer.device),
        k,
    )
    new_ids = proposed_ids[:, start_len:].detach()
    new_count = new_ids.shape[-1]
    stats["draft_generated_counts"][source_model] += new_count
    segments = []
    _append_segment(segments, source_model, new_count)
    return new_ids, segments


def propose_records_with_proxy(
    proposer,
    initial_input,
    prefix_ids,
    k,
    proxy_type,
    threshold,
    mavg_window,
    tokenizer,
    source_model,
    positions,
    block_index,
    cycle_index,
    stats,
):
    if k <= 0:
        return []
    start_len = prefix_ids.shape[-1]
    proposed_ids = proposer.propose_with_proxy(
        _move_token_ids(initial_input, proposer.device),
        _move_token_ids(prefix_ids, proposer.device),
        k,
        proxy_type=proxy_type,
        threshold=threshold,
        mavg_window=mavg_window,
    )
    new_tokens = proposed_ids[:, start_len:].detach().cpu().tolist()[0]
    stats["draft_generated_counts"][source_model] += len(new_tokens)
    trimmed_positions = positions[: len(new_tokens)]
    return make_records(
        token_ids=new_tokens,
        tokenizer=tokenizer,
        source_model=source_model,
        positions=trimmed_positions,
        block_index=block_index,
        cycle_index=cycle_index,
    )


def propose_ids_with_proxy(
    proposer,
    initial_input,
    prefix_ids,
    k,
    proxy_type,
    threshold,
    mavg_window,
    source_model,
    stats,
):
    if k <= 0:
        return empty_ids(proposer.device), []
    start_len = prefix_ids.shape[-1]
    proposed_ids = proposer.propose_with_proxy(
        _move_token_ids(initial_input, proposer.device),
        _move_token_ids(prefix_ids, proposer.device),
        k,
        proxy_type=proxy_type,
        threshold=threshold,
        mavg_window=mavg_window,
    )
    new_ids = proposed_ids[:, start_len:].detach()
    new_count = new_ids.shape[-1]
    stats["draft_generated_counts"][source_model] += new_count
    segments = []
    _append_segment(segments, source_model, new_count)
    return new_ids, segments


def _clone_record(record):
    return copy.deepcopy(record)


def verify_fixed_window(
    verifier,
    verifier_key,
    edge_key,
    initial_input,
    prefix_ids,
    candidate_records,
    tokenizer,
    stats,
):
    candidate_len = len(candidate_records)
    if candidate_len == 0:
        return [], {"accepted_count": 0, "generated_count": 0, "candidate_len": 0}

    verifier_device = verifier.device
    review_index = prefix_ids.shape[-1]
    candidate_ids = ids_from_records(candidate_records, verifier_device)
    review_ids = torch.cat([_move_token_ids(prefix_ids, verifier_device), candidate_ids], dim=1)

    stats["verification_calls"][verifier_key] += 1
    stats["verification_positions"][verifier_key] += candidate_len
    stats["edge_pass"][edge_key]["proposed"] += candidate_len

    reviewed_ids, _ = verifier.review(
        _move_token_ids(initial_input, verifier_device),
        review_ids,
        None,
        review_index,
        leniency=1,
    )
    reviewed_ext = reviewed_ids[:, review_index:]
    accepted_count = min(candidate_len, max(0, reviewed_ext.shape[-1] - 1))
    stats["edge_pass"][edge_key]["accepted"] += accepted_count

    final_records = []
    for record in candidate_records[:accepted_count]:
        accepted_record = _clone_record(record)
        if verifier_key == "middle":
            accepted_record["verified_by_middle"] = True
        if verifier_key == "large":
            accepted_record["verified_by_large"] = True
        final_records.append(accepted_record)

    if accepted_count == candidate_len:
        return final_records, {
            "accepted_count": accepted_count,
            "generated_count": 0,
            "candidate_len": candidate_len,
        }

    replacement_positions = [record["position"] for record in candidate_records[accepted_count:]]
    first_generated_token = int(reviewed_ext[0, -1].item())
    generated_records = make_records(
        token_ids=[first_generated_token],
        tokenizer=tokenizer,
        source_model=verifier_key,
        positions=[replacement_positions[0]],
        block_index=candidate_records[accepted_count]["large_block_index"],
        cycle_index=candidate_records[accepted_count].get("middle_cycle_index"),
    )
    final_records.extend(generated_records)
    stats["draft_generated_counts"][verifier_key] += 1

    if first_generated_token == _eos_token(verifier):
        return final_records, {
            "accepted_count": accepted_count,
            "generated_count": 1,
            "candidate_len": candidate_len,
        }

    remaining = candidate_len - len(final_records)
    if remaining > 0:
        correction_prefix = _current_sequence_ids(prefix_ids, final_records, verifier_device)
        filled_ids = verifier.propose(
            _move_token_ids(initial_input, verifier_device),
            correction_prefix,
            remaining,
        )
        new_tokens = filled_ids[:, correction_prefix.shape[-1]:].detach().cpu().tolist()[0]
        kept_tokens = []
        kept_positions = []
        for idx, token_id in enumerate(new_tokens):
            kept_tokens.append(token_id)
            kept_positions.append(replacement_positions[idx + 1])
            if token_id == _eos_token(verifier):
                break
        fill_records = make_records(
            token_ids=kept_tokens,
            tokenizer=tokenizer,
            source_model=verifier_key,
            positions=kept_positions,
            block_index=candidate_records[min(accepted_count, candidate_len - 1)]["large_block_index"],
            cycle_index=candidate_records[min(accepted_count, candidate_len - 1)].get("middle_cycle_index"),
        )
        final_records.extend(fill_records)
        stats["draft_generated_counts"][verifier_key] += len(fill_records)

    return final_records, {
        "accepted_count": accepted_count,
        "generated_count": len(final_records) - accepted_count,
        "candidate_len": candidate_len,
    }


def verify_prefix_plus_one(
    verifier,
    verifier_key,
    edge_key,
    initial_input,
    prefix_ids,
    candidate_records,
    tokenizer,
    stats,
    refill_policy="always",
):
    candidate_len = len(candidate_records)
    if candidate_len == 0:
        return [], {"accepted_count": 0, "generated_count": 0, "candidate_len": 0}

    verifier_device = verifier.device
    review_index = prefix_ids.shape[-1]
    candidate_ids = ids_from_records(candidate_records, verifier_device)
    review_ids = torch.cat([_move_token_ids(prefix_ids, verifier_device), candidate_ids], dim=1)

    stats["verification_calls"][verifier_key] += 1
    stats["verification_positions"][verifier_key] += candidate_len
    stats["edge_pass"][edge_key]["proposed"] += candidate_len

    reviewed_ids, _ = verifier.review(
        _move_token_ids(initial_input, verifier_device),
        review_ids,
        None,
        review_index,
        leniency=1,
    )
    reviewed_ext = reviewed_ids[:, review_index:]
    accepted_count = min(candidate_len, max(0, reviewed_ext.shape[-1] - 1))
    stats["edge_pass"][edge_key]["accepted"] += accepted_count

    final_records = []
    for record in candidate_records[:accepted_count]:
        accepted_record = _clone_record(record)
        if verifier_key == "middle":
            accepted_record["verified_by_middle"] = True
        if verifier_key == "large":
            accepted_record["verified_by_large"] = True
        final_records.append(accepted_record)

    generated_count = 0
    should_refill = (
        refill_policy == "always"
        or (refill_policy == "on_full_accept" and accepted_count == candidate_len)
    )
    if should_refill and reviewed_ext.shape[-1] > accepted_count:
        if accepted_count < candidate_len:
            gen_position = candidate_records[accepted_count]["position"]
            block_index = candidate_records[accepted_count]["large_block_index"]
            cycle_index = candidate_records[accepted_count].get("middle_cycle_index")
        else:
            gen_position = candidate_records[-1]["position"] + 1
            block_index = candidate_records[-1]["large_block_index"]
            cycle_index = candidate_records[-1].get("middle_cycle_index")

        generated_token = int(reviewed_ext[0, -1].item())
        final_records.extend(
            make_records(
                token_ids=[generated_token],
                tokenizer=tokenizer,
                source_model=verifier_key,
                positions=[gen_position],
                block_index=block_index,
                cycle_index=cycle_index,
            )
        )
        stats["draft_generated_counts"][verifier_key] += 1
        generated_count = 1

    return final_records, {
        "accepted_count": accepted_count,
        "generated_count": generated_count,
        "candidate_len": candidate_len,
    }


def verify_fixed_window_ids(
    verifier,
    verifier_key,
    edge_key,
    initial_input,
    prefix_ids,
    candidate_ids,
    candidate_sources,
    stats,
):
    candidate_len = candidate_ids.shape[-1]
    verifier_device = verifier.device
    if candidate_len == 0:
        return empty_ids(verifier_device), [], {
            "accepted_count": 0,
            "generated_count": 0,
            "candidate_len": 0,
        }

    review_index = prefix_ids.shape[-1]
    review_ids = _concat_ids(prefix_ids, candidate_ids, verifier_device)

    stats["verification_calls"][verifier_key] += 1
    stats["verification_positions"][verifier_key] += candidate_len
    stats["edge_pass"][edge_key]["proposed"] += candidate_len

    reviewed_ids, _ = verifier.review(
        _move_token_ids(initial_input, verifier_device),
        review_ids,
        None,
        review_index,
        leniency=1,
    )
    reviewed_ext = reviewed_ids[:, review_index:]
    accepted_count = min(candidate_len, max(0, reviewed_ext.shape[-1] - 1))
    stats["edge_pass"][edge_key]["accepted"] += accepted_count

    final_ids = _move_token_ids(candidate_ids[:, :accepted_count], verifier_device)
    final_sources = _take_segments_prefix(candidate_sources, accepted_count)
    generated_count = 0

    if accepted_count == candidate_len:
        return final_ids, final_sources, {
            "accepted_count": accepted_count,
            "generated_count": generated_count,
            "candidate_len": candidate_len,
        }

    if reviewed_ext.shape[-1] > 0:
        first_generated = _move_token_ids(reviewed_ext[:, -1:], verifier_device)
        final_ids = torch.cat([final_ids, first_generated], dim=1)
        _append_segment(final_sources, verifier_key, 1)
        stats["draft_generated_counts"][verifier_key] += 1
        generated_count += 1

        if int(first_generated[0, -1].item()) != _eos_token(verifier):
            remaining = candidate_len - final_ids.shape[-1]
            if remaining > 0:
                correction_prefix = _concat_ids(prefix_ids, final_ids, verifier_device)
                filled_ids = verifier.propose(
                    _move_token_ids(initial_input, verifier_device),
                    correction_prefix,
                    remaining,
                )
                new_ids = filled_ids[:, correction_prefix.shape[-1] :].detach()
                if new_ids.shape[-1] > 0:
                    eos_positions = (new_ids[0] == _eos_token(verifier)).nonzero(as_tuple=False)
                    kept_len = (
                        int(eos_positions[0].item()) + 1
                        if eos_positions.numel()
                        else new_ids.shape[-1]
                    )
                    new_ids = _move_token_ids(new_ids[:, :kept_len], verifier_device)
                    final_ids = torch.cat([final_ids, new_ids], dim=1)
                    _append_segment(final_sources, verifier_key, kept_len)
                    stats["draft_generated_counts"][verifier_key] += kept_len
                    generated_count += kept_len

    return final_ids, final_sources, {
        "accepted_count": accepted_count,
        "generated_count": generated_count,
        "candidate_len": candidate_len,
    }


def verify_prefix_plus_one_ids(
    verifier,
    verifier_key,
    edge_key,
    initial_input,
    prefix_ids,
    candidate_ids,
    candidate_sources,
    stats,
    refill_policy="always",
):
    candidate_len = candidate_ids.shape[-1]
    verifier_device = verifier.device
    if candidate_len == 0:
        return empty_ids(verifier_device), [], {
            "accepted_count": 0,
            "generated_count": 0,
            "candidate_len": 0,
        }

    review_index = prefix_ids.shape[-1]
    review_ids = _concat_ids(prefix_ids, candidate_ids, verifier_device)

    stats["verification_calls"][verifier_key] += 1
    stats["verification_positions"][verifier_key] += candidate_len
    stats["edge_pass"][edge_key]["proposed"] += candidate_len

    reviewed_ids, _ = verifier.review(
        _move_token_ids(initial_input, verifier_device),
        review_ids,
        None,
        review_index,
        leniency=1,
    )
    reviewed_ext = reviewed_ids[:, review_index:]
    accepted_count = min(candidate_len, max(0, reviewed_ext.shape[-1] - 1))
    stats["edge_pass"][edge_key]["accepted"] += accepted_count

    final_ids = _move_token_ids(candidate_ids[:, :accepted_count], verifier_device)
    final_sources = _take_segments_prefix(candidate_sources, accepted_count)
    generated_count = 0

    should_refill = (
        refill_policy == "always"
        or (refill_policy == "on_full_accept" and accepted_count == candidate_len)
    )
    if should_refill and reviewed_ext.shape[-1] > accepted_count:
        generated_ids = _move_token_ids(reviewed_ext[:, -1:], verifier_device)
        final_ids = torch.cat([final_ids, generated_ids], dim=1)
        _append_segment(final_sources, verifier_key, 1)
        stats["draft_generated_counts"][verifier_key] += 1
        generated_count = 1

    return final_ids, final_sources, {
        "accepted_count": accepted_count,
        "generated_count": generated_count,
        "candidate_len": candidate_len,
    }


def _truncate_records(records, limit):
    if len(records) <= limit:
        return records
    return records[:limit]


def _score_and_finalize(dataset_name, item, tokenizer, initial_input, final_records):
    generated_ids = [record["token_id"] for record in final_records]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    score = score_sample(dataset_name, item, generated_text)
    return generated_text, score


def should_stop_on_answer(cfg, tokenizer, final_records):
    if not cfg.get("stop_on_answer", True) or not final_records:
        return False
    generated_ids = [record["token_id"] for record in final_records]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return has_final_answer_signal(cfg["dataset"], generated_text)


def _per_sample_usage(stats, total_tokens):
    usage = {}
    for key in ("draft_generated_counts", "final_source_counts", "verification_positions", "verification_calls"):
        usage[key] = {}
        denom = sum(stats[key].values())
        for model_key, value in stats[key].items():
            usage[key][model_key] = {
                "count": value,
                "pct": (value / denom) if denom else 0.0,
            }
    usage["edge_pass_rates"] = {}
    for edge_key, counters in stats["edge_pass"].items():
        usage["edge_pass_rates"][edge_key] = (
            counters["accepted"] / counters["proposed"] if counters["proposed"] else None
        )
    usage["total_tokens"] = total_tokens
    return usage


def run_baseline_sample(
    cfg,
    item,
    initial_input,
    tokenizer,
    m_s,
    m_l,
    sample_index=0,
    capture_trace=True,
):
    if not capture_trace:
        return run_baseline_sample_fast(
            cfg=cfg,
            item=item,
            initial_input=initial_input,
            tokenizer=tokenizer,
            m_s=m_s,
            m_l=m_l,
            sample_index=sample_index,
        )

    reset_model_runtime(m_s)
    reset_model_runtime(m_l)

    initial_len = initial_input.shape[-1]
    current_ids = _move_token_ids(initial_input.clone(), m_l.device)
    final_records = []
    chunks = [] if capture_trace else None
    stats = new_run_stats()
    max_length = cfg["max_length"]
    small_window = cfg["small_window"]
    eos_token = _eos_token(m_l)

    start = time.time()
    chunk_index = 0
    with torch.no_grad():
        while len(final_records) < max_length:
            remaining = max_length - len(final_records)
            chunk_target = min(small_window, remaining)
            base_position = len(final_records)
            positions = list(range(base_position, base_position + chunk_target))
            small_records = propose_records(
                proposer=m_s,
                initial_input=initial_input,
                prefix_ids=current_ids,
                k=chunk_target,
                tokenizer=tokenizer,
                source_model="small",
                positions=positions,
                block_index=chunk_index,
                cycle_index=0,
                stats=stats,
            )
            final_chunk_records, meta = verify_fixed_window(
                verifier=m_l,
                verifier_key="large",
                edge_key="small_to_large",
                initial_input=initial_input,
                prefix_ids=current_ids,
                candidate_records=small_records,
                tokenizer=tokenizer,
                stats=stats,
            )
            final_records.extend(final_chunk_records)
            current_ids = _current_sequence_ids(_move_token_ids(initial_input, m_l.device), final_records, m_l.device)
            if not final_chunk_records:
                break
            if capture_trace:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "requested_tokens": chunk_target,
                        "small_draft": copy.deepcopy(small_records),
                        "large_accepted_count": meta["accepted_count"],
                        "large_generated_count": meta["generated_count"],
                        "final_chunk": copy.deepcopy(final_chunk_records),
                    }
                )
            if final_chunk_records and final_chunk_records[-1]["token_id"] == eos_token:
                break
            if should_stop_on_answer(cfg, tokenizer, final_records):
                break
            chunk_index += 1

    wall_time = time.time() - start
    for record in final_records:
        stats["final_source_counts"][record["source_model"]] += 1

    generated_text, score = _score_and_finalize(
        cfg["dataset"], item, tokenizer, initial_input, final_records
    )
    sample_usage = _per_sample_usage(stats, len(final_records))
    result = {
        "sample_index": sample_index,
        "generated_text": generated_text,
        "gold_answer": item.get("answer"),
        "score": score,
        "wall_time": wall_time,
        "tokens_generated": len(final_records),
        "usage": sample_usage,
        "raw_counters": stats,
        "model_runtime": {
            "small": model_runtime_snapshot(m_s),
            "middle": model_runtime_snapshot(None),
            "large": model_runtime_snapshot(m_l),
        },
        "ml_forward_calls": getattr(m_l, "forward_count", 0),
    }
    if capture_trace:
        result["prompt"] = tokenizer.decode(
            initial_input[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result["final_tokens"] = final_records
        result["chunks"] = chunks
    return result


def run_baseline_sample_fast(
    cfg,
    item,
    initial_input,
    tokenizer,
    m_s,
    m_l,
    sample_index=0,
):
    reset_model_runtime(m_s)
    reset_model_runtime(m_l)

    current_small_ids = _move_token_ids(initial_input.clone(), m_s.device)
    current_large_ids = _move_token_ids(initial_input.clone(), m_l.device)
    generated_ids = empty_ids(m_l.device)
    stats = new_run_stats()
    max_length = cfg["max_length"]
    small_window = cfg["small_window"]
    eos_token = _eos_token(m_l)

    start = time.time()
    with torch.no_grad():
        while generated_ids.shape[-1] < max_length:
            remaining = max_length - generated_ids.shape[-1]
            chunk_target = min(small_window, remaining)
            small_ids, small_sources = propose_ids(
                proposer=m_s,
                initial_input=initial_input,
                prefix_ids=current_small_ids,
                k=chunk_target,
                source_model="small",
                stats=stats,
            )
            final_chunk_ids, final_chunk_sources, _ = verify_fixed_window_ids(
                verifier=m_l,
                verifier_key="large",
                edge_key="small_to_large",
                initial_input=initial_input,
                prefix_ids=current_large_ids,
                candidate_ids=small_ids,
                candidate_sources=small_sources,
                stats=stats,
            )
            final_chunk_ids = final_chunk_ids[:, :remaining]
            final_chunk_sources = _take_segments_prefix(final_chunk_sources, remaining)
            if final_chunk_ids.shape[-1] == 0:
                break
            generated_ids = _append_ids(generated_ids, final_chunk_ids)
            current_large_ids = _append_ids(current_large_ids, final_chunk_ids)
            current_small_ids = _append_ids(current_small_ids, final_chunk_ids)
            _add_counts(stats["final_source_counts"], _segment_counts(final_chunk_sources))
            if final_chunk_ids.shape[-1] and int(final_chunk_ids[0, -1].item()) == eos_token:
                break
            if should_stop_on_ids(cfg, tokenizer, generated_ids):
                break

    wall_time = time.time() - start
    generated_text, score = _score_and_finalize_ids(
        cfg["dataset"], item, tokenizer, generated_ids
    )
    sample_usage = _per_sample_usage(stats, generated_ids.shape[-1])
    return {
        "sample_index": sample_index,
        "generated_text": generated_text,
        "gold_answer": item.get("answer"),
        "score": score,
        "wall_time": wall_time,
        "tokens_generated": generated_ids.shape[-1],
        "usage": sample_usage,
        "raw_counters": stats,
        "model_runtime": {
            "small": model_runtime_snapshot(m_s),
            "middle": model_runtime_snapshot(None),
            "large": model_runtime_snapshot(m_l),
        },
        "ml_forward_calls": getattr(m_l, "forward_count", 0),
    }


def run_hierarchical_sample(
    cfg,
    item,
    initial_input,
    tokenizer,
    m_s,
    m_m,
    m_l,
    sample_index=0,
    capture_trace=True,
):
    if not capture_trace:
        return run_hierarchical_sample_fast(
            cfg=cfg,
            item=item,
            initial_input=initial_input,
            tokenizer=tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=sample_index,
        )

    reset_model_runtime(m_s)
    reset_model_runtime(m_m)
    reset_model_runtime(m_l)

    current_ids = _move_token_ids(initial_input.clone(), m_l.device)
    final_records = []
    blocks = [] if capture_trace else None
    stats = new_run_stats()
    max_length = cfg["max_length"]
    small_window = cfg["small_window"]
    middle_window = cfg["middle_window"]
    eos_token = _eos_token(m_l)

    start = time.time()
    block_index = 0
    with torch.no_grad():
        while len(final_records) < max_length:
            block_target = min(middle_window, max_length - len(final_records))
            block_records = []
            middle_cycles = []
            cycle_index = 0

            while len(block_records) < block_target:
                remaining_in_block = block_target - len(block_records)
                chunk_target = min(small_window, remaining_in_block)
                tentative_prefix = _current_sequence_ids(current_ids, block_records, m_s.device)
                base_position = len(final_records) + len(block_records)
                positions = list(range(base_position, base_position + chunk_target))
                small_records = propose_records(
                    proposer=m_s,
                    initial_input=initial_input,
                    prefix_ids=tentative_prefix,
                    k=chunk_target,
                    tokenizer=tokenizer,
                    source_model="small",
                    positions=positions,
                    block_index=block_index,
                    cycle_index=cycle_index,
                    stats=stats,
                )
                middle_chunk_records, meta = verify_fixed_window(
                    verifier=m_m,
                    verifier_key="middle",
                    edge_key="small_to_middle",
                    initial_input=initial_input,
                    prefix_ids=_move_token_ids(tentative_prefix, m_m.device),
                    candidate_records=small_records,
                    tokenizer=tokenizer,
                    stats=stats,
                )
                block_records.extend(middle_chunk_records)
                if not middle_chunk_records:
                    break
                middle_cycles.append(
                    {
                        "cycle_index": cycle_index,
                        "requested_tokens": chunk_target,
                        "small_draft": copy.deepcopy(small_records),
                        "middle_accepted_count": meta["accepted_count"],
                        "middle_generated_count": meta["generated_count"],
                        "middle_result": copy.deepcopy(middle_chunk_records),
                    }
                )
                if middle_chunk_records and middle_chunk_records[-1]["token_id"] == eos_token:
                    break
                cycle_index += 1

            if not block_records:
                break
            final_block_records, meta = verify_fixed_window(
                verifier=m_l,
                verifier_key="large",
                edge_key="middle_to_large",
                initial_input=initial_input,
                prefix_ids=current_ids,
                candidate_records=block_records,
                tokenizer=tokenizer,
                stats=stats,
            )
            final_records.extend(final_block_records)
            current_ids = _current_sequence_ids(_move_token_ids(initial_input, m_l.device), final_records, m_l.device)
            if not final_block_records:
                break
            if capture_trace:
                blocks.append(
                    {
                        "block_index": block_index,
                        "requested_tokens": block_target,
                        "middle_cycles": middle_cycles,
                        "candidate_to_large": copy.deepcopy(block_records),
                        "large_accepted_count": meta["accepted_count"],
                        "large_generated_count": meta["generated_count"],
                        "final_block": copy.deepcopy(final_block_records),
                    }
                )
            if final_block_records and final_block_records[-1]["token_id"] == eos_token:
                break
            if should_stop_on_answer(cfg, tokenizer, final_records):
                break
            block_index += 1

    wall_time = time.time() - start
    for record in final_records:
        stats["final_source_counts"][record["source_model"]] += 1

    generated_text, score = _score_and_finalize(
        cfg["dataset"], item, tokenizer, initial_input, final_records
    )
    sample_usage = _per_sample_usage(stats, len(final_records))
    result = {
        "sample_index": sample_index,
        "generated_text": generated_text,
        "gold_answer": item.get("answer"),
        "score": score,
        "wall_time": wall_time,
        "tokens_generated": len(final_records),
        "usage": sample_usage,
        "raw_counters": stats,
        "model_runtime": {
            "small": model_runtime_snapshot(m_s),
            "middle": model_runtime_snapshot(m_m),
            "large": model_runtime_snapshot(m_l),
        },
        "ml_forward_calls": getattr(m_l, "forward_count", 0),
        "mm_forward_calls": getattr(m_m, "forward_count", 0),
    }
    if capture_trace:
        result["prompt"] = tokenizer.decode(
            initial_input[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result["final_tokens"] = final_records
        result["blocks"] = blocks
    return result


def run_hierarchical_sample_fast(
    cfg,
    item,
    initial_input,
    tokenizer,
    m_s,
    m_m,
    m_l,
    sample_index=0,
):
    reset_model_runtime(m_s)
    reset_model_runtime(m_m)
    reset_model_runtime(m_l)

    current_small_ids = _move_token_ids(initial_input.clone(), m_s.device)
    current_middle_ids = _move_token_ids(initial_input.clone(), m_m.device)
    current_large_ids = _move_token_ids(initial_input.clone(), m_l.device)
    generated_ids = empty_ids(m_l.device)
    stats = new_run_stats()
    max_length = cfg["max_length"]
    small_window = cfg["small_window"]
    middle_window = cfg["middle_window"]
    eos_token = _eos_token(m_l)

    start = time.time()
    with torch.no_grad():
        while generated_ids.shape[-1] < max_length:
            block_target = min(middle_window, max_length - generated_ids.shape[-1])
            block_small_ids = empty_ids(m_s.device)
            block_middle_ids = empty_ids(m_m.device)
            block_large_ids = empty_ids(m_l.device)
            block_sources = []

            while block_large_ids.shape[-1] < block_target:
                remaining_in_block = block_target - block_large_ids.shape[-1]
                chunk_target = min(small_window, remaining_in_block)
                small_prefix = _concat_ids(current_small_ids, block_small_ids, m_s.device)
                small_ids, small_sources = propose_ids(
                    proposer=m_s,
                    initial_input=initial_input,
                    prefix_ids=small_prefix,
                    k=chunk_target,
                    source_model="small",
                    stats=stats,
                )
                middle_prefix = _concat_ids(current_middle_ids, block_middle_ids, m_m.device)
                middle_chunk_ids, middle_chunk_sources, _ = verify_fixed_window_ids(
                    verifier=m_m,
                    verifier_key="middle",
                    edge_key="small_to_middle",
                    initial_input=initial_input,
                    prefix_ids=middle_prefix,
                    candidate_ids=small_ids,
                    candidate_sources=small_sources,
                    stats=stats,
                )
                middle_chunk_ids = middle_chunk_ids[:, :remaining_in_block]
                middle_chunk_sources = _take_segments_prefix(
                    middle_chunk_sources, remaining_in_block
                )
                block_small_ids = _append_ids(block_small_ids, middle_chunk_ids)
                block_middle_ids = _append_ids(block_middle_ids, middle_chunk_ids)
                block_large_ids = _append_ids(block_large_ids, middle_chunk_ids)
                _extend_segments(block_sources, middle_chunk_sources)
                if middle_chunk_ids.shape[-1] == 0:
                    break
                if middle_chunk_ids.shape[-1] and int(middle_chunk_ids[0, -1].item()) == eos_token:
                    break

            if block_large_ids.shape[-1] == 0:
                break
            remaining = max_length - generated_ids.shape[-1]
            final_block_ids, final_block_sources, _ = verify_fixed_window_ids(
                verifier=m_l,
                verifier_key="large",
                edge_key="middle_to_large",
                initial_input=initial_input,
                prefix_ids=current_large_ids,
                candidate_ids=block_large_ids,
                candidate_sources=block_sources,
                stats=stats,
            )
            final_block_ids = final_block_ids[:, :remaining]
            final_block_sources = _take_segments_prefix(final_block_sources, remaining)
            if final_block_ids.shape[-1] == 0:
                break
            generated_ids = _append_ids(generated_ids, final_block_ids)
            current_small_ids = _append_ids(current_small_ids, final_block_ids)
            current_middle_ids = _append_ids(current_middle_ids, final_block_ids)
            current_large_ids = _append_ids(current_large_ids, final_block_ids)
            _add_counts(stats["final_source_counts"], _segment_counts(final_block_sources))
            if final_block_ids.shape[-1] and int(final_block_ids[0, -1].item()) == eos_token:
                break
            if should_stop_on_ids(cfg, tokenizer, generated_ids):
                break

    wall_time = time.time() - start
    generated_text, score = _score_and_finalize_ids(
        cfg["dataset"], item, tokenizer, generated_ids
    )
    sample_usage = _per_sample_usage(stats, generated_ids.shape[-1])
    return {
        "sample_index": sample_index,
        "generated_text": generated_text,
        "gold_answer": item.get("answer"),
        "score": score,
        "wall_time": wall_time,
        "tokens_generated": generated_ids.shape[-1],
        "usage": sample_usage,
        "raw_counters": stats,
        "model_runtime": {
            "small": model_runtime_snapshot(m_s),
            "middle": model_runtime_snapshot(m_m),
            "large": model_runtime_snapshot(m_l),
        },
        "ml_forward_calls": getattr(m_l, "forward_count", 0),
        "mm_forward_calls": getattr(m_m, "forward_count", 0),
    }


def run_double_layer_sample(
    cfg,
    item,
    initial_input,
    tokenizer,
    m_s,
    m_m,
    m_l,
    sample_index=0,
    capture_trace=True,
):
    if not capture_trace or _selective_routing_enabled(cfg):
        return run_double_layer_sample_fast(
            cfg=cfg,
            item=item,
            initial_input=initial_input,
            tokenizer=tokenizer,
            m_s=m_s,
            m_m=m_m,
            m_l=m_l,
            sample_index=sample_index,
        )

    reset_model_runtime(m_s)
    reset_model_runtime(m_m)
    reset_model_runtime(m_l)

    current_ids = _move_token_ids(initial_input.clone(), m_l.device)
    final_records = []
    blocks = [] if capture_trace else None
    stats = new_run_stats()
    max_length = cfg["max_length"]
    eos_token = _eos_token(m_l)
    variant = cfg.get("hierarchical_variant", "double_layer")
    refill_policy = _double_layer_refill_policy(cfg)
    window_state = _init_window_state(cfg)
    proxy_type = None
    if variant in DOUBLE_LAYER_PROXY_VARIANTS:
        proxy_type = variant.split("_", 1)[1]
    proxy_threshold = cfg.get("proxy_threshold")
    proxy_defaults = {
        "entropy": 2.0,
        "top1": -1.5,
        "margin": 1.0,
        "mavg": -1.5,
    }
    if proxy_type is not None and proxy_threshold is None:
        proxy_threshold = proxy_defaults[proxy_type]
    mavg_window = cfg.get("mavg_window", 5)

    start = time.time()
    block_index = 0
    with torch.no_grad():
        while len(final_records) < max_length:
            block_target = _request_window(
                cfg,
                window_state,
                "middle",
                max_length - len(final_records),
            )
            cfg["_last_window_utility"] = None
            block_checkpoint = _runtime_checkpoint(m_s, m_m, m_l)
            block_records = []
            middle_cycles = []
            cycle_index = 0

            while len(block_records) < block_target:
                remaining_in_block = block_target - len(block_records)
                chunk_target = _request_window(
                    cfg,
                    window_state,
                    "small",
                    remaining_in_block,
                )
                tentative_prefix = _current_sequence_ids(current_ids, block_records, m_s.device)
                base_position = len(final_records) + len(block_records)
                positions = list(range(base_position, base_position + chunk_target + 1))
                if proxy_type is None:
                    small_records = propose_records(
                        proposer=m_s,
                        initial_input=initial_input,
                        prefix_ids=tentative_prefix,
                        k=chunk_target,
                        tokenizer=tokenizer,
                        source_model="small",
                        positions=positions,
                        block_index=block_index,
                        cycle_index=cycle_index,
                        stats=stats,
                    )
                else:
                    small_records = propose_records_with_proxy(
                        proposer=m_s,
                        initial_input=initial_input,
                        prefix_ids=tentative_prefix,
                        k=chunk_target,
                        proxy_type=proxy_type,
                        threshold=proxy_threshold,
                        mavg_window=mavg_window,
                        tokenizer=tokenizer,
                        source_model="small",
                        positions=positions,
                        block_index=block_index,
                        cycle_index=cycle_index,
                        stats=stats,
                    )

                middle_step_records, meta = verify_prefix_plus_one(
                    verifier=m_m,
                    verifier_key="middle",
                    edge_key="small_to_middle",
                    initial_input=initial_input,
                    prefix_ids=_move_token_ids(tentative_prefix, m_m.device),
                    candidate_records=small_records,
                    tokenizer=tokenizer,
                    stats=stats,
                    refill_policy=refill_policy,
                )
                small_acceptance_ratio = (
                    meta["accepted_count"] / meta["candidate_len"]
                    if meta["candidate_len"]
                    else 1.0
                )
                _update_window(
                    cfg,
                    window_state,
                    "small",
                    small_acceptance_ratio,
                    block_index * 1000 + cycle_index,
                )
                middle_step_records = _truncate_records(
                    middle_step_records,
                    block_target - len(block_records),
                )
                block_records.extend(middle_step_records)
                if capture_trace:
                    middle_cycles.append(
                        {
                            "cycle_index": cycle_index,
                            "requested_tokens": chunk_target,
                            "small_draft": copy.deepcopy(small_records),
                            "middle_accepted_count": meta["accepted_count"],
                            "middle_generated_count": meta["generated_count"],
                            "middle_result": copy.deepcopy(middle_step_records),
                            "small_window": chunk_target,
                            "small_window_acceptance_ratio": small_acceptance_ratio,
                            "middle_refill_policy": refill_policy,
                        }
                    )
                if not middle_step_records:
                    break
                if middle_step_records and middle_step_records[-1]["token_id"] == eos_token:
                    break
                cycle_index += 1

            if not block_records:
                fallback_records = propose_records(
                    proposer=m_l,
                    initial_input=initial_input,
                    prefix_ids=current_ids,
                    k=1,
                    tokenizer=tokenizer,
                    source_model="large",
                    positions=[len(final_records)],
                    block_index=block_index,
                    cycle_index=None,
                    stats=stats,
                )
                final_records.extend(fallback_records)
                current_ids = _current_sequence_ids(
                    _move_token_ids(initial_input, m_l.device), final_records, m_l.device
                )
                if not fallback_records:
                    break
                if fallback_records[-1]["token_id"] == eos_token:
                    break
                if should_stop_on_answer(cfg, tokenizer, final_records):
                    break
                block_index += 1
                continue
            final_block_records, meta = verify_prefix_plus_one(
                verifier=m_l,
                verifier_key="large",
                edge_key="middle_to_large",
                initial_input=initial_input,
                prefix_ids=current_ids,
                candidate_records=block_records,
                tokenizer=tokenizer,
                stats=stats,
                refill_policy="always",
            )
            middle_acceptance_ratio = (
                meta["accepted_count"] / meta["candidate_len"]
                if meta["candidate_len"]
                else 1.0
            )
            middle_utility_score = None
            if _window_policy(cfg) == "utility":
                middle_utility_score = _estimate_middle_window_utility(
                    cfg,
                    block_checkpoint,
                    m_s,
                    m_m,
                    m_l,
                    len(block_records),
                )
            cfg["_last_window_utility"] = middle_utility_score
            _update_window(
                cfg,
                window_state,
                "middle",
                middle_acceptance_ratio,
                block_index,
            )
            cfg["_last_window_utility"] = None
            final_block_records = _truncate_records(
                final_block_records,
                max_length - len(final_records),
            )
            final_records.extend(final_block_records)
            current_ids = _current_sequence_ids(
                _move_token_ids(initial_input, m_l.device), final_records, m_l.device
            )
            if not final_block_records:
                break
            if capture_trace:
                blocks.append(
                    {
                        "block_index": block_index,
                        "requested_tokens": block_target,
                        "middle_cycles": middle_cycles,
                        "candidate_to_large": copy.deepcopy(block_records),
                        "large_accepted_count": meta["accepted_count"],
                        "large_generated_count": meta["generated_count"],
                        "final_block": copy.deepcopy(final_block_records),
                        "middle_window": block_target,
                        "middle_window_acceptance_ratio": middle_acceptance_ratio,
                        "middle_window_utility_score": middle_utility_score,
                        "window_policy": window_state["policy"],
                    }
                )
            if final_block_records and final_block_records[-1]["token_id"] == eos_token:
                break
            if should_stop_on_answer(cfg, tokenizer, final_records):
                break
            block_index += 1

    wall_time = time.time() - start
    for record in final_records:
        stats["final_source_counts"][record["source_model"]] += 1

    generated_text, score = _score_and_finalize(
        cfg["dataset"], item, tokenizer, initial_input, final_records
    )
    sample_usage = _per_sample_usage(stats, len(final_records))
    result = {
        "sample_index": sample_index,
        "generated_text": generated_text,
        "gold_answer": item.get("answer"),
        "score": score,
        "wall_time": wall_time,
        "tokens_generated": len(final_records),
        "usage": sample_usage,
        "raw_counters": stats,
        "model_runtime": {
            "small": model_runtime_snapshot(m_s),
            "middle": model_runtime_snapshot(m_m),
            "large": model_runtime_snapshot(m_l),
        },
        "ml_forward_calls": getattr(m_l, "forward_count", 0),
        "mm_forward_calls": getattr(m_m, "forward_count", 0),
        "draft_window": _window_payload(window_state),
        "hierarchical_variant": variant,
        "middle_refill_policy": refill_policy,
    }
    if capture_trace:
        result["prompt"] = tokenizer.decode(
            initial_input[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result["final_tokens"] = final_records
        result["blocks"] = blocks
    return result


def run_double_layer_sample_fast(
    cfg,
    item,
    initial_input,
    tokenizer,
    m_s,
    m_m,
    m_l,
    sample_index=0,
):
    reset_model_runtime(m_s)
    reset_model_runtime(m_m)
    reset_model_runtime(m_l)

    current_small_ids = _move_token_ids(initial_input.clone(), m_s.device)
    current_middle_ids = _move_token_ids(initial_input.clone(), m_m.device)
    current_large_ids = _move_token_ids(initial_input.clone(), m_l.device)
    generated_ids = empty_ids(m_l.device)
    stats = new_run_stats()
    max_length = cfg["max_length"]
    eos_token = _eos_token(m_l)
    variant = cfg.get("hierarchical_variant", "double_layer")
    refill_policy = _double_layer_refill_policy(cfg)
    window_state = _init_window_state(cfg)
    route_state = _init_route_state(cfg)
    proxy_type = None
    if variant in DOUBLE_LAYER_PROXY_VARIANTS:
        proxy_type = variant.split("_", 1)[1]
    proxy_threshold = cfg.get("proxy_threshold")
    proxy_defaults = {
        "entropy": 2.0,
        "top1": -1.5,
        "margin": 1.0,
        "mavg": -1.5,
    }
    if proxy_type is not None and proxy_threshold is None:
        proxy_threshold = proxy_defaults[proxy_type]
    mavg_window = cfg.get("mavg_window", 5)

    start = time.time()
    block_index = 0
    with torch.no_grad():
        while generated_ids.shape[-1] < max_length:
            block_target = _request_window(
                cfg,
                window_state,
                "middle",
                max_length - generated_ids.shape[-1],
            )
            cfg["_last_window_utility"] = None
            block_checkpoint = _runtime_checkpoint(m_s, m_m, m_l)
            selected_route, route_reason = _select_route(cfg, route_state, block_index)
            if selected_route == "large":
                direct_block_ids, direct_block_sources, direct_meta = _run_direct_large_block_fast(
                    cfg=cfg,
                    window_state=window_state,
                    initial_input=initial_input,
                    current_small_ids=current_small_ids,
                    current_large_ids=current_large_ids,
                    m_s=m_s,
                    m_l=m_l,
                    stats=stats,
                    block_target=block_target,
                    eos_token=eos_token,
                    proxy_type=proxy_type,
                    proxy_threshold=proxy_threshold,
                    mavg_window=mavg_window,
                    block_index=block_index,
                )
                direct_block_ids = direct_block_ids[
                    :, : max_length - generated_ids.shape[-1]
                ]
                direct_block_sources = _take_segments_prefix(
                    direct_block_sources,
                    direct_block_ids.shape[-1],
                )
                if direct_block_ids.shape[-1] == 0:
                    break
                _record_route_outcome(
                    route_state,
                    block_index,
                    "large",
                    route_reason,
                    direct_meta["acceptance_ratio"],
                    None,
                    direct_block_ids.shape[-1],
                )
                generated_ids = _append_ids(generated_ids, direct_block_ids)
                current_small_ids = _append_ids(current_small_ids, direct_block_ids)
                current_middle_ids = _append_ids(current_middle_ids, direct_block_ids)
                current_large_ids = _append_ids(current_large_ids, direct_block_ids)
                _add_counts(stats["final_source_counts"], _segment_counts(direct_block_sources))
                if int(direct_block_ids[0, -1].item()) == eos_token:
                    break
                if should_stop_on_ids(cfg, tokenizer, generated_ids):
                    break
                block_index += 1
                continue
            block_small_ids = empty_ids(m_s.device)
            block_middle_ids = empty_ids(m_m.device)
            block_large_ids = empty_ids(m_l.device)
            block_sources = []
            cycle_index = 0

            while block_large_ids.shape[-1] < block_target:
                remaining_in_block = block_target - block_large_ids.shape[-1]
                chunk_target = _request_window(
                    cfg,
                    window_state,
                    "small",
                    remaining_in_block,
                )
                small_prefix = _concat_ids(current_small_ids, block_small_ids, m_s.device)
                if proxy_type is None:
                    small_ids, small_sources = propose_ids(
                        proposer=m_s,
                        initial_input=initial_input,
                        prefix_ids=small_prefix,
                        k=chunk_target,
                        source_model="small",
                        stats=stats,
                    )
                else:
                    small_ids, small_sources = propose_ids_with_proxy(
                        proposer=m_s,
                        initial_input=initial_input,
                        prefix_ids=small_prefix,
                        k=chunk_target,
                        proxy_type=proxy_type,
                        threshold=proxy_threshold,
                        mavg_window=mavg_window,
                        source_model="small",
                        stats=stats,
                    )

                middle_prefix = _concat_ids(current_middle_ids, block_middle_ids, m_m.device)
                middle_step_ids, middle_step_sources, middle_meta = verify_prefix_plus_one_ids(
                    verifier=m_m,
                    verifier_key="middle",
                    edge_key="small_to_middle",
                    initial_input=initial_input,
                    prefix_ids=middle_prefix,
                    candidate_ids=small_ids,
                    candidate_sources=small_sources,
                    stats=stats,
                    refill_policy=refill_policy,
                )
                small_acceptance_ratio = (
                    middle_meta["accepted_count"] / middle_meta["candidate_len"]
                    if middle_meta["candidate_len"]
                    else 1.0
                )
                _update_window(
                    cfg,
                    window_state,
                    "small",
                    small_acceptance_ratio,
                    block_index * 1000 + cycle_index,
                )
                middle_step_ids = middle_step_ids[:, :remaining_in_block]
                middle_step_sources = _take_segments_prefix(
                    middle_step_sources, remaining_in_block
                )
                block_small_ids = _append_ids(block_small_ids, middle_step_ids)
                block_middle_ids = _append_ids(block_middle_ids, middle_step_ids)
                block_large_ids = _append_ids(block_large_ids, middle_step_ids)
                _extend_segments(block_sources, middle_step_sources)
                if middle_step_ids.shape[-1] == 0:
                    break
                if middle_step_ids.shape[-1] and int(middle_step_ids[0, -1].item()) == eos_token:
                    break
                cycle_index += 1

            if block_large_ids.shape[-1] == 0:
                fallback_ids, fallback_sources = propose_ids(
                    proposer=m_l,
                    initial_input=initial_input,
                    prefix_ids=current_large_ids,
                    k=1,
                    source_model="large",
                    stats=stats,
                )
                if fallback_ids.shape[-1] == 0:
                    break
                generated_ids = _append_ids(generated_ids, fallback_ids)
                current_small_ids = _append_ids(current_small_ids, fallback_ids)
                current_middle_ids = _append_ids(current_middle_ids, fallback_ids)
                current_large_ids = _append_ids(current_large_ids, fallback_ids)
                _add_counts(stats["final_source_counts"], _segment_counts(fallback_sources))
                if int(fallback_ids[0, -1].item()) == eos_token:
                    break
                if should_stop_on_ids(cfg, tokenizer, generated_ids):
                    break
                block_index += 1
                continue
            remaining = max_length - generated_ids.shape[-1]
            final_block_ids, final_block_sources, large_meta = verify_prefix_plus_one_ids(
                verifier=m_l,
                verifier_key="large",
                edge_key="middle_to_large",
                initial_input=initial_input,
                prefix_ids=current_large_ids,
                candidate_ids=block_large_ids,
                candidate_sources=block_sources,
                stats=stats,
                refill_policy="always",
            )
            middle_acceptance_ratio = (
                large_meta["accepted_count"] / large_meta["candidate_len"]
                if large_meta["candidate_len"]
                else 1.0
            )
            middle_utility_score = None
            if _window_policy(cfg) == "utility":
                middle_utility_score = _estimate_middle_window_utility(
                    cfg,
                    block_checkpoint,
                    m_s,
                    m_m,
                    m_l,
                    block_large_ids.shape[-1],
                )
            cfg["_last_window_utility"] = middle_utility_score
            _update_window(
                cfg,
                window_state,
                "middle",
                middle_acceptance_ratio,
                block_index,
            )
            cfg["_last_window_utility"] = None
            final_block_ids = final_block_ids[:, :remaining]
            final_block_sources = _take_segments_prefix(final_block_sources, remaining)
            if final_block_ids.shape[-1] == 0:
                break
            _record_route_outcome(
                route_state,
                block_index,
                "middle",
                route_reason,
                middle_acceptance_ratio,
                middle_utility_score,
                final_block_ids.shape[-1],
            )
            generated_ids = _append_ids(generated_ids, final_block_ids)
            current_small_ids = _append_ids(current_small_ids, final_block_ids)
            current_middle_ids = _append_ids(current_middle_ids, final_block_ids)
            current_large_ids = _append_ids(current_large_ids, final_block_ids)
            _add_counts(stats["final_source_counts"], _segment_counts(final_block_sources))
            if final_block_ids.shape[-1] and int(final_block_ids[0, -1].item()) == eos_token:
                break
            if should_stop_on_ids(cfg, tokenizer, generated_ids):
                break
            block_index += 1

    wall_time = time.time() - start
    generated_text, score = _score_and_finalize_ids(
        cfg["dataset"], item, tokenizer, generated_ids
    )
    sample_usage = _per_sample_usage(stats, generated_ids.shape[-1])
    return {
        "sample_index": sample_index,
        "generated_text": generated_text,
        "gold_answer": item.get("answer"),
        "score": score,
        "wall_time": wall_time,
        "tokens_generated": generated_ids.shape[-1],
        "usage": sample_usage,
        "raw_counters": stats,
        "model_runtime": {
            "small": model_runtime_snapshot(m_s),
            "middle": model_runtime_snapshot(m_m),
            "large": model_runtime_snapshot(m_l),
        },
        "ml_forward_calls": getattr(m_l, "forward_count", 0),
        "mm_forward_calls": getattr(m_m, "forward_count", 0),
        "draft_window": _window_payload(window_state),
        "hierarchical_variant": variant,
        "middle_refill_policy": refill_policy,
        "routing": _route_payload(route_state),
    }


def summarize_run(sample_results, run_name, cfg):
    n = len(sample_results)
    total_tokens = sum(sample["tokens_generated"] for sample in sample_results)
    total_wall = sum(sample["wall_time"] for sample in sample_results)
    score_name = sample_results[0]["score"]["metric_name"] if sample_results else "score"
    total_score = sum(sample["score"]["score"] for sample in sample_results)
    runtime_aggregate = {
        model_key: {
            "propose_calls": 0,
            "review_calls": 0,
            "propose_wall_time": 0.0,
            "review_wall_time": 0.0,
            "total_wall_time": 0.0,
        }
        for model_key in MODEL_KEYS
    }

    aggregate = new_run_stats()
    window_totals = {
        "small": {"sum": 0, "count": 0},
        "middle": {"sum": 0, "count": 0},
    }
    window_change_count = 0
    window_policy = cfg.get("window_policy", cfg.get("draft_window_policy", "fixed"))
    routing_totals = {"middle": 0, "large": 0}
    routing_change_count = 0
    routing_seen = False
    for sample in sample_results:
        for counter_key in ("draft_generated_counts", "final_source_counts", "verification_positions", "verification_calls"):
            for model_key, value in sample["raw_counters"][counter_key].items():
                aggregate[counter_key][model_key] += value
        for edge_key, counters in sample["raw_counters"]["edge_pass"].items():
            aggregate["edge_pass"][edge_key]["accepted"] += counters["accepted"]
            aggregate["edge_pass"][edge_key]["proposed"] += counters["proposed"]
        for model_key, runtime in sample.get("model_runtime", {}).items():
            bucket = runtime_aggregate[model_key]
            for field in bucket:
                bucket[field] += runtime.get(field, 0)
        draft_window = sample.get("draft_window") or {}
        if draft_window:
            window_policy = draft_window.get("policy", window_policy)
            window_change_count += draft_window.get("change_count", 0)
            for model_key, totals in (draft_window.get("totals") or {}).items():
                if model_key not in window_totals:
                    continue
                window_totals[model_key]["sum"] += totals.get("sum", 0)
                window_totals[model_key]["count"] += totals.get("count", 0)
        routing = sample.get("routing")
        if routing:
            routing_seen = True
            routing_change_count += routing.get("change_count", 0)
            for route_key, value in (routing.get("counts") or {}).items():
                if route_key in routing_totals:
                    routing_totals[route_key] += value

    avg_draft_window = {}
    for model_key, totals in window_totals.items():
        avg_draft_window[model_key] = (
            totals["sum"] / totals["count"] if totals["count"] else 0.0
        )

    runtime_avg = {}
    runtime_total = sum(
        runtime_aggregate[model_key]["total_wall_time"] for model_key in MODEL_KEYS
    )
    for model_key in MODEL_KEYS:
        runtime_avg[model_key] = {
            "propose_calls": (
                runtime_aggregate[model_key]["propose_calls"] / n if n else 0.0
            ),
            "review_calls": (
                runtime_aggregate[model_key]["review_calls"] / n if n else 0.0
            ),
            "propose_wall_time": (
                runtime_aggregate[model_key]["propose_wall_time"] / n if n else 0.0
            ),
            "review_wall_time": (
                runtime_aggregate[model_key]["review_wall_time"] / n if n else 0.0
            ),
            "total_wall_time": (
                runtime_aggregate[model_key]["total_wall_time"] / n if n else 0.0
            ),
            "share": (
                runtime_aggregate[model_key]["total_wall_time"] / runtime_total
                if runtime_total
                else 0.0
            ),
        }

    summary = {
        "run_name": run_name,
        "tokens_per_sec": (total_tokens / total_wall) if total_wall else 0.0,
        "avg_wall_time": (total_wall / n) if n else 0.0,
        "avg_tokens_generated": (total_tokens / n) if n else 0.0,
        "benchmark_metric": score_name,
        "benchmark_score": (total_score / n) if n else 0.0,
        "usage": _per_sample_usage(aggregate, total_tokens),
        "avg_model_runtime": runtime_avg,
        "avg_ml_forward_calls": (
            sum(sample.get("ml_forward_calls", 0) for sample in sample_results) / n if n else 0.0
        ),
        "avg_mm_forward_calls": (
            sum(sample.get("mm_forward_calls", 0) for sample in sample_results) / n if n else 0.0
        ),
        "n_samples": n,
        "small_window": cfg["small_window"],
        "middle_window": cfg.get("middle_window"),
        "hierarchical_variant": cfg.get("hierarchical_variant"),
        "middle_refill_policy": _double_layer_refill_policy(cfg)
        if run_name == "hierarchical"
        else None,
        "window_policy": window_policy,
        "adapt_small_window": bool(cfg.get("adapt_small_window", False)),
        "dynamic_utility_margin": cfg.get("dynamic_utility_margin", 0.0),
        "avg_draft_window": avg_draft_window,
        "avg_draft_window_changes": window_change_count / n if n else 0.0,
        "max_length": cfg["max_length"],
    }
    if routing_seen:
        routing_total = sum(routing_totals.values())
        summary["routing"] = {
            "counts": routing_totals,
            "pct": {
                key: (value / routing_total) if routing_total else 0.0
                for key, value in routing_totals.items()
            },
            "avg_change_count": routing_change_count / n if n else 0.0,
        }
    return summary, aggregate


def comparison_summary(hierarchical_summary, baseline_summary):
    return {
        "throughput_delta": hierarchical_summary["tokens_per_sec"] - baseline_summary["tokens_per_sec"],
        "benchmark_delta": hierarchical_summary["benchmark_score"] - baseline_summary["benchmark_score"],
        "avg_wall_time_delta": hierarchical_summary["avg_wall_time"] - baseline_summary["avg_wall_time"],
        "hierarchical_pass_rates": hierarchical_summary["usage"]["edge_pass_rates"],
        "baseline_pass_rates": baseline_summary["usage"]["edge_pass_rates"],
    }


def summary_from_merge_stats(run_name, cfg, aggregate_counters, merge_stats):
    n = merge_stats["n_samples"]
    total_tokens = merge_stats["total_tokens"]
    total_wall = merge_stats["total_wall_time"]
    total_score = merge_stats["total_score"]
    runtime_totals = merge_stats["runtime_totals"]
    runtime_total_wall = sum(
        runtime_totals[model_key]["total_wall_time"] for model_key in MODEL_KEYS
    )

    avg_model_runtime = {}
    for model_key in MODEL_KEYS:
        bucket = runtime_totals[model_key]
        avg_model_runtime[model_key] = {
            "propose_calls": (bucket["propose_calls"] / n) if n else 0.0,
            "review_calls": (bucket["review_calls"] / n) if n else 0.0,
            "propose_wall_time": (bucket["propose_wall_time"] / n) if n else 0.0,
            "review_wall_time": (bucket["review_wall_time"] / n) if n else 0.0,
            "total_wall_time": (bucket["total_wall_time"] / n) if n else 0.0,
            "share": (
                bucket["total_wall_time"] / runtime_total_wall
                if runtime_total_wall
                else 0.0
            ),
        }

    draft_window_totals = merge_stats.get(
        "draft_window_totals",
        {
            "small": {"sum": 0, "count": 0},
            "middle": {"sum": 0, "count": 0},
        },
    )
    avg_draft_window = {}
    for model_key, totals in draft_window_totals.items():
        avg_draft_window[model_key] = (
            totals["sum"] / totals["count"] if totals["count"] else 0.0
        )

    is_hierarchical = cfg.get("run_type") == "hierarchical" or run_name == "hierarchical"

    summary = {
        "run_name": run_name,
        "tokens_per_sec": (total_tokens / total_wall) if total_wall else 0.0,
        "avg_wall_time": (total_wall / n) if n else 0.0,
        "avg_tokens_generated": (total_tokens / n) if n else 0.0,
        "benchmark_metric": merge_stats["benchmark_metric"],
        "benchmark_score": (total_score / n) if n else 0.0,
        "usage": _per_sample_usage(aggregate_counters, total_tokens),
        "avg_model_runtime": avg_model_runtime,
        "avg_ml_forward_calls": (
            merge_stats["ml_forward_call_sum"] / n if n else 0.0
        ),
        "avg_mm_forward_calls": (
            merge_stats["mm_forward_call_sum"] / n if n else 0.0
        ),
        "n_samples": n,
        "small_window": cfg["small_window"],
        "middle_window": cfg.get("middle_window"),
        "hierarchical_variant": cfg.get("hierarchical_variant"),
        "middle_refill_policy": _double_layer_refill_policy(cfg)
        if is_hierarchical
        else None,
        "window_policy": cfg.get("window_policy", cfg.get("draft_window_policy", "fixed")),
        "adapt_small_window": bool(cfg.get("adapt_small_window", False)),
        "dynamic_utility_margin": cfg.get("dynamic_utility_margin", 0.0),
        "avg_draft_window": avg_draft_window,
        "avg_draft_window_changes": (
            merge_stats.get("window_change_count", 0) / n if n else 0.0
        ),
        "max_length": cfg["max_length"],
    }
    if merge_stats.get("routing_seen"):
        routing_totals = merge_stats.get("routing_totals", {"middle": 0, "large": 0})
        routing_total = sum(routing_totals.values())
        summary["routing"] = {
            "counts": routing_totals,
            "pct": {
                key: (value / routing_total) if routing_total else 0.0
                for key, value in routing_totals.items()
            },
            "avg_change_count": (
                merge_stats.get("routing_change_count", 0) / n if n else 0.0
            ),
        }
    return summary
