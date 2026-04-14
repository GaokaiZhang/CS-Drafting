"""
Double-layer speculative decoding with optional proxy-guided M_s drafting.

Three-tier model system:
  M_s — small fast drafter      (TinyLlama-1.1B)
  M_m — middle verifier         (LLaMA-2-7B)   intermediate, always
  M_l — large final verifier    (LLaMA-2-13B)  final, always

acsd_double_layer  [base method]
    Per outer step:
      inner loop until k_m tokens accumulated from outer_review_index:
        1. M_s proposes k_s tokens greedily
        2. M_m verifies: accepts longest prefix + adds one correction token
        (repeat from new position)
      M_l verifies the full k_m-token batch, accepting a prefix + one M_l token.
    k_m > k_s so M_l sees a larger, M_m-filtered batch per forward pass.
    Losslessness: M_l always has final say.

acsd_proxy  [full method]
    Same inner-loop cascade, but M_s consults a local confidence proxy after
    each draft token and stops early when below threshold.
    Proxy types: entropy | top1 | margin | mavg  (see propose_with_proxy in model.py).
"""

import torch


# ── helpers ────────────────────────────────────────────────────────────────────

def _eos_token(model):
    return 2 if 't5' not in model.name.lower() else 1


def _make_one_hot_probs(token_ids, vocab_size, device):
    """
    One-hot probability tensor from a greedy token sequence.
    Convention: probs[k] = one-hot of token k+1 (first token dropped).
    Shape: [1, seq_len - 1, vocab_size]
    """
    return torch.nn.functional.one_hot(
        token_ids.squeeze(0)[1:], num_classes=vocab_size
    ).float().unsqueeze(0).to(device)


# ── base method ────────────────────────────────────────────────────────────────

def acsd_double_layer(m_s, m_m, m_l, initial_input, input_ids,
                      k_s=5, k_m=10, leniency=1, max_length=200):
    """
    Double-layer speculative decoding.

    Each outer step:
      inner loop (until k_m tokens accumulated):
        M_s.propose(k_s)  →  M_m.pre_verify()  →  append & advance
      M_l.review(k_m tokens)

    Returns: generated token sequence (tensor).
    """
    _EOS = _eos_token(m_l)
    initial_len = input_ids.shape[-1]
    initial_input = initial_input.to(m_l.device)
    input_ids = input_ids.to(m_l.device)

    with torch.no_grad():
        while input_ids.shape[-1] - initial_len < max_length:
            outer_review_index = input_ids.shape[-1]
            cur_ids = input_ids

            # ── inner loop: M_s → M_m cascade ────────────────────────────────
            while cur_ids.shape[-1] - outer_review_index < k_m:
                inner_review = cur_ids.shape[-1]

                ms_draft = m_s.propose(
                    initial_input.to(m_s.device),
                    cur_ids.to(m_s.device),
                    k_s,
                )
                ms_probs = _make_one_hot_probs(ms_draft, m_m.vocab_size, m_m.device)
                mm_verified, _ = m_m.pre_verify(
                    initial_input.to(m_m.device),
                    ms_draft.to(m_m.device),
                    ms_probs,
                    inner_review,
                    leniency=leniency,
                )
                mm_verified = mm_verified.to(m_l.device)

                if mm_verified.shape[-1] - outer_review_index >= k_m:
                    cur_ids = mm_verified[:, :outer_review_index + k_m]
                    break

                cur_ids = mm_verified
                if torch.any(mm_verified[0, inner_review:] == _EOS):
                    break

            # ── outer: M_l final verification ────────────────────────────────
            ml_probs = _make_one_hot_probs(cur_ids, m_l.vocab_size, m_l.device)
            input_ids, _ = m_l.review(
                initial_input.to(m_l.device),
                cur_ids.to(m_l.device),
                ml_probs,
                outer_review_index,
                leniency=1,
            )

            if input_ids.shape[-1] <= outer_review_index:
                break
            if torch.any(input_ids[0, outer_review_index:] == _EOS):
                break

    return input_ids


# ── full method ────────────────────────────────────────────────────────────────

def acsd_proxy(m_s, m_m, m_l, initial_input, input_ids,
               k_s=5, k_m=10,
               proxy_type='top1', threshold=-1.5, mavg_window=5,
               leniency=1, max_length=200):
    """
    Proxy-guided double-layer speculative decoding.

    Same inner loop as acsd_double_layer but M_s uses a local confidence proxy
    (propose_with_proxy) to stop drafting early when uncertain.  Fewer bad tokens
    reach M_m, improving inner-loop efficiency.

    proxy_type / threshold convention:
      'entropy'  stop if H(p) > threshold          (threshold > 0, e.g. 2.0)
      'top1'     stop if log p_max < threshold      (threshold < 0, e.g. -1.5)
      'margin'   stop if log p_1 - log p_2 < thr   (threshold > 0, e.g. 1.0)
      'mavg'     stop if mavg(log p_max) < thr      (threshold < 0, e.g. -1.5)

    Returns: generated token sequence (tensor).
    """
    _EOS = _eos_token(m_l)
    initial_len = input_ids.shape[-1]
    initial_input = initial_input.to(m_l.device)
    input_ids = input_ids.to(m_l.device)

    with torch.no_grad():
        while input_ids.shape[-1] - initial_len < max_length:
            outer_review_index = input_ids.shape[-1]
            cur_ids = input_ids

            # ── inner loop: proxy-guided M_s → M_m cascade ───────────────────
            while cur_ids.shape[-1] - outer_review_index < k_m:
                inner_review = cur_ids.shape[-1]

                ms_draft = m_s.propose_with_proxy(
                    initial_input.to(m_s.device),
                    cur_ids.to(m_s.device),
                    k_s,
                    proxy_type=proxy_type,
                    threshold=threshold,
                    mavg_window=mavg_window,
                )
                ms_probs = _make_one_hot_probs(ms_draft, m_m.vocab_size, m_m.device)
                mm_verified, _ = m_m.pre_verify(
                    initial_input.to(m_m.device),
                    ms_draft.to(m_m.device),
                    ms_probs,
                    inner_review,
                    leniency=leniency,
                )
                mm_verified = mm_verified.to(m_l.device)

                if mm_verified.shape[-1] - outer_review_index >= k_m:
                    cur_ids = mm_verified[:, :outer_review_index + k_m]
                    break

                cur_ids = mm_verified
                if torch.any(mm_verified[0, inner_review:] == _EOS):
                    break

            # ── outer: M_l final verification ────────────────────────────────
            ml_probs = _make_one_hot_probs(cur_ids, m_l.vocab_size, m_l.device)
            input_ids, _ = m_l.review(
                initial_input.to(m_l.device),
                cur_ids.to(m_l.device),
                ml_probs,
                outer_review_index,
                leniency=1,
            )

            if input_ids.shape[-1] <= outer_review_index:
                break
            if torch.any(input_ids[0, outer_review_index:] == _EOS):
                break

    return input_ids
