"""
Adaptive Cascaded Speculative Decoding (ACSD)

Three-tier system:
  M_s  — small fast drafter (e.g. TinyLlama-1.1B)
  M_m  — middle model (e.g. LLaMA-2-7B), acts as pre-verifier or drafter
  M_l  — large final verifier (e.g. LLaMA-2-13B), always runs last

Phase 2 (acsd_cascaded):
    M_s drafts → M_m pre-verifies → M_l final-verifies filtered sequence
    M_l only sees tokens M_m accepted, reducing M_l forward-pass cost.

Phase 3 (acsd_adaptive):
    Same as Phase 2, but when M_s rolling acceptance rate alpha < tau,
    M_m is promoted to drafter and M_s is bypassed entirely.
    M_l always runs final verification — losslessness is preserved.
"""

import torch
from collections import deque
from dataclasses import dataclass, field


# ── helpers ────────────────────────────────────────────────────────────────────

def _eos_token(model):
    return 2 if 't5' not in model.name.lower() else 1


def _make_one_hot_probs(token_ids, vocab_size, device):
    """
    Build a one-hot probability tensor from a greedy token sequence.
    Shape: [1, seq_len, vocab_size]
    """
    return torch.nn.functional.one_hot(
        token_ids.squeeze(0), num_classes=vocab_size
    ).float().unsqueeze(0).to(device)


# ── adaptive state ─────────────────────────────────────────────────────────────

@dataclass
class AdaptiveCSDState:
    """
    Tracks M_s rolling acceptance rate and decides which model drafts next.
    """
    tau: float = 0.4
    window_size: int = 20
    alpha_window: deque = field(default_factory=deque)
    current_drafter: str = 'ms'   # 'ms' or 'mm'

    def update(self, n_accepted: int, n_drafted: int):
        ratio = n_accepted / n_drafted if n_drafted > 0 else 1.0
        self.alpha_window.append(ratio)
        if len(self.alpha_window) > self.window_size:
            self.alpha_window.popleft()

    @property
    def rolling_alpha(self) -> float:
        if not self.alpha_window:
            return 1.0
        return sum(self.alpha_window) / len(self.alpha_window)

    def maybe_switch(self):
        """Update current_drafter based on rolling alpha."""
        if self.rolling_alpha < self.tau:
            self.current_drafter = 'mm'
        else:
            self.current_drafter = 'ms'


# ── Phase 2: cascaded pre-verification ────────────────────────────────────────

def acsd_cascaded(m_s, m_m, m_l, initial_input, input_ids,
                  k_s=5, leniency=1, max_length=200):
    """
    Two-tier cascade with M_m as a pre-verifier gate.

    Each step:
      1. M_s proposes k_s draft tokens (greedy)
      2. M_m reviews those drafts, accepting only ones it agrees with
      3. M_l reviews the filtered (shorter) sequence

    M_l forward cost is reduced because it processes fewer draft positions.
    Losslessness: M_l always performs final verification.

    Returns:
        input_ids: generated token sequence
    """
    _EOS = _eos_token(m_l)
    initial_len = input_ids.shape[-1]
    initial_input = initial_input.to(m_l.device)
    input_ids = input_ids.to(m_l.device)

    with torch.no_grad():
        while input_ids.shape[-1] - initial_len < max_length:
            review_index = input_ids.shape[-1]

            # 1. M_s drafts k_s tokens
            draft_ids = m_s.propose(
                initial_input.to(m_s.device),
                input_ids.to(m_s.device),
                k_s,
            )

            # 2. Build one-hot probs for M_s's greedy tokens
            ms_probs = _make_one_hot_probs(draft_ids, m_m.vocab_size, m_m.device)

            # 3. M_m pre-verifies (filters draft tokens it disagrees with)
            filtered_ids, mm_probs = m_m.pre_verify(
                initial_input.to(m_m.device),
                draft_ids.to(m_m.device),
                ms_probs,
                review_index,
                leniency=leniency,
            )

            # 4. M_l final-verifies the filtered sequence (leniency=1, always strict)
            if mm_probs is None:
                mm_probs = _make_one_hot_probs(filtered_ids, m_l.vocab_size, m_l.device)
            input_ids, _ = m_l.review(
                initial_input.to(m_l.device),
                filtered_ids.to(m_l.device),
                mm_probs.to(m_l.device),
                review_index,
                leniency=1,
            )

            if input_ids.shape[-1] <= review_index:
                break
            if torch.any(input_ids[0, review_index:] == _EOS):
                break

    return input_ids


# ── Phase 3: adaptive role-switching ──────────────────────────────────────────

def acsd_adaptive(m_s, m_m, m_l, initial_input, input_ids,
                  k_s=5, k_m=4, leniency=1,
                  tau=0.4, window_size=20, max_length=200):
    """
    Adaptive cascade: M_s drafts by default, but is bypassed when its rolling
    acceptance rate drops below tau, at which point M_m takes over as drafter.

    State machine per step:
      current_drafter == 'ms':
          M_s → M_m.pre_verify → M_l.review   (Phase 2 fast path)
      current_drafter == 'mm':
          M_m → M_l.review                     (M_m drafts, M_s idle)

    M_l always runs final verification — losslessness is preserved.

    Returns:
        input_ids: generated token sequence
        state: AdaptiveCSDState with acceptance history and switch events
    """
    _EOS = _eos_token(m_l)
    initial_len = input_ids.shape[-1]
    initial_input = initial_input.to(m_l.device)
    input_ids = input_ids.to(m_l.device)
    state = AdaptiveCSDState(tau=tau, window_size=window_size)
    mm_as_verifier = True   # whether M_m's KV cache is valid for verifier role

    with torch.no_grad():
        while input_ids.shape[-1] - initial_len < max_length:
            review_index = input_ids.shape[-1]

            if state.current_drafter == 'ms':
                # ── Fast path: M_s drafts → M_m pre-verifies → M_l verifies ──
                draft_ids = m_s.propose(
                    initial_input.to(m_s.device),
                    input_ids.to(m_s.device),
                    k_s,
                )
                n_drafted = draft_ids.shape[-1] - review_index
                ms_probs = _make_one_hot_probs(draft_ids, m_m.vocab_size, m_m.device)

                filtered_ids, mm_probs = m_m.pre_verify(
                    initial_input.to(m_m.device),
                    draft_ids.to(m_m.device),
                    ms_probs,
                    review_index,
                    leniency=leniency,
                )
                mm_as_verifier = True   # mark so next drafter-switch invalidates cache

                if mm_probs is None:
                    mm_probs = _make_one_hot_probs(filtered_ids, m_l.vocab_size, m_l.device)
                input_ids, _ = m_l.review(
                    initial_input.to(m_l.device),
                    filtered_ids.to(m_l.device),
                    mm_probs.to(m_l.device),
                    review_index,
                    leniency=1,
                )

                # Update M_s rolling acceptance rate
                n_accepted = input_ids.shape[-1] - review_index
                state.update(n_accepted, n_drafted)

            else:
                # ── Slow path: M_m drafts directly → M_l verifies ──
                # Invalidate M_m's KV cache on first step in drafter role
                if mm_as_verifier:
                    m_m.past_key_values = None
                    m_m.past_ids = None
                    mm_as_verifier = False

                draft_ids = m_m.propose(
                    initial_input.to(m_m.device),
                    input_ids.to(m_m.device),
                    k_m,
                )
                mm_probs = _make_one_hot_probs(draft_ids, m_l.vocab_size, m_l.device)
                input_ids, _ = m_l.review(
                    initial_input.to(m_l.device),
                    draft_ids.to(m_l.device),
                    mm_probs,
                    review_index,
                    leniency=1,
                )
                # Do not update M_s alpha while M_m is drafting

            # Decide who drafts next step
            state.maybe_switch()

            if input_ids.shape[-1] <= review_index:
                break
            if torch.any(input_ids[0, review_index:] == _EOS):
                break

    return input_ids, state
