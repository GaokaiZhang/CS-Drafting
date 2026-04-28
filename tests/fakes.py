import torch


class FakeTokenizer:
    TOKEN_MAP = {
        100: "Question prompt",
        101: "Reason the math question below step by step.",
        11: " The",
        12: " answer",
        13: " is",
        14: " (",
        15: "B",
        16: ")",
        17: ".",
        18: " wrong",
        19: " token",
        20: " path",
        21: " 42",
        22: "####",
        23: "42",
    }

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.TOKEN_MAP.get(int(token_id), f"<{int(token_id)}>") for token_id in ids)

    def __call__(self, text, truncation=True, padding=False, return_tensors="pt"):
        if "Question:" in text:
            token_id = 100
        else:
            token_id = 101
        return {"input_ids": torch.tensor([[token_id]], dtype=torch.long)}


class FakeModel:
    def __init__(self, proposal_sequence, review_sequence, name="llama"):
        self.device = torch.device("cpu")
        self.name = name
        self.vocab_size = 256
        self.proposal_sequence = proposal_sequence
        self.review_sequence = review_sequence
        self.forward_count = 0
        self.propose_count = 0
        self.review_count = 0
        self.wall_time = []
        self.propose_wall_time = []
        self.review_wall_time = []
        self.past_key_values = None
        self.past_ids = None

    def _generated_len(self, input_ids):
        return max(0, input_ids.shape[-1] - 1)

    def propose(self, initial_input, input_ids, k):
        self.propose_count += 1
        current = input_ids.clone()
        start = self._generated_len(current)
        for offset in range(k):
            index = start + offset
            if index >= len(self.proposal_sequence):
                break
            token = self.proposal_sequence[index]
            current = torch.cat([current, torch.tensor([[token]], dtype=torch.long)], dim=1)
            if token == 2:
                break
        return current

    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        self.forward_count += 1
        self.review_count += 1
        prefix_generated = max(0, review_index - 1)
        candidate = input_ids[0, review_index:].tolist()
        expected = self.review_sequence[prefix_generated:]
        accepted = 0
        for token, gold in zip(candidate, expected):
            if token == gold:
                accepted += 1
            else:
                break
        prefix = input_ids[:, :review_index]
        if accepted == len(candidate):
            next_index = prefix_generated + accepted
            if next_index < len(self.review_sequence):
                extra = [self.review_sequence[next_index]]
            else:
                extra = [2]
            return torch.cat(
                [prefix, input_ids[:, review_index:review_index + accepted], torch.tensor([extra], dtype=torch.long)],
                dim=1,
            ), None
        correction = [expected[accepted]]
        return torch.cat(
            [prefix, input_ids[:, review_index:review_index + accepted], torch.tensor([correction], dtype=torch.long)],
            dim=1,
        ), None
