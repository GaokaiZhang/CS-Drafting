import os
import sys

from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csd_datasets


def test_mmlu_falls_back_to_subject_caches(monkeypatch):
    calls = []

    def fake_load_dataset(name, config, split):
        calls.append((name, config, split))
        if config == "all":
            raise ValueError("Couldn't find cache for cais/mmlu for config 'all'")
        return Dataset.from_list(
            [
                {
                    "question": f"question-{config}",
                    "choices": ["A", "B", "C", "D"],
                    "answer": 0,
                }
            ]
        )

    monkeypatch.setattr(csd_datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(csd_datasets, "MMLU_SUBJECTS", ["subject_a", "subject_b"])

    rows = csd_datasets.get_test_set("mmlu")

    assert calls == [
        ("cais/mmlu", "all", "test"),
        ("cais/mmlu", "subject_a", "test"),
        ("cais/mmlu", "subject_b", "test"),
    ]
    assert len(rows) == 2
    assert {row["subject"] for row in rows} == {"subject_a", "subject_b"}
