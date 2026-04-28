import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from main_fixed_window import (
    _harmonize_model_vocab_sizes,
    _resolve_local_snapshot,
    _select_tokenizer_path,
)
from model import CountedCSDraftingDecoderModelKVCache


def _make_snapshot(root, model_name, revision):
    namespace, repo = model_name.split("/", 1)
    repo_cache = root / f"models--{namespace}--{repo}"
    snapshot_dir = repo_cache / "snapshots" / revision
    snapshot_dir.mkdir(parents=True)
    return repo_cache, snapshot_dir


def test_resolve_local_snapshot_prefers_hf_hub_cache_ref(tmp_path, monkeypatch):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    hub_cache = tmp_path / "hub"
    repo_cache, snapshot_dir = _make_snapshot(hub_cache, model_name, "rev-main")
    (repo_cache / "refs").mkdir(parents=True)
    (repo_cache / "refs" / "main").write_text("rev-main")

    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.delenv("HF_HOME", raising=False)

    assert _resolve_local_snapshot(model_name) == str(snapshot_dir)


def test_resolve_local_snapshot_falls_back_to_hf_home_hub(tmp_path, monkeypatch):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    hf_home = tmp_path / "hf-home"
    _, snapshot_dir = _make_snapshot(hf_home / "hub", model_name, "rev-alt")

    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(hf_home))

    assert _resolve_local_snapshot(model_name) == str(snapshot_dir)


def test_resolve_local_snapshot_returns_model_name_when_cache_missing(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    model_name = "Qwen/Qwen2.5-14B-Instruct"
    assert _resolve_local_snapshot(model_name) == model_name


def test_resolve_local_snapshot_keeps_absolute_path_even_if_missing():
    model_name = "/data/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/rev"
    assert _resolve_local_snapshot(model_name) == model_name


def test_harmonize_model_vocab_sizes_wraps_review_to_common_vocab():
    class DummyModel:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.seen_vocab_sizes = []
            self.device = torch.device("cpu")
            self.model = None

        def review(self, initial_input, input_ids, probs, review_index, leniency=1):
            self.seen_vocab_sizes.append(probs.shape[-1])
            return input_ids, probs

    small = DummyModel(4)
    large = DummyModel(6)
    _harmonize_model_vocab_sizes([small, None, large])

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


def test_select_tokenizer_path_prefers_small_model_snapshot():
    assert _select_tokenizer_path("/tmp/small", "/tmp/large") == "/tmp/small"
    assert _select_tokenizer_path("", "/tmp/large") == "/tmp/large"


def test_counted_kv_cache_wrapper_preserves_native_vocab_size():
    class DummyHFModel:
        def __init__(self, vocab_size):
            self.config = type("Config", (), {"vocab_size": vocab_size})()
            self.device = torch.device("cpu")

    wrapped = CountedCSDraftingDecoderModelKVCache(
        DummyHFModel(151936),
        name="Qwen/Qwen2.5-0.5B-Instruct",
        vocab_size=151936,
    )

    assert wrapped.vocab_size == 151936


def test_counted_kv_cache_wrapper_uses_embedding_vocab_over_config():
    class DummyEmbeddings:
        def __init__(self, vocab_size):
            self.num_embeddings = vocab_size

    class DummyHFModel:
        def __init__(self, config_vocab, embedding_vocab):
            self.config = type("Config", (), {"vocab_size": config_vocab})()
            self.device = torch.device("cpu")
            self.embeddings = DummyEmbeddings(embedding_vocab)

        def get_input_embeddings(self):
            return self.embeddings

        def get_output_embeddings(self):
            return self.embeddings

    wrapped = CountedCSDraftingDecoderModelKVCache(
        DummyHFModel(config_vocab=10, embedding_vocab=4),
        name="dummy",
        vocab_size=10,
    )

    assert wrapped.vocab_size == 4
    assert wrapped.native_vocab_size == 4


def test_harmonize_generation_vocab_uses_embedding_vocab_limit():
    class DummyEmbeddings:
        def __init__(self, vocab_size):
            self.num_embeddings = vocab_size

    class DummyHFModel:
        def __init__(self, config_vocab, embedding_vocab):
            self.config = type("Config", (), {"vocab_size": config_vocab})()
            self.device = torch.device("cpu")
            self.embeddings = DummyEmbeddings(embedding_vocab)

        def get_input_embeddings(self):
            return self.embeddings

        def get_output_embeddings(self):
            return self.embeddings

    small = CountedCSDraftingDecoderModelKVCache(
        DummyHFModel(config_vocab=10, embedding_vocab=4),
        name="small",
        vocab_size=10,
    )
    large = CountedCSDraftingDecoderModelKVCache(
        DummyHFModel(config_vocab=12, embedding_vocab=8),
        name="large",
        vocab_size=12,
    )

    _harmonize_model_vocab_sizes([small, None, large])

    assert small.native_vocab_size == 4
    assert large.native_vocab_size == 8
    assert small.generation_vocab_size == 4
    assert large.generation_vocab_size == 4
