import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from tailtrace.checkpoint import restore
from tailtrace.config import TrainConfig
from tailtrace.corpus import TokenCorpus, build_corpus
from tailtrace.model import CausalTransformer, make_batch
from tailtrace.train import run


def fixture(tmp_path):
    text = tmp_path / "text.txt"
    text.write_text(
        "A real byte corpus: CUDA, distributed gradients, and checkpoint recovery.\n" * 4
    )
    path = tmp_path / "corpus"
    build_corpus([text], path, max_length=16)
    config = TrainConfig(
        dataset_path=str(path),
        samples=8,
        min_length=2,
        max_length=16,
        batch_size=2,
        width=16,
        heads=2,
        layers=1,
        steps=6,
        warmup=0,
        strategy="single",
        checkpoint_every=6,
    )
    return path, config


def test_byte_collation_preserves_targets_and_masks_padding(tmp_path):
    path, config = fixture(tmp_path)
    with TokenCorpus(path) as corpus:
        ids = [0, len(corpus) - 1]
        lengths = [corpus.entry(i)[1] for i in range(len(corpus))]
        x, y = make_batch(ids, lengths, 256, config.seed, torch.device("cpu"), corpus)
        for row, i in enumerate(ids):
            n = lengths[i] - 1
            assert x[row, :n].tolist() == list(corpus[i][:-1])
            assert y[row, :n].tolist() == list(corpus[i][1:])
            assert (y[row, n:] == -100).all()


def test_real_text_mid_epoch_resume_restores_model_and_adamw_exactly(tmp_path):
    path, config = fixture(tmp_path)
    run(config, tmp_path / "whole")
    run(replace(config, steps=3, checkpoint_every=3), tmp_path / "partial")
    run(config, tmp_path / "resumed", str(tmp_path / "partial/checkpoint-3"))
    with TokenCorpus(path) as corpus:
        states = []
        for name in ("whole", "resumed"):
            model = CausalTransformer(config)
            optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
            assert (
                restore(
                    tmp_path / name / "checkpoint-6",
                    model,
                    optim,
                    config,
                    1,
                    dataset_sha256=corpus.sha256,
                )
                == 6
            )
            states.append((model.state_dict(), optim.state_dict()))
        for key in states[0][0]:
            torch.testing.assert_close(states[0][0][key], states[1][0][key], rtol=0, atol=0)
        for key, value in states[0][1]["state"].items():
            for field, tensor in value.items():
                torch.testing.assert_close(
                    tensor, states[1][1]["state"][key][field], rtol=0, atol=0
                )
        manifest = json.loads((tmp_path / "resumed/manifest.json").read_text())
        assert manifest["dataset"]["sha256"] == corpus.sha256
        assert manifest["start_step"] == 3


def test_dataset_identity_rejected_before_checkpoint_tensor_loading(tmp_path):
    path, config = fixture(tmp_path)
    run(replace(config, steps=1, checkpoint_every=1), tmp_path / "partial")
    with pytest.raises(ValueError, match="dataset content changed"):
        restore(
            tmp_path / "partial/checkpoint-1", None, None, config, 1, dataset_sha256="different"
        )
    payload = path / "tokens.bin"
    with TokenCorpus(path) as corpus:
        payload.write_bytes(b"X" + payload.read_bytes()[1:])
        with pytest.raises(ValueError, match="changed during training"):
            corpus[0]


def test_corpus_preflight_rejects_invalid_training_shapes(tmp_path):
    _, config = fixture(tmp_path)
    with pytest.raises(ValueError, match="samples exceeds"):
        run(replace(config, samples=1000), tmp_path / "bad-count")
    with pytest.raises(ValueError, match="exceed model"):
        run(replace(config, max_length=8), tmp_path / "bad-length")
    with pytest.raises(ValueError, match="vocab_size=256"):
        replace(config, vocab_size=32)
