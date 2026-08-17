from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from tailtrace.checkpoint import restore
from tailtrace.config import TrainConfig
from tailtrace.model import CausalTransformer
from tailtrace.train import run


def test_resume_mid_epoch_matches_uninterrupted_weights(tmp_path):
    config = TrainConfig(
        samples=8,
        batch_size=2,
        steps=8,
        warmup=1,
        min_length=2,
        max_length=8,
        width=16,
        heads=2,
        layers=1,
        checkpoint_every=8,
    )
    run(config, tmp_path / "full")
    run(replace(config, steps=6, checkpoint_every=5), tmp_path / "partial")
    run(config, tmp_path / "resumed", str(tmp_path / "partial" / "checkpoint-5"))
    models = [CausalTransformer(config) for _ in range(2)]
    for model, name in zip(models, ("full", "resumed"), strict=True):
        optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
        assert restore(tmp_path / name / "checkpoint-8", model, optim, config, 1) == 8
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(FileExistsError, match="already exists"):
        run(config, tmp_path / "full")


def test_profiler_emits_readable_rank_trace(tmp_path):
    from tailtrace.traces import analyze_trace

    config = TrainConfig(
        samples=16,
        batch_size=2,
        steps=5,
        warmup=1,
        min_length=2,
        max_length=8,
        width=16,
        heads=2,
        layers=1,
        profile=True,
        profile_steps=2,
    )
    run(config, tmp_path)
    trace = analyze_trace(tmp_path / "trace-rank0.json")
    assert trace["gpu_events"] == 0
    assert len(trace["steps"]) == 2
