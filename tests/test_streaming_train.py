import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from tailtrace.checkpoint import restore
from tailtrace.config import TrainConfig
from tailtrace.journal import inspect_run
from tailtrace.model import CausalTransformer
from tailtrace.recovery import compare_states
from tailtrace.report import summarize_run
from tailtrace.train import run


def config():
    return TrainConfig(
        samples=8,
        batch_size=2,
        steps=17,
        warmup=1,
        min_length=2,
        max_length=8,
        width=16,
        heads=2,
        layers=1,
        vocab_size=32,
        metrics_flush_every=3,
        checkpoint_every=17,
    )


def test_bounded_metrics_preserve_all_steps_and_exact_training_state(tmp_path):
    cfg = config()
    run(cfg, tmp_path / "bounded")
    run(replace(cfg, metrics_flush_every=0), tmp_path / "deferred")
    states = []
    for name in ("bounded", "deferred"):
        model = CausalTransformer(cfg)
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        assert restore(tmp_path / name / "checkpoint-17", model, optim, cfg, 1) == 17
        states.append({"model": model.state_dict(), "optimizer": optim.state_dict(), "step": 17})
        assert inspect_run(tmp_path / name)["status"] == "complete"
        assert inspect_run(tmp_path / name)["durable_through_step"] == 16
    assert compare_states(*states)["max_tensor_error"] == 0
    bounded = json.loads((tmp_path / "bounded/manifest.json").read_text())
    deferred = json.loads((tmp_path / "deferred/manifest.json").read_text())
    assert bounded["telemetry"]["peak_buffered_rows"] == 3
    assert deferred["telemetry"]["peak_buffered_rows"] == 17
    assert bounded["instrumented"]
    assert not deferred["instrumented"]
    assert bounded["telemetry"]["rank_progress"][0]["rows"] == 17
    assert len(summarize_run(tmp_path / "bounded")["steps"]) == 16
    summary = json.loads((tmp_path / "bounded/summary.json").read_text())
    assert not summary["steps_retained"] and not summary["steps"]
    assert summary["quantile_method"] == "exact_external_sqlite_sort"


def test_completed_telemetry_corruption_and_extra_suffix_are_rejected(tmp_path):
    run(config(), tmp_path)
    path = tmp_path / "metrics-rank0.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + b"extra suffix")
    with pytest.raises(ValueError, match="journal differs"):
        summarize_run(tmp_path)
    path.write_bytes(b"!" + original[1:])
    with pytest.raises(ValueError, match="prefix changed"):
        summarize_run(tmp_path)


def test_partial_output_reuse_cannot_overwrite_diagnostics(tmp_path):
    cfg = config()
    (tmp_path / "run.lock").touch()
    path = tmp_path / "failure-rank0.json"
    path.write_text('{"old": "failure"}')
    with pytest.raises(FileExistsError, match="preflight failed"):
        run(cfg, tmp_path)
    assert path.read_text() == '{"old": "failure"}'


@pytest.mark.parametrize("value", [-1, True, 2.5])
def test_metrics_cadence_validation(value):
    with pytest.raises(ValueError, match="metrics_flush_every"):
        replace(config(), metrics_flush_every=value)
