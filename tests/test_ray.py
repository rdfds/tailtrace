import json
from argparse import Namespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("ray")

from tailtrace.checkpoint_index import verify
from tailtrace.evidence import atomic_json
from tailtrace.journal import inspect_run
from tailtrace.ray_runner import launch


@pytest.mark.ray
def test_two_cpu_workers(tmp_path):
    config = tmp_path / "config.json"
    atomic_json(
        config,
        {
            "device": "cpu",
            "strategy": "ddp",
            "steps": 3,
            "checkpoint_every": 3,
            "metrics_flush_every": 1,
            "warmup": 1,
            "width": 16,
            "heads": 2,
            "layers": 1,
            "max_length": 24,
            "min_length": 2,
        },
    )
    launch(
        Namespace(
            config=str(config),
            out=str(tmp_path / "run"),
            workers=2,
            address=None,
            storage=str(tmp_path / "ray-storage"),
        )
    )
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["world_size"] == 2
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["measured_steps"] == 2 and not summary["steps_retained"]
    assert manifest["telemetry"]["peak_buffered_rows"] == 1
    assert manifest["instrumented"]
    assert all(p["rows"] == 3 and p["complete"] for p in manifest["telemetry"]["rank_progress"])
    assert len({h["source_sha256"] for h in manifest["hardware"]}) == 1
    assert inspect_run(tmp_path / "run")["status"] == "complete"
    assert verify(tmp_path / "run/checkpoint-3", require_sealed=True)["step"] == 3
