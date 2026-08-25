import json
from argparse import Namespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("ray")

from tailtrace.evidence import atomic_json
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
    assert (tmp_path / "run" / "summary.json").exists()
