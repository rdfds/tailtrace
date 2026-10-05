import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from tailtrace.config import TrainConfig
from tailtrace.evidence import atomic_json
from tailtrace.fleet import FleetProfile
from tailtrace.train import run


def test_single_rank_fleet_resume_pins_profile_content(tmp_path):
    path = tmp_path / "fleet.json"
    data = {
        "schema_version": 1,
        "evidence": "declared",
        "units": "proxy",
        "ranks": [{"model": {"quadratic": 1, "max_samples": 4}}],
    }
    atomic_json(path, data)
    config = TrainConfig(
        samples=8,
        batch_size=2,
        steps=4,
        warmup=1,
        width=16,
        heads=2,
        layers=1,
        min_length=2,
        max_length=8,
        vocab_size=32,
        planner="fleet",
        fleet_path=str(path),
        checkpoint_every=2,
    )
    run(config, tmp_path / "first")
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert manifest["fleet"]["sha256"] == FleetProfile.load(path).sha256
    run(replace(config, steps=6), tmp_path / "resumed", str(tmp_path / "first" / "checkpoint-4"))
    data["ranks"][0]["model"]["quadratic"] = 2
    atomic_json(path, data)
    with pytest.raises(ValueError, match="profile content changed"):
        run(
            replace(config, steps=6), tmp_path / "invalid", str(tmp_path / "first" / "checkpoint-4")
        )


def test_bad_profile_preflight_fails_before_training(tmp_path):
    config = TrainConfig(planner="fleet", fleet_path=str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="fleet preflight failed"):
        run(config, tmp_path / "run")
