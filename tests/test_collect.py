import json

import pytest

pytest.importorskip("torch")

from tailtrace.collect import collect
from tailtrace.config import TrainConfig
from tailtrace.fleet import FleetProfile


def test_actual_cpu_calibration_retains_raw_samples_and_scope(tmp_path):
    config = TrainConfig(width=16, heads=2, layers=1, min_length=2, max_length=16, vocab_size=32)
    out = tmp_path / "calibration"
    profile = collect(config, [1, 2, 4, 8], [4, 8, 12, 16], 3, 1, out)
    raw = json.loads((out / "observations.json").read_text())
    assert len(raw["ranks"][0]["rows"]) == 48
    assert raw["timing"] == "cpu_wall"
    assert profile["scope"] == "forward_backward_no_collectives"
    assert len(profile["ranks"][0]["fit"]["heldout_shapes"]) == 4
    loaded = FleetProfile.load(out / "fleet.json")
    loaded.check_workload(config, 1)
    loaded.check_hardware(0, profile["hardware"][0])
    with pytest.raises(FileExistsError):
        collect(config, [1, 2, 4, 8], [4, 8, 12, 16], 3, 1, out)
