import copy

import pytest

from tailtrace.config import TrainConfig
from tailtrace.fleet import FleetProfile, workload_signature


def test_hash_tracks_content_not_key_order_and_declared_is_not_observed():
    data = {
        "schema_version": 1,
        "evidence": "declared",
        "units": "proxy",
        "ranks": [{"model": {"quadratic": 1, "max_samples": 8}}],
    }
    a = FleetProfile(data)
    assert a.sha256 == FleetProfile(dict(reversed(list(data.items())))).sha256
    data["ranks"][0]["model"]["quadratic"] = 2
    assert a.sha256 != FleetProfile(data).sha256
    assert a.summary()["evidence"] == "declared"


def test_observed_requires_scope_and_rejects_workload_and_domain_drift():
    config = TrainConfig()
    data = {
        "schema_version": 1,
        "evidence": "observed_isolated",
        "units": "ms",
        "scope": "forward_backward_no_collectives",
        "workload": workload_signature(config),
        "hardware": [{"device": "cpu"}],
        "ranks": [
            {
                "model": {"quadratic": 1, "max_samples": 8},
                "domain": {"min_batch": 1, "max_batch": 8, "min_width": 8, "max_width": 128},
            }
        ],
    }
    fleet = FleetProfile(data)
    fleet.check_workload(config, 1)
    with pytest.raises(ValueError, match="workload changed"):
        fleet.check_workload(TrainConfig(width=128), 1)
    with pytest.raises(ValueError, match="outside"):
        fleet.check_domains([[0]], [3])
    bad = copy.deepcopy(data)
    bad["scope"] = "ddp_step"
    with pytest.raises(ValueError, match="isolated scope"):
        FleetProfile(bad)
