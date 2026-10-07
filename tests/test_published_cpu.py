import json
import math
import statistics
from pathlib import Path

import pytest

from tailtrace.certificate import certify
from tailtrace.config import TrainConfig
from tailtrace.corpus import TokenCorpus
from tailtrace.cost import RankCost
from tailtrace.scheduler import improve_reference

ROOT = Path(__file__).parents[1] / "results"


def test_published_scheduler_timings_recompute_ratios_and_exact_assignments():
    data = json.loads((ROOT / "planner-cpu/benchmark.json").read_text())
    assert data["evidence"] == "observed_cpu"
    assert len(data["cases"]) == 4
    for case in data["cases"]:
        models = [RankCost(**m) for m in case["models"]]
        reference = improve_reference(
            case["initial_groups"], case["lengths"], models, data["rounds"]
        )
        assert reference == case["assignment"]
        certify(range(case["samples"]), case["lengths"], reference, models)
        assert len(case["observations"]) == data["repeats"]
        for observation in case["observations"]:
            assert set(observation["order"]) == {"full_rescoring", "incremental"}
            assert all(math.isfinite(ms) and ms > 0 for ms in observation["ms"].values())
        medians = {
            name: statistics.median(o["ms"][name] for o in case["observations"])
            for name in ("full_rescoring", "incremental")
        }
        assert medians == case["median_ms"]
        assert medians["full_rescoring"] / medians["incremental"] == case["median_ratio"]


def test_published_distributed_checkpoints_reproduce_recovery_state_hash():
    torch = pytest.importorskip("torch")
    from tailtrace.recovery import compare_states, load_state

    torch.set_num_threads(1)
    root = ROOT / "recovery-cpu"
    proof = json.loads((root / "audit.json").read_text())
    config = TrainConfig.load(root / "whole-config.json")
    with TokenCorpus(root / "corpus") as corpus:
        assert corpus.sha256 == proof["dataset_sha256"]
    assert proof["runs"]["crashed"]["returncode"] != 0
    assert proof["runs"]["uninterrupted"]["returncode"] == 0
    assert proof["runs"]["resumed"]["returncode"] == 0
    assert "TAILTRACE_INJECTED_EXIT rank=1 code=86" in (root / "crash.log").read_text()
    assert not (root / "crashed/manifest.json").exists()
    metadata = json.loads((root / "crashed/checkpoint-2/complete.json").read_text())
    assert metadata["step"] == proof["fault_step"]
    manifest = json.loads((root / "resumed/manifest.json").read_text())
    assert manifest["start_step"] == proof["fault_step"]
    assert manifest["dataset"]["sha256"] == proof["dataset_sha256"]
    states = [
        load_state(
            root / name / "checkpoint-6",
            config,
            proof["dataset_sha256"],
            metadata.get("fleet_sha256"),
        )
        for name in ("whole", "resumed")
    ]
    assert compare_states(*states) == proof["comparison"]
