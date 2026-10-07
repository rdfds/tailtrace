"""Independent reload and accounting checks for the retained 0.4 observations."""

import json
from pathlib import Path

import pytest

from tailtrace.checkpoint_index import select, verify
from tailtrace.config import TrainConfig
from tailtrace.corpus import TokenCorpus, file_hash
from tailtrace.journal import inspect_run
from tailtrace.metrics_bench import make_fixture
from tailtrace.report import summarize_run

ROOT = Path(__file__).parents[1] / "results"


def test_retained_partial_write_and_corrupt_checkpoint_recover_exact_state():
    torch = pytest.importorskip("torch")
    from tailtrace.recovery import compare_states, load_state

    torch.set_num_threads(1)
    root = ROOT / "fault-recovery-cpu"
    proof = json.loads((root / "audit.json").read_text())
    cfg = TrainConfig.load(root / "whole-config.json")
    committed = verify(root / "failed/checkpoint-2", require_sealed=True)
    with TokenCorpus(root / "corpus") as corpus:
        assert corpus.sha256 == committed["dataset_sha256"]
    assert proof["runs"]["partial_write"]["returncode"] != 0
    assert (
        "TAILTRACE_PARTIAL_WRITE rank=1 step=4 bytes=4096 code=88"
        in (root / "partial-write.log").read_text()
    )
    assert any(p.stat().st_size == 4096 for p in (root / "failed/checkpoint-4").glob("*.distcp"))
    with pytest.raises(ValueError, match="no complete.json"):
        verify(root / "failed/checkpoint-4")
    with pytest.raises(ValueError, match="storage content changed"):
        verify(root / "failed/checkpoint-3")
    corruption = proof["corruption"]
    assert file_hash(root / corruption["path"]) == corruption["after_sha256"]
    assert corruption["before_byte"] ^ 1 == corruption["after_byte"]
    assert inspect_run(root / "failed") == proof["telemetry"]
    selected = select(
        root / "failed",
        cfg,
        2,
        dataset_sha256=committed["dataset_sha256"],
        torch_version=str(torch.__version__),
    )
    assert selected["step"] == proof["selection"]["step"] == 2
    assert selected["candidates"] == proof["selection"]["candidates"]
    states = [
        load_state(root / name / "checkpoint-6", cfg, committed["dataset_sha256"], None)
        for name in ("whole", "resumed")
    ]
    assert compare_states(*states) == proof["comparison"]


def test_retained_separate_node_agents_reproduce_canonical_state():
    torch = pytest.importorskip("torch")
    from tailtrace.recovery import compare_states, load_state

    torch.set_num_threads(1)
    root = ROOT / "logical-nodes-cpu"
    proof = json.loads((root / "audit.json").read_text())
    cfg = TrainConfig.load(root / "config.json")
    manifest = json.loads((root / "two_agents/manifest.json").read_text())
    topology = [
        [h[k] for k in ("rank", "local_rank", "group_rank", "local_world_size")]
        for h in manifest["hardware"]
    ]
    assert topology == proof["topology"] == [[0, 0, 0, 1], [1, 0, 1, 1]]
    assert proof["physical_hosts"] == 1
    assert len({h["hostname"] for h in manifest["hardware"]}) == 1
    assert all(
        agent["returncode"] == 0 and "--nnodes=2" in agent["command"] for agent in proof["agents"]
    )
    with TokenCorpus(root / "corpus") as corpus:
        assert corpus.sha256 == manifest["dataset"]["sha256"]
    states = [
        load_state(root / name / "checkpoint-6", cfg, manifest["dataset"]["sha256"], None)
        for name in ("single_agent", "two_agents")
    ]
    assert compare_states(*states) == proof["comparison"]
    assert all(
        inspect_run(root / name)["status"] == "complete" for name in ("single_agent", "two_agents")
    )


def test_retained_report_memory_inputs_and_aggregates_regenerate(tmp_path):
    proof = json.loads((ROOT / "telemetry-memory-cpu/benchmark.json").read_text())
    assert proof["input_evidence"] == "synthetic"
    for case in proof["cases"]:
        fixture = tmp_path / str(case["global_steps"])
        make_fixture(fixture, case["global_steps"])
        assert case["rank_rows"] == 2 * case["global_steps"]
        for entry in case["input_files"]:
            assert file_hash(fixture / entry["path"]) == entry["sha256"]
            assert (fixture / entry["path"]).stat().st_size == entry["bytes"]
        summary = summarize_run(fixture, retain_steps=False)
        external, retained = case["observations"]
        assert external["aggregates"] == retained["aggregates"]
        assert external["aggregates"] == {key: summary[key] for key in external["aggregates"]}
        assert external["retained_rows"] == 0 and retained["retained_rows"] == case["global_steps"]
        assert (
            case["python_peak_ratio"]
            == retained["python_peak_bytes"] / external["python_peak_bytes"]
        )
