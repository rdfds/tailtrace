import json
from copy import deepcopy

import pytest

from tailtrace.evidence import atomic_json
from tailtrace.report import compare_runs, summarize_run, write_html


def fixture_run(path, seed, planner, duration=10):
    manifest = {
        "schema_version": 1,
        "evidence": "observed",
        "world_size": 2,
        "timing": "cpu_wall",
        "instrumented": False,
        "start_step": 0,
        "resume": None,
        "config": {"seed": seed, "steps": 3, "warmup": 1, "planner": planner},
        "hardware": [{"rank": r, "hostname": "fixture", "torch": "2.8"} for r in range(2)],
    }
    atomic_json(path / "manifest.json", manifest)
    for rank in range(2):
        rows = [
            {
                "rank": rank,
                "step": s,
                "warmup": s < 1,
                "local_tokens": 4,
                "global_tokens": 8,
                "step_ms": duration,
                "loss_sum": 8,
                "predicted_cost": 4,
                "padded_tokens": 4,
                "peak_memory_bytes": 0,
            }
            for s in range(3)
        ]
        (path / f"metrics-rank{rank}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def test_summary_and_seed_bootstrap(tmp_path):
    base, candidate = [], []
    for seed in range(3):
        a, b = tmp_path / f"a{seed}", tmp_path / f"b{seed}"
        fixture_run(a, seed, "random", 10)
        fixture_run(b, seed, "balanced", 5)
        base.append(str(a))
        candidate.append(str(b))
    assert summarize_run(base[0])["tokens_per_second"] == 800
    report = compare_runs(base, candidate)
    assert report["geometric_mean_speedup"] == 2
    assert report["seed_bootstrap_95_ci"] == [2, 2]


def test_comparison_rejects_synthetic_or_uncontrolled(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fixture_run(a, 1, "random")
    manifest = fixture_run(b, 1, "balanced")
    modified = deepcopy(manifest)
    modified["evidence"] = "synthetic"
    atomic_json(b / "manifest.json", modified)
    with pytest.raises(ValueError, match="observed"):
        compare_runs([str(a)], [str(b)])
    modified = deepcopy(manifest)
    modified["config"]["seed"] = 2
    atomic_json(b / "manifest.json", modified)
    with pytest.raises(ValueError, match="uncontrolled"):
        compare_runs([str(a)], [str(b)])


def test_missing_rank_and_duplicate_rows_rejected(tmp_path):
    fixture_run(tmp_path, 0, "balanced")
    path = tmp_path / "metrics-rank1.jsonl"
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        summarize_run(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="rank files"):
        summarize_run(tmp_path)


def test_one_pair_has_no_confidence_interval_and_html_is_escaped(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fixture_run(a, 1, "random")
    fixture_run(b, 1, "balanced", 8)
    result = compare_runs([str(a)], [str(b)])
    assert result["seed_bootstrap_95_ci"] is None
    result["limitations"] = ["<script>alert(1)</script>"]
    path = tmp_path / "report.html"
    write_html(result, path)
    assert "<script>" not in path.read_text()


def test_fleet_content_cannot_drift_across_seed_pairs(tmp_path):
    bases, candidates = [], []
    for seed in (1, 2):
        a, b = tmp_path / f"a{seed}", tmp_path / f"b{seed}"
        baseline = fixture_run(a, seed, "random")
        candidate = fixture_run(b, seed, "fleet")
        baseline["config"]["fleet_path"] = None
        candidate["config"]["fleet_path"] = "same/path.json"
        candidate["fleet"] = {"sha256": str(seed), "evidence": "declared"}
        atomic_json(a / "manifest.json", baseline)
        atomic_json(b / "manifest.json", candidate)
        bases.append(str(a))
        candidates.append(str(b))
    with pytest.raises(ValueError, match="settings differ"):
        compare_runs(bases, candidates)


def test_paired_dataset_content_must_match(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for path, planner, sha in ((a, "random", "one"), (b, "balanced", "two")):
        data = fixture_run(path, 1, planner)
        data["config"]["dataset_path"] = "same/path"
        data["dataset"] = {"sha256": sha, "codec": "utf8_bytes"}
        atomic_json(path / "manifest.json", data)
    with pytest.raises(ValueError, match="different dataset contents"):
        compare_runs([str(a)], [str(b)])
