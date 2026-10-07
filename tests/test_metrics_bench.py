import json

import pytest

from tailtrace.metrics_bench import benchmark


def test_report_memory_experiment_compares_identical_generated_workloads(tmp_path):
    output = tmp_path / "benchmark"
    data = benchmark(output, sizes=(31,))
    assert data == json.loads((output / "benchmark.json").read_text())
    assert data["input_evidence"] == "synthetic"
    case = data["cases"][0]
    assert case["rank_rows"] == 62
    external, retained = case["observations"]
    assert external["retained_rows"] == 0 and retained["retained_rows"] == 31
    assert external["aggregates"] == retained["aggregates"]
    assert all(o["python_peak_bytes"] > 0 and o["wall_ms"] > 0 for o in case["observations"])
    with pytest.raises(FileExistsError):
        benchmark(output, sizes=(31,))
    with pytest.raises(ValueError):
        benchmark(tmp_path / "invalid", sizes=(0,))
