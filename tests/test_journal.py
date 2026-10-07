import json

import pytest

from tailtrace.evidence import atomic_json
from tailtrace.journal import RankJournal, committed_rows, inspect_run


def row(rank, step):
    return {"rank": rank, "step": step, "step_ms": 1.0, "loss_sum": 2.0}


def test_torn_uncommitted_suffix_is_ignored_and_committed_corruption_is_rejected(tmp_path):
    atomic_json(tmp_path / "attempt.json", {"world_size": 1})
    journal = RankJournal(tmp_path, 0, 7)
    journal.write([row(0, 7), row(0, 8)])
    journal.close()
    path = tmp_path / "metrics-rank0.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"rank": 0, "step": 9')
    rows, marker = committed_rows(tmp_path, 0)
    assert [r["step"] for r in rows] == [7, 8]
    assert marker["last_step"] == 8
    assert inspect_run(tmp_path)["durable_through_step"] == 8
    data = path.read_bytes()
    path.write_bytes(b"!" + data[1:])
    assert inspect_run(tmp_path)["status"] == "invalid"
    with pytest.raises(ValueError, match="prefix changed"):
        committed_rows(tmp_path, 0)


def test_minimum_rank_frontier_and_failed_job_do_not_become_complete(tmp_path):
    atomic_json(tmp_path / "attempt.json", {"world_size": 2})
    for rank, count in ((0, 3), (1, 2)):
        journal = RankJournal(tmp_path, rank, 0)
        journal.write([row(rank, i) for i in range(count)])
        journal.close()
    atomic_json(tmp_path / "failure-rank0.json", {"exception": "RuntimeError"})
    status = inspect_run(tmp_path)
    assert status["durable_through_step"] == 1
    assert status["status"] == "partial"
    assert status["ranks"][0]["failure"]["exception"] == "RuntimeError"
    with pytest.raises(FileExistsError):
        RankJournal(tmp_path, 0, 0)


def test_noncontiguous_write_does_not_commit_or_mutate_prefix(tmp_path):
    journal = RankJournal(tmp_path, 0, 0)
    journal.write([row(0, 0)])
    before = (tmp_path / "progress-rank0.json").read_bytes()
    with pytest.raises(ValueError, match="sequence differs"):
        journal.write([row(0, 2)])
    journal.close()
    assert (tmp_path / "progress-rank0.json").read_bytes() == before
    assert json.loads(before)["rows"] == 1


def test_completed_inspection_requires_every_sealed_rank(tmp_path):
    journal = RankJournal(tmp_path, 0, 0)
    journal.write([row(0, 0)], complete=True)
    journal.close()
    marker = json.loads((tmp_path / "progress-rank0.json").read_text())
    atomic_json(
        tmp_path / "manifest.json",
        {"world_size": 2, "telemetry": {"rank_progress": [marker, marker]}},
    )
    assert inspect_run(tmp_path)["status"] == "invalid"
    marker["bytes"] = True
    atomic_json(tmp_path / "progress-rank0.json", marker)
    with pytest.raises(ValueError, match="invalid progress"):
        committed_rows(tmp_path, 0)
