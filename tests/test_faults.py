import pytest

pytest.importorskip("torch")

from tailtrace.config import TrainConfig
from tailtrace.corpus import build_corpus
from tailtrace.faults import audit_faults
from tailtrace.multinode import audit_nodes


def fixture(tmp_path):
    text = tmp_path / "text.txt"
    text.write_text(
        "Actual failures must preserve a committed objective and optimizer state.\n" * 4
    )
    build_corpus([text], tmp_path / "corpus", max_length=16)
    return TrainConfig(
        dataset_path=str(tmp_path / "corpus"),
        samples=12,
        min_length=2,
        max_length=16,
        width=16,
        heads=2,
        layers=1,
        steps=6,
        warmup=0,
        batch_size=2,
    )


@pytest.mark.distributed
def test_actual_partial_dcp_write_corruption_and_safe_fallback(tmp_path):
    result = audit_faults(fixture(tmp_path), tmp_path / "audit")
    assert result["comparison"]["bitwise_equal"]
    assert result["comparison"]["max_tensor_error"] == 0
    assert result["selection"]["step"] == 2
    assert result["telemetry"]["durable_through_step"] == 3
    rejected = {
        r["path"]: r["reason"]
        for r in result["selection"]["candidates"]
        if r["status"] == "rejected"
    }
    assert "storage content changed" in rejected["checkpoint-3"]
    assert "commit marker" in rejected["checkpoint-4"]
    assert result["corruption"]["before_sha256"] != result["corruption"]["after_sha256"]


@pytest.mark.distributed
def test_actual_separate_node_agents_match_single_agent_state(tmp_path):
    result = audit_nodes(fixture(tmp_path), tmp_path / "audit")
    assert result["comparison"]["bitwise_equal"]
    assert result["topology"] == [(0, 0, 0, 1), (1, 0, 1, 1)]
    assert result["logical_nodes"] == 2
    assert result["physical_hosts"] == 1
