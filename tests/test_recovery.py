import pytest

torch = pytest.importorskip("torch")

from tailtrace.config import TrainConfig
from tailtrace.corpus import build_corpus
from tailtrace.recovery import audit, compare_states


def test_state_comparison_checks_adamw_metadata_and_tensor_contents():
    a = {"weights": torch.arange(4), "step": 6, "lr": [0.001]}
    assert compare_states(a, a)["bitwise_equal"]
    with pytest.raises(AssertionError, match="state value differs"):
        compare_states(a, {**a, "step": 5})
    with pytest.raises(AssertionError, match="state tensor differs"):
        compare_states(a, {**a, "weights": torch.arange(4) + 1})


@pytest.mark.distributed
def test_actual_process_crash_and_real_text_restart(tmp_path):
    text = tmp_path / "source.txt"
    text.write_text("Recover real text with exact distributed gradients and AdamW states.\n" * 4)
    corpus = build_corpus([text], tmp_path / "corpus", 16)
    config = TrainConfig(
        dataset_path=str(tmp_path / "corpus"),
        samples=12,
        max_length=16,
        min_length=2,
        width=16,
        heads=2,
        layers=1,
        steps=6,
        warmup=0,
        batch_size=2,
    )
    result = audit(config, tmp_path / "audit", fault_step=2)
    assert result["runs"]["crashed"]["returncode"] != 0
    assert result["runs"]["resumed"]["returncode"] == 0
    assert result["comparison"]["bitwise_equal"]
    assert result["comparison"]["max_tensor_error"] == 0
    assert result["dataset_sha256"] == corpus["sha256"]
    assert not (tmp_path / "audit/crashed/manifest.json").exists()
