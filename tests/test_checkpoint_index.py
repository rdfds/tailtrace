from dataclasses import replace

import pytest

from tailtrace.checkpoint_index import seal, select, verify
from tailtrace.config import TrainConfig
from tailtrace.corpus import content_hash
from tailtrace.evidence import atomic_json


def fixture(root, step, config):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    # Deliberately generated storage bytes exercise inventory guards, not DCP loading.
    (path / ".metadata").write_bytes(b"generated metadata fixture")
    (path / "__0_0.distcp").write_bytes(b"generated shard fixture")
    marker = seal(
        path,
        {
            "step": step,
            "world_size": 2,
            "config": config.to_dict(),
            "fleet_sha256": None,
            "dataset_sha256": "dataset",
        },
    )
    atomic_json(path / "complete.json", marker)
    return path, marker


def test_selection_skips_incomplete_corrupt_incompatible_and_final_checkpoints(tmp_path):
    config = TrainConfig(steps=8)
    fixture(tmp_path, 1, config)
    fixture(tmp_path, 2, config)
    corrupt, _ = fixture(tmp_path, 3, config)
    (corrupt / "__0_0.distcp").write_bytes(b"changed")
    (tmp_path / "checkpoint-4").mkdir()
    fixture(tmp_path, 5, replace(config, seed=18))
    fixture(tmp_path, 8, config)
    result = select(tmp_path, config, 2, dataset_sha256="dataset")
    assert result["step"] == 2
    assert len([r for r in result["candidates"] if r["status"] == "rejected"]) == 4
    with pytest.raises(ValueError, match="no verified compatible"):
        select(tmp_path, config, 2, dataset_sha256="different")


def test_corrupted_metadata_same_size_and_unsafe_inventory_rejected(tmp_path):
    path, marker = fixture(tmp_path, 1, TrainConfig())
    payload = path / ".metadata"
    payload.write_bytes(b"!" + payload.read_bytes()[1:])
    with pytest.raises(ValueError, match="storage content changed"):
        verify(path)
    marker["files"][0]["path"] = "../outside"
    marker["sha256"] = content_hash({k: v for k, v in marker.items() if k != "sha256"})
    atomic_json(path / "complete.json", marker)
    with pytest.raises(ValueError, match="invalid checkpoint storage path"):
        verify(path)


def test_legacy_explicit_restore_allowed_but_automatic_selection_requires_seals(tmp_path):
    path = tmp_path / "checkpoint-1"
    atomic_json(
        path / "complete.json", {"schema_version": 1, "step": 1, "world_size": 2, "config": {}}
    )
    assert verify(path)["schema_version"] == 1
    with pytest.raises(ValueError, match="no storage content seal"):
        verify(path, require_sealed=True)


def test_model_contract_and_torch_version_reject_before_storage_loading(tmp_path):
    config = TrainConfig(steps=8)
    path, marker = fixture(tmp_path, 2, config)
    marker["model_contract_sha256"] = "different-objective"
    marker["sha256"] = content_hash({k: v for k, v in marker.items() if k != "sha256"})
    atomic_json(path / "complete.json", marker)
    with pytest.raises(ValueError, match="model/objective implementation changed"):
        select(tmp_path, config, 2, dataset_sha256="dataset")
    from tailtrace.checkpoint_index import model_contract_hash

    marker["model_contract_sha256"] = model_contract_hash()
    marker["torch_version"] = "other-runtime"
    marker["sha256"] = content_hash({k: v for k, v in marker.items() if k != "sha256"})
    atomic_json(path / "complete.json", marker)
    with pytest.raises(ValueError, match="torch version changed"):
        select(tmp_path, config, 2, dataset_sha256="dataset", torch_version="2.8.0")
