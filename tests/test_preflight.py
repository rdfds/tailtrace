import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.mark.distributed
@pytest.mark.parametrize("mode", ["config", "source", "journal", "commit"])
def test_asymmetric_rank_errors_reject_on_every_rank_without_hanging(tmp_path, mode):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            str(Path(__file__).with_name("preflight_worker.py")),
            mode,
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proofs = [json.loads((tmp_path / f"proof-rank{rank}.json").read_text()) for rank in range(2)]
    assert [p["rank"] for p in proofs] == [0, 1]
    assert all(p["mode"] == mode for p in proofs)
    if mode in {"journal", "commit"}:
        assert all((tmp_path / "run" / f"failure-rank{rank}.json").exists() for rank in range(2))
    if mode == "commit":
        assert not (tmp_path / "run/checkpoint-1/complete.json").exists()
