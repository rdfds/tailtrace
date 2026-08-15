import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.mark.distributed
def test_real_two_rank_gradient_and_optimizer_equivalence(tmp_path):
    # Avoid torchrun's platform-dependent hostname selection on macOS.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    evidence = tmp_path / "equivalence.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        str(Path(__file__).with_name("distributed_worker.py")),
        str(evidence),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(evidence.read_text())["gradient_and_adamw_equivalence"] == "passed"
