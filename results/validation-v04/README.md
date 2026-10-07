# Local version 0.4 validation

The [core log](core.log) records **116 passed**, two optional-runtime skips, and 17
CUDA tests deselected. It includes actual two-rank objective checks, interrupted DCP
writes, corruption fallback, separate logical-node agents, asymmetric failure guards,
and independent reloads of all published CPU reliability proofs.

```bash
ruff check src tests
ruff format --check src tests
pytest -m 'not cuda and not ray and not spark'
pytest tests/test_ray.py -m ray
```

The core run preceded installing Ray locally, hence its collection skips for Ray and
Spark. Ray is validated separately with two actual CPU workers; its final test enables
per-step durable journal flushing and a sealed distributed checkpoint. See
[validation.json](validation.json) for exact commands, scope, and environment.

A supplemental four-test artifact run also passes after adding the retained Ray
checkpoint/journal check; three of those tests overlap the core suite.

The source SHA-256 matches every new retained experiment. Later evidence, test, and
documentation commits do not change package source. Published observations originate
from a clean code commit and retain their own provenance.

All runs use existing local CPU compute. No hosted workflows or paid GPU/cloud jobs
were launched. The hosted badge refers to an earlier version. These checks provide
no NVIDIA CUDA/FSDP2 or physical multi-host validation. Spark was not rerun locally;
its earlier hosted baseline remains separate evidence.
