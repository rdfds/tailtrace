# Ray Train: bounded journals and sealed checkpoint

This retained run uses two actual Ray Train CPU workers, the shared DDP loop,
`metrics_flush_every=1`, and a distributed checkpoint after step 3. Each rank seals
three metric rows; peak buffered rows are exactly one. The automatic summary streams
both journals and retains two measured steps after one warmup step. The DCP inventory
is verified by the actual Ray integration test and by the published evidence test.

```bash
pip install -e '.[ray,dev]'
pytest tests/test_ray.py -m ray
```

[validation.log](validation.log) records the passing integration test. Raw rank metrics,
progress markers, the manifest, checkpoint shards/metadata, config, and aggregate
summary are retained in `run/`. The code-uploaded workers record the same package
source SHA-256 as the release. Their Git metadata may be unavailable inside Ray's
uploaded package directory; the source content identity remains recorded.

Scope: one physical CPU host, Gloo/DDP, synthetic token training. This demonstrates
Ray placement, process-group reuse, journal flushing, and DCP integration. It is not
a real-data quality result, remote Ray-cluster validation, or a GPU/scaling benchmark.
