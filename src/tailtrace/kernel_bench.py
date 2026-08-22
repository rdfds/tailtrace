from __future__ import annotations

import statistics

import torch

from tailtrace.evidence import provenance
from tailtrace.kernels import fused_residual_rmsnorm
from tailtrace.model import residual_rmsnorm


def benchmark(rows: int, width: int, dtype: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("kernel-bench requires NVIDIA CUDA hardware and nvcc")
    if rows <= 0 or width <= 0:
        raise ValueError("rows and width must be positive")
    dtype = getattr(torch, dtype)
    torch.manual_seed(0)
    x, r = (
        torch.randn(rows, width, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)
    )
    w = torch.randn(width, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    ref = residual_rmsnorm(x, r, w)
    actual = fused_residual_rmsnorm(x, r, w)  # compile before timing
    tol = 3e-5 if dtype == torch.float32 else 0.04 if dtype == torch.bfloat16 else 0.005
    torch.testing.assert_close(actual, ref, rtol=tol, atol=tol)
    for a, b in zip(
        torch.autograd.grad(actual, (x, r, w), dy),
        torch.autograd.grad(ref, (x, r, w), dy),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)

    def measure(fn, backward):
        def invoke():
            y = fn(x, r, w)
            if backward:
                torch.autograd.grad(y, (x, r, w), dy)

        for _ in range(20):
            invoke()
        torch.cuda.synchronize()
        times = []
        for _ in range(30):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):
                invoke()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) / 10)
        return {"median_ms": statistics.median(times), "samples_ms": times}

    results = {}
    for name, fn in (("reference", residual_rmsnorm), ("cuda", fused_residual_rmsnorm)):
        for backward in (False, True):
            results[name + ("_forward_backward" if backward else "_forward")] = measure(
                fn, backward
            )
    return {
        "schema_version": 1,
        "evidence": "observed",
        "provenance": provenance(),
        "rows": rows,
        "width": width,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "correctness": "forward and first-order gradients passed",
        "results": results,
        "limitations": [
            "Includes Python launch gaps; eager reference, not torch.compile or vendor-fused norm.",
            "Measures one shape and dtype; this does not establish training speedup.",
        ],
    }
