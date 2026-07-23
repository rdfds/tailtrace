from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def region(name: str, nvtx: bool):
    with torch.profiler.record_function(name):
        if nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()
