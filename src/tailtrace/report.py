from __future__ import annotations

import html
import json
import math
import random
import statistics
from pathlib import Path


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        raise ValueError("empty distribution")
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


