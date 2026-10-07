import math
import random
import statistics
from pathlib import Path

import pytest

from tailtrace.quantiles import DiskQuantiles
from tailtrace.report import percentile


@pytest.mark.parametrize("size", [1, 2, 17, 1001])
def test_external_quantiles_match_exact_in_memory_definition(size):
    rng = random.Random(7)
    values = [rng.randrange(1, 1000) / 7 for _ in range(size)]
    with DiskQuantiles() as distribution:
        directory = Path(distribution.directory.name)
        for value in values:
            distribution.add(value)
        assert distribution.percentile(0.5) == statistics.median(values)
        for q in (0, 0.025, 0.95, 1):
            assert distribution.percentile(q) == percentile(values, q)
        with pytest.raises(ValueError, match="indexed"):
            distribution.add(1)
    assert not directory.exists()


def test_quantile_cleanup_and_invalid_values():
    with pytest.raises(RuntimeError):
        with DiskQuantiles() as distribution:
            directory = Path(distribution.directory.name)
            for value in (0, -1, math.nan, math.inf):
                with pytest.raises(ValueError):
                    distribution.add(value)
            with pytest.raises(ValueError):
                distribution.percentile(0.5)
            raise RuntimeError("injected consumer failure")
    assert not directory.exists()
