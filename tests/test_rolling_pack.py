"""Example rolling-stats feature pack: math + windowing correctness.

Each expectation is computed by an independent method (Python's ``statistics``
and ``math`` over the raw list) so the test cannot pass merely by re-running the
feature's own arithmetic.
"""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Importing the pack registers its features in the global REGISTRY.
import pit_feature_store.features.packs.rolling  # noqa: E402,F401
from pit_feature_store.features.context import AsOfContext
from pit_feature_store.features.registry import REGISTRY
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

_T0 = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)

# 10 ticks, 30s apart, spanning [_T0, _T0+270s]. as_of at _T0+300s so the
# trailing 300s window [_T0, _T0+300) captures all ten.
_VALUES = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 15.0, 17.0, 18.0]
_SIZES = [1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0]
_AS_OF = _T0 + timedelta(seconds=300)


def _seed(root: Path) -> None:
    w = ParquetWriter(SCHEMAS, root=root)
    rows = [
        {
            "event_ts": _T0 + timedelta(seconds=30 * i),
            "ingest_ts": _T0 + timedelta(seconds=30 * i),
            "seq": i, "raw": None,
            "value": _VALUES[i], "size": _SIZES[i],
        }
        for i in range(len(_VALUES))
    ]
    w.append_many("synth", "tick", "series_a", rows)
    w.flush_all()


def _ctx(root: Path) -> AsOfContext:
    return AsOfContext(as_of_ts=_AS_OF, raw_root=root)


def _f(name: str, root: Path):
    return REGISTRY.get(name).compute_fn(_ctx(root), "series_a")


def test_mean(tmp_path: Path):
    _seed(tmp_path)
    assert _f("mean_300s", tmp_path) == statistics.mean(_VALUES)


def test_std(tmp_path: Path):
    _seed(tmp_path)
    # approx: the pack and statistics.stdev use different-but-valid summation
    # orders, so they can differ in the last floating-point ULP.
    assert _f("std_300s", tmp_path) == pytest.approx(statistics.stdev(_VALUES))


def test_min_max_range(tmp_path: Path):
    _seed(tmp_path)
    assert _f("min_300s", tmp_path) == min(_VALUES)
    assert _f("max_300s", tmp_path) == max(_VALUES)
    assert _f("range_300s", tmp_path) == max(_VALUES) - min(_VALUES)


def test_last(tmp_path: Path):
    _seed(tmp_path)
    assert _f("last", tmp_path) == _VALUES[-1]


def test_zscore(tmp_path: Path):
    _seed(tmp_path)
    expected = (_VALUES[-1] - statistics.mean(_VALUES)) / statistics.stdev(_VALUES)
    assert _f("zscore_300s", tmp_path) == pytest.approx(expected)


def test_log_return(tmp_path: Path):
    _seed(tmp_path)
    assert _f("log_return_300s", tmp_path) == math.log(_VALUES[-1] / _VALUES[0])


def test_vwap(tmp_path: Path):
    _seed(tmp_path)
    expected = sum(v * s for v, s in zip(_VALUES, _SIZES, strict=True)) / sum(_SIZES)
    assert _f("vwap_300s", tmp_path) == expected


def test_count_60s_respects_window(tmp_path: Path):
    _seed(tmp_path)
    # Trailing 60s window is [_AS_OF-60, _AS_OF) = [_T0+240, _T0+300): ticks at
    # offsets 240 and 270 -> 2.
    assert _f("count_60s", tmp_path) == 2.0


def test_features_return_none_without_data(tmp_path: Path):
    # Empty lake -> every feature declines rather than raising.
    ctx = AsOfContext(as_of_ts=_AS_OF, raw_root=tmp_path)
    for name in ("mean_300s", "std_300s", "zscore_300s", "vwap_300s", "last"):
        assert REGISTRY.get(name).compute_fn(ctx, "series_a") is None


def test_unknown_entity_returns_none(tmp_path: Path):
    _seed(tmp_path)
    assert REGISTRY.get("mean_300s").compute_fn(_ctx(tmp_path), "not_wired") is None
