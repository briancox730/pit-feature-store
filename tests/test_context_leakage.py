"""Leakage-refusal tests — the core point-in-time guarantee.

Proven with synthetic data: a read (and any feature built on it) can never
observe a row stamped at or after ``as_of_ts``, regardless of the ``end_ts`` the
caller asks for.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pit_feature_store.features.context import AsOfContext
from pit_feature_store.features.registry import FeatureRegistry, feature
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def _seed_ticks(root: Path, n: int = 10) -> None:
    w = ParquetWriter(SCHEMAS, root=root)
    rows = [
        {
            "event_ts": _NOW + timedelta(minutes=i),
            "ingest_ts": _NOW + timedelta(minutes=i),
            "seq": i, "raw": None,
            "value": 100.0 + i, "size": 1.0,
        }
        for i in range(n)
    ]
    w.append_many("synth", "tick", "series_a", rows)
    w.flush_all()


def test_read_clips_end_to_as_of(tmp_path: Path):
    _seed_ticks(tmp_path)
    # Pinned 5 minutes in; ask for a full hour — the read must stop at as_of_ts.
    ctx = AsOfContext(as_of_ts=_NOW + timedelta(minutes=5), raw_root=tmp_path)
    tbl = ctx.read(
        "tick", source="synth", entity="series_a",
        start_ts=_NOW, end_ts=_NOW + timedelta(hours=1),
    )
    # Window is [start, as_of_ts) exclusive: minutes 0..4 -> 5 rows, values 100..104.
    assert tbl.num_rows == 5
    assert max(tbl.column("value").to_pylist()) == 104.0
    assert all(v < 105.0 for v in tbl.column("value").to_pylist())


def test_read_never_returns_future_rows_across_cutoffs(tmp_path: Path):
    _seed_ticks(tmp_path)
    for minutes in (1, 3, 7, 9):
        as_of = _NOW + timedelta(minutes=minutes)
        ctx = AsOfContext(as_of_ts=as_of, raw_root=tmp_path)
        tbl = ctx.read(
            "tick", source="synth", entity="series_a",
            start_ts=_NOW, end_ts=_NOW + timedelta(days=1),
        )
        for ts in tbl.column("event_ts").to_pylist():
            assert ts < as_of, f"leaked a row at {ts} >= as_of {as_of}"


def test_inverted_window_returns_empty(tmp_path: Path):
    _seed_ticks(tmp_path)
    ctx = AsOfContext(as_of_ts=_NOW, raw_root=tmp_path)
    tbl = ctx.read(
        "tick", source="synth", entity="series_a",
        start_ts=_NOW + timedelta(hours=1), end_ts=_NOW + timedelta(hours=2),
    )
    assert tbl.num_rows == 0


def test_feature_cannot_see_the_future(tmp_path: Path):
    """A registered feature that reads a wide window still only sees the past."""
    _seed_ticks(tmp_path)
    reg = FeatureRegistry()

    @feature(name="max_value", entity_type="series", registry=reg)
    def max_value(ctx, entity_id):
        tbl = ctx.read(
            "tick", source="synth", entity=entity_id,
            start_ts=ctx.as_of_ts - timedelta(days=1),   # ask far back and forward
            end_ts=ctx.as_of_ts + timedelta(days=1),     # ...the clip still applies
        )
        vals = tbl.column("value").to_pylist()
        return max(vals) if vals else None

    spec = reg.get("max_value")
    # Global max value is 109 (minute 9). As of minute 5, the feature must not
    # see it — the correct point-in-time answer is 104.
    ctx = AsOfContext(as_of_ts=_NOW + timedelta(minutes=5), raw_root=tmp_path)
    assert spec.compute_fn(ctx, "series_a") == 104.0

    # As of the very start there is no past data at all.
    ctx0 = AsOfContext(as_of_ts=_NOW, raw_root=tmp_path)
    assert spec.compute_fn(ctx0, "series_a") is None
