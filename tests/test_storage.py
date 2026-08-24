"""Storage layer: partition grammar, writer/reader round-trip, graceful empties."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pit_feature_store.storage.layout import dts_between, hour_keys, partition_dir
from pit_feature_store.storage.query import read_data_type
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

_NOW = datetime(2026, 5, 1, 12, 30, tzinfo=UTC)


def test_hour_keys_and_partition_dir():
    dt, hour = hour_keys(_NOW)
    assert (dt, hour) == ("2026-05-01", "12")
    p = partition_dir(Path("/root"), "synth", "tick", "series_a", dt, hour)
    assert p.as_posix().endswith(
        "root/source=synth/data_type=tick/entity=series_a/dt=2026-05-01/hour=12"
    )


def test_dts_between_enumerates_spanned_days():
    start = datetime(2026, 5, 1, 23, tzinfo=UTC)
    end = datetime(2026, 5, 3, 1, tzinfo=UTC)
    assert dts_between(start, end) == ["2026-05-01", "2026-05-02", "2026-05-03"]


def test_writer_reader_roundtrip(tmp_path: Path):
    w = ParquetWriter(SCHEMAS, root=tmp_path)
    rows = [
        {
            "event_ts": _NOW + timedelta(seconds=i),
            "ingest_ts": _NOW + timedelta(seconds=i),
            "seq": i, "raw": None, "value": float(i), "size": 1.0,
        }
        for i in range(5)
    ]
    w.append_many("synth", "tick", "series_a", rows)
    w.flush_all()

    tbl = read_data_type(
        "tick", source="synth", entity="series_a", root=tmp_path,
        start_ts=_NOW, end_ts=_NOW + timedelta(minutes=1),
    )
    assert tbl.num_rows == 5
    assert tbl.column("value").to_pylist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    # Rows come back ordered by event_ts.
    evs = tbl.column("event_ts").to_pylist()
    assert evs == sorted(evs)


def test_read_missing_lake_returns_empty_typed_table(tmp_path: Path):
    tbl = read_data_type("tick", source="synth", entity="nope", root=tmp_path)
    assert tbl.num_rows == 0
    assert "value" in tbl.column_names  # empty but correctly typed


def test_column_projection(tmp_path: Path):
    w = ParquetWriter(SCHEMAS, root=tmp_path)
    w.append_many("synth", "tick", "series_a", [{
        "event_ts": _NOW, "ingest_ts": _NOW, "seq": 1, "raw": None,
        "value": 42.0, "size": 2.0,
    }])
    w.flush_all()
    tbl = read_data_type(
        "tick", source="synth", entity="series_a", root=tmp_path,
        columns=["event_ts", "value"],
    )
    assert set(tbl.column_names) == {"event_ts", "value"}
    assert tbl.column("value").to_pylist() == [42.0]
