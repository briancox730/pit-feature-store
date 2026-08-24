"""Batch-builder unit tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pit_feature_store.features.builder import BatchBuilder
from pit_feature_store.features.registry import FeatureRegistry, feature
from pit_feature_store.storage.query import read_data_type
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def test_builder_writes_one_row_per_feature_per_event(tmp_path: Path):
    reg = FeatureRegistry()

    @feature(name="constant_one", entity_type="series", registry=reg)
    def one(ctx, entity_id):
        return 1.0

    @feature(name="constant_two", entity_type="series", registry=reg)
    def two(ctx, entity_id):
        return 2.0

    feature_root = tmp_path / "features"
    builder = BatchBuilder(
        features=reg.list(),
        entities=[("series", "series_a")],
        start_ts=_NOW,
        end_ts=_NOW + timedelta(minutes=4),
        interval=timedelta(minutes=1),
        writer=ParquetWriter(SCHEMAS, root=feature_root),
        raw_root=tmp_path,
    )
    result = builder.run()
    # 5 ticks on the grid x 2 features = 10 rows.
    assert result["rows_written"] == 10
    tbl = read_data_type(
        "feature", source="series", entity="series_a", root=feature_root,
        start_ts=_NOW, end_ts=_NOW + timedelta(hours=1),
    )
    assert tbl.num_rows == 10
    assert set(tbl.column("feature_name").to_pylist()) == {"constant_one", "constant_two"}
    # Every written row is point-in-time: as_of_ts == event_ts in batch mode.
    assert tbl.column("as_of_ts").to_pylist() == tbl.column("event_ts").to_pylist()


def test_builder_skips_features_returning_none(tmp_path: Path):
    reg = FeatureRegistry()

    @feature(name="sometimes_none", entity_type="series", registry=reg)
    def f(ctx, entity_id):
        return 42.0 if ctx.as_of_ts.minute % 2 == 0 else None

    feature_root = tmp_path / "features"
    BatchBuilder(
        features=reg.list(),
        entities=[("series", "series_a")],
        start_ts=_NOW, end_ts=_NOW + timedelta(minutes=3),
        interval=timedelta(minutes=1),
        writer=ParquetWriter(SCHEMAS, root=feature_root),
        raw_root=tmp_path,
    ).run()
    tbl = read_data_type(
        "feature", source="series", entity="series_a", root=feature_root,
        start_ts=_NOW, end_ts=_NOW + timedelta(hours=1),
    )
    # minutes 0, 2 are even -> 2 rows.
    assert tbl.num_rows == 2


def test_builder_isolates_entity_types(tmp_path: Path):
    reg = FeatureRegistry()

    @feature(name="series_only", entity_type="series", registry=reg)
    def sr(ctx, entity_id):
        return 1.0

    @feature(name="device_only", entity_type="device", registry=reg)
    def dv(ctx, entity_id):
        return 2.0

    feature_root = tmp_path / "features"
    BatchBuilder(
        features=reg.list(),
        entities=[("series", "series_a"), ("device", "device_1")],
        start_ts=_NOW, end_ts=_NOW + timedelta(minutes=2),
        interval=timedelta(minutes=1),
        writer=ParquetWriter(SCHEMAS, root=feature_root),
        raw_root=tmp_path,
    ).run()
    series_tbl = read_data_type(
        "feature", source="series", entity="series_a", root=feature_root,
        start_ts=_NOW, end_ts=_NOW + timedelta(hours=1),
    )
    device_tbl = read_data_type(
        "feature", source="device", entity="device_1", root=feature_root,
        start_ts=_NOW, end_ts=_NOW + timedelta(hours=1),
    )
    assert set(series_tbl.column("feature_name").to_pylist()) == {"series_only"}
    assert set(device_tbl.column("feature_name").to_pylist()) == {"device_only"}


def test_builder_rejects_empty_inputs(tmp_path: Path):
    writer = ParquetWriter(SCHEMAS, root=tmp_path / "features")
    with pytest.raises(ValueError):
        BatchBuilder(
            features=[], entities=[("series", "series_a")],
            start_ts=_NOW, end_ts=_NOW + timedelta(minutes=1),
            interval=timedelta(minutes=1), writer=writer,
        )
    reg = FeatureRegistry()

    @feature(name="x", entity_type="series", registry=reg)
    def x(ctx, entity_id):
        return 1.0

    with pytest.raises(ValueError):
        BatchBuilder(
            features=reg.list(), entities=[],
            start_ts=_NOW, end_ts=_NOW + timedelta(minutes=1),
            interval=timedelta(minutes=1), writer=writer,
        )
    with pytest.raises(ValueError):
        BatchBuilder(
            features=reg.list(), entities=[("series", "series_a")],
            start_ts=_NOW + timedelta(minutes=1), end_ts=_NOW,
            interval=timedelta(minutes=1), writer=writer,
        )
