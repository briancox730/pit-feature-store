"""Training-set export: round-trip fidelity, NULL handling, idempotent manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from pit_feature_store.features.training import (
    LabelSpec,
    TrainingSetSpec,
    export_training_set,
    load_training_set,
    manifest_hash,
)
from pit_feature_store.manifests import db as manifest_db
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def _seed_features(root: Path, entries: list[tuple[str, datetime, str, float]]) -> None:
    """entries = [(entity_id, event_ts, feature_name, value), ...]"""
    w = ParquetWriter(SCHEMAS, root=root)
    grouped: dict[str, list[dict]] = {}
    for entity_id, ts, fname, value in entries:
        grouped.setdefault(entity_id, []).append({
            "event_ts": ts, "ingest_ts": ts, "seq": None, "raw": None,
            "as_of_ts": ts, "feature_name": fname, "value": value,
        })
    for entity_id, rows in grouped.items():
        w.append_many("series", "feature", entity_id, rows)
    w.flush_all()


def _spec(start: datetime, end: datetime, interval: timedelta) -> TrainingSetSpec:
    return TrainingSetSpec(
        experiment="unit_test",
        version="1",
        feature_names=("mean_300s", "std_300s"),
        entities=(("series", "series_a"),),
        start_ts=start, end_ts=end, interval=interval,
        label=LabelSpec(name="next_move", description="forward-looking label"),
        code_sha="abc1234",
    )


def test_export_roundtrip_reloads_identically(tmp_path: Path):
    features_root = tmp_path / "features"
    training_root = tmp_path / "training"
    manifest_path = tmp_path / "manifests.sqlite"
    _seed_features(features_root, [
        ("series_a", _NOW, "mean_300s", 0.001),
        ("series_a", _NOW, "std_300s", 0.0008),
        ("series_a", _NOW + timedelta(minutes=1), "mean_300s", -0.002),
        ("series_a", _NOW + timedelta(minutes=1), "std_300s", 0.0009),
    ])
    spec = _spec(_NOW, _NOW + timedelta(minutes=1), timedelta(minutes=1))

    def label_fn(entity_id, event_ts):
        return 1.0 if event_ts == _NOW else 0.5

    path = export_training_set(
        spec, label_fn,
        features_root=features_root, training_root=training_root,
        manifest_path=manifest_path,
    )
    assert path.exists()
    assert path.name == "dataset.parquet"
    assert path.parent == training_root / "exp=unit_test" / "v=1"

    table = pq.read_table(path)
    assert table.num_rows == 2
    cols = table.column_names
    assert {"entity_type", "entity_id", "event_ts", "mean_300s", "std_300s", "label"} <= set(cols)
    assert table.column("mean_300s").to_pylist() == [0.001, -0.002]
    assert table.column("label").to_pylist() == [1.0, 0.5]

    # The frozen set re-loads identically through the public loader.
    df = load_training_set(path)
    assert list(df["mean_300s"]) == [0.001, -0.002]
    assert list(df["label"]) == [1.0, 0.5]
    # Reloading twice yields byte-for-byte equal frames.
    df2 = load_training_set(path)
    assert df.equals(df2)

    # Manifest JSON on disk.
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert manifest["manifest_id"] == "unit_test_v1"
    assert manifest["features"] == ["mean_300s", "std_300s"]
    assert manifest["label"]["name"] == "next_move"

    # Manifest ledger row, incl. the content hash.
    row = manifest_db.get_manifest("unit_test_v1", path=manifest_path)
    assert row is not None
    assert row["experiment"] == "unit_test"
    assert row["code_sha"] == "abc1234"
    assert row["manifest_sha256"] == manifest_hash(spec)


def test_missing_feature_cell_is_null_by_default(tmp_path: Path):
    features_root = tmp_path / "features"
    _seed_features(features_root, [("series_a", _NOW, "mean_300s", 0.001)])
    spec = _spec(_NOW, _NOW, timedelta(minutes=1))
    path = export_training_set(
        spec, lambda e, t: 0.0,
        features_root=features_root, training_root=tmp_path / "training",
        manifest_path=tmp_path / "m.sqlite",
    )
    table = pq.read_table(path)
    assert table.num_rows == 1
    assert table.column("mean_300s").to_pylist() == [0.001]
    assert table.column("std_300s").to_pylist() == [None]


def test_require_all_features_drops_partial_rows(tmp_path: Path):
    features_root = tmp_path / "features"
    _seed_features(features_root, [
        ("series_a", _NOW, "mean_300s", 0.001),   # std_300s missing -> drop
        ("series_a", _NOW + timedelta(minutes=1), "mean_300s", 0.002),
        ("series_a", _NOW + timedelta(minutes=1), "std_300s", 0.0009),
    ])
    spec = _spec(_NOW, _NOW + timedelta(minutes=1), timedelta(minutes=1))
    path = export_training_set(
        spec, lambda e, t: 0.0,
        features_root=features_root, training_root=tmp_path / "training",
        manifest_path=tmp_path / "m.sqlite", require_all_features=True,
    )
    table = pq.read_table(path)
    assert table.num_rows == 1
    assert table.column("event_ts").to_pylist()[0] == _NOW + timedelta(minutes=1)


def test_label_fn_returning_none(tmp_path: Path):
    features_root = tmp_path / "features"
    _seed_features(features_root, [("series_a", _NOW, "mean_300s", 0.001)])
    spec = _spec(_NOW, _NOW, timedelta(minutes=1))
    path = export_training_set(
        spec, lambda e, t: None,
        features_root=features_root, training_root=tmp_path / "training",
        manifest_path=tmp_path / "m.sqlite",
    )
    assert pq.read_table(path).column("label").to_pylist() == [None]


def test_reexport_is_idempotent(tmp_path: Path):
    features_root = tmp_path / "features"
    manifest_path = tmp_path / "m.sqlite"
    _seed_features(features_root, [("series_a", _NOW, "mean_300s", 0.001)])
    spec = _spec(_NOW, _NOW, timedelta(minutes=1))
    for _ in range(2):
        export_training_set(
            spec, lambda e, t: 0.0,
            features_root=features_root, training_root=tmp_path / "training",
            manifest_path=manifest_path,
        )
    conn = manifest_db.get_conn(manifest_path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM training_set_manifests WHERE manifest_id = ?",
            ("unit_test_v1",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 1
