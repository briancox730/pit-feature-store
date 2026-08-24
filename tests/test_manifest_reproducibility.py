"""Manifest reproducibility: identical inputs hash identically; any change moves it."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pit_feature_store.features.training import (
    LabelSpec,
    TrainingSetSpec,
    export_training_set,
    manifest_hash,
    manifest_json,
)
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def _spec(**overrides) -> TrainingSetSpec:
    base = dict(
        experiment="repro",
        version="1",
        feature_names=("mean_300s", "std_300s"),
        entities=(("series", "series_a"),),
        start_ts=_NOW,
        end_ts=_NOW + timedelta(hours=1),
        interval=timedelta(minutes=1),
        label=LabelSpec(name="next_move", description="d"),
        code_sha="deadbeef",
    )
    base.update(overrides)
    return TrainingSetSpec(**base)


def test_same_spec_hashes_identically():
    a = _spec()
    b = _spec()   # a distinct object with identical field values
    assert a is not b
    assert manifest_hash(a) == manifest_hash(b)
    # And it is stable across repeated calls on the same object.
    assert manifest_hash(a) == manifest_hash(a)


def test_hash_is_deterministic_content_hash():
    spec = _spec()
    expected = hashlib.sha256(manifest_json(spec).encode("utf-8")).hexdigest()
    assert manifest_hash(spec) == expected


def test_changing_feature_list_changes_hash():
    base = manifest_hash(_spec())
    assert manifest_hash(_spec(feature_names=("mean_300s",))) != base
    # Order matters — a reordering is a different dataset layout.
    assert manifest_hash(_spec(feature_names=("std_300s", "mean_300s"))) != base


def test_changing_definition_fields_changes_hash():
    base = manifest_hash(_spec())
    assert manifest_hash(_spec(code_sha="cafef00d")) != base
    assert manifest_hash(_spec(entities=(("series", "series_b"),))) != base
    assert manifest_hash(_spec(end_ts=_NOW + timedelta(hours=2))) != base
    assert manifest_hash(_spec(interval=timedelta(minutes=5))) != base
    assert manifest_hash(_spec(label=LabelSpec(name="other", description="d"))) != base


def test_written_manifest_file_matches_hash(tmp_path: Path):
    features_root = tmp_path / "features"
    w = ParquetWriter(SCHEMAS, root=features_root)
    w.append_many("series", "feature", "series_a", [{
        "event_ts": _NOW, "ingest_ts": _NOW, "seq": None, "raw": None,
        "as_of_ts": _NOW, "feature_name": "mean_300s", "value": 1.0,
    }])
    w.flush_all()

    spec = _spec(start_ts=_NOW, end_ts=_NOW, interval=timedelta(minutes=1))
    path = export_training_set(
        spec, lambda e, t: 0.0,
        features_root=features_root, training_root=tmp_path / "training",
        manifest_path=tmp_path / "m.sqlite",
    )
    on_disk = (path.parent / "manifest.json").read_bytes()
    assert hashlib.sha256(on_disk).hexdigest() == manifest_hash(spec)


def test_hash_ignores_object_identity_only_values_matter():
    # Reconstructing the spec from its own fields must not change the hash.
    spec = _spec()
    clone = dataclasses.replace(spec)
    assert manifest_hash(clone) == manifest_hash(spec)
