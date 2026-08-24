"""Frozen training-set export with a reproducible, content-hashed manifest.

A training set is a pinned-in-time slice of the feature lake joined to a
caller-supplied label callable, materialized to a single Parquet under
``<training_root>/exp=<experiment>/v=<version>/dataset.parquet`` and recorded in
the manifest ledger (SQLite) for reproducibility.

The manifest captures everything needed to regenerate the dataset — the feature
list, the entities, the time range + interval, the label definition, and an
optional code SHA — and is serialized *canonically* (sorted keys, no
wall-clock fields) so its content hash is stable across runs. Same definition
-> same hash; any change to the definition changes the hash.

Label semantics are intentionally minimal: the caller passes a
``label_fn(entity_id, event_ts) -> float | None``. A label may look at any data,
including data with ``event_ts`` beyond the row's ``event_ts`` — labels are
forward-looking by definition. Leakage protection applies to FEATURES, which the
AsOfContext already enforces; it deliberately does not constrain labels.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pit_feature_store.manifests import db as manifest_db
from pit_feature_store.paths import FEATURES_ROOT, TRAINING_ROOT
from pit_feature_store.storage.query import read_data_type

logger = logging.getLogger(__name__)

LabelFn = Callable[[str, datetime], "float | None"]


@dataclass(frozen=True)
class LabelSpec:
    name: str
    description: str = ""


@dataclass(frozen=True)
class TrainingSetSpec:
    experiment: str
    version: str
    feature_names: tuple[str, ...]
    entities: tuple[tuple[str, str], ...]   # ((entity_type, entity_id), ...)
    start_ts: datetime
    end_ts: datetime
    interval: timedelta
    label: LabelSpec
    code_sha: str = ""
    notes: str = ""

    @property
    def manifest_id(self) -> str:
        return f"{self.experiment}_v{self.version}"

    @property
    def export_dir(self) -> Path:
        return TRAINING_ROOT / f"exp={self.experiment}" / f"v={self.version}"


# --------------------------------------------------------------------------
# Manifest: canonical serialization + content hash
# --------------------------------------------------------------------------


def manifest_dict(spec: TrainingSetSpec) -> dict:
    """The reproducibility manifest as a plain dict.

    Contains only definitional fields — no wall-clock timestamps — so it is a
    pure function of the spec and therefore hashable to a stable digest.
    """
    return {
        "manifest_id": spec.manifest_id,
        "experiment": spec.experiment,
        "version": spec.version,
        "features": list(spec.feature_names),
        "label": {"name": spec.label.name, "description": spec.label.description},
        "entities": [list(e) for e in spec.entities],
        "date_range": {
            "start": spec.start_ts.isoformat(),
            "end": spec.end_ts.isoformat(),
            "interval_seconds": spec.interval.total_seconds(),
        },
        "code_sha": spec.code_sha,
        "notes": spec.notes,
    }


def manifest_json(spec: TrainingSetSpec) -> str:
    """Canonical JSON for the manifest: sorted keys, stable separators."""
    return json.dumps(manifest_dict(spec), sort_keys=True, indent=2)


def manifest_hash(spec: TrainingSetSpec) -> str:
    """SHA-256 of the canonical manifest JSON.

    Deterministic across runs and machines for a given spec; changes iff the
    training-set definition changes.
    """
    return hashlib.sha256(manifest_json(spec).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def _read_feature_rows(
    spec: TrainingSetSpec, features_root: Path,
) -> dict[tuple[str, str, datetime], dict[str, float]]:
    """Aggregate the feature lake into ``{(entity_type, entity_id, event_ts) -> {name: value}}``."""
    wanted = set(spec.feature_names)
    out: dict[tuple[str, str, datetime], dict[str, float]] = {}
    # read_data_type's end_ts is exclusive; pad by 1us so the grid's right
    # endpoint (which IS included in the dataset) finds its feature rows.
    query_end = spec.end_ts + timedelta(microseconds=1)
    for entity_type, entity_id in spec.entities:
        tbl = read_data_type(
            "feature", source=entity_type, entity=entity_id,
            root=features_root,
            start_ts=spec.start_ts, end_ts=query_end,
        )
        if tbl.num_rows == 0:
            continue
        evt = tbl.column("event_ts").to_pylist()
        names = tbl.column("feature_name").to_pylist()
        vals = tbl.column("value").to_pylist()
        for ts, name, value in zip(evt, names, vals, strict=True):
            if name not in wanted or value is None:
                continue
            out.setdefault((entity_type, entity_id, ts), {})[name] = float(value)
    return out


def _grid(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    pts: list[datetime] = []
    cur = start
    while cur <= end:
        pts.append(cur)
        cur += step
    return pts


def export_training_set(
    spec: TrainingSetSpec,
    label_fn: LabelFn,
    *,
    features_root: Path | None = None,
    training_root: Path | None = None,
    manifest_path: Path | None = None,
    require_all_features: bool = False,
) -> Path:
    """Materialize a training set and persist its manifest.

    Returns the path to the written ``dataset.parquet``.

    ``require_all_features``: if True, rows missing any requested feature are
    dropped; if False (default), missing cells become NULL.
    """
    features_root = features_root if features_root is not None else FEATURES_ROOT
    training_root = training_root if training_root is not None else TRAINING_ROOT
    feature_rows = _read_feature_rows(spec, features_root)
    grid = _grid(spec.start_ts, spec.end_ts, spec.interval)

    cols_entity_type: list[str] = []
    cols_entity_id: list[str] = []
    cols_event_ts: list[datetime] = []
    cols_features: dict[str, list[float | None]] = {n: [] for n in spec.feature_names}
    cols_label: list[float | None] = []

    for entity_type, entity_id in spec.entities:
        for event_ts in grid:
            cells = feature_rows.get((entity_type, entity_id, event_ts), {})
            if require_all_features and not all(n in cells for n in spec.feature_names):
                continue
            cols_entity_type.append(entity_type)
            cols_entity_id.append(entity_id)
            cols_event_ts.append(event_ts)
            for name in spec.feature_names:
                cols_features[name].append(cells.get(name))
            try:
                label_value = label_fn(entity_id, event_ts)
            except Exception:
                logger.exception("label_fn failed for %s/%s", entity_id, event_ts)
                label_value = None
            cols_label.append(float(label_value) if label_value is not None else None)

    out_dir = training_root / f"exp={spec.experiment}" / f"v={spec.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        [
            pa.field("entity_type", pa.string()),
            pa.field("entity_id", pa.string()),
            pa.field("event_ts", pa.timestamp("us", tz="UTC")),
        ]
        + [pa.field(n, pa.float64()) for n in spec.feature_names]
        + [pa.field("label", pa.float64())]
    )
    arrays = (
        [
            pa.array(cols_entity_type, type=pa.string()),
            pa.array(cols_entity_id, type=pa.string()),
            pa.array(cols_event_ts, type=pa.timestamp("us", tz="UTC")),
        ]
        + [pa.array(cols_features[n], type=pa.float64()) for n in spec.feature_names]
        + [pa.array(cols_label, type=pa.float64())]
    )
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    dataset_path = out_dir / "dataset.parquet"
    pq.write_table(table, dataset_path, compression="zstd")

    # Write the human-readable manifest next to the data. ``newline="\n"``
    # disables platform newline translation so the file is byte-identical
    # everywhere and hashes to the same digest on Windows and POSIX alike.
    (out_dir / "manifest.json").write_text(
        manifest_json(spec), encoding="utf-8", newline="\n"
    )
    # ...and record it in the ledger.
    _persist_manifest_row(spec, dataset_path, manifest_path)

    logger.info("training: wrote %d rows to %s", table.num_rows, dataset_path)
    return dataset_path


def load_training_set(dataset_path: Path) -> pd.DataFrame:
    """Load a frozen training set back into a pandas DataFrame.

    The complement of :func:`export_training_set`: what a model-training job
    calls. Column order and dtypes round-trip exactly.
    """
    return pq.read_table(dataset_path).to_pandas()


def _persist_manifest_row(
    spec: TrainingSetSpec, dataset_path: Path, manifest_path: Path | None,
) -> None:
    manifest_db.init_db(manifest_path)
    manifest_db.upsert_manifest(
        spec.manifest_id,
        experiment=spec.experiment,
        version=spec.version,
        features_json=json.dumps(list(spec.feature_names)),
        label_def_json=json.dumps(
            {"name": spec.label.name, "description": spec.label.description}
        ),
        entities_json=json.dumps([list(e) for e in spec.entities]),
        date_range_json=json.dumps(
            {
                "start": spec.start_ts.isoformat(),
                "end": spec.end_ts.isoformat(),
                "interval_seconds": spec.interval.total_seconds(),
            }
        ),
        manifest_sha256=manifest_hash(spec),
        dataset_path=str(dataset_path),
        code_sha=spec.code_sha,
        notes=spec.notes,
        path=manifest_path,
    )
