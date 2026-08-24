"""Batch builder: iterate a time grid x entity set, compute features, write rows.

The builder uses one DuckDB connection and one ParquetWriter for the whole run.
Features are written to a separate lake (``FEATURES_ROOT``) using the same
partition grammar as the raw lake, with:

  - ``source``    = entity_type   (e.g. ``series``)
  - ``data_type`` = ``feature``
  - ``entity``    = entity_id      (e.g. ``series_a``)

In batch mode ``as_of_ts = event_ts``: the feature may observe any input row
with ``input.event_ts < event_ts``. A live/incremental build would set
``as_of_ts < event_ts`` to model collector lag explicitly; the AsOfContext
contract is identical either way.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pit_feature_store.features._time import time_grid
from pit_feature_store.features.context import AsOfContext
from pit_feature_store.features.registry import FeatureSpec
from pit_feature_store.paths import FEATURES_ROOT, RAW_ROOT
from pit_feature_store.storage.query import connect
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

logger = logging.getLogger(__name__)

DATA_TYPE = "feature"
_BATCH_SIZE = 5000


class BatchBuilder:
    def __init__(
        self,
        features: Sequence[FeatureSpec],
        entities: Sequence[tuple[str, str]],
        *,
        start_ts: datetime,
        end_ts: datetime,
        interval: timedelta,
        writer: ParquetWriter,
        raw_root: Path | None = None,
    ) -> None:
        if not features:
            raise ValueError("no features supplied")
        if not entities:
            raise ValueError("no entities supplied")
        if end_ts < start_ts:
            raise ValueError("end_ts < start_ts")
        if interval.total_seconds() <= 0:
            raise ValueError("interval must be > 0")
        self.features = list(features)
        self.entities = list(entities)
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.interval = interval
        self.writer = writer
        self._raw_root = raw_root if raw_root is not None else RAW_ROOT

    def run(self) -> dict:
        """Compute and write all features x entities x time-grid.

        Returns ``{"rows_written": int, "per_entity": {(entity_type, entity_id): int}}``.
        """
        ingest_ts = datetime.now(UTC)
        per_entity: dict[tuple[str, str], int] = {}
        total = 0
        grid = time_grid(self.start_ts, self.end_ts, self.interval)
        con = connect()
        try:
            for entity_type, entity_id in self.entities:
                specs = [s for s in self.features if s.entity_type == entity_type]
                if not specs:
                    continue
                n_for_entity = 0
                rows_buf: list[dict] = []
                for event_ts in grid:
                    ctx = AsOfContext(as_of_ts=event_ts, raw_root=self._raw_root, con=con)
                    for spec in specs:
                        try:
                            value = spec.compute_fn(ctx, entity_id)
                        except Exception:
                            logger.exception(
                                "feature %s failed at %s/%s",
                                spec.name, entity_id, event_ts,
                            )
                            continue
                        if value is None:
                            continue
                        rows_buf.append({
                            "event_ts": event_ts,
                            "ingest_ts": ingest_ts,
                            "seq": None,
                            "raw": None,
                            "as_of_ts": event_ts,
                            "feature_name": spec.name,
                            "value": float(value),
                        })
                        if len(rows_buf) >= _BATCH_SIZE:
                            self._flush(entity_type, entity_id, rows_buf)
                            total += len(rows_buf)
                            n_for_entity += len(rows_buf)
                            rows_buf = []
                if rows_buf:
                    self._flush(entity_type, entity_id, rows_buf)
                    total += len(rows_buf)
                    n_for_entity += len(rows_buf)
                per_entity[(entity_type, entity_id)] = n_for_entity
        finally:
            con.close()
        self.writer.flush_all()
        return {"rows_written": total, "per_entity": per_entity}

    def _flush(self, entity_type: str, entity_id: str, rows: list[dict]) -> None:
        self.writer.append_many(entity_type, DATA_TYPE, entity_id, rows)


def make_features_writer() -> ParquetWriter:
    """Convenience: a ParquetWriter rooted at ``FEATURES_ROOT``."""
    return ParquetWriter(SCHEMAS, root=FEATURES_ROOT)
