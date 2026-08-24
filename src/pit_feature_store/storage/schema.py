"""PyArrow schemas for every data_type stored in the lake.

Every record carries a universal envelope (``event_ts``, ``ingest_ts``,
``seq``, ``raw``) plus type-specific fields:

  - ``event_ts``  when the event *happened* (the point-in-time axis).
  - ``ingest_ts`` when we recorded it (always >= ``event_ts`` in a live feed).
  - ``seq``       optional monotonic sequence number from the source.
  - ``raw``       optional original payload (JSON string) kept as source of truth.

Partition columns (``source``, ``data_type``, ``entity``) are NOT in any
schema: they live in the directory path and are recovered by DuckDB at read
time. See :mod:`pit_feature_store.storage.writer`.

The types are intentionally generic — a ``tick`` is any scalar observation, a
``stream`` is any windowed event, a ``label`` is any per-window outcome. Nothing
here is tied to a particular domain.
"""

from __future__ import annotations

import pyarrow as pa

TS_TYPE = pa.timestamp("us", tz="UTC")


ENVELOPE_FIELDS: list[pa.Field] = [
    pa.field("event_ts", TS_TYPE, nullable=False),
    pa.field("ingest_ts", TS_TYPE, nullable=False),
    pa.field("seq", pa.int64(), nullable=True),
    pa.field("raw", pa.string(), nullable=True),
]


def _schema(*extra_fields: pa.Field) -> pa.Schema:
    return pa.schema(ENVELOPE_FIELDS + list(extra_fields))


# A generic scalar observation stream: one ``value`` per event, with an optional
# ``size`` weight (volume, count, mass — whatever the domain measures). The
# rolling-stats example feature pack is built on this.
TICK = _schema(
    pa.field("value", pa.float64(), nullable=False),
    pa.field("size", pa.float64(), nullable=True),
)

# A windowed event stream: every event belongs to a fixed-cadence window
# (``window_start``). Used by the liveness auditor to derive per-window
# coverage. ``value`` lets the auditor tell "the stream was alive and moving"
# from "the stream was emitting a frozen value".
STREAM = _schema(
    pa.field("window_start", TS_TYPE, nullable=False),
    pa.field("value", pa.float64(), nullable=True),
)

# Per-window outcome/label. ``label`` is NULL until the window is determined;
# the auditor treats ``label IS NOT NULL`` as "this window carries a label".
LABEL = _schema(
    pa.field("window_start", TS_TYPE, nullable=False),
    pa.field("label", pa.float64(), nullable=True),
    pa.field("determined_ts", TS_TYPE, nullable=True),
)

# Derived feature values. Stored under FEATURES_ROOT (not RAW_ROOT), partitioned
# by entity_type (in the ``source`` slot) and entity_id (in the ``entity`` slot).
# Every row carries both ``event_ts`` (the time the feature describes) and
# ``as_of_ts`` (the latest input timestamp used to compute it). Because features
# are computed through an AsOfContext, ``as_of_ts <= event_ts`` always holds.
FEATURE = _schema(
    pa.field("as_of_ts", TS_TYPE, nullable=False),
    pa.field("feature_name", pa.string(), nullable=False),
    pa.field("value", pa.float64(), nullable=True),
)


SCHEMAS: dict[str, pa.Schema] = {
    "tick": TICK,
    "stream": STREAM,
    "label": LABEL,
    "feature": FEATURE,
}
