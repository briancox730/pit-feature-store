"""Buffered, partition-aware Parquet writer for the lake.

Layout: ``<root>/source=X/data_type=Y/entity=Z/dt=YYYY-MM-DD/hour=HH/part-NNNN.parquet``

A partition buffer flushes when *any* of these are true:
  - row count reaches ``max_rows``
  - estimated buffered bytes reach ``max_bytes``
  - the oldest buffered row is older than ``max_age_seconds``

Writes go to a hidden ``.part-NNNN.parquet.tmp`` then ``os.replace`` to the
final name, so a reader never sees a half-written file.

Schemas registered with the writer must NOT contain the partition columns
(``source``, ``data_type``, ``entity``); those values live in the directory
path and are recovered by DuckDB at read time. Extra keys in a row dict that
are absent from the schema are silently dropped by ``pa.Table.from_pylist``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pit_feature_store.paths import RAW_ROOT
from pit_feature_store.storage.layout import (
    PART_PREFIX,
    hour_keys,
    iter_part_files,
    partition_dir,
)

logger = logging.getLogger(__name__)

PartitionKey = tuple[str, str, str, str, str]  # (source, data_type, entity, dt, hour)


def _next_part_index(part_dir: Path) -> int:
    if not part_dir.exists():
        return 0
    max_idx = -1
    for p in iter_part_files(part_dir):
        try:
            idx = int(p.stem[len(PART_PREFIX):])
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    return max_idx + 1


@dataclass
class _Buffer:
    rows: list[dict[str, Any]] = field(default_factory=list)
    bytes_estimate: int = 0
    opened_at: float = field(default_factory=time.monotonic)


class ParquetWriter:
    """Thread-safe buffered Parquet writer.

    One instance per process is sufficient. Pass a ``schemas`` dict mapping
    ``data_type`` to its ``pyarrow.Schema``. Appending a row with an
    unregistered ``data_type`` raises ``KeyError``.
    """

    def __init__(
        self,
        schemas: dict[str, pa.Schema],
        *,
        root: Path | None = None,
        max_rows: int = 10_000,
        max_bytes: int = 16 * 1024 * 1024,
        max_age_seconds: float = 30.0,
        compression: str = "zstd",
    ) -> None:
        self._schemas = schemas
        self._root = root if root is not None else RAW_ROOT
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._max_age = max_age_seconds
        self._compression = compression
        self._buffers: dict[PartitionKey, _Buffer] = {}
        self._lock = threading.Lock()

    def append(self, source: str, data_type: str, entity: str, row: dict[str, Any]) -> None:
        self.append_many(source, data_type, entity, (row,))

    def append_many(self, source: str, data_type: str, entity: str, rows) -> None:
        """Append a batch of rows under a single lock acquire.

        Each row may carry its own ``event_ts`` and thus land in its own
        partition buffer — we group locally before touching ``self._buffers`` so
        the lock is held once per call rather than once per row.
        """
        if data_type not in self._schemas:
            raise KeyError(f"no schema registered for data_type={data_type!r}")
        grouped: dict[PartitionKey, list[tuple[dict[str, Any], int]]] = {}
        for row in rows:
            event_ts = row.get("event_ts")
            if not isinstance(event_ts, datetime):
                raise ValueError("row must include a datetime 'event_ts' for partitioning")
            dt, hour = hour_keys(event_ts)
            key: PartitionKey = (source, data_type, entity, dt, hour)
            grouped.setdefault(key, []).append((row, _estimate_row_bytes(row)))
        if not grouped:
            return
        with self._lock:
            for key, batch in grouped.items():
                buf = self._buffers.get(key)
                if buf is None:
                    buf = _Buffer()
                    self._buffers[key] = buf
                for row, est in batch:
                    buf.rows.append(row)
                    buf.bytes_estimate += est
                if (
                    len(buf.rows) >= self._max_rows
                    or buf.bytes_estimate >= self._max_bytes
                    or (time.monotonic() - buf.opened_at) >= self._max_age
                ):
                    self._flush_locked(key)

    def flush_idle(self) -> None:
        """Flush buffers older than ``max_age_seconds``. Call periodically."""
        with self._lock:
            now = time.monotonic()
            stale = [
                k for k, b in self._buffers.items()
                if b.rows and (now - b.opened_at) >= self._max_age
            ]
            for k in stale:
                self._flush_locked(k)

    def flush_all(self) -> None:
        """Force-flush every buffer. Call on shutdown / at the end of a build."""
        with self._lock:
            for k in list(self._buffers.keys()):
                self._flush_locked(k)

    def buffered_partitions(self) -> int:
        with self._lock:
            return sum(1 for b in self._buffers.values() if b.rows)

    def _flush_locked(self, key: PartitionKey) -> None:
        buf = self._buffers.pop(key, None)
        if buf is None or not buf.rows:
            return
        _, data_type, _, _, _ = key
        schema = self._schemas[data_type]
        table = pa.Table.from_pylist(buf.rows, schema=schema)
        out_dir = partition_dir(self._root, *key)
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = _next_part_index(out_dir)
        final_path = out_dir / f"part-{idx:04d}.parquet"
        tmp_path = out_dir / f".part-{idx:04d}.parquet.tmp"
        pq.write_table(table, tmp_path, compression=self._compression)
        os.replace(tmp_path, final_path)
        logger.debug("flushed %d rows -> %s", len(buf.rows), final_path)


def _estimate_row_bytes(row: dict[str, Any]) -> int:
    """Cheap byte estimate for flush thresholds. Intentionally approximate."""
    total = 0
    for v in row.values():
        if v is None or isinstance(v, bool):
            total += 1
        elif isinstance(v, (int, float, datetime)):
            total += 8
        elif isinstance(v, (str, bytes)):
            total += len(v)
        else:
            total += sys.getsizeof(v)
    return total
