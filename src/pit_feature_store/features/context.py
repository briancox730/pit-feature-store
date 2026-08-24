"""The read-side context handed to every feature compute function.

This is where point-in-time correctness is enforced. An :class:`AsOfContext`
pins an ``as_of_ts`` and **clips every read's upper bound to it**. A feature
compute function receives the context and reads its inputs through it, so there
is no code path by which a feature can observe a row stamped at or after the
cutoff. Leakage is prevented by construction, not by discipline.

The context also caches reads per ``(data_type, source, entity)`` for its own
lifetime: features in a pack typically read the same source over overlapping
windows, so the context keeps the widest window it has fetched and slices
narrower requests in memory instead of re-querying the lake.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc

from pit_feature_store.paths import RAW_ROOT
from pit_feature_store.storage.query import read_data_type
from pit_feature_store.storage.schema import SCHEMAS


class AsOfContext:
    """A leakage-proof reader pinned at a single ``as_of_ts``.

    Every :meth:`read` returns only rows with ``start_ts <= event_ts < as_of_ts``.
    The upper bound is *always* clamped to ``as_of_ts`` regardless of the
    ``end_ts`` a feature asks for — that clamp is the entire safety property.
    """

    def __init__(
        self,
        as_of_ts: datetime,
        *,
        raw_root: Path | None = None,
        con: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.as_of_ts = as_of_ts
        self._raw_root = raw_root if raw_root is not None else RAW_ROOT
        self._con = con
        # (data_type, source, entity) -> (cached_start_ts, table covering [cached_start_ts, as_of_ts))
        self._cache: dict[tuple[str, str, str], tuple[datetime, pa.Table]] = {}

    def read(
        self,
        data_type: str,
        *,
        source: str,
        entity: str,
        start_ts: datetime,
        end_ts: datetime | None = None,
    ) -> pa.Table:
        """Read raw rows in ``[start_ts, min(end_ts, as_of_ts))``.

        ``end_ts`` defaults to ``as_of_ts`` and is never allowed to exceed it.
        """
        eff_end = self.as_of_ts if end_ts is None else min(end_ts, self.as_of_ts)
        if eff_end <= start_ts:
            return SCHEMAS[data_type].empty_table()

        key = (data_type, source, entity)
        cached = self._cache.get(key)
        if cached is not None:
            cached_start, cached_tbl = cached
            if cached_start <= start_ts:
                return _slice_between(cached_tbl, start_ts, eff_end)
            widest = min(cached_start, start_ts)
        else:
            widest = start_ts

        tbl = read_data_type(
            data_type,
            root=self._raw_root,
            source=source,
            entity=entity,
            start_ts=widest,
            end_ts=self.as_of_ts,
            con=self._con,
        )
        self._cache[key] = (widest, tbl)
        return _slice_between(tbl, start_ts, eff_end)


def _slice_between(tbl: pa.Table, start_ts: datetime, end_ts: datetime) -> pa.Table:
    if tbl.num_rows == 0:
        return tbl
    ev = tbl.column("event_ts")
    mask = pc.and_(pc.greater_equal(ev, start_ts), pc.less(ev, end_ts))
    return tbl.filter(mask)
