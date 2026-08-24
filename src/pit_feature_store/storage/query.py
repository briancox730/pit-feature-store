"""DuckDB-backed read helper over the Parquet lake.

Reads ``<root>/source=*/data_type=<dt>/entity=*/dt=*/hour=*/part-*.parquet`` with
Hive partitioning enabled, so ``source``, ``data_type``, ``entity``, ``dt`` and
``hour`` are queryable as columns even though they are not stored in the files.

Returns ``pyarrow.Table`` instances. When no files match the requested
data_type, returns an empty table with that type's schema instead of raising —
callers can treat "no data yet" and "no matching rows" uniformly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from pit_feature_store.paths import RAW_ROOT
from pit_feature_store.storage.layout import dts_between, hive_glob
from pit_feature_store.storage.schema import SCHEMAS


def connect() -> duckdb.DuckDBPyConnection:
    """Open a fresh in-memory DuckDB connection pinned to UTC."""
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC';")
    return con


def read_data_type(
    data_type: str,
    *,
    root: Path | None = None,
    source: str | None = None,
    entity: str | None = None,
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
    columns: list[str] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pa.Table:
    """Read rows of one data_type with optional source/entity/time filters.

    ``start_ts`` is inclusive, ``end_ts`` is exclusive. ``columns`` projects a
    subset (must include ``event_ts``). Rows are returned ordered by
    ``event_ts``. An empty result returns an empty table with the registered
    schema for that data_type.
    """
    if data_type not in SCHEMAS:
        raise KeyError(f"unknown data_type={data_type!r}")
    root = root if root is not None else RAW_ROOT
    pattern = hive_glob(root, data_type)
    select = "*" if not columns else ", ".join(columns)

    own_conn = con is None
    if con is None:
        con = connect()
    try:
        where: list[str] = []
        params: list[Any] = []
        if source is not None:
            where.append("source = ?")
            params.append(source)
        if entity is not None:
            where.append("entity = ?")
            params.append(entity)
        if start_ts is not None:
            where.append("event_ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            where.append("event_ts < ?")
            params.append(end_ts)
        # When the window is bounded both sides, prune to its dt partitions so a
        # narrow read does not footer-scan the whole lake.
        if start_ts is not None and end_ts is not None:
            dts = dts_between(start_ts, end_ts)
            where.append("dt IN (" + ",".join(["?"] * len(dts)) + ")")
            params.extend(dts)
        where_clause = " AND ".join(where) if where else "TRUE"
        sql = (
            f"SELECT {select} FROM read_parquet('{pattern}', hive_partitioning=true) "
            f"WHERE {where_clause} "
            f"ORDER BY event_ts"
        )
        try:
            return con.execute(sql, params).to_arrow_table()
        except duckdb.IOException:
            return SCHEMAS[data_type].empty_table()
    finally:
        if own_conn:
            con.close()
