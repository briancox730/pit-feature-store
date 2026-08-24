"""Single source of truth for the lake partition grammar.

Layout::

    <root>/source=<>/data_type=<>/entity=<>/dt=YYYY-MM-DD/hour=HH/part-NNNN.parquet

Three semantic partition dimensions plus a time-based leaf:

  - ``source``    where the data came from (a feed, producer, or — for the
                  feature lake — the entity *type*).
  - ``data_type`` the kind of record (``tick``, ``stream``, ``label``,
                  ``feature``).
  - ``entity``    the thing being measured (a series id, device id, market id).
  - ``dt``/``hour`` derived from ``event_ts`` for cheap time pruning.

Writer, reader, and quality auditor all import this module so the format lives
in exactly one place; changing the leaf granularity is a single edit here.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

LEAF_PARTITION_GLOB = "hour=*"
PART_PREFIX = "part-"
PART_SUFFIX = ".parquet"


def is_part_file(path: Path) -> bool:
    return path.suffix == PART_SUFFIX and path.name.startswith(PART_PREFIX)


def iter_part_files(partition: Path) -> Iterable[Path]:
    """Yield ``part-NNNN.parquet`` files in one leaf partition directory."""
    if not partition.is_dir():
        return
    for p in partition.iterdir():
        if is_part_file(p):
            yield p


def partition_dir(
    root: Path,
    source: str,
    data_type: str,
    entity: str,
    dt: str,
    hour: str,
) -> Path:
    return (
        Path(root)
        / f"source={source}"
        / f"data_type={data_type}"
        / f"entity={entity}"
        / f"dt={dt}"
        / f"hour={hour}"
    )


def hive_glob(root: Path, data_type: str = "*") -> str:
    """Return the ``read_parquet`` glob for one (or all) data_types under ``root``.

    To narrow a scan to specific ``dt=YYYY-MM-DD`` partitions, keep this full
    ``dt=*`` glob and add a ``WHERE dt IN (...)`` predicate so the query engine
    prunes the non-matching partitions. Do NOT build a ``dt={a,b}`` brace glob:
    DuckDB does not expand brace patterns, so the read silently matches zero
    files.
    """
    return (
        Path(root).as_posix()
        + f"/source=*/data_type={data_type}/entity=*/dt=*/hour=*/part-*.parquet"
    )


def hour_keys(event_ts: datetime) -> tuple[str, str]:
    """``(dt, hour)`` strings for a UTC datetime — what the partition path uses.

    A naive datetime is assumed to already be UTC; an aware one is converted.
    """
    if event_ts.tzinfo is None:
        event_ts = event_ts.replace(tzinfo=UTC)
    ts = event_ts.astimezone(UTC)
    return ts.strftime("%Y-%m-%d"), ts.strftime("%H")


def dts_between(start: datetime, end: datetime) -> list[str]:
    """``dt=YYYY-MM-DD`` partition values a ``[start, end]`` span can touch.

    Feed this to a ``dt IN (...)`` predicate so a narrow time window does not
    open every Parquet footer in the lake.
    """
    from datetime import timedelta

    n = (end.date() - start.date()).days
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n + 1)]
