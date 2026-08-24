"""Example feature pack: rolling statistics over a generic tick stream.

Every feature reads the ``tick`` data_type (a stream of ``(event_ts, value,
size)`` observations) through the AsOfContext, over a trailing window ending at
``as_of_ts``. Because the read is clipped to ``as_of_ts``, each feature is
point-in-time correct by construction — swap the synthetic ``series_*`` entities
for your own and nothing here changes.

Features (entity_type ``series``), for a trailing window of length ``W``:

  - ``mean_<W>``       arithmetic mean of ``value``
  - ``std_<W>``        sample standard deviation of ``value``
  - ``min_<W>`` / ``max_<W>`` / ``range_<W>``
  - ``count_<W>``      number of ticks observed in the window
  - ``last``           most recent ``value`` strictly before ``as_of_ts``
  - ``zscore_<W>``     ``(last - mean) / std`` over the window
  - ``log_return_<W>`` ``log(last / first)`` across the window
  - ``vwap_<W>``       ``sum(value * size) / sum(size)`` (size-weighted mean)

The naming convention ``_<W>`` uses whole seconds (e.g. ``mean_60s``).
"""

from __future__ import annotations

import math
from datetime import timedelta

from pit_feature_store.features.registry import feature
from pit_feature_store.features.sources import resolve

ENTITY_TYPE = "series"

# Two trailing windows for the bundled examples.
W_SHORT = timedelta(seconds=60)
W_LONG = timedelta(seconds=300)


def _read_ticks(ctx, entity_id, lookback: timedelta):
    src = resolve(entity_id, "tick")
    if src is None:
        return None
    source, entity = src
    return ctx.read(
        "tick", source=source, entity=entity,
        start_ts=ctx.as_of_ts - lookback,
    )


def _values(ctx, entity_id, lookback: timedelta) -> list[float] | None:
    tbl = _read_ticks(ctx, entity_id, lookback)
    if tbl is None or tbl.num_rows == 0:
        return None
    vals = [v for v in tbl.column("value").to_pylist() if v is not None]
    return vals or None


def _stdev(xs: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))


# --------------------------------------------------------------------------
# Central tendency / dispersion
# --------------------------------------------------------------------------


@feature(name="mean_60s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Mean of value over the trailing 60s")
def _mean_60s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_SHORT)
    return sum(xs) / len(xs) if xs else None


@feature(name="mean_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Mean of value over the trailing 300s")
def _mean_300s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    return sum(xs) / len(xs) if xs else None


@feature(name="std_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Sample standard deviation of value over the trailing 300s")
def _std_300s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    return _stdev(xs) if xs else None


@feature(name="min_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Minimum value over the trailing 300s")
def _min_300s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    return min(xs) if xs else None


@feature(name="max_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Maximum value over the trailing 300s")
def _max_300s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    return max(xs) if xs else None


@feature(name="range_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="max - min of value over the trailing 300s")
def _range_300s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    return (max(xs) - min(xs)) if xs else None


@feature(name="count_60s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Number of ticks observed in the trailing 60s")
def _count_60s(ctx, entity_id):
    tbl = _read_ticks(ctx, entity_id, W_SHORT)
    if tbl is None or tbl.num_rows == 0:
        return None
    return float(tbl.num_rows)


# --------------------------------------------------------------------------
# Level / momentum
# --------------------------------------------------------------------------


@feature(name="last", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Most recent value strictly before as_of_ts (trailing 300s window)")
def _last(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    return xs[-1] if xs else None


@feature(name="zscore_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="(last - mean) / stdev of value over the trailing 300s")
def _zscore_300s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    if not xs:
        return None
    sd = _stdev(xs)
    if sd is None or sd == 0:
        return None
    mean = sum(xs) / len(xs)
    return (xs[-1] - mean) / sd


@feature(name="log_return_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="log(last / first) of value across the trailing 300s")
def _log_return_300s(ctx, entity_id):
    xs = _values(ctx, entity_id, W_LONG)
    if not xs or len(xs) < 2:
        return None
    first, last = xs[0], xs[-1]
    if first <= 0 or last <= 0:
        return None
    return math.log(last / first)


@feature(name="vwap_300s", entity_type=ENTITY_TYPE, inputs=("tick",),
         description="Size-weighted mean of value over the trailing 300s")
def _vwap_300s(ctx, entity_id):
    tbl = _read_ticks(ctx, entity_id, W_LONG)
    if tbl is None or tbl.num_rows == 0:
        return None
    vals = tbl.column("value").to_pylist()
    sizes = tbl.column("size").to_pylist()
    num = 0.0
    den = 0.0
    for v, s in zip(vals, sizes, strict=True):
        if v is None or s is None:
            continue
        num += v * s
        den += s
    if den == 0:
        return None
    return num / den
