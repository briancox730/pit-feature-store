"""Per-window liveness / coverage audit for a windowed event stream.

The model of the data: events arrive on a stream, each tagged with the
fixed-cadence ``window_start`` it belongs to, and each window is *supposed* to
be covered continuously until it closes. Separately, a label stream assigns an
outcome to a window after it closes. A training set is built by joining the
per-window features to that label.

The failure this module exists to catch is the one an in-stream sequence-gap log
cannot see, because *nothing streamed*:

  1. **process-down gaps** — the collector was not running, so whole windows
     have ZERO events yet still receive a label after the fact.
  2. **mid-window stalls** — the stream froze for minutes while the window
     stayed open, leaving a window that is mostly dead but still gets a full
     label. A model ingests it as a valid sample whose inputs were effectively
     absent. This is the *silent training trap*: it does not error, it does not
     look empty, it just quietly teaches the model on noise.

Liveness is derived **from the stream itself** — never from a gap log — and
joined to the label stream so ``dead-but-labeled`` and ``missing-but-labeled``
windows can be fenced before any training-set build. The module is pure compute
plus dataclasses plus a :func:`render` for a CLI/monitor, with programmatic
hooks (:func:`dead_but_labeled`, :func:`missing_but_labeled`) a job can exit
non-zero on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb

from pit_feature_store.paths import RAW_ROOT
from pit_feature_store.storage.layout import hive_glob

# Default window cadence and "the stream went quiet" gap. Both are overridable
# per call — a 15-minute window with sub-second cadence and a 1-hour window with
# per-minute cadence are both just parameter choices.
DEFAULT_WINDOW = timedelta(minutes=15)
DEFAULT_QUIET_GAP_S = 5.0

# Verdict severity order (worst first) — used for sorting and summaries.
VERDICTS = ("dead", "degraded", "thin", "ok")


@dataclass(frozen=True)
class Thresholds:
    """Verdict cutoffs, as fractions of the window (except row counts).

    Defaults are deliberately generic; tune them to a stream's real cadence. The
    fraction-based cutoffs make the thresholds independent of window length: a
    2-minute blackout in a 15-minute window is the same *fraction* dark as a
    20-second blackout in a 2.5-minute window.
    """

    dead_gap_frac: float = 0.13         # a single blackout >= 13% of the window -> dead
    dead_dark_frac: float = 0.33        # >= a third of the window dark in total -> dead
    degraded_gap_frac: float = 0.033    # a blackout >= ~3% of the window -> degraded
    degraded_dark_frac: float = 0.066   # >= ~7% of the window dark total -> degraded
    edge_offset_frac: float = 0.066     # capture started/ended >~7% of the window off -> degraded
    last_frac: float = 0.066            # the decision-critical final slice of the window
    thin_rows: int = 50                 # fewer rows than this (and otherwise clean) -> thin


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class WindowLiveness:
    entity: str
    window_start: datetime
    window_seconds: float
    n_rows: int
    first_ev: datetime
    last_ev: datetime
    max_gap_s: float
    dead_s: float            # summed seconds in gaps longer than the quiet threshold
    distinct_values: int     # did the stream actually move, or emit a frozen value?
    rows_last_slice: int     # rows in the final decision-critical slice before close
    start_offset_s: float    # first_ev - window_start (how late capture began)
    end_offset_s: float      # close - last_ev (how early capture ended; <0 = spilled past)
    verdict: str
    labeled: bool            # a label joined on (entity, window_start)

    @property
    def dead_fraction(self) -> float:
        return self.dead_s / self.window_seconds if self.window_seconds else 0.0

    @property
    def usable(self) -> bool:
        """True only for fully-live windows — the gate a training export uses."""
        return self.verdict == "ok"


@dataclass(frozen=True)
class MissingWindow:
    entity: str
    window_start: datetime
    labeled: bool


@dataclass(frozen=True)
class LivenessReport:
    taken_at: datetime
    window: timedelta
    lookback: timedelta | None
    windows: list[WindowLiveness]
    missing: list[MissingWindow]

    def by_verdict(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for w in self.windows:
            out[w.verdict] = out.get(w.verdict, 0) + 1
        return out


def classify(
    *,
    n_rows: int,
    window_seconds: float,
    max_gap_s: float,
    dead_s: float,
    rows_last_slice: int,
    start_offset_s: float,
    end_offset_s: float,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Map per-window coverage metrics to ``dead`` / ``degraded`` / ``thin`` / ``ok``.

    Precedence is worst-first: a long blackout is ``dead`` even if the window
    otherwise looks busy; a window blind in its final slice is at least
    ``degraded`` because the decision-critical moment was not observed.
    """
    ws = window_seconds
    if max_gap_s >= th.dead_gap_frac * ws or dead_s >= th.dead_dark_frac * ws:
        return "dead"
    if (
        rows_last_slice == 0
        or max_gap_s >= th.degraded_gap_frac * ws
        or dead_s >= th.degraded_dark_frac * ws
        or start_offset_s >= th.edge_offset_frac * ws
        or end_offset_s >= th.edge_offset_frac * ws
    ):
        return "degraded"
    if n_rows < th.thin_rows:
        return "thin"
    return "ok"


def _lookback_bounds(
    lookback: timedelta | None,
) -> tuple[datetime | None, list[str] | None]:
    """``(start, dts)`` restricting a scan to a trailing ``lookback``.

    ``dts`` is the set of ``dt=YYYY-MM-DD`` partitions the window can span, fed
    to a ``dt IN (...)`` predicate so the engine prunes the rest. ``start``
    bounds ``event_ts`` exactly within the edge partitions. Returns
    ``(None, None)`` for an unbounded scan.
    """
    if lookback is None:
        return None, None
    now = datetime.now(UTC)
    start = now - lookback
    dts = sorted({
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((now.date() - start.date()).days + 1)
    })
    return start, dts


def compute_stream_liveness(
    *,
    raw_root: Path | None = None,
    window: timedelta = DEFAULT_WINDOW,
    quiet_gap_s: float = DEFAULT_QUIET_GAP_S,
    entities: tuple[str, ...] | None = None,
    lookback: timedelta | None = None,
    th: Thresholds = DEFAULT_THRESHOLDS,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[WindowLiveness]:
    """Aggregate the ``stream`` data_type into one row per ``(entity, window_start)``.

    Returns ``[]`` if no ``stream`` files exist. ``labeled`` is left ``False``
    here; :func:`report` fills it by joining the label stream.
    """
    raw_root = raw_root if raw_root is not None else RAW_ROOT
    glob = hive_glob(raw_root, "stream")
    start, dts = _lookback_bounds(lookback)
    window_s = window.total_seconds()
    last_slice_s = int(window_s - th.last_frac * window_s)

    where: list[str] = []
    params: list = []
    if entities is not None:
        where.append("entity IN (" + ",".join(["?"] * len(entities)) + ")")
        params.extend(entities)
    if dts is not None:
        where.append("dt IN (" + ",".join(["?"] * len(dts)) + ")")
        params.extend(dts)
    if start is not None:
        where.append("epoch(event_ts) >= ?")
        params.append(start.timestamp())
    where_clause = " AND ".join(where) if where else "TRUE"

    # All time math is done in epoch seconds (``epoch(ts)`` is the absolute
    # instant, independent of any session timezone). This keeps the query free
    # of ICU/timezone-conversion machinery, so it runs on a bare DuckDB with no
    # optional extensions — the offline-CI guarantee.
    sql = f"""
    WITH s AS (
        SELECT entity, window_start, event_ts, value
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
        WHERE {where_clause}
    ),
    g AS (
        SELECT entity, window_start, value,
            epoch(event_ts) AS ev_s,
            epoch(event_ts) - lag(epoch(event_ts))
                OVER (PARTITION BY entity, window_start ORDER BY event_ts) AS gap_s
        FROM s
    )
    SELECT
        entity,
        epoch(window_start)                                          AS ws_s,
        count(*)                                                     AS n_rows,
        min(ev_s)                                                    AS first_ev_s,
        max(ev_s)                                                    AS last_ev_s,
        coalesce(max(gap_s), 0.0)                                    AS max_gap_s,
        coalesce(sum(CASE WHEN gap_s > {quiet_gap_s} THEN gap_s ELSE 0 END), 0.0) AS dead_s,
        count(DISTINCT value)                                        AS distinct_values,
        sum(CASE WHEN ev_s >= epoch(window_start) + {last_slice_s}
                 THEN 1 ELSE 0 END)                                  AS rows_last_slice,
        min(ev_s) - epoch(window_start)                             AS start_offset_s,
        (epoch(window_start) + {window_s}) - max(ev_s)             AS end_offset_s
    FROM g
    GROUP BY entity, window_start
    ORDER BY entity, window_start
    """

    own = con is None
    if con is None:
        con = duckdb.connect(":memory:")
    try:
        try:
            rows = con.execute(sql, params).fetchall()
        except duckdb.IOException:
            return []
    finally:
        if own:
            con.close()

    out: list[WindowLiveness] = []
    for (entity, ws_s, n_rows, first_ev_s, last_ev_s, max_gap_s, dead_s, distinct,
         last_slice, start_off, end_off) in rows:
        verdict = classify(
            n_rows=n_rows, window_seconds=window_s, max_gap_s=max_gap_s,
            dead_s=dead_s, rows_last_slice=last_slice,
            start_offset_s=start_off, end_offset_s=end_off, th=th,
        )
        out.append(WindowLiveness(
            entity=entity,
            window_start=datetime.fromtimestamp(ws_s, tz=UTC),
            window_seconds=window_s,
            n_rows=int(n_rows),
            first_ev=datetime.fromtimestamp(first_ev_s, tz=UTC),
            last_ev=datetime.fromtimestamp(last_ev_s, tz=UTC),
            max_gap_s=float(max_gap_s), dead_s=float(dead_s),
            distinct_values=int(distinct), rows_last_slice=int(last_slice),
            start_offset_s=float(start_off), end_offset_s=float(end_off),
            verdict=verdict, labeled=False,
        ))
    return out


def find_missing_windows(
    windows: list[WindowLiveness], window: timedelta = DEFAULT_WINDOW,
) -> list[tuple[str, datetime]]:
    """Per entity, the window slots with ZERO events between the first and last
    observed window — the process-down gaps an in-stream gap log cannot see."""
    missing: list[tuple[str, datetime]] = []
    by_entity: dict[str, list[datetime]] = {}
    for w in windows:
        by_entity.setdefault(w.entity, []).append(w.window_start)
    for entity, starts in by_entity.items():
        observed = set(starts)
        cur, last = min(starts), max(starts)
        while cur < last:
            cur = cur + window
            if cur < last and cur not in observed:
                missing.append((entity, cur))
    return missing


def _labeled_windows(
    raw_root: Path,
    lookback: timedelta | None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> set[tuple[str, datetime]]:
    """``(entity, window_start)`` pairs that carry a determined label."""
    glob = hive_glob(raw_root, "label")
    where = "label IS NOT NULL"
    params: list = []
    if lookback is not None:
        where += " AND epoch(window_start) >= ?"
        params.append((datetime.now(UTC) - lookback).timestamp())
    sql = (
        f"SELECT DISTINCT entity, epoch(window_start) AS ws_s "
        f"FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true) "
        f"WHERE {where}"
    )
    own = con is None
    if con is None:
        con = duckdb.connect(":memory:")
    try:
        try:
            return {
                (r[0], datetime.fromtimestamp(r[1], tz=UTC))
                for r in con.execute(sql, params).fetchall()
            }
        except duckdb.IOException:
            return set()
    finally:
        if own:
            con.close()


def report(
    *,
    raw_root: Path | None = None,
    window: timedelta = DEFAULT_WINDOW,
    quiet_gap_s: float = DEFAULT_QUIET_GAP_S,
    entities: tuple[str, ...] | None = None,
    lookback: timedelta | None = None,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> LivenessReport:
    """Full liveness report: per-window verdicts + missing windows, each tagged
    with whether a label exists (so the two labeled traps stand out)."""
    raw_root = raw_root if raw_root is not None else RAW_ROOT
    con = duckdb.connect(":memory:")
    try:
        windows = compute_stream_liveness(
            raw_root=raw_root, window=window, quiet_gap_s=quiet_gap_s,
            entities=entities, lookback=lookback, th=th, con=con,
        )
        labels = _labeled_windows(raw_root, lookback, con=con)
    finally:
        con.close()

    windows = [
        replace(w, labeled=(w.entity, w.window_start) in labels)
        for w in windows
    ]
    missing = [
        MissingWindow(entity=e, window_start=ws, labeled=(e, ws) in labels)
        for (e, ws) in find_missing_windows(windows, window)
    ]
    return LivenessReport(
        taken_at=datetime.now(UTC),
        window=window, lookback=lookback, windows=windows, missing=missing,
    )


def dead_but_labeled(rep: LivenessReport) -> list[WindowLiveness]:
    """The silent training trap: windows with a label but a stream that was
    effectively absent. A monitor exits non-zero when this is non-empty."""
    return [w for w in rep.windows if w.labeled and w.verdict == "dead"]


def missing_but_labeled(rep: LivenessReport) -> list[MissingWindow]:
    """Labeled windows with NO events at all (process-down gaps)."""
    return [m for m in rep.missing if m.labeled]


# --------------------------------------------------------------------------
# Plain-text render — kept here so a CLI is a thin caller.
# --------------------------------------------------------------------------


def render(rep: LivenessReport, *, max_examples: int = 12) -> str:
    lines: list[str] = []
    lines.append(f"Stream liveness @ {rep.taken_at.isoformat()}")
    lines.append(f"Window: {int(rep.window.total_seconds())}s")
    if rep.lookback is not None:
        lines.append(f"Lookback: {int(rep.lookback.total_seconds() / 3600)}h")
    else:
        lines.append("Lookback: all")
    counts = rep.by_verdict()
    dbl = dead_but_labeled(rep)
    mbl = missing_but_labeled(rep)
    lines.append("")

    if dbl:
        lines.append(
            f"** {len(dbl)} DEAD-BUT-LABELED window(s) -- a label sits on a "
            f"stream that was effectively absent. FENCE before training. **"
        )
    if mbl:
        lines.append(
            f"** {len(mbl)} MISSING-BUT-LABELED window(s) -- a label exists but "
            f"ZERO events were captured (process-down gap). **"
        )
    if not dbl and not mbl:
        lines.append("No dead-but-labeled or missing-but-labeled windows.")
    lines.append("")

    total = len(rep.windows)
    lines.append(
        f"WINDOWS  (n={total})  "
        + "  ".join(f"{v}={counts[v]}" for v in VERDICTS)
        + f"  missing={len(rep.missing)}"
    )

    entities = sorted({w.entity for w in rep.windows} | {m.entity for m in rep.missing})
    if entities:
        lines.append(
            f"  {'entity':12s}  {'ok':>4s} {'thin':>4s} {'degr':>4s} {'dead':>4s} "
            f"{'miss':>4s}  {'labeled':>7s}"
        )
        for e in entities:
            ew = [w for w in rep.windows if w.entity == e]
            em = [m for m in rep.missing if m.entity == e]
            c = {v: sum(1 for w in ew if w.verdict == v) for v in VERDICTS}
            labeled = sum(1 for w in ew if w.labeled)
            lines.append(
                f"  {e:12s}  {c['ok']:>4d} {c['thin']:>4d} {c['degraded']:>4d} "
                f"{c['dead']:>4d} {len(em):>4d}  {labeled:>7d}"
            )

    flagged = sorted(
        (w for w in rep.windows if w.verdict in ("dead", "degraded")),
        key=lambda w: (VERDICTS.index(w.verdict), w.entity, w.window_start),
    )
    if flagged:
        lines.append("")
        lines.append("FLAGGED WINDOWS (worst first)")
        lines.append(
            f"  {'entity':12s} {'window_start':25s} {'verdict':8s} "
            f"{'rows':>8s} {'max_gap':>8s} {'dead_s':>7s} {'last':>5s} {'lbl':>3s}"
        )
        for w in flagged[:max_examples]:
            lines.append(
                f"  {w.entity:12s} {w.window_start.isoformat():25s} {w.verdict:8s} "
                f"{w.n_rows:>8,d} {w.max_gap_s:>8.1f} {w.dead_s:>7.0f} "
                f"{w.rows_last_slice:>5d} {'Y' if w.labeled else '-':>3s}"
            )
        if len(flagged) > max_examples:
            lines.append(f"  ... and {len(flagged) - max_examples} more")

    if rep.missing:
        lines.append("")
        lines.append("MISSING WINDOWS (zero events)")
        for m in rep.missing[:max_examples]:
            lines.append(
                f"  {m.entity:12s} {m.window_start.isoformat():25s}"
                + ("  <== LABELED" if m.labeled else "")
            )
        if len(rep.missing) > max_examples:
            lines.append(f"  ... and {len(rep.missing) - max_examples} more")

    return "\n".join(lines)
