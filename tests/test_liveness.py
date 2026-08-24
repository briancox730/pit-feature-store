"""Stream-liveness auditor: verdict precedence, gap enumeration, trap detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pit_feature_store.quality.liveness import (
    LivenessReport,
    WindowLiveness,
    classify,
    dead_but_labeled,
    find_missing_windows,
    missing_but_labeled,
    render,
    report,
)
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter

UTC = UTC
WS = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
WINDOW = timedelta(minutes=15)
WINDOW_S = WINDOW.total_seconds()


# --------------------------------------------------------------------------
# classify() — pure verdict precedence
# --------------------------------------------------------------------------


def _classify(**over) -> str:
    base = dict(
        n_rows=500, window_seconds=WINDOW_S, max_gap_s=3.0, dead_s=0.0,
        rows_last_slice=200, start_offset_s=2.0, end_offset_s=2.0,
    )
    base.update(over)
    return classify(**base)


def test_classify_ok_when_dense_and_live():
    assert _classify() == "ok"


def test_classify_dead_on_long_blackout():
    # A multi-minute blackout is dead even with many rows overall.
    assert _classify(max_gap_s=200.0, n_rows=80_000) == "dead"


def test_classify_dead_on_large_total_dark():
    assert _classify(dead_s=400.0, max_gap_s=20.0) == "dead"


def test_classify_degraded_when_blind_at_close():
    assert _classify(rows_last_slice=0) == "degraded"


def test_classify_degraded_on_moderate_gap():
    assert _classify(max_gap_s=45.0) == "degraded"


def test_classify_degraded_on_late_start():
    assert _classify(start_offset_s=90.0) == "degraded"


def test_classify_degraded_on_early_end():
    assert _classify(end_offset_s=120.0) == "degraded"


def test_classify_thin_on_low_rowcount():
    assert _classify(n_rows=10) == "thin"


def test_dead_takes_precedence_over_thin():
    assert _classify(n_rows=5, max_gap_s=300.0, rows_last_slice=0) == "dead"


def test_thresholds_scale_with_window():
    # A 20s blackout is fine in a 15-min window but dead in a 2.5-min window,
    # because the cutoffs are fractions of the window length.
    assert _classify(max_gap_s=20.0) == "ok"
    assert classify(
        n_rows=500, window_seconds=150.0, max_gap_s=20.0, dead_s=0.0,
        rows_last_slice=200, start_offset_s=0.0, end_offset_s=0.0,
    ) == "dead"


# --------------------------------------------------------------------------
# find_missing_windows() — pure gap enumeration
# --------------------------------------------------------------------------


def _wl(ws: datetime, verdict: str = "ok") -> WindowLiveness:
    return WindowLiveness(
        entity="series_x", window_start=ws, window_seconds=WINDOW_S, n_rows=100,
        first_ev=ws, last_ev=ws, max_gap_s=1.0, dead_s=0.0, distinct_values=10,
        rows_last_slice=10, start_offset_s=0.0, end_offset_s=0.0,
        verdict=verdict, labeled=False,
    )


def test_find_missing_windows_reports_interior_gap():
    windows = [_wl(WS), _wl(WS + WINDOW), _wl(WS + 3 * WINDOW)]
    assert find_missing_windows(windows, WINDOW) == [("series_x", WS + 2 * WINDOW)]


def test_find_missing_windows_none_when_contiguous():
    windows = [_wl(WS), _wl(WS + WINDOW), _wl(WS + 2 * WINDOW)]
    assert find_missing_windows(windows, WINDOW) == []


def test_find_missing_windows_per_entity_isolated():
    import dataclasses
    a = _wl(WS)
    b = dataclasses.replace(_wl(WS + 2 * WINDOW), entity="series_y")
    assert find_missing_windows([a, b], WINDOW) == []


def test_usable_only_for_ok():
    assert _wl(WS, "ok").usable
    assert not _wl(WS, "dead").usable
    assert not _wl(WS, "degraded").usable
    assert not _wl(WS, "thin").usable


# --------------------------------------------------------------------------
# Integration over a tiny synthetic lake
# --------------------------------------------------------------------------


def _stream_row(ws: datetime, offset_s: float, value: float) -> dict:
    ev = ws + timedelta(seconds=offset_s)
    return {
        "event_ts": ev, "ingest_ts": ev, "seq": int(offset_s * 1000), "raw": None,
        "window_start": ws, "value": value,
    }


def _label_row(ws: datetime, label: float) -> dict:
    close = ws + WINDOW
    return {
        "event_ts": close, "ingest_ts": close + timedelta(minutes=30),
        "seq": None, "raw": None,
        "window_start": ws, "label": label, "determined_ts": close,
    }


def _healthy_window_rows(ws: datetime) -> list[dict]:
    # 180 rows on a 5s grid spanning the whole window, including the final slice.
    return [_stream_row(ws, s, 0.40 + (s % 7) * 0.01) for s in range(0, 900, 5)]


def _dead_window_rows(ws: datetime) -> list[dict]:
    # A few rows up front, then one row 250s later: a >2-min blackout and nothing
    # in the final slice -> a mid-window stall.
    early = [_stream_row(ws, s, 0.40 + s * 0.01) for s in range(0, 5)]
    return early + [_stream_row(ws, 250, 0.45)]


def _build_lake(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    w = ParquetWriter(SCHEMAS, root=raw_root)
    # Observed: healthy @ WS, dead @ WS+15, healthy @ WS+45. WS+30 left empty.
    w.append_many("feed", "stream", "series_x", _healthy_window_rows(WS))
    w.append_many("feed", "stream", "series_x", _dead_window_rows(WS + WINDOW))
    w.append_many("feed", "stream", "series_x", _healthy_window_rows(WS + 3 * WINDOW))
    # Labels for all four windows (including the missing WS+30).
    for k in range(4):
        w.append_many("feed", "label", "series_x", [_label_row(WS + k * WINDOW, 1.0)])
    w.flush_all()
    return raw_root


def test_report_classifies_and_flags_traps(tmp_path: Path):
    raw_root = _build_lake(tmp_path)
    rep = report(raw_root=raw_root, window=WINDOW, lookback=None, entities=("series_x",))

    verdicts = {w.window_start: w.verdict for w in rep.windows}
    assert verdicts[WS] == "ok"                         # healthy
    assert verdicts[WS + WINDOW] == "dead"              # mid-window stall
    assert verdicts[WS + 3 * WINDOW] == "ok"           # healthy

    # The dead window carries a label -> the silent training trap.
    assert [w.window_start for w in dead_but_labeled(rep)] == [WS + WINDOW]

    # WS+30 has a label but zero events -> missing-but-labeled.
    assert [(m.entity, m.window_start) for m in rep.missing] == [("series_x", WS + 2 * WINDOW)]
    assert [m.window_start for m in missing_but_labeled(rep)] == [WS + 2 * WINDOW]

    # All three observed windows joined a label.
    assert all(w.labeled for w in rep.windows)


def test_report_lookback_spans_two_dt_partitions(tmp_path: Path):
    # A lookback crossing UTC midnight spans two dt= partitions. The dt
    # restriction is a ``dt IN (...)`` predicate (not a brace glob, which DuckDB
    # silently matches to zero files), so both partitions' windows must surface.
    raw_root = tmp_path / "raw"
    w = ParquetWriter(SCHEMAS, root=raw_root)
    now = datetime.now(UTC)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ws_prev = midnight - WINDOW    # yesterday 23:45 -> dt=yesterday
    ws_today = midnight            # today 00:00     -> dt=today
    w.append_many("feed", "stream", "series_x", _healthy_window_rows(ws_prev))
    w.append_many("feed", "stream", "series_x", _healthy_window_rows(ws_today))
    for ws in (ws_prev, ws_today):
        w.append_many("feed", "label", "series_x", [_label_row(ws, 1.0)])
    w.flush_all()

    lookback = now - ws_prev + timedelta(hours=1)
    rep = report(raw_root=raw_root, window=WINDOW, lookback=lookback, entities=("series_x",))

    found = {wl.window_start for wl in rep.windows}
    assert ws_prev in found, "yesterday's partition was not scanned"
    assert ws_today in found, "today's partition was not scanned"
    assert all(wl.verdict == "ok" for wl in rep.windows)


def test_report_empty_lake_is_graceful(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    rep = report(raw_root=raw_root, window=WINDOW, lookback=None)
    assert rep.windows == []
    assert rep.missing == []
    assert dead_but_labeled(rep) == []


# --------------------------------------------------------------------------
# render()
# --------------------------------------------------------------------------


def _report_with(windows, missing) -> LivenessReport:
    return LivenessReport(
        taken_at=datetime(2026, 6, 11, 14, 0, tzinfo=UTC),
        window=WINDOW, lookback=timedelta(hours=48), windows=windows, missing=missing,
    )


def test_render_flags_dead_but_labeled():
    import dataclasses
    dead = dataclasses.replace(_wl(WS), verdict="dead", labeled=True)
    out = render(_report_with([dead], []))
    assert "DEAD-BUT-LABELED" in out
    assert "FLAGGED WINDOWS" in out


def test_render_clean_when_all_ok():
    out = render(_report_with([_wl(WS)], []))
    assert "No dead-but-labeled or missing-but-labeled windows." in out
    assert "DEAD-BUT-LABELED" not in out
