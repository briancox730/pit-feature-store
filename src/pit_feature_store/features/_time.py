"""Shared time-grid helper used by the batch builder and training-set export."""

from __future__ import annotations

from datetime import datetime, timedelta


def time_grid(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    """Inclusive grid: ``start, start+step, ..., last <= end``."""
    if step.total_seconds() <= 0:
        raise ValueError("step must be > 0")
    out: list[datetime] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += step
    return out
