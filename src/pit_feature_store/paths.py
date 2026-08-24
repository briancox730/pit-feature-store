"""Canonical filesystem paths for the feature store.

Every module resolves paths through this one, so the whole data footprint can
be relocated by setting ``PIT_FS_DATA_ROOT``. Tests always pass explicit roots
(``tmp_path``) and never touch these defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("PIT_FS_DATA_ROOT", _REPO_ROOT / "data"))

# Raw event streams (ticks, windowed streams, labels). ``v=1`` is the on-disk
# schema generation — bump it when a breaking layout change lands.
RAW_ROOT = DATA_ROOT / "raw" / "v=1"

# Derived feature values (one Parquet lake, same partition grammar as raw).
FEATURES_ROOT = DATA_ROOT / "features"

# Frozen training-set exports (one directory per experiment/version).
TRAINING_ROOT = DATA_ROOT / "training"

# SQLite store of training-set manifests (reproducibility ledger).
MANIFEST_DB_PATH = DATA_ROOT / "manifests.sqlite"


def ensure_data_dirs() -> None:
    """Create the standard data directories if they do not already exist."""
    for p in (RAW_ROOT, FEATURES_ROOT, TRAINING_ROOT):
        p.mkdir(parents=True, exist_ok=True)
