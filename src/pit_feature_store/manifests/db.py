"""SQLite store for training-set manifests — the reproducibility ledger.

One table, ``training_set_manifests``, records every frozen training set: what
features and entities it contains, over what time range, under what label
definition, at what code SHA, and the content hash of its manifest. Given a row
here, the exact dataset can be regenerated.

Timestamps are stored as ISO-8601 UTC strings (SQLite has no native
timestamptz; ISO keeps them human-readable and lexicographically sortable).
Schema evolution uses conditional ``PRAGMA table_info`` migrations so opening an
older database upgrades it in place.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pit_feature_store.paths import MANIFEST_DB_PATH


def get_conn(path: Path | None = None) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection. Defaults to ``MANIFEST_DB_PATH``."""
    p = path if path is not None else MANIFEST_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS training_set_manifests (
    manifest_id     TEXT PRIMARY KEY,    -- e.g. 'baseline_v1'
    experiment      TEXT NOT NULL,
    version         TEXT NOT NULL,
    created_at      TEXT NOT NULL,       -- ISO-8601 UTC, when the row was written
    features_json   TEXT NOT NULL,
    label_def_json  TEXT NOT NULL,
    entities_json   TEXT NOT NULL,
    date_range_json TEXT NOT NULL,
    code_sha        TEXT,
    manifest_sha256 TEXT NOT NULL,       -- content hash of the canonical manifest
    dataset_path    TEXT NOT NULL,
    notes           TEXT
);
"""


def init_db(path: Path | None = None) -> None:
    """Create the manifest table (and run idempotent migrations)."""
    with get_conn(path) as conn:
        conn.executescript(_SCHEMA_SQL)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Conditional schema migrations — extend here when fields are added.

    Pattern for the first additive migration::

        cols = {r[1] for r in conn.execute("PRAGMA table_info(training_set_manifests)")}
        if "new_col" not in cols:
            conn.execute("ALTER TABLE training_set_manifests ADD COLUMN new_col TEXT")
    """
    return


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def upsert_manifest(
    manifest_id: str,
    *,
    experiment: str,
    version: str,
    features_json: str,
    label_def_json: str,
    entities_json: str,
    date_range_json: str,
    manifest_sha256: str,
    dataset_path: str,
    code_sha: str = "",
    notes: str = "",
    conn: sqlite3.Connection | None = None,
    path: Path | None = None,
) -> None:
    """Insert-or-update one manifest row.

    All JSON fields are passed as already-encoded strings — the caller owns the
    schema inside each. Re-exporting the same ``manifest_id`` overwrites the row
    rather than raising, so a rebuild is idempotent.
    """
    own = conn is None
    if conn is None:
        conn = get_conn(path)
    try:
        conn.execute(
            """
            INSERT INTO training_set_manifests (
                manifest_id, experiment, version, created_at,
                features_json, label_def_json, entities_json, date_range_json,
                code_sha, manifest_sha256, dataset_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(manifest_id) DO UPDATE SET
                experiment = excluded.experiment,
                version = excluded.version,
                created_at = excluded.created_at,
                features_json = excluded.features_json,
                label_def_json = excluded.label_def_json,
                entities_json = excluded.entities_json,
                date_range_json = excluded.date_range_json,
                code_sha = excluded.code_sha,
                manifest_sha256 = excluded.manifest_sha256,
                dataset_path = excluded.dataset_path,
                notes = excluded.notes
            """,
            (
                manifest_id, experiment, version, utcnow_iso(),
                features_json, label_def_json, entities_json, date_range_json,
                code_sha, manifest_sha256, dataset_path, notes,
            ),
        )
        conn.commit()
    finally:
        if own:
            conn.close()


def get_manifest(
    manifest_id: str,
    *,
    conn: sqlite3.Connection | None = None,
    path: Path | None = None,
) -> sqlite3.Row | None:
    """Return the manifest row for ``manifest_id``, or ``None``."""
    own = conn is None
    if conn is None:
        conn = get_conn(path)
    try:
        return conn.execute(
            "SELECT * FROM training_set_manifests WHERE manifest_id = ?",
            (manifest_id,),
        ).fetchone()
    finally:
        if own:
            conn.close()
