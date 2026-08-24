# pit-feature-store

A point-in-time feature store with reproducible training-set exports and stream
data-quality auditing. Built on PyArrow, DuckDB, and pandas over
Hive-partitioned Parquet. Domain-agnostic: it models entities, event time, and
windowed streams, not any particular business.

It holds three guarantees:

1. **No leakage.** A feature cannot read the future because the read API will
   not return it.
2. **Reproducible training sets.** Every frozen dataset carries a
   content-hashed manifest. Same definition, same hash.
3. **Honest data quality.** A stream auditor catches windows that carry a label
   but had a dead or missing input stream.

Everything below is covered by the test suite (57 tests, fully offline) and CI
on Python 3.11 and 3.12.

---

## 1. No leakage

The most common way an offline result goes wrong is leakage: a feature computed
at time *t* picks up data that only existed after *t*. It inflates every metric,
you can't see it in the notebook, and you find out in production.

The usual defense is discipline, "remember to filter by timestamp". That
doesn't scale across dozens of features and years of edits. Here, features
never touch storage directly. They read through an **`AsOfContext`** that pins
an `as_of_ts` and clips every read's upper bound to it. There is no code path
where a feature can see a row stamped at or after the cutoff.

```python
from pit_feature_store.features.registry import feature
from pit_feature_store.features.sources import resolve

@feature(name="mean_60s", entity_type="series", inputs=("tick",),
         description="Mean of value over the trailing 60s")
def mean_60s(ctx, entity_id):
    source, entity = resolve(entity_id, "tick")
    tbl = ctx.read(                       # <-- the only way to read inputs
        "tick", source=source, entity=entity,
        start_ts=ctx.as_of_ts - timedelta(seconds=60),
        end_ts=ctx.as_of_ts + timedelta(days=365),   # ask for the future...
    )                                                 # ...you still get the past
    vals = tbl.column("value").to_pylist()
    return sum(vals) / len(vals) if vals else None
```

The `end_ts` above asks for a year of future data on purpose. It doesn't
matter. The context clamps the effective end to `as_of_ts` every time, and
[`tests/test_context_leakage.py`](tests/test_context_leakage.py) proves it: a
feature that requests a window spanning a full day on either side of the cutoff
still only sees rows before it.

```python
# from the tests: global max value is 109, but as of minute 5 the feature
# must return 104 — it cannot see the future rows.
ctx = AsOfContext(as_of_ts=_NOW + timedelta(minutes=5), raw_root=root)
assert max_value.compute_fn(ctx, "series_a") == 104.0
```

In a batch build, `as_of_ts` is each grid point's `event_ts`. In a live build
you set `as_of_ts < event_ts` to model collector lag explicitly. The contract
is identical in both, so a feature written once is correct in both.

---

## 2. Reproducible training sets

If you can't regenerate a training set, you can't trust anything built on it.
`export_training_set` freezes a slice of the feature lake, joined to a
caller-supplied label, into a single Parquet file, and records a **manifest**
with everything needed to rebuild it: the feature list, the entities, the time
range and interval, the label definition, and an optional code SHA.

The manifest is serialized canonically (sorted keys, no wall-clock fields) and
content-hashed. Same definition gives the same hash. Any change moves it.

```python
spec = TrainingSetSpec(
    experiment="baseline", version="1",
    feature_names=("mean_60s", "std_300s", "zscore_300s"),
    entities=(("series", "series_a"),),
    start_ts=start, end_ts=end, interval=timedelta(minutes=1),
    label=LabelSpec(name="next_move", description="forward-looking target"),
    code_sha="a1b2c3d",
)

path = export_training_set(spec, label_fn)   # writes dataset.parquet + manifest.json
df   = load_training_set(path)               # a pandas DataFrame, ready to train

manifest_hash(spec)   # -> stable 64-char sha256, identical across runs/machines
```

The hash is a pure function of the definition.
[`tests/test_manifest_reproducibility.py`](tests/test_manifest_reproducibility.py)
checks it from several angles: two independently constructed specs with
identical fields hash the same; changing the feature list (or just its order),
the code SHA, the entities, the time range, the interval, or the label each
moves the hash; and the manifest file on disk hashes to exactly
`manifest_hash(spec)` on both Windows and POSIX.

Labels are intentionally unconstrained. A supervised target is supposed to look
forward. Leakage protection applies to features, which `AsOfContext` already
covers, and keeping those two concerns separate is deliberate.

Every export also lands in a small SQLite ledger (`training_set_manifests`)
keyed by `manifest_id`, so re-exports are idempotent and the provenance of any
dataset is one query away.

---

## 3. The dead-but-labeled trap

Consider a windowed stream: events arrive tagged with the fixed-cadence window
they belong to, and after a window closes it receives a label. You join
per-window features to the label and train. Now suppose the collector was down
for one window, or the stream froze mid-window for a few minutes. The window
still closes. It still gets a label. It looks like an ordinary training row.

It isn't. Its features were computed over a stream that was effectively absent,
and the model learns on noise. No error is raised, nothing looks empty, and an
in-stream sequence-gap log can't see it either: there is no gap between
messages when there are no messages. Cheap to prevent, invisible until it has
already cost you.

The auditor derives liveness from the stream itself, never from a gap log, and
joins it to the label stream:

```python
from pit_feature_store.quality import liveness

rep = liveness.report(raw_root=raw, window=timedelta(minutes=15),
                      entities=("series_x",))
print(liveness.render(rep))

# programmatic hooks a monitor / CI job exits non-zero on:
liveness.dead_but_labeled(rep)      # labeled windows whose stream was dead
liveness.missing_but_labeled(rep)   # labeled windows with ZERO events at all
```

Each observed window is classified `ok` / `thin` / `degraded` / `dead` from
metrics computed in one DuckDB pass: row count, longest blackout, total dark
time, coverage of the decision-critical final slice, and start/end offsets.
Verdict cutoffs are fractions of the window length, so the same thresholds work
whether your window is 15 minutes or 15 seconds. Two failure modes get surfaced
by name:

- **dead-but-labeled** - a window classified `dead` that still carries a label.
  Fence these before any build.
- **missing-but-labeled** - a labeled window with no events at all, the
  process-down case an in-stream log cannot detect.

Sample output (from the end-to-end smoke: one healthy window, one mid-window
stall that still got a label):

```
Stream liveness @ 2026-05-01T...
Window: 900s
Lookback: all

** 1 DEAD-BUT-LABELED window(s) -- a label sits on a stream that was effectively absent. FENCE before training. **

WINDOWS  (n=2)  dead=1  degraded=0  thin=0  ok=1  missing=0
  entity          ok thin degr dead miss  labeled
  series_x         1    0    0    1    0        2

FLAGGED WINDOWS (worst first)
  entity       window_start              verdict      rows  max_gap  dead_s  last lbl
  series_x     2026-05-01T00:15:00+00:00 dead            6    246.0     246     0   Y
```

`WindowLiveness.usable` (true only for `ok`) is the gate a training export
joins on to drop or down-weight the bad windows. Healthy, dead-but-labeled, and
missing-but-labeled scenarios are each constructed and asserted in
[`tests/test_liveness.py`](tests/test_liveness.py).

---

## Architecture

```
src/pit_feature_store/
  paths.py                    canonical data-root resolution (PIT_FS_DATA_ROOT)
  storage/                    portable storage layer
    layout.py                 partition grammar: source / data_type / entity / dt / hour
    schema.py                 generic PyArrow schemas: tick, stream, label, feature
    writer.py                 buffered, atomic, partition-aware Parquet writer
    query.py                  DuckDB reader with dt-partition pruning
  features/
    registry.py               @feature decorator + FeatureRegistry
    context.py                AsOfContext  <-- the leakage guarantee lives here
    sources.py                entity -> (source, entity) routing table
    builder.py                BatchBuilder: grid x entities -> feature lake
    training.py               frozen export + canonical, content-hashed manifest
    packs/rolling.py          example pack: rolling stats over a tick stream
  quality/
    liveness.py               windowed-stream liveness / coverage auditor
  manifests/
    db.py                     SQLite reproducibility ledger
```

### Data model

The lake is Hive-partitioned Parquet under a single grammar:

```
<root>/source=<>/data_type=<>/entity=<>/dt=YYYY-MM-DD/hour=HH/part-NNNN.parquet
```

- **`source`** - where the data came from (a feed or producer; for the feature
  lake, the entity type).
- **`data_type`** - `tick`, `stream`, `label`, or `feature`.
- **`entity`** - the thing being measured (a series, device, or market id).
- **`dt` / `hour`** - derived from `event_ts` so time-range queries prune cheaply
  with a `WHERE dt IN (...)` predicate.

Every raw record carries a universal envelope (`event_ts`, `ingest_ts`, `seq`,
`raw`) plus type-specific fields. Partition columns live in the path, not the
files, and DuckDB recovers them at read time.

The bundled example wires a synthetic `series_*` tick stream. Swap
`features/sources.py` and the schemas for your own domain and nothing
downstream changes.

---

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate
pip install -e ".[dev]"

pytest        # 57 tests, fully offline
ruff check .  # clean
```

Build features, freeze a training set, and audit a stream, end to end:

```python
from datetime import datetime, timedelta, timezone
from pit_feature_store.storage.schema import SCHEMAS
from pit_feature_store.storage.writer import ParquetWriter
from pit_feature_store.features.builder import BatchBuilder
from pit_feature_store.features.registry import REGISTRY
import pit_feature_store.features.packs            # registers the rolling pack

# 1) land some raw ticks
w = ParquetWriter(SCHEMAS, root="data/raw/v=1")
w.append_many("synth", "tick", "series_a", rows)   # rows: dicts w/ event_ts, value, ...
w.flush_all()

# 2) build features over a time grid
BatchBuilder(
    features=REGISTRY.list("series"),
    entities=[("series", "series_a")],
    start_ts=t0, end_ts=t1, interval=timedelta(minutes=1),
    writer=ParquetWriter(SCHEMAS, root="data/features"),
    raw_root="data/raw/v=1",
).run()

# 3) freeze + reload a training set  (see section 2)
# 4) audit stream liveness           (see section 3)
```

---

## Design notes

- **Offline by construction.** The test suite builds its own Parquet lakes in
  `tmp_path` and reads them back with DuckDB. No network, no fixtures pulled
  from anywhere, no external services. The liveness auditor does its time math
  in epoch seconds so it runs on a bare DuckDB install with no ICU/timezone
  extensions.
- **DuckDB does the heavy lifting.** Per-window coverage (longest gap, dark
  time, distinct values, final-slice coverage, edge offsets) is one windowed
  SQL pass, not a Python loop over rows.
- **Atomic writes.** The writer stages to a hidden `.tmp` file and
  `os.replace`s it into place, so a reader never sees a half-written partition.
- **Small on purpose.** No orchestration framework, no service, no config
  system. It is a library with testable guarantees, the pieces you would
  otherwise get wrong.

## License

MIT © 2026 Brian Cox
