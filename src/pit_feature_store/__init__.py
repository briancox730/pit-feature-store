"""pit_feature_store: a point-in-time-correct feature store.

The three things this package exists to guarantee:

1. **No leakage, by construction.** Features are computed through an
   :class:`~pit_feature_store.features.context.AsOfContext` that pins an
   ``as_of_ts`` and clips every read to it. A feature *cannot* observe data
   stamped at or after the cutoff, so a backtest built on these features can
   never accidentally peek at the future.

2. **Reproducible training sets.** Every frozen training-set export writes a
   deterministic manifest (feature list, entities, time range, label
   definition, code SHA) whose content hash is stable across runs. Same inputs
   -> same hash; any change to the definition changes the hash.

3. **Honest data quality.** A stream-liveness auditor derives per-window
   coverage from the stream itself and flags the silent trap: windows that
   carry a label but whose stream was dead or entirely absent.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
