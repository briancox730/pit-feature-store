"""Point-in-time feature framework: registry, as-of context, builder, export.

The leakage guarantee lives in :mod:`pit_feature_store.features.context`: every
feature reads through an :class:`~pit_feature_store.features.context.AsOfContext`
that clips reads to ``as_of_ts``, so a feature can never observe the future.
"""
