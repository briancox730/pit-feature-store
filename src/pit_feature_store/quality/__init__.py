"""Stream data-quality auditing.

The headline check is the windowed-stream liveness auditor
(:mod:`pit_feature_store.quality.liveness`), which finds the silent trap that
corrupts training sets: windows that carry a label but whose stream was dead or
entirely absent.
"""
