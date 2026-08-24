"""Importing this package eagerly registers the bundled example feature packs.

Each submodule populates the global
:data:`pit_feature_store.features.registry.REGISTRY` at import time, so a caller
just does ``import pit_feature_store.features.packs`` (or imports one specific
pack module when it wants only that pack's features).
"""

from pit_feature_store.features.packs import rolling  # noqa: F401
