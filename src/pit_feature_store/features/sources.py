"""Single source of truth for entity -> (source, entity-in-lake) routing.

Feature packs ask ``resolve(entity_id, data_type)`` instead of carrying their
own private mappings, so pointing an entity at a different feed — or adding a
new entity — is a one-line change here.

The example table wires a few synthetic series to a ``synth`` source. Swap this
for your real routing (e.g. a device to its telemetry feed, a security to a
market-data vendor); nothing downstream needs to change.
"""

from __future__ import annotations

# (entity_id, data_type) -> (source, entity name in the lake)
SOURCES: dict[tuple[str, str], tuple[str, str]] = {
    ("series_a", "tick"): ("synth", "series_a"),
    ("series_b", "tick"): ("synth", "series_b"),
    ("series_c", "tick"): ("synth", "series_c"),
}


def resolve(entity_id: str, data_type: str) -> tuple[str, str] | None:
    """Return ``(source, entity)`` for this entity's data of this type, or None."""
    return SOURCES.get((entity_id, data_type))
