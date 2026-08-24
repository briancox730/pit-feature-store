"""Feature registry: declare features as Python callables with metadata.

A :class:`FeatureSpec` bundles a feature's name, the entity *type* it operates
on (``"series"``, ``"device"``, ...), a declarative list of input data_types
(informational — used for documentation and dependency-aware builds), and the
compute function. Use the :func:`feature` decorator to register one.

The compute function must accept ``(ctx, entity_id)`` and return either a single
float (the feature value) or ``None`` (skip — insufficient data). ``ctx`` is an
:class:`~pit_feature_store.features.context.AsOfContext` which already pins
``as_of_ts`` so reads cannot leak future data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pit_feature_store.features.context import AsOfContext


ComputeFn = Callable[["AsOfContext", str], float | None]
"""``compute_fn(ctx, entity_id)``. The cutoff is ``ctx.as_of_ts`` — a feature
must never look at any input row with ``event_ts >= as_of_ts``."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    entity_type: str
    inputs: tuple[str, ...]
    compute_fn: ComputeFn
    description: str = ""


class FeatureRegistry:
    """Process-local feature registry.

    The module-level :data:`REGISTRY` is the default; tests construct ad-hoc
    registries so the global one does not accumulate state across cases.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self._specs:
            raise ValueError(f"feature {spec.name!r} already registered")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def list(self, entity_type: str | None = None) -> list[FeatureSpec]:
        if entity_type is None:
            return list(self._specs.values())
        return [s for s in self._specs.values() if s.entity_type == entity_type]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def clear(self) -> None:
        """Drop all registrations — primarily for tests."""
        self._specs.clear()


REGISTRY = FeatureRegistry()


def feature(
    name: str,
    entity_type: str,
    inputs: tuple[str, ...] = (),
    description: str = "",
    registry: FeatureRegistry | None = None,
) -> Callable[[ComputeFn], FeatureSpec]:
    """Decorator: register the wrapped function as a feature.

    Returns the :class:`FeatureSpec` (the wrapped function is preserved on the
    spec's ``compute_fn``). Pass an explicit ``registry`` in tests so the global
    registry stays clean.
    """
    target = registry if registry is not None else REGISTRY

    def wrap(fn: ComputeFn) -> FeatureSpec:
        spec = FeatureSpec(
            name=name,
            entity_type=entity_type,
            inputs=tuple(inputs),
            compute_fn=fn,
            description=description,
        )
        target.register(spec)
        return spec

    return wrap
