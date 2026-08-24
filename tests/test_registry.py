"""Registry unit tests: registration, lookup, duplicate rejection, filtering."""

from __future__ import annotations

import pytest

from pit_feature_store.features.registry import FeatureRegistry, feature


def test_register_and_retrieve():
    reg = FeatureRegistry()

    @feature(name="x", entity_type="series", registry=reg)
    def x(ctx, entity_id):
        return 1.0

    assert "x" in reg
    assert len(reg) == 1
    assert reg.get("x").compute_fn is x.compute_fn
    assert reg.list() == [x]


def test_duplicate_register_raises():
    reg = FeatureRegistry()

    @feature(name="dup", entity_type="series", registry=reg)
    def f1(ctx, e):
        return 1.0

    with pytest.raises(ValueError):
        @feature(name="dup", entity_type="series", registry=reg)
        def f2(ctx, e):
            return 2.0


def test_list_filters_by_entity_type():
    reg = FeatureRegistry()

    @feature(name="s1", entity_type="series", registry=reg)
    def s1(ctx, e):
        return 1.0

    @feature(name="d1", entity_type="device", registry=reg)
    def d1(ctx, e):
        return 2.0

    assert {s.name for s in reg.list("series")} == {"s1"}
    assert {s.name for s in reg.list("device")} == {"d1"}
    assert {s.name for s in reg.list()} == {"s1", "d1"}


def test_clear_empties_registry():
    reg = FeatureRegistry()

    @feature(name="z", entity_type="series", registry=reg)
    def z(ctx, e):
        return 0.0

    reg.clear()
    assert len(reg) == 0
    assert "z" not in reg
