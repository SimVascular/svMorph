"""Tests for svmorph.core.units – unit scaling helpers."""

import pytest

import svmorph.core.units as units


def setup_function():
    """Reset unit scale to cm (1.0) before each test."""
    units.set_unit_scale(1.0)


def test_default_scale_is_cm():
    assert units.get_unit_scale() == 1.0


def test_L_returns_current_scale():
    assert units.L() == 1.0
    units.set_unit_scale(10.0)
    assert units.L() == 10.0


def test_set_and_get_unit_scale_mm():
    units.set_unit_scale(10.0)
    assert units.get_unit_scale() == 10.0


def test_unit_name_cm():
    units.set_unit_scale(1.0)
    assert units.unit_name() == "cm"


def test_unit_name_mm():
    units.set_unit_scale(10.0)
    assert units.unit_name() == "mm"


def test_unit_name_unknown():
    units.set_unit_scale(5.0)
    name = units.unit_name()
    assert "5.0" in name or "cm" in name


def test_arbitrary_scale():
    units.set_unit_scale(25.4)
    assert units.get_unit_scale() == pytest.approx(25.4)
    assert units.L() == pytest.approx(25.4)
