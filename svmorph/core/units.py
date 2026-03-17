"""Global length-unit configuration for svMorph.

All hardcoded spatial constants in the codebase were tuned for centimetres.
When input geometry uses millimetres (or another unit), call
:func:`set_unit_scale` once at startup with the appropriate factor
(e.g. 10.0 for mm) so that every ``L()``-scaled parameter adjusts
automatically.
"""

_UNIT_SCALE: float = 1.0  # 1.0 = cm (default), 10.0 = mm


def set_unit_scale(scale: float) -> None:
    """Set the global length scale factor (ratio of working unit to cm)."""
    global _UNIT_SCALE
    _UNIT_SCALE = scale


def get_unit_scale() -> float:
    """Return the current length scale factor."""
    return _UNIT_SCALE


def L() -> float:
    """Shorthand for the current length scale factor."""
    return _UNIT_SCALE


_UNIT_NAMES = {1.0: "cm", 10.0: "mm"}


def unit_name() -> str:
    """Return a human-readable abbreviation for the current length unit."""
    return _UNIT_NAMES.get(_UNIT_SCALE, f"{_UNIT_SCALE}x cm")
