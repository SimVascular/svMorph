"""Tests for svmorph.core.geometry – centerline resampling."""

import numpy as np
import pytest

from svmorph.core.geometry import resample_stent_axis


def _straight_line(n: int = 20, length: float = 2.0) -> np.ndarray:
    """Return n evenly-spaced points along the X axis from 0 to length."""
    x = np.linspace(0, length, n)
    return np.column_stack([x, np.zeros(n), np.zeros(n)])


def _flat_maps(n: int):
    """Trivial parent-tip map and segment-base mask for a single segment."""
    parent_tip_map = {}
    segment_base_mask = np.zeros(n, dtype=bool)
    return parent_tip_map, segment_base_mask


def test_output_is_jax_array():
    import jax.numpy as jnp
    pts = _straight_line()
    ptm, sbm = _flat_maps(len(pts))
    result = resample_stent_axis(pts, ptm, sbm, 10, 0.5, 0.1, sampling_direction=-1)
    assert isinstance(result, jnp.ndarray)


def test_resampled_length_proximal():
    pts = _straight_line(n=100, length=5.0)
    ptm, sbm = _flat_maps(len(pts))
    result = resample_stent_axis(pts, ptm, sbm, 50, 1.0, 0.1, sampling_direction=-1)
    # Arc length from idx=50 going backward 1.0 cm
    assert result.shape[0] >= 2
    coords = np.array(result)
    arc = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    assert arc == pytest.approx(1.0, abs=0.05)


def test_resampled_length_distal():
    pts = _straight_line(n=100, length=5.0)
    ptm, sbm = _flat_maps(len(pts))
    result = resample_stent_axis(pts, ptm, sbm, 10, 1.0, 0.1, sampling_direction=1)
    coords = np.array(result)
    arc = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    assert arc == pytest.approx(1.0, abs=0.05)


def test_segment_length_matches_spacing():
    pts = _straight_line(n=100, length=5.0)
    ptm, sbm = _flat_maps(len(pts))
    seg_len = 0.2
    result = resample_stent_axis(pts, ptm, sbm, 50, 1.0, seg_len, sampling_direction=-1)
    coords = np.array(result)
    spacings = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    for s in spacings:
        assert s == pytest.approx(seg_len, abs=1e-6)


def test_invalid_direction_raises():
    pts = _straight_line()
    ptm, sbm = _flat_maps(len(pts))
    with pytest.raises(ValueError, match="sampling_direction"):
        resample_stent_axis(pts, ptm, sbm, 5, 0.5, 0.1, sampling_direction=0)


def test_too_few_points_raises():
    pts = np.array([[0.0, 0.0, 0.0]])
    ptm, sbm = _flat_maps(1)
    with pytest.raises(ValueError, match="Not enough points"):
        resample_stent_axis(pts, ptm, sbm, 0, 0.5, 0.1, sampling_direction=-1)
