"""Tests for svmorph.core.deformation – tapered capsule-chain SDF."""

from __future__ import annotations

import numpy as np
import pytest

from svmorph.core.deformation import capsule_sdf, smin_sdf_capsule_contact_sculpt
from svmorph.core.units import set_unit_scale


@pytest.fixture(autouse=True)
def _unit_scale():
    set_unit_scale(1.0)


def _straight_axis(n: int = 11, length: float = 1.0) -> np.ndarray:
    """Return n evenly-spaced stent axis vertices along the Z axis from 0 to length."""
    z = np.linspace(0.0, length, n)
    return np.column_stack([np.zeros(n), np.zeros(n), z])


def _surface_radius_along_ray(axis_point: np.ndarray, radii, cap_height_fraction: float = 1.0,
                              vertices: np.ndarray | None = None) -> float:
    """Radial position of the SDF zero crossing on a +X ray from axis_point."""
    vertices = _straight_axis() if vertices is None else vertices
    radial = np.linspace(0.01, 2.0, 400)
    query = np.tile(np.asarray(axis_point, dtype=float), (len(radial), 1))
    query[:, 0] += radial
    distances = np.asarray(capsule_sdf(query, vertices, radii, cap_height_fraction))[:, 0]
    return float(radial[np.argmin(np.abs(distances))])


def test_scalar_radius_matches_constant_profile():
    vertices = _straight_axis()
    query = np.array([[0.35, 0.1, 0.4], [0.0, 0.0, -0.3], [0.5, -0.2, 1.2]])
    scalar = np.asarray(capsule_sdf(query, vertices, 0.25))
    per_vertex = np.asarray(capsule_sdf(query, vertices, np.full(len(vertices), 0.25)))
    np.testing.assert_allclose(scalar, per_vertex, atol=1e-6)


def test_tapered_radius_is_interpolated_along_segment():
    # Single cone segment from radius 0.2 to 0.4: surface radius at the midpoint is 0.3
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    radii = np.array([0.2, 0.4])
    midpoint_radius = _surface_radius_along_ray([0.0, 0.0, 0.5], radii, vertices=vertices)
    assert midpoint_radius == pytest.approx(0.3, abs=0.01)


def test_smin_sculpt_matches_capsule_sdf():
    vertices = _straight_axis()
    radii = np.linspace(0.2, 0.5, len(vertices))
    query = np.array([[0.4, 0.0, 0.2], [0.1, 0.3, 0.9], [0.0, 0.0, 1.4]])
    sdf = np.asarray(capsule_sdf(query, vertices, radii, 0.5))
    sculpt_dist, sculpt_dir = smin_sdf_capsule_contact_sculpt(query, vertices, radii, 0.5)
    np.testing.assert_allclose(sdf, np.asarray(sculpt_dist), atol=1e-6)
    assert np.asarray(sculpt_dir).shape == (len(query), 3)


def test_default_cap_is_spherical():
    # A point on the axis beyond the end of a uniform capsule chain is on the surface
    # at one radius past the end vertex (spherical end cap)
    vertices = _straight_axis()
    r = 0.3
    query = np.array([[0.0, 0.0, 1.0 + r]])
    distance = float(np.asarray(capsule_sdf(query, vertices, r))[0, 0])
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_flattened_cap_height():
    # With a flattened cap the surface on the axis is cap_height_fraction * radius
    # past the end vertex
    vertices = _straight_axis()
    r, fraction = 0.3, 0.35
    query = np.array([[0.0, 0.0, 1.0 + fraction * r]])
    distance = float(np.asarray(capsule_sdf(query, vertices, r, fraction))[0, 0])
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_concave_profile_preserved_by_flattened_caps():
    # Dumbbell profile: wide - narrow waist - wide. With spherical caps the wide
    # capsules' end caps bulge into the waist and fill it in; flattened caps keep it.
    vertices = _straight_axis(n=21, length=2.0)
    radii = np.full(len(vertices), 1.2)
    radii[6:15] = 0.5  # waist between z=0.6 and z=1.4
    waist_center = [0.0, 0.0, 1.0]
    flattened_waist = _surface_radius_along_ray(waist_center, radii, cap_height_fraction=0.35,
                                                vertices=vertices)
    spherical_waist = _surface_radius_along_ray(waist_center, radii, cap_height_fraction=1.0,
                                                vertices=vertices)
    assert flattened_waist == pytest.approx(0.5, abs=0.15)
    assert spherical_waist > 1.0
