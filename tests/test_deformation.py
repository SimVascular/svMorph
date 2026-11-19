"""Tests for svmorph.core.deformation – capsule-chain stent SDF, uniform and with a radius profile."""

from __future__ import annotations

import numpy as np
import pytest

from svmorph.core import deformation
from svmorph.core.deformation import (
    capsule_sdf,
    compute_sdf_contact_displacements,
    smin_sdf_capsule_contact_sculpt,
    smin_sdf_variable_radius_capsule_contact_sculpt,
    variable_radius_capsule_sdf,
)
from svmorph.core.geometry import flared_stent_radius_profile
from svmorph.core.units import set_unit_scale


@pytest.fixture(autouse=True)
def _unit_scale():
    set_unit_scale(1.0)


def _straight_axis(n: int = 11, length: float = 1.0) -> np.ndarray:
    """Return n evenly-spaced stent axis vertices along the Z axis from 0 to length."""
    z = np.linspace(0.0, length, n)
    return np.column_stack([np.zeros(n), np.zeros(n), z])


def _curved_axis(n: int = 21, length: float = 2.0, radius_of_curvature: float = 1.5) -> np.ndarray:
    """Return n evenly-spaced stent axis vertices along a circular arc in the XY plane."""
    angle = np.linspace(0.0, length, n) / radius_of_curvature
    return radius_of_curvature * np.column_stack([np.sin(angle), 1.0 - np.cos(angle), np.zeros(n)])


def _surface_radius(axial_positions, radii, cap_height_fraction: float = 1.0,
                    vertices: np.ndarray | None = None) -> np.ndarray:
    """Radial position of the SDF zero crossing on +X rays from points on the Z axis (bisection)."""
    vertices = _straight_axis() if vertices is None else vertices
    axial_positions = np.atleast_1d(np.asarray(axial_positions, dtype=float))
    inside = np.full(len(axial_positions), 1e-3)
    outside = np.full(len(axial_positions), 3.0)
    for _ in range(40):
        middle = 0.5 * (inside + outside)
        query = np.column_stack([middle, np.zeros_like(middle), axial_positions])
        distances = np.asarray(variable_radius_capsule_sdf(query, vertices, radii, cap_height_fraction))[:, 0]
        inside = np.where(distances < 0.0, middle, inside)
        outside = np.where(distances < 0.0, outside, middle)
    return 0.5 * (inside + outside)


def _vessel_data(radius: float = 0.25, length: float = 1.0) -> dict:
    """Minimal simulation data: a cylindrical vessel wall around the Z axis and its centerline."""
    angle, z = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False), np.linspace(-0.5, length + 0.5, 41))
    surface = np.column_stack([radius * np.cos(angle.ravel()), radius * np.sin(angle.ravel()), z.ravel()])
    centerline = np.column_stack([np.zeros(41), np.zeros(41), np.linspace(-0.5, length + 0.5, 41)])
    return {"nodes": {"force_center_point_id": 0}, "points": {"surface": surface, "centerline": centerline}}


def test_variable_radius_path_matches_uniform_stent():
    # For a uniform radius and default caps the variable-radius code path agrees with
    # the constant-radius capsule chain
    vertices = _curved_axis()
    rng = np.random.default_rng(0)
    query = rng.uniform(vertices.min(axis=0) - 0.8, vertices.max(axis=0) + 0.8, size=(2000, 3))
    uniform_dist, uniform_dir = smin_sdf_capsule_contact_sculpt(query, vertices, 0.3)
    dist, direction = smin_sdf_variable_radius_capsule_contact_sculpt(query, vertices, np.full(len(vertices), 0.3))
    np.testing.assert_allclose(np.asarray(dist), np.asarray(uniform_dist), atol=1e-5)
    np.testing.assert_allclose(np.asarray(direction), np.asarray(uniform_dir), atol=1e-5)
    np.testing.assert_allclose(np.asarray(variable_radius_capsule_sdf(query, vertices, 0.3)),
                               np.asarray(capsule_sdf(query, vertices, 0.3)), atol=1e-5)


def _forbid(name: str):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")
    return _fail


def test_uniform_stent_deployment_uses_constant_radius_path(monkeypatch):
    # A scalar radius with default caps must not go through the variable-radius code path
    monkeypatch.setattr(deformation, "smin_sdf_variable_radius_capsule_contact_sculpt",
                        _forbid("smin_sdf_variable_radius_capsule_contact_sculpt"))
    surface_disp, _, step = compute_sdf_contact_displacements(
        _vessel_data(), _straight_axis(), s=-1.0, target_stent_radius=0.4, current_stent_radius=0.26)
    assert step == pytest.approx(0.01)
    assert np.abs(surface_disp).max() > 0.0  # the wall is in contact and gets pushed


@pytest.mark.parametrize("radius, cap_height_fraction", [
    (np.full(11, 0.26), 1.0),  # per-vertex radii
    (0.26, 0.35),              # uniform radius with flattened caps
])
def test_variable_radius_deployment_uses_separate_path(monkeypatch, radius, cap_height_fraction):
    monkeypatch.setattr(deformation, "smin_sdf_capsule_contact_sculpt", _forbid("smin_sdf_capsule_contact_sculpt"))
    surface_disp, _, _ = compute_sdf_contact_displacements(
        _vessel_data(), _straight_axis(), s=-1.0, target_stent_radius=np.full(11, 0.4),
        current_stent_radius=radius, cap_height_fraction=cap_height_fraction)
    assert np.abs(surface_disp).max() > 0.0


def test_scalar_radius_matches_constant_profile():
    vertices = _straight_axis()
    query = np.array([[0.35, 0.1, 0.4], [0.0, 0.0, -0.3], [0.5, -0.2, 1.2]])
    scalar = np.asarray(variable_radius_capsule_sdf(query, vertices, 0.25))
    per_vertex = np.asarray(variable_radius_capsule_sdf(query, vertices, np.full(len(vertices), 0.25)))
    np.testing.assert_allclose(scalar, per_vertex, atol=1e-6)


def test_tapered_radius_is_interpolated_along_segment():
    # Single cone segment from radius 0.2 to 0.4: surface radius at the midpoint is 0.3
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    radii = np.array([0.2, 0.4])
    midpoint_radius = _surface_radius(0.5, radii, vertices=vertices)[0]
    assert midpoint_radius == pytest.approx(0.3, abs=0.01)


def test_smin_sculpt_matches_capsule_sdf():
    vertices = _straight_axis()
    radii = np.linspace(0.2, 0.5, len(vertices))
    query = np.array([[0.4, 0.0, 0.2], [0.1, 0.3, 0.9], [0.0, 0.0, 1.4]])
    sdf = np.asarray(variable_radius_capsule_sdf(query, vertices, radii, 0.5))
    sculpt_dist, sculpt_dir = smin_sdf_variable_radius_capsule_contact_sculpt(query, vertices, radii, 0.5)
    np.testing.assert_allclose(sdf, np.asarray(sculpt_dist), atol=1e-6)
    assert np.asarray(sculpt_dir).shape == (len(query), 3)


@pytest.mark.parametrize("sdf", [capsule_sdf, variable_radius_capsule_sdf])
def test_default_cap_is_spherical(sdf):
    # A point on the axis beyond the end of a uniform capsule chain is on the surface
    # at one radius past the end vertex (spherical end cap)
    vertices = _straight_axis()
    r = 0.3
    query = np.array([[0.0, 0.0, 1.0 + r]])
    distance = float(np.asarray(sdf(query, vertices, r))[0, 0])
    assert distance == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("end_position, outward", [(1.0, 1.0), (0.0, -1.0)])
def test_flattened_cap_height(end_position, outward):
    # With a flattened cap the surface on the axis is cap_height_fraction * radius
    # past the end vertex, at both stent ends
    vertices = _straight_axis()
    r, fraction = 0.3, 0.35
    query = np.array([[0.0, 0.0, end_position + outward * fraction * r]])
    distance = float(np.asarray(variable_radius_capsule_sdf(query, vertices, r, fraction))[0, 0])
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_flattened_cap_lies_inside_spherical_cap():
    # A point past the stent end that the spherical cap still covers is outside the
    # half-ellipsoid cap
    vertices = _straight_axis()
    r = 0.3
    query = np.array([[0.8 * r, 0.0, 1.0 + 0.5 * r]])
    assert float(np.asarray(variable_radius_capsule_sdf(query, vertices, r, 1.0))[0, 0]) < 0.0
    assert float(np.asarray(variable_radius_capsule_sdf(query, vertices, r, 0.35))[0, 0]) > 0.0


def test_flattened_caps_only_change_the_stent_ends():
    # Along the stent body the distance and direction do not depend on the cap shape
    vertices = _straight_axis(n=17, length=1.6)
    radii = flared_stent_radius_profile(vertices, 0.3, 0.45, 0.5, flare_end="both")
    rng = np.random.default_rng(1)
    query = np.column_stack([rng.uniform(-0.9, 0.9, 500), rng.uniform(-0.9, 0.9, 500),
                             rng.uniform(0.0, 1.6, 500)])
    spherical_dist, spherical_dir = smin_sdf_variable_radius_capsule_contact_sculpt(query, vertices, radii, 1.0)
    flattened_dist, flattened_dir = smin_sdf_variable_radius_capsule_contact_sculpt(query, vertices, radii, 0.35)
    np.testing.assert_allclose(np.asarray(flattened_dist), np.asarray(spherical_dist), atol=1e-5)
    np.testing.assert_allclose(np.asarray(flattened_dir), np.asarray(spherical_dir), atol=1e-6)


@pytest.mark.parametrize("cap_height_fraction", [1.0, 0.35])
def test_stent_body_has_no_ripple(cap_height_fraction):
    # The surface of a uniform stent must not pulse with the period of the capsule
    # segments, whatever the end cap shape
    vertices = _straight_axis(n=17, length=1.6)
    stations = np.linspace(0.4, 1.2, 161)  # several stations per 0.1-long segment
    surface_radius = _surface_radius(stations, 0.3, cap_height_fraction, vertices)
    assert np.ptp(surface_radius) < 1e-3


@pytest.mark.parametrize("cap_height_fraction", [1.0, 0.35])
def test_flare_follows_radius_profile(cap_height_fraction):
    # Flare 0.3 -> 0.45 over the last 0.6 of a 2.3-long stent
    vertices = _straight_axis(n=24, length=2.3)
    radii = flared_stent_radius_profile(vertices, 0.3, 0.45, 0.6)
    stations = np.linspace(0.4, 2.1, 341)
    t = np.clip(1.0 - (2.3 - stations) / 0.6, 0.0, 1.0)
    prescribed = 0.3 + 0.15 * t * t * (3.0 - 2.0 * t)
    surface_radius = _surface_radius(stations, radii, cap_height_fraction, vertices)
    # The surface tracks the prescribed profile up to the (nearly uniform) smooth-min offset
    assert np.ptp(surface_radius - prescribed) < 5e-3
    assert np.all(np.diff(surface_radius) > -1e-4)


@pytest.mark.parametrize("cap_height_fraction", [1.0, 0.35])
def test_concave_profile_is_preserved(cap_height_fraction):
    # Dumbbell profile: wide - narrow waist - wide.  The wide parts must not fill in
    # the waist, and the profile must not overshoot the prescribed radii.
    vertices = _straight_axis(n=21, length=2.0)
    radii = np.full(len(vertices), 1.2)
    radii[6:15] = 0.5  # waist between z=0.6 and z=1.4
    stations = np.linspace(0.3, 1.7, 141)
    surface_radius = _surface_radius(stations, radii, cap_height_fraction, vertices)
    waist = (stations > 0.65) & (stations < 1.35)
    np.testing.assert_allclose(surface_radius[waist], 0.5, atol=0.03)
    assert surface_radius.min() > 0.5 - 1e-3
    assert surface_radius.max() < 1.2 + 0.03


def test_direction_is_radial_along_the_body():
    # The radius profile does not tilt the push direction: it stays the (smooth-min
    # blended, hence only approximately unit) axis-to-point direction
    vertices = _straight_axis()
    radii = np.linspace(0.2, 0.4, len(vertices))
    query = np.array([[0.5, 0.0, 0.5], [0.0, -0.6, 0.35]])
    _, direction = smin_sdf_variable_radius_capsule_contact_sculpt(query, vertices, radii, 0.35)
    np.testing.assert_allclose(np.asarray(direction), [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], atol=0.02)
