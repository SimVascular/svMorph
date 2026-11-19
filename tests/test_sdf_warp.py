"""Tests for svmorph.core.sdf_warp – the Warp kernels against the JAX reference, one subroutine at a time.

Every test here runs once per Warp device this machine has (``warp_device`` in
tests/conftest.py): on the CPU everywhere, and on CUDA where there is a GPU.
The whole module is skipped where Warp is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

pytest.importorskip("warp", reason="the Warp backend needs warp-lang (pip install 'svmorph[warp]')")

from svmorph.core import deformation, sdf_warp  # noqa: E402
from svmorph.core.geometry import flared_stent_radius_profile  # noqa: E402
from svmorph.core.units import set_unit_scale  # noqa: E402
from tests.helpers import (  # noqa: E402
    copy_data,
    make_stent_axis,
    make_tube,
    reference_sdf,
    with_jax_reference,
    without_borderline_points,
)

# Agreement between two single-precision implementations of the same arithmetic.  Typical
# differences are a few 1e-7; the SDF bound is set by steep radius profiles (the dumbbell below),
# where a rounding error in a point's arc-length coordinate is multiplied by the slope of the
# profile and reaches about 1e-5 for points a few stent radii away from the axis.  For scale:
# the contact distance is 1e-3 and a deployment step 1e-2.
SDF_TOLERANCE = 3e-5
DIRECTION_TOLERANCE = 2e-5
DISPLACEMENT_TOLERANCE = 5e-6

SMOOTHING, CONTACT_DISTANCE, INFLUENCE_RADIUS, F_SCALE, TARGET_RADIUS = 0.01, 0.001, 0.65, 0.01, 0.8


@pytest.fixture(autouse=True)
def _unit_scale():
    set_unit_scale(1.0)


@pytest.fixture(autouse=True)
def _fresh_buffers():
    """No test starts with the device buffers, or the Morton order, another test left behind."""
    sdf_warp.clear_cache()
    yield
    sdf_warp.clear_cache()


def _curved_axis(n: int = 21, length: float = 2.0, radius_of_curvature: float = 1.5) -> np.ndarray:
    """Stent axis vertices along a circular arc in the XY plane."""
    angle = np.linspace(0.0, length, n) / radius_of_curvature
    return radius_of_curvature * np.column_stack([np.sin(angle), 1.0 - np.cos(angle), np.zeros(n)])


def _dumbbell(number_of_vertices: int) -> np.ndarray:
    """Wide - narrow waist - wide: the radius secants change sign, which the monotone cubic must respect."""
    radii = np.full(number_of_vertices, 0.56)
    radii[number_of_vertices // 3: 2 * number_of_vertices // 3] = 0.44
    return radii


# (name, radius as a function of the stent axis, cap_height_fraction)
STENTS = [
    ("uniform", lambda axis: 0.505, 1.0),
    ("uniform, flattened caps", lambda axis: 0.505, 0.35),
    ("taper", lambda axis: np.linspace(0.40, 0.56, len(axis)), 1.0),
    ("taper, flattened caps", lambda axis: np.linspace(0.40, 0.56, len(axis)), 0.35),
    ("flared ends", lambda axis: flared_stent_radius_profile(axis, 0.47, 0.58, 0.8, flare_end="both"), 0.35),
    ("dumbbell", lambda axis: _dumbbell(len(axis)), 1.0),
]
stent_cases = pytest.mark.parametrize("make_radius, cap_height_fraction",
                                      [case[1:] for case in STENTS], ids=[case[0] for case in STENTS])


def _step(data: dict, stent_vertices, radius, cap_height_fraction: float = 1.0, **overrides):
    """One deployment step through the public entry point, on the backend in use."""
    arguments = dict(s=-1.0, target_stent_radius=TARGET_RADIUS, current_stent_radius=radius,
                     cap_height_fraction=cap_height_fraction)
    arguments.update(overrides)
    return deformation.compute_sdf_contact_displacements(data, stent_vertices, **arguments)


def _kernel_arguments(cap_height_fraction: float = 1.0) -> dict:
    return deformation._warp_step_parameters(
        -1.0, TARGET_RADIUS, INFLUENCE_RADIUS, CONTACT_DISTANCE, F_SCALE, cap_height_fraction)


# ── Radius profile ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("make_radius", [
    lambda axis: np.linspace(0.40, 0.56, len(axis)),
    lambda axis: flared_stent_radius_profile(axis, 0.47, 0.58, 0.8, flare_end="both"),
    lambda axis: _dumbbell(len(axis)),
    lambda axis: 0.3,
], ids=["taper", "flared ends", "dumbbell", "scalar"])
@pytest.mark.parametrize("number_of_vertices", [2, 3, 21])
def test_radius_profile_matches_the_reference_slopes(make_radius, number_of_vertices):
    # The monotone-cubic slopes are computed on the host for the kernel; the reference computes
    # them inside its jitted kernel
    axis = _curved_axis(number_of_vertices).astype(np.float32)
    radius = make_radius(axis)
    profile = sdf_warp._radius_profile(axis, radius)

    arc_positions = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(axis, axis=0), axis=1))])
    radii = np.broadcast_to(np.asarray(radius, dtype=np.float32), (number_of_vertices,))
    expected_slopes = np.asarray(deformation._radius_profile_slopes(
        np.asarray(arc_positions, dtype=np.float32), radii))
    assert profile.shape == (number_of_vertices, 3) and profile.dtype == np.float32
    np.testing.assert_allclose(profile[:, 0], arc_positions, atol=1e-6)
    np.testing.assert_allclose(profile[:, 1], radii, atol=0)
    np.testing.assert_allclose(profile[:, 2], expected_slopes, atol=1e-5)


# ── Stent SDF ────────────────────────────────────────────────────────────────

@stent_cases
def test_sdf_agrees_with_the_reference(warp_device, make_radius, cap_height_fraction):
    axis = _curved_axis()
    radius = make_radius(axis)
    rng = np.random.default_rng(0)
    query = rng.uniform(axis.min(axis=0) - 0.9, axis.max(axis=0) + 0.9, size=(5000, 3))

    expected_distance, expected_direction = reference_sdf(query, axis, radius, cap_height_fraction)
    distance, direction = sdf_warp.capsule_chain_sdf(
        query, axis, radius, cap_height_fraction, smoothing=SMOOTHING)
    assert distance.shape == (len(query),) and distance.dtype == np.float32
    assert direction.shape == (len(query), 3) and direction.dtype == np.float32
    np.testing.assert_allclose(distance, expected_distance, atol=SDF_TOLERANCE)
    np.testing.assert_allclose(direction, expected_direction, atol=DIRECTION_TOLERANCE)


@pytest.mark.parametrize("number_of_vertices", [2, 3])
def test_sdf_of_the_shortest_stents_agrees_with_the_reference(warp_device, number_of_vertices):
    # One and two segments: the loops over segments and over the radius profile have least to do
    axis = _curved_axis(number_of_vertices, length=0.4)
    radii = np.linspace(0.2, 0.3, number_of_vertices)
    rng = np.random.default_rng(1)
    query = rng.uniform(axis.min(axis=0) - 0.6, axis.max(axis=0) + 0.6, size=(500, 3))
    for radius, cap_height_fraction in [(0.25, 1.0), (radii, 0.5)]:
        expected_distance, expected_direction = reference_sdf(query, axis, radius, cap_height_fraction)
        distance, direction = sdf_warp.capsule_chain_sdf(
            query, axis, radius, cap_height_fraction, smoothing=SMOOTHING)
        np.testing.assert_allclose(distance, expected_distance, atol=SDF_TOLERANCE)
        np.testing.assert_allclose(direction, expected_direction, atol=DIRECTION_TOLERANCE)


def test_default_cap_is_spherical_and_flattened_cap_has_its_height(warp_device):
    # The surface on the axis lies one radius past the end vertex with the default cap, and
    # cap_height_fraction * radius past it with a flattened one
    axis = np.column_stack([np.zeros(11), np.zeros(11), np.linspace(0.0, 1.0, 11)])
    radius, fraction = 0.3, 0.35
    spherical, _ = sdf_warp.capsule_chain_sdf(
        np.array([[0.0, 0.0, 1.0 + radius]]), axis, radius, smoothing=SMOOTHING)
    flattened, _ = sdf_warp.capsule_chain_sdf(
        np.array([[0.0, 0.0, 1.0 + fraction * radius], [0.0, 0.0, -fraction * radius]]), axis, radius, fraction,
        smoothing=SMOOTHING)
    assert spherical[0] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(flattened, 0.0, atol=1e-6)


def test_sdf_of_no_points_is_empty(warp_device):
    distance, direction = sdf_warp.capsule_chain_sdf(np.zeros((0, 3)), _curved_axis(), 0.3, smoothing=SMOOTHING)
    assert distance.shape == (0,) and direction.shape == (0, 3)


def test_stent_needs_two_axis_vertices(warp_device):
    with pytest.raises(ValueError, match="two axis vertices"):
        sdf_warp.capsule_chain_sdf(np.zeros((4, 3)), np.zeros((1, 3)), 0.3, smoothing=SMOOTHING)


# ── A whole deployment step ──────────────────────────────────────────────────

@stent_cases
def test_step_agrees_with_the_reference(warp_device, make_radius, cap_height_fraction):
    axis = make_stent_axis()
    radius = make_radius(axis)
    data = without_borderline_points(make_tube(), axis, radius, cap_height_fraction)

    expected_surface, expected_centerline, expected_step = with_jax_reference(
        _step, data, axis, radius, cap_height_fraction)
    surface, centerline, step = _step(data, axis, radius, cap_height_fraction)

    # The stent touches part of the tube, so that some points are in contact, some are blended
    # in by their distance to the contact front, and some do not move
    weights = np.linalg.norm(expected_surface, axis=1) / F_SCALE
    assert np.count_nonzero(weights > 0.99) > 50
    assert np.count_nonzero((weights > 0.05) & (weights < 0.95)) > 50
    assert np.count_nonzero(weights == 0.0) > 50

    assert surface.dtype == np.float32 and centerline.dtype == np.float32
    assert surface.shape == expected_surface.shape and centerline.shape == expected_centerline.shape
    np.testing.assert_allclose(surface, expected_surface, atol=DISPLACEMENT_TOLERANCE)
    np.testing.assert_allclose(centerline, expected_centerline, atol=DISPLACEMENT_TOLERANCE)
    assert step == expected_step


def test_step_moves_nothing_before_contact(warp_device):
    data, axis = make_tube(), make_stent_axis()
    surface, centerline, step = _step(data, axis, 0.3)
    assert surface.shape == data["points"]["surface"].shape and not surface.any()
    assert centerline.shape == data["points"]["centerline"].shape and not centerline.any()
    assert step == pytest.approx(F_SCALE)


def test_step_pushes_out_points_inside_the_stent(warp_device):
    # A stent wider than the vessel: every wall point beside it is inside the stent, and is
    # moved out by its penetration depth on top of the step
    axis = make_stent_axis()
    data = without_borderline_points(make_tube(), axis, 0.6)
    expected_surface, expected_centerline, _ = with_jax_reference(_step, data, axis, 0.6)
    surface, centerline, _ = _step(data, axis, 0.6)
    assert np.linalg.norm(expected_surface, axis=1).max() > 0.08
    np.testing.assert_allclose(surface, expected_surface, atol=DISPLACEMENT_TOLERANCE)
    np.testing.assert_allclose(centerline, expected_centerline, atol=DISPLACEMENT_TOLERANCE)


def test_step_honours_explicit_parameters(warp_device):
    # Values other than the defaults reach the kernels: a wider contact zone, a shorter
    # influence radius, a larger and weaker step
    axis = make_stent_axis()
    overrides = dict(influence_radius=0.3, contact_distance=0.004, f_scale=0.02, s=-0.5)
    data = without_borderline_points(make_tube(), axis, 0.505, contact_distance=0.004)
    expected_surface, expected_centerline, expected_step = with_jax_reference(
        _step, data, axis, 0.505, **overrides)
    surface, centerline, step = _step(data, axis, 0.505, **overrides)
    assert step == expected_step == pytest.approx(0.01)
    np.testing.assert_allclose(surface, expected_surface, atol=DISPLACEMENT_TOLERANCE)
    np.testing.assert_allclose(centerline, expected_centerline, atol=DISPLACEMENT_TOLERANCE)


def test_step_scales_with_the_length_unit(warp_device):
    # The same geometry in millimetres moves ten times as far: every length the kernels use
    # comes from the active unit, none is frozen into them
    axis = make_stent_axis()
    data = without_borderline_points(make_tube(), axis, 0.505)
    surface_cm, centerline_cm, step_cm = _step(data, axis, 0.505)

    data_mm = copy_data(data)
    for points in data_mm["points"].values():
        points *= 10.0
    try:
        set_unit_scale(10.0)
        surface_mm, centerline_mm, step_mm = _step(
            data_mm, 10.0 * axis, 5.05, target_stent_radius=10.0 * TARGET_RADIUS)
    finally:
        set_unit_scale(1.0)
    assert step_mm == pytest.approx(10.0 * step_cm)
    np.testing.assert_allclose(surface_mm, 10.0 * surface_cm, atol=5e-5)
    np.testing.assert_allclose(centerline_mm, 10.0 * centerline_cm, atol=5e-5)


def test_step_leaves_its_inputs_alone_and_returns_arrays_of_its_own(warp_device):
    # Double-precision points are accepted, and what one step returned is not overwritten by the next
    axis = make_stent_axis()
    data = make_tube(dtype=np.float64)
    surface_before, centerline_before = data["points"]["surface"].copy(), data["points"]["centerline"].copy()

    first_surface, first_centerline, _ = _step(data, axis, 0.505)
    kept_surface, kept_centerline = first_surface.copy(), first_centerline.copy()
    second_surface, _, _ = _step(data, axis, 0.56)

    np.testing.assert_array_equal(data["points"]["surface"], surface_before)
    np.testing.assert_array_equal(data["points"]["centerline"], centerline_before)
    assert np.abs(second_surface - kept_surface).max() > 1e-3, "the two steps should differ"
    np.testing.assert_array_equal(first_surface, kept_surface)
    np.testing.assert_array_equal(first_centerline, kept_centerline)


def test_step_without_surface_or_centerline_points(warp_device):
    axis = make_stent_axis()
    data = make_tube()
    no_points = np.zeros((0, 3), dtype=np.float32)

    surface, centerline = sdf_warp.sdf_contact_displacements(
        no_points, data["points"]["centerline"], axis, 0.505, **_kernel_arguments())
    assert surface.shape == (0, 3) and not centerline.any()

    expected_surface, _ = sdf_warp.sdf_contact_displacements(
        data["points"]["surface"], data["points"]["centerline"], axis, 0.505, **_kernel_arguments())
    surface, centerline = sdf_warp.sdf_contact_displacements(
        data["points"]["surface"], no_points, axis, 0.505, **_kernel_arguments())
    assert centerline.shape == (0, 3)
    np.testing.assert_array_equal(surface, expected_surface)


# ── The two code paths stay separate ─────────────────────────────────────────

def _forbid(name: str):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")
    return _fail


def test_uniform_stent_takes_the_constant_radius_path(warp_device, monkeypatch):
    # A scalar radius with default caps must not pay for the radius profile
    monkeypatch.setattr(sdf_warp._DeviceStep, "_variable_radius_classification",
                        _forbid("_variable_radius_classification"))
    monkeypatch.setattr(sdf_warp, "_radius_profile", _forbid("_radius_profile"))
    surface, _, _ = _step(make_tube(), make_stent_axis(), 0.505)
    assert np.abs(surface).max() > 0.0


@pytest.mark.parametrize("radius, cap_height_fraction", [
    (np.full(21, 0.505), 1.0),  # per-vertex radii
    (0.505, 0.35),              # uniform radius with flattened caps
], ids=["per-vertex radii", "flattened caps"])
def test_variable_radius_stent_takes_the_separate_path(warp_device, monkeypatch, radius, cap_height_fraction):
    monkeypatch.setattr(sdf_warp._DeviceStep, "_uniform_classification", _forbid("_uniform_classification"))
    surface, _, _ = _step(make_tube(), make_stent_axis(), radius, cap_height_fraction)
    assert np.abs(surface).max() > 0.0


# ── Contact search ───────────────────────────────────────────────────────────

def _scattered_data(number_of_points: int, seed: int, radius: float = 0.47) -> tuple[dict, np.ndarray]:
    """Points scattered in no order over a tube that narrows onto the stent in places.

    A mesh hands its points over in whatever order it was written in, and the
    search tree is built on an order of its own, so nothing here follows the
    surface.
    """
    stent_z = np.linspace(1.0, 3.0, 21)
    axis = np.column_stack([np.zeros_like(stent_z), np.zeros_like(stent_z), stent_z])
    rng = np.random.default_rng(seed)
    surface = np.zeros((0, 3), dtype=np.float32)
    while len(surface) < number_of_points:
        angle = rng.uniform(0.0, 2.0 * np.pi, 2 * number_of_points + 16)
        z = rng.uniform(-1.0, 5.0, 2 * number_of_points + 16)
        tube_radius = 0.5 + 0.1 * np.cos(2.0 * z)
        candidates = np.column_stack([tube_radius * np.cos(angle), tube_radius * np.sin(angle), z])
        candidates = candidates.astype(np.float32)
        distance, _ = reference_sdf(candidates, axis, radius)
        surface = np.concatenate([surface, candidates[np.abs(distance - CONTACT_DISTANCE) > 1e-4]])
    centerline_z = np.linspace(-1.0, 5.0, 61) + 0.013  # clear of the stent's ends
    centerline = np.column_stack([np.zeros_like(centerline_z), np.zeros_like(centerline_z), centerline_z])
    data = {"points": {"surface": np.ascontiguousarray(surface[:number_of_points]),
                       "centerline": centerline.astype(np.float32)},
            "nodes": {"force_center_point_id": 30}}
    return data, axis


# Numbers of points on either side of where the search tree gains a leaf, a node or a level
_LEAF, _BRANCHING = int(sdf_warp._LEAF_SIZE), int(sdf_warp._BRANCHING)
TREE_SIZES = [1, _LEAF - 1, _LEAF, _LEAF + 1, _LEAF * _BRANCHING, _LEAF * _BRANCHING + 1,
              _LEAF * _BRANCHING ** 2 + 1, _LEAF * _BRANCHING ** 3 + 3]


@pytest.mark.parametrize("number_of_points", TREE_SIZES)
def test_contact_search_agrees_with_the_reference_on_scattered_points(warp_device, number_of_points):
    data, axis = _scattered_data(number_of_points, seed=number_of_points)
    expected_surface, expected_centerline, _ = with_jax_reference(_step, data, axis, 0.47)
    surface, centerline, _ = _step(data, axis, 0.47)
    if number_of_points > 100:
        weights = np.linalg.norm(expected_surface, axis=1) / F_SCALE
        assert np.count_nonzero((weights > 0.05) & (weights < 0.95)) > number_of_points / 20, (
            "some points should be at a distance from the contact front, where the search decides")
    np.testing.assert_allclose(surface, expected_surface, atol=DISPLACEMENT_TOLERANCE)
    np.testing.assert_allclose(centerline, expected_centerline, atol=DISPLACEMENT_TOLERANCE)


def test_contact_search_finds_the_nearest_point_in_contact(warp_device):
    # The search on its own, against SciPy's k-d tree: the blending weight of a point outside
    # the stent is 1 - (distance to the nearest point in contact) / influence_radius
    data, axis = _scattered_data(4000, seed=11)
    surface_points = data["points"]["surface"]
    distance, direction = reference_sdf(surface_points, axis, 0.47)
    in_contact = distance < CONTACT_DISTANCE
    nearest, _ = cKDTree(surface_points[in_contact]).query(surface_points)

    surface, _, _ = _step(data, axis, 0.47)
    searched = ~in_contact & (distance > 0.0) & (distance < INFLUENCE_RADIUS) & (nearest < INFLUENCE_RADIUS)
    assert np.count_nonzero(searched) > 500
    # displacement = f_scale * bump * weight * direction, the direction being a smooth-min blend
    # and so only approximately of unit length
    bump = ((distance[searched] / INFLUENCE_RADIUS) ** 2 - 1.0) ** 2
    weights = (np.linalg.norm(surface[searched], axis=1)
               / (F_SCALE * bump * np.linalg.norm(direction[searched], axis=1)))
    np.testing.assert_allclose(weights, 1.0 - nearest[searched] / INFLUENCE_RADIUS, atol=2e-4)


def test_contact_search_is_exact_in_an_order_sorted_for_other_points(warp_device):
    # The Morton order is kept for many steps while the points move, and for the next deployment
    # on a mesh of the same size.  It may be a poor order by then; it must not be a wrong one.
    first, axis = _scattered_data(3000, seed=1)
    _step(first, axis, 0.47)

    data, axis = _scattered_data(3000, seed=2)
    expected_surface, expected_centerline, _ = with_jax_reference(_step, data, axis, 0.47)
    surface, centerline, _ = _step(data, axis, 0.47)
    assert sdf_warp._cached_step.steps_since_sort == 2, "the second step should not have sorted again"
    np.testing.assert_allclose(surface, expected_surface, atol=DISPLACEMENT_TOLERANCE)
    np.testing.assert_allclose(centerline, expected_centerline, atol=DISPLACEMENT_TOLERANCE)


def test_stent_axis_that_moves_is_followed(warp_device):
    # Straightening moves the stent axis a little at every step: the kernels must use the new
    # axis without sorting again, and a stent somewhere else altogether must sort again
    data, axis = _scattered_data(3000, seed=5)
    nudged = axis + np.array([0.004, -0.003, 0.002])
    data = without_borderline_points(data, nudged, 0.47)
    _step(data, axis, 0.47)

    expected_surface, _, _ = with_jax_reference(_step, data, nudged, 0.47)
    surface, _, _ = _step(data, nudged, 0.47)
    assert sdf_warp._cached_step.steps_since_sort == 2, "a small move should not sort again"
    np.testing.assert_allclose(surface, expected_surface, atol=DISPLACEMENT_TOLERANCE)

    moved = axis + np.array([0.0, 0.0, 1.5])
    data_moved = without_borderline_points(data, moved, 0.47)
    sdf_warp.clear_cache()
    _step(data_moved, axis, 0.3)
    expected_surface, _, _ = with_jax_reference(_step, data_moved, moved, 0.47)
    surface, _, _ = _step(data_moved, moved, 0.47)
    assert sdf_warp._cached_step.steps_since_sort == 1, "a stent elsewhere should sort again"
    np.testing.assert_allclose(surface, expected_surface, atol=DISPLACEMENT_TOLERANCE)


def test_order_is_refreshed_after_many_steps(warp_device, monkeypatch):
    monkeypatch.setattr(sdf_warp, "_STEPS_PER_SORT", 3)
    data, axis = _scattered_data(600, seed=7)
    counts = []
    for _ in range(7):
        _step(data, axis, 0.47)
        counts.append(sdf_warp._cached_step.steps_since_sort)
    assert counts == [1, 2, 3, 1, 2, 3, 1]


def test_buffers_are_kept_for_a_mesh_of_the_same_size_and_replaced_for_another(warp_device):
    data, axis = _scattered_data(600, seed=3)
    _step(data, axis, 0.47)
    buffers = sdf_warp._cached_step
    _step(data, axis, 0.48)
    assert sdf_warp._cached_step is buffers

    other, _ = _scattered_data(700, seed=3)
    _step(other, axis, 0.47)
    assert sdf_warp._cached_step is not buffers

    sdf_warp.clear_cache()
    assert sdf_warp._cached_step is None
