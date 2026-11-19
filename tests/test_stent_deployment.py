"""Tests for svmorph.core.stent_deployment – the step loop of a stent deployment, on either backend."""

from __future__ import annotations

import numpy as np
import pytest

from svmorph.core import deformation, mesh_data
from svmorph.core.geometry import flared_stent_radius_profile
from svmorph.core.stent_deployment import StentDeployment
from svmorph.core.units import set_unit_scale
from tests.helpers import copy_data, make_stent_axis, make_tube, with_jax_reference

TARGET_RADIUS, SMOOTHING_K = 0.8, 0.01


@pytest.fixture(autouse=True)
def _unit_scale():
    set_unit_scale(1.0)


def _radii(number_of_steps: int, start: float = 0.40):
    """Stent radii of a deployment that reaches the tube wall (radius 0.5) about a third of the way in."""
    return [start + 0.01 * step for step in range(number_of_steps)]


def _deploy_step_by_step(data: dict, axis, radii, cap_height_fraction: float = 1.0) -> dict:
    """The loop the session replaces: compute a step, apply it, on the backend in use."""
    for radius in radii:
        surface, centerline, _ = deformation.compute_sdf_contact_displacements(
            data, axis, -1.0, TARGET_RADIUS, radius, cap_height_fraction=cap_height_fraction)
        mesh_data.apply_displacements(data, surface, "surface")
        mesh_data.apply_displacements(data, centerline, "centerline")
    return data


def _deploy_with_session(data: dict, axis, radii, cap_height_fraction: float = 1.0) -> dict:
    with StentDeployment(data, target_stent_radius=TARGET_RADIUS,
                         cap_height_fraction=cap_height_fraction) as session:
        for radius in radii:
            session.step(axis, radius)
    return data


def _flared(axis, radius: float) -> np.ndarray:
    """A flared radius profile scaled to the nominal *radius*, as scripts/deploy_stent.py scales it."""
    fractions = flared_stent_radius_profile(axis, 1.0, 1.2, 0.6, flare_end="both")
    return fractions * (radius + SMOOTHING_K) - SMOOTHING_K


# ── Under the JAX reference ──────────────────────────────────────────────────

def test_session_is_the_compute_then_apply_loop():
    # Bit for bit: under JAX the session only wraps what the scripts used to spell out
    axis, radii = make_stent_axis(), _radii(20)
    expected = _deploy_step_by_step(make_tube(), axis, radii)
    data = _deploy_with_session(make_tube(), axis, radii)
    assert np.abs(expected["points"]["surface"] - make_tube()["points"]["surface"]).max() > 0.05
    np.testing.assert_array_equal(data["points"]["surface"], expected["points"]["surface"])
    np.testing.assert_array_equal(data["points"]["centerline"], expected["points"]["centerline"])


def test_session_reports_the_backend_and_the_step_size():
    session = StentDeployment(make_tube(), target_stent_radius=TARGET_RADIUS)
    assert session.backend == "jax"
    assert session.step_size() == pytest.approx(0.01)
    assert session.step_size(s=-0.5) == pytest.approx(0.005)
    assert session.step(make_stent_axis(), 0.3) == pytest.approx(0.01)
    assert StentDeployment(make_tube(), target_stent_radius=TARGET_RADIUS, f_scale=0.02).step_size() == 0.02


def test_session_displaces_the_arrays_it_was_given():
    # In place: the arrays are views of the VTK point buffers in the front ends
    data = make_tube()
    surface, centerline = data["points"]["surface"], data["points"]["centerline"]
    before = surface.copy()
    _deploy_with_session(data, make_stent_axis(), _radii(15))
    assert data["points"]["surface"] is surface and data["points"]["centerline"] is centerline
    assert np.abs(surface - before).max() > 0.01


def test_session_takes_the_largest_of_per_vertex_target_radii():
    session = StentDeployment(make_tube(), target_stent_radius=np.array([0.4, 0.9, 0.6]))
    assert session.target_stent_radius == pytest.approx(0.9)


# ── Under Warp ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cap_height_fraction, flared", [(1.0, False), (0.35, False), (0.35, True)],
                         ids=["uniform", "flattened caps", "flared"])
def test_warp_session_follows_the_reference_deployment(warp_device, cap_height_fraction, flared):
    # A whole deployment, from before contact until the wall has been pushed out.  Points are
    # free to be classified differently by the two backends at some step (the SDF agrees to
    # about 1e-6, not exactly), but a point in contact is held on the stent surface at every
    # step, so the trajectories do not drift apart.
    axis = make_stent_axis()
    radii = [_flared(axis, radius) for radius in _radii(25)] if flared else _radii(25)
    expected = with_jax_reference(_deploy_step_by_step, make_tube(), axis, radii, cap_height_fraction)
    data = _deploy_with_session(make_tube(), axis, radii, cap_height_fraction)

    moved = np.linalg.norm(expected["points"]["surface"] - make_tube()["points"]["surface"], axis=1)
    assert moved.max() > 0.1
    np.testing.assert_allclose(data["points"]["surface"], expected["points"]["surface"], atol=1e-4)
    np.testing.assert_allclose(data["points"]["centerline"], expected["points"]["centerline"], atol=1e-4)


def test_warp_session_on_the_device_equals_step_by_step(warp_device):
    # Keeping the mesh on the device changes where the displacements are added, not what they
    # are: bit for bit the result of uploading, computing, downloading and adding at every step
    axis, radii = make_stent_axis(), _radii(20)
    expected = _deploy_step_by_step(make_tube(), axis, radii)
    data = make_tube()
    session = StentDeployment(data, target_stent_radius=TARGET_RADIUS)
    assert session.backend == "warp" and session._device_deployment is not None
    for radius in radii:
        session.step(axis, radius)
    session.sync()
    np.testing.assert_array_equal(data["points"]["surface"], expected["points"]["surface"])
    np.testing.assert_array_equal(data["points"]["centerline"], expected["points"]["centerline"])


def test_warp_session_updates_the_host_arrays_on_sync(warp_device):
    data = make_tube()
    before = copy_data(data)
    session = StentDeployment(data, target_stent_radius=TARGET_RADIUS)
    session.sync()  # nothing to bring back yet
    np.testing.assert_array_equal(data["points"]["surface"], before["points"]["surface"])

    assert session.step(make_stent_axis(), 0.52) == pytest.approx(0.01)
    # The step was taken on the device; the host arrays have not been written to
    np.testing.assert_array_equal(data["points"]["surface"], before["points"]["surface"])
    session.sync()
    after = data["points"]["surface"].copy()
    assert np.abs(after - before["points"]["surface"]).max() > 0.005
    session.sync()  # a second sync changes nothing
    np.testing.assert_array_equal(data["points"]["surface"], after)


def test_warp_session_as_a_context_manager_syncs_on_exit(warp_device):
    data = make_tube()
    before = data["points"]["surface"].copy()
    with StentDeployment(data, target_stent_radius=TARGET_RADIUS) as session:
        session.step(make_stent_axis(), 0.52)
    assert np.abs(data["points"]["surface"] - before).max() > 0.005


def test_warp_session_follows_a_stent_axis_that_changes_length(warp_device):
    # Another number of axis vertices needs new device buffers; the mesh must carry over
    radii = _radii(12)
    short, long = make_stent_axis(1.4, 2.6, 13), make_stent_axis(1.0, 3.0, 21)
    expected = make_tube()
    _deploy_step_by_step(expected, short, radii[:6])
    _deploy_step_by_step(expected, long, radii[6:])

    data = make_tube()
    with StentDeployment(data, target_stent_radius=TARGET_RADIUS) as session:
        for radius in radii[:6]:
            session.step(short, radius)
        for radius in radii[6:]:
            session.step(long, radius)
    np.testing.assert_array_equal(data["points"]["surface"], expected["points"]["surface"])
    np.testing.assert_array_equal(data["points"]["centerline"], expected["points"]["centerline"])


def test_warp_session_switches_between_the_two_stent_code_paths(warp_device):
    # A uniform stent for some steps, then a radius profile, on the same device buffers
    axis, radii = make_stent_axis(), _radii(12)
    mixed = radii[:6] + [_flared(axis, radius) for radius in radii[6:]]
    expected = _deploy_step_by_step(make_tube(), axis, mixed, 1.0)
    data = _deploy_with_session(make_tube(), axis, mixed, 1.0)
    np.testing.assert_array_equal(data["points"]["surface"], expected["points"]["surface"])


def test_warp_session_computes_step_by_step_for_arrays_it_cannot_write_back(warp_device):
    # Positions come back from the device in single precision, so a double-precision mesh is
    # not kept there: each step is computed by Warp and added on the host instead
    axis, radii = make_stent_axis(), _radii(15)
    data = make_tube(dtype=np.float64)
    session = StentDeployment(data, target_stent_radius=TARGET_RADIUS)
    assert session.backend == "warp" and session._device_deployment is None
    before = data["points"]["surface"].copy()
    for radius in radii:
        session.step(axis, radius)
        # ... which keeps the host arrays current after every step
    assert data["points"]["surface"].dtype == np.float64
    assert np.abs(data["points"]["surface"] - before).max() > 0.01
    expected = with_jax_reference(_deploy_step_by_step, make_tube(dtype=np.float64), axis, radii)
    np.testing.assert_allclose(data["points"]["surface"], expected["points"]["surface"], atol=1e-4)


def test_warp_session_before_contact_moves_nothing(warp_device):
    data = make_tube()
    before = copy_data(data)
    with StentDeployment(data, target_stent_radius=TARGET_RADIUS) as session:
        for radius in (0.2, 0.25, 0.3):
            session.step(make_stent_axis(), radius)
    np.testing.assert_array_equal(data["points"]["surface"], before["points"]["surface"])
    np.testing.assert_array_equal(data["points"]["centerline"], before["points"]["centerline"])
