"""Tests for the stent deployment scripts – argument handling, and a deployment run end to end."""

import sys

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")  # the scripts import the VTK I/O layer; the rest of the suite does not need VTK

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # noqa: E402

from svmorph.core import backend  # noqa: E402
from svmorph.scripts import common, deploy_stent, deploy_stent_straighten  # noqa: E402

_REQUIRED = ["--mesh", "surface.vtp", "--cline", "centerline.vtp", "--start", "0"]


def test_flare_end_accepts_both():
    args = deploy_stent.build_parser().parse_args(_REQUIRED + ["--flare-R", "0.6", "--flare-end", "both"])
    assert args.flare_end == "both"
    assert args.cap_height_fraction == 1.0


@pytest.mark.parametrize("extra_args", [
    ["--cap-height-fraction", "0"],
    ["--cap-height-fraction", "1.5"],
    ["--flare-R", "0.6", "--flare-length", "0"],
])
def test_invalid_stent_shape_arguments_are_rejected(extra_args):
    # Rejected while parsing, before any input file is read
    with pytest.raises(SystemExit):
        deploy_stent.main(_REQUIRED + extra_args)


# ── Backend selection ────────────────────────────────────────────────────────

@pytest.mark.parametrize("script", [deploy_stent, deploy_stent_straighten])
def test_backend_flag_is_optional_and_restricted(script):
    parser = script.build_parser()
    assert parser.parse_args(_REQUIRED).backend is None  # the environment variable or "auto" decides
    assert parser.parse_args(_REQUIRED + ["--backend", "jax"]).backend == "jax"
    with pytest.raises(SystemExit):
        parser.parse_args(_REQUIRED + ["--backend", "torch"])


def test_backend_flag_selects_the_backend():
    backend.set_backend(None)
    common.setup_logging(deploy_stent.build_parser().parse_args(_REQUIRED + ["--backend", "jax"]))
    assert backend.requested_backend() == "jax"


def test_no_backend_flag_leaves_the_choice_alone(monkeypatch):
    monkeypatch.delenv(backend.ENVIRONMENT_VARIABLE, raising=False)
    backend.set_backend(None)
    common.setup_logging(deploy_stent.build_parser().parse_args(_REQUIRED))
    assert backend.requested_backend() == "auto"


def test_warp_backend_flag_without_warp_exits_with_the_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", None)  # as if warp-lang were not installed
    args = deploy_stent.build_parser().parse_args(_REQUIRED + ["--backend", "warp"])
    with pytest.raises(SystemExit, match=r"svmorph\[warp\]"):
        common.setup_logging(args)


@pytest.mark.parametrize("value", ["wrap", "warp"], ids=["a typo", "warp without Warp"])
def test_unusable_backend_in_the_environment_exits_before_any_work(monkeypatch, value):
    # Without a --backend flag the environment variable decides; a value that cannot be honoured
    # ends a stent script at startup, as the flag does, not with a traceback at its first step
    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, value)
    backend.set_backend(None)
    with pytest.raises(SystemExit) as exit_info:
        common.setup_logging(deploy_stent.build_parser().parse_args(_REQUIRED))
    assert str(exit_info.value.code).startswith("error: ")


def test_scripts_without_a_backend_flag_ignore_the_stent_backend(monkeypatch):
    # The aneurysm and stenosis scripts share setup_logging and never deploy a stent
    import argparse

    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, "wrap")
    backend.set_backend(None)
    common.setup_logging(argparse.Namespace(units="cm", verbose=False, debug=False))


# ── A deployment, end to end, on a synthetic vessel ──────────────────────────

VESSEL_RADIUS, VESSEL_LENGTH, START_POINT = 0.25, 4.0, 150


def _read_points(path) -> np.ndarray:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).copy()


def _write(polydata, path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    assert writer.Write() == 1


@pytest.fixture
def vessel(tmp_path):
    """A straight tube along z and its centerline, written as the .vtp pair the scripts read."""
    axis = vtk.vtkLineSource()
    axis.SetPoint1(0.0, 0.0, 0.0)
    axis.SetPoint2(0.0, 0.0, VESSEL_LENGTH)
    axis.SetResolution(100)
    tube = vtk.vtkTubeFilter()
    tube.SetInputConnection(axis.GetOutputPort())
    tube.SetRadius(VESSEL_RADIUS)
    tube.SetNumberOfSides(40)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(tube.GetOutputPort())
    triangles.Update()
    _write(triangles.GetOutput(), tmp_path / "surface.vtp")

    centerline = vtk.vtkLineSource()
    centerline.SetPoint1(0.0, 0.0, 0.0)
    centerline.SetPoint2(0.0, 0.0, VESSEL_LENGTH)
    centerline.SetResolution(200)
    centerline.Update()
    polyline = centerline.GetOutput()
    radii = numpy_to_vtk(np.full(polyline.GetNumberOfPoints(), VESSEL_RADIUS), deep=True)
    radii.SetName("MaximumInscribedSphereRadius")
    polyline.GetPointData().AddArray(radii)
    _write(polyline, tmp_path / "centerline.vtp")
    return tmp_path


def _deploy(script, vessel, name: str, *extra_args) -> tuple:
    """Run *script* on the synthetic vessel and return the deployed surface and centerline points."""
    surface_path, centerline_path = vessel / f"{name}_surface.vtp", vessel / f"{name}_centerline.vtp"
    script.main([
        "--mesh", str(vessel / "surface.vtp"), "--cline", str(vessel / "centerline.vtp"),
        "--start", str(START_POINT), "--target-R", "0.4",
        "--out-mesh", str(surface_path), "--out-cl", str(centerline_path), *extra_args])
    return _read_points(surface_path), _read_points(centerline_path)


def _radial_distance(points: np.ndarray) -> np.ndarray:
    return np.hypot(points[:, 0], points[:, 1])


@pytest.mark.parametrize("script, extra_args", [
    (deploy_stent, []),
    (deploy_stent, ["--flare-R", "0.5", "--flare-length", "0.4", "--cap-height-fraction", "0.35"]),
    (deploy_stent_straighten, []),
], ids=["uniform", "flared", "straightening"])
def test_deployment_expands_the_vessel_to_the_stent_radius(vessel, script, extra_args):
    original = _read_points(vessel / "surface.vtp")
    surface, centerline = _deploy(script, vessel, "deployed", "--backend", "jax", *extra_args)

    assert surface.shape == original.shape
    assert centerline.shape == _read_points(vessel / "centerline.vtp").shape
    # The stent runs 1.53 proximally from the start point at z = 3: beside its middle the wall has
    # been pushed out from the vessel radius to the stent radius, and far from it nothing moved
    beside_stent = np.abs(original[:, 2] - 2.2) < 0.3
    far_from_stent = original[:, 2] < 0.3
    np.testing.assert_allclose(_radial_distance(original[beside_stent]), VESSEL_RADIUS, atol=1e-3)
    np.testing.assert_allclose(_radial_distance(surface[beside_stent]), 0.4, atol=0.02)
    np.testing.assert_array_equal(surface[far_from_stent], original[far_from_stent])


@pytest.mark.parametrize("script, extra_args", [
    (deploy_stent, []),
    (deploy_stent, ["--flare-R", "0.5", "--flare-length", "0.4", "--cap-height-fraction", "0.35"]),
    (deploy_stent_straighten, []),
], ids=["uniform", "flared", "straightening"])
def test_warp_deployment_matches_the_jax_deployment(vessel, script, extra_args):
    pytest.importorskip("warp", reason="the Warp backend needs warp-lang (pip install 'svmorph[warp]')")
    expected_surface, expected_centerline = _deploy(script, vessel, "jax", "--backend", "jax", *extra_args)
    surface, centerline = _deploy(script, vessel, "warp", "--backend", "warp", *extra_args)
    assert np.abs(_radial_distance(expected_surface) - VESSEL_RADIUS).max() > 0.1
    np.testing.assert_allclose(surface, expected_surface, atol=1e-4)
    np.testing.assert_allclose(centerline, expected_centerline, atol=1e-4)


def test_snapshots_are_written_from_up_to_date_points(vessel):
    # Under Warp the mesh lives on the device between steps; a snapshot must be of the mesh as
    # deployed so far, not of the host arrays as they were when the deployment began
    pytest.importorskip("warp", reason="the Warp backend needs warp-lang (pip install 'svmorph[warp]')")
    original = _read_points(vessel / "surface.vtp")
    _deploy(deploy_stent, vessel, "jax", "--backend", "jax", "--save-step", "0.1")
    _deploy(deploy_stent, vessel, "warp", "--backend", "warp", "--save-step", "0.1")

    jax_snapshots = sorted((vessel / "jax_surface_intermediates").glob("jax_surface_*.vtp"))
    warp_snapshots = sorted((vessel / "warp_surface_intermediates").glob("warp_surface_*.vtp"))
    assert len(jax_snapshots) == len(warp_snapshots) == 3
    for jax_snapshot, warp_snapshot in zip(jax_snapshots, warp_snapshots):
        np.testing.assert_allclose(_read_points(warp_snapshot), _read_points(jax_snapshot), atol=1e-4)
    # The last snapshot is taken after the stent has reached the wall
    assert np.abs(_read_points(warp_snapshots[-1]) - original).max() > 0.05


def test_snapshot_manager_calls_before_save_only_when_it_writes(vessel):
    surface_pd = vtk.vtkXMLPolyDataReader()
    surface_pd.SetFileName(str(vessel / "surface.vtp"))
    surface_pd.Update()
    calls = []
    snapshots = common.SnapshotManager(
        start_value=0.05, target_value=0.4, save_step=0.1, out_mesh_path=str(vessel / "out.vtp"),
        out_cl_path=None, surface_pd=surface_pd.GetOutput(), centerline_pd=None, data={},
        before_save=lambda: calls.append(1))
    snapshots.check_and_save(0.10)
    assert calls == []
    snapshots.check_and_save(0.16)
    assert calls == [1]
    snapshots.check_and_save(0.17)
    assert calls == [1]
