"""Shared utilities for headless deformation scripts.

Provides reusable building blocks -- data loading, snapshot management,
CLI helpers -- so each script stays slim.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import vtk

from svmorph.core import deformation
from svmorph.logging import get_logger, setup_logging as _setup_logging, TIMING
from svmorph.visualization import vtk_io

logger = get_logger(__name__)


@dataclass
class SimulationContext:
    """Bundles all loaded simulation data for headless scripts."""

    data: dict
    parent_tip_map: dict[int, int]
    segment_base_mask: np.ndarray
    tangents: np.ndarray
    inscribed_sphere_radii: np.ndarray
    surface_pd: vtk.vtkPolyData
    centerline_pd: vtk.vtkPolyData


def load_simulation_data(mesh_path: str, cline_path: str) -> SimulationContext:
    """Read VTP files and assemble a :class:`SimulationContext`.

    Parameters
    ----------
    mesh_path : str
        Path to the surface mesh ``.vtp`` file.
    cline_path : str
        Path to the centerline ``.vtp`` file.
    """
    surface_pd = vtk_io.read_vtp(mesh_path)
    centerline_pd = vtk_io.read_vtp(cline_path)
    data = vtk_io.extract_mesh_arrays(surface_pd, centerline_pd)
    parent_tip_map, segment_base_mask = vtk_io.build_parent_tip_map(centerline_pd)
    tangents = vtk_io.extract_centerline_tangents(centerline_pd)
    inscribed_sphere_radii = vtk_io.extract_inscribed_sphere_radii(centerline_pd)
    return SimulationContext(
        data=data,
        parent_tip_map=parent_tip_map,
        segment_base_mask=segment_base_mask,
        tangents=tangents,
        inscribed_sphere_radii=inscribed_sphere_radii,
        surface_pd=surface_pd,
        centerline_pd=centerline_pd,
    )


def find_radius_representative(
    ctx: SimulationContext, center_id: int,
) -> tuple[np.ndarray, float]:
    """Identify the surface point that represents the vessel radius at *center_id*.

    Wraps :func:`deformation.find_stenosis_minimum_radius_representative` with
    the geometry setup boilerplate shared by aneurysm and stenosis scripts.

    Returns
    -------
    representative : np.ndarray
        Index (1-element array) into the surface point array.
    current_R : float
        Estimated current vessel radius.
    """
    data_points = ctx.data["points"]["surface"]
    centerline_points = ctx.data["points"]["centerline"]
    query_points = np.expand_dims(data_points, 1)
    query_points = np.tile(query_points, (1, 1, 1))
    centers = np.expand_dims(np.array([centerline_points[center_id]]), 0)
    normal = ctx.tangents[center_id]
    rotation_matrices = np.array(
        deformation.compute_householder_matrices(np.array([normal]))
    )
    original_radius = ctx.inscribed_sphere_radii[center_id]
    representative, current_R = deformation.find_stenosis_minimum_radius_representative(
        data_points, rotation_matrices, query_points, centers, original_radius,
    )
    return representative, current_R


class SnapshotManager:
    """Milestone-based intermediate saving for deformation loops.

    Works for both increasing (expansion/stent) and decreasing (contraction)
    radius tracking.  Creates a ``{out_mesh_stem}_intermediates/`` directory.

    Parameters
    ----------
    start_value, target_value : float
        Initial and target radius values used to determine milestones.
    save_step : float or None
        Interval between milestones.  ``None`` disables snapshots.
    out_mesh_path : str
        Base output mesh filename (used for snapshot naming).
    out_cl_path : str or None
        Base output centerline filename, or ``None`` to skip centerline saves.
    surface_pd, centerline_pd : vtk.vtkPolyData or None
        Polydata objects to write.
    data : dict
        Simulation data dict (for ``sync_polydata``).
    """

    def __init__(
        self,
        start_value: float,
        target_value: float,
        save_step: float | None,
        out_mesh_path: str,
        out_cl_path: str | None,
        surface_pd: vtk.vtkPolyData,
        centerline_pd: vtk.vtkPolyData | None,
        data: dict,
    ):
        self._surface_pd = surface_pd
        self._centerline_pd = centerline_pd
        self._data = data

        increasing = target_value > start_value

        mesh_stem = pathlib.Path(out_mesh_path).with_suffix("")
        self._snapshot_dir = pathlib.Path(f"{mesh_stem}_intermediates")
        self._mesh_prefix = self._snapshot_dir / pathlib.Path(out_mesh_path).stem
        self._cl_prefix = (
            self._snapshot_dir / pathlib.Path(out_cl_path).stem
            if out_cl_path
            else None
        )

        if not save_step or save_step <= 0:
            self._milestones = np.array([])
        elif increasing:
            self._milestones = np.arange(start_value + save_step, target_value, save_step)
        else:
            self._milestones = np.arange(start_value - save_step, target_value, -save_step)

        self._increasing = increasing
        self._next_idx = 0

    def check_and_save(self, current_value: float) -> None:
        """Write a snapshot if *current_value* has crossed the next milestone."""
        if self._next_idx >= len(self._milestones):
            return
        milestone = self._milestones[self._next_idx]
        crossed = (
            current_value >= milestone
            if self._increasing
            else current_value <= milestone
        )
        if crossed:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._write(current_value)
            self._next_idx += 1

    def _write(self, value: float) -> None:
        suffix = f"_{value:.3f}.vtp"
        vtk_io.sync_polydata(self._surface_pd, self._data, "surface")
        mesh_path = self._mesh_prefix.with_name(self._mesh_prefix.name + suffix)
        vtk_io.write_vtp(self._surface_pd, str(mesh_path))
        logger.info(f"Snapshot @ {value:.3f} -> {mesh_path}")
        if self._centerline_pd is not None and self._cl_prefix is not None:
            vtk_io.sync_polydata(self._centerline_pd, self._data, "centerline")
            cl_path = self._cl_prefix.with_name(self._cl_prefix.name + suffix)
            vtk_io.write_vtp(self._centerline_pd, str(cl_path))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI flags to *parser*."""
    parser.add_argument("--mesh", required=True, help="Input surface .vtp")
    parser.add_argument("--cline", required=True, help="Input centerline .vtp")
    parser.add_argument("--out-mesh", default="output_surface.vtp", help="Output surface mesh")
    parser.add_argument("--out-cl", default=None, help="Output centerline (optional)")
    parser.add_argument(
        "--save-step", type=float, default=None,
        help="Write intermediate snapshots every this many cm of radius change (omit to disable)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable TIMING-level logging")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")


def setup_logging(args: argparse.Namespace) -> None:
    """Configure svmorph logging from parsed CLI args."""
    if getattr(args, "debug", False):
        level = logging.DEBUG
    elif getattr(args, "verbose", False):
        level = TIMING
    else:
        level = logging.INFO
    _setup_logging(level)


def save_final(ctx: SimulationContext, args: argparse.Namespace) -> None:
    """Sync polydata and write final output files."""
    vtk_io.sync_polydata(ctx.surface_pd, ctx.data, "surface")
    vtk_io.write_vtp(ctx.surface_pd, args.out_mesh)
    logger.info(f"Saved surface -> {args.out_mesh}")
    if getattr(args, "out_cl", None):
        vtk_io.sync_polydata(ctx.centerline_pd, ctx.data, "centerline")
        vtk_io.write_vtp(ctx.centerline_pd, args.out_cl)
        logger.info(f"Saved centerline -> {args.out_cl}")
