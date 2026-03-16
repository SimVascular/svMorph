"""Kelvinlet contraction (stenosis creation) script.

Shrinks the vessel wall at a chosen centerline point using the truncated
spherical contraction kernel.  Only the surface mesh is displaced; the
centerline is left unchanged.

Usage::

    python -m svmorph.scripts.create_stenosis \\
        --mesh surface.vtp --cline centerline.vtp \\
        --center 789 --target-R 0.15 --stenosis-length 0.5 \\
        --force-scale 1.0 --save-step 0.02 \\
        --out-mesh stenosis_surface.vtp
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from svmorph.core import deformation, mesh_data
from svmorph.core.units import L
from svmorph.logging import get_logger
from svmorph.scripts import common
from svmorph.visualization import vtk_io

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a stenosis via Kelvinlet contraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_common_args(parser)
    parser.add_argument(
        "--center", type=int, required=True,
        help="Centerline point ID for stenosis center",
    )
    parser.add_argument("--target-R", type=float, default=None, help="Target lumen radius (default: 0.15 cm)")
    parser.add_argument(
        "--stenosis-length", type=float, default=None,
        help="Outer annular cutoff r_max (default: 0.5 cm)",
    )
    parser.add_argument(
        "--force-scale", type=float, default=1.0,
        help="Signed force scale (positive -> inward)",
    )
    parser.set_defaults(out_mesh="stenosis_surface.vtp")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common.setup_logging(args)

    if args.target_R is None:
        args.target_R = 0.15 * L()
    if args.stenosis_length is None:
        args.stenosis_length = 0.5 * L()

    logger.info("Loading simulation data...")
    ctx = common.load_simulation_data(args.mesh, args.cline)
    deformation.set_node_indices(ctx.data, [args.center])
    deformation.set_force_center(ctx.data, args.center)

    _, normal = vtk_io.get_centerline_point_and_normal(ctx.centerline_pd, args.center)

    representative, current_R = common.find_radius_representative(ctx, args.center)
    logger.info(f"Initial radius = {current_R:.4f}, target = {args.target_R:.4f}")

    snapshots = common.SnapshotManager(
        start_value=current_R,
        target_value=args.target_R,
        save_step=args.save_step,
        out_mesh_path=args.out_mesh,
        out_cl_path=None,
        surface_pd=ctx.surface_pd,
        centerline_pd=None,
        data=ctx.data,
    )

    iteration = 0
    t0 = time.time()
    while current_R > args.target_R + 2e-3 * L():
        surf_disp, step = deformation.compute_stenosis_displacements(
            ctx.data, args.force_scale, normal, args.target_R, args.stenosis_length,
        )
        mesh_data.apply_displacements(ctx.data, surf_disp, "surface")

        rep_point = ctx.data["points"]["surface"][representative]
        center_point = ctx.data["points"]["centerline"][args.center]
        current_R = float(np.linalg.norm(rep_point - center_point))

        iteration += 1
        logger.info(f"Step {iteration:3d}: R = {current_R:.5f}")
        snapshots.check_and_save(current_R)

    elapsed = time.time() - t0
    logger.info(f"Stenosis creation complete: {iteration} steps in {elapsed:.2f} s")
    common.save_final(ctx, args)


if __name__ == "__main__":
    main()
