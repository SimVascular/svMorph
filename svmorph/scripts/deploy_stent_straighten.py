"""SDF-contact stent deployment with concurrent axis straightening.

Identical to :mod:`deploy_stent` but after each displacement step the stent
axis vertices are projected toward the line connecting the first and last
vertices, gradually straightening the deployed stent.

Usage::

    python -m svmorph.scripts.deploy_stent_straight \\
        --mesh surface.vtp --cline centerline.vtp \\
        --start 123 --target-R 0.4 --start-R 0.05 \\
        --length 3.0 --straightening-strength 0.075 \\
        --out-mesh deployed_surface.vtp --out-cl deployed_centerline.vtp
"""

from __future__ import annotations

import argparse
import time

import jax.numpy as jnp

from svmorph.core import deformation, geometry, mesh_data
from svmorph.core.units import L
from svmorph.logging import get_logger
from svmorph.scripts import common

logger = get_logger(__name__)


def straighten_axis(vertices: jnp.ndarray, strength: float) -> jnp.ndarray:
    """Move each vertex toward the line connecting the first and last vertex.

    Parameters
    ----------
    vertices : jnp.ndarray
        Stent axis vertices, shape ``(M, 3)``.
    strength : float
        Fraction of the displacement toward the line applied per call.
    """
    start = vertices[0]
    end = vertices[-1]
    direction = end - start
    length = jnp.linalg.norm(direction)
    if length < 1e-6:
        return vertices
    direction = direction / length
    for i in range(len(vertices)):
        point = vertices[i]
        proj_len = jnp.dot(point - start, direction)
        closest = start + proj_len * direction
        vertices = vertices.at[i].set(point + strength * (closest - point))
    return vertices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy a crimped stent with concurrent axis straightening.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_common_args(parser)
    parser.add_argument(
        "--start", type=int, required=True,
        help="Centerline point ID for stent distal tip",
    )
    parser.add_argument("--target-R", type=float, default=None, help="Target deployed stent radius (default: 0.4 cm)")
    parser.add_argument("--start-R", type=float, default=None, help="Initial crimped stent radius (default: 0.05 cm)")
    parser.add_argument("--length", type=float, default=None, help="Stent length along centerline (default: 1.7 cm)")
    parser.add_argument(
        "--straightening-strength", type=float, default=0.075,
        help="Per-step fraction of displacement toward the straight line",
    )
    parser.set_defaults(out_mesh="deployed_surface.vtp", out_cl="deployed_centerline.vtp")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common.setup_logging(args)

    if args.target_R is None:
        args.target_R = 0.4 * L()
    if args.start_R is None:
        args.start_R = 0.05 * L()
    if args.length is None:
        args.length = 1.7 * L()

    smoothing_k = 0.01 * L()

    logger.info("Loading simulation data...")
    ctx = common.load_simulation_data(args.mesh, args.cline)
    deformation.set_node_indices(ctx.data, [args.start])
    deformation.set_force_center(ctx.data, args.start)

    foreshortening = 0.1
    deployed_length = args.length * (1 - foreshortening)
    axis_pts = geometry.resample_stent_axis(
        ctx.data["points"]["centerline_points_view_np"],
        ctx.parent_tip_map,
        ctx.segment_base_mask,
        args.start,
        deployed_length,
        0.1 * L(),
        sampling_direction=-1,
    )
    logger.info(f"Stent axis: {len(axis_pts)} vertices over {deployed_length:.2f}")

    a, b = mesh_data.compute_material_constants(1.0, 0.2)

    snapshots = common.SnapshotManager(
        start_value=args.start_R,
        target_value=args.target_R,
        save_step=args.save_step,
        out_mesh_path=args.out_mesh,
        out_cl_path=args.out_cl,
        surface_pd=ctx.surface_pd,
        centerline_pd=ctx.centerline_pd,
        data=ctx.data,
    )

    cur_R = args.start_R - smoothing_k
    logger.info(f"Starting R = {args.start_R:.4f}, target R = {args.target_R:.4f}")

    iteration = 0
    t0 = time.time()
    while True:
        surf_disp, cl_disp, dR = deformation.compute_sdf_contact_displacements(
            ctx.data,
            axis_pts,
            s=-1.0,
            target_stent_radius=args.target_R,
            current_stent_radius=cur_R,
        )
        if cur_R + dR > args.target_R:
            logger.info("Next increment would overshoot target -- done.")
            break
        mesh_data.apply_displacements(ctx.data, surf_disp, "surface")
        mesh_data.apply_displacements(ctx.data, cl_disp, "centerline")
        cur_R += dR
        axis_pts = straighten_axis(axis_pts, args.straightening_strength)
        iteration += 1
        displayed_R = cur_R + smoothing_k
        logger.info(f"Step {iteration:3d}: dR={dR:.5f}  R={displayed_R:.5f}")
        snapshots.check_and_save(displayed_R)

    elapsed = time.time() - t0
    logger.info(f"Deployment complete: {iteration} steps in {elapsed:.2f} s")
    common.save_final(ctx, args)


if __name__ == "__main__":
    main()
