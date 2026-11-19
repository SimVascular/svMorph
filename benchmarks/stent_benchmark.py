"""Time SDFStent deployment on each backend.

Two things are measured for every mesh and backend:

``step``
    One call of ``compute_sdf_contact_displacements`` with the stent in
    contact with the wall: what the GUI computes per frame.  Milliseconds,
    median of the repeats.
``deployment``
    A whole deployment through ``StentDeployment``, from the crimped radius to
    the target radius and including the final ``sync()``: what the stent
    scripts run.  Milliseconds, best of the repeats.

Compilation is excluded from both: JAX and Warp each compile their kernels the
first time they run, which a warm-up pays for before the clock starts.

Usage::

    # Synthetic tubes, every backend that is available here
    python benchmarks/stent_benchmark.py

    # Chosen mesh sizes and backends
    python benchmarks/stent_benchmark.py --sizes 13000,200000 --backends jax,warp:cuda:0

    # Your own geometry (same arguments as svmorph-deploy-stent)
    python benchmarks/stent_benchmark.py --mesh surface.vtp --cline centerline.vtp --start 123

A backend is written ``jax``, ``warp`` (Warp's default device) or
``warp:<device>``, e.g. ``warp:cuda:0`` or ``warp:cpu``.
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from svmorph.core import backend, deformation, geometry
from svmorph.core.stent_deployment import StentDeployment
from svmorph.core.units import L, set_unit_scale

# The tube is a vessel of this radius; a stent deploys in the middle of it
TUBE_RADIUS_CM = 0.25


def synthetic_tube(number_of_points: int, length: float) -> tuple[dict, np.ndarray]:
    """A straight tube of about *number_of_points* surface points, and a stent axis in the middle of it."""
    radius = TUBE_RADIUS_CM * L()
    circumference = 2.0 * np.pi * radius
    number_of_angles = max(8, int(round(np.sqrt(number_of_points * circumference / length))))
    number_of_rings = max(2, int(round(number_of_points / number_of_angles)))
    angle, z = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, number_of_angles, endpoint=False),
                           np.linspace(0.0, length, number_of_rings))
    surface = np.column_stack([radius * np.cos(angle.ravel()), radius * np.sin(angle.ravel()), z.ravel()])
    centerline_z = np.linspace(0.0, length, 401)
    centerline = np.column_stack([np.zeros_like(centerline_z), np.zeros_like(centerline_z), centerline_z])
    data = {"points": {"surface": surface.astype(np.float32), "centerline": centerline.astype(np.float32)},
            "nodes": {"force_center_point_id": 200}}
    return data, centerline.astype(np.float32)


def stent_axis_in_tube(length: float, stent_length: float) -> np.ndarray:
    """A slightly curved stent axis centred in the tube, sampled like ``resample_stent_axis`` samples it."""
    number_of_vertices = int(round(stent_length / (0.1 * L()))) + 1
    z = np.linspace(0.5 * (length - stent_length), 0.5 * (length + stent_length), number_of_vertices)
    return np.column_stack([0.01 * L() * np.sin(3.0 * z / L()), np.zeros_like(z), z])


def load_geometry(mesh_path: str, cline_path: str, start: int, stent_length: float) -> tuple[dict, np.ndarray]:
    """A surface/centerline pair from .vtp files and the stent axis the deployment script would build."""
    from svmorph.visualization import vtk_io

    surface_pd, centerline_pd = vtk_io.read_vtp(mesh_path), vtk_io.read_vtp(cline_path)
    data = vtk_io.extract_mesh_arrays(surface_pd, centerline_pd)
    parent_tip_map, segment_base_mask = vtk_io.build_parent_tip_map(centerline_pd)
    axis = geometry.resample_stent_axis(
        data["points"]["centerline_points_view_np"], parent_tip_map, segment_base_mask, start,
        stent_length, 0.1 * L(), sampling_direction=-1)
    # Copies: the benchmark deforms the points many times over
    data = {"points": {"surface": np.array(data["points"]["surface"]),
                       "centerline": np.array(data["points"]["centerline"])},
            "nodes": {"force_center_point_id": start}}
    return data, np.asarray(axis)


def copy_of(data: dict) -> dict:
    return {"points": {name: points.copy() for name, points in data["points"].items()},
            "nodes": dict(data["nodes"])}


def select_backend(name: str) -> str:
    """Select the backend written as ``jax``, ``warp`` or ``warp:<device>``; return a label for it."""
    kind, _, device = name.partition(":")
    backend.set_backend(kind, device=device or None)
    if kind == "jax":
        return "JAX (CPU)"
    return f"Warp ({backend.warp_device()})"


def deploy(data: dict, axis: np.ndarray, start_radius: float, target_radius: float) -> int:
    """The loop of ``scripts/deploy_stent.py`` for a uniform stent; returns the number of steps."""
    smoothing_k = 0.01 * L()
    session = StentDeployment(data, target_stent_radius=target_radius)
    step = session.step_size()
    radius = start_radius - smoothing_k
    steps = 0
    while radius + step <= target_radius:
        session.step(axis, radius)
        radius += step
        steps += 1
    session.sync()
    return steps


def radius_in_contact(data: dict, axis: np.ndarray, start_radius: float, target_radius: float) -> float:
    """A stent radius at which the stent presses on the wall: the first step that moves a point."""
    backend.set_backend("jax")
    radius = start_radius
    while radius < target_radius:
        surface, _, step = deformation.compute_sdf_contact_displacements(data, axis, -1.0, target_radius, radius)
        if np.abs(np.asarray(surface)).max() > 0.0:
            return radius + 2.0 * step
        radius += step
    raise SystemExit("The stent never touches the wall: choose a larger --target-R or another site")


def time_step(data: dict, axis: np.ndarray, radius: float, target_radius: float, repeats: int) -> float:
    for _ in range(3):
        deformation.compute_sdf_contact_displacements(data, axis, -1.0, target_radius, radius)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        deformation.compute_sdf_contact_displacements(data, axis, -1.0, target_radius, radius)
        samples.append(time.perf_counter() - start)
    return 1000.0 * float(np.median(samples))


def time_deployment(data: dict, axis: np.ndarray, start_radius: float, target_radius: float,
                    repeats: int) -> tuple[float, int]:
    deploy(copy_of(data), axis, start_radius, target_radius)  # compiles for the shapes of this mesh
    best, steps = np.inf, 0
    for _ in range(repeats):
        fresh = copy_of(data)
        start = time.perf_counter()
        steps = deploy(fresh, axis, start_radius, target_radius)
        best = min(best, time.perf_counter() - start)
    return 1000.0 * best, steps


def hardware() -> str:
    parts = [platform.processor() or platform.machine(), platform.system()]
    if "warp" in backend.available_backends():
        import warp as wp
        if wp.is_cuda_available():
            parts.append(wp.get_device("cuda:0").name)
        parts.append(f"warp-lang {wp.__version__}")
    import jax
    parts.append(f"jax {jax.__version__}")
    return ", ".join(parts)


def default_backends() -> list[str]:
    names = ["jax"]
    if "warp" in backend.available_backends():
        import warp as wp
        names.append("warp:cpu")
        if wp.is_cuda_available():
            names.append("warp:cuda:0")
    return names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="13000,50000,200000,1000000",
                        help="Surface point counts of the synthetic tubes (default: %(default)s)")
    parser.add_argument("--tube-length", type=float, default=None,
                        help="Length of the synthetic tubes (default: 12 cm, of which the stent's "
                             "neighbourhood is about a third)")
    parser.add_argument("--mesh", help="Surface .vtp to benchmark on instead of synthetic tubes")
    parser.add_argument("--cline", help="Centerline .vtp that goes with --mesh")
    parser.add_argument("--start", type=int, help="Centerline point ID of the stent's distal tip")
    parser.add_argument("--target-R", type=float, default=None, help="Target stent radius (default: 0.4 cm)")
    parser.add_argument("--start-R", type=float, default=None, help="Crimped stent radius (default: 0.05 cm)")
    parser.add_argument("--length", type=float, default=None, help="Stent length (default: 1.7 cm)")
    parser.add_argument("--units", choices=("cm", "mm"), default="cm")
    parser.add_argument("--backends", default=None,
                        help="Comma-separated backends: jax, warp, warp:<device> (default: all available)")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats of each measurement (default: 5)")
    args = parser.parse_args(argv)
    if (args.mesh is None) != (args.cline is None) or (args.mesh is not None and args.start is None):
        parser.error("--mesh, --cline and --start go together")

    set_unit_scale({"cm": 1.0, "mm": 10.0}[args.units])
    target_radius = args.target_R if args.target_R is not None else 0.4 * L()
    start_radius = args.start_R if args.start_R is not None else 0.05 * L()
    stent_length = (args.length if args.length is not None else 1.7 * L()) * 0.9  # foreshortened
    backends = args.backends.split(",") if args.backends else default_backends()

    meshes = []
    if args.mesh is not None:
        meshes.append(load_geometry(args.mesh, args.cline, args.start, stent_length))
    else:
        tube_length = args.tube_length if args.tube_length is not None else 12.0 * L()
        axis = stent_axis_in_tube(tube_length, stent_length)
        for size in args.sizes.split(","):
            meshes.append((synthetic_tube(int(size), tube_length)[0], axis))

    print(f"Hardware: {hardware()}")
    print(f"Stent: {start_radius:g} -> {target_radius:g} {args.units}, {stent_length:g} {args.units} long\n")
    labels, rows = [], []
    for data, axis in meshes:
        contact_radius = radius_in_contact(data, axis, start_radius, target_radius)
        row = []
        for name in backends:
            label = select_backend(name)
            if label not in labels:
                labels.append(label)
            step_ms = time_step(data, axis, contact_radius, target_radius, args.repeats * 4)
            deployment_ms, steps = time_deployment(data, axis, start_radius, target_radius, args.repeats)
            row.append((step_ms, deployment_ms, steps))
        box_min, box_max = axis.min(axis=0), axis.max(axis=0)
        padding = target_radius + 0.65 * L() + 0.011 * L()
        surface = data["points"]["surface"]
        in_box = np.all((surface >= box_min - padding) & (surface <= box_max + padding), axis=1).mean()
        rows.append((len(surface), in_box, row))

    for title, index in (("One step with the stent in contact (ms)", 0),
                         (f"Whole deployment of {rows[0][2][0][2]} steps (ms)", 1)):
        print(f"{title}\n")
        print("| Surface points | Near the stent | " + " | ".join(labels) + " | Fastest vs JAX |")
        print("|---:|---:|" + "---:|" * (len(labels) + 1))
        for number_of_points, in_box, row in rows:
            values = [entry[index] for entry in row]
            speedup = f"{values[0] / min(values):,.0f}x" if labels[0].startswith("JAX") else "-"
            print(f"| {number_of_points:,} | {100 * in_box:.0f}% | "
                  + " | ".join(f"{value:,.2f}" for value in values) + f" | {speedup} |")
        print()


if __name__ == "__main__":
    main()
