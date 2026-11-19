"""Synthetic geometry and reference results shared by the test modules."""

from __future__ import annotations

import numpy as np

from svmorph.core import backend, deformation


def make_tube(radius: float = 0.5, length: float = 4.0, number_of_angles: int = 48,
              number_of_rings: int = 81, number_of_centerline_points: int = 41,
              dtype=np.float32) -> dict:
    """Simulation data for a straight tube along z with its centerline on the axis."""
    angle, z = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, number_of_angles, endpoint=False),
                           np.linspace(0.0, length, number_of_rings))
    surface = np.column_stack([radius * np.cos(angle.ravel()), radius * np.sin(angle.ravel()), z.ravel()])
    centerline_z = np.linspace(0.0, length, number_of_centerline_points)
    centerline = np.column_stack([np.zeros_like(centerline_z), np.zeros_like(centerline_z), centerline_z])
    return {
        "points": {"surface": surface.astype(dtype), "centerline": centerline.astype(dtype)},
        "nodes": {"force_center_point_id": number_of_centerline_points // 2},
    }


def make_stent_axis(start: float = 1.0, end: float = 3.0, number_of_vertices: int = 21,
                    wobble: float = 0.02) -> np.ndarray:
    """A stent axis along z inside :func:`make_tube`, bent slightly so that no symmetry hides an error."""
    z = np.linspace(start, end, number_of_vertices)
    return np.column_stack([wobble * np.sin(3.0 * z), np.zeros_like(z), z])


def copy_data(data: dict) -> dict:
    """A copy of simulation *data* whose point arrays can be displaced independently."""
    return {"points": {name: points.copy() for name, points in data["points"].items()},
            "nodes": dict(data["nodes"])}


def with_jax_reference(function, *args, **kwargs):
    """Call *function* under the JAX reference backend, then put the backend in use back."""
    name, device = backend.get_backend(), backend.warp_device()
    backend.set_backend("jax")
    try:
        return function(*args, **kwargs)
    finally:
        backend.set_backend(name, device=device)


def reference_sdf(points, stent_vertices, radius, cap_height_fraction: float = 1.0):
    """Distance ``(N,)`` and direction ``(N, 3)`` from the JAX reference, on the code path a step would take."""
    if np.ndim(radius) == 0 and cap_height_fraction == 1.0:
        distance, direction = deformation.smin_sdf_capsule_contact_sculpt(points, stent_vertices, radius)
    else:
        distance, direction = deformation.smin_sdf_variable_radius_capsule_contact_sculpt(
            points, stent_vertices, radius, cap_height_fraction)
    return np.asarray(distance)[:, 0], np.asarray(direction)


def without_borderline_points(data: dict, stent_vertices, radius, cap_height_fraction: float = 1.0,
                              contact_distance: float = 0.001, margin: float = 1e-4) -> dict:
    """Return *data* without the points whose role in a step hangs on the last digits of the SDF.

    Two implementations agree on the stent SDF to about 1e-6.  A surface point
    whose distance is that close to the contact threshold, or a centerline
    point that close to the stent surface, may therefore be classified
    differently by them - a legitimate difference, but one that changes
    blending weights by far more than rounding.  Parity tests compare steps on
    geometry that has no such points.
    """
    surface, centerline = data["points"]["surface"], data["points"]["centerline"]
    surface_distance, _ = reference_sdf(surface, stent_vertices, radius, cap_height_fraction)
    centerline_distance, _ = reference_sdf(centerline, stent_vertices, radius, cap_height_fraction)
    filtered = copy_data(data)
    filtered["points"]["surface"] = np.ascontiguousarray(
        surface[np.abs(surface_distance - contact_distance) > margin])
    filtered["points"]["centerline"] = np.ascontiguousarray(centerline[np.abs(centerline_distance) > margin])
    return filtered
