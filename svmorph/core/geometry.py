"""Geometric utilities for centerline resampling and stent-axis construction.

Provides arc-length–based polyline resampling that supports branching
centerlines via a parent-tip map, used to generate evenly spaced stent
axis vertices along a selected centerline segment.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from svmorph.logging import get_logger

logger = get_logger(__name__)


def resample_stent_axis(
    points: np.ndarray,
    parent_tip_map: dict[int, int],
    segment_base_mask: np.ndarray,
    starting_point_idx: int,
    desired_total_length: float,
    desired_segment_length: float,
    sampling_direction: int = -1,
) -> jax.Array:
    """Extract and resample a polyline subsegment at uniform arc-length intervals.

    Walks along the centerline from *starting_point_idx* in the given
    *sampling_direction*, accumulating arc length up to *desired_total_length*.
    At branch bases the walk jumps to the parent segment via *parent_tip_map*.
    The collected vertices are then resampled at intervals of
    *desired_segment_length* using linear interpolation.

    Parameters
    ----------
    points : np.ndarray
        Centerline point coordinates, shape ``(N, 3)``.
    parent_tip_map : dict[int, int]
        Mapping from each point ID to the tip ID of its parent segment.
    segment_base_mask : np.ndarray
        Boolean mask marking the base point of each centerline segment.
    starting_point_idx : int
        Index of the centerline point at which to start walking.
    desired_total_length : float
        Target arc length of the extracted subsegment (cm).
    desired_segment_length : float
        Desired spacing between resampled vertices (cm).
    sampling_direction : int
        Walk direction: ``-1`` for proximal, ``+1`` for distal.

    Returns
    -------
    jax.Array
        Resampled stent axis vertices, shape ``(M, 3)``.
    """
    if sampling_direction not in (-1, 1):
        raise ValueError("sampling_direction must be integer -1 or +1")
    if len(points) < 2:
        raise ValueError("Not enough points to form a polyline.")

    diffs_all = np.diff(points, axis=0)
    distances_all = np.linalg.norm(diffs_all, axis=1)

    subsegment_points = []
    subsegment_s = []  # Track exact cumulative arclength at each vertex

    subsegment_points.append(points[starting_point_idx])
    subsegment_s.append(0.0)

    cumulative_length = 0.0
    n_points = len(points)

    # Walk along the polyline starting from starting_point_idx.
    i = starting_point_idx
    idx_end = 0 if sampling_direction == -1 else n_points - 1
    next_point_idx_offset = -1 if sampling_direction == -1 else 0

    while i != idx_end:
        if segment_base_mask[i]:
            next_i = parent_tip_map[i]
            d = np.linalg.norm(points[next_i] - points[i])
        else:
            next_i = i + sampling_direction
            d = distances_all[i + next_point_idx_offset]

        # If adding the full segment would exceed desired_total_length,
        # interpolate along this segment to hit the target exactly.
        if cumulative_length + d < desired_total_length:
            cumulative_length += d
            subsegment_points.append(points[next_i])
            subsegment_s.append(cumulative_length)
        else:
            remaining = desired_total_length - cumulative_length
            t = remaining / d
            new_point = (1 - t) * points[i] + t * points[next_i]
            subsegment_points.append(new_point)

            # Force the final tracked length to be EXACTLY the desired length
            cumulative_length = desired_total_length
            subsegment_s.append(desired_total_length)
            break  # desired total length achieved, exit loop

        i = next_i

    effective_total_length = cumulative_length
    if effective_total_length < desired_total_length:
        logger.warning(f"Subsegment truncated due to jump/end. Best achieved length = {effective_total_length:.4f} cm")

    # Convert lists to numpy arrays for vectorized interpolation
    subsegment_points = np.array(subsegment_points)
    subsegment_s = np.array(subsegment_s)

    # Generate new exact arc-length intervals
    new_s = np.arange(0, effective_total_length, desired_segment_length)

    # Ensure the final point is strictly included and exact
    if len(new_s) == 0 or not np.isclose(new_s[-1], effective_total_length):
        new_s = np.append(new_s, effective_total_length)

    # Interpolate x, y, and z coordinates independently using the exact tracked arclengths
    new_vertices = np.zeros((len(new_s), 3))
    for dim in range(3):
        new_vertices[:, dim] = np.interp(new_s, subsegment_s, subsegment_points[:, dim])

    return jnp.array(new_vertices)


def _normalized_arc_positions(stent_vertices: np.ndarray) -> np.ndarray:
    """Arc-length position of each stent axis vertex, normalized to [0, 1] from the first vertex."""
    points = np.asarray(stent_vertices, dtype=float)
    if len(points) < 2:
        return np.zeros(len(points))
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = arc_positions[-1]
    if total_length <= 0.0:
        return np.zeros(len(points))
    return arc_positions / total_length


def stent_radius_profile(stent_vertices: np.ndarray, control_points) -> np.ndarray:
    """Per-vertex stent radii from arbitrary (position, radius) control points.

    Builds a variable radius profile for the tapered capsule-chain stent SDF
    (see :func:`svmorph.core.deformation.compute_sdf_contact_displacements`):
    one radius per stent axis vertex, linearly interpolated between the
    control points along the stent axis.

    Parameters
    ----------
    stent_vertices : np.ndarray
        Stent axis vertices, shape ``(V, 3)`` (e.g. from
        :func:`resample_stent_axis`).
    control_points : iterable of (float, float)
        ``(position, radius)`` pairs, where *position* is the normalized
        arc-length position in ``[0, 1]`` measured from the first axis
        vertex.

    Returns
    -------
    np.ndarray
        Per-vertex stent radii, shape ``(V,)``.
    """
    control_points = sorted((float(position), float(radius)) for position, radius in control_points)
    if not control_points:
        raise ValueError("At least one radius profile control point is required")
    positions = [position for position, _ in control_points]
    radii = [radius for _, radius in control_points]
    return np.interp(_normalized_arc_positions(stent_vertices), positions, radii)


def flared_stent_radius_profile(
    stent_vertices: np.ndarray, body_radius: float, flare_radius: float,
    flare_length: float, flare_at_axis_start: bool = False,
) -> np.ndarray:
    """Per-vertex stent radii for a stent with one flared (funnel/trumpet) end.

    The radius transitions from *body_radius* to *flare_radius* over
    *flare_length* at one end of the stent axis with a smoothstep profile.
    The result can be passed (scaled to the current deployment radius) as the
    per-vertex ``current_stent_radius`` of
    :func:`svmorph.core.deformation.compute_sdf_contact_displacements`.

    Parameters
    ----------
    stent_vertices : np.ndarray
        Stent axis vertices, shape ``(V, 3)``.
    body_radius : float
        Stent radius away from the flared end.
    flare_radius : float
        Stent radius at the tip of the flared end (may also be smaller than
        *body_radius* for a tapered stent).
    flare_length : float
        Length of the radius transition, measured along the stent axis from
        the flared end, in the same unit as the vertex coordinates.
    flare_at_axis_start : bool
        Flare the first-vertex end of the stent axis instead of the
        last-vertex end.

    Returns
    -------
    np.ndarray
        Per-vertex stent radii, shape ``(V,)``.
    """
    points = np.asarray(stent_vertices, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = arc_positions[-1]
    distance_from_flared_end = arc_positions if flare_at_axis_start else total_length - arc_positions
    t = np.clip(1.0 - distance_from_flared_end / float(flare_length), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep
    return float(body_radius) + (float(flare_radius) - float(body_radius)) * t
