"""Kelvinlet-based deformation kernels and SDF-contact displacement computation.

Implements regularized Kelvinlet force responses, smooth-minimum SDF
evaluation against capsule stent geometries, and top-level routines that
assemble per-vertex displacement fields for aneurysm inflation, stenosis
creation, and SDF-contact stent deployment.  All heavy numerics use JAX
for automatic vectorization and JIT compilation.

This module is the reference implementation.  The stent deployment step can
also be computed by NVIDIA Warp kernels on a GPU (:mod:`svmorph.core.sdf_warp`),
chosen through :mod:`svmorph.core.backend`;
:func:`compute_sdf_contact_displacements` is where that choice takes effect.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

import jax as jx
import jax.numpy as jnp

from svmorph.core import backend
from svmorph.core.units import L
from svmorph.logging import get_logger

logger = get_logger(__name__)

def compute_householder_matrices(cross_section_normals: jx.Array) -> jx.Array:
    """Compute Householder reflection matrices that rotate each normal to the z-axis.

    Parameters
    ----------
    cross_section_normals : jx.Array
        Unit normals for each cross-section, shape ``(N, 3)``.

    Returns
    -------
    jx.Array
        Stack of 3x3 Householder matrices, shape ``(N, 3, 3)``.
    """
    z_axis = jnp.array([0, 0, 1])
    a_minus_b = cross_section_normals - z_axis
    denominator = jnp.linalg.norm(a_minus_b, axis=1) ** 2
    projection_matrix = 2 * jnp.einsum('ij,ik->ijk', a_minus_b, a_minus_b) / denominator[:, None, None]
    householder_matrices = jnp.eye(3) - projection_matrix
    return householder_matrices

def set_node_indices(data: dict, list_of_node_point_indices: list[int]) -> dict:
    """Store the selected centerline node indices in the simulation data dict.

    Parameters
    ----------
    data : dict
        Simulation data dictionary.
    list_of_node_point_indices : list[int]
        Indices of the selected centerline nodes.

    Returns
    -------
    dict
        Updated simulation data dictionary.
    """
    data["nodes"]["all_indices"] = jnp.array(list_of_node_point_indices)
    return data

def set_force_center(data: dict, point_id: int) -> dict:
    """Set the force-center point ID in the simulation data dict.

    Parameters
    ----------
    data : dict
        Simulation data dictionary.
    point_id : int
        Centerline point index to use as the force center.

    Returns
    -------
    dict
        Updated simulation data dictionary.
    """
    data["nodes"]["force_center_point_id"] = point_id
    return data

def mix(a: jx.Array, b: jx.Array, t: jx.Array) -> jx.Array:
    """Linearly interpolate between *a* and *b* by factor *t*.

    Parameters
    ----------
    a, b : jx.Array
        Endpoint values.
    t : jx.Array
        Interpolation factor(s); 0 gives *a*, 1 gives *b*.

    Returns
    -------
    jx.Array
        Interpolated result, same shape as the inputs.
    """
    return a + (b - a) * t

def smin_and_gradient(
    a: jx.Array, da: jx.Array, b: jx.Array, db: jx.Array, k: float | None = None,
) -> tuple[jx.Array, jx.Array]:
    """Smooth-minimum of two scalar fields with gradient blending.

    Uses polynomial smooth-min (k-smin) to approximate ``min(a, b)``
    while producing a C¹-continuous transition and a correspondingly
    blended gradient from *da* and *db*.

    Parameters
    ----------
    a, b : jx.Array
        Scalar distance fields.
    da, db : jx.Array
        Gradients (or direction vectors) associated with *a* and *b*.
    k : float or None
        Smoothing radius.  ``None`` (default) uses ``0.01 * L()``.

    Returns
    -------
    value : jx.Array
        Smooth minimum of *a* and *b*.
    grad : jx.Array
        Blended gradient.
    """
    if k is None:
        k = 0.01 * L()
    k = k * 4.0 # a quirk from the math in the paper
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    n = 0.5 * h
    m = h**2 * k / 4.0
    # Use jnp.where to choose between the two cases in a jittable way
    value = jnp.where(a < b, a - m, b - m)
    grad  = jnp.where(a < b, mix(da, db, n), mix(da, db, 1.0 - n))
    return value, grad

def fold_smin(
    carry: tuple[jx.Array, jx.Array], elem: tuple[jx.Array, jx.Array],
) -> tuple[tuple[jx.Array, jx.Array], None]:
    """``jax.lax.scan``-compatible fold step that accumulates smooth-min distance and direction.

    Parameters
    ----------
    carry : tuple[jx.Array, jx.Array]
        Running ``(min_distance, min_direction)`` accumulator.
    elem : tuple[jx.Array, jx.Array]
        Next ``(distance, direction)`` element.

    Returns
    -------
    tuple[tuple[jx.Array, jx.Array], None]
        Updated carry and a ``None`` scan placeholder.
    """
    cur_min_d, cur_min_dir = carry
    d, direction = elem
    new_d, new_dir = smin_and_gradient(cur_min_d, cur_min_dir, d, direction)
    return (new_d, new_dir), None

def compute_min_dist_and_direction(d: jx.Array, direction: jx.Array) -> tuple[jx.Array, jx.Array]:
    """Reduce per-segment distances and directions to a single smooth-minimum pair.

    Parameters
    ----------
    d : jx.Array
        Per-segment signed distances, shape ``(num_segments,)``.
    direction : jx.Array
        Per-segment direction vectors, shape ``(num_segments, 3)``.

    Returns
    -------
    final_d : jx.Array
        Smooth-minimum distance (scalar).
    final_dir : jx.Array
        Blended direction vector, shape ``(3,)``.
    """
    (final_d, final_dir), _ = jx.lax.scan(fold_smin, (d[0], direction[0]), (d[1:], direction[1:]))
    return final_d, final_dir

@jx.jit
def capsule_sdf(p: jx.Array, stent_vertices: jx.Array, r: float) -> jx.Array:
    """Evaluate the signed distance field of a capsule-chain stent.

    Each consecutive pair of *stent_vertices* defines a capsule segment
    with radius *r*.  The SDF is reduced via smooth-minimum so the
    iso-surface is C¹-continuous at segment junctions.

    Parameters
    ----------
    p : jx.Array
        Query points, shape ``(N, 3)``.
    stent_vertices : jx.Array
        Stent axis vertices, shape ``(V, 3)``.
    r : float
        Capsule radius.

    Returns
    -------
    jx.Array
        Signed distance for each query point, shape ``(N, 1)``.
    """
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = p[:, None, :] - stent_vertices[None, :-1, :]
    ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
    ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
    h_all = jnp.clip(ba_dot_pa_all / ba_dot_ba_all, 0, 1)
    axis_to_point_all = pa_all - h_all[:, :, None] * ba_all[None, :, :]
    dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
    direction_all = axis_to_point_all / dist_all
    dist_all_squeezed = jnp.squeeze(dist_all, axis=-1)  # shape: (num_mesh_points, num_segments)
    dist_to_surface_all = dist_all_squeezed - r
    final_dist_to_surface, _ = jx.vmap(compute_min_dist_and_direction)(dist_to_surface_all, direction_all)
    final_dist_to_surface = final_dist_to_surface[:, None]
    return final_dist_to_surface

def kelvinlets_truncated_spherical_contraction(
    rv: jx.Array, f_scale: float,
    s: float, r_min: float, r_max: float,
) -> jx.Array:
    """Compute inward radial displacements for stenosis creation.

    Points within the cylindrical annulus [*r_min*, *r_max*] are displaced
    radially inward using a quartic bump profile, with the axial component
    zeroed to prevent longitudinal drift.

    Parameters
    ----------
    rv : jx.Array
        Centerline-aligned relative positions, shape ``(N, 3)``.
    f_scale : float
        Force magnitude scaling factor.
    s : float
        Signed force scale (negative → inward).
    r_min : float
        Inner annular radius cutoff.
    r_max : float
        Outer annular radius cutoff.

    Returns
    -------
    jx.Array
        Per-point displacement vectors, shape ``(N, 3)``.
    """
    rx, ry, rz = rv[:, 0], rv[:, 1], rv[:, 2]
    re = jnp.sqrt(rx**2 + ry**2 + rz**2)
    re_no_z = jnp.sqrt(rx**2 + ry**2)
    inner_mask = (re_no_z >= r_min).astype(int)
    outer_mask = (re <= r_max).astype(int)
    rv = rv.at[:, 2].set(0 * rv[:, 2])
    displacements = f_scale * (r_max - r_min) * ((re[:, None] / r_max) ** 2 - 1) ** 2 * (-s) * rv
    displacements = displacements * inner_mask[:, None] * outer_mask[:, None]
    return displacements

def kelvinlets_truncated_spherical_expansion(
    rv: jx.Array, a: float, b: float, eps: float, s: float,
) -> jx.Array:
    """Compute outward radial displacements for aneurysm sculpting.

    Uses the scaling regularized Kelvinlet (F = s·I, Eq. 14 in de Goes &
    James 2017) to push surface points radially outward from the
    centerline, with the axial component zeroed.

    Parameters
    ----------
    rv : jx.Array
        Centerline-aligned relative positions, shape ``(N, 3)``.
    a, b : float
        Kelvinlet material parameters.
    eps : float
        Regularization parameter.
    s : float
        Signed force scale.

    Returns
    -------
    jx.Array
        Per-point displacement vectors, shape ``(N, 3)``.
    """
    f_scale = 0.01 * L()**3
    rx, ry, rz = rv[:, 0], rv[:, 1], rv[:, 2]
    re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    re3 = re[:, None]**3
    re5 = re[:, None]**5
    rv = rv.at[:, 2].set(0 * rv[:, 2])
    displacements = f_scale * (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv

    return displacements

@jx.jit
def smin_sdf_capsule_contact_sculpt(
    rv: jx.Array, stent_vertices: jx.Array,
    r_current: float,
) -> tuple[jx.Array, jx.Array]:
    """Compute smooth-min SDF distances and outward directions from a capsule-chain stent.

    Evaluates the signed distance from each query point to the nearest
    capsule segment surface (using smooth-minimum for C¹ continuity)
    and returns the distance and outward direction for SDF-contact
    displacement computation.

    Parameters
    ----------
    rv : jx.Array
        Query points, shape ``(N, 3)``.
    stent_vertices : jx.Array
        Stent axis vertices, shape ``(V, 3)``.
    r_current : float
        Current stent deployment radius.

    Returns
    -------
    final_dist_to_surface : jx.Array
        Signed distance to stent surface, shape ``(N, 1)``.
    final_direction : jx.Array
        Unit outward direction from the stent axis, shape ``(N, 3)``.
    """
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = rv[:, None, :] - stent_vertices[None, :-1, :]
    ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
    ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
    h_all = jnp.clip(ba_dot_pa_all / ba_dot_ba_all, 0, 1)
    axis_to_point_all = pa_all - h_all[:, :, None] * ba_all[None, :, :]
    dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
    direction_all = axis_to_point_all / dist_all
    dist_all_squeezed = jnp.squeeze(dist_all, axis=-1)  # shape: (num_mesh_points, num_segments)
    dist_to_surface_all = dist_all_squeezed - r_current
    # Vectorize the folding over all mesh points:
    final_dist_to_surface, final_direction = jx.vmap(compute_min_dist_and_direction)(dist_to_surface_all, direction_all)
    final_dist_to_surface = final_dist_to_surface[:, None]
    return final_dist_to_surface, final_direction

def _radius_profile_slopes(arc_positions: jx.Array, radii: jx.Array) -> jx.Array:
    """Monotone-cubic (PCHIP) slopes of the stent radius profile at the axis vertices.

    Interior slopes are the weighted harmonic mean of the two adjacent
    secant slopes, or zero where the secants vanish or change sign (local
    extremum), so the interpolated profile never overshoots the per-vertex
    radii.  End slopes are the secants of the terminal segments.

    Parameters
    ----------
    arc_positions : jx.Array
        Arc-length position of each stent axis vertex, shape ``(V,)``.
    radii : jx.Array
        Per-vertex stent radii, shape ``(V,)``.

    Returns
    -------
    jx.Array
        Radius slope (radius change per unit arc length) at each vertex,
        shape ``(V,)``.
    """
    segment_lengths = jnp.diff(arc_positions)
    secants = jnp.diff(radii) / segment_lengths
    weight_prev = 2.0 * segment_lengths[1:] + segment_lengths[:-1]
    weight_next = segment_lengths[1:] + 2.0 * segment_lengths[:-1]
    same_sign = secants[:-1] * secants[1:] > 0.0
    secants_prev = jnp.where(same_sign, secants[:-1], 1.0)
    secants_next = jnp.where(same_sign, secants[1:], 1.0)
    interior_slopes = jnp.where(
        same_sign, (weight_prev + weight_next) / (weight_prev / secants_prev + weight_next / secants_next), 0.0)
    return jnp.concatenate([secants[:1], interior_slopes, secants[-1:]])

def _radius_at_arc_position(arc_position: jx.Array, arc_positions: jx.Array, radii: jx.Array) -> jx.Array:
    """Evaluate the stent radius profile at arc-length positions along the stent axis.

    The per-vertex *radii* are interpolated with a monotone cubic (see
    :func:`_radius_profile_slopes`), so the profile passes through every
    vertex radius, has no creases at the vertices, and does not overshoot.

    Parameters
    ----------
    arc_position : jx.Array
        Arc-length positions to evaluate, within ``[0, arc_positions[-1]]``,
        shape ``(N,)``.
    arc_positions : jx.Array
        Arc-length position of each stent axis vertex, shape ``(V,)``.
    radii : jx.Array
        Per-vertex stent radii, shape ``(V,)``.

    Returns
    -------
    jx.Array
        Stent radius at each position, shape ``(N,)``.
    """
    slopes = _radius_profile_slopes(arc_positions, radii)
    i = jnp.clip(jnp.searchsorted(arc_positions, arc_position, side="right") - 1, 0, arc_positions.shape[0] - 2)
    segment_length = arc_positions[i + 1] - arc_positions[i]
    t = (arc_position - arc_positions[i]) / segment_length
    return (radii[i] + (radii[i + 1] - radii[i]) * (t * t * (3.0 - 2.0 * t))
            + segment_length * (slopes[i] * (t * (t - 1.0) ** 2) + slopes[i + 1] * (t * t * (t - 1.0))))

def _variable_radius_capsule_chain_sdf(
    p: jx.Array, stent_vertices: jx.Array, r: float | jx.Array, cap_height_fraction: float,
    with_direction: bool,
) -> tuple[jx.Array, jx.Array | None]:
    """Smooth-min signed distance (and outward direction) of a capsule-chain stent with a radius profile.

    Variable-radius counterpart of the constant-radius capsule chain
    (:func:`capsule_sdf`, :func:`smin_sdf_capsule_contact_sculpt`), kept as
    a separate code path so the uniform stent is evaluated exactly as
    before.  The radius is given per stent axis vertex and describes a
    profile along the stent axis (e.g. a flared/trumpet end, a taper, or a
    local narrowing).

    The distance is assembled in three steps:

    1. *Distance to the axis.*  The smooth minimum over the segments is taken
       of the distance to each axis segment, exactly as for a uniform
       radius.  The same fold also blends the arc-length coordinate at which
       every segment sees the query point along (the extension of) its own
       direction.
    2. *Radius profile.*  The stent radius at that arc-length coordinate is
       subtracted after the smooth minimum.  Subtracting per-segment radii
       inside the smooth minimum instead lets the spherical joint cap of a
       wide segment bulge into its narrower neighbors, and flattening the
       joint caps to prevent that confines the smooth-min blending to the
       joints, which shows up as a ripple with the period of the segment
       length along the whole stent.
    3. *End caps.*  An arc-length coordinate beyond either end of the axis is
       the axial overshoot past that stent end.  There the distance to the
       end vertex is measured with the overshoot scaled by the reciprocal of
       *cap_height_fraction*, which turns the spherical end cap into a half
       ellipsoid whose axial semi-axis is ``cap_height_fraction`` times the
       end radius.  The result is divided by its gradient magnitude so it
       stays a first-order distance, and is floored by the spherical-cap
       distance.  This keeps a flared stent end from inflating a ball into
       the vessel beyond the stent end.  Only the two stent ends are
       affected; the joints between segments are not.

    The returned direction is the blended axis-to-point direction in all
    cases, so the cap shape changes where the stent surface is but not the
    direction in which the vessel wall is pushed away from it.

    Parameters
    ----------
    p : jx.Array
        Query points, shape ``(N, 3)``.
    stent_vertices : jx.Array
        Stent axis vertices, shape ``(V, 3)``.
    r : float or jx.Array
        Stent radius: per-vertex radii of shape ``(V,)``, or a scalar.
    cap_height_fraction : float
        Axial semi-axis of the two stent end caps as a fraction of the end
        radius, in ``(0, 1]``.
    with_direction : bool
        Whether to also compute the outward direction.

    Returns
    -------
    dist_to_surface : jx.Array
        Signed distance to the stent surface, shape ``(N,)``.
    direction : jx.Array or None
        Outward direction from the stent axis, shape ``(N, 3)``, or ``None``
        when *with_direction* is false.
    """
    radii = jnp.broadcast_to(jnp.asarray(r), (stent_vertices.shape[0],))
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = p[:, None, :] - stent_vertices[None, :-1, :]
    ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
    ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
    h_unclamped_all = ba_dot_pa_all / ba_dot_ba_all
    h_all = jnp.clip(h_unclamped_all, 0, 1)
    axis_to_point_all = pa_all - h_all[:, :, None] * ba_all[None, :, :]
    dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
    dist_all_squeezed = jnp.squeeze(dist_all, axis=-1)  # shape: (num_mesh_points, num_segments)

    # Arc-length coordinate of each query point as seen by each segment (not clamped to the
    # segment, so all segments of a straight stretch agree on it)
    segment_lengths = jnp.sqrt(ba_dot_ba_all)
    arc_positions = jnp.concatenate([jnp.zeros(1, dtype=segment_lengths.dtype), jnp.cumsum(segment_lengths)])
    arc_position_all = (arc_positions[None, :-1] + h_unclamped_all * segment_lengths[None, :])[..., None]
    if with_direction:
        direction_all = axis_to_point_all / dist_all
        blended_all = jnp.concatenate([direction_all, arc_position_all], axis=-1)
    else:
        blended_all = arc_position_all

    # Vectorize the folding over all mesh points:
    axis_dist, blended = jx.vmap(compute_min_dist_and_direction)(dist_all_squeezed, blended_all)
    arc_position = blended[:, -1]
    arc_position_on_stent = jnp.clip(arc_position, 0.0, arc_positions[-1])
    dist_to_surface = axis_dist - _radius_at_arc_position(arc_position_on_stent, arc_positions, radii)

    # End caps: stretch the axial overshoot past the stent ends (zero along the stent body)
    cap_overshoot_ratio = jnp.clip((arc_position - arc_position_on_stent) / jnp.maximum(axis_dist, 1e-12), -1.0, 1.0)
    cap_stretch = 1.0 / cap_height_fraction**2 - 1.0
    stretched_dist_scale = jnp.sqrt(1.0 + cap_stretch * cap_overshoot_ratio**2)
    stretched_gradient_scale = jnp.sqrt(1.0 + cap_stretch * (cap_stretch + 2.0) * cap_overshoot_ratio**2)
    flattened_cap_dist = ((dist_to_surface + axis_dist * (stretched_dist_scale - 1.0))
                          * stretched_dist_scale / stretched_gradient_scale)
    # The flattened cap lies inside the spherical one, so it is never closer than the latter
    dist_to_surface = jnp.maximum(flattened_cap_dist, dist_to_surface)

    direction = blended[:, :3] if with_direction else None
    return dist_to_surface, direction

@jx.jit
def variable_radius_capsule_sdf(
    p: jx.Array, stent_vertices: jx.Array, r: float | jx.Array,
    cap_height_fraction: float = 1.0,
) -> jx.Array:
    """Evaluate the signed distance field of a capsule-chain stent with a radius profile.

    Variable-radius counterpart of :func:`capsule_sdf`; see
    :func:`_variable_radius_capsule_chain_sdf` for the formulation.

    Parameters
    ----------
    p : jx.Array
        Query points, shape ``(N, 3)``.
    stent_vertices : jx.Array
        Stent axis vertices, shape ``(V, 3)``.
    r : float or jx.Array
        Stent radius: per-vertex radii of shape ``(V,)`` that define a
        radius profile along the stent axis (e.g. a flared end), or a
        scalar.
    cap_height_fraction : float
        Axial semi-axis of the two stent end caps as a fraction of the end
        radius; ``1.0`` (default) gives spherical caps, smaller values
        flatten them into half ellipsoids.

    Returns
    -------
    jx.Array
        Signed distance for each query point, shape ``(N, 1)``.
    """
    dist_to_surface, _ = _variable_radius_capsule_chain_sdf(
        p, stent_vertices, r, cap_height_fraction, with_direction=False)
    return dist_to_surface[:, None]

@jx.jit
def smin_sdf_variable_radius_capsule_contact_sculpt(
    rv: jx.Array, stent_vertices: jx.Array,
    r_current: float | jx.Array,
    cap_height_fraction: float = 1.0,
) -> tuple[jx.Array, jx.Array]:
    """Compute smooth-min SDF distances and outward directions from a capsule-chain stent with a radius profile.

    Variable-radius counterpart of :func:`smin_sdf_capsule_contact_sculpt`;
    see :func:`_variable_radius_capsule_chain_sdf` for the formulation.

    Parameters
    ----------
    rv : jx.Array
        Query points, shape ``(N, 3)``.
    stent_vertices : jx.Array
        Stent axis vertices, shape ``(V, 3)``.
    r_current : float or jx.Array
        Current stent deployment radius: per-vertex radii of shape ``(V,)``
        that define a radius profile along the stent axis (e.g. a flared
        end), or a scalar.
    cap_height_fraction : float
        Axial semi-axis of the two stent end caps as a fraction of the end
        radius; ``1.0`` (default) gives spherical caps, smaller values
        flatten them into half ellipsoids.

    Returns
    -------
    final_dist_to_surface : jx.Array
        Signed distance to stent surface, shape ``(N, 1)``.
    final_direction : jx.Array
        Outward direction from the stent axis, shape ``(N, 3)``.
    """
    final_dist_to_surface, final_direction = _variable_radius_capsule_chain_sdf(
        rv, stent_vertices, r_current, cap_height_fraction, with_direction=True)
    final_dist_to_surface = final_dist_to_surface[:, None]
    return final_dist_to_surface, final_direction

@jx.jit
def get_scaling_kelvinlet_displacements_inner(
    data_points: jx.Array, rotation_matrices: jx.Array,
    centers: jx.Array, a: float, b: float, eps: float, s: float,
    surface_mesh_scale_factor: float | None,
) -> jx.Array:
    """JIT-compiled inner loop for scaling Kelvinlet displacement computation.

    Rotates mesh points into the center's local frame, computes sculpt
    displacements via the scaling Kelvinlet kernel (F = s·I), and rotates
    back to global coordinates.

    Parameters
    ----------
    data_points : jx.Array
        Surface mesh vertices, shape ``(N, 3)``.
    rotation_matrices : jx.Array
        Householder matrices, shape ``(1, 3, 3)``.
    centers : jx.Array
        Kelvinlet force center, shape ``(1, 3)``.
    a, b : float
        Kelvinlet material parameters.
    eps : float
        Regularization parameter.
    s : float
        Signed force scale.
    surface_mesh_scale_factor : float | None
        Optional global displacement scaling.

    Returns
    -------
    displacement : jx.Array
        Net displacement per mesh point, shape ``(N, 3)``.
    """
    rv = data_points - centers
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    displacement_local = kelvinlets_truncated_spherical_expansion(centerline_aligned_rv, a, b, eps, s)
    displacement = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
    return displacement

def find_stenosis_minimum_radius_representative(
    data_points: np.ndarray, rotation_matrices: np.ndarray,
    centers: np.ndarray, original_radius: float,
) -> tuple[np.intp, float]:
    """Find the surface point that best represents the minimum vessel radius.

    Among surface points on the annular shell between 1.0× and 1.1× the
    estimated current radius, selects the one with the smallest axial
    offset from the cross-section plane.  The current radius is estimated
    from Euclidean nearest neighbours projected onto the plane.

    Parameters
    ----------
    data_points : np.ndarray
        Surface mesh vertices, shape ``(N, 3)``.
    rotation_matrices : np.ndarray
        Householder matrices, shape ``(1, 3, 3)``.
    centers : np.ndarray
        Kelvinlet center, shape ``(1, 3)``.
    original_radius : float
        Maximum inscribed sphere radius at this centerline point.

    Returns
    -------
    index_for_min_rz : np.intp
        Index into *data_points* of the representative surface point.
    current_radius : float
        Estimated current vessel radius at this cross-section.
    """
    rv = data_points - centers
    rv = np.einsum('...ij,...j->...i', rotation_matrices, rv)
    rx, ry, rz = rv[:, 0], rv[:, 1], rv[:, 2]
    rz_magnitude = np.abs(rz)
    radial_magnitude_squared = rx**2 + ry**2

    # Estimate the actual current vessel radius:
    # 1. Euclidean nearest neighbors → local surface points
    # 2. Smallest |rz| among those → points on the cross-section plane
    # Their radial distances give the current vessel radius, even after
    # deformation.  This naturally excludes outlet cap points (which have
    # nonzero |rz|) unless the center is exactly on the cap.
    euclidean_dist_sq = rx**2 + ry**2 + rz**2
    nearest_dist = np.sqrt(np.min(euclidean_dist_sq))
    nearby = euclidean_dist_sq < (nearest_dist * 1.5)**2
    nearby_rz_mag = np.abs(rz[nearby])
    nearby_radial = np.sqrt(radial_magnitude_squared[nearby])
    rz_cutoff = np.percentile(nearby_rz_mag, 5)
    on_plane = nearby_rz_mag <= rz_cutoff
    if np.any(on_plane):
        current_radius = float(np.median(nearby_radial[on_plane]))
    else:
        current_radius = float(original_radius)

    mask = ((current_radius * 1.0) ** 2 <= radial_magnitude_squared) * (radial_magnitude_squared <= (current_radius * 1.1) ** 2)
    rz_magnitude_masked = np.where(mask, rz_magnitude, np.inf)
    index_for_min_rz = np.argmin(rz_magnitude_masked)
    logger.debug(f"Estimated current radius: {current_radius}")
    return index_for_min_rz, current_radius

@jx.jit
def get_stenosis_displacements_inner(
    data_points: jx.Array, rotation_matrices: jx.Array,
    centers: jx.Array, s: float,
    r_min: float, r_max: float,
) -> tuple[jx.Array, jx.Array]:
    """JIT-compiled inner loop for stenosis (inward shrink) displacements.

    Rotates mesh points into the local frame, applies the truncated-sphere
    warp, and rotates back to global coordinates.

    Parameters
    ----------
    data_points : jx.Array
        Surface mesh vertices, shape ``(N, 3)``.
    rotation_matrices : jx.Array
        Householder matrices, shape ``(1, 3, 3)``.
    centers : jx.Array
        Kelvinlet center, shape ``(1, 3)``.
    s : float
        Signed force scale.
    r_min : float
        Inner annular radius cutoff.
    r_max : float
        Outer annular radius cutoff.

    Returns
    -------
    displacement : jx.Array
        Per-point displacement vectors, shape ``(N, 3)``.
    step_size : jx.Array
        Scalar step size for radius tracking.
    """
    rv = data_points - centers
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    f_scale = 0.01 / (r_max - r_min)
    step_size = f_scale * (r_max - r_min) * s
    displacement_local = kelvinlets_truncated_spherical_contraction(centerline_aligned_rv, f_scale, s, r_min, r_max)
    displacement = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    return displacement, step_size

def compute_aneurysm_displacements(
    data: dict, a: float, b: float, eps: float, s: float,
    surface_mesh_scale_factor: float | None,     force_center_normal: jx.Array,
) -> np.ndarray:
    """Compute Kelvinlet-based outward surface displacements for aneurysm creation.

    Assembles query-point geometry, builds Householder rotation matrices,
    and delegates to the JIT-compiled scaling Kelvinlet inner kernel.

    Parameters
    ----------
    data : dict
        Simulation data dictionary with surface and centerline arrays.
    a, b : float
        Kelvinlet material parameters.
    eps : float
        Regularization parameter (pre-scaled by vessel radius).
    s : float
        Signed force scale.
    surface_mesh_scale_factor : float | None
        Optional displacement scaling.
    force_center_normal : jx.Array
        Tangent direction at the force center, shape ``(3,)``.

    Returns
    -------
    displacements : np.ndarray
        Per-vertex displacement vectors, shape ``(N, 3)``.
    """
    force_center_point_id = data["nodes"]["force_center_point_id"]
    logger.debug(f"Selected pointId: {force_center_point_id}")
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    centers = jnp.array([centerline_points[force_center_point_id]])
    rotation_matrices = compute_householder_matrices(jnp.array([force_center_normal]))
    displacements = get_scaling_kelvinlet_displacements_inner(
        data_points, rotation_matrices, centers, a, b, eps, s, surface_mesh_scale_factor
    )
    return np.array(displacements)

def compute_stenosis_displacements(
    data: dict, s: float,
    force_center_normal: jx.Array, r_min: float, r_max: float,
) -> tuple[np.ndarray, jx.Array]:
    """Compute inward surface displacements for stenosis creation.

    Assembles query-point geometry and delegates to the JIT-compiled
    stenosis displacement kernel.

    Parameters
    ----------
    data : dict
        Simulation data dictionary.
    s : float
        Signed force scale (positive → inward).
    force_center_normal : jx.Array
        Tangent direction at the force center, shape ``(3,)``.
    r_min : float
        Inner annular radius.
    r_max : float
        Outer annular radius.

    Returns
    -------
    displacements : np.ndarray
        Per-vertex displacement vectors, shape ``(N, 3)``.
    step_size : jx.Array
        Scalar step size for radius tracking.
    """
    force_center_point_id = data["nodes"]["force_center_point_id"]
    logger.debug(f"Selected pointId: {force_center_point_id}")
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    centers = jnp.array([centerline_points[force_center_point_id]])
    rotation_matrices = compute_householder_matrices(jnp.array([force_center_normal]))
    displacements, step_size = get_stenosis_displacements_inner(
        data_points, rotation_matrices, centers, s, r_min, r_max
    )
    return np.array(displacements), step_size

@jx.jit
def stent_bounding_box(
    data_points: jx.Array, stent_vertices: jx.Array,
    target_stent_radius: float, influence_radius: float, contact_distance: float,
) -> jx.Array:
    """Compute a boolean mask selecting points inside the padded stent bounding box.

    Parameters
    ----------
    data_points : jx.Array
        Mesh vertices, shape ``(N, 3)``.
    stent_vertices : jx.Array
        Stent axis vertices, shape ``(V, 3)``.
    target_stent_radius : float
        Target stent radius.
    influence_radius : float
        Additional radial padding for the influence zone.
    contact_distance : float
        Additional padding for the contact threshold.

    Returns
    -------
    jx.Array
        Boolean mask of length *N*.
    """
    min_coords = jnp.min(stent_vertices, axis=0) - target_stent_radius - influence_radius - contact_distance - 0.01 * L()
    max_coords = jnp.max(stent_vertices, axis=0) + target_stent_radius + influence_radius + contact_distance + 0.01 * L()
    mask = jnp.all((data_points >= min_coords) & (data_points <= max_coords), axis=1)
    return mask

def _sdf_contact_defaults(
    influence_radius: float | None, contact_distance: float | None, f_scale: float | None,
) -> tuple[float, float, float]:
    """Fill in the unit-scaled defaults of :func:`compute_sdf_contact_displacements`."""
    if influence_radius is None:
        influence_radius = 0.65 * L()
    if contact_distance is None:
        contact_distance = 0.001 * L()
    if f_scale is None:
        f_scale = 0.01 * L()
    return influence_radius, contact_distance, f_scale

def _warp_step_parameters(
    s: float, target_stent_radius: float | jx.Array,
    influence_radius: float, contact_distance: float, f_scale: float, cap_height_fraction: float,
) -> dict:
    """Translate the arguments of a deployment step into those of the Warp kernels.

    The kernels take every length explicitly, so the values that the JAX
    kernels read from :func:`svmorph.core.units.L` are spelled out here.
    """
    if np.ndim(target_stent_radius) > 0:
        target_stent_radius = float(np.max(target_stent_radius))
    return {
        # as stent_bounding_box pads the box
        "box_padding": target_stent_radius + influence_radius + contact_distance + 0.01 * L(),
        "cap_height_fraction": cap_height_fraction,
        # the smoothing radius smin_and_gradient uses when it is given none
        "smoothing": 0.01 * L(),
        "contact_distance": contact_distance,
        "influence_radius": influence_radius,
        "magnitude_scale": f_scale * (-s),
    }

def compute_sdf_contact_displacements(
    data: dict, stent_vertices: jx.Array,
    s: float,
    target_stent_radius: float | jx.Array, current_stent_radius: float | jx.Array, *,
    influence_radius: float | None = None, contact_distance: float | None = None,
    f_scale: float | None = None, cap_height_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute SDF-contact displacements for stent deployment.

    Main entry point for SDF-contact deformation.  The algorithm:

    1. Restricts evaluation to a bounding-box mask around the stent.
    2. Evaluates smooth-minimum capsule SDF for surface and centerline
       points.
    3. Identifies in-contact and in-influence point sets.
    4. Builds a KD-tree of contact points for influence-zone blending.
    5. Assembles displacement vectors that push the vessel wall outward.

    A uniform stent (scalar *current_stent_radius*, default spherical caps)
    is evaluated by the constant-radius capsule chain
    (:func:`smin_sdf_capsule_contact_sculpt`).  Per-vertex radii or
    flattened caps take the separate variable-radius code path
    (:func:`smin_sdf_variable_radius_capsule_contact_sculpt`).

    Under the ``warp`` backend (:mod:`svmorph.core.backend`) the same step is
    computed by the kernels of :mod:`svmorph.core.sdf_warp`, and the
    displacement arrays are returned in single precision.

    Parameters
    ----------
    data : dict
        Simulation data dictionary with surface and centerline arrays.
    stent_vertices : jx.Array
        Stent axis vertices, shape ``(V, 3)``.
    s : float
        Signed force scale.
    target_stent_radius : float or jx.Array
        Target stent radius for SDF computation (scalar, or per-vertex radii
        of shape ``(V,)``; only the maximum is used, for bounding-box
        culling).
    current_stent_radius : float or jx.Array
        Current deployment radius of the stent: scalar for a uniform stent,
        or per-vertex radii of shape ``(V,)`` so the stent can follow an
        arbitrary radius profile along the centerline (e.g. a
        flared/trumpet end, a taper, or a local narrowing).  The profile
        passes through the per-vertex radii and is interpolated smoothly
        between them.
    influence_radius : float
        Radial distance beyond the stent within which points are displaced.
    contact_distance : float
        Distance threshold for stent–wall contact.
    f_scale : float
        Force magnitude scaling factor.
    cap_height_fraction : float
        Axial semi-axis of the two stent end caps as a fraction of the end
        radius; ``1.0`` (default) gives spherical caps, smaller values
        (e.g. ``0.35``) flatten them into half ellipsoids, which is
        recommended for flared ends.

    Returns
    -------
    full_surface_displacements : np.ndarray
        Displacement vectors for all surface points, shape ``(N_surf, 3)``.
    full_centerline_displacements : np.ndarray
        Displacement vectors for all centerline points, shape ``(N_cl, 3)``.
    step_size : float
        Scalar step size for stent radius increment tracking.
    """
    influence_radius, contact_distance, f_scale = _sdf_contact_defaults(
        influence_radius, contact_distance, f_scale)
    if np.ndim(target_stent_radius) > 0:
        target_stent_radius = float(np.max(target_stent_radius))
    is_uniform_stent = np.ndim(current_stent_radius) == 0 and cap_height_fraction == 1.0

    # ── 1. Extract geometry from simulation data ──────────────────────
    force_center_point_id = data["nodes"]["force_center_point_id"]
    logger.debug(f"Selected pointId: {force_center_point_id}")
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]

    if backend.get_backend() == "warp":
        # Steps 2 to 9 below as Warp kernels on the device.  What follows this block is the
        # JAX reference implementation of the same computation.
        from svmorph.core import sdf_warp

        start_time = time.time()
        surface_displacements, centerline_displacements = sdf_warp.sdf_contact_displacements(
            data_points, centerline_points, stent_vertices, current_stent_radius,
            **_warp_step_parameters(
                s, target_stent_radius, influence_radius, contact_distance, f_scale, cap_height_fraction))
        logger.timing(f"Deployment step on the Warp device: {time.time() - start_time:.4f} s")
        return surface_displacements, centerline_displacements, f_scale * (-s)

    # ── 2. Bounding-box culling ──────────────────────────────────────
    # Restrict the expensive SDF evaluation to the axis-aligned bounding
    # box of the stent, padded by (target_radius + influence + contact).
    # Both surface and centerline points are culled independently.
    start_time = time.time()
    surface_bbox_mask = stent_bounding_box(data_points, stent_vertices, target_stent_radius, influence_radius, contact_distance)
    centerline_bbox_mask = stent_bounding_box(centerline_points, stent_vertices, target_stent_radius, influence_radius, contact_distance)
    logger.timing(f"Bounding box computation: {time.time() - start_time:.4f} s")

    start_time = time.time()
    surface_bbox_mask = np.array(surface_bbox_mask)
    centerline_bbox_mask = np.array(centerline_bbox_mask)
    data_points = np.array(data_points)
    centerline_points = np.array(centerline_points)
    data_points_masked = data_points[surface_bbox_mask]
    centerline_points_masked = centerline_points[centerline_bbox_mask]
    logger.timing(f"NumPy cast and bbox masking: {time.time() - start_time:.4f} s")

    num_mesh_points = data_points.shape[0]
    num_centerline_points = centerline_points.shape[0]
    num_in_bb_mesh_points = data_points_masked.shape[0]

    # Concatenate surface + centerline into one batch so the SDF kernel
    # is called only once (GPU kernel-launch overhead dominates otherwise).
    query_points = np.concatenate((data_points_masked, centerline_points_masked), axis=0)

    # ── 3. Smooth-min capsule SDF evaluation ─────────────────────────
    # Evaluate the signed distance from every candidate point to the
    # capsule-chain stent surface.  The smooth-min reduction over
    # segments ensures C¹-continuous distance and direction fields.
    start_time = time.time()
    logger.debug(f"Total # surface and centerline points combined: {query_points.shape[0]}")
    if is_uniform_stent:
        combined_final_dist_to_surface, combined_final_direction = smin_sdf_capsule_contact_sculpt(query_points, stent_vertices, current_stent_radius)
    else:
        combined_final_dist_to_surface, combined_final_direction = smin_sdf_variable_radius_capsule_contact_sculpt(query_points, stent_vertices, current_stent_radius, cap_height_fraction)
    combined_final_dist_to_surface = np.array(combined_final_dist_to_surface)
    combined_final_direction = np.array(combined_final_direction)

    # ── 4. Split SDF results and classify points ─────────────────────
    # Separate the batched SDF results back into surface points and
    # centerline points, then classify each into contact / movable sets.

    # 4a. Surface points: identify those within the contact threshold.
    final_dist_to_surface = combined_final_dist_to_surface[:num_in_bb_mesh_points]
    final_direction = combined_final_direction[:num_in_bb_mesh_points]
    new_contact_mask = (final_dist_to_surface < contact_distance).astype(bool)
    logger.timing(f"New contact points computation: {time.time() - start_time:.4f} s")

    # 4b. Centerline points: keep only those *outside* the stent
    #     (inside-stent centerline points are already enclosed and should
    #     not receive displacement).
    centerline_points_dist_to_surface = combined_final_dist_to_surface[num_in_bb_mesh_points:]
    centerline_points_final_direction = combined_final_direction[num_in_bb_mesh_points:]
    centerline_outside_stent_mask = (centerline_points_dist_to_surface[:, 0] > 0).astype(bool)
    movables_centerline_points = centerline_points_masked[centerline_outside_stent_mask]
    movables_centerline_points_dist_to_surface = centerline_points_dist_to_surface[centerline_outside_stent_mask]
    movables_centerline_points_final_direction = centerline_points_final_direction[centerline_outside_stent_mask]

    # 4c. Merge surface + movable-centerline into one "movables" set.
    final_movables_dist_to_surface = np.concatenate((final_dist_to_surface, movables_centerline_points_dist_to_surface))
    final_movables_direction = np.concatenate((final_direction, movables_centerline_points_final_direction))
    num_final_movables = final_movables_dist_to_surface.shape[0]

    # Build a full-size boolean mask to scatter centerline displacements
    # back to their original indices at the end.
    full_centerline_points_mask = np.zeros(num_centerline_points, dtype=bool)
    full_centerline_points_mask[centerline_bbox_mask] = centerline_outside_stent_mask

    # ── 5. Early exit: no contact → no displacement ──────────────────
    # If the stent surface has not yet reached any vessel wall point,
    # there is nothing to displace; return zeros and the nominal step.
    start_time = time.time()
    in_contact_vertices = data_points_masked[new_contact_mask[:,0]]
    logger.timing(f"In-contact vertices subslice: {time.time() - start_time:.4f} s")
    if in_contact_vertices.shape[0] == 0:
        step_size = f_scale * (-s)
        return np.zeros((num_mesh_points, 3)), np.zeros((num_centerline_points, 3)), step_size

    # ── 6. KD-tree influence-zone query ──────────────────────────────
    # Build a spatial index of the contact-set vertices and query every
    # candidate point to find its nearest contact neighbor.  Points
    # closer than `influence_radius` will receive a displacement that
    # decays smoothly with distance to the contact front.
    start_time = time.time()
    contact_tree = cKDTree(in_contact_vertices, leafsize=32)
    query_points = np.concatenate((data_points_masked, movables_centerline_points), axis=0)
    dist_min, _ = contact_tree.query(query_points, k=1, distance_upper_bound=influence_radius, workers=-1)
    logger.timing(f"KD-tree construction and query: {time.time() - start_time:.4f} s")

    start_time = time.time()
    in_influence_mask = dist_min < influence_radius
    in_influence_indices = np.flatnonzero(in_influence_mask)
    logger.timing(f"Flatnonzero: {time.time() - start_time:.4f} s")

    start_time = time.time()
    in_influence_to_in_contact_distances = dist_min[in_influence_mask]
    logger.timing(f"In-influence vertices: {time.time() - start_time:.4f} s")
    # ── 7. Influence blending weights ────────────────────────────────
    # Compute a per-point blending weight (alpha) that is 1 at the
    # contact front and linearly decays to 0 at `influence_radius`.
    # This prevents hard displacement discontinuities at the edge of
    # the influence zone.
    start_time = time.time()
    influence_radius_mask = (final_movables_dist_to_surface < influence_radius).astype(int)
    in_influence_vertices_blended_alpha_mask = np.zeros(num_final_movables)
    in_influence_vertices_blended_alpha = (1 - in_influence_to_in_contact_distances / influence_radius) # linear blending scheme
    logger.timing(f"JIT sculpt part two: {time.time() - start_time:.4f} s")

    start_time = time.time()
    in_influence_vertices_blended_alpha = np.array(in_influence_vertices_blended_alpha)
    in_influence_vertices_blended_alpha_mask[in_influence_indices] = in_influence_vertices_blended_alpha
    logger.timing(f"Blended alpha mask (NumPy): {time.time() - start_time:.4f} s")

    # ── 8. Assemble displacement vectors ─────────────────────────────
    # Displacement magnitude uses a quartic bump: ((d/R)² − 1)², which
    # is C¹ at both the stent surface (d=0) and the influence boundary
    # (d=R).  Points that have penetrated inside the stent (negative
    # SDF) get an additional offset to push them back to the surface.
    start_time = time.time()
    logger.debug(f"# negative SDF values to be corrected: {np.sum(final_movables_dist_to_surface < 0)}")

    # Interior correction: points with negative SDF are inside the stent
    # and need an extra push equal to their penetration depth to correct.
    interior_points_offset = np.maximum(-final_movables_dist_to_surface, 0.0)
    final_movables_dist_to_surface = np.maximum(final_movables_dist_to_surface, 0.0)

    # Quartic bump profile × force scale × influence blend
    displacements_magnitude = f_scale * ((final_movables_dist_to_surface / influence_radius) ** 2 - 1) ** 2 * (-s) * influence_radius_mask * in_influence_vertices_blended_alpha_mask[:, None]
    displacements_magnitude += interior_points_offset
    displacements = displacements_magnitude * final_movables_direction

    # ── 9. Scatter back to full-size arrays ──────────────────────────
    # The displacements were computed only for the bbox-culled subset.
    # Scatter them back into full-size zero arrays indexed by the
    # original vertex ordering.
    full_surface_displacements = np.zeros((num_mesh_points, 3))
    full_surface_displacements[surface_bbox_mask] = displacements[:num_in_bb_mesh_points]
    full_centerline_displacements = np.zeros((num_centerline_points, 3))
    full_centerline_displacements[full_centerline_points_mask] = displacements[num_in_bb_mesh_points:]
    step_size = f_scale * (-s)
    logger.timing(f"Remaining displacements: {time.time() - start_time:.4f} s")

    return full_surface_displacements, full_centerline_displacements, step_size
