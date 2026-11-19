"""The SDFStent deployment step as NVIDIA Warp kernels.

This is the ``warp`` backend of :mod:`svmorph.core.backend`: the arithmetic of
:func:`svmorph.core.deformation.compute_sdf_contact_displacements`, which
remains the reference, written as kernels that run once per mesh point on a
CUDA GPU.  Only the stent deployment step is here; every other computation in
svMorph stays on JAX.

What runs on the device, and the reference it mirrors:

==========================================  ==================================================
Kernel                                      Reference in ``deformation.py``
==========================================  ==================================================
``_uniform_chain_sdf``                      ``smin_sdf_capsule_contact_sculpt``
``_variable_radius_chain_sdf``              ``smin_sdf_variable_radius_capsule_contact_sculpt``
``_classify_*_kernel``                      bounding-box culling, contact and movable sets
``_leaf_boxes_kernel``, ``_node_boxes_kernel``,
``_nearest_contact_squared``                SciPy ``cKDTree`` nearest-contact query
``_displacement_kernel``                    influence blending and displacement assembly
==========================================  ==================================================

As in the reference, the uniform stent (scalar radius, spherical caps) and the
stent with a radius profile or flattened caps are two separate code paths, so
that the uniform stent keeps its speed and its results.

**Contact search.**  Warp has range queries for point sets but no
nearest-neighbour search, so the distance from each point to the nearest point
in contact is computed by kernels of this module.  They rely on what a
deployment has and a general search cannot assume: the points are the same from
one step to the next, and only which of them are in contact changes.  The
surface points are sorted along a Morton curve, which makes an implicit 8-ary
tree of them that is never rebuilt; a step only refits the bounding boxes of
the points currently in contact, from the leaves up, and then searches the tree
with a small stack.  The order affects speed only: the boxes are computed from
where the points are at each step, so the search is exact in any order.

**Where the time goes.**  On a GPU the kernels are a small part of a step; most
of it is copying the mesh to the device and the displacements back.  Two things
address that:

* :func:`sdf_contact_displacements`, the drop-in step, keeps its device arrays
  and recorded kernel launches from one call to the next and reads results back
  through page-locked host memory.
* :class:`DeviceDeployment` uploads the mesh once and keeps it on the device:
  a step applies its displacements there, and the positions are only copied
  back when :meth:`DeviceDeployment.sync` is called.

Everything is computed in single precision, like the JAX reference, and agrees
with it to about that precision.  Lengths are passed in explicitly, already in
the active unit, so nothing here depends on :func:`svmorph.core.units.L`.

Kernels are compiled the first time they run on a device and cached by Warp
afterwards.

The contact search, the persistent buffers and the recorded launches are
adapted from the Warp prototype by Andras Lasso
(https://github.com/lassoan/svMorph/tree/warp-backend).
"""

# No "from __future__ import annotations" in this module: it turns the kernels'
# argument annotations into strings, which older Warp releases cannot resolve.
# Annotations here therefore use typing.Optional rather than "X | None".

from typing import Optional, Tuple

import numpy as np
import warp as wp

from svmorph.core import backend
from svmorph.logging import get_logger

logger = get_logger(__name__)

# What the classification kernels make of a point
_OUTSIDE = wp.constant(0)  # outside the stent's bounding box, or a centerline point inside the stent
_MOVABLE = wp.constant(1)  # displaced if it is within the influence radius of a point in contact
_CONTACT = wp.constant(2)  # a surface point the stent has reached; movable as well

# Shape of the contact search tree: children per node and points per leaf.  The
# kernels' loops over them are unrolled, so they are module constants.
_BRANCHING = wp.constant(8)
_LEAF_SIZE = wp.constant(8)
# Nodes waiting to be visited during a search.  Visiting a node takes it off the
# stack and puts at most _BRANCHING children on, so a tree of depth d needs
# d * (_BRANCHING - 1) + 1 entries: 64 are enough for nine levels.
_STACK_SIZE = 64
_NodeStack = wp.types.vector(length=_STACK_SIZE, dtype=wp.int32)
_DistanceStack = wp.types.vector(length=_STACK_SIZE, dtype=wp.float32)
# Position given to a tree slot that holds no point in contact, and the corners
# of an empty box: far from everything, yet small enough that its square is finite.
_FAR = wp.constant(1.0e18)
# Morton codes use 10 bits per axis
_MORTON_CELLS = 1024
# The Morton order is refreshed after this many steps, and whenever the stent's
# bounding box has moved by more than _RESORT_BOX_FRACTION of its smallest side.
_STEPS_PER_SORT = 64
_RESORT_BOX_FRACTION = 0.05


# ── Stent SDF ────────────────────────────────────────────────────────────────

@wp.func
def _uniform_chain_sdf(
    point: wp.vec3,
    stent_vertices: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    smoothing: wp.float32,
):
    """Smooth-min distance and outward direction from *point* to a constant-radius capsule chain."""
    # 4 k, as in deformation.smin_and_gradient
    k = smoothing * 4.0

    min_distance = float(0.0)
    min_direction = wp.vec3(0.0, 0.0, 0.0)
    for segment in range(stent_vertices.shape[0] - 1):
        a = stent_vertices[segment]
        ba = stent_vertices[segment + 1] - a
        pa = point - a
        h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
        axis_to_point = pa - h * ba
        axis_distance = wp.length(axis_to_point)
        segment_direction = axis_to_point / axis_distance
        segment_distance = axis_distance - radius

        if segment == 0:
            min_distance = segment_distance
            min_direction = segment_direction
        else:
            # deformation.smin_and_gradient, with the running minimum as a and this segment as b
            blend = wp.max(k - wp.abs(min_distance - segment_distance), 0.0) / k
            n = 0.5 * blend
            m = blend * blend * k / 4.0
            if min_distance < segment_distance:
                min_direction = min_direction + (segment_direction - min_direction) * n
                min_distance = min_distance - m
            else:
                min_direction = min_direction + (segment_direction - min_direction) * (1.0 - n)
                min_distance = segment_distance - m

    return min_distance, min_direction


@wp.func
def _variable_radius_chain_sdf(
    point: wp.vec3,
    stent_vertices: wp.array(dtype=wp.vec3),
    radius_profile: wp.array(dtype=wp.vec3),
    cap_stretch: wp.float32,
    smoothing: wp.float32,
):
    """Smooth-min distance and outward direction from *point* to a capsule chain with a radius profile.

    Mirrors ``deformation._variable_radius_capsule_chain_sdf``.  Each entry of
    *radius_profile* holds, for one stent axis vertex, its arc-length position,
    its radius and the slope of the monotone cubic through the radii there
    (see :func:`_radius_profile`).
    """
    k = smoothing * 4.0
    number_of_vertices = stent_vertices.shape[0]

    # 1. Smooth minimum of the distance to the axis segments, blending along with it the outward
    #    direction and the arc-length coordinate at which each segment sees the point
    min_distance = float(0.0)
    min_direction = wp.vec3(0.0, 0.0, 0.0)
    min_arc = float(0.0)
    for segment in range(number_of_vertices - 1):
        a = stent_vertices[segment]
        ba = stent_vertices[segment + 1] - a
        pa = point - a
        ba_dot_ba = wp.dot(ba, ba)
        h_unclamped = wp.dot(pa, ba) / ba_dot_ba
        h = wp.clamp(h_unclamped, 0.0, 1.0)
        axis_to_point = pa - h * ba
        segment_distance = wp.length(axis_to_point)
        segment_direction = axis_to_point / segment_distance
        # Not clamped to the segment, so all segments of a straight stretch agree on it
        segment_arc = radius_profile[segment][0] + h_unclamped * wp.sqrt(ba_dot_ba)

        if segment == 0:
            min_distance = segment_distance
            min_direction = segment_direction
            min_arc = segment_arc
        else:
            blend = wp.max(k - wp.abs(min_distance - segment_distance), 0.0) / k
            n = 0.5 * blend
            m = blend * blend * k / 4.0
            if min_distance < segment_distance:
                min_direction = min_direction + (segment_direction - min_direction) * n
                min_arc = min_arc + (segment_arc - min_arc) * n
                min_distance = min_distance - m
            else:
                min_direction = min_direction + (segment_direction - min_direction) * (1.0 - n)
                min_arc = min_arc + (segment_arc - min_arc) * (1.0 - n)
                min_distance = segment_distance - m

    # 2. Radius profile at that arc-length coordinate, subtracted after the smooth minimum
    arc_on_stent = wp.clamp(min_arc, 0.0, radius_profile[number_of_vertices - 1][0])
    i = int(0)
    for j in range(1, number_of_vertices - 1):
        if radius_profile[j][0] <= arc_on_stent:
            i = j
    start = radius_profile[i]
    end = radius_profile[i + 1]
    segment_length = end[0] - start[0]
    t = (arc_on_stent - start[0]) / segment_length
    radius = (start[1] + (end[1] - start[1]) * (t * t * (3.0 - 2.0 * t))
              + segment_length * (start[2] * (t * (t - 1.0) * (t - 1.0)) + end[2] * (t * t * (t - 1.0))))
    distance = min_distance - radius

    # 3. End caps: stretch the axial overshoot past the stent ends (zero along the stent body)
    overshoot_ratio = wp.clamp((min_arc - arc_on_stent) / wp.max(min_distance, 1.0e-12), -1.0, 1.0)
    stretched_distance_scale = wp.sqrt(1.0 + cap_stretch * overshoot_ratio * overshoot_ratio)
    stretched_gradient_scale = wp.sqrt(
        1.0 + cap_stretch * (cap_stretch + 2.0) * overshoot_ratio * overshoot_ratio)
    flattened_cap_distance = ((distance + min_distance * (stretched_distance_scale - 1.0))
                              * stretched_distance_scale / stretched_gradient_scale)
    # The flattened cap lies inside the spherical one, so it is never closer than the latter
    distance = wp.max(flattened_cap_distance, distance)

    return distance, min_direction


@wp.kernel
def _uniform_sdf_kernel(
    points: wp.array(dtype=wp.vec3),
    stent_vertices: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    smoothing: wp.float32,
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    point_distance, point_direction = _uniform_chain_sdf(points[tid], stent_vertices, radius, smoothing)
    distance[tid] = point_distance
    direction[tid] = point_direction


@wp.kernel
def _variable_radius_sdf_kernel(
    points: wp.array(dtype=wp.vec3),
    stent_vertices: wp.array(dtype=wp.vec3),
    radius_profile: wp.array(dtype=wp.vec3),
    cap_stretch: wp.float32,
    smoothing: wp.float32,
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    point_distance, point_direction = _variable_radius_chain_sdf(
        points[tid], stent_vertices, radius_profile, cap_stretch, smoothing)
    distance[tid] = point_distance
    direction[tid] = point_direction


# ── Classification: bounding box, SDF, contact and movable sets ──────────────

@wp.func
def _outside_box(point: wp.vec3, box_min: wp.vec3, box_max: wp.vec3):
    return (point[0] < box_min[0] or point[1] < box_min[1] or point[2] < box_min[2]
            or point[0] > box_max[0] or point[1] > box_max[1] or point[2] > box_max[2])


@wp.func
def _point_state(is_surface_point: bool, distance: wp.float32, contact_distance: wp.float32):
    """What a point inside the bounding box is to the step, given its distance to the stent.

    A surface point is movable, and in contact once the stent is within
    *contact_distance* of it.  A centerline point is movable only while it is
    outside the stent: one the stent already encloses stays put.
    """
    if is_surface_point:
        if distance < contact_distance:
            return _CONTACT
        return _MOVABLE
    if distance > 0.0:
        return _MOVABLE
    return _OUTSIDE


@wp.kernel
def _classify_uniform_kernel(
    points: wp.array(dtype=wp.vec3),
    number_of_surface_points: wp.int32,
    stent_vertices: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    smoothing: wp.float32,
    box_min: wp.vec3,
    box_max: wp.vec3,
    contact_distance: wp.float32,
    order: wp.array(dtype=wp.int32),
    sorted_points: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
    state: wp.array(dtype=wp.int32),
    sorted_contact: wp.array(dtype=wp.vec3),
):
    """The SDF of every point in the stent's bounding box and its role in the step, for a uniform stent.

    *points* holds the surface points followed by the centerline points.  The
    surface points are visited in the order of the search tree, ``order[tid]``
    being the point at place *tid* in it, and everything written here is in
    that order; the centerline points keep their places after them.
    *sorted_contact* is the leaf level of the tree: a point in contact is
    written to its place there, and ``_FAR`` where a point is not in contact.
    """
    tid = wp.tid()
    is_surface_point = tid < number_of_surface_points
    point_id = tid
    if is_surface_point:
        point_id = order[tid]
        sorted_contact[tid] = wp.vec3(_FAR, _FAR, _FAR)
    point = points[point_id]
    sorted_points[tid] = point
    if _outside_box(point, box_min, box_max):
        state[tid] = _OUTSIDE
        return

    point_distance, point_direction = _uniform_chain_sdf(point, stent_vertices, radius, smoothing)
    distance[tid] = point_distance
    direction[tid] = point_direction
    point_state = _point_state(is_surface_point, point_distance, contact_distance)
    state[tid] = point_state
    if point_state == _CONTACT:
        sorted_contact[tid] = point


@wp.kernel
def _classify_variable_radius_kernel(
    points: wp.array(dtype=wp.vec3),
    number_of_surface_points: wp.int32,
    stent_vertices: wp.array(dtype=wp.vec3),
    radius_profile: wp.array(dtype=wp.vec3),
    cap_stretch: wp.float32,
    smoothing: wp.float32,
    box_min: wp.vec3,
    box_max: wp.vec3,
    contact_distance: wp.float32,
    order: wp.array(dtype=wp.int32),
    sorted_points: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
    state: wp.array(dtype=wp.int32),
    sorted_contact: wp.array(dtype=wp.vec3),
):
    """:func:`_classify_uniform_kernel` for a stent with a radius profile or flattened end caps."""
    tid = wp.tid()
    is_surface_point = tid < number_of_surface_points
    point_id = tid
    if is_surface_point:
        point_id = order[tid]
        sorted_contact[tid] = wp.vec3(_FAR, _FAR, _FAR)
    point = points[point_id]
    sorted_points[tid] = point
    if _outside_box(point, box_min, box_max):
        state[tid] = _OUTSIDE
        return

    point_distance, point_direction = _variable_radius_chain_sdf(
        point, stent_vertices, radius_profile, cap_stretch, smoothing)
    distance[tid] = point_distance
    direction[tid] = point_direction
    point_state = _point_state(is_surface_point, point_distance, contact_distance)
    state[tid] = point_state
    if point_state == _CONTACT:
        sorted_contact[tid] = point


# ── Contact search tree ──────────────────────────────────────────────────────

@wp.func
def _spread_bits(value: wp.int32):
    """The low 10 bits of *value*, with two zero bits inserted after each."""
    x = value & 1023
    x = (x | (x << 16)) & 50331903  # 0x030000FF
    x = (x | (x << 8)) & 50393103  # 0x0300F00F
    x = (x | (x << 4)) & 51130563  # 0x030C30C3
    x = (x | (x << 2)) & 153391689  # 0x09249249
    return x


@wp.kernel
def _morton_kernel(
    points: wp.array(dtype=wp.vec3),
    box_min: wp.vec3,
    cells_per_length: wp.vec3,
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
):
    """The place of each surface point along the Morton curve through the stent's bounding box.

    A point outside the box gets the code of the nearest cell of the box.  Such
    a point is never in contact, so its place only decides which leaf has an
    empty slot.
    """
    tid = wp.tid()
    cell = wp.cw_mul(points[tid] - box_min, cells_per_length)
    # Clamped as a float: converting a number too large for an int is undefined
    x = int(wp.clamp(cell[0], 0.0, 1023.0))
    y = int(wp.clamp(cell[1], 0.0, 1023.0))
    z = int(wp.clamp(cell[2], 0.0, 1023.0))
    keys[tid] = _spread_bits(x) | (_spread_bits(y) << 1) | (_spread_bits(z) << 2)
    values[tid] = tid


@wp.kernel
def _leaf_boxes_kernel(
    sorted_contact: wp.array(dtype=wp.vec3),
    leaf_offset: wp.int32,
    boxes: wp.array(dtype=wp.vec3),
):
    """The bounding box of the points in contact in each leaf.

    The corners of node ``n`` are ``boxes[2 n]`` and ``boxes[2 n + 1]``.
    """
    tid = wp.tid()
    lower = wp.vec3(_FAR, _FAR, _FAR)
    upper = wp.vec3(-_FAR, -_FAR, -_FAR)
    first = tid * _LEAF_SIZE
    for i in range(_LEAF_SIZE):
        point = sorted_contact[first + i]
        if point[0] < _FAR:
            lower = wp.min(lower, point)
            upper = wp.max(upper, point)
    node = leaf_offset + tid
    boxes[2 * node] = lower
    boxes[2 * node + 1] = upper


@wp.kernel
def _node_boxes_kernel(first_node: wp.int32, boxes: wp.array(dtype=wp.vec3)):
    """The bounding box of each node of one level, from those of its children on the level below."""
    node = first_node + wp.tid()
    child = node * _BRANCHING + 1
    lower = boxes[2 * child]
    upper = boxes[2 * child + 1]
    for c in range(1, _BRANCHING):
        lower = wp.min(lower, boxes[2 * (child + c)])
        upper = wp.max(upper, boxes[2 * (child + c) + 1])
    boxes[2 * node] = lower
    boxes[2 * node + 1] = upper


@wp.func
def _box_distance_squared(point: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    """Squared distance from *point* to a box; an empty box is too far to be near anything."""
    x = wp.max(wp.max(lower[0] - point[0], point[0] - upper[0]), 0.0)
    y = wp.max(wp.max(lower[1] - point[1], point[1] - upper[1]), 0.0)
    z = wp.max(wp.max(lower[2] - point[2], point[2] - upper[2]), 0.0)
    return x * x + y * y + z * z


@wp.func
def _nearest_contact_squared(
    point: wp.vec3,
    boxes: wp.array(dtype=wp.vec3),
    sorted_contact: wp.array(dtype=wp.vec3),
    leaf_offset: wp.int32,
    limit_squared: wp.float32,
):
    """Squared distance from *point* to the nearest point in contact; *limit_squared* if none is nearer."""
    best = float(limit_squared)
    nodes = _NodeStack()
    distances = _DistanceStack()
    nodes[0] = 0
    distances[0] = 0.0
    count = int(1)
    while count > 0:
        count -= 1
        node = nodes[count]
        # The nearest found so far may have come closer than this node's box since it was pushed
        if distances[count] < best:
            if node >= leaf_offset:
                first = (node - leaf_offset) * _LEAF_SIZE
                for i in range(_LEAF_SIZE):
                    offset = sorted_contact[first + i] - point
                    best = wp.min(best, wp.dot(offset, offset))
            else:
                # Children that can hold something nearer go on the stack, the nearest of them
                # last, so that it is visited first and most of the others are then dropped
                first_child = node * _BRANCHING + 1
                nearest_child = int(-1)
                nearest_distance = float(best)
                for c in range(_BRANCHING):
                    child = first_child + c
                    child_distance = _box_distance_squared(point, boxes[2 * child], boxes[2 * child + 1])
                    if child_distance < best:
                        if child_distance < nearest_distance:
                            if nearest_child >= 0:
                                nodes[count] = nearest_child
                                distances[count] = nearest_distance
                                count += 1
                            nearest_child = child
                            nearest_distance = child_distance
                        else:
                            nodes[count] = child
                            distances[count] = child_distance
                            count += 1
                if nearest_child >= 0:
                    nodes[count] = nearest_child
                    distances[count] = nearest_distance
                    count += 1
    return best


# ── Displacement ─────────────────────────────────────────────────────────────

@wp.kernel
def _displacement_kernel(
    sorted_points: wp.array(dtype=wp.vec3),
    number_of_surface_points: wp.int32,
    order: wp.array(dtype=wp.int32),
    state: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.vec3),
    boxes: wp.array(dtype=wp.vec3),
    sorted_contact: wp.array(dtype=wp.vec3),
    leaf_offset: wp.int32,
    influence_radius: wp.float32,
    magnitude_scale: wp.float32,
    displacement: wp.array(dtype=wp.vec3),
):
    """Influence blending and displacement assembly for one point.

    Visits the points in the order of the search tree, as the classification
    kernel left its results, and writes each displacement to the point's own
    place in the mesh.
    """
    tid = wp.tid()
    point_id = tid
    if tid < number_of_surface_points:
        point_id = order[tid]
    # The root's box is empty while the stent has not reached the wall, and then nothing moves
    if state[tid] == _OUTSIDE or boxes[0][0] > boxes[1][0]:
        displacement[point_id] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Blending weight: 1 at the contact front, decaying linearly to 0 at the influence radius.
    # A point in contact is its own nearest; a point further from the stent than the influence
    # radius has no use for a weight (see below), so neither is searched for.
    point_distance = distance[tid]
    alpha = float(0.0)
    if state[tid] == _CONTACT:
        alpha = 1.0
    elif point_distance < influence_radius:
        limit_squared = influence_radius * influence_radius
        nearest_squared = _nearest_contact_squared(
            sorted_points[tid], boxes, sorted_contact, leaf_offset, limit_squared)
        if nearest_squared < limit_squared:
            alpha = 1.0 - wp.sqrt(nearest_squared) / influence_radius

    # A point inside the stent is pushed out by its penetration depth, whatever its weight;
    # the quartic bump ((d/R)^2 - 1)^2 moves the rest, out to the influence radius.
    magnitude = wp.max(-point_distance, 0.0)
    if point_distance < influence_radius:
        ratio = wp.max(point_distance, 0.0) / influence_radius
        bump = ratio * ratio - 1.0
        magnitude = magnitude + magnitude_scale * bump * bump * alpha
    displacement[point_id] = magnitude * direction[tid]


@wp.kernel
def _apply_kernel(points: wp.array(dtype=wp.vec3), displacement: wp.array(dtype=wp.vec3)):
    """Add each point's displacement to it, on the device."""
    tid = wp.tid()
    points[tid] = points[tid] + displacement[tid]


# ── Host-side helpers ────────────────────────────────────────────────────────

def _get_device(device: Optional[str] = None):
    """The Warp device to compute on: *device* if given, otherwise the backend's choice."""
    if device is not None:
        return wp.get_device(device)
    if backend.get_backend() == "warp":
        return wp.get_device(backend.warp_device())
    # Called directly while the JAX backend is selected
    return wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")


def _is_uniform_stent(radius, cap_height_fraction: float) -> bool:
    """Whether the stent takes the constant-radius code path, as the reference decides it."""
    return np.ndim(radius) == 0 and cap_height_fraction == 1.0


def _as_points(points) -> np.ndarray:
    """*points* as a C-contiguous float32 ``(N, 3)`` array, without copying one that already is."""
    return np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)


def _as_stent_vertices(stent_vertices) -> np.ndarray:
    stent_vertices = _as_points(np.asarray(stent_vertices))
    if len(stent_vertices) < 2:
        raise ValueError("A capsule-chain stent needs at least two axis vertices")
    return stent_vertices


def _radius_profile(stent_vertices: np.ndarray, radius) -> np.ndarray:
    """Arc-length position, radius and radius slope at each stent axis vertex, shape ``(V, 3)``.

    The slopes are those of the monotone cubic (PCHIP) through the per-vertex
    radii, computed as ``deformation._radius_profile_slopes`` computes them and
    in single precision like it.
    """
    one = np.float32(1.0)
    segment_lengths = np.linalg.norm(np.diff(stent_vertices, axis=0), axis=1).astype(np.float32)
    arc_positions = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)])
    radii = np.broadcast_to(np.asarray(radius, dtype=np.float32), (len(stent_vertices),))
    secants = np.diff(radii) / segment_lengths
    weight_previous = np.float32(2.0) * segment_lengths[1:] + segment_lengths[:-1]
    weight_next = segment_lengths[1:] + np.float32(2.0) * segment_lengths[:-1]
    same_sign = secants[:-1] * secants[1:] > 0.0
    secants_previous = np.where(same_sign, secants[:-1], one)
    secants_next = np.where(same_sign, secants[1:], one)
    interior_slopes = np.where(
        same_sign,
        (weight_previous + weight_next) / (weight_previous / secants_previous + weight_next / secants_next),
        np.float32(0.0))
    slopes = np.concatenate([secants[:1], interior_slopes, secants[-1:]])
    return np.ascontiguousarray(np.column_stack([arc_positions, radii, slopes]), dtype=np.float32)


def _cap_stretch(cap_height_fraction: float) -> float:
    return 1.0 / float(cap_height_fraction) ** 2 - 1.0


def _bounding_box(stent_vertices: np.ndarray, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    """Corners of the stent axis' bounding box, grown by *padding* on every side."""
    padding = np.float32(padding)
    return stent_vertices.min(axis=0) - padding, stent_vertices.max(axis=0) + padding


def _upload(destination, host_array: np.ndarray, dtype, offset: int = 0) -> None:
    """Copy *host_array* into the device array *destination*, starting at *offset*.

    The NumPy array is wrapped, not copied, and Warp copies ordinary host
    memory to a device synchronously, so the caller is free to change the
    array as soon as this returns.
    """
    source = wp.array(host_array, dtype=dtype, device="cpu", copy=False)
    wp.copy(destination, source, offset, 0, len(host_array))


class _DeviceStep:
    """The device arrays and recorded kernel launches of deployment steps on one mesh.

    A deployment is many steps on the same mesh with the same stent axis, and
    on a small mesh most of a step is not arithmetic but allocating arrays and
    packing kernel arguments.  So both are kept: the arrays are filled in
    place, and each launch is recorded once with its arrays bound, after which
    a step only sets the few numbers that change.  The memory stays allocated
    on the device, about 90 bytes per mesh point, for as long as this object
    lives.

    The contact search tree lives here as well.  *order* lists the surface
    points along a Morton curve, so that points close in it are close in
    space.  Eight consecutive points of it form a leaf, eight consecutive
    leaves a node, and so on up to a single root: the tree is the order itself
    and needs no arrays of its own except *boxes*, the bounding box of the
    points in contact under each node, numbered as a heap (root 0, children of
    node ``n`` from ``n * _BRANCHING + 1``).  Refitting those boxes is all a
    step has to do, because the points in contact change at every step and the
    points do not.  A node with nothing in contact under it has an empty box,
    which a search never enters.
    """

    def __init__(self, device, number_of_surface_points: int, number_of_centerline_points: int,
                 number_of_stent_vertices: int):
        self.device = device
        self.sizes = (number_of_surface_points, number_of_centerline_points, number_of_stent_vertices)
        self.number_of_surface_points = number_of_surface_points
        self.number_of_points = number_of_points = number_of_surface_points + number_of_centerline_points

        self.stent_vertices = wp.empty(number_of_stent_vertices, dtype=wp.vec3, device=device)
        self.radius_profile = wp.empty(number_of_stent_vertices, dtype=wp.vec3, device=device)
        self.points = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        self.sorted_points = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        self.distance = wp.empty(number_of_points, dtype=wp.float32, device=device)
        self.direction = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        self.state = wp.empty(number_of_points, dtype=wp.int32, device=device)
        self.displacement = wp.empty(number_of_points, dtype=wp.vec3, device=device)
        # Results come back from a GPU through page-locked host memory, which is several times
        # faster to copy into than ordinary memory.  On the CPU device there is nothing to copy.
        self.staging = None
        self.staging_host = None
        if device.is_cuda:
            self.staging = wp.empty(number_of_points, dtype=wp.vec3, device="cpu", pinned=True)
            self.staging_host = self.staging.numpy()

        # The search tree: every level but the last is complete, so that nothing has to be stored
        # to find a child, and the leaves are the last level.
        number_of_leaves = -(-number_of_surface_points // _LEAF_SIZE)
        depth = 1
        while _BRANCHING ** depth < number_of_leaves:
            depth += 1
        if depth * (_BRANCHING - 1) + 1 > _STACK_SIZE:
            raise ValueError(
                f"A mesh of {number_of_surface_points} points is too large for the contact search")
        self.leaf_offset = (_BRANCHING ** depth - 1) // (_BRANCHING - 1)
        # Every node above the leaves reads all of its children, so the leaves are padded to a
        # multiple of _BRANCHING.  The padding, and the unused slots of the last leaf, keep the
        # empty values they are given here: no point is ever written to them.
        padded_leaves = -(-number_of_leaves // _BRANCHING) * _BRANCHING
        empty_box = np.array([[_FAR] * 3, [-_FAR] * 3], dtype=np.float32)
        self.boxes = wp.array(
            np.tile(empty_box, (self.leaf_offset + padded_leaves, 1)), dtype=wp.vec3, device=device)
        self.sorted_contact = wp.array(
            np.full((padded_leaves * _LEAF_SIZE, 3), _FAR, dtype=np.float32), dtype=wp.vec3, device=device)
        # The order: point indices sorted by Morton code.  Warp's radix sort needs room for as
        # many entries again in both arrays.
        self.morton_codes = wp.empty(2 * number_of_surface_points, dtype=wp.int32, device=device)
        self.order = wp.empty(2 * number_of_surface_points, dtype=wp.int32, device=device)
        self.sorted_box = None
        self.steps_since_sort = 0

        # The stent axis as last sent to the device, and the bounding box that follows from it:
        # they rarely change during a deployment, and are only sent and worked out again when they do
        self.host_stent_vertices = None
        self.box = None
        self.box_padding = None

        # record_cmd=True launches nothing; the scalar arguments are placeholders set at each step
        zero = wp.vec3(0.0, 0.0, 0.0)
        classify_outputs = [self.sorted_points, self.distance, self.direction, self.state, self.sorted_contact]
        self.classify_uniform = wp.launch(
            _classify_uniform_kernel,
            dim=number_of_points,
            inputs=[self.points, number_of_surface_points, self.stent_vertices,
                    0.0, 1.0, zero, zero, 0.0, self.order],
            outputs=classify_outputs,
            device=device, record_cmd=True)
        self.classify_variable_radius = wp.launch(
            _classify_variable_radius_kernel,
            dim=number_of_points,
            inputs=[self.points, number_of_surface_points, self.stent_vertices, self.radius_profile,
                    0.0, 1.0, zero, zero, 0.0, self.order],
            outputs=classify_outputs,
            device=device, record_cmd=True)
        # The boxes, from the leaves up: a level can only be computed after the one below it
        self.fit_boxes = [wp.launch(
            _leaf_boxes_kernel,
            dim=number_of_leaves,
            inputs=[self.sorted_contact, self.leaf_offset],
            outputs=[self.boxes],
            device=device, record_cmd=True)]
        nodes_on_level = number_of_leaves
        for level in range(depth - 1, -1, -1):
            nodes_on_level = -(-nodes_on_level // _BRANCHING)
            self.fit_boxes.append(wp.launch(
                _node_boxes_kernel,
                dim=nodes_on_level,
                inputs=[(_BRANCHING ** level - 1) // (_BRANCHING - 1), self.boxes],
                device=device, record_cmd=True))
        self.displace = wp.launch(
            _displacement_kernel,
            dim=number_of_points,
            inputs=[self.sorted_points, number_of_surface_points, self.order, self.state,
                    self.distance, self.direction, self.boxes, self.sorted_contact, self.leaf_offset,
                    0.0, 0.0],
            outputs=[self.displacement],
            device=device, record_cmd=True)
        self.apply = wp.launch(
            _apply_kernel,
            dim=number_of_points,
            inputs=[self.points, self.displacement],
            device=device, record_cmd=True)

    def fits(self, device, number_of_surface_points: int, number_of_centerline_points: int,
             number_of_stent_vertices: int) -> bool:
        return self.device == device and self.sizes == (
            number_of_surface_points, number_of_centerline_points, number_of_stent_vertices)

    def upload_points(self, surface_points: np.ndarray, centerline_points: np.ndarray) -> None:
        """Send the mesh to the device: surface points first, centerline points after them."""
        _upload(self.points, surface_points, wp.vec3)
        if len(centerline_points):
            _upload(self.points, centerline_points, wp.vec3, offset=self.number_of_surface_points)

    def _sort(self, box_min: np.ndarray, box_max: np.ndarray) -> None:
        """Order the surface points, which are on the device, along the Morton curve through the box."""
        extent = np.maximum(box_max.astype(float) - box_min.astype(float), 1.0e-12)
        wp.launch(
            _morton_kernel,
            dim=self.number_of_surface_points,
            inputs=[self.points, wp.vec3(*(float(value) for value in box_min)),
                    wp.vec3(*(float(value) for value in _MORTON_CELLS / extent))],
            outputs=[self.morton_codes, self.order],
            device=self.device)
        wp.utils.radix_sort_pairs(self.morton_codes, self.order, self.number_of_surface_points)
        self.sorted_box = (box_min.copy(), box_max.copy())
        self.steps_since_sort = 0

    def _update_stent(self, stent_vertices: np.ndarray, box_padding: float) -> None:
        """Send the stent axis to the device and recompute its bounding box, if either has changed."""
        stent_changed = (self.host_stent_vertices is None
                         or not np.array_equal(self.host_stent_vertices, stent_vertices))
        if stent_changed:
            _upload(self.stent_vertices, stent_vertices, wp.vec3)
            self.host_stent_vertices = stent_vertices.copy()
        if not stent_changed and box_padding == self.box_padding:
            return

        self.box = box_min, box_max = _bounding_box(stent_vertices, box_padding)
        self.box_padding = box_padding
        for classify in (self.classify_uniform, self.classify_variable_radius):
            classify.set_param_by_name("box_min", wp.vec3(*(float(value) for value in box_min)))
            classify.set_param_by_name("box_max", wp.vec3(*(float(value) for value in box_max)))
        if self.sorted_box is not None:
            # A box that has moved a little (an axis being straightened) leaves the Morton order as
            # good as it was; one that has moved a lot is another stent, and the order a poor one
            sorted_min, sorted_max = self.sorted_box
            tolerance = _RESORT_BOX_FRACTION * float(np.min(sorted_max - sorted_min))
            if max(np.max(np.abs(box_min - sorted_min)), np.max(np.abs(box_max - sorted_max))) > tolerance:
                self.sorted_box = None

    def _uniform_classification(self, radius):
        """The classification launch of a uniform stent, with its radius set."""
        self.classify_uniform.set_param_by_name("radius", float(radius))
        return self.classify_uniform

    def _variable_radius_classification(self, stent_vertices: np.ndarray, radius, cap_height_fraction: float):
        """The classification launch of a stent with a radius profile or flattened caps, with both set."""
        _upload(self.radius_profile, _radius_profile(stent_vertices, radius), wp.vec3)
        self.classify_variable_radius.set_param_by_name("cap_stretch", _cap_stretch(cap_height_fraction))
        return self.classify_variable_radius

    def run(self, stent_vertices: np.ndarray, radius, *, box_padding: float, cap_height_fraction: float,
            smoothing: float, contact_distance: float, influence_radius: float, magnitude_scale: float,
            apply: bool = False) -> None:
        """Compute one step's displacements for the points now on the device, into ``self.displacement``.

        With *apply* the displacements are also added to the points on the device.
        """
        self._update_stent(stent_vertices, box_padding)
        if self.sorted_box is None or self.steps_since_sort >= _STEPS_PER_SORT:
            self._sort(*self.box)
        self.steps_since_sort += 1

        # Two code paths, as in the reference: the uniform stent never pays for the radius profile
        if _is_uniform_stent(radius, cap_height_fraction):
            classify = self._uniform_classification(radius)
        else:
            classify = self._variable_radius_classification(stent_vertices, radius, cap_height_fraction)
        classify.set_param_by_name("smoothing", float(smoothing))
        classify.set_param_by_name("contact_distance", float(contact_distance))
        classify.launch()

        for fit_boxes in self.fit_boxes:
            fit_boxes.launch()

        displace = self.displace
        displace.set_param_by_name("influence_radius", float(influence_radius))
        displace.set_param_by_name("magnitude_scale", float(magnitude_scale))
        displace.launch()

        if apply:
            self.apply.launch()

    def _download(self, device_array) -> np.ndarray:
        """A host view of *device_array*, valid until the next download or step."""
        if self.staging is None:
            return device_array.numpy()  # the CPU device: a view of the array's own memory
        wp.copy(self.staging, device_array)
        wp.synchronize_device(self.device)
        return self.staging_host

    def download_displacements(self) -> np.ndarray:
        """The displacements of the last step, shape ``(N + M, 3)``; a view, see :meth:`_download`."""
        return self._download(self.displacement)

    def download_points(self) -> np.ndarray:
        """The points as they are on the device, shape ``(N + M, 3)``; a view, see :meth:`_download`."""
        return self._download(self.points)


# The buffers of the last drop-in step, reused by the next one on a mesh of the same size
_cached_step: Optional[_DeviceStep] = None


def clear_cache() -> None:
    """Release the device memory that :func:`sdf_contact_displacements` keeps between calls."""
    global _cached_step
    _cached_step = None


# ── Public entry points ──────────────────────────────────────────────────────

def capsule_chain_sdf(
    points, stent_vertices, radius, cap_height_fraction: float = 1.0, *,
    smoothing: float, device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the smooth-min capsule-chain stent SDF and its outward direction.

    The Warp counterpart of ``smin_sdf_capsule_contact_sculpt`` (scalar
    *radius* and ``cap_height_fraction == 1``) and of
    ``smin_sdf_variable_radius_capsule_contact_sculpt`` (otherwise) in
    :mod:`svmorph.core.deformation`.

    Parameters
    ----------
    points : array-like
        Query points, shape ``(N, 3)``.
    stent_vertices : array-like
        Stent axis vertices, shape ``(V, 3)``.
    radius : float or array-like
        Stent radius: scalar, or per-vertex radii of shape ``(V,)``.
    cap_height_fraction : float
        Axial semi-axis of the two stent end caps as a fraction of the end
        radius, in ``(0, 1]``.
    smoothing : float
        Smoothing radius ``k`` of the smooth minimum.
    device : str or None
        Warp device alias; by default the device of the backend in use.

    Returns
    -------
    distance : np.ndarray
        Signed distance to the stent surface, shape ``(N,)``, float32.
    direction : np.ndarray
        Outward direction from the stent axis, shape ``(N, 3)``, float32.
    """
    points = _as_points(points)
    number_of_points = len(points)
    if number_of_points == 0:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    device = _get_device(device)
    stent_vertices = _as_stent_vertices(stent_vertices)
    device_points = wp.array(points, dtype=wp.vec3, device=device)
    device_stent_vertices = wp.array(stent_vertices, dtype=wp.vec3, device=device)
    distance = wp.empty(number_of_points, dtype=wp.float32, device=device)
    direction = wp.empty(number_of_points, dtype=wp.vec3, device=device)
    if _is_uniform_stent(radius, cap_height_fraction):
        wp.launch(
            _uniform_sdf_kernel, dim=number_of_points,
            inputs=[device_points, device_stent_vertices, float(radius), float(smoothing)],
            outputs=[distance, direction], device=device)
    else:
        radius_profile = wp.array(_radius_profile(stent_vertices, radius), dtype=wp.vec3, device=device)
        wp.launch(
            _variable_radius_sdf_kernel, dim=number_of_points,
            inputs=[device_points, device_stent_vertices, radius_profile,
                    _cap_stretch(cap_height_fraction), float(smoothing)],
            outputs=[distance, direction], device=device)
    # numpy() of a CPU-device array is a view of memory the array owns, so both are copied
    return np.array(distance.numpy()), np.array(direction.numpy())


def sdf_contact_displacements(
    surface_points, centerline_points, stent_vertices, current_stent_radius, *,
    box_padding: float, cap_height_fraction: float, smoothing: float,
    contact_distance: float, influence_radius: float, magnitude_scale: float,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one SDF-contact deployment step: the displacement of every surface and centerline point.

    What :func:`svmorph.core.deformation.compute_sdf_contact_displacements`
    computes between reading the points from the simulation data and
    returning; that function documents the algorithm.

    Parameters
    ----------
    surface_points, centerline_points : array-like
        Mesh points, shapes ``(N, 3)`` and ``(M, 3)``.  Not modified.
    stent_vertices : array-like
        Stent axis vertices, shape ``(V, 3)``.
    current_stent_radius : float or array-like
        Current stent radius: scalar, or per-vertex radii of shape ``(V,)``.
    box_padding : float
        Distance by which the bounding box of the stent axis is grown; points
        outside the grown box are not evaluated.
    cap_height_fraction : float
        Axial semi-axis of the two stent end caps as a fraction of the end
        radius, in ``(0, 1]``.
    smoothing : float
        Smoothing radius ``k`` of the smooth minimum.
    contact_distance : float
        Distance threshold for stent-wall contact.
    influence_radius : float
        Distance from the contact front within which points are displaced.
    magnitude_scale : float
        Displacement of a point in contact on the stent surface, ``f_scale * (-s)``.
    device : str or None
        Warp device alias; by default the device of the backend in use.

    Returns
    -------
    surface_displacements : np.ndarray
        Shape ``(N, 3)``, float32.  All zero while nothing is in contact.
    centerline_displacements : np.ndarray
        Shape ``(M, 3)``, float32.
    """
    global _cached_step

    surface_points = _as_points(surface_points)
    centerline_points = _as_points(centerline_points)
    number_of_surface_points = len(surface_points)
    number_of_centerline_points = len(centerline_points)
    if number_of_surface_points == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((number_of_centerline_points, 3), dtype=np.float32))

    device = _get_device(device)
    stent_vertices = _as_stent_vertices(stent_vertices)
    step = _cached_step
    if step is None or not step.fits(
            device, number_of_surface_points, number_of_centerline_points, len(stent_vertices)):
        step = _cached_step = _DeviceStep(
            device, number_of_surface_points, number_of_centerline_points, len(stent_vertices))

    step.upload_points(surface_points, centerline_points)
    step.run(stent_vertices, current_stent_radius, box_padding=box_padding,
             cap_height_fraction=cap_height_fraction, smoothing=smoothing,
             contact_distance=contact_distance, influence_radius=influence_radius,
             magnitude_scale=magnitude_scale)
    # The download is a view of memory the next step overwrites; the caller gets arrays of its own
    displacements = step.download_displacements()
    return (displacements[:number_of_surface_points].copy(),
            displacements[number_of_surface_points:].copy())


class DeviceDeployment:
    """A stent deployment whose mesh stays on the device from one step to the next.

    The mesh is uploaded when the first step is taken.  Each :meth:`step` then
    computes the displacements and adds them to the points on the device,
    without any transfer of the mesh; :meth:`sync` copies the positions back
    into the host arrays.  Between a step and the next ``sync`` the host arrays
    are therefore out of date, and changes made to them are not seen by the
    device: after changing them, start a new deployment.

    Parameters
    ----------
    surface_points, centerline_points : np.ndarray
        The mesh, shapes ``(N, 3)`` and ``(M, 3)``.  They must be C-contiguous
        float32 arrays (as the point buffers of a VTK mesh are), because
        :meth:`sync` writes the positions back into them in place.
    device : str or None
        Warp device alias; by default the device of the backend in use.
    """

    def __init__(self, surface_points: np.ndarray, centerline_points: np.ndarray,
                 device: Optional[str] = None):
        for name, points in (("surface_points", surface_points), ("centerline_points", centerline_points)):
            if not self.supports(points):
                raise TypeError(f"{name} must be a C-contiguous float32 array of shape (N, 3)")
        self._surface_points = surface_points
        self._centerline_points = centerline_points
        self._device = _get_device(device)
        self._step: Optional[_DeviceStep] = None
        self._host_is_current = True

    @staticmethod
    def supports(points) -> bool:
        """Whether *points* is an array the positions can be written back into in place."""
        return (isinstance(points, np.ndarray) and points.dtype == np.float32 and points.ndim == 2
                and points.shape[1] == 3 and points.flags.c_contiguous and points.flags.writeable)

    def step(self, stent_vertices, current_stent_radius, *, box_padding: float, cap_height_fraction: float,
             smoothing: float, contact_distance: float, influence_radius: float,
             magnitude_scale: float) -> None:
        """Advance the deployment by one step on the device.

        The arguments are those of :func:`sdf_contact_displacements`.
        """
        number_of_surface_points = len(self._surface_points)
        if number_of_surface_points == 0:
            return
        stent_vertices = _as_stent_vertices(stent_vertices)
        sizes = (self._device, number_of_surface_points, len(self._centerline_points), len(stent_vertices))
        if self._step is None or not self._step.fits(*sizes):
            # A stent axis with another number of vertices needs new buffers; the mesh goes
            # through the host on its way into them
            self.sync()
            self._step = _DeviceStep(*sizes)
            self._step.upload_points(self._surface_points, self._centerline_points)
        self._step.run(stent_vertices, current_stent_radius, box_padding=box_padding,
                       cap_height_fraction=cap_height_fraction, smoothing=smoothing,
                       contact_distance=contact_distance, influence_radius=influence_radius,
                       magnitude_scale=magnitude_scale, apply=True)
        self._host_is_current = False

    def sync(self) -> None:
        """Copy the positions from the device into the host arrays, in place."""
        if self._host_is_current or self._step is None:
            return
        points = self._step.download_points()
        number_of_surface_points = len(self._surface_points)
        self._surface_points[:] = points[:number_of_surface_points]
        self._centerline_points[:] = points[number_of_surface_points:]
        self._host_is_current = True
