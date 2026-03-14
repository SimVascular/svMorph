import numpy as np
from scipy.spatial import cKDTree

import jax as jx
import jax.numpy as jnp
import time


# @jx.jit
def compute_householder_matrices(cross_section_normals):
    z_axis = jnp.array([0, 0, 1])
    a_minus_b = cross_section_normals - z_axis
    denominator = jnp.linalg.norm(a_minus_b, axis=1) ** 2
    # print("denominator=", denominator)
    # print("numerator=", jnp.einsum('ij,ik->ijk', a_minus_b, a_minus_b))
    projection_matrix = 2 * jnp.einsum('ij,ik->ijk', a_minus_b, a_minus_b) / denominator[:, None, None]
    householder_matrices = jnp.eye(3) - projection_matrix
    # print("householder_matrices=", householder_matrices)
    # print("to check each householder matrix is orthogonal, we need to check if the dot product of each householder matrix with its transpose is the identity matrix")
    # print("dot product of each householder matrix with its transpose=", jnp.einsum('ijk,ikl->ijl', householder_matrices, jnp.transpose(householder_matrices, axes=(0, 2, 1))))
    # assert jnp.allclose(jnp.einsum('ijk,ikl->ijl', householder_matrices, jnp.transpose(householder_matrices, axes=(0, 2, 1))), jnp.eye(3), atol=1e-3)
    return householder_matrices

def set_node_indices(data, list_of_node_point_indices):
    data["nodes"]["all_indices"] = jnp.array(list_of_node_point_indices)
    return data

def set_force_center(data, point_id):
    # assert(point_id in data["nodes"]["all_indices"])
    data["nodes"]["force_center_point_id"] = point_id
    return data

def interface_falloff(x, w_prime):
    power = 8
    # n = 10
    # second_term = (1/(1+n*x))**2
    return 1 / w_prime**power * (x - w_prime)**power 

def mix(a, b, t):
    return a + (b - a) * t

def smin_and_gradient(a, da, b, db, k=0.01):
    k = k * 4.0
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    n = 0.5 * h
    m = h**2 * k / 4.0
    # Use jnp.where to choose between the two cases in a jittable way
    value = jnp.where(a < b, a - m, b - m)
    grad  = jnp.where(a < b, mix(da, db, n), mix(da, db, 1.0 - n))
    return value, grad

def fold_smin(carry, elem):
    cur_min_d, cur_min_dir = carry 
    d, dir = elem      # new distance and direction to combine
    new_d, new_dir = smin_and_gradient(cur_min_d, cur_min_dir, d, dir)
    return (new_d, new_dir), None

def compute_min_dist_and_direction(d, dir):
    # d: (num_segments,), dir: (num_segments, ndims)
    (final_d, final_dir), _ = jx.lax.scan(fold_smin, (d[0], dir[0]), (d[1:], dir[1:]))
    return final_d, final_dir

@jx.jit
def capsule_sdf(p, stent_vertices, r):
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = p - stent_vertices[None, :-1, :]
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
    sdf = final_dist_to_surface
    return sdf

def kelvinlets_stent_edge(rv, a, b, eps, s, direction, w, r_target):
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # Extract components of rv
    f_scale = 0.5
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    cap_height_vector = jnp.maximum(0, -direction * rz - w)
    # interface_ratio = 0.5
    # w_prime = w * interface_ratio
    w_prime = 0.55555
    # cap_interface_cutoff_mask = (cap_height_vector < w_prime).astype(int)
    cap_height_vector = jnp.minimum(cap_height_vector, w_prime)
    cap_interface_falloff_mask = interface_falloff(cap_height_vector, w_prime)
    # rz = linear_heaviside(rz, w)
    # rz = linear_heaviside_stent_edge(rz) * 0
    # to convert rz into a mask that is 1 if rz is less than w and 0 otherwise
    # mask = (rz < 1e-6).astype(int) # consider using 1e-4 also possible
    asymmetry_mask = (-direction * rz < 0).astype(int)
    rz = rz * asymmetry_mask
    rv = rv.at[:, :, 2].set(rz)
    # re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    re = jnp.sqrt(rx**2 + ry**2 + rz**2)
    fall_off_mask = (re <= r_target).astype(int)
    assert re.shape == (num_mesh_points, num_kelvinlet_points)
    assert fall_off_mask.shape == (num_mesh_points, 1)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # Compute powers of re for the displacement formula
    # re3 = re**3
    # re5 = re**5
    # Calculate displacements
    # rv = rv.at[:, :, 2].set(0 * rv[:, :, 2])
    
    displacements = f_scale * r_target * ((re / r_target) ** 2 - 1) ** 2 * (-s) * rv
    # displacements = displacements * fall_off_mask[:, :, None] * cylinder_mask[:, :, None]
    displacements = displacements * fall_off_mask[:, :, None]
    displacements = displacements * cap_interface_falloff_mask[:, :, None]
    # displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)

    return displacements

def kelvinlets_truncated_sphere_warp_shrink(rv, a, b, eps, f_scale, s, r_min, r_max, r_original):
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # Extract components of rv
    # r_target = 0.05
    # r_max = 0.3
    # eps = 0.001
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    # Compute re with epsilon added
    # re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    re = jnp.sqrt(rx**2 + ry**2 + rz**2)
    re_no_z = jnp.sqrt(rx**2 + ry**2)
    # rz_magnitude = jnp.sqrt(rz**2)
    inner_mask = (re_no_z >= r_min).astype(int)
    outer_mask = (re <= r_max).astype(int)
    # vessel_selection_mask = (re_no_z <= r_original * 2.0 ).astype(int)
    assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # assert mask.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # Compute powers of re for the displacement formula
    # re3 = re**3
    # re5 = re**5
    # Calculate displacements
    rv = rv.at[:, :, 2].set(0 * rv[:, :, 2])
    # displacements = f_scale * (r_max - r_target) * (((re-r_max) / (r_target-r_max)) ** 2 - 1) ** 2 * (-s) * rv
    displacements = f_scale * (r_max - r_min) * ((re / (r_max)) ** 2 - 1) ** 2 * (-s) * rv
    displacements = displacements * inner_mask[:, :, None] * outer_mask[:, :, None]
    # displacements = f_scale * (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)
    # print("average_cross_section_displacement_distance: ", average_cross_section_displacement_distance)
    return displacements

def kelvinlets_truncated_sphere_warp_sculp(rv, a, b, eps, s, r_target):
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # Extract components of rv
    f_scale = 0.01
    # eps = 0.001
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    # Compute re with epsilon added
    # re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    re = jnp.sqrt(rx**2 + ry**2 + rz**2)
    # mask = (re <= r_target).astype(int)
    assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # assert mask.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # Compute powers of re for the displacement formula
    re3 = re**3
    re5 = re**5
    # Calculate displacements
    rv = rv.at[:, :, 2].set(0 * rv[:, :, 2])
    
    # displacements = f_scale * r_target * ((re / r_target) ** 2 - 1) ** 2 * (-s) * rv
    # displacements = displacements * mask[:, :, None]
    displacements = f_scale * (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)
    print("displacements norms: ", jnp.linalg.norm(displacements))

    return displacements

@jx.jit
def smin_sdf_capsule_contact_sculp(rv, a, b, stent_vertices, eps, s, r_target, r_current, doi, doc):
    # doi = 0.65 # distance of influence: width of the deformation zone
    # doc = 0.01 # distance within which contact is made
    # doi = 0.15
    # f_scale = 0.25 * doi * 0.1
    # f_scale = 0.25 * doi
    # rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = rv - stent_vertices[None, :-1, :]
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
    # mask = (final_dist_to_surface < doi).astype(int)
    # new_contact_mask = (final_dist_to_surface < doc).astype(bool)
    
    return final_dist_to_surface, final_direction

@jx.jit #TODO: comment/uncomment this to print kelvinlet quantities
def get_affine_laplacian_displacements_inner(data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor, w, r_target):
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    # Compute rv in the local frame
    rv = xs - centers
    # Rotate rv to the global frame
    rotation_matrices = jnp.expand_dims(rotation_matrices, 0)
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    # Compute Kelvinlet displacements
    average_displacement_distance = 0
    displacement_local = kelvinlets_truncated_sphere_warp_sculp(centerline_aligned_rv, a, b, eps, s, r_target)
    # displacement_local = kelvinlets_affine_laplacian(centerline_aligned_rv, a, b, eps, s, 1, 0.8)
    displacement_global = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    # Aggregate and normalize TODO: below use of sum is unnecessary if there is only 1 kelvinlet point
    displacement = jnp.sum(displacement_global, axis=1)
    # Scale if required
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
        average_displacement_distance *= surface_mesh_scale_factor
    return displacement, average_displacement_distance

@jx.jit
def get_stent_edge_displacements_inner(data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor, direction, w, r_target):
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    # Compute rv in the local frame
    rv = xs - centers
    # Rotate rv to the global frame
    rotation_matrices = jnp.expand_dims(rotation_matrices, 0)
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    # Compute Kelvinlet displacements
    displacement_local = kelvinlets_stent_edge(centerline_aligned_rv, a, b, eps, s, direction, w, r_target)
    displacement_global = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    # Aggregate and normalize
    displacement = jnp.sum(displacement_global, axis=1)
    # Scale if required
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
    return displacement

def find_stenosis_minimum_radius_representative(data_points, rotation_matrices, xs, centers, original_radius):
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    centers = np.tile(centers, (num_mesh_points, 1, 1))
    # Compute rv in the local frame
    rv = xs - centers
    # Rotate rv to the global frame
    rotation_matrices = np.expand_dims(rotation_matrices, 0)
    rv = np.einsum('...ij,...j->...i', rotation_matrices, rv)
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    rz_magnitude = np.sqrt(rz**2)
    radial_magnitude_squared = (rx**2 + ry**2)

    # Estimate the actual current vessel radius:
    # 1. Euclidean nearest neighbors → local surface points
    # 2. Smallest |rz| among those → points on the cross-section plane
    # Their radial distances give the current vessel radius, even after
    # deformation.  This naturally excludes outlet cap points (which have
    # nonzero |rz|) unless the center is exactly on the cap.
    euclidean_dist_sq = rx[:, 0]**2 + ry[:, 0]**2 + rz[:, 0]**2
    nearest_dist = np.sqrt(np.min(euclidean_dist_sq))
    nearby = euclidean_dist_sq < (nearest_dist * 1.5)**2
    nearby_rz_mag = np.abs(rz[nearby, 0])
    nearby_radial = np.sqrt(radial_magnitude_squared[nearby, 0])
    rz_cutoff = np.percentile(nearby_rz_mag, 5)
    on_plane = nearby_rz_mag <= rz_cutoff
    if np.any(on_plane):
        current_radius = float(np.median(nearby_radial[on_plane]))
    else:
        current_radius = float(original_radius)

    mask = ((current_radius * 1.0) ** 2 <= radial_magnitude_squared) * (radial_magnitude_squared <= (current_radius * 1.1) ** 2)
    # Set rz_magnitude to a large value where mask is False so they are not selected as min
    rz_magnitude_masked = np.where(mask, rz_magnitude, jnp.inf)
    index_for_min_rz = np.argmin(rz_magnitude_masked, axis=0)
    print("index_for_min_rz: ", index_for_min_rz)
    print("rx, ry, rz for min_rz: ", rx[index_for_min_rz], ry[index_for_min_rz], rz[index_for_min_rz])
    print("estimated current radius: ", current_radius)
    return index_for_min_rz, current_radius

@jx.jit #TODO: comment/uncomment this to print kelvinlet quantities
def get_stenosis_displacements_inner(data_points, rotation_matrices, xs, centers, a, b, eps, s, r_min, r_max, r_original):
    r_target = 0.05
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    # Compute rv in the local frame
    rv = xs - centers
    # Rotate rv to the global frame
    rotation_matrices = jnp.expand_dims(rotation_matrices, 0)
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    # Compute Kelvinlet displacements
    f_scale = 0.01 / (r_max - r_min)
    step_size = f_scale * (r_max - r_min) * s
    # print("step_size: ", step_size)
    displacement_local = kelvinlets_truncated_sphere_warp_shrink(centerline_aligned_rv, a, b, eps, f_scale, s, r_min, r_max, r_original)
    # displacement_local = kelvinlets_affine_laplacian(centerline_aligned_rv, a, b, eps, s, 1, 0.8)
    displacement_global = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    # Aggregate and normalize TODO: below use of sum is unnecessary if there is only 1 kelvinlet point
    displacement = jnp.sum(displacement_global, axis=1)
    return displacement, step_size

def compute_aneurysm_displacements(data, a, b, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, stent_radius):
    # Resolve all_indices and force_center_point_id outside JIT
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("force center: ", force_center_point_id)
    # Prepare other data
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    print("num_kelvinlet_points: ", num_kelvinlet_points)
    print("xs shape: ", xs.shape)
    centers = jnp.expand_dims(jnp.array([centerline_points[force_center_point_id]]), 0)
    kelvinlet_points_normals = jnp.array([force_center_normal])
    rotation_matrices = compute_householder_matrices(kelvinlet_points_normals)
    # Call the JIT-compiled function
    displacements, average_displacement_distance = get_affine_laplacian_displacements_inner(
        data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor, stent_halflength, stent_radius
    )
    # print("displacement: ", displacements)
    # print("to see which entry of displacement has a large numerical entry: ")
    # large_entries = jnp.where(jnp.abs(displacements) > 1)
    # print("large entries: ", large_entries)
    # print("the magnitude of the displacement is: ", jnp.linalg.norm(displacements))
    # displacement = get_affine_displacements_inner(
    #     data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    # ) / num_kelvinlet_points
    # xs = jnp.expand_dims(centerline_points, 1)
    # xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    # print("centerline xs shape: ", xs.shape)
    # centerline_displacements = get_affine_displacements_inner(
    #     centerline_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    # )
    # print("average_displacement_distance = ", average_displacement_distance)
    return np.array(displacements), average_displacement_distance#, centerline_displacements

def compute_stent_edge_displacements(data, a, b, eps, s, surface_mesh_scale_factor, force_center_normal, direction, w, r_target):
    # Resolve all_indices and force_center_point_id outside JIT
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("force center: ", force_center_point_id)
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    centers = jnp.expand_dims(jnp.array([centerline_points[force_center_point_id]]), 0)
    kelvinlet_points_normals = jnp.array([force_center_normal])
    rotation_matrices = compute_householder_matrices(kelvinlet_points_normals)
    # Call the JIT-compiled function
    displacements = get_stent_edge_displacements_inner(
        data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor, direction, w, r_target
    )
    # print("displacement: ", displacements)
    # print("to see which entry of displacement has a large numerical entry: ")
    # large_entries = jnp.where(jnp.abs(displacements) > 1)
    # print("large entries: ", large_entries)
    # print("the magnitude of the displacement is: ", jnp.linalg.norm(displacements))
    # displacement = get_affine_displacements_inner(
    #     data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    # ) / num_kelvinlet_points
    # xs = jnp.expand_dims(centerline_points, 1)
    # xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    # print("centerline xs shape: ", xs.shape)
    # centerline_displacements = get_affine_displacements_inner(
    #     centerline_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    # )
    return displacements#, centerline_displacements

def compute_stenosis_displacements(data, a, b, eps, s, force_center_normal, r_min, r_max, r_original):
    # Resolve all_indices and force_center_point_id outside JIT
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("selected stenosis center point ID: ", force_center_point_id)
    # Prepare other data
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    # print("num_kelvinlet_points: ", num_kelvinlet_points)
    # print("xs shape: ", xs.shape)
    centers = jnp.expand_dims(jnp.array([centerline_points[force_center_point_id]]), 0)
    kelvinlet_points_normals = jnp.array([force_center_normal])
    rotation_matrices = compute_householder_matrices(kelvinlet_points_normals)
    # Call the JIT-compiled function
    displacements, step_size = get_stenosis_displacements_inner(
        data_points, rotation_matrices, xs, centers, a, b, eps, s, r_min, r_max, r_original
    )
    # print("to see which entry of displacement has a large numerical entry: ")
    # large_entries = jnp.where(jnp.abs(displacements) > 1)
    # print("large entries: ", large_entries)
    # print("the magnitude of the displacement is: ", jnp.linalg.norm(displacements))
    # displacement = get_affine_displacements_inner(
    #     data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    # ) / num_kelvinlet_points
    # xs = jnp.expand_dims(centerline_points, 1)
    # xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    # print("centerline xs shape: ", xs.shape)
    # centerline_displacements = get_affine_displacements_inner(
    #     centerline_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    # )
    # print("average_displacement_distance = ", average_displacement_distance)
    return np.array(displacements), step_size#, centerline_displacements

@jx.jit
def stent_bounding_box(data_points, stent_vertices, target_stent_radius, doi, doc):
    # Compute the minimum and maximum coordinates of the bounding box
    min_coords = jnp.min(stent_vertices, axis=0) - target_stent_radius - doi - doc - 0.01
    max_coords = jnp.max(stent_vertices, axis=0) + target_stent_radius + doi + doc + 0.01
    mask = jnp.all((data_points >= min_coords) & (data_points <= max_coords), axis=1)
    return mask

def compute_sdf_contact_displacements(data, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, target_stent_radius, current_stent_radius):
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("selected point ID: ", force_center_point_id)
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    # doi = 0.9
    doi = 0.65
    # doc = 0.01
    doc = 0.001
    # doc = 0.1
    # f_scale = 0.25 * doi * 0.1
    f_scale = 0.01
    start_time = time.time()
    sbb_mask = stent_bounding_box(data_points, stent_vertices, target_stent_radius, doi, doc)
    cbb_mask = stent_bounding_box(centerline_points, stent_vertices, target_stent_radius, doi, doc)
    print("time taken to compute bounding box: ", time.time() - start_time)
    start_time = time.time()
    sbb_mask = np.array(sbb_mask)
    cbb_mask = np.array(cbb_mask)
    data_points = np.array(data_points)
    centerline_points = np.array(centerline_points)
    data_points_masked = data_points[sbb_mask]
    centerline_points_masked = centerline_points[cbb_mask]
    print("time taken to cast to numpy and mask out bounding box: ", time.time() - start_time)
    num_mesh_points = data_points.shape[0]
    num_centerline_points = centerline_points.shape[0]
    num_in_bb_mesh_points = data_points_masked.shape[0]
    data_and_centerline_points_masked = np.concatenate((data_points_masked, centerline_points_masked), axis=0)
    # data_and_centerline_points_masked = data_points_masked
    xs_np = np.expand_dims(data_and_centerline_points_masked, 1)
    # xs_np: (num_total_data_points, 1, 3)
    xs = np.tile(xs_np, (1, num_kelvinlet_points, 1))
    # xs = jnp.array(xs)
    # xs = jnp.expand_dims(data_points, 1)
    # xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
   
    start_time = time.time()
    total_num_vertices = xs.shape[0]
    print("num surface points and centerline points combined: ", total_num_vertices)
    combined_final_dist_to_surface, combined_final_direction = smin_sdf_capsule_contact_sculp(xs, a, b, stent_vertices, eps, s, target_stent_radius, current_stent_radius, doi, doc) #JIT-compiled
    combined_final_dist_to_surface = np.array(combined_final_dist_to_surface)
    combined_final_direction = np.array(combined_final_direction)

    final_dist_to_surface = combined_final_dist_to_surface[:num_in_bb_mesh_points]
    final_direction = combined_final_direction[:num_in_bb_mesh_points]
    new_contact_mask = (final_dist_to_surface < doc).astype(bool)
    # print("new_contact_mask shape: ", new_contact_mask.shape)
    print("time taken to compute new_contact points: ", time.time() - start_time)
    centerline_points_dist_to_surface = combined_final_dist_to_surface[num_in_bb_mesh_points:]
    centerline_points_final_direction = combined_final_direction[num_in_bb_mesh_points:]
    centerline_outside_stent_mask = (centerline_points_dist_to_surface[:, 0] > 0).astype(bool)
    # print("centerline_points_masked shape: ", centerline_points_masked.shape)
    # print("centerline_outside_stent_mask shape: ", centerline_outside_stent_mask.shape)
    movables_centerline_points = centerline_points_masked[centerline_outside_stent_mask]
    movables_centerline_points_dist_to_surface = centerline_points_dist_to_surface[centerline_outside_stent_mask]
    movables_centerline_points_final_direction = centerline_points_final_direction[centerline_outside_stent_mask]
    final_movables_dist_to_surface = np.concatenate((final_dist_to_surface, movables_centerline_points_dist_to_surface))
    final_movables_direction = np.concatenate((final_direction, movables_centerline_points_final_direction))
    num_final_movables = final_movables_dist_to_surface.shape[0]
    full_centerline_points_mask = np.zeros(num_centerline_points, dtype=bool)
    full_centerline_points_mask[cbb_mask] = centerline_outside_stent_mask
    
    # surface_xs = xs_np[:num_in_bb_mesh_points]
    start_time = time.time()
    in_contact_vertices = data_points_masked[new_contact_mask[:,0]]
    # print("in_contact_vertices shape: ", in_contact_vertices.shape)
    print("time taken to obtain in_contact vertices subslice: ", time.time() - start_time)
    if in_contact_vertices.shape[0] == 0: # things are in contact <=> things are in influence
        step_size = f_scale * (-s)
        return np.zeros((num_mesh_points, 3)), np.zeros((num_centerline_points, 3)), step_size 
    
    start_time = time.time()
    contact_tree = cKDTree(in_contact_vertices, leafsize=32)
    xs = np.concatenate((data_points_masked, movables_centerline_points), axis=0)
    dist_min, _ = contact_tree.query(xs, k=1, distance_upper_bound=doi, workers=-1)
    # print("dist_min shape: ", dist_min.shape)
    print("time for KD Tree construction and query: ", time.time() - start_time)
    start_time = time.time()
    in_influence_mask = dist_min < doi
    # print("in_influence_mask shape: ", in_influence_mask.shape)
    in_influence_indices = np.flatnonzero(in_influence_mask)
    # print("in_influence_indices shape: ", in_influence_indices.shape)
    print("time taken to flattennonzero: ", time.time() - start_time)
    
    start_time = time.time()
    in_influence_to_in_contact_distances = dist_min[in_influence_mask]
    print("time taken to obtain in-influence vertices: ", time.time() - start_time)
    part_two_start_time = time.time()
    
    print("time for converting to jnp arrays: ", time.time() - part_two_start_time)
    start_time = time.time()

    doi_mask = (final_movables_dist_to_surface < doi).astype(int)

    in_influence_vertices_blended_alpha_mask = np.zeros(num_final_movables)
    # in_influence_vertices_blended_alpha = smin_sdf_capsule_contact_sculp_part_two_KD(in_influence_vertices, in_contact_vertices, doi)
    in_influence_vertices_blended_alpha = (1 - in_influence_to_in_contact_distances / doi)  # linear blending
    # in_influence_vertices_blended_alpha = (1 - (in_influence_to_in_contact_distances / doi) ** 2) ** 2 # quadratic blending
    # in_influence_vertices_blended_alpha = in_influence_to_in_contact_distances / in_influence_to_in_contact_distances # no blending
    # k = -2.0
    # in_influence_vertices_blended_alpha = (np.exp(k*in_influence_to_in_contact_distances) - np.exp(k*doi)) / (1 - np.exp(k*doi))  # exponential blending

    print("time taken to compute JIT sculpt part two: ", time.time() - start_time)
    start_time = time.time()
    in_influence_vertices_blended_alpha = np.array(in_influence_vertices_blended_alpha)
    in_influence_vertices_blended_alpha_mask[in_influence_indices] = in_influence_vertices_blended_alpha
    print("time taken to compute blended alpha mask in np: ", time.time() - start_time)
    start_time = time.time()
    print("raw number of negative values in final_movables_dist_to_surface: ", np.sum(final_movables_dist_to_surface < 0))
    interior_points_offset = np.maximum(-final_movables_dist_to_surface, 0.0)
    final_movables_dist_to_surface = np.maximum(final_movables_dist_to_surface, 0.0)  # Clip the interior points to the surface
    # displacements = f_scale * ((final_movables_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * doi_mask * in_influence_vertices_blended_alpha_mask[:, None] * final_movables_direction
    displacements_magnitude = f_scale * ((final_movables_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * doi_mask * in_influence_vertices_blended_alpha_mask[:, None] 
    displacements_magnitude += interior_points_offset
    displacements = displacements_magnitude * final_movables_direction
    full_surface_displacements = np.zeros((num_mesh_points, 3))
    full_surface_displacements[sbb_mask] = displacements[:num_in_bb_mesh_points]
    full_centerline_displacements = np.zeros((num_centerline_points, 3))
    full_centerline_displacements[full_centerline_points_mask] = displacements[num_in_bb_mesh_points:]
    # displacements = f_scale * force_kernel(1.99, 0.88, 2.3, final_dist_to_surface) * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    step_size = f_scale * (-s)
    print("time taken to compute rest of the displacements: ", time.time() - start_time)   

    return full_surface_displacements, full_centerline_displacements, step_size
