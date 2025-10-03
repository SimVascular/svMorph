# import os
import sys
import vtk
# import math
import copy
# from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

from kelvinlet_core import vtk_utils
from kelvinlet_core import common
from kelvinlet_core import ring_points_optimizer

import jax as jx
import jax.numpy as jnp
# Profiling to check the bottleneck
import cProfile
import pstats
from functools import wraps
import time

def profile_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Print profiling results
        ps = pstats.Stats(profiler)
        ps.strip_dirs().sort_stats("cumulative").print_stats(10)
        
        return result
    return wrapper

@jx.jit
def compute_householder_matrix(cross_section_normal):
    z_axis = jnp.array([0, 0, 1])
    a_minus_b = cross_section_normal - z_axis
    denominator = jnp.linalg.norm(a_minus_b) ** 2
    projection_matrix = 2 * jnp.matmul(a_minus_b, jnp.transpose(a_minus_b)) / denominator
    householder_matrix = jnp.eye(3) - projection_matrix
    return householder_matrix
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

@jx.jit
def compute_rotation_matrix(cartesian_axis_vector, centerline_axis_vector):
    a_plus_b = cartesian_axis_vector + centerline_axis_vector
    denominator = jnp.expand_dims(jnp.linalg.norm(a_plus_b, axis=1) ** 2, 2)
    rotation_matrix = 2 * jnp.matmul(a_plus_b, jnp.transpose(a_plus_b, axes=(0, 2, 1))) / denominator
    rotation_matrix -= jnp.eye(3)
    return rotation_matrix

def define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas):
    # Convert to JAX-compatible arrays by using jnp.array
    centerline_points = jnp.array(copy.deepcopy(v2n(centerline_polydata.GetPoints().GetData())))
    surface_points = jnp.array(copy.deepcopy(v2n(surface_polydata.GetPoints().GetData())))
    # Check if points have the required shape
    assert centerline_points.shape[1] == 3  # Ensure (x, y, z) coordinates
    assert surface_points.shape[1] == 3

    # Create a dictionary to store data, including the JAX arrays
    data = {
        "points": {
            "centerline": centerline_points,
            "surface": surface_points
        },
        "nodes": {
            "all_indices": [],
            "force_center_point_id": -1
        },
        "centerline_coordinate": jnp.array([])
    }
    # Check for the "centerline_coordinate" array and convert if available
    if centerline_polydata.GetPointData().HasArray("centerline_coordinate"):
        num_centerline_points = data["points"]["centerline"].shape[0]
        data["centerline_coordinate"] = jnp.array(copy.deepcopy(
            v2n(centerline_polydata.GetPointData().GetArray("centerline_coordinate"))
        ))
        assert data["centerline_coordinate"].shape[0] == num_centerline_points
    else:
        print("We sys exited because there was no centerline coords")
        # sys.exit("'centerline_coordinate' is not a point array on the centerline polydata.")
    # Process and add other geometry points as JAX arrays
    for ig, polydata in enumerate(other_geometry_polydatas):
        other_geometry_points = jnp.array(copy.deepcopy(v2n(polydata.GetPoints().GetData())))
        assert other_geometry_points.shape[1] == 3
        data["points"][f"other_geometry_{ig}"] = other_geometry_points
    return data

def define_points_affine_jonathan(centerline_polydata, surface_polydata, other_geometry_polydatas):
    centerline_points = copy.deepcopy(v2n(centerline_polydata.GetPoints().GetData()))
    surface_points = copy.deepcopy(v2n(surface_polydata.GetPoints().GetData()))
    
    assert(centerline_points.shape[1] == 3) # (x, y, z) coordinates
    assert(surface_points.shape[1] == 3)
    
    data = {"points" : {"centerline" : centerline_points, "surface" : surface_points}, 
            "nodes" : {"all_indices" : [], "force_center_point_id" : -1},
            "centerline_coordinate" : np.array([]) }
    
    if centerline_polydata.GetPointData().HasArray("centerline_coordinate"):
        num_centerline_points = data["points"]["centerline"].shape[0]
        data["centerline_coordinate"] = copy.deepcopy(v2n(centerline_polydata.GetPointData().GetArray("centerline_coordinate")))
        assert(data["centerline_coordinate"].shape[0] == num_centerline_points)
    else:
        sys.exit("'centerline_coordinate' is not a point array on the centerline polydata.")
    
    for ig in range(len(other_geometry_polydatas)):
        other_geometry_points = copy.deepcopy(v2n(other_geometry_polydatas[ig].GetPoints().GetData()))
        assert(other_geometry_points.shape[1] == 3)
        data["points"]["other_geometry_" + str(ig)] = other_geometry_points
    
    return data

def define_nodes_affine(data, list_of_node_point_indices):
    data["nodes"]["all_indices"] = jnp.array(list_of_node_point_indices)
    return data

def add_node_data_to_centerline_polydata_affine(data, centerline_polydata):
    # TODO: this can be potentially deprecated as it is not useful anymore
    num_centerline_points = data["points"]["centerline"].shape[0]
    # Create a VTK array directly to store node information
    is_node = vtk.vtkIntArray()
    is_node.SetNumberOfComponents(1)
    is_node.SetNumberOfTuples(num_centerline_points)
    is_node.SetName("nodes")
    # Initialize array with zeros and update specific indices directly
    for i in range(num_centerline_points):
        is_node.SetValue(i, 0)  # Set all values to 0 initially
    for idx in data["nodes"]["all_indices"]:
        is_node.SetValue(idx, 1)  # Set specified indices to 1
    is_node.SetValue(data["nodes"]["force_center_point_id"], 2)  # Set force center point to 2
    # Add the "nodes" array directly to the VTK polydata
    centerline_polydata.GetPointData().AddArray(is_node)
    return centerline_polydata

def add_node_data_to_centerline_polydata_affine_jonathan(data, centerline_polydata):
    num_centerline_points = data["points"]["centerline"].shape[0]
    is_node = np.zeros(num_centerline_points)
    is_node[data["nodes"]["all_indices"]] = 1
    is_node[data["nodes"]["force_center_point_id"]] = 2
    centerline_polydata = vtk_utils.add_point_data_array_to_polydata(centerline_polydata, "nodes", is_node)
    return centerline_polydata

def assign_force_location_affine_v2(data, point_id):
    # assert(point_id in data["nodes"]["all_indices"])
    data["nodes"]["force_center_point_id"] = point_id
    return data

def get_force_matrix_scale(scale, a, b):
    return scale * (2 / 5) / (2 * b - a)

def linear_heaviside(x, w):
    alpha = 300
    return 0.5 * (1 + jnp.tanh(alpha*(jnp.abs(x)-w))) * x

def linear_heaviside_stent_edge(x):
    alpha = 20
    w = 0.2
    return 0.5 * (1 + jnp.tanh(alpha*(jnp.abs(x)-w)))

def interface_falloff(x, w_prime):
    power = 8
    # n = 10
    # second_term = (1/(1+n*x))**2
    return 1 / w_prime**power * (x - w_prime)**power 

def regularize_origin(r):
    return 0
    h = 100
    gamma_origin = 500
    r_0 = 0
    return h * (1 - 1 / (1 + jnp.exp(-gamma_origin * (r - r_0))))

def sigmoid_truncation(z, d):
    h = 1
    gamma = 10000
    w = 0.2
    return h * (1 - 1 / (1 + jnp.exp(-gamma * (-d * z - w))))

def regularize_radius(r):
    gamma_singularity = 0.001
    # gamma_singularity = 0.01
    r_1 = 0.5
    # r_1 = 0.8
    return regularize_origin(r) + r + jnp.sqrt((r - r_1)**2 + gamma_singularity**2) / 2

def mix(a, b, t):
    return a + (b - a) * t

def smin(a, b, k):
    k *= 4.0
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h**2 * k / 4.0 

def branching_smin_and_gradient(a, da, b, db, k):
    k *= 4.0
    h = max(k - abs(a - b), 0.0) / k
    n = 0.5 * h
    m = h**2 * k / 4.0
    return a - m, mix(da, db, n) if a < b else b - m, mix(da, db, 1.-n)

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

def kelvinlets_affine_laplacian_commentedout(rv, a, b, eps, s, w, r_target):
    # Ensure the input tensor has the correct dimensions
    print(f"rv shape: {rv.shape}")
    # assert rv.ndim == 3
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # assert ndims == 3
    print(f"num_mesh_points: {num_mesh_points}, num_kelvinlet_points: {num_kelvinlet_points}, ndims: {ndims}")

    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # Extract components of rv
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    rz = linear_heaviside(rz, w)
    rv = rv.at[:, :, 2].set(rz)
    # Compute re with epsilon added
    r = jnp.sqrt(rx**2 + ry**2 + rz**2)
    r = regularize_radius(r)
    print("r[21611:21613]", r[21611:21613])
    r2 = r**2
    re = jnp.sqrt(r2 + eps**2)
    # assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # r = jnp.expand_dims(r, 2)
    r2 = jnp.expand_dims(r2, 2)
    # Compute powers of re for the displacement formula
    re2 = re**2
    re7 = re**7
    re9 = re**9
    # Calculate displacements
    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    # displacements = ((-105*a*eps**4 / (2*re9)) - (b*(4*re2 - 5*(5*eps**2+2*r2))/re7) + (3*b*(4*re2-7*(7*eps**2+2*r2))*r2/re9) + (12*b*(7*eps**2+2*r2)/re7)) * s * rv
    displacements = ((b*(109*eps**2+34*r2-4*re2)/re7) + ((3*b*(4*re2-49*eps**2-14*r2)*r2 - 52.5*a*eps**4)/re9)) * s * rv
    # assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)
    return displacements

def kelvinlets_affine_laplacian_hard_cutoff_if_radius_over(rv, a, b, eps, s, w, r_target): #radius bounded this causes self-intersection (hard_cutoff_if_radius_over)
    # Ensure the input tensor has the correct dimensions
    print(f"rv shape: {rv.shape}")
    # assert rv.ndim == 3
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # assert ndims == 3
    print(f"num_mesh_points: {num_mesh_points}, num_kelvinlet_points: {num_kelvinlet_points}, ndims: {ndims}")

    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # Extract components of rv
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    rz = linear_heaviside(rz, w)
    # to convert rz into a mask that is 1 if rz is less than w and 0 otherwise
    mask = (rz < 1e-6).astype(int) # consider using 1e-4 also possible
    rv = rv.at[:, :, 2].set(rz)
    # Compute re with epsilon added
    r = jnp.sqrt(rx**2 + ry**2 + rz**2)
    # r = regularize_radius(r)
    # print("r[21611:21613]", r[21611:21613])
    r2 = r**2
    re = jnp.sqrt(r2 + eps**2)
    # assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # r = jnp.expand_dims(r, 2)
    r2 = jnp.expand_dims(r2, 2)
    # Compute powers of re for the displacement formula
    re2 = re**2
    re7 = re**7
    re9 = re**9
    # Calculate displacements
    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    # displacements = ((-105*a*eps**4 / (2*re9)) - (b*(4*re2 - 5*(5*eps**2+2*r2))/re7) + (3*b*(4*re2-7*(7*eps**2+2*r2))*r2/re9) + (12*b*(7*eps**2+2*r2)/re7)) * s * rv
    displacements = ((b*(109*eps**2+34*r2-4*re2)/re7) + ((3*b*(4*re2-49*eps**2-14*r2)*r2 - 52.5*a*eps**4)/re9)) * s * rv
    # assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)
    print("displacements[21611:21613]", displacements[21611:21613])
    print("shape of displacements: ", displacements.shape)
    radius_mask = (r < 0.509).astype(int)
    
    # import matplotlib.pyplot as plt
    # # Convert r to a numpy array and flatten it for plotting.
    # r_np = np.array(r).flatten()
    # plt.hist(r_np, bins=200, edgecolor="black")
    # plt.xlabel("r")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of r values over small buckets")
    # plt.show()

    mask = mask * radius_mask
    displacements = displacements * mask[:, :, None]
    cylindrical_wall_displacements = displacements * mask[:, :, None]
    norms = jnp.linalg.norm(cylindrical_wall_displacements, axis=2)
    # nonzero_mask = (norms > 0.001).astype(int)
    norms_sum = jnp.sum(norms, axis=0)[0]
    # total_nonzero_entries = jnp.sum(nonzero_mask, axis=0)[0]
    total_nonzero_entries = jnp.sum(mask, axis=0)[0]
    print("total_nonzero_entries: ", total_nonzero_entries)
    safe_total_nonzero_entries = total_nonzero_entries + (total_nonzero_entries == 0).astype(int)
    average_displacement_distance = norms_sum / (safe_total_nonzero_entries) * (total_nonzero_entries > 0).astype(int)
    print("average_displacement_distance: ", average_displacement_distance)
    return displacements, average_displacement_distance

def kelvinlets_affine_laplacian(rv, a, b, eps, s, w, r_target): #radius bounded, stops the whole field at once, maybe no self-intersections, current best
    # Ensure the input tensor has the correct dimensions
    print(f"rv shape: {rv.shape}")
    # assert rv.ndim == 3
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # assert ndims == 3
    print(f"num_mesh_points: {num_mesh_points}, num_kelvinlet_points: {num_kelvinlet_points}, ndims: {ndims}")

    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # Extract components of rv
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    rz = linear_heaviside(rz, w)
    # to convert rz into a mask that is 1 if rz is less than w and 0 otherwise
    mask = (rz < 1e-6).astype(int) # consider using 1e-4 also possible
    rv = rv.at[:, :, 2].set(rz)
    # Compute re with epsilon added
    r = jnp.sqrt(rx**2 + ry**2 + rz**2)
    # r = regularize_radius(r)
    # print("r[21611:21613]", r[21611:21613])
    r2 = r**2
    re = jnp.sqrt(r2 + eps**2)
    # assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # r = jnp.expand_dims(r, 2)
    r2 = jnp.expand_dims(r2, 2)
    # Compute powers of re for the displacement formula
    re2 = re**2
    re7 = re**7
    re9 = re**9
    # Calculate displacements
    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    # displacements = ((-105*a*eps**4 / (2*re9)) - (b*(4*re2 - 5*(5*eps**2+2*r2))/re7) + (3*b*(4*re2-7*(7*eps**2+2*r2))*r2/re9) + (12*b*(7*eps**2+2*r2)/re7)) * s * rv
    displacements = ((b*(109*eps**2+34*r2-4*re2)/re7) + ((3*b*(4*re2-49*eps**2-14*r2)*r2 - 52.5*a*eps**4)/re9)) * s * rv
    # assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)
    # print("displacements[21611:21613]", displacements[21611:21613])
    # print("shape of displacements: ", displacements.shape)
    # radius_mask = (r < 0.509).astype(int)
    radius_mask = (r < r_target).astype(int)
    # import matplotlib.pyplot as plt
    # # Convert r to a numpy array and flatten it for plotting.
    # r_np = np.array(r).flatten()
    # plt.hist(r_np, bins=200, edgecolor="black")
    # plt.xlabel("r")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of r values over small buckets")
    # plt.show()
    mask = mask * radius_mask
    # displacements = displacements * mask[:, :, None]
    cylindrical_wall_displacements = displacements * mask[:, :, None]
    norms = jnp.linalg.norm(cylindrical_wall_displacements, axis=2)
    # nonzero_mask = (norms > 0.001).astype(int)
    norms_sum = jnp.sum(norms, axis=0)[0]
    # total_nonzero_entries = jnp.sum(nonzero_mask, axis=0)[0]
    total_nonzero_entries = jnp.sum(mask, axis=0)[0]
    print("total_nonzero_entries: ", total_nonzero_entries)
    displacement_continue_flag = 1 - (total_nonzero_entries == 0).astype(int)
    safe_total_nonzero_entries = total_nonzero_entries + (total_nonzero_entries == 0).astype(int)
    average_displacement_distance = norms_sum / (safe_total_nonzero_entries) * (total_nonzero_entries > 0).astype(int)
    print("average_displacement_distance: ", average_displacement_distance)
    displacements = displacements * displacement_continue_flag
    return displacements, average_displacement_distance

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

def sdf_capsule_warp_sculp(rv, a, b, stent_vertices, eps, s, r_target, r_current):
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # Extract components of rv
    doi = 0.15
    f_scale = 0.25 * doi 
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    # Compute re with epsilon added
    # re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    # re = jnp.sqrt(rx**2 + ry**2 + rz**2)
    # mask = (re <= r_target).astype(int)
    # assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # assert mask.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    # re = jnp.expand_dims(re, 2)
    # Compute powers of re for the displacement formula
    # re3 = re**3
    # re5 = re**5
    # Calculate displacements
    # a = jnp.array([0, 0, 0.2])
    # b = jnp.array([0, 0, -0.2])
    # ba = b - a
    # pa = rv - a
    # ba_dot_pa = jnp.sum(ba * pa, axis=2)
    # ba_dot_ba = jnp.dot(ba, ba)
    # h = jnp.clip(ba_dot_pa / ba_dot_ba, 0, 1)
    # axis_to_point = pa - h[:, :, None] * ba
    # dist = jnp.linalg.norm(axis_to_point, axis=2)[..., None]
    # direction = axis_to_point / dist
    # dist_to_surface = dist - r_current
    ba_all = jnp.diff(stent_vertices, axis=0)
    pa_all = rv - stent_vertices[None, :-1, :]
    print("ba_test shape: ", ba_all.shape)
    print("pa_test shape: ", pa_all.shape)
    ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
    ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
    print("ba_dot_pa shape: ", ba_dot_pa_all.shape)
    print("ba_dot_ba shape: ", ba_dot_ba_all.shape)
    h_all = jnp.clip(ba_dot_pa_all / ba_dot_ba_all, 0, 1)
    axis_to_point_all = pa_all - h_all[:, :, None] * ba_all[None, :, :]
    dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
    direction_all = axis_to_point_all / dist_all
    print("h_test shape: ", h_all.shape)
    print("axis_to_point_test shape: ", axis_to_point_all.shape)
    print("dist_test shape: ", dist_all.shape)
    print("direction_test shape: ", direction_all.shape)
    dist_idx = jnp.argmin(dist_all[:,:,0], axis=1)
    print("dist_idx shape: ", dist_idx.shape)
    dist = dist_all[jnp.arange(dist_all.shape[0]), dist_idx, :]
    dist_to_surface = dist - r_current
    print("dist_min shape: ", dist.shape)
    direction = direction_all[jnp.arange(direction_all.shape[0]), dist_idx, :]
    print("direction_min shape: ", direction.shape)
    # rv = rv.at[:, :, 2].set(0 * rv[:, :, 2])
    
    # displacements = 0.1*f_scale * r_target * ((re / r_target) ** 2 - 1) ** 2 * (-s) * rv
    mask = (dist_to_surface < doi).astype(int)
    displacements = f_scale * ((dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * direction
    displacements = displacements * mask
    # displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    # assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)
    step_size = f_scale * (-s)

    return displacements, step_size

from jax import lax
def old_safe_max(arr):
    # Use lax.cond to choose a branch based on whether the array has any elements.
    print("arr shape: ", arr.shape)
    return lax.cond(
        arr.shape[1] > 0,             # Condition: is the array non-empty?
        lambda x: jnp.max(x, axis=1)[:, None], # True branch: max together.
        lambda x: jnp.ones((arr.shape[0],1)),  # False branch: passthrough when empty.
        arr                         # Input to the branch functions.
    )

def safe_max(arr):
    # Use lax.cond to choose a branch based on whether the array has any elements.
    print("alpha_array shape[1]: ", arr.shape[1])
    return lax.cond(
        arr.shape[1] > 0,             # Condition: if in_contact is non-empty
        lambda x: jnp.max(x, axis=1), # True branch: max together.
        lambda x: jnp.ones(arr.shape[0]),  # False branch: passthrough when empty.
        arr                         # Input to the branch functions.
    )
def smin_sdf_capsule_warp_sculp(rv, a, b, stent_vertices, eps, s, r_target, r_current):
    doi = 0.65 # distance of influence: width of the deformation zone
    # doi = 0.15
    f_scale = 0.25 * doi * 0.1
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
    mask = (final_dist_to_surface < doi).astype(int)
    displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction
    displacements = displacements * mask
    '''
    # displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    scaling_mask = (1 - final_dist_to_surface / doi)
    # following line is original unJITable code
    # vertices_close_to_stent = rv[(final_dist_to_surface < 0.01).astype(bool)] 
    # vertices_close_to_stent_indices = jnp.range(rv.shape[0]) * (final_dist_to_surface < 0.01).astype(int)
    # vertices_close_to_stent = jnp.take(rv, vertices_close_to_stent_indices, axis=0)
    vertices_close_to_stent_mask = (final_dist_to_surface < 0.01).astype(int)
    # print("vertices_close_to_stent shape: ", vertices_close_to_stent.shape)
    print("rv shape", rv.shape)
    print("mask shape: ", mask.shape)
    # distances_to_vertices_close_to_stent = jnp.linalg.norm(rv - vertices_close_to_stent[None, :, :], axis=-1)
    distances_to_vertices_close_to_stent = jnp.linalg.norm(rv - rv[None, :, 0, :], axis=-1)
    print("distances_to_vertices_close_to_stent shape: ", distances_to_vertices_close_to_stent.shape)
    affected_vertices_mask = (distances_to_vertices_close_to_stent < doi).astype(int)
    affected_vertices_alpha = (1 - distances_to_vertices_close_to_stent / doi) * affected_vertices_mask
    print("affected_vertices_alpha shape: ", affected_vertices_alpha.shape)
    print("affected_vertices_mask shape: ", affected_vertices_mask.shape)
    # affected_vertices_mask = jnp.any(affected_vertices_mask, axis=1)[:, None].astype(int)
    # following is the original unJITable code
    # affected_vertices_blended_alpha = jnp.max(affected_vertices_alpha, axis=1)[:, None] if affected_vertices_alpha.shape[1] > 0 else jnp.any(affected_vertices_mask, axis=1)[:, None].astype(int)
    print("vertices_close_to_stent_mask shape: ", vertices_close_to_stent_mask.shape)
    affected_vertices_alpha = affected_vertices_alpha * vertices_close_to_stent_mask[None, :, 0]
    affected_vertices_blended_alpha = old_safe_max(affected_vertices_alpha)

    print("affected_vertices_blended_alpha shape: ", affected_vertices_blended_alpha.shape)

    # displacements = displacements
    # displacements = displacements * jnp.any(affected_vertices_mask, axis=1)[:, None].astype(int)
    displacements = displacements * affected_vertices_blended_alpha #* scaling_mask 
    '''
    step_size = f_scale * (-s)

    return displacements, step_size

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
    # displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction
    # displacements = displacements * mask

@jx.jit
def smin_sdf_capsule_contact_sculp_part_two(in_influence_vertices, in_contact_vertices, doi):
    doi = 0.65
    # f_scale = 0.25 * doi * 0.1
    start_time = time.time()
    # in_influence_indices = jnp.nonzero(in_influence_mask, size = num_in_influence, fill_value=-1)[0] 
    # in_influence_vertices = xs[in_influence_indices, 0, :]
    in_influence_to_in_contact_distances = jnp.linalg.norm(in_influence_vertices[:, None, :] - in_contact_vertices[None, :, :], axis=-1)
    print("time to compute all pair distances: ", time.time() - start_time)
    start_time = time.time()
    # shape of in_influence_to_in_contact_distances: 
    # (num_in_influence_points, in_contact_points)
    in_influence_vertices_alpha = (1 - in_influence_to_in_contact_distances / doi)
    in_influence_vertices_blended_alpha = jnp.max(in_influence_vertices_alpha, axis=1)
    # print("in_influence_vertices_blended_alpha shape: ", in_influence_vertices_blended_alpha.shape)
    # print("final_dist_to_surface shape: ", final_dist_to_surface.shape)
    # print("in_influence_indices shape: ", in_influence_indices.shape)
    # in_influence_vertices_blended_alpha_mask = jnp.zeros(total_num_vertices)
    # in_influence_vertices_blended_alpha_mask = in_influence_vertices_blended_alpha_mask.at[in_influence_indices].set(in_influence_vertices_blended_alpha)
    print("time to compute the blend mask: ", time.time() - start_time)
    # start_time = time.time()

    # displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    # step_size = f_scale * (-s)
    # print("time to compute displacements: ", time.time() - start_time)

    # return displacements, step_size
    return in_influence_vertices_blended_alpha

def smin_sdf_capsule_contact_sculp_part_two_KD(in_influence_vertices, in_contact_vertices, doi):
    doi = 0.65
    start_time = time.time()
    # in_influence_to_in_contact_distances = jnp.linalg.norm(in_influence_vertices[:, None, :] - in_contact_vertices[None, :, :], axis=-1)
    tree = cKDTree(in_contact_vertices, leafsize=32)
    in_influence_to_in_contact_distances, indices = tree.query(in_influence_vertices, k=1)
    print("time to compute all pair distances: ", time.time() - start_time)
    # print("in_influence_to_in_contact_distances shape: ", in_influence_to_in_contact_distances[:100])
    start_time = time.time()
    # shape of in_influence_to_in_contact_distances: 
    # print("in_influence_to_in_contact_distances shape: ", in_influence_to_in_contact_distances.shape)
    # (num_in_influence_points, in_contact_points)
    in_influence_vertices_alpha = (1 - in_influence_to_in_contact_distances / doi)
    # in_influence_vertices_blended_alpha = jnp.max(in_influence_vertices_alpha, axis=1)
    print("time to compute the blend mask: ", time.time() - start_time)
    
    return in_influence_vertices_alpha

@ jx.jit
def smin_sdf_capsule_contact_sculp_default(final_dist_to_surface, final_direction, in_influence_indices, in_influence_vertices, in_contact_vertices, doi, s):
    doi = 0.65
    f_scale = 0.25 * doi * 0.1
    doi_mask = (final_dist_to_surface < doi).astype(int)
    in_influence_to_in_contact_distances = jnp.linalg.norm(in_influence_vertices[:, None, :] - in_contact_vertices[None, :, :], axis=-1)
    # shape of in_influence_to_in_contact_distances: 
    # (num_in_influence_points, in_contact_points)
    in_influence_vertices_alpha = (1 - in_influence_to_in_contact_distances / doi)
    in_influence_vertices_blended_alpha = safe_max(in_influence_vertices_alpha)
    print("in_influence_vertices_blended_alpha shape: ", in_influence_vertices_blended_alpha.shape)
    in_influence_vertices_blended_alpha_mask = jnp.ones(final_dist_to_surface.shape[0])
    in_influence_vertices_blended_alpha_mask = in_influence_vertices_blended_alpha_mask.at[in_influence_indices].set(in_influence_vertices_blended_alpha)

    displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    step_size = f_scale * (-s)

    return displacements, step_size    


def kelvinlets_stent_edge_nonaffine_cylinder_with_stop(rv, a, b, eps, s, direction): #radius bounded, stops the whole field at once, maybe no self-intersections, current best
    # Ensure the input tensor has the correct dimensions
    # assert rv.ndim == 3
    w = 0.2
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # assert ndims == 3

    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # Extract components of rv
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    rz = linear_heaviside(rz, w)
    # to convert rz into a mask that is 1 if rz is less than w and 0 otherwise
    mask = (rz < 1e-6).astype(int) # consider using 1e-4 also possible
    rv = rv.at[:, :, 2].set(rz)
    # Compute re with epsilon added
    r = jnp.sqrt(rx**2 + ry**2 + rz**2)
    # r = regularize_radius(r)
    # print("r[21611:21613]", r[21611:21613])
    r2 = r**2
    re = jnp.sqrt(r2 + eps**2)
    # assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # r = jnp.expand_dims(r, 2)
    r2 = jnp.expand_dims(r2, 2)
    # Compute powers of re for the displacement formula
    # re2 = re**2
    # re7 = re**7
    # re9 = re**9
    re3 = re**3
    re5 = re**5
    r_target = 0.9
    # Calculate displacements
    rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    # displacements = ((-105*a*eps**4 / (2*re9)) - (b*(4*re2 - 5*(5*eps**2+2*r2))/re7) + (3*b*(4*re2-7*(7*eps**2+2*r2))*r2/re9) + (12*b*(7*eps**2+2*r2)/re7)) * s * rv
    # displacements = ((b*(109*eps**2+34*r2-4*re2)/re7) + ((3*b*(4*re2-49*eps**2-14*r2)*r2 - 52.5*a*eps**4)/re9)) * s * rv
    # assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)
    print("displacements[21611:21613]", displacements[21611:21613])
    print("shape of displacements: ", displacements.shape)
    # radius_mask = (r < 0.509).astype(int)
    radius_mask = (r < r_target).astype(int)
    mask = mask * radius_mask
    # displacements = displacements * mask[:, :, None]
    cylindrical_wall_displacements = displacements * mask[:, :, None]
    norms = jnp.linalg.norm(cylindrical_wall_displacements, axis=2)
    # nonzero_mask = (norms > 0.001).astype(int)
    norms_sum = jnp.sum(norms, axis=0)[0]
    # total_nonzero_entries = jnp.sum(nonzero_mask, axis=0)[0]
    total_nonzero_entries = jnp.sum(mask, axis=0)[0]
    print("total_nonzero_entries: ", total_nonzero_entries)
    displacement_continue_flag = 1 - (total_nonzero_entries == 0).astype(int)
    safe_total_nonzero_entries = total_nonzero_entries + (total_nonzero_entries == 0).astype(int)
    average_displacement_distance = norms_sum / (safe_total_nonzero_entries) * (total_nonzero_entries > 0).astype(int)
    print("average_displacement_distance: ", average_displacement_distance)
    displacements = displacements * displacement_continue_flag
    return displacements

def kelvinlets_stent_edge_nonaffine_og(rv, a, b, eps, s, direction):
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    # Extract components of rv
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    # Compute re with epsilon added
    re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # Compute powers of re for the displacement formula
    re3 = re**3
    re5 = re**5
    # Calculate displacements
    # rv = rv.at[:, :, 2].set(0.001 * rv[:, :, 2])
    displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)

    return displacements

def kelvinlets_stent_edge_true_stent_edge(rv, a, b, eps, s, direction):
    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    rz_cylinder = linear_heaviside_stent_edge(rz)
    rv = rv.at[:, :, 2].set(rz_cylinder)
    r = jnp.sqrt(rx**2 + ry**2 + rz_cylinder**2)
    w = 0.2
    taper = 2
    base_damping = 1
    damping = base_damping + taper / 2 * sigmoid_truncation(rz, direction)*(1 + direction * rz / w)
    rx_damped = rx / damping**2
    ry_damped = ry / damping**2
    rv = rv.at[:, :, 0].set(rx_damped)
    rv = rv.at[:, :, 1].set(ry_damped)
    # r = regularize_radius(r)
    # print("r[21611:21613]", r[21611:21613])
    r2 = r**2
    re = jnp.sqrt(r2 + eps**2)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # r = jnp.expand_dims(r, 2)
    r2 = jnp.expand_dims(r2, 2)
    # Compute powers of re for the displacement formula
    re2 = re**2
    re7 = re**7
    re9 = re**9
    # Calculate displacements
    # rv = rv.at[:, :, 2].set(1e-6 * rv[:, :, 2])
    # displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    # displacements = ((-105*a*eps**4 / (2*re9)) - (b*(4*re2 - 5*(5*eps**2+2*r2))/re7) + (3*b*(4*re2-7*(7*eps**2+2*r2))*r2/re9) + (12*b*(7*eps**2+2*r2)/re7)) * s * rv
    displacements = ((b*(109*eps**2+34*r2-4*re2)/re7) + ((3*b*(4*re2-49*eps**2-14*r2)*r2 - 52.5*a*eps**4)/re9)) * s * rv
    # print("displacements[21611:21613]", displacements[21611:21613])
    return displacements

def kelvinlets_affine_v3(rv, a, b, eps, s):
    # Ensure the input tensor has the correct dimensions
    print(f"rv shape: {rv.shape}")
    assert rv.ndim == 3
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    assert ndims == 3
    print(f"num_mesh_points: {num_mesh_points}, num_kelvinlet_points: {num_kelvinlet_points}, ndims: {ndims}")
    # Extract components of rv
    rx, ry, rz = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2]
    # Compute re with epsilon added
    re = jnp.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    assert re.shape == (num_mesh_points, num_kelvinlet_points)
    # Expand re and tile to match the dimensions of rv
    re = jnp.expand_dims(re, 2)
    # Compute powers of re for the displacement formula
    re3 = re**3
    re5 = re**5
    # Calculate displacements
    rv = rv.at[:, :, 2].set(0.001 * rv[:, :, 2])
    displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    assert displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims)

    return displacements

"""
Evaluate equation 16 of De Goes 2017

Inputs:
    rv: array of shape (num_mesh_points, num_kelvinlet_points, 3) 
        where rv[i, j, 0] = x_i - x0_j
        where rv[i, j, 1] = y_i - y0_j
        where rv[i, j, 2] = z_i - z0_j
"""
def kelvinlets_affine_v3_jonathan(rv, a, b, eps, s):
    # https://stackoverflow.com/a/22778484
    assert(rv.ndim == 3) # https://stackoverflow.com/a/21299842
    num_mesh_points, num_kelvinlet_points, ndims = rv.shape
    assert(ndims == 3)
    rx = rv[:, :, 0]
    ry = rv[:, :, 1]
    rz = rv[:, :, 2]
    re = np.sqrt(rx**2 + ry**2 + rz**2 + eps**2)
    assert(re.shape == (num_mesh_points, num_kelvinlet_points))
    re = np.expand_dims(re, 2)
    re = np.dstack([re for i in range(ndims)])
    assert(re.shape == (num_mesh_points, num_kelvinlet_points, ndims))
    re3 = re**3
    re5 = re**5
    displacements = (2 * b - a) * (1 / re3 + 3 * eps**2 / (2 * re5)) * s * rv
    assert(displacements.shape == (num_mesh_points, num_kelvinlet_points, ndims))
    return displacements

def get_rotation_matrix_v2(data, first_centerline_point_id, last_centerline_point_id, cartesian_axis, num_copies):
    assert cartesian_axis in {"x", "y", "z"}
    last_centerline_point_id += 1
    num_pts = last_centerline_point_id - first_centerline_point_id

    num_centerline_points = data["points"]["centerline"].shape[0]
    assert num_centerline_points > 1

    temp1 = jnp.zeros((num_centerline_points + 2, 3))
    temp2 = jnp.zeros((num_centerline_points + 2, 3))
    temp1 = temp1.at[:-2, :].set(data["points"]["centerline"])
    temp1 = temp1.at[-2:, :].set(data["points"]["centerline"][-1, :])
    temp2 = temp2.at[0:2, :].set(data["points"]["centerline"][0, :])
    temp2 = temp2.at[2:, :].set(data["points"]["centerline"])

    centerline_axis_vector = (temp1 - temp2)[1 + first_centerline_point_id : 1 + last_centerline_point_id]
    centerline_axis_vector = jnp.expand_dims(centerline_axis_vector, 2)
    centerline_axis_vector /= jnp.linalg.norm(centerline_axis_vector, axis=1, keepdims=True)

    cartesian_axis_vector = jnp.array([{"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[cartesian_axis]]).reshape(1, 3, 1)
    cartesian_axis_vector = jnp.tile(cartesian_axis_vector, (num_pts, 1, 1))

    rotation_matrix = compute_rotation_matrix(cartesian_axis_vector, centerline_axis_vector)
    rotation_matrix = jnp.tile(jnp.expand_dims(rotation_matrix, 0), (num_copies, 1, 1, 1))

    return rotation_matrix, centerline_axis_vector

def get_rotation_matrix_v2_slow(data, first_centerline_point_id, last_centerline_point_id, cartesian_axis, num_copies):
    assert cartesian_axis in {"x", "y", "z"}

    # Adjust last_centerline_point_id for inclusive slicing
    last_centerline_point_id += 1
    num_pts = last_centerline_point_id - first_centerline_point_id

    num_centerline_points = data["points"]["centerline"].shape[0]
    assert num_centerline_points > 1

    # Create temporary arrays for forward and backward finite differences
    temp1 = jnp.zeros((num_centerline_points + 2, 3))
    temp2 = jnp.zeros((num_centerline_points + 2, 3))
    temp1 = temp1.at[:-2, :].set(data["points"]["centerline"])
    temp1 = temp1.at[-2:, :].set(data["points"]["centerline"][-1, :])
    temp2 = temp2.at[0:2, :].set(data["points"]["centerline"][0, :])
    temp2 = temp2.at[2:, :].set(data["points"]["centerline"])

    # Compute centerline axis vectors
    centerline_axis_vector = (temp1 - temp2)[1 + first_centerline_point_id : 1 + last_centerline_point_id]
    centerline_axis_vector = jnp.expand_dims(centerline_axis_vector, 2)
    assert centerline_axis_vector.shape == (num_pts, 3, 1)

    # Normalize centerline axis vectors
    centerline_axis_vector_norm = jnp.linalg.norm(centerline_axis_vector, axis=1, keepdims=True)
    centerline_axis_vector /= centerline_axis_vector_norm

    # Get the Cartesian axis vector
    if cartesian_axis == "x":
        cartesian_axis_vector = jnp.array([1, 0, 0]).reshape((1, 3, 1))
    elif cartesian_axis == "y":
        cartesian_axis_vector = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    elif cartesian_axis == "z":
        cartesian_axis_vector = jnp.array([0, 0, 1]).reshape((1, 3, 1))
    cartesian_axis_vector = jnp.tile(cartesian_axis_vector, (num_pts, 1, 1))
    assert cartesian_axis_vector.shape == (num_pts, 3, 1)

    # Compute the rotation matrices
    a_plus_b = cartesian_axis_vector + centerline_axis_vector
    denominator = jnp.expand_dims(jnp.linalg.norm(a_plus_b, axis=1) ** 2, 2)
    rotation_matrix = 2 * jnp.matmul(a_plus_b, jnp.transpose(a_plus_b, axes=(0, 2, 1))) / denominator
    assert rotation_matrix.shape == (num_pts, 3, 3)

    # Subtract the identity matrix
    identity = jnp.tile(jnp.eye(3), (num_pts, 1, 1))
    rotation_matrix -= identity
    assert rotation_matrix.shape == (num_pts, 3, 3)

    # Repeat rotation matrices for the number of copies
    rotation_matrix = jnp.expand_dims(rotation_matrix, 0)
    rotation_matrix = jnp.tile(rotation_matrix, (num_copies, 1, 1, 1))
    assert rotation_matrix.shape == (num_copies, num_pts, 3, 3)

    return rotation_matrix, centerline_axis_vector

"""
For each centerline point from first_centerline_point_id to last_centerline_point_id, get the rotation matrix needed to rotate a vector (a cartesian basis vector) from the cartesian coordinate system to the local centerline coordinate system (local system at the centerline point).
"""
def get_rotation_matrix_v2_jonathan(data, first_centerline_point_id, last_centerline_point_id, cartesian_axis, num_copies):
    assert(cartesian_axis == "x" or cartesian_axis == "y" or cartesian_axis == "z")

    last_centerline_point_id += 1 # for inclusivity in python array slicing
    num_pts = last_centerline_point_id - first_centerline_point_id

    num_centerline_points = data["points"]["centerline"].shape[0]
    assert(num_centerline_points > 1)

    # get normalized centerline axis at each centerline point, using forward finite difference for the first point, central finite difference for the middle points, and backward finite difference for the last point
    temp1 = np.empty((num_centerline_points + 2, 3))
    temp2 = np.empty((num_centerline_points + 2, 3))
    temp1[:-2, :] = data["points"]["centerline"]
    temp1[-2, :] = data["points"]["centerline"][-1, :]
    temp1[-1, :] = data["points"]["centerline"][-1, :]
    temp2[0, :] = data["points"]["centerline"][0, :]
    temp2[1, :] = data["points"]["centerline"][0, :]
    temp2[2:, :] = data["points"]["centerline"]
    centerline_axis_vector = (temp1 - temp2)[(1+first_centerline_point_id):(1+last_centerline_point_id)] # adding 1 to account for offset in temp1 and temp2
    centerline_axis_vector = np.expand_dims(centerline_axis_vector, 2)
    assert(centerline_axis_vector.shape == (num_pts, 3, 1))
    centerline_axis_vector_norm = np.linalg.norm(centerline_axis_vector, axis = 1).reshape((num_pts, 1, 1))
    centerline_axis_vector_norm = np.hstack([centerline_axis_vector_norm for i in range(3)])
    assert(centerline_axis_vector_norm.shape == (num_pts, 3, 1))
    centerline_axis_vector = np.divide(centerline_axis_vector, centerline_axis_vector_norm)

    # get cartesian axis vector
    if cartesian_axis == "x":
        cartesian_axis_vector = np.array([1, 0, 0]).reshape((1, 3, 1))
    elif cartesian_axis == "y":
        cartesian_axis_vector = np.array([0, 1, 0]).reshape((1, 3, 1))
    elif cartesian_axis == "z":
        cartesian_axis_vector = np.array([0, 0, 1]).reshape((1, 3, 1))
    cartesian_axis_vector = np.vstack([cartesian_axis_vector for i in range(num_pts)])
    assert(cartesian_axis_vector.shape == (num_pts, 3, 1))

    # for all centerline points, get the rotation matrix, R, needed to rotate the cartesian axis unit vector, a, into the centerline axis unit vector, b, so that R * a = b, where R = 2 * np.matmul(a + b, (a + b).T) / np.matmul((a + b).T, a + b)[0, 0] - np.eye(3) # reference: https://math.stackexchange.com/a/2672702
    a_plus_b = cartesian_axis_vector + centerline_axis_vector
    denominator = np.expand_dims(np.linalg.norm(a_plus_b, axis = 1) ** 2, 2)
    assert(denominator.shape == (num_pts, 1, 1))
    rotation_matrix = np.divide(2 * np.matmul(a_plus_b, np.transpose(a_plus_b, axes = (0, 2, 1))), denominator)
    assert(rotation_matrix.shape == (num_pts, 3, 3))
    identity = np.expand_dims(np.eye(3), 0)
    identity = np.vstack([identity for i in range(num_pts)])
    assert(identity.shape == (num_pts, 3, 3))
    rotation_matrix -= identity
    assert(rotation_matrix.shape == (num_pts, 3, 3))
    rotation_matrix = np.expand_dims(rotation_matrix, 0)
    rotation_matrix = np.vstack([rotation_matrix for i in range(num_copies)])
    assert(rotation_matrix.shape == (num_copies, num_pts, 3, 3))

    return rotation_matrix, centerline_axis_vector

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
def get_affine_displacements_inner(data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor):
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    # Compute rv in the local frame
    rv = xs - centers
    # Rotate rv to the global frame
    rotation_matrices = jnp.expand_dims(rotation_matrices, 0)
    centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    # Compute Kelvinlet displacements
    displacement_local = kelvinlets_affine_v3(centerline_aligned_rv, a, b, eps, s)
    displacement_global = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    # rv has shape (N, num_kelvinlet_points, 3)
    # rotation_matrices has shape (num_kelvinlet_points, 3, 3)
    # print("rotation matrix = ", rotation_matrices)
    # Aggregate and normalize
    displacement = jnp.sum(displacement_global, axis=1)
    # Scale if required
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
    return displacement

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
    radial_maginitude_squared = (rx**2 + ry**2) 
    # Compute mask for rx^2 + ry^2 <= (original_radius * 1.1)^2
    mask = ((original_radius * 1.0) ** 2 <= radial_maginitude_squared) * (radial_maginitude_squared <= (original_radius * 1.1) ** 2)
    # Set rz_magnitude to a large value where mask is False so they are not selected as min
    rz_magnitude_masked = np.where(mask, rz_magnitude, jnp.inf)
    index_for_min_rz = np.argmin(rz_magnitude_masked, axis=0)
    print("index_for_min_rz: ", index_for_min_rz)
    print("rx, ry, rz for min_rz: ", rx[index_for_min_rz], ry[index_for_min_rz], rz[index_for_min_rz])
    return index_for_min_rz
    


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

@jx.jit #TODO: comment/uncomment this to print kelvinlet quantities
def get_sdf_displacements_inner(data_points, xs, centers, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, w, r_target, r_current):
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    # Compute rv in the local frame
    # rv = xs - centers
    # print("rv shape: ", rv.shape)
    # print("xs shape: ", xs.shape)
    # Rotate rv to the global frame
    # rotation_matrices = jnp.expand_dims(rotation_matrices, 0)
    # centerline_aligned_rv = jnp.einsum('...ij,...j->...i', rotation_matrices, rv)
    # Compute Kelvinlet displacements
    # average_displacement_distance = 0
    displacement, step_size = smin_sdf_capsule_warp_sculp(xs, a, b, stent_vertices, eps, s, r_target, r_current)
    # displacement_local = kelvinlets_affine_laplacian(centerline_aligned_rv, a, b, eps, s, 1, 0.8)
    # displacement_global = jnp.einsum('...ij,...j->...i', rotation_matrices, displacement_local)
    # Aggregate and normalize TODO: below use of sum is unnecessary if there is only 1 kelvinlet point
    # displacement = jnp.sum(displacement_local, axis=1)
    # Scale if required
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
        # average_displacement_distance *= surface_mesh_scale_factor
    return displacement, step_size

@jx.jit #TODO: comment/uncomment this to print kelvinlet quantities
def get_sdf_contact_displacements_inner(data_points, xs, centers, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, w, r_target, r_current):
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    # Compute rv in the local frame
    # rv = xs - centers
    # print("rv shape: ", rv.shape)
    # print("xs shape: ", xs.shape)
    contact_mask = smin_sdf_capsule_contact_sculp(xs, a, b, stent_vertices, eps, s, r_target, r_current)

    # Scale if required
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
    return 

# Helper function to preprocess data
def get_affine_displacements_v2(data, a, b, eps, s, phi_type, mesh_type, surface_mesh_scale_factor):
    # Resolve all_indices and force_center_point_id outside JIT
    all_indices = data["nodes"]["all_indices"]
    force_center_point_id = data["nodes"]["force_center_point_id"]
    # index_of_force_center_point_id = all_indices.index(force_center_point_id)

    if phi_type == "constant":
        left_index = all_indices[0]
        right_index = all_indices[2] + 1
    else:  # phi_type == "point"
        left_index = force_center_point_id
        right_index = force_center_point_id + 1

    # Prepare other data
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    rotation_matrix, _ = get_rotation_matrix_v2(data, left_index, right_index - 1, "z", data_points.shape[0])
    num_kelvinlet_points = int(right_index - left_index)
    num_kelvinlet_points = 3
    # print("left_index: ", left_index)
    # print("right_index: ", right_index)
    # print("all_indices: ", all_indices)
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    # centers = jnp.expand_dims(centerline_points[left_index:right_index, :], 0)
    centers = jnp.expand_dims(jnp.array([centerline_points[left_index], centerline_points[force_center_point_id], centerline_points[right_index - 1]]), 0)
    # Call the JIT-compiled function
    displacement = get_affine_displacements_inner(
        data_points, centerline_points, rotation_matrix, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    ) / num_kelvinlet_points
    return displacement

def get_affine_displacements_v3(data, a, b, eps, s, phi_type, mesh_type, surface_mesh_scale_factor): # this puts dense kelvinlet points from left_index to right_index
    # Resolve all_indices and force_center_point_id outside JIT
    all_indices = data["nodes"]["all_indices"]
    force_center_point_id = data["nodes"]["force_center_point_id"]
    # index_of_force_center_point_id = all_indices.index(force_center_point_id)

    
    left_index = force_center_point_id
    right_index = force_center_point_id + 1
    left_index = all_indices[0]
    right_index = all_indices[2] + 1

    # Prepare other data
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    rotation_matrix, _ = get_rotation_matrix_v2(data, left_index, right_index - 1, "z", data_points.shape[0])
    # rotation_matrix = None
    # num_kelvinlet_points = 1
    num_kelvinlet_points = right_index - left_index
    # print("left_index: ", left_index)
    # print("right_index: ", right_index)
    # print("all_indices: ", all_indices)
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    print("num_kelvinlet_points: ", num_kelvinlet_points)
    print("xs shape: ", xs.shape)
    # centers = jnp.expand_dims(jnp.array([centerline_points[force_center_point_id]]), 0)
    centers = jnp.expand_dims(centerline_points[left_index:right_index, :], 0)
    # Call the JIT-compiled function
    displacement = get_affine_displacements_inner(
        data_points, rotation_matrix, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    ) / num_kelvinlet_points
    return displacement

def get_affine_displacements_point(data, a, b, eps, s, phi_type, mesh_type, surface_mesh_scale_factor, force_center_normal, stent_halflength):
    # Resolve all_indices and force_center_point_id outside JIT
    all_indices = data["nodes"]["all_indices"]
    # force_center_point_id = data["nodes"]["force_center_point_id"]
    # index_of_force_center_point_id = all_indices.index(force_center_point_id)
    left_index = all_indices[0]
    force_center_point_id = all_indices[1]
    right_index = all_indices[2] + 1
    print("left, force, right: ", left_index, force_center_point_id, right_index)
    # Prepare other data
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    # rotation_matrix, _ = get_rotation_matrix_v2(data, left_index, right_index - 1, "z", data_points.shape[0])
    left_index_normal = vtk_utils.get_normal_at_centerline_point(centerline_points, all_indices[0])
    right_index_normal = vtk_utils.get_normal_at_centerline_point(centerline_points, all_indices[2])
    num_kelvinlet_points = 3
    # print("left_index: ", left_index)
    # print("right_index: ", right_index)
    # print("all_indices: ", all_indices)
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    print("num_kelvinlet_points: ", num_kelvinlet_points)
    print("xs shape: ", xs.shape)
    centers = jnp.expand_dims(jnp.array([centerline_points[left_index], centerline_points[force_center_point_id], centerline_points[right_index]]), 0)
    kelvinlet_points_normals = jnp.array([left_index_normal, force_center_normal, right_index_normal])
    rotation_matrices = compute_householder_matrices(kelvinlet_points_normals)
    # rotation_matrices = [compute_householder_matrix(normal) for normal in kelvinlet_points_normals]
    # Call the JIT-compiled function
    # displacement, average_displacement_distance = get_affine_laplacian_displacements_inner(
    #     data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor, stent_halflength
    # ) / num_kelvinlet_points
    displacement = get_affine_displacements_inner(
        data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor
    ) / num_kelvinlet_points
    return displacement

def get_parallell_displacements(data, a, b, eps, s, surface_mesh_scale_factor, kelvinlet_points_normals, stent_halflength, stent_radius):
    # Resolve all_indices and force_center_point_id outside JIT
    all_indices = data["nodes"]["all_indices"]
    print("force centers: ", all_indices)
    # Prepare other data
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = kelvinlet_points_normals.shape[0]
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    print("num_kelvinlet_points: ", num_kelvinlet_points)
    print("xs shape: ", xs.shape)
    centers = jnp.expand_dims(jnp.array(centerline_points[all_indices]), 0)
    rotation_matrices = compute_householder_matrices(kelvinlet_points_normals)
    # Call the JIT-compiled function
    displacements, average_displacement_distance = get_affine_laplacian_displacements_inner(
        data_points, rotation_matrices, xs, centers, a, b, eps, s, surface_mesh_scale_factor, stent_halflength, stent_radius
    )
    # print("average_displacement_distance = ", average_displacement_distance)
    return displacements / 3, average_displacement_distance#, centerline_displacements

def get_displacements(data, a, b, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, stent_radius):
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

def get_stent_edge_displacements(data, a, b, eps, s, surface_mesh_scale_factor, force_center_normal, direction, w, r_target):
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

def get_stenosis_displacements(data, a, b, eps, s, force_center_normal, r_min, r_max, r_original):
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

def get_sdf_displacements(data, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, target_stent_radius, current_stent_radius):
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
    displacements, step_size = get_sdf_displacements_inner(
        data_points, xs, centers, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, stent_halflength, target_stent_radius, current_stent_radius
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
    return displacements, step_size#, centerline_displacements

def get_sdf_contact_displacements_april_23(data, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, target_stent_radius, current_stent_radius):
    # Resolve all_indices and force_center_point_id outside JIT
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("force center: ", force_center_point_id)
    # Prepare other data
    data_points = data["points"]["surface"]
    # centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    # print("num_kelvinlet_points: ", num_kelvinlet_points)
    # print("xs shape: ", xs.shape)
    # centers = jnp.expand_dims(jnp.array([centerline_points[force_center_point_id]]), 0)
   
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    # centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    start_time = time.time()
    total_num_vertices = xs.shape[0]
    doi = 0.65
    doc = 0.01
    f_scale = 0.25 * doi * 0.1
    final_dist_to_surface, final_direction = smin_sdf_capsule_contact_sculp(xs, a, b, stent_vertices, eps, s, target_stent_radius, current_stent_radius) #JIT-compiled
    new_contact_mask = (final_dist_to_surface < doc).astype(bool)
    # print("new_contact_mask shape: ", new_contact_mask.shape)
    print("time taken to compute new_contact points: ", time.time() - start_time)
    xs = np.array(xs)
    new_contact_mask = np.array(new_contact_mask)
    start_time = time.time()
    in_contact_vertices = xs[new_contact_mask, :]
    print("time taken to obtain in_contact vertices subslice: ", time.time() - start_time)
    # print("in contact vertices shape: ", in_contact_vertices.shape)
    if in_contact_vertices.shape[0] == 0: # things are in contact <=> things are in influence
        step_size = f_scale * (-s)
        displacements, step_size = jnp.zeros((num_mesh_points, 3)), step_size
        return displacements, step_size 
    # in_contact_mask = in_contact_mask.at[new_contact_indices].set(True)
    # not_in_contact_mask = not_in_contact_mask.at[new_contact_indices].set(False)
    start_time = time.time()
    contact_tree = cKDTree(in_contact_vertices, leafsize=32)
    dist_min, _ = contact_tree.query(xs[:, 0, :], k=1, distance_upper_bound=doi)
    # print("dist_min shape: ", dist_min.shape)
    # print("time for KD Tree query: ", time.time() - start_time)
    start_time = time.time()
    in_influence_mask = dist_min < doi
    # print("in_influence_mask shape: ", in_influence_mask.shape)
    print("time it takes to compute <", time.time() - start_time)
    flat_time = time.time()
    in_influence_indices = np.flatnonzero(in_influence_mask)
    print("in_influence_indices shape: ", in_influence_indices.shape)
    print("time taken to flattennonzero: ", time.time() - flat_time)
    # assert(in_influence_indices.shape[0] > 0)
    start_time = time.time()
    # in_influence_vertices = xs[in_influence_mask, 0, :]
    # print("in_influence_vertices shape: ", in_influence_vertices.shape)
    # print("in_influence x in-contact shape: (", in_influence_vertices.shape[0], ", ", in_contact_vertices.shape[0], ")")
    in_influence_vertices = xs[in_influence_mask, 0, :]
    # num_in_influence = jx.jit(lambda mask: jnp.sum(mask))(in_influence_mask)
    print("time taken to obtain in-influence vertices: ", time.time() - start_time)
    # # D = compute_all_pair_distances(in_influence_vertices, in_contact_vertices)
    # print("D shape: ", D.shape)
    part_two_start_time = time.time()
    # in_influence_vertices = jnp.array(in_influence_vertices)
    # in_influence_indices_jnp = jnp.array(in_influence_indices)
    # in_contact_vertices = jnp.array(in_contact_vertices)
    print("time for converting to jnp arrays: ", time.time() - part_two_start_time)
    start_time = time.time()
    doi_mask = (final_dist_to_surface < doi).astype(int)
    in_influence_vertices_blended_alpha_mask = np.zeros(total_num_vertices)
    in_influence_vertices_blended_alpha = smin_sdf_capsule_contact_sculp_part_two_KD(in_influence_vertices, in_contact_vertices, doi)
    print("time taken to compute JIT sculpt part two: ", time.time() - start_time)
    start_time = time.time()
    in_influence_vertices_blended_alpha = np.array(in_influence_vertices_blended_alpha)
    # in_influence_vertices_blended_alpha_mask = in_influence_vertices_blended_alpha_mask.at[in_influence_indices].set(in_influence_vertices_blended_alpha)
    in_influence_vertices_blended_alpha_mask[in_influence_indices] = in_influence_vertices_blended_alpha
    print("time taken to compute blended alpha mask in np: ", time.time() - start_time)
    start_time = time.time()
    displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    step_size = f_scale * (-s)
    print("time taken to compute rest of the displacements: ", time.time() - start_time)   

    return displacements, step_size

def get_sdf_contact_displacements_noboundingbox_nocenterline(data, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, target_stent_radius, current_stent_radius):
    # Resolve all_indices and force_center_point_id outside JIT
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("force center: ", force_center_point_id)
    # Prepare other data
    data_points = data["points"]["surface"]
    # centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    xs = jnp.expand_dims(data_points, 1)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    # print("num_kelvinlet_points: ", num_kelvinlet_points)
    # print("xs shape: ", xs.shape)
    # centers = jnp.expand_dims(jnp.array([centerline_points[force_center_point_id]]), 0)
   
    num_mesh_points = data_points.shape[0]
    # Prepare xs and centers using broadcasting
    # centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    start_time = time.time()
    total_num_vertices = xs.shape[0]
    doi = 0.65
    doc = 0.01
    f_scale = 0.25 * doi * 0.1
    final_dist_to_surface, final_direction = smin_sdf_capsule_contact_sculp(xs, a, b, stent_vertices, eps, s, target_stent_radius, current_stent_radius) #JIT-compiled
    new_contact_mask = (final_dist_to_surface < doc).astype(bool)
    # print("new_contact_mask shape: ", new_contact_mask.shape)
    print("time taken to compute new_contact points: ", time.time() - start_time)
    xs = np.array(xs)
    new_contact_mask = np.array(new_contact_mask)
    start_time = time.time()
    in_contact_vertices = xs[new_contact_mask, :]
    print("time taken to obtain in_contact vertices subslice: ", time.time() - start_time)
    # print("in contact vertices shape: ", in_contact_vertices.shape)
    if in_contact_vertices.shape[0] == 0: # things are in contact <=> things are in influence
        step_size = f_scale * (-s)
        displacements, step_size = jnp.zeros((num_mesh_points, 3)), step_size
        return displacements, step_size 
    # in_contact_mask = in_contact_mask.at[new_contact_indices].set(True)
    # not_in_contact_mask = not_in_contact_mask.at[new_contact_indices].set(False)
    start_time = time.time()
    contact_tree = cKDTree(in_contact_vertices, leafsize=32)
    dist_min, _ = contact_tree.query(xs[:, 0, :], k=1, distance_upper_bound=doi)
    print("dist_min shape: ", dist_min.shape)
    print("time for KD Tree query: ", time.time() - start_time)
    start_time = time.time()
    in_influence_mask = dist_min < doi
    # print("in_influence_mask shape: ", in_influence_mask.shape)
    in_influence_indices = np.flatnonzero(in_influence_mask)
    print("in_influence_indices shape: ", in_influence_indices.shape)
    print("time taken to flattennonzero: ", time.time() - start_time)
    # assert(in_influence_indices.shape[0] > 0)
    start_time = time.time()
    in_influence_to_in_contact_distances = dist_min[in_influence_mask]
    # print("influence_distances from KD Tree shape: ", influence_distances[:100])
    # in_influence_vertices = xs[in_influence_mask, 0, :]
    # print("in_influence_vertices shape: ", in_influence_vertices.shape)
    # print("in_influence x in-contact shape: (", in_influence_vertices.shape[0], ", ", in_contact_vertices.shape[0], ")")
    # in_influence_vertices = xs[in_influence_mask, 0, :]
    print("time taken to obtain in-influence vertices: ", time.time() - start_time)
    part_two_start_time = time.time()
    # in_influence_vertices = jnp.array(in_influence_vertices)
    # in_influence_indices_jnp = jnp.array(in_influence_indices)
    # in_contact_vertices = jnp.array(in_contact_vertices)
    print("time for converting to jnp arrays: ", time.time() - part_two_start_time)
    start_time = time.time()
    doi_mask = (final_dist_to_surface < doi).astype(int)
    in_influence_vertices_blended_alpha_mask = np.zeros(total_num_vertices)
    # in_influence_vertices_blended_alpha = smin_sdf_capsule_contact_sculp_part_two_KD(in_influence_vertices, in_contact_vertices, doi)
    in_influence_vertices_blended_alpha = (1 - in_influence_to_in_contact_distances / doi)
    print("time taken to compute JIT sculpt part two: ", time.time() - start_time)
    start_time = time.time()
    in_influence_vertices_blended_alpha = np.array(in_influence_vertices_blended_alpha)
    # in_influence_vertices_blended_alpha_mask = in_influence_vertices_blended_alpha_mask.at[in_influence_indices].set(in_influence_vertices_blended_alpha)
    in_influence_vertices_blended_alpha_mask[in_influence_indices] = in_influence_vertices_blended_alpha
    print("time taken to compute blended alpha mask in np: ", time.time() - start_time)
    start_time = time.time()
    # displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    displacements = f_scale * force_kernel(1.99, 0.88, 2.3, final_dist_to_surface) * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    step_size = f_scale * (-s)
    print("time taken to compute rest of the displacements: ", time.time() - start_time)   
    # print("displacement is a jnp array: ", type(displacements))

    return displacements, step_size

def force_kernel(a, b, eps, r):
    r_eps = (r**2 + eps**2)**0.5
    return (a-b)/r_eps + b/r_eps**3 + a/2*eps**2/r_eps**3

@jx.jit
def stent_bounding_box(data_points, stent_vertices, target_stent_radius, doi, doc):
    # Compute the minimum and maximum coordinates of the bounding box
    min_coords = jnp.min(stent_vertices, axis=0) - target_stent_radius - doi - doc - 0.01
    max_coords = jnp.max(stent_vertices, axis=0) + target_stent_radius + doi + doc + 0.01
    mask = jnp.all((data_points >= min_coords) & (data_points <= max_coords), axis=1)
    return mask

def get_sdf_contact_displacements(data, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, target_stent_radius, current_stent_radius):
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("force center: ", force_center_point_id)
    data_points = data["points"]["surface"]
    # centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    doi = 0.65
    doc = 0.01
    f_scale = 0.25 * doi * 0.1
    start_time = time.time()
    bb_mask = stent_bounding_box(data_points, stent_vertices, target_stent_radius, doi, doc)
    print("time taken to compute bounding box: ", time.time() - start_time)
    start_time = time.time()
    bb_mask = np.array(bb_mask)
    data_points = np.array(data_points)
    data_points_masked = data_points[bb_mask]
    print("time taken to cast to numpy and mask out bounding box: ", time.time() - start_time)

    xs_np = np.expand_dims(data_points_masked, 1)
    xs = np.tile(xs_np, (1, num_kelvinlet_points, 1))
    # xs = jnp.array(xs)
    # xs = jnp.expand_dims(data_points, 1)
    # xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
   
    num_mesh_points = data_points.shape[0]
    start_time = time.time()
    total_num_vertices = xs.shape[0]
    print("total_num_vertices: ", total_num_vertices)
    final_dist_to_surface, final_direction = smin_sdf_capsule_contact_sculp(xs, a, b, stent_vertices, eps, s, target_stent_radius, current_stent_radius) #JIT-compiled
    new_contact_mask = (final_dist_to_surface < doc).astype(bool)
    # print("new_contact_mask shape: ", new_contact_mask.shape)
    print("time taken to compute new_contact points: ", time.time() - start_time)
    # xs = np.array(xs)
    xs = xs_np
    new_contact_mask = np.array(new_contact_mask)
    start_time = time.time()
    in_contact_vertices = xs[new_contact_mask, :]
    print("in_contact_vertices type: ", type(in_contact_vertices))
    print("time taken to obtain in_contact vertices subslice: ", time.time() - start_time)
    if in_contact_vertices.shape[0] == 0: # things are in contact <=> things are in influence
        step_size = f_scale * (-s)
        displacements, step_size = jnp.zeros((num_mesh_points, 3)), step_size
        return displacements, step_size 
    
    start_time = time.time()
    contact_tree = cKDTree(in_contact_vertices, leafsize=32)
    dist_min, _ = contact_tree.query(xs[:, 0, :], k=1, distance_upper_bound=doi)
    print("dist_min shape: ", dist_min.shape)
    print("time for KD Tree query: ", time.time() - start_time)
    start_time = time.time()
    in_influence_mask = dist_min < doi
    # print("in_influence_mask shape: ", in_influence_mask.shape)
    in_influence_indices = np.flatnonzero(in_influence_mask)
    print("in_influence_indices shape: ", in_influence_indices.shape)
    print("time taken to flattennonzero: ", time.time() - start_time)
    
    start_time = time.time()
    in_influence_to_in_contact_distances = dist_min[in_influence_mask]
    print("time taken to obtain in-influence vertices: ", time.time() - start_time)
    part_two_start_time = time.time()
    
    print("time for converting to jnp arrays: ", time.time() - part_two_start_time)
    start_time = time.time()
    doi_mask = (final_dist_to_surface < doi).astype(int)
    in_influence_vertices_blended_alpha_mask = np.zeros(total_num_vertices)
    # in_influence_vertices_blended_alpha = smin_sdf_capsule_contact_sculp_part_two_KD(in_influence_vertices, in_contact_vertices, doi)
    in_influence_vertices_blended_alpha = (1 - in_influence_to_in_contact_distances / doi)
    print("time taken to compute JIT sculpt part two: ", time.time() - start_time)
    start_time = time.time()
    in_influence_vertices_blended_alpha = np.array(in_influence_vertices_blended_alpha)
    in_influence_vertices_blended_alpha_mask[in_influence_indices] = in_influence_vertices_blended_alpha
    print("time taken to compute blended alpha mask in np: ", time.time() - start_time)
    start_time = time.time()
    displacements = f_scale * ((final_dist_to_surface / doi) ** 2 - 1) ** 2 * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    full_displacements = np.zeros((num_mesh_points, 3))
    full_displacements[bb_mask] = displacements
    # displacements = f_scale * force_kernel(1.99, 0.88, 2.3, final_dist_to_surface) * (-s) * final_direction * doi_mask * in_influence_vertices_blended_alpha_mask[:, None]
    step_size = f_scale * (-s)
    print("time taken to compute rest of the displacements: ", time.time() - start_time)   
    # displacement is of type jnp array

    return full_displacements, step_size

def get_sdf_contact_surface_and_centerline_displacements(data, a, b, stent_vertices, eps, s, surface_mesh_scale_factor, force_center_normal, stent_halflength, target_stent_radius, current_stent_radius):
    force_center_point_id = data["nodes"]["force_center_point_id"]
    print("selected point ID: ", force_center_point_id)
    data_points = data["points"]["surface"]
    centerline_points = data["points"]["centerline"]
    num_kelvinlet_points = 1
    doi = 0.9
    doi = 0.65
    doc = 0.01
    doc = 0.001
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
    dist_min, _ = contact_tree.query(xs, k=1, distance_upper_bound=doi)
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

def get_affine_displacements_v2_no_timer(data, a, b, eps, s, phi_type, mesh_type, surface_mesh_scale_factor):
    assert mesh_type in {"surface", "centerline"} or mesh_type.startswith("other_geometry_")
    assert phi_type in {"point", "constant"}

    if surface_mesh_scale_factor is not None:
        assert mesh_type in {"centerline"} or mesh_type.startswith("other_geometry_")

    num_mesh_points = data["points"][mesh_type].shape[0]
    force_center_point_id = data["nodes"]["force_center_point_id"]
    index_of_force_center_point_id = data["nodes"]["all_indices"].index(force_center_point_id)

    if phi_type == "constant":
        left_index = data["nodes"]["all_indices"][index_of_force_center_point_id - 1]
        right_index = data["nodes"]["all_indices"][index_of_force_center_point_id + 1] + 1
    else:  # phi_type == "point"
        left_index = force_center_point_id
        right_index = force_center_point_id + 1
    num_kelvinlet_points = right_index - left_index

    rotation_matrix, _ = get_rotation_matrix_v2(data, left_index, right_index - 1, "z", num_mesh_points)
    assert rotation_matrix.shape == (num_mesh_points, num_kelvinlet_points, 3, 3)

    # Rotate mesh points to local centerline frame
    xs = jnp.expand_dims(data["points"][mesh_type], 1)
    assert xs.shape == (num_mesh_points, 1, 3)
    xs = jnp.tile(xs, (1, num_kelvinlet_points, 1))
    assert xs.shape == (num_mesh_points, num_kelvinlet_points, 3)

    centers = jnp.expand_dims(data["points"]["centerline"][left_index:right_index, :], 0)
    assert centers.shape == (1, num_kelvinlet_points, 3)
    centers = jnp.tile(centers, (num_mesh_points, 1, 1))
    assert centers.shape == (num_mesh_points, num_kelvinlet_points, 3)

    rv = jnp.expand_dims(xs - centers, 3)
    rv = jnp.matmul(jnp.transpose(rotation_matrix, axes=(0, 1, 3, 2)), rv)
    rv = rv[:, :, :, 0]
    # xs = rv + centers  # This line is unused in the original logic.

    # Compute displacements in the local centerline frame
    displacement = jnp.expand_dims(kelvinlets_affine_v3(rv, a, b, eps, s), 3)
    assert displacement.shape == (num_mesh_points, num_kelvinlet_points, 3, 1)

    # Rotate displacements back to world space
    displacement = jnp.matmul(rotation_matrix, displacement)
    assert displacement.shape == (num_mesh_points, num_kelvinlet_points, 3, 1)
    displacement = displacement[:, :, :, 0]
    assert displacement.shape == (num_mesh_points, num_kelvinlet_points, 3)

    # Sum and normalize displacements
    displacement = jnp.sum(displacement, axis=1)
    assert displacement.shape == (num_mesh_points, 3)
    displacement /= num_kelvinlet_points

    if mesh_type == "centerline" and phi_type == "point":
        assert jnp.allclose(displacement[force_center_point_id, :], jnp.zeros(3))

    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor

    return displacement

def get_affine_displacements_v2_jonathan(data, a, b, eps, s, phi_type, mesh_type, surface_mesh_scale_factor):

    assert(mesh_type == "surface" or mesh_type == "centerline" or mesh_type[:15] == "other_geometry_")
    assert(phi_type == "point" or phi_type == "constant")

    if surface_mesh_scale_factor is not None:
        assert(mesh_type == "centerline" or (mesh_type[:15] == "other_geometry_"))
    
    num_mesh_points = data["points"][mesh_type].shape[0]
    force_center_point_id = data["nodes"]["force_center_point_id"]
    index_of_force_center_point_id = data["nodes"]["all_indices"].index(force_center_point_id)
    if phi_type == "constant":
        left_index = data["nodes"]["all_indices"][index_of_force_center_point_id - 1]
        right_index = data["nodes"]["all_indices"][index_of_force_center_point_id + 1] + 1 # the additional "+1" is for inclusivity in python array slicing
    else: # phi_type == "point"
        left_index = force_center_point_id
        right_index = force_center_point_id + 1 # the additional "+1" is for inclusivity in python array slicing
    num_kelvinlet_points = right_index - left_index

    rotation_matrix, _ = get_rotation_matrix_v2(data, left_index, right_index - 1, "z", num_mesh_points) # rotation matrices that rotate the cartesian z-axis, about each centerline point, into the local centerline axis
    assert(rotation_matrix.shape == (num_mesh_points, num_kelvinlet_points, 3, 3))

    # Rotate mesh points, about each centerline point, from world space (global coordinates) into local frame (centerline frame), so that the z-axis aligns with centerline axis at each centerline point
    xs = np.expand_dims(data["points"][mesh_type], 1)
    assert(xs.shape == (num_mesh_points, 1, 3))
    xs = np.hstack([xs for i in range(num_kelvinlet_points)])
    assert(xs.shape == (num_mesh_points, num_kelvinlet_points, 3))
    centers = np.expand_dims(data["points"]["centerline"][left_index:right_index, :], 0)
    assert(centers.shape == (1, num_kelvinlet_points, 3))
    centers = np.vstack([centers for i in range(num_mesh_points)])
    assert(centers.shape == (num_mesh_points, num_kelvinlet_points, 3))
    rv = np.expand_dims(xs - centers, 3)
    rv = np.matmul(np.transpose(rotation_matrix, axes = (0, 1, 3, 2)), rv)
    rv = rv[:, :, :, 0] # similar to np.squeeze
    # xs = rv + centers

    # Compute the displacements in the local centerline frame
    displacement = np.expand_dims(kelvinlets_affine_v3(rv, a, b, eps, s), 3)
    assert(displacement.shape == (num_mesh_points, num_kelvinlet_points, 3, 1))

    # Rotate displacements from local centerline frame to world space (but still centered at centerline point because the kelvinlets are inherently defined as being relative to the force point (centerline point))
    displacement = np.matmul(rotation_matrix, displacement)
    assert(displacement.shape == (num_mesh_points, num_kelvinlet_points, 3, 1))
    displacement = displacement[:, :, :, 0] # similar to np.squeeze
    assert(displacement.shape == (num_mesh_points, num_kelvinlet_points, 3))

    displacement = np.sum(displacement, axis = 1)
    assert(displacement.shape == (num_mesh_points, 3))
    
    # normalize total displacement
    displacement /= num_kelvinlet_points

    if mesh_type == "centerline" and phi_type == "point":
        np.testing.assert_array_equal(displacement[force_center_point_id, :], np.zeros(3))
    
    if surface_mesh_scale_factor is not None:
        displacement *= surface_mesh_scale_factor
    
    return displacement

def get_cross_section_ring_points_timer(origin, normal, surface_polydata, num_ring_points, save_slice_to_file=False):
    start_time = time.time()

    # Slice the polydata using VTK and convert to JAX-compatible array
    start = time.time()
    cross_sectional_slice_polydata = vtk_utils.slice_polydata(surface_polydata, origin, normal)
    print(f"Time to slice polydata: {time.time() - start:.4f} seconds")
    
    start = time.time()
    cross_section_points = jnp.array(v2n(cross_sectional_slice_polydata.GetPoints().GetData()))
    print(f"Time to convert VTK points to JAX array: {time.time() - start:.4f} seconds")
    
    # Sort points in counter-clockwise (ccw) order
    start = time.time()
    ccw_sorted_cross_section_points = common.sort_ring_points_in_ccw(cross_section_points, normal)
    print(f"Time to sort points in CCW order: {time.time() - start:.4f} seconds")

    # Assertions to ensure correct shapes
    num_cross_section_points = ccw_sorted_cross_section_points.shape[0]
    assert num_cross_section_points > num_ring_points
    assert ccw_sorted_cross_section_points.shape == (num_cross_section_points, 3)

    # Expand dims for compatibility with downstream processing
    start = time.time()
    ccw_sorted_cross_section_points = jnp.expand_dims(ccw_sorted_cross_section_points, 2)
    print(f"Time to expand dims of sorted points: {time.time() - start:.4f} seconds")
    assert ccw_sorted_cross_section_points.shape == (num_cross_section_points, 3, 1)

    # Select evenly spaced indices for ring points
    start = time.time()
    step = num_cross_section_points // num_ring_points
    indices = jnp.arange(0, step * num_ring_points, step)
    ccw_sorted_ring_points = ccw_sorted_cross_section_points[indices]
    print(f"Time to select evenly spaced ring points: {time.time() - start:.4f} seconds")
    assert ccw_sorted_ring_points.shape == (num_ring_points, 3, 1)
    
    # Log selected indices
    print("ring point indices =", indices)

    total_time = time.time() - start_time
    print(f"Total execution time for get_cross_section_ring_points_timer: {total_time:.4f} seconds")

    return ccw_sorted_cross_section_points, ccw_sorted_ring_points

def get_cross_section_ring_points(origin, normal, surface_polydata, num_ring_points, save_slice_to_file=False):
    # Slice the polydata using VTK and convert to JAX-compatible array
    cross_sectional_slice_polydata = vtk_utils.slice_polydata(surface_polydata, origin, normal)
    cross_section_points = jnp.array(v2n(cross_sectional_slice_polydata.GetPoints().GetData()))
    # Sort points in counter-clockwise (ccw) order
    ccw_sorted_cross_section_points = jx.jit(common.sort_ring_points_in_ccw)(cross_section_points, normal)
    num_cross_section_points = ccw_sorted_cross_section_points.shape[0]
    # Assertions to ensure correct shapes
    assert num_cross_section_points > num_ring_points
    assert ccw_sorted_cross_section_points.shape == (num_cross_section_points, 3)
    # Expand dims for compatibility with downstream processing
    ccw_sorted_cross_section_points = jnp.expand_dims(ccw_sorted_cross_section_points, 2)
    assert ccw_sorted_cross_section_points.shape == (num_cross_section_points, 3, 1)
    # Select evenly spaced indices for ring points
    step = num_cross_section_points // num_ring_points
    indices = jnp.arange(0, step * num_ring_points, step)
    ccw_sorted_ring_points = ccw_sorted_cross_section_points[indices]
    # Ensure the resulting shape
    assert ccw_sorted_ring_points.shape == (num_ring_points, 3, 1)
    print("ring point indices =", indices)
    
    return ccw_sorted_cross_section_points, ccw_sorted_ring_points

def get_cross_section_ring_points_jonathan(data, origin, normal, surface_polydata, num_ring_points, save_slice_to_file = False):
    cross_sectional_slice_polydata = vtk_utils.slice_polydata(surface_polydata, origin, normal)
    cross_section_points = copy.deepcopy(v2n(cross_sectional_slice_polydata.GetPoints().GetData()))
    ccw_sorted_cross_section_points = common.sort_ring_points_in_ccw(cross_section_points, normal)
    num_cross_section_points = ccw_sorted_cross_section_points.shape[0]
    assert(num_cross_section_points > num_ring_points)
    assert(ccw_sorted_cross_section_points.shape == (num_cross_section_points, 3))
    ccw_sorted_cross_section_points = np.expand_dims(ccw_sorted_cross_section_points, 2)
    assert(ccw_sorted_cross_section_points.shape == (num_cross_section_points, 3, 1))
    
    indices = list(range(0, num_cross_section_points, num_cross_section_points // num_ring_points))
    assert(len(indices) >= num_ring_points)
    while len(indices) > num_ring_points:
        indices = indices[:-1]
    
    ccw_sorted_ring_points = ccw_sorted_cross_section_points[indices]
    assert(ccw_sorted_ring_points.shape == (num_ring_points, 3, 1))
    print("ring point indices = ", indices)
    
    if save_slice_to_file:
        ccw_sorted_cross_section_points_vtk = vtk.vtkPoints()
        ccw_sorted_cross_section_points_vtk.SetData(n2v( np.squeeze(ccw_sorted_cross_section_points) ))
        cross_sectional_slice_polydata.SetPoints(ccw_sorted_cross_section_points_vtk)
        is_ring_point = np.zeros(num_cross_section_points)
        is_ring_point[indices] = 1
        cross_sectional_slice_polydata = vtk_utils.add_point_data_array_to_polydata(cross_sectional_slice_polydata, "ring_point", is_ring_point)
        vtk_utils.write_polydata("./cross_sectional_slice_polydata.vtp", cross_sectional_slice_polydata)
    
    return ccw_sorted_cross_section_points, ccw_sorted_ring_points

def get_ring_point_and_forces_v2(data, a, b, eps, origin, normal, surface_polydata, num_ring_points, new_cross_section_area, falloff_type, kelvinlets_translation_jit):
    start_time = time.time()

    # Ensure origin has the correct shape
    assert origin.shape == (3,)
    print(f"Initial setup time: {time.time() - start_time:.4f} seconds")
    
    # Get ring and cross-sectional points as JAX arrays
    start = time.time()
    cross_section_points, ring_points = get_cross_section_ring_points(origin, normal, surface_polydata, num_ring_points)
    print(f"Time to get ring and cross-sectional points: {time.time() - start:.4f} seconds")
    
    # Assertions for ring points shape
    assert ring_points.shape == (num_ring_points, 3, 1)
    num_cross_section_points = cross_section_points.shape[0]
    assert cross_section_points.shape == (num_cross_section_points, 3, 1)

    # Initialize kappa matrix
    start = time.time()
    kappa = jnp.zeros((3 * num_ring_points, 3 * num_ring_points))
    print(f"Time to initialize kappa matrix: {time.time() - start:.4f} seconds")

    # Compute based on falloff type
    start = time.time()
    if falloff_type == "regular":
        kk = kelvinlets_translation_jit(
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            a, b, eps
        )
    elif falloff_type == "laplacian":
        kk = common.laplacian_kelvinlets_translation_v2_jit(
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            a, b, eps
        )
    elif falloff_type == "bilaplacian":
        kk = common.bilaplacian_kelvinlets_translation_v2(
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            a, b, eps
        )
    else:
        sys.exit(f"Error: falloff_type '{falloff_type}' is not recognized.")
    print(f"Time to compute kk matrix based on falloff type '{falloff_type}': {time.time() - start:.4f} seconds")
    
    # Populate kappa matrix
    start = time.time()
    # for l in range(num_ring_points):
    #     for i in range(num_ring_points):
    #         kappa = kappa.at[3 * i : 3 * (i + 1), 3 * l : 3 * (l + 1)].set(kk[i, l, :, :])
    # kappa = jnp.block([[kk[i, l, :, :] for l in range(num_ring_points)] for i in range(num_ring_points)])
    kappa = kk
   
    print(f"Time to populate kappa matrix: {time.time() - start:.4f} seconds")

    # Prepare closed ring of cross-sectional points for area scaling
    start = time.time()
    closed_cross_section_points = jnp.vstack([
        cross_section_points[:, :, 0], cross_section_points[0, :, 0]
    ])
    print(f"Time to prepare closed cross-sectional points: {time.time() - start:.4f} seconds")

    # Calculate centroid of cross-sectional points
    start = time.time()
    centroid = common.get_centroid(cross_section_points[:, :, 0])
    assert centroid.shape == (1, 3)
    centroid = centroid[0]
    print(f"Time to calculate centroid: {time.time() - start:.4f} seconds")

    # Define basis vectors in the ring plane
    start = time.time()
    e2 = normal / jnp.linalg.norm(normal)
    e0 = closed_cross_section_points[0] - centroid
    e0 /= jnp.linalg.norm(e0)
    e1 = jnp.cross(e2, e0)
    print(f"Time to define basis vectors in ring plane: {time.time() - start:.4f} seconds")

    # Project closed ring points onto the 2D ring plane
    start = time.time()
    # closed_cross_section_points_2d = jnp.array([
    #     [jnp.dot(closed_cross_section_points[ip] - centroid, e0), 
    #      jnp.dot(closed_cross_section_points[ip] - centroid, e1)]
    #     for ip in range(num_cross_section_points + 1)
    # ])
    # Step 1: Calculate relative positions to the centroid
    relative_positions = closed_cross_section_points - centroid  # Shape: (num_cross_section_points + 1, 3)
    # Step 2: Stack `e0` and `e1` to create a projection matrix of shape (3, 2)
    projection_matrix = jnp.stack([e0, e1], axis=1)  # Shape: (3, 2)
    # Step 3: Project all points onto the 2D plane using a single matrix multiplication
    closed_cross_section_points_2d = relative_positions @ projection_matrix  # Shape: (num_cross_section_points + 1, 2)
    print(f"Time to project closed ring points onto 2D plane: {time.time() - start:.4f} seconds")

    # Calculate scaling factor to match the desired cross-sectional area
    start = time.time()
    scale = 0.9
    # scale = ring_points_optimizer.find_scale_to_get_prescribed_area(closed_cross_section_points_2d, new_cross_section_area)
    print(f"Time to calculate scaling factor: {time.time() - start:.4f} seconds")

    # Scale ring points and adjust the centroid
    start = time.time()
    scaled_ring_points = scale * ring_points
    center = common.get_centroid(scaled_ring_points[:, :, 0])
    center = jnp.expand_dims(jnp.broadcast_to(center, (num_ring_points, 3)), 2)
    true_center = jnp.expand_dims(jnp.broadcast_to(jnp.expand_dims(centroid, 0), (num_ring_points, 3)), 2)
    scaled_ring_points += true_center - center
    print(f"Time to scale and adjust ring points: {time.time() - start:.4f} seconds")

    # Compute displacement vector `u_bar`
    start = time.time()
    u_bar = scaled_ring_points - ring_points
    assert u_bar.shape == (num_ring_points, 3, 1)
    u_bar = u_bar.reshape((3 * num_ring_points, 1))
    print(f"Time to compute displacement vector u_bar: {time.time() - start:.4f} seconds")

    # Solve for ring forces
    start = time.time()
    ring_forces = jnp.linalg.solve(kappa, u_bar)
    assert ring_forces.shape == (3 * num_ring_points, 1)
    print(f"Time to solve for ring forces: {time.time() - start:.4f} seconds")

    total_time = time.time() - start_time
    print(f"Total execution time for get_ring_point_and_forces_v2_timer: {total_time:.4f} seconds")

    return ring_points, ring_forces

# @profile_func
def get_ring_point_and_forces_v2_no_timer(data, a, b, eps, origin, normal, surface_polydata, num_ring_points, new_cross_section_area, falloff_type, kelvinlets_translation_jit):
    assert origin.shape == (3,)
    # Get ring and cross-sectional points as JAX arrays
    cross_section_points, ring_points = get_cross_section_ring_points(origin, normal, surface_polydata, num_ring_points)
    assert ring_points.shape == (num_ring_points, 3, 1)
    num_cross_section_points = cross_section_points.shape[0]
    assert cross_section_points.shape == (num_cross_section_points, 3, 1)
    # Initialize kappa matrix and compute based on falloff type
    kappa = jnp.zeros((3 * num_ring_points, 3 * num_ring_points))
    if falloff_type == "regular":
        # kk = common.kelvinlets_translation_v2(
        #     ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
        #     ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
        #     a, b, eps
        # )
        kk = kelvinlets_translation_jit(
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            a, b, eps
        )
    elif falloff_type == "laplacian":
        kk = common.laplacian_kelvinlets_translation_v2_jit(
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            a, b, eps
        )
    elif falloff_type == "bilaplacian":
        kk = common.bilaplacian_kelvinlets_translation_v2(
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            ring_points[:, 0, 0], ring_points[:, 1, 0], ring_points[:, 2, 0], 
            a, b, eps
        )
    else:
        sys.exit(f"Error: falloff_type '{falloff_type}' is not recognized.")
    assert kk.shape == (num_ring_points, num_ring_points, 3, 3)
    # Populate kappa matrix by expanding kk along the appropriate axes
    # kappa = jnp.block([
    #     [kk[i, j] if i == j else jnp.zeros((3, 3)) for j in range(num_ring_points)]
    #     for i in range(num_ring_points)
    # ]).reshape(3 * num_ring_points, 3 * num_ring_points)
    for l in range(num_ring_points):
        for i in range(num_ring_points):
            kappa = kappa.at[3 * i : 3 * (i + 1), 3 * l : 3 * (l + 1)].set(kk[i, l, :, :])

    # Prepare closed ring of cross-sectional points for area scaling
    closed_cross_section_points = jnp.vstack([
        cross_section_points[:, :, 0], cross_section_points[0, :, 0]
    ])
    # Calculate the centroid of the cross-sectional points
    centroid = common.get_centroid(cross_section_points[:, :, 0])
    assert centroid.shape == (1, 3)
    centroid = centroid[0]
    # Define basis vectors in the ring plane
    e2 = normal / jnp.linalg.norm(normal)
    e0 = closed_cross_section_points[0] - centroid
    e0 /= jnp.linalg.norm(e0)
    e1 = jnp.cross(e2, e0)
    # Project closed ring points onto the 2D ring plane
    closed_cross_section_points_2d = jnp.array([
        [jnp.dot(closed_cross_section_points[ip] - centroid, e0), 
         jnp.dot(closed_cross_section_points[ip] - centroid, e1)]
        for ip in range(num_cross_section_points + 1)
    ])
    # Calculate scaling factor to match the desired cross-sectional area
    scale = 0.9
    # scale = ring_points_optimizer.find_scale_to_get_prescribed_area(closed_cross_section_points_2d, new_cross_section_area)
    # Scale ring points and adjust the centroid
    scaled_ring_points = scale * ring_points
    # Calculate the centroid and broadcast once to the desired shape
    center = common.get_centroid(scaled_ring_points[:, :, 0])
    center = jnp.expand_dims(jnp.broadcast_to(center, (num_ring_points, 3)), 2)
    true_center = jnp.expand_dims(jnp.broadcast_to(jnp.expand_dims(centroid, 0), (num_ring_points, 3)), 2)
    scaled_ring_points += true_center - center
    # Compute displacement vector `u_bar`
    u_bar = scaled_ring_points - ring_points
    assert u_bar.shape == (num_ring_points, 3, 1)
    u_bar = u_bar.reshape((3 * num_ring_points, 1))
    # Solve for ring forces
    ring_forces = jnp.linalg.solve(kappa, u_bar)
    assert ring_forces.shape == (3 * num_ring_points, 1)

    return ring_points, ring_forces

def get_ring_point_and_forces_v2_jonathan(data, a, b, eps, origin, normal, surface_polydata, num_ring_points, new_cross_section_area, falloff_type):
    assert(origin.shape == (3, ))
    cross_section_points, ring_points = get_cross_section_ring_points(data, origin, normal, surface_polydata, num_ring_points)
    assert(ring_points.shape == (num_ring_points, 3, 1))
    num_cross_section_points = cross_section_points.shape[0]
    assert(cross_section_points.shape == (num_cross_section_points, 3, 1))
    
    kappa = np.zeros((3 * num_ring_points, 3 * num_ring_points))
    if falloff_type == "regular":
        kk = common.kelvinlets_translation_v2(  ring_points[:, 0, 0], 
                                                ring_points[:, 1, 0], 
                                                ring_points[:, 2, 0], 
                                                ring_points[:, 0, 0],
                                                ring_points[:, 1, 0],
                                                ring_points[:, 2, 0],
                                                a, b, eps)
    elif falloff_type == "laplacian":
        kk = common.laplacian_kelvinlets_translation_v2(ring_points[:, 0, 0], 
                                                        ring_points[:, 1, 0], 
                                                        ring_points[:, 2, 0], 
                                                        ring_points[:, 0, 0],
                                                        ring_points[:, 1, 0],
                                                        ring_points[:, 2, 0],
                                                        a, b, eps)
    elif falloff_type == "bilaplacian":
        kk = common.bilaplacian_kelvinlets_translation_v2(  ring_points[:, 0, 0], 
                                                            ring_points[:, 1, 0], 
                                                            ring_points[:, 2, 0], 
                                                            ring_points[:, 0, 0],
                                                            ring_points[:, 1, 0],
                                                            ring_points[:, 2, 0],
                                                            a, b, eps)
    else:
        sys.exit("Error. falloff_type, ", falloff_type, ", is not recognized.")
    assert(kk.shape == (num_ring_points, num_ring_points, 3, 3))
    
    for l in range(num_ring_points):
        for i in range(num_ring_points):
            kappa[3 * i : 3 * (i + 1), 3 * l : 3 * (l + 1)] = kk[i, l, :, :]
    
    # create a closed ring centered at the centroid, for area computation
    closed_cross_section_points = np.zeros((num_cross_section_points + 1, 3))
    closed_cross_section_points[:num_cross_section_points] = copy.deepcopy(cross_section_points[:, :, 0])
    closed_cross_section_points[num_cross_section_points] = copy.deepcopy(closed_cross_section_points[0])
    centroid = common.get_centroid(cross_section_points[:, :, 0])
    assert(centroid.shape == (1, 3))
    centroid = centroid[0]
    
    # get basis vectors defining the plane of the ring_points (assumes all ring points lie on the same plane), where basis vectors, e0 and e1, lie in the plane and basis vector, e2, is normal to the plane
    e2 = normal / np.linalg.norm(normal)
    e0 = closed_cross_section_points[0] - centroid
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(e2, e0)
    
    # project closed ring points onto ring plane to get 2d points, to get scale
    closed_cross_section_points_2d = np.zeros((num_cross_section_points + 1, 2))
    for ip in range(num_cross_section_points + 1):
        point_coordinate_relative = closed_cross_section_points[ip] - centroid
        closed_cross_section_points_2d[ip] = np.array([np.dot(point_coordinate_relative, e0), np.dot(point_coordinate_relative, e1)])
    
    scale = 0.9
    # scale = ring_points_optimizer.find_scale_to_get_prescribed_area(closed_cross_section_points_2d, new_cross_section_area) # multiplicative scale that scales ring points, such that the area of the scaled closed ring points matches desired area, new_cross_section_area # todo: replace this code with the section below (because the scale predicted from the optimizer is just square root of the area ratio):
    #       old_cross_section_area = ring_points_optimizer.get_area(closed_cross_section_points_2d)
    #       scale = np.sqrt(new_cross_section_area / old_cross_section_area)
    
    scaled_ring_points = scale * ring_points
    center = common.get_centroid(scaled_ring_points[:, :, 0])
    center = np.expand_dims(np.broadcast_to(center, (num_ring_points, 3)), 2)
    true_center = np.expand_dims(np.broadcast_to(np.expand_dims(centroid, 0), (num_ring_points, 3)), 2)
    scaled_ring_points += true_center - center
    
    u_bar = scaled_ring_points - ring_points
    assert(u_bar.shape == (num_ring_points, 3, 1))
    u_bar = u_bar.reshape((3 * num_ring_points, 1))
    ring_forces = np.linalg.solve(kappa, u_bar)
    assert(ring_forces.shape == (3 * num_ring_points, 1))
    
    return ring_points, ring_forces

def get_displacement_needed_for_prescribed_displacement(radius, area_percent_change, affine_type):
    # radius is radius at force center point id
    old_area = np.pi * radius ** 2
    new_area = old_area * area_percent_change / 100
    new_radius = np.sqrt(new_area / np.pi)
    if affine_type == "stenosis":
        delta_radius = radius - new_radius
    elif affine_type == "aneurysm":
        delta_radius = new_radius - radius
    else:
        sys.exit("Error. affine_type, " + affine_type + ", not recognized.")
    assert(delta_radius >= 0)
    return delta_radius

def run_aneurysm(affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, area_percent_change, num_time_steps, list_of_node_point_indices, list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names):

    affine_type = "aneurysm"
    brush_level = "uniscale"
    extension = ".vtp"
    
    a, b = common.get_a_b(mu, nu)
    
    centerline_polydata = vtk_utils.read_polydata_file(centerline_polydata_input_file_name)
    surface_polydata = vtk_utils.read_polydata_file(surface_polydata_input_file_name)
    surface_polydata_copy = vtk_utils.read_polydata_file(surface_polydata_input_file_name)
    
    other_geometry_polydatas = []
    for other_geometry_polydata_input_file_name in list_of_other_geometry_polydata_input_file_names:
        other_geometry_polydatas.append(vtk_utils.read_polydata_file(other_geometry_polydata_input_file_name))
    
    data = define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
    assert(list_of_node_point_indices is not None)
    data = define_nodes_affine(data, list_of_node_point_indices)
    assert(force_center_point_id is not None)
    data = assign_force_location_affine_v2(data, force_center_point_id)
        
    data_surface_copy = {"points" : {"surface" : copy.deepcopy(v2n(surface_polydata_copy.GetPoints().GetData()))}}
    
    centerline_polydata = add_node_data_to_centerline_polydata_affine(data, centerline_polydata)
    
    vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_run_scale_original" + extension, centerline_polydata)
    
    origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, data["nodes"]["force_center_point_id"])
    current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
    delta_radius = get_displacement_needed_for_prescribed_displacement(current_radius, area_percent_change, affine_type) / num_time_steps
    
    for it in range(num_time_steps):
        eps = affine_params["eps"][model] * current_radius
        s = get_force_matrix_scale(affine_params["scale"][model] * current_radius / num_time_steps, a, b)
        
        surface_displacements = get_affine_displacements_v2(data, a, b, eps, s, phi_type, "surface", None)
        
        ###################################
        # get what the resulting radius would be, to determine the approriate scaling factor to achieve the prescribed radius change
        data_surface_copy = common.update_points_with_displacements(data_surface_copy, surface_displacements, "surface")
        surface_polydata_copy = common.update_polydata_with_points(surface_polydata_copy, data_surface_copy, "surface")
        tentative_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata_copy, origin, normal) / np.pi)
        surface_displacements_norm = tentative_radius - current_radius
        surface_mesh_scale_factor = delta_radius / surface_displacements_norm
        surface_displacements *= surface_mesh_scale_factor
        ###################################
        
        # centerline_displacements = get_affine_displacements_v2(data, a, b, eps, s, phi_type, "centerline", surface_mesh_scale_factor)
        
        data = common.update_points_with_displacements(data, surface_displacements, "surface")
        # data = common.update_points_with_displacements(data, centerline_displacements, "centerline")
        
        surface_polydata = common.update_polydata_with_points(surface_polydata, data, "surface")
        # centerline_polydata = common.update_polydata_with_points(centerline_polydata, data, "centerline")
        
        for ig in range(len(other_geometry_polydatas)):
            other_geometry_displacements = get_affine_displacements_v2(data, a, b, eps, s, phi_type, "other_geometry_" + str(ig), surface_mesh_scale_factor)
            data = common.update_points_with_displacements(data, other_geometry_displacements, "other_geometry_" + str(ig))
            other_geometry_polydatas[ig] = common.update_polydata_with_points(other_geometry_polydatas[ig], data, "other_geometry_" + str(ig))
        
        data_surface_copy = {"points" : {"surface" : copy.deepcopy(v2n(surface_polydata.GetPoints().GetData()))}}
        surface_polydata_copy = common.update_polydata_with_points(surface_polydata_copy, data_surface_copy, "surface")
        current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        
        # centerline_polydata = vtk_utils.update_centerline_polydata_areas(centerline_polydata, surface_polydata, [data["nodes"]["force_center_point_id"]])
    
    vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + str(num_time_steps) + extension, centerline_polydata)
    
    surface_polydata = vtk_utils.update_surface_polydata_normals(surface_polydata)
        
    vtk_utils.write_polydata(surface_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + str(num_time_steps) + extension, surface_polydata)
    
    for ig in range(len(other_geometry_polydatas)):
        vtk_utils.write_polydata(list_of_other_geometry_polydata_output_file_names[ig] + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + str(num_time_steps) + extension, other_geometry_polydatas[ig])

def run_stent(affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, area_percent_change, num_time_steps, list_of_node_point_indices, list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names):

    affine_type = "aneurysm"
    brush_level = "uniscale"
    extension = ".vtp"
    
    a, b = common.get_a_b(mu, nu)
    
    centerline_polydata = vtk_utils.read_polydata_file(centerline_polydata_input_file_name)
    surface_polydata = vtk_utils.read_polydata_file(surface_polydata_input_file_name)
    surface_polydata_copy = vtk_utils.read_polydata_file(surface_polydata_input_file_name)
    
    other_geometry_polydatas = []
    for other_geometry_polydata_input_file_name in list_of_other_geometry_polydata_input_file_names:
        other_geometry_polydatas.append(vtk_utils.read_polydata_file(other_geometry_polydata_input_file_name))
    
    data = define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
    assert(list_of_node_point_indices is not None)
    data = define_nodes_affine(data, list_of_node_point_indices)
    assert(force_center_point_id is not None)
    data = assign_force_location_affine_v2(data, force_center_point_id)
        
    data_surface_copy = {"points" : {"surface" : copy.deepcopy(v2n(surface_polydata_copy.GetPoints().GetData()))}}
    
    centerline_polydata = add_node_data_to_centerline_polydata_affine(data, centerline_polydata)
    
    vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_run_scale_original" + extension, centerline_polydata)
    
    origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, data["nodes"]["force_center_point_id"])
    current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
    delta_radius = get_displacement_needed_for_prescribed_displacement(current_radius, area_percent_change, affine_type) / num_time_steps
    
    for it in range(num_time_steps):
        eps = affine_params["eps"][model] * current_radius
        s = get_force_matrix_scale(affine_params["scale"][model] * current_radius / num_time_steps, a, b)
        
        surface_displacements = get_affine_displacements_v2(data, a, b, eps, s, phi_type, "surface", None)
        
        ###################################
        # get what the resulting radius would be, to determine the approriate scaling factor to achieve the prescribed radius change
        data_surface_copy = common.update_points_with_displacements(data_surface_copy, surface_displacements, "surface")
        surface_polydata_copy = common.update_polydata_with_points(surface_polydata_copy, data_surface_copy, "surface")
        tentative_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata_copy, origin, normal) / np.pi)
        surface_displacements_norm = tentative_radius - current_radius
        surface_mesh_scale_factor = delta_radius / surface_displacements_norm
        surface_displacements *= surface_mesh_scale_factor
        ###################################
        
        centerline_displacements = get_affine_displacements_v2(data, a, b, eps, s, phi_type, "centerline", surface_mesh_scale_factor)
        
        data = common.update_points_with_displacements(data, surface_displacements, "surface")
        data = common.update_points_with_displacements(data, centerline_displacements, "centerline")
        
        surface_polydata = common.update_polydata_with_points(surface_polydata, data, "surface")
        centerline_polydata = common.update_polydata_with_points(centerline_polydata, data, "centerline")
        
        for ig in range(len(other_geometry_polydatas)):
            other_geometry_displacements = get_affine_displacements_v2(data, a, b, eps, s, phi_type, "other_geometry_" + str(ig), surface_mesh_scale_factor)
            data = common.update_points_with_displacements(data, other_geometry_displacements, "other_geometry_" + str(ig))
            other_geometry_polydatas[ig] = common.update_polydata_with_points(other_geometry_polydatas[ig], data, "other_geometry_" + str(ig))
        
        data_surface_copy = {"points" : {"surface" : copy.deepcopy(v2n(surface_polydata.GetPoints().GetData()))}}
        surface_polydata_copy = common.update_polydata_with_points(surface_polydata_copy, data_surface_copy, "surface")
        current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        
        centerline_polydata = vtk_utils.update_centerline_polydata_areas(centerline_polydata, surface_polydata, [data["nodes"]["force_center_point_id"]])
    
    vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + str(num_time_steps) + extension, centerline_polydata)
    
    surface_polydata = vtk_utils.update_surface_polydata_normals(surface_polydata)
        
    vtk_utils.write_polydata(surface_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + str(num_time_steps) + extension, surface_polydata)
    
    for ig in range(len(other_geometry_polydatas)):
        vtk_utils.write_polydata(list_of_other_geometry_polydata_output_file_names[ig] + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + str(num_time_steps) + extension, other_geometry_polydatas[ig])
