# import os
import sys
import vtk
# import math
import copy
# from time import perf_counter
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

import vtk_utils
import common
# import mask
import ring_points_optimizer

def define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas):
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
    data["nodes"]["all_indices"] = copy.deepcopy(list_of_node_point_indices)
    return data

def add_node_data_to_centerline_polydata_affine(data, centerline_polydata):
    num_centerline_points = data["points"]["centerline"].shape[0]
    is_node = np.zeros(num_centerline_points)
    is_node[data["nodes"]["all_indices"]] = 1
    is_node[data["nodes"]["force_center_point_id"]] = 2
    centerline_polydata = vtk_utils.add_point_data_array_to_polydata(centerline_polydata, "nodes", is_node)
    return centerline_polydata

def assign_force_location_affine_v2(data, point_id):
    assert(point_id in data["nodes"]["all_indices"])
    data["nodes"]["force_center_point_id"] = point_id
    return data

def get_force_matrix_scale(scale, a, b):
    return scale * (2 / 5) / (2 * b - a)

"""
Evaluate equation 16 of De Goes 2017

Inputs:
    rv: array of shape (num_mesh_points, num_kelvinlet_points, 3) 
        where rv[i, j, 0] = x_i - x0_j
        where rv[i, j, 1] = y_i - y0_j
        where rv[i, j, 2] = z_i - z0_j
"""
def kelvinlets_affine_v3(rv, a, b, eps, s):
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

"""
For each centerline point from first_centerline_point_id to last_centerline_point_id, get the rotation matrix needed to rotate a vector (a cartesian basis vector) from the cartesian coordinate system to the local centerline coordinate system (local system at the centerline point).
"""
def get_rotation_matrix_v2(data, first_centerline_point_id, last_centerline_point_id, cartesian_axis, num_copies):
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

def get_affine_displacements_v2(data, a, b, eps, s, phi_type, mesh_type, surface_mesh_scale_factor):

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

def get_cross_section_ring_points(data, origin, normal, surface_polydata, num_ring_points, save_slice_to_file = False):
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

def get_ring_point_and_forces_v2(data, a, b, eps, origin, normal, surface_polydata, num_ring_points, new_cross_section_area, falloff_type):
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
    
    scale = ring_points_optimizer.find_scale_to_get_prescribed_area(closed_cross_section_points_2d, new_cross_section_area) # multiplicative scale that scales ring points, such that the area of the scaled closed ring points matches desired area, new_cross_section_area # todo: replace this code with the section below (because the scale predicted from the optimizer is just square root of the area ratio):
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

def get_ring_displacements_v2(data, a, b, eps, mesh_type, ring_points, ring_forces, falloff_type):
    
    if mesh_type != "surface" and mesh_type != "centerline" and (mesh_type[:15] != "other_geometry_"):
        sys.exit("Error. mesh_type, " + mesh_type + ", is not valid.")
    
    num_mesh_points = data["points"][mesh_type].shape[0]
    force_center_point_id = data["nodes"]["force_center_point_id"]
    num_ring_points = ring_points.shape[0]
    
    # compute kelvinlets at each ring point
    kelvinlet_matrix = np.zeros((num_mesh_points, 3, 3 * num_ring_points))
    if falloff_type == "regular":
        kk = common.kelvinlets_translation_v2(  data["points"][mesh_type][:, 0], 
                                                data["points"][mesh_type][:, 1], 
                                                data["points"][mesh_type][:, 2], 
                                                ring_points[:, 0, 0],
                                                ring_points[:, 1, 0],
                                                ring_points[:, 2, 0],
                                                a, b, eps)
    elif falloff_type == "laplacian":
        kk = common.laplacian_kelvinlets_translation_v2(data["points"][mesh_type][:, 0], 
                                                        data["points"][mesh_type][:, 1], 
                                                        data["points"][mesh_type][:, 2], 
                                                        ring_points[:, 0, 0],
                                                        ring_points[:, 1, 0],
                                                        ring_points[:, 2, 0],
                                                        a, b, eps)
    elif falloff_type == "bilaplacian":
        kk = common.bilaplacian_kelvinlets_translation_v2(  data["points"][mesh_type][:, 0], 
                                                            data["points"][mesh_type][:, 1], 
                                                            data["points"][mesh_type][:, 2], 
                                                            ring_points[:, 0, 0],
                                                            ring_points[:, 1, 0],
                                                            ring_points[:, 2, 0],
                                                            a, b, eps)
    else:
        sys.exit("Error. falloff_type, ", falloff_type, ", is not recognized.")
    assert(kk.shape == (num_mesh_points, num_ring_points, 3, 3))
    
    for l in range(num_ring_points):
        kelvinlet_matrix[:, :, 3 * l : 3 * (l + 1)] = kk[:, l, :, :]
    
    # multiply the kelvinlets by the forces and add the displacements to get the total displacement due to the contribution of a force-applying kelvinlet located at each ring point
    displacement = np.matmul(kelvinlet_matrix, ring_forces)
    assert(displacement.shape == (num_mesh_points, 3, 1))
    displacement = np.squeeze(displacement)
    assert(displacement.shape == (num_mesh_points, 3))
    
    return displacement

def get_displacement_needed_for_prescribed_displacement(radius_at_force_center_point_id, area_percent_change, affine_type):
    radius = radius_at_force_center_point_id
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

def run_stenosis_v4(affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, force_center_point_id, num_ring_points, area_percent_change, num_time_steps, list_of_node_point_indices, falloff_type, list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names, weight_regularized_laplacian):
    
    affine_type = "stenosis"
    brush_level = "uniscale"
    phi_type = "point"
    extension = ".vtp"

    assert((0 <= weight_regularized_laplacian) and (weight_regularized_laplacian <= 1))
    
    a, b = common.get_a_b(mu, nu)
    
    centerline_polydata = vtk_utils.read_polydata_file(centerline_polydata_input_file_name)
    surface_polydata = vtk_utils.read_polydata_file(surface_polydata_input_file_name)
    
    other_geometry_polydatas = []
    for other_geometry_polydata_input_file_name in list_of_other_geometry_polydata_input_file_names:
        other_geometry_polydatas.append(vtk_utils.read_polydata_file(other_geometry_polydata_input_file_name))
    
    data = define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
    assert(list_of_node_point_indices is not None)
    data = define_nodes_affine(data, list_of_node_point_indices)
    assert(force_center_point_id is not None)
    data = assign_force_location_affine_v2(data, force_center_point_id)
    
    centerline_polydata = add_node_data_to_centerline_polydata_affine(data, centerline_polydata)
    
    vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + falloff_type + "_run_stenosis_original" + extension, centerline_polydata)
    
    origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, data["nodes"]["force_center_point_id"])
    original_area = vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal)
    target_area = original_area * area_percent_change / 100
    delta_area = (target_area - original_area) / num_time_steps
    
    for it in range(num_time_steps):
        
        print("---------------------------------------------------------------------- it = ", it)
        
        current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        eps = affine_params["eps"][model] * current_radius
        
        ring_points, ring_forces = get_ring_point_and_forces_v2(data, a, b, eps, np.array(origin), normal, surface_polydata, num_ring_points, original_area + delta_area * (it + 1), falloff_type)
        
        surface_displacements = get_ring_displacements_v2(data, a, b, eps, "surface", ring_points, ring_forces, falloff_type)
        
        centerline_displacements = get_ring_displacements_v2(data, a, b, eps, "centerline", ring_points, ring_forces, falloff_type)
        
        if falloff_type == "regular":
            ring_points_lap, ring_forces_lap = get_ring_point_and_forces_v2(data, a, b, eps, np.array(origin), normal, surface_polydata, num_ring_points, original_area + delta_area * (it + 1), "laplacian")
            surface_displacements_lap = get_ring_displacements_v2(data, a, b, eps, "surface", ring_points_lap, ring_forces_lap, "laplacian")
            centerline_displacements_lap = get_ring_displacements_v2(data, a, b, eps, "centerline", ring_points_lap, ring_forces_lap, "laplacian")
            surface_displacements = weight_regularized_laplacian * surface_displacements + (1 - weight_regularized_laplacian) * surface_displacements_lap
            centerline_displacements = weight_regularized_laplacian * centerline_displacements + (1 - weight_regularized_laplacian) * centerline_displacements_lap
        
        data = common.update_points_with_displacements(data, surface_displacements, "surface")
        data = common.update_points_with_displacements(data, centerline_displacements, "centerline")
        
        surface_polydata = common.update_polydata_with_points(surface_polydata, data, "surface")
        centerline_polydata = common.update_polydata_with_points(centerline_polydata, data, "centerline")
        
        for ig in range(len(other_geometry_polydatas)):
            other_geometry_displacements = get_ring_displacements_v2(data, a, b, eps, "other_geometry_" + str(ig), ring_points, ring_forces, falloff_type)
            if falloff_type == "regular":
                other_geometry_displacements_lap = get_ring_displacements_v2(data, a, b, eps, "other_geometry_" + str(ig), ring_points_lap, ring_forces_lap, "laplacian")
                other_geometry_displacements = weight_regularized_laplacian * other_geometry_displacements + (1 - weight_regularized_laplacian) * other_geometry_displacements_lap
            data = common.update_points_with_displacements(data, other_geometry_displacements, "other_geometry_" + str(ig))
            other_geometry_polydatas[ig] = common.update_polydata_with_points(other_geometry_polydatas[ig], data, "other_geometry_" + str(ig))
        
        centerline_polydata = vtk_utils.update_centerline_polydata_areas(centerline_polydata, surface_polydata, [data["nodes"]["force_center_point_id"]])
    
    vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + falloff_type + "_" + str(num_time_steps) + extension, centerline_polydata)
    
    surface_polydata = vtk_utils.update_surface_polydata_normals(surface_polydata)
        
    vtk_utils.write_polydata(surface_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + falloff_type + "_" + str(num_time_steps) + extension, surface_polydata)
    
    for ig in range(len(other_geometry_polydatas)):
        vtk_utils.write_polydata(list_of_other_geometry_polydata_output_file_names[ig] + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + falloff_type + "_" + str(num_time_steps) + extension, other_geometry_polydatas[ig])