# import os
# import sys
import vtk
# import math
import copy
# from time import perf_counter
# import textwrap
import numpy as np
import functools
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

from kelvinlet_core import vtk_utils

"""
As defined below eqn 3 in De Goes 2017
"""
def get_a_b(mu, nu):
    a = 1 / (4 * np.pi * mu)
    b = a / (4 * (1 - nu))
    return a, b

def phi_linear(x0, x1, y0, y1, x):
    assert(x0 <= x)
    assert(x <= x1)
    assert(x0 != x1)
    return (y1 - y0) / (x1 - x0) * (x - x0) + y0 # y = mx + b

def get_radius_at_point(centerline_polydata, point_id):
    area = centerline_polydata.GetPointData().GetArray("CenterlineSectionArea").GetTuple1(point_id)
    radius = np.sqrt(area / np.pi)
    return radius

def update_polydata_with_points(polydata, data, mesh_type):
    polypoints = polydata.GetPoints()
    polypoints.SetData(n2v(data["points"][mesh_type]))
    polydata.Modified()
    return polydata

def update_polydata_with_points_jonathan(polydata, data, mesh_type):
    new_polydata_points = vtk.vtkPoints()
    new_polydata_points.SetData(n2v(data["points"][mesh_type]))
    polydata.SetPoints(new_polydata_points)
    polydata.GetPoints().Modified()
    return polydata

def update_points_with_displacements(data, displacements, mesh_type):
    data["points"][mesh_type] += displacements
    return data

"""
Get the rotation matrix needed to rotate unit vector a into unit vector b, so that R * a = b.
"""
def get_rotation_matrix_between_two_vectors(a, b, num_copies = 1):
    # reference: https://math.stackexchange.com/a/2672702
    # assert(a != -b).all()
    if (a != b).all(): # todo: change this to "if not np.array_equal(a, b):"
        assert(a.shape == (3, 1))
        assert(b.shape == (3, 1))
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        assert(abs(np.linalg.norm(a) - 1) < 1e-6)
        assert(abs(np.linalg.norm(b) - 1) < 1e-6)
        rotation_matrix = 2 * np.matmul(a + b, (a + b).T) / np.matmul((a + b).T, a + b)[0, 0] - np.eye(3) # todo: figure out a better way to compute the rotation matrix that rotates vector a into b, because when a == b, the rotation_matrix that i get from this eqn is not the identity matrix even though it ideally should be...
        # if (a == b).all():
        #     np.testing.assert_equal(rotation_matrix, np.eye(3))
    else:
        rotation_matrix = np.eye(3)
    
    rotation_matrix = np.vstack([np.expand_dims(rotation_matrix, 0) for i in range(num_copies)])
    assert(rotation_matrix.shape == (num_copies, 3, 3))
    
    return rotation_matrix

def horizontal_broadcast(mask, num_copies):
    len_mask = len(mask)
    assert(mask.shape == (len_mask, ))
    broadcasted_mask = np.hstack(np.expand_dims(mask, 1) for i in range(num_copies))
    return broadcasted_mask

def apply_mask(displacements, mask):
    assert(displacements.shape == mask.shape)
    return np.multiply(displacements, mask)

def kelvinlets_translation_v2(x, y, z, x0, y0, z0, a, b, eps):
    n = len(x)
    m = len(x0)

    # Pre-allocate rv array to avoid dstack
    rv = np.empty((n, m, 3, 1))
    rv[:, :, 0, 0] = x.reshape((n, 1)) - x0.reshape((1, m))
    rv[:, :, 1, 0] = y.reshape((n, 1)) - y0.reshape((1, m))
    rv[:, :, 2, 0] = z.reshape((n, 1)) - z0.reshape((1, m))

    # Calculate re directly with added epsilon term
    re = np.sqrt(np.sum(rv[..., 0] ** 2, axis=2) + eps**2)
    re = re.reshape((n, m, 1, 1))  # Ensure re has shape (n, m, 1, 1)
    re3 = re ** 3

    # Preallocate identities
    identity3 = np.eye(3)
    identities = np.tile(identity3, (n, m, 1, 1))

    # Calculate K components
    K = ((a - b) / re) * identities
    rvT = rv.transpose((0, 1, 3, 2))  # Transpose rv to shape (n, m, 1, 3)
    K += (b / re3) * np.matmul(rv, rvT)
    K += (a / 2 * eps**2 / re3) * identities

    return K


"""
Eqn 6 of De Goes 2017

Inputs:
    x:  array of shape (n, )
    y:  array of shape (n, )
    z:  array of shape (n, )
    x0: array of shape (m, )
    y0: array of shape (m, )
    z0: array of shape (m, )
"""
def kelvinlets_translation_v2_jonathan(x, y, z, x0, y0, z0, a, b, eps):
    n = len(x) # number of points for which to compute the kelvinlets
    m = len(x0) # number of forces applied
    # assert(len(y) == n)
    # assert(len(z) == n)
    # assert(len(y0) == m)
    # assert(len(z0) == m)
    
    # start = perf_counter()
    x = x.reshape((n, 1, 1, 1))
    y = y.reshape((n, 1, 1, 1))
    z = z.reshape((n, 1, 1, 1))
    # print("1 reshaping time = ", perf_counter() - start)
    
    # start = perf_counter()
    x = np.broadcast_to(x, (n, m, 1, 1))
    y = np.broadcast_to(y, (n, m, 1, 1))
    z = np.broadcast_to(z, (n, m, 1, 1))
    # print("1 broadcast time = ", perf_counter() - start)
    
    # start = perf_counter()
    x0 = x0.reshape((1, m, 1, 1))
    y0 = y0.reshape((1, m, 1, 1))
    z0 = z0.reshape((1, m, 1, 1))
    # print("2 reshaping time = ", perf_counter() - start)
    
    # start = perf_counter()
    x0 = np.broadcast_to(x0, (n, m, 1, 1))
    y0 = np.broadcast_to(y0, (n, m, 1, 1))
    z0 = np.broadcast_to(z0, (n, m, 1, 1))
    # print("2 broadcast time = ", perf_counter() - start)
    
    # start = perf_counter()
    rv = np.dstack((x - x0, y - y0, z - z0))
    # assert(rv.shape == (n, m, 3, 1))
    # print("dstack time = ", perf_counter() - start)
    
    # start = perf_counter()
    r = np.linalg.norm(rv, 2, axis = 2)
    re = np.sqrt( r**2 + eps**2 )
    # print("norm time = ", perf_counter() - start)
    
    # start = perf_counter()
    re = np.broadcast_to(np.expand_dims(re, 3), (n, m, 3, 3))
    # print("3 broadcast time = ", perf_counter() - start)
    re3 = re ** 3
    
    # try not stacking but preallocating array bc stacking memory is not contiguous
    
    # start = perf_counter()
    identity3 = np.expand_dims(np.eye(3), 0)
    identities = np.vstack([identity3 for i in range(n)])
    identities = np.expand_dims(identities, 1)
    identities = np.hstack([identities for i in range(m)])
    # print("identity time = ", perf_counter() - start)
    
    # start = perf_counter()
    K = np.multiply( (a - b) / re, identities ) 
    # print("1 multiply math time = ", perf_counter() - start)
    
    # start = perf_counter()
    rvT = np.transpose(rv, axes = (0, 1, 3, 2))
    # print("transpose math time = ", perf_counter() - start)
    
    # start = perf_counter()
    K2 = np.matmul(rv, rvT)
    # print("matmul math time = ", perf_counter() - start)
    
    # start = perf_counter()
    K += np.multiply( b / re3, K2 ) 
    # print("3 multiply math time = ", perf_counter() - start)
    
    # start = perf_counter()
    K += np.multiply( a / 2 * eps**2 / re3, identities )
    # print("2 multiply math time = ", perf_counter() - start)
    
    assert(K.shape == (n, m, 3, 3))
    
    return K

"""
Eqn 15 and 2 of De Goes 2019

Inputs:
    x:  array of shape (n, )
    y:  array of shape (n, )
    z:  array of shape (n, )
    x0: array of shape (m, )
    y0: array of shape (m, )
    z0: array of shape (m, )
"""
def laplacian_kelvinlets_translation_v2(x, y, z, x0, y0, z0, a, b, eps):
    n = len(x)
    m = len(x0)

    # Pre-allocate rv array to avoid dstack
    rv = np.empty((n, m, 3, 1))
    rv[:, :, 0, 0] = x.reshape((n, 1)) - x0.reshape((1, m))
    rv[:, :, 1, 0] = y.reshape((n, 1)) - y0.reshape((1, m))
    rv[:, :, 2, 0] = z.reshape((n, 1)) - z0.reshape((1, m))

    # Calculate re directly, incorporating epsilon squared
    re = np.sqrt(np.sum(rv[..., 0] ** 2, axis=2) + eps**2)
    re = re.reshape((n, m, 1, 1))  # Ensure re has shape (n, m, 1, 1)
    re2 = re ** 2
    re7 = re ** 7

    # Calculate r^2 and broadcast it
    r2 = np.sum(rv[..., 0] ** 2, axis=2).reshape((n, m, 1, 1))

    # Preallocate identities without stacking
    identity3 = np.eye(3)
    identities = np.tile(identity3, (n, m, 1, 1))

    # Calculate the main component K
    term1 = 15 * a * eps**4
    term2 = 2 * b * re2 * (5 * eps**2 + 2 * r2)
    K = (term1 - term2) / (2 * re7) * identities

    # Calculate K2 component and add it to K
    rvT = rv.transpose((0, 1, 3, 2))  # Transpose rv to shape (n, m, 1, 3)
    K2 = np.matmul(rv, rvT)
    K += (3 * b * (7 * eps**2 + 2 * r2) / re7) * K2
    
    return K


def laplacian_kelvinlets_translation_v2_jonathan(x, y, z, x0, y0, z0, a, b, eps):
    n = len(x) # number of points for which to compute the kelvinlets
    m = len(x0) # number of forces applied
    assert(len(y) == n)
    assert(len(z) == n)
    assert(len(y0) == m)
    assert(len(z0) == m)
    
    x = x.reshape((n, 1, 1, 1))
    y = y.reshape((n, 1, 1, 1))
    z = z.reshape((n, 1, 1, 1))
    
    x = np.broadcast_to(x, (n, m, 1, 1))
    y = np.broadcast_to(y, (n, m, 1, 1))
    z = np.broadcast_to(z, (n, m, 1, 1))
    
    x0 = x0.reshape((1, m, 1, 1))
    y0 = y0.reshape((1, m, 1, 1))
    z0 = z0.reshape((1, m, 1, 1))
    
    x0 = np.broadcast_to(x0, (n, m, 1, 1))
    y0 = np.broadcast_to(y0, (n, m, 1, 1))
    z0 = np.broadcast_to(z0, (n, m, 1, 1))
    
    rv = np.dstack((x - x0, y - y0, z - z0))
    assert(rv.shape == (n, m, 3, 1))
    
    r = np.linalg.norm(rv, 2, axis = 2)
    re = np.sqrt( r**2 + eps**2 )
    
    r = np.broadcast_to(np.expand_dims(r, 3), (n, m, 3, 3))
    re = np.broadcast_to(np.expand_dims(re, 3), (n, m, 3, 3))
    r2  =  r ** 2
    re2 = re ** 2
    re7 = re ** 7
    
    identity3 = np.expand_dims(np.eye(3), 0)
    identities = np.vstack([identity3 for i in range(n)])
    identities = np.expand_dims(identities, 1)
    identities = np.hstack([identities for i in range(m)])
    
    K = np.multiply( np.divide(15 * a * eps**4 - np.multiply(2 * b * re2, 5 * eps**2 + 2 * r2), 2 * re7), identities )
    
    rvT = np.transpose(rv, axes = (0, 1, 3, 2))
    
    K2 = np.matmul(rv, rvT)
    
    K += np.multiply( np.divide(3 * b * (7 * eps**2 + 2 * r2), re7), K2 )
    
    # K *= -1
    
    assert(K.shape == (n, m, 3, 3))
    
    return K

"""
Eqn 15 and 2 of De Goes 2019

Inputs:
    x:  array of shape (n, )
    y:  array of shape (n, )
    z:  array of shape (n, )
    x0: array of shape (m, )
    y0: array of shape (m, )
    z0: array of shape (m, )
"""
def bilaplacian_kelvinlets_translation_v2(x, y, z, x0, y0, z0, a, b, eps): # todo: combine bilaplacian_kelvinlets_translation_v2, laplacian_kelvinlets_translation_v2, and kelvinlets_translation_v2 into one single function that can do all three, depending on the falloff_type specified
    n = len(x) # number of points for which to compute the kelvinlets
    m = len(x0) # number of forces applied
    assert(len(y) == n)
    assert(len(z) == n)
    assert(len(y0) == m)
    assert(len(z0) == m)
    
    x = x.reshape((n, 1, 1, 1))
    y = y.reshape((n, 1, 1, 1))
    z = z.reshape((n, 1, 1, 1))
    
    x = np.broadcast_to(x, (n, m, 1, 1))
    y = np.broadcast_to(y, (n, m, 1, 1))
    z = np.broadcast_to(z, (n, m, 1, 1))
    
    x0 = x0.reshape((1, m, 1, 1))
    y0 = y0.reshape((1, m, 1, 1))
    z0 = z0.reshape((1, m, 1, 1))
    
    x0 = np.broadcast_to(x0, (n, m, 1, 1))
    y0 = np.broadcast_to(y0, (n, m, 1, 1))
    z0 = np.broadcast_to(z0, (n, m, 1, 1))
    
    rv = np.dstack((x - x0, y - y0, z - z0))
    assert(rv.shape == (n, m, 3, 1))
    
    r = np.linalg.norm(rv, 2, axis = 2)
    re = np.sqrt( r**2 + eps**2 )
    
    r = np.broadcast_to(np.expand_dims(r, 3), (n, m, 3, 3))
    re = np.broadcast_to(np.expand_dims(re, 3), (n, m, 3, 3))
    r2  =  r ** 2
    re2 = re ** 2
    re11 = re ** 11
    
    identity3 = np.expand_dims(np.eye(3), 0)
    identities = np.vstack([identity3 for i in range(n)])
    identities = np.expand_dims(identities, 1)
    identities = np.hstack([identities for i in range(m)])
    
    K = np.multiply( np.divide(105 * eps**4 * (3 * a * (eps ** 2 - 2 * r2) - 2 * b * re2), 2 * re11), identities )
    
    rvT = np.transpose(rv, axes = (0, 1, 3, 2))
    
    K2 = np.matmul(rv, rvT)
    
    K += np.multiply( np.divide(945 * b * eps**4, re11), K2 )
    
    assert(K.shape == (n, m, 3, 3))
    
    return K

"""
Assumes points are sequentially numbered.
"""
def get_average_radius_between_two_points(surface_polydata, centerline_polydata, upstream_point_id, downstream_point_id):
    upstream_origin, upstream_normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, upstream_point_id)
    upstream_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, upstream_origin, upstream_normal) / np.pi)
    
    downstream_origin, downstream_normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, downstream_point_id)
    downstream_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, downstream_origin, downstream_normal) / np.pi)
    
    average_radius = (upstream_radius + downstream_radius) / 2
    return average_radius

def get_centroid(points):
    num_points = points.shape[0]
    assert(points.shape == (num_points, 2) or points.shape == (num_points, 3))
    centroid = np.array([np.sum(points, axis = 0) / num_points])
    if points.shape == (num_points, 2):
        assert(centroid.shape == (1, 2))
    else:
        assert(points.shape == (num_points, 3))
        assert(centroid.shape == (1, 3))
    return centroid

"""
Sort 2D points in CCW order, with points at 12 o'clock being ordered first. Returns -1 if point2 is greater than point1, in CCW order, meaning point1 comes before point2. Otherwise, return 1.
This function assumes that the centroid of the point cloud is located at coordinates, (x, y) = (0, 0), and that point1 and point2 and the centroid are not colinear.

Inputs:
    tuple point1
        = tuple with 2 entries, where the first entry is the coordinates of the point and the second entry is the point ID
    tuple point2
        = tuple with 2 entries, where the first entry is the coordinates of the point and the second entry is the point ID

References:
    https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
    https://stackoverflow.com/a/6989383
    https://stackoverflow.com/a/13239857
"""
def sort_ccw_predicate(point1, point2):
    # Points at 12 o'clock are "first". Then follow a CCW order (12 o'clock, 11 o'clock, ..., 2 o'clock, 1 o'clock)
    if point1[0][0] >= 0 and point2[0][0] < 0:
        return 1
    if point1[0][0] < 0 and point2[0][0] >= 0:
        return -1

    # Compute cross product
    determinant = point1[0][0] * point2[0][1] - point1[0][1] * point2[0][0]
    if determinant > 0:
        return -1
        
    return 1 # determinant < 0

def sort_ring_points_in_ccw(ring_points, plane_normal):
    num_ring_points = ring_points.shape[0]
    assert(ring_points.shape == (num_ring_points, 3))
    assert(plane_normal.shape == (3, ))
    
    centroid = get_centroid(ring_points).reshape(3)
    assert(centroid.shape == (3, ))
    
    # get basis vectors defining the plane of the ring_points (assumes all ring points lie on the same plane), where basis vectors, e0 and e1, lie in the plane and basis vector, e2, is normal to the plane
    e2 = plane_normal / np.linalg.norm(plane_normal)
    e0 = ring_points[0] - centroid
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(e2, e0)
    
    # project ring_points onto plane to get 2D points
    points_projected_2d = []
    for ip in range(num_ring_points):
        point_coordinate_relative = ring_points[ip] - centroid
        points_projected_2d.append((np.array([np.dot(point_coordinate_relative, e0), np.dot(point_coordinate_relative, e1)]), ip))
    
    # sort ring_points in CCW order, starting at 12 o'clock.
    points_projected_2d.sort(key = functools.cmp_to_key(sort_ccw_predicate)) # https://stackoverflow.com/a/13239857 # https://statisticsglobe.com/sort-list-custom-comparator-python
    
    points_projected_3d = np.zeros((num_ring_points, 3))
    for ip in range(num_ring_points):
        points_projected_3d[ip] = copy.deepcopy(ring_points[points_projected_2d[ip][1]])
    
    return points_projected_3d