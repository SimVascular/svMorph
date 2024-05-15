# import os
# import sys
import vtk
# import math
# import copy
# from time import perf_counter
# import textwrap
import numpy as np
# import collections
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

import vtk_utils
import common

def get_mesh_point_to_closest_centerline_point_map(centerline_polydata, mesh_polydata, mesh_type):
    mesh_point_to_closest_centerline_point_map = {}
    
    centerline_points_dataset = vtk.vtkPolyData()
    centerline_points_dataset.SetPoints(centerline_polydata.GetPoints())

    locator = vtk.vtkPointLocator()
    locator.Initialize()
    locator.SetDataSet(centerline_points_dataset)
    locator.BuildLocator()
    
    num_mesh_points = mesh_polydata.GetNumberOfPoints()
    for mesh_point_id in range(num_mesh_points):
        mesh_point_to_closest_centerline_point_map[mesh_point_id] = locator.FindClosestPoint(mesh_polydata.GetPoint(mesh_point_id))
    
    if (mesh_type == "centerline") and (centerline_polydata == mesh_polydata):
        for mesh_point_id in range(num_mesh_points): 
            assert(mesh_point_to_closest_centerline_point_map[mesh_point_id] == mesh_point_id)
    
    return mesh_point_to_closest_centerline_point_map

"""
Inputs:
    full_centerline_polydata_input_file_name
        - for the full centerline model of the entire surface, not just a single branch of the centerline. This is necessary because this mask is generated for each surface point based on its closest centerline point in the full centerline model.
"""
def mask_centerline_branch_bifurcation_id(num_mesh_points, mesh_type, surface_polydata, centerline_polydata, other_geometry_polydata, full_centerline_polydata_input_file_name, branch_ids_for_deformation, bifurcation_ids_for_deformation):
    assert(type(branch_ids_for_deformation) == list)
    full_centerline_polydata = vtk_utils.read_polydata_file(full_centerline_polydata_input_file_name)
    if mesh_type == "centerline":
        mesh_point_to_closest_centerline_point_map = get_mesh_point_to_closest_centerline_point_map(full_centerline_polydata, centerline_polydata, "centerline")
    elif mesh_type == "surface":
        mesh_point_to_closest_centerline_point_map = get_mesh_point_to_closest_centerline_point_map(full_centerline_polydata, surface_polydata, "surface")
    elif mesh_type[:15] == "other_geometry_":
        mesh_point_to_closest_centerline_point_map = get_mesh_point_to_closest_centerline_point_map(full_centerline_polydata, other_geometry_polydata, "other_geometry")
    else:
        raise Exception("Error. mesh_type, " + mesh_type + ", is not recognized.")
    mask = np.zeros(num_mesh_points)
    for mesh_point_id, full_centerline_point_id in mesh_point_to_closest_centerline_point_map.items():
        branch_id = full_centerline_polydata.GetPointData().GetArray("BranchId").GetTuple1(full_centerline_point_id)
        bifirucation_id = full_centerline_polydata.GetPointData().GetArray("BifurcationId").GetTuple1(full_centerline_point_id)
        if branch_id in branch_ids_for_deformation or bifirucation_id in bifurcation_ids_for_deformation:
            mask[mesh_point_id] = 1
    mask = common.horizontal_broadcast(mask, 3)
    return mask