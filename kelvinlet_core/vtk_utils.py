import os
# import sys
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v
import jax.numpy as jnp


def write_polydata(polydata_file_name, polydata):
    _, ext = os.path.splitext(polydata_file_name)
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError('File extension, ' + ext + ', is not recognized.')
    writer.SetFileName(polydata_file_name)
    writer.SetInputData(polydata)
    writer.Update()
    writer.Write()

def read_polydata_file(polydata_file_name):
    _, ext = os.path.splitext(polydata_file_name)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension, ' + ext + ', is not recognized.')
    reader.SetFileName(polydata_file_name)
    reader.Update()
    return reader.GetOutput()

def add_point_data_array_to_polydata(polydata, array_name, point_data_array):
    array = n2v(point_data_array)
    array.SetName(array_name)
    polydata.GetPointData().AddArray(array)
    return polydata

def get_closest_surface_point_to_centerline_point(data, centerline_polydata, surface_polydata, centerline_point_id):
    # todo: delete centerline_polydata since it doesnt get used here
    # todo: move this function to common, because we use "data" here which is not a vtk object
    surface_point_dataset = vtk.vtkPolyData()
    surface_point_dataset.SetPoints(surface_polydata.GetPoints())
    locator = vtk.vtkPointLocator()
    locator.Initialize()
    locator.SetDataSet(surface_point_dataset)
    locator.BuildLocator()
    closest_surface_point_id = locator.FindClosestPoint(data["points"]["centerline"][centerline_point_id])
    print("closest_surface_point_id = ", closest_surface_point_id)
    return closest_surface_point_id

def get_cross_sectional_area_of_triangulated_slice(triangulated_slice):
    # Ensure the slice has points; otherwise, raise an exception
    if not triangulated_slice.GetNumberOfPoints():
        raise Exception('Empty slice')
    # Set up VTK integrator to calculate the surface area
    integrator = vtk.vtkIntegrateAttributes()
    integrator.SetInputData(triangulated_slice)
    integrator.Update()
    # Access the calculated surface area directly from the VTK array
    surface_area = integrator.GetOutput().GetCellData().GetArray('Area').GetValue(0)
    return surface_area

"""
references: 
    /home/jonathanpham/Documents/software/svMorph/svMorph/vtk_utils.py
    https://github.com/StanfordCBCL/DataCuration/blob/master/vtk_functions.py

Inputs:
    vtk.vtkDelaunay2D triangulated_slice
Returns:
    float cross_sectional_area
"""
def get_cross_sectional_area_of_triangulated_slice_jonathan(triangulated_slice):
    integrator = vtk.vtkIntegrateAttributes()
    if not triangulated_slice.GetNumberOfPoints():
        raise Exception('Empty slice')
    integrator.SetInputData(triangulated_slice)
    integrator.Update()
    surface_area = v2n(integrator.GetOutput().GetCellData().GetArray('Area'))[0]
    return surface_area

"""
references: 
    /home/jonathanpham/Documents/software/svMorph/svMorph/vtk_utils.py
    https://github.com/StanfordCBCL/DataCuration/blob/master/vtk_functions.py -- cut_plane()
"""
def cut_polydata(polydata, origin, normal):
    # assert(len(origin) == 3)
    # assert(len(normal) == 3)
    cutting_plane = vtk.vtkPlane()
    cutting_plane.SetOrigin(origin[0], origin[1], origin[2])
    cutting_plane.SetNormal(normal[0], normal[1], normal[2])
    cut = vtk.vtkCutter()
    cut.SetInputData(polydata)
    cut.SetCutFunction(cutting_plane)
    cut.Update()
    return cut.GetOutput()

"""
references: 
    /home/jonathanpham/Documents/software/svMorph/svMorph/vtk_utils.py
    https://github.com/StanfordCBCL/DataCuration/blob/master/vtk_functions.py -- connectivity()
"""
def connectivity(polydata, origin):
    connection = vtk.vtkConnectivityFilter()
    connection.SetInputData(polydata)
    connection.SetExtractionModeToClosestPointRegion()
    connection.SetClosestPoint(origin[0], origin[1], origin[2])
    connection.Update()
    return connection
    
"""
references:
    /home/jonathanpham/Documents/software/svMorph/svMorph/vtk_utils.py
    https://github.com/StanfordCBCL/DataCuration/blob/master/get_mean_flow_3d.py -- slice_vessel()
"""
def slice_polydata(surface_polydata, origin, normal):
    cut = cut_polydata(surface_polydata, origin, normal)
    contour = connectivity(cut, origin) # get the contour closest to the origin, since the cut_polydata() might result in multiple contours extracted
    return contour.GetOutput()

"""
references:
    /home/jonathanpham/Documents/software/svMorph/svMorph/vtk_utils.py
    https://github.com/StanfordCBCL/DataCuration/blob/master/get_mean_flow_3d.py -- get_integral()
"""
def get_triangulated_slice(surface_polydata, origin, normal):
    assert(len(origin) == 3)
    assert(len(normal) == 3)
    triangulated_slice = vtk.vtkDelaunay2D()
    triangulated_slice.SetTolerance(1e-4)
    triangulated_slice.SetInputData(slice_polydata(surface_polydata, origin, normal))
    triangulated_slice.Update()
    return triangulated_slice.GetOutput()

def get_cross_sectional_area(surface_polydata, origin, normal):
    triangulated_slice = get_triangulated_slice(surface_polydata, origin, normal)
    area = get_cross_sectional_area_of_triangulated_slice(triangulated_slice)
    return area
    
def update_centerline_polydata_areas(centerline_polydata, surface_polydata, point_ids):
    # reference: /home/jonathanpham/Documents/software/svMorph/svMorph/graphics2.py - change_lumen_radius_via_vtk_sphere()
    assert(type(point_ids) == list or type(point_ids) == tuple)
    areas = centerline_polydata.GetPointData().GetArray("CenterlineSectionArea")
    for point_id in point_ids:
        origin, normal = get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, point_id)
        area = get_cross_sectional_area(surface_polydata, origin, normal)
        areas.InsertTuple(point_id, (area, ))
    centerline_polydata.GetPointData().AddArray(areas)
    return centerline_polydata

def update_surface_polydata_normals_slow(surface_polydata):
    normals = vtk.vtkPolyDataNormals() # https://kitware.github.io/vtk-examples/site/Cxx/PolyData/SmoothPolyDataFilter/
    normals.ComputeCellNormalsOn()
    normals.SetInputData(surface_polydata)
    # normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    surface_polydata.GetCellData().AddArray(normals.GetOutput().GetCellData().GetArray("Normals"))
    return surface_polydata

# def update_surface_polydata_normals(surface_polydata):
    
def compute_centerline_coordinates(centerline_polydata):
    num_centerline_points = centerline_polydata.GetNumberOfPoints()
    centerline_coordinate_array = np.zeros(num_centerline_points)
    for ip in range(1, num_centerline_points):
        centerline_coordinate_array[ip] = np.linalg.norm(np.array(centerline_polydata.GetPoint(ip)) - np.array(centerline_polydata.GetPoint(ip - 1)))
    centerline_coordinate_array = np.cumsum(centerline_coordinate_array)
    print("total length of centerline = ", centerline_coordinate_array[-1])
    return centerline_coordinate_array

def get_centerline_length(centerline_polydata):
    centerline_coordinate_array = compute_centerline_coordinates(centerline_polydata)
    total_length = centerline_coordinate_array[-1]
    return total_length

def get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, point_id):
    point = jnp.array(centerline_polydata.GetPoint(point_id))  # Directly convert to JAX array
    num_centerline_points = centerline_polydata.GetNumberOfPoints()
    # Compute the normal vector based on the position of point_id in the sequence
    if 0 < point_id < num_centerline_points - 1:
        next_point = jnp.array(centerline_polydata.GetPoint(point_id + 1))
        prev_point = jnp.array(centerline_polydata.GetPoint(point_id - 1))
        normal = next_point - prev_point
    elif point_id == num_centerline_points - 1:
        # Last point, calculate using only previous point
        prev_point = jnp.array(centerline_polydata.GetPoint(point_id - 1))
        normal = point - prev_point
    else:
        # First point, calculate using only the next point
        next_point = jnp.array(centerline_polydata.GetPoint(point_id + 1))
        normal = next_point - point
    # Normalize the normal vector
    normal /= jnp.linalg.norm(normal)
    return point, normal

def get_coordinates_and_normal_at_point_on_centerline_jonathan(centerline_polydata, point_id):
    point = centerline_polydata.GetPoint(point_id)
    num_centerline_points = centerline_polydata.GetNumberOfPoints()
    if 0 < point_id and point_id < num_centerline_points - 1:
        normal = np.array(centerline_polydata.GetPoint(point_id + 1)) - np.array(centerline_polydata.GetPoint(point_id - 1))
    elif 0 < point_id:
        assert(point_id == num_centerline_points - 1)
        normal = np.array(centerline_polydata.GetPoint(point_id)) - np.array(centerline_polydata.GetPoint(point_id - 1))
    else:
        assert(point_id == 0)
        normal = np.array(centerline_polydata.GetPoint(point_id + 1)) - np.array(centerline_polydata.GetPoint(point_id))
    normal /= np.linalg.norm(normal)
    return point, normal

def cap_polydata(uncapped_polydata, hole_size = 100000.0):
    """
    Fill holes in the given polydata and return the filled version.
    
    reference: 
        https://kitware.github.io/vtk-examples/site/Cxx/Meshes/FillHoles/
        https://kitware.github.io/vtk-examples/site/Cxx/Meshes/IdentifyHoles/
        /home/jonathanpham/Documents/software/svMorph/svMorph/simulation/demos/models/cylinder1/fill_hole.py
    """
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(uncapped_polydata)
    fill_holes.SetHoleSize(hole_size)
    fill_holes.Update()
    capped_polydata = fill_holes.GetOutput()
    return capped_polydata

def get_area_change_between_source_and_target(source_centerline_polydata, source_surface_polydata, target_surface_polydata, point_id):
    origin, normal = get_coordinates_and_normal_at_point_on_centerline(source_centerline_polydata, point_id)
    source_area = get_cross_sectional_area(source_surface_polydata, origin, normal)
    target_area = get_cross_sectional_area(target_surface_polydata, origin, normal)
    area_percent_change = (target_area - source_area) / source_area * 100
    return area_percent_change

def get_area_ratio_between_source_and_target(source_centerline_polydata, source_surface_polydata, target_surface_polydata, point_id):
    origin, normal = get_coordinates_and_normal_at_point_on_centerline(source_centerline_polydata, point_id)
    source_area = get_cross_sectional_area(source_surface_polydata, origin, normal)
    target_area = get_cross_sectional_area(target_surface_polydata, origin, normal)
    area_ratio = target_area / source_area * 100
    return area_ratio