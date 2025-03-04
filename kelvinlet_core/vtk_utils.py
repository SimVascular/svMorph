import os
# import sys
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v
import time
import jax.numpy as jnp
import jax as jx


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

# import matplotlib.pyplot as plt
@jx.jit
def estimate_radius(vertices, origin, normal, search_radius):
    # Calculate distances from the origin to all vertices
    distances = jnp.linalg.norm(vertices - origin, axis=1)
    # Sort distances and corresponding vertices
    sorted_indices = jnp.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_vertices = vertices[sorted_indices]
    # Truncate entries that are bigger than the search radius
    nearby_vertices = sorted_vertices[10:100]
    assert nearby_vertices.shape[0] > 0, "No nearby points found"
    # Normalize the normal vector
    normal = normal / jnp.linalg.norm(normal)
    # Project the nearby points onto the plane defined by the normal
    x = nearby_vertices - origin
    # Compute the projection of x onto the plane
    projections = x - jnp.outer(jnp.dot(x, normal), normal)
    projection_lengths = jnp.linalg.norm(projections, axis=1)
    # Estimate the radius as the minimum of the projection lengths
    estimated_radius = jnp.mean(projection_lengths)

    return estimated_radius, sorted_indices[10:100]


@jx.jit
def estimate_radius_nearby(vertices, sorted_indices, origin, normal):
    # Normalize the normal vector
    normal = normal / jnp.linalg.norm(normal)
    # Project the nearby points onto the plane defined by the normal
    nearby_vertices = vertices[sorted_indices]
    x = nearby_vertices - origin
    # Compute the projection of x onto the plane
    projections = x - jnp.outer(jnp.dot(x, normal), normal)
    projection_lengths = jnp.linalg.norm(projections, axis=1)
    # Estimate the radius as the minimum of the projection lengths
    estimated_radius = jnp.mean(projection_lengths)

    return estimated_radius

def estimate_radius_no_jit(vertices, origin, normal, search_radius):
    start_time = time.time()
    # Calculate distances from the origin to all vertices
    distances = jnp.linalg.norm(vertices - origin, axis=1)
    mid_time_1 = time.time()
    print(f"Distance calculation took {mid_time_1 - start_time:.6f} seconds")
    # Filter vertices within the search radius
    nearby_points = vertices[distances < search_radius]
    assert nearby_points.shape[0] > 0, "No nearby points found"
    print("nearby_points.shape[0] = ", nearby_points.shape[0])
    mid_time_2 = time.time()
    print(f"Filtering nearby points took {mid_time_2 - mid_time_1:.6f} seconds")
    # Normalize the normal vector
    normal = normal / jnp.linalg.norm(normal)
    # Project the nearby points onto the plane defined by the normal
    x = nearby_points - origin
    # Compute the projection of x onto the plane
    projections = x - jnp.outer(jnp.dot(x, normal), normal)
    projection_lengths = jnp.linalg.norm(projections, axis=1)
    mid_time_3 = time.time()
    print(f"Projection calculation took {mid_time_3 - mid_time_2:.6f} seconds")
    # Estimate the radius as the minimum of the projection lengths
    filtered_lengths = projection_lengths
    estimated_radius = jnp.min(filtered_lengths)
    end_time = time.time()
    print(f"Radius estimation took {end_time - mid_time_3:.6f} seconds")

    return estimated_radius


def estimate_radius_no_timer(vertices, origin, normal, search_radius):
    # Calculate distances from the origin to all vertices
    distances = jnp.linalg.norm(vertices - origin, axis=1)
    # Filter vertices within the search radius
    nearby_points = vertices[distances < search_radius]
    assert nearby_points.shape[0] > 0, "No nearby points found"
    print("nearby_points.shape[0] = ", nearby_points.shape[0])
    # Normalize the normal vector
    normal = normal / jnp.linalg.norm(normal)
    # Project the nearby points onto the plane defined by the normal
    x = nearby_points - origin
    # Compute the projection of x onto the plane
    projections = x - jnp.outer(jnp.dot(x, normal), normal)
    projection_lengths = jnp.linalg.norm(projections, axis=1)
    # # Sort the projection lengths
    # sorted_lengths = jx.jit(jnp.sort)(projection_lengths)
    # # Remove the bottom and top 10%
    # num_points = len(sorted_lengths)
    # lower_bound_index = int(num_points * 0.1)
    # upper_bound_index = int(num_points * 0.9)
    # filtered_lengths = sorted_lengths[lower_bound_index:upper_bound_index]
    # Estimate the radius as the average of the filtered lengths
    filtered_lengths = projection_lengths
    # estimated_radius = jnp.mean(filtered_lengths)
    estimated_radius = jnp.min(filtered_lengths)

    return estimated_radius

def estimate_radius_other_ideas(vertices, origin, normal, search_radius):
    # Calculate distances from the origin to all vertices
    distances = jnp.linalg.norm(vertices - origin, axis=1)
    # Filter vertices within the search radius
    nearby_points = vertices[distances < search_radius]
    assert nearby_points.shape[0] > 0, "No nearby points found"
    # Normalize the normal vector
    normal = normal / jnp.linalg.norm(normal)
    # Project the nearby points onto the plane defined by the normal
    x = nearby_points - origin
    # Compute the projection of x onto the plane
    projections = x - jnp.outer(jnp.dot(x, normal), normal)
    projection_lengths = jnp.linalg.norm(projections, axis=1)
    # # Remove outliers using the IQR method
    # Q1 = jnp.percentile(projection_lengths, 25)
    # Q3 = jnp.percentile(projection_lengths, 75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # filtered_lengths = projection_lengths[(projection_lengths >= lower_bound) & (projection_lengths <= upper_bound)]
    # Estimate the radius as the average of the filtered lengths
    # estimated_radius = jnp.mean(filtered_lengths)
    # Create a histogram of projection lengths with bin width 0.05
    # bin_width = 0.05
    # bins = jnp.arange(0, jnp.max(projection_lengths) + bin_width, bin_width)
    # histogram, bin_edges = jnp.histogram(projection_lengths, bins=bins)
    # # Find the most frequent bin
    # most_frequent_bin_index = jnp.argmax(histogram)
    # estimated_radius = (bin_edges[most_frequent_bin_index] + bin_edges[most_frequent_bin_index + 1]) / 2
    # # Plot the histogram
    # plt.hist(projection_lengths, bins=bins)
    # plt.xlabel('Projection Lengths')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Projection Lengths')
    # plt.show()
    return estimated_radius

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
    start_time = time.time()
    triangulated_slice = get_triangulated_slice(surface_polydata, origin, normal)
    mid_time = time.time()
    area = get_cross_sectional_area_of_triangulated_slice(triangulated_slice)
    end_time = time.time()
    print(f"get_triangulated_slice took {mid_time - start_time:.6f} seconds")
    print(f"get_cross_sectional_area_of_triangulated_slice took {end_time - mid_time:.6f} seconds")
    return area

def get_cross_sectional_area_no_timer(surface_polydata, origin, normal):
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

def get_centerline_tangents(centerline_polydata):
    num_centerline_points = centerline_polydata.GetNumberOfPoints()
    tangents = jnp.zeros((num_centerline_points, 3))
    for point_id in range(1, num_centerline_points - 1):
        tangents = tangents.at[point_id].set(jnp.array(centerline_polydata.GetPoint(point_id + 1)) - jnp.array(centerline_polydata.GetPoint(point_id - 1)))
        tangents = tangents.at[point_id].set(tangents[point_id] / jnp.linalg.norm(tangents[point_id]))
    tangents = tangents.at[0].set(jnp.array(centerline_polydata.GetPoint(1)) - jnp.array(centerline_polydata.GetPoint(0)))
    tangents = tangents.at[0].set(tangents[0] / jnp.linalg.norm(tangents[0]))
    tangents = tangents.at[-1].set(jnp.array(centerline_polydata.GetPoint(-1))
     - jnp.array(centerline_polydata.GetPoint(-2)))
    tangents = tangents.at[-1].set(tangents[-1] / jnp.linalg.norm(tangents[-1]))
    return tangents

def get_centerline_tangents_np(centerline_polydata):
    num_centerline_points = centerline_polydata.GetNumberOfPoints()
    tangents = np.zeros((num_centerline_points, 3))
    for point_id in range(1, num_centerline_points - 1):
        tangents[point_id] = np.array(centerline_polydata.GetPoint(point_id + 1)) - np.array(centerline_polydata.GetPoint(point_id - 1))
        tangents[point_id] /= np.linalg.norm(tangents[point_id])
    tangents[0] = np.array(centerline_polydata.GetPoint(1)) - np.array(centerline_polydata.GetPoint(0))
    tangents[0] /= np.linalg.norm(tangents[0])
    tangents[-1] = np.array(centerline_polydata.GetPoint(num_centerline_points - 1)) - np.array(centerline_polydata.GetPoint(num_centerline_points - 2))
    tangents[-1] /= np.linalg.norm(tangents[-1])
    return tangents

def get_centerline_cross_section_areas_np(centerline_polydata):
    areas = centerline_polydata.GetPointData().GetArray("CenterlineSectionArea")
    num_centerline_points = centerline_polydata.GetNumberOfPoints()
    centerline_areas = np.zeros(num_centerline_points)
    if areas is None:
        return centerline_areas
    for point_id in range(num_centerline_points):
        centerline_areas[point_id] = areas.GetValue(point_id)
    return centerline_areas

# @jx.jit
def get_normal_at_centerline_point(centerline_jnp_array, point_id):
    num_centerline_points = centerline_jnp_array.shape[0]
    if 0 < point_id and point_id < num_centerline_points - 1:
        normal = centerline_jnp_array[point_id + 1] - centerline_jnp_array[point_id - 1]
    elif 0 < point_id:
        assert(point_id == num_centerline_points - 1)
        normal = centerline_jnp_array[point_id] - centerline_jnp_array[point_id - 1]
    else:
        assert(point_id == 0)
        normal = centerline_jnp_array[point_id + 1] - centerline_jnp_array[point_id]
    normal /= np.linalg.norm(normal)
    return normal

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

def get_normal_at_point_on_centerline(centerline_polydata, point_id):
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
    return normal

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