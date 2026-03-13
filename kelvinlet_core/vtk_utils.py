import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from collections import defaultdict
import time
import jax.numpy as jnp


def polydata_to_np_jnp_data(surface_polydata, centerline_polydata):
    # Convert to JAX-compatible arrays by using jnp.array
    surface_points_view_np = v2n(surface_polydata.GetPoints().GetData())
    centerline_points_view_np = v2n(centerline_polydata.GetPoints().GetData())
    surface_points_jnp = jnp.array(surface_points_view_np)
    centerline_points_jnp = jnp.array(centerline_points_view_np)
    # Create a dictionary to store data, including the JAX arrays
    data = {
        "points": {
            "surface_points_view_np": surface_points_view_np,
            "centerline_points_view_np": centerline_points_view_np,
            "surface": surface_points_view_np,
            "centerline": centerline_points_view_np
        },
        "nodes": {
            "all_indices": [],
            "force_center_point_id": -1
        },
        "centerline_coordinate": jnp.array([])
    }
    
    return data

def polydata_to_parent_tip_map(centerline_polydata):
    """
    Build a mapping  pointId -> maxPointId_of_closest_parent_segment
    for a VTK centre‑line tree whose segments are encoded by a
    {0,1}-flag array (one component per leaf branch).
    Returns dict { pointId (int) : parent_tip_pointId (int) }.
    """
    vtk_arr = centerline_polydata.GetPointData().GetArray("CenterlineId")
    if vtk_arr is None:
        raise ValueError("Point array 'CenterlineId' not found.")
    flags = v2n(vtk_arr)            # (N, n_components)
    print("flags.shape = ", flags.shape)
    # n_pts, n_comp = flags.shape
    # Group points into segments
    #  -- unique_rows  :   (n_segments, n_components)
    #  -- inverse      :   length N vector   point i --> segment_id
    unique_rows, inverse = np.unique(flags, axis=0, return_inverse=True)
    if unique_rows.shape[0] == 1:
        return {pointId: -1 for pointId in range(flags.shape[0])}, np.zeros(flags.shape[0], dtype=bool)
    # n_segments = unique_rows.shape[0]
    segment_points = defaultdict(list)          # seg_id -> [pt_id, ...]
    for pointId, seg_id in enumerate(inverse):
        segment_points[seg_id].append(pointId)
    #  Pre‑compute the "tip" (largest point id) of every segment.
    seg_tip = {seg_id: max(pts) for seg_id, pts in segment_points.items()}
    segment_base_mask = np.zeros(flags.shape[0], dtype=bool)
    for seg_id, pts in segment_points.items():
        segment_base_mask[min(pts)] = True
    #  Pre‑compute bit counts (how many 1's) to choose closest parent.
    seg_bitcount = unique_rows.sum(axis=1)      # (n_segments,)
    # 3)  For every segment, find its closest ancestor
    #     (superset with minimal extra 1‑bits)
    parent_tip_for_segment = {}   # seg_id -> parent_tip_point_id
    for child_id, child_mask in enumerate(unique_rows):
        # Vectorised superset test:
        # parent is superset  <=>   all 1‑bits in child also 1 in parent
        mask_ok = np.logical_or(~child_mask.astype(bool), unique_rows.astype(bool))
        is_superset = mask_ok.all(axis=1)
        # Exclude itself, keep only strictly larger (superset) bit masks
        is_superset[child_id] = False
        # If no ancestor exists (root), map to its own tip.
        if not np.any(is_superset):
            parent_tip_for_segment[child_id] = seg_tip[child_id]
            continue
        # Among supersets pick the one with the fewest 1‑bits
        candidate_ids = np.nonzero(is_superset)[0]
        extra_bits = seg_bitcount[candidate_ids] - seg_bitcount[child_id]
        best_parent_idx = candidate_ids[np.argmin(extra_bits)]
        parent_tip_for_segment[child_id] = seg_tip[best_parent_idx]

    # 4)  Build the final point‑level dictionary
    point_to_parent_tip = {}
    for pt_id, seg_id in enumerate(inverse):
        point_to_parent_tip[pt_id] = parent_tip_for_segment[seg_id]
    return point_to_parent_tip, segment_base_mask

def sample_stent_axis_vertices_new(points, parent_tip_map, segment_base_mask, starting_point_idx, desired_total_length, desired_segment_length, sampling_direction=-1):
    """
    Extracts and resamples a subsegment of a polyline based strictly on original cumulative arclength.
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
        print(f"Subsegment truncated due to jump/end. Best achieved length = {effective_total_length:.4f} cm")
    
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

def cut_polydata(polydata, origin, normal):
    cutting_plane = vtk.vtkPlane()
    cutting_plane.SetOrigin(origin[0], origin[1], origin[2])
    cutting_plane.SetNormal(normal[0], normal[1], normal[2])
    cut = vtk.vtkCutter()
    cut.SetInputData(polydata)
    cut.SetCutFunction(cutting_plane)
    cut.Update()
    return cut.GetOutput()

def connectivity(polydata, origin):
    connection = vtk.vtkConnectivityFilter()
    connection.SetInputData(polydata)
    connection.SetExtractionModeToClosestPointRegion()
    connection.SetClosestPoint(origin[0], origin[1], origin[2])
    connection.Update()
    return connection
    
def slice_polydata(surface_polydata, origin, normal):
    cut = cut_polydata(surface_polydata, origin, normal)
    contour = connectivity(cut, origin) # get the contour closest to the origin, since the cut_polydata() might result in multiple contours extracted
    return contour.GetOutput()

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
    res = v2n(areas) if areas is not None else np.zeros(centerline_polydata.GetNumberOfPoints())
    return res

def get_maximum_inscribed_sphere_radius_np(centerline_polydata):
    radii = centerline_polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    res = v2n(radii) if radii is not None else np.zeros(centerline_polydata.GetNumberOfPoints())
    return res

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
