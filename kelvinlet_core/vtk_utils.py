import os
# import sys
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v
from collections import defaultdict
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
    #  Pre‑compute the “tip” (largest point id) of every segment.
    seg_tip = {seg_id: max(pts) for seg_id, pts in segment_points.items()}
    segment_base_mask = np.zeros(flags.shape[0], dtype=bool)
    for seg_id, pts in segment_points.items():
        segment_base_mask[min(pts)] = True
    #  Pre‑compute bit counts (how many 1’s) to choose closest parent.
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
    # print("unit test: parent id for 662 = ", point_to_parent_tip[662])
    # print("unit test: parent id for 7745 = ", point_to_parent_tip[7745])
    return point_to_parent_tip, segment_base_mask

def sample_stent_axis_vertices_old(points, parent_tip_map, segment_base_mask, starting_point_idx, desired_total_length, desired_segment_length, sampling_direction=-1):
    """
    Extracts and resamples a subsegment of a polyline.
    
    Parameters:
      points (np.array): Nx3 array of 3D coordinates representing the polyline.
      desired_segment_length (float): The spacing (in cm) between resampled points.
      starting_point_idx (int): Index in points where the subsegment starts.
      desired_total_length (float): The desired total arc length (in cm) for the subsegment.
      jump_threshold (float): Heuristic length for detecting a jump (default: distance more than 1.0 cm is considered a jump).
    
    Returns:
      jax.numpy.array: A new array of 3D coordinates representing the resampled subsegment.
    
    The function iterates from the starting index, accumulating arc length. If the next segment
    is longer than jump_threshold, a jump is assumed.
    If a jump is encountered before reaching the desired_total_length, the subsegment is terminated
    just before the jump and the achieved length is printed.
    The sampling direction can be either -1 (backward) or +1 (forward), this ensures we always sample from a branch towards a trunk so that stent location can be unambiguously identified with just the starting point index.
    """
    if sampling_direction not in (-1, 1):
        raise ValueError("sampling_direction must be integer -1 or +1")
    if len(points) < 2:
        raise ValueError("Not enough points to form a polyline.")
    
    diffs_all = np.diff(points, axis=0)
    distances_all = np.linalg.norm(diffs_all, axis=1)
    
    subsegment_points = []
    subsegment_points.append(points[starting_point_idx])
    cumulative_length = 0.0
    n_points = len(points)
    
    # Walk along the polyline starting from starting_point_idx.
    i = starting_point_idx
    idx_end = 0 if sampling_direction == -1 else n_points - 1
    next_point_idx_offset = -1 if sampling_direction == -1 else 0
    while i != idx_end:
        # d = distances_all[i+next_point_dist_idx]
        # if d > jump_threshold:
            # Jump detected; break out without including the jump segment.
            # print(f"Jump detected at segment {i} -> {i+sampling_direction} (distance {d:.4f} cm).")
            # break
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
        else:
            remaining = desired_total_length - cumulative_length
            t = remaining / d 
            new_point = (1 - t) * points[i] + t * points[next_i]
            subsegment_points.append(new_point)
            cumulative_length += remaining
            break  # desired total length achieved, exit loop

        i = next_i
    # print(f"subsegment_points = {subsegment_points}")
    effective_total_length = cumulative_length
    if effective_total_length < desired_total_length:
        print(f"Subsegment truncated due to jump. Best achieved length = {effective_total_length:.4f} cm")
    
    # Convert the subsegment to a numpy array.
    subsegment_points = np.array(subsegment_points)
    # Now resample the subsegment to have points uniformly spaced by desired_segment_length.
    # Compute cumulative arc-length for the subsegment.
    diffs = np.diff(subsegment_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumu_length = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = cumu_length[-1]
    # Generate new arc-length values from 0 to total_length, with spacing desired_segment_length.
    new_s = np.arange(0, total_length, desired_segment_length)
    new_s = np.append(new_s, total_length)
    new_vertices = []
    j = 0  # current segment index in the subsegment
    for s in new_s:
        # Find the segment that contains arc-length s.
        while j < len(cumu_length) - 2 and cumu_length[j+1] < s:
            j += 1
        seg_delta = cumu_length[j+1] - cumu_length[j]
        t = 0 if seg_delta == 0 else (s - cumu_length[j]) / seg_delta
        interpolated_vertex = (1 - t) * subsegment_points[j] + t * subsegment_points[j+1]
        new_vertices.append(interpolated_vertex)
    
    return jnp.array(new_vertices)

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

def sample_stent_axis_vertices(points, parent_tip_map, segment_base_mask, starting_point_idx, desired_total_length, desired_segment_length, stent_diameter, sampling_direction=-1):
    """
    Extract + resample a polyline subsegment whose *effective cylinder length*
    reaches `desired_total_length`, accounting for bend-induced extra length:
        deff = d + r * theta
    where d is segment length, r = stent_diameter / 2, and theta is the turning
    angle between consecutive segment tangents.  Parent-tip 'jumps' are treated
    as contiguous geometry, i.e. theta is computed across them as well.
    """
    if sampling_direction not in (-1, 1):
        raise ValueError("sampling_direction must be integer -1 or +1")
    if len(points) < 2:
        raise ValueError("Not enough points to form a polyline.")

    r = 0.5 * float(stent_diameter)
    pts = np.asarray(points, dtype=float)
    n_points = len(pts)

    def seg_len(a, b):
        return float(np.linalg.norm(b - a))

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    
    def turn_angle(t_prev, t_next):
        if np.allclose(t_prev, 0) or np.allclose(t_next, 0):
            return 0.0 # assume it is straight if one of the tangents is zero
        return float(np.arctan2(np.linalg.norm(np.cross(t_prev, t_next)), np.dot(t_prev, t_next)))

    sub_pts = [pts[starting_point_idx]]
    L_eff = 0.0  # accumulated effective (cylinder) length

    i = starting_point_idx
    idx_end = 0 if sampling_direction == -1 else n_points - 1

    have_prev_tan = False
    prev_tan = None

    while i != idx_end:
        # step to next index (handle parent-tip connectivity)
        if segment_base_mask[i]:
            next_i = parent_tip_map[i]
        else:
            next_i = i + sampling_direction

        cur = pts[i]
        nxt = pts[next_i]
        d = seg_len(cur, nxt)
        if d == 0.0:
            i = next_i
            continue

        cur_tan = unit(nxt - cur)
        theta = turn_angle(prev_tan, cur_tan) if have_prev_tan else 0.0

        # effective length increment for this step
        vertex_penalty = r * theta
        deff = vertex_penalty + d

        if L_eff + deff < desired_total_length:
            # take full step
            L_eff += deff
            sub_pts.append(nxt)
            i = next_i
            have_prev_tan = True
            prev_tan = cur_tan
        else:
            # finish inside this step
            remaining = desired_total_length - L_eff
            if remaining <= vertex_penalty + 1e-12:
                # stop exactly at current vertex
                break
            leftover_linear = remaining - vertex_penalty
            s = np.clip(leftover_linear / d, 0.0, 1.0)
            new_pt = (1.0 - s) * cur + s * nxt
            sub_pts.append(new_pt)
            L_eff = desired_total_length
            break

    subsegment_points = np.array(sub_pts)

    # resample uniformly by arclength along extracted subsegment
    if len(subsegment_points) == 1:
        return jnp.array(subsegment_points)

    diffs = np.diff(subsegment_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumu = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total = cumu[-1]

    if total == 0.0:
        return jnp.array(subsegment_points[:1])

    new_s = np.arange(0.0, total, float(desired_segment_length))
    if new_s.size == 0 or new_s[-1] < total:
        new_s = np.append(new_s, total)

    new_vertices = []
    j = 0
    for s in new_s:
        while j < len(cumu) - 2 and cumu[j+1] < s:
            j += 1
        seg_delta = cumu[j+1] - cumu[j]
        t = 0.0 if seg_delta == 0.0 else (s - cumu[j]) / seg_delta
        new_vertices.append((1.0 - t) * subsegment_points[j] + t * subsegment_points[j+1])

    return jnp.array(new_vertices)

def sample_stent_axis_vertices_straight_line(points, parent_tip_map, segment_base_mask, starting_point_idx, desired_total_length, desired_segment_length, sampling_direction=-1):
    """
    Extracts and resamples a subsegment of a polyline based on Euclidean distance from start.
    
    Parameters:
      points (np.array): Nx3 array of 3D coordinates representing the polyline.
      desired_segment_length (float): The spacing (in cm) between resampled points (output resolution).
      starting_point_idx (int): Index in points where the subsegment starts.
      desired_total_length (float): The maximum straight-line distance (in cm) allowed 
                                    between the start point and the end of the subsegment.
    
    Returns:
      jax.numpy.array: A new array of 3D coordinates representing the resampled subsegment.
    """
    if sampling_direction not in (-1, 1):
        raise ValueError("sampling_direction must be integer -1 or +1")
    if len(points) < 2:
        raise ValueError("Not enough points to form a polyline.")
    
    subsegment_points = []
    start_point = points[starting_point_idx]
    subsegment_points.append(start_point)
    
    n_points = len(points)
    
    # Walk along the polyline starting from starting_point_idx.
    i = starting_point_idx
    idx_end = 0 if sampling_direction == -1 else n_points - 1
    next_point_idx_offset = -1 if sampling_direction == -1 else 0
    
    while i != idx_end:
        # Determine the next index based on hierarchy (parent_tip_map) or linear traversal
        if segment_base_mask[i]:
            next_i = parent_tip_map[i]
        else:
            next_i = i + sampling_direction

        # Get the actual coordinates
        p_curr = points[i]
        p_next = points[next_i]

        # Calculate distance from the ORIGINAL start point to the NEXT point
        dist_from_start_to_next = np.linalg.norm(p_next - start_point)

        if dist_from_start_to_next < desired_total_length:
            # The next point is still within the desired radius
            subsegment_points.append(p_next)
        else:
            # The next point is outside the radius. 
            # We need to find t such that ||(p_curr + t * (p_next - p_curr)) - start_point|| = desired_total_length
            # Let U = p_curr - start_point
            # Let V = p_next - p_curr (the segment vector)
            # We solve ||U + tV||^2 = R^2 for t
            
            U = p_curr - start_point
            V = p_next - p_curr
            R = desired_total_length
            
            # Quadratic coefficients: At^2 + Bt + C = 0
            A = np.dot(V, V)
            B = 2 * np.dot(U, V)
            C = np.dot(U, U) - R**2
            
            # Calculate discriminant
            delta = B**2 - 4*A*C
            
            if delta >= 0 and A > 1e-8:
                # We want the positive root that usually falls between 0 and 1 for forward intersection
                t = (-B + np.sqrt(delta)) / (2*A)
                
                # Clamp t to [0, 1] just in case of precision issues, though math should hold
                t = max(0.0, min(1.0, t))
                
                new_point = p_curr + t * V
                subsegment_points.append(new_point)
            else:
                # Fallback if calculation fails (e.g. A=0), just take the current point
                subsegment_points.append(p_curr)
            
            break # Stop immediately after adding the interpolated end point

        i = next_i

    # Convert the subsegment to a numpy array.
    subsegment_points = np.array(subsegment_points)

    # --- Resampling Logic (Remains unchanged) ---
    # Now resample the subsegment to have points uniformly spaced by desired_segment_length.
    
    diffs = np.diff(subsegment_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumu_length = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = cumu_length[-1]
    
    if total_length == 0:
        return jnp.array([start_point])

    # Generate new arc-length values from 0 to total_length
    new_s = np.arange(0, total_length, desired_segment_length)
    # Ensure the final point is included if it's not close to the last step
    if len(new_s) == 0 or (total_length - new_s[-1] > 1e-6):
        new_s = np.append(new_s, total_length)
        
    new_vertices = []
    j = 0  
    for s in new_s:
        # Find the segment that contains arc-length s.
        while j < len(cumu_length) - 2 and cumu_length[j+1] < s:
            j += 1
        seg_delta = cumu_length[j+1] - cumu_length[j]
        t = 0 if seg_delta == 0 else (s - cumu_length[j]) / seg_delta
        interpolated_vertex = (1 - t) * subsegment_points[j] + t * subsegment_points[j+1]
        new_vertices.append(interpolated_vertex)
    
    return jnp.array(new_vertices)

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
    # num_centerline_points = centerline_polydata.GetNumberOfPoints()
    # centerline_areas = np.zeros(num_centerline_points)
    # if areas is None:
    #     return centerline_areas
    # for point_id in range(num_centerline_points):
    #     centerline_areas[point_id] = areas.GetValue(point_id)
    # return centerline_areas
    res = v2n(areas) if areas is not None else np.zeros(centerline_polydata.GetNumberOfPoints())
    return res

def get_maximum_inscribed_sphere_radius_np(centerline_polydata):
    radii = centerline_polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    res = v2n(radii) if radii is not None else np.zeros(centerline_polydata.GetNumberOfPoints())
    return res

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