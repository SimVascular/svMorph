import numpy as np
import jax.numpy as jnp
import vtk

def resample_subsegment(points, desired_segment_length, starting_point_idx, desired_total_length, jump_threshold=1.0):
    """
    Extracts and resamples a subsegment of a polyline.
    
    Parameters:
      points (np.array): Nx3 array of 3D coordinates representing the polyline.
      desired_segment_length (float): The spacing (in cm) between resampled points.
      starting_point_idx (int): Index in points where the subsegment starts.
      desired_total_length (float): The desired total arc length (in cm) for the subsegment.
      jump_factor (float): Heuristic factor for detecting a jump (default: 3.0).
    
    Returns:
      jax.numpy.array: A new array of 3D coordinates representing the resampled subsegment.
    
    The function iterates from the starting index, accumulating arc length. If the next segment
    is longer than (jump_factor * median_distance) [computed from the entire polyline], a jump is assumed.
    If a jump is encountered before reaching the desired_total_length, the subsegment is terminated
    just before the jump and the achieved length is printed.
    """
    # Compute the global median distance (heuristic for jump detection)
    if len(points) < 2:
        raise ValueError("Not enough points to form a polyline.")
    diffs_all = np.diff(points, axis=0)
    distances_all = np.linalg.norm(diffs_all, axis=1)
    
    subsegment_points = []
    # Start with the given starting point.
    subsegment_points.append(points[starting_point_idx])
    cumulative_length = 0.0
    n_points = len(points)
    
    # Walk along the polyline starting from starting_point_idx.
    for i in range(starting_point_idx, n_points - 1):
        # Compute the Euclidean distance to the next point.
        d = distances_all[i]
        # Check if this segment is a jump.
        if d > jump_threshold:
            # Jump detected; break out without including the jump segment.
            print(f"Jump detected at segment {i} -> {i+1} (distance {d:.4f} cm).")
            break
        # If adding the full segment would exceed desired_total_length,
        # interpolate along this segment to hit the target exactly.
        if cumulative_length + d < desired_total_length:
            cumulative_length += d
            subsegment_points.append(points[i+1])
        else:
            remaining = desired_total_length - cumulative_length
            t = remaining / d  # interpolation fraction
            new_point = (1 - t) * points[i] + t * points[i+1]
            subsegment_points.append(new_point)
            cumulative_length += remaining
            break  # desired total length achieved
    
    effective_total_length = cumulative_length
    if effective_total_length < desired_total_length:
        print(f"Subsegment truncated due to jump. Best achieved length = {effective_total_length:.4f} cm")
    
    # Convert the subsegment to a numpy array.
    subsegment_points = np.array(subsegment_points)
    
    # Now resample the subsegment to have points uniformly spaced by desired_segment_length.
    # Compute cumulative arc-length for the subsegment.
    diffs = np.diff(subsegment_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumu_length = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = cumu_length[-1]
    
    # Generate new arc-length values from 0 to total_length, with spacing desired_segment_length.
    new_s = np.arange(0, total_length, desired_segment_length)
    new_s = np.append(new_s, total_length)
    
    new_points = []
    j = 0  # current segment index in the subsegment
    for s in new_s:
        # Find the segment that contains arc-length s.
        while j < len(cumu_length) - 2 and cumu_length[j+1] < s:
            j += 1
        seg_delta = cumu_length[j+1] - cumu_length[j]
        t = 0 if seg_delta == 0 else (s - cumu_length[j]) / seg_delta
        interpolated_pt = (1 - t) * subsegment_points[j] + t * subsegment_points[j+1]
        new_points.append(interpolated_pt)
    
    # Convert the new resampled points to a jax.numpy array and return.
    return jnp.array(new_points)

# Example usage:
if __name__ == '__main__':
    # Example input: a simple polyline in 3D.
    # points = np.array([
    #     [0.0, 0.0, 0.0],
    #     [0.011, 0.0, 0.0],
    #     [0.023, 0.0, 0.0],
    #     [0.038, 0.02, 0.0],
    #     # A jump: next point is far away, representing a branch break.
    #     [1.70, 0.0, 0.0],
    #     [1.71, 0.0, 0.0],
    #     [1.72, 0.0, 0.0]
    # ])
    # Real file example
    # File names for input and output.
    input_filename = "/home/bohanjeffli/Full_Centerlines.vtp"
    output_filename = "/home/bohanjeffli/Stent_of_Full_Centerlines_resampled.vtp"
    # Read the input VTP file.
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_filename)
    reader.Update()
    polydata = reader.GetOutput()
    # Extract the points from the polydata.
    vtk_points = polydata.GetPoints()
    num_points = vtk_points.GetNumberOfPoints()
    points = np.array([vtk_points.GetPoint(i) for i in range(num_points)])
    

    desired_segment_length = 0.1  # in cm
    starting_point_idx = 5514 # around 5514 to 5575
    desired_total_length = 1.5  # in cm

    resampled_subsegment = resample_subsegment(points, desired_segment_length, starting_point_idx, desired_total_length)
    
    # Print the resulting resampled subsegment (as a jax.numpy array)
    print("Resampled subsegment points:")
    print(resampled_subsegment)