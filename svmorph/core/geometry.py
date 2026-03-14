import numpy as np
import jax.numpy as jnp


def resample_stent_axis(points, parent_tip_map, segment_base_mask, starting_point_idx, desired_total_length, desired_segment_length, sampling_direction=-1):
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
