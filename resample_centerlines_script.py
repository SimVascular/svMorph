import vtk
import numpy as np

def resample_polyline(points, segment_length):
    """
    Resamples a given polyline (Nx3 numpy array) so that each segment
    along the branch is approximately of length 'segment_length'.
    Uses arc-length parameterization and linear interpolation.
    """
    # Compute differences between consecutive points and their Euclidean lengths.
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    
    # Compute cumulative arc-length along the branch.
    cumulative_length = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = cumulative_length[-1]
    
    # Create new uniformly spaced arc-length values.
    new_s = np.arange(0, total_length, segment_length)
    if new_s[-1] != total_length:
        new_s = np.append(new_s, total_length)
    
    new_points = []
    j = 0  # current segment index in the original branch
    for s in new_s:
        # Find the segment such that cumulative_length[j] <= s <= cumulative_length[j+1]
        while j < len(cumulative_length) - 2 and s > cumulative_length[j+1]:
            j += 1
        # Compute the interpolation factor within the current segment.
        segment_delta = cumulative_length[j+1] - cumulative_length[j]
        t = 0 if segment_delta == 0 else (s - cumulative_length[j]) / segment_delta
        # Linear interpolation between the two endpoints.
        new_pt = (1 - t) * points[j] + t * points[j+1]
        new_points.append(new_pt)
    return np.array(new_points)

def split_into_branches(points, jump_threshold=1.0):
    """
    Heuristically splits a 1D polyline (given as a numpy array of points)
    into branches based on jumps in the distances between consecutive points.
    We compute the median distance of adjacent points and consider any gap 
    larger than (jump_factor * median) as a branch jump.
    """
    # Compute the distances between consecutive points.
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    # # Use the median distance as a baseline.
    # median_distance = np.median(distances)
    segments = []
    start_idx = 0
    for i, d in enumerate(distances):
        if d > jump_threshold:
            print(f"start idx = {start_idx}. Detected jump at index {i} (distance = {d:.4f} cm)")
            # We assume a jump: current branch goes from start_idx to i (inclusive).
            segments.append(points[start_idx:i+1])
            start_idx = i+1
    # Append any remaining points as the final branch.
    if start_idx < len(points):
        segments.append(points[start_idx:])
    return segments

def main():
    # File names for input and output.
    input_filename = "/home/bohanjeffli/Full_Centerlines.vtp"
    output_filename = "/home/bohanjeffli/Full_Centerlines_001_resampled.vtp"
    # Desired resampling segment length (cm).
    desired_segment_length = 0.01

    # Read the input VTP file.
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_filename)
    reader.Update()
    polydata = reader.GetOutput()

    # Extract the points from the polydata.
    vtk_points = polydata.GetPoints()
    num_points = vtk_points.GetNumberOfPoints()
    points = np.array([vtk_points.GetPoint(i) for i in range(num_points)])

    # Split the full list of points into branches using our heuristic.
    branches = split_into_branches(points, jump_threshold=1.0)
    print(f"Detected {len(branches)} branch(es) in the input centerline.")

    # Prepare a cell array for the new polydata.
    cells = vtk.vtkCellArray()
    new_vtk_points = vtk.vtkPoints()
    point_offset = 0  # to track indices across branches

    # Process each branch separately.
    for idx, branch in enumerate(branches):
        # Compute and print the total length of the branch.
        if branch.shape[0] < 2:
            # print("branch shape: ", branch.shape)
            branch_length = 0.0
        else:
            diffs = np.diff(branch, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            branch_length = np.sum(seg_lengths)
        print(f"Branch {idx+1}: total length = {branch_length:.4f} cm")

        # Resample the branch using the arc-length parameterization.
        if branch.shape[0] >= 2:
            resampled_branch = resample_polyline(branch, desired_segment_length)
        else:
            resampled_branch = branch  # single point branch; nothing to resample

        # Add the resampled points to the new vtkPoints array.
        num_branch_points = resampled_branch.shape[0]
        branch_point_ids = vtk.vtkIdList()
        for pt in resampled_branch:
            new_vtk_points.InsertNextPoint(pt)
            branch_point_ids.InsertNextId(point_offset)
            point_offset += 1

        # Create a polyline cell for this branch.
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(num_branch_points)
        for i in range(num_branch_points):
            line.GetPointIds().SetId(i, branch_point_ids.GetId(i))
        cells.InsertNextCell(line)

    # Build the new polydata with resampled branches.
    new_polydata = vtk.vtkPolyData()
    new_polydata.SetPoints(new_vtk_points)
    new_polydata.SetLines(cells)

    # Write the new polydata to a VTP file.
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(new_polydata)
    writer.Write()

if __name__ == '__main__':
    main()