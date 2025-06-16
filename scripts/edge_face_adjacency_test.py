import vtk
import numpy as np
import scipy.sparse as sp
import time

def build_adjacency_matrix(polyData):
    """
    Build a sparse adjacency matrix (in CSR format) for the mesh.
    Each non-zero entry (i, j) indicates that vertex i is adjacent to vertex j.
    """
    num_points = polyData.GetNumberOfPoints()
    rows = []
    cols = []
    
    polyData.BuildLinks()
    num_cells = polyData.GetNumberOfCells()
    
    for cell_id in range(num_cells):
        cell = polyData.GetCell(cell_id)
        num_ids = cell.GetNumberOfPoints()
        # Get all point IDs for this cell.
        point_ids = [cell.GetPointId(i) for i in range(num_ids)]
        # For each unique pair, add both directions.
        for i in range(len(point_ids)):
            for j in range(i+1, len(point_ids)):
                p1 = point_ids[i]
                p2 = point_ids[j]
                rows.extend([p1, p2])
                cols.extend([p2, p1])
    
    data = np.ones(len(rows), dtype=np.int8)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_points, num_points))
    return A.tocsr()

def k_ring_neighbors_sparse(A, start_vertex, k):
    """
    Compute the k-ring neighborhood using repeated sparse matrix multiplication.
    Returns a set of vertex indices that are within k rings of start_vertex.
    """
    num_points = A.shape[0]
    # Indicator vector for the start vertex (as int8).
    current = np.zeros(num_points, dtype=np.int8)
    current[start_vertex] = 1
    # Set to keep track of all visited vertices.
    visited = current.copy()

    for _ in range(k):
        # Multiply current by the adjacency matrix and threshold the result.
        current = A.dot(current)
        # Convert any nonzero value to 1 (vectorized thresholding).
        current = (current > 0).astype(np.int8)
        # Exclude vertices that have already been visited.
        current = current * (1 - visited)
        # Add the new vertices to the visited set.
        visited = np.maximum(visited, current)
    
    # The nonzero indices in visited represent the k-ring neighborhood.
    return set(np.nonzero(visited)[0])

# Example usage:
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("/home/bohanjeffli/mesh-complete-exterior.vtp")
reader.Update()
polyData = reader.GetOutput()

# Build the sparse adjacency matrix.
A = build_adjacency_matrix(polyData)

# Set the starting vertex and desired ring level.
start_vertex = 1  # Change as needed.
k = 2  # For the 2-ring neighborhood.
start_time = time.time()
neighbors_sparse = k_ring_neighbors_sparse(A, start_vertex, k)
# Print the result.
print(f"Time to compute k ring neighborhood: {time.time() - start_time:.4f} seconds")

print("K-ring neighborhood (sparse) for vertex {} with k={}:\n{}".format(start_vertex, k, neighbors_sparse))