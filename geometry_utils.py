import vtk
import math
from collections import defaultdict

# Global variable for input VTP file path
FILE_PATH1 = "/home/bohanjeffli/mesh-complete-exterior.vtp"
  # <-- Change this to your VTP file path
FILE_PATH2 = "/home/bohanjeffli/march-24-SI-Test-Two.vtp"
FILE_PATH = "/home/bohanjeffli/march-11-SI-Test-One.vtp"

# ===============================
# Module 1: Data Loading and Normals Computation
# ===============================
def load_polydata_with_normals(file_path):
    """
    Load a vtkPolyData from a VTP file and compute consistent,
    outward–facing cell (face) normals.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    
    # Compute cell (face) normals (only cell normals are computed)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    
    return normals.GetOutput()

# ===============================
# Module 2: Retrieve Cell Normals
# ===============================
def get_cell_normal(polydata, cell_id):
    """
    Retrieve the computed cell normal (as a list of three values) from the polydata.
    """
    normals = polydata.GetCellData().GetArray("Normals")
    if normals:
        return list(normals.GetTuple(cell_id))
    return None

# ===============================
# Module 3: Build Oriented Edge List
# ===============================
def build_oriented_edge_list(polydata):
    """
    Build an edge list from the polydata using an oriented-edge strategy.
    
    For each cell (assumed triangular), for each edge from vertex a to b
    with the next vertex c, decide whether (a,b) or (b,a) better aligns
    with the face’s orientation.
    
    The canonical key for the edge is the sorted pair (min(a,b), max(a,b)).
    For each adjacent cell, we also store an orientation flag:
      +1 if the cell’s ordering agrees with the canonical order,
      -1 if reversed.
    
    Returns a dictionary mapping canonical edges to a list of tuples:
      { (a, b): [ (cell_id, flag), ... ] }
    """
    polydata.BuildLinks()  # ensure connectivity is built
    edge_dict = defaultdict(list)
    num_cells = polydata.GetNumberOfCells()
    
    for cell_id in range(num_cells):
        cell = polydata.GetCell(cell_id)
        pt_ids = cell.GetPointIds()
        n_pts = pt_ids.GetNumberOfIds()
        if n_pts < 3:
            continue  # skip degenerate cells
        # For a triangle, we assume vertices are in some order.
        # For each edge, we also use the "next" vertex (c) to decide ordering.
        for i in range(n_pts):
            a = pt_ids.GetId(i)
            b = pt_ids.GetId((i+1) % n_pts)
            c = pt_ids.GetId((i+2) % n_pts)  # next vertex in the cycle
            
            # Retrieve point coordinates
            p_a = polydata.GetPoint(a)
            p_b = polydata.GetPoint(b)
            p_c = polydata.GetPoint(c)
            # Compute vectors: ab and bc
            ab = [p_b[j] - p_a[j] for j in range(3)]
            bc = [p_c[j] - p_b[j] for j in range(3)]
            # Get the cell's outward normal
            n = get_cell_normal(polydata, cell_id)
            if n is None:
                n = [0, 0, 1]
            # Option 1: Use ordering (a, b)
            # if a == 5830 and b == 6362:
            #     print("once.")
            # if a == 6362 and b == 5830:
            #     print("twice.")
            # Compute cross product: (ab) x (bc)
            cross1 = [ab[1]*bc[2] - ab[2]*bc[1],
                      ab[2]*bc[0] - ab[0]*bc[2],
                      ab[0]*bc[1] - ab[1]*bc[0]]
            dot1 = sum(cross1[j] * n[j] for j in range(3))
            # if a == 6362 and b == 5830:
            #     print("cross1 = ", cross1, "n = ", n, "dot1 = ", dot1)
            # Choose the ordering that gives a positive dot product.
            assert(dot1 >= 0)
            if dot1 >= 0:
                oriented_edge = (a, b)
                flag = 1
            else:
                oriented_edge = (b, a)
                print("this shouldn't happen.")
                flag = -1
            
            # Use the canonical key (sorted order) for uniqueness.
            key = tuple(sorted((a, b)))
            if (a > b):
                flag *= -1
            edge_dict[key].append((cell_id, flag))
    
    return edge_dict

# ===============================
# Module 4b: Compute Signed Face Bending Angle from Oriented Edge Info
# ===============================
def compute_signed_face_angle_from_edge(polydata, edge_key, cell_info_list):
    """
    Given a canonical edge key (a tuple of two point IDs) and its associated list
    of adjacent cell info [(cell_id, flag), ...] (expected length 2 for interior edges),
    compute the signed face bending angle.
    
    We use the canonical edge (from sorted order) as the reference.
    For each cell, we have an orientation flag indicating whether the cell’s
    local ordering of the edge agrees with the canonical order.
    
    Then, we compute:
         theta = acos( n1 · n2 )
         face_angle = sign * (180 - theta)
         
    where the sign is determined using the triple product of (n1 x n2) with the
    canonical edge vector.
    """
    if len(cell_info_list) != 2:
        return None  # boundary or non-manifold edge
    # Retrieve cell normals for both cells
    cell1, flag1 = cell_info_list[0]
    cell2, flag2 = cell_info_list[1]
    n1 = get_cell_normal(polydata, cell1) 
    n2 = get_cell_normal(polydata, cell2) 
    if n1 is None or n2 is None:
        return None

    # Get canonical edge vector from the sorted key
    p0 = polydata.GetPoint(edge_key[0]) if flag1 == +1 else polydata.GetPoint(edge_key[1])
    p1 = polydata.GetPoint(edge_key[1]) if flag1 == +1 else polydata.GetPoint(edge_key[0])

    edge_vector = [p1[i] - p0[i] for i in range(3)]
    edge_norm = math.sqrt(sum(c*c for c in edge_vector))
    if edge_norm == 0:
        return None
    edge_vector = [c/edge_norm for c in edge_vector]
    
    dot_val = sum(n1[i] * n2[i] for i in range(3))
    n1_norm = math.sqrt(sum(n1[i]*n1[i] for i in range(3)))
    n2_norm = math.sqrt(sum(n2[i]*n2[i] for i in range(3)))
    dot_val /= n1_norm * n2_norm
    theta = math.degrees(math.acos(dot_val))
    # Compute cross product of n1 and n2.
    cross = [n1[1]*n2[2] - n1[2]*n2[1],
             n1[2]*n2[0] - n1[0]*n2[2],
             n1[0]*n2[1] - n1[1]*n2[0]]
    # Second method: use the atan2 function to compute the signed angle
    theta = math.atan2(sum(edge_vector[i]*cross[i] for i in range(3)), dot_val)
    inside_angle = math.degrees(theta)
    dihedral_angle = 180 + inside_angle
    # print(f"First method: {first_method_angle:.2f}°, Second method: {second_method_angle:.2f}°")
    return dihedral_angle

# ===============================
# Module 5: Highlight Cells Based on Bending Angle
# ===============================
def highlight_flat_cells(polydata, edge_dict, angle_threshold=15):
    """
    For each interior edge (with exactly two adjacent cells), compute the signed face
    bending angle (using the oriented edge information). If the absolute face angle is
    below the threshold, mark both adjacent cells.
    
    This function returns a vtkUnsignedCharArray for cell colors with 4 components (RGBA):
      - Non-highlighted cells are white with 20% opacity.
      - Highlighted cells are red with 100% opacity.
    """
    num_cells = polydata.GetNumberOfCells()
    cell_colors = vtk.vtkUnsignedCharArray()
    cell_colors.SetNumberOfComponents(4)
    cell_colors.SetName("Colors")
    # Set default: white (255,255,255) with 20% opacity (51 out of 255)
    for _ in range(num_cells):
        cell_colors.InsertNextTuple4(255, 255, 255, 50)
    
    highlight_cells = set()
    for key, cell_info_list in edge_dict.items():
        if len(cell_info_list) == 2:
            angle = compute_signed_face_angle_from_edge(polydata, key, cell_info_list)
            if angle is None:
                continue
            # print(f"Edge {key}: face bending angle = {angle:.2f}°")
            if angle < angle_threshold:
                for (cell_id, _) in cell_info_list:
                    highlight_cells.add(cell_id)
    
    for cell_id in highlight_cells:
        # Set highlighted cells to red (255,0,0) with full opacity (255)
        cell_colors.SetTuple4(cell_id, 255, 0, 0, 255)
    
    return cell_colors

# ===============================
# Module 6: Create Glyph Actor for Normals at Cell Centers
# ===============================
def create_manual_normals_glyph_actor(polydata, scale_factor=0.1):
    """
    Create an actor to visualize cell normals as red arrows.
    Instead of using automatically computed normals from the polydata,
    this function manually queries the cell normals via get_cell_normal.
    Glyphs are placed at the center of each cell (using vtkCellCenters).
    """
    # Compute cell centers.
    centers = vtk.vtkCellCenters()
    centers.SetInputData(polydata)
    centers.Update()
    centers_poly = centers.GetOutput()
    
    # Create an array to hold the manually computed normals.
    manual_normals = vtk.vtkFloatArray()
    manual_normals.SetNumberOfComponents(3)
    manual_normals.SetName("ManualNormals")
    
    num_cells = polydata.GetNumberOfCells()
    for cell_id in range(num_cells):
        n = get_cell_normal(polydata, cell_id)
        if n is None:
            n = [0.0, 0.0, 1.0]
        manual_normals.InsertNextTuple(n)
    
    # Assign the manually computed normals to the cell centers polydata.
    centers_poly.GetPointData().SetVectors(manual_normals)
    
    # Create an arrow glyph source.
    arrowSource = vtk.vtkArrowSource()
    
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrowSource.GetOutputPort())
    glyph.SetInputData(centers_poly)
    # Use the vectors we just assigned.
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(scale_factor)
    glyph.OrientOn()
    glyph.Update()
    
    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph.GetOutputPort())
    
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.GetProperty().SetColor(1, 0, 0)  # red arrows
    
    return glyph_actor

def create_normals_glyph_actor(polydata, scale_factor=0.1):
    """
    Create an actor to visualize cell normals as red arrows.
    Glyphs are placed at the center of each cell (using vtkCellCenters).
    """
    centers = vtk.vtkCellCenters()
    centers.SetInputData(polydata)
    centers.Update()
    
    arrowSource = vtk.vtkArrowSource()
    
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrowSource.GetOutputPort())
    glyph.SetInputConnection(centers.GetOutputPort())
    glyph.SetVectorModeToUseNormal()   # Use cell normals stored in cell data
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(scale_factor)
    glyph.OrientOn()
    glyph.Update()
    
    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph.GetOutputPort())
    
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.GetProperty().SetColor(1, 0, 0)  # red arrows
    return glyph_actor

# ===============================
# Module 7: Visualization Setup
# ===============================
def setup_renderers(polydata, cell_colors, glyph_actor):
    """
    Setup and return a renderer, render window, and interactor.
    The scene displays:
      - The original mesh with cell colors (flat edges flagged cells are red).
      - The normals glyph actor.
    """
    polydata.GetCellData().SetScalars(cell_colors)
    
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(polydata)
    mesh_mapper.SetScalarModeToUseCellData()
    mesh_mapper.SetColorModeToDirectScalars()
    # mesh_mapper.SetInterpolateScalarsBeforeMapping(False)
    
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().EdgeVisibilityOn()
    mesh_actor.GetProperty().SetEdgeColor(0, 0, 0)
    # mesh_actor.GetProperty().SetOpacity(1)
    
    renderer = vtk.vtkRenderer()
    renderer.AddActor(mesh_actor)
    # renderer.AddActor(glyph_actor)
    renderer.SetBackground(1, 1, 1)
    
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1600, 1200)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(renderWindow)
    
    return renderWindow, interactor

# ===============================
# Module 8: Main Demo Function
# ===============================
def main():
    # 1. Load the polydata from file and compute consistent cell normals.
    polydata = load_polydata_with_normals(FILE_PATH)
    
    # 2. Build the oriented edge list (with per-cell orientation flags).
    edge_dict = build_oriented_edge_list(polydata)
    
    # 3. Highlight cells sharing an edge with a face bending angle (180 - angle between normals)
    #    below the threshold (here, < 15° absolute).
    cell_colors = highlight_flat_cells(polydata, edge_dict, angle_threshold=90)
    
    # 4. Create a glyph actor to display the cell normals as red arrows at cell centers.
    # glyph_actor = create_normals_glyph_actor(polydata, scale_factor=0.1)
    glyph_actor = create_manual_normals_glyph_actor(polydata, scale_factor=0.1)
    
    # 5. Setup visualization.
    renderWindow, interactor = setup_renderers(polydata, cell_colors, glyph_actor)
    
    renderWindow.Render()
    interactor.Initialize()
    interactor.Start()

if __name__ == "__main__":
    main()