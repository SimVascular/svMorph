import vtk
import math
import time
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

# Global variables for file paths (unused in this minimal example if you switch to manually created polydata)
FILE_PATH1 = "mesh-complete-exterior.vtp"
FILE_PATH = "march-24-SI-Test-Two.vtp"
FILE_PATH = "pymeshfix-output-may-7-minimum-fix-0.001-movement.vtp"
# FILE_PATH = "SU0243-stented-SI-test.vtp"
# FILE_PATH = "march-24-SI-Test-Fine-Mesh.vtp"
# FILE_PATH = "march-21-SI-Test-Two.vtp"
# FILE_PATH = "april-14-SI-perpendicular-typical-scale-test.vtp"

# Global variables for sharing data with callbacks.
global_polydata = None
global_edge_dict = None
global_cell_colors = None
global_renderer = None
global_renderWindow = None
global_glyph_actor = None
global_selected_vertices = set()  # Will store the vertices selected via grow selection

# =============================================================================
# Module 1: Data Loading and Normals Computation
# =============================================================================
def load_polydata_with_normals(file_path):
    """
    Load a vtkPolyData from a VTP file and compute consistent,
    outward–facing cell (face) normals.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()

# =============================================================================
# Module 1a: Create Minimal PolyData Examples
# =============================================================================
def create_two_triangle_polydata():
    """
    Create a minimal vtkPolyData from four vertices forming two triangles sharing an edge.
    """
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)       # v0
    points.InsertNextPoint(1.0, 0.0, 0.0)       # v1
    points.InsertNextPoint(0.5, 0.5, 0.0)       # v2
    points.InsertNextPoint(0.5, 0.5, -0.08)     # v3
    points.InsertNextPoint(2.5, 0.5, 0.08)      # v4

    triangles = vtk.vtkCellArray()
    # Triangle 1: v0, v1, v2
    triangle1 = vtk.vtkTriangle()
    triangle1.GetPointIds().SetId(0, 0)
    triangle1.GetPointIds().SetId(1, 1)
    triangle1.GetPointIds().SetId(2, 2)
    triangles.InsertNextCell(triangle1)
    # Triangle 2: v0, v1, v3
    triangle2 = vtk.vtkTriangle()
    triangle2.GetPointIds().SetId(0, 0)
    triangle2.GetPointIds().SetId(1, 1)
    triangle2.GetPointIds().SetId(2, 3)
    triangles.InsertNextCell(triangle2)
    # Triangle 3: v1, v3, v4
    triangle3 = vtk.vtkTriangle()
    triangle3.GetPointIds().SetId(0, 1)
    triangle3.GetPointIds().SetId(1, 3)
    triangle3.GetPointIds().SetId(2, 4)
    triangles.InsertNextCell(triangle3)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()

def create_simple_manifold_polydata():
    """
    Create a simple manifold vtkPolyData from several triangles.
    """
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)       # v0
    points.InsertNextPoint(1.0, 0.0, 0.0)       # v1
    points.InsertNextPoint(0.5, 0.5, 0.0)       # v2
    points.InsertNextPoint(0.5, 0.5, -0.08)     # v3
    points.InsertNextPoint(0.5, 0.0, 0.5)        # v4
    points.InsertNextPoint(0.5, -0.5, 0.0)       # v5
    points.InsertNextPoint(0.5, 0.0, -0.5)       # v6

    triangles = vtk.vtkCellArray()
    # Triangle 1: v0, v1, v2
    triangle1 = vtk.vtkTriangle()
    triangle1.GetPointIds().SetId(0, 0)
    triangle1.GetPointIds().SetId(1, 1)
    triangle1.GetPointIds().SetId(2, 2)
    triangles.InsertNextCell(triangle1)
    # Triangle 2: v0, v1, v3
    triangle2 = vtk.vtkTriangle()
    triangle2.GetPointIds().SetId(0, 0)
    triangle2.GetPointIds().SetId(1, 1)
    triangle2.GetPointIds().SetId(2, 3)
    triangles.InsertNextCell(triangle2)
    # Triangle 3: v1, v3, v4
    triangle3 = vtk.vtkTriangle()
    triangle3.GetPointIds().SetId(0, 1)
    triangle3.GetPointIds().SetId(1, 3)
    triangle3.GetPointIds().SetId(2, 4)
    triangles.InsertNextCell(triangle3)
    # Triangle 4: v0, v3, v4
    triangle4 = vtk.vtkTriangle()
    triangle4.GetPointIds().SetId(0, 0)
    triangle4.GetPointIds().SetId(1, 3)
    triangle4.GetPointIds().SetId(2, 4)
    triangles.InsertNextCell(triangle4)
    # Triangle 5: v0, v4, v5
    triangle5 = vtk.vtkTriangle()
    triangle5.GetPointIds().SetId(0, 0)
    triangle5.GetPointIds().SetId(1, 4)
    triangle5.GetPointIds().SetId(2, 5)
    triangles.InsertNextCell(triangle5)
    # Triangle 6: v1, v4, v5
    triangle6 = vtk.vtkTriangle()
    triangle6.GetPointIds().SetId(0, 1)
    triangle6.GetPointIds().SetId(1, 4)
    triangle6.GetPointIds().SetId(2, 5)
    triangles.InsertNextCell(triangle6)
    # Triangle 7: v0, v5, v6
    triangle7 = vtk.vtkTriangle()
    triangle7.GetPointIds().SetId(0, 0)
    triangle7.GetPointIds().SetId(1, 5)
    triangle7.GetPointIds().SetId(2, 6)
    triangles.InsertNextCell(triangle7)
    # Triangle 8: v1, v5, v6
    triangle8 = vtk.vtkTriangle()
    triangle8.GetPointIds().SetId(0, 1)
    triangle8.GetPointIds().SetId(1, 5)
    triangle8.GetPointIds().SetId(2, 6)
    triangles.InsertNextCell(triangle8)
    # Triangle 9: v0, v2, v6
    triangle9 = vtk.vtkTriangle()
    triangle9.GetPointIds().SetId(0, 0)
    triangle9.GetPointIds().SetId(1, 2)
    triangle9.GetPointIds().SetId(2, 6)
    triangles.InsertNextCell(triangle9)
    # Triangle 10: v1, v2, v6
    triangle10 = vtk.vtkTriangle()
    triangle10.GetPointIds().SetId(0, 1)
    triangle10.GetPointIds().SetId(1, 2)
    triangle10.GetPointIds().SetId(2, 6)
    triangles.InsertNextCell(triangle10)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()

# =============================================================================
# Module 2: Retrieve Cell Normals
# =============================================================================
def get_cell_normal(polydata, cell_id):
    """
    Retrieve the computed cell normal (as a list of three values) from the polydata.
    """
    normals = polydata.GetCellData().GetArray("Normals")
    if normals:
        return list(normals.GetTuple(cell_id))
    return None

# =============================================================================
# Module 3: Build Oriented Edge List
# =============================================================================
def build_oriented_edge_list(polydata):
    """
    Build an oriented edge dictionary from the polydata.
    Returns a dictionary mapping canonical edge (sorted tuple of vertex IDs)
    to a list of tuples: [(cell_id, orientation flag), ...].
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
        for i in range(n_pts):
            a = pt_ids.GetId(i)
            b = pt_ids.GetId((i+1) % n_pts)
            c = pt_ids.GetId((i+2) % n_pts)  # next vertex in cycle

            p_a = polydata.GetPoint(a)
            p_b = polydata.GetPoint(b)
            p_c = polydata.GetPoint(c)
            ab = [p_b[j] - p_a[j] for j in range(3)]
            bc = [p_c[j] - p_b[j] for j in range(3)]
            n = get_cell_normal(polydata, cell_id)
            if n is None:
                n = [0, 0, 1]
            cross1 = [ab[1]*bc[2] - ab[2]*bc[1],
                      ab[2]*bc[0] - ab[0]*bc[2],
                      ab[0]*bc[1] - ab[1]*bc[0]]
            dot1 = sum(cross1[j] * n[j] for j in range(3))
            if dot1 >= 0:
                flag = 1
            else:
                flag = -1

            key = tuple(sorted((a, b)))
            if a > b:
                flag *= -1
            edge_dict[key].append((cell_id, flag))
    return edge_dict

# =============================================================================
# Module 4: Compute Signed Face Bending Angle from Edge Info
# =============================================================================
def compute_signed_face_angle_from_edge(polydata, edge_key, cell_info_list):
    """
    Given a canonical edge key and its two adjacent cell infos, compute the dihedral angle.
    Returns the computed dihedral angle in degrees.
    """
    if len(cell_info_list) != 2:
        return None
    cell1, flag1 = cell_info_list[0]
    cell2, flag2 = cell_info_list[1]
    n1 = get_cell_normal(polydata, cell1)
    n2 = get_cell_normal(polydata, cell2)
    if n1 is None or n2 is None:
        return None

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
    dot_val /= (n1_norm * n2_norm)
    # Compute angle in degrees using atan2 for sign.
    cross = [n1[1]*n2[2] - n1[2]*n2[1],
             n1[2]*n2[0] - n1[0]*n2[2],
             n1[0]*n2[1] - n1[1]*n2[0]]
    theta = math.atan2(sum(edge_vector[i]*cross[i] for i in range(3)), dot_val)
    inside_angle = math.degrees(theta)
    dihedral_angle = 180 + inside_angle
    return dihedral_angle

# =============================================================================
# Module 5: Highlight Cells Based on Bending Angle
# =============================================================================
def highlight_flat_cells(polydata, edge_dict, angle_threshold=15):
    """
    For each interior edge (with exactly two adjacent cells), compute the dihedral angle.
    If the difference from 360° is less than the threshold (or the angle itself is small),
    mark both adjacent cells.
    Returns a vtkUnsignedCharArray of cell colors (RGBA).
    """
    num_cells = polydata.GetNumberOfCells()
    cell_colors = vtk.vtkUnsignedCharArray()
    cell_colors.SetNumberOfComponents(4)
    cell_colors.SetName("Colors")
    for _ in range(num_cells):
        cell_colors.InsertNextTuple4(255, 255, 255, 51)  # white, 20% opacity

    highlight_cells = set()
    for key, cell_info_list in edge_dict.items():
        if len(cell_info_list) == 2:
            angle = compute_signed_face_angle_from_edge(polydata, key, cell_info_list)
            if angle is None:
                continue
            if abs(360.0 - angle) < angle_threshold or abs(angle) < angle_threshold:
                for (cell_id, _) in cell_info_list:
                    highlight_cells.add(cell_id)

    for cell_id in highlight_cells:
        cell_colors.SetTuple4(cell_id, 255, 0, 0, 255)  # red, fully opaque
    return cell_colors

# =============================================================================
# Module 6: Create Normals Glyph Actor
# =============================================================================
def create_normals_glyph_actor(polydata, scale_factor=0.1):
    """
    Create an actor to visualize cell normals as arrows.
    Glyphs are placed at the center of each cell.
    """
    centers = vtk.vtkCellCenters()
    centers.SetInputData(polydata)
    centers.Update()

    arrowSource = vtk.vtkArrowSource()

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrowSource.GetOutputPort())
    glyph.SetInputConnection(centers.GetOutputPort())
    glyph.SetVectorModeToUseNormal()
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(scale_factor)
    glyph.OrientOn()
    glyph.SetColorModeToColorByScalar()
    glyph.Update()

    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph.GetOutputPort())

    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    return glyph_actor

# =============================================================================
# Module 7: Visualization Setup
# =============================================================================
def setup_renderers(polydata, cell_colors, glyph_actor):
    """
    Setup renderer, render window, and interactor.
    """
    polydata.GetCellData().SetScalars(cell_colors)

    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(polydata)
    mesh_mapper.SetScalarModeToUseCellData()
    mesh_mapper.SetColorModeToDirectScalars()
    mesh_mapper.SetInterpolateScalarsBeforeMapping(False)

    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().EdgeVisibilityOn()
    mesh_actor.GetProperty().SetEdgeColor(0, 0, 0)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(mesh_actor)
    renderer.AddActor(glyph_actor)
    renderer.SetBackground(1, 1, 1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 600)
    renderWindow.SetAlphaBitPlanes(True)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(renderWindow)

    return renderWindow, interactor

# =============================================================================
# Module 8: Sparse Matrix Based k-Ring Neighborhood
# =============================================================================
def build_adjacency_matrix(polyData):
    """
    Build a sparse adjacency matrix (CSR format) for the mesh.
    Each nonzero entry (i,j) indicates vertices i and j are adjacent.
    """
    num_points = polyData.GetNumberOfPoints()
    rows = []
    cols = []

    polyData.BuildLinks()
    num_cells = polyData.GetNumberOfCells()

    for cell_id in range(num_cells):
        cell = polyData.GetCell(cell_id)
        num_ids = cell.GetNumberOfPoints()
        # For each cell, add all unique vertex pairs.
        point_ids = [cell.GetPointId(i) for i in range(num_ids)]
        for i in range(len(point_ids)):
            for j in range(i+1, len(point_ids)):
                p1 = point_ids[i]
                p2 = point_ids[j]
                rows.extend([p1, p2])
                cols.extend([p2, p1])
    data = np.ones(len(rows), dtype=np.int8)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_points, num_points))
    return A.tocsr()

# -------------------------------------------------------------------------
# Build edge‑adjacency / vertex‑only‑adjacency **only for a subset** of
# triangles identified by their global cell‑ids.
# -------------------------------------------------------------------------
def build_subset_adjacency(polydata, cell_ids):
    """
    Parameters
    ----------
    polydata : vtkPolyData    (the global mesh)
    cell_ids : list[int]      (global cell ids of the working subset)

    Returns
    -------
    tris_global  : (m,3) int   vertex indices of the subset
    edge_adj     : list[ set[int] ] length m
    vert_adj     : list[ set[int] ] length m
    """
    # ----- gather triangles --------------------------------------------------
    tri_ids_map = {cid: k for k, cid in enumerate(cell_ids)}   # global→local
    tris_global = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:][cell_ids]

    m = len(cell_ids)
    edge_adj = [set() for _ in range(m)]
    vert_adj = [set() for _ in range(m)]

    # ----- edge adjacency ----------------------------------------------------
    edge2local = {}
    for loc_id, (a, b, c) in enumerate(tris_global):
        for e in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(e))
            edge2local.setdefault(key, []).append(loc_id)
    for ids in edge2local.values():
        if len(ids) > 1:
            for i in ids:
                edge_adj[i].update(ids)

    # ----- vertex‑only adjacency --------------------------------------------
    v2local = {}
    for loc_id, tri in enumerate(tris_global):
        for v in tri:
            v2local.setdefault(int(v), []).append(loc_id)
    for ids in v2local.values():
        if len(ids) > 1:
            for i in ids:
                for j in ids:
                    if j == i or j in edge_adj[i]:
                        continue
                    vert_adj[i].add(j)

    return tris_global, edge_adj, vert_adj

def k_ring_neighbors_sparse(A, start_vertex, k):
    """
    Compute the k-ring neighborhood using repeated sparse matrix multiplication.
    Returns a set of vertex indices within k rings of start_vertex.
    """
    num_points = A.shape[0]
    current = np.zeros(num_points, dtype=np.int8)
    current[start_vertex] = 1
    visited = current.copy()

    for _ in range(k):
        current = A.dot(current)
        current = (current > 0).astype(np.int8)
        current = current * (1 - visited)
        visited = np.maximum(visited, current)
    return set(np.nonzero(visited)[0])

# =============================================================================
# Module 9: Self-Intersection Correction (Existing Code)
# =============================================================================
EPS = 1e-12          # global geometric tolerance
# ------------------------------------------------------------------
#  Basic helpers
# ------------------------------------------------------------------
def plane_from_triangle(tri):
    """Return (n, d) for the plane n·x + d = 0 containing the triangle."""
    v0, v1, v2 = tri
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n)
    if n_norm < EPS:
        raise ValueError("Degenerate triangle")
    n /= n_norm
    d = -np.dot(n, v0)
    return n, d

def signed_dist(points, n, d):
    """Signed distance of point(s) to plane n·x + d = 0."""
    return np.dot(points, n) + d

def edge_plane_intersection(p0, p1, d0, d1):
    """Return the intersection point of edge (p0,p1) with a plane,
       given the signed distances d0, d1 of endpoints to that plane.
    """
    t = d0 / (d0 - d1)          # safe because d0 and d1 not both zero when called
    return p0 + t * (p1 - p0)

def unique_rows(pts):
    """Deduplicate points within EPS tolerance."""
    if len(pts) == 0:
        return pts
    # Sort lexicographically to cluster near‑duplicates
    idx = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    pts = pts[idx]
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > EPS:
            keep.append(i)
    return pts[keep]

def point_in_triangle(p, tri, n):
    """Barycentric test, robust to coplanar tolerance."""
    v0, v1, v2 = tri
    u = v1 - v0
    v = v2 - v0
    w = p  - v0
    uv = np.dot(u, v)
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    wu = np.dot(w, u)
    wv = np.dot(w, v)
    denom = uv * uv - uu * vv
    if abs(denom) < EPS:
        return False
    s = (uv * wv - vv * wu) / denom
    t = (uv * wu - uu * wv) / denom
    return (-EPS <= s <= 1+EPS) and (-EPS <= t <= 1+EPS) and (s + t <= 1+EPS)

# ------------------------------------------------------------------
#  Main routine
# ------------------------------------------------------------------
def triangle_triangle_intersection(T1, T2):
    """
    Parameters
    ----------
    T1, T2 : ndarray, shape (3, 3)
        Cartesian coordinates of the two triangles' vertices.

    Returns
    -------
    result : dict with keys
        'intersects' : bool
        'type'       : 'none' | 'point' | 'segment' | 'area'
        'points'     : ndarray (k, 3)  intersection vertices (empty if none)
    """
    # -- 1.  plane equations --------------------------------------------------
    n1, d1 = plane_from_triangle(T1)
    n2, d2 = plane_from_triangle(T2)

    # Signed distances of triangle vertices to opposite planes
    sd1 = signed_dist(T1, n2, d2)
    sd2 = signed_dist(T2, n1, d1)

    # Quick rejection: all on same strict side
    if (np.all(sd1 >  EPS) or np.all(sd1 < -EPS) or
        np.all(sd2 >  EPS) or np.all(sd2 < -EPS)):
        return {'intersects': False, 'type': 'none', 'points': np.empty((0, 3))}

    # -- 2.  Collect candidate intersection points ---------------------------
    pts = []

    # (a)  edges of T1 vs. plane of T2
    edges1 = [(0,1), (1,2), (2,0)]
    for i0, i1 in edges1:
        d0, d1_e = sd1[i0], sd1[i1]
        if d0 * d1_e < -EPS**2:                       # opposite signs
            pts.append(edge_plane_intersection(T1[i0], T1[i1], d0, d1_e))
        elif abs(d0) < EPS:                           # endpoint on plane
            pts.append(T1[i0])
        elif abs(d1_e) < EPS:
            pts.append(T1[i1])

    # (b)  edges of T2 vs. plane of T1
    edges2 = [(0,1), (1,2), (2,0)]
    for j0, j1 in edges2:
        d0, d1_e = sd2[j0], sd2[j1]
        if d0 * d1_e < -EPS**2:
            pts.append(edge_plane_intersection(T2[j0], T2[j1], d0, d1_e))
        elif abs(d0) < EPS:
            pts.append(T2[j0])
        elif abs(d1_e) < EPS:
            pts.append(T2[j1])

    if len(pts) == 0:
        # Coplanar but disjoint or numerical fall‑through
        return {'intersects': False, 'type': 'none', 'points': np.empty((0, 3))}

    pts = unique_rows(np.asarray(pts))

    # -- 3.  Filter points that truly lie inside both triangles --------------
    keep = []
    for p in pts:
        if point_in_triangle(p, T1, n1) and point_in_triangle(p, T2, n2):
            keep.append(p)
    pts = unique_rows(np.asarray(keep))

    if len(pts) == 0:
        return {'intersects': False, 'type': 'none', 'points': np.empty((0, 3))}

    # -- 4.  Classify the dimension of the intersection ----------------------
    if len(pts) == 1 or np.max(np.linalg.norm(pts - pts[0], axis=1)) < EPS:
        itype = 'point'
    elif abs(np.dot(n1, n2) - 1) < EPS and len(pts) >= 3:
        # Coplanar overlap with area (triangles not parallel exactly? They are.)
        # Compute convex hull area in plane coordinates (project to dominant axis)
        itype = 'area'
    else:
        # Non‑coplanar line segment (or coplanar segment)
        itype = 'segment'

    return {'intersects': True, 'type': itype, 'points': pts}


# -------------------------------------------------------------------------
# Push‑normal untangle  (works on the subset only)
# -------------------------------------------------------------------------
def untangle_push_normals(polydata, cell_ids, step=0.01,
                          max_iter=15, broad_r=0.05):
    pts = vtk_to_numpy(polydata.GetPoints().GetData())
    tris, edge_adj, vert_adj = build_subset_adjacency(polydata, cell_ids)

    # local → global helper
    loc2glob = np.asarray(cell_ids, int)

    for _ in range(max_iter):
        T   = pts[tris]                # (m,3,3)
        cen = T.mean(1)
        nrm = np.cross(T[:,1]-T[:,0], T[:,2]-T[:,0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

        kd   = cKDTree(cen)
        cand = kd.query_ball_tree(kd, r=broad_r)     # neighbour lists

        moved = False
        disp  = np.zeros_like(pts)

        for i, neighbors in enumerate(cand):
            neighbors = [j for j in neighbors
                     if j not in edge_adj[i] and j != i]
            if not neighbors:
                continue

            nontrivial = False
            for j in neighbors:
                info = triangle_triangle_intersection(T[i], T[j])
                if not info['intersects']:
                    continue
                if info['type'] != 'point' or j not in vert_adj[i]:
                    nontrivial = True
                    break
            if not nontrivial:
                continue

            # vec = cen[neighbors] - cen[i]
            # sgn = np.sign(np.einsum('ij,ij->i', vec, nrm[i])).mean()
            disp[tris[i]] += -step * nrm[i]
            moved = True

        if not moved:
            break
        pts += disp

    polydata.GetPoints().GetData().Modified()

def rotate_point_around_line(point, line_point1, line_point2, angle_degrees):
    """
    Rotate 'point' about the axis defined by line_point1 and line_point2 by angle_degrees.
    Uses Rodrigues' rotation formula.
    """
    v = [point[i] - line_point1[i] for i in range(3)]
    axis = [line_point2[i] - line_point1[i] for i in range(3)]
    axis_norm = math.sqrt(sum(x*x for x in axis))
    if axis_norm == 0:
        return point
    axis = [x/axis_norm for x in axis]
    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    dot = sum(axis[i]*v[i] for i in range(3))
    cross = [axis[1]*v[2] - axis[2]*v[1],
             axis[2]*v[0] - axis[0]*v[2],
             axis[0]*v[1] - axis[1]*v[0]]
    v_rot = [v[i]*cos_theta + cross[i]*sin_theta + axis[i]*dot*(1-cos_theta) for i in range(3)]
    new_point = [line_point1[i] + v_rot[i] for i in range(3)]
    return new_point

def correct_self_intersections(polydata, edge_dict, threshold, angle_increment):
    """
    For interior edges where the dihedral angle is near 360 or very small,
    rotate the non-edge vertex of each adjacent cell about the shared edge.
    """
    num_cells = polydata.GetNumberOfCells()
    points = polydata.GetPoints()
    num_corrected_edges = 0
    for key, cell_info_list in edge_dict.items():
        if len(cell_info_list) != 2:
            continue
        dihedral_angle = compute_signed_face_angle_from_edge(polydata, key, cell_info_list)
        if dihedral_angle is None:
            continue
        if abs(360.0 - dihedral_angle) < threshold or abs(dihedral_angle) < threshold:
            num_corrected_edges += 1
            print(f"Correcting edge {key}: dihedral angle = {dihedral_angle:.2f}°")
            for (cell_id, flag) in cell_info_list:
                cell = polydata.GetCell(cell_id)
                pt_ids = cell.GetPointIds()
                non_edge_vertex = None
                for i in range(pt_ids.GetNumberOfIds()):
                    v = pt_ids.GetId(i)
                    if v not in key:
                        non_edge_vertex = v
                        break
                if non_edge_vertex is None:
                    continue
                p = list(points.GetPoint(non_edge_vertex))
                axis_p0 = polydata.GetPoint(key[0])
                axis_p1 = polydata.GetPoint(key[1])
                if flag == 1:
                    new_p = rotate_point_around_line(p, axis_p0, axis_p1, -angle_increment)
                else:
                    new_p = rotate_point_around_line(p, axis_p0, axis_p1, angle_increment)
                points.SetPoint(non_edge_vertex, new_p)
                points.Modified()
    print(f"Corrected {num_corrected_edges} edges in this step.")

def correct_self_intersections_callback(obj, event):
    """
    Callback invoked when the "Correct Self-Intersections" button is pressed.
    """
    global global_polydata, global_edge_dict, global_cell_colors, global_renderWindow, global_glyph_actor
    print("Running self-intersection correction...")
    correct_self_intersections(global_polydata, global_edge_dict, threshold=15, angle_increment=2)
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(global_polydata)
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.SplittingOff()
    normals_filter.Update()
    global_polydata.ShallowCopy(normals_filter.GetOutput())
    global_cell_colors = highlight_flat_cells(global_polydata, global_edge_dict, angle_threshold=40)
    global_polydata.GetCellData().SetScalars(global_cell_colors)
    global_renderWindow.Render()
    new_glyph_actor = create_normals_glyph_actor(global_polydata, scale_factor=0.1)
    global_renderWindow.GetRenderers().GetFirstRenderer().RemoveActor(global_glyph_actor)
    global_renderWindow.GetRenderers().GetFirstRenderer().AddActor(new_glyph_actor)
    global_renderWindow.Render()
    global_glyph_actor = new_glyph_actor
    print("Self-intersection correction complete.")

def push_normals_callback(obj, event):
    global global_polydata, global_selected_vertices, global_renderWindow

    if not global_selected_vertices:
        print("No grow‑selection – nothing to correct.")
        return

    # collect global cell ids that touch at least one selected vertex
    sel_cells = []
    for cid in range(global_polydata.GetNumberOfCells()):
        cell = global_polydata.GetCell(cid)
        if any(cell.GetPointId(i) in global_selected_vertices for i in range(3)):
            sel_cells.append(cid)

    if not sel_cells:
        print("Selection contains no triangles.")
        return

    print(f"Push‑normal untangle on {len(sel_cells)} triangles …")
    untangle_push_normals(global_polydata, sel_cells,
                          step=0.001, max_iter=1, broad_r=0.1)
    global_renderWindow.Render()
    print("Done.")

def create_push_normals_button(interactor):
    txt = vtk.vtkTextActor()
    txt.SetInput("Correct Intersections (push‑normals)")
    tp  = txt.GetTextProperty(); tp.SetFontSize(24); tp.SetColor(0,0,0)
    rep = vtk.vtkTextRepresentation()
    rep.GetPositionCoordinate().SetValue(0.35, 0.46)
    rep.GetPosition2Coordinate().SetValue(0.3, 0.1)
    w   = vtk.vtkTextWidget()
    w.SetInteractor(interactor)
    w.SetRepresentation(rep)
    w.SetTextActor(txt)
    w.On()
    w.AddObserver("EndInteractionEvent", push_normals_callback)
    return w

def create_button_widget(interactor):
    """
    Create a text widget button that triggers self-intersection correction.
    """
    textActor = vtk.vtkTextActor()
    textActor.SetInput("Correct Self-Intersections")
    textProp = textActor.GetTextProperty()
    textProp.SetFontSize(24)
    textProp.SetColor(0, 0, 0)
    rep = vtk.vtkTextRepresentation()
    rep.GetPositionCoordinate().SetValue(0.35, 0.01)
    rep.GetPosition2Coordinate().SetValue(0.3, 0.1)
    buttonWidget = vtk.vtkTextWidget()
    buttonWidget.SetInteractor(interactor)
    buttonWidget.SetRepresentation(rep)
    buttonWidget.SetTextActor(textActor)
    buttonWidget.On()
    buttonWidget.AddObserver("EndInteractionEvent", correct_self_intersections_callback)
    return buttonWidget

# =============================================================================
# Module 10: Save Mesh Callback and Save Button Widget
# =============================================================================
def save_mesh_callback(obj, event):
    """
    Save the current global_polydata to a VTP file.
    """
    global global_polydata
    writer = vtk.vtkXMLPolyDataWriter()
    file_name = "april-8-laplacian-corrected-15-degrees.vtp"
    file_name = "march-24-SI-Test-Fine-Mesh.vtp"
    file_name = "april-14-SI-Laplacian-Fixed-perpendicular-typical-scale-test.vtp"
    file_name = "may-7-minimum-fix-0.001-movement.vtp"
    file_name = "may-7-march-24-SI-Test-Two-0.001-movement.vtp"
    file_name = "05-13-2025-dihedral-fixed-pymeshfix-minimum-fix-0.001.vtp"

    writer.SetFileName(file_name)
    writer.SetInputData(global_polydata)
    writer.Write()
    print("Mesh saved to", "'"+file_name+"'.")

def create_save_button_widget(interactor):
    """
    Create a "Save Mesh" button widget.
    """
    textActor = vtk.vtkTextActor()
    textActor.SetInput("Save Mesh")
    textProp = textActor.GetTextProperty()
    textProp.SetFontSize(24)
    textProp.SetColor(0, 0, 0)
    rep = vtk.vtkTextRepresentation()
    rep.GetPositionCoordinate().SetValue(0.35, 0.12)
    rep.GetPosition2Coordinate().SetValue(0.3, 0.1)
    saveButtonWidget = vtk.vtkTextWidget()
    saveButtonWidget.SetInteractor(interactor)
    saveButtonWidget.SetRepresentation(rep)
    saveButtonWidget.SetTextActor(textActor)
    saveButtonWidget.On()
    saveButtonWidget.AddObserver("EndInteractionEvent", save_mesh_callback)
    return saveButtonWidget

# =============================================================================
# Module 11: Grow Selection Using k-Ring Neighborhood
# =============================================================================
def grow_selection_callback(obj, event):
    """
    Callback for the "Grow Selection" button.
    Finds one edge whose dihedral angle triggers the threshold,
    extracts all vertices from its adjacent cells,
    computes their 2-ring neighborhood using the sparse adjacency matrix method, and then
    highlights all cells that have at least one vertex in the neighborhood.
    Also stores the selected vertices globally.
    """
    global global_polydata, global_edge_dict, global_cell_colors, global_renderWindow, global_selected_vertices

    print("Running Grow Selection...")

    vertices = set()
    for key, cell_info_list in global_edge_dict.items():
        if len(cell_info_list) != 2:
            continue
        angle = compute_signed_face_angle_from_edge(global_polydata, key, cell_info_list)
        if angle is None:
            continue
        if abs(360.0 - angle) < 15 or abs(angle) < 15:
            # For each adjacent cell, add all its vertices.
            for (cell_id, _) in cell_info_list:
                cell = global_polydata.GetCell(cell_id)
                pt_ids = cell.GetPointIds()
                for i in range(pt_ids.GetNumberOfIds()):
                    vertices.add(pt_ids.GetId(i))
    print("Initial selected vertices (from edge):", vertices)

    # Build the sparse adjacency matrix.
    A = build_adjacency_matrix(global_polydata)
    # Compute the union of 2-ring neighborhoods for all these vertices.
    k = 2
    selected_vertices = set()
    for v in vertices:
        selected_vertices |= k_ring_neighbors_sparse(A, v, k)
    print("2-ring neighborhood vertices:", selected_vertices)

    # Store the selected vertices globally for later smoothing.
    global_selected_vertices = selected_vertices

    # Now highlight cells that have at least one vertex in the selected vertex set.
    num_cells = global_polydata.GetNumberOfCells()
    existing_colors = global_polydata.GetCellData().GetScalars()
    new_colors = vtk.vtkUnsignedCharArray()
    new_colors.SetNumberOfComponents(4)
    new_colors.SetName("Colors")
    for cell_id in range(num_cells):
        cell = global_polydata.GetCell(cell_id)
        pt_ids = cell.GetPointIds()
        is_selected = any(pt_ids.GetId(i) in selected_vertices for i in range(pt_ids.GetNumberOfIds()))
        new_color = (0, 0, 255, 255) if is_selected else (255, 255, 255, 51)
        old_color = existing_colors.GetTuple4(cell_id)
        new_color = old_color if old_color == (255, 0, 0, 255) else new_color
        new_colors.InsertNextTuple4(*new_color)
    global_polydata.GetCellData().SetScalars(new_colors)
    global_renderWindow.Render()
    print("Grow selection complete.")

def create_grow_selection_button(interactor):
    """
    Create a "Grow Selection" button widget.
    """
    textActor = vtk.vtkTextActor()
    textActor.SetInput("Grow Selection (2-ring)")
    textProp = textActor.GetTextProperty()
    textProp.SetFontSize(24)
    textProp.SetColor(0, 0, 0)
    rep = vtk.vtkTextRepresentation()
    rep.GetPositionCoordinate().SetValue(0.35, 0.23)
    rep.GetPosition2Coordinate().SetValue(0.3, 0.1)
    buttonWidget = vtk.vtkTextWidget()
    buttonWidget.SetInteractor(interactor)
    buttonWidget.SetRepresentation(rep)
    buttonWidget.SetTextActor(textActor)
    buttonWidget.On()
    buttonWidget.AddObserver("EndInteractionEvent", grow_selection_callback)
    return buttonWidget

# =============================================================================
# Module 12: Laplacian Smoothing on Selected Vertices
# =============================================================================
def laplacian_smooth_selected(polydata, selected_vertices, num_iterations=1):
    """
    Perform Laplacian smoothing only on the vertices within 'selected_vertices'.
    Each vertex is updated to the average of its neighboring positions,
    where neighbors are taken only from within the selected set.
    """
    # Build the sparse adjacency matrix.
    A = build_adjacency_matrix(polydata)
    points = polydata.GetPoints()
    # For a number of iterations:
    for iteration in range(num_iterations):
        new_positions = {}
        for v in selected_vertices:
            # Extract the nonzero indices in row v.
            row = A.getrow(v).nonzero()[1]
            # Restrict neighbors to those in selected_vertices.
            neighbors = set(row).intersection(selected_vertices)
            if neighbors:
                pos_sum = np.zeros(3)
                count = 0
                for n in neighbors:
                    pos_sum += np.array(points.GetPoint(n))
                    count += 1
                avg_pos = (pos_sum / count).tolist()
                new_positions[v] = avg_pos
        # Update the positions after computing the new positions.
        for v, pos in new_positions.items():
            points.SetPoint(v, pos)
        points.Modified()
    print("Laplacian smoothing completed on selected vertices.")

def laplacian_smooth_callback(obj, event):
    """
    Callback for the Laplacian Smooth button.
    Performs Laplacian smoothing on the selected vertices and updates the rendering.
    """
    global global_polydata, global_selected_vertices, global_renderWindow
    print("Performing Laplacian smoothing on selected vertices...")
    if not global_selected_vertices:
        print("No vertices selected for smoothing. Please run Grow Selection first.")
        return
    laplacian_smooth_selected(global_polydata, global_selected_vertices, num_iterations=1)
    global_renderWindow.Render()
    print("Laplacian smoothing complete.")

def create_laplacian_smooth_button(interactor):
    """
    Create a "Laplacian Smooth" button widget.
    """
    textActor = vtk.vtkTextActor()
    textActor.SetInput("Laplacian Smooth Selection")
    textProp = textActor.GetTextProperty()
    textProp.SetFontSize(24)
    textProp.SetColor(0, 0, 0)
    rep = vtk.vtkTextRepresentation()
    # Position this widget above the Grow Selection button.
    rep.GetPositionCoordinate().SetValue(0.35, 0.35)
    rep.GetPosition2Coordinate().SetValue(0.3, 0.1)
    buttonWidget = vtk.vtkTextWidget()
    buttonWidget.SetInteractor(interactor)
    buttonWidget.SetRepresentation(rep)
    buttonWidget.SetTextActor(textActor)
    buttonWidget.On()
    buttonWidget.AddObserver("EndInteractionEvent", laplacian_smooth_callback)
    return buttonWidget

# =============================================================================
# Module 13: Main Demo Function
# =============================================================================
def main():
    global global_polydata, global_edge_dict, global_cell_colors, global_renderWindow, global_glyph_actor

    # Load polydata from file (or use one of the manually created examples).
    global_polydata = load_polydata_with_normals(FILE_PATH) 
    # Alternatively, for manual examples:
    # global_polydata = create_two_triangle_polydata()
    # global_polydata = create_simple_manifold_polydata()

    # Build the oriented edge list.
    global_edge_dict = build_oriented_edge_list(global_polydata)
    # Highlight cells based on bending angle.
    global_cell_colors = highlight_flat_cells(global_polydata, global_edge_dict, angle_threshold=40)

    # Create a normals glyph actor.
    glyph_actor = create_normals_glyph_actor(global_polydata, scale_factor=0.1)
    global_glyph_actor = glyph_actor

    # Setup the renderer, render window, and interactor.
    global_renderWindow, interactor = setup_renderers(global_polydata, global_cell_colors, glyph_actor)

    # Create and add the self-intersection correction button widget.
    correctionButtonWidget = vtk.vtkTextWidget()
    correctionTextActor = vtk.vtkTextActor()
    correctionTextActor.SetInput("Correct Self-Intersections")
    correctionTextProp = correctionTextActor.GetTextProperty()
    correctionTextProp.SetFontSize(24)
    correctionTextProp.SetColor(0, 0, 0)
    correctionRep = vtk.vtkTextRepresentation()
    correctionRep.GetPositionCoordinate().SetValue(0.35, 0.01)
    correctionRep.GetPosition2Coordinate().SetValue(0.3, 0.1)
    correctionButtonWidget.SetInteractor(interactor)
    correctionButtonWidget.SetRepresentation(correctionRep)
    correctionButtonWidget.SetTextActor(correctionTextActor)
    correctionButtonWidget.On()
    correctionButtonWidget.AddObserver("EndInteractionEvent", correct_self_intersections_callback)

    # Create and add the save mesh button widget.
    saveButtonWidget = create_save_button_widget(interactor)
    
    # Create and add the grow selection button widget.
    growSelectionButton = create_grow_selection_button(interactor)
    
    # Create and add the Laplacian smooth button widget.
    laplacianSmoothButton = create_laplacian_smooth_button(interactor)

    # Create and add the push normals button widget.
    pushNormalsButton = create_push_normals_button(interactor)

    global_renderWindow.Render()
    interactor.Initialize()
    interactor.Start()

if __name__ == "__main__":
    main()