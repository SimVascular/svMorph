'''
#!/usr/bin/env python3
# separate_with_button_demo.py
#
# python -m pip install vtk numpy scipy

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy.spatial import cKDTree


# ╭───────────────────────────────────────────────────────────────────────────╮
# │  0.  Simple utilities  (NumPy ↔ VTK)                                      │
# ╰───────────────────────────────────────────────────────────────────────────╯
def polydata_from(points: np.ndarray, tris: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(points, deep=True))
    cells = vtk.vtkCellArray()
    for t in tris:
        cells.InsertNextCell(3, t)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def numpy_triangles(poly: vtk.vtkPolyData) -> np.ndarray:
    pts = vtk_to_numpy(poly.GetPoints().GetData())
    ids = vtk.vtkIdList()
    tri = []
    for c in range(poly.GetNumberOfCells()):
        poly.GetCellPoints(c, ids)
        tri.append(pts[[ids.GetId(0), ids.GetId(1), ids.GetId(2)]])
    ret = np.asarray(tri)  # (n,3,3)
    print(f"ret shape: {ret.shape}")  # debug
    return np.asarray(tri)                        # (n,3,3)


def cell_normals(poly: vtk.vtkPolyData) -> np.ndarray:
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(poly)
    n.ComputeCellNormalsOn(); n.ComputePointNormalsOff(); n.ConsistencyOn()
    n.Update()
    return vtk_to_numpy(n.GetOutput().GetCellData().GetNormals())


# ╭───────────────────────────────────────────────────────────────────────────╮
# │  1.  Exact SAT triangle–triangle test  (vectorised NumPy)                 │
# ╰───────────────────────────────────────────────────────────────────────────╯
def tri_tri_intersect(T1, T2):
    """Vectorised Möller–Akenine SAT; T1,T2 (...,3,3) → bool (...,)"""
    u0, v0 = T1[..., 1, :] - T1[..., 0, :], T1[..., 2, :] - T1[..., 0, :]
    u1, v1 = T2[..., 1, :] - T2[..., 0, :], T2[..., 2, :] - T2[..., 0, :]
    n1, n2 = np.cross(u0, v0), np.cross(u1, v1)

    dB = np.einsum('...i,...ji->...j', n1, T2 - T1[..., :1, :])
    print(f"dB shape: {dB.shape}")  # debug
    sep1 = (np.sign(dB[..., 0]) * np.sign(dB[..., 1]) > 0) & \
           (np.sign(dB[..., 0]) * np.sign(dB[..., 2]) > 0)

    dA = np.einsum('...i,...ji->...j', n2, T1 - T2[..., :1, :])
    sep2 = (np.sign(dA[..., 0]) * np.sign(dA[..., 1]) > 0) & \
           (np.sign(dA[..., 0]) * np.sign(dA[..., 2]) > 0)

    return ~(sep1 | sep2)


# ╭───────────────────────────────────────────────────────────────────────────╮
# │  2.  KD-tree broad-phase → intersecting triangle index lists              │
# ╰───────────────────────────────────────────────────────────────────────────╯
def intersecting_cells(polyA, polyB, radius=1e-2):
    TA, TB = numpy_triangles(polyA), numpy_triangles(polyB)
    cA, cB = TA.mean(1), TB.mean(1)
    rA = np.linalg.norm(TA - cA[:, None, :], axis=2).max(1)
    rB = np.linalg.norm(TB - cB[:, None, :], axis=2).max(1)

    kd = cKDTree(cA)
    cand_lists = kd.query_ball_point(cB, (rA.max() + rB.max() + radius))

    hitA, hitB = [], []
    for j, nbr in enumerate(cand_lists):
        if nbr:
            ia = np.array(nbr, int)
            ib = np.full(len(ia), j, int)
            if tri_tri_intersect(TA[ia], TB[ib]).any():
                hitA.extend(ia); hitB.append(j)
    return np.unique(hitA), np.unique(hitB)


# ╭───────────────────────────────────────────────────────────────────────────╮
# │  3.  Build two slightly intersecting square patches                       │
# ╰───────────────────────────────────────────────────────────────────────────╯
quad_pts = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [1, 1, 0]], float)
quad_tris = np.array([[0, 1, 3], [0, 3, 2]], int)
quad_pts_A = quad_pts + np.array([[0, 0, -0.01],
                     [0.0, 0.0, -0.01],
                     [0.0, 0.0, -0.01],
                     [0.0, 0.0, -0.01]], float)
all_pts = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [1, 1, 0],
                     [1, 0, 0],
                     [1, 1, 0]], float) + np.array([[0.05, 0.05, -0.0],
                     [0.05, 0.05, -0.1],
                     [0.05, 0.05, -0.1],
                     [0.05, 0.05, -0.0],
                     [0.0, 0.0, -0.01],
                     [0.0, 0.0, -0.01]], float)

patchA = polydata_from(quad_pts_A, quad_tris)      # +z
patchA = polydata_from(all_pts, np.array([[0, 1, 3], [0, 3, 2], [0, 4, 5], [0, 5, 2]], int))  # +z
quad_pts_B = quad_pts + np.array([[0.05, 0.05, -0.0],
                     [0.05, 0.05, -0.1],
                     [0.05, 0.05, -0.1],
                     [0.05, 0.05, -0.0]], float)
patchB = polydata_from(quad_pts_B, quad_tris)  # –z


# ╭───────────────────────────────────────────────────────────────────────────╮
# │  4.  VTK actors with edge visibility                                      │
# ╰───────────────────────────────────────────────────────────────────────────╯
def make_actor(pd, rgb):
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(pd)
    act = vtk.vtkActor(); act.SetMapper(mapper)
    act.GetProperty().SetColor(*rgb)
    act.GetProperty().EdgeVisibilityOn()
    act.GetProperty().SetEdgeColor(0, 0, 0)
    act.GetProperty().SetLineWidth(1.5)
    return act

actorA = make_actor(patchA, (1, 0, 0))
actorB = make_actor(patchB, (0, 0.7, 0))


# ╭───────────────────────────────────────────────────────────────────────────╮
# │  5.  “Separate” button – text actor + picking                             │
# ╰───────────────────────────────────────────────────────────────────────────╯
txt = vtk.vtkTextActor()
txt.SetInput("Separate")
tp = txt.GetTextProperty()
tp.SetFontSize(14); tp.BoldOn(); tp.SetColor(0, 0, 0)
txt.SetDisplayPosition(10, 10)     # lower-left corner

# renderer, window, interactor
ren = vtk.vtkRenderer(); ren.SetBackground(1, 1, 1)
ren.AddActor(actorA); ren.AddActor(actorB); ren.AddActor2D(txt)

win = vtk.vtkRenderWindow(); win.AddRenderer(ren); win.SetSize(600, 600)
iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

#  pick helper: screen-space bbox of text
def text_bbox(act):
    x, y = act.GetPosition()
    tp = act.GetTextProperty()
    w = tp.GetFontSize() * len(act.GetInput()) * 0.6
    h = tp.GetFontSize() * 1.2
    return x, y, x + w, y + h

#  left-click callback
def on_left_click(obj, evt):
    x, y = obj.GetEventPosition()
    x0, y0, x1, y1 = text_bbox(txt)
    if x0 <= x <= x1 and y0 <= y <= y1:         # inside “button”
        hitA, hitB = intersecting_cells(patchA, patchB, radius=0.05)
        if hitA.size and hitB.size:
            nA = cell_normals(patchA)[hitA].mean(0); nA /= np.linalg.norm(nA)
            nB = -cell_normals(patchB)[hitB].mean(0); nB /= np.linalg.norm(nB)
            gap = 0.02                                       # 2 cm total
            for poly, n in ((patchA, nA), (patchB, nB)):
                tf = vtk.vtkTransform(); tf.Translate(0.5 * gap * n)
                tpf = vtk.vtkTransformPolyDataFilter()
                tpf.SetTransform(tf); tpf.SetInputData(poly); tpf.Update()
                poly.ShallowCopy(tpf.GetOutput())
        win.Render()                 # refresh
    # obj.OnLeftButtonDown()           # keep default trackball behaviour

iren.AddObserver("LeftButtonPressEvent", on_left_click, 1.0)  # high priority

win.Render(); iren.Start()
'''
#!/usr/bin/env python3
# untangle_self_intersect_demo.py

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree


# ────────────────────────── NumPy ↔ VTK helpers ────────────────────────────
def polydata_from(points, tris):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(points, deep=True))
    cells = vtk.vtkCellArray()
    for t in tris:
        cells.InsertNextCell(3, t)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts);  pd.SetPolys(cells)
    return pd


def np_triangles(poly):
    pts = vtk_to_numpy(poly.GetPoints().GetData())
    ids = vtk.vtkIdList();  out = []
    for c in range(poly.GetNumberOfCells()):
        poly.GetCellPoints(c, ids)
        out.append(pts[[ids.GetId(i) for i in range(3)]])
    return np.asarray(out)                       # (n,3,3)


def cell_normals(poly):
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(poly)
    n.ComputeCellNormalsOn(); n.ComputePointNormalsOff(); n.ConsistencyOn()
    n.Update()
    return vtk_to_numpy(n.GetOutput().GetCellData().GetNormals())


# ───────────────────── vectorised triangle–triangle SAT ────────────────────
def tri_tri_intersect_bad(T1, T2):
    u0, v0 = T1[..., 1]-T1[..., 0], T1[..., 2]-T1[..., 0]
    u1, v1 = T2[..., 1]-T2[..., 0], T2[..., 2]-T2[..., 0]
    n1, n2 = np.cross(u0, v0), np.cross(u1, v1)

    dB = np.einsum('...i,...ji->...j', n1, T2 - T1[..., :1, :])
    sep1 = (np.sign(dB[..., 0])*np.sign(dB[..., 1]) > 0) & \
           (np.sign(dB[..., 0])*np.sign(dB[..., 2]) > 0)

    dA = np.einsum('...i,...ji->...j', n2, T1 - T2[..., :1, :])
    sep2 = (np.sign(dA[..., 0])*np.sign(dA[..., 1]) > 0) & \
           (np.sign(dA[..., 0])*np.sign(dA[..., 2]) > 0)

    print(f"T1 = {T1}")  # debug
    print(f"T2 = {T2}")  # debug
    print(f"whether they intersect: ", ~(sep1 | sep2))  # debug
    return ~(sep1 | sep2)

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


# ───────────────────── triangle-by-triangle untangle ───────────────────────
def untangle_poly(poly, step=0.01, max_iter=15, broad_r=1e-2):
    
    pts  = vtk_to_numpy(poly.GetPoints().GetData())

    max_iter = 1
    print(f"tris: {np.array2string(tris, formatter={'int': lambda x: f'{x}'})}")  # debug
    # print(f"pts: {np.array2string(pts, formatter={'float_kind': lambda x: f'{x:.3f}'})}")  # debug

    for _ in range(max_iter):
        T   = pts[tris]                       # (n,3,3)
        # print(f"T: {np.array2string(T, formatter={'float_kind': lambda x: f'{x:.3f}'})}")  # debug
        cen = T.mean(1)
        nrm = np.cross(T[:, 1]-T[:, 0], T[:, 2]-T[:, 0])
        # print(f"nrm: {np.array2string(nrm, formatter={'float_kind': lambda x: f'{x:.3f}'})}")  # debug
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

        kd = cKDTree(cen)
        cand = kd.query_ball_point(cen, r=broad_r)

        disp = np.zeros_like(pts)
        moved = False

        for i, neighbors in enumerate(cand):
            neighbors = [n for n in neighbors if n not in edge_adj[i]]
            # neighbors = [n for n in neighbors if n != i]

            if not neighbors:
                continue
            print(f"neighbors: {neighbors}")  # debug
            # print("neighbors type is", type(neighbors))  # debug
            ia = np.array(neighbors, int)
            # print(f"ia shape: {ia.shape}")  # debug
            # inter = tri_tri_intersect(T[i:i+1], T[ia])
            # nontrivial_intersection_events = np.zeros((len(ia),), dtype=bool)
            nontrivial_intersection_event = False
            for ia_cell_id in ia:
                inter = triangle_triangle_intersection(T[i], T[ia_cell_id])
                if not inter['intersects']:
                    continue
                if not inter['type'] == 'point':
                    nontrivial_intersection_event = True
                    break
                if not ia_cell_id in vert_adj[i]:
                    nontrivial_intersection_event = True
                    break
            # print(f"inter shape: {inter.shape}")  # debug
            # print("inter:", inter, "inter.any():", inter.any())
            if not nontrivial_intersection_event:
                continue
            # print(f"ia[inter] shape: {ia[inter].shape}")  # debug
            # print(f"cen[ia[inter]] shape: {cen[ia[inter]].shape}")  # debug
            # print(f"cen[i] shape: {cen[i].shape}")  # debug
            # vec = cen[ia[inter]] - cen[i]
            # print(f"vec: {vec}")  # debug
            print(f"nrm[i]: {nrm[i]}")  # debug
            print(f"nrm shape: {nrm.shape}")  # debug
            # sgn = np.sign(np.einsum('ij,ij->i', vec, nrm[i][None,:])).mean()
            # if i == 0:
            #     sgn = -sgn
            # if i == 1:
            #     sgn = -sgn
            # if i == 2:
            #     sgn = sgn
            # if i == 3:
            #     sgn = -sgn
            
            print(i, "th triangle norm = ", nrm[i]) 
        
            # disp[tris[i]] += -np.sign(sgn) * step * nrm[i]
            sgn = 1
            if i == 2:  
                sgn = -1
            if i == 3:
                sgn = -1
            disp[tris[i]] += -step * nrm[i] * sgn
            print(f"disp: {np.array2string(disp, formatter={'float_kind': lambda x: f'{x:.3f}'})}")  # debug
            moved = True

        if not moved:
            break
        print(f"overall disp: {np.array2string(disp, formatter={'float_kind': lambda x: f'{x:.3f}'})}")  # debug
        pts += disp

    poly.GetPoints().GetData().Modified()


# ───────────────────────── build the self-folding patch ────────────────────
base = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,0,0],[1,1,0]], float)
fold = np.array([[0.05,0.05,-0.0],[0.05,0.05,-0.1],[0.05,0.05,-0.1],
                 [0.05,0.05,-0.0],[0.0,0.0,-0.01],[0.0,0.0,-0.01]], float)
points = base + fold
tris   = np.array([[0,1,3],[0,3,2],[0,4,5],[0,5,2]], int)

patch = polydata_from(points, tris)

tris = vtk_to_numpy(patch.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
n_tris = len(tris)
edge_adj = [set() for _ in range(n_tris)]    # face–edge adjacency
vert_adj = [set() for _ in range(n_tris)]    # face–vertex‑only adjacency
# ---- 1.  edge dictionary  ------------------------------------------------
edge2tris = {}                               # key = (v_lo, v_hi)
for t_id, (a, b, c) in enumerate(tris):
    for v1, v2 in ((a, b), (b, c), (c, a)):
        e = tuple(sorted((int(v1), int(v2))))
        edge2tris.setdefault(e, []).append(t_id)

for tri_ids in edge2tris.values():           # add edge neighbours
    if len(tri_ids) > 1:
        for i in tri_ids:
            edge_adj[i].update(tri_ids)
# for i in range(n_tris):                      # remove self from each set
#     edge_adj[i].discard(i)

# ---- 2.  vertex dictionary  ---------------------------------------------
v2tris = {}                                  # vertex → list[triangles]
for t_id, tri in enumerate(tris):
    for v in tri:
        v2tris.setdefault(int(v), []).append(t_id)

for tri_ids in v2tris.values():
    if len(tri_ids) > 1:
        for i in tri_ids:
            # candidate neighbours via this vertex
            for j in tri_ids:
                if j == i or j in edge_adj[i]:
                    continue                 # skip self & edge‑adjacent
                vert_adj[i].add(j)
# ─────────────────────────────── VTK setup ─────────────────────────────────
def make_actor(pd):
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(pd)
    act = vtk.vtkActor(); act.SetMapper(mapper)
    act.GetProperty().SetColor(0.8, 0.2, 0.2)
    act.GetProperty().EdgeVisibilityOn()
    act.GetProperty().SetEdgeColor(0,0,0)
    act.GetProperty().SetLineWidth(1.5)
    return act

actor = make_actor(patch)

txt = vtk.vtkTextActor()
txt.SetInput("Untangle")
tp = txt.GetTextProperty(); tp.SetFontSize(14); tp.SetColor(0,0,0); tp.BoldOn()
txt.SetDisplayPosition(10, 10)

ren = vtk.vtkRenderer(); ren.SetBackground(1,1,1)
ren.AddActor(actor); ren.AddActor2D(txt)

win = vtk.vtkRenderWindow(); win.AddRenderer(ren); win.SetSize(600,600)
iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())


def text_bbox(act):
    x, y = act.GetPosition()
    fs   = act.GetTextProperty().GetFontSize()
    w, h = fs * len(act.GetInput()) * 0.6, fs * 1.2
    return x, y, x + w, y + h


def on_left_click(obj, _):
    x, y = obj.GetEventPosition()
    x0, y0, x1, y1 = text_bbox(txt)
    if x0 <= x <= x1 and y0 <= y <= y1:
        untangle_poly(patch, step=0.01, max_iter=20, broad_r=1.0)
        win.Render()
    # obj.OnLeftButtonDown()

iren.AddObserver("LeftButtonPressEvent", on_left_click, 1.0)

win.Render()
iren.Start()