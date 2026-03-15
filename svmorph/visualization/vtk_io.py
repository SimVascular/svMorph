"""VTK polydata I/O and geometry extraction utilities.

Provides functions for reading/writing VTP files, extracting point arrays
(tangents, cross-section areas, inscribed-sphere radii) from centerline
polydata, slicing meshes with planes, and synchronising NumPy-backed
simulation data with VTK polydata objects.
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict

import vtk
import numpy as np
import jax
import jax.numpy as jnp
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

from svmorph.logging import get_logger

logger = get_logger(__name__)


def read_vtp(filename: str) -> vtk.vtkPolyData:
    """Read a VTK XML PolyData (.vtp) file and return the polydata.

    Parameters
    ----------
    filename : str
        Path to the ``.vtp`` file.

    Returns
    -------
    vtk.vtkPolyData
        Loaded polydata.
    """
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def write_vtp(polydata: vtk.vtkPolyData, filename: str) -> None:
    """Write a VTK polydata object to a ``.vtp`` file.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Polydata to write.
    filename : str
        Output file path.
    """
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def extract_mesh_arrays(
    surface_polydata: vtk.vtkPolyData, centerline_polydata: vtk.vtkPolyData,
) -> dict:
    """Extract surface and centerline point arrays into a simulation data dict.

    The returned dictionary contains NumPy views into the VTK point buffers
    (mutated in-place during deformation) as well as initial JAX copies.

    Parameters
    ----------
    surface_polydata : vtk.vtkPolyData
        Surface mesh polydata.
    centerline_polydata : vtk.vtkPolyData
        Centerline polydata.

    Returns
    -------
    dict
        Simulation data dictionary with ``"points"`` and ``"nodes"`` keys.
    """
    surface_points_view_np = v2n(surface_polydata.GetPoints().GetData())
    centerline_points_view_np = v2n(centerline_polydata.GetPoints().GetData())
    surface_points_jnp = jnp.array(surface_points_view_np)
    centerline_points_jnp = jnp.array(centerline_points_view_np)
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


def build_parent_tip_map(
    centerline_polydata: vtk.vtkPolyData,
) -> tuple[dict[int, int], np.ndarray]:
    """
    Build a mapping  pointId -> maxPointId_of_closest_parent_segment
    for a VTK centre-line tree whose segments are encoded by a
    {0,1}-flag array (one component per leaf branch).
    Returns dict { pointId (int) : parent_tip_pointId (int) }.
    """
    vtk_arr = centerline_polydata.GetPointData().GetArray("CenterlineId")
    if vtk_arr is None:
        raise ValueError("Point array 'CenterlineId' not found.")
    centerline_ids = v2n(vtk_arr)            # (N, n_components)
    unique_rows, inverse = np.unique(centerline_ids, axis=0, return_inverse=True)
    if unique_rows.shape[0] == 1:
        return {pointId: -1 for pointId in range(centerline_ids.shape[0])}, np.zeros(centerline_ids.shape[0], dtype=bool)
    segment_points = defaultdict(list)          # seg_id -> [pt_id, ...]
    for pointId, seg_id in enumerate(inverse):
        segment_points[seg_id].append(pointId)
    seg_tip = {seg_id: max(points) for seg_id, points in segment_points.items()}
    segment_base_mask = np.zeros(centerline_ids.shape[0], dtype=bool)
    for seg_id, points in segment_points.items():
        segment_base_mask[min(points)] = True
    seg_bitcount = unique_rows.sum(axis=1)      # (n_segments,)
    parent_tip_for_segment = {}   # seg_id -> parent_tip_point_id
    for child_id, child_mask in enumerate(unique_rows):
        mask_ok = np.logical_or(~child_mask.astype(bool), unique_rows.astype(bool))
        is_superset = mask_ok.all(axis=1)
        is_superset[child_id] = False
        if not np.any(is_superset):
            parent_tip_for_segment[child_id] = seg_tip[child_id]
            continue
        candidate_ids = np.nonzero(is_superset)[0]
        extra_bits = seg_bitcount[candidate_ids] - seg_bitcount[child_id]
        best_parent_idx = candidate_ids[np.argmin(extra_bits)]
        parent_tip_for_segment[child_id] = seg_tip[best_parent_idx]

    point_to_parent_tip = {}
    for pt_id, seg_id in enumerate(inverse):
        point_to_parent_tip[pt_id] = parent_tip_for_segment[seg_id]
    return point_to_parent_tip, segment_base_mask


def extract_centerline_tangents(centerline_polydata: vtk.vtkPolyData) -> np.ndarray:
    """Compute unit tangent vectors at every centerline point via central differences.

    Parameters
    ----------
    centerline_polydata : vtk.vtkPolyData
        Centerline polydata.

    Returns
    -------
    np.ndarray
        Unit tangent vectors, shape ``(N, 3)``.
    """
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


def extract_cross_section_areas(centerline_polydata: vtk.vtkPolyData) -> np.ndarray:
    """Extract the ``CenterlineSectionArea`` point array from the centerline.

    Parameters
    ----------
    centerline_polydata : vtk.vtkPolyData
        Centerline polydata with a ``CenterlineSectionArea`` point array.

    Returns
    -------
    np.ndarray
        Cross-section areas, shape ``(N,)``.  Zeros if the array is absent.
    """
    vtk_array = centerline_polydata.GetPointData().GetArray("CenterlineSectionArea")
    areas = v2n(vtk_array) if vtk_array is not None else np.zeros(centerline_polydata.GetNumberOfPoints())
    return areas


def extract_inscribed_sphere_radii(centerline_polydata: vtk.vtkPolyData) -> np.ndarray:
    """Extract the ``MaximumInscribedSphereRadius`` point array from the centerline.

    Parameters
    ----------
    centerline_polydata : vtk.vtkPolyData
        Centerline polydata with a ``MaximumInscribedSphereRadius`` point array.

    Returns
    -------
    np.ndarray
        Inscribed-sphere radii, shape ``(N,)``.  Zeros if the array is absent.
    """
    vtk_array = centerline_polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    radii = v2n(vtk_array) if vtk_array is not None else np.zeros(centerline_polydata.GetNumberOfPoints())
    return radii


def get_centerline_point_and_normal(
    centerline_polydata: vtk.vtkPolyData, point_id: int,
) -> tuple[jax.Array, jax.Array]:
    """Return the 3-D position and unit tangent at a centerline point.

    Parameters
    ----------
    centerline_polydata : vtk.vtkPolyData
        Centerline polydata.
    point_id : int
        Index of the centerline point.

    Returns
    -------
    point : jax.Array
        Coordinates of the point, shape ``(3,)``.
    normal : jax.Array
        Unit tangent vector at the point, shape ``(3,)``.
    """
    point = jnp.array(centerline_polydata.GetPoint(point_id))  # Directly convert to JAX array
    num_centerline_points = centerline_polydata.GetNumberOfPoints()
    if 0 < point_id < num_centerline_points - 1:
        next_point = jnp.array(centerline_polydata.GetPoint(point_id + 1))
        prev_point = jnp.array(centerline_polydata.GetPoint(point_id - 1))
        normal = next_point - prev_point
    elif point_id == num_centerline_points - 1:
        prev_point = jnp.array(centerline_polydata.GetPoint(point_id - 1))
        normal = point - prev_point
    else:
        next_point = jnp.array(centerline_polydata.GetPoint(point_id + 1))
        normal = next_point - point
    normal /= jnp.linalg.norm(normal)
    return point, normal


def get_cross_sectional_area_of_triangulated_slice(triangulated_slice: vtk.vtkPolyData) -> float:
    """Compute the area of a triangulated planar slice using ``vtkIntegrateAttributes``.

    Parameters
    ----------
    triangulated_slice : vtk.vtkPolyData
        Triangulated 2-D slice polydata.

    Returns
    -------
    float
        Surface area of the slice.

    Raises
    ------
    Exception
        If *triangulated_slice* is empty.
    """
    if not triangulated_slice.GetNumberOfPoints():
        raise Exception('Empty slice')
    integrator = vtk.vtkIntegrateAttributes()
    integrator.SetInputData(triangulated_slice)
    integrator.Update()
    surface_area = integrator.GetOutput().GetCellData().GetArray('Area').GetValue(0)
    return surface_area


def cut_polydata(
    polydata: vtk.vtkPolyData, origin: np.ndarray, normal: np.ndarray,
) -> vtk.vtkPolyData:
    """Cut a polydata with a plane and return the resulting contour.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Input polydata.
    origin : np.ndarray
        Plane origin, shape ``(3,)``.
    normal : np.ndarray
        Plane normal, shape ``(3,)``.

    Returns
    -------
    vtk.vtkPolyData
        Cut contour polydata.
    """
    cutting_plane = vtk.vtkPlane()
    cutting_plane.SetOrigin(origin[0], origin[1], origin[2])
    cutting_plane.SetNormal(normal[0], normal[1], normal[2])
    cut = vtk.vtkCutter()
    cut.SetInputData(polydata)
    cut.SetCutFunction(cutting_plane)
    cut.Update()
    return cut.GetOutput()


def connectivity(
    polydata: vtk.vtkPolyData, origin: np.ndarray,
) -> vtk.vtkConnectivityFilter:
    """Extract the connected region of *polydata* closest to *origin*.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Input polydata (typically a cut contour).
    origin : np.ndarray
        Reference point, shape ``(3,)``.

    Returns
    -------
    vtk.vtkConnectivityFilter
        Connectivity filter whose output is the closest region.
    """
    connection = vtk.vtkConnectivityFilter()
    connection.SetInputData(polydata)
    connection.SetExtractionModeToClosestPointRegion()
    connection.SetClosestPoint(origin[0], origin[1], origin[2])
    connection.Update()
    return connection


def slice_polydata(
    surface_polydata: vtk.vtkPolyData, origin: np.ndarray, normal: np.ndarray,
) -> vtk.vtkPolyData:
    """Cut a surface with a plane and return the closest connected contour.

    Parameters
    ----------
    surface_polydata : vtk.vtkPolyData
        Surface mesh polydata.
    origin : np.ndarray
        Plane origin, shape ``(3,)``.
    normal : np.ndarray
        Plane normal, shape ``(3,)``.

    Returns
    -------
    vtk.vtkPolyData
        Single-region contour polydata.
    """
    cut = cut_polydata(surface_polydata, origin, normal)
    contour = connectivity(cut, origin)
    return contour.GetOutput()


def get_triangulated_slice(
    surface_polydata: vtk.vtkPolyData, origin: np.ndarray, normal: np.ndarray,
) -> vtk.vtkPolyData:
    """Slice a surface with a plane and triangulate the resulting contour.

    Parameters
    ----------
    surface_polydata : vtk.vtkPolyData
        Surface mesh polydata.
    origin : np.ndarray
        Plane origin, shape ``(3,)``.
    normal : np.ndarray
        Plane normal, shape ``(3,)``.

    Returns
    -------
    vtk.vtkPolyData
        Triangulated 2-D slice polydata.
    """
    triangulated_slice = vtk.vtkDelaunay2D()
    triangulated_slice.SetTolerance(1e-4)
    triangulated_slice.SetInputData(slice_polydata(surface_polydata, origin, normal))
    triangulated_slice.Update()
    return triangulated_slice.GetOutput()


def get_cross_sectional_area(
    surface_polydata: vtk.vtkPolyData, origin: np.ndarray, normal: np.ndarray,
) -> float:
    """Compute the cross-sectional area of a surface sliced by a plane.

    Parameters
    ----------
    surface_polydata : vtk.vtkPolyData
        Surface mesh polydata.
    origin : np.ndarray
        Plane origin, shape ``(3,)``.
    normal : np.ndarray
        Plane normal, shape ``(3,)``.

    Returns
    -------
    float
        Cross-sectional area.
    """
    start_time = time.time()
    triangulated_slice = get_triangulated_slice(surface_polydata, origin, normal)
    mid_time = time.time()
    area = get_cross_sectional_area_of_triangulated_slice(triangulated_slice)
    end_time = time.time()
    logger.timing(f"get_triangulated_slice: {mid_time - start_time:.6f} s")
    logger.timing(f"get_cross_sectional_area_of_triangulated_slice: {end_time - mid_time:.6f} s")
    return area


def create_data_from_polydata(
    centerline_polydata: vtk.vtkPolyData,
    surface_polydata: vtk.vtkPolyData,
    other_geometry_polydatas: list[vtk.vtkPolyData],
) -> dict:
    """Build a simulation data dictionary from VTK polydata objects.

    Deep-copies point arrays into JAX arrays and optionally includes
    ``centerline_coordinate`` and additional geometry arrays.

    Parameters
    ----------
    centerline_polydata : vtk.vtkPolyData
        Centerline polydata.
    surface_polydata : vtk.vtkPolyData
        Surface mesh polydata.
    other_geometry_polydatas : list[vtk.vtkPolyData]
        Additional geometry polydata objects.

    Returns
    -------
    dict
        Simulation data dictionary.
    """
    centerline_points = jnp.array(copy.deepcopy(v2n(centerline_polydata.GetPoints().GetData())))
    surface_points = jnp.array(copy.deepcopy(v2n(surface_polydata.GetPoints().GetData())))

    # Create a dictionary to store data, including the JAX arrays
    data = {
        "points": {
            "centerline": centerline_points,
            "surface": surface_points
        },
        "nodes": {
            "all_indices": [],
            "force_center_point_id": -1
        },
        "centerline_coordinate": jnp.array([])
    }
    # Check for the "centerline_coordinate" array and convert if available
    if centerline_polydata.GetPointData().HasArray("centerline_coordinate"):
        num_centerline_points = data["points"]["centerline"].shape[0]
        data["centerline_coordinate"] = jnp.array(copy.deepcopy(
            v2n(centerline_polydata.GetPointData().GetArray("centerline_coordinate"))
        ))
    else:
        logger.warning("No centerline_coordinate array found on centerline polydata")
    # Process and add other geometry points as JAX arrays
    for geom_idx, polydata in enumerate(other_geometry_polydatas):
        other_geometry_points = jnp.array(copy.deepcopy(v2n(polydata.GetPoints().GetData())))
        data["points"][f"other_geometry_{geom_idx}"] = other_geometry_points
    return data


def sync_polydata(polydata: vtk.vtkPolyData, data: dict, mesh_type: str) -> vtk.vtkPolyData:
    """Mark the VTK point buffer as modified so the render pipeline picks up changes."""
    polydata.GetPoints().Modified()
    return polydata
