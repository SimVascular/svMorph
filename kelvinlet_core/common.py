"""Backward-compatibility shim — imports re-exported from new locations.

Pure math lives in svmorph.core.mesh_data; VTK helpers in svmorph.visualization.vtk_io.
"""

from svmorph.core.mesh_data import compute_material_constants as get_a_b
from svmorph.core.mesh_data import apply_displacements as update_points_with_displacements
from svmorph.core.mesh_data import get_centroid
from svmorph.visualization.vtk_io import sync_polydata as update_polydata_with_points
