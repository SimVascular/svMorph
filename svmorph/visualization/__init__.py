"""VTK-based rendering, interaction, and data bridging."""

from svmorph.visualization.vtk_io import (
    build_parent_tip_map,
    create_data_from_polydata,
    extract_centerline_tangents,
    extract_cross_section_areas,
    extract_inscribed_sphere_radii,
    extract_mesh_arrays,
    get_centerline_point_and_normal,
    read_vtp,
    sync_polydata,
    write_vtp,
)
from svmorph.visualization.renderer import SceneManager
from svmorph.visualization.interactor import MeshInteractor
