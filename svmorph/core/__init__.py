"""Headless-compatible computation core (no VTK, no Qt)."""

from svmorph.core.deformation import (
    capsule_sdf,
    compute_aneurysm_displacements,
    compute_householder_matrices,
    compute_sdf_contact_displacements,
    compute_stenosis_displacements,
    find_stenosis_minimum_radius_representative,
    set_force_center,
    set_node_indices,
    stent_bounding_box,
)
from svmorph.core.geometry import resample_stent_axis
from svmorph.core.mesh_data import (
    apply_displacements,
    compute_material_constants,
    get_centroid,
)
