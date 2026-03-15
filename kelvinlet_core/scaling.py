"""Backward-compatibility shim — all code has moved to svmorph.core.deformation
and svmorph.visualization.vtk_io.  This file re-exports the old names so that
existing callers continue to work until imports are fully migrated."""

from svmorph.core.deformation import (
    compute_householder_matrices,
    set_node_indices as define_nodes_affine,
    set_force_center as assign_force_location_affine_v2,
    interface_falloff,
    mix,
    smin_and_gradient,
    fold_smin,
    compute_min_dist_and_direction,
    kelvinlets_truncated_spherical_contraction as kelvinlets_truncated_sphere_warp_shrink,
    kelvinlets_truncated_spherical_expansion as kelvinlets_truncated_sphere_warp_sculp,
    smin_sdf_capsule_contact_sculp,
    get_scaling_kelvinlet_displacements_inner as get_affine_laplacian_displacements_inner,
    find_stenosis_minimum_radius_representative,
    get_stenosis_displacements_inner,
    stent_bounding_box,
    compute_aneurysm_displacements as get_displacements,
    compute_stenosis_displacements as get_stenosis_displacements,
    compute_sdf_contact_displacements as get_sdf_contact_surface_and_centerline_displacements,
)

from svmorph.visualization.vtk_io import (
    create_data_from_polydata as define_points_affine,
)
