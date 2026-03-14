import copy

import jax.numpy as jnp
from vtk.util.numpy_support import vtk_to_numpy as v2n


def create_data_from_polydata(centerline_polydata, surface_polydata, other_geometry_polydatas):
    # Convert to JAX-compatible arrays by using jnp.array
    centerline_points = jnp.array(copy.deepcopy(v2n(centerline_polydata.GetPoints().GetData())))
    surface_points = jnp.array(copy.deepcopy(v2n(surface_polydata.GetPoints().GetData())))
    # Check if points have the required shape
    assert centerline_points.shape[1] == 3  # Ensure (x, y, z) coordinates
    assert surface_points.shape[1] == 3

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
        assert data["centerline_coordinate"].shape[0] == num_centerline_points
    else:
        print("We sys exited because there was no centerline coords")
        # sys.exit("'centerline_coordinate' is not a point array on the centerline polydata.")
    # Process and add other geometry points as JAX arrays
    for ig, polydata in enumerate(other_geometry_polydatas):
        other_geometry_points = jnp.array(copy.deepcopy(v2n(polydata.GetPoints().GetData())))
        assert other_geometry_points.shape[1] == 3
        data["points"][f"other_geometry_{ig}"] = other_geometry_points
    return data
