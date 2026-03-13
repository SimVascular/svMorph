# import os
# import sys
import vtk
# import math
import copy
import time
# import textwrap
import numpy as np
import functools
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v
# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(linewidth=np.inf)
from kelvinlet_core import vtk_utils

import jax as jx
import jax.numpy as jnp

def get_a_b(mu, nu):
    a = 1 / (4 * jnp.pi * mu)
    b = a / (4 * (1 - nu))
    return a, b

def update_polydata_with_points(polydata, data, mesh_type):
    # polypoints = polydata.GetPoints()
    # polypoints.SetData(n2v(data["points"][mesh_type]))
    polydata.GetPoints().Modified()
    # polydata.Modified()
    return polydata

def update_points_with_displacements(data, displacements, mesh_type):
    data["points"][mesh_type] += displacements
    return data

def get_centroid(points):
    assert points.shape[1] in (2, 3)  # Ensure points are 2D or 3D
    # Calculate the centroid by taking the mean across the first axis
    centroid = jnp.mean(points, axis=0, keepdims=True)
    # Ensure centroid shape is correct
    assert centroid.shape == (1, points.shape[1])
    return centroid
