"""Data-dict management, material constants, and displacement updates.

Pure NumPy / JAX — no VTK or Qt imports.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def compute_material_constants(mu: float, nu: float) -> tuple[jax.Array, jax.Array]:
    """Return Kelvinlet material parameters (a, b) from shear modulus and Poisson ratio."""
    a = 1 / (4 * jnp.pi * mu)
    b = a / (4 * (1 - nu))
    return a, b


def apply_displacements(data: dict, displacements: np.ndarray, mesh_type: str) -> dict:
    """Add *displacements* to the point array stored under *mesh_type* in *data*."""
    data["points"][mesh_type] += displacements
    return data


def get_centroid(points: jax.Array) -> jax.Array:
    """Return the centroid of *points* as a (1, D) array."""
    assert points.shape[1] in (2, 3)
    centroid = jnp.mean(points, axis=0, keepdims=True)
    assert centroid.shape == (1, points.shape[1])
    return centroid
