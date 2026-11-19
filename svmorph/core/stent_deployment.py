"""A stent deployment in progress: repeated SDF-contact steps on one mesh.

:func:`svmorph.core.deformation.compute_sdf_contact_displacements` computes a
single step and hands the displacements back for the caller to apply.
:class:`StentDeployment` wraps the loop around it - compute a step, apply it -
which lets the ``warp`` backend keep the mesh on the GPU between steps instead
of copying it to the device and back at every step.  Under the ``jax`` backend
a step of the session is exactly the compute-then-apply sequence it replaces.
"""

from __future__ import annotations

import numpy as np

from svmorph.core import backend, deformation, mesh_data


class StentDeployment:
    """Advance an SDF-contact stent deployment step by step, on the backend in use.

    The session displaces the surface and centerline points of *data* in place.
    Under the ``warp`` backend the points live on the device between steps, and
    ``data`` is only brought up to date by :meth:`sync`: call it before reading
    the points (to save or render them), or use the session as a context
    manager, which syncs on exit.  The session owns the point arrays for its
    lifetime: after modifying them yourself, start a new session.

    Parameters
    ----------
    data : dict
        Simulation data dictionary; ``data["points"]["surface"]`` and
        ``data["points"]["centerline"]`` are displaced in place.
    target_stent_radius : float or array-like
        Target stent radius (scalar, or per-vertex radii of which the maximum
        is used); it bounds the region around the stent that is evaluated.
    influence_radius, contact_distance, f_scale : float or None
        As in :func:`~svmorph.core.deformation.compute_sdf_contact_displacements`;
        ``None`` selects the same unit-scaled defaults.
    cap_height_fraction : float
        Axial semi-axis of the two stent end caps as a fraction of the end
        radius, in ``(0, 1]``.
    """

    def __init__(
        self, data: dict, *, target_stent_radius,
        influence_radius: float | None = None, contact_distance: float | None = None,
        f_scale: float | None = None, cap_height_fraction: float = 1.0,
    ):
        self.data = data
        self.target_stent_radius = (
            float(np.max(target_stent_radius)) if np.ndim(target_stent_radius) > 0 else target_stent_radius)
        self.influence_radius, self.contact_distance, self.f_scale = deformation._sdf_contact_defaults(
            influence_radius, contact_distance, f_scale)
        self.cap_height_fraction = cap_height_fraction
        self.backend = backend.get_backend()

        # The mesh can stay on the device only if its positions can be written back in place
        # afterwards, which takes single-precision arrays such as the point buffers of a VTK mesh.
        # Any other arrays are still computed by Warp, a step at a time.
        self._device_deployment = None
        if self.backend == "warp":
            from svmorph.core import sdf_warp

            surface_points, centerline_points = data["points"]["surface"], data["points"]["centerline"]
            if (sdf_warp.DeviceDeployment.supports(surface_points)
                    and sdf_warp.DeviceDeployment.supports(centerline_points)):
                self._device_deployment = sdf_warp.DeviceDeployment(
                    surface_points, centerline_points, device=backend.warp_device())

    def step_size(self, s: float = -1.0) -> float:
        """Return the stent radius increment of one step at force scale *s*."""
        return self.f_scale * (-s)

    def step(self, stent_vertices, current_stent_radius, s: float = -1.0) -> float:
        """Advance the deployment by one step.

        Parameters
        ----------
        stent_vertices : array-like
            Stent axis vertices, shape ``(V, 3)``.
        current_stent_radius : float or array-like
            Current stent radius: scalar for a uniform stent, or per-vertex
            radii of shape ``(V,)``.
        s : float
            Signed force scale; negative values expand.

        Returns
        -------
        float
            Scalar step size for stent radius increment tracking.
        """
        if self._device_deployment is not None:
            self._device_deployment.step(
                stent_vertices, current_stent_radius,
                **deformation._warp_step_parameters(
                    s, self.target_stent_radius, self.influence_radius, self.contact_distance,
                    self.f_scale, self.cap_height_fraction))
            return self.step_size(s)

        surface_displacements, centerline_displacements, step_size = (
            deformation.compute_sdf_contact_displacements(
                self.data, stent_vertices, s, self.target_stent_radius, current_stent_radius,
                influence_radius=self.influence_radius, contact_distance=self.contact_distance,
                f_scale=self.f_scale, cap_height_fraction=self.cap_height_fraction))
        mesh_data.apply_displacements(self.data, surface_displacements, "surface")
        mesh_data.apply_displacements(self.data, centerline_displacements, "centerline")
        return step_size

    def sync(self) -> None:
        """Bring the point arrays of ``data`` up to date with the steps taken so far."""
        if self._device_deployment is not None:
            self._device_deployment.sync()

    def __enter__(self) -> StentDeployment:
        return self

    def __exit__(self, *exc_info) -> None:
        self.sync()
