"""VTK scene setup and lifetime management for svMorph.

Provides :class:`SceneManager`, which loads the surface mesh and
centerline, creates the VTK rendering pipeline (mappers, actors,
renderer), and exposes the :class:`~svmorph.visualization.interactor.MeshInteractor`.
"""

from __future__ import annotations

import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer

from svmorph.visualization import vtk_io
from svmorph.visualization.interactor import MeshInteractor


class SceneManager:
    """Manages the VTK renderer, actors, and interactor for a mesh–centerline pair."""

    def __init__(self, mesh_filename: str, centerline_filename: str) -> None:
        """Load the mesh and centerline VTP files and build the render pipeline.

        Parameters
        ----------
        mesh_filename : str
            Path to the surface-mesh VTP file.
        centerline_filename : str
            Path to the centerline VTP file.
        """
        self.mesh = vtk_io.read_vtp(mesh_filename)
        self.centerline = vtk_io.read_vtp(centerline_filename)
        self.mesh_filename = mesh_filename
        self.centerline_filename = centerline_filename

        self.mesh_mapper = vtkPolyDataMapper()
        self.mesh_mapper.SetInputData(self.mesh)
        self.centerline_mapper = vtkPolyDataMapper()
        self.centerline_mapper.SetInputData(self.centerline)

        self.mesh_actor = vtkActor()
        self.mesh_actor.SetMapper(self.mesh_mapper)
        self.mesh_actor.GetProperty().SetOpacity(0.8)
        self.mesh_actor.SetPickable(0)

        self.centerline_actor = vtkActor()
        self.centerline_actor.SetMapper(self.centerline_mapper)
        self.centerline_actor.SetPickable(0)

        self.renderer = vtkRenderer()
        self.renderer.AddActor(self.mesh_actor)
        self.renderer.AddActor(self.centerline_actor)
        self.renderer.SetBackground(1.0, 1.0, 1.0)

    def get_renderer(self) -> vtkRenderer:
        """Return the VTK renderer containing the mesh and centerline actors."""
        return self.renderer

    def get_interactor_style(self) -> MeshInteractor:
        """Create and return a :class:`MeshInteractor` wired to the loaded data."""
        return MeshInteractor(self.mesh, self.centerline, self.mesh_filename, self.centerline_filename, self.mesh_actor, self.centerline_actor)

    def save_mesh(self, filename: str) -> None:
        """Write the current surface mesh to a VTP file.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        vtk_io.write_vtp(self.mesh, filename)
