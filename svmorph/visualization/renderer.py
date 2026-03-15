from __future__ import annotations

import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer

from svmorph.visualization import vtk_io
from svmorph.visualization.interactor import MeshInteractor


class SceneManager:
    def __init__(self, mesh_filename: str, centerline_filename: str) -> None:
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
        return self.renderer

    def get_interactor_style(self) -> MeshInteractor:
        return MeshInteractor(self.mesh, self.centerline, self.mesh_filename, self.centerline_filename, self.mesh_actor, self.centerline_actor)

    def save_mesh(self, filename: str) -> None:
        vtk_io.write_vtp(self.mesh, filename)
