import vtk
# from vtkmodules.vtkRenderingOpenGL2 import vtkRenderingOpenGL2
# from vtkmodules.vtkCommonColor import vtkNamedColors
# from vtkmodules.vtkCommonTransforms import vtkTransform
# from vtkmodules.vtkFiltersSources import vtkConeSource, vtkSphereSource
# from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
# from vtkmodules.vtkInteractionWidgets import vtkBoxWidget
# from vtkmodules.vtkRenderingCore import (
#     vtkActor,
#     vtkPolyDataMapper,
#     vtkRenderWindow,
#     vtkRenderWindowInteractor,
#     vtkRenderer,
#     vtkPropPicker
# )
# from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        super().__init__()
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.selected_vertex_id = None
        self.mesh_actor = None
        self.picker = vtk.vtkPropPicker()

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
        picked_actor = self.picker.GetActor()
        
        if picked_actor and picked_actor == self.mesh_actor:
            picked_position = self.picker.GetPickPosition()
            points = self.mesh_actor.GetMapper().GetInput().GetPoints()
            min_distance = float('inf')
            for i in range(points.GetNumberOfPoints()):
                pt = points.GetPoint(i)
                dist = vtk.vtkMath.Distance2BetweenPoints(pt, picked_position)
                if dist < min_distance:
                    min_distance = dist
                    self.selected_vertex_id = i
            print(f"Selected vertex ID: {self.selected_vertex_id}")

        self.OnLeftButtonDown()

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'u' and self.selected_vertex_id is not None:
            self.move_selected_vertex([0, 10, 0])  # Move up by 10 units in y-axis
        self.OnKeyPress()

    def move_selected_vertex(self, vector):
        if self.mesh_actor and self.selected_vertex_id is not None:
            points = self.mesh_actor.GetMapper().GetInput().GetPoints()
            point = list(points.GetPoint(self.selected_vertex_id))
            point[0] += vector[0]  # x-component
            point[1] += vector[1]  # y-component
            point[2] += vector[2]  # z-component
            points.SetPoint(self.selected_vertex_id, point)
            points.Modified()  # Notify VTK that the points have been changed
            self.GetInteractor().GetRenderWindow().Render()

def main():
    # Renderer setup
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 600)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Create a cone
    cone_source = vtk.vtkConeSource()
    cone_source.SetHeight(20)
    cone_source.SetRadius(5)
    cone_source.SetResolution(20)
    cone_source.Update()

    # Mapper
    cone_mapper = vtk.vtkPolyDataMapper()
    cone_mapper.SetInputConnection(cone_source.GetOutputPort())

    # Actor
    cone_actor = vtk.vtkActor()
    cone_actor.SetMapper(cone_mapper)

    # Add actor to the renderer
    renderer.AddActor(cone_actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Background color dark blue

    # Custom interactor style
    customStyle = CustomInteractorStyle()
    customStyle.mesh_actor = cone_actor
    renderWindowInteractor.SetInteractorStyle(customStyle)

    # Initialize and start
    renderWindow.Render()
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()

if __name__ == "__main__":
    main()