# First access the VTK module (and any other needed modules) by importing them.
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkPropPicker
)
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
from kelvinlet_core import scaling  # Import the module containing the deformation function


def load_vtp_file(filename):
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def write_vtp_file(polydata, filename):
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def main(argv):
    colors = vtkNamedColors()

    if len(argv) > 2:
        mesh = load_vtp_file(argv[1])
        coneMapper = vtkPolyDataMapper()
        coneMapper.SetInputData(mesh)
        centerline = load_vtp_file(argv[2])
        centerlineMapper = vtkPolyDataMapper()
        centerlineMapper.SetInputData(centerline)
    else:
        raise ValueError("Please provide both the blood vessel mesh and centerline files as arguments.")

    coneActor = vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))
    coneActor.GetProperty().SetOpacity(0.7)
    coneActor.SetPickable(0)

    centerlineActor = vtkActor()
    centerlineActor.SetMapper(centerlineMapper)

    ren1 = vtkRenderer()
    ren1.AddActor(coneActor)
    ren1.AddActor(centerlineActor)
    ren1.SetBackground(0.1, 0.2, 0.3)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.SetSize(900, 900)
    renWin.SetWindowName('Interactive Mesh Viewer')

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    picker = vtkPropPicker()
    iren.SetPicker(picker)

    style = MouseInteractorStylePP(mesh, centerline, iren)
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()


class MouseInteractorStylePP(vtkInteractorStyleTrackballCamera):
    def __init__(self, mesh, centerline, interactor, parent=None):
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.Points = vtkmodules.vtkCommonCore.vtkPoints()
        self.vertexVisualizationActors = []
        self.mesh = mesh
        self.centerline = centerline
        # self.interactor = interactor
        self.selected_points = []

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'h':
            self.display_vertices()
        elif key == 'd':
            self.deform_mesh()
        self.OnKeyPress()

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        picker = self.GetInteractor().GetPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
        pickedActor = picker.GetActor()

        transform = vtkTransform()
        actor_matrix = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMatrix()
        transform.SetMatrix(actor_matrix)

        if pickedActor in self.vertexVisualizationActors:
            print(f"Picked actor centerpointID: {pickedActor.centerpointID}")
            pointID = pickedActor.centerpointID
            self.selected_points.append(pointID)
            sphere_center = pickedActor.GetMapper().GetInput().GetCenter()
            sphere_center_transformed = transform.TransformPoint(sphere_center)
            self.place_highlight_sphere(sphere_center_transformed)

        self.OnLeftButtonDown()

    def place_visualization_sphere(self, position, pointID):
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.03)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.centerpointID = pointID

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)
        self.vertexVisualizationActors.append(actor)

    def place_highlight_sphere(self, position):
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.05)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        actor.GetProperty().SetOpacity(0.5)

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)

    def display_vertices(self):
        points = self.centerline.GetPoints()
        transform = vtkTransform()
        actor_matrix = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMatrix()
        transform.SetMatrix(actor_matrix)

        for i in range(points.GetData().GetNumberOfTuples()):
            point = [0.0, 0.0, 0.0]
            points.GetPoint(i, point)
            transformed_point = transform.TransformPoint(point)
            self.place_visualization_sphere(transformed_point, i)

        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh(self):
        if len(self.selected_points) < 3:
            print("Please select at least 3 points along the centerline.")
            return

        create_aneurysm(self.mesh, self.centerline, self.selected_points)
        self.update_mesh_viewer()

    def update_mesh_viewer(self):
        updated_mesh = load_vtp_file("obtained_aneurysm_surface_aneurysm_constant_uniscale_25.vtp")
        updated_centerline = load_vtp_file("obtained_aneurysm_centerline_aneurysm_constant_uniscale_25.vtp")

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.RemoveAllViewProps()

        mesh_mapper = vtkPolyDataMapper()
        mesh_mapper.SetInputData(updated_mesh)
        mesh_actor = vtkActor()
        mesh_actor.SetMapper(mesh_mapper)
        mesh_actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('MistyRose'))
        mesh_actor.GetProperty().SetOpacity(0.7)
        mesh_actor.SetPickable(0)

        centerline_mapper = vtkPolyDataMapper()
        centerline_mapper.SetInputData(updated_centerline)
        centerline_actor = vtkActor()
        centerline_actor.SetMapper(centerline_mapper)

        ren.AddActor(mesh_actor)
        ren.AddActor(centerline_actor)

        iren = self.GetInteractor()
        self.mesh = updated_mesh
        self.centerline = updated_centerline
        self.selected_points = []
        # iren.SetInteractorStyle(style)

        ren.GetRenderWindow().Render()


def create_aneurysm(mesh, centerline, selected_points, model="test_aneurysm"):
    centerline_polydata_input_file_name = "centerline.vtp"
    surface_polydata_input_file_name = "mesh-complete-exterior.vtp"
    centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
    surface_polydata_output_file_name = "obtained_aneurysm_surface"

    list_of_other_geometry_polydata_input_file_names = []
    list_of_other_geometry_polydata_output_file_names = []

    force_center_point_id = selected_points[1]
    list_of_node_point_indices = selected_points
    area_percent_change = 500
    phi_type = "constant"
    affine_params = {"eps": {model: 1.0}, "scale": {model: 1.1}}

    mu = 1
    nu = 0.4
    num_time_steps = 25

    write_vtp_file(centerline, centerline_polydata_input_file_name)
    write_vtp_file(mesh, surface_polydata_input_file_name)

    scaling.run_aneurysm(
        affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name,
        centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, area_percent_change, num_time_steps, list_of_node_point_indices, 
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names
    )

if __name__ == '__main__':
    import sys

    main(sys.argv)