"""
=========================================================================

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

"""

# First access the VTK module (and any other needed modules) by importing them.
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersSources import vtkConeSource, vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkBoxWidget
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkPropPicker
)
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader


def load_vtp_file(filename):
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def main(argv):
    colors = vtkNamedColors()

    #
    # Next we create an instance of vtkConeSource and set some of its
    # properties. The instance of vtkConeSource 'cone' is part of a
    # visualization pipeline (it is a source process object) it produces data
    # (output type is vtkPolyData) which other filters may process. Functional?
    #
    # cone = vtkConeSource()
    # cone.SetHeight(3.0)
    # cone.SetRadius(1.0)
    # cone.SetResolution(4)
    # other filters could go here

    #
    # In this example we terminate the pipeline with a mapper process object.
    # (Intermediate filters such as vtkShrinkPolyData could be inserted in
    # between the source and the mapper.)  We create an instance of
    # vtkPolyDataMapper to map the polygonal data into graphics primitives. We
    # connect the output of the cone source to the input of this mapper.
    #
    if len(argv) > 2:
        mesh = load_vtp_file(argv[1])
        coneMapper = vtkPolyDataMapper()
        coneMapper.SetInputData(mesh)
        centerline = load_vtp_file(argv[2])
        centerlineMapper = vtkPolyDataMapper()
        centerlineMapper.SetInputData(centerline)
        
    else:
        cone = vtkSphereSource()
        coneMapper = vtkPolyDataMapper()
        coneMapper.SetInputConnection(cone.GetOutputPort())

    #
    # Create an actor to represent the cone. The actor orchestrates rendering
    # of the mapper's graphics primitives. An actor also refers to properties
    # via a vtkProperty instance, and includes an internal transformation
    # matrix. We set this actor's mapper to be coneMapper which we created
    # above.
    #
    coneActor = vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))
    coneActor.GetProperty().SetOpacity(0.7)
    coneActor.SetPickable(0)

    centerlineActor = vtkActor()
    centerlineActor.SetMapper(centerlineMapper)

    #
    # Create the Renderer and assign actors to it. A renderer is like a
    # viewport. It is part or all of a window on the screen and it is
    # responsible for drawing the actors it has.  We also set the background
    # color here.
    #
    ren1 = vtkRenderer()
    ren1.AddActor(coneActor)
    ren1.AddActor(centerlineActor)
    ren1.SetBackground(0.1, 0.2, 0.3)

    #
    # Finally we create the render window which will show up on the screen.
    # We put our renderer into the render window using AddRenderer. We also
    # set the size to be 300 pixels by 300.
    #
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.SetSize(900, 900)
    renWin.SetWindowName('Interactive Mesh Viewer')

    #
    # The vtkRenderWindowInteractor class watches for events (e.g., keypress,
    # mouse) in the vtkRenderWindow. These events are translated into
    # event invocations that VTK understands (see VTK/Common/vtkCommand.h
    # for all events that VTK processes). Then observers of these VTK
    # events can process them as appropriate.
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Picker
    # picker = vtkmodules.vtkRenderingCore.vtkPointPicker()
    picker = vtkPropPicker()
    iren.SetPicker(picker)

    #
    # By default the vtkRenderWindowInteractor instantiates an instance
    # of vtkInteractorStyle. vtkInteractorStyle translates a set of events
    # it observes into operations on the camera, actors, and/or properties
    # in the vtkRenderWindow associated with the vtkRenderWinodwInteractor.
    # Here we specify a particular interactor style.
    style = MouseInteractorStylePP()
    iren.SetInteractorStyle(style)

    #
    # Here we use a vtkBoxWidget to transform the underlying coneActor (by
    # manipulating its transformation matrix). Many other types of widgets
    # are available for use, see the documentation for more details.
    #
    # The SetInteractor method is how 3D widgets are associated with the render
    # window interactor. Internally, SetInteractor sets up a bunch of callbacks
    # using the Command/Observer mechanism (AddObserver()). The place factor
    # controls the initial size of the widget with respect to the bounding box
    # of the input to the widget.
    boxWidget = vtkBoxWidget()
    boxWidget.SetInteractor(iren)
    boxWidget.SetPlaceFactor(1.25)
    boxWidget.GetOutlineProperty().SetColor(colors.GetColor3d('Gold'))

    # adding seed widget
    # seedWidget = vtkSeedWidget()

    #
    # Place the interactor initially. The input to a 3D widget is used to
    # initially position and scale the widget. The EndInteractionEvent is
    # observed which invokes the SelectPolygons callback.
    #
    boxWidget.SetProp3D(coneActor)
    boxWidget.PlaceWidget()
    callback = vtkMyCallback()
    boxWidget.AddObserver('InteractionEvent', callback)

    #
    # Normally the user presses the 'i' key to bring a 3D widget to life. Here
    # we will manually enable it so it appears with the cone.
    #
    # boxWidget.On()

    #
    # Start the event loop.
    #
    iren.Initialize()
    iren.Start()

# Define custom interaction style class for point picking.
class MouseInteractorStylePP(vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.Points = vtkmodules.vtkCommonCore.vtkPoints()
        self.vertexVisualizationActors = [] # List of actors on the centerline vertices

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'h':
            self.display_vertices()
        self.OnKeyPress()

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        print(f"Picking pixel: {click_pos[0]} {click_pos[1]}")

        picker = self.GetInteractor().GetPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())

        # picked = picker.GetPickPosition()
        pickedActor = picker.GetActor()
        # print(f"Picked value: {picked[0]} {picked[1]} {picked[2]}")
        transform = vtkTransform()
        actor_matrix = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMatrix()
        transform.SetMatrix(actor_matrix)
        # print(f"pickedActor: {pickedActor}")
        if pickedActor in self.vertexVisualizationActors:
            print(f"Picked actor centerpointID: {pickedActor.centerpointID}")
            sphere_center = pickedActor.GetMapper().GetInput().GetCenter()
            print(f"Picked Sphere center: {sphere_center}")
            sphere_center_transformed = transform.TransformPoint(sphere_center)
            self.place_highlight_sphere(sphere_center_transformed)
        
        # Place a new "point"(sphere) at the picked position
        # self.place_highlight_sphere(picked)
        self.OnLeftButtonDown()

    def place_visualization_sphere(self, position, pointID):
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.02)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)

        # Assign the pointID of the center as a custom attribute to the actor
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
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color for visibility
        actor.GetProperty().SetOpacity(0.5)

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)
        # self.Spheres.append(actor)

    def display_vertices(self):
        mesh = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMapper().GetInput()
        points = mesh.GetPoints()

        transform = vtkTransform()
        actor_matrix = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMatrix()
        transform.SetMatrix(actor_matrix)

        for i in range(points.GetData().GetNumberOfTuples()):
            point = [0.0, 0.0, 0.0]
            points.GetPoint(i, point)  # Get the point at index i
            transformed_point = transform.TransformPoint(point)  # Transform the point

            # Place a sphere at the transformed location
            self.place_visualization_sphere(transformed_point, i)
        
        # Force the render window to update
        self.GetInteractor().GetRenderWindow().Render()

class vtkMyCallback(object):
    """
    Callback for the interaction.
    """

    def __call__(self, caller, ev):
        t = vtkTransform()
        widget = caller
        widget.GetTransform(t)
        widget.GetProp3D().SetUserTransform(t)


if __name__ == '__main__':
    import sys

    main(sys.argv)