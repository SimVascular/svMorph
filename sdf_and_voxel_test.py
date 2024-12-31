import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt6.QtCore import Qt
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkFiltersHybrid import vtkImplicitModeller
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkImagingHybrid import vtkSampleFunction, vtkVoxelModeller
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderWindow, vtkRenderer, vtkVolumeProperty, vtkVolume
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.numpy_support import vtk_to_numpy

class DemoApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SDF and Voxel Demo")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.layout = QVBoxLayout()

        # VTK render widget
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.layout.addWidget(self.vtk_widget)

        # Buttons
        self.import_button = QPushButton("Import .vtp Geometry")
        self.import_button.clicked.connect(self.import_geometry)
        self.layout.addWidget(self.import_button)

        self.sdf_button = QPushButton("Convert to SDF")
        self.sdf_button.clicked.connect(self.convert_to_sdf)
        self.layout.addWidget(self.sdf_button)

        self.voxel_button = QPushButton("Convert to Voxels")
        self.voxel_button.clicked.connect(self.convert_to_voxels)
        self.layout.addWidget(self.voxel_button)

        # Finalize layout
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # VTK setup
        self.vtk_renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.vtk_renderer.SetBackground(0.1, 0.1, 0.1)

        self.polydata = None

    def import_geometry(self):
        """Import a .vtp geometry file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Import .vtp Geometry", "", "VTK Files (*.vtp)")
        if not file_name:
            return

        reader = vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()

        self.polydata = reader.GetOutput()
        self.display_geometry(self.polydata)

    def display_geometry(self, polydata):
        """Display a polydata geometry in the renderer."""
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtkActor()
        actor.SetMapper(mapper)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def display_image_data(self, image_data):
        """Display a vtkImageData object in the renderer."""
        volume_mapper = vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)

        volume_property = vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddVolume(volume)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def convert_to_sdf_v1(self):
        """Convert the polydata to a Signed Distance Field (SDF) and display the isosurface."""
        if not self.polydata:
            return

        # Create SDF using vtkImplicitModeller
        modeller = vtkImplicitModeller()
        modeller.SetInputData(self.polydata)
        modeller.SetSampleDimensions(500, 500, 500)
        modeller.SetMaximumDistance(0.001)
        # modeller.SetModelBounds(self.polydata.GetBounds())

        # Sample the SDF to extract isosurface
        # sampler = vtkSampleFunction()
        # sampler.SetInputConnection(modeller.GetOutputPort())
        # sampler.Update()

        # sdf_polydata = sampler.GetOutput()
        # self.display_image_data(sdf_polydata)

        contour = vtkContourFilter()
        contour.SetInputConnection(modeller.GetOutputPort())
        # contour.SetValue(0, 0.1)
        contour.GenerateValues(3, -0.5, 0.5)
        # contour.UseScalarTreeOn()

        impMapper = vtkPolyDataMapper()
        impMapper.SetInputConnection(contour.GetOutputPort())
        # impMapper.ScalarVisibilityOff()
        impActor = vtkActor()
        impActor.SetMapper(impMapper)
        impActor.GetProperty().SetColor(vtkNamedColors().GetColor3d("Peacock"))
        impActor.GetProperty().SetOpacity(0.5)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(impActor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def convert_to_sdf(self):
        """
        Convert the polydata to a Signed Distance Field (SDF), extract the 0-level set 
        as a new surface, and display it. Also computes a similarity measure between 
        the original and reconstructed surfaces.
        """
        if not self.polydata:
            print("No polydata available. Please import a geometry.")
            return

        # Step 1: Create the SDF using vtkImplicitModeller
        modeller = vtkImplicitModeller()
        modeller.SetInputData(self.polydata)
        modeller.SetSampleDimensions(100, 100, 100)  # High resolution grid
        bounds = self.polydata.GetBounds()
        diagonal = ((bounds[1] - bounds[0])**2 + (bounds[3] - bounds[2])**2 + (bounds[5] - bounds[4])**2) ** 0.5
        modeller.SetMaximumDistance(diagonal * 0.01)  # 1% of the bounding box diagonal
        print("maximun distance: ", diagonal * 0.01)
        # modeller.SetMaximumDistance(0.001)  # High precision
        modeller.SetModelBounds(self.polydata.GetBounds())
        modeller.Update()
        sdf_range = modeller.GetOutput().GetScalarRange()
        print(f"SDF Range: {sdf_range}")
        # contour.SetValue(0, 0.5 * (sdf_range[0] + sdf_range[1]))  # Midpoint of the range

        # Step 2: Extract the 0-level set (isosurface) from the SDF
        contour = vtkContourFilter()
        contour.SetInputConnection(modeller.GetOutputPort())
        contour.SetValue(0, 0.14)  # 0-level set
        contour.Update()

        reconstructed_surface = contour.GetOutput()

        # Step 3: Compute similarity measure
        original_points = vtk_to_numpy(self.polydata.GetPoints().GetData())
        reconstructed_points = vtk_to_numpy(reconstructed_surface.GetPoints().GetData())

        # Compute Hausdorff distance
        from scipy.spatial import cKDTree
        tree_original = cKDTree(original_points)
        tree_reconstructed = cKDTree(reconstructed_points)

        distances_to_original, _ = tree_reconstructed.query(original_points)
        distances_to_reconstructed, _ = tree_original.query(reconstructed_points)

        hausdorff_distance = max(distances_to_original.max(), distances_to_reconstructed.max())
        mean_distance = (distances_to_original.mean() + distances_to_reconstructed.mean()) / 2

        print(f"Hausdorff Distance: {hausdorff_distance:.6f}")
        print(f"Mean Distance: {mean_distance:.6f}")

        # Step 4: Display the reconstructed surface
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(reconstructed_surface)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(vtkNamedColors().GetColor3d("Peacock"))
        actor.GetProperty().SetOpacity(0.6)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def convert_to_voxels(self):
        """Convert the polydata to a voxel representation and display the volume."""
        if not self.polydata:
            return

        # Create SDF for voxel conversion
        modeller = vtkVoxelModeller()
        modeller.SetInputData(self.polydata)
        modeller.SetSampleDimensions(50, 50, 50)
        modeller.SetMaximumDistance(0.5)

        # Sample the SDF into vtkImageData (voxel grid)
        sampler = vtkSampleFunction()
        sampler.SetInputConnection(modeller.GetOutputPort())
        sampler.SetSampleDimensions(50, 50, 50)
        sampler.Update()

        voxel_data = sampler.GetOutput()

        # Visualize voxel data as volume
        volume_mapper = vtkSmartVolumeMapper()
        volume_mapper.SetInputData(voxel_data)

        volume_property = vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddVolume(volume)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemoApp()
    window.show()
    sys.exit(app.exec())