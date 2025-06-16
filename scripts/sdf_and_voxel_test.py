import sys
import math
import numpy as np
import jax.numpy as jnp
import jax
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt6.QtCore import Qt

from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkFiltersHybrid import vtkImplicitModeller
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkMarchingCubes
from vtkmodules.vtkImagingHybrid import vtkSampleFunction, vtkVoxelModeller
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderWindow, vtkRenderer, vtkVolumeProperty, vtkVolume
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type
from vtk.util.numpy_support import vtk_to_numpy

def smin(a, b, k=0.01):
    k = k * 4.0
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    return jnp.minimum(a, b) - h**2 * k / 4.0 

def capsule_sdf(x, y, z, a, b, r):
    """
    Compute the signed distance from points (x,y,z) to a capsule defined by the line segment from a to b with radius r.
    x, y, z can be numpy arrays (from a meshgrid).
    """
    ax, ay, az = a
    bx, by, bz = b
    print("ax, ay, az: ", ax, ay, az)
    print("bx, by, bz: ", bx, by, bz)
    # Vector from a to b:
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    d2 = dx*dx + dy*dy + dz*dz
    # Vector from a to each grid point:
    px = x - ax
    py = y - ay
    pz = z - az
    # Projection parameter t (clamped to [0,1]):
    t = (px*dx + py*dy + pz*dz) / (d2 if d2 != 0 else 1)
    t = np.clip(t, 0.0, 1.0)
    # Closest point on the segment:
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    proj_z = az + t * dz
    # Euclidean distance from grid points to the projection:
    dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2 + (z - proj_z)**2)
    # print("dist: ", dist)
    return dist - r

def capsule_sdf_fast(p, a, b, r):
    ba = b - a
    pa = p - a
    ba_dot_pa = jnp.sum(ba * pa, axis=1)
    ba_dot_ba = jnp.dot(ba, ba)
    h = jnp.clip(ba_dot_pa / ba_dot_ba, 0, 1) # assume b neq a
    axis_to_point = pa - h[:, None] * ba
    dist = jnp.linalg.norm(axis_to_point, axis=1)[..., None]
    # direction = axis_to_point / (dist + 1e-6)
    dist_to_surface = dist - r
    return dist_to_surface

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

        # New button: Visualize Demo Capsule SDF
        self.demo_sdf_button = QPushButton("Visualize Demo Capsule SDF")
        self.demo_sdf_button.clicked.connect(self.visualize_demo_sdf)
        self.layout.addWidget(self.demo_sdf_button)

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

        modeller = vtkImplicitModeller()
        modeller.SetInputData(self.polydata)
        modeller.SetSampleDimensions(500, 500, 500)
        modeller.SetMaximumDistance(0.001)

        contour = vtkContourFilter()
        contour.SetInputConnection(modeller.GetOutputPort())
        contour.GenerateValues(3, -0.5, 0.5)

        impMapper = vtkPolyDataMapper()
        impMapper.SetInputConnection(contour.GetOutputPort())
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

        modeller = vtkImplicitModeller()
        modeller.SetInputData(self.polydata)
        modeller.SetSampleDimensions(100, 100, 100)
        bounds = self.polydata.GetBounds()
        diagonal = math.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
        modeller.SetMaximumDistance(diagonal * 0.01)
        print("maximum distance: ", diagonal * 0.01)
        modeller.SetModelBounds(self.polydata.GetBounds())
        modeller.Update()
        sdf_range = modeller.GetOutput().GetScalarRange()
        print(f"SDF Range: {sdf_range}")

        contour = vtkContourFilter()
        contour.SetInputConnection(modeller.GetOutputPort())
        contour.SetValue(0, 0.14)
        contour.Update()

        reconstructed_surface = contour.GetOutput()

        original_points = vtk_to_numpy(self.polydata.GetPoints().GetData())
        reconstructed_points = vtk_to_numpy(reconstructed_surface.GetPoints().GetData())

        from scipy.spatial import cKDTree
        tree_original = cKDTree(original_points)
        tree_reconstructed = cKDTree(reconstructed_points)

        distances_to_original, _ = tree_reconstructed.query(original_points)
        distances_to_reconstructed, _ = tree_original.query(reconstructed_points)

        hausdorff_distance = max(distances_to_original.max(), distances_to_reconstructed.max())
        mean_distance = (distances_to_original.mean() + distances_to_reconstructed.mean()) / 2

        print(f"Hausdorff Distance: {hausdorff_distance:.6f}")
        print(f"Mean Distance: {mean_distance:.6f}")

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

        modeller = vtkVoxelModeller()
        modeller.SetInputData(self.polydata)
        modeller.SetSampleDimensions(50, 50, 50)
        modeller.SetMaximumDistance(0.5)

        sampler = vtkSampleFunction()
        sampler.SetInputConnection(modeller.GetOutputPort())
        sampler.SetSampleDimensions(50, 50, 50)
        sampler.Update()

        voxel_data = sampler.GetOutput()

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

    def visualize_demo_sdf(self):
        """
        Manually sample the capsule SDF on a 3D voxel grid, convert it into a vtkImageData,
        then use vtkMarchingCubes to extract the 0-level isosurface and render it.
        """
        # Define the capsule parameters.
        a = (-0.5, 0.0, 0.0)  # segment start
        b = (0.0, 0.0, 0.0)   # segment end
        r = 0.2               # capsule radius

        # Define the sampling grid.
        nx, ny, nz = 100, 100, 100
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = -1.0, 1.0

        # Create coordinate arrays.
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        # print("X shape: ", X)
        # print("Y shape: ", Y)
        # print("Z shape: ", Z)
        p = np.stack((X.ravel(order='F'), Y.ravel(order='F'), Z.ravel(order='F')), axis=1)
        # print("p: ", p)

        # Compute the SDF values over the grid.
        # sdf = capsule_sdf(X, Y, Z, a, b, r)
        a = jnp.array(a)
        b = jnp.array(b)
        # print("a: ", a)
        # print("b: ", b)
        c = jnp.array([0.5, 0.2, 0.0])
        d = jnp.array([0.6, 0.5, 0.0])
        sdf_ab = capsule_sdf_fast(p, a, b, r)
        sdf_bc = capsule_sdf_fast(p, b, c, r)
        sdf_cd = capsule_sdf_fast(p, c, d, r)
        # sdf = jnp.minimum(sdf_ab, sdf_bc)
        sdf = jax.vmap(smin)(sdf_ab, sdf_bc)
        sdf = jax.vmap(smin)(sdf, sdf_cd)

        # Create a vtkImageData and populate it with the SDF values.
        imageData = vtkImageData()
        imageData.SetDimensions(nx, ny, nz)
        spacing = ((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), (zmax - zmin) / (nz - 1))
        imageData.SetSpacing(spacing)
        imageData.SetOrigin(xmin, ymin, zmin)

        # Flatten the sdf array in Fortran order (VTK expects Fortran order)
        sdf_flat = sdf.ravel(order='F')
        vtk_sdf = numpy_to_vtk(sdf_flat, deep=True, array_type=get_vtk_array_type(np.float32))
        vtk_sdf.SetName("SDF")
        imageData.GetPointData().SetScalars(vtk_sdf)

        # Use vtkMarchingCubes to extract the 0-level isosurface.
        mc = vtkMarchingCubes()
        mc.SetInputData(imageData)
        mc.SetValue(0, 0.0)
        mc.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(vtkNamedColors().GetColor3d("Tomato"))

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemoApp()
    window.show()
    sys.exit(app.exec())