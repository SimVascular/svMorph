# Profiling to check the bottleneck
import cProfile
import pstats
from functools import wraps

def profile_func(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(self)
        profiler.disable()
        
        # Print profiling results
        ps = pstats.Stats(profiler)
        ps.strip_dirs().sort_stats("cumulative").print_stats(10)
        
        return result
    return wrapper

###### Full VTK Integration ######
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QLabel, QHBoxLayout, QFileDialog, QLineEdit
from PyQt6.QtCore import Qt
import vtkmodules.all as vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk_module import VTKHandler

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        self.setWindowTitle("VTK with PyQt6")
        
        self.frame = QWidget()
        self.layout = QVBoxLayout()
        
        # Set the window size to be 800x600 pixels and position it at 100, 100
        self.setGeometry(100, 100, 1200, 900)
        
        # VTK Render Widget
        self.vtk_widget = QVTKRenderWindowInteractor(self.frame)
        self.layout.addWidget(self.vtk_widget)
        
        # Controls
        self.controls_layout = QHBoxLayout()

        # Import Buttons
        self.import_mesh_button = QPushButton("Import Mesh")
        self.import_mesh_button.clicked.connect(self.import_mesh)
        self.controls_layout.addWidget(self.import_mesh_button)
        self.import_centerline_button = QPushButton("Import Centerline")
        self.import_centerline_button.clicked.connect(self.import_centerline)
        self.controls_layout.addWidget(self.import_centerline_button)

        # Slider's label
        self.slider_label = QLabel("Area Percent Change:")
        self.controls_layout.addWidget(self.slider_label)
        # Slider for aneurysm area increase
        self.area_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_slider.setRange(100, 1000)
        self.area_slider.setValue(500.0)
        self.controls_layout.addWidget(self.area_slider)
        # Display the slider value
        self.slider_value = QLineEdit()
        self.slider_value.setText(f"{self.area_slider.value()}%")
        self.slider_value.setFixedWidth(50)
        self.controls_layout.addWidget(self.slider_value)
        self.area_slider.valueChanged.connect(lambda value: self.slider_value.setText(f"{value}%"))
        self.slider_value.textChanged.connect(lambda text: self.area_slider.setValue(float(text.replace("%", ""))))
        # Button for showing selectable nodes on the centerline
        self.show_nodes_button = QPushButton("Select Nodes")
        self.controls_layout.addWidget(self.show_nodes_button)
        # Button for running the deformation
        self.run_button = QPushButton("Create Aneurysm")
        self.controls_layout.addWidget(self.run_button)
        # Add controls layout to main layout
        self.layout.addLayout(self.controls_layout)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        # Connect the button to the run_deformation method
        self.run_button.clicked.connect(self.run_deformation)
        # Connect the button to the show_nodes method
        self.show_nodes_button.clicked.connect(self.display_centerline_nodes)

        # To add a second row of buttons, add another QHBoxLayout and add it to the main layout
        self.controls_layout2 = QHBoxLayout()
        # Add stenosis controls
        self.stenosis_slider_label = QLabel("Stenosis Area % Change:")
        self.controls_layout2.addWidget(self.stenosis_slider_label)
        
        self.stenosis_area_slider = QSlider(Qt.Orientation.Horizontal)
        self.stenosis_area_slider.setRange(1, 100)
        self.stenosis_area_slider.setValue(95)
        self.controls_layout2.addWidget(self.stenosis_area_slider)
        
        self.stenosis_slider_value = QLineEdit()
        self.stenosis_slider_value.setText(f"{self.stenosis_area_slider.value()}")
        self.stenosis_slider_value.setFixedWidth(50)
        self.controls_layout2.addWidget(self.stenosis_slider_value)
        
        self.num_ring_points_label = QLabel("Num Ring Points:")
        self.controls_layout2.addWidget(self.num_ring_points_label)
        
        self.num_ring_points_slider = QSlider(Qt.Orientation.Horizontal)
        self.num_ring_points_slider.setRange(1, 100)
        self.num_ring_points_slider.setValue(35)
        self.controls_layout2.addWidget(self.num_ring_points_slider)
        
        self.num_ring_points_value = QLineEdit()
        self.num_ring_points_value.setText(f"{self.num_ring_points_slider.value()}")
        self.num_ring_points_value.setFixedWidth(50)
        self.controls_layout2.addWidget(self.num_ring_points_value)
        
        self.run_stenosis_button = QPushButton("Create Stenosis")
        self.controls_layout2.addWidget(self.run_stenosis_button)
        self.layout.addLayout(self.controls_layout2)
        # Connect the slider and QLineEdit for stenosis area
        self.stenosis_area_slider.valueChanged.connect(lambda value: self.stenosis_slider_value.setText(f"{value}"))
        self.stenosis_slider_value.textChanged.connect(lambda text: self.stenosis_area_slider.setValue(int(text)))
        # Connect the slider and QLineEdit for num ring points
        self.num_ring_points_slider.valueChanged.connect(lambda value: self.num_ring_points_value.setText(f"{value}"))
        self.num_ring_points_value.textChanged.connect(lambda text: self.num_ring_points_slider.setValue(int(text)))
        # Connect the button to the run_stenosis method
        self.run_stenosis_button.clicked.connect(self.run_stenosis)
        
        # VTK Setup
        self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_handler = None
        # VTKHandler("input/mesh-complete-exterior.vtp", "input/centerline.vtp")

    def import_mesh(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Mesh", "", "VTK Files (*.vtp)")
        if file_name:
            self.mesh_file = file_name
            if hasattr(self, 'centerline_file'):
                self.initialize_vtk_handler()

    def import_centerline(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Centerline", "", "VTK Files (*.vtp)")
        if file_name:
            self.centerline_file = file_name
            if hasattr(self, 'mesh_file'):
                self.initialize_vtk_handler()

    def initialize_vtk_handler(self):
        self.vtk_handler = VTKHandler(self.mesh_file, self.centerline_file)
        self.ren = self.vtk_handler.get_renderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)

        self.style = self.vtk_handler.get_interactor_style(self.vtk_interactor)
        self.vtk_interactor.SetInteractorStyle(self.style)

        # self.vtk_widget.GetRenderWindow().Render()

        self.vtk_interactor.Initialize()
        self.vtk_interactor.Start()

    def run_deformation(self):
        area_percent_change = self.area_slider.value()
        print(f"Running deformation with area percent change: {area_percent_change}")
        self.style.deform_mesh(area_percent_change)
        # self.vtk_handler.get_interactor_style(self.vtk_interactor).deform_mesh()
        # self.vtk_widget.GetRenderWindow().Render()
    
    # @profile_func
    def run_stenosis(self):
        if self.vtk_handler is None:
            print("Please import both mesh and centerline files before running the stenosis.")
            return

        area_percent_change = self.stenosis_area_slider.value()
        num_ring_points = self.num_ring_points_slider.value()
        falloff_type = "regular"
        weight_regularized_laplacian = 1        
        print(f"Running stenosis with area percent change: {area_percent_change}, num ring points: {num_ring_points}")
        self.style.deform_mesh_stenosis(area_percent_change, num_ring_points, falloff_type, weight_regularized_laplacian)
        # self.vtk_widget.GetRenderWindow().Render()
        # self.ren.Render()


    def display_centerline_nodes(self):
        print("Please select three centerline nodes to generate aneurysm.")
        self.style.display_vertices()
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
