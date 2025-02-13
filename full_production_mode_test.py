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
from PyQt6.QtCore import QTimer
import math

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

        # Timer for continuous deformation and animated deformation
        self.timer = QTimer(self)
        self.stent_edge_timer = QTimer(self)
        self.animation_timer = QTimer(self)
        
        # Controls
        self.controls_layout = QHBoxLayout()
        '''
        # Slider's label
        self.slider_label = QLabel("Force Scale:")
        self.controls_layout.addWidget(self.slider_label)
        # Slider for aneurysm area increase
        self.area_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_slider.setRange(-1000, 1000)  # Use a larger range to simulate float values
        self.area_slider.setValue(20)
        self.controls_layout.addWidget(self.area_slider)
        # Display the slider value
        self.slider_value = QLineEdit()
        self.slider_value.setText(f"{self.area_slider.value() / 100.0}")
        self.slider_value.setFixedWidth(50)
        self.controls_layout.addWidget(self.slider_value)
        self.area_slider.valueChanged.connect(lambda value: [self.slider_value.setText(f"{value / 100.0}"), self.style.update_deformation_parameters(self.stenosis_area_slider.value() / 100.0, -self.area_slider.value() / 100.0)])
        self.slider_value.textChanged.connect(lambda text: [self.area_slider.setValue(int(float(text) * 100)), self.style.update_deformation_parameters(self.stenosis_area_slider.value() / 100.0, -self.area_slider.value() / 100.0)])
        '''
        # --- UI Setup ---
        # Slider's label
        self.slider_label = QLabel("Force Scale:")
        self.controls_layout.addWidget(self.slider_label)

        # Slider for the area adjustment (using a nonlinear mapping)
        self.area_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_slider.setRange(-1000, 1000)  # Underlying slider range
        # Initialize to a value that corresponds to 0.2 (for example)
        self.area_slider.setValue(self.force_scale_value_to_slider(0.2))
        self.controls_layout.addWidget(self.area_slider)
        # Display the slider value in a QLineEdit (to show the float value)
        self.slider_value = QLineEdit()
        # Set the initial text using the mapping function (format to three decimals)
        self.slider_value.setText(f"{self.force_scale_slider_to_value(self.area_slider.value()):.4f}")
        self.slider_value.setFixedWidth(50)
        # Connect signals to the handlers.
        self.area_slider.valueChanged.connect(self.on_force_scale_slider_change)
        self.slider_value.textChanged.connect(self.on_force_scale_text_change)
        self.controls_layout.addWidget(self.slider_value)

        # Button for showing selectable nodes on the centerline
        self.show_nodes_button = QPushButton("Select Point")
        self.show_nodes_button.setFixedWidth(120)
        self.controls_layout.addWidget(self.show_nodes_button)
        # Button for running the deformation
        self.run_button = QPushButton("Aneurysm Apply")
        self.run_button.setFixedWidth(120)
        self.controls_layout.addWidget(self.run_button)
        # Add controls layout to main layout
        self.layout.addLayout(self.controls_layout)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.run_button.clicked.connect(self.run_deformation)
        self.show_nodes_button.clicked.connect(self.display_centerline_nodes)

        # To add a second row of buttons, add another QHBoxLayout and add it to the main layout
        self.controls_layout2 = QHBoxLayout()
        # Add stenosis controls
        self.stenosis_slider_label = QLabel("Epsilon Negative Exponent:")
        self.controls_layout2.addWidget(self.stenosis_slider_label)
        
        self.stenosis_area_slider = QSlider(Qt.Orientation.Horizontal)
        # self.stenosis_area_slider.setRange(-3, 6)
        self.stenosis_area_slider.setRange(0, 500)
        self.stenosis_area_slider.setValue(20)
        self.controls_layout2.addWidget(self.stenosis_area_slider)
        self.stenosis_slider_value = QLineEdit()
        self.stenosis_slider_value.setText(f"{self.stenosis_area_slider.value() / 100.0}")
        self.stenosis_slider_value.setFixedWidth(50)
        self.stenosis_area_slider.valueChanged.connect(lambda value: [self.stenosis_slider_value.setText(f"{value / 100.0}"), self.style.update_deformation_parameters(self.stenosis_area_slider.value() / 100.0, -self.force_scale_slider_to_value(self.area_slider.value()))])
        self.stenosis_slider_value.textChanged.connect(lambda text: [self.stenosis_area_slider.setValue(int(float(text) * 100)), self.style.update_deformation_parameters(self.stenosis_area_slider.value() / 100.0, -self.force_scale_slider_to_value(self.area_slider.value()))])
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
        
        self.run_stenosis_button = QPushButton("Stenosis Apply")
        self.run_stenosis_button.setFixedWidth(120)
        self.controls_layout2.addWidget(self.run_stenosis_button)
        self.layout.addLayout(self.controls_layout2)
        # Connect the slider and QLineEdit for stenosis area
        # self.stenosis_area_slider.valueChanged.connect(lambda value: self.stenosis_slider_value.setText(f"{value}"))
        # self.stenosis_slider_value.textChanged.connect(lambda text: self.stenosis_area_slider.setValue(int(text)))
        # Connect the slider and QLineEdit for num ring points
        self.num_ring_points_slider.valueChanged.connect(lambda value: self.num_ring_points_value.setText(f"{value}"))
        self.num_ring_points_value.textChanged.connect(lambda text: self.num_ring_points_slider.setValue(int(text)))
        # Connect the button to the run_stenosis method
        self.run_stenosis_button.clicked.connect(self.run_stenosis)

        # Third row of buttons
        # self.controls_layout3 = QHBoxLayout()
        # self.animated_aneurysm_button = QPushButton("Animated Aneurysm Apply")
        # self.animated_aneurysm_button.setFixedWidth(200)
        # self.controls_layout3.addStretch(1)
        # self.controls_layout3.addWidget(self.animated_aneurysm_button)
        # self.layout.addLayout(self.controls_layout3)
        # self.animated_aneurysm_button.clicked.connect(self.run_deformation)

        # To add a fourth row of buttons, add another QHBoxLayout and add it to the main layout
        self.controls_layout4 = QHBoxLayout()
        # Import Buttons
        self.import_mesh_button = QPushButton("Import Mesh")
        self.import_mesh_button.setFixedWidth(120)
        self.import_mesh_button.clicked.connect(self.import_mesh)
        self.controls_layout4.addWidget(self.import_mesh_button)
        self.import_centerline_button = QPushButton("Import Centerline")
        self.import_centerline_button.setFixedWidth(150)
        self.import_centerline_button.clicked.connect(self.import_centerline)
        self.controls_layout4.addWidget(self.import_centerline_button)
        # Save Button
        self.run_save_button = QPushButton("Save")
        self.run_save_button.setFixedWidth(120)
        # to make the button right aligned
        self.controls_layout4.addStretch(1)
        self.controls_layout4.addWidget(self.run_save_button)
        self.layout.addLayout(self.controls_layout4)
        self.run_save_button.clicked.connect(self.save_mesh)
        # Button to toggle camera locking
        self.toggle_camera_lock_button = QPushButton("Camera Lock")
        self.toggle_camera_lock_button.setFixedWidth(120)
        self.controls_layout4.addWidget(self.toggle_camera_lock_button)
        self.toggle_camera_lock_button.clicked.connect(self.toggle_camera_lock)
        # Button to toggle interleave mode
        self.toggle_interleave_mode_button = QPushButton("Interleave Mode")
        self.toggle_interleave_mode_button.setFixedWidth(130)
        # self.toggle_interleave_mode_button.setStyleSheet("background-color: #d84005;")
        self.controls_layout4.addWidget(self.toggle_interleave_mode_button)
        self.toggle_interleave_mode_button.clicked.connect(self.toggle_interleave_mode)

        # Reverse animation direction button
        self.reverse_animation_button = QPushButton("Reverse Direction")
        self.reverse_animation_button.setFixedWidth(130)
        self.controls_layout4.addWidget(self.reverse_animation_button)
        self.reverse_animation_button.clicked.connect(self.reverse_animation_direction)
        
        # Continuous Stent Edge apply button
        self.stent_edge_button = QPushButton("Stent Edge")
        self.stent_edge_button.setFixedWidth(120)
        self.controls_layout4.addWidget(self.stent_edge_button)
        self.stent_edge_timer.timeout.connect(self.run_stent_edge)
        self.stent_edge_button.pressed.connect(self.start_stent_edge_deformation)
        self.stent_edge_button.released.connect(self.stop_stent_edge_deformation)

        # Animated Aneurysm Apply Button
        self.animated_aneurysm_button = QPushButton("Animated Aneurysm Apply")
        self.animated_aneurysm_button.setFixedWidth(200)
        self.controls_layout4.addWidget(self.animated_aneurysm_button)
        self.animated_aneurysm_button.pressed.connect(self.start_animated_deformation)
        self.animated_aneurysm_button.released.connect(self.stop_animated_deformation)

        # Add continuous aneurysm apply button
        self.continuous_run_button = QPushButton("Continuous Aneurysm Apply")
        self.continuous_run_button.setFixedWidth(200)
        self.controls_layout.addWidget(self.continuous_run_button)
        # Timer for continuous deformation
        self.timer.timeout.connect(self.run_deformation)
        # Connect button press and release events
        self.continuous_run_button.pressed.connect(self.start_continuous_deformation)
        self.continuous_run_button.released.connect(self.stop_continuous_deformation)
        self.animation_timer.timeout.connect(self.interleave_update_selected_points)

        # VTK Setup
        self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_handler = None
        # Pre-loaded Example mode for development
        ######################## DEMO 2 ########################
        # self.mesh_file = "/home/bohanjeffli/mesh-complete-exterior.vtp"
        # self.mesh_file = "/home/bohanjeffli/Unstented-Full-Tree-PA.vtp"
        # self.centerline_file = "/home/bohanjeffli/Full_Centerlines.vtp"
        ######################## DEMO 1 ########################
        # self.mesh_file = "/home/bohanjeffli/Unstented-Full-Tree-PA.vtp"
        self.mesh_file = "/home/bohanjeffli/mesh-complete-exterior.vtp"
        self.centerline_file = "/home/bohanjeffli/centerline.vtp"
        #######################################################
        # self.mesh_file = "/home/bohanjeffli/ImageToStl.com_9x9_square_grid_verbose.vtp"
        # self.centerline_file = "/home/bohanjeffli/ImageToStl.com_midline_polyline.vtp"
        ########################################################
        # self.mesh_file = "/home/bohanjeffli/Remeshed-200k-Stented-Truncated-PA.vtp"
        # self.centerline_file = "/home/bohanjeffli/full_m_L_R_2Daughters_Centerlines.vtp"
        ########################################################
        # self.mesh_file = None
        # self.centerline_file = None
        self.initialize_vtk_handler()

    def save_mesh(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Mesh", "mesh-name.vtp", "VTK Files (*.vtp)")
        if file_name:
            self.vtk_handler.save_mesh(file_name)
            print(f"Saved mesh to {file_name}")

    def import_mesh(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Mesh", "", "VTK Files (*.vtp)")
        if file_name:
            self.mesh_file = file_name
            if self.centerline_file:
                self.initialize_vtk_handler()

    def import_centerline(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Centerline", "", "VTK Files (*.vtp)")
        if file_name:
            self.centerline_file = file_name
            if self.mesh_file:
                self.initialize_vtk_handler()

    def initialize_vtk_handler(self):
        self.vtk_handler = VTKHandler(self.mesh_file, self.centerline_file)

        self.ren = self.vtk_handler.get_renderer()
        self.ren.SetBackground(1, 1, 1)
        if self.vtk_widget.GetRenderWindow().GetRenderers().GetNumberOfItems() > 0:
            self.vtk_widget.GetRenderWindow().GetRenderers().RemoveAllItems()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)

        self.style = self.vtk_handler.get_interactor_style()
        self.vtk_interactor.SetInteractorStyle(self.style)
        self.vtk_interactor.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.vtk_widget.GetRenderWindow().Render()

        self.vtk_interactor.Initialize()
        self.vtk_interactor.Start()

        self.style.update_deformation_parameters(self.stenosis_area_slider.value() / 100.0, -self.force_scale_slider_to_value(self.area_slider.value()))
        self.toggle_interleave_mode_button.setStyleSheet("background-color: #d84005;")
        self.toggle_camera_lock_button.setStyleSheet("background-color: white")

    def keyPressEvent(self, event):
        # when buttons are pressed focus shifts to PyQt window so key press events are not captured by VTK and needed to be handled here
        if event.key() == Qt.Key.Key_H:
            self.style.toggle_roi_cylinder()
        elif event.key() == Qt.Key.Key_D:
            # self.start_continuous_deformation()
            pass

    def run_deformation(self):
        # area_percent_change = self.area_slider.value()
        force_scale = -self.force_scale_slider_to_value(self.area_slider.value())
        print(f"Running stent with force scale: {force_scale}")
        # epsilon = 10 ** (-self.stenosis_area_slider.value()) # slider value 0 or 1 works well here
        epsilon = self.stenosis_area_slider.value() / 100.0
        print(f"Running stent with epsilon: {epsilon}")
        # self.style.deform_mesh(epsilon, force_scale)
        self.style.deform_mesh_sequential(epsilon, force_scale)
        # self.vtk_widget.GetRenderWindow().Render()
    
    def run_stent_edge(self):
        force_scale = -self.force_scale_slider_to_value(self.area_slider.value())
        epsilon = self.stenosis_area_slider.value() / 100.0
        print(f"Shaping stent edge with epsilon: {epsilon}")
        self.style.deform_mesh_stent_edge(epsilon, force_scale)

    def update_selected_point(self):
        # every 1 second, shift the selected force center to the next centerline node by incrementing or decrementing the index
        self.style.update_selected_point()

    def interleave_update_selected_points(self):
        self.style.interleave_update_selected_points()

    def reverse_animation_direction(self):
        if self.reverse_animation_button.styleSheet() == "background-color: #d84005;":
            self.reverse_animation_button.setStyleSheet("background-color: white")
        else:
            self.reverse_animation_button.setStyleSheet("background-color: #d84005;")
        self.style.reverse_animation_direction()

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

    def toggle_interleave_mode(self):
        if self.toggle_interleave_mode_button.styleSheet() == "background-color: #d84005;":
            self.toggle_interleave_mode_button.setStyleSheet("background-color: white")
        else:
            self.toggle_interleave_mode_button.setStyleSheet("background-color: #d84005;")
        if self.style.interleave_mode:
            self.animation_timer.timeout.disconnect()
            self.animation_timer.timeout.connect(self.update_selected_point)
        else:
            self.animation_timer.timeout.disconnect()
            self.animation_timer.timeout.connect(self.interleave_update_selected_points)
        self.style.toggle_interleave_mode()
    
    def toggle_camera_lock(self):
        if self.toggle_camera_lock_button.styleSheet() == "background-color: #d84005;":
            self.toggle_camera_lock_button.setStyleSheet("background-color: white")
        else:
            self.toggle_camera_lock_button.setStyleSheet("background-color: #d84005;")
        self.style.toggle_camera_lock()

    def display_centerline_nodes(self):
        print("Please select three centerline nodes to generate aneurysm.")
        force_scale = -self.force_scale_slider_to_value(self.area_slider.value())
        epsilon = self.stenosis_area_slider.value() / 100.0
        self.style.display_centerline_vertices()

    def start_continuous_deformation(self):
        self.timer.start(50)  # Run every 50 milliseconds

    def stop_continuous_deformation(self):
        self.timer.stop()

    def start_animated_deformation(self):
        self.timer.start(50)
        self.animation_timer.start(100)

    def stop_animated_deformation(self):
        self.timer.stop()
        self.animation_timer.stop()
    
    def start_stent_edge_deformation(self):
        self.stent_edge_timer.start(50)

    def stop_stent_edge_deformation(self):
        self.stent_edge_timer.stop()
    
    def on_force_scale_slider_change(self, raw_value):
        """
        Called when the slider is moved. Converts the slider’s raw integer value
        to the nonlinearly mapped float value, updates the text display, and calls
        update_deformation_parameters() accordingly.
        """
        float_value = self.force_scale_slider_to_value(raw_value)
        self.slider_value.setText(f"{float_value:.4f}")
        # Assuming self.stenosis_area_slider exists, convert its value similarly:
        stenosis_value = self.stenosis_area_slider.value() / 100.0
        self.style.update_deformation_parameters(stenosis_value, -float_value)

    def on_force_scale_text_change(self, text):
        """
        Called when the user types a new value into the QLineEdit.
        Parses the float value, converts it to the corresponding slider value,
        and updates the slider and deformation parameters.
        """
        try:
            # Convert entered text to float
            new_value = float(text)
        except ValueError:
            # If parsing fails, do nothing.
            return
        new_slider_val = self.force_scale_value_to_slider(new_value)
        self.area_slider.setValue(new_slider_val)
        # Again, update the deformation parameters.
        stenosis_value = self.stenosis_area_slider.value() / 100.
        self.style.update_deformation_parameters(stenosis_value, -new_value)

    @staticmethod
    def force_scale_slider_to_value(slider_val):
        """
        Converts the raw slider value (from -1000 to 1000) into the
        actual float value using a 7th degree polynomial mapping.
        """
        # Normalize slider value to [-1, 1]
        normalized = slider_val / 1000.0
        # Map nonlinearly. Near zero, changes are tiny;
        # at the extremes, the full range (10) is reached.
        mapped = math.copysign((abs(normalized) ** 7), normalized) * 10.0
        return mapped

    @staticmethod
    def force_scale_value_to_slider(value):
        """
        Converts the displayed float value back to the slider integer value.
        This is the inverse of the slider_to_value() function.
        """
        # To invert: if value = sign(x)*|x|^7*10, then
        # x = sign(value)* (|value|/10)^(1/7) and slider value = x * 1000.
        if value == 0:
            normalized = 0.0
        else:
            normalized = math.copysign((abs(value) / 10.0) ** (1.0 / 7), value)
        return int(normalized * 1000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

            