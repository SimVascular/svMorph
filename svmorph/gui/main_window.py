"""
Medical Visualization Application with VTK Integration

This module provides a PyQt6-based GUI application for medical mesh visualization
and deformation using VTK (Visualization Toolkit). The application supports:

- 3D mesh and centerline visualization
- Interactive deformation operations (aneurysm, stenosis, stent placement)
- Real-time parameter adjustment through sliders and controls
- SDF (Signed Distance Field) visualization and contact detection
- Mesh import/export functionality
- Animation and continuous deformation modes

Main Features:
1. Force Scale Control: Adjusts deformation intensity using nonlinear mapping
2. Stent Parameters: Length and diameter controls for stent placement
3. Stenosis Simulation: Creates narrowing in blood vessels
4. Contact Detection: SDF-based collision detection during deformation
5. Camera Controls: Locking and animation features
6. Multi-point Selection: Interactive point selection on centerlines

Dependencies:
- PyQt6: GUI framework
- VTK: 3D visualization and mesh processing
- svmorph: Core deformation and visualization modules
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QSlider,
    QLabel,
    QHBoxLayout,
    QFileDialog,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer
import vtkmodules.all as vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from svmorph.visualization.renderer import SceneManager
from svmorph.logging import get_logger

logger = get_logger(__name__)

# Application Constants
# Stent parameter ranges and steps
MAX_STENT_SIZE = 2.0  # 1cm = 10mm diameter
MIN_STENT_SIZE = 0.1  # 0.1cm = 1mm diameter
STENT_DIAMETER_NUM_STEPS = 190

MAX_STENT_LENGTH = 8.0  # 80mm length
MIN_STENT_LENGTH = 1.0  # 20mm length
STENT_LENGTH_NUM_STEPS = 70

# UI Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
WINDOW_X = 100
WINDOW_Y = 100

# Control widget dimensions
BUTTON_WIDTH_SMALL = 120
BUTTON_WIDTH_LARGE = 150
BUTTON_WIDTH_XLARGE = 200
TEXT_INPUT_WIDTH = 50
TEXT_INPUT_WIDTH_MEDIUM = 60

# Timer intervals (milliseconds)
DEFORMATION_TIMER_INTERVAL = 50
STENOSIS_TIMER_INTERVAL = 25

# Slider ranges and defaults
FORCE_SCALE_SLIDER_RANGE = (-1000, 1000)
FORCE_SCALE_DEFAULT = 1
EPSILON_SLIDER_RANGE = (0, 500)
EPSILON_DEFAULT = 20
STENT_DIAMETER_DEFAULT = 0.8  # 9mm stent
STENT_LENGTH_DEFAULT = 1.7  # 17mm stent

# Colors and styling
ACTIVE_BUTTON_COLOR = "#d84005"
INACTIVE_BUTTON_COLOR = "white"


class SliderMapper:
    """Helper class for managing slider value mappings"""

    @staticmethod
    def force_scale_slider_to_value(slider_val):
        """
        Converts the raw slider value (from -1000 to 1000) into the
        actual float value using linear mapping.
        """
        normalized = slider_val / 1000.0
        return normalized

    @staticmethod
    def force_scale_value_to_slider(value):
        """
        Converts the displayed float value back to the slider integer value.
        This is the inverse of the slider_to_value() function.
        """
        if value == 0:
            normalized = 0.0
        else:
            normalized = value
        return int(normalized * 1000)

    @staticmethod
    def stent_diameter_slider_to_value(slider_val):
        """Convert stent diameter slider value to actual diameter"""
        return MIN_STENT_SIZE + (slider_val / STENT_DIAMETER_NUM_STEPS) * (
            MAX_STENT_SIZE - MIN_STENT_SIZE
        )

    @staticmethod
    def stent_diameter_value_to_slider(value):
        """Convert stent diameter value to slider position"""
        clamped_value = max(MIN_STENT_SIZE, min(MAX_STENT_SIZE, value))
        return int(
            (clamped_value - MIN_STENT_SIZE)
            / (MAX_STENT_SIZE - MIN_STENT_SIZE)
            * STENT_DIAMETER_NUM_STEPS
        )

    @staticmethod
    def stent_length_slider_to_value(slider_val):
        """Convert stent length slider value to actual length"""
        return MIN_STENT_LENGTH + (slider_val / STENT_LENGTH_NUM_STEPS) * (
            MAX_STENT_LENGTH - MIN_STENT_LENGTH
        )

    @staticmethod
    def stent_length_value_to_slider(value):
        """Convert stent length value to slider position"""
        clamped_value = max(MIN_STENT_LENGTH, min(MAX_STENT_LENGTH, value))
        return int(
            (clamped_value - MIN_STENT_LENGTH)
            / (MAX_STENT_LENGTH - MIN_STENT_LENGTH)
            * STENT_LENGTH_NUM_STEPS
        )


class UIStyleManager:
    """Helper class for managing UI styling and button states"""

    @staticmethod
    def set_button_active(button, is_active=True):
        """Set button to active or inactive state"""
        color = ACTIVE_BUTTON_COLOR if is_active else INACTIVE_BUTTON_COLOR
        button.setStyleSheet(f"background-color: {color}")

    @staticmethod
    def toggle_button_style(button):
        """Toggle button between active and inactive states"""
        current_style = button.styleSheet()
        is_active = ACTIVE_BUTTON_COLOR in current_style
        UIStyleManager.set_button_active(button, not is_active)
        return not is_active


class MainWindow(QMainWindow):
    """Main application window for medical visualization with VTK integration"""

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle("VTK with PyQt6")
        self._setup_window_geometry()
        self._setup_main_layout()
        self._setup_timers()
        self._setup_ui_controls()
        self._setup_vtk_components()
        self.initialize_vtk_handler()

    def _setup_window_geometry(self):
        """Configure main window size and position"""
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

    def _setup_main_layout(self):
        """Setup the main frame and layout"""
        self.frame = QWidget()
        self.layout = QVBoxLayout()

        # VTK Render Widget
        self.vtk_widget = QVTKRenderWindowInteractor(self.frame)
        self.layout.addWidget(self.vtk_widget)

    def _setup_timers(self):
        """Initialize all timers used for animations and continuous operations"""
        self.timer = QTimer(self)
        self.kelvinlet_timer = QTimer(self)
        self.stenosis_timer = QTimer(self)

    def _setup_ui_controls(self):
        """Setup all UI control layouts and widgets"""
        self._setup_force_scale_controls()
        self._setup_stent_parameter_controls()
        self._setup_stenosis_controls()
        self._setup_import_export_controls()
        self._setup_epsilon_controls()

        # Finalize layout
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

    def _setup_force_scale_controls(self):
        """Setup the force scale slider and related controls (first row)"""
        self.controls_layout = QHBoxLayout()

        # Force scale slider with nonlinear mapping
        self.slider_label = QLabel("Force Scale:")
        self.controls_layout.addWidget(self.slider_label)

        self.area_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_slider.setRange(*FORCE_SCALE_SLIDER_RANGE)
        self.area_slider.setValue(
            SliderMapper.force_scale_value_to_slider(FORCE_SCALE_DEFAULT)
        )
        self.controls_layout.addWidget(self.area_slider)

        self.slider_value = QLineEdit()
        slider_value = SliderMapper.force_scale_slider_to_value(
            self.area_slider.value()
        )
        self.slider_value.setText(
            f"{slider_value:.3f}" if slider_value >= 0 else f"-{abs(slider_value):.3f}"
        )
        self.slider_value.setFixedWidth(TEXT_INPUT_WIDTH)
        self.controls_layout.addWidget(self.slider_value)

        # Action buttons
        self.show_nodes_button = QPushButton("Select Point")
        self.show_nodes_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout.addWidget(self.show_nodes_button)

        self.run_button = QPushButton("Contact Apply")
        self.run_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout.addWidget(self.run_button)

        self.continuous_sdf_button = QPushButton("Continuous Contact Apply")
        self.continuous_sdf_button.setFixedWidth(BUTTON_WIDTH_XLARGE)
        self.controls_layout.addWidget(self.continuous_sdf_button)

        self.layout.addLayout(self.controls_layout)

        # Connect signals
        self.area_slider.valueChanged.connect(self.on_force_scale_slider_change)
        self.slider_value.textChanged.connect(self.on_force_scale_text_change)
        self.run_button.clicked.connect(self.run_deformation_sdf)
        self.show_nodes_button.clicked.connect(self.display_centerline_nodes)
        self.timer.timeout.connect(self.run_deformation_sdf)
        self.continuous_sdf_button.pressed.connect(self.start_continuous_deformation)
        self.continuous_sdf_button.released.connect(self.stop_continuous_deformation)

    def _setup_stent_parameter_controls(self):
        """Setup stent length and diameter controls (second row)"""
        self.controls_layout2 = QHBoxLayout()

        # Stent length controls
        self.stent_length_label = QLabel("Stent Length:")
        self.controls_layout2.addWidget(self.stent_length_label)

        self.stent_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.stent_length_slider.setRange(0, STENT_LENGTH_NUM_STEPS)
        default_length_slider_value = SliderMapper.stent_length_value_to_slider(
            STENT_LENGTH_DEFAULT
        )
        self.stent_length_slider.setValue(default_length_slider_value)
        self.controls_layout2.addWidget(self.stent_length_slider)

        self.stent_length_value = QLineEdit()
        self.stent_length_value.setText(f"{STENT_LENGTH_DEFAULT:.4f}")
        self.stent_length_value.setFixedWidth(TEXT_INPUT_WIDTH)
        self.controls_layout2.addWidget(self.stent_length_value)

        # Stent diameter controls
        self.stent_radius_label = QLabel("Stent Diameter:")
        self.controls_layout2.addWidget(self.stent_radius_label)

        self.stent_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.stent_diameter_slider.setRange(0, STENT_DIAMETER_NUM_STEPS)
        default_diameter_slider_value = SliderMapper.stent_diameter_value_to_slider(
            STENT_DIAMETER_DEFAULT
        )
        self.stent_diameter_slider.setValue(default_diameter_slider_value)
        self.controls_layout2.addWidget(self.stent_diameter_slider)

        self.stent_diameter_value = QLineEdit()
        self.stent_diameter_value.setText(f"{STENT_DIAMETER_DEFAULT:.4f}")
        self.stent_diameter_value.setFixedWidth(TEXT_INPUT_WIDTH)
        self.controls_layout2.addWidget(self.stent_diameter_value)

        self.layout.addLayout(self.controls_layout2)

        # Connect signals
        self.stent_length_slider.valueChanged.connect(
            self._on_stent_length_slider_change
        )
        self.stent_length_value.textChanged.connect(self.update_stent_length_slider)
        self.stent_diameter_slider.valueChanged.connect(
            self._on_stent_diameter_slider_change
        )
        self.stent_diameter_value.textChanged.connect(self.update_stent_diameter_slider)

    def _setup_stenosis_controls(self):
        """Setup stenosis-related controls (third row)"""
        self.controls_layout3 = QHBoxLayout()

        self.run_stenosis_button = QPushButton("Stenosis Apply")
        self.run_stenosis_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout3.addWidget(self.run_stenosis_button)

        # Stenosis parameter inputs
        self.stenosis_radius_label = QLabel("Stenosis Minimum Radius:")
        self.controls_layout3.addWidget(self.stenosis_radius_label)
        self.stenosis_radius_value = QLineEdit()
        self.stenosis_radius_value.setText("0.1")
        self.stenosis_radius_value.setFixedWidth(TEXT_INPUT_WIDTH_MEDIUM)
        self.controls_layout3.addWidget(self.stenosis_radius_value)

        self.stenosis_length_label = QLabel("Stenosis Region Length:")
        self.controls_layout3.addWidget(self.stenosis_length_label)
        self.stenosis_length_value = QLineEdit()
        self.stenosis_length_value.setText("0.5")
        self.stenosis_length_value.setFixedWidth(TEXT_INPUT_WIDTH_MEDIUM)
        self.controls_layout3.addWidget(self.stenosis_length_value)

        self.controls_layout3.addStretch(1)

        self.continuous_stenosis_button = QPushButton("Continuous Stenosis Apply")
        self.continuous_stenosis_button.setFixedWidth(BUTTON_WIDTH_XLARGE)
        self.controls_layout3.addWidget(self.continuous_stenosis_button)

        self.layout.addLayout(self.controls_layout3)

        # Connect signals
        self.run_stenosis_button.clicked.connect(self.run_stenosis)
        self.stenosis_timer.timeout.connect(self.run_stenosis)
        self.continuous_stenosis_button.pressed.connect(
            self.start_continuous_stenosis_deformation
        )
        self.continuous_stenosis_button.released.connect(
            self.stop_continuous_stenosis_deformation
        )

    def _setup_import_export_controls(self):
        """Setup import/export and mode control buttons (fourth row)"""
        self.controls_layout4 = QHBoxLayout()

        # Import buttons
        self.import_mesh_button = QPushButton("Import Mesh")
        self.import_mesh_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.import_mesh_button.clicked.connect(self.import_mesh)
        self.controls_layout4.addWidget(self.import_mesh_button)

        self.import_centerline_button = QPushButton("Import Centerline")
        self.import_centerline_button.setFixedWidth(BUTTON_WIDTH_LARGE)
        self.import_centerline_button.clicked.connect(self.import_centerline)
        self.controls_layout4.addWidget(self.import_centerline_button)

        # Save button
        self.controls_layout4.addStretch(1)
        self.run_save_button = QPushButton("Save")
        self.run_save_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout4.addWidget(self.run_save_button)
        self.run_save_button.clicked.connect(self.save_mesh)

        # Mode control buttons
        self.toggle_camera_lock_button = QPushButton("Camera Lock")
        self.toggle_camera_lock_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout4.addWidget(self.toggle_camera_lock_button)
        self.toggle_camera_lock_button.clicked.connect(self.toggle_camera_lock)

        # Additional action buttons
        self.place_stent_button = QPushButton("Place Stent") # originally the "Stent Edge" button
        self.place_stent_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout4.addWidget(self.place_stent_button)
        self.place_stent_button.pressed.connect(self.save_current_stent)

        self.layout.addLayout(self.controls_layout4)

    def _setup_epsilon_controls(self):
        """Setup epsilon and additional control buttons (fifth row)"""
        self.controls_layout5 = QHBoxLayout()

        # Epsilon slider controls
        self.stenosis_slider_label = QLabel("Epsilon Negative Exponent:")
        self.controls_layout5.addWidget(self.stenosis_slider_label)

        self.stenosis_area_slider = QSlider(Qt.Orientation.Horizontal)
        self.stenosis_area_slider.setRange(*EPSILON_SLIDER_RANGE)
        self.stenosis_area_slider.setValue(EPSILON_DEFAULT)
        self.controls_layout5.addWidget(self.stenosis_area_slider)

        self.stenosis_slider_value = QLineEdit()
        self.stenosis_slider_value.setText(
            f"{self.stenosis_area_slider.value() / 100.0}"
        )
        self.stenosis_slider_value.setFixedWidth(TEXT_INPUT_WIDTH)
        self.controls_layout5.addWidget(self.stenosis_slider_value)

        self.controls_layout5.addStretch(1)

        self.render_sdf_button = QPushButton("Render SDF")
        self.render_sdf_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout5.addWidget(self.render_sdf_button)
        self.render_sdf_button.clicked.connect(self.render_sdf)

        self.simultaneous_apply_button = QPushButton("Straighten Apply")
        self.simultaneous_apply_button.setFixedWidth(BUTTON_WIDTH_XLARGE)
        self.controls_layout5.addWidget(self.simultaneous_apply_button)
        self.simultaneous_apply_button.clicked.connect(
            self.run_deformation_simultaneous
        )

        self.continuous_kelvinlet_button = QPushButton("Continuous Aneurysm Apply")
        self.continuous_kelvinlet_button.setFixedWidth(BUTTON_WIDTH_XLARGE)
        self.controls_layout5.addWidget(self.continuous_kelvinlet_button)
        self.kelvinlet_timer.timeout.connect(self.run_deformation)
        self.continuous_kelvinlet_button.pressed.connect(
            self.start_continuous_kelvinlet_deformation
        )
        self.continuous_kelvinlet_button.released.connect(
            self.stop_continuous_kelvinlet_deformation
        )

        self.layout.addLayout(self.controls_layout5)

        # Connect epsilon slider signals
        self.stenosis_area_slider.valueChanged.connect(self._on_epsilon_slider_change)
        self.stenosis_slider_value.textChanged.connect(self._on_epsilon_text_change)

    def _setup_vtk_components(self):
        """Initialize VTK-related components"""
        self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_handler = None
        self.mesh_file = "input/TST-STAN-3/TST-STAN-3-preop-FINAL-030426.vtp"
        self.centerline_file = "input/TST-STAN-3/TST-STAN-3-preop-FINAL-030426-centerlines.vtp"

    def _on_stent_length_slider_change(self, value):
        """Handle stent length slider changes"""
        length_value = SliderMapper.stent_length_slider_to_value(value)
        self.stent_length_value.setText(f"{length_value:.4f}")
        if hasattr(self, "style") and self.style:
            self.style.update_prescribed_stent_length(length_value)

    def _on_stent_diameter_slider_change(self, value):
        """Handle stent diameter slider changes"""
        diameter_value = SliderMapper.stent_diameter_slider_to_value(value)
        self.stent_diameter_value.setText(f"{diameter_value:.4f}")
        if hasattr(self, "style") and self.style:
            self.style.update_prescribed_stent_radius(diameter_value / 2.0)

    def _on_epsilon_slider_change(self, value):
        """Handle epsilon slider changes"""
        epsilon_value = value / 100.0
        self.stenosis_slider_value.setText(f"{epsilon_value}")
        if hasattr(self, "style") and self.style:
            force_scale_value = SliderMapper.force_scale_slider_to_value(
                self.area_slider.value()
            )
            self.style.update_deformation_parameters(epsilon_value, -force_scale_value)

    def _on_epsilon_text_change(self, text):
        """Handle epsilon text input changes"""
        try:
            value = float(text)
            self.stenosis_area_slider.setValue(int(value * 100))
            if hasattr(self, "style") and self.style:
                force_scale_value = SliderMapper.force_scale_slider_to_value(
                    self.area_slider.value()
                )
                self.style.update_deformation_parameters(value, -force_scale_value)
        except ValueError:
            pass

    def initialize_vtk_handler(self):
        """Initialize VTK handler and setup the visualization pipeline"""
        if not self.mesh_file or not self.centerline_file:
            # No data loaded yet — show an empty white viewport
            if not hasattr(self, 'ren'):
                self.ren = vtk.vtkRenderer()
                self.ren.SetBackground(1, 1, 1)
                self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)
                self.vtk_interactor.SetRenderWindow(self.vtk_widget.GetRenderWindow())
                self.vtk_widget.GetRenderWindow().Render()
                self.vtk_interactor.Initialize()
                self.vtk_interactor.Start()
            logger.warning("Both mesh and centerline files are required.")
            return

        self.vtk_handler = SceneManager(self.mesh_file, self.centerline_file)

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

        # Initialize deformation parameters
        epsilon_value = self.stenosis_area_slider.value() / 100.0
        force_scale_value = SliderMapper.force_scale_slider_to_value(
            self.area_slider.value()
        )
        self.style.update_deformation_parameters(epsilon_value, -force_scale_value)

        # Initialize UI styling
        UIStyleManager.set_button_active(self.toggle_camera_lock_button, False)

        # Display debugging information about radius as text in viewport
        self.style.display_radius_texts()

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # when buttons are pressed focus shifts to PyQt window so key press events
        # are not captured by VTK and needed to be handled here
        if event.key() == Qt.Key.Key_H:
            self.style.toggle_roi_cylinder()
        elif event.key() == Qt.Key.Key_D:
            # Reserved for future functionality
            pass

    def run_deformation(self):
        """Run kelvinlet-based mesh deformation"""
        force_scale = -SliderMapper.force_scale_slider_to_value(
            self.area_slider.value()
        )
        epsilon = self.stenosis_area_slider.value() / 100.0
        logger.info(f"Running Kelvinlet deformation with force_scale={force_scale}, epsilon={epsilon}")
        self.style.deform_mesh_sequential(epsilon, force_scale)

    def run_deformation_sdf(self):
        """Run SDF-based contact deformation"""
        force_scale = -SliderMapper.force_scale_slider_to_value(
            self.area_slider.value()
        )
        epsilon = self.stenosis_area_slider.value() / 100.0
        logger.info(
            f"Running SDF contact deformation with force_scale={force_scale}, epsilon={epsilon}"
        )
        self.style.deform_mesh_sdf_contact(epsilon, force_scale)

    def run_deformation_simultaneous(self):
        """Run simultaneous parallel mesh deformation"""
        force_scale = -SliderMapper.force_scale_slider_to_value(
            self.area_slider.value()
        )
        epsilon = self.stenosis_area_slider.value() / 100.0
        logger.info(f"Running straightening deformation with force_scale={force_scale}, epsilon={epsilon}")
        self.style.deform_mesh_with_straightening(epsilon, force_scale)

    def run_stenosis(self):
        """Run stenosis deformation with user-specified parameters"""
        if self.vtk_handler is None:
            logger.warning(
                "Please import both mesh and centerline files before running the stenosis."
            )
            return

        try:
            stenosis_radius = float(self.stenosis_radius_value.text())
            stenosis_length = float(self.stenosis_length_value.text())
        except ValueError:
            logger.warning("Invalid stenosis parameters. Please enter valid numbers.")
            return

        force_scale = SliderMapper.force_scale_slider_to_value(self.area_slider.value())
        area_percent_change = self.stenosis_area_slider.value()

        logger.info(
            f"Running stenosis with force_scale={force_scale}, stenosis_radius={stenosis_radius}, stenosis_length={stenosis_length}"
        )
        self.style.deform_mesh_stenosis(
            force_scale, area_percent_change, stenosis_radius, stenosis_length
        )

    def toggle_camera_lock(self):
        """Toggle camera lock and update button styling"""
        UIStyleManager.toggle_button_style(self.toggle_camera_lock_button)
        self.style.toggle_camera_lock()

    def display_centerline_nodes(self):
        """Display centerline nodes for single point selection"""
        logger.info("Please select centerline nodes to generate aneurysm.")
        self.style.display_centerline_vertices()

    def render_sdf(self):
        """Display signed distance field visualization"""
        logger.info("Displaying signed distance field.")
        self.style.render_sdf()

    def start_continuous_deformation(self):
        """Start continuous deformation timer"""
        self.timer.start(DEFORMATION_TIMER_INTERVAL)

    def stop_continuous_deformation(self):
        """Stop continuous deformation timer"""
        self.timer.stop()

    def start_continuous_kelvinlet_deformation(self):
        """Start continuous kelvinlet deformation timer"""
        self.kelvinlet_timer.start(DEFORMATION_TIMER_INTERVAL)

    def stop_continuous_kelvinlet_deformation(self):
        """Stop continuous kelvinlet deformation timer"""
        self.kelvinlet_timer.stop()

    def start_continuous_stenosis_deformation(self):
        """Start continuous stenosis deformation timer"""
        self.stenosis_timer.start(STENOSIS_TIMER_INTERVAL)

    def stop_continuous_stenosis_deformation(self):
        """Stop continuous stenosis deformation timer"""
        self.stenosis_timer.stop()

    def save_current_stent(self):
        """Save the current stent configuration"""
        if hasattr(self, "style") and self.style:
            self.style.save_current_stent()
            logger.info("Current stent placed.")
        else:
            logger.warning("VTK handler not initialized. Cannot place stent.")

    def update_stent_diameter_slider(self, text):
        """Update stent diameter slider from text input"""
        try:
            val = float(text)
            slider_val = SliderMapper.stent_diameter_value_to_slider(val)
            self.stent_diameter_slider.setValue(slider_val)
        except ValueError:
            pass

    def update_stent_length_slider(self, text):
        """Update stent length slider from text input"""
        try:
            val = float(text)
            slider_val = SliderMapper.stent_length_value_to_slider(val)
            self.stent_length_slider.setValue(slider_val)
        except ValueError:
            pass

    def on_force_scale_slider_change(self, raw_value):
        """
        Called when the slider is moved. Converts the slider's raw integer value
        to the nonlinearly mapped float value, updates the text display, and calls
        update_deformation_parameters() accordingly.
        """
        float_value = SliderMapper.force_scale_slider_to_value(raw_value)
        self.slider_value.setText(f"{float_value:.3f}")
        # Assuming self.stenosis_area_slider exists, convert its value similarly:
        stenosis_value = self.stenosis_area_slider.value() / 100.0
        if hasattr(self, "style") and self.style:
            self.style.update_deformation_parameters(stenosis_value, -float_value)

    def on_force_scale_text_change(self, text):
        """
        Called when the user types a new value into the QLineEdit.
        Parses the float value, converts it to the corresponding slider value,
        and updates the slider and deformation parameters.
        """
        try:
            new_value = float(text)
        except ValueError:
            return
        new_slider_val = SliderMapper.force_scale_value_to_slider(new_value)
        self.area_slider.setValue(new_slider_val)
        stenosis_value = self.stenosis_area_slider.value() / 100.0
        if hasattr(self, "style") and self.style:
            self.style.update_deformation_parameters(stenosis_value, -new_value)

    def save_mesh(self):
        """Save the current mesh to a file"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh", "mesh-name.vtp", "VTK Files (*.vtp)"
        )
        if file_name:
            self.vtk_handler.save_mesh(file_name)
            logger.info(f"Saved mesh to {file_name}")

    def import_mesh(self):
        """Import a mesh file and reinitialize if centerline is available"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Mesh", "", "VTK Files (*.vtp)"
        )
        if file_name:
            self.mesh_file = file_name
            if self.centerline_file:
                self.initialize_vtk_handler()

    def import_centerline(self):
        """Import a centerline file and reinitialize if mesh is available"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Centerline", "", "VTK Files (*.vtp)"
        )
        if file_name:
            self.centerline_file = file_name
            if self.mesh_file:
                self.initialize_vtk_handler()
