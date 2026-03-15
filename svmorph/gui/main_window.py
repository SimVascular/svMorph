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
SHARPNESS_SLIDER_RANGE = (1, 500)
SHARPNESS_DEFAULT = 100
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
        """Convert force-scale float to slider integer (inverse of slider_to_value)."""
        return int(value * 1000)

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

        self.setWindowTitle("svMorph")
        self.interactor = None
        self._setup_window_geometry()
        self._setup_main_layout()
        self._setup_timers()
        self._setup_ui_controls()
        self._setup_vtk_components()
        self.initialize_vtk_handler()

    @property
    def _current_sharpness(self):
        """Current sharpness value from the slider (0.01 – 5.0)."""
        return self.sharpness_slider.value() / 100.0

    @property
    def _current_force_scale_raw(self):
        """Current raw force-scale value from the slider (-1.0 – 1.0)."""
        return SliderMapper.force_scale_slider_to_value(self.force_scale_slider.value())

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
        self._setup_sharpness_controls()

        # Finalize layout
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

    def _setup_force_scale_controls(self):
        """Setup the force scale slider and related controls (first row)"""
        self.controls_layout = QHBoxLayout()

        # Force scale slider with nonlinear mapping
        self.slider_label = QLabel("Force Scale:")
        self.controls_layout.addWidget(self.slider_label)

        self.force_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.force_scale_slider.setRange(*FORCE_SCALE_SLIDER_RANGE)
        self.force_scale_slider.setValue(
            SliderMapper.force_scale_value_to_slider(FORCE_SCALE_DEFAULT)
        )
        self.controls_layout.addWidget(self.force_scale_slider)

        self.slider_value = QLineEdit()
        slider_value = SliderMapper.force_scale_slider_to_value(
            self.force_scale_slider.value()
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
        self.force_scale_slider.valueChanged.connect(self.on_force_scale_slider_change)
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
        self.place_stent_button = QPushButton("Place Stent")
        self.place_stent_button.setFixedWidth(BUTTON_WIDTH_SMALL)
        self.controls_layout4.addWidget(self.place_stent_button)
        self.place_stent_button.pressed.connect(self.save_current_stent)

        self.layout.addLayout(self.controls_layout4)

    def _setup_sharpness_controls(self):
        """Setup sharpness and additional control buttons (fifth row)"""
        self.controls_layout5 = QHBoxLayout()

        self.sharpness_label = QLabel("Sharpness:")
        self.controls_layout5.addWidget(self.sharpness_label)

        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setRange(*SHARPNESS_SLIDER_RANGE)
        self.sharpness_slider.setValue(SHARPNESS_DEFAULT)
        self.controls_layout5.addWidget(self.sharpness_slider)

        self.sharpness_value = QLineEdit()
        self.sharpness_value.setText(f"{self._current_sharpness}")
        self.sharpness_value.setFixedWidth(TEXT_INPUT_WIDTH)
        self.controls_layout5.addWidget(self.sharpness_value)

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

        self.sharpness_slider.valueChanged.connect(self._on_sharpness_slider_change)
        self.sharpness_value.textChanged.connect(self._on_sharpness_text_change)

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
        if self.interactor:
            self.interactor.update_prescribed_stent_length(length_value)

    def _on_stent_diameter_slider_change(self, value):
        """Handle stent diameter slider changes"""
        diameter_value = SliderMapper.stent_diameter_slider_to_value(value)
        self.stent_diameter_value.setText(f"{diameter_value:.4f}")
        if self.interactor:
            self.interactor.update_prescribed_stent_radius(diameter_value / 2.0)

    def _on_sharpness_slider_change(self, value):
        """Handle sharpness slider changes"""
        sharpness_value = value / 100.0
        self.sharpness_value.setText(f"{sharpness_value}")
        if self.interactor:
            self.interactor.update_deformation_parameters(sharpness_value, -self._current_force_scale_raw)

    def _on_sharpness_text_change(self, text):
        """Handle sharpness text input changes"""
        try:
            value = float(text)
            self.sharpness_slider.setValue(int(value * 100))
            if self.interactor:
                self.interactor.update_deformation_parameters(value, -self._current_force_scale_raw)
        except ValueError:
            pass

    def initialize_vtk_handler(self):
        """Initialize VTK handler and setup the visualization pipeline"""
        if not self.mesh_file or not self.centerline_file:
            # No data loaded yet — show an empty white viewport
            if not hasattr(self, 'renderer'):
                self.renderer = vtk.vtkRenderer()
                self.renderer.SetBackground(1, 1, 1)
                self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
                self.vtk_interactor.SetRenderWindow(self.vtk_widget.GetRenderWindow())
                self.vtk_widget.GetRenderWindow().Render()
                self.vtk_interactor.Initialize()
                self.vtk_interactor.Start()
            logger.warning("Both mesh and centerline files are required.")
            return

        self.vtk_handler = SceneManager(self.mesh_file, self.centerline_file)

        self.renderer = self.vtk_handler.get_renderer()
        self.renderer.SetBackground(1, 1, 1)
        if self.vtk_widget.GetRenderWindow().GetRenderers().GetNumberOfItems() > 0:
            self.vtk_widget.GetRenderWindow().GetRenderers().RemoveAllItems()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        self.interactor = self.vtk_handler.get_interactor_style()
        self.interactor._main_window = self
        self.vtk_interactor.SetInteractorStyle(self.interactor)
        self.vtk_interactor.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.vtk_widget.GetRenderWindow().Render()

        self.vtk_interactor.Initialize()
        self.vtk_interactor.Start()

        self.interactor.update_deformation_parameters(self._current_sharpness, -self._current_force_scale_raw)

        # Initialize UI styling
        UIStyleManager.set_button_active(self.toggle_camera_lock_button, False)

        # Display debugging information about radius as text in viewport
        self.interactor.display_radius_texts()

    def keyPressEvent(self, event):
        """Handle keyboard events forwarded from the Qt window to the VTK interactor."""
        if not self.interactor:
            return
        if event.key() == Qt.Key.Key_H:
            self.interactor.toggle_roi_cylinder()

    def run_deformation(self):
        """Run kelvinlet-based mesh deformation"""
        if self.vtk_handler is None:
            logger.warning("Please import mesh and centerline files first.")
            return
        force_scale = -self._current_force_scale_raw
        sharpness = self._current_sharpness
        logger.debug(f"Running Kelvinlet deformation with force_scale={force_scale}, sharpness={sharpness}")
        self.interactor.deform_mesh_sequential(sharpness, force_scale)

    def run_deformation_sdf(self):
        """Run SDF-based contact deformation"""
        if self.vtk_handler is None:
            logger.warning("Please import mesh and centerline files first.")
            return
        force_scale = -self._current_force_scale_raw
        logger.debug(f"Running SDF contact deformation with force_scale={force_scale}")
        self.interactor.deform_mesh_sdf_contact(force_scale)

    def run_deformation_simultaneous(self):
        """Run simultaneous parallel mesh deformation"""
        if self.vtk_handler is None:
            logger.warning("Please import mesh and centerline files first.")
            return
        force_scale = -self._current_force_scale_raw
        logger.debug(f"Running straightening deformation with force_scale={force_scale}")
        self.interactor.deform_mesh_with_straightening(force_scale)

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

        force_scale = self._current_force_scale_raw

        logger.debug(
            f"Running stenosis creation with force_scale={force_scale}, stenosis_radius={stenosis_radius}, stenosis_length={stenosis_length}"
        )
        self.interactor.deform_mesh_stenosis(
            force_scale, stenosis_radius, stenosis_length
        )

    def toggle_camera_lock(self):
        """Toggle camera lock and update button styling"""
        if not self.interactor:
            return
        UIStyleManager.toggle_button_style(self.toggle_camera_lock_button)
        self.interactor.toggle_camera_lock()

    def display_centerline_nodes(self):
        """Display centerline nodes for single point selection"""
        if not self.interactor:
            return
        logger.info("Please select a centerline node to generate aneurysm.")
        self.interactor.display_centerline_vertices()

    def render_sdf(self):
        """Display signed distance field visualization"""
        if not self.interactor:
            return
        logger.info("Rendering stent's signed distance field.")
        self.interactor.render_sdf()

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
        if self.interactor:
            self.interactor.save_current_stent()
            logger.info("Stent has been placed.")
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
        if self.interactor:
            self.interactor.update_deformation_parameters(self._current_sharpness, -float_value)

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
        self.force_scale_slider.setValue(new_slider_val)
        if self.interactor:
            self.interactor.update_deformation_parameters(self._current_sharpness, -new_value)

    def save_mesh(self):
        """Save the current mesh to a file"""
        if self.vtk_handler is None:
            logger.warning("No mesh loaded to save.")
            return
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
