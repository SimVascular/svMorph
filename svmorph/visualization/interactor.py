"""VTK interactor style for interactive Kelvinlet mesh deformation.

Provides :class:`MeshInteractor`, a ``vtkInteractorStyleTrackballCamera``
subclass that handles point selection on the centerline, stent placement
and visualisation, and dispatches the various deformation modes (aneurysm,
stenosis, SDF-contact, straightening).
"""

import vtkmodules.vtkRenderingCore
import vtkmodules.vtkFiltersGeneral
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkCylinderSource
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkPointPicker,
    vtkGlyph3DMapper,
    vtkAssembly
)
from svmorph.core import deformation
from svmorph.core import geometry
from svmorph.core import mesh_data
from svmorph.core.units import L, unit_name
from svmorph.core.defaults import (
    ANEURYSM_MAX_RADIUS_DEFAULT_CM,
    STENT_DIAMETER_DEFAULT_CM,
    STENT_LENGTH_DEFAULT_CM,
    SMOOTHING_K_CM,
    UNDEPLOYED_STENT_DIAMETER_CM,
    STENT_UNIT_SECTION_HALFLENGTH_CM,
    STENT_SEGMENT_LENGTH_CM,
    FORESHORTENING_PERCENTAGE,
    INFLUENCE_RADIUS_CM,
    CONTACT_DISTANCE_CM,
)
from svmorph.visualization import vtk_io
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type
import jax.numpy as jnp
import time
from svmorph.logging import get_logger

logger = get_logger(__name__)


class MeshInteractor(vtkInteractorStyleTrackballCamera):
    """Interactive VTK trackball camera style for Kelvinlet mesh deformation.

    Manages centerline point selection, stent placement/visualisation,
    deformation parameter updates, and dispatches the aneurysm, stenosis,
    SDF-contact, and straightening deformation pipelines.
    """

    def __init__(self, mesh, centerline, mesh_filename, centerline_filename, mesh_actor, centerline_actor, parent=None):
        """Initialise the interactor with mesh and centerline data.

        Parameters
        ----------
        mesh : vtk.vtkPolyData
            Surface mesh polydata.
        centerline : vtk.vtkPolyData
            Centerline polydata.
        mesh_filename : str
            Path to the surface mesh VTP file.
        centerline_filename : str
            Path to the centerline VTP file.
        mesh_actor : vtkActor
            VTK actor for the surface mesh.
        centerline_actor : vtkActor
            VTK actor for the centerline.
        parent : object, optional
            Parent widget (unused, kept for VTK compatibility).
        """
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("KeyPressEvent", self.key_press_event)
        self.AddObserver("KeyReleaseEvent", self.key_release_event)
        self.AddObserver("TimerEvent", self.timer_callback)
        self.vertex_actors = []
        self.highlight_actors = []
        self.mesh = mesh
        self.centerline = centerline
        self.centerline_tangents = vtk_io.extract_centerline_tangents(centerline)
        self.data = vtk_io.extract_mesh_arrays(mesh, centerline)
        self.parent_tip_map, self.segment_base_mask = vtk_io.build_parent_tip_map(centerline)
        self.centerline_section_areas = vtk_io.extract_cross_section_areas(centerline)
        self.maximum_inscribed_sphere_radius = vtk_io.extract_inscribed_sphere_radii(centerline)
        self.mesh_filename = mesh_filename
        self.centerline_filename = centerline_filename
        self.mesh_actor = mesh_actor
        self.centerline_actor = centerline_actor
        self._main_window = None

        self.selected_points = []
        self.select_multiple_points = False
        self.stent_axis_vertices = None
        self.stenosis_minimum_radius_representative = None

        self.sharpness = 1.0
        self.force_scale = -1.0
        self.aneurysm_radius = ANEURYSM_MAX_RADIUS_DEFAULT_CM * L()
        self.stent_radius = (STENT_DIAMETER_DEFAULT_CM / 2) * L()
        self.stent_length = STENT_LENGTH_DEFAULT_CM * L()
        self.smoothing_k = SMOOTHING_K_CM * L()
        self.undeployed_stent_radius = (UNDEPLOYED_STENT_DIAMETER_CM / 2) * L() - self.smoothing_k
        self.current_stent_radius = self.undeployed_stent_radius
        self.stent_unit_section_halflength = STENT_UNIT_SECTION_HALFLENGTH_CM * L()
        self.stent_segment_length = STENT_SEGMENT_LENGTH_CM * L()
        self.foreshortening_percentage = FORESHORTENING_PERCENTAGE
        self.influence_radius = INFLUENCE_RADIUS_CM * L()
        self.contact_distance = CONTACT_DISTANCE_CM * L()
        self.previous_stenosis_minimum_radius = None
        self.previous_aneurysm_maximum_radius = None

        self.stent_visualization_actors = []
        self.glyph_actor = None

        self.camera_lock = False
        self.previous_focal = [0.0, 0.0, 0.0]

    def key_press_event(self, obj, event):
        """Handle key-press events.  'h' toggles stent visibility; 'd' starts a repeating deformation timer."""
        key = self.GetInteractor().GetKeySym()
        if key == 'h':
            self.toggle_stent_visualization()
        elif key == 'd':
            self.timer_id = self.GetInteractor().CreateRepeatingTimer(100)
        self.OnKeyPress()

    def timer_callback(self, obj, event):
        """Repeating timer callback that drives one sequential deformation step."""
        self.deform_mesh_aneurysm(self.sharpness, self.force_scale, self.aneurysm_radius)

    def key_release_event(self, obj, event):
        """Handle key-release events.  Releasing 'd' destroys the deformation timer."""
        key = self.GetInteractor().GetKeySym()
        if key == 'd':
            self.GetInteractor().DestroyTimer(self.timer_id)
        self.OnKeyRelease()

    def left_button_press_event(self, obj, event):
        """Handle left-click: pick a centerline point and update selections and visualisation."""
        click_pos = self.GetInteractor().GetEventPosition()
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()

        picker = vtkPointPicker()
        picker.SetTolerance(0.01 * L())
        picker.PickFromListOn()
        picker.AddPickList(self.centerline_actor)
        picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        point_id = picker.GetPointId()

        if point_id >= 0:
            logger.info(f"Clicked and selected point ID: {point_id}")
            self.selected_points.append(point_id)
            polydata = self.centerline_actor.GetMapper().GetInput()
            sphere_center = [0.0, 0.0, 0.0]
            polydata.GetPoint(point_id, sphere_center)
            self.place_highlight_sphere(sphere_center, point_id)
            self.compute_prescribed_stent()
            self.compute_stenosis_minimum_radius_representative(point_id)

            if len(self.selected_points) > 1:
                self.selected_points.pop(0)
                renderer.RemoveActor(self.highlight_actors[0])
                self.highlight_actors.pop(0)
                if len(self.stent_visualization_actors) >= 2:
                    renderer.RemoveActor(self.stent_visualization_actors[0])
                    self.stent_visualization_actors.pop(0)
                self.current_stent_radius = self.undeployed_stent_radius
                self.update_current_stent_radius()
                self.GetInteractor().GetRenderWindow().Render()

            self.update_selected_point_radius_text()
        self.OnLeftButtonDown()

    def display_centerline_vertices(self):
        """Render a glyph sphere at every centerline vertex and make the mesh translucent."""
        polydata = self.centerline_actor.GetMapper().GetInput()
        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()
        logger.debug(f"# points in centerline: {num_points}")

        sphere_source = vtkSphereSource()
        sphere_source.SetRadius(0.01 * L())
        sphere_source.SetThetaResolution(8)
        sphere_source.SetPhiResolution(8)
        sphere_source.Update()

        self.glyph_mapper = vtkGlyph3DMapper()
        self.glyph_mapper.SetSourceConnection(sphere_source.GetOutputPort())
        self.glyph_mapper.SetInputData(self.centerline_actor.GetMapper().GetInput())
        self.glyph_mapper.ScalingOff()               # uniform size
        self.glyph_mapper.SetStatic(1)               # no per-glyph data changes expected

        self.glyph_actor = vtkActor()
        self.glyph_actor.SetMapper(self.glyph_mapper)
        self.glyph_actor.GetProperty().SetColor(0.0, 1.0, 1.0)

        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.glyph_actor)
        self.mesh_actor.GetProperty().SetOpacity(0.2)
        self.centerline_actor.SetPickable(1)
        self.GetInteractor().GetRenderWindow().Render()

    def update_selected_point_radius_text(self):
        """Show MIS and lumen effective radius in the window title bar."""
        if self._main_window is None:
            return
        if len(self.selected_points) == 0:
            self._main_window.setWindowTitle("svMorph")
            return
        point_id = self.selected_points[-1]
        area = self.centerline_section_areas[point_id]
        lumen_r = np.sqrt(area / np.pi)
        mis_r = self.maximum_inscribed_sphere_radius[point_id]
        u = unit_name()
        self._main_window.setWindowTitle(
            f"svMorph | MIS radius: {mis_r:.4f} {u} | "
            f"Lumen effective radius: {lumen_r:.4f} {u}"
        )

    def compute_prescribed_stent(self):
        """Resample the centerline to produce stent axis vertices and update the visualisation."""
        deployed_stent_length = self.stent_length * (1 - self.foreshortening_percentage)
        self.stent_axis_vertices = geometry.resample_stent_axis(self.data["points"]["centerline_points_view_np"], self.parent_tip_map, self.segment_base_mask, self.selected_points[-1], deployed_stent_length, self.stent_segment_length, sampling_direction=-1)
        logger.debug(f"# vertices for stent of length {self.stent_length} cm, segment length {self.stent_segment_length} cm: {len(self.stent_axis_vertices)}")
        self.place_sdf_stent_visualization()

    def compute_stenosis_minimum_radius_representative(self, point_id):
        """Identify the surface point representing minimum vessel radius at *point_id*."""
        data_points = self.data["points"]["surface"]
        centerline_points = self.data["points"]["centerline"]
        centers = np.array([centerline_points[point_id]])
        force_center_normal = self.centerline_tangents[point_id]
        rotation_matrices = deformation.compute_householder_matrices(np.array([force_center_normal]))
        original_radius = self.maximum_inscribed_sphere_radius[point_id]
        self.stenosis_minimum_radius_representative, current_radius = deformation.find_stenosis_minimum_radius_representative(data_points, rotation_matrices, centers, original_radius)
        logger.debug(f"Stenosis minimum radius representative index: {self.stenosis_minimum_radius_representative}")
        self.previous_stenosis_minimum_radius = current_radius
        self.previous_aneurysm_maximum_radius = current_radius

    def place_highlight_sphere(self, position, point_id):
        """Add a red highlight sphere at *position* and lock the camera to it."""
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.04 * L())

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        actor.GetProperty().SetOpacity(0.8)
        actor.center_point_id = point_id
        actor.sphere_source = sphere

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)
        self.highlight_actors.append(actor)

        self.lock_camera(position)

    def place_sdf_stent_visualization(self):
        """Build and display the cylinder-and-sphere stent visualisation along the axis vertices."""
        if self.stent_axis_vertices is None:
            return
        stent_assembly = vtkAssembly()

        for i in range(len(self.stent_axis_vertices)):
            if i < len(self.stent_axis_vertices) - 1:
                p0 = np.array(self.stent_axis_vertices[i])
                p1 = np.array(self.stent_axis_vertices[i+1])
                diff = p1 - p0
                length = np.linalg.norm(diff)
                if length < 1e-6:
                    continue
                midpoint = (p0 + p1) / 2.0

                default_axis = np.array([0, 1, 0])
                direction = diff / length
                rotation_axis = np.cross(default_axis, direction)
                if np.linalg.norm(rotation_axis) < 1e-6:
                    angle = 0.0
                    rotation_axis = [0, 0, 1]
                else:
                    angle = np.degrees(np.arccos(np.clip(np.dot(default_axis, direction), -1.0, 1.0)))

                cylinder = vtkCylinderSource()
                cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
                cylinder.SetHeight(length)
                cylinder.SetResolution(50)
                transform = vtkTransform()
                transform.Translate(midpoint)
                if angle != 0.0:
                    transform.RotateWXYZ(angle, rotation_axis)
                transform_filter = vtkTransformPolyDataFilter()
                transform_filter.SetInputConnection(cylinder.GetOutputPort())
                transform_filter.SetTransform(transform)
                transform_filter.Update()

                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(transform_filter.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                actor.GetProperty().SetOpacity(0.8)
                actor.geometrySource = cylinder
                stent_assembly.AddPart(actor)

            vertex = self.stent_axis_vertices[i]
            sphere = vtkSphereSource()
            sphere.SetCenter(vertex)
            sphere.SetRadius(self.current_stent_radius + self.smoothing_k)
            sphere.SetThetaResolution(50)
            sphere.SetPhiResolution(50)
            sphere.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 1.0, 1.0)
            actor.GetProperty().SetOpacity(0.0)
            actor.geometrySource = sphere
            stent_assembly.AddPart(actor)

        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        renderer.AddActor(stent_assembly)
        self.stent_visualization_actors.append(stent_assembly)

    def save_current_stent(self):
        """Retain the latest stent visualisation actor and discard older ones."""
        if len(self.stent_visualization_actors) == 0:
            return
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        last_stent_assembly = self.stent_visualization_actors[-1]
        renderer.AddActor(last_stent_assembly)
        self.stent_visualization_actors.pop(0)
        self.GetInteractor().GetRenderWindow().Render()

    def render_sdf(self):
        """Evaluate the stent capsule SDF on a regular grid and render its zero iso-surface."""
        time_start = time.time()
        if self.stent_axis_vertices is None:
            return
        r = self.current_stent_radius + self.smoothing_k            # capsule radius
        r_render = r + 0.1 * L()
        nx, ny, nz = 100, 100, 100
        xmin = jnp.min(self.stent_axis_vertices[:,0]) - r_render
        xmax = jnp.max(self.stent_axis_vertices[:,0]) + r_render
        ymin = jnp.min(self.stent_axis_vertices[:,1]) - r_render
        ymax = jnp.max(self.stent_axis_vertices[:,1]) + r_render
        zmin = jnp.min(self.stent_axis_vertices[:,2]) - r_render
        zmax = jnp.max(self.stent_axis_vertices[:,2]) + r_render
        logger.debug(f"SDF bounds: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}, zmin={zmin}, zmax={zmax}")
        x = jnp.linspace(xmin, xmax, nx)
        y = jnp.linspace(ymin, ymax, ny)
        z = jnp.linspace(zmin, zmax, nz)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        p = jnp.stack((X.ravel(order='F'), Y.ravel(order='F'), Z.ravel(order='F')), axis=1)
        sdf = deformation.capsule_sdf(p, self.stent_axis_vertices, r)
        logger.timing(f"Computing SDF: {time.time() - time_start:.4f} s")
        render_time_start = time.time()
        image_data = vtkImageData()
        image_data.SetDimensions(nx, ny, nz)
        spacing = ((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), (zmax - zmin) / (nz - 1))
        image_data.SetSpacing(spacing)
        image_data.SetOrigin(xmin, ymin, zmin)
        sdf_flat = sdf.ravel(order='F')
        vtk_sdf = numpy_to_vtk(sdf_flat, deep=True, array_type=get_vtk_array_type(np.float32))
        vtk_sdf.SetName("SDF")
        image_data.GetPointData().SetScalars(vtk_sdf)
        mc = vtkMarchingCubes()
        mc.SetInputData(image_data)
        mc.SetValue(0, 0.0)
        mc.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(vtkNamedColors().GetColor3d("Tomato"))
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        renderer.AddActor(actor)
        self.GetInteractor().GetRenderWindow().Render()
        logger.timing(f"Rendering SDF: {time.time() - render_time_start:.4f} s")

    def update_deformation_parameters(self, sharpness, force_scale):
        """Update the Kelvinlet sharpness and force-scale parameters and refresh the display."""
        self.sharpness = sharpness
        self.force_scale = force_scale
        self.GetInteractor().GetRenderWindow().Render()

    def update_prescribed_stent_radius(self, radius):
        """Set the target stent radius."""
        self.stent_radius = radius
        self.GetInteractor().GetRenderWindow().Render()

    def update_prescribed_stent_length(self, length):
        """Set the target stent length, recompute stent axis vertices, and refresh the display."""
        self.stent_length = length
        if len(self.selected_points) == 0:
            return
        self.compute_prescribed_stent()
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.stent_visualization_actors[0])
        self.stent_visualization_actors.pop(0)
        self.GetInteractor().GetRenderWindow().Render()

    def update_current_stent_radius(self):
        """Propagate the current deployment radius to stent visualisation actors."""
        for stent_visualization_assembly in self.stent_visualization_actors[-1:]:
            for stent_segment_actor in stent_visualization_assembly.GetParts():
                stent_segment_geometry = stent_segment_actor.geometrySource
                stent_segment_geometry.SetRadius(self.current_stent_radius + self.smoothing_k)

    def update_current_stent_curvature(self):
        """Straighten the stent axis vertices by projecting toward the start–end line."""
        straightening_strength = 0.075
        def lerp(vertices, start_point, end_point, strength):
            direction = end_point - start_point
            length = jnp.linalg.norm(direction)
            if length < 1e-6:
                return vertices
            direction /= length
            for i in range(len(vertices)):
                point = vertices[i]
                to_start = point - start_point
                projection_length = jnp.dot(to_start, direction)
                closest_point_on_line = start_point + projection_length * direction
                displacement = closest_point_on_line - point
                vertices = vertices.at[i].set(point + strength * displacement)
            return vertices

        start_point = self.stent_axis_vertices[0]
        end_point = self.stent_axis_vertices[-1]
        self.stent_axis_vertices = lerp(self.stent_axis_vertices, start_point, end_point, straightening_strength)
        self.place_sdf_stent_visualization()
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.stent_visualization_actors[0])
        self.stent_visualization_actors.pop(0)

    def toggle_stent_visualization(self):
        """Toggle visibility of the stent visualisation actors."""
        for stent_visualization_assembly in self.stent_visualization_actors:
            stent_visualization_assembly.SetVisibility(not stent_visualization_assembly.GetVisibility())
        self.GetInteractor().GetRenderWindow().Render()

    def toggle_camera_lock(self):
        """Toggle camera focal-point lock to the selected centerline point."""
        if self.camera_lock:
            self.lock_camera(self.previous_focal)
            self.camera_lock = False
        else:
            self.camera_lock = True
            if len(self.selected_points) >= 1:
                self.lock_camera(self.centerline.GetPoint(self.selected_points[-1]))

    def lock_camera(self, position):
        """Set the camera focal point to *position* when camera lock is active."""
        if self.camera_lock:
            camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            self.previous_focal = camera.GetFocalPoint()
            camera.SetFocalPoint(position)
            self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_aneurysm(self, sharpness, force_scale, aneurysm_radius):
        """Run one step of the aneurysm deformation pipeline."""
        # Remember the target so the repeating-timer path continues with the latest value
        self.aneurysm_radius = aneurysm_radius
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        epsilon = 0.08 / sharpness * L()
        force_center_point_id = self.selected_points[0]
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2
        if self.previous_aneurysm_maximum_radius >= aneurysm_radius - 2e-3 * L():
            logger.info("Target aneurysm radius reached.")
            return
        self.run_aneurysm(
            affine_params, model, mu, nu, force_center_point_id, force_scale, self.selected_points)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_stent(self, force_scale):
        """Run one step of the SDF-contact stent deployment deformation pipeline."""
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        if (self.current_stent_radius + self.smoothing_k) >= self.stent_radius - 1e-3 * L():
            logger.info("Target stent diameter reached.")
            return
        force_center_point_id = self.selected_points[0]
        self.run_stent(
            force_center_point_id, force_scale, self.selected_points, self.stent_radius)
        self.update_current_stent_radius()
        start_time = time.time()
        self.GetInteractor().GetRenderWindow().Render()
        logger.timing(f"Rendering new frame: {time.time() - start_time:.4f} s")

    def deform_mesh_stenosis(self, force_scale, stenosis_radius, stenosis_length):
        """Run one step of the stenosis creation deformation pipeline."""
        if len(self.selected_points) < 1:
            logger.warning("Please select 1 point along the centerline.")
            return
        if len(self.selected_points) > 1:
            logger.warning("Using only the most recent point picked.")
            self.selected_points = self.selected_points[-1:]

        if self.previous_stenosis_minimum_radius <= stenosis_radius + 2e-3 * L():
            logger.info("Target stenosis radius reached.")

        force_center_point_id = self.selected_points[0]
        self.run_stenosis(
            force_center_point_id, force_scale, stenosis_radius, stenosis_length, self.selected_points
        )
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_stent_straightening(self, force_scale):
        """Run one step of the SDF-contact stent deployment with stent-axis straightening."""
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        if (self.current_stent_radius + self.smoothing_k) >= self.stent_radius - 1e-3 * L():
            logger.info("Target stent diameter reached.")
            return
        force_center_point_id = self.selected_points[0]
        self.run_stent_straightening(
            force_center_point_id, force_scale, self.selected_points, self.stent_radius)
        self.update_current_stent_radius()
        self.update_current_stent_curvature()
        start_time = time.time()
        self.GetInteractor().GetRenderWindow().Render()
        logger.timing(f"Rendering new frame: {time.time() - start_time:.4f} s")

    def run_aneurysm(self, affine_params, model, mu, nu, force_center_point_id,
                        force_scale, node_point_indices):
        """Execute one aneurysm-inflation time step using scaling Kelvinlets.

        Computes material constants, assembles the displacement field, applies
        it to the surface mesh, and updates the running maximum-radius estimate.
        """
        total_start_time = time.time()  # Start total timer
        a, b = mesh_data.compute_material_constants(mu, nu)  # Material properties for Kelvinlet calculations
        logger.timing(f"Setting affine parameters: {time.time() - total_start_time:.4f} s")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        simulation_data = deformation.set_node_indices(self.data, node_point_indices)
        simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
        logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")

        calc_displacement_start_time = time.time()
        origin, normal = vtk_io.get_centerline_point_and_normal(self.centerline, simulation_data["nodes"]["force_center_point_id"])
        logger.timing(f"Getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} s")
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model]
        logger.debug(f"eps={eps}, force_scale={force_scale}")
        surface_displacements = deformation.compute_aneurysm_displacements(simulation_data, a, b, eps, force_scale, None, normal)
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")

        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        aneurysm_representative = simulation_data['points']['surface'][self.stenosis_minimum_radius_representative]
        selected_point = simulation_data['points']['centerline'][force_center_point_id]
        current_aneurysm_maximum_radius = np.linalg.norm(aneurysm_representative - selected_point)
        logger.debug(f"Current aneurysm maximum radius: {current_aneurysm_maximum_radius} cm")
        logger.debug(f"Delta to previous step: {current_aneurysm_maximum_radius - self.previous_aneurysm_maximum_radius} cm")
        delta_aneurysm = current_aneurysm_maximum_radius - self.previous_aneurysm_maximum_radius
        self.previous_aneurysm_maximum_radius = current_aneurysm_maximum_radius

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        elapsed = max(time.time() - total_start_time, 1e-9)
        logger.timing(f"Total simulation time: {elapsed:.4f} s")
        if self._main_window is not None:
            self._main_window.setWindowTitle(
                f"svMorph | FPS: {int(round(1 / elapsed))} | "
                f"Aneurysm radius: {current_aneurysm_maximum_radius:.4f} {unit_name()} | "
                f"Delta: {delta_aneurysm:.4f} {unit_name()}"
            )

    def run_stent(self, force_center_point_id, force_scale, node_point_indices, stent_radius):
        """Execute one SDF-contact stent deployment time step.

        Computes SDF-contact displacements for both surface and centerline,
        applies them, and increments the current stent radius.

        Returns
        -------
        float
            Step size (stent radius increment) for this iteration.
        """
        total_start_time = time.time()

        setup_start_time = time.time()
        simulation_data = deformation.set_node_indices(self.data, node_point_indices)
        simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
        logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")

        step_start_time = time.time()
        logger.debug(f"SDF contact force_scale={force_scale}")
        surface_displacements, centerline_displacements, step_size = deformation.compute_sdf_contact_displacements(simulation_data, self.stent_axis_vertices, force_scale, stent_radius, self.current_stent_radius, influence_radius=self.influence_radius, contact_distance=self.contact_distance)
        self.current_stent_radius += step_size
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")

        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = mesh_data.apply_displacements(simulation_data, centerline_displacements, "centerline")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        vtk_io.sync_polydata(self.centerline, simulation_data, "centerline")

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        elapsed = max(time.time() - total_start_time, 1e-9)
        logger.timing(f"Total simulation time: {elapsed:.4f} s")
        if self._main_window is not None:
            u = unit_name()
            r = self.current_stent_radius + self.smoothing_k
            d = 2 * r
            self._main_window.setWindowTitle(
                f"svMorph | FPS: {int(round(1 / elapsed))} | "
                f"Stent radius: {r:.4f} {u} | Stent diameter: {d:.4f} {u}"
            )
        return step_size

    def run_stent_straightening(self, force_center_point_id, force_scale, node_point_indices, stent_radius):
        """Execute one SDF-contact deployment step with concurrent stent-axis straightening.

        Identical to :meth:`run_stent` but additionally
        straightens the stent axis after each displacement step.

        Returns
        -------
        float
            Step size (stent radius increment) for this iteration.
        """
        total_start_time = time.time()

        setup_start_time = time.time()
        simulation_data = deformation.set_node_indices(self.data, node_point_indices)
        simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
        logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")

        step_start_time = time.time()
        logger.debug(f"SDF contact with straightening force_scale={force_scale}")
        surface_displacements, centerline_displacements, step_size = deformation.compute_sdf_contact_displacements(simulation_data, self.stent_axis_vertices, force_scale, stent_radius, self.current_stent_radius, influence_radius=self.influence_radius, contact_distance=self.contact_distance)
        self.current_stent_radius += step_size
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")

        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = mesh_data.apply_displacements(simulation_data, centerline_displacements, "centerline")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        vtk_io.sync_polydata(self.centerline, simulation_data, "centerline")

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        elapsed = max(time.time() - total_start_time, 1e-9)
        logger.timing(f"Total simulation time: {elapsed:.4f} s")
        if self._main_window is not None:
            u = unit_name()
            r = self.current_stent_radius + self.smoothing_k
            d = 2 * r
            self._main_window.setWindowTitle(
                f"svMorph | FPS: {int(round(1 / elapsed))} | "
                f"Stent radius: {r:.4f} {u} | Stent diameter: {d:.4f} {u}"
            )
        return step_size

    def run_stenosis(self, force_center_point_id, s, stenosis_radius, stenosis_length, node_point_indices):
        """Execute one stenosis-creation time step using the truncated-sphere warp.

        Computes inward displacements, applies them to the surface mesh,
        and updates the running minimum-radius estimate.

        Returns
        -------
        float
            Step size for this iteration.
        """
        total_start_time = time.time()

        setup_start_time = time.time()
        simulation_data = deformation.set_node_indices(self.data, node_point_indices)
        simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
        logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")

        calc_displacement_start_time = time.time()
        _, normal = vtk_io.get_centerline_point_and_normal(self.centerline, simulation_data["nodes"]["force_center_point_id"])
        logger.timing(f"Getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} s")

        step_start_time = time.time()
        logger.debug(f"Create stenosis s={s}")
        surface_displacements, step_size = deformation.compute_stenosis_displacements(simulation_data, s, normal, stenosis_radius, stenosis_length)
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")

        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        stenosis_representative = simulation_data['points']['surface'][self.stenosis_minimum_radius_representative]
        selected_point = simulation_data['points']['centerline'][force_center_point_id]
        current_stenosis_minimum_radius = np.linalg.norm(stenosis_representative - selected_point)
        logger.debug(f"Current stenosis minimum radius: {current_stenosis_minimum_radius} cm")
        logger.debug(f"Delta to previous step: {current_stenosis_minimum_radius - self.previous_stenosis_minimum_radius} cm")
        delta_stenosis = current_stenosis_minimum_radius - self.previous_stenosis_minimum_radius
        self.previous_stenosis_minimum_radius = current_stenosis_minimum_radius
        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        elapsed = max(time.time() - total_start_time, 1e-9)
        logger.timing(f"Total simulation time: {elapsed:.4f} s")
        if self._main_window is not None:
            self._main_window.setWindowTitle(
                f"svMorph | FPS: {int(round(1 / elapsed))} | "
                f"Stenosis radius: {current_stenosis_minimum_radius:.4f} {unit_name()} | "
                f"Delta: {delta_stenosis:.4f} {unit_name()}"
            )
        return step_size
