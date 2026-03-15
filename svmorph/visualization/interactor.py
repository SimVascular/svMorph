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
from svmorph.visualization import vtk_io
from svmorph.core import mesh_data
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
        start_time = time.time()
        self.centerline_tangents = vtk_io.extract_centerline_tangents(centerline)
        logger.timing(f"Getting centerline tangents: {time.time() - start_time:.4f} s")
        self.data = vtk_io.extract_mesh_arrays(mesh, centerline)
        self.parent_tip_map, self.segment_base_mask = vtk_io.build_parent_tip_map(centerline)
        self.centerline_section_areas = vtk_io.extract_cross_section_areas(centerline)
        self.maximum_inscribed_sphere_radius = vtk_io.extract_inscribed_sphere_radii(centerline)
        self.mesh_filename = mesh_filename
        self.centerline_filename = centerline_filename
        self.mesh_actor = mesh_actor
        self.centerline_actor = centerline_actor

        self.selected_points = []
        self.select_multiple_points = False
        self.force_center_idx = 0
        self.stent_axis_vertices = None
        self.stenosis_minimum_radius_representative = None

        self.sharpness = 1.0
        self.force_scale = -1.0
        self.stent_radius = 0.45
        self.stent_length = 1.7
        self.smoothing_k = 0.01
        self.undeployed_stent_radius = 0.05 - self.smoothing_k
        self.current_stent_radius = self.undeployed_stent_radius
        self.stent_unit_section_halflength = 0.2
        self.stent_segment_length = 0.1  # cm
        self.foreshortening_percentage = 0.1  # 10%
        self.influence_radius = 0.65  # doi from paper
        self.contact_distance = 0.001  # doc from paper
        self.previous_stenosis_minimum_radius = None
        self.previous_aneurysm_maximum_radius = None

        self.stent_visualization_actors = []
        self.roi_actors = []
        self.stent_actors = []
        self.radius_text_actor = None
        self.roi_text_actor = None
        self.glyph_actor = None
        self.roi_visible = True

        self.sampling_direction = -1
        self.animation_direction = 1
        self.num_kelvinlet_points = 1 # originally 3
        self.interleave_mode = False
        self.operation_count = 0
        self.total_displacement_distance = 0.0
        self.camera_lock = False
        self.previous_focal = [0.0, 0.0, 0.0]
    
    def key_press_event(self, obj, event):
        """Handle key-press events.  'h' toggles ROI visibility; 'd' starts a repeating deformation timer."""
        key = self.GetInteractor().GetKeySym()
        if key == 'h':
            self.toggle_roi_cylinder()
        elif key == 'd':
            self.timer_id = self.GetInteractor().CreateRepeatingTimer(100)
        self.OnKeyPress()

    def timer_callback(self, obj, event):
        """Repeating timer callback that drives one sequential deformation step."""
        self.deform_mesh_sequential(self.sharpness, self.force_scale)

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
        picker.SetTolerance(0.01)
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
            self.operation_count = 0
            self.total_displacement_distance = 0.0
            self.place_highlight_sphere(sphere_center, point_id)
            self.compute_prescribed_stent()
            self.compute_stenosis_minimum_radius_representative(point_id)
            
            if len(self.selected_points) > self.num_kelvinlet_points:
                self.selected_points.pop(0)
                renderer.RemoveActor(self.highlight_actors[0])
                self.highlight_actors.pop(0)
                renderer.RemoveActor(self.roi_actors[0])
                self.roi_actors.pop(0)
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
        logger.debug(f"Number of points in centerline: {num_points}")

        sphere_source = vtkSphereSource()
        sphere_source.SetRadius(0.01)  # Adjust radius as needed.
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
        self.mesh_actor.GetProperty().SetOpacity(0.2) # original opacity
        self.centerline_actor.SetPickable(1)
        self.GetInteractor().GetRenderWindow().Render()

    def display_radius_texts(self):
        """Create and display the on-screen radius and stent-radius text actors."""
        radius = 0.0
        selected_point_text_actor = vtkmodules.vtkRenderingCore.vtkTextActor()
        selected_point_text_actor.SetInput(f"MIS radius = {radius:.4f}, lumen effective radius = {radius:.4f}")
        selected_point_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        selected_point_text_actor.GetTextProperty().SetFontSize(16)
        selected_point_text_actor.SetPosition(10, 28)
        self.radius_text_actor = selected_point_text_actor
        roi_text_actor = vtkmodules.vtkRenderingCore.vtkTextActor()
        roi_text_actor.SetInput(f"stent radius = {self.current_stent_radius + self.smoothing_k:.4f}")
        roi_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        roi_text_actor.GetTextProperty().SetFontSize(16)
        roi_text_actor.SetPosition(10, 4)
        self.roi_text_actor = roi_text_actor
        
        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor2D(self.radius_text_actor)
        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor2D(self.roi_text_actor)
        self.GetInteractor().GetRenderWindow().Render()
        
    def update_selected_point_radius_text(self):
        """Refresh the MIS and lumen effective radius text for the most recently selected point."""
        if len(self.selected_points) == 0:
            radius = 0.0
        else:
            point_id = self.selected_points[-1]
            area = self.centerline_section_areas[point_id]
            radius = np.sqrt(area / np.pi)
        self.radius_text_actor.SetInput(f"MIS radius = {self.maximum_inscribed_sphere_radius[point_id]:.4f}, lumen effective radius = {radius:.4f}")
        self.GetInteractor().GetRenderWindow().Render()

    def update_roi_text(self):
        """Refresh the on-screen stent radius text actor."""
        if self.roi_text_actor is None:
            return
        self.roi_text_actor.SetInput(f"stent radius = {self.current_stent_radius + self.smoothing_k:.4f}")

    def compute_prescribed_stent(self):
        """Resample the centerline to produce stent axis vertices and update the visualisation."""
        deployed_stent_length = self.stent_length * (1 - self.foreshortening_percentage)
        self.stent_axis_vertices = geometry.resample_stent_axis(self.data["points"]["centerline_points_view_np"], self.parent_tip_map, self.segment_base_mask, self.selected_points[-1], deployed_stent_length, self.stent_segment_length, sampling_direction=self.sampling_direction)
        logger.debug(f"Num vertices for stent of length {self.stent_length} cm, segment length {self.stent_segment_length} cm: {len(self.stent_axis_vertices)}")
        self.place_sdf_stent_visualization()
        
    def compute_stenosis_minimum_radius_representative(self, point_id):
        """Identify the surface point representing minimum vessel radius at *point_id*."""
        logger.debug(f"Selected stenosis center point ID: {point_id}")
        data_points = self.data["points"]["surface"]
        centerline_points = self.data["points"]["centerline"]
        num_kelvinlet_points = 1
        xs = np.expand_dims(data_points, 1)
        xs = np.tile(xs, (1, num_kelvinlet_points, 1))
        centers = np.expand_dims(np.array([centerline_points[point_id]]), 0)
        force_center_normal = self.centerline_tangents[point_id]
        kelvinlet_points_normals = np.array([force_center_normal])
        rotation_matrices = deformation.compute_householder_matrices(kelvinlet_points_normals)
        original_radius = self.maximum_inscribed_sphere_radius[point_id]
        self.stenosis_minimum_radius_representative, current_radius = deformation.find_stenosis_minimum_radius_representative(data_points, rotation_matrices, xs, centers, original_radius)
        logger.debug(f"Stenosis minimum radius representative index: {self.stenosis_minimum_radius_representative}")
        self.previous_stenosis_minimum_radius = current_radius
        self.previous_aneurysm_maximum_radius = current_radius

    def place_highlight_sphere(self, position, point_id):
        """Add a red highlight sphere at *position* and lock the camera to it."""
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.04)

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
        self.place_radius_of_influence_cylinder(position, point_id)

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
                    angle = np.degrees(np.arccos(np.dot(default_axis, direction)))

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
        r_render = r + 0.1
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
        p = p[:, None, :]
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
        
    def place_radius_of_influence_cylinder(self, position, point_id):
        """Add a transparent cylinder actor representing the region of influence at *point_id*."""
        cylinder = vtkCylinderSource()
        cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
        cylinder.SetHeight(2 * self.stent_unit_section_halflength)
        cylinder.SetResolution(100)

        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[point_id]
        rotation_axis = np.cross(default_axis, tangent)
        angle = 180 / np.pi * np.arccos(np.dot(default_axis, tangent))

        transform = vtkTransform()
        transform.Translate(position)
        transform.RotateWXYZ(angle, rotation_axis)

        transform_filter = vtkmodules.vtkFiltersGeneral.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(cylinder.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        actor.GetProperty().SetOpacity(0.0)
        actor.SetPickable(0)
        actor.center_point_id = point_id
        actor.cylinderSource = cylinder 

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)

        actor.SetVisibility(self.roi_visible)
        self.roi_actors.append(actor)

    def place_stent_cylinder(self, position, point_id):
        """Add a semi-transparent stent cylinder actor at *position* aligned to the centerline tangent."""
        cylinder = vtkCylinderSource()
        cylinder.SetRadius(self.stent_radius)
        cylinder.SetHeight(2 * self.stent_unit_section_halflength)
        cylinder.SetResolution(100)

        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[point_id]
        rotation_axis = np.cross(default_axis, tangent)
        angle = 180 / np.pi * np.arccos(np.dot(default_axis, tangent))

        transform = vtkTransform()
        transform.Translate(position)
        transform.RotateWXYZ(angle, rotation_axis)

        transform_filter = vtkmodules.vtkFiltersGeneral.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(cylinder.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        actor.GetProperty().SetOpacity(0.2)
        actor.SetPickable(0)
        actor.center_point_id = point_id
        actor.cylinderSource = cylinder 

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)

        actor.SetVisibility(self.roi_visible)
        self.stent_actors.append(actor)

    def update_red_highlight_sphere_position(self, idx):
        """Move the highlight sphere for selection *idx* to the current centerline position."""
        point_id = self.selected_points[idx]
        polydata = self.centerline_actor.GetMapper().GetInput()
        sphere_center = [0.0, 0.0, 0.0]
        polydata.GetPoint(point_id, sphere_center)
        
        highlight_actor = self.highlight_actors[idx]
        highlight_sphere = highlight_actor.sphere_source
        highlight_sphere.SetCenter(sphere_center)

        self.lock_camera(sphere_center)
        if self.operation_count == 0:
            self.place_stent_cylinder(sphere_center, point_id)
        self.operation_count += 1
        if self.operation_count % 5 == 0:
            self.place_stent_cylinder(sphere_center, point_id)

    def update_radius_of_influence_cylinder(self, idx):
        """Reposition and reorient the ROI cylinder for selection *idx*."""
        point_id = self.selected_points[idx]
        polydata = self.centerline_actor.GetMapper().GetInput()
        position = [0.0, 0.0, 0.0]
        polydata.GetPoint(point_id, position)
        roi_actor = self.roi_actors[-1]
        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[point_id]
        rotation_axis = np.cross(default_axis, tangent)
        angle = 180 / np.pi * np.arccos(np.dot(default_axis, tangent))
        transform = vtkTransform()
        transform.Translate(position)
        transform.RotateWXYZ(angle, rotation_axis)
        transform_filter = vtkmodules.vtkFiltersGeneral.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(roi_actor.cylinderSource.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        roi_actor.GetMapper().SetInputConnection(transform_filter.GetOutputPort())
        self.GetInteractor().GetRenderWindow().Render()

    def update_deformation_parameters(self, sharpness, force_scale):
        """Update the Kelvinlet sharpness and force-scale parameters and refresh the display."""
        self.sharpness = sharpness
        self.force_scale = force_scale
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            roi_actor.GetProperty().SetOpacity(abs(force_scale) * 0.7)
        self.update_roi_text()
        self.GetInteractor().GetRenderWindow().Render()

    def update_prescribed_stent_radius(self, radius):
        """Set the target stent radius and update the ROI cylinder display."""
        self.stent_radius = radius
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            roi_cylinder.SetRadius(self.stent_radius)
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
        """Propagate the current deployment radius to ROI and stent visualisation actors."""
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            roi_cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
        for stent_visualization_assembly in self.stent_visualization_actors[-1:]:
            for stent_segment_actor in stent_visualization_assembly.GetParts():
                stent_segment_geometry = stent_segment_actor.geometrySource
                stent_segment_geometry.SetRadius(self.current_stent_radius + self.smoothing_k)
        self.update_roi_text()
    
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


    def update_selected_point(self):
        """Advance the single selected point along the centerline and refresh visual elements."""
        if len(self.selected_points) == 0:
            return
        self.selected_points[0] += self.animation_direction * 5
        if self.selected_points[0] >= self.centerline.GetNumberOfPoints():
            self.selected_points[0] = 0
        elif self.selected_points[0] < 0:
            self.selected_points[0] = self.centerline.GetNumberOfPoints() - 1
        self.update_red_highlight_sphere_position(0)
        self.update_radius_of_influence_cylinder(0)

    
    def interleave_update_selected_points(self):
        """Advance the active interleave point along the centerline, alternating direction."""
        if len(self.selected_points) == 0:
            return
        
        self.update_force_center_idx()
        self.selected_points[self.force_center_idx] += self.animation_direction * 2
        self.reverse_animation_direction()
        
        if self.selected_points[self.force_center_idx] >= self.centerline.GetNumberOfPoints():
            self.selected_points[self.force_center_idx] = 0
        elif self.selected_points[self.force_center_idx] < 0:
            self.selected_points[self.force_center_idx] = self.centerline.GetNumberOfPoints() - 1
        
        self.update_red_highlight_sphere_position(self.force_center_idx)

    def reverse_animation_direction(self):
        """Flip both the animation and sampling direction signs."""
        self.animation_direction *= -1
        self.sampling_direction *= -1

    def update_force_center_idx(self):
        """Set the force-center index to 0 or 1 based on the current animation direction."""
        self.force_center_idx = (self.animation_direction - 1) // 2

    def toggle_roi_cylinder(self):
        """Toggle visibility of the ROI cylinders, stent cylinders, and stent visualisation actors."""
        self.roi_visible = not self.roi_visible
        for roi_actor in self.roi_actors:
            roi_actor.SetVisibility(not roi_actor.GetVisibility())
        for stent_actor in self.stent_actors:
            stent_actor.SetVisibility(not stent_actor.GetVisibility())
        for stent_visualization_assembly in self.stent_visualization_actors:
            stent_visualization_assembly.SetVisibility(not stent_visualization_assembly.GetVisibility())
        self.GetInteractor().GetRenderWindow().Render()

    def toggle_interleave_mode(self):
        """Toggle interleave mode between single- and dual-point selection."""
        self.interleave_mode = not self.interleave_mode
        self.num_kelvinlet_points = 2 if self.interleave_mode else 1
        if not self.interleave_mode:
            if len(self.selected_points) > 1:
                self.selected_points.pop(0)
                renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
                renderer.RemoveActor(self.highlight_actors[0])
                self.highlight_actors.pop(0)
                renderer.RemoveActor(self.roi_actors[0])
                self.roi_actors.pop(0)
            self.GetInteractor().GetRenderWindow().Render()

    def toggle_camera_lock(self):
        """Toggle camera focal-point lock to the selected centerline point(s)."""
        if self.camera_lock:
            self.lock_camera(self.previous_focal)
            self.camera_lock = False
        else:
            self.camera_lock = True
            if len(self.selected_points) == 1:
                self.lock_camera(self.centerline.GetPoint(self.selected_points[-1]))
            elif self.interleave_mode and len(self.selected_points) >= 2:
                mid_point = [(self.centerline.GetPoint(self.selected_points[0])[i] + self.centerline.GetPoint(self.selected_points[-1])[i]) / 2.0 for i in range(3)]
                self.lock_camera_interleave(mid_point[0], mid_point[1], mid_point[2])

    def lock_camera(self, position):
        """Set the camera focal point to *position* when camera lock is active (non-interleave mode)."""
        if self.camera_lock and not self.interleave_mode:
            camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            self.previous_focal = camera.GetFocalPoint()
            camera.SetFocalPoint(position)
            self.GetInteractor().GetRenderWindow().Render()

    def lock_camera_interleave(self, x, y, z):
        """Set the camera focal point to *(x, y, z)* when camera lock is active (interleave mode)."""
        if self.camera_lock:
            camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            self.previous_focal = camera.GetFocalPoint()
            camera.SetFocalPoint(x, y, z)
            self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_sequential(self, sharpness, force_scale):
        """Run one step of the aneurysm sequential deformation pipeline."""
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        epsilon = 0.08 * sharpness
        force_center_point_id = self.selected_points[self.force_center_idx]
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2
        aneurysm_radius = 0.5
        if self.previous_aneurysm_maximum_radius >= aneurysm_radius - 2e-3:
            logger.info("Target aneurysm radius reached.")
            return
        self.run_aneurysm_sequential(
            affine_params, model, mu, nu, force_center_point_id, force_scale, self.selected_points)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_sdf_contact(self, force_scale):
        """Run one step of the SDF-contact stent deployment deformation pipeline."""
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        force_center_point_id = self.selected_points[self.force_center_idx]
        self.run_aneurysm_sdf_contact(
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

        if self.previous_stenosis_minimum_radius <= stenosis_radius + 2e-3:
            logger.info("Target stenosis radius reached.")

        self.create_stenosis(self.selected_points, force_scale, stenosis_radius, stenosis_length)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_with_straightening(self, force_scale):
        """Run one step of the SDF-contact deformation with stent-axis straightening."""
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        force_center_point_id = self.selected_points[self.force_center_idx]
        self.run_stent_with_straightening(
            force_center_point_id, force_scale, self.selected_points, self.stent_radius)
        self.update_current_stent_radius()
        self.update_current_stent_curvature()
        start_time = time.time()
        self.GetInteractor().GetRenderWindow().Render()
        logger.timing(f"Rendering new frame: {time.time() - start_time:.4f} s")

    def run_aneurysm_sequential(self, affine_params, model, mu, nu, force_center_point_id, 
                        force_scale, node_point_indices):
        """Execute one aneurysm-inflation time step using scaling Kelvinlets.

        Computes material constants, assembles the displacement field, applies
        it to the surface mesh, and updates the running maximum-radius estimate.

        Returns
        -------
        float
            Average displacement distance for this step.
        """
        total_start_time = time.time()  # Start total timer
        a, b = mesh_data.compute_material_constants(mu, nu)  # Material properties for Kelvinlet calculations
        logger.timing(f"Setting affine parameters: {time.time() - total_start_time:.4f} s")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        simulation_data = deformation.set_node_indices(self.data, node_point_indices)
        simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
        logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_io.get_centerline_point_and_normal(self.centerline, simulation_data["nodes"]["force_center_point_id"])
        logger.timing(f"Getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} s")
        cross_section_time = time.time()
        logger.timing(f"Getting cross sectional radius: {time.time() - cross_section_time:.4f} s")
        get_displacement_time = time.time()
        logger.timing(f"Computing initial radius and displacement: {time.time() - get_displacement_time:.4f} s")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model]
        logger.debug(f"eps={eps}, force_scale={force_scale}")
        surface_displacements, average_displacement_distance = deformation.compute_aneurysm_displacements(simulation_data, a, b, eps, force_scale, None, normal)
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
        average_displacement_distance = 0
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        aneurysm_representative = simulation_data['points']['surface'][self.stenosis_minimum_radius_representative]
        selected_point = simulation_data['points']['centerline'][force_center_point_id]
        current_aneurysm_maximum_radius = np.linalg.norm(aneurysm_representative - selected_point)
        logger.info(f"Current aneurysm maximum radius: {current_aneurysm_maximum_radius} cm")
        logger.info(f"Delta to previous step: {current_aneurysm_maximum_radius - self.previous_aneurysm_maximum_radius} cm")
        self.previous_aneurysm_maximum_radius = current_aneurysm_maximum_radius

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        total_simulation_time = time.time() - total_start_time
        logger.timing(f"Total simulation time: {total_simulation_time:.4f} s")
        logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return average_displacement_distance

    def run_aneurysm_sdf_contact(self, force_center_point_id, force_scale, node_point_indices, stent_radius):
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
        logger.debug(f"Main loop force_scale={force_scale}")
        surface_displacements, centerline_displacements, step_size = deformation.compute_sdf_contact_displacements(simulation_data, self.stent_axis_vertices, force_scale, stent_radius, self.current_stent_radius, influence_radius=self.influence_radius, contact_distance=self.contact_distance)
        self.current_stent_radius += step_size
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
        
        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = mesh_data.apply_displacements(simulation_data, centerline_displacements, "centerline")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        vtk_io.sync_polydata(self.centerline, simulation_data, "centerline")

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        logger.timing(f"Total simulation time: {time.time() - total_start_time:.4f} s")
        logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return step_size

    def run_stent_with_straightening(self, force_center_point_id, force_scale, node_point_indices, stent_radius):
        """Execute one SDF-contact deployment step with concurrent stent-axis straightening.

        Identical to :meth:`run_aneurysm_sdf_contact` but additionally
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
        logger.debug(f"Main loop force_scale={force_scale}")
        surface_displacements, centerline_displacements, step_size = deformation.compute_sdf_contact_displacements(simulation_data, self.stent_axis_vertices, force_scale, stent_radius, self.current_stent_radius, influence_radius=self.influence_radius, contact_distance=self.contact_distance)
        self.current_stent_radius += step_size
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
        
        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = mesh_data.apply_displacements(simulation_data, centerline_displacements, "centerline")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        vtk_io.sync_polydata(self.centerline, simulation_data, "centerline")

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        logger.timing(f"Total simulation time: {time.time() - total_start_time:.4f} s")
        logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return step_size

    def create_stenosis(self, selected_points, force_scale, stenosis_radius, stenosis_length):
        """Prepare parameters and delegate to :meth:`run_stenosis` for one stenosis step."""
        force_center_point_id = selected_points[0]
        self.run_stenosis(
            force_center_point_id, force_scale, stenosis_radius, stenosis_length, selected_points
        )

    def run_stenosis(self, force_center_point_id, s,
                            stenosis_radius, stenosis_length, node_point_indices):
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
            logger.debug(f"Main loop s={s}")
            surface_displacements, step_size = deformation.compute_stenosis_displacements(simulation_data, s, normal, stenosis_radius, stenosis_length)
            logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
            
            displacement_start_time = time.time()
            simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
            vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
            stenosis_representative = simulation_data['points']['surface'][self.stenosis_minimum_radius_representative]
            selected_point = simulation_data['points']['centerline'][force_center_point_id]
            current_stenosis_minimum_radius = np.linalg.norm(stenosis_representative - selected_point)
            logger.info(f"Current stenosis minimum radius: {current_stenosis_minimum_radius} cm")
            logger.info(f"Delta to previous step: {current_stenosis_minimum_radius - self.previous_stenosis_minimum_radius} cm")
            self.previous_stenosis_minimum_radius = current_stenosis_minimum_radius
            logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
            logger.timing(f"Total simulation time: {time.time() - total_start_time:.4f} s")
            logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
            return step_size
