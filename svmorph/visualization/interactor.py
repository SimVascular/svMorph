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
    def __init__(self, mesh, centerline, mesh_filename, centerline_filename, mesh_actor, centerline_actor, parent=None):
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("KeyPressEvent", self.key_press_event)
        self.AddObserver("KeyReleaseEvent", self.key_release_event)
        self.AddObserver("TimerEvent", self.timer_callback)
        self.vertexVisualizationActors = []
        self.redHighlightActors = []
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

        self.epsilon = 0.2
        self.force_scale = -1.0
        self.stent_radius = 0.45
        self.stent_length = 1.7
        self.smoothing_k = 0.01
        self.undeployed_stent_radius = 0.05 - self.smoothing_k
        self.current_stent_radius = self.undeployed_stent_radius
        self.stent_unit_section_halflength = 0.2
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
        key = self.GetInteractor().GetKeySym()
        if key == 'h':
            self.toggle_roi_cylinder()
        elif key == 'd':
            self.timer_id = self.GetInteractor().CreateRepeatingTimer(100)
        self.OnKeyPress()

    def timer_callback(self, obj, event):
        self.deform_mesh_sequential(self.epsilon, self.force_scale)

    def key_release_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'd':
            self.GetInteractor().DestroyTimer(self.timer_id)
        self.OnKeyRelease()

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()

        picker = vtkPointPicker()
        picker.SetTolerance(0.01)
        picker.PickFromListOn()
        picker.AddPickList(self.centerline_actor)
        picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        pointID = picker.GetPointId()

        if pointID >= 0:
            logger.info(f"Clicked and selected point ID: {pointID}")
            self.selected_points.append(pointID)
            polydata = self.centerline_actor.GetMapper().GetInput()
            sphere_center = [0.0, 0.0, 0.0]
            polydata.GetPoint(pointID, sphere_center)
            self.operation_count = 0
            self.total_displacement_distance = 0.0
            self.place_highlight_sphere(sphere_center, pointID)
            self.compute_prescribed_stent()
            self.compute_stenosis_minimum_radius_representative(pointID)
            
            if len(self.selected_points) > self.num_kelvinlet_points:
                self.selected_points.pop(0)
                renderer.RemoveActor(self.redHighlightActors[0])
                self.redHighlightActors.pop(0)
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
        polydata = self.centerline_actor.GetMapper().GetInput()
        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()
        logger.debug(f"Number of points in centerline: {num_points}")

        sphereSource = vtkSphereSource()
        sphereSource.SetRadius(0.01)  # Adjust radius as needed.
        sphereSource.SetThetaResolution(8)
        sphereSource.SetPhiResolution(8)
        sphereSource.Update()

        self.glyphMapper = vtkGlyph3DMapper()
        self.glyphMapper.SetSourceConnection(sphereSource.GetOutputPort())
        self.glyphMapper.SetInputData(self.centerline_actor.GetMapper().GetInput())
        self.glyphMapper.ScalingOff()               # uniform size
        self.glyphMapper.SetStatic(1)               # no per-glyph data changes expected

        self.glyphActor = vtkActor()
        self.glyphActor.SetMapper(self.glyphMapper)
        self.glyphActor.GetProperty().SetColor(0.0, 1.0, 1.0)

        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.glyphActor)
        self.mesh_actor.GetProperty().SetOpacity(0.2) # original opacity
        self.centerline_actor.SetPickable(1)
        self.GetInteractor().GetRenderWindow().Render()

    def display_radius_texts(self):
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
        if len(self.selected_points) == 0:
            radius = 0.0
        else:
            pointID = self.selected_points[-1]
            area = self.centerline_section_areas[pointID]
            radius = np.sqrt(area / np.pi)
        self.radius_text_actor.SetInput(f"MIS radius = {self.maximum_inscribed_sphere_radius[pointID]:.4f}, lumen effective radius = {radius:.4f}")
        self.GetInteractor().GetRenderWindow().Render()

    def update_roi_text(self):
        if self.roi_text_actor is None:
            return
        self.roi_text_actor.SetInput(f"stent radius = {self.current_stent_radius + self.smoothing_k:.4f}")

    def compute_prescribed_stent(self):
        segment_length = 0.1 # cm
        foreshortening_percentage = 0.1 # 10%
        deployed_stent_length = self.stent_length * (1 - foreshortening_percentage)
        self.stent_axis_vertices = geometry.resample_stent_axis(self.data["points"]["centerline_points_view_np"], self.parent_tip_map, self.segment_base_mask, self.selected_points[-1], deployed_stent_length, segment_length, sampling_direction=self.sampling_direction)
        logger.debug(f"Num vertices for stent of length {self.stent_length} cm, segment length {segment_length} cm: {len(self.stent_axis_vertices)}")
        self.place_sdf_stent_visualization()
        
    def compute_stenosis_minimum_radius_representative(self, pointID):
        logger.debug(f"Selected stenosis center point ID: {pointID}")
        data_points = self.data["points"]["surface"]
        centerline_points = self.data["points"]["centerline"]
        num_kelvinlet_points = 1
        xs = np.expand_dims(data_points, 1)
        xs = np.tile(xs, (1, num_kelvinlet_points, 1))
        centers = np.expand_dims(np.array([centerline_points[pointID]]), 0)
        force_center_normal = self.centerline_tangents[pointID]
        kelvinlet_points_normals = np.array([force_center_normal])
        rotation_matrices = deformation.compute_householder_matrices(kelvinlet_points_normals)
        original_radius = self.maximum_inscribed_sphere_radius[pointID]
        self.stenosis_minimum_radius_representative, current_radius = deformation.find_stenosis_minimum_radius_representative(data_points, rotation_matrices, xs, centers, original_radius)
        logger.debug(f"Stenosis minimum radius representative index: {self.stenosis_minimum_radius_representative}")
        self.previous_stenosis_minimum_radius = current_radius
        self.previous_aneurysm_maximum_radius = current_radius

    def place_highlight_sphere(self, position, pointID):
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.04)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        actor.GetProperty().SetOpacity(0.8)
        actor.centerpointID = pointID
        actor.sphereSource = sphere 

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)
        self.redHighlightActors.append(actor)

        self.lock_camera(position)
        self.place_radius_of_influence_cylinder(position, pointID)

    def place_sdf_stent_visualization(self):
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
        if len(self.stent_visualization_actors) == 0:
            return
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        last_stent_assembly = self.stent_visualization_actors[-1]
        renderer.AddActor(last_stent_assembly)
        self.stent_visualization_actors.pop(0)
        self.GetInteractor().GetRenderWindow().Render()

    def render_sdf(self):
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
        imageData = vtkImageData()
        imageData.SetDimensions(nx, ny, nz)
        spacing = ((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), (zmax - zmin) / (nz - 1))
        imageData.SetSpacing(spacing)
        imageData.SetOrigin(xmin, ymin, zmin)
        sdf_flat = sdf.ravel(order='F')
        vtk_sdf = numpy_to_vtk(sdf_flat, deep=True, array_type=get_vtk_array_type(np.float32))
        vtk_sdf.SetName("SDF")
        imageData.GetPointData().SetScalars(vtk_sdf)
        mc = vtkMarchingCubes()
        mc.SetInputData(imageData)
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
        
    def place_radius_of_influence_cylinder(self, position, pointID):
        cylinder = vtkCylinderSource()
        cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
        cylinder.SetHeight(2 * self.stent_unit_section_halflength)
        cylinder.SetResolution(100)

        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[pointID]
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
        actor.centerpointID = pointID
        actor.cylinderSource = cylinder 

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)

        actor.SetVisibility(self.roi_visible)
        self.roi_actors.append(actor)

    def place_stent_cylinder(self, position, pointID):
        cylinder = vtkCylinderSource()
        cylinder.SetRadius(self.stent_radius)
        cylinder.SetHeight(2 * self.stent_unit_section_halflength)
        cylinder.SetResolution(100)

        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[pointID]
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
        actor.centerpointID = pointID
        actor.cylinderSource = cylinder 

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)

        actor.SetVisibility(self.roi_visible)
        self.stent_actors.append(actor)

    def update_red_highlight_sphere_position(self, idx):
        pointID = self.selected_points[idx]
        polydata = self.centerline_actor.GetMapper().GetInput()
        sphere_center = [0.0, 0.0, 0.0]
        polydata.GetPoint(pointID, sphere_center)
        
        highlight_actor = self.redHighlightActors[idx]
        highlight_sphere = highlight_actor.sphereSource
        highlight_sphere.SetCenter(sphere_center)

        self.lock_camera(sphere_center)
        if self.operation_count == 0:
            self.place_stent_cylinder(sphere_center, pointID)
        self.operation_count += 1
        if self.operation_count % 5 == 0:
            self.place_stent_cylinder(sphere_center, pointID)

    def update_radius_of_influence_cylinder(self, idx):
        pointID = self.selected_points[idx]
        polydata = self.centerline_actor.GetMapper().GetInput()
        position = [0.0, 0.0, 0.0]
        polydata.GetPoint(pointID, position)
        roi_actor = self.roi_actors[-1]
        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[pointID]
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

    def update_deformation_parameters(self, epsilon, force_scale):
        a = 0.0795774715459 # TODO: dont hard code this lol
        b = 0.0331572798108
        self.epsilon = epsilon
        self.force_scale = force_scale
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            roi_actor.GetProperty().SetOpacity(abs(force_scale) * 0.7)
        self.update_roi_text()
        self.GetInteractor().GetRenderWindow().Render()

    def update_prescribed_stent_radius(self, radius):
        self.stent_radius = radius
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            roi_cylinder.SetRadius(self.stent_radius)
        self.GetInteractor().GetRenderWindow().Render()
    
    def update_prescribed_stent_length(self, length):
        self.stent_length = length
        if len(self.selected_points) == 0:
            return
        self.compute_prescribed_stent()
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        renderer.RemoveActor(self.stent_visualization_actors[0])
        self.stent_visualization_actors.pop(0)
        self.GetInteractor().GetRenderWindow().Render()

    def update_current_stent_radius(self):
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            roi_cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
        for stent_visualization_assembly in self.stent_visualization_actors[-1:]:
            for stent_segment_actor in stent_visualization_assembly.GetParts():
                stent_segment_geometry = stent_segment_actor.geometrySource
                stent_segment_geometry.SetRadius(self.current_stent_radius + self.smoothing_k)
        self.update_roi_text()
    
    def update_current_stent_curvature(self):
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
        self.animation_direction *= -1
        self.sampling_direction *= -1

    def update_force_center_idx(self):
        self.force_center_idx = (self.animation_direction - 1) // 2

    def toggle_roi_cylinder(self):
        self.roi_visible = not self.roi_visible
        for roi_actor in self.roi_actors:
            roi_actor.SetVisibility(not roi_actor.GetVisibility())
        for stent_actor in self.stent_actors:
            stent_actor.SetVisibility(not stent_actor.GetVisibility())
        for stent_visualization_assembly in self.stent_visualization_actors:
            stent_visualization_assembly.SetVisibility(not stent_visualization_assembly.GetVisibility())
        self.GetInteractor().GetRenderWindow().Render()

    def toggle_interleave_mode(self):
        self.interleave_mode = not self.interleave_mode
        self.num_kelvinlet_points = 2 if self.interleave_mode else 1
        if self.interleave_mode:
            self.num_kelvinlet_points = 2
        else:
            self.num_kelvinlet_points = 1
            if len(self.selected_points) > 1:
                self.selected_points.pop(0)
                renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
                renderer.RemoveActor(self.redHighlightActors[0])
                self.redHighlightActors.pop(0)
                renderer.RemoveActor(self.roi_actors[0])
                self.roi_actors.pop(0)
            self.GetInteractor().GetRenderWindow().Render()

    def toggle_camera_lock(self):
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
        if self.camera_lock and not self.interleave_mode:
            camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            self.previous_focal = camera.GetFocalPoint()
            camera.SetFocalPoint(position)
            self.GetInteractor().GetRenderWindow().Render()

    def lock_camera_interleave(self, x, y, z):
        if self.camera_lock:
            camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            self.previous_focal = camera.GetFocalPoint()
            camera.SetFocalPoint(x, y, z)
            self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_sequential(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        num_time_steps = 1
        aneurysm_radius = 0.5 # 20
        if self.previous_aneurysm_maximum_radius >= aneurysm_radius - 2e-3:
            logger.info("Target aneurysm radius reached.")
            return
        displacement_distance = self.run_aneurysm_sequential(
        affine_params, model, self.centerline_filename, self.mesh_filename, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, force_scale, num_time_steps, list_of_node_point_indices, self.stent_unit_section_halflength, self.stent_radius,
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_stent_edge(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        self.run_stent_edge(
        affine_params, model, mu, nu, phi_type, 
        force_center_point_id, force_scale, list_of_node_point_indices, self.animation_direction, self.stent_unit_section_halflength, self.stent_radius)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_sdf_contact(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        num_time_steps = 1
        step_size = self.run_aneurysm_sdf_contact(
        affine_params, model, self.centerline_filename, self.mesh_filename, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, force_scale, num_time_steps, list_of_node_point_indices, self.stent_unit_section_halflength, self.stent_radius,
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)
        self.update_current_stent_radius()
        start_time = time.time()
        self.GetInteractor().GetRenderWindow().Render()
        logger.timing(f"Rendering new frame: {time.time() - start_time:.4f} s")

    def deform_mesh_stenosis(self, force_scale, area_percent_change, stenosis_radius, stenosis_length):
        if len(self.selected_points) < 1:
            logger.warning("Please select 1 point along the centerline.")
            return
        if len(self.selected_points) > 1:
            logger.warning("Using only the most recent point picked.")
            self.selected_points = self.selected_points[-1:]

        if self.previous_stenosis_minimum_radius <= stenosis_radius + 2e-3:
            logger.info("Target stenosis radius reached.")

        self.create_stenosis(self.mesh_filename, self.centerline_filename, self.selected_points, force_scale, area_percent_change, stenosis_radius, stenosis_length)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_with_straightening(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            logger.warning("Please select the distal start of the stent along the centerline.")
            return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        num_time_steps = 1
        displacement_distance = self.run_stent_with_straightening(
        affine_params, model, self.centerline_filename, self.mesh_filename, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, force_scale, num_time_steps, list_of_node_point_indices, self.stent_unit_section_halflength, self.stent_radius,
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)
        self.update_current_stent_radius()
        self.update_current_stent_curvature()
        start_time = time.time()
        self.GetInteractor().GetRenderWindow().Render()
        logger.timing(f"Rendering new frame: {time.time() - start_time:.4f} s")

    def run_aneurysm_sequential(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius, 
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        a, b = mesh_data.compute_material_constants(mu, nu)  # Material properties for Kelvinlet calculations
        logger.timing(f"Setting affine parameters: {time.time() - total_start_time:.4f} s")
        # --- Load Polydata ---

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = deformation.set_node_indices(self.data, node_point_indices)
        simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
        logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_io.get_centerline_point_and_normal(self.centerline, simulation_data["nodes"]["force_center_point_id"])
        logger.timing(f"Getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} s")
        cross_section_time = time.time()
        original_radius = 0.42
        logger.info(f"Cross sectional radius estimated: {original_radius}")
        logger.timing(f"Getting cross sectional radius: {time.time() - cross_section_time:.4f} s")
        get_displacement_time = time.time()
        logger.timing(f"Computing initial radius and displacement: {time.time() - get_displacement_time:.4f} s")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        logger.debug(f"eps={eps}, force_scale={force_scale}")
        surface_displacements, average_displacement_distance = deformation.compute_aneurysm_displacements(simulation_data, a, b, eps, force_scale, None, normal, stent_halflength, stent_radius)
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

    def run_stent_edge(self, affine_params, model, mu, nu, phi_type, force_center_point_id, force_scale, node_point_indices, direction, stent_halflength, stent_radius):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        a, b = mesh_data.compute_material_constants(mu, nu)  # Material properties for Kelvinlet calculations
        logger.timing(f"Setting affine parameters: {time.time() - total_start_time:.4f} s")
        # --- Load Polydata ---
        load_start_time = time.time()
        centerline_polydata = self.centerline  # Loaded from self attributes
        surface_polydata = self.mesh
        logger.timing(f"Reading surface polydata: {time.time() - load_start_time:.4f} s")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = vtk_io.create_data_from_polydata(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = deformation.set_node_indices(simulation_data, node_point_indices)
        simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
        logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_io.get_centerline_point_and_normal(centerline_polydata, simulation_data["nodes"]["force_center_point_id"])
        logger.timing(f"Getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} s")
        cross_section_time = time.time()
        original_radius = 0.42 # TODO, account for this
        logger.info(f"Cross sectional radius estimated: {original_radius}")
        logger.timing(f"Getting cross sectional radius: {time.time() - cross_section_time:.4f} s")
        get_displacement_time = time.time()
        logger.timing(f"Computing initial radius and displacement: {time.time() - get_displacement_time:.4f} s")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        logger.debug(f"eps={eps}, force_scale={force_scale}")
        surface_displacements = deformation.compute_stent_edge_displacements(simulation_data, a, b, eps, force_scale, None, normal, direction, stent_halflength, stent_radius)
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        surface_polydata = vtk_io.sync_polydata(surface_polydata, simulation_data, "surface")
        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        total_simulation_time = time.time() - total_start_time
        logger.timing(f"Total simulation time: {total_simulation_time:.4f} s")
        logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")

    def run_aneurysm_sdf_contact(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius,
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        a, b = mesh_data.compute_material_constants(mu, nu)  # Material properties for Kelvinlet calculations
        logger.timing(f"Setting affine parameters: {time.time() - total_start_time:.4f} s")
        # --- Load Polydata ---
        load_start_time = time.time()
        logger.timing(f"Reading surface polydata: {time.time() - load_start_time:.4f} s")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = deformation.set_node_indices(self.data, node_point_indices) # TODO: this is going to be moved outside to be updated during mouse click selection
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
        logger.debug(f"Main loop eps={eps}, force_scale={force_scale}")
        surface_displacements, centerline_displacements, step_size = deformation.compute_sdf_contact_displacements(simulation_data, a, b, self.stent_axis_vertices, eps, force_scale, None, normal, stent_halflength, stent_radius, self.current_stent_radius) # TODO: remove those arguments that has self since can directly access 
        self.current_stent_radius += step_size
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = mesh_data.apply_displacements(simulation_data, centerline_displacements, "centerline")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        vtk_io.sync_polydata(self.centerline, simulation_data, "centerline")

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        total_simulation_time = time.time() - total_start_time
        logger.timing(f"Total simulation time: {total_simulation_time:.4f} s")
        logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return step_size

    def run_stent_with_straightening(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius, 
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        a, b = mesh_data.compute_material_constants(mu, nu)  # Material properties for Kelvinlet calculations
        logger.timing(f"Setting affine parameters: {time.time() - total_start_time:.4f} s")
        # --- Load Polydata ---
        load_start_time = time.time()
        logger.timing(f"Reading surface polydata: {time.time() - load_start_time:.4f} s")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = deformation.set_node_indices(self.data, node_point_indices) # TODO: this is going to be moved outside to be updated during mouse click selection
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
        logger.debug(f"Main loop eps={eps}, force_scale={force_scale}")
        surface_displacements, centerline_displacements, step_size = deformation.compute_sdf_contact_displacements(simulation_data, a, b, self.stent_axis_vertices, eps, force_scale, None, normal, stent_halflength, stent_radius, self.current_stent_radius) # TODO: remove those arguments that has self since can directly access 
        self.current_stent_radius += step_size
        logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = mesh_data.apply_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = mesh_data.apply_displacements(simulation_data, centerline_displacements, "centerline")
        vtk_io.sync_polydata(self.mesh, simulation_data, "surface")
        vtk_io.sync_polydata(self.centerline, simulation_data, "centerline")

        logger.timing(f"Updating points and polydata: {time.time() - displacement_start_time:.4f} s")
        total_simulation_time = time.time() - total_start_time
        logger.timing(f"Total simulation time: {total_simulation_time:.4f} s")
        logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return step_size

    def create_stenosis(self, mesh_filename, centerline_filename, selected_points, force_scale, area_percent_change, stenosis_radius, stenosis_length, model="test_aneurysm"):
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"

        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []

        force_center_point_id = selected_points[0]
        list_of_node_point_indices = selected_points
        phi_type = "constant"
        affine_params = {"eps": {model: 1.0}, "scale": {model: 1.1}}

        mu = 1
        nu = 0.4
        num_time_steps = 1

        self.run_stenosis(
            affine_params, model, centerline_filename, mesh_filename,
            centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
            force_center_point_id, force_scale, area_percent_change, stenosis_radius, stenosis_length, num_time_steps, list_of_node_point_indices, 
            list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names
        )

    def run_stenosis(self, affine_params, model, centerline_polydata_input_file_name, 
                            surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                            surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, s, area_percent_change, 
                            stenosis_radius, stenosis_length, num_time_steps, node_point_indices,
                            other_geometry_input_files, other_geometry_output_files):
            # --- Initialization and Parameter Setup ---
            total_start_time = time.time()  # Start total timer
            affine_type = "aneurysm"
            a, b = mesh_data.compute_material_constants(mu, nu)  # Material properties for Kelvinlet calculations
            logger.timing(f"Setting affine parameters: {time.time() - total_start_time:.4f} s")
            # --- Load Polydata ---
            load_start_time = time.time()
            logger.timing(f"Reading surface polydata: {time.time() - load_start_time:.4f} s")

            # --- Define Points and Nodes ---
            setup_start_time = time.time()
            other_geometry_polydatas = []  # Placeholder for additional geometries if needed
            simulation_data = deformation.set_node_indices(self.data, node_point_indices) # TODO: this is going to be moved outside to be updated during mouse click selection
            simulation_data = deformation.set_force_center(simulation_data, force_center_point_id)
            logger.timing(f"Converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} s")
            
            # --- Calculate Initial Displacements --- CURENTLY the longest step 
            calc_displacement_start_time = time.time()
            origin, normal = vtk_io.get_centerline_point_and_normal(self.centerline, simulation_data["nodes"]["force_center_point_id"])
            logger.timing(f"Getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} s")
            cross_section_time = time.time()
            original_radius = self.maximum_inscribed_sphere_radius[force_center_point_id]
            logger.timing(f"Getting cross sectional radius: {time.time() - cross_section_time:.4f} s")
            get_displacement_time = time.time()
            logger.timing(f"Computing initial radius and displacement: {time.time() - get_displacement_time:.4f} s")
            
            # --- Compute Initial Force Matrix and Displacements ---
            step_start_time = time.time()
            eps = affine_params["eps"][model]
            logger.debug(f"Main loop eps={eps}, s={s}")
            surface_displacements, step_size = deformation.compute_stenosis_displacements(simulation_data, a, b, eps, s, normal, stenosis_radius, stenosis_length, original_radius)
            logger.timing(f"Affine displacements calculation: {time.time() - step_start_time:.4f} s")
            
            # --- Scale Displacements to Match Desired Area ---
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
            total_simulation_time = time.time() - total_start_time
            logger.timing(f"Total simulation time: {total_simulation_time:.4f} s")
            logger.info(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
            return step_size
