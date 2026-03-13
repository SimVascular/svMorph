import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkCylinderSource
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkPointPicker,
    vtkGlyph3DMapper,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkPropPicker,
    vtkAssembly
)
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
from kelvinlet_core import scaling_v2 as scaling
from kelvinlet_core import vtk_utils
from kelvinlet_core import common
from scripts import calculate_radius_of_influence
import numpy as np
import copy
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type
import jax as jx
import jax.numpy as jnp
import time
import numpy as np
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray


def load_vtp_file(filename): # this is a less robust version of vtk_utils.read_polydata_file, TODO: replace usage with vtk_utils.read_polydata_file
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    # segmentation_array = reader.GetOutput().GetCellData().GetArray("ModelFaceID")
    # if segmentation_array is None:
    #     raise ValueError("Segmentation information (SegmentId) not found in .vtp file")
    # data = reader.GetOutput().GetCellData()
    # point_fields = []
    # for i in range(data.GetNumberOfArrays()):
    #     point_fields.append(data.GetArrayName(i))
    # print(f"Point fields: {point_fields}")
    # print(f"Segmentation array: {segmentation_array}")
    return reader.GetOutput()

def write_vtp_file(polydata, filename):
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

class VTKHandler:
    def __init__(self, mesh_filename, centerline_filename):
        self.colors = vtkNamedColors()
        self.mesh = load_vtp_file(mesh_filename)
        self.centerline = load_vtp_file(centerline_filename)
        self.mesh_filename = mesh_filename
        self.centerline_filename = centerline_filename

        self.mesh_mapper = vtkPolyDataMapper()
        self.mesh_mapper.SetInputData(self.mesh)
        self.centerline_mapper = vtkPolyDataMapper()
        self.centerline_mapper.SetInputData(self.centerline)

        self.mesh_actor = vtkActor()
        self.mesh_actor.SetMapper(self.mesh_mapper)
        # self.mesh_actor.GetProperty().SetColor(1.0, 0.8, 0.8)
        self.mesh_actor.GetProperty().SetOpacity(0.8)
        # self.mesh_actor.GetProperty().SetOpacity(0.1)
        self.mesh_actor.SetPickable(0)

        self.centerline_actor = vtkActor()
        self.centerline_actor.SetMapper(self.centerline_mapper)
        self.centerline_actor.SetPickable(0)

        self.renderer = vtkRenderer()
        self.renderer.AddActor(self.mesh_actor)
        self.renderer.AddActor(self.centerline_actor)
        self.renderer.SetBackground(1.0, 1.0, 1.0)

        # temporary code to overlay the reference mesh
        # self.reference_mesh = vtk_utils.read_polydata_file("SU0243-postop-estimated-cm.vtp")
        # self.reference_mesh_mapper = vtkPolyDataMapper()
        # self.reference_mesh_mapper.SetInputData(self.reference_mesh)
        # self.reference_mesh_actor = vtkActor()
        # self.reference_mesh_actor.SetMapper(self.reference_mesh_mapper)
        # self.reference_mesh_actor.GetProperty().SetColor(0.5, 0.0, 0.0)
        # colors = vtkUnsignedCharArray()
        # colors.SetNumberOfComponents(3)
        # colors.SetName("RGB")
        # num_pts = self.reference_mesh.GetNumberOfPoints()
        # for _ in range(num_pts):
        #     colors.InsertNextTuple3(255, 0, 0)
        # self.reference_mesh.GetPointData().SetScalars(colors)
        # self.reference_mesh_actor.GetProperty().SetOpacity(0.3)
        # self.renderer.AddActor(self.reference_mesh_actor)
        # to be deleted later

    def get_renderer(self):
        return self.renderer

    def get_interactor_style(self):
        return MouseInteractorStylePP(self.mesh, self.centerline, self.mesh_filename, self.centerline_filename, self.mesh_actor, self.centerline_actor)

    def save_mesh(self, filename):
        write_vtp_file(self.mesh, filename)

class MouseInteractorStylePP(vtkInteractorStyleTrackballCamera):
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
        self.centerline_tangents = vtk_utils.get_centerline_tangents_np(centerline)
        print(f"Time to get centerline tangents: {time.time() - start_time:.4f} seconds")
        self.data = vtk_utils.polydata_to_np_jnp_data(mesh, centerline)
        self.parent_tip_map, self.segment_base_mask = vtk_utils.polydata_to_parent_tip_map(centerline)
        # print("Should be equal:", np.allclose(self.data["points"]["centerline"], self.data["points"]["centerline_points_view_np"]))
        # print("centerline tangents length: ", len(self.centerline_tangents))
        # print("a few of the entries of centerline tangents: ", self.centerline_tangents[:5])
        self.centerline_section_areas = vtk_utils.get_centerline_cross_section_areas_np(centerline)
        self.maximum_inscribed_sphere_radius = vtk_utils.get_maximum_inscribed_sphere_radius_np(centerline)
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
        self.radius_of_influence = 0.0
        self.stent_radius = 0.45
        self.stent_length = 1.7
        self.smoothing_k = 0.01
        self.undeployed_stent_radius = 0.05 - self.smoothing_k
        # self.undeployed_stent_radius = 0.43
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

    # def left_button_press_event(self, obj, event):
    #     click_pos = self.GetInteractor().GetEventPosition()
    #     picker = self.GetInteractor().GetPicker()
    #     picker.Pick(click_pos[0], click_pos[1], 0, self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
    #     pickedActor = picker.GetActor()

    #     transform = vtkTransform()
    #     actor_matrix = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMatrix()
    #     transform.SetMatrix(actor_matrix)

    #     if pickedActor in self.vertexVisualizationActors:
    #         print(f"Picked actor centerpointID: {pickedActor.centerpointID}")
    #         pointID = pickedActor.centerpointID
    #         self.selected_points.append(pointID)
    #         sphere_center = pickedActor.GetMapper().GetInput().GetCenter()
    #         sphere_center_transformed = transform.TransformPoint(sphere_center)
    #         self.place_highlight_sphere(sphere_center_transformed, pointID)
    #         if len(self.selected_points) > self.num_kelvinlet_points:
    #             self.selected_points.pop(0)
    #             # remove the oldest highlight sphere
    #             self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.redHighlightActors[0])
    #             self.redHighlightActors.pop(0)
    #             self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.roi_actors[0])
    #             self.roi_actors.pop(0)
    #     self.OnLeftButtonDown()

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
            print("Clicked and selected point ID: ", pointID)
            self.selected_points.append(pointID)
            polydata = self.centerline_actor.GetMapper().GetInput()
            sphere_center = [0.0, 0.0, 0.0]
            polydata.GetPoint(pointID, sphere_center)
            self.operation_count = 0
            self.total_displacement_distance = 0.0
            self.place_highlight_sphere(sphere_center, pointID)
            # if self.roi_visible:
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
        print(f"We got here, and Number of points in centerline: {num_points}")

        sphereSource = vtkSphereSource()
        sphereSource.SetRadius(0.01)  # Adjust radius as needed.
        sphereSource.SetThetaResolution(8)
        sphereSource.SetPhiResolution(8)
        sphereSource.Update()

        # glyphFilter = vtkGlyph3D()
        # glyphFilter.SetSourceConnection(sphereSource.GetOutputPort())
        # glyphFilter.SetInputData(polydata)
        # # Disable data-driven scaling if you want uniform sphere sizes.
        # glyphFilter.SetScaleModeToDataScalingOff()
        # glyphFilter.Update()

        # glyphMapper = vtkPolyDataMapper()
        # glyphMapper.SetInputConnection(glyphFilter.GetOutputPort())

        # glyphActor = vtkActor()
        # glyphActor.SetMapper(glyphMapper)
        # glyphActor.GetProperty().SetColor(0.0, 1.0, 0.0)

        # 2. Use the GPU glyph mapper instead of vtkGlyph3D
        self.glyphMapper = vtkGlyph3DMapper()
        self.glyphMapper.SetSourceConnection(sphereSource.GetOutputPort())
        self.glyphMapper.SetInputData(self.centerline_actor.GetMapper().GetInput())
        self.glyphMapper.ScalingOff()               # uniform size
        self.glyphMapper.SetStatic(1)               # no per‐glyph data changes expected
        # optionally tweak:
        # self.glyphMapper.SetResolveCoincidentTopologyToPolygonOffset()
        # self.glyphMapper.SetScalarVisibility(0)

        # 3. Create the actor once
        self.glyphActor = vtkActor()
        self.glyphActor.SetMapper(self.glyphMapper)
        # self.glyphActor.GetProperty().SetColor(0.0, 1.0, 0.0)
        self.glyphActor.GetProperty().SetColor(0.0, 1.0, 1.0)

        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.glyphActor)
        # self.glyph_actor = glyphActor
        self.mesh_actor.GetProperty().SetOpacity(0.2) # original opacity
        # self.mesh_actor.GetProperty().SetOpacity(0.8) # temporarily increased opacity to delete
        self.centerline_actor.SetPickable(1)
        self.GetInteractor().GetRenderWindow().Render()

    def display_radius_texts(self):
        radius = 0.0
        # Create a text actor to display the radius at selected point
        selected_point_text_actor = vtkmodules.vtkRenderingCore.vtkTextActor()
        selected_point_text_actor.SetInput(f"MIS radius = {radius:.4f}, lumen effective radius = {radius:.4f}")
        selected_point_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        selected_point_text_actor.GetTextProperty().SetFontSize(16)
        selected_point_text_actor.SetPosition(10, 28)
        self.radius_text_actor = selected_point_text_actor
        # Create another text actor to display the radius of influence below it
        roi_text_actor = vtkmodules.vtkRenderingCore.vtkTextActor()
        roi_text_actor.SetInput(f"stent radius = {self.current_stent_radius + self.smoothing_k:.4f}")
        # roi_text_actor.SetInput(f"{self.current_stent_radius + self.smoothing_k:.4f}")
        roi_text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        roi_text_actor.GetTextProperty().SetFontSize(16)
        roi_text_actor.SetPosition(10, 4)
        self.roi_text_actor = roi_text_actor
        
        # Add the text actor to the renderer
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
        # self.roi_text_actor.SetInput(f"{self.current_stent_radius + self.smoothing_k:.4f}")

    def compute_prescribed_stent(self):
        segment_length = 0.1 # cm
        # self.stent_axis_vertices = vtk_utils.sample_stent_axis_vertices(self.data["points"]["centerline_points_view_np"], self.parent_tip_map, self.segment_base_mask, self.selected_points[-1], self.stent_length, segment_length, 2*self.stent_radius, sampling_direction=self.sampling_direction)
        foreshortening_percentage = 0.1 # 10%
        # foreshortening_percentage = 0.1032258065 # 0225
        # foreshortening_percentage = 0.135 # 0234
        # foreshortening_percentage = 0.12 # 0235
        deployed_stent_length = self.stent_length * (1 - foreshortening_percentage)
        self.stent_axis_vertices = vtk_utils.sample_stent_axis_vertices_new(self.data["points"]["centerline_points_view_np"], self.parent_tip_map, self.segment_base_mask, self.selected_points[-1], deployed_stent_length, segment_length, sampling_direction=self.sampling_direction)
        print(f"num vertices for stent of length {self.stent_length}cm, segment length {segment_length}cm: {len(self.stent_axis_vertices)}")
        self.place_sdf_stent_visualization()
        
    def compute_stenosis_minimum_radius_representative(self, pointID):
        print("selected stenosis center point ID: ", pointID)
        data_points = self.data["points"]["surface"]
        centerline_points = self.data["points"]["centerline"]
        num_kelvinlet_points = 1
        xs = np.expand_dims(data_points, 1)
        xs = np.tile(xs, (1, num_kelvinlet_points, 1))
        # print("xs shape: ", xs.shape)
        centers = np.expand_dims(np.array([centerline_points[pointID]]), 0)
        force_center_normal = self.centerline_tangents[pointID]
        kelvinlet_points_normals = np.array([force_center_normal])
        rotation_matrices = scaling.compute_householder_matrices(kelvinlet_points_normals)
        original_radius = self.maximum_inscribed_sphere_radius[pointID]
        self.stenosis_minimum_radius_representative = scaling.find_stenosis_minimum_radius_representative(data_points, rotation_matrices, xs, centers, original_radius)
        print(f"Stenosis minimum radius representative index found: {self.stenosis_minimum_radius_representative}")
        self.previous_stenosis_minimum_radius = original_radius
        self.previous_aneurysm_maximum_radius = original_radius

    # def place_visualization_sphere(self, position, pointID):
    #     sphere = vtkSphereSource()
    #     sphere.SetCenter(position)
    #     sphere.SetRadius(0.03)

    #     mapper = vtkPolyDataMapper()
    #     mapper.SetInputConnection(sphere.GetOutputPort())

    #     actor = vtkmodules.vtkRenderingCore.vtkActor()
    #     actor.SetMapper(mapper)
    #     actor.GetProperty().SetColor(0.0, 1.0, 0.0)
    #     actor.centerpointID = pointID
    #     actor.sphereSource = sphere  

    #     ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
    #     ren.AddActor(actor)
    #     self.vertexVisualizationActors.append(actor)

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

    def place_radius_of_influence_sphere(self, position, pointID):
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(self.stent_radius)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        actor.GetProperty().SetOpacity(abs(self.force_scale) * 0.7)
        actor.SetPickable(0)
        actor.centerpointID = pointID
        actor.sphereSource = sphere 

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)
        
        self.roi_actors.append(actor)

    def place_sdf_stent_visualization(self):
        if self.stent_axis_vertices is None:
            return
        # Create a single assembly to group all stent visualization parts
        stent_assembly = vtkAssembly()

        for i in range(len(self.stent_axis_vertices)):
            if i < len(self.stent_axis_vertices) - 1:
                # Get consecutive vertices
                p0 = np.array(self.stent_axis_vertices[i])
                p1 = np.array(self.stent_axis_vertices[i+1])
                diff = p1 - p0
                length = np.linalg.norm(diff)
                if length < 1e-6:
                    continue
                midpoint = (p0 + p1) / 2.0

                # Default cylinder is along (0,1,0)
                default_axis = np.array([0, 1, 0])
                direction = diff / length
                rotation_axis = np.cross(default_axis, direction)
                if np.linalg.norm(rotation_axis) < 1e-6:
                    angle = 0.0
                    rotation_axis = [0, 0, 1]
                else:
                    angle = np.degrees(np.arccos(np.dot(default_axis, direction)))

                # Create cylinder between p0 and p1
                cylinder = vtkCylinderSource()
                cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
                cylinder.SetHeight(length)
                cylinder.SetResolution(50)
                # Create transform: rotate and translate the cylinder so that its center is at midpoint
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
                # actor.GetProperty().SetColor(0.0, 0.0, 1.0)
                # actor.GetProperty().SetOpacity(abs(self.force_scale) * 0.95)
                actor.GetProperty().SetOpacity(0.8)
                actor.geometrySource = cylinder
                # Instead of adding directly to the renderer, add to the assembly
                stent_assembly.AddPart(actor)

            vertex = self.stent_axis_vertices[i]
            # Place a sphere at each vertex with the same radius
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
            # actor.GetProperty().SetOpacity(abs(self.force_scale) * 1)
            actor.geometrySource = sphere
            stent_assembly.AddPart(actor)

        # Add the entire assembly to the renderer and store it for easy removal/recreation.
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        renderer.AddActor(stent_assembly)
        self.stent_visualization_actors.append(stent_assembly)

    def save_current_stent(self):
        if len(self.stent_visualization_actors) == 0:
            return
        # Remove the last stent visualization assembly from the renderer
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        last_stent_assembly = self.stent_visualization_actors[-1]
        # renderer.RemoveActor(last_stent_assembly)
        # Add it back to the renderer to make it persist
        renderer.AddActor(last_stent_assembly)
        # Clear the list so that new stent visualizations can be added without removing this one
        self.stent_visualization_actors.pop(0)
        self.GetInteractor().GetRenderWindow().Render()

    @staticmethod
    @jx.jit
    def capsule_sdf_fast(p, stent_vertices, r):
        ba_all = jnp.diff(stent_vertices, axis=0)
        pa_all = p - stent_vertices[None, :-1, :]
        ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
        ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
        h_all = jnp.clip(ba_dot_pa_all / ba_dot_ba_all, 0, 1)
        axis_to_point_all = pa_all - h_all[:, :, None] * ba_all[None, :, :]
        dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
        direction_all = axis_to_point_all / dist_all
        dist_all_squeezed = jnp.squeeze(dist_all, axis=-1)  # shape: (num_mesh_points, num_segments)
        dist_to_surface_all = dist_all_squeezed - r
        # Vectorize the folding over all mesh points:
        final_dist_to_surface, _ = jx.vmap(scaling.compute_min_dist_and_direction)(dist_to_surface_all, direction_all)
        final_dist_to_surface = final_dist_to_surface[:, None]
        sdf = final_dist_to_surface
        return sdf

    def render_sdf(self):
        time_start = time.time()
        if self.stent_axis_vertices is None:
            return
        r = self.current_stent_radius + self.smoothing_k            # capsule radius
        r_render = r + 0.1
        # Define the sampling grid.
        nx, ny, nz = 100, 100, 100
        xmin = jnp.min(self.stent_axis_vertices[:,0]) - r_render
        xmax = jnp.max(self.stent_axis_vertices[:,0]) + r_render
        ymin = jnp.min(self.stent_axis_vertices[:,1]) - r_render
        ymax = jnp.max(self.stent_axis_vertices[:,1]) + r_render
        zmin = jnp.min(self.stent_axis_vertices[:,2]) - r_render
        zmax = jnp.max(self.stent_axis_vertices[:,2]) + r_render
        print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}, zmin: {zmin}, zmax: {zmax}")
        x = jnp.linspace(xmin, xmax, nx)
        y = jnp.linspace(ymin, ymax, ny)
        z = jnp.linspace(zmin, zmax, nz)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        p = jnp.stack((X.ravel(order='F'), Y.ravel(order='F'), Z.ravel(order='F')), axis=1)
        p = p[:, None, :]
        sdf = self.capsule_sdf_fast(p, self.stent_axis_vertices, r)
        print(f"Time to compute SDF: {time.time() - time_start:.4f} seconds")
        render_time_start = time.time()
        # Create a vtkImageData and populate it with the SDF values.
        imageData = vtkImageData()
        imageData.SetDimensions(nx, ny, nz)
        spacing = ((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), (zmax - zmin) / (nz - 1))
        imageData.SetSpacing(spacing)
        imageData.SetOrigin(xmin, ymin, zmin)
        # Flatten the sdf array in Fortran order(VTK expects Fortran order)
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
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        renderer.AddActor(actor)
        self.GetInteractor().GetRenderWindow().Render() 
        print(f"Time to render SDF: {time.time() - render_time_start:.4f} seconds")
        
    def place_radius_of_influence_cylinder(self, position, pointID):
        cylinder = vtkCylinderSource()
        # cylinder.SetCenter(position)
        # cylinder.SetRadius(self.radius_of_influence)
        cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
        cylinder.SetHeight(2 * self.stent_unit_section_halflength)
        cylinder.SetResolution(100)

        # compute the rotation
        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[pointID]
        rotation_axis = np.cross(default_axis, tangent)
        angle = 180 / np.pi * np.arccos(np.dot(default_axis, tangent))

        # Create a transform to align the cylinder with the vector (1, 2, 3)
        transform = vtkTransform()
        transform.Translate(position)  # Translate to origin
        transform.RotateWXYZ(angle, rotation_axis)  # Rotate about the origin

        transform_filter = vtkmodules.vtkFiltersGeneral.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(cylinder.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        # actor.GetProperty().SetOpacity(abs(self.force_scale) * 0.7)
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
        # cylinder.SetCenter(position)
        # cylinder.SetRadius(self.radius_of_influence)
        cylinder.SetRadius(self.stent_radius)
        cylinder.SetHeight(2 * self.stent_unit_section_halflength)
        cylinder.SetResolution(100)

        # compute the rotation
        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[pointID]
        rotation_axis = np.cross(default_axis, tangent)
        angle = 180 / np.pi * np.arccos(np.dot(default_axis, tangent))
        # print(f"tangent: {tangent}, orientation: {rotation_axis}, angle: {angle}")

        # Create a transform to align the cylinder with the vector (1, 2, 3)
        transform = vtkTransform()
        transform.Translate(position)  # Translate to origin
        transform.RotateWXYZ(angle, rotation_axis)  # Rotate about the origin
        

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

        # self.place_radius_of_influence_sphere(position, pointID)
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
        # compute the rotation
        default_axis = np.array([0, 1, 0])
        tangent = self.centerline_tangents[pointID]
        rotation_axis = np.cross(default_axis, tangent)
        angle = 180 / np.pi * np.arccos(np.dot(default_axis, tangent))
        # create a new transform with the updated rotation and translation
        transform = vtkTransform()
        transform.Translate(position)  # Translate to origin
        transform.RotateWXYZ(angle, rotation_axis)
        # update the mapper with a new transform filter based on the cylinder source
        transform_filter = vtkmodules.vtkFiltersGeneral.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(roi_actor.cylinderSource.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        # new_mapper = vtkPolyDataMapper()
        # new_mapper.SetInputConnection(transform_filter.GetOutputPort())
        # roi_actor.SetMapper(new_mapper)
        roi_actor.GetMapper().SetInputConnection(transform_filter.GetOutputPort())
        self.GetInteractor().GetRenderWindow().Render()

        # # Create a transform to align the cylinder with the vector (1, 2, 3)
        # transform = vtkTransform()
        # transform.Translate(position)  # Translate to origin
        # transform.RotateWXYZ(angle, rotation_axis)  # Rotate about the origin
        

        # transform_filter = vtkmodules.vtkFiltersGeneral.vtkTransformPolyDataFilter()
        # transform_filter.SetInputConnection(cylinder.GetOutputPort())
        # transform_filter.SetTransform(transform)
        # transform_filter.Update()

        # mapper = vtkPolyDataMapper()
        # mapper.SetInputConnection(transform_filter.GetOutputPort())

        # actor = vtkmodules.vtkRenderingCore.vtkActor()
        # actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        # actor.GetProperty().SetOpacity(0.2)
        # actor.SetPickable(0)
        # actor.centerpointID = pointID
        # actor.cylinderSource = cylinder 

        # ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        # ren.AddActor(actor)

        # actor.SetVisibility(self.roi_visible)
        # self.stent_actors.append(actor)

    def update_deformation_parameters(self, epsilon, force_scale):
        a = 0.0795774715459 # TODO: dont hard code this lol
        b = 0.0331572798108
        self.epsilon = epsilon
        self.force_scale = force_scale
        self.radius_of_influence = calculate_radius_of_influence.get_radius_of_influence(a, b, epsilon, force_scale)
        print(f"Updated epsilon: {epsilon}, force_scale: {force_scale}, radius_of_influence: {self.radius_of_influence}")
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            # roi_cylinder.SetRadius(self.radius_of_influence)
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
        # self.current_stent_radius = self.undeployed_stent_radius
        self.GetInteractor().GetRenderWindow().Render()

    def update_current_stent_radius(self):
        for roi_actor in self.roi_actors[-1:]:
            roi_cylinder = roi_actor.cylinderSource
            roi_cylinder.SetRadius(self.current_stent_radius + self.smoothing_k)
        for stent_visualization_assembly in self.stent_visualization_actors[-1:]:
            for stent_segment_actor in stent_visualization_assembly.GetParts():
                # opacity
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

    def update_centerline_vertices(self):
        for highlight_actor in self.redHighlightActors:
            highlight_sphere = highlight_actor.sphereSource
            point_ID = highlight_actor.centerpointID
            new_center = self.centerline.GetPoint(point_ID)
            highlight_sphere.SetCenter(new_center)

    def deform_mesh_parallel(self, epsilon, force_scale):
        # local_area = self.centerline_section_areas[self.selected_points[-1]]
        # local_radius = np.sqrt(local_area / np.pi)
        # if self.total_displacement_distance + local_radius >= self.radius_of_influence - 0.01:
            # print("Prescribed radius reached.")
            # return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        # area_percent_change = 500
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        num_time_steps = 1
        displacement_distance = self.run_aneurysm_parallel(
        affine_params, model, self.centerline_filename, self.mesh_filename, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, force_scale, num_time_steps, list_of_node_point_indices, self.stent_unit_section_halflength, self.stent_radius,
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)
        # self.total_displacement_distance += displacement_distance
        # print(f"Total displacement distance: {self.total_displacement_distance}")
        self.GetInteractor().GetRenderWindow().Render()
    
    def deform_mesh_sequential(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            print("Please select the distal start of the stent along the centerline.")
            return
        # local_area = self.centerline_section_areas[self.selected_points[-1]]
        # local_radius = np.sqrt(local_area / np.pi)
        # if self.total_displacement_distance + local_radius >= self.radius_of_influence - 0.01:
            # print("Prescribed radius reached.")
            # return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        # area_percent_change = 500
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        num_time_steps = 1
        aneurysm_radius = 0.5 # 20
        # force_scale = force_scale * 50000
        if self.previous_aneurysm_maximum_radius >= aneurysm_radius - 2e-3:
            print("Target aneurysm radius reached.")
            return
        displacement_distance = self.run_aneurysm_sequential(
        affine_params, model, self.centerline_filename, self.mesh_filename, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, force_scale, num_time_steps, list_of_node_point_indices, self.stent_unit_section_halflength, self.stent_radius,
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)
        # self.total_displacement_distance += displacement_distance
        # print(f"Total displacement distance: {self.total_displacement_distance}")
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_stent_edge(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            print("Please select the distal start of the stent along the centerline.")
            return
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        # area_percent_change = 500
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        self.run_stent_edge(
        affine_params, model, mu, nu, phi_type, 
        force_center_point_id, force_scale, list_of_node_point_indices, self.animation_direction, self.stent_unit_section_halflength, self.stent_radius)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_sdf(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            print("Please select the distal start of the stent along the centerline.")
            return
        # local_area = self.centerline_section_areas[self.selected_points[-1]]
        # local_radius = np.sqrt(local_area / np.pi)
        # if self.total_displacement_distance + local_radius >= self.radius_of_influence - 0.01:
            # print("Prescribed radius reached.")
            # return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        # area_percent_change = 500
        phi_type = "point"
        model = "test_aneurysm"
        affine_params = {"eps": {model: epsilon}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.2 # (0, 0.5), 0.4 originally
        num_time_steps = 1
        step_size = self.run_aneurysm_sdf(
        affine_params, model, self.centerline_filename, self.mesh_filename, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, force_scale, num_time_steps, list_of_node_point_indices, self.stent_unit_section_halflength, self.stent_radius,
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)
        # self.total_displacement_distance += displacement_distance
        # print(f"Total displacement distance: {self.total_displacement_distance}")
        self.update_current_stent_radius()
        self.GetInteractor().GetRenderWindow().Render()
    
    def deform_mesh_sdf_contact(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            print("Please select the distal start of the stent along the centerline.")
            return
        # local_area = self.centerline_section_areas[self.selected_points[-1]]
        # local_radius = np.sqrt(local_area / np.pi)
        # if self.total_displacement_distance + local_radius >= self.radius_of_influence - 0.01:
            # print("Prescribed radius reached.")
            # return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        # area_percent_change = 500
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
        # self.total_displacement_distance += displacement_distance
        # print(f"Total displacement distance: {self.total_displacement_distance}")
        self.update_current_stent_radius()
        start_time = time.time()
        self.GetInteractor().GetRenderWindow().Render()
        print(f"Time for rendering new frame: {time.time() - start_time:.4f} seconds")

    def deform_mesh_stenosis(self, force_scale, area_percent_change, stenosis_radius, stenosis_length):
        if len(self.selected_points) < 1:
            print("Please select 1 point along the centerline.")
            return
        if len(self.selected_points) > 1:
            print("Using only the most recent point picked.")
            self.selected_points = self.selected_points[-1:]

        if self.previous_stenosis_minimum_radius <= stenosis_radius + 2e-3:
            print("Target stenosis radius reached.")

        self.create_stenosis(self.mesh_filename, self.centerline_filename, self.selected_points, force_scale, area_percent_change, stenosis_radius, stenosis_length)
        self.GetInteractor().GetRenderWindow().Render()

    def deform_mesh_with_straightening(self, epsilon, force_scale):
        if len(self.selected_points) < 1:
            print("Please select the distal start of the stent along the centerline.")
            return
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        force_center_point_id = self.selected_points[self.force_center_idx]
        list_of_node_point_indices = self.selected_points
        # area_percent_change = 500
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
        print(f"Time for rendering new frame: {time.time() - start_time:.4f} seconds")

    def run_aneurysm_parallel(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius, 
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        # nu = 0.1
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        # --- Load Polydata ---
        load_start_time = time.time()
        centerline_polydata = self.centerline  # Loaded from self attributes
        surface_polydata = self.mesh
        print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas) #TODO, compute this jnp array before hand once instead of every time..
        simulation_data = scaling.define_nodes_affine(simulation_data, node_point_indices)
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Initialize Centerline Data ---
        # init_centerline_start_time = time.time()
        # centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(simulation_data, centerline_polydata)
        # print(f"Time for setting initial centerline data: {time.time() - init_centerline_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        # origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, simulation_data["nodes"]["force_center_point_id"])
        normals = jnp.array([vtk_utils.get_normal_at_point_on_centerline(centerline_polydata, i) for i in node_point_indices])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        original_radius = 0.42
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        # vertices = simulation_data["points"]["surface"]
        # original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Cross sectional radius is estimated to be: {original_radius}")
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        # force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        print("eps = ", eps, "force_scale = ", force_scale)
        # , centerline_displacements
        surface_displacements, average_displacement_distance = scaling.get_parallell_displacements(simulation_data, a, b, eps, force_scale, None, normals, stent_halflength, stent_radius)
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        surface_polydata = common.update_polydata_with_points(surface_polydata, simulation_data, "surface")
        # simulation_data = common.update_points_with_displacements(simulation_data, centerline_displacements, "centerline")
        # centerline_polydata = common.update_polydata_with_points(centerline_polydata, simulation_data, "centerline")
        print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
        total_simulation_time = time.time() - total_start_time
        print(f"Total simulation time: {total_simulation_time:.4f} seconds")
        print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return average_displacement_distance

    def run_aneurysm_sequential(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius, 
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        # nu = 0.1
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        # --- Load Polydata ---

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        # simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = scaling.define_nodes_affine(self.data, node_point_indices)
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Initialize Centerline Data ---
        # init_centerline_start_time = time.time()
        # centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(simulation_data, centerline_polydata)
        # print(f"Time for setting initial centerline data: {time.time() - init_centerline_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(self.centerline, simulation_data["nodes"]["force_center_point_id"])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        original_radius = 0.42
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        # vertices = simulation_data["points"]["surface"]
        # original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Cross sectional radius is estimated to be: {original_radius}")
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        # force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        print("eps = ", eps, "force_scale = ", force_scale)
        # , centerline_displacements
        surface_displacements, average_displacement_distance = scaling.get_displacements(simulation_data, a, b, eps, force_scale, None, normal, stent_halflength, stent_radius)
        # surface_displacements = scaling.get_affine_displacements_point(simulation_data, a, b, eps, 1000, None, None, None, normal, stent_halflength)
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        average_displacement_distance = 0
        # self.current_stent_radius += average_displacement_distance
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        common.update_polydata_with_points(self.mesh, simulation_data, "surface")
        # self.mesh = surface_polydata
        # simulation_data = common.update_points_with_displacements(simulation_data, centerline_displacements, "centerline")
        # centerline_polydata = common.update_polydata_with_points(centerline_polydata, simulation_data, "centerline")
        aneurysm_representative = simulation_data['points']['surface'][self.stenosis_minimum_radius_representative]
        selected_point = simulation_data['points']['centerline'][force_center_point_id]
        # print(f"Stenosis representative point: {stenosis_representative}")
        # print(f"Selected point: {selected_point}")
        current_aneurysm_maximum_radius = np.linalg.norm(aneurysm_representative - selected_point)
        print(f"Current aneurysm maximum radius: {current_aneurysm_maximum_radius} cm")
        print(f"Delta to previous step: {current_aneurysm_maximum_radius - self.previous_aneurysm_maximum_radius} cm")
        self.previous_aneurysm_maximum_radius = current_aneurysm_maximum_radius

        print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
        total_simulation_time = time.time() - total_start_time
        print(f"Total simulation time: {total_simulation_time:.4f} seconds")
        print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return average_displacement_distance

    def run_stent_edge(self, affine_params, model, mu, nu, phi_type, force_center_point_id, force_scale, node_point_indices, direction, stent_halflength, stent_radius):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        # nu = 0.1
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        # --- Load Polydata ---
        load_start_time = time.time()
        centerline_polydata = self.centerline  # Loaded from self attributes
        surface_polydata = self.mesh
        print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = scaling.define_nodes_affine(simulation_data, node_point_indices)
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, simulation_data["nodes"]["force_center_point_id"])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        original_radius = 0.42 # TODO, account for this
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        # vertices = simulation_data["points"]["surface"]
        # original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Cross sectional radius is estimated to be: {original_radius}")
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        # force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        print("eps = ", eps, "force_scale = ", force_scale)
        # , centerline_displacements
        surface_displacements = scaling.get_stent_edge_displacements(simulation_data, a, b, eps, force_scale, None, normal, direction, stent_halflength, stent_radius)
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        surface_polydata = common.update_polydata_with_points(surface_polydata, simulation_data, "surface")
        # simulation_data = common.update_points_with_displacements(simulation_data, centerline_displacements, "centerline")
        # centerline_polydata = common.update_polydata_with_points(centerline_polydata, simulation_data, "centerline")
        print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
        total_simulation_time = time.time() - total_start_time
        print(f"Total simulation time: {total_simulation_time:.4f} seconds")
        print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")

    def run_aneurysm_sdf(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius, 
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        # nu = 0.1
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        # --- Load Polydata ---
        load_start_time = time.time()
        centerline_polydata = self.centerline  # Loaded from self attributes
        surface_polydata = self.mesh
        print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        # simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = scaling.define_nodes_affine(self.data, node_point_indices) # TODO: this is going to be moved outside to be updated during mouse click selection
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Initialize Centerline Data ---
        # init_centerline_start_time = time.time()
        # centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(simulation_data, centerline_polydata)
        # print(f"Time for setting initial centerline data: {time.time() - init_centerline_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, simulation_data["nodes"]["force_center_point_id"])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        original_radius = 0.42
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        # vertices = simulation_data["points"]["surface"]
        # original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Cross sectional radius is estimated to be: {original_radius}")
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        # force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        print("eps = ", eps, "force_scale = ", force_scale)
        # , centerline_displacements
        surface_displacements, step_size = scaling.get_sdf_displacements(simulation_data, a, b, self.stent_axis_vertices, eps, force_scale, None, normal, stent_halflength, stent_radius, self.current_stent_radius) # TODO: remove those arguments that has self since can directly access 
        self.current_stent_radius += step_size
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        surface_polydata = common.update_polydata_with_points(surface_polydata, simulation_data, "surface")
        # simulation_data = common.update_points_with_displacements(simulation_data, centerline_displacements, "centerline")
        # centerline_polydata = common.update_polydata_with_points(centerline_polydata, simulation_data, "centerline")
        print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
        total_simulation_time = time.time() - total_start_time
        print(f"Total simulation time: {total_simulation_time:.4f} seconds")
        print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return step_size
    
    def run_aneurysm_sdf_contact(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius,
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        # nu = 0.1
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        # --- Load Polydata ---
        load_start_time = time.time()
        # centerline_polydata = self.centerline  # Loaded from self attributes
        # surface_polydata = self.mesh
        print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        # simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = scaling.define_nodes_affine(self.data, node_point_indices) # TODO: this is going to be moved outside to be updated during mouse click selection
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(self.centerline, simulation_data["nodes"]["force_center_point_id"])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        # vertices = simulation_data["points"]["surface"]
        # original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model]
        # force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        print("Main loop eps = ", eps, "force_scale = ", force_scale)
        # , centerline_displacements
        surface_displacements, centerline_displacements, step_size = scaling.get_sdf_contact_surface_and_centerline_displacements(simulation_data, a, b, self.stent_axis_vertices, eps, force_scale, None, normal, stent_halflength, stent_radius, self.current_stent_radius) # TODO: remove those arguments that has self since can directly access 
        self.current_stent_radius += step_size
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = common.update_points_with_displacements(simulation_data, centerline_displacements, "centerline")
        common.update_polydata_with_points(self.mesh, simulation_data, "surface")
        common.update_polydata_with_points(self.centerline, simulation_data, "centerline")

        print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
        total_simulation_time = time.time() - total_start_time
        print(f"Total simulation time: {total_simulation_time:.4f} seconds")
        print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return step_size

    def run_aneurysm_with_precribed_delta_radius(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, 
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        
        # --- Load Polydata ---
        load_start_time = time.time()
        centerline_polydata = self.centerline  # Loaded from self attributes
        surface_polydata = self.mesh
        print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = scaling.define_nodes_affine(simulation_data, node_point_indices)
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Initialize Centerline Data ---
        init_centerline_start_time = time.time()
        centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(simulation_data, centerline_polydata)
        print(f"Time for setting initial centerline data: {time.time() - init_centerline_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, simulation_data["nodes"]["force_center_point_id"])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        original_radius = 0.42
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        vertices = simulation_data["points"]["surface"]
        original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Cross sectional radius is estimated to be: {original_radius}")
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        # delta_radius = scaling.get_displacement_needed_for_prescribed_displacement(original_radius, area_percent_change, affine_type) / num_time_steps
        delta_radius = 0
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        # force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        # force_scale = -0.25
        print("eps = ", eps, "force_scale = ", force_scale)
        surface_displacements = scaling.get_affine_displacements_point(simulation_data, a, b, eps, force_scale, phi_type, "surface", None, normal, stent_halflength)  # 0 = "surface"
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        surface_polydata = common.update_polydata_with_points(surface_polydata, simulation_data, "surface")
        print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
        total_simulation_time = time.time() - total_start_time
        print(f"Total simulation time: {total_simulation_time:.4f} seconds")
        print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")

    def run_aneurysm_with_precribed_area_ratio(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        area_percent_change, num_time_steps, node_point_indices, 
                        other_geometry_input_files, other_geometry_output_files):
        """
        Run a single-step aneurysm deformation simulation, calculating and applying displacements 
        to a surface mesh based on centerline and scaling parameters.

        Parameters:
            affine_params: Dictionary of affine scaling parameters.
            model: Model identifier for affine_params.
            centerline_polydata_input_file_name: File path for centerline polydata input.
            surface_polydata_input_file_name: File path for surface polydata input.
            centerline_polydata_output_file_name: File path for saving updated centerline polydata.
            surface_polydata_output_file_name: File path for saving updated surface polydata.
            mu, nu: Material parameters for affine scaling.
            phi_type: Type of scaling (e.g., "constant" or "point").
            force_center_point_id: Centerline point ID where the force is applied.
            area_percent_change: Desired percentage change in cross-sectional area.
            num_time_steps: Number of time steps for applying the deformation.
            node_point_indices: List of node indices for displacement assignment.
            other_geometry_input_files: List of additional geometry input file paths.
            other_geometry_output_files: List of corresponding output file paths for additional geometry.
        """
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        
        # --- Load Polydata ---
        load_start_time = time.time()
        centerline_polydata = self.centerline  # Loaded from self attributes
        surface_polydata = self.mesh
        print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = scaling.define_nodes_affine(simulation_data, node_point_indices)
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Initialize Centerline Data ---
        init_centerline_start_time = time.time()
        centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(simulation_data, centerline_polydata)
        print(f"Time for setting initial centerline data: {time.time() - init_centerline_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, simulation_data["nodes"]["force_center_point_id"])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        original_radius = 0.42
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        vertices = simulation_data["points"]["surface"]
        original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Cross sectional radius is estimated to be: {original_radius}")
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        delta_radius = scaling.get_displacement_needed_for_prescribed_displacement(original_radius, area_percent_change, affine_type) / num_time_steps
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model] * original_radius
        force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        surface_displacements = scaling.get_affine_displacements_v2(simulation_data, a, b, eps, force_scale, phi_type, 0, None)  # 0 = "surface"
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        surface_polydata = common.update_polydata_with_points(surface_polydata, simulation_data, "surface")
        tentative_radius = 0.43
        # tentative_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        vertices = simulation_data["points"]["surface"]
        tentative_radius = vtk_utils.estimate_radius_nearby(vertices, sorted_indices, origin, normal)
        print(f"Area percent change: {area_percent_change}")
        print(f"delta_radius needed for prescribed area percent: {delta_radius}")
        print(f"Tentaive radius: {tentative_radius}")
        print(f"Original radius: {original_radius}")
        displacement_norm = tentative_radius - original_radius
        surface_mesh_scale_factor = delta_radius / displacement_norm
        surface_displacements = (surface_mesh_scale_factor - 1) * surface_displacements
        print(f"Time for displacement scaling: {time.time() - displacement_start_time:.4f} seconds")

        # --- Update Mesh Geometry ---
        update_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        surface_polydata = common.update_polydata_with_points(surface_polydata, simulation_data, "surface")
        # current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        print(f"Time for updating geometries: {time.time() - update_start_time:.4f} seconds")
        print(f"Total simulation time: {time.time() - total_start_time:.4f} seconds")

    def run_aneurysm(self, affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, area_percent_change, num_time_steps, list_of_node_point_indices, list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names):
        affine_type = "aneurysm"
        brush_level = "uniscale"
        extension = ".vtp"
        
        a, b = common.get_a_b(mu, nu)
        
        centerline_polydata = self.centerline
        surface_polydata = self.mesh
        surface_polydata_copy = vtk_utils.read_polydata_file(surface_polydata_input_file_name)
        
        other_geometry_polydatas = []
        for other_geometry_polydata_input_file_name in list_of_other_geometry_polydata_input_file_names:
            other_geometry_polydatas.append(vtk_utils.read_polydata_file(other_geometry_polydata_input_file_name))
        
        data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        assert(list_of_node_point_indices is not None)
        data = scaling.define_nodes_affine(data, list_of_node_point_indices)
        assert(force_center_point_id is not None)
        data = scaling.assign_force_location_affine_v2(data, force_center_point_id)
            
        data_surface_copy = {"points" : {"surface" : copy.deepcopy(v2n(surface_polydata_copy.GetPoints().GetData()))}}
        
        centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(data, centerline_polydata)
        
        # vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_run_scale_original" + extension, centerline_polydata)
        
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, data["nodes"]["force_center_point_id"])
        current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        delta_radius = scaling.get_displacement_needed_for_prescribed_displacement(current_radius, area_percent_change, affine_type) / num_time_steps
        
        for it in range(num_time_steps):
            eps = affine_params["eps"][model] * current_radius
            s = scaling.get_force_matrix_scale(affine_params["scale"][model] * current_radius / num_time_steps, a, b)
            
            surface_displacements = scaling.get_affine_displacements_v2(data, a, b, eps, s, phi_type, 0, None) # 0 = "surface"
            
            ###################################
            # get what the resulting radius would be, to determine the approriate scaling factor to achieve the prescribed radius change
            data_surface_copy = common.update_points_with_displacements(data_surface_copy, surface_displacements, "surface")
            surface_polydata_copy = common.update_polydata_with_points(surface_polydata_copy, data_surface_copy, "surface")
            tentative_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata_copy, origin, normal) / np.pi)
            surface_displacements_norm = tentative_radius - current_radius
            surface_mesh_scale_factor = delta_radius / surface_displacements_norm
            surface_displacements *= surface_mesh_scale_factor
            ###################################
            
            # centerline_displacements = get_affine_displacements_v2(data, a, b, eps, s, phi_type, "centerline", surface_mesh_scale_factor)
            
            data = common.update_points_with_displacements(data, surface_displacements, "surface")
            # data = common.update_points_with_displacements(data, centerline_displacements, "centerline")
            
            surface_polydata = common.update_polydata_with_points(surface_polydata, data, "surface")
            # centerline_polydata = common.update_polydata_with_points(centerline_polydata, data, "centerline")
            
            data_surface_copy = {"points" : {"surface" : copy.deepcopy(v2n(surface_polydata.GetPoints().GetData()))}}
            surface_polydata_copy = common.update_polydata_with_points(surface_polydata_copy, data_surface_copy, "surface")
            current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)

    def run_stent_with_straightening(self, affine_params, model, centerline_polydata_input_file_name, 
                        surface_polydata_input_file_name, centerline_polydata_output_file_name, 
                        surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, 
                        force_scale, num_time_steps, node_point_indices, stent_halflength, stent_radius, 
                        other_geometry_input_files, other_geometry_output_files):
        # --- Initialization and Parameter Setup ---
        total_start_time = time.time()  # Start total timer
        affine_type = "aneurysm"
        # nu = 0.1
        a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
        print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
        # --- Load Polydata ---
        load_start_time = time.time()
        # centerline_polydata = self.centerline  # Loaded from self attributes
        # surface_polydata = self.mesh
        print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

        # --- Define Points and Nodes ---
        setup_start_time = time.time()
        other_geometry_polydatas = []  # Placeholder for additional geometries if needed
        # simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        simulation_data = scaling.define_nodes_affine(self.data, node_point_indices) # TODO: this is going to be moved outside to be updated during mouse click selection
        simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
        print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
        
        # --- Calculate Initial Displacements --- CURENTLY the longest step 
        calc_displacement_start_time = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(self.centerline, simulation_data["nodes"]["force_center_point_id"])
        print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
        cross_section_time = time.time()
        # original_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)
        # vertices = simulation_data["points"]["surface"]
        # original_radius, sorted_indices = vtk_utils.estimate_radius(vertices, origin, normal, 1)
        print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
        get_displacement_time = time.time()
        print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
        
        # --- Compute Initial Force Matrix and Displacements ---
        step_start_time = time.time()
        eps = affine_params["eps"][model]
        # force_scale = scaling.get_force_matrix_scale(affine_params["scale"][model] * original_radius / num_time_steps, a, b)
        print("Main loop eps = ", eps, "force_scale = ", force_scale)
        # , centerline_displacements
        surface_displacements, centerline_displacements, step_size = scaling.get_sdf_contact_surface_and_centerline_displacements(simulation_data, a, b, self.stent_axis_vertices, eps, force_scale, None, normal, stent_halflength, stent_radius, self.current_stent_radius) # TODO: remove those arguments that has self since can directly access 
        self.current_stent_radius += step_size
        print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
        
        # --- Scale Displacements to Match Desired Area ---
        displacement_start_time = time.time()
        simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
        simulation_data = common.update_points_with_displacements(simulation_data, centerline_displacements, "centerline")
        common.update_polydata_with_points(self.mesh, simulation_data, "surface")
        common.update_polydata_with_points(self.centerline, simulation_data, "centerline")

        print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
        total_simulation_time = time.time() - total_start_time
        print(f"Total simulation time: {total_simulation_time:.4f} seconds")
        print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
        return step_size

    def update_mesh_viewer(self):
        # updated_mesh = load_vtp_file("obtained_aneurysm_surface_aneurysm_constant_uniscale_1.vtp")
        # updated_centerline = load_vtp_file("obtained_aneurysm_centerline_aneurysm_constant_uniscale_1.vtp")
        # self.mesh_filename = "obtained_aneurysm_surface_aneurysm_constant_uniscale_1.vtp"
        # self.centerline_filename = "obtained_aneurysm_centerline_aneurysm_constant_uniscale_1.vtp"
        
        # self.selected_points = []
        # ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        # for actor in self.redHighlightActors:
        #     ren.RemoveActor(actor)
        # self.redHighlightActors = []
        self.GetInteractor().GetRenderWindow().Render()

        '''
        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.RemoveAllViewProps()

        mesh_mapper = vtkPolyDataMapper()
        mesh_mapper.SetInputData(updated_mesh)
        mesh_actor = vtkActor()
        mesh_actor.SetMapper(mesh_mapper)
        mesh_actor.GetProperty().SetColor(0.8, 1.0, 1.0)
        mesh_actor.GetProperty().SetOpacity(0.6)
        mesh_actor.SetPickable(0)

        centerline_mapper = vtkPolyDataMapper()
        centerline_mapper.SetInputData(updated_centerline)
        centerline_actor = vtkActor()
        centerline_actor.SetMapper(centerline_mapper)

        ren.AddActor(mesh_actor)
        ren.AddActor(centerline_actor)

        self.mesh = updated_mesh
        self.centerline = updated_centerline
        # add temporary file names and make copies as intermediate files
        self.mesh_actor = mesh_actor
        self.selected_points = []
        ren.GetRenderWindow().Render()
        '''

    def create_stenosis(self, mesh_filename, centerline_filename, selected_points, force_scale, area_percent_change, stenosis_radius, stenosis_length, model="test_aneurysm"):
        # centerline_polydata_input_file_name = centerline_filename
        # surface_polydata_input_file_name = mesh_filename
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"

        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []

        # to make sure the selected points are in order regardless of picking order
        # selected_points.sort()
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
            # nu = 0.1
            a, b = common.get_a_b(mu, nu)  # Material properties for Kelvinlet calculations
            print(f"Time for setting affine parameters: {time.time() - total_start_time:.4f} seconds")
            # --- Load Polydata ---
            load_start_time = time.time()
            # centerline_polydata = self.centerline  # Loaded from self attributes
            # surface_polydata = self.mesh
            print(f"Time for reading surface polydata: {time.time() - load_start_time:.4f} seconds")

            # --- Define Points and Nodes ---
            setup_start_time = time.time()
            other_geometry_polydatas = []  # Placeholder for additional geometries if needed
            # simulation_data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
            simulation_data = scaling.define_nodes_affine(self.data, node_point_indices) # TODO: this is going to be moved outside to be updated during mouse click selection
            simulation_data = scaling.assign_force_location_affine_v2(simulation_data, force_center_point_id)
            print(f"Time for converting to jnp arrays and force location: {time.time() - setup_start_time:.4f} seconds")
            
            # --- Calculate Initial Displacements --- CURENTLY the longest step 
            calc_displacement_start_time = time.time()
            origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(self.centerline, simulation_data["nodes"]["force_center_point_id"])
            print(f"Time for getting coordinates and normal: {time.time() - calc_displacement_start_time:.4f} seconds")
            cross_section_time = time.time()
            original_radius = self.maximum_inscribed_sphere_radius[force_center_point_id]
            print(f"Time for getting cross sectional radius: {time.time() - cross_section_time:.4f} seconds")
            get_displacement_time = time.time()
            print(f"Time for computing initial radius and displacement: {time.time() - get_displacement_time:.4f} seconds")
            
            # --- Compute Initial Force Matrix and Displacements ---
            step_start_time = time.time()
            eps = affine_params["eps"][model]
            print("Main loop eps = ", eps, "s = ", s)
            surface_displacements, step_size = scaling.get_stenosis_displacements(simulation_data, a, b, eps, s, normal, stenosis_radius, stenosis_length, original_radius)
            self.current_stent_radius += step_size
            print(f"Time for affine displacements calculation: {time.time() - step_start_time:.4f} seconds")
            
            # --- Scale Displacements to Match Desired Area ---
            displacement_start_time = time.time()
            simulation_data = common.update_points_with_displacements(simulation_data, surface_displacements, "surface")
            # simulation_data = common.update_points_with_displacements(simulation_data, centerline_displacements, "centerline")
            common.update_polydata_with_points(self.mesh, simulation_data, "surface")
            # common.update_polydata_with_points(self.centerline, simulation_data, "centerline")
            stenosis_representative = simulation_data['points']['surface'][self.stenosis_minimum_radius_representative]
            selected_point = simulation_data['points']['centerline'][force_center_point_id]
            # print(f"Stenosis representative point: {stenosis_representative}")
            # print(f"Selected point: {selected_point}")
            current_stenosis_minimum_radius = np.linalg.norm(stenosis_representative - selected_point)
            print(f"Current stenosis minimum radius: {current_stenosis_minimum_radius} cm")
            print(f"Delta to previous step: {current_stenosis_minimum_radius - self.previous_stenosis_minimum_radius} cm")
            self.previous_stenosis_minimum_radius = current_stenosis_minimum_radius
            print(f"Time for updating points and polydata: {time.time() - displacement_start_time:.4f} seconds")
            total_simulation_time = time.time() - total_start_time
            print(f"Total simulation time: {total_simulation_time:.4f} seconds")
            print(f"FPS = {int(round(1 / (time.time() - total_start_time)))}")
            return step_size
