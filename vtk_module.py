import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkPropPicker
)
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
from kelvinlet_core import scaling
from kelvinlet_core import vtk_utils
from kelvinlet_core import common
import numpy as np
import copy
from vtk.util.numpy_support import vtk_to_numpy as v2n
import jax as jx
import jax.numpy as jnp
import time

def load_vtp_file(filename): # this is a less robust version of vtk_utils.read_polydata_file, TODO: replace usage with vtk_utils.read_polydata_file
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
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
        self.mesh_actor.GetProperty().SetColor(0.8, 1.0, 1.0)
        self.mesh_actor.GetProperty().SetOpacity(0.6)
        self.mesh_actor.SetPickable(0)

        self.centerline_actor = vtkActor()
        self.centerline_actor.SetMapper(self.centerline_mapper)

        self.renderer = vtkRenderer()
        self.renderer.AddActor(self.mesh_actor)
        self.renderer.AddActor(self.centerline_actor)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

    def get_renderer(self):
        return self.renderer

    def get_interactor_style(self, interactor):
        return MouseInteractorStylePP(self.mesh, self.centerline, self.mesh_filename, self.centerline_filename, self.mesh_actor, interactor)

class MouseInteractorStylePP(vtkInteractorStyleTrackballCamera):
    def __init__(self, mesh, centerline, mesh_filename, centerline_filename, mesh_actor, interactor, parent=None):
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.Points = vtkmodules.vtkCommonCore.vtkPoints()
        self.vertexVisualizationActors = []
        self.redHighlightActors = []
        self.mesh = mesh
        self.centerline = centerline
        self.mesh_filename = mesh_filename
        self.centerline_filename = centerline_filename
        self.interactor = interactor
        self.mesh_actor = mesh_actor
        self.selected_points = []

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'h':
            self.display_vertices()
        elif key == 'd':
            self.deform_mesh()
        self.OnKeyPress()

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        picker = self.GetInteractor().GetPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
        pickedActor = picker.GetActor()

        transform = vtkTransform()
        actor_matrix = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMatrix()
        transform.SetMatrix(actor_matrix)

        if pickedActor in self.vertexVisualizationActors:
            print(f"Picked actor centerpointID: {pickedActor.centerpointID}")
            pointID = pickedActor.centerpointID
            self.selected_points.append(pointID)
            sphere_center = pickedActor.GetMapper().GetInput().GetCenter()
            sphere_center_transformed = transform.TransformPoint(sphere_center)
            self.place_highlight_sphere(sphere_center_transformed)

        self.OnLeftButtonDown()

    def place_visualization_sphere(self, position, pointID):
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.03)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        actor.centerpointID = pointID

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)
        self.vertexVisualizationActors.append(actor)

    def place_highlight_sphere(self, position):
        sphere = vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.06)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkmodules.vtkRenderingCore.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        actor.GetProperty().SetOpacity(0.8)

        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        ren.AddActor(actor)
        self.redHighlightActors.append(actor)

    def jit_warm_up(self):
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        selected_points = [380, 400, 420]
        force_center_point_id = selected_points[1]
        list_of_node_point_indices = [force_center_point_id]
        model="test_stenosis"
        affine_params = {"eps": {model: 1.0}}
        mu = 1
        nu = 0.1
        num_time_steps = 1
        start_time = time.time()
        # Calculate `a` and `b` using material properties
        a, b = common.get_a_b(mu, nu)
        print(f"Time to get a and b: {time.time() - start_time:.4f} seconds")
        # Read centerline and surface polydata files
        centerline_polydata = self.centerline
        surface_polydata = self.mesh

        other_geometry_polydatas = []
        # Define affine points and nodes
        start = time.time()
        data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        data = scaling.define_nodes_affine(data, list_of_node_point_indices)
        print(f"Time to define affine points and nodes: {time.time() - start:.4f} seconds")
        # Assign force location
        start = time.time()
        data = scaling.assign_force_location_affine_v2(data, force_center_point_id)
        print(f"Time to assign force location: {time.time() - start:.4f} seconds")
        # Add node data to centerline
        start = time.time()
        centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(data, centerline_polydata)
        print(f"Time to add node data to centerline: {time.time() - start:.4f} seconds")
        # Calculate origin, normal, original area, and target area
        start = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, data["nodes"]["force_center_point_id"])
        original_area = vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal)
        target_area = original_area * 5 / 100
        delta_area = (target_area - original_area) / num_time_steps
        print(f"Time to calculate origin, normal, and target area: {time.time() - start:.4f} seconds")
        # Calculate current radius and `eps`
        start = time.time()
        current_radius = jnp.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / jnp.pi)
        eps = affine_params["eps"][model] * current_radius
        print(f"Time to calculate current radius and eps: {time.time() - start:.4f} seconds")
        # Precompile Kelvinlet translation function
        start = time.time()
        kelvinlets_translation_jit = jx.jit(common.kelvinlets_translation_v2_jax)
        kelvinlets_translation_jit_non_blocky = jx.jit(common.kelvinlets_translation_v2_jax_non_blocky)
        print(f"Time to precompile the two kelvinlets_translation_v2_jax functions: {time.time() - start:.4f} seconds")
        # Calculate ring points and forces
        num_ring_points = 35
        start = time.time()
        ring_points, ring_forces = scaling.get_ring_point_and_forces_v2(
            data, a, b, eps, jnp.array(origin), normal, surface_polydata, 
            num_ring_points, original_area + delta_area, "regular", kelvinlets_translation_jit
        )
        print(f"WARM UP Time to calculate ring points and forces: {time.time() - start:.4f} seconds")
        start = time.time()
        surface_displacements = scaling.get_ring_displacements_v2(data, a, b, eps, "surface", ring_points, ring_forces, "regular", kelvinlets_translation_jit_non_blocky)
        print(f"Time to calculate surface displacements: {time.time() - start:.4f} seconds")
        data = common.update_points_with_displacements(data, surface_displacements, "surface")
        # surface_polydata = common.update_polydata_with_points(surface_polydata, data, "surface")

    def display_vertices(self):
        points = self.centerline.GetPoints()
        transform = vtkTransform()
        actor_matrix = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActors().GetLastActor().GetMatrix()
        transform.SetMatrix(actor_matrix)

        for i in range(points.GetData().GetNumberOfTuples()):
            point = [0.0, 0.0, 0.0]
            points.GetPoint(i, point)
            transformed_point = transform.TransformPoint(point)
            self.place_visualization_sphere(transformed_point, i)

        self.mesh_actor.GetProperty().SetOpacity(0.3)
        self.GetInteractor().GetRenderWindow().Render()
        self.jit_warm_up()

    def deform_mesh(self, area_percent_change):
        if len(self.selected_points) < 3:
            print("Please select at least 3 points along the centerline.")
            return
        if len(self.selected_points) > 3:
            print("Using only the most recent 3 points picked.")
            self.selected_points = self.selected_points[-3:]

        # create_aneurysm(self.mesh_filename, self.centerline_filename, self.selected_points, area_percent_change)
        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        # to make sure the selected points are in order regardless of picking order
        self.selected_points.sort()
        force_center_point_id = self.selected_points[1]
        list_of_node_point_indices = self.selected_points
        # area_percent_change = 500
        phi_type = "constant"
        model = "test_aneurysm"
        affine_params = {"eps": {model: 1.0}, "scale": {model: 1.1}}
        mu = 1
        nu = 0.4
        num_time_steps = 25
        num_time_steps = 1
        self.run_aneurysm(
        affine_params, model, self.centerline_filename, self.mesh_filename,centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, area_percent_change, num_time_steps, list_of_node_point_indices, 
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)
        self.update_mesh_viewer()
        
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
            
            surface_displacements = scaling.get_affine_displacements_v2(data, a, b, eps, s, phi_type, "surface", None)
            
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
            
            for ig in range(len(other_geometry_polydatas)):
                other_geometry_displacements = scaling.get_affine_displacements_v2(data, a, b, eps, s, phi_type, "other_geometry_" + str(ig), surface_mesh_scale_factor)
                data = common.update_points_with_displacements(data, other_geometry_displacements, "other_geometry_" + str(ig))
                other_geometry_polydatas[ig] = common.update_polydata_with_points(other_geometry_polydatas[ig], data, "other_geometry_" + str(ig))
            
            data_surface_copy = {"points" : {"surface" : copy.deepcopy(v2n(surface_polydata.GetPoints().GetData()))}}
            surface_polydata_copy = common.update_polydata_with_points(surface_polydata_copy, data_surface_copy, "surface")
            current_radius = np.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / np.pi)

    def deform_mesh_jonathan(self, area_percent_change):
        if len(self.selected_points) < 3:
            print("Please select at least 3 points along the centerline.")
            return
        if len(self.selected_points) > 3:
            print("Using only the most recent 3 points picked.")
            self.selected_points = self.selected_points[-3:]

        create_aneurysm(self.mesh_filename, self.centerline_filename, self.selected_points, area_percent_change)
        self.update_mesh_viewer()
    
    def deform_mesh_stenosis(self, area_percent_change, num_ring_points, falloff_type, weight_regularized_laplacian):
        if len(self.selected_points) < 3:
            print("Please select at least 3 points along the centerline.")
            return
        if len(self.selected_points) > 3:
            print("Using only the most recent 3 points picked.")
            self.selected_points = self.selected_points[-3:]

        centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
        surface_polydata_output_file_name = "obtained_aneurysm_surface"
        list_of_other_geometry_polydata_input_file_names = []
        list_of_other_geometry_polydata_output_file_names = []
        self.selected_points.sort()
        force_center_point_id = self.selected_points[1]
        list_of_node_point_indices = [force_center_point_id]
        model="test_stenosis"
        affine_params = {"eps": {model: 1.0}}
        mu = 1
        nu = 0.1
        num_time_steps = 1
        self.run_stenosis_v5_timer(
        affine_params, model, self.centerline_filename, self.mesh_filename,
        centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, force_center_point_id, 
        num_ring_points, area_percent_change, num_time_steps, list_of_node_point_indices, falloff_type, 
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names, weight_regularized_laplacian
        )
        self.update_mesh_viewer()
    
    def run_stenosis_v5_timer(self, affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, force_center_point_id, num_ring_points, area_percent_change, num_time_steps, list_of_node_point_indices, falloff_type, list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names, weight_regularized_laplacian):
        affine_type = "stenosis"
        brush_level = "uniscale"
        phi_type = "point"
        extension = ".vtp"

        start_time = time.time()
        
        # Validate input
        assert((0 <= weight_regularized_laplacian) and (weight_regularized_laplacian <= 1))

        # Calculate `a` and `b` using material properties
        a, b = common.get_a_b(mu, nu)
        print(f"Time to get a and b: {time.time() - start_time:.4f} seconds")

        # Read centerline and surface polydata files
        centerline_polydata = vtk_utils.read_polydata_file(centerline_polydata_input_file_name)
        centerline_polydata = self.centerline
        surface_polydata = self.mesh

        start = time.time()
        other_geometry_polydatas = []
        for other_geometry_polydata_input_file_name in list_of_other_geometry_polydata_input_file_names:
            other_geometry_polydatas.append(vtk_utils.read_polydata_file(other_geometry_polydata_input_file_name))
        print(f"Time to read other geometry polydatas: {time.time() - start:.4f} seconds")

        # Define affine points and nodes
        start = time.time()
        data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        data = scaling.define_nodes_affine(data, list_of_node_point_indices)
        print(f"Time to define affine points and nodes: {time.time() - start:.4f} seconds")

        # Assign force location
        start = time.time()
        data = scaling.assign_force_location_affine_v2(data, force_center_point_id)
        print(f"Time to assign force location: {time.time() - start:.4f} seconds")

        # Add node data to centerline
        start = time.time()
        centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(data, centerline_polydata)
        print(f"Time to add node data to centerline: {time.time() - start:.4f} seconds")

        # Calculate origin, normal, original area, and target area
        start = time.time()
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, data["nodes"]["force_center_point_id"])
        print(f"Time to calculate origin, normal {time.time() - start:.4f} seconds")
        start = time.time()
        # original_area = vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal)
        # target_area = original_area * area_percent_change / 100
        # delta_area = (target_area - original_area) / num_time_steps
        original_area = 0
        delta_area = 0
        print(f"Time to calculate target area: {time.time() - start:.4f} seconds")

        # Main time-stepping loop
        for it in range(num_time_steps):
            print(f"---------------------------------------------------------------------- it = {it}")

            # Calculate current radius and `eps`
            start = time.time()
            current_radius = jnp.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / jnp.pi)
            eps = affine_params["eps"][model] * current_radius
            print(f"Time to calculate current radius and eps: {time.time() - start:.4f} seconds")

            # Precompile Kelvinlet translation function
            start = time.time()
            kelvinlets_translation_jit = jx.jit(common.kelvinlets_translation_v2_jax)
            print(f"Time to precompile kelvinlets_translation_v2_jax: {time.time() - start:.4f} seconds")
            # Precompile second Kelvinlet translation function
            start = time.time()
            kelvinlets_translation_jit_non_blocky = jx.jit(common.kelvinlets_translation_v2_jax_non_blocky)
            print(f"Time to precompile kelvinlets_translation_v2_jax_non_blocky: {time.time() - start:.4f} seconds")

            # Calculate ring points and forces
            start = time.time()
            ring_points, ring_forces = scaling.get_ring_point_and_forces_v2(
                data, a, b, eps, jnp.array(origin), normal, surface_polydata, 
                num_ring_points, original_area + delta_area * (it + 1), falloff_type, kelvinlets_translation_jit
            )
            print(f"First Time to calculate ring points and forces (iteration {it}): {time.time() - start:.4f} seconds")
            # Calculate ring points and forces
            # start = time.time()
            # ring_points, ring_forces = scaling.get_ring_point_and_forces_v2(
            #     data, a, b, eps, jnp.array(origin), normal, surface_polydata, 
            #     num_ring_points, original_area + delta_area * (it + 1), falloff_type, kelvinlets_translation_jit
            # )
            # print(f"Second Time to calculate ring points and forces (iteration {it}): {time.time() - start:.4f} seconds")

            # Calculate ring displacements
            start = time.time()
            surface_displacements = scaling.get_ring_displacements_v2(data, a, b, eps, "surface", ring_points, ring_forces, falloff_type, kelvinlets_translation_jit_non_blocky)
            print(f"Time to calculate surface displacements (iteration {it}): {time.time() - start:.4f} seconds")

            # Additional Laplacian calculation if required
            if falloff_type != "regular":
                start = time.time()
                ring_points_lap, ring_forces_lap = scaling.get_ring_point_and_forces_v2(
                    data, a, b, eps, jnp.array(origin), normal, surface_polydata, 
                    num_ring_points, original_area + delta_area * (it + 1), "laplacian"
                )
                surface_displacements_lap = scaling.get_ring_displacements_v2(data, a, b, eps, "surface", ring_points_lap, ring_forces_lap, "laplacian", kelvinlets_translation_jit_non_blocky)
                surface_displacements = weight_regularized_laplacian * surface_displacements + (1 - weight_regularized_laplacian) * surface_displacements_lap
                print(f"Time to calculate Laplacian adjustments (iteration {it}): {time.time() - start:.4f} seconds")

            # Update points and polydata
            start = time.time()
            data = common.update_points_with_displacements(data, surface_displacements, "surface")
            surface_polydata = common.update_polydata_with_points(surface_polydata, data, "surface")
            print(f"Time to update points and polydata (iteration {it}): {time.time() - start:.4f} seconds")

        # Update normals
        start = time.time()
        # surface_polydata = vtk_utils.update_surface_polydata_normals(surface_polydata)
        print(f"Time to update surface polydata normals: {time.time() - start:.4f} seconds")

        total_time = time.time() - start_time
        print(f"Total execution time for run_stenosis_v5_timer: {total_time:.4f} seconds")


    def run_stenosis_v5(self, affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, force_center_point_id, num_ring_points, area_percent_change, num_time_steps, list_of_node_point_indices, falloff_type, list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names, weight_regularized_laplacian):
        affine_type = "stenosis"
        brush_level = "uniscale"
        phi_type = "point"
        extension = ".vtp"

        assert((0 <= weight_regularized_laplacian) and (weight_regularized_laplacian <= 1))
        
        a, b = common.get_a_b(mu, nu)
        
        centerline_polydata = vtk_utils.read_polydata_file(centerline_polydata_input_file_name)
        # surface_polydata = vtk_utils.read_polydata_file(surface_polydata_input_file_name)
        centerline_polydata = self.centerline
        surface_polydata = self.mesh
        
        other_geometry_polydatas = []
        for other_geometry_polydata_input_file_name in list_of_other_geometry_polydata_input_file_names:
            other_geometry_polydatas.append(vtk_utils.read_polydata_file(other_geometry_polydata_input_file_name))
        
        data = scaling.define_points_affine(centerline_polydata, surface_polydata, other_geometry_polydatas)
        assert(list_of_node_point_indices is not None)
        data = scaling.define_nodes_affine(data, list_of_node_point_indices)
        assert(force_center_point_id is not None)
        data = scaling.assign_force_location_affine_v2(data, force_center_point_id)
        
        centerline_polydata = scaling.add_node_data_to_centerline_polydata_affine(data, centerline_polydata)
        
        # vtk_utils.write_polydata(centerline_polydata_output_file_name + "_" + affine_type + "_" + phi_type + "_" + brush_level + "_" + falloff_type + "_run_stenosis_original" + extension, centerline_polydata)
        
        origin, normal = vtk_utils.get_coordinates_and_normal_at_point_on_centerline(centerline_polydata, data["nodes"]["force_center_point_id"])
        original_area = vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal)
        target_area = original_area * area_percent_change / 100
        delta_area = (target_area - original_area) / num_time_steps

        # Precompile second Kelvinlet translation function
        kelvinlets_translation_jit_non_blocky = jx.jit(common.kelvinlets_translation_v2_jax_non_blocky)
        
        for it in range(num_time_steps):
            print("---------------------------------------------------------------------- it = ", it)
            
            current_radius = jnp.sqrt(vtk_utils.get_cross_sectional_area(surface_polydata, origin, normal) / jnp.pi)
            eps = affine_params["eps"][model] * current_radius
            
            kelvinlets_translation_jit = jx.jit(common.kelvinlets_translation_v2_jax)
            ring_points, ring_forces = scaling.get_ring_point_and_forces_v2(data, a, b, eps, jnp.array(origin), normal, surface_polydata, num_ring_points, original_area + delta_area * (it + 1), falloff_type, kelvinlets_translation_jit)
            ring_points, ring_forces = scaling.get_ring_point_and_forces_v2(data, a, b, eps, jnp.array(origin), normal, surface_polydata, num_ring_points, original_area + delta_area * (it + 1), falloff_type, kelvinlets_translation_jit)
            
            surface_displacements = scaling.get_ring_displacements_v2(data, a, b, eps, "surface", ring_points, ring_forces, falloff_type, kelvinlets_translation_jit_non_blocky)
            
            if falloff_type != "regular":
                ring_points_lap, ring_forces_lap = scaling.get_ring_point_and_forces_v2(data, a, b, eps, jnp.array(origin), normal, surface_polydata, num_ring_points, original_area + delta_area * (it + 1), "laplacian")
                surface_displacements_lap = scaling.get_ring_displacements_v2(data, a, b, eps, "surface", ring_points_lap, ring_forces_lap, "laplacian", kelvinlets_translation_jit_non_blocky)
                surface_displacements = weight_regularized_laplacian * surface_displacements + (1 - weight_regularized_laplacian) * surface_displacements_lap
            
            data = common.update_points_with_displacements(data, surface_displacements, "surface")
            
            surface_polydata = common.update_polydata_with_points(surface_polydata, data, "surface")
        
        surface_polydata = vtk_utils.update_surface_polydata_normals(surface_polydata)
            
        # vtk_utils.write_polydata(surface_polydata_output_file_name + "_" + "aneurysm" + "_" + "constant" + "_" + brush_level + "_" + str(num_time_steps) + extension, surface_polydata)

    def deform_mesh_stenosis_jonathan(self, area_percent_change, num_ring_points, falloff_type, weight_regularized_laplacian):
        if len(self.selected_points) < 3:
            print("Please select at least 3 points along the centerline.")
            return
        if len(self.selected_points) > 3:
            print("Using only the most recent 3 points picked.")
            self.selected_points = self.selected_points[-3:]

        create_stenosis(self.mesh_filename, self.centerline_filename, self.selected_points, area_percent_change, num_ring_points, falloff_type, weight_regularized_laplacian, model="test_stenosis")
        # self.update_mesh_viewer()

    def update_mesh_viewer(self):
        # updated_mesh = load_vtp_file("obtained_aneurysm_surface_aneurysm_constant_uniscale_1.vtp")
        # updated_centerline = load_vtp_file("obtained_aneurysm_centerline_aneurysm_constant_uniscale_1.vtp")
        self.mesh_filename = "obtained_aneurysm_surface_aneurysm_constant_uniscale_1.vtp"
        self.centerline_filename = "obtained_aneurysm_centerline_aneurysm_constant_uniscale_1.vtp"
        
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

def create_aneurysm(mesh_filename, centerline_filename, selected_points, area_percent_change, model="test_aneurysm"):
    # centerline_polydata_input_file_name = centerline_filename
    # surface_polydata_input_file_name = mesh_filename
    centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
    surface_polydata_output_file_name = "obtained_aneurysm_surface"

    list_of_other_geometry_polydata_input_file_names = []
    list_of_other_geometry_polydata_output_file_names = []

    # to make sure the selected points are in order regardless of picking order
    selected_points.sort()
    force_center_point_id = selected_points[1]
    list_of_node_point_indices = selected_points
    # area_percent_change = 500
    phi_type = "constant"
    affine_params = {"eps": {model: 1.0}, "scale": {model: 1.1}}

    mu = 1
    nu = 0.4
    num_time_steps = 25
    num_time_steps = 1

    # write_vtp_file(centerline, centerline_polydata_input_file_name)
    # write_vtp_file(mesh, surface_polydata_input_file_name)

    scaling.run_aneurysm(
        affine_params, model, centerline_filename, mesh_filename,
        centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, 
        force_center_point_id, area_percent_change, num_time_steps, list_of_node_point_indices, 
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names
    )

def create_stenosis(mesh_filename, centerline_filename, selected_points, area_percent_change, num_ring_points, falloff_type, weight_regularized_laplacian, model="test_stenosis"):
    centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
    surface_polydata_output_file_name = "obtained_aneurysm_surface"

    list_of_other_geometry_polydata_input_file_names = []
    list_of_other_geometry_polydata_output_file_names = []

    selected_points.sort()
    force_center_point_id = selected_points[1]
    list_of_node_point_indices = [force_center_point_id]
    affine_params = {"eps": {model: 1.0}}

    mu = 1
    nu = 0.1
    num_time_steps = 25
    num_time_steps = 1


    scaling.run_stenosis_v4(
        affine_params, model, centerline_filename, mesh_filename,
        centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, force_center_point_id, 
        num_ring_points, area_percent_change, num_time_steps, list_of_node_point_indices, falloff_type, 
        list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names, weight_regularized_laplacian
    )