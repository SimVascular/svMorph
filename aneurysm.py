# import os
# import sys
# import vtk
# import math
# import copy
import numpy as np
# from time import perf_counter
# from vtk.util.numpy_support import vtk_to_numpy as v2n
# from vtk.util.numpy_support import numpy_to_vtk as n2v

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# sys.path.append("../prototype")
import scaling
# import translate
# import vtk_utils
# import common
# sys.path.pop()

def create_aneurysm(model = "test_aneurysm"):
    
    # input files
    centerline_polydata_input_file_name = "centerline.vtp"
    surface_polydata_input_file_name = "mesh-complete-exterior.vtp"
    list_of_other_geometry_polydata_input_file_names = [] # can be empty list
    
    # output files
    centerline_polydata_output_file_name = "obtained_aneurysm_centerline"
    surface_polydata_output_file_name = "obtained_aneurysm_surface"
    list_of_other_geometry_polydata_output_file_names = [] # can be empty list

    # variable aneurysm creation parameters
    force_center_point_id = 152
    list_of_node_point_indices = [71, force_center_point_id, 234]
    area_percent_change = 500 # percent of original cross-sectional area
    phi_type = "constant" # "point"
    affine_params = {"eps" : {model : 1.0}, "scale" : {model : 1.1}}
    
    # most likely do not need to change these parameters
    mu = 1
    nu = 0.4
    num_time_steps = 25
    
    ########################################################################
    
    scaling.run_aneurysm(affine_params, model, centerline_polydata_input_file_name, surface_polydata_input_file_name, centerline_polydata_output_file_name, surface_polydata_output_file_name, mu, nu, phi_type, force_center_point_id, area_percent_change, num_time_steps, list_of_node_point_indices, list_of_other_geometry_polydata_input_file_names, list_of_other_geometry_polydata_output_file_names)

if __name__ == "__main__":
    create_aneurysm()
    print("Tests passed.")