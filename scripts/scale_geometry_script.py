import vtk

def scale_polydata(input_filename, output_filename, scale_factor):
    # Read the .vtp file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_filename)
    reader.Update()
    polydata = reader.GetOutput()

    # Create a transform that scales uniformly
    transform = vtk.vtkTransform()
    transform.Scale(scale_factor, scale_factor, scale_factor)

    # Apply the transformation to the polydata
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polydata)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    # Write the transformed polydata to a new .vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(transformFilter.GetOutput())
    writer.Write()

if __name__ == '__main__':
    # Scaling down by 10 means using a factor of 0.1
    scale_factor = 10 #0.1

    # File names for the input and output files
    # surface_input = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_geometry_remeshed.vtp"
    # centerline_input = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_fine_centerlines.vtp"
    # surface_input = "/home/bohanjeffli/SU0243-preop.vtp"
    # centerline_input = "/home/bohanjeffli/SU0243-preop-centerlines.vtp"
    # surface_input = "/home/bohanjeffli/SU0243-postop-estimated.vtp"
    # centerline_input = "/Users/bohanli/Downloads/f_1_rev_centerlines.vtp"
    # surface_input = "/Users/bohanli/Downloads/f_1_rev.vtp"

    surface_input = "/Users/bohanli/Downloads/Sanjib-Fontan/model_surface.vtp"
    centerline_input = "/Users/bohanli/Downloads/Sanjib-Fontan/centerlines.vtp"

    # centerline_input = "/home/bohanjeffli/SU0243-postop-estimated-centerlines.vtp"
    # surface_output = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_geometry_remeshed_scaled_down_10x.vtp"
    # centerline_output = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_fine_centerlines_scaled_down_10x.vtp"
    # surface_output = "/home/bohanjeffli/SU0243-preop-cm.vtp"
    # centerline_output = "/home/bohanjeffli/SU0243-preop-centerlines-cm.vtp"
    # surface_output = "/home/bohanjeffli/SU0243-postop-estimated-cm.vtp"
    # centerline_output = "/Users/bohanli/Downloads/f_1_rev_centerlines_cm.vtp"
    # surface_output = "/Users/bohanli/Downloads/f_1_rev_cm.vtp"

    surface_output = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_surface_cm.vtp"
    centerline_output = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_centerlines_cm.vtp"
    # centerline_output = "/home/bohanjeffli/SU0243-postop-estimated-centerlines-cm.vtp"

    surface_input1 = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_intra_ivc_16mm_cm.vtp"
    surface_output1 = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_intra_ivc_16mm.vtp"
    surface_input2 = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_rhv_10point6mm_cm.vtp"
    surface_output2 = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_rhv_10point6mm.vtp"
    surface_input3 = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_intra_ivc_16mm_rhv_10point6mm_cm.vtp"
    surface_output3 = "/Users/bohanli/Downloads/Sanjib-Fontan/fontan_intra_ivc_16mm_rhv_10point6mm.vtp"

    # Scale both the surface mesh and the centerline
    # scale_polydata(surface_input, surface_output, scale_factor)
    # scale_polydata(centerline_input, centerline_output, scale_factor)
    scale_polydata(surface_input1, surface_output1, scale_factor)
    scale_polydata(surface_input2, surface_output2, scale_factor)
    scale_polydata(surface_input3, surface_output3, scale_factor)

    print("Scaling complete. Output files:")
    print(" -", surface_output)
    print(" -", centerline_output)