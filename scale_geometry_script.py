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
    scale_factor = 0.1

    # File names for the input and output files
    # surface_input = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_geometry_remeshed.vtp"
    # centerline_input = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_fine_centerlines.vtp"
    surface_input = "/home/bohanjeffli/SU0243-preop.vtp"
    centerline_input = "/home/bohanjeffli/SU0243-preop-centerlines.vtp"
    surface_input = "/home/bohanjeffli/SU0243-postop-estimated.vtp"
    # centerline_input = "/home/bohanjeffli/SU0243-postop-estimated-centerlines.vtp"
    # surface_output = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_geometry_remeshed_scaled_down_10x.vtp"
    # centerline_output = "/home/bohanjeffli/MarsdenLab/my-vtk/input/potts_fine_centerlines_scaled_down_10x.vtp"
    surface_output = "/home/bohanjeffli/SU0243-preop-cm.vtp"
    centerline_output = "/home/bohanjeffli/SU0243-preop-centerlines-cm.vtp"
    surface_output = "/home/bohanjeffli/SU0243-postop-estimated-cm.vtp"
    # centerline_output = "/home/bohanjeffli/SU0243-postop-estimated-centerlines-cm.vtp"

    # Scale both the surface mesh and the centerline
    scale_polydata(surface_input, surface_output, scale_factor)
    scale_polydata(centerline_input, centerline_output, scale_factor)

    print("Scaling complete. Output files:")
    print(" -", surface_output)
    print(" -", centerline_output)