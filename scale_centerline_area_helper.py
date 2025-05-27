import vtk

def scale_polydata(input_filename, output_filename, area_scale=0.01, radius_scale=0.1):
    # Read the input VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_filename)
    reader.Update()
    polydata = reader.GetOutput()

    point_data = polydata.GetPointData()

    # Scale the CenterlineSectionArea by area_scale (0.01)
    area_array = point_data.GetArray("CenterlineSectionArea")
    if area_array is not None:
        for i in range(area_array.GetNumberOfTuples()):
            value = area_array.GetValue(i)
            area_array.SetValue(i, value * area_scale)
        area_array.Modified()
    else:
        print("CenterlineSectionArea array not found.")

    # Scale the maximuminscribedsphereradius by radius_scale (0.1)
    radius_array = point_data.GetArray("MaximumInscribedSphereRadius")
    if radius_array is not None:
        for i in range(radius_array.GetNumberOfTuples()):
            value = radius_array.GetValue(i)
            radius_array.SetValue(i, value * radius_scale)
        radius_array.Modified()
    else:
        print("MaximumInscribedSphereRadius array not found.")

    # Write the modified polydata to the output VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(polydata)
    writer.Write()

if __name__ == "__main__":
    input_file = "SU0243-preop-centerlines-cm.vtp"
    output_file = "corrected-SU0243-preop-centerlines-cm.vtp"
    scale_polydata(input_file, output_file)
    