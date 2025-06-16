import vtk
import numpy as np
import jax.numpy as jnp
from vtk.util.numpy_support import numpy_to_vtk as n2v
from time import sleep

def read_vtp_and_segment(vtp_file_path):
    """
    Reads a .vtp file, extracts the segmented subsections and their caps, and converts
    them into `jax.numpy` arrays for efficient processing.
    """
    # Step 1: Read the .vtp file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Step 2: Access segmentation information
    segmentation_array = polydata.GetCellData().GetArray("ModelFaceID")  # Example: segmentation ID array
    if segmentation_array is None:
        raise ValueError("Segmentation information (ModelFaceID) not found in .vtp file")
    print(f"Segmentation array: {segmentation_array}")
    segment_ids = [int(segmentation_array.GetValue(i)) for i in range(segmentation_array.GetNumberOfTuples())]
    segment_ids = set(segment_ids)
    print("segment_ids: ", segment_ids)

    # Step 3: Extract segments and caps into separate `jnp` arrays
    segments = {}
    for segment_id in segment_ids:
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(polydata)
        threshold.SetLowerThreshold(segment_id)
        threshold.SetUpperThreshold(segment_id)
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "ModelFaceID")
        threshold.Update()

        segment_polydata = vtk.vtkGeometryFilter()  # Convert the threshold output to polydata
        segment_polydata.SetInputConnection(threshold.GetOutputPort())
        segment_polydata.Update()
        segment_polydata = segment_polydata.GetOutput()

        # Convert segment vertices to numpy arrays
        points = segment_polydata.GetPoints()
        vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        cells = segment_polydata.GetPolys()
        cells.InitTraversal()
        cell_array = []
        id_list = vtk.vtkIdList()
        while cells.GetNextCell(id_list):
            cell_array.append([id_list.GetId(j) for j in range(id_list.GetNumberOfIds())])
        connectivity = np.array(cell_array, dtype=np.int32)

        # Convert to `jnp` arrays
        segments[segment_id] = {
            "vertices": jnp.array(vertices),
            "connectivity": jnp.array(connectivity),
            "vtk_points": points  # Store the original VTK points object
        }

    return segments

def main():
    # Example Usage:
    vtp_file = "/home/bohanjeffli/Unstented-Full-Tree-PA.vtp"
    segments = read_vtp_and_segment(vtp_file)

    # for segment_id, segment_data in segments.items():
        # print(f"Processing segment {segment_id}...")
        # scaled_vertices = example_process_segment(segment_data)
        # print(f"Updated vertices for segment {segment_id}:", scaled_vertices)
    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.Initialize()
    # Define a color map for the segments
    colors = vtk.vtkNamedColors()
    color_names = colors.GetColorNames().split('\n')

    # Display each segment with a different color
    for segment_id, segment_data in segments.items():
        # vertices = segment_data["vertices"]
        connectivity = segment_data["connectivity"]

        # Create a VTK points object
        points = segment_data["vtk_points"]

        # Create a VTK cell array
        polys = vtk.vtkCellArray()
        for cell in np.array(connectivity):
            polys.InsertNextCell(len(cell), cell)

        # Create a polydata object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set the color of the actor
        # print(f"Color names: {color_names}")
        color_name = color_names[4 * (segment_id-1) % len(color_names)]
        actor.GetProperty().SetColor(colors.GetColor3d(color_name))

        # Add the actor to the renderer
        renderer.AddActor(actor)

    # Set background color and size
    render_window_interactor.GetRenderWindow().SetSize(800, 600)
    render_window_interactor.GetRenderWindow().GetRenderers().GetFirstRenderer().SetBackground(colors.GetColor3d("White"))
    render_window_interactor.GetRenderWindow().Render()

    # Inflate a segment by increasing the scale of its vertices
    sleep(2)  # Sleep for 2 seconds before inflating the segment
    segment = segments[10]
    inflated_vertices = example_process_segment(segment)
    update_segment_vertices(segment, inflated_vertices)
    
    # Start the rendering loop
    render_window_interactor.GetRenderWindow().Render()
    # render_window_interactor.Start()

    # Inflate the segment whose ID is half of the number of segments
    sleep(2)  # Sleep for 2 seconds before inflating the segment
    segment_id_to_inflate = 5
    segment_to_inflate = segments[segment_id_to_inflate]
    inflated_vertices = example_process_segment(segment_to_inflate)
    # sleep(2)  # Sleep for 5 seconds to observe the inflated segment
    update_segment_vertices(segment_to_inflate, inflated_vertices)
    render_window_interactor.GetRenderWindow().Render()  # Re-render the window to see the updated mesh
    render_window_interactor.Start()
    # render_window_interactor.Start()
    # Example usage in main function
    output_vtp_file = "/home/bohanjeffli/MarsdenLab/my-vtk/combined_segments.vtp"
    save_segments_as_vtp(segments, output_vtp_file)
    

def update_segment_vertices(segment, new_vertices):
    """
    Update the vertices of a segment in the original VTK points object with the jnp array new_vertices.
    """
    vtk_points = segment["vtk_points"]
    vtk_points.SetData(n2v(new_vertices))
    vtk_points.Modified()

def example_process_segment(segment):
    """
    Example of how to process a segment represented by jnp arrays.
    """
    vertices = segment["vertices"]
    # Example: Apply a scaling transformation to the segment
    # Calculate the geometric center of the segment
    center = jnp.mean(vertices, axis=0)

    # Inflate the vertices by scaling them relative to the center
    scale_factor = 2  # Example scale factor
    scaled_vertices = (vertices - center) * scale_factor + center
    return scaled_vertices

def save_segments_as_vtp(segments, output_file_path):
    """
    Save the resulting segments as a combined geometry in a single VTP file.
    """
    # Create a new polydata object to hold the combined geometry
    combined_polydata = vtk.vtkPolyData()
    combined_points = vtk.vtkPoints()
    combined_polys = vtk.vtkCellArray()

    point_offset = 0
    for segment_id, segment_data in segments.items():
        points = segment_data["vtk_points"]
        connectivity = segment_data["connectivity"]

        # Add points to the combined points object
        for i in range(points.GetNumberOfPoints()):
            combined_points.InsertNextPoint(points.GetPoint(i))

        # Add cells to the combined cell array
        for cell in connectivity:
            new_cell = vtk.vtkIdList()
            for point_id in cell:
                new_cell.InsertNextId(point_id + point_offset)
            combined_polys.InsertNextCell(new_cell)

        point_offset += points.GetNumberOfPoints()

    combined_polydata.SetPoints(combined_points)
    combined_polydata.SetPolys(combined_polys)

    # Write the combined polydata to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file_path)
    writer.SetInputData(combined_polydata)
    writer.Write()

    

if __name__ == "__main__":
    main()
