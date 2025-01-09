# vtk-kelvinlet-3d

This project demonstrates the use of VTK for 3D visualization and processing, integrated with PyQt for GUI applications. The project includes various scripts for handling VTK data, performing segmentation, scaling, and voxelization.

## Files

- **full_production_mode_test.py**
  - Main script for the PyQt application.
  - Contains the 

MainWindow

 class which initializes the GUI.
  - Uses 

VTKHandler

 for VTK operations and 

QTimer

 for timing events.

- **scaling.py**
  - Contains functions for computing rotation and Householder matrices.
  - Includes methods for scaling and transforming VTK data.

- **sdf_and_voxel_test.py**
  - Demonstrates the use of VTK for signed distance functions (SDF) and voxelization.
  - Configures VTK modellers and generates contour values.

- **segmentation_functionality_test.py**
  - Handles segmentation of VTK polydata based on segment IDs.
  - Uses `vtkThreshold` and `vtkGeometryFilter` for extracting and processing segments.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/my-vtk-project.git
   cd my-vtk-project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main PyQt application:
   ```bash
   python full_production_mode_test.py
   ```

2. Explore the various functionalities provided by the scripts for VTK data processing and visualization.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [VTK](https://vtk.org/)
- [PyQt](https://riverbankcomputing.com/software/pyqt/intro)

Feel free to contribute to this project by submitting issues or pull requests.
