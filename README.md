# svMorph: VirtualCathLab – Stent‑deployment utilities for vascular meshes with centerlines
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](#license)
[![made with VTK](https://img.shields.io/badge/made%20with-VTK-398593)](https://vtk.org)
[![python 3.9](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/downloads/release/python-3913/)

Research code for *in‑silico* generation of post-stent geometry using SDF indenting based
surface deformation and associated analysis utilities. This project demonstrates the use 
of VTK for 3D visualization and processing, integrated with PyQt for GUI applications. 
The project includes various scripts for handling VTK data, performing segmentation, 
scaling, and voxelization.
  
*(last tested on macOS 15.4 / Apple‑silicon, Python 3.9, VTK 9.3, JAX 0.4.30).*

---

## Quick install (Mac with **Apple‑silicon**)

> **micromamba** is recommended for Mac with M-series chip as it is very fast, lightweight, and
> coexists happily with Homebrew and system Python.

### 1. Install **micromamba** on Mac with Apple-silicon

```bash
# Home in your $HOME/.local, no sudo needed
curl -L https://micromamba.snakepit.net/api/micromamba/osx-arm64/latest \
     | tar -xvj bin/micromamba
mkdir -p ~/micromamba
mv bin/micromamba ~/micromamba/
echo 'export PATH="$HOME/micromamba:$PATH"' >> ~/.zshrc   # or ~/.bash_profile
source ~/.zshrc                                          # reload shell
```
(See https://mamba.readthedocs.io/en/latest/installation.html for information about installing on other
platforms.)

### 2. Create the virtualcathlab environment

```
micromamba create -y -n virtualcathlab \
    python=3.9.19 \
    numpy=1.24.4 \
    scipy=1.10.1 \
    vtk=9.3.0 \
    -c conda-forge
```

### 3. Activate + add pip‑only packages

First initialize micromamba in the shell before first use:
```
eval "$(micromamba shell hook --shell zsh)"
```

Next, activate the environment and pip install the rest of the required packages:
```
micromamba activate virtualcathlab
pip install --upgrade pip
pip install "jax[cpu]"==0.4.30
pip install pyqt6==6.7
```

### 4. Verify the environment is correct

```
python installation_test.py
```

Expected output:
```
Python version: 3.9.19 | packaged by conda-forge | (main, Mar 20 2024, 12:55:20) 
[Clang 16.0.6 ]
NumPy version : 1.24.4
SciPy version : 1.10.1
JAX version   : 0.4.30
PyQt6 version : 6.7.0
Qt version    : 6.7.1
VTK version   : 9.3.0
```
If the versions match, the environment is
ready. 
Note: it is also expected to see a block of warning first about objc[32187]: Class QT_ROOT_LEVEL_POOL__THESE_OBJECTS_WILL_BE_RELEASED_WHEN_QAPP_GOES_OUT_OF_SCOPE, this does not affect usage)
 
⸻

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
   python main.py
   ```

2. Explore the various functionalities provided by the scripts for VTK data processing and visualization.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [VTK](https://vtk.org/)
- [PyQt](https://riverbankcomputing.com/software/pyqt/intro)

Feel free to contribute to this project by submitting issues or pull requests.
