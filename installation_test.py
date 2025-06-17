import sys
import numpy
import scipy
import jax
import PyQt6.QtCore as qc
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Version checks
print("Python version:", sys.version)
print("NumPy version :", numpy.__version__)
print("SciPy version :", scipy.__version__)
print("JAX version   :", jax.__version__)
print("PyQt6 version :", qc.PYQT_VERSION_STR)
print("Qt version    :", qc.QT_VERSION_STR)
print("VTK version   :", vtk.vtkVersion.GetVTKVersion())