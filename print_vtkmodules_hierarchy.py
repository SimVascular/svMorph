from __future__ import annotations

import importlib
import pkgutil
import warnings

import vtkmodules


def get_vtk_version() -> tuple[int, int, int]:
    """Returns the ``VTK`` version."""
    try:
        from vtkmodules.all import vtkVersion

        ver = vtkVersion()
    except AttributeError:
        warnings.warn('Unable to detect VTK version.')

    major = ver.GetVTKMajorVersion()
    minor = ver.GetVTKMinorVersion()
    patch = ver.GetVTKBuildVersion()

    return f'{major}.{minor}.{patch}'


if __name__ == '__main__':
    with open(
        f'vtkmodules_{get_vtk_version()}_hierarchy.txt',
        'w',
        encoding='utf-8',
    ) as w:
        for pkg in pkgutil.walk_packages(vtkmodules.__path__, vtkmodules.__name__ + '.'):
            try:
                module = importlib.import_module(pkg.name)
            except ImportError as err:
                continue

            w.write(f'+ {module.__name__}\n')

            for subitem in sorted(dir(module)):
                w.write(f'  - {subitem}\n')