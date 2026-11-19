# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

svMorph edits patient-specific vascular geometry — a triangulated surface `.vtp` plus its centerline `.vtp` — through three workflows: SDFStent virtual stenting (optionally with axis straightening), regularized-Kelvinlet aneurysm creation, and stenosis creation. Every workflow exists both in a PyQt6 + VTK GUI and as a headless CLI script. Published to PyPI as `svmorph`.

## Commands

```bash
# Install
pip install ".[dev]"             # what CI does: core deps + pytest/pytest-cov
pip install -e ".[dev,gui]"      # editable, plus PyQt6 for the GUI

# Tests
pytest tests/ -v --cov=svmorph --cov-report=term-missing        # CI invocation
pytest tests/test_geometry.py::test_resampled_length_proximal   # single test

# GUI from source (installed equivalent: svmorph-gui)
python main.py [--units mm] [--verbose | --debug]

# Headless workflows (installed equivalents: svmorph-deploy-stent, svmorph-create-aneurysm, ...)
python -m svmorph.scripts.deploy_stent --help
python -m svmorph.scripts.deploy_stent_straighten --help
python -m svmorph.scripts.create_aneurysm --help
python -m svmorph.scripts.create_stenosis --help
```

- No linter, formatter, or type checker is configured.
- Tests need only NumPy/SciPy/JAX (no VTK, no Qt) and import `svmorph` from the source tree, so they run without installing the package.
- CI runs Python 3.9–3.12. Code must stay 3.9-compatible, which is why modules using `X | None` annotations start with `from __future__ import annotations`.
- Pushing a `v*.*.*` tag triggers `.github/workflows/release.yml`, which publishes to PyPI. The version is hand-maintained in three places that must be bumped together: `pyproject.toml` (what the release uses), `__version__` in `svmorph/__init__.py`, and the README badge.

### What the tests do not cover

The tracked tests cover only `core/units`, `core/defaults`, `core/geometry`, and `logging`. Nothing exercises `core/deformation.py`, `scripts/`, `visualization/`, or `gui/`, and no geometry is tracked (`*.vtp`, `input/`, and `tests/input|data|results/` are gitignored). Verifying a kernel or CLI change therefore means supplying a surface/centerline pair yourself. A synthetic straight tube is enough to run all four scripts end to end: `vtkLineSource` → `vtkTubeFilter` → `vtkTriangleFilter` for the surface, and a `vtkLineSource` polyline carrying a `MaximumInscribedSphereRadius` point array for the centerline.

## Architecture

### Layers

Dependencies point one way: `gui` → `visualization` → `core`.

- `svmorph/core/` — all numerics, on NumPy/SciPy/JAX only. It must stay free of VTK and Qt imports: it is the public API (`core/__init__.py`) that downstream 3D Slicer / ParaView plugins import on its own.
- `svmorph/visualization/` — everything VTK. `vtk_io.py` is the VTK ↔ NumPy bridge used by both front ends; `renderer.py` (`SceneManager`) and `interactor.py` (`MeshInteractor`) are GUI-only. The package `__init__` eagerly imports all three, so the headless scripts' `from svmorph.visualization import vtk_io` also loads VTK's OpenGL rendering modules (but never PyQt6).
- `svmorph/gui/main_window.py` — PyQt6 widgets, slider ↔ value mapping (`SliderMapper`), and the `QTimer`s behind the hold-to-run buttons. It contains no deformation logic.
- `svmorph/scripts/` — one CLI per workflow; `common.py` holds the shared pieces (`SimulationContext`, `SnapshotManager`, common argparse flags, unit/logging setup).

### Two front ends, one engine, duplicated step loops

Every workflow is an iteration of small displacement steps until a target radius is reached. `core/deformation.py` computes one step; the loop around it is written twice:

- **GUI** — `MainWindow` → `SceneManager` (loads the VTPs, builds actors) → `MeshInteractor`, a `vtkInteractorStyleTrackballCamera` subclass that owns all simulation state (selected centerline point, stent axis vertices, current radius). Its `deform_mesh_*` methods check termination and its `run_*` methods perform one step; each button click or timer tick is one step. It reports progress by writing to the window title through a `_main_window` back-reference.
- **CLI** — each script runs its own `while` loop over the same `compute_*` functions.

Termination conditions, the foreshortening factor, axis straightening, the sharpness → epsilon mapping, and material constants are implemented in both `interactor.py` and the scripts, and the scripts hard-code their defaults as literals instead of importing `core/defaults.py`. A change to a workflow's behaviour or defaults has to be made in both places. The same goes for the GUI entry point: `main.py` (source runs) and `svmorph/_gui_entry.py` (the `svmorph-gui` console script) are parallel copies of the same argument handling.

### The simulation data dict is a live view of the VTK buffers

`vtk_io.extract_mesh_arrays` returns `data["points"]["surface"]` and `["centerline"]` as float32 NumPy views onto the `vtkPolyData` point arrays. `mesh_data.apply_displacements` adds to them in place, so a deformation step mutates the VTK objects directly and `vtk_io.sync_polydata` only has to call `Modified()`. Two consequences:

- Rebinding instead of mutating (`data["points"]["surface"] = new_array`) silently detaches the simulation from what is rendered and saved.
- No copy of the original geometry is kept; there is no undo short of reloading the files.

`compute_*` functions read the active site from `data["nodes"]["force_center_point_id"]` (set via `deformation.set_force_center`).

### Deformation kernels (`core/deformation.py`)

Each public `compute_*_displacements` is a NumPy-level wrapper around `@jx.jit` kernels and returns full-size per-vertex displacement arrays.

- **Aneurysm / stenosis** build a Householder matrix reflecting the centerline tangent onto z, evaluate the kernel in that local frame with the axial component zeroed, and reflect back. They move the surface only. Progress is the distance from the center point to a single "representative" surface vertex picked once up front by `find_stenosis_minimum_radius_representative` (used for aneurysms too, despite the name).
- **SDF contact (stent)** culls to the stent's padded bounding box, evaluates the smooth-min capsule-chain SDF for surface and centerline points in one batch, finds the contact set, uses a `cKDTree` over it to blend an influence zone, and scatters the result back to full size. It moves both surface and centerline. Progress is a radius incremented by the returned `step_size`.
- The stent radius handed to the SDF kernel is the nominal radius **minus** the smooth-min constant `k` (`SMOOTHING_K_CM`), because the smooth-min blend inflates the capsule surface by up to `k` at segment junctions; `k` is added back wherever a radius is displayed or compared with the target.
- Sign convention for the force scale `s`: aneurysm and SDF-contact kernels expand for `s < 0` (the GUI negates the slider value before passing it); stenosis contracts for `s > 0`.

JAX runs in its default float32 mode. Jitted kernels retrace whenever input shapes change (a different mesh, or a stent length that changes the number of axis vertices).

### Units

All spatial constants are stored in centimetres in `core/defaults.py` and multiplied by `L()` from `core/units.py` at the point of use (1.0 for cm, 10.0 for mm). Values supplied by the user are taken to be in the active unit already; only built-in defaults are scaled. Any new spatial constant must be scaled by `L()` the same way.

`L()` is read as a plain Python float inside the jitted kernels, so its value is frozen into each kernel the first time it is traced. `set_unit_scale` must run before any kernel does (the entry points call it straight after argument parsing); changing it later in the same process has no effect on already-compiled kernels unless `jax.clear_caches()` is called. Keep this in mind when writing tests that switch units.

### Centerline input

Centerlines are expected in SimVascular/VMTK form. `MaximumInscribedSphereRadius` and `CenterlineSectionArea` point arrays are read if present (zeros otherwise). Branching is derived from the multi-component `CenterlineId` flag array by `vtk_io.build_parent_tip_map`; without it, the file is treated as a single polyline. `geometry.resample_stent_axis` builds the stent axis by walking proximally (decreasing point id) from the selected distal point and jumping to the parent segment's tip at each segment base.

## Conventions

- Type-annotated code with NumPy-style docstrings.
- Logging goes through `svmorph/logging.py`: `logger = get_logger(__name__)`, with `logger.timing(...)` for performance instrumentation. `TIMING` is a custom level (15) between DEBUG and INFO; `--verbose` enables it and `--debug` enables everything.
- The `pyqt6>=6.7,<6.10` pin is deliberate: Qt 6.10 freezes at startup on macOS.
- `README.md` documents the GUI control rows, CLI flags, and the cm/mm default tables by hand; update it when any of those change.
