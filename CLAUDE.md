# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

svMorph edits patient-specific vascular geometry — a triangulated surface `.vtp` plus its centerline `.vtp` — through three workflows: SDFStent virtual stenting (optionally with axis straightening), regularized-Kelvinlet aneurysm creation, and stenosis creation. Every workflow exists both in a PyQt6 + VTK GUI and as a headless CLI script; the one exception is the flared (variable-radius) stent, which is CLI and Python API only. The numerics run on JAX; the stent deployment step can alternatively run as NVIDIA Warp kernels on a CUDA GPU, an optional backend. Published to PyPI as `svmorph`.

## Commands

```bash
# Install
pip install ".[dev]"             # what CI does: core deps + pytest/pytest-cov
pip install -e ".[dev,gui]"      # editable, plus PyQt6 for the GUI
pip install -e ".[dev,gui,warp]" # ... plus NVIDIA Warp, to run or test the GPU stent backend

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

# Timings of the stent backends (synthetic tubes, or --mesh/--cline/--start for real geometry)
python benchmarks/stent_benchmark.py
```

- No linter, formatter, or type checker is configured.
- Tests need only NumPy/SciPy/JAX (no VTK, no Qt) and import `svmorph` from the source tree, so they run without installing the package. Three groups need more and skip without it: `tests/test_deploy_stent_cli.py` imports the CLI scripts and with them VTK, everything that takes the `warp_device` fixture (all of `tests/test_sdf_warp.py`, parts of the others) needs `warp-lang`, and `tests/test_gui_entry.py` needs PyQt6. CI installs neither the `gui` extra nor a display, so that last one only ever runs locally.
- CI's `test` jobs install without Warp, so they show that the optional backend really is optional; the `test-warp` jobs install the `warp` extra and run the same suite with Warp's kernels on its CPU device. GitHub-hosted runners have no NVIDIA GPU, so **nothing in CI runs on CUDA**: after changing `svmorph/core/sdf_warp.py`, run `pytest tests/` on a machine with an NVIDIA GPU, where `warp_device` adds `cuda:0`.
- CI runs Python 3.9–3.12. Code must stay 3.9-compatible, which is why modules using `X | None` annotations start with `from __future__ import annotations`.
- Pushing a `v*.*.*` tag triggers `.github/workflows/release.yml`, which publishes to PyPI. The version is hand-maintained in three places that must be bumped together: `pyproject.toml` (what the release uses), `__version__` in `svmorph/__init__.py`, and the README badge. `README.md` is uploaded as the PyPI project description and is frozen once a version is published, so all three must already match in the commit that gets tagged; the release workflow checks this and refuses to publish otherwise. A README change made after a release only reaches PyPI with the next version.

### What the tests do not cover

The tracked tests cover `core/units`, `core/defaults`, `core/geometry`, `core/backend`, `logging`, the stent SDFs in `core/deformation.py` (uniform and variable-radius, on synthetic stent axes, plus the dispatch between the two in `compute_sdf_contact_displacements`), the Warp kernels in `core/sdf_warp.py` against that JAX reference one subroutine at a time, `core/stent_deployment`, and the two stent scripts: argument validation, and a deployment end to end on a synthetic tube (`tests/test_deploy_stent_cli.py` builds one). `tests/test_gui_entry.py` runs both GUI entry points as far as `--help`, which covers their imports and their options but opens no window. Nothing exercises the aneurysm or stenosis kernels or their scripts, `visualization/`, or `gui/`, and no geometry is tracked (`*.vtp`, `input/`, and `tests/input|data|results/` are gitignored). Verifying a change to those therefore means supplying a surface/centerline pair yourself. A synthetic straight tube is enough to run all four scripts end to end: `vtkLineSource` → `vtkTubeFilter` → `vtkTriangleFilter` for the surface, and a `vtkLineSource` polyline carrying a `MaximumInscribedSphereRadius` point array for the centerline.

`tests/conftest.py` pins every test to the JAX backend unless it asks for Warp through `warp_device`; without that, `"auto"` would pick Warp on a development machine with a GPU and the tests of the reference would stop testing it. Parity tests compare the two backends on geometry from `tests/helpers.without_borderline_points`: the backends agree on the SDF to about 1e-6, so a point that close to the contact threshold can legitimately be classified differently by them, which changes its blending weight by far more than rounding.

## Architecture

### Layers

Dependencies point one way: `gui` → `visualization` → `core`.

- `svmorph/core/` — all numerics, on NumPy/SciPy/JAX, plus Warp where it is installed and selected. It must stay free of VTK and Qt imports: it is the public API (`core/__init__.py`) that downstream 3D Slicer / ParaView plugins import on its own. Importing it must not import Warp either: `core/sdf_warp.py` is imported only when a deployment step actually runs on the Warp backend.
- `svmorph/visualization/` — everything VTK. `vtk_io.py` is the VTK ↔ NumPy bridge used by both front ends; `renderer.py` (`SceneManager`) and `interactor.py` (`MeshInteractor`) are GUI-only. The package `__init__` eagerly imports all three, so the headless scripts' `from svmorph.visualization import vtk_io` also loads VTK's OpenGL rendering modules (but never PyQt6).
- `svmorph/gui/main_window.py` — PyQt6 widgets, slider ↔ value mapping (`SliderMapper`), and the `QTimer`s behind the hold-to-run buttons. It contains no deformation logic.
- `svmorph/scripts/` — one CLI per workflow; `common.py` holds the shared pieces (`SimulationContext`, `SnapshotManager`, common argparse flags, unit/logging setup).

### Two front ends, one engine, duplicated step loops

Every workflow is an iteration of small displacement steps until a target radius is reached. `core/deformation.py` computes one step; the loop around it is written twice:

- **GUI** — `MainWindow` → `SceneManager` (loads the VTPs, builds actors) → `MeshInteractor`, a `vtkInteractorStyleTrackballCamera` subclass that owns all simulation state (selected centerline point, stent axis vertices, current radius). Its `deform_mesh_*` methods check termination and its `run_*` methods perform one step; each button click or timer tick is one step. It reports progress by writing to the window title through a `_main_window` back-reference.
- **CLI** — each script runs its own `while` loop: the aneurysm and stenosis scripts over the same `compute_*` functions, the two stent scripts over `StentDeployment.step` (`core/stent_deployment.py`), which is compute-then-apply under JAX and keeps the mesh on the GPU under Warp. The GUI does not use `StentDeployment`; it calls `compute_sdf_contact_displacements` and applies the result itself, which also runs on Warp when that backend is selected.

Termination conditions, the foreshortening factor, axis straightening, the sharpness → epsilon mapping, and material constants are implemented in both `interactor.py` and the scripts, and the scripts hard-code their defaults as literals instead of importing `core/defaults.py`. A change to a workflow's behaviour or defaults has to be made in both places. The flared stent is the exception: only `scripts/deploy_stent.py` builds a radius profile (`--flare-R`, `--flare-length`, `--flare-end`, `--cap-height-fraction`); the GUI and `deploy_stent_straighten.py` always deploy a uniform stent. The same goes for the GUI entry point: `main.py` (source runs) and `svmorph/_gui_entry.py` (the `svmorph-gui` console script) are parallel copies of the same argument handling. Both import `jax` before PyQt6, and that order matters: PyQt6's Windows wheels bundle an older Microsoft C++ runtime than recent jaxlib builds need, the first of the two to be imported decides which copy the process loads, and with Qt first the GUI dies on `import jax` with "DLL load failed while importing _jax".

### The simulation data dict is a live view of the VTK buffers

`vtk_io.extract_mesh_arrays` returns `data["points"]["surface"]` and `["centerline"]` as float32 NumPy views onto the `vtkPolyData` point arrays. `mesh_data.apply_displacements` adds to them in place, so a deformation step mutates the VTK objects directly and `vtk_io.sync_polydata` only has to call `Modified()`. Two consequences:

- Rebinding instead of mutating (`data["points"]["surface"] = new_array`) silently detaches the simulation from what is rendered and saved.
- No copy of the original geometry is kept; there is no undo short of reloading the files.
- Under the Warp backend a `StentDeployment` keeps the points on the GPU between steps, so the dict (and the VTK buffers behind it) is stale until `session.sync()`. The stent scripts sync before every snapshot (`SnapshotManager(before_save=...)`) and before the final save. Anything that reads or writes the points in the middle of a session has to sync first, and start a new session after writing.

`compute_*` functions read the active site from `data["nodes"]["force_center_point_id"]` (set via `deformation.set_force_center`).

### Stent deployment backends (`core/backend.py`, `core/sdf_warp.py`)

`core/deformation.py` is the reference implementation of every computation. For the SDFStent deployment step only, `core/sdf_warp.py` holds a second implementation as NVIDIA Warp kernels; aneurysm, stenosis and everything else always run on JAX.

- **One selection point.** `core/backend.py` resolves the request (`set_backend()` / the `--backend` flag of the stent scripts and GUI entry points, else the `SVMORPH_BACKEND` environment variable, else `"auto"`) to `"jax"` or `"warp"`. `"auto"` picks Warp only if a supported `warp-lang` is installed, Warp finds a CUDA device and the kernels build on it (`_auto_warp_device`, which catches everything: being the default, `"auto"` must never turn a deployment that works on JAX into one that fails). A backend that is named is never silently replaced (`"warp"` without a usable Warp raises `ImportError`; without a GPU it runs Warp on the CPU). The request is resolved at the first deployment step, not at import; the entry points call `backend.configure_from_command_line` at startup so that a named backend that cannot be used ends the run with one line instead of a traceback from inside a Qt callback. The result is read in exactly two places, `compute_sdf_contact_displacements` and `StentDeployment`, which also pass the device on to `sdf_warp`; do not add backend checks elsewhere, and `sdf_warp.py` itself does not consult the backend. The `--backend` flag is defined once, in `backend.add_backend_argument`, for the stent scripts and both GUI entry points.
- **The Warp kernels must match the JAX reference**, to single precision (about 1e-6 in the SDF and in a step's displacements). A change to the stent step in `deformation.py` has to be made in `sdf_warp.py` as well, and `tests/test_sdf_warp.py` compares them subroutine by subroutine. The module docstring of `sdf_warp.py` maps each kernel to the reference function it mirrors.
- **Two code paths there too**: `_uniform_chain_sdf` / `_classify_uniform_kernel` for the uniform stent and `_variable_radius_chain_sdf` / `_classify_variable_radius_kernel` for a radius profile or flattened caps, chosen by the predicate the reference uses (`deformation._is_uniform_stent`, the one definition of it) and guarded by the same kind of test.
- **No `L()` in the kernels.** They take every length as an argument; `deformation._warp_step_parameters` spells out the values the JAX kernels read from `L()` (the smooth-min `k`, the bounding-box padding). A new spatial constant in the stent step needs an entry there.
- **The contact search is svMorph's own.** Warp has no nearest-neighbour query for point sets, so the SciPy k-d tree of the reference is replaced by an implicit 8-ary tree over a Morton ordering of the surface points, whose bounding boxes are refitted to the current contact set at every step. The order only affects speed; correctness never depends on it being fresh.
- **Two entry points with different costs.** `sdf_warp.sdf_contact_displacements` (what `compute_sdf_contact_displacements` dispatches to) uploads the mesh and downloads the displacements at every call, and those transfers, not the kernels, are most of its time. `sdf_warp.DeviceDeployment` (behind `StentDeployment`) uploads once and applies displacements on the device. It needs C-contiguous float32 point arrays to write back into, which the VTK point buffers are; for anything else `StentDeployment` falls back to the per-call route.
- `sdf_warp.py` must not use `from __future__ import annotations`: it turns kernel argument annotations into strings, which older Warp releases cannot resolve. Use `typing.Optional` there.
- The `warp-lang>=1.5` floor of the `warp` extra is the oldest release the suite was run against, and `backend.MINIMUM_WARP_VERSION` enforces it at run time, because an older Warp can be installed without the extra ever being asked for. Keep the two in step. Releases from 1.13 need Python 3.10; on Python 3.9 pip resolves to 1.12.

### Deformation kernels (`core/deformation.py`)

Each public `compute_*_displacements` is a NumPy-level wrapper around `@jx.jit` kernels and returns full-size per-vertex displacement arrays.

- **Aneurysm / stenosis** build a Householder matrix reflecting the centerline tangent onto z, evaluate the kernel in that local frame with the axial component zeroed, and reflect back. They move the surface only. Progress is the distance from the center point to a single "representative" surface vertex picked once up front by `find_stenosis_minimum_radius_representative` (used for aneurysms too, despite the name).
- **SDF contact (stent)** culls to the stent's padded bounding box, evaluates the smooth-min capsule-chain SDF for surface and centerline points in one batch, finds the contact set, uses a `cKDTree` over it to blend an influence zone, and scatters the result back to full size. It moves both surface and centerline. Progress is a radius incremented by the returned `step_size`.
- **The stent SDF has two code paths, kept separate on purpose.** A scalar radius with the default spherical caps is evaluated by the constant-radius chain (`capsule_sdf`, `smin_sdf_capsule_contact_sculpt`). Per-vertex radii, or `cap_height_fraction` below 1, make `compute_sdf_contact_displacements` dispatch to `variable_radius_capsule_sdf` / `smin_sdf_variable_radius_capsule_contact_sculpt`, which cost about 25% more per evaluation. Do not route the uniform stent through the variable-radius kernel or merge the two: the uniform stent is what most users run and must keep its speed and its exact results. `tests/test_deformation.py` fails if either kind of deployment takes the other path.
- In the variable-radius path the smooth minimum runs over the distance to the axis segments only. The fold also blends each segment's unclamped arc-length coordinate of the query point; the radius (a monotone cubic through the per-vertex radii) is evaluated there and subtracted after the smooth minimum, and only arc length beyond the two stent ends is shaped into the flattened half-ellipsoid caps. Subtracting per-segment radii or flattening caps inside the smooth minimum confines its blending to the joints and ripples the stent surface with the period of the segment length.
- The stent radius handed to the SDF kernel is the nominal radius **minus** the smooth-min constant `k` (`SMOOTHING_K_CM`), because the smooth-min blend inflates the capsule surface by up to `k` at segment junctions; `k` is added back wherever a radius is displayed or compared with the target. For a flared stent the CLI scales the whole radius profile with the nominal radius and subtracts `k` per vertex.
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
- `README.md` documents the GUI control rows, CLI flags, and the cm/mm default tables by hand; update it when any of those change. Its GPU-acceleration timings come from `benchmarks/stent_benchmark.py`; re-measure with that script rather than editing numbers by hand, and state the hardware.
