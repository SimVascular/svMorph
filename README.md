# svMorph

**Real-time interactive vascular morphing with SDFStent virtual stenting and regularized Kelvinlets**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)](https://www.python.org)
[![VTK 9.3](https://img.shields.io/badge/VTK-9.3-398593.svg)](https://vtk.org)
[![JAX](https://img.shields.io/badge/JAX-0.4.30-A435F0.svg)](https://github.com/jax-ml/jax)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](#)

---

svMorph is a research tool for *in-silico* morphological editing of patient-specific
vascular geometries. It enables interactive and scriptable editing on triangulated
surface meshes with associated centerlines through three core workflows: **SDFStent**
for virtual stenting, **regularized Kelvinlet** virtual aneurysm creation, and
geometry-based synthetic stenosis creation.

The deformation engine includes **SDFStent**, a signed-distance-field (SDF)-based
virtual stenting method that mimics stent-wall interaction during deployment, and
**regularized Kelvinlet** displacement kernels (de Goes & James, 2017, Pham et al., 2024)
for virtual aneurysm and synthetic stenosis modeling.
All heavy numerics are JIT-compiled with [JAX](https://github.com/jax-ml/jax) for
real-time feedback on commodity hardware.  SDFStent deployment can additionally run on
an NVIDIA GPU through [NVIDIA Warp](https://github.com/NVIDIA/warp), an optional backend
(see [GPU acceleration](#gpu-acceleration-nvidia-warp)).

> **Citation guidance**
> - If you use **SDFStent** (virtual stenting), please cite the **SDFStent paper** (reference forthcoming).
> - If you use **virtual aneurysm** or **virtual stenosis** creation, please cite:
>   *svMorph: Interactive Geometry-Editing Tools for Virtual Patient-Specific Vascular Anatomies*,
>   DOI: [10.1115/1.4056055](https://doi.org/10.1115/1.4056055).

---

## Key capabilities

| Mode | Description |
|---|---|
| **SDFStent deployment** | Signed-distance-field virtual stenting with expansion of a crimped capsule-chain stent against the vessel wall, with bounding-box culling, KD-tree influence blending, and smooth-min C&sup1;-continuous distance fields. |
| **SDFStent + straightening** | Same SDFStent deployment, with concurrent projection of the stent axis toward a straight line at each step. |
| **SDFStent, flared stent** | Same SDFStent deployment, with a radius profile along the stent axis (one or both ends flared, tapers, local narrowings) and optionally flattened end caps. |
| **Aneurysm creation** | Outward inflation at a centerline point via the scaling regularized Kelvinlet (F&nbsp;=&nbsp;s&middot;I). |
| **Stenosis creation** | Inward contraction using a truncated-sphere quartic bump profile. |

All modes are available through both the **interactive GUI** (PyQt6 + VTK) and
**headless CLI scripts** suitable for batch processing and CI pipelines, except
the flared stent, which is available through the CLI and the Python API only.

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.9+ | Tested with 3.9.19 |
| NumPy | 1.24+ | |
| SciPy | 1.10+ | KD-tree for influence-zone queries |
| VTK | 9.3 | VTP mesh I/O and rendering |
| JAX (CPU) | 0.4.30 | JIT compilation of deformation kernels |
| PyQt6 | 6.7 | GUI only — not required for headless scripts |
| NVIDIA Warp | 1.5+ | Optional — runs SDFStent deployment on an NVIDIA GPU; see [GPU acceleration](#gpu-acceleration-nvidia-warp) |

---

## Installation

### Option A &mdash; Express installation (recommended) (no need to clone repo)
1. Create and activate an environment (with `conda` or `uv`, etc.):

- **Conda**
```bash
conda create -y -n svmorph python=3.9
conda activate svmorph
```

- **uv**
```bash
uv venv --python 3.9
source .venv/bin/activate
```

2. Decide based on the use case, then install with `pip`:  

- to install svMorph with the full interactive GUI:
```bash
pip install "svmorph[gui]"
```
or if using uv:
```bash
uv pip install "svmorph[gui]"
```
- to install only the core deformation engine and scripts without the GUI:
```bash
pip install svmorph
```
or if using uv:
```bash
uv pip install svmorph
```
- to also run SDFStent deployment on an NVIDIA GPU (Windows or Linux), add the `warp`
  extra to either of the above, e.g.:
```bash
pip install "svmorph[gui,warp]"
```

### Option B &mdash; Manual installation from source (development)
Clone the repository first, 
```bash
git clone https://github.com/SimVascular/svMorph
```
then install the dependencies.  
For svMorph with the optional full GUI:
```bash
pip install -r requirements-gui.txt
```
For the deformation engine and scripts without the GUI:
```bash
pip install -r requirements.txt
```
For the optional NVIDIA GPU backend, additionally:
```bash
pip install "warp-lang>=1.5"
```

### Option C &mdash; Micromamba (might be preferred by Apple Silicon developers)

[Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
resolves native VTK binaries quickly and coexists with Homebrew and system Python.

1. Create environment with conda-forge packages:
```bash
micromamba create -y -n svmorph \
    python=3.9.19 \
    numpy=1.24.4 \
    scipy=1.10.1 \
    vtk=9.3.0 \
    -c conda-forge
```
2. Activate and install pip-only packages:
```bash
micromamba activate svmorph
pip install "jax[cpu]==0.4.30" "pyqt6==6.7"
```
3. Remove the duplicate Qt runtime pulled by VTK's conda deps:
```bash
micromamba remove -n svmorph qt6-main --force
```

### Platform notes

| Platform | Status |
|---|---|
| **macOS (Apple Silicon)** | Primary development platform.  Use Option B above. |
| **macOS (Intel)** | Option A or B.  Replace `osx-arm64` with `osx-64` if building micromamba from source. |
| **Linux (x86_64)** | Option A or B.  Both pip wheels and conda packages are available for all dependencies. |
| **Windows** | Option A (`pip install -r requirements.txt`) in a standard Python 3.9+ environment. |

### Verify installation

```bash
python -c "
import numpy, scipy, jax, PyQt6.QtCore
from vtkmodules.vtkCommonCore import vtkVersion
print(f'NumPy  {numpy.__version__}')
print(f'SciPy  {scipy.__version__}')
print(f'VTK    {vtkVersion.GetVTKVersion()}')
print(f'JAX    {jax.__version__}')
print(f'Qt     {PyQt6.QtCore.PYQT_VERSION_STR}')
"
```

This uses `vtkmodules.vtkCommonCore.vtkVersion` instead of `import vtk` so the check stays quick; it is the same `vtkVersion` class exposed as `vtk.vtkVersion` when you import the full `vtk` package.

---

## Usage

### Interactive GUI

If you installed `svmorph` directly from pip (Option A), launch with the installed console entry point:
```bash
svmorph-gui                    # default: units in cm, INFO logging on
svmorph-gui --units mm         # for editing millimeter geometry, units in mm
svmorph-gui --verbose          # show per-step timing
svmorph-gui --debug            # full diagnostic output
svmorph-gui --backend jax      # compute stent deployment with JAX even where Warp and a GPU are available
```

If you cloned this repository and are running from source (Option B, developer workflow), launch with:
```bash
python main.py                  # default: units in cm, INFO logging on
python main.py --units mm       # for editing millimeter geometry, units in mm
python main.py --verbose        # show per-step timing
python main.py --debug          # full diagnostic output
python main.py --backend jax    # compute stent deployment with JAX even where Warp and a GPU are available
```

**Workflow overview:**

1. Import a surface mesh and its centerline (VTP format).
2. Click a centerline point to select the deformation site.
3. Adjust stent dimensions, force scale, or pathology parameters via the slider panel.
4. Apply deformation — one step at a time or continuously.
5. Save the modified mesh to a new VTP file.

**Keyboard shortcuts:**

| Key | Action |
|---|---|
| `H` | Toggle stent visualization visibility |
| `D` (hold) | Continuous aneurysm deformation while held |

### GUI control panel reference

The control panel sits below the 3D viewport and is organized into five
horizontal rows.  Each row groups related controls for a specific workflow.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                         3D VTK Viewport                                  │
│                    (trackball rotate / zoom / pan)                        │
│                                                                          │
├──── Row 1 ── Stent geometry ─────────────────────────────────────────────┤
│ Stent Length (cm): ◄══════════╪══════════► [1.7000]                      │
│ Stent Diameter (cm): ◄══════════╪══════════► [0.8000]                    │
├──── Row 2 ── SDFStent deployment ────────────────────────────────────────┤
│ Force Scale: ◄══════════╪══════════► [1.000]                             │
│ [Select Point] [Straighten Stent] [Expand Stent (One Step)] [Expand …]  │
├──── Row 3 ── Stenosis ──────────────────────────────────────────────────┤
│ Stenosis Min Radius (cm): [0.1]   Stenosis Region Length (cm): [0.5]    │
│                          [Apply Stenosis (One Step)] [Apply Stenosis]    │
├──── Row 4 ── Aneurysm ──────────────────────────────────────────────────┤
│ Aneurysm Max Radius (cm): [0.5]                                         │
│ Sharpness: ◄══════════╪══════════► [1.000]                              │
│                        [Apply Aneurysm (One Step)] [Apply Aneurysm]     │
├──── Row 5 ── Utility ───────────────────────────────────────────────────┤
│ [Import Mesh] [Import Centerline]                                        │
│                [Camera Lock] [Visualize SDF] [Place Stent] [Save Mesh]  │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Row 1 — Stent geometry

| Control | Type | Description |
|---|---|---|
| **Stent Length** | slider + text | Length of the capsule-chain stent along the centerline (1.0–8.0 cm). Adjusting this recomputes and redraws the stent axis vertices in real time. |
| **Stent Diameter** | slider + text | Target deployed diameter of the stent (0.1–2.0 cm). This sets the radius goal for the SDFStent expansion loop. |

Both controls are bidirectional: dragging the slider updates the text field, and
typing a value snaps the slider to match.

#### Row 2 — SDFStent deployment

| Control | Type | Description |
|---|---|---|
| **Force Scale** | slider + text | Scales the magnitude of each displacement step (&#8722;1.0 to 1.0).  Higher values produce larger per-step deformations; negative values reverse the direction. |
| **Select Point** | button | Enters point-selection mode: the mesh becomes translucent and cyan glyphs appear at every centerline vertex. Click a vertex to select the stent's distal starting point. |
| **Straighten Stent** | button | Runs one combined step of SDFStent expansion **and** axis straightening — the stent is projected toward the line connecting its endpoints. |
| **Expand Stent (One Step)** | button | Executes a single SDFStent displacement iteration: the stent radius increments by one step and the vessel wall deforms outward at contact points. |
| **Expand Stent** | hold button | Hold to run continuous SDFStent expansion (fires every 50 ms). Release to stop. The title bar shows live FPS, current stent radius, and diameter. |

#### Row 3 — Stenosis creation

| Control | Type | Description |
|---|---|---|
| **Stenosis Min Radius** | text | Target minimum lumen radius at the stenosis center.  The deformation loop stops when the representative surface point reaches this radius. |
| **Stenosis Region Length** | text | Outer annular cutoff that controls the axial extent of the narrowing.  Larger values produce longer, more gradual stenoses. |
| **Apply Stenosis (One Step)** | button | Executes a single inward contraction step using the truncated-sphere quartic bump profile. |
| **Apply Stenosis** | hold button | Hold for continuous stenosis creation (fires every 25 ms). Release to stop. |

#### Row 4 — Aneurysm creation

| Control | Type | Description |
|---|---|---|
| **Aneurysm Max Radius** | text | Target maximum vessel radius at the aneurysm site. The loop stops when the representative surface point reaches this radius. |
| **Sharpness** | slider + text | Controls the Kelvinlet regularization parameter &epsilon;.  Low sharpness (&lt;&nbsp;1) produces broad, diffuse bulges; high sharpness (&gt;&nbsp;1) produces focal, concentrated expansions. The mapping is exponential (slider center = 1.0). |
| **Apply Aneurysm (One Step)** | button | Executes a single outward scaling Kelvinlet displacement step. |
| **Apply Aneurysm** | hold button | Hold for continuous aneurysm inflation (fires every 50 ms). Release to stop. |

#### Row 5 — Utility

| Control | Type | Description |
|---|---|---|
| **Import Mesh** | button | Opens a file dialog to load a surface `.vtp`.  The viewport reinitializes once both mesh and centerline are loaded. |
| **Import Centerline** | button | Opens a file dialog to load a centerline `.vtp`. |
| **Camera Lock** | toggle button | Locks the camera focal point to the currently selected centerline vertex.  The button turns orange when active.  Click again to release. |
| **Visualize SDF** | button | Evaluates the capsule-chain SDF on a 100&times;100&times;100 regular grid and renders the zero iso-surface via marching cubes.  Useful for inspecting stent geometry before or during deployment. |
| **Place Stent** | button | Commits the current stent visualization as a persistent actor in the scene, so it remains visible when selecting a new point. |
| **Save Mesh** | button | Opens a "Save As" dialog to write the deformed surface mesh to a new `.vtp` file. |

### Headless CLI scripts

All scripts accept `--help` for full argument documentation.

**Deploy a stent (SDFStent)** — expands a crimped stent (initial radius 0.05 cm) to a
deployed radius of 0.4 cm (diameter 0.8 cm), with a total stent length of
1.7 cm.  The distal tip is placed at centerline point ID 123.  Intermediate
snapshots are saved every 0.1 cm of radius change.

> **Note:** `--start-R` must be smaller than the local vessel radius at the
> deployment site so the stent begins fully inside the lumen.  A value of
> 0.05 cm works well for typical cardiovascular geometries.

```bash
python -m svmorph.scripts.deploy_stent \
    --mesh surface.vtp --cline centerline.vtp \
    --start 123 --target-R 0.4 --start-R 0.05 --length 1.7 \
    --save-step 0.1 \
    --out-mesh deployed_surface.vtp --out-cl deployed_centerline.vtp
```

**Deploy a flared stent (SDFStent, variable radius)** — same as above, but the
stent radius follows a profile along the centerline instead of being uniform:
the end opposite the `--start` point flares from the 0.4 cm body radius to
0.6 cm over the last 0.5 cm of the stent (funnel/trumpet shape).  Use
`--flare-end start` to flare the `--start` point side instead, or
`--flare-end both` to flare both ends.
`--cap-height-fraction` flattens the two stent end caps into half ellipsoids
(axial semi-axis = fraction × end radius); this keeps a wide flared end from
inflating a spherical bulge into the vessel beyond the stent end.  It only
changes the stent ends: the stent body follows the radius profile regardless,
including concave profiles (local narrowings).  Arbitrary radius profiles can
be built programmatically with `svmorph.core.stent_radius_profile` (control
points) or `svmorph.core.flared_stent_radius_profile` and passed as the
per-vertex `current_stent_radius` of `compute_sdf_contact_displacements`.

```bash
python -m svmorph.scripts.deploy_stent \
    --mesh surface.vtp --cline centerline.vtp \
    --start 123 --target-R 0.4 --start-R 0.05 --length 1.7 \
    --flare-R 0.6 --flare-length 0.5 --flare-end end \
    --cap-height-fraction 0.35 \
    --out-mesh deployed_surface.vtp --out-cl deployed_centerline.vtp
```

| Flag | Description |
|---|---|
| `--flare-R` | Stent radius at the tip of a flared end; omit for a uniform stent |
| `--flare-length` | Length of each flare transition along the stent axis (default 0.5 cm) |
| `--flare-end` | `end` (default, the side opposite `--start`), `start`, or `both` |
| `--cap-height-fraction` | Axial height of the two stent end caps as a fraction of the end radius, in (0, 1]; `1.0` (default) keeps spherical caps |

A uniform stent (no `--flare-R`, default `--cap-height-fraction`) is still
evaluated by the original constant-radius SDF, so its results and run time are
unaffected by these options.

**Deploy with concurrent axis straightening (SDFStent)** — same stent geometry as above,
but after each expansion step the stent axis is projected toward the straight
line connecting its endpoints (strength 0.075), gradually removing curvature
from the deployed configuration.

```bash
python -m svmorph.scripts.deploy_stent_straighten \
    --mesh surface.vtp --cline centerline.vtp \
    --start 123 --target-R 0.4 --start-R 0.05 --length 1.7 \
    --straightening-strength 0.075 \
    --out-mesh deployed_surface.vtp --out-cl deployed_centerline.vtp
```

**Create an aneurysm** — inflates the vessel wall at centerline point ID 456
until the local maximum radius reaches 0.5 cm.  Sharpness 1.0 gives a moderate
focal bulge; lower values spread the deformation over a wider region.  Snapshots
are saved every 0.02 cm of radius growth.

```bash
python -m svmorph.scripts.create_aneurysm \
    --mesh surface.vtp --cline centerline.vtp \
    --center 456 --target-R 0.5 --sharpness 1.0 --force-scale -1.0 \
    --save-step 0.02 --out-mesh aneurysm_surface.vtp
```

**Create a stenosis** — narrows the vessel at centerline point ID 789 until the
minimum lumen radius shrinks to 0.1 cm.  The stenosis region extends 0.5 cm
axially from the center.  Snapshots are saved every 0.02 cm of radius reduction.

```bash
python -m svmorph.scripts.create_stenosis \
    --mesh surface.vtp --cline centerline.vtp \
    --center 789 --target-R 0.1 --stenosis-length 0.5 --force-scale 1.0 \
    --save-step 0.02 --out-mesh stenosis_surface.vtp
```

### Common CLI flags

| Flag | Description |
|---|---|
| `--mesh` | Input surface `.vtp` (required) |
| `--cline` | Input centerline `.vtp` (required) |
| `--out-mesh` | Output surface path |
| `--out-cl` | Output centerline path (stent scripts) |
| `--save-step` | Write intermediate snapshots at this radius interval |
| `--units` | `cm` (default) or `mm` |
| `--verbose` | TIMING-level log output |
| `--debug` | DEBUG-level log output |
| `--backend` | Stent scripts only: `auto` (default), `jax` or `warp` — see [GPU acceleration](#gpu-acceleration-nvidia-warp) |

When `--save-step` is provided, intermediate results are written to a
`{out_mesh_stem}_intermediates/` directory as milestone VTP files.

---

## Input data format

svMorph operates on **VTK XML PolyData (`.vtp`)** files, the standard output of
[SimVascular](https://simvascular.github.io/) and other cardiovascular modeling
pipelines.

**Surface mesh** — a triangulated surface with point coordinates.

**Centerline** — a polyline with the following expected point data arrays
(produced by SimVascular's centerline extraction or VMTK):

| Array name | Type | Description |
|---|---|---|
| `MaximumInscribedSphereRadius` | scalar | MIS radius at each centerline point |
| `CenterlineSectionArea` | scalar | Cross-sectional lumen area |

Branching centerlines are supported; the `CenterlineId` array is used to construct
a parent-tip map for arc-length walks across bifurcations.  If the array is absent,
the centerline is treated as a single unbranched polyline.

---

## Unit system (cm vs mm)

svMorph defaults to **centimeters** when `--units` is unspecified (equivalent to
`--units cm`).  This matches the convention used by 
[SimVascular](https://simvascular.github.io/), where exported surface
meshes, centerlines and TetGen'ed mesh exteriors are typically in cm.  
Some pipelines (e.g. certain VMTK or 3D Slicer workflows) produce geometry 
in **millimeters** instead.

The `--units` flag (available on both the GUI and all CLI scripts) tells svMorph
which coordinate system your input files use.  When you switch to `--units mm`,
**all built-in default parameters are automatically scaled by a factor of 10** so
they remain physically correct — you do not need to manually convert them.

| Parameter | Default (cm mode) | Default (mm mode) |
|---|---|---|
| Stent diameter | 0.8 cm | 8.0 mm |
| Stent length | 1.7 cm | 17.0 mm |
| Target stent radius | 0.4 cm | 4.0 mm |
| Initial crimped radius | 0.05 cm | 0.5 mm |
| Flare transition length | 0.5 cm | 5.0 mm |
| Stenosis target radius | 0.1 cm | 1.0 mm |
| Influence radius (doi in paper) | 0.65 cm | 6.5 mm |
| Contact distance (doc in paper) | 0.001 cm | 0.01 mm|

**Key rules:**

1. **Match `--units` to your mesh.**  If your VTP coordinates are in millimeters,
   pass `--units mm`.  If they are in centimeters (SimVascular default), use the
   default `--units cm` or omit the flag.

2. **User-supplied values must be in the active unit.**  When you provide
   explicit arguments such as `--target-R 4.0` or `--length 17.0`, those numbers
   are interpreted in the unit system you selected.  In mm mode, `--target-R 4.0`
   means 4.0 mm; in cm mode it would mean 4.0 cm.

3. **GUI labels update automatically.**  Slider labels and text fields display
   the active unit name (e.g. "Stent Length (mm):") so there is no ambiguity
   while interacting.

4. **Internally, all constants live in centimeters** in `defaults.py` and are
   multiplied by the runtime scale factor `L()` from `units.py`.  If you add new
   spatial constants, follow the same pattern.

---

## GPU acceleration (NVIDIA Warp)

SDFStent deployment can run as [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels on a
CUDA GPU.  This is optional: JAX remains the reference implementation, and an installation
without Warp behaves exactly as before.  Once Warp is installed and finds a CUDA GPU it is
used automatically, unless you choose otherwise (see below).  Unlike JAX, which has no GPU
build for Windows, Warp runs on NVIDIA GPUs natively on both Windows and Linux.

### Setup

```bash
pip install "svmorph[warp]"        # or "svmorph[gui,warp]"
```

This needs an NVIDIA GPU with a current driver; Warp brings its own CUDA runtime, so no
CUDA toolkit has to be installed.  To check what Warp finds:

```bash
python -c "import warp as wp; wp.init(); print(wp.is_cuda_available(), wp.get_devices())"
```

### Choosing the backend

| Backend | What computes a stent deployment step |
|---|---|
| `auto` (default) | Warp if `warp-lang` 1.5 or newer is installed, Warp finds a CUDA GPU **and** the kernels load on it; otherwise JAX. If anything about Warp goes wrong on the way, it computes with JAX and logs a warning. |
| `jax` | JAX on the CPU, always. |
| `warp` | Warp. Stops with an error if Warp cannot be used, rather than computing with JAX. Without a CUDA GPU the kernels run on a single CPU core, which works but is not the speed-up a GPU gives. |

The backend can be named on the command line, from Python, or in the environment.  The
command-line flag and `set_backend()` are the same choice and take precedence; the
environment variable is consulted only when neither was used.

```bash
# On the command line of the stent scripts and the GUI
svmorph-deploy-stent --backend warp --mesh surface.vtp --cline centerline.vtp --start 123
svmorph-gui --backend jax

# In the environment, for runs that do not pass --backend
export SVMORPH_BACKEND=jax         # PowerShell: $env:SVMORPH_BACKEND = "jax"
```

```python
# From Python
from svmorph.core import set_backend, get_backend
set_backend("warp")                 # optionally: set_backend("warp", device="cuda:1")
get_backend()                       # "warp"
```

svMorph logs the backend it ends up using (`Stent deployment backend: Warp on cuda:0 (...)`).

### What runs where

Warp accelerates the SDFStent deployment step and nothing else.  Every other computation
stays on JAX/NumPy under either backend.

| Computation | `jax` backend | `warp` backend |
|---|---|---|
| Stent SDF — uniform stent, and stent with a radius profile or flattened caps | JAX kernels | Warp kernels |
| Bounding-box culling, contact set, movable points | JAX + NumPy | Warp kernels |
| Distance to the nearest point in contact (influence blending) | SciPy k-d tree | Warp kernels (Morton-ordered bounding-box tree) |
| Displacement assembly | NumPy | Warp kernels |
| Applying a step to the mesh | NumPy | On the GPU in the stent scripts (`StentDeployment`); NumPy in the GUI |
| Stent axis resampling, radius profiles, axis straightening | NumPy / JAX | NumPy / JAX |
| "Visualize SDF" iso-surface in the GUI | JAX | JAX |
| Aneurysm and stenosis creation | JAX | JAX |

The two backends compute the same step in single precision and agree to about 10<sup>-6</sup>
in the stent SDF and in a step's displacements.  On the patient-specific models used for
the timings below, the deployed geometry differs by less than 10<sup>-5</sup> cm between them
after a whole deployment.  Every Warp kernel is tested against its JAX counterpart
(`tests/test_sdf_warp.py`).

### Performance

Measured with `benchmarks/stent_benchmark.py` on a Ryzen 7 5800X3D and a GeForce RTX 3080
(Windows 11, `warp-lang` 1.19, JAX 0.12 on the CPU), compilation excluded.  "Near the
stent" is the share of the mesh inside the stent's padded bounding box, which is all the
JAX backend evaluates.  A 1.7 cm stent (2.5 cm where noted) is deployed from a radius of
0.05 cm to 0.4-0.5 cm.

**One deployment step with the stent in contact** — what the GUI computes per frame (ms):

| Mesh | Surface points | Near the stent | JAX (CPU) | Warp (CUDA) | Speed-up |
|---|---:|---:|---:|---:|---:|
| Patient model A | 58,228 | 7% | 6.5 | 0.44 | 15× |
| Patient model A, larger vessel | 58,228 | 13% | 7.6 | 0.46 | 17× |
| Patient model B | 158,330 | 14% | 14.2 | 1.03 | 14× |
| Patient model B, larger vessel, 2.5 cm stent | 158,330 | 35% | 27.0 | 1.02 | 26× |
| Synthetic tube | 12,997 | 31% | 5.5 | 0.42 | 13× |
| Synthetic tube | 49,977 | 30% | 10.8 | 0.55 | 20× |
| Synthetic tube | 200,070 | 30% | 37.2 | 1.66 | 22× |
| Synthetic tube | 999,844 | 30% | 229 | 4.21 | 54× |

**A whole deployment** (35 to 45 steps) — what `svmorph-deploy-stent` runs (ms):

| Mesh | Surface points | Near the stent | JAX (CPU) | Warp (CUDA) | Speed-up |
|---|---:|---:|---:|---:|---:|
| Patient model A | 58,228 | 7% | 170 | 6.2 | 27× |
| Patient model A, larger vessel | 58,228 | 13% | 203 | 7.0 | 29× |
| Patient model B | 158,330 | 14% | 355 | 7.7 | 46× |
| Patient model B, larger vessel, 2.5 cm stent | 158,330 | 35% | 810 | 9.2 | 88× |
| Synthetic tube | 12,997 | 31% | 118 | 6.5 | 18× |
| Synthetic tube | 49,977 | 30% | 249 | 7.2 | 35× |
| Synthetic tube | 200,070 | 30% | 796 | 10.9 | 73× |
| Synthetic tube | 999,844 | 30% | 5,312 | 29.4 | 181× |

The whole deployment gains more than a single step because the stent scripts keep the
mesh on the GPU between steps (`StentDeployment`), where a step costs about 0.2 ms up to
several hundred thousand points; a single step called on its own spends most of its time
copying the mesh to the GPU and the displacements back.  In the GUI the step is then a
small part of a frame, and the frame rate is set by rendering.

To measure on your own hardware and geometry:

```bash
python benchmarks/stent_benchmark.py
python benchmarks/stent_benchmark.py --mesh surface.vtp --cline centerline.vtp --start 123
```

### Things to know

- **The first step compiles.**  Warp compiles its kernels the first time they run on a
  device (about 5 s for CUDA and 8 s for the CPU on the machine above) and caches them, so
  later runs start immediately.
- **NVIDIA only.**  Warp's GPU support is CUDA.  On machines with an AMD or Intel GPU, and
  on macOS, `auto` computes with JAX.
- **GPU memory is kept between steps.**  About 90 bytes per mesh point stay allocated on the
  GPU (plus 12 bytes per point of page-locked host memory) so that the next step does not
  allocate them again.  `svmorph.core.sdf_warp.clear_cache()` releases the buffers of
  `compute_sdf_contact_displacements`; a `StentDeployment` releases its own when it is
  discarded.
- **Python API.**  `compute_sdf_contact_displacements` runs on the selected backend with no
  change to calling code.  To keep the mesh on the GPU across steps, deploy through a session:

  ```python
  from svmorph.core import StentDeployment

  with StentDeployment(data, target_stent_radius=0.4) as session:
      radius = 0.05
      while radius + session.step_size() <= 0.4:
          radius += session.step(stent_axis_vertices, radius)
  # data["points"] is up to date again here; inside the block, call session.sync() before reading it
  ```

- **Testing.**  Continuous integration runs the Warp kernels on Warp's CPU device, since
  hosted runners have no GPU; the CUDA results above come from running the same test suite
  and benchmark locally.

The Warp backend builds on a prototype contributed by [Andras Lasso](https://github.com/lassoan).

---

## Architecture

```
svmorph/
├── core/                  # Pure computation — no VTK, no Qt
│   ├── deformation.py     #   Kelvinlet kernels, SDFStent (SDF-contact), displacement assembly
│   ├── backend.py          #   Choice of the stent deployment backend (JAX or NVIDIA Warp)
│   ├── sdf_warp.py         #   SDFStent deployment step as NVIDIA Warp kernels (optional, GPU)
│   ├── stent_deployment.py #   StentDeployment — the stent step loop, on either backend
│   ├── geometry.py         #   Arc-length centerline resampling (branch-aware), stent radius profiles
│   ├── mesh_data.py        #   Material constants, displacement application
│   ├── defaults.py         #   Spatial default constants (cm)
│   └── units.py            #   Runtime cm ↔ mm unit scaling
│
├── visualization/         # VTK rendering and I/O
│   ├── vtk_io.py           #   VTP read/write, mesh array extraction, centerline utilities
│   ├── renderer.py         #   SceneManager — VTK pipeline construction
│   └── interactor.py       #   MeshInteractor — interactive trackball camera + deformation dispatch
│
├── gui/                   # PyQt6 application
│   └── main_window.py      #   MainWindow with sliders, buttons, and VTK render widget
│
├── scripts/               # Headless CLI entry points
│   ├── common.py           #   SimulationContext, SnapshotManager, shared CLI helpers
│   ├── deploy_stent.py
│   ├── deploy_stent_straighten.py
│   ├── create_aneurysm.py
│   └── create_stenosis.py
│
└── logging.py             # Structured logging with custom TIMING level
```

The `core/` subpackage is **dependency-light** (JAX, NumPy and SciPy; NVIDIA Warp only if
installed and selected) and carries no VTK or Qt dependency, making it straightforward for downstream tools — such as
[3D Slicer](https://www.slicer.org/) extensions or [ParaView](https://www.paraview.org/)
plugins — to import and build upon the deformation engine independently:

```python
from svmorph.core import (
    compute_sdf_contact_displacements,
    compute_aneurysm_displacements,
    compute_stenosis_displacements,
    resample_stent_axis,
    stent_radius_profile,          # per-vertex stent radii from control points
    flared_stent_radius_profile,   # per-vertex stent radii for flared ends
    StentDeployment,               # stent step loop; keeps the mesh on the GPU under Warp
    set_backend,                   # "auto", "jax" or "warp" for stent deployment
)
```

---

## Extending svMorph

### For 3D Slicer / ParaView plugin developers

The `svmorph.core` subpackage is intentionally free of VTK and Qt imports.
To build a downstream plugin:

1. **Import the deformation engine** from `svmorph.core`.
2. **Bridge your own mesh representation** to NumPy arrays matching the
   simulation data dictionary layout (see `vtk_io.extract_mesh_arrays` for the
   reference schema).
3. **Call displacement routines** and apply the returned arrays to your mesh.

Simulation data dictionary schema:

```python
data = {
    "points": {
        "surface": np.ndarray,      # (N_surf, 3) surface vertex coordinates
        "centerline": np.ndarray,   # (N_cl, 3)   centerline vertex coordinates
    },
    "nodes": {
        "all_indices": jnp.ndarray, # selected centerline point indices
        "force_center_point_id": int,
    },
}
```

### For contributors

- **Coding style** — type-annotated Python 3.9+, NumPy-style docstrings.
- **Logging** — use `from svmorph.logging import get_logger; logger = get_logger(__name__)`.
  Use `logger.timing(...)` for performance instrumentation.
- **Units** — store constants in centimetres in `defaults.py`; multiply by `L()`
  at runtime.  Never hard-code unit-dependent values outside `defaults.py`.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

- [VTK](https://vtk.org/) — 3D visualization and mesh processing
- [JAX](https://github.com/jax-ml/jax) — composable transformations and JIT compilation
- [NVIDIA Warp](https://github.com/NVIDIA/warp) — GPU kernels for SDFStent deployment
- [PyQt6](https://riverbankcomputing.com/software/pyqt/) — cross-platform GUI framework
- [SimVascular](https://simvascular.github.io/) — cardiovascular modeling pipeline

---

## Troubleshooting

### macOS (Apple Silicon), Option A

If `import jax` fails after a successful `pip install`, your interpreter may be **x86_64**
(Rosetta) instead of native ARM. On Apple Silicon, the JAX wheels pip installs expect a
native **arm64** Python; an x86_64 interpreter will not load those wheels.

Create or recreate the environment forcing the **osx-arm64** package subdir (and use
Python 3.9 as elsewhere in this README) before installing:

```bash
CONDA_SUBDIR=osx-arm64 conda create -y -n svmorph python=3.9
conda activate svmorph
pip install -r requirements-gui.txt   # or requirements.txt
```

(With plain **conda**, `CONDA_SUBDIR=osx-arm64` applies for that command; with **mamba**
/ **micromamba**, the same variable works the same way. If an old x86_64 env already
exists, prefer a new env name or remove the old one rather than mixing architectures.)

---

### PyQt6 6.10 freezes on macOS (as of March 20, 2026)

Qt 6.10 (released October 2025) introduced a regression in its macOS platform integration that causes PyQt6 applications launched from the command line to freeze immediately on startup — the window never appears and the process shows "Application Not Responding." This affects macOS Sequoia and macOS 26 Tahoe.

PyQt6 versions **6.7, 6.8, and 6.9** all work correctly up through macOS Tahoe. The `requirements-gui.txt` is pinned to `pyqt6>=6.7,<6.10` to avoid the broken release. If you are seeing this freeze, check which PyQt6 version is installed:

```bash
python -c "import PyQt6.QtCore; print(PyQt6.QtCore.PYQT_VERSION_STR)"
```

If the output is `6.10.x`, force-downgrade to a working version:

```bash
pip install "pyqt6>=6.7,<6.10"
```

---

### First launch is slow (up to ~2 minutes)

This is expected and only happens once. The deformation engine uses **JAX** with JIT (Just-In-Time) compilation: the first time each `@jax.jit`-decorated function is called, JAX traces it and compiles it to optimized XLA machine code for your hardware. There are several such functions in the deformation module, and each compilation step can take 20–30 seconds, adding up to roughly 1–2 minutes on the very first run.

JAX automatically caches the compiled artifacts to disk (typically `~/.jax_cache`). Every subsequent run loads those precompiled binaries directly, so startup is effectively instant.

**This is not a bug** — it is the standard JAX/XLA warm-up cost, paid once in exchange for fast GPU-accelerated numerics on all future runs. The cache persists across terminal sessions, so you only recompile if you:

- Delete the JAX cache manually
- Upgrade or change JAX/XLA/Python versions
- Switch to a different machine or hardware

---

## Contact

**Jeff Bohan Li**  
[Cardiovascular Biomechanics Computation Lab](https://cbcl.stanford.edu/), Stanford University  
bohan1@stanford.edu  
