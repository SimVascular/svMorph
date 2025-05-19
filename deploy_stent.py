#!/usr/bin/env python3
"""deploy_stent.py
---------------------------------------------------------------
Command‑line tool that deploys a crimped stent inside a vascular
surface mesh until the stent radius reaches – but never exceeds –
a prescribed target radius.

The script wraps the Kelvinlet‑based *one‑step* deformation routine
`deform_mesh_sdf_contact(...)` you supplied.  We keep calling that
routine, accumulating the increment each time, **but we stop the
loop the moment adding the next increment would overshoot the target
radius.**  (In that case the last deformation is **not** applied –
this exactly matches your requirement "stop once this would overshoot".)

Usage
-----
python deploy_stent.py \
       --mesh   aneurysm_surface.vtp  \
       --cline  aneurysm_centerline.vtp  \
       --start   123                  # centre‑line point id of the distal tip of stent

optional flags  (see `-h` for the full list):
  --target‑R    0.40   # [cm] desired deployed stent radius
  --start‑R     0.05   # [cm] crimped radius (defaults to 0.05)
  --length      3.0    # [cm] stent length along centre‑line
  --out‑mesh    deployed_surface.vtp
  --out‑cl      deployed_centerline.vtp
"""
import argparse, sys, os, time
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v
from kelvinlet_core import vtk_utils, common
from kelvinlet_core import scaling_v2 as scaling

# -----------------------------------------------------------------------------
# minimal helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------
def read_vtp(fname:str)->vtk.vtkPolyData:
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(fname); r.Update();
    return r.GetOutput()

def write_vtp(poly:vtk.vtkPolyData, fname:str):
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(fname); w.SetInputData(poly); w.Write()

# -----------------------------------------------------------------------------
# core one‑step wrapper -------------------------------------------------------
# -----------------------------------------------------------------------------
def one_step(data,               # cached jnp data dict
             stent_vertices,     # jnp (m,3) sampled axis
             cur_R:float,
             eps:float, force_scale:float, a:float, b:float,
             halflength:float, target_R:float):
    """Perform one SDF‑contact Kelvinlet push; return ΔR actually produced."""
    # call the routine that returns displacements + step size
    disp, dR = scaling.get_sdf_contact_displacements(
        data, a, b, stent_vertices, eps, force_scale,
        surface_mesh_scale_factor=None,
        force_center_normal=None,
        stent_halflength=halflength,
        target_stent_radius=target_R,
        current_stent_radius=cur_R)

    # if applying dR would overshoot -> tell caller and *do not* change mesh
    if cur_R + dR > target_R:
        return 0.0  # target radius reached
    # otherwise apply displacement to surface mesh in‑place
    data["points"]["surface"] += disp
    return dR

# -----------------------------------------------------------------------------
# main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy a crimped stent by Kelvinlet SDF contact until its radius reaches the prescribed target.")
    ap.add_argument('--mesh',   required=True, help='input surface .vtp')
    ap.add_argument('--cline',  required=True, help='input center‑line .vtp')
    ap.add_argument('--start',   type=int, required=True, help='centre‑line vertex id indicating stent distal tip')
    ap.add_argument('--target-R', type=float, default=0.4, help='target deployed stent radius [cm]')
    ap.add_argument('--start-R',  type=float, default=0.05, help='initial crimped stent radius [cm]')
    ap.add_argument('--length',   type=float, default=3.0,  help='stent length along centre‑line [cm]')
    ap.add_argument('--out-mesh', default='deployed_surface.vtp', help='output surface mesh')
    ap.add_argument('--out-cl',   default='deployed_centerline.vtp', help='output center‑line (same topology, displaced verts)')
    args = ap.parse_args()

    mesh_pd  = read_vtp(args.mesh)
    cl_pd    = read_vtp(args.cline)

    # --- build cached JAX data once -----------------------------------------
    data = vtk_utils.polydata_to_np_jnp_data(mesh_pd, cl_pd)
    data = scaling.define_nodes_affine(data, [args.start])
    data = scaling.assign_force_location_affine_v2(data, args.start)

    # sample stent axis vertices --------------------------------------------
    parent_tip_map, seg_base_mask = vtk_utils.polydata_to_parent_tip_map(cl_pd)
    axis_pts = vtk_utils.sample_stent_axis_vertices(
        data['points']['centerline_points_view_np'], parent_tip_map, seg_base_mask,
        args.start, args.length, 0.1, sampling_direction=-1)

    # material constants for Kelvinlets --------------------------------------
    mu, nu = 1.0, 0.2
    a, b   = common.get_a_b(mu, nu)

    eps          = 0.2           # from your GUI defaults
    force_scale  = -1.0
    halflength   = 0.2           # per original variable naming

    cur_R = args.start_R
    print(f"Starting deployed‑radius = {cur_R:.4f} cm; target = {args.target_R:.4f} cm\n")
    it = 0
    while True:
        dR = one_step(data, axis_pts,
                       cur_R, eps, force_scale, a, b,
                       halflength, args.target_R)
        if dR <= 0.0:
            print("Next increment would overshoot – done.")
            break
        cur_R += dR
        it += 1
        print(f"  step {it:2d}:  ΔR = {dR:.5f}  →  R = {cur_R:.5f}")

    # ---------------------------------------------------------------------
    mesh_pd.GetPoints().SetData(n2v(data["points"]["surface"]))
    mesh_pd.Modified()
    write_vtp(mesh_pd, args.out_mesh)
    write_vtp(cl_pd,   args.out_cl)
    print(f"Saved:\n  surface  → {args.out_mesh}\n  center‑line → {args.out_cl}")

if __name__ == '__main__':
    main()
