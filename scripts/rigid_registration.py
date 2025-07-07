#!/usr/bin/env python3
"""
align_rigid.py

Compute the best-fit rigid (rotation + translation) transform between two meshes
given corresponding point IDs in a single text file.

Usage:
    pip install numpy scipy vtk
    python align_rigid.py mesh_preop.vtp mesh_postop.vtp correspondences.txt

- mesh_preop.vtp, mesh_postop.vtp: input VTK PolyData files
- correspondences.txt: each non-comment line contains two integers:
      <pointID_preop> <pointID_postop>
  Lines starting with '#' are ignored.
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import vtk


def rigid_transform(A: np.ndarray, B: np.ndarray):
    """
    Compute optimal rotation R and translation t
    to align A -> B in least-squares sense (Kabsch algorithm).
    A, B: (N×3) arrays of corresponding points.
    Returns (R_mat, t_vec).
    """
    # 1) Compute centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # 2) Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # 3) Compute covariance matrix
    H = AA.T @ BB

    # 4) SVD on covariance
    U, _, Vt = np.linalg.svd(H)

    # 5) Compute rotation matrix
    R_mat = Vt.T @ U.T

    # Reflection check (ensure a proper rotation, no flip)
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T

    # 6) Compute translation
    t = centroid_B - R_mat @ centroid_A

    return R_mat, t


def rotation_matrix_to_euler_xyz(R_mat: np.ndarray):
    """
    Convert a rotation matrix to extrinsic Euler angles (X→Y→Z) in degrees.
    Returns (roll_x, pitch_y, yaw_z).
    """
    rot = R.from_matrix(R_mat)
    angles = rot.as_euler('XYZ', degrees=True)
    return angles


def load_correspondences(pre_vtp: str, post_vtp: str, corr_file: str):
    """
    Reads two VTP meshes and a correspondence file with lines:
      id_preop id_postop
    Skips lines starting with '#'.
    Returns two arrays A, B of shape (N,3) containing point coordinates.
    """
    # Read pre-op mesh
    reader_pre = vtk.vtkXMLPolyDataReader()
    reader_pre.SetFileName(pre_vtp)
    reader_pre.Update()
    poly_pre = reader_pre.GetOutput()

    # Read post-op mesh
    reader_post = vtk.vtkXMLPolyDataReader()
    reader_post.SetFileName(post_vtp)
    reader_post.Update()
    poly_post = reader_post.GetOutput()

    A_pts = []
    B_pts = []
    with open(corr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line (expected 2 ints): {line}")
            id1, id2 = map(int, parts)

            # Validate point IDs
            n_pre = poly_pre.GetNumberOfPoints()
            n_post = poly_post.GetNumberOfPoints()
            if id1 < 0 or id1 >= n_pre:
                raise ValueError(f"Point ID {id1} out of range [0, {n_pre-1}] in {pre_vtp}")
            if id2 < 0 or id2 >= n_post:
                raise ValueError(f"Point ID {id2} out of range [0, {n_post-1}] in {post_vtp}")

            # Append coordinates
            A_pts.append(poly_pre.GetPoints().GetPoint(id1))
            B_pts.append(poly_post.GetPoints().GetPoint(id2))

    A = np.array(A_pts, dtype=float)
    B = np.array(B_pts, dtype=float)
    if A.shape[0] < 3:
        raise ValueError("At least 3 correspondences are required to compute a rigid transform.")

    return A, B


def main():
    parser = argparse.ArgumentParser(
        description="Rigid alignment from pre-op to post-op meshes using point ID correspondences."
    )
    parser.add_argument('pre_vtp', help="Path to pre-op mesh (.vtp)")
    parser.add_argument('post_vtp', help="Path to post-op mesh (.vtp)")
    parser.add_argument('corr_file', help="Correspondences file: each line 'id_pre id_post'")
    args = parser.parse_args()

    # Load corresponding point coordinates
    A, B = load_correspondences(args.pre_vtp, args.post_vtp, args.corr_file)

    # Compute rigid transform
    R_mat, t = rigid_transform(A, B)
    angles = rotation_matrix_to_euler_xyz(R_mat)

    # Output results
    print("\n🔧 Optimal rigid transform (pre-op → post-op):")
    print("Rotation matrix (R):")
    print(R_mat)
    print("\nEuler angles (extrinsic X→Y→Z) [degrees]:")
    print(f"  Roll  (X-axis): {angles[0]:.4f}°")
    print(f"  Pitch (Y-axis): {angles[1]:.4f}°")
    print(f"  Yaw   (Z-axis): {angles[2]:.4f}°")
    print("\nTranslation vector (t):")
    print(f"  t = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n")


if __name__ == '__main__':
    main()