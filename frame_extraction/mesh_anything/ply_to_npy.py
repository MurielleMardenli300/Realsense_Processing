import numpy as np
import open3d as o3d
import argparse
import os

def ply_to_npy(ply_path: str, output_path: str = None, save_colors: bool = True):
    """
    Convert a .ply point cloud file to .npy format.

    Saves:
        - xyz.npy        : float32 [N, 3] vertex positions
        - colors.npy     : float32 [N, 3] RGB colors in [0,1]  (if present)
        - combined.npy   : float32 [N, 6] xyz + rgb             (if colors present)
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"File not found: {ply_path}")

    print(f"Loading: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    n_down_points = len(pcd.points)
    print(f"Downsampled to: {n_down_points} points")

    # Compute normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.045, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)
    print(f"Normals OK: {pcd.has_normals()}")
    print(pcd)

    pts = np.asarray(pcd.points, dtype=np.float32)
    print(f"Points: {len(pts)}")

    # ── Determine output directory ────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(ply_path)[0]
        output_path = base + ".npy"

    out_dir  = os.path.dirname(os.path.abspath(output_path))
    out_stem = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    # ── Save XYZ with normals ──────────────────────────────────────────────────────────────
    xyz_path = os.path.join(out_dir, f"{out_stem}_xyz.npy")
    np.save(xyz_path, pts)
    print(f"Saved XYZ  → {xyz_path}  shape={pts.shape}")

    # ── Save normals if present ────────────────────────────────────────────────
    if save_colors and pcd.has_normals():
        cols = np.asarray(pcd.normals, dtype=np.float32)   # already in [0, 1]
        print(f"Normals of shape={cols.shape}")

        combined = np.concatenate([pts, cols], axis=1)     # [N, 6]
        comb_path = os.path.join(out_dir, f"{out_stem}_combined.npy")
        np.save(comb_path, combined)
        print(f"Saved XYZRGB → {comb_path}  shape={combined.shape}")
    else:
        print("No colors found — only XYZ saved.")

    # ── Also save normals if present ──────────────────────────────────────────
    # if pcd.has_normals():
    #     normals = np.asarray(pcd.normals, dtype=np.float32)
    #     norm_path = os.path.join(out_dir, f"{out_stem}_normals.npy")
    #     np.save(norm_path, normals)
    #     print(f"Saved normals → {norm_path}  shape={normals.shape}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .ply point cloud to .npy")
    parser.add_argument("ply_path",           help="Path to input .ply file", default='/home/rayan/Desktop/camera/Realsense_Processing/frame_extraction/results/6_30/test9/pointcloud_00020.ply')
    parser.add_argument("--output", "-o",     help="Output path/prefix (optional)", default='pc_npy')
    parser.add_argument("--no-colors",        action="store_true", help="Skip color export")
    args = parser.parse_args()

    ply_to_npy(args.ply_path, args.output, save_colors=not args.no_colors)