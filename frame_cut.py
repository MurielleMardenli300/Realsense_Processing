import open3d as o3d
import numpy as np

def filter_lower_fifth_flipped(input_path, output_path):
    pcd = o3d.io.read_point_cloud(input_path)
    pts = np.asarray(pcd.points)

    # Flip Y axis
    pts_flipped = pts.copy()
    pts_flipped[:, 1] = -pts[:, 1]

    # Now remove the lower fifth (lowest Y values in flipped space)
    y_min = pts_flipped[:, 1].min()
    y_max = pts_flipped[:, 1].max()
    y_threshold = y_min + (y_max - y_min) / 5

    mask = pts_flipped[:, 1] > y_threshold

    filtered_pts = pts_flipped[mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_pts)

    if pcd.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        normals_flipped = np.asarray(pcd.normals).copy()
        normals_flipped[:, 1] = -normals_flipped[:, 1]
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals_flipped[mask])

    o3d.io.write_point_cloud(output_path, filtered_pcd)
    print(f"Original:  {len(pts)} points")
    print(f"Filtered:  {len(filtered_pts)} points")
    print(f"Removed:   {len(pts) - len(filtered_pts)} points (y < {y_threshold:.4f})")
    print(f"Saved to:  {output_path}")

filter_lower_fifth_flipped("frame_extraction/results/test_aloys_dyn_calibration/pointcloud_00020.ply", "output_filtered_20.ply")