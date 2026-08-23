import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
import glob, sys

# ============================================================
# LOADING
# ============================================================

def load_ply(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)


# ============================================================
# TEMPORAL AVERAGING
# ============================================================

def average_pointclouds_nn(
    frames: list[np.ndarray],
    n_points: int = None
) -> np.ndarray:
    """
    Nearest-neighbor averaging.

    Frame 0 is used as the reference. For every reference point,
    the nearest point in every other frame is found and averaged.
    """

    reference = frames[0]

    if n_points is not None and len(reference) > n_points:
        idx = np.random.choice(
            len(reference),
            n_points,
            replace=False
        )
        reference = reference[idx]

    accumulated = reference.copy().astype(np.float64)

    for frame in frames[1:]:
        tree = cKDTree(frame)

        distances, indices = tree.query(
            reference,
            k=1,
            workers=-1
        )

        accumulated += frame[indices]

    return (accumulated / len(frames)).astype(np.float32)


def average_pointclouds_icp(
    frames: list[np.ndarray],
    n_points: int = None
) -> np.ndarray:
    """
    ICP-based temporal averaging.

    All frames are registered to frame 0 before averaging.
    """

    def to_o3d(pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    reference_pts = frames[0]

    if n_points is not None and len(reference_pts) > n_points:
        idx = np.random.choice(
            len(reference_pts),
            n_points,
            replace=False
        )
        reference_pts = reference_pts[idx]

    reference_pcd = to_o3d(reference_pts)

    aligned_frames = [reference_pts]

    for frame in frames[1:]:

        source_pcd = to_o3d(frame)

        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            reference_pcd,
            max_correspondence_distance=0.05,
            estimation_method=
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50
                )
        )

        aligned_pts = (
            result.transformation[:3, :3] @ frame.T
        ).T + result.transformation[:3, 3]

        aligned_frames.append(aligned_pts)

    return average_pointclouds_nn(
        aligned_frames,
        n_points=n_points
    )


# ============================================================
# SMOOTHNESS METRICS
# ============================================================

def estimate_normals(
    points: np.ndarray,
    k: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate local surface normals using PCA.

    Returns:
        normals
        neighbor_indices
    """

    tree = cKDTree(points)

    distances, indices = tree.query(
        points,
        k=k + 1,
        workers=-1
    )

    # Remove the point itself
    neighbors = indices[:, 1:]

    normals = np.zeros_like(points)

    for i in range(len(points)):

        local_points = points[neighbors[i]]

        center = np.mean(local_points, axis=0)

        centered = local_points - center

        covariance = centered.T @ centered
        covariance /= max(len(local_points) - 1, 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Eigenvector corresponding to smallest eigenvalue
        # represents the local surface normal.
        normals[i] = eigenvectors[:, 0]

    normals /= (
        np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    )

    return normals, neighbors


def surface_roughness(
    points: np.ndarray,
    k: int = 30
) -> float:
    """
    Local surface roughness.

    For each point, fit a local tangent plane and measure
    the absolute distance of neighboring points from that plane.

    Lower = smoother surface.
    """

    tree = cKDTree(points)

    distances, indices = tree.query(
        points,
        k=k + 1,
        workers=-1
    )

    roughness = []

    for i in range(len(points)):

        neighbors = points[indices[i, 1:]]

        center = np.mean(neighbors, axis=0)

        centered = neighbors - center

        covariance = centered.T @ centered

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        normal = eigenvectors[:, 0]

        # Signed distance from neighbors to local plane
        signed_distances = centered @ normal

        roughness.append(
            np.sqrt(np.mean(signed_distances ** 2))
        )

    return float(np.mean(roughness))


def normalized_surface_roughness(
    points: np.ndarray,
    k: int = 30
) -> float:
    """
    Surface roughness normalized by local point spacing.

    This makes the metric less dependent on the absolute
    scale and density of the point cloud.

    Lower = smoother.
    """

    tree = cKDTree(points)

    distances, indices = tree.query(
        points,
        k=k + 1,
        workers=-1
    )

    roughness_values = []

    for i in range(len(points)):

        neighbors = points[indices[i, 1:]]

        center = np.mean(neighbors, axis=0)

        centered = neighbors - center

        covariance = centered.T @ centered

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        normal = eigenvectors[:, 0]

        # Distance from neighbors to local tangent plane
        residuals = np.abs(centered @ normal)

        # Typical local spacing
        local_spacing = np.mean(distances[i, 1:])

        if local_spacing > 1e-12:
            roughness_values.append(
                np.sqrt(np.mean(residuals ** 2))
                / local_spacing
            )

    return float(np.mean(roughness_values))


def surface_variation(
    points: np.ndarray,
    k: int = 30
) -> float:
    """
    PCA-based surface variation.

    lambda_0 / (lambda_0 + lambda_1 + lambda_2)

    lambda_0 is the smallest eigenvalue.

    Lower values generally indicate a locally planar/smooth
    surface.

    Important:
        This is NOT pure noise. Anatomical curvature can
        legitimately increase this metric.
    """

    tree = cKDTree(points)

    distances, indices = tree.query(
        points,
        k=k + 1,
        workers=-1
    )

    variations = []

    for i in range(len(points)):

        neighbors = points[indices[i, 1:]]

        centered = neighbors - np.mean(
            neighbors,
            axis=0
        )

        covariance = centered.T @ centered

        eigenvalues = np.linalg.eigvalsh(covariance)

        eigenvalues = np.maximum(eigenvalues, 0)

        total = np.sum(eigenvalues)

        if total > 1e-12:
            variations.append(
                eigenvalues[0] / total
            )

    return float(np.mean(variations))


def normal_variation(
    points: np.ndarray,
    k: int = 30
) -> float:
    """
    Mean angular variation between a point's normal
    and the normals of its neighbors.

    Lower = smoother.
    """

    normals, neighbors = estimate_normals(
        points,
        k=k
    )

    angles = []

    for i in range(len(points)):

        n = normals[i]
        neighbor_normals = normals[neighbors[i]]

        # Absolute dot product because normal orientation
        # can be locally flipped.
        cos_angles = np.abs(
            neighbor_normals @ n
        )

        cos_angles = np.clip(
            cos_angles,
            -1.0,
            1.0
        )

        local_angles = np.arccos(cos_angles)

        angles.extend(local_angles)

    return float(
        np.mean(angles)
    )


def normal_consistency(
    points: np.ndarray,
    k: int = 30
) -> float:
    """
    Average absolute cosine similarity between neighboring
    surface normals.

    1.0 = perfectly consistent normals
    0.0 = highly inconsistent

    Higher = smoother.
    """

    normals, neighbors = estimate_normals(
        points,
        k=k
    )

    consistency = []

    for i in range(len(points)):

        n = normals[i]

        neighbor_normals = normals[neighbors[i]]

        cos_similarity = np.abs(
            neighbor_normals @ n
        )

        consistency.extend(
            cos_similarity
        )

    return float(
        np.mean(consistency)
    )


def normalized_laplacian_roughness(
    points: np.ndarray,
    k: int = 30
) -> float:
    """
    Measures how far each point is from the centroid
    of its local neighborhood.

    Normalized by local neighborhood size.

    Lower = smoother.
    """

    tree = cKDTree(points)

    distances, indices = tree.query(
        points,
        k=k + 1,
        workers=-1
    )

    values = []

    for i in range(len(points)):

        neighbors = points[indices[i, 1:]]

        local_center = np.mean(
            neighbors,
            axis=0
        )

        displacement = np.linalg.norm(
            points[i] - local_center
        )

        local_spacing = np.mean(
            distances[i, 1:]
        )

        if local_spacing > 1e-12:

            values.append(
                displacement / local_spacing
            )

    return float(
        np.mean(values)
    )


def point_spacing_cv(
    points: np.ndarray,
    k: int = 10
) -> float:
    """
    Coefficient of variation of local point spacing.

    Lower = more uniform sampling.
    """

    tree = cKDTree(points)

    distances, _ = tree.query(
        points,
        k=k + 1,
        workers=-1
    )

    local_spacing = np.mean(
        distances[:, 1:],
        axis=1
    )

    mean_spacing = np.mean(local_spacing)
    std_spacing = np.std(local_spacing)

    if mean_spacing < 1e-12:
        return 0.0

    return float(
        std_spacing / mean_spacing
    )


# ============================================================
# COMPLETE EVALUATION
# ============================================================

def evaluate_pointcloud_smoothness(
    points: np.ndarray,
    k: int = 30
) -> dict:
    """
    Calculate all smoothness metrics for one point cloud.
    """

    print(
        f"    Evaluating smoothness "
        f"({len(points)} points, k={k})..."
    )

    metrics = {}

    metrics["surface_roughness"] = (
        surface_roughness(points, k)
    )

    metrics["normalized_surface_roughness"] = (
        normalized_surface_roughness(points, k)
    )

    metrics["surface_variation"] = (
        surface_variation(points, k)
    )

    metrics["normal_variation_rad"] = (
        normal_variation(points, k)
    )

    metrics["normal_variation_deg"] = (
        np.degrees(
            metrics["normal_variation_rad"]
        )
    )

    metrics["normal_consistency"] = (
        normal_consistency(points, k)
    )

    metrics["normalized_laplacian_roughness"] = (
        normalized_laplacian_roughness(points, k)
    )

    metrics["point_spacing_cv"] = (
        point_spacing_cv(points, min(k, 10))
    )

    return metrics


# ============================================================
# EVALUATE SAVED AVERAGED CLOUDS
# ============================================================

def evaluate_averaged_clouds(
    averaged: list[np.ndarray],
    k: int = 30
) -> list[dict]:
    """
    Evaluate every averaged point cloud.
    """

    all_metrics = []

    print("\n" + "=" * 70)
    print("POINT CLOUD SMOOTHNESS EVALUATION")
    print("=" * 70)

    for i, points in enumerate(averaged):

        print(
            f"\nFrame {i + 1}/{len(averaged)}"
        )

        metrics = evaluate_pointcloud_smoothness(
            points,
            k=k
        )

        metrics["frame"] = i

        all_metrics.append(metrics)

    return all_metrics


def summarize_metrics(
    metrics: list[dict]
) -> dict:
    """
    Average metrics over all evaluated point clouds.
    """

    if not metrics:
        return {}

    keys = [
        key
        for key in metrics[0]
        if key != "frame"
    ]

    summary = {}

    for key in keys:

        values = [
            m[key]
            for m in metrics
        ]

        summary[key] = float(
            np.mean(values)
        )

    return summary


def print_metrics(
    metrics: list[dict]
):
    """
    Print per-frame and overall smoothness metrics.
    """

    if not metrics:
        return

    print("\n")
    print("=" * 90)
    print("SMOOTHNESS METRICS")
    print("=" * 90)

    print(
        f"{'Frame':>8} "
        f"{'Roughness':>14} "
        f"{'Norm Rough':>14} "
        f"{'Surface Var':>14} "
        f"{'Normal Var°':>14} "
        f"{'Normal Cons':>14} "
        f"{'Laplacian':>14} "
        f"{'Spacing CV':>14}"
    )

    print("-" * 110)

    for m in metrics:

        print(
            f"{m['frame']:>8d} "
            f"{m['surface_roughness']:>14.6f} "
            f"{m['normalized_surface_roughness']:>14.6f} "
            f"{m['surface_variation']:>14.6f} "
            f"{m['normal_variation_deg']:>14.4f} "
            f"{m['normal_consistency']:>14.6f} "
            f"{m['normalized_laplacian_roughness']:>14.6f} "
            f"{m['point_spacing_cv']:>14.6f}"
        )

    summary = summarize_metrics(metrics)

    print("-" * 110)
    print(
        f"{'MEAN':>8} "
        f"{summary['surface_roughness']:>14.6f} "
        f"{summary['normalized_surface_roughness']:>14.6f} "
        f"{summary['surface_variation']:>14.6f} "
        f"{summary['normal_variation_deg']:>14.4f} "
        f"{summary['normal_consistency']:>14.6f} "
        f"{summary['normalized_laplacian_roughness']:>14.6f} "
        f"{summary['point_spacing_cv']:>14.6f}"
    )

    print("=" * 90)


# ============================================================
# TEMPORAL WINDOW AVERAGING
# ============================================================

def temporal_window_average(
    ply_paths: list[str],
    window_size: int = 15,
    method: str = "nn",
    n_points: int = 10000,
):
    frames = [
        load_ply(p)
        for p in ply_paths
    ]

    n_frames = len(frames)
    n_windows = n_frames // window_size

    print(
        f"  {n_frames} frames → "
        f"{n_windows} windows of {window_size}"
    )

    avg_fn = (
        average_pointclouds_icp
        if method == "icp"
        else average_pointclouds_nn
    )

    averaged = []
    original_windows = []

    for i in range(n_windows):

        window = frames[
            i * window_size:
            (i + 1) * window_size
        ]

        print(
            f"  Averaging window "
            f"{i + 1}/{n_windows} "
            f"({len(window)} frames)..."
        )

        avg_pts = avg_fn(
            window,
            n_points=n_points
        )

        averaged.append(avg_pts)
        original_windows.append(window)

    return averaged, original_windows


# ============================================================
# SAVE
# ============================================================

def save_averaged(
    averaged: list[np.ndarray],
    out_dir: str
):
    out = Path(out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, pts in enumerate(averaged):

        pcd = o3d.geometry.PointCloud()

        pcd.points = (
            o3d.utility.Vector3dVector(pts)
        )

        path = (
            out /
            f"averaged_{i:05d}.ply"
        )

        o3d.io.write_point_cloud(
            str(path),
            pcd
        )

        print(
            f"  Saved: {path} "
            f"({len(pts)} points)"
        )

def chamfer_distance(
    source: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Bidirectional Chamfer distance between two point clouds.

    For every point in source, find its nearest point in target,
    and vice versa.

    Returns the mean squared nearest-neighbor distance.

    Lower = more geometrically similar.
    """

    tree_target = cKDTree(target)
    tree_source = cKDTree(source)

    distances_source, _ = tree_target.query(
        source,
        k=1,
        workers=-1
    )

    distances_target, _ = tree_source.query(
        target,
        k=1,
        workers=-1
    )

    chamfer = (
        np.mean(distances_source ** 2) +
        np.mean(distances_target ** 2)
    ) / 2.0

    return float(chamfer)


def chamfer_distance_l2(
    source: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Bidirectional Chamfer distance using Euclidean distances
    instead of squared distances.

    This is easier to interpret because the result has the
    same physical units as the point coordinates.
    """

    tree_target = cKDTree(target)
    tree_source = cKDTree(source)

    distances_source, _ = tree_target.query(
        source,
        k=1,
        workers=-1
    )

    distances_target, _ = tree_source.query(
        target,
        k=1,
        workers=-1
    )

    chamfer = (
        np.mean(distances_source) +
        np.mean(distances_target)
    ) / 2.0

    return float(chamfer)


def hausdorff_distance(
    source: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Symmetric Hausdorff distance.

    Measures the largest nearest-neighbor discrepancy between
    the two point clouds.

    Lower = more geometrically similar.

    Note:
        Hausdorff distance is very sensitive to outliers.
    """

    tree_target = cKDTree(target)
    tree_source = cKDTree(source)

    distances_source, _ = tree_target.query(
        source,
        k=1,
        workers=-1
    )

    distances_target, _ = tree_source.query(
        target,
        k=1,
        workers=-1
    )

    hausdorff = max(
        np.max(distances_source),
        np.max(distances_target)
    )

    return float(hausdorff)


def hausdorff_distance_percentile(
    source: np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0
) -> float:
    """
    Robust Hausdorff-like distance.

    Instead of using the single worst point, use a percentile
    of the bidirectional nearest-neighbor distances.

    This is useful for anatomical point clouds because a single
    noisy/outlier point can otherwise dominate the Hausdorff metric.

    Lower = more geometrically similar.
    """

    tree_target = cKDTree(target)
    tree_source = cKDTree(source)

    distances_source, _ = tree_target.query(
        source,
        k=1,
        workers=-1
    )

    distances_target, _ = tree_source.query(
        target,
        k=1,
        workers=-1
    )

    all_distances = np.concatenate([
        distances_source,
        distances_target
    ])

    return float(
        np.percentile(
            all_distances,
            percentile
        )
    )
    

def evaluate_geometry_preservation(
    original_frames: list[np.ndarray],
    averaged: np.ndarray
) -> dict:
    """
    Compare an averaged point cloud against all original
    frames that contributed to it.

    Returns mean Chamfer and Hausdorff distances across
    the temporal window.
    """

    chamfer_values = []
    chamfer_l2_values = []
    hausdorff_values = []
    hausdorff_95_values = []

    for frame in original_frames:

        chamfer_values.append(
            chamfer_distance(
                averaged,
                frame
            )
        )

        chamfer_l2_values.append(
            chamfer_distance_l2(
                averaged,
                frame
            )
        )

        hausdorff_values.append(
            hausdorff_distance(
                averaged,
                frame
            )
        )

        hausdorff_95_values.append(
            hausdorff_distance_percentile(
                averaged,
                frame,
                percentile=95
            )
        )

    return {
        "chamfer": float(
            np.mean(chamfer_values)
        ),

        "chamfer_l2": float(
            np.mean(chamfer_l2_values)
        ),

        "hausdorff": float(
            np.mean(hausdorff_values)
        ),

        "hausdorff_95": float(
            np.mean(hausdorff_95_values)
        )
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    ply_dir = "/volatile/Datasets/Varian_Motion/point_clouds/testY" 
    out_dir = "/volatile/Datasets/Varian_Motion/point_clouds/testY/average1" 
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 15 
    method = sys.argv[4] if len(sys.argv) > 4 else "nn" 
    paths = sorted(glob.glob(f"{ply_dir}/*.ply")) 
    print(f"Found {len(paths)} PLY files in {ply_dir}")

    averaged, original_windows = temporal_window_average(
        paths,
        window_size=window_size,
        method=method
    )

    save_averaged(
        averaged,
        out_dir
    )

    print("\n" + "=" * 90)
    print("GEOMETRY PRESERVATION METRICS")
    print("=" * 90)

    for i, (avg, original_window) in enumerate(
        zip(averaged, original_windows)
    ):

        metrics = evaluate_geometry_preservation(
            original_window,
            avg
        )

        print(f"\nWindow {i}")

        print(
            f"  Chamfer distance:       "
            f"{metrics['chamfer']:.6f}"
        )

        print(
            f"  Chamfer L2 distance:    "
            f"{metrics['chamfer_l2']:.6f}"
        )

        print(
            f"  Hausdorff distance:     "
            f"{metrics['hausdorff']:.6f}"
        )

        print(
            f"  95% Hausdorff distance: "
            f"{metrics['hausdorff_95']:.6f}"
        )

    print("=" * 90)