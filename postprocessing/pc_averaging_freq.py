"""
average_clouds_temporal.py — Time-aligned averaging of point cloud sequences.

Variant of average_clouds.py: instead of a fixed FRAME-COUNT window, windows
are sized to match a target TIME DURATION (e.g. the MRI acquisition period),
given the point cloud capture rate. This matters because the ratio of the two
frequencies is rarely an integer (15 fps / 590 ms MRI period ≈ 8.85 frames per
window, not 9) -- naively rounding to a fixed frame count every window causes
rounding error to accumulate, so later windows drift out of sync with the MRI
acquisitions. Window boundaries are instead computed from a cumulative
real-valued frame position and rounded only at each edge, so drift cannot
build up across the sequence (the same trick used for constant-rate frame
dropping/resampling elsewhere).

ASSUMPTION: point clouds were captured at a constant --pc_fps with no dropped
frames, and are named/sorted so that alphabetical order == acquisition order
(same assumption the original script made). If your capture pipeline logs a
real per-frame timestamp, prefer binning on those timestamps directly instead
of assuming constant fps -- that would be more robust to jitter/dropped
frames than this script's fixed-rate assumption. Ask if you'd like that
version; it's a small change (bin by loaded timestamp instead of by index).

Usage
-----
python average_clouds_temporal.py \\
    --input  /path/to/ply_folder \\
    --output /path/to/output_folder \\
    --pc_fps 15 \\
    --mri_interval_ms 590 \\
    --n_points 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers  (unchanged from average_clouds.py)
# ══════════════════════════════════════════════════════════════════════════════


def load_ply(path: str, n_points: int) -> np.ndarray:
    """Load a PLY, subsample/upsample to exactly n_points, return [N,3]."""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    N = len(pts)
    if N >= n_points:
        idx = np.random.choice(N, n_points, replace=False)
    else:
        idx = np.concatenate(
            [np.arange(N), np.random.choice(N, n_points - N, replace=True)]
        )
    return pts[idx]


def save_ply(pts: np.ndarray, path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=7),
        fast_normal_computation=True,
    )
    pcd.orient_normals_consistent_tangent_plane(7)
    o3d.io.write_point_cloud(path, pcd)


# ══════════════════════════════════════════════════════════════════════════════
# Time-aligned windowing  (NEW — replaces the fixed frame-count window)
# ══════════════════════════════════════════════════════════════════════════════


def compute_time_aligned_windows(
    n_total_frames: int, pc_fps: float, mri_interval_ms: float
) -> tuple[list[tuple[int, int]], float]:
    """
    Splits `n_total_frames` sequential point clouds (captured at a constant
    `pc_fps`) into windows whose duration matches `mri_interval_ms` as
    closely as possible.

    Window boundaries are computed from a running real-valued cumulative
    frame position (`cumulative += frames_per_window`), rounded to the
    nearest integer frame index only at each edge. This means individual
    windows will be either floor(frames_per_window) or ceil(frames_per_window)
    frames long, but the AVERAGE window length converges exactly to
    frames_per_window over the whole sequence -- so window N's start time
    never drifts away from N * mri_interval_ms, however long the recording.

    Returns
    -------
    windows           : list of (start_idx, end_idx) index pairs, end exclusive
    frames_per_window : the real-valued (non-rounded) frames-per-window ratio,
                         for reporting/diagnostics
    """
    frame_duration_ms = 1000.0 / pc_fps
    frames_per_window = mri_interval_ms / frame_duration_ms

    if frames_per_window < 1:
        raise ValueError(
            f"mri_interval_ms ({mri_interval_ms} ms) is shorter than a single "
            f"point cloud frame ({frame_duration_ms:.2f} ms at {pc_fps} fps) — "
            f"there's less than one frame per MRI period, so there's nothing "
            f"to average. Check --pc_fps / --mri_interval_ms."
        )

    windows = []
    cumulative = 0.0
    start_idx = 0

    while start_idx < n_total_frames:
        cumulative += frames_per_window
        end_idx = min(round(cumulative), n_total_frames)

        if end_idx <= start_idx:
            break

        windows.append((start_idx, end_idx))
        start_idx = end_idx

    # Drop a trailing window that's shorter than the others (not enough
    # frames left to represent a full MRI period) -- mirrors the original
    # script's floor(N/W) behaviour of discarding a partial final window.
    if len(windows) >= 2:
        typical_len = windows[0][1] - windows[0][0]
        last_len = windows[-1][1] - windows[-1][0]
        if last_len < typical_len:
            windows = windows[:-1]

    return windows, frames_per_window


# ══════════════════════════════════════════════════════════════════════════════
# Averaging methods  (unchanged from average_clouds.py)
# ══════════════════════════════════════════════════════════════════════════════


def average_nn(frames: list[np.ndarray]) -> np.ndarray:
    """
    Nearest-neighbour averaging.

    For each point in the reference frame (frame 0), find the single closest
    point in every other frame and average all their positions. Correct for
    small inter-frame motion (consecutive depth frames at 15-30 fps); produces
    a genuinely smoother cloud because random depth noise in each frame is
    suppressed by the average.

    Complexity: O(W · N · log N)  where W = window size, N = n_points.
    """
    reference = frames[0].copy().astype(np.float64)
    accumulated = reference.copy()

    for frame in frames[1:]:
        tree = cKDTree(frame)
        _, idx = tree.query(reference, k=1, workers=-1)
        accumulated += frame[idx]

    return (accumulated / len(frames)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Smoothness metrics  (unchanged from average_clouds.py)
# ══════════════════════════════════════════════════════════════════════════════


def compute_smoothness_metrics(
    pts: np.ndarray, k: int = 7, n_sample: int = 2000
) -> dict:
    """
    Compute three smoothness metrics on a point cloud.

    Returns
    -------
    dict with keys:
        mean_nn_dist        — mean nearest-neighbour distance (spacing)
        mean_surf_variation — mean local surface variation (curvature proxy)
        mean_normal_angle   — mean angular deviation between a point's normal
                              and its neighbours' normals (degrees)
    """
    N = len(pts)
    if N < k + 1:
        return dict(
            mean_nn_dist=np.nan, mean_surf_variation=np.nan, mean_normal_angle=np.nan
        )

    sample_idx = (
        np.random.choice(N, min(n_sample, N), replace=False)
        if N > n_sample
        else np.arange(N)
    )
    sample = pts[sample_idx]

    tree = cKDTree(pts)
    dists, neigh_idx = tree.query(sample, k=k + 1, workers=-1)

    mean_nn_dist = dists[:, 1].mean()

    surf_vars = []
    normal_angles = []

    for i in range(len(sample)):
        nb = pts[neigh_idx[i, 1:]]
        centroid = nb.mean(axis=0)
        centred = nb - centroid
        cov = (centred.T @ centred) / k
        eigvals, eigvecs = np.linalg.eigh(cov)
        sv = eigvals[0] / (eigvals.sum() + 1e-12)
        surf_vars.append(sv)
        normal_i = eigvecs[:, 0]
        nb_normals = []
        for j_idx in neigh_idx[i, 1:]:
            nb2 = pts[tree.query(pts[j_idx : j_idx + 1], k=k + 1)[1][0, 1:]]
            c2 = nb2.mean(0)
            ev2 = np.linalg.eigh((nb2 - c2).T @ (nb2 - c2) / k)[1]
            nb_normals.append(ev2[:, 0])
        nb_normals = np.array(nb_normals)
        cos_sim = np.clip(nb_normals @ normal_i, -1.0, 1.0)
        angles = np.degrees(np.arccos(np.abs(cos_sim)))
        normal_angles.append(angles.mean())

    return dict(
        mean_nn_dist=float(mean_nn_dist),
        mean_surf_variation=float(np.mean(surf_vars)),
        mean_normal_angle=float(np.mean(normal_angles)),
    )


def print_metrics(label: str, m: dict):
    print(f"  {label}")
    print(
        f"    mean NN dist:       {m['mean_nn_dist']:.6f} m  "
        f"(lower = denser spacing)"
    )
    print(
        f"    surface variation:  {m['mean_surf_variation']:.6f}    "
        f"(lower = smoother, 0=plane, 1=noisy)"
    )
    print(
        f"    normal deviation:   {m['mean_normal_angle']:.3f} deg  "
        f"(lower = more consistent normals)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Time-aligned NN-average of point cloud windows, matched "
        "to an external acquisition period (e.g. MRI), + smoothness metrics"
    )
    p.add_argument(
        "--input", "-i", required=True, help="Folder containing .ply point cloud files"
    )
    p.add_argument(
        "--output", "-o", required=True, help="Folder to write averaged .ply files"
    )

    p.add_argument(
        "--pc_fps",
        type=float,
        required=True,
        help="Point cloud acquisition rate in fps (e.g. 15)",
    )
    p.add_argument(
        "--mri_interval_ms",
        type=float,
        required=True,
        help="Target window duration in ms, i.e. the MRI "
        "acquisition period (e.g. 590)",
    )

    p.add_argument(
        "--n_points",
        type=int,
        default=10000,
        help="Resample each input cloud to this many points before "
        "averaging (default: 10000)",
    )
    p.add_argument(
        "--metric_samples",
        type=int,
        default=2000,
        help="Points used for smoothness metric computation "
        "(lower = faster, default: 2000)",
    )
    p.add_argument(
        "--metric_k",
        type=int,
        default=7,
        help="Neighbourhood size k for local PCA metrics (default: 7)",
    )
    p.add_argument(
        "--no_metrics",
        action="store_true",
        help="Skip metric computation (faster, just produce clouds)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(in_dir.glob("*.ply"))
    if not ply_files:
        print(f"No .ply files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    N = len(ply_files)
    windows, frames_per_window = compute_time_aligned_windows(
        N, args.pc_fps, args.mri_interval_ms
    )

    if not windows:
        print(
            f"Only {N} files, but need at least "
            f"{frames_per_window:.2f} frames per window.",
            file=sys.stderr,
        )
        sys.exit(1)

    frame_duration_ms = 1000.0 / args.pc_fps
    print(
        f"Found {N} point clouds at {args.pc_fps} fps "
        f"({frame_duration_ms:.2f} ms/frame)"
    )
    print(
        f"Target window duration: {args.mri_interval_ms} ms  "
        f"=> {frames_per_window:.3f} frames/window (non-integer, "
        f"so window lengths will alternate between "
        f"{int(frames_per_window)} and {int(frames_per_window) + 1} frames)"
    )
    print(f"{len(windows)} windows produced")
    print(f"n_points per cloud: {args.n_points}")
    print(f"Output: {out_dir}\n")

    orig_metrics_all = []
    avg_metrics_all = []

    for win_idx, (start_idx, end_idx) in enumerate(tqdm(windows, desc="Windows")):
        window_files = ply_files[start_idx:end_idx]
        window_len = end_idx - start_idx
        actual_duration_ms = window_len * frame_duration_ms

        frames = []
        for fp in window_files:
            try:
                frames.append(load_ply(str(fp), args.n_points))
            except Exception as e:
                print(f"  [warn] skipping {fp.name}: {e}")

        if len(frames) < 2:
            print(f"  [warn] window {win_idx}: fewer than 2 valid frames, skipping.")
            continue

        mid_frame = frames[len(frames) // 2]
        if not args.no_metrics:
            orig_m = compute_smoothness_metrics(
                mid_frame, k=args.metric_k, n_sample=args.metric_samples
            )
            orig_metrics_all.append(orig_m)

        avg_pts = average_nn(frames)

        if not args.no_metrics:
            avg_m = compute_smoothness_metrics(
                avg_pts, k=args.metric_k, n_sample=args.metric_samples
            )
            avg_metrics_all.append(avg_m)

        out_name = f"averaged_{win_idx:05d}.ply"
        save_ply(avg_pts, str(out_dir / out_name))

        if not args.no_metrics:
            tqdm.write(
                f"\nWindow {win_idx:3d}  "
                f"({window_files[0].name} … {window_files[-1].name})  "
                f"[{window_len} frames, {actual_duration_ms:.1f} ms "
                f"vs target {args.mri_interval_ms} ms]"
            )
            print_metrics("Original (mid frame)", orig_m)
            print_metrics("Averaged            ", avg_m)
            sv_imp = orig_m["mean_surf_variation"] - avg_m["mean_surf_variation"]
            na_imp = orig_m["mean_normal_angle"] - avg_m["mean_normal_angle"]
            tqdm.write(
                f"    Δ surface variation: {sv_imp:+.6f}  "
                f"({'improved' if sv_imp > 0 else 'worse'})"
            )
            tqdm.write(
                f"    Δ normal deviation:  {na_imp:+.3f} deg  "
                f"({'improved' if na_imp > 0 else 'worse'})"
            )

    if not args.no_metrics and orig_metrics_all:
        print(f"\n{'═'*60}")
        print(f"  SUMMARY  ({len(windows)} windows, target {args.mri_interval_ms} ms)")
        print(f"{'═'*60}")

        def agg(metric_list, key):
            vals = [m[key] for m in metric_list if not np.isnan(m[key])]
            return np.mean(vals), np.std(vals)

        keys = ["mean_nn_dist", "mean_surf_variation", "mean_normal_angle"]
        labels = ["mean NN dist (m)   ", "surface variation  ", "normal deviation°  "]

        print(
            f"  {'Metric':<24}  {'Original':>16}  {'Averaged':>16}  {'Improvement':>12}"
        )
        print(f"  {'-'*24}  {'-'*16}  {'-'*16}  {'-'*12}")
        for k, lab in zip(keys, labels):
            o_mu, o_sd = agg(orig_metrics_all, k)
            a_mu, a_sd = agg(avg_metrics_all, k)
            delta = o_mu - a_mu
            pct = 100 * delta / (o_mu + 1e-12)
            print(
                f"  {lab}  {o_mu:>8.5f}±{o_sd:.5f}  "
                f"{a_mu:>8.5f}±{a_sd:.5f}  {pct:>+10.1f}%"
            )
        print()

    print(f"Done. {len(windows)} averaged clouds saved to {out_dir}")


if __name__ == "__main__":
    main()
