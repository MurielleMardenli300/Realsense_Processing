"""
average_clouds.py — Temporal averaging of point cloud sequences.

Given a folder of N point clouds and a window size X, produces floor(N/X)
averaged point clouds using nearest-neighbour correspondence (fast, works well
for consecutive frames with small inter-frame motion) and reports smoothness
metrics for both the originals and the averages so you can quantify the
improvement.

Usage
-----
python average_clouds.py \\
    --input  /path/to/ply_folder \\
    --output /path/to/output_folder \\
    --window 7 \\
    --n_points 10000          # resample each cloud to this count before averaging
    --metric_samples 2000     # points used to compute smoothness metrics

Smoothness metrics reported (per cloud, then summarised)
---------------------------------------------------------
1. Mean k-NN distance (local point spacing)
   Lower = denser / smoother locally.

2. Surface variation (PCA curvature estimate)
   For each point, fit a local plane to its k nearest neighbours via PCA.
   Surface variation = λ_min / (λ_min + λ_mid + λ_max), where λ are the
   eigenvalues of the local covariance matrix.
   Range [0, 1].  0 = perfectly planar,  1 = isotropic / noisy ball.
   Lower = smoother surface.

3. Normal angular deviation (requires normals or estimates them)
   For each point, compute the mean angle between its normal and its
   neighbours' normals.  Lower = more consistent surface orientation = smoother.

These three metrics are standard in the point cloud processing literature
(see e.g. Rusu 2009, PCL paper; Bazazian et al. 2015 "Fast and Robust
Edge-Aware Normal Estimation").
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
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
        idx = np.concatenate([np.arange(N),
                               np.random.choice(N, n_points - N, replace=True)])
    return pts[idx]


def save_ply(pts: np.ndarray, path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    # estimate normals so the saved file is usable downstream
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=7),
        fast_normal_computation=True,
    )
    pcd.orient_normals_consistent_tangent_plane(7)
    o3d.io.write_point_cloud(path, pcd)


# ══════════════════════════════════════════════════════════════════════════════
# Averaging methods
# ══════════════════════════════════════════════════════════════════════════════

def average_nn(frames: list[np.ndarray]) -> np.ndarray:
    """
    Nearest-neighbour averaging.

    For each point in the reference frame (frame 0), find the single closest
    point in every other frame and average all their positions.  Correct for
    small inter-frame motion (consecutive depth frames at 15–30 fps); produces
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
# Smoothness metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_smoothness_metrics(pts: np.ndarray,
                                k: int = 7,
                                n_sample: int = 2000) -> dict:
    """
    Compute three smoothness metrics on a point cloud.

    Parameters
    ----------
    pts      : [N, 3] float32
    k        : neighbourhood size for all local estimates
    n_sample : number of points to sample for metric computation
               (set lower for speed, higher for accuracy)

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
        return dict(mean_nn_dist=np.nan,
                    mean_surf_variation=np.nan,
                    mean_normal_angle=np.nan)

    # subsample for speed
    sample_idx = (np.random.choice(N, min(n_sample, N), replace=False)
                  if N > n_sample else np.arange(N))
    sample = pts[sample_idx]

    tree = cKDTree(pts)
    dists, neigh_idx = tree.query(sample, k=k + 1, workers=-1)

    # 1. Mean nearest-neighbour distance (col 1 = closest non-self neighbour)
    mean_nn_dist = dists[:, 1].mean()

    # 2. Surface variation via local PCA
    surf_vars = []
    normal_angles = []

    # estimate normals for all points via PCA on k-NN
    # (reuse neighbour indices we already have for sample points)
    for i in range(len(sample)):
        nb = pts[neigh_idx[i, 1:]]          # [k, 3] — exclude self
        centroid = nb.mean(axis=0)
        centred  = nb - centroid
        cov      = (centred.T @ centred) / k
        eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
        # surface variation
        sv = eigvals[0] / (eigvals.sum() + 1e-12)
        surf_vars.append(sv)
        # normal of this point = eigenvector of smallest eigenvalue
        normal_i = eigvecs[:, 0]
        # compare to neighbour normals (compute inline for neighbours)
        nb_normals = []
        for j_idx in neigh_idx[i, 1:]:
            nb2 = pts[tree.query(pts[j_idx:j_idx+1], k=k+1)[1][0, 1:]]
            c2  = nb2.mean(0)
            ev2 = np.linalg.eigh((nb2 - c2).T @ (nb2 - c2) / k)[1]
            nb_normals.append(ev2[:, 0])
        nb_normals = np.array(nb_normals)   # [k, 3]
        # cos similarity, clamp to [-1,1] before arccos
        cos_sim  = np.clip(nb_normals @ normal_i, -1.0, 1.0)
        angles   = np.degrees(np.arccos(np.abs(cos_sim)))   # abs: ignore sign flip
        normal_angles.append(angles.mean())

    return dict(
        mean_nn_dist=float(mean_nn_dist),
        mean_surf_variation=float(np.mean(surf_vars)),
        mean_normal_angle=float(np.mean(normal_angles)),
    )


def print_metrics(label: str, m: dict):
    print(f"  {label}")
    print(f"    mean NN dist:       {m['mean_nn_dist']:.6f} m  "
          f"(lower = denser spacing)")
    print(f"    surface variation:  {m['mean_surf_variation']:.6f}    "
          f"(lower = smoother, 0=plane, 1=noisy)")
    print(f"    normal deviation:   {m['mean_normal_angle']:.3f} deg  "
          f"(lower = more consistent normals)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Temporal NN-average of point cloud windows + smoothness metrics"
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Folder containing .ply point cloud files")
    p.add_argument("--output", "-o", required=True,
                   help="Folder to write averaged .ply files")
    p.add_argument("--window", "-w", type=int, default=7,
                   help="Number of frames to average per output cloud (default: 7)")
    p.add_argument("--n_points", type=int, default=10000,
                   help="Resample each input cloud to this many points before "
                        "averaging (default: 10000)")
    p.add_argument("--metric_samples", type=int, default=2000,
                   help="Points used for smoothness metric computation "
                        "(lower = faster, default: 2000)")
    p.add_argument("--metric_k", type=int, default=7,
                   help="Neighbourhood size k for local PCA metrics (default: 7)")
    p.add_argument("--no_metrics", action="store_true",
                   help="Skip metric computation (faster, just produce clouds)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect and sort PLY files
    ply_files = sorted(in_dir.glob("*.ply"))
    if not ply_files:
        print(f"No .ply files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    W = args.window
    N = len(ply_files)
    n_windows = N // W

    if n_windows == 0:
        print(f"Only {N} files but window={W} — need at least {W} files.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {N} point clouds  →  {n_windows} windows of {W}")
    print(f"n_points per cloud: {args.n_points}")
    print(f"Output: {out_dir}\n")

    # ── per-window metrics accumulators ───────────────────────────────────────
    orig_metrics_all = []
    avg_metrics_all  = []

    for win_idx in tqdm(range(n_windows), desc="Windows"):
        window_files = ply_files[win_idx * W : (win_idx + 1) * W]

        # load all frames in this window
        frames = []
        for fp in window_files:
            try:
                frames.append(load_ply(str(fp), args.n_points))
            except Exception as e:
                print(f"  [warn] skipping {fp.name}: {e}")

        if len(frames) < 2:
            print(f"  [warn] window {win_idx}: fewer than 2 valid frames, skipping.")
            continue

        # ── metrics on the MIDDLE original frame (representative) ─────────────
        mid_frame = frames[len(frames) // 2]
        if not args.no_metrics:
            orig_m = compute_smoothness_metrics(
                mid_frame, k=args.metric_k, n_sample=args.metric_samples
            )
            orig_metrics_all.append(orig_m)

        # ── average ───────────────────────────────────────────────────────────
        avg_pts = average_nn(frames)

        # ── metrics on the averaged cloud ─────────────────────────────────────
        if not args.no_metrics:
            avg_m = compute_smoothness_metrics(
                avg_pts, k=args.metric_k, n_sample=args.metric_samples
            )
            avg_metrics_all.append(avg_m)

        # ── save ──────────────────────────────────────────────────────────────
        out_name = f"averaged_{win_idx:05d}.ply"
        save_ply(avg_pts, str(out_dir / out_name))

        # ── per-window report ─────────────────────────────────────────────────
        if not args.no_metrics:
            tqdm.write(f"\nWindow {win_idx:3d}  "
                       f"({window_files[0].name} … {window_files[-1].name})")
            print_metrics("Original (mid frame)", orig_m)
            print_metrics("Averaged            ", avg_m)
            sv_imp  = (orig_m['mean_surf_variation'] - avg_m['mean_surf_variation'])
            na_imp  = (orig_m['mean_normal_angle']   - avg_m['mean_normal_angle'])
            tqdm.write(f"    Δ surface variation: {sv_imp:+.6f}  "
                       f"({'improved' if sv_imp > 0 else 'worse'})")
            tqdm.write(f"    Δ normal deviation:  {na_imp:+.3f} deg  "
                       f"({'improved' if na_imp > 0 else 'worse'})")

    # ── aggregate summary ─────────────────────────────────────────────────────
    if not args.no_metrics and orig_metrics_all:
        print(f"\n{'═'*60}")
        print(f"  SUMMARY  ({n_windows} windows, W={W})")
        print(f"{'═'*60}")

        def agg(metric_list, key):
            vals = [m[key] for m in metric_list if not np.isnan(m[key])]
            return np.mean(vals), np.std(vals)

        keys = ['mean_nn_dist', 'mean_surf_variation', 'mean_normal_angle']
        labels = ['mean NN dist (m)   ', 'surface variation  ', 'normal deviation°  ']

        print(f"  {'Metric':<24}  {'Original':>16}  {'Averaged':>16}  {'Improvement':>12}")
        print(f"  {'-'*24}  {'-'*16}  {'-'*16}  {'-'*12}")
        for k, lab in zip(keys, labels):
            o_mu, o_sd = agg(orig_metrics_all, k)
            a_mu, a_sd = agg(avg_metrics_all,  k)
            delta = o_mu - a_mu
            pct   = 100 * delta / (o_mu + 1e-12)
            print(f"  {lab}  {o_mu:>8.5f}±{o_sd:.5f}  "
                  f"{a_mu:>8.5f}±{a_sd:.5f}  {pct:>+10.1f}%")

        print(f"\n  Interpretation:")
        print(f"  • surface variation: 0=plane, 1=isotropic noise. "
              f"Lower is smoother.")
        print(f"  • normal deviation:  angle between neighbouring normals. "
              f"Lower is smoother.")
        print(f"  • NN dist: local point spacing — averaging typically "
              f"doesn't change this much,")
        print(f"    since it's dominated by sampling density, not noise.")
        print()

    print(f"Done. {n_windows} averaged clouds saved to {out_dir}")


if __name__ == "__main__":
    main()