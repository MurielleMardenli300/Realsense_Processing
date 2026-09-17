"""
quantify_motion.py  —  Quantify frame-to-frame surface motion in a point cloud
                        sequence without requiring point correspondence.

Metrics computed per consecutive frame pair:
  1. Chamfer distance (mean + max)  — mean/max nearest-neighbour distance,
     a good proxy for "mean surface displacement in mm" without registration
  2. Hausdorff distance             — worst-case surface displacement
  3. Convex hull volume difference  — absolute volume change in mm³ (or m³
     depending on your coordinate units)
  4. Alpha-shape volume difference  — better than convex hull for non-convex
     surfaces like the abdomen (requires open3d >= 0.13)
  5. Centroid displacement          — how much the cloud's centre of mass moved

All distances are in whatever units your point clouds are stored in (typically
metres for RealSense output, so multiply by 1000 for mm).

Usage
-----
python quantify_motion.py \\
    --input  /path/to/ply_folder \\
    --unit_scale 1000             # multiply by 1000 to convert m → mm
    --output_csv results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_ply(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError(f"Empty cloud: {path}")
    return pts


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def chamfer(A: np.ndarray, B: np.ndarray):
    """
    Symmetric Chamfer distance.
    For each point in A find nearest in B, and vice versa.
    Returns (mean_A2B, mean_B2A, symmetric_mean, max_A2B, max_B2A).
    Units: same as input coordinates.
    """
    tree_B = cKDTree(B)
    tree_A = cKDTree(A)

    d_A2B, _ = tree_B.query(A, k=1, workers=-1)
    d_B2A, _ = tree_A.query(B, k=1, workers=-1)

    return dict(
        chamfer_mean_A2B = d_A2B.mean(),
        chamfer_mean_B2A = d_B2A.mean(),
        chamfer_mean_sym = (d_A2B.mean() + d_B2A.mean()) / 2,
        chamfer_max_A2B  = d_A2B.max(),
        chamfer_max_B2A  = d_B2A.max(),
        hausdorff        = max(d_A2B.max(), d_B2A.max()),
    )


def centroid_displacement(A: np.ndarray, B: np.ndarray) -> dict:
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    delta = cB - cA
    return dict(
        centroid_dx   = delta[0],
        centroid_dy   = delta[1],
        centroid_dz   = delta[2],
        centroid_dist = np.linalg.norm(delta),
    )


def convex_hull_volume(pts: np.ndarray) -> float:
    """
    Convex hull volume of a point cloud.
    For a concave surface like the abdomen this overestimates the true
    enclosed volume, but the CHANGE between frames is still meaningful.
    Returns 0.0 if hull construction fails (e.g. degenerate cloud).
    """
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def alpha_shape_volume(pts: np.ndarray, alpha: float = 0.05) -> float:
    """
    Volume estimate using open3d's alpha shape (better for concave surfaces).
    alpha controls how tightly the shape wraps around the points —
    smaller alpha = tighter fit, but more holes if points are sparse.

    Returns 0.0 if the resulting mesh is not watertight.
    """
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        if not mesh.is_watertight():
            return 0.0
        return mesh.get_volume()
    except Exception:
        return 0.0


def per_axis_displacement(A: np.ndarray, B: np.ndarray) -> dict:
    """
    For each point in A, find its nearest neighbour in B and record the
    displacement vector. Returns mean absolute displacement per axis.
    This is the closest thing to "absolute difference in x, y, z" without
    having explicit correspondences.
    """
    tree_B = cKDTree(B)
    _, idx  = tree_B.query(A, k=1, workers=-1)
    matched = B[idx]           # nearest point in B for each point in A
    delta   = matched - A      # displacement vector per point [N, 3]

    return dict(
        mean_abs_dx = np.abs(delta[:, 0]).mean(),
        mean_abs_dy = np.abs(delta[:, 1]).mean(),
        mean_abs_dz = np.abs(delta[:, 2]).mean(),
        mean_abs_d  = np.linalg.norm(delta, axis=1).mean(),
        std_abs_dx  = np.abs(delta[:, 0]).std(),
        std_abs_dy  = np.abs(delta[:, 1]).std(),
        std_abs_dz  = np.abs(delta[:, 2]).std(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  "-i", required=True,
                   help="Folder of .ply point cloud files (sorted alphabetically)")
    p.add_argument("--output_csv", "-o", default="motion_metrics.csv",
                   help="Output CSV path")
    p.add_argument("--unit_scale", type=float, default=1000.0,
                   help="Multiply coordinates by this to get mm. "
                        "1000 if stored in metres (RealSense default), "
                        "1 if already in mm.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Alpha parameter for alpha-shape volume (smaller = tighter).")
    p.add_argument("--skip_alpha", action="store_true",
                   help="Skip alpha-shape computation (faster).")
    p.add_argument("--no_summary", action="store_true",
                   help="Skip summary statistics at the end.")
    return p.parse_args()


def main():
    args = parse_args()
    s    = args.unit_scale     # convenience alias

    ply_files = sorted(Path(args.input).glob("*.ply"))
    if len(ply_files) < 2:
        print(f"Need at least 2 .ply files in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ply_files)} point clouds → {len(ply_files)-1} frame pairs")
    print(f"Unit scale: ×{s}  (output in {'mm' if s == 1000 else 'input units × ' + str(s)})\n")

    # pre-load all clouds
    clouds = []
    print("Loading clouds...")
    for fp in tqdm(ply_files):
        try:
            clouds.append((fp.name, load_ply(str(fp)) * s))
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")

    if len(clouds) < 2:
        print("Too few valid clouds.", file=sys.stderr)
        sys.exit(1)

    # compute metrics for each consecutive pair
    rows = []
    print("\nComputing metrics...")
    for i in tqdm(range(len(clouds) - 1)):
        name_A, A = clouds[i]
        name_B, B = clouds[i + 1]

        row = {"frame_A": name_A, "frame_B": name_B, "pair_idx": i}

        # Chamfer + Hausdorff
        row.update(chamfer(A, B))

        # per-axis displacement via nearest-neighbour correspondence
        row.update(per_axis_displacement(A, B))

        # centroid displacement
        row.update(centroid_displacement(A, B))

        # convex hull volumes
        vol_A = convex_hull_volume(A)
        vol_B = convex_hull_volume(B)
        row["convex_hull_vol_A"]    = vol_A
        row["convex_hull_vol_B"]    = vol_B
        row["convex_hull_vol_diff"] = abs(vol_B - vol_A)
        row["convex_hull_vol_pct"]  = (abs(vol_B - vol_A) / (vol_A + 1e-12) * 100
                                        if vol_A > 0 else 0.0)

        # alpha-shape volumes (optional)
        if not args.skip_alpha:
            # scale alpha by the same factor as coordinates
            al = args.alpha * s
            avol_A = alpha_shape_volume(A, alpha=al)
            avol_B = alpha_shape_volume(B, alpha=al)
            row["alpha_vol_A"]    = avol_A
            row["alpha_vol_B"]    = avol_B
            row["alpha_vol_diff"] = abs(avol_B - avol_A)
            row["alpha_vol_pct"]  = (abs(avol_B - avol_A) / (avol_A + 1e-12) * 100
                                      if avol_A > 0 else 0.0)

        rows.append(row)

    # write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.output_csv}")

    # summary
    if not args.no_summary and rows:
        unit = "mm" if s == 1000 else f"units×{s}"
        print(f"\n{'═'*58}")
        print(f"  SUMMARY  ({len(rows)} consecutive frame pairs)")
        print(f"  All distances in {unit},  volumes in {unit}³")
        print(f"{'═'*58}")

        def stat(key):
            vals = [r[key] for r in rows if key in r and r[key] is not None]
            if not vals:
                return "n/a"
            return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}  [max {np.max(vals):.3f}]"

        metrics = [
            ("Chamfer (symmetric mean)", "chamfer_mean_sym"),
            ("Hausdorff distance",       "hausdorff"),
            ("Mean abs ΔX",              "mean_abs_dx"),
            ("Mean abs ΔY",              "mean_abs_dy"),
            ("Mean abs ΔZ",              "mean_abs_dz"),
            ("Centroid displacement",    "centroid_dist"),
            ("Convex hull |ΔV|",        "convex_hull_vol_diff"),
            ("Convex hull |ΔV| %",       "convex_hull_vol_pct"),
        ]
        if not args.skip_alpha:
            metrics += [
                ("Alpha-shape |ΔV|",    "alpha_vol_diff"),
                ("Alpha-shape |ΔV| %",  "alpha_vol_pct"),
            ]

        for label, key in metrics:
            print(f"  {label:<30} {stat(key)}")

        # identify frames with largest motion
        max_chamfer_idx = max(range(len(rows)),
                              key=lambda i: rows[i]["chamfer_mean_sym"])
        print(f"\n  Largest motion pair: frames {rows[max_chamfer_idx]['frame_A']} → "
              f"{rows[max_chamfer_idx]['frame_B']} "
              f"(Chamfer = {rows[max_chamfer_idx]['chamfer_mean_sym']:.3f} {unit})")
        print()


if __name__ == "__main__":
    main()