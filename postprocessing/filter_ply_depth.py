"""
filter_pointclouds_by_bounds.py

Filters a folder of .ply point clouds by per-axis coordinate bounds and
saves the filtered clouds to a separate output folder, preserving filenames.

You can specify bounds for any subset of x, y, z. Axes you don't specify are
left unrestricted -- e.g. giving only --y_max keeps all x and z values and
filters purely by y-depth, keeping only points whose y-coordinate is at or
below that value.

Usage
-----
# keep only points with y <= 0.4 (all x, z kept)
python filter_pointclouds_by_bounds.py \\
    --input /path/to/ply_folder --output /path/to/filtered_folder \\
    --y_max 0.4

# keep only points within a box on all three axes
python filter_pointclouds_by_bounds.py \\
    --input /path/to/ply_folder --output /path/to/filtered_folder \\
    --x_min -0.3 --x_max 0.3 --y_max 0.5 --z_min -0.2 --z_max 0.2
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def build_mask(points: np.ndarray,
               x_min: float | None, x_max: float | None,
               y_min: float | None, y_max: float | None,
               z_min: float | None, z_max: float | None) -> np.ndarray:
    """
    Boolean keep-mask over `points` ([N,3]). An axis with both bounds None is
    left completely unrestricted (every point passes on that axis) -- this is
    what makes "only --y_max given" mean "keep all x, z, filter by y" rather
    than accidentally zeroing out x/z.
    """
    mask = np.ones(len(points), dtype=bool)

    bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    for axis, (lo, hi) in enumerate(bounds):
        if lo is not None:
            print(f"Filtering axis {axis} with lower bound {lo}")
            mask &= points[:, axis] >= lo
        if hi is not None:
            mask &= points[:, axis] <= hi

    return mask


def filter_ply(input_path: Path, output_path: Path,
                x_min, x_max, y_min, y_max, z_min, z_max) -> tuple[int, int]:
    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)
    has_normals = pcd.has_normals()
    has_colors = pcd.has_colors()

    mask = build_mask(points, x_min, x_max, y_min, y_max, z_min, z_max)

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[mask])
    if has_normals:
        normals = np.asarray(pcd.normals)
        filtered.normals = o3d.utility.Vector3dVector(normals[mask])
    if has_colors:
        colors = np.asarray(pcd.colors)
        filtered.colors = o3d.utility.Vector3dVector(colors[mask])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), filtered)

    return len(points), int(mask.sum())


def parse_args():
    p = argparse.ArgumentParser(
        description="Filter point clouds by per-axis coordinate bounds "
                     "(any subset of x/y/z -- unspecified axes are unrestricted)"
    )
    p.add_argument("--input", "-i", required=True, help="Folder of .ply files")
    p.add_argument("--output", "-o", required=True, help="Folder to write filtered .ply files")

    p.add_argument("--x_min", type=float, default=None)
    p.add_argument("--x_max", type=float, default=None)
    p.add_argument("--y_min", type=float, default=None)
    p.add_argument("--y_max", type=float, default=None)
    p.add_argument("--z_min", type=float, default=None)
    p.add_argument("--z_max", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if all(v is None for v in (args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)):
        raise ValueError("No bounds given -- specify at least one of "
                          "--x_min/--x_max/--y_min/--y_max/--z_min/--z_max.")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(input_dir.glob("*.ply"))
    if not ply_files:
        raise ValueError(f"No .ply files found in {input_dir}")

    print(f"Filtering {len(ply_files)} files")
    print(f"  x: [{args.x_min}, {args.x_max}]   "
          f"y: [{args.y_min}, {args.y_max}]   "
          f"z: [{args.z_min}, {args.z_max}]   (None = unrestricted)\n")

    total_before, total_after = 0, 0
    for fp in ply_files:
        output_path = output_dir / fp.name
        n_before, n_after = filter_ply(
            fp, output_path,
            args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max,
        )
        total_before += n_before
        total_after += n_after
        pct_kept = 100 * n_after / n_before if n_before else 0
        print(f"  {fp.name}: {n_before} -> {n_after} points ({pct_kept:.1f}% kept)")

    overall_pct = 100 * total_after / total_before if total_before else 0
    print(f"\nDone. {total_before} -> {total_after} points overall "
          f"({overall_pct:.1f}% kept). Saved to {output_dir}")


if __name__ == "__main__":
    main()