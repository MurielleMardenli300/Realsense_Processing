"""
inspect_ply.py

Quick diagnostic for .ply point clouds -- confirms the file's actual format
(binary/ASCII, detected automatically either way) and prints per-axis value
ranges, so you can see whether your filter thresholds are actually inside
the data's range. Also saves a histogram per axis so you can see the shape
of the distribution, not just min/max.

Usage
-----
# inspect one file
python inspect_ply.py --input /path/to/pointcloud_00001.ply

# inspect several files at once (e.g. first + middle + last of a sequence)
python inspect_ply.py --input /path/to/ply_folder --sample 3

# save histograms too
python inspect_ply.py --input /path/to/pointcloud_00001.ply --save_histograms ./output
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def peek_ply_header(path: Path) -> list[str]:
    """Reads just the raw PLY header (up to 'end_header') so you can see the
    declared format line directly, regardless of what open3d infers."""
    lines = []
    with open(path, "rb") as f:
        for raw_line in f:
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError:
                # hit binary vertex data with no more header lines to decode
                break
            lines.append(line)
            if line == "end_header":
                break
    return lines


def inspect_file(path: Path, save_histograms: Path | None):
    print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")

    header_lines = peek_ply_header(path)
    format_line = next((l for l in header_lines if l.startswith("format")), "unknown")
    element_lines = [l for l in header_lines if l.startswith("element")]
    property_lines = [l for l in header_lines if l.startswith("property")]
    print(f"Declared format : {format_line}")
    for l in element_lines:
        print(f"  {l}")
    print(f"Properties       : {[l.split()[-1] for l in property_lines]}")

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    print(f"\nLoaded by open3d : {len(points)} points, "
          f"has_normals={pcd.has_normals()}, has_colors={pcd.has_colors()}")

    if len(points) == 0:
        print("  [!] Zero points loaded -- check the header/format above for issues.")
        return

    axis_names = ["x", "y", "z"]
    print(f"\n{'axis':>6} {'min':>12} {'max':>12} {'mean':>12} {'std':>12} "
          f"{'p1':>12} {'p50':>12} {'p99':>12}")
    for axis, name in enumerate(axis_names):
        col = points[:, axis]
        p1, p50, p99 = np.percentile(col, [1, 50, 99])
        print(f"{name:>6} {col.min():>12.5f} {col.max():>12.5f} {col.mean():>12.5f} "
              f"{col.std():>12.5f} {p1:>12.5f} {p50:>12.5f} {p99:>12.5f}")

    print(f"\nFirst 3 points:\n{points[:3]}")

    if save_histograms is not None:
        import matplotlib.pyplot as plt
        save_histograms.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for axis, (ax, name) in enumerate(zip(axes, axis_names)):
            ax.hist(points[:, axis], bins=80)
            ax.set_title(f"{name} distribution")
        plt.tight_layout()
        out_path = save_histograms / f"{path.stem}_histograms.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"\nSaved histograms to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True,
                         help="A single .ply file, or a folder to sample from")
    parser.add_argument("--sample", type=int, default=1,
                         help="If --input is a folder: how many files to inspect "
                              "(evenly spaced through the sorted list). Default 1 (first file).")
    parser.add_argument("--save_histograms", type=Path, default=None,
                         help="Optional output folder to save per-axis histogram PNGs")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        files_to_check = [input_path]
    else:
        all_files = sorted(input_path.glob("*.ply"))
        if not all_files:
            raise ValueError(f"No .ply files found in {input_path}")
        if args.sample >= len(all_files):
            files_to_check = all_files
        else:
            indices = np.linspace(0, len(all_files) - 1, args.sample, dtype=int)
            files_to_check = [all_files[i] for i in indices]

    for fp in files_to_check:
        inspect_file(fp, args.save_histograms)


if __name__ == "__main__":
    main()