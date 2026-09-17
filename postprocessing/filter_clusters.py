"""
remove_clusters.py  —  Remove isolated outlier point clusters (e.g. MRI bore
                        artefacts) from abdominal surface point clouds.

Method: DBSCAN clustering.  The abdominal surface forms one large, dense,
spatially connected cluster.  Bore artefacts are smaller isolated clusters
far from the surface.  DBSCAN labels every cluster and we keep only the
largest one — the abdomen.

Why not a distance threshold?
  Because the outlier clusters overlap the abdomen in at least one axis (y),
  so axis-aligned filtering would remove real surface points.  DBSCAN uses
  3D spatial density and connectivity, so it cleanly separates clusters that
  are spatially isolated even if they share an axis range.

Key parameters (tune if needed):
  --eps       Neighbourhood radius for DBSCAN in the same units as your
              point clouds (metres for RealSense).  Points within this
              distance of each other are considered neighbours.
              Default: 0.03 m (3 cm).  If the abdomen gets split into
              multiple clusters, increase this.  If bore artefacts get
              merged with the abdomen, decrease this.
  --min_pts   Minimum points to form a dense cluster core.  Isolated sparse
              groups below this count are labelled as noise.
              Default: 20.

Usage
-----
# process a single folder, write cleaned files to a new folder
python remove_clusters.py \\
    --input  /path/to/raw_plys \\
    --output /path/to/cleaned_plys \\
    --eps 0.03 \\
    --min_pts 20

# preview what would be removed WITHOUT writing any files
python remove_clusters.py --input /path/to/raw_plys --dry_run

# process recursively (one folder per participant, each containing .ply files)
python remove_clusters.py \\
    --input  /volatile/Datasets/Varian_Motion/point_clouds/6_30 \\
    --output /volatile/Datasets/Varian_Motion/point_clouds/cleaned \\
    --recursive
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# Core filtering
# ══════════════════════════════════════════════════════════════════════════════

def keep_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_pts: int,
    verbose: bool = False,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """
    Run DBSCAN on the point cloud and return only the largest cluster.

    Parameters
    ----------
    pcd     : input Open3D PointCloud
    eps     : DBSCAN neighbourhood radius (same units as point coordinates)
    min_pts : DBSCAN minimum cluster size

    Returns
    -------
    cleaned_pcd : PointCloud containing only the largest cluster
    info        : dict with diagnostic statistics
    """
    pts = np.asarray(pcd.points)
    N   = len(pts)

    if N == 0:
        return pcd, {"n_input": 0, "n_output": 0, "n_removed": 0,
                     "n_clusters": 0, "largest_cluster_size": 0}

    # DBSCAN — label -1 means noise (too sparse to form a cluster)
    labels = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1).fit_predict(pts)

    unique_labels, counts = np.unique(labels, return_counts=True)

    # separate noise label (-1) from real clusters
    cluster_labels = unique_labels[unique_labels >= 0]
    cluster_counts = counts[unique_labels >= 0]
    n_clusters     = len(cluster_labels)
    n_noise        = int((labels == -1).sum())

    if n_clusters == 0:
        # everything is noise — return original cloud unchanged
        info = {"n_input": N, "n_output": N, "n_removed": 0,
                "n_clusters": 0, "largest_cluster_size": 0,
                "warning": "DBSCAN found no clusters — eps/min_pts may be wrong"}
        return pcd, info

    # keep only the largest cluster
    largest_label = cluster_labels[np.argmax(cluster_counts)]
    mask          = (labels == largest_label)
    keep_pts      = pts[mask]

    cleaned = o3d.geometry.PointCloud()
    cleaned.points = o3d.utility.Vector3dVector(keep_pts)

    # preserve colours if present
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)[mask]
        cleaned.colors = o3d.utility.Vector3dVector(cols)

    # preserve normals if present
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals)[mask]
        cleaned.normals = o3d.utility.Vector3dVector(nrm)

    n_removed = N - int(mask.sum())

    if verbose and n_clusters > 1:
        print(f"    Clusters found: {n_clusters}  "
              f"(sizes: {sorted(cluster_counts, reverse=True)[:6]})")
        if n_noise > 0:
            print(f"    Noise points:   {n_noise}")

    info = {
        "n_input":              N,
        "n_output":             int(mask.sum()),
        "n_removed":            n_removed,
        "pct_removed":          100 * n_removed / N,
        "n_clusters":           n_clusters,
        "n_noise":              n_noise,
        "largest_cluster_size": int(mask.sum()),
        "cluster_sizes":        sorted(cluster_counts.tolist(), reverse=True),
    }
    return cleaned, info


# ══════════════════════════════════════════════════════════════════════════════
# File processing
# ══════════════════════════════════════════════════════════════════════════════

def process_file(
    in_path:  Path,
    out_path: Path,
    eps:      float,
    min_pts:  int,
    dry_run:  bool,
    verbose:  bool,
) -> dict:
    """Process one PLY file. Returns info dict for summary."""
    try:
        pcd = o3d.io.read_point_cloud(str(in_path))
    except Exception as e:
        return {"file": in_path.name, "error": str(e)}

    cleaned, info = keep_largest_cluster(pcd, eps=eps, min_pts=min_pts,
                                          verbose=verbose)
    info["file"] = in_path.name

    if "warning" in info:
        tqdm.write(f"  [warn] {in_path.name}: {info['warning']}")

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_path), cleaned)

    if verbose or info.get("pct_removed", 0) > 5:
        tqdm.write(
            f"  {in_path.name}: "
            f"{info['n_input']} → {info['n_output']} pts  "
            f"(-{info.get('pct_removed', 0):.1f}%  "
            f"{info['n_clusters']} cluster(s))"
        )

    return info


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Remove isolated outlier clusters from abdominal point clouds "
                    "using DBSCAN — keeps only the largest connected cluster."
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Input folder containing .ply files")
    p.add_argument("--output", "-o", default=None,
                   help="Output folder for cleaned files. "
                        "Defaults to <input>_cleaned alongside the input folder.")
    p.add_argument("--eps", type=float, default=0.03,
                   help="DBSCAN neighbourhood radius in point cloud units "
                        "(default: 0.03 m = 3 cm for RealSense output). "
                        "Increase if abdomen gets split; decrease if bore "
                        "artefacts merge with abdomen.")
    p.add_argument("--min_pts", type=int, default=20,
                   help="DBSCAN minimum cluster size (default: 20). "
                        "Groups smaller than this are treated as noise.")
    p.add_argument("--recursive", action="store_true",
                   help="Process subdirectories recursively (one subfolder "
                        "per participant, each containing .ply files).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be done without writing any files.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-file cluster statistics.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files (skipped by default).")
    return p.parse_args()


def collect_pairs(in_dir: Path, out_dir: Path, recursive: bool
                  ) -> list[tuple[Path, Path]]:
    """Collect (input_ply, output_ply) path pairs."""
    pairs = []
    if recursive:
        # mirror the subdirectory structure
        for sub in sorted(in_dir.iterdir()):
            if sub.is_dir():
                for ply in sorted(sub.glob("*.ply")):
                    out = out_dir / sub.name / ply.name
                    pairs.append((ply, out))
    else:
        for ply in sorted(in_dir.glob("*.ply")):
            pairs.append((ply, out_dir / ply.name))
    return pairs


def main():
    args   = parse_args()
    in_dir = Path(args.input)

    if not in_dir.exists():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = (Path(args.output) if args.output
               else in_dir.parent / (in_dir.name + "_cleaned"))

    pairs = collect_pairs(in_dir, out_dir, args.recursive)

    if not pairs:
        print(f"No .ply files found under {in_dir} "
              f"({'recursive' if args.recursive else 'non-recursive'} search).",
              file=sys.stderr)
        sys.exit(1)

    # skip already-processed files unless --overwrite
    if not args.dry_run and not args.overwrite:
        before = len(pairs)
        pairs  = [(i, o) for i, o in pairs if not o.exists()]
        skipped = before - len(pairs)
        if skipped:
            print(f"Skipping {skipped} already-processed files "
                  f"(use --overwrite to reprocess).")

    print(f"Processing {len(pairs)} files")
    print(f"  eps={args.eps} m  |  min_pts={args.min_pts}")
    print(f"  output → {out_dir}")
    if args.dry_run:
        print("  DRY RUN — no files will be written\n")

    all_info = []
    for in_path, out_path in tqdm(pairs, desc="Cleaning"):
        info = process_file(in_path, out_path,
                            eps=args.eps, min_pts=args.min_pts,
                            dry_run=args.dry_run, verbose=args.verbose)
        all_info.append(info)

    # ── summary ──────────────────────────────────────────────────────────────
    ok      = [r for r in all_info if "error" not in r]
    errors  = [r for r in all_info if "error" in r]
    flagged = [r for r in ok if r.get("n_clusters", 1) > 1]

    print(f"\n{'═'*56}")
    print(f"  SUMMARY  ({len(ok)} files processed)")
    print(f"{'═'*56}")

    if ok:
        removed_pct = [r["pct_removed"] for r in ok if "pct_removed" in r]
        clusters    = [r["n_clusters"]   for r in ok if "n_clusters"  in r]
        import numpy as np
        print(f"  Files with >1 cluster detected:  {len(flagged)} / {len(ok)}")
        if removed_pct:
            print(f"  Points removed  mean ± std:      "
                  f"{np.mean(removed_pct):.1f}% ± {np.std(removed_pct):.1f}%  "
                  f"[max {np.max(removed_pct):.1f}%]")
        if clusters:
            print(f"  Clusters found  mean ± std:      "
                  f"{np.mean(clusters):.1f} ± {np.std(clusters):.1f}  "
                  f"[max {int(np.max(clusters))}]")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors:
            print(f"    {r['file']}: {r['error']}")

    print(f"\n  Tip — if results look wrong, tune these parameters:")
    print(f"    eps too large  → bore artefacts merge with abdomen (fewer clusters found)")
    print(f"    eps too small  → abdomen itself gets split into many clusters")
    print(f"    min_pts too large → small real surface patches treated as noise")
    print(f"    min_pts too small → sparse artefact groups not filtered out")
    print(f"\n  To preview without writing: add --dry_run --verbose")
    print()


if __name__ == "__main__":
    main()