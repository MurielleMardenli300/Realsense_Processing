"""
quantify_breathing_motion.py

Whole-surface, landmark-free quantification of respiratory motion amplitude,
directly comparable between an external point cloud surface (e.g. abdominal
skin captured with a depth camera) and an internal liver surface / volume
(from MRI, read directly from DICOM). No model is fit here -- this only
quantifies and lets you compare.

Rather than tracking a single anatomical point, each frame's difference from
a reference frame is decomposed into:
  1. a RIGID component (bulk translation + rotation, found via ICP), and
  2. a residual NON-RIGID / local-deformation component (per-point nearest-
     neighbour displacement after removing the rigid component),
and whole-surface aggregate statistics (mean / RMS / 95th percentile / max
displacement magnitude) of each are reported, per frame. This mirrors how
the DIR/DVF liver-motion literature decomposes deformation into shift,
rotation, and pure deformation and reports the 3D displacement vector
magnitude aggregated over the whole organ, applied here directly to point
sets so the SAME metric can be computed for both surfaces.

Three ways to get a whole-surface amplitude time series:

  pointcloud   Folder of .ply files (e.g. RealSense abdominal captures).
  liver-mask   Per-frame binary liver segmentation, read from DICOM (see
               DICOM LAYOUT below). Surface extracted via marching cubes,
               then goes through the SAME rigid+residual pipeline as
               `pointcloud`.
  volume-dvf   Per-frame MRI images, read from DICOM, NO segmentation
               needed. Falls back to deformable registration (SimpleITK
               Demons) and aggregates the resulting displacement field's
               magnitude (optionally restricted to a static liver mask,
               also given as DICOM).
  compare      Takes two CSVs produced above and reports Pearson
               correlation + cross-correlation lag between their amplitude
               time series, after resampling onto a common time axis.

DICOM LAYOUT -- `liver-mask` and `volume-dvf` accept an --input directory in
either of two layouts, auto-detected:
  (a) FLAT: the directory directly contains one DICOM file per frame (e.g. a
      single-slice real-time cine sequence -- your bTFE_2Dnav-style data).
      Files are detected by content (the DICM magic bytes), not extension,
      since these commonly have none.
  (b) PER-FRAME SUBDIRECTORIES: the directory contains one subdirectory per
      frame, each holding a full DICOM series (multiple slices) for a 3D
      volume at that timepoint.
Frames are ordered by DICOM AcquisitionTime when available (falls back to
filename order otherwise, with a warning) -- NOT by alphabetical filename,
since UID-based DICOM filenames have no guaranteed relationship to
acquisition order. When AcquisitionTime is available, time_ms in the output
CSV is the REAL elapsed time since the first frame (not an assumed constant
frame interval), which avoids the rounding drift that a fixed nominal
interval would accumulate over a long sequence.

IMPORTANT -- UNITS: for the comparison step to mean anything, both sides
need to end up in the same physical unit (mm recommended). RealSense .ply
files are commonly in metres; some pipelines (yours included, based on
earlier work) may have already normalized coordinates to an arbitrary unit
sphere, which is NOT physically comparable to MRI mm without undoing that
normalization first. `pointcloud` prints the loaded cloud's bounding-box
size so you can sanity-check this, and takes an explicit --unit_scale
(multiplier applied to raw coordinates) rather than guessing.

Usage
-----
# external surface
python quantify_breathing_motion.py pointcloud \\
    --input /path/to/ply_folder --output_csv ./surface_motion.csv \\
    --unit_scale 1000.0   # metres -> mm

# internal liver, if you have per-frame segmentations (DICOM)
python quantify_breathing_motion.py liver-mask \\
    --input /path/to/liver_masks_dicom --output_csv ./liver_motion.csv

# internal liver, no segmentation available (DICOM MRI directly)
python quantify_breathing_motion.py volume-dvf \\
    --input /path/to/mri_dicom --output_csv ./liver_motion.csv \\
    --mask /path/to/static_liver_mask_dicom

# compare the two
python quantify_breathing_motion.py compare \\
    --csv_a ./surface_motion.csv --column_a residual_rms_mm \\
    --csv_b ./liver_motion.csv   --column_b residual_rms_mm
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import open3d as o3d
import pydicom
import SimpleITK as sitk
from scipy.spatial import cKDTree
from scipy.signal import correlate


# ══════════════════════════════════════════════════════════════════════════
# DICOM discovery, temporal ordering, and loading
# ══════════════════════════════════════════════════════════════════════════

def looks_like_dicom(path: Path) -> bool:
    """Detects a DICOM file by its magic bytes (128-byte preamble + "DICM"),
    regardless of extension -- MRI DICOMs frequently have none."""
    if not path.is_file():
        return False
    if path.name.upper() == "DICOMDIR":
        return False
    try:
        with open(path, "rb") as f:
            preamble = f.read(132)
    except OSError:
        return False
    return len(preamble) == 132 and preamble[128:132] == b"DICM"


def parse_dicom_time(time_str: str | None) -> float | None:
    """DICOM TM field ("HHMMSS.FFFFFF") -> seconds since midnight, or None."""
    if not time_str:
        return None
    time_str = time_str.strip()
    if len(time_str) < 6:
        return None
    try:
        return int(time_str[0:2]) * 3600 + int(time_str[2:4]) * 60 + float(time_str[4:])
    except ValueError:
        return None


def get_acquisition_time(path: Path) -> float | None:
    """AcquisitionTime (seconds since midnight) of a single DICOM file, or of
    the first DICOM file inside a directory (for the per-frame-subdirectory
    layout, where the first slice's time represents that frame's time)."""
    target = path
    if path.is_dir():
        candidates = sorted(p for p in path.iterdir() if looks_like_dicom(p))
        if not candidates:
            return None
        target = candidates[0]
    try:
        ds = pydicom.dcmread(str(target), stop_before_pixels=True, force=True)
    except Exception:
        return None
    return parse_dicom_time(str(getattr(ds, "AcquisitionTime", "")))


def find_frame_paths(input_dir: Path) -> list[Path]:
    """Auto-detects the DICOM layout (see module docstring) and returns one
    path per frame -- a file for the flat layout, a subdirectory for the
    per-frame-subdirectory layout."""
    subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if subdirs:
        return subdirs
    files = sorted(p for p in input_dir.iterdir() if looks_like_dicom(p))
    if not files:
        raise ValueError(f"No DICOM files or per-frame subdirectories found in {input_dir}")
    return files


def sort_frames_temporally(frame_paths: list[Path]) -> tuple[list[Path], list[float | None]]:
    """Orders frames by AcquisitionTime; falls back to the given (filename)
    order if any frame is missing it. Returns (sorted_paths, acq_times_s),
    acq_times_s aligned to the returned order (seconds since midnight, or
    None for every entry if unavailable)."""
    times = [get_acquisition_time(p) for p in frame_paths]
    if all(t is not None for t in times):
        order = sorted(range(len(frame_paths)), key=lambda i: times[i])
        return [frame_paths[i] for i in order], [times[i] for i in order]
    print("    [warn] AcquisitionTime missing for at least one frame -- "
          "falling back to filename order (verify this matches real "
          "acquisition order for your data).")
    return frame_paths, [None] * len(frame_paths)


def compute_time_ms(acq_times_s: list[float | None],
                     frame_interval_ms: float | None) -> list[float | str]:
    """Real elapsed time (ms) since the first frame, from AcquisitionTime
    when available -- avoids the rounding drift a fixed nominal interval
    would accumulate over a long sequence. Falls back to
    index * frame_interval_ms if AcquisitionTime wasn't available at all,
    or to "" (unset) if neither is available."""
    if all(t is not None for t in acq_times_s):
        t0 = acq_times_s[0]
        return [(t - t0) * 1000.0 for t in acq_times_s]
    if frame_interval_ms is not None:
        return [i * frame_interval_ms for i in range(len(acq_times_s))]
    return ["" for _ in acq_times_s]


def read_dicom_as_image(path: Path) -> sitk.Image:
    """Reads a single DICOM file (2D, or a multi-frame DICOM) if `path` is a
    file, or an entire DICOM series (3D volume) if `path` is a directory."""
    if path.is_dir():
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise RuntimeError(f"No readable DICOM series in {path}")
        possible_series = []
        for series_id in series_ids:
            filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_id)
            possible_series.append((series_id, list(filenames)))
        _, filenames = max(possible_series, key=lambda item: len(item[1]))
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filenames)
        return reader.Execute()
    return sitk.ReadImage(str(path))


# ══════════════════════════════════════════════════════════════════════════
# Core: rigid + residual whole-surface displacement, shared by point clouds
# and segmentation-derived surfaces
# ══════════════════════════════════════════════════════════════════════════

def rigid_register(source: np.ndarray, target: np.ndarray,
                    max_correspondence_distance: float) -> np.ndarray:
    """ICP-aligns `source` onto `target` ([N,3] each, no correspondence or
    equal size required). Returns the 4x4 transform mapping source -> target."""
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source.astype(np.float64))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64))

    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return result.transformation


def decompose_transform(transform: np.ndarray) -> tuple[float, float]:
    """(translation_magnitude, rotation_angle_deg) of a 4x4 rigid transform."""
    translation_magnitude = float(np.linalg.norm(transform[:3, 3]))
    R = transform[:3, :3]
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rotation_angle_deg = float(np.degrees(np.arccos(trace)))
    return translation_magnitude, rotation_angle_deg


def whole_surface_stats(magnitudes: np.ndarray) -> dict:
    """Aggregate stats over a 1D array of per-point displacement magnitudes."""
    return {
        "mean": float(magnitudes.mean()),
        "rms": float(np.sqrt((magnitudes ** 2).mean())),
        "p95": float(np.percentile(magnitudes, 95)),
        "max": float(magnitudes.max()),
    }


def frame_vs_reference(reference_pts: np.ndarray, frame_pts: np.ndarray,
                        icp_max_corr_dist: float) -> dict:
    """
    Core per-frame computation, used identically for external point clouds
    and internal (segmentation-derived) surfaces:

    1. ICP-align frame -> reference: gives rigid translation + rotation
       (the "bulk" component of motion, e.g. the whole abdomen shifting).
    2. Apply that transform to bring the frame into the reference's pose,
       then for every reference point take the distance to its nearest
       neighbour in the rigid-aligned frame: this residual is the LOCAL /
       non-rigid deformation left after removing bulk motion (e.g. the
       abdominal wall bulging on inhalation), aggregated over the whole
       surface rather than at one landmark.
    """
    transform = rigid_register(frame_pts, reference_pts, icp_max_corr_dist)
    translation_mag, rotation_deg = decompose_transform(transform)

    frame_pts_h = np.hstack([frame_pts, np.ones((len(frame_pts), 1))])
    frame_aligned = (transform @ frame_pts_h.T).T[:, :3]

    tree = cKDTree(frame_aligned)
    residual_dist, _ = tree.query(reference_pts, k=1, workers=-1)
    residual_stats = whole_surface_stats(residual_dist)

    return {
        "rigid_translation_mm": translation_mag,
        "rigid_rotation_deg": rotation_deg,
        "residual_mean_mm": residual_stats["mean"],
        "residual_rms_mm": residual_stats["rms"],
        "residual_p95_mm": residual_stats["p95"],
        "residual_max_mm": residual_stats["max"],
    }


def auto_icp_distance(points: np.ndarray) -> float:
    """Default ICP correspondence cutoff: 5% of the point set's bbox diagonal,
    used only when the caller doesn't pass an explicit value -- keeps the
    default sane regardless of whether coordinates are in m, mm, or normalized
    units, though you should still confirm real units before comparing across
    modalities (see the unit warning at the top of this file)."""
    bbox_min, bbox_max = points.min(0), points.max(0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    return float(diagonal * 0.05)


# ══════════════════════════════════════════════════════════════════════════
# Path 1: point cloud sequence (.ply) -- unchanged, not a DICOM input
# ══════════════════════════════════════════════════════════════════════════

def load_ply_points(path: Path, unit_scale: float) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64) * unit_scale
    if len(pts) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts


def analyze_pointcloud_sequence(input_dir: Path,
                                 reference_index: int = 0,
                                 frame_interval_ms: float | None = None,
                                 icp_max_corr_dist: float | None = None,
                                 unit_scale: float = 1.0) -> list[dict]:
    ply_files = sorted(input_dir.glob("*.ply"))
    if len(ply_files) < 2:
        raise ValueError(f"Need at least 2 .ply files in {input_dir}")

    reference_pts = load_ply_points(ply_files[reference_index], unit_scale)

    bbox_size = reference_pts.max(0) - reference_pts.min(0)
    print(f"[units check] reference cloud bbox size (x,y,z): {bbox_size} "
          f"(after unit_scale={unit_scale}) -- for a real abdominal surface "
          f"this should be roughly a few hundred mm if working in mm.")

    if icp_max_corr_dist is None:
        icp_max_corr_dist = auto_icp_distance(reference_pts)
        print(f"[auto] icp_max_corr_dist = {icp_max_corr_dist:.3f}")

    rows = []
    for i, fp in enumerate(ply_files):
        frame_pts = load_ply_points(fp, unit_scale)
        stats = frame_vs_reference(reference_pts, frame_pts, icp_max_corr_dist)
        stats["frame_index"] = i
        stats["source_file"] = str(fp)
        stats["time_ms"] = i * frame_interval_ms if frame_interval_ms else ""
        rows.append(stats)

    return rows


# ══════════════════════════════════════════════════════════════════════════
# Path 2: segmented liver mask sequence (DICOM) -> surface via marching cubes
# ══════════════════════════════════════════════════════════════════════════

def extract_surface_from_mask_image(image: sitk.Image) -> np.ndarray:
    """Extracts a binary mask image's surface as a point set in physical
    (mm) coordinates via marching cubes -- the same kind of space (mm) your
    point clouds should be in for a fair comparison."""
    from skimage.measure import marching_cubes

    array = sitk.GetArrayFromImage(image)  # z, y, x  (or y, x if 2D)
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # treat a single 2D mask as one slice
    spacing = image.GetSpacing()[::-1]
    if len(spacing) == 2:
        spacing = (1.0,) + spacing  # nominal slice thickness for a lone 2D frame

    if array.max() == 0:
        raise ValueError("Mask is empty")

    verts, _, _, _ = marching_cubes(array.astype(np.float32), level=0.5, spacing=spacing)
    verts_xyz = verts[:, ::-1]  # (z,y,x) -> (x,y,z)
    origin = np.array(image.GetOrigin())
    origin = origin if len(origin) == 3 else np.append(origin, 0.0)
    return verts_xyz + origin


def analyze_liver_mask_sequence(input_dir: Path,
                                 reference_index: int = 0,
                                 frame_interval_ms: float | None = None,
                                 icp_max_corr_dist: float | None = None,
                                 max_surface_points: int = 20000) -> list[dict]:
    frame_paths = find_frame_paths(input_dir)
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {input_dir}")
    frame_paths, acq_times_s = sort_frames_temporally(frame_paths)
    time_ms_values = compute_time_ms(acq_times_s, frame_interval_ms)

    def load_and_subsample(path):
        pts = extract_surface_from_mask_image(read_dicom_as_image(path))
        if len(pts) > max_surface_points:
            idx = np.random.choice(len(pts), max_surface_points, replace=False)
            pts = pts[idx]
        return pts

    reference_pts = load_and_subsample(frame_paths[reference_index])

    if icp_max_corr_dist is None:
        icp_max_corr_dist = auto_icp_distance(reference_pts)
        print(f"[auto] icp_max_corr_dist = {icp_max_corr_dist:.3f} mm")

    rows = []
    for i, fp in enumerate(frame_paths):
        frame_pts = load_and_subsample(fp)
        stats = frame_vs_reference(reference_pts, frame_pts, icp_max_corr_dist)
        stats["frame_index"] = i
        stats["source_file"] = str(fp)
        stats["time_ms"] = time_ms_values[i]
        rows.append(stats)

    return rows


# ══════════════════════════════════════════════════════════════════════════
# Path 3: full MRI DICOM sequence, no segmentation -> deformable reg. (DVF)
# ══════════════════════════════════════════════════════════════════════════

def analyze_volume_dvf_sequence(input_dir: Path,
                                 reference_index: int = 0,
                                 frame_interval_ms: float | None = None,
                                 mask_path: Path | None = None) -> list[dict]:
    """
    Fallback when per-frame liver segmentations aren't available: registers
    each frame to the reference with SimpleITK's Demons algorithm and
    aggregates the resulting displacement vector field's magnitude -- the
    same mean/RMS/p95/max-of-|d| statistics used in the DIR/DVF liver-motion
    literature -- restricted to `mask_path` if given (a single static liver
    mask, itself read from DICOM, in the reference frame's space), otherwise
    over the whole image.
    """
    frame_paths = find_frame_paths(input_dir)
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {input_dir}")
    frame_paths, acq_times_s = sort_frames_temporally(frame_paths)
    time_ms_values = compute_time_ms(acq_times_s, frame_interval_ms)

    reference_image = sitk.Cast(read_dicom_as_image(frame_paths[reference_index]),
                                 sitk.sitkFloat32)

    mask_array = None
    if mask_path is not None:
        mask_image = read_dicom_as_image(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_image) > 0

    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(50)
    demons.SetStandardDeviations(1.5)

    rows = []
    for i, fp in enumerate(frame_paths):
        moving_image = sitk.Cast(read_dicom_as_image(fp), sitk.sitkFloat32)
        # NOTE: assumes frames are already roughly spatially aligned in
        # physical space (same patient/coil position across one session) --
        # true for a single dynamic acquisition, not across separate scans.
        moving_resampled = sitk.Resample(moving_image, reference_image)

        displacement_field = demons.Execute(reference_image, moving_resampled)
        field_array = sitk.GetArrayFromImage(displacement_field)
        magnitude = np.linalg.norm(field_array, axis=-1)

        if mask_array is not None:
            magnitude = magnitude[mask_array]

        stats = whole_surface_stats(magnitude.ravel())
        rows.append({
            "frame_index": i,
            "source_file": str(fp),
            "time_ms": time_ms_values[i],
            "dvf_mean_mm": stats["mean"],
            "dvf_rms_mm": stats["rms"],
            "dvf_p95_mm": stats["p95"],
            "dvf_max_mm": stats["max"],
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════
# Comparison: correlate two amplitude time series
# ══════════════════════════════════════════════════════════════════════════

def load_csv_series(path: Path, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("time_ms") in (None, ""):
                raise ValueError(
                    f"{path} has no time_ms values -- re-run the analysis "
                    f"with usable AcquisitionTime data or --frame_interval_ms "
                    f"so the two series can be time-aligned."
                )
            times.append(float(row["time_ms"]))
            values.append(float(row[value_column]))
    return np.array(times), np.array(values)


def compare_series(times_a, values_a, times_b, values_b,
                    max_lag_ms: float = 2000, resample_ms: float = 50) -> dict:
    """Resamples both series onto a common time grid (linear interpolation),
    reports Pearson correlation and the lag (within +/- max_lag_ms) that
    maximizes cross-correlation, plus peak-to-peak/RMS amplitude for each."""
    t_start = max(times_a.min(), times_b.min())
    t_end = min(times_a.max(), times_b.max())
    if t_end <= t_start:
        raise ValueError("The two series don't overlap in time.")

    common_t = np.arange(t_start, t_end, resample_ms)
    a_interp = np.interp(common_t, times_a, values_a)
    b_interp = np.interp(common_t, times_b, values_b)

    a_norm = (a_interp - a_interp.mean()) / (a_interp.std() + 1e-12)
    b_norm = (b_interp - b_interp.mean()) / (b_interp.std() + 1e-12)
    pearson_r = float(np.corrcoef(a_norm, b_norm)[0, 1])

    max_lag_samples = int(max_lag_ms / resample_ms)
    xcorr = correlate(a_norm, b_norm, mode="full") / len(a_norm)
    lags = np.arange(-len(a_norm) + 1, len(a_norm))
    valid = np.abs(lags) <= max_lag_samples
    best_idx = np.argmax(np.abs(xcorr[valid]))
    best_lag_ms = float(lags[valid][best_idx] * resample_ms)

    return {
        "pearson_r": pearson_r,
        "best_lag_ms": best_lag_ms,
        "xcorr_at_best_lag": float(xcorr[valid][best_idx]),
        "n_samples_compared": len(common_t),
        "amplitude_a_p2p": float(values_a.max() - values_a.min()),
        "amplitude_b_p2p": float(values_b.max() - values_b.min()),
        "amplitude_a_std": float(values_a.std()),
        "amplitude_b_std": float(values_b.std()),
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def print_summary(rows: list[dict], keys: list[str]):
    print(f"\n{'=' * 60}\nSummary over {len(rows)} frames\n{'=' * 60}")
    for k in keys:
        vals = np.array([r[k] for r in rows if r[k] not in (None, "")])
        print(f"  {k:24s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"p2p={vals.max() - vals.min():.4f}  max={vals.max():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pc = sub.add_parser("pointcloud", help="Analyze a folder of .ply files")
    p_pc.add_argument("--input", required=True)
    p_pc.add_argument("--output_csv", required=True)
    p_pc.add_argument("--reference_index", type=int, default=0)
    p_pc.add_argument("--frame_interval_ms", type=float, default=None,
                       help="e.g. 66.67 for 15 fps -- needed for 'compare' later "
                            "(point clouds have no AcquisitionTime to fall back on)")
    p_pc.add_argument("--icp_max_corr_dist", type=float, default=None,
                       help="Same units as your points (post unit_scale). "
                            "Omit to auto-set from bbox size.")
    p_pc.add_argument("--unit_scale", type=float, default=1.0,
                       help="Multiplier applied to raw point coordinates, "
                            "e.g. 1000.0 to convert metres to mm. See the "
                            "units warning at the top of this file.")

    p_mask = sub.add_parser("liver-mask",
                             help="Analyze per-frame liver segmentation, read from DICOM")
    p_mask.add_argument("--input", required=True,
                         help="Directory of per-frame DICOM masks (flat files or "
                              "per-frame subdirectories -- see module docstring)")
    p_mask.add_argument("--output_csv", required=True)
    p_mask.add_argument("--reference_index", type=int, default=0)
    p_mask.add_argument("--frame_interval_ms", type=float, default=None,
                         help="Only used as a fallback if AcquisitionTime is missing")
    p_mask.add_argument("--icp_max_corr_dist", type=float, default=None)
    p_mask.add_argument("--max_surface_points", type=int, default=20000)

    p_vol = sub.add_parser("volume-dvf",
                            help="Analyze per-frame MRI DICOM via deformable registration "
                                 "(no segmentation needed)")
    p_vol.add_argument("--input", required=True,
                        help="Directory of per-frame DICOM images (flat files or "
                             "per-frame subdirectories -- see module docstring)")
    p_vol.add_argument("--output_csv", required=True)
    p_vol.add_argument("--reference_index", type=int, default=0)
    p_vol.add_argument("--frame_interval_ms", type=float, default=None,
                        help="Only used as a fallback if AcquisitionTime is missing")
    p_vol.add_argument("--mask", default=None,
                        help="Optional static liver mask, read from DICOM (a single "
                             "file or a series directory), in the reference frame's "
                             "space, to restrict statistics to the liver")

    p_cmp = sub.add_parser("compare",
                            help="Compare two amplitude time series (CSVs from the commands above)")
    p_cmp.add_argument("--csv_a", required=True)
    p_cmp.add_argument("--column_a", required=True,
                        help="e.g. residual_rms_mm or dvf_rms_mm")
    p_cmp.add_argument("--csv_b", required=True)
    p_cmp.add_argument("--column_b", required=True)
    p_cmp.add_argument("--max_lag_ms", type=float, default=2000)
    p_cmp.add_argument("--resample_ms", type=float, default=50)

    args = parser.parse_args()

    if args.command == "pointcloud":
        rows = analyze_pointcloud_sequence(
            Path(args.input), args.reference_index, args.frame_interval_ms,
            args.icp_max_corr_dist, args.unit_scale)
        write_csv(rows, Path(args.output_csv))
        print_summary(rows, ["rigid_translation_mm", "residual_mean_mm", "residual_rms_mm"])

    elif args.command == "liver-mask":
        rows = analyze_liver_mask_sequence(
            Path(args.input), args.reference_index, args.frame_interval_ms,
            args.icp_max_corr_dist, args.max_surface_points)
        write_csv(rows, Path(args.output_csv))
        print_summary(rows, ["rigid_translation_mm", "residual_mean_mm", "residual_rms_mm"])

    elif args.command == "volume-dvf":
        mask_path = Path(args.mask) if args.mask else None
        rows = analyze_volume_dvf_sequence(
            Path(args.input), args.reference_index, args.frame_interval_ms, mask_path)
        write_csv(rows, Path(args.output_csv))
        print_summary(rows, ["dvf_mean_mm", "dvf_rms_mm"])

    elif args.command == "compare":
        times_a, values_a = load_csv_series(Path(args.csv_a), args.column_a)
        times_b, values_b = load_csv_series(Path(args.csv_b), args.column_b)
        result = compare_series(times_a, values_a, times_b, values_b,
                                 args.max_lag_ms, args.resample_ms)
        print(f"\n{'=' * 60}\nComparison: {args.csv_a} [{args.column_a}]  vs.  "
              f"{args.csv_b} [{args.column_b}]\n{'=' * 60}")
        for k, v in result.items():
            print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()