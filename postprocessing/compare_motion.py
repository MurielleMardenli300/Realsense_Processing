"""
quantify_breathing_motion.py

Whole-surface, landmark-free quantification of respiratory motion amplitude,
directly comparable between an external point cloud surface (e.g. abdominal
skin captured with a depth camera) and an internal liver surface / volume
(from MRI, read directly from DICOM). No model is fit here -- this only
quantifies and lets you compare.

Five subcommands, producing (in the recommended workflow) 3 CSVs plus 2
analysis-only commands:

  pointcloud    CSV 1: PLY vs. a reference PLY frame (rigid translation +
                rotation, and residual non-rigid deformation after removing
                the rigid component -- see the module-level docstring further
                down for what these mean).
  volume-dvf    CSV 2: MRI vs. a reference MRI frame, via deformable
                registration (Demons) displacement-field magnitude. Works
                WITHOUT any liver segmentation -- pass no --mask and it
                aggregates over the whole image. This is the path to use
                if you don't have liver masks.
  volume-simple CSV 2 alternative, registration-free: threshold-based
                centroid displacement + normalized intensity difference vs.
                a reference frame, no mask needed either. Much faster and
                more transparent than volume-dvf's Demons registration --
                use this if volume-dvf gives you trouble (e.g. registration
                not converging, or you just want a quicker sanity check).
  liver-mask    Alternative to volume-dvf IF you do have per-frame liver
                segmentations -- same rigid+residual pipeline as pointcloud,
                applied to a segmentation-derived surface. Skip this if you
                don't have masks; volume-dvf covers that case already.
  frame_to_frame
                CSV 3: per-modality amplitude, PLY and MRI merged by frame
                index. --comparison_mode consecutive (default) compares each
                frame to the PREVIOUS frame (instantaneous motion -- noisy,
                won't trace a coherent breathing curve). --comparison_mode
                reference compares each frame to a FIXED reference frame
                instead (default the first frame) -- use this one if you
                want to actually SEE the cyclical breathing-amplitude
                pattern; it's the same computation as CSVs 1/2, merged into
                one file here for direct two-modality comparison.
  compare       Analysis only (no new CSV data): Pearson correlation +
                cross-correlation lag between two amplitude time series
                (e.g. CSV 1's residual vs. CSV 2's DVF magnitude).
  breath_cycles Analysis only: detects individual breathing cycles in one
                CSV's signal via peak detection, and reports per-breath
                peak-to-peak amplitude (mean +/- std) -- a single number
                pair per modality, more directly "amplitude of breaths"
                than a continuous curve or a correlation coefficient.

DICOM LAYOUT -- `liver-mask` and `volume-dvf` accept an --input directory in
either of two layouts, auto-detected:
  (a) FLAT: the directory directly contains one DICOM file per frame (e.g. a
      single-slice real-time cine sequence).
  (b) PER-FRAME SUBDIRECTORIES: the directory contains one subdirectory per
      frame, each holding a full DICOM series (multiple slices) for a 3D
      volume at that timepoint.
Frames are ordered by DICOM AcquisitionTime when available (falls back to
filename order otherwise, with a warning).

WHAT "RIGID TRANSLATION", "RIGID ROTATION" AND "RESIDUAL" MEAN (pointcloud /
liver-mask outputs): every frame is ICP-aligned onto the reference frame.
- rigid_translation_mm / rigid_rotation_deg: how far/how much the whole
  surface moved as a solid, undeformed block (a bulk shift or tilt -- could
  be real physiological translation, or could be patient drift / camera
  mount slip; worth checking separately from the signal you actually care
  about).
- residual_*_mm: what's left after undoing that best-fit rigid transform --
  genuine local SHAPE CHANGE (e.g. the abdominal wall actually bulging),
  which is usually the more meaningful breathing-amplitude signal.

IMPORTANT -- UNITS: both sides need to end up in the same physical unit (mm
recommended) for `compare`/`breath_cycles` comparisons to mean anything.
`pointcloud` prints the loaded cloud's bounding-box size so you can
sanity-check --unit_scale.

Usage
-----
# CSV 1: external surface vs. reference
python quantify_breathing_motion.py pointcloud \\
    --input /path/to/ply_folder --output_csv ./output/ply_vs_reference.csv \\
    --unit_scale 1000.0

# CSV 2: internal MRI vs. reference, NO mask needed
python quantify_breathing_motion.py volume-dvf \\
    --input /path/to/mri_dicom --output_csv ./output/mri_vs_reference.csv

# CSV 3: frame-to-frame, both modalities merged
python quantify_breathing_motion.py frame_to_frame \\
    --ply_dir /path/to/ply_folder --mri_dir /path/to/mri_dicom \\
    --output_csv ./output/ply_mri_frame_to_frame.csv --unit_scale 1000.0

# Analysis: correlate CSV 1 and CSV 2
python quantify_breathing_motion.py compare \\
    --csv_a ./output/ply_vs_reference.csv --column_a residual_rms_mm \\
    --csv_b ./output/mri_vs_reference.csv --column_b dvf_rms_mm

# Analysis: per-breath amplitude from CSV 1
python quantify_breathing_motion.py breath_cycles \\
    --csv ./output/ply_vs_reference.csv --column residual_rms_mm
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import open3d as o3d
import pydicom
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree
from scipy.signal import correlate, find_peaks


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


def get_instance_number(path: Path) -> int | None:
    """InstanceNumber of a single DICOM file, or of the first DICOM file
    inside a directory (per-frame-subdirectory layout). Used for ordering
    instead of AcquisitionTime, which can repeat identically across
    genuinely different, visually distinct frames in some acquisitions
    (confirmed by visual inspection) and would otherwise group unrelated
    frames together under one stale timestamp."""
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
    value = getattr(ds, "InstanceNumber", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def looks_like_image_file(path: Path) -> bool:
    """DICOM (by magic bytes) OR a .nii/.nii.gz file -- lets find_frame_paths
    work directly on phase volumes built by build_phase_volumes.py, in
    addition to raw DICOM sequences."""
    if not path.is_file():
        return False
    if path.name.lower().endswith((".nii", ".nii.gz")):
        return True
    return looks_like_dicom(path)


def find_frame_paths(input_dir: Path) -> list[Path]:
    """Auto-detects the layout (see module docstring) and returns one path
    per frame -- a file for the flat layout, a subdirectory for the
    per-frame-subdirectory layout (DICOM series only; NIfTI is always flat)."""
    subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if subdirs:
        return subdirs
    files = sorted(p for p in input_dir.iterdir() if looks_like_image_file(p))
    if not files:
        raise ValueError(f"No DICOM or NIfTI files, or per-frame subdirectories, found in {input_dir}")
    return files


def sort_frames_temporally(frame_paths: list[Path]) -> tuple[list[Path], list[int | None]]:
    """Orders frames by DICOM InstanceNumber, NOT AcquisitionTime -- see
    get_instance_number for why. Falls back to filename order if
    InstanceNumber is missing for any frame. Returns (sorted_paths,
    instance_numbers) aligned to the returned order."""
    instance_numbers = [get_instance_number(p) for p in frame_paths]
    if all(n is not None for n in instance_numbers):
        order = sorted(range(len(frame_paths)), key=lambda i: instance_numbers[i])
        return [frame_paths[i] for i in order], [instance_numbers[i] for i in order]
    print("    [warn] InstanceNumber missing for at least one frame -- "
          "falling back to filename order.")
    return frame_paths, [None] * len(frame_paths)


def compute_time_ms(n_frames: int, frame_interval_ms: float | None) -> list[float | str]:
    """Nominal elapsed time (ms), index * frame_interval_ms. AcquisitionTime
    is deliberately not used for timing here -- see sort_frames_temporally."""
    if frame_interval_ms is not None:
        return [i * frame_interval_ms for i in range(n_frames)]
    return ["" for _ in range(n_frames)]


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
# Core: rigid + residual whole-surface displacement (pointcloud / liver-mask)
# ══════════════════════════════════════════════════════════════════════════

def rigid_register(source: np.ndarray, target: np.ndarray,
                    max_correspondence_distance: float) -> np.ndarray:
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
    translation_magnitude = float(np.linalg.norm(transform[:3, 3]))
    R = transform[:3, :3]
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rotation_angle_deg = float(np.degrees(np.arccos(trace)))
    return translation_magnitude, rotation_angle_deg


def whole_surface_stats(magnitudes: np.ndarray) -> dict:
    return {
        "mean": float(magnitudes.mean()),
        "rms": float(np.sqrt((magnitudes ** 2).mean())),
        "p95": float(np.percentile(magnitudes, 95)),
        "max": float(magnitudes.max()),
    }


def frame_vs_reference(reference_pts: np.ndarray, frame_pts: np.ndarray,
                        icp_max_corr_dist: float) -> dict:
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
    bbox_min, bbox_max = points.min(0), points.max(0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    return float(diagonal * 0.05)


# ══════════════════════════════════════════════════════════════════════════
# Consecutive-frame (t vs t+1) amplitude, for frame_to_frame
# ══════════════════════════════════════════════════════════════════════════

def pointcloud_pair_amplitude(pts_a: np.ndarray, pts_b: np.ndarray) -> dict:
    """Bidirectional NN distance between two point sets -> mean/rms/max, mm.
    No rigid alignment removed -- this is the RAW total displacement between
    consecutive frames (bulk shift and deformation both included), unlike
    frame_vs_reference's residual which deliberately excludes bulk shift."""
    tree_b = cKDTree(pts_b)
    d_a_to_b, _ = tree_b.query(pts_a, k=1, workers=-1)
    tree_a = cKDTree(pts_a)
    d_b_to_a, _ = tree_a.query(pts_b, k=1, workers=-1)
    d = np.concatenate([d_a_to_b, d_b_to_a])
    return whole_surface_stats(d)


def mri_pair_amplitude(image_a: sitk.Image, image_b: sitk.Image) -> dict:
    """Deformable (Demons) registration between two MRI frames -> displacement
    field magnitude mean/rms/max, mm. Works for 2D slices or 3D volumes."""
    a = sitk.Cast(image_a, sitk.sitkFloat32)
    b = sitk.Cast(image_b, sitk.sitkFloat32)
    b_resampled = sitk.Resample(b, a)

    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(50)
    demons.SetStandardDeviations(1.5)
    field = demons.Execute(a, b_resampled)

    field_array = sitk.GetArrayFromImage(field)
    magnitude = np.linalg.norm(field_array, axis=-1).ravel()
    return whole_surface_stats(magnitude)


# ══════════════════════════════════════════════════════════════════════════
# CSV 1: point cloud sequence (.ply) vs. reference
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
# Alternative to CSV 2: segmented liver mask sequence (only if you HAVE masks)
# ══════════════════════════════════════════════════════════════════════════

def extract_surface_from_mask_image(image: sitk.Image) -> np.ndarray:
    from skimage.measure import marching_cubes

    array = sitk.GetArrayFromImage(image)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    spacing = image.GetSpacing()[::-1]
    if len(spacing) == 2:
        spacing = (1.0,) + spacing

    if array.max() == 0:
        raise ValueError("Mask is empty")

    verts, _, _, _ = marching_cubes(array.astype(np.float32), level=0.5, spacing=spacing)
    verts_xyz = verts[:, ::-1]
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
    frame_paths, instance_numbers = sort_frames_temporally(frame_paths)
    time_ms_values = compute_time_ms(len(frame_paths), frame_interval_ms)

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
# CSV 2: full MRI DICOM sequence vs. reference, NO mask needed
# ══════════════════════════════════════════════════════════════════════════

def analyze_volume_dvf_sequence(input_dir: Path,
                                 reference_index: int = 0,
                                 frame_interval_ms: float | None = None,
                                 mask_path: Path | None = None) -> list[dict]:
    """
    Registers each frame to the reference with SimpleITK's Demons algorithm
    and aggregates the resulting displacement vector field's magnitude.
    `mask_path=None` (the default -- use this if you don't have liver
    segmentations) aggregates over the WHOLE image; pass a static mask only
    if you happen to have one and want to restrict statistics to the liver.
    """
    frame_paths = find_frame_paths(input_dir)
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {input_dir}")
    frame_paths, instance_numbers = sort_frames_temporally(frame_paths)
    time_ms_values = compute_time_ms(len(frame_paths), frame_interval_ms)

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
    prev_pixel_hash = None
    n_identical_warnings = 0

    for i, fp in enumerate(frame_paths):
        moving_image = sitk.Cast(read_dicom_as_image(fp), sitk.sitkFloat32)

        # diagnostic: hash the raw pixel data so identical-looking downstream
        # registration results can be traced back to genuinely duplicate
        # source data, rather than guessed at.
        moving_array_raw = sitk.GetArrayFromImage(moving_image)
        pixel_hash = hashlib.md5(moving_array_raw.tobytes()).hexdigest()[:12]
        identical_to_previous = (pixel_hash == prev_pixel_hash)
        if identical_to_previous:
            n_identical_warnings += 1
            if n_identical_warnings <= 5:
                print(f"    [warn] frame {i} ({fp.name}) has IDENTICAL pixel data "
                      f"to the previous frame -- likely duplicated source data, "
                      f"not a registration artifact. Any DVF stats for this frame "
                      f"vs. a fixed reference will trivially match whatever the "
                      f"previous frame's comparison gave.")
        prev_pixel_hash = pixel_hash

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
            "pixel_hash": pixel_hash,
            "identical_to_previous_frame": identical_to_previous,
            "dvf_mean_mm": stats["mean"],
            "dvf_rms_mm": stats["rms"],
            "dvf_p95_mm": stats["p95"],
            "dvf_max_mm": stats["max"],
        })

    if n_identical_warnings > 5:
        print(f"    [warn] ... plus {n_identical_warnings - 5} more duplicate-frame "
              f"warnings suppressed. {n_identical_warnings}/{len(frame_paths)} frames "
              f"were byte-identical to their immediate predecessor in total.")

    return rows


# ══════════════════════════════════════════════════════════════════════════
# CSV 2 alternative: registration-free centroid + intensity diff vs. reference
# ══════════════════════════════════════════════════════════════════════════

def normalize01(array: np.ndarray) -> np.ndarray:
    vmin, vmax = float(array.min()), float(array.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(array)
    return (array - vmin) / (vmax - vmin)


def centroid_of_array(array: np.ndarray, threshold_pct: float) -> np.ndarray | None:
    """Threshold-based center of mass, in voxel/pixel coordinates (array axis
    order, e.g. (z,y,x) for a 3D volume or (y,x) for a 2D slice)."""
    nonzero = array[array > 0]
    if nonzero.size == 0:
        return None
    thresh = np.percentile(nonzero, threshold_pct)
    mask = array > thresh
    if not mask.any():
        return None
    return np.array(center_of_mass(mask), dtype=np.float64)


def downsample_array(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array
    slicer = tuple(slice(None, None, factor) for _ in array.shape)
    return array[slicer]


def analyze_volume_simple_sequence(input_dir: Path,
                                    reference_index: int = 0,
                                    frame_interval_ms: float | None = None,
                                    threshold_pct: float = 50.0,
                                    downsample: int = 1) -> list[dict]:
    """
    Registration-free alternative to analyze_volume_dvf_sequence, for when
    Demons registration doesn't work well on your data. No mask needed.
    Every frame is compared to a fixed reference frame two ways:

    1. CENTROID DISPLACEMENT: threshold the frame to segment the anatomy,
       take its center of mass (in mm, via DICOM pixel/voxel spacing), and
       report its displacement FROM THE REFERENCE FRAME's centroid. Bulk
       positional motion, cheap and very robust, but blind to shape change
       that doesn't move the centroid (e.g. symmetric expansion/contraction).
    2. INTENSITY DIFFERENCE: normalize both frame and reference to [0, 1],
       take the mean absolute per-pixel difference. Sensitive to genuine
       local deformation, texture, and any misregistration between the two
       images -- a coarse but very fast and transparent proxy for "how
       different does this frame look from the reference".

    Also includes the same pixel-hash duplicate-frame diagnostic as
    analyze_volume_dvf_sequence, for the same reason: if source frames turn
    out to be byte-identical, both metrics will trivially read as ~0/flat
    for those frames, and that's a data issue, not an analysis bug.
    """
    frame_paths = find_frame_paths(input_dir)
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {input_dir}")
    frame_paths, instance_numbers = sort_frames_temporally(frame_paths)
    time_ms_values = compute_time_ms(len(frame_paths), frame_interval_ms)

    reference_image = read_dicom_as_image(frame_paths[reference_index])
    reference_array = sitk.GetArrayFromImage(reference_image).astype(np.float32)
    spacing = np.array(reference_image.GetSpacing()[::-1])  # matches array axis order

    reference_norm = normalize01(downsample_array(reference_array, downsample))
    reference_centroid_vox = centroid_of_array(reference_array, threshold_pct)
    reference_centroid_mm = (reference_centroid_vox * spacing
                              if reference_centroid_vox is not None else None)
    if reference_centroid_mm is None:
        print("    [warn] Couldn't find a centroid on the reference frame at "
              "this threshold_pct -- centroid_displacement_mm will be empty "
              "for every frame. Try a lower --threshold_pct.")

    rows = []
    prev_pixel_hash = None
    n_identical_warnings = 0

    for i, fp in enumerate(frame_paths):
        image = read_dicom_as_image(fp)
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        pixel_hash = hashlib.md5(array.tobytes()).hexdigest()[:12]
        identical_to_previous = (pixel_hash == prev_pixel_hash)
        if identical_to_previous:
            n_identical_warnings += 1
            if n_identical_warnings <= 5:
                print(f"    [warn] frame {i} ({fp.name}) has IDENTICAL pixel data "
                      f"to the previous frame -- likely duplicated source data.")
        prev_pixel_hash = pixel_hash

        centroid_vox = centroid_of_array(array, threshold_pct)
        if centroid_vox is not None and reference_centroid_mm is not None:
            centroid_mm = centroid_vox * spacing
            centroid_displacement_mm = float(np.linalg.norm(centroid_mm - reference_centroid_mm))
        else:
            centroid_displacement_mm = ""

        array_norm = normalize01(downsample_array(array, downsample))
        if array_norm.shape == reference_norm.shape:
            intensity_diff = float(np.mean(np.abs(array_norm - reference_norm)))
        else:
            intensity_diff = ""
            print(f"    [warn] frame {i} ({fp.name}) has a different shape than "
                  f"the reference frame ({array_norm.shape} vs {reference_norm.shape}) "
                  f"-- intensity_diff_vs_reference left blank for this frame.")

        rows.append({
            "frame_index": i,
            "source_file": str(fp),
            "time_ms": time_ms_values[i],
            "pixel_hash": pixel_hash,
            "identical_to_previous_frame": identical_to_previous,
            "centroid_displacement_mm": centroid_displacement_mm,
            "intensity_diff_vs_reference": intensity_diff,
        })

    if n_identical_warnings > 5:
        print(f"    [warn] ... plus {n_identical_warnings - 5} more duplicate-frame "
              f"warnings suppressed. {n_identical_warnings}/{len(frame_paths)} frames "
              f"were byte-identical to their immediate predecessor in total.")

    return rows


# ══════════════════════════════════════════════════════════════════════════
# CSV 3: cross-modal frame-to-frame (consecutive) amplitude, merged
# ══════════════════════════════════════════════════════════════════════════

def analyze_frame_to_frame(ply_dir: Path, mri_dir: Path,
                            frame_interval_ms: float | None,
                            unit_scale: float,
                            max_frames: int | None = None,
                            comparison_mode: str = "consecutive",
                            reference_index: int = 0) -> list[dict]:
    """
    Computes per-frame amplitude for PLY and MRI INDEPENDENTLY (each on its
    own sequence), then merges them into one CSV by frame index.

    comparison_mode:
      "consecutive" (default) -- each frame vs. the PREVIOUS frame (t vs
        t+1). Measures instantaneous motion / "breathing speed" -- this is
        noisy and does NOT trace out the smooth cyclical breathing-amplitude
        curve, since it's closer to a derivative of position than position
        itself.
      "reference" -- each frame vs. a FIXED reference frame (reference_index,
        default the first frame of each sequence). This is what traces out
        the actual coherent, cyclical amplitude pattern you'd expect to see
        for breathing -- the same computation CSV 1 (pointcloud) and CSV 2
        (volume-dvf/volume-simple) already do individually, just merged into
        one file here for direct two-modality comparison.

    Assumes PLY frame i and MRI frame i correspond to (approximately) the
    same real time. Uses the shorter of the two sequences if their lengths
    differ, further capped to max_frames if given.
    """
    if comparison_mode not in ("consecutive", "reference"):
        raise ValueError(f"comparison_mode must be 'consecutive' or 'reference', "
                          f"got {comparison_mode!r}")

    ply_files = sorted(ply_dir.glob("*.ply"))
    if len(ply_files) < 2:
        raise ValueError(f"Need at least 2 .ply files in {ply_dir}")

    mri_files = find_frame_paths(mri_dir)
    mri_files, mri_instance_numbers = sort_frames_temporally(mri_files)
    if len(mri_files) < 2:
        raise ValueError(f"Need at least 2 MRI frames in {mri_dir}")

    n_ply, n_mri = len(ply_files), len(mri_files)
    n_sync = min(n_ply, n_mri)
    if n_ply != n_mri:
        print(f"[note] PLY has {n_ply} frames, MRI has {n_mri} -- using the "
              f"first {n_sync} of each.")
    if max_frames is not None and n_sync > max_frames:
        print(f"[note] Limiting to the first {max_frames} frames (of {n_sync} "
              f"available) per --max_frames.")
        n_sync = max_frames

    ply_time_ms = [i * frame_interval_ms if frame_interval_ms else "" for i in range(n_sync)]
    mri_time_ms = compute_time_ms(n_sync, frame_interval_ms)

    rows = []

    if comparison_mode == "reference":
        reference_ply_pts = load_ply_points(ply_files[reference_index], unit_scale)
        reference_mri_img = read_dicom_as_image(mri_files[reference_index])
        reference_mri_hash = hashlib.md5(
            sitk.GetArrayFromImage(reference_mri_img).tobytes()).hexdigest()[:12]
        print(f"[reference mode] comparing every frame to reference_index={reference_index} "
              f"for both modalities")

        for i in range(n_sync):
            ply_pts = load_ply_points(ply_files[i], unit_scale)
            mri_img = read_dicom_as_image(mri_files[i])
            mri_array = sitk.GetArrayFromImage(mri_img)
            mri_hash = hashlib.md5(mri_array.tobytes()).hexdigest()[:12]
            identical_to_reference = (mri_hash == reference_mri_hash) and i != reference_index
            if identical_to_reference:
                print(f"    [warn] frame {i} ({mri_files[i].name}) has IDENTICAL pixel "
                      f"data to the reference frame -- source data problem, not a "
                      f"registration artifact.")

            row = {"frame_index": i, "time_ms": ply_time_ms[i], "mri_time_ms": mri_time_ms[i],
                   "ply_source_file": str(ply_files[i]), "mri_source_file": str(mri_files[i]),
                   "mri_instance_number": mri_instance_numbers[i],
                   "mri_pixel_hash": mri_hash, "mri_identical_to_reference": identical_to_reference}

            if i == reference_index:
                row.update({"ply_mean_mm": 0.0, "ply_rms_mm": 0.0, "ply_max_mm": 0.0,
                            "mri_mean_mm": 0.0, "mri_rms_mm": 0.0, "mri_max_mm": 0.0})
            else:
                ply_amp = pointcloud_pair_amplitude(reference_ply_pts, ply_pts)
                mri_amp = mri_pair_amplitude(reference_mri_img, mri_img)
                row.update({
                    "ply_mean_mm": ply_amp["mean"], "ply_rms_mm": ply_amp["rms"], "ply_max_mm": ply_amp["max"],
                    "mri_mean_mm": mri_amp["mean"], "mri_rms_mm": mri_amp["rms"], "mri_max_mm": mri_amp["max"],
                })

            rows.append(row)
            if i % 20 == 0:
                print(f"  processed {i + 1}/{n_sync}")

    else:  # consecutive
        prev_ply_pts, prev_mri_img, prev_mri_hash = None, None, None
        for i in range(n_sync):
            ply_pts = load_ply_points(ply_files[i], unit_scale)
            mri_img = read_dicom_as_image(mri_files[i])
            mri_array = sitk.GetArrayFromImage(mri_img)
            mri_hash = hashlib.md5(mri_array.tobytes()).hexdigest()[:12]
            identical_to_previous = (mri_hash == prev_mri_hash)
            if identical_to_previous:
                print(f"    [warn] frame {i} ({mri_files[i].name}) has IDENTICAL pixel "
                      f"data to the previous frame -- source data problem, not a "
                      f"registration artifact.")

            row = {"frame_index": i, "time_ms": ply_time_ms[i], "mri_time_ms": mri_time_ms[i],
                   "ply_source_file": str(ply_files[i]), "mri_source_file": str(mri_files[i]),
                   "mri_instance_number": mri_instance_numbers[i],
                   "mri_pixel_hash": mri_hash, "mri_identical_to_previous": identical_to_previous}

            if i == 0:
                row.update({"ply_mean_mm": "", "ply_rms_mm": "", "ply_max_mm": "",
                            "mri_mean_mm": "", "mri_rms_mm": "", "mri_max_mm": ""})
            else:
                ply_amp = pointcloud_pair_amplitude(prev_ply_pts, ply_pts)
                mri_amp = mri_pair_amplitude(prev_mri_img, mri_img)
                row.update({
                    "ply_mean_mm": ply_amp["mean"], "ply_rms_mm": ply_amp["rms"], "ply_max_mm": ply_amp["max"],
                    "mri_mean_mm": mri_amp["mean"], "mri_rms_mm": mri_amp["rms"], "mri_max_mm": mri_amp["max"],
                })

            rows.append(row)
            prev_ply_pts, prev_mri_img, prev_mri_hash = ply_pts, mri_img, mri_hash
            if i % 20 == 0:
                print(f"  processed {i + 1}/{n_sync}")

    return rows


# ══════════════════════════════════════════════════════════════════════════
# Analysis 1: correlate two amplitude time series (unchanged from before)
# ══════════════════════════════════════════════════════════════════════════

def load_csv_series(path: Path, value_column: str, frame_interval_ms: float = None) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(value_column) in (None, ""):
                continue  # skip frame 0, which has no vs-previous/vs-reference value
            values.append(float(row[value_column]))
            
            if frame_interval_ms is None:
                times.append(float(row["time_ms"]))
            else:
                times.append(len(times) * frame_interval_ms)
    return np.array(times), np.array(values)


def compare_series(times_a, values_a, times_b, values_b,
                    max_lag_ms: float = 2000, resample_ms: float = 50) -> dict:
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
# Analysis 2 (NEW): per-breath peak-to-peak amplitude via peak detection
# ══════════════════════════════════════════════════════════════════════════

def detect_breath_cycles(times_ms: np.ndarray, values: np.ndarray,
                          min_prominence: float | None,
                          min_distance_ms: float) -> dict:
    """
    Finds local maxima (peaks) in `values` -- e.g. the residual_rms_mm or
    dvf_rms_mm column of CSV 1/2 -- treating each peak-to-trough-to-peak span
    as one breathing cycle, and reports the peak-to-peak amplitude of each.

    `min_prominence`: how much a peak must stand out from its surroundings to
    count (in the same units as `values`, e.g. mm). None = auto (10% of the
    signal's own range) -- tune this up if noise is creating spurious peaks,
    or down if real, shallow breaths are being missed.
    `min_distance_ms`: minimum time between two accepted peaks -- set this
    roughly to your patient's fastest plausible breath period, to prevent
    noise from being counted as extra breaths.
    """
    if min_prominence is None:
        min_prominence = 0.1 * (values.max() - values.min())

    dt = np.median(np.diff(times_ms))
    min_distance_samples = max(1, int(round(min_distance_ms / dt)))

    peak_idx, _ = find_peaks(values, prominence=min_prominence, distance=min_distance_samples)
    trough_idx, _ = find_peaks(-values, prominence=min_prominence, distance=min_distance_samples)

    if len(peak_idx) < 2:
        return {"n_cycles": 0, "mean_amplitude_mm": None, "std_amplitude_mm": None,
                "peak_times_ms": [], "cycle_amplitudes_mm": []}

    cycle_amplitudes = []
    for p0, p1 in zip(peak_idx[:-1], peak_idx[1:]):
        troughs_between = trough_idx[(trough_idx > p0) & (trough_idx < p1)]
        if len(troughs_between) > 0:
            trough_val = values[troughs_between].min()
        else:
            trough_val = min(values[p0], values[p1])
        peak_val = max(values[p0], values[p1])
        cycle_amplitudes.append(float(peak_val - trough_val))

    return {
        "n_cycles": len(cycle_amplitudes),
        "mean_amplitude_mm": float(np.mean(cycle_amplitudes)),
        "std_amplitude_mm": float(np.std(cycle_amplitudes)),
        "peak_times_ms": [float(times_ms[i]) for i in peak_idx],
        "cycle_amplitudes_mm": cycle_amplitudes,
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
        if len(vals) == 0:
            continue
        print(f"  {k:24s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"p2p={vals.max() - vals.min():.4f}  max={vals.max():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pc = sub.add_parser("pointcloud", help="CSV 1: PLY vs. reference PLY frame")
    p_pc.add_argument("--input", required=True)
    p_pc.add_argument("--output_csv", required=True)
    p_pc.add_argument("--reference_index", type=int, default=0)
    p_pc.add_argument("--frame_interval_ms", type=float, default=None)
    p_pc.add_argument("--icp_max_corr_dist", type=float, default=None)
    p_pc.add_argument("--unit_scale", type=float, default=1.0)

    p_mask = sub.add_parser("liver-mask",
                             help="Alternative to volume-dvf IF you have per-frame liver segmentations")
    p_mask.add_argument("--input", required=True)
    p_mask.add_argument("--output_csv", required=True)
    p_mask.add_argument("--reference_index", type=int, default=0)
    p_mask.add_argument("--frame_interval_ms", type=float, default=None)
    p_mask.add_argument("--icp_max_corr_dist", type=float, default=None)
    p_mask.add_argument("--max_surface_points", type=int, default=20000)

    p_vol = sub.add_parser("volume-dvf",
                            help="CSV 2: MRI vs. reference MRI frame -- no mask needed")
    p_vol.add_argument("--input", required=True)
    p_vol.add_argument("--output_csv", required=True)
    p_vol.add_argument("--reference_index", type=int, default=0)
    p_vol.add_argument("--frame_interval_ms", type=float, default=None)
    p_vol.add_argument("--mask", default=None,
                        help="Optional; omit entirely if you don't have liver segmentations")

    p_simple = sub.add_parser("volume-simple",
                               help="CSV 2 alternative: registration-free centroid + intensity "
                                    "difference vs. reference (use this if volume-dvf doesn't "
                                    "work well on your data)")
    p_simple.add_argument("--input", required=True)
    p_simple.add_argument("--output_csv", required=True)
    p_simple.add_argument("--reference_index", type=int, default=0)
    p_simple.add_argument("--frame_interval_ms", type=float, default=None)
    p_simple.add_argument("--threshold_pct", type=float, default=50.0,
                           help="Percentile of nonzero intensity used as the centroid mask "
                                "threshold (default 50). Lower it if the centroid isn't found.")
    p_simple.add_argument("--downsample", type=int, default=1,
                           help="Take every Nth pixel/voxel per axis before the intensity-diff "
                                "calc, for speed (default 1 = no downsampling)")

    p_f2f = sub.add_parser("frame_to_frame",
                            help="CSV 3: consecutive-frame amplitude, PLY and MRI merged")
    p_f2f.add_argument("--ply_dir", required=True)
    p_f2f.add_argument("--mri_dir", required=True)
    p_f2f.add_argument("--output_csv", required=True)
    p_f2f.add_argument("--frame_interval_ms", type=float, default=None,
                        help="Fallback for PLY timestamps (PLY has no AcquisitionTime)")
    p_f2f.add_argument("--unit_scale", type=float, default=1.0)
    p_f2f.add_argument("--max_frames", type=int, default=None,
                        help="Limit to the first N frames of each modality (e.g. 200), "
                             "instead of the full synchronized sequence")
    p_f2f.add_argument("--comparison_mode", choices=["consecutive", "reference"],
                        default="consecutive",
                        help="'consecutive': each frame vs. the previous frame (noisy, "
                             "instantaneous motion). 'reference': each frame vs. a fixed "
                             "reference frame -- use this for a coherent, cyclical "
                             "breathing-amplitude curve.")
    p_f2f.add_argument("--reference_index", type=int, default=0,
                        help="Only used with --comparison_mode reference")

    p_cmp = sub.add_parser("compare", help="Correlate two amplitude time series (CSVs above)")
    p_cmp.add_argument("--csv_a", required=True)
    p_cmp.add_argument("--column_a", required=True)
    p_cmp.add_argument("--csv_b", required=True)
    p_cmp.add_argument("--column_b", required=True)
    p_cmp.add_argument("--max_lag_ms", type=float, default=2000)
    p_cmp.add_argument("--resample_ms", type=float, default=50)
    p_cmp.add_argument("--frame_interval_ms", type=float, default=None)

    p_breath = sub.add_parser("breath_cycles",
                               help="Per-breath peak-to-peak amplitude from one CSV's signal")
    p_breath.add_argument("--csv", required=True)
    p_breath.add_argument("--column", required=True,
                           help="e.g. residual_rms_mm (CSV 1) or dvf_rms_mm (CSV 2)")
    p_breath.add_argument("--min_prominence", type=float, default=None,
                           help="mm; None = auto (10%% of signal range)")
    p_breath.add_argument("--min_distance_ms", type=float, default=1500,
                           help="Minimum time between accepted breath peaks")

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

    elif args.command == "volume-simple":
        rows = analyze_volume_simple_sequence(
            Path(args.input), args.reference_index, args.frame_interval_ms,
            args.threshold_pct, args.downsample)
        write_csv(rows, Path(args.output_csv))
        print_summary(rows, ["centroid_displacement_mm", "intensity_diff_vs_reference"])

    elif args.command == "frame_to_frame":
        rows = analyze_frame_to_frame(
            Path(args.ply_dir), Path(args.mri_dir), args.frame_interval_ms,
            args.unit_scale, args.max_frames, args.comparison_mode, args.reference_index)
        write_csv(rows, Path(args.output_csv))
        print_summary(rows, ["ply_rms_mm", "mri_rms_mm"])

    elif args.command == "compare":
        times_a, values_a = load_csv_series(Path(args.csv_a), args.column_a, args.frame_interval_ms)
        times_b, values_b = load_csv_series(Path(args.csv_b), args.column_b, args.frame_interval_ms)
        result = compare_series(times_a, values_a, times_b, values_b,
                                 args.max_lag_ms, args.resample_ms)
        print(f"\n{'=' * 60}\nComparison: {args.csv_a} [{args.column_a}]  vs.  "
              f"{args.csv_b} [{args.column_b}]\n{'=' * 60}")
        for k, v in result.items():
            print(f"  {k:24s} {v}")

    elif args.command == "breath_cycles":
        times, values = load_csv_series(Path(args.csv), args.column)
        result = detect_breath_cycles(times, values, args.min_prominence, args.min_distance_ms)
        print(f"\n{'=' * 60}\nBreath cycles: {args.csv} [{args.column}]\n{'=' * 60}")
        print(f"  n_cycles          : {result['n_cycles']}")
        print(f"  mean_amplitude_mm : {result['mean_amplitude_mm']}")
        print(f"  std_amplitude_mm  : {result['std_amplitude_mm']}")
        if result["n_cycles"] > 0:
            print(f"  per-cycle amplitudes (mm): "
                  f"{[round(a, 3) for a in result['cycle_amplitudes_mm']]}")


if __name__ == "__main__":
    main()