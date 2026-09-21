"""
build_phase_volumes.py

Your MRI series is a multi-slice, phase-binned acquisition: for each of many
slice locations (stepping through the anatomy), the scanner captures 18
respiratory phase bins (TemporalPositionIdentifier 1-18) as navigator/volume
pairs (TriggerTime ~140/~365). This reconstructs the data the way it's meant
to be used: one real 3D volume PER PHASE, built by stacking every slice's
image for that phase, ordered by SliceLocation -- instead of the flat
single-slice-cine treatment used earlier, which was comparing different
anatomical cross-sections to each other rather than true breathing motion.

Usage
-----
python build_phase_volumes.py \\
    --input /volatile/Datasets/Varian_Motion/chum/9_11/mri/noantenna_Series0801 \\
    --output /volatile/Datasets/Varian_Motion/chum/9_11/mri/phase_volumes \\
    --trigger_time 365
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def looks_like_dicom(path: Path) -> bool:
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


def read_record(fp: Path) -> dict | None:
    try:
        ds = pydicom.dcmread(str(fp), force=True)
    except Exception as e:
        print(f"  [warn] couldn't read {fp}: {e}")
        return None
    return {
        "file": fp,
        "trigger_time": getattr(ds, "TriggerTime", None),
        "temporal_position": getattr(ds, "TemporalPositionIdentifier", None),
        "slice_location": getattr(ds, "SliceLocation", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "pixel_spacing": getattr(ds, "PixelSpacing", None),
        "pixel_array": ds.pixel_array,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of raw DICOM files (all slices, all phases)")
    parser.add_argument("--output", required=True, help="Folder to write phase_XX.nii.gz volumes into")
    parser.add_argument("--trigger_time", type=float, required=True,
                         help="TriggerTime value identifying the VOLUME images (not the "
                              "navigator) -- run explore_mri_series.py first if unsure; "
                              "365 was the volume trigger time in your data")
    parser.add_argument("--trigger_tolerance", type=float, default=5.0)
    parser.add_argument("--slice_thickness_override", type=float, default=None,
                         help="Force a specific slice spacing (mm) instead of inferring it "
                              "from consecutive SliceLocation values")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir() if looks_like_dicom(p))
    print(f"Found {len(files)} DICOM files in {input_dir}")

    records = []
    for fp in files:
        record = read_record(fp)
        if record is not None:
            records.append(record)

    volume_records = [
        r for r in records
        if r["trigger_time"] is not None
        and abs(float(r["trigger_time"]) - args.trigger_time) <= args.trigger_tolerance
    ]
    print(f"Kept {len(volume_records)}/{len(records)} files matching "
          f"trigger_time={args.trigger_time} (tolerance={args.trigger_tolerance})")

    if not volume_records:
        print("[error] Nothing matched --trigger_time. Distinct trigger_time values "
              f"seen: {sorted(set(r['trigger_time'] for r in records))}")
        return

    by_phase = defaultdict(list)
    for r in volume_records:
        by_phase[r["temporal_position"]].append(r)

    phases = sorted(by_phase.keys())
    print(f"Found {len(phases)} phase bins: {phases}")

    all_slice_locations = sorted(set(
        r["slice_location"] for r in volume_records if r["slice_location"] is not None
    ))
    n_slices_expected = len(all_slice_locations)
    print(f"{n_slices_expected} distinct slice locations across the whole series")

    if len(all_slice_locations) >= 2:
        diffs = np.diff(all_slice_locations)
        slice_spacing = args.slice_thickness_override or float(abs(np.median(diffs)))
        if not np.allclose(diffs, diffs[0], atol=0.05):
            print(f"  [warn] slice spacing isn't perfectly uniform (diffs: "
                  f"{np.round(diffs, 3)}) -- using median ({slice_spacing:.3f} mm). "
                  f"Volumes will still be built correctly (slices stay in the right "
                  f"order), but assume uniform mm spacing for the sitk.Image spacing tag.")
    else:
        slice_spacing = args.slice_thickness_override or 1.0

    for phase in phases:
        phase_records = sorted(by_phase[phase], key=lambda r: r["slice_location"])

        if len(phase_records) != n_slices_expected:
            print(f"  [warn] phase {phase}: {len(phase_records)}/{n_slices_expected} "
                  f"slices present (some slice locations missing this phase -- "
                  f"navigator likely rejected them). Volume will be built from "
                  f"whatever slices ARE present, in slice-location order; positions "
                  f"in the stack won't align exactly with other phases' stacks if "
                  f"different slices are missing per phase.")

        arrays = [r["pixel_array"].astype(np.float32) for r in phase_records]
        stack = np.stack(arrays, axis=0)  # [n_slices, rows, cols]

        image = sitk.GetImageFromArray(stack)

        pixel_spacing = phase_records[0]["pixel_spacing"]
        if pixel_spacing is not None:
            row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
        else:
            row_spacing = col_spacing = 1.0
        image.SetSpacing((col_spacing, row_spacing, slice_spacing))

        output_path = output_dir / f"phase_{int(phase):02d}.nii.gz"
        sitk.WriteImage(image, str(output_path))
        print(f"  phase {phase}: {stack.shape} -> {output_path}")

    print(f"\nDone. {len(phases)} phase volumes written to {output_dir}")
    print("Use these directly with quantify_breathing_motion.py's volume-dvf or "
          "volume-simple (find_frame_paths now also recognizes .nii/.nii.gz files, "
          "not just DICOM) -- e.g.:\n"
          f"  python quantify_breathing_motion.py volume-simple --input {output_dir} "
          f"--output_csv ./output/mri_phase_vs_reference.csv")


if __name__ == "__main__":
    main()