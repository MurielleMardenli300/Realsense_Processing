"""
split_dicom_by_attribute.py

Once explore_mri_series.py has shown you which DICOM tag (and which value of
it) separates the navigator frames from the volume-slice frames within a
series, use this to copy just the frames you want into a clean folder --
e.g. so MRI_DIR in the viewer notebook points at volume slices only.

Usage
-----
# split by TriggerTime (your case: navigator vs. volume slice differ here)
python split_dicom_by_attribute.py \\
    --input /path/to/mixed_series_folder \\
    --output /path/to/volume_slices_only \\
    --attribute trigger_time \\
    --keep_value 365

# multiple values to keep are comma-separated (in case more than one
# TriggerTime belongs to the volume slice, not just the navigator)
python split_dicom_by_attribute.py \\
    --input /path/to/mixed_series_folder \\
    --output /path/to/volume_slices_only \\
    --attribute trigger_time \\
    --keep_value 365,367.5

# split by ImageType or Rows x Columns instead
python split_dicom_by_attribute.py \\
    --input /path/to/mixed_series_folder \\
    --output /path/to/volume_slices_only \\
    --attribute image_type \\
    --keep_value "ORIGINAL,PRIMARY,M_FFE,M,FFE"
"""

import argparse
import shutil
from pathlib import Path

import pydicom

NUMERIC_ATTRIBUTES = {"trigger_time"}


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


def get_attribute_value(ds, attribute: str) -> str:
    if attribute == "image_type":
        return ",".join(getattr(ds, "ImageType", []))
    if attribute == "shape":
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        return f"{rows}x{cols}"
    if attribute == "sequence_name":
        return str(getattr(ds, "SequenceName", ""))
    if attribute == "trigger_time":
        return str(getattr(ds, "TriggerTime", ""))
    raise ValueError(f"Unknown --attribute {attribute!r}")


def matches(value: str, keep_values: list[str], attribute: str, tolerance: float) -> bool:
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return False
        return any(abs(value_f - float(kv)) <= tolerance for kv in keep_values)
    return value in keep_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of mixed DICOM files")
    parser.add_argument("--output", required=True, help="Folder to copy matching files into")
    parser.add_argument("--attribute", required=True,
                         choices=["image_type", "shape", "sequence_name", "trigger_time"],
                         help="Which tag to filter on -- see explore_mri_series.py's "
                              "output for what varies in your data")
    parser.add_argument("--keep_value", required=True,
                         help="Value(s) to KEEP, comma-separated if more than one, exactly "
                              "as printed by explore_mri_series.py (image_type: comma-"
                              "separated tuple values with no spaces; shape: e.g. '176x176'; "
                              "trigger_time: numeric, e.g. '365' or '365,367.5')")
    parser.add_argument("--tolerance", type=float, default=0.5,
                         help="For numeric attributes (trigger_time): how close a file's "
                              "value must be to one of --keep_value to count as a match "
                              "(handles minor float formatting/jitter). Default 0.5 ms.")
    parser.add_argument("--symlink", action="store_true",
                         help="Symlink instead of copy (faster, saves disk space)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_values = [v.strip() for v in args.keep_value.split(",")]

    files = [p for p in input_dir.iterdir() if looks_like_dicom(p)]
    print(f"Found {len(files)} DICOM files in {input_dir}")

    kept = 0
    seen_values_skipped = set()
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception as e:
            print(f"  [warn] couldn't read {fp}: {e}")
            continue

        value = get_attribute_value(ds, args.attribute)
        if not matches(value, keep_values, args.attribute, args.tolerance):
            seen_values_skipped.add(value)
            continue

        destination = output_dir / fp.name
        if args.symlink:
            if not destination.exists():
                destination.symlink_to(fp.resolve())
        else:
            shutil.copy2(fp, destination)
        kept += 1

    print(f"Kept {kept}/{len(files)} files matching "
          f"{args.attribute} in {keep_values} (tolerance={args.tolerance}) -> {output_dir}")

    if kept == 0:
        print(f"\n[warn] Nothing matched. Values actually seen but skipped: "
              f"{sorted(seen_values_skipped)[:10]}"
              f"{'...' if len(seen_values_skipped) > 10 else ''}. "
              f"Double-check --keep_value against these.")
    elif kept < len(files) and len(files) - kept > 0:
        print(f"Skipped {len(files) - kept} files with other {args.attribute} values: "
              f"{sorted(seen_values_skipped)[:10]}"
              f"{'...' if len(seen_values_skipped) > 10 else ''}")


if __name__ == "__main__":
    main()