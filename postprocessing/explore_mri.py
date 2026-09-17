"""
explore_mri_series.py

Run this against your MRI data folder to see what series are actually in
there. Unlike a naive per-series summary (which only keeps ONE file's tag
values per series and can silently hide real differences), this reports the
FULL DISTRIBUTION of Rows x Columns and ImageType seen across all files in
each series, plus a per-file CSV dump -- because a navigator and its paired
volume-slice image can be interleaved WITHIN a single SeriesInstanceUID
(same series, alternating file attributes), rather than living in two
separate series. If a series shows two distinct (Rows, Columns) or
ImageType groups each with roughly half the file count, that's almost
certainly your navigator/volume split.

Usage
-----
python explore_mri_series.py --input /path/to/mri_data_folder --output_dir ./output
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


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


def find_all_dicom_files(root: Path) -> list[Path]:
    if root.is_file() and looks_like_dicom(root):
        return [root]
    return [p for p in root.rglob("*") if looks_like_dicom(p)]


def read_file_record(fp: Path) -> dict | None:
    try:
        ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
    except Exception as e:
        print(f"  [warn] couldn't read {fp}: {e}")
        return None

    return {
        "file": fp,
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "UNKNOWN")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "protocol_name": str(getattr(ds, "ProtocolName", "")),
        "series_number": getattr(ds, "SeriesNumber", None),
        "modality": str(getattr(ds, "Modality", "")),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "image_type": tuple(getattr(ds, "ImageType", [])),
        "instance_number": getattr(ds, "InstanceNumber", None),
        "acquisition_number": getattr(ds, "AcquisitionNumber", None),
        "acquisition_time": str(getattr(ds, "AcquisitionTime", "")),
        "sequence_name": str(getattr(ds, "SequenceName", "")),
        "slice_location": getattr(ds, "SliceLocation", None),
        "echo_time": getattr(ds, "EchoTime", None),
        "trigger_time": getattr(ds, "TriggerTime", None),
        "temporal_position_identifier": getattr(ds, "TemporalPositionIdentifier", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                         help="Root folder to search recursively for DICOM files")
    parser.add_argument("--output_dir", default="./output",
                         help="Where to write per-series file-detail CSVs")
    parser.add_argument("--min_files_to_report", type=int, default=5,
                         help="Skip series with fewer files than this (filters out "
                              "1-file Presentation State / ExamCard objects, which "
                              "aren't image data)")
    args = parser.parse_args()

    root = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_all_dicom_files(root)
    print(f"Found {len(files)} DICOM files under {root}\n")

    records = []
    for fp in files:
        record = read_file_record(fp)
        if record is not None:
            records.append(record)

    by_series = defaultdict(list)
    for r in records:
        by_series[r["series_uid"]].append(r)

    real_series = {uid: recs for uid, recs in by_series.items()
                   if len(recs) >= args.min_files_to_report}
    skipped = {uid: recs for uid, recs in by_series.items()
               if len(recs) < args.min_files_to_report}

    if skipped:
        print(f"Skipping {len(skipped)} series with fewer than "
              f"{args.min_files_to_report} files (likely Presentation State / "
              f"ExamCard objects, not image data):")
        for uid, recs in skipped.items():
            r = recs[0]
            print(f"  {r['modality']:4s}  {r['series_description']!r:30s}  "
                  f"{len(recs)} file(s)  UID={uid}")
        print()

    print(f"{len(real_series)} image series found:\n")
    print(f"{'=' * 100}")

    for uid, recs in real_series.items():
        r0 = recs[0]
        shape_counts = Counter((r["rows"], r["columns"]) for r in recs)
        image_type_counts = Counter(r["image_type"] for r in recs)
        seq_name_counts = Counter(r["sequence_name"] for r in recs)

        print(f"SeriesInstanceUID : {uid}")
        print(f"  SeriesDescription : {r0['series_description']!r}")
        print(f"  ProtocolName      : {r0['protocol_name']!r}")
        print(f"  Modality          : {r0['modality']}")
        print(f"  SeriesNumber      : {r0['series_number']}")
        print(f"  Total files       : {len(recs)}")
        print(f"  Directory/ies     : {sorted(set(str(r['file'].parent) for r in recs))}")

        print(f"  Rows x Columns distribution:")
        for (rows, cols), count in shape_counts.most_common():
            print(f"    {rows} x {cols} : {count} files")

        print(f"  ImageType distribution:")
        for image_type, count in image_type_counts.most_common():
            print(f"    {image_type} : {count} files")

        if len(seq_name_counts) > 1:
            print(f"  SequenceName distribution (varies within series -- worth checking too):")
            for seq_name, count in seq_name_counts.most_common():
                print(f"    {seq_name!r} : {count} files")

        if len(shape_counts) > 1 or len(image_type_counts) > 1:
            print(f"  >>> This series has MULTIPLE distinct file types mixed together. "
                  f"If two groups each have roughly half the files, that's very likely "
                  f"your navigator/volume-slice split (see the CSV for how to separate "
                  f"them, and use split_dicom_by_attribute.py to do it automatically).")

        # per-file CSV, sorted by InstanceNumber (falls back to filename)
        csv_path = output_dir / f"series_{r0['series_number']}_file_details.csv"
        recs_sorted = sorted(
            recs, key=lambda r: (r["instance_number"] is None, r["instance_number"] or 0, str(r["file"]))
        )
        fieldnames = ["instance_number", "file", "rows", "columns", "image_type",
                      "acquisition_number", "acquisition_time", "sequence_name",
                      "slice_location", "echo_time", "trigger_time",
                      "temporal_position_identifier"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in recs_sorted:
                row = {k: r[k] for k in fieldnames if k != "file"}
                row["file"] = str(r["file"])
                writer.writerow(row)
        print(f"  Per-file details written to: {csv_path}")
        print(f"{'-' * 100}")

    print(
        "\nOpen the per-file CSV(s) above (sorted by InstanceNumber) and look at the "
        "rows/columns and image_type columns in sequence -- an alternating pattern "
        "every other row (or every Nth row) is the navigator/volume-slice split. "
        "Once you know which attribute distinguishes them and what value belongs to "
        "the volume slice, use split_dicom_by_attribute.py to copy just those files "
        "into a clean folder for the viewer notebook."
    )


if __name__ == "__main__":
    main()