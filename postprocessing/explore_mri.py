"""
Usage:
python explore_mri_series.py --input /path/to/mri_data_folder
"""

import argparse
from collections import defaultdict
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
    files = []
    for p in root.rglob("*"):
        if looks_like_dicom(p):
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                         help="Root folder to search recursively for DICOM files "
                              "(can contain both series mixed together or in subfolders)")
    args = parser.parse_args()

    root = Path(args.input)
    files = find_all_dicom_files(root)
    print(f"Found {len(files)} DICOM files under {root}\n")

    series = defaultdict(lambda: {"files": [], "directories": set()})

    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception as e:
            print(f"  [warn] couldn't read {fp}: {e}")
            continue

        uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
        entry = series[uid]
        entry["files"].append(fp)
        entry["directories"].add(str(fp.parent))
        entry["series_description"] = str(getattr(ds, "SeriesDescription", ""))
        entry["protocol_name"] = str(getattr(ds, "ProtocolName", ""))
        entry["series_number"] = getattr(ds, "SeriesNumber", None)
        entry["modality"] = str(getattr(ds, "Modality", ""))
        entry["rows"] = getattr(ds, "Rows", None)
        entry["columns"] = getattr(ds, "Columns", None)
        entry["image_type"] = list(getattr(ds, "ImageType", []))

    print(f"{len(series)} distinct series found:\n")
    print(f"{'=' * 100}")
    for uid, info in series.items():
        print(f"SeriesInstanceUID : {uid}")
        print(f"  SeriesDescription : {info['series_description']!r}")
        print(f"  ProtocolName      : {info['protocol_name']!r}")
        print(f"  Modality          : {info['modality']}")
        print(f"  SeriesNumber      : {info['series_number']}")
        print(f"  ImageType         : {info['image_type']}")
        print(f"  Dimensions        : {info['rows']} x {info['columns']}")
        print(f"  Number of files   : {len(info['files'])}")
        print(f"  Directory/ies     : {sorted(info['directories'])}")
        print(f"{'-' * 100}")

    print(
        "\nLook at Dimensions and SeriesDescription/ImageType above to tell the "
        "navigator series apart from the volume-slice series -- the navigator is "
        "typically a thin strip or has smaller/different Dimensions and a "
        "description mentioning 'nav', while the volume-slice series matches the "
        "square anatomical image (e.g. 176x176) you'd actually want to view.\n"
        "Once identified, point the viewer notebook's MRI_DIR at that series' "
        "directory (or, if both series are mixed in one flat folder, use the "
        "printed SeriesInstanceUID to filter -- ask if you'd like a copy/filter "
        "script for that case)."
    )


if __name__ == "__main__":
    main()