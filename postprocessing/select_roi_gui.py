"""
Select an ROI (region of interest) from a standalone ROS2 .db3 bag file
(no metadata.yaml required -- topic/type info is read directly from the
.db3 SQLite database), crop every frame on an image topic to that ROI,
and save as a video.

Usage:
    # List available topics/types in the file first if you don't know the topic name:
    python select_roi_db3.py --bag /path/to/file.db3 --list-topics

    # Interactive ROI selection (drag box on first frame, ENTER to confirm):
    python select_roi_db3.py --bag /path/to/file.db3 --topic /camera/color/image_raw --out cropped.mp4

    # Or skip interactive selection with known coordinates:
    python select_roi_db3.py --bag /path/to/file.db3 --topic /camera/color/image_raw \
        --out cropped.mp4 --x 100 --y 50 --w 400 --h 300

Requirements:
    pip install opencv-python rosbags
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from rosbags.serde import deserialize_cdr
except ImportError:
    print("This script requires the 'rosbags' package: pip install rosbags")
    sys.exit(1)


def list_topics(db3_path: str):
    """Reads the topics table directly from the .db3 SQLite file."""
    conn = sqlite3.connect(db3_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, type, serialization_format FROM topics")
    topics = cur.fetchall()
    conn.close()
    return topics  # list of (id, name, type, serialization_format)


def iter_bag_images(db3_path: str, topic: str):
    """Yields (timestamp_ns, bgr_image) for every message on `topic`,
    reading directly out of the .db3 SQLite file."""
    conn = sqlite3.connect(db3_path)
    cur = conn.cursor()

    cur.execute("SELECT id, type, serialization_format FROM topics WHERE name = ?", (topic,))
    row = cur.fetchone()
    if row is None:
        available = list_topics(db3_path)
        conn.close()
        raise ValueError(
            f"Topic '{topic}' not found in {db3_path}. Available topics:\n  " +
            "\n  ".join(f"{t[1]}  ({t[2]})" for t in available)
        )
    topic_id, msgtype, serialization_format = row
    if serialization_format != "cdr":
        print(f"[warn] serialization_format='{serialization_format}', expected 'cdr' -- "
              f"deserialization may fail.")

    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC",
        (topic_id,),
    )

    for timestamp, rawdata in cur:
        msg = deserialize_cdr(bytes(rawdata), msgtype)

        if msgtype.endswith("Image") and not msgtype.endswith("CompressedImage"):
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, -1)
            if img.shape[2] == 1:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif msg.encoding.lower() == "rgb8":
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img  # assume bgr8 or compatible layout
            yield timestamp, bgr

        elif msgtype.endswith("CompressedImage"):
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            yield timestamp, bgr

        else:
            raise ValueError(f"Topic '{topic}' has message type '{msgtype}', "
                              f"which isn't an Image/CompressedImage type.")

    conn.close()


def select_roi_interactive(frame: np.ndarray):
    print("Drag a rectangle over the ROI, then press ENTER/SPACE to confirm "
          "(or 'c' to cancel).")
    roi = cv2.selectROI("Select ROI (ENTER to confirm, c to cancel)", frame,
                         fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected (width or height was 0). Aborting.")
    return int(x), int(y), int(w), int(h)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", required=True, help="Path to the standalone .db3 file")
    parser.add_argument("--topic", default=None, help="Image topic to read, e.g. /camera/color/image_raw")
    parser.add_argument("--out", default=None, help="Output video path, e.g. cropped.mp4")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    parser.add_argument("--list-topics", action="store_true", help="Print available topics and exit")

    parser.add_argument("--x", type=int, default=None, help="ROI top-left x")
    parser.add_argument("--y", type=int, default=None, help="ROI top-left y")
    parser.add_argument("--w", type=int, default=None, help="ROI width")
    parser.add_argument("--h", type=int, default=None, help="ROI height")

    args = parser.parse_args()

    if args.list_topics:
        topics = list_topics(args.bag)
        print(f"Topics in {args.bag}:")
        for tid, name, msgtype, fmt in topics:
            print(f"  {name}   [{msgtype}]   ({fmt})")
        return

    if not args.topic or not args.out:
        parser.error("--topic and --out are required unless using --list-topics")

    frames_iter = iter_bag_images(args.bag, args.topic)

    try:
        first_ts, first_frame = next(frames_iter)
    except StopIteration:
        print(f"No messages found on topic '{args.topic}'.")
        sys.exit(1)

    manual_roi_given = all(v is not None for v in (args.x, args.y, args.w, args.h))
    if manual_roi_given:
        x, y, w, h = args.x, args.y, args.w, args.h
        print(f"Using provided ROI: x={x}, y={y}, w={w}, h={h}")
    else:
        x, y, w, h = select_roi_interactive(first_frame)
        print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")

    fh, fw = first_frame.shape[:2]
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        print(f"[warn] ROI ({x},{y},{w},{h}) exceeds frame bounds ({fw}x{fh}); clamping.")
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = min(w, fw - x)
        h = min(h, fh - y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

    def crop_and_write(frame):
        cropped = frame[y:y + h, x:x + w]
        writer.write(cropped)

    n_frames = 0
    crop_and_write(first_frame)
    n_frames += 1

    for ts, frame in frames_iter:
        crop_and_write(frame)
        n_frames += 1

    writer.release()
    print(f"Saved {n_frames} cropped frames to {out_path} (ROI: x={x}, y={y}, w={w}, h={h})")


if __name__ == "__main__":
    main()