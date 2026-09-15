"""
Extract colorized point cloud(s) as .ply from a RealSense recording.

IMPORTANT: this script wants the *original* native RealSense recording
(the file you get from RealSense Viewer's "Record" button, or `rs-record`),
which is usually a `.bag` file. It is NOT meant to read the converted
`.db3` file directly -- pyrealsense2 handles alignment, intrinsics, and
depth scale for you automatically, which is much more reliable than
reimplementing that by hand from the .db3 metadata strings.

If you only have the .db3 (the original .bag is gone), see the
`inspect_db3_metadata()` helper at the bottom -- run that first and send
me the printed output so I can write a manual-extraction path using the
real intrinsics/depth-scale values instead of guessing them.

Requirements:
    pip install pyrealsense2 numpy
"""

import argparse
import sys

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("This script requires pyrealsense2: pip install pyrealsense2")
    sys.exit(1)


def extract_ply_from_bag(bag_path: str, ply_path: str, frame_index: int = 0):
    """
    Play back a native RealSense .bag file, align depth to color,
    build a point cloud, and export it (with per-point color) as .ply.

    frame_index: which frame in the recording to export (0 = first).
    """
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)  # read as fast as possible, don't wait on wall-clock

    align = rs.align(rs.stream.color)  # aligns depth frame into the color frame's viewpoint
    pc = rs.pointcloud()

    try:
        frames = None
        for _ in range(frame_index + 1):
            frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Missing depth or color frame at that index.")

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        points.export_to_ply(ply_path, color_frame)
        print(f"Saved colorized point cloud to {ply_path}")
    finally:
        pipeline.stop()


def extract_ply_sequence(bag_path: str, out_dir: str, every_n: int = 1, max_frames: int | None = None):
    """
    Export a sequence of PLYs, one per (every_n-th) frame, into out_dir.
    Useful if you want the whole recording rather than a single frame.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    count = 0
    saved = 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break  # end of bag

            if count % every_n == 0:
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if depth_frame and color_frame:
                    pc.map_to(color_frame)
                    points = pc.calculate(depth_frame)
                    out_path = f"{out_dir}/frame_{count:06d}.ply"
                    points.export_to_ply(out_path, color_frame)
                    saved += 1

            count += 1
            if max_frames is not None and saved >= max_frames:
                break
    finally:
        pipeline.stop()

    print(f"Saved {saved} point clouds to {out_dir}/")


def inspect_db3_metadata(db3_path: str, topic_substring: str = "info"):
    """
    Fallback helper: if the original .bag is gone and you only have the
    .db3, dump the raw text of the metadata topics so we can see what
    intrinsics / depth-scale info actually survived the conversion.
    Run this and share the output before attempting manual extraction.
    """
    import sqlite3
    conn = sqlite3.connect(db3_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, type FROM topics WHERE name LIKE ?", (f"%{topic_substring}%",))
    topics = cur.fetchall()

    import struct

    for tid, name, msgtype in topics:
        cur.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (tid,))
        row = cur.fetchone()
        if row:
            raw = bytes(row[0])
            print(f"\n=== {name}  [{msgtype}] ===")
            # CDR-encoded std_msgs/String: 4-byte CDR header, then 4-byte
            # little-endian string length (includes null terminator),
            # then the string bytes themselves.
            try:
                (str_len,) = struct.unpack_from("<I", raw, 4)
                text = raw[8:8 + str_len].rstrip(b"\x00").decode("utf-8", errors="replace")
                print(text)
            except Exception as e:
                print(f"(couldn't decode: {e}) raw bytes: {raw[:200]!r}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", help="Path to the native .bag recording")
    parser.add_argument("--out", default="output.ply", help="Output .ply path (single-frame mode)")
    parser.add_argument("--frame-index", type=int, default=0, help="Which frame to export (single-frame mode)")
    parser.add_argument("--sequence", action="store_true", help="Export a sequence instead of one frame")
    parser.add_argument("--out-dir", default="ply_frames", help="Output dir for sequence mode")
    parser.add_argument("--every-n", type=int, default=1, help="Export every Nth frame in sequence mode")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of frames exported in sequence mode")
    parser.add_argument("--inspect-db3", help="Path to a .db3 file to dump metadata text from (fallback path)")
    parser.add_argument("--topic-filter", default="info",
                         help="Substring to filter topic names by in --inspect-db3 mode (default: 'info')")

    args = parser.parse_args()

    if args.inspect_db3:
        inspect_db3_metadata(args.inspect_db3, topic_substring=args.topic_filter)
        return

    if not args.bag:
        parser.error("--bag is required (or use --inspect-db3)")

    if args.sequence:
        extract_ply_sequence(args.bag, args.out_dir, every_n=args.every_n, max_frames=args.max_frames)
    else:
        extract_ply_from_bag(args.bag, args.out, frame_index=args.frame_index)


if __name__ == "__main__":
    main()