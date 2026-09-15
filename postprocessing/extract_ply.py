"""
Extract a colorized point cloud (.ply) directly from a converted RealSense
.db3 file -- no original .bag and no pyrealsense2 required.

This parses the recording's own metadata topics for:
  - depth camera intrinsics       (.../Depth_0/camera_info)
  - color camera intrinsics       (.../Color_0/camera_info)
  - depth scale (meters/unit)     (.../option/Depth_Units/value)
  - depth->color extrinsics       (.../Color_0/tf/ref_0)
so nothing about your specific camera/geometry is hardcoded.

Usage:
    # single frame (0 = first pair of depth+color frames)
    python extract_ply_from_db3.py --db3 recording.db3 --out output.ply --frame-index 0

    # every frame, one .ply per pair
    python extract_ply_from_db3.py --db3 recording.db3 --all --out-dir ply_frames

Requirements:
    pip install rosbags numpy
"""

import argparse
import bisect
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from rosbags.serde import deserialize_cdr
except ImportError:
    print("This script requires the 'rosbags' package: pip install rosbags")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def _decode_string_message(raw: bytes) -> str:
    """Decode a CDR-encoded std_msgs/String payload."""
    (str_len,) = struct.unpack_from("<I", raw, 4)
    return raw[8:8 + str_len].rstrip(b"\x00").decode("utf-8", errors="replace")


def _read_first_string_topic(conn: sqlite3.Connection, topic_name: str) -> str:
    cur = conn.cursor()
    cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Topic '{topic_name}' not found in this .db3.")
    topic_id = row[0]
    cur.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (topic_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Topic '{topic_name}' has no messages.")
    return _decode_string_message(bytes(row[0]))


def _parse_kv(text: str) -> dict:
    """Parse 'key=value;key=value;...' strings into a dict of raw string values."""
    out = {}
    for pair in text.split(";"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class Intrinsics:
    def __init__(self, width, height, fx, fy, ppx, ppy):
        self.width, self.height = width, height
        self.fx, self.fy, self.ppx, self.ppy = fx, fy, ppx, ppy


def get_intrinsics(conn: sqlite3.Connection, camera_info_topic: str) -> Intrinsics:
    kv = _parse_kv(_read_first_string_topic(conn, camera_info_topic))
    return Intrinsics(
        width=int(kv["width"]), height=int(kv["height"]),
        fx=float(kv["fx"]), fy=float(kv["fy"]),
        ppx=float(kv["ppx"]), ppy=float(kv["ppy"]),
    )


def get_depth_scale(conn: sqlite3.Connection, depth_units_topic: str) -> float:
    return float(_read_first_string_topic(conn, depth_units_topic).strip())


def get_extrinsics(conn: sqlite3.Connection, tf_topic: str):
    """
    Returns (R, t) such that: p_color = R @ p_depth + t
    librealsense stores the 'rotation' 9-tuple column-major.
    """
    kv = _parse_kv(_read_first_string_topic(conn, tf_topic))
    rot = [float(x) for x in kv["rotation"].split(",")]
    trans = np.array([float(x) for x in kv["translation"].split(",")], dtype=np.float64)
    # column-major -> R[row, col] = rot[col*3 + row]
    R = np.array(rot, dtype=np.float64).reshape(3, 3, order="F")
    return R, trans


# ---------------------------------------------------------------------------
# Image reading
# ---------------------------------------------------------------------------

def read_images(db3_path: str, topic_name: str):
    """Yields (timestamp_ns, array) for an Image topic, dtype/shape depending on encoding."""
    conn = sqlite3.connect(db3_path)
    cur = conn.cursor()
    cur.execute("SELECT id, type FROM topics WHERE name = ?", (topic_name,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Topic '{topic_name}' not found in this .db3.")
    topic_id, msgtype = row

    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (topic_id,))
    rows = cur.fetchall()
    conn.close()

    for ts, data in rows:
        msg = deserialize_cdr(bytes(data), msgtype)
        enc = msg.encoding.lower()
        if enc in ("z16", "mono16", "16uc1"):
            arr = np.frombuffer(msg.data, dtype="<u2").reshape(msg.height, msg.width)
        elif enc in ("y8", "mono8", "8uc1"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        elif enc == "rgb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif enc == "bgr8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)[..., ::-1]
        else:
            raise ValueError(f"Unhandled image encoding '{msg.encoding}' on topic '{topic_name}'.")
        yield ts, arr


def nearest_index(sorted_ts, target_ts):
    i = bisect.bisect_left(sorted_ts, target_ts)
    if i == 0:
        return 0
    if i == len(sorted_ts):
        return len(sorted_ts) - 1
    before, after = sorted_ts[i - 1], sorted_ts[i]
    return i - 1 if (target_ts - before) <= (after - target_ts) else i


# ---------------------------------------------------------------------------
# Core: backproject depth -> transform -> colorize -> PLY
# ---------------------------------------------------------------------------

def build_colored_point_cloud(depth_img, color_img, depth_scale,
                               depth_intr: Intrinsics, color_intr: Intrinsics,
                               R, t):
    h, w = depth_img.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth_img > 0

    z = depth_img[valid].astype(np.float64) * depth_scale
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)

    x = (u - depth_intr.ppx) * z / depth_intr.fx
    y = (v - depth_intr.ppy) * z / depth_intr.fy

    pts_depth = np.stack([x, y, z], axis=1)  # (N, 3) in depth camera frame

    # transform into color camera frame
    pts_color = pts_depth @ R.T + t

    # project into color image
    zc = pts_color[:, 2]
    in_front = zc > 0
    uc = np.zeros_like(zc)
    vc = np.zeros_like(zc)
    uc[in_front] = (pts_color[in_front, 0] / zc[in_front]) * color_intr.fx + color_intr.ppx
    vc[in_front] = (pts_color[in_front, 1] / zc[in_front]) * color_intr.fy + color_intr.ppy

    uc_i = np.round(uc).astype(np.int64)
    vc_i = np.round(vc).astype(np.int64)
    in_bounds = in_front & (uc_i >= 0) & (uc_i < color_intr.width) & (vc_i >= 0) & (vc_i < color_intr.height)

    pts_final = pts_depth[in_bounds]
    colors_final = color_img[vc_i[in_bounds], uc_i[in_bounds]]

    return pts_final, colors_final


def write_ply_binary(filepath, points: np.ndarray, colors: np.ndarray):
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    vertex = np.zeros(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(db3_path, out_path, frame_index,
            depth_topic, color_topic, depth_info_topic, color_info_topic,
            depth_units_topic, color_tf_topic):
    conn = sqlite3.connect(db3_path)
    depth_intr = get_intrinsics(conn, depth_info_topic)
    color_intr = get_intrinsics(conn, color_info_topic)
    depth_scale = get_depth_scale(conn, depth_units_topic)
    R, t = get_extrinsics(conn, color_tf_topic)
    conn.close()

    color_frames = list(read_images(db3_path, color_topic))
    color_ts = [c[0] for c in color_frames]

    depth_gen = read_images(db3_path, depth_topic)

    ts, depth_img = None, None
    for i, (d_ts, d_img) in enumerate(depth_gen):
        if i == frame_index:
            ts, depth_img = d_ts, d_img
            break
    if depth_img is None:
        raise ValueError(f"No depth frame at index {frame_index}.")

    ci = nearest_index(color_ts, ts)
    color_img = color_frames[ci][1]

    pts, colors = build_colored_point_cloud(depth_img, color_img, depth_scale, depth_intr, color_intr, R, t)
    write_ply_binary(out_path, pts, colors)
    print(f"Saved {pts.shape[0]} colored points to {out_path}")


def process_all(db3_path, out_dir, **topics):
    conn = sqlite3.connect(db3_path)
    depth_intr = get_intrinsics(conn, topics["depth_info_topic"])
    color_intr = get_intrinsics(conn, topics["color_info_topic"])
    depth_scale = get_depth_scale(conn, topics["depth_units_topic"])
    R, t = get_extrinsics(conn, topics["color_tf_topic"])
    conn.close()

    color_frames = list(read_images(db3_path, topics["color_topic"]))
    color_ts = [c[0] for c in color_frames]

    for i, (ts, depth_img) in enumerate(read_images(db3_path, topics["depth_topic"])):
        ci = nearest_index(color_ts, ts)
        color_img = color_frames[ci][1]
        pts, colors = build_colored_point_cloud(depth_img, color_img, depth_scale, depth_intr, color_intr, R, t)
        out_path = f"{out_dir}/frame_{i:06d}.ply"
        write_ply_binary(out_path, pts, colors)

    print(f"Saved {i + 1} colored point clouds to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db3", required=True, help="Path to the .db3 file")
    parser.add_argument("--out", default="output.ply", help="Output .ply path (single-frame mode)")
    parser.add_argument("--frame-index", type=int, default=0, help="Which depth frame to export (single-frame mode)")
    parser.add_argument("--all", action="store_true", help="Export every frame instead of a single one")
    parser.add_argument("--out-dir", default="ply_frames", help="Output dir for --all mode")

    # topic overrides, defaulting to what a standard D435 recording uses
    parser.add_argument("--depth-topic", default="/device_0/sensor_0/Depth_0/image/data")
    parser.add_argument("--color-topic", default="/device_0/sensor_1/Color_0/image/data")
    parser.add_argument("--depth-info-topic", default="/device_0/sensor_0/Depth_0/camera_info")
    parser.add_argument("--color-info-topic", default="/device_0/sensor_1/Color_0/camera_info")
    parser.add_argument("--depth-units-topic", default="/device_0/sensor_0/option/Depth_Units/value")
    parser.add_argument("--color-tf-topic", default="/device_0/sensor_1/Color_0/tf/ref_0")

    args = parser.parse_args()

    topics = dict(
        depth_topic=args.depth_topic, color_topic=args.color_topic,
        depth_info_topic=args.depth_info_topic, color_info_topic=args.color_info_topic,
        depth_units_topic=args.depth_units_topic, color_tf_topic=args.color_tf_topic,
    )

    if args.all:
        process_all(args.db3, args.out_dir, **topics)
    else:
        process(args.db3, args.out, args.frame_index, **topics)


if __name__ == "__main__":
    main()