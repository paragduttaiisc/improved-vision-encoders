"""One-time scan of the robot_jpg data dir into <data_dir>/manifest.jsonl.

Output files (in <data_dir>):
  manifest.jsonl        binary, 70 bytes/row, one row per frame:
                          [di:1 byte][tab][ep:4 bytes LE u32][tab]
                          [rel path: 62 bytes, NUL-padded][newline]
                        di = dataset index (0..255);
                        ep = per-dataset episode index (0-based order of
                             the episode dir, so 0..n_eps-1 for that
                             dataset);
                        rel path = "<dataset>/<episode>/<frame>.jpg"
                        relative to <data_dir>/robot_jpg/.
  manifest_meta.json    {"n_datasets": int, "n_frames": int,
                         "names": [dataset dir names...]}

The binary format lets load_robot_manifest() read ~20M entries in one
buffered read + vectorized numpy ops (a few seconds), and lets the
train/val split find episode blocks without rescanning the filesystem.

Re-run this if the robot data changes. ~20M frames -> ~1.4 GB file.

Usage:
    python build_manifest.py [--data_dir data]
"""

import argparse
import json
import os
import struct
import time


ROW_SIZE = 70  # 1 (di) + 1 (tab) + 4 (ep u32 LE) + 1 (tab) + 62 (rel) + 1 (\n)
CHUNK = 200_000


def _encode_row(di: int, ep: int, rel: str) -> bytes:
    rel_bytes = rel.encode("utf-8")
    if len(rel_bytes) >= 62:
        raise ValueError(f"path too long for manifest (>61 bytes): {rel}")
    return (bytes([di]) + b"\t" + struct.pack("<I", ep) + b"\t"
            + rel_bytes.ljust(62, b"\x00") + b"\n")


def build_manifest(robot_dir: str, out_path: str) -> dict:
    meta_path = out_path.replace(".jsonl", "_meta.json")
    tmp_path = out_path + ".tmp"
    datasets = sorted(d for d in os.listdir(robot_dir)
                      if os.path.isdir(os.path.join(robot_dir, d)))
    counts: list[int] = []
    n_frames = 0
    t0 = time.time()
    with open(tmp_path, "wb") as f:
        buf: list[bytes] = []
        for di, ds in enumerate(datasets):
            assert di < 256, "more than 256 datasets"
            ds_dir = os.path.join(robot_dir, ds)
            ep_dirs = [d for d in os.listdir(ds_dir)
                       if os.path.isdir(os.path.join(ds_dir, d))]
            ep_dirs.sort(key=int)  # "0" < "2" < "10"
            ds_count = 0
            for ep, ep_name in enumerate(ep_dirs):
                assert ep < 2**32
                ep_dir = os.path.join(ds_dir, ep_name)
                for fn in sorted(os.listdir(ep_dir)):
                    if fn.endswith(".jpg"):
                        buf.append(_encode_row(di, ep, f"{ds}/{ep_name}/{fn}"))
                        ds_count += 1
                        if len(buf) >= CHUNK:
                            f.write(b"".join(buf))
                            buf.clear()
            if buf:
                f.write(b"".join(buf))
                buf.clear()
            counts.append(ds_count)
            n_frames += ds_count
            print(f"[manifest] {ds}: {ds_count} frames, "
                  f"{len(ep_dirs)} episodes "
                  f"(total {n_frames}, {time.time() - t0:.0f}s)",
                  flush=True)
    os.replace(tmp_path, out_path)
    with open(meta_path, "w") as f:
        json.dump({"n_datasets": len(datasets), "n_frames": n_frames,
                   "names": datasets}, f, indent=2)
    return {"n_datasets": len(datasets), "n_frames": n_frames,
            "elapsed_s": time.time() - t0, "counts": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    robot_dir = os.path.join(args.data_dir, "robot_jpg")
    out_path = os.path.join(args.data_dir, "manifest.jsonl")
    summary = build_manifest(robot_dir, out_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}))
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
