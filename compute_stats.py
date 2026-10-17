"""Per-dataset pixel mean/std (on [0,1]) from a random frame sample.

Writes <data_dir>/norm_stats.json:
    {dataset_name: {"mean": [3], "std": [3], "n_samples": int}, ...}
The "imagenet" entry is the standard ImageNet mean/std (not sampled).

Samples >= 2000 random frames per dataset with a fixed seed (default
1234, 4096 samples), decodes with torchvision's C++ JPEG decoder, and
accumulates float64 sums.

Usage:
    python compute_stats.py [--data_dir data] [--n_samples 4096]
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from dataloader import _row_rel, decode_jpeg_bytes, load_robot_manifest


def sample_dataset_stats(man: dict, di: int, n_samples: int,
                         data_dir: str) -> dict:
    """Per-dataset channel mean/std over n random frames of dataset di."""
    rows = man["rows"]
    di_mask = rows[:, 0] == di
    idxs = np.flatnonzero(di_mask)
    assert len(idxs) >= 1, f"dataset {di} has no frames"
    n = min(n_samples, len(idxs))
    picks = np.array(random.Random(1234).sample(list(idxs), n),
                     dtype=np.int64)

    sum_x = torch.zeros(3, dtype=torch.float64)
    sum_x2 = torch.zeros(3, dtype=torch.float64)
    for i in picks:
        with open(
                os.path.join(data_dir, "robot_jpg",
                             _row_rel(rows[int(i)])), "rb") as fh:
            img = decode_jpeg_bytes(
                torch.frombuffer(fh.read(), dtype=torch.uint8)
            ).float() / 255.0
        sum_x += img.flatten(1).sum(1).double()
        sum_x2 += (img.flatten(1) ** 2).sum(1).double()

    n_px = n * 256 * 256
    mean = (sum_x / n_px).numpy()
    var = (sum_x2 / n_px - mean ** 2)
    return {"mean": [float(v) for v in mean],
            "std": [float(v) for v in np.sqrt(var)],
            "n_samples": int(n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_samples", type=int, default=4096,
                        help="Random frames sampled per dataset (>= 2000).")
    args = parser.parse_args()
    assert args.n_samples >= 2000, "Use at least 2000 samples per dataset."

    man = load_robot_manifest(args.data_dir)
    stats = {}
    for di, name in enumerate(man["names"]):
        stats[name] = sample_dataset_stats(man, di, args.n_samples,
                                           args.data_dir)
        print(f"[stats] {name}: mean={stats[name]['mean']} "
              f"std={stats[name]['std']} ({stats[name]['n_samples']} frames)",
              flush=True)

    stats["imagenet"] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "n_samples": None,
    }
    out = os.path.join(args.data_dir, "norm_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
