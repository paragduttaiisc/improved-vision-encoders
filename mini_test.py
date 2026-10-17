"""Local unit test of build_manifest + dataloader split logic on a
synthetic mini robot_jpg tree (no real data needed)."""
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)

import build_manifest as bm
import dataloader as dl

ROOT = "/tmp/mini_data"
shutil.rmtree(ROOT, ignore_errors=True)
rj = os.path.join(ROOT, "robot_jpg")

# dataset A: 3 episodes, frames 3/4/2; dataset B: 2 episodes, frames 5/1
plan = [
    ("ds_a", {0: 3, 1: 4, 10: 2}),
    ("ds_b", {0: 5, 2: 1}),
]
for ds, eps in plan:
    for ep, nframes in eps.items():
        d = os.path.join(rj, ds, str(ep))
        os.makedirs(d, exist_ok=True)
        for i in range(nframes):
            arr = (np.random.RandomState(ep * 100 + i).rand(256, 256, 3)
                   * 255).astype("uint8")
            Image.fromarray(arr).save(os.path.join(d, f"frame_{i:06d}.jpg"),
                                      "JPEG")

summary = bm.build_manifest(rj, os.path.join(ROOT, "manifest.jsonl"))
print("manifest summary:", summary)
assert os.path.getsize(os.path.join(ROOT, "manifest.jsonl")) == \
    summary["n_frames"] * 70

man = dl.load_robot_manifest(ROOT)
n = len(man["rows"])
print("n rows:", n, "names:", man["names"])
assert n == summary["n_frames"] == 15
assert man["names"] == ["ds_a", "ds_b"]
# episode blocks: ds_a:3,4,2 ; ds_b:5,1
print("bpos:", man["bpos"].tolist())
print("ep_counts:", man["ep_counts"].tolist())
assert man["ep_counts"].tolist() == [3, 4, 2, 5, 1]
# rel path extraction
for i in range(n):
    rel = dl._row_rel(man["rows"][i])
    assert os.path.exists(os.path.join(ROOT, "robot_jpg", rel)), rel
print("rel paths ok:", [dl._row_rel(man["rows"][i]) for i in range(n)])

# split with train_frac=0.9 -> per episode: 1 val frame (for m>=2),
# m=1 episode -> all train
splits = dl.build_robot_splits(man, ROOT, train_frac=0.9, seed=2026)
train, val = splits["train"], splits["val"]
assert len(train) + len(val) == n
assert len(np.intersect1d(train, val)) == 0
# per-episode val counts: 1 each for 4 eps with >=2 frames, 0 for m=1
print("val:", val.tolist(), "train:", train.tolist())
assert len(val) == 4
# determinism: delete cache, rebuild, same result
import dataloader
p = dataloader._splits_path(ROOT, 2026, 0.9)
os.remove(p)
splits2 = dl.build_robot_splits(man, ROOT, train_frac=0.9, seed=2026)
assert (np.sort(splits["train"]) == np.sort(splits2["train"])).all()
assert (np.sort(splits["val"]) == np.sort(splits2["val"])).all()
print("deterministic rebuild ok")

# dataloader end-to-end
import torch
tl, vl, names = dl.get_robot_dataloaders(ROOT, 8, 2)
imgs, labels = next(iter(tl))
assert imgs.dtype == torch.uint8
assert imgs.shape[-3:] == (3, 256, 256)
assert imgs.min() >= 0 and imgs.max() <= 255
print("batch:", tuple(imgs.shape), labels.tolist())
print("ALL SPLIT/LOADER TESTS PASSED")
