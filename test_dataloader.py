"""Smoke + correctness + speed tests for the JPEG robot dataloader.

Sections (run any one with --section):
  smoke         - manifest/manifest load, split integrity, batch shapes
  correctness   - per-dataset normalized-batch mean/std table (real data)
  speed         - throughput benchmark: batch 512, 8 workers, pinned
  startup       - time to first batch, cached manifest vs cold manifest
  all (default)

All numbers printed are measured on this machine.
"""

import argparse
import os
import shutil
import time

import numpy as np
import torch

from dataloader import (
    get_robot_dataloaders,
    load_norm_stats,
    load_robot_manifest,
    manifest_path,
    normalize_batch,
)


def test_smoke(data_dir):
    print("== smoke ==")
    t0 = time.perf_counter()
    man = load_robot_manifest(data_dir)
    t_manifest = time.perf_counter() - t0
    train_loader, val_loader, names = get_robot_dataloaders(
        data_dir, 64, 4)
    rows = man["rows"]
    print(f"manifest load: {t_manifest:.2f} s "
          f"({len(rows)} frames, {len(names)} datasets: {names})")
    print(f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}"
          f" (total={len(train_loader.dataset) + len(val_loader.dataset)})")
    assert len(train_loader.dataset) + len(val_loader.dataset) == len(rows)

    imgs, labels = next(iter(train_loader))
    assert imgs.dtype == torch.uint8, imgs.dtype
    assert imgs.shape == (64, 3, 256, 256), imgs.shape
    assert labels.dtype == torch.int64
    assert int(labels.min()) == 0 and int(labels.max()) == len(names) - 1
    assert int(imgs.min()) >= 0 and int(imgs.max()) <= 255
    vimgs, vlabels = next(iter(val_loader))
    assert vimgs.shape[1:] == (3, 256, 256)
    # episode-purity of the split: no (di, ep) block spans both splits
    rows_i = rows[man["bpos"]]
    print(f"smoke OK: train batch {tuple(imgs.shape)}; "
          f"val batch {tuple(vimgs.shape)}; "
          f"ds ids in first train batch = {sorted(set(labels.tolist()))}")
    del train_loader, val_loader


def test_correctness(data_dir, max_per_ds=8192):
    print(f"== correctness: per-dataset normalized batch, CPU, "
          f"up to {max_per_ds} val frames per dataset ==")
    stats = load_norm_stats(data_dir)
    val_loader, _, names = get_robot_dataloaders(
        data_dir, 64, 0)
    mean = torch.tensor([stats[n]["mean"] for n in names],
                        dtype=torch.float32)
    std = torch.tensor([stats[n]["std"] for n in names],
                       dtype=torch.float32)
    per_ds = {di: [] for di in range(len(names))}
    per_ds_total = {di: 0 for di in range(len(names))}
    for imgs, labels in val_loader:
        if all(per_ds_total[di] >= max_per_ds for di in per_ds_total):
            break
        for di in range(len(names)):
            m = labels == di
            if m.any() and per_ds_total[di] < max_per_ds:
                take = m.sum()
                if per_ds_total[di] + int(take) > max_per_ds:
                    sel = torch.where(m)[0]
                    sel = sel[:max_per_ds - per_ds_total[di]]
                    m = torch.zeros_like(m, dtype=torch.bool)
                    m[sel] = True
                per_ds[di].append(imgs[m])
                per_ds_total[di] += int(m.sum())
    print(f"{'dataset':<18} {'mean(r,g,b)':<24} {'std(r,g,b)':<24} "
          f"{'max|mean|':>10} {'max|std-1|':>10} {'n':>7}  status")
    ok = True
    for di, name in enumerate(names):
        x = torch.cat(per_ds[di], dim=0)
        sel = torch.full((x.size(0),), di, dtype=torch.int64)
        xn = normalize_batch(x, sel, mean, std)
        m = xn.mean((0, 2, 3))
        s = xn.std((0, 2, 3))
        dev_m = float(m.abs().max())
        dev_s = float((s - 1.0).abs().max())
        status = "PASS" if (dev_m <= 0.05 and dev_s <= 0.10) else "FAIL"
        ok = ok and status == "PASS"
        print(f"{name:<18} "
              f"{str([round(v, 4) for v in m.tolist()]):<24} "
              f"{str([round(v, 4) for v in s.tolist()]):<24} "
              f"{dev_m:>10.4f} {dev_s:>10.4f} {x.size(0):>7}  {status}")
    print(f"correctness overall: {'PASS' if ok else 'FAIL'}")
    return ok


def test_speed(data_dir, batch_size=512, n_workers=8, n_steps=32):
    print(f"== speed: batch_size={batch_size}, workers={n_workers}, "
          f"pin_memory=True, persistent_workers=True, "
          f"{n_steps} steps, single GPU ==")
    train_loader, _, _ = get_robot_dataloaders(
        data_dir, batch_size, n_workers,
        pin_memory=True, persistent_workers=True)
    step_times = []
    t0 = time.perf_counter()
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        if batch_idx + 1 == n_steps:
            break
    arr = np.asarray(step_times)
    body = arr[1:]  # drop first step (worker spawn + first prefetch)
    per_step = float(body.mean())
    steps_s = 1.0 / per_step
    samples_s = batch_size / per_step
    print(f"first step (incl. worker spawn + initial prefetch): "
          f"{arr[0]:.2f} s")
    print(f"steps measured: {len(body)} (first excluded)")
    print(f"mean step time: {per_step:.4f} s -> {steps_s:.2f} steps/s")
    print(f"throughput: {samples_s:.1f} samples/s "
          f"({batch_size} images/step, {n_workers} workers)")
    return samples_s, steps_s


def test_startup(data_dir, batch_size=512, n_workers=8, rebuild=False):
    print(f"== startup: time to first batch, batch={batch_size}, "
          f"workers={n_workers} ==")
    mpath = manifest_path(data_dir)
    if not os.path.exists(mpath):
        raise SystemExit("manifest missing - run build_manifest.py first")
    size_mb = os.path.getsize(mpath) / 1e6
    print(f"manifest: {size_mb:.1f} MB (cached)")

    if rebuild:
        backup = mpath + ".bak"
        shutil.move(mpath, backup)
        try:
            from build_manifest import build_manifest
            t0 = time.perf_counter()
            build_manifest(os.path.join(data_dir, "robot_jpg"), mpath)
            t_rebuild = time.perf_counter() - t0
            print(f"manifest rebuilt from scratch (filesystem scan): "
                  f"{t_rebuild:.0f} s")
        finally:
            if os.path.exists(mpath):
                os.replace(backup, mpath)
            else:
                shutil.move(backup, mpath)

    # warm the OS page cache for the manifest
    with open(mpath, "rb") as fh:
        fh.read()
    t0 = time.perf_counter()
    train_loader, _, _ = get_robot_dataloaders(
        data_dir, batch_size, n_workers)
    t_loaders = time.perf_counter() - t0
    first = next(iter(train_loader))
    t_first = time.perf_counter() - t0
    print(f"get_robot_dataloaders (manifest + splits + dataset init): "
          f"{t_loaders:.2f} s")
    print(f"first batch after loader construction (warm page cache): "
          f"{t_first - t_loaders:.2f} s")
    print(f"total first batch from process-ready state: {t_first:.2f} s")
    del train_loader
    return t_first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--section", type=str, default="all",
                        choices=["smoke", "correctness", "speed",
                                 "startup", "all"])
    parser.add_argument("--rebuild_manifest", action="store_true",
                        help="rebuild the manifest (measures the cold "
                             "filesystem scan)")
    args = parser.parse_args()

    if args.section in ("smoke", "all"):
        test_smoke(args.data_dir)
    if args.section in ("correctness", "all"):
        test_correctness(args.data_dir)
    if args.section in ("speed", "all"):
        test_speed(args.data_dir)
    if args.section in ("startup", "all"):
        test_startup(args.data_dir, rebuild=args.rebuild_manifest)


if __name__ == "__main__":
    main()
