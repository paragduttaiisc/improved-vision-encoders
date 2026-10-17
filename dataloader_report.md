# JPEG dataloader report (branch `jpeg-dataloader`)

Date: 2026-10-17. All numbers below come from executed runs on Mriya
(10.192.30.30), Python `/raid/home/paragdutta/miniconda3/envs/dl/bin/python`
(torch 2.14.0+cu130, torchvision 0.29.0), repo
`/raid/home/paragdutta/Codes/improved-vision-encoders`.

## What was built

- **`build_manifest.py`** — one-time filesystem scan of `data/robot_jpg/`
  into `data/manifest.jsonl` (fixed-width binary, 70 bytes/frame:
  `[di:1B][tab][ep:4B LE u32][tab][rel path:62B NUL-padded][\n]`) plus
  `data/manifest_meta.json`. Written dataset-major / episode-major /
  frame-sorted, so an episode is always a contiguous block.
- **`compute_stats.py`** — per-dataset channel mean/std on [0,1] pixels
  from 4096 random frames per dataset (seed 1234, documented in the
  script) → `data/norm_stats.json` (includes the ImageNet entry
  0.485/0.456/0.406, 0.229/0.224/0.225).
- **`dataloader.py`** (replaced) —
  - `load_robot_manifest()`: one buffered read + vectorized numpy, loads
    all 20,464,561 frames' manifest in **1.29 s** (measured).
  - `build_robot_splits()`: per-episode 90/10 split (seed 2026); an
    episode's frames always stay on one side; cached as
    `data/robot_splits_seed2026_frac0.9.npz` (int64 indices, ~300 MB).
  - `RobotJpgDataset`: holds the 1.43 GB manifest buffer + split index
    only — no 20M-string pickle per worker; paths extracted per item.
  - `normalize_batch()`: per-sample per-dataset (x/255 − mean)/std as one
    batched GPU op, stats gathered by dataset idx.
  - `get_robot_dataloaders()` / `get_imagenet_dataloaders()` /
    `get_dataloaders()` dispatcher (legacy ImageFolder path preserved,
    verified working, reads ImageNet mean/std from norm_stats.json).
  - Batches: uint8 [B,3,256,256] + dataset-idx int64 labels.
- **`pretrain.py`** (patched) — uses the new loaders, normalizes on GPU
  with per-dataset stats, auto-sets `n_classes` to 8 for robot data,
  adds `--max_steps` for smoke runs. `utils.py::denormalize` now takes
  optional mean/std (per-sample [B,3] supported).
- **`test_dataloader.py`** — `--section {smoke,correctness,speed,startup}`.
- `mini_test.py` — synthetic-tree unit test of manifest + split logic
  (kept in repo for regression).

## 1. Correctness — PASS

`test_dataloader.py --section correctness`: for each dataset, up to 8192
val frames normalized with `normalize_batch` then measured per-channel
mean/std on CPU. Thresholds: |mean| ≤ 0.05, |std − 1| ≤ 0.10.

```
dataset            mean(r,g,b)              std(r,g,b)              max|mean| max|std-1|     n  status
bc_z               [-0.0, 0.0003, 0.0006]   [1.0018, 1.002, 1.0021]   0.0006     0.0021   8192  PASS
bridge             [-0.0035, 0.002, 0.0045] [0.9997, 1.001, 1.0012]   0.0045     0.0012   8192  PASS
custom_robot       [0.0157, 0.0162, 0.016]  [0.9982, 0.9981, 0.9978]  0.0162     0.0022   8192  PASS
freiburg_play      [0.0005, 0.0005, 0.0005] [0.9994, 0.9993, 0.9992]  0.0005     0.0008   8192  PASS
furniture_bench    [0.0012, 0.0009, 0.0012] [0.9991, 0.9993, 0.9989]  0.0012     0.0011   8192  PASS
maniskill          [-0.002, -0.0023, -0.003] [1.0054, 1.0061, 1.0074] 0.0030     0.0074   8192  PASS
qt-opt             [0.0064, 0.0024, -0.0038] [0.9999, 0.9987, 0.9974] 0.0064     0.0026   8192  PASS
rt-1               [0.0017, 0.0017, 0.0005] [1.0001, 1.0002, 0.9999]  0.0017     0.0002   8192  PASS
correctness overall: PASS
```

## 2. Throughput — measured

`test_dataloader.py --section speed`: batch_size=512, num_workers=8,
pin_memory=True, persistent_workers=True, first step excluded, 31 steps
averaged, batch moved to GPU each step on one A100 (GPU 0, idle, no other
jobs). How measured: per-step wall time between consecutive
`imgs.to('cuda') + torch.cuda.synchronize()` returns.

```
first step (incl. worker spawn + initial prefetch): 5.81 s
steps measured: 31 (first excluded)
mean step time: 0.0582 s -> 17.19 steps/s
throughput: 8798.9 samples/s (512 images/step, 8 workers)
```

**Headline: ~8,800 samples/s, ~17.2 steps/s at batch 512.** This is the
dataloader-only number (no model compute); the loader was CPU-bound, not
I/O-bound, on RAID.

Decode backend choice: torchvision C++ `decode_image` (libjpeg) measured
0.254 ms/image vs PIL 0.433 ms/image in a direct A/B benchmark on Mriya
this session (256×256 frames, ~500 frames each). C++ decoder kept.

## 3. Smoke test — 50 train steps, 1 GPU, real loss

`pretrain.py --data_dir data --batch_size 256 --n_workers 8 --num_epochs 1
--max_steps 50 --run_name ''` (batch 256 because batch 512 OOMs on the
A100-40GB after step 1 with this 12-layer MoE encoder+decoder — step 1 at
512 did run, loss 4.4486, before OOM; the throughput number above is the
512 measurement and is dataloader-only, not model training). No wandb
(`--run_name ''`). Loss finite throughout; probe loss decreasing:

```
step    loss      pred     var      sw       probe_loss  probe_acc
1       4.4206    1.1036   0.1301   0.0651   2.1143      0.0547
6       4.4361    1.0864   0.1312   0.0698   0.8499      0.7422
11      4.2855    1.1270   0.1237   0.0655   0.5174      0.8398
16      4.0977    1.0662   0.1186   0.0665   0.5400      0.8008
21      3.9992    1.0912   0.1139   0.0612   0.5039      0.8242
26      4.0114    1.1065   0.1135   0.0679   0.3386      0.8867
31      3.8453    1.1080   0.1069   0.0653   0.3743      0.8945
36      3.8633    1.1279   0.1068   0.0660   0.2276      0.9258
41      3.7495    1.1306   0.1021   0.0663   0.1631      0.9727
46      3.5510    1.0924   0.0957   0.0668   0.1812      0.9453
```

Loss at step ~50 (last logged step 46): **3.5510** (started 4.4206);
probe loss 2.1143 → 0.1812, probe acc 0.0547 → 0.9453 (8 dataset classes,
so the probe is actually separating datasets — sanity check on labels).

## 4. Startup time to first batch

- **Warm (cached manifest)**: `get_robot_dataloaders` (manifest read +
  split load + dataset init) = **1.41 s**; first batch after loader
  construction = 5.24 s; **total 6.65 s** (512 batch, 8 workers).
- **Cold (manifest deleted, rebuilt from the 20.5M-file filesystem
  tree)**: rebuild = **39.6 s** (1.43 GB written); then loader init
  16.16 s (cold page cache) + first batch 5.71 s = **21.87 s** total.
  After that, manifest is cached and startup returns to ~1.4 s for
  loader construction.

## Design notes / deviations

- Normalization is done **on GPU** in the train loop
  (`normalize_batch`), not in-worker: workers only do I/O + C++ JPEG
  decode, which is the measured bottleneck; the per-batch gather + affine
  is microseconds on GPU. Rationale: in-worker `transforms.Normalize`
  would add per-item Python + float-cast cost on the critical path and
  would duplicate 8× stat tensors into worker code for no throughput
  gain — the A/B of decode backends was measured instead of a full
  in-worker-vs-GPU normalize A/B because the GPU path also keeps the
  worker output uint8 (less pinned-memory bandwidth: 256×256×3 bytes vs
  ×4).
- Manifest/splits/norm_stats live under `data/` (gitignored via the
  `data/` entries already in the repo; the generated files are not
  committed). Regeneration: `python build_manifest.py` (~40 s),
  `python compute_stats.py` (~3 min), splits rebuild automatically on
  first `get_robot_dataloaders` call (~1 min).
- Splits are per-episode (episode never straddles train/val); within
  each episode 90/10 by frame, seed 2026. `furniture_bench`/`qt-opt`
  etc. all fit in the 62-byte relative-path field (longest observed:
  30 bytes).
- Known quirk: `np.savez` on a path already ending `.npz` is fine
  (verified); the split cache is validated against the manifest frame
  count on load, so a stale cache fails loudly.
- `test.py` (legacy ImageNet eval) verified working through the new
  `get_dataloaders` dispatcher: train/test batches are normalized float
  224×224, `splits.pt` cache reused, no assertion changes.

## Reproduce

```bash
python build_manifest.py --data_dir data      # ~40 s, one-time
python compute_stats.py                       # ~3 min, one-time
python test_dataloader.py --section all
python pretrain.py --data_dir data --batch_size 256 --n_workers 8 \
       --num_epochs 1 --max_steps 50 --run_name ''
```
