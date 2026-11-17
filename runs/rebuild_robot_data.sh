#!/bin/bash
set -eu
PY=/raid/home/paragdutta/miniconda3/envs/dl/bin/python
cd /raid/home/paragdutta/Codes/improved-vision-encoders
$PY build_manifest.py --data_dir data
$PY compute_stats.py --data_dir data
$PY -c "
from dataloader import load_robot_manifest, build_robot_splits
m = load_robot_manifest(\"data\")
s = build_robot_splits(m, \"data\", 0.9, 2026)
print(\"splits: train\", len(s[\"train\"]), \"val\", len(s[\"val\"]))
"
echo REBUILD_DONE
