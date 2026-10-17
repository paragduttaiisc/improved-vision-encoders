#!/bin/bash
# Smoke test: 12 steps on robot data, no wandb, per-GPU batch 64 (=512 global), image_size 256.
# NOTE: --batch_size in this codebase is PER-GPU (dataloader.py uses it as-is).
# Resumes from the stage-1 ImageNet epoch-90-start checkpoint.
# Log: runs/robot_smoke.log ; VRAM sampler: /tmp/smoke_robot_vram.csv
# PASS = first total loss ~0.35-0.45 (continuous with stage-1, NOT ~4.7),
#        no OOM at 256px, peak VRAM recorded, exit code 0.
set -u
source /raid/home/paragdutta/miniconda3/etc/profile.d/conda.sh
conda activate dl
cd /raid/home/paragdutta/Codes/improved-vision-encoders
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

: > /tmp/smoke_robot_vram.csv
( while true; do
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 1
  done > /tmp/smoke_robot_vram.csv ) &
VRAM_PID=$!

accelerate launch --num_processes 8 pretrain.py \
  --data_dir data --image_size 256 --n_classes 8 \
  --batch_size 64 --n_workers 8 \
  --lr 5e-5 --probe_lr 6e-3 \
  --resume models/mae_moe_imagenet_100ep/checkpoint \
  --max_steps 12 --save_interval 99 --save_dir /tmp/robot_smoke_ckpt \
  --log_interval 1 --run_name "" \
  2>&1 | python3 "$(pwd)/runs/tsline.py" >> runs/robot_smoke.log
kill $VRAM_PID 2>/dev/null
echo "SMOKE_EXIT=${RC}" >> runs/robot_smoke.log
exit $RC
