#!/bin/bash
# Short 8-GPU ImageNet run with the mae_moe_imagenet_100ep settings, to check
# the current code end to end and read off throughput (Img/s in the log).
# Usage: runs/smoke_imagenet.sh [max_steps=200]   -> runs/smoke_imagenet.log
set -u
source /raid/home/paragdutta/miniconda3/etc/profile.d/conda.sh
conda activate dl
cd /raid/home/paragdutta/Codes/improved-vision-encoders
for i in 0 1 2 3 4 5 6 7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
  [ "${m:-99999}" -gt 1024 ] && { echo "REFUSING LAUNCH: GPU$i busy (${m} MiB)" >&2; exit 2; }
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "git commit: $(git rev-parse --short HEAD) ($(git branch --show-current))" > runs/smoke_imagenet.log
accelerate launch --num_processes 8 pretrain.py \
  --data_dir data/imagenet-540k-1k --image_size 224 --n_classes 1000 \
  --batch_size 512 --n_workers 8 --lr 2e-4 --probe_lr 6e-3 \
  --num_epochs 100 --warmup_epochs 10 --last_decay_epoch 90 \
  --max_steps "${1:-200}" --log_interval 10 \
  --save_dir /tmp/smoke_imagenet_models --output_dir /tmp/smoke_imagenet_outputs \
  2>&1 | python3 runs/tsline.py >> runs/smoke_imagenet.log
echo "EXIT=${PIPESTATUS[0]}" >> runs/smoke_imagenet.log
