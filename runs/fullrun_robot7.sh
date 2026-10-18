#!/bin/bash
# Full 9-epoch continue-pretrain on the robot dataset, resuming the
# ImageNet 100-ep checkpoint (epoch-90-start save, 89 clean epochs).
# 7-GPU DDP phase (GPU0 held by nikhils): CUDA_VISIBLE_DEVICES=1-7,
# num_processes 7, per-GPU batch 448 (=3136 global), image_size 256.
# DDP-verified 105+ clean steps, 0 OOM (peak ~39.9GB/GPU -- tight but stable).
# NOTE: --batch_size in this codebase is PER-GPU (dataloader.py uses it as-is).
# LR stays 5e-5 peak per user choice.
#
# Fresh LR schedule (NOT the stage-1 scheduler):
#   --lr 5e-5  -> 1 epoch linear warmup 5e-8 -> 5e-5,
#                 cosine 5e-5 -> 2.5e-6 (= lr/20) over the remaining 8 epochs.
# Probe: DISABLED for robot runs (--no_probe; no class concept in robot data).
#
# Checkpoints: per-epoch (save_interval 1) at end of each epoch's training
#   -> models/mae_moe_robot_9ep/checkpoint_epoch_{1..9} (~5.5 GB each),
#   plus checkpoint_final after the last epoch's validation.
#   The GPU0-free watcher (switch_7to8.sh) stops this run and resumes on
#   8 GPUs @ 400/GPU from the newest epoch checkpoint.
#
# Detach with:  setsid nohup bash runs/fullrun_robot7.sh >/dev/null 2>&1 &
#               echo $! > runs/robot_9ep7.pid
# Log: runs/robot_9ep.log
set -u
source /raid/home/paragdutta/miniconda3/etc/profile.d/conda.sh
conda activate dl
cd /raid/home/paragdutta/Codes/improved-vision-encoders

# GPU gate: require GPUs 1-7 free before touching anything
for i in 1 2 3 4 5 6 7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
  if [ "${m:-99999}" -gt 1024 ]; then
    echo "REFUSING LAUNCH: GPU$i busy (${m} MiB)" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs models/mae_moe_robot_9ep

{
  echo "=== launch banner $(date -u '+%Y-%m-%d %H:%M:%S UTC') (7-GPU phase) ==="
  echo "git commit: $(git rev-parse --short HEAD) ($(git branch --show-current))"
  echo "gpu check:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "=== pretrain.py launch ==="
} >> runs/robot_9ep.log

accelerate launch --num_processes 7 pretrain.py \
  --data_dir data --image_size 256 --n_classes 8 \
  --batch_size 448 --n_workers 8 \
  --lr 5e-5 --probe_lr 6e-3 --no_probe \
  --num_epochs 9 --warmup_epochs 1 --last_decay_epoch 9 \
  --save_interval 1 --val_samples_max 4000 \
  --resume models/mae_moe_imagenet_100ep/checkpoint \
  --save_dir models/mae_moe_robot_9ep --output_dir outputs \
  --run_name mae_moe_robot_9ep_7gpu \
  2>&1 | python3 "$(pwd)/runs/tsline.py" >> runs/robot_9ep.log
RC=${PIPESTATUS[0]}
echo "ROBOT_9EP_EXIT=${RC}" >> runs/robot_9ep.log
exit $RC
