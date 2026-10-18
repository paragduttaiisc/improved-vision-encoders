#!/bin/bash
# Full 9-epoch continue-pretrain on the robot dataset, resuming the
# ImageNet 100-ep checkpoint (epoch-90-start save, 89 clean epochs).
# 8-GPU DDP, wandb on, per-GPU batch 64 (=512 global), image_size 256.
# NOTE: --batch_size in this codebase is PER-GPU (dataloader.py uses it as-is);
# 64 x 8 = 512 global matches stage-1 exactly, so the lr=5e-5 peak calibration holds.
#
# Fresh LR schedule (NOT the stage-1 scheduler):
#   --lr 5e-5  -> 1 epoch linear warmup 5e-8 -> 5e-5,
#                 cosine 5e-5 -> 2.5e-6 (= lr/20) over the remaining 8 epochs.
# Probe keeps its own fresh --probe_lr 6e-3 (new 8-class head).
#
# Checkpoints: per-epoch (save_interval 1) at end of each epoch's training
#   -> models/mae_moe_robot_9ep/checkpoint_epoch_{1..9} (~5.5 GB each),
#   plus checkpoint_final after the last epoch's validation.
#
# Detach with:  setsid nohup bash runs/fullrun_robot.sh >/dev/null 2>&1 &
#               echo $! > runs/robot_9ep.pid
# Log: runs/robot_9ep.log (epoch-seconds-prefixed via runs/tsline.py)
set -u
source /raid/home/paragdutta/miniconda3/etc/profile.d/conda.sh
conda activate dl
cd /raid/home/paragdutta/Codes/improved-vision-encoders
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p runs models/mae_moe_robot_9ep

{
  echo "=== launch banner $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
  echo "git commit: $(git rev-parse --short HEAD) ($(git branch --show-current))"
  echo "gpu check:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "=== pretrain.py launch ==="
} >> runs/robot_9ep.log

accelerate launch --num_processes 8 pretrain.py \
  --data_dir data --image_size 256 --n_classes 8 \
  --batch_size 448 --n_workers 8 \
  --lr 5e-5 --probe_lr 6e-3 \
  --num_epochs 9 --warmup_epochs 1 --last_decay_epoch 9 \
  --save_interval 1 \
  --resume models/mae_moe_imagenet_100ep/checkpoint \
  --save_dir models/mae_moe_robot_9ep --output_dir outputs \
  --run_name mae_moe_robot_9ep \
  2>&1 | python3 "$(pwd)/runs/tsline.py" >> runs/robot_9ep.log
RC=${PIPESTATUS[0]}
echo "ROBOT_9EP_EXIT=${RC}" >> runs/robot_9ep.log
exit $RC
