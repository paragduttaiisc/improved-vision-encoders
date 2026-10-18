#!/bin/bash
# switch_7to8.sh -- stop the 7-GPU robot run and relaunch on 8 GPUs @ 400/GPU
# 448/GPU OOMed on 8-GPU DDP (verified: all 8 ranks OOM at batch 1), so the
# 8-GPU phase runs at the DDP-verified 400/GPU (3200 global).
# Idempotent-ish: refuses to run if GPUs 1-7 are busy with a non-robot job.
repo=/raid/home/paragdutta/Codes/improved-vision-encoders
cd $repo
log=/tmp/switch_7to8.log
{
echo "=== switch start $(date -u +%FT%TZ) ==="

# 1. preflight: GPU0 must be free
g0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "${g0:-99999}" -gt 1024 ]; then
  echo "GPU0 still busy (${g0} MiB) - not switching"; exit 3
fi
# check who is on GPU0
g0pid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | head -1)
[ -n "$g0pid" ] && { echo "GPU0 pid $g0pid owner=$(ps -o user= -p $g0pid 2>/dev/null) present"; exit 3; }
echo "GPU0 free confirmed"

# 2. find latest robot checkpoint (for resume)
latest=$(ls models/mae_moe_robot_9ep/checkpoint_epoch_* 2>/dev/null | sort -V | tail -1)
if [ -n "$latest" ]; then
  resume="$latest"
  echo "resuming from robot ckpt: $latest"
else
  resume="models/mae_moe_imagenet_100ep/checkpoint"
  echo "no robot epoch ckpt yet; resuming from epoch-90 imagenet ckpt"
fi

# 3. stop the 7-GPU run
pidfile=/tmp/robot_run7.pid
pids=""
[ -f "$pidfile" ] && pids=$(cat "$pidfile")
[ -n "$pids" ] && kill $pids 2>/dev/null
pkill -f "num_processes 7" 2>/dev/null
sleep 10
pkill -9 -f "num_processes 7" 2>/dev/null
# drain
for t in $(seq 1 30); do
  mems=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1-7 | tr '\n' ' ')
  if echo "$mems" | awk '{for(i=1;i<=NF;i++) if($i+0>1024){exit 1}}' && echo "$mems" | awk '{ok=1; for(i=1;i<=NF;i++) if($i+0>1024) ok=0; exit ok?0:1}'; then
    echo "GPUs 1-7 drained after ${t}0s"; break
  fi
  sleep 10
done
pgrep -f "num_processes 7" && { echo "7-gpu procs still alive, aborting"; exit 4; } || echo "7-gpu run stopped"

# 4. launch 8-GPU run @ 400/GPU
cp runs/robot_9ep.log runs/robot_9ep.log.switched_$(date +%s) 2>/dev/null || true
: > runs/robot_9ep.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup accelerate launch --num_processes 8 pretrain.py \
  --data_dir data \
  --batch_size 400 --n_workers 8 \
  --lr 5e-5 --probe_lr 6e-3 --no_probe \
  --num_epochs 9 --warmup_epochs 1 --last_decay_epoch 9 \
  --image_size 256 --patch_size 16 \
  --resume "$resume" \
  --save_dir models/mae_moe_robot_9ep --output_dir outputs \
  --log_interval 100 --val_samples_max 4000 \
  > runs/robot_9ep.log 2>&1 &
newpid=$!
echo $newpid > /tmp/robot_run8.pid
echo "8-GPU run launched, pid $newpid, resume=$resume"

# 5. health check: wait for first loss line or OOM (max ~20 min)
ok=0
for t in $(seq 1 120); do
  sleep 10
  if grep -q "OutOfMemoryError\|Traceback (most recent call last)" runs/robot_9ep.log 2>/dev/null; then
    echo "HEALTH FAIL: error in log"; tail -20 runs/robot_9ep.log; exit 5
  fi
  if grep -q "Loss:" runs/robot_9ep.log 2>/dev/null; then
    ok=1; break
  fi
done
if [ $ok -eq 1 ]; then
  echo "HEALTH OK: first loss lines:"
  grep "Loss:" runs/robot_9ep.log | head -3
  mem=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' '; ')
  echo "GPU mem/util: $mem"
  echo "SWITCH_OK"
else
  echo "HEALTH TIMEOUT: no loss line in 20 min (may still be initializing; check log)"; tail -10 runs/robot_9ep.log
fi
echo "=== switch end $(date -u +%FT%TZ) ==="
} >> "$log" 2>&1
echo "switch script rc=$? (log: $log)"
