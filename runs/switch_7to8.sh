#!/bin/bash
# switch_7to8.sh -- stop the 7-GPU robot run and resume on all 8 GPUs.
# 8-GPU phase: probe 416/GPU under real 8-GPU DDP (user target; 420 OOMed at
# batch1 before, so 416). If the short probe OOMs, fall back to verified
# 400/GPU (40 clean DDP steps, peak 37.54 GB).
# Uses --start_epoch N+1 so the 8-GPU phase continues at epoch N+1 with the
# LR scheduler fast-forwarded (no re-warmup, no clobbering 7-GPU ckpts).
set -u
repo=/raid/home/paragdutta/Codes/improved-vision-encoders
cd $repo
log=/tmp/switch_7to8.log
# idempotency: refuse if a switch already completed
[ -f /tmp/switch_7to8_done ] && { echo "already switched (marker present)"; exit 0; }
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

# 2. find latest COMPLETE robot epoch checkpoint (for resume)
# A checkpoint dir is "complete" only if save_state finished: accelerator.save_state
# writes model.safetensors/model_1/model_2 + optimizer.bin + random_states +
# accelerate version metadata. Require model.safetensors (the largest file,
# written last by safetensors single-file writer) AND random_states_0.pkl.
# ADDITIONAL GUARD: the checkpoint must be NEWER than the current 7-GPU run's
# launch banner, so a stale leftover from an abandoned earlier run cannot be
# resumed over in-progress work of the live run.
banner_ts=$(grep -m1 "launch banner" runs/robot_9ep.log 2>/dev/null | sed -E 's/.*banner ([0-9-]+ [0-9:]+).*UTC.*/\1/')
[ -z "$banner_ts" ] && { echo "no launch banner in robot_9ep.log; cannot date-gate"; exit 2; }
banner_epoch=$(date -d "$banner_ts" +%s 2>/dev/null)
[ -z "$banner_epoch" ] && { echo "could not parse banner ts '$banner_ts' to epoch"; exit 2; }
echo "7-GPU run launch banner: $banner_ts (epoch $banner_epoch)"

latest=""
for c in $(ls -d models/mae_moe_robot_9ep/checkpoint_epoch_* 2>/dev/null | sort -V); do
  if [ -f "$c/model.safetensors" ] && [ -f "$c/random_states_0.pkl" ]; then
    # mtime of the newest file in the ckpt dir must be >= banner time
    ckpt_epoch=$(stat -c %Y "$c/random_states_0.pkl" 2>/dev/null || echo 0)
    if [ "$ckpt_epoch" -ge "$banner_epoch" ]; then
      latest="$c"
    else
      echo "skipping STALE $c (random_states mtime $(date -d @$ckpt_epoch -u +%FT%TZ) < banner)"
    fi
  fi
done
if [ -z "$latest" ]; then
  echo "no complete+fresh epoch checkpoint yet (7-GPU run may still be in epoch 1)"
  echo "will retry on next watcher poll"; exit 2
fi
N=$(basename "$latest" | sed 's/checkpoint_epoch_//')
resume="$latest"
echo "resuming from robot ckpt: $latest (epoch $N), 8-GPU phase starts at epoch $((N+1))"

# 3. stop the 7-GPU run
echo "stopping 7-GPU run..."
pkill -f "num_processes 7" 2>/dev/null
sleep 10
pkill -9 -f "num_processes 7" 2>/dev/null
# drain GPUs 1-7
for t in $(seq 1 24); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1-7 | awk '$1+0>1024' | wc -l)
  [ "$busy" -eq 0 ] && { echo "GPUs 1-7 drained (${t}0s)"; break; }
  sleep 10
done
pgrep -f "num_processes 7" >/dev/null && { echo "7-GPU procs still alive - aborting"; exit 4; } || echo "7-GPU run stopped"

# 4. short 8-GPU DDP probe at 416/GPU (20 steps, ~3-5 min incl. ckpt load)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > /tmp/probe8.log
echo "probing 416/GPU on 8-GPU DDP..."
accelerate launch --num_processes 8 pretrain.py \
  --data_dir data --image_size 256 --n_classes 8 \
  --batch_size 416 --n_workers 8 \
  --lr 5e-5 --probe_lr 6e-3 --no_probe \
  --num_epochs 99 --warmup_epochs 1 --last_decay_epoch 99 \
  --start_epoch "$((N + 1))" \
  --max_steps 20 \
  --resume "$resume" \
  --save_dir /tmp/probe8_416 --output_dir /tmp/probe8_out \
  > /tmp/probe8.log 2>&1
probe_rc=$?
if [ $probe_rc -eq 0 ] && ! grep -q "OutOfMemoryError" /tmp/probe8.log; then
  BATCH=416
  echo "416/GPU 8-GPU DDP probe clean (rc=$probe_rc)"
else
  echo "416/GPU probe FAILED rc=$probe_rc oom=$(grep -c OutOfMemoryError /tmp/probe8.log 2>/dev/null) - falling back to 400/GPU"
  tail -5 /tmp/probe8.log
  BATCH=400
fi
echo "8-GPU phase batch size: $BATCH"

# drain any residual memory from the probe before the full launch
for t in $(seq 1 18); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1+0>1024' | wc -l)
  [ "$busy" -eq 0 ] && { echo "all GPUs drained post-probe (${t}0s)"; break; }
  sleep 10
done

# 5. launch the 8-GPU run
cp runs/robot_9ep.log "runs/robot_9ep.log.7gpu_$(date +%s)" 2>/dev/null || true
: > runs/robot_9ep.log
setsid nohup accelerate launch --num_processes 8 pretrain.py \
  --data_dir data --image_size 256 --n_classes 8 \
  --batch_size $BATCH --n_workers 8 \
  --lr 5e-5 --probe_lr 6e-3 --no_probe \
  --num_epochs 9 --warmup_epochs 1 --last_decay_epoch 9 \
  --start_epoch "$((N + 1))" \
  --save_interval 1 --val_samples_max 4000 \
  --resume "$resume" \
  --save_dir models/mae_moe_robot_9ep --output_dir outputs \
  --run_name mae_moe_robot_9ep_8gpu \
  2>&1 | python3 "$(pwd)/runs/tsline.py" >> runs/robot_9ep.log &
newpid=$!
echo $newpid > /tmp/robot_run8.pid
echo "8-GPU run launched, pid $newpid, resume=$resume, start_epoch=$((N+1)), batch=${BATCH}/GPU"

# 6. health check: first loss line OR error, within ~20 min
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
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' '; '; echo
  date -u +%FT%TZ > /tmp/switch_7to8_done
  echo "SWITCH_OK"
else
  echo "HEALTH TIMEOUT: no loss line in 20 min (may still be initializing; check log)"; tail -10 runs/robot_9ep.log
fi
echo "=== switch end $(date -u +%FT%TZ) ==="
} >> "$log" 2>&1
echo "switch rc=$? (log: $log)"
