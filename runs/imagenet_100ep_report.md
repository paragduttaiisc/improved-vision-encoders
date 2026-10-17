# ImageNet 100-epoch MAE pretrain — 8-GPU DDP launch report

Task: t_eb52c322 · Host: Mriya (10.192.30.30) · Date: 2026-10-17
Repo: /raid/home/paragdutta/Codes/improved-vision-encoders @ branch `jpeg-dataloader` (HEAD 21f12b4)
Model: Encoder ~482M params, 12 layers, n_embed 768, `--token_mixer MHA` (attention), MoE MLP default (`--n_experts 8 --n_active_experts 2 --activation SwiGLU`), bf16 mixed precision.

## Launch command (verbatim, full run)
```bash
cd /raid/home/paragdutta/Codes/improved-vision-encoders
source /raid/home/paragdutta/miniconda3/etc/profile.d/conda.sh && conda activate dl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
setsid nohup accelerate launch --num_processes 8 pretrain.py \
  --data_dir data/imagenet-540k-1k --image_size 224 --n_classes 1000 \
  --batch_size 512 --n_workers 8 --num_epochs 100 \
  --last_decay_epoch 90 --save_interval 10 \
  --save_dir models/mae_moe_imagenet_100ep --output_dir outputs \
  --run_name mae_moe_imagenet_100ep \
  > runs/imagenet_100ep.log 2>&1 &
```
- Launcher type: `accelerate launch --num_processes 8` (accelerate 1.15.0, Accelerator with `DistributedDataParallelKwargs(find_unused_parameters=True)`, bf16), wrapped in `setsid nohup` so it is detached from the launch session.
- Smoke run used the same launcher with `--num_epochs 1 --max_steps 12 --run_name '' --log_interval 1` (wandb off). Smoke passed at global batch 512 → **no batch reduction needed**.

## Batch / steps
- Global batch used: **512** (`--batch_size 512` is the per-GPU batch in this codebase; `accelerate.prepare` shards the dataloader → 64 samples/GPU/step × 8 GPUs = 512 global).
- Steps/epoch: **123** (500k train samples, logged per step in log).

## Measured metrics
- Per-GPU VRAM peak: smoke 38,140 MiB (GPU0, max over 8); full-run live peak 38,898 MiB (GPU0), others 38,362–38,416 MiB — headroom ~1.1–1.6 GB on 40 GB cards.
- Steady-state throughput (full run, log timestamps, steps 450–646 window, after compile warmup): **1.162 steps/s = 595 samples/s** (168.6 s / 196 logged-step interval, extrapolated; log lines every 10 steps).
- Steps/epoch = 123 → pure-train epoch time ≈ 105.8 s; measured epoch period incl. per-epoch validation ≈ 95.9 s between consecutive "Epoch Batch [1/123]" log lines (first two epochs; val loop ~15 s/epoch, non-blocking for train steps).
- Projected wall clock for 100 epochs: **~3.2–3.5 h** (100 × 123 steps / 1.162 steps/s ≈ 2.93 h training + ~0.25–0.4 h cumulative validation). ETA ≈ 2026-10-17 05:00–05:30 UTC+5:30 (started 01:34 IST).

## First logged loss line (full run)
```
Epoch [1/100] Batch [1/123] LR: 0.000000 Loss: 4.7622 Pred Loss: 1.5033 Var Loss: 0.1278 SW Loss: 0.0628 Probe Loss: 7.0051 Probe Acc: 0.0000 Router Loss: 0.0030
```
Loss trend healthy: 4.76 → 1.24 by epoch 3; first val loss 1.3060. No OOM/tracebacks in log.

## wandb
- Run: https://wandb.ai/statsml-csa-iisc/improved-vision-encoders/runs/hgyj91xq
- State at verification: `running` (checked via wandb API); project https://wandb.ai/statsml-csa-iisc/improved-vision-encoders

## Detached full-run process
- PID (launcher script `bash /tmp/fullrun.sh`): **1269207** (reparented to init via setsid; accelerate launcher PID 1269211, 8 worker PIDs 1269310–1269318).
- Log path: **/raid/home/paragdutta/Codes/improved-vision-encoders/runs/imagenet_100ep.log** (each line prefixed with epoch seconds, e.g. `1792249698.834 Epoch ...`).
- Verified at launch + ~6 min and ~12.5 min: 8/8 GPUs at 37.5–38.9 GB / 90–100% util, steps progressing (839 → 922 in the check window), no crashes.
- Checkpoints every 10 epochs → models/mae_moe_imagenet_100ep/checkpoint (accelerate.save_state); wandb syncs continuously.
