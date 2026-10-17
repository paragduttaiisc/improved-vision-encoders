# Report — mae_moe_robot_9ep (continue-pretrain, task t_47c3f6d8)

Status at write time: **READY_TO_LAUNCH — awaiting Mriya reboot to free all
8 GPUs.** All GPU-free work is done and committed. Smoke + full launch +
dataloader-throughput measurement happen on unblock.

## 1. LR-scheduler fix (CPU-verified)

**Bug.** Stage-1 was launched with
`--num_epochs 100 --warmup_epochs 10 --last_decay_epoch 90`. The hold phase
was `SequentialLR([LinearLR, CosineAnnealingLR, ConstantLR(factor=1.0)])`.
Reproduced on the server's exact torch build (2.14.0+cu130, CPU, dummy
optimizer): at the second milestone the `ConstantLR` "hold" child reset LR
from `lr/20 = 1e-5` back to the **full base `2e-4`** and kept it there
through the end of the run. Root cause: `SequentialLR` anchors every child
scheduler to the optimizer's original `base_lrs` (the initial `lr`), so the
"constant" phase returned `1.0 × base_lr` instead of holding the cosine
end value. `ConstantDecayLR` (the obvious replacement) is **not present** in
this torch build (only `ConstantLR`), so it was not an option.

**Fix.** Replaced the whole `SequentialLR` with a single `LambdaLR` over the
full step budget. The lambda computes the multiplier from the global step:
- `n < warmup_steps`: `1e-3 + (1 - 1e-3)·(n/warmup_steps)` (linear warmup,
  bit-identical to `LinearLR(start_factor=1e-3)`),
- `warmup ≤ n < warmup+cosine`: `lr/20 + 0.5·(1 - lr/20)·(1 + cos(π·t/cosine))`
  (cosine to `lr/20`, bit-identical to `CosineAnnealingLR(eta_min=lr/20)`),
- `n ≥ warmup+cosine`: hold at `lr/20` (the fix).

This sidesteps the `SequentialLR`/`ConstantLR` `base_lrs` pitfall entirely.

**Secondary fix.** `steps_per_epoch` used `len(train_loader) // nproc`
(floor), which undercounted by 1 (ImageNet: 122 vs true 123; robot: 4497 vs
true 4498), firing every milestone one step early. Now `math.ceil(...)`.

**Resume interaction (found during CPU verification).**
`_resume_from_checkpoint` calls `optimizer.load_state_dict(...)`, which
restores each param group's `lr` from the checkpoint (the stage-1 base
`2e-4`). Left alone, batch 1 would have trained at `2e-4` and the fresh
schedule only taken over from batch 2. Added a re-sync right after the
resume block: `for pg, new_lr in zip(optimizer.param_groups,
scheduler.get_lr()): pg["lr"] = new_lr`. Verified against the live
checkpoint: main-optimizer LR goes `2e-4 → 5e-8` (the stage-2 warmup start
= `5e-5 × 1e-3`), so batch 1 trains at the correct continued-run LR.

**Verification** (no GPU, no data) — `runs/lr_schedule_check.txt`:
- Stage-1 config (100ep, warmup 10, decay to ep90, sbe=123, lr=2e-4):
  monotonic non-increasing after warmup = True, no upward jump at
  milestone 2 = True, final LR = 1e-05 = lr/20 = True. Warmup region
  matches a real `LinearLR` to 2e-15; cosine region matches a real
  `CosineAnnealingLR` to 1e-14. PASS.
- Stage-2 config (9ep, warmup 1, decay to ep9, sbe=4498, lr=5e-5):
  same three checks = True/True/True, final LR = 2.5e-06 = lr/20.
  Warmup/cosine cross-checks 1e-14. PASS.

Commits on `jpeg-dataloader`: `8ad9d0c` (resume plumbing, prior task) +
`2d61e30` (this fix + per-epoch/final checkpoints + launchers + LR table).

## 2. Checkpoint chosen + why

`models/mae_moe_imagenet_100ep/checkpoint` — written at the **start of
epoch 90**, i.e. after 89 clean epochs and **before** any `2e-4` batch
(the LR jump first affected epoch 90, batch 41). This is the correct,
clean resume point. Nearest clean logged val: epoch 89 loss 0.3736 /
Acc@1 0.3095 (epoch-90 val 0.3841 was measured after partial contamination
and is not used). The `--resume` path (commit `8ad9d0c`) bilinearly
interpolates `pos_emb` 14×14 → 16×16 for encoder+decoder, re-initialises
the 1000→8-class probe, and leaves the probe optimizer fresh.

## 3. Save semantics (per-epoch requirement)

- `--save_interval 1` → a `save_state` after **each epoch's training**
  (before that epoch's val). The main process writes
  `models/mae_moe_robot_9ep/checkpoint_epoch_{1..9}` (one ~5.5 GB dir per
  epoch — 10 distinct dirs total with the final; ~55 GB, well within
  /raid's 8.9 TB free).
- `checkpoint_final` is written after the last epoch's validation, so the
  finished end-of-epoch-9 state is persisted (the per-epoch scheme alone
  would not cover epoch 9's end).
- Both saves are main-process only; the final save is gated off in
  `--max_steps` (smoke) mode.
- The previous single rolling `{save_dir}/checkpoint` path is gone — with
  it, `save_interval 1` would have kept only the last epoch's checkpoint.

## 4. Dataloader speed — reference vs plan

Stage-1 measured steady state: **1.1623 steps/s @ global batch 512 =
4761 global samples/s = 595 samples/s per GPU** (dataloader is the
bottleneck; dataloader-only ceiling ~8,800 samples/s @ batch 512 / 8
workers, Howard's dataloader report). Robot frames are 256×256 (vs 224) so
decode is expected to be somewhat slower.

**Plan on unblock:** run the smoke, then the full run; from
`runs/robot_9ep.log` (epoch-seconds-prefixed via `runs/tsline.py`) read a
~5-min steady-state window and compute (a) global samples/s, (b) implied
per-GPU samples/s, (c) dataloader share of step time. Threshold: if
< 85% of stage-1's 4761 global samples/s (~4046), investigate
`num_workers` / `pin_memory` before leaving the run running. Global batch
stays 512 (per-GPU) unless OOM at 256px, in which case fall back to 384
per-GPU and say so in the report.

## 5. Dataset size + real wall clock (CORRECTION to task body)

The task body's "~44,978 iters/epoch, ~404,802 steps, ~96 h" assumed the
robot training set was ~184 M frames. Measuring the live loader
(`RobotJpgDataset.__len__` over the 8 robot datasets, 90/10 split, seed
2026): **train_set = 18,420,724 frames**, val_set = 2,043,837.
Per-rank raw batches @ batch 512 = 35,978; DDP (8 procs, DistributedSampler
pads to a multiple of 8) → **4498 steps/epoch** (ceil; old floor gave 4497).
Total budget = 9 × 4498 = **40,482 steps** (not 404,802).

At stage-1's 1.1623 steps/s that projects to **~9.7 h**; if the 256px robot
frames are ~10–20% slower, ~10–12 h. I will report the **measured** wall
clock from the log, not this estimate.

## 6. Launch protocol (post-reboot)

All 8 GPUs must be free. The launchers now live in the repo (survive a
`/tmp` wipe on reboot): `runs/smoke_robot.sh` and `runs/fullrun_robot.sh`
(+ `runs/tsline.py`). Exact commands:

```
# smoke (12 steps, wandb off, ~3-4 min)
cd /raid/home/paragdutta/Codes/improved-vision-encoders
bash runs/smoke_robot.sh          # -> runs/robot_smoke.log, VRAM -> /tmp/smoke_robot_vram.csv

# full (9 ep, wandb on, detached ~10-12 h)
cd /raid/home/paragdutta/Codes/improved-vision-encoders
setsid nohup bash runs/fullrun_robot.sh >/dev/null 2>&1 &
echo $! > runs/robot_9ep.pid
```

Smoke PASS = first total loss ~0.35–0.45 (continuous with stage-1, NOT
~4.7), no OOM at 256px/batch 512, peak VRAM recorded, exit 0. Full-run
wandb: entity `statsml-csa-iisc`, project `improved-vision-encoders`,
run_name `mae_moe_robot_9ep` → https://wandb.ai/statsml-csa-iisc/improved-vision-encoders

## 7. Open / pending (on unblock)
- Smoke result (first loss, peak VRAM, steps/s) — fill in below.
- Full-launch PID + wandb URL + measured wall clock.
- Dataloader-throughput table vs stage-1 reference.
- Per-epoch save confirmation (checkpoint_epoch_1..9 + checkpoint_final).

## 8. Launch-time facts (filled after unblock)
- Smoke first total loss:
- Peak VRAM (smoke):
- Full-run PID:
- Full-run wandb URL:
- Steady-state steps/s + global samples/s (measured window):
- Per-GPU samples/s:
- Dataloader share of step time:
- Projected / measured wall clock:
