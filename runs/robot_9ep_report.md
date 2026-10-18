# Robot 9-epoch continue-pretrain — launch report (2026-10-18 00:46 UTC / 06:16 IST)

## Status: RUNNING (detached, 8/8 A100s)

- PID (fullrun bash): 129305 (runs/robot_9ep.pid)
- Log: runs/robot_9ep.log
- wandb: https://wandb.ai/statsml-csa-iisc/improved-vision-encoders/runs/g74q796d (run mae_moe_robot_9ep, entity statsml-csa-iisc, user parag1604)
- Git: branch jpeg-dataloader, commit 9f36c34
- Checkpoint dir: models/mae_moe_robot_9ep/
- Flags: --num_epochs 9 --warmup_epochs 1 --last_decay_epoch 9 --lr 5e-5 --probe_lr 6e-3 --save_interval 1 --batch_size 64 (per-GPU = 512 global) --image_size 256 --resume models/mae_moe_imagenet_100ep/checkpoint

## 1. LR-scheduler fix (commit 2d61e30)
- Bug: ConstantLR hold phase inside SequentialLR reset LR to the full base value at milestone 2
  (stage-1 log showed 1.0e-5 -> 2.0e-4 jump at epoch 90 batch 41, stayed there).
- Fix: single LambdaLR — linear warmup (5e-8 -> 5e-5 over epoch 1), then cosine decay to lr/20
  (2.5e-6) at last_decay_epoch; the "hold" is now implicit: after the decay completes the lambda
  returns eta_min for the remaining epochs. Also fixed steps_per_epoch ceil (was floor,
  firing milestones 1 step early).
- CPU table (runs/lr_schedule_check.txt, torch 2.14.0): stage-1 config holds 1e-5 (lr/20) after
  milestone 2, monotonic, final == lr/20; stage-2 config (9ep, sbe=4498, lr=5e-5): no jump at
  milestone 2, final == 2.5e-6 = lr/20; warmup matches LinearLR to 2e-15, cosine matches
  CosineAnnealingLR to 1e-14. PASS.

## 2. Resume checkpoint (unchanged choice, re-verified)
- models/mae_moe_imagenet_100ep/checkpoint = epoch-90-START save = after 89 fully clean epochs,
  written BEFORE any 2.0e-4 (bug) batch. Cleanest logged val neighbor: 0.3735 / Acc@1 0.3040 @ ep 80.
- pos_emb bilinear 14x14 -> 16x16 (encoder + decoder); probe re-initialized to 8 classes;
  fresh probe optimizer (commit 8ad9d0c).
- Main-optimizer state is NOT loaded (commit 5c273e8): checkpoint optimizer.bin is FP32
  exp_avg/exp_avg_sq vs bf16 params -> fused-Adam dtype assert at step 1. Fresh AdamW is the
  correct choice anyway (new dataset + fresh warmup at 5e-5); loss continuity comes from weights.

## 3. Smoke (PASS)
- 12 steps, 8-GPU, 256px, 64/GPU. SMOKE_EXIT=0 after RC-capture fix (commit 9f36c34 — the
  launcher referenced an unbound RC under set -u; training itself had finished all 12 steps).
- First total loss 0.4913 (Pred 0.2663, fresh probe head 2.1475 at 5% acc) — continuous with
  stage-1 (Pred ~0.31-0.37), NOT 4.7. Probe head converges fast (0.06 at step 12).
- Peak VRAM 16.4 GB/GPU (max of 8 GPUs over the 1 Hz nvidia-smi sampler). No OOM.
- Batch bug (commit deefd6a): --batch_size is PER-GPU in this codebase; 64/GPU = 512 global
  matches stage-1 exactly, so the 5e-5 LR calibration holds. 96/GPU was probed to fit earlier;
  64 chosen for stage-1 regime parity.

## 4. First losses (full run)
- Epoch [1/9] Batch 1: Loss 0.4913, Pred 0.2663, Probe 2.1475 (acc 5.3%) — identical to smoke
  (same checkpoint, deterministic init). By batch 241+ total loss ~0.39-0.42, probe ~0.0001.
- LR display confirms warmup: 1e-6 at batch 731 (731/35978 * 5e-5 = 1.0e-6). Log uses .6f, so
  warmup values print as 0.000000-0.000005 (see wandb for the curve).

## 5. Checkpoint semantics (save_interval 1) — VERIFIED against pretrain.py
- Saves at the END of each epoch's TRAIN loop (pretrain.py L388-390: after the last batch,
  `wait_for_everyone`, then save — i.e. end-of-training for that epoch, BEFORE that epoch's
  validation). So checkpoint_epoch_N = end-of-epoch-N state, for N = 1..8.
- The end-of-epoch-9 state IS also saved as checkpoint_epoch_9 (end of epoch-9 training);
  checkpoint_final (after the last epoch's val, pretrain.py L519-520) persists the finished
  model — same weights, written after val so the final val numbers are logged first.
- Net: a checkpoint after every epoch (end-of-epoch-1 .. end-of-epoch-9 + checkpoint_final;
  epoch_9 and final are the same weights) = 10 saves, ~5.5 GB each ~= 55 GB;
  /raid had 8.8 TB free at launch.

## 6. Dataloader throughput (measured, steady-state)
- Window: 363.4 s, batches 731 -> 1821 of epoch 1 (well past startup).
- 1090 steps / 363.4 s = 2.999 steps/s.

| metric                 | stage-1 (ImageNet, 224px) | robot 9ep (256px) | ratio  |
|------------------------|---------------------------|-------------------|--------|
| steps/s (global)       | 1.1623                    | 2.999             | 2.58x  |
| global samples/s       | 595                       | 1536              | 2.58x  |
| per-GPU samples/s      | 74.4                      | 192               | 2.58x  |

- Requirement check: 1536 samples/s = 258% of stage-1's 595 -> passes the 85% floor with wide
  margin; no worker/pin_memory investigation needed.
- Dataloader share: realized 1536 samples/s is below the dataloader-only ceiling (~8,800
  samples/s @ 512/8 workers, measured in t_ed6e219c), so the dataloader is still likely the
  dominant share of step time (robot JPEGs decode faster than ImageNet's, hence 2.58x stage-1).
- Batch size unchanged from stage-1 (512 global) — no OOM at 256px (peak ~16.3 GB/GPU, ~40% of
  the 40 GB A100s).

## 7. Wall clock (MEASURED, replaces earlier estimates)
- Iters/epoch = 35,978 (= ceil(18.42M samples / 8 GPUs / 64 per-GPU batch)). NOTE: this corrects
  both the task body's "44,978 iters/epoch" and my own READY_TO_LAUNCH "40,482 total steps"
  (I divided by 8 a second time). True budget: 9 x 35,978 = 323,802 steps.
- At measured 2.999 steps/s: 323,802 training steps ~= 108,000 s = ~30.0 h of pure training.
  Per-epoch validation, scaled from stage-1's measured val+save (~5 s at 10K samples ->
  ~17 min at the robot's 2.04M val frames): ~2.5 h total. Grand total ~32.5 h. Launched
  2026-10-18 00:46 UTC -> ETA ~2026-10-19 09:15 UTC (~14:45 IST 10-19); re-estimate from the
  log after the first epoch completes (its end-of-epoch save + val line fixes the real cadence).
- Per-epoch saves land at the END of each epoch's training (~3.5 h apart, training + val each
  epoch): checkpoint_epoch_1 ~04:25 UTC 10-18, epoch-2 ~07:55, epoch-3 ~11:25, epoch-4 ~14:55,
  epoch-5 ~18:25, epoch-6 ~21:55, epoch-7 ~01:25 (10-19), epoch-8 ~04:55, epoch-9 ~08:25,
  plus checkpoint_final ~08:30 UTC 10-19 after the last epoch's val.
- Monitoring: tail runs/robot_9ep.log; watchdog signal = ROBOT_9EP_EXIT=0 at file tail +
  checkpoint_epoch_1..9 + checkpoint_final all present in models/mae_moe_robot_9ep/.

## 8. Commits this task
- 8ad9d0c resume plumbing (pos_emb interp, fresh probe + probe optimizer)
- 2d61e30 LR-scheduler LambdaLR fix + ceil fix + per-epoch/final save paths + LR table + launchers
- 5c273e8 resume: skip main-optimizer state (FP32/bf16 dtype crash)
- deefd6a + a461c48 launcher batch-size fix (64/GPU = 512 global)
- 9f36c34 smoke launcher: capture accelerate RC via PIPESTATUS
