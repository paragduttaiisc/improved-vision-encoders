# Handoff note — mae_moe_robot_9ep (continue-pretrain, task t_47c3f6d8)

Status at write time: **BLOCKED on stage-1 (t_eb52c322) finishing.** All 8
GPUs were occupied by the ImageNet 100-ep run (epoch 40/100 at 16:18 IST).
Resume plumbing is DONE, CPU-verified, and committed. Everything below is
ready to execute the moment the GPUs are free.

## What is already done
- `--resume <ckpt_dir>` flag added to `pretrain.py`, commit `8ad9d0c` on
  branch `jpeg-dataloader`.
  - Loads encoder/decoder/probe from `model.safetensors` / `model_1` /
    `model_2`, and both optimizer states from `optimizer.bin` /
    `optimizer_1.bin`.
  - **pos_emb bilinear interpolation 224→256** (196→256 patches) for
    encoder and decoder — this is why the checkpoint does NOT load
    "either way" as the task body assumed: `pos_emb` is a plain
    `nn.Parameter` of shape `(1, num_patches, dim)`.
  - **Probe re-init** (1000 → 8 classes; both weight and bias skip with a
    printed warning, as intended). When the probe is re-initialized, the
    **probe optimizer is also left fresh** — stage-1's momentum buffers
    are `(1000, 768)`/`(1000,)` and crash the first AdamW step against
    the `(8, 768)` probe (verified by executing a step).
  - Scheduler is rebuilt fresh for the 9-epoch budget (stage-1 never saved
    scheduler state — it wasn't in `accelerator.prepare`).
- CPU verification vs the live stage-1 checkpoint (script output in
  kanban task t_47c3f6d8):
  - encoder: 364/364 tensors, 0 missing; decoder: 49/49, 0 missing
  - `patch_emb.weight` load diff = 0.0 (bit-exact)
  - optimizer state loaded: 413 param entries, running LR 1.03e-04
    (mid-decay, as expected at epoch ~40)
- Launchers staged on the server:
  - `/tmp/smoke_robot.sh` → `runs/robot_smoke.log`
  - `/tmp/fullrun_robot.sh` → `runs/robot_9ep.log`

## image_size decision: **256** (native robot resolution)
The robot dataloader yields fixed 256×256 uint8 frames (`RobotJpgDataset`,
no `image_size` parameter, no resize transform). Running at 224 would
resize 256→224 with no benefit beyond checkpoint compatibility, and the
checkpoint now loads at 256 anyway via pos_emb interpolation. 256 gives
256 patches (16×16) vs 196 (14×14) — sequence-length change is fine for
MHA. No assertion in the code path blocks 256 (verified by reading
model.py: only `image_size % patch_size == 0` is asserted).

## Exact next steps (when GPUs are free)
1. **Confirm stage-1 done:** `bash /tmp/imagenet_probe.sh` prints
   `ALL_DONE`, log tail shows `Epoch [100/100] ... Average Val Loss`,
   and `models/mae_moe_imagenet_100ep/checkpoint` files are newer than
   ~10 min. Use the FINAL epoch-100 checkpoint (refresh the checkpoint
   dir if stage-1 saved it after my CPU test — the loader reads whatever
   is on disk at run time, so no re-test needed unless files are missing).
2. **Smoke (wandb off, ~3–4 min):** `setsid nohup bash /tmp/smoke_robot.sh &`
   Wait for `SMOKE_EXIT=0` in `runs/robot_smoke.log`. Then verify:
   - First train loss ≈ stage-1 end-of-training level (~0.36–0.40 total
     loss; stage-1 was at 0.34–0.36 in epochs 39–40), NOT 4.7.
     Note: robot data is harder than ImageNet, so a few steps of slightly
     higher loss on robot val-style data is normal; the point is the
     reconstruction (Pred Loss) sits near stage-1's ~0.33–0.35, not 1.5.
   - No OOM at 512 global batch / 256px. Measure steps/s from the
     timestamped lines and note peak VRAM (expect ~40–44 GB/GPU at 256;
     A100s have 40 GB! If OOM, fall back to `--batch_size 384` global and
     re-smoke).
   - steps/epoch at 512 global = 18,420,724 / 512 / 8 ≈ 4497.
3. **Full run (detached, wandb on):** `setsid nohup bash /tmp/fullrun_robot.sh &`
   Record the setsid PID. ~40 s after launch: `nvidia-smi` should show 8
   GPUs at ~40 GB and the log should show `Epoch [1/9] Batch [1/4497]`.
4. **Report** `runs/robot_9ep_report.md`: launch cmd, image_size 256 +
   rationale (above), 512 global / 4497 steps-per-epoch, measured steps/s
   + projected wall clock, checkpoint path
   `models/mae_moe_imagenet_100ep/checkpoint` + loss-continuity number,
   wandb URL, PID + log path.
5. `kanban_complete` with the 3-line summary.

## VRAM note
At 256px, patches go from 196→256 per sample (+31% sequence length) and
the batch is identical, so peak VRAM should scale roughly by 1.2–1.3×.
Stage-1 peaked at ~38.9 GB/GPU on the A100-40G box. Projected peak at
256: ~40–44 GB. **This may exceed 40 GB** — if the smoke OOMs, retry with
`--batch_size 384` (per-GPU global semantics unchanged) and document the
reduced batch in the report.

## Provenance
- Stage-1 launch cmd: `accelerate launch --num_processes 8 pretrain.py
  --data_dir data/imagenet-540k-1k --image_size 224 --n_classes 1000
  --batch_size 512 --n_workers 8 --num_epochs 100 --last_decay_epoch 90
  --save_interval 10 --save_dir models/mae_moe_imagenet_100ep
  --output_dir outputs --run_name mae_moe_imagenet_100ep`
- Stage-1 wandb: https://wandb.ai/statsml-csa-iisc/improved-vision-encoders/runs/hgyj91xq
- Robot data: `data/robot_jpg` (18.42M train / 2.04M val, 8 datasets,
  norm_stats in `data/norm_stats.json`, normalization on-GPU).
- `--batch_size 512` is GLOBAL in this codebase (500k/512/8 ≈ 123
  steps/epoch matches stage-1's logged 123; robot: 18.42M/512/8 ≈ 4497).
