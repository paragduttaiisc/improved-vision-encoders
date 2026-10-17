# Launcher note — mae_moe_imagenet_100ep (ImageNet, 100 epochs, 8-GPU DDP)

Status at write time: RUNNING, detached (survives the launching session).

- Process: `bash /tmp/fullrun.sh` PID **1269207**, reparented to init (setsid).
  - accelerate launcher PID 1269211; 8 DDP worker PIDs 1269310–1269318 on GPUs 0–7.
- Log (authoritative): **/raid/home/paragdutta/Codes/improved-vision-encoders/runs/imagenet_100ep.log**
  - Every output line is prefixed with epoch-seconds (e.g. `1792249698.834 Epoch [1/100] Batch [1/123] ...`).
  - Health check: `tail -5 runs/imagenet_100ep.log` should show increasing step numbers; `nvidia-smi` should show ~38 GB used + high util on all 8 GPUs.
- wandb: https://wandb.ai/statsml-csa-iisc/improved-vision-encoders/runs/hgyj91xq (entity statsml-csa-iisc, project improved-vision-encoders).
- Checkpoints: models/mae_moe_imagenet_100ep/checkpoint every 10 epochs.
- Expected completion: ~2026-10-17 05:00–05:30 UTC+5:30 (~3.2–3.5 h from 01:34 launch). On exit the log gets a final `FULL_RUN_EXIT=0` line.
- If process 1269207 is gone and the log lacks `FULL_RUN_EXIT=0`, the run crashed — last ~50 log lines + `dmesg` on Mriya have the cause.
