import math
import os
import time
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F, torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim.lr_scheduler import LambdaLR

from model import Encoder, Decoder, convert_legacy_moe_state_dict
from utils import set_seed, human_readable_numbers as hrn,\
    patchify, unpatchify, denormalize, visualize
from dataloader import get_robot_dataloaders, get_imagenet_dataloaders,\
    load_norm_stats, normalize_batch
from losses import variance_loss_fn, sliced_wasserstein_loss_fn


def _raw_module(model):
    """The underlying module of a torch.compiled model (no-op otherwise)."""
    return getattr(model, "_orig_mod", model)


def _interp_pos_emb(old, new_shape):
    """Bilinearly resample a (1, P_old^2, D) positional-embedding tensor to
    (1, P_new^2, D), where P_new^2 == new_shape[1] (new patch grid)."""
    p_old = int(round(old.shape[1] ** 0.5))
    p_new = int(round(new_shape[1] ** 0.5))
    d = old.shape[2]
    if p_old * p_old != old.shape[1] or p_new * p_new != new_shape[1]:
        raise ValueError(f"pos_emb grid is not a perfect square: "
                         f"{tuple(old.shape)}")
    # (P_old^2, D) -> treat D as channels of a (p_old, p_old) grid.
    grid = old[0].float().reshape(p_old, p_old, d).permute(2, 0, 1)
    grid = torch.nn.functional.interpolate(
        grid.unsqueeze(0), size=(p_new, p_new), mode="bilinear",
        align_corners=False)
    return grid[0].permute(1, 2, 0).reshape(1, p_new * p_new, d)


def _load_state_dict(path):
    """Load a model state dict from safetensors or a torch .bin file."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu")


def _load_into(path, raw, label, interpolate):
    """Load the state dict at ``path`` into raw model ``raw`` (non-strict).

    Shape-mismatched keys are dropped, unless it is the ``pos_emb`` tensor
    and ``interpolate=True``: then it is bilinearly resampled to the
    current patch grid (needed when resuming across image_size, e.g.
    224 -> 256). All other parameters load verbatim. Returns True when
    every key the model expects was loaded with a matching shape (i.e.
    the module was not partially re-initialized).
    """
    sd = convert_legacy_moe_state_dict(_load_state_dict(path))
    cur = raw.state_dict()
    for key in list(sd):
        if key not in cur:
            print(f"[resume] {label}: unexpected key {key} ignored")
            del sd[key]
        elif sd[key].shape != cur[key].shape:
            if interpolate and key == "pos_emb":
                old_shape = tuple(sd[key].shape)
                sd[key] = _interp_pos_emb(sd[key], cur[key].shape)
                print(f"[resume] {label}: interpolated pos_emb "
                      f"{old_shape} -> {tuple(cur[key].shape)}")
            else:
                print(f"[resume] {label}: skipping {key} "
                      f"(shape {tuple(sd[key].shape)} != "
                      f"{tuple(cur[key].shape)})")
                del sd[key]
    res = raw.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if k != "pos_emb"]
    print(f"[resume] {label}: loaded {len(sd)}/{len(cur)} tensors; "
          f"still-missing={missing}")
    return not missing


def _expert_logs(encoder) -> dict[str, float]:
    """Per-expert load / token count / router importance, averaged over the
    encoder's MoE layers. Empty for a dense encoder (--n_experts 1)."""
    mixers = [l.channel_mixer for l in encoder.layers
              if getattr(l.channel_mixer, "last_load", None) is not None]
    if not mixers:
        return {}
    stats = {
        name: torch.stack([getattr(m, f"last_{name}").float() for m in mixers])
        .mean(0).tolist()
        for name in ("load", "num_tokens", "importance")
    }
    logs = {}
    for i in range(len(stats["load"])):
        logs[f"expert_{i}_load"] = stats["load"][i]
        logs[f"expert_{i}_tokens"] = stats["num_tokens"][i]
        logs[f"expert_{i}_importance"] = stats["importance"][i]
    return logs


def save_recon_grid(original, masked, reconstruction, out_path, n=12):
    """Save an Original / Masked-Input / Reconstruction grid image to disk.

    original/masked/reconstruction: [B,3,H,W] float tensors in display
    space (0..1). Each sample occupies one row: [original, masked, recon].
    """
    n = min(n, original.shape[0])
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(6.0, 6.0 * rows))
    if rows == 1:
        axes = axes[None, :]
    titles = ["Original", "Masked Input", "Reconstruction"]
    for i in range(n):
        r, c = divmod(i, 3)
        for j, im in enumerate([original[i], masked[i], reconstruction[i]]):
            ax = axes[r, j]
            ax.imshow(im.permute(1, 2, 0).detach().cpu().numpy())
            if r == 0:
                ax.set_title(titles[j], fontsize=9)
            ax.axis("off")
    for i in range(n, rows * 3):
        r, c = divmod(i, 3)
        axes[r, c].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _resume_from_checkpoint(ckpt_dir, args, encoder, decoder, probe,
                            optimizer, probe_optimizer):
    """Continue training from an accelerate checkpoint directory.

    accelerate's save_state() writes the prepared models split by
    prepare-order index (model.safetensors = encoder, model_1 = decoder,
    model_2 = probe) plus optimizer.bin / optimizer_1.bin. The LR
    scheduler is intentionally NOT restored: a continued run gets a fresh
    schedule for its own (shorter) epoch budget. This must be called
    before accelerator.prepare() -- DDP wrapping preserves parameter
    identity, so the loaded optimizer state stays valid after prepare().

    Optimizer states are only loaded for modules whose parameters all
    matched the checkpoint. This matters for the probe when n_classes
    differs (ImageNet 1000 -> robot 8): the probe is re-initialized and
    its stage-1 momentum buffers would have the wrong shape, so we leave
    the probe optimizer fresh in that case.
    """
    enc_p = os.path.join(ckpt_dir, "model.safetensors")
    dec_p = os.path.join(ckpt_dir, "model_1.safetensors")
    prb_p = os.path.join(ckpt_dir, "model_2.safetensors")
    if not os.path.isfile(enc_p):
        enc_p, dec_p, prb_p = (
            os.path.join(ckpt_dir, n)
            for n in ("model.bin", "model_1.bin", "model_2.bin"))
    print(f"[resume] continuing from {ckpt_dir} "
          f"(image_size={args.image_size})")
    enc_ok = _load_into(enc_p, _raw_module(encoder), "encoder", True)
    dec_ok = _load_into(dec_p, _raw_module(decoder), "decoder", True)
    prb_ok = (_load_into(prb_p, probe, "probe", False)
              if probe is not None else True)
    # The main optimizer is deliberately NOT resumed. Stage-1 ran with a
    # different LR regime (peak 2e-4 vs stage-2 peak 5e-5, fresh warmup) on
    # a different dataset, so ImageNet's AdamW momentum is not useful for the
    # continued run. It is also the source of a dtype crash: the saved
    # exp_avg/exp_avg_sq are FP32 while the resumed bf16 params make the
    # fused GPU Adam assert same-dtype and fail at step 1. Loss continuity
    # comes from the model weights, not the optimizer, so skipping this is
    # correct, not just a workaround. (If a future stage wants momentum,
    # cast the loaded moments to the param dtype before load_state_dict.)
    if enc_ok and dec_ok:
        print("[resume] main optimizer left fresh (stage-2 uses a new LR "
              "schedule on a new dataset; saved state is FP32 and is not "
              "compatible with the bf16 params)")
    else:
        print("[resume] WARNING: encoder/decoder did not fully load; "
              "main optimizer starts fresh (no state loaded)")
    if probe is None:
        print("[resume] probe disabled (--no_probe): probe weights and "
              "probe optimizer not loaded")
    elif prb_ok:
        popt_p = os.path.join(ckpt_dir, "optimizer_1.bin")
        if os.path.isfile(popt_p):
            probe_optimizer.load_state_dict(torch.load(popt_p,
                                                       map_location="cpu"))
            print("[resume] loaded probe optimizer state")
        else:
            print("[resume] WARNING: optimizer_1.bin not found; "
                  "probe optimizer starts fresh")
    else:
        print("[resume] probe was re-initialized (n_classes differs); "
              "probe optimizer left fresh")
    print(f"[resume] checkpoint {ckpt_dir} loaded successfully")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_float32_matmul_precision('medium')

    # Data: robot_jpg root (8 datasets, per-dataset normalization) or an
    # ImageFolder root such as data/imagenet-540k-1k (legacy path).
    if os.path.isdir(os.path.join(args.data_dir, "robot_jpg")):
        train_loader, val_loader, dataset_names = get_robot_dataloaders(
            args.data_dir, args.batch_size, args.n_workers)
        is_robot = True
        if args.n_classes != len(dataset_names):
            print(f"[data] robot data: setting n_classes="
                  f"{len(dataset_names)} (was {args.n_classes})")
            args.n_classes = len(dataset_names)
        stats = load_norm_stats(args.data_dir)
        norm_mean = torch.tensor(
            [stats[n]["mean"] for n in dataset_names], dtype=torch.float32)
        norm_std = torch.tensor(
            [stats[n]["std"] for n in dataset_names], dtype=torch.float32)
        print(f"[data] robot datasets: {dataset_names}")
        print(f"[data] norm stats: mean={norm_mean.tolist()}")
        print(f"[data] norm stats: std={norm_std.tolist()}")
    else:
        train_loader, val_loader, _ = get_imagenet_dataloaders(
            args.data_dir, args.batch_size, args.n_workers)
        is_robot = False
        norm_mean = norm_std = None

    ddp_kwargs = DistributedDataParallelKwargs(
        # Every parameter gets a gradient each step (the grouped-MoE runs
        # all experts' weights through grouped_mm even with 0 tokens), so
        # DDP does not need the extra unused-parameter graph traversal.
        find_unused_parameters=False
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="bf16",
        log_with="wandb" if args.run_name else None
    )

    if args.run_name:
        accelerator.init_trackers(
            project_name="improved-vision-encoders",
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "entity": "statsml-csa-iisc",
                    "name": args.run_name,
                }
            },
        )

    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.compile()
    decoder.compile()
    
    if args.no_probe:
        # Probe CODE is fully retained; --no_probe only blocks its
        # execution. Robot data has no class concept so the probe is
        # skipped there; ImageNet runs (without the flag) are unchanged.
        probe = None
        print("[probe] disabled via --no_probe: not built, not trained")
    else:
        probe = torch.nn.Linear(args.n_embed, args.n_classes)
        torch.nn.init.trunc_normal_(probe.weight, std=0.02)
        torch.nn.init.zeros_(probe.bias)
    
    decay = []
    no_decay = []
    for model in [encoder, decoder]:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

    optimizer = optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())
    if args.no_probe:
        probe_optimizer = None
    else:
        probe_optimizer = optim.AdamW(
            probe.parameters(), lr=args.probe_lr,
            betas=(0.9, 0.95), weight_decay=0.0)

    # steps_per_epoch: per-rank DDP step count. Use ceil (not floor) so the
    # schedule milestones do not fire 1 step early. len(train_loader) is the
    # raw per-rank dataloader length (before DDP wrapping).
    steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    cosine_steps = (args.last_decay_epoch - args.warmup_epochs) * steps_per_epoch
    lr_floor = 1.0 / 20.0  # final LR multiplier = args.lr / 20

    # Single LambdaLR for the whole budget. The old SequentialLR
    # [LinearLR -> CosineAnnealingLR -> ConstantLR] had a bug: the ConstantLR
    # "hold" phase reset LR to the full base value instead of holding at
    # lr/20, because SequentialLR anchors each child scheduler to the
    # original base_lrs (the initial lr), not to the cosine running value.
    # A LambdaLR sidesteps this entirely by computing the multiplier from the
    # global step, matching the warmup and cosine formulas exactly and
    # holding at lr/20 for the remainder.
    def _lr_lambda(n):
        if n < warmup_steps:
            # Linear warmup: 1e-3 -> 1.0 (matches LinearLR start_factor=1e-3)
            return 1e-3 + (1.0 - 1e-3) * (n / warmup_steps if warmup_steps > 0 else 0.0)
        elif n < warmup_steps + cosine_steps:
            # Cosine decay: 1.0 -> 1/20 (matches CosineAnnealingLR eta_min=lr/20)
            t = n - warmup_steps
            return lr_floor + 0.5 * (1.0 - lr_floor) * \
                   (1.0 + math.cos(math.pi * t / cosine_steps))
        else:
            # Hold at lr/20 (the fix: original ConstantLR reset to base_lr)
            return lr_floor

    scheduler = LambdaLR(optimizer, _lr_lambda)

    if args.start_epoch > 1:
        _ff = (args.start_epoch - 1) * steps_per_epoch
        scheduler.last_epoch = _ff
        for pg, new_lr in zip(optimizer.param_groups, scheduler.get_lr()):
            pg["lr"] = new_lr
        accelerator.print(
            f"[start_epoch={args.start_epoch}] scheduler fast-forwarded to step {_ff}, "
            f"LR={scheduler.get_last_lr()[0]:.6g}")

    get_param_count = lambda model: sum(p.numel() for p in model.parameters())
    accelerator.print(f"Train loader: {hrn(len(train_loader.dataset))} samples") # type: ignore
    accelerator.print(f"Val loader: {hrn(len(val_loader.dataset))} samples") # type: ignore
    accelerator.print(f"Encoder params: {hrn(get_param_count(encoder))}")
    accelerator.print(f"Decoder params: {hrn(get_param_count(decoder))}")
    if args.no_probe:
        accelerator.print("Probe params: 0 (disabled via --no_probe)")
    else:
        accelerator.print(f"Probe params: {hrn(get_param_count(probe))}")

    if args.resume:
        _resume_from_checkpoint(
            args.resume, args, encoder, decoder, probe,
            optimizer, probe_optimizer)
        # load_state_dict() inside _resume_from_checkpoint restores each
        # param group's 'lr' from the checkpoint (the stage-1 base LR), which
        # clobbers the fresh schedule. Re-sync the main optimizer LR to the
        # fresh scheduler's value for the current step so batch 1 trains at
        # the correct continued-run LR, not the stale checkpoint LR.
        for pg, new_lr in zip(optimizer.param_groups, scheduler.get_lr()):
            pg["lr"] = new_lr
        accelerator.print(
            f"[resume] main-optimizer LR re-synced to the fresh scheduler "
            f"(step-0 value {optimizer.param_groups[0]['lr']:.6g})")
    if args.no_probe:
        encoder, decoder, optimizer, train_loader, val_loader = \
            accelerator.prepare(encoder, decoder, optimizer,
                                train_loader, val_loader)
        probe, probe_optimizer = None, None
    else:
        encoder, decoder, probe, optimizer, probe_optimizer, train_loader,\
            val_loader = accelerator.prepare(
                encoder, decoder, probe, optimizer,
                probe_optimizer, train_loader, val_loader)

    if is_robot:
        norm_mean_dev = norm_mean.to(accelerator.device)
        norm_std_dev = norm_std.to(accelerator.device)
    else:
        norm_mean_dev = norm_std_dev = None

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        encoder.train()
        decoder.train()
        if not args.no_probe:
            probe.train()
        last_log_time, last_log_batch = time.perf_counter(), 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if is_robot:
                imgs = normalize_batch(imgs, labels,
                                       norm_mean_dev, norm_std_dev)
            patches = patchify(imgs, args.patch_size)
            
            B, N, _ = patches.shape
            noise = torch.rand(B, N, device=accelerator.device)
            ids_shuffle = noise.argsort(dim=1)
            ids_keep = ids_shuffle[:, :N // 4]
            ids_restore = ids_shuffle.argsort(dim=1)
            
            latents, router_loss = encoder(patches, ids_keep)
            reconstructed, _ = decoder(latents, ids_restore)
            
            pred_loss = F.mse_loss(
                reconstructed, patches, reduction='none').mean(-1)
            mask = torch.ones(B, N, device=accelerator.device)
            mask.scatter_(1, ids_keep, 0.)
            pred_loss = (pred_loss * mask).sum() / mask.sum()
            var_loss = variance_loss_fn(latents)
            sw_loss = sliced_wasserstein_loss_fn(latents)
            loss = (
                args.pred_weight * pred_loss\
                + args.var_weight * var_loss\
                + args.sw_weight * sw_loss
            )
            if router_loss is not None:
                loss += args.router_weight * router_loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            if not args.no_probe:
                encoder.eval()
                with torch.no_grad():
                    latents, _ = encoder(patches)
                    features = latents.mean(dim=1).detach()
                encoder.train()
                logits = probe(features)
                probe_loss = F.cross_entropy(logits, labels)
                probe_optimizer.zero_grad()
                accelerator.backward(probe_loss)
                probe_optimizer.step()

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            # Metrics are read back (.item(): a host sync each) and logged
            # only every log_interval steps, not on every step.
            if batch_idx % args.log_interval == 0:
                now = time.perf_counter()
                imgs_per_sec = (
                    (batch_idx - last_log_batch) * B * accelerator.num_processes
                    / (now - last_log_time)) if batch_idx > last_log_batch else 0.0
                last_log_time, last_log_batch = now, batch_idx
                _train_metrics = {
                    "train/epoch": (epoch-1) + (batch_idx + 1) / len(train_loader),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/imgs_per_sec": imgs_per_sec,
                    "train/loss": loss.item(),
                    "train/pred_loss": pred_loss.item(),
                    "train/var_loss": var_loss.item(),
                    "train/sw_loss": sw_loss.item(),
                    "train/router_loss": router_loss.item() if router_loss is not None else 0.0,
                    **_expert_logs(accelerator.unwrap_model(encoder)),
                }
                _probe_str = ""
                if not args.no_probe:
                    preds = logits.argmax(dim=-1)
                    preds, labels = accelerator.gather_for_metrics((preds, labels))
                    acc = (preds == labels).sum().item() / len(labels)
                    _train_metrics["train/probe_loss"] = probe_loss.item()
                    _train_metrics["train/probe_acc"] = acc
                    _probe_str = (f" Probe Loss: {probe_loss.item():.4f}"
                                  f" Probe Acc: {acc:.4f}")
                accelerator.log(_train_metrics, step=global_step)
                accelerator.print(
                    f"Epoch [{epoch}/{args.num_epochs}]"
                    f" Batch [{batch_idx + 1}/{len(train_loader)}]"
                    f" LR: {optimizer.param_groups[0]['lr']:.6f}"
                    f" Img/s: {imgs_per_sec:.0f}"
                    f" Loss: {loss.item():.4f}"
                    f" Pred Loss: {pred_loss.item():.4f}"
                    f" Var Loss: {var_loss.item():.4f}"
                    f" SW Loss: {sw_loss.item():.4f}"
                    f"{_probe_str}"
                    f" Router Loss: {router_loss.item() if router_loss is not None else 0.0:.4f}"
                )
            if args.max_steps and global_step + 1 >= args.max_steps:
                accelerator.print(
                    f"[max_steps] stopping after {args.max_steps} train steps")
                break
        if args.max_steps and global_step + 1 >= args.max_steps:
            break
            
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and epoch % args.save_interval == 0:
            ckpt_path = f"{args.save_dir}/checkpoint_epoch_{epoch}"
            accelerator.save_state(ckpt_path)
            accelerator.print(f"[save] epoch {epoch} checkpoint -> {ckpt_path}")

        if args.max_steps:
            # Smoke-test mode: no validation at epoch ends; training goes on
            # until max_steps is reached.
            accelerator.print(
                f"[max_steps] epoch {epoch} done; skipping validation "
                f"(train loss {loss.item():.4f})")
            continue
        
        encoder.eval()
        decoder.eval()
        if not args.no_probe:
            probe.eval()
        total_loss, total_mse, total_mae, total_psnr = 0.0, 0.0, 0.0, 0.0
        total_pred_loss, total_var_loss, total_sw_loss = 0.0, 0.0, 0.0
        total_correct_1, total_correct_5, total_correct_10 = 0, 0, 0
        with torch.no_grad():
            _max_val_batches = (args.val_samples_max // len(val_loader) + 1) \
                if args.val_samples_max > 0 else None
            _n_val_batches = 0
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                if _max_val_batches is not None and batch_idx >= _max_val_batches:
                    break
                _n_val_batches += 1
                if is_robot:
                    imgs = normalize_batch(imgs, labels,
                                           norm_mean_dev, norm_std_dev)
                patches = patchify(imgs, args.patch_size)
                
                B, N, _ = patches.shape
                noise = torch.rand(B, N, device=accelerator.device)
                ids_shuffle = noise.argsort(dim=1)
                ids_keep = ids_shuffle[:, :N // 4]
                ids_restore = ids_shuffle.argsort(dim=1)
                
                latents, _ = encoder(patches, ids_keep)
                reconstructed, _ = decoder(latents, ids_restore)
                
                pred_loss = F.mse_loss(
                    reconstructed, patches, reduction='none').mean(-1)
                if is_robot:
                    s_mean = norm_mean_dev[labels]
                    s_std = norm_std_dev[labels]
                else:
                    s_mean = s_std = None
                preds = denormalize(unpatchify(
                    reconstructed, args.patch_size, args.image_size, args.in_channels),
                                    mean=s_mean, std=s_std)
                targets = denormalize(unpatchify(
                    patches, args.patch_size, args.image_size, args.in_channels),
                                      mean=s_mean, std=s_std)
                mask = torch.ones(B, N, device=accelerator.device)
                mask.scatter_(1, ids_keep, 0.)
                if batch_idx == 0 and accelerator.is_main_process:
                    masked_patches = patches * (1 - mask.unsqueeze(-1))
                    masked_inputs = denormalize(unpatchify(
                        masked_patches, args.patch_size,
                        args.image_size, args.in_channels),
                        mean=s_mean, std=s_std)
                    fig = visualize(targets, masked_inputs, preds)
                    accelerator.log({"visualization": fig}, step=global_step)
                    # On-disk reconstruction grid: auto-enabled for
                    # --no_probe (robot) runs; opt-in elsewhere.
                    if (args.save_recon or args.no_probe) and args.output_dir:
                        os.makedirs(args.output_dir, exist_ok=True)
                        _gpath = os.path.join(
                            args.output_dir,
                            f"recon_epoch{epoch}_valbatch{batch_idx}.png")
                        save_recon_grid(targets, masked_inputs, preds, _gpath)
                        accelerator.print(
                            f"[recon] epoch {epoch} val-batch {batch_idx}"
                            f" -> {_gpath}")
                pred_loss = (pred_loss * mask).sum() / mask.sum()
                var_loss = variance_loss_fn(latents)
                sw_loss = sliced_wasserstein_loss_fn(latents)
                loss = (
                    args.pred_weight * pred_loss\
                    + args.var_weight * var_loss\
                    + args.sw_weight * sw_loss
                )
                # Router loss not included in validation loss
                mse = F.mse_loss(preds, targets)
                mae = F.l1_loss(preds, targets)
                psnr = -10 * torch.log10(mse)

                if not args.no_probe:
                    latents, _ = encoder(patches)
                    logits = probe(latents.mean(dim=1).detach())
                    logits, labels = accelerator.gather_for_metrics((logits, labels))
                    total_correct_1 +=\
                        (logits.argmax(dim=-1) == labels).sum().item()
                    labels = labels.unsqueeze(-1)
                    total_correct_5 +=\
                        (logits.topk(5, dim=-1).indices == labels).sum().item()
                    total_correct_10 +=\
                        (logits.topk(10, dim=-1).indices == labels).sum().item()

                total_pred_loss += pred_loss.item()
                total_var_loss += var_loss.item()
                total_sw_loss += sw_loss.item()
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                total_psnr += psnr.item()
        # Denominator = number of val batches ACTUALLY processed (capped by
        # --val_samples_max), not the full per-rank len(val_loader). Dividing
        # the capped sum by the full count understates every val metric by
        # len(val_loader)/n_processed (e.g. ~117x when val runs 6 of ~702).
        _n_val = max(_n_val_batches, 1)
        val_pred_loss = total_pred_loss / _n_val
        val_var_loss = total_var_loss / _n_val
        val_sw_loss = total_sw_loss / _n_val
        val_loss = total_loss / _n_val
        val_mse = total_mse / _n_val
        val_mae = total_mae / _n_val
        val_psnr = total_psnr / _n_val
        _n_val_ds = len(val_loader.dataset)
        correct_1 = total_correct_1 / _n_val_ds if not args.no_probe else 0.0
        correct_5 = total_correct_5 / _n_val_ds if not args.no_probe else 0.0
        correct_10 = total_correct_10 / _n_val_ds if not args.no_probe else 0.0
        _val_metrics = {
            "val/pred_loss": val_pred_loss,
            "val/var_loss": val_var_loss,
            "val/sw_loss": val_sw_loss,
            "val/loss": val_loss,
            "val/mse": val_mse,
            "val/rmse": math.sqrt(val_mse),
            "val/mae": val_mae,
            "val/psnr": val_psnr
        }
        if not args.no_probe:
            _val_metrics.update({
                "val/acc@1": correct_1,
                "val/acc@5": correct_5,
                "val/acc@10": correct_10
            })
        accelerator.log(_val_metrics, step=global_step)
        
        accelerator.print(
            f"Epoch [{epoch}/{args.num_epochs}]"
            f" Average Val Loss: {val_loss:.4f}"
            f" Average Val Pred Loss: {val_pred_loss:.4f}"
            f" Average Val Var Loss: {val_var_loss:.4f}"
            f" Average Val SW Loss: {val_sw_loss:.4f}"
            f" Average Val MSE: {val_mse:.6f}"
            f" Average Val RMSE: {math.sqrt(val_mse):.4f}"
            f" Average Val MAE: {val_mae:.4f}"
            f" Average Val PSNR: {val_psnr:.2f} dB")
        if not args.no_probe:
            accelerator.print(
                f" Average Val Acc@1: {correct_1:.4f}"
                f" Average Val Acc@5: {correct_5:.4f}"
                f" Average Val Acc@10: {correct_10:.4f}")
    # Final checkpoint: persist the finished model after the last epoch's
    # validation. With --save_interval 1, the per-epoch saves at each epoch
    # START cover epochs 1..N-1; this final save covers the end of epoch N.
    # Gated off in smoke-test (--max_steps) mode so a short run does not write
    # a final checkpoint into the model save dir.
    if accelerator.is_main_process and not args.max_steps:
        final_path = f"{args.save_dir}/checkpoint_final"
        accelerator.save_state(final_path)
        accelerator.print(f"[save] final checkpoint -> {final_path}")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_embed', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--n_experts', type=int, default=8)
    parser.add_argument('--n_active_experts', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--token_mixer', type=str, default='MHA')
    parser.add_argument('--activation', type=str, default='SwiGLU')
    parser.add_argument('--decoder_n_embed', type=int, default=256)
    parser.add_argument('--decoder_n_heads', type=int, default=4)
    parser.add_argument('--decoder_n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--n_classes', type=int, default=8)
    parser.add_argument('--no_probe', action='store_true',
                        help='optionally block probe execution (robot data '
                             'has no class concept). Probe code is retained; '
                             'build/train/eval are skipped. Default: probe runs '
                             '(ImageNet behavior unchanged).')
    parser.add_argument('--save_recon', action='store_true',
                        help='save a reconstruction image grid to output_dir '
                             'at each validation start (auto-enabled when '
                             '--no_probe is set)')
    parser.add_argument('--val_samples_max', type=int, default=0,
                        help='cap validation to this many samples '
                             '(0 = full val set)')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Continue training from this 1-indexed epoch; skip 1..start_epoch-1 and fast-forward the LR scheduler to the matching step.')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--last_decay_epoch', type=int, default=480)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_steps', type=int, default=0,
                        help='stop after this many train steps (0 = no '
                             'limit, e.g. 50 for a smoke test)')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--probe_lr', type=float, default=6e-3)
    parser.add_argument('--pred_weight', type=float, default=1.0)
    parser.add_argument('--var_weight', type=float, default=25.0)
    parser.add_argument('--sw_weight', type=float, default=1.0)
    parser.add_argument('--router_weight', type=float, default=0.01)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--resume', type=str, default='',
                        help='accelerate checkpoint dir to continue'
                             ' training from (encoder/decoder/probe '
                             '+ optimizers); the scheduler is rebuilt '
                             'fresh for the new epoch budget')
    args = parser.parse_args()
    main(args)
