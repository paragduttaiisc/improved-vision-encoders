import math
import argparse
import torch, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from accelerate import Accelerator

from model import Encoder, Decoder
from utils import set_seed, human_readable_numbers as hrn,\
    patchify, unpatchify, denormalize, visualize
from dataloader import get_dataloaders
from losses import variance_loss_fn, sliced_wasserstein_loss_fn


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_float32_matmul_precision('medium')
    accelerator = Accelerator(
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

    train_loader, val_loader, _ =\
        get_dataloaders(args.data_dir, args.batch_size, args.n_workers)

    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.compile()
    decoder.compile()
    
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
    probe_optimizer = optim.AdamW(
        probe.parameters(), lr=args.probe_lr,
        betas=(0.9, 0.95), weight_decay=0.0)

    total_steps = args.num_epochs * len(train_loader) // torch.cuda.device_count()

    warmup_steps = len(train_loader) // torch.cuda.device_count() * 3
    cosine_steps = total_steps - warmup_steps
    warmup = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=args.lr / 10)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    get_param_count = lambda model: sum(p.numel() for p in model.parameters())
    accelerator.print(f"Train loader: {hrn(len(train_loader.dataset))} samples") # type: ignore
    accelerator.print(f"Val loader: {hrn(len(val_loader.dataset))} samples") # type: ignore
    accelerator.print(f"Encoder params: {hrn(get_param_count(encoder))}")
    accelerator.print(f"Decoder params: {hrn(get_param_count(decoder))}")
    accelerator.print(f"Probe params: {hrn(get_param_count(probe))}")

    encoder, decoder, probe, optimizer, probe_optimizer, train_loader,\
        val_loader = accelerator.prepare(
            encoder, decoder, probe, optimizer,
            probe_optimizer, train_loader, val_loader)

    for epoch in range(1, args.num_epochs + 1):
        encoder.train()
        decoder.train()
        probe.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            patches = patchify(imgs, args.patch_size)
            
            B, N, _ = patches.shape
            noise = torch.rand(B, N, device=accelerator.device)
            ids_shuffle = noise.argsort(dim=1)
            ids_keep = ids_shuffle[:, :N // 4]
            ids_restore = ids_shuffle.argsort(dim=1)
            
            latents = encoder(patches, ids_keep)
            reconstructed = decoder(latents, ids_restore)
            
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

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            encoder.eval()
            with torch.no_grad():
                latents = encoder(patches)
                features = latents.mean(dim=1).detach()
            encoder.train()
            logits = probe(features)
            probe_loss = F.cross_entropy(logits, labels)
            probe_optimizer.zero_grad()
            accelerator.backward(probe_loss)
            probe_optimizer.step()
            preds = logits.argmax(dim=-1)
            preds, labels = accelerator.gather_for_metrics((preds, labels))
            acc = (preds == labels).sum().item() / len(labels)

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            accelerator.log({
                "train/epoch": (epoch-1) + (batch_idx + 1) / len(train_loader),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/loss": loss.item(),
                "train/pred_loss": pred_loss.item(),
                "train/var_loss": var_loss.item(),
                "train/sw_loss": sw_loss.item(),
                "train/probe_loss": probe_loss.item(),
                "train/probe_acc": acc,
            }, step=global_step)
            if batch_idx % args.log_interval == 0:
                accelerator.print(
                    f"Epoch [{epoch}/{args.num_epochs}]"
                    f" Batch [{batch_idx + 1}/{len(train_loader)}]"
                    f" LR: {optimizer.param_groups[0]['lr']:.6f}"
                    f" Loss: {loss.item():.4f}"
                    f" Pred Loss: {pred_loss.item():.4f}"
                    f" Var Loss: {var_loss.item():.4f}"
                    f" SW Loss: {sw_loss.item():.4f}"
                    f" Probe Loss: {probe_loss.item():.4f}"
                    f" Probe Acc: {acc:.4f}"
                )
            
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and epoch % args.save_interval == 0:
            accelerator.save_state(
                f"{args.save_dir}/checkpoint"
            )
        
        encoder.eval()
        decoder.eval()
        probe.eval()
        total_loss, total_mse, total_mae, total_psnr = 0.0, 0.0, 0.0, 0.0
        total_pred_loss, total_var_loss, total_sw_loss = 0.0, 0.0, 0.0
        total_correct_1, total_correct_5, total_correct_10 = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                patches = patchify(imgs, args.patch_size)
                
                B, N, _ = patches.shape
                noise = torch.rand(B, N, device=accelerator.device)
                ids_shuffle = noise.argsort(dim=1)
                ids_keep = ids_shuffle[:, :N // 4]
                ids_restore = ids_shuffle.argsort(dim=1)
                
                latents = encoder(patches, ids_keep)
                reconstructed = decoder(latents, ids_restore)
                
                pred_loss = F.mse_loss(
                    reconstructed, patches, reduction='none').mean(-1)
                preds = denormalize(unpatchify(
                    reconstructed, args.patch_size, args.image_size, args.in_channels))
                targets = denormalize(unpatchify(
                    patches, args.patch_size, args.image_size, args.in_channels))
                mask = torch.ones(B, N, device=accelerator.device)
                mask.scatter_(1, ids_keep, 0.)
                if batch_idx == 0 and accelerator.is_main_process:
                    masked_patches = patches * (1 - mask.unsqueeze(-1))
                    masked_inputs = denormalize(unpatchify(
                        masked_patches, args.patch_size,
                        args.image_size, args.in_channels))
                    fig = visualize(targets, masked_inputs, preds)
                    accelerator.log({"visualization": fig}, step=global_step)
                pred_loss = (pred_loss * mask).sum() / mask.sum()
                var_loss = variance_loss_fn(latents)
                sw_loss = sliced_wasserstein_loss_fn(latents)
                loss = (
                    args.pred_weight * pred_loss\
                    + args.var_weight * var_loss\
                    + args.sw_weight * sw_loss
                )
                mse = F.mse_loss(preds, targets)
                mae = F.l1_loss(preds, targets)
                psnr = -10 * torch.log10(mse)

                latents = encoder(patches)
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
        val_pred_loss = total_pred_loss / len(val_loader)
        val_var_loss = total_var_loss / len(val_loader)
        val_sw_loss = total_sw_loss / len(val_loader)
        val_loss = total_loss / len(val_loader)
        val_mse = total_mse / len(val_loader)
        val_mae = total_mae / len(val_loader)
        val_psnr = total_psnr / len(val_loader)
        correct_1 = total_correct_1 / len(val_loader.dataset)
        correct_5 = total_correct_5 / len(val_loader.dataset)
        correct_10 = total_correct_10 / len(val_loader.dataset)
        accelerator.log({
            "val/epoch": epoch,
            "val/pred_loss": val_pred_loss,
            "val/var_loss": val_var_loss,
            "val/sw_loss": val_sw_loss,
            "val/loss": val_loss,
            "val/mse": val_mse,
            "val/rmse": math.sqrt(val_mse),
            "val/mae": val_mae,
            "val/psnr": val_psnr,
            "val/acc@1": correct_1,
            "val/acc@5": correct_5,
            "val/acc@10": correct_10
        }, step=global_step)
        
        accelerator.print(
            f"Epoch [{epoch}/{args.num_epochs}]"
            f" Average Val Loss: {val_loss:.4f}"
            f" Average Val Pred Loss: {val_pred_loss:.4f}"
            f" Average Val Var Loss: {val_var_loss:.4f}"
            f" Average Val SW Loss: {val_sw_loss:.4f}"
            f" Average Val MSE: {val_mse:.4f}"
            f" Average Val RMSE: {math.sqrt(val_mse):.4f}"
            f" Average Val MAE: {val_mae:.4f}"
            f" Average Val PSNR: {val_psnr:.4f} dB"
            f" Average Val Acc@1: {correct_1:.4f}"
            f" Average Val Acc@5: {correct_5:.4f}"
            f" Average Val Acc@10: {correct_10:.4f}"
        )
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--data_dir', type=str, default='data/imagenet-540k-1k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_embed', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--decoder_n_embed', type=int, default=256)
    parser.add_argument('--decoder_n_heads', type=int, default=4)
    parser.add_argument('--decoder_n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_classes', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--probe_lr', type=float, default=6e-3)
    parser.add_argument('--pred_weight', type=float, default=1.0)
    parser.add_argument('--var_weight', type=float, default=25.0)
    parser.add_argument('--sw_weight', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()
    main(args)
