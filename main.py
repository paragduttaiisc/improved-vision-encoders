import math
import argparse
import torch, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from accelerate import Accelerator

from model import Encoder, Decoder
from utils import set_seed, patchify, unpatchify, denormalize, visualize, hrs
from dataloader import get_dataloaders


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_float32_matmul_precision('medium')
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb"
    )

    accelerator.init_trackers(
        project_name="improved-vision-encoders",
        config=vars(args),
        init_kwargs={
            "wandb": {
                "entity": "statsml-csa-iisc",
                "name": "ViT-MAE",
            }
        },
    )

    train_loader, val_loader, _ =\
        get_dataloaders(args.data_dir, args.batch_size, args.n_workers)
    accelerator.print(f"Train loader: {len(train_loader.dataset)} samples") # type: ignore
    accelerator.print(f"Validation loader: {len(val_loader.dataset)} samples") # type: ignore

    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.compile()
    decoder.compile()
    
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

    total_steps = args.num_epochs * len(train_loader) // torch.cuda.device_count()

    warmup_steps = len(train_loader) // torch.cuda.device_count() * 3
    cosine_steps = total_steps - warmup_steps
    warmup = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=args.lr / 10)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    accelerator.print(f"Encoder params: {hrs(encoder_params)}")
    accelerator.print(f"Decoder params: {hrs(decoder_params)}")

    encoder, decoder, optimizer, train_loader, val_loader = accelerator.prepare(
        encoder, decoder, optimizer, train_loader, val_loader)

    for epoch in range(1, args.num_epochs + 1):
        encoder.train()
        decoder.train()
        for batch_idx, (imgs, _) in enumerate(train_loader):
            patches = patchify(imgs, args.patch_size)
            
            B, N, _ = patches.shape
            noise = torch.rand(B, N, device=accelerator.device)
            ids_shuffle = noise.argsort(dim=1)
            ids_keep = ids_shuffle[:, :N // 4]
            ids_restore = ids_shuffle.argsort(dim=1)
            
            latents = encoder(patches, ids_keep)
            reconstructed = decoder(latents, ids_restore)
            
            loss = F.mse_loss(reconstructed, patches, reduction='none').mean(-1)
            mask = torch.ones(B, N, device=accelerator.device)
            mask.scatter_(1, ids_keep, 0.)
            loss = (loss * mask).sum() / mask.sum()

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_loader) + batch_idx
            accelerator.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

            if batch_idx % args.log_interval == 0:
                accelerator.print(
                    f"Epoch [{epoch}/{args.num_epochs}]"
                    f" Batch [{batch_idx + 1}/{len(train_loader)}]"
                    f" Loss: {loss.item():.4f}"
                )
            
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and epoch % args.save_interval == 0:
            accelerator.save_state(
                f"{args.save_dir}/checkpoint"
            )
        
        encoder.eval()
        decoder.eval()
        total_loss, total_mse, total_mae, total_psnr = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(val_loader):
                patches = patchify(imgs, args.patch_size)
                
                B, N, _ = patches.shape
                noise = torch.rand(B, N, device=accelerator.device)
                ids_shuffle = noise.argsort(dim=1)
                ids_keep = ids_shuffle[:, :N // 4]
                ids_restore = ids_shuffle.argsort(dim=1)
                
                latents = encoder(patches, ids_keep)
                reconstructed = decoder(latents, ids_restore)
                
                loss = F.mse_loss(reconstructed, patches, reduction='none').mean(-1)
                preds = denormalize(unpatchify(
                    reconstructed, args.patch_size, args.image_size, args.in_channels))
                targets = denormalize(unpatchify(
                    patches, args.patch_size, args.image_size, args.in_channels))
                if batch_idx == 0 and accelerator.is_main_process:
                    masked_patches = patches * (1 - mask.unsqueeze(-1))
                    masked_inputs = denormalize(unpatchify(
                        masked_patches, args.patch_size,
                        args.image_size, args.in_channels))
                    fig = visualize(targets, masked_inputs, preds)
                    accelerator.log({"visualization": fig}, step=global_step)
                mask = torch.ones(B, N, device=accelerator.device)
                mask.scatter_(1, ids_keep, 0.)
                loss = (loss * mask).sum() / mask.sum()
                mse = F.mse_loss(preds, targets)
                mae = F.l1_loss(preds, targets)
                psnr = -10 * torch.log10(mse)

                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                total_psnr += psnr.item()
        val_loss = total_loss / len(val_loader)
        val_mse = total_mse / len(val_loader)
        val_mae = total_mae / len(val_loader)
        val_psnr = total_psnr / len(val_loader)
        accelerator.log({
            "val/loss": val_loss,
            "val/mse": val_mse,
            "val/rmse": math.sqrt(val_mse),
            "val/mae": val_mae,
            "val/psnr": val_psnr,
        }, step=global_step)
        
        accelerator.print(
            f"Epoch [{epoch}/{args.num_epochs}]"
            f" Average Val Loss: {val_loss:.4f}"
            f" Average Val MSE: {val_mse:.4f}"
            f" Average Val RMSE: {math.sqrt(val_mse):.4f}"
            f" Average Val MAE: {val_mae:.4f}"
            f" Average Val PSNR: {val_psnr:.4f} dB")
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
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--decoder_n_embed', type=int, default=256)
    parser.add_argument('--decoder_n_heads', type=int, default=4)
    parser.add_argument('--decoder_n_layers', type=int, default=6)
    parser.add_argument('--n_classes', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    main(args)
