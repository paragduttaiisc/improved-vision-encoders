import math
import argparse
import torch, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from accelerate import Accelerator

from model import Model
from utils import set_seed, human_readable_numbers as hrn
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

    model = Model(args)
    model.compile()
    
    linear_probe = torch.nn.Linear(args.n_embed, args.n_classes)
    torch.nn.init.trunc_normal_(linear_probe.weight, std=0.02)
    torch.nn.init.zeros_(linear_probe.bias)
    
    decay = []
    no_decay = []
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
        linear_probe.parameters(), lr=args.probe_lr,
        betas=(0.9, 0.95), weight_decay=0.0)

    total_steps = args.num_epochs * len(train_loader) // torch.cuda.device_count()

    warmup_steps = len(train_loader) // torch.cuda.device_count() * 10
    cosine_steps = total_steps - warmup_steps
    warmup = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=args.lr / 10)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    accelerator.print(f"Train loader: {hrn(len(train_loader.dataset))} samples") # type: ignore
    accelerator.print(f"Validation loader: {hrn(len(val_loader.dataset))} samples") # type: ignore
    accelerator.print(f"Model has {hrn(sum(p.numel() for p in model.parameters()))} params")

    model, linear_probe, optimizer, probe_optimizer, train_loader, val_loader =\
        accelerator.prepare(
            model, linear_probe, optimizer, probe_optimizer, train_loader, val_loader)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        linear_probe.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            B = len(labels)
            global_batch = torch.cat(imgs[:2], dim=0)
            local_batch = torch.cat(imgs[2:], dim=0)
            global_latents = model(global_batch).mean(dim=1).view(2, B, -1)
            local_latents = model(local_batch).mean(dim=1).view(6, B, -1)
            all_latents = torch.cat([global_latents, local_latents], dim=0)

            mu = global_latents.mean(dim=0, keepdim=True)
            pred_loss = F.mse_loss(all_latents, mu.expand_as(all_latents))
            z = all_latents.reshape(-1, args.n_embed)
            var_loss = variance_loss_fn(z)
            sw_loss = sliced_wasserstein_loss_fn(z)
            loss = (
                args.pred_weight * pred_loss\
                + args.var_weight * var_loss\
                + args.sw_weight * sw_loss
            )

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            logits = linear_probe(global_latents[0].detach())
            probe_loss = F.cross_entropy(logits, labels)
            probe_optimizer.zero_grad()
            accelerator.backward(probe_loss)
            probe_optimizer.step()
            preds = logits.argmax(dim=-1)
            preds, labels = accelerator.gather_for_metrics((preds, labels))
            acc = (preds == labels).sum().item() / len(labels)

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            if batch_idx % args.log_interval == 0:
                accelerator.log({
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/loss": loss.item(),
                    "train/pred_loss": pred_loss.item(),
                    "train/var_loss": var_loss.item(),
                    "train/sw_loss": sw_loss.item(),
                    "train/probe_loss": probe_loss.item(),
                    "train/probe_acc": acc,
                }, step=global_step)
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
        
        model.eval()
        linear_probe.eval()
        total_correct_1, total_correct_5, total_correct_10 = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                B = len(labels)
                logits = linear_probe(model(imgs).mean(dim=1).detach())
                logits, labels = accelerator.gather_for_metrics((logits, labels))
                total_correct_1 +=\
                    (logits.argmax(dim=-1) == labels).sum().item()
                labels = labels.unsqueeze(-1)
                total_correct_5 +=\
                    (logits.topk(5, dim=-1).indices == labels).sum().item()
                total_correct_10 +=\
                    (logits.topk(10, dim=-1).indices == labels).sum().item()
        correct_1 = total_correct_1 / len(val_loader.dataset)
        correct_5 = total_correct_5 / len(val_loader.dataset)
        correct_10 = total_correct_10 / len(val_loader.dataset)
        accelerator.log({
            "val/acc@1": correct_1,
            "val/acc@5": correct_5,
            "val/acc@10": correct_10
        }, step=global_step)
        
        accelerator.print(
            f"Epoch [{epoch}/{args.num_epochs}]"
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
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_classes', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--probe_lr', type=float, default=1e-3)
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
