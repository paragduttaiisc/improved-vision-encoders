import argparse
import torch, torch.nn.functional as F
from safetensors.torch import load_file
from accelerate import Accelerator

from model import Encoder
from dataloader import get_dataloaders
from utils import patchify, set_seed


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_float32_matmul_precision('medium')
    accelerator = Accelerator(mixed_precision="bf16")

    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.n_workers)

    encoder = Encoder(args)
    encoder.load_state_dict(load_file("models/MAE/model.safetensors"))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    classifier = torch.nn.Linear(encoder.n_embed, args.n_classes)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0)

    encoder, classifier, optimizer, train_loader, val_loader =\
        accelerator.prepare(
            encoder, classifier, optimizer, train_loader, val_loader)
    
    for epoch in range(args.num_epochs):
        total_correct = 0
        for idx, (images, labels) in enumerate(train_loader):
            with torch.no_grad():
                patches = patchify(images, args.patch_size)
                features = encoder(patches).mean(dim=1).detach()
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            preds, labels = accelerator.gather_for_metrics((preds, labels))
            total_correct += (preds == labels).sum().item()
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            if idx % args.log_interval == 0:
                accelerator.print(
                    f"Epoch [{epoch+1}/{args.num_epochs}]"
                    f" Step [{idx}/{len(train_loader)}]"
                    f" Loss: {loss.item():.4f}")
        accelerator.print(
            f"Epoch [{epoch+1}/{args.num_epochs}]"
            f" Train Accuracy: {total_correct/len(train_loader.dataset):.4f}")
        
        total_correct = 0
        for idx, (images, labels) in enumerate(val_loader):
            with torch.no_grad():
                patches = patchify(images, args.patch_size)
                features = encoder(patches).mean(dim=1).detach()
                logits = classifier(features)
                preds = logits.argmax(dim=-1)
                preds, labels = accelerator.gather_for_metrics((preds, labels))
                total_correct += (preds == labels).sum().item()

        accelerator.print(
            f"Epoch [{epoch+1}/{args.num_epochs}]"
            f" Val Accuracy: {total_correct/len(val_loader.dataset):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--data_dir', type=str, default='data/imagenet-540k-1k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2048)
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
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--log_interval', type=int, default=10)
    args = parser.parse_args()
    main(args)
