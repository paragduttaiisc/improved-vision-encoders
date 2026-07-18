import argparse
import torch, torch.nn.functional as F

from model import Encoder, Decoder
from utils import human_readable_size as hrs, set_seed, patchify
from dataloader import get_dataloaders


def main(args: argparse.Namespace) -> None:    
    set_seed(args.seed)

    train_loader, val_loader, test_loader =\
        get_dataloaders(args.data_dir, args.batch_size)
    print(f"Train loader: {len(train_loader.dataset)} samples") # type: ignore
    print(f"Validation loader: {len(val_loader.dataset)} samples") # type: ignore
    print(f"Test loader: {len(test_loader.dataset)} samples") # type: ignore

    encoder = Encoder(args).to(args.device)
    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    decoder = Decoder(args).to(args.device)
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {hrs(encoder_params)}")
    print(f"Decoder params: {hrs(decoder_params)}")

    for epoch in range(1, args.num_epochs + 1):
        encoder.train()
        decoder.train()
        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            patches = patchify(imgs, args.patch_size)
            
            B, N, _ = patches.shape
            noise = torch.rand(B, N, device=args.device)
            ids_shuffle = noise.argsort(dim=1)
            ids_keep = ids_shuffle[:, :N // 4]
            ids_restore = ids_shuffle.argsort(dim=1)
            
            latents = encoder(patches, ids_keep)
            reconstructed = decoder(latents, ids_restore)
            
            loss = F.mse_loss(reconstructed, patches, reduction='none').mean(-1)
            mask = torch.ones(B, N, device=args.device)
            mask.scatter_(1, ids_keep, 0.)
            loss = (loss * mask).sum() / mask.sum()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            encoder_optimizer.step()
            decoder_optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch [{epoch}/{args.num_epochs}]"
                    f" Batch [{batch_idx + 1}/{len(train_loader)}]"
                    f" Loss: {loss.item():.4f}"
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--data_dir', type=str, default='data/imagenet')
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
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=10)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
