import argparse
import torch
import torch.nn as nn

from model import Encoder, Decoder
from dataloader import get_dataloaders


def main(args: argparse.Namespace) -> None:
    _, _, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.n_workers)

    encoder = Encoder(args)
    decoder = Decoder(args)

    encoder.load_state_dict(torch.load("models/encoder.pt"))
    decoder.load_state_dict(torch.load("models/decoder.pt"))

    encoder.eval()
    decoder.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/imagenet-540k-1k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=10)
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
    args = parser.parse_args()
    main(args)
