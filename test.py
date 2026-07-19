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

    _, _, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.n_workers)

    encoder = Encoder(args)
    encoder.load_state_dict(load_file("models/MAE/model.safetensors"))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    classifier = torch.nn.Linear(encoder.n_embed, args.n_classes)
    classifier.load_state_dict(load_file("models/MAE/model_2.safetensors"))

    encoder, classifier, test_loader =\
        accelerator.prepare(encoder, classifier, test_loader)

    correct_1, correct_5, correct_10 = 0, 0, 0
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            patches = patchify(images, args.patch_size)
            features = encoder(patches).mean(dim=1).detach()
            logits = classifier(features)
            logits, labels = accelerator.gather_for_metrics((logits, labels))
            correct_1 += (logits.argmax(dim=-1) == labels).sum().item()
            correct_5 +=\
                (logits.topk(5, dim=-1).indices == labels.unsqueeze(-1)).sum().item()
            correct_10 +=\
                (logits.topk(10, dim=-1).indices == labels.unsqueeze(-1)).sum().item()
    accelerator.print(
        f" Test Accuracy@1: {correct_1/len(test_loader.dataset):.4f}"
        f" Test Accuracy@5: {correct_5/len(test_loader.dataset):.4f}"
        f" Test Accuracy@10: {correct_10/len(test_loader.dataset):.4f}"
    )
    accelerator.end_training()


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
    parser.add_argument('--n_classes', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
