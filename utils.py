import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def human_readable_numbers(size: int) -> str:
    for unit in ['', 'K', 'M', 'G', 'T']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024 # type: ignore
    return f"{size:.2f}{unit}"


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    assert imgs.shape[2] % patch_size == 0
    assert imgs.shape[3] % patch_size == 0
    return rearrange(
        imgs,
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=patch_size,
        p2=patch_size,
    )


def unpatchify(
        patches: torch.Tensor,
        patch_size: int,
        image_size: int,
        in_channels: int = 3
) -> torch.Tensor:
    return rearrange(
        patches,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=image_size // patch_size,
        w=image_size // patch_size,
        p1=patch_size,
        p2=patch_size,
        c=in_channels,
    )


def denormalize(
        imgs: torch.Tensor,
        mean: torch.Tensor = torch.tensor([0.485,0.456,0.406]).view(1, 3, 1, 1),
        std: torch.Tensor = torch.tensor([0.229,0.224,0.225]).view(1, 3, 1, 1)
) -> torch.Tensor:
    if type(mean) is not torch.Tensor:
        mean = torch.tensor(mean, dtype=imgs.dtype, device=imgs.device)
        std = torch.tensor(std, dtype=imgs.dtype, device=imgs.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 3, 1, 1)
        std = std.view(-1, 3, 1, 1)
    if mean.device != imgs.device:
        mean = mean.to(imgs.device)
        std = std.to(imgs.device)
    if imgs.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    return (imgs * std + mean).clamp(0, 1)


def visualize(original, masked, reconstruction):
    num_examples = min(10, original.size(0))
    rows = (num_examples + 1) // 2
    fig, axes = plt.subplots(rows, 6, figsize=(18, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    for idx in range(num_examples):
        row = idx % rows
        col_offset = 0 if idx < rows else 3
        images = [original[idx], masked[idx], reconstruction[idx]]
        titles = ["Original", "Masked Input", "Reconstruction"]
        for j, (img, title) in enumerate(zip(images, titles)):
            ax = axes[row, col_offset + j]
            ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    return fig