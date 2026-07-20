import random
import torch
import numpy as np


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


def denormalize(
        imgs: torch.Tensor,
        mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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