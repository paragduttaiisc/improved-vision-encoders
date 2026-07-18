import random
import torch
import numpy as np
from einops import rearrange


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def human_readable_size(size: int) -> str:
    """Convert a size in bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024 # type: ignore
    return f"{size:.2f} {unit}"


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    assert imgs.shape[2] % patch_size == 0
    assert imgs.shape[3] % patch_size == 0
    return rearrange(
        imgs,
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=patch_size,
        p2=patch_size,
    )