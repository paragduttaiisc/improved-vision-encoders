import torch
import torch.nn.functional as F


def variance_loss_fn(
        z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def sliced_wasserstein_loss_fn(
        z: torch.Tensor, num_projections: int = 1024
) -> torch.Tensor:
    D = z.shape[-1]
    z = (z - z.mean(0)) / (z.std(0, unbiased=False) + 1e-6)
    directions = F.normalize(torch.randn(
        D, num_projections, device=z.device, dtype=z.dtype), dim=0)
    proj = z @ directions
    gaussian = torch.randn_like(proj)
    proj, _ = proj.sort(dim=0)
    gaussian, _ = gaussian.sort(dim=0)
    loss = (proj - gaussian).pow(2).mean()
    return loss