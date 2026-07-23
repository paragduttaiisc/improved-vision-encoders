import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FourierMixer(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fft(x, dim=1).real
        return self.dropout(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.token_mixer = FourierMixer(n_embed, dropout)
        self.channel_mixer = FeedForward(n_embed, dropout)
        self.norm1 = nn.RMSNorm(n_embed)
        self.norm2 = nn.RMSNorm(n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))
        return x


class BaseModel(nn.Module):
    def __init__(self, n_embed: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Block(n_embed, dropout) for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(n_embed)
    
    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class Encoder(BaseModel):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args.n_embed, args.n_layers, args.dropout)
        assert args.image_size % args.patch_size == 0
        num_patches = (args.image_size // args.patch_size) ** 2
        self.n_embed = args.n_embed
        self.patch_emb = nn.Linear(
            args.patch_size * args.patch_size * args.in_channels, args.n_embed)
        self.pos_emb = nn.Parameter(torch.empty(1, num_patches, args.n_embed))
        
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(
            self,
            patches: torch.Tensor,
            ids_to_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, _, D = patches.shape
        pos = self.pos_emb.expand(B, -1, -1)
        if ids_to_keep is not None:
            patch_ids = ids_to_keep.unsqueeze(-1).expand(-1, -1, D)
            patches = torch.gather(patches, dim=1, index=patch_ids)
            pos_ids = ids_to_keep.unsqueeze(-1).expand(-1, -1, self.n_embed)
            pos = torch.gather(pos, dim=1, index=pos_ids)
        x = self.patch_emb(patches) + pos
        return self.forward_features(x)


class Decoder(BaseModel):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            args.decoder_n_embed, args.decoder_n_layers, args.dropout)
        assert args.image_size % args.patch_size == 0
        self.num_patches = (args.image_size // args.patch_size) ** 2
        self.n_embed = args.decoder_n_embed
        self.mask_token = nn.Parameter(torch.empty(1, 1, args.decoder_n_embed))
        self.decoder_embed = nn.Linear(args.n_embed, args.decoder_n_embed)
        self.pos_emb = nn.Parameter(torch.empty(
            1, self.num_patches, args.decoder_n_embed))
        self.pixel_head = nn.Linear(
            args.decoder_n_embed, (args.patch_size ** 2) * args.in_channels)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
    
    def forward(
            self,
            latents: torch.Tensor,
            ids_to_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(latents)
        B, N, _ = x.shape
        num_mask = self.num_patches - N
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        idxs = ids_to_restore.unsqueeze(-1).expand(-1, -1, self.n_embed)
        x = torch.gather(x, dim=1, index=idxs)
        x = x + self.pos_emb
        x = self.forward_features(x)
        return self.pixel_head(x)