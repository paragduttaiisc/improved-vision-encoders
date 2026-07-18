import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embed // num_heads

        self.attn_mat = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
            self,
            x: torch.Tensor,
            attn_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.attn_mat(x).chunk(3, dim=-1)  # Each: B, T, n_embed
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        if attn_masks is not None:
            attn_masks = attn_masks[:, None, None, :]
        if self.flash: # Flash attention using PyTorch 2.0's built-in function
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_masks, is_causal=False,
                dropout_p=self.dropout.p if self.training else 0.0)
        else: # Fallback to manual SDPA implementation
            att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
            if attn_masks is not None:
                att = att.masked_fill(attn_masks, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))  # B, T, n_embed


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
    def __init__(self, n_embed: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads, n_embed, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.norm1 = nn.RMSNorm(n_embed)
        self.norm2 = nn.RMSNorm(n_embed)
    
    def forward(
            self,
            x: torch.Tensor,
            attn_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.sa_heads(self.norm1(x), attn_masks)
        x = x + self.ffwd(self.norm2(x))
        return x


class ViTBaseModel(nn.Module):
    def __init__(
            self, n_embed: int, n_heads: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        assert n_embed % n_heads == 0
        
        self.layers = nn.ModuleList([
            Block(n_embed, n_heads, dropout)
            for _ in range(n_layers)
        ])
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


class Encoder(ViTBaseModel):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            args.n_embed, args.n_heads, args.n_layers, args.dropout)
        assert args.image_size % args.patch_size == 0
        num_patches = (args.image_size // args.patch_size) ** 2
        self.n_embed = args.n_embed
        self.patch_emb = nn.Linear(
            args.patch_size * args.patch_size * args.in_channels, args.n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, args.n_embed))
        
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


class Decoder(ViTBaseModel):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            args.decoder_n_embed, args.decoder_n_heads,
            args.decoder_n_layers, args.dropout)
        assert args.image_size % args.patch_size == 0
        self.num_patches = (args.image_size // args.patch_size) ** 2
        self.n_embed = args.decoder_n_embed
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_n_embed))
        self.decoder_embed = nn.Linear(args.n_embed, args.decoder_n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(
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
