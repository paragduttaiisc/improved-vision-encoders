import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embed // num_heads

        self.attn_mat = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.attn_mat(x).chunk(3, dim=-1)  # Each: B, T, n_embed
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        if self.flash: # Flash attention using PyTorch 2.0's built-in function
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=False,
                dropout_p=self.dropout.p if self.training else 0.0)
        else: # Fallback to manual SDPA implementation
            att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa_heads(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        assert args.n_embed % args.n_heads == 0
        assert args.image_size % args.patch_size == 0
        self.n_embed = args.n_embed
        self.patch_size = args.patch_size
        self.base_grid_size = args.image_size // args.patch_size
        num_patches = self.base_grid_size ** 2

        self.patch_emb = nn.Linear(
            args.patch_size ** 2 * args.in_channels, args.n_embed)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, num_patches, args.n_embed))
        self.layers = nn.ModuleList([
            Block(args.n_embed, args.n_heads, args.dropout)
            for _ in range(args.n_layers)])
        self.norm_f = nn.RMSNorm(args.n_embed)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def interpolate_pos_emb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if grid_h == self.base_grid_size and grid_w == self.base_grid_size:
            return self.pos_emb
        D = self.pos_emb.shape[-1]
        pos = self.pos_emb.reshape(
            1, self.base_grid_size, self.base_grid_size, D)
        pos = pos.permute(0, 3, 1, 2)
        pos = F.interpolate(
            pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1)
        pos = pos.reshape(1, grid_h * grid_w, D)
        return pos
    
    @staticmethod
    def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
        return rearrange(
            imgs, "b c (gh ph) (gw pw) -> b (gh gw) (ph pw c)",
            ph=patch_size, pw=patch_size)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = imgs.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        patches = self.patchify(imgs, self.patch_size)
        pos = self.interpolate_pos_emb(grid_h, grid_w).expand(B, -1, -1)
        
        x = self.patch_emb(patches)
        x = x + pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x