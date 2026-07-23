import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, NamedTuple, Optional, Tuple


class MLPOutput(NamedTuple):
    value: torch.Tensor
    loss: Optional[torch.Tensor] = None


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
    

class FourierMixer(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fft(x, dim=1).real
        return self.dropout(self.proj(x))


class FeedForward(nn.Module):
    def __init__(
            self,
            n_embed: int,
            hidden_size: int,
            activation: str,
            dropout: float
    ) -> None:
        super().__init__()
        assert activation in ["GELU", "SwiGLU", "SqReLU"],\
            "Unsupported activation"
        self.act = nn.GELU(approximate="tanh") if activation == "GELU"\
                    else lambda x: F.relu(x).square()
        self.forward = self._forward_standard
        if activation == "SwiGLU":
            hidden_size = 8 * n_embed // 3
            hidden_size = 256 * ((hidden_size + 255) // 256) # for efficiency
            self.gate_proj = nn.Linear(n_embed, hidden_size, bias=False)
            self.forward = self._forward_SwiGLU
        self.up_proj = nn.Linear(n_embed, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _forward_SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.up_proj(x))
        x = self.down_proj(x)
        return self.dropout(x)

    def __call__(self, x: torch.Tensor) -> MLPOutput:
        return MLPOutput(value=self.forward(x))


class MoE(nn.Module):
    def __init__(
        self,
        n_embed: int,
        hidden_size: int,
        n_experts: int = 8,
        top_k: int = 2,
        activation: str = "SwiGLU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(n_embed, n_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(
                n_embed=n_embed,
                hidden_size=hidden_size,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(n_experts)
        ])

        self.last_load = None
        self.last_importance = None
        self.last_aux_loss = None
        self.last_num_tokens = None

    def forward(self, x: torch.Tensor) -> MLPOutput:

        B, T, C = x.shape

        router_logits = self.router(x) # (B,T,E)
        router_probs = F.softmax(router_logits, dim=-1)
        importance = router_probs.float().mean(dim=(0, 1))

        topk_probs, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        flat_indices = indices.reshape(-1, self.top_k)
        flat_weights = topk_probs.float().reshape(-1, self.top_k)
        load = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        load.scatter_add_(0, flat_indices.reshape(-1), flat_weights.reshape(-1))
        load /= load.sum()

        target = torch.full(
            (self.num_experts,),
            1.0 / self.num_experts,
            device=x.device,
            dtype=torch.float32,
        )
        aux_loss = F.mse_loss(load, target) + F.mse_loss(importance, target)

        self.last_load = load.detach()
        self.last_importance = importance.detach()
        self.last_aux_loss = aux_loss.detach()
        self.last_num_tokens = torch.bincount(
            flat_indices.reshape(-1), minlength=self.num_experts).detach()

        x_flat = x.reshape(-1, C)
        out = torch.zeros_like(x_flat)
        flat_weights = weights.reshape(-1, self.top_k)

        for expert_id, expert in enumerate(self.experts):
            token_idx, kth = torch.where(flat_indices == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_input = x_flat[token_idx]
            expert_output = expert.forward(expert_input).to(out.dtype)
            expert_output *= flat_weights[token_idx, kth].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_output)
        out = out.view(B, T, C)

        return MLPOutput(value=out, loss=aux_loss)


class Block(nn.Module):
    def __init__(
            self,
            n_embed: int,
            num_heads: int,
            num_experts: int,
            num_active_experts: int,
            dropout: float,
            token_mixer: str = "MHA",
            activation: str = "SwiGLU"
    ) -> None:
        super().__init__()
        assert token_mixer in ["MHA", "Fourier"], f"Unsupported token mixer: {token_mixer}"
        if token_mixer == "MHA":
            self.token_mixer = MultiHeadAttention(
                num_heads=num_heads, n_embed=n_embed, dropout=dropout)
        else:
            self.token_mixer = FourierMixer(n_embed, dropout)
        if num_experts == 1:
            self.channel_mixer =\
                FeedForward(n_embed, 4*n_embed, activation, dropout)
        else:
            self.channel_mixer = MoE(
                n_embed, 4*n_embed, num_experts,
                num_active_experts, activation, dropout)
        self.norm1 = nn.RMSNorm(n_embed)
        self.norm2 = nn.RMSNorm(n_embed)
    
    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x + self.token_mixer(self.norm1(x))
        outs = self.channel_mixer(self.norm2(x))
        x = x + outs.value
        return x, outs.loss


class BaseModel(nn.Module):
    def __init__(
            self,
            n_embed: int,
            n_heads: int,
            n_experts: int,
            n_active: int,
            n_layers: int,
            token_mixer: str,
            activation: str,
            dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Block(n_embed, n_heads, n_experts, n_active,
                  dropout, token_mixer, activation)
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
    
    def forward_features(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, router_loss = layer(x)
            if router_loss is not None:
                aux_loss += router_loss
        aux_loss = aux_loss / len(self.layers) if router_loss != 0 else None
        return self.norm_f(x), aux_loss


class Encoder(BaseModel):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            args.n_embed, args.n_heads, args.n_experts, args.n_active_experts,
            args.n_layers, args.token_mixer, args.activation, args.dropout)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
            args.decoder_n_embed, args.decoder_n_heads, 1, 1,
            args.decoder_n_layers, args.token_mixer, "SqReLU", args.dropout)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.decoder_embed(latents)
        B, N, _ = x.shape
        num_mask = self.num_patches - N
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        idxs = ids_to_restore.unsqueeze(-1).expand(-1, -1, self.n_embed)
        x = torch.gather(x, dim=1, index=idxs)
        x = x + self.pos_emb
        x, aux_loss = self.forward_features(x)
        return self.pixel_head(x), aux_loss