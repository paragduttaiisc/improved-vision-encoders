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


def ffn_hidden_size(n_embed: int, hidden_size: int, activation: str) -> int:
    """Actual FFN width. SwiGLU ignores `hidden_size` and uses 8/3 * n_embed
    rounded up to a multiple of 256 (same params as a 4x 2-matrix FFN)."""
    if activation != "SwiGLU":
        return hidden_size
    return 256 * ((8 * n_embed // 3 + 255) // 256)


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
        hidden_size = ffn_hidden_size(n_embed, hidden_size, activation)
        if activation == "SwiGLU":
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
    """Mixture-of-Experts FFN with top-k token-choice routing.

    Ported from modern_llms (model/feed_forward.py). Expert weights are
    stacked into 3-D parameters so all experts run in one `grouped_mm` call
    over expert-sorted tokens, instead of a Python loop with data-dependent
    shapes (which forced host syncs and torch.compile graph breaks):
        SwiGLU:        gate_up_weight [E, 2H, C] (rows [gate (H) | up (H)]),
                       down_weight    [E, C, H]
        GELU / SqReLU: up_weight      [E, H, C], down_weight [E, C, H]
    H follows `ffn_hidden_size`, so it matches the old per-expert
    FeedForward and legacy checkpoints convert exactly
    (`convert_legacy_moe_state_dict`).

    `grouped_mm` only runs in bf16 and is not covered by autocast, so the
    expert matmuls cast both activations and (fp32 master) weights to bf16,
    like every other matmul under bf16 autocast.

    Side effects of `forward` for logging, all detached: `last_load` [E]
    fraction of routed slots per expert, `last_importance` [E] mean router
    probability, `last_num_tokens` [E] slot counts, `last_aux_loss`.
    """
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
        if activation not in {"GELU", "SwiGLU", "SqReLU"}:
            raise ValueError(f"Unsupported activation: {activation}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(
                f"need 1 <= top_k <= n_experts, got {top_k=}, {n_experts=}")

        self.n_embed = n_embed
        self.num_experts = n_experts
        self.top_k = top_k
        self.activation = activation
        self.dropout_p = dropout
        self.hidden_size = H = ffn_hidden_size(n_embed, hidden_size, activation)

        self.router = nn.Linear(n_embed, n_experts, bias=False)
        self.gate_up_weight: nn.Parameter | None = None
        self.up_weight: nn.Parameter | None = None
        if activation == "SwiGLU":
            self.gate_up_weight = nn.Parameter(
                torch.empty(n_experts, 2 * H, n_embed))
        else:
            self.up_weight = nn.Parameter(torch.empty(n_experts, H, n_embed))
        self.down_weight = nn.Parameter(torch.empty(n_experts, n_embed, H))
        # Same init as the nn.Linear experts got from BaseModel._init_weights
        # (which does not reach these 3-D parameters).
        for p in (self.gate_up_weight, self.up_weight, self.down_weight):
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)

        self.last_load: torch.Tensor | None = None
        self.last_importance: torch.Tensor | None = None
        self.last_aux_loss: torch.Tensor | None = None
        self.last_num_tokens: torch.Tensor | None = None

    @staticmethod
    def _grouped_mm(
        x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """x [N_routed, in] with rows grouped by expert; weight [E, in, out];
        offsets [E] int32 cumulative row counts. Returns bf16 [N_routed, out]."""
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        if hasattr(F, "grouped_mm"):
            return F.grouped_mm(x, weight, offs=offsets)
        if hasattr(torch, "_grouped_mm"):
            return torch._grouped_mm(  # pyright: ignore[reportPrivateImportUsage]
                x, weight, offsets)
        out = x.new_empty(x.shape[0], weight.shape[-1])
        start = 0
        for e in range(weight.shape[0]):
            end = int(offsets[e].item())
            if end > start:
                out[start:end] = x[start:end] @ weight[e]
            start = end
        return out

    def forward(self, x: torch.Tensor) -> MLPOutput:
        B, T, C = x.shape
        N, H, K = B * T, self.hidden_size, self.top_k
        x_flat = x.reshape(N, C)

        router_probs = F.softmax(self.router(x_flat).float(), dim=-1)  # [N, E]
        topk_probs, topk_idx = torch.topk(router_probs, k=K, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Per-expert slot counts via a fixed-size scatter_add (bincount would
        # read its max index back to the host every step).
        flat_idx = topk_idx.reshape(-1)
        counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        importance = router_probs.mean(dim=0)  # differentiable
        load = counts.float() / float(N * K)    # counts: no gradient
        # Switch-Transformer balance loss; == 1.0 when perfectly balanced.
        aux_loss = self.num_experts * torch.sum(load * importance)

        self.last_load = load.detach()
        self.last_importance = importance.detach()
        self.last_aux_loss = aux_loss.detach()
        self.last_num_tokens = counts.detach()

        # Sort the N*K (token, expert) slots by expert so each expert's rows
        # are contiguous, which is the layout grouped_mm expects.
        sort_order = torch.argsort(flat_idx)
        token_idx = torch.arange(N, device=x.device).repeat_interleave(K)
        x_sorted = x_flat[token_idx[sort_order]]
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

        if self.gate_up_weight is not None:
            gate_up = self._grouped_mm(
                x_sorted, self.gate_up_weight.transpose(-1, -2), offsets)
            hidden = F.silu(gate_up[:, :H]) * gate_up[:, H:]
        else:
            assert self.up_weight is not None
            hidden = self._grouped_mm(
                x_sorted, self.up_weight.transpose(-1, -2), offsets)
            hidden = F.gelu(hidden, approximate="tanh") \
                if self.activation == "GELU" else F.relu(hidden).square()
        out_sorted = self._grouped_mm(
            hidden, self.down_weight.transpose(-1, -2), offsets)
        out_sorted = out_sorted * topk_weights.reshape(-1)[sort_order]\
            .to(out_sorted.dtype).unsqueeze(-1)

        # Unsort back to (token, slot) order and sum each token's K outputs.
        out = torch.empty_like(out_sorted)
        out[sort_order] = out_sorted
        out = out.view(N, K, C).float().sum(dim=1).to(x.dtype)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        return MLPOutput(value=out.view(B, T, C), loss=aux_loss)


def convert_legacy_moe_state_dict(
        sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert checkpoints from the old loop-based MoE (one FeedForward per
    expert: `...channel_mixer.experts.{e}.{gate,up,down}_proj.weight`) to the
    stacked layout (`gate_up_weight` / `up_weight` / `down_weight`). Keys of
    an already-converted (or dense) state dict pass through unchanged."""
    groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        head, sep, tail = k.partition(".experts.")
        parts = tail.split(".")
        if sep and len(parts) == 3 and parts[2] == "weight" and parts[0].isdigit():
            groups.setdefault(head, {}).setdefault(int(parts[0]), {})[parts[1]] = v
        else:
            out[k] = v
    for head, experts in groups.items():
        ex = [experts[e] for e in sorted(experts)]
        if "gate_proj" in ex[0]:
            out[f"{head}.gate_up_weight"] = torch.stack(
                [torch.cat([e["gate_proj"], e["up_proj"]], dim=0) for e in ex])
        else:
            out[f"{head}.up_weight"] = torch.stack([e["up_proj"] for e in ex])
        out[f"{head}.down_weight"] = torch.stack([e["down_proj"] for e in ex])
    return out


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
        router_losses = []
        for layer in self.layers:
            x, router_loss = layer(x)
            if router_loss is not None:
                router_losses.append(router_loss)
        aux_loss = torch.stack(router_losses).mean() if router_losses else None
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