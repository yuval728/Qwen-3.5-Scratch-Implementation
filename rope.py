from __future__ import annotations

import torch
from torch import nn
from utils import apply_interleaved_mrope


def rotate_half(x: torch.Tensor):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:

    # q, k: (batch, num_heads, seq_len, head_dim)
    # cos, sin: (batch, seq_len, pos_dim)

    # (batch, seq_len, dim) -> (batch, 1, seq_len, dim)

    cos = cos.unsqueeze(unsqueeze_dim)  # (batch, 1, seq_len, dim)
    sin = sin.unsqueeze(unsqueeze_dim)  # (batch, 1, seq_len, dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
        
    return q_embed, k_embed


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.pos_rotary_factor = 1.0
        self.theta = config.theta * self.pos_rotary_factor
        self.inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_heads, seq_len, head_dim)
        # pos_ids: (batch, seq_len)

        # (3, batch, seq_len)
        pos_ids = pos_ids[None, ...].expand(3, x.shape[0], -1)

        # (3, batch, dim//2, 1)
        inv_freq = self.inv_freq[None, None, :, None].expand(3, x.shape[0], -1, 1)

        # [3, batch, dim//2, 1] @ [3, batch, 1, seq_len] -> [3, batch, dim//2, seq_len]
        freq = inv_freq @ pos_ids[:, :, None, :].float()  # (3, batch, dim//2, seq_len)
        freq = freq.transpose(2, 3)  # (batch, 3, seq_len, dim//2)
        freq = apply_interleaved_mrope(freq)  # (batch, 3, seq_len, dim)

        embed = torch.cat((freq, freq), dim=-1)  # (batch, seq_len, dim)
        return embed.cos().to(dtype=x.dtype), embed.sin().to(dtype=x.dtype)



class Qwen35RotaryEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.theta = config.theta
        self.dim = min(config.dim, config.head_dim)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim != 3:
            raise ValueError(f"Expected 2D or 3D position ids, got {tuple(position_ids.shape)}")

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        theta=10000.0,
        rotaty_factor=1.0,
        dim=20,
        head_dim=20,
        mrope_section=[3, 2, 2],
    )
    rope = Qwen35RotaryEmbedding(config, "cpu")

    q = torch.randn(4, 8, 20, 20)
    k = torch.randn(4, 8, 20, 20)
    position_ids = torch.arange(20).unsqueeze(0).expand(4, -1)
    cos, sin = rope(q, position_ids)

    print(cos.shape)
    print(q.shape)
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
    print(q_embed.shape, k_embed.shape)