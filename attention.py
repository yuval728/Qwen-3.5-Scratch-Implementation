from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from norm import RMSNorm, Qwen35RMSNorm
from rope import apply_rotary_pos_emb
from utils import repeat_kv


class SelfAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        self.layer_idx = layer_idx

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.q_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_attention_heads * 2
        )

        self.k = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_kv_heads,
            bias=config.attention_bias,
        )
        self.v = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_kv_heads,
            bias=config.attention_bias,
        )

        self.proj_out = nn.Linear(
            self.head_dim * self.num_attention_heads,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states, pos_embeddings, attention_mask, cache=None):
        batch_size, seq_len, _ = hidden_states.size()

        q_proj = self.q_proj(
            hidden_states
        )  # (batch_size, seq_len, head_dim * num_attention_heads * 2)
        q, gate = torch.chunk(
            q_proj, 2, dim=-1
        )  # (batch_size, seq_len, head_dim * num_attention_heads), (batch_size, seq_len, head_dim * num_attention_heads)
        gate = gate.reshape(batch_size, seq_len, -1)

        q = self.q_norm(
            q.reshape(
                batch_size, seq_len, self.num_attention_heads, self.head_dim
            ).transpose(1, 2)
        )
        k = self.k_norm(
            self.k(hidden_states)
            .reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(hidden_states)
            .reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE to q and k
        cos, sin = (
            pos_embeddings  # (batch_size, seq_len, dim), (batch_size, seq_len, dim)
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)

        k = k.repeat_interleave(
            self.num_kv_groups, dim=1
        )  # (batch_size, num_attention_heads, seq_len, head_dim)
        v = v.repeat_interleave(
            self.num_kv_groups, dim=1
        )  # (batch_size, num_attention_heads, seq_len, head_dim)

        # Compute attention scores
        # attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(
            q, k.transpose(2, 3)
        )  # (batch_size, num_attention_heads, seq_len, seq_len)
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_out = torch.matmul(
            F.softmax(attn_weights, dim=-1), v
        )  # (batch_size, num_attention_heads, seq_len, head_dim)
        attn_out = attn_out.transpose(2, 3)

        attn_out = attn_out.reshape(
            batch_size, seq_len, -1
        )  # (batch_size, seq_len, head_dim * num_attention_heads)
        attn_out = attn_out * F.sigmoid(gate)

        out = self.proj_out(attn_out)  # (batch_size, seq_len, hidden_size)
        return out


class Qwen35Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).reshape(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)
