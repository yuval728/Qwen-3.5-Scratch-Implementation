from __future__ import annotations

import torch
import torch.nn.functional as F

ACT2FN = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}

def apply_interleaved_mrope(freqs: torch.Tensor) -> torch.Tensor:
    freqs_t = freqs[0].clone()
    mrope_section = [11, 11, 10]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_heads * n_rep, seq_len, head_dim)


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    kv_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    causal = torch.full((query_length, kv_length), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1 + kv_length - query_length)

    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, kv_length)
    if attention_mask is None:
        return causal

    padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_value # don't look to PAD tokens
    return causal + padding_mask 


if __name__ == "__main__":
    print(build_causal_mask(
        torch.randn(4, 20),
        4,
        20,
        20,
        'cpu',
        torch.float32
    ))