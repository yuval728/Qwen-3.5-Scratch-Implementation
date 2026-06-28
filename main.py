import torch
from torch import nn
import torch.nn.functional as F
from utils import build_causal_mask
from norm import RMSNorm
from rope import RoPE
from attention import SelfAttention
from mlp import MLP
from delta import GatedDeltaNet
from decoder import Decoder
from cache import DynamicCache
    
    
class TextModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self.config = config


        self.rope = RoPE(config)
        self.layers = nn.ModuleList([Decoder(config, i) for i in range(config.num_hidden_layers)])

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.out_proj = nn.Identity()
    
    def forward(
        self,
        input_ids=None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = True,
    ):
        # input_ids: [batch, seq_len]
        # inputs_embeds: [batch, seq_len, hidden_size]

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).to(device=input_ids.device)

        batch_size, seq_len, _ = inputs_embeds.shape

        cache = past_key_values
        if use_cache and cache is None:
            cache = DynamicCache(self.config)

        past_seen_tokens = cache.get_seq_len() if cache is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len + past_seen_tokens),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        kv_length = seq_len + past_seen_tokens
        
        causal_mask = build_causal_mask(
            attention_mask,
            batch_size,
            seq_len,
            kv_length,
            inputs_embeds.device,
            inputs_embeds.dtype
        )

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids[None, ...].expand(batch_size, -1)
        
        positional_embedding = self.rope(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                positional_embedding,
                mask,
                cache
            )
        
        hidden_states = self.norm(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states, cache
        
            
            
        
if __name__ == "__main__":
    from config import config
    norm = RMSNorm(64)
    out = norm(torch.randn(4, 20, 64))
    # print(out)

    batch_size = 4
    seq_len = 20

    pos_ids = torch.arange(0, seq_len).expand(batch_size, -1)
    rope = RoPE(config)
    out = rope(torch.randn(batch_size, seq_len, 128), pos_ids)
    # print(out)
    # print(out.shape)
    
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size) 
    pos_emb = torch.randn(batch_size, seq_len, config.head_dim)
    attention_mask = torch.randn(batch_size, 1, 1, seq_len)
    
    self_attn = SelfAttention(config)
    out = self_attn(hidden_states, (pos_emb, pos_emb), attention_mask)
    print(out.shape)
    
    
    gated_delta = GatedDeltaNet(config)
    out = gated_delta(hidden_states)
    print(out.shape)
    
    
    
    
    