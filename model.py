import torch
from torch import nn
import torch.nn.functional as F
from utils import build_causal_mask
from norm import RMSNorm, Qwen35RMSNorm
from rope import RoPE, Qwen35RotaryEmbedding
from decoder import Decoder, Qwen35DecoderLayer
from cache import DynamicCache, Qwen35DynamicCache

class TextModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.layers = nn.ModuleList([Decoder(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RoPE(config)
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
    ):
        input_embds = inputs_embeds
        cache = past_key_values
        pos_ids = position_ids
        # input_ids: [batch, seq_len]
        if input_embds is None:
            input_embds = self.embed_tokens(input_ids) # [batch, seq_len, hidden_size]


        batch_size, seq_len, _  = input_embds.shape

        if cache is None:
            cache = DynamicCache(self.config)

        past_seen_tokens = cache.get_seq_len()


        if pos_ids is None:
            pos_ids = torch.arange(0, seq_len, dtype=input_embds.dtype, device=input_embds.device) + past_seen_tokens
            pos_ids = pos_ids[None, :].expand(input_embds.shape[0], -1)

        pos_embeddings = self.rope(input_embds, pos_ids)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len + past_seen_tokens), dtype=input_embds.dtype, device=input_embds.device)

        kv_length = seq_len + past_seen_tokens
        
        causal = build_causal_mask(
            attention_mask, 
            batch_size, 
            seq_len, 
            kv_length, 
            dtype=input_embds.dtype, 
            device=input_embds.device)

        hidden_states = input_embds

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                pos_embeddings,
                attention_mask if layer.layer_type == "linear_attention" else causal,
                cache
            )
        
        hidden_states = self.norm(hidden_states)
        return self.out_proj(hidden_states), cache
    
    

class Qwen35TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Qwen35DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, Qwen35DynamicCache | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen35DynamicCache(self.config)

        batch_size, seq_len, _ = inputs_embeds.shape

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len + past_seen_tokens),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        kv_length = seq_len + past_seen_tokens
        causal_mask = build_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        hidden_states = inputs_embeds
        for layer in self.layers:
            layer_mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = (
            None
            if getattr(config, "tie_word_embeddings", False)
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        self.tie_weights()

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", False) and self.lm_head is not None:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Qwen35DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, Qwen35DynamicCache | None]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        if self.lm_head is None:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        return logits, past_key_values
