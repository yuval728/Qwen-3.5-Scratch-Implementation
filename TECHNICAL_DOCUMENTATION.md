# Technical Documentation

This document explains the implementation in this repository in detail. It is written for someone who wants to understand, debug, or extend the runtime, not just run the CLI.

## 1. High-Level Architecture

The repository implements an inference-only text runtime for a Qwen 3.5 style hybrid decoder language model.

At a high level, generation follows this pipeline:

```text
prompt text
  -> tokenizer
  -> input_ids and attention_mask
  -> Qwen35ForCausalLM
  -> Qwen35TextModel
  -> repeated Qwen35DecoderLayer blocks
  -> final RMSNorm
  -> LM logits
  -> greedy argmax next token
  -> append token and repeat
```

The central model path is:

```text
Qwen35ForCausalLM
  model: Qwen35TextModel
    embed_tokens
    layers: ModuleList[Qwen35DecoderLayer]
    norm
    rotary_emb
  lm_head or tied embedding projection
```

The code also contains older legacy classes (`TextModel`, `Decoder`, `SelfAttention`, `GatedDeltaNet`, `RMSNorm`, `MLP`, `DynamicCache`). These are useful as earlier references, but the Qwen 3.5 checkpoint-compatible path is the `Qwen35*` class family.

## 2. Entry Point: `main.py`

`main.py` is the CLI entry point. It does four jobs:

1. Parse command-line arguments.
2. Resolve and load model files.
3. Build the model and tokenizer.
4. Run greedy generation.

### 2.1 Download Patterns

`DOWNLOAD_PATTERNS` limits the files fetched by `snapshot_download()`:

```python
DOWNLOAD_PATTERNS = [
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "processor_config.json",
    "*.txt",
]
```

This avoids downloading unrelated files while keeping everything needed for text inference.

### 2.2 Tokenizer Loading

`load_tokenizer(model_dir)` first tries:

```python
AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
```

If that fails, it tries:

```python
AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
```

and returns `processor.tokenizer` if present. This fallback is useful for repos where tokenizer access is wrapped by a processor.

### 2.3 Runtime Sequence

`main()` does this:

```python
model_dir = resolve_model_path(args.model_id, allow_patterns=DOWNLOAD_PATTERNS)
config = Qwen35Config.from_dict(load_config_dict(model_dir))
model = Qwen35ForCausalLM(config).to(args.device)
missing, unexpected = load_weights(model, model_dir)
```

Then it tokenizes the prompt, moves tensors to the requested device, calls `greedy_generate()`, and decodes the output ids.

## 3. Configuration: `config.py`

`config.py` defines two things:

- `config`: a small `SimpleNamespace` toy config for local smoke checks.
- `Qwen35Config`: the real dataclass used by the Qwen 3.5 runtime.

### 3.1 Important Config Fields

Core vocabulary and token fields:

```text
vocab_size
pad_token_id
bos_token_id
eos_token_id
tie_word_embeddings
```

Model dimensions:

```text
hidden_size
num_hidden_layers
intermediate_size
```

Layer structure:

```text
layer_types
```

Each entry is either:

- `linear_attention`
- `full_attention`

Attention dimensions:

```text
num_attention_heads
num_key_value_heads
head_dim
attention_dropout
attention_bias
```

Linear attention dimensions:

```text
num_v_heads
num_k_heads
head_k_dim
head_v_dim
linear_conv_kernel_size
```

RoPE and MRoPE fields:

```text
theta
rotaty_factor
dim
mrope_section
```

`rotaty_factor` is spelled that way because the upstream config can use that spelling. The parser also accepts `rotary_factor` and `rope_parameters.partial_rotary_factor`.

### 3.2 `from_dict()`

`Qwen35Config.from_dict()` normalizes Hugging Face config shapes. If the input JSON has a `text_config` object, it uses that nested object for language-model fields.

It also maps several possible field names:

```text
linear_num_value_heads -> num_v_heads
linear_num_key_heads -> num_k_heads
linear_key_head_dim -> head_k_dim
linear_value_head_dim -> head_v_dim
linear_conv_kernel_dim -> linear_conv_kernel_size
rope_parameters.rope_theta -> theta
rope_parameters.mrope_section -> mrope_section
```

This lets the runtime tolerate minor naming differences in model config JSONs.

### 3.3 `__post_init__()`

If `dim` is missing, it is computed as:

```python
int(self.head_dim * self.rotaty_factor)
```

If `mrope_section` is missing, it defaults to:

```python
[11, 11, 10]
```

## 4. Model Wrapper: `model.py`

`model.py` contains both legacy and Qwen 3.5 model stacks.

## 4.1 Legacy `TextModel`

`TextModel` builds layers with the legacy `Decoder` class:

```python
self.layers = nn.ModuleList([Decoder(config, i) for i in range(config.num_hidden_layers)])
```

That legacy stack uses module names such as:

```text
self_attn.k
self_attn.v
self_attn.proj_out
```

Those names do not match Qwen 3.5 full-attention checkpoint keys.

## 4.2 `Qwen35TextModel`

`Qwen35TextModel` is the checkpoint-compatible text backbone:

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
self.layers = nn.ModuleList([Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.rotary_emb = Qwen35RotaryEmbedding(config)
```

### 4.2.1 Inputs

`forward()` accepts:

```text
input_ids: token ids, shape [batch, seq_len]
attention_mask: optional mask, shape [batch, total_seq_len]
position_ids: optional positions, shape [batch, seq_len]
past_key_values: optional Qwen35DynamicCache
inputs_embeds: optional embeddings, shape [batch, seq_len, hidden_size]
use_cache: whether to create/update cache
```

Exactly one of `input_ids` and `inputs_embeds` must be provided.

### 4.2.2 Embeddings

If `inputs_embeds` is missing:

```python
inputs_embeds = self.embed_tokens(input_ids)
```

Shape:

```text
input_ids:     [batch, seq_len]
inputs_embeds: [batch, seq_len, hidden_size]
```

### 4.2.3 Cache Creation

If `use_cache` is true and no cache is provided:

```python
past_key_values = Qwen35DynamicCache(self.config)
```

The cache stores full-attention KV tensors plus linear-attention convolution and recurrent states.

### 4.2.4 Position IDs

If no `position_ids` are supplied, they are built from the cache length:

```python
past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens
position_ids = position_ids.view(1, -1).expand(batch_size, -1)
```

This matters during generation because after the first pass each next-token call has `seq_len == 1`, but positions must continue from the prompt length.

### 4.2.5 Rotary Embeddings

```python
position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
```

This returns:

```text
cos: [batch, seq_len, rotary_dim]
sin: [batch, seq_len, rotary_dim]
```

The returned tuple is passed into each decoder layer.

### 4.2.6 Attention Masks

If no mask is supplied, the model builds an all-ones mask:

```python
attention_mask = torch.ones((batch_size, seq_len + past_seen_tokens), ...)
```

For full attention, `build_causal_mask()` converts this into a 4D additive mask:

```text
[batch, 1, query_length, kv_length]
```

For linear attention, the original 2D attention mask is passed through because linear attention uses it to zero padded hidden states.

### 4.2.7 Layer Loop

For each layer:

```python
layer_mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
hidden_states = layer(
    hidden_states,
    position_embeddings=position_embeddings,
    attention_mask=layer_mask,
    past_key_values=past_key_values,
)
```

After all layers:

```python
hidden_states = self.norm(hidden_states)
return hidden_states, past_key_values
```

## 4.3 `Qwen35ForCausalLM`

`Qwen35ForCausalLM` wraps `Qwen35TextModel` and turns hidden states into vocabulary logits.

### 4.3.1 Model Selection

The wrapper must use:

```python
self.model = Qwen35TextModel(config)
```

Using `TextModel` will cause Qwen checkpoint key mismatches in full-attention layers.

### 4.3.2 LM Head and Tied Embeddings

If `tie_word_embeddings` is false, the model creates a standard untied head:

```python
nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

If `tie_word_embeddings` is true, `self.lm_head` is `None`. The forward pass computes logits using the token embedding weight directly:

```python
F.linear(hidden_states, self.model.embed_tokens.weight)
```

This avoids allocating a duplicate `[vocab_size, hidden_size]` matrix. For large vocabularies and hidden sizes, that duplicate can be multiple GB.

## 5. Decoder Layers: `decoder.py`

`Qwen35DecoderLayer` represents one transformer block. Its structure is:

```text
input_layernorm
attention or linear_attention
residual add
post_attention_layernorm
MLP
residual add
```

Construction depends on `config.layer_types[layer_idx]`:

```python
if self.layer_type == "linear_attention":
    self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
elif self.layer_type == "full_attention":
    self.self_attn = Qwen35Attention(config, layer_idx)
else:
    raise ValueError(...)
```

Forward path:

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)

if linear_attention:
    hidden_states = self.linear_attn(...)
else:
    hidden_states = self.self_attn(...)

hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
return residual + hidden_states
```

This is a pre-norm decoder block.

## 6. Full Attention: `attention.py`

`Qwen35Attention` implements the checkpoint-compatible full self-attention layer.

### 6.1 Parameters

The important projections are:

```python
q_proj: hidden_size -> num_attention_heads * head_dim * 2
k_proj: hidden_size -> num_key_value_heads * head_dim
v_proj: hidden_size -> num_key_value_heads * head_dim
o_proj: num_attention_heads * head_dim -> hidden_size
```

The query projection has twice the usual output size because it contains both query content and an output gate.

Norms:

```python
q_norm = Qwen35RMSNorm(head_dim)
k_norm = Qwen35RMSNorm(head_dim)
```

### 6.2 Forward Tensor Flow

Input:

```text
hidden_states: [batch, seq_len, hidden_size]
```

Projection:

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
    2,
    dim=-1,
)
```

After chunking:

```text
query_states: [batch, seq_len, num_attention_heads, head_dim]
gate:         [batch, seq_len, num_attention_heads, head_dim]
```

Then `gate` is flattened to:

```text
[batch, seq_len, num_attention_heads * head_dim]
```

Keys and values are projected to KV heads:

```text
key_states:   [batch, seq_len, num_key_value_heads, head_dim]
value_states: [batch, seq_len, num_key_value_heads, head_dim]
```

Query and key states are normalized and transposed:

```text
query_states: [batch, num_attention_heads, seq_len, head_dim]
key_states:   [batch, num_key_value_heads, seq_len, head_dim]
value_states: [batch, num_key_value_heads, seq_len, head_dim]
```

### 6.3 Rotary Embedding Application

RoPE is applied to query and key only:

```python
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

Value states are not rotated.

### 6.4 KV Cache

If a cache is supplied:

```python
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

For prefill, the cache stores the full prompt keys and values. During generation, each next-token call appends one new key and value along the sequence dimension.

### 6.5 Grouped-Query Attention

Qwen 3.5 can have fewer KV heads than query heads. `repeat_kv()` repeats KV heads to match the number of query heads:

```python
key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)
```

If:

```text
num_attention_heads = 8
num_key_value_heads = 2
```

then:

```text
num_key_value_groups = 4
```

Each KV head is repeated four times.

### 6.6 Attention Scores

Attention weights:

```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
```

Shape:

```text
[batch, num_attention_heads, query_len, kv_len]
```

The scaling factor is:

```python
head_dim ** -0.5
```

The additive causal/padding mask is added before softmax.

### 6.7 Output Projection

After softmax and value aggregation:

```text
attn_output: [batch, seq_len, num_attention_heads * head_dim]
```

The gate is applied:

```python
attn_output = attn_output * torch.sigmoid(gate)
```

Then final projection:

```python
return self.o_proj(attn_output)
```

## 7. Linear Attention: `delta.py`

`Qwen35GatedDeltaNet` implements the Qwen 3.5 linear-attention layer.

This layer differs from full attention. It uses:

- a depthwise causal convolution over projected Q/K/V features
- recurrent delta-rule state
- gating via `z`
- per-value-head dynamic parameters `A_log` and `dt_bias`
- a gated RMSNorm before output projection

### 7.1 Parameters

Important dimensions:

```python
key_dim = head_k_dim * num_k_heads
value_dim = head_v_dim * num_v_heads
conv_dim = key_dim * 2 + value_dim
```

Projection layers:

```python
in_proj_qkv: hidden_size -> key_dim + key_dim + value_dim
in_proj_z:   hidden_size -> value_dim
in_proj_b:   hidden_size -> num_v_heads
in_proj_a:   hidden_size -> num_v_heads
out_proj:    value_dim -> hidden_size
```

Convolution:

```python
conv1d = nn.Conv1d(
    in_channels=conv_dim,
    out_channels=conv_dim,
    kernel_size=linear_conv_kernel_size,
    groups=conv_dim,
    bias=False,
    padding=linear_conv_kernel_size - 1,
)
```

Because `groups == conv_dim`, this is depthwise convolution.

### 7.2 Padding Mask Handling

`apply_mask_to_padding_states()` multiplies hidden states by the attention mask. If the mask contains more positions than the current hidden-state sequence, it uses the last `seq_len` entries.

This matters during generation because the full attention mask can contain prompt plus generated tokens, while the current hidden state can contain only the newest token.

### 7.3 Prefill vs Incremental Update

The linear-attention path detects incremental generation with:

```python
use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1
```

If true, it updates convolution state with `torch_causal_conv1d_update()`.

If false, it applies the full convolution over the current sequence and stores a padded convolution state for later single-token updates.

### 7.4 QKV Split

After projection and convolution:

```python
query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
```

Shapes become:

```text
query: [batch, seq_len, num_k_heads, head_k_dim]
key:   [batch, seq_len, num_k_heads, head_k_dim]
value: [batch, seq_len, num_v_heads, head_v_dim]
```

If value heads outnumber key heads, query and key are repeated to match value heads:

```python
query = query.repeat_interleave(rep, dim=2)
key = key.repeat_interleave(rep, dim=2)
```

### 7.5 Delta Rule

The recurrent delta rule maintains a state tensor shaped like:

```text
[batch, num_heads, key_head_dim, value_head_dim]
```

For each token position, it:

1. normalizes query and key with L2 normalization
2. decays the previous recurrent state by `g`
3. reads the current memory with the key
4. computes a delta between current value and memory readout
5. updates the recurrent state
6. reads output using the query

The implementation uses `torch_recurrent_gated_delta_rule()`. The `torch_chunk_gated_delta_rule()` function currently calls the recurrent implementation. It is a placeholder for a future optimized chunked implementation.

### 7.6 Output Gating and Projection

After the delta-rule output:

```python
core_attn_out = self.norm(core_attn_out, z)
```

`Qwen35RMSNormGated` normalizes and applies `F.silu(z)` as a gate.

The tensor is reshaped back to:

```text
[batch, seq_len, value_dim]
```

and projected to hidden size:

```python
return self.out_proj(core_attn_out)
```

## 8. Cache System: `cache.py`

The Qwen 3.5 cache class is `Qwen35DynamicCache`.

### 8.1 Stored State

It stores four lists, each with one entry per layer:

```python
conv_states
recurrent_states
key_cache
value_cache
```

Full-attention layers use `key_cache` and `value_cache`.

Linear-attention layers use `conv_states` and `recurrent_states`.

### 8.2 Full Attention Cache Update

`update(key_states, value_states, layer_idx)` appends new keys and values along sequence dimension `dim=2`:

```python
self.key_cache[layer_idx] = torch.cat([old_key, key_states], dim=2)
self.value_cache[layer_idx] = torch.cat([old_value, value_states], dim=2)
```

The expected cached shape is:

```text
[batch, kv_heads, cached_seq_len, head_dim]
```

### 8.3 Linear Attention Cache Update

Linear attention updates two states:

```python
update_conv_state(conv_state, layer_idx)
update_recurrent_state(recurrent_state, layer_idx)
```

The convolution state stores recent projected QKV features. The recurrent state stores accumulated delta-rule memory.

### 8.4 Sequence Length

`get_seq_length()` returns the sequence length of the first full-attention layer cache. If no full-attention layer has cached keys yet, it returns zero.

This value drives generated token positions.

### 8.5 Beam Reordering

`reorder_cache(beam_idx)` reorders cached tensors along batch dimension. It is useful for beam search, although this repository currently implements only greedy generation.

## 9. Rotary Embeddings: `rope.py`

The Qwen 3.5 path uses `Qwen35RotaryEmbedding`.

### 9.1 Inverse Frequencies

The constructor computes:

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
```

where:

```python
dim = min(config.dim, config.head_dim)
```

`inv_freq` is registered as a non-persistent buffer, so it is moved with the module across devices but is not saved as a checkpoint parameter.

### 9.2 Position ID Shape

If position ids are 2D:

```text
[batch, seq_len]
```

then they are expanded to:

```text
[3, batch, seq_len]
```

This supports the multi-section RoPE mixing logic.

### 9.3 MRoPE Mixing

`apply_interleaved_mrope()` starts from the first frequency stream and replaces interleaved sections from streams 1 and 2.

The default section list is:

```python
[11, 11, 10]
```

This is a simplified implementation of the model's multi-section rotary behavior.

### 9.4 Applying RoPE

`apply_rotary_pos_emb()` splits query/key heads into rotary and pass-through dimensions:

```text
q_rot, q_pass
k_rot, k_pass
```

It rotates only the rotary part and concatenates the pass-through part back.

## 10. Normalization: `norm.py`

There are three normalization modules.

### 10.1 `RMSNorm`

Legacy RMSNorm. Its `_norm()` currently multiplies by `sqrt(mean(x^2) + eps)` rather than multiplying by reciprocal sqrt. The Qwen 3.5 path does not use this for the final current model stack.

### 10.2 `Qwen35RMSNorm`

Qwen 3.5 standard RMSNorm:

```python
out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
out = out * (1.0 + weight)
```

The parameter is initialized to zeros, and the effective scale is `1 + weight`.

### 10.3 `Qwen35RMSNormGated`

Used in linear attention after the delta-rule core:

```python
hidden_states = self.weight * normalized_hidden_states
hidden_states = hidden_states * F.silu(gate)
```

This combines normalization and gate application.

## 11. MLP: `mlp.py`

`Qwen35MLP` is a SwiGLU-style feed-forward network:

```python
return down_proj(act_fn(gate_proj(x)) * up_proj(x))
```

Parameter shapes:

```text
gate_proj: [intermediate_size, hidden_size]
up_proj:   [intermediate_size, hidden_size]
down_proj: [hidden_size, intermediate_size]
```

The activation function is looked up from `ACT2FN` in `utils.py`.

## 12. Utilities: `utils.py`

### 12.1 `ACT2FN`

Maps config activation names to PyTorch functions:

```text
silu -> F.silu
swish -> F.silu
gelU/relu variants as configured
```

### 12.2 `repeat_kv()`

Used for grouped-query attention. It repeats KV heads to match query heads.

Input:

```text
[batch, num_key_heads, seq_len, head_dim]
```

Output:

```text
[batch, num_key_heads * n_rep, seq_len, head_dim]
```

### 12.3 `build_causal_mask()`

Builds a 4D additive attention mask.

Inputs:

```text
attention_mask: [batch, kv_length]
batch_size
query_length
kv_length
device
dtype
```

Output:

```text
[batch, 1, query_length, kv_length]
```

The mask uses `torch.finfo(dtype).min` for blocked positions. It combines:

- causal upper-triangular blocking
- padding-token blocking from `attention_mask`

## 13. Loader: `loader.py`

The loader handles model path resolution, safetensors loading, key filtering, and state-dict loading.

### 13.1 Path Resolution

`resolve_model_path()` checks whether `model_name_or_path` exists locally. If yes, it returns that path. Otherwise, it downloads from Hugging Face with `snapshot_download()`.

### 13.2 Config Loading

`load_config_dict()` reads:

```text
config.json
```

and returns the parsed JSON object.

### 13.3 Checkpoint Loading

`_load_sharded_state_dict()` supports two layouts.

Unindexed safetensors directory:

```text
*.safetensors
```

Indexed sharded directory:

```text
model.safetensors.index.json
model-00001-of-000NN.safetensors
...
```

If an index exists, it loads every unique shard referenced by `index["weight_map"]`.

### 13.4 Prefix Remapping

`remap_state_dict_keys()` maps language-model checkpoint prefixes to the local model path.

Text prefixes:

```text
model.language_model.
language_model.
```

become:

```text
model.
```

Ignored prefixes:

```text
model.visual.
visual.
model.vision
vision.
mtp.
model.mtp.
```

Those weights are excluded because this repo implements the text LM runtime, not vision or MTP modules.

### 13.5 Loading with Non-Strict Mode

`load_weights()` calls:

```python
missing, unexpected = model.load_state_dict(state_dict, strict=False)
```

Non-strict mode is used because the checkpoint may contain ignored modules or tied-embedding differences. The function still returns missing and unexpected keys so the caller can inspect them.

If word embeddings are tied, `lm_head.weight` is filtered from both missing and unexpected lists because it can be represented by `model.embed_tokens.weight` instead of a separate parameter.

## 14. Generation: `generate.py`

`greedy_generate()` is a minimal autoregressive loop.

### 14.1 First Pass

The full prompt is passed into the model:

```python
logits, cache = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
```

This fills caches for all cache-enabled layers.

### 14.2 Token Loop

For each new token:

```python
next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
generated = torch.cat([generated, next_token], dim=1)
```

If an attention mask exists, a one-valued column is appended to it.

Then only the new token is fed back:

```python
model(
    input_ids=next_token,
    attention_mask=attention_mask,
    past_key_values=cache,
    use_cache=True,
)
```

The cache lets the model avoid recomputing full-attention keys and values for earlier tokens.

### 14.3 Output

The function returns the full generated id tensor, including the original prompt ids.

It does not stop on EOS. It always generates exactly `max_new_tokens` tokens.

## 15. State Dict Expectations

For a Qwen 3.5 compatible model, important key families include:

Token embeddings:

```text
model.embed_tokens.weight
```

Full-attention layers:

```text
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.self_attn.q_norm.weight
model.layers.N.self_attn.k_norm.weight
```

Linear-attention layers:

```text
model.layers.N.linear_attn.conv1d.weight
model.layers.N.linear_attn.dt_bias
model.layers.N.linear_attn.A_log
model.layers.N.linear_attn.norm.weight
model.layers.N.linear_attn.out_proj.weight
model.layers.N.linear_attn.in_proj_qkv.weight
model.layers.N.linear_attn.in_proj_z.weight
model.layers.N.linear_attn.in_proj_b.weight
model.layers.N.linear_attn.in_proj_a.weight
```

Layer norms:

```text
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight
model.norm.weight
```

MLP:

```text
model.layers.N.mlp.gate_proj.weight
model.layers.N.mlp.up_proj.weight
model.layers.N.mlp.down_proj.weight
```

LM head, only if embeddings are not tied:

```text
lm_head.weight
```

## 16. Common Debugging Workflow

### 16.1 Check Model Class

Verify that `Qwen35ForCausalLM` uses:

```python
self.model = Qwen35TextModel(config)
```

not:

```python
self.model = TextModel(config)
```

### 16.2 Check Key Names

Run a local key check:

```powershell
$code = @"
from config import Qwen35Config, config as ns
from model import Qwen35ForCausalLM
cfg = Qwen35Config.from_namespace(ns)
model = Qwen35ForCausalLM(cfg)
print([k for k in model.state_dict() if k.startswith('model.layers.1.self_attn')])
"@
$code | python -
```

Expected full-attention keys include `k_proj`, `v_proj`, and `o_proj`.

### 16.3 Check Weight Loading

When running `main.py`, inspect:

```text
missing_keys
unexpected_keys
first_missing
first_unexpected
```

Expected clean text-model loading should have no meaningful missing/unexpected keys. A tied `lm_head.weight` difference is intentionally filtered.

### 16.4 Check RoPE and Mask Shapes

If output is garbled but weights load, add shape prints around:

- `Qwen35TextModel.forward()` position ids and causal mask
- `Qwen35Attention.forward()` query/key/value shapes
- `apply_rotary_pos_emb()` rotary dimension
- `Qwen35DynamicCache.update()` cached sequence length

### 16.5 Check Cache Growth

For full attention, cached sequence length should increase from prompt length to prompt length plus generated tokens.

For linear attention, `has_previous_state` should become true after prefill when at least one linear layer has stored convolution state.

## 17. Known Limitations and Extension Points

### 17.1 Generation Quality

The generator is greedy only. To improve generation quality, add sampling controls:

- temperature
- top-k
- top-p
- repetition penalty
- EOS stopping

### 17.2 Performance

The implementation is intentionally plain PyTorch. It does not use:

- FlashAttention
- fused RMSNorm
- fused RoPE
- optimized causal convolution kernels
- quantized linear layers

This keeps the code readable but slower than production runtimes.

### 17.3 Multimodal Support

The loader ignores vision weights. To support full multimodal Qwen models, the repository would need:

- vision encoder modules
- image/video preprocessing integration
- multimodal token insertion logic
- connector/projection modules
- output contract updates

### 17.4 MTP Support

The loader ignores MTP prefixes. To support MTP, add the corresponding modules and output handling before removing those ignored prefixes.

### 17.5 Testing

Recommended tests to add:

- config parsing tests for nested `text_config`
- state-dict key compatibility tests
- tied vs untied LM head tests
- cache length growth tests
- attention mask shape tests
- generation smoke tests on a tiny synthetic config

## 18. Mental Model Summary

The most important thing to remember is that this repository has two model families:

```text
legacy path: Decoder, SelfAttention, GatedDeltaNet, TextModel
Qwen path:   Qwen35DecoderLayer, Qwen35Attention, Qwen35GatedDeltaNet, Qwen35TextModel
```

For Qwen 3.5 checkpoints, use the Qwen path. The Qwen path matches checkpoint key names, tied embedding behavior, layer type routing, rotary embedding structure, and cache structure expected by the current loader and CLI.