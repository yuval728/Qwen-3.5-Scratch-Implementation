# Qwen 3.5 Minimal PyTorch Runtime

This repository is a small, readable PyTorch implementation for loading and running the text part of a Qwen 3.5 style hybrid language model. It focuses on inference, checkpoint key compatibility, and clarity of the core model components.

The code supports the Qwen 3.5 architecture pattern used by `Qwen/Qwen3.5-2B`, including:

- token embeddings and tied LM head support
- hybrid decoder layers with `linear_attention` and `full_attention`
- full attention with gated query projection, RoPE, KV cache, and grouped-query attention
- gated delta linear attention with convolutional and recurrent state caches
- Qwen-style RMSNorm variants
- safetensors checkpoint loading and Hugging Face snapshot resolution
- greedy token generation

For a deeper file-by-file and tensor-level explanation, see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## Repository Layout

| File | Purpose |
| --- | --- |
| `main.py` | CLI entry point. Resolves/downloads model files, loads config and weights, tokenizes a prompt, runs greedy generation, and prints output. |
| `config.py` | Defines `Qwen35Config` and a tiny sample namespace config for local smoke checks. |
| `model.py` | Top-level model classes: legacy `TextModel`, Qwen 3.5 `Qwen35TextModel`, and `Qwen35ForCausalLM`. |
| `decoder.py` | Decoder block implementations for legacy and Qwen 3.5 paths. |
| `attention.py` | Full self-attention implementations, including checkpoint-compatible `Qwen35Attention`. |
| `delta.py` | Gated delta linear attention implementation and recurrent/chunk update helpers. |
| `cache.py` | Cache containers for full attention KV states and linear attention convolution/recurrent states. |
| `rope.py` | Rotary positional embedding and multi-section RoPE helpers. |
| `norm.py` | RMSNorm implementations, including Qwen 3.5 offset-weight and gated variants. |
| `mlp.py` | SwiGLU-style feed-forward networks. |
| `utils.py` | Activation map, MRoPE frequency mixing, KV repetition, and causal mask construction. |
| `loader.py` | Hugging Face snapshot resolution, config loading, safetensors loading, and checkpoint key remapping. |
| `generate.py` | Simple greedy autoregressive decoding loop. |

## Requirements

The code expects Python 3.11+ and these Python packages:

```bash
pip install torch transformers huggingface_hub safetensors
```

A CUDA-enabled PyTorch install is recommended for real Qwen 3.5 checkpoints. CPU inference for the 2B model can require several GB of RAM and will be slow.

## Quick Start

Run the default model and prompt:

```bash
python main.py
```

Run with an explicit Hugging Face model id:

```bash
python main.py --model-id Qwen/Qwen3.5-2B --prompt "Hello from Qwen3.5" --max-new-tokens 16 --device cuda
```

Run from a local model directory:

```bash
python main.py --model-id C:\path\to\Qwen3.5-2B --prompt "Explain transformers briefly" --max-new-tokens 32 --device cuda
```

The local directory must contain at least:

- `config.json`
- one or more `*.safetensors` files, or `model.safetensors.index.json` plus all referenced shard files
- tokenizer files such as `tokenizer.json`, `tokenizer_config.json`, and related files

## CLI Options

`main.py` exposes these arguments:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model-id` | `Qwen/Qwen3.5-2B` | Hugging Face repo id or local model directory. |
| `--prompt` | `Hello from Qwen3.5` | Input text passed to the tokenizer. |
| `--max-new-tokens` | `8` | Number of greedy decoding steps. |
| `--device` | `cpu` | PyTorch device string, for example `cpu`, `cuda`, or `cuda:0`. |

## What Happens During Startup

1. `main.py` calls `resolve_model_path()` in `loader.py`.
2. If `--model-id` is an existing local path, that path is used directly.
3. Otherwise, `huggingface_hub.snapshot_download()` downloads a filtered snapshot with config, safetensors, tokenizer, processor, and text metadata files.
4. `config.json` is parsed into `Qwen35Config`.
5. `Qwen35ForCausalLM` is constructed.
6. `load_weights()` loads all safetensors shards, remaps language-model prefixes, ignores vision/MTP weights, and calls `load_state_dict(..., strict=False)`.
7. The tokenizer encodes the prompt.
8. `greedy_generate()` repeatedly takes `argmax` over the last-token logits and feeds the selected token back into the model with cache enabled.

## Checkpoint Compatibility

The Qwen 3.5 path uses checkpoint-compatible module names. Full-attention layers expose keys like:

```text
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
```

This matters because the legacy implementation uses shorter names such as `k`, `v`, and `proj_out`. Loading a Qwen 3.5 checkpoint into the legacy model produces missing and unexpected keys such as:

```text
missing:    model.layers.3.self_attn.k.weight
unexpected: model.layers.3.self_attn.k_proj.weight
```

`Qwen35ForCausalLM` now constructs `Qwen35TextModel`, which is the intended checkpoint-compatible model path.

## Tied Embeddings

Qwen 3.5 configs can set:

```json
"tie_word_embeddings": true
```

When this is true, the LM head shares the token embedding matrix. This repository avoids allocating a second full vocabulary projection matrix in that case. Instead, `Qwen35ForCausalLM.forward()` computes logits with:

```python
F.linear(hidden_states, self.model.embed_tokens.weight)
```

That saves memory and avoids a redundant `lm_head.weight` parameter.

## Generation Behavior

Generation is intentionally simple:

- no sampling
- no temperature
- no top-k or top-p
- no repetition penalty
- no early stop on EOS
- greedy `argmax` only

This makes the runtime easier to debug, but it is not a production text-generation stack.

## Validation Commands

Syntax check:

```bash
python -m compileall .
```

Small local key-shape sanity check:

```powershell
$code = @"
from config import Qwen35Config, config as ns
from model import Qwen35ForCausalLM

cfg = Qwen35Config.from_namespace(ns)
cfg.tie_word_embeddings = True
model = Qwen35ForCausalLM(cfg)
keys = list(model.state_dict())

assert type(model.model).__name__ == 'Qwen35TextModel'
assert 'lm_head.weight' not in keys
assert 'model.layers.1.self_attn.k_proj.weight' in keys
assert 'model.layers.1.self_attn.k.weight' not in keys
assert 'model.layers.1.self_attn.o_proj.weight' in keys
assert 'model.layers.1.self_attn.proj_out.weight' not in keys
print('ok')
"@
$code | python -
```

## Troubleshooting

### Missing and unexpected attention keys

If you see keys like this:

```text
missing_keys ... self_attn.k.weight, self_attn.v.weight, self_attn.proj_out.weight
unexpected_keys ... self_attn.k_proj.weight, self_attn.v_proj.weight, self_attn.o_proj.weight
```

then the Qwen checkpoint is being loaded into the legacy attention implementation. Make sure `Qwen35ForCausalLM` constructs `Qwen35TextModel`, not `TextModel`.

### Out of memory during model construction

If the failure happens while creating `lm_head`, check whether `tie_word_embeddings` is true. In tied mode, the model should not allocate a separate `nn.Linear(hidden_size, vocab_size)` head.

### Hugging Face download or proxy failures

`resolve_model_path()` uses `snapshot_download()` when `--model-id` is not a local path. If the network is blocked, pass a local model directory instead:

```bash
python main.py --model-id C:\path\to\downloaded\model
```

### Zero-length files in the Hugging Face cache

A cache directory can contain incomplete or pointer-like files if a previous download failed. If `*.safetensors` files are zero bytes, redownload the snapshot or use a verified local model directory.

### Garbled output

Garbled output after clean weight loading usually means one of these is wrong:

- wrong model class or checkpoint key mapping
- incorrect cache update behavior
- incorrect RoPE dimensions or frequency mixing
- attention mask shape mismatch
- incomplete or corrupt checkpoint files
- tokenizer mismatch

Start by checking that missing and unexpected key counts are zero, except for intentionally filtered tied embedding keys.

## Current Scope and Limitations

This project is an inference-focused educational/runtime implementation. It does not currently include:

- training
- gradient checkpointing
- batched sampling utilities
- beam search
- quantization
- FlashAttention or fused kernels
- full multimodal vision path
- MTP head execution
- robust unit test suite
- packaged dependency metadata such as `requirements.txt` or `pyproject.toml`

The loader intentionally ignores vision and MTP checkpoint prefixes because this runtime implements the text language model path.