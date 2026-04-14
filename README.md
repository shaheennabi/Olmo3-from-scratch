# Olmo-3 End-to-End Implementation

A complete PyTorch implementation of AllenAI's Olmo-3 language model series, supporting autoregressive text generation with efficient KV caching.

## Architecture Overview

## Architecture Flow

```text
x (input)                         # [B, T, D]
│
├── RMSNorm --------------------→ x_norm1
│                                # [B, T, D]
│
├── Multi-Head Attention
│     ├── Q, K, V projections
│     ├── RoPE applied to Q, K
│     ├── KV cache (append/read)
│     ├── causal masking
│     └── output projection
│
├── Residual Add --------------→ x = x + attn_out
│
├── RMSNorm --------------------→ x_norm2
│
├── MLP (SwiGLU/GEGLU)
│     ├── up projection
│     ├── gated activation
│     └── down projection
│
└── Residual Add --------------→ x = x + mlp_out

OUTPUT: x                        # [B, T, D]
```





This implementation features a modern transformer architecture with the following key components:

### Core Components
- **Grouped Query Attention (GQA)**: Reduces KV cache size by sharing keys/values across multiple query heads
- **Rotary Position Embeddings (RoPE)**: Uses YaRN scaling for extended context windows
- **Sliding Window Attention**: Local attention with periodic global refresh layers
- **SwiGLU Feed-Forward Networks**: Efficient gating mechanism in MLP layers
- **RMSNorm**: Stable normalization without learnable biases

### Model Variants
| Variant | Parameters | Layers | Heads | KV Heads | Hidden Dim | Context |
|---------|------------|--------|-------|----------|------------|---------|
| Olmo-3-7B | 7B | 32 | 32 | 32 | 11,008 | 65,536 |
| Olmo-3-32B | 32B | 64 | 40 | 8 | 27,648 | 65,536 |

## Inference Pipeline

The inference pipeline supports efficient autoregressive text generation:

1. **Tokenization**: Input text → token IDs using the model's tokenizer
2. **Initial Forward Pass**: Process prompt tokens to build initial KV cache
3. **Autoregressive Generation**:
   - Generate one token at a time
   - Use KV cache to avoid recomputing past tokens
   - Apply RoPE with position offsets for cache compatibility
   - Update cache with new key/value pairs
4. **Detokenization**: Generated token IDs → output text

### Key Optimizations
- **KV Cache Management**: Stores unrotated keys/values for efficient memory usage
- **Sliding Window**: Limits attention to recent tokens while maintaining global context
- **Mixed Precision**: Uses bfloat16 for computation and storage

## KV Cache Implementation

The KV cache is critical for efficient autoregressive inference:

```python
class KvCache:
    def __init__(self, n_layers: int):
        self.cache = [None] * n_layers  # One entry per layer

    def update(self, layer_idx: int, value):
        # value = (K, V) with shapes [B, H_kv, T_total, D]
        self.cache[layer_idx] = value
```

### Cache Operation Flow
1. **Retrieve Past KV**: `past_k, past_v = kv_cache.get(layer_idx)`
2. **Concatenate**: `keys_cat = torch.cat([past_k, keys_new], dim=2)`
3. **Apply RoPE**: Rotate new keys/queries with position offset
4. **Attention**: Compute attention over concatenated sequence
5. **Update Cache**: Store extended KV for next step

### Memory Benefits
- **GQA Reduction**: KV heads << query heads (8 vs 32/40)
- **Sliding Window**: Limited context per layer type
- **Incremental Updates**: Cache grows linearly with sequence length

## Repository Structure

```
├── configs/
│   └── config.py              # Model configuration parameters
├── inference/
│   ├── generate.py            # Text generation utilities
│   ├── kv_cache.py            # KV cache implementation
│   └── __init__.py
├── model/
│   ├── attention.py           # Multi-head attention with GQA
│   ├── block.py               # Transformer block
│   ├── embeddings.py          # Token embeddings + RoPE
│   ├── mlp.py                 # SwiGLU feed-forward network
│   ├── norm.py                # RMSNorm implementation
│   ├── rope.py                # YaRN RoPE implementation
│   └── transformer.py         # Main transformer model
├── notebooks/
│   └── olmo3.ipynb            # Complete implementation notebook
├── utils/
│   ├── device.py              # Device management utilities
│   ├── logging.py             # Logging configuration
│   └── seed.py                # Random seed management
├── weights/
│   └── load_weights.py        # Weight loading from HuggingFace
├── run_inference.py           # Inference script entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## How to Run

### Prerequisites
```bash
pip install torch safetensors huggingface-hub tokenizers
```

### Basic Inference
```python
from notebooks.olmo3 import Olmo3, KvCache, generate_text_basic_stream

# Load model (7B or 32B variant)
USE_MODEL = "Olmo-3-7B-Instruct"
model = Olmo3(OLMO3_CONFIG)
load_weights_into_olmo(model, OLMO3_CONFIG, weights_dict)

# Initialize KV cache
kv_cache = KvCache(model.cfg["n_layers"])

# Generate text
prompt = "The future of AI is"
token_ids = tokenizer.encode(prompt)

for token in generate_text_basic_stream(
    model, token_ids, max_new_tokens=100,
    eos_token_id=OLMO3_CONFIG["eos_token_id"]
):
    print(tokenizer.decode([token]), end="")
```

### Model Loading
```python
# Download weights from HuggingFace
from huggingface_hub import snapshot_download
repo_id = f"allenai/{'7b' if '7B' in USE_MODEL else '32b'}"
weights_path = snapshot_download(repo_id)
```

## Implementation Notes

### Key Design Decisions
- **GQA over MHA**: Reduces KV cache memory by 75% (32→8 heads for 7B, 40→8 for 32B)
- **Sliding Window Pattern**: 3:1 ratio of sliding to full attention layers balances efficiency and context
- **YaRN RoPE Scaling**: Enables 8x context extension (8K→65K) with minimal quality degradation
- **SwiGLU Activation**: Improves parameter efficiency in feed-forward networks
- **No Attention Bias**: Follows modern transformer best practices

### Technical Details
- **Precision**: bfloat16 throughout for memory efficiency
- **Normalization**: RMSNorm with ε=1e-6 for numerical stability
- **Position Embeddings**: RoPE applied to queries and keys only (values unchanged)
- **Cache Storage**: Unrotated KV pairs to maintain RoPE compatibility

## Limitations

- **Memory Scaling**: KV cache grows linearly with sequence length
- **Sliding Window**: Local attention may miss long-range dependencies
- **Single GPU**: No distributed inference support
- **No Quantization**: Full precision weights required
- **Training Not Included**: Inference-only implementation

## Next Steps

### Short Term
- **Quantization**: Add 4-bit/8-bit weight quantization for reduced memory
- **Distributed Inference**: Multi-GPU support for larger models
- **Batch Processing**: Parallel generation for multiple prompts
- **Performance Profiling**: Memory and latency optimizations

### Long Term
- **Training Pipeline**: Add pre-training and fine-tuning capabilities
- **Model Parallelism**: Tensor/model parallelism for 100B+ scale
- **Custom Architectures**: Experiment with alternative attention mechanisms
- **Multi-Modal Extensions**: Vision-language model variants

## Example Prompt

```python
prompt = """You are Olmo-3, a helpful and truthful AI assistant built by AllenAI.
Answer the following question concisely and accurately:

What are the key differences between transformers and convolutional neural networks?"""

# Generate response with temperature=0.7, max_tokens=200
response = generate_text(model, tokenizer.encode(prompt),
                        max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(response))
```

Expected output demonstrates the model's reasoning capabilities and factual knowledge about neural architectures.
