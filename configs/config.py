import torch

# ============================================================
# Select model
# ============================================================

USE_MODEL = "Olmo-3-7B-Instruct"   # change to "32B" if needed


# ============================================================
# Olmo3 Configuration (7B)
# ============================================================

OLMO3_CONFIG_7B = {
    # ---------------- Token / sequence ----------------
    "vocab_size": 100_278,
    "context_length": 65_536,

    # ---------------- Model dims ----------------
    "emb_dim": 4096,
    "n_heads": 32,
    "head_dim": 128,
    "n_kv_heads": 32,
    "n_layers": 32,
    "hidden_dim": 11008,

    # ---------------- Attention ----------------
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": 4096,

    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    ] * 8,

    # ---------------- RoPE ----------------
    "rope_base": 500_000.0,
    "rope_type": "yarn",
    "rope_factor": 8.0,
    "rope_orig_max": 8192,
    "rope_attention_factor": 1.2079441541679836,
    "beta_fast": 32.0,
    "beta_slow": 1.0,

    # ---------------- Norm ----------------
    "rms_norm_eps": 1e-6,

    # ---------------- Misc ----------------
    "dtype": torch.bfloat16,
    "eos_token_id": 100_257,
    "pad_token_id": 100_277,
}


# ============================================================
# Olmo3 Configuration (32B)
# ============================================================

OLMO3_CONFIG_32B = {
    # ---------------- Token / sequence ----------------
    "vocab_size": 100_278,
    "context_length": 65_536,

    # ---------------- Model dims ----------------
    "emb_dim": 5120,
    "n_heads": 40,
    "head_dim": 128,
    "n_kv_heads": 8,
    "n_layers": 64,
    "hidden_dim": 27648,

    # ---------------- Attention ----------------
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": 4096,

    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    ] * 16,

    # ---------------- RoPE ----------------
    "rope_base": 500_000.0,
    "rope_type": "yarn",
    "rope_factor": 8.0,
    "rope_orig_max": 8192,
    "rope_attention_factor": 1.2079441541679836,
    "beta_fast": 32.0,
    "beta_slow": 1.0,

    # ---------------- Norm ----------------
    "rms_norm_eps": 1e-6,

    # ---------------- Misc ----------------
    "dtype": torch.bfloat16,
    "eos_token_id": 100_257,
    "pad_token_id": 100_277,
}


# ============================================================
# Model selection
# ============================================================

OLMO3_CONFIG = (
    OLMO3_CONFIG_32B if "32" in USE_MODEL else OLMO3_CONFIG_7B
)