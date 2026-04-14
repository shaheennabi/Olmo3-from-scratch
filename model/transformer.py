import torch.nn as nn

from model.norm import RMSNorm
from model.attention import GroupQueryAttention
from model.mlp import FeedForward

## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg, attn_type):
        super().__init__()

        self.attn_type = attn_type                          # "sliding_attention" or "global_attention"
        self.sliding_window = cfg['sliding_window']         # int: max tokens to keep in cache

        # ---------------- ATTENTION ----------------
        # Input / Output:
        # x: (B, T, D)
        # → attention → (B, T, D)
        self.att = GroupQueryAttention(
            d_in=cfg['emb_dim'],           # D
            num_heads=cfg['n_heads'],      # number of query heads
            num_kv_groups=cfg['n_kv_heads'],  # KV heads (GQA)
            head_dim=cfg['head_dim'],      # per-head dimension
            attention_bias=cfg['attention_bias'],
            dtype=cfg['dtype'],
            sliding_window=cfg['sliding_window'],
            attn_type=attn_type,
        )

        # ---------------- FEEDFORWARD ----------------
        # (B, T, D) → (B, T, D)
        self.ff = FeedForward(cfg)

        # ---------------- NORMALIZATION ----------------
        # RMSNorm keeps shape same: (B, T, D)
        self.post_attention_layernorm = RMSNorm(cfg['emb_dim'], eps=cfg["rms_norm_eps"])
        self.post_feedforward_layernorm = RMSNorm(cfg['emb_dim'], eps=cfg["rms_norm_eps"])



    def forward(self, x, mask_global, mask_local, cos, sin, start_pos=0, cache=None):
        """
        x:              (B, T, D)                ← current tokens
        mask_global:    (1, 1, T, T)             ← full causal mask
        mask_local:     (1, 1, max_seq, max_seq) ← sliding window mask (precomputed)
        cache:
            k: (B, kv_heads, T_past, head_dim)
            v: (B, kv_heads, T_past, head_dim)
        """

        # ---------------- RESIDUAL SAVE ----------------
        shortcut = x                                   # (B, T, D)

        # ============================================================
        # ---------------- MASK SELECTION LOGIC -----------------------
        # ============================================================

        if self.attn_type == "sliding_attention":

            # --------- STEP 1: get past sequence length ----------
            if cache is not None and isinstance(cache, tuple):
                prev_k, _ = cache                      # (B, kv_heads, T_past, head_dim)

                # extract past token length
                prev_len = prev_k.size(2) if prev_k is not None else 0   # scalar
            else:
                prev_len = 0                           # no cache → no past tokens


            # --------- STEP 2: total tokens (past + current) ----------
            # x.size(1) = T (current tokens)
            eff_kv_len = prev_len + x.size(1)          # scalar

            # Example:
            # prev_len = 3, current T = 2 → eff_kv_len = 5
            # total tokens = [t1 t2 t3 t4 t5]


            # --------- STEP 3: slice mask to match actual tokens ----------
            # mask_local: (1, 1, max_seq, max_seq)
            # after slicing:
            # attn_mask: (1, 1, max_seq, eff_kv_len)
            attn_mask = mask_local[..., -eff_kv_len:]

            # This ensures mask matches number of keys (tokens)


        else:
            # --------- GLOBAL ATTENTION ----------
            # mask_global: (1, 1, T, T)
            attn_mask = mask_global



        # ============================================================
        # ---------------- ATTENTION COMPUTATION ----------------------
        # ============================================================

        # Input:
        # x: (B, T, D)
        # attn_mask: (1, 1, ?, eff_kv_len)
        # cache used inside attention

        x_attn, next_cache = self.att(
            x,
            attn_mask,
            cos,
            sin,
            start_pos=start_pos,
            cache=cache
        )

        # Output:
        # x_attn: (B, T, D)
        # next_cache:
        #   k: (B, kv_heads, T_total, head_dim)
        #   v: (B, kv_heads, T_total, head_dim)



        # ============================================================
        # ---------------- CACHE TRUNCATION ---------------------------
        # ============================================================

        if next_cache is not None and self.attn_type == "sliding_attention":
            k, v = next_cache                          # (B, kv_heads, T_total, head_dim)

            # If cache grows beyond window → truncate
            if k.size(2) > self.sliding_window:
                # keep only last W tokens
                k = k[:, :, -self.sliding_window:, :]  # (B, kv_heads, W, head_dim)
                v = v[:, :, -self.sliding_window:, :]

            next_cache = (k, v)  ## only-hold this much cache



        # ============================================================
        # ---------------- POST-ATTENTION BLOCK -----------------------
        # ============================================================

        # RMSNorm
        x_attn = self.post_attention_layernorm(x_attn)   # (B, T, D)

        # Residual connection
        x = shortcut + x_attn                           # (B, T, D)



        # ============================================================
        # ---------------- FEEDFORWARD BLOCK --------------------------
        # ============================================================

        shortcut = x                                    # (B, T, D)

        x_fnn = self.ff(x)                              # (B, T, D)

        x_fnn = self.post_feedforward_layernorm(x_fnn)   # (B, T, D)

        # Residual
        x = shortcut + x_fnn                            # (B, T, D)



        # ============================================================
        # ---------------- FINAL OUTPUT -------------------------------
        # ============================================================

        return x, next_cache
    


