import torch
import torch.nn as nn
from model.norm import RMSNorm
from model.transformer import TransformerBlock
from model.rope import RoPE


class Olmo3(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # --- Sanity: one attention type per layer ---
        assert cfg["layer_types"] is not None
        assert len(cfg["layer_types"]) == cfg["n_layers"]

        # ============================================================
        # Token embedding
        # input_ids: [B, T]
        # output:    [B, T, emb_dim]
        # ============================================================
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
            dtype=cfg["dtype"]
        )

        # ============================================================
        # Transformer blocks (stack)
        # Each block preserves shape: [B, T, emb_dim]
        # ============================================================
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, attn_type)
            for attn_type in cfg["layer_types"]
        ])

        # ============================================================
        # Final normalization (RMSNorm)
        # Input:  [B, T, emb_dim]
        # Output: [B, T, emb_dim]
        # ============================================================
        self.final_norm = RMSNorm(
            cfg["emb_dim"],
            eps=cfg["rms_norm_eps"]
        )

        # ============================================================
        # Output projection → vocabulary logits
        # Input:  [B, T, emb_dim]
        # Output: [B, T, vocab_size]
        # ============================================================
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
            dtype=cfg["dtype"]
        )

        self.cfg = cfg

        # ============================================================
        # Stateful decoding pointer
        # Tracks absolute position during KV-cache decoding
        # ============================================================
        self.current_pos = 0

        # ============================================================
        # Precompute RoPE tables
        # cos/sin: [max_seq_len, head_dim]
        # Stored as buffers (not trainable, move with device)
        # ============================================================
        cos, sin = RoPE.compute_rope_parameters(
            head_dim=cfg["head_dim"],
            context_length=cfg["context_length"],
            theta_base=cfg["rope_base"],
            attention_factor=cfg["rope_attention_factor"],
            rope_type=cfg["rope_type"],
            rope_factor=cfg["rope_factor"],
            rope_orig_max=cfg["rope_orig_max"],
            dtype=torch.float32,
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # ================================================================
    # MASK CREATION
    # ================================================================
    def create_masks(self, cur_len, device, pos_start=0, pos_end=None):

        """
        cur_len:   number of current tokens (query length)
        pos_start: starting absolute position in sequence
        pos_end:   total sequence length after adding current tokens

        Returns:
            mask_global: [1, 1, cur_len, total_len]
            mask_local:  [1, 1, cur_len, total_len]
        """

        if pos_end is None:
            pos_end = cur_len

        # total sequence length (past + current)
        total_len = pos_end

        # ============================================================
        # Full square mask
        # shape: [total_len, total_len]
        # ============================================================
        ones = torch.ones(
            (total_len, total_len),
            dtype=torch.bool,
            device=device
        )

        # ============================================================
        # GLOBAL CAUSAL MASK
        # True = masked (cannot attend)
        # Upper triangular (future tokens)
        # ============================================================
        mask_global_full = torch.triu(ones, diagonal=1)

        # ============================================================
        # LOCAL MASK (sliding window constraint)
        # Masks tokens that are too far in the past
        # ============================================================
        far_past_full = torch.triu(
            ones,
            diagonal=self.cfg["sliding_window"]
        )

        # Combine:
        # - future tokens
        # - far past tokens
        mask_local_full = mask_global_full | far_past_full

        # ============================================================
        # Slice only relevant queries (rows)
        # rows   = current tokens [pos_start : pos_end]
        # cols   = all tokens     [0 : pos_end]
        # ============================================================
        row_slice = slice(pos_start, pos_end)

        # shape before unsqueeze:
        # [cur_len, total_len]
        mask_global = mask_global_full[row_slice, :pos_end]
        mask_local  = mask_local_full[row_slice, :pos_end]

        # ============================================================
        # Expand for attention:
        # [B, heads, Q, K]
        # → broadcast over batch and heads
        # ============================================================
        mask_global = mask_global[None, None, :, :]
        mask_local  = mask_local[None, None, :, :]

        return mask_global, mask_local

    # ================================================================
    # FORWARD PASS
    # ================================================================
    def forward(self, input_ids, cache=None):

        """
        input_ids: [B, T]
        cache:     dict(layer_id → KV tensors) OR None

        Returns:
            logits: [B, T, vocab_size]
        """

        b, seq_len = input_ids.shape

        # ============================================================
        # Token embedding
        # [B, T] → [B, T, emb_dim]
        # ============================================================
        x = self.tok_emb(input_ids)

        # ============================================================
        # POSITION + MASK HANDLING
        # ============================================================
        if cache is not None:
            # Inference mode (incremental decoding)

            pos_start = self.current_pos
            pos_end   = pos_start + seq_len

            # Update global pointer
            self.current_pos = pos_end

            # Masks consider full context (past + current)
            mask_global, mask_local = self.create_masks(
                cur_len=seq_len,
                device=x.device,
                pos_start=pos_start,
                pos_end=pos_end
            )

        else:
            # Training mode (full sequence)

            pos_start = 0
            pos_end   = seq_len

            mask_global, mask_local = self.create_masks(
                cur_len=seq_len,
                device=x.device,
                pos_start=pos_start,
                pos_end=pos_end
            )

        # ============================================================
        # RoPE tables (shared across layers)
        # ============================================================
        cos = self.cos   # [max_seq_len, head_dim]
        sin = self.sin   # [max_seq_len, head_dim]

        # ============================================================
        # Transformer stack
        # ============================================================
        for i, block in enumerate(self.blocks):

            # Retrieve per-layer KV cache
            blk_cache = cache.get(i) if cache is not None else None

            # Forward through block
            # x: [B, T, emb_dim]
            x, new_blk_cache = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos=cos,
                sin=sin,
                start_pos=pos_start,   # critical for RoPE alignment
                cache=blk_cache,
            )

            # Update cache (append new KV)
            if cache is not None:
                if isinstance(cache, dict):
                    cache[i] = new_blk_cache
                else:
                    cache.update(i, new_blk_cache)
        # ============================================================
        # Final normalization + projection
        # ============================================================
        x = self.final_norm(x)  # [B, T, emb_dim]

        logits = self.out_head(
            x.to(self.cfg["dtype"])
        )  # [B, T, vocab_size]

        return logits

    # ================================================================
    # RESET STATE (NEW SEQUENCE)
    # ================================================================
    def reset_kv_cache(self):
        """
        Must be called before starting a new independent sequence.
        Resets positional state (and externally, KV cache should be cleared).
        """
        self.current_pos = 0