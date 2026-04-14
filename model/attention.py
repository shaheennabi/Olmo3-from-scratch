import torch
import torch.nn as nn

from model.norm import RMSNorm
from model.rope import RoPE


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        d_in,
        num_heads,
        num_kv_groups,
        head_dim,
        attention_bias=False,
        dtype=None,
        sliding_window=None,
        attn_type="full_attention",
    ):
        super().__init__()

        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim

        self.d_out = num_heads * head_dim

        self.attn_type = attn_type
        self.sliding_window = (
            sliding_window if attn_type == "sliding_window" else None
        )

        # ---------------- Projections ----------------

        self.W_query = nn.Linear(
            d_in, self.d_out, bias=attention_bias, dtype=dtype
        )

        self.W_key = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=attention_bias, dtype=dtype
        )

        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=attention_bias, dtype=dtype
        )

        self.out_proj = nn.Linear(
            self.d_out, d_in, bias=attention_bias, dtype=dtype
        )

        # ---------------- Norms ----------------

        self.q_norm = RMSNorm(self.d_out)
        self.k_norm = RMSNorm(num_kv_groups * head_dim)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """
        x:    [B, T, D]
        mask: [B, 1 or H, T_q, T_k]
        cos/sin: [max_seq_len, head_dim]
        """

        B, T, _ = x.shape

        # --------------------------------------------------
        # Projections
        # --------------------------------------------------

        q = self.W_query(x)   # [B, T, H*Dh]
        k = self.W_key(x)     # [B, T, G*Dh]
        v = self.W_value(x)   # [B, T, G*Dh]

        # --------------------------------------------------
        # Norm
        # --------------------------------------------------

        q = self.q_norm(q)
        k = self.k_norm(k)

        # --------------------------------------------------
        # Reshape → heads
        # --------------------------------------------------

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, H, T, Dh]

        k_new = k.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # [B, G, T, Dh]

        v_new = v.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # [B, G, T, Dh]

        # --------------------------------------------------
        # KV Cache (UNROTATED storage)
        # --------------------------------------------------

        prev_len = 0

        if cache is not None and cache[0] is not None:
            prev_k, prev_v = cache
            prev_len = prev_k.size(2)

            k_cat = torch.cat([prev_k, k_new], dim=2)
            v_cat = torch.cat([prev_v, v_new], dim=2)
        else:
            k_cat = k_new
            v_cat = v_new

        # --------------------------------------------------
        # Ensure dtype/device alignment
        # --------------------------------------------------

        cos = cos.to(x.device, dtype=x.dtype)
        sin = sin.to(x.device, dtype=x.dtype)

        # --------------------------------------------------
        # RoPE
        # --------------------------------------------------

        # queries → current positions
        q = RoPE.apply_rope(q, cos, sin, offset=start_pos)

        # keys → absolute positions (IMPORTANT FIX)
        k = RoPE.apply_rope(k_cat, cos, sin, offset=0)

        # --------------------------------------------------
        # Expand KV → match heads (GQA)
        # --------------------------------------------------

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v_cat.repeat_interleave(self.group_size, dim=1)
        else:
            v = v_cat

        # --------------------------------------------------
        # Scale queries
        # --------------------------------------------------

        q = q * (self.head_dim ** -0.5)

        # --------------------------------------------------
        # Update cache (reuse concatenated tensors)
        # --------------------------------------------------

        next_cache = (k_cat, v_cat)

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        # [B, H, T_q, T_k]
        attn_scores = q @ k.transpose(2, 3)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # [B, H, T_q, Dh]
        context = attn_weights @ v

        # --------------------------------------------------
        # Merge heads
        # --------------------------------------------------

        context = context.transpose(1, 2).reshape(B, T, self.d_out)

        # --------------------------------------------------
        # Output projection
        # --------------------------------------------------

        out = self.out_proj(context)

        return out, next_cache