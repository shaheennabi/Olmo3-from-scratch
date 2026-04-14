## RMSNorm 

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        # weight: [D]  (per-feature scaling)
        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x):
        """
        x: [B, T, D]
           B = batch size
           T = sequence length (e.g., 14 tokens)
           D = embedding dim (e.g., 64)
        """

        input_dtype = x.dtype

        # ---- Step 1: cast to float32 for numerical stability ----
        x_f = x.float()
        # x_f: [B, T, D]

        # ---- Step 2: compute mean square (RMS denominator) ----
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        """
        x_f.pow(2): [B, T, D]
        mean over last dim (D):

        var: [B, T, 1]

        Each token now has ONE scalar = mean(x_i^2)
        """

        # ---- Step 3: normalize ----
        x_norm = x_f * torch.rsqrt(var + self.eps)
        """
        rsqrt(var): [B, T, 1]

        broadcast multiply:
        [B, T, D] * [B, T, 1] → [B, T, D]

        x_norm: [B, T, D]
        """

        # ---- Step 4: scale with learned weight ----
        out = self.weight * x_norm
        """
        weight: [D]

        broadcast:
        [D] → [1, 1, D]

        [B, T, D] * [1, 1, D] → [B, T, D]
        """

        # ---- Step 5: cast back to original dtype ----
        return out.to(input_dtype)
        # final output: [B, T, D]