## Architecture  
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # cfg:
        # emb_dim    = d_model
        # hidden_dim = d_ff (typically ~4 * d_model or 8/3 * d_model for SwiGLU)
        # dtype      = torch.float16 / bfloat16 / float32

        self.fc1 = nn.Linear(
            cfg['emb_dim'], 
            cfg['hidden_dim'], 
            dtype=cfg["dtype"], 
            bias=False
        )  # (d_model → d_ff)

        self.fc2 = nn.Linear(
            cfg['emb_dim'], 
            cfg['hidden_dim'], 
            dtype=cfg['dtype'], 
            bias=False
        )  # (d_model → d_ff)  <-- gating branch

        self.fc3 = nn.Linear(
            cfg['hidden_dim'], 
            cfg['emb_dim'], 
            dtype=cfg['dtype'], 
            bias=False
        )  # (d_ff → d_model)

    def forward(self, x): 
        """
        x: (batch_size, seq_len, emb_dim)
           = (B, T, d_model)
        """

        # -------- First projections --------
        x_fc1 = self.fc1(x)
        # (B, T, d_model) → (B, T, d_ff)

        x_fc2 = self.fc2(x)
        # (B, T, d_model) → (B, T, d_ff)

        # -------- SwiGLU activation --------
        # silu(x_fc1): (B, T, d_ff)
        # elementwise multiply with x_fc2
        x = F.silu(x_fc1) * x_fc2
        # (B, T, d_ff) ⊙ (B, T, d_ff) → (B, T, d_ff)

        # -------- Projection back --------
        out = self.fc3(x)
        # (B, T, d_ff) → (B, T, d_model)

        return out
        # final: (B, T, d_model)