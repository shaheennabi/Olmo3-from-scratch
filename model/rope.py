import math
import torch


class RoPE:
    """
    Rotary Positional Embedding (RoPE)

    Handles:
    - Precomputation of cos/sin tables
    - Application to Q/K tensors
    """

    @staticmethod
    def compute_rope_parameters(
        head_dim: int,
        context_length: int,
        theta_base: float = 10000.0,
        attention_factor: float = 1.0,
        dtype=torch.float32,
    ):
        """
        Precompute RoPE cos/sin tables.

        Args:
            head_dim: dimension per head (must be even)
            context_length: max sequence length
            theta_base: base frequency
        Returns:
            cos, sin: [context_length, head_dim]
        """

        assert head_dim % 2 == 0, "head_dim must be even"

        # [num_freqs]
        inv_freq = 1.0 / (
            theta_base ** (
                torch.arange(0, head_dim, 2, dtype=dtype) / head_dim
            )
        )

        # [T]
        positions = torch.arange(context_length, dtype=dtype)

        # [T, num_freqs]
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

        # [T, D]
        angles = torch.cat([angles, angles], dim=1)

        cos = torch.cos(angles) * attention_factor
        sin = torch.sin(angles) * attention_factor

        return cos, sin

    @staticmethod
    def apply_rope(x, cos, sin, offset: int = 0):
        """
        Apply RoPE rotation.

        Args:
            x:   [B, H, T, D]
            cos: [context_len, D]
            sin: [context_len, D]
            offset: position offset (for KV cache decoding)

        Returns:
            x_rotated: [B, H, T, D]
        """

        B, H, T, D = x.shape
        assert D % 2 == 0

        # Split into pairs
        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]

        # Select correct positions
        cos = cos[offset : offset + T, :]   # [T, D]
        sin = sin[offset : offset + T, :]

        # Broadcast → [1, 1, T, D]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Perpendicular rotation
        rotated = torch.cat((-x2, x1), dim=-1)

        # Apply rotation
        x_rot = (x * cos) + (rotated * sin)

        return x_rot.to(dtype=x.dtype)