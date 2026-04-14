"""
KV Cache: Global storage for Keys and Values across transformer layers
Used during autoregressive inference (NOT needed for full-sequence training)
"""


class KvCache:
    def __init__(self, n_layers: int):
        """
        Args:
            n_layers (int): Number of transformer layers

        Internal:
            self.cache: list of length n_layers

        Each entry:
            None
            OR
            (K, V)

        Shapes per layer:
            K: [B, H_kv, T_cached, D]
            V: [B, H_kv, T_cached, D]

        Where:
            B        = batch size
            H_kv     = number of KV heads (important for GQA)
            T_cached = total cached sequence length (grows over time)
            D        = head_dim
        """
        self.cache = [None] * n_layers


    def get(self, layer_idx: int):
        """
        Retrieve cached KV for a given layer.

        Args:
            layer_idx (int)

        Returns:
            None
            OR
            (K, V)

        Shapes:
            K: [B, H_kv, T_cached, D]
            V: [B, H_kv, T_cached, D]
        """
        return self.cache[layer_idx]


    def update(self, layer_idx: int, value):
        """
        Update KV cache for a given layer.

        Args:
            layer_idx (int)
            value (tuple): (K, V)

        Expected Shapes:
            K: [B, H_kv, T_new_total, D]
            V: [B, H_kv, T_new_total, D]

        Important:
            - This should be called AFTER concatenating past + new KV
            - Overwrites previous cache with extended sequence

        Example (outside this class):
            past_k, past_v = kv_cache.get(layer_idx)

            if past_k is not None:
                K = torch.cat([past_k, new_k], dim=2)
                V = torch.cat([past_v, new_v], dim=2)
            else:
                K, V = new_k, new_v

            kv_cache.update(layer_idx, (K, V))
        """
        self.cache[layer_idx] = value


    def get_all(self):
        """
        Returns full KV cache across all layers.

        Returns:
            List of length n_layers:
                [
                    (K_0, V_0),
                    (K_1, V_1),
                    ...
                ]

        Shapes:
            Each:
                K_l: [B, H_kv, T_cached, D]
                V_l: [B, H_kv, T_cached, D]
        """
        return self.cache


    def reset(self):
        """
        Clears the cache (for new sequence).

        After reset:
            All entries → None

        Important:
            Must be called between independent generations
            to avoid leakage across prompts.
        """
        for i in range(len(self.cache)):
            self.cache[i] = None