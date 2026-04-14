import torch
import json
import os
from pathlib import Path
from configs.config import USE_MODEL


class OlmoWeightLoader:
    """
    Handles:
    - downloading weights
    - loading shards
    - mapping weights → model
    """

    def __init__(self, model, config, device, model_name=USE_MODEL):
        self.model = model
        self.config = config
        self.device = device
        self.model_name = model_name

    # ============================================================
    # Core assign function
    # ============================================================

    def _assign(self, left, right, name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in '{name}': {left.shape} vs {right.shape}"
            )

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right.to(left.device, dtype=left.dtype))
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    # ============================================================
    # Load weights into model
    # ============================================================

    def load_into_model(self, params):

        # ---- Embedding ----
        if "model.embed_tokens.weight" in params:
            self._assign(
                self.model.tok_emb.weight,
                params["model.embed_tokens.weight"],
                "embed_tokens",
            )

        # ---- Transformer ----
        for l in range(self.config["n_layers"]):
            block = self.model.blocks[l]
            att = block.att

            # Attention
            self._assign(att.W_query.weight, params[f"model.layers.{l}.self_attn.q_proj.weight"])
            self._assign(att.W_key.weight,   params[f"model.layers.{l}.self_attn.k_proj.weight"])
            self._assign(att.W_value.weight, params[f"model.layers.{l}.self_attn.v_proj.weight"])
            self._assign(att.out_proj.weight, params[f"model.layers.{l}.self_attn.o_proj.weight"])

            # Q/K norm
            self._assign(att.q_norm.weight, params[f"model.layers.{l}.self_attn.q_norm.weight"])
            self._assign(att.k_norm.weight, params[f"model.layers.{l}.self_attn.k_norm.weight"])

            # MLP
            self._assign(block.ff.fc1.weight, params[f"model.layers.{l}.mlp.gate_proj.weight"])
            self._assign(block.ff.fc2.weight, params[f"model.layers.{l}.mlp.up_proj.weight"])
            self._assign(block.ff.fc3.weight, params[f"model.layers.{l}.mlp.down_proj.weight"])

            # Norms
            self._assign(
                block.post_attention_layernorm.weight,
                params[f"model.layers.{l}.post_attention_layernorm.weight"],
            )

            self._assign(
                block.post_feedforward_layernorm.weight,
                params[f"model.layers.{l}.post_feedforward_layernorm.weight"],
            )

        # ---- Final norm ----
        if "model.norm.weight" in params:
            self._assign(
                self.model.final_norm.weight,
                params["model.norm.weight"],
            )

        # ---- LM head ----
        if "lm_head.weight" in params:
            self._assign(
                self.model.out_head.weight,
                params["lm_head.weight"],
            )
        else:
            self.model.out_head.weight = self.model.tok_emb.weight
            print("Using weight tying.")

    # ============================================================
    # Download + load all weights
    # ============================================================

    def load(self):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load Olmo weights. Install it with: pip install safetensors"
            ) from exc

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required to download Olmo weights. Install it with: pip install huggingface-hub"
            ) from exc

        repo_id = f"allenai/{self.model_name}"
        local_dir = Path(repo_id).parts[-1]

        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)

        # ---- Load index ----
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")

        with open(index_path, "r") as f:
            index = json.load(f)

        # ---- Load shards ----
        weights_dict = {}

        for filename in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)

        # ---- Apply weights ----
        self.load_into_model(weights_dict)

        # ---- Move model ----
        self.model = self.model.to(self.device)

        # ---- Free memory ----
        del weights_dict

        return self.model, local_dir