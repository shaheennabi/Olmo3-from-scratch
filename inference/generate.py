import torch
from inference.kv_cache import KvCache


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None, context_size=None):
    """
    model          : causal LM
    token_ids      : torch.Tensor  shape = (B, T)
                     B = batch size, T = current sequence length
    max_new_tokens : int
    eos_token_id   : int or None
    """

    model.eval()

    with torch.no_grad():
        cache = KvCache(n_layers=model.cfg["n_layers"])
        model.reset_kv_cache()

        logits = model(token_ids, cache=cache)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)
            logits = model(next_token, cache=cache)
