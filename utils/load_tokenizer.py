from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

from configs.config import USE_MODEL


class OlmoTokenizer:
    def __init__(self, tokenizer_file_path, eos_token_id, pad_token_id):
        """
        tokenizer_file_path : str  path to tokenizer.json
        eos_token_id        : int  fallback EOS id
        pad_token_id        : int  fallback PAD id
        """

        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))

        eos_from_tok = self._tok.token_to_id("<|endoftext|>")
        if eos_from_tok is None:
            eos_from_tok = self._tok.token_to_id("<end_of_turn>")

        self.eos_token_id = (
            eos_from_tok if eos_from_tok is not None else eos_token_id
        )

        pad_from_tok = self._tok.token_to_id("<|pad|>")
        if pad_from_tok is None:
            pad_from_tok = self._tok.token_to_id("<pad>")

        self.pad_token_id = (
            pad_from_tok if pad_from_tok is not None else pad_token_id
        )

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    @staticmethod
    def apply_chat_template(user_text):
        return (
            "<|im_start|>user\n"
            f"{user_text}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


# ============================================================
# Tokenizer file handling
# ============================================================

_TOKENIZER = None


def _get_tokenizer_path():
    repo_id = f"allenai/{USE_MODEL}"
    local_dir = Path(repo_id).parts[-1]
    tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    if not os.path.exists(tokenizer_file_path):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=local_dir,
                token=hf_token,
            )
        except Exception as e:
            print(f"Failed to download tokenizer.json: {e}")
            tokenizer_file_path = "tokenizer.json"

    return tokenizer_file_path


def get_tokenizer():
    global _TOKENIZER

    if _TOKENIZER is None:
        tokenizer_file_path = _get_tokenizer_path()
        _TOKENIZER = OlmoTokenizer(
            tokenizer_file_path=tokenizer_file_path,
            eos_token_id=100_257,
            pad_token_id=100_277,
        )

    return _TOKENIZER
