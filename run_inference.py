import argparse
import torch

from configs.config import OLMO3_CONFIG
from inference.generate import generate_text_basic_stream
from inference.kv_cache import KvCache
from model.block import Olmo3
from utils.device import Device
from utils.load_tokenizer import OlmoTokenizer, get_tokenizer
from weights.load_weights import OlmoWeightLoader


def build_model():
    config = dict(OLMO3_CONFIG)
    config["group_size"] = config["n_heads"] // config["n_kv_heads"]
    return Olmo3(config)


def load_weights(model, device):
    loader = OlmoWeightLoader(model, model.cfg, device)
    model, _ = loader.load()
    return model


def generate_text(model, tokenizer, prompt, max_new_tokens, eos_token_id):
    input_token_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_token_ids, device=model.out_head.weight.device).unsqueeze(0)

    model.eval()
    cache = KvCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output_tokens = []

    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    ):
        token_id = token.squeeze(0).tolist()
        token_text = tokenizer.decode(token_id)
        print(token_text, end="", flush=True)
        output_tokens.append(token_text)

    if torch.cuda.is_available():
        def calc_gpu_gb(x):
            return f"{x / 1024 / 1024 / 1024:.2f} GB"

        print(
            f"\n\nGPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}"
        )

    print()
    return "".join(output_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Olmo model inference")
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is a neural network in simple words",
        help="Text prompt to run through the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Do not wrap the prompt in the chat template",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = Device.get()

    print(f"Using device: {device}")
    model = build_model()
    model = model.to(device)

    print("Loading tokenizer and weights...")
    model = load_weights(model, device)
    tokenizer = get_tokenizer()

    prompt = args.prompt
    if not args.no_chat_template:
        prompt = OlmoTokenizer.apply_chat_template(prompt)

    print("Prompt:")
    print(prompt)
    print("\nGenerating...\n")

    generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )


