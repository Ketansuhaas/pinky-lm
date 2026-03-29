"""
Terminal chatbot using a trained (or quantized) PinkyLM checkpoint.

Usage:
    python3 chat.py --checkpoint checkpoints/<run>/best.pt
    python3 chat.py --checkpoint checkpoints/<run>/best_int8.zlib  --quantized

Commands during chat:
    /temp <float>   — set sampling temperature (default 1.0)
    /steps <int>    — set generation steps (default 200)
    /quit or /exit  — exit
"""

import argparse
import sys
from pathlib import Path

import torch

from src.model     import PinkyLM
from src.tokenizer import SentencePieceTokenizer

TOKENIZER_PATH = 'data/tokenizers/fineweb_1024_bpe.model'


def parse_args():
    parser = argparse.ArgumentParser(description='PinkyLM terminal chatbot')
    parser.add_argument('--checkpoint',  required=True, help='Path to .pt or .zlib checkpoint')
    parser.add_argument('--quantized',   action='store_true', help='Load as int8+zlib quantized checkpoint')
    parser.add_argument('--tokenizer',   default=TOKENIZER_PATH)
    parser.add_argument('--block-size',  type=int, default=128)
    parser.add_argument('--embed-dim',   type=int, default=64)
    parser.add_argument('--n-heads',     type=int, default=4)
    parser.add_argument('--n-layers',    type=int, default=4)
    parser.add_argument('--steps',       type=int, default=200,  help='Tokens to generate per prompt')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    return parser.parse_args()


def load_model(args, device):
    tokenizer  = SentencePieceTokenizer(args.tokenizer)
    ckpt_path  = Path(args.checkpoint)

    if args.quantized:
        from src.quantization import decompress, dequantize_state_dict
        print("decompressing and dequantizing ...", flush=True)
        quant_obj  = decompress(ckpt_path.read_bytes())
        state_dict = dequantize_state_dict(quant_obj)
    else:
        ckpt       = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model']
        step       = ckpt.get('step', '?')
        bpb        = ckpt.get('val_bpb', '?')
        print(f"loaded step {step} | val_bpb {bpb}", flush=True)

    model = PinkyLM(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f"device:     {device}")
    print(f"checkpoint: {args.checkpoint}")
    model, tokenizer = load_model(args, device)

    params = sum(p.numel() for p in model.parameters())
    print(f"params:     {params:,}")
    print("\nType a prompt and press Enter. Commands: /temp <f>, /steps <n>, /quit\n")

    temperature = args.temperature
    steps       = args.steps

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue

        if prompt.startswith('/quit') or prompt.startswith('/exit'):
            break

        if prompt.startswith('/temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f"  temperature → {temperature}")
            except (IndexError, ValueError):
                print("  usage: /temp <float>")
            continue

        if prompt.startswith('/steps '):
            try:
                steps = int(prompt.split()[1])
                print(f"  steps → {steps}")
            except (IndexError, ValueError):
                print("  usage: /steps <int>")
            continue

        output = model.generate(
            tokenizer, prompt,
            steps=steps,
            temperature=temperature,
            device=device,
        )
        print(f"\nModel: {output}\n")


if __name__ == '__main__':
    main()
