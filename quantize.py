"""
Quantize a fp32 checkpoint to int8 + zlib compress it.
Saves best_int8.zlib next to the input checkpoint.

Usage:
    python3 quantize.py --checkpoint checkpoints/<run>/best.pt
"""

import argparse
import os
from pathlib import Path

import torch

from src.quantization import quantize_state_dict, compress


def parse_args():
    parser = argparse.ArgumentParser(description='Quantize PinkyLM checkpoint to int8')
    parser.add_argument('--checkpoint', required=True, help='Path to fp32 checkpoint .pt file')
    return parser.parse_args()


if __name__ == '__main__':
    args     = parse_args()
    ckpt_path = Path(args.checkpoint)

    print(f"loading {ckpt_path} ...")
    ckpt       = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    fp32_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"fp32 model size: {fp32_bytes / 1024**2:.2f} MB")

    print("quantizing to int8 ...")
    quant_obj = quantize_state_dict(state_dict)

    print("compressing with zlib ...")
    compressed = compress(quant_obj)

    out_path = ckpt_path.parent / 'best_int8.zlib'
    out_path.write_bytes(compressed)

    print(f"\nfp32 size:       {fp32_bytes / 1024**2:.2f} MB")
    print(f"int8+zlib size:  {len(compressed) / 1024**2:.2f} MB")
    print(f"compression:     {fp32_bytes / len(compressed):.1f}x")
    print(f"saved →          {out_path}")

    bpb = ckpt.get('val_bpb', '?')
    print(f"checkpoint bpb:  {bpb}")
