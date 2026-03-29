"""
Full val eval on the quantized (int8+zlib) model.
Dequantizes to fp32 before running inference.

Usage:
    python3 eval_quantized.py --checkpoint checkpoints/<run>/best_int8.zlib
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.dataset      import BinDataset
from src.model        import PinkyLM
from src.tokenizer    import SentencePieceTokenizer
from src.quantization import decompress, dequantize_state_dict

TOKENIZER_PATH = 'data/tokenizers/fineweb_1024_bpe.model'
VAL_PATH       = 'data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin'


def parse_args():
    parser = argparse.ArgumentParser(description='Eval quantized PinkyLM on full val set')
    parser.add_argument('--checkpoint', required=True, help='Path to best_int8.zlib file')
    parser.add_argument('--tokenizer',  default=TOKENIZER_PATH)
    parser.add_argument('--val',        default=VAL_PATH)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--embed-dim',  type=int, default=64)
    parser.add_argument('--n-heads',    type=int, default=4)
    parser.add_argument('--n-layers',   type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, val_path, block_size, batch_size, base_bytes, has_leading_space, is_boundary, device):
    from torch.utils.data import DataLoader
    ds     = BinDataset(val_path, block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    loss_sum, token_count, byte_count = 0.0, 0, 0
    total = len(loader)

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum    += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum').item()
        token_count += y.numel()
        prev    = x.reshape(-1)
        tgt     = y.reshape(-1)
        tbytes  = base_bytes[tgt].to(torch.int32)
        tbytes += (has_leading_space[tgt] & ~is_boundary[prev]).to(torch.int32)
        byte_count += tbytes.sum().item()

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} batches...", flush=True)

    val_loss = loss_sum / token_count
    val_bpb  = (val_loss / math.log(2.0)) * (token_count / byte_count)
    return val_loss, val_bpb


if __name__ == '__main__':
    args      = parse_args()
    ckpt_path = Path(args.checkpoint)
    device    = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f"device:     {device}")
    print(f"checkpoint: {ckpt_path}")
    print(f"size:       {ckpt_path.stat().st_size / 1024**2:.2f} MB\n")

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    base_bytes, has_leading_space, is_boundary = tokenizer.build_bpb_luts(device=device)

    print("decompressing and dequantizing ...")
    quant_obj  = decompress(ckpt_path.read_bytes())
    state_dict = dequantize_state_dict(quant_obj)

    model = PinkyLM(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    model.load_state_dict(state_dict)

    print("running full val eval on quantized model...\n")
    val_loss, val_bpb = evaluate(
        model, args.val, args.block_size, args.batch_size,
        base_bytes, has_leading_space, is_boundary, device,
    )

    print(f"\nval_loss : {val_loss:.4f}")
    print(f"val_bpb  : {val_bpb:.4f}")
