import argparse
import os
from pathlib import Path

import torch

# Load .env if present
_env = Path(__file__).parent / '.env'
if _env.exists():
    for line in _env.read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.model     import PinkyLM
from src.tokenizer import SentencePieceTokenizer
from src.trainer   import Trainer, TrainerConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train PinkyLM on FineWeb')
    parser.add_argument('--tokenizer',   default='data/tokenizers/fineweb_1024_bpe.model')
    parser.add_argument('--train',       default='data/datasets/fineweb10B_sp1024')
    parser.add_argument('--val',         default='data/datasets/fineweb10B_sp1024')
    parser.add_argument('--steps',       type=int,   default=5000)
    parser.add_argument('--eval-every',  type=int,   default=500)
    parser.add_argument('--block-size',  type=int,   default=128)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--embed-dim',   type=int,   default=64)
    parser.add_argument('--n-heads',     type=int,   default=4)
    parser.add_argument('--n-layers',    type=int,   default=4)
    parser.add_argument('--ckpt-dir',        default='checkpoints')
    parser.add_argument('--max-val-batches', type=int, default=100)
    parser.add_argument('--wandb-project',  default='pinky-lm')
    parser.add_argument('--wandb-run-name', default='')
    return parser.parse_args()


if __name__ == '__main__':
    args      = parse_args()
    device    = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    model     = PinkyLM(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )

    print(f"vocab size:   {len(tokenizer)}")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"device:       {device}\n")

    config  = TrainerConfig(
        steps=args.steps,
        eval_every=args.eval_every,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        ckpt_dir=args.ckpt_dir,
        train_path=args.train,
        val_path=args.val,
        max_val_batches=args.max_val_batches,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    trainer = Trainer(model, tokenizer, config)
    trainer.run()
