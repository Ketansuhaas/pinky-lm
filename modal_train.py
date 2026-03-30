"""
Modal training for pinky-lm.

Setup (one-time):
    modal run modal_train.py::download_data

Train:
    modal run modal_train.py                                     # smoke test
    modal run modal_train.py --run run-01 --steps 5000           # real run

Download checkpoints:
    modal volume get pinky-lm-cache checkpoints ./checkpoints

Requires:
    .env with WANDB_API_KEY=<key>
"""

import os
import modal

# ---------------------------------------------------------------------------
# Image — source code is baked in at build time via add_local_dir
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "sentencepiece",
        "wandb",
        "psutil",
        "huggingface_hub",
    )
    .add_local_dir(".", remote_path="/src", copy=True)
)

# ---------------------------------------------------------------------------
# Persistent volume — stores data shards, tokenizer, and checkpoints
# ---------------------------------------------------------------------------

CACHE_DIR = "/cache"
DATA_DIR  = f"{CACHE_DIR}/data"
CKPT_DIR  = f"{CACHE_DIR}/checkpoints"

volume = modal.Volume.from_name("pinky-lm-cache", create_if_missing=True)

app = modal.App("pinky-lm", image=image)

# ---------------------------------------------------------------------------
# Data download (run once)
# ---------------------------------------------------------------------------

@app.function(
    volumes={CACHE_DIR: volume},
    timeout=60 * 60 * 2,   # 2h
)
def download_data(train_shards: int = 1, variant: str = "sp1024"):
    """Download FineWeb shards + tokenizer into the volume at /cache/data/."""
    import subprocess, sys
    from pathlib import Path

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "/src/data/cached_challenge_fineweb.py",
            "--variant", variant,
            "--train-shards", str(train_shards),
        ],
        check=True,
        env={**os.environ, "FINEWEB_DATA_ROOT": DATA_DIR},
    )
    volume.commit()
    print(f"downloaded {train_shards} train shard(s) + val shard + tokenizer → {DATA_DIR}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_dotenv()],
    timeout=60 * 60 * 12,   # 12h max
)
def train(
    run: str        = "",
    steps: int      = 5000,
    eval_every: int = 500,
    log_every: int  = 1,
    block_size: int = 1024,
    batch_size: int = 32,
    lr: float       = 1e-3,
    embed_dim: int  = 64,
    n_heads: int    = 4,
    n_layers: int   = 4,
    max_val_tokens: int = 500_000,
):
    import sys
    sys.path.insert(0, "/src")

    import torch
    from src.model     import PinkyLM
    from src.tokenizer import SentencePieceTokenizer
    from src.trainer   import Trainer, TrainerConfig

    tokenizer_path = f"{DATA_DIR}/tokenizers/fineweb_1024_bpe.model"
    data_path      = f"{DATA_DIR}/datasets/fineweb10B_sp1024"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:     {device}")
    print(f"GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    tokenizer = SentencePieceTokenizer(tokenizer_path)
    model     = PinkyLM(
        vocab_size=len(tokenizer),
        block_size=block_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    print(f"params:     {sum(p.numel() for p in model.parameters()):,}")

    config = TrainerConfig(
        steps=steps,
        eval_every=eval_every,
        log_every=log_every,
        block_size=block_size,
        batch_size=batch_size,
        lr=lr,
        device=device,
        ckpt_dir=CKPT_DIR,
        train_path=data_path,
        val_path=data_path,
        max_val_tokens=max_val_tokens,
        wandb_project="pinky-lm",
        wandb_run_name=run,
    )
    Trainer(model, tokenizer, config).run()
    volume.commit()


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    run: str        = "",
    steps: int      = 5000,
    eval_every: int = 500,
    log_every: int  = 1,
    block_size: int = 1024,
    batch_size: int = 32,
    lr: float       = 1e-3,
    embed_dim: int  = 64,
    n_heads: int    = 4,
    n_layers: int   = 4,
    max_val_tokens: int = 500_000,
):
    train.remote(
        run=run,
        steps=steps,
        eval_every=eval_every,
        log_every=log_every,
        block_size=block_size,
        batch_size=batch_size,
        lr=lr,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        max_val_tokens=max_val_tokens,
    )
