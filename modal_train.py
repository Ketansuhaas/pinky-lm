"""
Modal training for pinky-lm on tinyshakespeare.

Usage:
    modal run modal_train.py::prepare
    modal run modal_train.py::train
    modal run modal_train.py::train --args "--n_layer 8 --n_embd 512 --max_iters 10000"
"""

import modal

app = modal.App("pinky-lm")

volume = modal.Volume.from_name("pinky-lm-data", create_if_missing=True)
VOLUME_PATH = "/data"
TIKTOKEN_CACHE = "/data/tiktoken_cache"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "tiktoken", "numpy", "matplotlib", "wandb")
)

train_image = base_image.add_local_file("train.py", "/root/train.py")


@app.function(image=base_image, volumes={VOLUME_PATH: volume}, timeout=300)
def prepare():
    import os
    import urllib.request
    import numpy as np
    import tiktoken

    os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE
    os.makedirs(TIKTOKEN_CACHE, exist_ok=True)

    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    print("Downloading tinyshakespeare...")
    urllib.request.urlretrieve(DATA_URL, os.path.join(VOLUME_PATH, "input.txt"))

    with open(os.path.join(VOLUME_PATH, "input.txt")) as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters")

    n = len(text)
    train_text = text[:int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    enc = tiktoken.get_encoding("gpt2")
    train_tokens = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
    val_tokens = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)

    train_tokens.tofile(os.path.join(VOLUME_PATH, "train.bin"))
    val_tokens.tofile(os.path.join(VOLUME_PATH, "val.bin"))
    print(f"Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")

    volume.commit()
    print("Saved to volume.")


wandb_secret = modal.Secret.from_dotenv(".env")


@app.function(image=train_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600, secrets=[wandb_secret])
def train(args: str = ""):
    import subprocess
    import sys
    import os

    os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE

    ckpt_dir = os.path.join(VOLUME_PATH, "checkpoints")
    cmd = [
        sys.executable, "/root/train.py",
        "--data_dir", VOLUME_PATH,
        "--out_dir", ckpt_dir,
    ] + (args.split() if args else [])

    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"Done. Checkpoint at {ckpt_dir}/best.pt")
