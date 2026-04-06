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

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "tiktoken", "numpy", "matplotlib")
)

train_image = base_image.add_local_file("train.py", "/root/train.py")


@app.function(image=base_image, volumes={VOLUME_PATH: volume}, timeout=300)
def prepare():
    import os
    import urllib.request
    import numpy as np
    import tiktoken

    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    print("Downloading tinyshakespeare...")
    urllib.request.urlretrieve(DATA_URL, os.path.join(VOLUME_PATH, "input.txt"))

    with open(os.path.join(VOLUME_PATH, "input.txt")) as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters")

    enc = tiktoken.get_encoding("gpt2")
    tokens = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    print(f"Tokenized: {len(tokens):,} tokens")

    split = int(0.9 * len(tokens))
    tokens[:split].tofile(os.path.join(VOLUME_PATH, "train.bin"))
    tokens[split:].tofile(os.path.join(VOLUME_PATH, "val.bin"))
    print(f"Train: {split:,} | Val: {len(tokens) - split:,}")

    volume.commit()
    print("Saved to volume.")


@app.function(image=train_image, volumes={VOLUME_PATH: volume}, gpu="A10G", timeout=3600)
def train(args: str = ""):
    import subprocess
    import sys
    import os

    ckpt_dir = os.path.join(VOLUME_PATH, "checkpoints")
    cmd = [
        sys.executable, "/root/train.py",
        "--data_dir", VOLUME_PATH,
        "--out_dir", ckpt_dir,
    ] + (args.split() if args else [])

    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"Done. Checkpoint at {ckpt_dir}/best.pt")
