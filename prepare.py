"""
Download and tokenize tinyshakespeare with GPT-2 BPE.
Saves data/train.bin and data/val.bin as uint16 token arrays.

Usage:
    python prepare.py
"""

import os
import urllib.request
import numpy as np
import tiktoken

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    input_path = os.path.join(DATA_DIR, "input.txt")

    print("Downloading tinyshakespeare...")
    urllib.request.urlretrieve(DATA_URL, input_path)
    print(f"Saved to {input_path}")

    with open(input_path, "r") as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters")

    n = len(text)
    train_text = text[:int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    enc = tiktoken.get_encoding("gpt2")
    train_tokens = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
    val_tokens = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"Train: {len(train_tokens):,} tokens → {train_path}")
    print(f"Val:   {len(val_tokens):,} tokens → {val_path}")
    # expected: train ~301,966 tokens, val ~36,059 tokens


if __name__ == "__main__":
    main()
