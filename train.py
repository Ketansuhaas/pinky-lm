"""
Train a small GPT on tinyshakespeare using GPT-2 BPE tokenization.
Run prepare.py first to download and tokenize the data.

Usage:
    python train.py
    python train.py --n_layer 4 --n_embd 256 --max_iters 10000
    python train.py --resume checkpoints/best.pt
"""

import os
import math
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Load .env if present
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import wandb


# ── Model ──────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        nh, hs = self.n_head, C // self.n_head
        k = k.view(B, T, nh, hs).transpose(1, 2)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying: share token embedding and output projection weights
        self.tok_emb.weight = self.head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ── Data ───────────────────────────────────────────────────────────────────

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    model.eval()

    # Train: random sampling (dataset too large to sweep)
    train_losses = [
        model(*get_batch(train_data, block_size, batch_size, device))[1].item()
        for _ in range(eval_iters)
    ]
    train_loss = sum(train_losses) / len(train_losses)

    # Val: full sequential sweep of non-overlapping windows — exact, reproducible
    n_windows = (len(val_data) - 1) // block_size
    total_loss, total_windows = 0.0, 0
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        xs = [torch.from_numpy(val_data[w * block_size : w * block_size + block_size].astype(np.int64)) for w in range(batch_start, batch_end)]
        ys = [torch.from_numpy(val_data[w * block_size + 1 : w * block_size + block_size + 1].astype(np.int64)) for w in range(batch_start, batch_end)]
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        _, loss = model(x, y)
        n = batch_end - batch_start
        total_loss += loss.item() * n
        total_windows += n
    val_loss = total_loss / total_windows

    model.train()
    return {"train": train_loss, "val": val_loss}


# ── GPU Metrics ────────────────────────────────────────────────────────────

# A100 40GB peak FP32 throughput (19.5 TFLOPS)
_A100_FP32_PEAK_FLOPS = 19.5e12


def get_gpu_metrics(step_time, tokens_per_iter, n_params):
    """
    step_time: avg seconds per optimizer step over the last eval interval
               (measured with CUDA events). None on the first eval.
    """
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated()
    reserved  = torch.cuda.memory_reserved()
    total     = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    metrics = {
        "gpu/mem_allocated_mb":    allocated / 1024 ** 2,
        "gpu/mem_reserved_mb":     reserved  / 1024 ** 2,
        "gpu/mem_utilization_pct": 100 * allocated / total,
    }
    if step_time is not None:
        flops_per_step = 6 * n_params * tokens_per_iter   # fwd + bwd ≈ 6x params x tokens
        metrics["gpu/tokens_per_sec"] = tokens_per_iter / step_time
        metrics["gpu/mfu"]            = flops_per_step / (step_time * _A100_FP32_PEAK_FLOPS)
    return metrics


# ── LR Schedule ────────────────────────────────────────────────────────────

def get_lr(it, max_iters, warmup_iters, lr, min_lr):
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it >= max_iters:
        return min_lr
    decay = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (lr - min_lr)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="checkpoints")
    # GPT-2 vocab is 50257; pad to nearest multiple of 64 for efficiency
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="pinky-lm")
    parser.add_argument("--wandb_run", type=str, default=None)
    args = parser.parse_args()

    tokens_per_iter = args.batch_size * args.block_size * args.grad_accum_steps
    if args.max_tokens is not None:
        args.max_iters = args.max_tokens // tokens_per_iter
        print(f"max_tokens={args.max_tokens:,} → max_iters={args.max_iters:,} ({tokens_per_iter:,} tokens/iter)")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    model = GPT(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    wandb.init(
        project=args.wandb_project,
        name=f"l{args.n_layer}_e{args.n_embd}_h{args.n_head}_bs{args.batch_size}_lr{args.lr:.0e}",
        config={**vars(args), "n_params": n_params, "effective_batch_size": args.batch_size * args.grad_accum_steps, "tokens_per_iter": tokens_per_iter},
    )

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt["iter"]
        print(f"Resumed from {args.resume} at iter {start_iter}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)

    train_losses, val_losses, loss_iters = [], [], []
    best_val_loss = float("inf")
    t0 = time.time()

    # CUDA event timing: measure GPU time across each eval interval
    if device == "cuda":
        _interval_start = torch.cuda.Event(enable_timing=True)
        _interval_end   = torch.cuda.Event(enable_timing=True)
        _interval_start.record()
    avg_step_time = None  # seconds/step; None until first interval completes

    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it, args.max_iters, args.warmup_iters, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % args.eval_interval == 0:
            # Capture GPU time for this interval before entering eval (which uses GPU too)
            if device == "cuda" and it > 0:
                _interval_end.record()
                torch.cuda.synchronize()
                avg_step_time = _interval_start.elapsed_time(_interval_end) / 1000 / args.eval_interval
                _interval_start.record()

            losses = estimate_loss(
                model, train_data, val_data, args.block_size, args.batch_size, device, args.eval_iters
            )
            elapsed = time.time() - t0
            tps_str = f" | {tokens_per_iter / avg_step_time:,.0f} tok/s" if avg_step_time else ""
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                f" | lr {lr:.2e} | {elapsed:.1f}s{tps_str}"
            )
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            loss_iters.append(it)

            wandb.log({"train/loss": losses["train"], "val/loss": losses["val"], "lr": lr, "tokens_seen": it * tokens_per_iter, **get_gpu_metrics(avg_step_time, tokens_per_iter, n_params)}, step=it)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {"iter": it, "model": model.state_dict(), "args": vars(args), "val_loss": losses["val"]},
                    os.path.join(args.out_dir, "best.pt"),
                )
                print(f"  → checkpoint saved (val {best_val_loss:.4f})")

        if it == args.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            (loss / args.grad_accum_steps).backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    # Save loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(loss_iters, train_losses, label="train")
    plt.plot(loss_iters, val_losses, label="val")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Loss curves → {plot_path}")


if __name__ == "__main__":
    main()
