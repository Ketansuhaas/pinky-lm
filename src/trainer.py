import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import psutil
import torch
import torch.nn.functional as F
import wandb

from src.dataset import make_loaders


@dataclass
class TrainerConfig:
    steps          : int   = 5000
    eval_every     : int   = 500
    block_size     : int   = 1024
    batch_size     : int   = 8
    lr             : float = 1e-3
    grad_clip      : float = 1.0
    device         : str   = 'cpu'
    ckpt_dir       : str   = 'checkpoints'
    train_path     : str   = ''
    val_path       : str   = ''
    max_val_batches: int   = 100
    wandb_project  : str   = 'pinky-lm'
    wandb_run_name : str   = ''


class Trainer:
    def __init__(self, model, tokenizer, config: TrainerConfig):
        self.model     = model.to(config.device)
        self.tokenizer = tokenizer
        self.config    = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.step      = 0

        # BPB lookup tables — matches parameter-golf challenge's exact metric
        self.base_bytes, self.has_leading_space, self.is_boundary = \
            tokenizer.build_bpb_luts(device=config.device)

        # Average tokens-per-byte for converting train loss to bpb
        avg_bytes = self.base_bytes.float().mean().item()
        self.tokens_per_byte = 1.0 / avg_bytes if avg_bytes > 0 else 1.0

        train_loader, val_loader = make_loaders(
            config.block_size, config.batch_size,
            config.train_path, config.val_path,
        )
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.train_iter   = iter(train_loader)

        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or None,
            config=asdict(config),
            mode='online',
            dir='/tmp',
        )
        timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name       = config.wandb_run_name or run.id
        self.ckpt_dir  = os.path.join(config.ckpt_dir, f'{timestamp}_{run_name}')
        self.best_bpb  = float('inf')
        self.best_ckpt = None

    def _next_batch(self):
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
        return x.to(self.config.device), y.to(self.config.device)

    def _train_step(self):
        self.model.train()
        x, y   = self._next_batch()
        logits = self.model(x)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        loss_sum, token_count, byte_count, n = 0.0, 0, 0, 0
        for x, y in self.val_loader:
            x, y    = x.to(self.config.device), y.to(self.config.device)
            logits  = self.model(x)
            B, T    = y.shape
            loss_sum    += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum').item()
            token_count += B * T
            prev    = x.reshape(-1)
            tgt     = y.reshape(-1)
            tbytes  = self.base_bytes[tgt].to(torch.int32)
            tbytes += (self.has_leading_space[tgt] & ~self.is_boundary[prev]).to(torch.int32)
            byte_count += tbytes.sum().item()
            n += 1
            if n >= self.config.max_val_batches:
                break
        val_loss = loss_sum / token_count
        val_bpb  = (val_loss / math.log(2.0)) * (token_count / byte_count)
        return val_loss, val_bpb

    def _reset_peak_memory(self):
        if self.config.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    def _memory_stats(self):
        stats  = {}
        labels = {}
        total_ram = psutil.virtual_memory().total / 1024 ** 2

        if self.config.device == 'cuda':
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            alloc = torch.cuda.memory_allocated()  / 1024 ** 2
            peak  = torch.cuda.max_memory_allocated() / 1024 ** 2
            stats['gpu_mem_alloc_mb'] = alloc
            stats['gpu_mem_peak_mb']  = peak
            labels['gpu_mem_alloc_mb'] = f"{alloc:.0f}/{total:.0f} MB"
            labels['gpu_mem_peak_mb']  = f"peak {peak:.0f}/{total:.0f} MB"
        elif self.config.device == 'mps':
            total = psutil.virtual_memory().total / 1024 ** 2
            alloc  = torch.mps.current_allocated_memory() / 1024 ** 2
            driver = torch.mps.driver_allocated_memory()  / 1024 ** 2
            stats['mps_mem_alloc_mb']  = alloc
            stats['mps_mem_driver_mb'] = driver
            labels['mps_mem_alloc_mb']  = f"{alloc:.0f}/{total:.0f} MB"
            labels['mps_mem_driver_mb'] = f"driver {driver:.0f}/{total:.0f} MB"

        stats['cpu_ram_mb']  = psutil.Process().memory_info().rss / 1024 ** 2
        labels['cpu_ram_mb'] = f"{stats['cpu_ram_mb']:.0f}/{total_ram:.0f} MB"
        return stats, labels

    def save_checkpoint(self, val_bpb):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f'step_{self.step:05d}.pt')
        torch.save({
            'step'      : self.step,
            'model'     : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'val_bpb'   : val_bpb,
        }, path)
        print(f"  checkpoint → {path}")
        if val_bpb < self.best_bpb:
            self.best_bpb  = val_bpb
            self.best_ckpt = path
            best_path = os.path.join(self.ckpt_dir, 'best.pt')
            torch.save({'step': self.step, 'model': self.model.state_dict(), 'val_bpb': val_bpb}, best_path)
            print(f"  best checkpoint → {best_path} (bpb {val_bpb:.4f})")

    def run(self):
        cfg        = self.config
        train_loss = 0.0
        t0         = time.perf_counter()

        for _ in range(cfg.steps):
            self.step  += 1
            train_loss += self._train_step()

            if self.step % cfg.eval_every == 0:
                dt          = time.perf_counter() - t0
                train_loss /= cfg.eval_every
                tokens_seen = cfg.eval_every * cfg.batch_size * cfg.block_size
                tok_per_sec = tokens_seen / dt
                step_ms     = dt / cfg.eval_every * 1000

                val_loss, val_bpb = self._evaluate()
                train_bpb = (train_loss / math.log(2.0)) * self.tokens_per_byte
                mem, mem_labels = self._memory_stats()

                print(
                    f"step {self.step:5d} | "
                    f"train_bpb {train_bpb:.4f} | "
                    f"val_bpb {val_bpb:.4f} | "
                    f"{tok_per_sec:,.0f} tok/s | "
                    f"{step_ms:.1f} ms/step | "
                    + " | ".join(f"{k} {v}" for k, v in mem_labels.items())
                )
                wandb.log({
                    'train_bpb'  : train_bpb,
                    'val_bpb'    : val_bpb,
                    'tok_per_sec': tok_per_sec,
                    'step_ms'    : step_ms,
                    **mem,
                }, step=self.step)

                self.save_checkpoint(val_bpb)
                train_loss = 0.0
                t0         = time.perf_counter()
                self._reset_peak_memory()  # reset peak so next interval measures fresh peak

        wandb.finish()
        print(f"\ntraining done. best checkpoint: {self.best_ckpt} (bpb {self.best_bpb:.4f})")
