import os
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
import wandb

from dataset import make_loaders


@dataclass
class TrainerConfig:
    steps          : int   = 5000
    eval_every     : int   = 500
    block_size     : int   = 128
    batch_size     : int   = 32
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
    def __init__(self, model, config: TrainerConfig):
        self.model     = model.to(config.device)
        self.config    = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.step      = 0

        train_loader, val_loader = make_loaders(
            config.block_size, config.batch_size,
            config.train_path, config.val_path,
        )
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.train_iter   = iter(train_loader)

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or None,
            config=asdict(config),
            mode='online',
            dir='/tmp',
        )

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
        total, count = 0.0, 0
        for x, y in self.val_loader:
            x, y   = x.to(self.config.device), y.to(self.config.device)
            logits = self.model(x)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            count += 1
            if count >= self.config.max_val_batches:
                break
        return total / count

    def save_checkpoint(self):
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        path = os.path.join(self.config.ckpt_dir, f'step_{self.step:05d}.pt')
        torch.save({
            'step'      : self.step,
            'model'     : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }, path)
        print(f"  checkpoint saved → {path}")

    def run(self):
        cfg        = self.config
        train_loss = 0.0

        for _ in range(cfg.steps):
            self.step  += 1
            train_loss += self._train_step()

            if self.step % cfg.eval_every == 0:
                val_loss   = self._evaluate()
                train_loss /= cfg.eval_every
                print(f"step {self.step:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=self.step)
                self.save_checkpoint()
                train_loss = 0.0

        wandb.finish()
