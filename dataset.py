import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class BinDataset(Dataset):
    def __init__(self, path, block_size):
        self.data       = np.memmap(path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx     : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64))
        return x, y


def load_dataset(path, block_size, split):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, f'fineweb_{split}_*.bin')))
        if not files:
            raise FileNotFoundError(f"No {split} shards found in {path}")
        return ConcatDataset([BinDataset(f, block_size) for f in files])
    return BinDataset(path, block_size)


def make_loaders(block_size, batch_size, train_path, val_path):
    train_ds = load_dataset(train_path, block_size, split='train')
    val_ds   = load_dataset(val_path,   block_size, split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader
