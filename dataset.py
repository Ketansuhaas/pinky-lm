import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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


def make_loaders(block_size, batch_size, train_path, val_path):
    train_ds = BinDataset(train_path, block_size)
    val_ds   = BinDataset(val_path,   block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader
