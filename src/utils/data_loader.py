import torch
import os
import mmap
from torch.utils.data import Dataset
from config import settings as cfg


class TextDataset(Dataset):
    def __init__(self, file_path):
        self.block_size = cfg.MODEL_CONFIG["block_size"]
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)

        # 使用内存映射加快读取速度
        with open(file_path, "r") as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return (self.file_size // self.block_size) - 1

    def __getitem__(self, idx):
        # 计算读取位置
        start = idx * self.block_size
        end = start + self.block_size * 2  # 同时读取x和y

        # 从内存映射直接读取
        self.mm.seek(start)
        chars = self.mm.read(end - start).decode('utf-8', 'ignore')

        # 转换为ASCII编码
        x = [ord(c) if ord(c) < cfg.MODEL_CONFIG["vocab_size"] else 0
             for c in chars[:self.block_size]]
        y = [ord(c) if ord(c) < cfg.MODEL_CONFIG["vocab_size"] else 0
             for c in chars[1:self.block_size + 1]]

        return torch.tensor(x), torch.tensor(y)

    def __del__(self):
        self.mm.close()
