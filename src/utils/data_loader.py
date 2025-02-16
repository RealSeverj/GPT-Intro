from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from config import settings as cfg


class TextDataset(Dataset):
    def __init__(self, dataset_name, split='train', cache_dir='data', device=None):
        self.block_size = cfg.MODEL_CONFIG["block_size"]
        self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.vocab_size = cfg.MODEL_CONFIG["vocab_size"]
        self.device = device  # 添加 device 参数，用于在加载时移到 GPU

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        x = [ord(c) if ord(c) < self.vocab_size else 0 for c in text[:self.block_size]]
        y = [ord(c) if ord(c) < self.vocab_size else 0 for c in text[1:self.block_size + 1]]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    # Separate the inputs and targets
    inputs, targets = zip(*batch)

    # Pad the sequences
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return inputs_padded, targets_padded
