import torch
from torch.utils.data import Dataset
from mingpt.utils import CfgNode as CN

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.data.block_size

    def __len__(self):
        return len(self.data) - (self.config.data.block_size + self.config.medusa.num_heads + 1)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.config.data.block_size+1+self.config.medusa.num_heads]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:self.config.data.block_size], dtype=torch.long)
        y_gpt = torch.tensor(dix[1:self.config.data.block_size+1], dtype=torch.long)
        
        # Create outputs for medusa heads.
        y_medusa = []

        for i in range(1, self.config.medusa.num_heads+1):
            y_medusa.append(torch.tensor(dix[i+1:i+1+ self.config.data.block_size], dtype=torch.long))
        y_medusa = torch.stack(y_medusa, dim=0)
        
        return x, y_gpt, y_medusa