from typing import Any

import torch

from torch.utils.data import Dataset

class CausalCBOWDataset(Dataset):
    def __init__(self, token_ids:list[int], context_size:int):
        # token_idsで文章をぶっこむ
        self.tokens = token_ids
        self.context_size = context_size

    def __len__(self):
        # 長さNの文章からは、k個の窓で取れる分だけ取れる
        # k個未満の場合は0個。
        return max(0, len(self.tokens) - self.context_size)

    def __getitem__(self, index) :
        k = self.context_size
        context = self.tokens[index:index + k]
        target = self.tokens[index + k]

        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )

# 高速化のためデータローダーをGPU上でやってしまう
class CausalCBOWBatches:
    def __init__(
        self,
        token_ids: list[int],
        context_size:int,
        batch_size:int,
        shuffle:bool = False,
        device=None,
        seed: int|None = None
    ) -> None:
        self.tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.context_size = context_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = max(0, len(token_ids) - context_size)

        self.offset = torch.arange(context_size, device=device)
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None 
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size 
    def __iter__(self):
        N = self.num_samples
        if self.shuffle:
            starts = torch.randperm(N, generator=self.generator).to(self.tokens)
        else:
            starts = torch.arange(N, device=self.tokens.device)

        k = self.context_size
        for i in range(0, N, self.batch_size):
            idx = starts[i: i+self.batch_size]          # (B,)
            x = self.tokens[idx[:, None] + self.offset] # (B, K)
            # idx[:, None] (B, 1) + (K, ) ---> (B,1) + (1, K) broadcast
            y = self.tokens[idx + k] # (B, )
            yield x, y