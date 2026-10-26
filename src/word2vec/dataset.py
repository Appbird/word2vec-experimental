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