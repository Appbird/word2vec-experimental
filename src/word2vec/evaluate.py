import math
import torch
import torch.nn.functional as F

from torch.nn import Module
from torch.optim import Optimizer

@torch.no_grad()
def evaluate(model:Module, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)

        total_loss += -out.output.sum() # 正解のlog確率の合計
        total_samples += y.numel()

    avg_loss = (total_loss / total_samples).item()
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }