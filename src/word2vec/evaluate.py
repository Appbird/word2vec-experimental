import math
import torch
import torch.nn.functional as F

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

@torch.no_grad()
def evaluate(model:Module, dataloader:DataLoader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        loss = F.cross_entropy(logits, y, reduction="sum")

        total_loss += loss.item()
        total_samples += y.numel()

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }