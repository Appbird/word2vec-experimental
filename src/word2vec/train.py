import torch
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm

def train_epoch(model:Module, dataloader, optimizer:Optimizer, device):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        # yは非負整数？
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        # 非負整数yをonehotベクトルと見做した上でのCross Entropy Lossを見ている。

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.detach() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

