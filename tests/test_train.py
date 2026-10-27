
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from word2vec.model import CausalCBOW
from word2vec.train import train_epoch


def make_dataloader():
    """テスト用の小規模データセット。"""
    x = torch.tensor([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ])

    y = torch.tensor([4, 5, 6, 7])

    return DataLoader(
        TensorDataset(x, y),
        batch_size=4,
        shuffle=False,
    )


def test_parameters_updated():
    """学習によってパラメータが更新される。"""
    torch.manual_seed(42)

    model = CausalCBOW(vocab_size=10, embedding_dim=8)
    loader = make_dataloader()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    train_epoch(
        model, loader, optimizer, torch.device("cpu")
    )

    for name, param in model.named_parameters():
        assert not torch.equal(
            before[name], param
        ), f"{name} was not updated"


def test_loss_decreases():
    """反復学習によってLossが減少する。"""
    torch.manual_seed(42)

    model = CausalCBOW(vocab_size=10, embedding_dim=8)
    loader = make_dataloader()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    losses = []

    for _ in range(20):
        loss = train_epoch(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
        )
        losses.append(loss)

    assert losses[-1] < losses[0]


def test_returns_finite_loss():
    """Lossが有限の数値として返される。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=8)
    loader = make_dataloader()

    optimizer = torch.optim.Adam(model.parameters())

    loss = train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
    )
    assert math.isfinite(loss)


def test_training_mode_enabled():
    """学習時にはmodel.train()が呼ばれる。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=8)
    model.eval()

    optimizer = torch.optim.Adam(model.parameters())

    train_epoch(
        model,
        make_dataloader(),
        optimizer,
        torch.device("cpu"),
    )

    assert model.training is True
