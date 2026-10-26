
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from word2vec.model import CausalCBOW
from word2vec.evaluate import evaluate


def make_dataloader():
    x = torch.tensor([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ])

    y = torch.tensor([4, 5, 6, 7])

    return DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )


def make_uniform_model(vocab_size=10):
    """すべての単語に等確率を与えるモデル。"""
    model = CausalCBOW(
        vocab_size=vocab_size,
        embedding_dim=8,
    )

    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()

    return model


def test_uniform_perplexity():
    """一様分布のPerplexityは語彙数に等しい。"""
    vocab_size = 10
    model = make_uniform_model(vocab_size)

    metrics = evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    assert metrics["perplexity"] == pytest.approx(
        vocab_size, rel=1e-5
    )


def test_uniform_cross_entropy():
    """一様分布のCross Entropyはlog(V)。"""
    vocab_size = 10
    model = make_uniform_model(vocab_size)

    metrics = evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    assert metrics["loss"] == pytest.approx(
        math.log(vocab_size), rel=1e-5
    )


def test_perplexity_relation():
    """PPL = exp(loss)が成り立つ。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=8)

    metrics = evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    assert metrics["perplexity"] == pytest.approx(
        math.exp(metrics["loss"])
    )


def test_evaluation_mode():
    """評価時にはmodel.eval()が呼ばれる。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=8)
    model.train()

    evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    assert model.training is False


def test_parameters_unchanged():
    """評価によってモデルの重みが更新されない。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=8)

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    for name, param in model.named_parameters():
        torch.testing.assert_close(
            before[name],
            param,
            rtol=0,
            atol=0,
        )


def test_evaluate_returns_expected_keys():
    """評価結果にLossとPerplexityが含まれる。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=8)

    metrics = evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    assert set(metrics.keys()) == {
        "loss",
        "perplexity",
    }
