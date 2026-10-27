
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from word2vec.model import CausalCBOW
from word2vec.evaluate import evaluate


VOCAB_SIZE = 10
CUTOFFS = [5]


def make_dataloader():
    x = torch.tensor([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ])

    # 4はhead、5〜7はtailに入る
    y = torch.tensor([4, 5, 6, 7])

    return DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )


def make_model():
    return CausalCBOW(
        vocab_size=VOCAB_SIZE,
        embedding_dim=8,
        cutoffs=CUTOFFS,
    )


def make_zero_output_model():
    """出力層の重みをすべて0にしたモデル。

    headの (5語 + 1クラスタ) と、tail内の5語がそれぞれ等確率になる。
    """
    model = make_model()

    with torch.no_grad():
        for param in model.output.parameters():
            param.zero_()

    return model


def test_zero_output_cross_entropy():
    """出力層が0のときのCross Entropyが理論値と一致する。"""
    model = make_zero_output_model()

    metrics = evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    # head語: P = 1/6、tail語: P = 1/6 * 1/5 = 1/30
    expected = (math.log(6) + 3 * math.log(30)) / 4

    assert metrics["loss"] == pytest.approx(expected, rel=1e-5)


def test_loss_matches_log_prob():
    """lossは正解単語の負の対数確率の平均と一致する。"""
    torch.manual_seed(0)
    model = make_model()
    model.eval()

    loader = make_dataloader()
    metrics = evaluate(model, loader, torch.device("cpu"))

    with torch.no_grad():
        nll = [
            -model.log_prob(x).gather(1, y[:, None]).squeeze(1)
            for x, y in loader
        ]
    expected = torch.cat(nll).mean().item()

    assert metrics["loss"] == pytest.approx(expected, rel=1e-5)


def test_perplexity_relation():
    """PPL = exp(loss)が成り立つ。"""
    model = make_model()

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
    model = make_model()
    model.train()

    evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    assert model.training is False


def test_parameters_unchanged():
    """評価によってモデルの重みが更新されない。"""
    model = make_model()

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
    model = make_model()

    metrics = evaluate(
        model,
        make_dataloader(),
        torch.device("cpu"),
    )

    assert set(metrics.keys()) == {
        "loss",
        "perplexity",
    }
