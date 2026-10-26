
import pytest
import torch

from word2vec.dataset import CausalCBOWDataset


def test_length():
    """学習サンプル数はN-Kになる。"""
    dataset = CausalCBOWDataset(
        token_ids=[0, 1, 2, 3, 4, 5, 6],
        context_size=4,
    )

    assert len(dataset) == 3


def test_context_and_target():
    """直前K単語から次の単語を予測する。"""
    dataset = CausalCBOWDataset(
        token_ids=[0, 1, 2, 3, 4, 5],
        context_size=3,
    )

    x, y = dataset[0]

    assert x.tolist() == [0, 1, 2]
    assert y.item() == 3


def test_sliding_window():
    """ウィンドウが1単語ずつ進む。"""
    dataset = CausalCBOWDataset(
        token_ids=[0, 1, 2, 3, 4, 5],
        context_size=3,
    )

    assert dataset[0][0].tolist() == [0, 1, 2]
    assert dataset[1][0].tolist() == [1, 2, 3]
    assert dataset[2][0].tolist() == [2, 3, 4]


def test_no_future_leakage():
    """各サンプルには正解以降の単語が含まれない。"""
    tokens = [10, 11, 12, 13, 14, 15]
    k = 3

    dataset = CausalCBOWDataset(tokens, k)

    for i in range(len(dataset)):
        x, y = dataset[i]

        assert x.tolist() == tokens[i:i + k]
        assert y.item() == tokens[i + k]


def test_tensor_dtype():
    """Embeddingの入力に使えるlong型を返す。"""
    dataset = CausalCBOWDataset([1, 2, 3, 4], 2)

    x, y = dataset[0]

    assert x.dtype == torch.long
    assert y.dtype == torch.long


@pytest.mark.parametrize(
    "tokens,k,expected",
    [
        ([], 4, 0),
        ([1, 2], 4, 0),
        ([1, 2, 3, 4], 4, 0),
        ([1, 2, 3, 4, 5], 4, 1),
    ],
)
def test_short_sequences(tokens, k, expected):
    """文脈長以下の系列では学習サンプルが生成されない。"""
    dataset = CausalCBOWDataset(tokens, k)

    assert len(dataset) == expected
