import pytest
import torch

from word2vec.dataset import CausalCBOWBatches, CausalCBOWDataset
from word2vec.model import CausalCBOW


def collect(batches):
    """全バッチを結合して (X, Y) を返す。"""
    xs, ys = zip(*list(batches))
    return torch.cat(xs), torch.cat(ys)


@pytest.mark.parametrize(
    "n,k,b,expected",
    [
        (10, 3, 4, 2),   # 7サンプル → 4 + 3
        (11, 3, 4, 2),   # 8サンプル → 4 + 4
        (12, 3, 4, 3),   # 9サンプル → 4 + 4 + 1
        (4, 4, 4, 0),    # サンプルなし
        (0, 4, 4, 0),
    ],
)
def test_length(n, k, b, expected):
    """len() はバッチ数 ceil((N-K)/B) を返す。"""
    batches = CausalCBOWBatches(list(range(n)), context_size=k, batch_size=b)

    assert len(batches) == expected
    assert len(list(batches)) == expected


def test_empty_yields_nothing():
    """サンプルがない場合は何も生成しない。"""
    batches = CausalCBOWBatches([1, 2, 3], context_size=4, batch_size=2)

    assert list(batches) == []


def test_shapes():
    """x は (B, K)、y は (B,) で、最後のバッチは端数になる。"""
    batches = CausalCBOWBatches(list(range(12)), context_size=3, batch_size=4)

    sizes = []
    for x, y in batches:
        assert x.dim() == 2 and x.size(1) == 3
        assert y.dim() == 1
        assert x.size(0) == y.size(0)
        sizes.append(x.size(0))

    assert sizes == [4, 4, 1]


def test_matches_dataset_without_shuffle():
    """shuffle=False なら CausalCBOWDataset と同じサンプルを同じ順で返す。"""
    tokens = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]
    k = 3
    dataset = CausalCBOWDataset(tokens, k)

    X, Y = collect(CausalCBOWBatches(tokens, context_size=k, batch_size=3))

    assert len(X) == len(dataset)
    for i in range(len(dataset)):
        x, y = dataset[i]
        assert X[i].tolist() == x.tolist()
        assert Y[i].item() == y.item()


def test_no_future_leakage_with_shuffle():
    """shuffle=True でも各サンプルは連続した直前K単語と次の単語の組になる。"""
    # 値 = 位置 になる系列なので、xの先頭から窓の位置が分かる
    tokens = list(range(50))
    k = 4

    batches = CausalCBOWBatches(
        tokens, context_size=k, batch_size=8, shuffle=True, seed=0
    )

    for x, y in batches:
        for row, target in zip(x.tolist(), y.tolist()):
            start = row[0]
            assert row == tokens[start:start + k]
            assert target == tokens[start + k]


def test_shuffle_covers_every_sample_once():
    """shuffle=True では1エポックで全サンプルをちょうど1回ずつ返す。"""
    tokens = list(range(50))
    k = 4

    X, _ = collect(CausalCBOWBatches(
        tokens, context_size=k, batch_size=8, shuffle=True, seed=0
    ))
    starts = X[:, 0].tolist()

    assert sorted(starts) == list(range(len(tokens) - k))


def test_shuffle_changes_order():
    """shuffle=True では元の順序と異なる順序になる。"""
    tokens = list(range(100))

    X, _ = collect(CausalCBOWBatches(
        tokens, context_size=2, batch_size=16, shuffle=True, seed=0
    ))

    assert X[:, 0].tolist() != list(range(98))


def test_shuffle_reproducible_with_seed():
    """同じseedなら同じ順序になる。"""
    tokens = list(range(100))

    def make():
        return CausalCBOWBatches(
            tokens, context_size=2, batch_size=16, shuffle=True, seed=123
        )

    X1, Y1 = collect(make())
    X2, Y2 = collect(make())

    assert torch.equal(X1, X2)
    assert torch.equal(Y1, Y2)


def test_reshuffles_each_epoch():
    """同じインスタンスを再度回すと、別の順序でシャッフルされる。"""
    batches = CausalCBOWBatches(
        list(range(100)), context_size=2, batch_size=16, shuffle=True, seed=0
    )

    X1, _ = collect(batches)
    X2, _ = collect(batches)

    assert not torch.equal(X1, X2)


def test_iterable_multiple_times_without_shuffle():
    """shuffle=False なら何度回しても同じ結果になる。"""
    batches = CausalCBOWBatches(list(range(20)), context_size=3, batch_size=4)

    X1, Y1 = collect(batches)
    X2, Y2 = collect(batches)

    assert torch.equal(X1, X2)
    assert torch.equal(Y1, Y2)


def test_usable_for_model_and_loss():
    """出力をそのままモデルに渡してlossを計算できる。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=8, cutoffs=[5])
    batches = CausalCBOWBatches(
        [1, 2, 3, 4, 5, 6, 7, 8, 9], context_size=3, batch_size=4
    )

    for x, y in batches:
        loss = model(x, y).loss
        assert torch.isfinite(loss)


def test_device_cpu():
    """指定したデバイス上のテンソルを返す。"""
    batches = CausalCBOWBatches(
        list(range(10)), context_size=3, batch_size=4, device=torch.device("cpu")
    )

    for x, y in batches:
        assert x.device.type == "cpu"
        assert y.device.type == "cpu"
