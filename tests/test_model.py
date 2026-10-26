
import torch

from word2vec.model import CausalCBOW


def test_output_shape():
    """出力は[batch_size, vocab_size]。"""
    model = CausalCBOW(vocab_size=100, embedding_dim=16)

    x = torch.randint(0, 100, (8, 4))
    logits = model(x)

    assert logits.shape == (8, 100)


def test_single_sample():
    """バッチサイズ1でも動作する。"""
    model = CausalCBOW(vocab_size=50, embedding_dim=8)

    x = torch.tensor([[1, 2, 3, 4]])
    logits = model(x)

    assert logits.shape == (1, 50)


def test_permutation_invariance():
    """単語順序を変更しても出力が変わらない。"""
    model = CausalCBOW(vocab_size=20, embedding_dim=8)
    model.eval()

    x1 = torch.tensor([[1, 2, 3, 4]])
    x2 = torch.tensor([[4, 3, 2, 1]])

    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)

    torch.testing.assert_close(y1, y2)


def test_embedding_average():
    """内部の文脈表現がEmbeddingの平均と一致する。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=4)

    x = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        embeddings = model.embedding(x)
        expected = model.output(embeddings.mean(dim=1))
        actual = model(x)

    torch.testing.assert_close(actual, expected)


def test_backward():
    """Embeddingと出力層の両方に勾配が流れる。"""
    model = CausalCBOW(vocab_size=20, embedding_dim=8)

    x = torch.tensor([[1, 2, 3, 4]])
    target = torch.tensor([5])

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)

    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.output.weight.grad is not None
    assert model.output.bias.grad is not None


def test_output_is_logits():
    """モデルの出力は確率ではなくlogitsである。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=4)

    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(2.0)

    x = torch.tensor([[1, 2, 3]])
    logits = model(x)

    torch.testing.assert_close(
        logits,
        torch.full((1, 10), 2.0),
    )


def test_parameter_count():
    """パラメータ数が理論値と一致する。"""
    vocab_size = 100
    embedding_dim = 16

    model = CausalCBOW(vocab_size, embedding_dim)

    expected = (
        vocab_size * embedding_dim
        + embedding_dim * vocab_size
        + vocab_size
    )

    actual = sum(p.numel() for p in model.parameters())

    assert actual == expected
