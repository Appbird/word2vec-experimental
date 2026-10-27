
import torch

from word2vec.model import CausalCBOW


def test_log_prob_shape():
    """log_prob()の出力は[batch_size, vocab_size]。"""
    model = CausalCBOW(vocab_size=100, embedding_dim=16, cutoffs=[10, 50])

    x = torch.randint(0, 100, (8, 4))
    log_probs = model.log_prob(x)

    assert log_probs.shape == (8, 100)


def test_single_sample():
    """バッチサイズ1でも動作する。"""
    model = CausalCBOW(vocab_size=50, embedding_dim=8, cutoffs=[10])

    x = torch.tensor([[1, 2, 3, 4]])
    target = torch.tensor([5])

    assert model.log_prob(x).shape == (1, 50)
    assert model(x, target).output.shape == (1,)


def test_log_prob_is_normalized():
    """log_prob()は全語彙で和が1になる対数確率である。"""
    model = CausalCBOW(vocab_size=100, embedding_dim=16, cutoffs=[10, 50])

    x = torch.randint(0, 100, (8, 4))
    with torch.no_grad():
        probs = model.log_prob(x).exp()

    torch.testing.assert_close(probs.sum(dim=1), torch.ones(8))


def test_forward_output_shape():
    """forward()は正解単語の対数確率(B,)とスカラーのlossを返す。"""
    model = CausalCBOW(vocab_size=100, embedding_dim=16, cutoffs=[10, 50])

    x = torch.randint(0, 100, (8, 4))
    target = torch.randint(0, 100, (8,))
    out = model(x, target)

    assert out.output.shape == (8,)
    assert out.loss.dim() == 0


def test_forward_consistent_with_log_prob():
    """forward()の出力はlog_prob()から正解単語を取り出したものと一致する。"""
    model = CausalCBOW(vocab_size=100, embedding_dim=16, cutoffs=[10, 50])
    model.eval()

    x = torch.randint(0, 100, (8, 4))
    # head・tail1・tail2のすべてのクラスタを含める
    target = torch.tensor([0, 5, 9, 10, 30, 49, 50, 99])

    with torch.no_grad():
        out = model(x, target)
        expected = model.log_prob(x).gather(1, target[:, None]).squeeze(1)

    torch.testing.assert_close(out.output, expected)
    torch.testing.assert_close(out.loss, -expected.mean())


def test_permutation_invariance():
    """単語順序を変更しても出力が変わらない。"""
    model = CausalCBOW(vocab_size=20, embedding_dim=8, cutoffs=[5])
    model.eval()

    x1 = torch.tensor([[1, 2, 3, 4]])
    x2 = torch.tensor([[4, 3, 2, 1]])

    with torch.no_grad():
        y1 = model.log_prob(x1)
        y2 = model.log_prob(x2)

    torch.testing.assert_close(y1, y2)


def test_encode_is_embedding_average():
    """encode()による文脈表現がEmbeddingの平均と一致する。"""
    model = CausalCBOW(vocab_size=10, embedding_dim=4, cutoffs=[5])

    x = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        expected = model.embedding(x).mean(dim=1)
        actual = model.encode(x)

    assert actual.shape == (1, 4)
    torch.testing.assert_close(actual, expected)


def test_backward():
    """Embedding・head・各tailに勾配が流れる。"""
    model = CausalCBOW(vocab_size=100, embedding_dim=16, cutoffs=[10, 50])

    x = torch.randint(0, 100, (3, 4))
    # head・tail1・tail2に1つずつ正解を置く
    target = torch.tensor([5, 30, 80])

    model(x, target).loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.output.head.weight.grad is not None
    for cluster in model.output.tail:
        for param in cluster.parameters():
            assert param.grad is not None


def test_parameter_count():
    """パラメータ数が理論値と一致し、full softmaxより少ない。"""
    vocab_size = 100
    embedding_dim = 16
    cutoffs = [10, 50]
    div_value = 4.0

    model = CausalCBOW(vocab_size, embedding_dim, cutoffs=cutoffs, div_value=div_value)

    # head: 頻出語 + クラスタ数（バイアスなし）
    expected = vocab_size * embedding_dim
    expected += embedding_dim * (cutoffs[0] + len(cutoffs))
    # tail i: E → E/div^(i+1) → クラスタ内の語彙数（バイアスなし）
    bounds = cutoffs + [vocab_size]
    for i in range(len(cutoffs)):
        hidden = int(embedding_dim // (div_value ** (i + 1)))
        cluster_size = bounds[i + 1] - bounds[i]
        expected += embedding_dim * hidden + hidden * cluster_size

    actual = sum(p.numel() for p in model.parameters())
    full_softmax = vocab_size * embedding_dim + embedding_dim * vocab_size + vocab_size

    assert actual == expected
    assert actual < full_softmax
