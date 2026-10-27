import pytest

from word2vec.tokenizer import Tokenizer

def test_initial_vocabulary():
    """初期状態では<unk>のみが登録されている。"""
    tokenizer = Tokenizer()

    assert tokenizer.vocab_size == 1
    assert tokenizer.word_to_id == {"<unk>": 0}
    assert tokenizer.id_to_word == ["<unk>"]


def test_fit():
    """fit()で語彙が構築され、単語に一意なIDが割り当てられる。"""
    tokenizer = Tokenizer()
    tokenizer.fit("apple banana apple orange")

    assert tokenizer.vocab_size == 4
    assert tokenizer.word_to_id == {
        "<unk>": 0,
        "apple": 1,
        "banana": 2,
        "orange": 3,
    }


def test_encode():
    """テキストを単語ID列に変換できる。"""
    tokenizer = Tokenizer()
    tokenizer.fit("apple banana orange")

    assert tokenizer.encode("apple orange banana") == [1, 3, 2]


def test_decode():
    """単語ID列をテキストに復元できる。"""
    tokenizer = Tokenizer()
    tokenizer.fit("apple banana orange")

    assert tokenizer.decode([1, 3, 2]) == "apple orange banana"


def test_unknown_word():
    """未登録単語は<unk>（ID=0）に変換される。"""
    tokenizer = Tokenizer()
    tokenizer.fit("apple banana")

    assert tokenizer.encode("apple grape banana") == [1, 0, 2]
    assert tokenizer.decode([0]) == "<unk>"


def test_min_frequency():
    """出現頻度がmin_freq未満の単語は登録されない。"""
    tokenizer = Tokenizer(min_freq=2)
    tokenizer.fit("apple apple banana orange orange orange")

    assert tokenizer.vocab_size == 3
    # orange(3回) → apple(2回) の順にIDが振られる
    assert tokenizer.encode("apple banana orange") == [2, 0, 1]


def test_ids_sorted_by_frequency():
    """頻度の高い単語ほど小さいIDが振られる。"""
    tokenizer = Tokenizer()
    tokenizer.fit("c b c a b c d c b a")

    # c:4, b:3, a:2, d:1
    assert tokenizer.id_to_word == ["<unk>", "c", "b", "a", "d"]


def test_frequency_tie_keeps_appearance_order():
    """同じ頻度の単語は出現順にIDが振られる。"""
    tokenizer = Tokenizer()
    tokenizer.fit("banana apple cherry")

    assert tokenizer.id_to_word == ["<unk>", "banana", "apple", "cherry"]


def test_whitespace():
    """連続する空白・改行・タブを単語区切りとして扱う。"""
    tokenizer = Tokenizer()
    tokenizer.fit("apple   banana\norange\tgrape")

    assert tokenizer.vocab_size == 5
    assert tokenizer.encode("apple\nbanana\tgrape") == [1, 2, 4]


def test_empty_text():
    """空文字列を与えても語彙が壊れない。"""
    tokenizer = Tokenizer()
    tokenizer.fit("")

    assert tokenizer.vocab_size == 1
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_encode_decode_roundtrip():
    """登録済みの単語列はencode→decodeで復元できる。"""
    tokenizer = Tokenizer()
    tokenizer.fit("the cat sat on the mat")

    text = "the cat sat on the mat"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_token_not_duplicated():
    """<unk>が学習テキストに含まれてもID=0を維持する。"""
    tokenizer = Tokenizer()
    tokenizer.fit("<unk> apple banana")

    assert tokenizer.word_to_id["<unk>"] == 0
    assert tokenizer.id_to_word.count("<unk>") == 1
    assert tokenizer.vocab_size == 3
