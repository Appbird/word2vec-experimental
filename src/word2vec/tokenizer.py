from collections import Counter

class Tokenizer:
    def __init__(self, min_freq = 1):
        self.min_freq = min_freq
        self.word_to_id = {"<unk>": 0}
        self.id_to_word = ["<unk>"]
    def fit(self, text:str):
        counts = Counter(text.split())
        for word, freq in counts.items():
            if word == "<unk>":
                continue
            if freq >= self.min_freq:
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word.append(word)

    def encode(self, text:str) -> list[int]:
        return [
            self.word_to_id.get(word, 0)
            for word in text.split()
        ]

    def decode(self, ids: list) -> str:
        return " ".join(self.id_to_word[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.id_to_word)

