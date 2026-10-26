import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from word2vec.config import Config
from word2vec.tokenizer import Tokenizer
from word2vec.dataset import CausalCBOWDataset
from word2vec.model import CausalCBOW
from word2vec.train import train_epoch
from word2vec.evaluate import evaluate

def train_loop():
    config = Config()
    torch.manual_seed(config.seed)
    assert torch.backends.mps.is_available()
    device = torch.device("mps")
    print(f"Device: {device}")

    text = Path("corpus/ja.text8.txt").read_text(encoding="utf-8")
    words = text.split()
    split = int(len(words) * 0.9)
    train_text = " ".join(words[:split])
    valid_text =  " ".join(words[split:])

    tokenizer = Tokenizer(min_freq = config.min_freq)
    tokenizer.fit(train_text)

    print(f"Vocabulary: {tokenizer.vocab_size}")

    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)

    train_dataset = CausalCBOWDataset(train_ids, config.context_size)
    valid_dataset = CausalCBOWDataset(valid_ids, config.context_size)
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Corpus is too small.")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    model = CausalCBOW(vocab_size=tokenizer.vocab_size, embedding_dim=config.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, valid_loader, device)
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"valid_ppl={metrics['perplexity']:.2f}"
        )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")
    Path("checkpoints/vocab.json").write_text(
        json.dumps(tokenizer.id_to_word, ensure_ascii=False),
        encoding="utf-8"
    )

if __name__ == "__main__":
    train_loop()