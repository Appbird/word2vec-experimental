from dataclasses import dataclass

@dataclass
class Config:
    context_size: int = 4
    embedding_dim: int = 128

    batch_size: int = 4096
    learning_rate: float = 1e-3
    epochs: int = 10

    min_freq:int = 5
    seed:int = 42
