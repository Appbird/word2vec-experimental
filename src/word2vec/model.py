import torch
import torch.nn as nn

class CausalCBOW(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.output = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )
    def forward(self, x:torch.Tensor):
        # x : (B:バッチサイズ, K:コンテキスト長) 非負整数が中に入っている。実質的にOne-hot vectorを表している。
        embeddings = self.embedding(x)
        # x : (B, K, E:埋め込み次元) 
        context_vector = embeddings.mean(dim=1)
        # x : (B, E) コンテキストにわたって次元別に平均
        logits = self.output(context_vector)
        # x : (B, V) 
        return logits
