import torch
import torch.nn as nn

class CausalCBOW(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            embedding_dim:int,
            cutoffs: list[int],
            div_value: float = 4.0
        ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.output = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=embedding_dim,
            n_classes=vocab_size,
            cutoffs=cutoffs,
            div_value=div_value
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, K) --> (B, E)
        return self.embedding(x).mean(dim=1)
    
    def forward(self, x:torch.Tensor, target: torch.Tensor):
        # 戻り値
            # .output (B, )正解単語のloss確立
            # .loss: 平均NLL
        return self.output(self.encode(x), target)
    def log_prob(self, x:torch.Tensor) -> torch.Tensor:
        return self.output.log_prob(self.encode(x))
