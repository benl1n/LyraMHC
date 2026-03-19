import torch
import torch.nn as nn

from src.models.components.Lyra_encoder import Lyra


class SequenceEncoder(nn.Module):
    def __init__(self, d_input, d_model, dropout, encoder_cfg):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.encoder = Lyra(**encoder_cfg)


    def forward(self, x):
        # x: (B, C, L)

        x = self.proj(x)
        x = x.transpose(-1, -2)
        _, out = self.encoder(x, return_embeddings=True)
        return out
