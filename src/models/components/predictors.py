from torch import nn


class SequencePredictor(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_seq):
        return self.net(fused_seq)