import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(d_model, d_ffn) -> activation -> Linear(d_ffn, d_model)."""

    def __init__(self, d_model, d_ffn, dropout=0.0, activation=None):
        super().__init__()
        # The layer indices (net.0 and net.3) are part of the checkpoint format.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            activation if activation is not None else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
