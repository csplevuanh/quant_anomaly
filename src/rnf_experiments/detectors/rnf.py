import torch
from torch import nn

class RNFDetector(nn.Module):
    """Residual Norm–Frequency (RN‑F) detector (Section 3.2)."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Return contamination logits for a batch of features."""
        return self.net(feats).squeeze(-1)
