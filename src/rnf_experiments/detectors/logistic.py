import torch
from torch import nn

class LogisticResidual(nn.Module):
    """Simple logistic regression baseline on residual norm."""

    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor):
        return self.fc(x).squeeze(-1)
