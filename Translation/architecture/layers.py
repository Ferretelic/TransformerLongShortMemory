import torch

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> torch.nn.Module:
  return torch.nn.Sequential(
    torch.nn.Linear(dim_input, dim_feedforward),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_feedforward, dim_input)
  )

class Residual(torch.nn.Module):
  def __init__(self, sublayer: torch.nn.Module, dimension: int, dropout: float = 0.1):
    super().__init__()
    self.sublayer = sublayer
    self.norm = torch.nn.LayerNorm(dimension)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, *tensors):
    if len(tensors) == 1:
      value = tensors[0]
    else:
      value = tensors[2]
    return self.norm(value + self.dropout(self.sublayer(*tensors)))
