import torch

def position_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device("cuda")) -> torch.Tensor:
  pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
  dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
  phase = pos / 1e4 ** (dim / dim_model)

  return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

class PositionalEmbedding(torch.nn.Module):
  def __init__(self, max_length: int, dim_model: int):
    super(PositionalEmbedding, self).__init__()
    self.positional_embedding = torch.nn.Embedding(max_length, dim_model)
    positions = torch.arange(0, max_length)
    self.register_buffer("positions", positions)

  def forward(self, sequence: torch.Tensor) -> torch.Tensor:
    batch_size, _, seq_len = sequence.size()
    positions = self.positions[:seq_len].unsqueeze(0).repeat(batch_size, 1)
    return self.positional_embedding(positions)
