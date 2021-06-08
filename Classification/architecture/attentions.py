import torch

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
  temp = query.bmm(key.transpose(1, 2))
  scaled = temp / (query.size(-1) ** 0.5)

  softmax = torch.nn.functional.softmax(scaled, dim=-1)
  return softmax.bmm(value)

class AttentionHead(torch.nn.Module):
  def __init__(self, dim_in: int, dim_k: int, dim_v: int):
    super().__init__()
    self.q = torch.nn.Linear(dim_in, dim_k)
    self.k = torch.nn.Linear(dim_in, dim_k)
    self.v = torch.nn.Linear(dim_in, dim_v)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
    super().__init__()
    self.heads = torch.nn.ModuleList(
      [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
    )
    self.linear = torch.nn.Linear(num_heads * dim_v, dim_in)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return self.linear(torch.cat([h(query, key, value) for h in self.heads], dim=-1))