import torch
from torch import Tensor

from architecture.layers import feed_forward, Residual
from architecture.attentions import MultiHeadAttention

class TransformerEncoderLayer(torch.nn.Module):
  def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
    super().__init__()
    dim_k = dim_v = dim_model // num_heads
    self.attention = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.feedforward = Residual(
      feed_forward(dim_model, dim_feedforward),
      dimension = dim_model,
      dropout = dropout
    )

  def forward(self, src: Tensor, mask=None) -> Tensor:
    src = self.attention(src, src, src, mask)
    return self.feedforward(src)

class TransformerEncoder(torch.nn.Module):
  def __init__(self, num_layers, dim_model, num_heads, dim_feedforward, dropout):
    super().__init__()

    self.dim_model = dim_model
    self.layers = torch.nn.ModuleList([
      TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
      for _ in range(num_layers)
    ])

  def forward(self, src: Tensor, mask=None) -> Tensor:
    seq_len, dimension = src.size(1), src.size(2)
    for layer in self.layers:
      src = layer(src, mask)

    return src
