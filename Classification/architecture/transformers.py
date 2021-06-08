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

  def forward(self, source):
    source = self.attention(source, source, source)
    return self.feedforward(source)
