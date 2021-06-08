import torch

from architecture.transformers import TransformerEncoder, TransformerShortMemoryDecoder

class ShortMemory:
  def __init__(self, dim_model: int, memory_size: int, batch_size: int, max_len: int, device):

    self.batch_size = batch_size
    self.max_len = max_len
    self.device = device
    self.memory_index = 0
    self.memory_size = memory_size
    self.dim_model = dim_model

  def update_memory(self, encoder_output: torch.Tensor):
    self.short_memory[self.memory_index] = encoder_output
    if (self.memory_index + 1) == self.memory_size:
      self.memory_index = 0
    else:
      self.memory_index += 1

  def initialize_memory(self):
    self.short_memory = torch.zeros(self.batch_size, self.memory_size, self.max_len, self.dim_model, requires_grad=False, device=self.device)
    self.memory_index = 0

  def extract_memory(self):
    index = range(self.memory_index - (self.memory_size - 1), self.memory_index + 1)
    return self.short_memory[:, index, :].clone().detach().requires_grad(False)

class MemoryAttention(torch.nn.Module):
  def __init__(self, dim_model: int = 512, memory_size: int = 5):
    super().__init__()
    self.dim_model = dim_model
    self.linear = torch.nn.Linear(dim_model * 2, memory_size)

  def forward(self, encoder_output: torch.Tensor, decoder_output: torch.Tensor, short_memory: torch.Tensor):
    combined = self.linear(torch.cat([encoder_output, decoder_output], dim=2))
    alpha = torch.softmax(torch.mean(combined, dim=1), dim=1)
    weighted_memory = short_memory * alpha.unsqueeze(2).unsqueeze(3).repeat(1, 1, combined.size(1), self.dim_model)
    return torch.sum(weighted_memory, dim=1)

class TransformerShortMemoryDecoderLayer(torch.nn.Module):
  def __init__(self, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float, short_memory_size: int):
    super().__init__()
    dim_k = dim_v = dim_model // num_heads
    self.attention_1 = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.attention_2 = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.feed_forward = Residual(
      feed_forward(dim_model, dim_feedforward),
      dimension = dim_model,
      dropout = dropout
    )
    self.short_memory_attention = MemoryAttention(dim_model, short_memory_size)

  def forward(self, tgt: Tensor, encoder_output: Tensor, short_memory: Tensor, long_memory: Tensor, src_mask=None, tgt_mask=None) -> Tensor:
    decoder_output = self.attention_1(tgt, tgt, tgt, tgt_mask)

    short_memory = self.short_memory_attention(encoder_output, decoder_output, short_memory)
    decoder_output = self.attention_2(encoder_output, short_memory, decoder_output, src_mask)

    return self.feed_forward(decoder_output)

class TransformerShortMemoryDecoder(torch.nn.Module):
  def __init__(self, num_layers: int, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float, short_memory_size: int):
    super().__init__()
    self.layers = torch.nn.ModuleList([
      TransformerShortMemoryDecoderLayer(dim_model, num_heads, dim_feedforward, dropout, short_memory_size)
      for _ in range(num_layers)
    ])

  def forward(self, tgt: Tensor, memory: Tensor, short_memory: Tensor, src_mask=None, tgt_mask=None) -> Tensor:
    seq_len, dimension = tgt.size(1), tgt.size(2)
    for layer in self.layers:
      tgt = layer(tgt, memory, short_memory, src_mask, tgt_mask)
    return tgt

class TransformerShortMemory(torch.nn.Module):
  def __init__(self, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_model: int = 512, num_heads: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,  short_memory_size: int = 5, vocab_size: int = 20000, batch_size: int = 64, max_len: int = 10, device = torch.device("cuda")):
    super().__init__()

    self.short_memory = ShortMemory(dim_model, short_memory_size, batch_size, max_len, device)

    self.encoder = TransformerEncoder(
      num_layers = num_encoder_layers,
      dim_model = dim_model,
      num_heads = num_heads,
      dim_feedforward = dim_feedforward,
      dropout = dropout
    )
    self.decoder = TransformerShortMemoryDecoder(
      num_layers = num_decoder_layers,
      dim_model = dim_model,
      num_heads = num_heads,
      dim_feedforward = dim_feedforward,
      dropout = dropout
    )
    self.dim_model = dim_model
    self.token_embedding = torch.nn.Embedding(vocab_size, dim_model)
    self.linear = torch.nn.Linear(dim_model, vocab_size)

  def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, tgt_mask=None) -> torch.Tensor:
    src = self.token_embedding(src) + position_encoding(src.size(1), self.dim_model)
    tgt = self.token_embedding(tgt) + position_encoding(tgt.size(1), self.dim_model)
    encoder_output = self.encoder(src, src_mask)

    self.short_memory.update_memory(encoder_output)
    short_memory = self.short_memory.extract_memory()

    return torch.softmax(self.linear(self.decoder(tgt, encoder_output, short_memory, src_mask, tgt_mask)), dim=-1)

  def initialize_memory(self):
    self.short_memory.initialize_memory()
