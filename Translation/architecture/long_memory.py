import sys
sys.path.append("..")

import torch

from architecture.transformers import TransformerEncoder
from architecture.embeddings import position_encoding
from architecture.layers import Residual, feed_forward
from architecture.attentions import MultiHeadAttention

class TransformerLongShortMemoryDecoderLayer(torch.nn.Module):
  def __init__(self, dim_model, num_heads, dim_feedforward, dropout, short_memory_size, long_memory_size):
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
    self.attention_3 = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.attention_4 = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.feed_forward_1 = Residual(
      feed_forward(dim_model, dim_feedforward),
      dimension = dim_model,
      dropout = dropout
    )
    self.feed_forward_2 = Residual(
      feed_forward(dim_model, dim_feedforward),
      dimension = dim_model,
      dropout = dropout
    )
    self.short_memory_attention = MemoryAttention(dim_model, short_memory_size)
    self.long_memory_attention = MemoryAttention(dim_model, long_memory_size)

  def forward(self, tgt, encoder_output, short_memory, long_memory, src_mask=None, tgt_mask=None):
    decoder_output = self.attention_1(tgt, tgt, tgt, tgt_mask)

    long_memory = self.long_memory_attention(encoder_output, decoder_output, long_memory)
    decoder_output = self.attention_2(long_memory, long_memory, decoder_output)

    decoder_output = self.feed_forward_1(decoder_output)

    short_memory = self.short_memory_attention(encoder_output, decoder_output, short_memory)
    decoder_output = self.attention_3(encoder_output, short_memory, decoder_output)

    decoder_output = self.attention_4(encoder_output, encoder_output, decoder_output, src_mask)
    return self.feed_forward_2(decoder_output)

class TransformerLongShortMemoryDecoder(torch.nn.Module):
  def __init__(self, num_layers, dim_model, num_heads, dim_feedforward, dropout, short_memory_size, long_memory_size):
    super().__init__()
    self.layers = torch.nn.ModuleList([
      TransformerLongShortMemoryDecoderLayer(dim_model, num_heads, dim_feedforward, dropout, short_memory_size, long_memory_size)
      for _ in range(num_layers)
    ])

  def forward(self, tgt, encoder_output, short_memory, long_memory, src_mask=None, tgt_mask=None):
    seq_len, dimension = tgt.size(1), tgt.size(2)
    for layer in self.layers:
      tgt = layer(tgt, encoder_output, short_memory, long_memory, src_mask, tgt_mask)
    return tgt

class ShortMemory(torch.nn.Module):
  def __init__(self, dim_model, short_memory_size, batch_size, max_length):
    super().__init__()

    self.batch_size = batch_size
    self.memory_index = 0
    self.short_memory_size = short_memory_size
    self.dim_model = dim_model
    self.memory_compressor = torch.nn.Conv1d(max_length, 1, kernel_size=1, stride=1, padding=0)

  def update_memory(self, encoder_output):
    compressed_memory = self.memory_compressor(encoder_output).squeeze()
    self.short_memory[:, self.memory_index, :] = compressed_memory.clone().detach().cpu()

    if (self.memory_index + 1) == self.short_memory_size:
      self.memory_index = 0
    else:
      self.memory_index += 1

  def initialize_memory(self):
    self.short_memory = torch.zeros(self.batch_size, self.short_memory_size, self.dim_model, requires_grad=False, dtype=torch.float32).cpu()
    self.memory_index = 0

  def extract_memory(self):
    index = range(self.memory_index - self.short_memory_size, self.memory_index)
    return self.short_memory[:, index, :].clone().detach().cuda()

class LongMemory(torch.nn.Module):
  def __init__(self, dim_model, short_memory_size, long_memory_size, batch_size):
    super().__init__()

    self.long_memory_size = long_memory_size
    self.dim_model = dim_model
    self.batch_size = batch_size
    self.compression = torch.nn.Conv2d(short_memory_size, long_memory_size, kernel_size=1, stride=1, padding=0)

  def update_memory(self, short_memory):
    compressed_memory = self.compression(short_memory)
    memory_sigmoid = torch.sigmoid(compressed_memory)
    self.long_memory *= memory_sigmoid
    self.long_memory += (memory_sigmoid * torch.tanh(compressed_memory)).clone().detach().cpu()

  def initialize_memory(self):
    self.long_memory = torch.zeros(self.batch_size, self.long_memory_size, self.dim_model, requires_grad=False).cpu()

  def extract_memory(self):
    return self.long_memory.clone().detach().cuda()

class MemoryAttention(torch.nn.Module):
  def __init__(self, dim_model = 512, memory_size = 5):
    super().__init__()
    self.dim_model = dim_model
    self.linear = torch.nn.Linear(dim_model * 2, memory_size)

  def forward(self, encoder_output, decoder_output, short_memory):
    combined = self.linear(torch.cat([encoder_output, decoder_output], dim=2))
    alpha = torch.softmax(torch.mean(combined, dim=1), dim=1)
    weighted_memory = short_memory * alpha.unsqueeze(2).unsqueeze(3).repeat(1, 1, combined.size(1), self.dim_model)
    return torch.sum(weighted_memory, dim=1)

class TransformerLongShortMemory(torch.nn.Module):
  def __init__(self,
               num_encoder_layers = 6,
               num_decoder_layers = 3,
               dim_model = 512,
               num_heads = 6,
               dim_feedforward = 2048,
               dropout: float = 0.1,
               short_memory_size = 5,
               long_memory_size = 10,
               embedding_size = 20000,
               batch_size = 64,
               max_length = 10):
    super().__init__()

    self.short_memory = ShortMemory(dim_model, short_memory_size, batch_size, max_length)
    self.long_memory = LongMemory(dim_model, short_memory_size, long_memory_size, batch_size)

    self.encoder = TransformerEncoder(
      num_layers = num_encoder_layers,
      dim_model = dim_model,
      num_heads = num_heads,
      dim_feedforward = dim_feedforward,
      dropout = dropout
    )
    self.decoder = TransformerLongShortMemoryDecoder(
      num_layers = num_decoder_layers,
      dim_model = dim_model,
      num_heads = num_heads,
      dim_feedforward = dim_feedforward,
      dropout = dropout,
      short_memory_size = short_memory_size,
      long_memory_size = long_memory_size
    )
    self.dim_model = dim_model
    self.token_embedding = torch.nn.Embedding(embedding_size, dim_model)
    self.linear = torch.nn.Linear(dim_model, embedding_size)

  def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    src = self.token_embedding(src) + position_encoding(src.size(1), self.dim_model)
    tgt = self.token_embedding(tgt) + position_encoding(tgt.size(1), self.dim_model)
    encoder_output = self.encoder(src, src_mask)

    self.short_memory.update_memory(encoder_output)
    self.long_memory.update_memory(self.short_memory.extract_memory())

    short_memory = self.short_memory.extract_memory()
    long_memory = self.long_memory.extract_memory()

    decoder_output = self.decoder(tgt, encoder_output, short_memory, long_memory, src_mask, tgt_mask)
    return torch.softmax(self.linear(decoder_output), dim=-1)

  def initialize_memory(self):
    self.short_memory.initialize_memory()
    self.long_memory.initialize_memory()


short_memory = ShortMemory(512, 10, 64, 10).cuda()
input = torch.randn(64, 10, 512).cuda()
short_memory.initialize_memory()
short_memory.update_memory(input)
short_memory.extract_memory()