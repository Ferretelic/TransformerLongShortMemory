import sys
sys.path.append("..")

import torch

from architecture.transformers import TransformerEncoderLayer
from architecture.embeddings import position_encoding
from architecture.layers import Residual, feed_forward
from architecture.attentions import MultiHeadAttention

class ShortMemory(torch.nn.Module):
  def __init__(self, dim_model, short_memory_size, batch_size, max_length):
    super().__init__()

    self.batch_size = batch_size
    self.memory_index = 0
    self.short_memory_size = short_memory_size
    self.dim_model = dim_model
    self.memory_compressor = torch.nn.Conv1d(max_length, 1, kernel_size=1, stride=1, padding=0)

  def update_memory(self, encoder_output):
    batch_size = encoder_output.size(0)
    compressed_memory = self.memory_compressor(encoder_output).squeeze()
    self.short_memory[:batch_size, self.memory_index] = compressed_memory.clone().detach().cpu()

    if (self.memory_index + 1) == self.short_memory_size:
      self.memory_index = 0
    else:
      self.memory_index += 1

  def initialize_memory(self):
    self.short_memory = torch.zeros(self.batch_size, self.short_memory_size, self.dim_model, requires_grad=False, dtype=torch.float32).cpu()
    self.memory_index = 0

  def extract_memory(self, batch_size):
    index = range(self.memory_index - self.short_memory_size, self.memory_index)
    return self.short_memory[:batch_size, index, :].clone().detach().cuda()

class LongMemory(torch.nn.Module):
  def __init__(self, dim_model, short_memory_size, long_memory_size, batch_size):
    super().__init__()

    self.long_memory_size = long_memory_size
    self.dim_model = dim_model
    self.batch_size = batch_size
    self.compression = torch.nn.Conv1d(short_memory_size, long_memory_size, kernel_size=1, stride=1, padding=0)

  def update_memory(self, short_memory):
    batch_size = short_memory.size(0)
    compressed_memory = self.compression(short_memory)
    memory_sigmoid = torch.sigmoid(compressed_memory)
    self.long_memory[:batch_size] *= memory_sigmoid.clone().detach().cpu()
    self.long_memory[:batch_size] += (memory_sigmoid * torch.tanh(compressed_memory)).clone().detach().cpu()

  def initialize_memory(self):
    self.long_memory = torch.zeros(self.batch_size, self.long_memory_size, self.dim_model, requires_grad=False).cpu()

  def extract_memory(self, batch_size):
    return self.long_memory[:batch_size].clone().detach().cuda()

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

class TransformerEncoder(torch.nn.Module):
  def __init__(self, num_layers, dim_model, num_heads, dim_feedforward, dropout):
    super().__init__()

    self.dim_model = dim_model
    self.layers = torch.nn.ModuleList([
      TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
      for _ in range(num_layers)
    ])

  def forward(self, source, short_memory, long_memory):
    for layer in self.layers:
      source = layer(source)
    return source

class TransformerLongShortMemory(torch.nn.Module):
  def __init__(self,
               num_encoder_layers = 3,
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
    self.dim_model = dim_model
    self.token_embedding = torch.nn.Embedding(embedding_size, dim_model)
    self.linear = torch.nn.Linear(dim_model, embedding_size)

  def forward(self, source, target, sentence_index, sentence_lengths):
    source = self.token_embedding(source) + position_encoding(source.size(1), self.dim_model)
    batch_mask = sentence_lengths >= sentence_index
    batch_size = sum(batch_mask)
    source, target = source[batch_mask], target[batch_mask]

    short_memory = self.short_memory.extract_memory(batch_size)
    long_memory = self.long_memory.extract_memory(batch_size)

    if sentence_index != 0:
      encoder_output = self.encoder(source, short_memory, long_memory)
    else:
      encoder_output = self.encoder(source, source, source)

    self.short_memory.update_memory(encoder_output)
    self.long_memory.update_memory(short_memory)


    output = torch.softmax(self.linear(torch.mean(encoder_output, dim=1)), dim=-1)
    return output, target

  def initialize_memory(self):
    self.short_memory.initialize_memory()
    self.long_memory.initialize_memory()
