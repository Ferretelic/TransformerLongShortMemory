import os
from collections import Counter
import pickle
import re
import random

import torchtext
import torch
import numpy as np

def create_vocabulary(dataset_path):
  counter = Counter()
  tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

  with open(os.path.join(dataset_path, "texts", "wiki.train.tokens"), "r") as f:
      text = re.sub("[^A-Za-z0-9]+", " ", str(f.read()).lower())

  counter.update(tokenizer(text))
  vocab = torchtext.vocab.Vocab(counter)

  with open(os.path.join(dataset_path, "pickle", "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

class WikiText2Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, bpc_length, mode):

    with open(os.path.join(dataset_path, "texts", "wiki.{}.tokens".format(mode)), "r") as f:
      text = f.read()

    paragraphs = list(filter((" ").__ne__, text.split("\n =")))
    paragraphs = [" ".join(list(filter((" ").__ne__, paragraph.split("\n")[1:]))) for paragraph in paragraphs]
    self.texts = list(filter(None, paragraphs))

    self.vocab = self.load_vocabulary(dataset_path)
    self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    self.bpc_length = bpc_length
    self.texts = [text for text in self.texts if len(self.tokenizer(self.clean_text(text))) > self.bpc_length]

  def __getitem__(self, index):
    tokens = [self.vocab[token] for token in self.tokenizer(self.clean_text(self.texts[index]))]
    num_sentences = len(tokens) - self.bpc_length

    sources = np.zeros((num_sentences, self.bpc_length), dtype=np.int32)
    targets = np.zeros((num_sentences,), dtype=np.int32)
    for token_index in range(num_sentences):
      sources[token_index] = tokens[token_index: token_index + self.bpc_length]
      targets[token_index] = tokens[token_index + self.bpc_length]

    sources = torch.tensor(sources, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return sources, targets, num_sentences

  def __len__(self):
    return len(self.texts)

  def load_vocabulary(self, dataset_path):
    with open(os.path.join(dataset_path, "pickle", "vocab.pkl"), "rb") as f:
      vocab = pickle.load(f)
    return vocab

  def clean_text(self, text):
    return re.sub("[^A-Za-z0-9]+", " ", str(text).lower())

def collate_function(batch):
  sources, targets, num_sentences = zip(*batch)
  max_length = max(num_sentences)
  batch_sources = torch.zeros((len(sources), max_length, len(sources[0][0])), dtype=torch.long)
  batch_targets = torch.zeros((len(targets), max_length), dtype=torch.long)

  for index, (source, target, length) in enumerate(batch):
    batch_sources[index][:length] = source
    batch_targets[index][:length] = target

  sorted_index = np.argsort(num_sentences)
  num_sentences = torch.tensor(num_sentences, dtype=torch.long)
  sources, targets, num_sentences = batch_sources[sorted_index], batch_targets[sorted_index], num_sentences[sorted_index]
  return sources, targets, num_sentences

def create_dataloader(dataset_path, bpc_length, mode, batch_size, num_workers):
  dataset = WikiText2Dataset(dataset_path, bpc_length, mode)
  dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_function, batch_size=batch_size, num_workers=num_workers)
  return dataloader
