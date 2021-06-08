import os
from collections import Counter
import pickle
import re

import torch
import torchtext
import pyprind
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def create_pg19_vocabulary(dataset_path):
  tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
  counter = Counter()
  text_path = os.path.join(dataset_path, "train")
  bar = pyprind.ProgBar(len(os.listdir(text_path)))

  for text_name in os.listdir(text_path):
    with open(os.path.join(text_path, text_name), "r") as f:
      counter.update(tokenizer(f.read()))
    bar.update()

  vocab = torchtext.vocab.Vocab(counter, max_size=19998, specials=["<pad>", "<unk>"])
  with open(os.path.join(dataset_path, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

class PG19Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, max_len, mode):
    self.max_len = max_len
    self.vocab = self.load_vocab(dataset_path)
    text_path = os.path.join(dataset_path, mode)

    texts = [os.path.join(text_path, text_name) for text_name in os.listdir(text_path)]
    self.texts = texts
    self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

  def __getitem__(self, index):
    with open(self.texts[index], "r") as f:
      sentences = np.array([self.vocab[token] for token in self.tokenizer(f.read())])

    length = sentences.shape[0] - (sentences.shape[0] % self.max_len)
    sentences = np.split(sentences[:length], sentences.shape[0] // self.max_len)

    sources = []
    targets = []
    for index in range(len(sentences) - 1):
      sources.append(sentences[index])
      targets.append(sentences[index + 1])

    sources = torch.tensor(sources, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return sources, targets

  def __len__(self):
    return len(self.texts)

  def load_vocab(self, dataset_path):
    with open(os.path.join(dataset_path, "vocab.pkl"), "rb") as f:
      vocab = pickle.load(f)

    return vocab

  def get_vocab_size(self, dataset_path):
    return len(self.load_vocab(dataset_path))

def create_bbcnews_indices(dataset_path):
  dataset_size = pd.read_csv(os.path.join(dataset_path, "train.csv")).shape[0]
  kfold = KFold(n_splits=5, shuffle=True, random_state=0)

  indices = {}
  for index, (train_indices, validation_indices) in enumerate(kfold.split(np.arange(dataset_size))):
    indices[index] = [train_indices, validation_indices]

  with open("indice.pkl", "wb") as f:
    pickle.dump(indices, f)

def create_bbcnews_vocabulary(dataset_path):
  tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
  counter = Counter()
  texts = pd.read_csv(os.path.join(dataset_path, "train.csv"))["Text"].values
  for text in texts:
    counter.update(tokenizer(str(text)))

  vocab = torchtext.vocab.Vocab(counter, specials=["<pad>", "<unk>"])
  with open(os.path.join(dataset_path, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

def load_bbcnews_indices(mode):
  with open("indice.pkl", "rb") as f:
    train_indices, validation_indices = pickle.load(f)[mode]
  return train_indices, validation_indices

class BBCNewsDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, max_length, indices):
    self.max_length = max_length
    self.vocab = self.load_vocab(dataset_path)
    self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    data = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    self.label2index = {label: index for index, label in enumerate(np.unique(data["Category"].values))}

    self.texts = data["Text"].values[indices]
    self.labels = data["Category"].values[indices]

  def __getitem__(self, index):
    sentences = self.texts[index].split(".")[:-1]
    source = torch.zeros(len(sentences), self.max_length, dtype=torch.long)

    for order, text in enumerate(sentences):
      tokenized = self.tokenizer(self.clean_text(text))

      if len(tokenized) == 0:
        continue

      text = [self.vocab[token] for token in tokenized]
      if len(text) > self.max_length:
        source[order, :] = torch.tensor(text[:self.max_length], dtype=torch.long)
      else:
        source[order, :len(text)] = torch.tensor(text, dtype=torch.long)

    label = self.label2index[self.labels[index]]
    label = torch.tensor(label, dtype=torch.long)
    print(source.size())

  def __len__(self):
    return len(self.texts)

  def load_vocab(self, dataset_path):
    with open(os.path.join(dataset_path, "vocab.pkl"), "rb") as f:
      vocab = pickle.load(f)
    return vocab

  def clean_text(self, text):
    return re.sub("[^A-Za-z0-9]+", " ", str(text).lower())

class CNNDailyMailDataset(torch.utils.data.Dataset):
