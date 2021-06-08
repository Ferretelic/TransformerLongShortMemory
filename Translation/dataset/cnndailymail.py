import os
from collections import Counter
import pickle
import re
import random

import pyprind
from sklearn.model_selection import KFold, train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import torch
import numpy as np

def create_indices(dataset_path):
  kfold = KFold(n_splits=5, random_state=0, shuffle=True)

  indices = {}
  for index, (train_indices, test_indices) in enumerate(kfold.split(np.arange(len(os.listdir(os.path.join(dataset_path, "stories"))))[:20000])):
    train_indices, validation_indices = train_test_split(train_indices, random_state=0)
    indices[index] = [train_indices, validation_indices, test_indices]

  with open(os.path.join(dataset_path, "pickle", "indices.pkl"), "wb") as f:
    pickle.dump(indices, f)


def load_indices(dataset_path, mode):
  with open(os.path.join(dataset_path, "pickle", "indices.pkl"), "rb") as f:
    train_indices, validation_indices, test_indices = pickle.load(f)[mode]

  return train_indices, validation_indices, test_indices

def create_vocabulary(dataset_path):
  counter = Counter()
  tokenizer = get_tokenizer("basic_english")
  bar = pyprind.ProgBar(len(os.listdir(os.path.join(dataset_path, "stories"))))

  for file_name in os.listdir(os.path.join(dataset_path, "stories")):
    with open(os.path.join(dataset_path, "stories", file_name), "r") as f:
      for sentence in f.read().split("@highlight"):
        counter.update(tokenizer(re.sub("[^A-Za-z0-9]+", " ", str(sentence).lower())))
      bar.update()

  vocab = Vocab(counter, specials=["<pad>", "<unk>"], max_size=19998)

  with open(os.path.join(dataset_path, "pickle", "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

class CNNDailyMailDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, max_length, indices):

    file_names = os.listdir(os.path.join(dataset_path, "stories"))
    self.files = np.array([os.path.join(dataset_path, "stories", file_name) for file_name in file_names])[indices]

    self.vocab = self.load_vocabulary(dataset_path)
    self.tokenizer = get_tokenizer("basic_english")

    self.max_length = max_length

  def __getitem__(self, index):
    with open(self.files[index], "r") as f:
      text = [sentence for sentence in f.read().split("@highlight")]

    sentences = [[self.vocab[token] for token in self.tokenizer(self.clean_text(sentence))] for sentence in text[0].split(".")[:-1]]
    summary = [self.vocab[token] for token in self.tokenizer(self.clean_text(random.choice(text[1:])))]

    sentences = torch.tensor([(sentence + [0] * self.max_length)[:self.max_length] for sentence in sentences], dtype=torch.long)
    summary = torch.tensor((summary + [0] * self.max_length)[:self.max_length], dtype=torch.long)
    return sentences, summary


  def __len__(self):
    return len(self.files)

  def load_vocabulary(self, dataset_path):
    with open(os.path.join(dataset_path, "pickle", "vocab.pkl"), "rb") as f:
      vocab = pickle.load(f)
    return vocab

  def clean_text(self, text):
    return re.sub("[^A-Za-z0-9]+", " ", str(text).lower())
