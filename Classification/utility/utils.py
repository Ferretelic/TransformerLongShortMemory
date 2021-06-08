import shutil
import os
import sys
import json

import torch
import numpy as np
import pyprind

def output_setting(model_name):
  project_path = "/home/shouki/Desktop/Programming/Python/AI/Research/TransformerLongShortMemory/Classification"
  model_path, history_path = os.path.join(project_path, "models", model_name), os.path.join(project_path, "histories", model_name)

  if os.path.exists(model_path):
    shutil.rmtree(model_path)

  if os.path.exists(history_path):
    shutil.rmtree(history_path)

  os.mkdir(model_path)
  os.mkdir(history_path)

  return model_path, history_path

def load_train_setting(model_name):
  with open("./setting.json", "r") as f:
    setting = json.load(f)[model_name]

  dataset_path = setting["dataset_path"]
  short_memory_size = setting["short_memory_size"]
  long_memory_size = setting["long_memory_size"]
  bpc_length = setting["bpc_length"]
  num_epochs = setting["num_epochs"]
  batch_size = setting["batch_size"]
  num_workers = setting["num_workers"]
  device = torch.device(setting["device"])

  return dataset_path, short_memory_size, long_memory_size, bpc_length, num_epochs, batch_size, num_workers, device


def train_model(model, criterion, optimizer, scheduler, num_epochs, train_dataloader, validation_dataloader, device, model_path):
  train_losses, validation_losses = [], []

  for epoch in range(num_epochs):
    model.train()
    bar = pyprind.ProgBar(len(train_dataloader), stream=sys.stdout)
    running_loss, evaluating_loss = 0.0, 0.0

    print("Epoch: {}".format(epoch + 1))
    print("Training Model")

    for sources, targets, num_sentences in train_dataloader:
      sources, targets, num_sentences = sources.to(device), targets.to(device), num_sentences.to(device)
      batch_loss = 0.0

      model.initialize_memory()
      for sentence_index, (source, target) in enumerate(zip(sources.transpose(0, 1), targets.transpose(0, 1))):
        output, target = model(source, target, sentence_index, num_sentences)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()

      running_loss += batch_loss / sources.size(0)
      bar.update()

    with torch.no_grad():
      model.eval()
      bar = pyprind.ProgBar(len(validation_dataloader))

      print("Evaluating Model")

      for sources, targets, num_sentences in validation_dataloader:
        sources, targets, num_sentences = sources.to(device), targets.to(device), num_sentences.to(device)
        batch_loss = 0.0

        model.initialize_memory()
        for sentence_index, source in enumerate(source.transpose(0, 1)):
          output = model(source, sentence_index, num_sentences)
          loss = criterion(output, num_sentences)
          batch_loss += loss.item()

        evaluating_loss += batch_loss / sources.size(0)
        bar.update()

    torch.save(model, os.path.join(model_path, "model_{}.pth".format(epoch + 1)))

    train_losses.append(running_loss / float(len(train_dataset)))
    validation_losses.append(evaluating_loss / float(len(validation_dataset)))
    scheduler.step(validation_losses[-1])

  history = {"train_losses": train_losses, "validation_losses": validation_losses}
  return history

def generate_mask(src, tgt):
  source_mask = (src != 0).unsqueeze(1)
  target_mask = (tgt != 0).unsqueeze(1)
  nopeak_mask = np.triu(np.ones((1, tgt.size(1), tgt.size(1))), k=1).astype(np.uint8)
  nopeak_mask = torch.tensor(nopeak_mask == 0).cuda()
  target_mask = target_mask & nopeak_mask

  return source_mask, target_mask