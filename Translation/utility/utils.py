import shutil
import os
import sys
import json

import torch
import numpy as np
import pyprind

def output_setting(model_name):
  project_path = "/home/shouki/Desktop/Programming/Python/AI/Research/TransformerLongShortMemory"
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
    running_loss, validation_epoch_loss = 0.0, 0.0

    print("Epoch: {}".format(epoch + 1))
    print("Training Model")

    for sources, targets, num_sentences in train_dataloader:
      sources, targets, num_sentences = sources.to(device), targets.to(device), num_sentences.to(device)
      model.initialize_memory()
      for source, target in zip(sources.transpose(0, 1), targets.transpose(0, 1)):
        print(source.size(), target.size())
        break
      break
    break

  #       running_loss = 0.0

  #       for sentence in sentences:
  #         sentence, summary = sentence.to(device), summary.to(device)
  #         optimizer.zero_grad()

  #         source_mask, target_mask = generate_mask(sentence, summary)
  #         output = model(sentence, summary, source_mask, target_mask)
  #         loss = criterion(output.squeeze(), summary.squeeze())

  #         loss.backward()
  #         optimizer.step()
  #         running_loss += loss.item()

  #       train_epoch_loss += (running_loss / float(sentences.size(0)))

  #     if (document_index + 1) % 10 == 0:
  #       print("Index: {}".format(document_index + 1))
  #       print("Loss: {}".format(loss.item()))

  #   with torch.no_grad():
  #     model.eval()
  #     print("Evaluating Model")
  #     bar = pyprind.ProgBar(len(validation_dataset), stream=sys.stdout)

  #     for document_index in range(len(validation_dataset)):
  #       sentences, summary = [data.to(device) for data in validation_dataset[document_index]]
  #       if sentences.size(0) != 0:
  #         sentences, summary = sentences.unsqueeze(1), summary.unsqueeze(0)
  #         model.initialize_memory()

  #         for sentence in sentences:
  #           source_mask, target_mask = generate_mask(sentence, summary)
  #           output = model(sentence, summary, source_mask, target_mask)
  #         loss = criterion(output.squeeze(), summary.squeeze())

  #         validation_epoch_loss += (loss.item())
  #         bar.update()

  #   torch.save(model, os.path.join(model_path, "model_{}.pth".format(epoch + 1)))

  #   train_losses.append(train_epoch_loss / float(len(train_dataset)))
  #   validation_losses.append(validation_epoch_loss / float(len(validation_dataset)))
  #   scheduler.step(validation_losses[-1])

  # history = {"train_losses": train_losses, "validation_losses": validation_losses}
  # return history

def generate_mask(src, tgt):
  source_mask = (src != 0).unsqueeze(1)
  target_mask = (tgt != 0).unsqueeze(1)
  nopeak_mask = np.triu(np.ones((1, tgt.size(1), tgt.size(1))), k=1).astype(np.uint8)
  nopeak_mask = torch.tensor(nopeak_mask == 0).cuda()
  target_mask = target_mask & nopeak_mask

  return source_mask, target_mask