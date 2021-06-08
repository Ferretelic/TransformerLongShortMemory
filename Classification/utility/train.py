import sys
sys.path.append("..")

import torch

from dataset.wikitext2 import create_dataloader
from architecture.long_memory import TransformerLongShortMemory
from utility.utils import train_model, output_setting, load_train_setting
from utility.history import plot_history

model_name = "WikiText2"
dataset_path, short_memory_size, long_memory_size, bpc_length, num_epochs, batch_size, num_workers, device = load_train_setting(model_name)
model_path, history_path = output_setting(model_name)

train_dataloader = create_dataloader(dataset_path, bpc_length, "train", batch_size, num_workers)
validation_dataloader = create_dataloader(dataset_path, bpc_length, "validation", batch_size, num_workers)
test_dataloader = create_dataloader(dataset_path, bpc_length, "test", batch_size, num_workers)

embedding_size = len(train_dataloader.dataset.vocab)
transformer = TransformerLongShortMemory(short_memory_size=short_memory_size, long_memory_size = long_memory_size, embedding_size=embedding_size, batch_size=batch_size, max_length=bpc_length).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

history = train_model(transformer, criterion, optimizer, scheduler, num_epochs, train_dataloader, validation_dataloader, device, model_path)

plot_history(history, history_path)
