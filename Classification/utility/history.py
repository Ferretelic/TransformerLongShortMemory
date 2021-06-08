import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns

def plot_history(history, history_path):
  train_losses, validation_losses = history["train_losses"], history["validation_losses"]

  plt.figure()
  plt.title("Loss")
  sns.lineplot(x=range(len(train_losses)), y=train_losses, legend="brief", label="train loss")
  sns.lineplot(x=range(len(validation_losses)), y=validation_losses, legend="brief", label="validation loss")
  plt.xlabel("Epoch")
  plt.ylabel("Cross Entropy Loss")
  plt.savefig(os.path.join(history_path, "loss.png"))

  with open(os.path.join(history_path, "history.pkl"), "wb") as f:
    pickle.dump(history, f)
