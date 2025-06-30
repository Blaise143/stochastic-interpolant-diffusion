import torch
import torch.nn as nn
from training.flow_trainer import train, train_one_example

if __name__ == "__main__":
    train(epochs=1000)
