import torch
import torch.nn as nn

from models import Diffusion
from sampling import EulerMaruyamaSolver
from training import train, train_one_example
if __name__ == "__main__":
    num_steps, num_epochs = 1000, 500
    epochs = num_epochs*num_epochs
    train(num_steps=num_steps, epochs=epochs, batch_size=200)

    # train_one_example(num_steps=num_steps, epochs=epochs, lr=3e-4)
