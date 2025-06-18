import torch
import torch.nn as nn

from models import Diffusion
from sampling import EulerMaruyamaSolver
from training import train
if __name__ == "__main__":
    train(num_steps=1000, epochs=1000, batch_size=200)
