import torch
import torch.nn as nn

from models import Diffusion
from sampling import EulerMaruyamaSolver
from training import train
if __name__ == "__main__":
    train()
