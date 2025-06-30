import torch
import torch.nn as nn
import argparse
from training.flow_trainer import train, train_one_example


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flow Matching")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Bath size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="ode solver sampling steps")
    parser.add_argument("--guided", action="store_true",
                        help="if flagged, guidance gotta kick in")
    parser.add_argument("--guidance_scale", type=float,
                        default=7.5, help="scale for classifier-free guidance")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_steps=args.num_steps,
        guided=args.guided,
        guidance_scale=args.guidance_scale
    )
