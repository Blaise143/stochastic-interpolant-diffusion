import torch
import torch.nn as nn
import argparse
from training.flow_trainer import train as train_flow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flow Matching")
    parser.add_argument("--mode", type=str, default="train_flow",
                        choices=["train_flow", "sample"],
                        help="Mode to run in")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Bath size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="ode solver sampling steps")

    parser.add_argument("--guided", action="store_true",
                        help="if flagged, guidance gotta kick in.. classifier freee guidance")
    parser.add_argument("--guidance_scale", type=float,
                        default=7.5, help="guidance scale")

    parser.add_argument("--model_path", type=str, default="checkpoints/flow_model.pth",
                        help="Path to flow model checkpoint for sampling")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="number of samples per class")
    parser.add_argument("--save_path", type=str, default="samples.png",
                        help="Generated samples")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device used.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    train_flow(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_steps=args.num_steps,
        guided=args.guided,
        guidance_scale=args.guidance_scale,
        model_path=args.model_path
    )
