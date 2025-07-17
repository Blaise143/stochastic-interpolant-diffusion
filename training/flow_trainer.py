import torch
from torch.optim import Adam
import wandb
from models.flow import Flow
from sampling.ode_solver import EulerSolver
from data.dataset import EMNISTDataLoader
import torchvision
from torch.utils.data import TensorDataset, DataLoader  # for sanity check later
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train(
    data_dir="dataset",
    epochs=10,
    batch_size=200,
    lr=1e-3,
    num_steps=500,
    device=get_device(),  # "cuda" if torch.cuda.is_available() else "cpu"
    guided=False,
    guidance_scale=7.,
    model_path="checkpoint/flow_model.pth"
):
    wandb.init(project="Flow-Matching-Emnist", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_steps": num_steps,
        "guided": guided,
        "guidance_scale": guidance_scale
    })

    dataloader = EMNISTDataLoader(data_dir=data_dir, batch_size=batch_size)
    train_loader = dataloader.get_train_dataloader()

    model = Flow(guidance_scale=guidance_scale).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    solver = EulerSolver(model=model, num_steps=num_steps)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if guided:
                loss = model.compute_loss(batch, labels)
            else:
                loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_name = "guided loss" if guided else "unguided loss"

        wandb.log({loss_name: avg_loss, "epoch": epoch + 1})

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        model.eval()
        if epoch % 5 == 0 or epoch == 0:
            with torch.no_grad():
                if guided:
                    all_samples = list()
                    for digit in range(10):
                        labels = torch.full((2,), digit, device=device)
                        samples = solver.sample_loop(
                            shape=(2, 1, 28, 28), labels=labels, guidance_scale=guidance_scale)
                        all_samples.append(samples)
                    samples = torch.cat(all_samples, dim=0)
                    samples = samples.clamp(0, 1).cpu()
                    grid = torchvision.utils.make_grid(samples, nrow=5)
                    wandb.log({"guided samples": wandb.Image(grid)})

                unguided_samples = solver.sample_loop(shape=(8, 1, 28, 28))
                unguided_samples = unguided_samples.clamp(0, 1).cpu()
                unguided_grid = torchvision.utils.make_grid(
                    unguided_samples, nrow=4)
                wandb.log({"unguided_samples": wandb.Image(unguided_grid)})
    torch.save(model.state_dict(), model_path)
    wandb.finish()
