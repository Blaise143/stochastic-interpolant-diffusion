import torch
from torch.optim import Adam
import wandb
from models.diffusion import Diffusion
from sampling.sde_solver import EulerMaruyamaSolver
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
    device=get_device()  # "cuda" if torch.cuda.is_available() else "cpu"
):
    wandb.init(project="diffusion-emnist", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_steps": num_steps,
    })

    dataloader = EMNISTDataLoader(data_dir=data_dir, batch_size=batch_size)
    train_loader = dataloader.get_train_dataloader()

    model = Diffusion().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    solver = EulerMaruyamaSolver(model, num_steps=num_steps)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        wandb.log({"loss": avg_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        model.eval()
        if epoch % 500 == 0:
            with torch.no_grad():
                samples = solver.sample_loop(shape=(8, 1, 28, 28))
                samples = samples.clamp(0, 1).cpu()
                grid = torchvision.utils.make_grid(samples, nrow=4)
                # , caption="samples")], "epoch": epoch + 1})
                wandb.log({"samples": wandb.Image(grid)})
    #            images = [wandb.Image(
    #               samples[i][0], caption=f"Sample {i+1}") for i in range(4)]
    #          wandb.log({"samples": images, "epoch": epoch + 1})

    torch.save(model.state_dict(), "diffusion_model.pth")
    wandb.finish()


def train_one_example(
    data_dir="dataset",
    epochs=10,
    lr=1e-3,
    num_steps=10,
    device=get_device()
):
    wandb.init(project="diffusion-emnist", config={
        "epochs": epochs,
        "learning_rate": lr,
        "num_steps": num_steps,
    })

    orig_loader = EMNISTDataLoader(
        data_dir=data_dir, batch_size=1, num_workers=0).get_train_dataloader()
    one_batch = next(iter(orig_loader))
    x, y = one_batch
    # x, y, = x[0], y[0]

    # import torchvision
    # real_img = x.clamp(0, 1).cpu()
    # grid = torchvision.utils.make_grid(real_img, nrow=1)
    # wandb.log({"real_sample": wandb.Image(
    #     grid, caption="Real EMNIST Example")})

    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=1,
                              shuffle=False, num_workers=0)

    model = Diffusion().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    solver = EulerMaruyamaSolver(model, num_steps=num_steps)

    for epoch in range(epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item(), "epoch": epoch + 1})
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            samples = solver.sample_loop(shape=(1, 1, 28, 28))
            samples = samples.clamp(0, 1).cpu()
            grid = torchvision.utils.make_grid(samples, nrow=1)
            wandb.log({"samples": wandb.Image(grid)})

    torch.save(model.state_dict(), "diffusion_model_one_example.pth")
    wandb.finish()


if __name__ == "__main__":
    train()
