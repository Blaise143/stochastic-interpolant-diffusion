import torch
from torch.optim import Adam
import wandb
from models.diffusion import Diffusion
from sampling.sde_solver import EulerMaruyamaSolver
from data.dataset import EMNISTDataLoader
import torchvision

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
    device=get_device()#"cuda" if torch.cuda.is_available() else "cpu"
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

    for epoch in range(epochs):
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
        with torch.no_grad():
            samples = solver.sample_loop(shape=(8, 1, 28, 28))
            samples = samples.clamp(0, 1).cpu()
            grid = torchvision.utils.make_grid(samples, nrow=4)
            wandb.log({"samples": wandb.Image(grid)})#, caption="samples")], "epoch": epoch + 1})
#            images = [wandb.Image(
 #               samples[i][0], caption=f"Sample {i+1}") for i in range(4)]
  #          wandb.log({"samples": images, "epoch": epoch + 1})

    torch.save(model.state_dict(), "diffusion_model.pth")
    wandb.finish()


if __name__ == "__main__":
    train()
