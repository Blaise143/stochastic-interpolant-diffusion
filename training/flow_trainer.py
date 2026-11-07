import torch
from torch.optim import Adam
from models.flow import Flow
from sampling.ode_solver import EulerSolver
from data.dataset import EMNISTDataLoader
import torchvision
from tqdm import tqdm
from training.logger import ExperimentLogger


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
    device=get_device(),
    guided=False,
    guidance_scale=7.,
    model_path="checkpoint/flow_model.pth",
    logger_type="none",
    scheduler_type="linear",
    use_attention=False,
):
    """Trains a `Flow` model on EMNIST digits and checkpoints it.

    Every 5th epoch (and the first), logs generated samples (and guided
    samples per digit, if `guided`) to the configured experiment logger.

    Args:
        data_dir: Directory to read/download the EMNIST dataset into.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Adam learning rate.
        num_steps: Number of Euler steps used when logging sample images
            during training.
        device: Torch device to train on.
        guided: If True, trains with classifier-free guidance conditioning
            and also logs per-digit guided samples.
        guidance_scale: Guidance scale used when logging guided samples.
        model_path: Path to save the trained model's state dict to.
        logger_type: Experiment logger backend ('none', 'tensorboard', or
            'wandb').
        scheduler_type: Probability path scheduler ('linear',
            'trigonometric', or 'polynomial').
        use_attention: If True, uses a `Unet` with a bottleneck
            self-attention block instead of the pure-CNN default.
    """
    logger = ExperimentLogger(
        logger_type=logger_type,
        project_name="Flow-Matching-Emnist",
        run_name=f"Flow-Training-{'guided' if guided else 'unguided'}-{scheduler_type}"
    )

    dataloader = EMNISTDataLoader(data_dir=data_dir, batch_size=batch_size)
    train_loader = dataloader.get_train_dataloader()

    model = Flow(
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        use_attention=use_attention,
    ).to(device)
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

        logger.log_scalar(loss_name, avg_loss, epoch + 1)

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
                    logger.log_image("guided samples", grid, step=epoch + 1, caption="Guided Samples")

                unguided_samples = solver.sample_loop(shape=(8, 1, 28, 28))
                unguided_samples = unguided_samples.clamp(0, 1).cpu()
                unguided_grid = torchvision.utils.make_grid(
                    unguided_samples, nrow=4)
                logger.log_image("unguided_samples", unguided_grid, step=epoch + 1, caption="Unguided Samples")
    torch.save(model.state_dict(), model_path)
    logger.finish()
