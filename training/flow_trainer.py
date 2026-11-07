import os
from typing import cast
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler
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


def is_distributed() -> bool:
    """Whether this process was launched by `torchrun` for multi-GPU DDP training."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    """Joins the DDP process group and pins this process to its assigned GPU.

    Expects to be launched via `torchrun`, which sets `LOCAL_RANK`, `RANK`,
    and `WORLD_SIZE` in the environment.

    Returns:
        (local_rank, rank, world_size)
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def train(
    data_dir="dataset",
    epochs=10,
    batch_size=200,
    lr=1e-3,
    num_steps=500,
    device=None,
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

    Runs single-GPU/CPU/MPS by default. For multi-GPU DDP training, launch
    with `torchrun --nproc_per_node=<num_gpus> main.py ...` instead of
    `python main.py ...`; this function then automatically wraps the model
    in `DistributedDataParallel`, shards the dataset across GPUs, and
    restricts logging/checkpointing/sampling to rank 0.

    Args:
        data_dir: Directory to read/download the EMNIST dataset into.
        epochs: Number of training epochs.
        batch_size: Per-GPU training batch size.
        lr: Adam learning rate.
        num_steps: Number of Euler steps used when logging sample images
            during training.
        device: Torch device to train on. Ignored (and derived from
            `LOCAL_RANK` instead) under DDP; defaults to `get_device()`
            otherwise.
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
    distributed = is_distributed()
    local_rank = 0
    if distributed:
        local_rank, rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = device or get_device()

    is_main = rank == 0

    logger = ExperimentLogger(
        logger_type=logger_type if is_main else "none",
        project_name="Flow-Matching-Emnist",
        run_name=f"Flow-Training-{'guided' if guided else 'unguided'}-{scheduler_type}"
    )

    dataloader = EMNISTDataLoader(data_dir=data_dir, batch_size=batch_size)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataloader.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = dataloader.get_train_dataloader(sampler=sampler)

    model = Flow(
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        use_attention=use_attention,
    ).to(device)

    # Sampling/checkpointing always go through the undecorated model: it
    # only ever runs on rank 0, and DDP's forward() broadcasts buffers (e.g.
    # BatchNorm stats) on every call, so calling it on one rank only (not in
    # lockstep with the others) would deadlock the other ranks.
    raw_model = model
    if distributed:
        raw_model = cast(Flow, torch.nn.SyncBatchNorm.convert_sync_batchnorm(model))
        model = DDP(raw_model, device_ids=[local_rank])

    optimizer = Adam(model.parameters(), lr=lr)
    solver = EulerSolver(model=raw_model, num_steps=num_steps)

    epoch_iter = tqdm(range(epochs)) if is_main else range(epochs)
    for epoch in epoch_iter:
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Calling `model(...)` (not `model.compute_loss(...)`) is required
            # under DDP: only forward() triggers gradient synchronization.
            loss = model(batch, labels=labels if guided else None, return_loss=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        if distributed:
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / world_size).item()

        if is_main:
            loss_name = "guided loss" if guided else "unguided loss"
            logger.log_scalar(loss_name, avg_loss, epoch + 1)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

            if epoch % 5 == 0 or epoch == 0:
                raw_model.eval()
                with torch.no_grad():
                    if guided:
                        all_samples = list()
                        for digit in range(10):
                            digit_labels = torch.full((2,), digit, device=device)
                            samples = solver.sample_loop(
                                shape=(2, 1, 28, 28), labels=digit_labels, guidance_scale=guidance_scale)
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

        if distributed:
            dist.barrier()

    if is_main:
        torch.save(raw_model.state_dict(), model_path)
        logger.finish()

    if distributed:
        dist.destroy_process_group()
