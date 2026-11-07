import os
import torch
import torchvision


class ExperimentLogger:
    """
    Unified experiment logger supporting 'none', 'tensorboard', and 'wandb'.
    """

    def __init__(self, logger_type: str = "none", project_name: str = "Flow-Matching", run_name: str = None, log_dir: str = "runs"):
        """
        Args:
            logger_type: Backend to use: 'none', 'tensorboard', or 'wandb'.
                Falls back to 'none' if the chosen backend isn't installed.
            project_name: Project name passed to the wandb backend.
            run_name: Run name passed to the wandb backend.
            log_dir: Directory for TensorBoard event files.
        """
        self.logger_type = logger_type.lower()
        self.writer = None
        self.wandb_run = None

        if self.logger_type == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"Initialized TensorBoard logger at {log_dir}")
            except ImportError:
                print("Warning: tensorboard not installed. Falling back to 'none' logger.")
                self.logger_type = "none"

        elif self.logger_type == "wandb":
            try:
                import wandb
                self.wandb_run = wandb.init(project=project_name, name=run_name)
                print(f"Initialized wandb logger for project {project_name}")
            except ImportError:
                print("Warning: wandb not installed. Falling back to 'none' logger.")
                self.logger_type = "none"
        else:
            print("Logger set to 'none'. No remote/board logging will be performed.")

    def log_scalar(self, name: str, value: float, step: int):
        """Logs a scalar value (no-op for the 'none' backend).

        Args:
            name: Metric name.
            value: Scalar value to log.
            step: Step/epoch the value corresponds to.
        """
        if self.logger_type == "tensorboard" and self.writer:
            self.writer.add_scalar(name, value, step)
        elif self.logger_type == "wandb":
            import wandb
            wandb.log({name: value, "epoch": step})

    def log_image(self, name: str, image_tensor: torch.Tensor, step: int = None, caption: str = None):
        """Logs an image (no-op for the 'none' backend).

        Args:
            name: Image/metric name.
            image_tensor: Image tensor; a batch (ndim 4) is combined into a
                grid for TensorBoard.
            step: Step/epoch the image corresponds to.
            caption: Caption shown alongside the image (wandb only).
        """
        if self.logger_type == "tensorboard" and self.writer:
            if image_tensor.ndim == 4:
                grid = image_tensor if image_tensor.shape[0] == 1 else torchvision.utils.make_grid(image_tensor)
            else:
                grid = image_tensor
            self.writer.add_image(name, grid, step if step is not None else 0)
        elif self.logger_type == "wandb":
            import wandb
            wandb.log({name: wandb.Image(image_tensor, caption=caption)})

    def finish(self):
        """Closes/finalizes the active logging backend, if any."""
        if self.logger_type == "tensorboard" and self.writer:
            self.writer.close()
        elif self.logger_type == "wandb":
            import wandb
            wandb.finish()
