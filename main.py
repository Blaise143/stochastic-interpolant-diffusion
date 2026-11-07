import torch
import hydra
from omegaconf import DictConfig
import torchvision
from training.flow_trainer import train as train_flow
from models.flow import Flow
from sampling.ode_solver import EulerSolver, MidpointSolver
from sampling.sde_solver import EulerMaruyamaSolver
from training.logger import ExperimentLogger


def get_device(device_arg=None):
    """Resolves the torch device to use.

    Args:
        device_arg: Explicit device request ('cuda', 'mps', or 'cpu'), or
            None to auto-detect the best available device.

    Returns:
        A torch.device.
    """
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_arg == "cpu":
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def sample(
    model_type="flow",
    solver_type="euler",
    scheduler_type="linear",
    model_path="checkpoints/flow_model.pth",
    num_steps=500,
    num_samples=10,
    guided=True,
    guidance_scale=7.5,
    device=None,
    logger_type="none",
    digits=None,
    save_path=None,
    use_attention=False,
):
    """
    Sample images from a flow or diffusion model.

    Returns:
        torch.Tensor: Generated samples
    """
    device = get_device(device)

    logger = ExperimentLogger(
        logger_type=logger_type,
        project_name="Flow-Matching-Sampling",
        run_name=f"{model_type.capitalize()}-Sampling"
    )

    model = Flow(
        num_classes=10,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        use_attention=use_attention,
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")

    model.eval()

    if model_type == "flow":
        if solver_type.lower() == "midpoint":
            solver = MidpointSolver(model=model, num_steps=num_steps)
            print(f"Using Midpoint ODE solver for {model_type} model")
        else:
            solver = EulerSolver(model=model, num_steps=num_steps)
            print(f"Using Euler ODE solver for {model_type} model")
    else:  # diffusion
        solver = EulerMaruyamaSolver(model=model, num_steps=num_steps)
        print(f"Using Euler Maruyama solver for {model_type} model")

    if digits is not None:
        try:
            if isinstance(digits, str):
                parsed_digits = [int(d.strip()) for d in digits.split(',')]
            else:
                parsed_digits = [int(d) for d in digits]
            valid_digits = []

            for d in parsed_digits:
                if 0 <= d <= 9:
                    valid_digits.append(d)
                else:
                    print(f"Warning: Digit {d} is not in range [0-9] and will be ignored.")

            if valid_digits:
                digits_to_sample = valid_digits
                num_samples = 1
            else:
                digits_to_sample = list(range(10))
        except ValueError:
            digits_to_sample = list(range(10))
    else:
        digits_to_sample = list(range(10))
    samples_by_digit = []

    with torch.no_grad():
        for digit in digits_to_sample:
            print(f"Sampling digit {digit}...")
            labels = torch.full((num_samples,), digit, device=device)
            samples = solver.sample_loop(
                shape=(num_samples, 1, 28, 28), 
                labels=labels if guided else None, 
                guidance_scale=guidance_scale if guided else None
            )
            samples = samples.clamp(0, 1).cpu()
            samples_by_digit.append(samples)

    all_samples = torch.cat(samples_by_digit, dim=0)

    if digits is not None and num_samples == 1:
        grid = torchvision.utils.make_grid(all_samples, nrow=len(digits_to_sample))
    else:
        grid = torchvision.utils.make_grid(all_samples, nrow=num_samples)

    if model_type == "flow":
        if solver_type.lower() == "midpoint":
            method = "Midpoint ODE"
        else:
            method = "Euler ODE"
    else:
        method = "Euler Maruyama SDE"

    if len(digits_to_sample) == 10:
        caption = f"{method} Method: {num_samples} samples per digit. Rows from top to bottom: digits 0-9"
    else:
        digits_str = ", ".join(str(d) for d in digits_to_sample)
        if digits is not None and num_samples == 1:
            caption = f"{method} Method: One sample per digit. Digits: {digits_str}"
        else:
            caption = f"{method} Method: {num_samples} samples per digit. Rows from top to bottom: digits {digits_str}"

    log_title = f"{model_type.capitalize()} Model Samples"
    if digits is not None and num_samples == 1:
        log_title += " (One per digit)"
    else:
        log_title += f" ({num_samples} per digit)"

    logger.log_image(log_title, grid, step=0, caption=caption)
    logger.finish()

    if save_path is None:
        solver_suffix = f"_{solver_type.lower()}" if model_type == "flow" else ""
        if len(digits_to_sample) == 10:
            save_path = f"images/{model_type}{solver_suffix}_samples.png"
        else:
            digits_str = "_".join(str(d) for d in digits_to_sample)
            save_path = f"images/{model_type}{solver_suffix}_samples_digits_{digits_str}.png"

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torchvision.utils.save_image(grid, save_path)
    return all_samples


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "train_flow":
        train_flow(
            data_dir=cfg.data_dir,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            num_steps=cfg.num_steps,
            guided=cfg.guided,
            guidance_scale=cfg.guidance_scale,
            model_path=cfg.model_path,
            logger_type=cfg.logger_type,
            scheduler_type=cfg.scheduler_type,
            use_attention=cfg.attention,
        )
    elif cfg.mode == "sample":
        sample(
            model_type=cfg.model_type,
            solver_type=cfg.solver_type,
            scheduler_type=cfg.scheduler_type,
            model_path=cfg.model_path,
            num_steps=cfg.num_steps,
            num_samples=cfg.num_samples,
            guided=cfg.guided,
            guidance_scale=cfg.guidance_scale,
            device=cfg.device,
            logger_type=cfg.logger_type,
            digits=cfg.digits,
            save_path=cfg.save_path,
            use_attention=cfg.attention,
        )
    else:
        raise ValueError(f"Unknown mode {cfg.mode!r}, expected 'train_flow' or 'sample'")


if __name__ == "__main__":
    main()
