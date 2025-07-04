import torch
from models.flow import Flow
from models.classifier import NoisyClassifier
from sampling.ode_solver import EulerSolver


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class ClassifierGuidedSampler:
    """
    So far only works for probability flow, so euler's method added.
    """

    def __init__(
        self,
        flow_model: Flow,
        classifier: NoisyClassifier,
        num_steps: int = 500,
        guidance_scale: float = 1.0,
    ):
        """
        Sampler that uses classifier guidance for the flow model
        """
        self.flow_model = flow_model
        self.classifier = classifier
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.device = next(flow_model.parameters()).device
        self.solver = EulerSolver(model=flow_model, num_steps=num_steps)

    def step(self, xt, t, labels=None):
        """
        One step of euler but with the classifier guidance added 
        """
        flow_drift = self.flow_model(xt, t, labels)

        if labels is not None and self.guidance_scale > 0:
            with torch.enable_grad():
                guidance_ = self.classifier.get_guidance(xt, t, labels)

            drift = flow_drift + self.guidance_scale * guidance_
        else:
            drift = flow_drift

        x_next = xt + drift * (1.0 / self.num_steps)
        return x_next

    def sample(self, shape=(4, 1, 28, 28), labels=None):
        """
        Samples using classifier guidance
        """
        xt = torch.randn(shape, device=self.device)

        for i in range(self.num_steps):
            t = torch.full((shape[0], 1), i /
                           self.num_steps, device=self.device)
            xt = self.step(xt, t, labels)

        return xt


def sample_with_classifier_guidance(
    flow_model_path,
    classifier_model_path,
    num_samples=2,
    num_steps=500,
    guidance_scale=1.0,
    device=get_device(),
    save_path=None,
):
    """
    Generate samples using classifier guidance
    """
    flow_model = Flow().to(device)
    flow_model.load_state_dict(torch.load(
        flow_model_path, map_location=device))
    flow_model.eval()

    classifier = NoisyClassifier().to(device)
    classifier.load_state_dict(torch.load(
        classifier_model_path, map_location=device))
    classifier.eval()

    sampler = ClassifierGuidedSampler(
        flow_model=flow_model,
        classifier=classifier,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
    )

    all_samples = []
    for digit in range(10):
        labels = torch.full((num_samples,), digit, device=device)
        samples = sampler.sample(shape=(num_samples, 1, 28, 28), labels=labels)
        all_samples.append(samples)

    samples = torch.cat(all_samples, dim=0)

    if save_path:
        import torchvision
        samples_grid = torchvision.utils.make_grid(samples, nrow=num_samples)
        torchvision.utils.save_image(samples_grid, save_path)

    return samples
