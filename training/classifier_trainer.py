import torch
from torch.optim import Adam
import wandb
from models.classifier import NoisyClassifier
from models.flow import Flow
from data.dataset import EMNISTDataLoader
import torchvision
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
    device=get_device(),
):
    wandb.init(project="Noisy Classifier", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    })

    dataloader = EMNISTDataLoader(data_dir=data_dir, batch_size=batch_size)
    train_loader = dataloader.get_train_dataloader()

    classifier = NoisyClassifier(
        in_channels=1,
        num_classes=10,
        channels_list=[32, 64, 128],
        embedding_dim=100,
    ).to(device)

    optimizer = Adam(classifier.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        classifier.train()
        total_loss = 0

        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)

            t = torch.rand(batch.size(0), 1, device=device)

            noise = torch.randn_like(batch)
            t_broadcast = t.view(-1, 1, 1, 1)
            noisy_batch = (1 - t_broadcast) * batch + t_broadcast * noise

            optimizer.zero_grad()
            loss = classifier.compute_loss(noisy_batch, t, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        wandb.log({"loss": avg_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == epochs - 1:
            classifier.eval()
            with torch.no_grad():
                test_batch, test_labels = next(
                    iter(dataloader.get_test_dataloader()))
                test_batch = test_batch.to(device)
                test_labels = test_labels.to(device)

                timesteps = torch.linspace(0, 0.9, 5).to(device)
                all_samples = []
                all_predictions = []

                for t_val in timesteps:
                    t = torch.full((test_batch.size(0), 1),
                                   t_val, device=device)

                    noise = torch.randn_like(test_batch)
                    t_broadcast = t.view(-1, 1, 1, 1)
                    noisy_batch = (1 - t_broadcast) * \
                        test_batch + t_broadcast * noise

                    logits = classifier(noisy_batch, t)
                    predictions = torch.argmax(logits, dim=1)

                    all_samples.append(noisy_batch[:8])
                    all_predictions.append(predictions[:8])

                samples = torch.cat(all_samples, dim=0)
                samples = samples.clamp(0, 1).cpu()
                grid = torchvision.utils.make_grid(samples, nrow=8)

                predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
                true_labels = test_labels[:8].repeat(
                    len(timesteps)).cpu().numpy()
                caption = f"Timesteps: {timesteps.cpu().numpy()}\n"
                caption += f"Predictions: {predictions}\n"
                caption += f"True labels: {true_labels}"

                wandb.log({
                    "samples": wandb.Image(grid, caption=caption),
                    "epoch": epoch + 1
                })

    torch.save(classifier.state_dict(), "classifier_model.pth")
    wandb.finish()
