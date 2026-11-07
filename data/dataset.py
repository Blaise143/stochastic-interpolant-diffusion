import torch
from torch.utils.data import DataLoader
from typing import Callable, Optional
from data.base import BaseDataset, BaseDataLoader
from data.transforms import emnist_transform
import torchvision

TransformType = Callable[[torch.Tensor], torch.Tensor]


class EMNIST(BaseDataset):
    """The EMNIST 'digits' split, wrapped as a `BaseDataset`.

    Attributes:
        transforms: Preprocessing pipeline applied to each image.
        dataset: The underlying torchvision EMNIST dataset.
    """

    def __init__(
        self,
        data_dir: str,
        transforms: Optional[TransformType] = None,
        train: bool = True,
        download: bool = True,
    ):
        """
        Args:
            data_dir: Directory to read/download the raw EMNIST data into.
            transforms: Preprocessing pipeline; defaults to `emnist_transform()`.
            train: If True, loads the training split, else the test split.
            download: If True, downloads the dataset to `data_dir` if missing.
        """
        super().__init__()

        self.transforms = transforms or emnist_transform()

        self.dataset = torchvision.datasets.EMNIST(
            root=data_dir,
            split="digits",
            train=train,
            download=download,
            transform=self.transforms,
        )

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Returns the (image, label) pair at `idx`."""
        image, label = self.dataset[idx]
        return image, label


class EMNISTDataLoader(BaseDataLoader):
    """Builds train/test `DataLoader`s over the EMNIST digits dataset.

    Attributes:
        batch_size: Batch size used by both dataloaders.
        num_workers: Number of worker processes used by both dataloaders.
        train_dataset: The training split, as an `EMNIST` dataset.
        test_dataset: The test split, as an `EMNIST` dataset.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 50,
        num_workers: int = 4,
        transforms: Optional[TransformType] = None,
    ):
        """
        Args:
            data_dir: Directory to read/download the raw EMNIST data into.
            batch_size: Batch size for both train and test dataloaders.
            num_workers: Number of worker processes for data loading.
            transforms: Preprocessing pipeline shared by both splits.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = EMNIST(
            data_dir=data_dir, transforms=transforms, train=True, download=True
        )

        self.test_dataset = EMNIST(
            data_dir=data_dir, transforms=transforms, train=False, download=True
        )

    def get_train_dataloader(self, sampler: Optional[torch.utils.data.Sampler] = None) -> DataLoader:
        """Returns a DataLoader over the training split.

        Args:
            sampler: Optional sampler (e.g. `DistributedSampler`) controlling
                how the data is split/ordered. When given, `shuffle` is left
                unset since a sampler and `shuffle=True` are mutually
                exclusive in `DataLoader`.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def get_test_dataloader(self) -> DataLoader:
        """Returns an unshuffled DataLoader over the test split."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    data_dir = "dataset"
    dataloader = EMNISTDataLoader(data_dir=data_dir)
    test_loader = dataloader.get_test_dataloader()
    train_loader = dataloader.get_train_dataloader()
    for batch, labels in train_loader:
        print(batch[0].min(), batch[0].max(), labels[0])
