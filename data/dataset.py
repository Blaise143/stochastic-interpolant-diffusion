from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from typing import Callable
from data.base import BaseDataset, BaseDataLoader
from data.transforms import emnist_transform

import torchvision
from typing import Optional

TransformType = Callable[[torch.Tensor], torch.Tensor]


class EMNIST(BaseDataset):
    def __init__(self, data_dir: str, transforms: Optional[TransformType] = None, train: bool = True, download: bool = True):
        super().__init__()

        self.transforms = transforms or emnist_transform()

        self.dataset = torchvision.datasets.EMNIST(
            root=data_dir,
            split="digits",
            train=train,
            download=download,
            transform=self.transforms
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, label


class EMNISTDataLoader(BaseDataLoader):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 50,
                 num_workers: int = 4,
                 transforms: Optional[TransformType] = None):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = EMNIST(
            data_dir=data_dir,
            transforms=transforms,
            train=True,
            download=True
        )

        self.test_dataset = EMNIST(
            data_dir=data_dir,
            transforms=transforms,
            train=False,
            download=True
        )

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def get_test_dataloader(self) -> DataLoader:
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
