from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    @abstractmethod
    def get_train_dataloader(self):
        ...

    @abstractmethod
    def get_test_dataloader(self):
        ...


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, idx: int):
        ...
