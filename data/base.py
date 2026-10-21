from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseDataLoader(ABC):
    @abstractmethod
    def get_train_dataloader(self) -> DataLoader: ...

    @abstractmethod
    def get_test_dataloader(self) -> DataLoader: ...


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple: ...
