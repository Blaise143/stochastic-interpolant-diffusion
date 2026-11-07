from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseDataLoader(ABC):
    """Base class for dataset-specific dataloader wrappers."""

    @abstractmethod
    def get_train_dataloader(self) -> DataLoader:
        """Returns the training set DataLoader."""
        ...

    @abstractmethod
    def get_test_dataloader(self) -> DataLoader:
        """Returns the held-out test set DataLoader."""
        ...


class BaseDataset(Dataset, ABC):
    """Base class for datasets used by this project."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """Returns the (sample, label) pair at `idx`."""
        ...
