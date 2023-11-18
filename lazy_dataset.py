from typing import *

from torch.utils import data


class BaseLazyDataset:
    def __init__(
            self, data_args: Optional[Iterable] = None, split: Optional[str] = None,
    ):
        self._data_args, self._split = data_args, split
        from datasets import DatasetDict
        self._data: Optional[DatasetDict] = None

    def init(self):
        if self._data is None:
            self._prepare_data()

    @property
    def data(self):
        if self._data is None:
            self._prepare_data()
        return self._data

    def clean(self):
        self._data = None

    def _prepare_data(self):
        from datasets import load_dataset
        if self._data_args is not None and self._data is None:
            self._data = load_dataset(*self._data_args, split=self._split)


class IterableLazyDataset(BaseLazyDataset, data.IterableDataset):
    pass


class LazyDataset(BaseLazyDataset, data.Dataset):
    def __init__(
            self, data_args: Optional[Iterable] = None, split: Optional[str] = None,
            data_length: Optional[int] = None
    ):
        super().__init__(data_args, split)
        self._data_length = data_length

    def __len__(self):
        return self._data_length
