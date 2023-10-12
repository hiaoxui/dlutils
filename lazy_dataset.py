from typing import *

from torch.utils import data
from transformers import PreTrainedTokenizer, AutoTokenizer


class BaseLazyDataset:
    def __init__(
            self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any] = None,
            data_args: Optional[Iterable] = None, split: Optional[str] = None,
    ):
        self._tokenizer_name, self._tokenizer_kwargs = tokenizer_name, tokenizer_kwargs
        self._data_args, self._split = data_args, split
        from datasets import DatasetDict
        self._data: Optional[DatasetDict] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None

    def init(self):
        if self._tokenizer is None:
            kwargs = dict() if self._tokenizer_kwargs is None else self._tokenizer_kwargs
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name, **kwargs, use_fast=False)
        if self._data is None:
            self._prepare_data()

    def clean(self):
        self._tokenizer = self._data = None

    def _prepare_data(self):
        from datasets import load_dataset
        if self._data_args is not None and self._data is None:
            self._data = load_dataset(*self._data_args, split=self._split)


class IterableLazyDataset(BaseLazyDataset, data.IterableDataset):
    pass


class LazyDataset(BaseLazyDataset, data.Dataset):
    def __init__(
            self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any] = None,
            data_args: Optional[Iterable] = None, split: Optional[str] = None,
            data_length: Optional[int] = None
    ):
        super().__init__(tokenizer_name, tokenizer_kwargs, data_args, split)
        self._data_length = data_length

    def __len__(self):
        return self._data_length

    def clean(self):
        self._tokenizer = self._data = None

