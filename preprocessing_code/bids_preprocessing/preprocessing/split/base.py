import abc
from typing import Optional, Sequence, Union

import numpy as np

from bids_preprocessing.preprocessing.base import PreprocessingStep
from bids_preprocessing.preprocessing.split.operations.base import SplitterOperation


class Splitter(PreprocessingStep, abc.ABC):
    def __init__(
        self,
        feature_mapping,
        split_fractions: Sequence[Union[int, float]],
        split_names: Sequence[str],
        extra_operation: Optional[SplitterOperation] = None,
        axis=0,
    ):
        self.feature_mapping = feature_mapping
        self.split_fractions = self._normalize_split_fraction(split_fractions)
        self.split_names = split_names
        self.extra_operation = extra_operation
        self.axis = axis

    def _normalize_split_fraction(self, split_fractions):
        return [fraction / sum(split_fractions) for fraction in split_fractions]

    @abc.abstractmethod
    def split(self, data, shortest_length, split_fraction, start_index):
        pass

    def __call__(self, data_dict):
        shortest_length = min(
            [data_dict[key].shape[self.axis] for key in self.feature_mapping.keys()]
        )
        for from_key, to_key in self.feature_mapping.items():
            data = np.take(
                data_dict[from_key], np.arange(0, shortest_length), axis=self.axis
            )
            resulting_data = {}

            self.extra_operation.reset()
            start_index = 0
            for split_name, split_fraction in zip(
                self.split_names, self.split_fractions
            ):
                split_data, start_index = self.split(
                    data, shortest_length, split_fraction, start_index
                )
                if self.extra_operation is not None:
                    split_data = self.extra_operation(split_data)
                resulting_data[split_name] = split_data
            del data_dict[from_key]
            data_dict[to_key] = resulting_data

        return data_dict
