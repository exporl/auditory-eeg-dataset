from typing import Sequence, Union

import numpy as np

from bids_preprocessing.preprocessing.split.base import Splitter


class MidSplit(Splitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sort_split_names_fractions()

    def _sort_split_names_fractions(self):
        split_name_fraction = zip(self.split_names, self.split_fractions)
        new_names = []
        new_fractions = []
        for name, fraction in sorted(split_name_fraction, key=lambda x: x[1]):
            new_names += [name]
            new_fractions += [fraction]
        self.split_names = new_names
        self.split_fractions = new_fractions

    def split(self, data, shortest_length, split_fraction, start_index):
        max_fraction = np.max(self.split_fractions)
        data_length = data.shape[self.axis]
        if split_fraction == max_fraction:
            end_index_1 = int(np.round(shortest_length * split_fraction / 2))
            end_index_2 = data_length - end_index_1
            split_data = np.take(
                data,
                np.concatenate(
                    (
                        np.arange(0, end_index_1),
                        np.arange(data_length - end_index_2, data_length),
                    ),
                    axis=self.axis,
                ),
                axis=self.axis,
            )
            return split_data, end_index_1
        else:
            end_index = start_index + int(np.round(shortest_length * split_fraction))
            split_data = np.take(
                data, np.arange(start_index, end_index), axis=self.axis
            )
            return split_data, end_index
