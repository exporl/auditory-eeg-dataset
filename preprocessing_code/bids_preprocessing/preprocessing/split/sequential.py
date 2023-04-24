from typing import Sequence, Union

import numpy as np

from bids_preprocessing.preprocessing.split.base import Splitter


class SequentialSplit(Splitter):
    def split(self, data, shortest_length, split_fraction, start_index):
        end_index = int(np.round(shortest_length * split_fraction))
        split_data = np.take(data, np.arange(start_index, end_index), axis=self.axis)
        return split_data, end_index
