import numpy as np

from bids_preprocessing.preprocessing.split.operations.base import SplitterOperation


class Standardize(SplitterOperation):
    def __init__(self, axis=0):
        self.axis = axis
        self.mean = None
        self.std = None

    def __call__(self, data):
        if self.mean is None:
            self.mean = np.mean(data, axis=self.axis, keepdims=True)
            self.std = np.std(data, axis=self.axis, keepdims=True)
        return (data - self.mean) / self.std

    def reset(self):
        self.mean = None
        self.std = None
