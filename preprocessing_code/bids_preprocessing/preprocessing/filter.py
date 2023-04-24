"""Preprocessing steps for filtering."""
import copy
import logging
from typing import Callable

from scipy import signal

from bids_preprocessing.preprocessing.base import PreprocessingStep


class SosFiltFilt(PreprocessingStep):
    """Filter data with a second-order section filter.


    Notes
    -----
    Uses scipy.signal.sosfiltfilt.
    """

    def __init__(
        self,
        filter_,
        data_key="data",
        emulate_matlab=True,
        *sosfiltfilt_args,
        copy_data_dict=False,
        **sosfiltfilt_kwargs
    ):
        super(SosFiltFilt, self).__init__(copy_data_dict=copy_data_dict)
        self.data_keys = self.parse_dict_keys(data_key)
        self.filter_ = filter_
        self.sosfiltfilt_args = sosfiltfilt_args
        self.sosfiltfilt_kwargs = sosfiltfilt_kwargs
        self.emulate_matlab = emulate_matlab

        # If the filter is not callable, we can precalculate the MATLAB
        # arguments if necessary
        if emulate_matlab and not isinstance(filter_, Callable):
            self.sosfiltfilt_args, self.sosfiltfilt_kwargs = self.get_matlab_arguments(
                filter_, self.sosfiltfilt_args, self.sosfiltfilt_kwargs
            )

    def get_matlab_arguments(self, filter_, sosfiltfilt_args, sosfiltfilt_kwargs):

        # The only real difference with matlab is the padding
        padlen = 3 * (
            2 * len(filter_)
            - min((filter_[:, 2] == 0).sum(), (filter_[:, 5] == 0).sum())
        )
        # Force the arguments to be the same as MATLAB's
        sosfiltfilt_kwargs = copy.deepcopy(sosfiltfilt_kwargs)
        sosfiltfilt_kwargs["padlen"] = padlen
        sosfiltfilt_kwargs["padtype"] = "odd"

        logging.warning(
            "Emulate MATLAB mode is on, all (keyword) arguments will be "
            "overwritten to match MATLAB's (sos)filtfilt."
        )
        return sosfiltfilt_args[:1], sosfiltfilt_kwargs

    def __call__(self, data_dict):
        data_dict = super(SosFiltFilt, self).__call__(data_dict)
        filter_ = self.filter_
        # Adjustable filter
        if isinstance(filter_, Callable):
            filter_ = filter_(data_dict)

        # Emulate MATLAB filtering
        sosfiltfilt_args = self.sosfiltfilt_args
        sosfiltfilt_kwargs = self.sosfiltfilt_kwargs
        if self.emulate_matlab and isinstance(self.filter_, Callable):
            sosfiltfilt_args, sosfiltfilt_kwargs = self.get_matlab_arguments(
                filter_, sosfiltfilt_args, sosfiltfilt_kwargs
            )

        for from_key, to_key in self.data_keys.items():
            data_dict[to_key] = signal.sosfiltfilt(
                filter_, data_dict[from_key], *sosfiltfilt_args, **sosfiltfilt_kwargs
            )
        return data_dict
