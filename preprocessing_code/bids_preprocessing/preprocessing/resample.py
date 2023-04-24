"""Resample data to a target frequency."""
from typing import Sequence, Union, Any, Dict, OrderedDict

from scipy import signal

from bids_preprocessing.preprocessing.base import PreprocessingStep


class ResamplePoly(PreprocessingStep):
    """Resample data to a target frequency with polyphase filtering.

    Notes
    -----
    Uses scipy.signal.resample_poly.
    """

    def __init__(
        self,
        target_frequency: Union[int, float],
        data_key: Union[str, Sequence[str]] = "data",
        sampling_frequency_key: Union[str, Sequence[str]] = "data_fs",
        copy_data_dict: bool = False,
        *resample_args: Any,
        **resample_kwargs: Any,
    ):
        """Create a new ResamplePoly instance.

        Parameters
        ----------
        data_key: Union[str, Sequence[str]]
            Keys in the data dict containing the data to resample.
        sampling_frequency_key: Union[str, Sequence[str]]
            Keys in the data dict containing the sampling frequency of the data.
        target_frequency
        resample_args
        resample_kwargs
        """
        super(ResamplePoly, self).__init__(copy_data_dict=copy_data_dict)
        self.data_keys = self.parse_dict_keys(
            data_key, "data_key", require_ordered_dict=True
        )
        self.sampling_frequency_keys = self.parse_dict_keys(
            sampling_frequency_key, "sampling_frequency_key", require_ordered_dict=True
        )
        self.target_frequency = target_frequency
        self.resample_args = resample_args
        self.resample_kwargs = resample_kwargs

        if len(self.data_keys) != len(self.sampling_frequency_keys):
            raise ValueError(
                "The number of data keys and sampling frequency keys must be equal."
            )

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Resample the data to the target frequency.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to resample.

        Returns
        -------
        Dict[str, Any]
            The data dict with the resampled data.
        """

        for (from_key, to_key), (from_fs_key, to_fs_key) in zip(
            self.data_keys.items(), self.sampling_frequency_keys.items()
        ):

            data_dict[to_key] = signal.resample_poly(
                data_dict[from_key],
                self.target_frequency,
                data_dict[from_fs_key],
                *self.resample_args,
                **self.resample_kwargs,
            )
            data_dict[to_fs_key] = self.target_frequency

        return data_dict
