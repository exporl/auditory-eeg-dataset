"""Rereference the data."""
import numpy as np

from typing import Dict, Any
from bids_preprocessing.preprocessing.base import PreprocessingStep


class CommonAverageRereference(PreprocessingStep):
    """Re-reference multivariate data to the common average of all channels."""

    def __init__(
        self, data_key: str = "data", axis: int = 0, *args, **kwargs: Any
    ) -> None:
        """Create a new CommonAverageRereference instance.

        Parameters
        ----------
        data_key: str
            The key of the EEG data in the data dict.
        """
        super().__init__(*args, **kwargs)
        self.data_keys = self.parse_dict_keys(data_key)
        self.axis = axis

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Rereference the EEG data to the common average of all electrodes.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the EEG data.

        Returns
        -------
        Dict[str, Any]
            A data dict containing the rereferenced EEG data.
        """
        for from_key, to_key in self.data_keys.items():
            data = data_dict[from_key]
            data_dict[to_key] = data - np.mean(data, axis=self.axis, keepdims=True)
        return data_dict
