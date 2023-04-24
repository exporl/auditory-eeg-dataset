"""Code to handle epoching of brain response data."""
import copy

from typing import Dict, Any, Sequence, OrderedDict, Union
from bids_preprocessing.preprocessing.base import PreprocessingStep


class SplitEpochs(PreprocessingStep):
    """Split epochs in individual data dicts."""

    def __init__(
        self,
        data_key: Union[str, Sequence[str], OrderedDict[str, str]] = "data",
        stimulus_key: Union[str, Sequence[str], OrderedDict[str, str]] = "stimuli",
        keys_to_index: Sequence[str] = ("event_info", ),
        *args,
        **kwargs
    ):
        """Create a new SplitEpochs instance.

        Parameters
        ----------
        data_key: str
            The key of the EEG data in the data dict.
        stimulus_key: str
            The key of the stimulus data in the data dict.

        Raises
        ------
        TypeError
            If data_key or stimulus_key is not an ordered dict, string or sequence of strings.
        """
        super().__init__(*args, **kwargs)
        self.data_key = self.parse_dict_keys(
            data_key, "data_key", require_ordered_dict=True
        )
        self.stimulus_key = self.parse_dict_keys(
            stimulus_key, "stimulus_key", require_ordered_dict=True
        )
        self.keys_to_index = keys_to_index

    def data_dict_copy(
        self, data_dict: Dict[str, Any], to_exclude: Sequence[str]
    ) -> Dict[str, Any]:
        """Copy a data dict.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict to copy.

        Returns
        -------
        Dict[str, Any]
            The copied data dict.
        """
        new_data_dict = {}
        for key, value in data_dict.items():
            if key not in to_exclude:
                new_data_dict[key] = copy.deepcopy(value)
        return new_data_dict

    def __call__(self, data_dict: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        """Split epochs in individual data dicts.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the EEG data and the stimulus data.
            This dict must contain the keys self.eeg_data_key and
            self.stimulus_key.

        Returns
        -------
        Sequence[Dict[str, Any]]
            A sequence of data dicts, each containing the EEG data and the
            stimulus data for a single epoch.
        """
        data_dict = super(SplitEpochs, self).__call__(data_dict)
        new_data_dicts = []
        for (from_data_key, to_data_key), (from_stimulus_key, to_stimulus_key) in zip(
            self.data_key.items(), self.stimulus_key.items()
        ):
            for index, (epoch, stimulus_dict) in enumerate(zip(
                data_dict[from_data_key], data_dict[from_stimulus_key]
            )):
                new_data_dict = self.data_dict_copy(
                    data_dict,
                    list(self.data_key.keys()) + list(self.stimulus_key.keys()),
                )
                new_data_dict[to_data_key] = epoch
                for key, value in stimulus_dict.items():
                    new_data_dict[key] = value

                for key in self.keys_to_index:
                    if key in new_data_dict and new_data_dict[key] is not None:
                        new_data_dict[key] = new_data_dict[key][index]

                new_data_dicts += [new_data_dict]
        if len(new_data_dicts) == 0:
            raise ValueError(
                "No epochs found. Make sure that the data dict contains the "
                "keys self.eeg_data_key and self.stimulus_key."
            )
        return new_data_dicts
