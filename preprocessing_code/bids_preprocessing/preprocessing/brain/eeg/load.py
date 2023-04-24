"""Load EEG data from a BDF file."""
from typing import Dict, Any, Sequence, Optional, Union, Mapping

import numpy as np

from bids_preprocessing.preprocessing.brain.load import MNELoader


class LoadEEGNumpy(MNELoader):
    """Load EEG data.

    This step uses MNE to load EEG data.
    """

    def __init__(
        self,
        eeg_path_key: Union[str, Sequence[str], Dict[str, str]] = {"data_path": "raw"},
        eeg_data_key: Dict[str, str] = {"raw": "data"},
        eeg_trigger_data_key: Dict[str, str] = {"raw": "trigger_data"},
        info_prefix: Dict[str, str] = {"raw": "eeg_"},
        channels_to_select: Optional[Union[Sequence[str], Sequence[int]]] = None,
        trigger_channel: Union[str, int] = "Status",
        unit_multiplier: Union[float, int] = 1.0,
        additional_mapping: Mapping[str, str] = {"eeg_sfreq": "data_fs"},
        copy_data_dict: bool = False,
        *mne_args,
        **mne_kwargs,
    ):
        """Create a new LoadEEG instance.

        Parameters
        ----------
        eeg_path_key: str
            The key of the EEG path in the data dict.
        eeg_data_key: str
            The key of the EEG data in the data dict.
        eeg_trigger_data_key: str
            The key of the EEG trigger data in the data dict.
        info_prefix: str
            The prefix to add to the keys of the EEG info dict.
        channels_to_select: Optional[Union[Sequence[str], Sequence[int]]]
            The names of the channels to select. If None, all channels are
            selected. If a sequence of integers, the channels at the given
            indices are selected. If a sequence of strings, the channels with
            the given names are selected.
        trigger_channel: Union[str, int]
            The name or index of the trigger channel.
        unit_multiplier: Union[float, int]
            The multiplier to apply to the EEG data. MNE loads the data in
            Volts.
        """
        super().__init__(
            eeg_path_key, *mne_args, copy_data_dict=copy_data_dict, **mne_kwargs
        )
        self.eeg_data_key = self.parse_dict_keys(eeg_data_key, "eeg_data_key")
        self.eeg_trigger_data_key = self.parse_dict_keys(
            eeg_trigger_data_key, "eeg_trigger_data_key"
        )
        self.info_prefix = self.parse_dict_keys(info_prefix, "info_prefix")
        self.channels_to_select = channels_to_select
        self.trigger_channel = trigger_channel
        self.unit_multiplier = unit_multiplier
        self.additional_mapping = additional_mapping
        # Necessary if working with zipped files, and data will be loaded
        # here anyway
        self.mne_kwargs["preload"] = True

    def get_channels(
        self,
        eeg: np.ndarray,
        channel_names: Sequence[str],
        selected_channels: Optional[Union[Sequence[str], Sequence[int]]],
    ):
        """Select channels from EEG data.

        Parameters
        ----------
        eeg: np.ndarray
            All available EEG data. Shape: (n_channels, n_samples)
        channel_names: Sequence[str]
            The names of the channels in the EEG data.
        selected_channels: Optional[Union[Sequence[str], Sequence[int]]]
            The names of the channels to select. If None, all channels are
            selected. If a sequence of integers, the channels at the given
            indices are selected. If a sequence of strings, the channels with
            the given names are selected.

        Returns
        -------
        np.ndarray
            The selected EEG data. Shape: (n_selected_channels, n_samples)

        Raises
        ------
        KeyError:
            If a channel name is not found in the EEG data.
        IndexError:
            If a channel index is out of range.
        """
        channels = []
        if isinstance(selected_channels, Sequence):
            for channel in selected_channels:
                if isinstance(channel, int):
                    channel_index = channel
                else:
                    if channel not in channel_names:
                        raise KeyError(
                            f"Channel {channel} not found in EEG data. "
                            f"Available channels: {channel_names}"
                        )
                    channel_index = channel_names.index(channel)
                channels.append(eeg[channel_index])
            return np.stack(channels, axis=0)
        elif selected_channels is None:
            return eeg

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load EEG data from a BDF file.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the EEG path.

        Returns
        -------
        Dict[str, Any]
            The data dict with the EEG data and the EEG info.
        """
        data_dict = super(LoadEEGNumpy, self).__call__(data_dict)
        for raw_key in self.keys.values():
            raw = data_dict[raw_key]

            # Get the EEG data and the trigger data
            all_eeg_data = raw.get_data()
            eeg = self.get_channels(
                all_eeg_data, raw.info["ch_names"], self.channels_to_select
            )
            trigger = self.get_channels(
                all_eeg_data, raw.info["ch_names"], [self.trigger_channel]
            )

            # Set the correct keys of the data_dict
            data_dict[self.eeg_data_key[raw_key]] = eeg * self.unit_multiplier
            data_dict[self.eeg_trigger_data_key[raw_key]] = trigger
            for info_key in raw.info.keys():
                data_dict[self.info_prefix[raw_key] + info_key] = raw.info[info_key]

            for from_key, to_key in self.additional_mapping.items():
                data_dict[to_key] = data_dict[from_key]

            # Remove the data from the raw object
            del data_dict[raw_key]._data

        return data_dict
