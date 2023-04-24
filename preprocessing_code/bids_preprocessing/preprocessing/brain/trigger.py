"""Utilities for working with trigger data."""
import logging

import numpy as np
import scipy.signal
import scipy.interpolate

from bids_preprocessing.preprocessing.base import PreprocessingStep


def biosemi_trigger_processing_fn(trigger):
    triggers = trigger.flatten().astype(np.int32) & (2**16 - 1)
    values, counts = np.unique(triggers, return_counts=True)
    valid_mask = (0 < values) & (values < 256)
    val_indices = np.argsort(counts[valid_mask])
    most_common = values[valid_mask][val_indices[-1]]
    if triggers[0] != most_common:
        logging.warning("First value of the EEG triggers is on, shouldn't be the case")
    return np.int32(triggers != most_common)




def default_drift_correction(
    brain_data, brain_trigger_indices, brain_fs, stimulus_trigger_indices, stimulus_fs
):
    """Correct the drift between the brain response data and the stimulus.

    When the brain response data and the stimulus data are not recorded on
    the same system (i.e. using the same clock), clock drift may cause the
    brain response data to be misaligned with the stimulus. This function
    tries to correct for this by interpolating the brain response data to
    the same length as the stimulus.

    Parameters
    ----------
    brain_data
    brain_trigger_indices
    brain_fs
    stimulus_trigger_indices
    stimulus_fs

    Returns
    -------

    """
    expected_length = int(
        np.ceil(
            (stimulus_trigger_indices[-1] - stimulus_trigger_indices[0])
            / stimulus_fs
            * brain_fs
        )
    )
    real_length = brain_trigger_indices[-1] - brain_trigger_indices[0]
    # resulting_triggers = np.ceil((stimulus_trigger_indices - stimulus_trigger_indices[0]) / stimulus_fs * eeg_fs) + eeg_trigger_indices[0]
    tmp_eeg = brain_data[:, brain_trigger_indices[0] : brain_trigger_indices[-1]]
    idx_real = np.linspace(0, 1, real_length)
    idx_expected = np.linspace(0, 1, expected_length)
    interpolate_fn = scipy.interpolate.interp1d(idx_real, tmp_eeg, "linear", axis=1)
    new_eeg = interpolate_fn(idx_expected)

    new_start = brain_trigger_indices[0]
    begin_eeg = brain_data[:, :new_start]
    end_eeg = brain_data[:, brain_trigger_indices[-1] + 1 :]

    new_end = int(brain_trigger_indices[-1] + 2 * brain_fs)
    # Make length multiple of samplerate
    new_end = int(np.ceil((new_end - new_start) / brain_fs) * brain_fs + new_start - 1)

    total_eeg = begin_eeg[:, new_start:]
    new_eeg_start = max(new_start - begin_eeg.shape[1], 0)
    new_eeg_end = min(new_end - begin_eeg.shape[1], new_eeg.shape[1])
    total_eeg = np.concatenate(
        (total_eeg, new_eeg[:, new_eeg_start:new_eeg_end]), axis=1
    )
    end_eeg_start = max(new_start - begin_eeg.shape[1] - new_eeg.shape[1], 0)
    end_eeg_end = min(
        new_end - begin_eeg.shape[1] - new_eeg.shape[1], end_eeg.shape[1]
    )
    total_eeg = np.concatenate(
        (total_eeg, end_eeg[:, end_eeg_start:end_eeg_end]), axis=1
    )
    if total_eeg.shape[1] % brain_fs != 0:
        nb_seconds = np.floor(brain_data.shape[1] / brain_fs)
        total_eeg = total_eeg[:, : int(nb_seconds * brain_fs)]
    return total_eeg


class AlignPeriodicBlockTriggers(PreprocessingStep):
    def __init__(
        self,
        brain_trigger_processing_fn=lambda x: x,
        postprocessing_fn=default_drift_correction,
        data_key="data",
        data_trigger_key="trigger_data",
        data_sampling_rate_key="eeg_sfreq",
        stimulus_trigger_data_key="trigger_data",
        stimulus_trigger_sampling_rate_key="trigger_sr",
    ):
        """Create a new MatchTriggersToStimulus instance.

        Parameters
        ----------
        data_key: str
        data_trigger_key: str
        data_sampling_rate_key: str
        stimulus_trigger_data_key: str
        stimulus_trigger_sampling_rate_key: str
        """
        super().__init__()
        self.brain_trigger_processing_fn = brain_trigger_processing_fn
        self.postprocessing_fn = postprocessing_fn
        self.data_key = data_key
        self.eeg_trigger_key = data_trigger_key
        self.eeg_sampling_rate_key = data_sampling_rate_key
        self.stimulus_trigger_data_key = stimulus_trigger_data_key
        self.stimulus_trigger_sampling_rate_key = stimulus_trigger_sampling_rate_key

    def get_trigger_indices(self, triggers: np.ndarray) -> np.ndarray:
        """Get the indices of the triggers.

        Parameters
        ----------
        triggers: np.ndarray
            Raw trigger data. Should be a 1D array of 0s and 1s.

        Returns
        -------
        np.ndarray
            Indices of the triggers.
        """
        all_indices = np.where(triggers > 0.5)[0]
        diff_trigger_indices = all_indices[1:] - all_indices[:-1]

        #estimated_trigger_indices = (diff_trigger_indices > 1)

        #trigger_duration = np.round(len(diff_trigger_indices)/len(estimated_trigger_indices))

        # Keep only the gaps between triggers, not the duration of triggers
        indices_to_keep = diff_trigger_indices > 1
        # Assumption that the EEG doesn't start with a trigger
        # in the first sample (shouldn't be the case)
        return all_indices[np.concatenate(([True], indices_to_keep))]

    def __call__(self, data_dict):
        """Match stimulus triggers to triggers from the brain response data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            Dictionary containing the data to be preprocessed.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the preprocessed data.
        """
        raw_brain_trigger = data_dict[self.eeg_trigger_key]
        brain_fs = data_dict[self.eeg_sampling_rate_key]
        brain_data = data_dict[self.data_key]

        # Process the brain trigger data
        brain_trigger = self.brain_trigger_processing_fn(raw_brain_trigger)
        brain_trigger_indices = self.get_trigger_indices(brain_trigger)
        brain_trigger_indices_left = brain_trigger_indices.copy()

        eeg_epochs = []
        all_stimulation_indices = []
        for stimulus_dict in data_dict["stimuli"]:
            nb_eeg_triggers_left = len(brain_trigger_indices_left)
            stimulus_trigger = stimulus_dict[self.stimulus_trigger_data_key]
            stimulus_fs = stimulus_dict[self.stimulus_trigger_sampling_rate_key]
            stimulus_trigger_indices = self.get_trigger_indices(stimulus_trigger)
            nb_stimulus_triggers = len(stimulus_trigger_indices)


            if nb_eeg_triggers_left < nb_stimulus_triggers:
                raise ValueError(
                    f"Number of triggers does not match "
                    f"(in eeg {nb_eeg_triggers_left} were found, "
                    f"in stimulus {nb_stimulus_triggers})"
                )
            cut_brain_trigger_indices = brain_trigger_indices_left[
                : len(stimulus_trigger_indices)
            ]
            new_eeg = self.postprocessing_fn(
                brain_data,
                cut_brain_trigger_indices,
                brain_fs,
                stimulus_trigger_indices,
                stimulus_fs,
            )

            eeg_epochs += [new_eeg]
            brain_trigger_indices_left = brain_trigger_indices_left[
                len(stimulus_trigger_indices) :
            ]
            all_stimulation_indices += [stimulus_trigger_indices]


        if len(brain_trigger_indices_left) != 0:
            raise ValueError(
                f"Found surplus of EEG triggers ({len(brain_trigger_indices_left)} surplus found)"
            )

        data_dict[self.data_key] = eeg_epochs
        return data_dict
