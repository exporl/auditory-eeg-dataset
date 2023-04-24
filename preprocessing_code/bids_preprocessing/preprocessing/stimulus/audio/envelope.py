"""Code to calculate speech envelopes."""
import logging

import numpy as np

from brian2 import Hz
from brian2hears import Sound, erbspace, Gammatone, Filterbank

from bids_preprocessing.preprocessing.base import PreprocessingStep


class EnvelopeFromGammatone(Filterbank):
    """Converts the output of a GammatoneFilterbank to an envelope."""

    def __init__(self, source, power_factor):
        """Initialize the envelope transformation.

        Parameters
        ----------
        source : Gammatone
            Gammatone filterbank output to convert to envelope
        power_factor : float
            The power factor for each sample.
        """
        super().__init__(source)
        self.power_factor = power_factor
        self.nchannels = 1

    def buffer_apply(self, input_):
        return np.reshape(
            np.sum(np.power(np.abs(input_), self.power_factor), axis=1, keepdims=True),
            (np.shape(input_)[0], self.nchannels),
        )


class GammatoneEnvelope(PreprocessingStep):
    """Calculates a gammatone envelope."""

    def __init__(
        self,
        stimulus_data_key="stimulus_data",
        stimulus_sr_key="stimulus_sr",
        output_key="envelope_data",
        power_factor=0.6,
        min_freq=50,
        max_freq=5000,
        bands=28,
    ):
        """Initialize the gammatone envelope FeatureExtractor.

        Parameters
        ----------
        power_factor : float
            The power factor for each sample
        target_fs : int
            The target sampling frequency
        """
        self.stimulus_data_key = stimulus_data_key
        self.stimulus_sr_key = stimulus_sr_key
        self.output_key = output_key
        self.power_factor = power_factor
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.bands = bands

    def __call__(self, data_dict):
        if self.stimulus_data_key not in data_dict:
            logging.warning(
                f"Found no stimulus data in {data_dict} for step "
                f"{self.__class__.__name__}, skipping..."
            )
            return data_dict
        data = data_dict[self.stimulus_data_key]
        sr = data_dict[self.stimulus_sr_key]

        sound = Sound(data, samplerate=sr * Hz)
        # 28 center frequencies from 50 Hz till 5kHz
        center_frequencies = erbspace(
            self.min_freq * Hz, self.max_freq * Hz, self.bands
        )
        filter_bank = Gammatone(sound, center_frequencies)
        envelope_calculation = EnvelopeFromGammatone(filter_bank, self.power_factor)
        output = envelope_calculation.process()

        data_dict[self.output_key] = output
        return data_dict
