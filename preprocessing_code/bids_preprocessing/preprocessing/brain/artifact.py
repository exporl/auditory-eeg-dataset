"""Artifact removal techniques for brain response data.

All the classes assume the MNE layout of the data: (channels, time).
"""
import numpy as np
import scipy
from typing import Union

from bids_preprocessing.preprocessing.base import PreprocessingStep


class InterpolateArtifacts(PreprocessingStep):
    """Blanking of large spikes in EEG."""

    def __init__(
        self,
        data_key: str = "data",
        threshold: Union[int, float] = 500,
        *args,
        **kwargs,
    ):
        """Create a BlankEEGArtifacts object.

        Parameters
        ----------

        data_key: str
            Key of the data_dict to access to obtain the eeg_data.
        threshold: Union[int,float]
            Threshold to determine whether a spike should be removed.
            Default is 500 microVolts, seems to work well.
        """
        super().__init__(*args, **kwargs)
        self.data_key = self.parse_dict_keys(data_key, "data_key")
        self.threshold = threshold

    def __call__(self, data_dict):
        """Interpolate artifacts in the EEG data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the EEG data. Should contain the key
            self.data_key, and the data should be structured as (channels, time).

        Returns
        -------
        Dict[str, Any]
            The data dict with the interpolated artifacts.
        """
        data_dict = super(InterpolateArtifacts, self).__call__(data_dict)
        for from_key, to_key in self.data_key.items():
            eeg = data_dict[from_key]
            # Iterate over channels
            for channel_index in range(eeg.shape[0]):
                artifact_indices = np.abs(eeg[channel_index, :]) > self.threshold
                concat = np.concatenate(([0], artifact_indices, [0]), axis=0)
                # Find rising and falling edges
                diff = np.diff(concat)
                rising_edges = diff[:-1]
                falling_edges = diff[1:]
                indices = np.arange(eeg.shape[1])
                start_indices = indices[rising_edges == 1]
                stop_indices = indices[falling_edges == -1]
                for start_index, stop_index in zip(start_indices, stop_indices):
                    start_sample = eeg[channel_index, start_index]
                    stop_sample = eeg[channel_index, stop_index]
                    # Linearly interpolate between to beginning/fall of the spike
                    eeg[channel_index, start_index + 1 : stop_index] = np.linspace(
                        start_sample, stop_sample, max(stop_index - start_index - 1, 0)
                    )
            data_dict[to_key] = eeg
        return data_dict


class ArtifactRemovalMWF(PreprocessingStep):
    """Remove (eyeblink) artifacts with an MWF.

    This code was based on the excellent library of Somers et al.

    References
    ----------

    """

    def __init__(
        self,
        data_key="data",
        fs_key="data_fs",
        reference_channels=(0, 1, 2, 32, 33, 34, 35, 36),
        delay=3,
        *args,
        **kwargs,
    ):
        """Create an ArtifactRemovalMWF object.

        Parameters
        ----------
        data_key: str
            Key in the data_dict for the brain data
        fs_key: str
            Key in the data_dict for the brain data sampling frequency.
        reference_channels: Sequence[int]
            List of channel indices to use as reference channels. By default, the
            frontal channels of a Biosemi64 channel system are used.
        delay: int
            Delay that has to be taken into account.
        """
        super().__init__(*args, **kwargs)
        self.data_key = self.parse_dict_keys(
            data_key, "data_key", require_ordered_dict=True
        )
        self.fs_key = self.parse_dict_keys(fs_key, "fs_key", require_ordered_dict=True)
        self.axis = 1
        self.reference_channels = reference_channels
        self.delay = delay

    def get_artifact_segments(self, data, fs):
        """Create a mask to select segments of data where artifacts are.

        Parameters
        ----------
        data: np.ndarray
            Brain response data with shape of (time, channels)
        fs: int
            Sampling frequency of the data

        Returns
        -------
        np.ndarray
            Mask of shape (time,) with True for artifact segments.
        """
        ref = np.sum(data[:, self.reference_channels] ** 2, axis=self.axis)
        threshold = 5 * np.mean(ref)
        mask = ref > threshold
        indices = np.where(mask)[0]
        window_len = int(np.round(fs / 2))
        n_frames = data.shape[0]
        for i in range(len(indices)):
            if indices[i] < window_len:
                mask[: indices[i] + window_len + 1] = True
            elif n_frames - indices[i] < window_len:
                mask[indices[i] - window_len :] = True
            else:
                mask[indices[i] - window_len : indices[i] + window_len + 1] = True
        return mask

    def stack_delayed(self, data, delay):
        """Stack delayed versions of the data.

        Parameters
        ----------
        data: np.ndarray
            Brain response data with shape of (channels, time)
        delay

        Returns
        -------
        Tuple[np.ndarray, int]
            Stacked data with shape of (channels * (2 * delay + 1), time), and the
            number of channels in the stacked data.
        """
        nb_channels = data.shape[0]
        nb_shifted_channels = (2 * delay + 1) * nb_channels
        data_s = np.zeros((nb_shifted_channels, data.shape[1]))
        for tau in range(-delay, delay + 1):
            start_ = (tau + delay) * nb_channels
            end_ = (tau + delay + 1) * nb_channels
            shifted = np.roll(data, tau, axis=self.axis)
            if tau > 0:
                shifted[:, :tau] = 0
            elif tau < 0:
                shifted[:, tau:] = 0
            data_s[start_:end_, :] = shifted
        return data_s, nb_shifted_channels

    def check_symmetric(self, data, rtol=1e-05, atol=1e-08):
        """Check whether a matrix is symmetric.

        Parameters
        ----------
        data: np.ndarray
            Matrix to check
        rtol: float
            Relative tolerance
        atol: float
            Absolute tolerance

        Returns
        -------
        bool
            True if the matrix is symmetric, False otherwise.
        """
        return np.allclose(data, data.T, rtol=rtol, atol=atol)

    def fix_symmetric(self, data):
        """Fix a matrix to be symmetric.

        Parameters
        ----------
        data: np.ndarray
            Matrix to fix

        Returns
        -------
        np.ndarray
            Symmetric matrix.
        """
        if self.check_symmetric(data):
            return data
        else:
            return (data.T + data) / 2

    def sort_evd(self, eig_values, eig_vectors):
        """Sort the eigenvalues and eigenvectors.

        Parameters
        ----------
        eig_values: np.ndarray
            Eigenvalues
        eig_vectors: np.ndarray
            Column eigenvectors

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Sorted eigenvalues and eigenvectors.
        """
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
        return eig_values, eig_vectors

    def compute_mwf(self, data, mask):
        """Compute the MWF.

        Parameters
        ----------
        data: np.ndarray
            Brain response data with shape of (channels, time)
        mask: np.ndarray
            Mask of shape (time,) with True for artifact segments.

        Returns
        -------
        np.ndarray
            The MWF with dimension (channels * (2 * delay + 1), channels)
        """
        data, nb_shifted_channels = self.stack_delayed(data, self.delay)
        ryy = self.fix_symmetric(np.cov(data[:, mask]))
        rnn = self.fix_symmetric(np.cov(data[:, mask == False]))
        eig_values, eig_vectors = scipy.linalg.eig(ryy, rnn)
        eig_values, eig_vectors = self.sort_evd(eig_values, eig_vectors)

        temp_eig_values = np.diag(eig_vectors.T @ ryy @ eig_vectors)
        # Denormalize eigenvectors. The original matlab code relied on
        # un-normalized eigenvectors while scipy only uses normalized
        # eigenvectors
        unnormalization_factors = np.repeat(
            np.sqrt(temp_eig_values / eig_values)[np.newaxis],
            eig_vectors.shape[0],
            axis=0,
        )
        unnorm_eig_vectors = eig_vectors / unnormalization_factors
        eig_values_y = unnorm_eig_vectors.T @ ryy @ unnorm_eig_vectors
        eig_values_n = unnorm_eig_vectors.T @ rnn @ unnorm_eig_vectors
        delta = eig_values_y - eig_values_n
        rank_w = nb_shifted_channels - np.sum(np.diagonal(delta) < 0)
        eig_values_to_truncate = range(rank_w, delta.shape[1])
        indices = (eig_values_to_truncate, eig_values_to_truncate)
        delta[indices] = 0
        eig_values_mat = eig_values * np.eye(delta.shape[0])

        left = -np.linalg.solve(eig_values_mat.T, unnorm_eig_vectors.T).T
        right = -np.linalg.solve(unnorm_eig_vectors.T, delta.T).T
        return left @ right

    def apply_mwf(self, data, mwf_weights):
        """Apply the MWF.

        Parameters
        ----------
        data: np.ndarray
            Brain response data with shape of (channels, time)
        mwf_weights: np.ndarray
            MWF weights with shape of (channels * (2 * delay + 1), channels)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered data with shape of (channels, time) and artifacts with
            shape of (channels, time).
        """
        channels, time = data.shape
        nb_weights = mwf_weights.shape[0]
        tau = (nb_weights - channels) // (2 * channels)
        channel_means = data.mean(axis=self.axis, keepdims=True)
        data = data - channel_means
        data_shifted, _ = self.stack_delayed(data, tau)
        orig_chans = range(tau * channels, (tau + 1) * channels)
        artifacts = mwf_weights[:, orig_chans].T @ data_shifted
        filtered_data = data - artifacts
        return filtered_data + channel_means, artifacts

    def __call__(self, data_dict):
        """Perform MWF filtering.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            Dictionary containing the data to filter.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the filtered data.
        """
        for (from_data_key, to_data_key), (from_fs_key, _) in zip(
            self.data_key.items(), self.fs_key.items()
        ):
            data = data_dict[from_data_key].T
            fs = data_dict[from_fs_key]
            mask = self.get_artifact_segments(data, fs)
            mwf_weights = self.compute_mwf(data.T, mask)
            filtered_data, artifacts = self.apply_mwf(data.T, mwf_weights)
            data_dict[to_data_key] = np.real(filtered_data)
        return data_dict
