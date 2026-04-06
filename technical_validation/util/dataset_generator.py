"""Code for the dataset_generator for task1."""
import itertools
import os
import numpy as np
import scipy.signal


class RegressionDataGenerator:
    """Generate data for the regression task."""

    def __init__(
        self,
        files,
        window_length= None,
        high_pass_freq= None,
        low_pass_freq= None,
        return_filenames=False,
    ):
        """Initialize the DataGenerator.

        Parameters
        ----------
        files: Sequence[Union[str, pathlib.Path]]
            Files to load.
        window_length: int
            Length of the decision window.
        """
        self.files = self.group_recordings(files)
        self.return_filenames = return_filenames
        self.high_pass_freq = high_pass_freq
        self.low_pass_freq = low_pass_freq

        if self.high_pass_freq and self.low_pass_freq:
            self.filter_ = scipy.signal.butter(N= 1,
                                          Wn =[self.high_pass_freq, self.low_pass_freq],
                                          btype= "bandpass",
                                          fs=64,
                                          output="sos")
        if self.high_pass_freq and not self.low_pass_freq:
            self.filter_ = scipy.signal.butter(N= 1,
                                          Wn = self.high_pass_freq,
                                          btype= "highpass",
                                          fs=64,
                                          output="sos")
        if not self.high_pass_freq and self.low_pass_freq:
            self.filter_ = scipy.signal.butter(N= 1,
                                          Wn = self.low_pass_freq,
                                          btype= "lowpass",
                                          fs=64,
                                          output="sos")
        if not self.high_pass_freq and not self.low_pass_freq:
            self.filter_ = None


    def group_recordings(self, files):
        """Group recordings and corresponding stimuli.

        Parameters
        ----------
        files : Sequence[Union[str, pathlib.Path]]
            List of filepaths to preprocessed and split EEG and speech features

        Returns
        -------
        list
            Files grouped by the self.group_key_fn and subsequently sorted
            by the self.feature_sort_fn.
        """
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, recording_index):
        """Get data for a certain recording.

        Parameters
        ----------
        recording_index: int
            Index of the recording in this dataset

        Returns
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        data = []
        for feature in self.files[recording_index]:
            data += [np.load(feature).astype(np.float32)]

        data = self.prepare_data(data)
        if self.return_filenames:
            return self.files[recording_index], tuple(data)
        else:
            return tuple(data)


    def __call__(self):
        """Load data for the next recording.

        Yields
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

            if idx == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """Change state at the end of an epoch."""
        np.random.shuffle(self.files)

    def prepare_data(self, data):
       """ If specified, filter the data between highpass and lowpass
       :param data:  list of numpy arrays, eeg and envelope
       :return: filtered data

       """

       if self.filter_ is not None:
           resulting_data = []
           # assuming time is the first dimension and channels the second
           resulting_data.append(scipy.signal.sosfiltfilt(self.filter_, data[0], axis=0))

           for stimulus_feature in data[1:]:
               resulting_data.append(scipy.signal.sosfiltfilt(self.filter_, stimulus_feature, axis=0))

       else:
           resulting_data = data

       return resulting_data


