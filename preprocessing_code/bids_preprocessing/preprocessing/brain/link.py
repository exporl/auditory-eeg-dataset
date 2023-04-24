"""Link stimulus to brain response data."""
import csv
import os
import pathlib
from typing import Sequence, Dict, Any, Optional, Union, Callable

from bids_preprocessing.preprocessing.base import PreprocessingStep
from bids_preprocessing.preprocessing.cache.base import PipelineCache
from bids_preprocessing.utils.list import flatten
from bids_preprocessing.utils.multiprocessing import MultiprocessingSingleton
from bids_preprocessing.utils.path import BIDSStimulusGrouper


class BIDSStimulusInfoExtractor:
    """Extract BIDS compliant stimulus information from an events.tsv file."""

    def __init__(
        self, brain_path_key: str = "data_path", event_info_key: str = "event_info"
    ):
        """Create a new BIDSStimulusInfoExtractor instance.

        Parameters
        ----------
        brain_path_key: str
            The key of the brain data path in the data dict.
        event_info_key: str
            The key store the event information in the data dict.
        """
        self.brain_path_key = brain_path_key
        self.event_info_key = event_info_key

    def __call__(self, brain_dict: Dict[str, Any]):
        """Extract BIDS compliant stimulus information from an events.tsv file.

        Parameters
        ----------
        brain_dict: Dict[str, Any]
            The data dict containing the brain data path.

        Returns
        -------
        Sequence[Dict[str, Any]]
            The extracted event information. Each dict contains the information
            of one row in the events.tsv file
        """
        path = brain_dict[self.brain_path_key]
        # Find BIDS compliant events
        events_path = "_".join(path.split("_")[:-1]) + "_events.tsv"
        # Read events
        event_info = self.read_events(events_path)
        brain_dict[self.event_info_key] = event_info
        return event_info

    def read_events(self, events_path: str):
        """Read events from a BIDS compliant events.tsv file.

        Parameters
        ----------
        events_path: str
            The path to the events.tsv file.

        Returns
        -------
        Sequence[Dict[str, Any]]
            The extracted event information. Each dict contains the information
            of one row in the events.tsv file
        """
        with open(events_path) as fp:
            reader = csv.DictReader(fp, dialect="excel-tab")
            event_info = []
            for row in reader:
                event_info += [row]
        return event_info

class BasenameComparisonFn:
    def __init__(self, stim_path_key: str = "stimulus_path", ignore_extension: bool = False):
        self.stim_path_key = stim_path_key
        self.ignore_extension = ignore_extension

    def process_path(self, path: Any) -> Optional[str]:
        if not isinstance(path, Union[str, pathlib.Path]):
            return None
        if self.ignore_extension:
            return ".".join(os.path.basename(path).split(".")[:-1])
        return path

    def __call__(self, extracted_stim_info, stim_dict):
        all_values = flatten([list(x.values()) for x in extracted_stim_info])
        all_processed_keys = [self.process_path(value) for value in all_values]
        all_processed_keys = [x for x in all_processed_keys if x is not None]
        stimulus_name = self.process_path(stim_dict[self.stim_path_key])
        return stimulus_name in all_processed_keys

def default_multiprocessing_key_fn(data_dict):
    return str(data_dict["stimulus_path"])

class LinkStimulusToBrainResponse(PreprocessingStep):
    """Link stimulus to EEG data."""

    multiprocessing_dict = MultiprocessingSingleton.manager.dict()
    multiprocessing_condition = MultiprocessingSingleton.manager.Event()

    def __init__(
        self,
        stimulus_data: Union[Sequence[Dict[str, Any]], PreprocessingStep],
        extract_stimuli_information_fn=BIDSStimulusInfoExtractor(),
        comparison_fn=BasenameComparisonFn(),
        stimulus_path_key="stimulus_path",
        stimuli_key="stimuli",
        grouper: Optional[BIDSStimulusGrouper] = None,
        key_fn_for_multiprocessing=default_multiprocessing_key_fn,
        *args,
        **kwargs
    ):
        """Create a new LinkStimulusToEEG instance.

        Parameters
        ----------
        stimulus_data_dicts: Sequence[Dict[str, Any]]
            A sequence of data dicts containing the stimulus data.
        find_stimulus_fn: Callable[[str], Sequence[str]]
            A function that takes the path to the EEG recording and returns
            a sequence of corresponding stimulus paths.
        comparison_fn: Callable[[Dict[str, Any], str], bool]
            A function that takes a data dict and a stimulus path and returns
            True if the data dict corresponds to the stimulus path.
        stimulus_path_key: str
            The key in the data dict that contains the path to the stimulus.
        stimuli_key: str
            The key in the data dict that contains the stimulus data.
        pipeline_cache: Optional[PipelineCache]
            The pipeline cache to use to load the stimulus data dict if
            necessary. This is only necessary if the stimulus data dicts
            are cached.
        """
        super(LinkStimulusToBrainResponse, self).__init__(*args, **kwargs)
        self.stimulus_data = stimulus_data
        self.extract_stimuli_information_fn = extract_stimuli_information_fn
        self.comparison_fn = comparison_fn
        self.stimulus_path_key = stimulus_path_key
        self.stimuli_key = stimuli_key
        if grouper is None and isinstance(stimulus_data, Callable):
            raise ValueError(
                "`grouper` must be set if stimulus data is a "
                "preprocessing step/callable."
            )
        self.grouper = grouper
        self.key_fn_for_multiprocessing = key_fn_for_multiprocessing

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Link the stimulus to the EEG data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the EEG data.

        Returns
        -------
        Dict[str, Any]
            The data dict with the corresponding stimulus data added.
        """
        data_dict = super(LinkStimulusToBrainResponse, self).__call__(data_dict)
        # Find the corresponding stimuli data_dicts
        stimulus_info_from_brain = self.extract_stimuli_information_fn(data_dict)
        all_stimuli = []
        if isinstance(self.stimulus_data, Sequence):
            for stimulus_dict in self.stimulus_data:
                if self.comparison_fn(stimulus_info_from_brain, stimulus_dict):
                    all_stimuli += [stimulus_dict]
        else:
            for stim_info in stimulus_info_from_brain:
                prototype_stim_dict = self.grouper(stim_info)
                key = self.key_fn_for_multiprocessing(prototype_stim_dict)
                # Check if no other processes are already running this
                while key in self.multiprocessing_dict:
                    # Wait for the process to finish
                    self.multiprocessing_condition.wait()
                self.multiprocessing_dict[key] = True
                self.multiprocessing_condition.clear()
                try:
                    stimulus_dicts = self.stimulus_data(prototype_stim_dict)
                finally:
                    # Remove the key from the multiprocessing dict to signal that this
                    # specific stimulus is processed
                    del self.multiprocessing_dict[key]
                    # Notify all waiting processes of that this is done
                    self.multiprocessing_condition.set()
                if isinstance(stimulus_dicts, dict):
                    stimulus_dicts = [stimulus_dicts]
                all_stimuli += stimulus_dicts

        data_dict[self.stimuli_key] = all_stimuli
        return data_dict
