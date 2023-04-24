"""Utilities for working with BIDS paths."""
import glob
import logging
import os
import pathlib
from typing import Sequence, Union, Dict


class BIDSPathGenerator:
    """Generate BIDS paths for a given root directory."""

    def __init__(self, root_dir):
        """Create a new BIDSPathGenerator.

        Parameters
        ----------
        root_dir: str
            The root directory of the BIDS dataset.
        """
        self.root_dir = root_dir

    def _parse_part(self, part):
        # Select everything when part is None.
        if part is None:
            return ["*"]
        elif isinstance(part, str):
            return [part]
        elif isinstance(part, Sequence):
            return part
        else:
            raise ValueError(f"Invalid part for BIDS path: {part}")



    def select_paths(
        self,
        subjects=None,
        sessions=None,
        tasks=None,
        runs=None,
        extensions="eeg",
        suffix="bdf",
    ):
        """Select BIDS paths for a given set of parameters.

        Parameters
        ----------
        subjects: Optional[Union[str, Sequence[str]]]
            The subjects to select. When None, all subjects are selected.
            When a string, only the subject with the given name is selected.
            When a sequence of strings, all subjects with the given names are
            selected.
        sessions: Optional[Union[str, Sequence[str]]]
            The sessions to select. When None, all sessions are selected.
            When a string, only the session with the given name is selected.
            When a sequence of strings, all sessions with the given names are
            selected.
        tasks: Optional[Union[str, Sequence[str]]]
            The tasks to select. When None, all tasks are selected.
            When a string, only the task with the given name is selected.
            When a sequence of strings, all tasks with the given names are
            selected.
        runs: Optional[Union[str, Sequence[str]]]
            The runs to select. When None, all runs are selected.
            When a string, only the run with the given name is selected.
            When a sequence of strings, all runs with the given names are
            selected.
        extensions: Optional[Union[str, Sequence[str]]]
            The extensions to select. When None, all extensions are selected.
            When a string, only the extension with the given name is selected.
            When a sequence of strings, all extensions with the given names are
            selected.
        suffix: Optional[Union[str, Sequence[str]]]
            The suffixes to select. When None, all suffixes are selected.
            When a string, only the suffix with the given name is selected.
            When a sequence of strings, all suffixes with the given names are
            selected.

        Returns
        -------
        List[str]
            A list of paths that match the given parameters.
        """
        paths = []

        for subject in self._parse_part(subjects):
            for session in self._parse_part(sessions):
                for task in self._parse_part(tasks):
                    for run in self._parse_part(runs):
                        for extension in self._parse_part(extensions):
                            for suffix in self._parse_part(suffix):
                                search_path = os.path.join(
                                    self.root_dir,
                                    f"sub-{subject}",
                                    f"ses-{session}",
                                    f"{extension}",
                                    f"sub-{subject}_ses-{session}_task-{task}_run-{run}_{extension}.{suffix}",
                                )
                                paths += glob.glob(search_path)
                                # session is not required in BIDS
                                if session == "*":
                                    search_path = os.path.join(
                                        self.root_dir,
                                        f"sub-{subject}",
                                        f"{extension}",
                                        f"sub-{subject}_task-{task}_run-{run}_{extension}.{suffix}",
                                    )
                                    paths += glob.glob(search_path)
        return paths


def default_is_trigger_fn(path: Union[str, pathlib.Path]):
    return os.path.basename(path).startswith("t_")


def default_is_noise_fn(path: Union[str, pathlib.Path]):
    return os.path.basename(path).startswith("noise_")


def default_is_video_fn(path: Union[str, pathlib.Path]):
    return os.path.basename(path).startswith("VIDEO")


def default_key_fn(path: Union[str, pathlib.Path]):
    basename = os.path.basename(path)
    for prefix in ("t_", "noise_"):
        if basename.startswith(prefix):
            return prefix.join(basename.split(prefix)[1:])
    return basename


class StimulusGrouper:
    def __init__(
        self,
        key_fn=default_key_fn,
        is_trigger_fn=default_is_trigger_fn,
        is_noise_fn=default_is_noise_fn,
        is_video_fn=default_is_video_fn,
        filter_no_triggers=True,
    ):
        self.key_fn = key_fn
        self.is_trigger_fn = is_trigger_fn
        self.is_noise_fn = is_noise_fn
        self.is_video_fn = is_video_fn
        self.filter_no_triggers = filter_no_triggers

    def _postprocess(self, data_dicts):
        new_data_dicts = []
        for data_dict in data_dicts.values():

            if data_dict["stimulus_path"] is None:
                if data_dict["trigger_path"] is None:
                    raise ValueError(
                        "Found data dict without stimulus and trigger, "
                        f"which should not be possible: {data_dict}"
                    )
                else:
                    logging.warning(
                        f"Found a data_dict with no stimulus: {data_dict}. "
                        f"This is fine if the data was collected in silence. "
                        f"Otherwise, adapt the `key_fn` and/or the `is_*_fn` "
                        f"of the StimulusGrouper."
                    )

            if data_dict["trigger_path"] is None:
                logging.error(
                    f"No trigger path found for {data_dict['stimulus_path']}."
                    f"If a trigger path shoud be present, adapt the `key_fn` "
                    f"and/or the `is_*_fn` of the StimulusGrouper."
                )
                if self.filter_no_triggers:
                    logging.error(
                        f"\tFiltering out stimulus data for "
                        f"{data_dict['stimulus_path']}"
                    )
                    continue
            new_data_dicts += [data_dict]
        return new_data_dicts

    def __call__(self, files: Sequence[Union[str, pathlib.Path]]) -> Sequence[Dict]:
        data_dicts = {}
        for path in files:
            key = self.key_fn(path)
            if key not in data_dicts:
                data_dicts[key] = {
                    "trigger_path": None,
                    "noise_path": None,
                    "video_path": None,
                    "stimulus_path": None,
                }
            if self.is_trigger_fn(path):
                data_dicts[key]["trigger_path"] = path
            elif self.is_noise_fn(path):
                data_dicts[key]["noise_path"] = path
            elif self.is_video_fn(path):
                data_dicts[key]["video_path"] = path
            else:
                data_dicts[key]["stimulus_path"] = path
        logging.info(f"Found {len(data_dicts)} stimulus groups")
        data_dict_list = self._postprocess(data_dicts)
        return data_dict_list


class BIDSStimulusGrouper:
    def __init__(
        self,
        bids_root,
        mapping={"stim_file": "stimulus_path"},
        subfolders=["stimuli"],
        na_values=["n/a"],
    ):
        self.bids_root = bids_root
        self.mapping = mapping
        self.subfolders = subfolders
        self.na_values = na_values

    def __call__(self, events_row):
        stimulus_dict = {}
        for from_key, to_key in self.mapping.items():
            if from_key not in events_row:
                logging.warning(
                    f"Could not find {from_key} in events row, skipping stimulus "
                    f"file {events_row}"
                )
                stimulus_dict[to_key] = None
                continue
            events_item = events_row[from_key]
            if events_item in self.na_values:
                stimulus_dict[to_key] = None
                continue

            subfolders = [folder for folder in self.subfolders if folder not in events_item.split('/')]

            stimulus_dict[to_key] = os.path.join(
                self.bids_root, *subfolders, events_item
            )
        return stimulus_dict







