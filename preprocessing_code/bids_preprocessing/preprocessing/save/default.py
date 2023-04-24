"""Default save class."""
import json
import logging
import os
from typing import Any, Dict

import numpy as np

from bids_preprocessing.preprocessing.cache.base import (
    pickle_dump_wrapper,
    pickle_load_wrapper,
)
from bids_preprocessing.preprocessing.save.base import BaseSave
from bids_preprocessing.utils.multiprocessing import MultiprocessingSingleton


def pickle_dump_wrapper_t(path: str, data: Any):
    """Wrapper for pickle.dump to make it compatible with the save function.

    Parameters
    ----------
    path: str
        The path to save the data to.
    data: Any
        The data to save.
    """
    pickle_dump_wrapper(data, path)

def pickle_dump_wrapper_env_t(path: str, data: Any):
    """Wrapper for pickle.dump to make it compatible with the save function.

    Parameters
    ----------
    path: str
        The path to save the data to.
    data: Any
        The data to save.
    """
    pickle_dump_wrapper(data, path)

    # check if envelope is in data
    if 'envelope_data' in data:
        envelope_path = path.replace('.data_dict', '_envelope.npy')
        np.save(envelope_path, data['envelope_data'])


DEFAULT_SAVE_FUNCTIONS = {
    "npy": np.save,
    "pickle": pickle_dump_wrapper_t,
    "data_dict": pickle_dump_wrapper_env_t,
}


def default_metadata_key_fn(data_dict: Dict[str, Any]) -> str:
    """Default function to generate a key for the metadata.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.

    Returns
    -------
    str
        The key for the metadata.
    """
    if "data_path" in data_dict:
        return os.path.basename(data_dict["data_path"])

    if "stimulus_path" in data_dict and data_dict["stimulus_path"] is not None:
        return os.path.basename(data_dict["stimulus_path"])

    if "trigger_path" in data_dict and data_dict["trigger_path"] is not None:
        return os.path.basename(data_dict["trigger_path"])

    raise ValueError("No data_path or stimulus_path in data_dict.")


def default_filename_fn(data_dict, feature_name, set_name=None, separator="_-_"):
    """Default function to generate a filename for the data.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.
    feature_name: str
        The name of the feature.
    set_name: Optional[str]
        The name of the set. If no set name is given, the set name is not
        included in the filename.
    separator: str
        The separator to use between the different parts of the filename.

    Returns
    -------
    str
        The filename.
    """
    parts = []
    if "data_path" in data_dict:
        parts += [os.path.basename(data_dict["data_path"]).split(".")[0]]

    if "stimulus_path" in data_dict:
        parts += [os.path.basename(data_dict["stimulus_path"]).split(".")[0]]

    if feature_name is None and set_name is None:
        return separator.join(parts) + ".data_dict"

    if 'event_info' in data_dict and 'snr' in data_dict['event_info']:
        parts += [str(data_dict['event_info']['snr'])]

    keys = parts + [feature_name]
    if set_name is not None:
        keys = [set_name] + keys
    return separator.join(keys) + ".npy"

def BIDS_filename_fn(data_dict, feature_name, set_name=None):
    """Default function to generate a filename for the data.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.
    feature_name: str
        The name of the feature.
    set_name: Optional[str]
        The name of the set. If no set name is given, the set name is not
        included in the filename.
    separator: str
        The separator to use between the different parts of the filename.

    Returns
    -------
    str
        The filename.
    """

    filename = os.path.basename(data_dict["data_path"]).split("_eeg")[0]

    subject = filename.split("_")[0]
    session = filename.split("_")[1]
    filename += f"_desc-preproc-audio-{os.path.basename(data_dict['stimulus_path']).split('.')[0]}_{feature_name}"

    if set_name is not None:
        filename += f"_set-{set_name}"
    print(os.path.join(subject, session,filename + ".npy"))
    return os.path.join(subject, session,filename + ".npy")


class Save(BaseSave):
    lock = MultiprocessingSingleton.manager.Lock()

    def __init__(
        self,
        root_dir,
        to_save=None,
        overwrite=False,
        clear_output=False,
        filename_fn=default_filename_fn,
        save_fn=DEFAULT_SAVE_FUNCTIONS,
        metadata_filename=".save_metadata.json",
        metadata_key_fn=default_metadata_key_fn,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.to_save = to_save
        self.overwrite = overwrite
        self.clear_output = clear_output
        self.filename_fn = filename_fn
        self.save_fn = save_fn
        self.metadata_filename = metadata_filename
        self.metadata_key_fn = metadata_key_fn

    def is_already_done(self, data_dict):
        if self.overwrite:
            return False
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        if key not in metadata:
            return False
        is_done = True
        for filename in metadata[key]:
            is_done = is_done and os.path.exists(os.path.join(self.root_dir, filename))
        return is_done

    def _get_metadata(self):
        metadata_path = os.path.join(self.root_dir, self.metadata_filename)
        if not os.path.exists(metadata_path):
            return {}
        self.lock.acquire()
        with open(metadata_path) as fp:
            metadata = json.load(fp)
        self.lock.release()
        return metadata

    def _add_metadata(self, data_dict, filepath):
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        if key not in metadata:
            metadata[key] = []
        if not isinstance(filepath, (list, tuple)):
            filepath = [filepath]
        for path in filepath:
            filename = os.path.basename(path)
            if filename not in metadata[key]:
                metadata[key] += [filename]
        self._write_metadata(metadata)

    def _write_metadata(self, metadata):
        metadata_path = os.path.join(self.root_dir, self.metadata_filename)
        self.lock.acquire()
        with open(metadata_path, "w") as fp:
            json.dump(metadata, fp)
        self.lock.release()

    def _fn_wrapper(self, fn, filepath, *args, **kwargs):
        if not isinstance(fn, dict):
            return fn(filepath, *args, **kwargs)
        suffix = os.path.basename(filepath).split(".")[-1]
        if suffix not in fn:
            raise ValueError(
                f"Can't find an appropriate function to save '{filepath}'."
            )
        return fn[suffix](filepath, *args, **kwargs)

    def _apply_to_data(self, data_dict, fn):
        if self.to_save is None:
            path = os.path.join(self.root_dir, self.filename_fn(data_dict, None, None))
            self._fn_wrapper(fn, path, data_dict)
            self._add_metadata(data_dict, [path])
            return

        paths = []
        for feature_name, feature_loc in self.to_save.items():
            data = data_dict[feature_loc]
            if isinstance(data, dict):
                for set_name, set_data in data.items():
                    filename = self.filename_fn(data_dict, feature_name, set_name)
                    path = os.path.join(self.root_dir, filename)
                    # make sure path folders exist, otherwise create them recusively
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    self._fn_wrapper(fn, path, set_data)
                    paths += [path]
            else:
                filename = self.filename_fn(data_dict, feature_name)
                path = os.path.join(self.root_dir, filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._fn_wrapper(fn, path, data)
                paths += [path]
        self._add_metadata(data_dict, paths)

    def is_reloadable(self, data_dict: Dict[str, Any]) -> bool:
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        if key not in metadata:
            return False
        if len(metadata[key]) != 1:
            return False
        path = os.path.join(self.root_dir, metadata[key][0])
        if path.endswith(".data_dict") and os.path.exists(path):
            return True
        else:
            return False

    def reload(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self._get_metadata()
        key = self.metadata_key_fn(data_dict)
        return pickle_load_wrapper(os.path.join(self.root_dir, metadata[key][0]))

    def __call__(self, data_dict):
        os.makedirs(self.root_dir, exist_ok=True)
        self._apply_to_data(data_dict, self.save_fn)
        # Save some RAM space
        if self.clear_output:
            return None
        return data_dict

