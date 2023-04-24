import gzip
import logging
from typing import Any, Dict

import librosa
import numpy as np

from bids_preprocessing.preprocessing.base import PreprocessingStep


def default_librosa_load_fn(path):
    data, sr = librosa.load(path, sr=None)
    return {"data": data, "sr": sr}


def default_npz_load_fn(path):
    np_data = np.load(path)
    return {
        "data": np_data["audio"],
        "sr": np_data["fs"],
    }


DEFAULT_LOAD_FNS = {
    ".wav": default_librosa_load_fn,
    ".npz": default_npz_load_fn,
}


class LoadStimuli(PreprocessingStep):
    def __init__(
        self,
        load_from={"stimulus_path": "stimulus", "trigger_path": "trigger"},
        load_fn=DEFAULT_LOAD_FNS,
        separator="_",
        **kwargs,
    ):
        super(LoadStimuli, self).__init__(**kwargs)
        self.load_from = load_from
        self.load_fn = load_fn
        self.separator = separator

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        for key, new_key in self.load_from.items():
            if key not in data_dict:
                raise ValueError(
                    f"Can't find {key} in the data dictionary. Available "
                    f"dictionary keys are {str(list(data_dict.keys()))}."
                )
            if data_dict[key] is None:
                logging.warning(f"'{key}' was None, skipping loading {data_dict}.")
                continue

            if isinstance(self.load_fn, dict):
                extension = "." + ".".join(data_dict[key].split(".")[1:])
                if extension not in self.load_fn:
                    raise ValueError(
                        f"Can't find a load function for extension {extension}. "
                        f"Available extensions are {str(list(self.load_fn.keys()))}."
                    )
                load_fn = self.load_fn[extension]
            else:
                load_fn = self.load_fn

            loaded_data = load_fn(data_dict[key])
            for loaded_key, loaded_value in loaded_data.items():
                data_dict[new_key + self.separator + loaded_key] = loaded_value
        return data_dict
