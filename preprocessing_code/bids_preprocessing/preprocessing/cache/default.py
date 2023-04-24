import os
from typing import Sequence, Dict, Any, Optional, Union

from bids_preprocessing.preprocessing.base import PreprocessingStep
from bids_preprocessing.preprocessing.cache.base import (
    PipelineCache,
    pickle_dump_wrapper,
    pickle_load_wrapper,
)


class DefaultPipelineCache(PipelineCache):
    def __init__(
        self,
        cache_root: str,
        cache_folder_overrides: Optional[Dict[PreprocessingStep, str]] = None,
        cache_key="cache",
        previous_cache_folder_key="previous_cache",
        previous_caches_key="previous_caches",
        serializer_fn=pickle_dump_wrapper,
        deserializer_fn=pickle_load_wrapper,
        filename_keys=(
            "eeg_path",
            "stimulus_path",
            ["stimuli", "stimulus_path"],
            "trigger_path",
        ),
        separator="_-_",
    ):
        super().__init__(
            cache_root,
            cache_key,
            previous_cache_folder_key,
            serializer_fn,
            deserializer_fn,
        )
        self.separator = separator
        self.filename_keys = filename_keys
        self.previous_caches_key = previous_caches_key
        if cache_folder_overrides is None:
            self.cache_folder_overrides = {}

    def _add_filename(self, old_filename: str, added_filename: str) -> str:
        if len(old_filename):
            old_filename += self.separator
        return old_filename + os.path.basename(added_filename).split(".")[0]

    def _get_or_predict_filenames_from_data_dict(
        self, data_dict: Dict[str, Any], keys, predict=False
    ) -> Union[str, Sequence[str]]:
        possible_filenames = []
        filename = ""
        for key in keys:
            if isinstance(key, str):
                if key not in data_dict or data_dict[key] is None:
                    continue
                filename = self._add_filename(filename, data_dict[key])
                possible_filenames += [[filename]]
            elif isinstance(key, Sequence) and len(key) == 2:
                if key[0] not in data_dict:
                    continue
                old_filename = filename
                sub_filenames = []
                list_of_dicts = data_dict[key[0]]
                if not isinstance(list_of_dicts, Sequence):
                    continue
                for sub_dict in list_of_dicts:
                    if key[1] not in sub_dict:
                        continue
                    filename = self._add_filename(filename, sub_dict[key[1]])
                    temp_filename = self._add_filename(old_filename, sub_dict[key[1]])
                    sub_filenames += [temp_filename]
                possible_filenames += [sub_filenames]
            elif isinstance(key, Sequence) and len(key) != 2:
                raise NotImplementedError(
                    "If key is a list or tuple, it must have length 2."
                )
            else:
                raise NotImplementedError("Key must be a string or a list or tuple.")
        if filename == "":
            raise ValueError(
                "Can't create a cache for this step as no unique filename to "
                "can be created for it. Try adjusting the `filename_keys` of "
                f"the {self.__class__.__name__}"
            )
        if predict:
            return possible_filenames
        else:
            return filename

    def predict_filenames_from_previous_filename(
        self, previous_filename: str
    ) -> Sequence[Sequence[str]]:
        split_filename = previous_filename.split(self.separator)
        possible_filenames = []
        nb_parts = len(split_filename)
        current_filename = ""

        for index1 in range(nb_parts):
            current_filename = self._add_filename(
                current_filename, split_filename[index1]
            )
            possible_filenames += [[current_filename]]
            sub_filenames = []
            for index2 in range(index1 + 1, nb_parts):
                sub_filenames += [
                    self._add_filename(current_filename, split_filename[index2])
                ]
            if len(sub_filenames):
                possible_filenames += [sub_filenames]
        return possible_filenames

    def predict_filenames_from_data_dict(
        self, data_dict: Dict[str, Any]
    ) -> Sequence[str]:
        return self._get_or_predict_filenames_from_data_dict(
            data_dict, self.filename_keys, predict=True
        )

    def get_filename(
        self, data_dict: Dict[str, Any], keys: Optional[Sequence[str]] = None
    ) -> str:
        if self.cache_key in data_dict:
            keys = [self.cache_key]
        else:
            keys = self.filename_keys
        return self._get_or_predict_filenames_from_data_dict(
            data_dict, keys, predict=False
        )

    def get_foldername(self, step: PreprocessingStep, step_index: Optional[int]) -> str:
        # Get the name from the overrides
        if step in self.cache_folder_overrides:
            return self.cache_folder_overrides[step]

        # Get the name from the step
        name = step.__class__.__name__
        if step_index is not None:
            name = f"{step_index}_{name}"
        return name
