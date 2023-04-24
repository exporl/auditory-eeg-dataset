import logging
from typing import Sequence, Dict, Any, Union

from bids_preprocessing.preprocessing.base import PreprocessingStep
from bids_preprocessing.preprocessing.cache.base import PipelineCache
from bids_preprocessing.preprocessing.pipeline import PreprocessingPipeline


class CachingPreprocessingPipeline(PreprocessingPipeline):
    def __init__(
        self,
        steps: Sequence[PreprocessingStep],
        pipeline_cache: PipelineCache,
        overwrite=False,
    ):
        super().__init__(steps)
        self.pipeline_cache = pipeline_cache
        self.overwrite = overwrite

    def run_step(
        self, step: PreprocessingStep, data_dict: Dict[str, Any], step_index=None
    ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:

        # Check if the passed data_dict is already cached
        existing_cache = self.pipeline_cache.get_existing_cache_paths(
            step, data_dict, step_index
        )

        # Check if there is already a cached version of the next step
        if len(existing_cache) and not self.overwrite:
            logging.info("Step was already run, skipping to next step")
            new_data_dicts = []
            # Get the cached data_dicts for the next step
            for existing_path in existing_cache:
                new_data_dicts += [
                    self.pipeline_cache.get_cache_dict(existing_path, step, step_index)
                ]
            return new_data_dicts
        elif self.overwrite:
            logging.info("Overwrite is set to True, running step again")

        # Load the data_dict from the previous step
        if self.pipeline_cache.previous_cache_folder_key in data_dict:
            previous_data_dict = self.pipeline_cache.load_from_data_dict(data_dict)
        else:
            previous_data_dict = data_dict

        new_data_dict = super(CachingPreprocessingPipeline, self).run_step(
            step, previous_data_dict
        )

        # Handle the case where the step returns a list of data_dicts
        if isinstance(new_data_dict, dict):
            new_data_dicts = [new_data_dict]
        else:
            new_data_dicts = new_data_dict

        # Save the data_dicts to the cache
        resulting_dicts = []
        for new_data_dict in new_data_dicts:
            save_path = self.pipeline_cache.get_path(step, new_data_dict, step_index)
            self.pipeline_cache.save(save_path, new_data_dict)
            logging.debug(f"Saved data_dict to {save_path}")
            resulting_dicts += [
                self.pipeline_cache.get_cache_dict(save_path, step, step_index)
            ]
        return resulting_dicts
