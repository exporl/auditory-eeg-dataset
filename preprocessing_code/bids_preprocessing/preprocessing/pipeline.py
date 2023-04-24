import logging
import os
import time
from enum import Enum
from typing import Sequence, Dict, Any, Optional, Union

from bids_preprocessing.preprocessing.base import PreprocessingStep
from bids_preprocessing.preprocessing.save.base import BaseSave
from bids_preprocessing.utils.list import flatten


def default_error_handler_fn(error, data_dict):
    """Default error handler for a Preprocessing pipeline.

    This handler will log the error and traceback.

    Parameters
    ----------
    error: Exception
        Exception that occurred during running of the pipeline
    data_dict: Dict[str, Any]
        Dictionary containing the data
    """

    error_string = f"Error encountered for: {data_dict}\n"
    logging.error(error_string)
    logging.error(error, exc_info=True)


class PreprocessingPipeline(PreprocessingStep):
    """A pipeline (sequence) of PreprocessingSteps."""

    CONTINUE = "continue"
    STOP = "stop"
    RAISE = "raise"

    ON_ERROR = [CONTINUE, STOP, RAISE]

    def __init__(
            self,
            steps: Sequence[PreprocessingStep],
            previous_steps_key: str = "previous_steps",
            error_handler_fn=default_error_handler_fn,
            on_error: str = STOP,
            *args,
            **kwargs
    ):
        """Create a preprocessing pipeline.

        Parameters
        ----------
        steps: Sequence[PreprocessingStep]
            A sequence of preprocessing steps to be applied in a
            specific order.
        previous_steps_key: str
            Key to use to store the metadata of the preprocessing in the
            data_dicts.
        error_handler_fn: Callable[[Exception, Dict[str,Any]], Any]
            Function that handles exceptions when they occur
        """
        super().__init__(*args, **kwargs)
        self.steps = steps
        self.previous_steps_key = previous_steps_key
        self.error_handler_fn = error_handler_fn
        self._on_error = None
        self.on_error = on_error

    @property
    def on_error(self):
        return self._on_error

    @on_error.setter
    def on_error(self, value):
        if value not in self.ON_ERROR:
            raise ValueError(
                f"Invalid value for on_error: {value}. "
                f"Valid values are {self.ON_ERROR}."
            )
        self._on_error = value

    def run_step(
            self,
            step: PreprocessingStep,
            data_dict: Dict[str, Any],
            step_index: Optional[int] = None,
    ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
        """Run a single preprocessing step.

        Parameters
        ----------
        step: PreprocessingStep
            The preprocessing step to be run.
        data_dict: Dict[str, Any]
            The data dictionary to be preprocessed.
        step_index: Optional[int]
            The index of the step in the pipeline.

        Returns
        -------
        Union[Dict[str, Any], Sequence[Dict[str, Any]]]
            A data dictionary or a sequence of data dictionaries.
        """
        step_name = step.__class__.__name__
        step_index_string = f"{step_index}. " if step_index is not None else ""
        logging.info(f"{step_index_string}Running {step_name}...")
        logging.debug(f"{step_name}[Input]: {data_dict}")

        # Add previous steps to data_dict
        start_time = time.time()
        try:
            new_data_dict = step(data_dict)
        except Exception as error:
            if self.on_error == self.CONTINUE:
                new_data_dict = data_dict
            else:
                raise error

        logging.debug(f"{step_name}[Output]: {new_data_dict}")
        logging.info(
            f"{step_index_string}Finished {step_name} " f"in {time.time() - start_time:.2f} seconds."
        )
        self._add_step_information_to_data_dict(data_dict, step, step_index)
        return new_data_dict

    def _add_step_information_to_data_dict(
            self,
            data_dict: Dict[str, Any],
            step: PreprocessingStep,
            step_index: Optional[int],
    ) -> Dict[str, Any]:
        """Add information about the Preprocessing step that was just run.

        The information will be added to the data_dict using the
        self.previous_steps_key.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            Dictionary containing the data
        step: PreprocessingStep
            The PreprocessingStep that will be run on the data_dict
        step_index: Optional[int]
            Optional: the number of the step in the pipeline.

        Returns
        -------
        Dict[str,Any]
            The data_dict with updated information about the PreprocessingSteps
            that were applied
        """
        step_information = dict(vars(step))
        step_information["step_index"] = step_index
        step_information["step_name"] = step.__class__.__name__
        if self.previous_steps_key not in data_dict:
            data_dict[self.previous_steps_key] = []
        data_dict[self.previous_steps_key] += [step_information]
        return data_dict

    def iterate_over_steps(
            self,
            steps: Sequence[PreprocessingStep],
            data_dict: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    ) -> Sequence[Dict[str, Any]]:
        """Iterate over a sequence of preprocessing steps.

        Parameters
        ----------
        steps: Sequence[PreprocessingStep]
            A sequence of preprocessing steps to be applied in a
        data_dict: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
            A data dictionary or a sequence of data dictionaries.

        Returns
        -------
        Sequence[Dict[str, Any]]
            A sequence of data dictionaries.
        """

        if isinstance(data_dict, Sequence):
            current_data_dicts = data_dict
        else:
            current_data_dicts = [data_dict]

        # Check if the last step is a save step
        reloaded_data_dicts = []
        if isinstance(steps[-1], BaseSave):
            new_current_data_dicts = []
            for data_dict in current_data_dicts:
                if steps[-1].is_already_done(data_dict):
                    logging.info("Already found output")
                    if steps[-1].is_reloadable(data_dict):
                        logging.info("Reloading output")
                        reloaded_data_dicts += [steps[-1].reload(data_dict)]
                    else:
                        logging.info("Output is not reloadable, skipping...")
                else:
                    new_current_data_dicts += [data_dict]
            current_data_dicts = new_current_data_dicts

        for step_index, step in enumerate(steps):
            new_data_dicts = []
            for data_dict in current_data_dicts:
                new_data_dicts += [self.run_step(step, data_dict, step_index)]
            current_data_dicts = flatten(new_data_dicts)

        return current_data_dicts + reloaded_data_dicts

    def __call__(
            self, data_dict: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
    ) -> Sequence[Dict[str, Any]]:
        """Apply a preprocessing pipeline to a data dictionary.

        Parameters
        ----------
        data_dict: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
            A data dictionary or a sequence of data dictionaries.

        Returns
        -------
        Sequence[Dict[str, Any]]
            A sequence of data dictionaries.
        """
        logging.info(
            f"Starting {self.__class__.__name__} " f"({len(self.steps)} steps)..."
        )
        start_time = time.time()
        try:
            processed_data_dict = self.iterate_over_steps(self.steps, data_dict)
        except Exception as error:
            if self.on_error == self.RAISE:
                raise error
            else:
                # self.STOP
                self.error_handler_fn(error, data_dict)
                processed_data_dict = data_dict

        logging.info(
            f"Finished {self.__class__.__name__} "
            f"in {time.time() - start_time:.2f} seconds."
        )
        return processed_data_dict
