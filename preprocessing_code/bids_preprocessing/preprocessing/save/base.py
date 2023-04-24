"""Base class for all save steps."""
import abc
from typing import Any, Dict

from bids_preprocessing.preprocessing.base import PreprocessingStep


class BaseSave(PreprocessingStep, abc.ABC):
    """Base class for all save steps."""

    @abc.abstractmethod
    def is_already_done(self, data_dict: Dict[str, Any]) -> bool:
        """Check if the step was already done.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        bool
            True if the step was already done, False otherwise.
        """
        pass

    @abc.abstractmethod
    def is_reloadable(self, data_dict: Dict[str, Any]) -> bool:
        """Check if the data can be reloaded.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        bool
            True if the data can be reloaded, False otherwise.
        """
        pass

    @abc.abstractmethod
    def reload(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Reload the data.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the data to save.

        Returns
        -------
        Dict[str, Any]
            The data dict containing the reloaded data.
        """
        pass
