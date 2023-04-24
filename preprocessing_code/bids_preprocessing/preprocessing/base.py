import abc
import copy
from collections import OrderedDict
from typing import Sequence, Any, Dict, Union, Mapping


class PreprocessingStep(abc.ABC):
    """A preprocessing step."""

    def __init__(self, copy_data_dict=False):
        self.copy_data_dict = copy_data_dict

    @abc.abstractmethod
    def __call__(
        self, data_dict: Dict[str, Any]
    ) -> Union[Dict[str, Any], Sequence[Dict[str, Any]]]:
        """Apply a preprocessing step to a data dictionary.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            A dictionary containing the data to be preprocessed.
            This dictionary is modified in-place and returned.


        Returns
        -------
        Union[Dict[str, Any], Sequence[Dict[str, Any]]]
            Preprocessed data in a dictionary format. Can be a single
            dictionary or a sequence of dictionaries.


        Notes
        -----
        In general, it is a good idea to keep the structure of data_dict as
        flat as possible, as this makes it easier to write interoperable
        preprocessing steps.
        """
        if self.copy_data_dict:
            return copy.deepcopy(data_dict)
        else:
            return data_dict

    def parse_dict_keys(
        self,
        key: Union[str, Sequence[str], Mapping[str, str]],
        name="key",
        require_ordered_dict=False,
    ) -> OrderedDict[str, str]:
        """Parse a key or a sequence of keys.

        Parameters
        ----------
        key: Union[str, Sequence[str], Mapping[str,str]]
            A key or a sequence of keys.

        Returns
        -------
        OrderedDict[str, str]
            A mapping of input keys to output keys.
        """
        if isinstance(key, str):
            return OrderedDict([(key, key)])
        elif isinstance(key, Mapping):
            if require_ordered_dict and not isinstance(key, OrderedDict):
                raise TypeError(
                    f"If the '{name}' is a mapping, it must be an ordered mapping "
                    "(e.g. not an ordinary `dict` but an `OrderedDict`)."
                )
            return OrderedDict(key.items())
        elif isinstance(key, Sequence):
            return OrderedDict([(k, k) for k in key])
        else:
            extra_msg = ""
            if require_ordered_dict:
                extra_msg = "n ordered"
            raise TypeError(
                f"The '{name}' must be a string, a sequence of strings or "
                f"a{extra_msg} mapping of strings."
            )
