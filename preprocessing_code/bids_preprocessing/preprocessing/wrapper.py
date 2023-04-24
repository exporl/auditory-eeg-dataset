"""Wrapper for any function that can be used as a preprocessing step."""

from bids_preprocessing.preprocessing.base import PreprocessingStep


class Wrapper(PreprocessingStep):
    """A preprocessing step that calls a function on the data_dict."""

    def __init__(
        self,
        function_,
        key=None,
        *extra_args,
        copy_data_dict: bool = False,
        **extra_kwargs
    ):
        """Create a new LambdaPreprocessingStep instance.

        Parameters
        ----------
        function_: Callable[[Dict[str, Any]], Dict[str, Any]]
            The function to call on the data_dict.
        extra_args: Sequence[Any]
            Extra arguments to pass to the function.
        extra_kwargs: Dict[str, Any]
            Extra keyword arguments to pass to the function.
        """
        super().__init__(copy_data_dict=copy_data_dict)
        self.function = function_
        if key is not None:
            self.keys = self.parse_dict_keys(key)
        else:
            self.keys = key
        self.extra_args = extra_args
        self.extra_kwargs = extra_kwargs

    def __call__(self, data_dict):
        """Apply the function to the data_dict.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict to pass to the function.

        Returns
        -------
        Dict[str, Any]
            The result of the function.
        """
        data_dict = super(Wrapper, self).__call__(data_dict)
        if self.keys is None:
            return self.function(data_dict, *self.extra_args, **self.extra_kwargs)
        for from_key, to_key in self.keys.items():
            data_dict[to_key] = self.function(
                data_dict[from_key], *self.extra_args, **self.extra_kwargs
            )
        return data_dict
