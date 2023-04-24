from bids_preprocessing.preprocessing.base import PreprocessingStep


class Transpose(PreprocessingStep):
    def __init__(self, keys):
        self.keys = self._clean_keys(keys)

    def _clean_keys(self, keys):
        if isinstance(keys, str):
            return [keys]
        elif isinstance(keys, (list, tuple)):
            return keys
        else:
            raise TypeError("keys must be a string or a list of strings")

    def __call__(self, data_dict):
        for key in self.keys:
            data_dict[key] = data_dict[key].T
        return data_dict
