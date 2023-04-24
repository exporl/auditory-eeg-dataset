import gzip
import logging
import shutil
import tempfile
import time

import mne

from bids_preprocessing.preprocessing.base import PreprocessingStep


class MNELoader(PreprocessingStep):
    def __init__(
        self, keys={"data_path": "raw"}, copy_data_dict=False, *mne_args, **mne_kwargs
    ):
        """Load a file using MNE.

        Parameters
        ----------
        key: str
            The key of the path in the data dict.
        mne_args: Sequence[Any]
            Extra arguments to pass to MNE's read_raw.
        mne_kwargs: Dict[str, Any]
            Extra keyword arguments to pass to MNE's read_raw.
        """
        super().__init__(copy_data_dict=copy_data_dict)
        self.keys = self.parse_dict_keys(keys, "keys")
        self.mne_args = mne_args
        self.mne_kwargs = mne_kwargs

    def __call__(self, data_dict):
        for from_key, to_key in self.keys.items():
            path = data_dict[from_key]
            fp_out = None
            # Support for gzipped files.
            if path.endswith(".gz"):
                logging.debug("File was gzipped, unzipping...")
                suffix = path.split(".")[-2]
                # MNE only expects paths, so we need to unzip the file first
                # as a named temporary file.
                fp_out = tempfile.NamedTemporaryFile(suffix="." + suffix)
                with gzip.open(path, "rb") as fp_in:
                    start_time = time.time()
                    shutil.copyfileobj(fp_in, fp_out)
                    logging.debug(
                        f"Unzipping took {start_time - time.time():.2f} seconds"
                    )
                path = fp_out.name
            raw = mne.io.read_raw(path, *self.mne_args, **self.mne_kwargs)
            # Clean up the temporary file.
            if fp_out is not None:
                fp_out.close()
            data_dict[to_key] = raw
        return data_dict
