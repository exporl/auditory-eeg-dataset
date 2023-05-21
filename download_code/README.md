Tools/info to download the dataset
==================================

Code to make downloading of the dataset more convenient.

You can download the full dataset in `.zip` format [from the KU Leuven RDR site by clicking this link.](https://rdr.kuleuven.be/api/access/dataset/:persistentId/?persistentId=doi:10.48804/K3VSND)

If you want more control over what you download and don't want to fully restart when downloading fails, you can use [download_script.py](./download_script.py) to download the dataset.

The code in [download_script.py](./download_script.py)  is meant to be used to download [the auditory EEG dataset from the RDR website](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND), but can (possibly) be used to download other datasets from DataVerse servers.

# Usage

```bash
python download_script.py --help
```

```text
usage: download_script.py [-h] [--server SERVER] [--dataset-id DATASET_ID] [--overwrite] [--skip-checksum] [--multiprocessing MULTIPROCESSING] [--subset {full,preprocessed,stimuli}] download_directory

Download the auditory EEG dataset from RDR.

positional arguments:
  download_directory    Path to download the dataset to.

options:
  -h, --help            show this help message and exit
  --server SERVER       The server to download the dataset from. Default: "rdr.kuleuven.be"
  --dataset-id DATASET_ID
                        The dataset ID to download. Default: "doi:10.48804/K3VSND"
  --overwrite           Overwrite existing files.
  --skip-checksum       Whether to skip checksums.
  --multiprocessing MULTIPROCESSING
                        Number of cores to use for multiprocessing. Default: -1 (all cores), set to 0 or 1 to disable multiprocessing.
  --subset {full,preprocessed,stimuli}
                        Download only a subset of the dataset. "full" downloads the full dataset, "preprocessed" downloads only the preprocessed data and "stimuli" downloads only the stimuli files. Default: full

```
