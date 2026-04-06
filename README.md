Code to preprocess the SparrKULee dataset
=========================================
This is the codebase to preprocess and validate the [SparrKULee](https://doi.org/10.48804/K3VSND) dataset.
This codebase consist of two main parts: 
1) preprocessing code, to preprocess the raw data into an easily usable format 
2) technical validation code, to validate the technical quality of the dataset. 
This code is used to generate the results in the dataset paper and assumes that the preprocessing pipeline has been run

Requirements
------------

Python >= 3.7

# General setup

Steps to get a working setup:

## 1. Clone this repository and install the [requirements.txt](requirements.txt)
```bash
# Clone this repository
git clone https://github.com/exporl/auditory-eeg-dataset

# Go to the root folder
cd auditory-eeg-dataset

# Optional: install a virtual environment
python3 -m venv venv # Optional
source venv/bin/activate # Optional

# Install requirements.txt
python3 -m install requirements.txt
```

## 2. Mount the data using R-clone (recommended)

Install [R-clone](https://rclone.org/) and follow [the instructions to mount the dataset using R-clone](https://homes.esat.kuleuven.be/~spchdata/corpora/auditory_eeg_data/#mounting).


## 2. [Download (parts of the) data (deprecated)](download_code/README.md) 


> [!WARNING]
> This path of working with the SparrKULee dataset is not supported anymore. It might still work, but we recommend follwing the steps above to mount the data instead.

A version of dataset is hosted on the [KU Leuven RDR website](https://doi.org/10.48804/K3VSND) and is accessible through DOI ([https://doi.org/10.48804/K3VSND](https://doi.org/10.48804/K3VSND)).

However, due to the dataset size/structure and the limitations of the UI of the KU Leuven RDR website, we also provide a [direct download link for the entire dataset in `.zip` format](https://rdr.kuleuven.be/api/access/dataset/:persistentId/?persistentId=doi:10.48804/K3VSND), a [onedrive repository containing then entire dataset split up into smaller files](https://kuleuven-my.sharepoint.com/:f:/g/personal/lies_bollens_kuleuven_be/EulH76nkcwxIuK--XJhLxKQBaX8_GgAX-rTKK7mskzmAZA?e=N6M5Ll) and a [tool to download (subsets of) the dataset robustly](download_code/README.md).
For more information about the tool, see [download_code/README.md](download_code/README.md). 

Due to privacy concerns, not all data is publically available. Users requesting access to these files should send a mail to the authors [sparrkulee@kuleuven.be](mailto:sparrkulee@kuleuven.be) , stating what they want to use the data for. Access will be granted to non-commercial users, complying to the CC-BY-NC-4.0 licence

When you want to directly start from the preprocessed data (which is the output you will get when running the file 
[preprocessing_code/examples/auditory_eeg_dataset.py](preprocessing_code/examples/auditory_eeg_dataset.py)), 
you can download the **derivatives** folder. This folder contains all the necessary files to run the technical validation. This can also be downloaded using [the download tool](download_code/README.md) as follows:

```bash
python3 download_code/download_script_from_rdr.py --subset preprocessed /path/to/local/folder
```


## 3. Adjust the [config.json](config.json) accordingly

The [config.json](config.json) defining the folder names and structure for the data and derivatives folder.
Adjust `dataset_folder` in the [config.json](config.json) file from `null` to the absolute path to the (mounted) folder containing all data.
  

OK, you should be all setup now!

# Next steps

See
* the [preprocessing_code/README.md](preprocessing_code/README.md) for instructions on how to run the preprocessing pipeline
* the [technical_validation/README.md](technical_validation/README.md) for instructions on how to run the technical validation code.
