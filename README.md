Code to preprocess the SparrKULee dataset
=========================================
This is the codebase to preprocess and validate the [SparrKULee](https://doi.org/10.48804/K3VSND).
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

## 2. [Download (parts of the) data](download_code/README.md) 

The official dataset is hosted on the [KU Leuven RDR website](https://doi.org/10.48804/K3VSND) and is accessible through DOI ([https://doi.org/10.48804/K3VSND](https://doi.org/10.48804/K3VSND)).

However, due to the dataset size/structure and the limitations of the UI of the KU Leuven RDR website, we also provide a [direct download link for the entire dataset in `.zip` format](https://rdr.kuleuven.be/api/access/dataset/:persistentId/?persistentId=doi:10.48804/K3VSND), a [onedrive repository containing then entire dataset split up into smaller files](https://kuleuven-my.sharepoint.com/:f:/g/personal/lies_bollens_kuleuven_be/EulH76nkcwxIuK--XJhLxKQBaX8_GgAX-rTKK7mskzmAZA?e=N6M5Ll) and a [tool to download (subsets of) the dataset robustly](download_code/README.md).
For more information about the tool, see [download_code/README.md](download_code/README.md). 

Due to privacy concerns, not all data is publically available. Users requesting access to these files should send a mail to the authors (lies.bollens@kuleuven.be ; bernd.accou@kuleuven.be) , stating what they want to use the data for. Access will be granted to non-commercial users, complying to the CC-BY-NC-4.0 licence

When you want to directly start from the preprocessed data (which is the output you will get when running the file 
[preprocessing_code/examples/auditory_eeg_dataset.py](preprocessing_code/examples/auditory_eeg_dataset.py)), 
you can download the **derivatives** folder. This folder contains all the necessary files to run the technical validation. This can also be downloaded using [the download tool](download_code/README.md) as follows:

```bash
python3 download_code/download_script.py --subset preprocessed /path/to/local/folder
```


## 3. Adjust the [config.json](config.json) accordingly

The [config.json](config.json) defining the folder names and structure for the data and derivatives folder.
Adjust `dataset_folder` in the [config.json](config.json) file from `null` to the absolute path to the folder containing all data.
  

OK, you should be all setup now!

Preprocessing code 
==================  

This repository uses the [brain_pipe package](https://github.com/exporl/brain_pipe) 
to preprocess the data. It is installed automatically when installing the [requirements.txt](requirements.txt).
You are invited to contribute to the [brain_pipe package](https://github.com/exporl/brain_pipe)  package, if you want to add new preprocessing steps.
Documentation for the brain_pipe package can be found [here](https://exporl.github.io/brain_pipe/).

Example usage
-------------

There are multiple ways to run the preprocessing pipeline:

### 1. Use the python script [preprocessing_code/sparrKULee.py](preprocessing_code/sparrKULee.py)

```bash
python3 preprocessing_code/sparrKULee.py
```

Different options (such as the number of parallel processes) can be specified from the command line.
For more information, run :

```bash
python3 preprocessing_code/sparrKULee.py --help.
```

### 2. Use the YAML file with the [brain_pipe](https://github.com/exporl/brain_pipe) CLI

For this option, you will have to fill in the `--dataset_folder`, `--derivatives_folder`,
`--preprocessed_stimuli_dir` and `--preprocessed_eeg_dir` with the values from the [config.json](config.json) file.

```bash
brain_pipe preprocessing_code/sparrKULee.yaml --dataset_folder {/path/to/dataset} --derivatives_folder {derivatives_folder} --preprocessed_stimuli_dir {preprocessed_stimuli_dir} --preprocessed_eeg_dir {preprocessed_eeg_dir}
```

Optionally, you could read the [config.json](config.json) file directly from the command line:

```bash
brain_pipe preprocessing_code/sparrKULee.yaml $(python3 -c "import json; f=open('config.json'); d=json.load(f); f.close(); print(' '.join([f'--{x}={y}' for x,y in d.items() if 'split_folder' != x]))")
```

For more information about the [brain_pipe](https://github.com/exporl/brain_pipe) CLI,
see the appriopriate documentation for the [CLI](https://exporl.github.io/brain_pipe/cli.html) and [configuration files (e.g. YAML)](https://exporl.github.io/brain_pipe/configuration.html)

Technical validation
====================
This repository contains code to validate the preprocessed dataset using esthablished models.
Running this code will yield the results summarized in the paper. (LINK TO PAPER)

Prerequisites 
-------------
The technical validation code assumes that the preprocessing pipeline has been run and that the derivatives folder is available.
The derivatives folder contains the preprocessed data and the necessary files to run the technical validation code.
Either download the derivatives folder directly from the online dataset or run the preprocessing pipeline yourself [preprocessing_code/examples/auditory_eeg_dataset.py](preprocessing_code/examples/auditory_eeg_dataset.py).

Example usage
-------------

We have defined some ready-to-go experiments, to replicate the results summarized in the dataset paper. 
All these experiments use split (into training/validation/test partitions) and normalised data, which can be obtained by 
running [technical_validation/util/split_and_normalize.py](technical_validation/util/split_and_normalize.py).

The experiment files live in the  [technical_validation/experiments](technical_validation/experiments) folder. The training log,
best model and evaluation results will be stored in a folder called
`results_{experiment_name}`.

To replicate the results summarized in the dataset paper, run the following experiments:
```bash
# train the dilated convolutional model introduced by Accou et al.(https://doi.org/10.1088/1741-2552/ac33e9) 
match_mismatch_dilated_convolutional_model.py

# train a simple linear backward model, reconstructing the envelope from EEG
# using filtered data in different frequency bands
# simple linear baseline model with Pearson correlation as a loss function, similar to the baseline model used in Accou et al (2022) (https://www.biorxiv.org/content/10.1101/2022.09.28.509945).

regression_linear_backwards_model.py --highpass 0.5 -lowpass 30
regression_linear_backwards_model.py --highpass 0.5 -lowpass 4
regression_linear_backwards_model.py --highpass 4 -lowpass 8
regression_linear_backwards_model.py --highpass 8 -lowpass 14
regression_linear_backwards_model.py --highpass 14 -lowpass 30

# train a simple linear forward model, predicting the EEG response from the envelope, 
# using filtered data in different frequency bands
regression_linear_forward.py --highpass 0.5 -lowpass 30
regression_linear_forward.py --highpass 0.5 -lowpass 4

# train/evaluate the VLAAI model as proposed by Accou et al (2022) (https://www.biorxiv.org/content/10.1101/2022.09.28.509945). You can find a pre-trained model at VLAAI's github page (https://github.com/exporl/vlaai).
regression_vlaai.py 
```

Finally, you can generate the plots as shown in the dataset paper by running the [technical_validation/util/plot_results.py](technical_validation/util/plot_results.py) script 

