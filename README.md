Code to preprocess BIDS datasets in python and to validate the preprocessed dataset using esthablished models.
==========================================
This is the codebase for the [2023  Auditory EEG Dataset](https://doi.org/10.48804/K3VSND).
This codebase consist of two main parts: 
1) preprocessing code, to preprocess the raw data into an easily usable format 
2) technical validation code, to validate the technical quality of the dataset. 
This code is used to generate the results in the dataset paper and assumes that the preprocessing pipeline has been run


Requirements
------------

Python > 3.6

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

## 2. [Download the data](https://doi.org/10.48804/K3VSND) or [Easy download](https://kuleuven-my.sharepoint.com/:f:/g/personal/lies_bollens_kuleuven_be/EulH76nkcwxIuK--XJhLxKQBaX8_GgAX-rTKK7mskzmAZA?e=N6M5Ll)

When you want to directly start from the preprocessed data, which is the output you will get when running the file 
[preprocessing_code/examples/auditory_eeg_dataset.py](preprocessing_code/examples/auditory_eeg_dataset.py), 
you can download the **derivatives** folder, which contains all the necessary files to run the technical validation. 

When you want to download the raw data and preprocess it yourself, you can download the whole dataset.


## 3. Adjust the `config.json` accordingly

The `config.json` defining the folder names and structure for the data and derivatives folder.
Adjust `dataset_folder` in the `config.json` file from `null` to the absolute path to the folder containing all data.
  

OK, you should be all setup now!

Preprocessing code 
==================  

This repository contains code to efficiently preprocess BIDS datasets in python3

Currently, only EEG datasets are supported. The initial main goal of this code is to use
it for the [public auditory EEG dataset](https://doi.org/10.48804/K3VSND)



Example usage
-------------

See [preprocessing_code/examples/auditory_eeg_dataset.py](preprocessing_code/examples/auditory_eeg_dataset.py)

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

