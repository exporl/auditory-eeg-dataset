Code to preprocess BIDS datasets in python and to validate the preprocessed dataset using esthablished models.
==========================================



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



# Run the experiments for the dataset paper

1) split_and_normlaize.py

2) run_experiment

match_mismatch_dilated_convolutional_model.py

regression_linear_backwards_model.py --highpass 0.5 -lowpass 30
regression_linear_backwards_model.py --highpass 0.5 -lowpass 4
regression_linear_backwards_model.py --highpass 4 -lowpass 8
regression_linear_backwards_model.py --highpass 8 -lowpass 14
regression_linear_backwards_model.py --highpass 14 -lowpass 30

regression_linear_forward.py --highpass 0.5 -lowpass 30
regression_linear_forward.py --highpass 0.5 -lowpass 4

regression_vlaai.py 


Auditory-eeg-challenge-2023-code
================================
This is the codebase for the [2023  Auditory EEG Dataset](https://exporl.github.io/auditory-eeg-challenge-2023).
This codebase consist of two main parts: 
1) preprocessing code, to preprocess the raw data into an easily usable format 
2) technical validation code, to validate the technical quality of the dataset. 
This code is used to generate the results in the dataset paper and assumes that the preprocessing pipeline has been run 


    

# Running the tasks

Each task has already some ready-to-go experiments files defined to give you a
baseline and make you acquainted with the problem. The experiment files live
in the `experiment` subfolder for each task. The training log,
best model and evaluation results will be stored in a folder called
`results_{experiment_name}`.

## Task1: Match-mismatch
    
By running [task1_match_mismatch/experiments/dilated_convolutional_model.py](./task1_match_mismatch/experiments/dilated_convolutional_model.py),
you can train the dilated convolutional model introduced by Accou et al. [(2021a)](https://doi.org/10.23919/Eusipco47968.2020.9287417) and [(2021b)](https://doi.org/10.1088/1741-2552/ac33e9).

Other models you might find interesting are [Decheveigné et al (2021)](https://www.sciencedirect.com/science/article/pii/S1053811918300338), [Monesi et al. (2020)](https://ieeexplore.ieee.org/abstract/document/9054000), [Monesi et al. (2021)](https://arxiv.org/abs/2106.09622),….

## Task2: Regression (reconstructing envelope from EEG)

By running [task2_regression/experiments/linear_baseline.py](./task2_regression/experiments/linear_baseline.py), you can 
train and evaluate a simple linear baseline model with Pearson correlation as a loss function, similar to the baseline model used in [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945).

By running [task2_regression/experiments/vlaai.py](./task2_regression/experiments/vlaai.py), you can train/evaluate
the VLAAI model as proposed by [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945). You can find a pre-trained model at [VLAAI's github page](https://github.com/exporl/vlaai).

Other models you might find interesting are: [Thornton et al. (2022)](https://iopscience.iop.org/article/10.1088/1741-2552/ac7976),...

# Previous version

If you are still using a previous version of this example code, we recommend updating to this version, as the test-set code and data will be made compatible for this version.
If you still like access to the previous version, you can find it [here](https://github.com/exporl/auditory-eeg-challenge-2023-code/tree/258b2d48bab4f2ac1da01b8c2aa30f6396063ff5)
