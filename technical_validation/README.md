

Technical validation
====================
This repository contains code to validate the preprocessed dataset using esthablished models.
Running this code will yield the results summarized in the [paper](https://doi.org/10.3390/data9080094)

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
