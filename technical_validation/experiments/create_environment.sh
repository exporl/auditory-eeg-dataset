#!/bin/bash

#create the necesary environment variables
source ~/.bashrc
source ~/.local/bin/load_conda_environment.sh bids_preprocessing

printenv

# Fix for unrecognized 'CUDA0' CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/CUDA//g' | sed -e 's/\://g')
export PYTHONPATH="/esat/spchtemp/scratch/baccou/auditory-eeg-dataset/"
echo "$@"

#run the original
$@

