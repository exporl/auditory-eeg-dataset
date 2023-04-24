"""Example experiment for a linear baseline method."""


import sys
import argparse


import numpy as np
import glob
import json
import logging
import os
import tensorflow as tf
import scipy.stats

from technical_validation.util.dataset_generator import RegressionDataGenerator, create_tf_dataset



def time_lag_matrix(eeg, num_lags):
    """Create a time-lag matrix from a 2D numpy array.

    Parameters
    ----------
    eeg: np.ndarray
        2D numpy array with shape (n_samples, n_channels)
    num_lags: int
        Number of time lags to use.

    Returns
    -------
    np.ndarray
        2D numpy array with shape (n_samples, n_channels* num_lags)
    """
    # Create a time-lag matrix
    numChannels = eeg.shape[1]

    for i in range(0,num_lags):
        # roll the array to the right
        eeg_t = np.roll(eeg, i, axis=0)

        if i == 0:
            final_array = eeg_t
        else:
            final_array = np.concatenate((final_array, eeg_t), axis=1)

    # shuffle the columns such that they are ordered by time lag
    final_array = final_array[:, list(np.concatenate([np.arange(i, final_array.shape[1], numChannels) for i in range(numChannels)]))]
    return final_array[num_lags:, :]


if __name__ == "__main__":
    # Parameters

    # frequency band chosen for the experiment
    # delta (0.5 -4 )
    # theta (4 - 8)
    # alpha (8 - 14)
    # beta (14 - 30)
    # broadband (0.5 - 32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--highpass', type=float, default=0.5)
    parser.add_argument('--lowpass', type=float, default=31)

    args = parser.parse_args()
    highpass = args.highpass
    lowpass = args.lowpass


    numChannels = 64
    time_window = 26
    ridge_param = 10000



    results_filename = f'eval_filter_{highpass}_{lowpass}.json'

    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    main_folder = os.path.dirname(os.path.dirname(experiments_folder))
    config_path = os.path.join(main_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test

    data_folder = os.path.join(config["dataset_folder"], config["derivatives"], config["split_folder"])
    features = ["envelope", "eeg"]

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_linear_forward")
    os.makedirs(results_folder, exist_ok=True)


    # get all the subjects
    all_files = glob.glob(os.path.join(data_folder, "train_-_*"))
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in all_files]))

    evaluation_all_subs = {}

    # train one model per subject
    for subject in subjects:
        print(f"Training model for subject {subject}")

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features  and subject in x]
        train_generator = RegressionDataGenerator(train_files, high_pass_freq=highpass, low_pass_freq=lowpass)
        all_data = [x for x in train_generator]

        #concatenate eeg and envelope
        eeg = np.concatenate([x[0] for x in all_data], axis=0)
        env = np.concatenate([x[1] for x in all_data], axis=0)

        # create the model
        # closed-form solution,
        train_env = time_lag_matrix(env, time_window)
        # shorten eeg to match the length of the env
        train_eeg = eeg[time_window:, :]

        # train the model
        correlationMatrix = np.matmul(train_env.T, train_env)

        correlationMatrix = correlationMatrix + ridge_param * np.eye(correlationMatrix.shape[0])
        crossCorrelationMatrix = np.matmul(train_env.T, train_eeg)
        model = np.linalg.solve(correlationMatrix, crossCorrelationMatrix)


        # # evaluate the model on the test set

        test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features and subject in x]
        test_generator = RegressionDataGenerator(test_files, high_pass_freq=highpass, low_pass_freq=lowpass)
        # test_data = [ x for x in test_generator]

        # calculate pearson correlation, per test segment
        # and average over all test segments
        for test_seg in test_generator:
            test_eeg = test_seg[0]
            test_env = test_seg[1]


            test_env = time_lag_matrix(test_env, time_window)
            # shorten eeg to match the length of the env
            test_eeg = test_eeg[time_window:, :]
            # predict
            test_eeg_pred = np.matmul(test_env, model)

            # evaluate
            pearson_scores = []
            for i in range(test_eeg.shape[1]):
                pearson_r = scipy.stats.pearsonr(test_eeg_pred[:,i], test_eeg[:,i])[0]
                pearson_scores.append(pearson_r)

            evaluation_all_subs[subject].append(pearson_scores)

        # average over all the test segments
        evaluation_all_subs[subject] = np.mean(evaluation_all_subs[subject], axis=0)

    # save the results
    with open(os.path.join(results_folder, results_filename), 'w') as fp:
        json.dump(evaluation_all_subs, fp)






