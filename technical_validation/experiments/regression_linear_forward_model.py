"""Example experiment for a linear baseline method."""


import sys
import argparse


import numpy as np
import glob
import json
import logging
import os
import scipy.stats

from technical_validation.util.dataset_generator import RegressionDataGenerator, create_tf_dataset



def time_lag_matrix(input_, tmin, tmax):
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
    numChannels = input_.shape[1]

    final_array = np.zeros((input_.shape[0], numChannels * (tmax - tmin)))

    for index, shift in enumerate(range(tmin, tmax)):
        # roll the array to the right
        shifted_data = np.roll(input_, -shift, axis=0)
        final_array[:, index * numChannels: (index + 1) * numChannels] = shifted_data

    if tmin < 0:
        return final_array[np.abs(tmin):-tmax+1, :]
    else:
        return final_array[:-tmax+1, :]


def train_model_cov(cxx, cxy, ridge_param):
    return np.linalg.solve(cxx + ridge_param * np.eye(cxx.shape[0]), cxy)

def evaluate_model(model, test_env, test_eeg):
    pred_eeg = np.matmul(test_env, model)
    channel_scores = []
    for channel in range(test_eeg.shape[1]):
        score = scipy.stats.pearsonr(pred_eeg[:, channel], test_eeg[:, channel])[0]
        channel_scores.append(score.tolist())
    return channel_scores

def permutation_test(model, eeg, env, tmin, tmax, numPermutations=100):
    pred_eeg = np.matmul(env, model)
    corrs = []
    for permutation_index in range(numPermutations):
        print(f'Permutation {permutation_index+1:03d}\r', end='')
        random_shift = np.random.randint(tmax-tmin, eeg.shape[0] - (tmax-tmin))
        temp_eeg = np.roll(eeg, random_shift, axis=0)
        # temp_eeg = temp_eeg[random_shift:, :]
        # temp_pred_eeg = pred_eeg[:-random_shift, :]
        temp_pred_eeg = pred_eeg
        channel_corrs = []
        for channel in range(eeg.shape[1]):
            channel_corrs.append(scipy.stats.pearsonr(temp_eeg[:, channel], temp_pred_eeg[:, channel])[0])
        corrs.append(channel_corrs)
    print()
    return np.array(corrs)

def crossval_over_recordings(all_data, tmin, tmax, ridge_param):
    # Cross validation loop to determine the optimal ridge parameter
    fold_scores = []
    for fold in range(len(all_data)):
        print(f'fold {fold}...')

        # train_folds
        train_eeg_folds = [x[0] for i, x in enumerate(all_data) if i != fold]
        train_env_folds = [x[1] for i, x in enumerate(all_data) if i != fold]

        # test_fold
        test_eeg_fold = [x[0] for i, x in enumerate(all_data) if i == fold][0]
        test_env_fold = [x[1] for i, x in enumerate(all_data) if i == fold][0]

        # create the model
        # closed-form solution,
        train_envs = [time_lag_matrix(env, tmin, tmax) for env in train_env_folds]
        cxx = np.sum([np.matmul(x.T, x) for x in train_envs], axis=0)
        cxy = np.sum([np.matmul(x.T, y[:-(tmax - tmin) + 1, :]) for x, y in
                      zip(train_envs, train_eeg_folds)], axis=0)

        if not isinstance(ridge_param, (int, float)):
            ridge_scores = []
            for lambd in ridge_param:
                model = train_model_cov(cxx, cxy, lambd)
                # evaluate the model on the test set
                score = evaluate_model(model,
                                       time_lag_matrix(test_env_fold, tmin, tmax),
                                       test_eeg_fold[:-(tmax - tmin) + 1, :])
                ridge_scores.append(score)
            fold_scores.append(ridge_scores)
        else:
            model = train_model_cov(cxx, cxy, ridge_param)
            score = evaluate_model(model, time_lag_matrix(test_env_fold, tmin, tmax),
                                   test_eeg_fold[:-(tmax - tmin) + 1, :])
            fold_scores.append(score)
    return fold_scores

def training_loop(subject, data_folder, features, highpass, lowpass, tmin, tmax, ridge_param):
    print(f"Training model for subject {subject}")

    train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features and subject in x]
    train_files = [x for x in train_files if 'audiobook_15_' not in x]
    train_generator = RegressionDataGenerator(train_files, high_pass_freq=highpass, low_pass_freq=lowpass)
    all_data = [x for x in train_generator]

    # Leave-one-out cross validation based on number of recordings
    # Done to determine the optimal ridge parameter
    numFolds = len(all_data)

    fold_scores = crossval_over_recordings(all_data, tmin, tmax, ridge_param)

    if not isinstance(ridge_param, (int, float)):
        fold_scores = np.array(fold_scores)
        # Take the average across channels and folds to obtain 1 correlation value
        # per lambda value
        fold_scores = fold_scores.mean(axis=2).mean(axis=0)
        best_ridge = ridge_param[np.argmax(fold_scores)]
        print(f"Best lambda: {best_ridge}")
    else:
        best_ridge = ridge_param


    # Actual training of the model on all training folds
    train_eegs = [x[0] for x in all_data]
    train_envs = [time_lag_matrix(x[1], tmin, tmax) for x in all_data]
    cxx = np.sum([np.matmul(x.T, x) for x in train_envs], axis=0)
    cxy = np.sum([np.matmul(x.T, y[:-(tmax-tmin)+1, :]) for x, y in zip(train_envs, train_eegs)], axis=0)
    model = train_model_cov(cxx, cxy, best_ridge)

    # # evaluate the model on the test set
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features and subject in x]
    test_files = [x for x in test_files if 'audiobook_15_' not in x]
    test_generator = RegressionDataGenerator(test_files, high_pass_freq=highpass, low_pass_freq=lowpass, return_filenames=True)

    # calculate pearson correlation, per test segment
    # and average over all test segments
    test_info = {'subject': subject, 'stim_filename':[], 'score': [], 'null_distr': [], 'ridge_param': best_ridge, 'model_weights': model.tolist(), 'highpass':highpass, 'lowpass':lowpass, 'numFolds':numFolds}
    print(f"Testing model on test data... ({numFolds})")
    for test_filenames, test_seg in test_generator:
        test_eeg = test_seg[0]
        test_env = test_seg[1]
        stim_filename = os.path.basename(test_filenames[1]).split("_-_")[-2]

        test_env = time_lag_matrix(test_env, tmin, tmax)
        # shorten eeg to match the length of the env
        test_eeg = test_eeg[:-(tmax-tmin)+1, :]
        # predict
        pearson_scores = evaluate_model(model, test_env, test_eeg)

        test_info['stim_filename'].append(stim_filename)
        test_info['score'].append(pearson_scores)
        # null distribution
        null_distr = permutation_test(model, test_eeg, test_env, tmin, tmax).tolist()
        test_info['null_distr'].append(null_distr)
    test_info['mean_score_per_channel'] = np.mean(test_info['score'], axis=0).tolist()
    test_info['mean_score'] = np.mean(test_info['score'])
    null_distr = np.reshape(test_info['null_distr'], (-1, 64))
    test_info['95_percentile_per_channel'] = np.percentile(null_distr, 95, axis=0).tolist()
    test_info['95_percentile'] = np.percentile(test_info['null_distr'], 95)

    return test_info



if __name__ == "__main__":
    # Parameters

    # frequency band chosen for the experiment
    # delta (0.5 -4 )
    # theta (4 - 8)
    # alpha (8 - 14)
    # beta (14 - 30)
    # broadband (0.5 - 32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--highpass', type=float, default=None)
    parser.add_argument('--lowpass', type=float, default=4)

    args = parser.parse_args()
    highpass = args.highpass
    lowpass = args.lowpass

    for highpass, lowpass in [(None, 4), (4, 8), (8, 14), (14, 30), (None, None)]:


        numChannels = 64
        tmin = -np.round(0.1*64).astype(int) # -100 ms
        tmax = np.round(0.4*64).astype(int)  # 400 ms
        ridge_param =  [10**x for x in range(-6, 7, 2)]
        overwrite = False

        results_filename = 'eval_filter_{subject}_{tmin}_{tmax}_{highpass}_{lowpass}.json'

        # Get the path to the config gile
        experiments_folder = os.path.dirname(__file__)
        main_folder = os.path.dirname(os.path.dirname(experiments_folder))
        config_path = os.path.join(main_folder, 'config.json')

        # Load the config
        with open(config_path) as fp:
            config = json.load(fp)

        # Provide the path of the dataset
        # which is split already to train, val, test

        data_folder = os.path.join(config["dataset_folder"], config["derivatives_folder"], config["split_folder"])
        features = ["envelope", "eeg"]

        # Create a directory to store (intermediate) results
        results_folder = os.path.join(experiments_folder, "results_linear_forward")
        os.makedirs(results_folder, exist_ok=True)


        # get all the subjects
        all_files = glob.glob(os.path.join(data_folder, "train_-_*"))
        subjects = list(set([os.path.basename(x).split("_-_")[1] for x in all_files]))

        evaluation_all_subs = {}
        chance_level_all_subs = {}

        # train one model per subject
        for subject in subjects:
            save_path = os.path.join(results_folder, results_filename.format(subject=subject, tmin=tmin, tmax=tmax, highpass=highpass, lowpass=lowpass))
            if not os.path.exists(save_path) or overwrite:
                result = training_loop(subject, data_folder, features, highpass, lowpass, tmin, tmax, ridge_param)

                # save the results
                with open(save_path, 'w') as fp:
                    json.dump(result, fp)
            else:
                print(f"Results for {subject} already exist, skipping...")






