"""Example experiment for a linear baseline method."""


import sys


import glob
import json
import logging
import os
import tensorflow as tf

import argparse

from technical_validation.models.linear import simple_linear_model
from technical_validation.util.dataset_generator import RegressionDataGenerator, create_tf_dataset


def evaluate_model(model, ds_test):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: tf.data.Dataset
        a tf.data.Dataset containing the test
        set for a subject.

    Returns
    -------
    dict
        Mapping between the metrics and the loss/evaluation score on the test set
    """

    results = model.evaluate(ds_test, verbose=2)
    metrics = model.metrics_names

    evaluation = dict(zip(metrics, results))
    return evaluation


if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length = 10 * 64  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64

    # choose the frequency band you want to filter the data
    # delta (0.5 -4 )
    # theta (4 - 8)
    # alpha (8 - 14)
    # beta (14 - 30)
    # broadband (0.5 - 32)
    # add argument input
    parser = argparse.ArgumentParser()
    parser.add_argument('--highpass', type=float, default=0.5)
    parser.add_argument('--lowpass', type=float, default=31)

    args = parser.parse_args()
    highpass = args.highpass
    lowpass = args.lowpass
    epochs = 100
    patience = 5
    batch_size = 64
    only_evaluate = False

    results_filename = f'eval_window_{window_length}_filter_{highpass}_{lowpass}.json'


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
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_linear_backward")
    os.makedirs(results_folder, exist_ok=True)

    # get all the subjects
    all_files = glob.glob(os.path.join(data_folder, "train_-_*"))
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in all_files]))

    evaluation_all_subs = {}

    # train one model per subject
    for subject in subjects:

        # create a simple linear model
        model = simple_linear_model()
        model.summary()
        model_path = os.path.join(results_folder, f"model_{subject}_window_{window_length}_filter_{highpass}_{lowpass}.h5" )
        training_log_filename = f"training_log_{subject}_window_{window_length}_filter_{highpass}_{lowpass}.csv"

        if only_evaluate:
            model = tf.keras.models.load_model(model_path)
        else:

            train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features and 'matrix' not in x and subject in x]
            # Create list of numpy array files
            train_generator = RegressionDataGenerator(train_files, window_length, high_pass_freq=highpass, low_pass_freq=lowpass)
            dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64,1))


            # Create the generator for the validation set
            val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features and 'matrix' not in x and subject in x]
            val_generator = RegressionDataGenerator(val_files, window_length, high_pass_freq=highpass, low_pass_freq=lowpass)

            dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size,
                                              data_types=(tf.float32, tf.float32), feature_dims=(64, 1))

            # Train the model
            model.fit(
                dataset_train,
                epochs=epochs,
                validation_data=dataset_val,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
                    tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                    tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
                ],
            )

        # Evaluate the model on test set
        # Create a dataset generator for each test subject
        test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features and 'matrix' not in x and subject in x]

        test_generator = RegressionDataGenerator(test_files, window_length, high_pass_freq=highpass, low_pass_freq=lowpass)

        dataset_test = create_tf_dataset(test_generator, window_length, None, hop_length, batch_size,
                          data_types=(tf.float32, tf.float32), feature_dims=(64, 1))


        # Evaluate the model
        logging.info(f"Evaluating model on test set for subject {subject}")
        evaluation_all_subs[subject] = evaluate_model(model, dataset_test)




    # We can save our results in a json encoded file
    results_path = os.path.join(results_folder, results_filename)
    with open(results_path, "w") as fp:
        json.dump(evaluation_all_subs, fp)
    logging.info(f"Results saved at {results_path}")
