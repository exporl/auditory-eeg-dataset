"""Example experiment for the VLAAI model."""
import glob
import json
import logging
import os
import tensorflow as tf
import sys


from technical_validation.models.vlaai import vlaai, pearson_loss, pearson_metric
from technical_validation.util.dataset_generator import RegressionDataGenerator, create_tf_dataset


def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    # Parameters
    # Length of the decision window
    window_length = 10 * 64  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64
    epochs = 100
    patience = 5
    batch_size = 64
    only_evaluate = False
    training_log_filename = "training_log_%d.csv" %window_length
    results_filename = 'eval_%d.json' % window_length


    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    main_folder = os.path.dirname(os.path.dirname(experiments_folder))
    config_path = os.path.join(main_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test

    data_folder = os.path.join(config["dataset_folder"], config["derivatives"],  config["split_folder"])
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_vlaai")
    os.makedirs(results_folder, exist_ok=True)

    # create the model
    model = vlaai()
    model.compile(tf.keras.optimizers.Adam(), loss=pearson_loss, metrics=[pearson_metric])
    model_path = os.path.join(results_folder, "model_%d.h5" % window_length)

    if only_evaluate:
        model.load_weights(model_path)
    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features ]
        # Create list of numpy array files
        train_generator = RegressionDataGenerator(train_files, window_length)
        dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64,1))

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features ]
        val_generator = RegressionDataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64,1))

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
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    # Get all different subjects from the test set
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    datasets_test = {}
    results_filename = 'eval_pretrained_%d.json' % window_length
    # Create a generator for each subject
    for sub in subjects:
        files_test_sub = [f for f in test_files if sub in os.path.basename(f)]

        test_generator = RegressionDataGenerator(files_test_sub, window_length)
        datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64,1))

    # Evaluate the model
    evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    results_path = os.path.join(results_folder, results_filename)
    with open(results_path, "w") as fp:
        json.dump(evaluation, fp)
    logging.info(f"Results saved at {results_path}")

