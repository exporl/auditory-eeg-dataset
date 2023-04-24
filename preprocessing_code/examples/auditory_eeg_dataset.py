import gzip
import logging
import os
import time
import sys

import numpy as np
import scipy.signal
from bids import BIDSLayout

from bids_preprocessing.preprocessing.brain.artifact import (
    InterpolateArtifacts,
    ArtifactRemovalMWF,
)
from bids_preprocessing.preprocessing.brain.eeg.load import LoadEEGNumpy
from bids_preprocessing.preprocessing.brain.epochs import SplitEpochs
from bids_preprocessing.preprocessing.brain.link import LinkStimulusToBrainResponse
from bids_preprocessing.preprocessing.brain.rereference import CommonAverageRereference
from bids_preprocessing.preprocessing.brain.trigger import (
    AlignPeriodicBlockTriggers,
    biosemi_trigger_processing_fn,
)
from bids_preprocessing.preprocessing.filter import SosFiltFilt
from bids_preprocessing.preprocessing.pipeline import PreprocessingPipeline
from bids_preprocessing.preprocessing.resample import ResamplePoly
from bids_preprocessing.preprocessing.save.default import Save, BIDS_filename_fn
from bids_preprocessing.preprocessing.stimulus.audio.envelope import GammatoneEnvelope
from bids_preprocessing.preprocessing.stimulus.load import LoadStimuli, DEFAULT_LOAD_FNS
from bids_preprocessing.utils.logging import default_logging, DefaultFormatter
from bids_preprocessing.utils.multiprocessing import MultiprocessingSingleton
from bids_preprocessing.utils.path import BIDSStimulusGrouper, BIDSPathGenerator


def temp_stimulus_load_fn(path):
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        with gzip.open(path + '.gz', 'rb') as f_in:
            data = np.load(f_in)
        return data

    extension = "." + ".".join(path.split(".")[1:])
    if extension not in DEFAULT_LOAD_FNS:
        raise ValueError(
            f"Can't find a load function for extension {extension}. "
            f"Available extensions are {str(list(DEFAULT_LOAD_FNS.keys()))}."
        )
    load_fn = DEFAULT_LOAD_FNS[extension]
    return load_fn(path)

if __name__ == "__main__":

    ####################
    # GENERAL SETTINGS #
    ####################

    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    main_folder = os.path.dirname(os.path.dirname(experiments_folder))
    config_path = os.path.join(main_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # GENERAL PATHS #
    #################
    config_file =
    root_dir = config['dataset_folder']
    output_dir = os.path.join(config['dataset_folder'], config['derivatives_folder'], config['preprocessed_eeg_folder'])
    stimuli_dir = os.path.join(config['dataset_folder'], config['derivatives_folder'], config['preprocessed_stimuli_folder'])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stimuli_dir, exist_ok=True)

    # SUBJECT/STIMULUS SELECTION CRITERIA #
    #######################################
    # select the number of processes that will be concurrently run, depending on cpu/ram availibility 
    nb_processes = 1 # 8

    # if empty array, all found values will be processed (default)
    subjects = []
    sessions = []
    tasks = []
    runs = []
    extensions = ['bdf', 'bdf.gz']
    suffix = "eeg"


    handler = logging.FileHandler('auditory_eeg_dataset.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])
    map_fn = MultiprocessingSingleton.get_map_fn(nb_processes)

    logging.info("Retrieving BIDS layout...")
    logging.info("Retrieving BIDS layout...")
    start_time = time.time()
    layout = BIDSLayout(root_dir, validate=False)
    logging.info(f"BIDS layout retrieved in {time.time()-start_time:.2f} seconds")
    

    eeg_paths = layout.get(
        return_type="file",
        extension=extensions,
        run=runs,
        task=tasks,
        subject=subjects,
        suffix=suffix,
        session=sessions
    )

    # STEPS #
    #########

    stimulus_steps = PreprocessingPipeline(
        steps=[
            LoadStimuli(load_fn=temp_stimulus_load_fn),
            GammatoneEnvelope(),
            ResamplePoly(64, "envelope_data", "stimulus_sr"),
            Save(stimuli_dir),
        ],
        on_error=PreprocessingPipeline.RAISE,
    )

    eeg_steps = [
        LinkStimulusToBrainResponse(
            stimulus_data=stimulus_steps,
            grouper=BIDSStimulusGrouper(
                bids_root=root_dir,
                mapping={"stim_file": "stimulus_path", "trigger_file": "trigger_path"},
                subfolders=["stimuli", "eeg"],
            ),
        ),
        LoadEEGNumpy(unit_multiplier=1e6, channels_to_select=list(range(64))),
        SosFiltFilt(
            scipy.signal.butter(1, 0.5, "highpass", fs=1024, output="sos"),
            emulate_matlab=True,
            axis=1,
        ),
        InterpolateArtifacts(),
        AlignPeriodicBlockTriggers(biosemi_trigger_processing_fn),
        SplitEpochs(),
        ArtifactRemovalMWF(),
        CommonAverageRereference(),
        ResamplePoly(64, axis=1),
        Save(
            output_dir,
            {"eeg": "data"},
            overwrite=False,
            clear_output=True,
            filename_fn=BIDS_filename_fn
        ),
    ]

    ##########################
    # Preprocessing pipeline #
    ##########################

    logging.info("Starting with the EEG preprocessing")
    logging.info("===================================")
    logging.info(f"Found {len(eeg_paths)} EEG files")


    # Create data_dicts for the EEG files
    data_dicts_eeg = [{"data_path": p} for p in eeg_paths]
    # Create the EEG pipeline
    eeg_pipeline = PreprocessingPipeline(steps=eeg_steps)
    # Execute the EEG pipeline
    preprocessed_eeg_data_dicts = list(map_fn(eeg_pipeline, data_dicts_eeg))
    MultiprocessingSingleton.clean()