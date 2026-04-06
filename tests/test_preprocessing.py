import os
import shutil

from preprocessing_code.sparrKULee import run_preprocessing_pipeline
from tests import BaseTestCase


class TestPreprocessing(BaseTestCase):

    def test_prep_1_recording(self):
        super().setUp()
        # Set the correct paths as default arguments
        dataset_folder = self.config['dataset_folder']
        shutil.rmtree(self.test_output_folder, ignore_errors=True)
        preprocessed_stimuli_folder = os.path.join(self.test_output_folder, "preprocessed_stimuli")
        preprocessed_eeg_folder = os.path.join(self.test_output_folder, "preprocessed_eeg")

        # Run the preprocessing
        run_preprocessing_pipeline(
            root_dir=dataset_folder,
            preprocessed_stimuli_dir=preprocessed_stimuli_folder,
            preprocessed_eeg_dir=preprocessed_eeg_folder,
            glob_patterns=[os.path.join(dataset_folder, "sub-027", "*", "eeg", "*run-04_eeg.bdf.gz")],
            overwrite=True,
            nb_processes=0,
            log_path=None
        )
        # Check if appriopriate files are generated
        self.assertTrue(os.path.exists(os.path.join(preprocessed_stimuli_folder, "audiobook_7_1.data_dict")))
        self.assertTrue(os.path.exists(os.path.join(preprocessed_stimuli_folder, "audiobook_7_1_-_envelope.npy")))
        self.assertTrue(os.path.exists(os.path.join(preprocessed_stimuli_folder, "audiobook_7_1_-_mel.npy")))
        self.assertTrue(os.path.exists(os.path.join(preprocessed_stimuli_folder, ".save_metadata.json")))

        # EEG file
        self.assertTrue(os.path.exists(os.path.join(preprocessed_eeg_folder, "sub-027","ses-varyingStories01", "sub-027_ses-varyingStories01_task-listeningActive_run-04_desc-preproc-audio-audiobook_7_1_eeg.npy")))
        self.assertTrue(os.path.exists(os.path.join(preprocessed_eeg_folder, ".save_metadata.json")))






