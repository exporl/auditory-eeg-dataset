import json
import os
import shutil
import unittest


class BaseTestCase(unittest.TestCase):

    def setUp(self):
        # Dummy config
        self.project_folder = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(self.project_folder, "config.json"), "r") as f:
            self.config = json.load(f)

        if self.config['dataset_folder'] == "null":
            self.config['dataset_folder'] = os.path.join(self.project_folder, 'sparrkulee')

        self.test_output_folder = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(self.test_output_folder, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_output_folder, ignore_errors=True)
