import glob
import json
import os

import numpy as np

from tests import BaseTestCase


class TechnicalValidationTest(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.all_files = glob.glob(
            os.path.join(self.config['dataset_folder'], 'derivatives', 'split_data',
                         '*_-_sub-001_-_audiobook_6_*_-_*.npy'
                         )
        )


    def test_linear_backward_model(self):
        import technical_validation.experiments.regression_linear_backward_model as backward
        backward.main(self.test_output_folder, self.all_files,None, 4)
        # Load results
        with open(os.path.join(self.test_output_folder, 'eval_filter_sub-001_-6_26_4_None.json')) as fp:
            result = json.load(fp)
        self.assertTrue(np.isclose(result['mean_score'], 0.07857143962834898))

    def test_linear_forward_model(self):
        import technical_validation.experiments.regression_linear_forward_model as forward
        forward.main(self.test_output_folder, self.all_files,None, 4)
        # Load results
        with open(os.path.join(self.test_output_folder, 'eval_filter_sub-001_-6_26_4_None.json')) as fp:
            result = json.load(fp)
        self.assertTrue(np.isclose(result['mean_score'], -0.003165490068297306))
