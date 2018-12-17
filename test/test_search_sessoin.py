import unittest
import pandas as pd
from preprocess import Dataset
from hypertune.search_session import SearchSession
from hypertune.reflexive_import import ReflexiveImporter


class TestSearchSession(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSearchSession, self).__init__(*args, **kwargs)
        self.test_build_search_session()
        self.test_fitting_process()
        self.test_testing_process()

    def test_build_search_session(self):
        importer = ReflexiveImporter("neural_net_adam")
        dataset = Dataset(folder="../dataset")
        self.session = SearchSession(importer.model, importer.param_dist, dataset, n_iter=1, cv=3)

    def test_fitting_process(self):
        self.session.fit()

    def test_search_results(self):
        self.assertIsInstance(self.session.search_results, pd.DataFrame)

    def test_testing_process(self):
        self.session.test()

    def test_test_results(self):
        self.assertIsInstance(self.session.test_result, dict)
        self.assertIn("accuracy", self.session.test_result)
        self.assertIn("precision", self.session.test_result)
        self.assertIn("recall", self.session.test_result)
        self.assertIn("f1-score", self.session.test_result)
        self.assertIn("support", self.session.test_result)
