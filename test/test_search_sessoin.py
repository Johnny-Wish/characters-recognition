import unittest
from preprocess import Dataset
from hypertune.search_session import SearchSession
from hypertune.reflexive_import import ReflexiveImporter


class TestSearchSession(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSearchSession, self).__init__(*args, **kwargs)
        importer = ReflexiveImporter("neural_net_adam")
        dataset = Dataset(folder="../dataset")
        self.session = SearchSession(importer.model, importer.param_dist, dataset, n_iter=1, cv=3)
