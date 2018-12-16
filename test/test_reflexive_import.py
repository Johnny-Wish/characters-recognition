import unittest
from hypertune.reflexive_import import ReflexiveImporter


class TestReflexiveImporter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestReflexiveImporter, self).__init__(*args, **kwargs)
        self.test_build_importer()

    def test_build_importer(self):
        self.importer = ReflexiveImporter(
            module_name="temp_module",
            package_name="temp",
            model_name="model_var",
            param_name="param_var"
        )

    def test_value(self):
        self.assertEqual(self.importer.param_dist, "This is a param variable")
        self.assertEqual(self.importer.model, "This is a model variable")
