import unittest
from reflexive_import import ReflexiveImporter
from api import DeprecatedError


class TestReflexiveImporter(unittest.TestCase):
    def test_legacy(self):
        importer = ReflexiveImporter(
            module_name="temp_module",
            package_name="temp",
            var_list=[],
        )
        with self.assertRaises(DeprecatedError):
            print(importer.param_dist)

        with self.assertRaises(DeprecatedError):
            print(importer.model)

    def test_import_var_with_alias(self):
        import temp.temp_module as tmp

        importer = ReflexiveImporter(
            module_name="temp_module",
            package_name="temp",
            var_list=["some_long_variable_name", "another_long_variable_name", "some_long_function_name"],
            alias_list=["var1", "var2", "func"],
        )

        self.assertEqual(importer['var1'], tmp.some_long_variable_name)
        self.assertEqual(importer['var2'], tmp.another_long_variable_name)
        self.assertEqual(importer['func'], tmp.some_long_function_name)

    def test_import_var_without_alias(self):
        from temp.temp_module import some_long_variable_name as var1
        from temp.temp_module import another_long_variable_name as var2
        from temp.temp_module import some_long_function_name as func
        importer = ReflexiveImporter(
            module_name="temp_module",
            package_name="temp",
            var_list=["some_long_variable_name", "another_long_variable_name", "some_long_function_name"],
        )
        self.assertEqual(importer["some_long_variable_name"], var1)
        self.assertEqual(importer["another_long_variable_name"], var2)
        self.assertEqual(importer["some_long_function_name"], func)

    def test_import_error(self):
        importer1 = ReflexiveImporter(
            module_name="temp_module",
            package_name="some_nonexistent_name",
            var_list=["some_long_variable_name"],
        )
        with self.assertRaises(ImportError):
            print(importer1["some_long_variable_name"])

        importer2 = ReflexiveImporter(
            module_name="some_nonexistent_name",
            package_name="temp",
            var_list=["some_long_variable_name"],
        )
        with self.assertRaises(ImportError):
            print(importer2["some_long_variable_name"])

        importer3 = ReflexiveImporter(
            module_name="temp_module",
            package_name="temp",
            var_list=["some_long_variable_name"],
        )
        with self.assertRaises(LookupError):
            print(importer3["no_such_alias"])

    def test_attribute_error(self):
        importer1 = ReflexiveImporter(
            module_name="temp_module",
            package_name="temp",
            var_list=["model_var"],
            alias_list=["model"],
        )
        with self.assertRaises(LookupError):
            print(importer1['some_nonexistent_alias'])

        with self.assertRaises(LookupError):
            print(importer1['model_var'])  # `model_var` is only accessible as `model`
