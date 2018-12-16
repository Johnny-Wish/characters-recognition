import sys
import os


class ReflexiveImporter:
    def __init__(self, module_name, package_name="sklearn_models",
                 model_name="model", param_name="parameter_distribution"):
        root_dir = "/".join(__file__.split("/")[:-2])
        sys.path.append(os.path.join(root_dir, package_name))
        self._package_name = package_name
        self._module_name = module_name
        self._model_name = model_name
        self._param_name = param_name
        self._module = None
        self._param = None
        self._model = None

    def _set_module(self):
        try:
            self._module = __import__(self._module_name)
        except ModuleNotFoundError as e:
            print("there is no module {}.py under package {}".format(self._module_name, self._package_name))
            raise e  # re-raise error

    def _set_model(self):
        if self._module is None:
            self._set_module()
        try:
            self._model = getattr(self._module, self._model_name)
        except AttributeError as e:
            print("module {}.py does not have a {}".format(self._module_name, self._model_name))
            raise e  # re-raise error

    def _set_param(self):
        if self._module is None:
            self._set_module()
        try:
            self._param = getattr(self._module, self._param_name)
        except AttributeError as e:
            print("module {}.py does not have a {}".format(self._module_name, self._param_name))
            raise e  # re-raise error

    @property
    def model(self):
        if self._model is None:
            self._set_model()
        return self._model

    @property
    def param_dist(self):
        if self._param is None:
            self._set_param()
        return self._param
