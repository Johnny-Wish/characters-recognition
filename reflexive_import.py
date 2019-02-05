import importlib
from api import deprecated


class Null:
    """a custom Null close representing a Null object"""
    pass


class ReflexiveImporter:
    def __init__(self, module_name, var_list, package_name=None, alias_list=None,
                 model_name="model", param_name="parameter_distribution"):
        # set the package
        if package_name == ".":
            self._package_name = None
        else:
            self._package_name = package_name

        # set the module based on package
        if self._package_name is not None:
            if module_name.startswith("."):
                self._module_name = module_name
            else:
                self._module_name = "." + module_name

        # set a custom Null instance, which will be used as a placeholder
        self.null = Null()

        # set a mapping from var_names to variables
        self._vars = {var: self.null for var in var_list}

        # set a mapping from aliases (if specified) to var_names
        if alias_list is None:
            self._alias_lookup = {var: var for var in var_list}
        else:
            self._alias_lookup = {alias: var for alias, var in zip(alias_list, var_list)}

        self._model_name = model_name  # deprecated
        self._param_name = param_name  # deprecated
        self._vars[self._model_name] = self.null  # deprecated, for backward compatibility
        self._vars[self._param_name] = self.null  # deprecated, for backward compatibility

        # use self.null as a placeholder for the module object
        self._module = self.null

    def _set_module(self):
        try:
            if self._package_name is not None:
                importlib.import_module(self._package_name)
            self._module = importlib.import_module(self._module_name, self._package_name)
        except ModuleNotFoundError as e:
            print("there is no module {}.py under package {}".format(self._module_name, self._package_name))
            raise e  # re-raise error

    @deprecated
    def _set_model(self):
        if self._module is self.null:
            self._set_module()
        try:
            self._vars[self._model_name] = getattr(self._module, self._model_name)
        except AttributeError as e:
            print("module {}.py does not have a {}".format(self._module_name, self._model_name))
            raise e  # re-raise error

    @deprecated
    def _set_param(self):
        if self._module is self.null:
            self._set_module()
        try:
            self._vars[self._param_name] = getattr(self._module, self._param_name)
        except AttributeError as e:
            print("module {}.py does not have a {}".format(self._module_name, self._param_name))
            raise e  # re-raise error

    @property
    @deprecated  # deprecation decorator must be nested inside property decorator
    def model(self):
        if self._vars[self._model_name] is self.null:
            self._set_model()
        return self._vars[self._model_name]

    @property
    @deprecated  # deprecation decorator must be nested inside property decorator
    def param_dist(self):
        if self._vars[self._param_name] is self.null:
            self._set_param()
        return self._vars[self._param_name]

    def _set_var(self, var_name):
        if self._module is self.null:
            self._set_module()
        try:
            self._vars[var_name] = getattr(self._module, var_name)
        except AttributeError as e:
            print("module {}.py does not have a {}".format(self._module_name, self._param_name))
            raise e  # re-raise error

    def __getitem__(self, item):
        if item not in self._alias_lookup:
            raise LookupError("Unknown alias for importer: {}".format(item))
        var_name = self._alias_lookup[item]
        if self._vars[var_name] is self.null:
            self._set_var(var_name)
        return self._vars[var_name]
