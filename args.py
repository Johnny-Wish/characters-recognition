import argparse


class _StaticParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        return self


class _StaticRegister:
    def __init__(self, parser: argparse.ArgumentParser):
        self.args = parser.parse_args()

    def __str__(self):
        return self.args.__str__()

    def __repr__(self):
        return self.args.__repr__()


class BaseDatasetParser(_StaticParser):
    def _setup(self):
        _StaticParser._setup(self)
        self.add_argument("--dataroot", default="../dataset", help="folder to hold dataset")
        self.add_argument("--datafile", default="emnist-byclass.mat", help="filename of dataset")
        return self


class BaseDatasetArgs(_StaticRegister):
    @property
    def dataroot(self):
        return self.args.dataroot

    @property
    def datafile(self):
        return self.args.datafile


class BaseModelParser(_StaticParser):
    def _setup(self):
        _StaticParser._setup(self)
        self.add_argument("--model", required=True, help="python module (with package prefix) for dynamic model import")
        return self


class BaseModelArgs(_StaticRegister):
    @property
    def model(self):
        return self.args.model


class BaseSessionParser(_StaticParser):
    def _setup(self):
        self.add_argument("--output", default="/output", type=str, help="folder to store trained parameters")


class BaseSessionArgs(_StaticRegister):
    @property
    def output(self):
        return self.args.output
