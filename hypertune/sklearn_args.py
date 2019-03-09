from cli import BaseModelParser, BaseModelArgs
from cli import BaseDatasetParser, BaseDatasetArgs
from cli import BaseSessionParser, BaseSessionArgs


class SklearnModelParser(BaseModelParser):
    pass


class SklearnModelArgs(BaseModelArgs):
    pass


class SklearnDatasetParser(BaseDatasetParser):
    pass


class SklearnDatasetArgs(BaseDatasetArgs):
    pass


class SklearnSessionParser(SklearnModelParser, SklearnDatasetParser, BaseSessionParser):
    def _setup(self):
        SklearnModelParser._setup(self)
        SklearnDatasetParser._setup(self)
        BaseSessionParser._setup(self)
        self.add_argument("--n_iter", default=200, type=int, help="number of iterations for random searching")
        self.add_argument("--cv", default=5, type=int, help="number of folds for cross validation")
        return self


class SklearnSessionArgs(SklearnModelArgs, SklearnDatasetArgs, BaseSessionArgs):
    @property
    def n_iter(self):
        return self.args.n_iter

    @property
    def cv(self):
        return self.args.cv
