from args import BaseModelParser, BaseModelArgs
from args import BaseDatasetParser, BaseDatasetArgs
from args import BaseSessionParser, BaseSessionArgs


class SklearnModelParser(BaseModelParser):
    pass


class SklearnModelArgs(BaseModelArgs):
    pass


class SklearnDatasetParser(BaseDatasetParser):
    def _setup(self):
        BaseDatasetParser.__init__(self)
        self.add_argument("--train_rate", default=0.01, type=float, help="ratio of training set to be used")
        self.add_argument("--test_rate", default=0.03, type=float, help="ratio of testing set to be used")


class SklearnDatasetArgs(BaseDatasetArgs):
    @property
    def train_rate(self):
        return self.args.train_rate

    @property
    def test_rate(self):
        return self.args.test_rate


class SklearnSessionParser(SklearnModelParser, SklearnDatasetParser, BaseSessionParser):
    def _setup(self):
        SklearnModelParser._setup(self)
        SklearnDatasetParser._setup(self)
        BaseSessionParser._setup(self)
        self.add_argument("--n_iter", default=200, type=int, help="number of iterations to run random searching")
        self.add_argument("--cv", default=5, type=int, help="number of folds for cross validation while searching")
        return self


class SklearnSessionArgs(SklearnModelArgs, SklearnDatasetArgs, BaseSessionArgs):
    @property
    def n_iter(self):
        return self.args.n_iter

    @property
    def cv(self):
        return self.args.cv
