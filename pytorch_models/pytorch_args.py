from args import BaseDatasetArgs, BaseDatasetParser
from args import BaseModelArgs, BaseModelParser
from args import BaseSessionArgs, BaseSessionParser


class TorchDatasetParser(BaseDatasetParser):
    def _setup(self):
        BaseDatasetParser._setup(self)
        self.add_argument("--batch", default=512, type=int, help="size of mini-batch in this dataset")
        return self


class TorchDatasetArgs(BaseDatasetArgs):
    @property
    def batch(self):
        return self.args.batch


class TorchModelParser(BaseModelParser):
    def _setup(self):
        BaseModelParser._setup(self)
        self.add_argument("--pretrained", default=None, help="pretrained path to be passed to the model getter")
        return self


class TorchModelArgs(BaseModelArgs):
    @property
    def pretrained(self):
        return self.args.pretrained


class TorchSessionParser(TorchDatasetParser, TorchModelParser, BaseSessionParser):
    def _setup(self):
        TorchDatasetParser._setup(self)
        TorchModelParser._setup(self)
        BaseSessionParser._setup(self)
        self.add_argument("--report_period", default=30, type=int, help="how frequently to report session metrics, "
                                                                        "in number of steps (mini-batches)")
        self.add_argument("--cuda", action="store_true", help="whether to use cuda (if available)")
        return self


class TorchSessionArgs(TorchDatasetArgs, TorchModelArgs, BaseSessionArgs):
    @property
    def report_period(self):
        return self.args.report_period

    @property
    def cuda(self):
        return self.args.cuda


class TorchTrainParser(TorchSessionParser):
    def _setup(self):
        BaseSessionParser._setup(self)
        self.add_argument("--param_summarize_period", default=25, type=int, help="how frequently to summarize "
                                                                                 "parameter distributions, "
                                                                                 "in number  of steps (mini-batches)")
        self.add_argument("--max_steps", default=1500, type=int, help="max number of steps to run before training is "
                                                                      "terminated")
        self.add_argument("--train_features", action="store_true", help="to train the feature layers (disabled by "
                                                                        "default)")
        self.add_argument("--logdir", default="/output", help="folder to store tensorboard summaries")
        self.add_argument("--checkpoint", action="store_true", help="do checkpoint for the model (disabled by default)")
        return self


class TorchTrainArgs(TorchSessionArgs):
    @property
    def param_summarize_period(self):
        return self.args.param_summarize_period

    @property
    def max_steps(self):
        return self.args.max_steps

    @property
    def train_features(self):
        return self.args.train_features

    @property
    def logdir(self):
        return self.args.logdir

    @property
    def checkpoint(self):
        return self.args.checkpoint


class TorchInferParser(TorchSessionParser):
    def _setup(self):
        BaseSessionParser._setup(self)
        return self


class TorchInferArgs(TorchSessionArgs):
    pass
