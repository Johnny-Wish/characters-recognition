import torch
from tensorboardX import SummaryWriter
from pytorch_models.pytorch_args import TorchSessionArgs
from preprocess import Dataset
from reflexive_import import ReflexiveImporter


class BaseSessionBuilder:
    def __init__(self, args: TorchSessionArgs):
        self.args = args
        if self.args.verbose:
            print(self.args)

        self.importer = ReflexiveImporter(
            module_name=self.args.model,
            var_list=["builder_class", "model_args", "model_kwargs", "transformer"],
            package_name="pytorch_models",
        )

        self._dataset = None
        self._model = None
        self._device = None
        self._writer = None
        self._session = None
        self._set_device()

        self.static_model_kwargs = dict(
            pretrained_path=self.args.pretrained,
            device=self._device,
        )

    def _set_dataset(self):
        if self._dataset is not None:
            return

        self._dataset = Dataset(
            filename=self.args.datafile,
            folder=self.args.dataroot,
            transformer=self.importer["transformer"]
        )
        if self.args.verbose:
            print("dataset loaded")

    def _set_model(self):
        if self._model is not None:
            return

        self._set_dataset()

        builder_class = self.importer["builder_class"]  # type: callable
        model_args = self.importer["model_args"]  # type: tuple
        model_kwargs = self.importer["model_kwargs"]  # type: dict
        model_kwargs.update(self.static_model_kwargs)
        model_kwargs.update(dict(num_classes=self._dataset.num_classes))

        model_builder = builder_class(*model_args, **model_kwargs)
        self._model = model_builder()

        if self.args.verbose:
            print("using model", self._model)

    def _set_device(self):
        if self._device is not None:
            return

        self._device = torch.device("cuda" if self.args.cuda or torch.cuda.is_available() else "cpu")
        if self.args.verbose:
            print("using device: {}".format(self._device))

    def _set_writer(self):
        if self._writer is not None:
            return

        self._writer = SummaryWriter(log_dir=self.args.logdir)
        if self.args.verbose:
            print("logging summaries at", self._writer.log_dir)

    def _set_session(self):
        if self._session is not None:
            return

        self._set_dataset()
        self._set_model()
        self._set_device()
        self._set_writer()

    @property
    def dataset(self):
        self._set_dataset()
        return self._dataset

    @property
    def model(self):
        self._set_model()
        return self._model

    @property
    def device(self):
        self._set_device()
        return self._device

    @property
    def writer(self):
        self._set_writer()
        return self._writer

    @property
    def session(self):
        self._set_session()
        return self._session

    def __call__(self, *args, **kwargs):
        return self.session
