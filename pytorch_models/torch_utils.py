import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from pytorch_models.pytorch_args import TorchSessionArgs, TorchTrainArgs, TorchInferArgs
from reflexive_import import ReflexiveImporter
from preprocess import Dataset


class EmbedModule(nn.Module):
    def __init__(self):
        super(EmbedModule, self).__init__()

    def embed(self, input):
        raise NotImplementedError


def get_metrics_dict(labels: torch.Tensor, preds: torch.Tensor):
    acc = (preds == labels).float().mean()
    preds, labels = preds.cpu(), labels.cpu()
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average="micro")
    _, _, _, support = precision_recall_fscore_support(labels, preds, warn_for=())
    std_support = np.std(support)

    # torch.Tensor are not serializable, therefore casting metrics to built-in python classes
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "fscore": float(fscore),
        "std_support": float(std_support),
    }


def prepend_tag(metrics: dict, tag, sep="/"):
    return {sep.join([tag, key]): value for key, value in metrics.items()}


class LossRegister:
    """
    a base class which has a `lowest_loss` property, and a `update_lowest_loss` method
    """

    def __init__(self):
        self._lowest_loss = None

    def update_lowest_loss(self, new_loss, verbose=True):
        if self._lowest_loss is None or new_loss < self._lowest_loss:
            self._lowest_loss = new_loss
            if verbose:
                print("Lowest loss updated to {}".format(self._lowest_loss))
            return True
        else:
            if verbose:
                print("Lowest loss remained to {}".format(self._lowest_loss))
            return False

    @property
    def lowest_loss(self):
        return self._lowest_loss


class Checkpointer:
    """
    a base class which has a `checkpoint_path` property, and a `checkpoint` method
    """

    def __init__(self, checkpoint_path):
        self._checkpoint_path = checkpoint_path

    def checkpoint(self):
        pass


def get_dataset_and_model(args: TorchSessionArgs):
    importer = ReflexiveImporter(
        module_name=args.model,
        var_list=["get_model", "model_args", "model_kwargs", "transformer"],
        package_name="pytorch_models",
    )

    dataset = Dataset(filename=args.datafile, folder=args.dataroot, transformer=importer["transformer"])
    if args.verbose:
        print("dataset loaded")

    get_model = importer["get_model"]  # type: callable
    model_args = importer["model_args"]  # type: tuple
    model_kwargs = importer["model_kwargs"]  # type: dict
    model_kwargs.update(dict(
        num_classes=dataset.num_classes,
        pretrained_path=args.pretrained,
    ))

    # for a training session, model needs to be specified whether the features are trained
    if isinstance(args, TorchTrainArgs):
        model_kwargs.update(dict(
            train_features=args.train_features,
        ))

    model = get_model(*model_args, **model_kwargs)
    if args.verbose:
        print("using model", model)
    return dataset, model
