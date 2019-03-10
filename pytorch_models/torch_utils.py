import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class EmbedModule(nn.Module):
    def __init__(self):
        super(EmbedModule, self).__init__()

    def embed(self, input):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


class EmbedModuleBuilder:
    def __init__(self, num_channels, num_classes, pretrained_path=None, train_features=False, device="cpu"):
        """
        constructs a callable class, which, when called, returns a desired model
        :param num_channels: number of channels in the input tensor
        :param num_classes: number of labels in the classification problem
        :param pretrained_path: path to state dicts saved on hard drive; if None, randomly initiate the model
        :param train_features: whether to train the features layers (i.e., layers before MLP classifier)
        :param device: device to use when loading model params in memory, default is cpu
        """
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.train_features = train_features
        self.device = device
        self._model = None

    def _instantiate_model(self) -> EmbedModule:
        raise NotImplementedError

    def _get_state_dict(self):
        return torch.load(self.pretrained_path, map_location=self.device)

    def _process_state_dict(self, d):
        raise NotImplementedError

    def _load_state_dict(self, d):
        self._model: EmbedModule
        self._model.load_state_dict(d)

    def _set_trainable(self):
        if self.pretrained_path is None and not self.train_features:
            raise ValueError("Cannot skip training features when the model is not pretrained")

    def _set_model(self):
        self._instantiate_model()
        if self.pretrained_path is not None:
            pretrained = self._get_state_dict()
            pretrained = self._process_state_dict(pretrained)
            self._load_state_dict(pretrained)

        self._set_trainable()

    def __call__(self):
        if self._model is None:
            self._set_model()
        return self._model


def get_metrics_dict(labels: torch.Tensor, preds: torch.Tensor):
    acc = (preds == labels).float().mean()
    preds, labels = preds.cpu(), labels.cpu()
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average="micro")
    _, _, _, support = precision_recall_fscore_support(labels, preds, warn_for=())
    std_support = np.std(support)

    # torch.Tensor are not serializable, therefore we cast metrics to built-in python classes
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
                print("Lowest loss remained {}".format(self._lowest_loss))
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
