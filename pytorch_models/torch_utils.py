import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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

    def update_lowest_loss(self, new_loss):
        if self._lowest_loss is None or new_loss < self._lowest_loss:
            self._lowest_loss = new_loss
            return True
        else:
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
