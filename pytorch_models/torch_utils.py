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
