import torch
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support


class ArrayTransform:
    """
    A callable that transforms  np.ndarray`s as if they are images
    """

    def __init__(self, img_transform):
        self.img_transform = img_transform

    def __call__(self, x, mode="L"):
        image = Image.fromarray(x, mode)
        resized = self.img_transform(image)
        return np.stack([np.asarray(resized, dtype=np.double)], axis=0)


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

