from global_utils import load
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class ConfusionMatrixAnalyzer:
    def __init__(self, matrix, labels=None):
        self.full_matrix = matrix  # type: np.ndarray
        self.misclassified = matrix.copy()

        np.fill_diagonal(self.misclassified, 0)
        if labels is None:
            labels = list(range(matrix.shape[0]))
        elif not isinstance(labels, list):
            labels = list(labels)

        _convert_label = lambda idx: labels[idx]
        convert_labels = lambda indices: list(map(_convert_label, indices))

        preds_count = self.full_matrix.sum(0)  # type: np.ndarray
        correct_count = self.full_matrix.diagonal()
        confidence = correct_count / preds_count

        suspects = self.misclassified.argsort(0)[::-1]
        suspicion = self.misclassified / preds_count
        suspicion.sort(0)
        suspicion = suspicion[::-1]

        self.df = pd.DataFrame(
            data={
                "predicted": preds_count,
                "correct": correct_count,
                "confidence": confidence,
                "suspect1": convert_labels(suspects[0]),
                "suspect2": convert_labels(suspects[1]),
                "suspect3": convert_labels(suspects[2]),
                "suspicion1": suspicion[0],
                "suspicion2": suspicion[1],
                "suspicion3": suspicion[2],
                "residual_suspicion": 1. - suspicion[:3].sum(0) - confidence,
            },
            index=labels,
        )

    @property
    def predicted(self):
        return self.df.predicted

    @property
    def correct(self):
        return self.df.correct

    @property
    def confidence(self):
        return self.df.confidence

    @property
    def labels(self):
        return self.df.index

    @property
    def suspect1(self):
        return self.df.suspect1

    @property
    def suspect2(self):
        return self.df.suspect2

    @property
    def suspect3(self):
        return self.df.suspect3

    @property
    def suspicion1(self):
        return self.df.suspicion1

    @property
    def suspicion2(self):
        return self.df.suspicion2

    @property
    def suspicion3(self):
        return self.df.suspicion3

    @property
    def residual_suspicion(self):
        return self.df.residual_suspicion


if __name__ == '__main__':
    matrix = load("pytorch_models/confusion-matrix.pth")
    labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    analyzer = ConfusionMatrixAnalyzer(matrix, labels)
