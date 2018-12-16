"""
Created:   Dec 16, 2018
encoding:  UTF-8
Author:    Shuheng Liu
Contact:   wish1104@outlook.com
GitHub:    https://github.com/Johnny-Wish

(c) All Rights Reserved
License: https://choosealicense.com/licenses/gpl-3.0/

This file contains pre-processing tools of the dataset, assumed to be of a standard .mat format
"""

import os
from scipy.io import loadmat


class Subset:
    def __init__(self, X=None, y=None):
        if X is None:
            self._X = []
        else:
            self._X = X

        if y is None:
            self._y = []
        else:
            self._y = y

        assert len(self._X) == len(self._y), "lengths of X and y differ: {} != {}".format(len(self._X), len(self._y))

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def __dict__(self):
        return {"X": self._X, "y": self._y}

    @classmethod
    def from_dict(cls, d):
        return cls(
            X=d.get("X", None),
            y=d.get("y", None),
        )


class Dataset:
    def __init__(self, filename="emnist-byclass.mat", folder="dataset"):
        dataset = loadmat(os.path.join(folder, filename)).get("dataset", None)
        train, test, mapping = dataset[0][0]
        train, test = train[0][0], test[0][0]  # `train` and `tests` are tuples of (images, labels, writers)
        self.train = Subset(X=train[0], y=train[1])
        self.test = Subset(X=test[0], y=test[1])
        self.mapping = mapping

    def __dict__(self):
        return {"train": self.train, "test": self.test, "mapping": self.mapping}

    @classmethod
    def from_dict(cls, d):
        instance = Dataset.__new__(cls)
        instance.train = d.get("train")
        instance.test = d.get("test")
        instance.mapping = d.get("mapping")
        return instance

    def __repr__(self):
        return "<Dataset: train={}, test={}>".format(self.train, self.test)
