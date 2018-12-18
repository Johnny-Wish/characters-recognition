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
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


class Subset:
    def __init__(self, X=None, y=None, encoder=None):
        """
        A object simulating the training set / testing test / validation set of a super dataset
        :param X: A 2-dim sequence, n_samples x n_feature_dims
        :param y: A 1-dim/2-dim sequence, n_samples or n_samples x 1
        :param encoder: a fitted LabelEncoder instance or a callable that returns an encoded label vector
        """
        if X is None:
            self._X = []
        else:
            self._X = X

        if y is None:
            self._y = []
        else:
            self._y = y

        if isinstance(self._y, np.ndarray) and len(self._y.shape) == 2:
            self._y = self._y.flatten()  # the method np.ndarray.flatten() is stupid and doesn't update `self`

        # if an encoder is given, encode labels accordingly
        if isinstance(encoder, LabelEncoder):
            self._y = encoder.transform(self._y)
        elif callable(encoder):
            self._y = encoder(self._y)

        assert len(self._X) == len(self._y), "X and y differ in length {} != {}".format(len(self._X), len(self._y))

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

    def __repr__(self):
        return "<Subset: X={}, y={}>".format(self._X, self._y)


class Dataset:
    def __init__(self, filename="emnist-byclass.mat", folder="dataset", label_order=None):
        dataset = loadmat(os.path.join(folder, filename)).get("dataset", None)
        train, test, mapping = dataset[0][0]
        train, test = train[0][0], test[0][0]  # `train` and `tests` are tuples of (images, labels, writers)

        if label_order == "shift":
            lower_bound = min(np.min(train[1]), np.min(test[1]))
            self._encoder = lambda old: old - lower_bound
            self._decoder = lambda new: new + lower_bound
        elif label_order == "reorder":
            self._encoder = LabelEncoder().fit(train[1])
            self._decoder = self._encoder
        else:
            if label_order is not None:
                print("unrecognized label order {}".format(label_order))
            self._encoder = None
            self._decoder = None

        self._train = Subset(X=train[0], y=train[1], encoder=self._encoder)
        self._test = Subset(X=test[0], y=test[1], encoder=self._encoder)
        self._mapping = mapping

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def mapping(self):
        return self._mapping

    def __dict__(self):
        return {"train": self._train, "test": self._test, "mapping": self._mapping}

    @classmethod
    def from_dict(cls, d):
        instance = Dataset.__new__(cls)
        instance._train = d.get("train")
        instance._test = d.get("test")
        instance._mapping = d.get("mapping")
        return instance

    def __repr__(self):
        return "<Dataset: train={}, test={}>".format(self._train, self._test)
