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
from torchvision.transforms import Compose


class Reshape:
    def __init__(self, *new_shape):
        self.shape = new_shape

    def __call__(self, array: np.ndarray):
        return array.reshape(*self.shape)


class TransposeFlatten2D:
    def __init__(self, n_rows, n_cols):
        transpose_flatten = Compose([
            Reshape(n_rows, n_cols),
            np.transpose,
            Reshape(n_rows * n_cols),
        ])
        indices = np.arange(0, n_rows * n_cols)
        self.reordered_indices = transpose_flatten(indices)

    def __call__(self, arr):
        return arr[self.reordered_indices]


class Subset:
    def __init__(self, X, y, mapping, transformer=None):
        """
        An object simulating the training set / testing test / validation set of a super dataset
        :param X: A 2-dim np.ndarray, n_samples x n_feature_dims
        :param y: A 1-dim/2-dim np.ndarray, n_samples or n_samples x 1
        :param transformer: a callable instance that transforms the input X, (and leaves y untouched)
        """
        if isinstance(y, np.ndarray) and len(y.shape) == 2:
            self._y = y.flatten()  # the method np.ndarray.flatten() is stupid and doesn't update `self`

        self._X = X if transformer is None else [transformer(sample) for sample in X]

        self.transformer = transformer
        self._mapping = mapping

        if len(self._X) != len(self._y):
            raise ValueError("X and y differ in length {} != {}".format(len(self._X), len(self._y)))

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def mapping(self):
        return self._mapping

    def __dict__(self):
        return {"X": self._X, "y": self._y}

    @classmethod
    def from_dict(cls, d):
        return cls(
            X=d.get("X", None),
            y=d.get("y", None),
            mapping=d.get("mapping", None),
        )

    def sampled(self, size=1.0):
        """
        return a new Subset instance which is contained in `self`
        :param size: float or int, size of the new instance
            if float. construed as the ratio to the original instance
            if size, construed as the number of elements in the returned instance
            size of new instance will be ceiled if it exceeds the maximum
        :return: a new Subset instance
        """
        total = len(self)
        if isinstance(size, (int, np.integer)):
            sample = min(total, size)
        else:
            sample = int(total * min(size, 1.0))
        index = np.random.choice(np.arange(total), sample, replace=False)
        return Subset(self._X[index], self._y[index], self._mapping)

    def __len__(self):
        return min(len(self._X), len(self._y))  # in case X and y differ in length, which should not happen

    def __getitem__(self, item):
            return {"X": self._X[item], "y": self._y[item]}

    def __repr__(self):
        return "<Subset: X={}, y={}, mapping={}>".format(self._X, self._y, self.mapping)


class Dataset:
    def __init__(self, filename="emnist-byclass.mat", folder="dataset", transformer=None):
        """
        An object representing a dataset, consisting of training set and testing set
        :param filename: name of the .mat file, including extensions
        :param folder: path to the folder containing data files
        :param label_order: "shift", "reorder" or None, whether and how to treat label orders
        """
        dataset = loadmat(os.path.join(folder, filename)).get("dataset", None)
        train, test, mapping = dataset[0][0]
        train, test = train[0][0], test[0][0]  # `train` and `tests` are tuples of (images, labels, writers)

        self._mapping = {key: "".join(map(chr, values)) for key, *values, in mapping}
        self._train = Subset(X=train[0], y=train[1], mapping=self._mapping, transformer=transformer)
        self._sampled_train = self._train
        self._train_size = len(self._train)
        self._test = Subset(X=test[0], y=test[1], mapping=self._mapping, transformer=transformer)
        self._sampled_test = self._test
        self._test_size = len(self._test)
        self._num_classes = len(np.unique(self.train.y))

    def sample_train(self, size=0.1):
        self._sampled_train = self._train.sampled(size)
        self._train_size = len(self._sampled_train)
        return self

    def sample_test(self, size=0.1):
        self._sampled_test = self._test.sampled(size)
        self._test_size = len(self._sampled_test)
        return self

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def train_size(self):
        return self._train_size

    @property
    def test_size(self):
        return self._test_size

    @property
    def train(self):
        return self._sampled_train

    @property
    def test(self):
        return self._sampled_test

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
