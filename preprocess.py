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

from scipy.io import loadmat


class Subset:
    def __init__(self, X=None, y=None):
        if X is None:
            self.X = []
        else:
            self.X = X

        if y is None:
            self.y = []
        else:
            self.y = y

        assert len(self.X) == len(self.y)

    def to_dict(self):
        return {"X": self.X, "y": self.y}

    @classmethod
    def from_dict(cls, d):
        return cls(
            X=d.get("X", None),
            y=d.get("y", None),
        )