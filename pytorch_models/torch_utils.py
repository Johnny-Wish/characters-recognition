import numpy as np
from PIL import Image


class ArrayTransform:
    """
    A callable that transforms  np.ndarray`s as if they are images
    """
    def __init__(self, img_transform):
        self.img_transform = img_transform

    def __call__(self, x):
        image = Image.fromarray(x, "L")
        resized = self.img_transform(image)
        return np.stack([np.asarray(resized, dtype=np.double)], axis=0)
