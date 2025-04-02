import numpy as np
from typing import Tuple


class Normalizer:
    """
    Simple intensity normaliser (linearly rescales intensity values to a given range)
    """

    def __init__(self, new_range: Tuple[float, float]):
        self._new_min = new_range[0]
        self._new_max = new_range[1]

    def fit_transform(self, img: np.ndarray):
        img_min = float(img.min())
        img_max = float(img.max())
        return (img - img_min) * (self._new_max - self._new_min) / (
            img_max - img_min
        ) + self._new_min
