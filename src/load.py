from typing import List
import numpy as np
from PIL import Image


class Loader:
    """Basic image loader"""

    def __init__(self):
        pass

    def load(self, filepath):
        out = self._load(filepath)
        return out

    @staticmethod
    def _load(filepath) -> np.ndarray:
        """
        Loads a file and converts it to a grayscale NumPy array.

        :param filepath: Path to the file
        :return: image (numpy.ndarray): Grayscale image as an array
        """
        image = Image.open(filepath)
        image = np.array(image)

        # convert to grayscale
        if len(image.shape) == 3 and image.shape[2] >= 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        return image.astype(np.float64)

    @staticmethod
    def load_batch(filepath) -> List[np.ndarray]:
        """
        Loads a file made of a series of images into a list of grayscale-converted arrays.
        """
        raise NotImplementedError()
