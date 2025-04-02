import numpy as np
from PIL import Image


class Loader:
    """Basic *.tiff image loader"""

    def __init__(self, ext="tiff"):
        self.ext = ext

    def load(self, filepath):
        if self.ext == "tiff":
            out = self._load_tiff(filepath)
            return out

    @staticmethod
    def _load_tiff(filepath) -> np.ndarray:
        """
        Loads a TIFF file and converts it to a grayscale NumPy array.

        :param filepath: Path to the TIFF file.

        :return: image (numpy.ndarray): Grayscale image as a NumPy array.
        """
        image = Image.open(filepath)
        image = np.array(image)

        # convert to grayscale
        if len(image.shape) == 3 and image.shape[2] >= 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        return image.astype(np.float64)
