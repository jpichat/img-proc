import os
import sys
import numpy as np
from numpy.polynomial import Legendre, Chebyshev
import matplotlib.pyplot as plt

from normalise import Normalizer
from load import Loader

sys.path.insert(0, os.path.join(os.getcwd(), "../data"))


class BiasCorrector:
    """
    Estimates the 2D bias field function as a linear combination of 1D Legendre or Chebyshev polynomials.
    """

    def __init__(self, m: int = 5, n: int = 5, poly_basis="Legendre"):
        self.m = m  # deg of the polynomial along the x-axis.
        self.n = n  # deg of the polynomial along the y-axis.

        if poly_basis not in ("Legendre", "Chebyshev"):
            raise ValueError("Unknown polynomial basis.")

        self.poly_basis = poly_basis

        # args set during fit
        self.A = None  # design matrix
        self.x = None
        self._img_h = None  # image height
        self._img_w = None  # image width

    def fit(self, img: np.ndarray):
        """

        :param img: greyscale (2D) image
        :return:
        """
        b = img.flatten()
        self._img_h, self._img_w = img.shape

        # normalized grid coordinates ([0, 1] range)
        span_h, span_w = np.mgrid[0 : 1 : self._img_h * 1j, 0 : 1 : self._img_w * 1j]
        span_h_flat = span_h.flatten()
        span_w_flat = span_w.flatten()

        A_ = []  # design matrix A

        for xdeg in range(self.m + 1):

            for ydeg in range(self.n + 1):

                if self.poly_basis == "Legendre":
                    px = Legendre.basis(xdeg)(2 * span_w_flat - 1)
                    py = Legendre.basis(ydeg)(2 * span_h_flat - 1)

                elif self.poly_basis == "Chebyshev":
                    px = Chebyshev.basis(xdeg)(2 * span_w_flat - 1)
                    py = Chebyshev.basis(ydeg)(2 * span_h_flat - 1)

                A_.append(px * py)  # outer product of 1D polynomials

        self.A = np.column_stack(A_)

        # solve for coefficients x using pseudo-inverse
        self.x = np.dot(np.linalg.pinv(self.A), b)

    @property
    def bias_image(self) -> np.ndarray:
        if self.A is None or self.x is None:
            raise ValueError("BiasCorrector should be fitted first!")
        bias = np.dot(self.A, self.x)
        return np.reshape(bias, (self._img_h, self._img_w))

    def transform(self, img: np.ndarray, bias_img: np.ndarray = None):
        bias_img = bias_img or self.bias_image
        if bias_img is None:
            raise ValueError("BiasCorrector should be fitted first!")

        return Normalizer(new_range=(0, 1)).fit_transform(img / bias_img)

    def plot(self, original_img, corrected_img, bias_img=None, cmap="grey"):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        titles = ["Original Image", "Estimated Bias Field", "Bias-Corrected Image"]

        images = [original_img, bias_img or self.bias_image, corrected_img]

        for i, (ax, title, im) in enumerate(zip(axes, titles, images)):
            ax.imshow(im, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    img = Loader().load("data/pd.tiff")

    # estimate bias field
    bias_corr = BiasCorrector(m=5, n=5, poly_basis="Legendre")
    bias_corr.fit(img)

    # correct bias
    corr_img = bias_corr.transform(img)

    # visualise
    bias_corr.plot(img, corr_img)
