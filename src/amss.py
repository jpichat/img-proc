import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

from load import Loader


class AMSS:
    def __init__(self, dt: float, R: int = 3, grad_thresh: float = 4, dtype=np.float32):
        """Affine curvature motion filtering.

        :param dt: time step
        :param R: normalization scale
        :param grad_thresh: gradient threshold
        :return:
        """
        self.dt = dt
        self.R = R
        self.grad_thresh = grad_thresh
        self.dtype = dtype

    def apply(self, input_img):
        """
        filter image
        """
        N = int((0.75 * self.R ** (1 + 1 / 3) / self.dt) + 0.5)
        u = np.copy(input_img.astype(self.dtype))
        thresh = self.grad_thresh

        for iter_num in range(N):
            u = self._update(u, self.dt, thresh)
            thresh = 1  # force back to 1 for next iterations

        # saturate values to 0-255
        u[u > 255] = 255
        u[u < 0] = 0

        return u

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _update(u, dt, grad_thresh):
        n, m = u.shape
        u_temp = np.empty((n + 2, m + 2))

        # u_temp with mirrored borders
        for i in prange(n + 2):
            if i == 0:
                src_row = u[0]
            elif i <= n:
                src_row = u[i - 1]
            else:
                src_row = u[-1]
            for j in range(m + 2):
                if j == 0:
                    u_temp[i, j] = src_row[0]
                elif j < m + 1:
                    u_temp[i, j] = src_row[j - 1]
                else:
                    u_temp[i, j] = src_row[-1]

        # iterate over u_temp to compute updates
        for i in prange(1, n + 1):
            for j in range(1, m + 1):
                # Compute gradients
                s_x = (
                    2.0 * (u_temp[i, j + 1] - u_temp[i, j - 1])
                    + (u_temp[i - 1, j + 1] - u_temp[i - 1, j - 1])
                    + (u_temp[i + 1, j + 1] - u_temp[i + 1, j - 1])
                )
                s_y = (
                    2.0 * (u_temp[i + 1, j] - u_temp[i - 1, j])
                    + (u_temp[i + 1, j + 1] - u_temp[i - 1, j + 1])
                    + (u_temp[i + 1, j - 1] - u_temp[i - 1, j - 1])
                )
                grad = (1.0 / 8.0) * (s_x ** 2 + s_y ** 2) ** (1 / 2)

                if grad < grad_thresh:
                    # Heat diffusion
                    K = 0.5
                    L = (
                        -4.0 * u_temp[i, j]
                        + u_temp[i - 1, j]
                        + u_temp[i, j - 1]
                        + u_temp[i, j + 1]
                        + u_temp[i + 1, j]
                    )
                    u[i - 1, j - 1] += K * dt * L
                else:
                    # AMSS diffusion
                    s_x_sq = s_x ** 2
                    s_y_sq = s_y ** 2
                    sum_sq = s_x_sq + s_y_sq
                    eta0 = 0.5 * sum_sq - (s_x_sq * s_y_sq) / sum_sq
                    eta1 = 2 * eta0 - s_x_sq
                    eta2 = 2 * eta0 - s_y_sq
                    eta3 = -eta0 + 0.5 * (sum_sq - s_x * s_y)
                    eta4 = -eta0 + 0.5 * (sum_sq + s_x * s_y)
                    l_comb = (
                        -4.0 * eta0 * u_temp[i, j]
                        + eta1 * (u_temp[i, j + 1] + u_temp[i, j - 1])
                        + eta2 * (u_temp[i + 1, j] + u_temp[i - 1, j])
                        + eta3 * (u_temp[i - 1, j - 1] + u_temp[i + 1, j + 1])
                        + eta4 * (u_temp[i - 1, j + 1] + u_temp[i + 1, j - 1])
                    )

                    if l_comb > 0:
                        delta = 0.25 * dt * l_comb ** (1 / 3)
                    else:
                        delta = -0.25 * dt * -(l_comb ** (1 / 3))
                    u[i - 1, j - 1] += delta
        return u

    @staticmethod
    def plot(original_image, filtered_image, cmap="grey"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        titles = ["Original image", "Filtered image"]

        images = [original_image, filtered_image]

        for i, (ax, title, im) in enumerate(zip(axes, titles, images)):
            ax.imshow(im, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    img = Loader().load("data/ny1.jpg")

    # initialise filter
    amss = AMSS(0.1, R=3)

    # apply
    filtered_img = amss.apply(img)

    # visualise
    amss.plot(img, filtered_img)
