import numpy as np
from scipy.signal import convolve2d


class Bonus:
    def __init__(self):
        pass

    @staticmethod
    def sad_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SADD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of absolute differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        sadd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

        kernel = np.ones((win_size, win_size), dtype=np.float64)

        for label_idx, disp in enumerate(disparity_values):
            right_image_shifted = np.zeros_like(right_image)

            if disp == 0:
                right_image_shifted[:] = right_image
            elif disp > 0:
                right_image_shifted[:, :-disp, :] = right_image[:, disp:, :]
            else:
                right_image_shifted[:, -disp:, :] = right_image[:, :disp, :]

            diff_sq = np.sum(np.abs(left_image - right_image_shifted), axis=2)

            sadd_tensor[:, :, label_idx] = convolve2d(diff_sq, kernel, mode="same", boundary="fill", fillvalue=0.0)

        sadd_tensor -= sadd_tensor.min()
        sadd_tensor /= sadd_tensor.max()
        sadd_tensor *= 255.0
        return sadd_tensor
