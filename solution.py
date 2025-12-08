"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d

class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
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

            diff_sq = np.sum((left_image - right_image_shifted) ** 2, axis=2)

            ssdd_tensor[:, :, label_idx] = convolve2d(diff_sq, kernel, mode="same", boundary="fill", fillvalue=0.0)

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        idx = np.arange(num_labels)
        diff = np.abs(idx[:, None] - idx[None, :])

        neighbor_mask = diff == 1
        far_mask = diff >= 2

        l_slice[:, 0] = c_slice[:, 0]

        for col in range(1, num_of_cols):
            prev = l_slice[:, col - 1]
            prev_mat = prev[None, :]

            neigh = np.where(neighbor_mask, prev_mat, np.inf).min(axis=1) + p1
            far = np.where(far_mask, prev_mat, np.inf).min(axis=1) + p2
            m_d = np.minimum.reduce([prev, neigh, far])

            l_slice[:, col] = c_slice[:, col] + m_d - prev.min()

        # plt.figure()
        # plt.imshow(np.transpose(l_slice))
        # plt.title('Slices')
        # plt.colorbar()
        # plt.title('Smooth Depth - DP')
        # plt.show()
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)

        for row in range(ssdd_tensor.shape[0]):
            c_slice = ssdd_tensor[row, :, :].T
            l_slice = self.dp_grade_slice(c_slice, p1, p2)
            l[row, :, :] = l_slice.T

        label_smooth_dp = np.argmin(l, axis=2)
        return label_smooth_dp


    def _iter_direction_slices(self, ssdd_tensor: np.ndarray, direction: int):
        """
        Directions (arrows point TOWARD the center):
            1: left  -> right
            2: top-left  -> bottom-right  (NW->SE)
            3: top  -> bottom
            4: top-right -> bottom-left   (NE->SW)
            5: right -> left
            6: bottom-right -> top-left   (SE->NW)
            7: bottom -> top
            8: bottom-left -> top-right   (SW->NE)

        Returns: list of (c_slice, coords)
            c_slice: (n_labels, N)
            coords:  (N, 2)  [row, col]
        """
        H, W, n_labels = ssdd_tensor.shape
        slices = []

        # map to canonical kind + reverse flag
        if   direction == 1: kind, rev = "h",  False
        elif direction == 5: kind, rev = "h",  True
        elif direction == 3: kind, rev = "v",  False
        elif direction == 7: kind, rev = "v",  True
        elif direction == 2: kind, rev = "md", False  # NW->SE
        elif direction == 6: kind, rev = "md", True   # SE->NW
        elif direction == 4: kind, rev = "ad", False  # NE->SW
        elif direction == 8: kind, rev = "ad", True   # SW->NE
        else:
            raise ValueError(f"Invalid direction {direction}")

        # ---------- horizontal & vertical ----------
        if kind in ("h", "v"):
            if kind == "h":   # rows
                rows_mat = np.repeat(np.arange(H)[:, None], W, axis=1)
                cols_mat = np.repeat(np.arange(W)[None, :], H, axis=0)
                n_slices = H
            else:             # "v" -> cols
                rows_mat = np.repeat(np.arange(H)[None, :], W, axis=0)
                cols_mat = np.repeat(np.arange(W)[:, None], H, axis=1)
                n_slices = W

            if rev:
                rows_mat = rows_mat[:, ::-1]
                cols_mat = cols_mat[:, ::-1]

            for i in range(n_slices):
                rows = rows_mat[i]
                cols = cols_mat[i]
                coords = np.column_stack((rows, cols))        # (N, 2)
                c_slice = ssdd_tensor[rows, cols, :].T        # (n_labels, N)
                slices.append((c_slice, coords))
            return slices

        # ---------- main diagonals (NW<->SE) ----------
        if kind == "md":
            # k = col - row
            for k in range(-(H - 1), W):
                rows = np.arange(H)
                cols = rows + k
                mask = (cols >= 0) & (cols < W)
                rows, cols = rows[mask], cols[mask]
                if rows.size == 0:
                    continue

                lin = rows * W + cols
                r_idx, c_idx = np.unravel_index(lin, (H, W))
                coords = np.column_stack((r_idx, c_idx))
                if rev:
                    coords = coords[::-1]

                r_idx, c_idx = coords[:, 0], coords[:, 1]
                c_slice = ssdd_tensor[r_idx, c_idx, :].T
                slices.append((c_slice, coords))
            return slices

        # ---------- anti-diagonals (NE<->SW) ----------
        if kind == "ad":
            # s = row + col
            for s in range(0, H + W - 1):
                rows = np.arange(H)
                cols = s - rows
                mask = (cols >= 0) & (cols < W)
                rows, cols = rows[mask], cols[mask]
                if rows.size == 0:
                    continue

                lin = rows * W + cols
                r_idx, c_idx = np.unravel_index(lin, (H, W))
                coords = np.column_stack((r_idx, c_idx))
                if rev:
                    coords = coords[::-1]

                r_idx, c_idx = coords[:, 0], coords[:, 1]
                c_slice = ssdd_tensor[r_idx, c_idx, :].T
                slices.append((c_slice, coords))
            return slices




    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        direction_to_slice = {}

        for direction in range(1, num_of_directions + 1):
            l_dir = np.zeros_like(ssdd_tensor, float)

            for c_slice, coords in self._iter_direction_slices(ssdd_tensor, direction):
                l_slice = self.dp_grade_slice(c_slice, p1, p2)   # (D,L)
                l_dir[coords[:,0], coords[:,1], :] = l_slice.T   # (L,D)

            direction_to_slice[direction] = np.argmin(l_dir, axis=2)

        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)
