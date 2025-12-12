from scipy.signal import convolve2d
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from solution import Solution
from scipy.ndimage import median_filter

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

    @staticmethod
    def smooth_depth_median(depth_map: np.ndarray, win_size: int = 5) -> np.ndarray:
        """Smooth a depth (disparity) map using a local median filter.

        This method replaces the exercise smoothing approaches by applying
        post-processing smoothing directly on the final disparity map.

        Args:
            depth_map: HxW disparity map (e.g., naive / DP / SGM output).
            win_size: Odd integer window size for the median filter.

        Returns:
            Smoothed depth map of shape HxW.
        """
        return median_filter(depth_map, size=win_size, mode="nearest")


COST1 = 0.5
COST2 = 3.0
WIN_SIZE = 3
DISPARITY_RANGE = 20
##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = '315297408'
ID2 = '211336888'
##########################################################


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


def forward_map(left_image, labels):
    labels -= DISPARITY_RANGE
    mapped = np.zeros_like(left_image)
    for row in range(left_image.shape[0]):
        cols = range(left_image.shape[1])
        mapped[row,
               np.clip(cols - labels[row, ...], 0, left_image.shape[1] - 1),
               ...] = left_image[row, cols, ...]
    return mapped


def load_data(is_your_data=False):
    # Read the data:
    if is_your_data:
        left_image = mpimg.imread('my_image_left.png')
        right_image = mpimg.imread('my_image_right.png')
    else:
        left_image = mpimg.imread('image_left.png')
        right_image = mpimg.imread('image_right.png')
    return left_image, right_image


def main_bonus():

    COST1 = 0.5
    COST2 = 3.0
    WIN_SIZE = 3
    DISPARITY_RANGE = 20

    left_image, right_image = load_data()
    solution = Solution()
    # Compute Sum-Abs-Diff distance
    tt = tic()
    sadd = Bonus.sad_distance(left_image.astype(np.float64),
                                 right_image.astype(np.float64),
                                 win_size=WIN_SIZE,
                                 dsp_range=DISPARITY_RANGE)

    print(f"SADD calculation done in {toc(tt):.4f}[seconds]")

    #Plot part A
    disparity_values = range(-DISPARITY_RANGE, DISPARITY_RANGE+1)
    chosen_disparities = [-10, -5, 0, 5, 10]
    chosen_disparities_indices = np.where(np.isin(disparity_values, chosen_disparities))[0].astype(int)
    plt.figure()
    for idx, d in enumerate(chosen_disparities):
        plt.subplot(1, len(chosen_disparities), idx+1)
        curr_disparity_idx = chosen_disparities_indices[idx]
        plt.imshow(sadd[:, :, chosen_disparities_indices[idx]], cmap='inferno')
        plt.title(f"Disparity = {d}")
        plt.axis('off')
    plt.show()

    # Construct naive disparity image
    tt = tic()
    label_map = solution.naive_labeling(sadd)
    print(f"Naive labeling done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.subplot(1, 2, 2)
    plt.imshow(label_map)
    plt.colorbar()
    plt.title('Naive Depth')
    plt.show()

    # Smooth the Naive depth map using Median Filter
    tt = tic()
    median_smooth_label_maps = []
    window_sizes = [3, 5, 9, 15]
    for w in window_sizes:
        median_smooth_label_maps.append(Bonus.smooth_depth_median(label_map, w))
    print(f"Median Filter done in {toc(tt):.4f}[seconds]")

    plt.figure()
    plt.subplot(1, len(window_sizes) + 1, 1)
    plt.imshow(label_map)
    plt.title('Naive depth')
    for idx, w in enumerate(window_sizes):
        plt.subplot(1, len(window_sizes) + 1, idx + 2)
        plt.imshow(median_smooth_label_maps[idx])
        plt.title(f"Median Win={w}")
    plt.show()


    # Compute forward map of the left image to the right image.
    mapped_image_smooth_dp = forward_map(left_image, labels=label_map)
    # plot left image, forward map image and right image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_smooth_dp)
    plt.title('Smooth Forward map - Naive')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')
    plt.show()

    # Smooth disparity image - Dynamic Programming
    tt = tic()
    label_smooth_dp = solution.dp_labeling(sadd, COST1, COST2)
    print(f"Dynamic Programming done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(label_map)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_dp)
    plt.colorbar()
    plt.title('Smooth Depth - DP')
    plt.show()
    #
    # # Compare naive method and Dynamic Programming
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(label_map)
    plt.title('Naive Method')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_dp)
    plt.title('Dynamic Programming')
    plt.show()

    # Compute forward map of the left image to the right image.
    mapped_image_smooth_dp = forward_map(left_image, labels=label_smooth_dp)
    # plot left image, forward map image and right image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_smooth_dp)
    plt.title('Smooth Forward map - DP')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')
    plt.show()

    # Generate a dictionary which maps each direction to a label map:
    tt = tic()
    direction_to_vote = solution.dp_labeling_per_direction(sadd, COST1, COST2)
    print(f"Dynamic programming in all directions done in {toc(tt):.4f}"
          f"[seconds]")

    # Plot all directions as well as the image, in the center of the plot:
    plt.figure(figsize=(10,10))

    direction_arrow = {
        1: "➡️",  3: "⬇️",  2: "↘️",  4: "↙️",
        5: "⬅️",  7: "⬆️",  6: "↖️",  8: "↗️"
    }

    for i in range(1, 10):
        plt.subplot(3, 3, i)

        if i == 5:
            plt.imshow(left_image)
            plt.title("Left Image")
            plt.axis("off")
            continue

        direction = i if i < 5 else i - 1

        plt.imshow(direction_to_vote[direction])
        plt.title(f"Direction {direction} {direction_arrow[direction]}")
        plt.axis("off")
    plt.show()


    # Smooth disparity image - Semi-Global Mapping
    tt = tic()
    label_smooth_sgm = solution.sgm_labeling(sadd, COST1, COST2)
    print(f"SGM done in {toc(tt):.4f}[seconds]")

    # Plot Semi-Global Mapping result:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_sgm)
    plt.colorbar()
    plt.title('Smooth Depth - SGM')
    plt.show()

    # Plot the forward map based on the Semi-Global Mapping result:
    mapped_image_smooth_sgm = forward_map(left_image, labels=label_smooth_sgm)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_smooth_sgm)
    plt.title('Smooth Forward map - SGM')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')
    plt.show()

    # Compare naive method and SGM
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(label_map)
    plt.title('Naive Method')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_sgm)
    plt.title('Smooth Forward map - SGM')
    plt.show()

