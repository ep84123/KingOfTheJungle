import numpy as np

from perception.horizon_detection.create_histogram import get_histogram, cos_weight, get_theta_phi_from_index
from perception.horizon_detection.search_windows import binary_find_windows, simple_loss_function
from tools import pfm_tool


def get_direction_from_image(image: np.ndarray):
    res = (70, 90)
    hist = get_histogram(image, res, 20, 7, weight_func=cos_weight)
    thresh = np.nanpercentile(hist[hist != 0], 10) * 2
    binary = pfm_tool.convert_to_binary(hist, thresh)
    i, j = binary_find_windows(binary, simple_loss_function)
    theta_min, phi_min = get_theta_phi_from_index(image.shape[0] - 1, 0)
    theta_max, phi_max = get_theta_phi_from_index(0, image.shape[1] - 1)
    return theta_min + j / res[1] * (theta_max - theta_min), phi_min + (phi_max - phi_min) * i / res[0]
