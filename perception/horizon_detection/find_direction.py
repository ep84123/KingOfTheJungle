import numpy as np

from perception.horizon_detection.create_histogram import get_histogram, cos_weight, get_theta_phi_from_index
from perception.horizon_detection.search_windows import binary_find_windows, simple_loss_function
from tools import pfm_tool


def get_direction_from_image(image: np.ndarray):
    hist = get_histogram(image, (70, 90), 20, 7, weight_func=cos_weight)
    thresh = np.nanpercentile(hist[hist != 0], 10) * 2
    binary = pfm_tool.convert_to_binary(hist, thresh)
    i, j = binary_find_windows(binary, simple_loss_function)
    return get_theta_phi_from_index(i, j)
