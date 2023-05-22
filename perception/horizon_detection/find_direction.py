import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from perception.horizon_detection.create_histogram import get_histogram, cos_weight, get_theta_phi_from_index
from perception.horizon_detection.search_windows import binary_find_windows, simple_loss_function
from tools import pfm_tool
import time


def go_down_loss(mat):
    return simple_loss_function(mat, target_direction=(mat.shape[0] * 0.8, mat.shape[1] / 2), x_weight=0,
                                y_weight=0.5) + simple_loss_function(mat)


def get_direction_from_image(image: np.ndarray, iter_num: int, pdf: PdfPages):
    max_depth = 20
    pfm_tool.display_np(np.clip(image,0,max_depth),f"frame {iter_num}",pdf)
    es = time.time()
    res = (70, 90)
    hist = get_histogram(image, res, max_depth, weight_func=cos_weight)
    thresh = np.nanpercentile(hist[hist != 0], 10) * 2
    binary = pfm_tool.convert_to_binary(hist, thresh, 10 ** 6, 0)
    i, j = binary_find_windows(binary, go_down_loss)
    theta_min, phi_min = get_theta_phi_from_index(image.shape[0] - 1, 0)
    theta_max, phi_max = get_theta_phi_from_index(0, image.shape[1] - 1)
    ef = time.time()
    pfm_tool.display_np(binary, f"binary hist {iter_num}", pdf)
    print(f"The time iter{iter_num} took is: {ef - es}")
    return theta_min + j / res[1] * (theta_max - theta_min), phi_min + (phi_max - phi_min) * i / res[0]
