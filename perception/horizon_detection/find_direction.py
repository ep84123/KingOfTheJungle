import math

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from perception.horizon_detection.create_histogram import get_histogram, cos_weight, get_theta_phi_from_index
from perception.horizon_detection.search_windows import binary_find_windows, simple_loss_function
from tools import pfm_tool
import time


def go_down_loss(mat):
    return simple_loss_function(mat, target_direction=(mat.shape[0] * 0.53, mat.shape[1] / 2), x_weight=1,
                                y_weight=1)


def get_direction_from_image(image: np.ndarray, iter_num: int, pdf):
    max_depth = 10
    thresh_loss_sesitivity = 1000
    pfm_tool.display_np(np.clip(image,0,max_depth),f"frame {iter_num}", pdf)
    es = time.time()
    res = (60, 90)
    hist = get_histogram(image, res, max_depth,octree_depth=6, weight_func=cos_weight)
    thresh = np.nanpercentile(hist[hist != 0], 10) + 5
    thresh = np.nanpercentile(hist[hist != 0], 10) / 2
    thresh = 200
    binary = pfm_tool.convert_to_binary(hist, thresh, 10 ** 9, 0)
    trunced_hist = np.maximum(binary,hist/thresh) * thresh_loss_sesitivity
    i, j, value = binary_find_windows(trunced_hist, go_down_loss,thresh_loss_sesitivity,pdf)

    cfd_level = np.power(np.e,-value/thresh_loss_sesitivity)
    print(f"indexes chosen: {i},{j}")
    theta_min, phi_min = get_theta_phi_from_index(image.shape[0] - 1, 0)
    theta_max, phi_max = get_theta_phi_from_index(0, image.shape[1] - 1)
    ef = time.time()
    # for thresh in [ np.nanpercentile(hist[hist != 0], 10),0.1,10,100]:
        # pfm_tool.display_np( pfm_tool.convert_to_binary(hist, thresh, 10 ** 6, 0), f"binary hist {iter_num},thresh{thresh}\ntime:{round(ef-es,2)}" ,pdf)
    # pfm_tool.display_np(binary,f"binary hist {iter_num},thresh{thresh}\ntime:{round(ef - es, 2)}", pdf)
    # pfm_tool.display_np(np.clip(trunced_hist, 0, 1.1*thresh_loss_sesitivity),f"binary hist {iter_num},thresh{thresh}\ntime:{round(ef - es, 2)}", pdf)
    print(f"The time iter{iter_num} took is: {ef - es}")
    return math.degrees(theta_min + j / res[1] * (theta_max - theta_min)), math.degrees(phi_min + (phi_max - phi_min) * i / res[0]), cfd_level