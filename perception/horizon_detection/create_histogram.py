import copy
import math
import os
import glob
import numpy as np
import open3d as o3d
from matplotlib.backends.backend_pdf import PdfPages

import tools.pfm_tool as pfm_tool
import time

# image_path = "../../data/basic enviroment/simulation data/img_ComputerVision_1_2_1679521978574868600.pfm"
image_paths = []  # ['../../data/complex/img_SimpleFlight_1_2_1679841217449199400.pfm']

# image_path = '../../data/complex/img_SimpleFlight_1_2_1679841217189484200.pfm'
image_dirs = ['..\\..\\data\\new_data_900X700']
for directory in image_dirs:
    image_paths += glob.glob(os.path.join(directory, '*.pfm'))

image_res = (900, 700)
fov_degrees = (90, 90)
fov = np.radians(fov_degrees)
intrinsic_matrix = o3d.open3d.camera.PinholeCameraIntrinsic(image_res[0], image_res[1],
                                                            fx=image_res[0] / (2 * math.tan(fov[0] / 2)),
                                                            fy=image_res[1] / (2 * math.tan(fov[1] / 2)),
                                                            cx=image_res[0] / 2, cy=image_res[1] / 2).intrinsic_matrix

robot_size = 0.3


def get_theta_phi(x, y, z):
    return np.arctan(x / z), np.arctan(y / np.sqrt(x ** 2 + z ** 2))


def get_theta_phi_from_index(i, j):
    d = intrinsic_matrix
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    x = (j - CX_DEPTH) / FX_DEPTH
    y = - (i - CY_DEPTH) / FY_DEPTH
    return get_theta_phi(x, y, 1)


def get_theta_phi_from_point(point):
    return get_theta_phi(point[0], point[1], point[2])


def get_point_from_image(i, j, depth_image):
    d = intrinsic_matrix
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    z = depth_image[i, j] / (math.sqrt(1 + ((j - CX_DEPTH) / FX_DEPTH) ** 2 + ((i - CY_DEPTH) / FY_DEPTH) ** 2))
    x = z * (j - CX_DEPTH) / FX_DEPTH
    y = -z * (i - CY_DEPTH) / FY_DEPTH
    return x, y, z


def add_in_angle_range(histogram, phi_index, theta_index, ver_diff, hor_diff, occupation_certianty):
    histogram[max(0, phi_index - ver_diff):min(histogram.shape[0], phi_index + ver_diff),
    max(theta_index - hor_diff, 0):min(theta_index + hor_diff, histogram.shape[1])] += occupation_certianty


def find_paths_in_histogram(histogram: np.ndarray, window_height, window_width):
    window = np.ones(window_height, window_width)
    result = np.zeros(np.asarray(histogram.shape) - np.asarray(window))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = histogram[i:i + window_height, j:j + window_width] * window
    return result


def pcd_from_np(depth_image: np.ndarray, max_distance: int):
    d = intrinsic_matrix
    res = depth_image.shape
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    image = depth_image.reshape((res[0], res[1], 1))
    x = np.arange(0, res[1])
    y = np.arange(0, res[0])
    mat = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]).reshape((res[0], res[1], 2))

    image = np.concatenate([image, mat], 2)

    Xs = ((image[:, :, 1] - CX_DEPTH) / FX_DEPTH)
    Ys = ((image[:, :, 2] - CY_DEPTH) / FY_DEPTH)
    Z = image[:, :, 0] / np.sqrt(1 + Xs * Xs + Ys * Ys)
    X = -Z * Xs
    Y = -Z * Ys
    points = np.stack((X, Y, Z), axis=2)
    points = points.reshape((points.shape[0] * points.shape[1], 3))
    bools = (depth_image < max_distance).flatten()
    points = points[bools, :]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pcd


def cubic_weight(x, d):
    return 2 * (x / d) ** 3 - 3 * (x / d) ** 2 + 1


def linear_weight(x, d):
    return 1 - x / d


def cos_weight(x, d):
    return 0.5 * (1 + math.cos(math.pi * x / d))


def double_cos_weight(x, d):
    return 0.5 * (1 + math.cos(0.5 * math.pi * (1 - math.cos(math.pi * x / d))))


def base_weight(x, d):
    return x


def add_leaf_to_histogram(node: o3d.geometry.OctreeNode, node_info: o3d.geometry.OctreeNodeInfo, max_depth,
                          histogram: np.ndarray, theta_min, theta_max, phi_min, phi_max, max_dist, weight_func):
    ver, hor = histogram.shape
    if node_info.depth != max_depth:
        return
    x, y, z = node_info.origin
    theta, phi = get_theta_phi(x, y, z)
    dist = math.sqrt(x * x + y * y + z * z)
    angle_range = math.asin(min([(robot_size + node_info.size) / dist, 1]))
    phi_index = math.floor(ver * (phi_max - phi) / (phi_max - phi_min))
    theta_index = math.floor(hor * (theta_max - theta) / (theta_max - theta_min))
    ver_index_diff = math.floor(ver * angle_range / (phi_max - phi_min))
    hor_index_diff = math.floor(hor * angle_range / (theta_max - theta_min))
    add_in_angle_range(histogram, phi_index, theta_index, ver_index_diff, hor_index_diff,
                       (len(node.indices) * len(node.indices)) * weight_func(dist, max_dist))


def get_histogram(depth_image: np.ndarray, shape, max_distance=15, octree_depth=7, visualize=False,
                  weight_func=lambda x, d: x):
    """
    create polar histogram from the given image.
    :param weight_func:
    :param depth_image:
    :param shape: the shape of the resulting histogram
    :param max_distance: how far should the algorithm look in
    :param octree_depth: depth of the octree data structure, should be at least 7
    :param visualize: True if we want to display the depth_map and the resulting histogram
    :return:
    """
    es1 = time.time()
    pcd = pcd_from_np(depth_image, max_distance)
    es2 = time.time()
    octree = o3d.geometry.Octree(max_depth=octree_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    es3 = time.time()
    histogram = np.zeros(shape)
    theta_min, phi_min = get_theta_phi_from_index(depth_image.shape[0] - 1, 0)
    theta_max, phi_max = get_theta_phi_from_index(0, depth_image.shape[1] - 1)

    es4 = time.time()
    octree.traverse(
        lambda x, y: add_leaf_to_histogram(x, y, octree_depth, histogram, theta_min, theta_max, phi_min, phi_max,
                                           max_distance, weight_func=weight_func))

    es5 = time.time()
    # print("point cloud: ", es2 - es1)
    # print("octree: ", es3 - es2)
    # print("min,max angles", es4 - es3)
    # print("histogram from octree", es5 - es4)
    print("total time:", es5 - es1)
    if visualize:
        pfm_tool.display_np(np.clip(depth_image, 0, max_distance))
        # pfm_tool.display_np(histogram)
        # o3d.visualization.draw([pcd])
    return histogram


# def display_dir_images(dir_path,max_depth):
#     directory = dir
#     for filename in os.listdir(directory):
#         print(filename)
#         arr = np.clip(pfm_tool.pfm2np(directory + '/' + filename),0,max_depth)
#         pfm_tool.display_np(arr)

def analize_image(image_path, i, pdf):
    weight_funcs = [cos_weight]  # [base_weight, linear_weight, cos_weight,double_cos_weight]
    thresholds = [lambda x: 100, lambda x: 1000, lambda x: max(np.percentile(x[x > 0], 10), 5)]
    depth_image = pfm_tool.pfm2np(image_path)
    max_dist = 20
    hist_res = (70, 90)
    pfm_tool.display_np(np.clip(depth_image, 0, max_dist), f"Image {i + 1}", pdf=pdf)
    for func in weight_funcs:
        hist = get_histogram(depth_image, hist_res, max_dist, octree_depth=7, visualize=False, weight_func=func)
        for thresh_func in thresholds:
            try:
                threshold = thresh_func(hist)
            except:
                threshold = -1
            pfm_tool.display_np(pfm_tool.convert_to_binary(hist, threshhold=threshold),
                                f"Image {i + 1}, Threshhold: {threshold}, Func: {func.__name__}", pdf=pdf)


def main():
    with PdfPages("result.pdf") as pdf:
        for i, image_path in enumerate(image_paths[50:60]):
            analize_image(image_path, i, pdf)

        # image = np.clip(pfm_tool.pfm2np(image_path),0,20)
        # pfm_tool.display_np(image,f"{i+1}")


if __name__ == "__main__":
    main()
