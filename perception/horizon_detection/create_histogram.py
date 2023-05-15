import copy
import math
import os

import numpy as np
import open3d as o3d
import tools.pfm_tool as pfm_tool
import time

# image_path = "../../data/basic enviroment/simulation data/img_ComputerVision_1_2_1679521978574868600.pfm"
image_path = '../../data/complex/img_SimpleFlight_1_2_1679841217449199400.pfm'
# image_path = '../../data/complex/img_SimpleFlight_1_2_1679841217189484200.pfm'


intrinsic_matrix = copy.deepcopy(pfm_tool.DEFAULT_INTRINSIC.intrinsic_matrix)
intrinsic_matrix[0, 2], intrinsic_matrix[1, 2] = 450, 350


phi_min = 0
phi_max = 0
theta_min = 0
theta_max = 0
robot_size = 0.3


def get_theta_phi(x, y, z):
    return math.atan(x / z), math.atan(y / math.sqrt(x ** 2 + z ** 2))


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


def pcd_from_np(depth_image, trunk=20):
    """
    create a point cloud from a numpy array
    :param depth_image:
    :param trunk: the max distance of points in the point cloud from the origin
    :return:
    """
    x = time.time()
    d = intrinsic_matrix
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    height, width = depth_image.shape
    points = []
    ind = [0, 0]

    # def add_point(depth):
    #     if depth < trunk:
    #         ind[1] += 1
    #         if ind[1] == height:
    #             ind[0] += 1
    #             ind[1] = 0
    #         x_fac = (500 - CX_DEPTH) / FX_DEPTH
    #         y_fac = (500 - CY_DEPTH) / FY_DEPTH
    #         z = depth / (math.sqrt(1 + x_fac * x_fac + (y_fac * y_fac)))
    #         points.append([z * x_fac, -z * y_fac, z])
    #
    # np.vectorize(add_point)(depth_image)

    for i in range(height):
        for j in range(width):
            if depth_image[i, j] < trunk:
                x_fac = (j - CX_DEPTH) / FX_DEPTH
                y_fac = (i - CY_DEPTH) / FY_DEPTH
                z = depth_image[i, j] / (math.sqrt(1 + x_fac * x_fac + (y_fac * y_fac)))
                points.append([z * x_fac, -z * y_fac, z])
    # x = time.time()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    print("slow pcd:", time.time() - x)
    return pcd


def pcd_from_np_fast(depth_image: np.ndarray, max_distance: int):
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
    X = Z * Xs
    Y = -Z * Ys
    points = np.stack((X, Y, Z), axis=2)
    points = points.reshape((points.shape[0] * points.shape[1], 3))
    bools = (depth_image < max_distance).flatten()
    points = points[bools, :]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pcd


def add_leaf_to_histogram(node: o3d.geometry.OctreeNode, node_info: o3d.geometry.OctreeNodeInfo, max_depth,
                          histogram: np.ndarray, theta_min, theta_max, phi_min, phi_max):
    ver, hor = histogram.shape
    if node_info.depth != max_depth:
        return
    x, y, z = node_info.origin
    theta, phi = get_theta_phi(x, y, z)
    angle_range = math.asin((robot_size + node_info.size) / math.sqrt(x * x + y * y + z * z))
    phi_index = math.floor(ver * (phi_max - phi) / (phi_max - phi_min))
    theta_index = math.floor(hor * (theta - theta_min) / (theta_max - theta_min))
    ver_index_diff = math.floor(ver * angle_range)
    hor_index_diff = math.floor(hor * angle_range)
    add_in_angle_range(histogram, phi_index, theta_index, ver_index_diff, hor_index_diff,
                       (len(node.indices) * len(node.indices)) * (max([0,a - b * (math.sqrt(x*x+y*y+z*z) - robot_size)]))
)


def get_histogram(depth_image: np.ndarray, shape, max_distance=15, octree_depth=7, visualize=False):
    """
    create polar histogram from the given image.
    :param depth_image:
    :param shape: the shape of the resulting histogram
    :param max_distance: how far should the algorithm look in
    :param octree_depth: depth of the octree data structure, should be at least 7
    :param visualize: True if we want to display the depth_map and the resulting histogram
    :return:
    """
    es1 = time.time()
    pcd = pcd_from_np_fast(depth_image, max_distance)
    # pcd = pfm_tool.depth_np2pcd(depth_image,max_distance)
    es2 = time.time()
    octree = o3d.geometry.Octree(max_depth=octree_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    es3 = time.time()
    histogram = np.zeros(shape)
    theta_min, phi_min = get_theta_phi_from_point(get_point_from_image(depth_image.shape[0] - 1, 0, depth_image))
    theta_max, phi_max = get_theta_phi_from_point(get_point_from_image(0, depth_image.shape[1] - 1, depth_image))

    es4 = time.time()
    octree.traverse(
        lambda x, y: add_leaf_to_histogram(x, y, octree_depth, histogram, theta_min, theta_max, phi_min, phi_max))

    # def pass1():
    #     pass
    # octree.traverse(lambda x,y: pass1())
    es5 = time.time()
    print("point cloud: ", es2 - es1)
    print("octree: ", es3 - es2)
    print("min,max angles", es4 - es3)
    print("histogram from octree", es5 - es4)
    print("total time:", es5 - es1)
    if visualize:
        pfm_tool.display_np(np.clip(depth_image, 0, max_distance))
        pfm_tool.display_np(histogram)
        # o3d.visualization.draw([pcd])
    return histogram


def main():
    # directory = "../../data/complex"
    # for filename in os.listdir(directory):
    #     print(filename)
    #     arr = np.clip(pfm_tool.pfm2np(directory + '/' + filename),0,100)
    #     pfm_tool.display_np(arr)
    depth_image = pfm_tool.pfm2np(image_path)
    # depth_image = depth_image[300:500, 300:700]
    hist = get_histogram(depth_image, (350,450), 20, octree_depth=7, visualize=True)
    pfm_tool.display_np(pfm_tool.convert_to_binary(hist, threshhold=10))
    voxels.sort(reverse=True)
    for i in range (1000):
        print(voxels[i])
    print(len(voxels))


if __name__ == "__main__":
    main()
