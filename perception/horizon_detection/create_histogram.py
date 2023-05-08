import copy
import math
import numpy as np
import open3d as o3d
import tools.pfm_tool as pfm_tool
import time

image_path = "../../data/basic enviroment/simulation data/img_ComputerVision_1_2_1679521978574868600.pfm"
# image_path = "../../data/basic enviroment/blitz/single_tree.pfm"


intrinsic_matrix = copy.deepcopy(pfm_tool.DEFAULT_INTRINSIC.intrinsic_matrix)
# intrinsic_matrix[0, 2], intrinsic_matrix[1, 2] = 450, 350
intrinsic_matrix[0, 2], intrinsic_matrix[1, 2] = 100, 200


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
    d = intrinsic_matrix
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    height, width = depth_image.shape
    points = []
    ind = [0,0]
    def add_point(depth):
        if depth < trunk:
            ind[1] += 1
            if ind[1] == height:
                ind[0] += 1
                ind[1] = 0
            x_fac = (500 - CX_DEPTH) / FX_DEPTH
            y_fac = (500 - CY_DEPTH) / FY_DEPTH
            z = depth / (math.sqrt(1 + x_fac * x_fac + (y_fac * y_fac)))
            points.append([z * x_fac, -z * y_fac, z])
    np.vectorize(add_point)(depth_image)

    for i in range(height):
        for j in range(width):
            if depth_image[i, j] < trunk:
                x_fac = (j - CX_DEPTH) / FX_DEPTH
                y_fac = (i - CY_DEPTH) / FY_DEPTH
                z = depth_image[i, j] / (math.sqrt(1 + x_fac * x_fac + (y_fac * y_fac)))
                points.append([z * x_fac, -z * y_fac, z])
    x = time.process_time()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    print("pcd:",time.process_time()-x)
    return pcd


def add_leaf_to_histogram(node: o3d.geometry.OctreeNode, node_info: o3d.geometry.OctreeNodeInfo, max_depth,
                          histogram: np.ndarray, phi_min, phi_max, theta_min, theta_max, robot_size=0.3):
    ver, hor = histogram.shape
    if node_info.depth != max_depth:
        return
    x, y, z = node_info.origin

    theta, phi = get_theta_phi(x, y, z)
    angle_range = math.asin((robot_size + node_info.size) / math.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi_index = math.floor(ver * (phi_max - phi) / (phi_max - phi_min))
    theta_index = math.floor(hor * (theta - theta_min) / (theta_max - theta_min))
    ver_index_diff = math.floor(ver * angle_range)
    hor_index_diff = math.floor(hor * angle_range)
    add_in_angle_range(histogram, phi_index, theta_index, ver_index_diff, hor_index_diff, len(node.indices) ** 2)


def get_histogram(depth_image: np.ndarray, shape, max_distance=15, octree_depth=7, drone_size=0.3, visualize=False):
    """
    create polar histogram from the given image.
    :param depth_image:
    :param shape: the shape of the resulting histogram
    :param max_distance: how far should the algorithm look in
    :param octree_depth: depth of the octree data structure, should be at least 7
    :param drone_size:
    :param visualize: True if we want to display the depth_map and the resulting histogram
    :return:
    """
    es1 = time.process_time()
    pcd = pcd_from_np(depth_image, max_distance)
    # pcd = pfm_tool.depth_np2pcd(depth_image,max_distance)
    es2 = time.process_time()
    octree = o3d.geometry.Octree(max_depth=octree_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    es3 = time.process_time()
    histogram = np.zeros(shape)
    min_theta, min_phi = get_theta_phi_from_point(get_point_from_image(depth_image.shape[0] - 1, 0, depth_image))
    max_theta, max_phi = get_theta_phi_from_point(get_point_from_image(0, depth_image.shape[1] - 1, depth_image))
    es4 = time.process_time()
    octree.traverse(lambda x, y: add_leaf_to_histogram(
        x, y, octree_depth, histogram, min_phi, max_phi, min_theta, max_theta, drone_size))
    es5 = time.process_time()
    print(es2 - es1)
    print(es3 - es2)
    print(es4 - es3)
    print(es5 - es4)
    if visualize:
        pfm_tool.display_np(np.clip(depth_image, 0, max_distance))
        pfm_tool.display_np(histogram)
    return histogram


def main():
    depth_image = pfm_tool.pfm2np(image_path)
    depth_image = depth_image
    hist = get_histogram(depth_image, (500,500), 15, drone_size=0.5, octree_depth=8, visualize=True)
    pfm_tool.display_np(pfm_tool.convert_to_binary(hist, threshhold=10))


if __name__ == "__main__":
    main()
