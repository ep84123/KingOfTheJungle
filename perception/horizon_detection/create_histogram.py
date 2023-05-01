import math
import open3d

import numpy as np
import open3d as o3d
import tools.pfm_tool as pfm_tool

image_path = "../../data/basic enviroment/simulation data/img_ComputerVision_1_2_1679521978574868600.pfm"
image_path = "../../data/basic enviroment/blitz/single_tree.pfm"


def get_theta_phi(x, y, z):
    return math.atan(x / z), math.atan(y / math.sqrt(x ** 2 + z ** 2))


def get_theta_phi_from_point(point):
    return get_theta_phi(point[0], point[1], point[2])


def get_point_from_image(i, j, depth_image):
    d = pfm_tool.DEFAULT_INTRINSIC.intrinsic_matrix
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    z = depth_image[i, j] / (math.sqrt(1 + ((j - CX_DEPTH) / FX_DEPTH) ** 2 + ((i - CY_DEPTH) / FY_DEPTH) ** 2))
    x = z * (j - CX_DEPTH) / FX_DEPTH
    y = -z * (i - CY_DEPTH) / FY_DEPTH
    return x, y, z


def get_angle_arrays(depth_image: np.ndarray):
    d = pfm_tool.DEFAULT_INTRINSIC.intrinsic_matrix
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    height, width = depth_image.shape
    theta_histogram = np.zeros(depth_image.shape)
    phi_histogram = np.zeros(depth_image.shape)
    for i in range(height):
        for j in range(width):
            x = (j - CX_DEPTH) / FX_DEPTH
            y = (i - CY_DEPTH) / FY_DEPTH
            theta = math.atan2(y, 1)
            phi = math.acos(x / math.sqrt(x ** 2 + y ** 2 + 1))
            if theta < 0:
                theta = 2 * math.pi + theta
            theta_histogram[i][j] = theta
            phi_histogram[i][j] = phi
    return theta_histogram, phi_histogram


def add_in_angle_range(histogram, phi_index, theta_index, ver_diff, hor_diff, occupation_certianty):
    for i in range(max(theta_index - hor_diff, 0), min(theta_index + hor_diff, histogram.shape[0])):
        for j in range(max(0, phi_index - ver_diff), min(histogram.shape[1], phi_index + ver_diff)):
            histogram[i, j] += occupation_certianty


def find_paths_in_histogram(histogram: np.ndarray, window_height, window_width):
    window = np.ones(window_height, window_width)
    result = np.zeros(np.asarray(histogram.shape) - np.asarray(window))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = histogram[i:i + window_height, j:j + window_width] * window
    return result


def pcd_from_np(depth_image, trunk=20):
    d = pfm_tool.DEFAULT_INTRINSIC.intrinsic_matrix
    FX_DEPTH, FY_DEPTH = d[0, 0], d[1, 1]
    CX_DEPTH, CY_DEPTH = d[0, 2], d[1, 2]
    height, width = depth_image.shape
    points = []
    for i in range(height):
        for j in range(width):
            z = depth_image[i, j] / (math.sqrt(1 + ((j - CX_DEPTH) / FX_DEPTH) ** 2 + ((i - CY_DEPTH) / FY_DEPTH) ** 2))
            x = z * (j - CX_DEPTH) / FX_DEPTH
            y = -z * (i - CY_DEPTH) / FY_DEPTH
            theta = math.atan2(y, 1)
            phi = math.acos(x / math.sqrt(x ** 2 + y ** 2 + z))

            if x ** 2 + y ** 2 + z ** 2 < trunk ** 2:
                points.append([x, y, z])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pcd


def add_leaf_to_histogram(node: o3d.geometry.OctreeNode, node_info: o3d.geometry.OctreeNodeInfo, max_depth,
                          histogram: np.ndarray, phi_min, phi_max, theta_min, theta_max, robot_size=0.3):
    ver, hor = histogram.shape
    if node_info.depth != max_depth:
        return
    x, y, z = node_info.origin

    theta, phi = get_theta_phi(x, y, z)
    angle_range = math.asin((robot_size + node_info.size) / math.sqrt(x**2 + y**2 + z**2))

    phi_index = math.floor(ver * (phi - phi_min) / (phi_max - phi_min))
    theta_index = math.floor(hor * (theta - theta_min) / (theta_max - theta_min))
    ver_index_diff = math.floor(ver * angle_range)
    hor_index_diff = math.floor(hor * angle_range)
    add_in_angle_range(histogram, phi_index, theta_index, ver_index_diff, hor_index_diff, len(node.indices)**2)


def testing():
    depth = 50
    depth_image = pfm_tool.pfm2np(image_path)
    cropped = np.clip(depth_image, 0, depth)
    pfm_tool.display_np(cropped)
    pcd = pcd_from_np(depth_image, depth)
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(len(pcd.points), 3)))
    # o3d.visualization.draw_geometries([pcd])

    octree_depth = 7
    octree = o3d.geometry.Octree(max_depth=octree_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    o3d.visualization.draw_geometries([octree])
    histogram = np.zeros((300, 300))
    min_theta, min_phi = get_theta_phi_from_point(get_point_from_image(depth_image.shape[0] - 1, 0, depth_image))
    max_theta, max_phi = get_theta_phi_from_point(get_point_from_image(0, depth_image.shape[1] - 1, depth_image))
    octree.traverse(lambda x, y: add_leaf_to_histogram(x, y, 7, histogram, min_phi, max_phi, min_theta, max_theta))
    pfm_tool.display_np(histogram)


if __name__ == "__main__":
    testing()
