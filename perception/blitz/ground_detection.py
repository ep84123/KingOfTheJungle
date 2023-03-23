import numpy
import open3d as o3d
import numpy as np


def split_to_chunks(arr: np.ndarray, chunk_size):
    """

    :param chunk_size: the size of each chunk
    :param arr: a numpy array of points in 3D that is split into (x,z) chunks of the specified size
    :return: a list of chunks with x,z sizes as specified
    """
    result = {}
    scale = 1 / chunk_size
    for a in arr:
        if (int(scale * a[0]), int(scale * a[2])) not in result:
            result[(int(scale * a[0]), int(scale * a[2]))] = []
        result[(int(scale * a[0]), int(scale * a[2]))].append(a)
    return [np.array(chunk) for chunk in result.values()]


def remove_ground_from_array(arr: np.ndarray, depth):
    """
    remove the lowest points in the array, according to the depth parameter
    :param arr: an array of 3D points
    :param depth: the height from the lowest point in the array, where points will be removed
    :return: the array of points that are above the given depth
    """
    y_min = min(arr, key=lambda x: x[1])[1]
    arr = arr[arr[:, 1] - y_min > depth]
    return arr


def remove_ground(arr, chunk_size=1, depth=2):
    """
    the main function of this file, removes the ground by splitting the point cloud into chunks,
    removing the minimum depth from each chunk and returning a new np array
    :param arr: the np arr
    :param chunk_size: the size of chunks
    :param depth: the depth to be removed from each chunk
    :return: a np array with the ground removed
    """
    chunks = [remove_ground_from_array(a, depth) for a in split_to_chunks(arr, chunk_size)]
    return np.concatenate(chunks, axis=0)


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("../../data/basic enviroment/blitz/two_trees.ply")
    pts = remove_ground(pcd, 1, 2)

    # all_clusters.append(pcd)
    # o3d.visualization.draw(all_clusters)
