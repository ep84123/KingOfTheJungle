import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

from KingOfTheJungle.perception.blitz.ground_detection import remove_ground


def get_clusters(pts: np.ndarray, eps=1, min_samples=100):
    """
    convert a np array to clusters
    :param eps: the eps of the DBSCAN algorithm
    :param min_samples: the min_samples of the DBSCAN
    :param pts: a np array of 3D points to be clustered
    :return: a list of point clouds
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(pts)
    labels = db.labels_
    all_clusters = []

    for i in range(np.max(labels) + 1):
        cluster_np = pts[i == labels]
        cluster_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_np))
        all_clusters.append(cluster_pcd)
    return all_clusters


def get_AABBs(arr):
    """
    return a list of axis aligned bounding boxes for clusters in the numpy array
    :param arr: a np array of the pointcloud
    :return: a list of clusters
    """
    groundless = remove_ground(arr)
    clusters = get_clusters(groundless)
    return [c.get_axis_aligned_bounding_box() for c in clusters]
