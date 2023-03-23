import open3d as o3d
import numpy as np


def remove_ground(point_cloud: o3d.geometry.PointCloud):
    y_min = min(point_cloud.points, key=lambda x: x[1])[1]
    arr = np.asarray(point_cloud.points)
    arr = arr[arr[:, 1] - y_min > 5]
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))


pcd = o3d.io.read_point_cloud("../../data/basic enviroment/blitz/two_trees.ply")
removed = remove_ground(pcd)
o3d.visualization.draw([removed])
