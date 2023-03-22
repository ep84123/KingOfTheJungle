import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


pcd = o3d.io.read_point_cloud("../../data/basic enviroment/tree.ply")

vector_labels = pcd.cluster_dbscan(eps=0.3, min_points=200, print_progress=True)
labels = np.array(vector_labels)
print(vector_labels)
print(labels.max())
colors = plt.get_cmap("tab20")(labels / labels.max())
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd],
                                   zoom=0.455,
                                   front=[-0.4999, -0.1659, -0.8499],
                                   lookat=[2.1813, 2.0619, 2.0999],
                                   up=[0.1204, -0.9852, 0.1215])
o3d.visualization.draw([pcd, vector_labels.get_axis_aligned_bounding_box()])