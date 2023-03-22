import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("rabbit point cloud.ply")

print(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points()))
print(np.asarray(pcd.get_axis_aligned_bounding_box().get_extent()))

# draw pov
pov = o3d.geometry.PointCloud()
pov.points = o3d.utility.Vector3dVector([np.array([0,0,1])])
o3d.visualization.draw([pcd, pcd.get_axis_aligned_bounding_box(), pov])

#min distance print
print(min(np.asarray(pcd.compute_point_cloud_distance(pov))))