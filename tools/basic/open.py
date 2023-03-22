import open3d as o3d

# Load the PLY file
pcd = o3d.io.read_point_cloud("tree.ply")

# Visualize the point cloud
o3d.visualization.draw([pcd])
