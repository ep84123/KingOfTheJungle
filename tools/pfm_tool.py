import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.geometry
from matplotlib.backends.backend_pdf import PdfPages

DEFAULT_INTRINSIC = o3d.open3d.camera.PinholeCameraIntrinsic(900, 700, fx=400, fy=400, cx=500, cy=400)
INT16_MAX = 65536


# Make sure to install .pfm plugin for imageio.v3 by first running:
# import imageio
# imageio.plugins.freeimage.download()
def pfm2np(path):
    return np.asarray(iio.imread(path))


def display_pfm(path):
    display_np(pfm2np(path))


def display_np(np_arr, title="", pdf= None):
    if pdf is None:
        plt.imshow(np_arr)
        plt.title(title)
        plt.show()
    else:
        pdf.append((np_arr, title))


def display_pdf(pdf_list, filename):
    with PdfPages(filename) as pdf:
        for data,title in pdf_list:
            plt.imshow(data)
            plt.title(title)
            pdf.savefig()
            plt.close()
def convert_dtype_to_int16(image, scaling_factor=1000.0):
    depth_image = scaling_factor * image
    for i in range(len(depth_image)):
        for j in range(len(depth_image[0])):
            if depth_image[i][j] >= INT16_MAX:
                depth_image[i][j] = INT16_MAX - 1
    return np.flip(depth_image, axis=0)


# Performs conversion of .pfm file to a pcd object using int16 data type
def pfm2pcd(path, depth_trunc=20.0, intrinsic=DEFAULT_INTRINSIC):
    return depth_np2pcd(pfm2np(path), depth_trunc, intrinsic)


# Performs conversion of numpy depth map to a pcd object using int16 data type
def depth_np2pcd(depth_np, depth_trunc=20.0, intrinsic=DEFAULT_INTRINSIC):
    depth_scale = INT16_MAX / (depth_trunc + 1)
    depth = o3d.geometry.Image(np.array(convert_dtype_to_int16(depth_np, depth_scale), dtype=np.uint16))
    return o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic, depth_scale=depth_scale,
                                                           depth_trunc=depth_trunc)


def pfm2ply(input_path, output_path, depth_trunc=20.0, intrinsic=DEFAULT_INTRINSIC):
    o3d.io.write_point_cloud(output_path, pfm2pcd(input_path, depth_trunc, intrinsic))


def pfm2greyscale(path, depth_trunc=20.0):
    depth_scale = INT16_MAX / (depth_trunc + 1)
    iio.imwrite('grayscale.png', np.array(convert_dtype_to_int16(pfm2np(path), depth_scale), dtype=np.uint16))


def convert_to_binary(depth_map, threshhold, false_val=0, true_val=1):
    return np.where(depth_map < threshhold, true_val, false_val)
