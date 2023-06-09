from masking import get_hsv_threshold, video_masking
from object_tracking import multiple_object_tracking
from edge_detection import detect_edge_tree , substract_frames
from utils import *
from stabilize_image import stabilize
import os

if __name__ == '__main__':
    # multiple_object_tracking("../../data/Baby_Sensory_-_Color_Animation_#4_-_Spirals_-_Infant_Visual_Stimulation_(Soothe_colic_baby).mp4")
    # get_hsv_threshold("../../data/simulation_data/img_ComputerVision_0_0_1685643569315383200.png")
    # for i in range(0, 195, 5  ):
    #     print(i )
    #     detect_edge(f"../../data/Video_Frames/sample-{i}.jpg", 25)
    detect_edge_tree(f"../../data/simulation_data/img_ComputerVision_0_0_1685643575410431700.png")
    # assign directory
    # directory = '../../data/simulation_data'
    # # iterate over files in
    # # that directory
    # for filename in os.scandir(directory):
    #     if filename.is_file():
    #         print(filename.path)
    #         detect_edge_tree(filename.path )
    # get_frames(name="sample", path="../../data/Baby_Sensory_-_Color_Animation_#4_-_Spirals_-_Infant_Visual_Stimulation_(Soothe_colic_baby).mp4", delta=5)
    # get_frames(name="0457", path="C:\\Users\\TLP-300\\Desktop\\DJI_0457.MP4")
    # substract_frames()
    # stabilize()
    # change_vid_res()
    # create_vid_from_images("test_video", "../../data/simulation_data/*.png")
