from masking import get_hsv_threshold, video_masking
from object_tracking import multiple_object_tracking
from edge_detection import detect_edge, substract_frames
from utils import get_frames
from stabilize_image import stabilize

if __name__ == '__main__':
    multiple_object_tracking("../../data/Baby_Sensory_-_Color_Animation_#4_-_Spirals_-_Infant_Visual_Stimulation_(Soothe_colic_baby).mp4")
    # get_hsv_threshold()
    # detect_edge("../../data/Video_Frames/0457-25.jpg", 25)
    # get_frames(name="0457", path="../../data/DJI_0457.MP4")
    # get_frames(name="0457", path="C:\\Users\\TLP-300\\Desktop\\DJI_0457.MP4")
    # substract_frames()
    # stabilize()
    # change_vid_res()

