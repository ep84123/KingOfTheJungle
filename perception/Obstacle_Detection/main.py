from masking import get_hsv_threshold, video_masking
from object_tracking import multiple_object_tracking
from edge_detection import detect_edge
from utils import get_frames
from trying_substract import subtract_frames
if __name__ == '__main__':
   # multiple_object_tracking("../../data/DJI_0457.MP4")
   # get_hsv_threshold()
   detect_edge("Screenshot 2023-05-03 142446.png")
   # get_frames(name="0457", path="../../data/DJI_0457.MP4", delta=1)
   # substract_frames()
   # stabilize()
   #subtract_frames()