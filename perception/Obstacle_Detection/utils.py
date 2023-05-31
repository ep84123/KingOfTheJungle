from random import randint
import cv2
import sys
import time
import os
# from pytube import YouTube
import sys


EXIT_KEY = 27


def get_random_color():
    return randint(127, 255), randint(127, 255), randint(127, 255)


def open_video(path: str, start_point: int):
    # Read video
    video = cv2.VideoCapture(path)
    video.set(cv2.CAP_PROP_POS_MSEC, start_point)  # select start point

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    return video


def calc_fps(timer):
    return str(int(cv2.getTickFrequency() / (cv2.getTickCount() - timer)))


def init_clock():
    return time.time()


def img_show(name: str, img, t0):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    print(name, ", time passed: ", time.time() - t0)


def get_frames(name, path, delta):
    video = open_video(path, 2e4)
    for i in range(200):
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        path = '../../data/Video_Frames/'
        if i % delta == 0:
            cv2.imwrite(os.path.join(path, name + '-' + str(i) + '.jpg'), frame)


# def change_vid_res():
#     video = cv2.VideoCapture(
#         "Baby_Sensory_-_Color_Animation_#4_-_Spirals_-_Infant_Visual_Stimulation_(Soothe_colic_baby).mp4")
#     success, image = video.read()
#     # Declare the variable with value 0
#     count = 0
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi', fourcc, 5, (720, 480))
#     while True:
#         ret, frame = video.read()
#         if ret:
#             b = cv2.resize(frame, (720, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
#             out.write(b)
#         else:
#             break
#
#     video.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print("finish")
#
#
# videoURL = ""
# if (len(sys.argv) > 1):
#     videoURL = sys.argv[1]
# if ("youtube.com" not in videoURL):
#     videoURL = input("Enter YouTube URL: ")
# yt = YouTube(videoURL, use_oauth=True, allow_oauth_cache=True)
# filename = yt.title.replace(" ", "_")
# print("Downloading YouTube File: " + yt.title)
# yt.streams.first().download(filename=filename + ".mp4", )
