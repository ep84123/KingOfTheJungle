from random import randint
import cv2
import sys
import time
import os

EXIT_KEY = 27
t0 = time.time()


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
    t0 = time.time()


def img_show(name: str, img):
    # cv2.imshow(name, img)
    # cv2.waitKey(0)
    print(name, ", time passed: ", time.time() - t0)


def get_frames(name, path, delta):
    video = open_video(path, 2e4)
    for i in range(200):
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        path = 'Video_Frames/'
        if i % delta == 0:
            cv2.imwrite(os.path.join(path, name + '-' + str(i) + '.jpg'), frame)
