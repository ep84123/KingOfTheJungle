from random import randint
import cv2
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
