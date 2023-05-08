import cv2
import numpy as np
from utils import open_video, EXIT_KEY

# TODO: use the output from the get_threshold
low_green = np.array([0, 0, 0])
high_green = np.array([35, 189, 255])


# Mask each frame of 'video', saves to 'output'
def mask_frames(video, output):
    while True:
        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            # cv2.imshow("Original frame", frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # cv2.imshow('HSV', hsv)
            mask = cv2.inRange(hsv, low_green, high_green)
            # cv2.imshow('Masked Frame', mask)
            output.write(mask)
            # Exit if ESC pressed
            if (cv2.waitKey(1) & 0xff) == EXIT_KEY:
                break
        else:
            break


# Mask video by pre-chosen range, save the result in output file
def video_masking(path: str, output_filename: str):
    video = open_video(path, start_point=1e4)
    result = cv2.VideoWriter(output_filename + ".mp4v", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    mask_frames()

    video.release()
    result.release()
    cv2.destroyAllWindows()

    print("video saved successfully")


# empty function
def doNothing(x):
    pass


# creating track bars for gathering threshold values of red green and blue
def create_trackbars(func):
    cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_red', 'Track Bars', 0, 255, doNothing)

    cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_red', 'Track Bars', 0, 255, doNothing)


# reading the trackbar values for thresholds
def get_mask_from_trackbars(hsv_image):
    min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
    min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
    min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

    max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
    max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
    max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

    # printing the threshold values for usage in detection application
    # print(f'min_blue {min_blue}  min_green {min_green} min_red {min_red}')
    # print(f'max_blue {max_blue}  max_green {max_green} max_red {max_red}')

    # using inrange function to turn on the image pixels where object threshold is matched
    return cv2.inRange(hsv_image, (min_blue, min_green, min_red), (max_blue, max_green, max_red))




def get_hsv_threshold():
    # creating a resizable window named Track Bars
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    create_trackbars(doNothing)

    # reading the image
    object_image = cv2.imread("Screenshot 2023-05-03 142446.png")

    # resizing the image for viewing purposes
    resized_image = cv2.resize(object_image, (800, 626))
    # converting into HSV color model
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # showing both resized and hsv image in named windows
    cv2.imshow('Base Image', resized_image)
    cv2.imshow('HSV Image', hsv_image)

    # creating a loop to get the feedback of the changes in trackbars
    while True:
        mask = get_mask_from_trackbars(hsv_image)
        # showing the mask image
        cv2.imshow('Mask Image', mask)
        # Exit if ESC pressed
        if (cv2.waitKey(1) & 0xff) == EXIT_KEY:
            break

    # destroying all windows
    cv2.destroyAllWindows()
