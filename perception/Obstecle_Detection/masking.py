import cv2
import numpy as np
import sys


def video_masking(path: str):
    low_green = np.array([0, 0, 0])
    high_green = np.array([35, 189, 255])

    #  Access primary camera feed
    video = cv2.VideoCapture(path)
    video.set(cv2.CAP_PROP_POS_MSEC, 1e4)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    filename = 'my_output.mp4v'
    result = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
    # Mask each frame
    while True:
        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            # cv2.imshow("Original frame", frame)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # cv2.imshow('HSV', hsv)
            mask = cv2.inRange(hsv, low_green, high_green)
            # cv2.imshow('Masked Frame', mask)
            result.write(mask)
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
    video.release()
    result.release()

    cv2.destroyAllWindows()
    print("video saved successfully")
    return filename


# empty function
def doNothing(x):
    pass


def get_hsv_threshold():
    # creating a resizable window named Track Bars
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

    # creating track bars for gathering threshold values of red green and blue
    cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_red', 'Track Bars', 0, 255, doNothing)

    cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_red', 'Track Bars', 0, 255, doNothing)

    # reading the image
    object_image = cv2.imread("../../data/real drone camera/Screenshot 2023-05-03 145135.png")

    # resizing the image for viewing purposes
    resized_image = cv2.resize(object_image, (800, 626))
    # converting into HSV color model
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # showing both resized and hsv image in named windows
    cv2.imshow('Base Image', resized_image)
    cv2.imshow('HSV Image', hsv_image)

    # creating a loop to get the feedback of the changes in trackbars
    while True:
        # reading the trackbar values for thresholds
        min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
        min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
        min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

        max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
        max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
        max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

        # using inrange function to turn on the image pixels where object threshold is matched
        mask = cv2.inRange(hsv_image, (min_blue, min_green, min_red), (max_blue, max_green, max_red))
        # showing the mask image
        cv2.imshow('Mask Image', mask)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # printing the threshold values for usage in detection application
        print(f'min_blue {min_blue}  min_green {min_green} min_red {min_red}')
        print(f'max_blue {max_blue}  max_green {max_green} max_red {max_red}')
        # destroying all windows
    cv2.destroyAllWindows()
