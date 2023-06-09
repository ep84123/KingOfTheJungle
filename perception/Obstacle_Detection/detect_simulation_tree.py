import os
import numpy as np
import cv2
from utils import img_show, init_clock


def process_img(raw_img, t0):
    # Convert to graycsale
    prs_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    # img_show("Grayscale", prs_img, t0)
    hsv_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    mask = get_hsv_mask(hsv_img)

    # # Bitwise-AND mask and original image
    prs_img = cv2.bitwise_and(prs_img, prs_img, mask=mask)
    # img_show("Masked image", prs_img, t0)


    return prs_img


def detect_edge_tree(frame):
    t0 = init_clock()  # for runtime test only
    # Read the original image
    raw_img = frame

    prss_img = process_img(raw_img, t0)
    kernel = np.ones((3, 3), np.uint8)

    prss_img = cv2.erode(prss_img, kernel, iterations=1)
    # img_show("img_erosion", prss_img, t0)
    prss_img = cv2.dilate(prss_img, kernel, iterations=2)

    # Finding Contours
    contours, hierarchy = cv2.findContours(prss_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw all contours
    # cv2.drawContours(raw_img, contours, -1, (0, 255, 0), 3)  # -1: draw all, color, thickness
    # img_show("Canny after contouring", raw_img, t0)

    cv2.destroyAllWindows()
    return contours


# reading the trackbar values for thresholds
def get_hsv_mask(hsv_image):
    min_val = np.array([19, 45, 31])
    max_val = np.array([101, 139, 255])

    mask = cv2.inRange(hsv_image, min_val, max_val)
    # img_show("hsv_mask", mask, 0)
    # using inrange function to turn on the image pixels where object threshold is matched
    return mask
